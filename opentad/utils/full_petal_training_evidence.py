"""Authenticated optimizer-event evidence for Full PETAL formal runs."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from collections.abc import Iterable, Mapping
import re
import time

from mmengine import Config

from .evidence_bundle import (
    EvidenceBundleError,
    bundle_file_reference,
    contained_file,
    publish_exclusive_pair,
    read_verified_bundle_bytes,
    verify_bundle_reference,
)
from .full_petal_attestation import AttestationError, _sign_payload, verify_payload
from .full_petal_identity import (
    IdentityError,
    canonical_json_sha256,
    derive_training_trace_identity,
    validate_evaluation_artifact_bindings,
)
from .full_petal_launch import (
    FullPetalLaunchError,
    LAUNCH_RECEIPT_SCHEMA,
    LAUNCH_TICKET_SCHEMA,
    resolved_config_sha256,
    verify_launch_receipt,
)
from .immutable_event_ledger import LedgerError, load_verified_ledger


TRAINING_TRACE_SCHEMA = "full-petal-training-trace-v1"
TRAINING_COMMITMENT_SCHEMA = "full-petal-training-trace-commitment-v1"
VISUAL_PARAMETER_TRACE_SCHEMA = "full-petal-visual-parameter-trace-v1"
VISUAL_PARAMETER_COMMITMENT_SCHEMA = "full-petal-visual-parameter-commitment-v1"
FORMAL_RUN_MANIFEST_SCHEMA = "full-petal-formal-run-manifest-v1"
FORMAL_RUN_ATTESTATION_ROLE = "formal-run"
GENESIS_HASH = "0" * 64
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")

_COMMON_FORMAL_ARTIFACT_ROLES = {
    "ledger",
    "commitment",
    "ground_truth",
    "allowed_videos",
    "evaluator_spec",
    "config",
    "checkpoint",
    "data_identity",
    "training_launch_ticket",
    "training_launch_receipt",
    "evaluation_launch_ticket",
    "evaluation_launch_receipt",
    "training_trace",
    "training_commitment",
}
_LAUNCH_TICKET_FIELDS = {
    "schema_version",
    "mode",
    "commit_sha",
    "source_tree_sha256",
    "config_file_sha256",
    "resolved_config_sha256",
    "scientific_config_sha256",
    "data_identity",
    "runtime_identity",
    "b0_evidence",
    "review_evidence",
    "profile_evidence",
}
_LAUNCH_RECEIPT_FIELDS = {
    "schema_version",
    "mode",
    "commit_sha",
    "source_tree_sha256",
    "data_identity_sha256",
    "runtime_identity_sha256",
    "resolved_config_sha256",
    "scientific_config_sha256",
    "launch_ticket",
    "b0_artifact_sha256",
    "review_artifact_sha256",
    "profile_artifact_sha256",
    "world_size",
    "slurm_job_id",
    "slurm_allocation",
}
_VISUAL_EVENT_FIELDS = {
    "optimizer_event_id",
    "parameter_name",
    "requires_grad",
    "optimizer_member",
    "numel",
    "dtype",
    "before_sha256",
    "after_sha256",
    "gradient_sha256",
    "gradient_finite",
    "gradient_norm",
    "delta_norm",
}

_TRACE_FIELDS = {
    "event_id",
    "episode_id",
    "input_tokens",
    "effective_batch_size",
    "world_size",
    "elapsed_seconds",
    "peak_memory_bytes",
    "precision",
    "optimizer_config_sha256",
    "scheduler_config_sha256",
    "data_order_sha256",
    "loss_normalization_sha256",
    "skipped",
}
_IDENTITY_FIELDS = (
    "precision",
    "optimizer_config_sha256",
    "scheduler_config_sha256",
    "data_order_sha256",
    "loss_normalization_sha256",
    "effective_batch_size",
    "world_size",
)


class TrainingEvidenceError(ValueError):
    """Raised when formal optimizer-event evidence cannot be verified."""


class OptimizerEventTraceRecorder:
    """Collect one immutable cost record per optimizer episode."""

    def __init__(
        self,
        *,
        precision,
        effective_batch_size,
        world_size,
        optimizer_config_sha256,
        scheduler_config_sha256,
        data_order_sha256,
        loss_normalization_sha256,
        clock=None,
        peak_memory_reader=None,
    ):
        self.identity = {
            "precision": precision,
            "effective_batch_size": effective_batch_size,
            "world_size": world_size,
            "optimizer_config_sha256": optimizer_config_sha256,
            "scheduler_config_sha256": scheduler_config_sha256,
            "data_order_sha256": data_order_sha256,
            "loss_normalization_sha256": loss_normalization_sha256,
        }
        probe = {
            "event_id": "validation-event",
            "episode_id": "validation-episode",
            "input_tokens": 1,
            "elapsed_seconds": 1.0,
            "peak_memory_bytes": 0,
            "skipped": False,
            **self.identity,
        }
        _normalize_event(probe)
        self._clock = clock or time.perf_counter
        self._peak_memory_reader = peak_memory_reader or (lambda: 0)
        self._started_at = float(self._clock())
        self._last_elapsed = 0.0
        self._events = []

    @property
    def events(self):
        return tuple(json.loads(canonical_json(event)) for event in self._events)

    def state_dict(self):
        return {
            "identity": json.loads(canonical_json(self.identity)),
            "last_elapsed": float(self._last_elapsed),
            "events": [json.loads(canonical_json(event)) for event in self._events],
        }

    def load_state_dict(self, state):
        if not isinstance(state, Mapping) or set(state) != {
            "identity",
            "last_elapsed",
            "events",
        }:
            raise TrainingEvidenceError("optimizer event recorder state fields differ")
        identity = json.loads(canonical_json(state["identity"]))
        if identity != self.identity:
            raise TrainingEvidenceError("optimizer event recorder identity cannot change")
        events = [_normalize_event(event) for event in state["events"]]
        build_training_rows(events) if events else None
        last_elapsed = _finite(
            state["last_elapsed"], "optimizer event recorder last_elapsed"
        )
        if events and last_elapsed != events[-1]["elapsed_seconds"]:
            raise TrainingEvidenceError(
                "optimizer event recorder elapsed state differs from its event tail"
            )
        if not events and last_elapsed != 0.0:
            raise TrainingEvidenceError(
                "empty optimizer event recorder must have zero elapsed state"
            )
        self._events = events
        self._last_elapsed = last_elapsed

    def record(self, *, epoch, episode_id, input_tokens, skipped):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise TrainingEvidenceError("optimizer event epoch must be non-negative")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise TrainingEvidenceError("optimizer event episode_id must be non-empty")
        sequence = len(self._events)
        elapsed = float(self._clock()) - self._started_at
        elapsed = max(elapsed, self._last_elapsed + 1e-9)
        event = {
            "event_id": f"optimizer-event-{sequence:08d}",
            "episode_id": f"epoch={epoch}|episode={episode_id}|sequence={sequence}",
            "input_tokens": input_tokens,
            "elapsed_seconds": elapsed,
            "peak_memory_bytes": int(self._peak_memory_reader()),
            "skipped": skipped,
            **self.identity,
        }
        normalized = _normalize_event(event)
        self._events.append(normalized)
        self._last_elapsed = elapsed
        return normalized

    def persist(self, trace_path, commitment_path):
        if not self._events:
            raise TrainingEvidenceError("cannot persist an empty optimizer-event trace")
        return persist_training_trace(trace_path, commitment_path, self._events)


def canonical_json(value):
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TrainingEvidenceError(f"training evidence is not canonical JSON: {exc}") from exc


def sha256_file(path):
    path = Path(path)
    if not path.is_file():
        raise TrainingEvidenceError(f"training evidence file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingEvidenceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nonempty_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise TrainingEvidenceError(f"{label} must be a non-empty string")
    return value


def _git_sha(value, label):
    if not isinstance(value, str) or not _GIT_SHA.fullmatch(value):
        raise TrainingEvidenceError(f"{label} must be a lowercase 40-character git SHA")
    return value


def _nonnegative_int(value, label, *, positive=False):
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingEvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingEvidenceError(f"{label} must be finite")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (not positive and value < 0):
        raise TrainingEvidenceError(f"{label} has an invalid value")
    return value


def _normalize_event(event):
    if not isinstance(event, Mapping) or set(event) != _TRACE_FIELDS:
        raise TrainingEvidenceError(
            f"training event fields differ; expected={sorted(_TRACE_FIELDS)}, "
            f"found={sorted(event) if isinstance(event, Mapping) else type(event).__name__}"
        )
    result = dict(event)
    for field in ("event_id", "episode_id"):
        if not isinstance(result[field], str) or not result[field].strip():
            raise TrainingEvidenceError(f"training event {field} must be non-empty")
    result["input_tokens"] = _nonnegative_int(result["input_tokens"], "input_tokens", positive=True)
    result["effective_batch_size"] = _nonnegative_int(
        result["effective_batch_size"], "effective_batch_size", positive=True
    )
    result["world_size"] = _nonnegative_int(
        result["world_size"], "world_size", positive=True
    )
    result["elapsed_seconds"] = _finite(result["elapsed_seconds"], "elapsed_seconds", positive=True)
    result["peak_memory_bytes"] = _nonnegative_int(result["peak_memory_bytes"], "peak_memory_bytes")
    if result["precision"] not in {"fp32", "fp16", "bf16"}:
        raise TrainingEvidenceError("training event precision is unsupported")
    for field in _IDENTITY_FIELDS[1:5]:
        _sha(result[field], field)
    if not isinstance(result["skipped"], bool):
        raise TrainingEvidenceError("training event skipped must be boolean")
    return json.loads(canonical_json(result))


def _row_hash(row):
    payload = dict(row)
    payload.pop("row_hash", None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_training_rows(events):
    rows = []
    previous = GENESIS_HASH
    seen_events = set()
    seen_episodes = set()
    last_elapsed = 0.0
    identity = None
    for sequence, event in enumerate(events):
        payload = _normalize_event(event)
        if payload["event_id"] in seen_events or payload["episode_id"] in seen_episodes:
            raise TrainingEvidenceError("training trace repeats an event or episode identity")
        if payload["elapsed_seconds"] <= last_elapsed:
            raise TrainingEvidenceError("training trace elapsed time must strictly increase")
        row_identity = {field: payload[field] for field in _IDENTITY_FIELDS}
        if identity is None:
            identity = row_identity
        elif identity != row_identity:
            raise TrainingEvidenceError("training trace changes optimizer/scheduler/data identity")
        row = {
            "schema_version": TRAINING_TRACE_SCHEMA,
            "sequence": sequence,
            "previous_hash": previous,
            **payload,
        }
        row["row_hash"] = _row_hash(row)
        rows.append(row)
        previous = row["row_hash"]
        last_elapsed = payload["elapsed_seconds"]
        seen_events.add(payload["event_id"])
        seen_episodes.add(payload["episode_id"])
    if not rows:
        raise TrainingEvidenceError("training trace cannot be empty")
    return tuple(rows)


def verify_training_rows(rows):
    if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, Iterable):
        raise TrainingEvidenceError("training rows must be a sequence")
    events = []
    previous = GENESIS_HASH
    for sequence, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TrainingEvidenceError("training row must be an object")
        required = _TRACE_FIELDS | {
            "schema_version",
            "sequence",
            "previous_hash",
            "row_hash",
        }
        if set(raw) != required:
            raise TrainingEvidenceError("training row envelope/schema fields differ")
        row = json.loads(canonical_json(raw))
        if row["schema_version"] != TRAINING_TRACE_SCHEMA or row["sequence"] != sequence:
            raise TrainingEvidenceError("training row schema or sequence differs")
        if row["previous_hash"] != previous or _row_hash(row) != row["row_hash"]:
            raise TrainingEvidenceError("training trace hash chain differs or is truncated")
        events.append({field: row[field] for field in _TRACE_FIELDS})
        previous = row["row_hash"]
    rebuilt = build_training_rows(events)
    if tuple(json.loads(canonical_json(row)) for row in rows) != rebuilt:
        raise TrainingEvidenceError("training trace is not canonical")
    return rebuilt


def persist_training_trace(trace_path, commitment_path, events):
    rows = build_training_rows(events)
    encoded = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    commitment = {
        "schema_version": TRAINING_COMMITMENT_SCHEMA,
        "trace_filename": Path(trace_path).name,
        "trace_sha256": hashlib.sha256(encoded).hexdigest(),
        "count": len(rows),
        "final_hash": rows[-1]["row_hash"],
    }
    try:
        publish_exclusive_pair(
            trace_path,
            encoded,
            commitment_path,
            (canonical_json(commitment) + "\n").encode("utf-8"),
        )
    except EvidenceBundleError as exc:
        raise TrainingEvidenceError(str(exc)) from exc
    return commitment


def _read_jsonl(path):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingEvidenceError(f"failed to read training trace: {exc}") from exc
    if any(canonical_json(row) != line for row, line in zip(rows, lines)):
        raise TrainingEvidenceError("training trace does not use canonical JSONL")
    return rows


def load_training_trace(trace_path, commitment_path):
    try:
        raw_commitment = Path(commitment_path).read_text(encoding="utf-8")
        if not raw_commitment.endswith("\n") or raw_commitment.count("\n") != 1:
            raise TrainingEvidenceError("training commitment must be one canonical JSON line")
        commitment = json.loads(raw_commitment)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingEvidenceError(f"failed to read training commitment: {exc}") from exc
    expected = {
        "schema_version",
        "trace_filename",
        "trace_sha256",
        "count",
        "final_hash",
    }
    if set(commitment) != expected or canonical_json(commitment) + "\n" != raw_commitment:
        raise TrainingEvidenceError("training commitment fields/encoding differ")
    if commitment["schema_version"] != TRAINING_COMMITMENT_SCHEMA:
        raise TrainingEvidenceError("training commitment schema is unsupported")
    if commitment["trace_filename"] != Path(trace_path).name:
        raise TrainingEvidenceError("training commitment filename differs")
    if sha256_file(trace_path) != commitment["trace_sha256"]:
        raise TrainingEvidenceError("training trace file hash differs from commitment")
    rows = verify_training_rows(_read_jsonl(trace_path))
    if len(rows) != commitment["count"] or rows[-1]["row_hash"] != commitment["final_hash"]:
        raise TrainingEvidenceError("training trace tail differs from commitment")
    return rows


def tensor_sha256(tensor):
    """Hash tensor metadata and exact CPU bytes for parameter-event binding."""

    try:
        import torch
    except ImportError as exc:
        raise TrainingEvidenceError("tensor hashing requires PyTorch") from exc
    if not isinstance(tensor, torch.Tensor):
        raise TrainingEvidenceError("parameter inventory contains a non-tensor value")
    value = tensor.detach().cpu().contiguous()
    metadata = canonical_json(
        {"dtype": str(value.dtype), "shape": list(value.shape)}
    ).encode("utf-8")
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(metadata + b"\n" + raw).hexdigest()


def _normalize_visual_event(event):
    if not isinstance(event, Mapping) or set(event) != _VISUAL_EVENT_FIELDS:
        raise TrainingEvidenceError(
            "visual parameter event fields differ; "
            f"expected={sorted(_VISUAL_EVENT_FIELDS)}, "
            f"found={sorted(event) if isinstance(event, Mapping) else type(event).__name__}"
        )
    result = dict(event)
    for field in ("optimizer_event_id", "parameter_name", "dtype"):
        _nonempty_string(result[field], f"visual parameter {field}")
    for field in ("requires_grad", "optimizer_member", "gradient_finite"):
        if not isinstance(result[field], bool):
            raise TrainingEvidenceError(f"visual parameter {field} must be boolean")
    result["numel"] = _nonnegative_int(result["numel"], "visual parameter numel", positive=True)
    for field in ("before_sha256", "after_sha256"):
        _sha(result[field], f"visual parameter {field}")
    gradient_sha = result["gradient_sha256"]
    if gradient_sha is not None:
        _sha(gradient_sha, "visual parameter gradient_sha256")
    result["gradient_norm"] = _finite(
        result["gradient_norm"], "visual parameter gradient_norm"
    )
    result["delta_norm"] = _finite(
        result["delta_norm"], "visual parameter delta_norm"
    )
    if result["gradient_finite"] != (gradient_sha is not None):
        raise TrainingEvidenceError(
            "visual parameter gradient digest/finite flag is inconsistent"
        )
    if not result["requires_grad"] and (
        result["optimizer_member"]
        or gradient_sha is not None
        or result["gradient_norm"] != 0.0
    ):
        raise TrainingEvidenceError(
            "frozen visual parameter cannot belong to optimizer or record a gradient"
        )
    return json.loads(canonical_json(result))


def build_visual_parameter_rows(events):
    rows = []
    previous = GENESIS_HASH
    seen = set()
    for sequence, raw in enumerate(events):
        event = _normalize_visual_event(raw)
        identity = (event["optimizer_event_id"], event["parameter_name"])
        if identity in seen:
            raise TrainingEvidenceError(
                "visual parameter trace repeats an optimizer-event/parameter pair"
            )
        row = {
            "schema_version": VISUAL_PARAMETER_TRACE_SCHEMA,
            "sequence": sequence,
            "previous_hash": previous,
            **event,
        }
        row["row_hash"] = _row_hash(row)
        rows.append(row)
        previous = row["row_hash"]
        seen.add(identity)
    if not rows:
        raise TrainingEvidenceError("visual parameter trace cannot be empty")
    return tuple(rows)


def persist_visual_parameter_trace(trace_path, commitment_path, events):
    rows = build_visual_parameter_rows(events)
    encoded = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    commitment = {
        "schema_version": VISUAL_PARAMETER_COMMITMENT_SCHEMA,
        "trace_filename": Path(trace_path).name,
        "trace_sha256": hashlib.sha256(encoded).hexdigest(),
        "count": len(rows),
        "final_hash": rows[-1]["row_hash"],
    }
    try:
        publish_exclusive_pair(
            trace_path,
            encoded,
            commitment_path,
            (canonical_json(commitment) + "\n").encode("utf-8"),
        )
    except EvidenceBundleError as exc:
        raise TrainingEvidenceError(str(exc)) from exc
    return commitment


def load_visual_parameter_trace(trace_path, commitment_path):
    commitment = _load_json_object(commitment_path, "visual parameter commitment")
    required = {
        "schema_version",
        "trace_filename",
        "trace_sha256",
        "count",
        "final_hash",
    }
    if set(commitment) != required:
        raise TrainingEvidenceError("visual parameter commitment fields differ")
    if commitment["schema_version"] != VISUAL_PARAMETER_COMMITMENT_SCHEMA:
        raise TrainingEvidenceError("visual parameter commitment schema is unsupported")
    if commitment["trace_filename"] != Path(trace_path).name:
        raise TrainingEvidenceError("visual parameter commitment filename differs")
    if sha256_file(trace_path) != commitment["trace_sha256"]:
        raise TrainingEvidenceError("visual parameter trace hash differs from commitment")
    raw_rows = _read_jsonl(trace_path)
    rebuilt = build_visual_parameter_rows(
        [{field: row[field] for field in _VISUAL_EVENT_FIELDS} for row in raw_rows]
    )
    if tuple(raw_rows) != rebuilt:
        raise TrainingEvidenceError("visual parameter trace chain or encoding differs")
    if len(rebuilt) != commitment["count"] or rebuilt[-1]["row_hash"] != commitment["final_hash"]:
        raise TrainingEvidenceError("visual parameter trace tail differs from commitment")
    return rebuilt


def _matches_parameter_prefix(name, prefixes):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


class VisualParameterEventRecorder:
    """Record complete per-event evidence for a configured visual parameter scope."""

    def __init__(self, model, optimizer, *, parameter_prefixes):
        prefixes = tuple(parameter_prefixes)
        if not prefixes or not all(isinstance(value, str) and value for value in prefixes):
            raise TrainingEvidenceError(
                "visual parameter recorder prefixes must be non-empty strings"
            )
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if _matches_parameter_prefix(name, prefixes)
        }
        if not parameters:
            raise TrainingEvidenceError(
                "visual parameter recorder scope selects no model parameters"
            )
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.parameter_prefixes = prefixes
        self.parameter_names = tuple(sorted(parameters))
        self._optimizer_parameter_ids = optimizer_parameter_ids
        self._events = []
        self._staged = None

    @property
    def events(self):
        return tuple(json.loads(canonical_json(event)) for event in self._events)

    def state_dict(self):
        if self._staged is not None:
            raise TrainingEvidenceError(
                "cannot snapshot visual parameter evidence during a staged update"
            )
        return {
            "parameter_prefixes": list(self.parameter_prefixes),
            "parameter_names": list(self.parameter_names),
            "events": [json.loads(canonical_json(event)) for event in self._events],
        }

    def load_state_dict(self, state):
        required = {"parameter_prefixes", "parameter_names", "events"}
        if not isinstance(state, Mapping) or set(state) != required:
            raise TrainingEvidenceError("visual parameter recorder state fields differ")
        if tuple(state["parameter_prefixes"]) != self.parameter_prefixes:
            raise TrainingEvidenceError("visual parameter recorder prefixes cannot change")
        if tuple(state["parameter_names"]) != self.parameter_names:
            raise TrainingEvidenceError("visual parameter recorder inventory cannot change")
        events = [_normalize_visual_event(event) for event in state["events"]]
        build_visual_parameter_rows(events) if events else None
        self._events = events
        self._staged = None

    def _selected_parameters(self, model):
        parameters = dict(model.named_parameters())
        selected = {
            name
            for name in parameters
            if _matches_parameter_prefix(name, self.parameter_prefixes)
        }
        if selected != set(self.parameter_names):
            raise TrainingEvidenceError(
                "visual parameter inventory changed during training"
            )
        return {name: parameters[name] for name in self.parameter_names}

    def capture_before(self, model, optimizer):
        import torch

        if self._staged is not None:
            raise TrainingEvidenceError(
                "visual parameter recorder already has a staged update"
            )
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_parameter_ids != self._optimizer_parameter_ids:
            raise TrainingEvidenceError(
                "optimizer parameter membership changed during visual evidence capture"
            )
        staged = {}
        for name, parameter in self._selected_parameters(model).items():
            value = parameter.detach().cpu().contiguous().clone()
            gradient = parameter.grad
            if gradient is None:
                gradient_sha256 = None
                gradient_norm = 0.0
            else:
                gradient = gradient.detach()
                if not torch.isfinite(gradient).all().item():
                    raise TrainingEvidenceError(
                        f"visual parameter gradient is non-finite: {name}"
                    )
                gradient_sha256 = tensor_sha256(gradient)
                gradient_norm = float(gradient.float().norm().item())
            staged[name] = {
                "before": value,
                "before_sha256": tensor_sha256(value),
                "requires_grad": bool(parameter.requires_grad),
                "optimizer_member": id(parameter) in optimizer_parameter_ids,
                "numel": int(parameter.numel()),
                "dtype": str(parameter.dtype),
                "gradient_sha256": gradient_sha256,
                "gradient_finite": gradient_sha256 is not None,
                "gradient_norm": gradient_norm,
            }
        self._staged = staged

    def record_after(self, optimizer_event_id, model):
        _nonempty_string(optimizer_event_id, "visual optimizer_event_id")
        if self._staged is None:
            raise TrainingEvidenceError(
                "visual parameter recorder has no staged pre-update evidence"
            )
        events = []
        for name, parameter in self._selected_parameters(model).items():
            before = self._staged[name]
            after = parameter.detach().cpu().contiguous()
            delta_norm = float((after.float() - before["before"].float()).norm().item())
            events.append(
                _normalize_visual_event(
                    {
                        "optimizer_event_id": optimizer_event_id,
                        "parameter_name": name,
                        "requires_grad": before["requires_grad"],
                        "optimizer_member": before["optimizer_member"],
                        "numel": before["numel"],
                        "dtype": before["dtype"],
                        "before_sha256": before["before_sha256"],
                        "after_sha256": tensor_sha256(after),
                        "gradient_sha256": before["gradient_sha256"],
                        "gradient_finite": before["gradient_finite"],
                        "gradient_norm": before["gradient_norm"],
                        "delta_norm": delta_norm,
                    }
                )
            )
        self._events.extend(events)
        self._staged = None
        return tuple(events)

    def persist(self, trace_path, commitment_path):
        if self._staged is not None:
            raise TrainingEvidenceError(
                "cannot persist visual parameter evidence during a staged update"
            )
        if not self._events:
            raise TrainingEvidenceError("cannot persist empty visual parameter evidence")
        return persist_visual_parameter_trace(trace_path, commitment_path, self._events)


def _checkpoint_parameter_inventory(checkpoint_path, prefixes, *, checkpoint_bytes=None):
    try:
        import torch

        checkpoint = torch.load(
            io.BytesIO(checkpoint_bytes)
            if checkpoint_bytes is not None
            else checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise TrainingEvidenceError(
            f"cannot load checkpoint parameter inventory: {exc}"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise TrainingEvidenceError("checkpoint must contain a parameter mapping")
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            checkpoint = candidate
            break
    inventory = {}
    for name, tensor in checkpoint.items():
        name = str(name)
        if not _matches_parameter_prefix(name, prefixes):
            continue
        try:
            inventory[name] = {
                "numel": int(tensor.numel()),
                "dtype": str(tensor.dtype),
                "sha256": tensor_sha256(tensor),
            }
        except (AttributeError, TypeError) as exc:
            raise TrainingEvidenceError(
                f"checkpoint visual parameter is not a tensor: {name}"
            ) from exc
    if not inventory:
        raise TrainingEvidenceError(
            "checkpoint contains no parameters in the configured visual scope"
        )
    return dict(sorted(inventory.items()))


def derive_visual_parameter_evidence(
    cfg,
    *,
    variant,
    checkpoint_path,
    training_trace_path,
    training_commitment_path,
    visual_trace_path,
    visual_commitment_path,
    checkpoint_bytes=None,
):
    """Derive C2 update facts from complete parameter events and final checkpoint."""

    contract = getattr(cfg, "visual_parameter_contract", None)
    if not isinstance(contract, Mapping) or set(contract) != {
        "parameter_prefixes",
        "adapted_trainable_prefixes",
    }:
        raise TrainingEvidenceError(
            "C2 config requires an exact visual_parameter_contract"
        )
    prefixes = tuple(contract["parameter_prefixes"])
    trainable_prefixes = tuple(contract["adapted_trainable_prefixes"])
    for values, label in (
        (prefixes, "visual parameter prefixes"),
        (trainable_prefixes, "adapted trainable prefixes"),
    ):
        if not values or not all(isinstance(value, str) and value for value in values):
            raise TrainingEvidenceError(f"{label} must be non-empty strings")
    if variant not in {"frozen", "adapted"}:
        raise TrainingEvidenceError("C2 visual parameter variant is unsupported")
    inventory = _checkpoint_parameter_inventory(
        checkpoint_path, prefixes, checkpoint_bytes=checkpoint_bytes
    )
    expected_trainable = {
        name for name in inventory if _matches_parameter_prefix(name, trainable_prefixes)
    }
    if not expected_trainable:
        raise TrainingEvidenceError(
            "visual parameter contract selects no adapted trainable parameters"
        )
    training_rows = load_training_trace(training_trace_path, training_commitment_path)
    if any(row["skipped"] for row in training_rows):
        raise TrainingEvidenceError(
            "C2 visual parameter evidence does not permit skipped optimizer events"
        )
    event_ids = [row["event_id"] for row in training_rows]
    visual_rows = load_visual_parameter_trace(
        visual_trace_path, visual_commitment_path
    )
    grouped = {event_id: {} for event_id in event_ids}
    for row in visual_rows:
        event_id = row["optimizer_event_id"]
        if event_id not in grouped:
            raise TrainingEvidenceError(
                "visual parameter trace refers to an unknown optimizer event"
            )
        grouped[event_id][row["parameter_name"]] = row
    names = set(inventory)
    previous_after = {}
    nonzero_grad_events = 0
    changed_events = 0
    for event_id in event_ids:
        records = grouped[event_id]
        if set(records) != names:
            raise TrainingEvidenceError(
                "visual parameter trace does not cover the complete checkpoint inventory"
            )
        for name in sorted(names):
            record = records[name]
            checkpoint_record = inventory[name]
            if (
                record["numel"] != checkpoint_record["numel"]
                or record["dtype"] != checkpoint_record["dtype"]
            ):
                raise TrainingEvidenceError(
                    f"visual parameter geometry differs from checkpoint: {name}"
                )
            if name in previous_after and record["before_sha256"] != previous_after[name]:
                raise TrainingEvidenceError(
                    f"visual parameter event continuity differs: {name}"
                )
            hash_changed = record["before_sha256"] != record["after_sha256"]
            if hash_changed != (record["delta_norm"] > 0.0):
                raise TrainingEvidenceError(
                    f"visual parameter hash/delta evidence is inconsistent: {name}"
                )
            is_trainable = name in expected_trainable and variant == "adapted"
            if is_trainable:
                if not (
                    record["requires_grad"]
                    and record["optimizer_member"]
                    and record["gradient_finite"]
                    and record["gradient_sha256"] is not None
                ):
                    raise TrainingEvidenceError(
                        f"adapted visual parameter lacks optimizer/gradient evidence: {name}"
                    )
                nonzero_grad_events += int(record["gradient_norm"] > 0.0)
                changed_events += int(record["delta_norm"] > 0.0)
            else:
                if (
                    record["requires_grad"]
                    or record["optimizer_member"]
                    or record["gradient_sha256"] is not None
                    or record["gradient_norm"] != 0.0
                    or record["delta_norm"] != 0.0
                    or hash_changed
                ):
                    raise TrainingEvidenceError(
                        f"frozen visual parameter records an update: {name}"
                    )
            previous_after[name] = record["after_sha256"]
    for name, checkpoint_record in inventory.items():
        if previous_after.get(name) != checkpoint_record["sha256"]:
            raise TrainingEvidenceError(
                f"visual parameter trace tail differs from checkpoint: {name}"
            )
    if variant == "adapted" and (nonzero_grad_events == 0 or changed_events == 0):
        raise TrainingEvidenceError(
            "adapted visual route lacks a nonzero gradient and parameter update"
        )
    return {
        "optimizer_events": len(event_ids),
        "visual_parameters": len(inventory),
        "adapted_trainable_parameters": len(expected_trainable) if variant == "adapted" else 0,
        "nonzero_gradient_parameter_events": nonzero_grad_events,
        "changed_parameter_events": changed_events,
        "inventory_sha256": canonical_json_sha256(inventory),
    }


def derive_training_cost(trace_path, commitment_path):
    rows = load_training_trace(trace_path, commitment_path)
    skipped = sum(int(row["skipped"]) for row in rows)
    elapsed = float(rows[-1]["elapsed_seconds"])
    identity = {field: rows[0][field] for field in _IDENTITY_FIELDS}
    return {
        "optimizer_events": len(rows),
        "successful_optimizer_events": len(rows) - skipped,
        "skipped_optimizer_events": skipped,
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "episodes": len(rows),
        "effective_batch_size": identity["effective_batch_size"],
        "gpu_hours": elapsed * identity["world_size"] / 3600.0,
        "wall_clock_sec": elapsed,
        "peak_vram_gb": max(row["peak_memory_bytes"] for row in rows) / float(1024**3),
        **{field: identity[field] for field in _IDENTITY_FIELDS},
    }


def derive_fixed_step_profile_measurements(
    trace_path,
    commitment_path,
    *,
    warmup_optimizer_events,
    measured_optimizer_events,
    expected_identity=None,
):
    """Derive fixed-step cost only from the committed optimizer-event trace."""

    warmup = _nonnegative_int(
        warmup_optimizer_events, "profile warmup optimizer events"
    )
    measured = _nonnegative_int(
        measured_optimizer_events,
        "profile measured optimizer events",
        positive=True,
    )
    rows = load_training_trace(trace_path, commitment_path)
    if len(rows) != warmup + measured:
        raise TrainingEvidenceError(
            "profile optimizer-event trace length differs from the fixed-step contract"
        )
    if any(row["skipped"] for row in rows):
        raise TrainingEvidenceError(
            "profile optimizer-event trace contains a skipped event"
        )
    identity = {field: rows[0][field] for field in _IDENTITY_FIELDS}
    if expected_identity is not None:
        normalized_expected = {
            field: expected_identity[field] for field in _IDENTITY_FIELDS
        }
        if identity != normalized_expected:
            raise TrainingEvidenceError(
                "profile optimizer-event trace identity differs from the launch config"
            )
    start_elapsed = rows[warmup - 1]["elapsed_seconds"] if warmup else 0.0
    elapsed = float(rows[-1]["elapsed_seconds"]) - float(start_elapsed)
    if elapsed <= 0:
        raise TrainingEvidenceError(
            "profile measured optimizer-event interval must be positive"
        )
    measured_rows = rows[warmup:]
    return {
        "warmup_optimizer_events": warmup,
        "measured_optimizer_events": measured,
        "total_optimizer_events": len(rows),
        "skipped_optimizer_events": 0,
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": max(row["peak_memory_bytes"] for row in measured_rows),
        "throughput_optimizer_events_per_second": measured / elapsed,
    }


def _load_json_object(path, label):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingEvidenceError(f"failed to load {label}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TrainingEvidenceError(f"{label} must contain one JSON object")
    return dict(payload)


def _formal_artifact_paths(artifacts, claim, *, bundle_root):
    expected = set(_COMMON_FORMAL_ARTIFACT_ROLES)
    if claim == "C2":
        expected.update(
            {"visual_parameter_trace", "visual_parameter_commitment"}
        )
    elif claim != "C1":
        raise TrainingEvidenceError("formal run claim must be C1 or C2")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected:
        found = set(artifacts) if isinstance(artifacts, Mapping) else set()
        raise TrainingEvidenceError(
            "formal run artifact roles differ; "
            f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
        )
    paths = {}
    for role, raw_path in artifacts.items():
        try:
            paths[role] = contained_file(
                raw_path, bundle_root, f"formal run artifact {role}"
            )
        except EvidenceBundleError as exc:
            raise TrainingEvidenceError(str(exc)) from exc
    return paths


def _profile_trust_root(cfg):
    try:
        root = cfg.launch_contract.attestation_trust_roots.profile
        root = dict(root)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TrainingEvidenceError(
            "formal run config lacks the execution attestation trust root"
        ) from exc
    if set(root) != {"key_id", "public_key"}:
        raise TrainingEvidenceError("formal run profile trust root fields differ")
    return root


def _validate_formal_ticket(path, *, entrypoint, seed, commit_sha, config_path):
    ticket = _load_json_object(path, f"formal {entrypoint} launch ticket")
    if set(ticket) != _LAUNCH_TICKET_FIELDS or ticket.get("schema_version") != LAUNCH_TICKET_SCHEMA:
        raise TrainingEvidenceError(f"formal {entrypoint} launch ticket schema/fields differ")
    if ticket.get("mode") != "formal" or ticket.get("commit_sha") != commit_sha:
        raise TrainingEvidenceError(f"formal {entrypoint} launch ticket mode/commit differs")
    if ticket.get("config_file_sha256") != sha256_file(config_path):
        raise TrainingEvidenceError(f"formal {entrypoint} launch ticket config hash differs")
    runtime = ticket.get("runtime_identity")
    runtime_fields = {
        "schema_version",
        "entrypoint",
        "seed",
        "run_id",
        "deterministic",
        "not_eval",
        "resume_checkpoint",
        "cfg_overrides",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != runtime_fields:
        raise TrainingEvidenceError(f"formal {entrypoint} runtime fields differ")
    if (
        runtime.get("entrypoint") != entrypoint
        or runtime.get("seed") != seed
        or runtime.get("deterministic") is not True
        or runtime.get("not_eval") is not False
    ):
        raise TrainingEvidenceError(f"formal {entrypoint} runtime differs from the run")
    if not isinstance(runtime.get("cfg_overrides"), Mapping) or set(runtime["cfg_overrides"]) - {
        "work_dir"
    }:
        raise TrainingEvidenceError(f"formal {entrypoint} runtime overrides differ")
    if ticket.get("profile_evidence") is None:
        raise TrainingEvidenceError(f"formal {entrypoint} ticket lacks profile evidence")
    return ticket


def _validate_formal_receipt(
    path,
    *,
    ticket,
    ticket_path,
    trust_root,
    commit_sha,
):
    signed = _load_json_object(path, "formal launch receipt")
    try:
        receipt = verify_launch_receipt(signed, trust_root=trust_root)
    except FullPetalLaunchError as exc:
        raise TrainingEvidenceError(f"formal launch receipt is not trusted: {exc}") from exc
    if set(receipt) != _LAUNCH_RECEIPT_FIELDS or receipt.get("schema_version") != LAUNCH_RECEIPT_SCHEMA:
        raise TrainingEvidenceError("formal launch receipt schema/fields differ")
    expected = {
        "mode": "formal",
        "commit_sha": commit_sha,
        "source_tree_sha256": ticket["source_tree_sha256"],
        "data_identity_sha256": ticket["data_identity"].get("identity_sha256"),
        "runtime_identity_sha256": canonical_json_sha256(ticket["runtime_identity"]),
        "resolved_config_sha256": ticket["resolved_config_sha256"],
        "scientific_config_sha256": ticket["scientific_config_sha256"],
        "launch_ticket": {
            "path": Path(ticket_path).name,
            "sha256": sha256_file(ticket_path),
        },
        "b0_artifact_sha256": ticket["b0_evidence"].get("sha256"),
        "review_artifact_sha256": ticket["review_evidence"].get("sha256"),
        "profile_artifact_sha256": ticket["profile_evidence"].get("sha256"),
        "world_size": 1,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise TrainingEvidenceError(f"formal launch receipt {field} differs")
    allocation = receipt.get("slurm_allocation")
    if (
        not isinstance(receipt.get("slurm_job_id"), str)
        or not receipt["slurm_job_id"].strip()
        or not isinstance(allocation, Mapping)
        or allocation.get("job_id") != receipt["slurm_job_id"]
        or allocation.get("state") != "RUNNING"
        or allocation.get("nodes") != 1
        or allocation.get("tasks") != 1
        or allocation.get("gpus") != 1
    ):
        raise TrainingEvidenceError("formal launch receipt lacks an active one-GPU allocation")
    return receipt


def validate_formal_run_artifacts(
    *, claim, variant, seed, commit_sha, artifacts, bundle_root
):
    """Validate the execution-bound run bundle before a formal signature is issued."""

    paths = _formal_artifact_paths(artifacts, claim, bundle_root=bundle_root)
    allowed_variants = {"C1": {"fixed", "rematch"}, "C2": {"frozen", "adapted"}}
    if variant not in allowed_variants[claim]:
        raise TrainingEvidenceError(f"formal {claim} variant is unsupported: {variant}")
    try:
        cfg = Config.fromfile(str(paths["config"]))
    except Exception as exc:
        raise TrainingEvidenceError(f"formal run config cannot be loaded: {exc}") from exc
    training_ticket = _validate_formal_ticket(
        paths["training_launch_ticket"],
        entrypoint="train",
        seed=seed,
        commit_sha=commit_sha,
        config_path=paths["config"],
    )
    evaluation_ticket = _validate_formal_ticket(
        paths["evaluation_launch_ticket"],
        entrypoint="test",
        seed=seed,
        commit_sha=commit_sha,
        config_path=paths["config"],
    )
    shared = _LAUNCH_TICKET_FIELDS - {"runtime_identity"}
    for field in shared:
        if training_ticket[field] != evaluation_ticket[field]:
            raise TrainingEvidenceError(
                f"formal training/evaluation launch tickets disagree on {field}"
            )
    if training_ticket["runtime_identity"]["resume_checkpoint"] is not None:
        raise TrainingEvidenceError("formal training ticket unexpectedly resumes")
    checkpoint = evaluation_ticket["runtime_identity"]["resume_checkpoint"]
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise TrainingEvidenceError(
            "formal evaluation ticket checkpoint fields differ"
        )
    try:
        bound_checkpoint_path, checkpoint_bytes = read_verified_bundle_bytes(
            {"path": checkpoint["path"], "sha256": checkpoint["sha256"]},
            bundle_root,
            "formal evaluation checkpoint",
        )
    except EvidenceBundleError as exc:
        raise TrainingEvidenceError(str(exc)) from exc
    if (
        bound_checkpoint_path != paths["checkpoint"]
        or len(checkpoint_bytes) != checkpoint["size_bytes"]
    ):
        raise TrainingEvidenceError(
            "formal evaluation ticket does not bind the supplied checkpoint"
        )
    if training_ticket["resolved_config_sha256"] != resolved_config_sha256(cfg):
        raise TrainingEvidenceError("formal ticket resolved config digest differs")
    if training_ticket["scientific_config_sha256"] != resolved_config_sha256(
        cfg, scientific=True
    ):
        raise TrainingEvidenceError("formal ticket scientific config digest differs")
    data_identity = _load_json_object(paths["data_identity"], "formal data identity")
    if data_identity != training_ticket["data_identity"]:
        raise TrainingEvidenceError("formal data identity differs from the launch tickets")
    evaluator_spec = _load_json_object(paths["evaluator_spec"], "formal evaluator spec")
    try:
        validate_evaluation_artifact_bindings(
            cfg,
            data_identity,
            ground_truth_path=paths["ground_truth"],
            allowed_videos_path=paths["allowed_videos"],
            evaluator_spec=evaluator_spec,
        )
        expected_training_identity = derive_training_trace_identity(
            cfg, data_identity, seed=seed, world_size=1
        )
    except IdentityError as exc:
        raise TrainingEvidenceError(f"formal config/data binding is invalid: {exc}") from exc
    cost = derive_training_cost(paths["training_trace"], paths["training_commitment"])
    if {field: cost[field] for field in expected_training_identity} != expected_training_identity:
        raise TrainingEvidenceError(
            "formal training trace identity differs from config/data-derived values"
        )
    if claim == "C2":
        derive_visual_parameter_evidence(
            cfg,
            variant=variant,
            checkpoint_path=paths["checkpoint"],
            training_trace_path=paths["training_trace"],
            training_commitment_path=paths["training_commitment"],
            visual_trace_path=paths["visual_parameter_trace"],
            visual_commitment_path=paths["visual_parameter_commitment"],
            checkpoint_bytes=checkpoint_bytes,
        )
    try:
        load_verified_ledger(paths["ledger"], paths["commitment"])
    except LedgerError as exc:
        raise TrainingEvidenceError(f"formal emission ledger is invalid: {exc}") from exc
    trust_root = _profile_trust_root(cfg)
    _validate_formal_receipt(
        paths["training_launch_receipt"],
        ticket=training_ticket,
        ticket_path=paths["training_launch_ticket"],
        trust_root=trust_root,
        commit_sha=commit_sha,
    )
    _validate_formal_receipt(
        paths["evaluation_launch_receipt"],
        ticket=evaluation_ticket,
        ticket_path=paths["evaluation_launch_ticket"],
        trust_root=trust_root,
        commit_sha=commit_sha,
    )
    return paths


def build_formal_run_manifest(
    *,
    claim,
    variant,
    seed,
    commit_sha,
    protocol_sha256,
    artifacts,
    bundle_root,
    private_key_path,
    key_id,
):
    claim = _nonempty_string(claim, "formal run claim")
    variant = _nonempty_string(variant, "formal run variant")
    seed = _nonnegative_int(seed, "formal run seed")
    commit_sha = _git_sha(commit_sha, "formal run commit_sha")
    paths = validate_formal_run_artifacts(
        claim=claim,
        variant=variant,
        seed=seed,
        commit_sha=commit_sha,
        artifacts=artifacts,
        bundle_root=bundle_root,
    )
    references = {}
    for role, path in sorted(paths.items()):
        role = _nonempty_string(role, "formal run artifact role")
        try:
            references[role] = bundle_file_reference(
                path, bundle_root, f"formal run artifact {role}"
            )
        except EvidenceBundleError as exc:
            raise TrainingEvidenceError(str(exc)) from exc
    evaluation_ticket = _load_json_object(
        paths["evaluation_launch_ticket"], "formal evaluation launch ticket"
    )
    bound_checkpoint = evaluation_ticket["runtime_identity"]["resume_checkpoint"]
    if references["checkpoint"] != {
        "path": bound_checkpoint["path"],
        "sha256": bound_checkpoint["sha256"],
    }:
        raise TrainingEvidenceError(
            "checkpoint changed between validation and formal manifest signing"
        )
    body = {
        "schema_version": FORMAL_RUN_MANIFEST_SCHEMA,
        "claim": claim,
        "variant": variant,
        "seed": seed,
        "commit_sha": commit_sha,
        "protocol_sha256": _sha(protocol_sha256, "protocol_sha256"),
        "artifacts": references,
    }
    try:
        return _sign_payload(
            body,
            private_key_path=private_key_path,
            key_id=key_id,
            role=FORMAL_RUN_ATTESTATION_ROLE,
        )
    except AttestationError as exc:
        raise TrainingEvidenceError(f"failed to attest formal run manifest: {exc}") from exc


def verify_formal_run_manifest(payload, *, trust_root, base_dir):
    try:
        body = verify_payload(
            payload,
            trust_root=trust_root,
            role=FORMAL_RUN_ATTESTATION_ROLE,
        )
    except AttestationError as exc:
        raise TrainingEvidenceError(f"formal run attestation is invalid: {exc}") from exc
    required = {
        "schema_version",
        "claim",
        "variant",
        "seed",
        "commit_sha",
        "protocol_sha256",
        "artifacts",
    }
    if set(body) != required or body["schema_version"] != FORMAL_RUN_MANIFEST_SCHEMA:
        raise TrainingEvidenceError("formal run manifest schema/fields differ")
    _nonempty_string(body["claim"], "formal run claim")
    _nonempty_string(body["variant"], "formal run variant")
    _nonnegative_int(body["seed"], "formal run seed")
    _git_sha(body["commit_sha"], "formal run commit_sha")
    _sha(body["protocol_sha256"], "protocol_sha256")
    if not isinstance(body["artifacts"], Mapping) or not body["artifacts"]:
        raise TrainingEvidenceError("formal run artifacts must be a non-empty mapping")
    for role, reference in body["artifacts"].items():
        _nonempty_string(role, "formal run artifact role")
        try:
            verify_bundle_reference(
                reference, base_dir, f"formal run artifact {role}"
            )
        except EvidenceBundleError as exc:
            raise TrainingEvidenceError(str(exc)) from exc
    return body


__all__ = [
    "FORMAL_RUN_ATTESTATION_ROLE",
    "FORMAL_RUN_MANIFEST_SCHEMA",
    "OptimizerEventTraceRecorder",
    "VisualParameterEventRecorder",
    "TRAINING_COMMITMENT_SCHEMA",
    "TRAINING_TRACE_SCHEMA",
    "VISUAL_PARAMETER_COMMITMENT_SCHEMA",
    "VISUAL_PARAMETER_TRACE_SCHEMA",
    "TrainingEvidenceError",
    "build_formal_run_manifest",
    "build_training_rows",
    "derive_training_cost",
    "derive_fixed_step_profile_measurements",
    "derive_visual_parameter_evidence",
    "load_training_trace",
    "persist_training_trace",
    "persist_visual_parameter_trace",
    "tensor_sha256",
    "verify_formal_run_manifest",
    "validate_formal_run_artifacts",
    "verify_training_rows",
]
