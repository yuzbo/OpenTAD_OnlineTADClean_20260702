"""Authenticated optimizer-event evidence for Full PETAL formal runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from collections.abc import Iterable, Mapping
import re
import time

from .full_petal_attestation import AttestationError, sign_payload, verify_payload


TRAINING_TRACE_SCHEMA = "full-petal-training-trace-v1"
TRAINING_COMMITMENT_SCHEMA = "full-petal-training-trace-commitment-v1"
FORMAL_RUN_MANIFEST_SCHEMA = "full-petal-formal-run-manifest-v1"
FORMAL_RUN_ATTESTATION_ROLE = "formal-run"
GENESIS_HASH = "0" * 64
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")

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


def _write_exclusive(path, data):
    path = Path(path)
    if path.exists():
        raise TrainingEvidenceError(f"refusing to overwrite training evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        written = os.write(descriptor, data)
        if written != len(data):
            raise OSError("partial training evidence write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def persist_training_trace(trace_path, commitment_path, events):
    rows = build_training_rows(events)
    encoded = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    _write_exclusive(trace_path, encoded)
    commitment = {
        "schema_version": TRAINING_COMMITMENT_SCHEMA,
        "trace_filename": Path(trace_path).name,
        "trace_sha256": hashlib.sha256(encoded).hexdigest(),
        "count": len(rows),
        "final_hash": rows[-1]["row_hash"],
    }
    _write_exclusive(
        commitment_path,
        (canonical_json(commitment) + "\n").encode("utf-8"),
    )
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
        **{field: identity[field] for field in _IDENTITY_FIELDS[:-1]},
    }


def build_formal_run_manifest(
    *,
    claim,
    variant,
    seed,
    commit_sha,
    protocol_sha256,
    artifacts,
    private_key_path,
    key_id,
):
    claim = _nonempty_string(claim, "formal run claim")
    variant = _nonempty_string(variant, "formal run variant")
    seed = _nonnegative_int(seed, "formal run seed")
    commit_sha = _git_sha(commit_sha, "formal run commit_sha")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise TrainingEvidenceError("formal run artifacts must be a non-empty mapping")
    references = {}
    for role, path in sorted(artifacts.items()):
        role = _nonempty_string(role, "formal run artifact role")
        resolved = Path(path).resolve()
        references[role] = {"path": str(resolved), "sha256": sha256_file(resolved)}
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
        return sign_payload(
            body,
            private_key_path=private_key_path,
            key_id=key_id,
            role=FORMAL_RUN_ATTESTATION_ROLE,
        )
    except AttestationError as exc:
        raise TrainingEvidenceError(f"failed to attest formal run manifest: {exc}") from exc


def verify_formal_run_manifest(payload, *, trust_root):
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
        if set(reference) != {"path", "sha256"}:
            raise TrainingEvidenceError(f"formal run artifact reference differs: {role}")
        if sha256_file(reference["path"]) != reference["sha256"]:
            raise TrainingEvidenceError(f"formal run artifact hash differs: {role}")
    return body


__all__ = [
    "FORMAL_RUN_ATTESTATION_ROLE",
    "FORMAL_RUN_MANIFEST_SCHEMA",
    "OptimizerEventTraceRecorder",
    "TRAINING_COMMITMENT_SCHEMA",
    "TRAINING_TRACE_SCHEMA",
    "TrainingEvidenceError",
    "build_formal_run_manifest",
    "build_training_rows",
    "derive_training_cost",
    "load_training_trace",
    "persist_training_trace",
    "verify_formal_run_manifest",
    "verify_training_rows",
]
