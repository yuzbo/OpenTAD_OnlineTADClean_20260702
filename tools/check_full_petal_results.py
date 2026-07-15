#!/usr/bin/env python3
"""Evaluate separate Full PETAL C1, C2, and project result gates."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

from mmengine import Config


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentad.evaluations.online_budgeted_map import OnlineAPBudgeted
from opentad.utils.evidence_bundle import (
    EvidenceBundleError,
    read_stable_file_bytes,
    read_verified_bundle_bytes,
    read_verified_path_bytes,
    strict_json_from_bytes,
)
from opentad.utils.full_petal_attestation import (
    AttestationError,
    verify_payload,
)
from opentad.utils.full_petal_b0 import (
    B0EvidenceError,
    validate_b0_evidence,
)
from opentad.utils.full_petal_data_contract import (
    ContractValidationError,
    FINEACTION_SOURCE_KEYS,
    validate_fineaction_qualification_report,
    validate_reporting_artifacts_from_sources,
    verify_content_hash,
)
from opentad.utils.full_petal_identity import (
    DATA_IDENTITY_SCHEMA,
    EVALUATOR_SPEC_SCHEMA,
    IdentityError,
    RUNTIME_IDENTITY_SCHEMA,
    authorization_only_diff,
    derive_training_trace_identity,
    validate_evaluation_artifact_bindings,
)
from opentad.utils.full_petal_launch import (
    FullPetalLaunchError,
    LAUNCH_RECEIPT_ATTESTATION_ROLE,
    LAUNCH_RECEIPT_SCHEMA,
    LAUNCH_TICKET_SCHEMA,
    PROFILE_ATTESTATION_ROLE,
    PROFILE_SCHEMA,
    REVIEW_ATTESTATION_ROLE,
    REVIEW_SCHEMA,
    _validate_g0_artifact,
    resolved_config_sha256,
    verify_launch_receipt,
)
from opentad.utils.full_petal_training_evidence import (
    TrainingEvidenceError,
    derive_fixed_step_profile_measurements,
    derive_training_cost,
    derive_visual_parameter_evidence,
    verify_formal_run_manifest,
)
from opentad.utils.full_petal_runtime_attestation import (
    RuntimeAttestationError,
    runtime_profile_trust_root,
)


SCHEMA_VERSION = "full_petal_result_gate.v3"
TRUST_CONFIG = REPO_ROOT / "configs" / "causaltad" / "thumos_pes_q2_base.py"

C1_VARIANTS = {
    "fixed": "fixed",
    "persist_fixed": "fixed",
    "persistent_fixed": "fixed",
    "fixed_birth_slot": "fixed",
    "thumos_pes_q2_persist_fixed": "fixed",
    "rematch": "rematch",
    "persist_rematch": "rematch",
    "persistent_rematch": "rematch",
    "prefix_rematch_active_pool": "rematch",
    "thumos_pes_q2_persist_rematch": "rematch",
}
C2_VARIANTS = {
    "frozen": "frozen",
    "frozen_encoder": "frozen",
    "raw_frozen": "frozen",
    "adapted": "adapted",
    "peft": "adapted",
    "adapter": "adapted",
    "lora": "adapted",
    "top_block": "adapted",
    "visual_adapted": "adapted",
}

METRIC_ALIASES = {
    "average_mAP": ("average_mAP",),
    "average_mOnlineAP": ("average_mOnlineAP",),
    "identity_recall": ("identity_recall",),
    "duplicate_per_gt": ("duplicate_per_gt",),
    "fragmentation_rate": ("fragmentation_rate",),
    "false_emission_rate": ("false_emission_rate",),
    "endpoint_latency_frames_mean": (
        "endpoint_latency_frames_mean",
        "endpoint_detection_latency_frames_mean",
    ),
    "high_tiou_mAP": ("high_tiou_mAP", "mAP@0.7"),
    "short_action_mAP": ("short_action_mAP",),
}

COMMON_RUN_ARTIFACTS = {
    "ledger",
    "commitment",
    "ground_truth",
    "allowed_videos",
    "fit_core",
    "feature_cache_manifest",
    "evaluator_spec",
    "config",
    "resolved_config",
    "checkpoint",
    "data_identity",
    "training_launch_ticket",
    "training_launch_receipt",
    "evaluation_launch_ticket",
    "evaluation_launch_receipt",
    "training_trace",
    "training_commitment",
}

PROTOCOL_REQUIRED_VALUES = {
    "decision_cadence": "packet_end",
    "nms": False,
    "offline_nms": False,
    "immutable_emissions": True,
    "load_from_raw_predictions": False,
    "runtime_identity": "persistent_slots",
    "lifecycle": "canonical_shared",
    "birth_rule": "first_threshold_crossing",
    "start_parameterization": "scalar",
    "endpoint_parameterization": "binary_first_crossing",
}

C1_COST_NUMERIC_FIELDS = (
    "optimizer_events",
    "successful_optimizer_events",
    "skipped_optimizer_events",
    "input_tokens",
    "episodes",
    "effective_batch_size",
    "world_size",
    "gpu_hours",
    "wall_clock_sec",
    "peak_vram_gb",
)
C1_COST_IDENTITY_FIELDS = (
    "precision",
    "optimizer_config_sha256",
    "scheduler_config_sha256",
    "data_order_sha256",
    "loss_normalization_sha256",
)


class ResultGateError(ValueError):
    """Base class for result-gate validation errors."""


class ResultGateInputError(ResultGateError):
    """Raised for malformed or incomplete result artifacts."""


class ResultGateProtocolError(ResultGateError):
    """Raised when an artifact records a protocol violation or mismatch."""


_VERIFIED_BYTES = {}


def _cache_verified_bytes(path, payload):
    _VERIFIED_BYTES[str(Path(path).resolve())] = bytes(payload)


def _cached_bytes(path, label):
    key = str(Path(path).resolve())
    payload = _VERIFIED_BYTES.get(key)
    if payload is None:
        try:
            _, payload = read_stable_file_bytes(path, label)
        except EvidenceBundleError as exc:
            raise ResultGateInputError(str(exc)) from exc
        _cache_verified_bytes(path, payload)
    return payload


def _attestation_trust_roots():
    try:
        namespace = runpy.run_path(str(TRUST_CONFIG))
        roots = namespace["launch_contract"]["attestation_trust_roots"]
    except (OSError, KeyError, TypeError) as exc:
        raise ResultGateInputError(
            f"cannot load the repository trust roots from {TRUST_CONFIG}: {exc}"
        ) from exc
    if not isinstance(roots, dict) or set(roots) != {
        "b0",
        "review",
        "g0",
        "profile",
        "formal",
    }:
        raise ResultGateInputError("repository attestation trust roots are incomplete")
    return roots


def _route_launch_requirements():
    try:
        namespace = runpy.run_path(str(TRUST_CONFIG))
        launch = namespace["launch_contract"]
        profile = namespace["profile_contract"]
        return {
            "reviewer_id": launch["required_reviewer_id"],
            "review_scope": list(launch["required_review_scope"]),
            "warmup_events": profile["warmup_steps"],
            "measured_events": profile["measured_steps"],
            "world_size": profile["world_size"],
            "dimensions": {
                "batch_size": namespace["solver"]["train"]["batch_size"],
                "chunk_size": namespace["chunk_size"],
                "memory_size": namespace["memory_size"],
                "slot_count": namespace["num_slots"],
            },
            "fineaction_trust_roots": namespace["reporting_contract"][
                "fineaction_trust_roots"
            ],
        }
    except (OSError, KeyError, TypeError) as exc:
        raise ResultGateInputError(
            f"cannot load the repository launch requirements from {TRUST_CONFIG}: {exc}"
        ) from exc


def _canonical_json(value, label):
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ResultGateInputError(f"{label} must be canonical JSON data: {exc}") from exc


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value, label):
    return _sha256_bytes(_canonical_json(value, label).encode("utf-8"))


def _required_sha256(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ResultGateInputError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _required_git_sha(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ResultGateInputError(f"{label} must be a 40-character lowercase git SHA")
    return value


def _evidence_file(evidence, prefix, label):
    path_key = f"{prefix}_path"
    hash_key = f"{prefix}_sha256"
    if path_key not in evidence or hash_key not in evidence:
        raise ResultGateInputError(
            f"{label} requires verified evidence fields {path_key} and {hash_key}"
        )
    raw_path = evidence[path_key]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ResultGateInputError(f"{label}.{path_key} must be a non-empty path")
    path = Path(raw_path).expanduser()
    expected = _required_sha256(evidence[hash_key], f"{label}.{hash_key}")
    try:
        path, payload = read_verified_path_bytes(
            path, expected, f"{label}.{prefix} evidence"
        )
    except EvidenceBundleError as exc:
        raise ResultGateInputError(str(exc)) from exc
    _cache_verified_bytes(path, payload)
    return path, expected


def _bundle_reference_file(reference, base_dir, label):
    try:
        path, payload = read_verified_bundle_bytes(reference, base_dir, label)
    except EvidenceBundleError as exc:
        raise ResultGateInputError(str(exc)) from exc
    _cache_verified_bytes(path, payload)
    return path, reference["sha256"]


def _finite_number(value, label, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise ResultGateInputError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ResultGateInputError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ResultGateInputError(f"{label} must be a finite number")
    if minimum is not None and number < minimum:
        raise ResultGateInputError(f"{label} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ResultGateInputError(f"{label} must be <= {maximum}")
    return number


def _nonnegative_integer(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        qualifier = "positive" if positive else "non-negative"
        raise ResultGateInputError(f"{label} must be a {qualifier} integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ResultGateInputError(f"{label} must be a {qualifier} integer")
    return value


def _mean(values):
    values = list(values)
    return sum(values) / len(values)


def _claim_name(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"c1", "mechanism", "c1_mechanism"}:
        return "C1"
    if normalized in {"c2", "raw_video", "c2_raw_video"}:
        return "C2"
    raise ResultGateInputError(f"unexpected claim: {value!r}")


def _variant_name(claim, value):
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = C1_VARIANTS if claim == "C1" else C2_VARIANTS
    if normalized not in aliases:
        raise ResultGateInputError(f"unexpected {claim} variant: {value!r}")
    return aliases[normalized]


def _seed(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResultGateInputError(f"{label} must be a non-negative integer")
    return value


def _has_violations(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _finite_number(value, "protocol_violations", minimum=0.0) > 0
    if isinstance(value, (list, tuple, set, dict, str)):
        return bool(value)
    raise ResultGateInputError("protocol_violations has an unsupported type")


def _validate_row_protocol(row, claim, label):
    protocol = row.get("protocol")
    if not isinstance(protocol, dict) or not protocol:
        raise ResultGateInputError(f"{label}.protocol must be a non-empty object")
    expected_input = "matched_features" if claim == "C1" else "raw_video"
    if protocol.get("input") != expected_input:
        raise ResultGateProtocolError(
            f"{label}.protocol.input must be {expected_input!r} for {claim}"
        )
    for field, expected in PROTOCOL_REQUIRED_VALUES.items():
        if field not in protocol:
            raise ResultGateInputError(f"{label}.protocol is missing required field {field}")
        if protocol[field] != expected:
            raise ResultGateProtocolError(
                f"{label}.protocol.{field} must be {expected!r}, found {protocol[field]!r}"
            )
    capacity = protocol.get("capacity")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ResultGateInputError(f"{label}.protocol.capacity must be a positive integer")
    _canonical_json(protocol, f"{label}.protocol")
    return protocol


def _metric_candidates(metrics, canonical_name):
    candidates = []
    for alias in METRIC_ALIASES[canonical_name]:
        if alias in metrics:
            candidates.append((alias, metrics[alias]))
    if canonical_name == "endpoint_latency_frames_mean":
        summary = metrics.get("endpoint_detection_latency_frames")
        if isinstance(summary, dict) and "mean" in summary:
            candidates.append(("endpoint_detection_latency_frames.mean", summary["mean"]))
    return candidates


def _metric(metrics, canonical_name, label, required=True):
    candidates = _metric_candidates(metrics, canonical_name)
    if not candidates:
        if required:
            raise ResultGateInputError(f"{label} is missing metric {canonical_name}")
        return None
    bounds = {
        "average_mAP": (0.0, 1.0),
        "average_mOnlineAP": (0.0, 1.0),
        "identity_recall": (0.0, 1.0),
        "duplicate_per_gt": (0.0, None),
        "fragmentation_rate": (0.0, 1.0),
        "false_emission_rate": (0.0, 1.0),
        "endpoint_latency_frames_mean": (0.0, None),
        "high_tiou_mAP": (0.0, 1.0),
        "short_action_mAP": (0.0, 1.0),
    }
    minimum, maximum = bounds[canonical_name]
    values = [
        _finite_number(
            value,
            f"{label}.{name}",
            minimum=minimum,
            maximum=maximum,
        )
        for name, value in candidates
    ]
    if any(abs(value - values[0]) > 1e-12 for value in values[1:]):
        names = [name for name, _ in candidates]
        raise ResultGateInputError(f"{label} has conflicting aliases for {canonical_name}: {names}")
    return values[0]


def _normalized_metrics(row, claim, label):
    metrics = row.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ResultGateInputError(f"{label}.metrics must be a non-empty object")
    if claim == "C1":
        required = (
            "average_mAP",
            "average_mOnlineAP",
            "identity_recall",
            "duplicate_per_gt",
            "fragmentation_rate",
            "false_emission_rate",
            "endpoint_latency_frames_mean",
        )
        return {name: _metric(metrics, name, label) for name in required}
    normalized = {"average_mAP": _metric(metrics, "average_mAP", label)}
    for name in ("high_tiou_mAP", "short_action_mAP"):
        value = _metric(metrics, name, label, required=False)
        if value is not None:
            normalized[name] = value
    return normalized


def _normalized_cost(row, label):
    cost = row.get("cost")
    if not isinstance(cost, dict) or not cost:
        raise ResultGateInputError(f"{label}.cost must be a non-empty object for C1")
    required = set(C1_COST_NUMERIC_FIELDS) | set(C1_COST_IDENTITY_FIELDS)
    if set(cost) != required:
        raise ResultGateInputError(
            f"{label}.cost fields differ: expected {sorted(required)}, found {sorted(cost)}"
        )
    normalized = {
        field: _finite_number(cost[field], f"{label}.cost.{field}", minimum=0.0)
        for field in C1_COST_NUMERIC_FIELDS
    }
    for field in (
        "optimizer_events",
        "successful_optimizer_events",
        "skipped_optimizer_events",
        "input_tokens",
        "episodes",
        "effective_batch_size",
        "world_size",
    ):
        normalized[field] = _nonnegative_integer(
            cost[field],
            f"{label}.cost.{field}",
            positive=field
            in {
                "optimizer_events",
                "input_tokens",
                "episodes",
                "effective_batch_size",
                "world_size",
            },
        )
    if normalized["optimizer_events"] != (
        normalized["successful_optimizer_events"] + normalized["skipped_optimizer_events"]
    ):
        raise ResultGateInputError(
            f"{label}.cost optimizer event accounting is inconsistent"
        )
    if normalized["skipped_optimizer_events"] != 0:
        raise ResultGateProtocolError(f"{label}.cost records skipped optimizer events")
    precision = cost["precision"]
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ResultGateInputError(
            f"{label}.cost.precision must be one of fp32, fp16, or bf16"
        )
    normalized["precision"] = precision
    for field in C1_COST_IDENTITY_FIELDS[1:]:
        normalized[field] = _required_sha256(cost[field], f"{label}.cost.{field}")
    return normalized


def _load_evidence_json(path, label):
    try:
        return strict_json_from_bytes(
            _cached_bytes(path, label), label, require_object=True
        )
    except EvidenceBundleError as exc:
        raise ResultGateInputError(str(exc)) from exc


def _validate_data_identity_artifact(path, ticket_identity, label):
    identity = _load_evidence_json(path, f"{label}.data_identity")
    if identity != ticket_identity:
        raise ResultGateInputError(f"{label} data identity differs from its launch ticket")
    required = {
        "schema_version",
        "files",
        "feature_inventory_count",
        "feature_inventory_sha256",
        "optional_qualifications",
        "identity_sha256",
    }
    if set(identity) != required or identity["schema_version"] != DATA_IDENTITY_SCHEMA:
        raise ResultGateInputError(f"{label} data identity schema/fields differ")
    body = dict(identity)
    declared_identity = body.pop("identity_sha256")
    if _sha256_json(body, f"{label}.data_identity") != declared_identity:
        raise ResultGateInputError(f"{label} data identity digest is invalid")
    records = list(identity["files"].items())
    records.extend(
        (name, record)
        for name, record in identity["optional_qualifications"].items()
        if record is not None
    )
    for name, record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
            raise ResultGateInputError(f"{label} data record {name} fields differ")
        reference = {f"{name}_path": record["path"], f"{name}_sha256": record["sha256"]}
        record_path, _ = _evidence_file(reference, name, f"{label}.data_identity")
        if len(_cached_bytes(record_path, f"{label}.data_identity.{name}")) != record[
            "size_bytes"
        ]:
            raise ResultGateInputError(f"{label} data record {name} size differs")
    return identity


def _validate_signed_review(path, *, commit_sha, b0_sha256, label):
    signed = _load_evidence_json(path, f"{label}.signed_review")
    try:
        review = verify_payload(
            signed,
            trust_root=_attestation_trust_roots()["review"],
            role=REVIEW_ATTESTATION_ROLE,
        )
    except AttestationError as exc:
        raise ResultGateInputError(
            f"{label} independent review attestation is invalid: {exc}"
        ) from exc
    required = {
        "schema_version",
        "reviewer_id",
        "reviewed_commit",
        "b0_artifact_sha256",
        "scope",
        "verdict",
        "blocking_findings",
        "protocol_violations",
    }
    requirements = _route_launch_requirements()
    if set(review) != required or review["schema_version"] != REVIEW_SCHEMA:
        raise ResultGateInputError(f"{label} independent review schema/fields differ")
    if (
        review["reviewer_id"] != requirements["reviewer_id"]
        or review["reviewed_commit"] != commit_sha
        or review["b0_artifact_sha256"] != b0_sha256
        or review["scope"] != requirements["review_scope"]
    ):
        raise ResultGateInputError(f"{label} independent review binding differs")
    if (
        review["verdict"] != "PASS"
        or not isinstance(review["blocking_findings"], list)
        or review["blocking_findings"]
        or not isinstance(review["protocol_violations"], list)
        or review["protocol_violations"]
    ):
        raise ResultGateProtocolError(f"{label} independent review has not reached PASS")
    return review


def _validate_signed_profile(path, *, ticket, manifest, config_path, label):
    signed = _load_evidence_json(path, f"{label}.signed_profile")
    try:
        profile_receipt_path, _ = _bundle_reference_file(
            {
                "path": signed.get("launch_receipt_path"),
                "sha256": signed.get("launch_receipt_sha256"),
            },
            Path(path).parent,
            f"{label}.profile_launch_receipt",
        )
        profile_signed_receipt = _load_evidence_json(
            profile_receipt_path, f"{label}.profile_launch_receipt"
        )
        profile_receipt = verify_launch_receipt(
            profile_signed_receipt,
            trust_root=_attestation_trust_roots()["profile"],
        )
        dynamic_profile_root = runtime_profile_trust_root(
            profile_receipt["execution_session"]
        )
        profile = verify_payload(
            signed,
            trust_root=dynamic_profile_root,
            role=PROFILE_ATTESTATION_ROLE,
        )
    except (AttestationError, FullPetalLaunchError, RuntimeAttestationError) as exc:
        raise ResultGateInputError(
            f"{label} fixed-step profile attestation is invalid: {exc}"
        ) from exc
    required = {
        "schema_version",
        "status",
        "commit_sha",
        "source_tree_sha256",
        "data_identity_sha256",
        "runtime_identity_sha256",
        "resolved_config_sha256",
        "scientific_config_sha256",
        "launch_ticket_path",
        "launch_ticket_sha256",
        "launch_receipt_path",
        "launch_receipt_sha256",
        "slurm_job_id",
        "slurm_allocation",
        "world_size",
        "precision",
        "hardware",
        "dimensions",
        "measurements",
        "optimizer_event_trace",
        "optimizer_event_commitment",
    }
    requirements = _route_launch_requirements()
    if set(profile) != required or profile["schema_version"] != PROFILE_SCHEMA:
        raise ResultGateInputError(f"{label} fixed-step profile schema/fields differ")
    if profile["status"] != "PASS":
        raise ResultGateProtocolError(f"{label} fixed-step profile has not reached PASS")
    profile_commit = _required_git_sha(
        profile["commit_sha"], f"{label}.fixed_step_profile.commit_sha"
    )
    if profile_commit == manifest["commit_sha"]:
        raise ResultGateProtocolError(
            f"{label} formal commit is not later than the profile commit"
        )
    try:
        authorization_only_diff(REPO_ROOT, profile_commit, manifest["commit_sha"])
    except IdentityError as exc:
        raise ResultGateProtocolError(
            f"{label} profile-to-formal authorization diff is invalid: {exc}"
        ) from exc
    expected_bindings = {
        "source_tree_sha256": ticket["source_tree_sha256"],
        "data_identity_sha256": ticket["data_identity"]["identity_sha256"],
        "scientific_config_sha256": ticket["scientific_config_sha256"],
        "world_size": requirements["world_size"],
        "dimensions": requirements["dimensions"],
    }
    for field, expected in expected_bindings.items():
        if profile[field] != expected:
            raise ResultGateInputError(f"{label} fixed-step profile {field} differs")
    _required_sha256(
        profile["runtime_identity_sha256"],
        f"{label}.fixed_step_profile.runtime_identity_sha256",
    )
    _required_sha256(
        profile["resolved_config_sha256"],
        f"{label}.fixed_step_profile.resolved_config_sha256",
    )
    if profile["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ResultGateInputError(f"{label} fixed-step profile precision differs")
    hardware = profile["hardware"]
    if not isinstance(hardware, dict) or set(hardware) != {
        "gpu_name",
        "torch_version",
        "cuda_version",
    }:
        raise ResultGateInputError(f"{label} fixed-step profile hardware fields differ")
    for field in ("gpu_name", "torch_version"):
        if not isinstance(hardware[field], str) or not hardware[field].strip():
            raise ResultGateInputError(f"{label} fixed-step profile lacks {field}")

    measurements = profile["measurements"]
    measurement_fields = {
        "warmup_optimizer_events",
        "measured_optimizer_events",
        "total_optimizer_events",
        "skipped_optimizer_events",
        "measurement_start_after_event_id",
        "measurement_end_event_id",
        "elapsed_seconds",
        "peak_memory_bytes",
        "throughput_optimizer_events_per_second",
    }
    if not isinstance(measurements, dict) or set(measurements) != measurement_fields:
        raise ResultGateInputError(f"{label} fixed-step profile measurements differ")
    warmup = requirements["warmup_events"]
    measured = requirements["measured_events"]
    if (
        measurements["warmup_optimizer_events"] != warmup
        or measurements["measured_optimizer_events"] != measured
        or measurements["total_optimizer_events"] != warmup + measured
        or measurements["skipped_optimizer_events"] != 0
    ):
        raise ResultGateProtocolError(f"{label} fixed-step profile event counts differ")
    for field in (
        "measurement_start_after_event_id",
        "measurement_end_event_id",
    ):
        if not isinstance(measurements[field], str) or not measurements[field].strip():
            raise ResultGateInputError(
                f"{label} fixed-step profile {field} is invalid"
            )
    _finite_number(
        measurements["elapsed_seconds"],
        f"{label}.fixed_step_profile.elapsed_seconds",
        minimum=1e-12,
    )
    _nonnegative_integer(
        measurements["peak_memory_bytes"],
        f"{label}.fixed_step_profile.peak_memory_bytes",
    )
    _finite_number(
        measurements["throughput_optimizer_events_per_second"],
        f"{label}.fixed_step_profile.throughput",
        minimum=1e-12,
    )

    allocation = profile["slurm_allocation"]
    allocation_fields = {
        "job_id",
        "state",
        "user",
        "nodes",
        "tasks",
        "gpus",
        "command",
        "work_dir",
    }
    if not isinstance(allocation, dict) or set(allocation) != allocation_fields:
        raise ResultGateInputError(f"{label} profile Slurm allocation fields differ")
    if (
        allocation["job_id"] != profile["slurm_job_id"]
        or allocation["state"] != "RUNNING"
        or allocation["nodes"] != 1
        or allocation["tasks"] != requirements["world_size"]
        or allocation["gpus"] != requirements["world_size"]
    ):
        raise ResultGateProtocolError(f"{label} profile Slurm allocation differs")

    profile_ticket_path, _ = _bundle_reference_file(
        {
            "path": profile["launch_ticket_path"],
            "sha256": profile["launch_ticket_sha256"],
        },
        Path(path).parent,
        f"{label}.fixed_step_profile.profile_ticket",
    )
    profile_ticket = _load_evidence_json(
        profile_ticket_path, f"{label}.profile_launch_ticket"
    )
    ticket_fields = {
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
        "g0_evidence",
        "profile_evidence",
    }
    if (
        set(profile_ticket) != ticket_fields
        or profile_ticket["schema_version"] != LAUNCH_TICKET_SCHEMA
        or profile_ticket["mode"] != "profile"
        or profile_ticket["commit_sha"] != profile_commit
        or profile_ticket["profile_evidence"] is not None
        or profile_ticket["source_tree_sha256"] != ticket["source_tree_sha256"]
        or profile_ticket["resolved_config_sha256"]
        != profile["resolved_config_sha256"]
        or profile_ticket["scientific_config_sha256"]
        != ticket["scientific_config_sha256"]
        or profile_ticket["data_identity"] != ticket["data_identity"]
        or _sha256_json(
            profile_ticket["runtime_identity"], f"{label}.profile_runtime"
        )
        != profile["runtime_identity_sha256"]
    ):
        raise ResultGateInputError(f"{label} profile launch ticket binding differs")
    profile_runtime = profile_ticket["runtime_identity"]
    if (
        not isinstance(profile_runtime, dict)
        or profile_runtime.get("schema_version") != RUNTIME_IDENTITY_SCHEMA
        or profile_runtime.get("entrypoint") != "train"
        or profile_runtime.get("deterministic") is not True
        or profile_runtime.get("not_eval") is not False
        or profile_runtime.get("resume_checkpoint") is not None
    ):
        raise ResultGateProtocolError(
            f"{label} profile runtime identity is not a fresh deterministic train run"
        )
    expected_profile_receipt = {
        "schema_version": LAUNCH_RECEIPT_SCHEMA,
        "mode": "profile",
        "commit_sha": profile_commit,
        "source_tree_sha256": profile["source_tree_sha256"],
        "data_identity_sha256": profile["data_identity_sha256"],
        "runtime_identity_sha256": profile["runtime_identity_sha256"],
        "resolved_config_sha256": profile["resolved_config_sha256"],
        "scientific_config_sha256": profile["scientific_config_sha256"],
        "launch_ticket": {
            "path": profile_ticket_path.name,
            "sha256": profile["launch_ticket_sha256"],
        },
        "b0_artifact_sha256": profile_ticket["b0_evidence"]["sha256"],
        "review_artifact_sha256": profile_ticket["review_evidence"]["sha256"],
        "g0_artifact_sha256": (
            None
            if profile_ticket["g0_evidence"] is None
            else profile_ticket["g0_evidence"]["sha256"]
        ),
        "profile_artifact_sha256": None,
        "world_size": profile["world_size"],
        "slurm_job_id": profile["slurm_job_id"],
        "slurm_allocation": profile["slurm_allocation"],
        "execution_session": profile_receipt["execution_session"],
    }
    if profile_receipt != expected_profile_receipt:
        raise ResultGateInputError(
            f"{label} profile launch receipt binding differs"
        )

    profile_b0_reference = profile_ticket["b0_evidence"]
    profile_review_reference = profile_ticket["review_evidence"]
    for role, reference in (
        ("profile_b0", profile_b0_reference),
        ("profile_review", profile_review_reference),
    ):
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ResultGateInputError(f"{label} {role} reference differs")
    try:
        validate_b0_evidence(
            profile_b0_reference,
            base_dir=profile_ticket_path.parent,
            expected_commit=profile_commit,
            trust_root=_attestation_trust_roots()["b0"],
        )
    except B0EvidenceError as exc:
        raise ResultGateInputError(
            f"{label} profile B0 evidence is invalid: {exc}"
        ) from exc
    profile_review_path, profile_review_sha256 = _bundle_reference_file(
        profile_review_reference,
        profile_ticket_path.parent,
        f"{label}.profile_review",
    )
    del profile_review_sha256
    _validate_signed_review(
        profile_review_path,
        commit_sha=profile_commit,
        b0_sha256=profile_b0_reference["sha256"],
        label=f"{label}.profile",
    )

    trace_path, _ = _bundle_reference_file(
        profile["optimizer_event_trace"],
        Path(path).parent,
        f"{label}.profile_trace",
    )
    commitment_path, _ = _bundle_reference_file(
        profile["optimizer_event_commitment"],
        Path(path).parent,
        f"{label}.profile_commitment",
    )
    try:
        profile_cfg = _resolved_run_config(config_path, f"{label}.profile")
        expected_trace_identity = derive_training_trace_identity(
            profile_cfg,
            profile_ticket["data_identity"],
            seed=profile_runtime["seed"],
            world_size=requirements["world_size"],
        )
        derived_measurements = derive_fixed_step_profile_measurements(
            trace_path,
            commitment_path,
            warmup_optimizer_events=warmup,
            measured_optimizer_events=measured,
            expected_identity=expected_trace_identity,
            profiler_measurements=profile["measurements"],
            trace_bytes=_cached_bytes(trace_path, f"{label}.profile_trace"),
            commitment_bytes=_cached_bytes(
                commitment_path, f"{label}.profile_commitment"
            ),
            runtime_binding=profile_receipt["execution_session"],
        )
    except (OSError, ValueError, TrainingEvidenceError, IdentityError) as exc:
        raise ResultGateInputError(
            f"{label} profile optimizer-event evidence is invalid: {exc}"
        ) from exc
    if profile["measurements"] != derived_measurements:
        raise ResultGateInputError(
            f"{label} fixed-step profile measurements differ from its event trace"
        )
    return profile


def _validate_launch_ticket(
    path,
    receipt_path,
    manifest,
    paths,
    hashes,
    seed,
    label,
    *,
    entrypoint,
):
    ticket_role = "training" if entrypoint == "train" else "evaluation"
    ticket = _load_evidence_json(path, f"{label}.launch_ticket")
    required = {
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
        "g0_evidence",
        "profile_evidence",
    }
    if set(ticket) != required or ticket["schema_version"] != LAUNCH_TICKET_SCHEMA:
        raise ResultGateInputError(f"{label} launch ticket schema/fields differ")
    if ticket["mode"] != "formal" or ticket["commit_sha"] != manifest["commit_sha"]:
        raise ResultGateInputError(f"{label} launch ticket mode/commit differs")
    if ticket["config_file_sha256"] != hashes["config"]:
        raise ResultGateInputError(f"{label} launch ticket config hash differs")
    runtime = ticket["runtime_identity"]
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
    if (
        not isinstance(runtime, dict)
        or set(runtime) != runtime_fields
        or runtime.get("schema_version") != RUNTIME_IDENTITY_SCHEMA
    ):
        raise ResultGateInputError(f"{label} launch ticket runtime schema/fields differ")
    if runtime.get("seed") != seed:
        raise ResultGateInputError(f"{label} launch ticket runtime seed differs")
    if runtime.get("entrypoint") != entrypoint:
        raise ResultGateInputError(f"{label} launch ticket entrypoint differs")
    _nonnegative_integer(runtime.get("run_id"), f"{label}.runtime.run_id")
    if runtime.get("deterministic") is not True or runtime.get("not_eval") is not False:
        raise ResultGateProtocolError(f"{label} formal runtime was not deterministic")
    overrides = runtime.get("cfg_overrides")
    if not isinstance(overrides, dict) or set(overrides) - {"work_dir"}:
        raise ResultGateInputError(f"{label} launch ticket cfg overrides differ")
    checkpoint = runtime.get("resume_checkpoint")
    if entrypoint == "train":
        if checkpoint is not None:
            raise ResultGateProtocolError(f"{label} formal training unexpectedly resumed")
    else:
        expected_checkpoint = {
            "path": paths["checkpoint"]
            .resolve()
            .relative_to(Path(path).resolve().parent)
            .as_posix(),
            "sha256": hashes["checkpoint"],
            "size_bytes": len(
                _cached_bytes(paths["checkpoint"], f"{label}.checkpoint")
            ),
        }
        if checkpoint != expected_checkpoint:
            raise ResultGateInputError(
                f"{label} evaluation ticket does not bind the evaluated checkpoint"
            )
    if ticket["profile_evidence"] is None:
        raise ResultGateInputError(f"{label} formal ticket lacks fixed-step profile evidence")
    prerequisite_hashes = {}
    prerequisite_paths = {}
    for role in ("b0", "review", "profile"):
        reference = ticket[f"{role}_evidence"]
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ResultGateInputError(f"{label} launch ticket {role} reference differs")
        prerequisite_paths[role], prerequisite_hashes[role] = _bundle_reference_file(
            reference,
            Path(path).parent,
            f"{label}.launch_ticket.{role}",
        )
    resolved_cfg = _resolved_run_config(paths["resolved_config"], label)
    if resolved_cfg.get("crs_eps_contract") is not None:
        if ticket["g0_evidence"] is None:
            raise ResultGateInputError(f"{label} CRS-EPS launch ticket lacks G0 evidence")
        try:
            _validate_g0_artifact(
                ticket["g0_evidence"],
                Path(path).parent,
                commit_sha=manifest["commit_sha"],
                source_tree_sha256=ticket["source_tree_sha256"],
                config_file_sha256=ticket["config_file_sha256"],
                resolved_config_sha256=ticket["resolved_config_sha256"],
                scientific_config_sha256=ticket["scientific_config_sha256"],
                data_identity_sha256=ticket["data_identity"]["identity_sha256"],
                trust_root=_attestation_trust_roots()["g0"],
            )
        except FullPetalLaunchError as exc:
            raise ResultGateInputError(f"{label} G0 evidence is invalid: {exc}") from exc
    elif ticket["g0_evidence"] is not None:
        raise ResultGateInputError(f"{label} non-CRS ticket contains G0 evidence")
    _validate_data_identity_artifact(paths["data_identity"], ticket["data_identity"], label)
    _validate_signed_review(
        prerequisite_paths["review"],
        commit_sha=manifest["commit_sha"],
        b0_sha256=prerequisite_hashes["b0"],
        label=label,
    )
    _validate_signed_profile(
        prerequisite_paths["profile"],
        ticket=ticket,
        manifest=manifest,
        config_path=paths["resolved_config"],
        label=label,
    )

    signed_receipt = _load_evidence_json(receipt_path, f"{label}.launch_receipt")
    try:
        receipt = verify_launch_receipt(
            signed_receipt,
            trust_root=_attestation_trust_roots()["profile"],
        )
    except FullPetalLaunchError as exc:
        raise ResultGateInputError(
            f"{label} launch receipt attestation is invalid: {exc}"
        ) from exc
    receipt_fields = {
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
        "g0_artifact_sha256",
        "profile_artifact_sha256",
        "world_size",
        "slurm_job_id",
        "slurm_allocation",
        "execution_session",
    }
    if set(receipt) != receipt_fields or receipt["schema_version"] != LAUNCH_RECEIPT_SCHEMA:
        raise ResultGateInputError(f"{label} launch receipt schema/fields differ")
    expected_receipt_values = {
        "mode": "formal",
        "commit_sha": manifest["commit_sha"],
        "source_tree_sha256": ticket["source_tree_sha256"],
        "data_identity_sha256": ticket["data_identity"]["identity_sha256"],
        "runtime_identity_sha256": _sha256_json(runtime, f"{label}.runtime_identity"),
        "resolved_config_sha256": ticket["resolved_config_sha256"],
        "scientific_config_sha256": ticket["scientific_config_sha256"],
        "launch_ticket": {
            "path": Path(path).name,
            "sha256": hashes[f"{ticket_role}_launch_ticket"],
        },
        "b0_artifact_sha256": prerequisite_hashes["b0"],
        "review_artifact_sha256": prerequisite_hashes["review"],
        "g0_artifact_sha256": (
            None
            if ticket["g0_evidence"] is None
            else ticket["g0_evidence"]["sha256"]
        ),
        "profile_artifact_sha256": prerequisite_hashes["profile"],
    }
    for field, expected in expected_receipt_values.items():
        if receipt[field] != expected:
            raise ResultGateInputError(f"{label} launch receipt {field} differs")
    world_size = _nonnegative_integer(
        receipt["world_size"], f"{label}.launch_receipt.world_size"
    )
    if world_size != 1:
        raise ResultGateProtocolError(f"{label} launch receipt violates locked world_size=1")
    allocation = receipt["slurm_allocation"]
    allocation_fields = {
        "job_id",
        "state",
        "user",
        "nodes",
        "tasks",
        "gpus",
        "command",
        "work_dir",
    }
    if not isinstance(allocation, dict) or set(allocation) != allocation_fields:
        raise ResultGateInputError(f"{label} Slurm allocation fields differ")
    if (
        not isinstance(receipt["slurm_job_id"], str)
        or not receipt["slurm_job_id"].strip()
        or allocation["job_id"] != receipt["slurm_job_id"]
        or allocation["state"] != "RUNNING"
    ):
        raise ResultGateProtocolError(f"{label} launch receipt lacks an active Slurm job")
    if (
        allocation["nodes"] != 1
        or allocation["tasks"] != world_size
        or allocation["gpus"] != world_size
    ):
        raise ResultGateProtocolError(f"{label} Slurm allocation geometry differs")
    for field in ("user", "command", "work_dir"):
        if not isinstance(allocation[field], str) or not allocation[field].strip():
            raise ResultGateInputError(f"{label} Slurm allocation lacks {field}")
    try:
        runtime_profile_trust_root(receipt["execution_session"])
    except RuntimeAttestationError as exc:
        raise ResultGateInputError(
            f"{label} launch receipt runtime session is invalid: {exc}"
        ) from exc
    return ticket, receipt


def _evaluate_signed_run(paths, claim, label):
    spec = _load_evidence_json(paths["evaluator_spec"], f"{label}.evaluator_spec")
    required = {
        "schema_version",
        "type",
        "subset",
        "tiou_thresholds",
        "latency_budgets_sec",
        "fps",
        "identity_tiou_threshold",
        "identity_latency_budget_sec",
    }
    if set(spec) != required or spec["schema_version"] != EVALUATOR_SPEC_SCHEMA:
        raise ResultGateInputError(f"{label} evaluator specification fields differ")
    if spec["type"] != "OnlineAPBudgeted":
        raise ResultGateInputError(f"{label} evaluator type is unsupported")
    try:
        ground_truth = strict_json_from_bytes(
            _cached_bytes(paths["ground_truth"], f"{label}.ground_truth"),
            f"{label}.ground_truth",
            require_object=True,
        )
        allowed_video_bytes = _cached_bytes(
            paths["allowed_videos"], f"{label}.allowed_videos"
        )
        if paths["allowed_videos"].suffix.lower() == ".json":
            allowed_videos = strict_json_from_bytes(
                allowed_video_bytes, f"{label}.allowed_videos"
            )
        else:
            allowed_videos = [
                line.strip()
                for line in allowed_video_bytes.decode("utf-8").splitlines()
                if line.strip()
            ]
        if not isinstance(allowed_videos, list):
            raise ResultGateInputError(
                f"{label}.allowed_videos must be a JSON list"
            )
        evaluator = OnlineAPBudgeted(
            ground_truth_filename=ground_truth,
            prediction_filename={
                "ledger_bytes": _cached_bytes(paths["ledger"], f"{label}.ledger"),
                "commitment_bytes": _cached_bytes(
                    paths["commitment"], f"{label}.commitment"
                ),
                "ledger_filename": paths["ledger"].name,
            },
            allowed_videos=allowed_videos,
            subset=spec["subset"],
            tiou_thresholds=spec["tiou_thresholds"],
            latency_budgets_sec=spec["latency_budgets_sec"],
            fps=spec["fps"],
            require_ledger=True,
            require_no_future=True,
            include_identity_diagnostics=True,
            identity_tiou_threshold=spec["identity_tiou_threshold"],
            identity_latency_budget_sec=spec["identity_latency_budget_sec"],
        )
        computed = evaluator.evaluate()
    except (EvidenceBundleError, OSError, TypeError, ValueError) as exc:
        raise ResultGateProtocolError(f"{label} evaluator replay failed: {exc}") from exc
    return _normalized_metrics({"metrics": computed}, claim, f"{label}.recomputed_metrics")


def _resolved_run_config(path, label):
    try:
        return Config(
            strict_json_from_bytes(
                _cached_bytes(path, f"{label}.resolved_config"),
                f"{label}.resolved_config",
                require_object=True,
            )
        )
    except (EvidenceBundleError, TypeError, ValueError) as exc:
        raise ResultGateInputError(
            f"{label} resolved config cannot be loaded: {exc}"
        ) from exc


def _validate_run_evidence(row, claim, variant, seed, label):
    evidence = row.get("evidence")
    expected_evidence = {"run_manifest_path", "run_manifest_sha256"}
    if not isinstance(evidence, dict) or set(evidence) != expected_evidence:
        raise ResultGateInputError(
            f"{label}.evidence requires exactly a signed run manifest path and hash"
        )
    manifest_path, manifest_hash = _evidence_file(evidence, "run_manifest", label)
    signed_manifest = _load_evidence_json(manifest_path, f"{label}.run_manifest")
    verified_artifacts = {}
    try:
        manifest = verify_formal_run_manifest(
            signed_manifest,
            trust_root=_attestation_trust_roots()["formal"],
            base_dir=manifest_path.parent,
            verified_artifacts=verified_artifacts,
        )
    except TrainingEvidenceError as exc:
        raise ResultGateInputError(f"{label} formal run manifest is invalid: {exc}") from exc
    if (
        _claim_name(manifest["claim"]) != claim
        or _variant_name(claim, manifest["variant"]) != variant
        or _seed(manifest["seed"], f"{label}.run_manifest.seed") != seed
    ):
        raise ResultGateInputError(f"{label} formal run identity differs")
    _required_git_sha(manifest["commit_sha"], f"{label}.run_manifest.commit_sha")
    if manifest["protocol_sha256"] != _sha256_json(row["protocol"], f"{label}.protocol"):
        raise ResultGateInputError(f"{label} formal run protocol hash differs")

    expected_artifacts = set(COMMON_RUN_ARTIFACTS)
    if claim == "C2":
        expected_artifacts.update(
            {"visual_parameter_trace", "visual_parameter_commitment"}
        )
    if set(manifest["artifacts"]) != expected_artifacts:
        raise ResultGateInputError(
            f"{label} formal run artifact roles differ; "
            f"missing={sorted(expected_artifacts - set(manifest['artifacts']))}, "
            f"extra={sorted(set(manifest['artifacts']) - expected_artifacts)}"
        )
    paths = {}
    artifact_bytes = {}
    try:
        for role, (path, payload) in verified_artifacts.items():
            paths[role] = path
            artifact_bytes[role] = payload
            _cache_verified_bytes(path, payload)
    except EvidenceBundleError as exc:
        raise ResultGateInputError(str(exc)) from exc
    checkpoint_bytes = artifact_bytes["checkpoint"]
    hashes = {
        role: reference["sha256"] for role, reference in manifest["artifacts"].items()
    }
    training_ticket, training_receipt = _validate_launch_ticket(
        paths["training_launch_ticket"],
        paths["training_launch_receipt"],
        manifest,
        paths,
        hashes,
        seed,
        f"{label}.training",
        entrypoint="train",
    )
    evaluation_ticket, _ = _validate_launch_ticket(
        paths["evaluation_launch_ticket"],
        paths["evaluation_launch_receipt"],
        manifest,
        paths,
        hashes,
        seed,
        f"{label}.evaluation",
        entrypoint="test",
    )
    shared_ticket_fields = {
        "commit_sha",
        "source_tree_sha256",
        "config_file_sha256",
        "resolved_config_sha256",
        "scientific_config_sha256",
        "data_identity",
        "b0_evidence",
        "review_evidence",
        "g0_evidence",
        "profile_evidence",
    }
    for field in shared_ticket_fields:
        if training_ticket[field] != evaluation_ticket[field]:
            raise ResultGateInputError(
                f"{label} training/evaluation launch tickets disagree on {field}"
            )
    cfg = _resolved_run_config(paths["resolved_config"], label)
    if training_ticket["resolved_config_sha256"] != resolved_config_sha256(cfg):
        raise ResultGateInputError(
            f"{label} launch ticket resolved config hash differs from the config artifact"
        )
    if training_ticket["scientific_config_sha256"] != resolved_config_sha256(
        cfg, scientific=True
    ):
        raise ResultGateInputError(
            f"{label} launch ticket scientific config hash differs from the config artifact"
        )
    evaluator_spec = _load_evidence_json(
        paths["evaluator_spec"], f"{label}.evaluator_spec"
    )
    try:
        validate_evaluation_artifact_bindings(
            cfg,
            training_ticket["data_identity"],
            ground_truth_path=paths["ground_truth"],
            allowed_videos_path=paths["allowed_videos"],
            evaluator_spec=evaluator_spec,
            ground_truth_bytes=artifact_bytes["ground_truth"],
            allowed_videos_bytes=artifact_bytes["allowed_videos"],
        )
        expected_training_identity = derive_training_trace_identity(
            cfg,
            training_ticket["data_identity"],
            seed=seed,
            world_size=1,
            annotation_bytes=artifact_bytes["ground_truth"],
            allow_list_bytes=artifact_bytes["fit_core"],
            cache_manifest_bytes=artifact_bytes["feature_cache_manifest"],
        )
    except IdentityError as exc:
        raise ResultGateInputError(
            f"{label} bound config/data identity is invalid: {exc}"
        ) from exc
    metrics = _evaluate_signed_run(paths, claim, label)
    try:
        cost = derive_training_cost(
            paths["training_trace"],
            paths["training_commitment"],
            trace_bytes=artifact_bytes["training_trace"],
            commitment_bytes=artifact_bytes["training_commitment"],
            runtime_binding=training_receipt["execution_session"],
            require_contiguous_runtime=claim == "C1",
        )
        if cost["skipped_optimizer_events"] != 0:
            raise TrainingEvidenceError(
                "formal training trace contains an uncommitted optimizer boundary"
            )
    except TrainingEvidenceError as exc:
        raise ResultGateInputError(f"{label} training trace is invalid: {exc}") from exc
    actual_training_identity = {
        field: cost[field] for field in expected_training_identity
    }
    if actual_training_identity != expected_training_identity:
        raise ResultGateInputError(
            f"{label} training trace identity differs from the bound config/data order"
        )
    normalized_cost = _normalized_cost({"cost": cost}, label) if claim == "C1" else None

    visual_parameter_evidence = None
    if claim == "C2":
        try:
            visual_parameter_evidence = derive_visual_parameter_evidence(
                cfg,
                variant=variant,
                checkpoint_path=paths["checkpoint"],
                training_trace_path=paths["training_trace"],
                training_commitment_path=paths["training_commitment"],
                visual_trace_path=paths["visual_parameter_trace"],
                visual_commitment_path=paths["visual_parameter_commitment"],
                checkpoint_bytes=checkpoint_bytes,
                training_trace_bytes=artifact_bytes["training_trace"],
                training_commitment_bytes=artifact_bytes["training_commitment"],
                visual_trace_bytes=artifact_bytes["visual_parameter_trace"],
                visual_commitment_bytes=artifact_bytes[
                    "visual_parameter_commitment"
                ],
                runtime_binding=training_receipt["execution_session"],
            )
        except TrainingEvidenceError as exc:
            raise ResultGateInputError(
                f"{label} visual parameter evidence is invalid: {exc}"
            ) from exc
    return {
        "manifest_sha256": manifest_hash,
        "commit_sha": manifest["commit_sha"],
        "b0_evidence": training_ticket["b0_evidence"],
        "metrics": metrics,
        "cost": normalized_cost,
        "visual_parameter_evidence": visual_parameter_evidence,
    }


def _collect_payloads(value):
    payloads = [value] if isinstance(value, dict) else list(value) if isinstance(value, (list, tuple)) else None
    if payloads is None or not payloads:
        raise ResultGateInputError("result artifacts must be a non-empty object or sequence")
    runs = []
    containers = []
    for index, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            raise ResultGateInputError(f"artifact[{index}] must be a JSON object")
        if "claim" in payload:
            runs.append(payload)
            continue
        unknown = set(payload) - {"runs", "b0_evidence", "reporting_evidence"}
        if unknown:
            raise ResultGateInputError(
                f"artifact[{index}] contains unsupported self-attested fields: {sorted(unknown)}"
            )
        containers.append(payload)
        artifact_runs = payload.get("runs", ())
        if not isinstance(artifact_runs, (list, tuple)):
            raise ResultGateInputError(f"artifact[{index}].runs must be a sequence")
        runs.extend(artifact_runs)
    return containers, runs


def _b0_evidence_entry(containers):
    forbidden = {"prerequisites", "protocol_b0", "protocol_passed", "b0_passed"}
    entries = []
    for index, container in enumerate(containers):
        legacy = sorted(forbidden.intersection(container))
        if legacy:
            raise ResultGateInputError(
                "protocol/B0 status cannot be self-attested; "
                f"remove {legacy} and provide b0_evidence"
            )
        if "b0_evidence" in container:
            entries.append((index, container["b0_evidence"]))
    if len(entries) > 1:
        canonical = {
            _canonical_json(value, f"artifact[{index}].b0_evidence")
            for index, value in entries
        }
        if len(canonical) > 1:
            raise ResultGateInputError("conflicting B0 evidence artifacts")
    return entries[0][1] if entries else None


def _validate_b0_evidence(containers, *, expected_commit, ticket_b0_hashes):
    evidence = _b0_evidence_entry(containers)
    if evidence is None:
        return {"provided": False, "passed": False, "violations": []}
    try:
        artifact = validate_b0_evidence(
            evidence,
            base_dir=REPO_ROOT,
            expected_commit=expected_commit,
            trust_root=_attestation_trust_roots()["b0"],
        )
    except B0EvidenceError as exc:
        raise ResultGateInputError(f"B0 evidence is invalid: {exc}") from exc
    b0_hash = artifact["artifact_sha256"]
    if ticket_b0_hashes != {b0_hash}:
        raise ResultGateInputError(
            "formal launch tickets do not all bind the supplied B0 artifact"
        )
    return {
        "provided": True,
        "passed": True,
        "violations": [],
        "artifact_sha256": b0_hash,
        "commit_sha": artifact["commit_sha"],
        "test_count": artifact["test_count"],
    }


def _single_container_entry(containers, name):
    entries = [(index, container[name]) for index, container in enumerate(containers) if name in container]
    if not entries:
        return None
    canonical = {_canonical_json(value, f"artifact[{index}].{name}") for index, value in entries}
    if len(canonical) != 1:
        raise ResultGateInputError(f"conflicting {name} artifacts")
    return entries[0][1]


def _load_content_manifest(reference, label):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ResultGateInputError(f"{label} reference requires exactly path and sha256")
    wrapper = {f"{label}_path": reference["path"], f"{label}_sha256": reference["sha256"]}
    path, digest = _evidence_file(wrapper, label, label)
    manifest = _load_evidence_json(path, label)
    try:
        valid = verify_content_hash(manifest)
    except ContractValidationError as exc:
        raise ResultGateInputError(f"{label} content hash validation failed: {exc}") from exc
    if not valid:
        raise ResultGateInputError(f"{label} content hash is invalid")
    return manifest, path, digest


def _load_source_reference(reference, label):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ResultGateInputError(f"{label} reference requires exactly path and sha256")
    wrapper = {f"{label}_path": reference["path"], f"{label}_sha256": reference["sha256"]}
    return _evidence_file(wrapper, label, label)


def _validate_reporting_evidence(containers, *, require_fineaction):
    evidence = _single_container_entry(containers, "reporting_evidence")
    if evidence is None:
        return {
            "reporting_211_213": {"provided": False, "passed": False},
            "FineAction": {"provided": False, "passed": not require_fineaction},
        }
    required_fields = {
        "universe",
        "comparison",
        "locked_ids",
        "observed_ids",
        "difference_reasons",
        "fineaction",
        "fineaction_sources",
    }
    if not isinstance(evidence, dict) or set(evidence) != required_fields:
        raise ResultGateInputError(
            "reporting_evidence requires reporting manifests, their raw sources, "
            "and FineAction source references"
        )
    universe, _, universe_hash = _load_content_manifest(
        evidence["universe"], "reporting_universe"
    )
    comparison, _, comparison_hash = _load_content_manifest(
        evidence["comparison"], "reporting_comparison"
    )
    locked_ids_path, _ = _load_source_reference(
        evidence["locked_ids"], "reporting_locked_ids"
    )
    observed_ids_path, _ = _load_source_reference(
        evidence["observed_ids"], "reporting_observed_ids"
    )
    difference_reasons_path, _ = _load_source_reference(
        evidence["difference_reasons"], "reporting_difference_reasons"
    )
    try:
        validate_reporting_artifacts_from_sources(
            universe,
            comparison,
            locked_ids_path=locked_ids_path,
            observed_ids_path=observed_ids_path,
            difference_reasons_path=difference_reasons_path,
        )
    except ContractValidationError as exc:
        raise ResultGateInputError(
            f"reporting evidence is not source-derived: {exc}"
        ) from exc
    if (
        universe.get("schema") != "full_petal.historical_reporting_universe"
        or universe.get("universe", {}).get("count") != 211
    ):
        raise ResultGateInputError("historical reporting universe is not the locked 211-ID artifact")
    if (
        comparison.get("schema") != "full_petal.reporting_universe_comparison"
        or comparison.get("strict") is not True
        or comparison.get("observed_universe", {}).get("count") != 213
        or comparison.get("historical_universe", {}).get("manifest_content_sha256")
        != universe.get("content_sha256")
        or comparison.get("comparison", {}).get("unexplained_ids")
        or comparison.get("status") not in {"MATCH", "EXPLAINED_MISMATCH"}
    ):
        raise ResultGateInputError("211-vs-213 reporting comparison is not fully explained")

    fineaction_result = {"provided": False, "passed": not require_fineaction}
    if evidence["fineaction"] is not None:
        source_references = evidence["fineaction_sources"]
        if not isinstance(source_references, dict) or set(source_references) != set(
            FINEACTION_SOURCE_KEYS
        ):
            raise ResultGateInputError(
                "FineAction evidence requires the exact source reference set"
            )
        fineaction, _, fineaction_hash = _load_content_manifest(
            evidence["fineaction"], "fineaction_qualification"
        )
        if (
            fineaction.get("schema") != "full_petal.fineaction_qualification"
            or fineaction.get("qualified") is not True
            or fineaction.get("status") != "PASS"
        ):
            raise ResultGateInputError("FineAction qualification has not reached PASS")
        source_paths = {}
        source_bytes = {}
        for name in FINEACTION_SOURCE_KEYS:
            source_path, _ = _load_source_reference(
                source_references[name], f"fineaction_{name}"
            )
            source_paths[name] = source_path
            source_bytes[name] = _cached_bytes(
                source_path, f"fineaction_{name}"
            )
        try:
            validate_fineaction_qualification_report(
                fineaction,
                source_paths,
                trust_roots=_route_launch_requirements()[
                    "fineaction_trust_roots"
                ],
                source_bytes=source_bytes,
            )
        except ContractValidationError as exc:
            raise ResultGateInputError(
                f"FineAction qualification is not source-derived: {exc}"
            ) from exc
        fineaction_result = {
            "provided": True,
            "passed": True,
            "artifact_sha256": fineaction_hash,
        }
    elif evidence["fineaction_sources"] is not None:
        raise ResultGateInputError(
            "FineAction sources cannot be supplied without a qualification report"
        )
    return {
        "reporting_211_213": {
            "provided": True,
            "passed": True,
            "universe_sha256": universe_hash,
            "comparison_sha256": comparison_hash,
        },
        "FineAction": fineaction_result,
    }


def _run_evidence_rows(grouped):
    return [
        row["evidence"]
        for claim in grouped.values()
        for variant in claim.values()
        for row in variant.values()
    ]


def _normalize_prerequisites(containers, grouped, run_count):
    evidence_rows = _run_evidence_rows(grouped)
    if not evidence_rows:
        reporting = _validate_reporting_evidence(containers, require_fineaction=False)
        return {
            "protocol": {
                "provided": False,
                "passed": False,
                "violations": [],
                "source": "verified_run_evidence",
            },
            "B0": {"provided": False, "passed": False, "violations": []},
            **reporting,
        }
    commits = {row["commit_sha"] for row in evidence_rows}
    if len(commits) != 1:
        raise ResultGateInputError("all formal runs must bind one exact source commit")
    expected_commit = _required_git_sha(next(iter(commits)), "formal run commit")
    ticket_b0_hashes = {
        row["b0_evidence"].get("sha256")
        for row in evidence_rows
        if isinstance(row.get("b0_evidence"), dict)
    }
    b0 = _validate_b0_evidence(
        containers,
        expected_commit=expected_commit,
        ticket_b0_hashes=ticket_b0_hashes,
    )
    require_fineaction = any(grouped["C2"][name] for name in grouped["C2"])
    reporting = _validate_reporting_evidence(
        containers, require_fineaction=require_fineaction
    )
    return {
        "protocol": {
            "provided": run_count > 0,
            "passed": run_count > 0,
            "violations": [],
            "source": "verified_run_evidence",
        },
        "B0": b0,
        **reporting,
    }


def _normalize_runs(raw_runs):
    grouped = {
        "C1": {"fixed": {}, "rematch": {}},
        "C2": {"frozen": {}, "adapted": {}},
    }
    for index, row in enumerate(raw_runs):
        if not isinstance(row, dict):
            raise ResultGateInputError(f"runs[{index}] must be a JSON object")
        required_fields = {"claim", "variant", "seed", "protocol", "evidence"}
        if set(row) != required_fields:
            raise ResultGateInputError(
                f"runs[{index}] fields differ; expected {sorted(required_fields)}, "
                f"found {sorted(row)}. Metrics, costs, and PASS flags must be recomputed."
            )
        claim = _claim_name(row["claim"])
        variant = _variant_name(claim, row["variant"])
        seed = _seed(row["seed"], f"runs[{index}].seed")
        if seed in grouped[claim][variant]:
            raise ResultGateInputError(f"duplicate {claim}/{variant} run for seed {seed}")
        label = f"runs[{index}]({claim}/{variant}/seed={seed})"
        evidence = _validate_run_evidence(row, claim, variant, seed, label)
        protocol = _validate_row_protocol(row, claim, label)
        normalized = {
            "seed": seed,
            "protocol": protocol,
            "metrics": evidence["metrics"],
            "evidence": evidence,
        }
        if claim == "C1":
            normalized["cost"] = evidence["cost"]
        grouped[claim][variant][seed] = normalized
    return grouped


def _paired_claim(grouped, claim, variants):
    left_name, right_name = variants
    left = grouped[claim][left_name]
    right = grouped[claim][right_name]
    if not left and not right:
        return None
    left_seeds = set(left)
    right_seeds = set(right)
    if not left_seeds or not right_seeds or left_seeds != right_seeds:
        raise ResultGateInputError(
            f"{claim} paired seed coverage requires identical non-empty seed sets: "
            f"{left_name}={sorted(left_seeds)}, {right_name}={sorted(right_seeds)}"
        )
    seeds = sorted(left_seeds)
    protocol_values = {
        _canonical_json(row["protocol"], f"{claim}.protocol")
        for name in variants
        for row in grouped[claim][name].values()
    }
    if len(protocol_values) != 1:
        raise ResultGateProtocolError(f"{claim} protocol mismatch across paired runs")
    return seeds


def _relative_reduction(candidate, baseline):
    if baseline > 0:
        return (baseline - candidate) / baseline
    return 0.0 if candidate == 0 else -1.0


def _cost_parity(grouped, seeds, tolerance):
    comparisons = []
    passed = True
    for seed in seeds:
        fixed = grouped["C1"]["fixed"][seed]["cost"]
        rematch = grouped["C1"]["rematch"][seed]["cost"]
        if set(fixed) != set(rematch):
            raise ResultGateInputError(
                f"C1 cost fields differ for seed {seed}: "
                f"fixed={sorted(fixed)}, rematch={sorted(rematch)}"
            )
        metrics = {}
        for key in sorted(fixed):
            if key in C1_COST_IDENTITY_FIELDS:
                relative_gap = 0.0 if fixed[key] == rematch[key] else None
                metric_pass = fixed[key] == rematch[key]
            else:
                scale = max(abs(fixed[key]), abs(rematch[key]), 1e-12)
                relative_gap = abs(fixed[key] - rematch[key]) / scale
                metric_pass = relative_gap <= tolerance + 1e-12
            passed = passed and metric_pass
            metrics[key] = {
                "fixed": fixed[key],
                "rematch": rematch[key],
                "relative_gap": relative_gap,
                "passed": metric_pass,
            }
        comparisons.append({"seed": seed, "metrics": metrics})
    return {
        "passed": passed,
        "relative_tolerance": tolerance,
        "per_seed": comparisons,
    }


def _c1_scientific_metrics(
    grouped,
    seeds,
    map_gain_points,
    error_reduction,
    map_parity_tolerance_points,
    recall_tolerance_points,
    false_emission_tolerance,
    latency_tolerance_frames,
):
    per_seed = []
    for seed in seeds:
        fixed = grouped["C1"]["fixed"][seed]["metrics"]
        rematch = grouped["C1"]["rematch"][seed]["metrics"]
        per_seed.append(
            {
                "seed": seed,
                "online_map_gain_points": 100.0
                * (fixed["average_mOnlineAP"] - rematch["average_mOnlineAP"]),
                "standard_map_delta_points": 100.0
                * (fixed["average_mAP"] - rematch["average_mAP"]),
                "recall_delta_points": 100.0
                * (fixed["identity_recall"] - rematch["identity_recall"]),
                "false_emission_rate_delta": (
                    fixed["false_emission_rate"] - rematch["false_emission_rate"]
                ),
                "endpoint_latency_frames_delta": (
                    fixed["endpoint_latency_frames_mean"]
                    - rematch["endpoint_latency_frames_mean"]
                ),
                "error_reduction": {
                    name: _relative_reduction(fixed[name], rematch[name])
                    for name in ("duplicate_per_gt", "fragmentation_rate")
                },
            }
        )

    aggregate = {
        "online_map_gain_points": _mean(
            row["online_map_gain_points"] for row in per_seed
        ),
        "standard_map_delta_points": _mean(
            row["standard_map_delta_points"] for row in per_seed
        ),
        "recall_delta_points": _mean(row["recall_delta_points"] for row in per_seed),
        "false_emission_rate_delta": _mean(
            row["false_emission_rate_delta"] for row in per_seed
        ),
        "endpoint_latency_frames_delta": _mean(
            row["endpoint_latency_frames_delta"] for row in per_seed
        ),
        "error_reduction": {
            name: _mean(row["error_reduction"][name] for row in per_seed)
            for name in ("duplicate_per_gt", "fragmentation_rate")
        },
    }
    best_error_reduction = max(aggregate["error_reduction"].values())
    identity_noninferior = all(
        reduction >= -1e-12
        for reduction in aggregate["error_reduction"].values()
    )
    map_path = aggregate["online_map_gain_points"] >= map_gain_points - 1e-12
    error_path = (
        best_error_reduction >= error_reduction - 1e-12
        and aggregate["online_map_gain_points"]
        >= -map_parity_tolerance_points - 1e-12
    )
    standard_map_pass = (
        aggregate["standard_map_delta_points"]
        >= -map_parity_tolerance_points - 1e-12
    )
    recall_pass = aggregate["recall_delta_points"] >= -recall_tolerance_points - 1e-12
    false_emission_pass = (
        aggregate["false_emission_rate_delta"] <= false_emission_tolerance + 1e-12
    )
    latency_pass = (
        aggregate["endpoint_latency_frames_delta"] <= latency_tolerance_frames + 1e-12
    )
    passed = (
        (map_path or error_path)
        and standard_map_pass
        and recall_pass
        and false_emission_pass
        and latency_pass
        and identity_noninferior
    )
    return {
        "passed": passed,
        "effect_paths": {
            "map_gain": map_path,
            "duplicate_or_fragmentation_reduction_at_map_parity": error_path,
        },
        "safety_checks": {
            "standard_map_noninferior": standard_map_pass,
            "recall_noninferior": recall_pass,
            "false_emission_noninferior": false_emission_pass,
            "endpoint_latency_noninferior": latency_pass,
            "duplicate_and_fragmentation_noninferior": identity_noninferior,
        },
        "aggregate": aggregate,
        "per_seed": per_seed,
        "thresholds": {
            "map_gain_points": map_gain_points,
            "error_reduction_fraction": error_reduction,
            "map_parity_tolerance_points": map_parity_tolerance_points,
            "recall_tolerance_points": recall_tolerance_points,
            "false_emission_rate_tolerance": false_emission_tolerance,
            "endpoint_latency_frames_tolerance": latency_tolerance_frames,
        },
    }


def _not_evaluated(claim, variants):
    return {
        "claim": claim,
        "status": "NOT_EVALUATED",
        "requirements": {
            "paired_seed_coverage": False,
            "protocol_equality": False,
            "scientific_metrics": False,
            **({"cost_parity": False} if claim == "C1" else {}),
            **({"visual_parameter_events": False} if claim == "C2" else {}),
        },
        "paired_seed_coverage": {
            "passed": False,
            "seeds": [],
            "variants": list(variants),
        },
        "reasons": [f"no {claim} runs were supplied"],
    }


def _evaluate_c1(
    grouped,
    cost_tolerance,
    map_gain_points,
    error_reduction,
    map_parity_tolerance_points,
    recall_tolerance_points,
    false_emission_tolerance,
    latency_tolerance_frames,
):
    variants = ("fixed", "rematch")
    seeds = _paired_claim(grouped, "C1", variants)
    if seeds is None:
        return _not_evaluated("C1", variants)
    cost = _cost_parity(grouped, seeds, cost_tolerance)
    scientific = _c1_scientific_metrics(
        grouped,
        seeds,
        map_gain_points,
        error_reduction,
        map_parity_tolerance_points,
        recall_tolerance_points,
        false_emission_tolerance,
        latency_tolerance_frames,
    )
    passed = cost["passed"] and scientific["passed"]
    reasons = []
    if not scientific["passed"]:
        reasons.append("C1 scientific effect or non-inferiority checks did not pass")
    if not cost["passed"]:
        reasons.append("C1 fixed/rematch cost parity did not pass")
    requirements = {
        "paired_seed_coverage": True,
        "protocol_equality": True,
        "scientific_metrics": scientific["passed"],
        "cost_parity": cost["passed"],
    }
    return {
        "claim": "C1",
        "status": "PASS" if passed else "FAIL",
        "requirements": requirements,
        "paired_seed_coverage": {
            "passed": True,
            "seeds": seeds,
            "variants": list(variants),
        },
        "protocol_equality": {"passed": True},
        "scientific_metrics": scientific,
        "cost_parity": cost,
        "reasons": reasons,
    }


def _evaluate_c2(grouped, map_gain_points, specialized_gain_points, map_parity_tolerance_points):
    variants = ("adapted", "frozen")
    seeds = _paired_claim(grouped, "C2", variants)
    if seeds is None:
        return _not_evaluated("C2", variants)

    per_seed = []
    specialized_names = set()
    for seed in seeds:
        adapted = grouped["C2"]["adapted"][seed]["metrics"]
        frozen = grouped["C2"]["frozen"][seed]["metrics"]
        row = {
            "seed": seed,
            "map_gain_points": 100.0 * (adapted["average_mAP"] - frozen["average_mAP"]),
            "specialized_gain_points": {},
        }
        for name in ("high_tiou_mAP", "short_action_mAP"):
            if (name in adapted) != (name in frozen):
                raise ResultGateInputError(
                    f"C2 paired runs must both report {name} for seed {seed} or neither"
                )
            if name in adapted:
                specialized_names.add(name)
                row["specialized_gain_points"][name] = 100.0 * (
                    adapted[name] - frozen[name]
                )
        per_seed.append(row)

    aggregate_map_gain = _mean(row["map_gain_points"] for row in per_seed)
    aggregate_specialized = {
        name: _mean(row["specialized_gain_points"][name] for row in per_seed)
        for name in sorted(specialized_names)
    }
    map_path = aggregate_map_gain >= map_gain_points - 1e-12
    specialized_path = bool(aggregate_specialized) and (
        max(aggregate_specialized.values()) >= specialized_gain_points - 1e-12
        and aggregate_map_gain >= -map_parity_tolerance_points - 1e-12
    )
    scientific_pass = map_path or specialized_path
    scientific = {
        "passed": scientific_pass,
        "effect_paths": {
            "average_map_gain": map_path,
            "preregistered_specialized_gain_at_map_parity": specialized_path,
        },
        "aggregate": {
            "map_gain_points": aggregate_map_gain,
            "specialized_gain_points": aggregate_specialized,
        },
        "per_seed": per_seed,
        "thresholds": {
            "map_gain_points": map_gain_points,
            "specialized_gain_points": specialized_gain_points,
            "map_parity_tolerance_points": map_parity_tolerance_points,
        },
    }
    return {
        "claim": "C2",
        "status": "PASS" if scientific_pass else "FAIL",
        "requirements": {
            "paired_seed_coverage": True,
            "protocol_equality": True,
            "scientific_metrics": scientific_pass,
            "visual_parameter_events": True,
        },
        "paired_seed_coverage": {
            "passed": True,
            "seeds": seeds,
            "variants": list(variants),
        },
        "protocol_equality": {"passed": True},
        "visual_parameter_events": {
            "passed": True,
            "per_seed": [
                {
                    "seed": seed,
                    "adapted": grouped["C2"]["adapted"][seed]["evidence"][
                        "visual_parameter_evidence"
                    ],
                    "frozen": grouped["C2"]["frozen"][seed]["evidence"][
                        "visual_parameter_evidence"
                    ],
                }
                for seed in seeds
            ],
        },
        "scientific_metrics": scientific,
        "reasons": [] if scientific_pass else ["C2 raw-video scientific checks did not pass"],
    }


def evaluate_result_gates(
    artifacts,
    c1_map_gain_points=2.0,
    c1_error_reduction=0.20,
    c1_map_parity_tolerance_points=0.5,
    c1_recall_tolerance_points=1.0,
    c1_false_emission_tolerance=0.01,
    c1_latency_tolerance_frames=0.0,
    c1_cost_relative_tolerance=0.05,
    c2_map_gain_points=2.0,
    c2_specialized_gain_points=2.0,
    c2_map_parity_tolerance_points=0.5,
):
    """Return independent claim decisions; never collapse them into one pass flag."""

    _VERIFIED_BYTES.clear()
    thresholds = {
        "c1_map_gain_points": c1_map_gain_points,
        "c1_error_reduction": c1_error_reduction,
        "c1_map_parity_tolerance_points": c1_map_parity_tolerance_points,
        "c1_recall_tolerance_points": c1_recall_tolerance_points,
        "c1_false_emission_tolerance": c1_false_emission_tolerance,
        "c1_latency_tolerance_frames": c1_latency_tolerance_frames,
        "c1_cost_relative_tolerance": c1_cost_relative_tolerance,
        "c2_map_gain_points": c2_map_gain_points,
        "c2_specialized_gain_points": c2_specialized_gain_points,
        "c2_map_parity_tolerance_points": c2_map_parity_tolerance_points,
    }
    for name, value in thresholds.items():
        thresholds[name] = _finite_number(value, name, minimum=0.0)

    containers, raw_runs = _collect_payloads(artifacts)
    grouped = _normalize_runs(raw_runs)
    prerequisites = _normalize_prerequisites(containers, grouped, len(raw_runs))
    c1 = _evaluate_c1(
        grouped,
        thresholds["c1_cost_relative_tolerance"],
        thresholds["c1_map_gain_points"],
        thresholds["c1_error_reduction"],
        thresholds["c1_map_parity_tolerance_points"],
        thresholds["c1_recall_tolerance_points"],
        thresholds["c1_false_emission_tolerance"],
        thresholds["c1_latency_tolerance_frames"],
    )
    c2 = _evaluate_c2(
        grouped,
        thresholds["c2_map_gain_points"],
        thresholds["c2_specialized_gain_points"],
        thresholds["c2_map_parity_tolerance_points"],
    )
    project_requirements = {
        "C1": c1["status"] == "PASS",
        "C2": c2["status"] == "PASS",
        "protocol": prerequisites["protocol"]["passed"],
        "B0": prerequisites["B0"]["passed"],
        "reporting_211_213": prerequisites["reporting_211_213"]["passed"],
        "FineAction": prerequisites["FineAction"]["passed"],
    }
    screen_pass = all(project_requirements.values())
    project = {
        "status": "NARROW" if screen_pass else "KILL",
        "scientific_status": "INCONCLUSIVE",
        "paper_level_pass": False,
        "scope": "development_screen_not_paper_claim",
        "requirements": project_requirements,
        "reasons": (
            [
                "development screens passed, but B5 evidence requires at least five "
                "seeds, two qualified datasets, paired uncertainty, sample qualification, "
                "and ID-shuffle sensitivity"
            ]
            if screen_pass
            else [
                f"required decision/prerequisite did not pass: {name}"
                for name, passed in project_requirements.items()
                if not passed
            ]
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "decisions": {
            "C1": c1,
            "C2": c2,
            "project_full_system": project,
        },
        "prerequisites": prerequisites,
    }


def _load_json(path):
    try:
        payload = _cached_bytes(path, f"JSON input {path}")
        return strict_json_from_bytes(payload, f"JSON input {path}")
    except EvidenceBundleError as exc:
        raise ResultGateInputError(str(exc)) from exc


def _serialize(payload):
    return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)


def _emit(payload, output=None):
    serialized = _serialize(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="exit nonzero unless the project/full-system decision passes",
    )
    parser.add_argument("--c1-map-gain-points", type=float, default=2.0)
    parser.add_argument("--c1-error-reduction", type=float, default=0.20)
    parser.add_argument("--c1-cost-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--c2-map-gain-points", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        payloads = [_load_json(path) for path in args.artifacts]
        verdict = evaluate_result_gates(
            payloads,
            c1_map_gain_points=args.c1_map_gain_points,
            c1_error_reduction=args.c1_error_reduction,
            c1_cost_relative_tolerance=args.c1_cost_relative_tolerance,
            c2_map_gain_points=args.c2_map_gain_points,
        )
    except ResultGateProtocolError as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "error": {"kind": "protocol_violation", "message": str(exc)},
            },
            args.output,
        )
        return 3
    except ResultGateInputError as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "error": {"kind": "malformed_input", "message": str(exc)},
            },
            args.output,
        )
        return 2

    _emit(verdict, args.output)
    if args.require_pass and verdict["decisions"]["project_full_system"]["status"] != "PASS":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
