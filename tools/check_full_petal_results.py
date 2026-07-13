#!/usr/bin/env python3
"""Evaluate separate Full PETAL C1, C2, and project result gates."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentad.utils.immutable_event_ledger import (
    LedgerError,
    load_verified_ledger,
)


SCHEMA_VERSION = "full_petal_result_gate.v2"

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

SHA256_FIELDS = (
    "config",
    "model",
    "dataset_manifest",
    "evaluator",
    "metrics",
    "run_manifest",
)
RUN_MANIFEST_SCHEMA = "full-petal-run-manifest-v1"
B0_SCHEMA = "full-petal-b0-v1"
B0_TEST_REPORT_SCHEMA = "full-petal-b0-test-report-v1"
B0_AUDIT_REPORT_SCHEMA = "full-petal-b0-audit-report-v1"
RAW_VISUAL_AUDIT_SCHEMA = "full-petal-raw-visual-audit-v1"

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


def _sha256_file(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ResultGateInputError(f"failed to hash evidence file {path}: {exc}") from exc
    return digest.hexdigest()


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
    if not path.is_file():
        raise ResultGateInputError(f"{label}.{path_key} is not a file: {path}")
    expected = _required_sha256(evidence[hash_key], f"{label}.{hash_key}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ResultGateInputError(
            f"{label}.{prefix} evidence hash mismatch: expected {expected}, found {actual}"
        )
    return path, actual


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
    metrics = row.get("metrics")
    violation_values = [row.get("protocol_violations")]
    if isinstance(metrics, dict):
        violation_values.append(metrics.get("protocol_violations"))
        causal = metrics.get("causal_validation")
        if isinstance(causal, dict):
            if causal.get("passed") is False:
                raise ResultGateProtocolError(f"{label} failed causal emission validation")
            violation_values.append(causal.get("violations"))
            counts = causal.get("violation_counts")
            if isinstance(counts, dict) and any(
                _finite_number(value, f"{label}.causal_validation.{key}", minimum=0.0) > 0
                for key, value in counts.items()
            ):
                raise ResultGateProtocolError(f"{label} records causal protocol violations")
    if any(_has_violations(value) for value in violation_values):
        raise ResultGateProtocolError(f"{label} records protocol violations")

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
    ):
        normalized[field] = _nonnegative_integer(
            cost[field],
            f"{label}.cost.{field}",
            positive=field in {"optimizer_events", "input_tokens", "episodes", "effective_batch_size"},
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
    payload = _load_json(Path(path))
    if not isinstance(payload, dict):
        raise ResultGateInputError(f"{label} must be a JSON object")
    return payload


def _validate_raw_visual_audit(path, claim, variant, seed, hashes, label):
    audit = _load_evidence_json(path, f"{label}.raw_visual_audit")
    required = {
        "schema_version",
        "claim",
        "variant",
        "seed",
        "model_sha256",
        "dataset_manifest_sha256",
        "registered_visual_params",
        "nonzero_finite_grad_params",
        "changed_visual_params",
        "frozen_param_delta_max",
        "status",
    }
    if set(audit) != required:
        raise ResultGateInputError(
            f"{label} raw visual audit fields differ: "
            f"expected {sorted(required)}, found {sorted(audit)}"
        )
    if audit["schema_version"] != RAW_VISUAL_AUDIT_SCHEMA:
        raise ResultGateInputError(f"{label} raw visual audit schema is unsupported")
    if (
        _claim_name(audit["claim"]) != claim
        or _variant_name(claim, audit["variant"]) != variant
        or _seed(audit["seed"], f"{label}.raw_visual_audit.seed") != seed
    ):
        raise ResultGateInputError(f"{label} raw visual audit run identity mismatch")
    if audit["model_sha256"] != hashes["model"]:
        raise ResultGateInputError(f"{label} raw visual audit model hash mismatch")
    if audit["dataset_manifest_sha256"] != hashes["dataset_manifest"]:
        raise ResultGateInputError(f"{label} raw visual audit dataset hash mismatch")

    registered = _nonnegative_integer(
        audit["registered_visual_params"],
        f"{label}.raw_visual_audit.registered_visual_params",
    )
    finite_grad = _nonnegative_integer(
        audit["nonzero_finite_grad_params"],
        f"{label}.raw_visual_audit.nonzero_finite_grad_params",
    )
    changed = _nonnegative_integer(
        audit["changed_visual_params"],
        f"{label}.raw_visual_audit.changed_visual_params",
    )
    frozen_delta = _finite_number(
        audit["frozen_param_delta_max"],
        f"{label}.raw_visual_audit.frozen_param_delta_max",
        minimum=0.0,
    )
    if finite_grad > registered or changed > registered:
        raise ResultGateInputError(
            f"{label} raw visual audit counts exceed registered visual parameters"
        )
    if audit["status"] != "PASS":
        raise ResultGateProtocolError(f"{label} raw visual audit did not pass")
    if variant == "frozen":
        if any((registered, finite_grad, changed)) or frozen_delta > 1e-12:
            raise ResultGateProtocolError(
                f"{label} frozen raw visual audit records trainable or changed parameters"
            )
    else:
        if registered <= 0 or finite_grad <= 0 or changed <= 0:
            raise ResultGateProtocolError(
                f"{label} adapted raw visual audit lacks nonzero gradients or parameter changes"
            )
        if frozen_delta > 1e-12:
            raise ResultGateProtocolError(
                f"{label} adapted raw visual audit changed frozen parameters"
            )
    return audit


def _validate_run_manifest(
    path,
    row,
    claim,
    variant,
    seed,
    hashes,
    raw_visual_hash,
    label,
):
    manifest = _load_evidence_json(path, f"{label}.run_manifest")
    required = {
        "schema_version",
        "claim",
        "variant",
        "seed",
        "ledger_sha256",
        "commitment_sha256",
        "config_sha256",
        "model_sha256",
        "dataset_manifest_sha256",
        "evaluator_sha256",
        "metrics_sha256",
        "cost_sha256",
        "protocol_sha256",
        "raw_visual_audit_sha256",
    }
    if set(manifest) != required:
        raise ResultGateInputError(
            f"{label}.run_manifest fields differ: "
            f"expected {sorted(required)}, found {sorted(manifest)}"
        )
    if manifest["schema_version"] != RUN_MANIFEST_SCHEMA:
        raise ResultGateInputError(f"{label}.run_manifest schema is unsupported")
    if (
        _claim_name(manifest["claim"]) != claim
        or _variant_name(claim, manifest["variant"]) != variant
        or _seed(manifest["seed"], f"{label}.run_manifest.seed") != seed
    ):
        raise ResultGateInputError(f"{label}.run_manifest run identity mismatch")

    expected = {
        "ledger_sha256": hashes["ledger"],
        "commitment_sha256": hashes["commitment"],
        "config_sha256": hashes["config"],
        "model_sha256": hashes["model"],
        "dataset_manifest_sha256": hashes["dataset_manifest"],
        "evaluator_sha256": hashes["evaluator"],
        "metrics_sha256": hashes["metrics"],
        "cost_sha256": hashes.get("cost"),
        "protocol_sha256": _sha256_json(row["protocol"], f"{label}.protocol"),
        "raw_visual_audit_sha256": raw_visual_hash,
    }
    for field, expected_value in expected.items():
        if manifest[field] != expected_value:
            raise ResultGateInputError(f"{label}.run_manifest {field} mismatch")
    return manifest


def _validate_run_evidence(row, claim, variant, seed, label):
    evidence = row.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ResultGateInputError(f"{label} requires verified ledger evidence")

    prefixes = ["ledger", "commitment", *SHA256_FIELDS]
    if claim == "C1":
        prefixes.append("cost")
    else:
        prefixes.append("raw_visual_audit")
    expected_fields = {"ledger_count", "ledger_final_hashes"}
    for prefix in prefixes:
        expected_fields.update({f"{prefix}_path", f"{prefix}_sha256"})
    if set(evidence) != expected_fields:
        raise ResultGateInputError(
            f"{label}.evidence fields differ: "
            f"expected {sorted(expected_fields)}, found {sorted(evidence)}"
        )

    paths = {}
    hashes = {}
    for prefix in prefixes:
        paths[prefix], hashes[prefix] = _evidence_file(evidence, prefix, label)

    try:
        ledger = load_verified_ledger(paths["ledger"], paths["commitment"])
    except LedgerError as exc:
        raise ResultGateInputError(f"{label} verified ledger evidence failed: {exc}") from exc
    expected_count = _nonnegative_integer(
        evidence["ledger_count"], f"{label}.evidence.ledger_count"
    )
    if ledger.count != expected_count:
        raise ResultGateInputError(f"{label} ledger count differs from evidence")
    expected_final_hashes = evidence["ledger_final_hashes"]
    if not isinstance(expected_final_hashes, dict):
        raise ResultGateInputError(
            f"{label}.evidence.ledger_final_hashes must be an object"
        )
    for stream_id, final_hash in expected_final_hashes.items():
        if not isinstance(stream_id, str) or not stream_id:
            raise ResultGateInputError(f"{label} ledger final-hash stream id is invalid")
        _required_sha256(final_hash, f"{label}.ledger_final_hashes.{stream_id}")
    if ledger.final_hashes != expected_final_hashes:
        raise ResultGateInputError(f"{label} ledger final hashes differ from evidence")

    metrics_artifact = _load_evidence_json(paths["metrics"], f"{label}.metrics evidence")
    if metrics_artifact != row.get("metrics"):
        raise ResultGateInputError(f"{label} metrics differ from verified metrics evidence")
    if claim == "C1":
        cost_artifact = _load_evidence_json(paths["cost"], f"{label}.cost evidence")
        if cost_artifact != row.get("cost"):
            raise ResultGateInputError(f"{label} cost differs from verified cost evidence")

    raw_visual_hash = None
    raw_visual_audit = None
    if claim == "C2":
        raw_visual_hash = hashes["raw_visual_audit"]
        raw_visual_audit = _validate_raw_visual_audit(
            paths["raw_visual_audit"],
            claim,
            variant,
            seed,
            hashes,
            label,
        )
    _validate_run_manifest(
        paths["run_manifest"],
        row,
        claim,
        variant,
        seed,
        hashes,
        raw_visual_hash,
        label,
    )
    return {
        "hashes": hashes,
        "ledger_count": ledger.count,
        "ledger_final_hashes": ledger.final_hashes,
        "raw_visual_audit": raw_visual_audit,
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


def _validate_b0_evidence(containers):
    evidence = _b0_evidence_entry(containers)
    if evidence is None:
        return {"provided": False, "passed": False, "violations": []}
    if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256"}:
        raise ResultGateInputError("b0_evidence requires exactly path and sha256")
    wrapper = {
        "b0_path": evidence["path"],
        "b0_sha256": evidence["sha256"],
    }
    b0_path, b0_hash = _evidence_file(wrapper, "b0", "b0_evidence")
    artifact = _load_evidence_json(b0_path, "b0_evidence artifact")
    required = {
        "schema_version",
        "status",
        "commit_sha",
        "test_count",
        "blocking_findings",
        "protocol_violations",
        "test_report_path",
        "test_report_sha256",
        "audit_report_path",
        "audit_report_sha256",
    }
    if set(artifact) != required:
        raise ResultGateInputError(
            "B0 artifact fields differ: "
            f"expected {sorted(required)}, found {sorted(artifact)}"
        )
    if artifact["schema_version"] != B0_SCHEMA:
        raise ResultGateInputError("unsupported B0 artifact schema")
    commit_sha = _required_git_sha(artifact["commit_sha"], "B0.commit_sha")
    test_count = _nonnegative_integer(artifact["test_count"], "B0.test_count", positive=True)
    blocking_findings = _nonnegative_integer(
        artifact["blocking_findings"], "B0.blocking_findings"
    )
    protocol_violations = _nonnegative_integer(
        artifact["protocol_violations"], "B0.protocol_violations"
    )

    linked = {
        "test_report_path": artifact["test_report_path"],
        "test_report_sha256": artifact["test_report_sha256"],
        "audit_report_path": artifact["audit_report_path"],
        "audit_report_sha256": artifact["audit_report_sha256"],
    }
    test_path, _ = _evidence_file(linked, "test_report", "B0")
    audit_path, _ = _evidence_file(linked, "audit_report", "B0")
    test_report = _load_evidence_json(test_path, "B0 test report")
    audit_report = _load_evidence_json(audit_path, "B0 audit report")

    test_required = {"schema_version", "status", "commit_sha", "collected", "passed", "failed"}
    if set(test_report) != test_required or test_report.get("schema_version") != B0_TEST_REPORT_SCHEMA:
        raise ResultGateInputError("B0 test report schema or fields are invalid")
    if _required_git_sha(test_report["commit_sha"], "B0.test_report.commit_sha") != commit_sha:
        raise ResultGateInputError("B0 test report commit does not match B0 artifact")
    collected = _nonnegative_integer(test_report["collected"], "B0.test_report.collected")
    passed = _nonnegative_integer(test_report["passed"], "B0.test_report.passed")
    failed = _nonnegative_integer(test_report["failed"], "B0.test_report.failed")
    test_pass = (
        test_report["status"] == "PASS"
        and collected == test_count
        and passed == test_count
        and failed == 0
    )

    audit_required = {
        "schema_version",
        "status",
        "commit_sha",
        "blocking_findings",
        "protocol_violations",
    }
    if set(audit_report) != audit_required or audit_report.get("schema_version") != B0_AUDIT_REPORT_SCHEMA:
        raise ResultGateInputError("B0 audit report schema or fields are invalid")
    if _required_git_sha(audit_report["commit_sha"], "B0.audit_report.commit_sha") != commit_sha:
        raise ResultGateInputError("B0 audit report commit does not match B0 artifact")
    audit_blockers = _nonnegative_integer(
        audit_report["blocking_findings"], "B0.audit_report.blocking_findings"
    )
    audit_violations = _nonnegative_integer(
        audit_report["protocol_violations"], "B0.audit_report.protocol_violations"
    )
    audit_pass = (
        audit_report["status"] == "PASS"
        and audit_blockers == blocking_findings == 0
        and audit_violations == protocol_violations == 0
    )
    overall_pass = artifact["status"] == "PASS" and test_pass and audit_pass
    violations = []
    if not test_pass:
        violations.append("B0 test report did not pass or count accounting differs")
    if not audit_pass:
        violations.append("B0 audit report contains blockers or protocol violations")
    if artifact["status"] not in {"PASS", "FAIL"}:
        raise ResultGateInputError("B0.status must be PASS or FAIL")
    return {
        "provided": True,
        "passed": overall_pass,
        "violations": violations,
        "artifact_sha256": b0_hash,
        "commit_sha": commit_sha,
        "test_count": test_count,
    }


def _normalize_prerequisites(containers, run_count):
    b0 = _validate_b0_evidence(containers)
    return {
        "protocol": {
            "provided": run_count > 0,
            "passed": run_count > 0,
            "violations": [],
            "source": "verified_run_evidence",
        },
        "B0": b0,
    }


def _normalize_runs(raw_runs):
    grouped = {
        "C1": {"fixed": {}, "rematch": {}},
        "C2": {"frozen": {}, "adapted": {}},
    }
    for index, row in enumerate(raw_runs):
        if not isinstance(row, dict):
            raise ResultGateInputError(f"runs[{index}] must be a JSON object")
        if "claim" not in row or "variant" not in row or "seed" not in row:
            raise ResultGateInputError(
                f"runs[{index}] requires claim, variant, and seed fields"
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
            "metrics": _normalized_metrics(row, claim, label),
            "evidence": evidence,
        }
        if claim == "C1":
            normalized["cost"] = _normalized_cost(row, label)
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
            **({"raw_visual_audit": False} if claim == "C2" else {}),
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
            "raw_visual_audit": True,
        },
        "paired_seed_coverage": {
            "passed": True,
            "seeds": seeds,
            "variants": list(variants),
        },
        "protocol_equality": {"passed": True},
        "raw_visual_audit": {
            "passed": True,
            "per_seed": [
                {
                    "seed": seed,
                    "adapted": grouped["C2"]["adapted"][seed]["evidence"]["raw_visual_audit"],
                    "frozen": grouped["C2"]["frozen"][seed]["evidence"]["raw_visual_audit"],
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
    prerequisites = _normalize_prerequisites(containers, len(raw_runs))
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


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResultGateInputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise ResultGateInputError(f"non-finite JSON number is not allowed: {value}")


def _load_json(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
    except ResultGateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResultGateInputError(f"failed to read {path}: {exc}") from exc


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
