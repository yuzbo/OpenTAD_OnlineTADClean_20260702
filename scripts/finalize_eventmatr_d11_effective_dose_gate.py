"""Close the D1.1 effective-dose training, terminal, and delta evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required D1.1 gate artifact is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"D1.1 gate artifact is not an object: {path}")
    return payload


def _require_identity(identity: dict, commit: str, tree: str, label: str) -> None:
    if not isinstance(identity, dict):
        raise ValueError(f"{label} source identity is absent")
    if identity.get("commit") != commit or identity.get("tree") != tree:
        raise ValueError(f"{label} source identity mismatch")


def _nonnegative_integer_count(counts: dict, field: str) -> int:
    value = counts.get(field)
    if isinstance(value, bool):
        raise ValueError(f"D1.1 terminal lifecycle count is invalid: {field}={value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"D1.1 terminal lifecycle count is invalid: {field}={value!r}"
        ) from error
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(
            f"D1.1 terminal lifecycle count is invalid: {field}={value!r}"
        )
    return int(numeric)


def validate_effective_dose_gate(
    mechanism: dict,
    scan: dict,
    delta: dict,
    *,
    checkpoint_sha256: str,
    training_source_commit: str,
    training_source_tree: str,
) -> dict:
    if mechanism.get("status") != "PASS" or mechanism.get("protocol") != (
        "eventmatr_d11_seed52_effective_dose_mechanism_v2"
    ):
        raise ValueError("D1.1 effective-dose training receipt is not PASS/v2")
    if (
        mechanism.get("test_access") is not False
        or mechanism.get("checkpoint_updated") is not True
        or mechanism.get("strict_causal_paper_result_valid") is not False
        or mechanism.get("performance_gate_applied") is not False
        or mechanism.get("existing_five_epoch_matrix_release") is not False
    ):
        raise ValueError("D1.1 training receipt crossed its evidence boundary")
    _require_identity(
        mechanism.get("source_identity"),
        training_source_commit,
        training_source_tree,
        "training receipt",
    )
    if mechanism.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ValueError("D1.1 training receipt checkpoint hash mismatch")
    effective_dose = mechanism.get("effective_dose", {})
    effective_dose_expected = {
        "d11_effective_dose_enabled": 1.0,
        "d11_optimizer_step_count": 3270.0,
        "d11_initial_learning_rate": 1e-8,
        "d11_learning_rate_first": 3.34e-6,
        "d11_learning_rate_minimum": 3.34e-6,
        "d11_learning_rate_maximum": 3.34e-6,
        "d11_learning_rate_last": 3.34e-6,
        "d11_scheduler_last_epoch_at_train_start": 1.0,
        "d11_scheduler_t_cur_at_train_start": 1.0,
    }
    for field, expected in effective_dose_expected.items():
        value = float(effective_dose.get(field, float("nan")))
        if not math.isfinite(value) or not math.isclose(
            value,
            expected,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"D1.1 training receipt effective dose drifted: "
                f"{field}={value} != {expected}"
            )
    trace = mechanism.get("effective_dose_update_trace", {})
    trace_expected = {
        "row_count": 3270,
        "first_optimizer_step": 1,
        "last_optimizer_step": 3270,
    }
    for field, expected in trace_expected.items():
        if int(trace.get(field, -1)) != expected:
            raise ValueError(
                f"D1.1 training receipt update trace drifted: "
                f"{field}={trace.get(field)!r} != {expected}"
            )
    trace_learning_rate = float(trace.get("learning_rate", float("nan")))
    if not math.isfinite(trace_learning_rate) or not math.isclose(
        trace_learning_rate,
        3.34e-6,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("D1.1 training receipt update-trace learning rate drifted")
    trace_sha256 = trace.get("sha256")
    if (
        not isinstance(trace_sha256, str)
        or len(trace_sha256) != 64
        or any(character not in "0123456789abcdef" for character in trace_sha256)
    ):
        raise ValueError("D1.1 training receipt update-trace hash is invalid")
    trace_path_value = trace.get("path")
    if not isinstance(trace_path_value, str) or not trace_path_value:
        raise ValueError("D1.1 training receipt update-trace path is absent")
    trace_path = Path(trace_path_value).expanduser().resolve()
    if not trace_path.is_file():
        raise ValueError("D1.1 training receipt update-trace artifact is absent")
    if trace_path.stat().st_size != int(trace.get("bytes", -1)):
        raise ValueError("D1.1 training receipt update-trace size mismatch")
    if _sha256(trace_path) != trace_sha256:
        raise ValueError("D1.1 training receipt update-trace hash mismatch")

    scan_expected = {
        "status": "DIAGNOSTIC_COMPLETE",
        "execution_status": "PASS",
        "protocol": "eventmatr_d11_terminal_association_scan_v4",
        "complete_scan": True,
        "test_access": False,
        "checkpoint_updated": False,
        "strict_causal_paper_result_valid": False,
        "performance_gate_release": False,
        "threshold_search": False,
        "one_epoch_mechanism_gate_status": "PASS_TRAIN_ONLY",
    }
    for field, expected in scan_expected.items():
        if scan.get(field) != expected:
            raise ValueError(
                f"D1.1 terminal scan {field} mismatch: "
                f"{scan.get(field)!r} != {expected!r}"
            )
    if scan.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ValueError("D1.1 terminal scan checkpoint hash mismatch")
    _require_identity(
        scan.get("training_source_identity"),
        training_source_commit,
        training_source_tree,
        "terminal scan training",
    )
    _require_identity(
        scan.get("source_identity"),
        training_source_commit,
        training_source_tree,
        "terminal scan execution",
    )
    counts = scan.get("barrier_counts")
    if not isinstance(counts, dict):
        raise ValueError("D1.1 terminal scan has no barrier counts")
    lifecycle_fields = (
        "predicted_start_active_query_count",
        "predicted_birth_query_count",
        "assignment_count",
        "runtime_birth_count",
        "runtime_cancel_count",
        "runtime_end_count",
        "runtime_emit_count",
        "runtime_capacity_exhaustion_count",
    )
    lifecycle_counts = {
        field: _nonnegative_integer_count(counts, field)
        for field in lifecycle_fields
    }
    required_positive = (
        "predicted_start_active_query_count",
        "predicted_birth_query_count",
        "assignment_count",
        "runtime_birth_count",
        "runtime_cancel_count",
        "runtime_end_count",
    )
    for field in required_positive:
        value = lifecycle_counts[field]
        if value <= 0:
            raise ValueError(f"D1.1 terminal lifecycle is not live: {field}={value}")
    if lifecycle_counts["runtime_capacity_exhaustion_count"] != 0:
        raise ValueError("D1.1 terminal lifecycle exhausted dynamic capacity")
    runtime_end = lifecycle_counts["runtime_end_count"]
    runtime_emit = lifecycle_counts["runtime_emit_count"]
    if runtime_emit != runtime_end:
        raise ValueError("D1.1 terminal END and immutable emission counts differ")

    delta_expected = {
        "status": "PASS",
        "protocol": "eventmatr_d11_epoch1_parameter_delta_audit_v1",
        "test_access": False,
        "checkpoint_updated": False,
        "model_forward_executed": False,
        "optimizer_step_executed": False,
        "strict_causal_paper_result_valid": False,
        "initialization_reconstructed_from_exact_training_commit": True,
        "seed": 52,
        "construction_order": "seed_then_train_dataset_then_model",
    }
    for field, expected in delta_expected.items():
        if delta.get(field) != expected:
            raise ValueError(
                f"D1.1 parameter audit {field} mismatch: "
                f"{delta.get(field)!r} != {expected!r}"
            )
    if delta.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ValueError("D1.1 parameter audit checkpoint hash mismatch")
    _require_identity(
        delta.get("training_source_identity"),
        training_source_commit,
        training_source_tree,
        "parameter audit training",
    )
    _require_identity(
        delta.get("audit_source_identity"),
        training_source_commit,
        training_source_tree,
        "parameter audit execution",
    )
    if delta.get("initialization_source_commit") != training_source_commit:
        raise ValueError("D1.1 parameter audit initialization source mismatch")
    recorded_learning_rate = float(
        delta.get("metrics", {}).get(
            "recorded_epoch_average_learning_rate",
            float("nan"),
        )
    )
    if not math.isfinite(recorded_learning_rate) or not math.isclose(
        recorded_learning_rate,
        3.34e-6,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("D1.1 parameter audit learning-rate record drifted")
    optimizer = delta.get("optimizer", {})
    if (
        optimizer.get("step_count_closed") is not True
        or int(optimizer.get("expected_step", -1)) != 3270
        or int(optimizer.get("maximum_step", -1)) != 3270
    ):
        raise ValueError("D1.1 parameter audit optimizer steps did not close")
    delta_groups = delta.get("parameter_delta_by_group", {})
    component_groups = (
        "event_transition_head",
        "event_owner_decoder",
        "other_event_parameters",
        "inherited_parent_parameters",
    )
    all_parameters = delta_groups.get("all_parameters", {})
    all_tensor_count = int(all_parameters.get("parameter_tensor_count", -1))
    all_element_count = int(all_parameters.get("element_count", -1))
    all_changed_count = int(all_parameters.get("changed_element_count", -1))
    if (
        all_tensor_count <= 0
        or all_element_count <= 0
        or all_changed_count <= 0
    ):
        raise ValueError("D1.1 parameter audit has no complete all-parameter delta")
    component_tensor_count = 0
    component_element_count = 0
    component_changed_count = 0
    component_delta_l2_squared = 0.0
    component_initial_l2_squared = 0.0
    for group in component_groups:
        group_delta = delta_groups.get(group, {})
        tensor_count = int(group_delta.get("parameter_tensor_count", -1))
        element_count = int(group_delta.get("element_count", -1))
        changed_count = int(group_delta.get("changed_element_count", -1))
        delta_l2 = float(group_delta.get("delta_l2", float("nan")))
        initial_l2 = float(group_delta.get("initial_l2", float("nan")))
        if (
            tensor_count < 0
            or element_count < 0
            or changed_count < 0
            or changed_count > element_count
            or not math.isfinite(delta_l2)
            or delta_l2 < 0.0
            or not math.isfinite(initial_l2)
            or initial_l2 < 0.0
        ):
            raise ValueError(f"D1.1 parameter audit group is invalid: {group}")
        component_tensor_count += tensor_count
        component_element_count += element_count
        component_changed_count += changed_count
        component_delta_l2_squared += delta_l2 * delta_l2
        component_initial_l2_squared += initial_l2 * initial_l2
    if (
        component_tensor_count != all_tensor_count
        or component_element_count != all_element_count
        or component_changed_count != all_changed_count
    ):
        raise ValueError("D1.1 parameter audit group counts did not close")
    all_delta_l2 = float(all_parameters.get("delta_l2", float("nan")))
    all_initial_l2 = float(all_parameters.get("initial_l2", float("nan")))
    if (
        not math.isfinite(all_delta_l2)
        or all_delta_l2 <= 0.0
        or not math.isclose(
            all_delta_l2 * all_delta_l2,
            component_delta_l2_squared,
            rel_tol=1e-10,
            abs_tol=1e-20,
        )
        or not math.isfinite(all_initial_l2)
        or all_initial_l2 <= 0.0
        or not math.isclose(
            all_initial_l2 * all_initial_l2,
            component_initial_l2_squared,
            rel_tol=1e-10,
            abs_tol=1e-20,
        )
    ):
        raise ValueError("D1.1 all-parameter delta norms did not close")
    for group in ("event_transition_head", "event_owner_decoder"):
        group_delta = delta_groups.get(group, {})
        value = float(group_delta.get("delta_l2", float("nan")))
        relative = float(group_delta.get("relative_l2_delta", float("nan")))
        if (
            not math.isfinite(value)
            or not math.isfinite(relative)
            or value <= 0.0
            or relative <= 0.0
        ):
            raise ValueError(f"D1.1 parameter group did not update: {group}")

    return {
        "terminal_lifecycle": {
            field: lifecycle_counts[field]
            for field in lifecycle_fields
        },
        "parameter_delta": {
            group: {
                "delta_l2": float(delta_groups[group]["delta_l2"]),
                "relative_l2_delta": float(
                    delta_groups[group]["relative_l2_delta"]
                ),
            }
            for group in ("event_transition_head", "event_owner_decoder")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-receipt", required=True, type=Path)
    parser.add_argument("--association-scan", required=True, type=Path)
    parser.add_argument("--parameter-delta-audit", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--training-source-commit", required=True)
    parser.add_argument("--training-source-tree", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    artifact_paths = {
        "mechanism_receipt": args.mechanism_receipt.expanduser().resolve(),
        "association_scan": args.association_scan.expanduser().resolve(),
        "parameter_delta_audit": args.parameter_delta_audit.expanduser().resolve(),
    }
    payloads = {name: _load(path) for name, path in artifact_paths.items()}
    evidence = validate_effective_dose_gate(
        payloads["mechanism_receipt"],
        payloads["association_scan"],
        payloads["parameter_delta_audit"],
        checkpoint_sha256=args.expected_checkpoint_sha256,
        training_source_commit=args.training_source_commit,
        training_source_tree=args.training_source_tree,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "PASS",
        "protocol": "eventmatr_d11_effective_dose_complete_gate_v1",
        "test_access": False,
        "strict_causal_paper_result_valid": False,
        "performance_gate_release": False,
        "threshold_search": False,
        "training_mechanism_gate_pass": True,
        "five_epoch_science_contract_eligible_for_freeze": True,
        "existing_five_epoch_matrix_release": False,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "training_source": {
            "commit": args.training_source_commit,
            "tree": args.training_source_tree,
        },
        "artifacts": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in artifact_paths.items()
        },
        **evidence,
    }
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
