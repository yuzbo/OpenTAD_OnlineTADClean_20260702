"""Validate one arm of the paired D1.6 one-epoch mechanism experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

try:
    from scripts.finalize_eventmatr_d1_pilot import FIXED_OPTIONS, validate_metric_lines
    from scripts.finalize_eventmatr_d11_mechanism import (
        EFFECTIVE_DOSE_METRICS,
        validate_effective_dose_metrics,
        validate_effective_dose_trace,
    )
except ModuleNotFoundError:
    from finalize_eventmatr_d1_pilot import FIXED_OPTIONS, validate_metric_lines
    from finalize_eventmatr_d11_mechanism import (
        EFFECTIVE_DOSE_METRICS,
        validate_effective_dose_metrics,
        validate_effective_dose_trace,
    )


CHECKPOINT_SCHEMA = "eventmatr_d16_policy_independent_risk_mechanism_v1"
ARM_VARIANTS = {"control": "none", "risk": "policy_independent"}
RISK_CONTRACTS = {
    "control": "runtime_conditioned_owner_risk_v1",
    "risk": "policy_independent_competing_risk_v1",
}
COMMON_CONTRACTS = {
    "association_contract": "soft_target_class_log_probability_v1",
    "birth_risk_contract": "event_matched_hard_negative_v1",
    "birth_objective_contract": "decision_aligned_interval_bag_v1",
}
D16_CENSUS_SUFFIXES = (
    "alive_positive_count",
    "end_positive_count",
    "ragged_track_count",
    "false_track_cancel_group_count",
    "source_target_visible_row_count",
    "source_target_visible_group_count",
    "source_target_visible_class_row_count",
    "source_target_visible_end_risk_group_count",
    "source_predicted_unresolved_row_count",
    "source_predicted_unresolved_group_count",
    "source_predicted_unresolved_class_row_count",
    "source_predicted_unresolved_end_risk_group_count",
)
FROZEN_BIRTH_EXACT = {
    "birth_positive_count": 3003,
    "birth_selected_negative_count": 3003,
    "birth_positive_batch_count": 1685,
    "birth_zero_positive_batch_count": 1585,
    "birth_interval_fallback_count": 0,
    "birth_prebirth_exposure_count": 180182,
    "birth_interval_exposure_count": 5666,
    "birth_selected_risk_logit_count": 185852,
    "birth_postinterval_ignored_exposure_count": 4,
    "birth_normalized_survival_event_count": 0,
    "birth_decision_aligned_positive_bag_count": 3003,
    "birth_decision_aligned_negative_bag_count": 3003,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_structure_sha256(state_dict: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in state_dict.items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def validate_d16_mechanism_metrics(metrics: dict, arm: str) -> dict:
    if arm not in ARM_VARIANTS:
        raise ValueError(f"invalid D1.6 arm: {arm}")

    required = {
        "loss_event_owner_state_unscaled",
        "loss_event_end_unscaled",
        "event_transition_gradient_norm",
        "event_birth_gradient_norm",
        "event_owner_gradient_norm",
        "event_runtime_capacity_exhaustions_unscaled",
        "event_association_audit_class_argmax_reject_pair_count_unscaled",
        "event_association_audit_admissible_class_argmax_mismatch_pair_count_unscaled",
        "d16_epoch_physical_batch_count",
        *(f"d16_epoch_{suffix}_total" for suffix in FROZEN_BIRTH_EXACT),
        *(f"d16_epoch_{suffix}_total" for suffix in D16_CENSUS_SUFFIXES),
    }
    missing = sorted(required.difference(metrics))
    if missing:
        raise ValueError(f"D1.6 mechanism metrics are missing keys: {missing}")
    census = {}
    for name in sorted(required):
        value = float(metrics[name])
        if not math.isfinite(value):
            raise ValueError(f"D1.6 mechanism metric is non-finite: {name}={value}")
        if name.startswith("d16_epoch_"):
            if not value.is_integer() or value < 0:
                raise ValueError(
                    f"D1.6 census must be a non-negative integer: {name}={value}"
                )
            census[name] = int(value)

    if float(metrics["loss_event_owner_state_unscaled"]) <= 0:
        raise ValueError("D1.6 owner-state objective is not live")
    for name in (
        "event_transition_gradient_norm",
        "event_birth_gradient_norm",
        "event_owner_gradient_norm",
    ):
        if float(metrics[name]) <= 0:
            raise ValueError(f"D1.6 gradient is not live: {name}")
    if float(metrics["event_runtime_capacity_exhaustions_unscaled"]) != 0:
        raise ValueError("D1.6 exhausted dynamic lifecycle capacity")
    if float(metrics["d16_epoch_physical_batch_count"]) != 3270:
        raise ValueError("D1.6 physical batch count did not close at 3270")
    for suffix, expected in FROZEN_BIRTH_EXACT.items():
        name = f"d16_epoch_{suffix}_total"
        if census[name] != expected:
            raise ValueError(
                f"D1.6 frozen birth census drifted: {name}={census[name]} != {expected}"
            )
    if float(metrics["event_association_audit_class_argmax_reject_pair_count_unscaled"]) != 0:
        raise ValueError("D1.6 restored the rejected hard class gate")
    if float(
        metrics[
            "event_association_audit_admissible_class_argmax_mismatch_pair_count_unscaled"
        ]
    ) <= 0:
        raise ValueError("D1.6 did not preserve the soft-assignment intervention")
    target_prefix = "d16_epoch_source_target_visible_"
    unresolved_prefix = "d16_epoch_source_predicted_unresolved_"
    if arm == "control":
        for name, value in census.items():
            if (name.startswith(target_prefix) or name.startswith(unresolved_prefix)) and value != 0:
                raise ValueError(
                    f"D1.6 control unexpectedly entered the independent risk path: {name}={value}"
                )
        if float(metrics["loss_event_end_unscaled"]) <= 0:
            raise ValueError("D1.6 control lost its frozen auxiliary END hazard")
    else:
        if census["d16_epoch_end_positive_count_total"] != 3001:
            raise ValueError(
                "D1.6 independent risk did not close the 3001 observable endpoints"
            )
        if census["d16_epoch_source_target_visible_row_count_total"] <= 0:
            raise ValueError("D1.6 independent target-visible risk path is empty")
        if census["d16_epoch_source_target_visible_class_row_count_total"] <= 0:
            raise ValueError("D1.6 target-visible risk lost class supervision")
        if census["d16_epoch_source_target_visible_end_risk_group_count_total"] < 3001:
            raise ValueError("D1.6 target-visible END risk coverage is incomplete")
        if float(metrics["loss_event_end_unscaled"]) != 0:
            raise ValueError("D1.6 duplicated END with the frozen auxiliary hazard")

    return {
        "arm": arm,
        "event_d16_variant": ARM_VARIANTS[arm],
        "owner_risk_contract": RISK_CONTRACTS[arm],
        "d16_census": census,
        "frozen_birth_census": {
            name: census[f"d16_epoch_{name}_total"]
            for name in FROZEN_BIRTH_EXACT
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=tuple(ARM_VARIANTS))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    result_dirs = sorted((output_root / "result").glob("*__TH__*"))
    if len(result_dirs) != 1:
        raise SystemExit(f"expected exactly one TH result directory, got {result_dirs}")
    result_dir = result_dirs[0]
    options_path = result_dir / "opts.json"
    metrics_path = result_dir / "mechanism_epoch_metrics.jsonl"
    trace_path = result_dir / "effective_dose_update_trace.jsonl"
    options = json.loads(options_path.read_text(encoding="utf-8"))
    metric_lines = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    try:
        validate_metric_lines(metric_lines, "TH", 1)
        metric_row = metric_lines[0]
        expected_variant = ARM_VARIANTS[args.arm]
        expected_metric_fields = {
            "study_protocol": "d16_mechanism",
            "event_d13_variant": "combined",
            "event_d14_variant": "decision_aligned_bag",
            "event_d16_variant": expected_variant,
        }
        for field, expected in expected_metric_fields.items():
            if metric_row.get(field) != expected:
                raise ValueError(
                    f"D1.6 metric {field} drifted: {metric_row.get(field)!r} != {expected!r}"
                )
        initialization_sha256 = metric_row.get("model_initialization_sha256")
        if not isinstance(initialization_sha256, str) or len(initialization_sha256) != 64:
            raise ValueError("D1.6 metric row lacks an initialization SHA-256")
        validate_effective_dose_metrics(metric_row["metrics"])
        trace_receipt = validate_effective_dose_trace(trace_path)
        mechanism_validation = validate_d16_mechanism_metrics(
            metric_row["metrics"], args.arm
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    checkpoint = result_dir / "terminal_epoch1.pth"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise SystemExit(f"terminal checkpoint is absent or empty: {checkpoint}")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    expected_checkpoint = {
        "epoch": 1,
        "study_protocol": "d16_mechanism",
        "model_variant": "eventmatr",
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "event_lifecycle_version": "d1_censored",
        "event_d1_lane": "th",
        "owner_state_count": 3,
        "birth_head": "independent_binary_hazard",
        "event_d13_variant": "combined",
        "event_d14_variant": "decision_aligned_bag",
        "event_d16_variant": ARM_VARIANTS[args.arm],
        "owner_risk_contract": RISK_CONTRACTS[args.arm],
        "model_initialization_sha256": initialization_sha256,
        **COMMON_CONTRACTS,
    }
    for field, expected in expected_checkpoint.items():
        if checkpoint_payload.get(field) != expected:
            raise SystemExit(
                f"terminal checkpoint {field} mismatch: "
                f"{checkpoint_payload.get(field)!r} != {expected!r}"
            )
    state_dict = checkpoint_payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise SystemExit("terminal checkpoint has no model state")

    locked_test_sentinel = (output_root / "LOCKED_TEST_NOT_MOUNTED.pickle").resolve()
    if locked_test_sentinel.exists():
        raise SystemExit("locked-test sentinel unexpectedly exists")
    expected_options = {
        **FIXED_OPTIONS,
        "epochs": 1,
        "train_eval_step": 1,
        "study_protocol": "d16_mechanism",
        "model_variant": "eventmatr",
        "event_lifecycle_version": "d1_censored",
        "event_d1_lane": "th",
        "event_d13_variant": "combined",
        "event_d14_variant": "decision_aligned_bag",
        "event_d16_variant": ARM_VARIANTS[args.arm],
        "birth_mode": "instant_transition",
        "ownership_mode": "sticky_owner",
        "event_arm": "b1o1",
        "event_teacher_forcing_ratio": 0.5,
        "d11_effective_dose": True,
        "event_resource_limit": 0,
        "reduce": 1,
        "make_output": True,
        "load_model": False,
    }
    for field, expected in expected_options.items():
        if options.get(field) != expected:
            raise SystemExit(
                f"mechanism option {field} mismatch: {options.get(field)!r} != {expected!r}"
            )
    if Path(options["video_feature_all_test"]).resolve() != locked_test_sentinel:
        raise SystemExit("D1.6 run did not retain the absent locked-test sentinel")
    if options.get("event_birth_logit_threshold") is not None:
        raise SystemExit("D1.6 mechanism used a fixed birth threshold")
    if options.get("event_end_logit_threshold") is not None:
        raise SystemExit("D1.6 mechanism used a fixed end threshold")

    expected_identity = {
        "commit": args.source_commit,
        "tree": args.source_tree,
        "manifest_sha256": args.manifest_sha256,
    }
    source_identity_receipts = {}
    for phase in ("start", "final"):
        identity_path = output_root / f"source_identity_{phase}.json"
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if identity.get("status") != "PASS" or identity.get("clean") is not True:
            raise SystemExit(f"D1.6 {phase} identity receipt is not PASS/clean")
        for field, expected in expected_identity.items():
            if identity.get(field) != expected:
                raise SystemExit(f"D1.6 {phase} identity {field} mismatch")
        smoke = identity.get("smoke")
        if (
            not isinstance(smoke, dict)
            or smoke.get("status") != "PASS"
            or smoke.get("test_access") is not False
            or smoke.get("formal_training_started") is not False
        ):
            raise SystemExit(f"D1.6 {phase} identity lacks the passing no-test smoke")
        source_identity_receipts[phase] = identity

    final_metrics = metric_lines[0]["metrics"]
    receipt = {
        "status": "PASS_TRAIN_MECHANISM_ONLY",
        "protocol": "eventmatr_d16_seed52_paired_risk_mechanism_v1",
        "arm": args.arm,
        "event_d16_variant": ARM_VARIANTS[args.arm],
        "owner_risk_contract": RISK_CONTRACTS[args.arm],
        "lane": "TH",
        "epochs": 1,
        "seed": 52,
        "fresh_start": True,
        "test_access": False,
        "checkpoint_updated": True,
        "strict_causal_paper_result_valid": False,
        "official_paper_performance_valid": False,
        "train_prefix_metrics_diagnostic_only": True,
        "threshold_search": False,
        "threshold_lowering": False,
        "multi_seed": False,
        "official_comparison_release": False,
        "query_internalization_release": False,
        "locked_test_release": False,
        "endpoint_margin_gate_pending": True,
        "model_initialization_sha256": initialization_sha256,
        "model_state_structure_sha256": _state_structure_sha256(state_dict),
        "source_identity": expected_identity,
        "source_identity_receipts": source_identity_receipts,
        "smoke_gate": source_identity_receipts["final"]["smoke"],
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
            **expected_checkpoint,
        },
        "options": {
            "path": str(options_path),
            "bytes": options_path.stat().st_size,
            "sha256": _sha256(options_path),
        },
        "metrics_artifact": {
            "path": str(metrics_path),
            "bytes": metrics_path.stat().st_size,
            "sha256": _sha256(metrics_path),
        },
        "mechanism_validation": mechanism_validation,
        "effective_dose": {
            name: final_metrics[name] for name in EFFECTIVE_DOSE_METRICS
        },
        "effective_dose_update_trace": trace_receipt,
        "final_epoch_metrics": final_metrics,
    }
    receipt_path = output_root / "mechanism_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(receipt_path)


if __name__ == "__main__":
    main()
