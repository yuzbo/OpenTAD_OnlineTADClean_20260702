"""Validate one prospective D1.4 train-only decision-alignment run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

try:
    from scripts.finalize_eventmatr_d1_pilot import (
        FIXED_OPTIONS,
        validate_metric_lines,
    )
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


CHECKPOINT_SCHEMA = "eventmatr_d14_decision_alignment_mechanism_v1"
COMMON_CONTRACTS = {
    "association_contract": "soft_target_class_log_probability_v1",
    "birth_risk_contract": "event_matched_hard_negative_v1",
}
VARIANT_OBJECTIVES = {
    "normalized_survival": "event_normalized_censored_hazard_v1",
    "decision_aligned_bag": "decision_aligned_interval_bag_v1",
}
COMMON_POSITIVE_METRICS = (
    "event_transition_gradient_norm",
    "event_birth_gradient_norm",
    "event_owner_gradient_norm",
    "event_birth_positive_count_unscaled",
    "event_end_positive_count_unscaled",
    "event_owner_assignment_count_unscaled",
    "event_ragged_track_count_unscaled",
    "event_false_track_cancel_group_count_unscaled",
    "event_source_predicted_unmatched_row_count_unscaled",
    "event_source_teacher_birth_row_count_unscaled",
)
CENSUS_SUFFIXES = (
    "birth_positive_count",
    "birth_selected_negative_count",
    "birth_negative_candidate_count",
    "birth_positive_batch_count",
    "birth_zero_positive_batch_count",
    "birth_interval_fallback_count",
    "birth_prebirth_exposure_count",
    "birth_interval_exposure_count",
    "birth_selected_risk_logit_count",
    "birth_postinterval_ignored_exposure_count",
    "birth_prebirth_group_count",
    "birth_zero_prebirth_group_count",
    "birth_normalized_survival_event_count",
    "birth_decision_aligned_positive_bag_count",
    "birth_decision_aligned_negative_bag_count",
)
CENSUS_KEYS = tuple(f"d14_epoch_{suffix}_total" for suffix in CENSUS_SUFFIXES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_d14_mechanism_metrics(metrics: dict, variant: str) -> dict:
    if variant not in VARIANT_OBJECTIVES:
        raise ValueError(f"invalid D1.4 variant: {variant}")
    required = (
        set(COMMON_POSITIVE_METRICS)
        .union(CENSUS_KEYS)
        .union(
            {
                "d14_epoch_physical_batch_count",
                "event_runtime_capacity_exhaustions_unscaled",
                "event_association_audit_class_argmax_reject_pair_count_unscaled",
                (
                    "event_association_audit_"
                    "admissible_class_argmax_mismatch_pair_count_unscaled"
                ),
            }
        )
    )
    missing = sorted(required.difference(metrics))
    if missing:
        raise ValueError(f"D1.4 mechanism metrics are missing keys: {missing}")
    for name in COMMON_POSITIVE_METRICS:
        value = float(metrics[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"D1.4 mechanism metric is not live: {name}={value}")
    capacity = float(metrics["event_runtime_capacity_exhaustions_unscaled"])
    if not math.isfinite(capacity) or capacity != 0:
        raise ValueError(f"D1.4 mechanism exhausted dynamic capacity: {capacity}")
    if float(metrics["d14_epoch_physical_batch_count"]) != 3270.0:
        raise ValueError("D1.4 physical batch count did not close at 3270")

    census = {}
    for name in CENSUS_KEYS:
        value = float(metrics[name])
        if not math.isfinite(value) or not value.is_integer() or value < 0:
            raise ValueError(
                "D1.4 census metric must be a finite non-negative integer: "
                f"{name}={value}"
            )
        census[name] = int(value)
    exact = {
        "d14_epoch_birth_positive_count_total": 3003,
        "d14_epoch_birth_selected_negative_count_total": 3003,
        "d14_epoch_birth_positive_batch_count_total": 1685,
        "d14_epoch_birth_zero_positive_batch_count_total": 1585,
        "d14_epoch_birth_interval_fallback_count_total": 0,
        "d14_epoch_birth_prebirth_exposure_count_total": 180182,
        "d14_epoch_birth_interval_exposure_count_total": 5666,
        "d14_epoch_birth_selected_risk_logit_count_total": 185852,
        "d14_epoch_birth_postinterval_ignored_exposure_count_total": 4,
    }
    for name, expected in exact.items():
        if census[name] != expected:
            raise ValueError(
                f"D1.4 official-train census drifted: {name}="
                f"{census[name]} != {expected}"
            )
    if (
        census["d14_epoch_birth_positive_batch_count_total"]
        + census["d14_epoch_birth_zero_positive_batch_count_total"]
        != 3270
    ):
        raise ValueError("D1.4 positive/zero-positive batch census did not close")
    if (
        census["d14_epoch_birth_prebirth_group_count_total"]
        + census["d14_epoch_birth_zero_prebirth_group_count_total"]
        != 3003
    ):
        raise ValueError("D1.4 prebirth group census did not close")
    if census["d14_epoch_birth_negative_candidate_count_total"] <= 0:
        raise ValueError("D1.4 birth negative candidate pool was empty")

    normalized_count = census[
        "d14_epoch_birth_normalized_survival_event_count_total"
    ]
    positive_bags = census[
        "d14_epoch_birth_decision_aligned_positive_bag_count_total"
    ]
    negative_bags = census[
        "d14_epoch_birth_decision_aligned_negative_bag_count_total"
    ]
    if variant == "normalized_survival":
        if (normalized_count, positive_bags, negative_bags) != (3003, 0, 0):
            raise ValueError("D1.4 normalized-survival objective census drifted")
    elif (normalized_count, positive_bags, negative_bags) != (0, 3003, 3003):
        raise ValueError("D1.4 decision-aligned bag objective census drifted")

    hard_reject = float(
        metrics[
            "event_association_audit_class_argmax_reject_pair_count_unscaled"
        ]
    )
    admissible_mismatch = float(
        metrics[
            "event_association_audit_"
            "admissible_class_argmax_mismatch_pair_count_unscaled"
        ]
    )
    if (
        not math.isfinite(hard_reject)
        or hard_reject != 0
        or not math.isfinite(admissible_mismatch)
        or admissible_mismatch <= 0
    ):
        raise ValueError(
            "D1.4 drifted from the frozen soft-assignment intervention"
        )
    return {
        "census": census,
        "birth_objective_contract": VARIANT_OBJECTIVES[variant],
        "soft_assignment_wrong_argmax_path_observed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=tuple(VARIANT_OBJECTIVES))
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
        if metric_row.get("study_protocol") != "d14_mechanism":
            raise ValueError("D1.4 mechanism metric protocol drifted")
        if metric_row.get("event_d13_variant") != "combined":
            raise ValueError("D1.4 lost the frozen D1.3 combined contract")
        if metric_row.get("event_d14_variant") != args.variant:
            raise ValueError("D1.4 mechanism metric variant drifted")
        validate_effective_dose_metrics(metric_row["metrics"])
        trace_receipt = validate_effective_dose_trace(trace_path)
        mechanism_validation = validate_d14_mechanism_metrics(
            metric_row["metrics"], args.variant
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    checkpoint = result_dir / "terminal_epoch1.pth"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise SystemExit(f"terminal checkpoint is absent or empty: {checkpoint}")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    expected_checkpoint = {
        "epoch": 1,
        "study_protocol": "d14_mechanism",
        "model_variant": "eventmatr",
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "event_lifecycle_version": "d1_censored",
        "event_d1_lane": "th",
        "owner_state_count": 3,
        "birth_head": "independent_binary_hazard",
        "event_d13_variant": "combined",
        "event_d14_variant": args.variant,
        "birth_objective_contract": VARIANT_OBJECTIVES[args.variant],
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

    locked_test_sentinel = (
        output_root / "LOCKED_TEST_NOT_MOUNTED.pickle"
    ).resolve()
    if locked_test_sentinel.exists():
        raise SystemExit("locked-test sentinel unexpectedly exists")
    expected_options = {
        **FIXED_OPTIONS,
        "epochs": 1,
        "train_eval_step": 1,
        "study_protocol": "d14_mechanism",
        "model_variant": "eventmatr",
        "event_lifecycle_version": "d1_censored",
        "event_d1_lane": "th",
        "event_d13_variant": "combined",
        "event_d14_variant": args.variant,
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
                f"mechanism option {field} mismatch: "
                f"{options.get(field)!r} != {expected!r}"
            )
    if Path(options["video_feature_all_test"]).resolve() != locked_test_sentinel:
        raise SystemExit("D1.4 run did not retain the absent locked-test sentinel")
    if options.get("event_birth_logit_threshold") is not None:
        raise SystemExit("D1.4 mechanism used a fixed birth threshold")
    if options.get("event_end_logit_threshold") is not None:
        raise SystemExit("D1.4 mechanism used a fixed end threshold")

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
            raise SystemExit(f"D1.4 {phase} identity receipt is not PASS/clean")
        for field, expected in expected_identity.items():
            if identity.get(field) != expected:
                raise SystemExit(f"D1.4 {phase} identity {field} mismatch")
        smoke = identity.get("smoke")
        if (
            not isinstance(smoke, dict)
            or smoke.get("status") != "PASS"
            or smoke.get("test_access") is not False
        ):
            raise SystemExit(
                f"D1.4 {phase} identity lacks a passing no-test smoke gate"
            )
        source_identity_receipts[phase] = identity
    smoke = source_identity_receipts["final"]["smoke"]

    final_metrics = metric_lines[0]["metrics"]
    receipt = {
        "status": "PASS_TRAIN_MECHANISM_ONLY",
        "protocol": "eventmatr_d14_seed52_decision_alignment_mechanism_v1",
        "variant": args.variant,
        "contracts": {
            **COMMON_CONTRACTS,
            "birth_objective_contract": VARIANT_OBJECTIVES[args.variant],
        },
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
        "structure_gate_release": False,
        "locked_test_release": False,
        "source_identity": expected_identity,
        "source_identity_receipts": source_identity_receipts,
        "smoke_gate": smoke,
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
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(receipt_path)


if __name__ == "__main__":
    main()
