"""Validate one prospective D1.3 train-only factorial mechanism run."""

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
    from finalize_eventmatr_d1_pilot import (
        FIXED_OPTIONS,
        validate_metric_lines,
    )
    from finalize_eventmatr_d11_mechanism import (
        EFFECTIVE_DOSE_METRICS,
        validate_effective_dose_metrics,
        validate_effective_dose_trace,
    )


CHECKPOINT_SCHEMA = "eventmatr_d13_factorial_mechanism_v1"
VARIANT_CONTRACTS = {
    "soft_assignment_only": {
        "association_contract": "soft_target_class_log_probability_v1",
        "birth_risk_contract": "prefix_hard_negative_v1",
    },
    "event_matched_birth_only": {
        "association_contract": "hard_class_argmax_gate_v1",
        "birth_risk_contract": "event_matched_hard_negative_v1",
    },
    "combined": {
        "association_contract": "soft_target_class_log_probability_v1",
        "birth_risk_contract": "event_matched_hard_negative_v1",
    },
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
CENSUS_KEYS = (
    "d13_epoch_birth_positive_count_total",
    "d13_epoch_birth_selected_negative_count_total",
    "d13_epoch_birth_negative_candidate_count_total",
    "d13_epoch_birth_positive_batch_count_total",
    "d13_epoch_birth_zero_positive_batch_count_total",
    "d13_epoch_birth_interval_fallback_count_total",
    "d13_epoch_birth_prebirth_exposure_count_total",
    "d13_epoch_birth_interval_exposure_count_total",
    "d13_epoch_birth_selected_risk_logit_count_total",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_d13_mechanism_metrics(metrics: dict, variant: str) -> dict:
    missing = sorted(
        set(COMMON_POSITIVE_METRICS)
        .union(CENSUS_KEYS)
        .union(
            {
                "d13_epoch_physical_batch_count",
                "event_runtime_capacity_exhaustions_unscaled",
                "event_association_audit_class_argmax_reject_pair_count_unscaled",
                (
                    "event_association_audit_"
                    "admissible_class_argmax_mismatch_pair_count_unscaled"
                ),
                "event_association_predicted_associated_count_unscaled",
            }
        )
        .difference(metrics)
    )
    if missing:
        raise ValueError(f"D1.3 mechanism metrics are missing keys: {missing}")
    for name in COMMON_POSITIVE_METRICS:
        value = float(metrics[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"D1.3 mechanism metric is not live: {name}={value}")
    capacity = float(metrics["event_runtime_capacity_exhaustions_unscaled"])
    if not math.isfinite(capacity) or capacity != 0:
        raise ValueError(f"D1.3 mechanism exhausted dynamic capacity: {capacity}")

    census = {}
    for name in CENSUS_KEYS:
        value = float(metrics[name])
        if not math.isfinite(value) or not value.is_integer() or value < 0:
            raise ValueError(
                f"D1.3 census metric must be a finite non-negative integer: "
                f"{name}={value}"
            )
        census[name] = int(value)
    if float(metrics["d13_epoch_physical_batch_count"]) != 3270.0:
        raise ValueError("D1.3 physical batch count did not close at 3270")
    exact = {
        "d13_epoch_birth_positive_count_total": 3003,
        "d13_epoch_birth_positive_batch_count_total": 1685,
        "d13_epoch_birth_zero_positive_batch_count_total": 1585,
    }
    for name, expected in exact.items():
        if census[name] != expected:
            raise ValueError(
                f"D1.3 official-train census drifted: {name}={census[name]} "
                f"!= {expected}"
            )
    if (
        census["d13_epoch_birth_positive_batch_count_total"]
        + census["d13_epoch_birth_zero_positive_batch_count_total"]
        != 3270
    ):
        raise ValueError("D1.3 positive/zero-positive batch census did not close")
    if census["d13_epoch_birth_negative_candidate_count_total"] <= 0:
        raise ValueError("D1.3 birth negative candidate pool was empty")
    if census["d13_epoch_birth_selected_risk_logit_count_total"] < 3003:
        raise ValueError("D1.3 positive risk exposure count is incomplete")
    event_matched = (
        VARIANT_CONTRACTS[variant]["birth_risk_contract"]
        == "event_matched_hard_negative_v1"
    )
    expected_negative_count = 3003 if event_matched else 203363
    if census["d13_epoch_birth_selected_negative_count_total"] != expected_negative_count:
        raise ValueError(
            "D1.3 selected birth-negative count drifted: "
            f"{census['d13_epoch_birth_selected_negative_count_total']} "
            f"!= {expected_negative_count}"
        )

    soft_assignment = (
        VARIANT_CONTRACTS[variant]["association_contract"]
        == "soft_target_class_log_probability_v1"
    )
    associated = float(
        metrics["event_association_predicted_associated_count_unscaled"]
    )
    admissible_mismatch = float(
        metrics[
            "event_association_audit_"
            "admissible_class_argmax_mismatch_pair_count_unscaled"
        ]
    )
    hard_reject = float(
        metrics[
            "event_association_audit_class_argmax_reject_pair_count_unscaled"
        ]
    )
    for name, value in (
        ("predicted association", associated),
        ("admissible class-argmax mismatch", admissible_mismatch),
        ("hard class-argmax rejection", hard_reject),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"D1.3 {name} metric is invalid: {value}")
    if soft_assignment:
        if associated <= 0 or admissible_mismatch <= 0:
            raise ValueError(
                "D1.3 soft assignment did not create a real "
                "wrong-argmax admissible/associated path"
            )
        if hard_reject != 0:
            raise ValueError("D1.3 soft assignment still applied a hard class gate")
    elif hard_reject <= 0:
        raise ValueError("D1.3 hard-gate control observed no class rejection")
    return {
        "event_matched_birth": event_matched,
        "soft_assignment": soft_assignment,
        "census": census,
        "predicted_association_metric": associated,
        "admissible_class_argmax_mismatch_metric": admissible_mismatch,
        "class_argmax_reject_metric": hard_reject,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=tuple(VARIANT_CONTRACTS))
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
        if metric_row.get("study_protocol") != "d13_mechanism":
            raise ValueError("D1.3 mechanism metric protocol drifted")
        if metric_row.get("event_d13_variant") != args.variant:
            raise ValueError("D1.3 mechanism metric variant drifted")
        validate_effective_dose_metrics(metric_row["metrics"])
        trace_receipt = validate_effective_dose_trace(trace_path)
        mechanism_validation = validate_d13_mechanism_metrics(
            metric_row["metrics"], args.variant
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    contracts = VARIANT_CONTRACTS[args.variant]
    checkpoint = result_dir / "terminal_epoch1.pth"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise SystemExit(f"terminal checkpoint is absent or empty: {checkpoint}")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    expected_checkpoint = {
        "epoch": 1,
        "study_protocol": "d13_mechanism",
        "model_variant": "eventmatr",
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "event_lifecycle_version": "d1_censored",
        "event_d1_lane": "th",
        "owner_state_count": 3,
        "birth_head": "independent_binary_hazard",
        "event_d13_variant": args.variant,
        **contracts,
    }
    for field, expected in expected_checkpoint.items():
        if checkpoint_payload.get(field) != expected:
            raise SystemExit(
                f"terminal checkpoint {field} mismatch: "
                f"{checkpoint_payload.get(field)!r} != {expected!r}"
            )
    if not checkpoint_payload.get("state_dict"):
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
        "study_protocol": "d13_mechanism",
        "model_variant": "eventmatr",
        "event_lifecycle_version": "d1_censored",
        "event_d1_lane": "th",
        "event_d13_variant": args.variant,
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
        raise SystemExit("D1.3 run did not retain the absent locked-test sentinel")
    if options.get("event_birth_logit_threshold") is not None:
        raise SystemExit("D1.3 mechanism used a fixed birth threshold")
    if options.get("event_end_logit_threshold") is not None:
        raise SystemExit("D1.3 mechanism used a fixed end threshold")

    identity = json.loads(
        (output_root / "source_identity_start.json").read_text(encoding="utf-8")
    )
    expected_identity = {
        "commit": args.source_commit,
        "tree": args.source_tree,
        "manifest_sha256": args.manifest_sha256,
    }
    if identity.get("status") != "PASS" or identity.get("clean") is not True:
        raise SystemExit("D1.3 start identity receipt is not PASS/clean")
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise SystemExit(f"D1.3 start identity {field} mismatch")
    smoke = identity.get("smoke")
    if (
        not isinstance(smoke, dict)
        or smoke.get("status") != "PASS"
        or smoke.get("test_access") is not False
    ):
        raise SystemExit("D1.3 start identity lacks a passing no-test smoke gate")

    final_metrics = metric_lines[0]["metrics"]
    receipt = {
        "status": "PASS_TRAIN_MECHANISM_ONLY",
        "protocol": "eventmatr_d13_seed52_factorial_mechanism_v1",
        "variant": args.variant,
        "contracts": contracts,
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
        "official_comparison_release": False,
        "locked_test_release": False,
        "source_identity": expected_identity,
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
