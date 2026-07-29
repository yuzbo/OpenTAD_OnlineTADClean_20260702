"""Validate the single registered D1.1 train-only mechanism run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

try:
    from scripts.finalize_eventmatr_d1_pilot import (
        FIXED_OPTIONS,
        validate_metric_lines,
    )
except ModuleNotFoundError:
    from finalize_eventmatr_d1_pilot import (
        FIXED_OPTIONS,
        validate_metric_lines,
    )


CHECKPOINT_SCHEMA = "eventmatr_d11_ternary_owner_v1"
POSITIVE_MECHANISM_METRICS = (
    "event_transition_gradient_norm",
    "event_owner_gradient_norm",
    "event_birth_positive_count_unscaled",
    "event_end_positive_count_unscaled",
    "event_owner_assignment_count_unscaled",
    "event_ragged_track_count_unscaled",
    "event_false_track_cancel_group_count_unscaled",
    "event_source_predicted_associated_row_count_unscaled",
    "event_source_predicted_unmatched_row_count_unscaled",
    "event_source_teacher_birth_row_count_unscaled",
    "event_association_predicted_associated_count_unscaled",
)


def validate_mechanism_metrics(metrics: dict) -> None:
    missing = sorted(
        set(POSITIVE_MECHANISM_METRICS).difference(metrics)
        | {"event_runtime_capacity_exhaustions_unscaled"}.difference(metrics)
    )
    if missing:
        raise ValueError(f"D1.1 mechanism metrics are missing keys: {missing}")
    for name in POSITIVE_MECHANISM_METRICS:
        value = float(metrics[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"D1.1 mechanism metric is not live: {name}={value}")
    capacity = float(metrics["event_runtime_capacity_exhaustions_unscaled"])
    if not math.isfinite(capacity) or capacity != 0:
        raise ValueError(
            "D1.1 mechanism exhausted dynamic capacity: "
            f"event_runtime_capacity_exhaustions_unscaled={capacity}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    result_dirs = sorted((output_root / "result").glob("*__TH__*"))
    if len(result_dirs) != 1:
        raise SystemExit(f"expected exactly one TH result directory, got {result_dirs}")
    result_dir = result_dirs[0]
    options = json.loads((result_dir / "opts.json").read_text(encoding="utf-8"))
    metric_lines = [
        json.loads(line)
        for line in (result_dir / "mechanism_epoch_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    try:
        validate_metric_lines(metric_lines, "TH", 1)
        if metric_lines[0].get("study_protocol") != "d11_mechanism":
            raise ValueError("D1.1 mechanism metric protocol drifted")
        validate_mechanism_metrics(metric_lines[0]["metrics"])
    except ValueError as error:
        raise SystemExit(str(error)) from error

    checkpoint = result_dir / "terminal_epoch1.pth"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise SystemExit(f"terminal checkpoint is absent or empty: {checkpoint}")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    expected_checkpoint = {
        "epoch": 1,
        "study_protocol": "d11_mechanism",
        "model_variant": "eventmatr",
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "event_lifecycle_version": "d1_censored",
        "event_d1_lane": "th",
        "owner_state_count": 3,
    }
    for field, value in expected_checkpoint.items():
        if checkpoint_payload.get(field) != value:
            raise SystemExit(
                f"terminal checkpoint {field} mismatch: "
                f"{checkpoint_payload.get(field)!r} != {value!r}"
            )
    if not isinstance(checkpoint_payload.get("state_dict"), dict) or not checkpoint_payload[
        "state_dict"
    ]:
        raise SystemExit("terminal checkpoint has no model state")

    locked_test_sentinel = (output_root / "LOCKED_TEST_NOT_MOUNTED.pickle").resolve()
    if locked_test_sentinel.exists():
        raise SystemExit("locked-test sentinel unexpectedly exists")
    expected_options = {
        **FIXED_OPTIONS,
        "epochs": 1,
        "train_eval_step": 1,
        "study_protocol": "d11_mechanism",
        "model_variant": "eventmatr",
        "event_lifecycle_version": "d1_censored",
        "event_d1_lane": "th",
        "birth_mode": "instant_transition",
        "ownership_mode": "sticky_owner",
        "event_arm": "b1o1",
        "event_teacher_forcing_ratio": 0.5,
        "event_resource_limit": 0,
        "reduce": 1,
        "make_output": True,
        "load_model": False,
    }
    for field, value in expected_options.items():
        if options.get(field) != value:
            raise SystemExit(
                f"mechanism option {field} mismatch: "
                f"{options.get(field)!r} != {value!r}"
            )
    if Path(options["video_feature_all_test"]).resolve() != locked_test_sentinel:
        raise SystemExit("mechanism run did not retain the absent locked-test sentinel")
    if options.get("event_birth_logit_threshold") is not None:
        raise SystemExit("mechanism run used a fixed birth threshold")
    if options.get("event_end_logit_threshold") is not None:
        raise SystemExit("mechanism run used a fixed end threshold")

    identity = json.loads(
        (output_root / "source_identity_start.json").read_text(encoding="utf-8")
    )
    expected_identity = {
        "commit": args.source_commit,
        "tree": args.source_tree,
        "manifest_sha256": args.manifest_sha256,
    }
    if identity.get("status") != "PASS" or identity.get("clean") is not True:
        raise SystemExit("mechanism start identity receipt is not PASS/clean")
    for field, value in expected_identity.items():
        if identity.get(field) != value:
            raise SystemExit(f"mechanism start identity {field} mismatch")
    smoke = identity.get("smoke")
    if (
        not isinstance(smoke, dict)
        or smoke.get("status") != "PASS"
        or smoke.get("test_access") is not False
    ):
        raise SystemExit("mechanism start identity lacks a passing no-test smoke gate")

    final_metrics = metric_lines[0]["metrics"]
    source_metrics = {
        key: value
        for key, value in final_metrics.items()
        if key.startswith("event_source_") or key.startswith("event_association_")
    }
    runtime_metrics = {
        key: value
        for key, value in final_metrics.items()
        if key.startswith("event_runtime_")
    }
    receipt = {
        "status": "PASS",
        "protocol": "eventmatr_d11_seed52_one_epoch_mechanism_v1",
        "lane": "TH",
        "epochs": 1,
        "seed": 52,
        "fresh_start": True,
        "test_access": False,
        "checkpoint_updated": True,
        "strict_causal_paper_result_valid": False,
        "train_prefix_metrics_diagnostic_only": True,
        "performance_gate_applied": False,
        "five_epoch_contract_revision_only": True,
        "existing_five_epoch_matrix_release": False,
        "source_identity": expected_identity,
        "smoke_gate": smoke,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            **expected_checkpoint,
        },
        "mechanism_liveness": {
            name: final_metrics[name] for name in POSITIVE_MECHANISM_METRICS
        },
        "runtime_metrics": runtime_metrics,
        "source_metrics": source_metrics,
        "final_epoch_metrics": final_metrics,
    }
    receipt_path = output_root / "mechanism_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(receipt_path)


if __name__ == "__main__":
    main()
