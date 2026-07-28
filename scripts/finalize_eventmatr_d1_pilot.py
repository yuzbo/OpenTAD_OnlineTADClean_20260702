"""Validate one train-only D1 pilot and write a compact scientific receipt."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch


COMMON_METRICS = {
    "loss",
    "loss_cls",
    "loss_flag",
    "loss_reg_l1",
    "loss_reg_diou",
    "loss_reg_stcls",
    "lr",
    "mAP_train",
    "mAP_03_train",
    "mAP_04_train",
    "mAP_05_train",
    "mAP_06_train",
    "mAP_07_train",
}
D1_METRICS = {
    "loss_event_birth",
    "loss_event_owner_state",
    "event_birth_positive_count_unscaled",
    "event_end_positive_count_unscaled",
    "event_owner_assignment_count_unscaled",
    "event_ragged_track_count_unscaled",
}
FIXED_OPTIONS = {
    "mode": "train",
    "rgb": True,
    "flow": True,
    "feat_dim": 4096,
    "hidden_dim": 1024,
    "ffn_dim": 2048,
    "enc_layers": 3,
    "dec_layers": 5,
    "num_frame": 64,
    "num_queries": 10,
    "p_videos": 1,
    "detect_len": 16,
    "anti_len": 16,
    "max_memory_len": 7,
    "memory_sampler": "gap2",
    "batch": 64,
    "min_lr": 1e-8,
    "max_lr": 1e-5,
    "weight_decay": 1e-4,
    "lr_Tup": 3,
    "lr_Tcycle": 10,
    "lr_gamma": 0.9,
    "random_seed": 52,
    "use_focal": True,
    "use_flag": True,
    "flag_threshold": 0.5,
    "cls_threshold": 0.1,
    "nms_threshold": 0.3,
}


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def validate_metric_lines(metric_lines: list[dict], lane: str, epochs: int) -> None:
    if [item.get("epoch") for item in metric_lines] != list(
        range(1, epochs + 1)
    ):
        raise ValueError("pilot epoch metrics are incomplete or non-chronological")
    required = set(COMMON_METRICS)
    if lane != "N":
        required.update(D1_METRICS)
    for item in metric_lines:
        metrics = item.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("pilot metrics must be a non-empty dictionary")
        missing = sorted(required.difference(metrics))
        if missing:
            raise ValueError(f"pilot metrics are missing required keys: {missing}")
        if item.get("test_access") is not False:
            raise ValueError("pilot epoch did not certify test_access=false")
        if item.get("strict_causal_paper_result_valid") is not False:
            raise ValueError("pilot epoch was incorrectly marked paper-valid")
    if not _finite(metric_lines):
        raise ValueError("pilot metrics contain a non-finite value")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--lane", required=True, choices=("N", "R", "T", "H", "TH"))
    parser.add_argument("--epochs", required=True, choices=(5, 10, 20), type=int)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    result_dirs = sorted((output_root / "result").glob(f"*__{args.lane}__*"))
    if len(result_dirs) != 1:
        raise SystemExit(f"expected exactly one result directory, got {result_dirs}")
    result_dir = result_dirs[0]
    options = json.loads((result_dir / "opts.json").read_text(encoding="utf-8"))
    metric_lines = [
        json.loads(line)
        for line in (result_dir / "pilot_epoch_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    try:
        validate_metric_lines(metric_lines, args.lane, args.epochs)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    checkpoint = result_dir / f"terminal_epoch{args.epochs}.pth"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise SystemExit(f"terminal checkpoint is absent or empty: {checkpoint}")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    if checkpoint_payload.get("epoch") != args.epochs:
        raise SystemExit("terminal checkpoint epoch mismatch")
    if checkpoint_payload.get("study_protocol") != "d1_preexperiment":
        raise SystemExit("terminal checkpoint protocol mismatch")
    if not isinstance(checkpoint_payload.get("state_dict"), dict) or not checkpoint_payload[
        "state_dict"
    ]:
        raise SystemExit("terminal checkpoint has no model state")
    locked_test_sentinel = (output_root / "LOCKED_TEST_NOT_MOUNTED.pickle").resolve()
    if locked_test_sentinel.exists():
        raise SystemExit("locked-test sentinel unexpectedly exists")
    if Path(options["video_feature_all_test"]).resolve() != locked_test_sentinel:
        raise SystemExit("pilot did not retain the absent locked-test sentinel")
    if options.get("study_protocol") != "d1_preexperiment":
        raise SystemExit("pilot study protocol drifted")
    if options.get("epochs") != args.epochs:
        raise SystemExit("pilot seed or horizon drifted")
    if options.get("train_eval_step") != args.epochs:
        raise SystemExit("pilot terminal train-evaluation step drifted")
    for field, value in FIXED_OPTIONS.items():
        if options.get(field) != value:
            raise SystemExit(
                f"pilot option {field} mismatch: {options.get(field)!r} != {value!r}"
            )

    if args.lane == "N":
        expected = {
            "model_variant": "native_matr",
            "event_lifecycle_version": "v1_dense",
        }
    else:
        expected = {
            "model_variant": "eventmatr",
            "event_lifecycle_version": "d1_censored",
            "event_d1_lane": args.lane.lower(),
        }
    for field, value in expected.items():
        if options.get(field) != value:
            raise SystemExit(
                f"pilot option {field} mismatch: {options.get(field)!r} != {value!r}"
            )
    if checkpoint_payload.get("model_variant") != expected["model_variant"]:
        raise SystemExit("terminal checkpoint model variant mismatch")

    identity_path = output_root / "source_identity_start.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    expected_identity = {
        "commit": args.source_commit,
        "tree": args.source_tree,
        "manifest_sha256": args.manifest_sha256,
    }
    if identity.get("status") != "PASS" or identity.get("clean") is not True:
        raise SystemExit("pilot start identity receipt is not PASS/clean")
    for field, value in expected_identity.items():
        if identity.get(field) != value:
            raise SystemExit(f"pilot start identity {field} mismatch")

    receipt = {
        "status": "PASS",
        "protocol": "eventmatr_d1_seed52_short_pilot_v1",
        "lane": args.lane,
        "epochs": args.epochs,
        "seed": 52,
        "test_access": False,
        "checkpoint_updated": True,
        "strict_causal_paper_result_valid": False,
        "train_prefix_metrics_diagnostic_only": True,
        "source_identity": expected_identity,
        "result_dir": str(result_dir),
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "final_epoch_metrics": metric_lines[-1]["metrics"],
        "metric_epochs": len(metric_lines),
    }
    receipt_path = output_root / "pilot_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(receipt_path)


if __name__ == "__main__":
    main()
