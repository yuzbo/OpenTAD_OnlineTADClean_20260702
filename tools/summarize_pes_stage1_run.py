#!/usr/bin/env python3
"""Create one machine-readable Stage-1 run row from an immutable ledger."""

import argparse
import json
import math
from pathlib import Path


def _load_json(value):
    if isinstance(value, dict):
        return value
    with Path(value).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def analyze_emission_errors(
    ground_truth,
    predictions,
    subset,
    tiou_threshold=0.5,
    latency_budget_sec=2.0,
    fps=30.0,
):
    from opentad.evaluations.online_budgeted_map import OnlineAPBudgeted

    prediction_data = _load_json(predictions)
    for video_rows in prediction_data.get("results", {}).values():
        for row in video_rows:
            if row.get("immutable") is not True:
                raise ValueError("Stage-1 diagnostics require immutable emission rows")
    evaluator = OnlineAPBudgeted(
        ground_truth_filename=_load_json(ground_truth),
        prediction_filename=prediction_data,
        subset=subset,
        tiou_thresholds=(tiou_threshold,),
        latency_budgets_sec=(latency_budget_sec,),
        fps=fps,
        require_ledger=False,
        require_no_future=True,
        include_identity_diagnostics=True,
        identity_tiou_threshold=tiou_threshold,
        identity_latency_budget_sec=latency_budget_sec,
    )
    identity = evaluator.evaluate()["identity_diagnostics"]
    return {
        "definition": identity["matching"],
        "num_ground_truth": identity["counts"]["ground_truth"],
        "num_predictions": identity["counts"]["emissions"],
        "true_positives": identity["counts"]["matched_ground_truth"],
        "false_negatives": identity["counts"]["unmatched_ground_truth"],
        "duplicate_false_positives": identity["counts"]["duplicate_emissions"],
        "fragmented_ground_truth": identity["fragmentation"][
            "fragmented_ground_truth_count"
        ],
        "duplicate_per_gt": identity["duplicate_per_gt"],
        "duplicate_fraction": identity["duplicate_fraction"],
        "fragmentation_rate": identity["fragmentation_rate"],
        "canonical_identity_diagnostics": identity,
    }


def checkpoint_slot_exhaustion(path):
    import torch

    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    values = [
        value
        for key, value in state_dict.items()
        if str(key).endswith("audit_slot_exhaustion_total")
    ]
    if len(values) != 1:
        raise ValueError("checkpoint must contain exactly one slot-exhaustion audit counter")
    return int(values[0].item())


def build_run_summary(
    annotation,
    ledger,
    subset,
    variant,
    seed,
    gpu_hours,
    slot_exhaustion,
    tiou_thresholds,
    latency_budgets_sec,
    diagnostic_tiou=0.5,
    diagnostic_budget_sec=2.0,
):
    from opentad.evaluations.online_budgeted_map import OnlineAPBudgeted

    prediction_data = _load_json(ledger)
    evaluator = OnlineAPBudgeted(
        ground_truth_filename=annotation,
        prediction_filename=prediction_data,
        subset=subset,
        tiou_thresholds=tiou_thresholds,
        latency_budgets_sec=latency_budgets_sec,
        require_ledger=True,
        require_no_future=True,
        identity_tiou_threshold=diagnostic_tiou,
        identity_latency_budget_sec=diagnostic_budget_sec,
    )
    metrics = evaluator.evaluate()
    errors = metrics["identity_diagnostics"]
    if not math.isfinite(float(gpu_hours)) or float(gpu_hours) < 0:
        raise ValueError("gpu_hours must be finite and non-negative")
    return {
        "variant": str(variant),
        "seed": int(seed),
        "average_mOnlineAP": metrics["average_mOnlineAP"],
        "duplicate_per_gt": errors["duplicate_per_gt"],
        "duplicate_fraction": errors["duplicate_fraction"],
        "fragmentation_rate": errors["fragmentation_rate"],
        "gpu_hours": float(gpu_hours),
        "protocol_violations": 0,
        "slot_exhaustion": int(slot_exhaustion),
        "online_metrics": metrics,
        "error_diagnostics": errors,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--variant", choices=("fresh", "trackformer", "persistent"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-hours", type=float, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--slot-exhaustion", type=int)
    parser.add_argument("--tiou-thresholds", type=float, nargs="+", default=(0.3, 0.4, 0.5, 0.6, 0.7))
    parser.add_argument("--latency-budgets-sec", type=float, nargs="+", default=(0.5, 1.0, 2.0, 4.0))
    parser.add_argument("--diagnostic-tiou", type=float, default=0.5)
    parser.add_argument("--diagnostic-budget-sec", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.checkpoint is None) == (args.slot_exhaustion is None):
        raise SystemExit("provide exactly one of --checkpoint or --slot-exhaustion")
    slot_exhaustion = (
        checkpoint_slot_exhaustion(args.checkpoint)
        if args.checkpoint is not None
        else args.slot_exhaustion
    )
    summary = build_run_summary(
        annotation=args.annotation,
        ledger=args.ledger,
        subset=args.subset,
        variant=args.variant,
        seed=args.seed,
        gpu_hours=args.gpu_hours,
        slot_exhaustion=slot_exhaustion,
        tiou_thresholds=args.tiou_thresholds,
        latency_budgets_sec=args.latency_budgets_sec,
        diagnostic_tiou=args.diagnostic_tiou,
        diagnostic_budget_sec=args.diagnostic_budget_sec,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PES_RUN_SUMMARY={output}")


if __name__ == "__main__":
    main()
