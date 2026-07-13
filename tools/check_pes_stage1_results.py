#!/usr/bin/env python3
"""Apply the registered Stage-1 persistent-event-set mechanism gate."""

import argparse
import json
import math
from pathlib import Path


VARIANTS = ("fresh", "trackformer", "persistent")
REQUIRED_METRICS = (
    "average_mOnlineAP",
    "duplicate_per_gt",
    "fragmentation_rate",
    "gpu_hours",
    "protocol_violations",
    "slot_exhaustion",
)


def _mean(values):
    return sum(values) / len(values)


def _validate_number(row, key, minimum=None):
    value = float(row[key])
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"invalid {key} for {row.get('variant')}/{row.get('seed')}: {value}")
    return value


def evaluate_gate(
    report,
    map_gain_threshold_points=2.0,
    error_reduction_threshold=0.20,
    map_parity_tolerance_points=0.5,
):
    rows = list(report.get("runs", ()))
    if not rows:
        raise ValueError("result report must contain non-empty runs")
    grouped = {variant: {} for variant in VARIANTS}
    for row in rows:
        variant = str(row.get("variant"))
        if variant not in grouped:
            raise ValueError(f"unexpected Stage-1 variant: {variant}")
        missing = [key for key in REQUIRED_METRICS if key not in row]
        if missing:
            raise ValueError(f"run is missing required metrics: {missing}")
        seed = int(row["seed"])
        if seed in grouped[variant]:
            raise ValueError(f"duplicate run for {variant} seed {seed}")
        normalized = dict(row)
        for key in REQUIRED_METRICS:
            normalized[key] = _validate_number(row, key, minimum=0.0)
        grouped[variant][seed] = normalized

    seed_sets = {variant: set(seed_rows) for variant, seed_rows in grouped.items()}
    if any(not seeds for seeds in seed_sets.values()) or len({frozenset(v) for v in seed_sets.values()}) != 1:
        raise ValueError(f"Stage-1 variants require matched seed sets: {seed_sets}")
    seeds = sorted(next(iter(seed_sets.values())))

    aggregate = {}
    for variant in VARIANTS:
        variant_rows = [grouped[variant][seed] for seed in seeds]
        aggregate[variant] = {
            key: _mean([row[key] for row in variant_rows]) for key in REQUIRED_METRICS
        }

    total_gpu_hours = sum(row["gpu_hours"] for rows_by_seed in grouped.values() for row in rows_by_seed.values())
    hard_budget = float(report.get("hard_budget_gpu_hours", 10.0))
    invalid_reasons = []
    if total_gpu_hours > hard_budget + 1e-9:
        invalid_reasons.append(
            f"GPU-hour budget exceeded: {total_gpu_hours:.3f} > {hard_budget:.3f}"
        )
    protocol_violations = sum(
        row["protocol_violations"] for rows_by_seed in grouped.values() for row in rows_by_seed.values()
    )
    if protocol_violations:
        invalid_reasons.append(f"protocol violations observed: {protocol_violations:g}")
    slot_exhaustion = sum(
        row["slot_exhaustion"] for rows_by_seed in grouped.values() for row in rows_by_seed.values()
    )
    if slot_exhaustion:
        invalid_reasons.append(f"slot exhaustion observed: {slot_exhaustion:g}")

    persistent = aggregate["persistent"]
    comparisons = {}
    for baseline_name in ("fresh", "trackformer"):
        baseline = aggregate[baseline_name]
        map_gain_points = 100.0 * (
            persistent["average_mOnlineAP"] - baseline["average_mOnlineAP"]
        )
        error_reductions = {}
        for key in ("duplicate_per_gt", "fragmentation_rate"):
            error_reductions[key] = (
                (baseline[key] - persistent[key]) / baseline[key]
                if baseline[key] > 0
                else 0.0
            )
        best_error_reduction = max(error_reductions.values())
        map_pass = map_gain_points >= float(map_gain_threshold_points)
        error_pass = (
            best_error_reduction >= float(error_reduction_threshold)
            and map_gain_points >= -float(map_parity_tolerance_points)
        )
        comparisons[baseline_name] = {
            "map_gain_points": map_gain_points,
            "error_reduction_fraction": error_reductions,
            "best_error_reduction_fraction": best_error_reduction,
            "map_threshold_pass": map_pass,
            "error_threshold_at_map_parity_pass": error_pass,
            "passes": bool(map_pass or error_pass),
        }

    if invalid_reasons:
        status = "INVALID"
    elif all(comparison["passes"] for comparison in comparisons.values()):
        status = "PASS_MECHANISM_GATE"
    else:
        status = "KILL_OR_REVISE"
    return {
        "status": status,
        "seeds": seeds,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "thresholds": {
            "map_gain_points": float(map_gain_threshold_points),
            "error_reduction_fraction": float(error_reduction_threshold),
            "map_parity_tolerance_points": float(map_parity_tolerance_points),
            "hard_budget_gpu_hours": hard_budget,
        },
        "total_gpu_hours": total_gpu_hours,
        "invalid_reasons": invalid_reasons,
        "needs_five_seed_confirmation": True,
        "raw_video_training_allowed": False,
        "interpretation": (
            "A pass only retains the feature-level mechanism for novelty review and five-seed "
            "confirmation; it is not a final effectiveness or novelty claim."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.report.open("r", encoding="utf-8") as file:
        verdict = evaluate_gate(json.load(file))
    serialized = json.dumps(verdict, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    raise SystemExit({"PASS_MECHANISM_GATE": 0, "KILL_OR_REVISE": 2, "INVALID": 3}[verdict["status"]])


if __name__ == "__main__":
    main()
