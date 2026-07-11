#!/usr/bin/env python3
"""Create one machine-readable Stage-1 run row from an immutable ledger."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def _load_json(value):
    if isinstance(value, dict):
        return value
    with Path(value).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _segment_iou(left, right):
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def _union_coverage(segments, target):
    clipped = sorted(
        (max(segment[0], target[0]), min(segment[1], target[1]))
        for segment in segments
        if min(segment[1], target[1]) > max(segment[0], target[0])
    )
    if not clipped:
        return 0.0
    merged = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    covered = sum(end - start for start, end in merged)
    return covered / max(target[1] - target[0], 1e-12)


def _row_time(row, prefix):
    for key in (f"{prefix}_time_sec", f"{prefix}_sec"):
        if key in row:
            return float(row[key])
    frame_key = f"{prefix}_frame"
    if frame_key in row and float(row.get("fps", 0.0)) > 0:
        return float(row[frame_key]) / float(row["fps"])
    return None


def analyze_emission_errors(
    ground_truth,
    predictions,
    subset,
    tiou_threshold=0.5,
    latency_budget_sec=2.0,
):
    ground_truth = _load_json(ground_truth)
    predictions = _load_json(predictions)
    targets = []
    seen_targets = set()
    for video_id, video in ground_truth.get("database", {}).items():
        if video.get("subset") != subset:
            continue
        for annotation in video.get("annotations", ()):
            segment = tuple(float(value) for value in annotation["segment"])
            if len(segment) != 2 or segment[1] <= segment[0]:
                raise ValueError(f"invalid ground-truth segment for {video_id}: {segment}")
            key = (str(video_id), str(annotation["label"]), segment)
            if key in seen_targets:
                continue
            seen_targets.add(key)
            targets.append(
                {
                    "id": len(targets),
                    "video_id": str(video_id),
                    "label": str(annotation["label"]),
                    "segment": segment,
                }
            )

    rows = []
    for video_id, video_rows in predictions.get("results", {}).items():
        for raw in video_rows:
            if not bool(raw.get("immutable", False)):
                raise ValueError("Stage-1 diagnostics require immutable emission rows")
            segment = tuple(float(value) for value in raw["segment"])
            if len(segment) != 2 or segment[1] <= segment[0]:
                raise ValueError(f"invalid prediction segment for {video_id}: {segment}")
            emit_time = _row_time(raw, "emit")
            source_time = _row_time(raw, "source")
            if emit_time is None or source_time is None:
                raise ValueError("emission rows require emit and source provenance")
            if segment[1] > emit_time + 1e-9:
                raise ValueError("prediction endpoint uses future time")
            if source_time > emit_time + 1e-9:
                raise ValueError("prediction source provenance uses future time")
            rows.append(
                {
                    "id": len(rows),
                    "video_id": str(video_id),
                    "label": str(raw["label"]),
                    "score": float(raw["score"]),
                    "segment": segment,
                    "emit_time": emit_time,
                }
            )

    targets_by_group = defaultdict(list)
    rows_by_group = defaultdict(list)
    for target in targets:
        targets_by_group[(target["video_id"], target["label"])].append(target)
    for row in rows:
        rows_by_group[(row["video_id"], row["label"])].append(row)

    locked = set()
    duplicate_false_positives = 0
    for group, group_rows in rows_by_group.items():
        group_targets = targets_by_group.get(group, ())
        for row in sorted(group_rows, key=lambda item: (-item["score"], item["id"])):
            eligible = []
            for target in group_targets:
                latency = row["emit_time"] - target["segment"][1]
                iou = _segment_iou(row["segment"], target["segment"])
                if iou >= tiou_threshold and -1e-9 <= latency <= latency_budget_sec + 1e-9:
                    eligible.append((iou, -abs(latency), target))
            available = [item for item in eligible if item[2]["id"] not in locked]
            if available:
                target = max(available, key=lambda item: (item[0], item[1]))[2]
                locked.add(target["id"])
            elif eligible:
                duplicate_false_positives += 1

    fragmented = 0
    for target in targets:
        if target["id"] in locked:
            continue
        candidates = []
        for row in rows_by_group.get((target["video_id"], target["label"]), ()):
            latency = row["emit_time"] - target["segment"][1]
            iou = _segment_iou(row["segment"], target["segment"])
            if -1e-9 <= latency <= latency_budget_sec + 1e-9 and 0.0 < iou < tiou_threshold:
                candidates.append(row["segment"])
        if len(candidates) >= 2 and _union_coverage(candidates, target["segment"]) >= tiou_threshold:
            fragmented += 1

    num_predictions = len(rows)
    num_targets = len(targets)
    return {
        "definition": {
            "tiou_threshold": float(tiou_threshold),
            "latency_budget_sec": float(latency_budget_sec),
            "duplicate": "extra ranked emission eligible for an already matched GT",
            "fragmentation": (
                "unmatched GT covered above threshold by at least two timely same-class pieces, "
                "with no piece individually reaching the tIoU threshold"
            ),
        },
        "num_ground_truth": num_targets,
        "num_predictions": num_predictions,
        "true_positives": len(locked),
        "false_negatives": num_targets - len(locked),
        "duplicate_false_positives": duplicate_false_positives,
        "fragmented_ground_truth": fragmented,
        "duplicate_rate": duplicate_false_positives / num_predictions if num_predictions else 0.0,
        "fragmentation_rate": fragmented / num_targets if num_targets else 0.0,
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
    )
    metrics = evaluator.evaluate()
    errors = analyze_emission_errors(
        annotation,
        prediction_data,
        subset=subset,
        tiou_threshold=diagnostic_tiou,
        latency_budget_sec=diagnostic_budget_sec,
    )
    if not math.isfinite(float(gpu_hours)) or float(gpu_hours) < 0:
        raise ValueError("gpu_hours must be finite and non-negative")
    return {
        "variant": str(variant),
        "seed": int(seed),
        "average_mOnlineAP": metrics["average_mOnlineAP"],
        "duplicate_rate": errors["duplicate_rate"],
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
