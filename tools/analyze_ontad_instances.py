#!/usr/bin/env python3
"""Audit instance concurrency and duration statistics in On-TAD annotations."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def _percentile(values, percentile):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    rank = (len(values) - 1) * float(percentile) / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _max_concurrency(annotations, same_class=False):
    events = []
    for annotation in annotations:
        start, end = (float(value) for value in annotation["segment"])
        label = str(annotation["label"])
        events.append((start, 1, label))
        events.append((end, -1, label))
    # End events precede starts at the same timestamp: touching intervals do not overlap.
    events.sort(key=lambda item: (item[0], item[1]))
    if same_class:
        active = defaultdict(int)
        maximum = 0
        for _, delta, label in events:
            active[label] += delta
            maximum = max(maximum, active[label])
        return maximum
    active = 0
    maximum = 0
    for _, delta, _ in events:
        active += delta
        maximum = max(maximum, active)
    return maximum


def analyze_database(data, subsets=None):
    if "database" not in data:
        raise ValueError("annotation file must contain a database field")
    allowed = None if not subsets else {str(value) for value in subsets}
    per_video = []
    durations = []
    labels = Counter()
    invalid = []
    for video_id, video in sorted(data["database"].items()):
        subset = str(video.get("subset", ""))
        if allowed is not None and subset not in allowed:
            continue
        annotations = []
        for index, annotation in enumerate(video.get("annotations", [])):
            segment = tuple(float(value) for value in annotation.get("segment", ()))
            if len(segment) != 2 or segment[1] <= segment[0]:
                invalid.append({"video_id": video_id, "index": index, "segment": segment})
                continue
            row = {"segment": segment, "label": str(annotation["label"])}
            annotations.append(row)
            durations.append(segment[1] - segment[0])
            labels[row["label"]] += 1
        maximum = _max_concurrency(annotations)
        same_class_maximum = _max_concurrency(annotations, same_class=True)
        per_video.append(
            {
                "video_id": str(video_id),
                "subset": subset,
                "instances": len(annotations),
                "max_concurrency": maximum,
                "max_same_class_concurrency": same_class_maximum,
            }
        )

    concurrency = [row["max_concurrency"] for row in per_video]
    same_class = [row["max_same_class_concurrency"] for row in per_video]
    global_max = max(concurrency, default=0)
    suggested_slots = max(1, global_max + 1)
    return {
        "subsets": sorted(allowed) if allowed is not None else "all",
        "num_videos": len(per_video),
        "num_instances": len(durations),
        "num_classes": len(labels),
        "invalid_instances": invalid,
        "duration_sec": {
            "mean": sum(durations) / len(durations) if durations else None,
            "p50": _percentile(durations, 50),
            "p90": _percentile(durations, 90),
            "p95": _percentile(durations, 95),
            "max": max(durations, default=None),
        },
        "concurrency": {
            "global_max": global_max,
            "p95_video_max": _percentile(concurrency, 95),
            "videos_with_overlap": sum(value >= 2 for value in concurrency),
            "histogram": dict(sorted(Counter(concurrency).items())),
        },
        "same_class_concurrency": {
            "global_max": max(same_class, default=0),
            "p95_video_max": _percentile(same_class, 95),
            "videos_with_overlap": sum(value >= 2 for value in same_class),
            "histogram": dict(sorted(Counter(same_class).items())),
        },
        "suggested_num_slots": suggested_slots,
        "per_video": per_video,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("--subset", action="append", dest="subsets")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.annotation.open("r", encoding="utf-8") as file:
        report = analyze_database(json.load(file), subsets=args.subsets)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
