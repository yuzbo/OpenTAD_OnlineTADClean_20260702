"""Deterministic, model-outcome-blind R0 annotation census for Protocol V2."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import random


R0_REPORT_SCHEMA = "prefix-route-r0-derived-report-v2"
R0_DETAIL_SCHEMA = "prefix-route-r0-reviewer-detail-v2"
R0_COLLECTOR_VERSION = "20260717.2"
FEATURE_STRIDE_FRAMES = 8
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 2026071701
PRIMARY_FAMILIES = (
    "same_class_repetition",
    "same_class_overlap",
    "same_bin_end_start",
    "direct_complete",
)


class PrefixRouteR0Error(ValueError):
    """Raised when R0 inputs do not define one reproducible census."""


def canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _finite(value, label):
    if isinstance(value, bool):
        raise PrefixRouteR0Error(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PrefixRouteR0Error(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise PrefixRouteR0Error(f"{label} must be finite")
    return parsed


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise PrefixRouteR0Error(f"{label} must be a positive integer")
    parsed = value
    if parsed <= 0:
        raise PrefixRouteR0Error(f"{label} must be a positive integer")
    return parsed


def parse_class_map_bytes(payload):
    if not isinstance(payload, bytes):
        raise PrefixRouteR0Error("class map must be supplied as bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PrefixRouteR0Error("class map must be UTF-8") from exc
    if text and not text.endswith("\n"):
        raise PrefixRouteR0Error("class map must end with a newline")
    names = [line.rstrip("\r") for line in text.splitlines()]
    if not names or any(not name or name != name.strip() for name in names):
        raise PrefixRouteR0Error("class map contains an empty or padded class")
    if len(names) != len(set(names)):
        raise PrefixRouteR0Error("class map contains duplicate classes")
    if "Ambiguous" in names:
        raise PrefixRouteR0Error("Ambiguous must not be a legal class")
    return tuple(names)


def _type7_percentile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0.0 <= probability <= 1.0:
        raise PrefixRouteR0Error("percentile probability is outside [0, 1]")
    rank = (len(ordered) - 1) * probability
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values):
    values = [float(value) for value in values]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "p25": _type7_percentile(values, 0.25),
        "p50": _type7_percentile(values, 0.50),
        "p75": _type7_percentile(values, 0.75),
        "p90": _type7_percentile(values, 0.90),
        "p95": _type7_percentile(values, 0.95),
        "max": max(values),
    }


def _decision_bin_for_start(start_frame, stride):
    return int(start_frame) // int(stride)


def _decision_bin_for_end(end_observation_count, stride):
    return (int(end_observation_count) - 1) // int(stride)


def _frame_coordinates(start_sec, end_sec, duration_sec, frame_count):
    scale = frame_count / duration_sec
    start_frame = min(
        frame_count - 1,
        max(0, int(math.floor(scale * start_sec))),
    )
    end_observation_count = min(
        frame_count,
        max(1, int(math.ceil(scale * end_sec))),
    )
    return start_frame, end_observation_count


def _interval_overlap(left, right):
    return max(
        0.0,
        min(left["end_sec"], right["end_sec"])
        - max(left["start_sec"], right["start_sec"]),
    )


def _max_concurrency(instances, *, same_class):
    events = []
    for instance in instances:
        label = instance["label"]
        events.append((instance["start_sec"], 1, label, instance["instance_index"]))
        events.append((instance["end_sec"], -1, label, instance["instance_index"]))
    # End before start makes touching half-open intervals non-overlapping.
    events.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    if same_class:
        active = defaultdict(int)
        maximum = 0
        for _, delta, label, _ in events:
            active[label] += delta
            maximum = max(maximum, active[label])
        return maximum
    active = 0
    maximum = 0
    for _, delta, _, _ in events:
        active += delta
        maximum = max(maximum, active)
    return maximum


def _parse_video(video_id, video, class_order, stride, expected_subset):
    if not isinstance(video, dict):
        raise PrefixRouteR0Error(f"video {video_id} must be an object")
    duration = _finite(video.get("duration"), f"{video_id}.duration")
    if duration <= 0:
        raise PrefixRouteR0Error(f"{video_id}.duration must be positive")
    frame_count = _positive_int(video.get("frame"), f"{video_id}.frame")
    subset = video.get("subset")
    if subset != expected_subset:
        raise PrefixRouteR0Error(
            f"{video_id}.subset must equal registered reporting subset "
            f"{expected_subset!r}"
        )
    annotations = video.get("annotations", [])
    if not isinstance(annotations, list):
        raise PrefixRouteR0Error(f"{video_id}.annotations must be an array")

    parsed = []
    ambiguous = []
    for source_index, annotation in enumerate(annotations):
        reasons = []
        if not isinstance(annotation, dict):
            ambiguous.append(
                {
                    "source_index": source_index,
                    "reason": "MALFORMED_ANNOTATION",
                    "duration_sec": None,
                }
            )
            continue
        label = annotation.get("label")
        segment = annotation.get("segment")
        if label == "Ambiguous":
            reason = "AMBIGUOUS_LABEL"
        elif not isinstance(label, str) or not label:
            reason = "MISSING_OR_INVALID_LABEL"
        elif label not in class_order:
            raise PrefixRouteR0Error(
                f"{video_id}.annotations[{source_index}] has unknown class {label!r}"
            )
        else:
            reason = None
        if not isinstance(segment, list) or len(segment) != 2:
            reasons.append("MALFORMED_SEGMENT")
            start = end = None
        else:
            try:
                start = _finite(segment[0], f"{video_id}.segment.start")
                end = _finite(segment[1], f"{video_id}.segment.end")
            except PrefixRouteR0Error:
                start = end = None
                reasons.append("NONFINITE_SEGMENT")
        if start is not None:
            if not 0.0 <= start < end <= duration:
                reasons.append("OUT_OF_BOUNDS_OR_NONPOSITIVE_INTERVAL")
        if reason is not None:
            reasons.append(reason)
        if reasons:
            ambiguous.append(
                {
                    "source_index": source_index,
                    "reason": "+".join(sorted(set(reasons))),
                    "duration_sec": (
                        max(0.0, end - start)
                        if start is not None and end is not None
                        else None
                    ),
                }
            )
            continue
        start_frame, end_count = _frame_coordinates(
            start,
            end,
            duration,
            frame_count,
        )
        parsed.append(
            {
                "source_index": source_index,
                "label": label,
                "class_index": class_order[label],
                "start_sec": start,
                "end_sec": end,
                "duration_sec": end - start,
                "start_frame": start_frame,
                "end_observation_count": end_count,
                "start_bin": _decision_bin_for_start(start_frame, stride),
                "end_bin": _decision_bin_for_end(end_count, stride),
            }
        )

    duplicate_groups = defaultdict(list)
    for row in parsed:
        duplicate_groups[
            (row["label"], row["start_sec"], row["end_sec"])
        ].append(row)
    duplicate_source_indexes = {
        row["source_index"]
        for rows in duplicate_groups.values()
        if len(rows) > 1
        for row in rows
    }
    legal = []
    for row in parsed:
        if row["source_index"] in duplicate_source_indexes:
            ambiguous.append(
                {
                    "source_index": row["source_index"],
                    "reason": "DUPLICATE_EXACT_ANNOTATION",
                    "duration_sec": row["duration_sec"],
                }
            )
        else:
            legal.append(row)

    legal.sort(
        key=lambda row: (
            row["start_frame"],
            row["end_observation_count"],
            row["class_index"],
            row["source_index"],
        )
    )
    for instance_index, row in enumerate(legal):
        row["instance_index"] = instance_index

    by_class = defaultdict(list)
    for row in legal:
        by_class[row["label"]].append(row)

    family_indexes = {family: set() for family in PRIMARY_FAMILIES}
    sequential_gaps_sec = []
    sequential_gaps_bins = []
    for rows in by_class.values():
        if len(rows) >= 2:
            family_indexes["same_class_repetition"].update(
                row["instance_index"] for row in rows
            )
        for left, right in zip(rows, rows[1:]):
            sequential_gaps_sec.append(right["start_sec"] - left["end_sec"])
            sequential_gaps_bins.append(right["start_bin"] - left["end_bin"])

    overlap = {
        "same_class": {
            "pair_count": 0,
            "instance_indexes": set(),
            "duration_sec": [],
        },
        "cross_class": {
            "pair_count": 0,
            "instance_indexes": set(),
            "duration_sec": [],
        },
    }
    same_bin = {
        key: {
            relation: {"pair_count": 0, "instance_indexes": set()}
            for relation in ("handoff_or_touching", "overlap_transition")
        }
        for key in ("same_class", "cross_class")
    }
    for left_index, left in enumerate(legal):
        for right in legal[left_index + 1 :]:
            class_key = (
                "same_class" if left["label"] == right["label"] else "cross_class"
            )
            duration_overlap = _interval_overlap(left, right)
            if duration_overlap > 0:
                overlap[class_key]["pair_count"] += 1
                overlap[class_key]["instance_indexes"].update(
                    (left["instance_index"], right["instance_index"])
                )
                overlap[class_key]["duration_sec"].append(duration_overlap)
                if class_key == "same_class":
                    family_indexes["same_class_overlap"].update(
                        (left["instance_index"], right["instance_index"])
                    )

            # One unordered pair yields exactly one earlier-end -> other-start relation.
            predecessor, successor = min(
                ((left, right), (right, left)),
                key=lambda pair: (
                    pair[0]["end_sec"],
                    pair[1]["start_sec"],
                    pair[0]["instance_index"],
                    pair[1]["instance_index"],
                ),
            )
            if predecessor["end_bin"] == successor["start_bin"]:
                relation = (
                    "handoff_or_touching"
                    if predecessor["end_sec"] <= successor["start_sec"]
                    else "overlap_transition"
                )
                same_bin[class_key][relation]["pair_count"] += 1
                same_bin[class_key][relation]["instance_indexes"].update(
                    (
                        predecessor["instance_index"],
                        successor["instance_index"],
                    )
                )
                family_indexes["same_bin_end_start"].update(
                    (
                        predecessor["instance_index"],
                        successor["instance_index"],
                    )
                )

    direct_complete = {
        row["instance_index"]
        for row in legal
        if row["start_bin"] == row["end_bin"]
    }
    family_indexes["direct_complete"].update(direct_complete)

    family_classes = {
        family: sorted(
            {
                legal[index]["label"]
                for index in indexes
            },
            key=lambda label: class_order[label],
        )
        for family, indexes in family_indexes.items()
    }
    return {
        "video_id": video_id,
        "subset": subset,
        "duration_sec": duration,
        "frame_count": frame_count,
        "legal_instances": legal,
        "ambiguous_intervals": sorted(
            ambiguous,
            key=lambda row: (row["source_index"], row["reason"]),
        ),
        "sequential_gaps_sec": sequential_gaps_sec,
        "sequential_gaps_bins": sequential_gaps_bins,
        "overlap": overlap,
        "same_bin": same_bin,
        "direct_complete_indexes": direct_complete,
        "family_indexes": family_indexes,
        "family_classes": family_classes,
        "max_total_concurrency": _max_concurrency(legal, same_class=False),
        "max_same_class_concurrency": _max_concurrency(legal, same_class=True),
    }


def _family_eligibility(details, family):
    cluster_sizes = [len(row["family_indexes"][family]) for row in details]
    positive_videos = sum(size > 0 for size in cluster_sizes)
    instances = sum(cluster_sizes)
    classes = {
        label
        for row in details
        for label in row["family_classes"][family]
    }
    positive_fraction = positive_videos / len(details) if details else 0.0
    primary_checks = {
        "independent_videos": positive_videos >= 30,
        "gt_instances": instances >= 100,
        "fraction_of_positive_videos": positive_fraction >= 0.05,
        "classes": len(classes) >= 3,
    }
    secondary_checks = {
        "independent_videos": positive_videos >= 10,
        "gt_instances": instances >= 30,
    }
    if all(primary_checks.values()):
        label = "DESCRIPTIVE_PRIMARY_COVERAGE"
    elif all(secondary_checks.values()):
        label = "DESCRIPTIVE_SECONDARY_COVERAGE"
    else:
        label = "DESCRIPTIVE_SPARSE"
    return {
        "role": "DESCRIPTIVE_SUPPORT_ONLY_NOT_DOWNSTREAM_POWER_OR_CLAIM",
        "label": label,
        "independent_videos": positive_videos,
        "gt_instances": instances,
        "fraction_of_positive_videos": positive_fraction,
        "classes": len(classes),
        "class_names": sorted(classes),
        "cluster_size_vector_includes_zero_videos": True,
        "primary_checks": primary_checks,
        "secondary_checks": secondary_checks,
    }


def _bootstrap(details, resamples, seed):
    if not details:
        raise PrefixRouteR0Error("R0 population is empty")
    rng = random.Random(int(seed))
    count = len(details)
    target_names = ["legal_instances_per_video"]
    target_names.extend(f"{family}_positive_video_fraction" for family in PRIMARY_FAMILIES)
    target_names.extend(f"{family}_instances_per_video" for family in PRIMARY_FAMILIES)
    samples = {target: [] for target in target_names}
    for _ in range(int(resamples)):
        indexes = [rng.randrange(count) for _ in range(count)]
        sampled = [details[index] for index in indexes]
        samples["legal_instances_per_video"].append(
            sum(len(row["legal_instances"]) for row in sampled) / count
        )
        for family in PRIMARY_FAMILIES:
            sizes = [len(row["family_indexes"][family]) for row in sampled]
            samples[f"{family}_positive_video_fraction"].append(
                sum(size > 0 for size in sizes) / count
            )
            samples[f"{family}_instances_per_video"].append(sum(sizes) / count)
    estimates = {
        "legal_instances_per_video": (
            sum(len(row["legal_instances"]) for row in details) / count
        )
    }
    for family in PRIMARY_FAMILIES:
        sizes = [len(row["family_indexes"][family]) for row in details]
        estimates[f"{family}_positive_video_fraction"] = (
            sum(size > 0 for size in sizes) / count
        )
        estimates[f"{family}_instances_per_video"] = sum(sizes) / count
    return {
        "method": "nonparametric_video_cluster_percentile_type7",
        "resamples": int(resamples),
        "seed": int(seed),
        "confidence_level": 0.95,
        "same_resample_indices_for_all_targets": True,
        "targets": {
            target: {
                "estimate": estimates[target],
                "lower": _type7_percentile(values, 0.025),
                "upper": _type7_percentile(values, 0.975),
            }
            for target, values in sorted(samples.items())
        },
    }


def collect_r0_census(
    annotation,
    class_names,
    reporting_video_ids,
    *,
    feature_stride_frames=FEATURE_STRIDE_FRAMES,
    bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    bootstrap_seed=BOOTSTRAP_SEED,
    expected_subset="validation",
    annotation_exposure_status=(
        "DESIGN_EXPOSED_ROUTE_SELECTION_AND_BENCHMARK"
    ),
):
    """Derive aggregate R0 evidence and reviewer detail from source objects."""

    if not isinstance(annotation, dict) or set(annotation) < {"database"}:
        raise PrefixRouteR0Error("annotation must contain a database object")
    database = annotation["database"]
    if not isinstance(database, dict):
        raise PrefixRouteR0Error("annotation database must be an object")
    class_names = tuple(class_names)
    if not class_names or len(class_names) != len(set(class_names)):
        raise PrefixRouteR0Error("class names must be unique and non-empty")
    class_order = {name: index for index, name in enumerate(class_names)}
    video_ids = list(reporting_video_ids)
    if (
        not video_ids
        or any(not isinstance(video_id, str) or not video_id for video_id in video_ids)
        or video_ids != sorted(video_ids)
        or len(video_ids) != len(set(video_ids))
    ):
        raise PrefixRouteR0Error(
            "reporting video IDs must be sorted unique non-empty strings"
        )
    missing = [video_id for video_id in video_ids if video_id not in database]
    if missing:
        raise PrefixRouteR0Error(
            f"annotation database misses reporting IDs: {missing[:5]}"
        )
    stride = _positive_int(feature_stride_frames, "feature_stride_frames")
    if expected_subset != "validation":
        raise PrefixRouteR0Error("registered reporting subset must be validation")
    if annotation_exposure_status != (
        "DESIGN_EXPOSED_ROUTE_SELECTION_AND_BENCHMARK"
    ):
        raise PrefixRouteR0Error("R0 annotation exposure status differs")
    details = [
        _parse_video(
            video_id,
            database[video_id],
            class_order,
            stride,
            expected_subset,
        )
        for video_id in video_ids
    ]

    legal_instances = [
        instance for row in details for instance in row["legal_instances"]
    ]
    ambiguous_rows = [
        interval for row in details for interval in row["ambiguous_intervals"]
    ]
    per_class = Counter(instance["label"] for instance in legal_instances)
    repeated_videos = sum(
        bool(row["family_indexes"]["same_class_repetition"]) for row in details
    )
    repeated_instances = sum(
        len(row["family_indexes"]["same_class_repetition"]) for row in details
    )
    overlap_report = {}
    for class_key in ("same_class", "cross_class"):
        pair_count = sum(
            row["overlap"][class_key]["pair_count"] for row in details
        )
        involved_instances = sum(
            len(row["overlap"][class_key]["instance_indexes"]) for row in details
        )
        positive_videos = sum(
            row["overlap"][class_key]["pair_count"] > 0 for row in details
        )
        durations = [
            value
            for row in details
            for value in row["overlap"][class_key]["duration_sec"]
        ]
        overlap_report[class_key] = {
            "pair_count": pair_count,
            "involved_instance_count_summed_within_video": involved_instances,
            "positive_video_count": positive_videos,
            "overlap_duration_sec": _summary(durations),
        }

    same_bin_report = {}
    for class_key in ("same_class", "cross_class"):
        same_bin_report[class_key] = {}
        for relation in ("handoff_or_touching", "overlap_transition"):
            rows = [row["same_bin"][class_key][relation] for row in details]
            same_bin_report[class_key][relation] = {
                "pair_count": sum(row["pair_count"] for row in rows),
                "involved_instance_count_summed_within_video": sum(
                    len(row["instance_indexes"]) for row in rows
                ),
                "positive_video_count": sum(row["pair_count"] > 0 for row in rows),
            }

    direct_counts = [
        len(row["direct_complete_indexes"]) for row in details
    ]
    eligibility = {
        family: _family_eligibility(details, family)
        for family in PRIMARY_FAMILIES
    }
    aggregate = {
        "video_count": len(details),
        "legal_instance_count": len(legal_instances),
        "class_count_and_order": {
            "count": len(class_names),
            "order": list(class_names),
        },
        "per_class_instance_count": {
            name: per_class.get(name, 0) for name in class_names
        },
        "per_video_instance_count_distribution": _summary(
            len(row["legal_instances"]) for row in details
        ),
        "action_duration_seconds_distribution": _summary(
            instance["duration_sec"] for instance in legal_instances
        ),
        "action_duration_bins_distribution": _summary(
            instance["end_bin"] - instance["start_bin"] + 1
            for instance in legal_instances
        ),
        "same_class_repeated": {
            "positive_video_count": repeated_videos,
            "involved_instance_count": repeated_instances,
        },
        "same_class_sequential_gap_seconds_distribution": _summary(
            value for row in details for value in row["sequential_gaps_sec"]
        ),
        "same_class_sequential_gap_bins_distribution": _summary(
            value for row in details for value in row["sequential_gaps_bins"]
        ),
        "overlap_pairs": overlap_report,
        "same_bin_end_start_pairs": same_bin_report,
        "direct_complete": {
            "positive_video_count": sum(value > 0 for value in direct_counts),
            "instance_count": sum(direct_counts),
        },
        "maximum_total_concurrency": max(
            (row["max_total_concurrency"] for row in details),
            default=0,
        ),
        "maximum_same_class_concurrency": max(
            (row["max_same_class_concurrency"] for row in details),
            default=0,
        ),
        "zero_action_video_count": sum(
            not row["legal_instances"] for row in details
        ),
        "ambiguous": {
            "positive_video_count": sum(
                bool(row["ambiguous_intervals"]) for row in details
            ),
            "interval_count": len(ambiguous_rows),
            "duration_sec": _summary(
                row["duration_sec"]
                for row in ambiguous_rows
                if row["duration_sec"] is not None
            ),
            "reason_counts": dict(
                sorted(Counter(row["reason"] for row in ambiguous_rows).items())
            ),
        },
    }
    detail_payload = {
        "schema_version": R0_DETAIL_SCHEMA,
        "collector_version": R0_COLLECTOR_VERSION,
        "definitions": {
            "interval": "half_open_[start_sec,end_sec)",
            "touching_is_overlap": False,
            "first_previous_observation_count": 0,
            "start_frame": "floor(frame_count*start_sec/duration_sec)",
            "end_observation_count": "ceil(frame_count*end_sec/duration_sec)",
            "terminal_end_observation_count": "frame_count",
            "start_bin": "start_frame//feature_stride_frames",
            "end_bin": "(end_observation_count-1)//feature_stride_frames",
            "exact_duplicate_policy": "exclude_all_members_as_ambiguous",
            "pair_counting": "unordered_pair_once",
            "concurrency_tie": "end_before_start",
            "sequential_gap": "next_start-current_end",
        },
        "feature_stride_frames": stride,
        "videos": [
            {
                "video_id": row["video_id"],
                "subset": row["subset"],
                "duration_sec": row["duration_sec"],
                "frame_count": row["frame_count"],
                "legal_instances": row["legal_instances"],
                "ambiguous_intervals": row["ambiguous_intervals"],
                "family_instance_indexes": {
                    family: sorted(indexes)
                    for family, indexes in row["family_indexes"].items()
                },
                "family_classes": row["family_classes"],
                "overlap": {
                    class_key: {
                        "pair_count": values["pair_count"],
                        "instance_indexes": sorted(values["instance_indexes"]),
                        "duration_sec": values["duration_sec"],
                    }
                    for class_key, values in row["overlap"].items()
                },
                "same_bin": {
                    class_key: {
                        relation: {
                            "pair_count": values["pair_count"],
                            "instance_indexes": sorted(values["instance_indexes"]),
                        }
                        for relation, values in relations.items()
                    }
                    for class_key, relations in row["same_bin"].items()
                },
                "max_total_concurrency": row["max_total_concurrency"],
                "max_same_class_concurrency": row["max_same_class_concurrency"],
            }
            for row in details
        ],
    }
    detail_sha256 = hashlib.sha256(
        canonical_json_bytes(detail_payload)
    ).hexdigest()
    report = {
        "schema_version": R0_REPORT_SCHEMA,
        "collector_version": R0_COLLECTOR_VERSION,
        "status": "PASS_R0_COMPLETE",
        "reporting_population_role": "canonical_reporting_213",
        "annotation_exposure_status": annotation_exposure_status,
        "author_disclosure": "aggregate_only_no_video_ids",
        "feature_stride_frames": stride,
        "aggregate": aggregate,
        "claim_eligibility": eligibility,
        "bootstrap": _bootstrap(
            details,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "reviewer_detail_commitment_sha256": detail_sha256,
    }
    return report, detail_payload


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "FEATURE_STRIDE_FRAMES",
    "PRIMARY_FAMILIES",
    "PrefixRouteR0Error",
    "canonical_json_bytes",
    "collect_r0_census",
    "parse_class_map_bytes",
]
