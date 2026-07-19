"""Executable negative controls frozen by Prefix-Route Protocol V2."""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
import hashlib
import json
import math


CONTROL_SCHEMA = "prefix-route-negative-control-v2"
FEATURE_TIME_SHUFFLE_SEED = 2026071703
SEMANTIC_DERANGEMENT_SEED = 2026071704
SEMANTIC_CLASS_AP_DROP_MARGIN = 0.05


class PrefixRouteControlError(ValueError):
    pass


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


def _digest(*parts):
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_time_shuffle_permutation(
    video_id,
    token_count,
    *,
    seed=FEATURE_TIME_SHUFFLE_SEED,
):
    """Return destination->source indices for a no-fixed-point token rotation."""

    if not isinstance(video_id, str) or not video_id:
        raise PrefixRouteControlError("video_id must be non-empty text")
    if isinstance(token_count, bool) or not isinstance(token_count, int):
        raise PrefixRouteControlError("token_count must be an integer")
    if token_count < 2:
        raise PrefixRouteControlError("length-one video is unconstructible")
    ranked = sorted(
        range(token_count),
        key=lambda index: (_digest(video_id, index, seed), index),
    )
    permutation = list(range(token_count))
    for rank, destination in enumerate(ranked):
        permutation[destination] = ranked[(rank + 1) % token_count]
    if any(destination == source for destination, source in enumerate(permutation)):
        raise PrefixRouteControlError("feature shuffle contains a fixed point")
    if sorted(permutation) != list(range(token_count)):
        raise PrefixRouteControlError("feature shuffle is not a permutation")
    return tuple(permutation)


def count_only_schedule(
    video_id,
    decision_bins,
    *,
    event_count_prior,
    class_prior,
):
    """Construct the frozen feature-free count/class-prior schedule."""

    if not isinstance(video_id, str) or not video_id:
        raise PrefixRouteControlError("video_id must be non-empty text")
    if (
        not isinstance(decision_bins, list)
        or not decision_bins
        or decision_bins != sorted(set(decision_bins))
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in decision_bins)
    ):
        raise PrefixRouteControlError("decision_bins must be sorted unique integers")
    if (
        isinstance(event_count_prior, bool)
        or not isinstance(event_count_prior, int)
        or event_count_prior < 0
    ):
        raise PrefixRouteControlError("event_count_prior must be non-negative")
    if (
        not isinstance(class_prior, list)
        or not class_prior
        or any(
            not isinstance(row, dict)
            or set(row) != {"label", "probability"}
            or not isinstance(row["label"], str)
            or not row["label"]
            or isinstance(row["probability"], bool)
            or not isinstance(row["probability"], (int, float))
            or row["probability"] < 0
            for row in class_prior
        )
    ):
        raise PrefixRouteControlError("class_prior fields differ")
    total = sum(float(row["probability"]) for row in class_prior)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise PrefixRouteControlError("class_prior must sum to one")
    count = min(event_count_prior, len(decision_bins))
    if count == 0:
        selected_bins = []
    else:
        selected_bins = [
            decision_bins[(index * len(decision_bins)) // count]
            for index in range(count)
        ]
    ranked_classes = sorted(
        class_prior,
        key=lambda row: (-float(row["probability"]), row["label"]),
    )
    return {
        "video_id": video_id,
        "scheduled_bins": selected_bins,
        "labels": [
            ranked_classes[index % len(ranked_classes)]["label"]
            for index in range(count)
        ],
    }


def template_timing_schedule(
    video_id,
    decision_bins,
    *,
    duration_template_bins,
    gap_template_bins,
):
    """Construct the frozen feature-free duration/gap template schedule."""

    if not isinstance(video_id, str) or not video_id:
        raise PrefixRouteControlError("video_id must be non-empty text")
    if (
        not isinstance(decision_bins, list)
        or not decision_bins
        or decision_bins != sorted(set(decision_bins))
    ):
        raise PrefixRouteControlError("decision_bins differ")
    for values, label, minimum in (
        (duration_template_bins, "duration_template_bins", 1),
        (gap_template_bins, "gap_template_bins", 0),
    ):
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
                for value in values
            )
        ):
            raise PrefixRouteControlError(f"{label} differs")
    allowed = set(decision_bins)
    rows = []
    cursor = decision_bins[0]
    template_index = 0
    while cursor in allowed:
        duration = duration_template_bins[
            template_index % len(duration_template_bins)
        ]
        rows.append(
            {
                "start_bin": cursor,
                "end_bin": cursor + duration,
            }
        )
        gap = gap_template_bins[template_index % len(gap_template_bins)]
        cursor = cursor + duration + gap
        template_index += 1
    return {"video_id": video_id, "events": rows}


def _validate_instances(instances):
    if not isinstance(instances, list) or not instances:
        raise PrefixRouteControlError("instances must be a non-empty array")
    required = {"video_id", "instance_id", "start_frame", "end_frame", "label"}
    normalized = []
    for index, row in enumerate(instances):
        if not isinstance(row, dict) or set(row) != required:
            raise PrefixRouteControlError(
                f"instances[{index}] fields differ from {sorted(required)}"
            )
        video_id = row["video_id"]
        instance_id = row["instance_id"]
        label = row["label"]
        if not isinstance(video_id, str) or not video_id:
            raise PrefixRouteControlError("video_id must be non-empty text")
        if not isinstance(instance_id, (str, int)) or isinstance(instance_id, bool):
            raise PrefixRouteControlError("instance_id must be string or integer")
        if not isinstance(label, str) or not label:
            raise PrefixRouteControlError("label must be non-empty text")
        try:
            start = float(row["start_frame"])
            end = float(row["end_frame"])
        except (TypeError, ValueError) as exc:
            raise PrefixRouteControlError("frame bounds must be numeric") from exc
        if not start < end:
            raise PrefixRouteControlError("instance interval must be positive")
        normalized.append(
            {
                "video_id": video_id,
                "instance_id": instance_id,
                "start_frame": start,
                "end_frame": end,
                "label": label,
            }
        )
    identities = [
        (row["video_id"], str(row["instance_id"])) for row in normalized
    ]
    if len(identities) != len(set(identities)):
        raise PrefixRouteControlError("instance identities must be unique")
    return normalized


def _is_global_consistent_rename(original, permuted):
    forward = defaultdict(set)
    reverse = defaultdict(set)
    for source, target in zip(original, permuted):
        forward[source].add(target)
        reverse[target].add(source)
    return (
        all(len(targets) == 1 for targets in forward.values())
        and all(len(sources) == 1 for sources in reverse.values())
    )


def semantic_derangement(
    instances,
    *,
    split_id,
    seed=SEMANTIC_DERANGEMENT_SEED,
):
    """Derange labels within one frozen split while preserving its label multiset."""

    rows = _validate_instances(instances)
    if not isinstance(split_id, str) or not split_id:
        raise PrefixRouteControlError("split_id must be non-empty text")
    ranked_indexes = sorted(
        range(len(rows)),
        key=lambda index: (
            _digest(
                split_id,
                rows[index]["video_id"],
                rows[index]["start_frame"],
                rows[index]["end_frame"],
                rows[index]["label"],
                rows[index]["instance_id"],
                seed,
            ),
            rows[index]["video_id"],
            str(rows[index]["instance_id"]),
        ),
    )
    ranked_labels = [rows[index]["label"] for index in ranked_indexes]
    counts = Counter(ranked_labels)
    maximum_count = max(counts.values())
    if maximum_count * 2 > len(rows):
        raise PrefixRouteControlError(
            "FAIL_CONTROL_UNCONSTRUCTIBLE: maximum label frequency exceeds half"
        )
    grouped_positions = sorted(
        range(len(ranked_labels)),
        key=lambda rank: (ranked_labels[rank], rank),
    )
    grouped_tokens = [
        ranked_labels[rank]
        for rank in grouped_positions
    ]
    selected_labels = [None] * len(rows)
    for grouped_rank, position_rank in enumerate(grouped_positions):
        selected_labels[position_rank] = grouped_tokens[
            (grouped_rank + maximum_count) % len(rows)
        ]
    if any(source == target for source, target in zip(ranked_labels, selected_labels)):
        raise PrefixRouteControlError(
            "FAIL_CONTROL_UNCONSTRUCTIBLE: multiset rotation has a fixed label"
        )
    non_global_swap = None
    if _is_global_consistent_rename(ranked_labels, selected_labels):
        for left in range(len(rows)):
            source_left = ranked_labels[left]
            target_left = selected_labels[left]
            if counts[source_left] < 2:
                continue
            for right in range(left + 1, len(rows)):
                source_right = ranked_labels[right]
                target_right = selected_labels[right]
                if (
                    target_right != source_left
                    and target_left != source_right
                    and target_right != target_left
                ):
                    selected_labels[left], selected_labels[right] = (
                        target_right,
                        target_left,
                    )
                    non_global_swap = [left, right]
                    break
            if non_global_swap is not None:
                break
        if non_global_swap is None or _is_global_consistent_rename(
            ranked_labels,
            selected_labels,
        ):
            raise PrefixRouteControlError(
                "FAIL_CONTROL_UNCONSTRUCTIBLE: only global class rename exists"
            )

    permuted_by_index = [None] * len(rows)
    for rank, row_index in enumerate(ranked_indexes):
        permuted_by_index[row_index] = selected_labels[rank]
    if sorted(permuted_by_index) != sorted(row["label"] for row in rows):
        raise PrefixRouteControlError("semantic derangement changed label multiset")
    if any(
        row["label"] == target
        for row, target in zip(rows, permuted_by_index)
    ):
        raise PrefixRouteControlError("semantic derangement contains a fixed label")
    if _is_global_consistent_rename(
        [row["label"] for row in rows],
        permuted_by_index,
    ):
        raise PrefixRouteControlError("global consistent class rename is forbidden")

    output_rows = [
        {
            **row,
            "original_label": row["label"],
            "label": target,
        }
        for row, target in zip(rows, permuted_by_index)
    ]
    source_sha256 = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    output_sha256 = hashlib.sha256(canonical_json_bytes(output_rows)).hexdigest()
    return {
        "schema_version": CONTROL_SCHEMA,
        "control": "semantic_derangement",
        "scope": "within_one_frozen_evaluation_split",
        "split_id": split_id,
        "seed": int(seed),
        "algorithm": (
            "sha256_rank_then_max_frequency_multiset_rotation_with_non_global_swap"
        ),
        "rotation_offset": maximum_count,
        "non_global_swap_rank_indexes": non_global_swap,
        "instance_count": len(rows),
        "source_sha256": source_sha256,
        "output_sha256": output_sha256,
        "global_label_multiset_preserved": True,
        "all_instance_labels_changed": True,
        "global_consistent_class_rename": False,
        "class_ap_drop_margin": SEMANTIC_CLASS_AP_DROP_MARGIN,
        "instances": output_rows,
    }


__all__ = [
    "FEATURE_TIME_SHUFFLE_SEED",
    "PrefixRouteControlError",
    "SEMANTIC_CLASS_AP_DROP_MARGIN",
    "SEMANTIC_DERANGEMENT_SEED",
    "canonical_json_bytes",
    "count_only_schedule",
    "feature_time_shuffle_permutation",
    "semantic_derangement",
    "template_timing_schedule",
]
