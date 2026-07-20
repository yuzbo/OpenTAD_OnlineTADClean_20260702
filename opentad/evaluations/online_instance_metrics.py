"""Auditable instance metrics for immutable strictly causal On-TAD emissions.

The evaluator intentionally performs no NMS, segment merging, score filtering,
or event reordering. Protocol auditing retains the online chronology, while
instance scoring uses a frozen deterministic global one-to-one assignment.
"""

from collections import Counter, defaultdict
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

SCHEMA_VERSION = "online_instance_metrics.v2"
_EPSILON = 1e-9


class OnlineInstanceMetricError(ValueError):
    """Base error for malformed Full PETAL metric inputs."""


class OnlineInstanceInputError(OnlineInstanceMetricError):
    """Raised when rows do not satisfy the metric input schema."""


class OnlineInstanceProtocolError(OnlineInstanceMetricError):
    """Raised when immutable online emission rules are violated."""

    def __init__(self, violations):
        self.violations = tuple(violations)
        details = "; ".join(
            "{code}[{stream_key}:{row_index}]".format(**violation)
            for violation in self.violations
        )
        super().__init__("Full PETAL emission protocol violations: " + details)


def _finite_number(value, label):
    if isinstance(value, bool):
        raise OnlineInstanceInputError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OnlineInstanceInputError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise OnlineInstanceInputError(f"{label} must be a finite number")
    return number


def _scalar_id(value, label):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise OnlineInstanceInputError(f"{label} must be a JSON scalar identifier")
    if isinstance(value, float) and not math.isfinite(value):
        raise OnlineInstanceInputError(f"{label} must be finite")
    if isinstance(value, str) and not value:
        raise OnlineInstanceInputError(f"{label} must not be empty")
    return value


def _first_present(row, keys, label, required=True):
    present = [key for key in keys if key in row]
    if not present:
        if required:
            raise OnlineInstanceInputError(f"{label} is required")
        return None
    value = row[present[0]]
    for key in present[1:]:
        if row[key] != value:
            raise OnlineInstanceInputError(
                f"conflicting aliases for {label}: {present[0]} and {key}"
            )
    return value


def _inject_stream(row, stream_key):
    if not isinstance(row, dict):
        raise OnlineInstanceInputError("metric rows must be JSON objects")
    copied = dict(row)
    copied.setdefault("video_id", stream_key)
    return copied


def _flatten_stream_mapping(mapping):
    rows = []
    for stream_key, stream_rows in mapping.items():
        if not isinstance(stream_rows, (list, tuple)):
            raise OnlineInstanceInputError(
                f"rows for stream {stream_key!r} must be a sequence"
            )
        rows.extend(_inject_stream(row, stream_key) for row in stream_rows)
    return rows


def _flatten_rows(value, kind):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        rows = list(value)
    elif isinstance(value, dict):
        if kind == "ground truth" and "database" in value:
            database = value["database"]
            if not isinstance(database, dict):
                raise OnlineInstanceInputError("ground-truth database must be an object")
            rows = []
            for stream_key, video in database.items():
                if not isinstance(video, dict):
                    raise OnlineInstanceInputError("ground-truth video records must be objects")
                for raw_row in video.get("annotations", ()):
                    row = _inject_stream(raw_row, stream_key)
                    row.setdefault(
                        "coordinate_system",
                        video.get("coordinate_system", "seconds"),
                    )
                    if "fps" in video:
                        row.setdefault("fps", video["fps"])
                    rows.append(row)
        else:
            container_keys = (
                ("ground_truth", "annotations")
                if kind == "ground truth"
                else ("emissions", "results")
            )
            container = None
            for key in container_keys:
                if key in value:
                    container = value[key]
                    break
            if container is None:
                if any(key in value for key in ("label", "segment", "end_frame")):
                    rows = [value]
                else:
                    rows = _flatten_stream_mapping(value)
            elif isinstance(container, dict):
                rows = _flatten_stream_mapping(container)
            elif isinstance(container, (list, tuple)):
                rows = list(container)
            else:
                raise OnlineInstanceInputError(f"{kind} rows must be a sequence or stream mapping")
    else:
        raise OnlineInstanceInputError(f"{kind} must be a sequence or JSON object")
    if not all(isinstance(row, dict) for row in rows):
        raise OnlineInstanceInputError(f"all {kind} rows must be JSON objects")
    return [dict(row) for row in rows]


def _segment(row, label, default_fps):
    start_keys = ("start_frame", "predicted_start_frame")
    end_keys = ("end_frame", "predicted_end_frame")
    has_start_frame = any(key in row for key in start_keys)
    has_end_frame = any(key in row for key in end_keys)
    if has_start_frame or has_end_frame:
        if not has_start_frame or not has_end_frame:
            raise OnlineInstanceInputError(
                f"{label} must provide both explicit start_frame and end_frame bounds"
            )
        start = _finite_number(
            _first_present(row, start_keys, f"{label}.start_frame"),
            f"{label}.start_frame",
        )
        end = _finite_number(
            _first_present(row, end_keys, f"{label}.end_frame"),
            f"{label}.end_frame",
        )
    elif "segment" in row:
        segment = row["segment"]
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            raise OnlineInstanceInputError(f"{label}.segment must contain [start, end]")
        start = _finite_number(segment[0], f"{label}.segment[0]")
        end = _finite_number(segment[1], f"{label}.segment[1]")
        coordinate_system = str(row.get("coordinate_system", "frames")).lower()
        if coordinate_system == "seconds":
            fps = _finite_number(row.get("fps", default_fps), f"{label}.fps")
            if fps <= 0:
                raise OnlineInstanceInputError(f"{label}.fps must be positive")
            start *= fps
            end *= fps
        elif coordinate_system != "frames":
            raise OnlineInstanceInputError(
                f"{label}.coordinate_system must be 'frames' or 'seconds'"
            )
    else:
        raise OnlineInstanceInputError(f"{label} requires frame bounds or a segment")
    if end <= start:
        raise OnlineInstanceInputError(f"{label} segment end must be greater than start")
    return (start, end)


def _video_id(row, label):
    value = row.get("video_id")
    if value is None:
        value = _first_present(
            row,
            ("stream_key", "stream_id"),
            f"{label}.video_id",
        )
    return str(_scalar_id(value, f"{label}.video_id"))


def _stream_key(row, label):
    value = _first_present(
        row,
        ("runtime_stream_key", "stream_key", "stream_id"),
        f"{label}.runtime_stream_key",
        required=False,
    )
    if value is None:
        value = _first_present(
            row,
            ("video_id",),
            f"{label}.runtime_stream_key",
        )
    return str(_scalar_id(value, f"{label}.runtime_stream_key"))


def _normalize_ground_truth(value, default_fps):
    normalized = []
    seen_ids = set()
    for index, row in enumerate(_flatten_rows(value, "ground truth")):
        label = f"ground_truth[{index}]"
        gt_id = _first_present(row, ("gt_id", "instance_id", "id"), f"{label}.gt_id", False)
        if gt_id is None:
            gt_id = f"gt:{index}"
        gt_id = _scalar_id(gt_id, f"{label}.gt_id")
        video_id = _video_id(row, label)
        identity = (video_id, type(gt_id).__name__, repr(gt_id))
        if identity in seen_ids:
            raise OnlineInstanceInputError(
                f"duplicate ground-truth ID {gt_id!r} in video {video_id!r}"
            )
        seen_ids.add(identity)
        target_label = _first_present(row, ("label", "class", "class_id"), f"{label}.label")
        normalized.append(
            {
                "index": index,
                "id": gt_id,
                "video_id": video_id,
                "label": target_label,
                "segment": _segment(row, label, default_fps),
            }
        )
    return normalized


def _normalize_emissions(value, default_fps):
    normalized = []
    seen_ids = set()
    for index, row in enumerate(_flatten_rows(value, "emissions")):
        label = f"emissions[{index}]"
        emission_id = _first_present(
            row,
            ("emission_id", "event_id", "id"),
            f"{label}.emission_id",
            False,
        )
        if emission_id is None:
            emission_id = f"emission:{index}"
        emission_id = _scalar_id(emission_id, f"{label}.emission_id")
        identity = (type(emission_id).__name__, repr(emission_id))
        if identity in seen_ids:
            raise OnlineInstanceInputError(f"duplicate emission ID {emission_id!r}")
        seen_ids.add(identity)
        sequence = _first_present(
            row,
            ("sequence_id", "sequence", "emission_index"),
            f"{label}.sequence",
            False,
        )
        normalized.append(
            {
                "index": index,
                "id": emission_id,
                "video_id": _video_id(row, label),
                "stream_key": _stream_key(row, label),
                "label": _first_present(
                    row, ("label", "class", "class_id"), f"{label}.label"
                ),
                "segment": _segment(row, label, default_fps),
                "emit_frame": _finite_number(
                    _first_present(row, ("emit_frame",), f"{label}.emit_frame"),
                    f"{label}.emit_frame",
                ),
                "source_frame": _finite_number(
                    _first_present(row, ("source_frame",), f"{label}.source_frame"),
                    f"{label}.source_frame",
                ),
                "sequence": (
                    None
                    if sequence is None
                    else _finite_number(sequence, f"{label}.sequence")
                ),
                "immutable": row.get("immutable"),
            }
        )
    return normalized


def _protocol_audit(rows):
    violations = []
    last_emit_by_stream = {}
    last_sequence_by_stream = {}
    sequence_mode_by_stream = {}

    def record(code, row):
        violations.append(
            {
                "code": code,
                "stream_key": row["stream_key"],
                "row_index": row["index"],
                "emission_id": row["id"],
            }
        )

    for row in rows:
        stream_key = row["stream_key"]
        if row["immutable"] is not True:
            record("mutable_emission", row)
        if row["segment"][1] > row["emit_frame"] + _EPSILON:
            record("future_predicted_end", row)
        if row["source_frame"] > row["emit_frame"] + _EPSILON:
            record("future_source_frame", row)

        previous_emit = last_emit_by_stream.get(stream_key)
        if previous_emit is not None and row["emit_frame"] < previous_emit - _EPSILON:
            record("non_monotonic_emit_frame", row)
        last_emit_by_stream[stream_key] = row["emit_frame"]

        has_sequence = row["sequence"] is not None
        if stream_key not in sequence_mode_by_stream:
            sequence_mode_by_stream[stream_key] = has_sequence
        elif sequence_mode_by_stream[stream_key] != has_sequence:
            raise OnlineInstanceInputError(
                f"stream {stream_key!r} must use an explicit sequence on every row or none"
            )
        if has_sequence:
            previous_sequence = last_sequence_by_stream.get(stream_key)
            if previous_sequence is not None and row["sequence"] <= previous_sequence:
                record("non_monotonic_sequence", row)
            last_sequence_by_stream[stream_key] = row["sequence"]

    counts = Counter(violation["code"] for violation in violations)
    violation_names = (
        "mutable_emission",
        "future_predicted_end",
        "future_source_frame",
        "non_monotonic_emit_frame",
        "non_monotonic_sequence",
    )
    summary = {
        "passed": not violations,
        "num_emissions": len(rows),
        "num_streams": len({row["stream_key"] for row in rows}),
        "violation_counts": {name: int(counts.get(name, 0)) for name in violation_names},
        "violations": violations,
        "monotonicity_definition": (
            "Rows retain input order; emit_frame must be nondecreasing per stream and "
            "an explicit sequence, when present, must be strictly increasing."
        ),
        "no_silent_correction": True,
    }
    return summary, violations


def audit_causal_emissions(emissions, fps=30.0):
    """Return a causal protocol audit without filtering or reordering rows."""

    rows = _normalize_emissions(emissions, fps)
    summary, _ = _protocol_audit(rows)
    return summary


def validate_causal_emissions(emissions, fps=30.0):
    """Validate immutable causal emissions and return the passing audit summary."""

    rows = _normalize_emissions(emissions, fps)
    summary, violations = _protocol_audit(rows)
    if violations:
        raise OnlineInstanceProtocolError(violations)
    return summary


def _temporal_iou(left, right):
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values):
    values = [float(value) for value in values]
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
    }


def _rate(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _trace_rows(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        rows = list(value)
    elif isinstance(value, dict):
        rows = _flatten_stream_mapping(value)
    else:
        raise OnlineInstanceInputError("lifecycle_traces must be a sequence or stream mapping")
    if not all(isinstance(row, dict) for row in rows):
        raise OnlineInstanceInputError("lifecycle trace snapshots must be JSON objects")
    return [dict(row) for row in rows]


def _snapshot_bindings(row, label):
    raw_bindings = _first_present(
        row,
        ("slots", "bindings", "canonical_bindings"),
        f"{label}.slots",
    )
    if isinstance(raw_bindings, dict):
        pairs = list(raw_bindings.items())
    elif isinstance(raw_bindings, (list, tuple)):
        pairs = []
        for index, binding in enumerate(raw_bindings):
            if not isinstance(binding, dict):
                raise OnlineInstanceInputError(f"{label}.slots[{index}] must be an object")
            slot_id = _first_present(
                binding, ("slot_id", "slot"), f"{label}.slots[{index}].slot_id"
            )
            target_id = _first_present(
                binding,
                ("target_id", "instance_id", "gt_id"),
                f"{label}.slots[{index}].target_id",
                False,
            )
            pairs.append((slot_id, target_id))
    else:
        raise OnlineInstanceInputError(f"{label}.slots must be a mapping or sequence")

    slot_to_target = {}
    target_to_slot = {}
    for slot_id, target_id in pairs:
        slot_id = _scalar_id(slot_id, f"{label}.slot_id")
        if target_id is None:
            continue
        target_id = _scalar_id(target_id, f"{label}.target_id")
        slot_key = (type(slot_id).__name__, repr(slot_id))
        target_key = (type(target_id).__name__, repr(target_id))
        if slot_key in slot_to_target:
            raise OnlineInstanceInputError(f"{label} repeats slot {slot_id!r}")
        if target_key in target_to_slot:
            raise OnlineInstanceInputError(f"{label} binds target {target_id!r} more than once")
        slot_to_target[slot_key] = target_key
        target_to_slot[target_key] = slot_key
    return target_to_slot


def compute_lifecycle_trace_metrics(lifecycle_traces):
    """Measure target-to-slot continuity across consecutive trace prefixes."""

    rows = _trace_rows(lifecycle_traces)
    if not rows:
        return {"provided": False}

    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        label = f"lifecycle_traces[{index}]"
        stream_key = _stream_key(row, label)
        prefix_frame = _finite_number(
            _first_present(
                row,
                ("prefix_frame", "frame", "emit_frame"),
                f"{label}.prefix_frame",
            ),
            f"{label}.prefix_frame",
        )
        grouped[stream_key].append(
            {
                "index": index,
                "prefix_frame": prefix_frame,
                "target_to_slot": _snapshot_bindings(row, label),
            }
        )

    opportunities = 0
    survival_count = 0
    shuffle_count = 0
    for stream_key, snapshots in grouped.items():
        previous_frame = None
        for snapshot in snapshots:
            if previous_frame is not None and snapshot["prefix_frame"] <= previous_frame:
                raise OnlineInstanceInputError(
                    f"lifecycle prefixes must increase in stream {stream_key!r}"
                )
            previous_frame = snapshot["prefix_frame"]
        for previous, current in zip(snapshots, snapshots[1:]):
            shared_targets = set(previous["target_to_slot"]).intersection(
                current["target_to_slot"]
            )
            opportunities += len(shared_targets)
            for target in shared_targets:
                if previous["target_to_slot"][target] == current["target_to_slot"][target]:
                    survival_count += 1
                else:
                    shuffle_count += 1

    denominator = {
        "name": "targets_bound_in_consecutive_prefixes",
        "value": opportunities,
    }
    return {
        "provided": True,
        "num_streams": len(grouped),
        "num_prefixes": len(rows),
        "slot_survival": {
            "count": survival_count,
            "rate": _rate(survival_count, opportunities),
            "denominator": dict(denominator),
        },
        "binding_shuffle": {
            "count": shuffle_count,
            "rate": _rate(shuffle_count, opportunities),
            "denominator": dict(denominator),
        },
        "definition": (
            "A survival keeps one target on the same slot across consecutive prefixes; "
            "a shuffle keeps the target present but changes its slot ID."
        ),
    }


def _global_primary_assignment(targets, predictions, threshold):
    matches = {}
    target_groups = defaultdict(list)
    prediction_groups = defaultdict(list)

    def group_key(row):
        return (
            row["video_id"],
            type(row["label"]).__name__,
            repr(row["label"]),
        )

    for target in targets:
        target_groups[group_key(target)].append(target)
    for prediction in predictions:
        prediction_groups[group_key(prediction)].append(prediction)

    for key in sorted(set(target_groups).union(prediction_groups)):
        group_targets = sorted(target_groups[key], key=lambda row: row["index"])
        group_predictions = sorted(
            prediction_groups[key],
            key=lambda row: row["index"],
        )
        if not group_targets or not group_predictions:
            continue
        num_predictions = len(group_predictions)
        num_targets = len(group_targets)
        cardinality_bonus = float(max(num_predictions, num_targets) + 1)
        rewards = np.zeros(
            (num_predictions, num_targets + num_predictions),
            dtype=np.float64,
        )
        rewards[:, :num_targets] = -cardinality_bonus
        tiou_matrix = np.zeros(
            (num_predictions, num_targets),
            dtype=np.float64,
        )
        eligible = np.zeros(
            (num_predictions, num_targets),
            dtype=bool,
        )
        for prediction_index, prediction in enumerate(group_predictions):
            for target_index, target in enumerate(group_targets):
                tiou = _temporal_iou(
                    prediction["segment"],
                    target["segment"],
                )
                tiou_matrix[prediction_index, target_index] = tiou
                if tiou + _EPSILON >= threshold:
                    eligible[prediction_index, target_index] = True
                    rewards[prediction_index, target_index] = (
                        cardinality_bonus + tiou
                    )
        row_indexes, column_indexes = linear_sum_assignment(-rewards)
        for prediction_index, target_index in zip(
            row_indexes.tolist(),
            column_indexes.tolist(),
        ):
            if target_index >= num_targets:
                continue
            if not eligible[prediction_index, target_index]:
                continue
            prediction = group_predictions[prediction_index]
            target = group_targets[target_index]
            matches[prediction["index"]] = (
                target,
                float(tiou_matrix[prediction_index, target_index]),
            )
    return matches


def _covered_components(target_segment, assignments):
    clipped = []
    for item in assignments:
        start = max(
            float(target_segment[0]),
            float(item["prediction"]["segment"][0]),
        )
        end = min(
            float(target_segment[1]),
            float(item["prediction"]["segment"][1]),
        )
        if end > start + _EPSILON:
            clipped.append((start, end))
    components = []
    for start, end in sorted(set(clipped)):
        if not components or start > components[-1][1] + _EPSILON:
            components.append([start, end])
        else:
            components[-1][1] = max(components[-1][1], end)
    return components


def compute_online_instance_metrics(
    ground_truth,
    emissions,
    tiou_threshold=0.5,
    lifecycle_traces=None,
    fps=30.0,
):
    """Compute frozen v2 causal instance metrics over every final emission."""

    threshold = _finite_number(tiou_threshold, "tiou_threshold")
    if threshold <= 0 or threshold > 1:
        raise OnlineInstanceInputError("tiou_threshold must be in (0, 1]")
    default_fps = _finite_number(fps, "fps")
    if default_fps <= 0:
        raise OnlineInstanceInputError("fps must be positive")

    targets = _normalize_ground_truth(ground_truth, default_fps)
    predictions = _normalize_emissions(emissions, default_fps)
    causal_validation, violations = _protocol_audit(predictions)
    if violations:
        raise OnlineInstanceProtocolError(violations)

    primary_by_prediction = _global_primary_assignment(
        targets,
        predictions,
        threshold,
    )
    assignments_by_target = defaultdict(list)
    primary_pairs = []
    emission_assignments = []
    unmatched_emission_ids = []
    pair_by_target = {}

    for prediction in predictions:
        primary = primary_by_prediction.get(prediction["index"])
        if primary is not None:
            target, tiou = primary
            role = "primary_match"
            pair = {
                "ground_truth_id": target["id"],
                "emission_id": prediction["id"],
                "video_id": target["video_id"],
                "runtime_stream_key": prediction["stream_key"],
                "label": target["label"],
                "tiou": tiou,
                "endpoint_latency_frames": (
                    prediction["emit_frame"] - target["segment"][1]
                ),
            }
            primary_pairs.append(pair)
            pair_by_target[target["index"]] = pair
        else:
            candidates = []
            for target in targets:
                if prediction["video_id"] != target["video_id"]:
                    continue
                if prediction["label"] != target["label"]:
                    continue
                tiou = _temporal_iou(
                    prediction["segment"],
                    target["segment"],
                )
                if tiou + _EPSILON >= threshold:
                    candidates.append((tiou, target))
            if candidates:
                tiou, target = min(
                    candidates,
                    key=lambda item: (-item[0], item[1]["index"]),
                )
                role = "duplicate"
            else:
                target = None
                tiou = None
                role = "unmatched"

        emission_assignments.append(
            {
                "emission_id": prediction["id"],
                "status": role,
                "ground_truth_id": None if target is None else target["id"],
                "tiou": tiou,
            }
        )
        if target is None:
            unmatched_emission_ids.append(prediction["id"])
        else:
            assignments_by_target[target["index"]].append(
                {
                    "prediction": prediction,
                    "role": role,
                    "tiou": tiou,
                }
            )

    primary_pairs.sort(
        key=lambda row: (
            row["video_id"],
            type(row["ground_truth_id"]).__name__,
            repr(row["ground_truth_id"]),
        )
    )
    duplicate_count = 0
    duplicate_per_target = []
    fragmented_targets = []
    covered_component_count = 0
    excess_fragment_count = 0
    for target in targets:
        pair = pair_by_target.get(target["index"])
        if pair is None:
            continue
        assignments = assignments_by_target[target["index"]]
        duplicate_ids = [
            item["prediction"]["id"]
            for item in assignments
            if item["role"] == "duplicate"
        ]
        duplicate_count += len(duplicate_ids)
        duplicate_per_target.append(
            {
                "ground_truth_id": target["id"],
                "primary_emission_id": pair["emission_id"],
                "duplicate_count": len(duplicate_ids),
                "duplicate_emission_ids": duplicate_ids,
            }
        )
        components = _covered_components(target["segment"], assignments)
        if len(components) >= 2:
            fragmented_targets.append(
                {
                    "ground_truth_id": target["id"],
                    "covered_component_count": len(components),
                    "components": components,
                }
            )
            covered_component_count += len(components)
            excess_fragment_count += len(components) - 1

    matched_count = len(primary_pairs)
    emission_count = len(predictions)
    unmatched_count = len(unmatched_emission_ids)
    identity_denominator = {"name": "all_ground_truth", "value": len(targets)}
    endpoint_latencies = [
        pair["endpoint_latency_frames"] for pair in primary_pairs
    ]
    duplicate_rate = _rate(duplicate_count, len(targets))
    fragmentation_rate = _rate(excess_fragment_count, len(targets))
    fragmented_target_rate = _rate(len(fragmented_targets), len(targets))
    false_emission_rate = _rate(unmatched_count, emission_count)

    return {
        "schema_version": SCHEMA_VERSION,
        "causal_validation": causal_validation,
        "counts": {
            "ground_truth": len(targets),
            "emissions": emission_count,
            "matched_ground_truth": matched_count,
            "unmatched_ground_truth": len(targets) - matched_count,
            "primary_matches": matched_count,
            "duplicate_emissions": duplicate_count,
            "unmatched_emissions": unmatched_count,
        },
        "matching": {
            "policy": "global_max_cardinality_max_total_tiou",
            "identity_key": "video_id",
            "protocol_stream_key": "runtime_stream_key",
            "coordinate_system": "frames",
            "tiou_threshold": threshold,
            "tie_breaking": (
                "Stable prediction and ground-truth input order before "
                "deterministic SciPy linear assignment."
            ),
            "pairs": primary_pairs,
            "emission_assignments": emission_assignments,
            "no_nms_or_output_correction": True,
        },
        "duplicates": {
            "duplicate_emission_count": duplicate_count,
            "rate": duplicate_rate,
            "denominator": identity_denominator,
            "per_matched_ground_truth": duplicate_per_target,
        },
        "fragmentation": {
            "fragmented_ground_truth_count": len(fragmented_targets),
            "covered_component_count": covered_component_count,
            "excess_fragment_count": excess_fragment_count,
            "rate": fragmentation_rate,
            "fragmented_target_rate": fragmented_target_rate,
            "denominator": identity_denominator,
            "definition": (
                "Fragmentation is the number of disjoint connected components "
                "in assigned prediction coverage clipped to each ground-truth "
                "interval, minus one; nested or overlapping bounds stay one "
                "component."
            ),
            "per_ground_truth": fragmented_targets,
        },
        "false_emissions": {
            "unmatched_emission_count": unmatched_count,
            "rate": false_emission_rate,
            "denominator": {"name": "all_emissions", "value": emission_count},
            "emission_ids": unmatched_emission_ids,
        },
        "endpoint_detection_latency_frames": _summary(endpoint_latencies),
        "lifecycle": compute_lifecycle_trace_metrics(lifecycle_traces),
        "duplicate_rate": duplicate_rate,
        "fragmentation_rate": fragmentation_rate,
        "false_emission_rate": false_emission_rate,
    }


__all__ = [
    "OnlineInstanceInputError",
    "OnlineInstanceMetricError",
    "OnlineInstanceProtocolError",
    "audit_causal_emissions",
    "compute_online_instance_metrics",
    "compute_lifecycle_trace_metrics",
    "validate_causal_emissions",
]
