"""Auditable instance metrics for immutable Full PETAL emissions.

The evaluator intentionally performs no NMS, segment merging, score filtering,
or ledger reordering. Input order is the online chronology used for matching.
"""

from collections import Counter, defaultdict
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


SCHEMA_VERSION = "full_petal_metrics.v2"
_EPSILON = 1e-9


class FullPetalMetricError(ValueError):
    """Base error for malformed Full PETAL metric inputs."""


class FullPetalInputError(FullPetalMetricError):
    """Raised when rows do not satisfy the metric input schema."""


class FullPetalProtocolError(FullPetalMetricError):
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
        raise FullPetalInputError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FullPetalInputError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise FullPetalInputError(f"{label} must be a finite number")
    return number


def _scalar_id(value, label):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise FullPetalInputError(f"{label} must be a JSON scalar identifier")
    if isinstance(value, float) and not math.isfinite(value):
        raise FullPetalInputError(f"{label} must be finite")
    if isinstance(value, str) and not value:
        raise FullPetalInputError(f"{label} must not be empty")
    return value


def _first_present(row, keys, label, required=True):
    present = [key for key in keys if key in row]
    if not present:
        if required:
            raise FullPetalInputError(f"{label} is required")
        return None
    value = row[present[0]]
    for key in present[1:]:
        if row[key] != value:
            raise FullPetalInputError(
                f"conflicting aliases for {label}: {present[0]} and {key}"
            )
    return value


def _inject_stream(row, stream_key):
    if not isinstance(row, dict):
        raise FullPetalInputError("metric rows must be JSON objects")
    copied = dict(row)
    if not any(key in copied for key in ("stream_key", "stream_id", "video_id")):
        copied["stream_key"] = stream_key
    return copied


def _flatten_stream_mapping(mapping):
    rows = []
    for stream_key, stream_rows in mapping.items():
        if not isinstance(stream_rows, (list, tuple)):
            raise FullPetalInputError(
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
                raise FullPetalInputError("ground-truth database must be an object")
            rows = []
            for stream_key, video in database.items():
                if not isinstance(video, dict):
                    raise FullPetalInputError("ground-truth video records must be objects")
                rows.extend(
                    _inject_stream(row, stream_key)
                    for row in video.get("annotations", ())
                )
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
                raise FullPetalInputError(f"{kind} rows must be a sequence or stream mapping")
    else:
        raise FullPetalInputError(f"{kind} must be a sequence or JSON object")
    if not all(isinstance(row, dict) for row in rows):
        raise FullPetalInputError(f"all {kind} rows must be JSON objects")
    return [dict(row) for row in rows]


def _segment(row, label):
    start_keys = ("start_frame", "predicted_start_frame")
    end_keys = ("end_frame", "predicted_end_frame")
    has_start_frame = any(key in row for key in start_keys)
    has_end_frame = any(key in row for key in end_keys)
    if has_start_frame or has_end_frame:
        if not has_start_frame or not has_end_frame:
            raise FullPetalInputError(
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
            raise FullPetalInputError(f"{label}.segment must contain [start, end]")
        start = _finite_number(segment[0], f"{label}.segment[0]")
        end = _finite_number(segment[1], f"{label}.segment[1]")
    else:
        raise FullPetalInputError(f"{label} requires frame bounds or a segment")
    if end <= start:
        raise FullPetalInputError(f"{label} segment end must be greater than start")
    return (start, end)


def _stream_key(row, label):
    value = _first_present(
        row,
        ("stream_key", "stream_id", "video_id"),
        f"{label}.stream_key",
    )
    return str(_scalar_id(value, f"{label}.stream_key"))


def _normalize_ground_truth(value):
    normalized = []
    seen_ids = set()
    for index, row in enumerate(_flatten_rows(value, "ground truth")):
        label = f"ground_truth[{index}]"
        gt_id = _first_present(row, ("gt_id", "instance_id", "id"), f"{label}.gt_id", False)
        if gt_id is None:
            gt_id = f"gt:{index}"
        gt_id = _scalar_id(gt_id, f"{label}.gt_id")
        stream_key = _stream_key(row, label)
        identity = (stream_key, type(gt_id).__name__, repr(gt_id))
        if identity in seen_ids:
            raise FullPetalInputError(
                f"duplicate ground-truth ID {gt_id!r} in stream {stream_key!r}"
            )
        seen_ids.add(identity)
        target_label = _first_present(row, ("label", "class", "class_id"), f"{label}.label")
        normalized.append(
            {
                "index": index,
                "id": gt_id,
                "stream_key": stream_key,
                "label": target_label,
                "segment": _segment(row, label),
            }
        )
    return normalized


def _normalize_emissions(value):
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
            raise FullPetalInputError(f"duplicate emission ID {emission_id!r}")
        seen_ids.add(identity)
        sequence = _first_present(
            row,
            ("sequence_id", "sequence", "emission_index"),
            f"{label}.sequence",
            False,
        )
        row_fps = _first_present(row, ("fps",), f"{label}.fps", False)
        if row_fps is not None:
            row_fps = _finite_number(row_fps, f"{label}.fps")
            if row_fps <= 0:
                raise FullPetalInputError(f"{label}.fps must be positive")
        normalized.append(
            {
                "index": index,
                "id": emission_id,
                "stream_key": _stream_key(row, label),
                "label": _first_present(
                    row, ("label", "class", "class_id"), f"{label}.label"
                ),
                "segment": _segment(row, label),
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
                "score": _finite_number(
                    _first_present(row, ("score",), f"{label}.score"),
                    f"{label}.score",
                ),
                "fps": row_fps,
                "immutable": row.get("immutable"),
            }
        )
        if not 0.0 <= normalized[-1]["score"] <= 1.0:
            raise FullPetalInputError(f"{label}.score must lie in [0, 1]")
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
            raise FullPetalInputError(
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


def audit_causal_emissions(emissions):
    """Return a causal protocol audit without filtering or reordering rows."""

    rows = _normalize_emissions(emissions)
    summary, _ = _protocol_audit(rows)
    return summary


def validate_causal_emissions(emissions):
    """Validate immutable causal emissions and return the passing audit summary."""

    rows = _normalize_emissions(emissions)
    summary, violations = _protocol_audit(rows)
    if violations:
        raise FullPetalProtocolError(violations)
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


def _stable_id_key(value):
    return (type(value).__name__, repr(value))


def _timely(prediction, target, latency_budget_sec, default_fps):
    latency = prediction["emit_frame"] - target["segment"][1]
    prediction_fps = prediction.get("fps") or default_fps
    return -_EPSILON <= latency <= (
        latency_budget_sec * prediction_fps + _EPSILON
    )


def _maximum_cardinality_tiou_matching(
    targets,
    predictions,
    threshold,
    latency_budget_sec,
    default_fps,
):
    """Solve the preregistered lexicographic diagnostic assignment."""

    pairs = []
    grouped_targets = defaultdict(list)
    grouped_predictions = defaultdict(list)
    for target in targets:
        grouped_targets[(target["stream_key"], target["label"])].append(target)
    for prediction in predictions:
        grouped_predictions[(prediction["stream_key"], prediction["label"])].append(
            prediction
        )

    for group_key in sorted(
        set(grouped_targets).intersection(grouped_predictions),
        key=lambda value: (str(value[0]), _stable_id_key(value[1])),
    ):
        group_targets = sorted(
            grouped_targets[group_key],
            key=lambda row: (_stable_id_key(row["id"]), row["index"]),
        )
        group_predictions = sorted(
            grouped_predictions[group_key],
            key=lambda row: (-row["score"], _stable_id_key(row["id"]), row["index"]),
        )
        num_predictions = len(group_predictions)
        num_targets = len(group_targets)
        if not num_predictions or not num_targets:
            continue

        max_pairs = min(num_predictions, num_targets)
        cardinality_bonus = float(max_pairs + 1)
        invalid_weight = -cardinality_bonus
        weights = np.zeros(
            (num_predictions, num_targets + num_predictions),
            dtype=np.float64,
        )
        weights[:, :num_targets] = invalid_weight
        eligible = {}
        tie_scale = 1e-9 / float(max(1, num_predictions * num_targets))
        for prediction_index, prediction in enumerate(group_predictions):
            for target_index, target in enumerate(group_targets):
                tiou = _temporal_iou(prediction["segment"], target["segment"])
                if tiou + _EPSILON < threshold or not _timely(
                    prediction,
                    target,
                    latency_budget_sec,
                    default_fps,
                ):
                    continue
                eligible[(prediction_index, target_index)] = tiou
                score_tie = prediction["score"] * tie_scale
                id_tie = (
                    (num_predictions - prediction_index)
                    * (num_targets - target_index)
                    * tie_scale
                    * 1e-3
                )
                weights[prediction_index, target_index] = (
                    cardinality_bonus + tiou + score_tie + id_tie
                )

        row_indices, column_indices = linear_sum_assignment(weights, maximize=True)
        for prediction_index, target_index in zip(row_indices, column_indices):
            if (prediction_index, target_index) not in eligible:
                continue
            prediction = group_predictions[prediction_index]
            target = group_targets[target_index]
            pairs.append(
                {
                    "prediction": prediction,
                    "target": target,
                    "tiou": eligible[(prediction_index, target_index)],
                }
            )

    return sorted(pairs, key=lambda item: item["prediction"]["index"])


def _union_length(intervals):
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end + _EPSILON:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def _trace_rows(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        rows = list(value)
    elif isinstance(value, dict):
        rows = _flatten_stream_mapping(value)
    else:
        raise FullPetalInputError("lifecycle_traces must be a sequence or stream mapping")
    if not all(isinstance(row, dict) for row in rows):
        raise FullPetalInputError("lifecycle trace snapshots must be JSON objects")
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
                raise FullPetalInputError(f"{label}.slots[{index}] must be an object")
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
        raise FullPetalInputError(f"{label}.slots must be a mapping or sequence")

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
            raise FullPetalInputError(f"{label} repeats slot {slot_id!r}")
        if target_key in target_to_slot:
            raise FullPetalInputError(f"{label} binds target {target_id!r} more than once")
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
                raise FullPetalInputError(
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


def compute_full_petal_metrics(
    ground_truth,
    emissions,
    tiou_threshold=0.5,
    fps=30.0,
    latency_budget_sec=2.0,
    lifecycle_traces=None,
):
    """Compute the preregistered identity diagnostics over immutable emissions."""

    threshold = _finite_number(tiou_threshold, "tiou_threshold")
    if threshold <= 0 or threshold > 1:
        raise FullPetalInputError("tiou_threshold must be in (0, 1]")
    fps = _finite_number(fps, "fps")
    if fps <= 0:
        raise FullPetalInputError("fps must be positive")
    latency_budget_sec = _finite_number(
        latency_budget_sec,
        "latency_budget_sec",
    )
    if latency_budget_sec < 0:
        raise FullPetalInputError("latency_budget_sec must be non-negative")
    targets = _normalize_ground_truth(ground_truth)
    predictions = _normalize_emissions(emissions)
    for prediction in predictions:
        if prediction["fps"] is None:
            prediction["fps"] = fps
    causal_validation, violations = _protocol_audit(predictions)
    if violations:
        raise FullPetalProtocolError(violations)

    matched_pairs = _maximum_cardinality_tiou_matching(
        targets,
        predictions,
        threshold,
        latency_budget_sec,
        fps,
    )
    matched_target_indexes = {
        item["target"]["index"] for item in matched_pairs
    }
    matched_prediction_indexes = {
        item["prediction"]["index"] for item in matched_pairs
    }
    assignments_by_target = defaultdict(list)
    primary_pairs = []
    assignment_by_prediction = {}
    for item in matched_pairs:
        prediction = item["prediction"]
        target = item["target"]
        pair = {
            "ground_truth_id": target["id"],
            "emission_id": prediction["id"],
            "stream_key": target["stream_key"],
            "label": target["label"],
            "tiou": item["tiou"],
            "endpoint_latency_frames": (
                prediction["emit_frame"] - target["segment"][1]
            ),
        }
        primary_pairs.append(pair)
        assignment_by_prediction[prediction["index"]] = {
            "emission_id": prediction["id"],
            "status": "primary_match",
            "ground_truth_id": target["id"],
            "tiou": item["tiou"],
        }
        assignments_by_target[target["index"]].append(
            {"prediction": prediction, "role": "primary_match", "tiou": item["tiou"]}
        )

    duplicate_prediction_indexes = set()
    for prediction in predictions:
        if prediction["index"] in matched_prediction_indexes:
            continue
        candidates = []
        for target in targets:
            if target["index"] not in matched_target_indexes:
                continue
            if prediction["stream_key"] != target["stream_key"]:
                continue
            if prediction["label"] != target["label"]:
                continue
            tiou = _temporal_iou(prediction["segment"], target["segment"])
            if tiou + _EPSILON >= threshold and _timely(
                prediction,
                target,
                latency_budget_sec,
                fps,
            ):
                candidates.append((tiou, target))
        if not candidates:
            continue
        tiou, target = min(
            candidates,
            key=lambda item: (-item[0], _stable_id_key(item[1]["id"]), item[1]["index"]),
        )
        duplicate_prediction_indexes.add(prediction["index"])
        assignment_by_prediction[prediction["index"]] = {
            "emission_id": prediction["id"],
            "status": "duplicate",
            "ground_truth_id": target["id"],
            "tiou": tiou,
        }
        assignments_by_target[target["index"]].append(
            {"prediction": prediction, "role": "duplicate", "tiou": tiou}
        )

    unmatched_primary_predictions = [
        prediction
        for prediction in predictions
        if prediction["index"] not in matched_prediction_indexes
    ]
    fragment_prediction_indexes = set()
    fragmented_targets = []
    fragment_count = 0
    for target in targets:
        if target["index"] in matched_target_indexes:
            continue
        candidates = []
        intersections = []
        for prediction in unmatched_primary_predictions:
            if prediction["stream_key"] != target["stream_key"]:
                continue
            if prediction["label"] != target["label"]:
                continue
            if not _timely(prediction, target, latency_budget_sec, fps):
                continue
            tiou = _temporal_iou(prediction["segment"], target["segment"])
            intersection = (
                max(prediction["segment"][0], target["segment"][0]),
                min(prediction["segment"][1], target["segment"][1]),
            )
            if tiou + _EPSILON >= threshold or intersection[1] <= intersection[0]:
                continue
            candidates.append((prediction, tiou, intersection))
            intersections.append(intersection)
        if len(candidates) < 2:
            continue
        coverage = _union_length(intersections) / (
            target["segment"][1] - target["segment"][0]
        )
        if coverage + _EPSILON < threshold:
            continue
        fragment_ids = [item[0]["id"] for item in candidates]
        fragment_prediction_indexes.update(item[0]["index"] for item in candidates)
        fragment_count += len(candidates)
        fragmented_targets.append(
            {
                "ground_truth_id": target["id"],
                "fragment_count": len(candidates),
                "fragment_emission_ids": fragment_ids,
                "intersections": [list(item[2]) for item in candidates],
                "union_coverage": coverage,
            }
        )
        for prediction, tiou, _ in candidates:
            assignment = assignment_by_prediction.get(prediction["index"])
            if assignment is None:
                assignment_by_prediction[prediction["index"]] = {
                    "emission_id": prediction["id"],
                    "status": "fragment",
                    "ground_truth_id": target["id"],
                    "tiou": tiou,
                }
            else:
                assignment.setdefault("fragment_ground_truth_ids", []).append(
                    target["id"]
                )

    unmatched_emission_ids = []
    for prediction in predictions:
        if prediction["index"] not in assignment_by_prediction:
            unmatched_emission_ids.append(prediction["id"])
            assignment_by_prediction[prediction["index"]] = {
                "emission_id": prediction["id"],
                "status": "unmatched",
                "ground_truth_id": None,
                "tiou": None,
            }
    emission_assignments = [
        assignment_by_prediction[prediction["index"]] for prediction in predictions
    ]

    duplicate_count = len(duplicate_prediction_indexes)
    duplicate_per_target = []
    for pair in primary_pairs:
        target = next(
            item
            for item in targets
            if item["stream_key"] == pair["stream_key"]
            and item["id"] == pair["ground_truth_id"]
        )
        assignments = assignments_by_target[target["index"]]
        duplicate_ids = [
            item["prediction"]["id"]
            for item in assignments
            if item["role"] == "duplicate"
        ]
        duplicate_per_target.append(
            {
                "ground_truth_id": target["id"],
                "primary_emission_id": pair["emission_id"],
                "duplicate_count": len(duplicate_ids),
                "duplicate_emission_ids": duplicate_ids,
            }
        )

    matched_count = len(primary_pairs)
    emission_count = len(predictions)
    unmatched_count = len(unmatched_emission_ids)
    fragmentation_denominator = {
        "name": "all_ground_truth",
        "value": len(targets),
    }
    endpoint_latencies = [pair["endpoint_latency_frames"] for pair in primary_pairs]
    duplicate_per_gt = _rate(duplicate_count, len(targets))
    duplicate_fraction = _rate(duplicate_count, emission_count)
    duplicate_gt_count = sum(
        int(item["duplicate_count"] > 0) for item in duplicate_per_target
    )
    duplicate_gt_rate = _rate(duplicate_gt_count, len(targets))
    fragmentation_rate = _rate(len(fragmented_targets), len(targets))
    false_emission_rate = _rate(unmatched_count, emission_count)
    recall = _rate(matched_count, len(targets))

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
            "fragment_emissions": len(fragment_prediction_indexes),
            "unmatched_emissions": unmatched_count,
        },
        "matching": {
            "policy": "maximum_cardinality_then_total_tiou",
            "coordinate_system": "frames",
            "tiou_threshold": threshold,
            "latency_budget_sec": latency_budget_sec,
            "default_fps": fps,
            "latency_budget_frame_policy": "latency_budget_sec * emission_fps",
            "tie_breaking": (
                "Maximize cardinality, then total tIoU; numerical ties use descending "
                "prediction score and deterministic emission/GT IDs."
            ),
            "pairs": primary_pairs,
            "emission_assignments": emission_assignments,
            "no_nms_or_output_correction": True,
        },
        "duplicates": {
            "duplicate_emission_count": duplicate_count,
            "duplicate_ground_truth_count": duplicate_gt_count,
            "duplicate_per_gt": duplicate_per_gt,
            "duplicate_fraction": duplicate_fraction,
            "duplicate_gt_rate": duplicate_gt_rate,
            "denominators": {
                "duplicate_per_gt": {
                    "name": "all_ground_truth",
                    "value": len(targets),
                },
                "duplicate_fraction": {
                    "name": "all_emissions",
                    "value": emission_count,
                },
            },
            "per_matched_ground_truth": duplicate_per_target,
        },
        "fragmentation": {
            "fragmented_ground_truth_count": len(fragmented_targets),
            "fragment_count": fragment_count,
            "rate": fragmentation_rate,
            "denominator": fragmentation_denominator,
            "definition": (
                "An unmatched GT is fragmented when at least two timely, same-class primary-"
                "unmatched "
                "emissions each have positive overlap but are individually below the tIoU "
                "threshold, while their intersection union covers at least that fraction of GT."
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
        "recall": recall,
        "duplicate_per_gt": duplicate_per_gt,
        "duplicate_fraction": duplicate_fraction,
        "duplicate_gt_rate": duplicate_gt_rate,
        "fragmentation_rate": fragmentation_rate,
        "false_emission_rate": false_emission_rate,
    }


__all__ = [
    "FullPetalInputError",
    "FullPetalMetricError",
    "FullPetalProtocolError",
    "audit_causal_emissions",
    "compute_full_petal_metrics",
    "compute_lifecycle_trace_metrics",
    "validate_causal_emissions",
]
