"""Auditable instance metrics for immutable Full PETAL emissions.

The evaluator intentionally performs no NMS, segment merging, score filtering,
or event reordering. Input order is the online chronology used for matching.
"""

from collections import Counter, defaultdict
import math


SCHEMA_VERSION = "online_instance_metrics.v1"
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
    if not any(key in copied for key in ("stream_key", "stream_id", "video_id")):
        copied["stream_key"] = stream_key
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
                raise OnlineInstanceInputError(f"{kind} rows must be a sequence or stream mapping")
    else:
        raise OnlineInstanceInputError(f"{kind} must be a sequence or JSON object")
    if not all(isinstance(row, dict) for row in rows):
        raise OnlineInstanceInputError(f"all {kind} rows must be JSON objects")
    return [dict(row) for row in rows]


def _segment(row, label):
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
    else:
        raise OnlineInstanceInputError(f"{label} requires frame bounds or a segment")
    if end <= start:
        raise OnlineInstanceInputError(f"{label} segment end must be greater than start")
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
            raise OnlineInstanceInputError(
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


def compute_online_instance_metrics(
    ground_truth,
    emissions,
    tiou_threshold=0.5,
    lifecycle_traces=None,
):
    """Compute auditable Full PETAL metrics over every immutable emission."""

    threshold = _finite_number(tiou_threshold, "tiou_threshold")
    if threshold <= 0 or threshold > 1:
        raise OnlineInstanceInputError("tiou_threshold must be in (0, 1]")

    targets = _normalize_ground_truth(ground_truth)
    predictions = _normalize_emissions(emissions)
    causal_validation, violations = _protocol_audit(predictions)
    if violations:
        raise OnlineInstanceProtocolError(violations)

    matched_target_indexes = set()
    assignments_by_target = defaultdict(list)
    primary_pairs = []
    emission_assignments = []
    unmatched_emission_ids = []

    for prediction in predictions:
        candidates = []
        for target in targets:
            if prediction["stream_key"] != target["stream_key"]:
                continue
            if prediction["label"] != target["label"]:
                continue
            tiou = _temporal_iou(prediction["segment"], target["segment"])
            if tiou + _EPSILON >= threshold:
                candidates.append((tiou, target))

        available = [
            candidate
            for candidate in candidates
            if candidate[1]["index"] not in matched_target_indexes
        ]
        if available:
            tiou, target = min(available, key=lambda item: (-item[0], item[1]["index"]))
            matched_target_indexes.add(target["index"])
            role = "primary_match"
            latency = prediction["emit_frame"] - target["segment"][1]
            pair = {
                "ground_truth_id": target["id"],
                "emission_id": prediction["id"],
                "stream_key": target["stream_key"],
                "label": target["label"],
                "tiou": tiou,
                "endpoint_latency_frames": latency,
            }
            primary_pairs.append(pair)
        elif candidates:
            tiou, target = min(candidates, key=lambda item: (-item[0], item[1]["index"]))
            role = "duplicate"
        else:
            target = None
            tiou = None
            role = "unmatched"

        assignment = {
            "emission_id": prediction["id"],
            "status": role,
            "ground_truth_id": None if target is None else target["id"],
            "tiou": tiou,
        }
        emission_assignments.append(assignment)
        if target is None:
            unmatched_emission_ids.append(prediction["id"])
        else:
            assignments_by_target[target["index"]].append(
                {"prediction": prediction, "role": role, "tiou": tiou}
            )

    duplicate_count = 0
    duplicate_per_target = []
    fragmented_targets = []
    distinct_fragment_count = 0
    excess_fragment_count = 0
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
        duplicate_count += len(duplicate_ids)
        duplicate_per_target.append(
            {
                "ground_truth_id": target["id"],
                "primary_emission_id": pair["emission_id"],
                "duplicate_count": len(duplicate_ids),
                "duplicate_emission_ids": duplicate_ids,
            }
        )

        distinct_segments = []
        seen_segments = set()
        for item in assignments:
            segment = item["prediction"]["segment"]
            if segment not in seen_segments:
                seen_segments.add(segment)
                distinct_segments.append(list(segment))
        if len(distinct_segments) >= 2:
            fragmented_targets.append(
                {
                    "ground_truth_id": target["id"],
                    "distinct_fragment_count": len(distinct_segments),
                    "segments": distinct_segments,
                }
            )
            distinct_fragment_count += len(distinct_segments)
            excess_fragment_count += len(distinct_segments) - 1

    matched_count = len(primary_pairs)
    emission_count = len(predictions)
    unmatched_count = len(unmatched_emission_ids)
    fragmentation_denominator = {
        "name": "matched_ground_truth",
        "value": matched_count,
    }
    endpoint_latencies = [pair["endpoint_latency_frames"] for pair in primary_pairs]
    duplicate_rate = _rate(duplicate_count, emission_count)
    fragmentation_rate = _rate(len(fragmented_targets), matched_count)
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
            "policy": "chronological_greedy_class_aware_tiou",
            "coordinate_system": "frames",
            "tiou_threshold": threshold,
            "tie_breaking": (
                "Input emission order, then highest tIoU among unmatched GT, then GT input order; "
                "later matches to locked GT are duplicates."
            ),
            "pairs": primary_pairs,
            "emission_assignments": emission_assignments,
            "no_nms_or_output_correction": True,
        },
        "duplicates": {
            "duplicate_emission_count": duplicate_count,
            "rate": duplicate_rate,
            "denominator": {"name": "all_emissions", "value": emission_count},
            "per_matched_ground_truth": duplicate_per_target,
        },
        "fragmentation": {
            "fragmented_ground_truth_count": len(fragmented_targets),
            "distinct_fragment_count": distinct_fragment_count,
            "excess_fragment_count": excess_fragment_count,
            "rate": fragmentation_rate,
            "denominator": fragmentation_denominator,
            "definition": (
                "A matched GT is fragmented when its assigned emissions contain at least two "
                "distinct segment bounds, whether disjoint or overlapping; exact repeated bounds "
                "remain duplicates but are not an additional fragment."
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
