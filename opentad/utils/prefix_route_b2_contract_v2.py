"""Executable temporal-MOTR attack contract for Prefix-Route Protocol V2."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import hashlib
import json
import math


B2_CONTRACT_SCHEMA = "prefix-route-temporal-motr-b2-v2"
B2_STREAM_STATE_SCHEMA = "prefix-route-b2-stream-state-v2"
B2_STREAM_TRANSCRIPT_SCHEMA = "prefix-route-b2-stream-transcript-v2"
TRACK_CAPACITY = 64
NEWBORN_QUERY_COUNT = 64
MAX_DECODER_QUERY_COUNT = TRACK_CAPACITY + NEWBORN_QUERY_COUNT
THRESHOLD_GRID = tuple(index / 20.0 for index in range(1, 20))
DUSTBIN_COST = 10.0
FEATURE_STRIDE_FRAMES = 8
_COST_SCALE = 1_000_000
_GENESIS_STATE_SHA256 = "0" * 64


class PrefixRouteB2Error(ValueError):
    pass


def _finite(value, label):
    if isinstance(value, bool):
        raise PrefixRouteB2Error(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PrefixRouteB2Error(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise PrefixRouteB2Error(f"{label} must be finite")
    return parsed


def _unit(value, label):
    parsed = _finite(value, label)
    if not 0.0 <= parsed <= 1.0:
        raise PrefixRouteB2Error(f"{label} must lie in [0, 1]")
    return parsed


def _nonnegative_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrefixRouteB2Error(f"{label} must be a non-negative integer")
    return value


def _identifier(value, label):
    if not isinstance(value, str) or not value:
        raise PrefixRouteB2Error(f"{label} must be non-empty text")
    return value


def _canonical_json_bytes(value):
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


def _state_sha256(value):
    unsigned = dict(value)
    unsigned.pop("state_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def _quantized_cost_units(value, label):
    cost = _finite(value, label)
    if cost < 0:
        raise PrefixRouteB2Error(f"{label} must be non-negative")
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as exc:
        raise PrefixRouteB2Error(f"{label} must be a decimal number") from exc
    quantized = decimal.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
    if decimal != quantized:
        raise PrefixRouteB2Error(
            f"{label} must be quantized to 1e-6"
        )
    return int(quantized * _COST_SCALE)


def _hungarian_rectangular_integer(costs):
    """Return the minimum assignment for an integer n-by-m matrix, n <= m."""

    if not costs or not costs[0]:
        raise PrefixRouteB2Error("assignment matrix must be non-empty")
    row_count = len(costs)
    column_count = len(costs[0])
    if row_count > column_count:
        raise PrefixRouteB2Error("assignment matrix must have at least as many columns")
    if any(len(row) != column_count for row in costs):
        raise PrefixRouteB2Error("assignment matrix rows differ in width")

    u = [0] * (row_count + 1)
    v = [0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row_index in range(1, row_count + 1):
        matched_row[0] = row_index
        minimum = [None] * (column_count + 1)
        used = [False] * (column_count + 1)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = None
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[current_row - 1][candidate - 1]
                    - u[current_row]
                    - v[candidate]
                )
                if minimum[candidate] is None or reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if (
                    delta is None
                    or minimum[candidate] < delta
                    or (
                        minimum[candidate] == delta
                        and candidate < next_column
                    )
                ):
                    delta = minimum[candidate]
                    next_column = candidate
            if delta is None:
                raise PrefixRouteB2Error("assignment matrix has no complete matching")
            for candidate in range(column_count + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                elif minimum[candidate] is not None:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column]:
            assignment[matched_row[column] - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise PrefixRouteB2Error("assignment matrix produced an incomplete matching")
    return assignment


def _lexicographic_exact_assignment(primary_costs):
    """Minimize primary integer cost, then the row-wise column vector exactly."""

    row_count = len(primary_costs)
    column_count = len(primary_costs[0])
    base = column_count + 1
    primary_multiplier = base ** row_count
    row_weights = [
        base ** (row_count - row_index - 1)
        for row_index in range(row_count)
    ]
    combined = [
        [
            primary_costs[row_index][column_index] * primary_multiplier
            + column_index * row_weights[row_index]
            for column_index in range(column_count)
        ]
        for row_index in range(row_count)
    ]
    return _hungarian_rectangular_integer(combined)


def validate_calibrated_threshold(value):
    threshold = _unit(value, "calibrated threshold")
    if threshold not in THRESHOLD_GRID:
        raise PrefixRouteB2Error("calibrated threshold is outside the frozen grid")
    return threshold


def instance_aware_risk_set(
    instances,
    *,
    previous_observation_count,
    decision_observation_count,
    emitted_instance_ids=(),
):
    """Derive prefix targets without dropping completed but not-yet-emitted GT."""

    previous = _nonnegative_int(
        previous_observation_count,
        "previous_observation_count",
    )
    current = _nonnegative_int(
        decision_observation_count,
        "decision_observation_count",
    )
    if current <= previous:
        raise PrefixRouteB2Error("decision observations must advance")
    emitted_rows = tuple(emitted_instance_ids)
    emitted = set(emitted_rows)
    if len(emitted) != len(emitted_rows):
        raise PrefixRouteB2Error("emitted instance IDs are duplicated")
    rows = []
    seen = set()
    for instance in instances:
        required = {
            "instance_id",
            "start_frame",
            "end_observation_count",
            "label",
        }
        if not isinstance(instance, dict) or set(instance) != required:
            raise PrefixRouteB2Error("risk-set instance fields differ")
        instance_id = _identifier(instance["instance_id"], "instance_id")
        if instance_id in seen:
            raise PrefixRouteB2Error("risk-set instance IDs are duplicated")
        seen.add(instance_id)
        start = _nonnegative_int(instance["start_frame"], "start_frame")
        end = _nonnegative_int(
            instance["end_observation_count"],
            "end_observation_count",
        )
        if end <= start:
            raise PrefixRouteB2Error("risk-set instance interval is invalid")
        _identifier(instance["label"], "instance label")
        if instance_id in emitted or start >= current:
            continue
        completed_now = end <= current
        rows.append(
            {
                **instance,
                "active_at_decision": start < current < end,
                "completion_target": int(completed_now),
                "first_emission_target": int(completed_now),
                "newly_observable": previous <= start < current,
                "delayed_unemitted_completion": end <= previous,
            }
        )
    return sorted(rows, key=lambda row: (row["start_frame"], row["instance_id"]))


def build_temporal_tracklet_assignment(
    *,
    risk_set,
    previous_track_assignments,
    newborn_costs,
):
    """Lock surviving track identities, then match newborn slots one-to-one."""

    if not isinstance(previous_track_assignments, dict):
        raise PrefixRouteB2Error("previous track assignments must be an object")
    if len(previous_track_assignments) > TRACK_CAPACITY:
        raise PrefixRouteB2Error("previous track assignment capacity is exceeded")
    by_instance = {row["instance_id"]: row for row in risk_set}
    if len(by_instance) != len(risk_set):
        raise PrefixRouteB2Error("risk-set instance IDs are duplicated")
    locked = {}
    used_instances = set()
    for track_id, instance_id in sorted(previous_track_assignments.items()):
        _identifier(track_id, "track_id")
        _identifier(instance_id, "assigned instance_id")
        if instance_id in by_instance:
            if instance_id in used_instances:
                raise PrefixRouteB2Error(
                    "multiple propagated tracks target one instance"
                )
            locked[track_id] = instance_id
            used_instances.add(instance_id)
        else:
            locked[track_id] = "DUSTBIN"
    if not isinstance(newborn_costs, dict) or set(newborn_costs) != set(
        range(NEWBORN_QUERY_COUNT)
    ):
        raise PrefixRouteB2Error("newborn cost rows must cover slots 0..63")
    available_instances = sorted(set(by_instance) - used_instances)
    newborn_assignment = {
        slot: "DUSTBIN" for slot in range(NEWBORN_QUERY_COUNT)
    }
    if not available_instances and any(newborn_costs[slot] for slot in newborn_costs):
        raise PrefixRouteB2Error(
            "newborn costs must be empty when the risk set is fully assigned"
        )
    if available_instances:
        target_count = len(available_instances)
        matrix = []
        for slot in range(NEWBORN_QUERY_COUNT):
            row = newborn_costs[slot]
            if not isinstance(row, dict) or set(row) != set(available_instances):
                raise PrefixRouteB2Error(
                    "newborn cost columns differ from unassigned risk set"
                )
            matrix_row = []
            for column, instance_id in enumerate(available_instances):
                del column
                matrix_row.append(
                    _quantized_cost_units(
                        row[instance_id],
                        "newborn assignment cost",
                    )
                )
            for dustbin in range(NEWBORN_QUERY_COUNT):
                del dustbin
                matrix_row.append(int(DUSTBIN_COST * _COST_SCALE))
            matrix.append(matrix_row)
        assignment = _lexicographic_exact_assignment(matrix)
        for slot, column in enumerate(assignment):
            if column < target_count:
                newborn_assignment[slot] = available_instances[column]
    return {
        "schema_version": B2_CONTRACT_SCHEMA,
        "propagated_assignment": locked,
        "newborn_assignment": newborn_assignment,
        "one_to_one": True,
        "unmatched_target": "DUSTBIN",
        "dustbin_cost": DUSTBIN_COST,
        "cost_quantization": "round_half_even_1e-6",
        "tie_break": "exact_primary_integer_then_rowwise_lexicographic_column",
    }


def _validate_previous_tracks(previous_tracks):
    if not isinstance(previous_tracks, list):
        raise PrefixRouteB2Error("previous tracks must be an array")
    required = {
        "track_id",
        "state_ref",
        "started_at_bin",
        "last_decision_bin",
    }
    rows = []
    for row in previous_tracks:
        if not isinstance(row, dict) or set(row) != required:
            raise PrefixRouteB2Error("previous track fields differ")
        rows.append(
            {
                "track_id": _identifier(row["track_id"], "track_id"),
                "state_ref": _identifier(row["state_ref"], "state_ref"),
                "started_at_bin": _nonnegative_int(
                    row["started_at_bin"],
                    "started_at_bin",
                ),
                "last_decision_bin": _nonnegative_int(
                    row["last_decision_bin"],
                    "last_decision_bin",
                ),
            }
        )
    if len(rows) > TRACK_CAPACITY:
        raise PrefixRouteB2Error("previous track capacity is exceeded")
    if len({row["track_id"] for row in rows}) != len(rows):
        raise PrefixRouteB2Error("previous track IDs are duplicated")
    return sorted(rows, key=lambda row: row["track_id"])


def _validate_decoder_rows(decoder_rows, previous_tracks):
    if not isinstance(decoder_rows, list):
        raise PrefixRouteB2Error("decoder rows must be an array")
    required = {
        "pool",
        "slot",
        "track_id",
        "active_score",
        "completion_score",
        "class_score",
        "start",
        "end",
        "label",
        "state_ref",
    }
    rows = []
    for row in decoder_rows:
        if not isinstance(row, dict) or set(row) != required:
            raise PrefixRouteB2Error("decoder row fields differ")
        pool = row["pool"]
        if pool not in {"propagated", "newborn"}:
            raise PrefixRouteB2Error("decoder query pool differs")
        rows.append(
            {
                **row,
                "slot": _nonnegative_int(row["slot"], "decoder slot"),
                "active_score": _unit(row["active_score"], "active_score"),
                "completion_score": _unit(
                    row["completion_score"],
                    "completion_score",
                ),
                "class_score": _unit(row["class_score"], "class_score"),
                "start": _finite(row["start"], "predicted start"),
                "end": _finite(row["end"], "predicted end"),
                "label": _identifier(row["label"], "predicted label"),
                "state_ref": _identifier(row["state_ref"], "state_ref"),
            }
        )
    propagated = [row for row in rows if row["pool"] == "propagated"]
    newborn = [row for row in rows if row["pool"] == "newborn"]
    if len(propagated) != len(previous_tracks):
        raise PrefixRouteB2Error("propagated decoder row count differs")
    if len(newborn) != NEWBORN_QUERY_COUNT:
        raise PrefixRouteB2Error("all 64 newborn queries must run every decision")
    if sorted(row["slot"] for row in propagated) != list(
        range(len(previous_tracks))
    ):
        raise PrefixRouteB2Error("propagated slots differ")
    if sorted(row["slot"] for row in newborn) != list(
        range(NEWBORN_QUERY_COUNT)
    ):
        raise PrefixRouteB2Error("newborn slots differ")
    expected_tracks = [row["track_id"] for row in previous_tracks]
    observed_tracks = [
        row["track_id"] for row in sorted(propagated, key=lambda row: row["slot"])
    ]
    if observed_tracks != expected_tracks:
        raise PrefixRouteB2Error("propagated query identity differs")
    if any(row["track_id"] is not None for row in newborn):
        raise PrefixRouteB2Error("newborn queries may not assert prior track identity")
    return propagated, newborn


def temporal_motr_transition(
    *,
    stream_key,
    decision_bin,
    decision_observation_count,
    calibrated_threshold,
    previous_tracks,
    decoder_rows,
    next_track_serial,
    next_emission_sequence,
):
    """Execute the exact B2 lifecycle, including its no-same-bin-reuse limit."""

    stream_key = _identifier(stream_key, "stream_key")
    decision_bin = _nonnegative_int(decision_bin, "decision_bin")
    observation_count = _nonnegative_int(
        decision_observation_count,
        "decision_observation_count",
    )
    if observation_count <= 0:
        raise PrefixRouteB2Error(
            "decision_observation_count must be a positive integer"
        )
    if decision_bin != (observation_count - 1) // 8:
        raise PrefixRouteB2Error(
            "decision bin differs from the frozen observation coordinate"
        )
    threshold = validate_calibrated_threshold(calibrated_threshold)
    serial = _nonnegative_int(next_track_serial, "next_track_serial")
    sequence = _nonnegative_int(
        next_emission_sequence,
        "next_emission_sequence",
    )
    previous = _validate_previous_tracks(previous_tracks)
    if any(row["last_decision_bin"] >= decision_bin for row in previous):
        raise PrefixRouteB2Error("previous tracks do not precede this decision")
    propagated, newborn = _validate_decoder_rows(decoder_rows, previous)
    rows_by_track = {row["track_id"]: row for row in propagated}
    emissions = []
    survivors = []

    def emit(row, identity):
        nonlocal sequence
        if (
            row["start"] < 0
            or not row["start"] < row["end"] <= observation_count
        ):
            raise PrefixRouteB2Error("emitted interval escapes observed prefix")
        emissions.append(
            {
                "emission_id": f"{stream_key}:b2:{identity}:{sequence}",
                "stream_key": stream_key,
                "sequence_id": sequence,
                "start": row["start"],
                "end": row["end"],
                "class": row["label"],
                "score": row["class_score"],
                "source_frame": max(0, observation_count - 1),
                "emit_frame": observation_count,
            }
        )
        sequence += 1

    for track in previous:
        row = rows_by_track[track["track_id"]]
        completed = (
            row["completion_score"] >= threshold
            and row["class_score"] >= threshold
        )
        if completed:
            emit(row, track["track_id"])
        elif row["active_score"] >= threshold:
            survivors.append(
                {
                    "track_id": track["track_id"],
                    "state_ref": row["state_ref"],
                    "started_at_bin": track["started_at_bin"],
                    "last_decision_bin": decision_bin,
                }
            )

    direct_newborn = [
        row for row in newborn
        if row["completion_score"] >= threshold
        and row["class_score"] >= threshold
    ]
    for row in sorted(direct_newborn, key=lambda item: item["slot"]):
        emit(row, f"direct-{row['slot']}")

    # B2 may allocate only capacity free before this decision. Slots released by
    # completion/drop become usable at the next decision, which isolates D1.
    predecision_free = TRACK_CAPACITY - len(previous)
    birth_candidates = [
        row for row in newborn
        if not (
            row["completion_score"] >= threshold
            and row["class_score"] >= threshold
        )
        and row["active_score"] >= threshold
    ]
    birth_candidates.sort(
        key=lambda row: (
            -row["active_score"],
            -row["class_score"],
            row["slot"],
        )
    )
    births = []
    for row in birth_candidates[:predecision_free]:
        track_id = f"{stream_key}:b2-track:{serial}"
        serial += 1
        births.append(
            {
                "track_id": track_id,
                "state_ref": row["state_ref"],
                "started_at_bin": decision_bin,
                "last_decision_bin": decision_bin,
            }
        )
    next_tracks = sorted(survivors + births, key=lambda row: row["track_id"])
    if len(next_tracks) > TRACK_CAPACITY:
        raise PrefixRouteB2Error("post-transition track capacity is exceeded")
    return {
        "schema_version": B2_CONTRACT_SCHEMA,
        "evidence_role": "DIAGNOSTIC_ONLY_NOT_A_FORMAL_STREAM_CERTIFICATE",
        "stream_key": stream_key,
        "decision_bin": decision_bin,
        "decision_observation_count": observation_count,
        "query_counts": {
            "propagated": len(previous),
            "newborn": NEWBORN_QUERY_COUNT,
            "decoder_total": len(previous) + NEWBORN_QUERY_COUNT,
            "maximum_decoder_total": MAX_DECODER_QUERY_COUNT,
        },
        "predecision_free_track_slots": predecision_free,
        "released_slots_reusable_same_decision": False,
        "next_tracks": next_tracks,
        "emissions": emissions,
        "next_track_serial": serial,
        "next_emission_sequence": sequence,
    }


class B2StreamMachine:
    """Own the indivisible B2 clock, lifecycle state, and transcript."""

    def __init__(
        self,
        *,
        stream_key,
        frame_count,
        calibrated_threshold,
        feature_stride_frames=FEATURE_STRIDE_FRAMES,
        run_binding_sha256,
    ):
        self._stream_key = _identifier(stream_key, "stream_key")
        self._frame_count = _nonnegative_int(frame_count, "frame_count")
        if self._frame_count <= 0:
            raise PrefixRouteB2Error("frame_count must be positive")
        self._stride = _nonnegative_int(
            feature_stride_frames,
            "feature_stride_frames",
        )
        if self._stride != FEATURE_STRIDE_FRAMES:
            raise PrefixRouteB2Error(
                "feature_stride_frames differs from the frozen B2 contract"
            )
        self._threshold = validate_calibrated_threshold(calibrated_threshold)
        if (
            not isinstance(run_binding_sha256, str)
            or len(run_binding_sha256) != 64
            or any(character not in "0123456789abcdef" for character in run_binding_sha256)
        ):
            raise PrefixRouteB2Error("run_binding_sha256 must be a lowercase SHA-256")
        self._run_binding_sha256 = run_binding_sha256
        self._decision_index = 0
        self._observation_count = 0
        self._tracks = []
        self._next_track_serial = 0
        self._next_emission_sequence = 0
        self._previous_state_sha256 = _GENESIS_STATE_SHA256
        self._terminal = False
        self._transcript = []

    def _next_observation_count(self):
        if self._terminal:
            return None
        return min(
            (self._decision_index + 1) * self._stride,
            self._frame_count,
        )

    def snapshot(self):
        state = {
            "schema_version": B2_STREAM_STATE_SCHEMA,
            "stream_key": self._stream_key,
            "run_binding_sha256": self._run_binding_sha256,
            "frame_count": self._frame_count,
            "feature_stride_frames": self._stride,
            "calibrated_threshold": self._threshold,
            "decision_index": self._decision_index,
            "observation_count": self._observation_count,
            "next_decision_observation_count": self._next_observation_count(),
            "next_track_serial": self._next_track_serial,
            "next_emission_sequence": self._next_emission_sequence,
            "tracks": [dict(row) for row in self._tracks],
            "terminal": self._terminal,
            "previous_state_sha256": self._previous_state_sha256,
        }
        state["state_sha256"] = _state_sha256(state)
        return state

    def advance(self, *, decision_observation_count, decoder_rows):
        expected = self._next_observation_count()
        if expected is None:
            raise PrefixRouteB2Error("B2 stream is already terminal")
        if decision_observation_count != expected:
            raise PrefixRouteB2Error(
                "B2 stream clock rejects repeated, skipped, reordered, or reset input"
            )
        before = self.snapshot()
        decision_bin = (decision_observation_count - 1) // self._stride
        result = temporal_motr_transition(
            stream_key=self._stream_key,
            decision_bin=decision_bin,
            decision_observation_count=decision_observation_count,
            calibrated_threshold=self._threshold,
            previous_tracks=self._tracks,
            decoder_rows=decoder_rows,
            next_track_serial=self._next_track_serial,
            next_emission_sequence=self._next_emission_sequence,
        )
        self._tracks = [dict(row) for row in result["next_tracks"]]
        self._next_track_serial = result["next_track_serial"]
        self._next_emission_sequence = result["next_emission_sequence"]
        self._observation_count = decision_observation_count
        self._decision_index += 1
        self._previous_state_sha256 = before["state_sha256"]
        self._terminal = decision_observation_count == self._frame_count
        after = self.snapshot()
        transcript_row = {
            "schema_version": B2_STREAM_TRANSCRIPT_SCHEMA,
            "stream_key": self._stream_key,
            "run_binding_sha256": self._run_binding_sha256,
            "decision_index": self._decision_index - 1,
            "decision_observation_count": decision_observation_count,
            "state_before_sha256": before["state_sha256"],
            "decoder_rows_sha256": hashlib.sha256(
                _canonical_json_bytes(decoder_rows)
            ).hexdigest(),
            "emissions_sha256": hashlib.sha256(
                _canonical_json_bytes(result["emissions"])
            ).hexdigest(),
            "state_after_sha256": after["state_sha256"],
        }
        self._transcript.append(transcript_row)
        return {
            "state_before": before,
            "transition": result,
            "state_after": after,
            "transcript_row": dict(transcript_row),
        }

    def transcript(self):
        return tuple(dict(row) for row in self._transcript)


__all__ = [
    "B2_CONTRACT_SCHEMA",
    "B2_STREAM_STATE_SCHEMA",
    "B2_STREAM_TRANSCRIPT_SCHEMA",
    "B2StreamMachine",
    "DUSTBIN_COST",
    "FEATURE_STRIDE_FRAMES",
    "MAX_DECODER_QUERY_COUNT",
    "NEWBORN_QUERY_COUNT",
    "THRESHOLD_GRID",
    "TRACK_CAPACITY",
    "PrefixRouteB2Error",
    "build_temporal_tracklet_assignment",
    "instance_aware_risk_set",
    "temporal_motr_transition",
    "validate_calibrated_threshold",
]
