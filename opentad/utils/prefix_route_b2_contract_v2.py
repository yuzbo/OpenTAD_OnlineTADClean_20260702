"""Executable temporal-MOTR attack contract for Prefix-Route Protocol V2."""

from __future__ import annotations

import math


B2_CONTRACT_SCHEMA = "prefix-route-temporal-motr-b2-v2"
TRACK_CAPACITY = 64
NEWBORN_QUERY_COUNT = 64
MAX_DECODER_QUERY_COUNT = TRACK_CAPACITY + NEWBORN_QUERY_COUNT
THRESHOLD_GRID = tuple(index / 20.0 for index in range(1, 20))
DUSTBIN_COST = 10.0


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
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment
        except Exception as exc:
            raise PrefixRouteB2Error(
                "temporal tracklet assignment requires NumPy and SciPy"
            ) from exc
        matrix = np.empty(
            (NEWBORN_QUERY_COUNT, len(available_instances)),
            dtype=np.float64,
        )
        for slot in range(NEWBORN_QUERY_COUNT):
            row = newborn_costs[slot]
            if not isinstance(row, dict) or set(row) != set(available_instances):
                raise PrefixRouteB2Error(
                    "newborn cost columns differ from unassigned risk set"
                )
            for column, instance_id in enumerate(available_instances):
                cost = _finite(row[instance_id], "newborn assignment cost")
                if cost < 0:
                    raise PrefixRouteB2Error(
                        "newborn assignment cost must be non-negative"
                    )
                quantized = round(cost, 6)
                if not math.isclose(cost, quantized, rel_tol=0.0, abs_tol=1e-12):
                    raise PrefixRouteB2Error(
                        "newborn assignment costs must be quantized to 1e-6"
                    )
                matrix[slot, column] = (
                    quantized + slot * 1e-9 + column * 1e-12
                )
        row_indexes, column_indexes = linear_sum_assignment(matrix)
        for slot, column in zip(row_indexes.tolist(), column_indexes.tolist()):
            raw_cost = matrix[slot, column] - slot * 1e-9 - column * 1e-12
            if raw_cost < DUSTBIN_COST:
                newborn_assignment[slot] = available_instances[column]
    return {
        "schema_version": B2_CONTRACT_SCHEMA,
        "propagated_assignment": locked,
        "newborn_assignment": newborn_assignment,
        "one_to_one": True,
        "unmatched_target": "DUSTBIN",
        "dustbin_cost": DUSTBIN_COST,
        "cost_quantization": "round_half_even_1e-6",
        "tie_break": "newborn_slot_then_instance_id",
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


__all__ = [
    "B2_CONTRACT_SCHEMA",
    "DUSTBIN_COST",
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
