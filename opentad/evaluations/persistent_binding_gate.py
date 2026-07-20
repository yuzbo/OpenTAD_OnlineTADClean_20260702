import math


REQUIRED_SEEDS = (705, 706, 707)

TECHNICAL_THRESHOLDS = {
    "prediction_gt_ratio_min": 0.25,
    "prediction_gt_ratio_max": 4.0,
    "recall_tiou_0p3_min": 0.25,
}

SCIENTIFIC_THRESHOLDS = {
    "identity_error_relative_reduction_min": 0.20,
    "improved_seed_count_min": 2,
    "map_delta_points_min": -0.5,
}


class PersistentBindingGateInputError(ValueError):
    pass


def _finite(row, field):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise PersistentBindingGateInputError(
            f"{field} must be a finite number"
        ) from exc
    if not math.isfinite(value):
        raise PersistentBindingGateInputError(f"{field} must be a finite number")
    return value


def _integer(row, field):
    value = _finite(row, field)
    if not value.is_integer():
        raise PersistentBindingGateInputError(f"{field} must be an integer")
    return int(value)


def _normalize(rows, arm):
    if not isinstance(rows, (list, tuple)):
        raise PersistentBindingGateInputError(f"{arm} rows must be a sequence")
    normalized = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise PersistentBindingGateInputError(f"{arm} rows must be dictionaries")
        row = dict(raw)
        seed = _integer(row, "seed")
        if seed in normalized:
            raise PersistentBindingGateInputError(f"{arm} repeats seed {seed}")
        row["seed"] = seed
        row["average_map"] = _finite(row, "average_map")
        row["duplicate_rate"] = _finite(row, "duplicate_rate")
        row["fragmentation_rate"] = _finite(row, "fragmentation_rate")
        row["prediction_gt_ratio"] = _finite(row, "prediction_gt_ratio")
        row["recall_tiou_0p3"] = _finite(row, "recall_tiou_0p3")
        for field in (
            "protocol_violations",
            "dropped_gt_birth_targets",
            "runtime_capacity_exhaustions",
            "committed_predictions",
        ):
            row[field] = _integer(row, field)
        if min(
            row["duplicate_rate"],
            row["fragmentation_rate"],
            row["prediction_gt_ratio"],
            row["recall_tiou_0p3"],
        ) < 0:
            raise PersistentBindingGateInputError(
                f"{arm} seed {seed} contains a negative rate"
            )
        row["identity_error"] = 0.5 * (
            row["duplicate_rate"] + row["fragmentation_rate"]
        )
        normalized[seed] = row

    if tuple(sorted(normalized)) != REQUIRED_SEEDS:
        raise PersistentBindingGateInputError(
            f"{arm} must contain exactly seeds {REQUIRED_SEEDS}"
        )
    return normalized


def _technical_violations(arm, row):
    failures = []
    seed = row["seed"]
    for field in (
        "protocol_violations",
        "dropped_gt_birth_targets",
        "runtime_capacity_exhaustions",
    ):
        if row[field] != 0:
            failures.append(f"{arm}/{seed}: {field}={row[field]}")
    if row["committed_predictions"] <= 0:
        failures.append(f"{arm}/{seed}: no committed predictions")
    if not (
        TECHNICAL_THRESHOLDS["prediction_gt_ratio_min"]
        <= row["prediction_gt_ratio"]
        <= TECHNICAL_THRESHOLDS["prediction_gt_ratio_max"]
    ):
        failures.append(
            f"{arm}/{seed}: prediction_gt_ratio={row['prediction_gt_ratio']}"
        )
    if row["recall_tiou_0p3"] < TECHNICAL_THRESHOLDS["recall_tiou_0p3_min"]:
        failures.append(
            f"{arm}/{seed}: recall_tiou_0p3={row['recall_tiou_0p3']}"
        )
    return failures


def evaluate_persistent_binding_gate(fixed_rows, rematch_rows):
    fixed = _normalize(fixed_rows, "fixed")
    rematch = _normalize(rematch_rows, "rematch")

    technical_failures = []
    for arm, rows in (("fixed", fixed), ("rematch", rematch)):
        for seed in REQUIRED_SEEDS:
            technical_failures.extend(_technical_violations(arm, rows[seed]))
    technical_pass = not technical_failures

    fixed_identity = sum(fixed[seed]["identity_error"] for seed in REQUIRED_SEEDS) / 3
    rematch_identity = (
        sum(rematch[seed]["identity_error"] for seed in REQUIRED_SEEDS) / 3
    )
    relative_reduction = (
        None
        if rematch_identity == 0
        else (rematch_identity - fixed_identity) / rematch_identity
    )
    improved_seeds = sum(
        fixed[seed]["identity_error"] < rematch[seed]["identity_error"]
        for seed in REQUIRED_SEEDS
    )
    fixed_map = sum(fixed[seed]["average_map"] for seed in REQUIRED_SEEDS) / 3
    rematch_map = sum(rematch[seed]["average_map"] for seed in REQUIRED_SEEDS) / 3
    map_delta = fixed_map - rematch_map

    scientific_failures = []
    if relative_reduction is None:
        scientific_failures.append("rematch identity error is zero")
    elif (
        relative_reduction
        < SCIENTIFIC_THRESHOLDS["identity_error_relative_reduction_min"]
    ):
        scientific_failures.append(
            f"identity error relative reduction={relative_reduction:.6f}"
        )
    if improved_seeds < SCIENTIFIC_THRESHOLDS["improved_seed_count_min"]:
        scientific_failures.append(f"improved seed count={improved_seeds}")
    if map_delta < SCIENTIFIC_THRESHOLDS["map_delta_points_min"]:
        scientific_failures.append(f"mAP delta={map_delta:.6f} points")
    scientific_pass = not scientific_failures

    return {
        "schema_version": "persistent_binding_gate.v1",
        "required_seeds": REQUIRED_SEEDS,
        "technical_thresholds": dict(TECHNICAL_THRESHOLDS),
        "scientific_thresholds": dict(SCIENTIFIC_THRESHOLDS),
        "technical_pass": technical_pass,
        "technical_failures": tuple(technical_failures),
        "scientific_pass": scientific_pass,
        "scientific_failures": tuple(scientific_failures),
        "overall_pass": technical_pass and scientific_pass,
        "fixed_identity_error": fixed_identity,
        "rematch_identity_error": rematch_identity,
        "identity_error_relative_reduction": relative_reduction,
        "improved_seed_count": improved_seeds,
        "fixed_average_map": fixed_map,
        "rematch_average_map": rematch_map,
        "map_delta_points": map_delta,
    }


__all__ = [
    "PersistentBindingGateInputError",
    "REQUIRED_SEEDS",
    "SCIENTIFIC_THRESHOLDS",
    "TECHNICAL_THRESHOLDS",
    "evaluate_persistent_binding_gate",
]
