import math
import re


REQUIRED_SEEDS = (705, 706, 707)
REQUIRED_TIOU_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
METRIC_SCHEMA = "standard_ontad_map.v1"
INSTANCE_METRIC_SCHEMA = "online_instance_metrics.v2"
MAP_UNIT = "percentage_points"

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

PROVENANCE_SHA256_FIELDS = (
    "annotation_sha256",
    "calibration_protocol_sha256",
    "checkpoint_sha256",
    "census_sha256",
    "config_sha256",
    "emission_ledger_sha256",
    "reporting_manifest_sha256",
)
COMMON_PROVENANCE_FIELDS = (
    "annotation_sha256",
    "calibration_protocol_sha256",
    "census_sha256",
    "reporting_manifest_sha256",
)


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


def _exact(row, field, expected):
    if row.get(field) != expected:
        raise PersistentBindingGateInputError(
            f"{field} must equal {expected!r}"
        )
    return expected


def _sha256(value, field):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise PersistentBindingGateInputError(
            f"provenance.{field} must be a 64-character SHA-256"
        )
    return value.lower()


def _provenance(row):
    value = row.get("provenance")
    if not isinstance(value, dict):
        raise PersistentBindingGateInputError("provenance must be a dictionary")
    code_commit = value.get("code_commit")
    if not isinstance(code_commit, str) or re.fullmatch(
        r"[0-9a-fA-F]{40}",
        code_commit,
    ) is None:
        raise PersistentBindingGateInputError(
            "provenance.code_commit must be a full 40-character Git SHA"
        )
    normalized = {"code_commit": code_commit.lower()}
    for field in PROVENANCE_SHA256_FIELDS:
        normalized[field] = _sha256(value.get(field), field)
    return normalized


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
        _exact(row, "metric_schema", METRIC_SCHEMA)
        _exact(row, "instance_metric_schema", INSTANCE_METRIC_SCHEMA)
        _exact(row, "map_unit", MAP_UNIT)
        try:
            thresholds = tuple(float(value) for value in row["tiou_thresholds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PersistentBindingGateInputError(
                "tiou_thresholds must be a numeric sequence"
            ) from exc
        if thresholds != REQUIRED_TIOU_THRESHOLDS:
            raise PersistentBindingGateInputError(
                f"tiou_thresholds must equal {REQUIRED_TIOU_THRESHOLDS}"
            )
        row["tiou_thresholds"] = thresholds
        row["average_map_pct"] = _finite(row, "average_map_pct")
        if not 0.0 <= row["average_map_pct"] <= 100.0:
            raise PersistentBindingGateInputError(
                "average_map_pct must be in [0, 100]"
            )
        row["duplicate_rate"] = _finite(row, "duplicate_rate")
        row["fragmentation_rate"] = _finite(row, "fragmentation_rate")
        row["prediction_gt_ratio"] = _finite(row, "prediction_gt_ratio")
        row["recall_tiou_0p3"] = _finite(row, "recall_tiou_0p3")
        for field in (
            "protocol_violations",
            "gt_supervision_exhaustions",
            "gt_birth_runtime_entry_free_collisions",
            "candidate_arbitration_suppressions",
            "candidate_cancellations",
            "active_abandonments",
            "deferred_birth_due_to_release",
            "committed_predictions",
            "expected_updates",
            "successful_updates",
            "scheduler_steps",
            "skipped_updates",
        ):
            row[field] = _integer(row, field)
            if row[field] < 0:
                raise PersistentBindingGateInputError(
                    f"{arm} seed {seed} contains negative {field}"
                )
        if min(
            row["duplicate_rate"],
            row["fragmentation_rate"],
            row["prediction_gt_ratio"],
            row["recall_tiou_0p3"],
        ) < 0:
            raise PersistentBindingGateInputError(
                f"{arm} seed {seed} contains a negative rate"
            )
        if row["recall_tiou_0p3"] > 1:
            raise PersistentBindingGateInputError(
                f"{arm} seed {seed} recall_tiou_0p3 exceeds one"
            )
        row["provenance"] = _provenance(row)
        row["identity_error"] = 0.5 * (
            row["duplicate_rate"] + row["fragmentation_rate"]
        )
        normalized[seed] = row

    if tuple(sorted(normalized)) != REQUIRED_SEEDS:
        raise PersistentBindingGateInputError(
            f"{arm} must contain exactly seeds {REQUIRED_SEEDS}"
        )
    return normalized


def _validate_common_provenance(fixed, rematch):
    rows = [
        (arm, seed, arm_rows[seed])
        for arm, arm_rows in (("fixed", fixed), ("rematch", rematch))
        for seed in REQUIRED_SEEDS
    ]
    first_arm, first_seed, first = rows[0]
    for arm, seed, row in rows[1:]:
        if row["provenance"]["code_commit"] != first["provenance"]["code_commit"]:
            raise PersistentBindingGateInputError(
                "all rows must use one identical code commit; "
                f"{first_arm}/{first_seed} differs from {arm}/{seed}"
            )
        for field in COMMON_PROVENANCE_FIELDS:
            if row["provenance"][field] != first["provenance"][field]:
                raise PersistentBindingGateInputError(
                    f"all rows must share provenance.{field}; "
                    f"{first_arm}/{first_seed} differs from {arm}/{seed}"
                )


def _technical_violations(arm, row):
    failures = []
    seed = row["seed"]
    for field in (
        "protocol_violations",
        "gt_supervision_exhaustions",
        "gt_birth_runtime_entry_free_collisions",
        "skipped_updates",
    ):
        if row[field] != 0:
            failures.append(f"{arm}/{seed}: {field}={row[field]}")
    if row["committed_predictions"] <= 0:
        failures.append(f"{arm}/{seed}: no committed predictions")
    if row["expected_updates"] <= 0:
        failures.append(f"{arm}/{seed}: expected_updates must be positive")
    if not (
        row["successful_updates"]
        == row["expected_updates"]
        == row["scheduler_steps"]
    ):
        failures.append(
            f"{arm}/{seed}: updates expected={row['expected_updates']} "
            f"successful={row['successful_updates']} "
            f"scheduler={row['scheduler_steps']}"
        )
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
    _validate_common_provenance(fixed, rematch)

    technical_failures = []
    for arm, rows in (("fixed", fixed), ("rematch", rematch)):
        for seed in REQUIRED_SEEDS:
            technical_failures.extend(_technical_violations(arm, rows[seed]))
    technical_pass = not technical_failures

    fixed_identity = sum(
        fixed[seed]["identity_error"] for seed in REQUIRED_SEEDS
    ) / 3
    rematch_identity = sum(
        rematch[seed]["identity_error"] for seed in REQUIRED_SEEDS
    ) / 3
    relative_reduction = (
        None
        if rematch_identity == 0
        else (rematch_identity - fixed_identity) / rematch_identity
    )
    improved_seeds = sum(
        fixed[seed]["identity_error"] < rematch[seed]["identity_error"]
        for seed in REQUIRED_SEEDS
    )
    fixed_map = sum(
        fixed[seed]["average_map_pct"] for seed in REQUIRED_SEEDS
    ) / 3
    rematch_map = sum(
        rematch[seed]["average_map_pct"] for seed in REQUIRED_SEEDS
    ) / 3
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
        "schema_version": "persistent_binding_gate.v2",
        "metric_schema": METRIC_SCHEMA,
        "instance_metric_schema": INSTANCE_METRIC_SCHEMA,
        "map_unit": MAP_UNIT,
        "tiou_thresholds": REQUIRED_TIOU_THRESHOLDS,
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
        "fixed_average_map_pct": fixed_map,
        "rematch_average_map_pct": rematch_map,
        "map_delta_points": map_delta,
    }


__all__ = [
    "INSTANCE_METRIC_SCHEMA",
    "MAP_UNIT",
    "METRIC_SCHEMA",
    "PersistentBindingGateInputError",
    "REQUIRED_SEEDS",
    "REQUIRED_TIOU_THRESHOLDS",
    "SCIENTIFIC_THRESHOLDS",
    "TECHNICAL_THRESHOLDS",
    "evaluate_persistent_binding_gate",
]
