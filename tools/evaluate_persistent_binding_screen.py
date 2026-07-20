"""Apply the frozen technical gate to the paired seed-705 screen."""

import argparse
import json
import math
from pathlib import Path
import re
import sys


RESULT_SCHEMA = "persistent_binding_screen_result.v1"
RESOURCE_SCHEMA = "persistent_binding_resource.v1"
EXPECTED_SEED = 705
EXPECTED_TIOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
TECHNICAL_THRESHOLDS = {
    "prediction_gt_ratio_min": 0.25,
    "prediction_gt_ratio_max": 4.0,
    "recall_tiou_0p3_min": 0.25,
}
COMMON_PROVENANCE = (
    "code_commit",
    "annotation_sha256",
    "calibration_manifest_sha256",
    "fit_manifest_sha256",
    "reporting_manifest_sha256",
    "census_sha256",
    "profile_gate_sha256",
    "smoke_gate_sha256",
    "screening_contract_sha256",
)


def _load(path):
    with Path(path).open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _integer(row, field):
    value = _finite(row, field)
    if not value.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _validate_sha(value, field, length=64):
    if not isinstance(value, str) or re.fullmatch(
        rf"[0-9a-f]{{{length}}}",
        value,
    ) is None:
        raise ValueError(f"{field} is not a lowercase hash")


def _normalize(payload, arm):
    if payload.get("schema_version") != RESULT_SCHEMA:
        raise ValueError(f"{arm} has an unexpected screen-result schema")
    if payload.get("arm") != arm:
        raise ValueError(f"{arm} result has the wrong arm")
    if int(payload.get("seed", -1)) != EXPECTED_SEED:
        raise ValueError(f"{arm} screen seed must be {EXPECTED_SEED}")
    if payload.get("reporting_accessed") is not False:
        raise ValueError(f"{arm} screen accessed reporting data")
    if payload.get("effectiveness_claim_authorized") is not False:
        raise ValueError(f"{arm} screen improperly authorizes effectiveness")
    if payload.get("raw_rgb_authorized") is not False:
        raise ValueError(f"{arm} screen improperly authorizes raw RGB")
    expected_binding = {
        "fixed": "fixed_birth_slot",
        "rematch": "prefix_rematch_active_pool",
    }[arm]
    if payload.get("binding_mode") != expected_binding:
        raise ValueError(f"{arm} screen has the wrong binding mode")
    row = dict(payload["gate_row"])
    if row.get("arm") != arm or row.get("binding_mode") != expected_binding:
        raise ValueError(f"{arm} gate row has the wrong arm or binding mode")
    for field in (
        "reporting_accessed",
        "effectiveness_claim_authorized",
        "raw_rgb_authorized",
    ):
        if row.get(field) is not False:
            raise ValueError(f"{arm} gate row changed {field}")
    if tuple(float(value) for value in row["tiou_thresholds"]) != EXPECTED_TIOUS:
        raise ValueError(f"{arm} screen changed tIoU thresholds")
    if row.get("metric_schema") != "standard_ontad_map.v1":
        raise ValueError(f"{arm} screen changed the standard-mAP schema")
    if row.get("instance_metric_schema") != "online_instance_metrics.v2":
        raise ValueError(f"{arm} screen changed the instance metric schema")
    if row.get("map_unit") != "percentage_points":
        raise ValueError(f"{arm} screen mAP must use percentage points")
    if int(row.get("screen_epochs", -1)) != 1:
        raise ValueError(f"{arm} screen must contain one epoch")
    for field in (
        "average_map_pct",
        "duplicate_rate",
        "fragmentation_rate",
        "prediction_gt_ratio",
        "recall_tiou_0p3",
        "allocated_gpu_hours",
        "projected_pair_gpu_hours",
    ):
        row[field] = _finite(row, field)
    for field in (
        "protocol_violations",
        "gt_supervision_exhaustions",
        "gt_birth_runtime_entry_free_collisions",
        "committed_predictions",
        "expected_updates",
        "successful_updates",
        "scheduler_steps",
        "skipped_updates",
    ):
        row[field] = _integer(row, field)
        if row[field] < 0:
            raise ValueError(f"{arm} contains negative {field}")
    provenance = dict(row["provenance"])
    for field in COMMON_PROVENANCE:
        _validate_sha(
            provenance.get(field),
            field,
            length=40 if field == "code_commit" else 64,
        )
    for field, value in provenance.items():
        if field not in COMMON_PROVENANCE:
            _validate_sha(value, field)
    row["provenance"] = provenance
    row["identity_error"] = 0.5 * (
        row["duplicate_rate"] + row["fragmentation_rate"]
    )
    return row


def _technical_failures(arm, row):
    failures = []
    for field in (
        "protocol_violations",
        "gt_supervision_exhaustions",
        "gt_birth_runtime_entry_free_collisions",
        "skipped_updates",
    ):
        if row[field] != 0:
            failures.append(f"{arm}: {field}={row[field]}")
    if row["committed_predictions"] <= 0:
        failures.append(f"{arm}: no committed predictions")
    if row["expected_updates"] <= 0:
        failures.append(f"{arm}: expected_updates must be positive")
    if not (
        row["successful_updates"]
        == row["expected_updates"]
        == row["scheduler_steps"]
    ):
        failures.append(
            f"{arm}: updates expected={row['expected_updates']} "
            f"successful={row['successful_updates']} "
            f"scheduler={row['scheduler_steps']}"
        )
    if not (
        TECHNICAL_THRESHOLDS["prediction_gt_ratio_min"]
        <= row["prediction_gt_ratio"]
        <= TECHNICAL_THRESHOLDS["prediction_gt_ratio_max"]
    ):
        failures.append(
            f"{arm}: prediction_gt_ratio={row['prediction_gt_ratio']}"
        )
    if (
        row["recall_tiou_0p3"]
        < TECHNICAL_THRESHOLDS["recall_tiou_0p3_min"]
    ):
        failures.append(
            f"{arm}: recall_tiou_0p3={row['recall_tiou_0p3']}"
        )
    return failures


def evaluate_screen(fixed_payload, rematch_payload, resource_payload):
    fixed = _normalize(fixed_payload, "fixed")
    rematch = _normalize(rematch_payload, "rematch")
    for field in COMMON_PROVENANCE:
        if fixed["provenance"].get(field) != rematch["provenance"].get(field):
            raise ValueError(f"paired screen differs on provenance.{field}")
    if resource_payload.get("schema_version") != RESOURCE_SCHEMA:
        raise ValueError("paired screen has an unexpected resource schema")
    if resource_payload.get("scope") != "pair":
        raise ValueError("paired resource scope must be pair")
    if (
        resource_payload.get("code_commit")
        != fixed["provenance"]["code_commit"]
    ):
        raise ValueError("paired resource report code commit mismatch")
    actual_pair_gpu_hours = _finite(
        resource_payload,
        "allocated_gpu_hours",
    )
    if actual_pair_gpu_hours <= 0:
        raise ValueError("paired allocated GPU hours must be positive")
    paired_cap = 2.0
    budget_pass = actual_pair_gpu_hours <= paired_cap

    failures = []
    failures.extend(_technical_failures("fixed", fixed))
    failures.extend(_technical_failures("rematch", rematch))
    if (
        fixed["projected_pair_gpu_hours"]
        != rematch["projected_pair_gpu_hours"]
    ):
        failures.append("pair: projected GPU hours differ between arms")
    if fixed["projected_pair_gpu_hours"] > paired_cap:
        failures.append(
            "pair: projected GPU hours exceed the frozen cap"
        )
    if not budget_pass:
        failures.append(
            f"pair: allocated_gpu_hours={actual_pair_gpu_hours}"
        )
    technical_pass = not failures
    rematch_identity = rematch["identity_error"]
    relative_reduction = (
        None
        if rematch_identity == 0
        else (
            rematch_identity - fixed["identity_error"]
        )
        / rematch_identity
    )
    return {
        "schema_version": "persistent_binding_seed705_screen_gate.v1",
        "seed": EXPECTED_SEED,
        "purpose": "convergence_and_non_degeneracy_only",
        "screen_pass": technical_pass,
        "technical_pass": technical_pass,
        "technical_failures": failures,
        "technical_thresholds": dict(TECHNICAL_THRESHOLDS),
        "budget_pass": budget_pass,
        "paired_gpu_hour_cap": paired_cap,
        "actual_pair_gpu_hours": actual_pair_gpu_hours,
        "fixed": fixed,
        "rematch": rematch,
        "directional_diagnostics": {
            "fixed_identity_error": fixed["identity_error"],
            "rematch_identity_error": rematch_identity,
            "identity_error_relative_reduction": relative_reduction,
            "fixed_average_map_pct": fixed["average_map_pct"],
            "rematch_average_map_pct": rematch["average_map_pct"],
            "map_delta_points": (
                fixed["average_map_pct"] - rematch["average_map_pct"]
            ),
        },
        "reporting_accessed": False,
        "effectiveness_claim_authorized": False,
        "raw_rgb_authorized": False,
        "next_gate": (
            "budgeted_multi_epoch_feature_protocol"
            if technical_pass
            else "diagnose_without_reporting_or_threshold_changes"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--rematch", required=True)
    parser.add_argument("--resource-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = evaluate_screen(
            _load(args.fixed),
            _load(args.rematch),
            _load(args.resource_report),
        )
    except Exception as error:
        payload = {
            "schema_version": "persistent_binding_seed705_screen_gate.v1",
            "screen_pass": False,
            "technical_pass": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "effectiveness_claim_authorized": False,
            "raw_rgb_authorized": False,
        }
        with output.open("x", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
        raise
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    if not payload["screen_pass"]:
        raise SystemExit("seed-705 screen failed its frozen technical gate")


if __name__ == "__main__":
    main()
