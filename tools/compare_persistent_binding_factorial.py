"""Compare the frozen reserve6 calibration-by-boundary 2x2 pilot."""

import argparse
import json
import math
from pathlib import Path


VARIANTS = {
    "reserve": (0, 0),
    "calibration": (1, 0),
    "boundary": (0, 1),
    "boundary_calibration": (1, 1),
}
DEPLOYMENT_VARIANTS = {
    "reserve": ("reserve",),
    "calibration": ("calibration", "calibration_batched"),
    "boundary": ("boundary",),
    "boundary_calibration": (
        "boundary_calibration",
        "boundary_calibration_batched",
    ),
}
ARMS = {
    "fixed": "fixed_birth_slot",
    "rematch": "prefix_rematch_active_pool",
}
COMMON_HASHES = (
    "annotation_sha256",
    "calibration_manifest_sha256",
    "fit_manifest_sha256",
    "reporting_manifest_sha256",
    "census_sha256",
)
FACTORIAL_METRICS = (
    "committed_predictions",
    "prediction_gt_ratio",
    "recall_tiou_0p3",
    "average_map_pct",
)


def _load(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _mechanism_gate(variant, diagnoses):
    failures = []
    needs_calibration = VARIANTS[variant][0] == 1
    needs_boundary = VARIANTS[variant][1] == 1
    for arm, diagnosis in diagnoses.items():
        if diagnosis.get("reporting_accessed") is not False:
            failures.append(f"{arm}: score diagnosis accessed reporting")
        if diagnosis.get("raw_rgb_authorized") is not False:
            failures.append(f"{arm}: score diagnosis authorized raw RGB")
        if diagnosis.get("binding_mode") != ARMS[arm]:
            failures.append(f"{arm}: score diagnosis binding mismatch")
        if needs_calibration:
            audit = diagnosis.get("lifecycle_calibration_invariance", {})
            if audit.get("mode") != "monotone_affine":
                failures.append(f"{arm}: monotone calibration did not activate")
            if audit.get("passed") is not True:
                failures.append(f"{arm}: calibration AUC invariance failed")
            scales = audit.get("scale")
            if not isinstance(scales, list) or len(scales) != 3 or not all(
                math.isfinite(float(value)) and float(value) > 0
                for value in scales
            ):
                failures.append(f"{arm}: calibration scales are not positive")
        if needs_boundary:
            audit = diagnosis.get("boundary_factorization", {})
            pointer = audit.get("endpoint_pointer", {})
            if audit.get("end_transition_mode") != "causal_delta_mlp":
                failures.append(f"{arm}: causal transition end did not activate")
            if audit.get("endpoint_start_mode") != "past_pointer":
                failures.append(f"{arm}: endpoint pointer did not activate")
            if int(pointer.get("decisions", 0)) <= 0:
                failures.append(f"{arm}: endpoint pointer has no decisions")
            if pointer.get("past_only") is not True:
                failures.append(f"{arm}: endpoint pointer accessed future memory")
            if audit.get("runtime_state_contains_gt") is not False:
                failures.append(f"{arm}: runtime state contains GT")
    return {
        "passed": not failures,
        "failures": failures,
    }


def _arm_row(screen, diagnosis, arm):
    row = screen.get(arm)
    if not isinstance(row, dict):
        raise ValueError(f"screen gate lacks {arm} result")
    if row.get("binding_mode") != ARMS[arm]:
        raise ValueError(f"screen gate {arm} binding mismatch")
    target = diagnosis.get("target_conditioned_score_distributions")
    if not isinstance(target, dict):
        raise ValueError(f"{arm} diagnosis lacks target-conditioned scores")
    channels = {}
    for channel in ("birth", "alive", "end"):
        channel_row = target.get(channel)
        if not isinstance(channel_row, dict):
            raise ValueError(f"{arm} diagnosis lacks {channel}")
        channels[channel] = {
            "pairwise_auc": _finite(
                channel_row["pairwise_auc"],
                f"{arm}.{channel}.pairwise_auc",
            ),
            "threshold_true_positive_rate": _finite(
                channel_row["threshold_true_positive_rate"],
                f"{arm}.{channel}.threshold_true_positive_rate",
            ),
            "threshold_false_positive_rate": _finite(
                channel_row["threshold_false_positive_rate"],
                f"{arm}.{channel}.threshold_false_positive_rate",
            ),
        }
    metrics = {
        key: _finite(row[key], f"{arm}.{key}")
        for key in FACTORIAL_METRICS
    }
    metrics["identity_error"] = 0.5 * (
        _finite(row["duplicate_rate"], f"{arm}.duplicate_rate")
        + _finite(row["fragmentation_rate"], f"{arm}.fragmentation_rate")
    )
    return {
        "metrics": metrics,
        "channels": channels,
        "lifecycle_counts": dict(diagnosis.get("lifecycle_counts", {})),
    }


def _normalize_run(root, variant):
    root = Path(root).resolve()
    contract = _load(root / "pilot_contract.json")
    activation = _load(root / "optimization_activation.json")
    screen = _load(root / "screen_gate.json")
    diagnoses = {
        arm: _load(root / f"{arm}_score_diagnosis.json")
        for arm in ARMS
    }
    deployment_variant = contract.get("variant")
    if deployment_variant not in DEPLOYMENT_VARIANTS[variant]:
        raise ValueError(f"{variant}: pilot contract variant mismatch")
    if contract.get("reporting_accessed") is not False:
        raise ValueError(f"{variant}: pilot contract accessed reporting")
    if contract.get("threshold_search") is not False:
        raise ValueError(f"{variant}: pilot contract searched thresholds")
    if contract.get("raw_rgb_authorized") is not False:
        raise ValueError(f"{variant}: pilot contract authorized raw RGB")
    if activation.get("variant") != deployment_variant:
        raise ValueError(f"{variant}: activation variant mismatch")
    if screen.get("reporting_accessed") is not False:
        raise ValueError(f"{variant}: screen accessed reporting")
    if screen.get("raw_rgb_authorized") is not False:
        raise ValueError(f"{variant}: screen authorized raw RGB")
    mechanism = _mechanism_gate(variant, diagnoses)
    technical_pass = screen.get("technical_pass") is True
    learning_readiness_pass = screen.get(
        "learning_readiness_pass",
        technical_pass,
    ) is True
    # v1 called the epoch-1 operational gate "technical_pass". v2 keeps
    # fixed-threshold operation diagnostic until the formal multi-epoch gate.
    operational_pass = screen.get(
        "operational_pass",
        technical_pass,
    ) is True
    activation_pass = activation.get("passed") is True
    learning_ready = (
        learning_readiness_pass
        and activation_pass
        and mechanism["passed"]
    )
    eligible = operational_pass and activation_pass and mechanism["passed"]
    return {
        "root": str(root),
        "factor_levels": {
            "calibration": VARIANTS[variant][0],
            "boundary": VARIANTS[variant][1],
        },
        "deployment_variant": deployment_variant,
        "code_commit": contract.get("code_commit"),
        "technical_pass": technical_pass,
        "learning_readiness_pass": learning_readiness_pass,
        "operational_pass": operational_pass,
        "activation_pass": activation_pass,
        "mechanism_gate": mechanism,
        "multi_epoch_learning_ready": learning_ready,
        "eligible": eligible,
        "technical_failures": list(screen.get("technical_failures", [])),
        "operational_failures": list(
            screen.get("operational_failures", [])
        ),
        "actual_pair_gpu_hours": _finite(
            screen["actual_pair_gpu_hours"],
            f"{variant}.actual_pair_gpu_hours",
        ),
        "common_hashes": {
            key: screen["fixed"]["provenance"].get(key)
            for key in COMMON_HASHES
        },
        "arms": {
            arm: _arm_row(screen, diagnoses[arm], arm)
            for arm in ARMS
        },
    }


def _factorial_effect(values):
    reserve = float(values["reserve"])
    calibration = float(values["calibration"])
    boundary = float(values["boundary"])
    interaction = float(values["boundary_calibration"])
    return {
        "cells": {key: float(value) for key, value in values.items()},
        "calibration_main_effect": 0.5
        * ((calibration - reserve) + (interaction - boundary)),
        "boundary_main_effect": 0.5
        * ((boundary - reserve) + (interaction - calibration)),
        "interaction": interaction - calibration - boundary + reserve,
    }


def compare_factorial(run_paths):
    if set(run_paths) != set(VARIANTS):
        raise ValueError("factorial comparison requires exactly E/F/G/H runs")
    runs = {
        variant: _normalize_run(run_paths[variant], variant)
        for variant in VARIANTS
    }
    reference_hashes = runs["reserve"]["common_hashes"]
    for variant, row in runs.items():
        if row["common_hashes"] != reference_hashes:
            raise ValueError(f"{variant}: frozen data provenance differs")
    calibration_batched = (
        runs["calibration"]["deployment_variant"] == "calibration_batched"
    )
    interaction_batched = (
        runs["boundary_calibration"]["deployment_variant"]
        == "boundary_calibration_batched"
    )
    if calibration_batched != interaction_batched:
        raise ValueError(
            "calibration and interaction cells must use the same aggregation"
        )

    factorial = {}
    for arm in ARMS:
        factorial[arm] = {}
        for metric in FACTORIAL_METRICS + ("identity_error",):
            factorial[arm][metric] = _factorial_effect(
                {
                    variant: runs[variant]["arms"][arm]["metrics"][metric]
                    for variant in VARIANTS
                }
            )
        for channel in ("birth", "alive", "end"):
            for metric in (
                "pairwise_auc",
                "threshold_true_positive_rate",
                "threshold_false_positive_rate",
            ):
                key = f"{channel}_{metric}"
                factorial[arm][key] = _factorial_effect(
                    {
                        variant: runs[variant]["arms"][arm]["channels"][
                            channel
                        ][metric]
                        for variant in VARIANTS
                    }
                )

    interaction_eligible = runs["boundary_calibration"]["eligible"]
    interaction_learning_ready = runs["boundary_calibration"][
        "multi_epoch_learning_ready"
    ]
    selected_variant = runs["boundary_calibration"]["deployment_variant"]
    return {
        "schema_version": "persistent_binding_factorial_comparison.v2",
        "design": (
            "reserve6_calibration_batched_x_boundary_2x2_v2"
            if calibration_batched
            else "reserve6_calibration_x_boundary_2x2_v1"
        ),
        "variants": runs,
        "factorial_effects": factorial,
        "selected_variant": (
            selected_variant if interaction_eligible else None
        ),
        "multi_epoch_candidate": (
            selected_variant if interaction_learning_ready else None
        ),
        "multi_epoch_feature_training_authorized_next": (
            interaction_learning_ready
        ),
        "multi_seed_authorized_next": interaction_eligible,
        "raw_rgb_authorized_next": False,
        "reporting_accessed": False,
        "threshold_search": False,
        "next_gate": (
            "multi_seed_feature_validation"
            if interaction_eligible
            else (
                "paired_12_epoch_feature_training"
                if interaction_learning_ready
                else "diagnose_feature_factorial_without_threshold_search"
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve-run", required=True)
    parser.add_argument("--calibration-run", required=True)
    parser.add_argument("--boundary-run", required=True)
    parser.add_argument("--interaction-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = compare_factorial(
        {
            "reserve": args.reserve_run,
            "calibration": args.calibration_run,
            "boundary": args.boundary_run,
            "boundary_calibration": args.interaction_run,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
