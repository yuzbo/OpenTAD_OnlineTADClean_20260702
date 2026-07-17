"""Derived capacity and resource matching checks for Prefix-Route Protocol V2."""

from __future__ import annotations

import math


FAIRNESS_SCHEMA = "prefix-route-arm-resource-audit-v2"
CORE_ARMS = ("B0", "B1", "B2", "B3", "B4")
PARAMETER_TOLERANCE = 0.05
RESOURCE_TOLERANCE = 0.10
LATENCY_UPPER_RATIO = 1.10


class PrefixRouteFairnessError(ValueError):
    pass


def _finite_nonnegative(value, label):
    if isinstance(value, bool):
        raise PrefixRouteFairnessError(f"{label} must be finite and non-negative")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PrefixRouteFairnessError(
            f"{label} must be finite and non-negative"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise PrefixRouteFairnessError(f"{label} must be finite and non-negative")
    return parsed


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PrefixRouteFairnessError(f"{label} must be a positive integer")
    return value


def _relative_difference(value, anchor):
    if anchor == 0:
        return 0.0 if value == 0 else math.inf
    return abs(value - anchor) / anchor


def derive_fairness_audit(rows, parameter_inventory):
    """Compute all fairness results from measured rows and parameter inventories."""

    required = {
        "arm",
        "trainable_parameters",
        "train_macs_per_optimizer_event",
        "inference_macs_per_decision",
        "live_causal_state_bytes",
        "peak_training_memory_bytes",
        "peak_inference_memory_bytes",
        "latency_median_ms",
        "latency_p95_ms",
        "optimizer_event_count",
        "effective_token_count",
        "gradient_accumulation_steps",
        "hyperparameter_trial_count",
        "calibration_video_count",
        "seed_count",
    }
    by_arm = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required:
            raise PrefixRouteFairnessError(
                f"resource row {index} fields differ from frozen schema"
            )
        arm = row["arm"]
        if arm not in CORE_ARMS or arm in by_arm:
            raise PrefixRouteFairnessError("resource arm is invalid or duplicated")
        normalized = {"arm": arm}
        normalized["trainable_parameters"] = _positive_int(
            row["trainable_parameters"],
            f"{arm}.trainable_parameters",
        )
        for field in (
            "train_macs_per_optimizer_event",
            "inference_macs_per_decision",
            "live_causal_state_bytes",
            "peak_training_memory_bytes",
            "peak_inference_memory_bytes",
            "latency_median_ms",
            "latency_p95_ms",
        ):
            normalized[field] = _finite_nonnegative(row[field], f"{arm}.{field}")
        for field in (
            "optimizer_event_count",
            "effective_token_count",
            "gradient_accumulation_steps",
            "hyperparameter_trial_count",
            "calibration_video_count",
            "seed_count",
        ):
            normalized[field] = _positive_int(row[field], f"{arm}.{field}")
        if normalized["latency_p95_ms"] < normalized["latency_median_ms"]:
            raise PrefixRouteFairnessError("p95 latency is below median latency")
        by_arm[arm] = normalized
    if set(by_arm) != set(CORE_ARMS):
        raise PrefixRouteFairnessError("B0-B4 resource rows are incomplete")

    inventory_required = {
        "name",
        "numel",
        "requires_grad",
        "used_by_forward",
        "gradient_observed_in_contract_smoke",
        "purpose",
    }
    inventory_by_arm = {}
    for arm, entries in parameter_inventory.items():
        if arm not in CORE_ARMS or not isinstance(entries, list) or not entries:
            raise PrefixRouteFairnessError("parameter inventory arm is invalid")
        names = set()
        total = 0
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != inventory_required:
                raise PrefixRouteFairnessError("parameter inventory fields differ")
            name = entry["name"]
            if not isinstance(name, str) or not name or name in names:
                raise PrefixRouteFairnessError("parameter name is invalid or duplicated")
            names.add(name)
            numel = _positive_int(entry["numel"], f"{arm}.{name}.numel")
            if entry["requires_grad"] is not True:
                raise PrefixRouteFairnessError(
                    "inventory may contain only trainable parameters"
                )
            if entry["used_by_forward"] is not True:
                raise PrefixRouteFairnessError(
                    f"dummy or unused parameter detected: {arm}.{name}"
                )
            if entry["gradient_observed_in_contract_smoke"] is not True:
                raise PrefixRouteFairnessError(
                    f"no gradient evidence for parameter: {arm}.{name}"
                )
            if not isinstance(entry["purpose"], str) or not entry["purpose"]:
                raise PrefixRouteFairnessError("parameter purpose is missing")
            total += numel
        if total != by_arm[arm]["trainable_parameters"]:
            raise PrefixRouteFairnessError(
                f"parameter inventory total differs for {arm}"
            )
        inventory_by_arm[arm] = {"parameter_count": total, "tensor_count": len(entries)}
    if set(inventory_by_arm) != set(CORE_ARMS):
        raise PrefixRouteFairnessError("B0-B4 parameter inventories are incomplete")

    anchor = by_arm["B2"]
    exact_fields = (
        "optimizer_event_count",
        "effective_token_count",
        "gradient_accumulation_steps",
        "hyperparameter_trial_count",
        "calibration_video_count",
        "seed_count",
    )
    relative_fields = {
        "trainable_parameters": PARAMETER_TOLERANCE,
        "train_macs_per_optimizer_event": RESOURCE_TOLERANCE,
        "inference_macs_per_decision": RESOURCE_TOLERANCE,
        "live_causal_state_bytes": RESOURCE_TOLERANCE,
        "peak_training_memory_bytes": RESOURCE_TOLERANCE,
        "peak_inference_memory_bytes": RESOURCE_TOLERANCE,
    }
    checks = {}
    for arm in CORE_ARMS:
        row = by_arm[arm]
        exact = {
            field: row[field] == anchor[field] for field in exact_fields
        }
        relative = {
            field: {
                "relative_difference": _relative_difference(
                    row[field],
                    anchor[field],
                ),
                "tolerance": tolerance,
                "pass": _relative_difference(row[field], anchor[field]) <= tolerance,
            }
            for field, tolerance in relative_fields.items()
        }
        latency = {
            "median_ratio_to_B2": (
                row["latency_median_ms"] / anchor["latency_median_ms"]
                if anchor["latency_median_ms"] > 0
                else math.inf
            ),
            "p95_ratio_to_B2": (
                row["latency_p95_ms"] / anchor["latency_p95_ms"]
                if anchor["latency_p95_ms"] > 0
                else math.inf
            ),
        }
        latency["pass"] = (
            latency["median_ratio_to_B2"] <= LATENCY_UPPER_RATIO
            and latency["p95_ratio_to_B2"] <= LATENCY_UPPER_RATIO
        )
        checks[arm] = {
            "exact_budget_checks": exact,
            "relative_resource_checks": relative,
            "latency_check": latency,
            "pass": (
                all(exact.values())
                and all(item["pass"] for item in relative.values())
                and latency["pass"]
            ),
        }
    return {
        "schema_version": FAIRNESS_SCHEMA,
        "anchor_arm": "B2",
        "parameter_relative_tolerance": PARAMETER_TOLERANCE,
        "resource_relative_tolerance": RESOURCE_TOLERANCE,
        "latency_upper_ratio": LATENCY_UPPER_RATIO,
        "rows": by_arm,
        "parameter_inventory_summary": inventory_by_arm,
        "checks": checks,
        "status": (
            "PASS_BOTH_CAPACITY_AND_RESOURCE_MATCHED"
            if all(result["pass"] for result in checks.values())
            else "FAIL_ARM_FAIRNESS"
        ),
    }


__all__ = [
    "CORE_ARMS",
    "FAIRNESS_SCHEMA",
    "LATENCY_UPPER_RATIO",
    "PARAMETER_TOLERANCE",
    "RESOURCE_TOLERANCE",
    "PrefixRouteFairnessError",
    "derive_fairness_audit",
]
