#!/usr/bin/env python3
"""Evaluate separate Full PETAL C1, C2, and project result gates."""

import argparse
import json
import math
from pathlib import Path


SCHEMA_VERSION = "full_petal_result_gate.v1"

C1_VARIANTS = {
    "fixed": "fixed",
    "persist_fixed": "fixed",
    "persistent_fixed": "fixed",
    "fixed_birth_slot": "fixed",
    "thumos_pes_q2_persist_fixed": "fixed",
    "rematch": "rematch",
    "persist_rematch": "rematch",
    "persistent_rematch": "rematch",
    "prefix_rematch_active_pool": "rematch",
    "thumos_pes_q2_persist_rematch": "rematch",
}
C2_VARIANTS = {
    "frozen": "frozen",
    "frozen_encoder": "frozen",
    "raw_frozen": "frozen",
    "adapted": "adapted",
    "peft": "adapted",
    "adapter": "adapted",
    "lora": "adapted",
    "top_block": "adapted",
    "visual_adapted": "adapted",
}

METRIC_ALIASES = {
    "average_mAP": ("average_mAP", "average_mOnlineAP", "mAP", "map"),
    "recall": ("recall", "instance_recall"),
    "duplicate_rate": ("duplicate_rate",),
    "fragmentation_rate": ("fragmentation_rate",),
    "false_emission_rate": ("false_emission_rate", "unmatched_emission_rate"),
    "endpoint_latency_frames_mean": (
        "endpoint_latency_frames_mean",
        "endpoint_detection_latency_frames_mean",
    ),
    "high_tiou_mAP": ("high_tiou_mAP", "mAP@0.7", "map_at_0.7"),
    "short_action_mAP": ("short_action_mAP",),
}


class ResultGateError(ValueError):
    """Base class for result-gate validation errors."""


class ResultGateInputError(ResultGateError):
    """Raised for malformed or incomplete result artifacts."""


class ResultGateProtocolError(ResultGateError):
    """Raised when an artifact records a protocol violation or mismatch."""


def _canonical_json(value, label):
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ResultGateInputError(f"{label} must be canonical JSON data: {exc}") from exc


def _finite_number(value, label, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise ResultGateInputError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ResultGateInputError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ResultGateInputError(f"{label} must be a finite number")
    if minimum is not None and number < minimum:
        raise ResultGateInputError(f"{label} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ResultGateInputError(f"{label} must be <= {maximum}")
    return number


def _mean(values):
    values = list(values)
    return sum(values) / len(values)


def _claim_name(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"c1", "mechanism", "c1_mechanism"}:
        return "C1"
    if normalized in {"c2", "raw_video", "c2_raw_video"}:
        return "C2"
    raise ResultGateInputError(f"unexpected claim: {value!r}")


def _variant_name(claim, value):
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = C1_VARIANTS if claim == "C1" else C2_VARIANTS
    if normalized not in aliases:
        raise ResultGateInputError(f"unexpected {claim} variant: {value!r}")
    return aliases[normalized]


def _seed(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResultGateInputError(f"{label} must be a non-negative integer")
    return value


def _has_violations(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _finite_number(value, "protocol_violations", minimum=0.0) > 0
    if isinstance(value, (list, tuple, set, dict, str)):
        return bool(value)
    raise ResultGateInputError("protocol_violations has an unsupported type")


def _validate_row_protocol(row, label):
    metrics = row.get("metrics")
    violation_values = [row.get("protocol_violations")]
    if isinstance(metrics, dict):
        violation_values.append(metrics.get("protocol_violations"))
        causal = metrics.get("causal_validation")
        if isinstance(causal, dict):
            if causal.get("passed") is False:
                raise ResultGateProtocolError(f"{label} failed causal emission validation")
            violation_values.append(causal.get("violations"))
            counts = causal.get("violation_counts")
            if isinstance(counts, dict) and any(
                _finite_number(value, f"{label}.causal_validation.{key}", minimum=0.0) > 0
                for key, value in counts.items()
            ):
                raise ResultGateProtocolError(f"{label} records causal protocol violations")
    if any(_has_violations(value) for value in violation_values):
        raise ResultGateProtocolError(f"{label} records protocol violations")

    protocol = row.get("protocol")
    if not isinstance(protocol, dict) or not protocol:
        raise ResultGateInputError(f"{label}.protocol must be a non-empty object")
    if protocol.get("offline_nms") is True or protocol.get("nms") not in (None, False):
        raise ResultGateProtocolError(f"{label} enables forbidden NMS")
    if protocol.get("immutable_emissions") is False:
        raise ResultGateProtocolError(f"{label} disables immutable emissions")
    if protocol.get("load_from_raw_predictions") is True:
        raise ResultGateProtocolError(f"{label} enables raw-prediction loading")
    _canonical_json(protocol, f"{label}.protocol")
    return protocol


def _metric_candidates(metrics, canonical_name):
    candidates = []
    for alias in METRIC_ALIASES[canonical_name]:
        if alias in metrics:
            candidates.append((alias, metrics[alias]))
    if canonical_name == "endpoint_latency_frames_mean":
        summary = metrics.get("endpoint_detection_latency_frames")
        if isinstance(summary, dict) and "mean" in summary:
            candidates.append(("endpoint_detection_latency_frames.mean", summary["mean"]))
    return candidates


def _metric(metrics, canonical_name, label, required=True):
    candidates = _metric_candidates(metrics, canonical_name)
    if not candidates:
        if required:
            raise ResultGateInputError(f"{label} is missing metric {canonical_name}")
        return None
    values = [
        _finite_number(
            value,
            f"{label}.{name}",
            minimum=(None if canonical_name == "endpoint_latency_frames_mean" else 0.0),
            maximum=(
                None
                if canonical_name == "endpoint_latency_frames_mean"
                else 1.0
            ),
        )
        for name, value in candidates
    ]
    if any(abs(value - values[0]) > 1e-12 for value in values[1:]):
        names = [name for name, _ in candidates]
        raise ResultGateInputError(f"{label} has conflicting aliases for {canonical_name}: {names}")
    return values[0]


def _normalized_metrics(row, claim, label):
    metrics = row.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ResultGateInputError(f"{label}.metrics must be a non-empty object")
    if claim == "C1":
        required = (
            "average_mAP",
            "recall",
            "duplicate_rate",
            "fragmentation_rate",
            "false_emission_rate",
            "endpoint_latency_frames_mean",
        )
        return {name: _metric(metrics, name, label) for name in required}
    normalized = {"average_mAP": _metric(metrics, "average_mAP", label)}
    for name in ("high_tiou_mAP", "short_action_mAP"):
        value = _metric(metrics, name, label, required=False)
        if value is not None:
            normalized[name] = value
    return normalized


def _normalized_cost(row, label):
    cost = row.get("cost")
    if not isinstance(cost, dict) or not cost:
        raise ResultGateInputError(f"{label}.cost must be a non-empty object for C1")
    normalized = {}
    for key, value in cost.items():
        normalized[str(key)] = _finite_number(value, f"{label}.cost.{key}", minimum=0.0)
    return normalized


def _collect_payloads(value):
    payloads = [value] if isinstance(value, dict) else list(value) if isinstance(value, (list, tuple)) else None
    if payloads is None or not payloads:
        raise ResultGateInputError("result artifacts must be a non-empty object or sequence")
    runs = []
    containers = []
    for index, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            raise ResultGateInputError(f"artifact[{index}] must be a JSON object")
        if "claim" in payload:
            runs.append(payload)
            continue
        containers.append(payload)
        artifact_runs = payload.get("runs", ())
        if not isinstance(artifact_runs, (list, tuple)):
            raise ResultGateInputError(f"artifact[{index}].runs must be a sequence")
        runs.extend(artifact_runs)
    return containers, runs


def _status_value(value, label):
    if isinstance(value, bool):
        return value, []
    if not isinstance(value, dict):
        raise ResultGateInputError(f"{label} must be a boolean or object")
    if "passed" not in value or not isinstance(value["passed"], bool):
        raise ResultGateInputError(f"{label}.passed must be a boolean")
    violations = value.get("violations", [])
    if not isinstance(violations, (list, tuple)):
        raise ResultGateInputError(f"{label}.violations must be a sequence")
    return value["passed"], list(violations)


def _prerequisite_entries(container):
    entries = {"protocol": [], "B0": []}
    prerequisites = container.get("prerequisites")
    if prerequisites is not None:
        if not isinstance(prerequisites, dict):
            raise ResultGateInputError("prerequisites must be an object")
        if "protocol" in prerequisites:
            entries["protocol"].append(
                _status_value(prerequisites["protocol"], "prerequisites.protocol")
            )
        b0_key = "B0" if "B0" in prerequisites else "b0" if "b0" in prerequisites else None
        if b0_key is not None:
            entries["B0"].append(
                _status_value(prerequisites[b0_key], f"prerequisites.{b0_key}")
            )

    legacy = container.get("protocol_b0")
    if legacy is not None:
        if not isinstance(legacy, dict):
            raise ResultGateInputError("protocol_b0 must be an object")
        if "protocol_passed" in legacy:
            entries["protocol"].append(
                _status_value(legacy["protocol_passed"], "protocol_b0.protocol_passed")
            )
        if "b0_passed" in legacy:
            entries["B0"].append(
                _status_value(legacy["b0_passed"], "protocol_b0.b0_passed")
            )
    if "protocol_passed" in container:
        entries["protocol"].append(
            _status_value(container["protocol_passed"], "protocol_passed")
        )
    if "b0_passed" in container:
        entries["B0"].append(_status_value(container["b0_passed"], "b0_passed"))
    return entries


def _normalize_prerequisites(containers):
    combined = {"protocol": [], "B0": []}
    for container in containers:
        entries = _prerequisite_entries(container)
        combined["protocol"].extend(entries["protocol"])
        combined["B0"].extend(entries["B0"])

    result = {}
    for name in ("protocol", "B0"):
        entries = combined[name]
        values = {passed for passed, _ in entries}
        if len(values) > 1:
            raise ResultGateInputError(f"conflicting {name} prerequisite statuses")
        violations = [violation for _, found in entries for violation in found]
        passed = next(iter(values)) if values else False
        if name == "protocol" and (violations or (entries and not passed)):
            reason = violations or ["protocol prerequisite did not pass"]
            raise ResultGateProtocolError(f"protocol prerequisite violation: {reason}")
        result[name] = {
            "provided": bool(entries),
            "passed": bool(passed),
            "violations": violations,
        }
    return result


def _normalize_runs(raw_runs):
    grouped = {
        "C1": {"fixed": {}, "rematch": {}},
        "C2": {"frozen": {}, "adapted": {}},
    }
    for index, row in enumerate(raw_runs):
        if not isinstance(row, dict):
            raise ResultGateInputError(f"runs[{index}] must be a JSON object")
        if "claim" not in row or "variant" not in row or "seed" not in row:
            raise ResultGateInputError(
                f"runs[{index}] requires claim, variant, and seed fields"
            )
        claim = _claim_name(row["claim"])
        variant = _variant_name(claim, row["variant"])
        seed = _seed(row["seed"], f"runs[{index}].seed")
        if seed in grouped[claim][variant]:
            raise ResultGateInputError(f"duplicate {claim}/{variant} run for seed {seed}")
        label = f"runs[{index}]({claim}/{variant}/seed={seed})"
        protocol = _validate_row_protocol(row, label)
        normalized = {
            "seed": seed,
            "protocol": protocol,
            "metrics": _normalized_metrics(row, claim, label),
        }
        if claim == "C1":
            normalized["cost"] = _normalized_cost(row, label)
        grouped[claim][variant][seed] = normalized
    return grouped


def _paired_claim(grouped, claim, variants):
    left_name, right_name = variants
    left = grouped[claim][left_name]
    right = grouped[claim][right_name]
    if not left and not right:
        return None
    left_seeds = set(left)
    right_seeds = set(right)
    if not left_seeds or not right_seeds or left_seeds != right_seeds:
        raise ResultGateInputError(
            f"{claim} paired seed coverage requires identical non-empty seed sets: "
            f"{left_name}={sorted(left_seeds)}, {right_name}={sorted(right_seeds)}"
        )
    seeds = sorted(left_seeds)
    protocol_values = {
        _canonical_json(row["protocol"], f"{claim}.protocol")
        for name in variants
        for row in grouped[claim][name].values()
    }
    if len(protocol_values) != 1:
        raise ResultGateProtocolError(f"{claim} protocol mismatch across paired runs")
    return seeds


def _relative_reduction(candidate, baseline):
    if baseline > 0:
        return (baseline - candidate) / baseline
    return 0.0 if candidate == 0 else -1.0


def _cost_parity(grouped, seeds, tolerance):
    comparisons = []
    passed = True
    for seed in seeds:
        fixed = grouped["C1"]["fixed"][seed]["cost"]
        rematch = grouped["C1"]["rematch"][seed]["cost"]
        if set(fixed) != set(rematch):
            raise ResultGateInputError(
                f"C1 cost fields differ for seed {seed}: "
                f"fixed={sorted(fixed)}, rematch={sorted(rematch)}"
            )
        metrics = {}
        for key in sorted(fixed):
            scale = max(abs(fixed[key]), abs(rematch[key]), 1e-12)
            relative_gap = abs(fixed[key] - rematch[key]) / scale
            metric_pass = relative_gap <= tolerance + 1e-12
            passed = passed and metric_pass
            metrics[key] = {
                "fixed": fixed[key],
                "rematch": rematch[key],
                "relative_gap": relative_gap,
                "passed": metric_pass,
            }
        comparisons.append({"seed": seed, "metrics": metrics})
    return {
        "passed": passed,
        "relative_tolerance": tolerance,
        "per_seed": comparisons,
    }


def _c1_scientific_metrics(
    grouped,
    seeds,
    map_gain_points,
    error_reduction,
    map_parity_tolerance_points,
    recall_tolerance_points,
    false_emission_tolerance,
    latency_tolerance_frames,
):
    per_seed = []
    for seed in seeds:
        fixed = grouped["C1"]["fixed"][seed]["metrics"]
        rematch = grouped["C1"]["rematch"][seed]["metrics"]
        per_seed.append(
            {
                "seed": seed,
                "map_gain_points": 100.0 * (fixed["average_mAP"] - rematch["average_mAP"]),
                "recall_delta_points": 100.0 * (fixed["recall"] - rematch["recall"]),
                "false_emission_rate_delta": (
                    fixed["false_emission_rate"] - rematch["false_emission_rate"]
                ),
                "endpoint_latency_frames_delta": (
                    fixed["endpoint_latency_frames_mean"]
                    - rematch["endpoint_latency_frames_mean"]
                ),
                "error_reduction": {
                    name: _relative_reduction(fixed[name], rematch[name])
                    for name in ("duplicate_rate", "fragmentation_rate")
                },
            }
        )

    aggregate = {
        "map_gain_points": _mean(row["map_gain_points"] for row in per_seed),
        "recall_delta_points": _mean(row["recall_delta_points"] for row in per_seed),
        "false_emission_rate_delta": _mean(
            row["false_emission_rate_delta"] for row in per_seed
        ),
        "endpoint_latency_frames_delta": _mean(
            row["endpoint_latency_frames_delta"] for row in per_seed
        ),
        "error_reduction": {
            name: _mean(row["error_reduction"][name] for row in per_seed)
            for name in ("duplicate_rate", "fragmentation_rate")
        },
    }
    best_error_reduction = max(aggregate["error_reduction"].values())
    map_path = aggregate["map_gain_points"] >= map_gain_points - 1e-12
    error_path = (
        best_error_reduction >= error_reduction - 1e-12
        and aggregate["map_gain_points"] >= -map_parity_tolerance_points - 1e-12
    )
    recall_pass = aggregate["recall_delta_points"] >= -recall_tolerance_points - 1e-12
    false_emission_pass = (
        aggregate["false_emission_rate_delta"] <= false_emission_tolerance + 1e-12
    )
    latency_pass = (
        aggregate["endpoint_latency_frames_delta"] <= latency_tolerance_frames + 1e-12
    )
    passed = (map_path or error_path) and recall_pass and false_emission_pass and latency_pass
    return {
        "passed": passed,
        "effect_paths": {
            "map_gain": map_path,
            "duplicate_or_fragmentation_reduction_at_map_parity": error_path,
        },
        "safety_checks": {
            "recall_noninferior": recall_pass,
            "false_emission_noninferior": false_emission_pass,
            "endpoint_latency_noninferior": latency_pass,
        },
        "aggregate": aggregate,
        "per_seed": per_seed,
        "thresholds": {
            "map_gain_points": map_gain_points,
            "error_reduction_fraction": error_reduction,
            "map_parity_tolerance_points": map_parity_tolerance_points,
            "recall_tolerance_points": recall_tolerance_points,
            "false_emission_rate_tolerance": false_emission_tolerance,
            "endpoint_latency_frames_tolerance": latency_tolerance_frames,
        },
    }


def _not_evaluated(claim, variants):
    return {
        "claim": claim,
        "status": "NOT_EVALUATED",
        "requirements": {
            "paired_seed_coverage": False,
            "protocol_equality": False,
            "scientific_metrics": False,
            **({"cost_parity": False} if claim == "C1" else {}),
        },
        "paired_seed_coverage": {
            "passed": False,
            "seeds": [],
            "variants": list(variants),
        },
        "reasons": [f"no {claim} runs were supplied"],
    }


def _evaluate_c1(
    grouped,
    cost_tolerance,
    map_gain_points,
    error_reduction,
    map_parity_tolerance_points,
    recall_tolerance_points,
    false_emission_tolerance,
    latency_tolerance_frames,
):
    variants = ("fixed", "rematch")
    seeds = _paired_claim(grouped, "C1", variants)
    if seeds is None:
        return _not_evaluated("C1", variants)
    cost = _cost_parity(grouped, seeds, cost_tolerance)
    scientific = _c1_scientific_metrics(
        grouped,
        seeds,
        map_gain_points,
        error_reduction,
        map_parity_tolerance_points,
        recall_tolerance_points,
        false_emission_tolerance,
        latency_tolerance_frames,
    )
    passed = cost["passed"] and scientific["passed"]
    reasons = []
    if not scientific["passed"]:
        reasons.append("C1 scientific effect or non-inferiority checks did not pass")
    if not cost["passed"]:
        reasons.append("C1 fixed/rematch cost parity did not pass")
    requirements = {
        "paired_seed_coverage": True,
        "protocol_equality": True,
        "scientific_metrics": scientific["passed"],
        "cost_parity": cost["passed"],
    }
    return {
        "claim": "C1",
        "status": "PASS" if passed else "FAIL",
        "requirements": requirements,
        "paired_seed_coverage": {
            "passed": True,
            "seeds": seeds,
            "variants": list(variants),
        },
        "protocol_equality": {"passed": True},
        "scientific_metrics": scientific,
        "cost_parity": cost,
        "reasons": reasons,
    }


def _evaluate_c2(grouped, map_gain_points, specialized_gain_points, map_parity_tolerance_points):
    variants = ("adapted", "frozen")
    seeds = _paired_claim(grouped, "C2", variants)
    if seeds is None:
        return _not_evaluated("C2", variants)

    per_seed = []
    specialized_names = set()
    for seed in seeds:
        adapted = grouped["C2"]["adapted"][seed]["metrics"]
        frozen = grouped["C2"]["frozen"][seed]["metrics"]
        row = {
            "seed": seed,
            "map_gain_points": 100.0 * (adapted["average_mAP"] - frozen["average_mAP"]),
            "specialized_gain_points": {},
        }
        for name in ("high_tiou_mAP", "short_action_mAP"):
            if (name in adapted) != (name in frozen):
                raise ResultGateInputError(
                    f"C2 paired runs must both report {name} for seed {seed} or neither"
                )
            if name in adapted:
                specialized_names.add(name)
                row["specialized_gain_points"][name] = 100.0 * (
                    adapted[name] - frozen[name]
                )
        per_seed.append(row)

    aggregate_map_gain = _mean(row["map_gain_points"] for row in per_seed)
    aggregate_specialized = {
        name: _mean(row["specialized_gain_points"][name] for row in per_seed)
        for name in sorted(specialized_names)
    }
    map_path = aggregate_map_gain >= map_gain_points - 1e-12
    specialized_path = bool(aggregate_specialized) and (
        max(aggregate_specialized.values()) >= specialized_gain_points - 1e-12
        and aggregate_map_gain >= -map_parity_tolerance_points - 1e-12
    )
    scientific_pass = map_path or specialized_path
    scientific = {
        "passed": scientific_pass,
        "effect_paths": {
            "average_map_gain": map_path,
            "preregistered_specialized_gain_at_map_parity": specialized_path,
        },
        "aggregate": {
            "map_gain_points": aggregate_map_gain,
            "specialized_gain_points": aggregate_specialized,
        },
        "per_seed": per_seed,
        "thresholds": {
            "map_gain_points": map_gain_points,
            "specialized_gain_points": specialized_gain_points,
            "map_parity_tolerance_points": map_parity_tolerance_points,
        },
    }
    return {
        "claim": "C2",
        "status": "PASS" if scientific_pass else "FAIL",
        "requirements": {
            "paired_seed_coverage": True,
            "protocol_equality": True,
            "scientific_metrics": scientific_pass,
        },
        "paired_seed_coverage": {
            "passed": True,
            "seeds": seeds,
            "variants": list(variants),
        },
        "protocol_equality": {"passed": True},
        "scientific_metrics": scientific,
        "reasons": [] if scientific_pass else ["C2 raw-video scientific checks did not pass"],
    }


def evaluate_result_gates(
    artifacts,
    c1_map_gain_points=2.0,
    c1_error_reduction=0.20,
    c1_map_parity_tolerance_points=0.5,
    c1_recall_tolerance_points=1.0,
    c1_false_emission_tolerance=0.01,
    c1_latency_tolerance_frames=0.0,
    c1_cost_relative_tolerance=0.05,
    c2_map_gain_points=2.0,
    c2_specialized_gain_points=2.0,
    c2_map_parity_tolerance_points=0.5,
):
    """Return independent claim decisions; never collapse them into one pass flag."""

    thresholds = {
        "c1_map_gain_points": c1_map_gain_points,
        "c1_error_reduction": c1_error_reduction,
        "c1_map_parity_tolerance_points": c1_map_parity_tolerance_points,
        "c1_recall_tolerance_points": c1_recall_tolerance_points,
        "c1_false_emission_tolerance": c1_false_emission_tolerance,
        "c1_latency_tolerance_frames": c1_latency_tolerance_frames,
        "c1_cost_relative_tolerance": c1_cost_relative_tolerance,
        "c2_map_gain_points": c2_map_gain_points,
        "c2_specialized_gain_points": c2_specialized_gain_points,
        "c2_map_parity_tolerance_points": c2_map_parity_tolerance_points,
    }
    for name, value in thresholds.items():
        thresholds[name] = _finite_number(value, name, minimum=0.0)

    containers, raw_runs = _collect_payloads(artifacts)
    prerequisites = _normalize_prerequisites(containers)
    grouped = _normalize_runs(raw_runs)
    c1 = _evaluate_c1(
        grouped,
        thresholds["c1_cost_relative_tolerance"],
        thresholds["c1_map_gain_points"],
        thresholds["c1_error_reduction"],
        thresholds["c1_map_parity_tolerance_points"],
        thresholds["c1_recall_tolerance_points"],
        thresholds["c1_false_emission_tolerance"],
        thresholds["c1_latency_tolerance_frames"],
    )
    c2 = _evaluate_c2(
        grouped,
        thresholds["c2_map_gain_points"],
        thresholds["c2_specialized_gain_points"],
        thresholds["c2_map_parity_tolerance_points"],
    )
    project_requirements = {
        "C1": c1["status"] == "PASS",
        "C2": c2["status"] == "PASS",
        "protocol": prerequisites["protocol"]["passed"],
        "B0": prerequisites["B0"]["passed"],
    }
    project_pass = all(project_requirements.values())
    project = {
        "status": "PASS" if project_pass else "FAIL",
        "requirements": project_requirements,
        "reasons": [
            f"required decision/prerequisite did not pass: {name}"
            for name, passed in project_requirements.items()
            if not passed
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "decisions": {
            "C1": c1,
            "C2": c2,
            "project_full_system": project,
        },
        "prerequisites": prerequisites,
    }


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResultGateInputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise ResultGateInputError(f"non-finite JSON number is not allowed: {value}")


def _load_json(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
    except ResultGateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResultGateInputError(f"failed to read {path}: {exc}") from exc


def _serialize(payload):
    return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)


def _emit(payload, output=None):
    serialized = _serialize(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="exit nonzero unless the project/full-system decision passes",
    )
    parser.add_argument("--c1-map-gain-points", type=float, default=2.0)
    parser.add_argument("--c1-error-reduction", type=float, default=0.20)
    parser.add_argument("--c1-cost-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--c2-map-gain-points", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        payloads = [_load_json(path) for path in args.artifacts]
        verdict = evaluate_result_gates(
            payloads,
            c1_map_gain_points=args.c1_map_gain_points,
            c1_error_reduction=args.c1_error_reduction,
            c1_cost_relative_tolerance=args.c1_cost_relative_tolerance,
            c2_map_gain_points=args.c2_map_gain_points,
        )
    except ResultGateProtocolError as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "error": {"kind": "protocol_violation", "message": str(exc)},
            },
            args.output,
        )
        return 3
    except ResultGateInputError as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "error": {"kind": "malformed_input", "message": str(exc)},
            },
            args.output,
        )
        return 2

    _emit(verdict, args.output)
    if args.require_pass and verdict["decisions"]["project_full_system"]["status"] != "PASS":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

