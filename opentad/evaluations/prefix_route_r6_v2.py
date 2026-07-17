"""Exact paired inference and terminal decisions for Prefix-Route Protocol V2."""

from __future__ import annotations

import hashlib
import json
import math
import random

from .full_petal_metrics import compute_full_petal_metrics
from .online_budgeted_map import OnlineAPBudgeted


GLOBAL_METRICS = (
    "mOnlineAP",
    "event_recall",
    "false_emission_per_video",
    "duplicate_per_gt",
    "fragmentation_per_gt",
    "endpoint_latency_bins",
)
CONTROL_METRICS = ("class_mOnlineAP",)
STRESS_METRICS = (
    "same_class_repetition_recall",
    "same_class_overlap_recall",
    "same_bin_end_start_recall",
    "direct_complete_recall",
)
HIGHER_IS_BETTER = {
    "mOnlineAP",
    "class_mOnlineAP",
    "event_recall",
    *STRESS_METRICS,
}
LOWER_IS_BETTER = {
    "false_emission_per_video",
    "duplicate_per_gt",
    "fragmentation_per_gt",
    "endpoint_latency_bins",
}
REQUIRED_ARMS = (
    "B2",
    "B3",
    "B4",
    "A1_NO_UOT",
    "A2_BINARY_MASS",
    "A1_A2_JOINT",
    "A3_NO_CONSISTENCY",
    "A4_NO_SAME_BIN_REUSE",
    "A5_NO_NEURAL_LATCH",
)
CONTROL_ARMS = (
    "COUNT_ONLY",
    "TEMPLATE_TIMING",
    "LEDGER_ONLY",
    "HISTORY_OFF",
    "FEATURE_TIME_SHUFFLE",
    "SEMANTIC_DERANGEMENT",
)
INFERENCE_ARMS = REQUIRED_ARMS + CONTROL_ARMS
SIMPLE_CONTROL_ARMS = (
    "COUNT_ONLY",
    "TEMPLATE_TIMING",
    "LEDGER_ONLY",
)
REQUIRED_CONTRASTS = (
    ("B4", "B2"),
    ("B4", "B3"),
    ("B4", "A1_NO_UOT"),
    ("B4", "A2_BINARY_MASS"),
    ("B4", "A1_A2_JOINT"),
    ("B4", "A3_NO_CONSISTENCY"),
    ("B4", "A4_NO_SAME_BIN_REUSE"),
    ("B4", "A5_NO_NEURAL_LATCH"),
    ("B4", "COUNT_ONLY"),
    ("B4", "TEMPLATE_TIMING"),
    ("B4", "LEDGER_ONLY"),
    ("B4", "HISTORY_OFF"),
    ("B4", "FEATURE_TIME_SHUFFLE"),
    ("B4", "SEMANTIC_DERANGEMENT"),
)
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 2026071707
FAMILY_WISE_ALPHA = 0.05
R6_RAW_SCHEMA = "prefix-route-r6-raw-evidence-v2"
R6_RESULT_SCHEMA = "prefix-route-r6-derived-result-v2"
R6_SEEDS = (705, 706, 707)
R6_TIOU_THRESHOLD = 0.5
R6_LATENCY_BUDGET_SEC = 2.0
R6_FEATURE_STRIDE_FRAMES = 8


class PrefixRouteR6Error(ValueError):
    pass


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


def _finite(value, label):
    if isinstance(value, bool):
        raise PrefixRouteR6Error(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PrefixRouteR6Error(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise PrefixRouteR6Error(f"{label} must be finite")
    return parsed


def _nonnegative_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrefixRouteR6Error(f"{label} must be a non-negative integer")
    return value


def _type7_percentile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise PrefixRouteR6Error("cannot compute an empty interval")
    rank = (len(ordered) - 1) * probability
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def canonical_emission(row):
    """Map one immutable model output to the evaluator's exact field names."""

    required = {
        "emission_id",
        "stream_key",
        "sequence_id",
        "start",
        "end",
        "class",
        "score",
        "source_frame",
        "emit_frame",
    }
    if not isinstance(row, dict) or set(row) != required:
        raise PrefixRouteR6Error(
            f"formal emission fields differ from {sorted(required)}"
        )
    start = _finite(row["start"], "start")
    end = _finite(row["end"], "end")
    source = _finite(row["source_frame"], "source_frame")
    emitted = _finite(row["emit_frame"], "emit_frame")
    score = _finite(row["score"], "score")
    if start < 0 or not start < end:
        raise PrefixRouteR6Error("emission interval must be positive")
    if source < 0 or emitted < 0:
        raise PrefixRouteR6Error("emission provenance coordinates must be non-negative")
    if end > emitted:
        raise PrefixRouteR6Error("emission reveals a future endpoint")
    if source > emitted:
        raise PrefixRouteR6Error("source_frame exceeds emit_frame")
    if not 0.0 <= score <= 1.0:
        raise PrefixRouteR6Error("score must be in [0, 1]")
    if not isinstance(row["emission_id"], str) or not row["emission_id"]:
        raise PrefixRouteR6Error("emission_id must be non-empty text")
    if not isinstance(row["stream_key"], str) or not row["stream_key"]:
        raise PrefixRouteR6Error("stream_key must be non-empty text")
    if (
        isinstance(row["sequence_id"], bool)
        or not isinstance(row["sequence_id"], int)
        or row["sequence_id"] < 0
    ):
        raise PrefixRouteR6Error("sequence_id must be a non-negative integer")
    if not isinstance(row["class"], str) or not row["class"]:
        raise PrefixRouteR6Error("class must be non-empty text")
    return {
        **row,
        "start": start,
        "end": end,
        "score": score,
        "source_frame": source,
        "emit_frame": emitted,
    }


def derive_per_video_cell(
    *,
    arm,
    seed,
    video_id,
    ground_truth,
    emissions,
    stress_ground_truth_ids,
    tiou_threshold=0.5,
    fps=30.0,
    latency_budget_sec=2.0,
    feature_stride_frames=8,
):
    """Derive one R6 cell from raw GT and immutable emissions."""

    if arm not in INFERENCE_ARMS:
        raise PrefixRouteR6Error("R6 cell arm is not registered")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PrefixRouteR6Error("R6 cell seed must be an integer")
    if not isinstance(video_id, str) or not video_id:
        raise PrefixRouteR6Error("R6 cell video_id must be non-empty text")
    if (
        isinstance(feature_stride_frames, bool)
        or not isinstance(feature_stride_frames, int)
        or feature_stride_frames <= 0
    ):
        raise PrefixRouteR6Error("feature stride must be a positive integer")
    if not isinstance(ground_truth, list):
        raise PrefixRouteR6Error("ground truth must be an array")
    if not isinstance(emissions, list):
        raise PrefixRouteR6Error("emissions must be an array")
    canonical_emissions = [canonical_emission(dict(row)) for row in emissions]
    metric_emissions = [
        {
            "emission_id": row["emission_id"],
            "stream_key": row["stream_key"],
            "sequence_id": row["sequence_id"],
            "start_frame": row["start"],
            "end_frame": row["end"],
            "label": row["class"],
            "score": row["score"],
            "source_frame": row["source_frame"],
            "emit_frame": row["emit_frame"],
            "immutable": True,
            "fps": float(fps),
        }
        for row in canonical_emissions
    ]
    gt_ids = []
    for index, row in enumerate(ground_truth):
        if not isinstance(row, dict):
            raise PrefixRouteR6Error("ground-truth row must be an object")
        stream = row.get("stream_key", row.get("video_id"))
        if str(stream) != video_id:
            raise PrefixRouteR6Error("ground-truth stream differs from video_id")
        gt_id = row.get("gt_id", row.get("instance_id", row.get("id")))
        if gt_id is None:
            gt_id = f"gt:{index}"
        try:
            hash(gt_id)
        except TypeError as exc:
            raise PrefixRouteR6Error("ground-truth ID must be hashable") from exc
        gt_ids.append(gt_id)
    if len(gt_ids) != len(set(gt_ids)):
        raise PrefixRouteR6Error("ground-truth IDs are duplicated")
    expected_families = {
        "same_class_repetition",
        "same_class_overlap",
        "same_bin_end_start",
        "direct_complete",
    }
    if (
        not isinstance(stress_ground_truth_ids, dict)
        or not set(stress_ground_truth_ids).issubset(expected_families)
    ):
        raise PrefixRouteR6Error("stress family set is invalid")
    gt_id_set = set(gt_ids)
    stress_sets = {}
    for family, values in stress_ground_truth_ids.items():
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise PrefixRouteR6Error("stress IDs must be unique arrays")
        values = set(values)
        if not values.issubset(gt_id_set):
            raise PrefixRouteR6Error("stress IDs are outside ground truth")
        stress_sets[family] = values
    metrics = compute_full_petal_metrics(
        ground_truth,
        metric_emissions,
        tiou_threshold=tiou_threshold,
        fps=fps,
        latency_budget_sec=latency_budget_sec,
    )
    matched_ids = {
        pair["ground_truth_id"] for pair in metrics["matching"]["pairs"]
    }
    endpoint_latencies = [
        pair["endpoint_latency_frames"] / feature_stride_frames
        for pair in metrics["matching"]["pairs"]
    ]
    return {
        "arm": arm,
        "seed": seed,
        "video_id": video_id,
        "ground_truth_count": metrics["counts"]["ground_truth"],
        "matched_ground_truth_count": metrics["counts"]["matched_ground_truth"],
        "false_emission_count": metrics["counts"]["unmatched_emissions"],
        "duplicate_emission_count": metrics["counts"]["duplicate_emissions"],
        "fragmented_ground_truth_count": metrics["fragmentation"][
            "fragmented_ground_truth_count"
        ],
        "endpoint_latency_bins_sum": sum(endpoint_latencies),
        "endpoint_latency_observed_count": len(endpoint_latencies),
        "stress": {
            family: {
                "ground_truth_count": len(values),
                "matched_ground_truth_count": len(values.intersection(matched_ids)),
            }
            for family, values in sorted(stress_sets.items())
        },
    }


def _validate_cell(row, eligible_stress):
    required = {
        "arm",
        "seed",
        "video_id",
        "ground_truth_count",
        "matched_ground_truth_count",
        "false_emission_count",
        "duplicate_emission_count",
        "fragmented_ground_truth_count",
        "endpoint_latency_bins_sum",
        "endpoint_latency_observed_count",
        "stress",
    }
    if not isinstance(row, dict) or set(row) != required:
        raise PrefixRouteR6Error("per-video metric cell fields differ")
    if row["arm"] not in INFERENCE_ARMS:
        raise PrefixRouteR6Error("metric cell arm is not registered")
    if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
        raise PrefixRouteR6Error("seed must be an integer")
    if not isinstance(row["video_id"], str) or not row["video_id"]:
        raise PrefixRouteR6Error("video_id must be non-empty text")
    for field in (
        "ground_truth_count",
        "matched_ground_truth_count",
        "false_emission_count",
        "duplicate_emission_count",
        "fragmented_ground_truth_count",
        "endpoint_latency_observed_count",
    ):
        _nonnegative_int(row[field], field)
    if row["matched_ground_truth_count"] > row["ground_truth_count"]:
        raise PrefixRouteR6Error("matched count exceeds ground truth count")
    if row["fragmented_ground_truth_count"] > row["ground_truth_count"]:
        raise PrefixRouteR6Error("fragmented count exceeds ground truth count")
    _finite(row["endpoint_latency_bins_sum"], "endpoint_latency_bins_sum")
    stress = row["stress"]
    if not isinstance(stress, dict) or set(stress) != set(eligible_stress):
        raise PrefixRouteR6Error("stress metric cell set differs")
    for family, counts in stress.items():
        if not isinstance(counts, dict) or set(counts) != {
            "ground_truth_count",
            "matched_ground_truth_count",
        }:
            raise PrefixRouteR6Error(f"stress counts differ for {family}")
        total = _nonnegative_int(
            counts["ground_truth_count"],
            f"{family}.ground_truth_count",
        )
        matched = _nonnegative_int(
            counts["matched_ground_truth_count"],
            f"{family}.matched_ground_truth_count",
        )
        if matched > total:
            raise PrefixRouteR6Error(f"stress matched count exceeds total for {family}")
    return row


def _validate_dataset_metrics(rows, seeds):
    required = {"arm", "seed", "mOnlineAP", "class_mOnlineAP"}
    mapping = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise PrefixRouteR6Error("dataset metric row fields differ")
        arm = row["arm"]
        seed = row["seed"]
        if arm not in INFERENCE_ARMS or seed not in seeds:
            raise PrefixRouteR6Error("dataset metric row key is unregistered")
        values = {
            metric: _finite(row[metric], metric)
            for metric in ("mOnlineAP", "class_mOnlineAP")
        }
        if any(not 0.0 <= value <= 1.0 for value in values.values()):
            raise PrefixRouteR6Error(
                "dataset AP metrics must be in [0, 1]"
            )
        key = (arm, seed)
        if key in mapping:
            raise PrefixRouteR6Error("duplicate dataset metric row")
        mapping[key] = values
    expected = {(arm, seed) for arm in INFERENCE_ARMS for seed in seeds}
    if set(mapping) != expected:
        raise PrefixRouteR6Error("dataset metric Cartesian product is incomplete")
    return mapping


def _safe_rate(numerator, denominator, label):
    if denominator <= 0:
        raise PrefixRouteR6Error(f"{label} denominator is zero")
    return numerator / denominator


def _aggregate(rows, dataset_map, arm, sampled_seeds, eligible_stress):
    selected = [row for row in rows if row["arm"] == arm]
    ground_truth = sum(row["ground_truth_count"] for row in selected)
    matched = sum(row["matched_ground_truth_count"] for row in selected)
    false_emissions = sum(row["false_emission_count"] for row in selected)
    duplicates = sum(row["duplicate_emission_count"] for row in selected)
    fragmented = sum(row["fragmented_ground_truth_count"] for row in selected)
    latency_count = sum(row["endpoint_latency_observed_count"] for row in selected)
    metrics = {
        "mOnlineAP": (
            sum(
                dataset_map[(arm, seed)]["mOnlineAP"]
                for seed in sampled_seeds
            )
            / len(sampled_seeds)
        ),
        "class_mOnlineAP": (
            sum(
                dataset_map[(arm, seed)]["class_mOnlineAP"]
                for seed in sampled_seeds
            )
            / len(sampled_seeds)
        ),
        "event_recall": _safe_rate(matched, ground_truth, "event_recall"),
        "false_emission_per_video": _safe_rate(
            false_emissions,
            len(selected),
            "false_emission_per_video",
        ),
        "duplicate_per_gt": _safe_rate(
            duplicates,
            ground_truth,
            "duplicate_per_gt",
        ),
        "fragmentation_per_gt": _safe_rate(
            fragmented,
            ground_truth,
            "fragmentation_per_gt",
        ),
        "endpoint_latency_bins": _safe_rate(
            sum(row["endpoint_latency_bins_sum"] for row in selected),
            latency_count,
            "endpoint_latency_bins",
        ),
    }
    for family in eligible_stress:
        total = sum(
            row["stress"][family]["ground_truth_count"] for row in selected
        )
        family_matched = sum(
            row["stress"][family]["matched_ground_truth_count"]
            for row in selected
        )
        metrics[f"{family}_recall"] = _safe_rate(
            family_matched,
            total,
            f"{family}_recall",
        )
    return metrics


def _oriented_delta(left, right, metric):
    if metric in HIGHER_IS_BETTER:
        return left - right
    if metric in LOWER_IS_BETTER:
        return right - left
    raise PrefixRouteR6Error(f"metric direction is not registered: {metric}")


def _practical_margins(rows, eligible_stress):
    first_seed = min(row["seed"] for row in rows)
    one_arm = [
        row for row in rows if row["arm"] == "B4" and row["seed"] == first_seed
    ]
    gt = sum(row["ground_truth_count"] for row in one_arm)
    video_count = len({row["video_id"] for row in one_arm})
    if gt <= 0 or video_count <= 0:
        raise PrefixRouteR6Error("B4 denominators are empty")
    margins = {
        "mOnlineAP": 0.005,
        "class_mOnlineAP": 0.05,
        "event_recall": max(0.01, 2.0 / gt),
        "false_emission_per_video": max(0.01, 2.0 / video_count),
        "duplicate_per_gt": max(0.01, 2.0 / gt),
        "fragmentation_per_gt": max(0.01, 2.0 / gt),
        "endpoint_latency_bins": 1.0,
    }
    for family in eligible_stress:
        family_gt = sum(
            row["stress"][family]["ground_truth_count"] for row in one_arm
        )
        if family_gt <= 0:
            raise PrefixRouteR6Error(f"eligible stress family {family} is empty")
        margins[f"{family}_recall"] = max(0.02, 2.0 / family_gt)
    return margins


def _paired_crossed_bootstrap(
    per_video_cells,
    dataset_metrics,
    *,
    eligible_stress_families,
    resamples,
    seed,
    family_wise_alpha,
):
    """Use global paired seed and video resamples for every arm and contrast."""

    eligible_stress = tuple(eligible_stress_families)
    if any(family not in {
        "same_class_repetition",
        "same_class_overlap",
        "same_bin_end_start",
        "direct_complete",
    } for family in eligible_stress):
        raise PrefixRouteR6Error("eligible stress family is unregistered")
    if len(eligible_stress) != len(set(eligible_stress)):
        raise PrefixRouteR6Error("eligible stress families contain duplicates")
    cells = [_validate_cell(dict(row), eligible_stress) for row in per_video_cells]
    seeds = sorted({row["seed"] for row in cells})
    videos = sorted({row["video_id"] for row in cells})
    if not seeds or not videos:
        raise PrefixRouteR6Error("paired inference population is empty")
    cell_map = {}
    for row in cells:
        key = (row["arm"], row["seed"], row["video_id"])
        if key in cell_map:
            raise PrefixRouteR6Error("duplicate arm-seed-video cell")
        cell_map[key] = row
    expected = {
        (arm, run_seed, video_id)
        for arm in INFERENCE_ARMS
        for run_seed in seeds
        for video_id in videos
    }
    if set(cell_map) != expected:
        raise PrefixRouteR6Error("arm-seed-video Cartesian product is incomplete")
    for video_id in videos:
        reference = cell_map[(INFERENCE_ARMS[0], seeds[0], video_id)]
        reference_stress = {
            family: reference["stress"][family]["ground_truth_count"]
            for family in eligible_stress
        }
        for arm in INFERENCE_ARMS:
            for run_seed in seeds:
                row = cell_map[(arm, run_seed, video_id)]
                if row["ground_truth_count"] != reference["ground_truth_count"]:
                    raise PrefixRouteR6Error(
                        "ground-truth denominator differs across paired cells"
                    )
                if {
                    family: row["stress"][family]["ground_truth_count"]
                    for family in eligible_stress
                } != reference_stress:
                    raise PrefixRouteR6Error(
                        "stress denominator differs across paired cells"
                    )
    dataset_map = _validate_dataset_metrics(dataset_metrics, seeds)
    metrics = GLOBAL_METRICS + CONTROL_METRICS + tuple(
        f"{family}_recall" for family in eligible_stress
    )
    comparison_count = len(REQUIRED_CONTRASTS) * len(metrics)
    alpha_each = float(family_wise_alpha) / comparison_count
    lower_probability = alpha_each / 2.0
    upper_probability = 1.0 - alpha_each / 2.0
    margins = _practical_margins(cells, eligible_stress)

    observed = {}
    all_seed_video_rows = [
        cell_map[(arm, run_seed, video_id)]
        for arm in INFERENCE_ARMS
        for run_seed in seeds
        for video_id in videos
    ]
    observed_by_arm = {
        arm: _aggregate(
            [row for row in all_seed_video_rows if row["arm"] == arm],
            dataset_map,
            arm,
            seeds,
            eligible_stress,
        )
        for arm in INFERENCE_ARMS
    }
    for left, right in REQUIRED_CONTRASTS:
        observed[(left, right)] = {
            metric: _oriented_delta(
                observed_by_arm[left][metric],
                observed_by_arm[right][metric],
                metric,
            )
            for metric in metrics
        }

    samples = {
        (left, right, metric): []
        for left, right in REQUIRED_CONTRASTS
        for metric in metrics
    }
    rng = random.Random(int(seed))
    for _ in range(int(resamples)):
        sampled_seeds = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        sampled_videos = [videos[rng.randrange(len(videos))] for _ in videos]
        arm_metrics = {}
        for arm in INFERENCE_ARMS:
            sampled_rows = [
                cell_map[(arm, run_seed, video_id)]
                for run_seed in sampled_seeds
                for video_id in sampled_videos
            ]
            arm_metrics[arm] = _aggregate(
                sampled_rows,
                dataset_map,
                arm,
                sampled_seeds,
                eligible_stress,
            )
        for left, right in REQUIRED_CONTRASTS:
            for metric in metrics:
                samples[(left, right, metric)].append(
                    _oriented_delta(
                        arm_metrics[left][metric],
                        arm_metrics[right][metric],
                        metric,
                    )
                )

    intervals = {}
    for left, right in REQUIRED_CONTRASTS:
        contrast_id = f"{left}_vs_{right}"
        intervals[contrast_id] = {}
        for metric in metrics:
            values = samples[(left, right, metric)]
            intervals[contrast_id][metric] = {
                "orientation": "positive_means_left_arm_is_better",
                "estimate": observed[(left, right)][metric],
                "lower": _type7_percentile(values, lower_probability),
                "upper": _type7_percentile(values, upper_probability),
                "practical_margin": margins[metric],
            }
    return {
        "schema_version": "prefix-route-r6-inference-v2",
        "method": "paired_crossed_video_and_global_seed_percentile_bootstrap",
        "fixed_population_estimand": True,
        "mOnlineAP_resampling_unit": "paired_global_training_seed",
        "additive_metric_resampling_units": ["video", "paired_global_training_seed"],
        "resamples": int(resamples),
        "seed": int(seed),
        "family_wise_alpha": float(family_wise_alpha),
        "contrast_count": len(REQUIRED_CONTRASTS),
        "metric_count": len(metrics),
        "simultaneous_comparison_count": comparison_count,
        "per_comparison_two_sided_alpha": alpha_each,
        "lower_probability": lower_probability,
        "upper_probability": upper_probability,
        "eligible_stress_families": list(eligible_stress),
        "margins": margins,
        "arm_estimates": observed_by_arm,
        "intervals": intervals,
    }


def _sealed_paired_crossed_bootstrap(
    per_video_cells,
    dataset_metrics,
    *,
    eligible_stress_families,
):
    """Run the only registered production inference parameterization."""

    return _paired_crossed_bootstrap(
        per_video_cells,
        dataset_metrics,
        eligible_stress_families=eligible_stress_families,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        family_wise_alpha=FAMILY_WISE_ALPHA,
    )


def _all_noninferior(intervals, metric_names):
    return all(
        intervals[metric]["lower"] >= -intervals[metric]["practical_margin"]
        for metric in metric_names
    )


def _terminal_route_decision(inference):
    """Compute the frozen B4 survival decision from sealed inference."""

    intervals = inference["intervals"]
    global_metrics = tuple(GLOBAL_METRICS)
    equivalent_controls = []
    dominating_controls = []
    for arm in SIMPLE_CONTROL_ARMS:
        control = intervals[f"B4_vs_{arm}"]
        if all(
            control[metric]["lower"] >= -control[metric]["practical_margin"]
            and control[metric]["upper"] <= control[metric]["practical_margin"]
            for metric in global_metrics
        ):
            equivalent_controls.append(arm)
        elif (
            all(
                control[metric]["upper"]
                <= control[metric]["practical_margin"]
                for metric in global_metrics
            )
            and any(
                control[metric]["upper"]
                < -control[metric]["practical_margin"]
                for metric in global_metrics
            )
        ):
            dominating_controls.append(arm)
    if equivalent_controls or dominating_controls:
        return {
            "status": "KILL_BENCHMARK_NOT_IDENTIFIABLE",
            "equivalent_simple_controls": equivalent_controls,
            "dominating_simple_controls": dominating_controls,
            "temporal_control_established": False,
            "semantic_control_established": False,
            "b2_global_noninferior": False,
            "d1_established": False,
            "d2_established": False,
            "route_claim_allowed": False,
            "automatic_extra_seeds_allowed": False,
        }

    temporal = intervals["B4_vs_FEATURE_TIME_SHUFFLE"]
    temporal_metrics = ("mOnlineAP", "event_recall")
    temporal_established = any(
        temporal[metric]["lower"] > temporal[metric]["practical_margin"]
        for metric in temporal_metrics
    )
    temporal_ruled_out = all(
        temporal[metric]["upper"] <= temporal[metric]["practical_margin"]
        for metric in temporal_metrics
    )
    if temporal_ruled_out:
        return {
            "status": "KILL_TEMPORAL_IDENTIFIABILITY_CLAIM",
            "equivalent_simple_controls": [],
            "dominating_simple_controls": [],
            "temporal_control_established": False,
            "semantic_control_established": False,
            "b2_global_noninferior": False,
            "d1_established": False,
            "d2_established": False,
            "route_claim_allowed": False,
            "automatic_extra_seeds_allowed": False,
        }

    semantic = intervals["B4_vs_SEMANTIC_DERANGEMENT"][
        "class_mOnlineAP"
    ]
    semantic_established = semantic["lower"] > semantic["practical_margin"]
    if semantic["upper"] <= semantic["practical_margin"]:
        return {
            "status": "KILL_SEMANTIC_IDENTIFIABILITY_CLAIM",
            "equivalent_simple_controls": [],
            "dominating_simple_controls": [],
            "temporal_control_established": temporal_established,
            "semantic_control_established": False,
            "b2_global_noninferior": False,
            "d1_established": False,
            "d2_established": False,
            "route_claim_allowed": False,
            "automatic_extra_seeds_allowed": False,
        }

    b2 = intervals["B4_vs_B2"]
    if any(
        b2[metric]["upper"] < -b2[metric]["practical_margin"]
        for metric in global_metrics
    ):
        return {
            "status": "KILL_TEMPORAL_MOTR_INFERIOR",
            "equivalent_simple_controls": [],
            "dominating_simple_controls": [],
            "temporal_control_established": temporal_established,
            "semantic_control_established": semantic_established,
            "b2_global_noninferior": False,
            "d1_established": False,
            "d2_established": False,
            "route_claim_allowed": False,
            "automatic_extra_seeds_allowed": False,
        }
    b2_noninferior = _all_noninferior(b2, global_metrics)
    d1_intervals = intervals["B4_vs_A4_NO_SAME_BIN_REUSE"]
    d1_metric = "same_bin_end_start_recall"
    d1_available = d1_metric in d1_intervals
    d1_established = (
        d1_available
        and d1_intervals[d1_metric]["lower"]
        > d1_intervals[d1_metric]["practical_margin"]
        and _all_noninferior(d1_intervals, global_metrics)
    )
    d2_intervals = intervals["B4_vs_A1_A2_JOINT"]
    d2_candidates = (
        "same_class_repetition_recall",
        "same_class_overlap_recall",
    )
    d2_established = (
        any(
            metric in d2_intervals
            and d2_intervals[metric]["lower"]
            > d2_intervals[metric]["practical_margin"]
            for metric in d2_candidates
        )
        and _all_noninferior(d2_intervals, global_metrics)
    )
    controls_established = temporal_established and semantic_established
    if b2_noninferior and controls_established and (
        d1_established or d2_established
    ):
        status = "PASS_B4_ROUTE_SURVIVES"
        claim_allowed = True
    else:
        d1_ruled_out = (
            not d1_available
            or d1_intervals[d1_metric]["upper"]
            <= d1_intervals[d1_metric]["practical_margin"]
        )
        d2_ruled_out = all(
            metric not in d2_intervals
            or d2_intervals[metric]["upper"]
            <= d2_intervals[metric]["practical_margin"]
            for metric in d2_candidates
        )
        if d1_ruled_out and d2_ruled_out:
            status = "KILL_D1_D2_NOT_ESTABLISHED"
        else:
            status = "INDETERMINATE_NO_ROUTE_CLAIM"
        claim_allowed = False
    return {
        "status": status,
        "equivalent_simple_controls": [],
        "dominating_simple_controls": [],
        "temporal_control_established": temporal_established,
        "semantic_control_established": semantic_established,
        "b2_global_noninferior": b2_noninferior,
        "d1_established": d1_established,
        "d2_established": d2_established,
        "route_claim_allowed": claim_allowed,
        "automatic_extra_seeds_allowed": False,
    }


def _sha256_text(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PrefixRouteR6Error(f"{label} must be a lowercase SHA-256")
    return value


def _r0_ground_truth(r0_report, r0_detail, *, expected_video_count):
    if not isinstance(r0_report, dict) or not isinstance(r0_detail, dict):
        raise PrefixRouteR6Error("R0 report and detail must be objects")
    report_fields = {
        "schema_version",
        "collector_version",
        "status",
        "reporting_population_role",
        "annotation_exposure_status",
        "author_disclosure",
        "feature_stride_frames",
        "aggregate",
        "claim_eligibility",
        "bootstrap",
        "reviewer_detail_commitment_sha256",
    }
    detail_fields = {
        "schema_version",
        "collector_version",
        "definitions",
        "feature_stride_frames",
        "videos",
    }
    if set(r0_report) != report_fields or set(r0_detail) != detail_fields:
        raise PrefixRouteR6Error("R0 source record fields differ")
    from opentad.utils.prefix_route_r0_v2 import (
        R0_DETAIL_SCHEMA,
        R0_REPORT_SCHEMA,
    )

    if (
        r0_report["schema_version"] != R0_REPORT_SCHEMA
        or r0_detail["schema_version"] != R0_DETAIL_SCHEMA
        or r0_report["collector_version"] != r0_detail["collector_version"]
        or r0_report["status"] != "PASS_R0_COMPLETE"
        or r0_report["reporting_population_role"] != "canonical_reporting_213"
        or r0_report["annotation_exposure_status"]
        != "DESIGN_EXPOSED_ROUTE_SELECTION_AND_BENCHMARK"
        or r0_report["author_disclosure"] != "aggregate_only_no_video_ids"
        or r0_report["feature_stride_frames"] != R6_FEATURE_STRIDE_FRAMES
        or r0_detail["feature_stride_frames"] != R6_FEATURE_STRIDE_FRAMES
    ):
        raise PrefixRouteR6Error("R0 source identity differs")
    detail_sha256 = hashlib.sha256(_canonical_json_bytes(r0_detail)).hexdigest()
    if r0_report.get("reviewer_detail_commitment_sha256") != detail_sha256:
        raise PrefixRouteR6Error("R0 detail differs from its report commitment")
    videos = r0_detail.get("videos")
    if (
        not isinstance(videos, list)
        or len(videos) != expected_video_count
        or r0_report["aggregate"].get("video_count") != expected_video_count
    ):
        raise PrefixRouteR6Error("R0 detail video population differs")
    by_video = {}
    family_totals = {family: 0 for family in (
        "same_class_repetition",
        "same_class_overlap",
        "same_bin_end_start",
        "direct_complete",
    )}
    for video in videos:
        required = {
            "video_id",
            "subset",
            "duration_sec",
            "frame_count",
            "legal_instances",
            "ambiguous_intervals",
            "family_instance_indexes",
            "family_classes",
            "overlap",
            "same_bin",
            "max_total_concurrency",
            "max_same_class_concurrency",
        }
        if not isinstance(video, dict) or set(video) != required:
            raise PrefixRouteR6Error("R0 detail video fields differ")
        video_id = video["video_id"]
        if (
            not isinstance(video_id, str)
            or not video_id
            or video_id in by_video
            or video["subset"] != "validation"
        ):
            raise PrefixRouteR6Error("R0 detail video identity differs")
        frame_count = _nonnegative_int(video["frame_count"], "frame_count")
        duration_sec = _finite(video["duration_sec"], "duration_sec")
        if frame_count <= 0 or duration_sec <= 0:
            raise PrefixRouteR6Error("R0 video geometry must be positive")
        legal = video["legal_instances"]
        if not isinstance(legal, list):
            raise PrefixRouteR6Error("R0 legal instances must be an array")
        gt = []
        index_to_id = {}
        for instance in legal:
            if not isinstance(instance, dict):
                raise PrefixRouteR6Error("R0 legal instance must be an object")
            index = _nonnegative_int(
                instance.get("instance_index"),
                "instance_index",
            )
            if index in index_to_id:
                raise PrefixRouteR6Error("R0 instance indexes are duplicated")
            start = _finite(instance.get("start_frame"), "start_frame")
            end = _finite(
                instance.get("end_observation_count"),
                "end_observation_count",
            )
            label = instance.get("label")
            if (
                start < 0
                or not start < end <= frame_count
                or not isinstance(label, str)
                or not label
            ):
                raise PrefixRouteR6Error("R0 legal instance geometry differs")
            gt_id = f"{video_id}:instance:{index}"
            index_to_id[index] = gt_id
            gt.append(
                {
                    "gt_id": gt_id,
                    "stream_key": video_id,
                    "label": label,
                    "start_frame": start,
                    "end_frame": end,
                    "start_sec": _finite(instance["start_sec"], "start_sec"),
                    "end_sec": _finite(instance["end_sec"], "end_sec"),
                }
            )
        families = video["family_instance_indexes"]
        if not isinstance(families, dict) or set(families) != set(family_totals):
            raise PrefixRouteR6Error("R0 stress family set differs")
        stress_ids = {}
        for family, indexes in families.items():
            if (
                not isinstance(indexes, list)
                or len(indexes) != len(set(indexes))
                or any(index not in index_to_id for index in indexes)
            ):
                raise PrefixRouteR6Error("R0 stress membership differs")
            stress_ids[family] = [index_to_id[index] for index in indexes]
            family_totals[family] += len(indexes)
        by_video[video_id] = {
            "frame_count": frame_count,
            "duration_sec": duration_sec,
            "fps": frame_count / duration_sec,
            "ground_truth": gt,
            "stress_ground_truth_ids": stress_ids,
        }
    if list(by_video) != sorted(by_video):
        raise PrefixRouteR6Error("R0 detail videos must be sorted")
    eligible = tuple(
        family
        for family in (
            "same_class_repetition",
            "same_class_overlap",
            "same_bin_end_start",
            "direct_complete",
        )
        if family_totals[family] > 0
    )
    return by_video, eligible, detail_sha256


def _r0_evidence(protocol_record, r0_envelope, r0_detail):
    from opentad.utils.prefix_route_protocol_v2 import R0_ENVELOPE_SCHEMA
    from opentad.utils.prefix_route_r0_v2 import (
        BOOTSTRAP_RESAMPLES as R0_BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED as R0_BOOTSTRAP_SEED,
    )

    required = {
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "review_attestation_sha256",
        "population_derived_sha256",
        "source_sha256",
        "report",
        "status",
        "derived_sha256",
    }
    if not isinstance(r0_envelope, dict) or set(r0_envelope) != required:
        raise PrefixRouteR6Error("R0 evidence envelope fields differ")
    source_sha256 = r0_envelope["source_sha256"]
    if not isinstance(source_sha256, dict) or set(source_sha256) != {
        "annotation",
        "class_map",
        "exposure_ledger",
    }:
        raise PrefixRouteR6Error("R0 source commitment fields differ")
    for field, value in source_sha256.items():
        _sha256_text(value, f"R0 {field} source")
    for field in (
        "review_attestation_sha256",
        "population_derived_sha256",
        "derived_sha256",
    ):
        _sha256_text(r0_envelope[field], f"R0 {field}")
    unsigned = dict(r0_envelope)
    supplied_derived = unsigned.pop("derived_sha256")
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != supplied_derived:
        raise PrefixRouteR6Error("R0 evidence envelope commitment differs")
    if (
        r0_envelope["schema_version"] != R0_ENVELOPE_SCHEMA
        or r0_envelope["protocol_id"]
        != protocol_record["protocol"]["protocol_id"]
        or r0_envelope["protocol_sha256"] != protocol_record["sha256"]
        or r0_envelope["status"] != "PASS_R0_COMPLETE"
    ):
        raise PrefixRouteR6Error("R0 evidence envelope identity differs")
    registration = protocol_record["protocol"]["population"][
        "source_registration"
    ]
    if registration["state"] != "REGISTERED_IN_FIXED_REVIEWED_PROTOCOL":
        raise PrefixRouteR6Error(
            "R6 is blocked until reporting source identity is registered"
        )
    if (
        source_sha256["annotation"]
        != registration["authoritative_annotation_sha256"]
    ):
        raise PrefixRouteR6Error(
            "R0 annotation source differs from registered protocol"
        )
    report = r0_envelope["report"]
    if not isinstance(report, dict):
        raise PrefixRouteR6Error("R0 envelope report must be an object")
    bootstrap = report.get("bootstrap")
    if (
        not isinstance(bootstrap, dict)
        or bootstrap.get("method")
        != "nonparametric_video_cluster_percentile_type7"
        or bootstrap.get("resamples") != R0_BOOTSTRAP_RESAMPLES
        or bootstrap.get("seed") != R0_BOOTSTRAP_SEED
        or bootstrap.get("confidence_level") != 0.95
        or bootstrap.get("same_resample_indices_for_all_targets") is not True
    ):
        raise PrefixRouteR6Error("R0 bootstrap identity differs")
    expected_video_count = protocol_record["protocol"]["population"][
        "reporting"
    ]["canonical_count"]
    video_sources, eligible_stress, detail_sha256 = _r0_ground_truth(
        report,
        r0_detail,
        expected_video_count=expected_video_count,
    )
    return report, video_sources, eligible_stress, detail_sha256


def _canonical_video_emissions(rows, video_id, frame_count):
    if not isinstance(rows, list):
        raise PrefixRouteR6Error("per-video emissions must be an array")
    canonical = [canonical_emission(dict(row)) for row in rows]
    if [row["sequence_id"] for row in canonical] != list(range(len(canonical))):
        raise PrefixRouteR6Error(
            "per-video sequence_id must be contiguous append order"
        )
    if len({row["emission_id"] for row in canonical}) != len(canonical):
        raise PrefixRouteR6Error("per-video emission IDs are duplicated")
    previous_emit = -math.inf
    for row in canonical:
        if row["stream_key"] != video_id:
            raise PrefixRouteR6Error("emission stream differs from R0 video")
        if row["end"] > frame_count or row["emit_frame"] > frame_count:
            raise PrefixRouteR6Error("emission escapes the observed video prefix")
        if row["emit_frame"] < previous_emit:
            raise PrefixRouteR6Error("emissions are not append-ordered")
        previous_emit = row["emit_frame"]
    return canonical


def _online_ap_from_raw(video_sources, emissions_by_video):
    ground_truth = {"database": {}}
    predictions = {"results": {}}
    for video_id, source in video_sources.items():
        ground_truth["database"][video_id] = {
            "subset": "validation",
            "duration": source["duration_sec"],
            "frame": source["frame_count"],
            "annotations": [
                {
                    "segment": [row["start_sec"], row["end_sec"]],
                    "label": row["label"],
                    "instance_id": row["gt_id"],
                }
                for row in source["ground_truth"]
            ],
        }
        fps = source["fps"]
        predictions["results"][video_id] = [
            {
                "segment": [row["start"] / fps, row["end"] / fps],
                "label": row["class"],
                "score": row["score"],
                "source_frame": row["source_frame"],
                "emit_frame": row["emit_frame"],
                "fps": fps,
            }
            for row in emissions_by_video[video_id]
        ]
    evaluator = OnlineAPBudgeted(
        ground_truth,
        predictions,
        subset="validation",
        tiou_thresholds=(R6_TIOU_THRESHOLD,),
        latency_budgets_sec=(R6_LATENCY_BUDGET_SEC,),
        require_ledger=False,
        include_identity_diagnostics=False,
    )
    metrics = evaluator.evaluate()
    class_scores = metrics["class_AP@2s@tIoU0.5"]
    return {
        "mOnlineAP": float(metrics["average_mOnlineAP"]),
        "class_mOnlineAP": (
            sum(class_scores.values()) / len(class_scores)
            if class_scores
            else 0.0
        ),
    }


def evaluate_r6_raw_evidence(
    raw_evidence,
    *,
    protocol_path,
    r0_envelope,
    r0_detail,
):
    """Derive every R6 metric, interval, control, and terminal state from raw rows."""

    from opentad.utils.prefix_route_protocol_v2 import load_protocol

    protocol_record = load_protocol(protocol_path)
    required = {
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "r0_envelope_sha256",
        "r0_detail_sha256",
        "reporting_subset",
        "seeds",
        "runs",
    }
    if not isinstance(raw_evidence, dict) or set(raw_evidence) != required:
        raise PrefixRouteR6Error("R6 raw evidence fields differ")
    if raw_evidence["schema_version"] != R6_RAW_SCHEMA:
        raise PrefixRouteR6Error("R6 raw evidence schema differs")
    if (
        raw_evidence["protocol_id"]
        != protocol_record["protocol"]["protocol_id"]
        or raw_evidence["protocol_sha256"] != protocol_record["sha256"]
        or raw_evidence["reporting_subset"]
        != protocol_record["protocol"]["r0"]["reporting_subset"]
        or tuple(raw_evidence["seeds"])
        != tuple(protocol_record["protocol"]["r6"]["run_seeds"])
    ):
        raise PrefixRouteR6Error("R6 frozen identity differs")
    _sha256_text(raw_evidence["protocol_sha256"], "protocol_sha256")
    envelope_sha256 = hashlib.sha256(
        _canonical_json_bytes(r0_envelope)
    ).hexdigest()
    if raw_evidence["r0_envelope_sha256"] != envelope_sha256:
        raise PrefixRouteR6Error("R6 R0 envelope binding differs")
    _, video_sources, eligible_stress, detail_sha256 = _r0_evidence(
        protocol_record,
        r0_envelope,
        r0_detail,
    )
    if raw_evidence["r0_detail_sha256"] != detail_sha256:
        raise PrefixRouteR6Error("R6 R0 detail binding differs")
    runs = raw_evidence["runs"]
    if not isinstance(runs, list):
        raise PrefixRouteR6Error("R6 runs must be an array")
    run_map = {}
    cells = []
    dataset_metrics = []
    expected_videos = list(video_sources)
    for run in runs:
        if not isinstance(run, dict) or set(run) != {"arm", "seed", "videos"}:
            raise PrefixRouteR6Error("R6 run fields differ")
        arm = run["arm"]
        seed = run["seed"]
        if arm not in INFERENCE_ARMS or seed not in R6_SEEDS:
            raise PrefixRouteR6Error("R6 run identity is unregistered")
        key = (arm, seed)
        if key in run_map:
            raise PrefixRouteR6Error("R6 run is duplicated")
        videos = run["videos"]
        if not isinstance(videos, list):
            raise PrefixRouteR6Error("R6 run videos must be an array")
        emissions_by_video = {}
        for video in videos:
            if (
                not isinstance(video, dict)
                or set(video) != {"video_id", "emissions"}
            ):
                raise PrefixRouteR6Error("R6 run video fields differ")
            video_id = video["video_id"]
            if video_id not in video_sources or video_id in emissions_by_video:
                raise PrefixRouteR6Error("R6 run video identity differs")
            source = video_sources[video_id]
            emissions = _canonical_video_emissions(
                video["emissions"],
                video_id,
                source["frame_count"],
            )
            emissions_by_video[video_id] = emissions
            cells.append(
                derive_per_video_cell(
                    arm=arm,
                    seed=seed,
                    video_id=video_id,
                    ground_truth=source["ground_truth"],
                    emissions=emissions,
                    stress_ground_truth_ids={
                        family: source["stress_ground_truth_ids"][family]
                        for family in eligible_stress
                    },
                    tiou_threshold=R6_TIOU_THRESHOLD,
                    fps=source["fps"],
                    latency_budget_sec=R6_LATENCY_BUDGET_SEC,
                    feature_stride_frames=R6_FEATURE_STRIDE_FRAMES,
                )
            )
        if list(emissions_by_video) != expected_videos:
            raise PrefixRouteR6Error("R6 run video population differs from R0")
        if len({
            row["emission_id"]
            for emissions in emissions_by_video.values()
            for row in emissions
        }) != sum(len(rows) for rows in emissions_by_video.values()):
            raise PrefixRouteR6Error("R6 emission IDs are duplicated across videos")
        dataset_metrics.append(
            {
                "arm": arm,
                "seed": seed,
                **_online_ap_from_raw(video_sources, emissions_by_video),
            }
        )
        run_map[key] = True
    expected_runs = {
        (arm, seed) for arm in INFERENCE_ARMS for seed in R6_SEEDS
    }
    if set(run_map) != expected_runs:
        raise PrefixRouteR6Error("R6 arm-seed Cartesian product is incomplete")
    inference = _sealed_paired_crossed_bootstrap(
        cells,
        dataset_metrics,
        eligible_stress_families=eligible_stress,
    )
    decision = _terminal_route_decision(inference)
    result = {
        "schema_version": R6_RESULT_SCHEMA,
        "raw_evidence_sha256": hashlib.sha256(
            _canonical_json_bytes(raw_evidence)
        ).hexdigest(),
        "r0_envelope_sha256": envelope_sha256,
        "r0_detail_sha256": detail_sha256,
        "source_derived_ground_truth": True,
        "caller_supplied_metrics_or_intervals": False,
        "online_ap_evaluator": (
            "opentad.evaluations.online_budgeted_map.OnlineAPBudgeted"
        ),
        "eligible_stress_families": list(eligible_stress),
        "inference": inference,
        "decision": decision,
    }
    result["derived_sha256"] = hashlib.sha256(
        _canonical_json_bytes(result)
    ).hexdigest()
    return result


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONTROL_ARMS",
    "CONTROL_METRICS",
    "FAMILY_WISE_ALPHA",
    "GLOBAL_METRICS",
    "INFERENCE_ARMS",
    "R6_RAW_SCHEMA",
    "R6_RESULT_SCHEMA",
    "R6_SEEDS",
    "REQUIRED_ARMS",
    "REQUIRED_CONTRASTS",
    "STRESS_METRICS",
    "PrefixRouteR6Error",
    "canonical_emission",
    "derive_per_video_cell",
    "evaluate_r6_raw_evidence",
]
