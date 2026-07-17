"""Exact paired inference and terminal decisions for Prefix-Route Protocol V2."""

from __future__ import annotations

import math
import random

from .full_petal_metrics import compute_full_petal_metrics


GLOBAL_METRICS = (
    "mOnlineAP",
    "event_recall",
    "false_emission_per_video",
    "duplicate_per_gt",
    "fragmentation_per_gt",
    "endpoint_latency_bins",
)
STRESS_METRICS = (
    "same_class_repetition_recall",
    "same_class_overlap_recall",
    "same_bin_end_start_recall",
    "direct_complete_recall",
)
HIGHER_IS_BETTER = {
    "mOnlineAP",
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
REQUIRED_CONTRASTS = (
    ("B4", "B2"),
    ("B4", "B3"),
    ("B4", "A1_NO_UOT"),
    ("B4", "A2_BINARY_MASS"),
    ("B4", "A1_A2_JOINT"),
    ("B4", "A3_NO_CONSISTENCY"),
    ("B4", "A4_NO_SAME_BIN_REUSE"),
    ("B4", "A5_NO_NEURAL_LATCH"),
)
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 2026071707
FAMILY_WISE_ALPHA = 0.05


class PrefixRouteR6Error(ValueError):
    pass


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
    if not start < end:
        raise PrefixRouteR6Error("emission interval must be positive")
    if end > emitted:
        raise PrefixRouteR6Error("emission reveals a future endpoint")
    if source > emitted:
        raise PrefixRouteR6Error("source_frame exceeds emit_frame")
    if not 0.0 <= score <= 1.0:
        raise PrefixRouteR6Error("score must be in [0, 1]")
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

    if arm not in REQUIRED_ARMS:
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
    if row["arm"] not in REQUIRED_ARMS:
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
    required = {"arm", "seed", "mOnlineAP"}
    mapping = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise PrefixRouteR6Error("dataset metric row fields differ")
        arm = row["arm"]
        seed = row["seed"]
        if arm not in REQUIRED_ARMS or seed not in seeds:
            raise PrefixRouteR6Error("dataset metric row key is unregistered")
        value = _finite(row["mOnlineAP"], "mOnlineAP")
        if not 0.0 <= value <= 1.0:
            raise PrefixRouteR6Error("mOnlineAP must be in [0, 1]")
        key = (arm, seed)
        if key in mapping:
            raise PrefixRouteR6Error("duplicate dataset metric row")
        mapping[key] = value
    expected = {(arm, seed) for arm in REQUIRED_ARMS for seed in seeds}
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
            sum(dataset_map[(arm, seed)] for seed in sampled_seeds)
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


def paired_crossed_bootstrap(
    per_video_cells,
    dataset_metrics,
    *,
    eligible_stress_families,
    resamples=BOOTSTRAP_RESAMPLES,
    seed=BOOTSTRAP_SEED,
    family_wise_alpha=FAMILY_WISE_ALPHA,
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
        for arm in REQUIRED_ARMS
        for run_seed in seeds
        for video_id in videos
    }
    if set(cell_map) != expected:
        raise PrefixRouteR6Error("arm-seed-video Cartesian product is incomplete")
    for video_id in videos:
        reference = cell_map[(REQUIRED_ARMS[0], seeds[0], video_id)]
        reference_stress = {
            family: reference["stress"][family]["ground_truth_count"]
            for family in eligible_stress
        }
        for arm in REQUIRED_ARMS:
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
    metrics = GLOBAL_METRICS + tuple(
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
        for arm in REQUIRED_ARMS
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
        for arm in REQUIRED_ARMS
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
        for arm in REQUIRED_ARMS:
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


def _all_noninferior(intervals, metric_names):
    return all(
        intervals[metric]["lower"] >= -intervals[metric]["practical_margin"]
        for metric in metric_names
    )


def terminal_route_decision(inference):
    """Compute, rather than assert, the frozen B4 survival decision."""

    intervals = inference["intervals"]
    global_metrics = tuple(GLOBAL_METRICS)
    b2 = intervals["B4_vs_B2"]
    if any(
        b2[metric]["upper"] < -b2[metric]["practical_margin"]
        for metric in ("mOnlineAP", "event_recall")
    ):
        return {
            "status": "KILL_TEMPORAL_MOTR_INFERIOR",
            "b2_global_noninferior": False,
            "d1_established": False,
            "d2_established": False,
            "route_claim_allowed": False,
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
    if b2_noninferior and (d1_established or d2_established):
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
        "b2_global_noninferior": b2_noninferior,
        "d1_established": d1_established,
        "d2_established": d2_established,
        "route_claim_allowed": claim_allowed,
        "automatic_extra_seeds_allowed": False,
    }


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "FAMILY_WISE_ALPHA",
    "GLOBAL_METRICS",
    "REQUIRED_ARMS",
    "REQUIRED_CONTRASTS",
    "STRESS_METRICS",
    "PrefixRouteR6Error",
    "canonical_emission",
    "derive_per_video_cell",
    "paired_crossed_bootstrap",
    "terminal_route_decision",
]
