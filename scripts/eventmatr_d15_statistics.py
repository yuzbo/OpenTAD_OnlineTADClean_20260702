"""Pure lifecycle-census and paired-effect contracts for EventMATR D1.5 v2."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from typing import Iterable, Mapping


LIFECYCLE_CENSUS_PROTOCOL = "eventmatr_d15_lifecycle_census_v1"
PAIRED_ANALYSIS_PROTOCOL = "eventmatr_d15_paired_event_analysis_v1"

EFFECT_SIZE_THRESHOLD = 0.05
FAMILYWISE_ALPHA = 0.05
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 52_015
PERMUTATION_REPLICATES = 10_000
PERMUTATION_SEED = 52_016

EXPECTED_LIFECYCLE_COUNTS = {
    "video_count": 200,
    "annotation_count": 3007,
    "visible_birth_target_count": 3003,
    "observable_end_target_count": 3001,
    "fully_observed_event_count": 3001,
    "right_censored_event_count": 2,
    "left_truncated_event_count": 0,
    "completely_unobservable_event_count": 4,
    "birth_supervision_row_count": 3003,
    "alive_supervision_row_count": 56551,
    "end_supervision_row_count": 3001,
}

EXPECTED_RIGHT_CENSORED_IDENTITIES = (
    ("video_validation_0000318", 22, "HammerThrow"),
    ("video_validation_0000985", 9, "VolleyballSpiking"),
)
EXPECTED_UNOBSERVABLE_IDENTITIES = (
    ("video_validation_0000364", 4, "HighJump"),
    ("video_validation_0000364", 5, "HighJump"),
    ("video_validation_0000856", 3, "SoccerPenalty"),
    ("video_validation_0000856", 4, "SoccerPenalty"),
)

FORMAL_COMPARISONS = (
    {
        "name": "identity_refresh_under_predicted_admission",
        "treatment_channel": "PR",
        "treatment_route": "formal",
        "control_channel": "PF",
        "control_route": "formal",
    },
    {
        "name": "identity_refresh_under_oracle_visible_admission",
        "treatment_channel": "OR",
        "treatment_route": "formal",
        "control_channel": "OF",
        "control_route": "formal",
    },
    {
        "name": "oracle_visible_admission_under_free_identity",
        "treatment_channel": "OF",
        "treatment_route": "formal",
        "control_channel": "PF",
        "control_route": "formal",
    },
)

SHADOW_COMPARISONS = tuple(
    {
        "name": f"no_cancel_shadow_under_{channel}",
        "treatment_channel": channel,
        "treatment_route": "shadow",
        "control_channel": channel,
        "control_route": "formal",
    }
    for channel in ("PF", "PR", "OF", "OR")
)


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_float(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _nonnegative_integer(value, label: str) -> int:
    result = _finite_float(value, label)
    if result < 0 or not result.is_integer():
        raise ValueError(f"{label} is not a non-negative integer")
    return int(result)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability is outside [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution_summary(events: list[dict]) -> dict:
    alive_rows = [int(event["alive_supervision_rows"]) for event in events]
    durations = [float(event["duration_frames"]) for event in events]
    observable = [
        event for event in events if event["observation_status"] == "fully_observed"
    ]
    alive_bins = Counter()
    for value in alive_rows:
        if value == 0:
            label = "0"
        elif value == 1:
            label = "1"
        elif value == 2:
            label = "2"
        elif value <= 7:
            label = "3_to_7"
        elif value <= 15:
            label = "8_to_15"
        elif value <= 31:
            label = "16_to_31"
        elif value <= 63:
            label = "32_to_63"
        else:
            label = "64_plus"
        alive_bins[label] += 1

    duration_bins = Counter()
    for value in durations:
        if value <= 8.0:
            label = "at_most_8"
        elif value <= 16.0:
            label = "over_8_to_16"
        elif value <= 32.0:
            label = "over_16_to_32"
        elif value <= 64.0:
            label = "over_32_to_64"
        else:
            label = "over_64"
        duration_bins[label] += 1

    probabilities = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    return {
        "alive_supervision_row_histogram": dict(sorted(alive_bins.items())),
        "duration_frame_histogram": dict(sorted(duration_bins.items())),
        "alive_supervision_row_quantiles": {
            f"p{int(probability * 100):02d}": _quantile(alive_rows, probability)
            for probability in probabilities
        },
        "duration_frame_quantiles": {
            f"p{int(probability * 100):02d}": _quantile(durations, probability)
            for probability in probabilities
        },
        "mean_alive_supervision_rows": sum(alive_rows) / len(alive_rows),
        "mean_duration_frames": sum(durations) / len(durations),
        "fully_observed_direct_birth_to_end_count": sum(
            int(event["alive_supervision_rows"] == 0) for event in observable
        ),
        "fully_observed_at_most_one_alive_update_count": sum(
            int(event["alive_supervision_rows"] <= 1) for event in observable
        ),
        "fully_observed_at_most_two_alive_updates_count": sum(
            int(event["alive_supervision_rows"] <= 2) for event in observable
        ),
    }


def build_lifecycle_census(
    *,
    video_dict: Mapping[str, dict],
    video_len: Mapping[str, int],
    label_names: Iterable[str],
    birth_mode: str,
    anti_len: int,
) -> dict:
    """Reconstruct EventMATR lifecycle visibility from official metadata only."""

    if birth_mode not in {"instant_transition", "matr_delayed"}:
        raise ValueError(f"unsupported birth mode: {birth_mode!r}")
    label_names = tuple(label_names)
    label_to_id = {name: index for index, name in enumerate(label_names)}
    if len(label_to_id) != len(label_names):
        raise ValueError("class labels are not unique")
    if set(video_dict) != set(video_len):
        raise ValueError("annotation videos and feature-length videos differ")

    events = []
    for video_name in sorted(video_dict):
        info = video_dict[video_name]
        duration_seconds = _finite_float(
            info.get("duration"), f"{video_name}.duration"
        )
        feature_length = _nonnegative_integer(
            video_len[video_name], f"{video_name}.feature_length"
        )
        if duration_seconds <= 0.0 or feature_length <= 0:
            raise ValueError(f"{video_name} has a non-positive observation window")
        annotations = info.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError(f"{video_name}.annotations is not a list")
        frame_scale = feature_length / duration_seconds
        last_observed_frame = feature_length - 1
        for event_id, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                raise ValueError(f"{video_name}/{event_id} annotation is malformed")
            class_label = annotation.get("label")
            if class_label not in label_to_id:
                raise ValueError(f"{video_name}/{event_id} class is not registered")
            segment = annotation.get("segment")
            if not isinstance(segment, list) or len(segment) != 2:
                raise ValueError(f"{video_name}/{event_id} segment is malformed")
            start_frame = _finite_float(
                segment[0], f"{video_name}/{event_id}.start"
            ) * frame_scale
            end_frame = _finite_float(
                segment[1], f"{video_name}/{event_id}.end"
            ) * frame_scale
            if end_frame < start_frame:
                raise ValueError(f"{video_name}/{event_id} has reversed endpoints")
            if birth_mode == "instant_transition":
                admission_frame = float(math.ceil(start_frame))
            else:
                admission_frame = max(
                    float(math.ceil(start_frame)),
                    float(math.floor(end_frame - anti_len) + 1),
                )
            end_crossing_frame = max(
                float(math.ceil(end_frame)), admission_frame + 1.0
            )
            birth_observed = -1.0 < admission_frame <= last_observed_frame
            end_observed = -1.0 < end_crossing_frame <= last_observed_frame
            action_intersects_window = not (
                end_frame < 0.0 or start_frame > last_observed_frame
            )
            if not action_intersects_window:
                observation_status = "completely_unobservable"
            elif start_frame < 0.0:
                observation_status = "left_truncated"
            elif birth_observed and end_observed:
                observation_status = "fully_observed"
            elif birth_observed and end_crossing_frame > last_observed_frame:
                observation_status = "right_censored"
            elif not birth_observed:
                observation_status = "completely_unobservable"
            else:
                raise ValueError(
                    f"{video_name}/{event_id} has an unclassified observation status"
                )

            alive_first = max(0, int(admission_frame) + 1)
            alive_last = min(last_observed_frame, int(end_crossing_frame) - 1)
            alive_rows = max(0, alive_last - alive_first + 1)
            events.append(
                {
                    "video_name": video_name,
                    "event_id": event_id,
                    "class_id": label_to_id[class_label],
                    "class_label": class_label,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_frames": end_frame - start_frame,
                    "admission_frame": admission_frame,
                    "end_crossing_frame": end_crossing_frame,
                    "last_observed_frame": last_observed_frame,
                    "birth_observed": birth_observed,
                    "end_observed": end_observed,
                    "birth_supervision_rows": int(birth_observed),
                    "alive_supervision_rows": alive_rows,
                    "end_supervision_rows": int(end_observed),
                    "observation_status": observation_status,
                }
            )

    counts = Counter()
    counts["video_count"] = len(video_dict)
    counts["annotation_count"] = len(events)
    counts["visible_birth_target_count"] = sum(
        event["birth_supervision_rows"] for event in events
    )
    counts["observable_end_target_count"] = sum(
        event["end_supervision_rows"] for event in events
    )
    for status in (
        "fully_observed",
        "right_censored",
        "left_truncated",
        "completely_unobservable",
    ):
        counts[f"{status}_event_count"] = sum(
            event["observation_status"] == status for event in events
        )
    for state in ("birth", "alive", "end"):
        counts[f"{state}_supervision_row_count"] = sum(
            event[f"{state}_supervision_rows"] for event in events
        )

    census = {
        "protocol": LIFECYCLE_CENSUS_PROTOCOL,
        "birth_mode": birth_mode,
        "anti_len": int(anti_len),
        "counts": dict(sorted(counts.items())),
        "events": events,
        "event_list_sha256": canonical_sha256(events),
        "video_list_sha256": canonical_sha256(sorted(video_dict)),
        "supervision_distribution": _distribution_summary(events),
    }
    validate_lifecycle_census(census, require_official_counts=False)
    return census


def validate_lifecycle_census(
    census: dict, *, require_official_counts: bool = True
) -> dict:
    if not isinstance(census, dict):
        raise ValueError("D1.5 lifecycle census is absent")
    if census.get("protocol") != LIFECYCLE_CENSUS_PROTOCOL:
        raise ValueError("D1.5 lifecycle census protocol drifted")
    if census.get("birth_mode") not in {"instant_transition", "matr_delayed"}:
        raise ValueError("D1.5 lifecycle census birth mode drifted")
    _nonnegative_integer(census.get("anti_len"), "lifecycle_census.anti_len")
    events = census.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("D1.5 lifecycle census events are absent")
    if census.get("event_list_sha256") != canonical_sha256(events):
        raise ValueError("D1.5 lifecycle census event-list hash drifted")

    required_fields = {
        "video_name",
        "event_id",
        "class_id",
        "class_label",
        "start_frame",
        "end_frame",
        "duration_frames",
        "admission_frame",
        "end_crossing_frame",
        "last_observed_frame",
        "birth_observed",
        "end_observed",
        "birth_supervision_rows",
        "alive_supervision_rows",
        "end_supervision_rows",
        "observation_status",
    }
    normalized_keys = set()
    videos = set()
    for index, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != required_fields:
            raise ValueError(f"lifecycle event {index} schema drifted")
        video_name = event["video_name"]
        class_label = event["class_label"]
        if not isinstance(video_name, str) or not video_name:
            raise ValueError(f"lifecycle event {index} video is invalid")
        if not isinstance(class_label, str) or not class_label:
            raise ValueError(f"lifecycle event {index} class is invalid")
        event_id = _nonnegative_integer(event["event_id"], f"event[{index}].id")
        _nonnegative_integer(event["class_id"], f"event[{index}].class_id")
        for field in (
            "start_frame",
            "end_frame",
            "duration_frames",
            "admission_frame",
            "end_crossing_frame",
            "last_observed_frame",
        ):
            _finite_float(event[field], f"event[{index}].{field}")
        for field in ("birth_observed", "end_observed"):
            if not isinstance(event[field], bool):
                raise ValueError(f"event[{index}].{field} is not Boolean")
        for field in (
            "birth_supervision_rows",
            "alive_supervision_rows",
            "end_supervision_rows",
        ):
            _nonnegative_integer(event[field], f"event[{index}].{field}")
        if event["observation_status"] not in {
            "fully_observed",
            "right_censored",
            "left_truncated",
            "completely_unobservable",
        }:
            raise ValueError(f"event[{index}] observation status drifted")
        key = (video_name, event_id)
        if key in normalized_keys:
            raise ValueError("D1.5 lifecycle census duplicated an event identity")
        normalized_keys.add(key)
        videos.add(video_name)

        start_frame = float(event["start_frame"])
        end_frame = float(event["end_frame"])
        admission_frame = float(event["admission_frame"])
        end_crossing_frame = float(event["end_crossing_frame"])
        last_observed_frame = float(event["last_observed_frame"])
        if end_frame < start_frame or last_observed_frame < 0.0:
            raise ValueError(f"lifecycle event {index} endpoints are invalid")
        expected_admission = (
            float(math.ceil(start_frame))
            if census["birth_mode"] == "instant_transition"
            else max(
                float(math.ceil(start_frame)),
                float(math.floor(end_frame - int(census["anti_len"])) + 1),
            )
        )
        expected_end_crossing = max(
            float(math.ceil(end_frame)), expected_admission + 1.0
        )
        if not math.isclose(
            admission_frame, expected_admission, rel_tol=0.0, abs_tol=1e-9
        ) or not math.isclose(
            end_crossing_frame,
            expected_end_crossing,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"lifecycle event {index} crossing frames drifted")
        if not math.isclose(
            float(event["duration_frames"]),
            end_frame - start_frame,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(f"lifecycle event {index} duration drifted")
        expected_birth_observed = -1.0 < admission_frame <= last_observed_frame
        expected_end_observed = -1.0 < end_crossing_frame <= last_observed_frame
        action_intersects_window = not (
            end_frame < 0.0 or start_frame > last_observed_frame
        )
        if not action_intersects_window:
            expected_status = "completely_unobservable"
        elif start_frame < 0.0:
            expected_status = "left_truncated"
        elif expected_birth_observed and expected_end_observed:
            expected_status = "fully_observed"
        elif expected_birth_observed and end_crossing_frame > last_observed_frame:
            expected_status = "right_censored"
        elif not expected_birth_observed:
            expected_status = "completely_unobservable"
        else:
            raise ValueError(f"lifecycle event {index} status cannot be reconstructed")
        alive_first = max(0, int(admission_frame) + 1)
        alive_last = min(int(last_observed_frame), int(end_crossing_frame) - 1)
        expected_alive_rows = max(0, alive_last - alive_first + 1)
        expected_supervision = {
            "birth_supervision_rows": int(expected_birth_observed),
            "alive_supervision_rows": expected_alive_rows,
            "end_supervision_rows": int(expected_end_observed),
        }
        if (
            event["birth_observed"] != expected_birth_observed
            or event["end_observed"] != expected_end_observed
            or event["observation_status"] != expected_status
            or any(event[name] != value for name, value in expected_supervision.items())
        ):
            raise ValueError(f"lifecycle event {index} visibility semantics drifted")

    if census.get("video_list_sha256") != canonical_sha256(sorted(videos)):
        raise ValueError("D1.5 lifecycle census video-list hash drifted")
    observed_counts = {
        "video_count": len(videos),
        "annotation_count": len(events),
        "visible_birth_target_count": sum(
            event["birth_supervision_rows"] for event in events
        ),
        "observable_end_target_count": sum(
            event["end_supervision_rows"] for event in events
        ),
        "fully_observed_event_count": sum(
            event["observation_status"] == "fully_observed" for event in events
        ),
        "right_censored_event_count": sum(
            event["observation_status"] == "right_censored" for event in events
        ),
        "left_truncated_event_count": sum(
            event["observation_status"] == "left_truncated" for event in events
        ),
        "completely_unobservable_event_count": sum(
            event["observation_status"] == "completely_unobservable"
            for event in events
        ),
        "birth_supervision_row_count": sum(
            event["birth_supervision_rows"] for event in events
        ),
        "alive_supervision_row_count": sum(
            event["alive_supervision_rows"] for event in events
        ),
        "end_supervision_row_count": sum(
            event["end_supervision_rows"] for event in events
        ),
    }
    if census.get("counts") != observed_counts:
        raise ValueError("D1.5 lifecycle census counts do not reconstruct")
    if census.get("supervision_distribution") != _distribution_summary(events):
        raise ValueError("D1.5 lifecycle supervision distribution drifted")
    if require_official_counts and observed_counts != EXPECTED_LIFECYCLE_COUNTS:
        raise ValueError(
            f"official D1.5 lifecycle census drifted: {observed_counts!r}"
        )

    right_censored = tuple(
        sorted(
            (event["video_name"], event["event_id"], event["class_label"])
            for event in events
            if event["observation_status"] == "right_censored"
        )
    )
    unobservable = tuple(
        sorted(
            (event["video_name"], event["event_id"], event["class_label"])
            for event in events
            if event["observation_status"] == "completely_unobservable"
        )
    )
    if require_official_counts:
        if right_censored != EXPECTED_RIGHT_CENSORED_IDENTITIES:
            raise ValueError("official right-censored event identities drifted")
        if unobservable != EXPECTED_UNOBSERVABLE_IDENTITIES:
            raise ValueError("official unobservable event identities drifted")
    return census


def _percentile(values: list[float], probability: float) -> float:
    return _quantile(values, probability)


def _holm_adjust(raw_p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw_p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted = {}
    running_maximum = 0.0
    family_size = len(ordered)
    for index, (name, raw_value) in enumerate(ordered):
        candidate = min(1.0, (family_size - index) * float(raw_value))
        running_maximum = max(running_maximum, candidate)
        adjusted[name] = running_maximum
    return adjusted


def _analyze_effect_family(
    *,
    name: str,
    comparisons: tuple[dict, ...],
    outcome_by_key: Mapping[tuple[str, str, str, int], dict],
    observable_events: list[dict],
    video_names: list[str],
) -> dict:
    event_differences = {}
    cluster_sums = {}
    cluster_counts = {}
    observed = {}
    for comparison in comparisons:
        comparison_name = comparison["name"]
        differences = []
        sums = defaultdict(int)
        counts = defaultdict(int)
        improved = 0
        regressed = 0
        for event in observable_events:
            target_key = (event["video_name"], int(event["event_id"]))
            treatment = outcome_by_key[
                (
                    comparison["treatment_channel"],
                    comparison["treatment_route"],
                    *target_key,
                )
            ]
            control = outcome_by_key[
                (
                    comparison["control_channel"],
                    comparison["control_route"],
                    *target_key,
                )
            ]
            difference = int(treatment["primary_success"]) - int(
                control["primary_success"]
            )
            differences.append(difference)
            sums[event["video_name"]] += difference
            counts[event["video_name"]] += 1
            improved += int(difference > 0)
            regressed += int(difference < 0)
        denominator = len(differences)
        effect = sum(differences) / denominator
        event_differences[comparison_name] = differences
        cluster_sums[comparison_name] = sums
        cluster_counts[comparison_name] = counts
        observed[comparison_name] = {
            **comparison,
            "denominator_event_count": denominator,
            "treatment_success_count": sum(
                outcome_by_key[
                    (
                        comparison["treatment_channel"],
                        comparison["treatment_route"],
                        event["video_name"],
                        int(event["event_id"]),
                    )
                ]["primary_success"]
                for event in observable_events
            ),
            "control_success_count": sum(
                outcome_by_key[
                    (
                        comparison["control_channel"],
                        comparison["control_route"],
                        event["video_name"],
                        int(event["event_id"]),
                    )
                ]["primary_success"]
                for event in observable_events
            ),
            "improved_event_count": improved,
            "regressed_event_count": regressed,
            "unchanged_event_count": denominator - improved - regressed,
            "net_improved_event_count": sum(differences),
            "paired_effect": effect,
        }

    bootstrap_values = {comparison["name"]: [] for comparison in comparisons}
    bootstrap_rng = random.Random(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_videos = [
            video_names[bootstrap_rng.randrange(len(video_names))]
            for _ in video_names
        ]
        for comparison in comparisons:
            comparison_name = comparison["name"]
            numerator = sum(
                cluster_sums[comparison_name].get(video_name, 0)
                for video_name in sampled_videos
            )
            denominator = sum(
                cluster_counts[comparison_name].get(video_name, 0)
                for video_name in sampled_videos
            )
            if denominator <= 0:
                raise ValueError("cluster bootstrap sampled no observable event")
            bootstrap_values[comparison_name].append(numerator / denominator)

    extreme_counts = Counter()
    permutation_rng = random.Random(PERMUTATION_SEED)
    for _ in range(PERMUTATION_REPLICATES):
        signs = {
            video_name: (-1 if permutation_rng.getrandbits(1) else 1)
            for video_name in video_names
        }
        for comparison in comparisons:
            comparison_name = comparison["name"]
            permuted_effect = sum(
                signs[video_name]
                * cluster_sums[comparison_name].get(video_name, 0)
                for video_name in video_names
            ) / observed[comparison_name]["denominator_event_count"]
            if abs(permuted_effect) + 1e-15 >= abs(
                observed[comparison_name]["paired_effect"]
            ):
                extreme_counts[comparison_name] += 1

    raw_p_values = {
        comparison["name"]: (
            extreme_counts[comparison["name"]] + 1
        )
        / (PERMUTATION_REPLICATES + 1)
        for comparison in comparisons
    }
    adjusted_p_values = _holm_adjust(raw_p_values)
    minimum_net = math.ceil(EFFECT_SIZE_THRESHOLD * len(observable_events))
    results = {}
    for comparison in comparisons:
        comparison_name = comparison["name"]
        lower = _percentile(bootstrap_values[comparison_name], 0.025)
        upper = _percentile(bootstrap_values[comparison_name], 0.975)
        effect = observed[comparison_name]["paired_effect"]
        adjusted_p = adjusted_p_values[comparison_name]
        effect_pass = effect >= EFFECT_SIZE_THRESHOLD
        interval_pass = lower > 0.0
        holm_pass = adjusted_p <= FAMILYWISE_ALPHA
        results[comparison_name] = {
            **observed[comparison_name],
            "minimum_effect_threshold": EFFECT_SIZE_THRESHOLD,
            "minimum_net_improved_event_count": minimum_net,
            "cluster_bootstrap_ci95": [lower, upper],
            "raw_two_sided_cluster_sign_flip_p": raw_p_values[comparison_name],
            "holm_adjusted_p": adjusted_p,
            "effect_size_pass": effect_pass,
            "positive_interval_pass": interval_pass,
            "holm_familywise_pass": holm_pass,
            "scientific_gate_pass": effect_pass and interval_pass and holm_pass,
        }
    return {
        "family": name,
        "comparison_count": len(comparisons),
        "effect_metric": "same_event_near_end_end_and_immutable_emission_difference",
        "effect_size_threshold": EFFECT_SIZE_THRESHOLD,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "bootstrap": {
            "unit": "video",
            "method": "percentile",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "common_resamples_across_comparisons": True,
        },
        "permutation": {
            "unit": "video",
            "method": "two_sided_cluster_sign_flip",
            "replicates": PERMUTATION_REPLICATES,
            "seed": PERMUTATION_SEED,
            "common_signs_across_comparisons": True,
        },
        "multiple_comparison_correction": "holm",
        "comparisons": results,
    }


def build_paired_event_analysis(
    *,
    census: dict,
    outcome_rows: list[dict],
    channels: Iterable[str],
    routes: Iterable[str],
    require_official_counts: bool = True,
) -> dict:
    """Build the preregistered event-paired gate from validated route outcomes."""

    validate_lifecycle_census(
        census, require_official_counts=require_official_counts
    )
    channels = tuple(channels)
    routes = tuple(routes)
    visible_events = [
        event for event in census["events"] if event["birth_observed"]
    ]
    observable_events = [
        event
        for event in visible_events
        if event["observation_status"] == "fully_observed"
    ]
    expected_keys = {
        (channel, route, event["video_name"], int(event["event_id"]))
        for channel in channels
        for route in routes
        for event in visible_events
    }
    outcome_by_key = {}
    for index, row in enumerate(outcome_rows):
        if not isinstance(row, dict):
            raise ValueError(f"event outcome {index} is malformed")
        key = (
            row.get("channel"),
            row.get("route"),
            row.get("video_name"),
            _nonnegative_integer(row.get("event_id"), f"outcome[{index}].event_id"),
        )
        if key in outcome_by_key:
            raise ValueError("paired event outcomes duplicated a route/event")
        binary_fields = (
            "primary_success",
            "near_end_end",
            "immutable_emission",
            "premature_cancel",
            "unresolved_identity",
            "ambiguous_identity",
        )
        for field in binary_fields:
            value = row.get(field)
            if not isinstance(value, (bool, int)) or value not in (0, 1):
                raise ValueError(
                    f"paired event outcome {field} is not binary"
                )
        if row["unresolved_identity"] and row["ambiguous_identity"]:
            raise ValueError("paired event outcome has conflicting identity states")
        if row["primary_success"] and (
            row["unresolved_identity"]
            or row["ambiguous_identity"]
            or not row["near_end_end"]
            or not row["immutable_emission"]
        ):
            raise ValueError("paired event outcome primary-success semantics drifted")
        outcome_by_key[key] = row
    if set(outcome_by_key) != expected_keys:
        missing = len(expected_keys.difference(outcome_by_key))
        extra = len(set(outcome_by_key).difference(expected_keys))
        raise ValueError(
            f"paired event outcome coverage drifted: missing={missing}, extra={extra}"
        )

    route_summaries = {}
    for channel in channels:
        for route in routes:
            rows = [
                outcome_by_key[(channel, route, event["video_name"], event["event_id"])]
                for event in observable_events
            ]
            route_summaries[f"{channel}/{route}"] = {
                "observable_event_count": len(rows),
                "primary_success_count": sum(row["primary_success"] for row in rows),
                "near_end_end_count": sum(row["near_end_end"] for row in rows),
                "immutable_emission_count": sum(
                    row["immutable_emission"] for row in rows
                ),
                "premature_cancel_count": sum(
                    row["premature_cancel"] for row in rows
                ),
                "unresolved_identity_count": sum(
                    row["unresolved_identity"] for row in rows
                ),
                "ambiguous_identity_count": sum(
                    row["ambiguous_identity"] for row in rows
                ),
            }

    video_names = sorted({event["video_name"] for event in census["events"]})
    return {
        "protocol": PAIRED_ANALYSIS_PROTOCOL,
        "primary_metric": {
            "definition": (
                "target-backed END and immutable emission within a symmetric "
                "one-segment window around the annotated end"
            ),
            "window_feature_steps": 64,
            "right_censored_events_excluded_from_endpoint_denominator": True,
            "unresolved_identity_counted_as_primary_failure": True,
            "ambiguous_identity_counted_as_primary_failure": True,
            "unresolved_identity_interpreted_as_false_action": False,
        },
        "event_list_sha256": census["event_list_sha256"],
        "video_list_sha256": census["video_list_sha256"],
        "visible_event_count": len(visible_events),
        "observable_endpoint_event_count": len(observable_events),
        "right_censored_event_count": len(visible_events) - len(observable_events),
        "route_summaries": route_summaries,
        "formal_effect_family": _analyze_effect_family(
            name="formal_factorial_effects",
            comparisons=FORMAL_COMPARISONS,
            outcome_by_key=outcome_by_key,
            observable_events=observable_events,
            video_names=video_names,
        ),
        "no_cancel_effect_family": _analyze_effect_family(
            name="no_cancel_shadow_effects",
            comparisons=SHADOW_COMPARISONS,
            outcome_by_key=outcome_by_key,
            observable_events=observable_events,
            video_names=video_names,
        ),
        "outcome_rows": sorted(
            outcome_rows,
            key=lambda row: (
                row["video_name"],
                row["event_id"],
                row["channel"],
                row["route"],
            ),
        ),
    }
