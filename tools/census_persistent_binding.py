"""Build the frozen split/lifecycle census consumed by formal launchers."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

from mmengine.config import Config

from opentad.utils.prefix_instance_schedule import (
    build_prefix_instance_schedule,
)


SCHEMA_VERSION = "persistent_binding_split_census.v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_ids(path):
    with Path(path).open("r", encoding="utf-8") as file:
        values = [line.strip() for line in file if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"split manifest repeats a video: {path}")
    return values


def _video_segments(video):
    duration = float(video["duration"])
    frames = int(video["frame"])
    if duration <= 0 or frames <= 0:
        raise ValueError("video duration and frame count must be positive")
    scale = frames / duration
    segments = []
    labels = []
    for index, annotation in enumerate(video.get("annotations", ())):
        if annotation.get("label") == "Ambiguous":
            continue
        start, end = (float(value) * scale for value in annotation["segment"])
        segments.append((start, end))
        labels.append(index)
    return segments, labels


def _census_video(
    video_id,
    video,
    cached,
    *,
    num_slots,
    max_start_offset_frames,
    feature_stride,
):
    source_frames = tuple(int(value) for value in cached.get("source_frames", ()))
    if int(cached.get("num_tokens", -1)) != len(source_frames):
        raise ValueError(f"{video_id}: cache token/source-frame mismatch")
    if not source_frames:
        raise ValueError(f"{video_id}: cache has no source frames")
    if any(
        right <= left
        for left, right in zip(source_frames, source_frames[1:])
    ):
        raise ValueError(f"{video_id}: source frames are not strictly increasing")
    if source_frames[-1] >= int(video["frame"]):
        raise ValueError(f"{video_id}: source frame exceeds video bounds")

    segments, labels = _video_segments(video)
    schedule = build_prefix_instance_schedule(
        segments,
        labels,
        decision_frames=source_frames,
        previous_frame=source_frames[0] - feature_stride,
    )
    occupied = set()
    seen_births = set()
    seen_ends = set()
    result = Counter()
    result["videos"] = 1
    result["tokens"] = len(source_frames)
    result["instances"] = len(segments)
    result["last_source_frame"] = source_frames[-1]
    result["video_frames"] = int(video["frame"])
    max_births = 0
    max_visible = 0
    max_birth_start_offset_tokens = 0.0

    for step_index, step in enumerate(schedule):
        births = {int(item.instance_id) for item in step.births}
        active = {int(item.instance_id) for item in step.active}
        ends = {int(item.instance_id) for item in step.ends}
        visible = active | ends
        same_instance_short = births & ends
        old_ends = ends - births
        new_births = births - ends
        free_at_entry = max(0, num_slots - len(occupied))
        entry_deficit = max(0, len(births) - free_at_entry)
        result["birth_events"] += len(births)
        result["end_events"] += len(ends)
        result["birth_positive_targets"] += len(births)
        result["birth_supervised_targets"] += free_at_entry
        result["alive_positive_targets"] += len(visible)
        result["alive_supervised_targets"] += num_slots
        result["end_positive_targets"] += len(ends)
        result["end_supervised_targets"] += len(visible)
        result["gt_entry_free_deficits"] += entry_deficit
        result["same_step_short_instances"] += len(same_instance_short)
        if old_ends and new_births:
            result["same_step_old_end_new_birth_steps"] += 1
            result["same_step_old_end_new_birth_pairs"] += (
                len(old_ends) * len(new_births)
            )
        if same_instance_short and step_index == len(schedule) - 1:
            result["final_token_short_instances"] += len(
                same_instance_short
            )
        max_births = max(max_births, len(births))
        max_visible = max(max_visible, len(visible))

        birth_items = {
            int(item.instance_id): item
            for item in step.births
        }
        for item in birth_items.values():
            offset_frames = (
                float(step.current_frame) - float(item.start_frame)
            )
            offset_tokens = offset_frames / float(feature_stride)
            max_birth_start_offset_tokens = max(
                max_birth_start_offset_tokens,
                offset_tokens,
            )
            if (
                offset_frames
                > float(max_start_offset_frames) + 1e-9
            ):
                result["clipped_start_supervision_targets"] += 1

        occupied.update(births)
        if len(occupied) > num_slots:
            result["oracle_capacity_overflow_steps"] += 1
        occupied.difference_update(ends)
        seen_births.update(births)
        seen_ends.update(ends)

    result["uncovered_birth_instances"] = len(segments) - len(seen_births)
    result["uncovered_endpoint_instances"] = len(segments) - len(seen_ends)
    result["max_births_per_step"] = max_births
    result["max_visible_instances"] = max_visible
    result["max_birth_start_offset_tokens"] = (
        max_birth_start_offset_tokens
    )
    return dict(result)


def _balance_summary(totals):
    summary = {}
    for channel in ("birth", "alive", "end"):
        positive = int(totals[f"{channel}_positive_targets"])
        supervised = int(totals[f"{channel}_supervised_targets"])
        if positive <= 0 or supervised <= positive:
            raise ValueError(
                f"{channel} supervision balance requires positives and negatives"
            )
        negative = supervised - positive
        summary[channel] = {
            "positive_targets": positive,
            "negative_targets": negative,
            "supervised_targets": supervised,
            "positive_rate": positive / supervised,
            "sqrt_negative_to_positive_ratio": math.sqrt(
                negative / positive
            ),
        }
    return summary


def build_census(config_path):
    cfg = Config.fromfile(config_path)
    contract = cfg.census_contract
    annotation_path = Path(cfg.annotation_path)
    cache_manifest_path = Path(cfg.feature_cache_manifest)
    annotation = _load_json(annotation_path)
    cache = _load_json(cache_manifest_path)
    if cache.get("annotation_sha256") != _sha256(annotation_path):
        raise ValueError("cache annotation hash differs from current annotations")
    if int(cache.get("feature_stride", -1)) != int(cfg.feature_stride):
        raise ValueError("cache feature stride differs from the route config")

    split_specs = {
        "fit_core": (
            Path(cfg.fit_core_manifest),
            "training",
        ),
        "calibration": (
            Path(cfg.calibration_manifest),
            "training",
        ),
        "reporting_locked": (
            Path(cfg.reporting_manifest),
            "validation",
        ),
    }
    split_ids = {
        name: _load_ids(path)
        for name, (path, _) in split_specs.items()
    }
    overlap = {}
    names = tuple(split_ids)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = sorted(set(split_ids[left]).intersection(split_ids[right]))
            overlap[f"{left}__{right}"] = shared

    failures = []
    if any(overlap.values()):
        failures.append("frozen split manifests overlap")
    expected_counts = dict(contract.expected_split_counts)
    for name, expected in expected_counts.items():
        actual = len(split_ids[name])
        if actual != int(expected):
            failures.append(
                f"{name} contains {actual} videos, expected {int(expected)}"
            )

    database = annotation["database"]
    cache_videos = cache["videos"]
    split_reports = {}
    global_totals = Counter()
    global_max_births = 0
    global_max_visible = 0
    memory_frames = int(cfg.memory_size) * int(cfg.feature_stride)
    max_start_offset_tokens = float(
        cfg.model.head.get("max_start_offset", cfg.memory_size)
    )
    max_start_offset_frames = (
        max_start_offset_tokens * int(cfg.feature_stride)
    )
    for name, (manifest_path, expected_subset) in split_specs.items():
        totals = Counter()
        per_video = {}
        for video_id in split_ids[name]:
            if video_id not in database:
                raise ValueError(f"{name}: annotation misses {video_id}")
            if video_id not in cache_videos:
                raise ValueError(f"{name}: cache misses {video_id}")
            video = database[video_id]
            if video.get("subset") != expected_subset:
                raise ValueError(
                    f"{name}: {video_id} has subset {video.get('subset')!r}"
                )
            row = _census_video(
                video_id,
                video,
                cache_videos[video_id],
                num_slots=int(cfg.num_slots),
                max_start_offset_frames=max_start_offset_frames,
                feature_stride=int(cfg.feature_stride),
            )
            per_video[video_id] = row
            totals.update(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {
                        "max_births_per_step",
                        "max_visible_instances",
                        "last_source_frame",
                        "video_frames",
                        "max_birth_start_offset_tokens",
                    }
                }
            )
            totals["max_births_per_step"] = max(
                totals["max_births_per_step"],
                row["max_births_per_step"],
            )
            totals["max_visible_instances"] = max(
                totals["max_visible_instances"],
                row["max_visible_instances"],
            )
            totals["max_birth_start_offset_tokens"] = max(
                totals["max_birth_start_offset_tokens"],
                row["max_birth_start_offset_tokens"],
            )
        split_reports[name] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "video_ids_sha256": hashlib.sha256(
                ("\n".join(split_ids[name]) + "\n").encode("utf-8")
            ).hexdigest(),
            "totals": dict(totals),
            "per_video": per_video,
        }
        global_max_births = max(
            global_max_births,
            totals["max_births_per_step"],
        )
        global_max_visible = max(
            global_max_visible,
            totals["max_visible_instances"],
        )
        global_totals.update(
            {
                key: value
                for key, value in totals.items()
                if key not in {
                    "max_births_per_step",
                    "max_visible_instances",
                    "max_birth_start_offset_tokens",
                }
            }
        )
    global_totals["max_births_per_step"] = global_max_births
    global_totals["max_visible_instances"] = global_max_visible
    global_totals["max_birth_start_offset_tokens"] = max(
        report["totals"]["max_birth_start_offset_tokens"]
        for report in split_reports.values()
    )

    limits = {
        "max_births_per_step": int(cfg.model.head.max_births_per_step),
        "max_visible_instances": int(cfg.num_slots),
        "gt_entry_free_deficits": int(
            contract.max_gt_entry_free_deficits
        ),
        "oracle_capacity_overflow_steps": int(
            contract.max_oracle_capacity_overflow_steps
        ),
        "uncovered_birth_instances": int(
            contract.max_uncovered_birth_instances
        ),
        "uncovered_endpoint_instances": int(
            contract.max_uncovered_endpoint_instances
        ),
        "clipped_start_supervision_targets": int(
            contract.max_clipped_start_supervision_targets
        ),
    }
    for field, limit in limits.items():
        actual = int(global_totals[field])
        if actual > limit:
            failures.append(f"{field}={actual} exceeds frozen limit {limit}")
    if (
        float(global_totals["max_birth_start_offset_tokens"])
        > max_start_offset_tokens + 1e-9
    ):
        failures.append(
            "max_birth_start_offset_tokens="
            f"{global_totals['max_birth_start_offset_tokens']} exceeds "
            f"head limit {max_start_offset_tokens}"
        )

    fit_balance = _balance_summary(
        split_reports["fit_core"]["totals"]
    )
    balance_contract = cfg.get("supervision_balance_contract")
    if balance_contract is not None:
        expected_balance_counts = {
            "tokens": int(balance_contract.tokens),
            "birth_positive_targets": int(
                balance_contract.birth_positive_targets
            ),
            "birth_supervised_targets": int(
                balance_contract.birth_supervised_targets
            ),
            "alive_positive_targets": int(
                balance_contract.alive_positive_targets
            ),
            "alive_supervised_targets": int(
                balance_contract.alive_supervised_targets
            ),
            "end_positive_targets": int(
                balance_contract.end_positive_targets
            ),
            "end_supervised_targets": int(
                balance_contract.end_supervised_targets
            ),
        }
        fit_totals = split_reports["fit_core"]["totals"]
        for field, expected in expected_balance_counts.items():
            if int(fit_totals[field]) != expected:
                failures.append(
                    f"fit_core {field}={fit_totals[field]} "
                    f"differs from registered {expected}"
                )
        channel_settings = {
            "birth": (
                float(cfg.model.head.birth_prior_probability),
                float(cfg.model.birth_positive_weight),
            ),
            "alive": (
                float(cfg.model.head.alive_prior_probability),
                float(cfg.model.alive_positive_weight),
            ),
            "end": (
                float(cfg.model.head.end_prior_probability),
                float(cfg.model.end_positive_weight),
            ),
        }
        for channel, (prior, positive_weight) in channel_settings.items():
            measured = fit_balance[channel]
            if not math.isclose(
                prior,
                measured["positive_rate"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                failures.append(
                    f"{channel} prior {prior} differs from fit-only "
                    f"rate {measured['positive_rate']}"
                )
            if not math.isclose(
                positive_weight,
                measured["sqrt_negative_to_positive_ratio"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                failures.append(
                    f"{channel} positive weight {positive_weight} differs "
                    "from the registered fit-only square-root balance "
                    f"{measured['sqrt_negative_to_positive_ratio']}"
                )

    return {
        "schema_version": SCHEMA_VERSION,
        "passed": not failures,
        "failures": failures,
        "route_stage": cfg.route_stage,
        "coordinate_system": "zero_based_source_frames",
        "annotation_sha256": _sha256(annotation_path),
        "cache_manifest_sha256": _sha256(cache_manifest_path),
        "cache_encoder_id": cache.get("encoder_id"),
        "feature_stride": int(cfg.feature_stride),
        "num_slots": int(cfg.num_slots),
        "memory_size_tokens": int(cfg.memory_size),
        "memory_horizon_frames": memory_frames,
        "max_start_offset_tokens": max_start_offset_tokens,
        "max_start_offset_frames": max_start_offset_frames,
        "fit_supervision_balance": fit_balance,
        "split_overlap": overlap,
        "expected_split_counts": expected_counts,
        "limits": limits,
        "totals": dict(global_totals),
        "splits": split_reports,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="write a failed census without returning a failing exit code",
    )
    args = parser.parse_args()
    payload = build_census(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    if not payload["passed"] and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
