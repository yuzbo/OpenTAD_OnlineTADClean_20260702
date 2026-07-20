import argparse
import json
import math
from pathlib import Path
import re


EXPECTED_PROFILE_SEED = 705
EXPECTED_WARMUP_STEPS = 50
EXPECTED_MEASURED_STEPS = 200
NO_FUTURE_KEYS = {
    "future_end_violations",
    "future_source_violations",
    "negative_latency_rows",
    "non_monotonic_emit_rows",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply the frozen cost and stability gate to binding profiles"
    )
    parser.add_argument("--fixed-train", required=True)
    parser.add_argument("--rematch-train", required=True)
    parser.add_argument("--fixed-calibration-inference", required=True)
    parser.add_argument("--rematch-calibration-inference", required=True)
    parser.add_argument("--reporting-chunks", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--safety-factor", type=float, default=1.25)
    parser.add_argument("--paired-gpu-hour-cap", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _require_positive_finite(value, message):
    value = float(value)
    _require(math.isfinite(value) and value > 0, message)
    return value


def _validate_common_profile(profile, mode, binding_mode):
    _require(profile.get("passed") is True, "an input profile did not pass")
    _require(profile.get("mode") == mode, f"expected a {mode} profile")
    _require(
        profile.get("binding_mode") == binding_mode,
        f"profile does not use {binding_mode}",
    )
    _require(
        int(profile.get("seed", -1)) == EXPECTED_PROFILE_SEED,
        f"profile seed must be frozen at {EXPECTED_PROFILE_SEED}",
    )
    _require(
        int(profile.get("warmup_steps", -1)) == EXPECTED_WARMUP_STEPS,
        f"profile warmup must be frozen at {EXPECTED_WARMUP_STEPS} steps",
    )
    _require(
        int(profile.get("measured_steps", -1)) == EXPECTED_MEASURED_STEPS,
        f"profile measurement must be frozen at {EXPECTED_MEASURED_STEPS} steps",
    )
    _require(profile.get("amp") is False, "feature profiles must use FP32")
    _require(
        profile.get("load_from_raw_predictions") is False,
        "profile loaded raw predictions",
    )
    _require(
        profile.get("save_raw_prediction") is False,
        "profile enabled a raw-prediction cache",
    )
    _require(profile.get("raw_video_finetuning") is False, "profile used raw video")
    _require(
        profile.get("strict_causal_control") is True,
        "profile did not use strict causal stream control",
    )
    _require(
        profile.get("streaming_safe_emission") is True,
        "profile did not use immutable online emissions",
    )
    _require(
        profile.get("sliding_window") is False,
        "profile enabled offline sliding-window merging",
    )
    _require(
        int(profile.get("dataset_chunks", 0))
        >= EXPECTED_WARMUP_STEPS + EXPECTED_MEASURED_STEPS,
        "profile split is too short for the frozen measurement window",
    )
    _require(int(profile.get("dataset_videos", 0)) > 0, "profile has no videos")
    _require(
        re.fullmatch(r"[0-9a-f]{64}", str(profile.get("dataset_video_ids_sha256", "")))
        is not None,
        "profile is missing its dataset identity digest",
    )
    _require_positive_finite(
        profile.get("timing_seconds", {}).get("mean"),
        "profile mean step time must be positive and finite",
    )
    _require_positive_finite(
        profile.get("estimated_full_pass_gpu_hours"),
        "profile full-pass estimate must be positive and finite",
    )
    _require_positive_finite(
        profile.get("max_gpu_memory_mib"),
        "profile peak GPU memory must be positive and finite",
    )


def evaluate_profiles(
    fixed,
    rematch,
    fixed_inference,
    rematch_inference,
    reporting_chunks,
    epochs=12,
    safety_factor=1.25,
    paired_gpu_hour_cap=2.0,
):
    binding_modes = ("fixed_birth_slot", "prefix_rematch_active_pool")
    train_profiles = (fixed, rematch)
    inference_profiles = (fixed_inference, rematch_inference)
    for profile, binding_mode in zip(train_profiles, binding_modes):
        _validate_common_profile(profile, "train", binding_mode)
        _require(profile.get("split") == "train", "training profile must use fit split")
        _require(
            profile.get("dataset_test_mode") is False,
            "training profile unexpectedly disabled supervision",
        )
        _require(
            profile.get("update_audit", {}).get("passed") is True,
            "training profile did not update the head",
        )
        _require(
            int(profile.get("dropped_gt_birth_targets", -1)) == 0,
            "training profile dropped GT birth targets",
        )
        _require(
            int(profile.get("runtime_capacity_exhaustions", -1)) == 0,
            "training profile exhausted predicted runtime capacity",
        )
    _require(
        len({profile["dataset_chunks"] for profile in train_profiles}) == 1,
        "FIXED and REMATCH training profiles used different fit data",
    )
    _require(
        len({profile["dataset_videos"] for profile in train_profiles}) == 1,
        "FIXED and REMATCH training profiles used different fit videos",
    )
    _require(
        len({profile["dataset_video_ids_sha256"] for profile in train_profiles}) == 1,
        "FIXED and REMATCH training profiles used different fit identities",
    )

    for profile, binding_mode in zip(inference_profiles, binding_modes):
        _validate_common_profile(profile, "inference", binding_mode)
        _require(
            profile.get("split") == "val",
            "inference profile must use the calibration split",
        )
        _require(
            profile.get("dataset_test_mode") is True,
            "inference profile must not construct supervision schedules",
        )
        summary = profile.get("emission_summary", {})
        _require(
            int(summary.get("num_emissions", 0)) > 0,
            "inference profile produced no final intervals",
        )
        no_future = summary.get("no_future", {})
        _require(
            set(no_future) == NO_FUTURE_KEYS,
            "inference profile has an incomplete no-future audit",
        )
        _require(
            all(int(no_future[key]) == 0 for key in NO_FUTURE_KEYS),
            "inference profile violated causality",
        )
        _require(
            re.fullmatch(r"[0-9a-f]{64}", str(profile.get("emission_ledger_sha256", "")))
            is not None,
            "inference profile is missing its canonical ledger digest",
        )
    _require(
        len({profile["dataset_chunks"] for profile in inference_profiles}) == 1,
        "FIXED and REMATCH inference profiles used different calibration data",
    )
    _require(
        len({profile["dataset_videos"] for profile in inference_profiles}) == 1,
        "FIXED and REMATCH inference profiles used different calibration videos",
    )
    _require(
        len(
            {
                profile["dataset_video_ids_sha256"]
                for profile in inference_profiles
            }
        )
        == 1,
        "FIXED and REMATCH inference profiles used different calibration identities",
    )
    _require(
        fixed_inference["emission_summary"] == rematch_inference["emission_summary"],
        "FIXED and REMATCH changed inference behavior before training",
    )
    _require(
        fixed_inference["emission_ledger_sha256"]
        == rematch_inference["emission_ledger_sha256"],
        "FIXED and REMATCH changed the exact inference ledger before training",
    )
    _require(int(reporting_chunks) > 0, "reporting chunk count must be positive")
    _require(int(epochs) > 0, "epochs must be positive")
    _require(float(safety_factor) >= 1.0, "safety factor must be at least one")
    _require(
        float(paired_gpu_hour_cap) > 0,
        "paired GPU-hour cap must be positive",
    )

    train_hours = sum(
        float(profile["estimated_full_pass_gpu_hours"]) * int(epochs)
        for profile in train_profiles
    )
    calibration_hours = sum(
        float(profile["estimated_full_pass_gpu_hours"])
        for profile in inference_profiles
    )
    reporting_hours = sum(
        float(profile["timing_seconds"]["mean"]) * int(reporting_chunks) / 3600.0
        for profile in inference_profiles
    )
    raw_paired_hours = train_hours + calibration_hours + reporting_hours
    budgeted_paired_hours = raw_paired_hours * float(safety_factor)
    budget_passed = budgeted_paired_hours <= float(paired_gpu_hour_cap)
    return {
        "passed": budget_passed,
        "stability_passed": True,
        "budget_passed": budget_passed,
        "binding_inference_equivalence_passed": True,
        "epochs": int(epochs),
        "safety_factor": float(safety_factor),
        "paired_gpu_hour_cap": float(paired_gpu_hour_cap),
        "estimated_pair_gpu_hours": {
            "training": train_hours,
            "calibration_inference": calibration_hours,
            "reporting_inference": reporting_hours,
            "raw_total": raw_paired_hours,
            "with_safety_factor": budgeted_paired_hours,
        },
        "fixed_train": {
            "step_seconds": fixed["timing_seconds"],
            "max_gpu_memory_mib": fixed["max_gpu_memory_mib"],
            "dataset_chunks": fixed["dataset_chunks"],
            "runtime_capacity_exhaustions": fixed["runtime_capacity_exhaustions"],
        },
        "rematch_train": {
            "step_seconds": rematch["timing_seconds"],
            "max_gpu_memory_mib": rematch["max_gpu_memory_mib"],
            "dataset_chunks": rematch["dataset_chunks"],
            "runtime_capacity_exhaustions": rematch["runtime_capacity_exhaustions"],
        },
        "fixed_calibration_inference": {
            "step_seconds": fixed_inference["timing_seconds"],
            "max_gpu_memory_mib": fixed_inference["max_gpu_memory_mib"],
            "dataset_chunks": fixed_inference["dataset_chunks"],
            "emission_summary": fixed_inference["emission_summary"],
        },
        "rematch_calibration_inference": {
            "step_seconds": rematch_inference["timing_seconds"],
            "max_gpu_memory_mib": rematch_inference["max_gpu_memory_mib"],
            "dataset_chunks": rematch_inference["dataset_chunks"],
            "emission_summary": rematch_inference["emission_summary"],
        },
        "reporting_chunks": int(reporting_chunks),
    }


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = evaluate_profiles(
            _load(args.fixed_train),
            _load(args.rematch_train),
            _load(args.fixed_calibration_inference),
            _load(args.rematch_calibration_inference),
            reporting_chunks=args.reporting_chunks,
            epochs=args.epochs,
            safety_factor=args.safety_factor,
            paired_gpu_hour_cap=args.paired_gpu_hour_cap,
        )
    except Exception as error:
        payload = {
            "passed": False,
            "stability_passed": False,
            "budget_passed": False,
            "error": str(error),
            "error_type": type(error).__name__,
        }
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        raise
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit("paired seed estimate exceeds the frozen GPU-hour cap")


if __name__ == "__main__":
    main()
