import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_persistent_binding_profile import evaluate_profiles  # noqa: E402


def _train_profile(binding_mode, step_seconds=0.05, runtime_exhaustions=0):
    return {
        "passed": True,
        "mode": "train",
        "binding_mode": binding_mode,
        "split": "train",
        "seed": 705,
        "warmup_steps": 50,
        "measured_steps": 200,
        "amp": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "raw_video_finetuning": False,
        "strict_causal_control": True,
        "streaming_safe_emission": True,
        "sliding_window": False,
        "estimated_full_pass_gpu_hours": step_seconds * 2010 / 3600,
        "timing_seconds": {"mean": step_seconds},
        "max_gpu_memory_mib": 256,
        "dataset_chunks": 2010,
        "dataset_videos": 160,
        "dataset_test_mode": False,
        "dataset_video_ids_sha256": "c" * 64,
        "dropped_gt_birth_targets": 0,
        "runtime_capacity_exhaustions": runtime_exhaustions,
        "update_audit": {"passed": True},
    }


def _inference_profile(binding_mode, step_seconds=0.02):
    return {
        "passed": True,
        "mode": "inference",
        "binding_mode": binding_mode,
        "split": "val",
        "seed": 705,
        "warmup_steps": 50,
        "measured_steps": 200,
        "amp": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "raw_video_finetuning": False,
        "strict_causal_control": True,
        "streaming_safe_emission": True,
        "sliding_window": False,
        "estimated_full_pass_gpu_hours": step_seconds * 469 / 3600,
        "timing_seconds": {"mean": step_seconds},
        "max_gpu_memory_mib": 128,
        "dataset_chunks": 469,
        "dataset_videos": 40,
        "dataset_test_mode": True,
        "dataset_video_ids_sha256": "d" * 64,
        "emission_ledger_sha256": "a" * 64,
        "emission_summary": {
            "num_emissions": 10,
            "no_future": {
                "future_end_violations": 0,
                "future_source_violations": 0,
                "negative_latency_rows": 0,
                "non_monotonic_emit_rows": 0,
            },
        },
    }


def test_profile_gate_accepts_stable_pair_inside_frozen_budget():
    result = evaluate_profiles(
        _train_profile("fixed_birth_slot"),
        _train_profile("prefix_rematch_active_pool"),
        _inference_profile("fixed_birth_slot"),
        _inference_profile("prefix_rematch_active_pool"),
        reporting_chunks=2719,
    )

    assert result["passed"] is True
    assert result["stability_passed"] is True
    assert result["budget_passed"] is True
    assert result["estimated_pair_gpu_hours"]["with_safety_factor"] <= 2


def test_profile_gate_rejects_runtime_exhaustion_before_full_seed():
    try:
        evaluate_profiles(
            _train_profile("fixed_birth_slot", runtime_exhaustions=1),
            _train_profile("prefix_rematch_active_pool"),
            _inference_profile("fixed_birth_slot"),
            _inference_profile("prefix_rematch_active_pool"),
            reporting_chunks=2719,
        )
    except ValueError as error:
        assert "runtime capacity" in str(error)
    else:
        raise AssertionError("runtime exhaustion must reject the profile")


def test_profile_gate_rejects_pair_above_two_gpu_hours():
    result = evaluate_profiles(
        _train_profile("fixed_birth_slot", step_seconds=0.2),
        _train_profile("prefix_rematch_active_pool", step_seconds=0.2),
        _inference_profile("fixed_birth_slot", step_seconds=0.1),
        _inference_profile("prefix_rematch_active_pool", step_seconds=0.1),
        reporting_chunks=2719,
    )

    assert result["passed"] is False
    assert result["budget_passed"] is False


def test_profile_gate_rejects_inference_behavior_changed_by_binding_control():
    rematch_inference = _inference_profile("prefix_rematch_active_pool")
    rematch_inference["emission_ledger_sha256"] = "b" * 64

    try:
        evaluate_profiles(
            _train_profile("fixed_birth_slot"),
            _train_profile("prefix_rematch_active_pool"),
            _inference_profile("fixed_birth_slot"),
            rematch_inference,
            reporting_chunks=2719,
        )
    except ValueError as error:
        assert "exact inference ledger" in str(error)
    else:
        raise AssertionError("binding control must not alter untrained inference")


def test_profiler_and_n16r4_launcher_freeze_scope_and_measurement_contracts():
    profiler = (ROOT / "tools/profile_persistent_binding.py").read_text(
        encoding="utf-8"
    )
    submitter = (
        ROOT / "tools/remote/submit_persistent_binding_profile_n16r4.sh"
    ).read_text(encoding="utf-8")

    assert "warmup-steps" in profiler
    assert "measured-steps" in profiler
    assert "snapshot_trainable_parameters" in profiler
    assert "audit_training_update" in profiler
    assert "dropped_gt_birth_targets" in profiler
    assert "runtime_capacity_exhaustions" in profiler
    assert "validate_emission_ledger_summary" in profiler
    assert "emission_ledger_sha256" in profiler
    assert "profiling cannot load raw predictions" in profiler

    assert "WARMUP_STEPS=50" in submitter
    assert "MEASURED_STEPS=200" in submitter
    assert "PAIRED_GPU_HOUR_CAP=2" in submitter
    assert "SMOKE_RUN_DIR" in submitter
    assert "gate_summary.json" in submitter
    assert "#SBATCH --gres=gpu:1" in submitter
    assert "sbatch" in submitter
    assert "dataset.val.test_mode=True" in submitter
    assert "fixed_calibration_inference_profile.json" in submitter
    assert "rematch_calibration_inference_profile.json" in submitter
    assert "REPORTING_CHUNKS" in submitter
    assert "tools/evaluate_persistent_binding_profile.py" in submitter
    assert "tools/train.py" not in submitter
