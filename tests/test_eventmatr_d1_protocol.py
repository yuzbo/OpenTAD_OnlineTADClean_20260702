"""Source-level fail-closed contracts for the D1 pre-experiment DAG."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.finalize_eventmatr_d1_pilot import validate_metric_lines
from scripts.finalize_eventmatr_d11_mechanism import validate_mechanism_metrics


ROOT = Path(__file__).resolve().parents[1]


def test_d1_preexperiment_factorization_and_access_policy() -> None:
    protocol = json.loads(
        (
            ROOT
            / "experiment_configs"
            / "eventmatr_d1_preexperiments.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["base_training_source"]["commit"] == (
        "92cf34aa07bebee2a7a7e3661431d5055804b29b"
    )
    assert list(protocol["lanes"]) == ["N", "R", "T", "H", "TH"]
    assert protocol["lanes"]["N"]["model_variant"] == "native_matr"
    assert [
        protocol["lanes"][lane]["event_d1_lane"]
        for lane in ("R", "T", "H", "TH")
    ] == ["r", "t", "h", "th"]
    assert protocol["fixed_preexperiment_seed"] == 52
    assert protocol["test_access"] is False
    assert protocol["strict_causal_paper_result_valid"] is False
    assert all(protocol["forbidden"].values())
    assert protocol["gates"][2]["stage"] == "d11_seed52_one_epoch_mechanism"
    assert protocol["gates"][2]["epochs"] == [1]
    assert protocol["gates"][2]["lanes"] == ["TH"]
    assert protocol["gates"][2]["release_condition"] == (
        "all prior hard contracts pass"
    )
    assert protocol["gates"][3]["release_condition"] == (
        "D1.1 one-epoch mechanism receipt is PASS"
    )


def test_d1_microexperiment_receipt_is_diagnostic_only() -> None:
    source = (
        ROOT / "scripts" / "run_eventmatr_d1_microexperiments.py"
    ).read_text(encoding="utf-8")
    assert '"test_access": False' in source
    assert '"checkpoint_updated": False' in source
    assert '"strict_causal_paper_result_valid": False' in source
    assert "gradient_dilution_experiment" in source
    assert "censoring_experiment" in source
    assert "runtime_stress_experiment" in source
    assert "integrated_lane_experiment" in source


def test_d1_model_boundary_lists_every_forbidden_future_field() -> None:
    source = (ROOT / "on_tal_task.py").read_text(encoding="utf-8")
    for field in (
        "duration",
        "true_duration",
        "video_time",
        "frame_to_time",
        "segment_flag",
    ):
        assert f'"{field}"' in source
    assert "make_model_inputs" in source


def test_d1_real_smoke_covers_registered_lanes_without_test_access() -> None:
    source = (
        ROOT / "scripts" / "run_eventmatr_d1_real_smoke.py"
    ).read_text(encoding="utf-8")
    assert 'REGISTERED_LANES = ("N", "R", "T", "H", "TH")' in source
    assert "THUMOS14Dataset" in source
    assert "_load_real_batch" in source
    assert "_run_lane" in source
    assert '"test_access": False' in source
    assert '"checkpoint_updated": False' in source
    assert '"strict_causal_paper_result_valid": False' in source
    assert "torch.randn" not in source


def test_d1_slurm_smoke_is_identity_gated_and_runs_real_batch_last() -> None:
    source = (
        ROOT / "scripts" / "slurm_eventmatr_d1_smoke.sh"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in source
    assert "#SBATCH --mem" not in source
    assert "verify_source_identity.py" in source
    assert "eventmatr_d1_preexperiments.json" in source
    assert "run_eventmatr_d1_microexperiments.py" in source
    assert "run_eventmatr_d1_real_smoke.py" in source
    assert source.find("run_eventmatr_d1_real_smoke.py") > source.find("pytest")
    assert "MATR_D1_SMOKE_RECEIPT" in source


def test_d1_seed52_pilot_array_is_registered_and_train_only() -> None:
    slurm = (
        ROOT / "scripts" / "slurm_eventmatr_d1_pilot_array.sh"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT / "scripts" / "train_eventmatr_d1_pilot.sh"
    ).read_text(encoding="utf-8")
    finalizer = (
        ROOT / "scripts" / "finalize_eventmatr_d1_pilot.py"
    ).read_text(encoding="utf-8")
    assert "LANES=(N R T H TH)" in slurm
    assert "HORIZONS=(5 10 20)" in slurm
    assert "TASK_ID >= 15" in slurm
    assert "#SBATCH --gpus=1" in slurm
    assert "#SBATCH --mem" not in slurm
    assert "verify_source_identity.py" in slurm
    assert "--smoke-receipt" in slurm
    assert "--study_protocol d1_preexperiment" in launcher
    assert "--random_seed 52" in launcher
    assert "LOCKED_TEST_NOT_MOUNTED.pickle" in launcher
    assert "--event_teacher_forcing_ratio 0.5" in launcher
    assert '"test_access": False' in finalizer
    assert '"strict_causal_paper_result_valid": False' in finalizer
    assert '"train_prefix_metrics_diagnostic_only": True' in finalizer


def test_d11_one_epoch_mechanism_is_singleton_fresh_and_train_only() -> None:
    slurm = (
        ROOT / "scripts" / "slurm_eventmatr_d11_mechanism.sh"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT / "scripts" / "train_eventmatr_d11_mechanism.sh"
    ).read_text(encoding="utf-8")
    finalizer = (
        ROOT / "scripts" / "finalize_eventmatr_d11_mechanism.py"
    ).read_text(encoding="utf-8")
    task = (ROOT / "on_tal_task.py").read_text(encoding="utf-8")

    assert "#SBATCH --gpus=1" in slurm
    assert "SLURM_ARRAY_TASK_ID" not in slurm
    assert "MATR_LANE=TH" in slurm
    assert "--smoke-receipt" in slurm
    assert "--epochs 1" in launcher
    assert "--train_eval_step 1" in launcher
    assert "--study_protocol d11_mechanism" in launcher
    assert "--event_d1_lane th" in launcher
    assert "--random_seed 52" in launcher
    assert "LOCKED_TEST_NOT_MOUNTED.pickle" in launcher
    assert "--load_model" not in launcher
    assert '"performance_gate_applied": False' in finalizer
    assert '"five_epoch_matrix_release": True' in finalizer
    assert "eventmatr_d11_ternary_owner_v1" in finalizer
    assert "validate_d1_checkpoint_compatibility" in task
    assert "D11_CHECKPOINT_SCHEMA" in task


def test_d11_mechanism_gate_requires_liveness_without_effect_thresholds() -> None:
    metrics = {
        "event_transition_gradient_norm": 1.0,
        "event_owner_gradient_norm": 1.0,
        "event_birth_positive_count_unscaled": 1.0,
        "event_end_positive_count_unscaled": 1.0,
        "event_owner_assignment_count_unscaled": 1.0,
        "event_ragged_track_count_unscaled": 1.0,
        "event_false_track_cancel_group_count_unscaled": 1.0,
        "event_source_predicted_associated_row_count_unscaled": 1.0,
        "event_source_predicted_unmatched_row_count_unscaled": 1.0,
        "event_source_teacher_birth_row_count_unscaled": 1.0,
        "event_association_predicted_associated_count_unscaled": 1.0,
        "event_runtime_capacity_exhaustions_unscaled": 0.0,
    }
    validate_mechanism_metrics(metrics)
    with pytest.raises(ValueError, match="not live"):
        validate_mechanism_metrics({**metrics, "event_owner_gradient_norm": 0.0})
    with pytest.raises(ValueError, match="exhausted"):
        validate_mechanism_metrics(
            {**metrics, "event_runtime_capacity_exhaustions_unscaled": 1.0}
        )


def test_d1_pilot_finalizer_rejects_empty_or_incomplete_metrics() -> None:
    empty = [
        {
            "epoch": epoch,
            "metrics": {},
            "test_access": False,
            "strict_causal_paper_result_valid": False,
        }
        for epoch in range(1, 6)
    ]
    with pytest.raises(ValueError, match="non-empty"):
        validate_metric_lines(empty, "N", 5)

    common_only = [
        {
            "epoch": epoch,
            "metrics": {
                key: 0.0
                for key in (
                    "loss",
                    "loss_cls",
                    "loss_flag",
                    "loss_reg_l1",
                    "loss_reg_diou",
                    "loss_reg_stcls",
                    "lr",
                    "mAP_train",
                    "mAP_03_train",
                    "mAP_04_train",
                    "mAP_05_train",
                    "mAP_06_train",
                    "mAP_07_train",
                )
            },
            "test_access": False,
            "strict_causal_paper_result_valid": False,
        }
        for epoch in range(1, 6)
    ]
    with pytest.raises(ValueError, match="required keys"):
        validate_metric_lines(common_only, "TH", 5)


def test_d1_checkpoint_replay_is_train_only_read_only_and_threshold_free() -> None:
    manifest = json.loads(
        (
            ROOT
            / "experiment_configs"
            / "eventmatr_d1_checkpoint_replay.json"
        ).read_text(encoding="utf-8")
    )
    runner = (
        ROOT / "scripts" / "run_eventmatr_d1_checkpoint_replay.py"
    ).read_text(encoding="utf-8")
    slurm = (
        ROOT / "scripts" / "slurm_eventmatr_d1_checkpoint_replay.sh"
    ).read_text(encoding="utf-8")
    finalizer = (
        ROOT / "scripts" / "finalize_eventmatr_d1_checkpoint_replay.py"
    ).read_text(encoding="utf-8")

    assert manifest["lanes"] == ["R", "T", "H", "TH"]
    assert manifest["scope"]["dataset_subset"] == "official_train_validation_only"
    assert manifest["scope"]["ground_truth_visible_to_model"] is False
    assert manifest["scope"]["locked_test_access"] is False
    assert manifest["scope"]["checkpoint_update"] is False
    assert manifest["fixed_decisions"]["birth_logit_threshold"] is None
    assert manifest["fixed_decisions"]["end_logit_threshold"] is None
    assert manifest["fixed_decisions"]["threshold_search"] is False

    assert 'THUMOS14Dataset(args, subset="train")' in runner
    assert "make_model_inputs(args, features, infos)" in runner
    assert "make_model_inputs(args, features, infos, target" not in runner
    assert "D1_RUNTIME_FORBIDDEN_MODEL_INFO" in runner
    assert '"test_access": False' in runner
    assert '"checkpoint_updated": False' in runner
    assert '"threshold_search": False' in runner
    assert "model.load_state_dict" in runner and "strict=True" in runner
    assert "optimizer" not in runner

    assert "LANES=(R T H TH)" in slurm
    assert "TASK_ID >= 4" in slurm
    assert "#SBATCH --gpus=1" in slurm
    assert "LOCKED_TEST" not in slurm
    assert "--trace" in slurm
    assert "verify_source_identity.py" in slurm

    assert '"test_access": False' in finalizer
    assert '"checkpoint_updated": False' in finalizer
    assert '"threshold_search": False' in finalizer
    assert "stage[\"ledger_rows\"] != stage[\"emissions\"]" in finalizer
