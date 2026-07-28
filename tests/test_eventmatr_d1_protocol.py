"""Source-level fail-closed contracts for the D1 pre-experiment DAG."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.finalize_eventmatr_d1_pilot import validate_metric_lines


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
    assert protocol["gates"][2]["release_condition"] == (
        "all prior hard contracts pass"
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
    for field in ("duration", "true_duration", "video_time", "frame_to_time"):
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
