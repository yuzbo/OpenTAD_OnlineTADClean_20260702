"""Source-level fail-closed contracts for the D1 pre-experiment DAG."""

from __future__ import annotations

import json
from pathlib import Path


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
