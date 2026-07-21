import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "tools/evaluate_persistent_binding_formal12.py"
    spec = importlib.util.spec_from_file_location(
        "evaluate_persistent_binding_formal12",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(arm, *, predictions=100, ratio=1.0, recall=0.5, gpu_hours=6.0):
    binding = {
        "fixed": "fixed_birth_slot",
        "rematch": "prefix_rematch_active_pool",
    }[arm]
    provenance = {
        "code_commit": "a" * 40,
        "annotation_sha256": "b" * 64,
        "calibration_manifest_sha256": "c" * 64,
        "fit_manifest_sha256": "d" * 64,
        "reporting_manifest_sha256": "e" * 64,
        "census_sha256": "f" * 64,
        "formal_contract_sha256": "1" * 64,
        "calibration_protocol_sha256": "2" * 64,
        "checkpoint_sha256": ("3" if arm == "fixed" else "4") * 64,
        "config_sha256": ("5" if arm == "fixed" else "6") * 64,
        "emission_ledger_sha256": ("7" if arm == "fixed" else "8") * 64,
    }
    row = {
        "arm": arm,
        "binding_mode": binding,
        "seed": 705,
        "training_epochs": 12,
        "fixed_threshold": 0.5,
        "metric_schema": "standard_ontad_map.v1",
        "instance_metric_schema": "online_instance_metrics.v2",
        "map_unit": "percentage_points",
        "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        "average_map_pct": 10.0,
        "duplicate_rate": 0.2,
        "fragmentation_rate": 0.2,
        "prediction_gt_ratio": ratio,
        "recall_tiou_0p3": recall,
        "protocol_violations": 0,
        "committed_predictions": predictions,
        "expected_updates": 24120,
        "successful_updates": 24120,
        "scheduler_steps": 24120,
        "skipped_updates": 0,
        "gt_supervision_exhaustions": 0,
        "gt_birth_runtime_entry_free_collisions": 0,
        "allocated_gpu_hours": gpu_hours,
        "reporting_accessed": False,
        "threshold_search": False,
        "raw_rgb_authorized": False,
        "provenance": provenance,
    }
    return {
        "schema_version": "persistent_binding_formal12_arm_result.v1",
        "arm": arm,
        "training_epochs": 12,
        "checkpoint_epoch": 12,
        "fixed_thresholds": {"birth": 0.5, "alive": 0.5, "end": 0.5},
        "gate_row": row,
        "reporting_accessed": False,
        "threshold_search": False,
        "effectiveness_claim_authorized": False,
        "raw_rgb_authorized": False,
    }


def test_formal12_gate_passes_only_a_complete_operational_pair():
    module = _module()

    result = module.evaluate_gate(_payload("fixed"), _payload("rematch"))

    assert result["technical_pass"] is True
    assert result["operational_pass"] is True
    assert result["formal_fixed_threshold_pass"] is True
    assert result["arm_actual_gpu_hours"] == 12.0
    assert result["finalizer_reserved_gpu_hours"] == 0.5
    assert result["budgeted_pair_gpu_hours"] == 12.5
    assert result["next_gate"] == "paired_multi_seed_feature_protocol"
    assert result["reporting_accessed"] is False
    assert result["raw_rgb_authorized"] is False


def test_formal12_gate_records_epoch12_silence_as_a_formal_failure():
    module = _module()

    result = module.evaluate_gate(
        _payload("fixed", predictions=0, ratio=0.0, recall=0.0),
        _payload("rematch"),
    )

    assert result["technical_pass"] is True
    assert result["operational_pass"] is False
    assert result["formal_fixed_threshold_pass"] is False
    assert any("epoch 12" in item for item in result["operational_failures"])
    assert result["next_gate"] == "model_optimization_on_fit_and_calibration_only"


def test_formal12_gate_rejects_mismatched_pair_provenance():
    module = _module()
    rematch = _payload("rematch")
    rematch["gate_row"]["provenance"]["census_sha256"] = "9" * 64

    with pytest.raises(ValueError, match="provenance.census_sha256"):
        module.evaluate_gate(_payload("fixed"), rematch)


def test_formal12_gate_counts_reserved_finalizer_gpu_budget():
    module = _module()

    result = module.evaluate_gate(
        _payload("fixed", gpu_hours=7.8),
        _payload("rematch", gpu_hours=7.8),
    )

    assert result["arm_actual_gpu_hours"] == pytest.approx(15.6)
    assert result["budgeted_pair_gpu_hours"] == pytest.approx(16.1)
    assert result["budget_pass"] is False
    assert result["technical_pass"] is False


def test_formal12_submitter_encodes_pairing_and_locked_split_contract():
    source = (
        ROOT / "tools/remote/submit_persistent_binding_formal12_n16r4.sh"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --exclude=$SBATCH_EXCLUDE" in source
    assert "#SBATCH --gpus=1" in source
    assert "#SBATCH --mem=" not in source
    assert "--evaluation-role calibration" in source
    assert "--evaluation-role reporting" not in source
    assert "3:2 6:5 9:8 12:11" in source
    assert "formal_fixed_threshold_gate_epoch" in source
    assert "tools/evaluate_persistent_binding_formal12.py" in source
    assert "--dependency=afterok:" in source
    assert "sbatch --parsable" in source
    assert "scancel" in source
    assert '"finalizer_reserved_gpu_hours": 0.5' in source
