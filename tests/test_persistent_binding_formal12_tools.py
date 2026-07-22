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


def _recovery_module():
    path = ROOT / "tools/stage_persistent_binding_training_recovery.py"
    spec = importlib.util.spec_from_file_location(
        "stage_persistent_binding_training_recovery",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replay_module():
    path = ROOT / "tools/verify_persistent_binding_calibration_replay.py"
    spec = importlib.util.spec_from_file_location(
        "verify_persistent_binding_calibration_replay",
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
    assert "stage_persistent_binding_training_recovery.py" in source
    assert '--quarantine-output "\\$QUARANTINE"' in source
    assert r'checkpoint="\$RECOVERY/checkpoint/epoch_\${zero}.pth"' in source
    assert "positive_duration_hard_transition_reserve_" in source
    assert "precalibration_quarantine.v2" in source


def test_formal12_calibration_replay_reuses_checkpoints_without_training():
    source = (
        ROOT
        / "tools/remote/submit_persistent_binding_formal12_calibration_replay_n16r4.sh"
    ).read_text(encoding="utf-8")

    assert "SOURCE_RUN" in source
    assert "SOURCE_COMMIT" in source
    assert "calibration_replay_only" in source
    assert "training_reused" in source
    assert "verify_persistent_binding_calibration_replay.py" in source
    assert "replay_event_payload_equality_required" in source
    assert "cumulative_source_plus_replay" in source
    assert "--evaluation-role calibration" in source
    assert "--evaluation-role reporting" not in source
    assert "tools/train.py" not in source
    assert "3:2 6:5 9:8 12:11" in source
    assert "tools/evaluate_persistent_binding_formal12.py" in source
    assert "--dependency=afterok:" in source
    assert "replay_silent_ledger_correction" in source


def _replay_row(event_id, sequence_id, emit_frame, *, score=0.5):
    return {
        "event_id": event_id,
        "video_id": "v1",
        "stream_key": "video=v1|stream=calibration",
        "sequence_id": sequence_id,
        "start_frame": 0,
        "end_frame": emit_frame,
        "source_frame": emit_frame,
        "emit_frame": emit_frame,
        "segment": [0, emit_frame],
        "label": "A",
        "score": score,
        "immutable": True,
    }


def test_calibration_replay_verifier_accepts_tied_frame_reorder_only(tmp_path):
    module = _replay_module()
    row_0 = _replay_row("e0", 0, 8, score=0.4)
    row_1 = _replay_row("e1", 1, 8, score=0.6)
    row_2 = _replay_row("e2", 2, 16)
    source = tmp_path / "source.json"
    replay = tmp_path / "replay.json"
    source.write_text(
        __import__("json").dumps({"results": {"v1": [row_1, row_0, row_2]}}),
        encoding="utf-8",
    )
    replay.write_text(
        __import__("json").dumps({"results": {"v1": [row_0, row_1, row_2]}}),
        encoding="utf-8",
    )

    result = module.verify_replay(source_ledger=source, replay_ledger=replay)

    assert result["passed"] is True
    assert result["reorder_only"] is True
    assert result["event_payloads_identical"] is True
    assert result["source_tied_sequence_violations"] == 1
    assert not any(result["replay_violation_counts"].values())


def test_calibration_replay_verifier_rejects_payload_change(tmp_path):
    module = _replay_module()
    source = tmp_path / "source.json"
    replay = tmp_path / "replay.json"
    source.write_text(
        __import__("json").dumps(
            {"results": {"v1": [_replay_row("e0", 0, 8, score=0.4)]}}
        ),
        encoding="utf-8",
    )
    replay.write_text(
        __import__("json").dumps(
            {"results": {"v1": [_replay_row("e0", 0, 8, score=0.5)]}}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed an emitted event payload"):
        module.verify_replay(source_ledger=source, replay_ledger=replay)


def test_formal12_training_recovery_is_validated_and_atomic(tmp_path):
    module = _recovery_module()
    train_root = tmp_path / "train"
    checkpoint_root = train_root / "checkpoint"
    checkpoint_root.mkdir(parents=True)
    audit = {
        "schema_version": "persistent_binding_training_audit.v1",
        "fit_only": True,
        "screening_only": False,
        "route_stage": "persistent_binding_feature_multi_epoch_calibration",
        "epochs": [{"epoch": epoch} for epoch in range(12)],
        "totals": {
            "expected_updates": 24120,
            "successful_updates": 24120,
            "scheduler_steps": 24120,
            "skipped_updates": 0,
            "gt_supervision_exhaustions": 0,
            "gt_birth_runtime_entry_free_collisions": 0,
        },
    }
    (train_root / "training_audit.json").write_text(
        __import__("json").dumps(audit), encoding="utf-8"
    )
    for _, zero in module.CHECKPOINTS:
        (checkpoint_root / f"epoch_{zero}.pth").write_bytes(
            f"checkpoint-{zero}".encode()
        )
    config = tmp_path / "config.py"
    config.write_text("formal_training_ready = True\n", encoding="utf-8")
    output = tmp_path / "recovery"

    manifest = module.stage_recovery(
        train_root=train_root,
        config=config,
        output=output,
        arm="fixed",
        commit="a" * 40,
        seed=705,
    )

    assert output.is_dir()
    assert not list(tmp_path.glob(".recovery.*.tmp"))
    assert manifest["schema_version"] == "persistent_binding_training_recovery.v1"
    assert manifest["checkpoint_epochs"] == [3, 6, 9, 12]
    assert manifest["successful_updates"] == 24120
    assert manifest["reporting_accessed"] is False
    assert len(manifest["checkpoints"]) == 4
    assert all(len(row["sha256"]) == 64 for row in manifest["checkpoints"])
    assert (output / "training_audit.json").is_file()
    assert (output / "checkpoint/epoch_11.pth").is_file()


def test_formal12_training_recovery_rejects_incomplete_updates():
    module = _recovery_module()
    audit = {
        "schema_version": "persistent_binding_training_audit.v1",
        "fit_only": True,
        "screening_only": False,
        "route_stage": "persistent_binding_feature_multi_epoch_calibration",
        "epochs": [{"epoch": epoch} for epoch in range(12)],
        "totals": {
            "expected_updates": 24120,
            "successful_updates": 24119,
            "scheduler_steps": 24120,
            "skipped_updates": 0,
            "gt_supervision_exhaustions": 0,
            "gt_birth_runtime_entry_free_collisions": 0,
        },
    }

    with pytest.raises(ValueError, match="successful_updates"):
        module.validate_training_audit(audit)


def test_formal12_failed_audit_is_atomically_quarantined(tmp_path):
    module = _recovery_module()
    train_root = tmp_path / "train"
    checkpoint_root = train_root / "checkpoint"
    checkpoint_root.mkdir(parents=True)
    audit = {
        "schema_version": "persistent_binding_training_audit.v1",
        "fit_only": True,
        "screening_only": False,
        "route_stage": "persistent_binding_feature_multi_epoch_calibration",
        "epochs": [{"epoch": epoch} for epoch in range(12)],
        "totals": {
            "expected_updates": 24120,
            "successful_updates": 24120,
            "scheduler_steps": 24120,
            "skipped_updates": 0,
            "gt_supervision_exhaustions": 0,
            "gt_birth_runtime_entry_free_collisions": 1,
        },
    }
    (train_root / "training_audit.json").write_text(
        __import__("json").dumps(audit),
        encoding="utf-8",
    )
    for _, zero in module.CHECKPOINTS:
        (checkpoint_root / f"epoch_{zero}.pth").write_bytes(
            f"checkpoint-{zero}".encode()
        )
    config = tmp_path / "config.py"
    config.write_text("formal_training_ready = True\n", encoding="utf-8")
    recovery = tmp_path / "recovery"
    quarantine = tmp_path / "quarantine"

    with pytest.raises(
        ValueError,
        match="gt_birth_runtime_entry_free_collisions",
    ):
        module.stage_recovery(
            train_root=train_root,
            config=config,
            output=recovery,
            arm="fixed",
            commit="a" * 40,
            seed=705,
            quarantine=quarantine,
        )

    assert not recovery.exists()
    assert quarantine.is_dir()
    assert not list(tmp_path.glob(".quarantine.*.tmp"))
    manifest = __import__("json").loads(
        (quarantine / "quarantine_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema_version"] == (
        "persistent_binding_training_quarantine.v1"
    )
    assert manifest["calibration_authorized"] is False
    assert manifest["recovery_manifest"] is False
    assert len(manifest["checkpoints"]) == 4
    assert all(len(row["sha256"]) == 64 for row in manifest["checkpoints"])
    assert (quarantine / "training_audit.json").is_file()
    assert (quarantine / "checkpoint/epoch_11.pth").is_file()
