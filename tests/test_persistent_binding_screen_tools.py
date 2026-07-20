import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(arm, *, ratio=1.0, recall=0.5, identity_shift=0.0):
    binding_mode = {
        "fixed": "fixed_birth_slot",
        "rematch": "prefix_rematch_active_pool",
    }[arm]
    provenance = {
        "code_commit": "a" * 40,
        "annotation_sha256": "b" * 64,
        "calibration_manifest_sha256": "c" * 64,
        "fit_manifest_sha256": "d" * 64,
        "reporting_manifest_sha256": "e" * 64,
        "checkpoint_sha256": ("1" if arm == "fixed" else "2") * 64,
        "census_sha256": "3" * 64,
        "config_sha256": ("4" if arm == "fixed" else "5") * 64,
        "emission_ledger_sha256": ("6" if arm == "fixed" else "7") * 64,
        "profile_gate_sha256": "8" * 64,
        "smoke_gate_sha256": "9" * 64,
        "screening_contract_sha256": "f" * 64,
    }
    return {
        "schema_version": "persistent_binding_screen_result.v1",
        "arm": arm,
        "binding_mode": binding_mode,
        "seed": 705,
        "reporting_accessed": False,
        "effectiveness_claim_authorized": False,
        "raw_rgb_authorized": False,
        "gate_row": {
            "arm": arm,
            "binding_mode": binding_mode,
            "seed": 705,
            "screen_epochs": 1,
            "metric_schema": "standard_ontad_map.v1",
            "instance_metric_schema": "online_instance_metrics.v2",
            "map_unit": "percentage_points",
            "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
            "average_map_pct": 10.0,
            "duplicate_rate": 0.2 + identity_shift,
            "fragmentation_rate": 0.1 + identity_shift,
            "prediction_gt_ratio": ratio,
            "recall_tiou_0p3": recall,
            "protocol_violations": 0,
            "gt_supervision_exhaustions": 0,
            "gt_birth_runtime_entry_free_collisions": 0,
            "committed_predictions": 0 if ratio == 0 else 100,
            "expected_updates": 2010,
            "successful_updates": 2010,
            "scheduler_steps": 2010,
            "skipped_updates": 0,
            "reporting_accessed": False,
            "effectiveness_claim_authorized": False,
            "raw_rgb_authorized": False,
            "allocated_gpu_hours": 0.4,
            "projected_pair_gpu_hours": 1.7,
            "provenance": provenance,
        },
    }


def _resource(gpu_hours=0.9):
    return {
        "schema_version": "persistent_binding_resource.v1",
        "scope": "pair",
        "allocated_gpu_hours": gpu_hours,
        "code_commit": "a" * 40,
    }


def test_seed705_screen_pass_is_technical_not_an_effectiveness_claim():
    evaluator = _module(
        "tools/evaluate_persistent_binding_screen.py",
        "evaluate_persistent_binding_screen",
    )

    result = evaluator.evaluate_screen(
        _payload("fixed", identity_shift=0.1),
        _payload("rematch"),
        _resource(),
    )

    assert result["screen_pass"] is True
    assert result["effectiveness_claim_authorized"] is False
    assert result["raw_rgb_authorized"] is False
    assert result["reporting_accessed"] is False
    assert result["directional_diagnostics"][
        "identity_error_relative_reduction"
    ] < 0


def test_seed705_screen_rejects_silent_or_explosive_outputs():
    evaluator = _module(
        "tools/evaluate_persistent_binding_screen.py",
        "evaluate_persistent_binding_screen_failures",
    )

    silent = evaluator.evaluate_screen(
        _payload("fixed", ratio=0.0, recall=0.0),
        _payload("rematch"),
        _resource(),
    )
    explosive = evaluator.evaluate_screen(
        _payload("fixed", ratio=9.0),
        _payload("rematch"),
        _resource(),
    )

    assert silent["screen_pass"] is False
    assert any("no committed" in item for item in silent["technical_failures"])
    assert explosive["screen_pass"] is False
    assert any(
        "prediction_gt_ratio" in item
        for item in explosive["technical_failures"]
    )


def test_seed705_screen_rejects_reporting_access_or_budget_overrun():
    evaluator = _module(
        "tools/evaluate_persistent_binding_screen.py",
        "evaluate_persistent_binding_screen_isolation",
    )
    leaked = _payload("fixed")
    leaked["reporting_accessed"] = True

    with pytest.raises(ValueError, match="reporting"):
        evaluator.evaluate_screen(
            leaked,
            _payload("rematch"),
            _resource(),
        )

    over_budget = evaluator.evaluate_screen(
        _payload("fixed"),
        _payload("rematch"),
        _resource(gpu_hours=2.1),
    )
    assert over_budget["screen_pass"] is False
    assert over_budget["budget_pass"] is False

    incomplete = _payload("fixed")
    del incomplete["gate_row"]["provenance"]["smoke_gate_sha256"]
    with pytest.raises(ValueError, match="smoke_gate_sha256"):
        evaluator.evaluate_screen(
            incomplete,
            _payload("rematch"),
            _resource(),
        )


def test_resource_report_counts_allocated_gpu_time_and_requires_clean_repo(
    tmp_path,
):
    resource = _module(
        "tools/build_persistent_binding_resource_report.py",
        "build_persistent_binding_resource_report",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    report = resource.build_resource_report(
        repo=repo,
        scope="pair",
        started_unix=100,
        ended_unix=3700,
        gpu_count=1,
        gpu_name="RTX 4090",
        slurm_job_id="1",
    )

    assert report["elapsed_seconds"] == 3600
    assert report["allocated_gpu_hours"] == 1.0
    assert len(report["code_commit"]) == 40


def test_screen_sources_forbid_reporting_execution_and_formal_claims():
    builder = (
        ROOT / "tools/build_persistent_binding_screen_result.py"
    ).read_text(encoding="utf-8")
    trainer = (ROOT / "tools/train.py").read_text(encoding="utf-8")
    submitter = (
        ROOT / "tools/remote/submit_persistent_binding_screen_n16r4.sh"
    ).read_text(encoding="utf-8")

    assert "calibration_manifest" in builder
    assert "reporting_manifest" in builder
    assert "reporting_accessed" in builder
    assert "effectiveness_claim_authorized" in builder
    assert "raw_rgb_authorized" in builder
    assert "reporting_receipt" not in builder
    assert "--allow-unready-screen" in trainer
    assert "screening_training_ready" in trainer
    assert "#SBATCH --gres=gpu:1" in submitter
    assert "tools/train.py" in submitter
    assert "--evaluation-role calibration" in submitter
    assert "--evaluation-role reporting" not in submitter
    assert "seed705_screen_profile_gate.json" in submitter
    assert "screen_gate.json" in submitter
    assert "sbatch" in submitter
