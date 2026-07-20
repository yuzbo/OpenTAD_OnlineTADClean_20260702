import importlib.util
import json
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]


def _load_module(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_smoke_config_is_one_epoch_feature_only_and_not_formally_unlocked():
    cfg = Config.fromfile(ROOT / "configs/causaltad/thumos_persistent_binding_smoke.py")

    assert cfg.route_stage == "persistent_binding_feature_slurm_smoke"
    assert cfg.smoke_only is True
    assert cfg.formal_training_ready is False
    assert cfg.raw_video_finetuning is False
    assert cfg.model.trajectory_binding_mode == "fixed_birth_slot"
    assert cfg.workflow.end_epoch == 1
    assert cfg.workflow.checkpoint_interval == 1
    assert cfg.workflow.val_eval_interval == 1
    assert cfg.workflow.fit_only is False
    assert cfg.workflow.fail_on_nonfinite is True
    assert cfg.solver.amp is False
    assert cfg.scheduler.warmup_start_lr == cfg.optimizer.lr
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    assert cfg.post_processing.emission_ledger_filename.endswith(".json")
    assert cfg.post_processing.streaming_safe_emission is True


def test_manifest_selector_uses_shortest_annotated_real_video_per_frozen_split(tmp_path):
    tool = _load_module(
        "tools/prepare_persistent_binding_smoke_manifests.py",
        "prepare_persistent_binding_smoke_manifests",
    )
    annotation = tmp_path / "annotation.json"
    cache = tmp_path / "manifest.json"
    fit = tmp_path / "fit.txt"
    calibration = tmp_path / "calibration.txt"
    reporting = tmp_path / "reporting.txt"
    output = tmp_path / "output"

    database = {
        "train_long": {
            "subset": "training",
            "annotations": [{"label": "A", "segment": [1, 2]}],
        },
        "train_short": {
            "subset": "training",
            "annotations": [{"label": "A", "segment": [1, 2]}],
        },
        "cal_short": {
            "subset": "training",
            "annotations": [{"label": "A", "segment": [1, 2]}],
        },
        "test_empty": {"subset": "validation", "annotations": []},
        "test_valid": {
            "subset": "validation",
            "annotations": [{"label": "A", "segment": [1, 2]}],
        },
    }
    videos = {
        "train_long": {"num_tokens": 20},
        "train_short": {"num_tokens": 5},
        "cal_short": {"num_tokens": 6},
        "test_empty": {"num_tokens": 3},
        "test_valid": {"num_tokens": 7},
    }
    _write_json(annotation, {"database": database})
    _write_json(cache, {"videos": videos})
    fit.write_text("train_long\ntrain_short\n", encoding="utf-8")
    calibration.write_text("cal_short\n", encoding="utf-8")
    reporting.write_text("test_empty\ntest_valid\n", encoding="utf-8")

    payload = tool.prepare_manifests(
        annotation,
        cache,
        fit,
        calibration,
        reporting,
        output,
    )

    assert payload["selections"]["train"]["video_id"] == "train_short"
    assert payload["selections"]["val"]["video_id"] == "cal_short"
    assert payload["selections"]["test"]["video_id"] == "test_valid"
    assert (output / "train.txt").read_text(encoding="utf-8") == "train_short\n"
    assert json.loads((output / "selection.json").read_text(encoding="utf-8")) == payload


def test_smoke_runner_and_verifier_enforce_real_updates_causality_and_reload():
    smoke = (ROOT / "tools/smoke_persistent_binding.py").read_text(encoding="utf-8")
    verify = (ROOT / "tools/verify_persistent_binding_smoke.py").read_text(
        encoding="utf-8"
    )

    assert "snapshot_trainable_parameters" in smoke
    assert "audit_training_update" in smoke
    assert 'required_module_prefixes = ("head",)' in smoke
    assert "dropped_gt_birth_targets" in smoke
    assert "load_from_raw_predictions is forbidden" in smoke
    assert "summarize_emission_ledger" in smoke

    assert "checkpoint reload changed deterministic streaming emissions" in verify
    assert "checkpoint contains no optimizer update state" in verify
    assert "checkpoint is identical to seeded initialization" in verify
    assert "validate_emission_ledger_summary" in verify
    assert "build_evaluator" in verify
    assert "smoke model produced no final intervals" in verify
    assert "raw_video_finetuning" in verify


def test_n16r4_submitter_is_clean_checkout_slurm_only_and_runs_standard_entrypoints():
    submit = (
        ROOT / "tools/remote/submit_persistent_binding_n16r4.sh"
    ).read_text(encoding="utf-8")
    check = (
        ROOT / "tools/remote/check_persistent_binding_n16r4.sh"
    ).read_text(encoding="utf-8")

    assert "status --porcelain" in submit
    assert "branch --show-current" in submit
    assert "codex/ontad-science-fixed-rematch" in submit
    assert "EXPECTED_COMMIT must be a full lowercase Git SHA" in submit
    assert "Detached deployment requires EXPECTED_COMMIT" in submit
    assert "CURRENT_COMMIT" in submit
    assert "#SBATCH -p gpu" in submit
    assert "#SBATCH --gres=gpu:1" in submit
    assert "sbatch" in submit
    assert "torchrun" in submit
    assert "tools/train.py" in submit
    assert "tools/test.py" in submit
    assert "tools/smoke_persistent_binding.py" in submit
    assert "tools/verify_persistent_binding_smoke.py" in submit
    assert "tools/census_persistent_binding.py" in submit
    assert "--allow-unready-smoke" in submit
    assert "persistent_binding_emissions.json" in submit
    assert "training_audit.json" in submit
    assert "load_from_raw_predictions=True" not in submit
    assert 'cat >> "$SCRIPT_PATH" <<\'SBATCH\'' in submit

    assert "squeue" in check
    assert "sacct" in check
    assert "gate_summary.json" in check
    assert "tail -n 120" in check


def test_all_persistent_binding_submitters_guard_detached_exact_commits():
    names = (
        "submit_persistent_binding_n16r4.sh",
        "submit_persistent_binding_profile_n16r4.sh",
        "submit_persistent_binding_screen_n16r4.sh",
        "submit_persistent_binding_score_diagnosis_n16r4.sh",
        "submit_persistent_binding_optimization_pilot_n16r4.sh",
    )

    for name in names:
        source = (ROOT / "tools" / "remote" / name).read_text(
            encoding="utf-8"
        )
        assert "status --porcelain" in source
        assert "EXPECTED_COMMIT=${EXPECTED_COMMIT:-}" in source
        assert "EXPECTED_COMMIT must be a full lowercase Git SHA" in source
        assert "Deployment checkout does not match EXPECTED_COMMIT" in source
        assert "Detached deployment requires EXPECTED_COMMIT" in source
        assert "COMMIT_SHA=$CURRENT_COMMIT" in source


def test_model_optimization_submitter_deploys_three_feature_only_variants():
    submit = (
        ROOT
        / "tools/remote/submit_persistent_binding_optimization_pilot_n16r4.sh"
    ).read_text(encoding="utf-8")
    check = (
        ROOT
        / "tools/remote/check_persistent_binding_optimization_pilot_n16r4.sh"
    ).read_text(encoding="utf-8")

    for variant in ("sw", "margin", "transport"):
        assert f"{variant})" in submit
    assert "thumos_persistent_binding_opt_sw_fixed.py" in submit
    assert "thumos_persistent_binding_opt_margin_fixed.py" in submit
    assert "thumos_persistent_binding_opt_transport_fixed.py" in submit
    assert "--allow-unready-screen" in submit
    assert "profile_persistent_binding.py" in submit
    assert "evaluate_persistent_binding_profile.py" in submit
    assert "--epochs 1" in submit
    assert "same_commit_profile_required" in submit
    assert "soft_sinkhorn_temperature_0p5_v1" in submit
    assert "--evaluation-role calibration" in submit
    assert "diagnose_persistent_binding_scores.py" in submit
    assert "evaluate_persistent_binding_optimization_activation.py" in submit
    assert "optimization_activation.json" in submit
    assert "reporting_accessed" in submit
    assert '"raw_rgb_authorized": False' in submit
    assert "#SBATCH --gres=gpu:1" in submit
    assert "sbatch" in submit
    assert "squeue" in check
    assert "sacct" in check
    assert "fixed_score_diagnosis.json" in check
