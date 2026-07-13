from copy import deepcopy
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "causaltad"


def _load(name):
    return Config.fromfile(CONFIG_ROOT / name)


def _without_changed_axis(cfg):
    data = deepcopy(dict(cfg))
    data["model"] = dict(data["model"])
    data["model"].pop("trajectory_binding_mode")
    data.pop("work_dir")
    return data


def test_q2_bridge_configs_change_only_post_birth_loss_binding():
    rematch = _load("thumos_pes_q2_persist_rematch.py")
    fixed = _load("thumos_pes_q2_persist_fixed.py")

    assert _without_changed_axis(rematch) == _without_changed_axis(fixed)
    assert rematch.model.trajectory_binding_mode == "prefix_rematch_active_pool"
    assert fixed.model.trajectory_binding_mode == "fixed_birth_slot"


def test_q2_bridge_freezes_scientific_and_cost_contracts():
    cfg = _load("thumos_pes_q2_persist_fixed.py")

    assert cfg.route_stage == "q2_persistent_binding_one_factor"
    assert cfg.formal_training_ready is False
    assert cfg.visual_training_allowed is False
    assert cfg.raw_video_finetuning is False
    assert cfg.gpu_authorization == "BLOCKED_UNTIL_B0_AND_PROFILE"
    assert cfg.profile_contract.warmup_steps == 50
    assert cfg.profile_contract.measured_steps == 200
    assert cfg.profile_contract.b1_total_gpu_hour_cap == 2
    assert cfg.profile_contract.b2_total_gpu_hour_cap == 10

    assert cfg.experiment_contract.changed_axis == "post_birth_target_to_slot_loss_binding"
    assert cfg.experiment_contract.shared_first_crossing_birth is True
    assert cfg.experiment_contract.canonical_lifecycle_shared is True
    assert cfg.experiment_contract.runtime_state_contains_gt is False
    assert cfg.experiment_contract.rematch_is_supervision_control_only is True

    assert cfg.model.type == "PersistentTrajectoryOnlineDetector"
    assert cfg.model.birth_assignment_mode == "first_crossing_shared"
    assert cfg.model.canonical_supervision_lifecycle is True
    assert cfg.model.head.query_mode == "persistent"
    assert cfg.model.head.start_mode == "scalar"
    assert cfg.model.head.endpoint_mode == "binary"
    assert cfg.model.head.num_slots == 4
    assert "backbone" not in cfg.model

    assert cfg.dataset.train.subset_name == "training"
    assert cfg.dataset.val.subset_name == "training"
    assert cfg.dataset.test.subset_name == "training"
    assert cfg.dataset.train.allow_list.endswith("thumos_fit_core_160.txt")
    assert cfg.dataset.val.allow_list.endswith("thumos_calibration_40.txt")
    assert cfg.dataset.test.allow_list.endswith("thumos_calibration_40.txt")
    assert cfg.dataset.train.split_role == "fit_core"
    assert cfg.dataset.val.split_role == "calibration"
    assert cfg.dataset.test.split_role == "calibration"
    assert cfg.dataset.train.split_seed == 20260713
    assert cfg.dataset.train.split_manifest.endswith("thumos_development_split.json")
    assert cfg.evaluation.subset == "training"
    assert cfg.evaluation.allowed_videos.endswith("thumos_calibration_40.txt")
    assert cfg.reporting_contract.allow_during_training is False
    assert cfg.reporting_contract.locked_population.endswith(
        "thumos_reporting_locked_211.txt"
    )
    assert cfg.solver.train.streaming is True
    assert cfg.solver.train.batch_size == 1
    assert cfg.inference.load_from_raw_predictions is False
    assert "nms" not in cfg.post_processing
    assert cfg.post_processing.streaming_safe_emission is True
    assert cfg.post_processing.emission_ledger_filename.endswith(".jsonl")
    assert cfg.post_processing.emission_ledger_commitment_filename.endswith(
        ".commitment.json"
    )


def test_q2_configs_keep_pointer_hazard_and_raw_video_out_of_c1_gate():
    for name in (
        "thumos_pes_q2_persist_rematch.py",
        "thumos_pes_q2_persist_fixed.py",
    ):
        cfg = _load(name)
        assert cfg.dataset.train.type == "StreamingFeatureDataset"
        assert cfg.model.head.start_mode != "pointer"
        assert cfg.model.head.endpoint_mode != "hazard"
        assert cfg.experiment_contract.input == "fixed_cached_causal_features"
        assert cfg.experiment_contract.offline_nms is False


def test_streaming_evaluator_does_not_inject_rank_into_model_metadata():
    source = (ROOT / "opentad" / "cores" / "test_engine.py").read_text(
        encoding="utf-8"
    )

    assert 'meta["eval_rank"]' not in source
