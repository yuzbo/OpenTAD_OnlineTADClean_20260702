from copy import deepcopy
from pathlib import Path

from mmengine.config import Config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "causaltad"


def _load(name):
    return Config.fromfile(CONFIG_DIR / name)


def _normalized(cfg):
    value = deepcopy(cfg.to_dict())
    value["model"].pop("trajectory_binding_mode")
    value.pop("work_dir")
    return value


def test_fixed_and_rematch_are_a_single_variable_comparison():
    fixed = _load("thumos_persistent_binding_fixed.py")
    rematch = _load("thumos_persistent_binding_rematch.py")

    assert fixed.model.trajectory_binding_mode == "fixed_birth_slot"
    assert rematch.model.trajectory_binding_mode == "prefix_rematch_active_pool"
    assert _normalized(fixed) == _normalized(rematch)


def test_feature_route_is_strictly_causal_and_raw_rgb_remains_blocked():
    cfg = _load("thumos_persistent_binding_fixed.py")

    assert cfg.experiment_contract.input == "fixed_cached_causal_features"
    assert cfg.experiment_contract.runtime_state_contains_gt is False
    assert cfg.experiment_contract.future_endpoint_prediction is False
    assert cfg.experiment_contract.offline_nms is False
    assert cfg.experiment_contract.raw_video_joint_training is False
    assert cfg.raw_video_finetuning is False
    assert cfg.solver.amp is False
    assert cfg.workflow.fail_on_nonfinite is True
    assert cfg.visual_training_allowed is False
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    assert cfg.dataset.train.strict_causal_control is True
    assert cfg.dataset.val.strict_causal_control is True
    assert cfg.dataset.test.strict_causal_control is True


def test_route_uses_candidate_recycle_without_capacity_holding_refractory():
    cfg = _load("thumos_persistent_binding_fixed.py")
    head = cfg.model.head

    assert head.lifecycle_mode == "candidate_recycle"
    assert head.candidate_confirmation_steps == 1
    assert head.max_births_per_step == 2
    assert head.refractory_steps == 0
    assert head.num_slots == 4


def test_route_sources_do_not_reference_historical_route_labels():
    paths = (
        CONFIG_DIR / "thumos_persistent_binding_base.py",
        CONFIG_DIR / "thumos_persistent_binding_fixed.py",
        CONFIG_DIR / "thumos_persistent_binding_rematch.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)

    assert "full_petal" not in source
    assert "pes_q2" not in source
