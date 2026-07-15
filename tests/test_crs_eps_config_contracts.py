from pathlib import Path
from copy import deepcopy

from mmengine import Config
import pytest

from opentad.utils.full_petal_launch import FullPetalLaunchError, _profile_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "causaltad"


def _config(name):
    return Config.fromfile(str(CONFIG_ROOT / name))


def test_crs_eps_fixed_and_rematch_share_the_frozen_training_protocol():
    fixed = _config("thumos_pes_q2_crs_eps_fixed.py")
    rematch = _config("thumos_pes_q2_crs_eps_rematch.py")

    for cfg in (fixed, rematch):
        assert cfg.route_stage == "q2_crs_eps_hh_ipw_implementation_gate"
        assert cfg.dataset.train.type == "CrsEpsFeatureDataset"
        assert cfg.dataset.train.draws_per_video == 4
        assert cfg.dataset.test.type == "StreamingFeatureDataset"
        assert cfg.dataset.test.chunk_size == 1
        assert cfg.dataset.val.chunk_size == 64
        assert cfg.profile_contract.step_unit == "video_group_optimizer_event"
        assert cfg.profile_contract.warmup_steps == 8
        assert cfg.profile_contract.measured_steps == 32
        assert cfg.profile_contract.required_workload_denominators == [
            "temporal_forward_tokens",
            "temporal_backward_tokens",
            "replay_tokens",
            "supervised_exposures",
            "unique_supervised_bins",
            "effective_sample_size",
            "visual_forward_frames",
            "visual_backward_frames",
            "data_wait_seconds",
            "control_unroll_seconds",
            "wall_seconds",
            "peak_memory_bytes",
            "gpu_hours",
        ]
        assert cfg.gpu_authorization == "BLOCKED_UNTIL_B0_AND_PROFILE"

    assert fixed.model.trajectory_binding_mode == "fixed_birth_slot"
    assert rematch.model.trajectory_binding_mode == "prefix_rematch_active_pool"


def test_crs_eps_contract_keeps_cached_features_as_an_audited_cost_surrogate():
    cfg = _config("thumos_pes_q2_crs_eps_fixed.py")

    assert cfg.dataset.train.type == "CrsEpsFeatureDataset"
    assert cfg.dataset.train.cache_manifest.endswith("manifest.json")
    assert "backbone" not in cfg.model
    assert "projection" not in cfg.model
    assert cfg.experiment_contract.dynamic_replay_is_state_surrogate is True
    assert cfg.experiment_contract.exact_ipw_does_not_correct_state_bias is True
    assert cfg.crs_eps_contract.final_evaluation == (
        "complete_chronological_one_token_streaming"
    )


def test_crs_eps_launch_rejects_drifted_cost_denominators():
    cfg = _config("thumos_pes_q2_crs_eps_fixed.py")
    assert _profile_contract(cfg) == (8, 32, 1)
    drifted = deepcopy(cfg)
    drifted.profile_contract.required_workload_denominators.pop()

    with pytest.raises(FullPetalLaunchError, match="workload denominators"):
        _profile_contract(drifted)
