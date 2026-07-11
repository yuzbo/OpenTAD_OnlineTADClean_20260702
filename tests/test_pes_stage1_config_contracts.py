from copy import deepcopy
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "causaltad"


def _load(name):
    return Config.fromfile(CONFIG_ROOT / name)


def _mechanism_free_model(cfg):
    model = deepcopy(dict(cfg.model))
    model.pop("assignment_mode")
    head = dict(model["head"])
    for key in ("query_mode", "start_mode", "endpoint_mode"):
        head.pop(key)
    model["head"] = head
    return model


def test_stage1_variants_change_only_registered_mechanism_fields():
    fresh = _load("thumos_pes_stage1_fresh.py")
    track = _load("thumos_pes_stage1_trackformer.py")
    persistent = _load("thumos_pes_stage1_persistent.py")
    configs = (fresh, track, persistent)

    for cfg in configs[1:]:
        assert cfg.dataset == configs[0].dataset
        assert cfg.optimizer == configs[0].optimizer
        assert cfg.scheduler == configs[0].scheduler
        assert cfg.solver == configs[0].solver
        assert cfg.workflow == configs[0].workflow
        assert cfg.evaluation == configs[0].evaluation
        assert cfg.post_processing == configs[0].post_processing
        assert cfg.pilot_seeds == configs[0].pilot_seeds
        assert _mechanism_free_model(cfg) == _mechanism_free_model(configs[0])

    assert fresh.model.assignment_mode == "per_step"
    assert fresh.model.head.query_mode == "fresh"
    assert fresh.model.head.start_mode == "scalar"
    assert fresh.model.head.endpoint_mode == "binary"

    assert track.model.assignment_mode == "prefix"
    assert track.model.head.query_mode == "persistent"
    assert track.model.head.start_mode == "scalar"
    assert track.model.head.endpoint_mode == "binary"

    assert persistent.model.assignment_mode == "prefix"
    assert persistent.model.head.query_mode == "persistent"
    assert persistent.model.head.start_mode == "pointer"
    assert persistent.model.head.endpoint_mode == "hazard"


def test_stage1_contract_freezes_visual_training_nms_and_raw_prediction_shortcuts():
    cfg = _load("thumos_pes_stage1_persistent.py")

    assert cfg.route_stage == "pes_stage1_matched_feature_kill_test"
    assert cfg.formal_training_ready is False
    assert cfg.visual_training_allowed is False
    assert cfg.hard_budget_gpu_hours == 10
    assert cfg.pilot_seeds == [705, 706, 707]
    assert cfg.dataset.train.type == "StreamingFeatureDataset"
    assert cfg.dataset.train.data_path == cfg.feature_cache_path
    assert cfg.dataset.train.cache_manifest == cfg.feature_cache_manifest
    assert cfg.dataset.test.test_mode is True
    assert cfg.model.type == "PersistentEventSetOnlineDetector"
    assert cfg.model.head.num_slots == 4
    assert "backbone" not in cfg.model
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    assert "nms" not in cfg.post_processing
    assert cfg.post_processing.streaming_safe_emission is True
    assert cfg.solver.train.streaming is True
    assert cfg.solver.train.batch_size == 1
    assert cfg.solver.train.stream_batch_size == 1
