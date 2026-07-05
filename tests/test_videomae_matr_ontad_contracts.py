from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_matr_head_source_declares_online_start_end_memory_contract():
    source = read("opentad/models/dense_heads/matr_head.py")

    assert "class MATRHead" in source
    assert "@HEADS.register_module()" in source
    assert "start_head" in source
    assert "end_head" in source
    assert "actionness_head" in source
    assert "emit_threshold" in source
    assert "online = True" in source
    assert "get_valid_proposals_scores" in source
    assert "_append_stream_memory" in source
    assert "start_score" in source
    assert "end_score" in source
    assert "clamp_end_to_current" in source


def test_online_videomae_adapter_source_declares_causal_frame_adapter_contract():
    source = read("opentad/models/backbones/online_videomae_adapter.py")

    assert "class CausalTemporalAdapter" in source
    assert "class OnlineVideoMAEAdapter" in source
    assert "@MODELS.register_module()" in source
    assert "input_format = \"raw_frames\"" in source
    assert "online = True" in source
    assert "left_pad" in source
    assert "freeze_backbone" in source
    assert "use_stub_backbone" in source
    assert "get_optim_groups" in source
    assert "strict_online" in source


def test_strict_causal_projection_source_has_no_required_future_path():
    source = read("opentad/models/projections/causal_proj.py")

    assert "strict_causal=False" in source
    assert "strict_causal=True" in source
    assert "CausalConvModule" in source
    assert "if not self.strict_causal" in source
    assert "_causal_downsample" in source


def test_videomae_matr_config_declares_raw_frame_online_protocol():
    Config = __import__("mmengine.config", fromlist=["Config"]).Config

    cfg = Config.fromfile(str(ROOT / "configs/causaltad/thumos_videomae_adapter_matr_ontad.py"))
    source = read("configs/causaltad/thumos_videomae_adapter_matr_ontad.py")

    assert "thumos_i3d.py" not in source
    assert "LoadFeats" not in source
    assert cfg.model.type == "VideoMambaSuite"
    assert cfg.model.backbone.type == "OnlineVideoMAEAdapter"
    assert cfg.model.backbone.input_format == "raw_frames"
    assert cfg.model.backbone.online is True
    assert cfg.model.backbone.strict_online is True
    assert cfg.model.projection.strict_causal is True
    assert cfg.model.rpn_head.type == "MATRHead"
    assert cfg.model.rpn_head.online is True
    assert cfg.model.rpn_head.emit_threshold >= 0
    assert cfg.dataset.train.input_format == "raw_frames"
    assert not bool(cfg.inference.load_from_raw_predictions)
    assert not bool(cfg.inference.save_raw_prediction)


def test_optimizer_source_keeps_trainable_adapter_params_when_backbone_frozen():
    source = read("opentad/cores/optimizer.py")

    assert "get_optim_groups" in source
    assert "Train frozen-backbone adapters" in source


def test_raw_frame_dataset_and_loader_are_registered_for_training_route():
    dataset_init = read("opentad/datasets/__init__.py")
    transform_init = read("opentad/datasets/transforms/__init__.py")
    raw_dataset = read("opentad/datasets/raw_frame.py")
    loading = read("opentad/datasets/transforms/loading.py")

    assert "FrameWindowDataset" in dataset_init
    assert "LoadRawFrames" in transform_init
    assert "@DATASETS.register_module()" in raw_dataset
    assert "class FrameWindowDataset" in raw_dataset
    assert "gt_segments" in raw_dataset
    assert "frame_inds" in raw_dataset
    assert "@PIPELINES.register_module()" in loading
    assert "class LoadRawFrames" in loading
    assert "results[\"frames\"]" in loading


def test_videomae_matr_config_has_trainable_runtime_fields_and_batch_keys():
    Config = __import__("mmengine.config", fromlist=["Config"]).Config

    cfg = Config.fromfile(str(ROOT / "configs/causaltad/thumos_videomae_adapter_matr_ontad.py"))

    for field in ("solver", "scheduler", "workflow", "evaluation"):
        assert field in cfg

    assert cfg.dataset.train.type == "FrameWindowDataset"
    assert cfg.dataset.train.subset_name == "training"
    assert cfg.dataset.val.subset_name == "validation"
    assert cfg.dataset.test.test_mode is True

    train_collect = cfg.dataset.train.pipeline[-1]
    val_collect = cfg.dataset.val.pipeline[-1]
    test_collect = cfg.dataset.test.pipeline[-1]
    assert train_collect.type == "Collect"
    assert train_collect.inputs == "frames"
    assert train_collect["keys"] == ["masks", "gt_segments", "gt_labels"]
    assert val_collect["keys"] == ["masks", "gt_segments", "gt_labels"]
    assert test_collect["keys"] == ["masks"]


def test_matr_head_has_supervised_online_branches_and_video_aware_state_reset():
    head_source = read("opentad/models/dense_heads/matr_head.py")
    detector_source = read("opentad/models/detectors/single_stage.py")

    assert "boundary_loss_weight" in head_source
    assert "emit_loss_weight" in head_source
    assert "_online_branch_losses" in head_source
    assert "use_boundary_scores" in head_source
    assert "self._stream_video_names" in head_source
    assert "_maybe_reset_stream_state" in head_source
    assert "video_name" in head_source
    assert "metas=metas" in detector_source


def test_raw_frame_route_has_no_missing_pseudo_boundary_import_dependency():
    pseudo_boundary = ROOT / "opentad/datasets/transforms/pseudo_boundary.py"

    assert pseudo_boundary.exists()
    source = pseudo_boundary.read_text(encoding="utf-8")
    assert "def load_boundary_scores" in source
    assert "def slice_global_scores_for_window" in source
    assert "def select_pseudo_boundary_hybrid_positions" in source
    assert "def select_pseudo_boundary_snap_positions" in source


def test_matr_stream_state_reset_uses_window_start_frame_not_only_video_name():
    head_source = read("opentad/models/dense_heads/matr_head.py")

    assert "window_start_frame" in head_source
    assert "_stream_window_start_frames" in head_source


def test_matr_stream_memory_reuses_only_contiguous_non_overlapping_windows():
    head_source = read("opentad/models/dense_heads/matr_head.py")
    raw_dataset = read("opentad/datasets/raw_frame.py")
    formatting = read("opentad/datasets/transforms/formatting.py")

    assert "_stream_window_end_frames" in head_source
    assert "_stream_window_end_by_key" in head_source
    assert "_clear_stream_memory_for_key" in head_source
    assert "current_start == previous_end" in head_source
    assert "current_start < previous_end" in head_source
    assert "window_end_frame" in raw_dataset
    assert "window_end_frame" in formatting


def test_runtime_smoke_does_not_skip_opentad_import_or_build_errors():
    source = read("tests/test_videomae_matr_runtime_smoke.py")

    assert "pytest.skip(f\"OpenTAD" not in source
    assert "pytest.fail(f\"OpenTAD" in source


def test_raw_videomae_route_marks_stub_backbone_as_contract_only():
    Config = __import__("mmengine.config", fromlist=["Config"]).Config

    cfg = Config.fromfile(str(ROOT / "configs/causaltad/thumos_videomae_adapter_matr_ontad.py"))
    source = read("configs/causaltad/thumos_videomae_adapter_matr_ontad.py")

    assert cfg.model.backbone.use_stub_backbone is True
    assert cfg.model.backbone.stub_backbone_contract_only is True
    assert cfg.formal_training_ready is False
    assert "stub_backbone_contract_only" in source
    assert "formal_training_ready = False" in source


def test_causal_adapter_does_not_start_with_double_zero_branch():
    source = read("opentad/models/backbones/online_videomae_adapter.py")

    assert "init_scale=0.0" not in source
