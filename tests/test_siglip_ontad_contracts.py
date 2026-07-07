from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_siglip_online_encoder_source_declares_real_frozen_frame_route():
    encoder_path = ROOT / "opentad/models/backbones/online_siglip_adapter.py"

    assert encoder_path.exists()
    source = encoder_path.read_text(encoding="utf-8")

    assert "@MODELS.register_module()" in source
    assert "class OnlineSigLIPFrameEncoder" in source
    assert "input_format = \"raw_frames\"" in source
    assert "online = True" in source
    assert "strict_online = True" in source
    assert "AutoImageProcessor" in source
    assert "AutoModel" in source
    assert "frame_chunk_size" in source
    assert "freeze_vision_encoder" in source
    assert "CausalMotionBranch" in source
    assert "use_motion_branch" in source
    assert "torch.no_grad()" in source
    assert "pixel_attention_mask" in source
    assert "attention_mask" in source
    assert "spatial_shapes" in source
    assert "_build_vision_inputs" in source
    assert "_masked_mean_pool" in source
    assert "self.processor(" in source
    assert "do_rescale=False" in source
    assert "frame_stride != 1" in source


def test_backbone_registry_exposes_siglip_online_encoder():
    source = read("opentad/models/backbones/__init__.py")

    assert "OnlineSigLIPFrameEncoder" in source
    assert "CausalMotionBranch" in source
    assert '"OnlineSigLIPFrameEncoder"' in source


def test_detector_and_adapter_pass_metas_and_updated_masks_through_backbone():
    detector_source = read("opentad/models/detectors/single_stage.py")
    adapter_source = read("opentad/models/backbones/online_videomae_adapter.py")

    assert "_forward_backbone" in detector_source
    assert "metas=metas" in detector_source
    assert "_unpack_backbone_output" in detector_source
    assert "x, masks = self._forward_backbone(inputs, masks, metas)" in detector_source

    assert "del metas" not in adapter_source
    assert "masks=masks" in adapter_source
    assert "metas=metas" in adapter_source
    assert "_call_frame_backbone" in adapter_source
    assert "_unpack_backbone_output" in adapter_source
    assert "return x.to(torch.float32), masks" in adapter_source


def test_p0_siglip2_config_is_frozen_recent_frame_no_memory_baseline():
    Config = __import__("mmengine.config", fromlist=["Config"]).Config

    cfg = Config.fromfile(str(ROOT / "configs/causaltad/thumos_siglip2_matr_ontad_p0.py"))

    assert cfg.route_stage == "P0"
    assert cfg.model.backbone.type == "OnlineVideoMAEAdapter"
    assert cfg.model.backbone.use_stub_backbone is False
    assert cfg.model.backbone.freeze_backbone is True
    assert cfg.model.backbone.backbone.type == "OnlineSigLIPFrameEncoder"
    assert cfg.model.backbone.backbone.model_name.endswith("/hf_models/google-siglip2-base-patch16-224")
    assert cfg.model.backbone.backbone.local_files_only is True
    assert cfg.model.backbone.backbone.freeze_vision_encoder is True
    assert cfg.model.backbone.backbone.strict_online is True
    assert cfg.model.projection.type == "CausalTemporalMaxerProj"
    assert cfg.model.projection.strict_causal is True
    assert cfg.model.rpn_head.memory_size == 0
    assert cfg.model.rpn_head.use_boundary_scores is False
    assert cfg.model.rpn_head.use_emit_scores is True
    assert cfg.dataset.test.window_overlap_ratio == 0.0
    assert cfg.post_processing.streaming is True
    assert cfg.post_processing.sliding_window is False
    assert cfg.post_processing.streaming_safe_emission is True
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.fixed_raw_frame_protocol.image_size in (224, 256, 384)
    assert cfg.fixed_raw_frame_protocol.snippet_stride_frames == cfg.dataset.train.feature_stride
    assert cfg.data_path.endswith("/raw_data/video")
    assert cfg.dataset.train.pipeline[1].frame_format == "video"
    assert cfg.dataset.train.pipeline[1].video_filename_tmpl == "{}.mp4"
    assert cfg.dataset.train.stream_id == "thumos_siglip2_p0"
    assert cfg.dataset.train.processor_id == cfg.fixed_raw_frame_protocol.processor
    assert cfg.dataset.train.encoder_id == cfg.fixed_raw_frame_protocol.encoder_hub_id
    assert cfg.dataset.train.image_size == cfg.fixed_raw_frame_protocol.image_size
    assert cfg.dataset.train.frame_policy == cfg.fixed_raw_frame_protocol.frame_policy
    assert "processor_id" in cfg.dataset.train.pipeline[-1].meta_keys
    assert "encoder_id" in cfg.dataset.train.pipeline[-1].meta_keys
    assert "image_size" in cfg.dataset.train.pipeline[-1].meta_keys
    assert "frame_policy" in cfg.dataset.train.pipeline[-1].meta_keys
    assert "input_format" in cfg.dataset.train.pipeline[-1].meta_keys


def test_p1_siglip2_config_declares_trainable_raw_frame_adapter_protocol():
    Config = __import__("mmengine.config", fromlist=["Config"]).Config

    cfg = Config.fromfile(str(ROOT / "configs/causaltad/thumos_siglip2_matr_ontad_p1.py"))

    assert cfg.route_stage == "P1"
    assert cfg.model.backbone.use_stub_backbone is False
    assert cfg.model.backbone.backbone.type == "OnlineSigLIPFrameEncoder"
    assert cfg.model.backbone.backbone.freeze_vision_encoder is True
    assert cfg.model.projection.type == "CausalTemporalMaxerProj"
    assert cfg.model.projection.strict_causal is True
    assert cfg.model.rpn_head.memory_size == 0
    assert cfg.model.rpn_head.use_boundary_scores is False
    assert cfg.model.rpn_head.online_censored_training is True
    assert cfg.model.rpn_head.max_future_offset == 0.0
    assert cfg.dataset.train.input_format == "raw_frames"
    assert cfg.dataset.train.window_overlap_ratio == 0.0
    assert cfg.dataset.val.window_overlap_ratio == 0.0
    assert cfg.dataset.test.window_overlap_ratio == 0.0
    assert cfg.fixed_raw_frame_protocol.fps > 0
    assert cfg.fixed_raw_frame_protocol.snippet_stride_frames > 0
    assert cfg.temporal_roundtrip_check.enabled is True
    assert "backbone.adapter" in cfg.trainable_scope.modules
    assert "projection" in cfg.trainable_scope.modules
    assert "rpn_head" in cfg.trainable_scope.modules
    assert cfg.dataset.train.stream_id == "thumos_siglip2_p1"
    assert cfg.dataset.train.processor_id == cfg.fixed_raw_frame_protocol.processor
    assert cfg.dataset.train.encoder_id == cfg.fixed_raw_frame_protocol.encoder_hub_id
    assert cfg.dataset.train.image_size == cfg.fixed_raw_frame_protocol.image_size
    assert cfg.dataset.train.frame_policy == cfg.fixed_raw_frame_protocol.frame_policy
    assert cfg.post_processing.save_emission_ledger is True
    assert cfg.post_processing.save_latency_summary is True


def test_p1_pilot_config_runs_real_eval_with_emission_reports():
    Config = __import__("mmengine.config", fromlist=["Config"]).Config

    cfg = Config.fromfile(str(ROOT / "configs/causaltad/thumos_siglip2_matr_ontad_p1_pilot.py"))

    assert cfg.route_stage == "P1-pilot"
    assert cfg.formal_training_ready is False
    assert cfg.model.backbone.use_stub_backbone is False
    assert cfg.model.projection.type == "CausalTemporalMaxerProj"
    assert cfg.model.rpn_head.memory_size == 0
    assert cfg.model.rpn_head.online_censored_training is True
    assert cfg.dataset.train.input_format == "raw_frames"
    assert len(cfg.dataset.train.allow_list) >= 4
    assert len(cfg.dataset.test.allow_list) >= 4
    assert cfg.workflow.val_start_epoch == 0
    assert cfg.workflow.val_eval_interval == 1
    assert cfg.workflow.end_epoch == cfg.scheduler.max_epoch
    assert cfg.post_processing.streaming_safe_emission is True
    assert cfg.post_processing.max_latency_frames == cfg.window_size * 8
    assert cfg.post_processing.save_emission_ledger is True
    assert cfg.post_processing.save_latency_summary is True
    assert cfg.post_processing.emission_ledger_filename.endswith(".json")
    assert cfg.post_processing.latency_summary_filename.endswith(".json")
    assert cfg.evaluation.thread == 4


def test_p1_fix_config_lowers_lr_controls_emissions_and_uses_subset_eval():
    Config = __import__("mmengine.config", fromlist=["Config"]).Config

    cfg = Config.fromfile(str(ROOT / "configs/causaltad/thumos_siglip2_matr_ontad_p1_fix.py"))

    assert cfg.route_stage == "P1-fix"
    assert cfg.formal_training_ready is False
    assert cfg.optimizer.lr <= 2e-4
    assert cfg.optimizer.backbone.lr <= 1e-4
    assert cfg.model.rpn_head.online_censored_training is True
    assert cfg.scheduler.warmup_epoch == 1
    assert cfg.workflow.val_eval_interval == 1
    assert cfg.post_processing.pre_nms_thresh >= 0.05
    assert cfg.post_processing.pre_nms_topk <= 500
    assert cfg.post_processing.max_latency_frames == 96 * 8
    assert cfg.post_processing.save_emission_ledger is True
    assert cfg.post_processing.save_latency_summary is True
    assert cfg.evaluation.allowed_videos == cfg.dataset.test.allow_list
    assert cfg.evaluation.thread == 4


def test_p1_full_60_config_uses_full_dataset_and_full_validation_eval():
    Config = __import__("mmengine.config", fromlist=["Config"]).Config

    cfg = Config.fromfile(str(ROOT / "configs/causaltad/thumos_siglip2_matr_ontad_p1_full_60.py"))

    assert cfg.route_stage == "P1-full-60"
    assert cfg.formal_training_ready is False
    assert cfg.dataset.train.get("allow_list", None) is None
    assert cfg.dataset.val.get("allow_list", None) is None
    assert cfg.dataset.test.get("allow_list", None) is None
    assert cfg.evaluation.get("allowed_videos", None) is None
    assert cfg.scheduler.max_epoch == 60
    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.val_eval_interval == 10
    assert cfg.workflow.checkpoint_interval == 5
    assert cfg.optimizer.lr <= 1e-4
    assert cfg.model.rpn_head.online_censored_training is True
    assert cfg.post_processing.pre_nms_thresh >= 0.05
    assert cfg.post_processing.pre_nms_topk <= 300
    assert cfg.post_processing.max_latency_frames == 96 * 8
    assert cfg.post_processing.emission_ledger_filename == "p1_full_60_emission_ledger.json"
    assert cfg.post_processing.latency_summary_filename == "p1_full_60_latency_summary.json"


def test_p1_fix_and_full_configs_explicitly_preserve_streaming_safe_post_processing():
    Config = __import__("mmengine.config", fromlist=["Config"]).Config

    for config_name in (
        "thumos_siglip2_matr_ontad_p1_fix.py",
        "thumos_siglip2_matr_ontad_p1_full_60.py",
    ):
        source = read(f"configs/causaltad/{config_name}")
        cfg = Config.fromfile(str(ROOT / f"configs/causaltad/{config_name}"))

        assert "streaming=True" in source
        assert "sliding_window=False" in source
        assert "streaming_safe_emission=True" in source
        assert "max_latency=0.0" in source
        assert "max_latency_frames" in source
        assert cfg.post_processing.streaming is True
        assert cfg.post_processing.sliding_window is False
        assert cfg.post_processing.streaming_safe_emission is True
        assert cfg.post_processing.max_latency == 0.0
        assert cfg.post_processing.max_latency_frames == 96 * 8


def test_p0_smoke_config_limits_remote_validation_run():
    Config = __import__("mmengine.config", fromlist=["Config"]).Config

    cfg = Config.fromfile(str(ROOT / "configs/causaltad/thumos_siglip2_matr_ontad_p0_smoke.py"))

    assert cfg.route_stage == "P0-smoke"
    assert cfg.model.projection.type == "CausalTemporalMaxerProj"
    assert cfg.model.projection.strict_causal is True
    assert "use_abs_pe" not in cfg.model.projection
    assert "mamba_kernel_size" not in cfg.model.projection
    assert cfg.workflow.end_epoch == 1
    assert cfg.workflow.disable_checkpoint is True
    assert cfg.solver.train.num_workers == 0
    assert cfg.dataset.train.allow_list == ["video_validation_0000051"]
    assert cfg.dataset.test.allow_list == ["video_test_0000004"]
    assert cfg.post_processing.max_latency_frames == cfg.window_size * 8


def test_p0_smoke_overfit_config_uses_effective_lr_for_learning_check():
    Config = __import__("mmengine.config", fromlist=["Config"]).Config

    cfg = Config.fromfile(str(ROOT / "configs/causaltad/thumos_siglip2_matr_ontad_p0_smoke_overfit.py"))

    assert cfg.route_stage == "P0-smoke-overfit"
    assert cfg.formal_training_ready is False
    assert cfg.model.projection.type == "CausalTemporalMaxerProj"
    assert cfg.model.projection.strict_causal is True
    assert cfg.scheduler.warmup_epoch == 1
    assert cfg.scheduler.max_epoch == cfg.workflow.end_epoch
    assert cfg.optimizer.lr > 1e-4
    assert cfg.workflow.end_epoch >= 6
    assert cfg.workflow.disable_checkpoint is True
    assert cfg.dataset.train.allow_list == ["video_validation_0000051"]
    assert cfg.dataset.train.stream_id == "thumos_siglip2_p0_smoke_overfit"


def test_p2_config_adds_causal_motion_or_streaming_safe_emission_contribution():
    Config = __import__("mmengine.config", fromlist=["Config"]).Config

    cfg = Config.fromfile(str(ROOT / "configs/causaltad/thumos_siglip2_motion_matr_ontad_p2.py"))

    assert cfg.route_stage == "P2"
    assert cfg.model.backbone.backbone.type == "OnlineSigLIPFrameEncoder"
    assert cfg.model.backbone.backbone.use_motion_branch is True
    assert cfg.model.backbone.backbone.motion_branch.causal is True
    assert cfg.post_processing.streaming_safe_emission is True
    assert cfg.post_processing.max_latency == 0.0
    assert cfg.post_processing.max_latency_frames == cfg.window_size * cfg.snippet_stride_frames
    assert cfg.model.rpn_head.clamp_end_to_current is True
    assert cfg.model.rpn_head.max_future_offset == 0.0
    assert cfg.model.rpn_head.online_censored_training is True
    assert cfg.model.rpn_head.use_boundary_scores is False
    assert cfg.model.rpn_head.use_emit_scores is True


def test_online_censored_training_masks_future_endpoint_supervision():
    anchor_source = read("opentad/models/dense_heads/anchor_free_head.py")
    matr_source = read("opentad/models/dense_heads/matr_head.py")

    assert "online_censored_training" in anchor_source
    assert "online_censored_max_future_offset" in anchor_source
    assert "endpoint_observed" in anchor_source
    assert "target_segments[:, 1]" in anchor_source
    assert "reg_loss_values *" in anchor_source

    assert "online_censored_training" in matr_source
    assert "endpoint_observed" in matr_source
    assert "end_mask = torch.logical_and(end_mask, endpoint_observed)" in matr_source
    assert "emit_target[batch_idx, end_mask, 0] = 1.0" in matr_source


def test_frame_grid_seconds_roundtrip_helpers_are_declared():
    source = read("opentad/models/utils/post_processing/utils.py")

    assert "def grid_to_seconds" in source
    assert "def seconds_to_grid" in source
    assert "window_start_frame" in source
    assert "snippet_stride" in source


def test_matr_boundary_score_fusion_is_not_same_point_start_end_product():
    source = read("opentad/models/dense_heads/matr_head.py")

    assert "Same-point start and end multiplication suppresses long actions." in source
    assert "scores = scores * start_score\n\n        if self.use_boundary_scores and end_pred is not None" not in source
    assert "kwargs.pop(\"online\", None)" in source


def test_raw_frame_loader_supports_remote_mp4_video_files():
    source = read("opentad/datasets/transforms/loading.py")

    assert '"video"' in source
    assert "def _load_from_video" in source
    assert "cv2.VideoCapture" in source
    assert "video_filename_tmpl" in source


def test_raw_frame_dataset_aligns_gt_with_fixed_fps_protocol():
    source = read("opentad/datasets/raw_frame.py")

    assert "effective_frames" in source
    assert "resolve_raw_video_total_frames" in source
    assert "def get_num_frames" in source
    assert "self.get_num_frames(video_info)" in source


def test_remote_siglip_submit_uses_slurm_not_login_node_training():
    script = read("tools/remote/submit_siglip_ontad_n16r4.sh")

    assert "sbatch" in script
    assert "tools/train.py" in script
    assert "activate_n16r4_causaltad.sh" in script
    assert "thumos_siglip2_matr_ontad_p0.py" in script
    assert "python tools/train.py" not in script
    assert "torchrun" in script
    assert "MASTER_PORT" in script
    assert "SLURM_JOB_ID" in script
    assert "--rdzv_endpoint" in script
    assert "127.0.0.1:${MASTER_PORT}" in script or "127.0.0.1:\\${MASTER_PORT}" in script
    assert "PREFLIGHT" in script
    assert "transformers" in script
    assert "opencv" in script
    assert "streaming-safe SigLIP/SigLIP2 online TAD routes require GPUS_PER_NODE=1" in script
