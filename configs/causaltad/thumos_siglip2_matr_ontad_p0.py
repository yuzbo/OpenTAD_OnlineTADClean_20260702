# P0: Frozen SigLIP2 recent-frame baseline for online raw-frame TAD.
#
# This route is intentionally conservative:
# - raw frames are loaded directly; no feature cache or raw-prediction shortcut;
# - one sampled frame produces one TAD grid token, so the visual tower has no
#   future-frame path;
# - MATR streaming memory is disabled for the first trainable baseline.

route_stage = "P0"
formal_training_ready = False

annotation_path = "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json"
class_map = "/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt"
data_path = "/data/run01/sczc063/yuzibo/thumos14/raw_data/video"

fixed_raw_frame_protocol = dict(
    encoder="/data/run01/sczc063/yuzibo/hf_models/google-siglip2-base-patch16-224",
    encoder_hub_id="google/siglip2-base-patch16-224",
    processor="AutoImageProcessor",
    image_size=224,
    fps=30.0,
    snippet_stride_frames=8,
    sample_stride=1,
    token_seconds=8.0 / 30.0,
    frame_policy="recent_frame_only",
)
temporal_roundtrip_check = dict(enabled=True, helpers=["grid_to_seconds", "seconds_to_grid"])
trainable_scope = dict(modules=["backbone.adapter", "backbone.out_proj", "backbone.norm", "projection", "neck", "rpn_head"])

window_size = 192
snippet_stride_frames = fixed_raw_frame_protocol["snippet_stride_frames"]
image_size = fixed_raw_frame_protocol["image_size"]
raw_frame_stream_id = "thumos_siglip2_p0"
online_meta_keys = [
    "video_name",
    "data_path",
    "input_format",
    "stream_id",
    "processor_id",
    "encoder_id",
    "image_size",
    "frame_policy",
    "fps",
    "duration",
    "snippet_stride",
    "window_start_frame",
    "window_end_frame",
    "window_size",
    "offset_frames",
]

data_shape = dict(
    input_format="raw_frames",
    frame_layout="[B, N, 3, T, H, W]",
    online=True,
    visual_encoder="Frozen SigLIP2 frame tower",
    notes="No feature cache, dataset, checkpoint, or generated result archive is stored in this repo.",
)

_train_pipeline = [
    dict(type="LoadFrames", num_clips=1, scale_factor=1, method="sliding_window"),
    dict(
        type="LoadRawFrames",
        frame_format="video",
        video_filename_tmpl="{}.mp4",
        start_index=0,
        frame_size=(image_size, image_size),
        layout="[N,3,T,H,W]",
    ),
    dict(type="ConvertToTensor", keys=["gt_segments", "gt_labels"]),
    dict(type="Collect", inputs="frames", keys=["masks", "gt_segments", "gt_labels"], meta_keys=online_meta_keys),
]

_test_pipeline = [
    dict(type="LoadFrames", num_clips=1, scale_factor=1, method="sliding_window"),
    dict(
        type="LoadRawFrames",
        frame_format="video",
        video_filename_tmpl="{}.mp4",
        start_index=0,
        frame_size=(image_size, image_size),
        layout="[N,3,T,H,W]",
    ),
    dict(type="Collect", inputs="frames", keys=["masks"], meta_keys=online_meta_keys),
]

dataset = dict(
    train=dict(
        type="FrameWindowDataset",
        input_format="raw_frames",
        online=True,
        stream_id=raw_frame_stream_id,
        processor_id=fixed_raw_frame_protocol["processor"],
        encoder_id=fixed_raw_frame_protocol["encoder_hub_id"],
        image_size=image_size,
        frame_policy=fixed_raw_frame_protocol["frame_policy"],
        ann_file=annotation_path,
        subset_name="training",
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        feature_stride=snippet_stride_frames,
        sample_stride=1,
        fps=fixed_raw_frame_protocol["fps"],
        offset_frames=0,
        window_size=window_size,
        window_overlap_ratio=0.0,
        ioa_thresh=0.75,
        pipeline=_train_pipeline,
    ),
    val=dict(
        type="FrameWindowDataset",
        input_format="raw_frames",
        online=True,
        stream_id=raw_frame_stream_id,
        processor_id=fixed_raw_frame_protocol["processor"],
        encoder_id=fixed_raw_frame_protocol["encoder_hub_id"],
        image_size=image_size,
        frame_policy=fixed_raw_frame_protocol["frame_policy"],
        ann_file=annotation_path,
        subset_name="validation",
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        feature_stride=snippet_stride_frames,
        sample_stride=1,
        fps=fixed_raw_frame_protocol["fps"],
        offset_frames=0,
        window_size=window_size,
        window_overlap_ratio=0.0,
        ioa_thresh=0.75,
        pipeline=_train_pipeline,
    ),
    test=dict(
        type="FrameWindowDataset",
        input_format="raw_frames",
        online=True,
        stream_id=raw_frame_stream_id,
        processor_id=fixed_raw_frame_protocol["processor"],
        encoder_id=fixed_raw_frame_protocol["encoder_hub_id"],
        image_size=image_size,
        frame_policy=fixed_raw_frame_protocol["frame_policy"],
        ann_file=annotation_path,
        subset_name="validation",
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        feature_stride=snippet_stride_frames,
        sample_stride=1,
        fps=fixed_raw_frame_protocol["fps"],
        offset_frames=0,
        window_size=window_size,
        window_overlap_ratio=0.0,
        ioa_thresh=0.0,
        pipeline=_test_pipeline,
    ),
)

model = dict(
    type="VideoMambaSuite",
    backbone=dict(
        type="OnlineVideoMAEAdapter",
        input_format="raw_frames",
        online=True,
        strict_online=True,
        in_channels=3,
        embed_dims=768,
        adapter_channels=192,
        out_channels=768,
        causal_kernel_size=3,
        freeze_backbone=True,
        use_stub_backbone=False,
        return_masks=True,
        output_layout="bct",
        norm_cfg=dict(type="LN"),
        backbone=dict(
            type="OnlineSigLIPFrameEncoder",
            model_name=fixed_raw_frame_protocol["encoder"],
            backend="transformers",
            embed_dims=768,
            image_size=image_size,
            pooling="mean",
            frame_stride=1,
            frame_chunk_size=32,
            freeze_vision_encoder=True,
            strict_online=True,
            local_files_only=True,
            use_motion_branch=False,
            output_layout="bct",
        ),
    ),
    projection=dict(
        type="CausalProj",
        in_channels=768,
        out_channels=512,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3),
        norm_cfg=dict(type="LN"),
        use_abs_pe=False,
        max_seq_len=1024,
        input_pdrop=0.1,
        mamba_kernel_size=4,
        channel_expand=2,
        num_head=4,
        drop_path_rate=0.1,
        strict_causal=True,
    ),
    neck=dict(
        type="FPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
    rpn_head=dict(
        type="MATRHead",
        online=True,
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        kernel_size=3,
        emit_threshold=0.0,
        memory_size=0,
        clamp_end_to_current=True,
        max_future_offset=0.0,
        boundary_loss_weight=0.2,
        actionness_loss_weight=0.2,
        emit_loss_weight=0.2,
        use_boundary_scores=False,
        use_actionness_scores=True,
        use_emit_scores=True,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        ),
        loss_normalizer=250,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(lr=5e-5, weight_decay=0.05),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=2, max_epoch=10)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    streaming=True,
    streaming_safe_emission=True,
    sliding_window=False,
    max_latency=0.0,
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.0,
    ),
    save_dict=False,
)
solver = dict(
    train=dict(batch_size=1, num_workers=4),
    val=dict(batch_size=1, num_workers=4),
    test=dict(batch_size=1, num_workers=4),
    clip_grad_norm=1,
    ema=True,
    amp=True,
)

evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=8,
)

work_dir = "exps/thumos/siglip2_matr_ontad_p0"
