# Experimental raw-frame online TAD entry.
#
# This file is intentionally standalone. It must not inherit feature-level
# THUMOS configs, because those configs load pre-extracted features. The
# declared FrameWindowDataset pipeline is real, but the backbone below remains
# a contract-only stub until a causal/streaming VideoMAE is wired in.

formal_training_ready = False

data_shape = dict(
    input_format="raw_frames",
    frame_layout="[B, N, 3, T, H, W]",
    online=True,
    requires_frame_dataset=True,
    notes="No feature cache, dataset, checkpoint, or generated result archive is stored in this repo.",
)

annotation_path = "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json"
class_map = "/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt"
data_path = "/data/run01/sczc063/yuzibo/thumos14/rawframes"
window_size = 768

dataset = dict(
    train=dict(
        type="FrameWindowDataset",
        input_format="raw_frames",
        online=True,
        ann_file=annotation_path,
        subset_name="training",
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        feature_stride=1,
        sample_stride=1,
        offset_frames=0,
        window_size=window_size,
        window_overlap_ratio=0.25,
        ioa_thresh=0.75,
        pipeline=[
            dict(type="LoadFrames", num_clips=1, scale_factor=1, method="sliding_window"),
            dict(
                type="LoadRawFrames",
                frame_format="dir",
                filename_tmpl="{:05d}.jpg",
                start_index=1,
                frame_size=(160, 160),
                layout="[N,3,T,H,W]",
            ),
            dict(type="ConvertToTensor", keys=["gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="frames", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type="FrameWindowDataset",
        input_format="raw_frames",
        online=True,
        ann_file=annotation_path,
        subset_name="validation",
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        feature_stride=1,
        sample_stride=1,
        offset_frames=0,
        window_size=window_size,
        window_overlap_ratio=0.25,
        ioa_thresh=0.75,
        pipeline=[
            dict(type="LoadFrames", num_clips=1, scale_factor=1, method="sliding_window"),
            dict(
                type="LoadRawFrames",
                frame_format="dir",
                filename_tmpl="{:05d}.jpg",
                start_index=1,
                frame_size=(160, 160),
                layout="[N,3,T,H,W]",
            ),
            dict(type="ConvertToTensor", keys=["gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="frames", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type="FrameWindowDataset",
        input_format="raw_frames",
        online=True,
        ann_file=annotation_path,
        subset_name="validation",
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        feature_stride=1,
        sample_stride=1,
        offset_frames=0,
        window_size=window_size,
        window_overlap_ratio=0.5,
        ioa_thresh=0.0,
        pipeline=[
            dict(type="LoadFrames", num_clips=1, scale_factor=1, method="sliding_window"),
            dict(
                type="LoadRawFrames",
                frame_format="dir",
                filename_tmpl="{:05d}.jpg",
                start_index=1,
                frame_size=(160, 160),
                layout="[N,3,T,H,W]",
            ),
            dict(type="Collect", inputs="frames", keys=["masks"]),
        ],
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
        use_stub_backbone=True,
        stub_backbone_contract_only=True,
        output_layout="bct",
        norm_cfg=dict(type="LN"),
    ),
    projection=dict(
        type="CausalProj",
        in_channels=768,
        out_channels=512,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3),
        norm_cfg=dict(type="LN"),
        use_abs_pe=False,
        max_seq_len=2304,
        input_pdrop=0.1,
        mamba_kernel_size=4,
        channel_expand=2,
        num_head=4,
        drop_path_rate=0.3,
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
        memory_size=256,
        clamp_end_to_current=True,
        max_future_offset=0.0,
        boundary_loss_weight=0.2,
        actionness_loss_weight=0.2,
        emit_loss_weight=0.2,
        use_boundary_scores=True,
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
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=30)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    streaming=True,
    sliding_window=False,
    max_latency=0.0,
    max_latency_frames=window_size,
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
    val_start_epoch=20,
)

work_dir = "exps/thumos/videomae_adapter_matr_ontad"
