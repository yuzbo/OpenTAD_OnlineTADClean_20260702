# PCEH-OnTAD candidate: Prefix-Censored Event-Emission Hazard Learning.
#
# This config is executable but intentionally not marked formal-training ready.
# Gate promotion requires remote dataset/model-path validation, causal audits,
# optimizer audit output, and a single-rank smoke run. The strict stream never
# loads cached features or saved raw predictions.

route_stage = "pceh_ontad_candidate"
formal_training_ready = False

annotation_path = "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json"
class_map = "/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt"
data_path = "/data/run01/sczc063/yuzibo/thumos14/raw_data/video"
encoder_path = "/data/run01/sczc063/yuzibo/hf_models/google-siglip2-base-patch16-224"

fps = 30.0
packet_size_frames = 8
primary_latency_budget_sec = 1.0
delay_budget_frames = int(primary_latency_budget_sec * fps)
cache_size_tokens = 192

stream_protocol = dict(
    protocol="strict_prefix_packets_v1",
    acquisition_policy="packet_recent_frame",
    packet_size_frames=packet_size_frames,
    decision_cadence_frames=packet_size_frames,
    decision_cadence_sec=packet_size_frames / fps,
    context_cache_tokens=cache_size_tokens,
    primary_latency_budget_sec=primary_latency_budget_sec,
    latency_budgets_sec=[0.5, 1.0, 2.0, 4.0],
    latency_definition="emit_time_minus_matched_gt_end",
    predicted_end_latency_role="diagnostic_only",
    emissions="immutable",
    late_predictions="retained_as_false_positives",
    evaluation_world_size=1,
)

raw_frame_stream_id = "thumos_pceh_ontad_v1"
image_size = 224
online_meta_keys = [
    "video_name",
    "input_format",
    "stream_id",
    "processor_id",
    "encoder_id",
    "image_size",
    "frame_policy",
    "fps",
    "snippet_stride",
    "offset_frames",
    "packet_start_frame",
    "packet_end_frame",
    "is_video_start",
    "is_video_end",
    "delay_budget_frames",
    "encoded_source_frames",
]

_packet_pipeline = [
    dict(type="LoadStreamPacketFrames", policy=None, stride=1),
    dict(
        type="LoadRawFrames",
        frame_format="video",
        video_filename_tmpl="{}.mp4",
        start_index=0,
        frame_size=(image_size, image_size),
        layout="[N,3,T,H,W]",
    ),
    dict(
        type="Collect",
        inputs="frames",
        keys=["masks"],
        meta_keys=online_meta_keys,
    ),
]

_dataset_common = dict(
    type="StreamingRawFrameDataset",
    input_format="raw_frames",
    online=True,
    stream_id=raw_frame_stream_id,
    processor_id="AutoImageProcessor",
    encoder_id="google/siglip2-base-patch16-224",
    image_size=image_size,
    frame_policy=stream_protocol["acquisition_policy"],
    packet_size_frames=packet_size_frames,
    delay_budget_frames=delay_budget_frames,
    expose_terminal_duration=True,
    ann_file=annotation_path,
    class_map=class_map,
    data_path=data_path,
    filter_gt=False,
    feature_stride=1,
    sample_stride=1,
    fps=fps,
    offset_frames=0,
    window_size=packet_size_frames,
    window_overlap_ratio=0.0,
    ioa_thresh=0.0,
    pipeline=_packet_pipeline,
)

dataset = dict(
    train=dict(**_dataset_common, subset_name="training"),
    val=dict(**_dataset_common, subset_name="validation"),
    test=dict(**_dataset_common, subset_name="validation"),
)

model = dict(
    type="PCEHOnlineDetector",
    cache_size=cache_size_tokens,
    detach_stream_state=True,
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
            model_name=encoder_path,
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
        type="CausalTemporalMaxerProj",
        in_channels=768,
        out_channels=512,
        arch=(2, 0, 0),
        conv_cfg=dict(kernel_size=3),
        norm_cfg=dict(type="LN"),
        drop_out=0.1,
        strict_causal=True,
    ),
    head=dict(
        type="PrefixEventEmissionHead",
        in_channels=512,
        num_classes=20,
        emission_policy="pceh",
        class_threshold=0.5,
        start_threshold=0.5,
        end_threshold=0.5,
        completion_threshold=0.5,
        emission_threshold=0.5,
        class_loss_weight=1.0,
        start_loss_weight=0.25,
        ongoing_loss_weight=0.25,
        end_loss_weight=1.0,
        completion_loss_weight=0.5,
        emission_loss_weight=1.0,
        delay_loss_weight=0.25,
        calibration_loss_weight=0.1,
    ),
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    backbone=dict(lr=5e-5, weight_decay=0.05),
    audit=dict(fail_on_frozen=True),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=2, max_epoch=30)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    streaming=True,
    streaming_safe_emission=True,
    sliding_window=False,
    save_emission_ledger=True,
    emission_ledger_filename="pceh_emission_ledger.json",
    save_latency_summary=True,
    latency_summary_filename="pceh_predicted_end_latency_summary.json",
    save_dict=False,
)

solver = dict(
    train=dict(
        batch_size=1,
        stream_batch_size=1,
        streaming=True,
        num_workers=0,
    ),
    val=dict(
        batch_size=1,
        stream_batch_size=1,
        streaming=True,
        num_workers=0,
    ),
    test=dict(
        batch_size=1,
        stream_batch_size=1,
        streaming=True,
        num_workers=0,
    ),
    static_graph=False,
    clip_grad_norm=1.0,
    ema=False,
    amp=True,
)

evaluation = dict(
    type="OnlineAPBudgeted",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    latency_budgets_sec=stream_protocol["latency_budgets_sec"],
    fps=fps,
    require_ledger=True,
    require_no_future=True,
    ground_truth_filename=annotation_path,
)

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=0,
    end_epoch=30,
)

work_dir = "exps/thumos/pceh_ontad"
