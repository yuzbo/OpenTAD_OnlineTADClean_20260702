_base_ = "./thumos_siglip2_matr_ontad_p1_pilot.py"

# P1-fix: stabilize the raw-frame online TAD pilot after the first P1 run.
# Changes versus p1_pilot:
# - lower adapter/head LR to avoid late-epoch loss growth;
# - raise pre-NMS threshold and lower top-k to reduce online emission volume;
# - evaluate mAP only on the same validation videos used by the pilot route.

route_stage = "P1-fix"
formal_training_ready = False
raw_frame_stream_id = "thumos_siglip2_p1_fix"

pilot_eval_videos = [
    "video_test_0000896",
    "video_test_0000897",
    "video_test_0001078",
    "video_test_0000179",
]

dataset = dict(
    train=dict(stream_id=raw_frame_stream_id),
    val=dict(
        allow_list=pilot_eval_videos,
        stream_id=raw_frame_stream_id,
    ),
    test=dict(
        allow_list=pilot_eval_videos,
        stream_id=raw_frame_stream_id,
    ),
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.02,
    paramwise=True,
    backbone=dict(lr=5e-5, weight_decay=0.02),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=1, max_epoch=8)

workflow = dict(
    logging_interval=5,
    checkpoint_interval=2,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=0,
    end_epoch=8,
    disable_checkpoint=False,
)

post_processing = dict(
    streaming=True,
    sliding_window=False,
    streaming_safe_emission=True,
    max_latency=0.0,
    pre_nms_thresh=0.05,
    pre_nms_topk=300,
    streaming_nms_iou=0.5,
    save_emission_ledger=True,
    save_latency_summary=True,
    emission_ledger_filename="p1_fix_emission_ledger.json",
    latency_summary_filename="p1_fix_latency_summary.json",
)

evaluation = dict(
    allowed_videos=pilot_eval_videos,
    thread=4,
)

work_dir = "exps/thumos/siglip2_matr_ontad_p1_fix"
