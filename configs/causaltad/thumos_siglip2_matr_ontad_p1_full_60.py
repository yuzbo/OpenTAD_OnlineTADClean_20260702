_base_ = "./thumos_siglip2_matr_ontad_p1_fix.py"

# P1-full-60: full THUMOS raw-frame online TAD training/evaluation run.
# This removes all pilot allow-lists, keeps the P1-fix stability settings, and
# evaluates the full validation split every 10 epochs including epoch 60.

route_stage = "P1-full-60"
formal_training_ready = False
raw_frame_stream_id = "thumos_siglip2_p1_full_60"

dataset = dict(
    train=dict(
        allow_list=None,
        stream_id=raw_frame_stream_id,
    ),
    val=dict(
        allow_list=None,
        stream_id=raw_frame_stream_id,
    ),
    test=dict(
        allow_list=None,
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
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=1, max_epoch=60)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=10,
    val_start_epoch=9,
    end_epoch=60,
    disable_checkpoint=False,
)

post_processing = dict(
    pre_nms_thresh=0.05,
    pre_nms_topk=300,
    streaming_nms_iou=0.5,
    save_emission_ledger=True,
    save_latency_summary=True,
    emission_ledger_filename="p1_full_60_emission_ledger.json",
    latency_summary_filename="p1_full_60_latency_summary.json",
)

evaluation = dict(
    allowed_videos=None,
    thread=8,
)

work_dir = "exps/thumos/siglip2_matr_ontad_p1_full_60"
