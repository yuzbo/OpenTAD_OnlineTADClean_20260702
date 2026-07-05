_base_ = "./thumos_siglip2_matr_ontad_p1.py"

# P1 pilot: real raw-MP4 online TAD training with validation mAP and
# streaming emission-ledger/latency reports.  This is a controlled multi-video
# run for pipeline validation, not a paper-number config.

route_stage = "P1-pilot"
formal_training_ready = False
raw_frame_stream_id = "thumos_siglip2_p1_pilot"

window_size = 96

dataset = dict(
    train=dict(
        allow_list=[
            "video_validation_0000950",
            "video_validation_0000270",
            "video_validation_0000178",
            "video_validation_0000179",
            "video_validation_0000174",
            "video_validation_0000175",
        ],
        stream_id=raw_frame_stream_id,
        window_size=window_size,
    ),
    val=dict(
        allow_list=[
            "video_test_0000896",
            "video_test_0000897",
            "video_test_0001078",
            "video_test_0000179",
        ],
        stream_id=raw_frame_stream_id,
        window_size=window_size,
    ),
    test=dict(
        allow_list=[
            "video_test_0000896",
            "video_test_0000897",
            "video_test_0001078",
            "video_test_0000179",
        ],
        stream_id=raw_frame_stream_id,
        window_size=window_size,
    ),
)

optimizer = dict(
    type="AdamW",
    lr=7e-4,
    weight_decay=0.02,
    paramwise=True,
    backbone=dict(lr=3e-4, weight_decay=0.02),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=1, max_epoch=5)

solver = dict(
    train=dict(batch_size=1, num_workers=0),
    val=dict(batch_size=1, num_workers=0),
    test=dict(batch_size=1, num_workers=0),
    clip_grad_norm=1,
    ema=False,
    amp=True,
)

workflow = dict(
    logging_interval=5,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=0,
    end_epoch=5,
    disable_checkpoint=False,
)

post_processing = dict(
    save_emission_ledger=True,
    save_latency_summary=True,
    emission_ledger_filename="p1_pilot_emission_ledger.json",
    latency_summary_filename="p1_pilot_latency_summary.json",
)

evaluation = dict(thread=4)

work_dir = "exps/thumos/siglip2_matr_ontad_p1_pilot"
