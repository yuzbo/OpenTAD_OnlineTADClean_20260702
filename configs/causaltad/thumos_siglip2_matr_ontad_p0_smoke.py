_base_ = "./thumos_siglip2_matr_ontad_p0.py"

# One-video remote smoke for the real raw-MP4 SigLIP2 route.
# Use through tools/remote/submit_siglip_ontad_n16r4.sh before launching the
# full P0/P1 sweeps.

route_stage = "P0-smoke"
formal_training_ready = False
raw_frame_stream_id = "thumos_siglip2_p0_smoke"

window_size = 64

dataset = dict(
    train=dict(
        allow_list=["video_validation_0000051"],
        stream_id=raw_frame_stream_id,
        window_size=window_size,
    ),
    val=dict(
        allow_list=["video_test_0000004"],
        stream_id=raw_frame_stream_id,
        window_size=window_size,
    ),
    test=dict(
        allow_list=["video_test_0000004"],
        stream_id=raw_frame_stream_id,
        window_size=window_size,
    ),
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=0, max_epoch=1)
solver = dict(
    train=dict(batch_size=1, num_workers=0),
    val=dict(batch_size=1, num_workers=0),
    test=dict(batch_size=1, num_workers=0),
    clip_grad_norm=1,
    ema=False,
    amp=True,
)
workflow = dict(
    logging_interval=1,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=99,
    end_epoch=1,
    disable_checkpoint=True,
)

work_dir = "exps/thumos/siglip2_matr_ontad_p0_smoke"
