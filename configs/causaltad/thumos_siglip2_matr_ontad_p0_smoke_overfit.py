_base_ = "./thumos_siglip2_matr_ontad_p0_smoke.py"

# Diagnostic overfit run for the frozen-SigLIP2 raw-frame chain.
# This inherits the smoke projection on purpose: it verifies real MP4 frames,
# SigLIP2 preprocessing/encoding, trainable adapters/projection, and MATR losses
# can optimize end to end before the formal CausalProj ABI path is repaired.

route_stage = "P0-smoke-overfit"
formal_training_ready = False
raw_frame_stream_id = "thumos_siglip2_p0_smoke_overfit"

dataset = dict(
    train=dict(stream_id=raw_frame_stream_id),
    val=dict(stream_id=raw_frame_stream_id),
    test=dict(stream_id=raw_frame_stream_id),
)

optimizer = dict(
    type="AdamW",
    lr=1e-3,
    weight_decay=0.01,
    paramwise=True,
    backbone=dict(lr=5e-4, weight_decay=0.01),
)

# warmup_epoch must be positive with the current per-iteration scheduler.  The
# one-epoch smoke run uses eta_min-scale LR; this run is meant to prove learning.
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=1, max_epoch=8)

workflow = dict(
    logging_interval=1,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=99,
    end_epoch=8,
    disable_checkpoint=True,
)

work_dir = "exps/thumos/siglip2_matr_ontad_p0_smoke_overfit"
