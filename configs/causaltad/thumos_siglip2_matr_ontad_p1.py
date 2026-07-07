_base_ = "./thumos_siglip2_matr_ontad_p0.py"

# P1: raw-frame online TAD validation candidate under validation.
#
# This keeps the visual tower frozen and trains the causal adapter,
# projection, FPN normalization, and MATR head from raw frames under one fixed
# processor/FPS/stride protocol. It is not formal-training-ready until the
# streaming-safe evaluator and emission-ledger checks pass on real runs.

route_stage = "P1"
formal_training_ready = False
raw_frame_stream_id = "thumos_siglip2_p1"

dataset = dict(
    train=dict(stream_id=raw_frame_stream_id),
    val=dict(stream_id=raw_frame_stream_id),
    test=dict(stream_id=raw_frame_stream_id),
)

model = dict(
    rpn_head=dict(
        # P1 trains online emissions from raw frames. Keep classification and
        # actionness labels, but do not regress or emit a future endpoint before
        # that endpoint is observable under the streaming latency budget.
        online_censored_training=True,
    )
)

trainable_scope = dict(
    modules=[
        "backbone.adapter",
        "backbone.out_proj",
        "backbone.norm",
        "projection",
        "neck",
        "rpn_head",
    ]
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=30)
workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=20,
)

work_dir = "exps/thumos/siglip2_matr_ontad_p1"
