_base_ = "./thumos_siglip2_matr_ontad_p0.py"

# P1: formal raw-frame online TAD training route.
#
# This keeps the visual tower frozen and trains the causal adapter,
# projection, FPN normalization, and MATR head from raw frames under one fixed
# processor/FPS/stride protocol.

route_stage = "P1"
formal_training_ready = False

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
