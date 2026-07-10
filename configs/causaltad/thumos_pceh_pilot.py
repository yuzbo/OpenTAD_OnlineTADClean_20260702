_base_ = "./thumos_pceh_ontad.py"

# Three-epoch single-seed decision pilot. This is not a formal result config.
route_stage = "pceh_short_pilot"
formal_training_ready = False

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=0,
    end_epoch=3,
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=1, max_epoch=3)

work_dir = "exps/thumos/pceh_short_pilot"
