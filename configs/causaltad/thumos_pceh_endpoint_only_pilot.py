_base_ = "./thumos_pceh_endpoint_only.py"

# Three-epoch fair endpoint-only control for the PCEH decision pilot.
route_stage = "endpoint_only_short_pilot"
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

work_dir = "exps/thumos/pceh_endpoint_only_short_pilot"
