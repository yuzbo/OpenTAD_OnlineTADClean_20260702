_base_ = ["./thumos_persistent_binding_fixed.py"]

route_stage = "persistent_binding_feature_slurm_smoke"
formal_training_ready = False
smoke_only = True

# The integration smoke must exercise serialization before meaningful
# training. Formal/screen configs retain their fit-derived low event priors.
model = dict(
    prior_bias_mode="raw_probability",
    head=dict(
        birth_prior_probability=0.5,
        alive_prior_probability=0.5,
        end_prior_probability=0.5,
    )
)

scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=0,
    warmup_start_lr=2e-4,
    max_epoch=1,
)

workflow = dict(
    fit_only=False,
    logging_interval=1,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=0,
    end_epoch=1,
    runtime_debug_interval=1,
)

work_dir = "exps/thumos/persistent_binding_smoke"
