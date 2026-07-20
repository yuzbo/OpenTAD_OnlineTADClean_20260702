_base_ = ["./thumos_persistent_binding_base.py"]

# Budgeted seed-705 convergence/non-degeneracy screen. This deliberately keeps
# formal_training_ready false: it is not a paper-effectiveness run.
route_stage = "persistent_binding_feature_seed705_screen"
formal_training_ready = False
screening_only = True
screening_training_ready = True

screening_contract = dict(
    schema_version="persistent_binding_seed705_screen.v1",
    seed=705,
    epochs=1,
    fit_videos=160,
    calibration_videos=40,
    reporting_videos_accessed=0,
    paired_gpu_hour_cap=2.0,
    preoptimization_profile_upper_bound_gpu_hours=1.6913390777756898,
    same_commit_profile_gate_required=True,
    purpose="convergence_and_non_degeneracy_only",
    effectiveness_claim_authorized=False,
    raw_rgb_authorized=False,
    shared_schedule_revision="short_warmup_v1",
    changed_axis="shared_linear_warmup_fraction_only",
    previous_warmup_epoch=1.0,
    warmup_epoch=0.1,
    peak_learning_rate=2e-4,
    mean_lr_exposure_ratio_vs_full_epoch_warmup=1.89066304675978,
)

# The previous one-epoch screen spent all 2,010 updates in linear warm-up and
# moved each repaired lifecycle bias by less than 0.02 logit. This shared
# schedule repair keeps the update count and peak LR fixed while reaching the
# peak after the first 10% of the fit epoch.
workflow = dict(
    fit_only=True,
    logging_interval=50,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=0,
    end_epoch=1,
    fail_on_nonfinite=True,
)
