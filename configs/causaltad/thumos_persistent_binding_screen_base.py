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
)

# Preserve the exact first epoch of the registered 12-epoch optimizer and
# scheduler. Only the stopping point changes.
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
