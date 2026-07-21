_base_ = [
    "./thumos_persistent_binding_opt_boundary_calibration_batched_fixed.py"
]

route_stage = "persistent_binding_feature_multi_epoch_calibration"
formal_training_ready = True
screening_only = False
screening_training_ready = False

multi_epoch_training_contract = dict(
    schema_version="persistent_binding_feature_multi_epoch.v1",
    variant="reserve6_boundary_monotone_calibration_batched",
    seed=705,
    epochs=12,
    checkpoint_epochs=(3, 6, 9, 12),
    initialization="seed_initialization_not_pilot_resume",
    input="fixed_cached_causal_features",
    fit_only=True,
    checkpoint_selection_split="calibration_only",
    reporting_accessed=False,
    threshold_search=False,
    frozen_birth_threshold=0.5,
    frozen_alive_threshold=0.5,
    frozen_end_threshold=0.5,
    epoch1_fixed_threshold_role="diagnostic_only",
    formal_fixed_threshold_gate_epoch=12,
    extension_to_epoch24_rule="only_if_epoch9_to_epoch12_improves_without_overfit",
    fixed_rematch_shared=True,
    raw_rgb_authorized=False,
)

workflow = dict(
    fit_only=True,
    logging_interval=50,
    checkpoint_interval=3,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=0,
    end_epoch=12,
    fail_on_nonfinite=True,
)

work_dir = (
    "exps/thumos/persistent_binding_formal12_boundary_calibration_batched_fixed_seed705"
)
