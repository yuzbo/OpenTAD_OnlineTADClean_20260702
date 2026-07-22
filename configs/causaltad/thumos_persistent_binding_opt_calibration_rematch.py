_base_ = ["./thumos_persistent_binding_opt_reserve_rematch.py"]

screening_contract = dict(
    optimization_variant="reserve6_monotone_lifecycle_calibration",
    changed_axis="detached_raw_monotone_affine_lifecycle_calibration",
)
optimization_pilot_contract = dict(
    optimization_variant="reserve6_monotone_lifecycle_calibration",
    reference_variant="short_warmup_lifecycle_margin_reserve6",
    lifecycle_calibration_revision="detached_raw_positive_scale_bias_balanced_bce_v1",
)
model = dict(
    birth_calibration_loss_weight=1.0,
    alive_calibration_loss_weight=1.0,
    end_calibration_loss_weight=1.0,
    head=dict(lifecycle_calibration_mode="monotone_affine"),
)
work_dir = "exps/thumos/persistent_binding_opt_calibration_rematch_seed705"
