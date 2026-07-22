_base_ = ["./thumos_persistent_binding_opt_boundary_rematch.py"]

screening_contract = dict(
    optimization_variant="reserve6_boundary_monotone_calibration",
    changed_axis="add_detached_monotone_calibration_to_causal_boundary",
)
optimization_pilot_contract = dict(
    optimization_variant="reserve6_boundary_monotone_calibration",
    reference_variant="reserve6_causal_boundary_factorization",
    factorial_design="reserve6_calibration_x_boundary_2x2_v1",
    lifecycle_calibration_revision="detached_raw_positive_scale_bias_balanced_bce_v1",
)
model = dict(
    birth_calibration_loss_weight=1.0,
    alive_calibration_loss_weight=1.0,
    end_calibration_loss_weight=1.0,
    head=dict(lifecycle_calibration_mode="monotone_affine"),
)
work_dir = "exps/thumos/persistent_binding_opt_boundary_calibration_rematch_seed705"
