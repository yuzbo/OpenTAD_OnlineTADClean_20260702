_base_ = ["./thumos_persistent_binding_opt_boundary_calibration_fixed.py"]

screening_contract = dict(
    optimization_variant="reserve6_boundary_monotone_calibration_batched",
    changed_axis="episode_balanced_calibration_on_causal_boundary",
)
optimization_pilot_contract = dict(
    optimization_variant="reserve6_boundary_monotone_calibration_batched",
    reference_variant="reserve6_causal_boundary_factorization",
    factorial_design="reserve6_calibration_batched_x_boundary_2x2_v2",
    lifecycle_calibration_revision=(
        "detached_raw_positive_scale_bias_episode_balanced_bce_v2"
    ),
    lifecycle_calibration_aggregation="episode_balanced",
)
model = dict(lifecycle_calibration_aggregation="episode_balanced")
work_dir = (
    "exps/thumos/persistent_binding_opt_boundary_calibration_batched_fixed_seed705"
)
