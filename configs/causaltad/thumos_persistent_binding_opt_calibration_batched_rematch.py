_base_ = ["./thumos_persistent_binding_opt_calibration_rematch.py"]

screening_contract = dict(
    optimization_variant="reserve6_monotone_lifecycle_calibration_batched",
    changed_axis="episode_balanced_lifecycle_calibration_aggregation",
)
optimization_pilot_contract = dict(
    optimization_variant="reserve6_monotone_lifecycle_calibration_batched",
    reference_variant="short_warmup_lifecycle_margin_reserve6",
    factorial_design="reserve6_calibration_batched_x_boundary_2x2_v2",
    lifecycle_calibration_revision=(
        "detached_raw_positive_scale_bias_episode_balanced_bce_v2"
    ),
    lifecycle_calibration_aggregation="episode_balanced",
)
model = dict(lifecycle_calibration_aggregation="episode_balanced")
work_dir = "exps/thumos/persistent_binding_opt_calibration_batched_rematch_seed705"
