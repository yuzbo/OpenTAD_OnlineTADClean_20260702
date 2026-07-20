_base_ = ["./thumos_persistent_binding_optimization_pilot_base.py"]

screening_contract = dict(
    optimization_variant="short_warmup",
    changed_axis="shared_short_warmup_only",
)
optimization_pilot_contract = dict(
    optimization_variant="short_warmup",
)
model = dict(trajectory_binding_mode="prefix_rematch_active_pool")
work_dir = "exps/thumos/persistent_binding_opt_sw_rematch_seed705"

