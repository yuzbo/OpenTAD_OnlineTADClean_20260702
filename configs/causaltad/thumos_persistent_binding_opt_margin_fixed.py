_base_ = ["./thumos_persistent_binding_optimization_pilot_base.py"]

screening_contract = dict(
    optimization_variant="short_warmup_birth_margin",
    changed_axis="shared_birth_frame_balanced_logit_margin",
)
optimization_pilot_contract = dict(
    optimization_variant="short_warmup_birth_margin",
)
model = dict(
    trajectory_binding_mode="fixed_birth_slot",
    birth_logit_margin_loss_weight=0.5,
)
work_dir = "exps/thumos/persistent_binding_opt_margin_fixed_seed705"

