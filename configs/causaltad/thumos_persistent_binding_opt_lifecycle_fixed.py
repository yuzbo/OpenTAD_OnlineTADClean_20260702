_base_ = ["./thumos_persistent_binding_optimization_pilot_base.py"]

screening_contract = dict(
    optimization_variant="short_warmup_lifecycle_margin",
    changed_axis="shared_current_step_three_head_balanced_logit_margin",
)
optimization_pilot_contract = dict(
    optimization_variant="short_warmup_lifecycle_margin",
    lifecycle_margin_revision="balanced_birth_alive_end_0p1x0p25_v1",
)
model = dict(
    trajectory_binding_mode="fixed_birth_slot",
    birth_logit_margin_loss_weight=0.1,
    alive_logit_margin_loss_weight=0.1,
    end_logit_margin_loss_weight=0.1,
)
work_dir = "exps/thumos/persistent_binding_opt_lifecycle_fixed_seed705"
