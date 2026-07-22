_base_ = ["./thumos_persistent_binding_optimization_pilot_base.py"]

screening_contract = dict(
    optimization_variant="short_warmup_causal_transport",
    changed_axis="shared_prediction_only_causal_query_transport",
)
optimization_pilot_contract = dict(
    optimization_variant="short_warmup_causal_transport",
)
model = dict(
    trajectory_binding_mode="fixed_birth_slot",
    causal_query_transport_loss_weight=0.05,
)
work_dir = "exps/thumos/persistent_binding_opt_transport_fixed_seed705"

