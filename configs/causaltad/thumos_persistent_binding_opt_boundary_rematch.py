_base_ = ["./thumos_persistent_binding_opt_reserve_rematch.py"]

screening_contract = dict(
    optimization_variant="reserve6_causal_boundary_factorization",
    changed_axis="previous_current_delta_end_and_endpoint_past_start",
)
optimization_pilot_contract = dict(
    optimization_variant="reserve6_causal_boundary_factorization",
    reference_variant="short_warmup_lifecycle_margin_reserve6",
    boundary_revision="qprev_qcurrent_delta_xt_end_endpoint_past_pointer_v1",
    runtime_start_fallback="birth_time_frozen_start_state",
)
model = dict(
    endpoint_start_pointer_loss_weight=1.0,
    head=dict(
        end_transition_mode="causal_delta_mlp",
        endpoint_start_mode="past_pointer",
    ),
)
work_dir = "exps/thumos/persistent_binding_opt_boundary_rematch_seed705"
