_base_ = ["./thumos_persistent_binding_optimization_pilot_base.py"]

screening_contract = dict(
    optimization_variant="short_warmup_lifecycle_margin_reserve6",
    changed_axis="shared_transition_reserve_slots_4_to_6",
)
optimization_pilot_contract = dict(
    optimization_variant="short_warmup_lifecycle_margin_reserve6",
    reference_variant="short_warmup_lifecycle_margin",
    lifecycle_margin_revision="balanced_birth_alive_end_0p1x0p25_v1",
    transition_capacity_revision="hard_entry_free_4_plus_2_reserve_v2",
)
transition_capacity_contract = dict(
    resident_visible_slots=4,
    transition_birth_reserve_slots=2,
    total_slots=6,
    admission_policy="max_occupied_4_keep_entry_free_2",
    implementation_revision="hard_transition_birth_reserve_v2",
    source="frozen_full_split_census",
    released_slot_reuse="next_causal_decision",
    same_step_release_before_birth=False,
    expand_further_on_failure=False,
)
model = dict(
    trajectory_binding_mode="prefix_rematch_active_pool",
    birth_logit_margin_loss_weight=0.1,
    alive_logit_margin_loss_weight=0.1,
    end_logit_margin_loss_weight=0.1,
    head=dict(
        num_slots=6,
        transition_birth_reserve_slots=2,
    ),
)
work_dir = "exps/thumos/persistent_binding_opt_reserve_rematch_seed705"
