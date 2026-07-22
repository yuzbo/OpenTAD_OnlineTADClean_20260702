_base_ = ["./thumos_persistent_binding_screen_base.py"]

# One-epoch, feature-only mechanism pilots. These configs are not paper-result
# configs and never access the reporting split during training or diagnosis.
route_stage = "persistent_binding_feature_model_optimization_pilot"
formal_training_ready = False
screening_only = True
screening_training_ready = True

screening_contract = dict(
    schema_version="persistent_binding_model_optimization_pilot.v1",
    seed=705,
    epochs=1,
    reporting_videos_accessed=0,
    purpose="shared_model_optimization_non_degeneracy",
    epoch1_fixed_threshold_role="diagnostic_only",
    formal_fixed_threshold_gate_epoch=12,
    convergence_claim_authorized=False,
    effectiveness_claim_authorized=False,
    raw_rgb_authorized=False,
    optimization_variant="UNSET_BY_VARIANT",
    changed_axis="UNSET_BY_VARIANT",
    comparison_control="short_warmup_v1",
)

optimization_pilot_contract = dict(
    schema_version="persistent_binding_model_optimization_pilot.v1",
    optimization_variant="UNSET_BY_VARIANT",
    reference_variant="short_warmup",
    seed=705,
    epochs=1,
    input="fixed_cached_causal_features",
    fit_only=True,
    calibration_diagnosis_only=True,
    reporting_accessed=False,
    fixed_rematch_shared=True,
    threshold_search=False,
    frozen_birth_threshold=0.5,
    epoch1_fixed_threshold_role="diagnostic_only",
    formal_fixed_threshold_gate_epoch=12,
    raw_rgb_authorized=False,
    transport_numerical_revision="batched_soft_sinkhorn_0p25_iter56_v3",
    causal_query_transport_temperature=0.25,
    causal_query_transport_iterations=56,
)

model = dict(
    birth_logit_margin_loss_weight=0.0,
    birth_logit_margin=0.25,
    alive_logit_margin_loss_weight=0.0,
    alive_logit_margin=0.25,
    end_logit_margin_loss_weight=0.0,
    end_logit_margin=0.25,
    causal_query_transport_loss_weight=0.0,
    causal_query_transport_temperature=0.25,
    causal_query_transport_identity_cost=0.25,
    causal_query_transport_iterations=56,
    causal_query_transport_mass_floor=0.05,
    birth_calibration_loss_weight=0.0,
    alive_calibration_loss_weight=0.0,
    end_calibration_loss_weight=0.0,
    endpoint_start_pointer_loss_weight=0.0,
)
