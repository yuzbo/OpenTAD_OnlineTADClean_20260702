from copy import deepcopy
from pathlib import Path

from mmengine.config import Config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "causaltad"


def _load(name):
    return Config.fromfile(CONFIG_DIR / name)


def _normalized(cfg):
    value = deepcopy(cfg.to_dict())
    value["model"].pop("trajectory_binding_mode")
    value.pop("work_dir")
    return value


def test_fixed_and_rematch_are_a_single_variable_comparison():
    fixed = _load("thumos_persistent_binding_fixed.py")
    rematch = _load("thumos_persistent_binding_rematch.py")

    assert fixed.model.trajectory_binding_mode == "fixed_birth_slot"
    assert rematch.model.trajectory_binding_mode == "prefix_rematch_active_pool"
    assert _normalized(fixed) == _normalized(rematch)


def test_seed705_screen_is_one_epoch_feature_only_and_single_variable():
    fixed = _load("thumos_persistent_binding_fixed_screen.py")
    rematch = _load("thumos_persistent_binding_rematch_screen.py")

    assert fixed.model.trajectory_binding_mode == "fixed_birth_slot"
    assert rematch.model.trajectory_binding_mode == "prefix_rematch_active_pool"
    assert _normalized(fixed) == _normalized(rematch)
    assert fixed.route_stage == "persistent_binding_feature_seed705_screen"
    assert fixed.formal_training_ready is False
    assert fixed.screening_only is True
    assert fixed.screening_training_ready is True
    assert fixed.screening_contract.seed == 705
    assert fixed.screening_contract.epochs == 1
    assert fixed.screening_contract.reporting_videos_accessed == 0
    assert fixed.screening_contract.effectiveness_claim_authorized is False
    assert fixed.screening_contract.raw_rgb_authorized is False
    assert fixed.workflow.end_epoch == 1
    assert fixed.workflow.fit_only is True
    assert fixed.workflow.val_eval_interval == -1
    assert fixed.scheduler.max_epoch == 12
    assert fixed.scheduler.warmup_epoch == 0.1
    assert fixed.screening_contract.shared_schedule_revision == "short_warmup_v1"
    assert (
        fixed.screening_contract.changed_axis
        == "shared_linear_warmup_fraction_only"
    )
    assert fixed.screening_contract.previous_warmup_epoch == 1.0
    assert fixed.screening_contract.warmup_epoch == 0.1
    assert fixed.screening_contract.peak_learning_rate == 2e-4
    assert (
        fixed.screening_contract.warmup_epoch
        == fixed.optimization_contract.warmup_epoch
        == fixed.scheduler.warmup_epoch
    )
    assert (
        fixed.screening_contract.shared_schedule_revision
        == fixed.optimization_contract.schedule_revision
    )
    assert (
        fixed.screening_contract.peak_learning_rate
        == fixed.optimization_contract.peak_learning_rate
        == fixed.optimizer.lr
    )
    assert (
        fixed.screening_contract.mean_lr_exposure_ratio_vs_full_epoch_warmup
        > 1.8
    )
    assert fixed.raw_video_finetuning is False


def test_model_optimization_pilots_are_matched_and_isolated():
    variants = {
        "sw": (
            "thumos_persistent_binding_opt_sw_fixed.py",
            "thumos_persistent_binding_opt_sw_rematch.py",
            0.0,
            0.0,
            0.0,
            0.0,
            "short_warmup",
        ),
        "margin": (
            "thumos_persistent_binding_opt_margin_fixed.py",
            "thumos_persistent_binding_opt_margin_rematch.py",
            0.5,
            0.0,
            0.0,
            0.0,
            "short_warmup_birth_margin",
        ),
        "transport": (
            "thumos_persistent_binding_opt_transport_fixed.py",
            "thumos_persistent_binding_opt_transport_rematch.py",
            0.0,
            0.0,
            0.0,
            0.05,
            "short_warmup_causal_transport",
        ),
        "lifecycle": (
            "thumos_persistent_binding_opt_lifecycle_fixed.py",
            "thumos_persistent_binding_opt_lifecycle_rematch.py",
            0.1,
            0.1,
            0.1,
            0.0,
            "short_warmup_lifecycle_margin",
        ),
    }
    for (
        fixed_name,
        rematch_name,
        birth_margin_weight,
        alive_margin_weight,
        end_margin_weight,
        transport_weight,
        variant,
    ) in variants.values():
        fixed = _load(fixed_name)
        rematch = _load(rematch_name)

        assert fixed.model.trajectory_binding_mode == "fixed_birth_slot"
        assert (
            rematch.model.trajectory_binding_mode
            == "prefix_rematch_active_pool"
        )
        assert _normalized(fixed) == _normalized(rematch)
        assert fixed.route_stage == (
            "persistent_binding_feature_model_optimization_pilot"
        )
        assert fixed.screening_contract.optimization_variant == variant
        assert fixed.optimization_pilot_contract.optimization_variant == variant
        assert (
            fixed.model.birth_logit_margin_loss_weight
            == birth_margin_weight
        )
        assert (
            fixed.model.alive_logit_margin_loss_weight
            == alive_margin_weight
        )
        assert fixed.model.end_logit_margin_loss_weight == end_margin_weight
        assert (
            fixed.model.causal_query_transport_loss_weight
            == transport_weight
        )
        assert fixed.model.causal_query_transport_temperature == 0.25
        assert fixed.model.causal_query_transport_iterations == 56
        assert fixed.model.birth_logit_margin == 0.25
        assert fixed.model.alive_logit_margin == 0.25
        assert fixed.model.end_logit_margin == 0.25
        assert (
            fixed.optimization_pilot_contract.transport_numerical_revision
            == "batched_soft_sinkhorn_0p25_iter56_v3"
        )
        assert (
            fixed.optimization_pilot_contract.causal_query_transport_temperature
            == fixed.model.causal_query_transport_temperature
        )
        assert (
            fixed.optimization_pilot_contract.causal_query_transport_iterations
            == fixed.model.causal_query_transport_iterations
        )
        assert fixed.scheduler.warmup_epoch == 0.1
        assert fixed.workflow.end_epoch == 1
        assert fixed.workflow.fit_only is True
        assert fixed.screening_contract.reporting_videos_accessed == 0
        assert fixed.optimization_pilot_contract.reporting_accessed is False
        assert fixed.optimization_pilot_contract.threshold_search is False
        assert fixed.optimization_pilot_contract.frozen_birth_threshold == 0.5
        assert (
            fixed.optimization_pilot_contract.epoch1_fixed_threshold_role
            == "diagnostic_only"
        )
        assert (
            fixed.optimization_pilot_contract.formal_fixed_threshold_gate_epoch
            == 12
        )
        assert fixed.screening_contract.convergence_claim_authorized is False
        assert fixed.optimization_pilot_contract.raw_rgb_authorized is False
        assert fixed.raw_video_finetuning is False

        if variant == "short_warmup_lifecycle_margin":
            assert (
                fixed.optimization_pilot_contract.lifecycle_margin_revision
                == "balanced_birth_alive_end_0p1x0p25_v1"
            )


def test_feature_route_is_strictly_causal_and_raw_rgb_remains_blocked():
    cfg = _load("thumos_persistent_binding_fixed.py")

    assert cfg.experiment_contract.input == "fixed_cached_causal_features"
    assert cfg.experiment_contract.runtime_state_contains_gt is False
    assert cfg.experiment_contract.future_endpoint_prediction is False
    assert cfg.experiment_contract.offline_nms is False
    assert cfg.experiment_contract.raw_video_joint_training is False
    assert cfg.raw_video_finetuning is False
    assert cfg.solver.amp is False
    assert cfg.optimization_contract.schedule_revision == "short_warmup_v1"
    assert cfg.optimization_contract.changed_axis == (
        "shared_linear_warmup_fraction_only"
    )
    assert cfg.optimization_contract.update_count_unchanged is True
    assert cfg.optimization_contract.fixed_rematch_shared is True
    assert cfg.scheduler.warmup_epoch == 0.1
    assert cfg.scheduler.max_epoch == 12
    assert cfg.optimizer.lr == 2e-4
    assert cfg.workflow.fail_on_nonfinite is True
    assert cfg.visual_training_allowed is False
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    assert cfg.inference.require_explicit_checkpoint is True
    assert cfg.workflow.fit_only is True
    assert cfg.workflow.val_eval_interval == -1
    assert cfg.model.fail_on_supervision_exhaustion is True
    assert cfg.dataset.train.strict_causal_control is True
    assert cfg.dataset.val.strict_causal_control is True
    assert cfg.dataset.test.strict_causal_control is True


def test_transition_reserve_pilot_is_matched_and_census_bounded():
    fixed = _load("thumos_persistent_binding_opt_reserve_fixed.py")
    rematch = _load("thumos_persistent_binding_opt_reserve_rematch.py")

    assert fixed.model.trajectory_binding_mode == "fixed_birth_slot"
    assert rematch.model.trajectory_binding_mode == "prefix_rematch_active_pool"
    assert _normalized(fixed) == _normalized(rematch)
    assert fixed.model.head.num_slots == 6
    assert fixed.transition_capacity_contract.resident_visible_slots == 4
    assert fixed.transition_capacity_contract.transition_birth_reserve_slots == 2
    assert fixed.transition_capacity_contract.total_slots == 6
    assert fixed.transition_capacity_contract.released_slot_reuse == (
        "next_causal_decision"
    )
    assert fixed.transition_capacity_contract.same_step_release_before_birth is False
    assert fixed.transition_capacity_contract.expand_further_on_failure is False
    assert fixed.screening_contract.changed_axis == (
        "shared_transition_reserve_slots_4_to_6"
    )
    assert fixed.model.birth_logit_margin_loss_weight == 0.1
    assert fixed.model.alive_logit_margin_loss_weight == 0.1
    assert fixed.model.end_logit_margin_loss_weight == 0.1
    assert fixed.model.causal_query_transport_loss_weight == 0.0
    assert fixed.optimization_pilot_contract.threshold_search is False
    assert fixed.optimization_pilot_contract.frozen_birth_threshold == 0.5
    assert fixed.optimization_pilot_contract.reporting_accessed is False
    assert fixed.optimization_pilot_contract.raw_rgb_authorized is False


def test_monotone_calibration_pilot_is_matched_and_keeps_threshold_frozen():
    fixed = _load("thumos_persistent_binding_opt_calibration_fixed.py")
    rematch = _load("thumos_persistent_binding_opt_calibration_rematch.py")

    assert fixed.model.trajectory_binding_mode == "fixed_birth_slot"
    assert rematch.model.trajectory_binding_mode == "prefix_rematch_active_pool"
    assert _normalized(fixed) == _normalized(rematch)
    assert fixed.model.head.num_slots == 6
    assert fixed.model.head.lifecycle_calibration_mode == "monotone_affine"
    assert fixed.model.birth_calibration_loss_weight == 1.0
    assert fixed.model.alive_calibration_loss_weight == 1.0
    assert fixed.model.end_calibration_loss_weight == 1.0
    assert fixed.model.birth_logit_margin_loss_weight == 0.1
    assert fixed.model.alive_logit_margin_loss_weight == 0.1
    assert fixed.model.end_logit_margin_loss_weight == 0.1
    assert fixed.model.causal_query_transport_loss_weight == 0.0
    assert fixed.model.head.birth_threshold == 0.5
    assert fixed.model.head.alive_threshold == 0.5
    assert fixed.model.head.end_threshold == 0.5
    assert fixed.optimization_pilot_contract.threshold_search is False
    assert fixed.optimization_pilot_contract.reporting_accessed is False
    assert fixed.optimization_pilot_contract.raw_rgb_authorized is False
    assert fixed.optimization_pilot_contract.lifecycle_calibration_revision == (
        "detached_raw_positive_scale_bias_balanced_bce_v1"
    )


def test_causal_boundary_pilot_is_matched_and_uses_only_past_start_memory():
    fixed = _load("thumos_persistent_binding_opt_boundary_fixed.py")
    rematch = _load("thumos_persistent_binding_opt_boundary_rematch.py")

    assert fixed.model.trajectory_binding_mode == "fixed_birth_slot"
    assert rematch.model.trajectory_binding_mode == "prefix_rematch_active_pool"
    assert _normalized(fixed) == _normalized(rematch)
    assert fixed.model.head.num_slots == 6
    assert fixed.model.head.end_transition_mode == "causal_delta_mlp"
    assert fixed.model.head.endpoint_start_mode == "past_pointer"
    assert fixed.model.endpoint_start_pointer_loss_weight == 1.0
    assert fixed.model.birth_logit_margin_loss_weight == 0.1
    assert fixed.model.alive_logit_margin_loss_weight == 0.1
    assert fixed.model.end_logit_margin_loss_weight == 0.1
    assert fixed.model.birth_calibration_loss_weight == 0.0
    assert fixed.model.alive_calibration_loss_weight == 0.0
    assert fixed.model.end_calibration_loss_weight == 0.0
    assert fixed.model.causal_query_transport_loss_weight == 0.0
    assert fixed.model.head.birth_threshold == 0.5
    assert fixed.model.head.alive_threshold == 0.5
    assert fixed.model.head.end_threshold == 0.5
    assert fixed.optimization_pilot_contract.runtime_start_fallback == (
        "birth_time_frozen_start_state"
    )
    assert fixed.optimization_pilot_contract.threshold_search is False
    assert fixed.optimization_pilot_contract.reporting_accessed is False
    assert fixed.optimization_pilot_contract.raw_rgb_authorized is False


def test_boundary_calibration_completes_registered_two_by_two_factorial():
    fixed = _load(
        "thumos_persistent_binding_opt_boundary_calibration_fixed.py"
    )
    rematch = _load(
        "thumos_persistent_binding_opt_boundary_calibration_rematch.py"
    )

    assert fixed.model.trajectory_binding_mode == "fixed_birth_slot"
    assert rematch.model.trajectory_binding_mode == (
        "prefix_rematch_active_pool"
    )
    assert _normalized(fixed) == _normalized(rematch)
    assert fixed.model.head.num_slots == 6
    assert fixed.model.head.lifecycle_calibration_mode == "monotone_affine"
    assert fixed.model.head.end_transition_mode == "causal_delta_mlp"
    assert fixed.model.head.endpoint_start_mode == "past_pointer"
    assert fixed.model.birth_calibration_loss_weight == 1.0
    assert fixed.model.alive_calibration_loss_weight == 1.0
    assert fixed.model.end_calibration_loss_weight == 1.0
    assert fixed.model.endpoint_start_pointer_loss_weight == 1.0
    assert fixed.model.birth_logit_margin_loss_weight == 0.1
    assert fixed.model.alive_logit_margin_loss_weight == 0.1
    assert fixed.model.end_logit_margin_loss_weight == 0.1
    assert fixed.model.causal_query_transport_loss_weight == 0.0
    assert fixed.optimization_pilot_contract.factorial_design == (
        "reserve6_calibration_x_boundary_2x2_v1"
    )
    assert fixed.optimization_pilot_contract.threshold_search is False
    assert fixed.optimization_pilot_contract.reporting_accessed is False
    assert fixed.optimization_pilot_contract.raw_rgb_authorized is False


def test_batched_calibration_v2_is_matched_and_changes_only_aggregation():
    pairs = (
        (
            "thumos_persistent_binding_opt_calibration_batched_fixed.py",
            "thumos_persistent_binding_opt_calibration_batched_rematch.py",
            False,
        ),
        (
            "thumos_persistent_binding_opt_boundary_calibration_batched_fixed.py",
            "thumos_persistent_binding_opt_boundary_calibration_batched_rematch.py",
            True,
        ),
    )
    for fixed_name, rematch_name, boundary in pairs:
        fixed = _load(fixed_name)
        rematch = _load(rematch_name)

        assert fixed.model.trajectory_binding_mode == "fixed_birth_slot"
        assert rematch.model.trajectory_binding_mode == (
            "prefix_rematch_active_pool"
        )
        assert _normalized(fixed) == _normalized(rematch)
        assert fixed.model.lifecycle_calibration_aggregation == (
            "episode_balanced"
        )
        assert fixed.model.head.lifecycle_calibration_mode == "monotone_affine"
        assert fixed.model.birth_calibration_loss_weight == 1.0
        assert fixed.model.alive_calibration_loss_weight == 1.0
        assert fixed.model.end_calibration_loss_weight == 1.0
        assert fixed.model.endpoint_start_pointer_loss_weight == float(boundary)
        assert fixed.optimization_pilot_contract.factorial_design == (
            "reserve6_calibration_batched_x_boundary_2x2_v2"
        )
        assert fixed.optimization_pilot_contract.lifecycle_calibration_revision == (
            "detached_raw_positive_scale_bias_episode_balanced_bce_v2"
        )
        assert fixed.optimization_pilot_contract.threshold_search is False
        assert fixed.optimization_pilot_contract.reporting_accessed is False
        assert fixed.optimization_pilot_contract.raw_rgb_authorized is False


def test_route_uses_candidate_recycle_without_capacity_holding_refractory():
    cfg = _load("thumos_persistent_binding_fixed.py")
    head = cfg.model.head

    assert head.lifecycle_mode == "candidate_recycle"
    assert head.candidate_confirmation_steps == 1
    assert head.max_births_per_step == 2
    assert head.refractory_steps == 0
    assert head.num_slots == 4
    assert "max_endpoint_offset" not in head
    assert head.max_start_offset == 1.0
    assert cfg.model.birth_positive_weight > 1.0
    assert cfg.model.alive_positive_weight > 1.0
    assert cfg.model.end_positive_weight > 1.0
    assert cfg.model.prior_bias_mode == "weighted_bce_stationary"
    assert (
        head.birth_prior_probability
        == cfg.supervision_balance_contract.birth_positive_rate
    )
    assert (
        head.alive_prior_probability
        == cfg.supervision_balance_contract.alive_positive_rate
    )
    assert (
        head.end_prior_probability
        == cfg.supervision_balance_contract.end_positive_rate
    )
    assert cfg.census_contract.max_gt_entry_free_deficits == 0


def test_smoke_keeps_explicit_raw_probability_biases_for_serialization():
    cfg = _load("thumos_persistent_binding_smoke.py")

    assert cfg.model.prior_bias_mode == "raw_probability"
    assert cfg.model.head.birth_prior_probability == 0.5
    assert cfg.model.head.alive_prior_probability == 0.5
    assert cfg.model.head.end_prior_probability == 0.5


def test_route_sources_do_not_reference_historical_route_labels():
    paths = (
        CONFIG_DIR / "thumos_persistent_binding_base.py",
        CONFIG_DIR / "thumos_persistent_binding_fixed.py",
        CONFIG_DIR / "thumos_persistent_binding_rematch.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)

    assert "full_petal" not in source
    assert "pes_q2" not in source
