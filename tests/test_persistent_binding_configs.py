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


def test_three_model_optimization_pilots_are_matched_and_isolated():
    variants = {
        "sw": (
            "thumos_persistent_binding_opt_sw_fixed.py",
            "thumos_persistent_binding_opt_sw_rematch.py",
            0.0,
            0.0,
            "short_warmup",
        ),
        "margin": (
            "thumos_persistent_binding_opt_margin_fixed.py",
            "thumos_persistent_binding_opt_margin_rematch.py",
            0.5,
            0.0,
            "short_warmup_birth_margin",
        ),
        "transport": (
            "thumos_persistent_binding_opt_transport_fixed.py",
            "thumos_persistent_binding_opt_transport_rematch.py",
            0.0,
            0.05,
            "short_warmup_causal_transport",
        ),
    }
    for (
        fixed_name,
        rematch_name,
        margin_weight,
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
        assert fixed.model.birth_logit_margin_loss_weight == margin_weight
        assert (
            fixed.model.causal_query_transport_loss_weight
            == transport_weight
        )
        assert fixed.model.causal_query_transport_temperature == 0.25
        assert fixed.model.causal_query_transport_iterations == 56
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
        assert fixed.optimization_pilot_contract.raw_rgb_authorized is False
        assert fixed.raw_video_finetuning is False


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
