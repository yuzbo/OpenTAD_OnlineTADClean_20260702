from dataclasses import fields, replace
import hashlib
import json
import math

import numpy as np
import pytest
import torch

import opentad.models.detectors.persistent_trajectory_ontad as trajectory_module
from opentad.datasets.streaming_feature import StreamingFeatureDataset
from opentad.evaluations import compute_online_instance_metrics
from opentad.evaluations.online_budgeted_map import OnlineAPBudgeted
from opentad.models.dense_heads.persistent_event_set_head import (
    EventSetEmissionRecord,
    PersistentEventSetHead,
)
from opentad.models.detectors.persistent_trajectory_ontad import (
    PersistentTrajectoryOnlineDetector,
    PersistentTrajectoryRuntimeState,
    _balanced_binary_logit_margin,
    _balanced_binary_calibration_bce,
    _causal_query_transport_loss,
    _masked_bce,
    _sinkhorn_plan,
)
from opentad.utils.online_protocol import (
    ProtocolViolation,
    summarize_emission_ledger,
    validate_emission_ledger_summary,
)
from opentad.utils.prefix_instance_schedule import build_prefix_instance_schedule
from opentad.utils.prefix_trajectory_supervision import (
    PrefixTrajectorySupervisionState,
)


def _head():
    torch.manual_seed(23)
    return PersistentEventSetHead(
        in_channels=4,
        hidden_dim=8,
        num_classes=3,
        num_slots=2,
        memory_size=4,
        num_heads=2,
        dropout=0.0,
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
        birth_threshold=1.1,
        alive_threshold=1.1,
        end_threshold=1.1,
        refractory_steps=0,
        lifecycle_mode="candidate_recycle",
        candidate_confirmation_steps=1,
        max_births_per_step=2,
    )


def _detector(binding_mode="fixed_birth_slot", **kwargs):
    return PersistentTrajectoryOnlineDetector(
        head=_head(),
        trajectory_binding_mode=binding_mode,
        detach_stream_state=True,
        **kwargs,
    )


def _meta(source_frames):
    return {
        "video_name": "video",
        "video_id": "video",
        "stream_id": "fixed-rematch-contract",
        "input_format": "cached_features",
        "feature_stride": 8,
        "fps": 30.0,
        "source_frames": tuple(source_frames),
        "current_frame": int(source_frames[-1]),
    }


def test_runtime_state_schema_has_no_training_identity_or_terminal_fields():
    names = {field.name.lower() for field in fields(PersistentTrajectoryRuntimeState)}
    forbidden_fragments = {
        "instance",
        "ground_truth",
        "gt_",
        "endpoint_seen",
        "duration",
        "total_frames",
        "is_video_end",
        "eof",
        "ledger",
        "refractory",
    }

    assert all(
        fragment not in name
        for name in names
        for fragment in forbidden_fragments
    )


def test_inference_rejects_terminal_metadata_and_has_no_supervision_argument():
    detector = _detector().eval()
    inputs = torch.randn(1, 4, 1)
    masks = torch.ones(1, 1, dtype=torch.bool)

    tainted = _meta((7,))
    tainted["is_video_end"] = True
    with pytest.raises(ProtocolViolation, match="terminal|forbidden|taint"):
        detector.infer_step(inputs, masks, tainted)

    nested_taint = _meta((7,))
    nested_taint["extra"] = {"total_frames": 100}
    with pytest.raises(ProtocolViolation, match="forbidden"):
        detector.infer_step(inputs, masks, nested_taint)

    with pytest.raises((TypeError, ProtocolViolation)):
        detector.infer_step(
            inputs,
            masks,
            _meta((7,)),
            supervision_schedule=(),
        )


def test_final_emission_contains_frame_and_standard_second_coordinates():
    detector = _detector().eval()
    record = EventSetEmissionRecord(
        stream_key="stream",
        slot_id=0,
        label=1,
        score=0.8,
        start_frame=30,
        end_frame=60,
        emit_frame=66,
        max_source_frame=66,
    )

    committed, rows = detector._append_emissions(
        (),
        (record,),
        video_id="video",
        class_names=("a", "b", "c"),
        fps=30.0,
    )

    assert committed == rows
    assert rows[0]["segment"] == [1.0, 2.0]
    assert rows[0]["label"] == "b"
    assert rows[0]["video_id"] == "video"
    assert rows[0]["runtime_stream_key"] == "stream"
    assert rows[0]["sequence_id"] == 0
    assert rows[0]["latency_sec"] == pytest.approx(0.2)
    assert rows[0]["start_frame"] <= rows[0]["end_frame"] <= rows[0]["emit_frame"]


def test_single_active_instance_has_identical_fixed_and_rematch_losses():
    fixed = _detector("fixed_birth_slot").train()
    rematch = _detector("prefix_rematch_active_pool").train()
    rematch.load_state_dict(fixed.state_dict())
    frames = (7, 15, 23)
    schedule = build_prefix_instance_schedule(
        segments=[[2.0, 17.0]],
        labels=[1],
        decision_frames=frames,
        previous_frame=-1,
    )
    inputs = torch.randn(1, 4, len(frames))
    masks = torch.ones(1, len(frames), dtype=torch.bool)

    fixed_output = fixed.train_episode(inputs, masks, _meta(frames), schedule)
    rematch_output = rematch.train_episode(inputs, masks, _meta(frames), schedule)

    assert fixed_output.audit["birth_assignments"] == rematch_output.audit["birth_assignments"]
    assert fixed_output.audit["canonical_lifecycle"] == rematch_output.audit["canonical_lifecycle"]
    assert fixed_output.audit["birth_mask_trace"] == rematch_output.audit["birth_mask_trace"]
    assert fixed_output.audit["alive_mask_trace"] == rematch_output.audit["alive_mask_trace"]
    assert fixed_output.losses.keys() == rematch_output.losses.keys()
    for name in fixed_output.losses:
        assert torch.equal(fixed_output.losses[name], rematch_output.losses[name]), name


def test_start_regression_is_supervised_only_on_the_shared_birth_assignment():
    detector = _detector("prefix_rematch_active_pool").train()
    frames = (7, 15)
    schedule = build_prefix_instance_schedule(
        segments=[[2.0, 30.0]],
        labels=[1],
        decision_frames=frames,
        previous_frame=-1,
    )
    supervision = PrefixTrajectorySupervisionState(
        num_slots=2,
        mode="rematch",
    )
    head_state = detector.head.initial_state(
        torch.device("cpu"),
        torch.float32,
        "stream",
    )

    for index, frame in enumerate(frames):
        outputs, head_state = detector.head.step(
            torch.randn(1, 4),
            head_state,
            frame,
        )
        outputs["start_offset"] = torch.full_like(
            outputs["start_offset"],
            99.0,
            requires_grad=True,
        )
        transition = supervision.transition(
            schedule[index],
            detector._cost_provider(outputs, schedule[index], 8),
        )
        losses = detector._step_losses(
            outputs,
            schedule[index],
            transition,
            8,
        )
        if index == 0:
            assert transition.birth_assignments
            assert losses["start_loss"].item() > 0
        else:
            assert transition.birth_assignments == ()
            assert losses["start_loss"].item() == 0


def test_positive_weight_temperately_amplifies_rare_positive_binary_targets():
    logits = torch.zeros(1, 2)
    target = torch.tensor([[1.0, 0.0]])
    mask = torch.ones_like(target, dtype=torch.bool)

    unweighted = _masked_bce(logits, target, mask, positive_weight=1.0)
    balanced = _masked_bce(logits, target, mask, positive_weight=4.0)
    negative_only = _masked_bce(
        logits,
        torch.zeros_like(target),
        mask,
        positive_weight=4.0,
    )
    negative_reference = _masked_bce(
        logits,
        torch.zeros_like(target),
        mask,
        positive_weight=1.0,
    )

    assert balanced > unweighted
    assert torch.equal(negative_only, negative_reference)


def test_birth_logit_margin_pushes_positive_and_negative_across_zero():
    logits = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0]])
    mask = torch.ones_like(target, dtype=torch.bool)

    loss = _balanced_binary_logit_margin(
        logits,
        target,
        mask,
        margin=0.25,
    )
    loss.backward()

    assert loss.item() > 0
    assert logits.grad[0, 0].item() < 0
    assert logits.grad[0, 1].item() > 0
    no_birth = _balanced_binary_logit_margin(
        logits.detach(),
        torch.zeros_like(target),
        mask,
        margin=0.25,
    )
    assert no_birth.item() == 0


def test_sinkhorn_plan_matches_requested_lifecycle_marginals():
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    source = torch.tensor([0.7, 0.3])
    target = torch.tensor([0.4, 0.6])

    plan = _sinkhorn_plan(
        cost,
        source,
        target,
        temperature=0.2,
        iterations=50,
    )

    assert torch.allclose(plan.sum(dim=1), source, atol=1e-5)
    assert torch.allclose(plan.sum(dim=0), target, atol=1e-5)
    assert torch.isfinite(plan).all()


def test_runtime_transport_temperature_converges_with_identity_prior():
    torch.manual_seed(705)
    num_slots = 4
    off_diagonal = 1.0 - torch.eye(num_slots)
    for _ in range(128):
        cost = torch.rand(num_slots, num_slots) * 2.0
        cost = cost + 0.25 * off_diagonal
        source = torch.rand(num_slots).clamp_min(0.05)
        target = torch.rand(num_slots).clamp_min(0.05)
        source = source / source.sum()
        target = target / target.sum()

        plan = _sinkhorn_plan(
            cost,
            source,
            target,
            temperature=0.25,
            iterations=56,
        )

        assert torch.allclose(
            plan.sum(dim=1),
            source,
            atol=1e-3,
            rtol=0,
        )
        assert torch.allclose(
            plan.sum(dim=0),
            target,
            atol=1e-3,
            rtol=0,
        )
        assert torch.isfinite(plan).all()


def test_runtime_transport_uses_first_pressure_test_iteration_below_tolerance():
    cost = torch.tensor(
        [
            [0.42022443, 0.94815993, 2.08324742, 1.52870035],
            [1.45719934, 0.63024175, 0.38382852, 1.76248419],
            [1.30959499, 0.69627583, 0.13552845, 0.38356495],
            [0.65937150, 2.21395874, 1.89271379, 0.06057763],
        ]
    )
    source = torch.tensor(
        [0.02257955, 0.24191631, 0.32210892, 0.41339517]
    )
    target = torch.tensor(
        [0.06650352, 0.03992320, 0.53413475, 0.35943845]
    )

    plan_55 = _sinkhorn_plan(
        cost,
        source,
        target,
        temperature=0.25,
        iterations=55,
    )
    plan_56 = _sinkhorn_plan(
        cost,
        source,
        target,
        temperature=0.25,
        iterations=56,
    )

    def marginal_error(plan):
        return torch.maximum(
            (plan.sum(dim=1) - source).abs().max(),
            (plan.sum(dim=0) - target).abs().max(),
        )

    assert marginal_error(plan_55).item() > 1e-3
    assert marginal_error(plan_56).item() <= 1e-3


def test_batched_sinkhorn_matches_individual_transport_plans():
    torch.manual_seed(705)
    cost = torch.rand(3, 4, 4)
    source = torch.rand(3, 4).clamp_min(0.05)
    target = torch.rand(3, 4).clamp_min(0.05)
    source = source / source.sum(dim=-1, keepdim=True)
    target = target / target.sum(dim=-1, keepdim=True)

    batched = _sinkhorn_plan(
        cost,
        source,
        target,
        temperature=0.25,
        iterations=56,
    )
    individual = torch.stack(
        [
            _sinkhorn_plan(
                cost[index],
                source[index],
                target[index],
                temperature=0.25,
                iterations=56,
            )
            for index in range(cost.shape[0])
        ]
    )

    assert torch.allclose(batched, individual, atol=1e-6, rtol=0)


def test_batched_causal_transport_equals_sum_of_individual_pair_losses():
    torch.manual_seed(706)
    previous = torch.randn(3, 4, 8)
    current = torch.randn(3, 4, 8, requires_grad=True)
    previous_mass = torch.rand(3, 4)
    current_mass = torch.rand(3, 4)
    kwargs = dict(
        temperature=0.25,
        identity_cost=0.25,
        iterations=56,
        mass_floor=0.05,
    )

    batched = _causal_query_transport_loss(
        previous,
        current,
        previous_mass,
        current_mass,
        **kwargs,
    )
    individual = torch.stack(
        [
            _causal_query_transport_loss(
                previous[index : index + 1],
                current[index : index + 1],
                previous_mass[index : index + 1],
                current_mass[index : index + 1],
                **kwargs,
            )
            for index in range(previous.shape[0])
        ]
    ).sum()

    assert torch.allclose(batched, individual, atol=1e-6, rtol=0)
    batched.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


def test_disabled_transport_skips_lifecycle_mass_construction(monkeypatch):
    def unexpected_mass_construction(_outputs):
        raise AssertionError("disabled transport constructed lifecycle mass")

    monkeypatch.setattr(
        trajectory_module,
        "_predicted_lifecycle_mass",
        unexpected_mass_construction,
    )
    frames = (7, 15, 23)
    schedule = build_prefix_instance_schedule(
        segments=[[2.0, 30.0]],
        labels=[1],
        decision_frames=frames,
        previous_frame=-1,
    )
    detector = _detector(causal_query_transport_loss_weight=0.0).train()

    output = detector.train_episode(
        torch.randn(1, 4, len(frames)),
        torch.ones(1, len(frames), dtype=torch.bool),
        _meta(frames),
        schedule,
    )

    assert output.losses["causal_transport_loss"].item() == 0


def test_causal_query_transport_is_one_way_from_past_to_current():
    previous = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]],
        requires_grad=True,
    )
    current = torch.tensor(
        [[[0.8, 0.2], [0.3, 0.7]]],
        requires_grad=True,
    )
    previous_mass = torch.tensor([[0.8, 0.2]])
    current_mass = torch.tensor([[0.6, 0.4]])

    loss = _causal_query_transport_loss(
        previous,
        current,
        previous_mass,
        current_mass,
        temperature=0.1,
        identity_cost=0.25,
        iterations=30,
        mass_floor=0.05,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() >= 0
    assert previous.grad is None
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


def test_enabled_model_optimization_losses_are_finite_and_differentiable():
    frames = (7, 15, 23)
    schedule = build_prefix_instance_schedule(
        segments=[[2.0, 30.0]],
        labels=[1],
        decision_frames=frames,
        previous_frame=-1,
    )
    inputs = torch.randn(1, 4, len(frames))
    masks = torch.ones(1, len(frames), dtype=torch.bool)
    detector = _detector(
        birth_logit_margin_loss_weight=0.5,
        causal_query_transport_loss_weight=0.05,
    ).train()

    output = detector.train_episode(
        inputs,
        masks,
        _meta(frames),
        schedule,
    )
    output.losses["cost"].backward()

    assert output.losses["birth_margin_loss"].item() >= 0
    assert output.losses["causal_transport_loss"].item() >= 0
    assert torch.isfinite(output.losses["cost"])
    assert detector.last_episode_audit[
        "birth_logit_margin_loss_weight"
    ] == pytest.approx(0.5)
    assert detector.last_episode_audit[
        "causal_query_transport_loss_weight"
    ] == pytest.approx(0.05)
    assert detector.head.birth_head.weight.grad is not None


def test_lifecycle_logit_margins_cover_birth_alive_and_end_boundaries():
    frames = (7, 15, 23)
    schedule = build_prefix_instance_schedule(
        segments=[[2.0, 17.0]],
        labels=[1],
        decision_frames=frames,
        previous_frame=-1,
    )
    detector = _detector(
        birth_logit_margin_loss_weight=0.5,
        alive_logit_margin_loss_weight=0.5,
        end_logit_margin_loss_weight=0.5,
    ).train()

    output = detector.train_episode(
        torch.randn(1, 4, len(frames)),
        torch.ones(1, len(frames), dtype=torch.bool),
        _meta(frames),
        schedule,
    )
    output.losses["cost"].backward()

    for key in (
        "birth_margin_loss",
        "alive_margin_loss",
        "end_margin_loss",
    ):
        assert output.losses[key].item() > 0
    assert torch.isfinite(output.losses["cost"])
    assert detector.last_episode_audit[
        "alive_logit_margin_loss_weight"
    ] == pytest.approx(0.5)
    assert detector.last_episode_audit[
        "end_logit_margin_loss_weight"
    ] == pytest.approx(0.5)
    assert detector.head.birth_head.weight.grad is not None
    assert detector.head.alive_head.weight.grad is not None
    assert detector.head.end_head.weight.grad is not None


def test_balanced_calibration_bce_requires_a_current_positive():
    logits = torch.tensor([[0.0, 1.0]], requires_grad=True)
    mask = torch.ones_like(logits, dtype=torch.bool)

    absent = _balanced_binary_calibration_bce(
        logits,
        torch.zeros_like(logits),
        mask,
    )
    present = _balanced_binary_calibration_bce(
        logits,
        torch.tensor([[1.0, 0.0]]),
        mask,
    )

    assert absent.item() == 0
    assert present.item() > 0


def test_calibration_losses_update_only_monotone_calibrator():
    head = _head()
    head.lifecycle_calibration_mode = "monotone_affine"
    head.lifecycle_calibration_log_scale = torch.nn.Parameter(torch.zeros(3))
    head.lifecycle_calibration_bias = torch.nn.Parameter(torch.zeros(3))
    detector = PersistentTrajectoryOnlineDetector(
        head=head,
        trajectory_binding_mode="fixed_birth_slot",
        birth_loss_weight=0.0,
        alive_loss_weight=0.0,
        class_loss_weight=0.0,
        start_loss_weight=0.0,
        end_loss_weight=0.0,
        birth_calibration_loss_weight=1.0,
        alive_calibration_loss_weight=1.0,
        end_calibration_loss_weight=1.0,
    ).train()
    frames = (7, 15, 23)
    schedule = build_prefix_instance_schedule(
        segments=[[2.0, 17.0]],
        labels=[1],
        decision_frames=frames,
        previous_frame=-1,
    )

    output = detector.train_episode(
        torch.randn(1, 4, len(frames)),
        torch.ones(1, len(frames), dtype=torch.bool),
        _meta(frames),
        schedule,
    )
    calibration_cost = sum(
        output.losses[key]
        for key in (
            "birth_calibration_loss",
            "alive_calibration_loss",
            "end_calibration_loss",
        )
    )
    calibration_cost.backward()

    assert head.lifecycle_calibration_log_scale.grad is not None
    assert head.lifecycle_calibration_bias.grad is not None
    assert head.birth_head.weight.grad is None
    assert head.alive_head.weight.grad is None
    assert head.end_head.weight.grad is None
    assert detector.last_episode_audit["lifecycle_calibration_mode"] == (
        "monotone_affine"
    )


def test_detector_aligns_prior_biases_with_weighted_binary_losses():
    head = PersistentEventSetHead(
        in_channels=4,
        hidden_dim=8,
        num_classes=3,
        num_slots=2,
        memory_size=4,
        num_heads=2,
        dropout=0.0,
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
        refractory_steps=0,
        lifecycle_mode="candidate_recycle",
        candidate_confirmation_steps=1,
        max_births_per_step=2,
        birth_prior_probability=0.01,
        alive_prior_probability=0.08,
        end_prior_probability=0.06,
    )
    weights = {
        "birth": 9.0,
        "alive": 3.0,
        "end": 4.0,
    }

    detector = PersistentTrajectoryOnlineDetector(
        head=head,
        trajectory_binding_mode="fixed_birth_slot",
        birth_positive_weight=weights["birth"],
        alive_positive_weight=weights["alive"],
        end_positive_weight=weights["end"],
        prior_bias_mode="weighted_bce_stationary",
    )

    assert detector.prior_bias_mode == "weighted_bce_stationary"
    for channel in ("birth", "alive", "end"):
        prior = getattr(head, f"{channel}_prior_probability")
        expected_logit = math.log(prior / (1.0 - prior)) + math.log(
            weights[channel]
        )
        assert getattr(head, f"{channel}_head").bias.item() == pytest.approx(
            expected_logit
        )


def test_detector_rejects_unknown_prior_bias_mode():
    with pytest.raises(ValueError, match="prior_bias_mode"):
        _detector(prior_bias_mode="guess")


def test_post_birth_rematch_cost_ignores_unsupervised_start_prediction():
    detector = _detector("prefix_rematch_active_pool").train()
    frames = (7, 15)
    schedule = build_prefix_instance_schedule(
        segments=[[2.0, 30.0], [3.0, 31.0]],
        labels=[0, 1],
        decision_frames=frames,
        previous_frame=-1,
    )
    state = detector.head.initial_state(
        torch.device("cpu"),
        torch.float32,
        "stream",
    )
    _, state = detector.head.step(torch.randn(1, 4), state, frames[0])
    outputs, _ = detector.head.step(torch.randn(1, 4), state, frames[1])
    instance_ids = tuple(
        sorted(item.instance_id for item in schedule[1].active)
    )
    slot_ids = (0, 1)

    baseline = detector._cost_provider(
        outputs,
        schedule[1],
        feature_stride=8,
    )("rematch", instance_ids, slot_ids)
    changed = dict(outputs)
    changed["start_offset"] = torch.tensor(
        [[0.0, 999.0]],
        dtype=outputs["start_offset"].dtype,
    )
    perturbed = detector._cost_provider(
        changed,
        schedule[1],
        feature_stride=8,
    )("rematch", instance_ids, slot_ids)

    assert (baseline == perturbed).all()


def test_predicted_runtime_occupancy_cannot_delete_ground_truth_birth_supervision():
    detector = _detector("fixed_birth_slot").train()
    frames = (7,)
    schedule = build_prefix_instance_schedule(
        segments=[[2.0, 17.0]],
        labels=[1],
        decision_frames=frames,
        previous_frame=-1,
    )
    inputs = torch.randn(1, 4, 1)
    masks = torch.ones(1, 1, dtype=torch.bool)
    runtime = detector._initial_runtime_state(
        inputs,
        "video=video|stream=fixed-rematch-contract|input=cached_features|stride=8",
    )
    runtime = replace(
        runtime,
        slot_status=torch.full_like(runtime.slot_status, 1),
    )

    output = detector.train_episode(
        inputs,
        masks,
        _meta(frames),
        schedule,
        initial_runtime_state=runtime,
    )

    assert len(output.audit["birth_assignments"][0]) == 1
    assert output.audit["slot_exhaustion"] == 0
    assert output.audit["dropped_gt_birth_targets"] == 0
    assert output.audit["runtime_capacity_exhaustions"] == 1
    assert output.audit["gt_supervision_exhaustions"] == 0
    assert output.audit["gt_birth_runtime_entry_free_collisions"] == 1


def test_formal_supervision_exhaustion_fails_without_mutating_input_state():
    detector = _detector(
        "fixed_birth_slot",
        fail_on_supervision_exhaustion=True,
    ).train()
    frames = (7,)
    schedule = build_prefix_instance_schedule(
        segments=[[1.0, 17.0], [2.0, 18.0], [3.0, 19.0]],
        labels=[0, 1, 2],
        decision_frames=frames,
        previous_frame=-1,
    )
    supervision = PrefixTrajectorySupervisionState(
        num_slots=2,
        mode="fixed",
    )
    pristine = supervision.snapshot()

    with pytest.raises(ProtocolViolation, match="supervision exhaustion"):
        detector.train_episode(
            torch.randn(1, 4, 1),
            torch.ones(1, 1, dtype=torch.bool),
            _meta(frames),
            schedule,
            initial_supervision_state=supervision,
        )

    assert supervision.snapshot() == pristine
    assert detector.last_episode_audit == {}


def test_episode_scan_matches_incremental_cached_execution():
    episode = _detector().eval()
    incremental = _detector().eval()
    incremental.load_state_dict(episode.state_dict())
    source_frames = (7, 15, 23, 31)
    inputs = torch.randn(1, 4, len(source_frames))
    masks = torch.ones(1, len(source_frames), dtype=torch.bool)

    episode_output = episode.infer_step(inputs, masks, _meta(source_frames))
    state = None
    step_logits = []
    step_emissions = []
    for index, frame in enumerate(source_frames):
        output = incremental.infer_step(
            inputs[:, :, index : index + 1],
            masks[:, index : index + 1],
            _meta((frame,)),
            runtime_state=state,
        )
        state = output.runtime_state
        step_logits.extend(output.logits)
        step_emissions.extend(output.emissions)

    assert len(episode_output.logits) == len(step_logits)
    for batched, stepwise in zip(episode_output.logits, step_logits):
        for key in (
            "birth_logits",
            "alive_logits",
            "class_logits",
            "end_hazard_logits",
            "start_offset",
        ):
            assert torch.allclose(batched[key], stepwise[key], atol=1e-6), key
    assert episode_output.emissions == tuple(step_emissions)
    assert torch.allclose(
        episode_output.runtime_state.queries,
        state.queries,
        atol=1e-6,
    )
    assert episode_output.runtime_state.source_frames == state.source_frames


def test_future_feature_perturbation_does_not_change_earlier_prefix_logits():
    detector = _detector().eval()
    source_frames = (7, 15, 23, 31)
    inputs = torch.randn(1, 4, len(source_frames))
    perturbed = inputs.clone()
    perturbed[:, :, 2:] += 100.0

    baseline = detector.infer_step(
        inputs,
        torch.ones(1, len(source_frames), dtype=torch.bool),
        _meta(source_frames),
    )
    changed = detector.infer_step(
        perturbed,
        torch.ones(1, len(source_frames), dtype=torch.bool),
        _meta(source_frames),
    )

    for index in (0, 1):
        assert torch.allclose(
            baseline.logits[index]["class_logits"],
            changed.logits[index]["class_logits"],
            atol=1e-6,
        )


def test_adjacent_actions_survive_dataset_detector_ledger_and_metrics(
    tmp_path,
    monkeypatch,
):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.save(
        feature_dir / "adjacent.npy",
        np.arange(20, dtype=np.float32).reshape(5, 4),
    )
    annotation = {
        "database": {
            "adjacent": {
                "subset": "validation",
                "duration": 5.0,
                "frame": 5,
                "annotations": [
                    {"segment": [0.0, 2.0], "label": "A"},
                    {"segment": [2.0, 4.0], "label": "A"},
                ],
            }
        }
    }
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    class_map_path = tmp_path / "classes.txt"
    class_map_path.write_text("A\n", encoding="utf-8")
    manifest = {
        "schema": "ontad_feature_cache_v1",
        "annotation_sha256": hashlib.sha256(
            annotation_path.read_bytes()
        ).hexdigest(),
        "encoder_id": "deterministic-adjacent-action-fixture",
        "feature_policy": "packet_recent_frame",
        "timestamp_convention": "zero_based_source_frame",
        "feature_stride": 1,
        "feature_dim": 4,
        "dtype": "float32",
        "videos": {
            "adjacent": {
                "file": "adjacent.npy",
                "num_tokens": 5,
                "source_frames": [0, 1, 2, 3, 4],
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset = StreamingFeatureDataset(
        ann_file=annotation_path,
        subset_name="validation",
        class_map=class_map_path,
        data_path=feature_dir,
        cache_manifest=manifest_path,
        chunk_size=5,
        feature_stride=1,
        stream_id="adjacent-action-e2e",
        test_mode=True,
        strict_causal_control=True,
    )

    head = PersistentEventSetHead(
        in_channels=4,
        hidden_dim=8,
        num_classes=1,
        num_slots=1,
        memory_size=4,
        num_heads=2,
        dropout=0.0,
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
        birth_threshold=0.5,
        alive_threshold=0.5,
        end_threshold=0.5,
        refractory_steps=0,
        max_start_offset=1.0,
        lifecycle_mode="candidate_recycle",
        candidate_confirmation_steps=1,
        max_births_per_step=1,
    )
    original_step = head.step

    def scripted_step(feature, state, source_frame):
        outputs, next_state = original_step(feature, state, source_frame)
        frame = int(source_frame)
        outputs.update(
            birth_logits=feature.new_tensor(
                [[10.0 if frame in (0, 2, 3) else -10.0]]
            ),
            alive_logits=feature.new_tensor(
                [[10.0 if frame == 1 else -10.0]]
            ),
            class_logits=feature.new_tensor([[[10.0]]]),
            end_hazard_logits=feature.new_tensor(
                [[10.0 if frame in (2, 4) else -10.0]]
            ),
            start_offset=feature.new_tensor(
                [[1.0 if frame == 3 else 0.0]]
            ),
        )
        return outputs, next_state

    monkeypatch.setattr(head, "step", scripted_step)
    detector = PersistentTrajectoryOnlineDetector(
        head=head,
        trajectory_binding_mode="fixed_birth_slot",
        detach_stream_state=True,
    ).eval()
    sample = dataset[0]
    results = detector.forward(
        torch.from_numpy(sample["inputs"]).unsqueeze(0),
        torch.from_numpy(sample["masks"]).unsqueeze(0),
        metas=[sample["metas"]],
        stream_control=[sample["stream_control"]],
        return_loss=False,
        ext_cls=dataset.class_map,
    )

    rows = results["adjacent"]
    assert [(row["start_frame"], row["end_frame"]) for row in rows] == [
        (0, 2),
        (2, 4),
    ]
    assert [row["sequence_id"] for row in rows] == [0, 1]
    assert all(row["immutable"] is True and row["final"] is True for row in rows)
    runtime = next(iter(detector._runtime_states.values()))
    assert runtime.deferred_birth_due_to_release == 1

    summary = summarize_emission_ledger(results)
    validate_emission_ledger_summary(summary)
    assert summary["num_emissions"] == 2
    assert not any(summary["no_future"].values())

    instance_metrics = compute_online_instance_metrics(
        annotation,
        results,
        tiou_threshold=0.5,
        fps=1.0,
    )
    assert instance_metrics["counts"]["matched_ground_truth"] == 2
    assert instance_metrics["counts"]["duplicate_emissions"] == 0
    assert instance_metrics["fragmentation_rate"] == 0.0

    budgeted = OnlineAPBudgeted(
        ground_truth_filename=annotation,
        prediction_filename={"results": results},
        subset="validation",
        tiou_thresholds=[0.5],
        latency_budgets_sec=[0.0],
        allowed_videos=["adjacent"],
        fps=1.0,
        require_ledger=True,
        require_no_future=True,
    ).evaluate()
    assert budgeted["average_mOnlineAP"] == 1.0
