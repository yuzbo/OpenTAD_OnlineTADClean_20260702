from dataclasses import replace

import pytest
import torch

from opentad.models.dense_heads.persistent_event_set_head import (
    SLOT_ACTIVE,
    SLOT_CANDIDATE,
    SLOT_FREE,
    SLOT_REFRACTORY,
    PersistentEventSetHead,
)


def _head(**kwargs):
    return PersistentEventSetHead(
        in_channels=4,
        hidden_dim=8,
        num_classes=3,
        num_slots=2,
        memory_size=4,
        num_heads=2,
        dropout=0.0,
        **kwargs,
    ).eval()


def _state(head):
    return head.initial_state(device=torch.device("cpu"), dtype=torch.float32, stream_key="stream")


def test_fresh_queries_ignore_carried_query_state_but_persistent_queries_use_it():
    torch.manual_seed(4)
    feature = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    fresh = _head(query_mode="fresh", start_mode="scalar")
    fresh_state = _state(fresh)
    perturbed_fresh = replace(fresh_state, queries=fresh_state.queries + 10.0)
    fresh_a, _ = fresh.step(feature, fresh_state, source_frame=7)
    fresh_b, _ = fresh.step(feature, perturbed_fresh, source_frame=7)
    assert torch.allclose(fresh_a["class_logits"], fresh_b["class_logits"])

    persistent = _head(query_mode="persistent", start_mode="scalar")
    persistent_state = _state(persistent)
    perturbed_persistent = replace(persistent_state, queries=persistent_state.queries + 10.0)
    persistent_a, _ = persistent.step(feature, persistent_state, source_frame=7)
    persistent_b, _ = persistent.step(feature, perturbed_persistent, source_frame=7)
    assert not torch.allclose(persistent_a["class_logits"], persistent_b["class_logits"])


def test_pointer_target_has_explicit_before_memory_bin():
    head = _head(query_mode="persistent", start_mode="pointer")

    assert head.pointer_target((7, 15, 23), start_frame=2.0) == 0
    assert head.pointer_target((7, 15, 23), start_frame=15.0) == 2
    assert head.pointer_target((7, 15, 23), start_frame=22.0) == 3


def test_scalar_start_offset_is_decoded_in_feature_steps_not_raw_frames():
    head = _head(query_mode="persistent", start_mode="scalar")
    outputs = {"start_offset": torch.tensor([[2.0, 0.0]])}

    assert head._decode_start(outputs, slot=0, current_frame=31, feature_stride=8) == 15


def test_scalar_route_uses_bounded_offset_and_skips_pointer_branch():
    head = _head(
        query_mode="persistent",
        start_mode="scalar",
        max_start_offset=1.0,
    )
    outputs, _ = head.step(
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        _state(head),
        source_frame=7,
    )

    assert outputs["start_offset"].min().item() >= 0.0
    assert outputs["start_offset"].max().item() <= 1.0
    assert "start_pointer_logits" not in outputs
    assert head.before_memory is None


def test_monotone_lifecycle_calibration_is_identity_and_ranking_preserving():
    head = _head(
        query_mode="persistent",
        start_mode="scalar",
        lifecycle_calibration_mode="monotone_affine",
    ).train()
    raw = torch.tensor(
        [[[-2.0, 0.0, 1.0], [1.0, 2.0, 3.0]]],
        requires_grad=True,
    )

    calibrated = head.calibrate_lifecycle_logits(raw)

    assert torch.equal(calibrated, raw)
    assert torch.all(head.lifecycle_calibration_log_scale.exp() > 0)
    for channel in range(3):
        assert torch.equal(
            calibrated[..., channel].argsort(dim=1),
            raw[..., channel].argsort(dim=1),
        )


def test_lifecycle_calibration_gradient_is_isolated_from_raw_logits():
    head = _head(
        query_mode="persistent",
        start_mode="scalar",
        lifecycle_calibration_mode="monotone_affine",
    ).train()
    raw = torch.tensor(
        [[[-1.0, 0.5, 2.0], [2.0, -0.5, -1.0]]],
        requires_grad=True,
    )

    head.calibrate_lifecycle_logits(raw).sum().backward()

    assert raw.grad is None
    assert head.lifecycle_calibration_log_scale.grad is not None
    assert head.lifecycle_calibration_bias.grad is not None


def test_causal_delta_end_and_endpoint_pointer_use_only_observed_prefix():
    torch.manual_seed(9)
    head = _head(
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
        end_transition_mode="causal_delta_mlp",
        endpoint_start_mode="past_pointer",
    ).train()
    state = _state(head)

    first, state = head.step(
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        state,
        source_frame=7,
    )
    second, _ = head.step(
        torch.tensor([[0.4, 0.3, 0.2, 0.1]]),
        state,
        source_frame=15,
    )

    assert first["memory_frames"] == (7,)
    assert second["memory_frames"] == (7, 15)
    assert second["endpoint_start_pointer_logits"].shape == (1, 2, 3)
    assert max(second["memory_frames"]) <= 15
    loss = (
        second["raw_end_hazard_logits"].sum()
        + second["endpoint_start_pointer_logits"].sum()
    )
    loss.backward()
    assert head.end_transition_proj[0].weight.grad is not None
    assert head.endpoint_start_query.weight.grad is not None
    assert head.endpoint_before_memory.grad is not None


def test_endpoint_pointer_sentinel_falls_back_to_frozen_birth_start():
    head = _head(
        query_mode="persistent",
        start_mode="scalar",
        endpoint_start_mode="past_pointer",
    )
    outputs = {
        "endpoint_start_pointer_logits": torch.tensor(
            [[[10.0, -10.0, -10.0], [10.0, -10.0, -10.0]]]
        ),
        "memory_frames": (31, 39),
    }

    assert head._decode_endpoint_start(
        outputs,
        slot=0,
        fallback_start_frame=7,
        current_frame=39,
    ) == 7


def test_fit_prior_probabilities_are_encoded_exactly_in_output_biases():
    priors = dict(
        birth_prior_probability=0.01,
        alive_prior_probability=0.08,
        end_prior_probability=0.06,
    )
    head = _head(
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
        **priors,
    )

    assert torch.sigmoid(head.birth_head.bias).item() == pytest.approx(0.01)
    assert torch.sigmoid(head.alive_head.bias).item() == pytest.approx(0.08)
    assert torch.sigmoid(head.end_head.bias).item() == pytest.approx(0.06)


def test_prior_biases_can_match_weighted_bce_stationary_probabilities():
    priors = dict(
        birth_prior_probability=0.01,
        alive_prior_probability=0.08,
        end_prior_probability=0.06,
    )
    weights = dict(
        birth_positive_weight=9.0,
        alive_positive_weight=3.0,
        end_positive_weight=4.0,
    )
    head = _head(
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
        **priors,
    )

    head.initialize_prior_biases(**weights)

    for channel in ("birth", "alive", "end"):
        prior = priors[f"{channel}_prior_probability"]
        weight = weights[f"{channel}_positive_weight"]
        expected = weight * prior / (weight * prior + 1.0 - prior)
        layer = getattr(head, f"{channel}_head")
        assert torch.sigmoid(layer.bias).item() == pytest.approx(expected)


def test_weighted_prior_initialization_requires_registered_priors():
    head = _head(
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
    )

    with pytest.raises(ValueError, match="requires all binary priors"):
        head.initialize_prior_biases(birth_positive_weight=2.0)


def test_end_hazard_targets_cover_only_instance_aware_risk_slots():
    head = _head(query_mode="persistent", start_mode="pointer")
    reference = torch.zeros(1, 2)

    target, mask = head.build_end_hazard_targets(
        reference,
        at_risk_slots=(0, 1),
        end_event_slots=(1,),
    )
    assert target.tolist() == [[0.0, 1.0]]
    assert mask.tolist() == [[True, True]]

    target, mask = head.build_end_hazard_targets(reference, at_risk_slots=(), end_event_slots=())
    assert target.tolist() == [[0.0, 0.0]]
    assert mask.tolist() == [[False, False]]


def test_active_false_birth_is_released_when_alive_evidence_disappears():
    head = _head(
        query_mode="persistent",
        start_mode="scalar",
        birth_threshold=0.5,
        alive_threshold=0.5,
        end_threshold=0.5,
    )
    state = _state(head)
    outputs = {
        "birth_logits": torch.tensor([[10.0, -10.0]]),
        "alive_logits": torch.tensor([[10.0, -10.0]]),
        "class_logits": torch.tensor([[[0.0, 10.0, 0.0], [10.0, 0.0, 0.0]]]),
        "end_hazard_logits": torch.tensor([[-10.0, -10.0]]),
        "endpoint_offset": torch.zeros(1, 2),
        "start_offset": torch.zeros(1, 2),
        "memory_frames": (7,),
    }
    _, state = head.decode_step(outputs, state, current_frame=7, feature_stride=8)
    assert state.slot_status.tolist() == [SLOT_ACTIVE, SLOT_FREE]

    quiet = dict(outputs)
    quiet.update(
        birth_logits=torch.tensor([[-10.0, -10.0]]),
        alive_logits=torch.tensor([[-10.0, -10.0]]),
    )
    _, state = head.decode_step(quiet, state, current_frame=15, feature_stride=8)

    assert state.slot_status.tolist() == [SLOT_FREE, SLOT_FREE]
    assert torch.isnan(state.start_frames[0])


def test_two_same_class_slots_emit_independently_once_then_rearm():
    head = _head(
        query_mode="persistent",
        start_mode="pointer",
        birth_threshold=0.5,
        alive_threshold=0.5,
        end_threshold=0.5,
        endpoint_mode="binary",
        refractory_steps=1,
    )
    state = _state(head)
    start_outputs = {
        "birth_logits": torch.tensor([[10.0, 10.0]]),
        "alive_logits": torch.tensor([[10.0, 10.0]]),
        "class_logits": torch.tensor([[[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]]]),
        "end_hazard_logits": torch.tensor([[-10.0, -10.0]]),
        "start_pointer_logits": torch.tensor(
            [[[-10.0, 10.0, 0.0], [-10.0, 0.0, 10.0]]]
        ),
        "memory_frames": (7, 15),
    }
    emissions, state = head.decode_step(start_outputs, state, current_frame=15)
    assert emissions == []
    assert state.slot_status.tolist() == [SLOT_ACTIVE, SLOT_ACTIVE]
    assert state.start_frames.tolist() == [7.0, 15.0]

    end_outputs = dict(start_outputs)
    end_outputs.update(
        birth_logits=torch.tensor([[-10.0, -10.0]]),
        end_hazard_logits=torch.tensor([[10.0, 10.0]]),
    )
    emissions, state = head.decode_step(end_outputs, state, current_frame=23)
    assert len(emissions) == 2
    assert {record.slot_id for record in emissions} == {0, 1}
    assert {record.label for record in emissions} == {1}
    assert {record.end_frame for record in emissions} == {23}
    assert state.slot_status.tolist() == [SLOT_REFRACTORY, SLOT_REFRACTORY]

    repeated, state = head.decode_step(end_outputs, state, current_frame=31)
    assert repeated == []
    assert state.slot_status.tolist() == [SLOT_REFRACTORY, SLOT_REFRACTORY]

    quiet_outputs = dict(end_outputs)
    quiet_outputs.update(
        alive_logits=torch.tensor([[-10.0, -10.0]]),
        end_hazard_logits=torch.tensor([[-10.0, -10.0]]),
    )
    repeated, state = head.decode_step(quiet_outputs, state, current_frame=39)
    assert repeated == []
    assert state.slot_status.tolist() == [SLOT_FREE, SLOT_FREE]


def _candidate_outputs(*, birth, alive, end):
    return {
        "birth_logits": torch.tensor([birth], dtype=torch.float32),
        "alive_logits": torch.tensor([alive], dtype=torch.float32),
        "class_logits": torch.tensor(
            [[[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]]],
            dtype=torch.float32,
        ),
        "end_hazard_logits": torch.tensor([end], dtype=torch.float32),
        "start_offset": torch.zeros(1, 2),
        "memory_frames": (7,),
    }


def test_candidate_lifecycle_bounds_false_birth_and_recycles_without_refractory():
    head = _head(
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
        lifecycle_mode="candidate_recycle",
        candidate_confirmation_steps=1,
        max_births_per_step=2,
        refractory_steps=0,
    )
    state = _state(head)

    _, state = head.decode_step(
        _candidate_outputs(birth=[10.0, -10.0], alive=[-10.0, -10.0], end=[-10.0, -10.0]),
        state,
        current_frame=7,
    )
    assert state.slot_status.tolist() == [SLOT_CANDIDATE, SLOT_FREE]

    _, state = head.decode_step(
        _candidate_outputs(birth=[-10.0, -10.0], alive=[-10.0, -10.0], end=[-10.0, -10.0]),
        state,
        current_frame=15,
    )
    assert state.slot_status.tolist() == [SLOT_FREE, SLOT_FREE]
    assert state.candidate_cancellations == 1


def test_candidate_confirmation_then_end_commits_once_and_immediately_frees_slot():
    head = _head(
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
        lifecycle_mode="candidate_recycle",
        candidate_confirmation_steps=1,
        max_births_per_step=2,
        refractory_steps=0,
    )
    state = _state(head)
    _, state = head.decode_step(
        _candidate_outputs(birth=[10.0, -10.0], alive=[-10.0, -10.0], end=[-10.0, -10.0]),
        state,
        current_frame=7,
    )
    _, state = head.decode_step(
        _candidate_outputs(birth=[-10.0, -10.0], alive=[10.0, -10.0], end=[-10.0, -10.0]),
        state,
        current_frame=15,
    )
    assert state.slot_status.tolist() == [SLOT_ACTIVE, SLOT_FREE]

    emissions, state = head.decode_step(
        _candidate_outputs(birth=[-10.0, -10.0], alive=[10.0, -10.0], end=[10.0, -10.0]),
        state,
        current_frame=23,
    )
    assert len(emissions) == 1
    assert state.slot_status.tolist() == [SLOT_FREE, SLOT_FREE]

    repeated, state = head.decode_step(
        _candidate_outputs(birth=[-10.0, -10.0], alive=[-10.0, -10.0], end=[10.0, -10.0]),
        state,
        current_frame=31,
    )
    assert repeated == []
    assert state.slot_status.tolist() == [SLOT_FREE, SLOT_FREE]


def test_candidate_birth_arbitration_admits_only_frozen_per_step_budget():
    head = PersistentEventSetHead(
        in_channels=4,
        hidden_dim=8,
        num_classes=3,
        num_slots=4,
        memory_size=4,
        num_heads=2,
        dropout=0.0,
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
        lifecycle_mode="candidate_recycle",
        candidate_confirmation_steps=1,
        max_births_per_step=2,
        refractory_steps=0,
    ).eval()
    state = head.initial_state(torch.device("cpu"), torch.float32, "stream")
    outputs = {
        "birth_logits": torch.tensor([[1.0, 4.0, 3.0, 2.0]]),
        "alive_logits": torch.full((1, 4), -10.0),
        "class_logits": torch.zeros(1, 4, 3),
        "end_hazard_logits": torch.full((1, 4), -10.0),
        "start_offset": torch.zeros(1, 4),
        "memory_frames": (7,),
    }

    _, state = head.decode_step(outputs, state, current_frame=7)

    assert state.slot_status.tolist() == [
        SLOT_FREE,
        SLOT_CANDIDATE,
        SLOT_CANDIDATE,
        SLOT_FREE,
    ]
    assert state.birth_proposals == 4
    assert state.birth_admissions == 2
    assert state.arbitration_suppressions == 2


def test_binary_endpoint_has_no_trainable_offset_and_emits_at_decision_frame():
    head = _head(
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
        lifecycle_mode="candidate_recycle",
        candidate_confirmation_steps=1,
        max_births_per_step=2,
        refractory_steps=0,
    )
    outputs, _ = head.step(
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        _state(head),
        source_frame=7,
    )

    assert "endpoint_offset" not in outputs
    assert not any("endpoint_offset" in name for name, _ in head.named_parameters())


def test_binary_newborn_end_crossing_commits_once_in_the_same_step():
    head = _head(
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
        lifecycle_mode="candidate_recycle",
        candidate_confirmation_steps=1,
        max_births_per_step=2,
        refractory_steps=0,
    )
    state = _state(head)

    emissions, state = head.decode_step(
        _candidate_outputs(
            birth=[10.0, -10.0],
            alive=[-10.0, -10.0],
            end=[10.0, -10.0],
        ),
        state,
        current_frame=7,
    )

    assert len(emissions) == 1
    assert emissions[0].start_frame == emissions[0].end_frame == 7
    assert state.slot_status.tolist() == [SLOT_FREE, SLOT_FREE]
    repeated, state = head.decode_step(
        _candidate_outputs(
            birth=[-10.0, -10.0],
            alive=[-10.0, -10.0],
            end=[10.0, -10.0],
        ),
        state,
        current_frame=15,
    )
    assert repeated == []
    assert len(state.ledger) == 1


def test_release_step_birth_is_deferred_and_audited_until_next_step():
    head = _head(
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
        lifecycle_mode="candidate_recycle",
        candidate_confirmation_steps=1,
        max_births_per_step=2,
        refractory_steps=0,
    )
    state = _state(head)
    _, state = head.decode_step(
        _candidate_outputs(
            birth=[10.0, -10.0],
            alive=[-10.0, -10.0],
            end=[-10.0, -10.0],
        ),
        state,
        current_frame=7,
    )
    _, state = head.decode_step(
        _candidate_outputs(
            birth=[-10.0, -10.0],
            alive=[10.0, -10.0],
            end=[-10.0, -10.0],
        ),
        state,
        current_frame=15,
    )

    _, state = head.decode_step(
        _candidate_outputs(
            birth=[10.0, -10.0],
            alive=[-10.0, -10.0],
            end=[-10.0, -10.0],
        ),
        state,
        current_frame=23,
    )
    assert state.slot_status.tolist() == [SLOT_FREE, SLOT_FREE]
    assert state.active_abandonments == 1
    assert state.deferred_birth_due_to_release == 1

    _, state = head.decode_step(
        _candidate_outputs(
            birth=[10.0, -10.0],
            alive=[-10.0, -10.0],
            end=[-10.0, -10.0],
        ),
        state,
        current_frame=31,
    )
    assert state.slot_status.tolist() == [SLOT_CANDIDATE, SLOT_FREE]
    assert state.birth_admissions == 2


def test_all_binary_route_parameters_receive_gradient():
    head = _head(
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
        lifecycle_mode="candidate_recycle",
        candidate_confirmation_steps=1,
        max_births_per_step=2,
        refractory_steps=0,
    ).train()
    outputs, _ = head.step(
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        _state(head),
        source_frame=7,
    )
    loss = (
        outputs["birth_logits"].sum()
        + outputs["alive_logits"].sum()
        + outputs["class_logits"].sum()
        + outputs["end_hazard_logits"].sum()
        + outputs["start_offset"].sum()
    )
    loss.backward()

    missing = [name for name, parameter in head.named_parameters() if parameter.grad is None]
    assert missing == []
