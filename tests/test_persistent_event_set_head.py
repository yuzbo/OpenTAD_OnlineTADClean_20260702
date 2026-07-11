from dataclasses import replace

import torch

from opentad.models.dense_heads.persistent_event_set_head import (
    SLOT_ACTIVE,
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
        refractory_steps=1,
        max_endpoint_offset=8.0,
    )
    state = _state(head)
    start_outputs = {
        "birth_logits": torch.tensor([[10.0, 10.0]]),
        "alive_logits": torch.tensor([[10.0, 10.0]]),
        "class_logits": torch.tensor([[[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]]]),
        "end_hazard_logits": torch.tensor([[-10.0, -10.0]]),
        "endpoint_offset": torch.zeros(1, 2),
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
        endpoint_offset=torch.tensor([[1.0, 2.0]]),
    )
    emissions, state = head.decode_step(end_outputs, state, current_frame=23)
    assert len(emissions) == 2
    assert {record.slot_id for record in emissions} == {0, 1}
    assert {record.label for record in emissions} == {1}
    assert all(record.end_frame <= record.emit_frame for record in emissions)
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
