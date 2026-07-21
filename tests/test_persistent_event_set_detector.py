import pytest
import torch

from opentad.models.dense_heads.persistent_event_set_head import PersistentEventSetHead
from opentad.models.detectors.persistent_event_set_ontad import (
    PersistentEventSetOnlineDetector,
    _solve_linear_assignment,
)
from opentad.utils.online_protocol import ProtocolViolation
from opentad.utils.prefix_instance_schedule import build_prefix_instance_schedule


def _head(num_slots=2, **kwargs):
    torch.manual_seed(11)
    kwargs.setdefault("max_births_per_step", min(2, num_slots))
    return PersistentEventSetHead(
        in_channels=4,
        hidden_dim=8,
        num_classes=3,
        num_slots=num_slots,
        memory_size=4,
        num_heads=2,
        dropout=0.0,
        query_mode="persistent",
        start_mode="pointer",
        endpoint_mode="hazard",
        refractory_steps=1,
        **kwargs,
    )


def _meta(source_frames):
    return {
        "video_name": "video",
        "video_id": "video",
        "stream_id": "pes-test",
        "input_format": "cached_features",
        "feature_stride": 8,
        "source_frames": tuple(source_frames),
        "current_frame": int(source_frames[-1]),
    }


def _control(start=True, end=True):
    return {
        "video_id": "video",
        "chunk_index": 0,
        "is_video_start": start,
        "is_video_end": end,
    }


def test_assignment_is_globally_optimal_instead_of_pairwise_greedy():
    cost = torch.tensor([[1.0, 2.0], [2.0, 100.0]])

    assert _solve_linear_assignment(cost) == ((0, 1), (1, 0))


def test_unexpected_forward_inputs_are_rejected_as_potential_gt_taint():
    detector = PersistentEventSetOnlineDetector(head=_head(), assignment_mode="prefix")

    with pytest.raises(ProtocolViolation, match="GT or terminal taint"):
        detector(
            inputs=torch.randn(1, 4, 1),
            masks=torch.ones(1, 1, dtype=torch.bool),
            metas=[_meta((7,))],
            prefix_schedule=[()],
            stream_control=[_control()],
            return_loss=True,
            gt_segments=torch.tensor([[1.0, 7.0]]),
        )


def test_training_uses_one_first_event_endpoint_positive_and_has_gradients():
    detector = PersistentEventSetOnlineDetector(head=_head(), assignment_mode="prefix")
    source_frames = (7, 15, 23)
    schedule = build_prefix_instance_schedule(
        segments=[[2.0, 17.0]],
        labels=[1],
        decision_frames=source_frames,
        previous_frame=-1,
    )
    inputs = torch.randn(1, 4, 3, requires_grad=True)

    losses = detector(
        inputs=inputs,
        masks=torch.ones(1, 3, dtype=torch.bool),
        metas=[_meta(source_frames)],
        prefix_schedule=[schedule],
        stream_control=[_control()],
        return_loss=True,
    )
    losses["cost"].backward()

    assert torch.isfinite(losses["cost"])
    assert inputs.grad is not None
    assert detector.last_chunk_audit["end_event_frames"] == [23]
    assert detector.last_chunk_audit["endpoint_positive_count"] == 1
    assert detector.last_chunk_audit["slot_exhaustion"] == 0


def test_prefix_assignment_preserves_two_overlapping_same_class_instances():
    detector = PersistentEventSetOnlineDetector(head=_head(num_slots=2), assignment_mode="prefix")
    source_frames = (7, 15, 23)
    schedule = build_prefix_instance_schedule(
        segments=[[1.0, 18.0], [2.0, 19.0]],
        labels=[1, 1],
        decision_frames=source_frames,
        previous_frame=-1,
    )

    detector(
        inputs=torch.randn(1, 4, 3),
        masks=torch.ones(1, 3, dtype=torch.bool),
        metas=[_meta(source_frames)],
        prefix_schedule=[schedule],
        stream_control=[_control()],
        return_loss=True,
    )

    assert detector.last_chunk_audit["max_assigned_instances"] == 2
    assert detector.last_chunk_audit["slot_exhaustion"] == 0
    assert detector.last_chunk_audit["same_class_concurrent_max"] == 2


def test_slot_exhaustion_is_counted_instead_of_merging_instances():
    detector = PersistentEventSetOnlineDetector(head=_head(num_slots=1), assignment_mode="prefix")
    source_frames = (7, 15)
    schedule = build_prefix_instance_schedule(
        segments=[[1.0, 18.0], [2.0, 19.0]],
        labels=[1, 1],
        decision_frames=source_frames,
        previous_frame=-1,
    )

    detector(
        inputs=torch.randn(1, 4, 2),
        masks=torch.ones(1, 2, dtype=torch.bool),
        metas=[_meta(source_frames)],
        prefix_schedule=[schedule],
        stream_control=[_control()],
        return_loss=True,
    )

    assert detector.last_chunk_audit["slot_exhaustion"] >= 1
    assert detector.audit_slot_exhaustion_total.item() >= 1


class _ScriptedHead(PersistentEventSetHead):
    def step(self, feature, state, source_frame):
        outputs, state = super().step(feature, state, source_frame)
        end_now = int(source_frame) >= 15
        outputs.update(
            birth_logits=torch.tensor([[10.0, -10.0]], device=feature.device),
            alive_logits=torch.tensor([[10.0, -10.0]], device=feature.device),
            class_logits=torch.tensor(
                [[[0.0, 10.0, 0.0], [10.0, 0.0, 0.0]]],
                device=feature.device,
            ),
            end_hazard_logits=torch.tensor(
                [[10.0 if end_now else -10.0, -10.0]],
                device=feature.device,
            ),
            endpoint_offset=torch.zeros(1, 2, device=feature.device),
        )
        pointer = torch.full(
            (1, 2, len(outputs["memory_frames"]) + 1),
            -10.0,
            device=feature.device,
        )
        pointer[:, :, 1] = 10.0
        outputs["start_pointer_logits"] = pointer
        return outputs, state


def test_inference_writes_one_immutable_completion_without_terminal_meta_taint():
    head = _ScriptedHead(
        in_channels=4,
        hidden_dim=8,
        num_classes=3,
        num_slots=2,
        memory_size=4,
        num_heads=2,
        dropout=0.0,
        query_mode="persistent",
        start_mode="pointer",
        endpoint_mode="hazard",
        refractory_steps=2,
    )
    detector = PersistentEventSetOnlineDetector(head=head, assignment_mode="prefix").eval()
    source_frames = (7, 15, 23)

    results = detector(
        inputs=torch.randn(1, 4, 3),
        masks=torch.ones(1, 3, dtype=torch.bool),
        metas=[_meta(source_frames)],
        stream_control=[_control()],
        return_loss=False,
        ext_cls=["C0", "C1", "C2"],
    )

    rows = results["video"]
    assert len(rows) == 1
    assert rows[0]["label"] == "C1"
    assert rows[0]["end_frame"] <= rows[0]["emit_frame"]
    assert rows[0]["immutable"] is True
    assert rows[0]["latency_definition"] == "emit_time_minus_predicted_end_time"
    assert detector._stream_states == {}
    assert {"duration", "is_video_end", "total_frames"}.isdisjoint(
        detector.last_chunk_audit["model_meta_keys"]
    )


def test_chunk_scan_matches_incremental_prefixes_and_future_perturbation():
    head_full = _head().eval()
    head_step = _head().eval()
    head_step.load_state_dict(head_full.state_dict())
    full = PersistentEventSetOnlineDetector(head=head_full, assignment_mode="prefix").eval()
    step = PersistentEventSetOnlineDetector(head=head_step, assignment_mode="prefix").eval()
    inputs = torch.randn(1, 4, 4)
    perturbed = inputs.clone()
    perturbed[:, :, 2:] += 100.0
    source_frames = (7, 15, 23, 31)

    full_outputs, _ = full.scan_features(
        inputs,
        torch.ones(1, 4, dtype=torch.bool),
        _meta(source_frames),
        state=None,
    )
    perturbed_outputs, _ = full.scan_features(
        perturbed,
        torch.ones(1, 4, dtype=torch.bool),
        _meta(source_frames),
        state=None,
    )
    assert torch.allclose(full_outputs[0]["class_logits"], perturbed_outputs[0]["class_logits"])
    assert torch.allclose(full_outputs[1]["class_logits"], perturbed_outputs[1]["class_logits"])

    state = None
    step_outputs = []
    for index, frame in enumerate(source_frames):
        outputs, state = step.scan_features(
            inputs[:, :, index : index + 1],
            torch.ones(1, 1, dtype=torch.bool),
            _meta((frame,)),
            state=state,
        )
        step_outputs.extend(outputs)

    for chunked, incremental in zip(full_outputs, step_outputs):
        assert torch.allclose(chunked["class_logits"], incremental["class_logits"], atol=1e-6)
