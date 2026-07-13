from dataclasses import fields

import pytest
import torch

from opentad.models.dense_heads.persistent_event_set_head import PersistentEventSetHead
from opentad.models.detectors.persistent_trajectory_ontad import (
    PersistentTrajectoryOnlineDetector,
    PersistentTrajectoryRuntimeState,
)
from opentad.utils.online_protocol import ProtocolViolation
from opentad.utils.prefix_instance_schedule import build_prefix_instance_schedule
from opentad.utils.immutable_event_ledger import verify_rows


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
        refractory_steps=1,
    )


def _detector(binding_mode="fixed_birth_slot"):
    return PersistentTrajectoryOnlineDetector(
        head=_head(),
        trajectory_binding_mode=binding_mode,
        detach_stream_state=True,
    )


def _meta(source_frames):
    return {
        "video_name": "video",
        "video_id": "video",
        "stream_id": "full-petal-contract",
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

    with pytest.raises((TypeError, ProtocolViolation)):
        detector.infer_step(
            inputs,
            masks,
            _meta((7,)),
            supervision_schedule=(),
        )


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


class _ScriptedTrajectoryHead(PersistentEventSetHead):
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
            start_offset=torch.ones(1, 2, device=feature.device),
        )
        return outputs, state


def test_formal_emission_is_hash_chained_and_standard_evaluator_ready():
    head = _ScriptedTrajectoryHead(
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
        refractory_steps=1,
    )
    detector = PersistentTrajectoryOnlineDetector(head=head).eval()
    source_frames = (7, 15)

    output = detector.infer_step(
        torch.randn(1, 4, 2),
        torch.ones(1, 2, dtype=torch.bool),
        _meta(source_frames),
        class_names=("C0", "C1", "C2"),
    )

    assert len(output.emissions) == 1
    row = output.emissions[0]
    assert row["immutable"] is True
    assert row["label"] == "C1"
    assert row["segment"] == [row["start_frame"] / 30.0, row["end_frame"] / 30.0]
    assert row["emit_time_sec"] == row["emit_frame"] / 30.0
    assert row["source_time_sec"] == row["source_frame"] / 30.0
    assert row["source_frame"] <= row["emit_frame"]
    assert row["end_frame"] <= row["emit_frame"]
    assert row["sequence"] == 0
    assert len(row["row_hash"]) == 64
    assert verify_rows(output.runtime_state.ledger_rows).count == 1
