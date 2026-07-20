from dataclasses import fields, replace

import pytest
import torch

from opentad.models.dense_heads.persistent_event_set_head import (
    EventSetEmissionRecord,
    PersistentEventSetHead,
)
from opentad.models.detectors.persistent_trajectory_ontad import (
    PersistentTrajectoryOnlineDetector,
    PersistentTrajectoryRuntimeState,
)
from opentad.utils.online_protocol import ProtocolViolation
from opentad.utils.prefix_instance_schedule import build_prefix_instance_schedule


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
        class_names=("a", "b", "c"),
        fps=30.0,
    )

    assert committed == rows
    assert rows[0]["segment"] == [1.0, 2.0]
    assert rows[0]["label"] == "b"
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
