from dataclasses import replace

import pytest
import torch

from opentad.cores.train_engine import train_one_epoch
from opentad.models.dense_heads.persistent_event_set_head import (
    SLOT_FREE,
    SLOT_REFRACTORY,
    PersistentEventSetHead,
)
from opentad.models.detectors.persistent_trajectory_ontad import (
    PersistentTrajectoryOnlineDetector,
    _stream_key,
)
from opentad.utils.online_protocol import ProtocolViolation
from opentad.utils.prefix_instance_schedule import (
    PrefixInstanceTarget,
    PrefixScheduleStep,
    build_prefix_instance_schedule,
)
from opentad.utils.prefix_trajectory_supervision import PrefixTrajectorySupervisionState

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


def _meta(source_frames):
    return {
        "video_name": "video",
        "video_id": "video",
        "stream_id": "full-petal-transaction",
        "input_format": "cached_features",
        "input_provenance_digest": "a" * 64,
        "feature_stride": 8,
        "fps": 30.0,
        "source_frames": tuple(source_frames),
        "current_frame": int(source_frames[-1]),
    }


def _detector():
    return PersistentTrajectoryOnlineDetector(head=_head(), detach_stream_state=True)


def _control(*, start, end, training_targets=None):
    control = {
        "video_id": "video",
        "chunk_index": 0 if start else 1,
        "is_video_start": start,
        "is_video_end": end,
    }
    if training_targets is not None:
        control["training_targets"] = training_targets
    return control


def _schedule(frames, previous_frame):
    return build_prefix_instance_schedule(
        segments=[[2.0, 30.0]],
        labels=[1],
        decision_frames=frames,
        previous_frame=previous_frame,
    )


def test_training_forward_stages_state_across_chunks_until_atomic_commit():
    detector = _detector().train()
    first = (7,)
    second = (15,)

    detector(
        inputs=torch.randn(1, 4, 1),
        masks=torch.ones(1, 1, dtype=torch.bool),
        metas=[_meta(first)],
        prefix_schedule=[_schedule(first, -1)],
        stream_control=[_control(start=True, end=False)],
        return_loss=True,
    )
    assert detector.has_pending_online_update() is True
    assert detector._runtime_states == {}
    assert detector._supervision_states == {}
    with pytest.raises(ProtocolViolation, match="complete.*episode|terminal"):
        detector.commit_online_update()

    detector(
        inputs=torch.randn(1, 4, 1),
        masks=torch.ones(1, 1, dtype=torch.bool),
        metas=[_meta(second)],
        prefix_schedule=[_schedule(second, 7)],
        stream_control=[_control(start=False, end=True)],
        return_loss=True,
    )
    assert detector._runtime_states == {}
    detector.commit_online_update()

    assert detector.has_pending_online_update() is False
    assert detector._runtime_states == {}
    assert detector._supervision_states == {}


def test_training_rollback_discards_staged_state_and_allows_clean_restart():
    detector = _detector().train()
    frames = (7,)
    args = dict(
        inputs=torch.randn(1, 4, 1),
        masks=torch.ones(1, 1, dtype=torch.bool),
        metas=[_meta(frames)],
        prefix_schedule=[_schedule(frames, -1)],
        stream_control=[_control(start=True, end=False)],
        return_loss=True,
    )

    detector(**args)
    detector.rollback_online_update()
    assert detector.has_pending_online_update() is False
    assert detector._runtime_states == {}

    detector(**args)
    assert detector.has_pending_online_update() is True


def test_failed_episode_does_not_mutate_caller_supervision_state():
    detector = _detector().train()
    frames = (7, 15)
    valid_first = _schedule((7,), -1)[0]
    future_end = PrefixScheduleStep(
        current_frame=15,
        births=(),
        active=(),
        ends=(PrefixInstanceTarget(0, 1, 2.0, 99.0),),
    )
    supervision = PrefixTrajectorySupervisionState(
        num_slots=detector.head.num_slots,
        mode=detector.supervision_mode,
    )
    pristine = supervision.snapshot()

    with pytest.raises((ProtocolViolation, ValueError), match="future endpoint"):
        detector.train_episode(
            torch.randn(1, 4, 2),
            torch.ones(1, 2, dtype=torch.bool),
            _meta(frames),
            (valid_first, future_end),
            initial_supervision_state=supervision,
        )

    assert supervision.snapshot() == pristine


def test_masked_tail_neither_advances_state_nor_accepts_supervision():
    detector = _detector().train()
    frames = (7, 15)
    first = _schedule((7,), -1)[0]
    empty_tail = PrefixScheduleStep(current_frame=15, births=(), active=(), ends=())

    output = detector.train_episode(
        torch.randn(1, 4, 2),
        torch.tensor([[True, False]]),
        _meta(frames),
        (first, empty_tail),
    )

    assert output.runtime_state.last_decision_frame == 7
    assert output.audit["max_source_frame"] == 7
    with pytest.raises(ProtocolViolation, match="masked.*supervision"):
        detector.train_episode(
            torch.randn(1, 4, 2),
            torch.tensor([[True, False]]),
            _meta(frames),
            _schedule(frames, -1),
        )


def test_training_birth_mask_excludes_runtime_refractory_slot():
    detector = _detector().train()
    frames = (7,)
    reference = torch.randn(1, 4, 1)
    runtime = detector._initial_runtime_state(reference, _stream_key(_meta(frames)))
    runtime = replace(
        runtime,
        slot_status=torch.tensor([SLOT_REFRACTORY, SLOT_FREE]),
        refractory=torch.tensor([1, 0]),
    )

    output = detector.train_episode(
        reference,
        torch.ones(1, 1, dtype=torch.bool),
        _meta(frames),
        _schedule(frames, -1),
        initial_runtime_state=runtime,
    )

    assert output.audit["birth_mask_trace"] == ((False, True),)
    assert output.audit["birth_assignments"][0][0][1] == 1


def test_inference_rejects_training_targets_even_in_control_plane():
    detector = _detector().eval()
    with pytest.raises(ProtocolViolation, match="inference.*training targets"):
        detector(
            inputs=torch.randn(1, 4, 1),
            masks=torch.ones(1, 1, dtype=torch.bool),
            metas=[_meta((7,))],
            stream_control=[
                _control(
                    start=True,
                    end=True,
                    training_targets={"gt_segments": [[1, 7]]},
                )
            ],
            return_loss=False,
        )


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class _Scheduler:
    def __init__(self):
        self.steps = 0

    def get_last_lr(self):
        return [0.1]

    def step(self):
        self.steps += 1


class _TransactionalToy(torch.nn.Module):
    def __init__(self, *, fail_second=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.fail_second = fail_second
        self.forward_calls = 0
        self.pending = False
        self.commits = 0
        self.rollbacks = 0

    def reset_online_states(self):
        self.pending = False

    def has_pending_online_update(self):
        return self.pending

    def commit_online_update(self):
        assert self.pending
        self.pending = False
        self.commits += 1

    def rollback_online_update(self):
        self.pending = False
        self.rollbacks += 1

    def forward(self, inputs, stream_control, return_loss=False):
        assert return_loss is True
        self.forward_calls += 1
        self.pending = True
        if self.fail_second and self.forward_calls == 2:
            cost = self.weight * torch.tensor(float("nan"))
        else:
            cost = (self.weight - inputs).pow(2).mean()
        return {"cost": cost}


def _toy_loader():
    return [
        {
            "inputs": torch.tensor([1.0]),
            "stream_control": [_control(start=True, end=False)],
        },
        {
            "inputs": torch.tensor([2.0]),
            "stream_control": [_control(start=False, end=True)],
        },
    ]


def test_train_engine_steps_once_per_complete_episode():
    model = _TransactionalToy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _Scheduler()

    train_one_epoch(
        _toy_loader(),
        model,
        optimizer,
        scheduler,
        curr_epoch=0,
        logger=_Logger(),
        logging_interval=10,
    )

    assert model.forward_calls == 2
    assert model.commits == 1
    assert model.rollbacks == 0
    assert scheduler.steps == 1


def test_nonfinite_episode_rolls_back_state_and_skips_optimizer_step():
    model = _TransactionalToy(fail_second=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _Scheduler()
    before = model.weight.detach().clone()

    train_one_epoch(
        _toy_loader(),
        model,
        optimizer,
        scheduler,
        curr_epoch=0,
        logger=_Logger(),
        logging_interval=10,
    )

    assert model.commits == 0
    assert model.rollbacks == 1
    assert scheduler.steps == 0
    assert torch.equal(model.weight.detach(), before)
