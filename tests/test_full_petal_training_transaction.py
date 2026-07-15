from dataclasses import replace
import hashlib
import json

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
from opentad.utils.fixed_step_profile import FixedStepProfiler
from opentad.utils.full_petal_runtime_attestation import issue_runtime_session
from opentad.utils.full_petal_training_evidence import (
    OptimizerEventTraceRecorder,
    VisualParameterEventRecorder,
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


def test_stream_key_uses_structured_identity_without_delimiter_collisions():
    left = _stream_key(
        {
            "video_id": "x|stream=y",
            "stream_id": "z",
            "input_format": "cached",
            "feature_stride": 8,
        }
    )
    right = _stream_key(
        {
            "video_id": "x",
            "stream_id": "y|stream=z",
            "input_format": "cached",
            "feature_stride": 8,
        }
    )

    assert left != right
    assert left.startswith("stream:") and len(left) == len("stream:") + 64


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
    def __init__(self, *, fail_step=False):
        self.steps = 0
        self.fail_step = fail_step

    def get_last_lr(self):
        return [0.1]

    def step(self):
        self.steps += 1
        if self.fail_step:
            raise RuntimeError("injected scheduler failure")

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.steps = int(state["steps"])


class _TransactionalToy(torch.nn.Module):
    def __init__(self, *, fail_second=False, fail_commit=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.fail_second = fail_second
        self.fail_commit = fail_commit
        self.forward_calls = 0
        self.pending = False
        self.commits = 0
        self.rollbacks = 0

    def reset_online_states(self):
        self.pending = False

    def has_pending_online_update(self):
        return self.pending

    def snapshot_online_update(self):
        return {"pending": self.pending}

    def restore_online_update(self, snapshot):
        self.pending = bool(snapshot["pending"])

    def commit_online_update(self):
        assert self.pending
        if self.fail_commit:
            raise RuntimeError("injected commit failure")
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


def _two_episode_loader():
    return _toy_loader() + [
        {
            "inputs": torch.tensor([3.0]),
            "stream_control": [_control(start=True, end=False)],
        },
        {
            "inputs": torch.tensor([4.0]),
            "stream_control": [_control(start=False, end=True)],
        },
    ]


class _ProfileBackend:
    def __init__(self):
        self.synchronizations = 0
        self.peak_resets = 0

    def synchronize(self):
        self.synchronizations += 1

    def reset_peak_memory(self):
        self.peak_resets += 1

    def peak_memory_bytes(self):
        return 4096


class _StatefulScaler:
    def __init__(self, *, fail_at=None):
        self.calls = {"unscale": 0, "step": 0, "update": 0}
        self.fail_at = fail_at

    def scale(self, value):
        return value

    def unscale_(self, optimizer):
        del optimizer
        self.calls["unscale"] += 1
        if self.fail_at == "unscale":
            raise RuntimeError("injected scaler unscale failure")

    def step(self, optimizer):
        self.calls["step"] += 1
        optimizer.step()
        if self.fail_at == "step":
            raise RuntimeError("injected scaler step failure")

    def update(self):
        self.calls["update"] += 1
        if self.fail_at == "update":
            raise RuntimeError("injected scaler update failure")

    def state_dict(self):
        return dict(self.calls)

    def load_state_dict(self, state):
        self.calls = dict(state)


class _OptimizerEventRecorder:
    def __init__(self, *, fail_record=False):
        self.calls = []
        self.fail_record = fail_record
        self.active_boundary = None

    def begin_optimizer_boundary(self):
        assert self.active_boundary is None
        self.active_boundary = {
            "optimizer_step_completed": False,
            "transaction_commit_completed": False,
        }
        return self.active_boundary

    def execute_optimizer_step(self, proof, optimizer, *, scaler=None):
        assert proof is self.active_boundary
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        proof["optimizer_step_completed"] = True

    def commit_online_transaction(self, proof, transaction):
        assert proof is self.active_boundary
        assert proof["optimizer_step_completed"] is True
        transaction.commit_online_update()
        proof["transaction_commit_completed"] = True

    def abort_optimizer_boundary(self, proof):
        if self.active_boundary is not None:
            assert proof is self.active_boundary
            self.active_boundary = None

    def record(self, **event):
        proof = event.pop("boundary_proof")
        assert proof is self.active_boundary
        assert proof == {
            "optimizer_step_completed": True,
            "transaction_commit_completed": True,
        }
        self.calls.append(dict(event))
        self.active_boundary = None
        if self.fail_record:
            raise RuntimeError("injected recorder failure")
        return {"event_id": f"event-{len(self.calls) - 1}"}

    def state_dict(self):
        assert self.active_boundary is None
        return {"calls": list(self.calls)}

    def load_state_dict(self, state):
        self.calls = list(state["calls"])
        self.active_boundary = None


class _FaultAfterStepAdamW(torch.optim.AdamW):
    def step(self, closure=None):
        result = super().step(closure)
        raise RuntimeError("injected optimizer failure")


class _StatefulEma(torch.nn.Module):
    def __init__(self, model, *, fail_update=False):
        super().__init__()
        self.register_buffer("shadow", model.weight.detach().clone())
        self.fail_update = fail_update

    def update(self, model):
        self.shadow.copy_(model.weight.detach())
        if self.fail_update:
            raise RuntimeError("injected EMA failure")


def _authenticated_optimizer_recorder(runtime_session):
    timestamps = iter((10.0, 11.0))
    return OptimizerEventTraceRecorder(
        precision="fp32",
        effective_batch_size=1,
        world_size=1,
        optimizer_config_sha256="1" * 64,
        scheduler_config_sha256="2" * 64,
        data_order_sha256="3" * 64,
        loss_normalization_sha256="4" * 64,
        runtime_session=runtime_session,
        clock=lambda: next(timestamps),
        peak_memory_reader=lambda: 0,
    )


class _FaultingVisualRecorder(VisualParameterEventRecorder):
    def record_after(self, optimizer_event_id, model):
        super().record_after(optimizer_event_id, model)
        raise RuntimeError("injected visual evidence failure")


def _state_digest(value):
    digest = hashlib.sha256()

    def update(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode("utf-8"))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                update(child)
        else:
            digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


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


def test_train_engine_records_one_successful_event_per_complete_episode():
    model = _TransactionalToy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _Scheduler()
    recorder = _OptimizerEventRecorder()

    train_one_epoch(
        _toy_loader(),
        model,
        optimizer,
        scheduler,
        curr_epoch=4,
        logger=_Logger(),
        logging_interval=10,
        optimizer_event_recorder=recorder,
    )

    assert recorder.calls == [
        {
            "epoch": 4,
            "episode_id": "video",
            "input_tokens": 2,
            "skipped": False,
        }
    ]


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


def test_nonfinite_episode_blocks_authenticated_optimizer_evidence():
    model = _TransactionalToy(fail_second=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _Scheduler()
    recorder = _OptimizerEventRecorder()

    with pytest.raises(RuntimeError, match="cannot continue after a skipped"):
        train_one_epoch(
            _toy_loader(),
            model,
            optimizer,
            scheduler,
            curr_epoch=2,
            logger=_Logger(),
            logging_interval=10,
            optimizer_event_recorder=recorder,
        )

    assert recorder.calls == []


def test_commit_failure_restores_all_parameter_optimizer_scheduler_and_scaler_mutation():
    model = _TransactionalToy(fail_commit=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = _Scheduler()
    scaler = _StatefulScaler()
    recorder = _OptimizerEventRecorder()
    before = {
        "model": _state_digest(model.state_dict()),
        "optimizer": _state_digest(optimizer.state_dict()),
        "scheduler": scheduler.steps,
        "scaler": _state_digest(scaler.state_dict()),
    }

    with pytest.raises(RuntimeError, match="injected commit failure"):
        train_one_epoch(
            _toy_loader(),
            model,
            optimizer,
            scheduler,
            curr_epoch=0,
            logger=_Logger(),
            logging_interval=10,
            scaler=scaler,
            optimizer_event_recorder=recorder,
        )

    after = {
        "model": _state_digest(model.state_dict()),
        "optimizer": _state_digest(optimizer.state_dict()),
        "scheduler": scheduler.steps,
        "scaler": _state_digest(scaler.state_dict()),
    }
    assert after == before
    assert model.rollbacks == 1
    assert model.pending is False
    assert recorder.calls == []


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("unscale", "injected scaler unscale failure"),
        ("clip", "injected gradient clip failure"),
        ("optimizer", "injected optimizer failure"),
        ("scaler_step", "injected scaler step failure"),
        ("scheduler", "injected scheduler failure"),
        ("recorder", "injected recorder failure"),
        ("ema", "injected EMA failure"),
    ],
)
def test_every_post_gradient_failure_restores_the_complete_training_transaction(
    monkeypatch, fault, message
):
    model = _TransactionalToy()
    optimizer_cls = _FaultAfterStepAdamW if fault == "optimizer" else torch.optim.AdamW
    optimizer = optimizer_cls(model.parameters(), lr=0.1)
    scheduler = _Scheduler(fail_step=fault == "scheduler")
    scaler = (
        _StatefulScaler(
            fail_at=(
                "unscale"
                if fault == "unscale"
                else "step" if fault == "scaler_step" else None
            )
        )
        if fault in {"unscale", "scaler_step"}
        else None
    )
    recorder = _OptimizerEventRecorder(fail_record=fault == "recorder")
    model_ema = _StatefulEma(model, fail_update=fault == "ema")
    if fault == "clip":
        def fail_clip(parameters, max_norm):
            del max_norm
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(0.5)
            raise RuntimeError("injected gradient clip failure")

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fail_clip)

    before = {
        "model": _state_digest(model.state_dict()),
        "optimizer": _state_digest(optimizer.state_dict()),
        "scheduler": _state_digest(scheduler.state_dict()),
        "scaler": None if scaler is None else _state_digest(scaler.state_dict()),
        "recorder": _state_digest(recorder.state_dict()),
        "ema": _state_digest(model_ema.state_dict()),
    }

    with pytest.raises(RuntimeError, match=message):
        train_one_epoch(
            _toy_loader(),
            model,
            optimizer,
            scheduler,
            curr_epoch=0,
            logger=_Logger(),
            logging_interval=10,
            scaler=scaler,
            clip_grad_l2norm=1.0 if fault == "clip" else -1,
            optimizer_event_recorder=recorder,
            model_ema=model_ema,
        )

    after = {
        "model": _state_digest(model.state_dict()),
        "optimizer": _state_digest(optimizer.state_dict()),
        "scheduler": _state_digest(scheduler.state_dict()),
        "scaler": None if scaler is None else _state_digest(scaler.state_dict()),
        "recorder": _state_digest(recorder.state_dict()),
        "ema": _state_digest(model_ema.state_dict()),
    }
    assert after == before
    assert model.commits == (1 if fault == "recorder" else 0)
    assert model.rollbacks == 1
    assert model.pending is False


def test_training_produces_event_bound_visual_parameter_evidence():
    model = _TransactionalToy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _Scheduler()
    runtime_session = issue_runtime_session()
    optimizer_recorder = _authenticated_optimizer_recorder(runtime_session)
    visual_recorder = VisualParameterEventRecorder(
        model,
        optimizer,
        parameter_prefixes=("weight",),
        runtime_session=runtime_session,
    )

    train_one_epoch(
        _toy_loader(),
        model,
        optimizer,
        scheduler,
        curr_epoch=0,
        logger=_Logger(),
        logging_interval=10,
        optimizer_event_recorder=optimizer_recorder,
        visual_parameter_event_recorder=visual_recorder,
    )

    assert len(optimizer_recorder.events) == 1
    assert len(visual_recorder.events) == 1
    event = visual_recorder.events[0]
    assert event["optimizer_event_id"] == optimizer_recorder.events[0]["event_id"]
    assert event["parameter_name"] == "weight"
    assert event["gradient_finite"] is True
    assert event["gradient_norm"] > 0.0
    assert event["delta_norm"] > 0.0
    assert event["before_sha256"] != event["after_sha256"]


def test_visual_evidence_failure_rolls_back_optimizer_and_both_evidence_streams():
    model = _TransactionalToy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = _Scheduler()
    runtime_session = issue_runtime_session()
    optimizer_recorder = _authenticated_optimizer_recorder(runtime_session)
    visual_recorder = _FaultingVisualRecorder(
        model,
        optimizer,
        parameter_prefixes=("weight",),
        runtime_session=runtime_session,
    )
    before = {
        "model": _state_digest(model.state_dict()),
        "optimizer": _state_digest(optimizer.state_dict()),
        "scheduler": _state_digest(scheduler.state_dict()),
    }

    with pytest.raises(RuntimeError, match="injected visual evidence failure"):
        train_one_epoch(
            _toy_loader(),
            model,
            optimizer,
            scheduler,
            curr_epoch=0,
            logger=_Logger(),
            logging_interval=10,
            optimizer_event_recorder=optimizer_recorder,
            visual_parameter_event_recorder=visual_recorder,
        )

    assert {
        "model": _state_digest(model.state_dict()),
        "optimizer": _state_digest(optimizer.state_dict()),
        "scheduler": _state_digest(scheduler.state_dict()),
    } == before
    assert optimizer_recorder.events == ()
    assert visual_recorder.events == ()
    assert model.commits == 1
    assert model.rollbacks == 1


def test_fixed_step_profile_stops_only_after_committed_episode_boundary():
    model = _TransactionalToy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _Scheduler()
    backend = _ProfileBackend()
    timestamps = iter((10.0, 12.0))
    profiler = FixedStepProfiler(
        0,
        1,
        backend=backend,
        clock=lambda: next(timestamps),
    )
    optimizer_recorder = _authenticated_optimizer_recorder(issue_runtime_session())

    stats = train_one_epoch(
        _two_episode_loader(),
        model,
        optimizer,
        scheduler,
        curr_epoch=0,
        logger=_Logger(),
        logging_interval=10,
        fixed_step_profiler=profiler,
        optimizer_event_recorder=optimizer_recorder,
    )

    assert stats == {
        "optimizer_events": 1,
        "successful_optimizer_events": 1,
        "skipped_optimizer_events": 0,
        "fixed_step_profile_complete": True,
    }
    assert model.forward_calls == 2
    assert model.commits == 1
    assert model.rollbacks == 0
    assert model.pending is False
    assert scheduler.steps == 1
    assert backend.synchronizations == 2
    assert backend.peak_resets == 1
    assert profiler.measurements()["total_optimizer_events"] == 1
