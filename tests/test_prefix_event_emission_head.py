import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


_TORCH_PROBE = subprocess.run(
    [sys.executable, "-c", "import torch"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
    timeout=20,
)
if _TORCH_PROBE.returncode != 0:
    pytest.skip("torch is unavailable in this environment", allow_module_level=True)
import torch


HEAD_PATH = (
    Path(__file__).resolve().parents[1]
    / "opentad"
    / "models"
    / "dense_heads"
    / "prefix_event_emission_head.py"
)
MODULE_NAME = "prefix_event_emission_head_under_test"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, HEAD_PATH)
HEAD_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = HEAD_MODULE
SPEC.loader.exec_module(HEAD_MODULE)
PrefixEventEmissionHead = HEAD_MODULE.PrefixEventEmissionHead


def _target(**overrides):
    values = {
        "class_target": torch.tensor([[1.0]]),
        "start_target": torch.tensor([[0.0]]),
        "ongoing_target": torch.tensor([[1.0]]),
        "end_event": torch.tensor([[0.0]]),
        "censor_mask": torch.tensor([[1.0]]),
        "completion_target": torch.tensor([[0.0]]),
        "emit_allowed": torch.tensor([[0.0]]),
        "emit_forbidden": torch.tensor([[1.0]]),
        "emit_event": torch.tensor([[0.0]]),
        "late_target": torch.tensor([[0.0]]),
    }
    values.update(overrides)
    return values


def _logits(class_value, start, ongoing, end, completion, emit):
    return {
        "class_logits": torch.tensor([[class_value]]),
        "start_logits": torch.tensor([[start]]),
        "ongoing_logits": torch.tensor([[ongoing]]),
        "end_hazard_logits": torch.tensor([[end]]),
        "completion_logits": torch.tensor([[completion]]),
        "emission_hazard_logits": torch.tensor([[emit]]),
    }


def test_endpoint_and_emission_heads_have_disjoint_parameters():
    head = PrefixEventEmissionHead(in_channels=4, num_classes=2)

    assert head.end_hazard_head.weight.data_ptr() != head.emission_hazard_head.weight.data_ptr()
    assert head.end_hazard_head.bias.data_ptr() != head.emission_hazard_head.bias.data_ptr()


def test_censored_survival_and_forbidden_emission_train_independent_heads():
    torch.manual_seed(0)
    head = PrefixEventEmissionHead(in_channels=4, num_classes=1)
    logits = head(torch.randn(1, 4))

    losses = head.losses(logits, _target())
    losses["cost"].backward()

    assert torch.isfinite(losses["end_survival_loss"])
    assert torch.isfinite(losses["emission_loss"])
    assert head.end_hazard_head.weight.grad is not None
    assert head.emission_hazard_head.weight.grad is not None
    assert head.end_hazard_head.weight.grad.data_ptr() != head.emission_hazard_head.weight.grad.data_ptr()


def test_late_target_adds_positive_delay_penalty():
    head = PrefixEventEmissionHead(in_channels=4, num_classes=1)
    logits = _logits(2.0, -2.0, -2.0, 2.0, 2.0, -4.0)

    on_time = head.losses(logits, _target(censor_mask=torch.tensor([[0.0]])))
    late = head.losses(
        logits,
        _target(
            censor_mask=torch.tensor([[0.0]]),
            completion_target=torch.tensor([[1.0]]),
            emit_forbidden=torch.tensor([[0.0]]),
            late_target=torch.tensor([[1.0]]),
        ),
    )

    assert late["delay_loss"] > on_time["delay_loss"]


def test_emission_state_is_absorbing_until_start_signal_rearms():
    head = PrefixEventEmissionHead(
        in_channels=4,
        num_classes=1,
        class_threshold=0.5,
        start_threshold=0.5,
        end_threshold=0.5,
        completion_threshold=0.5,
        emission_threshold=0.5,
    )
    state = head.initial_state(stream_key="video=v1")

    emitted, state = head.decode_step(
        _logits(5.0, 5.0, 5.0, -5.0, -5.0, -5.0),
        state,
        {
            "current_frame": 2,
            "max_raw_frame_read": 2,
            "max_cache_source_frame": 2,
        },
    )
    assert emitted == []
    assert 0 in state.active_tracks

    emitted, state = head.decode_step(
        _logits(5.0, -5.0, -5.0, 5.0, 5.0, 5.0),
        state,
        {
            "current_frame": 5,
            "max_raw_frame_read": 5,
            "max_cache_source_frame": 5,
        },
    )
    assert len(emitted) == 1
    frozen_record = emitted[0]
    assert frozen_record.start_frame == 2
    assert frozen_record.end_frame == 5
    assert frozen_record.emit_frame == 5
    assert state.ledger == [frozen_record]

    emitted, state = head.decode_step(
        _logits(5.0, 5.0, 5.0, 5.0, 5.0, 5.0),
        state,
        {
            "current_frame": 6,
            "max_raw_frame_read": 6,
            "max_cache_source_frame": 6,
        },
    )
    assert emitted == []
    assert state.ledger == [frozen_record]

    _, state = head.decode_step(
        _logits(-5.0, -5.0, -5.0, -5.0, -5.0, -5.0),
        state,
        {
            "current_frame": 7,
            "max_raw_frame_read": 7,
            "max_cache_source_frame": 7,
        },
    )
    assert state.armed[0] is True


def test_decode_rejects_future_read_provenance():
    head = PrefixEventEmissionHead(in_channels=4, num_classes=1)
    state = head.initial_state(stream_key="video=v1")

    try:
        head.decode_step(
            _logits(5.0, 5.0, 5.0, -5.0, -5.0, -5.0),
            state,
            {
                "current_frame": 4,
                "max_raw_frame_read": 5,
                "max_cache_source_frame": 4,
            },
        )
    except ValueError as exc:
        assert "future raw frame" in str(exc)
    else:
        raise AssertionError("future read provenance must fail closed")


def test_endpoint_only_policy_emits_without_completion_or_emission_gate():
    endpoint_head = PrefixEventEmissionHead(
        in_channels=4,
        num_classes=1,
        emission_policy="endpoint_only",
    )
    pceh_head = PrefixEventEmissionHead(
        in_channels=4,
        num_classes=1,
        emission_policy="pceh",
    )
    meta_start = {
        "current_frame": 2,
        "max_raw_frame_read": 2,
        "max_cache_source_frame": 2,
    }
    meta_end = {
        "current_frame": 5,
        "max_raw_frame_read": 5,
        "max_cache_source_frame": 5,
    }
    start_logits = _logits(5.0, 5.0, 5.0, -5.0, -5.0, -5.0)
    end_only_logits = _logits(5.0, -5.0, -5.0, 5.0, -5.0, -5.0)

    endpoint_state = endpoint_head.initial_state("v1")
    pceh_state = pceh_head.initial_state("v1")
    _, endpoint_state = endpoint_head.decode_step(start_logits, endpoint_state, meta_start)
    _, pceh_state = pceh_head.decode_step(start_logits, pceh_state, meta_start)

    endpoint_emitted, _ = endpoint_head.decode_step(end_only_logits, endpoint_state, meta_end)
    pceh_emitted, _ = pceh_head.decode_step(end_only_logits, pceh_state, meta_end)

    assert len(endpoint_emitted) == 1
    assert pceh_emitted == []


def test_unknown_emission_policy_fails_closed():
    with pytest.raises(ValueError, match="emission_policy"):
        PrefixEventEmissionHead(in_channels=4, num_classes=1, emission_policy="future_oracle")
