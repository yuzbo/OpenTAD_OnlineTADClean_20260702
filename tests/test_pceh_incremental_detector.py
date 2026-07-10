import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
DETECTOR_PATH = ROOT / "opentad" / "models" / "detectors" / "pceh_ontad.py"
MODULE_NAME = "pceh_ontad_under_test"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, DETECTOR_PATH)
DETECTOR_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = DETECTOR_MODULE
SPEC.loader.exec_module(DETECTOR_MODULE)
PCEHOnlineDetector = DETECTOR_MODULE.PCEHOnlineDetector
ProtocolViolation = DETECTOR_MODULE.ProtocolViolation


class CountingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoded_frame_count = 0

    def forward(self, frames):
        self.encoded_frame_count += int(frames.shape[-1])
        return frames


class IdentityProjection(nn.Module):
    def forward(self, features, masks):
        return features, masks


class RecordingHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, newest):
        batch = newest.shape[0]
        zeros = newest.new_zeros((batch, 1))
        return {
            "class_logits": zeros,
            "start_logits": zeros,
            "ongoing_logits": zeros,
            "end_hazard_logits": zeros,
            "completion_logits": zeros,
            "emission_hazard_logits": zeros,
        }

    def initial_state(self, stream_key):
        return {"stream_key": stream_key}

    def decode_step(self, logits, state, meta):
        state["last_meta"] = dict(meta)
        return [], state


def _detector(cache_size=3):
    return PCEHOnlineDetector(
        backbone=CountingBackbone(),
        projection=IdentityProjection(),
        head=RecordingHead(channels=4),
        cache_size=cache_size,
        detach_stream_state=True,
    )


def _meta(start, end, is_start=False, is_end=False, video_id="v1"):
    return {
        "video_id": video_id,
        "stream_id": "strict",
        "packet_start_frame": start,
        "packet_end_frame": end,
        "is_video_start": is_start,
        "is_video_end": is_end,
    }


def test_forward_step_encodes_only_new_packet_and_bounds_cache():
    detector = _detector(cache_size=3)

    _, state, first_trace = detector.forward_step(
        new_frames=torch.randn(1, 4, 2),
        state=None,
        packet_meta=_meta(0, 2, is_start=True),
    )
    _, state, second_trace = detector.forward_step(
        new_frames=torch.randn(1, 4, 2),
        state=state,
        packet_meta=_meta(2, 4),
    )

    assert detector.backbone.encoded_frame_count == 4
    assert state.feature_cache.shape[-1] == 3
    assert state.cache_source_frames == (1, 2, 3)
    assert first_trace.encoded_source_frames == (0, 1)
    assert second_trace.encoded_source_frames == (2, 3)


def test_read_trace_comes_from_packet_and_cache_not_prediction():
    detector = _detector(cache_size=2)

    output, state, trace = detector.forward_step(
        new_frames=torch.randn(1, 4, 2),
        state=None,
        packet_meta=_meta(10, 12, is_start=True),
    )

    assert output["emissions"] == []
    assert trace.max_raw_frame_read == 11
    assert trace.max_cache_source_frame == 11
    assert state.head_state["last_meta"]["max_raw_frame_read"] == 11
    assert state.head_state["last_meta"]["max_cache_source_frame"] == 11


def test_explicit_encoded_source_frames_are_preserved():
    detector = _detector(cache_size=4)
    meta = _meta(20, 28, is_start=True)
    meta["encoded_source_frames"] = [21, 27]

    _, state, trace = detector.forward_step(
        new_frames=torch.randn(1, 4, 2),
        state=None,
        packet_meta=meta,
    )

    assert trace.encoded_source_frames == (21, 27)
    assert state.cache_source_frames == (21, 27)


def test_non_monotonic_or_cross_video_packet_fails_closed():
    detector = _detector()
    _, state, _ = detector.forward_step(
        new_frames=torch.randn(1, 4, 2),
        state=None,
        packet_meta=_meta(0, 2, is_start=True),
    )

    with pytest.raises(ProtocolViolation, match="chronological"):
        detector.forward_step(
            new_frames=torch.randn(1, 4, 2),
            state=state,
            packet_meta=_meta(1, 3),
        )
    with pytest.raises(ProtocolViolation, match="stream key mismatch"):
        detector.forward_step(
            new_frames=torch.randn(1, 4, 2),
            state=state,
            packet_meta=_meta(2, 4, video_id="v2"),
        )


def test_first_packet_requires_explicit_video_start():
    detector = _detector()

    with pytest.raises(ProtocolViolation, match="is_video_start"):
        detector.forward_step(
            new_frames=torch.randn(1, 4, 2),
            state=None,
            packet_meta=_meta(0, 2, is_start=False),
        )


def test_videomamba_optimizer_grouping_no_longer_drops_trainable_backbone_adapters():
    source = (ROOT / "opentad" / "models" / "detectors" / "mamba.py").read_text(encoding="utf-8")

    assert 'if fpn.startswith("backbone"):\n                    continue' not in source
    assert "if p.requires_grad" in source
    assert "trainable parameters" in source
