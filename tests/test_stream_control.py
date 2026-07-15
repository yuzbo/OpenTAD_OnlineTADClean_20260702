import json
from dataclasses import dataclass

import pytest

from opentad.utils.stream_control import (
    FORBIDDEN_MODEL_META_KEYS,
    MODEL_META_ALLOWLIST,
    STREAM_CONTROL_ALLOWLIST,
    TRAINING_TARGET_ALLOWLIST,
    StreamMetadataError,
    sanitize_stream_metadata,
    validate_model_meta,
    validate_stream_control,
)


def _causal_meta(**overrides):
    meta = {
        "video_name": "video-1",
        "video_id": "video-1",
        "stream_id": "route-a",
        "current_frame": 15,
        "source_frames": (7, 15),
        "fps": 30.0,
        "feature_stride": 8,
        "input_format": "cached_features",
        "processor_policy": {"resize": 224, "crop": "center"},
        "encoder_policy": {"encoder_id": "siglip-so400m", "frozen": True},
    }
    meta.update(overrides)
    return meta


def test_allowlists_keep_terminal_and_target_fields_out_of_model_meta():
    assert {
        "video_id",
        "video_name",
        "stream_id",
        "current_frame",
        "source_frames",
        "fps",
        "feature_stride",
        "input_format",
        "processor_policy",
        "encoder_policy",
    } <= MODEL_META_ALLOWLIST
    assert {"is_video_end", "reset_stream", "total_frames"} <= STREAM_CONTROL_ALLOWLIST
    assert {"annotations", "gt_segments", "prefix_schedule"} <= TRAINING_TARGET_ALLOWLIST
    assert {"is_video_end", "duration", "annotations", "prefix_schedule"} <= FORBIDDEN_MODEL_META_KEYS
    assert MODEL_META_ALLOWLIST.isdisjoint(FORBIDDEN_MODEL_META_KEYS)


def test_sanitizer_splits_causal_terminal_and_training_target_fields():
    raw = _causal_meta(
        chunk_index=2,
        is_video_start=False,
        is_video_end=True,
        reset_stream=True,
        duration=12.5,
        total_frames=375,
        video_end_frame=374,
        annotations=[{"segment": [1.0, 2.0], "label": "A"}],
        gt_segments=[[30, 60]],
        gt_labels=[1],
        prefix_schedule=[{"current_frame": 15}],
    )

    model_meta, stream_control = sanitize_stream_metadata(raw)

    assert set(model_meta) <= MODEL_META_ALLOWLIST
    assert model_meta["source_frames"] == [7, 15]
    assert "is_video_end" not in model_meta
    assert "annotations" not in model_meta
    assert stream_control["is_video_end"] is True
    assert stream_control["reset_stream"] is True
    assert stream_control["video_end_frame"] == 374
    assert stream_control["training_targets"] == {
        "annotations": [{"label": "A", "segment": [1.0, 2.0]}],
        "gt_labels": [1],
        "gt_segments": [[30, 60]],
        "prefix_schedule": [{"current_frame": 15}],
    }


@pytest.mark.parametrize(
    "tainted_policy",
    [
        {"preprocess": {"isVideoEnd": True}},
        {"audit": [{"groundTruth": {"label": 1}}]},
        {"future": {"schedule": [1, 2, 3]}},
        {"nested": {"video-duration": 12.0}},
    ],
)
def test_nested_terminal_or_target_aliases_are_rejected_from_model_values(tainted_policy):
    with pytest.raises(StreamMetadataError, match="taint"):
        sanitize_stream_metadata(_causal_meta(processor_policy=tainted_policy))


@pytest.mark.parametrize(
    "tainted_policy",
    [
        {"futureContext": {"frames": [17]}},
        {"lookaheadFrames": 8},
        {"gtHints": {"class": 1}},
        {"targetEndpoint": 31},
    ],
)
def test_compound_nested_future_and_gt_taint_is_rejected(tainted_policy):
    with pytest.raises(StreamMetadataError, match="taint"):
        validate_model_meta(_causal_meta(encoder_policy=tainted_policy))


def test_aliases_are_canonicalized_into_the_correct_plane():
    model_meta, stream_control = sanitize_stream_metadata(
        {
            "video": "video-1",
            "stream": "route-a",
            "now_frame": 9,
            "source_indices": (1, 5, 9),
            "frame_rate": 25,
            "sampling_stride": 4,
            "input_mode": "raw_frames",
            "eos": True,
            "reset_state": True,
            "frame_count": 100,
            "video_duration": 4.0,
            "gt": {"segments": [[1, 9]], "labels": [2]},
            "schedule": [{"current_frame": 9}],
        }
    )

    assert model_meta == {
        "current_frame": 9,
        "fps": 25,
        "input_format": "raw_frames",
        "sample_stride": 4,
        "source_frames": [1, 5, 9],
        "stream_id": "route-a",
        "video_id": "video-1",
    }
    assert stream_control == {
        "duration": 4.0,
        "is_video_end": True,
        "num_frames": 100,
        "reset_stream": True,
        "training_targets": {
            "ground_truth": {"labels": [2], "segments": [[1, 9]]},
            "prefix_schedule": [{"current_frame": 9}],
        },
    }


def test_conflicting_aliases_are_rejected():
    with pytest.raises(StreamMetadataError, match="conflicting values"):
        sanitize_stream_metadata(
            _causal_meta(video_id="video-a", video="video-b")
        )


def test_unknown_fields_fail_closed_in_both_planes():
    with pytest.raises(StreamMetadataError, match="unknown metadata field"):
        sanitize_stream_metadata(_causal_meta(mystery_field="future hint"))
    with pytest.raises(StreamMetadataError, match="unknown model metadata"):
        validate_model_meta({**_causal_meta(), "mystery_field": 1})
    with pytest.raises(StreamMetadataError, match="unknown stream control"):
        validate_stream_control({"is_video_end": True, "mystery_field": 1})


@pytest.mark.parametrize(
    "leak",
    [
        {"is_video_end": True},
        {"eos": True},
        {"duration": 10.0},
        {"annotations": []},
        {"groundTruth": {}},
        {"prefix_schedule": []},
    ],
)
def test_model_validator_rejects_terminal_target_and_alias_leaks(leak):
    with pytest.raises(StreamMetadataError, match="forbidden|taint"):
        validate_model_meta({**_causal_meta(), **leak})


def test_end_of_stream_reset_is_legal_only_in_control_plane():
    control = validate_stream_control(
        {
            "video_id": "video-1",
            "chunk_index": 3,
            "is_video_end": True,
            "reset_stream": True,
        }
    )

    assert control["is_video_end"] is True
    assert control["reset_stream"] is True
    with pytest.raises(StreamMetadataError, match="stream boundary"):
        validate_stream_control({"reset_stream": True})
    with pytest.raises(StreamMetadataError, match="start.*reset|reset.*start"):
        validate_stream_control(
            {"is_video_start": True, "is_video_end": False, "reset_stream": True}
        )


def test_targets_must_use_the_training_target_structure_in_control():
    with pytest.raises(StreamMetadataError, match="training_targets"):
        validate_stream_control({"annotations": []})

    validated = validate_stream_control(
        {
            "is_video_end": False,
            "training_targets": {
                "annotations": [],
                "gt_segments": [[1, 2]],
                "gt_labels": [0],
            },
        }
    )
    assert validated["training_targets"]["gt_segments"] == [[1, 2]]


def test_explicit_control_and_targets_merge_without_entering_model_meta():
    model_meta, stream_control = sanitize_stream_metadata(
        _causal_meta(),
        stream_control={"video_id": "video-1", "is_video_start": True},
        training_targets={"gt_segments": [[1, 4]], "gt_labels": [2]},
    )

    assert "gt_segments" not in model_meta
    assert stream_control["training_targets"] == {
        "gt_labels": [2],
        "gt_segments": [[1, 4]],
    }


def test_training_schedule_objects_remain_in_control_without_tainting_model_meta():
    @dataclass(frozen=True)
    class ScheduleStep:
        current_frame: int
        active_ids: tuple

    step = ScheduleStep(current_frame=15, active_ids=(1, 2))
    model_meta, stream_control = sanitize_stream_metadata(
        _causal_meta(prefix_schedule=(step,))
    )

    assert stream_control["training_targets"]["prefix_schedule"] == (step,)
    assert json.loads(json.dumps(model_meta, allow_nan=False)) == model_meta


def test_model_metadata_is_detached_json_native_and_causally_validated():
    raw = _causal_meta(source_frames=(3, 7, 11), current_frame=11)
    model_meta, _ = sanitize_stream_metadata(raw)
    raw["processor_policy"]["resize"] = 999

    assert model_meta["processor_policy"]["resize"] == 224
    assert isinstance(model_meta["source_frames"], list)
    assert json.loads(json.dumps(model_meta, allow_nan=False)) == model_meta
    assert validate_model_meta(model_meta) == model_meta

    with pytest.raises(StreamMetadataError, match="future source frame"):
        validate_model_meta(_causal_meta(source_frames=(3, 12), current_frame=11))
    for field in ("packet_end_frame", "window_end_frame"):
        with pytest.raises(StreamMetadataError, match="future source frame"):
            validate_model_meta(_causal_meta(current_frame=11, **{field: 12}))
    with pytest.raises(StreamMetadataError, match="JSON serializable"):
        sanitize_stream_metadata(_causal_meta(encoder_policy={"layers": {1, 2}}))
