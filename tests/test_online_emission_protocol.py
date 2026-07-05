from types import SimpleNamespace

import pytest

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _torch_or_skip():
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("torch import timed out in a subprocess")
    if probe.returncode != 0:
        pytest.skip(f"torch import failed in a subprocess with exit code {probe.returncode}")
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"torch is unavailable in this environment: {exc}")

    return torch


def test_streaming_safe_config_disables_video_level_sliding_window_nms():
    from opentad.utils.online_protocol import resolve_sliding_window_for_post_processing

    post_cfg = SimpleNamespace(streaming_safe_emission=True, sliding_window=True)

    assert resolve_sliding_window_for_post_processing(post_cfg, dataset_is_sliding_window=True) is False


def test_online_emitter_filters_future_segments_and_records_latency():
    from opentad.utils.online_protocol import GridSpec, OnlineCandidate, OnlineEmitter, OnlineState

    grid = GridSpec(fps=30.0, snippet_stride=2, window_start_frame=100, offset_frames=0)
    state = OnlineState(video_name="v1")
    emitter = OnlineEmitter(grid_spec=grid, score_threshold=0.1, nms_iou_threshold=0.5, latency_frames=0)
    candidates = [
        OnlineCandidate(source_grid=0, label="A", score=0.9, start_grid=0.0, end_grid=1.0),
        OnlineCandidate(source_grid=1, label="A", score=0.8, start_grid=1.0, end_grid=3.0),
    ]

    emitted = emitter.step(video_name="v1", now_frame=104, state=state, candidates=candidates)

    assert len(emitted) == 1
    assert emitted[0].source_grid == 0
    assert emitted[0].emit_frame == 104
    assert emitted[0].end_frame == 102
    assert emitted[0].latency_sec == pytest.approx((104 - 102) / 30.0)
    assert state.last_emitted_grid == 0


def test_online_emitter_rejects_non_monotonic_stream():
    from opentad.utils.online_protocol import GridSpec, OnlineEmitter, OnlineState, ProtocolViolation

    grid = GridSpec(fps=30.0, snippet_stride=1, window_start_frame=0, offset_frames=0)
    state = OnlineState(video_name="v1", last_emit_frame=10)
    emitter = OnlineEmitter(grid_spec=grid)

    with pytest.raises(ProtocolViolation, match="non-monotonic"):
        emitter.step(video_name="v1", now_frame=9, state=state, candidates=[])


def test_online_emitter_suppresses_duplicate_against_prior_prefix():
    from opentad.utils.online_protocol import GridSpec, OnlineCandidate, OnlineEmitter, OnlineState

    grid = GridSpec(fps=10.0, snippet_stride=1, window_start_frame=0, offset_frames=0)
    state = OnlineState(video_name="v1")
    emitter = OnlineEmitter(grid_spec=grid, score_threshold=0.0, nms_iou_threshold=0.5, latency_frames=0)

    first = emitter.step(
        video_name="v1",
        now_frame=4,
        state=state,
        candidates=[OnlineCandidate(source_grid=1, label="A", score=0.5, start_grid=0.0, end_grid=2.0)],
    )
    second = emitter.step(
        video_name="v1",
        now_frame=5,
        state=state,
        candidates=[OnlineCandidate(source_grid=3, label="A", score=0.9, start_grid=0.0, end_grid=2.0)],
    )

    assert len(first) == 1
    assert second == []
    assert len(state.emitted) == 1


def test_candidate_source_grid_uses_absolute_proposal_end_not_flattened_index():
    from opentad.utils.online_protocol import candidate_source_grid

    assert candidate_source_grid(end_grid=2.0, grid_offset=192, flattened_index=377) == 194
    assert candidate_source_grid(end_grid=2.2, grid_offset=192, flattened_index=377) == 195


def test_stream_key_includes_protocol_identity_not_just_video_name():
    from opentad.utils.online_protocol import make_stream_key

    base = dict(
        video_name="same_video",
        stream_id="eval",
        fps=30.0,
        snippet_stride=8,
        offset_frames=0,
        processor_id="siglip2-224",
        encoder_id="siglip2-base",
        image_size=224,
        frame_policy="recent_frame_only",
        input_format="raw_frames",
    )

    assert make_stream_key(base) != make_stream_key({**base, "fps": 15.0})
    assert make_stream_key(base) != make_stream_key({**base, "snippet_stride": 4})
    assert make_stream_key(base) != make_stream_key({**base, "offset_frames": 4})
    assert make_stream_key(base) != make_stream_key({**base, "processor_id": "siglip2-384"})
    assert make_stream_key(base) != make_stream_key({**base, "encoder_id": "other"})
    assert make_stream_key(base) != make_stream_key({**base, "image_size": 384})
    assert make_stream_key(base) != make_stream_key({**base, "frame_policy": "clip_center"})
    assert make_stream_key(base) != make_stream_key({**base, "input_format": "features"})


def test_streaming_emission_ledger_sort_is_rank_order_independent():
    from opentad.utils.online_protocol import sort_emission_ledger

    result_dict = {
        "v1": [
            {"emit_frame": 32, "source_grid": 4, "segment": [0.8, 1.0], "label": "A", "score": 0.7},
            {"emit_frame": 16, "source_grid": 2, "segment": [0.2, 0.4], "label": "A", "score": 0.8},
            {"emit_frame": 16, "source_grid": 1, "segment": [0.1, 0.3], "label": "B", "score": 0.9},
        ],
        "v2": [
            {"emit_frame": 8, "source_grid": 1, "segment": [0.1, 0.2], "label": "A", "score": 0.5},
        ],
    }

    sorted_dict = sort_emission_ledger(result_dict)

    assert [row["source_grid"] for row in sorted_dict["v1"]] == [1, 2, 4]
    assert sorted_dict["v2"][0]["emit_frame"] == 8


def test_streaming_safe_online_eval_rejects_multi_rank_state_splitting():
    from opentad.utils.online_protocol import ProtocolViolation, validate_streaming_safe_world_size

    post_cfg = SimpleNamespace(streaming_safe_emission=True)

    validate_streaming_safe_world_size(post_cfg, world_size=1)
    with pytest.raises(ProtocolViolation, match="single-rank"):
        validate_streaming_safe_world_size(post_cfg, world_size=2)


def test_streaming_safe_online_eval_rejects_callable_external_classifiers():
    from opentad.utils.online_protocol import ProtocolViolation, validate_streaming_safe_ext_cls

    validate_streaming_safe_ext_cls(["Action"], post_cfg=SimpleNamespace(streaming_safe_emission=True))
    with pytest.raises(ProtocolViolation, match="callable external classifiers"):
        validate_streaming_safe_ext_cls(lambda *_args: None, post_cfg=SimpleNamespace(streaming_safe_emission=True))


def test_eval_engine_uses_streaming_safe_sliding_window_resolution():
    source = (ROOT / "opentad/cores/test_engine.py").read_text(encoding="utf-8")

    assert "resolve_sliding_window_for_post_processing" in source
    assert "should_run_video_level_nms" in source
    assert "sort_emission_ledger" in source
    assert "validate_streaming_safe_world_size" in source
    assert "validate_streaming_safe_ext_cls" in source
    assert "reset_online_states" in source


def test_single_stage_streaming_results_include_emission_ledger_fields():
    source = (ROOT / "opentad/models/detectors/single_stage.py").read_text(encoding="utf-8")

    assert "streaming_safe_emission" in source
    assert "emit_frame" in source
    assert "source_grid" in source
    assert "latency_sec" in source
    assert "self._online_states" in source
    assert "reset_online_states" in source
    assert "make_stream_key" in source
    assert "grid_offset" in source
    assert "callable(ext_cls)" not in source
    assert "candidate_source_grid" in source
    assert "window_start_frame" in source
    assert "window_end_frame" in source
    assert "processor_id" in source
    assert "encoder_id" in source


def test_matr_stream_memory_uses_protocol_stream_key():
    source = (ROOT / "opentad/models/dense_heads/matr_head.py").read_text(encoding="utf-8")

    assert "make_stream_key" in source


def test_single_stage_streaming_post_processing_filters_future_and_writes_ledger():
    torch = _torch_or_skip()
    from opentad.models.detectors.single_stage import SingleStageDetector

    detector = SingleStageDetector()
    predictions = (
        [torch.tensor([[0.0, 1.0], [1.0, 3.0]])],
        [torch.tensor([[0.9], [0.8]])],
    )
    metas = [
        dict(
            video_name="v1",
            fps=30.0,
            snippet_stride=2,
            window_start_frame=100,
            window_end_frame=104,
            offset_frames=0,
            duration=10.0,
        )
    ]
    post_cfg = SimpleNamespace(
        streaming_safe_emission=True,
        sliding_window=True,
        nms=dict(max_seg_num=1, min_score=0.0),
        pre_nms_thresh=0.001,
        pre_nms_topk=10,
        max_latency=0.0,
    )

    results = detector.post_processing(predictions, metas, post_cfg=post_cfg, ext_cls=["Action"])

    assert list(results) == ["v1"]
    assert len(results["v1"]) == 1
    row = results["v1"][0]
    assert row["label"] == "Action"
    assert row["emit_frame"] == 104
    assert row["source_grid"] == 51
    assert row["end_frame"] == 102
    assert row["latency_sec"] > 0
    assert row["stream_id"] == "default"
    assert row["window_start_frame"] == 100
    assert row["window_end_frame"] == 104
