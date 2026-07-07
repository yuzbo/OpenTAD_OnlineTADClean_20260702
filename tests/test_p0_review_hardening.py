import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return (ROOT / path).read_text(encoding="utf-8")


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


def _ledger_row(video="keep", latency_sec=0.1, score=0.9):
    del video
    return dict(
        segment=[0.0, 1.0],
        label="Action",
        score=score,
        emit_frame=10,
        source_frame=10,
        start_frame=0,
        end_frame=10,
        latency_sec=latency_sec,
        source_grid=10,
        stream_key="stream",
    )


def test_online_emitter_latency_frames_is_max_allowed_delay():
    from opentad.utils.online_protocol import GridSpec, OnlineCandidate, OnlineEmitter, OnlineState

    grid = GridSpec(fps=1.0, snippet_stride=1, window_start_frame=0, offset_frames=0)
    emitter = OnlineEmitter(grid_spec=grid, score_threshold=0.0, nms_iou_threshold=1.0, latency_frames=5)
    state = OnlineState(video_name="v1")

    emitted = emitter.step(
        video_name="v1",
        now_frame=10,
        state=state,
        candidates=[
            OnlineCandidate(source_grid=10, label="A", score=0.9, start_grid=8.0, end_grid=10.0),
            OnlineCandidate(source_grid=7, label="B", score=0.9, start_grid=5.0, end_grid=7.0),
            OnlineCandidate(source_grid=4, label="C", score=0.9, start_grid=2.0, end_grid=4.0),
            OnlineCandidate(source_grid=11, label="D", score=0.9, start_grid=10.0, end_grid=11.0),
        ],
    )

    labels = [det.label for det in emitted]
    assert labels == ["B", "A"]
    assert [det.latency_sec for det in emitted] == [pytest.approx(3.0), pytest.approx(0.0)]
    assert state.last_emit_frame == 10


def test_online_map_stats_are_computed_after_allowed_and_latency_filtering():
    from opentad.evaluations.online_map import OnlineMAP

    evaluator = object.__new__(OnlineMAP)
    evaluator.require_ledger = True
    evaluator.require_no_future = True
    evaluator.max_latency_sec = 0.5
    evaluator.pred_fields = ["results"]
    evaluator.blocked_videos = []
    evaluator.allowed_videos = {"keep"}
    evaluator.activity_index = {"Action": 0}
    evaluator.online_metric_dict = {}

    prediction = {
        "results": {
            "keep": [
                _ledger_row(latency_sec=0.1, score=0.9),
                _ledger_row(latency_sec=0.8, score=0.8),
            ],
            "drop": [_ledger_row(latency_sec=0.1, score=0.7)],
        }
    }

    dataframe = evaluator._import_prediction(prediction)

    assert len(dataframe) == 1
    assert evaluator.online_metric_dict["online"]["num_emissions"] == 1
    assert evaluator.online_metric_dict["num_predictions"] == 1
    assert evaluator.online_metric_dict["online"]["emission_latency_sec"]["count"] == 1


def test_selector_source_fails_closed_for_all_invalid_masks():
    source = _read("opentad/models/selectors/causal_frame_selector.py")

    assert "return torch.zeros(1" not in source
    assert "all-invalid mask" in source


def test_siglip_selected_only_runtime_assert_is_strict():
    source = _read("opentad/models/backbones/online_siglip_adapter.py")

    assert "selected_only requires selected frames at runtime" in source
    assert "pixels.shape[0] != expected" in source


def test_matr_irregular_metadata_batch_contract_is_explicit_fail_fast():
    source = _read("opentad/models/dense_heads/matr_head.py")

    assert "len(metas) > 1" in source
    assert "adaptive irregular MATRHead currently requires batch_size=1" in source


def test_selected_only_encoder_runtime_counts_encoded_frames():
    torch = _torch_or_skip()

    import opentad.models  # noqa: F401
    from mmengine.registry import MODELS

    encoder = MODELS.build(
        dict(
            type="OnlineSigLIPFrameEncoder",
            model_name=None,
            backend="tiny",
            embed_dims=4,
            image_size=4,
            frame_chunk_size=8,
            freeze_vision_encoder=True,
            frame_selector=dict(type="CausalFrameSelector", policy="causal_stride", stride=2, keep_ratio=0.5),
            encode_policy="selected_only",
            assert_selected_only=True,
            output_layout="bct",
        )
    )
    seen = {}

    def fake_encode(pixels):
        seen["count"] = int(pixels.shape[0])
        return torch.ones(pixels.shape[0], encoder.embed_dims, device=pixels.device)

    encoder._encode_pixels_chunked = fake_encode
    frames = torch.randn(1, 1, 3, 8, 4, 4)
    masks = torch.ones(1, 8, dtype=torch.bool)

    feats, out_masks = encoder(frames, masks=masks, metas=[dict(fps=1.0, snippet_stride=1)])

    assert seen["count"] == 4
    assert feats.shape == (1, 4, 4)
    assert out_masks.tolist() == [[True, True, True, True]]


def test_selector_rejects_all_invalid_masks_at_runtime():
    torch = _torch_or_skip()

    import opentad.models  # noqa: F401
    from mmengine.registry import MODELS

    selector = MODELS.build(dict(type="CausalFrameSelector", policy="causal_stride", stride=2))
    frames = torch.randn(1, 8, 3, 4, 4)
    masks = torch.zeros(1, 8, dtype=torch.bool)

    with pytest.raises(ValueError, match="all-invalid mask"):
        selector.select(frames, masks=masks)

