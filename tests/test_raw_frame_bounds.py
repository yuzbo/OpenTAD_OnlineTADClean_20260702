import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_frame_bounds_module():
    path = ROOT / "opentad/datasets/frame_bounds.py"
    assert path.exists(), "raw-frame bounds helper module should exist"
    spec = importlib.util.spec_from_file_location("frame_bounds_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_frame_total_frames_do_not_exceed_annotation_frame_count():
    frame_bounds = _load_frame_bounds_module()
    video_info = {"duration": 699.087, "frame": 17475}

    assert frame_bounds.resolve_raw_video_total_frames(video_info, fps=30.0) == 17475


def test_raw_frame_total_frames_fall_back_to_protocol_fps_without_frame_count():
    frame_bounds = _load_frame_bounds_module()
    video_info = {"duration": 10.0}

    assert frame_bounds.resolve_raw_video_total_frames(video_info, fps=30.0) == 300


def test_raw_frame_total_frames_use_annotation_frames_without_fixed_fps():
    frame_bounds = _load_frame_bounds_module()
    video_info = {"duration": 699.087, "frame": 17475}

    assert frame_bounds.resolve_raw_video_total_frames(video_info, fps=-1) == 17475
