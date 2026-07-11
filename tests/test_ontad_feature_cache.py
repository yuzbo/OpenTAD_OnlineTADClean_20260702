import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "tools" / "cache_ontad_features.py"
    name = "_cache_ontad_features_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_recent_packet_source_frames_include_short_terminal_packet():
    assert _module().recent_packet_source_frames(total_frames=10, feature_stride=4) == (3, 7, 9)
    assert _module().recent_packet_source_frames(total_frames=8, feature_stride=4) == (3, 7)


def test_selected_frames_are_encoded_once_in_fixed_batches():
    module = _module()
    calls = []

    def encode_batch(frames):
        calls.append(np.asarray(frames).copy())
        return np.asarray([[float(frame.mean()), float(frame.max())] for frame in frames])

    frame_iter = ((index, np.full((2, 2, 3), index, dtype=np.uint8)) for index in range(10))
    features = module.encode_selected_frames(
        frame_iter,
        source_frames=(3, 7, 9),
        encode_batch=encode_batch,
        batch_size=2,
    )

    assert [batch.shape[0] for batch in calls] == [2, 1]
    assert features.tolist() == [[3.0, 3.0], [7.0, 7.0], [9.0, 9.0]]


def test_feature_file_write_is_atomic_and_resume_checks_geometry(tmp_path):
    module = _module()
    path = tmp_path / "video.npy"
    features = np.arange(12, dtype=np.float32).reshape(3, 4)

    metadata = module.write_feature_file(
        path,
        features,
        source_frames=(3, 7, 9),
        resume=False,
    )
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp"))
    assert metadata["num_tokens"] == 3
    assert metadata["feature_dim"] == 4
    assert metadata["dtype"] == "float32"

    resumed = module.write_feature_file(
        path,
        None,
        source_frames=(3, 7, 9),
        resume=True,
    )
    assert resumed["resumed"] is True
    assert np.load(path).tolist() == features.tolist()

    with pytest.raises(ValueError, match="geometry"):
        module.write_feature_file(
            path,
            np.ones((2, 4), dtype=np.float32),
            source_frames=(3, 7),
            resume=True,
        )

    with pytest.raises(ValueError, match="dtype"):
        module.write_feature_file(
            path,
            None,
            source_frames=(3, 7, 9),
            resume=True,
            expected_dtype=np.float16,
        )


def test_per_video_resume_record_fails_closed_on_encoder_or_stride_change(tmp_path):
    module = _module()
    feature_path = tmp_path / "video.npy"
    metadata = module.write_feature_file(
        feature_path,
        np.ones((2, 4), dtype=np.float16),
        source_frames=(3, 7),
    )
    record_path = tmp_path / "video.json"
    module.write_video_cache_record(
        record_path,
        metadata,
        encoder_id="encoder-a",
        feature_stride=4,
    )

    resumed = module.write_feature_file(
        feature_path,
        None,
        source_frames=(3, 7),
        resume=True,
        expected_dtype=np.float16,
    )
    module.validate_video_cache_record(record_path, resumed, "encoder-a", 4)
    with pytest.raises(ValueError, match="resume contract"):
        module.validate_video_cache_record(record_path, resumed, "encoder-b", 4)


def test_manifest_records_hashes_and_exact_source_frames(tmp_path):
    module = _module()
    annotation = tmp_path / "ann.json"
    annotation.write_text('{"database": {}}', encoding="utf-8")
    output = tmp_path / "manifest.json"
    videos = {
        "video": {
            "file": "video.npy",
            "num_tokens": 2,
            "feature_dim": 4,
            "source_frames": [3, 7],
        }
    }

    module.write_cache_manifest(
        output,
        annotation_path=annotation,
        encoder_id="tiny-test",
        feature_stride=4,
        feature_dim=4,
        dtype="float32",
        videos=videos,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "ontad_feature_cache_v1"
    assert len(payload["annotation_sha256"]) == 64
    assert payload["videos"]["video"]["source_frames"] == [3, 7]
