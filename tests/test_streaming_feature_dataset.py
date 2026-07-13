import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _dataset_class():
    path = ROOT / "opentad" / "datasets" / "streaming_feature.py"
    name = "_streaming_feature_dataset_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.StreamingFeatureDataset


def _fixture(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.save(feature_dir / "train_a.npy", np.arange(20, dtype=np.float32).reshape(5, 4))
    np.save(feature_dir / "train_b.npy", np.arange(12, dtype=np.float32).reshape(3, 4))

    annotation = {
        "database": {
            "train_a": {
                "subset": "training",
                "duration": 4.0,
                "frame": 120,
                "annotations": [
                    {"segment": [0.2, 1.8], "label": "A"},
                    {"segment": [1.0, 2.5], "label": "A"},
                ],
            },
            "train_b": {
                "subset": "training",
                "duration": 2.0,
                "frame": 60,
                "annotations": [{"segment": [0.5, 1.5], "label": "B"}],
            },
        }
    }
    ann_file = tmp_path / "annotations.json"
    ann_file.write_text(json.dumps(annotation), encoding="utf-8")
    class_map = tmp_path / "classes.txt"
    class_map.write_text("A\nB\n", encoding="utf-8")

    manifest = {
        "schema": "ontad_feature_cache_v1",
        "annotation_sha256": hashlib.sha256(ann_file.read_bytes()).hexdigest(),
        "encoder_id": "tiny-test-encoder",
        "feature_policy": "packet_recent_frame",
        "timestamp_convention": "zero_based_source_frame",
        "feature_stride": 8,
        "feature_dim": 4,
        "dtype": "float32",
        "videos": {
            "train_a": {
                "file": "train_a.npy",
                "sha256": hashlib.sha256((feature_dir / "train_a.npy").read_bytes()).hexdigest(),
                "num_tokens": 5,
                "source_frames": [7, 15, 23, 31, 39],
            },
            "train_b": {
                "file": "train_b.npy",
                "sha256": hashlib.sha256((feature_dir / "train_b.npy").read_bytes()).hexdigest(),
                "num_tokens": 3,
                "source_frames": [7, 15, 23],
            },
        },
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    return ann_file, class_map, feature_dir, manifest_file


def test_dataset_builds_complete_video_manifests_and_contiguous_chunks(tmp_path):
    ann_file, class_map, feature_dir, manifest_file = _fixture(tmp_path)
    dataset = _dataset_class()(
        ann_file=ann_file,
        subset_name="training",
        class_map=class_map,
        data_path=feature_dir,
        cache_manifest=manifest_file,
        chunk_size=2,
        feature_stride=8,
        stream_id="pes-stage1-test",
    )

    assert len(dataset) == 5
    assert dataset.packet_manifests == {"train_a": [0, 1, 2], "train_b": [3, 4]}

    first = dataset[0]
    assert first["inputs"].shape == (4, 2)
    assert first["masks"].tolist() == [True, True]
    assert first["metas"]["source_frames"] == (7, 15)
    assert first["metas"]["current_frame"] == 15
    assert first["metas"]["fps"] == pytest.approx(30.0)
    assert len(first["metas"]["input_provenance_digest"]) == 64
    assert first["stream_control"]["is_video_start"] is True
    assert first["stream_control"]["is_video_end"] is False

    last_a = dataset[2]
    assert last_a["inputs"].shape == (4, 1)
    assert last_a["metas"]["source_frames"] == (39,)
    assert last_a["stream_control"]["is_video_end"] is True


def test_dataset_keeps_terminal_metadata_out_of_model_meta(tmp_path):
    ann_file, class_map, feature_dir, manifest_file = _fixture(tmp_path)
    dataset = _dataset_class()(
        ann_file=ann_file,
        subset_name="training",
        class_map=class_map,
        data_path=feature_dir,
        cache_manifest=manifest_file,
        chunk_size=8,
        feature_stride=8,
    )

    sample = dataset[0]
    forbidden = {"duration", "total_frames", "num_frames", "is_video_end", "video_end_frame"}
    assert forbidden.isdisjoint(sample["metas"])
    assert sample["stream_control"]["is_video_end"] is True


def test_dataset_constructs_training_only_prefix_observable_schedule(tmp_path):
    ann_file, class_map, feature_dir, manifest_file = _fixture(tmp_path)
    Dataset = _dataset_class()
    train = Dataset(
        ann_file=ann_file,
        subset_name="training",
        class_map=class_map,
        data_path=feature_dir,
        cache_manifest=manifest_file,
        chunk_size=5,
        feature_stride=8,
    )
    test = Dataset(
        ann_file=ann_file,
        subset_name="training",
        class_map=class_map,
        data_path=feature_dir,
        cache_manifest=manifest_file,
        chunk_size=5,
        feature_stride=8,
        test_mode=True,
    )

    schedule = train[0]["prefix_schedule"]
    assert len(schedule) == 5
    assert all(item.end_frame is None for step in schedule for item in step.births + step.active)
    assert "prefix_schedule" not in test[0]


def test_dataset_rejects_cache_manifest_geometry_mismatch(tmp_path):
    ann_file, class_map, feature_dir, manifest_file = _fixture(tmp_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["videos"]["train_a"]["num_tokens"] = 99
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="num_tokens"):
        _dataset_class()(
            ann_file=ann_file,
            subset_name="training",
            class_map=class_map,
            data_path=feature_dir,
            cache_manifest=manifest_file,
            chunk_size=2,
            feature_stride=8,
        )


def test_dataset_rejects_manifest_from_different_annotations(tmp_path):
    ann_file, class_map, feature_dir, manifest_file = _fixture(tmp_path)
    annotation = json.loads(ann_file.read_text(encoding="utf-8"))
    annotation["database"]["train_a"]["frame"] = 121
    ann_file.write_text(json.dumps(annotation), encoding="utf-8")

    with pytest.raises(ValueError, match="annotation hash"):
        _dataset_class()(
            ann_file=ann_file,
            subset_name="training",
            class_map=class_map,
            data_path=feature_dir,
            cache_manifest=manifest_file,
            chunk_size=2,
            feature_stride=8,
        )


def test_dataset_rejects_feature_bytes_that_do_not_match_manifest(tmp_path):
    ann_file, class_map, feature_dir, manifest_file = _fixture(tmp_path)
    np.save(feature_dir / "train_a.npy", np.zeros((5, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="feature hash"):
        _dataset_class()(
            ann_file=ann_file,
            subset_name="training",
            class_map=class_map,
            data_path=feature_dir,
            cache_manifest=manifest_file,
            chunk_size=2,
            feature_stride=8,
        )
