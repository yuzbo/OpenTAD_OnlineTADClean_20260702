import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from opentad.utils.full_petal_data_contract import (
    canonical_json_sha256,
    load_json,
)


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


def _split_contract(tmp_path, ann_file):
    fit_path = tmp_path / "thumos_fit_core_160.txt"
    calibration_path = tmp_path / "thumos_calibration_40.txt"
    fit_ids = ["train_a"]
    calibration_ids = ["train_b"]
    fit_path.write_text("train_a\n", encoding="utf-8")
    calibration_path.write_text("train_b\n", encoding="utf-8")
    annotation = load_json(ann_file)
    manifest = {
        "schema": "full_petal.thumos_development_split",
        "schema_version": "thumos-development-split-v1",
        "dataset": "THUMOS14",
        "seed": 20260713,
        "annotation": {
            "canonical_sha256": canonical_json_sha256(annotation),
        },
        "universe": {
            "count": 2,
            "ids": ["train_a", "train_b"],
            "ids_sha256": canonical_json_sha256(["train_a", "train_b"]),
        },
        "splits": {
            "fit_core": {
                "count": 1,
                "ids": fit_ids,
                "ids_sha256": canonical_json_sha256(fit_ids),
            },
            "calibration": {
                "count": 1,
                "ids": calibration_ids,
                "ids_sha256": canonical_json_sha256(calibration_ids),
            },
        },
        "artifacts": {
            "fit_ids": {
                "name": fit_path.name,
                "sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
                "content_sha256": canonical_json_sha256(fit_ids),
            },
            "calibration_ids": {
                "name": calibration_path.name,
                "sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
                "content_sha256": canonical_json_sha256(calibration_ids),
            },
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = tmp_path / "thumos_development_split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, fit_path, calibration_path


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

    specs = train.iter_crs_eps_sampling_specs()
    assert [spec.video_id for spec in specs] == ["train_a", "train_b"]
    assert [spec.num_bins for spec in specs] == [5, 3]
    assert [instance.instance_id for instance in specs[0].instances] == [0, 1]
    assert specs[0].instances[0].birth_bin == 0
    assert specs[0].instances[0].end_bin is None
    with pytest.raises(RuntimeError, match="test datasets"):
        test.iter_crs_eps_sampling_specs()


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


def test_dataset_rejects_selected_video_with_zero_cached_tokens(tmp_path):
    ann_file, class_map, feature_dir, manifest_file = _fixture(tmp_path)
    empty_path = feature_dir / "train_a.npy"
    np.save(empty_path, np.empty((0, 4), dtype=np.float32))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["videos"]["train_a"].update(
        sha256=hashlib.sha256(empty_path.read_bytes()).hexdigest(),
        num_tokens=0,
        source_frames=[],
    )
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="at least one token"):
        _dataset_class()(
            ann_file=ann_file,
            subset_name="training",
            class_map=class_map,
            data_path=feature_dir,
            cache_manifest=manifest_file,
            chunk_size=2,
            feature_stride=8,
        )


def test_dataset_rejects_duplicate_cache_manifest_json_keys(tmp_path):
    ann_file, class_map, feature_dir, manifest_file = _fixture(tmp_path)
    text = manifest_file.read_text(encoding="utf-8")
    text = text.replace(
        '"schema": "ontad_feature_cache_v1"',
        '"schema": "ontad_feature_cache_v1", "schema": "ontad_feature_cache_v1"',
        1,
    )
    manifest_file.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
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


def test_dataset_rejects_feature_replaced_after_index_before_first_use(tmp_path):
    ann_file, class_map, feature_dir, manifest_file = _fixture(tmp_path)
    dataset = _dataset_class()(
        ann_file=ann_file,
        subset_name="training",
        class_map=class_map,
        data_path=feature_dir,
        cache_manifest=manifest_file,
        chunk_size=2,
        feature_stride=8,
    )
    np.save(
        feature_dir / "train_a.npy",
        np.full((5, 4), 777.0, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="feature hash/content mismatch"):
        dataset[0]


def test_dataset_reuses_only_previously_verified_feature_bytes(tmp_path):
    ann_file, class_map, feature_dir, manifest_file = _fixture(tmp_path)
    dataset = _dataset_class()(
        ann_file=ann_file,
        subset_name="training",
        class_map=class_map,
        data_path=feature_dir,
        cache_manifest=manifest_file,
        chunk_size=2,
        feature_stride=8,
    )
    first = dataset[0]
    np.save(
        feature_dir / "train_a.npy",
        np.full((5, 4), 777.0, dtype=np.float32),
    )
    second = dataset[1]

    assert float(first["inputs"].max()) < 777.0
    assert np.array_equal(
        second["inputs"],
        np.arange(20, dtype=np.float32).reshape(5, 4)[2:4].T,
    )


def test_dataset_consumes_hashed_split_role_and_binds_it_to_provenance(tmp_path):
    ann_file, class_map, feature_dir, cache_manifest = _fixture(tmp_path)
    split_manifest, fit_path, _ = _split_contract(tmp_path, ann_file)

    dataset = _dataset_class()(
        ann_file=ann_file,
        subset_name="training",
        class_map=class_map,
        data_path=feature_dir,
        cache_manifest=cache_manifest,
        allow_list=fit_path,
        split_manifest=split_manifest,
        split_role="fit_core",
        split_seed=20260713,
        chunk_size=8,
        feature_stride=8,
    )

    assert set(dataset.packet_manifests) == {"train_a"}
    assert dataset.split_role == "fit_core"
    assert dataset.split_seed == 20260713
    assert len(dataset.split_manifest_sha256) == 64
    assert len(dataset[0]["metas"]["input_provenance_digest"]) == 64


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda context: context.update(split_seed=20260714), "seed"),
        (lambda context: context.update(split_role="calibration"), "allow-list"),
        (
            lambda context: context["fit_path"].write_text(
                "train_a\ntrain_b\n", encoding="utf-8"
            ),
            "hash",
        ),
    ],
)
def test_dataset_rejects_split_seed_role_or_id_file_drift(tmp_path, mutate, message):
    ann_file, class_map, feature_dir, cache_manifest = _fixture(tmp_path)
    split_manifest, fit_path, _ = _split_contract(tmp_path, ann_file)
    context = {
        "split_seed": 20260713,
        "split_role": "fit_core",
        "fit_path": fit_path,
    }
    mutate(context)

    with pytest.raises(ValueError, match=message):
        _dataset_class()(
            ann_file=ann_file,
            subset_name="training",
            class_map=class_map,
            data_path=feature_dir,
            cache_manifest=cache_manifest,
            allow_list=fit_path,
            split_manifest=split_manifest,
            split_role=context["split_role"],
            split_seed=context["split_seed"],
            chunk_size=8,
            feature_stride=8,
        )


def test_dataset_rejects_rehashed_but_semantically_overlapping_split(tmp_path):
    ann_file, class_map, feature_dir, cache_manifest = _fixture(tmp_path)
    split_manifest, fit_path, _ = _split_contract(tmp_path, ann_file)
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    manifest["splits"]["calibration"] = dict(manifest["splits"]["fit_core"])
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    split_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        _dataset_class()(
            ann_file=ann_file,
            subset_name="training",
            class_map=class_map,
            data_path=feature_dir,
            cache_manifest=cache_manifest,
            allow_list=fit_path,
            split_manifest=split_manifest,
            split_role="fit_core",
            split_seed=20260713,
            chunk_size=8,
            feature_stride=8,
        )
