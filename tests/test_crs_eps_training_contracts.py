import hashlib
import json

import numpy as np
import pytest
import torch

from opentad.cores.train_engine import train_one_epoch
from opentad.datasets.builder import build_dataloader
from opentad.datasets.crs_eps_feature import CrsEpsFeatureDataset
from opentad.models.dense_heads.persistent_event_set_head import PersistentEventSetHead
from opentad.models.detectors.persistent_trajectory_ontad import (
    PersistentTrajectoryOnlineDetector,
)
from opentad.utils.evidence_bundle import EvidenceBundleError
from opentad.utils.online_protocol import ProtocolViolation
from opentad.utils.prefix_instance_schedule import build_prefix_instance_schedule


def _detector():
    torch.manual_seed(91)
    head = PersistentEventSetHead(
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
    return PersistentTrajectoryOnlineDetector(
        head=head,
        trajectory_binding_mode="fixed_birth_slot",
        detach_stream_state=True,
    )


def _meta(frames, *, stream_id="episode", replay_start=0):
    return {
        "video_name": "video",
        "video_id": "video",
        "stream_id": stream_id,
        "input_format": "cached_features",
        "input_provenance_digest": "a" * 64,
        "feature_stride": 1,
        "fps": 30.0,
        "source_frames": tuple(frames),
        "current_frame": int(frames[-1]),
        "packet_start_token": replay_start,
        "packet_end_token": replay_start + len(frames),
    }


def _crs_control(
    *,
    replay=(0, 76),
    supervised=(68, 76),
    gradient=((64, 76),),
    draw_index=0,
    group_size=1,
):
    return {
        "draw_index": draw_index,
        "episode_id": f"episode-{draw_index}",
        "supervised_range": list(supervised),
        "replay_range": list(replay),
        "gradient_ranges": [list(value) for value in gradient],
        "raw_weight_by_bin": [0.25] * (supervised[1] - supervised[0]),
        "final_weight_by_bin": [0.25] * (supervised[1] - supervised[0]),
        "video_group_size": group_size,
        "is_video_group_start": draw_index == 0,
        "is_video_group_end": draw_index + 1 == group_size,
        "episode_manifest_sha256": "b" * 64,
    }


def test_crs_eps_gradient_starts_at_original_global_detach_boundary():
    detector = _detector().train()
    frames = tuple(range(76))
    inputs = torch.randn(1, 4, len(frames), requires_grad=True)
    masks = torch.ones(1, len(frames), dtype=torch.bool)
    schedule = build_prefix_instance_schedule([], [], frames, previous_frame=-1)

    output = detector.train_crs_eps_episode(
        inputs,
        masks,
        _meta(frames),
        schedule,
        _crs_control(),
    )
    output.losses["cost"].backward()

    assert torch.equal(inputs.grad[:, :, :64], torch.zeros_like(inputs.grad[:, :, :64]))
    assert torch.count_nonzero(inputs.grad[:, :, 64:]).item() > 0
    assert output.audit["gradient_ranges"] == ((64, 76),)
    assert output.audit["per_loss_denominator"] == 8


def test_crs_eps_rejects_clipped_or_drifted_training_weights():
    detector = _detector().train()
    frames = tuple(range(8))
    inputs = torch.randn(1, 4, 8)
    masks = torch.ones(1, 8, dtype=torch.bool)
    schedule = build_prefix_instance_schedule([], [], frames, previous_frame=-1)
    control = _crs_control(replay=(0, 8), supervised=(0, 8), gradient=((0, 8),))
    control["final_weight_by_bin"][0] = 0.1

    with pytest.raises(ProtocolViolation, match="uncapped"):
        detector.train_crs_eps_episode(inputs, masks, _meta(frames), schedule, control)


def test_independent_draws_stage_unique_terminal_states_and_commit_once():
    detector = _detector().train()
    frames = tuple(range(8))
    inputs = torch.randn(1, 4, 8)
    masks = torch.ones(1, 8, dtype=torch.bool)
    schedule = build_prefix_instance_schedule([], [], frames, previous_frame=-1)

    for draw_index in range(2):
        stream_id = f"episode-{draw_index}"
        detector(
            inputs,
            masks,
            [_meta(frames, stream_id=stream_id)],
            return_loss=True,
            prefix_schedule=[schedule],
            stream_control=[
                {
                    "video_id": "video",
                    "stream_id": stream_id,
                    "chunk_index": 0,
                    "is_video_start": True,
                    "is_video_end": True,
                }
            ],
            crs_eps=[
                _crs_control(
                    replay=(0, 8),
                    supervised=(0, 8),
                    gradient=((0, 8),),
                    draw_index=draw_index,
                    group_size=2,
                )
            ],
        )
        assert detector._runtime_states == {}
        assert detector._supervision_states == {}
        assert len(detector._staged_runtime_states) == draw_index + 1

    detector.commit_online_update()
    assert detector._runtime_states == {}
    assert detector._supervision_states == {}
    assert detector.has_pending_online_update() is False


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class _Scheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1

    def get_last_lr(self):
        return [0.1]

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.steps = state["steps"]


class _CrsToy(torch.nn.Module):
    def __init__(self, mutate_buffer=False, slot_exhaustion=0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("audit_buffer", torch.tensor(0))
        self.mutate_buffer = mutate_buffer
        self.slot_exhaustion = int(slot_exhaustion)
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
        self.pending = False
        self.commits += 1

    def rollback_online_update(self):
        self.pending = False
        self.rollbacks += 1

    def forward(self, inputs, stream_control, crs_eps, return_loss=False):
        assert return_loss
        self.pending = True
        if self.mutate_buffer:
            self.audit_buffer.add_(1)
        self.last_episode_audit = {
            "slot_exhaustion": self.slot_exhaustion,
            "control_unroll_seconds": 0.0,
        }
        cost = (self.weight - inputs).pow(2).mean()
        scale = cost.new_tensor(1.0 / crs_eps[0]["video_group_size"])
        return {
            "cost": cost,
            "_optimizer_weight": scale,
            "_optimizer_denominator": scale,
        }


def _engine_crs_control(draw_index, group_size=2):
    return {
        "video_id": "video",
        "video_group_index": 0,
        "video_group_size": group_size,
        "draw_index": draw_index,
        "is_video_group_start": draw_index == 0,
        "is_video_group_end": draw_index + 1 == group_size,
        "episode_manifest_sha256": "c" * 64,
        "replay_range": [0, 1],
        "supervised_range": [0, 1],
        "gradient_ranges": [[0, 1]],
        "video_covered_unique_bins": 1,
        "video_effective_sample_size": 2.0,
        "video_ipw_weight_sum": 2.0,
        "video_ipw_weight_squared_sum": 2.0,
    }


def _crs_loader():
    return [
        {
            "inputs": torch.tensor([1.0]),
            "stream_control": [
                {"video_id": "video", "is_video_start": True, "is_video_end": True}
            ],
            "crs_eps": [_engine_crs_control(0)],
        },
        {
            "inputs": torch.tensor([3.0]),
            "stream_control": [
                {"video_id": "video", "is_video_start": True, "is_video_end": True}
            ],
            "crs_eps": [_engine_crs_control(1)],
        },
    ]


def test_engine_applies_one_hh_update_per_video_group_without_double_division():
    model = _CrsToy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _Scheduler()

    stats = train_one_epoch(
        _crs_loader(),
        model,
        optimizer,
        scheduler,
        curr_epoch=0,
        logger=_Logger(),
        logging_interval=100,
    )

    assert model.weight.item() == pytest.approx(0.4)
    assert model.commits == 1
    assert scheduler.steps == 1
    assert stats["successful_optimizer_events"] == 1


def test_engine_fails_closed_on_mutable_buffer_leak_and_rolls_back_group():
    model = _CrsToy(mutate_buffer=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="buffers leaked"):
        train_one_epoch(
            _crs_loader(),
            model,
            optimizer,
            _Scheduler(),
            curr_epoch=0,
            logger=_Logger(),
            logging_interval=100,
        )

    assert model.rollbacks == 1
    assert model.weight.item() == 0.0


def test_engine_treats_slot_exhaustion_as_scientific_failure():
    model = _CrsToy(slot_exhaustion=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="scientific failure: slot exhaustion"):
        train_one_epoch(
            _crs_loader(),
            model,
            optimizer,
            _Scheduler(),
            curr_epoch=0,
            logger=_Logger(),
            logging_interval=100,
        )

    assert model.rollbacks == 1
    assert model.weight.item() == 0.0


def _dataset_fixture(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    features = np.arange(48, dtype=np.float32).reshape(12, 4)
    np.save(feature_dir / "video.npy", features)
    annotation = {
        "database": {
            "video": {
                "subset": "training",
                "duration": 4.0,
                "frame": 120,
                "annotations": [{"segment": [0.2, 2.0], "label": "A"}],
            }
        }
    }
    ann_file = tmp_path / "annotations.json"
    ann_file.write_text(json.dumps(annotation), encoding="utf-8")
    class_map = tmp_path / "classes.txt"
    class_map.write_text("A\n", encoding="utf-8")
    feature_file = feature_dir / "video.npy"
    manifest = {
        "schema": "ontad_feature_cache_v1",
        "annotation_sha256": hashlib.sha256(ann_file.read_bytes()).hexdigest(),
        "encoder_id": "test-encoder",
        "feature_policy": "packet_recent_frame",
        "timestamp_convention": "zero_based_source_frame",
        "feature_stride": 8,
        "feature_dim": 4,
        "dtype": "float32",
        "videos": {
            "video": {
                "file": "video.npy",
                "sha256": hashlib.sha256(feature_file.read_bytes()).hexdigest(),
                "num_tokens": 12,
                "source_frames": [7 + 8 * value for value in range(12)],
            }
        },
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    return ann_file, class_map, feature_dir, manifest_file


def test_crs_dataset_rebuilds_epoch_manifest_and_keeps_gt_out_of_model_meta(tmp_path):
    ann_file, class_map, feature_dir, manifest_file = _dataset_fixture(tmp_path)
    dataset = CrsEpsFeatureDataset(
        ann_file=ann_file,
        subset_name="training",
        class_map=class_map,
        data_path=feature_dir,
        cache_manifest=manifest_file,
        chunk_size=64,
        feature_stride=8,
        sampling_seed=705,
        draws_per_video=3,
    )

    assert len(dataset) == 3
    first_hash = dataset.current_episode_manifest_sha256
    sample = dataset[0]
    assert sample["crs_eps"]["is_video_group_start"] is True
    assert sample["crs_eps"]["video_group_size"] == 3
    assert {
        "anchor_bin",
        "supervised_range",
        "raw_weight_by_bin",
        "left_censored_instance_ids",
    }.isdisjoint(sample["metas"])
    dataset.set_epoch(1)
    assert dataset.current_episode_manifest_sha256 != first_hash

    output_dir = tmp_path / "evidence"
    path = dataset.persist_current_manifest(output_dir)
    assert path.is_file()
    with pytest.raises(EvidenceBundleError, match="overwrite"):
        dataset.persist_current_manifest(output_dir)


def test_dataset_dataloader_detector_optimizer_cpu_closure(tmp_path):
    ann_file, class_map, feature_dir, manifest_file = _dataset_fixture(tmp_path)
    dataset = CrsEpsFeatureDataset(
        ann_file=ann_file,
        subset_name="training",
        class_map=class_map,
        data_path=feature_dir,
        cache_manifest=manifest_file,
        chunk_size=64,
        feature_stride=8,
        sampling_seed=705,
        draws_per_video=2,
    )
    loader = build_dataloader(
        dataset,
        batch_size=1,
        rank=0,
        world_size=1,
        streaming=True,
        stream_batch_size=1,
        num_workers=0,
    )
    model = _detector()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = _Scheduler()

    stats = train_one_epoch(
        loader,
        model,
        optimizer,
        scheduler,
        curr_epoch=0,
        logger=_Logger(),
        logging_interval=100,
    )

    assert len(loader) == 2
    assert stats["successful_optimizer_events"] == 1
    assert stats["optimizer_events"] == 1
    assert scheduler.steps == 1
    assert model.has_pending_online_update() is False


def test_gold_audit_replay_arms_are_explicit_and_do_not_mutate_training_manifest(tmp_path):
    ann_file, class_map, feature_dir, manifest_file = _dataset_fixture(tmp_path)
    dataset = CrsEpsFeatureDataset(
        ann_file=ann_file,
        subset_name="training",
        class_map=class_map,
        data_path=feature_dir,
        cache_manifest=manifest_file,
        chunk_size=64,
        feature_stride=8,
        sampling_seed=705,
        draws_per_video=2,
        context_bins=2,
    )
    video = dataset.current_episode_manifest["videos"][0]
    draw = video["draws"][0]
    original_hash = dataset.current_episode_manifest_sha256
    samples = {
        mode: dataset.build_gold_audit_sample(
            video, draw, mode, manifest_sha256=original_hash
        )
        for mode in ("video_start_full", "fixed_192", "dynamic_birth", "reset")
    }

    supervised_start = draw["supervised_range"][0]
    assert samples["video_start_full"]["crs_eps"]["replay_range"][0] == 0
    assert samples["fixed_192"]["crs_eps"]["replay_range"][0] == max(
        0, supervised_start - 2
    )
    assert samples["dynamic_birth"]["crs_eps"]["replay_range"] == draw["replay_range"]
    assert samples["reset"]["crs_eps"]["replay_range"][0] == supervised_start
    assert samples["reset"]["crs_eps"]["gradient_ranges"] == [
        draw["supervised_range"]
    ]
    assert all(sample["crs_eps"]["video_group_size"] == 1 for sample in samples.values())
    assert dataset.current_episode_manifest_sha256 == original_hash
