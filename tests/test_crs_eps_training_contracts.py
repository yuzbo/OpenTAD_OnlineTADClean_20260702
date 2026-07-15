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
from opentad.utils.crs_eps_sampling import (
    VideoSamplingSpec,
    build_epoch_manifest,
    canonical_json_sha256,
    episode_payload_sha256,
    epoch_manifest_data_order_sha256,
    epoch_manifest_runtime_control,
)
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
    def payload(index):
        return {
            "draw_index": index,
            "episode_id": f"episode-{index}",
            "proposal_component": "uniform",
            "component_fallback_to_uniform": False,
            "anchor_bin": supervised[0],
            "supervised_range": list(supervised),
            "replay_range": list(replay),
            "gradient_ranges": [list(value) for value in gradient],
            "true_left_censored": False,
            "left_censored_instance_ids": [],
            "dynamic_extension_instance_ids": [],
            "video_start_fallback": replay[0] == 0,
            "q_component": 0.4,
            "q_anchor_given_component": 1.0,
            "q_anchor_marginal": 0.1,
            "rho_by_supervised_bin": [0.4] * (supervised[1] - supervised[0]),
            "union_pi_by_supervised_bin": [0.4] * (supervised[1] - supervised[0]),
            "raw_weight_by_bin": [0.25] * (supervised[1] - supervised[0]),
            "final_weight_by_bin": [0.25] * (supervised[1] - supervised[0]),
            "rng_key": f"rng-{index}",
        }

    current = payload(draw_index)
    current["episode_payload_sha256"] = episode_payload_sha256(current)
    return {
        **current,
        "video_group_size": group_size,
        "is_video_group_start": draw_index == 0,
        "is_video_group_end": draw_index + 1 == group_size,
        "episode_manifest_sha256": "b" * 64,
        "episode_sequence_sha256": canonical_json_sha256(
            [episode_payload_sha256(payload(index)) for index in range(group_size)]
        ),
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
    def __init__(
        self,
        mutate_buffer=False,
        slot_exhaustion=0,
        fail_on_call=None,
        nonfinite_on_call=None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("audit_buffer", torch.tensor(0))
        self.mutate_buffer = mutate_buffer
        self.slot_exhaustion = int(slot_exhaustion)
        self.fail_on_call = fail_on_call
        self.nonfinite_on_call = nonfinite_on_call
        self.forward_calls = 0
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
        self.forward_calls += 1
        self.pending = True
        if self.forward_calls == self.fail_on_call:
            self.audit_buffer.add_(1)
            raise RuntimeError("injected interior draw failure")
        if self.mutate_buffer:
            self.audit_buffer.add_(1)
        self.last_episode_audit = {
            "slot_exhaustion": self.slot_exhaustion,
            "control_unroll_seconds": 0.0,
        }
        cost = (self.weight - inputs).pow(2).mean()
        if self.forward_calls == self.nonfinite_on_call:
            cost = cost * cost.new_tensor(float("nan"))
        scale = cost.new_tensor(1.0 / crs_eps[0]["video_group_size"])
        return {
            "cost": cost,
            "_optimizer_weight": scale,
            "_optimizer_denominator": scale,
        }


def _engine_manifest(group_size=2, *, seed=705):
    return build_epoch_manifest(
        [VideoSamplingSpec(video_id="video", num_bins=1, instances=())],
        epoch=0,
        seed=seed,
        draws_per_video=group_size,
        suffix_bins=1,
        context_bins=1,
        detach_interval=1,
    )


def _engine_crs_control(draw_index, group_size=2, *, manifest=None):
    manifest = _engine_manifest(group_size) if manifest is None else manifest
    return epoch_manifest_runtime_control(manifest, 0, draw_index)


def test_crs_data_order_identity_is_manifest_derived_and_epoch_stable():
    epoch_zero = _engine_manifest(4, seed=705)
    epoch_one = build_epoch_manifest(
        [VideoSamplingSpec(video_id="video", num_bins=1, instances=())],
        epoch=1,
        seed=705,
        draws_per_video=4,
        suffix_bins=1,
        context_bins=1,
        detach_interval=1,
    )

    assert epoch_manifest_data_order_sha256(
        epoch_zero
    ) == epoch_manifest_data_order_sha256(epoch_one)
    assert epoch_manifest_data_order_sha256(
        epoch_zero
    ) != epoch_manifest_data_order_sha256(_engine_manifest(4, seed=706))
    assert epoch_manifest_data_order_sha256(
        epoch_zero
    ) != epoch_manifest_data_order_sha256(_engine_manifest(3, seed=705))


def _crs_loader(*, manifest=None):
    manifest = _engine_manifest() if manifest is None else manifest
    return [
        {
            "inputs": torch.tensor([1.0]),
            "stream_control": [
                {"video_id": "video", "is_video_start": True, "is_video_end": True}
            ],
            "crs_eps": [_engine_crs_control(0, manifest=manifest)],
        },
        {
            "inputs": torch.tensor([3.0]),
            "stream_control": [
                {"video_id": "video", "is_video_start": True, "is_video_end": True}
            ],
            "crs_eps": [_engine_crs_control(1, manifest=manifest)],
        },
    ]


def _crs_loader_with_indices(
    indices, *, group_size=4, cross_manifest_at=None, manifest=None
):
    manifest = _engine_manifest(group_size) if manifest is None else manifest
    rows = []
    for position, draw_index in enumerate(indices):
        control = _engine_crs_control(
            draw_index, group_size=group_size, manifest=manifest
        )
        if position == cross_manifest_at:
            control["episode_manifest_sha256"] = "d" * 64
        rows.append(
            {
                "inputs": torch.tensor([float(position + 1)]),
                "stream_control": [
                    {
                        "video_id": "video",
                        "is_video_start": True,
                        "is_video_end": True,
                    }
                ],
                "crs_eps": [control],
            }
        )
    return rows


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
        crs_eps_manifest=_engine_manifest(),
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
            crs_eps_manifest=_engine_manifest(),
        )

    assert model.rollbacks == 1
    assert model.weight.item() == 0.0
    assert model.audit_buffer.item() == 0


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
            crs_eps_manifest=_engine_manifest(),
        )

    assert model.rollbacks == 1
    assert model.weight.item() == 0.0


@pytest.mark.parametrize("indices", ([0, 1, 1, 3], [0, 2, 1, 3]))
def test_engine_rejects_duplicate_or_reordered_manifest_draws(indices):
    model = _CrsToy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="membership/order"):
        train_one_epoch(
            _crs_loader_with_indices(indices),
            model,
            optimizer,
            _Scheduler(),
            curr_epoch=0,
            logger=_Logger(),
            crs_eps_manifest=_engine_manifest(4),
        )

    assert model.rollbacks == 1
    assert model.weight.item() == 0.0


def test_engine_rejects_missing_manifest_draw_and_rolls_back_at_epoch_end():
    model = _CrsToy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="epoch ended inside"):
        train_one_epoch(
            _crs_loader_with_indices([0, 1, 2]),
            model,
            optimizer,
            _Scheduler(),
            curr_epoch=0,
            logger=_Logger(),
            crs_eps_manifest=_engine_manifest(4),
        )

    assert model.rollbacks == 1
    assert model.weight.item() == 0.0


def test_engine_rejects_cross_manifest_draw_before_optimizer_boundary():
    model = _CrsToy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="escaped its active video group"):
        train_one_epoch(
            _crs_loader_with_indices([0, 1, 2, 3], cross_manifest_at=1),
            model,
            optimizer,
            _Scheduler(),
            curr_epoch=0,
            logger=_Logger(),
            crs_eps_manifest=_engine_manifest(4),
        )

    assert model.rollbacks == 1


def test_engine_rejects_tampered_episode_membership_digest_at_boundary():
    model = _CrsToy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = _crs_loader_with_indices([0, 1, 2, 3])
    loader[2]["crs_eps"][0]["episode_id"] = "substituted-episode"
    loader[2]["crs_eps"][0]["episode_payload_sha256"] = episode_payload_sha256(
        loader[2]["crs_eps"][0]
    )

    with pytest.raises(RuntimeError, match="draw payload differs"):
        train_one_epoch(
            loader,
            model,
            optimizer,
            _Scheduler(),
            curr_epoch=0,
            logger=_Logger(),
            crs_eps_manifest=_engine_manifest(4),
        )

    assert model.rollbacks == 1
    assert model.weight.item() == 0.0


def test_engine_rejects_substituted_draw_payload_even_with_expected_episode_id():
    model = _CrsToy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = _crs_loader_with_indices([0, 1, 2, 3])
    loader[1]["crs_eps"][0]["proposal_component"] = "end"

    with pytest.raises(RuntimeError, match="draw payload differs"):
        train_one_epoch(
            loader,
            model,
            optimizer,
            _Scheduler(),
            curr_epoch=0,
            logger=_Logger(),
            crs_eps_manifest=_engine_manifest(4),
        )

    assert model.rollbacks == 1
    assert model.weight.item() == 0.0


def test_engine_rejects_whole_rehashed_group_not_owned_by_trusted_manifest():
    trusted_manifest = _engine_manifest(4, seed=705)
    substituted_manifest = _engine_manifest(4, seed=706)
    loader = _crs_loader_with_indices(
        [0, 1, 2, 3], manifest=substituted_manifest
    )
    model = _CrsToy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="immutable manifest"):
        train_one_epoch(
            loader,
            model,
            optimizer,
            _Scheduler(),
            curr_epoch=0,
            logger=_Logger(),
            crs_eps_manifest=trusted_manifest,
        )

    assert model.commits == 0
    assert model.weight.item() == 0.0


def test_second_draw_control_parse_failure_rolls_back_group():
    manifest = _engine_manifest(4)
    loader = _crs_loader_with_indices([0, 1, 2, 3], manifest=manifest)
    loader[1]["crs_eps"][0].pop("episode_payload_sha256")
    model = _CrsToy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="lacks video-group fields"):
        train_one_epoch(
            loader,
            model,
            optimizer,
            _Scheduler(),
            curr_epoch=0,
            logger=_Logger(),
            crs_eps_manifest=manifest,
        )

    assert model.pending is False
    assert model.rollbacks == 1
    assert model.weight.grad is None


class _InterruptedCrsLoader:
    def __init__(self, first_row, *, exception):
        self.first_row = first_row
        self.exception = exception

    def __len__(self):
        return 2

    def __iter__(self):
        return _InterruptedCrsIterator(self.first_row, self.exception)


class _InterruptedCrsIterator:
    def __init__(self, first_row, exception):
        self.first_row = first_row
        self.exception = exception
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position == 0:
            self.position += 1
            return self.first_row
        raise self.exception


@pytest.mark.parametrize(
    "exception, match",
    ((RuntimeError("loader failed"), "loader failed"), (StopIteration(), "")),
)
def test_mid_group_loader_failure_rolls_back_staged_state(exception, match):
    manifest = _engine_manifest(4)
    first_row = _crs_loader_with_indices([0], manifest=manifest)[0]
    loader = _InterruptedCrsLoader(first_row, exception=exception)
    model = _CrsToy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    expected = pytest.raises(type(exception), match=match) if match else pytest.raises(
        type(exception)
    )
    with expected:
        train_one_epoch(
            loader,
            model,
            optimizer,
            _Scheduler(),
            curr_epoch=0,
            logger=_Logger(),
            crs_eps_manifest=manifest,
        )

    assert model.pending is False
    assert model.rollbacks == 1
    assert model.weight.grad is None


def test_interior_forward_exception_restores_group_buffers_and_staged_state():
    model = _CrsToy(fail_on_call=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="injected interior draw failure"):
        train_one_epoch(
            _crs_loader_with_indices([0, 1, 2, 3]),
            model,
            optimizer,
            _Scheduler(),
            curr_epoch=0,
            logger=_Logger(),
            crs_eps_manifest=_engine_manifest(4),
        )

    assert model.audit_buffer.item() == 0
    assert model.pending is False
    assert model.rollbacks == 1
    assert model.weight.item() == 0.0


def test_interior_nonfinite_loss_rolls_back_before_skipping_to_group_end():
    model = _CrsToy(nonfinite_on_call=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    stats = train_one_epoch(
        _crs_loader_with_indices([0, 1, 2, 3]),
        model,
        optimizer,
        _Scheduler(),
        curr_epoch=0,
        logger=_Logger(),
        crs_eps_manifest=_engine_manifest(4),
    )

    assert stats["successful_optimizer_events"] == 0
    assert stats["skipped_optimizer_events"] == 1
    assert model.pending is False
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


def test_crs_dataset_uses_verified_feature_bytes_after_path_replacement(tmp_path):
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
    np.save(
        feature_dir / "video.npy",
        np.full((12, 4), 777.0, dtype=np.float32),
    )

    sample = dataset[0]
    assert float(sample["inputs"].max()) < 777.0


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
        crs_eps_manifest=dataset.current_episode_manifest,
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
