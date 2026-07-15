import json
from pathlib import Path

import numpy as np

from opentad.utils.crs_eps_sampling import (
    DEFAULT_CONTEXT_BINS,
    DEFAULT_DETACH_INTERVAL,
    DEFAULT_MIXTURE,
    DEFAULT_SUFFIX_BINS,
    build_epoch_manifest,
    canonical_json_sha256,
    episode_payload_sha256,
    validate_epoch_manifest,
)
from opentad.utils.evidence_bundle import publish_exclusive_file
from opentad.utils.prefix_instance_schedule import (
    PrefixScheduleStep,
    build_prefix_instance_schedule,
)

from .streaming_feature import StreamingFeatureDataset


class CrsEpsFeatureDataset(StreamingFeatureDataset):
    """Cached-feature CRS-EPS draws with annotation data confined to control."""

    sampling_protocol = "crs_eps"

    def __init__(
        self,
        *args,
        sampling_seed=None,
        draws_per_video,
        suffix_bins=DEFAULT_SUFFIX_BINS,
        context_bins=DEFAULT_CONTEXT_BINS,
        detach_interval=DEFAULT_DETACH_INTERVAL,
        proposal_mixture=None,
        **kwargs,
    ):
        if kwargs.get("test_mode", False):
            raise ValueError("CRS-EPS is a training-only dataset")
        self.sampling_seed = 0 if sampling_seed is None else int(sampling_seed)
        self.sampling_seed_bound_to_runtime = sampling_seed is not None
        self.draws_per_video = int(draws_per_video)
        self.suffix_bins = int(suffix_bins)
        self.context_bins = int(context_bins)
        self.detach_interval = int(detach_interval)
        self.proposal_mixture = dict(
            DEFAULT_MIXTURE if proposal_mixture is None else proposal_mixture
        )
        if self.sampling_seed < 0 or self.draws_per_video <= 0:
            raise ValueError("sampling seed/draw count are invalid")
        super().__init__(*args, **kwargs)
        self._manifest_provenance = {
            "annotation_sha256": self._manifest["annotation_sha256"],
            "feature_cache_manifest_sha256": self.cache_manifest_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "split_manifest_file_sha256": self.split_manifest_file_sha256,
            "split_role": self.split_role,
            "split_seed": self.split_seed,
        }
        self.current_sampling_epoch = None
        self.current_episode_manifest = None
        self.current_episode_manifest_sha256 = None
        self.set_epoch(0)

    def bind_sampling_seed(self, seed):
        if self.current_sampling_epoch not in (None, 0):
            raise RuntimeError("sampling seed must be bound before training advances")
        seed = int(seed)
        if seed < 0:
            raise ValueError("sampling seed must be non-negative")
        if self.sampling_seed_bound_to_runtime and self.sampling_seed != seed:
            raise RuntimeError("configured sampling seed conflicts with runtime seed")
        self.sampling_seed = seed
        self.sampling_seed_bound_to_runtime = True
        self.set_epoch(0)

    def bind_manifest_provenance(self, **provenance):
        if self.current_sampling_epoch not in (None, 0):
            raise RuntimeError("manifest provenance must be bound before training advances")
        for key, value in provenance.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                self._manifest_provenance[str(key)] = value
            else:
                raise TypeError("manifest provenance values must be JSON scalars")
        self.set_epoch(0)

    def set_epoch(self, epoch):
        epoch = int(epoch)
        manifest = build_epoch_manifest(
            self.iter_crs_eps_sampling_specs(),
            epoch=epoch,
            seed=self.sampling_seed,
            draws_per_video=self.draws_per_video,
            provenance=self._manifest_provenance,
            suffix_bins=self.suffix_bins,
            context_bins=self.context_bins,
            detach_interval=self.detach_interval,
            mixture=self.proposal_mixture,
        )
        validate_epoch_manifest(manifest)
        self.current_sampling_epoch = epoch
        self.current_episode_manifest = manifest
        self.current_episode_manifest_sha256 = manifest["manifest_sha256"]
        self.data_list = []
        self.packet_manifests = {}
        for video_group_index, video in enumerate(manifest["videos"]):
            indices = []
            for draw in video["draws"]:
                indices.append(len(self.data_list))
                self.data_list.append(
                    {
                        "video_group_index": video_group_index,
                        "video": video,
                        "draw": draw,
                    }
                )
            self.packet_manifests[video["video_id"]] = indices

    def persist_current_manifest(self, output_dir):
        output = Path(output_dir) / f"crs_eps_epoch_{self.current_sampling_epoch:04d}.json"
        encoded = (
            json.dumps(
                self.current_episode_manifest,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        publish_exclusive_file(output, encoded)
        return output

    @property
    def optimizer_events_per_epoch(self):
        return len(self.packet_manifests)

    @staticmethod
    def _remove_left_censored(schedule, instance_ids):
        censored = set(int(value) for value in instance_ids)
        if not censored:
            return schedule
        return tuple(
            PrefixScheduleStep(
                current_frame=step.current_frame,
                births=tuple(
                    target for target in step.births if target.instance_id not in censored
                ),
                active=tuple(
                    target for target in step.active if target.instance_id not in censored
                ),
                ends=tuple(
                    target for target in step.ends if target.instance_id not in censored
                ),
            )
            for step in schedule
        )

    @staticmethod
    def _left_censored_instance_ids(video, replay_start):
        replay_start = int(replay_start)
        return [
            int(instance["instance_id"])
            for instance in video["instances"]
            if replay_start in instance["active_bins"]
            and (
                instance["birth_bin"] is None
                or int(instance["birth_bin"]) < replay_start
            )
        ]

    def _build_sample(
        self,
        video,
        draw,
        *,
        video_group_index,
        manifest_sha256,
        draw_index=None,
        group_size=None,
    ):
        record = self.video_records[video["video_id"]]
        replay_start, replay_end = draw["replay_range"]
        source_frames = tuple(record["source_frames"][replay_start:replay_end])
        features = self._load_verified_feature_array(
            record["feature_path"],
            record["feature_sha256"],
            video["video_id"],
        )[replay_start:replay_end]
        inputs = np.asarray(features, dtype=np.float32).T.copy()
        masks = np.ones(replay_end - replay_start, dtype=np.bool_)
        previous_frame = (
            source_frames[0] - self.feature_stride
            if replay_start == 0
            else record["source_frames"][replay_start - 1]
        )
        schedule = build_prefix_instance_schedule(
            record["segments"],
            record["labels"],
            decision_frames=source_frames,
            previous_frame=previous_frame,
        )
        schedule = self._remove_left_censored(
            schedule, draw["left_censored_instance_ids"]
        )
        episode_id = draw["episode_id"]
        stream_id = f"{self.stream_id}:crs-eps:{episode_id}"
        meta = dict(
            video_name=video["video_id"],
            video_id=video["video_id"],
            stream_id=stream_id,
            input_format="cached_features",
            input_provenance_digest=record["input_provenance_digest"],
            feature_stride=self.feature_stride,
            feature_dim=record["feature_dim"],
            fps=record["fps"],
            source_frames=source_frames,
            current_frame=source_frames[-1],
            packet_start_token=replay_start,
            packet_end_token=replay_end,
        )
        draw_index = int(draw["draw_index"] if draw_index is None else draw_index)
        group_size = int(
            video["draws_per_video"] if group_size is None else group_size
        )
        sequence_sha256 = (
            video["episode_sequence_sha256"]
            if group_size == video["draws_per_video"] and draw in video["draws"]
            else canonical_json_sha256([draw["episode_payload_sha256"]])
        )
        return {
            "inputs": inputs,
            "masks": masks,
            "metas": meta,
            "stream_control": {
                "video_id": video["video_id"],
                "stream_id": stream_id,
                "chunk_index": 0,
                "is_video_start": True,
                "is_video_end": True,
            },
            "prefix_schedule": schedule,
            "crs_eps": {
                **draw,
                "video_id": video["video_id"],
                "draw_index": draw_index,
                "video_group_index": int(video_group_index),
                "video_group_size": group_size,
                "is_video_group_start": draw_index == 0,
                "is_video_group_end": draw_index + 1 == group_size,
                "episode_manifest_sha256": manifest_sha256,
                "episode_sequence_sha256": sequence_sha256,
                "video_covered_unique_bins": video["covered_unique_bins"],
                "video_effective_sample_size": video["effective_sample_size"],
                "video_ipw_weight_sum": video["ipw_weight_sum"],
                "video_ipw_weight_squared_sum": video["ipw_weight_squared_sum"],
            },
        }

    def build_gold_audit_sample(self, video, draw, mode, *, manifest_sha256):
        """Build one of the four frozen G0 replay arms without changing the dataset."""

        if mode not in {"video_start_full", "fixed_192", "dynamic_birth", "reset"}:
            raise ValueError(f"unsupported CRS-EPS gold audit mode: {mode}")
        if video["video_id"] not in self.video_records or draw not in video["draws"]:
            raise ValueError("gold audit draw is not owned by the supplied video")
        supervised_start, supervised_end = (int(value) for value in draw["supervised_range"])
        if mode == "video_start_full":
            replay_start = 0
        elif mode == "fixed_192":
            replay_start = max(0, supervised_start - self.context_bins)
        elif mode == "dynamic_birth":
            replay_start = int(draw["replay_range"][0])
        else:
            replay_start = supervised_start
        audit_draw = dict(draw)
        audit_draw["episode_id"] = f'{draw["episode_id"]}:g0:{mode}'
        audit_draw["replay_range"] = [replay_start, supervised_end]
        audit_draw["left_censored_instance_ids"] = self._left_censored_instance_ids(
            video, replay_start
        )
        if mode == "reset":
            audit_draw["gradient_ranges"] = [[supervised_start, supervised_end]]
        audit_draw["episode_payload_sha256"] = episode_payload_sha256(audit_draw)
        return self._build_sample(
            video,
            audit_draw,
            video_group_index=0,
            manifest_sha256=manifest_sha256,
            draw_index=0,
            group_size=1,
        )

    def __getitem__(self, index):
        item = self.data_list[index]
        return self._build_sample(
            item["video"],
            item["draw"],
            video_group_index=item["video_group_index"],
            manifest_sha256=self.current_episode_manifest_sha256,
        )


__all__ = ["CrsEpsFeatureDataset"]
