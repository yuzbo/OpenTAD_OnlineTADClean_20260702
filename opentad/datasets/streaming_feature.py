import hashlib
import json
import numbers
from pathlib import Path

import numpy as np

from opentad.utils.full_petal_data_contract import (
    THUMOS_DEVELOPMENT_SPLIT_SCHEMA_VERSION,
    canonical_json_sha256,
    load_id_file,
    load_json,
    verify_content_hash,
)
from opentad.utils.prefix_instance_schedule import build_prefix_instance_schedule
from opentad.utils.crs_eps_sampling import video_sampling_spec_from_schedule


_FORBIDDEN_MODEL_META = {
    "duration",
    "total_frames",
    "num_frames",
    "is_video_end",
    "video_end_frame",
}


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value):
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StreamingFeatureDataset:
    """Chronological fixed-feature chunks with training-only prefix labels."""

    def __init__(
        self,
        ann_file,
        subset_name,
        class_map,
        data_path,
        cache_manifest,
        chunk_size=128,
        feature_stride=8,
        stream_id="pes-stage1-v1",
        test_mode=False,
        allow_list=None,
        block_list=None,
        split_manifest=None,
        split_role=None,
        split_seed=None,
        logger=None,
    ):
        self.ann_file = Path(ann_file)
        self.data_path = Path(data_path)
        self.cache_manifest_path = Path(cache_manifest)
        self.cache_manifest_sha256 = _sha256_file(self.cache_manifest_path)
        self.chunk_size = int(chunk_size)
        self.feature_stride = int(feature_stride)
        self.stream_id = str(stream_id)
        self.test_mode = bool(test_mode)
        self.logger = logger.info if logger is not None else (lambda *_: None)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.feature_stride <= 0:
            raise ValueError("feature_stride must be positive")

        self.class_map = self._load_class_map(class_map)
        self._class_to_index = {name: index for index, name in enumerate(self.class_map)}
        self.allow_list_path = (
            Path(allow_list)
            if allow_list is not None and not isinstance(allow_list, (list, tuple, set))
            else None
        )
        self._allowed = self._load_name_filter(allow_list)
        self._blocked = self._load_name_filter(block_list) or set()
        self._subsets = {subset_name} if isinstance(subset_name, str) else set(subset_name)
        self._split_contract = self._load_split_contract(
            split_manifest,
            split_role,
            split_seed,
        )
        self._manifest = self._load_manifest()
        self.data_list = []
        self.packet_manifests = {}
        self.crs_eps_sampling_specs = {}
        self.video_records = {}
        self._build_index()
        if self._split_contract is not None:
            selected_ids = sorted(self.packet_manifests)
            if selected_ids != self._split_contract["ids"]:
                raise ValueError(
                    "dataset videos do not exactly consume the locked split role; "
                    f"expected {self._split_contract['ids']}, found {selected_ids}"
                )
        if not self.data_list:
            raise ValueError(f"no cached feature chunks found for subsets {sorted(self._subsets)}")

    @property
    def optimizer_events_per_epoch(self):
        """One transactional optimizer event is emitted per complete video episode."""

        return len(self.packet_manifests)

    @staticmethod
    def _load_class_map(path):
        with Path(path).open("r", encoding="utf-8") as handle:
            return tuple(line.strip() for line in handle if line.strip())

    @staticmethod
    def _load_name_filter(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            return {str(item) for item in value}
        with Path(value).open("r", encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}

    def _load_split_contract(self, split_manifest, split_role, split_seed):
        provided = (split_manifest is not None, split_role is not None, split_seed is not None)
        if not any(provided):
            self.split_manifest_path = None
            self.split_manifest_sha256 = None
            self.split_manifest_file_sha256 = None
            self.split_role = None
            self.split_seed = None
            return None
        if not all(provided):
            raise ValueError(
                "split_manifest, split_role, and split_seed must be provided together"
            )
        if self.allow_list_path is None:
            raise ValueError("a locked split contract requires a physical allow-list file")
        if self._blocked:
            raise ValueError("a locked split contract forbids an additional block-list")
        if isinstance(split_seed, bool) or not isinstance(split_seed, numbers.Integral):
            raise ValueError("split seed must be an integer")
        split_seed = int(split_seed)

        manifest_path = Path(split_manifest)
        manifest = load_json(manifest_path)
        if manifest.get("schema") != "full_petal.thumos_development_split":
            raise ValueError("split manifest has an unsupported schema")
        if manifest.get("schema_version") != THUMOS_DEVELOPMENT_SPLIT_SCHEMA_VERSION:
            raise ValueError("split manifest has an unsupported schema version")
        if not verify_content_hash(manifest):
            raise ValueError("split manifest content hash does not verify")
        if manifest.get("seed") != split_seed:
            raise ValueError(
                f"split seed mismatch: expected {split_seed}, found {manifest.get('seed')}"
            )
        role = str(split_role)
        artifact_names = {
            "fit_core": "fit_ids",
            "calibration": "calibration_ids",
        }
        if role not in artifact_names:
            raise ValueError(f"unsupported split role {role!r}")
        split_records = manifest.get("splits")
        if not isinstance(split_records, dict):
            raise ValueError("split manifest is missing split records")
        role_ids = {}
        for declared_role in artifact_names:
            split_record = split_records.get(declared_role)
            if not isinstance(split_record, dict):
                raise ValueError(f"split manifest is missing role {declared_role!r}")
            declared_ids = split_record.get("ids")
            if not isinstance(declared_ids, list) or not all(
                isinstance(video_id, str) and video_id for video_id in declared_ids
            ):
                raise ValueError(
                    f"split role {declared_role!r} must contain non-empty text IDs"
                )
            if declared_ids != sorted(set(declared_ids)):
                raise ValueError(
                    f"split role {declared_role!r} IDs must be sorted and unique"
                )
            if split_record.get("count") != len(declared_ids):
                raise ValueError(
                    f"split role {declared_role!r} count does not match its IDs"
                )
            if split_record.get("ids_sha256") != canonical_json_sha256(declared_ids):
                raise ValueError(f"split role {declared_role!r} ID hash does not verify")
            role_ids[declared_role] = declared_ids
        overlap = sorted(set(role_ids["fit_core"]).intersection(role_ids["calibration"]))
        if overlap:
            raise ValueError("fit_core and calibration split roles overlap: " + ", ".join(overlap))
        universe = manifest.get("universe")
        if not isinstance(universe, dict):
            raise ValueError("split manifest is missing its development universe")
        universe_ids = universe.get("ids")
        combined_ids = sorted([*role_ids["fit_core"], *role_ids["calibration"]])
        if universe_ids != combined_ids:
            raise ValueError("split roles do not exactly partition the development universe")
        if universe.get("count") != len(universe_ids):
            raise ValueError("development universe count does not match its IDs")
        if universe.get("ids_sha256") != canonical_json_sha256(universe_ids):
            raise ValueError("development universe ID hash does not verify")
        ids = role_ids[role]

        annotation = load_json(self.ann_file)
        expected_annotation_hash = manifest.get("annotation", {}).get(
            "canonical_sha256"
        )
        if expected_annotation_hash != canonical_json_sha256(annotation):
            raise ValueError("split manifest annotation hash does not match dataset")

        artifact = manifest.get("artifacts", {}).get(artifact_names[role])
        if not isinstance(artifact, dict):
            raise ValueError(f"split manifest is missing the {role!r} allow-list artifact")
        if artifact.get("name") != self.allow_list_path.name:
            raise ValueError("split allow-list filename does not match manifest")
        if artifact.get("sha256") != _sha256_file(self.allow_list_path):
            raise ValueError("split allow-list file hash does not verify")
        allow_ids = load_id_file(self.allow_list_path)
        if artifact.get("content_sha256") != canonical_json_sha256(allow_ids):
            raise ValueError("split allow-list content hash does not verify")
        if allow_ids != ids:
            raise ValueError(
                f"allow-list does not exactly match locked split role {role!r}"
            )

        self._allowed = set(ids)
        self.split_manifest_path = manifest_path
        self.split_manifest_sha256 = manifest["manifest_sha256"]
        self.split_manifest_file_sha256 = _sha256_file(manifest_path)
        self.split_role = role
        self.split_seed = split_seed
        return {
            "ids": ids,
            "manifest": manifest,
        }

    def _load_manifest(self):
        manifest = load_json(self.cache_manifest_path)
        if manifest.get("schema") != "ontad_feature_cache_v1":
            raise ValueError("cache manifest must use schema ontad_feature_cache_v1")
        if manifest.get("feature_policy") != "packet_recent_frame":
            raise ValueError("cache manifest must use packet_recent_frame features")
        if manifest.get("timestamp_convention") != "zero_based_source_frame":
            raise ValueError("cache manifest timestamp convention is incompatible")
        if int(manifest.get("feature_stride", -1)) != self.feature_stride:
            raise ValueError("cache manifest feature_stride does not match dataset")
        if manifest.get("annotation_sha256") != _sha256_file(self.ann_file):
            raise ValueError("cache manifest annotation hash does not match dataset annotations")
        if not str(manifest.get("encoder_id", "")).strip():
            raise ValueError("cache manifest must identify the frozen encoder")
        try:
            dtype = np.dtype(manifest["dtype"])
        except (KeyError, TypeError) as exc:
            raise ValueError("cache manifest must contain a valid dtype") from exc
        if dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError("cache manifest dtype must be float16 or float32")
        if not isinstance(manifest.get("videos"), dict):
            raise ValueError("cache manifest must contain a videos mapping")
        return manifest

    def _feature_path(self, video_name, video_manifest):
        raw_path = Path(video_manifest.get("file", f"{video_name}.npy"))
        return raw_path if raw_path.is_absolute() else self.data_path / raw_path

    def _annotations(self, video_info):
        if self.test_mode:
            return (), ()
        duration = float(video_info["duration"])
        num_frames = int(video_info["frame"])
        scale = num_frames / max(duration, 1e-6)
        segments = []
        labels = []
        for annotation in video_info.get("annotations", ()):
            label_name = annotation["label"]
            if label_name == "Ambiguous":
                continue
            if label_name not in self._class_to_index:
                raise ValueError(f"unknown action label {label_name!r}")
            start, end = annotation["segment"]
            segments.append((float(start) * scale, float(end) * scale))
            labels.append(self._class_to_index[label_name])
        return tuple(segments), tuple(labels)

    def _validate_video_cache(self, video_name, video_manifest, video_info):
        feature_path = self._feature_path(video_name, video_manifest)
        if not feature_path.is_file():
            raise FileNotFoundError(f"missing cached feature file: {feature_path}")
        expected_feature_hash = video_manifest.get("sha256")
        if not isinstance(expected_feature_hash, str) or len(expected_feature_hash) != 64:
            raise ValueError(f"cache manifest is missing feature hash for {video_name}")
        if _sha256_file(feature_path) != expected_feature_hash:
            raise ValueError(f"cached feature hash mismatch for {video_name}")
        features = np.load(feature_path, mmap_mode="r")
        if features.ndim != 2:
            raise ValueError(f"cached features for {video_name} must have shape [T,C]")
        num_tokens, feature_dim = map(int, features.shape)
        if num_tokens <= 0:
            raise ValueError(f"cached features for {video_name} must contain at least one token")
        if int(video_manifest.get("num_tokens", -1)) != num_tokens:
            raise ValueError(f"cache num_tokens mismatch for {video_name}")
        expected_dim = int(self._manifest.get("feature_dim", feature_dim))
        if feature_dim != expected_dim:
            raise ValueError(f"cache feature_dim mismatch for {video_name}")
        if features.dtype != np.dtype(self._manifest["dtype"]):
            raise ValueError(f"cache dtype mismatch for {video_name}")
        source_frames = tuple(int(frame) for frame in video_manifest.get("source_frames", ()))
        if len(source_frames) != num_tokens:
            raise ValueError(f"cache source_frames mismatch for {video_name}")
        if any(right <= left for left, right in zip(source_frames, source_frames[1:])):
            raise ValueError(f"cache source_frames must be strictly increasing for {video_name}")
        if source_frames and source_frames[-1] >= int(video_info["frame"]):
            raise ValueError(f"cache source frame exceeds video length for {video_name}")
        return feature_path, source_frames, feature_dim

    def _build_index(self):
        database = load_json(self.ann_file)["database"]
        manifest_videos = self._manifest["videos"]

        for video_name, video_info in database.items():
            if video_info.get("subset") not in self._subsets:
                continue
            if video_name in self._blocked:
                continue
            if self._allowed is not None and video_name not in self._allowed:
                continue
            if video_name not in manifest_videos:
                raise ValueError(f"cache manifest is missing video {video_name}")

            feature_path, source_frames, feature_dim = self._validate_video_cache(
                video_name,
                manifest_videos[video_name],
                video_info,
            )
            segments, labels = self._annotations(video_info)
            if not self.test_mode:
                full_schedule = build_prefix_instance_schedule(
                    segments,
                    labels,
                    decision_frames=source_frames,
                    previous_frame=source_frames[0] - self.feature_stride,
                )
                self.crs_eps_sampling_specs[video_name] = (
                    video_sampling_spec_from_schedule(video_name, full_schedule)
                )
            input_provenance_digest = _canonical_json_sha256(
                {
                    "annotation_sha256": self._manifest["annotation_sha256"],
                    "cache_manifest_sha256": self.cache_manifest_sha256,
                    "encoder_id": self._manifest["encoder_id"],
                    "feature_sha256": manifest_videos[video_name]["sha256"],
                    "source_frames": source_frames,
                    "split_manifest_sha256": self.split_manifest_sha256,
                    "split_manifest_file_sha256": self.split_manifest_file_sha256,
                    "split_role": self.split_role,
                    "split_seed": self.split_seed,
                    "video_id": video_name,
                }
            )
            self.video_records[video_name] = dict(
                video_name=video_name,
                feature_path=feature_path,
                feature_dim=feature_dim,
                fps=float(video_info["frame"])
                / max(float(video_info["duration"]), 1e-6),
                source_frames=source_frames,
                segments=segments,
                labels=labels,
                input_provenance_digest=input_provenance_digest,
            )
            packet_indices = []
            for chunk_index, start in enumerate(range(0, len(source_frames), self.chunk_size)):
                end = min(start + self.chunk_size, len(source_frames))
                packet_indices.append(len(self.data_list))
                self.data_list.append(
                    dict(
                        video_name=video_name,
                        feature_path=feature_path,
                        feature_dim=feature_dim,
                        fps=float(video_info["frame"]) / max(float(video_info["duration"]), 1e-6),
                        source_frames=source_frames,
                        chunk_index=chunk_index,
                        start_token=start,
                        end_token=end,
                        segments=segments,
                        labels=labels,
                        input_provenance_digest=input_provenance_digest,
                    )
                )
            self.packet_manifests[video_name] = packet_indices

        self.logger(
            f"{len(self.packet_manifests)} videos, {len(self.data_list)} chronological feature chunks"
        )

    def iter_crs_eps_sampling_specs(self):
        if self.test_mode:
            raise RuntimeError("test datasets cannot expose annotation-guided sampling specs")
        return tuple(
            self.crs_eps_sampling_specs[video_id]
            for video_id in sorted(self.crs_eps_sampling_specs)
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]
        start = item["start_token"]
        end = item["end_token"]
        source_frames = tuple(item["source_frames"][start:end])
        features = np.load(item["feature_path"], mmap_mode="r")[start:end]
        inputs = np.asarray(features, dtype=np.float32).T.copy()
        masks = np.ones(end - start, dtype=np.bool_)

        meta = dict(
            video_name=item["video_name"],
            video_id=item["video_name"],
            stream_id=self.stream_id,
            input_format="cached_features",
            input_provenance_digest=item["input_provenance_digest"],
            feature_stride=self.feature_stride,
            feature_dim=item["feature_dim"],
            fps=item["fps"],
            source_frames=source_frames,
            current_frame=source_frames[-1],
            packet_start_token=start,
            packet_end_token=end,
        )
        if not _FORBIDDEN_MODEL_META.isdisjoint(meta):
            raise RuntimeError("terminal metadata escaped into model metadata")

        is_start = start == 0
        is_end = end == len(item["source_frames"])
        sample = dict(
            inputs=inputs,
            masks=masks,
            metas=meta,
            stream_control=dict(
                video_id=item["video_name"],
                chunk_index=item["chunk_index"],
                is_video_start=is_start,
                is_video_end=is_end,
            ),
        )
        if not self.test_mode:
            previous_frame = source_frames[0] - self.feature_stride if is_start else item["source_frames"][start - 1]
            sample["prefix_schedule"] = build_prefix_instance_schedule(
                item["segments"],
                item["labels"],
                decision_frames=source_frames,
                previous_frame=previous_frame,
            )
        return sample
