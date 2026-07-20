import hashlib
import json
from pathlib import Path

import numpy as np

from opentad.utils.prefix_instance_schedule import build_prefix_instance_schedule


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
        strict_causal_control=False,
        logger=None,
    ):
        self.ann_file = Path(ann_file)
        self.data_path = Path(data_path)
        self.cache_manifest_path = Path(cache_manifest)
        self.chunk_size = int(chunk_size)
        self.feature_stride = int(feature_stride)
        self.stream_id = str(stream_id)
        self.test_mode = bool(test_mode)
        self.strict_causal_control = bool(strict_causal_control)
        self.logger = logger.info if logger is not None else (lambda *_: None)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.feature_stride <= 0:
            raise ValueError("feature_stride must be positive")

        self.class_map = self._load_class_map(class_map)
        self._class_to_index = {name: index for index, name in enumerate(self.class_map)}
        self._allowed = self._load_name_filter(allow_list)
        self._blocked = self._load_name_filter(block_list) or set()
        self._subsets = {subset_name} if isinstance(subset_name, str) else set(subset_name)
        self._manifest = self._load_manifest()
        self.data_list = []
        self.packet_manifests = {}
        self._build_index()
        if not self.data_list:
            raise ValueError(f"no cached feature chunks found for subsets {sorted(self._subsets)}")

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

    def _load_manifest(self):
        with self.cache_manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
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
        features = np.load(feature_path, mmap_mode="r")
        if features.ndim != 2:
            raise ValueError(f"cached features for {video_name} must have shape [T,C]")
        num_tokens, feature_dim = map(int, features.shape)
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
        with self.ann_file.open("r", encoding="utf-8") as handle:
            database = json.load(handle)["database"]
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
                    )
                )
            self.packet_manifests[video_name] = packet_indices

        self.logger(
            f"{len(self.packet_manifests)} videos, {len(self.data_list)} chronological feature chunks"
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
        stream_control = dict(
            video_id=item["video_name"],
            is_video_start=is_start,
        )
        if not self.strict_causal_control:
            stream_control.update(
                chunk_index=item["chunk_index"],
                is_video_end=is_end,
            )
        sample = dict(
            inputs=inputs,
            masks=masks,
            metas=meta,
            stream_control=stream_control,
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
