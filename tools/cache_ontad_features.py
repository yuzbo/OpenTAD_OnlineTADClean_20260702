import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def recent_packet_source_frames(total_frames, feature_stride):
    total_frames = int(total_frames)
    feature_stride = int(feature_stride)
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if feature_stride <= 0:
        raise ValueError("feature_stride must be positive")
    return tuple(min(start + feature_stride, total_frames) - 1 for start in range(0, total_frames, feature_stride))


def encode_selected_frames(frame_iter, source_frames, encode_batch, batch_size):
    source_frames = tuple(int(frame) for frame in source_frames)
    if any(right <= left for left, right in zip(source_frames, source_frames[1:])):
        raise ValueError("source_frames must be strictly increasing")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    selected = set(source_frames)
    batch = []
    outputs = []
    seen = []
    for frame_index, frame in frame_iter:
        frame_index = int(frame_index)
        if frame_index not in selected:
            continue
        batch.append(np.asarray(frame))
        seen.append(frame_index)
        if len(batch) == batch_size:
            encoded = np.asarray(encode_batch(np.stack(batch, axis=0)))
            if encoded.ndim != 2 or encoded.shape[0] != len(batch):
                raise ValueError("encode_batch must return [batch, feature_dim]")
            outputs.append(encoded)
            batch = []
    if batch:
        encoded = np.asarray(encode_batch(np.stack(batch, axis=0)))
        if encoded.ndim != 2 or encoded.shape[0] != len(batch):
            raise ValueError("encode_batch must return [batch, feature_dim]")
        outputs.append(encoded)
    if tuple(seen) != source_frames:
        missing = sorted(set(source_frames).difference(seen))
        raise RuntimeError(f"video reader did not produce selected source frames: {missing}")
    if not outputs:
        raise RuntimeError("feature extraction produced no outputs")
    return np.concatenate(outputs, axis=0)


def write_feature_file(path, features, source_frames, resume=False, expected_dtype=None):
    path = Path(path)
    source_frames = tuple(int(frame) for frame in source_frames)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and resume:
        existing = np.load(path, mmap_mode="r")
        expected_shape = None if features is None else tuple(np.asarray(features).shape)
        if existing.ndim != 2 or existing.shape[0] != len(source_frames):
            raise ValueError("existing feature geometry does not match requested cache")
        if expected_shape is not None and tuple(existing.shape) != expected_shape:
            raise ValueError("existing feature geometry does not match requested cache")
        if expected_dtype is not None and existing.dtype != np.dtype(expected_dtype):
            raise ValueError("existing feature dtype does not match requested cache")
        return {
            "file": path.name,
            "num_tokens": int(existing.shape[0]),
            "feature_dim": int(existing.shape[1]),
            "dtype": str(existing.dtype),
            "source_frames": list(source_frames),
            "sha256": sha256_file(path),
            "resumed": True,
        }
    if path.exists():
        raise FileExistsError(f"feature cache already exists: {path}; use --resume")
    if features is None:
        raise ValueError("features are required when creating a cache file")
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[0] != len(source_frames):
        raise ValueError("feature/source geometry mismatch")

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, features)
    os.replace(temporary, path)
    return {
        "file": path.name,
        "num_tokens": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "dtype": str(features.dtype),
        "source_frames": list(source_frames),
        "sha256": sha256_file(path),
        "resumed": False,
    }


def write_video_cache_record(path, metadata, encoder_id, feature_stride):
    path = Path(path)
    payload = {
        "schema": "ontad_feature_video_v1",
        "encoder_id": str(encoder_id),
        "feature_stride": int(feature_stride),
        **dict(metadata),
    }
    payload.pop("resumed", None)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return payload


def validate_video_cache_record(path, metadata, encoder_id, feature_stride):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"resume requires a per-video cache record: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "schema": "ontad_feature_video_v1",
        "encoder_id": str(encoder_id),
        "feature_stride": int(feature_stride),
        **{key: value for key, value in metadata.items() if key != "resumed"},
    }
    if payload != expected:
        raise ValueError(f"per-video cache record does not match resume contract: {path}")
    return payload


def write_cache_manifest(
    path,
    annotation_path,
    encoder_id,
    feature_stride,
    feature_dim,
    dtype,
    videos,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ontad_feature_cache_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "annotation_file": Path(annotation_path).name,
        "annotation_sha256": sha256_file(annotation_path),
        "encoder_id": str(encoder_id),
        "feature_policy": "packet_recent_frame",
        "timestamp_convention": "zero_based_source_frame",
        "feature_stride": int(feature_stride),
        "feature_dim": int(feature_dim),
        "dtype": str(dtype),
        "videos": videos,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return payload


def iter_video_frames(video_path, image_size):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("cache_ontad_features requires opencv-python") from exc
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if image_size:
                frame = cv2.resize(frame, (int(image_size), int(image_size)), interpolation=cv2.INTER_LINEAR)
            yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()


def build_siglip_encoder(model_name, device, image_size, batch_size, local_files_only=True):
    import torch

    from opentad.models.backbones.online_siglip_adapter import OnlineSigLIPFrameEncoder

    encoder = OnlineSigLIPFrameEncoder(
        model_name=model_name,
        backend="transformers",
        image_size=int(image_size),
        pooling="mean",
        frame_stride=1,
        frame_chunk_size=int(batch_size),
        freeze_vision_encoder=True,
        strict_online=True,
        local_files_only=bool(local_files_only),
        output_layout="bct",
    ).to(device).eval()

    def encode(frames):
        pixels = torch.from_numpy(np.ascontiguousarray(frames)).to(device=device)
        pixels = pixels.permute(0, 3, 1, 2).float().div_(255.0)
        pixels = pixels.permute(1, 0, 2, 3).unsqueeze(0)
        masks = torch.ones(1, pixels.shape[2], dtype=torch.bool, device=device)
        with torch.no_grad():
            features, _ = encoder(pixels, masks=masks, metas=None)
        return features[0].transpose(0, 1).float().cpu().numpy()

    return encode


def parse_args():
    parser = argparse.ArgumentParser(description="Cache fixed causal frame features for matched On-TAD pilots")
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--subset", action="append", default=None)
    parser.add_argument("--video-suffix", default=".mp4")
    parser.add_argument("--feature-stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--allow-list")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    return parser.parse_args()


def _load_allow_list(path):
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def main():
    args = parse_args()
    annotation_path = Path(args.ann_file)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with annotation_path.open("r", encoding="utf-8") as handle:
        database = json.load(handle)["database"]
    subsets = set(args.subset or ("training", "validation"))
    allow_list = _load_allow_list(args.allow_list)
    selected_videos = [
        (name, info)
        for name, info in sorted(database.items())
        if info.get("subset") in subsets and (allow_list is None or name in allow_list)
    ]
    if args.max_videos is not None:
        selected_videos = selected_videos[: int(args.max_videos)]
    if not selected_videos:
        raise SystemExit("no videos selected for feature caching")

    encode_batch = build_siglip_encoder(
        args.model_name,
        device=args.device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        local_files_only=not args.allow_network,
    )
    output_dtype = np.float16 if args.dtype == "float16" else np.float32
    manifest_videos = {}
    feature_dim = None
    for video_index, (video_name, video_info) in enumerate(selected_videos, start=1):
        total_frames = int(video_info["frame"])
        source_frames = recent_packet_source_frames(total_frames, args.feature_stride)
        output_path = output_dir / f"{video_name}.npy"
        record_path = output_dir / f"{video_name}.json"
        if output_path.exists() and args.resume:
            metadata = write_feature_file(
                output_path,
                None,
                source_frames,
                resume=True,
                expected_dtype=output_dtype,
            )
            validate_video_cache_record(
                record_path,
                metadata,
                encoder_id=args.model_name,
                feature_stride=args.feature_stride,
            )
        else:
            video_path = video_dir / f"{video_name}{args.video_suffix}"
            features = encode_selected_frames(
                iter_video_frames(video_path, args.image_size),
                source_frames,
                encode_batch=encode_batch,
                batch_size=args.batch_size,
            ).astype(output_dtype, copy=False)
            metadata = write_feature_file(output_path, features, source_frames, resume=False)
            write_video_cache_record(
                record_path,
                metadata,
                encoder_id=args.model_name,
                feature_stride=args.feature_stride,
            )
        feature_dim = int(metadata["feature_dim"]) if feature_dim is None else feature_dim
        if int(metadata["feature_dim"]) != feature_dim:
            raise RuntimeError("encoder produced inconsistent feature dimensions")
        metadata.pop("resumed", None)
        manifest_videos[video_name] = metadata
        print(f"[{video_index}/{len(selected_videos)}] cached {video_name}: {metadata['num_tokens']} tokens")

    manifest_path = output_dir / "manifest.json"
    write_cache_manifest(
        manifest_path,
        annotation_path=annotation_path,
        encoder_id=args.model_name,
        feature_stride=args.feature_stride,
        feature_dim=feature_dim,
        dtype=args.dtype,
        videos=manifest_videos,
    )
    print(f"CACHE_MANIFEST={manifest_path}")


if __name__ == "__main__":
    main()
