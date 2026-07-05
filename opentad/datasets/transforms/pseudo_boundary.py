import hashlib
import json
import os

import numpy as np


def _stable_rng(sample_key):
    value = "unknown" if sample_key is None else str(sample_key)
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], byteorder="little", signed=False))


def _as_score_vector(scores):
    if scores is None:
        return None
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 0:
        return scores.reshape(1)
    if scores.ndim > 1:
        scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    return scores


def load_boundary_scores(cache_dir, video_name):
    """Load optional per-video boundary scores.

    The cache is deliberately optional: routes that do not use pseudo-boundary
    sampling should still import cleanly when no cache exists.
    """

    if not cache_dir:
        return None

    base = os.path.join(cache_dir, str(video_name))
    for path in (base + ".npy", base + ".npz", base + ".json"):
        if not os.path.exists(path):
            continue
        if path.endswith(".npy"):
            return _as_score_vector(np.load(path))
        if path.endswith(".npz"):
            data = np.load(path)
            key = "scores" if "scores" in data.files else data.files[0]
            return _as_score_vector(data[key])
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("scores", payload.get("boundary_scores"))
        return _as_score_vector(payload)
    return None


def slice_global_scores_for_window(boundary_scores, global_indices):
    scores = _as_score_vector(boundary_scores)
    if scores is None:
        return None
    global_indices = np.asarray(global_indices, dtype=np.int64)
    if global_indices.size == 0:
        return np.zeros((0,), dtype=np.float32)
    valid = np.logical_and(global_indices >= 0, global_indices < scores.shape[0])
    window_scores = np.zeros(global_indices.shape, dtype=np.float32)
    window_scores[valid] = scores[global_indices[valid]]
    return window_scores


def _regular_positions(valid_len, target_frame_num):
    if valid_len <= 0 or target_frame_num <= 0:
        return np.zeros((0,), dtype=np.int64)
    if target_frame_num >= valid_len:
        return np.arange(valid_len, dtype=np.int64)
    return np.rint(np.linspace(0, valid_len - 1, target_frame_num)).astype(np.int64)


def _random_fill(valid_len, target_frame_num, sample_key, selected):
    selected = set(int(pos) for pos in selected if 0 <= int(pos) < valid_len)
    if len(selected) >= target_frame_num:
        return np.array(sorted(selected), dtype=np.int64)[:target_frame_num]

    remaining = np.array([idx for idx in range(valid_len) if idx not in selected], dtype=np.int64)
    if remaining.size > 0:
        need = min(target_frame_num - len(selected), remaining.size)
        selected.update(_stable_rng(sample_key).choice(remaining, size=need, replace=False).tolist())
    return np.array(sorted(selected), dtype=np.int64)


def _apply_group_size(positions, group_size, valid_len):
    group_size = max(int(group_size or 1), 1)
    if group_size == 1 or len(positions) == 0:
        return np.asarray(positions, dtype=np.int64)
    expanded = []
    half = group_size // 2
    for pos in positions:
        start = max(int(pos) - half, 0)
        end = min(start + group_size, valid_len)
        expanded.extend(range(start, end))
    return np.unique(np.asarray(expanded, dtype=np.int64))


def _fallback_positions(valid_len, target_frame_num, sample_key, fallback, group_size):
    if fallback == "regular":
        return _regular_positions(valid_len, target_frame_num)
    if fallback in {"random", "random_fixed", "stratified_random_fixed"}:
        return _random_fill(valid_len, target_frame_num, sample_key, [])
    return _regular_positions(valid_len, target_frame_num)


def select_pseudo_boundary_hybrid_positions(
    valid_len,
    target_frame_num,
    sample_key,
    boundary_scores,
    pseudo_quota=64,
    pseudo_radius=1,
    pseudo_min_score=0.0,
    fallback="random_fixed",
    group_size=1,
):
    if valid_len <= 0 or target_frame_num <= 0:
        return np.zeros((0,), dtype=np.int64)

    scores = _as_score_vector(boundary_scores)
    if scores is None or scores.size == 0:
        return _fallback_positions(valid_len, target_frame_num, sample_key, fallback, group_size)

    scores = scores[:valid_len]
    candidates = np.flatnonzero(scores >= float(pseudo_min_score))
    if candidates.size == 0:
        return _fallback_positions(valid_len, target_frame_num, sample_key, fallback, group_size)

    quota = min(int(pseudo_quota), target_frame_num, candidates.size)
    ranked = candidates[np.argsort(scores[candidates])[-quota:]]
    selected = []
    radius = max(int(pseudo_radius), 0)
    for pos in ranked:
        selected.extend(range(max(int(pos) - radius, 0), min(int(pos) + radius + 1, valid_len)))
    selected = _apply_group_size(selected, group_size, valid_len)
    selected = _random_fill(valid_len, target_frame_num, sample_key, selected)
    if selected.size < target_frame_num:
        selected = np.unique(np.concatenate([selected, _regular_positions(valid_len, target_frame_num)]))
    return selected[: min(target_frame_num, selected.size)]


def select_pseudo_boundary_snap_positions(
    valid_len,
    target_frame_num,
    sample_key,
    boundary_scores,
    pseudo_quota=64,
    pseudo_snap_distance=2,
    pseudo_min_score=0.0,
    fallback="random_fixed",
    group_size=1,
):
    if valid_len <= 0 or target_frame_num <= 0:
        return np.zeros((0,), dtype=np.int64)

    base = _fallback_positions(valid_len, target_frame_num, sample_key, fallback, group_size)
    scores = _as_score_vector(boundary_scores)
    if scores is None or scores.size == 0:
        return base

    scores = scores[:valid_len]
    candidates = np.flatnonzero(scores >= float(pseudo_min_score))
    if candidates.size == 0:
        return base

    quota = min(int(pseudo_quota), candidates.size)
    candidates = candidates[np.argsort(scores[candidates])[-quota:]]
    snap_distance = max(int(pseudo_snap_distance), 0)
    snapped = []
    for pos in base:
        close = candidates[np.abs(candidates - int(pos)) <= snap_distance]
        if close.size == 0:
            snapped.append(int(pos))
        else:
            best = close[np.argmax(scores[close])]
            snapped.append(int(best))
    snapped = _apply_group_size(snapped, group_size, valid_len)
    snapped = _random_fill(valid_len, target_frame_num, sample_key, snapped)
    return snapped[: min(target_frame_num, snapped.size)]
