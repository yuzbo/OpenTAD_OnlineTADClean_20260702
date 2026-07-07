import bisect
import math


def _is_tensor(value):
    return hasattr(value, "dim") and hasattr(value, "new_tensor")


def _validate_token_times(token_times_sec):
    times = [float(item) for item in token_times_sec]
    if len(times) == 0:
        raise ValueError("token_times_sec must not be empty")
    for value in times:
        if not math.isfinite(value):
            raise ValueError("token_times_sec must contain finite values")
    for left, right in zip(times, times[1:]):
        if right < left:
            raise ValueError("token_times_sec must be monotonic non-decreasing")
    return times


def _decode_scalar(coord, times, duration=None):
    right_time = float(duration) if duration is not None else times[-1]
    xp = list(range(len(times))) + [len(times)]
    fp = times + [right_time]
    coord = max(0.0, min(float(coord), float(xp[-1])))
    right_idx = bisect.bisect_right(xp, coord)
    right_idx = min(max(right_idx, 1), len(xp) - 1)
    left_idx = right_idx - 1
    x0, x1 = float(xp[left_idx]), float(xp[right_idx])
    y0, y1 = float(fp[left_idx]), float(fp[right_idx])
    if x1 <= x0:
        return y0
    weight = (coord - x0) / (x1 - x0)
    return y0 + weight * (y1 - y0)


def decode_irregular_segments_to_seconds(segments, token_times_sec, duration=None):
    times = _validate_token_times(token_times_sec)
    if _is_tensor(segments):
        import torch

        xp = torch.arange(len(times) + 1, dtype=segments.dtype, device=segments.device)
        end_time = float(duration) if duration is not None else times[-1]
        fp = segments.new_tensor(times + [end_time])
        coord_shape = segments.shape
        flat = segments.reshape(-1).clamp(min=0.0, max=float(len(times)))
        right_idx = torch.searchsorted(xp, flat, right=True).clamp(min=1, max=xp.numel() - 1)
        left_idx = right_idx - 1
        x0 = xp[left_idx]
        x1 = xp[right_idx]
        y0 = fp[left_idx]
        y1 = fp[right_idx]
        weight = (flat - x0) / (x1 - x0).clamp(min=1e-6)
        return (y0 + weight * (y1 - y0)).reshape(coord_shape)

    return [[_decode_scalar(point, times, duration=duration) for point in segment] for segment in segments]


def encode_seconds_to_irregular_grid(seconds, token_times_sec, duration=None):
    times = _validate_token_times(token_times_sec)
    right_time = float(duration) if duration is not None else times[-1]
    xp = times + [right_time]
    fp = list(range(len(times))) + [len(times)]

    def encode_scalar(value):
        value = max(float(xp[0]), min(float(value), float(xp[-1])))
        right_idx = bisect.bisect_right(xp, value)
        right_idx = min(max(right_idx, 1), len(xp) - 1)
        left_idx = right_idx - 1
        x0, x1 = float(xp[left_idx]), float(xp[right_idx])
        y0, y1 = float(fp[left_idx]), float(fp[right_idx])
        if x1 <= x0:
            return y0
        return y0 + (value - x0) * (y1 - y0) / (x1 - x0)

    if _is_tensor(seconds):
        import torch

        xp_tensor = seconds.new_tensor(xp)
        fp_tensor = seconds.new_tensor(fp)
        shape = seconds.shape
        flat = seconds.reshape(-1).clamp(min=float(xp[0]), max=float(xp[-1]))
        right_idx = torch.searchsorted(xp_tensor, flat, right=True).clamp(min=1, max=xp_tensor.numel() - 1)
        left_idx = right_idx - 1
        x0 = xp_tensor[left_idx]
        x1 = xp_tensor[right_idx]
        y0 = fp_tensor[left_idx]
        y1 = fp_tensor[right_idx]
        weight = (flat - x0) / (x1 - x0).clamp(min=1e-6)
        return (y0 + weight * (y1 - y0)).reshape(shape)

    return [[encode_scalar(point) for point in segment] for segment in seconds]
