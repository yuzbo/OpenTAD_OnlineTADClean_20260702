from typing import Tuple

import torch
from torch import Tensor, nn


class TimeAlignedRasterizer(nn.Module):
    """Map irregular temporal Adapter features through a uniform work axis.

    The Adapter calls this module around its temporal convolution. The branch is
    residual-gated by default, so restoring this missing module fixes clean-clone
    imports without changing baseline behavior unless a config explicitly opens
    the branch scale.
    """

    def __init__(
        self,
        channels: int,
        mode: str = "content_adaptive",
        residual: bool = True,
        residual_init: float = 0.0,
        center_feature_index: int = 0,
        valid_feature_index: int = 1,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.mode = mode
        self.use_branch_residual = bool(residual)
        self.center_feature_index = int(center_feature_index)
        self.valid_feature_index = int(valid_feature_index)
        self.eps = float(eps)
        self.extra_cfg = dict(kwargs)
        self.branch_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def _fallback_centers(self, batch_size: int, steps: int, dtype, device) -> Tensor:
        if steps <= 1:
            centers = torch.zeros(steps, dtype=dtype, device=device)
        else:
            centers = torch.linspace(0.0, 1.0, steps, dtype=dtype, device=device)
        return centers[None, :].expand(batch_size, steps)

    def _time_centers_and_valid(self, time_features: Tensor) -> Tuple[Tensor, Tensor]:
        if time_features.dim() != 3:
            raise ValueError(f"time_features must have shape [B, T, D], got {tuple(time_features.shape)}")

        batch_size, steps, feat_dim = time_features.shape
        fallback = self._fallback_centers(batch_size, steps, time_features.dtype, time_features.device)
        if feat_dim <= self.center_feature_index:
            centers = fallback
        else:
            centers = time_features[..., self.center_feature_index]
            centers = torch.where(torch.isfinite(centers), centers, fallback)

        if feat_dim <= self.valid_feature_index:
            valid = torch.ones(batch_size, steps, dtype=torch.bool, device=time_features.device)
        else:
            feature_nonzero = time_features.abs().sum(dim=-1) > self.eps
            valid = (time_features[..., self.valid_feature_index] > 0.5) | feature_nonzero
            valid = valid & torch.isfinite(centers)

        return centers, valid

    def _uniform_centers(self, source_center: Tensor, source_valid: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, steps = source_center.shape
        fallback = self._fallback_centers(batch_size, steps, source_center.dtype, source_center.device)
        centers = []
        valid_rows = []
        for batch_idx in range(batch_size):
            row_valid = source_valid[batch_idx]
            if int(row_valid.sum().item()) >= 2:
                row_center = source_center[batch_idx, row_valid]
                lo = row_center.min()
                hi = row_center.max()
                if torch.isfinite(lo) and torch.isfinite(hi) and bool((hi - lo).abs() > self.eps):
                    centers.append(torch.linspace(lo, hi, steps, dtype=source_center.dtype, device=source_center.device))
                    valid_rows.append(torch.ones(steps, dtype=torch.bool, device=source_center.device))
                    continue
            centers.append(fallback[batch_idx])
            valid_rows.append(torch.ones(steps, dtype=torch.bool, device=source_center.device))
        return torch.stack(centers, dim=0), torch.stack(valid_rows, dim=0)

    def _resample(self, values: Tensor, src_center: Tensor, dst_center: Tensor, src_valid: Tensor) -> Tensor:
        if values.dim() != 4:
            raise ValueError(f"values must have shape [B, S, C, T], got {tuple(values.shape)}")
        batch_size, spatial_tokens, channels, src_steps = values.shape
        if channels != self.channels:
            raise ValueError(f"values channel count ({channels}) does not match rasterizer channels ({self.channels}).")
        if src_center.shape != (batch_size, src_steps):
            raise ValueError(f"src_center must have shape {(batch_size, src_steps)}, got {tuple(src_center.shape)}.")
        if src_valid.shape != (batch_size, src_steps):
            raise ValueError(f"src_valid must have shape {(batch_size, src_steps)}, got {tuple(src_valid.shape)}.")

        dst_steps = dst_center.shape[1]
        output = values.new_zeros(batch_size, spatial_tokens, channels, dst_steps)
        for batch_idx in range(batch_size):
            valid = src_valid[batch_idx] & torch.isfinite(src_center[batch_idx])
            if int(valid.sum().item()) < 2:
                if dst_steps == src_steps:
                    output[batch_idx] = values[batch_idx]
                else:
                    flat = values[batch_idx].reshape(spatial_tokens * channels, 1, src_steps)
                    output[batch_idx] = torch.nn.functional.interpolate(
                        flat,
                        size=dst_steps,
                        mode="linear",
                        align_corners=True,
                    ).reshape(spatial_tokens, channels, dst_steps)
                continue

            valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
            centers = src_center[batch_idx, valid_indices]
            order = torch.argsort(centers)
            centers = centers[order]
            samples = values[batch_idx].index_select(-1, valid_indices.index_select(0, order))

            dst = dst_center[batch_idx].to(dtype=centers.dtype).clamp(min=centers[0], max=centers[-1])
            upper = torch.searchsorted(centers.contiguous(), dst.contiguous(), right=False)
            upper = upper.clamp(min=1, max=centers.numel() - 1)
            lower = upper - 1
            lower_center = centers.index_select(0, lower)
            upper_center = centers.index_select(0, upper)
            denom = (upper_center - lower_center).clamp_min(self.eps)
            weight = ((dst - lower_center) / denom).to(dtype=values.dtype)

            lower_values = samples.index_select(-1, lower)
            upper_values = samples.index_select(-1, upper)
            output[batch_idx] = lower_values + (upper_values - lower_values) * weight.view(1, 1, dst_steps)
        return output

    def source_to_uniform(self, source: Tensor, time_features: Tensor):
        if self.mode in {"identity", "none", None}:
            source_center, source_valid = self._time_centers_and_valid(time_features)
            uniform_center = source_center
            uniform_valid = source_valid
            return source, uniform_center, uniform_valid, source_center, source_valid

        source_center, source_valid = self._time_centers_and_valid(time_features)
        uniform_center, uniform_valid = self._uniform_centers(source_center, source_valid)
        uniform = self._resample(source, source_center, uniform_center, source_valid)
        return uniform, uniform_center, uniform_valid, source_center, source_valid

    def uniform_to_source(
        self,
        uniform: Tensor,
        uniform_center: Tensor,
        uniform_valid: Tensor,
        source_center: Tensor,
        source_valid: Tensor,
    ) -> Tensor:
        source = self._resample(uniform, uniform_center, source_center, uniform_valid)
        return source * source_valid[:, None, None, :].to(dtype=source.dtype)

    def mix_with_source(self, source: Tensor, target: Tensor) -> Tensor:
        if not self.use_branch_residual:
            return target
        return source + self.branch_scale * (target - source)
