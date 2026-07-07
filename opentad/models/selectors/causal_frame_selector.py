from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from mmengine.registry import MODELS


@dataclass
class SelectedFrameBatch:
    frames: torch.Tensor
    batch_indices: torch.Tensor
    slot_indices: torch.Tensor
    selected_positions: torch.Tensor
    selected_masks: torch.Tensor
    dense_lengths: torch.Tensor


@MODELS.register_module()
class CausalFrameSelector(nn.Module):
    """Streaming-safe frame selector that runs before the heavy visual tower."""

    def __init__(
        self,
        policy="causal_stride",
        keep_ratio=1.0,
        stride=None,
        min_gap=1,
        max_gap=None,
        motion_threshold=0.08,
        max_tokens=None,
        always_first=True,
    ):
        super().__init__()
        if keep_ratio <= 0:
            raise ValueError("keep_ratio must be positive")
        if policy not in {"causal_stride", "causal_motion"}:
            raise ValueError("policy must be 'causal_stride' or 'causal_motion'")
        self.policy = policy
        self.keep_ratio = float(keep_ratio)
        self.stride = int(stride) if stride is not None else max(1, int(math.ceil(1.0 / self.keep_ratio)))
        self.min_gap = max(1, int(min_gap))
        self.max_gap = int(max_gap) if max_gap is not None else self.stride
        self.motion_threshold = float(motion_threshold)
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.always_first = bool(always_first)

    def _valid_indices(self, mask, seq_len, device):
        if mask is None:
            return torch.arange(seq_len, device=device, dtype=torch.long)
        valid = torch.nonzero(mask.bool(), as_tuple=False).flatten().to(device=device)
        if valid.numel() == 0:
            return torch.zeros(1, device=device, dtype=torch.long)
        return valid

    def _select_stride(self, valid):
        first = int(valid[0].item())
        selected = [first] if self.always_first else []
        for idx in valid.tolist():
            if selected and idx == selected[-1]:
                continue
            if (idx - first) % self.stride == 0:
                selected.append(int(idx))
            if self.max_tokens is not None and len(selected) >= self.max_tokens:
                break
        if not selected:
            selected.append(first)
        return selected

    def _select_motion(self, frames, valid):
        selected = [int(valid[0].item())]
        previous_idx = selected[0]
        previous_frame = frames[previous_idx]
        for idx in valid[1:].tolist():
            gap = int(idx) - previous_idx
            diff = (frames[int(idx)] - previous_frame).abs().mean().item()
            should_select = (gap >= self.max_gap) or (gap >= self.min_gap and diff >= self.motion_threshold)
            if should_select:
                selected.append(int(idx))
                previous_idx = int(idx)
                previous_frame = frames[int(idx)]
            if self.max_tokens is not None and len(selected) >= self.max_tokens:
                break
        return selected

    def select(self, frames, masks=None):
        if frames.dim() != 5:
            raise ValueError(f"CausalFrameSelector expects [B,T,C,H,W], got {tuple(frames.shape)}")
        batch_size, seq_len = frames.shape[:2]
        device = frames.device

        positions_per_sample = []
        dense_lengths = []
        for batch_idx in range(batch_size):
            mask = masks[batch_idx] if masks is not None else None
            valid = self._valid_indices(mask, seq_len, device)
            dense_lengths.append(int(valid.numel()))
            if self.policy == "causal_motion":
                selected = self._select_motion(frames[batch_idx], valid)
            else:
                selected = self._select_stride(valid)
            positions_per_sample.append(selected)

        max_selected = max(len(item) for item in positions_per_sample)
        selected_positions = torch.zeros(batch_size, max_selected, device=device, dtype=torch.long)
        selected_masks = torch.zeros(batch_size, max_selected, device=device, dtype=torch.bool)
        packed_frames = []
        batch_indices = []
        slot_indices = []
        for batch_idx, selected in enumerate(positions_per_sample):
            selected_positions[batch_idx, : len(selected)] = torch.tensor(selected, device=device, dtype=torch.long)
            selected_masks[batch_idx, : len(selected)] = True
            for slot_idx, frame_idx in enumerate(selected):
                packed_frames.append(frames[batch_idx, int(frame_idx)])
                batch_indices.append(batch_idx)
                slot_indices.append(slot_idx)

        selected_frames = torch.stack(packed_frames, dim=0)
        return SelectedFrameBatch(
            frames=selected_frames,
            batch_indices=torch.tensor(batch_indices, device=device, dtype=torch.long),
            slot_indices=torch.tensor(slot_indices, device=device, dtype=torch.long),
            selected_positions=selected_positions,
            selected_masks=selected_masks,
            dense_lengths=torch.tensor(dense_lengths, device=device, dtype=torch.long),
        )

    def forward(self, frames, masks=None):
        return self.select(frames, masks=masks)
