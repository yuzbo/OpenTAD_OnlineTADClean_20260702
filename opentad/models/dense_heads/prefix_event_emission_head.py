from dataclasses import dataclass, field
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActiveActionTrack:
    label: int
    start_frame: int
    last_frame: int
    peak_class_score: float


@dataclass(frozen=True)
class ImmutableEmissionRecord:
    stream_key: str
    label: int
    score: float
    start_frame: int
    end_frame: int
    emit_frame: int
    max_raw_frame_read: int
    max_cache_source_frame: int


@dataclass
class PrefixEmissionState:
    stream_key: str
    num_classes: int
    active_tracks: Dict[int, ActiveActionTrack] = field(default_factory=dict)
    ledger: List[ImmutableEmissionRecord] = field(default_factory=list)
    armed: List[bool] = field(default_factory=list)

    def __post_init__(self):
        if not self.armed:
            self.armed = [True for _ in range(self.num_classes)]


class PrefixEventEmissionHead(nn.Module):
    """Independent lifecycle hazards for prefix-censored streaming TAD."""

    def __init__(
        self,
        in_channels,
        num_classes,
        class_threshold=0.5,
        start_threshold=0.5,
        end_threshold=0.5,
        completion_threshold=0.5,
        emission_threshold=0.5,
        class_loss_weight=1.0,
        start_loss_weight=0.25,
        ongoing_loss_weight=0.25,
        end_loss_weight=1.0,
        completion_loss_weight=0.5,
        emission_loss_weight=1.0,
        delay_loss_weight=0.25,
        calibration_loss_weight=0.1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        if self.in_channels <= 0 or self.num_classes <= 0:
            raise ValueError("in_channels and num_classes must be positive")

        self.class_threshold = float(class_threshold)
        self.start_threshold = float(start_threshold)
        self.end_threshold = float(end_threshold)
        self.completion_threshold = float(completion_threshold)
        self.emission_threshold = float(emission_threshold)
        self.loss_weights = {
            "class_loss": float(class_loss_weight),
            "start_loss": float(start_loss_weight),
            "ongoing_loss": float(ongoing_loss_weight),
            "end_survival_loss": float(end_loss_weight),
            "completion_loss": float(completion_loss_weight),
            "emission_loss": float(emission_loss_weight),
            "delay_loss": float(delay_loss_weight),
            "calibration_loss": float(calibration_loss_weight),
        }

        self.class_head = nn.Conv1d(self.in_channels, self.num_classes, kernel_size=1)
        self.start_memory_head = nn.Conv1d(self.in_channels, self.num_classes, kernel_size=1)
        self.ongoing_head = nn.Conv1d(self.in_channels, self.num_classes, kernel_size=1)
        self.end_hazard_head = nn.Conv1d(self.in_channels, self.num_classes, kernel_size=1)
        self.completion_head = nn.Conv1d(self.in_channels, self.num_classes, kernel_size=1)
        self.emission_hazard_head = nn.Conv1d(self.in_channels, self.num_classes, kernel_size=1)

    def forward(self, features):
        if features.ndim == 2:
            features = features.unsqueeze(-1)
        if features.ndim != 3:
            raise ValueError(f"features must have shape [B,C] or [B,C,T], got {tuple(features.shape)}")
        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"feature channel mismatch: expected {self.in_channels}, got {features.shape[1]}"
            )
        newest = features[..., -1:]
        return {
            "class_logits": self.class_head(newest).squeeze(-1),
            "start_logits": self.start_memory_head(newest).squeeze(-1),
            "ongoing_logits": self.ongoing_head(newest).squeeze(-1),
            "end_hazard_logits": self.end_hazard_head(newest).squeeze(-1),
            "completion_logits": self.completion_head(newest).squeeze(-1),
            "emission_hazard_logits": self.emission_hazard_head(newest).squeeze(-1),
        }

    @staticmethod
    def _as_target(targets, key, reference):
        if key not in targets:
            raise KeyError(f"missing prefix target: {key}")
        value = targets[key]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.to(device=reference.device, dtype=reference.dtype)
        if value.shape != reference.shape:
            raise ValueError(
                f"target {key} shape {tuple(value.shape)} does not match logits {tuple(reference.shape)}"
            )
        return value

    @staticmethod
    def _masked_bce(logits, targets, mask):
        mask = mask.to(dtype=torch.bool)
        if not mask.any():
            return logits.float().sum() * 0.0
        return F.binary_cross_entropy_with_logits(logits[mask], targets[mask], reduction="mean")

    def losses(self, logits, targets):
        class_logits = logits["class_logits"]
        start_logits = logits["start_logits"]
        ongoing_logits = logits["ongoing_logits"]
        end_logits = logits["end_hazard_logits"]
        completion_logits = logits["completion_logits"]
        emit_logits = logits["emission_hazard_logits"]

        class_target = self._as_target(targets, "class_target", class_logits)
        start_target = self._as_target(targets, "start_target", start_logits)
        ongoing_target = self._as_target(targets, "ongoing_target", ongoing_logits)
        end_event = self._as_target(targets, "end_event", end_logits)
        censor_mask = self._as_target(targets, "censor_mask", end_logits)
        completion_target = self._as_target(targets, "completion_target", completion_logits)
        emit_allowed = self._as_target(targets, "emit_allowed", emit_logits)
        emit_forbidden = self._as_target(targets, "emit_forbidden", emit_logits)
        emit_event = self._as_target(targets, "emit_event", emit_logits)
        late_target = self._as_target(targets, "late_target", emit_logits)

        all_mask = torch.ones_like(class_target, dtype=torch.bool)
        end_mask = (end_event + censor_mask) > 0
        emit_mask = (emit_forbidden + emit_allowed + emit_event + late_target) > 0
        emit_target = torch.clamp(emit_allowed + emit_event + late_target, min=0.0, max=1.0)

        raw_losses = {
            "class_loss": self._masked_bce(class_logits, class_target, all_mask),
            "start_loss": self._masked_bce(start_logits, start_target, all_mask),
            "ongoing_loss": self._masked_bce(ongoing_logits, ongoing_target, all_mask),
            "end_survival_loss": self._masked_bce(end_logits, end_event, end_mask),
            "completion_loss": self._masked_bce(completion_logits, completion_target, all_mask),
            "emission_loss": self._masked_bce(emit_logits, emit_target, emit_mask),
        }

        late_mask = late_target > 0
        if late_mask.any():
            raw_losses["delay_loss"] = F.softplus(-emit_logits[late_mask]).mean()
        else:
            raw_losses["delay_loss"] = emit_logits.float().sum() * 0.0
        raw_losses["calibration_loss"] = F.mse_loss(
            completion_logits.sigmoid(),
            completion_target,
            reduction="mean",
        )

        weighted = {
            key: value * self.loss_weights[key]
            for key, value in raw_losses.items()
        }
        weighted["cost"] = sum(weighted.values())
        return weighted

    def initial_state(self, stream_key):
        return PrefixEmissionState(stream_key=str(stream_key), num_classes=self.num_classes)

    @staticmethod
    def _probability_vector(logits, key, num_classes):
        value = logits[key]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32)
        if value.ndim == 2:
            if value.shape[0] != 1:
                raise ValueError("decode_step supports one stream lane at a time")
            value = value[0]
        if value.ndim != 1 or value.numel() != num_classes:
            raise ValueError(f"{key} must contain {num_classes} class logits")
        return value.detach().float().sigmoid().cpu().tolist()

    def decode_step(self, logits, state, meta):
        if state.num_classes != self.num_classes:
            raise ValueError("state/head class-count mismatch")
        current_frame = int(meta["current_frame"])
        max_raw_frame_read = int(meta["max_raw_frame_read"])
        max_cache_source_frame = int(meta["max_cache_source_frame"])
        if max_raw_frame_read > current_frame:
            raise ValueError(
                f"future raw frame read: source={max_raw_frame_read} current={current_frame}"
            )
        if max_cache_source_frame > current_frame:
            raise ValueError(
                f"future cache source frame: source={max_cache_source_frame} current={current_frame}"
            )

        class_probs = self._probability_vector(logits, "class_logits", self.num_classes)
        start_probs = self._probability_vector(logits, "start_logits", self.num_classes)
        end_probs = self._probability_vector(logits, "end_hazard_logits", self.num_classes)
        completion_probs = self._probability_vector(logits, "completion_logits", self.num_classes)
        emit_probs = self._probability_vector(logits, "emission_hazard_logits", self.num_classes)

        emitted = []
        for label in range(self.num_classes):
            if not state.armed[label] and start_probs[label] < self.start_threshold:
                state.armed[label] = True

            track = state.active_tracks.get(label)
            if track is not None:
                track.last_frame = current_frame
                track.peak_class_score = max(track.peak_class_score, class_probs[label])
                should_emit = (
                    end_probs[label] >= self.end_threshold
                    and completion_probs[label] >= self.completion_threshold
                    and emit_probs[label] >= self.emission_threshold
                )
                if should_emit:
                    score = math.pow(
                        max(track.peak_class_score * end_probs[label] * emit_probs[label], 0.0),
                        1.0 / 3.0,
                    )
                    record = ImmutableEmissionRecord(
                        stream_key=state.stream_key,
                        label=label,
                        score=float(score),
                        start_frame=int(track.start_frame),
                        end_frame=current_frame,
                        emit_frame=current_frame,
                        max_raw_frame_read=max_raw_frame_read,
                        max_cache_source_frame=max_cache_source_frame,
                    )
                    state.ledger.append(record)
                    emitted.append(record)
                    del state.active_tracks[label]
                    state.armed[label] = False
                continue

            if (
                state.armed[label]
                and class_probs[label] >= self.class_threshold
                and start_probs[label] >= self.start_threshold
            ):
                state.active_tracks[label] = ActiveActionTrack(
                    label=label,
                    start_frame=current_frame,
                    last_frame=current_frame,
                    peak_class_score=class_probs[label],
                )

        return emitted, state
