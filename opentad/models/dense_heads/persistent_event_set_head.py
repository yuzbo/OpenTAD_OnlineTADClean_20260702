from dataclasses import dataclass, field
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ..builder import HEADS


SLOT_FREE = 0
SLOT_ACTIVE = 1
SLOT_REFRACTORY = 2


@dataclass(frozen=True)
class EventSetEmissionRecord:
    stream_key: str
    slot_id: int
    label: int
    score: float
    start_frame: int
    end_frame: int
    emit_frame: int
    max_source_frame: int


@dataclass
class PersistentEventSetState:
    stream_key: str
    queries: torch.Tensor
    feature_memory: torch.Tensor
    memory_frames: Tuple[int, ...]
    slot_status: torch.Tensor
    refractory: torch.Tensor
    start_frames: torch.Tensor
    peak_class_scores: torch.Tensor
    peak_class_labels: torch.Tensor
    ledger: List[EventSetEmissionRecord] = field(default_factory=list)
    instance_to_slot: Dict[int, int] = field(default_factory=dict)
    slot_to_instance: Dict[int, int] = field(default_factory=dict)


@HEADS.register_module()
class PersistentEventSetHead(nn.Module):
    """Shared causal query head for the matched Stage-1 mechanism test."""

    def __init__(
        self,
        in_channels,
        hidden_dim,
        num_classes,
        num_slots,
        memory_size=32,
        num_heads=4,
        dropout=0.0,
        query_mode="persistent",
        start_mode="pointer",
        endpoint_mode="hazard",
        birth_threshold=0.5,
        alive_threshold=0.5,
        end_threshold=0.5,
        refractory_steps=2,
        max_endpoint_offset=8.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.num_slots = int(num_slots)
        self.memory_size = int(memory_size)
        self.query_mode = str(query_mode)
        self.start_mode = str(start_mode)
        self.endpoint_mode = str(endpoint_mode)
        self.birth_threshold = float(birth_threshold)
        self.alive_threshold = float(alive_threshold)
        self.end_threshold = float(end_threshold)
        self.refractory_steps = int(refractory_steps)
        self.max_endpoint_offset = float(max_endpoint_offset)
        if self.query_mode not in {"fresh", "persistent"}:
            raise ValueError("query_mode must be 'fresh' or 'persistent'")
        if self.start_mode not in {"scalar", "pointer"}:
            raise ValueError("start_mode must be 'scalar' or 'pointer'")
        if self.endpoint_mode not in {"binary", "hazard"}:
            raise ValueError("endpoint_mode must be 'binary' or 'hazard'")
        if min(self.in_channels, self.hidden_dim, self.num_classes, self.num_slots, self.memory_size) <= 0:
            raise ValueError("head dimensions and memory_size must be positive")
        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.refractory_steps < 0:
            raise ValueError("refractory_steps must be non-negative")

        self.input_proj = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.query_embed = nn.Embedding(self.num_slots, self.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_dim,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.slot_attention = nn.MultiheadAttention(
            self.hidden_dim,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.update_cell = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.before_memory = nn.Parameter(torch.empty(self.hidden_dim))

        self.birth_head = nn.Linear(self.hidden_dim, 1)
        self.alive_head = nn.Linear(self.hidden_dim, 1)
        self.class_head = nn.Linear(self.hidden_dim, self.num_classes)
        self.end_head = nn.Linear(self.hidden_dim, 1)
        self.endpoint_offset_head = (
            nn.Linear(self.hidden_dim, 1) if self.endpoint_mode == "hazard" else None
        )
        self.start_offset_head = nn.Linear(self.hidden_dim, 1)
        nn.init.normal_(self.before_memory, std=0.02)

    def initial_state(self, device, dtype, stream_key):
        queries = self.query_embed.weight.to(device=device, dtype=dtype).unsqueeze(0)
        return PersistentEventSetState(
            stream_key=str(stream_key),
            queries=queries,
            feature_memory=queries.new_zeros((1, 0, self.hidden_dim)),
            memory_frames=(),
            slot_status=torch.full((self.num_slots,), SLOT_FREE, dtype=torch.long, device=device),
            refractory=torch.zeros(self.num_slots, dtype=torch.long, device=device),
            start_frames=torch.full((self.num_slots,), float("nan"), dtype=dtype, device=device),
            peak_class_scores=torch.zeros(self.num_slots, dtype=dtype, device=device),
            peak_class_labels=torch.full(
                (self.num_slots,), -1, dtype=torch.long, device=device
            ),
        )

    def _base_queries(self, state, batch_size, device, dtype):
        if self.query_mode == "fresh":
            return self.query_embed.weight.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        return state.queries.to(device=device, dtype=dtype)

    def step(self, feature, state, source_frame):
        if feature.ndim != 2 or feature.shape[1] != self.in_channels:
            raise ValueError(f"feature must have shape [B,{self.in_channels}]")
        if feature.shape[0] != 1:
            raise ValueError("Stage-1 event-set head currently supports one stream lane")
        projected = self.input_proj(feature).unsqueeze(1)
        memory = torch.cat([state.feature_memory.to(projected), projected], dim=1)
        memory = memory[:, -self.memory_size :]
        memory_frames = (tuple(state.memory_frames) + (int(source_frame),))[-self.memory_size :]

        base = self._base_queries(state, feature.shape[0], feature.device, projected.dtype)
        cross, _ = self.cross_attention(base, memory, memory, need_weights=False)
        slots, _ = self.slot_attention(base, base, base, need_weights=False)
        proposal = self.query_norm(base + cross + slots)
        queries = self.update_cell(
            proposal.reshape(-1, self.hidden_dim),
            base.reshape(-1, self.hidden_dim),
        ).reshape_as(base)
        queries = self.query_norm(queries)

        pointer_scores = torch.einsum("bkd,bmd->bkm", queries, memory) / math.sqrt(self.hidden_dim)
        sentinel = torch.einsum("bkd,d->bk", queries, self.before_memory.to(queries)).unsqueeze(-1)
        endpoint_offset = (
            self.endpoint_offset_head(queries).sigmoid().squeeze(-1)
            * self.max_endpoint_offset
            if self.endpoint_offset_head is not None
            else queries.new_zeros((queries.shape[0], self.num_slots))
        )
        outputs = {
            "birth_logits": self.birth_head(queries).squeeze(-1),
            "alive_logits": self.alive_head(queries).squeeze(-1),
            "class_logits": self.class_head(queries),
            "end_hazard_logits": self.end_head(queries).squeeze(-1),
            "endpoint_offset": endpoint_offset,
            "start_offset": self.start_offset_head(queries).sigmoid().squeeze(-1) * self.memory_size,
            "start_pointer_logits": torch.cat([sentinel, pointer_scores], dim=-1),
            "memory_frames": memory_frames,
        }
        next_queries = queries if self.query_mode == "persistent" else state.queries.to(queries)
        next_state = PersistentEventSetState(
            stream_key=state.stream_key,
            queries=next_queries,
            feature_memory=memory,
            memory_frames=memory_frames,
            slot_status=state.slot_status,
            refractory=state.refractory,
            start_frames=state.start_frames,
            peak_class_scores=state.peak_class_scores,
            peak_class_labels=state.peak_class_labels,
            ledger=state.ledger,
            instance_to_slot=state.instance_to_slot,
            slot_to_instance=state.slot_to_instance,
        )
        return outputs, next_state

    @staticmethod
    def pointer_target(memory_frames, start_frame):
        if not memory_frames or float(start_frame) < float(memory_frames[0]):
            return 0
        distances = [abs(float(frame) - float(start_frame)) for frame in memory_frames]
        return int(min(range(len(distances)), key=distances.__getitem__)) + 1

    def build_end_hazard_targets(self, reference, at_risk_slots, end_event_slots):
        target = torch.zeros_like(reference)
        mask = torch.zeros_like(reference, dtype=torch.bool)
        for slot in at_risk_slots:
            mask[:, int(slot)] = True
        for slot in end_event_slots:
            slot = int(slot)
            if not mask[:, slot].all():
                raise ValueError("end-event slot must be in the current instance risk set")
            target[:, slot] = 1.0
        return target, mask

    def _decode_start(self, outputs, slot, current_frame, feature_stride=1):
        if self.start_mode == "pointer":
            pointer_index = int(outputs["start_pointer_logits"][0, slot].argmax().item())
            memory_frames = tuple(int(frame) for frame in outputs["memory_frames"])
            if not memory_frames:
                return int(current_frame)
            if pointer_index == 0:
                return int(memory_frames[0])
            return int(memory_frames[min(pointer_index - 1, len(memory_frames) - 1)])
        offset = float(outputs["start_offset"][0, slot].detach().item())
        return max(0, int(round(float(current_frame) - offset * float(feature_stride))))

    def decode_step(self, outputs, state, current_frame, feature_stride=1):
        current_frame = int(current_frame)
        birth = outputs["birth_logits"].detach().sigmoid()[0]
        alive = outputs["alive_logits"].detach().sigmoid()[0]
        end = outputs["end_hazard_logits"].detach().sigmoid()[0]
        class_probs = outputs["class_logits"].detach().softmax(dim=-1)[0]
        class_scores, labels = class_probs.max(dim=-1)
        offsets = (
            outputs["endpoint_offset"].detach()[0]
            if self.endpoint_mode == "hazard"
            else None
        )

        status = state.slot_status.clone()
        refractory = state.refractory.clone()
        start_frames = state.start_frames.clone()
        peak_scores = state.peak_class_scores.clone()
        peak_labels = state.peak_class_labels.clone()
        ledger = list(state.ledger)
        emitted = []

        for slot in range(self.num_slots):
            just_born = False
            if int(status[slot].item()) == SLOT_REFRACTORY:
                if int(refractory[slot].item()) > 0:
                    refractory[slot] -= 1
                    continue
                if alive[slot] < self.alive_threshold and birth[slot] < self.birth_threshold:
                    status[slot] = SLOT_FREE
                continue

            if int(status[slot].item()) == SLOT_FREE:
                if birth[slot] < self.birth_threshold:
                    continue
                status[slot] = SLOT_ACTIVE
                start_frames[slot] = float(
                    self._decode_start(outputs, slot, current_frame, feature_stride)
                )
                peak_scores[slot] = class_scores[slot]
                peak_labels[slot] = labels[slot]
                just_born = True

            if (
                not just_born
                and alive[slot] < self.alive_threshold
                and end[slot] < self.end_threshold
            ):
                status[slot] = SLOT_FREE
                start_frames[slot] = float("nan")
                peak_scores[slot] = 0.0
                peak_labels[slot] = -1
                continue

            if class_scores[slot] > peak_scores[slot]:
                peak_scores[slot] = class_scores[slot]
                peak_labels[slot] = labels[slot]
            if end[slot] < self.end_threshold:
                continue

            start_frame = int(round(float(start_frames[slot].item())))
            if self.endpoint_mode == "binary":
                end_frame = current_frame
            else:
                end_frame = int(round(current_frame - float(offsets[slot].item())))
            end_frame = min(current_frame, max(start_frame, end_frame))
            score = math.sqrt(max(float(peak_scores[slot].item()) * float(end[slot].item()), 0.0))
            record = EventSetEmissionRecord(
                stream_key=state.stream_key,
                slot_id=slot,
                label=int(peak_labels[slot].item()),
                score=score,
                start_frame=start_frame,
                end_frame=end_frame,
                emit_frame=current_frame,
                max_source_frame=max(int(frame) for frame in outputs["memory_frames"]),
            )
            ledger.append(record)
            emitted.append(record)
            status[slot] = SLOT_REFRACTORY
            refractory[slot] = self.refractory_steps
            start_frames[slot] = float("nan")
            peak_scores[slot] = 0.0
            peak_labels[slot] = -1

        next_state = PersistentEventSetState(
            stream_key=state.stream_key,
            queries=state.queries,
            feature_memory=state.feature_memory,
            memory_frames=state.memory_frames,
            slot_status=status,
            refractory=refractory,
            start_frames=start_frames,
            peak_class_scores=peak_scores,
            peak_class_labels=peak_labels,
            ledger=ledger,
            instance_to_slot=state.instance_to_slot,
            slot_to_instance=state.slot_to_instance,
        )
        return emitted, next_state


__all__ = [
    "SLOT_FREE",
    "SLOT_ACTIVE",
    "SLOT_REFRACTORY",
    "EventSetEmissionRecord",
    "PersistentEventSetState",
    "PersistentEventSetHead",
]
