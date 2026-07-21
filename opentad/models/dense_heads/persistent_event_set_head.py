from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..builder import HEADS


SLOT_FREE = 0
SLOT_ACTIVE = 1
SLOT_REFRACTORY = 2
SLOT_CANDIDATE = 3


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
    ledger: List[EventSetEmissionRecord] = field(default_factory=list)
    instance_to_slot: Dict[int, int] = field(default_factory=dict)
    slot_to_instance: Dict[int, int] = field(default_factory=dict)
    candidate_age: Optional[torch.Tensor] = None
    peak_class_labels: Optional[torch.Tensor] = None
    birth_proposals: int = 0
    birth_admissions: int = 0
    arbitration_suppressions: int = 0
    candidate_cancellations: int = 0
    active_abandonments: int = 0
    deferred_birth_due_to_release: int = 0


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
        max_start_offset=None,
        lifecycle_mode="legacy",
        candidate_confirmation_steps=1,
        max_births_per_step=2,
        birth_prior_probability=None,
        alive_prior_probability=None,
        end_prior_probability=None,
        lifecycle_calibration_mode="none",
        end_transition_mode="current_query",
        endpoint_start_mode="none",
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
        self.max_start_offset = float(
            self.memory_size
            if max_start_offset is None
            else max_start_offset
        )
        self.lifecycle_mode = str(lifecycle_mode)
        self.candidate_confirmation_steps = int(candidate_confirmation_steps)
        self.max_births_per_step = int(max_births_per_step)
        self.lifecycle_calibration_mode = str(lifecycle_calibration_mode)
        self.end_transition_mode = str(end_transition_mode)
        self.endpoint_start_mode = str(endpoint_start_mode)
        self.birth_prior_probability = self._validated_prior(
            birth_prior_probability,
            "birth_prior_probability",
        )
        self.alive_prior_probability = self._validated_prior(
            alive_prior_probability,
            "alive_prior_probability",
        )
        self.end_prior_probability = self._validated_prior(
            end_prior_probability,
            "end_prior_probability",
        )
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
        if not math.isfinite(self.max_start_offset) or self.max_start_offset <= 0:
            raise ValueError("max_start_offset must be positive and finite")
        if self.lifecycle_mode not in {"legacy", "candidate_recycle"}:
            raise ValueError("lifecycle_mode must be 'legacy' or 'candidate_recycle'")
        if self.lifecycle_mode == "candidate_recycle" and self.refractory_steps != 0:
            raise ValueError("candidate_recycle does not permit refractory occupancy")
        if self.candidate_confirmation_steps != 1:
            raise ValueError("the scientific route freezes one-step candidate confirmation")
        if not 1 <= self.max_births_per_step <= self.num_slots:
            raise ValueError("max_births_per_step must be in [1, num_slots]")
        if self.lifecycle_calibration_mode not in {
            "none",
            "monotone_affine",
        }:
            raise ValueError(
                "lifecycle_calibration_mode must be none or monotone_affine"
            )
        if self.end_transition_mode not in {
            "current_query",
            "causal_delta_mlp",
        }:
            raise ValueError(
                "end_transition_mode must be current_query or causal_delta_mlp"
            )
        if (
            self.end_transition_mode == "causal_delta_mlp"
            and self.query_mode != "persistent"
        ):
            raise ValueError(
                "causal_delta_mlp requires persistent query recurrence"
            )
        if self.endpoint_start_mode not in {"none", "past_pointer"}:
            raise ValueError(
                "endpoint_start_mode must be none or past_pointer"
            )

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
        if self.end_transition_mode == "causal_delta_mlp":
            self.end_transition_proj = nn.Sequential(
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
            )
        else:
            self.end_transition_proj = None
        if self.start_mode == "pointer":
            self.before_memory = nn.Parameter(torch.empty(self.hidden_dim))
        else:
            self.register_parameter("before_memory", None)

        self.birth_head = nn.Linear(self.hidden_dim, 1)
        self.alive_head = nn.Linear(self.hidden_dim, 1)
        self.class_head = nn.Linear(self.hidden_dim, self.num_classes)
        self.end_head = nn.Linear(self.hidden_dim, 1)
        if self.endpoint_start_mode == "past_pointer":
            self.endpoint_start_query = nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            )
            self.endpoint_before_memory = nn.Parameter(
                torch.empty(self.hidden_dim)
            )
            nn.init.normal_(self.endpoint_before_memory, std=0.02)
        else:
            self.endpoint_start_query = None
            self.register_parameter("endpoint_before_memory", None)
        if self.lifecycle_calibration_mode == "monotone_affine":
            self.lifecycle_calibration_log_scale = nn.Parameter(
                torch.zeros(3)
            )
            self.lifecycle_calibration_bias = nn.Parameter(torch.zeros(3))
        else:
            self.register_parameter(
                "lifecycle_calibration_log_scale",
                None,
            )
            self.register_parameter("lifecycle_calibration_bias", None)
        if self.endpoint_mode == "hazard":
            self.endpoint_offset_head = nn.Linear(self.hidden_dim, 1)
        self.start_offset_head = nn.Linear(self.hidden_dim, 1)
        if self.before_memory is not None:
            nn.init.normal_(self.before_memory, std=0.02)
        self._initialize_prior_bias(
            self.birth_head,
            self.birth_prior_probability,
        )
        self._initialize_prior_bias(
            self.alive_head,
            self.alive_prior_probability,
        )
        self._initialize_prior_bias(
            self.end_head,
            self.end_prior_probability,
        )

    @staticmethod
    def _validated_prior(value, name):
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1)")
        return value

    @staticmethod
    def prior_bias_logit(probability, positive_weight=1.0):
        if probability is None:
            return None
        probability = float(probability)
        positive_weight = float(positive_weight)
        if not 0.0 < probability < 1.0:
            raise ValueError("prior probability must be in (0, 1)")
        if not math.isfinite(positive_weight) or positive_weight <= 0:
            raise ValueError("prior positive weight must be positive and finite")
        return (
            math.log(probability / (1.0 - probability))
            + math.log(positive_weight)
        )

    @classmethod
    def _initialize_prior_bias(
        cls,
        layer,
        probability,
        positive_weight=1.0,
    ):
        logit = cls.prior_bias_logit(probability, positive_weight)
        if logit is None:
            return
        nn.init.constant_(layer.bias, logit)

    def initialize_prior_biases(
        self,
        *,
        birth_positive_weight=1.0,
        alive_positive_weight=1.0,
        end_positive_weight=1.0,
    ):
        settings = (
            (
                self.birth_head,
                self.birth_prior_probability,
                birth_positive_weight,
            ),
            (
                self.alive_head,
                self.alive_prior_probability,
                alive_positive_weight,
            ),
            (
                self.end_head,
                self.end_prior_probability,
                end_positive_weight,
            ),
        )
        if any(probability is None for _, probability, _ in settings):
            raise ValueError(
                "weighted prior initialization requires all binary priors"
            )
        logits = tuple(
            self.prior_bias_logit(probability, positive_weight)
            for _, probability, positive_weight in settings
        )
        for (layer, _, _), logit in zip(settings, logits):
            nn.init.constant_(layer.bias, logit)

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
            candidate_age=torch.zeros(self.num_slots, dtype=torch.long, device=device),
            peak_class_labels=torch.full(
                (self.num_slots,),
                -1,
                dtype=torch.long,
                device=device,
            ),
        )

    def _base_queries(self, state, batch_size, device, dtype):
        if self.query_mode == "fresh":
            return self.query_embed.weight.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        return state.queries.to(device=device, dtype=dtype)

    def calibrate_lifecycle_logits(self, raw_logits):
        """Apply a ranking-preserving calibration trained independently of raw heads."""

        if raw_logits.shape[-1] != 3:
            raise ValueError(
                "lifecycle logits must end with birth/alive/end channels"
            )
        if self.lifecycle_calibration_mode == "none":
            return raw_logits
        scale = self.lifecycle_calibration_log_scale.exp().to(raw_logits)
        bias = self.lifecycle_calibration_bias.to(raw_logits)
        return raw_logits.detach() * scale + bias

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

        raw_birth_logits = self.birth_head(queries).squeeze(-1)
        raw_alive_logits = self.alive_head(queries).squeeze(-1)
        if self.end_transition_mode == "causal_delta_mlp":
            current_token = projected.expand(
                -1,
                self.num_slots,
                -1,
            )
            end_features = self.end_transition_proj(
                torch.cat(
                    [base, queries, queries - base, current_token],
                    dim=-1,
                )
            )
        else:
            end_features = queries
        raw_end_logits = self.end_head(end_features).squeeze(-1)
        calibrated_lifecycle_logits = self.calibrate_lifecycle_logits(
            torch.stack(
                [raw_birth_logits, raw_alive_logits, raw_end_logits],
                dim=-1,
            )
        )
        outputs = {
            "raw_birth_logits": raw_birth_logits,
            "raw_alive_logits": raw_alive_logits,
            "raw_end_hazard_logits": raw_end_logits,
            "birth_logits": calibrated_lifecycle_logits[..., 0],
            "alive_logits": calibrated_lifecycle_logits[..., 1],
            "class_logits": self.class_head(queries),
            "end_hazard_logits": calibrated_lifecycle_logits[..., 2],
            "start_offset": (
                self.start_offset_head(queries).sigmoid().squeeze(-1)
                * self.max_start_offset
            ),
            "memory_frames": memory_frames,
        }
        if self.start_mode == "pointer":
            pointer_scores = (
                torch.einsum("bkd,bmd->bkm", queries, memory)
                / math.sqrt(self.hidden_dim)
            )
            sentinel = torch.einsum(
                "bkd,d->bk",
                queries,
                self.before_memory.to(queries),
            ).unsqueeze(-1)
            outputs["start_pointer_logits"] = torch.cat(
                [sentinel, pointer_scores],
                dim=-1,
            )
        if self.endpoint_start_mode == "past_pointer":
            endpoint_query = self.endpoint_start_query(queries)
            endpoint_pointer_scores = (
                torch.einsum("bkd,bmd->bkm", endpoint_query, memory)
                / math.sqrt(self.hidden_dim)
            )
            endpoint_sentinel = torch.einsum(
                "bkd,d->bk",
                endpoint_query,
                self.endpoint_before_memory.to(endpoint_query),
            ).unsqueeze(-1)
            outputs["endpoint_start_pointer_logits"] = torch.cat(
                [endpoint_sentinel, endpoint_pointer_scores],
                dim=-1,
            )
        if self.endpoint_mode == "hazard":
            outputs["endpoint_offset"] = (
                self.endpoint_offset_head(queries).sigmoid().squeeze(-1)
                * self.max_endpoint_offset
            )
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
            ledger=state.ledger,
            instance_to_slot=state.instance_to_slot,
            slot_to_instance=state.slot_to_instance,
            candidate_age=state.candidate_age,
            peak_class_labels=state.peak_class_labels,
            birth_proposals=state.birth_proposals,
            birth_admissions=state.birth_admissions,
            arbitration_suppressions=state.arbitration_suppressions,
            candidate_cancellations=state.candidate_cancellations,
            active_abandonments=state.active_abandonments,
            deferred_birth_due_to_release=state.deferred_birth_due_to_release,
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

    def _decode_end(self, outputs, slot, start_frame, current_frame):
        if self.endpoint_mode == "binary":
            return int(current_frame)
        offset = float(outputs["endpoint_offset"][0, slot].detach().item())
        decoded = int(round(float(current_frame) - offset))
        return min(int(current_frame), max(int(start_frame), decoded))

    @staticmethod
    def _positive_causal_interval(start_frame, end_frame, feature_stride=1):
        """Quantize a same-decision short action to one observed feature cell."""

        start_frame = int(start_frame)
        end_frame = int(end_frame)
        feature_stride = int(feature_stride)
        if feature_stride <= 0:
            raise ValueError("feature_stride must be positive")
        if start_frame > end_frame:
            raise ValueError("decoded action start must not exceed its endpoint")
        if start_frame == end_frame:
            start_frame = max(0, end_frame - feature_stride)
        if not 0 <= start_frame < end_frame:
            raise ValueError(
                "a final action must have positive duration within observed frames"
            )
        return start_frame, end_frame

    def _decode_endpoint_start(
        self,
        outputs,
        slot,
        fallback_start_frame,
        current_frame,
    ):
        fallback = int(fallback_start_frame)
        if self.endpoint_start_mode != "past_pointer":
            return fallback
        pointer_index = int(
            outputs["endpoint_start_pointer_logits"][0, slot]
            .detach()
            .argmax()
            .item()
        )
        memory_frames = tuple(int(frame) for frame in outputs["memory_frames"])
        if pointer_index == 0 or not memory_frames:
            return fallback
        selected = memory_frames[
            min(pointer_index - 1, len(memory_frames) - 1)
        ]
        return min(int(current_frame), max(0, int(selected)))

    def _decode_legacy(self, outputs, state, current_frame, feature_stride=1):
        current_frame = int(current_frame)
        birth = outputs["birth_logits"].detach().sigmoid()[0]
        alive = outputs["alive_logits"].detach().sigmoid()[0]
        end = outputs["end_hazard_logits"].detach().sigmoid()[0]
        class_probs = outputs["class_logits"].detach().softmax(dim=-1)[0]
        class_scores, labels = class_probs.max(dim=-1)

        status = state.slot_status.clone()
        refractory = state.refractory.clone()
        start_frames = state.start_frames.clone()
        peak_scores = state.peak_class_scores.clone()
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
                just_born = True

            if (
                not just_born
                and alive[slot] < self.alive_threshold
                and end[slot] < self.end_threshold
            ):
                status[slot] = SLOT_FREE
                start_frames[slot] = float("nan")
                peak_scores[slot] = 0.0
                continue

            peak_scores[slot] = torch.maximum(peak_scores[slot], class_scores[slot])
            if end[slot] < self.end_threshold:
                continue

            start_frame = int(round(float(start_frames[slot].item())))
            end_frame = self._decode_end(outputs, slot, start_frame, current_frame)
            start_frame, end_frame = self._positive_causal_interval(
                start_frame,
                end_frame,
                feature_stride,
            )
            score = math.sqrt(max(float(peak_scores[slot].item()) * float(end[slot].item()), 0.0))
            record = EventSetEmissionRecord(
                stream_key=state.stream_key,
                slot_id=slot,
                label=int(labels[slot].item()),
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

        next_state = PersistentEventSetState(
            stream_key=state.stream_key,
            queries=state.queries,
            feature_memory=state.feature_memory,
            memory_frames=state.memory_frames,
            slot_status=status,
            refractory=refractory,
            start_frames=start_frames,
            peak_class_scores=peak_scores,
            ledger=ledger,
            instance_to_slot=state.instance_to_slot,
            slot_to_instance=state.slot_to_instance,
            candidate_age=state.candidate_age,
            peak_class_labels=state.peak_class_labels,
            birth_proposals=state.birth_proposals,
            birth_admissions=state.birth_admissions,
            arbitration_suppressions=state.arbitration_suppressions,
            candidate_cancellations=state.candidate_cancellations,
            active_abandonments=state.active_abandonments,
            deferred_birth_due_to_release=state.deferred_birth_due_to_release,
        )
        return emitted, next_state

    def _decode_candidate(self, outputs, state, current_frame, feature_stride=1):
        current_frame = int(current_frame)
        birth = outputs["birth_logits"].detach().sigmoid()[0]
        alive = outputs["alive_logits"].detach().sigmoid()[0]
        end = outputs["end_hazard_logits"].detach().sigmoid()[0]
        class_probs = outputs["class_logits"].detach().softmax(dim=-1)[0]
        class_scores, labels = class_probs.max(dim=-1)

        status = state.slot_status.clone()
        free_at_entry = status.eq(SLOT_FREE)
        candidate_age = (
            state.candidate_age.clone()
            if state.candidate_age is not None
            else torch.zeros_like(status)
        )
        start_frames = state.start_frames.clone()
        peak_scores = state.peak_class_scores.clone()
        peak_labels = (
            state.peak_class_labels.clone()
            if state.peak_class_labels is not None
            else torch.full_like(status, -1)
        )
        ledger = list(state.ledger)
        emitted = []
        released_slots = []
        candidate_cancellations = int(state.candidate_cancellations)
        active_abandonments = int(state.active_abandonments)

        def reset_slot(slot):
            status[slot] = SLOT_FREE
            candidate_age[slot] = 0
            start_frames[slot] = float("nan")
            peak_scores[slot] = 0.0
            peak_labels[slot] = -1
            released_slots.append(int(slot))

        def update_peak(slot):
            if class_scores[slot] > peak_scores[slot]:
                peak_scores[slot] = class_scores[slot]
                peak_labels[slot] = labels[slot]

        def commit(slot):
            update_peak(slot)
            fallback_start_frame = int(
                round(float(start_frames[slot].item()))
            )
            start_frame = self._decode_endpoint_start(
                outputs,
                slot,
                fallback_start_frame,
                current_frame,
            )
            end_frame = self._decode_end(outputs, slot, start_frame, current_frame)
            start_frame, end_frame = self._positive_causal_interval(
                start_frame,
                end_frame,
                feature_stride,
            )
            score = math.sqrt(
                max(
                    float(peak_scores[slot].item()) * float(end[slot].item()),
                    0.0,
                )
            )
            record = EventSetEmissionRecord(
                stream_key=state.stream_key,
                slot_id=int(slot),
                label=int(peak_labels[slot].item()),
                score=score,
                start_frame=start_frame,
                end_frame=end_frame,
                emit_frame=current_frame,
                max_source_frame=max(int(frame) for frame in outputs["memory_frames"]),
            )
            ledger.append(record)
            emitted.append(record)
            reset_slot(slot)

        for slot in range(self.num_slots):
            slot_status = int(status[slot].item())
            if slot_status == SLOT_CANDIDATE:
                update_peak(slot)
                if end[slot] >= self.end_threshold:
                    commit(slot)
                elif alive[slot] >= self.alive_threshold:
                    status[slot] = SLOT_ACTIVE
                    candidate_age[slot] = 0
                else:
                    candidate_cancellations += 1
                    reset_slot(slot)
                continue
            if slot_status != SLOT_ACTIVE:
                continue
            update_peak(slot)
            if end[slot] >= self.end_threshold:
                commit(slot)
            elif alive[slot] < self.alive_threshold:
                active_abandonments += 1
                reset_slot(slot)

        eligible = tuple(
            slot
            for slot in range(self.num_slots)
            if bool(free_at_entry[slot].item()) and birth[slot] >= self.birth_threshold
        )
        ranked = tuple(
            sorted(
                eligible,
                key=lambda slot: (-float(birth[slot].item()), int(slot)),
            )
        )
        admitted = ranked[: self.max_births_per_step]
        for slot in admitted:
            status[slot] = SLOT_CANDIDATE
            candidate_age[slot] = self.candidate_confirmation_steps
            start_frames[slot] = float(
                self._decode_start(outputs, slot, current_frame, feature_stride)
            )
            peak_scores[slot] = class_scores[slot]
            peak_labels[slot] = labels[slot]
            if end[slot] >= self.end_threshold:
                commit(slot)

        deferred_birth_due_to_release = int(state.deferred_birth_due_to_release)
        deferred_birth_due_to_release += sum(
            int(
                not bool(free_at_entry[slot].item())
                and birth[slot] >= self.birth_threshold
            )
            for slot in released_slots
        )

        queries = state.queries
        if released_slots:
            reset_mask = torch.zeros(
                self.num_slots,
                dtype=torch.bool,
                device=queries.device,
            )
            reset_mask[released_slots] = True
            learned = self.query_embed.weight.to(device=queries.device, dtype=queries.dtype)
            queries = torch.where(
                reset_mask.view(1, -1, 1),
                learned.unsqueeze(0),
                queries,
            )

        next_state = PersistentEventSetState(
            stream_key=state.stream_key,
            queries=queries,
            feature_memory=state.feature_memory,
            memory_frames=state.memory_frames,
            slot_status=status,
            refractory=torch.zeros_like(state.refractory),
            start_frames=start_frames,
            peak_class_scores=peak_scores,
            ledger=ledger,
            instance_to_slot=state.instance_to_slot,
            slot_to_instance=state.slot_to_instance,
            candidate_age=candidate_age,
            peak_class_labels=peak_labels,
            birth_proposals=int(state.birth_proposals) + len(ranked),
            birth_admissions=int(state.birth_admissions) + len(admitted),
            arbitration_suppressions=(
                int(state.arbitration_suppressions) + len(ranked) - len(admitted)
            ),
            candidate_cancellations=candidate_cancellations,
            active_abandonments=active_abandonments,
            deferred_birth_due_to_release=deferred_birth_due_to_release,
        )
        return emitted, next_state

    def decode_step(self, outputs, state, current_frame, feature_stride=1):
        if self.lifecycle_mode == "candidate_recycle":
            return self._decode_candidate(
                outputs,
                state,
                current_frame,
                feature_stride=feature_stride,
            )
        return self._decode_legacy(
            outputs,
            state,
            current_frame,
            feature_stride=feature_stride,
        )


__all__ = [
    "SLOT_FREE",
    "SLOT_ACTIVE",
    "SLOT_CANDIDATE",
    "SLOT_REFRACTORY",
    "EventSetEmissionRecord",
    "PersistentEventSetState",
    "PersistentEventSetHead",
]
