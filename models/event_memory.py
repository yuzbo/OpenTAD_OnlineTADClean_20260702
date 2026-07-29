"""Dynamic causal event records for MATR-derived online localization.

This module deliberately contains no dataset or ground-truth access.  It is a
runtime state machine driven only by the current model outputs and immutable
past state.  The MATR backbone keeps its fixed query bandwidth, while event
records are allocated dynamically and therefore are not a hand-sized semantic
slot bank.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_BIRTH_MODES = {"matr_delayed", "instant_transition"}
_OWNERSHIP_MODES = {"fresh_rematch", "sticky_owner"}


def temporal_viterbi_assignment(
    state_logits: torch.Tensor,
    class_logits: torch.Tensor,
    query_features: torch.Tensor,
    class_id: int,
) -> List[int]:
    """Return one stable, causal query path for a pre-birth event window.

    The matcher is deliberately stop-gradient: it constructs a temporally
    consistent supervision path, while gradients flow through the logits and
    features selected by that path.  Emissions prefer START/class evidence and
    transitions prefer feature continuity.  No future prefix outside the
    supplied chronological window is inspected.
    """

    if state_logits.ndim != 3 or state_logits.size(-1) != 4:
        raise ValueError("state_logits must be [T,Q,4]")
    if class_logits.shape[:2] != state_logits.shape[:2]:
        raise ValueError("class_logits must start with [T,Q]")
    if query_features.shape[:2] != state_logits.shape[:2]:
        raise ValueError("query_features must start with [T,Q]")
    time_steps, query_count, _ = state_logits.shape
    if time_steps == 0 or query_count == 0:
        return []

    foreground_classes = max(1, class_logits.size(-1) - 1)
    class_id = min(max(0, int(class_id)), foreground_classes - 1)
    with torch.no_grad():
        emission = (
            state_logits.log_softmax(dim=-1)[..., 1]
            + class_logits.log_softmax(dim=-1)[..., class_id]
        )
        normalized = F.normalize(query_features, dim=-1)
        score = emission[0]
        backpointers = []
        for time_index in range(1, time_steps):
            continuity = normalized[time_index - 1] @ normalized[time_index].transpose(
                0, 1
            )
            candidate = score.unsqueeze(1) + continuity
            best_previous = candidate.argmax(dim=0)
            score = emission[time_index] + candidate.max(dim=0).values
            backpointers.append(best_previous)
        query_index = int(score.argmax().item())
        path = [query_index]
        for best_previous in reversed(backpointers):
            query_index = int(best_previous[query_index].item())
            path.append(query_index)
        path.reverse()
    return path


def resolve_model_variant(args) -> str:
    """Keep exact upstream MATR separate from every EventMATR factor cell."""
    variant = getattr(args, "model_variant", None)
    if variant is None:
        variant = "eventmatr" if getattr(args, "event_arm", None) else "native_matr"
    if variant not in {"native_matr", "eventmatr"}:
        raise ValueError(
            "model_variant must be 'native_matr' or 'eventmatr', got {!r}".format(
                variant
            )
        )
    if variant == "native_matr" and getattr(args, "event_arm", None):
        raise ValueError("native_matr must not carry an EventMATR event_arm")
    return variant


def resolve_event_modes(args) -> Tuple[str, str, str]:
    """Resolve the public two-factor interface without changing MATR defaults."""
    birth_mode = getattr(
        args, "birth_mode", getattr(args, "event_birth_mode", "matr_delayed")
    )
    ownership_mode = getattr(
        args,
        "ownership_mode",
        getattr(args, "event_owner_mode", "fresh_rematch"),
    )
    if birth_mode not in _BIRTH_MODES:
        raise ValueError(
            "birth_mode must be one of {}, got {!r}".format(
                sorted(_BIRTH_MODES), birth_mode
            )
        )
    if ownership_mode not in _OWNERSHIP_MODES:
        raise ValueError(
            "ownership_mode must be one of {}, got {!r}".format(
                sorted(_OWNERSHIP_MODES), ownership_mode
            )
        )

    arm = {
        ("matr_delayed", "fresh_rematch"): "b0o0",
        ("instant_transition", "fresh_rematch"): "b1o0",
        ("matr_delayed", "sticky_owner"): "b0o1",
        ("instant_transition", "sticky_owner"): "b1o1",
    }[(birth_mode, ownership_mode)]
    requested_arm = getattr(args, "event_arm", None)
    if requested_arm not in (None, "", arm):
        raise ValueError(
            "event_arm={!r} conflicts with birth_mode={!r}, "
            "ownership_mode={!r} (derived {!r})".format(
                requested_arm, birth_mode, ownership_mode, arm
            )
        )
    return birth_mode, ownership_mode, arm


class EventTransitionHead(nn.Module):
    """Competitive background/start/alive/end state prediction.

    Runtime transitions use the winning-state margin.  Consequently the
    scientific rule is learned competition among four states, not a universal
    sigmoid probability threshold of 0.5.  Optional logit-margin thresholds in
    ``DynamicEventMemory`` are only calibration/test injection points.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        fused_dim = hidden_dim * 2
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.state = nn.Linear(hidden_dim, 4)

    def forward(
        self, class_query: torch.Tensor, regression_query: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        state = self.fuse(torch.cat((class_query, regression_query), dim=-1))
        state_logits = self.state(state)
        margins = []
        for state_index in (1, 2, 3):
            competing = torch.cat(
                (
                    state_logits[..., :state_index],
                    state_logits[..., state_index + 1 :],
                ),
                dim=-1,
            )
            margins.append(
                state_logits[..., state_index] - competing.max(dim=-1).values
            )
        return (
            state_logits,
            margins[0],
            margins[1],
            margins[2],
            state,
        )


class OwnerEventDecoder(nn.Module):
    """Decode an arbitrary number of persistent owner records.

    Owners cross-attend to the current fixed-width MATR query set.  The owner
    axis is ragged/padded and may exceed ``num_queries``; it is not a semantic
    slot bank.  The same decoder is trained on per-query owner prototypes and
    applied to copied persistent owner embeddings at runtime.
    """

    def __init__(self, hidden_dim: int, num_classes: int, num_heads: int = 4):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.state = nn.Linear(hidden_dim, 4)
        self.end_offset = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())
        self.classification = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        owner_embeddings: torch.Tensor,
        current_queries: torch.Tensor,
        owner_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attended = self.cross_attention(
            owner_embeddings, current_queries, current_queries
        )[0]
        owner_state = self.norm1(owner_embeddings + attended)
        owner_state = self.norm2(owner_state + self.ffn(owner_state))
        state_logits = self.state(owner_state)
        end_offsets = self.end_offset(owner_state).squeeze(-1)
        class_logits = self.classification(owner_state)
        if owner_padding_mask is not None:
            state_logits = state_logits.masked_fill(
                owner_padding_mask.unsqueeze(-1), 0.0
            )
            end_offsets = end_offsets.masked_fill(owner_padding_mask, 0.0)
            class_logits = class_logits.masked_fill(
                owner_padding_mask.unsqueeze(-1), 0.0
            )
        return state_logits, end_offsets, class_logits, owner_state


@dataclass
class EventRecord:
    event_id: int
    video_name: str
    start_frame: float
    owner_query_id: int
    owner_embedding: torch.Tensor
    class_distribution: torch.Tensor
    birth_score: float
    created_frame: float
    end_frame: Optional[float] = None
    end_score: Optional[float] = None
    emit_frame: Optional[float] = None
    status: str = "active"
    target_event_id: Optional[int] = None
    source: str = "predicted"
    reacquisition_count: int = 0


class DynamicEventMemory:
    """GT-free dynamic event memory with explicit end/emit separation.

    ``resource_limit`` is an optional physical fail-closed guard.  Zero means
    unlimited and is the default; no event is silently evicted or truncated.
    """

    def __init__(
        self,
        birth_mode: str,
        ownership_mode: str,
        birth_logit_threshold: Optional[float] = None,
        end_logit_threshold: Optional[float] = None,
        min_duration_frames: int = 1,
        emit_delay_frames: int = 0,
        resource_limit: int = 0,
        segment_size: int = 64,
        enable_reacquisition: bool = False,
        strict_causal_boundary: bool = False,
    ):
        if birth_mode not in _BIRTH_MODES:
            raise ValueError("invalid birth_mode: {!r}".format(birth_mode))
        if ownership_mode not in _OWNERSHIP_MODES:
            raise ValueError("invalid ownership_mode: {!r}".format(ownership_mode))
        if min_duration_frames < 1:
            raise ValueError("min_duration_frames must be positive")
        if emit_delay_frames < 0:
            raise ValueError("emit_delay_frames must be non-negative")
        if resource_limit < 0:
            raise ValueError("resource_limit must be non-negative")
        self.birth_mode = birth_mode
        self.ownership_mode = ownership_mode
        self.birth_logit_threshold = (
            None
            if birth_logit_threshold is None
            else float(birth_logit_threshold)
        )
        self.end_logit_threshold = (
            None if end_logit_threshold is None else float(end_logit_threshold)
        )
        self.min_duration_frames = int(min_duration_frames)
        self.emit_delay_frames = int(emit_delay_frames)
        self.resource_limit = int(resource_limit)
        self.segment_size = int(segment_size)
        self.enable_reacquisition = bool(enable_reacquisition)
        self.strict_causal_boundary = bool(strict_causal_boundary)

        self._records: Dict[str, List[EventRecord]] = {}
        self._cancelled_records: Dict[str, List[EventRecord]] = {}
        self._ledger: Dict[str, List[dict]] = {}
        self._previous_start_active: Dict[str, torch.Tensor] = {}
        self._last_frame: Dict[str, float] = {}
        # Frozen v1 compatibility only.  D1 rejects this metadata at its model
        # boundary and never populates the map.
        self._true_last_frame: Dict[str, float] = {}
        self._next_event_id: Dict[str, int] = {}
        self._next_sequence_id: Dict[str, int] = {}
        self.last_audit = {
            "births": 0,
            "ends": 0,
            "emits": 0,
            "cancellations": [],
            "reacquisitions": 0,
            "runtime_capacity_exhaustions": 0,
            "padding_prefixes_ignored": 0,
            "eos_observed": 0,
        }

    def reset(self, video_name: Optional[str] = None) -> None:
        if video_name is None:
            self._records.clear()
            self._cancelled_records.clear()
            self._ledger.clear()
            self._previous_start_active.clear()
            self._last_frame.clear()
            self._true_last_frame.clear()
            self._next_event_id.clear()
            self._next_sequence_id.clear()
        else:
            self._records.pop(video_name, None)
            self._cancelled_records.pop(video_name, None)
            self._ledger.pop(video_name, None)
            self._previous_start_active.pop(video_name, None)
            self._last_frame.pop(video_name, None)
            self._true_last_frame.pop(video_name, None)
            self._next_event_id.pop(video_name, None)
            self._next_sequence_id.pop(video_name, None)
        self.last_audit = {
            "births": 0,
            "ends": 0,
            "emits": 0,
            "cancellations": [],
            "reacquisitions": 0,
            "runtime_capacity_exhaustions": 0,
            "padding_prefixes_ignored": 0,
            "eos_observed": 0,
        }

    def records(self, video_name: str) -> Tuple[EventRecord, ...]:
        return tuple(self._records.get(str(video_name), ()))

    def ledger(self, video_name: str) -> Tuple[dict, ...]:
        rows = sorted(
            self._ledger.get(str(video_name), ()),
            key=lambda row: (row["emit_frame"], row["sequence_id"]),
        )
        return tuple(dict(row) for row in rows)

    def owner_batch(
        self, video_names, *, device, dtype, embedding_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return padded active owner embeddings and stable event ids."""
        names = self._as_names(video_names)
        rows = [
            [record for record in self._records.get(name, []) if record.status == "active"]
            for name in names
        ]
        max_records = max((len(records) for records in rows), default=0)
        owners = torch.zeros(
            (len(names), max_records, embedding_dim), device=device, dtype=dtype
        )
        padding_mask = torch.ones(
            (len(names), max_records), device=device, dtype=torch.bool
        )
        event_ids = torch.full(
            (len(names), max_records), -1, device=device, dtype=torch.long
        )
        for batch_index, records in enumerate(rows):
            for record_index, record in enumerate(records):
                owners[batch_index, record_index] = record.owner_embedding.to(
                    device=device, dtype=dtype
                )
                padding_mask[batch_index, record_index] = False
                event_ids[batch_index, record_index] = int(record.event_id)
        return owners, padding_mask, event_ids

    def record_metadata(self, video_name: str, event_ids: torch.Tensor) -> List[dict]:
        """Expose non-tensor identity metadata for ragged training supervision."""

        records = {
            int(record.event_id): record
            for record in self._records.get(str(video_name), ())
            if record.status == "active"
        }
        metadata = []
        for event_id in event_ids.detach().cpu().reshape(-1).tolist():
            record = records.get(int(event_id))
            if record is None:
                metadata.append(
                    {
                        "event_id": int(event_id),
                        "target_event_id": None,
                        "source": "missing",
                        "owner_query_id": -1,
                    }
                )
            else:
                metadata.append(
                    {
                        "event_id": int(record.event_id),
                        "target_event_id": record.target_event_id,
                        "source": record.source,
                        "owner_query_id": int(record.owner_query_id),
                    }
                )
        return metadata

    def detach_graph(self) -> None:
        """Detach carried state after one chronological training unroll.

        Owner updates inside a batch remain differentiable.  Detaching only the
        persistent copies prevents a graph from leaking across optimizer steps.
        """

        for records in self._records.values():
            for record in records:
                record.owner_embedding = record.owner_embedding.detach()
                record.class_distribution = record.class_distribution.detach()

    @torch.no_grad()
    def rematch_active_owners(
        self,
        video_name: str,
        query_features: torch.Tensor,
        class_logits: torch.Tensor,
    ) -> None:
        """Refresh O0 owner copies from the current prefix before decoding."""
        if self.ownership_mode != "fresh_rematch":
            return
        video_name = str(video_name)
        records = [
            record
            for record in self._records.get(video_name, [])
            if record.status == "active"
        ]
        foreground = self._foreground_distribution(class_logits.detach())
        assignments = self._match_records(records, query_features.detach(), foreground)
        for record_index, query_index in assignments.items():
            records[record_index].class_distribution = foreground[
                query_index
            ].detach().clone()

    def cancel(
        self, video_name: str, event_id: int, reason: str = "false_birth"
    ) -> bool:
        video_name = str(video_name)
        records = self._records.get(video_name, [])
        for index, record in enumerate(records):
            if record.event_id == int(event_id) and record.status != "emitted":
                record.status = "cancelled"
                del records[index]
                if self.strict_causal_boundary:
                    self._cancelled_records.setdefault(video_name, []).append(record)
                self.last_audit.setdefault("cancellations", []).append(
                    {
                        "video_name": video_name,
                        "event_id": int(event_id),
                        "reason": str(reason),
                    }
                )
                return True
        return False

    @staticmethod
    def _as_names(video_names: Sequence) -> List[str]:
        return [str(name) for name in video_names]

    @staticmethod
    def _as_frames(current_frames, batch_size: int) -> List[float]:
        if torch.is_tensor(current_frames):
            values = current_frames.detach().cpu().reshape(-1).tolist()
        elif isinstance(current_frames, (list, tuple)):
            values = list(current_frames)
        else:
            values = [current_frames]
        if len(values) != batch_size:
            raise ValueError("current_frames must have one value per video")
        return [float(value) for value in values]

    @staticmethod
    def _foreground_distribution(class_logits: torch.Tensor) -> torch.Tensor:
        probs = class_logits.softmax(dim=-1)
        if probs.size(-1) <= 1:
            return probs
        foreground = probs[..., :-1]
        return foreground / foreground.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _match_records(
        self,
        records: List[EventRecord],
        query_features: torch.Tensor,
        class_distribution: torch.Tensor,
    ) -> Dict[int, int]:
        query_count = query_features.size(0)
        if not records or query_count == 0:
            return {}
        if self.ownership_mode == "sticky_owner":
            # Sticky means the persistent record owns its birth query identity.
            # Multiple records may exist beyond Q; the query bandwidth is not
            # used as a semantic storage limit.
            assignments = {}
            for index, record in enumerate(records):
                query_index = int(record.owner_query_id)
                if query_index < 0 or query_index >= query_count:
                    raise RuntimeError(
                        "sticky owner query {} is outside current query bandwidth {}".format(
                            query_index, query_count
                        )
                    )
                assignments[index] = query_index
            return assignments

        anchors = torch.stack(
            [record.owner_embedding.to(query_features) for record in records], dim=0
        )
        classes = torch.stack(
            [record.class_distribution.to(class_distribution) for record in records],
            dim=0,
        )
        embedding_score = F.normalize(anchors, dim=-1) @ F.normalize(
            query_features, dim=-1
        ).transpose(0, 1)
        class_score = classes @ class_distribution.transpose(0, 1)
        score = embedding_score + class_score

        # Deterministic greedy one-to-one association.  If records outnumber
        # the official query bandwidth, unmatched records remain active.
        assignments: Dict[int, int] = {}
        available_records = set(range(score.size(0)))
        available_queries = set(range(score.size(1)))
        while available_records and available_queries:
            best = None
            for record_index in sorted(available_records):
                for query_index in sorted(available_queries):
                    candidate = (
                        float(score[record_index, query_index].item()),
                        -record_index,
                        -query_index,
                    )
                    if best is None or candidate > best[0]:
                        best = (candidate, record_index, query_index)
            _, record_index, query_index = best
            assignments[record_index] = query_index
            available_records.remove(record_index)
            available_queries.remove(query_index)
        for record_index, query_index in assignments.items():
            records[record_index].owner_query_id = int(query_index)
            records[record_index].owner_embedding = query_features[
                query_index
            ].detach().clone()
        return assignments

    def _check_frame_order(self, video_name: str, frame: float) -> None:
        previous = self._last_frame.get(video_name)
        if previous is not None and frame < previous:
            raise RuntimeError(
                "non-causal frame order for {}: {} after {}".format(
                    video_name, frame, previous
                )
            )
        self._last_frame[video_name] = frame

    def _emit_record(self, record: EventRecord, emit_frame: float) -> dict:
        if record.end_frame is None:
            raise RuntimeError("cannot emit an event before its end")
        if record.end_frame <= record.start_frame:
            raise RuntimeError("event interval must have strictly positive length")
        true_last = self._true_last_frame.get(record.video_name)
        if true_last is not None and (
            record.end_frame > true_last or float(emit_frame) > true_last
        ):
            raise RuntimeError(
                "event end/emit exceeds the last real prefix for {}".format(
                    record.video_name
                )
            )
        record.emit_frame = float(emit_frame)
        record.status = "emitted"
        class_id = int(record.class_distribution.argmax().item())
        class_confidence = float(record.class_distribution.max().item())
        if record.end_score is None:
            raise RuntimeError("emitted EventMATR record has no end-state confidence")
        score = max(
            0.0,
            min(
                1.0,
                float(record.birth_score)
                * class_confidence
                * float(record.end_score),
            ),
        ) ** (1.0 / 3.0)
        sequence_id = self._next_sequence_id.get(record.video_name, 0)
        self._next_sequence_id[record.video_name] = sequence_id + 1
        row = {
            "event_id": int(record.event_id),
            "sequence_id": int(sequence_id),
            "video_name": record.video_name,
            "start_frame": float(record.start_frame),
            "end_frame": float(record.end_frame),
            "emit_frame": float(record.emit_frame),
            "owner_query_id": int(record.owner_query_id),
            "class_id": class_id,
            "birth_state_confidence": float(record.birth_score),
            "class_confidence": class_confidence,
            "end_state_confidence": float(record.end_score),
            "score": score,
            "status": "emitted",
            "source": record.source,
            "reacquisition_count": int(record.reacquisition_count),
        }
        self._ledger.setdefault(record.video_name, []).append(row)
        return row

    def step(
        self,
        *,
        video_names,
        current_frames,
        birth_logits: torch.Tensor,
        alive_logits: torch.Tensor,
        end_logits: torch.Tensor,
        end_offsets: torch.Tensor,
        class_logits: torch.Tensor,
        query_features: torch.Tensor,
        candidate_start_frames: torch.Tensor,
        candidate_state_logits: Optional[torch.Tensor] = None,
        is_real_prefix=None,
        true_durations=None,
        is_eos=None,
        preserve_graph: bool = False,
        oracle_births=None,
        owner_state_logits: Optional[torch.Tensor] = None,
        owner_end_offsets: Optional[torch.Tensor] = None,
        owner_class_logits: Optional[torch.Tensor] = None,
        owner_updated_embeddings: Optional[torch.Tensor] = None,
        owner_valid_mask: Optional[torch.Tensor] = None,
        owner_record_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if birth_logits.ndim != 2:
            raise ValueError("birth_logits must be [B,Q]")
        batch_size, query_count = birth_logits.shape
        expected = (batch_size, query_count)
        for name, value in (
            ("alive_logits", alive_logits),
            ("end_logits", end_logits),
            ("end_offsets", end_offsets),
            ("candidate_start_frames", candidate_start_frames),
        ):
            if tuple(value.shape) != expected:
                raise ValueError("{} must have shape {}".format(name, expected))
        if class_logits.shape[:2] != expected or query_features.shape[:2] != expected:
            raise ValueError("class_logits/query_features must start with [B,Q]")
        if candidate_state_logits is not None and tuple(
            candidate_state_logits.shape
        ) != (batch_size, query_count, 4):
            raise ValueError("candidate_state_logits must be [B,Q,4]")
        if self.birth_logit_threshold is None and candidate_state_logits is None:
            raise ValueError(
                "formal learned START decisions require candidate_state_logits"
            )
        if (
            self.end_logit_threshold is None
            and self.ownership_mode == "fresh_rematch"
            and candidate_state_logits is None
        ):
            raise ValueError("formal learned END decisions require state logits")

        device = birth_logits.device
        new_birth_mask = torch.zeros(expected, dtype=torch.bool, device=device)
        ended_mask = torch.zeros(expected, dtype=torch.bool, device=device)
        emitted_mask = torch.zeros(expected, dtype=torch.bool, device=device)
        cancelled_mask = torch.zeros(expected, dtype=torch.bool, device=device)
        active_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        birth_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        end_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        emit_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        cancellation_count = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        reacquisition_count = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        capacity_exhaustions = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        padding_ignored = torch.zeros(batch_size, dtype=torch.long, device=device)
        eos_observed = torch.zeros(batch_size, dtype=torch.long, device=device)
        names = self._as_names(video_names)
        if len(names) != batch_size:
            raise ValueError("video_names must have one value per batch item")
        frames = self._as_frames(current_frames, batch_size)
        foreground = self._foreground_distribution(class_logits.detach())

        def optional_values(values, default):
            if values is None:
                return [default] * batch_size
            if torch.is_tensor(values):
                values = values.detach().cpu().reshape(-1).tolist()
            elif not isinstance(values, (list, tuple)):
                values = [values]
            if len(values) != batch_size:
                raise ValueError("prefix metadata must match batch size")
            return list(values)

        real_prefixes = [bool(value) for value in optional_values(is_real_prefix, True)]
        durations = optional_values(true_durations, None)
        eos_flags = [bool(value) for value in optional_values(is_eos, False)]
        oracle_rows = optional_values(oracle_births, ())

        step_audit = {
            "births": 0,
            "ends": 0,
            "emits": 0,
            "cancellations": list(self.last_audit.get("cancellations", [])),
            "reacquisitions": 0,
            "runtime_capacity_exhaustions": 0,
            "padding_prefixes_ignored": 0,
            "eos_observed": 0,
        }
        for batch_index, (video_name, frame) in enumerate(zip(names, frames)):
            records = self._records.setdefault(video_name, [])
            if not real_prefixes[batch_index]:
                # Padding is not a causal observation.  It must not alter any
                # lifecycle state, including previous START, owner updates, a
                # delayed emit, or sequence numbering.
                padding_ignored[batch_index] = 1
                step_audit["padding_prefixes_ignored"] += 1
                active_count[batch_index] = len(records)
                continue

            duration = durations[batch_index]
            if duration is not None:
                if self.enable_reacquisition:
                    raise RuntimeError(
                        "D1 runtime forbids true_duration/full-video metadata"
                    )
                true_last = float(duration) - 1.0
                if true_last < 0:
                    raise ValueError("true_duration must be positive")
                known_last = self._true_last_frame.get(video_name)
                if known_last is not None and known_last != true_last:
                    raise RuntimeError("true_duration changed within a video stream")
                self._true_last_frame[video_name] = true_last
                if frame > true_last:
                    raise RuntimeError("real prefix exceeds true video duration")
            if eos_flags[batch_index]:
                # EOS is a current observation only.  The runtime neither knows
                # nor validates a complete-video duration.
                true_last = self._true_last_frame.get(video_name)
                if true_last is not None and frame != true_last:
                    raise RuntimeError("EOS must identify the final real prefix")
                eos_observed[batch_index] = 1
                step_audit["eos_observed"] += 1
            self._check_frame_order(video_name, frame)

            # First update and close events that existed before this prefix.
            active_records = [record for record in records if record.status == "active"]
            use_owner_decode = (
                owner_state_logits is not None and owner_record_ids is not None
            )
            if use_owner_decode:
                id_to_record = {record.event_id: record for record in active_records}
                width = owner_state_logits.size(1)
                for owner_index in range(width):
                    if owner_valid_mask is not None and not bool(
                        owner_valid_mask[batch_index, owner_index].item()
                    ):
                        continue
                    event_id = int(owner_record_ids[batch_index, owner_index].item())
                    record = id_to_record.get(event_id)
                    if record is None:
                        continue
                    if owner_updated_embeddings is not None:
                        updated_embedding = owner_updated_embeddings[
                            batch_index, owner_index
                        ]
                        record.owner_embedding = (
                            updated_embedding.clone()
                            if preserve_graph
                            else updated_embedding.detach().clone()
                        )
                    if owner_class_logits is not None:
                        owner_probs = owner_class_logits[
                            batch_index, owner_index
                        ].softmax(dim=-1)
                        if owner_probs.numel() > 1:
                            owner_probs = owner_probs[:-1]
                            owner_probs = owner_probs / owner_probs.sum().clamp_min(1e-8)
                        record.class_distribution = (
                            owner_probs.clone()
                            if preserve_graph
                            else owner_probs.detach().clone()
                        )
                    state = int(
                        owner_state_logits[batch_index, owner_index].argmax().item()
                    )
                    if state == 0:
                        # BACKGROUND is a learned competitive owner state, not
                        # a probability threshold.  For an already-active
                        # record it means the proposed birth was rejected.
                        record.status = "cancelled"
                        if self.enable_reacquisition:
                            self._cancelled_records.setdefault(video_name, []).append(
                                record
                            )
                        cancelled_mask[
                            batch_index, int(record.owner_query_id)
                        ] = True
                        step_audit["cancellations"].append(
                            {
                                "video_name": video_name,
                                "event_id": int(record.event_id),
                                "reason": "learned_owner_background",
                                "frame": float(frame),
                            }
                        )
                        cancellation_count[batch_index] += 1
                        continue
                    end_margin = owner_state_logits[
                        batch_index, owner_index, 3
                    ] - owner_state_logits[batch_index, owner_index, :3].max()
                    if self.end_logit_threshold is None:
                        should_end = state == 3
                        end_confidence = float(
                            owner_state_logits[
                                batch_index, owner_index
                            ].softmax(dim=-1)[3].item()
                        )
                    else:
                        should_end = (
                            state == 3
                            and float(end_margin.item())
                            >= self.end_logit_threshold
                        )
                        end_confidence = float(torch.sigmoid(end_margin).item())
                    if should_end and frame >= (
                        record.start_frame + self.min_duration_frames
                    ):
                        offset = 0.0
                        if owner_end_offsets is not None:
                            offset = max(
                                0.0,
                                float(
                                    owner_end_offsets[
                                        batch_index, owner_index
                                    ].item()
                                )
                                * self.segment_size,
                            )
                        predicted_end = max(
                            record.start_frame + self.min_duration_frames,
                            min(frame, frame - offset),
                        )
                        true_last = self._true_last_frame.get(video_name)
                        if true_last is not None:
                            predicted_end = min(predicted_end, true_last)
                        record.end_frame = predicted_end
                        record.end_score = end_confidence
                        record.status = "ended"
                        ended_mask[
                            batch_index, int(record.owner_query_id)
                        ] = True
                        step_audit["ends"] += 1
                        end_count[batch_index] += 1
            else:
                assignments = self._match_records(
                    active_records,
                    query_features[batch_index].detach(),
                    foreground[batch_index],
                )
                for record_index, query_index in assignments.items():
                    record = active_records[record_index]
                    record.class_distribution = (
                        0.8 * record.class_distribution.to(foreground)
                        + 0.2 * foreground[batch_index, query_index]
                    ).detach()
                    if self.end_logit_threshold is None:
                        query_state = candidate_state_logits[
                            batch_index, query_index
                        ]
                        should_end = int(query_state.argmax().item()) == 3
                        end_confidence = float(
                            query_state.softmax(dim=-1)[3].item()
                        )
                    else:
                        raw_end = end_logits[batch_index, query_index]
                        should_end = (
                            float(raw_end.item()) >= self.end_logit_threshold
                        )
                        end_confidence = float(torch.sigmoid(raw_end).item())
                    if should_end and frame >= (
                        record.start_frame + self.min_duration_frames
                    ):
                        predicted_end = frame - max(
                            0.0,
                            float(end_offsets[batch_index, query_index].item())
                            * self.segment_size,
                        )
                        predicted_end = min(frame, predicted_end)
                        predicted_end = max(
                            record.start_frame + self.min_duration_frames,
                            predicted_end,
                        )
                        true_last = self._true_last_frame.get(video_name)
                        if true_last is not None:
                            predicted_end = min(predicted_end, true_last)
                        record.end_frame = predicted_end
                        record.end_score = end_confidence
                        record.status = "ended"
                        ended_mask[batch_index, query_index] = True
                        step_audit["ends"] += 1
                        end_count[batch_index] += 1

            # End and emit are separate state transitions, even when delay=0.
            for record in records:
                if (
                    record.status == "ended"
                    and frame >= record.end_frame + self.emit_delay_frames
                ):
                    query_index = int(record.owner_query_id)
                    if query_index < 0 or query_index >= query_count:
                        raise RuntimeError(
                            "record owner query is outside current query bandwidth"
                        )
                    self._emit_record(record, frame)
                    emitted_mask[batch_index, query_index] = True
                    step_audit["emits"] += 1
                    emit_count[batch_index] += 1
            records[:] = [
                record
                for record in records
                if record.status not in {"emitted", "cancelled"}
            ]

            current_birth = birth_logits[batch_index].detach()
            if self.birth_logit_threshold is None:
                current_start = (
                    candidate_state_logits[batch_index].argmax(dim=-1) == 1
                )
                birth_confidences = candidate_state_logits[
                    batch_index
                ].softmax(dim=-1)[:, 1]
            else:
                current_start = current_birth >= self.birth_logit_threshold
                birth_confidences = torch.sigmoid(current_birth)
            previous_start = self._previous_start_active.get(video_name)
            if previous_start is None or previous_start.numel() != query_count:
                previous_start = torch.zeros_like(current_start)
            rising = current_start & ~previous_start
            self._previous_start_active[video_name] = current_start.clone()
            birth_indices = rising.nonzero(as_tuple=False).reshape(-1).tolist()

            oracle_by_query = {}
            for oracle_row in oracle_rows[batch_index] or ():
                query_index = int(oracle_row["query_index"])
                if query_index < 0 or query_index >= query_count:
                    raise RuntimeError(
                        "oracle assignment query {} exceeds bandwidth {}".format(
                            query_index, query_count
                        )
                    )
                oracle_by_query[query_index] = dict(oracle_row)

            # A predicted START that agrees with an oracle path is one mixed
            # predicted/oracle track, not a duplicated semantic event.
            birth_specs = []
            for query_index in birth_indices:
                spec = {
                    "query_index": int(query_index),
                    "target_event_id": None,
                    "source": "predicted",
                    "start_frame": float(
                        candidate_start_frames[batch_index, query_index].item()
                    ),
                }
                if query_index in oracle_by_query:
                    oracle_row = oracle_by_query.pop(query_index)
                    spec.update(oracle_row)
                    spec["source"] = "predicted_oracle"
                birth_specs.append(spec)
            birth_specs.extend(
                spec
                for spec in oracle_by_query.values()
                if bool(spec.get("force_create", True))
            )

            active_target_ids = {
                record.target_event_id
                for record in records
                if record.status == "active" and record.target_event_id is not None
            }
            birth_specs = [
                spec
                for spec in birth_specs
                if spec.get("target_event_id") not in active_target_ids
            ]

            potential_new = len(birth_specs)
            if self.enable_reacquisition:
                cancelled = self._cancelled_records.get(video_name, [])
                cancelled_targets = {
                    record.target_event_id
                    for record in cancelled
                    if record.target_event_id is not None
                }
                potential_new -= sum(
                    spec.get("target_event_id") in cancelled_targets
                    for spec in birth_specs
                )

            if self.resource_limit and len(records) + potential_new > self.resource_limit:
                capacity_exhaustions[batch_index] = 1
                step_audit["runtime_capacity_exhaustions"] += 1
                self.last_audit = step_audit
                raise RuntimeError(
                    "dynamic event resource limit exhausted for {}: {} active + "
                    "{} births > {}; no event was discarded".format(
                        video_name,
                        len(records),
                        potential_new,
                        self.resource_limit,
                    )
                )

            next_id = self._next_event_id.get(video_name, 0)
            for spec in birth_specs:
                query_index = int(spec["query_index"])
                # Both B0 and B1 store the causal past-pointer estimate.  B1's
                # immediate property is its creation time, not a forced
                # equality between estimated start and the current frame.
                start_frame = max(
                    0.0,
                    min(
                        frame,
                        float(spec.get("start_frame", frame)),
                    ),
                )
                target_event_id = spec.get("target_event_id")
                reacquired = None
                if self.enable_reacquisition:
                    cancelled = self._cancelled_records.setdefault(video_name, [])
                    if target_event_id is not None:
                        for archived in reversed(cancelled):
                            if archived.target_event_id == int(target_event_id):
                                reacquired = archived
                                break
                    else:
                        predicted_class = int(
                            foreground[batch_index, query_index].argmax().item()
                        )
                        for archived in reversed(cancelled):
                            archived_class = int(
                                archived.class_distribution.argmax().item()
                            )
                            if (
                                archived.target_event_id is None
                                and archived_class == predicted_class
                            ):
                                reacquired = archived
                                break
                    if reacquired is not None:
                        cancelled.remove(reacquired)
                        reacquired.status = "active"
                        reacquired.owner_query_id = query_index
                        reacquired.owner_embedding = (
                            query_features[batch_index, query_index].clone()
                            if preserve_graph
                            else query_features[
                                batch_index, query_index
                            ].detach().clone()
                        )
                        reacquired.class_distribution = (
                            foreground[batch_index, query_index].clone()
                            if preserve_graph
                            else foreground[
                                batch_index, query_index
                            ].detach().clone()
                        )
                        reacquired.start_frame = min(
                            reacquired.start_frame, start_frame
                        )
                        reacquired.created_frame = frame
                        reacquired.end_frame = None
                        reacquired.end_score = None
                        reacquired.emit_frame = None
                        reacquired.source = "reacquired"
                        reacquired.reacquisition_count += 1
                        records.append(reacquired)
                        new_birth_mask[batch_index, query_index] = True
                        step_audit["births"] += 1
                        step_audit["reacquisitions"] += 1
                        birth_count[batch_index] += 1
                        reacquisition_count[batch_index] += 1
                        continue
                record = EventRecord(
                    event_id=next_id,
                    video_name=video_name,
                    start_frame=float(start_frame),
                    owner_query_id=int(query_index),
                    owner_embedding=(
                        query_features[batch_index, query_index].clone()
                        if preserve_graph
                        else query_features[
                            batch_index, query_index
                        ].detach().clone()
                    ),
                    class_distribution=(
                        foreground[batch_index, query_index].clone()
                        if preserve_graph
                        else foreground[
                            batch_index, query_index
                        ].detach().clone()
                    ),
                    birth_score=float(birth_confidences[query_index].item()),
                    created_frame=frame,
                    target_event_id=(
                        None if target_event_id is None else int(target_event_id)
                    ),
                    source=str(spec.get("source", "predicted")),
                )
                records.append(record)
                new_birth_mask[batch_index, query_index] = True
                next_id += 1
                step_audit["births"] += 1
                birth_count[batch_index] += 1
            self._next_event_id[video_name] = next_id
            active_count[batch_index] = len(records)

        self.last_audit = step_audit
        return {
            "new_birth_mask": new_birth_mask,
            "ended_mask": ended_mask,
            "emitted_mask": emitted_mask,
            "cancelled_mask": cancelled_mask,
            "active_count": active_count,
            "birth_count": birth_count,
            "end_count": end_count,
            "emit_count": emit_count,
            "cancellation_count": cancellation_count,
            "reacquisition_count": reacquisition_count,
            "runtime_capacity_exhaustions": capacity_exhaustions,
            "padding_prefixes_ignored": padding_ignored,
            "eos_observed": eos_observed,
        }
