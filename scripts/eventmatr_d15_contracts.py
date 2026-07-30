"""Pure deterministic contracts shared by the EventMATR D1.5 scan and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TargetView:
    event_id: int
    class_id: int
    start_frame: float
    end_frame: float
    admission_frame: float
    is_birth: bool
    is_alive: bool
    is_end: bool


def query_compatibility_scores(
    targets: Iterable[TargetView],
    *,
    class_logits: torch.Tensor,
    birth_logits: torch.Tensor,
    candidate_start_frames: torch.Tensor,
    segment_size: int,
    include_birth_evidence: bool,
) -> dict[int, torch.Tensor]:
    if class_logits.ndim != 2 or birth_logits.ndim != 1:
        raise ValueError("query compatibility expects [Q,C] classes and [Q] births")
    if candidate_start_frames.shape != birth_logits.shape:
        raise ValueError("candidate start and birth shapes differ")
    class_log_probs = class_logits.log_softmax(dim=-1)
    scores = {}
    for target in targets:
        if target.class_id < 0 or target.class_id >= class_logits.size(-1) - 1:
            raise RuntimeError("target foreground class is outside model vocabulary")
        score = class_log_probs[:, target.class_id] - (
            candidate_start_frames - float(target.start_frame)
        ).abs() / float(segment_size)
        if include_birth_evidence and target.is_birth:
            score = score + F.logsigmoid(birth_logits)
        scores[target.event_id] = score.detach()
    return scores


def deterministic_target_query_assignment(
    targets: Iterable[TargetView],
    *,
    class_logits: torch.Tensor,
    birth_logits: torch.Tensor,
    candidate_start_frames: torch.Tensor,
    segment_size: int,
    include_birth_evidence: bool,
) -> tuple[dict[int, int], int]:
    """Frozen one-to-one target/query assignment with index-based tie breaking."""

    targets = tuple(sorted(targets, key=lambda target: target.event_id))
    if len(targets) > int(class_logits.size(0)):
        raise RuntimeError("visible targets exceed the frozen query bandwidth")
    scores = query_compatibility_scores(
        targets,
        class_logits=class_logits,
        birth_logits=birth_logits,
        candidate_start_frames=candidate_start_frames,
        segment_size=segment_size,
        include_birth_evidence=include_birth_evidence,
    )
    remaining_targets = {target.event_id for target in targets}
    remaining_queries = set(range(int(class_logits.size(0))))
    assignments: dict[int, int] = {}
    exact_ties = 0
    while remaining_targets:
        candidates = []
        for target_id in sorted(remaining_targets):
            for query_index in sorted(remaining_queries):
                candidates.append(
                    (
                        float(scores[target_id][query_index].item()),
                        int(target_id),
                        int(query_index),
                    )
                )
        best_score = max(candidate[0] for candidate in candidates)
        tied = [
            candidate for candidate in candidates if candidate[0] == best_score
        ]
        if len(tied) > 1:
            exact_ties += 1
        _, target_id, query_index = min(
            tied, key=lambda candidate: (candidate[1], candidate[2])
        )
        assignments[target_id] = query_index
        remaining_targets.remove(target_id)
        remaining_queries.remove(query_index)
    return assignments, exact_ties
