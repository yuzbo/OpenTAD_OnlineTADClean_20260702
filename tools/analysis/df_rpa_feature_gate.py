import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CandidateRecord:
    video_id: str
    time: float
    candidate_start: float
    candidate_end: float
    scores: dict[str, float]
    label_useful: int


@dataclass(frozen=True)
class ScoreSummary:
    precision_at_k: float
    mean_positive_score: float
    num_candidates: int
    num_positive: int


def load_candidate_records(path: str | Path) -> list[CandidateRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(_parse_record(json.loads(line)))
    return records


def precision_at_k(records: Iterable[CandidateRecord], score_field: str, k: int) -> float:
    checked = _materialize_records(records, score_field)
    if k <= 0:
        raise ValueError("k must be positive")
    if not checked:
        return 0.0

    top_k = sorted(checked, key=lambda record: record.scores[score_field], reverse=True)[:k]
    return sum(1 for record in top_k if record.label_useful) / len(top_k)


def summarize_score_fields(
    records: Iterable[CandidateRecord],
    score_fields: Iterable[str],
    k: int,
) -> dict[str, ScoreSummary]:
    checked = list(records)
    return {
        score_field: ScoreSummary(
            precision_at_k=precision_at_k(checked, score_field, k),
            mean_positive_score=_mean_positive_score(checked, score_field),
            num_candidates=len(checked),
            num_positive=sum(1 for record in checked if record.label_useful),
        )
        for score_field in score_fields
    }


def _parse_record(raw: dict) -> CandidateRecord:
    scores = {
        key.removeprefix("score_"): float(value)
        for key, value in raw.items()
        if key.startswith("score_")
    }
    return CandidateRecord(
        video_id=str(raw["video_id"]),
        time=float(raw["time"]),
        candidate_start=float(raw["candidate_start"]),
        candidate_end=float(raw["candidate_end"]),
        scores=scores,
        label_useful=int(raw["label_useful"]),
    )


def _materialize_records(
    records: Iterable[CandidateRecord],
    score_field: str,
) -> list[CandidateRecord]:
    checked = list(records)
    for record in checked:
        if score_field not in record.scores:
            raise KeyError(f"missing score field: {score_field}")
    return checked


def _mean_positive_score(records: Iterable[CandidateRecord], score_field: str) -> float:
    checked = _materialize_records(records, score_field)
    positive_scores = [record.scores[score_field] for record in checked if record.label_useful]
    if not positive_scores:
        return 0.0
    return sum(positive_scores) / len(positive_scores)
