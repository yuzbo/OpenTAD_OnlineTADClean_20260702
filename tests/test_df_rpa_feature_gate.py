from pathlib import Path

import pytest

from tools.analysis.df_rpa_feature_gate import (
    CandidateRecord,
    load_candidate_records,
    precision_at_k,
    summarize_score_fields,
)


def test_precision_at_k_uses_same_candidate_records_for_all_scores():
    records = [
        CandidateRecord("v1", 10.0, 7.0, 8.0, {"feedback": 0.9, "boundary": 0.2}, 1),
        CandidateRecord("v1", 10.0, 6.0, 7.0, {"feedback": 0.8, "boundary": 0.1}, 1),
        CandidateRecord("v1", 10.0, 8.0, 9.0, {"feedback": 0.1, "boundary": 0.95}, 0),
        CandidateRecord("v1", 10.0, 5.0, 6.0, {"feedback": 0.2, "boundary": 0.85}, 0),
    ]

    summary = summarize_score_fields(records, ["feedback", "boundary"], k=2)

    assert summary["feedback"].precision_at_k == pytest.approx(1.0)
    assert summary["boundary"].precision_at_k == pytest.approx(0.0)
    assert summary["feedback"].num_candidates == 4
    assert summary["boundary"].num_candidates == 4


def test_load_candidate_records_from_jsonl(tmp_path: Path):
    jsonl_path = tmp_path / "candidates.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                '{"video_id":"v1","time":10.0,"candidate_start":7.0,'
                '"candidate_end":8.0,"score_feedback":0.9,'
                '"score_uniform":0.2,"label_useful":1}',
                '{"video_id":"v1","time":10.0,"candidate_start":8.0,'
                '"candidate_end":9.0,"score_feedback":0.1,'
                '"score_uniform":0.8,"label_useful":0}',
            ]
        ),
        encoding="utf-8",
    )

    records = load_candidate_records(jsonl_path)

    assert records == [
        CandidateRecord("v1", 10.0, 7.0, 8.0, {"feedback": 0.9, "uniform": 0.2}, 1),
        CandidateRecord("v1", 10.0, 8.0, 9.0, {"feedback": 0.1, "uniform": 0.8}, 0),
    ]


def test_precision_at_k_rejects_missing_score_field():
    records = [
        CandidateRecord("v1", 10.0, 7.0, 8.0, {"feedback": 0.9}, 1),
    ]

    with pytest.raises(KeyError, match="missing score field"):
        precision_at_k(records, "boundary", k=1)
