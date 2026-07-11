import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "summarize_pes_stage1_run.py"
SPEC = importlib.util.spec_from_file_location("summarize_pes_stage1_run", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _ground_truth():
    return {
        "database": {
            "video": {
                "subset": "validation",
                "annotations": [{"segment": [0.0, 10.0], "label": "Action"}],
            }
        }
    }


def _row(segment, score=0.9, emit=10.2):
    return {
        "segment": segment,
        "label": "Action",
        "score": score,
        "emit_time_sec": emit,
        "source_time_sec": emit,
        "stream_key": "video=video|stream=test",
        "immutable": True,
    }


def test_duplicate_rate_counts_extra_eligible_immutable_emissions():
    predictions = {
        "results": {
            "video": [
                _row([0.0, 10.0], score=0.9),
                _row([0.0, 10.0], score=0.8),
            ]
        }
    }

    report = MODULE.analyze_emission_errors(
        _ground_truth(),
        predictions,
        subset="validation",
        tiou_threshold=0.5,
        latency_budget_sec=1.0,
    )

    assert report["true_positives"] == 1
    assert report["duplicate_false_positives"] == 1
    assert report["duplicate_rate"] == pytest.approx(0.5)


def test_fragmentation_requires_multiple_timely_pieces_with_sufficient_union_coverage():
    predictions = {
        "results": {
            "video": [
                _row([0.0, 4.0], score=0.9),
                _row([4.0, 10.0], score=0.8),
            ]
        }
    }

    report = MODULE.analyze_emission_errors(
        _ground_truth(),
        predictions,
        subset="validation",
        tiou_threshold=0.7,
        latency_budget_sec=1.0,
    )

    assert report["true_positives"] == 0
    assert report["fragmented_ground_truth"] == 1
    assert report["fragmentation_rate"] == pytest.approx(1.0)


def test_future_or_mutable_rows_are_protocol_violations():
    row = _row([0.0, 11.0], emit=10.2)
    row["immutable"] = False

    with pytest.raises(ValueError, match="immutable"):
        MODULE.analyze_emission_errors(
            _ground_truth(),
            {"results": {"video": [row]}},
            subset="validation",
        )
