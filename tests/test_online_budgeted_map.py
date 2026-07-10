import json

import pytest


def _ground_truth(tmp_path, annotations):
    path = tmp_path / "gt.json"
    path.write_text(
        json.dumps(
            {
                "database": {
                    "video_1": {
                        "subset": "validation",
                        "annotations": annotations,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _row(segment, score, emit_time_sec, label="Action"):
    return {
        "segment": list(segment),
        "label": label,
        "score": score,
        "emit_time_sec": emit_time_sec,
        "source_time_sec": emit_time_sec,
        "stream_key": "video=video_1|stream=default",
        "immutable": True,
    }


def _evaluator(tmp_path, annotations, rows, budgets=(1.0,), tiou=(0.5,)):
    from opentad.evaluations.online_budgeted_map import OnlineAPBudgeted

    return OnlineAPBudgeted(
        ground_truth_filename=_ground_truth(tmp_path, annotations),
        prediction_filename={"results": {"video_1": rows}},
        subset="validation",
        tiou_thresholds=tiou,
        latency_budgets_sec=budgets,
    )


def test_late_prediction_is_false_positive_and_does_not_delete_ground_truth(tmp_path):
    evaluator = _evaluator(
        tmp_path,
        [{"segment": [0.0, 10.0], "label": "Action"}],
        [
            _row([0.0, 10.0], score=0.9, emit_time_sec=12.0),
            _row([0.0, 10.0], score=0.8, emit_time_sec=10.5),
        ],
    )

    metrics = evaluator.evaluate()

    assert metrics["OnlineAP@1s@tIoU0.5"] == pytest.approx(0.5)
    assert metrics["counts@1s@tIoU0.5"] == {"tp": 1, "fp": 1, "fn": 0}
    assert metrics["num_predictions"] == 2


def test_late_only_prediction_is_fp_while_gt_remains_fn(tmp_path):
    evaluator = _evaluator(
        tmp_path,
        [{"segment": [0.0, 10.0], "label": "Action"}],
        [_row([0.0, 10.0], score=0.9, emit_time_sec=11.01)],
    )

    metrics = evaluator.evaluate()

    assert metrics["OnlineAP@1s@tIoU0.5"] == 0.0
    assert metrics["counts@1s@tIoU0.5"] == {"tp": 0, "fp": 1, "fn": 1}


def test_latency_budget_sweep_changes_eligibility_without_dropping_rows(tmp_path):
    evaluator = _evaluator(
        tmp_path,
        [{"segment": [0.0, 10.0], "label": "Action"}],
        [_row([0.0, 10.0], score=0.9, emit_time_sec=11.5)],
        budgets=(1.0, 2.0),
    )

    metrics = evaluator.evaluate()

    assert metrics["OnlineAP@1s@tIoU0.5"] == 0.0
    assert metrics["OnlineAP@2s@tIoU0.5"] == 1.0
    assert metrics["num_predictions"] == 1


def test_primary_latency_uses_matched_gt_end_not_predicted_end(tmp_path):
    evaluator = _evaluator(
        tmp_path,
        [{"segment": [0.0, 10.0], "label": "Action"}],
        [_row([0.0, 9.0], score=0.9, emit_time_sec=10.5)],
        budgets=(1.0,),
        tiou=(0.8,),
    )

    metrics = evaluator.evaluate()

    assert metrics["matched_gt_latency_sec@1s@tIoU0.8"]["p50"] == pytest.approx(0.5)
    assert metrics["predicted_end_latency_sec"]["p50"] == pytest.approx(1.5)


def test_emission_before_ground_truth_end_cannot_match(tmp_path):
    evaluator = _evaluator(
        tmp_path,
        [{"segment": [0.0, 10.0], "label": "Action"}],
        [_row([0.0, 9.8], score=0.9, emit_time_sec=9.9)],
    )

    metrics = evaluator.evaluate()

    assert metrics["counts@1s@tIoU0.5"] == {"tp": 0, "fp": 1, "fn": 1}


def test_duplicate_timely_prediction_is_false_positive(tmp_path):
    evaluator = _evaluator(
        tmp_path,
        [{"segment": [0.0, 10.0], "label": "Action"}],
        [
            _row([0.0, 10.0], score=0.9, emit_time_sec=10.2),
            _row([0.0, 10.0], score=0.8, emit_time_sec=10.3),
        ],
    )

    metrics = evaluator.evaluate()

    assert metrics["counts@1s@tIoU0.5"] == {"tp": 1, "fp": 1, "fn": 0}


def test_future_read_provenance_is_rejected(tmp_path):
    row = _row([0.0, 10.0], score=0.9, emit_time_sec=10.5)
    row["source_time_sec"] = 10.6
    with pytest.raises(ValueError, match="future source"):
        _evaluator(
            tmp_path,
            [{"segment": [0.0, 10.0], "label": "Action"}],
            [row],
        )


def test_frame_ledger_uses_explicit_evaluator_fps(tmp_path):
    from opentad.evaluations.online_budgeted_map import OnlineAPBudgeted

    row = _row([0.0, 10.0], score=0.9, emit_time_sec=10.5)
    row.pop("emit_time_sec")
    row.pop("source_time_sec")
    row.update(emit_frame=315, source_frame=315)
    evaluator = OnlineAPBudgeted(
        ground_truth_filename=_ground_truth(
            tmp_path,
            [{"segment": [0.0, 10.0], "label": "Action"}],
        ),
        prediction_filename={"results": {"video_1": [row]}},
        subset="validation",
        tiou_thresholds=(0.5,),
        latency_budgets_sec=(1.0,),
        fps=30,
    )

    assert evaluator.evaluate()["OnlineAP@1s@tIoU0.5"] == 1.0
