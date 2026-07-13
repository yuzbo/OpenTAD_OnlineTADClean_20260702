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
        "video_id": "video_1",
        "immutable": True,
    }


def _evaluator(
    tmp_path,
    annotations,
    rows,
    budgets=(1.0,),
    tiou=(0.5,),
    **overrides,
):
    from opentad.evaluations.online_budgeted_map import OnlineAPBudgeted

    arguments = dict(
        ground_truth_filename=_ground_truth(tmp_path, annotations),
        prediction_filename={"results": {"video_1": rows}},
        subset="validation",
        tiou_thresholds=tiou,
        latency_budgets_sec=budgets,
        fps=30,
        require_ledger=False,
    )
    arguments.update(overrides)
    return OnlineAPBudgeted(**arguments)


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
        require_ledger=False,
    )

    assert evaluator.evaluate()["OnlineAP@1s@tIoU0.5"] == 1.0


def test_distinct_identical_ground_truth_instances_are_not_deduplicated(tmp_path):
    evaluator = _evaluator(
        tmp_path,
        [
            {"instance_id": "first", "segment": [0.0, 10.0], "label": "Action"},
            {"instance_id": "second", "segment": [0.0, 10.0], "label": "Action"},
        ],
        [
            _row([0.0, 10.0], score=0.9, emit_time_sec=10.1),
            _row([0.0, 10.0], score=0.8, emit_time_sec=10.2),
        ],
    )

    metrics = evaluator.evaluate()

    assert metrics["num_ground_truth"] == 2
    assert metrics["counts@1s@tIoU0.5"] == {"tp": 2, "fp": 0, "fn": 0}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score", float("nan")),
        ("score", float("inf")),
        ("emit_time_sec", float("nan")),
        ("source_time_sec", float("inf")),
    ],
)
def test_nonfinite_prediction_values_are_rejected(tmp_path, field, value):
    row = _row([0.0, 10.0], score=0.9, emit_time_sec=10.5)
    row[field] = value

    with pytest.raises(ValueError, match="finite"):
        _evaluator(
            tmp_path,
            [{"segment": [0.0, 10.0], "label": "Action"}],
            [row],
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tiou_thresholds": (float("nan"),)}, "tiou_thresholds"),
        ({"latency_budgets_sec": (float("inf"),)}, "latency_budgets_sec"),
        ({"fps": float("nan")}, "fps"),
    ],
)
def test_nonfinite_protocol_parameters_are_rejected(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        _evaluator(
            tmp_path,
            [{"segment": [0.0, 10.0], "label": "Action"}],
            [_row([0.0, 10.0], score=0.9, emit_time_sec=10.5)],
            **overrides,
        )


def test_formal_evaluator_requires_a_hash_verified_ledger(tmp_path):
    with pytest.raises(ValueError, match="envelope"):
        _evaluator(
            tmp_path,
            [{"segment": [0.0, 10.0], "label": "Action"}],
            [_row([0.0, 10.0], score=0.9, emit_time_sec=10.5)],
            require_ledger=True,
        )


def test_production_evaluator_reports_distinct_standard_online_and_identity_metrics(
    tmp_path,
):
    import hashlib

    from opentad.utils.immutable_event_ledger import ImmutableEventLedger

    ledger = ImmutableEventLedger()
    rows = []
    for index, (score, emit_frame) in enumerate(((0.9, 306), (0.8, 309))):
        rows.append(
            ledger.append(
                {
                    "event_id": f"event-{index}",
                    "stream_id": "stream:video_1",
                    "stream_key": "stream:video_1",
                    "video_id": "video_1",
                    "immutable": True,
                    "slot_id": index,
                    "label": "Action",
                    "score": score,
                    "start_frame": 0,
                    "end_frame": 300,
                    "emit_frame": emit_frame,
                    "source_frame": emit_frame,
                    "provenance_digest": hashlib.sha256(
                        f"event-{index}".encode("utf-8")
                    ).hexdigest(),
                    "segment": [0.0, 10.0],
                }
            )
        )

    evaluator = _evaluator(
        tmp_path,
        [{"instance_id": "gt", "segment": [0.0, 10.0], "label": "Action"}],
        rows,
        budgets=(1.0,),
        tiou=(0.5,),
        require_ledger=True,
    )

    metrics = evaluator.evaluate()

    assert metrics["average_mAP"] == pytest.approx(1.0)
    assert metrics["average_mOnlineAP"] == pytest.approx(1.0)
    assert metrics["duplicate_per_gt"] == pytest.approx(1.0)
    assert metrics["duplicate_fraction"] == pytest.approx(0.5)
    assert metrics["fragmentation_rate"] == pytest.approx(0.0)
    assert metrics["identity_diagnostics"]["matching"]["latency_budget_sec"] == 2.0


def test_per_video_annotation_fps_controls_frame_latency_and_rejects_drift(tmp_path):
    from opentad.evaluations.online_budgeted_map import OnlineAPBudgeted

    ground_truth = tmp_path / "variable-fps-gt.json"
    ground_truth.write_text(
        json.dumps(
            {
                "database": {
                    "video_1": {
                        "subset": "validation",
                        "frame": 250,
                        "duration": 10.0,
                        "annotations": [
                            {"segment": [0.0, 10.0], "label": "Action"}
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    row = _row([0.0, 10.0], score=0.9, emit_time_sec=10.2)
    row.pop("emit_time_sec")
    row.pop("source_time_sec")
    row.update(
        start_frame=0,
        end_frame=250,
        emit_frame=255,
        source_frame=255,
        fps=25.0,
    )

    evaluator = OnlineAPBudgeted(
        ground_truth_filename=str(ground_truth),
        prediction_filename={"results": {"video_1": [row]}},
        subset="validation",
        tiou_thresholds=(0.5,),
        latency_budgets_sec=(0.5,),
        fps=30.0,
        require_ledger=False,
    )
    assert evaluator.evaluate()["OnlineAP@0.5s@tIoU0.5"] == pytest.approx(1.0)

    drifted = dict(row, fps=30.0)
    with pytest.raises(ValueError, match="ground-truth fps"):
        OnlineAPBudgeted(
            ground_truth_filename=str(ground_truth),
            prediction_filename={"results": {"video_1": [drifted]}},
            subset="validation",
            tiou_thresholds=(0.5,),
            latency_budgets_sec=(0.5,),
            require_ledger=False,
        )
