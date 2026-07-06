import json

import pytest


def test_map_evaluator_filters_ground_truth_and_predictions_to_allowed_videos(tmp_path):
    from opentad.evaluations.mAP import mAP

    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps(
            {
                "database": {
                    "v_keep": {
                        "subset": "validation",
                        "annotations": [{"segment": [0.0, 1.0], "label": "A"}],
                    },
                    "v_drop": {
                        "subset": "validation",
                        "annotations": [{"segment": [0.0, 1.0], "label": "B"}],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    predictions = {
        "results": {
            "v_keep": [{"segment": [0.0, 1.0], "label": "A", "score": 0.9}],
            "v_drop": [{"segment": [0.0, 1.0], "label": "B", "score": 0.99}],
        }
    }

    evaluator = mAP(
        ground_truth_filename=str(gt_path),
        prediction_filename=predictions,
        subset="validation",
        tiou_thresholds=[0.5],
        allowed_videos=["v_keep"],
        thread=1,
    )

    assert set(evaluator.ground_truth["video-id"]) == {"v_keep"}
    assert set(evaluator.prediction["video-id"]) == {"v_keep"}
    assert evaluator.evaluate()["average_mAP"] == pytest.approx(1.0)
