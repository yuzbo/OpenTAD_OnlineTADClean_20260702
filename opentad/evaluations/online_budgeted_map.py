"""Budget-aware online temporal action detection evaluation.

Unlike a latency pre-filter followed by offline mAP, this evaluator keeps every
immutable emission in the ranked prediction list. A detection is a true
positive only when class, temporal IoU, and the ground-truth-end latency budget
all match. Late detections therefore remain false positives and cannot erase
the corresponding false negative.
"""

import json
from collections import defaultdict

import numpy as np

from .builder import EVALUATORS


def _number_key(value):
    return f"{float(value):g}"


def _segment_iou(prediction, ground_truth):
    intersection = max(
        0.0,
        min(float(prediction[1]), float(ground_truth[1]))
        - max(float(prediction[0]), float(ground_truth[0])),
    )
    union = max(float(prediction[1]), float(ground_truth[1])) - min(
        float(prediction[0]), float(ground_truth[0])
    )
    return intersection / union if union > 0 else 0.0


def _interpolated_ap(tp, fp, num_ground_truth):
    if num_ground_truth <= 0:
        return None
    tp = np.cumsum(np.asarray(tp, dtype=np.float64))
    fp = np.cumsum(np.asarray(fp, dtype=np.float64))
    recall = tp / float(num_ground_truth)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def _summary(values):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


@EVALUATORS.register_module()
class OnlineAPBudgeted:
    """Compute OnlineAP@B without deleting early, late, or duplicate rows."""

    def __init__(
        self,
        ground_truth_filename,
        prediction_filename,
        subset,
        tiou_thresholds,
        latency_budgets_sec=(0.5, 1.0, 2.0, 4.0),
        blocked_videos=None,
        allowed_videos=None,
        fps=None,
        require_ledger=True,
        require_no_future=True,
        **kwargs,
    ):
        del kwargs
        self.subset = subset
        self.tiou_thresholds = tuple(float(value) for value in tiou_thresholds)
        self.latency_budgets_sec = tuple(float(value) for value in latency_budgets_sec)
        if not self.tiou_thresholds:
            raise ValueError("tiou_thresholds must not be empty")
        if not self.latency_budgets_sec or any(value < 0 for value in self.latency_budgets_sec):
            raise ValueError("latency_budgets_sec must contain non-negative budgets")
        self.fps = None if fps is None else float(fps)
        self.require_ledger = bool(require_ledger)
        self.require_no_future = bool(require_no_future)
        self.blocked_videos = self._load_video_set(blocked_videos) or set()
        self.allowed_videos = self._load_video_set(allowed_videos)
        self.ground_truth = self._load_ground_truth(ground_truth_filename)
        self.predictions = self._load_predictions(prediction_filename)
        self.metrics = None

    @staticmethod
    def _load_video_set(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            return {str(item) for item in value}
        with open(value, "r", encoding="utf-8") as file:
            if str(value).lower().endswith(".json"):
                loaded = json.load(file)
                return {str(item) for item in loaded}
            return {line.strip() for line in file if line.strip()}

    @staticmethod
    def _load_json_or_dict(value):
        if isinstance(value, dict):
            return value
        with open(value, "r", encoding="utf-8") as file:
            return json.load(file)

    def _video_allowed(self, video_id):
        return video_id not in self.blocked_videos and (
            self.allowed_videos is None or video_id in self.allowed_videos
        )
    def _load_ground_truth(self, filename):
        data = self._load_json_or_dict(filename)
        if "database" not in data:
            raise IOError("ground truth must contain a database field")
        rows = []
        seen = set()
        for video_id, video in data["database"].items():
            if video.get("subset") != self.subset or not self._video_allowed(video_id):
                continue
            for annotation in video.get("annotations", []):
                segment = tuple(float(value) for value in annotation["segment"])
                if len(segment) != 2 or segment[1] <= segment[0]:
                    raise ValueError(f"invalid ground-truth segment for {video_id}: {segment}")
                key = (video_id, str(annotation["label"]), segment)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "id": len(rows),
                        "video_id": str(video_id),
                        "label": str(annotation["label"]),
                        "segment": segment,
                    }
                )
        return rows

    def _time_from_row(self, row, prefix):
        for key in (f"{prefix}_time_sec", f"{prefix}_sec"):
            if key in row:
                return float(row[key])
        frame_key = f"{prefix}_frame"
        if frame_key in row:
            fps = float(row.get("fps", self.fps or 0.0))
            if fps <= 0:
                raise ValueError(f"{frame_key} requires a positive row or evaluator fps")
            return float(row[frame_key]) / fps
        return None

    def _validate_prediction(self, video_id, row, index):
        required = {"segment", "label", "score"}
        if self.require_ledger:
            required.update({"stream_key", "immutable"})
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"prediction row is missing fields {missing}")
        if self.require_ledger and not bool(row["immutable"]):
            raise ValueError("online prediction rows must be immutable emissions")
        segment = tuple(float(value) for value in row["segment"])
        if len(segment) != 2 or segment[1] <= segment[0]:
            raise ValueError(f"invalid prediction segment for {video_id}: {segment}")
        emit_time = self._time_from_row(row, "emit")
        source_time = self._time_from_row(row, "source")
        if emit_time is None:
            raise ValueError("online prediction row requires emit_time_sec or emit_frame")
        if self.require_ledger and source_time is None:
            raise ValueError("online prediction row requires source_time_sec or source_frame")
        if self.require_no_future and segment[1] > emit_time + 1e-9:
            raise ValueError(
                f"future predicted end for {video_id}: end={segment[1]} emit={emit_time}"
            )
        if self.require_no_future and source_time is not None and source_time > emit_time + 1e-9:
            raise ValueError(
                f"future source read for {video_id}: source={source_time} emit={emit_time}"
            )
        return {
            "id": index,
            "video_id": str(video_id),
            "label": str(row["label"]),
            "score": float(row["score"]),
            "segment": segment,
            "emit_time_sec": emit_time,
            "source_time_sec": source_time,
        }

    def _load_predictions(self, filename):
        data = self._load_json_or_dict(filename)
        if "results" not in data:
            raise IOError("predictions must contain a results field")
        rows = []
        for video_id, video_rows in data["results"].items():
            video_id = str(video_id)
            if not self._video_allowed(video_id):
                continue
            for row in video_rows:
                rows.append(self._validate_prediction(video_id, row, len(rows)))
        return rows

    def _evaluate_setting(self, budget, tiou_threshold):
        ground_truth_by_label = defaultdict(list)
        predictions_by_label = defaultdict(list)
        for row in self.ground_truth:
            ground_truth_by_label[row["label"]].append(row)
        for row in self.predictions:
            predictions_by_label[row["label"]].append(row)

        class_ap = {}
        total_tp = 0
        total_fp = 0
        matched_latencies = []
        labels = sorted(set(ground_truth_by_label) | set(predictions_by_label))
        for label in labels:
            label_ground_truth = ground_truth_by_label[label]
            predictions = sorted(
                predictions_by_label[label],
                key=lambda row: (-row["score"], row["id"]),
            )
            locked = set()
            tp = np.zeros(len(predictions), dtype=np.float64)
            fp = np.zeros(len(predictions), dtype=np.float64)
            for prediction_index, prediction in enumerate(predictions):
                candidates = []
                for target in label_ground_truth:
                    if target["id"] in locked or target["video_id"] != prediction["video_id"]:
                        continue
                    iou = _segment_iou(prediction["segment"], target["segment"])
                    latency = prediction["emit_time_sec"] - target["segment"][1]
                    if iou >= tiou_threshold and -1e-9 <= latency <= budget + 1e-9:
                        candidates.append((iou, -abs(latency), target, max(0.0, latency)))
                if not candidates:
                    fp[prediction_index] = 1.0
                    continue
                _, _, target, latency = max(candidates, key=lambda item: (item[0], item[1]))
                locked.add(target["id"])
                tp[prediction_index] = 1.0
                matched_latencies.append(latency)
            ap = _interpolated_ap(tp, fp, len(label_ground_truth))
            if ap is not None:
                class_ap[label] = ap
            total_tp += int(tp.sum())
            total_fp += int(fp.sum())

        return {
            "online_ap": float(np.mean(list(class_ap.values()))) if class_ap else 0.0,
            "class_ap": class_ap,
            "counts": {
                "tp": total_tp,
                "fp": total_fp,
                "fn": len(self.ground_truth) - total_tp,
            },
            "matched_gt_latency_sec": _summary(matched_latencies),
        }

    def evaluate(self):
        predicted_end_latencies = [
            row["emit_time_sec"] - row["segment"][1] for row in self.predictions
        ]
        metrics = {
            "metric": "OnlineAPBudgeted",
            "primary_latency_definition": "emit_time_sec - matched_ground_truth_end_sec",
            "num_ground_truth": len(self.ground_truth),
            "num_predictions": len(self.predictions),
            "latency_budgets_sec": list(self.latency_budgets_sec),
            "tiou_thresholds": list(self.tiou_thresholds),
            "predicted_end_latency_sec": _summary(predicted_end_latencies),
        }
        all_scores = []
        for budget in self.latency_budgets_sec:
            budget_scores = []
            for tiou_threshold in self.tiou_thresholds:
                result = self._evaluate_setting(budget, tiou_threshold)
                suffix = f"{_number_key(budget)}s@tIoU{_number_key(tiou_threshold)}"
                metrics[f"OnlineAP@{suffix}"] = result["online_ap"]
                metrics[f"class_AP@{suffix}"] = result["class_ap"]
                metrics[f"counts@{suffix}"] = result["counts"]
                metrics[f"matched_gt_latency_sec@{suffix}"] = result[
                    "matched_gt_latency_sec"
                ]
                budget_scores.append(result["online_ap"])
                all_scores.append(result["online_ap"])
            metrics[f"mOnlineAP@{_number_key(budget)}s"] = float(np.mean(budget_scores))
        metrics["average_mOnlineAP"] = float(np.mean(all_scores)) if all_scores else 0.0
        self.metrics = metrics
        return metrics

    def logging(self, logger=None):
        if self.metrics is None:
            self.evaluate()
        pprint = print if logger is None else logger.info
        pprint(
            f"OnlineAPBudgeted: {len(self.ground_truth)} GT, "
            f"{len(self.predictions)} immutable emissions"
        )
        for budget in self.latency_budgets_sec:
            key = f"mOnlineAP@{_number_key(budget)}s"
            pprint(f"{key}: {self.metrics[key] * 100:.2f}%")
        pprint(
            "Primary latency (matched GT end): "
            + str(
                self.metrics[
                    "matched_gt_latency_sec@"
                    f"{_number_key(self.latency_budgets_sec[0])}s@"
                    f"tIoU{_number_key(self.tiou_thresholds[0])}"
                ]
            )
        )
