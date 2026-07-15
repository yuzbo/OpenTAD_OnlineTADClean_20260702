"""Budget-aware online temporal action detection evaluation.

Unlike a latency pre-filter followed by offline mAP, this evaluator keeps every
immutable emission in the ranked prediction list. A detection is a true
positive only when class, temporal IoU, and the ground-truth-end latency budget
all match. Late detections therefore remain false positives and cannot erase
the corresponding false negative.
"""

import json
import math
from collections import defaultdict

import numpy as np

from .builder import EVALUATORS
from .full_petal_metrics import compute_full_petal_metrics
from opentad.utils.online_protocol import (
    verified_emission_result_dict,
    verified_emission_result_dict_bytes,
)


def _number_key(value):
    return f"{float(value):g}"


def _finite_number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


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
        include_identity_diagnostics=None,
        identity_tiou_threshold=0.5,
        identity_latency_budget_sec=2.0,
        **kwargs,
    ):
        del kwargs
        self.subset = subset
        self.tiou_thresholds = tuple(
            _finite_number(value, "tiou_thresholds") for value in tiou_thresholds
        )
        self.latency_budgets_sec = tuple(
            _finite_number(value, "latency_budgets_sec")
            for value in latency_budgets_sec
        )
        if not self.tiou_thresholds or any(
            value <= 0 or value > 1 for value in self.tiou_thresholds
        ):
            raise ValueError("tiou_thresholds must contain values in (0, 1]")
        if not self.latency_budgets_sec or any(
            value < 0 for value in self.latency_budgets_sec
        ):
            raise ValueError("latency_budgets_sec must contain finite non-negative budgets")
        self.fps = None if fps is None else _finite_number(fps, "fps")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be a finite positive number")
        self.require_ledger = bool(require_ledger)
        self.require_no_future = bool(require_no_future)
        self.include_identity_diagnostics = (
            self.require_ledger
            if include_identity_diagnostics is None
            else bool(include_identity_diagnostics)
        )
        self.identity_tiou_threshold = _finite_number(
            identity_tiou_threshold,
            "identity_tiou_threshold",
        )
        if not 0 < self.identity_tiou_threshold <= 1:
            raise ValueError("identity_tiou_threshold must lie in (0, 1]")
        self.identity_latency_budget_sec = _finite_number(
            identity_latency_budget_sec,
            "identity_latency_budget_sec",
        )
        if self.identity_latency_budget_sec < 0:
            raise ValueError("identity_latency_budget_sec must be non-negative")
        self.blocked_videos = self._load_video_set(blocked_videos) or set()
        self.allowed_videos = self._load_video_set(allowed_videos)
        self._video_fps_by_id = {}
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
        seen_ids = set()
        for video_id, video in data["database"].items():
            if video.get("subset") != self.subset or not self._video_allowed(video_id):
                continue
            video_fps = self.fps
            has_frame = "frame" in video
            has_duration = "duration" in video
            if has_frame != has_duration:
                raise ValueError(
                    f"ground-truth video {video_id} must provide frame and duration together"
                )
            if has_frame:
                frame_count = _finite_number(video["frame"], f"{video_id}.frame")
                duration = _finite_number(video["duration"], f"{video_id}.duration")
                if frame_count <= 0 or duration <= 0:
                    raise ValueError(
                        f"ground-truth video {video_id} frame and duration must be positive"
                    )
                video_fps = frame_count / duration
            self._video_fps_by_id[str(video_id)] = video_fps
            for annotation_index, annotation in enumerate(video.get("annotations", [])):
                raw_segment = annotation.get("segment")
                if not isinstance(raw_segment, (list, tuple)) or len(raw_segment) != 2:
                    raise ValueError(
                        f"invalid ground-truth segment for {video_id}: {raw_segment}"
                    )
                segment = tuple(
                    _finite_number(value, f"ground-truth segment for {video_id}")
                    for value in raw_segment
                )
                if segment[1] <= segment[0]:
                    raise ValueError(f"invalid ground-truth segment for {video_id}: {segment}")
                instance_id = annotation.get(
                    "instance_id",
                    annotation.get("id", f"annotation-{annotation_index}"),
                )
                instance_key = (str(video_id), type(instance_id).__name__, repr(instance_id))
                if instance_key in seen_ids:
                    raise ValueError(
                        f"duplicate ground-truth instance ID {instance_id!r} for {video_id}"
                    )
                seen_ids.add(instance_key)
                rows.append(
                    {
                        "id": (
                            f"{video_id}:annotation:{type(instance_id).__name__}:"
                            f"{instance_id!r}"
                        ),
                        "video_id": str(video_id),
                        "label": str(annotation["label"]),
                        "segment": segment,
                        "fps": video_fps,
                    }
                )
        return rows

    def _row_fps(self, row, video_id):
        row_fps = row.get("fps")
        if row_fps is not None:
            row_fps = _finite_number(row_fps, "prediction row fps")
            if row_fps <= 0:
                raise ValueError("prediction row fps must be finite and positive")
        expected_fps = self._video_fps_by_id.get(str(video_id))
        resolved_fps = row_fps if row_fps is not None else expected_fps
        if resolved_fps is None:
            resolved_fps = self.fps
        if row_fps is not None and expected_fps is not None and not math.isclose(
            row_fps,
            expected_fps,
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"prediction row fps conflicts with ground-truth fps for {video_id}"
            )
        return resolved_fps

    def _time_from_row(self, row, prefix, video_id):
        explicit = []
        for key in (f"{prefix}_time_sec", f"{prefix}_sec"):
            if key in row:
                explicit.append(_finite_number(row[key], key))
        if len(explicit) == 2 and not math.isclose(
            explicit[0], explicit[1], rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"conflicting {prefix} time aliases")
        explicit_time = explicit[0] if explicit else None
        frame_key = f"{prefix}_frame"
        if frame_key in row:
            frame = _finite_number(row[frame_key], frame_key)
            fps = self._row_fps(row, video_id)
            if fps is None:
                raise ValueError(f"{frame_key} requires a positive row or evaluator fps")
            frame_time = frame / fps
            if explicit_time is not None and not math.isclose(
                explicit_time,
                frame_time,
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                raise ValueError(f"{prefix} frame and time coordinates conflict")
            return frame_time
        return explicit_time

    def _frame_from_row(self, row, prefix, time_value, fps):
        key = f"{prefix}_frame"
        if key in row:
            return _finite_number(row[key], key)
        if time_value is None or fps is None:
            return None
        return time_value * fps

    def _validate_prediction(self, video_id, row, index):
        required = {"segment", "label", "score"}
        if self.require_ledger:
            required.update({"stream_key", "immutable"})
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"prediction row is missing fields {missing}")
        if self.require_ledger and not bool(row["immutable"]):
            raise ValueError("online prediction rows must be immutable emissions")
        raw_segment = row["segment"]
        if not isinstance(raw_segment, (list, tuple)) or len(raw_segment) != 2:
            raise ValueError(f"invalid prediction segment for {video_id}: {raw_segment}")
        segment = tuple(
            _finite_number(value, f"prediction segment for {video_id}")
            for value in raw_segment
        )
        if segment[1] <= segment[0]:
            raise ValueError(f"invalid prediction segment for {video_id}: {segment}")
        score = _finite_number(row["score"], "prediction score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("prediction score must be finite and lie in [0, 1]")
        row_fps = self._row_fps(row, video_id)
        emit_time = self._time_from_row(row, "emit", video_id)
        source_time = self._time_from_row(row, "source", video_id)
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
        start_frame = self._frame_from_row(row, "start", segment[0], row_fps)
        end_frame = self._frame_from_row(row, "end", segment[1], row_fps)
        emit_frame = self._frame_from_row(row, "emit", emit_time, row_fps)
        source_frame = self._frame_from_row(row, "source", source_time, row_fps)
        if row_fps is not None and start_frame is not None and end_frame is not None:
            expected_segment = (start_frame / row_fps, end_frame / row_fps)
            if any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-7)
                for actual, expected in zip(segment, expected_segment)
            ):
                raise ValueError("prediction frame and segment coordinates conflict")
        return {
            "id": index,
            "event_id": row.get("event_id", f"prediction-{index}"),
            "video_id": str(video_id),
            "stream_key": str(row.get("stream_key", video_id)),
            "label": str(row["label"]),
            "score": score,
            "segment": segment,
            "emit_time_sec": emit_time,
            "source_time_sec": source_time,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "emit_frame": emit_frame,
            "source_frame": source_frame,
            "sequence": row.get("sequence"),
            "immutable": row.get("immutable") is True,
            "fps": row_fps,
        }

    def _load_predictions(self, filename):
        data = self._load_json_or_dict(filename)
        if self.require_ledger:
            path_fields = {"ledger_path", "commitment_path"}
            byte_fields = {"ledger_bytes", "commitment_bytes", "ledger_filename"}
            if set(data) == path_fields:
                result_mapping = verified_emission_result_dict(
                    data["ledger_path"], data["commitment_path"]
                )
            elif set(data) == byte_fields:
                result_mapping = verified_emission_result_dict_bytes(
                    data["ledger_bytes"],
                    data["commitment_bytes"],
                    ledger_filename=data["ledger_filename"],
                )
            else:
                raise ValueError(
                    "formal predictions require ledger_path/commitment_path or "
                    "ledger_bytes/commitment_bytes/ledger_filename"
                )
        else:
            if "results" not in data:
                raise IOError("predictions must contain a results field")
            result_mapping = data["results"]
        rows = []
        for video_id, video_rows in result_mapping.items():
            video_id = str(video_id)
            if not self._video_allowed(video_id):
                continue
            for row in video_rows:
                rows.append(self._validate_prediction(video_id, row, len(rows)))
        return rows

    def _evaluate_standard_setting(self, tiou_threshold):
        ground_truth_by_label = defaultdict(list)
        predictions_by_label = defaultdict(list)
        for row in self.ground_truth:
            ground_truth_by_label[row["label"]].append(row)
        for row in self.predictions:
            predictions_by_label[row["label"]].append(row)

        class_ap = {}
        total_tp = 0
        total_fp = 0
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
                    candidates.append(
                        (_segment_iou(prediction["segment"], target["segment"]), target)
                    )
                if not candidates:
                    fp[prediction_index] = 1.0
                    continue
                tiou, target = max(candidates, key=lambda item: item[0])
                if tiou + 1e-9 < tiou_threshold:
                    fp[prediction_index] = 1.0
                    continue
                locked.add(target["id"])
                tp[prediction_index] = 1.0
            ap = _interpolated_ap(tp, fp, len(label_ground_truth))
            if ap is not None:
                class_ap[label] = ap
            total_tp += int(tp.sum())
            total_fp += int(fp.sum())
        return {
            "ap": float(np.mean(list(class_ap.values()))) if class_ap else 0.0,
            "class_ap": class_ap,
            "counts": {
                "tp": total_tp,
                "fp": total_fp,
                "fn": len(self.ground_truth) - total_tp,
            },
        }

    def _compute_identity_diagnostics(self):
        if any(row["fps"] is None for row in self.ground_truth):
            raise ValueError(
                "identity diagnostics require annotation frame/duration or evaluator fps"
            )
        if any(row["fps"] is None for row in self.predictions):
            raise ValueError("identity diagnostics require a resolved fps for every emission")
        ground_truth = [
            {
                "gt_id": row["id"],
                "stream_key": row["video_id"],
                "label": row["label"],
                "start_frame": row["segment"][0] * row["fps"],
                "end_frame": row["segment"][1] * row["fps"],
            }
            for row in self.ground_truth
        ]
        emissions = [
            {
                "emission_id": row["event_id"],
                "stream_key": row["video_id"],
                "label": row["label"],
                "score": row["score"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "emit_frame": row["emit_frame"],
                "source_frame": row["source_frame"],
                "sequence_id": row["sequence"],
                "immutable": row["immutable"],
                "fps": row["fps"],
            }
            for row in self.predictions
        ]
        return compute_full_petal_metrics(
            ground_truth,
            emissions,
            tiou_threshold=self.identity_tiou_threshold,
            fps=self.fps or 1.0,
            latency_budget_sec=self.identity_latency_budget_sec,
        )

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
        standard_scores = []
        for tiou_threshold in self.tiou_thresholds:
            result = self._evaluate_standard_setting(tiou_threshold)
            suffix = _number_key(tiou_threshold)
            metrics[f"mAP@{suffix}"] = result["ap"]
            metrics[f"class_AP@tIoU{suffix}"] = result["class_ap"]
            metrics[f"standard_counts@tIoU{suffix}"] = result["counts"]
            standard_scores.append(result["ap"])
        metrics["average_mAP"] = (
            float(np.mean(standard_scores)) if standard_scores else 0.0
        )
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
        if self.include_identity_diagnostics:
            identity = self._compute_identity_diagnostics()
            metrics["identity_diagnostics"] = identity
            metrics["identity_recall"] = identity["recall"]
            metrics["duplicate_per_gt"] = identity["duplicate_per_gt"]
            metrics["duplicate_fraction"] = identity["duplicate_fraction"]
            metrics["duplicate_gt_rate"] = identity["duplicate_gt_rate"]
            metrics["fragmentation_rate"] = identity["fragmentation_rate"]
            metrics["false_emission_rate"] = identity["false_emission_rate"]
            metrics["endpoint_latency_frames_mean"] = identity[
                "endpoint_detection_latency_frames"
            ]["mean"]
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
