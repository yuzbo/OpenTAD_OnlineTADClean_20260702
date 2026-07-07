import json

import pandas as pd

from .builder import EVALUATORS
from .mAP import mAP
from opentad.utils.online_protocol import summarize_emission_ledger, validate_emission_ledger_summary


LEDGER_FIELDS = {
    "emit_frame",
    "start_frame",
    "end_frame",
    "source_frame",
    "latency_sec",
    "source_grid",
    "stream_key",
}


def compute_online_detection_metrics(result_dict, require_no_future=True):
    summary = summarize_emission_ledger(result_dict)
    if require_no_future:
        validate_emission_ledger_summary(summary)
    no_future = summary["no_future"]
    return dict(
        num_predictions=sum(len(rows) for rows in result_dict.values()),
        online=dict(
            num_emissions=summary["num_emissions"],
            num_streams=summary["num_streams"],
            no_future=no_future,
            future_end_violations=int(no_future.get("future_end_violations", 0)),
            future_source_violations=int(no_future.get("future_source_violations", 0)),
            emission_latency_sec=summary["latency_sec"],
        ),
    )


@EVALUATORS.register_module()
class OnlineMAP(mAP):
    """mAP over online-emitted detections with ledger-aware validity stats."""

    def __init__(
        self,
        *args,
        require_ledger=True,
        require_no_future=True,
        max_latency_sec=None,
        online_map=None,
        **kwargs,
    ):
        del online_map
        self.require_ledger = bool(require_ledger)
        self.require_no_future = bool(require_no_future)
        self.max_latency_sec = max_latency_sec
        self.online_metric_dict = {}
        super().__init__(*args, **kwargs)

    def _load_prediction_data(self, prediction_filename):
        if isinstance(prediction_filename, str):
            with open(prediction_filename, "r") as fobj:
                return json.load(fobj)
        if isinstance(prediction_filename, dict):
            return prediction_filename
        raise IOError(f"Type of prediction file is {type(prediction_filename)}.")

    def _validate_ledger_row(self, row):
        missing = sorted(LEDGER_FIELDS.difference(row.keys()))
        if self.require_ledger and missing:
            raise IOError(f"OnlineMAP requires ledger fields: {missing}")
        if self.max_latency_sec is not None and float(row.get("latency_sec", 0.0)) > float(self.max_latency_sec):
            return False
        return True

    def _filter_prediction_results(self, results):
        filtered = {}
        for video_id, rows in results.items():
            if video_id in self.blocked_videos:
                continue
            if self.allowed_videos is not None and video_id not in self.allowed_videos:
                continue
            kept_rows = []
            for row in rows:
                if self._validate_ledger_row(row):
                    kept_rows.append(row)
            filtered[video_id] = kept_rows
        return filtered

    def _import_prediction(self, prediction_filename):
        data = self._load_prediction_data(prediction_filename)
        if not all([field in list(data.keys()) for field in self.pred_fields]):
            raise IOError("Please input a valid prediction file.")

        filtered_results = self._filter_prediction_results(data["results"])
        self.online_metric_dict = compute_online_detection_metrics(
            filtered_results,
            require_no_future=self.require_no_future,
        )

        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        latency_lst, emit_frame_lst = [], []
        for video_id, rows in filtered_results.items():
            for result in rows:
                try:
                    label = self.activity_index[result["label"]]
                except Exception:
                    label = len(self.activity_index)
                video_lst.append(video_id)
                t_start_lst.append(float(result["segment"][0]))
                t_end_lst.append(float(result["segment"][1]))
                label_lst.append(label)
                score_lst.append(float(result["score"]))
                latency_lst.append(float(result.get("latency_sec", 0.0)))
                emit_frame_lst.append(float(result.get("emit_frame", -1)))

        return pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
                "score": score_lst,
                "latency_sec": latency_lst,
                "emit_frame": emit_frame_lst,
            }
        )

    def evaluate(self):
        metric_dict = super().evaluate()
        metric_dict["online"] = self.online_metric_dict.get("online", {})
        return metric_dict

    def logging(self, logger=None):
        super().logging(logger=logger)
        pprint = print if logger is None else logger.info
        online = self.online_metric_dict.get("online", {})
        if online:
            pprint(f"Online emissions: {online.get('num_emissions', 0)}")
            pprint(f"Online latency summary: {online.get('emission_latency_sec', {})}")
            pprint(f"Online no-future summary: {online.get('no_future', {})}")
