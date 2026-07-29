"""Strict-causal, train-only checkpoint replay for EventMATR D1.

This diagnostic never updates a checkpoint and never constructs the locked
test dataset.  Ground truth is retained outside the model boundary and is used
only after each forward pass to compute explicitly labelled diagnostics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import THUMOS14Dataset  # noqa: E402
from models import build_model  # noqa: E402
from on_tal_task import D1_RUNTIME_FORBIDDEN_MODEL_INFO, make_model_inputs  # noqa: E402
from util.utils import memory_initialize, parrallel_collate_fn  # noqa: E402


LANE_SETTINGS = {
    "R": ("r", "fresh_rematch", "b1o0"),
    "T": ("t", "sticky_owner", "b1o1"),
    "H": ("h", "fresh_rematch", "b1o0"),
    "TH": ("th", "sticky_owner", "b1o1"),
}
TRAINING_COMMIT = "1f4bb29ad58dddcc33f6ff2bdc57a5934ee5c53d"
TRAINING_TREE = "aad757531cfcbfb79b616756466251f946574035"
CALIBRATION_BINS = 15
BIRTH_TOLERANCE_FRAMES = 64.0
NEARBY_GAP_FRAMES = 64.0
TIOU_THRESHOLDS = (0.3, 0.5, 0.7)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True, choices=tuple(LANE_SETTINGS))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--options", required=True, type=Path)
    parser.add_argument("--pilot-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiment_configs" / "eventmatr_d1_checkpoint_replay.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-batches",
        default=0,
        type=int,
        help="Nonzero is smoke-only and is rejected as a complete replay.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Write a compressed per-real-prefix trace outside the repository.",
    )
    args = parser.parse_args()
    for field in ("checkpoint", "options", "pilot_receipt", "manifest"):
        path = getattr(args, field).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"{field} does not exist: {path}")
        setattr(args, field, path)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_batches < 0:
        raise SystemExit("--max-batches must be non-negative")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total
    )
    return [
        max(0.0, (centre - radius) / denominator),
        min(1.0, (centre + radius) / denominator),
    ]


class BinaryCalibration:
    def __init__(self, bins: int = CALIBRATION_BINS) -> None:
        self.bins = int(bins)
        self.count = 0
        self.positive = 0
        self.brier_sum = 0.0
        self.nll_sum = 0.0
        self.bin_count = [0] * self.bins
        self.bin_confidence = [0.0] * self.bins
        self.bin_positive = [0] * self.bins

    def update(self, probabilities: Iterable[float], labels: Iterable[bool]) -> None:
        for probability, label in zip(probabilities, labels):
            probability = min(1.0 - 1e-7, max(1e-7, float(probability)))
            target = 1.0 if bool(label) else 0.0
            index = min(self.bins - 1, int(probability * self.bins))
            self.count += 1
            self.positive += int(target)
            self.brier_sum += (probability - target) ** 2
            self.nll_sum += -(
                target * math.log(probability)
                + (1.0 - target) * math.log(1.0 - probability)
            )
            self.bin_count[index] += 1
            self.bin_confidence[index] += probability
            self.bin_positive[index] += int(target)

    def summary(self) -> dict:
        if self.count == 0:
            return {
                "count": 0,
                "positive_count": 0,
                "brier": None,
                "nll": None,
                "ece": None,
                "bins": [],
            }
        ece = 0.0
        rows = []
        for index in range(self.bins):
            count = self.bin_count[index]
            if count:
                confidence = self.bin_confidence[index] / count
                frequency = self.bin_positive[index] / count
                ece += count / self.count * abs(confidence - frequency)
            else:
                confidence = None
                frequency = None
            rows.append(
                {
                    "index": index,
                    "count": count,
                    "mean_confidence": confidence,
                    "positive_frequency": frequency,
                }
            )
        return {
            "count": self.count,
            "positive_count": self.positive,
            "brier": self.brier_sum / self.count,
            "nll": self.nll_sum / self.count,
            "ece": ece,
            "bins": rows,
        }


class StreamingValues:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values: Iterable[float]) -> None:
        for value in values:
            value = float(value)
            if not math.isfinite(value):
                raise RuntimeError("non-finite diagnostic value")
            self.count += 1
            self.total += value
            self.total_square += value * value
            self.minimum = min(self.minimum, value)
            self.maximum = max(self.maximum, value)

    def summary(self) -> dict:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_square / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


class OwnerStateCollector:
    """Capture only ragged runtime owner decodes, excluding dense prototypes."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, _module, inputs, output) -> None:
        if len(inputs) < 3 or inputs[2] is None:
            return
        padding = inputs[2].detach().bool()
        logits = output[0].detach()
        valid_logits = logits[~padding]
        if valid_logits.ndim != 2 or valid_logits.size(-1) != 4:
            raise RuntimeError("owner diagnostic expected [N,4] valid logits")
        if valid_logits.numel() == 0:
            return
        probabilities = valid_logits.softmax(dim=-1)
        argmax = valid_logits.argmax(dim=-1)
        counts = torch.bincount(argmax, minlength=4)
        end_margin = valid_logits[:, 3] - valid_logits[:, :3].max(dim=-1).values
        self.calls.append(
            {
                "rows": int(valid_logits.size(0)),
                "argmax_counts": [int(value) for value in counts.cpu().tolist()],
                "probability_sums": [
                    float(value) for value in probabilities.sum(dim=0).cpu().tolist()
                ],
                "max_end_probability": float(probabilities[:, 3].max().item()),
                "end_margins": [
                    float(value) for value in end_margin.cpu().reshape(-1).tolist()
                ],
            }
        )

    def drain(self) -> list[dict]:
        rows = self.calls
        self.calls = []
        return rows


def _temporal_iou(left: dict, right: dict) -> float:
    intersection = max(
        0.0,
        min(float(left["end_frame"]), float(right["end_frame"]))
        - max(float(left["start_frame"]), float(right["start_frame"])),
    )
    union = max(
        1e-8,
        max(float(left["end_frame"]), float(right["end_frame"]))
        - min(float(left["start_frame"]), float(right["start_frame"])),
    )
    return intersection / union


def _greedy_interval_match(
    predictions: list[dict], ground_truth: list[dict], threshold: float
) -> list[dict]:
    unmatched = set(range(len(ground_truth)))
    matches = []
    ordered_predictions = sorted(
        range(len(predictions)),
        key=lambda index: (
            -float(predictions[index]["score"]),
            float(predictions[index]["emit_frame"]),
            index,
        ),
    )
    for prediction_index in ordered_predictions:
        prediction = predictions[prediction_index]
        candidates = []
        for gt_index in sorted(unmatched):
            target = ground_truth[gt_index]
            if int(prediction["class_id"]) != int(target["class_id"]):
                continue
            overlap = _temporal_iou(prediction, target)
            if overlap >= threshold:
                candidates.append((overlap, -gt_index, gt_index))
        if not candidates:
            continue
        overlap, _, gt_index = max(candidates)
        unmatched.remove(gt_index)
        matches.append(
            {
                "prediction_index": prediction_index,
                "ground_truth_index": gt_index,
                "tiou": float(overlap),
            }
        )
    return matches


def _greedy_birth_match(
    predicted_births: list[dict],
    ground_truth: list[dict],
    tolerance: float = BIRTH_TOLERANCE_FRAMES,
) -> list[dict]:
    unmatched = set(range(len(ground_truth)))
    matches = []
    ordered = sorted(
        range(len(predicted_births)),
        key=lambda index: (
            -float(predicted_births[index]["score"]),
            float(predicted_births[index]["frame"]),
            index,
        ),
    )
    for prediction_index in ordered:
        prediction = predicted_births[prediction_index]
        candidates = []
        for gt_index in sorted(unmatched):
            target = ground_truth[gt_index]
            if int(prediction["class_id"]) != int(target["class_id"]):
                continue
            delay = float(prediction["frame"]) - float(target["admission_frame"])
            # A prediction before the event becomes observable is a false
            # alarm, never a negatively delayed true positive.
            if 0.0 <= delay <= tolerance:
                candidates.append((-abs(delay), -gt_index, gt_index, delay))
        if not candidates:
            continue
        _, _, gt_index, delay = max(candidates)
        unmatched.remove(gt_index)
        matches.append(
            {
                "prediction_index": prediction_index,
                "ground_truth_index": gt_index,
                "delay_frames": float(delay),
            }
        )
    return matches


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90_abs": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90_abs": float(np.quantile(np.abs(array), 0.90)),
    }


def _diagnose(stage: dict) -> dict:
    births = int(stage["start_rising_births"])
    cancellations = int(stage["cancellations"])
    ends = int(stage["ends"])
    emissions = int(stage["emissions"])
    unresolved = int(stage["active_at_observed_eos"])
    if births == 0:
        label = "candidate_start_starvation"
    elif ends == 0 and cancellations > 0 and unresolved == 0:
        label = "owner_cancellation_before_end"
    elif ends == 0 and unresolved > 0 and cancellations == 0:
        label = "end_state_starvation"
    elif ends == 0 and unresolved > 0 and cancellations > 0:
        label = "mixed_owner_cancellation_and_end_state_starvation"
    elif ends > 0 and emissions == 0:
        label = "end_to_writer_failure"
    elif emissions > 0:
        label = "live_but_stage_rates_require_comparison"
    else:
        label = "unclassified_lifecycle_failure"
    return {
        "label": label,
        "birth_to_cancel_ratio": (
            cancellations / births if births else None
        ),
        "birth_to_end_ratio": ends / births if births else None,
        "birth_to_emission_ratio": emissions / births if births else None,
        "end_to_emission_ratio": emissions / ends if ends else None,
    }


def _validate_options(options: dict, lane: str) -> None:
    d1_lane, ownership, arm = LANE_SETTINGS[lane]
    expected = {
        "model_variant": "eventmatr",
        "event_lifecycle_version": "d1_censored",
        "event_d1_lane": d1_lane,
        "ownership_mode": ownership,
        "event_arm": arm,
        "birth_mode": "instant_transition",
        "event_birth_logit_threshold": None,
        "event_end_logit_threshold": None,
        "study_protocol": "d1_preexperiment",
        "epochs": 5,
        "random_seed": 52,
        "p_videos": 1,
        "batch": 64,
        "use_flag": True,
    }
    mismatches = {
        key: {"expected": value, "actual": options.get(key)}
        for key, value in expected.items()
        if options.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "checkpoint replay option mismatch:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )
    if int(options.get("event_emit_delay_frames", 0)) != 0:
        raise RuntimeError(
            "replay owner alignment requires the registered zero-frame emit delay"
        )
    locked_path = Path(options["video_feature_all_test"]).expanduser().resolve()
    if locked_path.exists():
        raise RuntimeError(f"locked-test sentinel unexpectedly exists: {locked_path}")
    if locked_path.name != "LOCKED_TEST_NOT_MOUNTED.pickle":
        raise RuntimeError("options do not retain the absent locked-test sentinel")


def _validate_pilot_receipt(receipt: dict, lane: str, checkpoint: Path) -> None:
    expected = {
        "status": "PASS",
        "protocol": "eventmatr_d1_seed52_short_pilot_v1",
        "lane": lane,
        "epochs": 5,
        "seed": 52,
        "test_access": False,
        "strict_causal_paper_result_valid": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(
                f"pilot receipt {key} mismatch: {receipt.get(key)!r} != {value!r}"
            )
    identity = receipt.get("source_identity", {})
    if identity.get("commit") != TRAINING_COMMIT:
        raise RuntimeError("pilot checkpoint training commit mismatch")
    if identity.get("tree") != TRAINING_TREE:
        raise RuntimeError("pilot checkpoint training tree mismatch")
    checkpoint_row = receipt.get("checkpoint", {})
    if int(checkpoint_row.get("bytes", -1)) != checkpoint.stat().st_size:
        raise RuntimeError("pilot receipt checkpoint size mismatch")
    if Path(checkpoint_row.get("path", "")).resolve() != checkpoint:
        raise RuntimeError("pilot receipt checkpoint path mismatch")


def _ground_truth_by_video(dataset: THUMOS14Dataset) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for video_name in dataset.video_list:
        rows = []
        for event_id, row in enumerate(dataset.gt_action[video_name]):
            end_frame = float(row[0])
            start_frame = end_frame - float(row[1])
            rows.append(
                {
                    "event_id": int(event_id),
                    "class_id": int(row[2]),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "admission_frame": float(math.ceil(start_frame)),
                    "nearby_same_class": False,
                }
            )
        for left_index, left in enumerate(rows):
            for right_index, right in enumerate(rows):
                if left_index == right_index or left["class_id"] != right["class_id"]:
                    continue
                gap = max(
                    0.0,
                    max(left["start_frame"], right["start_frame"])
                    - min(left["end_frame"], right["end_frame"]),
                )
                if gap <= NEARBY_GAP_FRAMES:
                    left["nearby_same_class"] = True
                    break
        output[str(video_name)] = rows
    return output


def _make_scientific_metrics(
    ground_truth: dict[str, list[dict]],
    ledgers: dict[str, list[dict]],
    predicted_births: dict[str, list[dict]],
) -> dict:
    interval_metrics = {}
    matches_by_threshold: dict[float, dict[str, list[dict]]] = {}
    all_predictions = [
        row for rows in ledgers.values() for row in rows
    ]
    all_ground_truth = [
        row for rows in ground_truth.values() for row in rows
    ]
    for threshold in TIOU_THRESHOLDS:
        per_video = {}
        match_count = 0
        for video_name, targets in ground_truth.items():
            matches = _greedy_interval_match(
                ledgers.get(video_name, []), targets, threshold
            )
            per_video[video_name] = matches
            match_count += len(matches)
        matches_by_threshold[threshold] = per_video
        interval_metrics[str(threshold)] = {
            "matches": match_count,
            "precision": (
                match_count / len(all_predictions) if all_predictions else None
            ),
            "precision_wilson_95": _wilson(match_count, len(all_predictions)),
            "recall": (
                match_count / len(all_ground_truth) if all_ground_truth else None
            ),
            "recall_wilson_95": _wilson(match_count, len(all_ground_truth)),
        }

    start_errors = []
    end_errors = []
    emit_delays = []
    matched_prediction_indices: dict[str, set[int]] = defaultdict(set)
    matched_gt_indices: dict[str, set[int]] = defaultdict(set)
    for video_name, matches in matches_by_threshold[0.3].items():
        predictions = ledgers.get(video_name, [])
        targets = ground_truth[video_name]
        for match in matches:
            prediction_index = int(match["prediction_index"])
            gt_index = int(match["ground_truth_index"])
            prediction = predictions[prediction_index]
            target = targets[gt_index]
            matched_prediction_indices[video_name].add(prediction_index)
            matched_gt_indices[video_name].add(gt_index)
            start_errors.append(
                float(prediction["start_frame"]) - float(target["start_frame"])
            )
            end_errors.append(
                float(prediction["end_frame"]) - float(target["end_frame"])
            )
            emit_delays.append(
                float(prediction["emit_frame"]) - float(target["end_frame"])
            )

    duplicate_predictions = 0
    fragmented_gt = 0
    for video_name, targets in ground_truth.items():
        predictions = ledgers.get(video_name, [])
        for target in targets:
            count = sum(
                int(prediction["class_id"]) == int(target["class_id"])
                and _temporal_iou(prediction, target) >= 0.3
                for prediction in predictions
            )
            duplicate_predictions += max(0, count - 1)
            fragmented_gt += int(count > 1)

    nearby_total = 0
    nearby_matched = 0
    for video_name, targets in ground_truth.items():
        for gt_index, target in enumerate(targets):
            if target["nearby_same_class"]:
                nearby_total += 1
                nearby_matched += int(gt_index in matched_gt_indices[video_name])

    birth_matches = []
    total_predicted_births = 0
    for video_name, targets in ground_truth.items():
        predictions = predicted_births.get(video_name, [])
        total_predicted_births += len(predictions)
        birth_matches.extend(_greedy_birth_match(predictions, targets))
    birth_delays = [float(row["delay_frames"]) for row in birth_matches]

    emission_calibration = BinaryCalibration()
    correct_at_05: dict[str, set[int]] = defaultdict(set)
    for video_name, matches in matches_by_threshold[0.5].items():
        correct_at_05[video_name] = {
            int(match["prediction_index"]) for match in matches
        }
    for video_name, predictions in ledgers.items():
        emission_calibration.update(
            [float(row["score"]) for row in predictions],
            [
                prediction_index in correct_at_05[video_name]
                for prediction_index in range(len(predictions))
            ],
        )

    return {
        "ground_truth_event_count": len(all_ground_truth),
        "proposal_count": len(all_predictions),
        "videos_with_proposals": sum(bool(rows) for rows in ledgers.values()),
        "video_count": len(ground_truth),
        "video_coverage": (
            sum(bool(rows) for rows in ledgers.values()) / len(ground_truth)
            if ground_truth
            else None
        ),
        "interval_detection": interval_metrics,
        "timing_frames_at_tiou_0.3": {
            "start_error": _distribution(start_errors),
            "end_error": _distribution(end_errors),
            "emit_delay": _distribution(emit_delays),
        },
        "birth_within_64_frames": {
            "predicted_births": total_predicted_births,
            "matches": len(birth_matches),
            "precision": (
                len(birth_matches) / total_predicted_births
                if total_predicted_births
                else None
            ),
            "recall": (
                len(birth_matches) / len(all_ground_truth)
                if all_ground_truth
                else None
            ),
            "delay_frames": _distribution(birth_delays),
        },
        "identity": {
            "duplicate_predictions_at_tiou_0.3": duplicate_predictions,
            "fragmented_ground_truth_events_at_tiou_0.3": fragmented_gt,
            "nearby_same_class_ground_truth_events": nearby_total,
            "nearby_same_class_matches_at_tiou_0.3": nearby_matched,
            "nearby_same_class_recall_at_tiou_0.3": (
                nearby_matched / nearby_total if nearby_total else None
            ),
        },
        "emission_calibration_at_tiou_0.5": emission_calibration.summary(),
    }


def main() -> None:
    cli = _parse_args()
    if cli.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("formal checkpoint replay requires one CUDA device")
    if torch.cuda.device_count() != 1:
        raise SystemExit("formal checkpoint replay requires exactly one visible GPU")

    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise SystemExit(f"checkpoint replay source is dirty:\n{status}")
    source_identity = {
        "commit": _git("rev-parse", "HEAD"),
        "tree": _git("rev-parse", "HEAD^{tree}"),
        "manifest_sha256": _sha256(cli.manifest),
    }

    options = json.loads(cli.options.read_text(encoding="utf-8"))
    _validate_options(options, cli.lane)
    receipt = json.loads(cli.pilot_receipt.read_text(encoding="utf-8"))
    _validate_pilot_receipt(receipt, cli.lane, cli.checkpoint)
    checkpoint_stat_before = {
        "bytes": cli.checkpoint.stat().st_size,
        "mtime_ns": cli.checkpoint.stat().st_mtime_ns,
        "sha256": _sha256(cli.checkpoint),
    }

    random.seed(52)
    np.random.seed(52)
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    options["training"] = False
    options["mode"] = "eval"
    options["make_output"] = False
    options["num_workers"] = int(options.get("num_workers", 4))
    args = SimpleNamespace(**options)
    device = torch.device("cuda")

    dataset = THUMOS14Dataset(args, subset="train")
    if len(dataset.video_list) != 200 and cli.max_batches == 0:
        raise RuntimeError(
            f"complete replay expected 200 train/validation videos, got {len(dataset.video_list)}"
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    model = torch.nn.DataParallel(build_model(args)).to(device)
    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    if checkpoint.get("epoch") != 5:
        raise RuntimeError("checkpoint epoch mismatch")
    if checkpoint.get("study_protocol") != "d1_preexperiment":
        raise RuntimeError("checkpoint protocol mismatch")
    if checkpoint.get("model_variant") != "eventmatr":
        raise RuntimeError("checkpoint model variant mismatch")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model.eval()
    memory_initialize(model, args)

    owner_collector = OwnerStateCollector()
    hook = model.module.event_owner_decoder.register_forward_hook(owner_collector)
    torch.cuda.reset_peak_memory_stats(device)
    trace_path = cli.output_dir / f"{cli.lane}_prefix_trace.jsonl.gz"
    trace_handle = (
        gzip.open(trace_path, "wt", encoding="utf-8") if cli.trace else None
    )

    stage = {
        "real_prefixes": 0,
        "candidate_queries": 0,
        "candidate_state_argmax": [0, 0, 0, 0],
        "candidate_start_prefixes": 0,
        "start_rising_births": 0,
        "gt_birth_prefixes": 0,
        "gt_birth_prefixes_with_candidate_start": 0,
        "gt_birth_prefixes_with_predicted_birth": 0,
        "owner_rows": 0,
        "owner_state_argmax": [0, 0, 0, 0],
        "gt_end_prefixes": 0,
        "gt_end_prefixes_with_owner": 0,
        "gt_end_prefixes_with_owner_end": 0,
        "cancellations": 0,
        "ends": 0,
        "emissions": 0,
        "reacquisitions": 0,
        "runtime_capacity_exhaustions": 0,
        "padding_prefixes_ignored": 0,
        "observed_eos": 0,
        "active_at_observed_eos": 0,
        "ledger_rows": 0,
        "native_gate_predicted_on": 0,
        "native_gate_gt_on": 0,
        "native_gate_true_positive": 0,
        "native_gate_false_positive": 0,
        "native_gate_false_negative": 0,
    }
    candidate_start_margin = StreamingValues()
    owner_end_margin = StreamingValues()
    candidate_birth_calibration = BinaryCalibration()
    lifecycle_end_calibration = BinaryCalibration()
    native_gate_calibration = BinaryCalibration()
    predicted_births: dict[str, list[dict]] = defaultdict(list)
    per_video: dict[str, dict] = defaultdict(
        lambda: {
            "real_prefixes": 0,
            "births": 0,
            "cancellations": 0,
            "ends": 0,
            "emissions": 0,
            "reacquisitions": 0,
            "observed_eos": 0,
            "active_at_observed_eos": 0,
        }
    )
    last_real_frame: dict[str, float] = {}
    started_at = time.perf_counter()
    processed_batches = 0

    try:
        with torch.no_grad():
            for batch_index, (features, targets, infos) in enumerate(loader):
                if cli.max_batches and batch_index >= cli.max_batches:
                    break
                processed_batches += 1
                features, targets, infos = parrallel_collate_fn(
                    features, targets, infos, args.p_videos
                )
                features = features.to(device, non_blocking=True)
                target_tensors = {
                    key: value.to(device, non_blocking=True)
                    for key, value in targets.items()
                }
                payload = make_model_inputs(args, features, infos)
                forbidden = D1_RUNTIME_FORBIDDEN_MODEL_INFO.intersection(
                    payload["infos"]
                )
                if forbidden:
                    raise RuntimeError(
                        f"future/full-video metadata reached model: {sorted(forbidden)}"
                    )
                outputs = model(payload, device)
                owner_calls = owner_collector.drain()

                real_mask = infos["is_real_prefix"]
                if not torch.is_tensor(real_mask):
                    real_mask = torch.tensor(real_mask)
                real_mask = real_mask.detach().cpu().bool().reshape(-1)
                eos_mask = infos["is_eos"]
                if not torch.is_tensor(eos_mask):
                    eos_mask = torch.tensor(eos_mask)
                eos_mask = eos_mask.detach().cpu().bool().reshape(-1)
                frames = infos["current_frame"]
                if torch.is_tensor(frames):
                    frames = frames.detach().cpu().reshape(-1).tolist()
                else:
                    frames = list(frames)
                names = [str(value) for value in infos["video_name"]]

                state_logits = outputs["event_state_logits"].detach().cpu()
                state_argmax = state_logits.argmax(dim=-1)
                state_probabilities = state_logits.softmax(dim=-1)
                birth_margin = outputs["event_birth_logits"].detach().cpu()
                birth_mask = outputs["event_new_birth_mask"].detach().cpu().bool()
                cancel_mask = outputs["event_cancelled_mask"].detach().cpu().bool()
                end_mask = outputs["event_ended_mask"].detach().cpu().bool()
                emit_mask = outputs["event_emitted_mask"].detach().cpu().bool()
                active_after = outputs["event_active_count"].detach().cpu().long()
                birth_count = outputs["event_birth_count"].detach().cpu().long()
                end_count = outputs["event_end_count"].detach().cpu().long()
                emit_count = outputs["event_emit_count"].detach().cpu().long()
                cancellation_count = outputs[
                    "event_cancellation_count"
                ].detach().cpu().long()
                reacquisition_count = outputs[
                    "event_reacquisition_count"
                ].detach().cpu().long()
                capacity = outputs[
                    "event_runtime_capacity_exhaustions"
                ].detach().cpu().long()
                padding = outputs["event_padding_prefixes_ignored"].detach().cpu().long()
                observed_eos = outputs["event_eos_observed"].detach().cpu().long()
                class_logits = outputs["pred_cls"].detach().cpu()
                flag_probabilities = outputs["pred_flag"].detach().cpu().reshape(-1)
                gt_valid = target_tensors["event_valid_mask"].detach().cpu().bool()
                gt_rows = target_tensors["event_targets"].detach().cpu()
                gt_birth_any = ((gt_rows[..., 5] > 0.5) & gt_valid).any(dim=1)
                gt_end_any = ((gt_rows[..., 7] > 0.5) & gt_valid).any(dim=1)
                gt_native_gate = infos["segment_flag"]
                if not torch.is_tensor(gt_native_gate):
                    gt_native_gate = torch.tensor(gt_native_gate)
                gt_native_gate = (
                    gt_native_gate.detach().cpu().bool().reshape(-1)
                )

                active_before = (
                    active_after
                    - birth_count
                    + cancellation_count
                    + emit_count
                )
                if bool((active_before < 0).any()):
                    raise RuntimeError("negative reconstructed owner count")
                owner_positions = [
                    index
                    for index in range(len(names))
                    if int(active_before[index].item()) > 0
                ]
                if len(owner_positions) != len(owner_calls):
                    raise RuntimeError(
                        "owner hook/runtime alignment mismatch: "
                        f"{len(owner_positions)} active prefixes != {len(owner_calls)} calls"
                    )
                owner_by_position = {
                    position: owner_calls[index]
                    for index, position in enumerate(owner_positions)
                }

                for row_index, video_name in enumerate(names):
                    if not bool(real_mask[row_index]):
                        stage["padding_prefixes_ignored"] += int(
                            padding[row_index].item()
                        )
                        continue
                    frame = float(frames[row_index])
                    previous_frame = last_real_frame.get(video_name)
                    if previous_frame is not None and frame < previous_frame:
                        raise RuntimeError(
                            f"non-monotonic replay frame for {video_name}: "
                            f"{frame} after {previous_frame}"
                        )
                    last_real_frame[video_name] = frame
                    row_states = state_argmax[row_index]
                    candidate_counts = torch.bincount(row_states, minlength=4)
                    for state_index, count in enumerate(candidate_counts.tolist()):
                        stage["candidate_state_argmax"][state_index] += int(count)
                    stage["real_prefixes"] += 1
                    stage["candidate_queries"] += int(row_states.numel())
                    has_candidate_start = bool((row_states == 1).any().item())
                    stage["candidate_start_prefixes"] += int(has_candidate_start)
                    candidate_start_margin.update(
                        birth_margin[row_index].reshape(-1).tolist()
                    )

                    births = int(birth_count[row_index].item())
                    cancellations = int(cancellation_count[row_index].item())
                    ends = int(end_count[row_index].item())
                    emissions = int(emit_count[row_index].item())
                    reacquisitions = int(reacquisition_count[row_index].item())
                    if births != int(birth_mask[row_index].sum().item()):
                        raise RuntimeError(
                            "predicted-only replay birth count/mask mismatch"
                        )
                    if cancellations < int(cancel_mask[row_index].sum().item()):
                        raise RuntimeError("cancellation count is smaller than its mask")
                    if ends < int(end_mask[row_index].sum().item()):
                        raise RuntimeError("end count is smaller than its mask")
                    if emissions < int(emit_mask[row_index].sum().item()):
                        raise RuntimeError("emission count is smaller than its mask")
                    stage["start_rising_births"] += births
                    stage["cancellations"] += cancellations
                    stage["ends"] += ends
                    stage["emissions"] += emissions
                    stage["reacquisitions"] += reacquisitions
                    stage["runtime_capacity_exhaustions"] += int(
                        capacity[row_index].item()
                    )
                    stage["observed_eos"] += int(observed_eos[row_index].item())
                    has_gt_birth = bool(gt_birth_any[row_index].item())
                    has_gt_end = bool(gt_end_any[row_index].item())
                    stage["gt_birth_prefixes"] += int(has_gt_birth)
                    stage["gt_birth_prefixes_with_candidate_start"] += int(
                        has_gt_birth and has_candidate_start
                    )
                    stage["gt_birth_prefixes_with_predicted_birth"] += int(
                        has_gt_birth and births > 0
                    )
                    if bool(eos_mask[row_index]):
                        stage["active_at_observed_eos"] += int(
                            active_after[row_index].item()
                        )

                    video_row = per_video[video_name]
                    video_row["real_prefixes"] += 1
                    video_row["births"] += births
                    video_row["cancellations"] += cancellations
                    video_row["ends"] += ends
                    video_row["emissions"] += emissions
                    video_row["reacquisitions"] += reacquisitions
                    video_row["observed_eos"] += int(observed_eos[row_index].item())
                    if bool(eos_mask[row_index]):
                        video_row["active_at_observed_eos"] = int(
                            active_after[row_index].item()
                        )

                    birth_score = float(
                        state_probabilities[row_index, :, 1].max().item()
                    )
                    candidate_birth_calibration.update(
                        [birth_score], [bool(gt_birth_any[row_index].item())]
                    )
                    owner_row = owner_by_position.get(
                        row_index,
                        {
                            "rows": 0,
                            "argmax_counts": [0, 0, 0, 0],
                            "probability_sums": [0.0, 0.0, 0.0, 0.0],
                            "max_end_probability": 0.0,
                            "end_margins": [],
                        },
                    )
                    stage["owner_rows"] += int(owner_row["rows"])
                    for state_index, count in enumerate(owner_row["argmax_counts"]):
                        stage["owner_state_argmax"][state_index] += int(count)
                    owner_end_margin.update(owner_row["end_margins"])
                    has_owner = int(owner_row["rows"]) > 0
                    has_owner_end = int(owner_row["argmax_counts"][3]) > 0
                    stage["gt_end_prefixes"] += int(has_gt_end)
                    stage["gt_end_prefixes_with_owner"] += int(
                        has_gt_end and has_owner
                    )
                    stage["gt_end_prefixes_with_owner_end"] += int(
                        has_gt_end and has_owner_end
                    )
                    if has_owner:
                        lifecycle_end_calibration.update(
                            [float(owner_row["max_end_probability"])],
                            [has_gt_end],
                        )

                    gate_probability = float(flag_probabilities[row_index].item())
                    gate_prediction = gate_probability > float(args.flag_threshold)
                    gate_target = bool(gt_native_gate[row_index].item())
                    native_gate_calibration.update(
                        [gate_probability], [gate_target]
                    )
                    stage["native_gate_predicted_on"] += int(gate_prediction)
                    stage["native_gate_gt_on"] += int(gate_target)
                    stage["native_gate_true_positive"] += int(
                        gate_prediction and gate_target
                    )
                    stage["native_gate_false_positive"] += int(
                        gate_prediction and not gate_target
                    )
                    stage["native_gate_false_negative"] += int(
                        not gate_prediction and gate_target
                    )

                    for query_index in birth_mask[row_index].nonzero(
                        as_tuple=False
                    ).reshape(-1).tolist():
                        predicted_class = int(
                            class_logits[row_index, query_index].argmax().item()
                        )
                        predicted_births[video_name].append(
                            {
                                "frame": frame,
                                "query_index": int(query_index),
                                "class_id": predicted_class,
                                "score": float(
                                    state_probabilities[
                                        row_index, query_index, 1
                                    ].item()
                                ),
                            }
                        )

                    if trace_handle is not None:
                        trace_handle.write(
                            json.dumps(
                                {
                                    "video_name": video_name,
                                    "frame": frame,
                                    "is_eos": bool(eos_mask[row_index]),
                                    "gt_birth": bool(
                                        gt_birth_any[row_index].item()
                                    ),
                                    "gt_end": bool(gt_end_any[row_index].item()),
                                    "native_gate_probability": gate_probability,
                                    "native_gate_prediction": gate_prediction,
                                    "native_gate_target": gate_target,
                                    "candidate_state_argmax": [
                                        int(value)
                                        for value in candidate_counts.tolist()
                                    ],
                                    "candidate_max_start_probability": birth_score,
                                    "owner_rows": int(owner_row["rows"]),
                                    "owner_state_argmax": owner_row[
                                        "argmax_counts"
                                    ],
                                    "owner_max_end_probability": float(
                                        owner_row["max_end_probability"]
                                    ),
                                    "births": births,
                                    "cancellations": cancellations,
                                    "ends": ends,
                                    "emissions": emissions,
                                    "reacquisitions": reacquisitions,
                                    "active_after": int(
                                        active_after[row_index].item()
                                    ),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
    finally:
        hook.remove()
        if trace_handle is not None:
            trace_handle.close()

    elapsed = time.perf_counter() - started_at
    module = model.module
    ground_truth = _ground_truth_by_video(dataset)
    ledgers = {
        video_name: [dict(row) for row in module.event_memory.ledger(video_name)]
        for video_name in dataset.video_list
    }
    stage["ledger_rows"] = sum(len(rows) for rows in ledgers.values())
    if stage["ledger_rows"] != stage["emissions"]:
        raise RuntimeError(
            f"ledger/emission mismatch: {stage['ledger_rows']} != {stage['emissions']}"
        )
    if stage["runtime_capacity_exhaustions"] != 0:
        raise RuntimeError("replay encountered runtime capacity exhaustion")
    if cli.max_batches == 0 and stage["observed_eos"] != len(dataset.video_list):
        raise RuntimeError(
            f"complete replay observed {stage['observed_eos']} EOS markers for "
            f"{len(dataset.video_list)} videos"
        )

    distinct_event_ids = set()
    for video_name, rows in ledgers.items():
        sequence_ids = set()
        for row in rows:
            key = (video_name, int(row["event_id"]))
            if key in distinct_event_ids:
                raise RuntimeError(f"duplicate immutable event id: {key}")
            distinct_event_ids.add(key)
            sequence_id = int(row["sequence_id"])
            if sequence_id in sequence_ids:
                raise RuntimeError(f"duplicate sequence id for {video_name}")
            sequence_ids.add(sequence_id)
            if not (
                0.0
                <= float(row["start_frame"])
                < float(row["end_frame"])
                <= float(row["emit_frame"])
            ):
                raise RuntimeError(f"invalid emitted interval: {row}")
            if float(row["emit_frame"]) > last_real_frame.get(video_name, -math.inf):
                raise RuntimeError(f"emission exceeds observed prefix: {row}")

    total_created_records = sum(
        int(value) for value in module.event_memory._next_event_id.values()
    )
    reacquisitions = int(stage["reacquisitions"])
    if stage["start_rising_births"] - reacquisitions != total_created_records:
        raise RuntimeError(
            "birth/reacquisition accounting mismatch: "
            f"{stage['start_rising_births']} - {reacquisitions} "
            f"!= {total_created_records}"
        )

    scientific_metrics = _make_scientific_metrics(
        ground_truth, ledgers, predicted_births
    )
    checkpoint_stat_after = {
        "bytes": cli.checkpoint.stat().st_size,
        "mtime_ns": cli.checkpoint.stat().st_mtime_ns,
        "sha256": _sha256(cli.checkpoint),
    }
    if checkpoint_stat_after != checkpoint_stat_before:
        raise RuntimeError("checkpoint changed during read-only replay")

    result = {
        "status": "PASS",
        "protocol": "eventmatr_d1_strict_causal_checkpoint_replay_v1",
        "lane": cli.lane,
        "complete_replay": cli.max_batches == 0,
        "processed_batches": processed_batches,
        "strict_causal_paper_result_valid": False,
        "test_access": False,
        "checkpoint_updated": False,
        "threshold_search": False,
        "ground_truth_visible_to_model": False,
        "source_identity": source_identity,
        "checkpoint_training_source": {
            "commit": TRAINING_COMMIT,
            "tree": TRAINING_TREE,
            "seed": 52,
            "epoch": 5,
        },
        "checkpoint": {
            "path": str(cli.checkpoint),
            **checkpoint_stat_before,
        },
        "options": {
            "path": str(cli.options),
            "sha256": _sha256(cli.options),
        },
        "pilot_receipt": {
            "path": str(cli.pilot_receipt),
            "sha256": _sha256(cli.pilot_receipt),
        },
        "manifest": {
            "path": str(cli.manifest),
            "sha256": source_identity["manifest_sha256"],
        },
        "fixed_decisions": {
            "birth_logit_threshold": None,
            "end_logit_threshold": None,
            "birth_rule": "four_state_argmax_start_rising_edge",
            "end_rule": "owner_four_state_argmax_end",
        },
        "stage_counts": stage,
        "state_statistics": {
            "candidate_start_margin": candidate_start_margin.summary(),
            "owner_end_margin": owner_end_margin.summary(),
            "candidate_birth_calibration": candidate_birth_calibration.summary(),
            "lifecycle_end_opportunity_calibration": (
                lifecycle_end_calibration.summary()
            ),
            "native_memory_gate_calibration": native_gate_calibration.summary(),
        },
        "lifecycle_accounting": {
            "distinct_created_records": total_created_records,
            "reacquisitions": reacquisitions,
            "closure_ratio": (
                stage["emissions"] / stage["start_rising_births"]
                if stage["start_rising_births"]
                else None
            ),
            "cancel_ratio": (
                stage["cancellations"] / stage["start_rising_births"]
                if stage["start_rising_births"]
                else None
            ),
            "birth_wilson_95": _wilson(
                stage["start_rising_births"], stage["candidate_queries"]
            ),
            "emission_per_birth_wilson_95": _wilson(
                stage["emissions"], stage["start_rising_births"]
            ),
        },
        "root_cause": _diagnose(stage),
        "scientific_metrics": scientific_metrics,
        "per_video_stage_counts": {
            video_name: dict(rows) for video_name, rows in sorted(per_video.items())
        },
        "systems": {
            "device": torch.cuda.get_device_name(0),
            "elapsed_seconds": elapsed,
            "prefixes_per_second": (
                stage["real_prefixes"] / elapsed if elapsed > 0 else None
            ),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "trace": (
            {
                "path": str(trace_path),
                "sha256": _sha256(trace_path),
                "compressed_jsonl": True,
            }
            if cli.trace
            else None
        ),
    }
    if not _finite_tree(result):
        raise RuntimeError("replay result contains non-finite values")
    output_path = cli.output_dir / f"{cli.lane}_replay_summary.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_path)
    print(json.dumps({"lane": cli.lane, "root_cause": result["root_cause"], "stage_counts": stage}, sort_keys=True))


if __name__ == "__main__":
    main()
