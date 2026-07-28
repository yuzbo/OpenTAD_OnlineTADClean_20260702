"""Stage-D0 audit for already-trained EventMATR-v1 checkpoints.

This script does not train or update a checkpoint.  It answers two questions
that the matched 100-epoch training loop could not answer:

1. Did an event checkpoint learn non-trivial START/ALIVE/END scores and
   gradients on a real official training batch?
2. When the same checkpoint is put in ``eval()`` and replayed from the first
   prefix of every training video, does its runtime ledger actually emit
   intervals?

Only the official THUMOS14 validation/train feature pickle is opened.  The
locked test feature path remains an absent sentinel.  The resulting train
replay mAP is diagnostic evidence, not a paper result: EventMATR-v1 consumes
``true_duration``/EOS metadata and therefore does not satisfy the revised
strict-causal endpoint-metadata contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from criterion import build_criterion
from dataset import THUMOS14Dataset
from eval import evaluation_detection
from models import build_model
from on_tal_task import write_model_predictions
from util.utils import memory_initialize, online_nms, parrallel_collate_fn


ARM_MODES = {
    "b0o0": ("matr_delayed", "fresh_rematch"),
    "b1o0": ("instant_transition", "fresh_rematch"),
    "b0o1": ("matr_delayed", "sticky_owner"),
    "b1o1": ("instant_transition", "sticky_owner"),
}
EVENT_LOGIT_KEYS = (
    "event_birth_logits",
    "event_alive_logits",
    "event_end_logits",
)


class RunningTensorStats:
    """Numerically stable-enough streaming moments for diagnostic logits."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, value: torch.Tensor) -> None:
        flat = value.detach().float().reshape(-1)
        if flat.numel() == 0:
            return
        if not torch.isfinite(flat).all():
            raise RuntimeError("non-finite tensor encountered during D0 replay")
        self.count += int(flat.numel())
        self.total += float(flat.sum().item())
        self.total_sq += float((flat * flat).sum().item())
        self.minimum = min(self.minimum, float(flat.min().item()))
        self.maximum = max(self.maximum, float(flat.max().item()))

    def receipt(self) -> dict:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _binary_constant_optimum(positive: int, total: int) -> dict:
    if total <= 0:
        return {
            "positive": int(positive),
            "total": int(total),
            "positive_ratio": None,
            "optimal_constant_logit": None,
            "minimum_constant_bce": None,
        }
    ratio = float(positive) / float(total)
    if ratio <= 0.0:
        logit = None
        entropy = 0.0
    elif ratio >= 1.0:
        logit = None
        entropy = 0.0
    else:
        logit = math.log(ratio / (1.0 - ratio))
        entropy = -ratio * math.log(ratio) - (1.0 - ratio) * math.log(
            1.0 - ratio
        )
    return {
        "positive": int(positive),
        "total": int(total),
        "positive_ratio": ratio,
        "optimal_constant_logit": logit,
        "minimum_constant_bce": entropy,
    }


def _categorical_constant_optimum(histogram: Sequence[int]) -> dict:
    counts = [int(value) for value in histogram]
    total = sum(counts)
    if total <= 0:
        probabilities = [None for _ in counts]
        entropy = None
    else:
        probabilities = [float(value) / float(total) for value in counts]
        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0.0
        )
    return {
        "histogram": counts,
        "total": total,
        "optimal_constant_probabilities": probabilities,
        "minimum_constant_cross_entropy": entropy,
    }


def _tensor_stats(value: torch.Tensor) -> dict:
    detached = value.detach().float()
    if not torch.isfinite(detached).all():
        raise RuntimeError("non-finite tensor in D0 batch audit")
    return {
        "shape": list(detached.shape),
        "mean": float(detached.mean().item()),
        "std": float(detached.std(unbiased=False).item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
    }


def _gradient_norm(parameter: torch.nn.Parameter) -> Optional[float]:
    if parameter.grad is None:
        return None
    gradient = parameter.grad.detach().float()
    if not torch.isfinite(gradient).all():
        raise RuntimeError("non-finite parameter gradient in D0 batch audit")
    return float(gradient.norm().item())


def _gradient_mass(value: torch.Tensor) -> dict:
    if value.grad is None:
        return {
            "available": False,
            "negative_mass": None,
            "positive_mass": None,
            "zero_count": None,
        }
    gradient = value.grad.detach().float()
    if not torch.isfinite(gradient).all():
        raise RuntimeError("non-finite retained gradient in D0 batch audit")
    return {
        "available": True,
        # For BCE logits, negative derivatives are positive-label pressure and
        # positive derivatives are background pressure.
        "negative_mass": float((-gradient[gradient < 0]).sum().item()),
        "positive_mass": float(gradient[gradient > 0].sum().item()),
        "zero_count": int((gradient == 0).sum().item()),
    }


def _required_absolute_file(value: str, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SystemExit("{} must be absolute: {}".format(name, path))
    if not path.is_file():
        raise SystemExit("{} does not exist: {}".format(name, path))
    return path.resolve()


def _discover_lane(source_run: Path, lane: str) -> Tuple[Path, Path, Path]:
    result_root = source_run / "result"
    if not result_root.is_dir():
        raise SystemExit("source run has no result directory: {}".format(result_root))
    lane_dirs = sorted(
        path
        for path in result_root.iterdir()
        if path.is_dir() and "__{}__".format(lane) in path.name
    )
    if len(lane_dirs) != 1:
        raise SystemExit(
            "expected one {} result directory, found {}".format(
                lane, [str(path) for path in lane_dirs]
            )
        )
    lane_dir = lane_dirs[0]
    checkpoint = lane_dir / "terminal_epoch100.pth"
    opts = lane_dir / "opts.json"
    if not checkpoint.is_file() or not opts.is_file():
        raise SystemExit("incomplete source lane: {}".format(lane_dir))
    return lane_dir, checkpoint, opts


def _load_args(opts_path: Path, lane: str) -> SimpleNamespace:
    payload = _read_json(opts_path)
    expected_birth, expected_owner = ARM_MODES[lane]
    checks = {
        "model_variant": "eventmatr",
        "event_arm": lane,
        "birth_mode": expected_birth,
        "ownership_mode": expected_owner,
        "study_protocol": "matched_study",
        "epochs": 100,
        "p_videos": 1,
        "batch": 64,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise SystemExit(
                "source opts mismatch for {}: expected {!r}, got {!r}".format(
                    key, expected, payload.get(key)
                )
            )
    args = SimpleNamespace(**payload)
    # The model is first reconstructed in its original training configuration
    # for the gradient audit.  The same object is then switched to eval replay.
    args.training = True
    args.event_runtime_during_training = False
    args.make_output = True
    args.dataset = "thumos14"
    args.num_workers = 0
    return args


def _validate_training_only_paths(args: SimpleNamespace) -> None:
    _required_absolute_file(args.video_anno, "video_anno")
    _required_absolute_file(args.video_feature_all_train, "video_feature_all_train")
    test_path = Path(args.video_feature_all_test)
    if test_path.exists():
        raise SystemExit(
            "D0 forbids any mounted test feature path: {}".format(test_path)
        )
    if "LOCKED_TEST_NOT_MOUNTED" not in test_path.name:
        raise SystemExit(
            "D0 requires the locked-test absent sentinel, got {}".format(test_path)
        )


def _assert_chronological_dataset(dataset: THUMOS14Dataset) -> dict:
    previous_video = None
    previous_frame = None
    real_prefixes = 0
    padding_prefixes = 0
    video_transitions = 0
    for video_name, _, end_exclusive in dataset.inputs[0]:
        frame = int(end_exclusive) - 1
        if video_name != previous_video:
            if frame != 0:
                raise RuntimeError(
                    "video {} does not begin at prefix frame 0".format(video_name)
                )
            previous_video = video_name
            previous_frame = frame
            video_transitions += 1
        else:
            if frame != int(previous_frame) + 1:
                raise RuntimeError(
                    "non-chronological prefix for {}: {} after {}".format(
                        video_name, frame, previous_frame
                    )
                )
            previous_frame = frame
        if frame < int(dataset.video_len[video_name]):
            real_prefixes += 1
        else:
            padding_prefixes += 1
    if len(dataset.inputs[0]) % int(dataset.batch) != 0:
        raise RuntimeError(
            "padded replay dataset length {} is not divisible by batch {}".format(
                len(dataset.inputs[0]), dataset.batch
            )
        )
    aligned_single_video_batches = 0
    cross_video_batches = 0
    for start in range(0, len(dataset.inputs[0]), int(dataset.batch)):
        batch_rows = dataset.inputs[0][start : start + int(dataset.batch)]
        batch_videos = {str(row[0]) for row in batch_rows}
        if len(batch_videos) != 1:
            cross_video_batches += 1
            continue
        batch_frames = [int(row[2]) - 1 for row in batch_rows]
        if any(
            right != left + 1
            for left, right in zip(batch_frames[:-1], batch_frames[1:])
        ):
            raise RuntimeError(
                "non-consecutive frames in padded replay batch starting at {}".format(
                    start
                )
            )
        aligned_single_video_batches += 1
    if cross_video_batches:
        raise RuntimeError(
            "official padded replay layout contains {} cross-video batches".format(
                cross_video_batches
            )
        )
    return {
        "video_count": len(dataset.video_list),
        "video_transitions": video_transitions,
        "real_prefixes": real_prefixes,
        "padding_prefixes": padding_prefixes,
        "total_prefixes": len(dataset.inputs[0]),
        "batch_size": int(dataset.batch),
        "batch_boundary_count": len(dataset.inputs[0]) // int(dataset.batch),
        "aligned_single_video_batches": aligned_single_video_batches,
        "cross_video_batches": cross_video_batches,
    }


def _collate_batch(batch, p_videos: int):
    inputs, targets, infos = batch
    return parrallel_collate_fn(inputs, targets, infos, p_videos)


def _find_supervised_batch(dataset: THUMOS14Dataset) -> Tuple[int, List[int]]:
    supervised_index = None
    for index, (video_name, _, end_exclusive) in enumerate(dataset.inputs[0]):
        current_frame = int(end_exclusive) - 1
        if current_frame >= int(dataset.video_len[video_name]):
            continue
        _, valid = dataset._make_event_targets(video_name, current_frame)
        if bool(valid.any().item()):
            supervised_index = index
            break
    if supervised_index is None:
        raise RuntimeError("no real supervised EventMATR prefix exists")
    # The official dataset pads every video to an exact batch multiple.  Use
    # that aligned block instead of a sliding window, so the gradient audit can
    # never mix the tail of one video with the start of another.
    start = (supervised_index // int(dataset.batch)) * int(dataset.batch)
    stop = start + int(dataset.batch)
    indices = list(range(start, stop))
    if len(indices) != dataset.batch or supervised_index not in indices:
        raise RuntimeError("could not form an official-size supervised batch")
    batch_videos = {str(dataset.inputs[0][index][0]) for index in indices}
    if len(batch_videos) != 1:
        raise RuntimeError("supervised D0 gradient batch crosses a video boundary")
    return supervised_index, indices


def _event_target_counts(
    targets: Dict[str, torch.Tensor], infos: dict, num_queries: int
) -> dict:
    real = infos["is_real_prefix"]
    if not torch.is_tensor(real):
        real = torch.as_tensor(real)
    real = real.detach().cpu().bool().reshape(-1)
    valid = targets["event_valid_mask"].detach().cpu().bool()
    rows = targets["event_targets"].detach().cpu()
    valid = valid & real.unsqueeze(1)
    selected = rows[valid]
    state_counts = [0, 0, 0]
    class_histogram: Dict[str, int] = {}
    if selected.numel() > 0:
        state_counts = [
            int(selected[:, column].sum().item()) for column in (5, 6, 7)
        ]
        for class_id in selected[:, 1].long().tolist():
            key = str(int(class_id))
            class_histogram[key] = class_histogram.get(key, 0) + 1
    real_queries = int(real.sum().item()) * int(num_queries)
    valid_rows = int(valid.sum().item())
    return {
        "real_prefixes": int(real.sum().item()),
        "real_queries": real_queries,
        "valid_rows": valid_rows,
        "birth": state_counts[0],
        "alive": state_counts[1],
        "end": state_counts[2],
        "owner_state_histogram": [
            real_queries - valid_rows,
            state_counts[0],
            state_counts[1],
            state_counts[2],
        ],
        "class_histogram": class_histogram,
    }


def _merge_counts(total: dict, update: dict) -> None:
    for key in ("real_prefixes", "real_queries", "valid_rows", "birth", "alive", "end"):
        total[key] = int(total.get(key, 0)) + int(update[key])
    if "owner_state_histogram" not in total:
        total["owner_state_histogram"] = [0, 0, 0, 0]
    total["owner_state_histogram"] = [
        int(left) + int(right)
        for left, right in zip(
            total["owner_state_histogram"], update["owner_state_histogram"]
        )
    ]
    classes = total.setdefault("class_histogram", {})
    for key, value in update["class_histogram"].items():
        classes[key] = int(classes.get(key, 0)) + int(value)


def _batch_gradient_audit(
    args: SimpleNamespace,
    dataset: THUMOS14Dataset,
    model: torch.nn.DataParallel,
    device: torch.device,
) -> dict:
    supervised_index, indices = _find_supervised_batch(dataset)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=dataset.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    inputs, targets, infos = _collate_batch(next(iter(loader)), dataset.p_videos)
    inputs = inputs.to(device)
    targets = {key: value.to(device) for key, value in targets.items()}
    model.train()
    args.training = True
    criterion = build_criterion(args, device)
    criterion.train()
    model.zero_grad(set_to_none=True)
    outputs = model({"inputs": inputs, "infos": infos}, device)
    retained_keys = (
        "event_birth_logits",
        "event_alive_logits",
        "event_end_logits",
        "event_owner_state_logits",
        "event_owner_class_logits",
    )
    for key in retained_keys:
        outputs[key].retain_grad()
    loss_dict = criterion(outputs, targets, infos, device)
    weighted = {
        name: value * criterion.weight_dict[name]
        for name, value in loss_dict.items()
        if name in criterion.weight_dict
    }
    loss = sum(weighted.values())
    if not torch.isfinite(loss):
        raise RuntimeError("D0 gradient audit produced non-finite total loss")
    loss.backward()

    module = model.module
    counts = _event_target_counts(targets, infos, args.num_queries)
    event_distribution = {
        "birth": _binary_constant_optimum(counts["birth"], counts["real_queries"]),
        "alive": _binary_constant_optimum(counts["alive"], counts["real_queries"]),
        "end": _binary_constant_optimum(counts["end"], counts["real_queries"]),
        "owner_state": _categorical_constant_optimum(
            counts["owner_state_histogram"]
        ),
        "class_histogram": counts["class_histogram"],
    }
    receipt = {
        "supervised_index": supervised_index,
        "batch_indices": [indices[0], indices[-1]],
        "input_shape": list(inputs.shape),
        "target_counts": counts,
        "constant_predictor_optima": event_distribution,
        "raw_losses": {
            name: float(value.detach().cpu().item())
            for name, value in sorted(loss_dict.items())
        },
        "weighted_losses": {
            name: float(value.detach().cpu().item())
            for name, value in sorted(weighted.items())
        },
        "total_weighted_loss": float(loss.detach().cpu().item()),
        "logits": {
            key: _tensor_stats(outputs[key]) for key in retained_keys
        },
        "retained_gradient_mass": {
            key: _gradient_mass(outputs[key]) for key in retained_keys
        },
        "parameter_gradient_norms": {
            "shared_feature_reduction_rgb": _gradient_norm(
                module.feature_reduction_rgb.weight
            ),
            "event_transition_state": _gradient_norm(
                module.event_transition_head.state.weight
            ),
            "owner_state": _gradient_norm(
                module.event_owner_decoder.state.weight
            ),
            "owner_cross_attention": _gradient_norm(
                module.event_owner_decoder.cross_attention.in_proj_weight
            ),
        },
    }
    finite_gradient_tensors = 0
    zero_gradient_tensors = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("D0 batch audit found a non-finite model gradient")
        finite_gradient_tensors += 1
        if float(parameter.grad.detach().float().norm().item()) == 0.0:
            zero_gradient_tensors += 1
    receipt["finite_gradient_tensors"] = finite_gradient_tensors
    receipt["zero_gradient_tensors"] = zero_gradient_tensors

    del criterion, outputs, loss_dict, weighted, loss, inputs, targets
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return receipt


def _validate_ledger(model, dataset: THUMOS14Dataset) -> dict:
    module = model.module
    total_rows = 0
    negative_starts = 0
    nonpositive_intervals = 0
    end_after_emit = 0
    emit_after_last_real = 0
    duplicate_event_ids = 0
    sequence_violations = 0
    invalid_classes = 0
    active_after_eos = 0
    rows_by_video: Dict[str, int] = {}
    samples: List[dict] = []
    for video_name in dataset.video_list:
        ledger = list(module.event_memory.ledger(video_name))
        rows_by_video[video_name] = len(ledger)
        total_rows += len(ledger)
        seen = set()
        previous_key = None
        last_real = float(dataset.video_len[video_name] - 1)
        for row in ledger:
            if int(row["event_id"]) in seen:
                duplicate_event_ids += 1
            seen.add(int(row["event_id"]))
            key = (float(row["emit_frame"]), int(row["sequence_id"]))
            if previous_key is not None and key < previous_key:
                sequence_violations += 1
            previous_key = key
            if float(row["start_frame"]) < 0.0:
                negative_starts += 1
            if float(row["end_frame"]) <= float(row["start_frame"]):
                nonpositive_intervals += 1
            if float(row["emit_frame"]) < float(row["end_frame"]):
                end_after_emit += 1
            if float(row["emit_frame"]) > last_real:
                emit_after_last_real += 1
            if not 0 <= int(row["class_id"]) < len(dataset.label_name):
                invalid_classes += 1
            if len(samples) < 20:
                samples.append(dict(row))
        active_after_eos += sum(
            1
            for record in module.event_memory.records(video_name)
            if record.status == "active"
        )
    return {
        "total_rows": total_rows,
        "videos_with_rows": sum(value > 0 for value in rows_by_video.values()),
        "max_rows_per_video": max(rows_by_video.values(), default=0),
        "negative_start_rows": negative_starts,
        "nonpositive_interval_rows": nonpositive_intervals,
        "emit_before_end_rows": end_after_emit,
        "emit_after_last_real_rows": emit_after_last_real,
        "duplicate_event_ids": duplicate_event_ids,
        "sequence_violations": sequence_violations,
        "invalid_class_rows": invalid_classes,
        "active_records_after_eos": active_after_eos,
        "sample_rows": samples,
    }


@torch.no_grad()
def _full_train_replay(
    args: SimpleNamespace,
    dataset: THUMOS14Dataset,
    model: torch.nn.DataParallel,
    device: torch.device,
    output_dir: Path,
    progress_every: int,
) -> dict:
    args.training = False
    model.eval()
    memory_initialize(model, args)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    proposal_txt = output_dir / "proposal_pred_train_replay.txt"
    # Explicit truncation is essential: the inherited writer only touches and
    # appends, which previously made stale-output reuse possible.
    proposal_txt.write_text("", encoding="utf-8")
    proposal_template = str(output_dir / "proposal_{}_train_replay.txt")
    if Path(proposal_template.format("pred")) != proposal_txt:
        raise RuntimeError("D0 proposal writer/checker path contract diverged")

    stats = {
        "event_state_logits": RunningTensorStats(),
        "event_birth_logits": RunningTensorStats(),
        "event_alive_logits": RunningTensorStats(),
        "event_end_logits": RunningTensorStats(),
        "pred_cls": RunningTensorStats(),
    }
    state_argmax = [0, 0, 0, 0]
    runtime = {
        "births": 0,
        "ends": 0,
        "emits": 0,
        "cancellations": 0,
        "capacity_exhaustions": 0,
        "padding_prefixes_ignored": 0,
        "eos_observed": 0,
        "max_active_count": 0,
    }
    supervision: dict = {}
    replay_started = time.time()
    last_video = None
    for batch_index, batch in enumerate(loader):
        inputs, targets, infos = _collate_batch(batch, dataset.p_videos)
        video_names = [str(value) for value in infos["video_name"]]
        if len(set(video_names)) != 1:
            raise RuntimeError(
                "D0 requires each replay batch to contain one chronological video"
            )
        frames = infos["current_frame"].detach().cpu().reshape(-1).tolist()
        if any(
            int(right) != int(left) + 1
            for left, right in zip(frames[:-1], frames[1:])
        ):
            raise RuntimeError("non-consecutive frames inside replay batch")
        if last_video is not None and video_names[0] != last_video and int(frames[0]) != 0:
            raise RuntimeError("new replay video did not begin at frame zero")
        last_video = video_names[-1]

        counts = _event_target_counts(targets, infos, args.num_queries)
        _merge_counts(supervision, counts)
        inputs = inputs.to(device, non_blocking=True)
        model_inputs = {"inputs": inputs, "infos": infos}
        outputs = model(model_inputs, device)
        for key, accumulator in stats.items():
            accumulator.update(outputs[key])
        winners = outputs["event_state_logits"].argmax(dim=-1)
        for state_index in range(4):
            state_argmax[state_index] += int(
                (winners == state_index).sum().item()
            )
        runtime["births"] += int(outputs["event_new_birth_mask"].sum().item())
        runtime["ends"] += int(outputs["event_ended_mask"].sum().item())
        runtime["emits"] += int(outputs["event_emitted_mask"].sum().item())
        runtime["cancellations"] += int(
            outputs["event_cancelled_mask"].sum().item()
        )
        runtime["capacity_exhaustions"] += int(
            outputs["event_runtime_capacity_exhaustions"].sum().item()
        )
        runtime["padding_prefixes_ignored"] += int(
            outputs["event_padding_prefixes_ignored"].sum().item()
        )
        runtime["eos_observed"] += int(
            outputs["event_eos_observed"].sum().item()
        )
        runtime["max_active_count"] = max(
            runtime["max_active_count"],
            int(outputs["event_active_count"].max().item()),
        )
        write_model_predictions(
            args,
            model,
            infos,
            outputs,
            proposal_template,
            dataset.label_name,
        )
        if progress_every > 0 and (
            (batch_index + 1) % progress_every == 0
            or batch_index + 1 == len(loader)
        ):
            print(
                "D0_PROGRESS lane={} batch={}/{} births={} ends={} emits={} active_max={}".format(
                    args.event_arm,
                    batch_index + 1,
                    len(loader),
                    runtime["births"],
                    runtime["ends"],
                    runtime["emits"],
                    runtime["max_active_count"],
                ),
                flush=True,
            )
        del inputs, outputs

    ledger = _validate_ledger(model, dataset)
    proposal_lines = sum(
        1
        for line in proposal_txt.read_text(
            encoding="utf-8", errors="strict"
        ).splitlines()
        if line.strip()
    )
    if proposal_lines != ledger["total_rows"]:
        raise RuntimeError(
            "proposal/ledger mismatch: {} lines vs {} rows".format(
                proposal_lines, ledger["total_rows"]
            )
        )
    proposal_json = output_dir / "proposal_pred_train_replay.json"
    result_dict = online_nms(args, str(proposal_txt), dataset)
    proposal_json.write_text(
        json.dumps(
            {"version": "VERSION 1", "results": result_dict, "external_data": {}},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tiou_thresholds = np.linspace(0.3, 0.70, 5)
    maps = evaluation_detection(
        args,
        str(proposal_json),
        subset="train",
        tiou_thresholds=tiou_thresholds,
        verbose=False,
    )
    supervision_receipt = {
        "counts": supervision,
        "constant_predictor_optima": {
            "birth": _binary_constant_optimum(
                supervision["birth"], supervision["real_queries"]
            ),
            "alive": _binary_constant_optimum(
                supervision["alive"], supervision["real_queries"]
            ),
            "end": _binary_constant_optimum(
                supervision["end"], supervision["real_queries"]
            ),
            "owner_state": _categorical_constant_optimum(
                supervision["owner_state_histogram"]
            ),
            "class_histogram": supervision["class_histogram"],
        },
    }
    if runtime["births"] == 0:
        operational_verdict = "learned_runtime_zero_births"
    elif runtime["emits"] == 0:
        operational_verdict = "nonzero_births_but_zero_final_emissions"
    else:
        operational_verdict = "runtime_emits_final_intervals"
    return {
        "batches": len(loader),
        "elapsed_seconds": time.time() - replay_started,
        "proposal_txt": str(proposal_txt),
        "proposal_json": str(proposal_json),
        "proposal_lines": proposal_lines,
        "runtime_counts": runtime,
        "operational_verdict": operational_verdict,
        "ledger_audit": ledger,
        "logit_distributions": {
            key: accumulator.receipt() for key, accumulator in stats.items()
        },
        "event_state_argmax_histogram": {
            str(index): count for index, count in enumerate(state_argmax)
        },
        "supervision_distribution": supervision_receipt,
        "train_replay_map_percent": {
            "average_0.3_0.7": float(np.asarray(maps).mean()),
            "0.3": float(maps[0]),
            "0.4": float(maps[1]),
            "0.5": float(maps[2]),
            "0.6": float(maps[3]),
            "0.7": float(maps[4]),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--lane", required=True, choices=tuple(ARM_MODES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-training-commit", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--diagnostic-commit", required=True)
    parser.add_argument("--diagnostic-tree", required=True)
    parser.add_argument("--progress-every", default=100, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.source_run = Path(args.source_run).expanduser().resolve()
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    if not args.source_run.is_dir():
        raise SystemExit("source run does not exist: {}".format(args.source_run))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("D0 output directory must be fresh: {}".format(args.output_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def main() -> None:
    cli = _parse_args()
    if cli.device != "cuda":
        raise SystemExit("formal D0 audit requires --device cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise SystemExit("D0 requires exactly one visible GPU per process")

    source_identity_path = cli.source_run / "source_identity.json"
    source_identity = _read_json(source_identity_path)
    if source_identity.get("commit") != cli.expected_training_commit:
        raise SystemExit(
            "source training commit mismatch: {}".format(source_identity.get("commit"))
        )
    if source_identity.get("tree") != cli.expected_training_tree:
        raise SystemExit(
            "source training tree mismatch: {}".format(source_identity.get("tree"))
        )
    if source_identity.get("clean") is not True:
        raise SystemExit("source training identity is not clean")
    lane_dir, checkpoint_path, opts_path = _discover_lane(
        cli.source_run, cli.lane
    )
    args = _load_args(opts_path, cli.lane)
    _validate_training_only_paths(args)

    seed = int(args.random_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    dataset = THUMOS14Dataset(args, subset="train")
    chronology = _assert_chronological_dataset(dataset)

    checkpoint_sha256 = _sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if int(checkpoint.get("epoch", -1)) != 100:
        raise SystemExit("D0 requires a terminal epoch-100 checkpoint")
    if checkpoint.get("model_variant") != "eventmatr":
        raise SystemExit("checkpoint is not EventMATR")
    model = torch.nn.DataParallel(build_model(args))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model = model.to(device)

    gradient_audit = _batch_gradient_audit(args, dataset, model, device)
    memory_initialize(model, args)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    replay = _full_train_replay(
        args,
        dataset,
        model,
        device,
        cli.output_dir,
        cli.progress_every,
    )

    receipt = {
        "status": "PASS",
        "protocol_id": "eventmatr_v1_d0_checkpoint_replay_v1",
        "scientific_scope": (
            "diagnostic replay on official validation/train features; "
            "not a locked-test or raw-RGB result"
        ),
        "test_access": False,
        "checkpoint_updated": False,
        "strict_causal_paper_result_valid": False,
        "strict_causal_invalid_reasons": [
            "EventMATR-v1 runtime consumes true_duration",
            "EventMATR-v1 runtime consumes offline EOS metadata",
            "frame_to_time is derived from complete-video metadata",
        ],
        "training_zero_interpretation": (
            "matched-study training ran model.train() with "
            "event_runtime_during_training=false, so empty training proposal "
            "files cannot establish learned all-background collapse"
        ),
        "diagnostic_source": {
            "commit": cli.diagnostic_commit,
            "tree": cli.diagnostic_tree,
        },
        "source_training": {
            "run": str(cli.source_run),
            "identity": source_identity,
            "lane": cli.lane,
            "lane_dir": str(lane_dir),
            "opts_json": str(opts_path),
            "opts_sha256": _sha256(opts_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "epoch": 100,
        },
        "dataset": {
            "subset": "train",
            "annotation": str(Path(args.video_anno).resolve()),
            "train_feature": str(Path(args.video_feature_all_train).resolve()),
            "locked_test_sentinel": str(Path(args.video_feature_all_test)),
            "chronology": chronology,
        },
        "gradient_audit": gradient_audit,
        "eval_full_prefix_replay": replay,
        "device": {
            "name": torch.cuda.get_device_name(0),
            "visible_gpu_count": torch.cuda.device_count(),
        },
    }
    receipt_path = cli.output_dir / "d0_audit_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "D0_RESULT lane={} verdict={} births={} ends={} emits={} map={:.6f}".format(
            cli.lane,
            replay["operational_verdict"],
            replay["runtime_counts"]["births"],
            replay["runtime_counts"]["ends"],
            replay["runtime_counts"]["emits"],
            replay["train_replay_map_percent"]["average_0.3_0.7"],
        ),
        flush=True,
    )
    print("D0_RECEIPT {}".format(receipt_path), flush=True)


if __name__ == "__main__":
    main()
