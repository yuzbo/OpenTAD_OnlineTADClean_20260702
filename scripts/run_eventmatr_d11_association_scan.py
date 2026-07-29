"""Strict-causal, train-only association-opportunity scan for D1.1.

The model runs predicted-only and never receives ground truth.  Prefix-visible
training targets are consulted only after each forward pass to decompose the
causal association barrier.  The checkpoint is read-only and the locked test
feature path must remain an absent sentinel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import THUMOS14Dataset  # noqa: E402
from models import build_model  # noqa: E402
from models.event_memory import (  # noqa: E402
    CausalTemporalHistory,
    causal_single_assignment,
    temporal_viterbi_assignment,
)
from on_tal_task import (  # noqa: E402
    D1_RUNTIME_FORBIDDEN_MODEL_INFO,
    D11_CHECKPOINT_SCHEMA,
    make_model_inputs,
    validate_d1_checkpoint_compatibility,
)
from util.utils import memory_initialize, parrallel_collate_fn  # noqa: E402


AUDIT_FIELDS = (
    "predicted_query_count",
    "target_count",
    "pair_count",
    "class_mismatch_pair_count",
    "start_distance_reject_pair_count",
    "admissible_pair_count",
)
SCORE_QUANTILES = (
    0.0,
    0.001,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.99,
    0.999,
    1.0,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--options", required=True, type=Path)
    parser.add_argument("--source-identity", required=True, type=Path)
    parser.add_argument("--expected-training-source-commit", required=True)
    parser.add_argument("--expected-training-source-tree", required=True)
    parser.add_argument(
        "--one-epoch-mechanism-gate-status",
        required=True,
        choices=("FAIL_UNCHANGED", "PASS_TRAIN_ONLY"),
    )
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-options-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", default=0, type=int)
    args = parser.parse_args()
    for field in ("checkpoint", "options", "source_identity"):
        path = getattr(args, field).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"{field} does not exist: {path}")
        setattr(args, field, path)
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
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


def _validate_inputs(
    options: dict,
    checkpoint: dict,
    identity: dict,
    *,
    expected_training_source_commit: str,
    expected_training_source_tree: str,
    one_epoch_mechanism_gate_status: str,
) -> None:
    expected_options = {
        "study_protocol": "d11_mechanism",
        "model_variant": "eventmatr",
        "event_lifecycle_version": "d1_censored",
        "event_d1_lane": "th",
        "epochs": 1,
        "random_seed": 52,
        "batch": 64,
        "p_videos": 1,
        "event_birth_logit_threshold": None,
        "event_end_logit_threshold": None,
        "event_resource_limit": 0,
    }
    for field, expected in expected_options.items():
        if options.get(field) != expected:
            raise RuntimeError(
                f"association scan option {field} mismatch: "
                f"{options.get(field)!r} != {expected!r}"
            )
    locked_test = Path(options["video_feature_all_test"]).expanduser().resolve()
    if locked_test.exists() or locked_test.name != "LOCKED_TEST_NOT_MOUNTED.pickle":
        raise RuntimeError("association scan requires the absent locked-test sentinel")
    expected_checkpoint = {
        "epoch": 1,
        "study_protocol": "d11_mechanism",
        "model_variant": "eventmatr",
        "checkpoint_schema": D11_CHECKPOINT_SCHEMA,
        "event_lifecycle_version": "d1_censored",
        "event_d1_lane": "th",
        "owner_state_count": 3,
    }
    for field, expected in expected_checkpoint.items():
        if checkpoint.get(field) != expected:
            raise RuntimeError(
                f"association scan checkpoint {field} mismatch: "
                f"{checkpoint.get(field)!r} != {expected!r}"
            )
    if identity.get("status") != "PASS" or identity.get("clean") is not True:
        raise RuntimeError("training start source identity is not PASS/clean")
    if identity.get("commit") != expected_training_source_commit:
        raise RuntimeError("unexpected D1.1 training commit")
    if identity.get("tree") != expected_training_source_tree:
        raise RuntimeError("unexpected D1.1 training tree")
    effective_dose = bool(options.get("d11_effective_dose", False))
    if (
        one_epoch_mechanism_gate_status == "PASS_TRAIN_ONLY"
        and not effective_dose
    ):
        raise RuntimeError("passing D1.1 training receipt lacks effective dose")
    if one_epoch_mechanism_gate_status == "FAIL_UNCHANGED" and effective_dose:
        raise RuntimeError("failed legacy D1.1 scan unexpectedly used effective dose")
    smoke = identity.get("smoke")
    if (
        not isinstance(smoke, dict)
        or smoke.get("status") != "PASS"
        or smoke.get("test_access") is not False
    ):
        raise RuntimeError("training identity lacks a passing no-test smoke gate")


def _new_counts() -> dict:
    counts = {
        "real_prefix_count": 0,
        "padding_prefix_count": 0,
        "padding_noop_count": 0,
        "visible_target_prefix_count": 0,
        "visible_birth_target_prefix_count": 0,
        "visible_recovery_target_prefix_count": 0,
        "predicted_birth_prefix_count": 0,
        "predicted_start_active_prefix_count": 0,
        "cooccurrence_prefix_count": 0,
        "visible_target_count": 0,
        "visible_birth_target_count": 0,
        "visible_recovery_target_count": 0,
        "predicted_birth_query_count": 0,
        "predicted_start_active_query_count": 0,
        "pair_count": 0,
        "class_mismatch_pair_count": 0,
        "start_distance_reject_pair_count": 0,
        "admissible_pair_count": 0,
        "ambiguous_query_count": 0,
        "ambiguous_target_count": 0,
        "assignment_count": 0,
        "runtime_birth_count": 0,
        "runtime_cancel_count": 0,
        "runtime_end_count": 0,
        "runtime_emit_count": 0,
        "runtime_reacquisition_count": 0,
        "runtime_capacity_exhaustion_count": 0,
        "observed_eos_count": 0,
    }
    return counts


def _add_counts(total: dict, row: dict) -> None:
    for key, value in row.items():
        total[key] = int(total.get(key, 0)) + int(value)


def _validate_count_closure(counts: dict, label: str) -> None:
    if (
        counts["pair_count"]
        != counts["class_mismatch_pair_count"]
        + counts["start_distance_reject_pair_count"]
        + counts["admissible_pair_count"]
    ):
        raise RuntimeError(f"association barrier accounting does not close for {label}")
    if counts["visible_target_count"] != (
        counts["visible_birth_target_count"]
        + counts["visible_recovery_target_count"]
    ):
        raise RuntimeError(f"visible target accounting does not close for {label}")
    if counts["predicted_birth_query_count"] != counts["runtime_birth_count"]:
        raise RuntimeError(f"predicted birth accounting does not close for {label}")
    if counts["padding_prefix_count"] != counts["padding_noop_count"]:
        raise RuntimeError(f"padding no-op accounting does not close for {label}")


def _metadata_values(value, *, batch_size: int, label: str) -> list:
    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if len(values) != batch_size:
        raise RuntimeError(
            f"{label} metadata has {len(values)} rows for physical batch {batch_size}"
        )
    return values


def _score_summary(values, *, label: str) -> dict:
    if isinstance(values, list) and values and isinstance(values[0], np.ndarray):
        array = np.concatenate(values)
    else:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0}
    if not np.isfinite(array).all():
        raise RuntimeError(f"non-finite score diagnostic: {label}")
    quantiles = np.quantile(array, SCORE_QUANTILES)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "positive_count": int((array > 0.0).sum()),
        "zero_count": int((array == 0.0).sum()),
        "quantiles": {
            format(float(level), ".3g"): float(value)
            for level, value in zip(SCORE_QUANTILES, quantiles)
        },
    }


def _rank_summary(ranks: list[int], *, query_count: int) -> dict:
    if not ranks:
        return {"count": 0}
    values = np.asarray(ranks, dtype=np.int64)
    if (values < 1).any() or (values > int(query_count)).any():
        raise RuntimeError("oracle-path query rank is outside query bandwidth")
    return {
        "count": int(values.size),
        "mean_rank": float(values.mean()),
        "mean_reciprocal_rank": float((1.0 / values).mean()),
        "top1_count": int((values <= 1).sum()),
        "top1_rate": float((values <= 1).mean()),
        "top3_count": int((values <= min(3, query_count)).sum()),
        "top3_rate": float((values <= min(3, query_count)).mean()),
        "top5_count": int((values <= min(5, query_count)).sum()),
        "top5_rate": float((values <= min(5, query_count)).mean()),
    }


def main() -> None:
    cli = _parse_args()
    if cli.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("formal D1.1 association scan requires one CUDA device")
    if torch.cuda.device_count() != 1:
        raise SystemExit("formal D1.1 association scan requires one visible GPU")
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise SystemExit(f"association scan source is dirty:\n{status}")

    options = json.loads(cli.options.read_text(encoding="utf-8"))
    identity = json.loads(cli.source_identity.read_text(encoding="utf-8"))
    checkpoint_stat_before = {
        "bytes": cli.checkpoint.stat().st_size,
        "mtime_ns": cli.checkpoint.stat().st_mtime_ns,
        "sha256": _sha256(cli.checkpoint),
    }
    if checkpoint_stat_before["sha256"] != cli.expected_checkpoint_sha256:
        raise RuntimeError("association scan checkpoint SHA-256 mismatch")
    options_sha256 = _sha256(cli.options)
    if options_sha256 != cli.expected_options_sha256:
        raise RuntimeError("association scan options SHA-256 mismatch")
    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    _validate_inputs(
        options,
        checkpoint,
        identity,
        expected_training_source_commit=cli.expected_training_source_commit,
        expected_training_source_tree=cli.expected_training_source_tree,
        one_epoch_mechanism_gate_status=cli.one_epoch_mechanism_gate_status,
    )

    random.seed(52)
    np.random.seed(52)
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    options["training"] = False
    options["mode"] = "eval"
    options["make_output"] = False
    options["load_model"] = False
    options["num_workers"] = int(options.get("num_workers", 4))
    args = SimpleNamespace(**options)
    device = torch.device("cuda")

    dataset_cache_paths = {
        "video_len": Path(args.video_len_file.format("train")).expanduser().resolve(),
        "proposal_labels": Path(
            args.ontal_label_file.format(
                "train",
                args.num_frame,
                args.num_queries,
                args.detect_len,
                args.anti_len,
                args.max_memory_len,
                args.p_videos,
            )
        )
        .expanduser()
        .resolve(),
    }
    for cache_name, cache_path in dataset_cache_paths.items():
        if not cache_path.is_file():
            raise RuntimeError(
                f"read-only association scan requires existing {cache_name} cache: "
                f"{cache_path}"
            )
    dataset_cache_stats_before = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for name, path in dataset_cache_paths.items()
    }
    dataset = THUMOS14Dataset(args, subset="train")
    if len(dataset.video_list) != 200 and cli.max_batches == 0:
        raise RuntimeError(
            f"complete scan expected 200 train/validation videos, "
            f"got {len(dataset.video_list)}"
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
    validate_d1_checkpoint_compatibility(checkpoint, model, args)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model.eval()
    memory_initialize(model, args)

    history = CausalTemporalHistory(window_size=int(args.num_frame))
    totals = _new_counts()
    per_video: dict[str, dict] = {}
    processed_batches = 0
    forward_batches = 0
    last_real_frame: dict[str, float] = {}
    last_physical_frame: dict[str, float] = {}
    closed_streams: set[str] = set()
    score_chunks: dict[str, list[np.ndarray]] = {
        "all_query_margin": [],
        "prefix_max_margin": [],
    }
    score_values: dict[str, list[float]] = {
        "hardest_background_margin": [],
        "birth_oracle_path_margin": [],
        "birth_competing_query_margin": [],
        "birth_oracle_path_start_distance": [],
        "birth_oracle_path_interval_probability": [],
        "birth_oracle_path_pre_survival_nll": [],
        "birth_oracle_path_interval_event_nll": [],
        "birth_oracle_path_pre_interval_margin": [],
        "birth_oracle_path_in_interval_margin": [],
        "alive_opportunity_oracle_path_margin": [],
        "alive_opportunity_competing_query_margin": [],
        "alive_opportunity_oracle_path_start_distance": [],
    }
    birth_oracle_ranks: list[int] = []
    alive_opportunity_oracle_ranks: list[int] = []
    birth_pairwise_preference_rates: list[float] = []
    alive_opportunity_pairwise_preference_rates: list[float] = []
    birth_oracle_state_winners = {str(index): 0 for index in range(4)}
    alive_opportunity_oracle_state_winners = {
        str(index): 0 for index in range(4)
    }
    compatibility_counts = {
        "birth_class_match_count": 0,
        "birth_start_distance_pass_count": 0,
        "birth_joint_class_and_distance_pass_count": 0,
        "alive_opportunity_class_match_count": 0,
        "alive_opportunity_start_distance_pass_count": 0,
        "alive_opportunity_joint_class_and_distance_pass_count": 0,
        "birth_interval_fallback_count": 0,
        "birth_interval_probability_clamp_count": 0,
    }
    started_at = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for batch_index, (features, targets, infos) in enumerate(loader):
            if cli.max_batches and batch_index >= cli.max_batches:
                break
            processed_batches += 1
            features, targets, infos = parrallel_collate_fn(
                features, targets, infos, args.p_videos
            )
            batch_size = int(features.size(0))
            if batch_size != int(args.batch):
                raise RuntimeError(
                    f"frozen MATR requires physical batch {args.batch}, got {batch_size}"
                )
            names = [
                str(value)
                for value in _metadata_values(
                    infos["video_name"],
                    batch_size=batch_size,
                    label="video_name",
                )
            ]
            frames = [
                float(value)
                for value in _metadata_values(
                    infos["current_frame"],
                    batch_size=batch_size,
                    label="current_frame",
                )
            ]
            real_flags = [
                bool(value)
                for value in _metadata_values(
                    infos["is_real_prefix"],
                    batch_size=batch_size,
                    label="is_real_prefix",
                )
            ]
            eos_flags = [
                bool(value)
                for value in _metadata_values(
                    infos["is_eos"],
                    batch_size=batch_size,
                    label="is_eos",
                )
            ]
            for row_index, (video_name, frame, is_real, is_eos) in enumerate(
                zip(names, frames, real_flags, eos_flags)
            ):
                previous_physical = last_physical_frame.get(video_name)
                if previous_physical is not None and frame <= previous_physical:
                    raise RuntimeError(
                        f"non-monotonic physical row for {video_name}: "
                        f"{frame} after {previous_physical}"
                    )
                last_physical_frame[video_name] = frame
                if is_eos and not is_real:
                    raise RuntimeError(
                        f"padding row {row_index} for {video_name} cannot signal EOS"
                    )
                if is_real:
                    if video_name in closed_streams:
                        raise RuntimeError(
                            f"real prefix for {video_name} appeared after observed EOS"
                        )
                    previous_real = last_real_frame.get(video_name)
                    if previous_real is not None and frame <= previous_real:
                        raise RuntimeError(
                            f"non-monotonic real prefix for {video_name}: "
                            f"{frame} after {previous_real}"
                        )
                    last_real_frame[video_name] = frame
                    if is_eos:
                        closed_streams.add(video_name)
                elif video_name not in closed_streams:
                    raise RuntimeError(
                        f"padding for {video_name} appeared before observed EOS"
                    )

            forward_batches += 1
            features = features.to(device, non_blocking=True)
            target_tensors = {
                key: value.to(device, non_blocking=True)
                for key, value in targets.items()
            }
            if "segment_flag" not in infos:
                raise RuntimeError(
                    "diagnostic loader omitted the inherited supervision flag"
                )
            stripped_boundary_fields = (
                D1_RUNTIME_FORBIDDEN_MODEL_INFO.intersection(infos)
            )
            if stripped_boundary_fields != D1_RUNTIME_FORBIDDEN_MODEL_INFO:
                raise RuntimeError(
                    "diagnostic loader omitted fields required to audit stripping: "
                    f"{sorted(D1_RUNTIME_FORBIDDEN_MODEL_INFO - stripped_boundary_fields)}"
                )
            payload = make_model_inputs(args, features, infos)
            expected_model_info_keys = (
                set(infos) - D1_RUNTIME_FORBIDDEN_MODEL_INFO
            )
            if set(payload["infos"]) != expected_model_info_keys:
                raise RuntimeError(
                    "runtime model-boundary stripping contract changed: "
                    f"{sorted(payload['infos'])} != "
                    f"{sorted(expected_model_info_keys)}"
                )
            forbidden = D1_RUNTIME_FORBIDDEN_MODEL_INFO.intersection(payload["infos"])
            if forbidden:
                raise RuntimeError(
                    f"future/full-video metadata reached model: {sorted(forbidden)}"
                )
            if "event_targets" in payload or "event_valid_mask" in payload:
                raise RuntimeError("ground truth reached the predicted-only model payload")
            outputs = model(payload, device)

            state_logits = outputs["event_state_logits"].detach()
            birth_logits = outputs["event_birth_logits"].detach()
            class_logits = outputs["pred_cls"].detach()
            query_features = outputs["event_query_features"].detach()
            start_frames = outputs["event_candidate_start_frames"].detach()
            predicted_birth_mask = outputs["event_new_birth_mask"].detach().bool()
            predicted_start_active_mask = birth_logits > 0.0
            valid_mask = target_tensors["event_valid_mask"].detach().bool()
            event_targets = target_tensors["event_targets"].detach()

            runtime_fields = {
                "runtime_birth_count": outputs["event_birth_count"],
                "runtime_cancel_count": outputs["event_cancellation_count"],
                "runtime_end_count": outputs["event_end_count"],
                "runtime_emit_count": outputs["event_emit_count"],
                "runtime_reacquisition_count": outputs[
                    "event_reacquisition_count"
                ],
                "runtime_capacity_exhaustion_count": outputs[
                    "event_runtime_capacity_exhaustions"
                ],
                "observed_eos_count": outputs["event_eos_observed"],
            }
            padding_ignored = outputs["event_padding_prefixes_ignored"]
            transition_masks = {
                "birth": outputs["event_new_birth_mask"],
                "end": outputs["event_ended_mask"],
                "emit": outputs["event_emitted_mask"],
                "cancel": outputs["event_cancelled_mask"],
            }
            birth_margin_rows = (
                birth_logits.detach().cpu().to(torch.float64).numpy()
            )
            state_winner_rows = (
                state_logits.detach().argmax(dim=-1).cpu().numpy()
            )
            foreground_class_winner_rows = (
                class_logits.detach()[..., :-1].argmax(dim=-1).cpu().numpy()
            )
            candidate_start_rows = (
                start_frames.detach().cpu().to(torch.float64).numpy()
            )
            real_row_indices = np.asarray(
                [
                    row_index
                    for row_index, is_real in enumerate(real_flags)
                    if is_real
                ],
                dtype=np.int64,
            )
            if real_row_indices.size:
                real_birth_margins = birth_margin_rows[real_row_indices]
                score_chunks["all_query_margin"].append(
                    real_birth_margins.reshape(-1)
                )
                score_chunks["prefix_max_margin"].append(
                    real_birth_margins.max(axis=1)
                )

            for row_index, video_name in enumerate(names):
                if not real_flags[row_index]:
                    if int(padding_ignored[row_index].item()) != 1:
                        raise RuntimeError(
                            f"padding row for {video_name} was not ignored exactly once"
                        )
                    nonzero_runtime = {
                        field: int(tensor[row_index].item())
                        for field, tensor in runtime_fields.items()
                        if int(tensor[row_index].item()) != 0
                    }
                    nonzero_transitions = {
                        field: int(tensor[row_index].sum().item())
                        for field, tensor in transition_masks.items()
                        if bool(tensor[row_index].any().item())
                    }
                    if nonzero_runtime or nonzero_transitions:
                        raise RuntimeError(
                            f"padding row mutated lifecycle for {video_name}: "
                            f"runtime={nonzero_runtime}, transitions={nonzero_transitions}"
                        )
                    row_counts = _new_counts()
                    row_counts["padding_prefix_count"] = 1
                    row_counts["padding_noop_count"] = 1
                    _add_counts(totals, row_counts)
                    video_counts = per_video.setdefault(video_name, _new_counts())
                    _add_counts(video_counts, row_counts)
                    continue
                if int(padding_ignored[row_index].item()) != 0:
                    raise RuntimeError(
                        f"real prefix for {video_name} was treated as padding"
                    )
                if int(runtime_fields["observed_eos_count"][row_index].item()) != int(
                    eos_flags[row_index]
                ):
                    raise RuntimeError(
                        f"runtime EOS observation disagrees with current signal for "
                        f"{video_name} at frame {frames[row_index]}"
                    )
                frame = float(frames[row_index])
                history.append(
                    video_name=video_name,
                    frame=frame,
                    state_logits=state_logits[row_index],
                    birth_logits=birth_logits[row_index],
                    class_logits=class_logits[row_index],
                    query_features=query_features[row_index],
                )
                rows = event_targets[row_index][valid_mask[row_index]]
                target_specs = []
                birth_terminal_queries: set[int] = set()
                visible_birth_target_count = 0
                visible_alive_opportunity_count = 0
                for row in rows:
                    is_birth = bool(row[5].item())
                    is_alive = bool(row[6].item())
                    if not (is_birth or is_alive):
                        continue
                    start_frame = float(row[2].item())
                    window = history.window(
                        video_name,
                        lower=max(0.0, start_frame - int(args.num_frame) + 1.0),
                        upper=frame,
                    )
                    if not window:
                        raise RuntimeError(
                            "post-forward scan omitted the current causal prefix"
                        )
                    path = temporal_viterbi_assignment(
                        torch.stack(
                            [entry["state_logits"] for entry in window], dim=0
                        ),
                        torch.stack(
                            [entry["class_logits"] for entry in window], dim=0
                        ),
                        torch.stack(
                            [entry["query_features"] for entry in window], dim=0
                        ),
                        int(row[1].item()),
                    )
                    if not path:
                        continue
                    terminal_query = int(path[-1])
                    row_margins = birth_margin_rows[row_index]
                    oracle_margin = float(row_margins[terminal_query])
                    competing_margins = np.delete(row_margins, terminal_query)
                    rank = 1 + int((competing_margins > oracle_margin).sum())
                    pairwise_rank = float(
                        (
                            (oracle_margin > competing_margins).astype(np.float64)
                            + 0.5
                            * (oracle_margin == competing_margins).astype(np.float64)
                        ).mean()
                    )
                    target_class = int(row[1].item())
                    class_match = int(
                        foreground_class_winner_rows[row_index, terminal_query]
                    ) == target_class
                    start_distance = abs(
                        float(candidate_start_rows[row_index, terminal_query])
                        - start_frame
                    )
                    start_distance_pass = start_distance <= float(args.num_frame)
                    state_winner = str(
                        int(state_winner_rows[row_index, terminal_query])
                    )
                    if is_birth:
                        birth_terminal_queries.add(terminal_query)
                        score_values["birth_oracle_path_margin"].append(
                            oracle_margin
                        )
                        score_values["birth_competing_query_margin"].extend(
                            float(value) for value in competing_margins
                        )
                        score_values[
                            "birth_oracle_path_start_distance"
                        ].append(start_distance)
                        birth_oracle_ranks.append(rank)
                        birth_pairwise_preference_rates.append(pairwise_rank)
                        birth_oracle_state_winners[state_winner] += 1
                        compatibility_counts["birth_class_match_count"] += int(
                            class_match
                        )
                        compatibility_counts[
                            "birth_start_distance_pass_count"
                        ] += int(start_distance_pass)
                        compatibility_counts[
                            "birth_joint_class_and_distance_pass_count"
                        ] += int(class_match and start_distance_pass)

                        selected = torch.stack(
                            [
                                entry["birth_logits"][int(query_index)]
                                for entry, query_index in zip(window, path)
                            ],
                            dim=0,
                        )
                        selected_frames = torch.tensor(
                            [float(entry["frame"]) for entry in window],
                            device=selected.device,
                            dtype=selected.dtype,
                        )
                        interval_left = float(np.floor(start_frame))
                        interval_right = float(np.ceil(start_frame))
                        pre_birth = selected_frames < interval_left
                        in_interval = (selected_frames >= interval_left) & (
                            selected_frames <= interval_right
                        )
                        if not bool(in_interval.any().item()):
                            in_interval[-1] = True
                            pre_birth[-1] = False
                            compatibility_counts[
                                "birth_interval_fallback_count"
                            ] += 1
                        pre_survival_nll = F.softplus(selected[pre_birth]).sum()
                        log_interval_survival = F.logsigmoid(
                            -selected[in_interval]
                        ).sum()
                        raw_interval_probability = -torch.expm1(
                            log_interval_survival
                        )
                        if float(raw_interval_probability.item()) < 1e-8:
                            compatibility_counts[
                                "birth_interval_probability_clamp_count"
                            ] += 1
                        interval_probability = raw_interval_probability.clamp_min(
                            1e-8
                        )
                        event_nll = (
                            pre_survival_nll - interval_probability.log()
                        )
                        score_values[
                            "birth_oracle_path_interval_probability"
                        ].append(float(interval_probability.item()))
                        score_values[
                            "birth_oracle_path_pre_survival_nll"
                        ].append(float(pre_survival_nll.item()))
                        score_values[
                            "birth_oracle_path_interval_event_nll"
                        ].append(float(event_nll.item()))
                        score_values[
                            "birth_oracle_path_pre_interval_margin"
                        ].extend(
                            float(value)
                            for value in selected[pre_birth]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        score_values[
                            "birth_oracle_path_in_interval_margin"
                        ].extend(
                            float(value)
                            for value in selected[in_interval]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                    else:
                        score_values[
                            "alive_opportunity_oracle_path_margin"
                        ].append(oracle_margin)
                        score_values[
                            "alive_opportunity_competing_query_margin"
                        ].extend(
                            float(value) for value in competing_margins
                        )
                        score_values[
                            "alive_opportunity_oracle_path_start_distance"
                        ].append(start_distance)
                        alive_opportunity_oracle_ranks.append(rank)
                        alive_opportunity_pairwise_preference_rates.append(
                            pairwise_rank
                        )
                        alive_opportunity_oracle_state_winners[state_winner] += 1
                        compatibility_counts[
                            "alive_opportunity_class_match_count"
                        ] += int(class_match)
                        compatibility_counts[
                            "alive_opportunity_start_distance_pass_count"
                        ] += int(start_distance_pass)
                        compatibility_counts[
                            "alive_opportunity_joint_class_and_distance_pass_count"
                        ] += int(class_match and start_distance_pass)

                    visible_birth_target_count += int(is_birth)
                    visible_alive_opportunity_count += int(is_alive)
                    target_specs.append(
                        {
                            "target_event_id": int(row[0].item()),
                            "class_id": target_class,
                            "start_frame": start_frame,
                            "anchor_feature": window[-1]["query_features"][
                                terminal_query
                            ],
                        }
                    )

                available_background_queries = [
                    query_index
                    for query_index in range(birth_margin_rows.shape[1])
                    if query_index not in birth_terminal_queries
                ]
                if available_background_queries:
                    score_values["hardest_background_margin"].append(
                        float(
                            birth_margin_rows[
                                row_index, available_background_queries
                            ].max()
                        )
                    )

                predicted_queries = tuple(
                    int(index)
                    for index in predicted_birth_mask[row_index]
                    .nonzero(as_tuple=False)
                    .reshape(-1)
                    .tolist()
                )
                predicted_start_active_count = int(
                    predicted_start_active_mask[row_index].sum().item()
                )
                association = causal_single_assignment(
                    predicted_query_indices=predicted_queries,
                    candidate_start_frames=start_frames[row_index],
                    class_logits=class_logits[row_index],
                    query_features=query_features[row_index],
                    target_specs=target_specs,
                    max_start_distance=float(args.num_frame),
                )
                row_counts = _new_counts()
                row_counts["real_prefix_count"] = 1
                row_counts["visible_target_prefix_count"] = int(
                    bool(target_specs)
                )
                row_counts["visible_birth_target_prefix_count"] = int(
                    visible_birth_target_count > 0
                )
                row_counts["visible_recovery_target_prefix_count"] = int(
                    visible_alive_opportunity_count > 0
                )
                row_counts["predicted_birth_prefix_count"] = int(
                    bool(predicted_queries)
                )
                row_counts["predicted_start_active_prefix_count"] = int(
                    predicted_start_active_count > 0
                )
                row_counts["cooccurrence_prefix_count"] = int(
                    bool(target_specs) and bool(predicted_queries)
                )
                for field in AUDIT_FIELDS:
                    output_field = (
                        "visible_target_count"
                        if field == "target_count"
                        else "predicted_birth_query_count"
                        if field == "predicted_query_count"
                        else field
                    )
                    row_counts[output_field] = int(getattr(association, field))
                row_counts["visible_birth_target_count"] = (
                    visible_birth_target_count
                )
                row_counts["visible_recovery_target_count"] = (
                    visible_alive_opportunity_count
                )
                row_counts["predicted_start_active_query_count"] = (
                    predicted_start_active_count
                )
                row_counts["ambiguous_query_count"] = len(
                    association.ambiguous_queries
                )
                row_counts["ambiguous_target_count"] = len(
                    association.ambiguous_targets
                )
                row_counts["assignment_count"] = len(association.assignments)
                for output_field, tensor in runtime_fields.items():
                    row_counts[output_field] = int(tensor[row_index].item())
                _add_counts(totals, row_counts)
                video_counts = per_video.setdefault(video_name, _new_counts())
                _add_counts(video_counts, row_counts)
            history.detach()

    elapsed = time.perf_counter() - started_at
    checkpoint_stat_after = {
        "bytes": cli.checkpoint.stat().st_size,
        "mtime_ns": cli.checkpoint.stat().st_mtime_ns,
        "sha256": _sha256(cli.checkpoint),
    }
    if checkpoint_stat_after != checkpoint_stat_before:
        raise RuntimeError("checkpoint changed during read-only association scan")
    if _sha256(cli.options) != options_sha256:
        raise RuntimeError("options changed during read-only association scan")
    dataset_cache_stats_after = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for name, path in dataset_cache_paths.items()
    }
    if dataset_cache_stats_after != dataset_cache_stats_before:
        raise RuntimeError("dataset caches changed during read-only association scan")
    if totals["runtime_capacity_exhaustion_count"] != 0:
        raise RuntimeError("association scan encountered capacity exhaustion")
    if cli.max_batches == 0 and totals["observed_eos_count"] != len(
        dataset.video_list
    ):
        raise RuntimeError(
            f"complete scan observed {totals['observed_eos_count']} EOS markers "
            f"for {len(dataset.video_list)} videos"
        )
    if cli.max_batches == 0 and closed_streams != set(dataset.video_list):
        missing = sorted(set(dataset.video_list) - closed_streams)
        unexpected = sorted(closed_streams - set(dataset.video_list))
        raise RuntimeError(
            "complete scan stream closure mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if processed_batches != forward_batches:
        raise RuntimeError(
            "association scan changed the frozen physical batch schedule"
        )
    _validate_count_closure(totals, "global")
    for video_name, counts in per_video.items():
        _validate_count_closure(counts, video_name)
        if cli.max_batches == 0 and counts["observed_eos_count"] != 1:
            raise RuntimeError(
                f"complete scan expected one observed EOS for {video_name}, "
                f"got {counts['observed_eos_count']}"
            )
    if cli.max_batches == 0 and len(per_video) != len(dataset.video_list):
        raise RuntimeError(
            f"complete scan produced {len(per_video)} per-video rows for "
            f"{len(dataset.video_list)} videos"
        )

    score_summaries = {
        label: _score_summary(values, label=label)
        for label, values in {**score_chunks, **score_values}.items()
    }
    if (
        score_summaries["all_query_margin"].get("positive_count", 0)
        != totals["predicted_start_active_query_count"]
    ):
        raise RuntimeError(
            "score audit and predicted START-active query count disagree"
        )
    if (
        score_summaries["prefix_max_margin"].get("positive_count", 0)
        != totals["predicted_start_active_prefix_count"]
    ):
        raise RuntimeError(
            "score audit and predicted START-active prefix count disagree"
        )
    if (
        score_summaries["birth_oracle_path_margin"]["count"]
        != totals["visible_birth_target_count"]
    ):
        raise RuntimeError("birth oracle-path score count does not close")
    if (
        score_summaries["alive_opportunity_oracle_path_margin"]["count"]
        != totals["visible_recovery_target_count"]
    ):
        raise RuntimeError("alive-opportunity oracle-path score count does not close")

    def compatibility_summary(prefix: str, denominator: int) -> dict:
        fields = {
            "class_match": compatibility_counts[
                f"{prefix}_class_match_count"
            ],
            "start_distance_pass": compatibility_counts[
                f"{prefix}_start_distance_pass_count"
            ],
            "joint_class_and_distance_pass": compatibility_counts[
                f"{prefix}_joint_class_and_distance_pass_count"
            ],
        }
        counts = {
            f"{name}_count": int(value)
            for name, value in fields.items()
        }
        rates = {
            f"{name}_rate": (
                float(value) / int(denominator) if denominator else None
            )
            for name, value in fields.items()
        }
        return {**counts, **rates}

    birth_score_diagnostics = {
        "scope": "terminal_checkpoint_eval_mode_train_only_post_forward_gt",
        "decision_boundary_semantics": (
            "zero_is_start_equal_to_strongest_competitor_not_a_tuned_threshold"
        ),
        "oracle_path_diagnostic_semantics": (
            "post_forward_ground_truth_class_conditioned_temporal_viterbi"
        ),
        "oracle_path_is_runtime_association": False,
        "counterfactual_intervention_performed": False,
        "risk_calibration_valid": False,
        "fixed_start_distance_gate_frames": int(args.num_frame),
        "fixed_start_distance_gate_semantics": (
            "registered_geometry_not_a_learned_or_searched_threshold"
        ),
        "query_count": int(args.num_queries),
        "score_summaries": score_summaries,
        "birth_oracle_query_rank": _rank_summary(
            birth_oracle_ranks, query_count=int(args.num_queries)
        ),
        "alive_opportunity_oracle_query_rank": _rank_summary(
            alive_opportunity_oracle_ranks, query_count=int(args.num_queries)
        ),
        "birth_oracle_pairwise_preference_rate": (
            float(np.mean(birth_pairwise_preference_rates))
            if birth_pairwise_preference_rates
            else None
        ),
        "alive_opportunity_oracle_pairwise_preference_rate": (
            float(np.mean(alive_opportunity_pairwise_preference_rates))
            if alive_opportunity_pairwise_preference_rates
            else None
        ),
        "birth_oracle_state_winner_counts": birth_oracle_state_winners,
        "alive_opportunity_oracle_state_winner_counts": (
            alive_opportunity_oracle_state_winners
        ),
        "birth_oracle_path_observational_compatibility": compatibility_summary(
            "birth", totals["visible_birth_target_count"]
        ),
        "alive_opportunity_oracle_path_observational_compatibility": (
            compatibility_summary(
                "alive_opportunity", totals["visible_recovery_target_count"]
            )
        ),
        "birth_interval_fallback_count": int(
            compatibility_counts["birth_interval_fallback_count"]
        ),
        "birth_interval_probability_clamp_count": int(
            compatibility_counts["birth_interval_probability_clamp_count"]
        ),
    }

    result = {
        "status": "DIAGNOSTIC_COMPLETE",
        "execution_status": "PASS",
        "status_semantics": "scan_completed_not_mechanism_or_performance_pass",
        "protocol": "eventmatr_d11_terminal_association_scan_v4",
        "complete_scan": cli.max_batches == 0,
        "processed_batches": processed_batches,
        "forward_batches": forward_batches,
        "seed": 52,
        "test_access": False,
        "checkpoint_updated": False,
        "strict_causal_paper_result_valid": False,
        "ground_truth_visible_to_model": False,
        "padding_metadata_visible_to_model": True,
        "padding_metadata_semantics": (
            "structural_no_op_only_after_observed_eos"
        ),
        "padding_contract_verified": True,
        "physical_batch_contract": "frozen_matr_fixed_width_preserved",
        "eos_semantics": "current_stream_termination_observation_only",
        "association_semantics": (
            "post_forward_opportunity_scan_without_target_ownership_exclusion"
        ),
        "association_target_semantics": (
            "birth_targets_plus_alive_opportunity_upper_bound_without_teacher_"
            "ownership_reconstruction"
        ),
        "visible_recovery_field_semantics": (
            "legacy_field_name_means_alive_opportunity_without_target_ownership_"
            "exclusion_not_reconstructed_recovery"
        ),
        "root_cause_scope": (
            "necessary_condition_opportunity_scan_not_teacher_ownership_reconstruction"
        ),
        "performance_gate_release": False,
        "one_epoch_mechanism_gate_status": (
            cli.one_epoch_mechanism_gate_status
        ),
        "threshold_search": False,
        "source_identity": {
            "commit": _git("rev-parse", "HEAD"),
            "tree": _git("rev-parse", "HEAD^{tree}"),
        },
        "training_source_identity": identity,
        "checkpoint": {
            "path": str(cli.checkpoint),
            **checkpoint_stat_before,
        },
        "options": {
            "path": str(cli.options),
            "sha256": options_sha256,
        },
        "dataset_caches": dataset_cache_stats_before,
        "barrier_counts": totals,
        "birth_score_diagnostics": birth_score_diagnostics,
        "videos_scanned": len(per_video),
        "per_video_barrier_counts": {
            video_name: dict(counts)
            for video_name, counts in sorted(per_video.items())
        },
        "systems": {
            "device": torch.cuda.get_device_name(0),
            "elapsed_seconds": elapsed,
            "real_prefixes_per_second": (
                totals["real_prefix_count"] / elapsed if elapsed > 0 else None
            ),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }
    if not all(
        math.isfinite(float(value))
        for value in result["barrier_counts"].values()
    ):
        raise RuntimeError("association scan produced non-finite counts")
    cli.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(cli.output)
    print(json.dumps(result["barrier_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
