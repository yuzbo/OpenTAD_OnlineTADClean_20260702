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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--options", required=True, type=Path)
    parser.add_argument("--source-identity", required=True, type=Path)
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


def _validate_inputs(options: dict, checkpoint: dict, identity: dict) -> None:
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
    if identity.get("commit") != "de0837cf38e05d65a40f0744b863056edc2f433a":
        raise RuntimeError("unexpected D1.1 training commit")
    if identity.get("tree") != "91d998e787c42895b08571ef0ed5ab5af5458a47":
        raise RuntimeError("unexpected D1.1 training tree")
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


def _select_rows(value, indices: torch.Tensor):
    if torch.is_tensor(value):
        return value.index_select(0, indices.to(value.device))
    if isinstance(value, np.ndarray):
        return value[indices.cpu().numpy()]
    if isinstance(value, (list, tuple)):
        selected = [value[int(index)] for index in indices.tolist()]
        return tuple(selected) if isinstance(value, tuple) else selected
    raise TypeError(f"cannot select diagnostic batch rows from {type(value)!r}")


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
    _validate_inputs(options, checkpoint, identity)

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
            real_mask = infos["is_real_prefix"]
            if not torch.is_tensor(real_mask):
                real_mask = torch.tensor(real_mask)
            real_indices = (
                real_mask.detach().cpu().bool().reshape(-1).nonzero(
                    as_tuple=False
                ).reshape(-1)
            )
            if real_indices.numel() == 0:
                continue
            forward_batches += 1
            features = _select_rows(features, real_indices).to(
                device, non_blocking=True
            )
            target_tensors = {
                key: _select_rows(value, real_indices).to(
                    device, non_blocking=True
                )
                for key, value in targets.items()
            }
            infos = {
                key: _select_rows(value, real_indices)
                for key, value in infos.items()
            }
            if not all(bool(value) for value in infos["is_real_prefix"]):
                raise RuntimeError("diagnostic real-prefix filtering failed")
            if "segment_flag" not in infos:
                raise RuntimeError(
                    "diagnostic loader omitted the inherited supervision flag"
                )
            runtime_infos = dict(infos)
            runtime_infos.pop("is_real_prefix")
            payload = make_model_inputs(args, features, runtime_infos)
            expected_model_info_keys = (
                set(runtime_infos) - D1_RUNTIME_FORBIDDEN_MODEL_INFO
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
            outputs = model(payload, device)

            real_mask = infos["is_real_prefix"]
            if not torch.is_tensor(real_mask):
                real_mask = torch.tensor(real_mask)
            real_mask = real_mask.detach().cpu().bool().reshape(-1)
            names = [str(value) for value in infos["video_name"]]
            frames = infos["current_frame"]
            if torch.is_tensor(frames):
                frames = frames.detach().cpu().reshape(-1).tolist()
            else:
                frames = list(frames)

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

            for row_index, video_name in enumerate(names):
                if not bool(real_mask[row_index]):
                    continue
                frame = float(frames[row_index])
                previous_frame = last_real_frame.get(video_name)
                if previous_frame is not None and frame <= previous_frame:
                    raise RuntimeError(
                        f"non-monotonic diagnostic frame for {video_name}: "
                        f"{frame} after {previous_frame}"
                    )
                last_real_frame[video_name] = frame
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
                visible_birth_target_count = 0
                visible_recovery_target_count = 0
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
                    visible_birth_target_count += int(is_birth)
                    visible_recovery_target_count += int(is_alive)
                    target_specs.append(
                        {
                            "target_event_id": int(row[0].item()),
                            "class_id": int(row[1].item()),
                            "start_frame": start_frame,
                            "anchor_feature": window[-1]["query_features"][
                                int(path[-1])
                            ],
                        }
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
                    visible_recovery_target_count > 0
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
                    visible_recovery_target_count
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

    result = {
        "status": "DIAGNOSTIC_COMPLETE",
        "execution_status": "PASS",
        "status_semantics": "scan_completed_not_mechanism_or_performance_pass",
        "protocol": "eventmatr_d11_failed_one_epoch_association_scan_v1",
        "complete_scan": cli.max_batches == 0,
        "processed_batches": processed_batches,
        "forward_batches": forward_batches,
        "seed": 52,
        "test_access": False,
        "checkpoint_updated": False,
        "strict_causal_paper_result_valid": False,
        "ground_truth_visible_to_model": False,
        "padding_metadata_visible_to_model": False,
        "eos_semantics": "current_stream_termination_observation_only",
        "association_semantics": (
            "post_forward_opportunity_scan_without_target_ownership_exclusion"
        ),
        "root_cause_scope": (
            "necessary_condition_opportunity_scan_not_teacher_ownership_reconstruction"
        ),
        "performance_gate_release": False,
        "one_epoch_mechanism_gate_status": "FAIL_UNCHANGED",
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
