#!/usr/bin/env python3
"""Run the frozen full-fit-core Q2 capacity/lifecycle audit on CPU."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid

import torch
from mmengine import Config


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.datasets import build_dataset  # noqa: E402
from opentad.models import build_detector  # noqa: E402
from opentad.utils.evidence_bundle import (  # noqa: E402
    EvidenceBundleError,
    read_stable_file_bytes,
    resolve_bundle_path,
    strict_json_from_bytes,
)
from opentad.utils.full_petal_identity import canonical_json_sha256  # noqa: E402
from opentad.utils.full_petal_launch import resolved_config_sha256  # noqa: E402
from opentad.utils.misc import set_seed  # noqa: E402
from opentad.utils.prefix_instance_schedule import (  # noqa: E402
    build_prefix_instance_schedule,
)
from opentad.utils.prefix_trajectory_supervision import (  # noqa: E402
    PrefixTrajectorySupervisionState,
    SupervisionMode,
)
from opentad.utils.q2_capacity_audit import (  # noqa: E402
    AUDIT_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    EXHAUSTION_CAUSES,
    Q2CapacityAuditError,
    Q2LifecycleState,
    adjusted_outputs,
    attribute_exhaustions,
    audit_annotation_schedule,
    build_assignment_cost_provider,
    canonical_available_slots,
    frozen_policy_grid,
    policy_summary_template,
    replay_lifecycle_step,
    state_dict_fingerprint,
)


FROZEN_SEEDS = (705, 706, 707)
FROZEN_FIT_CORE_COUNT = 160
FROZEN_CPU_HOUR_CAP = 48.0
FORBIDDEN_HEAD_META = {
    "annotations",
    "segments",
    "labels",
    "ground_truth",
    "future_endpoint",
    "video_end_frame",
    "eof",
}


class CapacityAuditRunnerError(RuntimeError):
    pass


class CapacityAuditBudgetExceeded(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_commit():
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise CapacityAuditRunnerError("capacity audit requires a clean checkout")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise CapacityAuditRunnerError("git commit identity is malformed")
    return commit


def external_new_directory(path):
    output = Path(path).expanduser().resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise CapacityAuditRunnerError(
            "capacity evidence must stay outside the repository"
        )
    if output.exists():
        raise CapacityAuditRunnerError(
            f"refusing to overwrite capacity evidence: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def load_checkpoint_manifest(bundle):
    bundle = Path(bundle).expanduser().resolve(strict=True)
    if not bundle.is_dir():
        raise CapacityAuditRunnerError("checkpoint bundle must be a directory")
    manifest_path = bundle / "manifest.json"
    _, raw = read_stable_file_bytes(manifest_path, "capacity checkpoint manifest")
    manifest = dict(
        strict_json_from_bytes(raw, "capacity checkpoint manifest", require_object=True)
    )
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CapacityAuditRunnerError("capacity checkpoint schema is incompatible")
    claimed = manifest.get("manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if claimed != canonical_json_sha256(body):
        raise CapacityAuditRunnerError("capacity checkpoint manifest hash is invalid")
    return bundle, manifest, sha256_file(manifest_path)


def load_checkpoint_record(bundle, record):
    path = resolve_bundle_path(record.get("path"), bundle, "capacity checkpoint")
    resolved, raw = read_stable_file_bytes(path, "capacity checkpoint")
    if hashlib.sha256(raw).hexdigest() != record.get("sha256"):
        raise CapacityAuditRunnerError("capacity checkpoint bytes differ from manifest")
    if len(raw) != int(record.get("byte_size", -1)):
        raise CapacityAuditRunnerError("capacity checkpoint size differs from manifest")
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(io.BytesIO(raw), map_location="cpu")
    if not isinstance(payload, dict) or payload.get("seed") != record.get("seed"):
        raise CapacityAuditRunnerError(
            "capacity checkpoint payload identity is invalid"
        )
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CapacityAuditRunnerError("capacity checkpoint payload schema is invalid")
    state = payload.get(record.get("state_key"))
    if not isinstance(state, dict):
        raise CapacityAuditRunnerError(
            "capacity checkpoint state dictionary is missing"
        )
    if state_dict_fingerprint(state) != record.get("tensor_fingerprint_sha256"):
        raise CapacityAuditRunnerError(
            "capacity checkpoint tensor fingerprint is invalid"
        )
    return resolved, state


def validate_q2_config(cfg):
    if cfg.get("route_stage") != "q2_persistent_binding_one_factor":
        raise CapacityAuditRunnerError("capacity audit requires the frozen Q2 route")
    if cfg.get("formal_training_ready") is not False:
        raise CapacityAuditRunnerError(
            "capacity audit cannot run from a formal-ready config"
        )
    if tuple(int(seed) for seed in cfg.get("pilot_seeds", ())) != FROZEN_SEEDS:
        raise CapacityAuditRunnerError("Q2 pilot seeds drifted from 705/706/707")
    head = cfg.model.head
    expected = {
        "num_slots": 4,
        "dropout": 0.1,
        "birth_threshold": 0.5,
        "alive_threshold": 0.5,
        "end_threshold": 0.5,
        "refractory_steps": 2,
    }
    observed = {key: head.get(key) for key in expected}
    if observed != expected:
        raise CapacityAuditRunnerError(
            f"Q2 lifecycle config drifted: expected {expected}, observed {observed}"
        )


def assert_zero_gpu_runtime():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", "-1"):
        raise CapacityAuditRunnerError(
            "capacity audit requires CUDA_VISIBLE_DEVICES empty"
        )
    if os.environ.get("SLURM_JOB_GPUS", "").strip():
        raise CapacityAuditRunnerError("capacity audit received a Slurm GPU allocation")
    if visible is None and torch.cuda.is_available():
        raise CapacityAuditRunnerError(
            "CUDA is visible; explicitly hide devices for the zero-GPU audit"
        )


def assert_lifecycle_independent_head(head):
    original_rng = torch.get_rng_state()
    base = head.initial_state(torch.device("cpu"), torch.float32, "invariant")
    status = torch.tensor([0, 1, 2, 1][: head.num_slots], dtype=torch.long)
    if len(status) < head.num_slots:
        status = torch.cat(
            [status, torch.zeros(head.num_slots - len(status), dtype=torch.long)]
        )
    changed = replace(
        base,
        slot_status=status,
        refractory=torch.arange(head.num_slots, dtype=torch.long),
        start_frames=torch.arange(head.num_slots, dtype=torch.float32),
        peak_class_scores=torch.ones(head.num_slots),
        peak_class_labels=torch.arange(head.num_slots, dtype=torch.long),
        instance_to_slot={100 + slot: slot for slot in range(head.num_slots)},
        slot_to_instance={slot: 100 + slot for slot in range(head.num_slots)},
    )
    feature = torch.zeros((1, head.in_channels), dtype=torch.float32)
    try:
        torch.set_rng_state(original_rng)
        left_outputs, left_state = head.step(feature, base, 0)
        torch.set_rng_state(original_rng)
        right_outputs, right_state = head.step(feature, changed, 0)
    finally:
        torch.set_rng_state(original_rng)
    for key in left_outputs:
        left = left_outputs[key]
        right = right_outputs[key]
        if torch.is_tensor(left):
            if not torch.equal(left, right):
                raise CapacityAuditRunnerError(
                    f"head logits now depend on lifecycle field: {key}"
                )
        elif left != right:
            raise CapacityAuditRunnerError(
                f"head output now depends on lifecycle field: {key}"
            )
    if not torch.equal(left_state.queries, right_state.queries) or not torch.equal(
        left_state.feature_memory, right_state.feature_memory
    ):
        raise CapacityAuditRunnerError("head numerical recurrence depends on lifecycle")


def write_trace(handle, row):
    handle.write(
        json.dumps(
            row,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def binding_rows(bindings):
    return [
        {"instance_id": int(item.instance_id), "slot_id": int(item.slot_id)}
        for item in bindings
    ]


def canonical_rows(state):
    return [
        {"instance_id": int(instance), "slot_id": int(slot)}
        for instance, slot in sorted(state.instance_to_slot.items())
    ]


def aggregate_policy_summaries(by_seed, policies):
    aggregate = {}
    for policy in policies:
        combined = policy_summary_template(policy)
        for seed in FROZEN_SEEDS:
            source = by_seed[str(seed)][policy.name]
            for key in (
                "bins",
                "births",
                "assignments",
                "exhaustions",
                "emissions",
                "false_active_slot_bins",
                "refractory_slot_bins",
            ):
                combined[key] += int(source[key])
            for cause in EXHAUSTION_CAUSES:
                combined["cause_counts"][cause] += int(source["cause_counts"][cause])
            combined["attribution_closed"] = bool(
                combined["attribution_closed"] and source["attribution_closed"]
            )
            combined["fixed_rematch_canonical_equal"] = bool(
                combined["fixed_rematch_canonical_equal"]
                and source["fixed_rematch_canonical_equal"]
            )
        aggregate[policy.name] = combined
    return aggregate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--wall-time-budget-seconds", type=int, default=21000)
    parser.add_argument("--cpu-hour-cap", type=float, default=FROZEN_CPU_HOUR_CAP)
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    partial = None
    trace_text = None
    try:
        if not 1 <= args.cpu_threads <= 16:
            raise CapacityAuditRunnerError("cpu-threads must lie in [1,16]")
        if args.wall_time_budget_seconds <= 0:
            raise CapacityAuditRunnerError("wall-time budget must be positive")
        if float(args.cpu_hour_cap) != FROZEN_CPU_HOUR_CAP:
            raise CapacityAuditRunnerError("CPU-hour cap is frozen at exactly 48")
        declared_cpu_hours = args.cpu_threads * args.wall_time_budget_seconds / 3600.0
        if declared_cpu_hours > FROZEN_CPU_HOUR_CAP:
            raise CapacityAuditRunnerError(
                "declared wall/thread budget exceeds 48 CPU-hours"
            )
        assert_zero_gpu_runtime()
        output = external_new_directory(args.output)
        config_path = args.config.resolve(strict=True)
        config_path.relative_to(ROOT.resolve())
        if config_path.name != "thumos_pes_q2_persist_fixed.py":
            raise CapacityAuditRunnerError(
                "capacity audit requires thumos_pes_q2_persist_fixed.py"
            )
        commit_sha = clean_commit()
        cfg = Config.fromfile(str(config_path))
        validate_q2_config(cfg)
        (
            bundle,
            checkpoint_manifest,
            checkpoint_manifest_file_sha,
        ) = load_checkpoint_manifest(args.checkpoint_bundle)
        if checkpoint_manifest.get("commit_sha") != commit_sha:
            raise CapacityAuditRunnerError(
                "checkpoint bundle commit differs from audit"
            )
        checkpoint_config = checkpoint_manifest.get("config", {})
        expected_config = {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(config_path),
            "resolved_config_sha256": resolved_config_sha256(cfg),
            "scientific_config_sha256": resolved_config_sha256(cfg, scientific=True),
        }
        if checkpoint_config != expected_config:
            raise CapacityAuditRunnerError("checkpoint bundle config identity drifted")
        checkpoint_records = checkpoint_manifest.get("checkpoints")
        if not isinstance(checkpoint_records, list):
            raise CapacityAuditRunnerError(
                "checkpoint manifest lacks checkpoint records"
            )
        records_by_seed = {int(record["seed"]): record for record in checkpoint_records}
        if (
            len(checkpoint_records) != len(FROZEN_SEEDS)
            or tuple(sorted(records_by_seed)) != FROZEN_SEEDS
        ):
            raise CapacityAuditRunnerError(
                "checkpoint bundle seeds are not 705/706/707"
            )

        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        dataset = build_dataset(dict(cfg.dataset.train))
        video_ids = tuple(sorted(dataset.packet_manifests))
        if len(video_ids) != FROZEN_FIT_CORE_COUNT:
            raise CapacityAuditRunnerError(
                f"fit core must contain exactly 160 videos, observed {len(video_ids)}"
            )
        policies = frozen_policy_grid(int(cfg.num_slots))
        policy_sha = canonical_json_sha256([policy.as_record() for policy in policies])

        partial = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
        partial.mkdir()
        raw_trace = (partial / "capacity_trace.jsonl.gz").open("wb")
        gzip_trace = gzip.GzipFile(fileobj=raw_trace, mode="wb", mtime=0)
        trace_text = io.TextIOWrapper(gzip_trace, encoding="utf-8", newline="\n")
        started_at = time.monotonic()
        budget_exceeded = False
        budget_message = None
        trace_rows = 0
        annotation_complete = False
        replay_complete_seeds = []
        annotation_maxima = Counter()
        annotation_video_summaries = []
        k_census_bin_counts = Counter()
        by_seed = {
            str(seed): {
                policy.name: policy_summary_template(policy) for policy in policies
            }
            for seed in FROZEN_SEEDS
        }

        def check_budget():
            elapsed = time.monotonic() - started_at
            if elapsed >= args.wall_time_budget_seconds:
                raise CapacityAuditBudgetExceeded(
                    "frozen wall-time budget reached before audit completion"
                )

        try:
            for video_index, video_id in enumerate(video_ids):
                record = dataset.video_records[video_id]
                schedule = build_prefix_instance_schedule(
                    record["segments"],
                    record["labels"],
                    decision_frames=record["source_frames"],
                    previous_frame=record["source_frames"][0] - cfg.feature_stride,
                )
                video_summary, annotation_rows = audit_annotation_schedule(
                    video_id,
                    schedule,
                    num_slots=cfg.num_slots,
                    refractory_steps=cfg.model.head.refractory_steps,
                )
                annotation_video_summaries.append(video_summary)
                for key, value in video_summary["maxima"].items():
                    annotation_maxima[key] = max(annotation_maxima[key], int(value))
                for row in annotation_rows:
                    demand = int(row["minimum_oracle_free_k"])
                    for capacity in range(2, max(8, demand) + 1):
                        if demand > capacity:
                            k_census_bin_counts[capacity] += 1
                    write_trace(trace_text, {"record_type": "annotation_bin", **row})
                    trace_rows += 1
                if video_index % 16 == 0:
                    check_budget()
            annotation_complete = True

            for seed in FROZEN_SEEDS:
                set_seed(seed)
                model = build_detector(dict(cfg.model)).cpu().train()
                checkpoint_path, checkpoint_state = load_checkpoint_record(
                    bundle, records_by_seed[seed]
                )
                initialized_fingerprint = state_dict_fingerprint(model.state_dict())
                expected_fingerprint = records_by_seed[seed][
                    "tensor_fingerprint_sha256"
                ]
                if initialized_fingerprint != expected_fingerprint:
                    raise CapacityAuditRunnerError(
                        f"seed {seed} no longer reproduces its init checkpoint"
                    )
                model.load_state_dict(checkpoint_state, strict=True)
                assert_lifecycle_independent_head(model.head)
                print(
                    f"Q2 capacity replay seed={seed} checkpoint={checkpoint_path}",
                    flush=True,
                )
                seed_summary = by_seed[str(seed)]
                for video_index, video_id in enumerate(video_ids):
                    numerical_state = model.head.initial_state(
                        torch.device("cpu"), torch.float32, f"{seed}:{video_id}"
                    )
                    runtime_states = {
                        policy.name: Q2LifecycleState.initial(policy.num_slots)
                        for policy in policies
                    }
                    supervision_states = {
                        policy.name: PrefixTrajectorySupervisionState(
                            policy.num_slots, SupervisionMode.FIXED
                        )
                        for policy in policies
                    }
                    actual_rematch_state = PrefixTrajectorySupervisionState(
                        int(cfg.num_slots), SupervisionMode.REMATCH
                    )
                    for packet_index in dataset.packet_manifests[video_id]:
                        sample = dataset[packet_index]
                        if not FORBIDDEN_HEAD_META.isdisjoint(sample["metas"]):
                            raise CapacityAuditRunnerError(
                                "GT-tainted metadata reached the numerical head path"
                            )
                        features = torch.from_numpy(sample["inputs"]).float()
                        source_frames = tuple(sample["metas"]["source_frames"])
                        schedule = tuple(sample["prefix_schedule"])
                        if features.shape[1] != len(source_frames) or len(
                            schedule
                        ) != len(source_frames):
                            raise CapacityAuditRunnerError(
                                "feature, provenance, and schedule lengths differ"
                            )
                        for local_index, (source_frame, schedule_step) in enumerate(
                            zip(source_frames, schedule)
                        ):
                            with torch.no_grad():
                                outputs, numerical_state = model.head.step(
                                    features[:, local_index].unsqueeze(0),
                                    numerical_state,
                                    int(source_frame),
                                )
                            global_bin = (
                                int(sample["metas"]["packet_start_token"]) + local_index
                            )
                            providers = {}
                            for policy in policies:
                                summary = seed_summary[policy.name]
                                runtime_before = runtime_states[policy.name]
                                lifecycle = replay_lifecycle_step(
                                    runtime_before, outputs, policy
                                )
                                bias_key = float(policy.birth_logit_bias)
                                if bias_key not in providers:
                                    policy_outputs = adjusted_outputs(outputs, policy)
                                    providers[
                                        bias_key
                                    ] = build_assignment_cost_provider(
                                        policy_outputs,
                                        schedule_step,
                                        feature_stride=cfg.feature_stride,
                                        memory_size=cfg.model.head.memory_size,
                                    )
                                provider = providers[bias_key]
                                fixed_state = supervision_states[policy.name]
                                fixed_before = dict(fixed_state.instance_to_slot)
                                if policy.privileged_canonical_availability:
                                    available = canonical_available_slots(
                                        fixed_state, policy.num_slots
                                    )
                                else:
                                    available = lifecycle.available_slots
                                fixed = fixed_state.transition(
                                    schedule_step, provider, available_slots=available
                                )
                                equal = True
                                if policy.name == "actual":
                                    rematch_before = dict(
                                        actual_rematch_state.instance_to_slot
                                    )
                                    if fixed_before != rematch_before:
                                        raise CapacityAuditRunnerError(
                                            "fixed/rematch canonical state diverged before transition"
                                        )
                                    rematch = actual_rematch_state.transition(
                                        schedule_step,
                                        provider,
                                        available_slots=available,
                                    )
                                    equal = (
                                        fixed.birth_assignments
                                        == rematch.birth_assignments
                                        and fixed.canonical_bindings
                                        == rematch.canonical_bindings
                                        and fixed.exhausted_instance_ids
                                        == rematch.exhausted_instance_ids
                                        and fixed_state.instance_to_slot
                                        == actual_rematch_state.instance_to_slot
                                    )
                                if not equal:
                                    summary["fixed_rematch_canonical_equal"] = False
                                    raise CapacityAuditRunnerError(
                                        "fixed/rematch capacity lifecycle diverged"
                                    )
                                attribution = attribute_exhaustions(
                                    fixed.exhausted_instance_ids,
                                    fixed.birth_instance_ids,
                                    [item.instance_id for item in schedule_step.ends],
                                    fixed_before,
                                    runtime_before,
                                    fixed.birth_candidates,
                                )
                                summary["bins"] += 1
                                summary["births"] += len(fixed.birth_instance_ids)
                                summary["assignments"] += len(fixed.birth_assignments)
                                summary["exhaustions"] += int(fixed.exhaustion)
                                summary["emissions"] += len(lifecycle.emitted_slots)
                                canonical_slots = set(fixed_before.values())
                                false_active_slots = [
                                    slot
                                    for slot, status in enumerate(
                                        runtime_before.slot_status
                                    )
                                    if status == 1 and slot not in canonical_slots
                                ]
                                refractory_free_slots = [
                                    slot
                                    for slot, status in enumerate(
                                        runtime_before.slot_status
                                    )
                                    if status == 2 and slot not in canonical_slots
                                ]
                                ending_ids = {
                                    int(item.instance_id) for item in schedule_step.ends
                                }
                                same_bin_ending_slots = [
                                    int(slot)
                                    for instance, slot in sorted(fixed_before.items())
                                    if int(instance) in ending_ids
                                ]
                                same_bin_runtime_free_slots = [
                                    slot
                                    for slot in same_bin_ending_slots
                                    if runtime_before.slot_status[slot] == 0
                                ]
                                summary["false_active_slot_bins"] += len(
                                    false_active_slots
                                )
                                summary["refractory_slot_bins"] += len(
                                    refractory_free_slots
                                )
                                summary["attribution_closed"] = bool(
                                    summary["attribution_closed"]
                                    and attribution.attribution_closed
                                )
                                for cause, count in attribution.counts.items():
                                    summary["cause_counts"][cause] += int(count)
                                runtime_states[policy.name] = lifecycle.state_after

                                if policy.name == "actual" or fixed.exhaustion:
                                    row = {
                                        "record_type": (
                                            "actual_bin"
                                            if policy.name == "actual"
                                            else "counterfactual_exhaustion"
                                        ),
                                        "seed": seed,
                                        "video_id": video_id,
                                        "bin_index": global_bin,
                                        "current_frame": int(source_frame),
                                        "policy": policy.name,
                                        "birth_probabilities": list(
                                            lifecycle.birth_probabilities
                                        ),
                                        "alive_probabilities": list(
                                            lifecycle.alive_probabilities
                                        ),
                                        "end_probabilities": list(
                                            lifecycle.end_probabilities
                                        ),
                                        "runtime_status_before": list(
                                            runtime_before.slot_status
                                        ),
                                        "runtime_refractory_before": list(
                                            runtime_before.refractory
                                        ),
                                        "runtime_available_slots": list(
                                            lifecycle.available_slots
                                        ),
                                        "birth_candidates": list(
                                            fixed.birth_candidates
                                        ),
                                        "runtime_status_after": list(
                                            lifecycle.state_after.slot_status
                                        ),
                                        "runtime_refractory_after": list(
                                            lifecycle.state_after.refractory
                                        ),
                                        "false_active_slots": false_active_slots,
                                        "canonically_free_refractory_slots": (
                                            refractory_free_slots
                                        ),
                                        "same_bin_ending_slots": (
                                            same_bin_ending_slots
                                        ),
                                        "same_bin_runtime_free_slots": (
                                            same_bin_runtime_free_slots
                                        ),
                                        "canonical_before": [
                                            {
                                                "instance_id": int(instance),
                                                "slot_id": int(slot),
                                            }
                                            for instance, slot in sorted(
                                                fixed_before.items()
                                            )
                                        ],
                                        "canonical_after": canonical_rows(fixed_state),
                                        "gt_birth_instance_ids": list(
                                            fixed.birth_instance_ids
                                        ),
                                        "gt_end_instance_ids": [
                                            int(item.instance_id)
                                            for item in schedule_step.ends
                                        ],
                                        "birth_assignments": binding_rows(
                                            fixed.birth_assignments
                                        ),
                                        "exhausted_instance_ids": list(
                                            fixed.exhausted_instance_ids
                                        ),
                                        "emitted_slots": list(lifecycle.emitted_slots),
                                        "attribution": attribution.as_record(),
                                        "fixed_rematch_canonical_equal": equal,
                                    }
                                    write_trace(trace_text, row)
                                    trace_rows += 1
                            if global_bin % 64 == 0:
                                check_budget()
                    for policy in policies:
                        if supervision_states[policy.name].instance_to_slot:
                            raise CapacityAuditRunnerError(
                                f"{video_id} ended with live fixed canonical instances"
                            )
                    if actual_rematch_state.instance_to_slot:
                        raise CapacityAuditRunnerError(
                            f"{video_id} ended with live rematch canonical instances"
                        )
                    if (video_index + 1) % 10 == 0:
                        print(
                            f"Q2 capacity seed={seed} videos={video_index + 1}/160",
                            flush=True,
                        )
                        check_budget()
                replay_complete_seeds.append(seed)
                del model
        except CapacityAuditBudgetExceeded as exc:
            budget_exceeded = True
            budget_message = str(exc)
        finally:
            trace_text.flush()
            trace_text.close()
            trace_text = None

        aggregate = aggregate_policy_summaries(by_seed, policies)
        all_closed = all(row["attribution_closed"] for row in aggregate.values())
        all_equal = all(
            row["fixed_rematch_canonical_equal"] for row in aggregate.values()
        )
        actual_exhaustions = int(aggregate["actual"]["exhaustions"])
        legal_zero = [
            policy.name
            for policy in policies
            if policy.eligible_shared_contract
            and int(aggregate[policy.name]["exhaustions"]) == 0
        ]
        oracle_k = int(annotation_maxima.get("minimum_oracle_free_k", 0))
        if budget_exceeded:
            gate_status = "BUDGET_EXCEEDED"
            selected_contract = None
        elif not annotation_complete or tuple(replay_complete_seeds) != FROZEN_SEEDS:
            gate_status = "INVALID_EVIDENCE"
            selected_contract = None
        elif not all_closed or not all_equal:
            gate_status = "INVALID_EVIDENCE"
            selected_contract = None
        elif oracle_k > int(cfg.num_slots):
            gate_status = "Q2_INVALID_STRUCTURAL_CAPACITY"
            selected_contract = None
        elif actual_exhaustions == 0:
            gate_status = "PASS_FREEZE_ACTUAL"
            selected_contract = "actual"
        elif legal_zero:
            gate_status = "REVISE_REQUIRED"
            selected_contract = None
        else:
            gate_status = "Q2_INVALID_NO_SHARED_POLICY"
            selected_contract = None

        trace_path = partial / "capacity_trace.jsonl.gz"
        elapsed = time.monotonic() - started_at
        data_identity = {
            "video_ids_sha256": canonical_json_sha256(list(video_ids)),
            "video_count": len(video_ids),
            "cache_manifest_sha256": dataset.cache_manifest_sha256,
            "split_manifest_sha256": dataset.split_manifest_sha256,
            "split_manifest_file_sha256": dataset.split_manifest_file_sha256,
            "split_role": dataset.split_role,
            "split_seed": dataset.split_seed,
        }
        report = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "status": gate_status,
            "commit_sha": commit_sha,
            "config": expected_config,
            "checkpoint_manifest": {
                "path": str((bundle / "manifest.json").resolve()),
                "file_sha256": checkpoint_manifest_file_sha,
                "manifest_sha256": checkpoint_manifest["manifest_sha256"],
            },
            "data_identity": data_identity,
            "data_identity_sha256": canonical_json_sha256(data_identity),
            "gt_taint_audit": {
                "status": "PASS",
                "forbidden_model_meta_keys": sorted(FORBIDDEN_HEAD_META),
                "schedule_consumed_by_head": False,
                "canonical_only_policy_eligible_for_training": False,
            },
            "resource_contract": {
                "device": "cpu",
                "gpu_hours": 0,
                "cpu_threads": args.cpu_threads,
                "wall_time_budget_seconds": args.wall_time_budget_seconds,
                "cpu_hour_cap": FROZEN_CPU_HOUR_CAP,
                "declared_cpu_hours": declared_cpu_hours,
                "elapsed_seconds": elapsed,
                "budget_exceeded": budget_exceeded,
                "budget_message": budget_message,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
                "torch_version": str(torch.__version__),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            "policy_contract_sha256": policy_sha,
            "same_logits_contract": {
                "head_lifecycle_independence_revalidated": True,
                "fixed_rematch_verification": (
                    "actual policy replayed in both modes at every bin; "
                    "counterfactual policies reuse the same mode-independent "
                    "canonical transition proven by production-code unit tests"
                ),
                "native_checkpoint_k": int(cfg.num_slots),
                "same_logits_contractions": [
                    policy.num_slots
                    for policy in policies
                    if policy.name.startswith("capacity_k")
                ],
                "same_logits_expansion_supported": False,
                "expansion_reason": (
                    "K greater than checkpoint K has no corresponding frozen logits; "
                    "structural expansion is reported by annotation-only K census"
                ),
            },
            "annotation": {
                "complete": annotation_complete,
                "maxima": dict(annotation_maxima),
                "minimum_oracle_free_k": oracle_k,
                "k_census_bins_exceeding_capacity": {
                    str(capacity): int(count)
                    for capacity, count in sorted(k_census_bin_counts.items())
                },
                "videos": annotation_video_summaries,
            },
            "chronological_replay": {
                "complete_seeds": replay_complete_seeds,
                "expected_seeds": list(FROZEN_SEEDS),
                "by_seed": by_seed,
                "aggregate": aggregate,
                "trace_rows": trace_rows,
                "trace_file": trace_path.name,
                "trace_sha256": sha256_file(trace_path),
            },
            "gate": {
                "status": gate_status,
                "actual_exhaustions": actual_exhaustions,
                "legal_zero_exhaustion_candidates": legal_zero,
                "selected_contract": selected_contract,
                "fixed_rematch_canonical_equal": all_equal,
                "all_exhaustions_attributed": all_closed,
                "r1_implementation_allowed": gate_status == "PASS_FREEZE_ACTUAL",
                "gpu_profile_allowed": False,
                "formal_training_allowed": False,
            },
        }
        report["content_sha256"] = canonical_json_sha256(report)
        summary_path = partial / "capacity_summary.json"
        summary_path.write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        commitment = {
            "schema_version": "q2-capacity-evidence-commitment-v1",
            "commit_sha": commit_sha,
            "summary_sha256": sha256_file(summary_path),
            "trace_sha256": sha256_file(trace_path),
            "checkpoint_manifest_sha256": checkpoint_manifest_file_sha,
            "policy_contract_sha256": policy_sha,
            "status": gate_status,
        }
        commitment["content_sha256"] = canonical_json_sha256(commitment)
        (partial / "commitment.json").write_text(
            json.dumps(commitment, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, output)
        partial = None
    except (
        CapacityAuditRunnerError,
        EvidenceBundleError,
        KeyError,
        OSError,
        Q2CapacityAuditError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as exc:
        if trace_text is not None:
            trace_text.close()
        if partial is not None and partial.exists():
            shutil.rmtree(partial)
        parser.error(str(exc))
    print(f"Q2_CAPACITY_AUDIT={output}")
    print(f"Q2_CAPACITY_STATUS={gate_status}")
    print(f"Q2_CAPACITY_SUMMARY_SHA256={sha256_file(output / 'capacity_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
