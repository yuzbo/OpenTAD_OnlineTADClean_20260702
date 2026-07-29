"""Reconstruct the D1.1 initialization and measure the epoch-one parameter delta.

The audit performs no model forward, optimizer step, or checkpoint write.  It
replays the training seed and construction order on the official train-only
dataset, then compares every model parameter with the frozen terminal
checkpoint.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_eventmatr_d11_association_scan import (  # noqa: E402
    _sha256,
    _validate_inputs,
)


TRAINING_COMMIT = "de0837cf38e05d65a40f0744b863056edc2f433a"
INITIALIZATION_HELPER = r"""
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

root = Path.cwd()
sys.path.insert(0, str(root))
from dataset import THUMOS14Dataset
from models import build_model

options = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
output = Path(sys.argv[2])
seed = int(options.get("random_seed", -1))
if seed != 52:
    raise RuntimeError(f"expected frozen training seed 52, got {seed}")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
args = SimpleNamespace(**options)

def rng_sha256():
    return hashlib.sha256(
        torch.get_rng_state().detach().cpu().numpy().tobytes()
    ).hexdigest()

rng_before_dataset = rng_sha256()
dataset = THUMOS14Dataset(args, subset="train")
rng_after_dataset = rng_sha256()
if len(dataset.video_list) != 200:
    raise RuntimeError(
        f"initialization reconstruction expected 200 videos, "
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
loader_batch_size = int(args.batch)
dataset_item_count = len(dataset)
loader_batch_count = len(loader)
model = torch.nn.DataParallel(build_model(args))
torch.save(
    {
        "state_dict": model.state_dict(),
        "parameter_names": [name for name, _ in model.named_parameters()],
        "dataset_consumed_torch_rng": rng_before_dataset != rng_after_dataset,
        "video_count": len(dataset.video_list),
        "dataset_item_count": dataset_item_count,
        "loader_batch_size": loader_batch_size,
        "loader_batch_count": loader_batch_count,
    },
    output,
)
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--options", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--source-identity", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-options-sha256", required=True)
    parser.add_argument("--expected-optimizer-steps", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.expected_optimizer_steps <= 0:
        raise SystemExit("expected optimizer steps must be positive")
    for field in ("checkpoint", "options", "metrics", "source_identity"):
        path = getattr(args, field).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"{field} does not exist: {path}")
        setattr(args, field, path)
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


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


def _reconstruct_initial_state(options: Path, work_root: Path) -> dict:
    with tempfile.TemporaryDirectory(
        prefix="eventmatr_d11_initial_",
        dir=str(work_root),
    ) as temporary:
        temporary_root = Path(temporary).resolve()
        source_root = temporary_root / "source"
        source_root.mkdir()
        archive_process = subprocess.run(
            ["git", "archive", "--format=tar", TRAINING_COMMIT],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if archive_process.returncode != 0:
            raise RuntimeError(
                archive_process.stderr.decode("utf-8", errors="replace").strip()
                or "failed to archive exact training source"
            )
        with tarfile.open(
            fileobj=io.BytesIO(archive_process.stdout),
            mode="r:",
        ) as archive:
            for member in archive.getmembers():
                destination = (source_root / member.name).resolve()
                if source_root != destination and source_root not in destination.parents:
                    raise RuntimeError("training-source archive attempted path traversal")
                if not (member.isdir() or member.isfile()):
                    raise RuntimeError(
                        "training-source archive contains unsupported non-regular "
                        f"member: {member.name}"
                    )
            archive.extractall(source_root)
        state_path = temporary_root / "initial_state.pth"
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(source_root)
        completed = subprocess.run(
            [sys.executable, "-c", INITIALIZATION_HELPER, str(options), str(state_path)],
            cwd=source_root,
            env=environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                completed.stderr.strip()
                or completed.stdout.strip()
                or "exact training-source initialization reconstruction failed"
            )
        payload = torch.load(state_path, map_location="cpu")
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("state_dict"), dict)
        or not isinstance(payload.get("parameter_names"), list)
    ):
        raise RuntimeError("initialization reconstruction payload is invalid")
    return payload


def _new_delta_stats() -> dict:
    return {
        "parameter_tensor_count": 0,
        "element_count": 0,
        "changed_element_count": 0,
        "absolute_delta_sum": 0.0,
        "squared_delta_sum": 0.0,
        "squared_initial_sum": 0.0,
        "max_absolute_delta": 0.0,
    }


def _accumulate_delta(stats: dict, initial: torch.Tensor, terminal: torch.Tensor) -> None:
    initial = torch.as_tensor(initial)
    terminal = torch.as_tensor(terminal)
    if initial.shape != terminal.shape:
        raise RuntimeError(
            f"parameter shape mismatch: {tuple(initial.shape)} != {tuple(terminal.shape)}"
        )
    initial64 = initial.detach().cpu().to(torch.float64)
    terminal64 = terminal.detach().cpu().to(torch.float64)
    delta = terminal64 - initial64
    stats["parameter_tensor_count"] += 1
    stats["element_count"] += int(delta.numel())
    stats["changed_element_count"] += int(torch.count_nonzero(delta).item())
    stats["absolute_delta_sum"] += float(delta.abs().sum().item())
    stats["squared_delta_sum"] += float(delta.square().sum().item())
    stats["squared_initial_sum"] += float(initial64.square().sum().item())
    if delta.numel():
        stats["max_absolute_delta"] = max(
            float(stats["max_absolute_delta"]),
            float(delta.abs().max().item()),
        )


def _finalize_delta_stats(stats: dict) -> dict:
    element_count = int(stats["element_count"])
    delta_l2 = math.sqrt(float(stats["squared_delta_sum"]))
    initial_l2 = math.sqrt(float(stats["squared_initial_sum"]))
    return {
        "parameter_tensor_count": int(stats["parameter_tensor_count"]),
        "element_count": element_count,
        "changed_element_count": int(stats["changed_element_count"]),
        "changed_element_fraction": (
            float(stats["changed_element_count"]) / element_count
            if element_count
            else None
        ),
        "mean_absolute_delta": (
            float(stats["absolute_delta_sum"]) / element_count
            if element_count
            else None
        ),
        "max_absolute_delta": float(stats["max_absolute_delta"]),
        "delta_l2": delta_l2,
        "initial_l2": initial_l2,
        "relative_l2_delta": delta_l2 / initial_l2 if initial_l2 > 0.0 else None,
    }


def _parameter_group(name: str) -> str:
    if ".event_transition_head." in name:
        return "event_transition_head"
    if ".event_owner_decoder." in name:
        return "event_owner_decoder"
    if ".event_" in name:
        return "other_event_parameters"
    return "inherited_parent_parameters"


def _optimizer_step_summary(
    checkpoint: dict,
    expected_steps: int,
    parameter_names: list[str],
) -> dict:
    if expected_steps <= 0:
        raise RuntimeError("expected optimizer steps must be positive")
    if not parameter_names or len(set(parameter_names)) != len(parameter_names):
        raise RuntimeError("optimizer audit requires unique parameter names")
    optimizer = checkpoint.get("optimizer")
    if not isinstance(optimizer, dict):
        raise RuntimeError("checkpoint has no optimizer state")
    state = optimizer.get("state")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("checkpoint optimizer has no parameter state")
    param_groups = optimizer.get("param_groups")
    if not isinstance(param_groups, list) or len(param_groups) != 1:
        raise RuntimeError("expected exactly one optimizer parameter group")
    serialized_parameters = param_groups[0].get("params")
    if (
        not isinstance(serialized_parameters, list)
        or not all(isinstance(parameter_id, int) for parameter_id in serialized_parameters)
        or len(serialized_parameters) != len(parameter_names)
        or len(set(serialized_parameters)) != len(serialized_parameters)
    ):
        raise RuntimeError(
            "checkpoint optimizer parameter order does not match reconstructed model"
        )
    name_by_id = dict(zip(serialized_parameters, parameter_names))
    if not set(state).issubset(name_by_id):
        raise RuntimeError("checkpoint optimizer state contains an unknown parameter")
    steps_by_name = {}
    for parameter_id, parameter_state in state.items():
        if not isinstance(parameter_state, dict) or "step" not in parameter_state:
            raise RuntimeError("checkpoint optimizer parameter state has no step")
        value = parameter_state["step"]
        step = int(value.item()) if torch.is_tensor(value) else int(value)
        if step <= 0 or step > expected_steps:
            raise RuntimeError(
                "checkpoint optimizer step-count closure failed: "
                f"{name_by_id[parameter_id]} has {step}, expected 1..{expected_steps}"
            )
        steps_by_name[name_by_id[parameter_id]] = step
    steps = list(steps_by_name.values())
    unique_steps = sorted(set(steps))
    if max(steps) != expected_steps:
        raise RuntimeError(
            "checkpoint optimizer step-count closure failed: "
            f"no parameter state reached the registered {expected_steps} batches"
        )
    step_histogram = {
        str(step): sum(value == step for value in steps)
        for step in unique_steps
    }
    if sum(step_histogram.values()) != len(steps):
        raise RuntimeError("checkpoint optimizer global step histogram did not close")
    group_names = {
        group: [name for name in parameter_names if _parameter_group(name) == group]
        for group in (
            "event_transition_head",
            "event_owner_decoder",
            "other_event_parameters",
            "inherited_parent_parameters",
        )
    }
    step_summary_by_group = {}
    for group, names in group_names.items():
        group_steps = [steps_by_name[name] for name in names if name in steps_by_name]
        group_unique = sorted(set(group_steps))
        step_summary_by_group[group] = {
            "parameter_count": len(names),
            "state_initialized_parameter_count": len(group_steps),
            "state_uninitialized_parameter_count": len(names) - len(group_steps),
            "minimum_step": min(group_steps) if group_steps else None,
            "maximum_step": max(group_steps) if group_steps else None,
            "unique_steps": group_unique,
            "step_histogram": {
                str(step): sum(value == step for value in group_steps)
                for step in group_unique
            },
            "full_batch_step_parameter_count": sum(
                value == expected_steps for value in group_steps
            ),
            "conditional_step_parameter_count": sum(
                value < expected_steps for value in group_steps
            ),
        }
    uninitialized_names = [
        name for name in parameter_names if name not in steps_by_name
    ]
    if sum(len(names) for names in group_names.values()) != len(parameter_names):
        raise RuntimeError("optimizer parameter groups did not close over the model")
    if sum(
        summary["state_initialized_parameter_count"]
        for summary in step_summary_by_group.values()
    ) != len(steps):
        raise RuntimeError("optimizer initialized parameter groups did not close")
    if sum(
        summary["state_uninitialized_parameter_count"]
        for summary in step_summary_by_group.values()
    ) != len(uninitialized_names):
        raise RuntimeError("optimizer uninitialized parameter groups did not close")
    for group, summary in step_summary_by_group.items():
        if (
            summary["state_initialized_parameter_count"]
            + summary["state_uninitialized_parameter_count"]
            != summary["parameter_count"]
        ):
            raise RuntimeError(f"optimizer parameter counts did not close for {group}")
        if (
            sum(summary["step_histogram"].values())
            != summary["state_initialized_parameter_count"]
        ):
            raise RuntimeError(f"optimizer step histogram did not close for {group}")
    return {
        "expected_step": expected_steps,
        "step_count_closed": True,
        "step_count_semantics": (
            "global_batches_closed_by_completed_loader_and_max_parameter_step;"
            "per_parameter_steps_count_only_non_none_gradients"
        ),
        "parameter_count": len(parameter_names),
        "state_parameter_count": len(steps),
        "state_uninitialized_parameter_count": len(uninitialized_names),
        "state_uninitialized_parameter_names": uninitialized_names,
        "minimum_step": min(steps),
        "maximum_step": max(steps),
        "unique_steps": unique_steps,
        "step_histogram": step_histogram,
        "step_summary_by_group": step_summary_by_group,
        "checkpoint_saved_learning_rate": float(param_groups[0]["lr"]),
    }


def main() -> None:
    cli = _parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(
            "formal D1.1 parameter-delta audit requires one visible CUDA device"
        )
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise SystemExit(f"parameter-delta audit source is dirty:\n{status}")

    options = json.loads(cli.options.read_text(encoding="utf-8"))
    identity = json.loads(cli.source_identity.read_text(encoding="utf-8"))
    checkpoint_stat_before = {
        "bytes": cli.checkpoint.stat().st_size,
        "mtime_ns": cli.checkpoint.stat().st_mtime_ns,
        "sha256": _sha256(cli.checkpoint),
    }
    if checkpoint_stat_before["sha256"] != cli.expected_checkpoint_sha256:
        raise RuntimeError("parameter-delta checkpoint SHA-256 mismatch")
    options_sha256 = _sha256(cli.options)
    if options_sha256 != cli.expected_options_sha256:
        raise RuntimeError("parameter-delta options SHA-256 mismatch")
    if options.get("mode") != "train" or options.get("load_model") is not False:
        raise RuntimeError(
            "parameter-delta audit requires the original fresh train construction"
        )
    if int(options.get("random_seed", -1)) != 52:
        raise RuntimeError("parameter-delta audit requires frozen random_seed=52")

    metric_lines = [
        json.loads(line)
        for line in cli.metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(metric_lines) != 1 or int(metric_lines[0].get("epoch", -1)) != 1:
        raise RuntimeError("parameter-delta audit requires one epoch metric row")
    if metric_lines[0].get("test_access") is not False:
        raise RuntimeError("parameter-delta metric row did not certify test_access=false")
    recorded_lr = float(metric_lines[0].get("metrics", {}).get("lr", float("nan")))
    if not math.isfinite(recorded_lr) or recorded_lr <= 0.0:
        raise RuntimeError("parameter-delta metric row has no finite positive LR")

    dataset_cache_paths = {
        "annotation": Path(options["video_anno"]).expanduser().resolve(),
        "train_features": (
            Path(options["video_feature_all_train"]).expanduser().resolve()
        ),
        "video_len": (
            Path(options["video_len_file"].format("train")).expanduser().resolve()
        ),
        "proposal_labels": Path(
            options["ontal_label_file"].format(
                "train",
                options["num_frame"],
                options["num_queries"],
                options["detect_len"],
                options["anti_len"],
                options["max_memory_len"],
                options["p_videos"],
            )
        )
        .expanduser()
        .resolve(),
    }
    for cache_name, cache_path in dataset_cache_paths.items():
        if not cache_path.is_file():
            raise RuntimeError(
                f"parameter-delta audit requires existing {cache_name} cache: "
                f"{cache_path}"
            )
    dataset_cache_stats_before = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": _sha256(path),
        }
        for name, path in dataset_cache_paths.items()
    }

    initial_payload = _reconstruct_initial_state(cli.options, cli.output.parent)
    if int(initial_payload.get("loader_batch_count", -1)) != (
        cli.expected_optimizer_steps
    ):
        raise RuntimeError(
            "reconstructed loader batch count does not match expected optimizer "
            f"steps: {initial_payload.get('loader_batch_count')} != "
            f"{cli.expected_optimizer_steps}"
        )
    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    _validate_inputs(options, checkpoint, identity)
    if int(checkpoint.get("epoch", -1)) != 1:
        raise RuntimeError("parameter-delta audit requires an epoch-one checkpoint")
    terminal_state = checkpoint.get("state_dict")
    if not isinstance(terminal_state, dict) or not terminal_state:
        raise RuntimeError("parameter-delta checkpoint has no model state")
    fresh_state = initial_payload["state_dict"]
    if set(fresh_state) != set(terminal_state):
        raise RuntimeError(
            "fresh and terminal model state keys differ: "
            f"missing={sorted(set(fresh_state) - set(terminal_state))}, "
            f"unexpected={sorted(set(terminal_state) - set(fresh_state))}"
        )
    for name, initial in fresh_state.items():
        terminal = terminal_state[name]
        if initial.shape != terminal.shape or initial.dtype != terminal.dtype:
            raise RuntimeError(
                f"fresh/terminal state contract mismatch for {name}: "
                f"{tuple(initial.shape)}/{initial.dtype} != "
                f"{tuple(terminal.shape)}/{terminal.dtype}"
            )
    group_stats = {
        "all_parameters": _new_delta_stats(),
        "event_transition_head": _new_delta_stats(),
        "event_owner_decoder": _new_delta_stats(),
        "other_event_parameters": _new_delta_stats(),
        "inherited_parent_parameters": _new_delta_stats(),
    }
    per_parameter = []
    parameter_names = [str(name) for name in initial_payload["parameter_names"]]
    if len(set(parameter_names)) != len(parameter_names):
        raise RuntimeError("initialization reconstruction duplicated parameter names")
    for name in parameter_names:
        initial = fresh_state[name]
        if name not in terminal_state:
            raise RuntimeError(f"terminal checkpoint omitted parameter: {name}")
        terminal = terminal_state[name]
        local = _new_delta_stats()
        _accumulate_delta(local, initial, terminal)
        _accumulate_delta(group_stats["all_parameters"], initial, terminal)
        _accumulate_delta(group_stats[_parameter_group(name)], initial, terminal)
        summary = _finalize_delta_stats(local)
        per_parameter.append({"name": name, **summary})

    dataset_cache_stats_after = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": _sha256(path),
        }
        for name, path in dataset_cache_paths.items()
    }
    if dataset_cache_stats_after != dataset_cache_stats_before:
        raise RuntimeError("dataset caches changed during parameter-delta audit")
    checkpoint_stat_after = {
        "bytes": cli.checkpoint.stat().st_size,
        "mtime_ns": cli.checkpoint.stat().st_mtime_ns,
        "sha256": _sha256(cli.checkpoint),
    }
    if checkpoint_stat_after != checkpoint_stat_before:
        raise RuntimeError("checkpoint changed during parameter-delta audit")
    if _sha256(cli.options) != options_sha256:
        raise RuntimeError("options changed during parameter-delta audit")

    result = {
        "status": "PASS",
        "protocol": "eventmatr_d11_epoch1_parameter_delta_audit_v1",
        "test_access": False,
        "checkpoint_updated": False,
        "model_forward_executed": False,
        "optimizer_step_executed": False,
        "strict_causal_paper_result_valid": False,
        "training_source_identity": identity,
        "audit_source_identity": {
            "commit": _git("rev-parse", "HEAD"),
            "tree": _git("rev-parse", "HEAD^{tree}"),
        },
        "initialization_reconstructed_from_exact_training_commit": True,
        "initialization_source_commit": TRAINING_COMMIT,
        "seed": int(options["random_seed"]),
        "construction_order": "seed_then_train_dataset_then_model",
        "dataset_consumed_torch_rng": bool(
            initial_payload["dataset_consumed_torch_rng"]
        ),
        "initialization_video_count": int(initial_payload["video_count"]),
        "initialization_dataset_item_count": int(
            initial_payload["dataset_item_count"]
        ),
        "initialization_loader_batch_size": int(
            initial_payload["loader_batch_size"]
        ),
        "initialization_loader_batch_count": int(
            initial_payload["loader_batch_count"]
        ),
        "checkpoint": {
            "path": str(cli.checkpoint),
            **checkpoint_stat_before,
        },
        "options": {
            "path": str(cli.options),
            "sha256": options_sha256,
        },
        "metrics": {
            "path": str(cli.metrics),
            "sha256": _sha256(cli.metrics),
            "recorded_epoch_average_learning_rate": recorded_lr,
        },
        "optimizer": _optimizer_step_summary(
            checkpoint,
            cli.expected_optimizer_steps,
            parameter_names,
        ),
        "scheduler": checkpoint.get("scheduler"),
        "dataset_caches": dataset_cache_stats_before,
        "parameter_delta_by_group": {
            name: _finalize_delta_stats(stats)
            for name, stats in group_stats.items()
        },
        "largest_relative_parameter_deltas": sorted(
            per_parameter,
            key=lambda row: (
                -float(row["relative_l2_delta"] or 0.0),
                row["name"],
            ),
        )[:25],
        "largest_absolute_parameter_deltas": sorted(
            per_parameter,
            key=lambda row: (
                -float(row["max_absolute_delta"]),
                row["name"],
            ),
        )[:25],
    }
    cli.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(cli.output)
    print(json.dumps(result["parameter_delta_by_group"], sort_keys=True))


if __name__ == "__main__":
    main()
