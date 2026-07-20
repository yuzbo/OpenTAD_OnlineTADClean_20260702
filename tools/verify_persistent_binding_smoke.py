import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402

from opentad.evaluations import build_evaluator  # noqa: E402
from opentad.models import build_detector  # noqa: E402
from opentad.utils import set_seed  # noqa: E402
from opentad.utils.online_protocol import (  # noqa: E402
    summarize_emission_ledger,
    validate_emission_ledger_summary,
)


class SmokeVerificationError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify the standard-runner persistent-binding Slurm smoke"
    )
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-ledger", required=True)
    parser.add_argument("--reload-ledger", required=True)
    parser.add_argument("--allowed-videos", required=True)
    parser.add_argument("--direct-report", action="append", default=[])
    parser.add_argument("--seed", type=int, default=705)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-empty-emissions", action="store_true")
    return parser.parse_args()


def _require(condition, message):
    if not condition:
        raise SmokeVerificationError(message)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allowed_videos(path):
    with open(path, "r", encoding="utf-8") as handle:
        values = [line.strip() for line in handle if line.strip()]
    _require(values, "allowed-videos file is empty")
    _require(len(values) == len(set(values)), "allowed-videos file contains duplicates")
    return values


def _verify_checkpoint(path, cfg, seed):
    payload = torch.load(path, map_location="cpu")
    required = {"epoch", "state_dict", "optimizer", "scheduler"}
    _require(required.issubset(payload), f"checkpoint is missing {sorted(required - set(payload))}")
    _require(int(payload["epoch"]) == 0, "smoke checkpoint must be epoch 0")
    state_dict = payload["state_dict"]
    _require(state_dict, "checkpoint state_dict is empty")
    optimizer_state = payload["optimizer"].get("state", {})
    _require(optimizer_state, "standard runner checkpoint contains no optimizer update state")
    tensor_count = 0
    for name, value in state_dict.items():
        if torch.is_tensor(value):
            tensor_count += 1
            _require(bool(torch.isfinite(value).all().item()), f"non-finite checkpoint tensor: {name}")
    _require(tensor_count > 0, "checkpoint has no tensor state")

    set_seed(seed)
    initial_state = build_detector(cfg.model).state_dict()
    normalized_state = {
        name[len("module.") :] if name.startswith("module.") else name: value
        for name, value in state_dict.items()
    }
    _require(
        set(normalized_state) == set(initial_state),
        "checkpoint state keys differ from a fresh model of the same config",
    )
    changed_tensors = []
    total_delta_l2 = 0.0
    for name, initial in initial_state.items():
        saved = normalized_state[name]
        if not torch.is_tensor(initial) or not torch.is_tensor(saved):
            continue
        delta = saved.detach().cpu().float() - initial.detach().cpu().float()
        delta_l2 = float(delta.norm().item())
        if delta_l2 > 0:
            changed_tensors.append(name)
            total_delta_l2 += delta_l2
    _require(changed_tensors, "standard runner checkpoint is identical to seeded initialization")
    return {
        "epoch": int(payload["epoch"]),
        "state_tensor_count": tensor_count,
        "optimizer_state_entries": len(optimizer_state),
        "changed_state_tensors": len(changed_tensors),
        "total_state_delta_l2": total_delta_l2,
        "sha256": _sha256(path),
    }


def _verify_direct_reports(paths):
    reports = []
    modes = set()
    for path in paths:
        report = _load_json(path)
        _require(report.get("passed") is True, f"direct smoke did not pass: {path}")
        _require(report.get("mode") == "train_step", f"unexpected direct smoke mode: {path}")
        _require(
            int(report.get("dropped_gt_birth_targets", -1)) == 0,
            f"direct smoke dropped GT birth targets: {path}",
        )
        _require(
            report.get("all_update_audits_passed") is True,
            f"direct smoke parameter update audit failed: {path}",
        )
        modes.add(report.get("binding_mode"))
        reports.append(
            {
                "path": str(Path(path).resolve()),
                "binding_mode": report.get("binding_mode"),
                "chunks_processed": int(report.get("chunks_processed", 0)),
                "dropped_gt_birth_targets": int(report["dropped_gt_birth_targets"]),
                "runtime_capacity_exhaustions": int(
                    report.get("runtime_capacity_exhaustions", 0)
                ),
            }
        )
    _require(
        modes == {"fixed_birth_slot", "prefix_rematch_active_pool"},
        f"direct smoke must cover both binding modes, got {sorted(str(item) for item in modes)}",
    )
    return reports


def _verify_ledgers(train_path, reload_path, allowed, require_emissions):
    train_payload = _load_json(train_path)
    reload_payload = _load_json(reload_path)
    _require("results" in train_payload and "summary" in train_payload, "train ledger schema is incomplete")
    _require(
        "results" in reload_payload and "summary" in reload_payload,
        "reload ledger schema is incomplete",
    )
    _require(
        train_payload["results"] == reload_payload["results"],
        "checkpoint reload changed deterministic streaming emissions",
    )
    _require(
        train_payload["summary"] == reload_payload["summary"],
        "checkpoint reload changed the emission summary",
    )
    _require(
        set(reload_payload["results"]) == set(allowed),
        "ledger video IDs do not exactly match the smoke allow-list",
    )
    recomputed = summarize_emission_ledger(reload_payload["results"])
    validate_emission_ledger_summary(recomputed)
    _require(recomputed == reload_payload["summary"], "saved ledger summary is stale or inconsistent")
    if require_emissions:
        _require(recomputed["num_emissions"] > 0, "smoke model produced no final intervals")
    return reload_payload, recomputed


def _verify(args):
    cfg = Config.fromfile(args.config)
    _require(cfg.inference.load_from_raw_predictions is False, "raw prediction loading is forbidden")
    _require(cfg.raw_video_finetuning is False, "raw-RGB training must remain disabled")
    _require(cfg.solver.amp is False, "persistent-binding feature smoke must use stable FP32")
    _require(cfg.workflow.fail_on_nonfinite is True, "non-finite training must be a hard failure")
    allowed = _allowed_videos(args.allowed_videos)
    checkpoint = _verify_checkpoint(args.checkpoint, cfg, args.seed)
    direct_reports = _verify_direct_reports(args.direct_report)
    ledger, emission_summary = _verify_ledgers(
        args.train_ledger,
        args.reload_ledger,
        allowed,
        require_emissions=not args.allow_empty_emissions,
    )

    evaluation = dict(cfg.evaluation)
    evaluation.update(
        prediction_filename=ledger,
        allowed_videos=allowed,
    )
    evaluator = build_evaluator(evaluation)
    metrics = evaluator.evaluate()
    _require(metrics["num_ground_truth"] > 0, "smoke reporting video has no ground truth")
    _require(
        metrics["num_predictions"] == emission_summary["num_emissions"],
        "evaluator prediction count differs from the immutable ledger",
    )
    return {
        "passed": True,
        "config": str(Path(args.config).resolve()),
        "checkpoint": checkpoint,
        "allowed_videos": allowed,
        "direct_reports": direct_reports,
        "train_ledger": {
            "path": str(Path(args.train_ledger).resolve()),
            "sha256": _sha256(args.train_ledger),
        },
        "reload_ledger": {
            "path": str(Path(args.reload_ledger).resolve()),
            "sha256": _sha256(args.reload_ledger),
        },
        "checkpoint_reload_emissions_identical": True,
        "emission_summary": emission_summary,
        "evaluation": metrics,
        "formal_training_ready": bool(cfg.formal_training_ready),
        "raw_video_finetuning": bool(cfg.raw_video_finetuning),
    }


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = _verify(args)
    except Exception as error:
        payload = {
            "passed": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
        }
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
            newline="\n",
        )
        raise
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
