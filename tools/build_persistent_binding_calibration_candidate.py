"""Build a checkpoint-selection candidate from calibration-only emissions."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mmengine.config import Config

from opentad.evaluations.online_budgeted_map import OnlineAPBudgeted


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _load(path):
    with Path(path).open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _ids(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def _git_commit(repo):
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("calibration candidate requires a clean worktree")
    return commit


def build_candidate(
    *,
    repo,
    config,
    checkpoint,
    ledger,
    census,
    seed,
    arm,
    epoch,
):
    cfg = Config.fromfile(config)
    expected_mode = {
        "fixed": "fixed_birth_slot",
        "rematch": "prefix_rematch_active_pool",
    }[arm]
    if cfg.model.trajectory_binding_mode != expected_mode:
        raise ValueError("arm does not match config")
    census_payload = _load(census)
    if census_payload.get("passed") is not True:
        raise ValueError("calibration cannot consume a failed census")
    ledger_payload = _load(ledger)
    results = ledger_payload.get("results")
    if not isinstance(results, dict):
        raise ValueError("calibration ledger requires a results mapping")
    calibration_ids = _ids(cfg.calibration_manifest)
    reporting_ids = _ids(cfg.reporting_manifest)
    if set(results).difference(calibration_ids):
        raise ValueError("calibration ledger contains a non-calibration video")
    if set(results).intersection(reporting_ids):
        raise ValueError("calibration ledger accessed reporting videos")
    evaluation_cfg = dict(cfg.calibration_evaluation)
    evaluation_cfg.pop("type", None)
    metrics = OnlineAPBudgeted(
        prediction_filename={"results": results},
        **evaluation_cfg,
    ).evaluate()
    metric_name = str(cfg.calibration_contract.selection_metric)
    if metric_name not in metrics:
        raise ValueError(f"calibration metric is absent: {metric_name}")
    checkpoint = Path(checkpoint).resolve()
    return {
        "schema_version": "persistent_binding_calibration_candidate.v1",
        "arm": arm,
        "binding_mode": expected_mode,
        "seed": int(seed),
        "epoch": int(epoch),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "metric_name": metric_name,
        "metric_value": float(metrics[metric_name]),
        "metric_unit": "fraction",
        "code_commit": _git_commit(repo),
        "config_sha256": _sha256(config),
        "calibration_manifest_sha256": _sha256(cfg.calibration_manifest),
        "census_sha256": _sha256(census),
        "calibration_protocol_sha256": _canonical_sha256(
            dict(cfg.calibration_contract)
        ),
        "emission_ledger_sha256": _sha256(ledger),
        "reporting_accessed": False,
        "calibration_metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--census", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=("fixed", "rematch"), required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_candidate(
        repo=args.repo,
        config=args.config,
        checkpoint=args.checkpoint,
        ledger=args.ledger,
        census=args.census,
        seed=args.seed,
        arm=args.arm,
        epoch=args.epoch,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
