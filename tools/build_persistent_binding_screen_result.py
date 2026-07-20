"""Build a calibration-only result for the registered seed-705 screen."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from mmengine.config import Config

from opentad.evaluations.mAP import mAP
from opentad.evaluations.online_budgeted_map import OnlineAPBudgeted
from opentad.evaluations.online_instance_metrics import (
    compute_online_instance_metrics,
)


SCHEMA_VERSION = "persistent_binding_screen_result.v1"
TIOU_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


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


def _load(path, label):
    with Path(path).open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _ids(path):
    with Path(path).open("r", encoding="utf-8") as file:
        values = [line.strip() for line in file if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"manifest repeats video IDs: {path}")
    return values


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
        raise ValueError("screen result construction requires a clean worktree")
    return commit


def _filter_ground_truth(annotation, allowed):
    database = {
        video_id: video
        for video_id, video in annotation["database"].items()
        if video_id in allowed
    }
    if set(database) != set(allowed):
        missing = sorted(set(allowed).difference(database))
        raise ValueError(f"calibration annotations miss videos: {missing}")
    return {"database": database}


def _validate_ledger(ledger, calibration_ids, reporting_ids):
    results = ledger.get("results")
    if not isinstance(results, dict):
        raise ValueError("screen ledger requires a results mapping")
    unknown = sorted(set(results).difference(calibration_ids))
    if unknown:
        raise ValueError(f"screen ledger contains non-calibration videos: {unknown}")
    leaked = sorted(set(results).intersection(reporting_ids))
    if leaked:
        raise ValueError(f"screen ledger accessed reporting videos: {leaked}")
    for video_id, rows in results.items():
        if not isinstance(rows, list):
            raise ValueError(f"ledger rows for {video_id} must be a list")
        for row in rows:
            if str(row.get("video_id")) != str(video_id):
                raise ValueError(f"ledger video identity mismatch for {video_id}")
    return results


def _percentage_metrics(metrics):
    result = {
        "schema_version": "standard_ontad_map.v1",
        "unit": "percentage_points",
        "tiou_thresholds": list(TIOU_THRESHOLDS),
        "average_map_pct": float(metrics["average_mAP"]) * 100.0,
        "map_pct": {},
    }
    for threshold in TIOU_THRESHOLDS:
        result["map_pct"][f"{threshold:g}"] = (
            float(metrics[f"mAP@{threshold}"]) * 100.0
        )
    return result


def build_screen_result(
    *,
    repo,
    config,
    checkpoint,
    ledger,
    training_audit,
    census,
    calibration_receipt,
    resource_report,
    profile_gate,
    smoke_gate,
    seed,
    arm,
):
    paths = {
        "repo": Path(repo).resolve(),
        "config": Path(config).resolve(),
        "checkpoint": Path(checkpoint).resolve(),
        "ledger": Path(ledger).resolve(),
        "training_audit": Path(training_audit).resolve(),
        "census": Path(census).resolve(),
        "calibration_receipt": Path(calibration_receipt).resolve(),
        "resource_report": Path(resource_report).resolve(),
        "profile_gate": Path(profile_gate).resolve(),
        "smoke_gate": Path(smoke_gate).resolve(),
    }
    for label, path in paths.items():
        if not path.exists():
            raise ValueError(f"{label} does not exist: {path}")

    cfg = Config.fromfile(paths["config"])
    expected_mode = {
        "fixed": "fixed_birth_slot",
        "rematch": "prefix_rematch_active_pool",
    }.get(arm)
    if expected_mode is None or cfg.model.trajectory_binding_mode != expected_mode:
        raise ValueError("screen arm does not match config")
    if cfg.get("screening_only") is not True:
        raise ValueError("screen result requires screening_only=True")
    if cfg.get("screening_training_ready") is not True:
        raise ValueError("screen config is not registered as ready")
    contract = dict(cfg.screening_contract)
    if int(seed) != int(contract["seed"]) or int(contract["epochs"]) != 1:
        raise ValueError("screen seed/epoch contract mismatch")
    if bool(contract["effectiveness_claim_authorized"]):
        raise ValueError("technical screen cannot authorize effectiveness")
    if bool(contract["raw_rgb_authorized"]):
        raise ValueError("technical screen cannot authorize raw RGB")

    commit = _git_commit(paths["repo"])
    checkpoint_sha256 = _sha256(paths["checkpoint"])
    ledger_sha256 = _sha256(paths["ledger"])
    audit = _load(paths["training_audit"], "training audit")
    census_payload = _load(paths["census"], "split census")
    calibration = _load(
        paths["calibration_receipt"],
        "calibration receipt",
    )
    resources = _load(paths["resource_report"], "resource report")
    profile = _load(paths["profile_gate"], "screen profile gate")
    smoke = _load(paths["smoke_gate"], "smoke gate")

    if audit.get("schema_version") != "persistent_binding_training_audit.v1":
        raise ValueError("unexpected training audit schema")
    if int(audit.get("seed", -1)) != int(seed):
        raise ValueError("training audit seed mismatch")
    if audit.get("fit_only") is not True:
        raise ValueError("screen training audit is not fit-only")
    if audit.get("screening_only") is not True:
        raise ValueError("training audit does not identify a screen run")
    if audit.get("route_stage") != cfg.route_stage:
        raise ValueError("training audit route stage mismatch")
    if audit.get("binding_mode") != expected_mode:
        raise ValueError("training audit binding mode mismatch")
    if len(audit.get("epochs", ())) != 1:
        raise ValueError("screen training audit must contain exactly one epoch")
    totals = audit.get("totals", {})
    if int(totals.get("expected_updates", 0)) <= 0:
        raise ValueError("screen training contains no optimizer updates")
    if census_payload.get("schema_version") != "persistent_binding_split_census.v1":
        raise ValueError("unexpected census schema")
    if census_payload.get("passed") is not True:
        raise ValueError("screen cannot consume a failed census")
    if calibration.get("schema_version") != "persistent_binding_calibration_receipt.v1":
        raise ValueError("unexpected calibration receipt schema")
    expected_calibration = {
        "arm": arm,
        "seed": int(seed),
        "code_commit": commit,
        "config_sha256": _sha256(paths["config"]),
        "calibration_manifest_sha256": _sha256(
            cfg.calibration_manifest
        ),
        "census_sha256": _sha256(paths["census"]),
        "calibration_protocol_sha256": _canonical_sha256(
            dict(cfg.calibration_contract)
        ),
        "selection_metric": str(
            cfg.calibration_contract.selection_metric
        ),
        "selected_emission_ledger_sha256": ledger_sha256,
    }
    for field, expected in expected_calibration.items():
        if calibration.get(field) != expected:
            raise ValueError(
                f"calibration receipt {field} mismatch"
            )
    if calibration.get("selected_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("calibration did not select the screen checkpoint")
    if calibration.get("reporting_accessed") is not False:
        raise ValueError("screen calibration accessed reporting data")
    if int(calibration.get("selected_epoch", -1)) != 0:
        raise ValueError("one-epoch screen must select epoch_0")
    if int(calibration.get("candidate_count", -1)) != 1:
        raise ValueError("one-epoch screen must have one calibration candidate")
    if resources.get("schema_version") != "persistent_binding_resource.v1":
        raise ValueError("unexpected resource-report schema")
    if resources.get("scope") != arm:
        raise ValueError("resource scope does not match screen arm")
    if resources.get("code_commit") != commit:
        raise ValueError("resource report code commit mismatch")
    if profile.get("passed") is not True or int(profile.get("epochs", -1)) != 1:
        raise ValueError("screen profile gate did not authorize one epoch")
    if (
        float(profile["estimated_pair_gpu_hours"]["with_safety_factor"])
        > float(profile["paired_gpu_hour_cap"])
    ):
        raise ValueError("screen profile exceeds its paired GPU-hour cap")
    if smoke.get("passed") is not True:
        raise ValueError("screen requires a passed repaired smoke")

    calibration_ids = _ids(cfg.calibration_manifest)
    reporting_ids = _ids(cfg.reporting_manifest)
    if set(calibration_ids).intersection(reporting_ids):
        raise ValueError("calibration and reporting manifests overlap")
    annotation = _load(cfg.annotation_path, "annotation")
    ground_truth = _filter_ground_truth(annotation, set(calibration_ids))
    ledger_payload = _load(paths["ledger"], "emission ledger")
    results = _validate_ledger(
        ledger_payload,
        set(calibration_ids),
        set(reporting_ids),
    )

    standard = mAP(
        ground_truth_filename=cfg.annotation_path,
        prediction_filename={"results": results},
        subset="training",
        tiou_thresholds=list(TIOU_THRESHOLDS),
        allowed_videos=calibration_ids,
        thread=8,
    ).evaluate()
    standard_metrics = _percentage_metrics(standard)
    budget_cfg = dict(cfg.calibration_evaluation)
    budget_cfg.pop("type", None)
    budgeted = OnlineAPBudgeted(
        prediction_filename={"results": results},
        **budget_cfg,
    ).evaluate()
    instance_03 = compute_online_instance_metrics(
        ground_truth,
        ledger_payload,
        tiou_threshold=0.3,
        fps=float(cfg.fps),
    )
    instance_05 = compute_online_instance_metrics(
        ground_truth,
        ledger_payload,
        tiou_threshold=0.5,
        fps=float(cfg.fps),
    )
    violation_counts = instance_05["causal_validation"][
        "violation_counts"
    ]
    protocol_violations = sum(int(value) for value in violation_counts.values())
    ground_truth_count = int(instance_03["counts"]["ground_truth"])
    prediction_count = int(instance_03["counts"]["emissions"])
    recall_03 = (
        float(instance_03["counts"]["matched_ground_truth"])
        / ground_truth_count
        if ground_truth_count
        else 0.0
    )
    provenance = {
        "code_commit": commit,
        "annotation_sha256": _sha256(cfg.annotation_path),
        "calibration_manifest_sha256": _sha256(cfg.calibration_manifest),
        "fit_manifest_sha256": _sha256(cfg.fit_core_manifest),
        "reporting_manifest_sha256": _sha256(cfg.reporting_manifest),
        "checkpoint_sha256": checkpoint_sha256,
        "census_sha256": _sha256(paths["census"]),
        "config_sha256": _sha256(paths["config"]),
        "emission_ledger_sha256": ledger_sha256,
        "profile_gate_sha256": _sha256(paths["profile_gate"]),
        "smoke_gate_sha256": _sha256(paths["smoke_gate"]),
        "screening_contract_sha256": _canonical_sha256(contract),
    }
    gate_row = {
        "arm": arm,
        "binding_mode": expected_mode,
        "seed": int(seed),
        "screen_epochs": 1,
        "metric_schema": standard_metrics["schema_version"],
        "instance_metric_schema": instance_05["schema_version"],
        "map_unit": standard_metrics["unit"],
        "tiou_thresholds": list(TIOU_THRESHOLDS),
        "average_map_pct": standard_metrics["average_map_pct"],
        "duplicate_rate": float(instance_05["duplicate_rate"]),
        "fragmentation_rate": float(instance_05["fragmentation_rate"]),
        "prediction_gt_ratio": (
            float(prediction_count) / ground_truth_count
            if ground_truth_count
            else 0.0
        ),
        "recall_tiou_0p3": recall_03,
        "protocol_violations": protocol_violations,
        "gt_supervision_exhaustions": int(
            totals["gt_supervision_exhaustions"]
        ),
        "gt_birth_runtime_entry_free_collisions": int(
            totals["gt_birth_runtime_entry_free_collisions"]
        ),
        "committed_predictions": prediction_count,
        "expected_updates": int(totals["expected_updates"]),
        "successful_updates": int(totals["successful_updates"]),
        "scheduler_steps": int(totals["scheduler_steps"]),
        "skipped_updates": int(totals["skipped_updates"]),
        "reporting_accessed": False,
        "effectiveness_claim_authorized": False,
        "raw_rgb_authorized": False,
        "allocated_gpu_hours": float(resources["allocated_gpu_hours"]),
        "projected_pair_gpu_hours": float(
            profile["estimated_pair_gpu_hours"]["with_safety_factor"]
        ),
        "provenance": provenance,
    }
    for field in (
        "average_map_pct",
        "duplicate_rate",
        "fragmentation_rate",
        "prediction_gt_ratio",
        "recall_tiou_0p3",
        "allocated_gpu_hours",
        "projected_pair_gpu_hours",
    ):
        if not math.isfinite(float(gate_row[field])):
            raise ValueError(f"screen result contains non-finite {field}")

    return {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "binding_mode": expected_mode,
        "seed": int(seed),
        "purpose": "convergence_and_non_degeneracy_only",
        "reporting_accessed": False,
        "effectiveness_claim_authorized": False,
        "raw_rgb_authorized": False,
        "standard_metrics": standard_metrics,
        "budgeted_metrics": budgeted,
        "instance_metrics_tiou_0p3": instance_03,
        "instance_metrics_tiou_0p5": instance_05,
        "latency": ledger_payload.get("summary"),
        "training_audit": audit,
        "split_census": census_payload,
        "calibration_receipt": calibration,
        "resource_use": resources,
        "profile_gate": profile,
        "provenance": provenance,
        "gate_row": gate_row,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--training-audit", required=True)
    parser.add_argument("--census", required=True)
    parser.add_argument("--calibration-receipt", required=True)
    parser.add_argument("--resource-report", required=True)
    parser.add_argument("--profile-gate", required=True)
    parser.add_argument("--smoke-gate", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--arm", choices=("fixed", "rematch"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_screen_result(
        repo=args.repo,
        config=args.config,
        checkpoint=args.checkpoint,
        ledger=args.ledger,
        training_audit=args.training_audit,
        census=args.census,
        calibration_receipt=args.calibration_receipt,
        resource_report=args.resource_report,
        profile_gate=args.profile_gate,
        smoke_gate=args.smoke_gate,
        seed=args.seed,
        arm=args.arm,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    print(f"SCREEN_RESULT={output}")
    print(f"SCREEN_RESULT_SHA256={_sha256(output)}")
    json.dump(payload["gate_row"], sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
