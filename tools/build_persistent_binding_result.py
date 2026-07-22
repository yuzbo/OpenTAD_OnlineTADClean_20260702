"""Join one locked reporting run into the repository-owned result artifact."""

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


SCHEMA_VERSION = "persistent_binding_result.v1"
TIOU_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        raise ValueError("reporting manifest repeats video IDs")
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
        raise ValueError("result construction requires a clean Git worktree")
    return commit


def _filter_ground_truth(annotation, allowed):
    database = {
        video_id: video
        for video_id, video in annotation["database"].items()
        if video_id in allowed
    }
    if set(database) != set(allowed):
        missing = sorted(set(allowed).difference(database))
        raise ValueError(f"reporting annotations miss videos: {missing}")
    return {"database": database}


def _validate_ledger(ledger, allowed):
    results = ledger.get("results")
    if not isinstance(results, dict):
        raise ValueError("emission ledger must contain a results mapping")
    unknown = sorted(set(results).difference(allowed))
    if unknown:
        raise ValueError(f"ledger contains non-reporting videos: {unknown}")
    for video_id, rows in results.items():
        if not isinstance(rows, list):
            raise ValueError(f"ledger rows for {video_id} must be a list")
        for row in rows:
            if str(row.get("video_id")) != str(video_id):
                raise ValueError(
                    f"ledger video identity mismatch for {video_id}"
                )
    return results


def _percentage_metrics(metrics):
    output = {
        "schema_version": "standard_ontad_map.v1",
        "unit": "percentage_points",
        "tiou_thresholds": list(TIOU_THRESHOLDS),
        "average_map_pct": float(metrics["average_mAP"]) * 100.0,
        "map_pct": {},
    }
    for threshold in TIOU_THRESHOLDS:
        output["map_pct"][f"{threshold:g}"] = (
            float(metrics[f"mAP@{threshold}"]) * 100.0
        )
    return output


def build_result(
    *,
    repo,
    config,
    checkpoint,
    ledger,
    training_audit,
    census,
    calibration_receipt,
    reporting_receipt,
    resource_report,
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
        "reporting_receipt": Path(reporting_receipt).resolve(),
        "resource_report": Path(resource_report).resolve(),
    }
    for label, path in paths.items():
        if not path.exists():
            raise ValueError(f"{label} does not exist: {path}")
    cfg = Config.fromfile(paths["config"])
    binding_mode = str(cfg.model.trajectory_binding_mode)
    expected_mode = {
        "fixed": "fixed_birth_slot",
        "rematch": "prefix_rematch_active_pool",
    }.get(arm)
    if expected_mode is None or binding_mode != expected_mode:
        raise ValueError("arm does not match config trajectory_binding_mode")
    thresholds = tuple(
        float(value)
        for value in cfg.reporting_evaluation.tiou_thresholds
    )
    if thresholds != TIOU_THRESHOLDS:
        raise ValueError("reporting tIoU thresholds differ from the frozen schema")

    commit = _git_commit(paths["repo"])
    checkpoint_sha256 = _sha256(paths["checkpoint"])
    ledger_sha256 = _sha256(paths["ledger"])
    calibration = _load(
        paths["calibration_receipt"],
        "calibration receipt",
    )
    reporting = _load(paths["reporting_receipt"], "reporting receipt")
    audit = _load(paths["training_audit"], "training audit")
    census_payload = _load(paths["census"], "census")
    resources = _load(paths["resource_report"], "resource report")
    if calibration.get("schema_version") != "persistent_binding_calibration_receipt.v1":
        raise ValueError("unexpected calibration receipt schema")
    if reporting.get("schema_version") != "persistent_binding_reporting_receipt.v1":
        raise ValueError("unexpected reporting receipt schema")
    if reporting.get("status") != "completed":
        raise ValueError("reporting receipt is not completed")
    if reporting.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("reporting receipt checkpoint hash mismatch")
    if reporting.get("emission_ledger_sha256") != ledger_sha256:
        raise ValueError("reporting receipt ledger hash mismatch")
    if reporting.get("code_commit") != commit:
        raise ValueError("reporting receipt code commit mismatch")
    if calibration.get("selected_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("calibration did not select this checkpoint")
    if calibration.get("reporting_accessed") is not False:
        raise ValueError("calibration receipt accessed reporting data")
    if audit.get("schema_version") != "persistent_binding_training_audit.v1":
        raise ValueError("unexpected training audit schema")
    if int(audit.get("seed", -1)) != int(seed):
        raise ValueError("training audit seed mismatch")
    if audit.get("fit_only") is not True:
        raise ValueError("training audit does not prove fit-only execution")
    if census_payload.get("schema_version") != "persistent_binding_split_census.v1":
        raise ValueError("unexpected census schema")
    if census_payload.get("passed") is not True:
        raise ValueError("formal result cannot consume a failed census")

    reporting_ids = _ids(cfg.reporting_manifest)
    allowed = set(reporting_ids)
    annotation = _load(cfg.annotation_path, "annotation")
    ground_truth = _filter_ground_truth(annotation, allowed)
    ledger_payload = _load(paths["ledger"], "emission ledger")
    results = _validate_ledger(ledger_payload, allowed)

    standard_evaluator = mAP(
        ground_truth_filename=cfg.annotation_path,
        prediction_filename={"results": results},
        subset="validation",
        tiou_thresholds=list(TIOU_THRESHOLDS),
        allowed_videos=reporting_ids,
        thread=8,
    )
    standard_metrics = _percentage_metrics(standard_evaluator.evaluate())
    budget_cfg = dict(cfg.reporting_evaluation)
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
    protocol_violations = sum(
        int(value)
        for value in instance_05["causal_validation"][
            "violation_counts"
        ].values()
    )
    ground_truth_count = int(instance_03["counts"]["ground_truth"])
    prediction_count = int(instance_03["counts"]["emissions"])
    recall_03 = (
        float(instance_03["counts"]["matched_ground_truth"])
        / ground_truth_count
        if ground_truth_count
        else 0.0
    )
    totals = audit["totals"]
    provenance = {
        "code_commit": commit,
        "annotation_sha256": _sha256(cfg.annotation_path),
        "calibration_protocol_sha256": _canonical_sha256(
            dict(cfg.calibration_contract)
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "census_sha256": _sha256(paths["census"]),
        "config_sha256": _sha256(paths["config"]),
        "emission_ledger_sha256": ledger_sha256,
        "reporting_manifest_sha256": _sha256(cfg.reporting_manifest),
    }
    gate_row = {
        "seed": int(seed),
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
        "candidate_arbitration_suppressions": int(
            totals["candidate_arbitration_suppressions"]
        ),
        "candidate_cancellations": int(
            totals["candidate_cancellations"]
        ),
        "active_abandonments": int(totals["active_abandonments"]),
        "deferred_birth_due_to_release": int(
            totals["deferred_birth_due_to_release"]
        ),
        "committed_predictions": prediction_count,
        "expected_updates": int(totals["expected_updates"]),
        "successful_updates": int(totals["successful_updates"]),
        "scheduler_steps": int(totals["scheduler_steps"]),
        "skipped_updates": int(totals["skipped_updates"]),
        "provenance": provenance,
    }
    for value in (
        gate_row["average_map_pct"],
        gate_row["duplicate_rate"],
        gate_row["fragmentation_rate"],
        gate_row["prediction_gt_ratio"],
        gate_row["recall_tiou_0p3"],
    ):
        if not math.isfinite(float(value)):
            raise ValueError("result artifact contains a non-finite metric")
    return {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "binding_mode": binding_mode,
        "seed": int(seed),
        "standard_metrics": standard_metrics,
        "budgeted_metrics": budgeted,
        "instance_metrics_tiou_0p3": instance_03,
        "instance_metrics_tiou_0p5": instance_05,
        "latency": ledger_payload.get("summary"),
        "training_audit": audit,
        "split_census": census_payload,
        "resource_use": resources,
        "calibration_receipt": calibration,
        "reporting_receipt": reporting,
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
    parser.add_argument("--reporting-receipt", required=True)
    parser.add_argument("--resource-report", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--arm", choices=("fixed", "rematch"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_result(
        repo=args.repo,
        config=args.config,
        checkpoint=args.checkpoint,
        ledger=args.ledger,
        training_audit=args.training_audit,
        census=args.census,
        calibration_receipt=args.calibration_receipt,
        reporting_receipt=args.reporting_receipt,
        resource_report=args.resource_report,
        seed=args.seed,
        arm=args.arm,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    print(f"RESULT_ARTIFACT={output}")
    print(f"RESULT_ARTIFACT_SHA256={_sha256(output)}")
    json.dump(payload["gate_row"], sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
