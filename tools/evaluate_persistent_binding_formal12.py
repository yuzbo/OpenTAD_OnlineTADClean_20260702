"""Build and gate the paired epoch-12 calibration-only feature experiment."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from mmengine.config import Config

from opentad.evaluations.mAP import mAP
from opentad.evaluations.online_budgeted_map import OnlineAPBudgeted
from opentad.evaluations.online_instance_metrics import (
    compute_online_instance_metrics,
)


ARM_SCHEMA = "persistent_binding_formal12_arm_result.v1"
GATE_SCHEMA = "persistent_binding_formal12_gate.v1"
CONTRACT_SCHEMA = "persistent_binding_feature_multi_epoch_deployment.v1"
EXPECTED_SEED = 705
EXPECTED_EPOCHS = 12
EXPECTED_UPDATES_PER_EPOCH = 2010
EXPECTED_UPDATES = EXPECTED_EPOCHS * EXPECTED_UPDATES_PER_EPOCH
EXPECTED_CHECKPOINT_EPOCHS = (3, 6, 9, 12)
EXPECTED_TIOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
PAIR_GPU_HOUR_CAP = 16.0
OPERATIONAL_THRESHOLDS = {
    "prediction_gt_ratio_min": 0.25,
    "prediction_gt_ratio_max": 4.0,
    "recall_tiou_0p3_min": 0.25,
}
COMMON_PROVENANCE = (
    "code_commit",
    "annotation_sha256",
    "calibration_manifest_sha256",
    "fit_manifest_sha256",
    "reporting_manifest_sha256",
    "census_sha256",
    "formal_contract_sha256",
    "calibration_protocol_sha256",
)
ARM_PROVENANCE = COMMON_PROVENANCE + (
    "checkpoint_sha256",
    "config_sha256",
    "emission_ledger_sha256",
)


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
        raise ValueError("formal12 evaluation requires a clean worktree")
    return commit


def _finite(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


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
        raise ValueError("formal12 ledger requires a results mapping")
    unknown = sorted(set(results).difference(calibration_ids))
    if unknown:
        raise ValueError(
            f"formal12 ledger contains non-calibration videos: {unknown}"
        )
    leaked = sorted(set(results).intersection(reporting_ids))
    if leaked:
        raise ValueError(f"formal12 ledger accessed reporting videos: {leaked}")
    for video_id, rows in results.items():
        if not isinstance(rows, list):
            raise ValueError(f"ledger rows for {video_id} must be a list")
        for row in rows:
            if str(row.get("video_id")) != str(video_id):
                raise ValueError(f"ledger video identity mismatch for {video_id}")
    return results


def _percentage_metrics(metrics):
    output = {
        "schema_version": "standard_ontad_map.v1",
        "unit": "percentage_points",
        "tiou_thresholds": list(EXPECTED_TIOUS),
        "average_map_pct": float(metrics["average_mAP"]) * 100.0,
        "map_pct": {},
    }
    for threshold in EXPECTED_TIOUS:
        output["map_pct"][f"{threshold:g}"] = (
            float(metrics[f"mAP@{threshold}"]) * 100.0
        )
    return output


def _require_formal_config(cfg, arm):
    expected_binding = {
        "fixed": "fixed_birth_slot",
        "rematch": "prefix_rematch_active_pool",
    }[arm]
    if cfg.get("formal_training_ready") is not True:
        raise ValueError("formal12 config is not registered as training-ready")
    if cfg.get("screening_only") is not False:
        raise ValueError("formal12 config is still marked screening-only")
    if cfg.get("route_stage") != (
        "persistent_binding_feature_multi_epoch_calibration"
    ):
        raise ValueError("formal12 route stage mismatch")
    if cfg.model.trajectory_binding_mode != expected_binding:
        raise ValueError("formal12 arm/config binding mismatch")
    if int(cfg.workflow.end_epoch) != EXPECTED_EPOCHS:
        raise ValueError("formal12 config does not train exactly 12 epochs")
    if int(cfg.workflow.checkpoint_interval) != 3:
        raise ValueError("formal12 checkpoint interval must be three epochs")
    contract = dict(cfg.multi_epoch_training_contract)
    if tuple(contract["checkpoint_epochs"]) != EXPECTED_CHECKPOINT_EPOCHS:
        raise ValueError("formal12 checkpoint epochs changed")
    if contract["checkpoint_selection_split"] != "calibration_only":
        raise ValueError("formal12 checkpoint selection is not calibration-only")
    for field in ("reporting_accessed", "threshold_search", "raw_rgb_authorized"):
        if contract[field] is not False:
            raise ValueError(f"formal12 contract changed {field}")
    for channel in ("birth", "alive", "end"):
        if float(cfg.model.head[f"{channel}_threshold"]) != 0.5:
            raise ValueError(f"formal12 changed the frozen {channel} threshold")
    return expected_binding


def _validate_training_audit(audit, cfg, arm):
    if audit.get("schema_version") != "persistent_binding_training_audit.v1":
        raise ValueError("unexpected formal12 training-audit schema")
    if int(audit.get("seed", -1)) != EXPECTED_SEED:
        raise ValueError("formal12 training seed mismatch")
    if audit.get("fit_only") is not True or audit.get("screening_only") is not False:
        raise ValueError("formal12 audit route flags are invalid")
    if audit.get("route_stage") != cfg.route_stage:
        raise ValueError("formal12 audit route stage mismatch")
    expected_binding = {
        "fixed": "fixed_birth_slot",
        "rematch": "prefix_rematch_active_pool",
    }[arm]
    if audit.get("binding_mode") != expected_binding:
        raise ValueError("formal12 audit binding mode mismatch")
    epochs = audit.get("epochs", [])
    if [int(row.get("epoch", -1)) for row in epochs] != list(
        range(EXPECTED_EPOCHS)
    ):
        raise ValueError("formal12 audit does not contain epochs 0..11")
    totals = audit.get("totals", {})
    if int(totals.get("expected_updates", -1)) != EXPECTED_UPDATES:
        raise ValueError("formal12 expected-update count mismatch")
    for field in ("successful_updates", "scheduler_steps"):
        if int(totals.get(field, -1)) != EXPECTED_UPDATES:
            raise ValueError(f"formal12 {field} mismatch")
    for field in (
        "skipped_updates",
        "gt_supervision_exhaustions",
        "gt_birth_runtime_entry_free_collisions",
    ):
        if int(totals.get(field, -1)) != 0:
            raise ValueError(f"formal12 {field} is nonzero")
    return totals


def build_arm_result(*, repo, config, arm_root, census, formal_contract, arm):
    repo = Path(repo).resolve()
    config = Path(config).resolve()
    arm_root = Path(arm_root).resolve()
    artifacts = arm_root / "artifacts"
    paths = {
        "checkpoint": artifacts / "checkpoint/epoch_12_resume.pth",
        "ledger": artifacts / "calibration_epoch_12_emissions.json",
        "candidate": artifacts / "calibration_candidate_epoch_12.json",
        "training_audit": artifacts / "training_audit.json",
        "resource_report": artifacts / "resource_report.json",
        "calibration_receipt": artifacts / "calibration_receipt.json",
        "promotion": artifacts / "calibration_promotion.json",
        "completion": artifacts / "formal12_arm_completion.json",
        "census": Path(census).resolve(),
        "formal_contract": Path(formal_contract).resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f"formal12 {label} does not exist: {path}")

    cfg = Config.fromfile(config)
    binding_mode = _require_formal_config(cfg, arm)
    commit = _git_commit(repo)
    deployment = _load(paths["formal_contract"], "formal12 contract")
    if deployment.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("unexpected formal12 deployment-contract schema")
    if deployment.get("code_commit") != commit:
        raise ValueError("formal12 deployment commit mismatch")
    expected_deployment = {
        "variant": "boundary_calibration_batched",
        "seed": EXPECTED_SEED,
        "epochs": EXPECTED_EPOCHS,
        "checkpoint_epochs": list(EXPECTED_CHECKPOINT_EPOCHS),
        "initialization": "seed_initialization_not_pilot_resume",
        "input": "fixed_cached_causal_features",
        "fit_only": True,
        "checkpoint_selection_split": "calibration_only",
        "formal_fixed_threshold_gate_epoch": 12,
        "paired_gpu_hour_cap": PAIR_GPU_HOUR_CAP,
        "submit_via_slurm_only": True,
    }
    for field, expected in expected_deployment.items():
        if deployment.get(field) != expected:
            raise ValueError(f"formal12 deployment {field} mismatch")
    if deployment.get("fixed_thresholds") != {
        "birth": 0.5,
        "alive": 0.5,
        "end": 0.5,
    }:
        raise ValueError("formal12 deployment changed frozen thresholds")
    if deployment.get("split_census_sha256") != _sha256(paths["census"]):
        raise ValueError("formal12 deployment census hash mismatch")
    if deployment.get("reporting_accessed") is not False:
        raise ValueError("formal12 deployment accessed reporting")
    if deployment.get("threshold_search") is not False:
        raise ValueError("formal12 deployment performed threshold search")
    if deployment.get("raw_rgb_authorized") is not False:
        raise ValueError("formal12 deployment authorized raw RGB")

    audit = _load(paths["training_audit"], "formal12 training audit")
    totals = _validate_training_audit(audit, cfg, arm)
    census_payload = _load(paths["census"], "split census")
    if census_payload.get("schema_version") != "persistent_binding_split_census.v1":
        raise ValueError("unexpected formal12 census schema")
    if census_payload.get("passed") is not True:
        raise ValueError("formal12 cannot consume a failed census")

    checkpoint_sha = _sha256(paths["checkpoint"])
    ledger_sha = _sha256(paths["ledger"])
    candidate = _load(paths["candidate"], "epoch-12 calibration candidate")
    expected_candidate = {
        "schema_version": "persistent_binding_calibration_candidate.v1",
        "arm": arm,
        "seed": EXPECTED_SEED,
        "epoch": 11,
        "code_commit": commit,
        "config_sha256": _sha256(config),
        "calibration_manifest_sha256": _sha256(cfg.calibration_manifest),
        "census_sha256": _sha256(paths["census"]),
        "calibration_protocol_sha256": _canonical_sha256(
            dict(cfg.calibration_contract)
        ),
        "checkpoint_sha256": checkpoint_sha,
        "emission_ledger_sha256": ledger_sha,
        "reporting_accessed": False,
    }
    for field, expected in expected_candidate.items():
        if candidate.get(field) != expected:
            raise ValueError(f"epoch-12 calibration candidate {field} mismatch")

    receipt = _load(paths["calibration_receipt"], "calibration receipt")
    if receipt.get("schema_version") != "persistent_binding_calibration_receipt.v1":
        raise ValueError("unexpected formal12 calibration-receipt schema")
    if int(receipt.get("candidate_count", -1)) != 4:
        raise ValueError("formal12 calibration did not compare four checkpoints")
    if int(receipt.get("selected_epoch", -1)) not in (2, 5, 8, 11):
        raise ValueError("formal12 selected checkpoint is outside 3/6/9/12")
    if receipt.get("reporting_accessed") is not False:
        raise ValueError("formal12 calibration receipt accessed reporting")
    for field, expected in (
        ("arm", arm),
        ("seed", EXPECTED_SEED),
        ("code_commit", commit),
        ("config_sha256", _sha256(config)),
    ):
        if receipt.get(field) != expected:
            raise ValueError(f"formal12 calibration receipt {field} mismatch")

    promotion = _load(paths["promotion"], "calibration promotion")
    if promotion.get("schema_version") != (
        "persistent_binding_calibration_promotion.v1"
    ):
        raise ValueError("unexpected formal12 promotion schema")
    if promotion.get("arm") != arm or promotion.get("code_commit") != commit:
        raise ValueError("formal12 promotion identity mismatch")
    if promotion.get("epoch12_resume_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("formal12 promoted epoch-12 checkpoint hash mismatch")
    for field in ("reporting_accessed", "threshold_search", "raw_rgb_authorized"):
        if promotion.get(field) is not False:
            raise ValueError(f"formal12 promotion changed {field}")

    completion = _load(paths["completion"], "formal12 arm completion")
    if completion.get("schema_version") != (
        "persistent_binding_feature_multi_epoch_arm.v1"
    ):
        raise ValueError("unexpected formal12 arm-completion schema")
    if completion.get("arm") != arm or completion.get("code_commit") != commit:
        raise ValueError("formal12 arm completion identity mismatch")
    if tuple(row["epoch"] for row in completion["calibration_curve"]) != (
        EXPECTED_CHECKPOINT_EPOCHS
    ):
        raise ValueError("formal12 calibration curve is incomplete")

    resources = _load(paths["resource_report"], "formal12 resource report")
    if resources.get("schema_version") != "persistent_binding_resource.v1":
        raise ValueError("unexpected formal12 resource schema")
    if resources.get("scope") != arm or resources.get("code_commit") != commit:
        raise ValueError("formal12 resource identity mismatch")
    allocated_gpu_hours = _finite(
        resources.get("allocated_gpu_hours"),
        "allocated_gpu_hours",
    )
    if allocated_gpu_hours <= 0:
        raise ValueError("formal12 allocated GPU hours must be positive")

    calibration_ids = _ids(cfg.calibration_manifest)
    reporting_ids = _ids(cfg.reporting_manifest)
    if set(calibration_ids).intersection(reporting_ids):
        raise ValueError("calibration and reporting manifests overlap")
    annotation = _load(cfg.annotation_path, "annotation")
    ground_truth = _filter_ground_truth(annotation, set(calibration_ids))
    ledger_payload = _load(paths["ledger"], "epoch-12 emission ledger")
    results = _validate_ledger(
        ledger_payload,
        set(calibration_ids),
        set(reporting_ids),
    )

    standard = mAP(
        ground_truth_filename=cfg.annotation_path,
        prediction_filename={"results": results},
        subset="training",
        tiou_thresholds=list(EXPECTED_TIOUS),
        allowed_videos=calibration_ids,
        thread=8,
    ).evaluate()
    standard_metrics = _percentage_metrics(standard)
    budget_cfg = dict(cfg.calibration_evaluation)
    budget_cfg.pop("type", None)
    budgeted_metrics = OnlineAPBudgeted(
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
    violation_counts = instance_05["causal_validation"]["violation_counts"]
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
        "checkpoint_sha256": checkpoint_sha,
        "census_sha256": _sha256(paths["census"]),
        "config_sha256": _sha256(config),
        "emission_ledger_sha256": ledger_sha,
        "formal_contract_sha256": _sha256(paths["formal_contract"]),
        "calibration_protocol_sha256": _canonical_sha256(
            dict(cfg.calibration_contract)
        ),
    }
    gate_row = {
        "arm": arm,
        "binding_mode": binding_mode,
        "seed": EXPECTED_SEED,
        "training_epochs": EXPECTED_EPOCHS,
        "fixed_threshold": 0.5,
        "metric_schema": standard_metrics["schema_version"],
        "instance_metric_schema": instance_05["schema_version"],
        "map_unit": standard_metrics["unit"],
        "tiou_thresholds": list(EXPECTED_TIOUS),
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
        "committed_predictions": prediction_count,
        "expected_updates": int(totals["expected_updates"]),
        "successful_updates": int(totals["successful_updates"]),
        "scheduler_steps": int(totals["scheduler_steps"]),
        "skipped_updates": int(totals["skipped_updates"]),
        "gt_supervision_exhaustions": int(totals["gt_supervision_exhaustions"]),
        "gt_birth_runtime_entry_free_collisions": int(
            totals["gt_birth_runtime_entry_free_collisions"]
        ),
        "allocated_gpu_hours": allocated_gpu_hours,
        "reporting_accessed": False,
        "threshold_search": False,
        "raw_rgb_authorized": False,
        "provenance": provenance,
    }
    for field in (
        "average_map_pct",
        "duplicate_rate",
        "fragmentation_rate",
        "prediction_gt_ratio",
        "recall_tiou_0p3",
    ):
        _finite(gate_row[field], field)
    return {
        "schema_version": ARM_SCHEMA,
        "purpose": "epoch12_fixed_threshold_calibration_gate",
        "arm": arm,
        "binding_mode": binding_mode,
        "seed": EXPECTED_SEED,
        "training_epochs": EXPECTED_EPOCHS,
        "checkpoint_epoch": 12,
        "fixed_thresholds": {"birth": 0.5, "alive": 0.5, "end": 0.5},
        "standard_metrics": standard_metrics,
        "budgeted_metrics": budgeted_metrics,
        "instance_metrics_tiou_0p3": instance_03,
        "instance_metrics_tiou_0p5": instance_05,
        "latency": ledger_payload.get("summary"),
        "calibration_curve": completion["calibration_curve"],
        "selected_checkpoint_epoch": int(completion["selected_epoch"]),
        "gate_row": gate_row,
        "reporting_accessed": False,
        "threshold_search": False,
        "effectiveness_claim_authorized": False,
        "raw_rgb_authorized": False,
    }


def _normalize_gate_row(payload, arm):
    if payload.get("schema_version") != ARM_SCHEMA:
        raise ValueError(f"{arm} formal12 result schema mismatch")
    if payload.get("arm") != arm:
        raise ValueError(f"{arm} formal12 result arm mismatch")
    if int(payload.get("training_epochs", -1)) != EXPECTED_EPOCHS:
        raise ValueError(f"{arm} formal12 result epoch count mismatch")
    if int(payload.get("checkpoint_epoch", -1)) != 12:
        raise ValueError(f"{arm} formal12 result checkpoint mismatch")
    if payload.get("fixed_thresholds") != {
        "birth": 0.5,
        "alive": 0.5,
        "end": 0.5,
    }:
        raise ValueError(f"{arm} formal12 frozen thresholds changed")
    for field in (
        "reporting_accessed",
        "threshold_search",
        "effectiveness_claim_authorized",
        "raw_rgb_authorized",
    ):
        if payload.get(field) is not False:
            raise ValueError(f"{arm} formal12 result changed {field}")
    row = dict(payload["gate_row"])
    if int(row.get("seed", -1)) != EXPECTED_SEED:
        raise ValueError(f"{arm} formal12 seed mismatch")
    if int(row.get("training_epochs", -1)) != EXPECTED_EPOCHS:
        raise ValueError(f"{arm} formal12 epoch count mismatch")
    if float(row.get("fixed_threshold", -1)) != 0.5:
        raise ValueError(f"{arm} formal12 threshold mismatch")
    expected_binding = {
        "fixed": "fixed_birth_slot",
        "rematch": "prefix_rematch_active_pool",
    }[arm]
    if row.get("binding_mode") != expected_binding:
        raise ValueError(f"{arm} formal12 binding mismatch")
    if row.get("metric_schema") != "standard_ontad_map.v1":
        raise ValueError(f"{arm} formal12 metric schema changed")
    if row.get("instance_metric_schema") != "online_instance_metrics.v2":
        raise ValueError(f"{arm} formal12 instance metric schema changed")
    if row.get("map_unit") != "percentage_points":
        raise ValueError(f"{arm} formal12 mAP unit changed")
    for field in (
        "reporting_accessed",
        "threshold_search",
        "raw_rgb_authorized",
    ):
        if row.get(field) is not False:
            raise ValueError(f"{arm} formal12 gate row changed {field}")
    if tuple(float(value) for value in row["tiou_thresholds"]) != EXPECTED_TIOUS:
        raise ValueError(f"{arm} formal12 tIoU thresholds changed")
    for field in (
        "average_map_pct",
        "duplicate_rate",
        "fragmentation_rate",
        "prediction_gt_ratio",
        "recall_tiou_0p3",
        "allocated_gpu_hours",
    ):
        row[field] = _finite(row[field], field)
    if not 0 <= row["average_map_pct"] <= 100:
        raise ValueError(f"{arm} formal12 mAP is outside [0, 100]")
    if min(
        row["duplicate_rate"],
        row["fragmentation_rate"],
        row["prediction_gt_ratio"],
        row["recall_tiou_0p3"],
    ) < 0:
        raise ValueError(f"{arm} formal12 contains a negative rate")
    if row["recall_tiou_0p3"] > 1:
        raise ValueError(f"{arm} formal12 recall exceeds one")
    if row["allocated_gpu_hours"] <= 0:
        raise ValueError(f"{arm} formal12 GPU hours must be positive")
    for field in (
        "protocol_violations",
        "committed_predictions",
        "expected_updates",
        "successful_updates",
        "scheduler_steps",
        "skipped_updates",
        "gt_supervision_exhaustions",
        "gt_birth_runtime_entry_free_collisions",
    ):
        value = _finite(row[field], field)
        if not value.is_integer() or value < 0:
            raise ValueError(f"{arm} formal12 {field} must be a nonnegative integer")
        row[field] = int(value)
    provenance = dict(row["provenance"])
    for field in ARM_PROVENANCE:
        value = provenance.get(field)
        length = 40 if field == "code_commit" else 64
        if not isinstance(value, str) or re.fullmatch(
            rf"[0-9a-f]{{{length}}}",
            value,
        ) is None:
            raise ValueError(f"{arm} formal12 provenance.{field} is invalid")
    row["provenance"] = provenance
    row["identity_error"] = 0.5 * (
        row["duplicate_rate"] + row["fragmentation_rate"]
    )
    return row


def evaluate_gate(fixed_payload, rematch_payload):
    fixed = _normalize_gate_row(fixed_payload, "fixed")
    rematch = _normalize_gate_row(rematch_payload, "rematch")
    for field in COMMON_PROVENANCE:
        if fixed["provenance"][field] != rematch["provenance"][field]:
            raise ValueError(f"formal12 pair differs on provenance.{field}")

    integrity_failures = []
    operational_failures = []
    for arm, row in (("fixed", fixed), ("rematch", rematch)):
        for field in (
            "protocol_violations",
            "skipped_updates",
            "gt_supervision_exhaustions",
            "gt_birth_runtime_entry_free_collisions",
        ):
            if row[field] != 0:
                integrity_failures.append(f"{arm}: {field}={row[field]}")
        if not (
            row["expected_updates"]
            == row["successful_updates"]
            == row["scheduler_steps"]
            == EXPECTED_UPDATES
        ):
            integrity_failures.append(
                f"{arm}: updates expected={row['expected_updates']} "
                f"successful={row['successful_updates']} "
                f"scheduler={row['scheduler_steps']}"
            )
        if row["committed_predictions"] <= 0:
            operational_failures.append(
                f"{arm}: no committed predictions at epoch 12"
            )
        if not (
            OPERATIONAL_THRESHOLDS["prediction_gt_ratio_min"]
            <= row["prediction_gt_ratio"]
            <= OPERATIONAL_THRESHOLDS["prediction_gt_ratio_max"]
        ):
            operational_failures.append(
                f"{arm}: prediction_gt_ratio={row['prediction_gt_ratio']}"
            )
        if row["recall_tiou_0p3"] < OPERATIONAL_THRESHOLDS[
            "recall_tiou_0p3_min"
        ]:
            operational_failures.append(
                f"{arm}: recall_tiou_0p3={row['recall_tiou_0p3']}"
            )

    actual_pair_gpu_hours = (
        fixed["allocated_gpu_hours"] + rematch["allocated_gpu_hours"]
    )
    budget_pass = 0 < actual_pair_gpu_hours <= PAIR_GPU_HOUR_CAP
    if not budget_pass:
        integrity_failures.append(
            f"pair: allocated_gpu_hours={actual_pair_gpu_hours}"
        )
    technical_pass = not integrity_failures
    operational_pass = technical_pass and not operational_failures
    rematch_identity = rematch["identity_error"]
    relative_reduction = (
        None
        if rematch_identity == 0
        else (
            rematch_identity - fixed["identity_error"]
        )
        / rematch_identity
    )
    return {
        "schema_version": GATE_SCHEMA,
        "seed": EXPECTED_SEED,
        "training_epochs": EXPECTED_EPOCHS,
        "checkpoint_epoch": 12,
        "fixed_threshold": 0.5,
        "operational_thresholds": dict(OPERATIONAL_THRESHOLDS),
        "technical_pass": technical_pass,
        "technical_failures": integrity_failures,
        "operational_pass": operational_pass,
        "operational_failures": operational_failures,
        "formal_fixed_threshold_pass": operational_pass,
        "budget_pass": budget_pass,
        "paired_gpu_hour_cap": PAIR_GPU_HOUR_CAP,
        "actual_pair_gpu_hours": actual_pair_gpu_hours,
        "fixed": fixed,
        "rematch": rematch,
        "directional_diagnostics": {
            "fixed_identity_error": fixed["identity_error"],
            "rematch_identity_error": rematch_identity,
            "identity_error_relative_reduction": relative_reduction,
            "fixed_average_map_pct": fixed["average_map_pct"],
            "rematch_average_map_pct": rematch["average_map_pct"],
            "map_delta_points": (
                fixed["average_map_pct"] - rematch["average_map_pct"]
            ),
        },
        "reporting_accessed": False,
        "threshold_search": False,
        "effectiveness_claim_authorized": False,
        "raw_rgb_authorized": False,
        "next_gate": (
            "paired_multi_seed_feature_protocol"
            if operational_pass
            else "model_optimization_on_fit_and_calibration_only"
        ),
    }


def _write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--fixed-config", required=True)
    parser.add_argument("--rematch-config", required=True)
    parser.add_argument("--fixed-root", required=True)
    parser.add_argument("--rematch-root", required=True)
    parser.add_argument("--census", required=True)
    parser.add_argument("--formal-contract", required=True)
    parser.add_argument("--fixed-output", required=True)
    parser.add_argument("--rematch-output", required=True)
    parser.add_argument("--gate-output", required=True)
    args = parser.parse_args()
    fixed = build_arm_result(
        repo=args.repo,
        config=args.fixed_config,
        arm_root=args.fixed_root,
        census=args.census,
        formal_contract=args.formal_contract,
        arm="fixed",
    )
    rematch = build_arm_result(
        repo=args.repo,
        config=args.rematch_config,
        arm_root=args.rematch_root,
        census=args.census,
        formal_contract=args.formal_contract,
        arm="rematch",
    )
    gate = evaluate_gate(fixed, rematch)
    _write_new(args.fixed_output, fixed)
    _write_new(args.rematch_output, rematch)
    _write_new(args.gate_output, gate)
    json.dump(gate, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
