#!/usr/bin/env python3
"""Atomically preserve validated formal12 checkpoints before calibration."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


CHECKPOINTS = ((3, 2), (6, 5), (9, 8), (12, 11))
EXPECTED_UPDATES = 12 * 2010


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_audit(audit):
    if audit.get("schema_version") != "persistent_binding_training_audit.v1":
        raise ValueError("unexpected training audit schema")
    if audit.get("fit_only") is not True or audit.get("screening_only") is not False:
        raise ValueError("formal12 audit route flags are invalid")
    if audit.get("route_stage") != "persistent_binding_feature_multi_epoch_calibration":
        raise ValueError("formal12 audit route stage mismatch")
    epochs = audit.get("epochs", [])
    if [int(row.get("epoch", -1)) for row in epochs] != list(range(12)):
        raise ValueError("formal12 audit does not contain epochs 0..11")
    totals = audit.get("totals", {})
    if int(totals.get("expected_updates", -1)) != EXPECTED_UPDATES:
        raise ValueError("formal12 expected updates mismatch")
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


def _checkpoint_sources(train_root):
    sources = []
    for human, zero in CHECKPOINTS:
        source = train_root / "checkpoint" / f"epoch_{zero}.pth"
        if not source.is_file():
            raise FileNotFoundError(
                f"epoch {human} checkpoint is missing: {source}"
            )
        sources.append((human, zero, source))
    return sources


def _stage_quarantine(
    *,
    audit_source,
    config,
    output,
    checkpoint_sources,
    arm,
    commit,
    seed,
    validation_error,
):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(
            f"training quarantine output already exists: {output}"
        )
    if not output.parent.is_dir():
        raise FileNotFoundError(
            f"training quarantine parent does not exist: {output.parent}"
        )
    staging = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if staging.exists():
        raise FileExistsError(
            f"training quarantine staging path already exists: {staging}"
        )
    try:
        checkpoint_dir = staging / "checkpoint"
        checkpoint_dir.mkdir(parents=True)
        shutil.copy2(audit_source, staging / "training_audit.json")
        shutil.copy2(config, staging / "config.py")
        checkpoint_rows = []
        for human, zero, source in checkpoint_sources:
            target = checkpoint_dir / source.name
            shutil.copy2(source, target)
            checkpoint_rows.append(
                {
                    "epoch": human,
                    "checkpoint_epoch": zero,
                    "path": str(target.relative_to(staging)),
                    "bytes": target.stat().st_size,
                    "sha256": _sha256(target),
                }
            )
        manifest = {
            "schema_version": "persistent_binding_training_quarantine.v1",
            "arm": arm,
            "seed": int(seed),
            "code_commit": str(commit),
            "validation_error": str(validation_error),
            "training_audit_sha256": _sha256(
                staging / "training_audit.json"
            ),
            "config_sha256": _sha256(staging / "config.py"),
            "checkpoints": checkpoint_rows,
            "calibration_authorized": False,
            "recovery_manifest": False,
            "reporting_accessed": False,
            "threshold_search": False,
            "raw_rgb_authorized": False,
        }
        (staging / "quarantine_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def stage_recovery(
    *,
    train_root,
    config,
    output,
    arm,
    commit,
    seed,
    quarantine=None,
):
    train_root = Path(train_root).resolve()
    config = Path(config).resolve()
    output = Path(output).resolve()
    if arm not in {"fixed", "rematch"}:
        raise ValueError("arm must be fixed or rematch")
    if len(str(commit)) != 40:
        raise ValueError("commit must be a full 40-character hash")
    if output.exists():
        raise FileExistsError(f"recovery output already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"recovery parent does not exist: {output.parent}")
    if not config.is_file():
        raise FileNotFoundError(f"formal12 config is missing: {config}")

    audit_source = train_root / "training_audit.json"
    audit = _load(audit_source)
    checkpoint_sources = _checkpoint_sources(train_root)
    try:
        totals = validate_training_audit(audit)
    except ValueError as error:
        if quarantine is not None:
            _stage_quarantine(
                audit_source=audit_source,
                config=config,
                output=quarantine,
                checkpoint_sources=checkpoint_sources,
                arm=arm,
                commit=commit,
                seed=seed,
                validation_error=error,
            )
        raise

    staging = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if staging.exists():
        raise FileExistsError(f"recovery staging path already exists: {staging}")
    try:
        checkpoint_dir = staging / "checkpoint"
        checkpoint_dir.mkdir(parents=True)
        shutil.copy2(audit_source, staging / "training_audit.json")
        shutil.copy2(config, staging / "config.py")
        checkpoint_rows = []
        for human, zero, source in checkpoint_sources:
            target = checkpoint_dir / source.name
            shutil.copy2(source, target)
            checkpoint_rows.append(
                {
                    "epoch": human,
                    "checkpoint_epoch": zero,
                    "path": str(target.relative_to(staging)),
                    "bytes": target.stat().st_size,
                    "sha256": _sha256(target),
                }
            )
        manifest = {
            "schema_version": "persistent_binding_training_recovery.v1",
            "arm": arm,
            "seed": int(seed),
            "code_commit": str(commit),
            "epochs": 12,
            "checkpoint_epochs": [human for human, _ in CHECKPOINTS],
            "expected_updates": EXPECTED_UPDATES,
            "successful_updates": int(totals["successful_updates"]),
            "scheduler_steps": int(totals["scheduler_steps"]),
            "training_audit_sha256": _sha256(staging / "training_audit.json"),
            "config_sha256": _sha256(staging / "config.py"),
            "checkpoints": checkpoint_rows,
            "calibration_complete": False,
            "reporting_accessed": False,
            "threshold_search": False,
            "raw_rgb_authorized": False,
        }
        (staging / "recovery_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=("fixed", "rematch"), required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--quarantine-output")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = stage_recovery(
        train_root=args.train_root,
        config=args.config,
        output=args.output,
        arm=args.arm,
        commit=args.commit,
        seed=args.seed,
        quarantine=args.quarantine_output,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
