"""Run the reporting split exactly once and leave an auditable receipt."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path, label):
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _git_state(repo):
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
        raise ValueError("locked reporting requires a clean Git worktree")
    return commit


def _validate_command(command, checkpoint):
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a reporting command is required after --")
    try:
        role_index = command.index("--evaluation-role")
    except ValueError as exc:
        raise ValueError(
            "reporting command must include --evaluation-role reporting"
        ) from exc
    if role_index + 1 >= len(command) or command[role_index + 1] != "reporting":
        raise ValueError(
            "reporting command must include --evaluation-role reporting"
        )
    try:
        checkpoint_index = command.index("--checkpoint")
    except ValueError as exc:
        raise ValueError(
            "reporting command must include an explicit --checkpoint"
        ) from exc
    if checkpoint_index + 1 >= len(command):
        raise ValueError("--checkpoint requires a path")
    commanded = Path(command[checkpoint_index + 1]).resolve()
    if commanded != Path(checkpoint).resolve():
        raise ValueError(
            "reporting command checkpoint differs from the calibrated checkpoint"
        )
    return command


def run_locked_report(
    *,
    repo,
    config,
    checkpoint,
    calibration_receipt,
    reporting_manifest,
    ledger,
    lock,
    receipt,
    command,
):
    paths = {
        "repo": Path(repo).resolve(),
        "config": Path(config).resolve(),
        "checkpoint": Path(checkpoint).resolve(),
        "calibration_receipt": Path(calibration_receipt).resolve(),
        "reporting_manifest": Path(reporting_manifest).resolve(),
        "ledger": Path(ledger).resolve(),
        "lock": Path(lock).resolve(),
        "receipt": Path(receipt).resolve(),
    }
    for label in (
        "repo",
        "config",
        "checkpoint",
        "calibration_receipt",
        "reporting_manifest",
    ):
        if not paths[label].exists():
            raise ValueError(f"{label} does not exist: {paths[label]}")
    if paths["lock"].exists() or paths["receipt"].exists():
        raise ValueError(
            "reporting is already locked or has a prior receipt; refusing rerun"
        )
    calibration = _load_json(
        paths["calibration_receipt"],
        "calibration receipt",
    )
    if calibration.get("schema_version") != "persistent_binding_calibration_receipt.v1":
        raise ValueError("unexpected calibration receipt schema")
    checkpoint_sha256 = _sha256(paths["checkpoint"])
    if calibration.get("selected_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(
            "calibration receipt does not select the requested checkpoint"
        )
    if calibration.get("reporting_accessed") is not False:
        raise ValueError(
            "calibration receipt must state reporting_accessed=false"
        )
    command = _validate_command(list(command), paths["checkpoint"])
    commit = _git_state(paths["repo"])
    started_at = datetime.now(timezone.utc).isoformat()
    lock_payload = {
        "schema_version": "persistent_binding_reporting_lock.v1",
        "status": "started",
        "started_at_utc": started_at,
        "code_commit": commit,
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_receipt_sha256": _sha256(
            paths["calibration_receipt"]
        ),
        "reporting_manifest_sha256": _sha256(
            paths["reporting_manifest"]
        ),
        "command": command,
    }
    paths["lock"].parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        paths["lock"],
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump(lock_payload, file, indent=2, sort_keys=True)

    completed = subprocess.run(command, cwd=paths["repo"], check=False)
    receipt_payload = {
        **lock_payload,
        "schema_version": "persistent_binding_reporting_receipt.v1",
        "status": "completed" if completed.returncode == 0 else "failed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": int(completed.returncode),
        "config_sha256": _sha256(paths["config"]),
    }
    if completed.returncode == 0:
        if not paths["ledger"].is_file():
            receipt_payload["status"] = "failed"
            receipt_payload["error"] = "reporting command produced no ledger"
        else:
            receipt_payload["emission_ledger_sha256"] = _sha256(
                paths["ledger"]
            )
    paths["receipt"].parent.mkdir(parents=True, exist_ok=True)
    with paths["receipt"].open("x", encoding="utf-8") as file:
        json.dump(receipt_payload, file, indent=2, sort_keys=True)
    if receipt_payload["status"] != "completed":
        raise RuntimeError(
            "locked reporting failed; lock and receipt are retained for audit"
        )
    return receipt_payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-receipt", required=True)
    parser.add_argument("--reporting-manifest", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = run_locked_report(
        repo=args.repo,
        config=args.config,
        checkpoint=args.checkpoint,
        calibration_receipt=args.calibration_receipt,
        reporting_manifest=args.reporting_manifest,
        ledger=args.ledger,
        lock=args.lock,
        receipt=args.receipt,
        command=args.command,
    )
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
