"""Fail closed unless a worker is running the exact clean experiment source.

The identity is deliberately small: Git commit, Git tree, and the SHA-256 of
the frozen scientific manifest.  A generated identity receipt itself may live
inside a temporary test repository and is therefore the only untracked path
ignored by the clean-worktree check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _git(project_dir: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=project_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_status(project_dir: Path, output: Path) -> str:
    status = _git(project_dir, "status", "--porcelain=v1", "--untracked-files=all")
    if not status:
        return ""

    try:
        output_relative = output.resolve().relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        output_relative = None
    lines = []
    for line in status.splitlines():
        # The verifier may be rerun in a temporary contract-test repository.
        # Ignore only its own untracked receipt, never a tracked modification.
        if output_relative is not None and line == f"?? {output_relative}":
            continue
        lines.append(line)
    return "\n".join(lines)


def _expected(cli_value: str | None, env_name: str) -> str | None:
    return cli_value if cli_value is not None else os.environ.get(env_name)


def _load_smoke(path: Path, identity: dict[str, str]) -> dict[str, Any]:
    smoke = json.loads(path.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASS":
        raise RuntimeError(f"real-data smoke is not PASS: {path}")
    if smoke.get("test_access") is not False:
        raise RuntimeError("real-data smoke did not certify test_access=false")
    smoke_identity = smoke.get("source_identity")
    if not isinstance(smoke_identity, dict):
        raise RuntimeError("real-data smoke lacks source_identity")
    for field, value in identity.items():
        if smoke_identity.get(field) != value:
            raise RuntimeError(
                f"real-data smoke source {field} mismatch: "
                f"{smoke_identity.get(field)!r} != {value!r}"
            )
    return smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-tree")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--smoke-receipt", type=Path)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    if not manifest.is_file():
        raise SystemExit(f"manifest not found: {manifest}")

    try:
        status = _clean_status(project_dir, output)
        if status:
            raise RuntimeError(f"source worktree is dirty:\n{status}")

        identity = {
            "commit": _git(project_dir, "rev-parse", "HEAD"),
            "tree": _git(project_dir, "rev-parse", "HEAD^{tree}"),
            "manifest_sha256": _sha256(manifest),
        }
        expected = {
            "commit": _expected(args.expected_commit, "MATR_SOURCE_COMMIT"),
            "tree": _expected(args.expected_tree, "MATR_SOURCE_TREE"),
            "manifest_sha256": _expected(
                args.expected_manifest_sha256, "MATR_MANIFEST_SHA256"
            ),
        }
        for field, expected_value in expected.items():
            if expected_value is not None and identity[field] != expected_value:
                raise RuntimeError(
                    f"source {field} mismatch: {identity[field]!r} != {expected_value!r}"
                )

        smoke_path = args.smoke_receipt
        smoke_summary = None
        if smoke_path is not None:
            smoke_path = smoke_path.resolve()
            smoke = _load_smoke(smoke_path, identity)
            smoke_summary = {
                "path": str(smoke_path),
                "sha256": _sha256(smoke_path),
                "status": smoke["status"],
                "test_access": smoke["test_access"],
            }

        receipt: dict[str, Any] = {
            "status": "PASS",
            "clean": True,
            "status_porcelain": "",
            **identity,
            "manifest": str(manifest),
        }
        if smoke_summary is not None:
            receipt["smoke"] = smoke_summary
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output)
    except Exception as error:
        output.parent.mkdir(parents=True, exist_ok=True)
        failure = {"status": "FAIL", "clean": False, "error": str(error)}
        output.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
