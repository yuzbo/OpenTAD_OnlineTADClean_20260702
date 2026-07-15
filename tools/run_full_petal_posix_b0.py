#!/usr/bin/env python3
"""Run and sign the target-Linux Full PETAL B0 leaf."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.utils.evidence_bundle import publish_exclusive_file  # noqa: E402
from opentad.utils.full_petal_attestation import AttestationError  # noqa: E402
from opentad.utils.full_petal_b0 import B0_POSIX_LEAF_SCHEMA  # noqa: E402
from opentad.utils.full_petal_role_signing import sign_posix_b0_leaf  # noqa: E402
from run_full_petal_b0 import (  # noqa: E402
    DEFAULT_MANIFEST,
    _is_relative_to,
    _load_manifest,
    _repository_state,
    _run_suite,
)


def _publish_json(path, payload):
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    publish_exclusive_file(path, encoded)
    return encoded


def _python_platform(torch_python):
    command = [
        str(torch_python),
        "-c",
        (
            "import json,platform,sys,torch;"
            "print(json.dumps({'machine':platform.machine(),"
            "'python_version':platform.python_version(),"
            "'torch_version':torch.__version__,"
            "'sys_platform':sys.platform},sort_keys=True))"
        ),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
        identity = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot establish target Python/Torch identity: {exc}") from exc
    if identity.get("sys_platform") != "linux":
        raise SystemExit("POSIX B0 requires a Linux Torch runtime")
    return {
        "os_name": "posix",
        "sys_platform": "linux",
        "machine": str(identity.get("machine") or platform.machine()),
        "python_version": str(identity.get("python_version") or ""),
        "torch_version": str(identity.get("torch_version") or ""),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--torch-python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--attestation-private-key", type=Path, required=True)
    parser.add_argument("--attestation-key-id", required=True)
    args = parser.parse_args(argv)

    if os.name != "posix" or sys.platform != "linux":
        raise SystemExit("POSIX B0 leaf can only be issued on Linux")
    commit, initial_status = _repository_state()
    clean_before = not bool(initial_status)
    if not clean_before:
        raise SystemExit("POSIX B0 requires a clean committed checkout")
    output_dir = args.output_dir.resolve()
    if _is_relative_to(output_dir, ROOT.resolve()):
        raise SystemExit("POSIX B0 evidence must remain outside the repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"POSIX B0 output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_python = args.torch_python.resolve(strict=True)
    private_key = args.attestation_private_key.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    if not _is_relative_to(manifest_path, ROOT.resolve()):
        raise SystemExit("POSIX B0 manifest must be a tracked repository file")
    manifest, manifest_bytes = _load_manifest(manifest_path)

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    suites = [
        _run_suite(suite, output_dir, torch_python, env)
        for suite in manifest["suites"]
    ]
    totals = {
        field: sum(suite[field] for suite in suites)
        for field in ("collected", "passed", "failed", "errors", "skipped")
    }
    final_commit, final_status = _repository_state()
    clean_after = final_commit == commit and not bool(final_status)
    passed = (
        clean_before
        and clean_after
        and totals["collected"] > 0
        and totals["collected"] == totals["passed"]
        and totals["failed"] == totals["errors"] == totals["skipped"] == 0
        and all(suite["status"] == "PASS" for suite in suites)
    )
    leaf = {
        "schema_version": B0_POSIX_LEAF_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "commit_sha": commit,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "platform": _python_platform(torch_python),
        "repository_clean_before": clean_before,
        "repository_clean_after": clean_after,
        **totals,
        "suites": suites,
    }
    if not passed:
        failure_path = output_dir / "posix-b0-failure.json"
        _publish_json(failure_path, leaf)
        print(f"FULL_PETAL_POSIX_B0_FAILURE={failure_path}")
        return 1
    try:
        signed = sign_posix_b0_leaf(
            leaf,
            private_key_path=private_key,
            key_id=args.attestation_key_id,
        )
    except AttestationError as exc:
        raise SystemExit(f"failed to attest POSIX B0 leaf: {exc}") from exc
    leaf_path = output_dir / "posix-b0.json"
    leaf_bytes = _publish_json(leaf_path, signed)
    print(f"FULL_PETAL_POSIX_B0={leaf_path}")
    print(f"FULL_PETAL_POSIX_B0_SHA256={hashlib.sha256(leaf_bytes).hexdigest()}")
    print("FULL_PETAL_POSIX_B0_STATUS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
