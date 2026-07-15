#!/usr/bin/env python3
"""Run the exhaustive Full PETAL CPU B0 matrix and sign its evidence root."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tokenize


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.utils.full_petal_attestation import (  # noqa: E402
    AttestationError,
    _sign_payload,
)
from opentad.utils.full_petal_b0 import (  # noqa: E402
    B0_ATTESTATION_ROLE,
    B0_AUDIT_REPORT_SCHEMA,
    B0_SCHEMA,
    B0_TEST_REPORT_SCHEMA,
    canonical_json_sha256,
    junit_cases,
    sha256_file,
    validate_manifest,
)


DEFAULT_MANIFEST = ROOT / "tools" / "testing" / "full_petal_b0_manifest.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--torch-python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--attestation-private-key", type=Path, required=True)
    parser.add_argument("--attestation-key-id", required=True)
    return parser.parse_args(argv)


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(*args, check=True):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _repository_state():
    commit = _git("rev-parse", "HEAD").stdout.strip()
    status = _git("status", "--porcelain").stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("repository HEAD is not a full lowercase git SHA")
    return commit, status


def _is_relative_to(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_manifest(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to load B0 manifest {path}: {exc}") from exc
    validate_manifest(payload, repository_root=ROOT)
    return payload


def _actual_command(canonical, *, torch_python, junit_path):
    replacements = {
        "$TORCH_PYTHON": str(torch_python),
        "$REPO_ROOT": str(ROOT),
        "$JUNIT": str(junit_path),
    }
    return [replacements.get(item, item) for item in canonical]


def _run_suite(suite, output_dir, torch_python, env):
    name = suite["name"]
    log_path = output_dir / f"{name}.log"
    junit_path = output_dir / f"{name}.junit.xml"
    command = _actual_command(
        suite["canonical_argv"],
        torch_python=torch_python,
        junit_path=junit_path,
    )
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_path.write_text(
        "CANONICAL_ARGV="
        + json.dumps(suite["canonical_argv"], separators=(",", ":"))
        + "\nACTUAL_ARGV="
        + json.dumps(command, separators=(",", ":"))
        + "\nRETURN_CODE="
        + str(result.returncode)
        + "\n\nSTDOUT\n"
        + result.stdout
        + "\nSTDERR\n"
        + result.stderr,
        encoding="utf-8",
    )
    if junit_path.is_file():
        try:
            counts, cases = junit_cases(junit_path)
        except Exception as exc:
            counts = {
                "collected": 0,
                "passed": 0,
                "failed": 0,
                "errors": 1,
                "skipped": 0,
            }
            cases = [{"classname": "b0.runner", "name": f"junit_parse_error:{exc}"}]
    else:
        junit_path.write_text("<testsuites/>\n", encoding="utf-8")
        counts = {
            "collected": 0,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
        }
        cases = []
    status = (
        "PASS"
        if result.returncode == 0
        and counts["collected"] > 0
        and not any(counts[key] for key in ("failed", "errors", "skipped"))
        else "FAIL"
    )
    return {
        "name": name,
        "status": status,
        "canonical_argv": suite["canonical_argv"],
        "python_executable": str(torch_python),
        **counts,
        "log_path": log_path.name,
        "log_sha256": sha256_file(log_path),
        "junit_path": junit_path.name,
        "junit_sha256": sha256_file(junit_path),
        "testcase_manifest_sha256": canonical_json_sha256(cases),
    }


def _write_check(output_dir, name, canonical_argv, status, content):
    log_path = output_dir / f"audit-{name}.log"
    log_path.write_text(content, encoding="utf-8")
    return {
        "name": name,
        "status": status,
        "canonical_argv": list(canonical_argv),
        "log_path": log_path.name,
        "log_sha256": sha256_file(log_path),
    }


def _syntax_check(output_dir):
    tracked = _git("ls-files", "--", "*.py").stdout.splitlines()
    failures = []
    for relative_path in tracked:
        path = ROOT / relative_path
        try:
            with tokenize.open(path) as handle:
                compile(handle.read(), str(path), "exec")
        except Exception as exc:
            failures.append(f"{relative_path}: {exc!r}")
    content = f"tracked_python_files={len(tracked)}\n" + "\n".join(failures)
    return _write_check(
        output_dir,
        "python_syntax",
        ["$B0_PYTHON", "compile", "$ALL_TRACKED_PYTHON"],
        "PASS" if tracked and not failures else "FAIL",
        content + "\n",
    )


def _git_check(output_dir, name, canonical_argv, *args):
    result = _git(*args, check=False)
    status = "PASS" if result.returncode == 0 and not result.stdout.strip() else "FAIL"
    return _write_check(
        output_dir,
        name,
        canonical_argv,
        status,
        result.stdout + result.stderr,
    )


def main(argv=None):
    args = parse_args(argv)
    commit, initial_status = _repository_state()
    if initial_status:
        raise SystemExit("Full PETAL B0 requires a clean committed checkout")

    output_dir = args.output_dir.resolve()
    if _is_relative_to(output_dir, ROOT.resolve()):
        raise SystemExit("B0 evidence must be written outside the repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"B0 output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_python = args.torch_python.resolve()
    if not torch_python.is_file():
        raise SystemExit(f"Torch Python does not exist: {torch_python}")
    if not args.attestation_private_key.resolve().is_file():
        raise SystemExit("B0 attestation private key does not exist")
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file() or not _is_relative_to(manifest_path, ROOT.resolve()):
        raise SystemExit("B0 manifest must be a tracked repository file")
    manifest = _load_manifest(manifest_path)
    evidence_manifest_path = output_dir / "b0-manifest.json"
    shutil.copyfile(manifest_path, evidence_manifest_path)

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
    tests_pass = all(suite["status"] == "PASS" for suite in suites)
    manifest_sha = sha256_file(evidence_manifest_path)
    test_report = {
        "schema_version": B0_TEST_REPORT_SCHEMA,
        "status": "PASS" if tests_pass else "FAIL",
        "commit_sha": commit,
        "manifest_sha256": manifest_sha,
        **totals,
        "suites": suites,
    }
    test_report_path = output_dir / "b0-test-report.json"
    _write_json(test_report_path, test_report)

    checks = [
        _syntax_check(output_dir),
        _git_check(
            output_dir,
            "git_diff_check",
            ["git", "-C", "$REPO_ROOT", "diff", "--check"],
            "diff",
            "--check",
        ),
        _git_check(
            output_dir,
            "repository_clean_after",
            ["git", "-C", "$REPO_ROOT", "status", "--porcelain"],
            "status",
            "--porcelain",
        ),
    ]
    final_commit, final_status = _repository_state()
    audit_pass = (
        final_commit == commit
        and not final_status
        and all(check["status"] == "PASS" for check in checks)
    )
    audit_report = {
        "schema_version": B0_AUDIT_REPORT_SCHEMA,
        "status": "PASS" if audit_pass else "FAIL",
        "commit_sha": commit,
        "manifest_sha256": manifest_sha,
        "blocking_findings": 0 if audit_pass else 1,
        "protocol_violations": 0,
        "checks": checks,
    }
    audit_report_path = output_dir / "b0-audit-report.json"
    _write_json(audit_report_path, audit_report)

    b0_pass = tests_pass and audit_pass
    unsigned_b0 = {
        "schema_version": B0_SCHEMA,
        "status": "PASS" if b0_pass else "FAIL",
        "commit_sha": commit,
        "test_count": totals["collected"],
        "blocking_findings": 0 if b0_pass else 1,
        "protocol_violations": 0,
        "manifest_path": evidence_manifest_path.name,
        "manifest_sha256": manifest_sha,
        "test_report_path": test_report_path.name,
        "test_report_sha256": sha256_file(test_report_path),
        "audit_report_path": audit_report_path.name,
        "audit_report_sha256": sha256_file(audit_report_path),
    }
    try:
        b0 = _sign_payload(
            unsigned_b0,
            private_key_path=args.attestation_private_key,
            key_id=args.attestation_key_id,
            role=B0_ATTESTATION_ROLE,
        )
    except AttestationError as exc:
        raise SystemExit(f"failed to attest B0 evidence: {exc}") from exc
    b0_path = output_dir / "b0.json"
    _write_json(b0_path, b0)
    print(f"FULL_PETAL_B0={b0_path}")
    print(f"FULL_PETAL_B0_SHA256={sha256_file(b0_path)}")
    print(f"FULL_PETAL_B0_STATUS={unsigned_b0['status']}")
    return 0 if b0_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
