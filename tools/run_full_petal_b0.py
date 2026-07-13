#!/usr/bin/env python3
"""Run the locked Full PETAL CPU B0 matrix and emit hash-bound evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tokenize
import xml.etree.ElementTree as ET


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
B0_SCHEMA = "full-petal-b0-v1"
TEST_REPORT_SCHEMA = "full-petal-b0-test-report-v1"
AUDIT_REPORT_SCHEMA = "full-petal-b0-audit-report-v1"
TORCH_TARGETS = (
    "tests/test_full_petal_training_transaction.py",
    "tests/test_full_petal_detector_contracts.py",
    "tests/test_persistent_event_set_detector.py",
    "tests/test_persistent_event_set_head.py",
    "tests/test_optimizer_audit.py",
    "tests/test_full_petal_training_cost_controls.py",
    "tests/test_training_update_audit.py",
)
TORCH_IMPORT = re.compile(r"(^|\s)(import torch|from torch)", re.MULTILINE)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--torch-python", type=Path, required=True)
    parser.add_argument("--dependency-site", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _junit_counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {key: 0 for key in ("collected", "failed", "errors", "skipped")}
    for suite in suites:
        totals["collected"] += int(suite.attrib.get("tests", 0))
        totals["failed"] += int(suite.attrib.get("failures", 0))
        totals["errors"] += int(suite.attrib.get("errors", 0))
        totals["skipped"] += int(suite.attrib.get("skipped", 0))
    totals["passed"] = totals["collected"] - sum(
        totals[key] for key in ("failed", "errors", "skipped")
    )
    return totals


def _run_pytest(name, command, output_dir, env):
    log_path = output_dir / f"{name}.log"
    junit_path = output_dir / f"{name}.junit.xml"
    command = [*command, "--junitxml", str(junit_path)]
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
        "COMMAND=" + json.dumps(command) + "\n\nSTDOUT\n" + result.stdout
        + "\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    if junit_path.is_file():
        counts = _junit_counts(junit_path)
    else:
        junit_path.write_text("<testsuites/>\n", encoding="utf-8")
        counts = {
            "collected": 0,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
        }
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
        "command": command,
        "python_executable": command[0],
        **counts,
        "log_path": str(log_path.resolve()),
        "log_sha256": _sha256(log_path),
        "junit_path": str(junit_path.resolve()),
        "junit_sha256": _sha256(junit_path),
    }


def _write_check(output_dir, name, command, status, content):
    log_path = output_dir / f"audit-{name}.log"
    log_path.write_text(content, encoding="utf-8")
    return {
        "name": name,
        "status": status,
        "command": list(command),
        "log_path": str(log_path.resolve()),
        "log_sha256": _sha256(log_path),
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
    status = "PASS" if tracked and not failures else "FAIL"
    return _write_check(
        output_dir,
        "python_syntax",
        [sys.executable, "compile(all tracked Python sources)"],
        status,
        content + "\n",
    )


def _git_check(output_dir, name, *args):
    result = _git(*args, check=False)
    status = "PASS" if result.returncode == 0 and not result.stdout.strip() else "FAIL"
    return _write_check(
        output_dir,
        name,
        ["git", "-C", str(ROOT), *args],
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
    dependency_sites = [path.resolve() for path in args.dependency_site]
    for dependency_site in dependency_sites:
        if not dependency_site.is_dir():
            raise SystemExit(f"dependency site does not exist: {dependency_site}")

    pure_targets = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        if not TORCH_IMPORT.search(path.read_text(encoding="utf-8")):
            pure_targets.append(str(path.relative_to(ROOT)))
    common_pytest = ["-q", "-p", "no:cacheprovider"]
    pure_command = [sys.executable, "-m", "pytest", *pure_targets, *common_pytest]

    wrapper = ROOT / "tools" / "testing" / "run_isolated_torch_pytest.py"
    torch_command = [str(torch_python), str(wrapper)]
    for dependency_site in dependency_sites:
        torch_command.extend(["--dependency-site", str(dependency_site)])
    torch_command.extend(["--", *TORCH_TARGETS, *common_pytest])

    base_env = dict(os.environ)
    base_env["PYTHONDONTWRITEBYTECODE"] = "1"
    suites = [
        _run_pytest("pure_contracts", pure_command, output_dir, base_env),
    ]
    torch_env = dict(base_env)
    torch_env["PYTHONNOUSERSITE"] = "1"
    suites.append(
        _run_pytest("focused_torch_contracts", torch_command, output_dir, torch_env)
    )

    totals = {
        field: sum(suite[field] for suite in suites)
        for field in ("collected", "passed", "failed", "errors", "skipped")
    }
    tests_pass = all(suite["status"] == "PASS" for suite in suites)
    test_report = {
        "schema_version": TEST_REPORT_SCHEMA,
        "status": "PASS" if tests_pass else "FAIL",
        "commit_sha": commit,
        **totals,
        "suites": suites,
    }
    test_report_path = output_dir / "b0-test-report.json"
    _write_json(test_report_path, test_report)

    checks = [
        _syntax_check(output_dir),
        _git_check(output_dir, "git_diff_check", "diff", "--check"),
        _git_check(output_dir, "repository_clean_after", "status", "--porcelain"),
    ]
    final_commit, final_status = _repository_state()
    audit_pass = (
        final_commit == commit
        and not final_status
        and all(check["status"] == "PASS" for check in checks)
    )
    audit_report = {
        "schema_version": AUDIT_REPORT_SCHEMA,
        "status": "PASS" if audit_pass else "FAIL",
        "commit_sha": commit,
        "blocking_findings": 0 if audit_pass else 1,
        "protocol_violations": 0,
        "checks": checks,
    }
    audit_report_path = output_dir / "b0-audit-report.json"
    _write_json(audit_report_path, audit_report)

    b0_pass = tests_pass and audit_pass
    b0 = {
        "schema_version": B0_SCHEMA,
        "status": "PASS" if b0_pass else "FAIL",
        "commit_sha": commit,
        "test_count": totals["collected"],
        "blocking_findings": 0 if b0_pass else 1,
        "protocol_violations": 0,
        "test_report_path": str(test_report_path.resolve()),
        "test_report_sha256": _sha256(test_report_path),
        "audit_report_path": str(audit_report_path.resolve()),
        "audit_report_sha256": _sha256(audit_report_path),
    }
    b0_path = output_dir / "b0.json"
    _write_json(b0_path, b0)
    print(f"FULL_PETAL_B0={b0_path}")
    print(f"FULL_PETAL_B0_STATUS={b0['status']}")
    return 0 if b0_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
