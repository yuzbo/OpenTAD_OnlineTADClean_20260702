"""Canonical Full PETAL B0 manifest and evidence validation."""

from __future__ import annotations

import ast
import hashlib
import hmac
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

from .full_petal_attestation import AttestationError, verify_payload


B0_SCHEMA = "full-petal-b0-v2"
B0_TEST_REPORT_SCHEMA = "full-petal-b0-test-report-v2"
B0_AUDIT_REPORT_SCHEMA = "full-petal-b0-audit-report-v2"
B0_MANIFEST_SCHEMA = "full-petal-b0-manifest-v1"
B0_ATTESTATION_ROLE = "b0-runner"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


class B0EvidenceError(ValueError):
    """Raised when B0 evidence is incomplete, forged, or internally inconsistent."""


def sha256_file(path):
    path = Path(path)
    if not path.is_file():
        raise B0EvidenceError(f"evidence file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value):
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise B0EvidenceError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _exact(payload, fields, label):
    if not isinstance(payload, Mapping):
        raise B0EvidenceError(f"{label} must be an object")
    expected = set(fields)
    found = set(payload)
    if found != expected:
        raise B0EvidenceError(
            f"{label} fields differ; missing={sorted(expected - found)}, "
            f"extra={sorted(found - expected)}"
        )


def _sha(value, label):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise B0EvidenceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _git_sha(value, label):
    if not isinstance(value, str) or not _GIT_SHA.fullmatch(value):
        raise B0EvidenceError(f"{label} must be a 40-character lowercase git SHA")
    return value


def _count(value, label, *, positive=False):
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise B0EvidenceError(f"{label} must be a {qualifier} integer")
    return value


def _path(value, base_dir, label):
    if not isinstance(value, str) or not value.strip():
        raise B0EvidenceError(f"{label} path must be non-empty text")
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = Path(base_dir) / result
    return result.resolve()


def _load_json(path, label):
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise B0EvidenceError(f"failed to load {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise B0EvidenceError(f"{label} must contain one JSON object")
    return payload


def _verified_path(path_value, digest_value, base_dir, label):
    path = _path(path_value, base_dir, label)
    expected = _sha(digest_value, f"{label}.sha256")
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected):
        raise B0EvidenceError(
            f"{label} hash mismatch: expected {expected}, found {actual}"
        )
    return path


def test_functions(path):
    """Return the exact top-level pytest function declarations in source order."""

    path = Path(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise B0EvidenceError(f"cannot parse test source {path}: {exc}") from exc
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]


def junit_cases(path):
    """Parse actual JUnit leaves, including parametrized testcase names."""

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise B0EvidenceError(f"cannot parse JUnit report {path}: {exc}") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {name: 0 for name in ("collected", "failed", "errors", "skipped")}
    cases = []
    for suite in suites:
        counts["collected"] += int(suite.attrib.get("tests", 0))
        counts["failed"] += int(suite.attrib.get("failures", 0))
        counts["errors"] += int(suite.attrib.get("errors", 0))
        counts["skipped"] += int(suite.attrib.get("skipped", 0))
        for case in suite.findall("testcase"):
            classname = case.attrib.get("classname", "")
            name = case.attrib.get("name", "")
            if not classname or not name:
                raise B0EvidenceError("JUnit testcase lacks classname or name")
            cases.append({"classname": classname, "name": name})
    counts["passed"] = counts["collected"] - sum(
        counts[name] for name in ("failed", "errors", "skipped")
    )
    if counts["collected"] != len(cases):
        raise B0EvidenceError("JUnit testcase leaves differ from declared test count")
    return counts, cases


def validate_manifest(manifest, *, repository_root=None):
    _exact(
        manifest,
        {
            "schema_version",
            "suite_order",
            "suites",
            "runner_sources",
            "audit_checks",
        },
        "B0 manifest",
    )
    if manifest["schema_version"] != B0_MANIFEST_SCHEMA:
        raise B0EvidenceError("B0 manifest schema is unsupported")
    suites = manifest["suites"]
    order = manifest["suite_order"]
    if not isinstance(suites, list) or not suites:
        raise B0EvidenceError("B0 manifest requires at least one suite")
    if order != [suite.get("name") for suite in suites]:
        raise B0EvidenceError("B0 manifest suite order is not exact")
    seen_files = set()
    declared_cases = {}
    for suite_index, suite in enumerate(suites):
        label = f"B0 manifest suite {suite_index}"
        _exact(
            suite,
            {"name", "runner", "canonical_argv", "test_files"},
            label,
        )
        if suite["runner"] != "isolated_torch":
            raise B0EvidenceError(f"{label} uses an unsupported runner")
        if not isinstance(suite["canonical_argv"], list) or not all(
            isinstance(item, str) and item for item in suite["canonical_argv"]
        ):
            raise B0EvidenceError(f"{label} canonical argv is invalid")
        files = suite["test_files"]
        if not isinstance(files, list) or not files:
            raise B0EvidenceError(f"{label} has no classified tests")
        suite_cases = {}
        for source in files:
            _exact(source, {"path", "sha256", "test_functions"}, "B0 test source")
            relative = source["path"]
            if (
                not isinstance(relative, str)
                or not relative.startswith("tests/test_")
                or not relative.endswith(".py")
                or relative in seen_files
            ):
                raise B0EvidenceError(f"invalid or duplicate B0 test source: {relative!r}")
            seen_files.add(relative)
            functions = source["test_functions"]
            if not isinstance(functions, list) or not functions or len(functions) != len(set(functions)):
                raise B0EvidenceError(f"B0 test source {relative} has an invalid testcase list")
            if not all(isinstance(name, str) and name.startswith("test_") for name in functions):
                raise B0EvidenceError(f"B0 test source {relative} has invalid testcase names")
            _sha(source["sha256"], f"B0 test source {relative}")
            module = relative[:-3].replace("/", ".")
            suite_cases[module] = set(functions)
            if repository_root is not None:
                absolute = Path(repository_root) / relative
                if sha256_file(absolute) != source["sha256"]:
                    raise B0EvidenceError(f"B0 test source hash differs: {relative}")
                if test_functions(absolute) != functions:
                    raise B0EvidenceError(f"B0 testcase manifest differs: {relative}")
        declared_cases[suite["name"]] = suite_cases

    runners = manifest["runner_sources"]
    if not isinstance(runners, list) or not runners:
        raise B0EvidenceError("B0 manifest requires runner source hashes")
    for source in runners:
        _exact(source, {"path", "sha256"}, "B0 runner source")
        _sha(source["sha256"], f"B0 runner source {source['path']}")
        if repository_root is not None:
            if sha256_file(Path(repository_root) / source["path"]) != source["sha256"]:
                raise B0EvidenceError(f"B0 runner source hash differs: {source['path']}")
    if manifest["audit_checks"] != [
        "python_syntax",
        "git_diff_check",
        "repository_clean_after",
    ]:
        raise B0EvidenceError("B0 audit check manifest is incomplete or reordered")

    if repository_root is not None:
        discovered = {
            path.relative_to(repository_root).as_posix()
            for path in (Path(repository_root) / "tests").glob("test_*.py")
        }
        if discovered != seen_files:
            raise B0EvidenceError(
                "B0 manifest leaves tests unclassified; "
                f"missing={sorted(discovered - seen_files)}, extra={sorted(seen_files - discovered)}"
            )
    return declared_cases


def validate_b0_evidence(
    reference,
    *,
    base_dir,
    expected_commit=None,
    trust_root,
    repository_root=None,
):
    """Validate the complete signed B0 chain and return its unsigned root body."""

    _exact(reference, {"path", "sha256"}, "B0 evidence reference")
    root_path = _verified_path(reference["path"], reference["sha256"], base_dir, "B0 evidence")
    signed_root = _load_json(root_path, "B0 evidence")
    try:
        root = verify_payload(signed_root, trust_root=trust_root, role=B0_ATTESTATION_ROLE)
    except AttestationError as exc:
        raise B0EvidenceError(str(exc)) from exc
    _exact(
        root,
        {
            "schema_version",
            "status",
            "commit_sha",
            "test_count",
            "blocking_findings",
            "protocol_violations",
            "manifest_path",
            "manifest_sha256",
            "test_report_path",
            "test_report_sha256",
            "audit_report_path",
            "audit_report_sha256",
        },
        "B0 artifact",
    )
    if root["schema_version"] != B0_SCHEMA:
        raise B0EvidenceError("B0 artifact schema is unsupported")
    commit = _git_sha(root["commit_sha"], "B0.commit_sha")
    if expected_commit is not None and commit != expected_commit:
        raise B0EvidenceError("B0 artifact does not bind the expected commit")
    test_count = _count(root["test_count"], "B0.test_count", positive=True)
    blockers = _count(root["blocking_findings"], "B0.blocking_findings")
    violations = _count(root["protocol_violations"], "B0.protocol_violations")

    manifest_path = _verified_path(
        root["manifest_path"], root["manifest_sha256"], root_path.parent, "B0 manifest"
    )
    manifest = _load_json(manifest_path, "B0 manifest")
    declared_cases = validate_manifest(manifest, repository_root=repository_root)

    report_path = _verified_path(
        root["test_report_path"],
        root["test_report_sha256"],
        root_path.parent,
        "B0 test report",
    )
    report = _load_json(report_path, "B0 test report")
    _exact(
        report,
        {
            "schema_version",
            "status",
            "commit_sha",
            "manifest_sha256",
            "collected",
            "passed",
            "failed",
            "errors",
            "skipped",
            "suites",
        },
        "B0 test report",
    )
    if report["schema_version"] != B0_TEST_REPORT_SCHEMA:
        raise B0EvidenceError("B0 test report schema is unsupported")
    if report["commit_sha"] != commit or report["manifest_sha256"] != root["manifest_sha256"]:
        raise B0EvidenceError("B0 test report identity differs")
    if [suite.get("name") for suite in report["suites"]] != manifest["suite_order"]:
        raise B0EvidenceError("B0 test suite execution differs from the locked order")
    totals = {name: 0 for name in ("collected", "passed", "failed", "errors", "skipped")}
    for index, suite in enumerate(report["suites"]):
        label = f"B0 test suite {index}"
        _exact(
            suite,
            {
                "name",
                "status",
                "canonical_argv",
                "python_executable",
                "collected",
                "passed",
                "failed",
                "errors",
                "skipped",
                "log_path",
                "log_sha256",
                "junit_path",
                "junit_sha256",
                "testcase_manifest_sha256",
            },
            label,
        )
        manifest_suite = manifest["suites"][index]
        if suite["name"] != manifest_suite["name"] or suite["canonical_argv"] != manifest_suite["canonical_argv"]:
            raise B0EvidenceError(f"{label} command differs from the locked manifest")
        log_path = _verified_path(suite["log_path"], suite["log_sha256"], root_path.parent, f"{label} log")
        del log_path
        junit_path = _verified_path(
            suite["junit_path"], suite["junit_sha256"], root_path.parent, f"{label} JUnit"
        )
        counts, cases = junit_cases(junit_path)
        for name in totals:
            declared = _count(suite[name], f"{label}.{name}")
            if declared != counts[name]:
                raise B0EvidenceError(f"{label} counts differ from parsed JUnit")
            totals[name] += declared
        if canonical_json_sha256(cases) != suite["testcase_manifest_sha256"]:
            raise B0EvidenceError(f"{label} testcase manifest digest differs")
        seen_functions = {module: set() for module in declared_cases[suite["name"]]}
        for case in cases:
            module = case["classname"]
            function = case["name"].split("[", 1)[0]
            if module not in seen_functions or function not in declared_cases[suite["name"]][module]:
                raise B0EvidenceError(f"{label} contains an unclassified testcase: {module}::{function}")
            seen_functions[module].add(function)
        missing = {
            f"{module}::{name}"
            for module, names in declared_cases[suite["name"]].items()
            for name in names - seen_functions[module]
        }
        if missing:
            raise B0EvidenceError(f"{label} did not execute declared testcases: {sorted(missing)}")
        if (
            suite["status"] != "PASS"
            or counts["failed"]
            or counts["errors"]
            or counts["skipped"]
        ):
            raise B0EvidenceError(f"{label} has not reached PASS")

    report_totals = {name: _count(report[name], f"B0 test report.{name}") for name in totals}
    if report_totals != totals:
        raise B0EvidenceError("B0 test report totals differ from suite totals")

    audit_path = _verified_path(
        root["audit_report_path"],
        root["audit_report_sha256"],
        root_path.parent,
        "B0 audit report",
    )
    audit = _load_json(audit_path, "B0 audit report")
    _exact(
        audit,
        {
            "schema_version",
            "status",
            "commit_sha",
            "manifest_sha256",
            "blocking_findings",
            "protocol_violations",
            "checks",
        },
        "B0 audit report",
    )
    if audit["schema_version"] != B0_AUDIT_REPORT_SCHEMA:
        raise B0EvidenceError("B0 audit report schema is unsupported")
    if audit["commit_sha"] != commit or audit["manifest_sha256"] != root["manifest_sha256"]:
        raise B0EvidenceError("B0 audit report identity differs")
    if [check.get("name") for check in audit["checks"]] != manifest["audit_checks"]:
        raise B0EvidenceError("B0 audit checks differ from the locked manifest")
    for index, check in enumerate(audit["checks"]):
        label = f"B0 audit check {index}"
        _exact(check, {"name", "status", "canonical_argv", "log_path", "log_sha256"}, label)
        _verified_path(check["log_path"], check["log_sha256"], root_path.parent, f"{label} log")
        if check["status"] != "PASS":
            raise B0EvidenceError(f"{label} has not reached PASS")

    passed = (
        root["status"] == "PASS"
        and report["status"] == "PASS"
        and audit["status"] == "PASS"
        and totals["collected"] == totals["passed"] == test_count
        and totals["failed"] == totals["errors"] == totals["skipped"] == 0
        and blockers == violations == 0
        and _count(audit["blocking_findings"], "B0 audit blockers") == 0
        and _count(audit["protocol_violations"], "B0 audit violations") == 0
    )
    if not passed:
        raise B0EvidenceError("B0 evidence chain has not reached PASS")
    return {
        **root,
        "artifact_path": str(root_path),
        "artifact_sha256": sha256_file(root_path),
        "manifest": manifest,
    }


__all__ = [
    "B0_ATTESTATION_ROLE",
    "B0_AUDIT_REPORT_SCHEMA",
    "B0_MANIFEST_SCHEMA",
    "B0_SCHEMA",
    "B0_TEST_REPORT_SCHEMA",
    "B0EvidenceError",
    "canonical_json_sha256",
    "junit_cases",
    "sha256_file",
    "test_functions",
    "validate_b0_evidence",
    "validate_manifest",
]
