"""Canonical Full PETAL B0 manifest and evidence validation."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

from .full_petal_attestation import AttestationError, verify_payload
from .evidence_bundle import (
    EvidenceBundleError,
    read_stable_file_bytes,
    read_verified_path_bytes,
    resolve_bundle_path,
    strict_json_from_bytes,
)


B0_SCHEMA = "full-petal-b0-v3"
B0_TEST_REPORT_SCHEMA = "full-petal-b0-test-report-v2"
B0_AUDIT_REPORT_SCHEMA = "full-petal-b0-audit-report-v2"
B0_MANIFEST_SCHEMA = "full-petal-b0-manifest-v1"
B0_ATTESTATION_ROLE = "b0-runner"
B0_POSIX_LEAF_SCHEMA = "full-petal-b0-posix-leaf-v1"
B0_POSIX_ATTESTATION_ROLE = "b0-posix-runner"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


class B0EvidenceError(ValueError):
    """Raised when B0 evidence is incomplete, forged, or internally inconsistent."""


def sha256_file(path):
    try:
        _, payload = read_stable_file_bytes(path, f"B0 evidence file {path}")
    except EvidenceBundleError as exc:
        raise B0EvidenceError(str(exc)) from exc
    return hashlib.sha256(payload).hexdigest()


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


def _path(value, base_dir, label, *, require_relative=True):
    if not isinstance(value, str) or not value.strip():
        raise B0EvidenceError(f"{label} path must be non-empty text")
    result = Path(value).expanduser()
    if result.is_absolute():
        if require_relative:
            raise B0EvidenceError(f"{label} must use a relative bundle path")
        return result.resolve()
    try:
        return resolve_bundle_path(value, base_dir, label)
    except EvidenceBundleError as exc:
        raise B0EvidenceError(str(exc)) from exc


def _load_json_bytes(payload, label):
    try:
        return strict_json_from_bytes(payload, label, require_object=True)
    except EvidenceBundleError as exc:
        raise B0EvidenceError(str(exc)) from exc


def _verified_bytes(
    path_value, digest_value, base_dir, label, *, require_relative=True
):
    path = _path(
        path_value, base_dir, label, require_relative=require_relative
    )
    expected = _sha(digest_value, f"{label}.sha256")
    try:
        return read_verified_path_bytes(path, expected, label)
    except EvidenceBundleError as exc:
        raise B0EvidenceError(str(exc)) from exc


def test_functions(path, *, source_bytes=None):
    """Return the exact top-level pytest function declarations in source order."""

    path = Path(path)
    try:
        if source_bytes is None:
            _, source_bytes = read_stable_file_bytes(path, f"test source {path}")
        tree = ast.parse(source_bytes.decode("utf-8"), filename=str(path))
    except (EvidenceBundleError, UnicodeError, SyntaxError) as exc:
        raise B0EvidenceError(f"cannot parse test source {path}: {exc}") from exc
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]


def junit_cases(path=None, *, junit_bytes=None):
    """Parse actual JUnit leaves, including parametrized testcase names."""

    try:
        if junit_bytes is None:
            _, junit_bytes = read_stable_file_bytes(path, f"JUnit report {path}")
        root = ET.fromstring(junit_bytes)
    except (EvidenceBundleError, ET.ParseError) as exc:
        raise B0EvidenceError(f"cannot parse JUnit report {path or '<bytes>'}: {exc}") from exc
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
                try:
                    _, source_bytes = read_verified_path_bytes(
                        absolute,
                        source["sha256"],
                        f"B0 test source {relative}",
                    )
                except EvidenceBundleError as exc:
                    raise B0EvidenceError(str(exc)) from exc
                if test_functions(absolute, source_bytes=source_bytes) != functions:
                    raise B0EvidenceError(f"B0 testcase manifest differs: {relative}")
        declared_cases[suite["name"]] = suite_cases

    runners = manifest["runner_sources"]
    if not isinstance(runners, list) or not runners:
        raise B0EvidenceError("B0 manifest requires runner source hashes")
    for source in runners:
        _exact(source, {"path", "sha256"}, "B0 runner source")
        _sha(source["sha256"], f"B0 runner source {source['path']}")
        if repository_root is not None:
            try:
                read_verified_path_bytes(
                    Path(repository_root) / source["path"],
                    source["sha256"],
                    f"B0 runner source {source['path']}",
                )
            except EvidenceBundleError as exc:
                raise B0EvidenceError(str(exc)) from exc
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
    root_path, root_bytes = _verified_bytes(
        reference["path"],
        reference["sha256"],
        base_dir,
        "B0 evidence",
        require_relative=False,
    )
    signed_root = _load_json_bytes(root_bytes, "B0 evidence")
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
            "posix_leaf_path",
            "posix_leaf_sha256",
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

    manifest_path, manifest_bytes = _verified_bytes(
        root["manifest_path"], root["manifest_sha256"], root_path.parent, "B0 manifest"
    )
    manifest = _load_json_bytes(manifest_bytes, "B0 manifest")
    declared_cases = validate_manifest(manifest, repository_root=repository_root)

    posix_leaf = validate_posix_b0_leaf(
        {
            "path": root["posix_leaf_path"],
            "sha256": root["posix_leaf_sha256"],
        },
        base_dir=root_path.parent,
        expected_commit=commit,
        expected_manifest_sha256=root["manifest_sha256"],
        manifest=manifest,
        declared_cases=declared_cases,
        trust_root=trust_root,
    )

    report_path, report_bytes = _verified_bytes(
        root["test_report_path"],
        root["test_report_sha256"],
        root_path.parent,
        "B0 test report",
    )
    report = _load_json_bytes(report_bytes, "B0 test report")
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
        _verified_bytes(
            suite["log_path"], suite["log_sha256"], root_path.parent, f"{label} log"
        )
        junit_path, junit_bytes = _verified_bytes(
            suite["junit_path"], suite["junit_sha256"], root_path.parent, f"{label} JUnit"
        )
        counts, cases = junit_cases(junit_path, junit_bytes=junit_bytes)
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

    audit_path, audit_bytes = _verified_bytes(
        root["audit_report_path"],
        root["audit_report_sha256"],
        root_path.parent,
        "B0 audit report",
    )
    audit = _load_json_bytes(audit_bytes, "B0 audit report")
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
        _verified_bytes(check["log_path"], check["log_sha256"], root_path.parent, f"{label} log")
        if check["status"] != "PASS":
            raise B0EvidenceError(f"{label} has not reached PASS")

    passed = (
        root["status"] == "PASS"
        and report["status"] == "PASS"
        and audit["status"] == "PASS"
        and totals["collected"] == totals["passed"] == test_count
        and posix_leaf["collected"] == posix_leaf["passed"] == test_count
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
        "artifact_sha256": hashlib.sha256(root_bytes).hexdigest(),
        "manifest": manifest,
    }


def validate_posix_b0_leaf(
    reference,
    *,
    base_dir,
    expected_commit,
    expected_manifest_sha256,
    manifest,
    declared_cases,
    trust_root,
):
    """Verify a target-Linux B0 leaf and every JUnit/log byte it commits."""

    _exact(reference, {"path", "sha256"}, "POSIX B0 leaf reference")
    leaf_path, leaf_bytes = _verified_bytes(
        reference["path"],
        reference["sha256"],
        base_dir,
        "POSIX B0 leaf",
        require_relative=False,
    )
    signed_leaf = _load_json_bytes(leaf_bytes, "POSIX B0 leaf")
    try:
        leaf = verify_payload(
            signed_leaf,
            trust_root=trust_root,
            role=B0_POSIX_ATTESTATION_ROLE,
        )
    except AttestationError as exc:
        raise B0EvidenceError(str(exc)) from exc
    _exact(
        leaf,
        {
            "schema_version",
            "status",
            "commit_sha",
            "manifest_sha256",
            "platform",
            "repository_clean_before",
            "repository_clean_after",
            "collected",
            "passed",
            "failed",
            "errors",
            "skipped",
            "suites",
        },
        "POSIX B0 leaf",
    )
    if leaf["schema_version"] != B0_POSIX_LEAF_SCHEMA:
        raise B0EvidenceError("POSIX B0 leaf schema is unsupported")
    if leaf["commit_sha"] != expected_commit:
        raise B0EvidenceError("POSIX B0 leaf commit differs")
    if leaf["manifest_sha256"] != expected_manifest_sha256:
        raise B0EvidenceError("POSIX B0 leaf manifest differs")
    platform = leaf["platform"]
    _exact(
        platform,
        {"os_name", "sys_platform", "machine", "python_version", "torch_version"},
        "POSIX B0 platform",
    )
    if platform["os_name"] != "posix" or platform["sys_platform"] != "linux":
        raise B0EvidenceError("POSIX B0 leaf was not produced on Linux")
    if not all(
        isinstance(platform[field], str) and platform[field].strip()
        for field in ("machine", "python_version", "torch_version")
    ):
        raise B0EvidenceError("POSIX B0 platform identity is incomplete")
    if (
        leaf["status"] != "PASS"
        or leaf["repository_clean_before"] is not True
        or leaf["repository_clean_after"] is not True
    ):
        raise B0EvidenceError("POSIX B0 leaf has not reached clean PASS")
    if [suite.get("name") for suite in leaf["suites"]] != manifest["suite_order"]:
        raise B0EvidenceError("POSIX B0 suite order differs from the manifest")

    totals = {name: 0 for name in ("collected", "passed", "failed", "errors", "skipped")}
    for index, suite in enumerate(leaf["suites"]):
        label = f"POSIX B0 suite {index}"
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
        if (
            suite["name"] != manifest_suite["name"]
            or suite["canonical_argv"] != manifest_suite["canonical_argv"]
        ):
            raise B0EvidenceError(f"{label} command differs from the manifest")
        if not isinstance(suite["python_executable"], str) or not suite[
            "python_executable"
        ].strip():
            raise B0EvidenceError(f"{label} Python executable is invalid")
        _verified_bytes(
            suite["log_path"], suite["log_sha256"], leaf_path.parent, f"{label} log"
        )
        junit_path, junit_bytes = _verified_bytes(
            suite["junit_path"],
            suite["junit_sha256"],
            leaf_path.parent,
            f"{label} JUnit",
        )
        counts, cases = junit_cases(junit_path, junit_bytes=junit_bytes)
        for name in totals:
            declared = _count(suite[name], f"{label}.{name}")
            if declared != counts[name]:
                raise B0EvidenceError(f"{label} counts differ from parsed JUnit")
            totals[name] += declared
        if canonical_json_sha256(cases) != suite["testcase_manifest_sha256"]:
            raise B0EvidenceError(f"{label} testcase digest differs")
        seen_functions = {
            module: set() for module in declared_cases[suite["name"]]
        }
        for case in cases:
            module = case["classname"]
            function = case["name"].split("[", 1)[0]
            if (
                module not in seen_functions
                or function not in declared_cases[suite["name"]][module]
            ):
                raise B0EvidenceError(
                    f"{label} contains an unclassified testcase: {module}::{function}"
                )
            seen_functions[module].add(function)
        missing = {
            f"{module}::{name}"
            for module, names in declared_cases[suite["name"]].items()
            for name in names - seen_functions[module]
        }
        if missing:
            raise B0EvidenceError(
                f"{label} did not execute declared testcases: {sorted(missing)}"
            )
        if (
            suite["status"] != "PASS"
            or counts["failed"]
            or counts["errors"]
            or counts["skipped"]
        ):
            raise B0EvidenceError(f"{label} has not reached PASS")
    leaf_totals = {name: _count(leaf[name], f"POSIX B0.{name}") for name in totals}
    if leaf_totals != totals or totals["collected"] <= 0:
        raise B0EvidenceError("POSIX B0 totals differ from suite totals")
    return {
        **leaf,
        "artifact_path": str(leaf_path),
        "artifact_sha256": hashlib.sha256(leaf_bytes).hexdigest(),
    }


__all__ = [
    "B0_ATTESTATION_ROLE",
    "B0_AUDIT_REPORT_SCHEMA",
    "B0_MANIFEST_SCHEMA",
    "B0_POSIX_ATTESTATION_ROLE",
    "B0_POSIX_LEAF_SCHEMA",
    "B0_SCHEMA",
    "B0_TEST_REPORT_SCHEMA",
    "B0EvidenceError",
    "canonical_json_sha256",
    "junit_cases",
    "sha256_file",
    "test_functions",
    "validate_b0_evidence",
    "validate_manifest",
    "validate_posix_b0_leaf",
]
