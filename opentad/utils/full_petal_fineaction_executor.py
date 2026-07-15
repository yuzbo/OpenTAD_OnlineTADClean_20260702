"""Locked FineAction evidence executor for Full PETAL qualification.

The execution key is loaded only after one fixed pytest subprocess has produced a
clean JUnit report and its captured log. The signed root therefore cannot be
created by pairing independently authored source, JUnit, and log files.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .evidence_bundle import (
    EvidenceBundleError,
    publish_exclusive_file,
    read_stable_file_bytes,
)
from .full_petal_attestation import (
    ATTESTATION_ALGORITHM,
    ATTESTATION_FIELD,
    ATTESTATION_SCHEMA,
    AttestationError,
    _message,
)
from .full_petal_b0 import (
    B0EvidenceError,
    canonical_json_sha256,
    junit_cases,
    test_functions,
)
from .full_petal_role_signing import (
    _load_ed25519_key,
    _public_digest,
    _validated_key_id,
)


FINEACTION_EXECUTOR_SCHEMA = "full-petal-fineaction-locked-executor-v1"
FINEACTION_PREPROCESSING_SCHEMA = "full-petal-fineaction-preprocessing-run-v2"
FINEACTION_LOADER_SCHEMA = "full-petal-fineaction-loader-run-v2"
FINEACTION_PREPROCESSING_ROLE = "fineaction-causal-preprocessing-run"
FINEACTION_LOADER_ROLE = "fineaction-loader-smoke-run"
LOCKED_CWD = "."
LOCKED_SOURCE_PATH = "source.py"
LOCKED_JUNIT_PATH = "junit.xml"
LOCKED_LOG_PATH = "execution.log"
LOCKED_RECORD_PATH = "execution.json"
LOCKED_ENVIRONMENT_KEYS = (
    "PYTHONHASHSEED",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONIOENCODING",
    "PYTHONPATH",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
)


class FineActionExecutionError(RuntimeError):
    """Raised when a locked FineAction evidence run cannot be attested."""


@dataclass(frozen=True)
class _ExecutionSpec:
    kind: str
    schema: str
    role: str
    pass_marker: str
    metadata_fields: frozenset[str]


_PREPROCESSING = _ExecutionSpec(
    kind="preprocessing",
    schema=FINEACTION_PREPROCESSING_SCHEMA,
    role=FINEACTION_PREPROCESSING_ROLE,
    pass_marker="FINEACTION_CAUSAL_PREPROCESS_PASS",
    metadata_fields=frozenset(
        {
            "annotation_sha256",
            "media_inventory_sha256",
            "future_frames_allowed",
            "timestamp_convention",
            "frame_stride",
        }
    ),
)
_LOADER = _ExecutionSpec(
    kind="loader",
    schema=FINEACTION_LOADER_SCHEMA,
    role=FINEACTION_LOADER_ROLE,
    pass_marker="FINEACTION_LOADER_SMOKE_PASS",
    metadata_fields=frozenset(
        {
            "annotation_sha256",
            "media_inventory_sha256",
            "preprocessing_sha256",
        }
    ),
)
_SPECS = (_PREPROCESSING, _LOADER)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reference(path, root, payload):
    return {
        "path": Path(path).relative_to(root).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _json_bytes(payload):
    return (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def fineaction_junit_prefix(kind, source_sha256):
    if kind not in {spec.kind for spec in _SPECS} or not _is_sha256(source_sha256):
        raise FineActionExecutionError("FineAction JUnit prefix inputs are invalid")
    return f"full_petal_fineaction_{kind}_{source_sha256}"


def _locked_command(python_executable, junit_prefix):
    return [
        str(python_executable),
        "-m",
        "pytest",
        LOCKED_SOURCE_PATH,
        "-q",
        "-p",
        "no:cacheprovider",
        "--junitxml",
        LOCKED_JUNIT_PATH,
        "--junit-prefix",
        junit_prefix,
    ]


def locked_command_for_kind(kind, python_executable, source_sha256):
    return _locked_command(
        str(python_executable), fineaction_junit_prefix(kind, source_sha256)
    )


def fineaction_invocation_sha256(command, environment_overrides, source):
    return canonical_json_sha256(
        {
            "executor_schema": FINEACTION_EXECUTOR_SCHEMA,
            "command": list(command),
            "cwd": LOCKED_CWD,
            "environment_overrides": dict(environment_overrides),
            "source": dict(source),
        }
    )


def fineaction_execution_sha256(invocation_sha256, exit_code, junit, log):
    return canonical_json_sha256(
        {
            "invocation_sha256": invocation_sha256,
            "exit_code": exit_code,
            "junit": dict(junit),
            "log": dict(log),
        }
    )


def _locked_environment_overrides():
    return {
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": str(_REPOSITORY_ROOT),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }


def _validate_metadata(spec, metadata):
    if set(metadata) != set(spec.metadata_fields):
        raise FineActionExecutionError("FineAction execution metadata fields differ")
    for field in spec.metadata_fields:
        if field.endswith("_sha256") and not _is_sha256(metadata[field]):
            raise FineActionExecutionError(f"FineAction {field} is not a SHA-256")
    if spec is _PREPROCESSING:
        if metadata["future_frames_allowed"] is not False:
            raise FineActionExecutionError("FineAction preprocessing must forbid future frames")
        if not isinstance(metadata["timestamp_convention"], str) or not metadata[
            "timestamp_convention"
        ].strip():
            raise FineActionExecutionError("FineAction timestamp convention is empty")
        stride = metadata["frame_stride"]
        if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
            raise FineActionExecutionError("FineAction frame stride must be positive")


def _validate_junit(junit_bytes, *, junit_prefix, declared_tests):
    try:
        counts, cases = junit_cases(junit_bytes=junit_bytes)
    except B0EvidenceError as exc:
        raise FineActionExecutionError(f"FineAction JUnit is invalid: {exc}") from exc
    if (
        counts["collected"] <= 0
        or counts["passed"] != counts["collected"]
        or counts["failed"]
        or counts["errors"]
        or counts["skipped"]
    ):
        raise FineActionExecutionError("FineAction JUnit did not reach a clean PASS")
    declared = set(declared_tests)
    for case in cases:
        if not case["classname"].startswith(f"{junit_prefix}."):
            raise FineActionExecutionError("FineAction JUnit is not bound to its source hash")
        function_name = case["name"].split("[", 1)[0]
        if function_name not in declared:
            raise FineActionExecutionError("FineAction JUnit testcase is absent from its source")
    return counts, cases


def _log_bytes(
    *, command, environment_overrides, invocation_sha256, result, pass_marker=None
):
    content = (
        f"FULL_PETAL_FINEACTION_EXECUTOR={FINEACTION_EXECUTOR_SCHEMA}\n"
        f"INVOCATION_SHA256={invocation_sha256}\n"
        "ACTUAL_ARGV="
        + json.dumps(command, separators=(",", ":"))
        + "\nCWD="
        + LOCKED_CWD
        + "\nENVIRONMENT_OVERRIDES="
        + json.dumps(environment_overrides, separators=(",", ":"), sort_keys=True)
        + f"\nRETURN_CODE={result.returncode}\n\nSTDOUT\n"
        + result.stdout
        + "\nSTDERR\n"
        + result.stderr
    )
    if pass_marker is not None:
        content += f"\n{pass_marker}\n"
    return content.encode("utf-8")


def _run_locked_fineaction_evidence(
    spec,
    *,
    source_path,
    output_dir,
    private_key_path,
    key_id,
    python_executable,
    timeout_seconds,
    metadata,
):
    if spec not in _SPECS:
        raise FineActionExecutionError("FineAction execution specification is not locked")
    _validate_metadata(spec, metadata)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise FineActionExecutionError("FineAction execution timeout must be positive")
    try:
        source_path, source_bytes = read_stable_file_bytes(
            source_path, f"FineAction {spec.kind} pytest source"
        )
    except EvidenceBundleError as exc:
        raise FineActionExecutionError(str(exc)) from exc
    if source_path.suffix != ".py" or not source_bytes.strip():
        raise FineActionExecutionError("FineAction execution source must be non-empty Python")
    try:
        declared_tests = test_functions(source_path, source_bytes=source_bytes)
    except B0EvidenceError as exc:
        raise FineActionExecutionError(str(exc)) from exc
    if not declared_tests:
        raise FineActionExecutionError("FineAction execution source declares no tests")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir()
    except FileExistsError as exc:
        raise FineActionExecutionError(
            f"FineAction evidence directory already exists: {output_dir}"
        ) from exc
    staged_source = output_dir / LOCKED_SOURCE_PATH
    junit_path = output_dir / LOCKED_JUNIT_PATH
    log_path = output_dir / LOCKED_LOG_PATH
    record_path = output_dir / LOCKED_RECORD_PATH
    try:
        publish_exclusive_file(staged_source, source_bytes)
    except EvidenceBundleError as exc:
        raise FineActionExecutionError(str(exc)) from exc

    python_path = Path(python_executable or sys.executable).expanduser().resolve()
    if not python_path.is_file():
        raise FineActionExecutionError(f"FineAction Python executable is missing: {python_path}")
    source_reference = _reference(staged_source, output_dir, source_bytes)
    junit_prefix = fineaction_junit_prefix(spec.kind, source_reference["sha256"])
    command = _locked_command(str(python_path), junit_prefix)
    environment_overrides = _locked_environment_overrides()
    invocation_sha256 = fineaction_invocation_sha256(
        command, environment_overrides, source_reference
    )
    child_environment = os.environ.copy()
    child_environment.update(environment_overrides)
    try:
        result = subprocess.run(
            command,
            cwd=output_dir,
            env=child_environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        raise FineActionExecutionError(
            f"FineAction {spec.kind} execution timed out"
        ) from exc

    if result.returncode != 0 or not junit_path.is_file():
        try:
            publish_exclusive_file(
                log_path,
                _log_bytes(
                    command=command,
                    environment_overrides=environment_overrides,
                    invocation_sha256=invocation_sha256,
                    result=result,
                ),
            )
        except EvidenceBundleError as exc:
            raise FineActionExecutionError(str(exc)) from exc
        raise FineActionExecutionError(
            f"FineAction {spec.kind} subprocess failed with exit code {result.returncode}"
        )
    try:
        _, junit_bytes = read_stable_file_bytes(
            junit_path, f"FineAction {spec.kind} JUnit"
        )
    except EvidenceBundleError as exc:
        raise FineActionExecutionError(str(exc)) from exc
    _validate_junit(
        junit_bytes, junit_prefix=junit_prefix, declared_tests=declared_tests
    )
    log_bytes = _log_bytes(
        command=command,
        environment_overrides=environment_overrides,
        invocation_sha256=invocation_sha256,
        result=result,
        pass_marker=spec.pass_marker,
    )
    try:
        publish_exclusive_file(log_path, log_bytes)
    except EvidenceBundleError as exc:
        raise FineActionExecutionError(str(exc)) from exc

    junit_reference = _reference(junit_path, output_dir, junit_bytes)
    log_reference = _reference(log_path, output_dir, log_bytes)
    body = {
        "schema_version": spec.schema,
        "executor_schema": FINEACTION_EXECUTOR_SCHEMA,
        "dataset": "FineAction",
        "status": "PASS",
        "command": command,
        "cwd": LOCKED_CWD,
        "environment_overrides": environment_overrides,
        "exit_code": 0,
        "source": source_reference,
        "junit": junit_reference,
        "log": log_reference,
        "invocation_sha256": invocation_sha256,
        "execution_sha256": fineaction_execution_sha256(
            invocation_sha256, 0, junit_reference, log_reference
        ),
        **metadata,
    }
    try:
        validated_key_id = _validated_key_id(key_id)
        key = _load_ed25519_key(private_key_path)
    except AttestationError as exc:
        raise FineActionExecutionError(str(exc)) from exc
    body[ATTESTATION_FIELD] = {
        "schema_version": ATTESTATION_SCHEMA,
        "algorithm": ATTESTATION_ALGORITHM,
        "role": spec.role,
        "key_id": validated_key_id,
        "public_key_sha256": _public_digest(key),
        "signature": base64.b64encode(
            key.sign(_message(body, spec.role))
        ).decode("ascii"),
    }
    try:
        publish_exclusive_file(record_path, _json_bytes(body))
    except EvidenceBundleError as exc:
        raise FineActionExecutionError(str(exc)) from exc
    return record_path


def run_fineaction_preprocessing_evidence(
    source_path,
    output_dir,
    *,
    annotation_sha256,
    media_inventory_sha256,
    timestamp_convention,
    frame_stride,
    private_key_path,
    key_id,
    python_executable=None,
    timeout_seconds=300,
):
    return _run_locked_fineaction_evidence(
        _PREPROCESSING,
        source_path=source_path,
        output_dir=output_dir,
        private_key_path=private_key_path,
        key_id=key_id,
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
        metadata={
            "annotation_sha256": annotation_sha256,
            "media_inventory_sha256": media_inventory_sha256,
            "future_frames_allowed": False,
            "timestamp_convention": timestamp_convention,
            "frame_stride": frame_stride,
        },
    )


def run_fineaction_loader_evidence(
    source_path,
    output_dir,
    *,
    annotation_sha256,
    media_inventory_sha256,
    preprocessing_sha256,
    private_key_path,
    key_id,
    python_executable=None,
    timeout_seconds=300,
):
    return _run_locked_fineaction_evidence(
        _LOADER,
        source_path=source_path,
        output_dir=output_dir,
        private_key_path=private_key_path,
        key_id=key_id,
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
        metadata={
            "annotation_sha256": annotation_sha256,
            "media_inventory_sha256": media_inventory_sha256,
            "preprocessing_sha256": preprocessing_sha256,
        },
    )


__all__ = [
    "FINEACTION_EXECUTOR_SCHEMA",
    "FINEACTION_LOADER_SCHEMA",
    "FINEACTION_PREPROCESSING_SCHEMA",
    "FineActionExecutionError",
    "run_fineaction_loader_evidence",
    "run_fineaction_preprocessing_evidence",
]
