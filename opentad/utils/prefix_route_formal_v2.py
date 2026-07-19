"""Clean-process and repository import-closure checks for Prefix Route V2."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


FORMAL_PROCESS_SCHEMA = "prefix-route-formal-process-attestation-v2"


class PrefixRouteFormalError(ValueError):
    pass


_FORMAL_PROCESS_CAPABILITY_TOKEN = object()


class VerifiedFormalProcess:
    __slots__ = ("attestation", "attestation_sha256", "_token")

    def __init__(
        self,
        *,
        attestation,
        attestation_sha256,
        token,
    ):
        self.attestation = json.loads(canonical_json_bytes(attestation))
        self.attestation_sha256 = attestation_sha256
        self._token = token


def canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _source_path(module_file):
    path = Path(module_file)
    if path.suffix in {".pyc", ".pyo"}:
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except ValueError as exc:
            raise PrefixRouteFormalError(
                f"cannot resolve imported bytecode origin: {module_file}"
            ) from exc
    return path.resolve()


def repository_module_origins(repo_root):
    repo_root = Path(repo_root).resolve()
    origins = {}
    for module_name, module in sorted(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        source_path = _source_path(module_file)
        try:
            relative = source_path.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        if source_path.suffix != ".py":
            continue
        if relative in origins and origins[relative] != module_name:
            raise PrefixRouteFormalError(
                f"repository source loaded under multiple module names: {relative}"
            )
        origins[relative] = module_name
    return origins


def required_parent_initializers(repo_root, source_paths):
    repo_root = Path(repo_root).resolve()
    required = set()
    for relative in source_paths:
        path = repo_root.joinpath(*Path(relative).parts)
        if path.suffix != ".py":
            continue
        parent = path.parent
        while parent != repo_root:
            initializer = parent / "__init__.py"
            if initializer.is_file():
                required.add(initializer.relative_to(repo_root).as_posix())
            parent = parent.parent
    return required


def verify_runtime_import_closure(*, repo_root, manifest_record):
    """Reject every loaded repository module absent from the reviewed manifest."""

    repo_root = Path(repo_root).resolve()
    entries = manifest_record["manifest"]["entries"]
    parent_initializers = required_parent_initializers(repo_root, entries)
    missing_parents = sorted(parent_initializers - set(entries))
    if missing_parents:
        raise PrefixRouteFormalError(
            f"source manifest omits package initializers: {missing_parents}"
        )
    origins = repository_module_origins(repo_root)
    missing = sorted(set(origins) - set(entries))
    if missing:
        raise PrefixRouteFormalError(
            f"runtime imported repository modules outside manifest: {missing}"
        )
    rows = []
    for relative, module_name in sorted(origins.items()):
        path = repo_root.joinpath(*Path(relative).parts)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise PrefixRouteFormalError(
                f"cannot read imported module origin: {relative}"
            ) from exc
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entries[relative]:
            raise PrefixRouteFormalError(
                f"imported module bytes differ from manifest: {relative}"
            )
        rows.append(
            {
                "module": module_name,
                "path": relative,
                "sha256": digest,
            }
        )
    return rows


def _git(repo_root, *args):
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrefixRouteFormalError(
            f"git command failed: git {' '.join(args)}"
        ) from exc


def build_formal_process_attestation(
    *,
    repo_root,
    manifest_record,
    protocol_record,
    review_record,
    argv,
):
    """Bind the child executable, exact tree, argv, and loaded source origins."""

    repo_root = Path(repo_root).resolve()
    if Path.cwd().resolve() != repo_root:
        raise PrefixRouteFormalError(
            "formal process working directory differs from repository root"
        )
    commit = _git(repo_root, "rev-parse", "HEAD")
    tree = _git(repo_root, "rev-parse", "HEAD^{tree}")
    review = review_record["attestation"]
    if (
        commit != review["protocol_commit"]
        or tree != review["protocol_tree_sha1"]
    ):
        raise PrefixRouteFormalError(
            "formal process checkout differs from signed review"
        )
    modules = verify_runtime_import_closure(
        repo_root=repo_root,
        manifest_record=manifest_record,
    )
    relevant_environment = {}
    for name in (
        "PATH",
        "PYTHONPATH",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        value = os.environ.get(name)
        relevant_environment[name] = (
            None
            if value is None
            else hashlib.sha256(value.encode("utf-8")).hexdigest()
        )
    attestation = {
        "schema_version": FORMAL_PROCESS_SCHEMA,
        "protocol_id": protocol_record["protocol"]["protocol_id"],
        "protocol_sha256": protocol_record["sha256"],
        "source_manifest_sha256": manifest_record["sha256"],
        "review_attestation_sha256": review_record["attestation_sha256"],
        "repository_root": str(repo_root),
        "repository_commit": commit,
        "repository_tree_sha1": tree,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "working_directory": str(Path.cwd().resolve()),
        "argv": [str(value) for value in argv],
        "environment_variable_sha256": relevant_environment,
        "environment_identity_sha256": hashlib.sha256(
            canonical_json_bytes(relevant_environment)
        ).hexdigest(),
        "repository_modules": modules,
    }
    attestation["derived_sha256"] = hashlib.sha256(
        canonical_json_bytes(attestation)
    ).hexdigest()
    return attestation


def _mint_formal_process_capability(attestation):
    if (
        not isinstance(attestation, dict)
        or attestation.get("schema_version") != FORMAL_PROCESS_SCHEMA
    ):
        raise PrefixRouteFormalError("formal process attestation differs")
    unsigned = dict(attestation)
    supplied = unsigned.pop("derived_sha256", None)
    derived = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if supplied != derived:
        raise PrefixRouteFormalError(
            "formal process attestation commitment differs"
        )
    return VerifiedFormalProcess(
        attestation=attestation,
        attestation_sha256=hashlib.sha256(
            canonical_json_bytes(attestation)
        ).hexdigest(),
        token=_FORMAL_PROCESS_CAPABILITY_TOKEN,
    )


def require_formal_process_capability(capability):
    if (
        type(capability) is not VerifiedFormalProcess
        or capability._token is not _FORMAL_PROCESS_CAPABILITY_TOKEN
    ):
        raise PrefixRouteFormalError(
            "formal R6 requires a clean-process capability"
        )
    attestation = capability.attestation
    if hashlib.sha256(canonical_json_bytes(attestation)).hexdigest() != (
        capability.attestation_sha256
    ):
        raise PrefixRouteFormalError(
            "formal process capability changed after attestation"
        )
    unsigned = dict(attestation)
    supplied = unsigned.pop("derived_sha256", None)
    if (
        attestation.get("schema_version") != FORMAL_PROCESS_SCHEMA
        or supplied
        != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    ):
        raise PrefixRouteFormalError("formal process capability is invalid")
    return attestation


__all__ = [
    "FORMAL_PROCESS_SCHEMA",
    "PrefixRouteFormalError",
    "build_formal_process_attestation",
    "canonical_json_bytes",
    "repository_module_origins",
    "required_parent_initializers",
    "require_formal_process_capability",
    "verify_runtime_import_closure",
]
