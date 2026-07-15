"""Source, data, runtime, and scheduler identities for Full PETAL launches."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from collections.abc import Mapping


DATA_IDENTITY_SCHEMA = "full-petal-data-identity-v1"
RUNTIME_IDENTITY_SCHEMA = "full-petal-runtime-identity-v1"
SCIENTIFIC_SOURCE_SCHEMA = "full-petal-scientific-source-v1"
AUTHORIZATION_CONFIG_PATH = "configs/causaltad/thumos_pes_q2_base.py"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class IdentityError(ValueError):
    """Raised when a launch identity cannot be proven exactly."""


@dataclass(frozen=True)
class SlurmAllocation:
    job_id: str
    state: str
    user: str
    nodes: int
    tasks: int
    gpus: int
    command: str
    work_dir: str


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
        raise IdentityError(f"identity is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path):
    path = Path(path)
    if not path.is_file():
        raise IdentityError(f"identity file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value, label):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise IdentityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _file_record(path, label):
    if not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise IdentityError(f"{label} path must be non-empty")
    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _load_json(path, label):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IdentityError(f"failed to load {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IdentityError(f"{label} must contain one JSON object")
    return payload


def _feature_inventory(cache_manifest_path, cache_root):
    manifest = _load_json(cache_manifest_path, "feature cache manifest")
    if manifest.get("schema") != "ontad_feature_cache_v1":
        raise IdentityError("feature cache manifest schema is unsupported")
    videos = manifest.get("videos")
    if not isinstance(videos, dict) or not videos:
        raise IdentityError("feature cache manifest has no video inventory")
    inventory = []
    for video_id, record in sorted(videos.items()):
        if not isinstance(video_id, str) or not video_id or not isinstance(record, dict):
            raise IdentityError("feature cache inventory contains an invalid video record")
        required = {"file", "sha256", "num_tokens", "feature_dim", "dtype", "source_frames"}
        if not required.issubset(record):
            raise IdentityError(f"feature cache record {video_id} lacks hashed geometry")
        feature_path = (Path(cache_root) / record["file"]).resolve()
        expected = _require_sha(record["sha256"], f"feature cache {video_id}.sha256")
        actual = sha256_file(feature_path)
        if actual != expected:
            raise IdentityError(f"feature cache content hash differs for {video_id}")
        source_frames = record["source_frames"]
        if (
            not isinstance(source_frames, list)
            or not source_frames
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in source_frames)
            or source_frames != sorted(set(source_frames))
        ):
            raise IdentityError(f"feature cache source frames are invalid for {video_id}")
        if record["num_tokens"] != len(source_frames):
            raise IdentityError(f"feature cache token geometry differs for {video_id}")
        inventory.append(
            {
                "video_id": video_id,
                "file": record["file"],
                "sha256": expected,
                "num_tokens": record["num_tokens"],
                "feature_dim": record["feature_dim"],
                "dtype": record["dtype"],
                "source_frames_sha256": canonical_json_sha256(source_frames),
            }
        )
    return manifest, inventory


def build_data_identity(cfg):
    """Dereference and hash every data object used by the locked Q2 route."""

    role_paths = {
        "annotation": _cfg_get(cfg, "annotation_path"),
        "class_map": _cfg_get(cfg, "class_map"),
        "development_split": _cfg_get(cfg, "development_split_manifest"),
        "fit_core": _cfg_get(cfg, "fit_core_manifest"),
        "calibration": _cfg_get(cfg, "calibration_manifest"),
        "reporting_ids": _cfg_get(cfg, "reporting_manifest"),
        "reporting_universe": _cfg_get(cfg, "reporting_universe_manifest"),
        "reporting_comparison": _cfg_get(cfg, "reporting_comparison_manifest"),
        "feature_cache_manifest": _cfg_get(cfg, "feature_cache_manifest"),
    }
    files = {role: _file_record(path, role) for role, path in role_paths.items()}
    cache_manifest, inventory = _feature_inventory(
        role_paths["feature_cache_manifest"],
        _cfg_get(cfg, "feature_cache_path"),
    )
    if cache_manifest.get("annotation_sha256") != files["annotation"]["sha256"]:
        raise IdentityError("feature cache was built from a different annotation file")

    optional = {}
    fineaction_path = _cfg_get(cfg, "fineaction_qualification_manifest")
    optional["fineaction_qualification"] = (
        None if fineaction_path is None else _file_record(fineaction_path, "FineAction qualification")
    )
    payload = {
        "schema_version": DATA_IDENTITY_SCHEMA,
        "files": files,
        "feature_inventory_count": len(inventory),
        "feature_inventory_sha256": canonical_json_sha256(inventory),
        "optional_qualifications": optional,
    }
    payload["identity_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_data_identity(identity, cfg):
    actual = build_data_identity(cfg)
    if identity != actual:
        raise IdentityError("launch data identity differs from dereferenced data")
    return actual


def _git(root, *args):
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise IdentityError(f"failed to inspect repository source identity: {exc}") from exc


def scientific_source_identity(repository_root, scientific_config_sha256):
    """Hash every tracked source except the two explicit authorization literals."""

    repository_root = Path(repository_root).resolve()
    tracked = [item for item in _git(repository_root, "ls-files", "-z").split("\0") if item]
    if not tracked:
        raise IdentityError("repository has no tracked files")
    records = []
    for relative in sorted(tracked):
        normalized = relative.replace("\\", "/")
        if normalized == AUTHORIZATION_CONFIG_PATH:
            continue
        records.append({"path": normalized, "sha256": sha256_file(repository_root / relative)})
    payload = {
        "schema_version": SCIENTIFIC_SOURCE_SCHEMA,
        "authorization_config_path": AUTHORIZATION_CONFIG_PATH,
        "scientific_config_sha256": _require_sha(
            scientific_config_sha256, "scientific config hash"
        ),
        "tracked_files": records,
    }
    return canonical_json_sha256(payload)


def authorization_only_diff(repository_root, profile_commit, formal_commit):
    """Require the profile-to-formal commit delta to be exactly two auth literals."""

    root = Path(repository_root).resolve()
    names = [
        line.strip().replace("\\", "/")
        for line in _git(root, "diff", "--name-only", profile_commit, formal_commit).splitlines()
        if line.strip()
    ]
    if names != [AUTHORIZATION_CONFIG_PATH]:
        raise IdentityError(
            "profile reuse requires an authorization-only diff; "
            f"changed files={names}"
        )
    diff = _git(
        root,
        "diff",
        "--unified=0",
        profile_commit,
        formal_commit,
        "--",
        AUTHORIZATION_CONFIG_PATH,
    )
    removed = sorted(
        line[1:].strip()
        for line in diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    added = sorted(
        line[1:].strip()
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    expected_removed = sorted(
        [
            'formal_training_ready = False',
            'gpu_authorization = "BLOCKED_UNTIL_B0_AND_PROFILE"',
        ]
    )
    expected_added = sorted(
        [
            'formal_training_ready = True',
            'gpu_authorization = "FORMAL_TRAINING_APPROVED"',
        ]
    )
    if removed != expected_removed or added != expected_added:
        raise IdentityError(
            "profile reuse diff contains changes beyond the two authorization literals"
        )
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def build_runtime_identity(
    *,
    entrypoint,
    seed,
    run_id,
    deterministic,
    not_eval,
    resume_path,
    cfg_overrides,
):
    if entrypoint not in {"train", "test"}:
        raise IdentityError("runtime entrypoint must be train or test")
    for value, label in ((seed, "seed"), (run_id, "run_id")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise IdentityError(f"runtime {label} must be a non-negative integer")
    if not isinstance(deterministic, bool) or not isinstance(not_eval, bool):
        raise IdentityError("runtime deterministic/not_eval flags must be booleans")
    if not isinstance(cfg_overrides, Mapping):
        raise IdentityError("runtime cfg overrides must be an object")
    checkpoint = None if resume_path is None else _file_record(resume_path, "resume checkpoint")
    return {
        "schema_version": RUNTIME_IDENTITY_SCHEMA,
        "entrypoint": entrypoint,
        "seed": seed,
        "run_id": run_id,
        "deterministic": deterministic,
        "not_eval": not_eval,
        "resume_checkpoint": checkpoint,
        "cfg_overrides": dict(cfg_overrides),
    }


def inspect_slurm_allocation(job_id, *, scontrol="scontrol"):
    if not isinstance(job_id, str) or not job_id.strip():
        raise IdentityError("Slurm job id must be non-empty")
    try:
        output = subprocess.run(
            [scontrol, "show", "job", "--oneliner", job_id],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise IdentityError(f"cannot authenticate Slurm allocation {job_id}: {exc}") from exc
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9]*)=([^\s]*)", output))
    if fields.get("JobId") != job_id:
        raise IdentityError("scontrol returned a different Slurm job")
    user = fields.get("UserId", "").split("(", 1)[0]
    gres = " ".join(
        fields.get(name, "") for name in ("TresPerNode", "Gres", "TresAlloc")
    )
    gpu_matches = [int(value) for value in re.findall(r"(?:gpu(?::[^:=,]+)?[:=])(\d+)", gres)]
    return SlurmAllocation(
        job_id=job_id,
        state=fields.get("JobState", ""),
        user=user,
        nodes=int(fields.get("NumNodes", "0") or 0),
        tasks=int(fields.get("NumTasks", "0") or 0),
        gpus=max(gpu_matches, default=0),
        command=fields.get("Command", ""),
        work_dir=fields.get("WorkDir", ""),
    )


def validate_slurm_allocation(allocation, *, job_id, world_size, expected_user=None):
    if not isinstance(allocation, SlurmAllocation):
        raise IdentityError("Slurm allocation evidence has the wrong type")
    expected_user = expected_user or getpass.getuser()
    if allocation.job_id != job_id or allocation.state != "RUNNING":
        raise IdentityError("Slurm allocation is not the active requested job")
    if allocation.user != expected_user:
        raise IdentityError("Slurm allocation belongs to a different user")
    if allocation.nodes != 1 or allocation.tasks != world_size or allocation.gpus != world_size:
        raise IdentityError(
            "Slurm allocation geometry differs from the one-node/one-rank-per-GPU contract"
        )
    if not allocation.command or not allocation.work_dir:
        raise IdentityError("Slurm allocation lacks command/work-directory provenance")
    return allocation


__all__ = [
    "AUTHORIZATION_CONFIG_PATH",
    "DATA_IDENTITY_SCHEMA",
    "IdentityError",
    "RUNTIME_IDENTITY_SCHEMA",
    "SlurmAllocation",
    "authorization_only_diff",
    "build_data_identity",
    "build_runtime_identity",
    "canonical_json_sha256",
    "inspect_slurm_allocation",
    "scientific_source_identity",
    "validate_data_identity",
    "validate_slurm_allocation",
]
