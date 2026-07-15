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

from .evidence_bundle import EvidenceBundleError, relative_bundle_path


DATA_IDENTITY_SCHEMA = "full-petal-data-identity-v1"
RUNTIME_IDENTITY_SCHEMA = "full-petal-runtime-identity-v1"
SCIENTIFIC_SOURCE_SCHEMA = "full-petal-scientific-source-v1"
EVALUATOR_SPEC_SCHEMA = "full-petal-evaluator-spec-v1"
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


def _file_record(path, label, *, bundle_root=None):
    if not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise IdentityError(f"{label} path must be non-empty")
    resolved = Path(path).expanduser().resolve()
    try:
        recorded_path = (
            str(resolved)
            if bundle_root is None
            else relative_bundle_path(resolved, bundle_root, label)
        )
    except EvidenceBundleError as exc:
        raise IdentityError(str(exc)) from exc
    return {
        "path": recorded_path,
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


def _nested_cfg_get(cfg, *keys):
    value = cfg
    for key in keys:
        value = _cfg_get(value, key)
        if value is None:
            raise IdentityError(f"config is missing {'.'.join(keys)}")
    return value


def _plain_mapping(value, label):
    if not isinstance(value, Mapping):
        raise IdentityError(f"{label} must be an object")
    return {key: item for key, item in value.items()}


def _identity_file(identity, role):
    if not isinstance(identity, Mapping):
        raise IdentityError("data identity must be an object")
    files = identity.get("files")
    if not isinstance(files, Mapping) or role not in files:
        raise IdentityError(f"data identity lacks the {role} file record")
    record = files[role]
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "size_bytes"}:
        raise IdentityError(f"data identity {role} record fields differ")
    resolved = Path(record["path"]).expanduser().resolve()
    if not resolved.is_file():
        raise IdentityError(f"data identity {role} file does not exist: {resolved}")
    if sha256_file(resolved) != _require_sha(record["sha256"], f"{role}.sha256"):
        raise IdentityError(f"data identity {role} file hash differs")
    if resolved.stat().st_size != record["size_bytes"]:
        raise IdentityError(f"data identity {role} file size differs")
    return resolved, dict(record)


def _require_same_identity_file(path, identity, role, label):
    expected_path, record = _identity_file(identity, role)
    actual_path = Path(path).expanduser().resolve()
    if actual_path != expected_path:
        raise IdentityError(f"{label} path differs from data identity {role}")
    if sha256_file(actual_path) != record["sha256"]:
        raise IdentityError(f"{label} content differs from data identity {role}")
    return actual_path


def derive_evaluator_spec(cfg):
    """Derive the only evaluator specification accepted for a resolved config."""

    evaluation = _plain_mapping(_nested_cfg_get(cfg, "evaluation"), "evaluation config")
    required = {
        "type",
        "subset",
        "allowed_videos",
        "tiou_thresholds",
        "latency_budgets_sec",
        "fps",
        "require_ledger",
        "require_no_future",
        "ground_truth_filename",
        "identity_tiou_threshold",
        "identity_latency_budget_sec",
    }
    if set(evaluation) != required:
        raise IdentityError(
            "evaluation config fields differ; "
            f"missing={sorted(required - set(evaluation))}, "
            f"extra={sorted(set(evaluation) - required)}"
        )
    if evaluation["type"] != "OnlineAPBudgeted":
        raise IdentityError("evaluation type must be OnlineAPBudgeted")
    if evaluation["require_ledger"] is not True or evaluation["require_no_future"] is not True:
        raise IdentityError("evaluation must require an immutable no-future ledger")
    return {
        "schema_version": EVALUATOR_SPEC_SCHEMA,
        "type": evaluation["type"],
        "subset": evaluation["subset"],
        "tiou_thresholds": list(evaluation["tiou_thresholds"]),
        "latency_budgets_sec": list(evaluation["latency_budgets_sec"]),
        "fps": evaluation["fps"],
        "identity_tiou_threshold": evaluation["identity_tiou_threshold"],
        "identity_latency_budget_sec": evaluation["identity_latency_budget_sec"],
    }


def validate_evaluation_artifact_bindings(
    cfg,
    data_identity,
    *,
    ground_truth_path,
    allowed_videos_path,
    evaluator_spec,
):
    """Bind evaluator inputs and thresholds to the launch data/config identity."""

    evaluation = _plain_mapping(_nested_cfg_get(cfg, "evaluation"), "evaluation config")
    ground_truth = _require_same_identity_file(
        ground_truth_path,
        data_identity,
        "annotation",
        "ground-truth artifact",
    )
    allowed_videos = _require_same_identity_file(
        allowed_videos_path,
        data_identity,
        "calibration",
        "allowed-videos artifact",
    )
    if Path(evaluation["ground_truth_filename"]).expanduser().resolve() != ground_truth:
        raise IdentityError("evaluation ground truth differs from the bound annotation")
    if Path(evaluation["allowed_videos"]).expanduser().resolve() != allowed_videos:
        raise IdentityError("evaluation allow-list differs from the bound calibration split")
    expected_spec = derive_evaluator_spec(cfg)
    if evaluator_spec != expected_spec:
        raise IdentityError("evaluator specification differs from the resolved config")
    return expected_spec


def derive_training_data_order_sha256(cfg, data_identity, *, seed):
    """Reconstruct the exact chronological packet order from bound data artifacts."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise IdentityError("training data-order seed must be a non-negative integer")
    train_cfg = _plain_mapping(_nested_cfg_get(cfg, "dataset", "train"), "training dataset")
    annotation_path = _require_same_identity_file(
        train_cfg.get("ann_file"), data_identity, "annotation", "training annotation"
    )
    allow_path = _require_same_identity_file(
        train_cfg.get("allow_list"), data_identity, "fit_core", "training allow-list"
    )
    cache_path = _require_same_identity_file(
        train_cfg.get("cache_manifest"),
        data_identity,
        "feature_cache_manifest",
        "training cache manifest",
    )
    chunk_size = train_cfg.get("chunk_size")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise IdentityError("training chunk_size must be a positive integer")
    subsets = train_cfg.get("subset_name")
    subsets = {subsets} if isinstance(subsets, str) else set(subsets or ())
    if not subsets or not all(isinstance(item, str) and item for item in subsets):
        raise IdentityError("training subset_name is invalid")
    try:
        annotation = _load_json(annotation_path, "training annotation")
        cache_manifest = _load_json(cache_path, "training feature cache manifest")
        allowed_ids = [
            line.strip()
            for line in allow_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError) as exc:
        raise IdentityError(f"failed to load training order inputs: {exc}") from exc
    if len(allowed_ids) != len(set(allowed_ids)) or not allowed_ids:
        raise IdentityError("training allow-list must contain unique video IDs")
    database = annotation.get("database")
    videos = cache_manifest.get("videos")
    if not isinstance(database, Mapping) or not isinstance(videos, Mapping):
        raise IdentityError("training annotation/cache inventory is malformed")
    allowed = set(allowed_ids)
    selected = []
    episodes = []
    packet_offset = 0
    for video_id, video_info in database.items():
        if not isinstance(video_info, Mapping) or video_info.get("subset") not in subsets:
            continue
        if video_id not in allowed:
            continue
        record = videos.get(video_id)
        if not isinstance(record, Mapping):
            raise IdentityError(f"training cache lacks video {video_id}")
        source_frames = record.get("source_frames")
        if not isinstance(source_frames, list) or not source_frames:
            raise IdentityError(f"training cache source frames are invalid for {video_id}")
        packet_count = (len(source_frames) + chunk_size - 1) // chunk_size
        packet_indices = list(range(packet_offset, packet_offset + packet_count))
        episodes.append({"video_id": str(video_id), "packet_indices": packet_indices})
        selected.append(video_id)
        packet_offset += packet_count
    if set(selected) != allowed:
        raise IdentityError(
            "training order does not consume the exact fit-core allow-list; "
            f"missing={sorted(allowed - set(selected))}"
        )
    return canonical_json_sha256({"seed": seed, "episodes": episodes})


def derive_training_trace_identity(cfg, data_identity, *, seed, world_size):
    """Derive every immutable identity field expected in optimizer-event traces."""

    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise IdentityError("training world_size must be a positive integer")
    optimizer = _plain_mapping(_nested_cfg_get(cfg, "optimizer"), "optimizer config")
    scheduler = _plain_mapping(_nested_cfg_get(cfg, "scheduler"), "scheduler config")
    model = _plain_mapping(_nested_cfg_get(cfg, "model"), "model config")
    solver = _plain_mapping(_nested_cfg_get(cfg, "solver"), "solver config")
    train_solver = _plain_mapping(_nested_cfg_get(solver, "train"), "training solver config")
    batch_size = train_solver.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise IdentityError("training batch_size must be a positive integer")
    if not bool(solver.get("amp", False)):
        precision = "fp32"
    else:
        amp_dtype = str(solver.get("amp_dtype", "fp16")).strip().lower()
        precision_aliases = {
            "fp16": "fp16",
            "float16": "fp16",
            "bf16": "bf16",
            "bfloat16": "bf16",
        }
        if amp_dtype not in precision_aliases:
            raise IdentityError("training amp_dtype is unsupported")
        precision = precision_aliases[amp_dtype]
    return {
        "precision": precision,
        "optimizer_config_sha256": canonical_json_sha256(optimizer),
        "scheduler_config_sha256": canonical_json_sha256(scheduler),
        "data_order_sha256": derive_training_data_order_sha256(
            cfg, data_identity, seed=seed
        ),
        "loss_normalization_sha256": canonical_json_sha256(
            {"model": model, "solver": solver}
        ),
        "effective_batch_size": batch_size * world_size,
        "world_size": world_size,
    }


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
    try:
        ancestry = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                profile_commit,
                formal_commit,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise IdentityError(f"failed to inspect profile commit ancestry: {exc}") from exc
    if ancestry.returncode == 1:
        raise IdentityError("profile commit is not an ancestor of the formal commit")
    if ancestry.returncode != 0:
        detail = (ancestry.stderr or ancestry.stdout).strip()
        raise IdentityError(
            "failed to inspect profile commit ancestry"
            + (f": {detail}" if detail else "")
        )
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
    bundle_root=None,
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
    checkpoint = (
        None
        if resume_path is None
        else _file_record(
            resume_path, "resume checkpoint", bundle_root=bundle_root
        )
    )
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


def inspect_slurm_allocation(job_id, *, scontrol_path):
    if not isinstance(job_id, str) or not job_id.strip():
        raise IdentityError("Slurm job id must be non-empty")
    if (
        not isinstance(scontrol_path, (str, os.PathLike))
        or not Path(scontrol_path).is_absolute()
    ):
        raise IdentityError("trusted scontrol path must be absolute")
    lexical = Path(os.path.abspath(os.fspath(scontrol_path)))
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IdentityError("trusted scontrol executable does not exist") from exc
    if resolved != lexical or lexical.is_symlink() or not resolved.is_file():
        raise IdentityError("trusted scontrol executable cannot be a link")
    try:
        output = subprocess.run(
            [str(resolved), "show", "job", "--oneliner", job_id],
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
    "EVALUATOR_SPEC_SCHEMA",
    "IdentityError",
    "RUNTIME_IDENTITY_SCHEMA",
    "SlurmAllocation",
    "authorization_only_diff",
    "build_data_identity",
    "build_runtime_identity",
    "canonical_json_sha256",
    "derive_evaluator_spec",
    "derive_training_data_order_sha256",
    "derive_training_trace_identity",
    "inspect_slurm_allocation",
    "scientific_source_identity",
    "validate_data_identity",
    "validate_evaluation_artifact_bindings",
    "validate_slurm_allocation",
]
