import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import opentad.utils.full_petal_identity as identity_module
from opentad.utils.full_petal_identity import (
    IdentityError,
    SlurmAllocation,
    authorization_only_diff,
    build_data_identity,
    build_runtime_identity,
    inspect_slurm_allocation,
    scientific_source_identity,
    validate_data_identity,
    validate_slurm_allocation,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def _data_fixture(root):
    annotation = _write(root / "annotation.json", "{}\n")
    class_map = _write(root / "classes.txt", "action\n")
    paths = {
        "development_split_manifest": _write(root / "development.json", "{}\n"),
        "fit_core_manifest": _write(root / "fit.txt", "video-a\n"),
        "calibration_manifest": _write(root / "calibration.txt", "video-a\n"),
        "reporting_manifest": _write(root / "reporting.txt", "video-a\n"),
        "reporting_universe_manifest": _write(root / "reporting.json", "{}\n"),
        "reporting_comparison_manifest": _write(root / "comparison.json", "{}\n"),
    }
    cache = root / "features"
    feature = _write(cache / "video-a.npy", "immutable-feature-bytes\n")
    manifest = {
        "schema": "ontad_feature_cache_v1",
        "annotation_sha256": _sha(annotation),
        "videos": {
            "video-a": {
                "file": feature.name,
                "sha256": _sha(feature),
                "num_tokens": 2,
                "feature_dim": 4,
                "dtype": "float32",
                "source_frames": [7, 15],
            }
        },
    }
    manifest_path = cache / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "annotation_path": str(annotation),
        "class_map": str(class_map),
        "feature_cache_path": str(cache),
        "feature_cache_manifest": str(manifest_path),
        "fineaction_qualification_manifest": None,
        **{key: str(value) for key, value in paths.items()},
    }, feature


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_repository(root):
    _git(root, "init")
    _git(root, "config", "user.email", "full-petal@example.invalid")
    _git(root, "config", "user.name", "Full PETAL Test")
    config = _write(
        root / "configs" / "causaltad" / "thumos_pes_q2_base.py",
        'formal_training_ready = False\n'
        'gpu_authorization = "BLOCKED_UNTIL_B0_AND_PROFILE"\n',
    )
    source = _write(root / "opentad" / "model.py", "SCIENTIFIC_VALUE = 1\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "profile source")
    return config, source, _git(root, "rev-parse", "HEAD")


def test_data_identity_dereferences_every_feature_and_rejects_tampering(tmp_path):
    cfg, feature = _data_fixture(tmp_path)
    locked = build_data_identity(cfg)

    assert locked["feature_inventory_count"] == 1
    assert locked["files"]["annotation"]["sha256"] == _sha(cfg["annotation_path"])
    assert validate_data_identity(locked, cfg) == locked

    feature.write_text("tampered\n", encoding="utf-8", newline="\n")
    with pytest.raises(IdentityError, match="content hash differs"):
        validate_data_identity(locked, cfg)


def test_scientific_source_identity_excludes_only_authorization_config(tmp_path):
    config, source, _ = _init_repository(tmp_path)
    initial = scientific_source_identity(tmp_path, "a" * 64)

    config.write_text(
        'formal_training_ready = True\n'
        'gpu_authorization = "FORMAL_TRAINING_APPROVED"\n',
        encoding="utf-8",
        newline="\n",
    )
    assert scientific_source_identity(tmp_path, "a" * 64) == initial

    source.write_text("SCIENTIFIC_VALUE = 2\n", encoding="utf-8", newline="\n")
    assert scientific_source_identity(tmp_path, "a" * 64) != initial


def test_profile_reuse_accepts_exact_authorization_literals_only(tmp_path):
    config, _, profile_commit = _init_repository(tmp_path)
    config.write_text(
        'formal_training_ready = True\n'
        'gpu_authorization = "FORMAL_TRAINING_APPROVED"\n',
        encoding="utf-8",
        newline="\n",
    )
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "authorize formal training")
    formal_commit = _git(tmp_path, "rev-parse", "HEAD")

    assert len(authorization_only_diff(tmp_path, profile_commit, formal_commit)) == 64

    with config.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("SCIENTIFIC_AXIS = 2\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "forbidden scientific change")
    forbidden_commit = _git(tmp_path, "rev-parse", "HEAD")
    with pytest.raises(IdentityError, match="beyond the two authorization literals"):
        authorization_only_diff(tmp_path, profile_commit, forbidden_commit)


def test_runtime_identity_hash_binds_resume_checkpoint_contents(tmp_path):
    checkpoint = _write(tmp_path / "checkpoint.pth", "checkpoint-v1\n")
    first = build_runtime_identity(
        entrypoint="train",
        seed=705,
        run_id=0,
        deterministic=True,
        not_eval=False,
        resume_path=checkpoint,
        cfg_overrides={"work_dir": "/run/a"},
    )
    checkpoint.write_text("checkpoint-v2\n", encoding="utf-8", newline="\n")
    second = build_runtime_identity(
        entrypoint="train",
        seed=705,
        run_id=0,
        deterministic=True,
        not_eval=False,
        resume_path=checkpoint,
        cfg_overrides={"work_dir": "/run/a"},
    )

    assert first["resume_checkpoint"]["sha256"] != second["resume_checkpoint"]["sha256"]


def test_slurm_identity_is_parsed_and_bound_to_active_allocation(monkeypatch):
    output = (
        "JobId=12345 JobState=RUNNING UserId=fixture-user(1000) "
        "NumNodes=1 NumTasks=1 TresPerNode=gres:gpu:1 "
        "Command=/repo/tools/train.py WorkDir=/repo\n"
    )

    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(identity_module.subprocess, "run", run)
    allocation = inspect_slurm_allocation("12345")

    assert allocation.gpus == 1
    assert validate_slurm_allocation(
        allocation,
        job_id="12345",
        world_size=1,
        expected_user="fixture-user",
    ) == allocation
    with pytest.raises(IdentityError, match="geometry"):
        validate_slurm_allocation(
            SlurmAllocation(**{**allocation.__dict__, "gpus": 2}),
            job_id="12345",
            world_size=1,
            expected_user="fixture-user",
        )
