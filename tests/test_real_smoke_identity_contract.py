"""Fail-closed contracts for the real-data smoke and exact source identity.

These tests intentionally reject a unit-test-only smoke.  The release gate must
touch the official training inputs, execute one optimisation batch through all
five scientific lanes, reload the resulting checkpoint, and bind every worker
to the same clean Git source and protocol manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MANIFEST = ROOT / "experiment_configs" / "eventmatr_bxo_official_thumos14.json"
REAL_SMOKE = SCRIPTS / "run_eventmatr_real_smoke.py"
IDENTITY = SCRIPTS / "verify_source_identity.py"
SMOKE_WORKER = SCRIPTS / "slurm_eventmatr_smoke.sh"
TRAIN_WORKER = SCRIPTS / "slurm_eventmatr_train.sh"
LAUNCHER = SCRIPTS / "submit_eventmatr_bxo_official_slurm.sh"
COMPLETION = SCRIPTS / "verify_eventmatr_completion.py"

LANES = ("native_matr", "b0o0", "b1o0", "b0o1", "b1o1")
IDENTITY_ENV = (
    "MATR_SOURCE_COMMIT",
    "MATR_SOURCE_TREE",
    "MATR_MANIFEST_SHA256",
    "MATR_SMOKE_RECEIPT",
)


def _text(path: Path) -> str:
    assert path.is_file(), f"missing scientific gate implementation: {path}"
    return path.read_text(encoding="utf-8")


def _run(*argv: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_manifest_freezes_real_smoke_and_exact_source_identity() -> None:
    manifest = json.loads(_text(MANIFEST))
    smoke = manifest["smoke_protocol"]
    assert smoke["input"] == "real_official_training_data"
    assert smoke["batches_per_lane"] == 1
    assert smoke["batch_size"] == 64
    assert smoke["lanes"] == list(LANES)
    assert smoke["test_access"] is False
    assert smoke["receipt"] == "eventmatr_real_smoke.json"
    assert smoke["required_status"] == "PASS"
    assert smoke["forward"] is True
    assert smoke["backward"] is True
    assert smoke["optimizer_step"] is True
    assert smoke["checkpoint_reload"] is True
    assert smoke["finite_loss"] is True
    assert smoke["event_gradient_for_event_lanes"] is True
    assert smoke["owner_gradient_for_event_lanes"] is True

    identity = manifest["source_identity_protocol"]
    assert identity["clean_worktree_required"] is True
    assert identity["fields"] == ["commit", "tree", "manifest_sha256"]
    assert identity["worker_reverify"] is True
    assert identity["dirty_fails"] is True


def test_real_smoke_uses_official_inputs_and_one_batch_for_all_lanes() -> None:
    source = _text(REAL_SMOKE)
    lower = source.lower()
    for variable in (
        "MATR_TRAIN_FEATURE",
        "MATR_PROTOCOL_ANNO",
        "MATR_VIDEO_LEN_PATTERN",
        "MATR_LABEL_PATTERN",
    ):
        assert variable in source
    for lane in LANES:
        assert lane in source

    # A synthetic/random tensor smoke is not acceptable evidence.
    assert "torch.randn" not in source
    assert "torch.rand(" not in source
    assert "synthetic" not in lower
    assert "fake" not in lower
    assert re.search(r"(?:next\s*\(\s*iter|islice\s*\().*(?:loader|dataloader)", source, re.DOTALL)

    # The smoke must exercise a real train step and persistence boundary.
    assert ".backward(" in source
    assert ".step(" in source
    assert "load_state_dict" in source
    assert "torch.load" in source
    assert "isfinite" in lower
    assert "event_grad" in lower
    assert "owner_grad" in lower

    # Receipt semantics are scientific data, not a successful exit code alone.
    assert "eventmatr_real_smoke.json" in source
    assert '"status"' in source or "'status'" in source
    assert "PASS" in source
    assert "test_access" in source
    assert re.search(r"test_access[^\n]*(?:false|False)", source)


def test_smoke_slurm_runs_identity_check_and_real_smoke_not_only_pytest() -> None:
    source = _text(SMOKE_WORKER)
    assert "verify_source_identity.py" in source
    assert "run_eventmatr_real_smoke.py" in source
    assert "MATR_SMOKE_RECEIPT" in source
    for variable in (
        "MATR_TRAIN_FEATURE",
        "MATR_PROTOCOL_ANNO",
        "MATR_VIDEO_LEN_PATTERN",
        "MATR_LABEL_PATTERN",
    ):
        assert variable in source
    # Unit tests remain useful, but cannot be the terminal smoke action.
    assert source.find("run_eventmatr_real_smoke.py") > source.find("pytest")


def test_launcher_captures_identity_and_every_training_job_is_afterok_smoke() -> None:
    source = _text(LAUNCHER)
    for variable in IDENTITY_ENV:
        assert variable in source
        assert re.search(rf"export\s+{variable}(?:=|\b)", source)
    assert "verify_source_identity.py" in source
    assert "source_identity.json" in source
    assert "eventmatr_real_smoke.json" in source

    lane_loop = re.search(
        r"for\s+LANE\s+in\s+native_matr\s+b0o0\s+b1o0\s+b0o1\s+b1o1;\s*do(?P<body>.*?)done",
        source,
        re.DOTALL,
    )
    assert lane_loop, "launcher must submit exactly the five registered lanes"
    body = lane_loop.group("body")
    assert body.count("sbatch") == 1
    assert re.search(r"--dependency=[\"']?afterok:\$\{SMOKE_JOB\}", body)
    assert "slurm_eventmatr_train.sh" in body


def test_train_worker_reverifies_identity_before_lane_exec_and_writes_receipt() -> None:
    source = _text(TRAIN_WORKER)
    for variable in IDENTITY_ENV:
        assert variable in source
    verify_at = source.find("verify_source_identity.py")
    assert verify_at >= 0
    assert "source_identity_lanes" in source
    assert "${MATR_LANE}.json" in source
    for command in ("train_native_matr.sh", '"scripts/train_${MATR_LANE}.sh"'):
        train_at = source.find(command)
        assert train_at > verify_at, f"{command} can run before source identity verification"


def test_completion_receipt_binds_smoke_and_all_lane_identities() -> None:
    source = _text(COMPLETION)
    lower = source.lower()
    assert "eventmatr_real_smoke.json" in source
    assert "source_identity.json" in source
    assert "source_identity_lanes" in source
    for field in ("commit", "tree", "manifest_sha256"):
        assert field in lower
    assert "PASS" in source
    assert "test_access" in lower
    for lane in LANES:
        assert lane in source


def test_identity_verifier_accepts_clean_repo_and_rejects_dirty_or_mismatch(
    tmp_path: Path,
) -> None:
    assert IDENTITY.is_file(), f"missing identity verifier: {IDENTITY}"
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = repo / "protocol.json"
    tracked = repo / "tracked.txt"
    manifest.write_text('{"protocol":"test"}\n', encoding="utf-8")
    tracked.write_text("clean\n", encoding="utf-8")

    assert _run("git", "init", cwd=repo).returncode == 0
    assert _run("git", "config", "user.email", "contract@example.invalid", cwd=repo).returncode == 0
    assert _run("git", "config", "user.name", "Contract Test", cwd=repo).returncode == 0
    assert _run("git", "add", "protocol.json", "tracked.txt", cwd=repo).returncode == 0
    assert _run("git", "commit", "-m", "clean source", cwd=repo).returncode == 0

    output = repo / "identity.json"
    clean = _run(
        sys.executable,
        str(IDENTITY),
        "--project-dir",
        str(repo),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        cwd=repo,
    )
    assert clean.returncode == 0, clean.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    commit = _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
    tree = _run("git", "rev-parse", "HEAD^{tree}", cwd=repo).stdout.strip()
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert receipt["status"] == "PASS"
    assert receipt["clean"] is True
    assert receipt["commit"] == commit
    assert receipt["tree"] == tree
    assert receipt["manifest_sha256"] == manifest_sha256

    expected = (
        "--expected-commit",
        commit,
        "--expected-tree",
        tree,
        "--expected-manifest-sha256",
        manifest_sha256,
    )
    smoke_path = tmp_path / "preflight.json"
    smoke_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "test_access": False,
                "formal_training_started": False,
                "paper_performance_valid": False,
                "threshold_search": False,
                "checkpoint_updated": False,
                "source_identity": {
                    "commit": commit,
                    "tree": tree,
                    "manifest_sha256": manifest_sha256,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with_smoke = _run(
        sys.executable,
        str(IDENTITY),
        "--project-dir",
        str(repo),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        *expected,
        "--smoke-receipt",
        str(smoke_path),
        cwd=repo,
    )
    assert with_smoke.returncode == 0, with_smoke.stderr
    compact_smoke = json.loads(output.read_text(encoding="utf-8"))["smoke"]
    assert compact_smoke["test_access"] is False
    assert compact_smoke["formal_training_started"] is False
    assert compact_smoke["paper_performance_valid"] is False
    assert compact_smoke["threshold_search"] is False
    assert compact_smoke["checkpoint_updated"] is False

    verified = _run(
        sys.executable,
        str(IDENTITY),
        "--project-dir",
        str(repo),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        *expected,
        cwd=repo,
    )
    assert verified.returncode == 0, verified.stderr

    tracked.write_text("dirty\n", encoding="utf-8")
    dirty = _run(
        sys.executable,
        str(IDENTITY),
        "--project-dir",
        str(repo),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        *expected,
        cwd=repo,
    )
    assert dirty.returncode != 0

    tracked.write_text("clean\n", encoding="utf-8")
    untracked_path = repo / "untracked.txt"
    untracked_path.write_text("must fail\n", encoding="utf-8")
    untracked = _run(
        sys.executable,
        str(IDENTITY),
        "--project-dir",
        str(repo),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        *expected,
        cwd=repo,
    )
    assert untracked.returncode != 0
    untracked_path.unlink()

    mismatch = _run(
        sys.executable,
        str(IDENTITY),
        "--project-dir",
        str(repo),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--expected-commit",
        "0" * 40,
        "--expected-tree",
        tree,
        "--expected-manifest-sha256",
        manifest_sha256,
        cwd=repo,
        env={**os.environ},
    )
    assert mismatch.returncode != 0


def test_identity_verifier_does_not_inherit_launch_identity_for_another_repo(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "isolated-repo"
    repo.mkdir()
    manifest = repo / "protocol.json"
    tracked = repo / "tracked.txt"
    manifest.write_text('{"protocol":"test"}\n', encoding="utf-8")
    tracked.write_text("clean\n", encoding="utf-8")
    assert _run("git", "init", cwd=repo).returncode == 0
    assert _run("git", "config", "user.email", "contract@example.invalid", cwd=repo).returncode == 0
    assert _run("git", "config", "user.name", "Contract Test", cwd=repo).returncode == 0
    assert _run("git", "add", "protocol.json", "tracked.txt", cwd=repo).returncode == 0
    assert _run("git", "commit", "-m", "clean source", cwd=repo).returncode == 0

    monkeypatch.setenv("MATR_SOURCE_COMMIT", "0" * 40)
    result = _run(
        sys.executable,
        str(IDENTITY),
        "--project-dir",
        str(repo),
        "--manifest",
        str(manifest),
        "--output",
        str(repo / "identity.json"),
        cwd=repo,
    )
    assert result.returncode == 0, result.stderr
