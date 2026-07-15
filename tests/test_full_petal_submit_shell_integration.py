import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
READER_SCRIPT = ROOT / "tools" / "read_full_petal_launch_ticket.py"
SUBMIT_SCRIPT = ROOT / "tools" / "remote" / "submit_full_petal_q2_n16r4.sh"
SPEC = importlib.util.spec_from_file_location(
    "read_full_petal_launch_ticket", READER_SCRIPT
)
READER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READER)


def _ticket(path, *, work_dir=None, overrides=None, resume=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = str(path.parent / "work") if work_dir is None else work_dir
    payload = {
        "mode": "profile",
        "runtime_identity": {
            "schema_version": "full-petal-runtime-identity-v1",
            "entrypoint": "train",
            "seed": 705,
            "run_id": 9,
            "deterministic": True,
            "not_eval": False,
            "resume_checkpoint": resume,
            "cfg_overrides": overrides or {"work_dir": work_dir},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_ticket_reader_accepts_only_ticket_root_work_dir(tmp_path):
    ticket = _ticket(tmp_path / "run" / "launch_ticket.json")

    fields = READER.read_launcher_fields(ticket)

    assert fields["mode"] == "profile"
    assert fields["seed"] == "705"
    assert fields["run_id"] == "9"
    assert fields["work_dir"] == str(ticket.parent / "work")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"work_dir": "/tmp/drifted"}, "artifact root"),
        (
            {
                "overrides": {
                    "work_dir": "/tmp/work",
                    "optimizer.lr": 1e-3,
                }
            },
            "only work_dir",
        ),
        ({"resume": {"path": "/tmp/checkpoint.pth"}}, "resume"),
    ],
)
def test_ticket_reader_rejects_runtime_identity_drift(tmp_path, kwargs, match):
    ticket = _ticket(tmp_path / "run" / "launch_ticket.json", **kwargs)

    with pytest.raises(READER.TicketReadError, match=match):
        READER.read_launcher_fields(ticket)


def _run(command, cwd, **kwargs):
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        **kwargs,
    )


def test_submit_shell_consumes_ticket_identity_and_calls_fake_sbatch(tmp_path):
    if os.name == "nt":
        source = SUBMIT_SCRIPT.read_text(encoding="utf-8")
        assert 'TICKET_FIELDS[work_dir]' in source
        assert 'SCRIPT_PATH="$RUN_DIR/job.sbatch"' in source
        assert '"$SBATCH_BIN" "$SCRIPT_PATH"' in source
        assert "$(date" not in source
        git_bash = next(
            (
                candidate
                for candidate in (
                    Path("C:/Program Files/Git/bin/bash.exe"),
                    Path("C:/Program Files/Git/usr/bin/bash.exe"),
                )
                if candidate.is_file()
            ),
            None,
        )
        if git_bash is not None:
            _run([str(git_bash), "-n", str(SUBMIT_SCRIPT)], ROOT)
        return

    bash = shutil.which("bash")
    assert bash is not None, "the production POSIX launcher requires bash"
    deployment = tmp_path / "deployment"
    tools = deployment / "tools"
    tools.mkdir(parents=True)
    shutil.copy2(READER_SCRIPT, tools / READER_SCRIPT.name)
    _run(["git", "init"], deployment)
    _run(["git", "config", "user.email", "test@example.invalid"], deployment)
    _run(["git", "config", "user.name", "Full PETAL Test"], deployment)
    _run(["git", "add", "."], deployment)
    _run(["git", "commit", "-m", "test deployment"], deployment)

    artifact_root = tmp_path / "evidence" / "profile-run"
    ticket = _ticket(artifact_root / "launch_ticket.json")
    capture = tmp_path / "sbatch.capture"
    fake_sbatch = tmp_path / "fake-sbatch"
    fake_sbatch.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$1" >"$SBATCH_CAPTURE"\n',
        encoding="utf-8",
    )
    fake_sbatch.chmod(fake_sbatch.stat().st_mode | stat.S_IXUSR)
    env = os.environ.copy()
    env.update(
        {
            "BASE_DIR": str(deployment),
            "PYTHON_BIN": sys.executable,
            "SBATCH_BIN": str(fake_sbatch),
            "SBATCH_CAPTURE": str(capture),
        }
    )

    result = _run(
        [
            bash,
            str(SUBMIT_SCRIPT),
            "profile",
            "configs/causaltad/thumos_pes_q2_persist_fixed.py",
            str(ticket),
        ],
        ROOT,
        env=env,
    )

    script = artifact_root / "job.sbatch"
    source = script.read_text(encoding="utf-8")
    assert f"FULL_PETAL_RUN_DIR={artifact_root}" in result.stdout
    assert capture.read_text(encoding="utf-8").strip() == str(script)
    assert f"work_dir={artifact_root / 'work'}" in source
    assert "SEED=705" in source
    assert "RUN_ID=9" in source
    assert "$(date" not in source

    failed = subprocess.run(
        [
            bash,
            str(SUBMIT_SCRIPT),
            "profile",
            "configs/causaltad/thumos_pes_q2_persist_fixed.py",
            str(ticket),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 2
    assert "Refusing to overwrite" in failed.stderr
