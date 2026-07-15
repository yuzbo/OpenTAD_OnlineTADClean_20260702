import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest
from opentad.utils.evidence_bundle import EvidenceBundleError


ROOT = Path(__file__).resolve().parents[1]
READER_SCRIPT = ROOT / "tools" / "read_full_petal_launch_ticket.py"
SUBMITTER_SCRIPT = ROOT / "tools" / "submit_full_petal_slurm_script.py"
SUBMIT_SCRIPT = ROOT / "tools" / "remote" / "submit_full_petal_q2_n16r4.sh"
SPEC = importlib.util.spec_from_file_location(
    "read_full_petal_launch_ticket", READER_SCRIPT
)
READER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READER)
SUBMITTER_SPEC = importlib.util.spec_from_file_location(
    "submit_full_petal_slurm_script", SUBMITTER_SCRIPT
)
SUBMITTER = importlib.util.module_from_spec(SUBMITTER_SPEC)
SUBMITTER_SPEC.loader.exec_module(SUBMITTER)


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


def _bash_executable():
    candidates = []
    if os.name == "nt":
        candidates.extend(
            (
                Path("C:/Program Files/Git/bin/bash.exe"),
                Path("C:/Program Files/Git/usr/bin/bash.exe"),
            )
        )
    else:
        candidates.extend((Path("/usr/bin/bash"), Path("/bin/bash")))
    return next((str(path) for path in candidates if path.is_file()), None)


def test_submit_shell_uses_validated_atomic_byte_submission():
    source = SUBMIT_SCRIPT.read_text(encoding="utf-8")
    assert 'TICKET_FIELDS[work_dir]' in source
    assert 'SCRIPT_PATH="$RUN_DIR/job.sbatch"' in source
    assert "CFG_WORK_DIR_ARG=work_dir=$Q_WORK_DIR" in source
    assert '--cfg-options "$CFG_WORK_DIR_ARG"' in source
    assert "submit_full_petal_slurm_script.py" in source
    assert "SBATCH_BIN" not in source
    assert 'cat >"$SCRIPT_PATH"' not in source
    assert "$(date" not in source
    bash = _bash_executable()
    if bash is not None:
        _run([bash, "-n", str(SUBMIT_SCRIPT)], ROOT)


@pytest.mark.parametrize(
    "variable,value,message",
    (
        ("PROFILE_TIME", "01:00:00\n#SBATCH --gres=gpu:8", "Slurm time"),
        ("CPUS_PER_TASK", "4\nid", "CPUS_PER_TASK"),
        ("CPUS_PER_TASK", "65", "CPUS_PER_TASK"),
    ),
)
def test_submit_shell_rejects_resource_injection_before_ticket_access(
    variable, value, message
):
    bash = _bash_executable()
    if bash is None:
        pytest.skip("Bash is unavailable")
    env = os.environ.copy()
    env[variable] = value
    result = subprocess.run(
        [
            bash,
            str(SUBMIT_SCRIPT),
            "profile",
            "configs/causaltad/thumos_pes_q2_persist_fixed.py",
            "/missing/ticket.json",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert message in result.stderr


def _completed_submission():
    return subprocess.CompletedProcess(["/usr/bin/sbatch"], 0, b"Submitted\n", b"")


def test_submitter_publishes_and_submits_identical_in_memory_bytes(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    script = run_dir / "job.sbatch"
    payload = b"#!/usr/bin/env bash\necho exact\n"
    captured = {}

    def submitter(submitted, cwd):
        captured["payload"] = submitted
        captured["cwd"] = cwd
        script.write_bytes(b"#!/usr/bin/env bash\necho replaced\n")
        return _completed_submission()

    _, digest = SUBMITTER.publish_and_submit(
        script,
        payload,
        submission_cwd=tmp_path,
        submitter=submitter,
    )

    assert captured == {"payload": payload, "cwd": tmp_path.resolve()}
    assert digest == hashlib.sha256(payload).hexdigest()
    assert script.read_bytes() != captured["payload"]


def test_submitter_exclusive_publication_has_one_concurrent_winner(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    script = run_dir / "job.sbatch"

    def attempt(index):
        payload = f"#!/usr/bin/env bash\necho {index}\n".encode("ascii")
        return SUBMITTER.publish_and_submit(
            script,
            payload,
            submission_cwd=tmp_path,
            submitter=lambda submitted, cwd: _completed_submission(),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attempt, index) for index in range(2)]
    outcomes = []
    for future in futures:
        try:
            future.result()
            outcomes.append("submitted")
        except Exception as exc:
            outcomes.append(type(exc).__name__)

    assert outcomes.count("submitted") == 1
    assert outcomes.count("EvidenceBundleError") == 1
    assert len(outcomes) == 2


def test_submitter_refuses_preexisting_script(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    script = run_dir / "job.sbatch"
    script.write_text("occupied\n", encoding="utf-8")

    with pytest.raises(EvidenceBundleError, match="overwrite"):
        SUBMITTER.publish_and_submit(
            script,
            b"#!/usr/bin/env bash\necho exact\n",
            submission_cwd=tmp_path,
            submitter=lambda submitted, cwd: _completed_submission(),
        )
