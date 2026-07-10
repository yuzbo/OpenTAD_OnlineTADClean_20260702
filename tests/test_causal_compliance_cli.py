import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "check_causal_compliance.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(CLI), *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_packet_cli_returns_machine_readable_pass_report(tmp_path):
    path = tmp_path / "packets.json"
    path.write_text(
        json.dumps(
            [
                {
                    "video_id": "v1",
                    "packet_start_frame": 0,
                    "packet_end_frame": 8,
                    "is_video_start": True,
                    "is_video_end": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = _run("packets", path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["passed"] is True


def test_recorded_future_perturbation_cli_fails_at_first_changed_prefix(tmp_path):
    reference = tmp_path / "reference.json"
    perturbed = tmp_path / "perturbed.json"
    reference.write_text(json.dumps([{"time": 0, "score": 0.1}]), encoding="utf-8")
    perturbed.write_text(json.dumps([{"time": 0, "score": 0.2}]), encoding="utf-8")

    result = _run("future-perturbation", reference, perturbed, "--cut", "0")

    assert result.returncode == 2
    assert "future perturbation" in result.stderr
    assert "time=0" in result.stderr
