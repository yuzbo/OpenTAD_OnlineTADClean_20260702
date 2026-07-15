import importlib.util
import json
from pathlib import Path

import pytest

from opentad.utils.full_petal_launch import FullPetalLaunchError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_full_petal_launch_ticket.py"
SPEC = importlib.util.spec_from_file_location("build_full_petal_launch_ticket", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_launch_ticket_cli_exclusively_publishes_validated_payload(
    tmp_path, monkeypatch
):
    config = tmp_path / "config.py"
    config.write_text("value = 1\n", encoding="utf-8")
    output = tmp_path / "evidence" / "ticket.json"
    expected = {"schema_version": "synthetic-ticket", "mode": "profile"}

    def validated_builder(cfg, config_path, **kwargs):
        assert cfg.value == 1
        assert cfg.work_dir == str(output.parent.resolve() / "work")
        assert Path(config_path).resolve() == config.resolve()
        assert kwargs["mode"] == "profile"
        assert kwargs["entrypoint"] == "train"
        assert kwargs["cfg_overrides"] == {
            "work_dir": str(output.parent.resolve() / "work")
        }
        return expected

    monkeypatch.setattr(MODULE, "build_launch_ticket", validated_builder)
    argv = [
        str(config),
        "--mode",
        "profile",
        "--b0",
        str(tmp_path / "b0.json"),
        "--review",
        str(tmp_path / "review.json"),
        "--output",
        str(output),
        "--entrypoint",
        "train",
        "--seed",
        "705",
        "--id",
        "0",
    ]

    assert MODULE.main(argv) == 0
    assert json.loads(output.read_bytes()) == expected
    with pytest.raises(SystemExit):
        MODULE.main(argv)


def test_launch_ticket_cli_rejects_work_dir_override(tmp_path):
    config = tmp_path / "config.py"
    config.write_text("value = 1\n", encoding="utf-8")
    argv = [
        str(config),
        "--mode",
        "profile",
        "--b0",
        str(tmp_path / "b0.json"),
        "--review",
        str(tmp_path / "review.json"),
        "--output",
        str(tmp_path / "evidence" / "ticket.json"),
        "--entrypoint",
        "train",
        "--seed",
        "705",
        "--id",
        "0",
        "--cfg-options",
        "work_dir=/tmp/drifted",
    ]

    with pytest.raises(SystemExit):
        MODULE.main(argv)


def test_launch_ticket_cli_rejects_repository_local_evidence():
    with pytest.raises(FullPetalLaunchError, match="outside"):
        MODULE._external_output(ROOT / "ticket.json")
