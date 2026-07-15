import importlib.util
import json
from pathlib import Path

from mmengine import Config
import pytest

from opentad.utils.evidence_bundle import EvidenceBundleError
from opentad.utils.full_petal_launch import resolved_config_sha256


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "export_full_petal_resolved_config.py"
SPEC = importlib.util.spec_from_file_location(
    "export_full_petal_resolved_config", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_exported_snapshot_resolves_inheritance_without_base_file(tmp_path):
    base = tmp_path / "base.py"
    base.write_text(
        "model = dict(type='PersistentTrajectoryOnlineDetector', hidden_dim=64)\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.py"
    child.write_text(
        "_base_ = ['./base.py']\nmodel = dict(hidden_dim=128)\nwork_dir = 'run-a'\n",
        encoding="utf-8",
    )
    output = tmp_path / "evidence" / "resolved.json"

    result = MODULE.export_resolved_config(child, output)
    raw = output.read_bytes()
    payload = json.loads(raw)
    snapshot = Config(payload)
    source = Config.fromfile(str(child))

    assert raw == (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    assert snapshot.model.hidden_dim == 128
    assert resolved_config_sha256(snapshot) == resolved_config_sha256(source)
    assert result["resolved_config_sha256"] == resolved_config_sha256(source)

    with pytest.raises(EvidenceBundleError, match="refusing to overwrite"):
        MODULE.export_resolved_config(child, output)
    base.unlink()
    assert Config(json.loads(output.read_bytes())).model.hidden_dim == 128


def test_resolved_snapshot_must_stay_outside_repository():
    with pytest.raises(EvidenceBundleError, match="outside"):
        MODULE._outside_repository(ROOT / "resolved.json")
