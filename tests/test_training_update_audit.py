import json
from pathlib import Path
import subprocess
import sys

import pytest


_TORCH_PROBE = subprocess.run(
    [sys.executable, "-c", "import torch"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
    timeout=20,
)
if _TORCH_PROBE.returncode != 0:
    pytest.skip("torch is unavailable in this environment", allow_module_level=True)

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


class TinyRoute(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 3)
        self.head = nn.Linear(3, 1)
        self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, inputs):
        return self.head(torch.tanh(self.backbone(inputs)))


def test_recursive_device_transfer_preserves_metadata_and_container_shapes():
    from opentad.utils.device import move_data_to_device

    batch = {
        "inputs": torch.ones(1, 2),
        "nested": [torch.zeros(1), {"mask": torch.ones(1, dtype=torch.bool)}],
        "metas": [{"video_id": "v1"}],
        "tuple_value": (torch.ones(1), "keep"),
    }

    moved = move_data_to_device(batch, torch.device("cpu"))

    assert moved["inputs"].device.type == "cpu"
    assert moved["nested"][1]["mask"].dtype == torch.bool
    assert moved["metas"] == [{"video_id": "v1"}]
    assert isinstance(moved["tuple_value"], tuple)
    assert moved["tuple_value"][1] == "keep"


def test_training_update_audit_proves_required_modules_receive_updates():
    from opentad.utils.training_audit import (
        audit_training_update,
        snapshot_trainable_parameters,
    )

    model = TinyRoute()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = snapshot_trainable_parameters(model)

    loss = model(torch.tensor([[0.5, -0.25]])).square().mean()
    loss.backward()
    optimizer.step()
    report = audit_training_update(
        model,
        before,
        required_module_prefixes=("backbone", "head"),
    )

    assert report.passed
    assert report.module_summaries["backbone"]["nonzero_grad_parameters"] > 0
    assert report.module_summaries["backbone"]["changed_parameters"] > 0
    assert report.module_summaries["head"]["nonzero_grad_parameters"] > 0
    assert "frozen" not in report.trainable_parameters
    json.dumps(report.as_dict())


def test_training_update_audit_fails_when_required_module_is_disconnected():
    from opentad.utils.training_audit import (
        TrainingUpdateAuditError,
        audit_training_update,
        snapshot_trainable_parameters,
    )

    model = TinyRoute()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = snapshot_trainable_parameters(model)

    loss = model.head(torch.ones(1, 3)).square().mean()
    loss.backward()
    optimizer.step()
    report = audit_training_update(
        model,
        before,
        required_module_prefixes=("backbone", "head"),
    )

    assert not report.passed
    assert report.missing_gradient_modules == ("backbone",)
    assert report.missing_update_modules == ("backbone",)
    with pytest.raises(TrainingUpdateAuditError, match="backbone"):
        report.raise_for_errors()


def test_training_engines_move_batches_to_the_model_device():
    train_source = (ROOT / "opentad" / "cores" / "train_engine.py").read_text(encoding="utf-8")
    test_source = (ROOT / "opentad" / "cores" / "test_engine.py").read_text(encoding="utf-8")

    assert "move_data_to_device" in train_source
    assert "move_data_to_device" in test_source
