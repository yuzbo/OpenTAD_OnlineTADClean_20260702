import subprocess
import sys
from copy import deepcopy

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


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.frozen = nn.Linear(2, 1)
        for parameter in self.frozen.parameters():
            parameter.requires_grad = False


class FakeOptimizer:
    def __init__(self, groups):
        self.param_groups = groups


class HeadOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = None
        self.head = nn.Linear(2, 2)
        self.frozen_aux = nn.Linear(2, 1)
        for parameter in self.frozen_aux.parameters():
            parameter.requires_grad = False


def test_optimizer_audit_accepts_exact_trainable_coverage():
    from opentad.utils.optimizer_audit import audit_optimizer_coverage

    model = TinyModel()
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=1e-3)

    report = audit_optimizer_coverage(model, optimizer)

    assert report.passed
    assert report.missing_trainable == ()
    assert report.duplicate_parameters == ()
    assert report.frozen_in_optimizer == ()
    assert report.group_summaries[0]["num_parameters"] == 2


def test_optimizer_audit_fails_for_missing_trainable_parameter():
    from opentad.utils.optimizer_audit import OptimizerAuditError, audit_optimizer_coverage

    model = TinyModel()
    optimizer = torch.optim.AdamW([model.encoder.weight], lr=1e-3)

    report = audit_optimizer_coverage(model, optimizer)

    assert report.missing_trainable == ("encoder.bias",)
    with pytest.raises(OptimizerAuditError, match="encoder.bias"):
        report.raise_for_errors()


def test_optimizer_audit_detects_duplicate_groups_even_for_external_optimizer():
    from opentad.utils.optimizer_audit import audit_optimizer_coverage

    model = TinyModel()
    optimizer = FakeOptimizer(
        [
            {"params": [model.encoder.weight, model.encoder.bias], "lr": 1e-3},
            {"params": [model.encoder.weight], "lr": 1e-4},
        ]
    )

    report = audit_optimizer_coverage(model, optimizer)

    assert report.duplicate_parameters == ("encoder.weight",)
    assert not report.passed


def test_optimizer_audit_can_fail_on_frozen_parameters_in_groups():
    from opentad.utils.optimizer_audit import audit_optimizer_coverage

    model = TinyModel()
    optimizer = FakeOptimizer(
        [{"params": list(model.parameters()), "lr": 1e-3, "weight_decay": 0.01}]
    )

    warning_report = audit_optimizer_coverage(model, optimizer, fail_on_frozen=False)
    strict_report = audit_optimizer_coverage(model, optimizer, fail_on_frozen=True)

    assert warning_report.passed
    assert warning_report.frozen_in_optimizer == ("frozen.bias", "frozen.weight")
    assert not strict_report.passed


def test_optimizer_audit_enforces_declared_trainable_plan():
    from opentad.utils.optimizer_audit import audit_optimizer_coverage

    model = TinyModel()
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=1e-3)

    report = audit_optimizer_coverage(
        model,
        optimizer,
        expected_trainable=("encoder.weight", "encoder.bias", "frozen.weight"),
    )

    assert report.expected_trainable_not_enabled == ("frozen.weight",)
    assert not report.passed


def test_optimizer_audit_rejects_unknown_parameter_objects():
    from opentad.utils.optimizer_audit import audit_optimizer_coverage

    model = TinyModel()
    external = nn.Parameter(torch.ones(1))
    optimizer = FakeOptimizer(
        [{"params": [model.encoder.weight, model.encoder.bias, external], "lr": 1e-3}]
    )

    report = audit_optimizer_coverage(model, optimizer)

    assert report.unknown_optimizer_parameters == ("<unknown:0>",)
    assert not report.passed


def test_optimizer_builder_handles_none_backbone_without_mutating_config():
    from opentad.cores.optimizer import build_optimizer

    model = HeadOnlyModel()
    config = {
        "type": "AdamW",
        "lr": 1e-3,
        "weight_decay": 0.01,
        "audit": {"fail_on_frozen": True},
    }
    original = deepcopy(config)

    optimizer = build_optimizer(config, model, logger=None)

    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_ids == {id(parameter) for parameter in model.head.parameters()}
    assert config == original
