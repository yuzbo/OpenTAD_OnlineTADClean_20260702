import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = __import__("pathlib", fromlist=["Path"]).Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_core_training_helpers_do_not_assume_ddp_module_wrapper():
    train_source = read("opentad/cores/train_engine.py")
    optimizer_source = read("opentad/cores/optimizer.py")
    layer_decay_source = read("opentad/cores/layer_decay_optimizer.py")
    misc_source = read("opentad/utils/misc.py")

    assert "hasattr(model.module" not in train_source
    assert "model.module.backbone" not in train_source
    assert "model.module" not in optimizer_source
    assert "model.module" not in layer_decay_source
    assert "dist.is_initialized()" in misc_source


def _torch_or_skip():
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("torch import timed out in a subprocess")
    if probe.returncode != 0:
        pytest.skip(f"torch import failed in a subprocess with exit code {probe.returncode}")
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"torch is unavailable in this environment: {exc}")
    return torch


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, *args, **kwargs):
        self.messages.append(("info", args, kwargs))

    def error(self, *args, **kwargs):
        self.messages.append(("error", args, kwargs))


def test_train_one_epoch_supports_unwrapped_single_process_model():
    torch = _torch_or_skip()

    from opentad.cores.train_engine import train_one_epoch

    class BareTrainModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = SimpleNamespace(freeze_backbone=False)
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.seen_epochs = []

        def set_train_epoch(self, epoch):
            self.seen_epochs.append(epoch)

        def forward(self, inputs, return_loss=False):
            assert return_loss is True
            loss = (self.weight - inputs).pow(2).mean()
            return {"cost": loss, "aux": loss.detach() + 0.0 * loss}

    model = BareTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    train_loader = [{"inputs": torch.tensor([1.0])}]

    train_one_epoch(
        train_loader,
        model,
        optimizer,
        scheduler,
        curr_epoch=3,
        logger=_Logger(),
        logging_interval=1,
    )

    assert model.seen_epochs == [3]
    assert model.weight.detach().item() != 0.0


def test_build_optimizer_supports_unwrapped_single_process_model():
    torch = _torch_or_skip()

    from opentad.cores.optimizer import build_optimizer

    class BareBackbone(torch.nn.Module):
        freeze_backbone = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    class BareDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = BareBackbone()
            self.head = torch.nn.Linear(1, 1)

    model = BareDetector()
    optimizer = build_optimizer(
        dict(
            type="AdamW",
            lr=1e-4,
            weight_decay=0.01,
            backbone=dict(lr=5e-5, weight_decay=0.02),
        ),
        model,
        _Logger(),
    )

    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}

    assert isinstance(optimizer, torch.optim.AdamW)
    assert id(model.backbone.weight) in optimizer_param_ids
    assert id(model.head.weight) in optimizer_param_ids
