from pathlib import Path

import pytest
import torch
from mmengine.config import Config

from opentad.cores.scheduler import build_scheduler


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "causaltad"
    / "thumos_persistent_binding_fixed_screen.py"
)
FIT_UPDATES = 2010


def _optimizer_update_lrs(warmup_epoch):
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=2e-4)
    scheduler, max_epoch = build_scheduler(
        dict(
            type="LinearWarmupCosineAnnealingLR",
            warmup_epoch=warmup_epoch,
            max_epoch=12,
        ),
        optimizer,
        FIT_UPDATES,
    )
    used_lrs = []
    for _ in range(FIT_UPDATES):
        used_lrs.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()
    return used_lrs, scheduler, max_epoch


def test_short_warmup_uses_201_steps_and_preserves_peak_lr():
    used_lrs, scheduler, max_epoch = _optimizer_update_lrs(0.1)

    assert scheduler.warmup_epoch == pytest.approx(201.0)
    assert scheduler.max_epoch == pytest.approx(24120.0)
    assert max_epoch == 12
    assert used_lrs[0] == pytest.approx(0.0)
    assert used_lrs[200] == pytest.approx(2e-4)
    assert max(used_lrs) == pytest.approx(2e-4)
    assert used_lrs[-1] < 2e-4


def test_registered_lr_exposure_matches_actual_optimizer_update_order():
    cfg = Config.fromfile(CONFIG)
    candidate_lrs, _, _ = _optimizer_update_lrs(
        cfg.scheduler.warmup_epoch
    )
    previous_lrs, _, _ = _optimizer_update_lrs(
        cfg.screening_contract.previous_warmup_epoch
    )
    candidate_sum = sum(candidate_lrs)
    previous_sum = sum(previous_lrs)
    ratio = candidate_sum / previous_sum

    assert previous_sum == pytest.approx(0.201)
    assert candidate_sum == pytest.approx(0.3802042142882046)
    assert ratio == pytest.approx(1.891563255165197)
    assert candidate_sum == pytest.approx(
        cfg.screening_contract.optimizer_update_lr_sum
    )
    assert previous_sum == pytest.approx(
        cfg.screening_contract.previous_optimizer_update_lr_sum
    )
    assert ratio == pytest.approx(
        cfg.screening_contract.mean_lr_exposure_ratio_vs_full_epoch_warmup
    )
