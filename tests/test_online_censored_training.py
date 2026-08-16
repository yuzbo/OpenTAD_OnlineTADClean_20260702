import subprocess
import sys

import pytest


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


def _tiny_matr_head(**kwargs):
    import opentad.models  # noqa: F401
    from opentad.models.dense_heads.matr_head import MATRHead

    cfg = dict(
        num_classes=1,
        in_channels=4,
        feat_channels=4,
        num_convs=1,
        prior_generator=dict(type="PointGenerator", strides=[1], regression_range=[(0, 100)]),
        loss=dict(cls_loss=dict(type="FocalLoss"), reg_loss=dict(type="DIOULoss")),
        online=True,
        max_future_offset=0.0,
        online_censored_training=True,
    )
    cfg.update(kwargs)
    return MATRHead(**cfg)


def test_future_endpoint_regression_weight_is_zero_until_observed():
    torch = _torch_or_skip()
    head = _tiny_matr_head()

    points = [torch.tensor([[0.0, 0.0, 100.0, 1.0], [5.0, 0.0, 100.0, 1.0]])]
    pos_mask = torch.tensor([[True, True]])
    target_segments = torch.tensor([[0.0, 10.0], [0.0, 5.0]])

    weights = head._online_censored_regression_weights(points, pos_mask, target_segments)

    assert weights.tolist() == [0.0, 1.0]


def test_future_endpoint_is_not_positive_end_or_emit_target():
    torch = _torch_or_skip()
    head = _tiny_matr_head(boundary_target_radius=1.5)

    points = [torch.tensor([[9.0, 0.0, 100.0, 1.0], [10.0, 0.0, 100.0, 1.0]])]
    mask_list = [torch.ones(1, 2, dtype=torch.bool)]
    gt_segments = [torch.tensor([[0.0, 10.0]])]
    gt_labels = [torch.tensor([0])]

    _, end_target, _, emit_target, _ = head._build_online_branch_targets(
        points,
        mask_list,
        gt_segments,
        gt_labels,
    )

    assert end_target[0, 0, 0].item() == 0.0
    assert emit_target[0, 0, 0].item() == 0.0
    assert end_target[0, 1, 0].item() == 1.0
    assert emit_target[0, 1, 0].item() == 1.0


def test_cofie_field_is_causal_and_reads_the_earliest_same_instance_point():
    torch = _torch_or_skip()
    head = _tiny_matr_head(
        cofie_enabled=True,
        cofie_channels=2,
        cofie_loss_weight=1.0,
        cofie_threshold=0.5,
        memory_size=0,
    )

    points = [
        torch.tensor(
            [
                [0.0, 0.0, 100.0, 1.0],
                [1.0, 0.0, 100.0, 1.0],
                [2.0, 0.0, 100.0, 1.0],
                [3.0, 0.0, 100.0, 1.0],
            ]
        )
    ]
    mask_list = [torch.ones(1, 4, dtype=torch.bool)]
    gt_segments = [torch.tensor([[1.0, 3.0]])]

    targets, valid_pairs = head._build_cofie_targets(points, mask_list, gt_segments)
    assert targets[0][0, 3, 1].item() == 1.0
    assert targets[0][0, 1, 3].item() == 0.0
    assert valid_pairs[0][0, 1, 3].item() is False

    logits = torch.full((1, 4, 4), -10.0, requires_grad=True)
    with torch.no_grad():
        logits[0, 3, 1] = 10.0
    loss = head._cofie_loss(points, mask_list, gt_segments, [logits])["cofie_loss"]
    assert torch.isfinite(loss)
    loss.backward()

    starts, has_support = head._cofie_start_coordinates(points, mask_list, [logits.detach()])
    assert has_support[0, 3].item() is True
    assert starts[0, 3].item() == 1.0
