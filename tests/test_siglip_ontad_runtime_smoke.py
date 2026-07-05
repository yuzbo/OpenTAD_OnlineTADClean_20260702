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


def _tiny_siglip_model_cfg():
    return dict(
        type="VideoMambaSuite",
        backbone=dict(
            type="OnlineVideoMAEAdapter",
            input_format="raw_frames",
            online=True,
            strict_online=True,
            in_channels=3,
            embed_dims=8,
            adapter_channels=4,
            out_channels=8,
            causal_kernel_size=3,
            freeze_backbone=True,
            use_stub_backbone=False,
            output_layout="bct",
            return_masks=True,
            norm_cfg=dict(type="LN"),
            backbone=dict(
                type="OnlineSigLIPFrameEncoder",
                model_name=None,
                backend="tiny",
                embed_dims=8,
                image_size=8,
                freeze_vision_encoder=True,
                strict_online=True,
                frame_chunk_size=4,
                use_motion_branch=True,
                motion_branch=dict(channels=8, kernel_size=3, causal=True),
            ),
        ),
        projection=dict(
            type="TemporalMaxerProj",
            in_channels=8,
            out_channels=8,
            arch=(1, 0, 0),
            conv_cfg=dict(kernel_size=1),
            norm_cfg=None,
            drop_out=0.0,
        ),
        neck=dict(type="FPNIdentity", in_channels=8, out_channels=8, num_levels=1, norm_cfg=None),
        rpn_head=dict(
            type="MATRHead",
            online=True,
            num_classes=2,
            in_channels=8,
            feat_channels=8,
            num_convs=1,
            kernel_size=3,
            emit_threshold=0.0,
            memory_size=0,
            clamp_end_to_current=True,
            max_future_offset=0.0,
            boundary_loss_weight=0.1,
            actionness_loss_weight=0.1,
            emit_loss_weight=0.1,
            cls_prior_prob=0.01,
            prior_generator=dict(type="PointGenerator", strides=[1], regression_range=[(0, 32)]),
            loss_normalizer=8,
            loss_normalizer_momentum=0.9,
            center_sample="radius",
            center_sample_radius=1.5,
            label_smoothing=0.0,
            loss=dict(cls_loss=dict(type="FocalLoss"), reg_loss=dict(type="DIOULoss")),
        ),
    )


def test_tiny_siglip_encoder_is_shape_stable_and_no_future():
    torch = _torch_or_skip()

    try:
        import opentad.models  # noqa: F401
        from mmengine.registry import MODELS
    except Exception as exc:
        pytest.fail(f"OpenTAD model imports failed: {exc}")

    encoder = MODELS.build(
        dict(
            type="OnlineSigLIPFrameEncoder",
            model_name=None,
            backend="tiny",
            embed_dims=8,
            image_size=8,
            freeze_vision_encoder=True,
            strict_online=True,
            frame_chunk_size=3,
            use_motion_branch=True,
            motion_branch=dict(channels=8, kernel_size=3, causal=True),
        )
    )
    encoder.eval()

    frames = torch.randn(2, 1, 3, 6, 8, 8)
    masks = torch.ones(2, 6, dtype=torch.bool)
    with torch.no_grad():
        base, out_masks = encoder(frames, masks=masks, metas=[{"video_name": "v"}] * 2)
        perturbed = frames.clone()
        perturbed[:, :, :, 4:] = perturbed[:, :, :, 4:] + 100.0
        changed, _ = encoder(perturbed, masks=masks, metas=[{"video_name": "v"}] * 2)

    assert base.shape == (2, 8, 6)
    assert out_masks.shape == (2, 6)
    assert torch.equal(out_masks, masks)
    assert torch.allclose(base[..., :4], changed[..., :4], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(base[..., 4:], changed[..., 4:], atol=1e-5, rtol=1e-5)


def test_tiny_siglip_detector_forward_backward_and_frozen_encoder_grads():
    torch = _torch_or_skip()

    try:
        import opentad.models  # noqa: F401
        from opentad.models import build_detector
    except Exception as exc:
        pytest.fail(f"OpenTAD model imports failed: {exc}")

    model = build_detector(_tiny_siglip_model_cfg())
    model.train()

    inputs = torch.randn(2, 1, 3, 8, 8, 8)
    masks = torch.ones(2, 8, dtype=torch.bool)
    metas = [
        {"video_name": "video_a", "window_start_frame": 0, "snippet_stride": 1, "fps": 1.0},
        {"video_name": "video_b", "window_start_frame": 0, "snippet_stride": 1, "fps": 1.0},
    ]
    gt_segments = [torch.tensor([[1.0, 4.0]]), torch.tensor([[2.0, 6.0]])]
    gt_labels = [torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long)]

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-3)
    losses_seen = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        losses = model(inputs, masks, metas, gt_segments=gt_segments, gt_labels=gt_labels, return_loss=True)
        assert "cost" in losses
        assert torch.isfinite(losses["cost"])
        losses_seen.append(float(losses["cost"].detach().cpu()))
        losses["cost"].backward()
        optimizer.step()

    frozen_encoder_grads = [
        p.grad
        for name, p in model.backbone.backbone.named_parameters()
        if name.startswith("vision_encoder.") and not p.requires_grad
    ]
    trainable_grads = [
        p.grad
        for name, p in model.named_parameters()
        if p.requires_grad and ("adapter." in name or "out_proj." in name or "projection." in name or "rpn_head." in name)
    ]

    assert all(grad is None for grad in frozen_encoder_grads)
    assert any(grad is not None and torch.isfinite(grad).all() for grad in trainable_grads)
    assert min(losses_seen) <= losses_seen[0]


def test_frame_grid_seconds_roundtrip_runtime():
    torch = _torch_or_skip()

    try:
        from opentad.models.utils.post_processing.utils import grid_to_seconds, seconds_to_grid
    except Exception as exc:
        pytest.fail(f"OpenTAD post-processing imports failed: {exc}")

    meta = dict(
        fps=30.0,
        snippet_stride=8,
        offset_frames=0,
        window_start_frame=16,
        duration=100.0,
    )
    grid = torch.tensor([[0.0, 4.0], [2.5, 7.25]])
    seconds = grid_to_seconds(grid, meta)
    roundtrip = seconds_to_grid(seconds, meta)

    assert torch.allclose(seconds[0], torch.tensor([16.0 / 30.0, 48.0 / 30.0]))
    assert torch.allclose(roundtrip, grid, atol=1e-5)
