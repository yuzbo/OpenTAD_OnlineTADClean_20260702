import json
import subprocess
import sys
from pathlib import Path

import numpy as np
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


def _write_tiny_raw_frame_dataset(tmp_path):
    ann_file = tmp_path / "ann.json"
    class_map = tmp_path / "class_map.txt"
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()

    ann_file.write_text(
        json.dumps(
            {
                "database": {
                    "video_test": {
                        "subset": "training",
                        "duration": 8.0,
                        "frame": 8,
                        "annotations": [{"segment": [2.0, 5.0], "label": "Action"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    class_map.write_text("Action\n", encoding="utf-8")
    frames = np.arange(8 * 8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 8, 3)
    np.save(frame_dir / "video_test.npy", frames)
    return ann_file, class_map, frame_dir


def test_build_raw_frame_dataset_from_synthetic_frames(tmp_path):
    torch = _torch_or_skip()

    try:
        import opentad.datasets  # noqa: F401
        from opentad.datasets.builder import build_dataset
    except Exception as exc:
        pytest.fail(f"OpenTAD dataset imports failed: {exc}")

    ann_file, class_map, frame_dir = _write_tiny_raw_frame_dataset(tmp_path)
    dataset = build_dataset(
        dict(
            type="FrameWindowDataset",
            input_format="raw_frames",
            online=True,
            ann_file=str(ann_file),
            subset_name="training",
            class_map=str(class_map),
            data_path=str(frame_dir),
            filter_gt=False,
            feature_stride=1,
            sample_stride=1,
            offset_frames=0,
            window_size=8,
            window_overlap_ratio=0.0,
            ioa_thresh=0.0,
            pipeline=[
                dict(type="LoadFrames", num_clips=1, scale_factor=1, method="sliding_window"),
                dict(type="LoadRawFrames", frame_format="npy", frame_size=None),
                dict(type="ConvertToTensor", keys=["gt_segments", "gt_labels"]),
                dict(type="Collect", inputs="frames", keys=["masks", "gt_segments", "gt_labels"]),
            ],
        )
    )

    sample = dataset[0]
    assert sample["inputs"].shape == (1, 3, 8, 8, 8)
    assert sample["inputs"].dtype == torch.float32
    assert sample["masks"].shape == (8,)
    assert sample["masks"].all()
    assert sample["gt_segments"].shape == (1, 2)
    assert sample["gt_labels"].tolist() == [0]
    assert sample["metas"]["video_name"] == "video_test"
    assert sample["metas"]["window_start_frame"] == 0


def _tiny_model_cfg():
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
            use_stub_backbone=True,
            stub_backbone_contract_only=True,
            output_layout="bct",
            norm_cfg=dict(type="LN"),
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


def test_build_model_synthetic_raw_frames_forward_backward():
    torch = _torch_or_skip()

    try:
        import opentad.models  # noqa: F401
        from opentad.models import build_detector
    except Exception as exc:
        pytest.fail(f"OpenTAD model imports failed: {exc}")

    model = build_detector(_tiny_model_cfg())
    model.train()

    inputs = torch.randn(2, 1, 3, 8, 8, 8)
    masks = torch.ones(2, 8, dtype=torch.bool)
    metas = [
        {"video_name": "video_a", "window_start_frame": 0},
        {"video_name": "video_b", "window_start_frame": 0},
    ]
    gt_segments = [torch.tensor([[1.0, 4.0]]), torch.tensor([[2.0, 6.0]])]
    gt_labels = [torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long)]

    losses = model(inputs, masks, metas, gt_segments=gt_segments, gt_labels=gt_labels, return_loss=True)
    assert "cost" in losses
    assert torch.isfinite(losses["cost"])
    losses["cost"].backward()

    trainable_grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in trainable_grads)


def test_optimizer_param_groups_include_frozen_backbone_adapters():
    torch = _torch_or_skip()

    try:
        import opentad.models  # noqa: F401
        from opentad.cores.optimizer import build_optimizer
        from opentad.models import build_detector
    except Exception as exc:
        pytest.fail(f"OpenTAD model imports failed: {exc}")

    class _Wrapped:
        def __init__(self, module):
            self.module = module

    class _Logger:
        def info(self, message):
            self.last_message = message

    model = build_detector(_tiny_model_cfg())
    optimizer = build_optimizer(
        dict(
            type="AdamW",
            lr=1e-4,
            weight_decay=0.05,
            paramwise=True,
            backbone=dict(lr=5e-5, weight_decay=0.05),
        ),
        _Wrapped(model),
        _Logger(),
    )

    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    trainable_backbone_ids = {
        id(param)
        for name, param in model.backbone.named_parameters()
        if param.requires_grad and (name.startswith("adapter.") or name.startswith("out_proj.") or name.startswith("norm."))
    }
    frozen_stub_ids = {
        id(param)
        for name, param in model.backbone.named_parameters()
        if name.startswith("stub_backbone.") and not param.requires_grad
    }

    assert trainable_backbone_ids
    assert trainable_backbone_ids <= optimizer_param_ids
    assert frozen_stub_ids.isdisjoint(optimizer_param_ids)
    assert isinstance(optimizer, torch.optim.AdamW)
