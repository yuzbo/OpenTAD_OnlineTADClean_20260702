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


def test_causal_temporalmaxer_proj_does_not_change_past_outputs_when_future_changes():
    torch = _torch_or_skip()
    from opentad.models.projections.causal_temporalmaxer_proj import CausalTemporalMaxerProj

    torch.manual_seed(7)
    module = CausalTemporalMaxerProj(
        in_channels=3,
        out_channels=5,
        arch=(2, 0, 5),
        conv_cfg=dict(kernel_size=3),
        norm_cfg=dict(type="LN"),
        drop_out=0.0,
        strict_causal=True,
    )
    module.eval()

    length = 17
    cutoff = 8
    x = torch.randn(2, 3, length)
    mask = torch.ones(2, length, dtype=torch.bool)

    future_changed = x.clone()
    future_changed[..., cutoff + 1 :] = torch.randn_like(future_changed[..., cutoff + 1 :]) * 100.0

    with torch.no_grad():
        base_feats, base_masks = module(x, mask)
        changed_feats, changed_masks = module(future_changed, mask)

    for level, (base_feat, changed_feat, base_mask, changed_mask) in enumerate(
        zip(base_feats, changed_feats, base_masks, changed_masks)
    ):
        causal_stride = 2**level
        safe_length = min(base_feat.shape[-1], cutoff // causal_stride + 1)
        assert safe_length > 0
        assert torch.equal(base_mask[..., :safe_length], changed_mask[..., :safe_length])
        torch.testing.assert_close(
            base_feat[..., :safe_length],
            changed_feat[..., :safe_length],
            rtol=1e-5,
            atol=1e-5,
        )


def test_causal_temporalmaxer_proj_rejects_noncausal_stem_or_flag():
    _torch_or_skip()
    from opentad.models.projections.causal_temporalmaxer_proj import CausalTemporalMaxerProj

    try:
        CausalTemporalMaxerProj(
            in_channels=3,
            out_channels=5,
            arch=(2, 1, 5),
            conv_cfg=dict(kernel_size=3),
            strict_causal=True,
        )
    except ValueError as exc:
        assert "no attention stem" in str(exc)
    else:
        raise AssertionError("Expected nonzero stem arch to be rejected.")

    try:
        CausalTemporalMaxerProj(
            in_channels=3,
            out_channels=5,
            arch=(2, 0, 5),
            conv_cfg=dict(kernel_size=3),
            strict_causal=False,
        )
    except ValueError as exc:
        assert "strict causal" in str(exc)
    else:
        raise AssertionError("Expected strict_causal=False to be rejected.")
