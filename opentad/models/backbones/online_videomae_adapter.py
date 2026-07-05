import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS


class CausalTemporalAdapter(nn.Module):
    """Lightweight causal temporal adapter for frame-level VideoMAE features."""

    def __init__(self, channels, adapter_channels=None, kernel_size=3, init_scale=1.0):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")

        adapter_channels = adapter_channels or max(channels // 4, 1)
        self.is_causal = True
        self.left_pad = kernel_size - 1
        self.down_proj = nn.Conv1d(channels, adapter_channels, kernel_size=1)
        self.dwconv = nn.Conv1d(
            adapter_channels,
            adapter_channels,
            kernel_size=kernel_size,
            groups=adapter_channels,
            padding=0,
        )
        self.mix = nn.Conv1d(adapter_channels, adapter_channels, kernel_size=1)
        self.up_proj = nn.Conv1d(adapter_channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, mask=None):
        residual = x
        x = self.down_proj(x)
        x = self.act(x)
        x = F.pad(x, (self.left_pad, 0))
        x = self.dwconv(x)
        x = self.mix(x)
        x = self.act(x)
        x = self.up_proj(x)
        x = residual + self.scale.to(dtype=x.dtype) * x
        if mask is not None:
            x = x * mask.unsqueeze(1).to(dtype=x.dtype)
        return x


class _StubFrameBackbone(nn.Module):
    """Tiny frame encoder used for local contracts when VideoMAE is unavailable."""

    def __init__(self, in_channels, embed_dims):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dims, kernel_size=1)

    def forward(self, frames, masks=None, metas=None):
        del masks, metas
        # frames: [B, 1, C, T, H, W] or [B, C, T, H, W]
        if frames.dim() == 6:
            frames = frames.mean(dim=1)
        if frames.dim() != 5:
            raise ValueError(f"Expected frames with 5 or 6 dims, got {tuple(frames.shape)}")
        x = frames.mean(dim=(-1, -2))
        return self.proj(x)


@MODELS.register_module()
class OnlineVideoMAEAdapter(nn.Module):
    """Frame-level online VideoMAE adapter front-end.

    The production path can wrap a real VideoMAE-style backbone through
    ``backbone``. The local stub path preserves the same [B, C, T] contract
    without downloading weights or requiring CUDA.
    """

    input_format = "raw_frames"
    online = True

    def __init__(
        self,
        in_channels=3,
        embed_dims=768,
        adapter_channels=None,
        out_channels=None,
        causal_kernel_size=3,
        input_format="raw_frames",
        online=True,
        strict_online=True,
        freeze_backbone=True,
        use_stub_backbone=False,
        stub_backbone_contract_only=False,
        backbone=None,
        norm_cfg=None,
        output_layout="bct",
        return_masks=False,
    ):
        super().__init__()
        if input_format != "raw_frames":
            raise ValueError("OnlineVideoMAEAdapter only accepts input_format='raw_frames'")
        if strict_online and backbone is not None:
            backbone_is_online = bool(
                backbone.get("online", False)
                or backbone.get("strict_online", False)
                or backbone.get("causal_attention", False)
            )
            if not backbone_is_online:
                raise ValueError("strict_online=True requires a backbone that declares online/causal attention.")
        self.input_format = input_format
        self.online = bool(online)
        self.strict_online = bool(strict_online)
        self.freeze_backbone = bool(freeze_backbone)
        self.use_stub_backbone = bool(use_stub_backbone)
        self.stub_backbone_contract_only = bool(stub_backbone_contract_only)
        self.out_channels = out_channels or embed_dims
        self.output_layout = output_layout.lower()
        self.return_masks = bool(return_masks)
        if self.output_layout not in {"bct", "btc"}:
            raise ValueError("output_layout must be either 'bct' or 'btc'")

        if self.use_stub_backbone:
            if not self.stub_backbone_contract_only:
                raise ValueError(
                    "use_stub_backbone=True is only allowed for explicit contract tests; "
                    "set stub_backbone_contract_only=True or provide a real causal VideoMAE backbone."
                )
            self.stub_backbone = _StubFrameBackbone(in_channels, embed_dims)
            self.backbone = self.stub_backbone
        elif backbone is not None:
            self.backbone = MODELS.build(backbone)
        else:
            raise ValueError(
                "OnlineVideoMAEAdapter requires a real backbone config unless "
                "use_stub_backbone=True and stub_backbone_contract_only=True"
            )

        self.adapter = CausalTemporalAdapter(
            channels=embed_dims,
            adapter_channels=adapter_channels,
            kernel_size=causal_kernel_size,
        )
        self.out_proj = nn.Conv1d(embed_dims, self.out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(self.out_channels, eps=1e-6) if norm_cfg is not None else None
        self._configure_trainability()

    def _configure_trainability(self):
        if not self.freeze_backbone:
            return
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        for param in self.adapter.parameters():
            param.requires_grad = True
        for param in self.out_proj.parameters():
            param.requires_grad = True
        if self.norm is not None:
            for param in self.norm.parameters():
                param.requires_grad = True

    def get_optim_groups(self, cfg):
        lr = cfg.get("lr", 1e-4)
        weight_decay = cfg.get("weight_decay", 0.05)
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("bias") or name.endswith("scale") or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        optim_groups = []
        if decay:
            optim_groups.append(dict(params=decay, lr=lr, weight_decay=weight_decay))
        if no_decay:
            optim_groups.append(dict(params=no_decay, lr=lr, weight_decay=0.0))
        return optim_groups

    @staticmethod
    def _unpack_backbone_output(output, masks):
        updated_masks = masks
        x = output
        if isinstance(output, dict):
            x = output.get("features", output.get("feats", output.get("last_hidden_state")))
            updated_masks = output.get("masks", masks)
        elif isinstance(output, (tuple, list)):
            if len(output) >= 2 and torch.is_tensor(output[1]) and output[1].dim() <= 2:
                x = output[0]
                updated_masks = output[1]
            else:
                x = output[-1]
        if x is None:
            raise ValueError("Backbone output did not contain feature tensors")
        return x, updated_masks

    def _call_frame_backbone(self, frames, masks=None, metas=None):
        try:
            signature = inspect.signature(self.backbone.forward)
        except (TypeError, ValueError):
            return self.backbone(frames)

        params = signature.parameters
        accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in params.values())
        if accepts_kwargs or ("masks" in params and "metas" in params):
            return self.backbone(frames, masks=masks, metas=metas)
        if "masks" in params:
            return self.backbone(frames, masks=masks)
        return self.backbone(frames)

    def forward(self, frames, masks=None, metas=None):
        if self.freeze_backbone:
            with torch.no_grad():
                output = self._call_frame_backbone(frames, masks=masks, metas=metas)
        else:
            output = self._call_frame_backbone(frames, masks=masks, metas=metas)

        x, masks = self._unpack_backbone_output(output, masks)
        if x.dim() == 5:
            x = x.mean(dim=(-1, -2))
        if x.dim() == 3 and self.output_layout == "btc":
            x = x.transpose(1, 2)
        if x.dim() != 3:
            raise ValueError(f"Backbone output must be [B,C,T] or [B,T,C], got {tuple(x.shape)}")

        if masks is not None and masks.shape[-1] != x.shape[-1]:
            masks = F.interpolate(masks.unsqueeze(1).float(), size=x.shape[-1], mode="nearest").squeeze(1).bool()

        x = self.adapter(x, masks)
        x = self.out_proj(x)
        if self.norm is not None:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        if masks is not None:
            x = x * masks.unsqueeze(1).to(dtype=x.dtype)
        if self.return_masks:
            return x.to(torch.float32), masks
        return x.to(torch.float32)
