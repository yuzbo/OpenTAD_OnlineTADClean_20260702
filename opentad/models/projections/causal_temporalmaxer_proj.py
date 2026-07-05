import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bricks import ConvModule
from ..builder import PROJECTIONS


class CausalConvModule(ConvModule):
    """ConvModule with left-only temporal padding."""

    def __init__(self, *args, kernel_size=1, padding=0, **kwargs):
        if padding not in (0, None):
            raise ValueError("CausalConvModule pads on the left; pass padding=0.")
        if not isinstance(kernel_size, int):
            raise TypeError("CausalConvModule expects an integer temporal kernel_size.")
        self.left_pad = kernel_size - 1
        super().__init__(*args, kernel_size=kernel_size, padding=0, **kwargs)

    def forward(self, x, mask=None):
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        return super().forward(x, mask)


@PROJECTIONS.register_module()
class CausalTemporalMaxerProj(nn.Module):
    """Portable causal temporal pyramid for online raw-frame baselines.

    This module intentionally avoids Mamba and full self-attention.  It is a
    strict no-future projection: every convolution and downsampling level uses
    left-only padding, so output index t never consumes input indices > t.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 0, 5),
        conv_cfg=None,
        norm_cfg=None,
        drop_out=0.0,
        strict_causal=True,
    ):
        super().__init__()
        if len(arch) != 3:
            raise ValueError("arch must be (#convs, #stem, #branch).")
        if arch[1] != 0:
            raise ValueError("CausalTemporalMaxerProj has no attention stem; set arch[1] to 0.")
        if not strict_causal:
            raise ValueError("CausalTemporalMaxerProj is always strict causal.")
        if conv_cfg is None or "kernel_size" not in conv_cfg:
            raise ValueError("conv_cfg with kernel_size is required.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.scale_factor = 2
        self.with_norm = norm_cfg is not None
        self.strict_causal = True

        self.drop_out = nn.Dropout1d(p=drop_out) if drop_out > 0 else None

        if isinstance(self.in_channels, (list, tuple)):
            if not isinstance(self.out_channels, (list, tuple)) or len(self.in_channels) != len(self.out_channels):
                raise ValueError("List input channels require matching list output channels.")
            self.proj = nn.ModuleList()
            for n_in, n_out in zip(self.in_channels, self.out_channels):
                self.proj.append(
                    CausalConvModule(
                        n_in,
                        n_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            in_channels = out_channels = sum(self.out_channels)
        else:
            self.proj = None

        self.embed = nn.ModuleList()
        for i in range(arch[0]):
            self.embed.append(
                CausalConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="relu"),
                )
            )

        self.branch = nn.ModuleList()
        for _ in range(arch[2]):
            self.branch.append(CausalTemporalMaxer(kernel_size=3, stride=self.scale_factor))

    def forward(self, x, mask):
        if self.drop_out is not None:
            x = self.drop_out(x)

        if self.proj is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))], dim=1)

        for embed in self.embed:
            x, mask = embed(x, mask)

        out_feats = (x,)
        out_masks = (mask,)

        for branch in self.branch:
            x, mask = branch(x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks


class CausalTemporalMaxer(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        if kernel_size < 1 or stride < 1:
            raise ValueError("kernel_size and stride must be positive.")
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.left_pad = self.kernel_size - 1

    def forward(self, x, mask):
        mask_float = mask.float().unsqueeze(1)
        padded_mask = F.pad(mask_float, (self.left_pad, 0))
        out_mask = F.max_pool1d(padded_mask, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        out_mask = out_mask.squeeze(1).bool()

        if torch.is_floating_point(x):
            pad_value = torch.finfo(x.dtype).min
        else:
            pad_value = 0
        x = x.masked_fill(~mask.unsqueeze(1), pad_value)
        padded_x = F.pad(x, (self.left_pad, 0), value=pad_value)
        out = F.max_pool1d(padded_x, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        out = out.masked_fill(~out_mask.unsqueeze(1), 0.0)

        return out, out_mask
