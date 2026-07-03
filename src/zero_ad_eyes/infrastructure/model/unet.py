"""U-Net architecture for the learned segmentation adapter (MP4).

The delivered ``best.pt`` is a bare PyTorch ``state_dict`` (no architecture, no
metadata), so the module tree here must reproduce the training model *exactly* —
the parameter names below were reverse-engineered from the checkpoint keys:

    inc.block.{0,1,3,4}          DoubleConv  (Conv-BN-ReLU ×2, bias-free convs)
    down{1..4}.block.{0,1}       MaxPool → DoubleConv
    up{1..4}.upsample            ConvTranspose2d (weight+bias)
    up{1..4}.conv.block.*        DoubleConv over cat(skip, upsampled)
    outc.{weight,bias}           1×1 Conv2d → n_classes logits

``base_channels`` (32) and ``num_classes`` (17) come from the checkpoint scalars.
Torch is imported lazily by the adapter; importing this module still needs torch,
so keep it out of any headless import path.
"""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """(Conv3×3 → BN → ReLU) twice; convs are bias-free (BN carries the shift)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscale by max-pool, then a double convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Learned upscale (transpose conv), concat the skip, then a double conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After upsample (in/2) concatenated with the same-width skip → in channels in.
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Guard against off-by-one from odd input dims (mirrors the reference U-Net).
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(
            x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """The segmentation network whose weights ``best.pt`` carries."""

    def __init__(self, n_channels: int = 3, n_classes: int = 17, base_channels: int = 32) -> None:
        super().__init__()
        b = base_channels
        self.inc = DoubleConv(n_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 16)
        self.up1 = Up(b * 16, b * 8)
        self.up2 = Up(b * 8, b * 4)
        self.up3 = Up(b * 4, b * 2)
        self.up4 = Up(b * 2, b)
        self.outc = nn.Conv2d(b, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
