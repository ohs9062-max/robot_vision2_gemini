# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""CSPNeXt Backbone for RTMDet.
Reference: RTMDet (Lyu et al., arXiv:2212.07784)
Pure PyTorch implementation without mmcv/mmdet dependencies.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class ConvBNAct(nn.Module):
    """Standard Convolution + BatchNorm + SiLU activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution with 5x5 kernel as specified in CSPNeXt."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.dw_act = nn.SiLU(inplace=True)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.pw_act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_act(self.dw_bn(self.dw(x)))
        x = self.pw_act(self.pw_bn(self.pw(x)))
        return x


class CSPNeXtBlock(nn.Module):
    """CSPNeXt basic bottleneck block."""
    def __init__(self, in_channels: int, out_channels: int, expansion: float = 0.5, kernel_size: int = 5, residual: bool = True):
        super().__init__()
        mid_channels = int(out_channels * expansion)
        self.conv1 = ConvBNAct(in_channels, mid_channels, 3, 1, 1)
        self.conv2 = DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=kernel_size)
        self.has_residual = residual and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.conv1(x))
        return x + res if self.has_residual else res


class CSPNeXtStage(nn.Module):
    """CSPNeXt Stage combining downsampling conv and cross-stage partial blocks."""
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, expansion: float = 0.5):
        super().__init__()
        mid_channels = int(out_channels * expansion)
        self.main_conv = ConvBNAct(in_channels, mid_channels, 1, 1, 0)
        self.short_conv = ConvBNAct(in_channels, mid_channels, 1, 1, 0)
        self.blocks = nn.Sequential(*[
            CSPNeXtBlock(mid_channels, mid_channels, expansion=1.0, kernel_size=5, residual=True)
            for _ in range(num_blocks)
        ])
        self.final_conv = ConvBNAct(2 * mid_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_main = self.blocks(self.main_conv(x))
        x_short = self.short_conv(x)
        return self.final_conv(torch.cat([x_main, x_short], dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        mid_channels = in_channels // 2
        self.cv1 = ConvBNAct(in_channels, mid_channels, 1, 1, 0)
        self.cv2 = ConvBNAct(mid_channels * 4, out_channels, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class CSPNeXt(nn.Module):
    """CSPNeXt backbone for RTMDet-s.
    Width multiplier = 0.5, Depth multiplier = 0.33
    Outputs 3 feature maps: P3 (stride 8), P4 (stride 16), P5 (stride 32)
    """
    def __init__(self, width: float = 0.5, depth: float = 0.33):
        super().__init__()
        base_channels = [64, 128, 256, 512, 1024]
        channels = [max(round(c * width), 16) for c in base_channels]
        base_blocks = [1, 2, 2, 1]
        num_blocks = [max(round(b * depth), 1) for b in base_blocks]

        # Stem (downsample 4x)
        self.stem = nn.Sequential(
            ConvBNAct(3, channels[0] // 2, 3, 2, 1),
            ConvBNAct(channels[0] // 2, channels[0] // 2, 3, 1, 1),
            ConvBNAct(channels[0] // 2, channels[0], 3, 1, 1),
        )

        # Stage 1: stride 4 -> stride 4
        self.stage1 = nn.Sequential(
            ConvBNAct(channels[0], channels[1], 3, 2, 1),
            CSPNeXtStage(channels[1], channels[1], num_blocks[0]),
        )

        # Stage 2 (P3): stride 8
        self.stage2 = nn.Sequential(
            ConvBNAct(channels[1], channels[2], 3, 2, 1),
            CSPNeXtStage(channels[2], channels[2], num_blocks[1]),
        )

        # Stage 3 (P4): stride 16
        self.stage3 = nn.Sequential(
            ConvBNAct(channels[2], channels[3], 3, 2, 1),
            CSPNeXtStage(channels[3], channels[3], num_blocks[2]),
        )

        # Stage 4 (P5): stride 32 + SPPF
        self.stage4 = nn.Sequential(
            ConvBNAct(channels[3], channels[4], 3, 2, 1),
            SPPF(channels[4], channels[4]),
            CSPNeXtStage(channels[4], channels[4], num_blocks[3]),
        )

        self.out_channels = [channels[2], channels[3], channels[4]]  # P3, P4, P5

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)   # stride 8
        c4 = self.stage3(c3)  # stride 16
        c5 = self.stage4(c4)  # stride 32
        return c3, c4, c5
