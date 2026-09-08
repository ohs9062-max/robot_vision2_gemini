# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""CSPNeXt-PAFPN Neck for RTMDet.
Reference: RTMDet (Lyu et al., arXiv:2212.07784)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .cspnext import ConvBNAct, CSPNeXtBlock


class CSPNeXtPAFPN(nn.Module):
    """Path Aggregation Feature Pyramid Network with CSPNeXt blocks."""
    def __init__(self, in_channels: List[int] = [128, 256, 512], out_channels: int = 128, num_blocks: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Lateral convs for top-down
        self.reduce_conv5 = ConvBNAct(in_channels[2], out_channels, 1, 1, 0)
        self.top_down_block4 = nn.Sequential(
            ConvBNAct(out_channels + in_channels[1], out_channels, 1, 1, 0),
            CSPNeXtBlock(out_channels, out_channels, expansion=1.0, kernel_size=5, residual=True)
        )
        self.reduce_conv4 = ConvBNAct(out_channels, out_channels, 1, 1, 0)
        self.top_down_block3 = nn.Sequential(
            ConvBNAct(out_channels + in_channels[0], out_channels, 1, 1, 0),
            CSPNeXtBlock(out_channels, out_channels, expansion=1.0, kernel_size=5, residual=True)
        )

        # Bottom-up convs
        self.down_conv3 = ConvBNAct(out_channels, out_channels, 3, 2, 1)
        self.bottom_up_block4 = nn.Sequential(
            ConvBNAct(out_channels * 2, out_channels, 1, 1, 0),
            CSPNeXtBlock(out_channels, out_channels, expansion=1.0, kernel_size=5, residual=True)
        )
        self.down_conv4 = ConvBNAct(out_channels, out_channels, 3, 2, 1)
        self.bottom_up_block5 = nn.Sequential(
            ConvBNAct(out_channels * 2, out_channels, 1, 1, 0),
            CSPNeXtBlock(out_channels, out_channels, expansion=1.0, kernel_size=5, residual=True)
        )

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = inputs  # stride 8, 16, 32

        # Top-down pathway
        p5 = self.reduce_conv5(c5)
        p5_upsample = F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p4 = self.top_down_block4(torch.cat([p5_upsample, c4], dim=1))

        p4_reduced = self.reduce_conv4(p4)
        p4_upsample = F.interpolate(p4_reduced, size=c3.shape[2:], mode="nearest")
        p3_out = self.top_down_block3(torch.cat([p4_upsample, c3], dim=1))

        # Bottom-up pathway
        p3_down = self.down_conv3(p3_out)
        p4_out = self.bottom_up_block4(torch.cat([p3_down, p4], dim=1))

        p4_down = self.down_conv4(p4_out)
        p5_out = self.bottom_up_block5(torch.cat([p4_down, p5], dim=1))

        return p3_out, p4_out, p5_out
