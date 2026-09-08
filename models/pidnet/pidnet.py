# --------------------------------------------------------
# PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# Licensed under the MIT License.
# --------------------------------------------------------

"""PIDNet-S Model Implementation.
Classes: 0: drivable, 1: caution, 2: non_drivable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any

from .blocks import ConvBNReLU, BasicBlock, Pag, Bag, PAPPM, SegmentHead


class PIDNetS(nn.Module):
    """PIDNet-S for real-time semantic segmentation in autonomous driving and robot vision.
    m = 32 (base channel multiplier)
    """
    def __init__(self, num_classes: int = 3, m: int = 32, class_weights: Optional[list] = None):
        super().__init__()
        self.num_classes = num_classes
        self.m = m
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

        # 1. Stem (downsample to 1/4)
        self.stem = nn.Sequential(
            ConvBNReLU(3, m, 3, 2, 1),
            ConvBNReLU(m, m * 2, 3, 2, 1),
            BasicBlock(m * 2, m * 2),
            BasicBlock(m * 2, m * 2),
        )

        # 2. Stage 1 -> downsample to 1/8
        self.down_stage1 = nn.Sequential(
            nn.Conv2d(m * 2, m * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(m * 2),
            nn.ReLU(inplace=True),
        )

        # 3. P-Branch (Proportional: high resolution, stride 8)
        self.p_layer1 = nn.Sequential(
            BasicBlock(m * 2, m * 2),
            BasicBlock(m * 2, m * 2),
        )
        self.p_layer2 = nn.Sequential(
            BasicBlock(m * 2, m * 2),
            BasicBlock(m * 2, m * 2),
        )

        # 4. I-Branch (Integral: deep context, stride 8 -> 16 -> 32)
        self.i_down1 = nn.Sequential(
            nn.Conv2d(m * 2, m * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(m * 4),
            nn.ReLU(inplace=True),
        )
        self.i_layer1 = nn.Sequential(
            BasicBlock(m * 4, m * 4),
            BasicBlock(m * 4, m * 4),
        )

        self.i_down2 = nn.Sequential(
            nn.Conv2d(m * 4, m * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(m * 8),
            nn.ReLU(inplace=True),
        )
        self.i_layer2 = nn.Sequential(
            BasicBlock(m * 8, m * 8),
            BasicBlock(m * 8, m * 8),
        )

        # Context Aggregation: PAPPM
        self.pappm = PAPPM(m * 8, m * 2, m * 4)

        # 5. Information interaction between P and I branches (Pag)
        self.pag1 = Pag(p_channels=m * 2, i_channels=m * 4, mid_channels=m * 2)
        self.pag2 = Pag(p_channels=m * 2, i_channels=m * 4, mid_channels=m * 2)

        # 6. D-Branch (Derivative: edge details, stride 8)
        self.d_layer = nn.Sequential(
            ConvBNReLU(m * 2, m * 2, 3, 1, 1),
            ConvBNReLU(m * 2, m * 2, 3, 1, 1),
            ConvBNReLU(m * 2, m * 2, 3, 1, 1),
        )

        # 7. Fusion: Bag block
        self.bag = Bag(p_channels=m * 2, i_channels=m * 4, d_channels=m * 2, out_channels=m * 4)

        # 8. Heads
        # Main segmentation head (stride 8 -> full resolution)
        self.seg_head = SegmentHead(m * 4, m * 2, num_classes, scale_factor=8)
        # Auxiliary head for D-branch boundary (stride 8 -> full resolution, 1 channel boundary)
        self.d_head = SegmentHead(m * 2, m, 1, scale_factor=8)

    def forward(self, x: torch.Tensor, target_mask: Optional[torch.Tensor] = None, export_onnx: bool = False) -> Any:
        h_orig, w_orig = x.shape[2:]

        # Stem
        stem_out = self.stem(x)  # stride 4, channels 64

        # Down to stride 8
        s8 = self.down_stage1(stem_out)  # stride 8, channels 64

        # Initial P, I, D features at stride 8
        p_feat = self.p_layer1(s8)
        d_feat = self.d_layer(s8)

        # I-branch path
        i_s16 = self.i_down1(s8)     # stride 16, channels 128
        i_feat16 = self.i_layer1(i_s16)

        # First Pag interaction
        p_feat = self.pag1(p_feat, i_feat16)

        i_s32 = self.i_down2(i_feat16)  # stride 32, channels 256
        i_feat32 = self.i_layer2(i_s32)
        i_context = self.pappm(i_feat32)  # stride 32, channels 128

        # Second Pag interaction
        p_feat = self.p_layer2(p_feat)
        p_feat = self.pag2(p_feat, i_context)

        # Bag fusion guided by D-branch
        fused = self.bag(p_feat, i_context, d_feat)

        # Main Seg prediction
        logits = self.seg_head(fused)

        # Ensure exact input resolution
        if logits.shape[2:] != (h_orig, w_orig):
            logits = F.interpolate(logits, size=(h_orig, w_orig), mode="bilinear", align_corners=True)

        if export_onnx:
            # ONNX export returns class probabilities
            return F.softmax(logits, dim=1)

        if self.training:
            if target_mask is None:
                raise ValueError("target_mask must be provided in training mode")

            # Weighted Cross Entropy Loss
            ce_loss = F.cross_entropy(logits, target_mask, weight=self.class_weights)

            # Dice Loss for underrepresented caution class (class 1)
            target_one_hot = F.one_hot(target_mask.clamp(0, self.num_classes - 1), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            probs = F.softmax(logits, dim=1)
            intersection = (probs * target_one_hot).sum(dim=(2, 3))
            union = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
            dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
            dice_loss = dice_loss.mean()

            total_loss = ce_loss + 0.5 * dice_loss
            return {
                "loss_ce": ce_loss,
                "loss_dice": dice_loss,
                "loss_total": total_loss,
            }

        return logits


def build_pidnet_s(num_classes: int = 3, m: int = 32, class_weights: Optional[list] = None) -> PIDNetS:
    """Build pure PyTorch PIDNet-S model."""
    return PIDNetS(num_classes=num_classes, m=m, class_weights=class_weights)
