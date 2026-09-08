# --------------------------------------------------------
# PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# Licensed under the MIT License.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """ResNet BasicBlock used in PIDNet."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Pag(nn.Module):
    """Pixel-attention guided (Pag) fusion block.
    Transfers context from I-branch to P-branch.
    """
    def __init__(self, p_channels: int, i_channels: int, mid_channels: int):
        super().__init__()
        self.f_p = nn.Sequential(
            nn.Conv2d(p_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.f_i = nn.Sequential(
            nn.Conv2d(i_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Sigmoid(),
        )

    def forward(self, p: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        if i.shape[2:] != p.shape[2:]:
            i = F.interpolate(i, size=p.shape[2:], mode="bilinear", align_corners=True)
        att = self.f_i(i)
        p_feat = self.f_p(p)
        return p + p_feat * att


class Bag(nn.Module):
    """Boundary-attention guided (Bag) fusion block.
    Fuses P-branch and I-branch guided by D-branch edge features.
    """
    def __init__(self, p_channels: int, i_channels: int, d_channels: int, out_channels: int):
        super().__init__()
        self.conv_i = nn.Sequential(
            nn.Conv2d(i_channels, p_channels, 1, bias=False),
            nn.BatchNorm2d(p_channels),
            nn.ReLU(inplace=True),
        ) if i_channels != p_channels else nn.Identity()

        self.conv_d = nn.Sequential(
            nn.Conv2d(d_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
        ) if d_channels != 1 else nn.Identity()

        self.conv_out = nn.Sequential(
            nn.Conv2d(p_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, p: torch.Tensor, i: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if i.shape[2:] != p.shape[2:]:
            i = F.interpolate(i, size=p.shape[2:], mode="bilinear", align_corners=True)
        if d.shape[2:] != p.shape[2:]:
            d = F.interpolate(d, size=p.shape[2:], mode="bilinear", align_corners=True)

        i_aligned = self.conv_i(i)
        edge_att = torch.sigmoid(self.conv_d(d))
        fused = p * (1.0 - edge_att) + i_aligned * edge_att
        return self.conv_out(fused)


class PAPPM(nn.Module):
    """Parallel Aggregation Pyramid Pooling Module (PAPPM) - Official ONNX-compliant design."""
    def __init__(self, in_channels: int, branch_channels: int, out_channels: int):
        super().__init__()
        self.scale0 = ConvBNReLU(in_channels, branch_channels, 1, 1, 0)
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            ConvBNReLU(in_channels, branch_channels, 1, 1, 0),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            ConvBNReLU(in_channels, branch_channels, 1, 1, 0),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            ConvBNReLU(in_channels, branch_channels, 1, 1, 0),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBNReLU(in_channels, branch_channels, 1, 1, 0),
        )
        self.scale_process = nn.Sequential(
            ConvBNReLU(branch_channels * 4, branch_channels * 4, 3, 1, 1),
        )
        self.compression = ConvBNReLU(branch_channels * 5, out_channels, 1, 1, 0)
        self.shortcut = ConvBNReLU(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        x0 = self.scale0(x)
        s1 = F.interpolate(self.scale1(x), size=(h, w), mode="bilinear", align_corners=True) + x0
        s2 = F.interpolate(self.scale2(x), size=(h, w), mode="bilinear", align_corners=True) + x0
        s3 = F.interpolate(self.scale3(x), size=(h, w), mode="bilinear", align_corners=True) + x0
        s4 = F.interpolate(self.scale4(x), size=(h, w), mode="bilinear", align_corners=True) + x0

        scale_cat = torch.cat([s1, s2, s3, s4], dim=1)
        scale_proc = self.scale_process(scale_cat)

        out = self.compression(torch.cat([x0, scale_proc], dim=1)) + self.shortcut(x)
        return out


class SegmentHead(nn.Module):
    """Prediction head for semantic segmentation."""
    def __init__(self, in_channels: int, mid_channels: int, num_classes: int, scale_factor: int = 8):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = ConvBNReLU(in_channels, mid_channels, 3, 1, 1)
        self.cls = nn.Conv2d(mid_channels, num_classes, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cls(self.conv(x))
        if self.scale_factor > 1:
            out = F.interpolate(out, scale_factor=self.scale_factor, mode="bilinear", align_corners=True)
        return out
