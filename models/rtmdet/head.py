# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Decoupled Anchor-free Head for RTMDet with Robust Focal Loss & Positive Normalization.
Fixes:
1. Replaces diluted BCE (divided by 8400) with proper Sigmoid Focal Loss normalized by num_pos.
2. Replaces premature soft-IoU target with stable positive targets during scratch training.
3. Strict center-in-box candidate sampling to prevent noisy background assignments.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from .cspnext import ConvBNAct


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute Complete IoU (CIoU) between two sets of boxes (xyxy format)."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1 + area2 - inter_area + eps
    iou = inter_area / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps

    b1_cx = (b1_x1 + b1_x2) / 2.0
    b1_cy = (b1_y1 + b1_y2) / 2.0
    b2_cx = (b2_x1 + b2_x2) / 2.0
    b2_cy = (b2_y1 + b2_y2) / 2.0
    rho2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    w1, h1 = (b1_x2 - b1_x1).clamp(min=eps), (b1_y2 - b1_y1).clamp(min=eps)
    w2, h2 = (b2_x2 - b2_x1).clamp(min=eps), (b2_y2 - b2_y1).clamp(min=eps)
    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - (rho2 / c2 + alpha * v)
    return ciou.clamp(min=-1.0, max=1.0)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Original Sigmoid Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)"""
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    return loss


class RTMDetHead(nn.Module):
    """RTMDet Decoupled Anchor-free Head."""
    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 128,
        feat_channels: int = 128,
        stacked_convs: int = 2,
        strides: List[int] = [8, 16, 32],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in range(stacked_convs):
            self.cls_convs.append(ConvBNAct(feat_channels, feat_channels, 3, 1, 1))
            self.reg_convs.append(ConvBNAct(feat_channels, feat_channels, 3, 1, 1))

        self.rtm_cls = nn.Conv2d(feat_channels, num_classes, 3, 1, 1)
        self.rtm_reg = nn.Conv2d(feat_channels, 4, 3, 1, 1)

        self.scales = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in strides])

        # Prior bias initialization for classification (p=0.01)
        bias_init = -math.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.rtm_cls.bias, bias_init)

    def forward(self, feats: Tuple[torch.Tensor, ...]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_scores = []
        bbox_preds = []
        for feat, stride, scale in zip(feats, self.strides, self.scales):
            cls_feat = feat
            reg_feat = feat
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)

            cls_score = self.rtm_cls(cls_feat)
            bbox_pred = F.relu(self.rtm_reg(reg_feat) * scale) * stride

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        return cls_scores, bbox_preds

    def generate_anchors(self, feats: List[torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor_points = []
        stride_tensor = []
        for feat, stride in zip(feats, self.strides):
            _, _, h, w = feat.shape
            y = torch.arange(h, device=device).float() * stride + stride / 2.0
            x = torch.arange(w, device=device).float() * stride + stride / 2.0
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            points = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
            anchor_points.append(points)
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float32, device=device))

        return torch.cat(anchor_points, dim=0), torch.cat(stride_tensor, dim=0)

    def decode_boxes(self, anchor_points: torch.Tensor, reg_preds: torch.Tensor) -> torch.Tensor:
        cx, cy = anchor_points[:, 0], anchor_points[:, 1]
        l = reg_preds[..., 0]
        t = reg_preds[..., 1]
        r = reg_preds[..., 2]
        b = reg_preds[..., 3]

        x1 = cx - l
        y1 = cy - t
        x2 = cx + r
        y2 = cy + b
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def compute_loss(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        gt_boxes_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]

        anchor_points, stride_tensor = self.generate_anchors(cls_scores, device)
        total_anchors = anchor_points.shape[0]

        cls_preds_flat = torch.cat(
            [s.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes) for s in cls_scores], dim=1
        )
        reg_preds_flat = torch.cat(
            [r.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for r in bbox_preds], dim=1
        )

        decoded_boxes = self.decode_boxes(anchor_points, reg_preds_flat)

        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        num_pos_total = 0

        anc_x = anchor_points[:, 0:1]  # [total_anchors, 1]
        anc_y = anchor_points[:, 1:2]

        for b in range(batch_size):
            gt_boxes = gt_boxes_list[b].to(device)
            gt_labels = gt_labels_list[b].to(device)
            pred_cls = cls_preds_flat[b]       # [total_anchors, num_classes]
            pred_box = decoded_boxes[b]        # [total_anchors, 4]

            target_cls = torch.zeros_like(pred_cls)
            num_gt = gt_boxes.shape[0]

            if num_gt > 0:
                gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
                gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0

                dist_x = torch.abs(anc_x - gt_cx.unsqueeze(0))
                dist_y = torch.abs(anc_y - gt_cy.unsqueeze(0))

                # Positive candidates: Inside GT box AND within center radius
                in_box = (
                    (anc_x >= gt_boxes[:, 0:1].T) & (anc_x <= gt_boxes[:, 2:3].T) &
                    (anc_y >= gt_boxes[:, 1:2].T) & (anc_y <= gt_boxes[:, 3:4].T)
                )
                near_center = (dist_x <= 2.5 * stride_tensor) & (dist_y <= 2.5 * stride_tensor)
                candidate_mask = in_box & near_center

                pos_anchor_indices = []
                matched_gt_indices = []

                for gt_idx in range(num_gt):
                    cands = torch.where(candidate_mask[:, gt_idx])[0]
                    if len(cands) == 0:
                        # Fallback: points strictly in box, or nearest anchor to center
                        cands_in_box = torch.where(in_box[:, gt_idx])[0]
                        if len(cands_in_box) > 0:
                            cands = cands_in_box
                        else:
                            dists = (anc_x - gt_cx[gt_idx]) ** 2 + (anc_y - gt_cy[gt_idx]) ** 2
                            cands = torch.argmin(dists).unsqueeze(0)

                    cls_id = gt_labels[gt_idx]
                    # Assign confident positive target (1.0) during scratch training
                    target_cls[cands, cls_id] = 1.0

                    pos_anchor_indices.append(cands)
                    matched_gt_indices.append(torch.full_like(cands, gt_idx))

                if pos_anchor_indices:
                    pos_idx = torch.cat(pos_anchor_indices)
                    gt_idx_all = torch.cat(matched_gt_indices)

                    pos_pred_boxes = pred_box[pos_idx]
                    pos_target_boxes = gt_boxes[gt_idx_all]
                    pos_ciou = bbox_ciou(pos_pred_boxes, pos_target_boxes)
                    box_loss = (1.0 - pos_ciou).sum()
                    total_box_loss += box_loss
                    num_pos_total += len(pos_idx)

            # Compute Focal Loss for this image (sum over all anchors and classes)
            focal_loss_img = sigmoid_focal_loss(pred_cls, target_cls, alpha=0.25, gamma=2.0).sum()
            total_cls_loss += focal_loss_img

        # Normalize by total number of positive anchors across the batch (Standard RetinaNet/RTMDet normalization)
        norm_pos = max(num_pos_total, 1)
        loss_cls_norm = total_cls_loss / norm_pos
        loss_box_norm = total_box_loss / norm_pos

        loss_total = loss_cls_norm + 2.0 * loss_box_norm

        return {
            "loss_cls": loss_cls_norm,
            "loss_box": loss_box_norm,
            "loss_total": loss_total,
            "num_pos": num_pos_total,
        }
