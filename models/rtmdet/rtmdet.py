# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""RTMDet-s Complete Detector.
Reference: RTMDet (Lyu et al., arXiv:2212.07784)
Apache-2.0 License.
"""

import torch
import torch.nn as nn
from torchvision.ops import nms
from typing import List, Tuple, Dict, Optional, Any

from .cspnext import CSPNeXt
from .pafpn import CSPNeXtPAFPN
from .head import RTMDetHead


class RTMDetS(nn.Module):
    """RTMDet-s Object Detector for mobile/edge robot vision.
    Classes: 0: step, 1: ditch_hole, 2: puddle, 3: obstacle.
    """
    def __init__(
        self,
        num_classes: int = 4,
        width: float = 0.5,
        depth: float = 0.33,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.backbone = CSPNeXt(width=width, depth=depth)
        neck_channels = int(256 * width)  # 128
        self.neck = CSPNeXtPAFPN(
            in_channels=self.backbone.out_channels,
            out_channels=neck_channels,
            num_blocks=1,
        )
        self.head = RTMDetHead(
            num_classes=num_classes,
            in_channels=neck_channels,
            feat_channels=neck_channels,
            stacked_convs=2,
            strides=[8, 16, 32],
        )

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[Dict[str, List[torch.Tensor]]] = None,
        export_onnx: bool = False,
    ) -> Any:
        feats = self.backbone(x)
        neck_feats = self.neck(feats)
        cls_scores, bbox_preds = self.head(neck_feats)

        # 1. ONNX Export mode
        if export_onnx:
            device = x.device
            batch_size = x.shape[0]
            anchor_points, _ = self.head.generate_anchors(cls_scores, device)

            cls_preds_flat = torch.cat(
                [s.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes) for s in cls_scores], dim=1
            )
            reg_preds_flat = torch.cat(
                [r.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for r in bbox_preds], dim=1
            )

            decoded_boxes = self.head.decode_boxes(anchor_points, reg_preds_flat)
            cls_probs = torch.sigmoid(cls_preds_flat)
            return decoded_boxes, cls_probs

        # 2. Training mode
        if self.training:
            if targets is None:
                raise ValueError("targets must be provided when training=True")
            return self.head.compute_loss(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                gt_boxes_list=targets["boxes"],
                gt_labels_list=targets["labels"],
            )

        # 3. PyTorch Inference mode with NMS
        device = x.device
        batch_size = x.shape[0]
        anchor_points, _ = self.head.generate_anchors(cls_scores, device)

        cls_preds_flat = torch.cat(
            [s.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes) for s in cls_scores], dim=1
        )
        reg_preds_flat = torch.cat(
            [r.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for r in bbox_preds], dim=1
        )

        decoded_boxes = self.head.decode_boxes(anchor_points, reg_preds_flat)
        cls_probs = torch.sigmoid(cls_preds_flat)

        results = []
        for b in range(batch_size):
            boxes_b = decoded_boxes[b]     # [N, 4]
            probs_b = cls_probs[b]         # [N, num_classes]

            # Max score and label per anchor
            scores_b, labels_b = torch.max(probs_b, dim=-1)

            # Filter by confidence threshold
            keep = scores_b >= self.conf_threshold
            filtered_boxes = boxes_b[keep]
            filtered_scores = scores_b[keep]
            filtered_labels = labels_b[keep]

            if len(filtered_boxes) == 0:
                results.append({
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros((0,), device=device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                })
                continue

            # Class-aware NMS
            max_coord = filtered_boxes.max() + 1.0
            offsets = filtered_labels.float() * max_coord
            nms_boxes = filtered_boxes + offsets.unsqueeze(1)
            keep_indices = nms(nms_boxes, filtered_scores, self.nms_threshold)

            results.append({
                "boxes": filtered_boxes[keep_indices],
                "scores": filtered_scores[keep_indices],
                "labels": filtered_labels[keep_indices],
            })

        return results


def build_rtmdet_s(num_classes: int = 4, conf_thresh: float = 0.25, nms_thresh: float = 0.45) -> RTMDetS:
    """Build pure PyTorch RTMDet-s model."""
    return RTMDetS(
        num_classes=num_classes,
        width=0.5,
        depth=0.33,
        conf_threshold=conf_thresh,
        nms_threshold=nms_thresh,
    )
