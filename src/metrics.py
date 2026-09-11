#!/usr/bin/env python3
"""Evaluation metrics for Object Detection and Semantic Segmentation."""

import numpy as np
import torch
from typing import List, Dict, Tuple, Any


class DetectionMetrics:
    """Computes Precision, Recall, mAP50, and mAP50-95 for Object Detection.
    Standard COCO-style evaluation with IoU thresholds [0.50:0.05:0.95].
    """
    def __init__(self, num_classes: int = 4, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.iou_thresholds = np.linspace(0.50, 0.95, 10)
        self.reset()

    def reset(self):
        # List of dicts for each image: {"pred_boxes", "pred_scores", "pred_labels", "gt_boxes", "gt_labels"}
        self.records = []

    def update(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ):
        """Add predictions and ground truths for one image. Boxes in xyxy format."""
        self.records.append({
            "pred_boxes": pred_boxes.detach().cpu().numpy(),
            "pred_scores": pred_scores.detach().cpu().numpy(),
            "pred_labels": pred_labels.detach().cpu().numpy(),
            "gt_boxes": gt_boxes.detach().cpu().numpy(),
            "gt_labels": gt_labels.detach().cpu().numpy(),
        })

    def _box_iou_np(self, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between b1 [N, 4] and b2 [M, 4]."""
        if len(b1) == 0 or len(b2) == 0:
            return np.zeros((len(b1), len(b2)), dtype=np.float32)

        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])

        inter_x1 = np.maximum(b1[:, 0:1], b2[:, 0:1].T)
        inter_y1 = np.maximum(b1[:, 1:2], b2[:, 1:2].T)
        inter_x2 = np.minimum(b1[:, 2:3], b2[:, 2:3].T)
        inter_y2 = np.minimum(b1[:, 3:4], b2[:, 3:4].T)

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union = area1[:, None] + area2[None, :] - inter_area
        return inter_area / np.maximum(union, 1e-6)

    def compute(self) -> Dict[str, Any]:
        """Compute mAP50, mAP50-95, Precision, and Recall."""
        ap_per_class_all_iou = np.zeros((self.num_classes, len(self.iou_thresholds)))
        tp_50_total = 0
        fp_50_total = 0
        fn_50_total = 0

        for c in range(self.num_classes):
            c_preds = []
            n_gt_class = 0

            # Gather predictions and ground truths for class c
            for rec_idx, rec in enumerate(self.records):
                gt_mask = rec["gt_labels"] == c
                gt_boxes_c = rec["gt_boxes"][gt_mask]
                n_gt_class += len(gt_boxes_c)

                pred_mask = rec["pred_labels"] == c
                pred_boxes_c = rec["pred_boxes"][pred_mask]
                pred_scores_c = rec["pred_scores"][pred_mask]

                if len(pred_boxes_c) > 0:
                    ious = self._box_iou_np(pred_boxes_c, gt_boxes_c)
                    for i in range(len(pred_boxes_c)):
                        c_preds.append({
                            "score": pred_scores_c[i],
                            "ious": ious[i] if len(gt_boxes_c) > 0 else np.zeros((0,)),
                            # image this prediction belongs to -- needed so gt_idx below is
                            # scoped per-image, not treated as a dataset-global GT index.
                            "rec_idx": rec_idx,
                        })

            if n_gt_class == 0:
                # Class had 0 ground truths across dataset
                continue

            if len(c_preds) == 0:
                fn_50_total += n_gt_class
                continue

            # Sort predictions by descending score
            c_preds.sort(key=lambda x: x["score"], reverse=True)

            # Evaluate for each IoU threshold. Standard greedy matching:
            # predictions sorted by descending score, each claims its best
            # still-unclaimed ground-truth box (>= iou_thresh) in ITS OWN
            # image. The match key must be (rec_idx, gt_idx) -- gt_idx alone
            # is only meaningful within one image, and using id(p) (unique
            # per prediction) as previously done never actually blocked a
            # second prediction from claiming an already-matched GT box,
            # silently double-counting true positives.
            for t_idx, iou_thresh in enumerate(self.iou_thresholds):
                tp = np.zeros(len(c_preds))
                fp = np.zeros(len(c_preds))
                gt_matched_flags = set()

                for idx, p in enumerate(c_preds):
                    ious = p["ious"]
                    best_gt = -1
                    best_iou = 0.0
                    for g_idx in range(len(ious)):
                        if (p["rec_idx"], g_idx) not in gt_matched_flags and ious[g_idx] >= iou_thresh:
                            if ious[g_idx] > best_iou:
                                best_iou = ious[g_idx]
                                best_gt = g_idx
                    if best_gt >= 0:
                        tp[idx] = 1
                        gt_matched_flags.add((p["rec_idx"], best_gt))
                    else:
                        fp[idx] = 1

                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                recalls = tp_cumsum / max(n_gt_class, 1)
                precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-6)

                # Area Under PR Curve
                ap = np.trapezoid(precisions, recalls) if hasattr(np, "trapezoid") else np.trapz(precisions, recalls) if len(recalls) > 1 else (precisions[0] * recalls[0] if len(recalls) == 1 else 0.0)
                ap_per_class_all_iou[c, t_idx] = max(0.0, float(ap))

                if t_idx == 0:  # IoU = 0.50
                    tp_50_total += int(np.sum(tp))
                    fp_50_total += int(np.sum(fp))
                    fn_50_total += (n_gt_class - int(np.sum(tp)))

        # Summary metrics
        # Valid classes that actually appeared in GT
        valid_classes = [c for c in range(self.num_classes) if any(np.sum(rec["gt_labels"] == c) > 0 for rec in self.records)]
        if not valid_classes:
            valid_classes = list(range(self.num_classes))

        ap50_per_class = ap_per_class_all_iou[:, 0]
        ap50_95_per_class = ap_per_class_all_iou.mean(axis=1)

        map50 = float(np.mean([ap50_per_class[c] for c in valid_classes]))
        map50_95 = float(np.mean([ap50_95_per_class[c] for c in valid_classes]))

        precision_50 = tp_50_total / max(tp_50_total + fp_50_total, 1)
        recall_50 = tp_50_total / max(tp_50_total + fn_50_total, 1)

        class_details = {}
        for c in range(self.num_classes):
            name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            class_details[name] = {
                "AP50": round(float(ap50_per_class[c]), 4),
                "AP50_95": round(float(ap50_95_per_class[c]), 4),
            }

        return {
            "Precision": round(float(precision_50), 4),
            "Recall": round(float(recall_50), 4),
            "mAP50": round(float(map50), 4),
            "mAP50_95": round(float(map50_95), 4),
            "per_class": class_details,
            "samples": len(self.records),
        }


class SegmentationMetrics:
    """Computes Confusion Matrix, Pixel Accuracy, Class-wise IoU, and mIoU.
    Classes: 0: drivable, 1: caution, 2: non_drivable
    """
    def __init__(self, num_classes: int = 3, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or ["drivable", "caution", "non_drivable"]
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Add batch of predicted masks [B, H, W] and target masks [B, H, W]."""
        pred_np = pred.detach().cpu().numpy().astype(np.int64).flatten()
        target_np = target.detach().cpu().numpy().astype(np.int64).flatten()

        mask = (target_np >= 0) & (target_np < self.num_classes)
        hist = np.bincount(
            self.num_classes * target_np[mask] + pred_np[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += hist

    def compute(self) -> Dict[str, Any]:
        """Return Pixel Accuracy, mIoU, and per-class IoU."""
        hist = self.confusion_matrix
        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp

        denom = tp + fp + fn
        ious = np.zeros(self.num_classes, dtype=np.float64)
        for c in range(self.num_classes):
            ious[c] = tp[c] / denom[c] if denom[c] > 0 else 0.0

        pixel_acc = float(tp.sum() / max(hist.sum(), 1))
        miou = float(np.mean(ious))

        per_class_iou = {}
        for c in range(self.num_classes):
            name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            per_class_iou[name] = round(float(ious[c]), 4)

        return {
            "Pixel_Accuracy": round(pixel_acc, 4),
            "mIoU": round(miou, 4),
            "class_iou": per_class_iou,
        }
