#!/usr/bin/env python3
"""Inference and visualization script for RTMDet-s best checkpoint.

Evaluates on 20 test split images with Ground Truth detections,
drawing distinct GT bboxes (dashed) and Prediction bboxes (solid)
with class names (step, ditch_hole, puddle, obstacle) and confidence scores.
"""

import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.rtmdet import build_rtmdet_s

# Class names and BGR colors according to AGENTS.md:
# step: 보라 (Purple), ditch_hole: 파랑 (Blue), puddle: 하늘색 (Sky Blue), obstacle: 주황 (Orange)
CLASS_NAMES = ["step", "ditch_hole", "puddle", "obstacle"]
CLASS_COLORS_BGR = {
    0: (240, 32, 160),    # step: 보라 (Purple)
    1: (255, 102, 0),     # ditch_hole: 파랑 (Blue)
    2: (255, 191, 0),     # puddle: 하늘색 (Sky Blue)
    3: (0, 140, 255),     # obstacle: 주황 (Orange)
}


def draw_dashed_line(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                     color: Tuple[int, int, int], thickness: int = 2,
                     dash_length: int = 12, gap_length: int = 8):
    """Draw an anti-aliased dashed line between pt1 and pt2."""
    dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist <= 0:
        return
    vx = (pt2[0] - pt1[0]) / dist
    vy = (pt2[1] - pt1[1]) / dist
    curr = 0.0
    while curr < dist:
        start_x = int(round(pt1[0] + vx * curr))
        start_y = int(round(pt1[1] + vy * curr))
        end_val = min(curr + dash_length, dist)
        end_x = int(round(pt1[0] + vx * end_val))
        end_y = int(round(pt1[1] + vy * end_val))
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness, cv2.LINE_AA)
        curr += dash_length + gap_length


def draw_dashed_rect(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                     color: Tuple[int, int, int], thickness: int = 2,
                     dash_length: int = 12, gap_length: int = 8):
    """Draw a dashed rectangle using 4 dashed lines."""
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash_length, gap_length)
    draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash_length, gap_length)
    draw_dashed_line(img, (x2, y2), (x1, y2), color, thickness, dash_length, gap_length)
    draw_dashed_line(img, (x1, y2), (x1, y1), color, thickness, dash_length, gap_length)


def draw_label(img: np.ndarray, text: str, pt: Tuple[int, int],
               bg_color: Tuple[int, int, int], is_gt: bool = False):
    """Draw a clean label box with text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    font_thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    x, y = int(pt[0]), int(pt[1])
    if y - th - 8 < 48:
        y1 = max(48, y)
        y2 = y1 + th + 8
    else:
        y1 = y - th - 8
        y2 = y

    x1 = max(0, x)
    x2 = min(img.shape[1] - 1, x1 + tw + 10)

    overlay = img.copy()
    if is_gt:
        # GT: Dark background with colored border and white text
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, 1, cv2.LINE_AA)
        cv2.putText(img, text, (x1 + 5, y2 - baseline - 2), font, font_scale,
                    (255, 255, 255), font_thickness, cv2.LINE_AA)
    else:
        # Pred: Solid colored background with high-contrast text
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
        brightness = (bg_color[0] * 0.114 + bg_color[1] * 0.587 + bg_color[2] * 0.299)
        txt_col = (0, 0, 0) if brightness > 140 else (255, 255, 255)
        cv2.putText(img, text, (x1 + 5, y2 - baseline - 2), font, font_scale,
                    txt_col, font_thickness, cv2.LINE_AA)


def draw_header_banner(img: np.ndarray, frame_id: str, gt_count: int, pred_count: int):
    """Draw top header banner with legend and class color indicators."""
    h, w = img.shape[:2]
    header_h = 44
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    info_text = f"[{frame_id}]   GT: {gt_count}   |   Pred: {pred_count}"
    cv2.putText(img, info_text, (16, 29), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Legend GT style sample (Dashed)
    cv2.putText(img, "GT:", (450, 28), font, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
    draw_dashed_line(img, (485, 22), (535, 22), (255, 255, 255), 3, dash_length=8, gap_length=5)

    # Legend Pred style sample (Solid)
    cv2.putText(img, "Pred:", (560, 28), font, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.line(img, (615, 22), (665, 22), (255, 255, 255), 3, cv2.LINE_AA)

    # Class colors
    classes_info = [
        ("step", CLASS_COLORS_BGR[0]),
        ("ditch_hole", CLASS_COLORS_BGR[1]),
        ("puddle", CLASS_COLORS_BGR[2]),
        ("obstacle", CLASS_COLORS_BGR[3]),
    ]
    curr_x = 705
    for name, col in classes_info:
        cv2.rectangle(img, (curr_x, 14), (curr_x + 16, 30), col, -1)
        cv2.rectangle(img, (curr_x, 14), (curr_x + 16, 30), (255, 255, 255), 1)
        cv2.putText(img, name, (curr_x + 22, 28), font, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
        curr_x += 22 + int(len(name) * 11) + 20


def load_gt_boxes(label_path: Path, orig_w: int, orig_h: int) -> List[Tuple[int, float, float, float, float]]:
    """Load Ground Truth YOLO bounding boxes scaled to original image dimensions."""
    gt_boxes = []
    if label_path.exists() and label_path.stat().st_size > 0:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    x1 = max(0.0, (cx - w / 2.0) * orig_w)
                    y1 = max(0.0, (cy - h / 2.0) * orig_h)
                    x2 = min(float(orig_w), (cx + w / 2.0) * orig_w)
                    y2 = min(float(orig_h), (cy + h / 2.0) * orig_h)
                    gt_boxes.append((cid, x1, y1, x2, y2))
    return gt_boxes


def main():
    splits_path = PROJECT_ROOT / "data/splits/v0_splits.json"
    ckpt_path = PROJECT_ROOT / "outputs/checkpoints/detection/best.pt"
    dataset_root = PROJECT_ROOT / "dataset/validation_v0"
    images_dir = dataset_root / "images"
    det_dir = dataset_root / "detection"
    output_dir = PROJECT_ROOT / "outputs/rtmdet_v0_preview"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load test split
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_frames = splits["test_frames"]

    # 2. Filter positive test frames (having GT detection annotations)
    pos_frames = []
    for fid in test_frames:
        lbl_file = det_dir / f"{fid}.txt"
        if lbl_file.exists() and lbl_file.stat().st_size > 0:
            with open(lbl_file, "r", encoding="utf-8") as lf:
                lines = [l.strip() for l in lf if l.strip()]
                if lines:
                    pos_frames.append(fid)

    print(f"Total test frames: {len(test_frames)}")
    print(f"Positive test frames with GT detection: {len(pos_frames)}")

    # 3. Load model without modifying any code or weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_rtmdet_s(num_classes=4).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 4. Select 20 representative positive test frames spanning across the split
    valid_candidates = []
    for fid in pos_frames:
        img_path = images_dir / f"{fid}.jpg"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            preds = model(tensor)[0]
        if len(preds["boxes"]) > 0:
            valid_candidates.append(fid)

    print(f"Candidate test frames with both GT and model predictions: {len(valid_candidates)}")

    step = len(valid_candidates) / 20.0
    selected_frames = [valid_candidates[int(i * step)] for i in range(20)]
    print(f"Selected 20 frames: {selected_frames}")

    # 5. Run inference and generate preview images
    summary_data = []

    for idx, fid in enumerate(selected_frames, 1):
        img_path = images_dir / f"{fid}.jpg"
        lbl_path = det_dir / f"{fid}.txt"
        img_bgr = cv2.imread(str(img_path))
        orig_h, orig_w = img_bgr.shape[:2]

        # Load GT
        gt_boxes = load_gt_boxes(lbl_path, orig_w, orig_h)

        # Preprocess & Inference
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            preds = model(tensor)[0]

        pred_boxes_raw = preds["boxes"].cpu().numpy()
        pred_scores_raw = preds["scores"].cpu().numpy()
        pred_labels_raw = preds["labels"].cpu().numpy()

        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0

        vis_img = img_bgr.copy()

        # A. Draw Prediction boxes first (Solid lines, upper label)
        frame_preds_info = []
        for box, score, label in zip(pred_boxes_raw, pred_scores_raw, pred_labels_raw):
            x1, y1 = float(box[0] * scale_x), float(box[1] * scale_y)
            x2, y2 = float(box[2] * scale_x), float(box[3] * scale_y)
            cid = int(label)
            sc = float(score)
            col = CLASS_COLORS_BGR[cid]
            cname = CLASS_NAMES[cid]

            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), col, 3, cv2.LINE_AA)
            pred_label_text = f"[PRED] {cname} {sc:.2f}"
            draw_label(vis_img, pred_label_text, (x1, y1), col, is_gt=False)

            frame_preds_info.append({
                "class_id": cid,
                "class_name": cname,
                "confidence": round(sc, 4),
                "box_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })

        # B. Draw Ground Truth boxes second (Dashed lines, lower label)
        frame_gt_info = []
        for cid, x1, y1, x2, y2 in gt_boxes:
            col = CLASS_COLORS_BGR[cid]
            cname = CLASS_NAMES[cid]

            # Slight 3px outer expansion to keep both lines visible if perfectly overlapping
            draw_dashed_rect(vis_img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), col,
                             thickness=3, dash_length=12, gap_length=8)

            gt_label_text = f"[GT] {cname}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(gt_label_text, font, 0.52, 1)
            lx = int(x1)
            ly = int(y2 + th + 10)
            if ly > orig_h - 10:
                ly = int(y2 - 8)
            draw_label(vis_img, gt_label_text, (lx, ly), col, is_gt=True)

            frame_gt_info.append({
                "class_id": cid,
                "class_name": cname,
                "box_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })

        # C. Draw Header Banner
        draw_header_banner(vis_img, fid, len(gt_boxes), len(pred_boxes_raw))

        # D. Save visualized image
        out_file = output_dir / f"{fid}.jpg"
        cv2.imwrite(str(out_file), vis_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[{idx:02d}/20] Saved {out_file.name} | GT: {len(gt_boxes)} | Pred: {len(pred_boxes_raw)}")

        summary_data.append({
            "frame_id": fid,
            "output_path": str(out_file),
            "num_gt": len(gt_boxes),
            "gt": frame_gt_info,
            "num_pred": len(pred_boxes_raw),
            "pred": frame_preds_info
        })

    # Save summary JSON
    summary_path = output_dir / "preview_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\nAll 20 preview images and summary saved to {output_dir}")


if __name__ == "__main__":
    main()
