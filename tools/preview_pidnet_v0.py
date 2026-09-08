#!/usr/bin/env python3
"""PIDNet-S inference and 3-panel overlay preview generator.

Generates side-by-side comparison images:
[ ORIGINAL | GROUND TRUTH | PREDICTION (PIDNet-S) ]
for 20 frames from the test split.

Color specification (ODT / AI모델_데이터라벨링.md):
- drivable (0)     : Green  (0, 255, 0)
- caution (1)      : Yellow (0, 255, 255)
- non_drivable (2) : Red    (0, 0, 255)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import ensure_directory, load_yaml, select_device
from models.pidnet import build_pidnet_s

# Segmentation palette (OpenCV BGR format)
SEG_COLORS = {
    0: (0, 255, 0),      # drivable     = 초록
    1: (0, 255, 255),    # caution      = 노랑
    2: (0, 0, 255),      # non_drivable = 빨강
}

CLASS_NAMES = {
    0: "drivable",
    1: "caution",
    2: "non_drivable",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PIDNet-S 3-panel preview overlays")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/segmentation/best.pt",
        help="Path to PIDNet-S model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_segmentation.yaml",
        help="Path to segmentation training config",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="data/splits/v0_splits.json",
        help="Path to splits JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pidnet_v0_preview",
        help="Output directory for preview images",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=20,
        help="Number of test frames to evaluate and visualize",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=640,
        help="Width of each panel tile (default: 640)",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=360,
        help="Height of each panel tile (default: 360)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Overlay mask alpha blending weight (default: 0.35)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device (auto, cuda, cpu)",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Tuple[int, int]]:
    """Load PIDNet-S model with checkpoint weights."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    num_classes = model_cfg.get("num_classes", 3)
    m = model_cfg.get("m", 32)
    input_size = tuple(model_cfg.get("input_size", [288, 512]))  # [H, W]
    class_weights = train_cfg.get("class_weights", [2.0, 10.0, 0.5])

    model = build_pidnet_s(num_classes=num_classes, m=m, class_weights=class_weights).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "unknown")
    best_metric = ckpt.get("best_metric", None)
    metric_str = f"{best_metric:.4f}" if isinstance(best_metric, (int, float)) else str(best_metric)
    print(f"[Model] Loaded PIDNet-S (epoch {epoch}, best_mIoU: {metric_str}) from {checkpoint_path}")
    print(f"[Model] Input size: [H={input_size[0]}, W={input_size[1]}] | Classes: {num_classes}")

    return model, input_size


def create_color_overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Create color overlay on original BGR image using segmentation mask.
    Drivable: Green, Caution: Yellow, Non-drivable: Red.
    """
    color_mask = np.zeros_like(image_bgr)
    for class_id, color in SEG_COLORS.items():
        color_mask[mask == class_id] = color

    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, color_mask, alpha, 0)
    return overlay


def add_panel_header(panel: np.ndarray, title: str, accent_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Add a clean top banner with title to a panel."""
    h, w = panel.shape[:2]
    banner_h = 34

    # Create top banner with dark semi-transparent bar
    banner = panel.copy()
    cv2.rectangle(banner, (0, 0), (w, banner_h), (20, 20, 20), -1)
    panel_with_banner = cv2.addWeighted(panel, 0.25, banner, 0.75, 0)

    # Accent color small pill on left
    cv2.rectangle(panel_with_banner, (10, 8), (14, banner_h - 8), accent_color, -1)

    # Title text
    cv2.putText(
        panel_with_banner,
        title,
        (22, banner_h - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel_with_banner


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, Any]:
    """Compute per-class IoU and mean IoU."""
    class_ious = {}
    valid_ious = []

    for c in range(3):
        c_name = CLASS_NAMES[c]
        p_c = pred_mask == c
        g_c = gt_mask == c

        inter = np.logical_and(p_c, g_c).sum()
        union = np.logical_or(p_c, g_c).sum()

        if union == 0:
            iou = 1.0 if inter == 0 else 0.0
            class_ious[c_name] = {"iou": iou, "present": False, "inter": int(inter), "union": int(union)}
        else:
            iou = float(inter) / float(union)
            class_ious[c_name] = {"iou": iou, "present": True, "inter": int(inter), "union": int(union)}
            valid_ious.append(iou)

    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    pixel_acc = float((pred_mask == gt_mask).mean())

    return {
        "mIoU": miou,
        "pixel_accuracy": pixel_acc,
        "drivable_iou": class_ious["drivable"]["iou"],
        "class_ious": class_ious,
    }


def make_3panel_image(
    original_bgr: np.ndarray,
    gt_overlay: np.ndarray,
    pred_overlay: np.ndarray,
    frame_id: str,
    metrics: Dict[str, Any],
    tile_w: int = 640,
    tile_h: int = 360,
) -> np.ndarray:
    """Combine Original, GT Overlay, and Prediction Overlay into a single 3-panel image."""
    p1 = cv2.resize(original_bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    p2 = cv2.resize(gt_overlay, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    p3 = cv2.resize(pred_overlay, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

    p1 = add_panel_header(p1, "[1] ORIGINAL", (200, 200, 200))
    p2 = add_panel_header(p2, "[2] GROUND TRUTH", (0, 255, 0))
    p3 = add_panel_header(p3, "[3] PREDICTION (PIDNet-S)", (0, 220, 255))

    combined = np.hstack([p1, p2, p3])
    total_w = combined.shape[1]

    # Footer bar
    footer_h = 38
    footer = np.zeros((footer_h, total_w, 3), dtype=np.uint8)
    footer[:] = (24, 24, 24)

    # Top border line of footer
    cv2.line(footer, (0, 0), (total_w, 0), (60, 60, 60), 1)

    # Info text: Frame ID & Metrics
    drivable_iou = metrics["drivable_iou"]
    miou = metrics["mIoU"]
    pix_acc = metrics["pixel_accuracy"]
    info_text = (
        f"Frame: {frame_id}  |  Drivable IoU: {drivable_iou * 100:.1f}%  |  "
        f"mIoU: {miou * 100:.1f}%  |  Pixel Acc: {pix_acc * 100:.1f}%"
    )
    cv2.putText(
        footer,
        info_text,
        (16, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    # Legend on the right
    # Green: Drivable, Yellow: Caution, Red: Non-drivable
    legend_start_x = total_w - 480
    legends = [
        ("Drivable", SEG_COLORS[0]),
        ("Caution", SEG_COLORS[1]),
        ("Non-drivable", SEG_COLORS[2]),
    ]
    cur_x = legend_start_x
    for name, color in legends:
        cv2.rectangle(footer, (cur_x, 12), (cur_x + 14, 26), color, -1)
        cv2.rectangle(footer, (cur_x, 12), (cur_x + 14, 26), (255, 255, 255), 1)
        cv2.putText(
            footer,
            name,
            (cur_x + 20, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        cur_x += 150

    return np.vstack([combined, footer])


def main():
    args = parse_args()
    device_name = select_device(args.device)
    device = torch.device(device_name)

    print("=" * 70)
    print("PIDNet-S Test Split 3-Panel Overlay Generator")
    print(f"Device: {device_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 70)

    # Load configurations
    cfg_path = PROJECT_ROOT / args.config
    config = load_yaml(cfg_path)
    dataset_cfg = load_yaml(PROJECT_ROOT / config["dataset"]["config"])

    dataset_root = Path(dataset_cfg["dataset_root"])
    img_dir = dataset_root / "images"
    seg_dir = dataset_root / "segmentation"

    splits_path = PROJECT_ROOT / args.splits
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_frames = splits.get("test_frames", [])
    if not test_frames:
        raise ValueError(f"No test_frames found in {splits_path}")

    selected_frames = test_frames[: args.num_frames]
    print(f"Total test frames available: {len(test_frames):,}")
    print(f"Selected frames to process: {len(selected_frames)}")

    # Load model
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    model, (input_h, input_w) = load_model(checkpoint_path, device)

    # Output directory
    output_dir = ensure_directory(PROJECT_ROOT / args.output_dir)

    results_summary: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    for idx, fid in enumerate(selected_frames, start=1):
        img_p = img_dir / f"{fid}.jpg"
        seg_p = seg_dir / f"{fid}.png"

        if not img_p.exists():
            print(f"[{idx:02d}/{len(selected_frames)}] ERROR: Missing image {img_p}")
            continue
        if not seg_p.exists():
            print(f"[{idx:02d}/{len(selected_frames)}] ERROR: Missing mask {seg_p}")
            continue

        # 1. Read Original and GT
        img_bgr = cv2.imread(str(img_p))
        gt_mask = cv2.imread(str(seg_p), cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = img_bgr.shape[:2]

        # 2. Preprocess for PIDNet-S
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(img_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        img_tensor = (
            torch.from_numpy(resized_rgb.transpose(2, 0, 1))
            .float()
            .unsqueeze(0)
            .to(device)
            / 255.0
        )

        # 3. Model Inference
        with torch.no_grad():
            logits = model(img_tensor)  # [1, 3, input_h, input_w]
            # Interpolate logits back to original image resolution for high-quality mask
            logits_orig = F.interpolate(
                logits, size=(orig_h, orig_w), mode="bilinear", align_corners=True
            )
            pred_mask = (
                torch.argmax(logits_orig, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            )

        # 4. Compute Metrics
        metrics = compute_metrics(pred_mask, gt_mask)

        # 5. Generate Overlays
        gt_overlay = create_color_overlay(img_bgr, gt_mask, alpha=args.alpha)
        pred_overlay = create_color_overlay(img_bgr, pred_mask, alpha=args.alpha)

        # 6. Compose 3-panel image
        panel_image = make_3panel_image(
            original_bgr=img_bgr,
            gt_overlay=gt_overlay,
            pred_overlay=pred_overlay,
            frame_id=fid,
            metrics=metrics,
            tile_w=args.tile_width,
            tile_h=args.tile_height,
        )

        # 7. Save output
        out_filename = f"{fid}.jpg"
        out_path = output_dir / out_filename
        cv2.imwrite(str(out_path), panel_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        record = {
            "frame_id": fid,
            "filename": out_filename,
            "drivable_iou": round(metrics["drivable_iou"], 4),
            "mIoU": round(metrics["mIoU"], 4),
            "pixel_accuracy": round(metrics["pixel_accuracy"], 4),
            "class_ious": {
                c: round(v["iou"], 4) for c, v in metrics["class_ious"].items()
            },
        }
        results_summary.append(record)

        print(
            f"[{idx:02d}/{len(selected_frames)}] {fid} -> {out_filename} "
            f"| Drivable IoU: {metrics['drivable_iou'] * 100:.1f}% "
            f"| mIoU: {metrics['mIoU'] * 100:.1f}% "
            f"| PixAcc: {metrics['pixel_accuracy'] * 100:.1f}%"
        )

    total_time = time.perf_counter() - total_start

    # Save summary report JSON
    avg_drivable = float(np.mean([r["drivable_iou"] for r in results_summary]))
    avg_miou = float(np.mean([r["mIoU"] for r in results_summary]))
    avg_pixacc = float(np.mean([r["pixel_accuracy"] for r in results_summary]))

    summary_data = {
        "model": "PIDNet-S",
        "checkpoint": str(args.checkpoint),
        "split": "test",
        "num_frames": len(results_summary),
        "total_time_seconds": round(total_time, 2),
        "fps": round(len(results_summary) / (total_time + 1e-6), 2),
        "average_metrics": {
            "drivable_iou": round(avg_drivable, 4),
            "mIoU": round(avg_miou, 4),
            "pixel_accuracy": round(avg_pixacc, 4),
        },
        "frames": results_summary,
    }

    summary_json_path = output_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("Inference and 3-Panel Overlay Generation Complete!")
    print(f"Processed: {len(results_summary)} frames in {total_time:.2f}s ({summary_data['fps']} FPS)")
    print(f"Average Drivable IoU : {avg_drivable * 100:.2f}%")
    print(f"Average mIoU         : {avg_miou * 100:.2f}%")
    print(f"Average Pixel Acc    : {avg_pixacc * 100:.2f}%")
    print(f"Summary JSON saved   : {summary_json_path}")
    print(f"Output Directory     : {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
