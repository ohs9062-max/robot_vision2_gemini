#!/usr/bin/env python3
"""Train RTMDet-s object detector on road hazard dataset.
Pure PyTorch implementation (Apache-2.0 License).
Scratch training without unverified commercial pretrained weights.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import load_yaml, ensure_directory, select_device, setup_logging
from dataset import DetectionDataset, detection_collate_fn, load_split_frames
from metrics import DetectionMetrics
from models.rtmdet import build_rtmdet_s


def parse_args():
    parser = argparse.ArgumentParser(description="Train RTMDet-s detector")
    parser.add_argument("--config", default="config/train_detection.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test with small subset")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_detection(model, val_loader, device: str, class_names: list) -> Dict[str, Any]:
    model.eval()
    metric = DetectionMetrics(num_classes=len(class_names), class_names=class_names)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            targets_boxes = batch["boxes"]
            targets_labels = batch["labels"]

            predictions = model(images)  # List of dicts: {"boxes", "scores", "labels"}

            for pred, gt_b, gt_l in zip(predictions, targets_boxes, targets_labels):
                metric.update(
                    pred_boxes=pred["boxes"],
                    pred_scores=pred["scores"],
                    pred_labels=pred["labels"],
                    gt_boxes=gt_b.to(device),
                    gt_labels=gt_l.to(device),
                )

    return metric.compute()


def main():
    args = parse_args()
    logger = setup_logging(args.verbose)

    config = load_yaml(PROJECT_ROOT / args.config)
    dataset_cfg = load_yaml(PROJECT_ROOT / config["dataset"]["config"])
    classes_cfg = load_yaml(PROJECT_ROOT / config["dataset"]["classes"])

    class_names = classes_cfg["detection"]["classes"]
    num_classes = len(class_names)
    img_size = tuple(config["model"]["input_size"])

    # same fix as train_segmentation.py -- config declared a seed nothing
    # actually consumed, so every run was genuinely unreproducible.
    seed = int(config["training"].get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    epochs = args.epochs or int(config["training"]["epochs"])
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    lr = args.lr or float(config["training"]["learning_rate"])
    device_name = select_device(config["training"].get("device", "auto"))
    device = torch.device(device_name)
    use_amp = bool(config["training"].get("amp", True)) and device_name == "cuda"

    # Dataset splits
    dataset_root = Path(dataset_cfg["dataset_root"])
    splits_path = PROJECT_ROOT / dataset_cfg["splits_file"]
    train_frames = load_split_frames(splits_path, "train")
    val_frames = load_split_frames(splits_path, "val")

    if args.smoke:
        logger.info("[SMOKE TEST MODE] Limiting to 64 train and 32 val samples, 2 epochs.")
        train_frames = train_frames[:64]
        val_frames = val_frames[:32]
        epochs = 2
        batch_size = min(batch_size, 8)

    neg_ratio = config["training"].get("negative_ratio", None)
    train_ds = DetectionDataset(dataset_root, train_frames, img_size=img_size, is_train=True, negative_ratio=neg_ratio)
    val_ds = DetectionDataset(dataset_root, val_frames, img_size=img_size, is_train=False)

    num_workers = 0 if args.smoke else int(config["training"].get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate_fn)

    # Model
    model = build_rtmdet_s(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(config["training"].get("weight_decay", 0.05)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(config["training"].get("min_lr", 1e-5))
    )
    scaler = GradScaler("cuda") if use_amp else None

    start_epoch = 1
    best_map50 = -1.0
    out_dir = ensure_directory(PROJECT_ROOT / config["output"]["dir"])

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_map50 = ckpt.get("best_metric", -1.0)
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    logger.info("=== Starting RTMDet-s Training ===")
    logger.info("Device: %s | AMP: %s | Image size: %s | Classes: %s", device_name, use_amp, img_size, class_names)
    logger.info("Train samples: %d | Val samples: %d | Batch size: %d | Epochs: %d", len(train_ds), len(val_ds), batch_size, epochs)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_box_loss = 0.0

        for batch in train_loader:
            images = batch["images"].to(device)
            targets = {
                "boxes": [b.to(device) for b in batch["boxes"]],
                "labels": [l.to(device) for l in batch["labels"]],
            }

            optimizer.zero_grad()
            if use_amp:
                with autocast("cuda"):
                    loss_dict = model(images, targets)
                    loss = loss_dict["loss_total"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 35.0)))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                loss = loss_dict["loss_total"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 35.0)))
                optimizer.step()

            total_loss += float(loss.item())
            total_cls_loss += float(loss_dict["loss_cls"].item())
            total_box_loss += float(loss_dict["loss_box"].item())

        scheduler.step()
        train_time = time.perf_counter() - epoch_start
        num_batches = max(len(train_loader), 1)
        avg_loss = total_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_box_loss = total_box_loss / num_batches

        # Validation
        val_res = validate_detection(model, val_loader, device_name, class_names)
        map50 = val_res["mAP50"]
        map50_95 = val_res["mAP50_95"]
        precision = val_res["Precision"]
        recall = val_res["Recall"]

        logger.info(
            "Epoch %d/%d (%.1fs) | Train Loss: %.4f (cls: %.4f, box: %.4f) | Val mAP50: %.4f | mAP50-95: %.4f | Prec: %.4f | Rec: %.4f | LR: %.6f",
            epoch, epochs, train_time, avg_loss, avg_cls_loss, avg_box_loss, map50, map50_95, precision, recall, scheduler.get_last_lr()[0]
        )

        # Checkpoints
        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "classes": class_names,
            "config": config,
            "val_metrics": val_res,
            "best_metric": best_map50,
        }
        torch.save(ckpt, out_dir / "latest.pt")

        if map50 >= best_map50:
            best_map50 = map50
            ckpt["best_metric"] = best_map50
            torch.save(ckpt, out_dir / "best.pt")
            logger.info("-> Saved new best checkpoint to %s (mAP50: %.4f)", out_dir / "best.pt", best_map50)

    logger.info("Training complete. Best checkpoint: %s", out_dir / "best.pt")
    return 0


if __name__ == "__main__":
    main()
