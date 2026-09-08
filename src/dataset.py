#!/usr/bin/env python3
"""Dataset loaders for Object Detection, Semantic Segmentation, and Unified Vision."""

import json
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from common import require_file, require_directory, load_yaml


class DetectionDataset(Dataset):
    """Dataset for Object Detection (YOLO format text annotations).
    Supports 0-byte negative frames with balanced negative sampling during training.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        frames: List[str],
        img_size: Tuple[int, int] = (640, 640),
        is_train: bool = True,
        negative_ratio: Optional[float] = None,
        seed: int = 42,
    ):
        self.dataset_root = Path(dataset_root)
        self.img_dir = self.dataset_root / "images"
        self.det_dir = self.dataset_root / "detection"
        self.img_h, self.img_w = img_size
        self.is_train = is_train

        # Optional balanced sampling between positive and negative frames
        if is_train and negative_ratio is not None:
            pos_frames = []
            neg_frames = []
            for fid in frames:
                lbl_path = self.det_dir / f"{fid}.txt"
                if lbl_path.exists() and lbl_path.stat().st_size > 0:
                    pos_frames.append(fid)
                else:
                    neg_frames.append(fid)

            max_neg = int(len(pos_frames) * negative_ratio)
            rng = random.Random(seed)
            if len(neg_frames) > max_neg:
                neg_frames = rng.sample(neg_frames, max_neg)

            self.frames = pos_frames + neg_frames
            rng.shuffle(self.frames)
        else:
            self.frames = list(frames)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fid = self.frames[idx]
        img_path = self.img_dir / f"{fid}.jpg"
        label_path = self.det_dir / f"{fid}.txt"

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        if label_path.exists() and label_path.stat().st_size > 0:
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    x1 = max(0.0, cx - w / 2.0)
                    y1 = max(0.0, cy - h / 2.0)
                    x2 = min(1.0, cx + w / 2.0)
                    y2 = min(1.0, cy + h / 2.0)
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls_id)

        # Horizontal Flip augmentation during training
        if self.is_train and np.random.rand() > 0.5:
            img_rgb = np.ascontiguousarray(np.fliplr(img_rgb))
            new_boxes = []
            for b in boxes:
                new_boxes.append([1.0 - b[2], b[1], 1.0 - b[0], b[3]])
            boxes = new_boxes

        resized_img = cv2.resize(img_rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(resized_img.transpose(2, 0, 1)).float() / 255.0

        if boxes:
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
            box_tensor[:, [0, 2]] *= self.img_w
            box_tensor[:, [1, 3]] *= self.img_h
            label_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            label_tensor = torch.zeros((0,), dtype=torch.int64)

        return {
            "image": img_tensor,
            "boxes": box_tensor,
            "labels": label_tensor,
            "frame_id": fid,
            "orig_size": (orig_h, orig_w),
            "input_size": (self.img_h, self.img_w),
        }


def detection_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    boxes = [item["boxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    frame_ids = [item["frame_id"] for item in batch]
    orig_sizes = [item["orig_size"] for item in batch]
    input_sizes = [item["input_size"] for item in batch]
    return {
        "images": images,
        "boxes": boxes,
        "labels": labels,
        "frame_ids": frame_ids,
        "orig_sizes": orig_sizes,
        "input_sizes": input_sizes,
    }


class SegmentationDataset(Dataset):
    """Dataset for Semantic Segmentation (single channel uint8 PNG).
    Classes: 0: drivable, 1: caution, 2: non_drivable
    """

    def __init__(
        self,
        dataset_root: str | Path,
        frames: List[str],
        img_size: Tuple[int, int] = (288, 512),
        is_train: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.img_dir = self.dataset_root / "images"
        self.seg_dir = self.dataset_root / "segmentation"
        self.frames = frames
        self.img_h, self.img_w = img_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fid = self.frames[idx]
        img_path = self.img_dir / f"{fid}.jpg"
        mask_path = self.seg_dir / f"{fid}.png"

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        if self.is_train and np.random.rand() > 0.5:
            img_rgb = np.ascontiguousarray(np.fliplr(img_rgb))
            mask = np.ascontiguousarray(np.fliplr(mask))

        resized_img = cv2.resize(img_rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        resized_mask = np.clip(resized_mask, 0, 2)

        img_tensor = torch.from_numpy(resized_img.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(resized_mask).long()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "frame_id": fid,
            "orig_size": (orig_h, orig_w),
        }


class UnifiedVisionDataset(Dataset):
    """Dataset providing both Detection and Segmentation targets for the same frame."""

    def __init__(
        self,
        dataset_root: str | Path,
        frames: List[str],
        det_img_size: Tuple[int, int] = (640, 640),
        seg_img_size: Tuple[int, int] = (288, 512),
        is_train: bool = False,
    ):
        self.det_ds = DetectionDataset(dataset_root, frames, img_size=det_img_size, is_train=is_train)
        self.seg_ds = SegmentationDataset(dataset_root, frames, img_size=seg_img_size, is_train=is_train)

    def __len__(self) -> int:
        return len(self.det_ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        det_item = self.det_ds[idx]
        seg_item = self.seg_ds[idx]
        return {
            "det_image": det_item["image"],
            "boxes": det_item["boxes"],
            "labels": det_item["labels"],
            "seg_image": seg_item["image"],
            "mask": seg_item["mask"],
            "frame_id": det_item["frame_id"],
            "orig_size": det_item["orig_size"],
        }


def load_split_frames(splits_path: str | Path, split_name: str) -> List[str]:
    path = Path(splits_path)
    if not path.is_file():
        raise FileNotFoundError(f"Splits file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    key = f"{split_name}_frames"
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {path}. Available: {list(data.keys())}")
    return data[key]
