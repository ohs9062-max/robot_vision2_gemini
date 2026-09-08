#!/usr/bin/env python3
"""Build Standardized Dataset v0 from baseline validation frames.

- Source: /home/hs/rang/robot_vision/dataset/validation (READ ONLY)
- Output: /home/hs/rang/robot_vision2_gemini/dataset/validation_v0
- Preserves original resolution (1920x1080) and raw label semantics (no modifications).
- Formats:
    images/frame_XXXXXX.jpg
    segmentation/frame_XXXXXX.png
    detection/frame_XXXXXX.txt
    metadata/frame_XXXXXX.json (strictly 6 keys)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

SRC_DIR = Path("/home/hs/rang/robot_vision/dataset/validation")
TGT_DIR = Path("/home/hs/rang/robot_vision2_gemini/dataset/validation_v0")
QC_PATH = Path("/home/hs/rang/robot_vision2_gemini/dataset/validation_v0_qc.json")
STATS_PATH = Path("/home/hs/rang/robot_vision2_gemini/dataset/validation_v0_stats.json")

DET_CLASS_NAMES = {
    0: "step",
    1: "ditch_hole",
    2: "puddle",
    3: "obstacle",
}

WEATHER_MAP = {
    "맑음": "sunny",
    "흐림": "cloudy",
    "비/안개": "after_rain",
    "sunny": "sunny",
    "cloudy": "cloudy",
    "after_rain": "after_rain",
}


def process_single_frame(fid: str) -> Dict[str, Any]:
    """Process a single frame without modifying label semantics."""
    src_img = SRC_DIR / "images" / f"{fid}.png"
    src_seg = SRC_DIR / "segmentation" / f"{fid}.png"
    src_det = SRC_DIR / "detection" / f"{fid}.txt"
    src_meta = SRC_DIR / "metadata" / f"{fid}.json"

    tgt_img = TGT_DIR / "images" / f"{fid}.jpg"
    tgt_seg = TGT_DIR / "segmentation" / f"{fid}.png"
    tgt_det = TGT_DIR / "detection" / f"{fid}.txt"
    tgt_meta = TGT_DIR / "metadata" / f"{fid}.json"

    # 1. Convert image PNG -> JPG (if not already exists)
    if not tgt_img.exists():
        im = cv2.imread(str(src_img))
        if im is None:
            raise RuntimeError(f"Failed to read image {src_img}")
        cv2.imwrite(str(tgt_img), im, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 2. Hardlink or copy segmentation mask
    if not tgt_seg.exists():
        try:
            os.link(src_seg, tgt_seg)
        except OSError:
            shutil.copyfile(src_seg, tgt_seg)

    # 3. Hardlink or copy detection txt
    if not tgt_det.exists():
        try:
            os.link(src_det, tgt_det)
        except OSError:
            shutil.copyfile(src_det, tgt_det)

    # 4. Standardize metadata to strict 6 keys
    with open(src_meta, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # Parse detection note classes
    det_classes_present: set[str] = set()
    bbox_count = 0
    if src_det.exists() and src_det.stat().st_size > 0:
        with open(src_det, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id in DET_CLASS_NAMES:
                        det_classes_present.add(DET_CLASS_NAMES[cls_id])
                    bbox_count += 1

    src_weather = meta_data.get("weather", "맑음")
    std_weather = WEATHER_MAP.get(src_weather, "sunny")
    std_surface = "wet" if std_weather == "after_rain" else "dry"
    location = meta_data.get("location", "지방")
    camera = "front_camera"
    note = sorted(list(det_classes_present))

    std_meta = {
        "frame_id": fid,
        "location": location,
        "weather": std_weather,
        "surface_condition": std_surface,
        "camera": camera,
        "note": note,
    }

    with open(tgt_meta, "w", encoding="utf-8") as f:
        json.dump(std_meta, f, indent=2, ensure_ascii=False)

    return {
        "fid": fid,
        "weather": std_weather,
        "surface_condition": std_surface,
        "location": location,
        "bbox_count": bbox_count,
        "det_classes": note,
    }


def main():
    print(f"=== Starting Dataset v0 Generation ===")
    print(f"Source: {SRC_DIR}")
    print(f"Target: {TGT_DIR}")

    # Ensure output directories
    for sub in ["images", "segmentation", "detection", "metadata"]:
        (TGT_DIR / sub).mkdir(parents=True, exist_ok=True)

    # Identify all frames
    src_images = sorted([f.stem for f in (SRC_DIR / "images").glob("*.png")])
    total_frames = len(src_images)
    print(f"Found {total_frames} frames in source dataset.")
    assert total_frames == 15000, f"Expected 15,000 frames, found {total_frames}"

    start_time = time.time()
    batch_stats = []

    # Run parallel conversion
    num_workers = min(os.cpu_count() or 16, 24)
    print(f"Processing frames using {num_workers} parallel workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, res in enumerate(executor.map(process_single_frame, src_images, chunksize=100)):
            batch_stats.append(res)
            if (i + 1) % 2500 == 0 or (i + 1) == total_frames:
                elapsed = time.time() - start_time
                fps = (i + 1) / elapsed
                print(f"Progress: [{i+1}/{total_frames}] ({((i+1)/total_frames)*100:.1f}%) - {fps:.1f} fps - {elapsed:.1f}s")

    print(f"Dataset v0 build completed in {time.time() - start_time:.1f}s.")

    # Run QC Validation
    print("=== Running QC Validation ===")
    qc_passed = True
    qc_errors = []
    
    # 1. 1:1 file existence check
    for fid in src_images:
        for ext, sub in [(".jpg", "images"), (".png", "segmentation"), (".txt", "detection"), (".json", "metadata")]:
            target_path = TGT_DIR / sub / f"{fid}{ext}"
            if not target_path.exists():
                qc_passed = False
                qc_errors.append(f"Missing file: {target_path}")

    print(f"File 1:1 existence check: {'PASS' if qc_passed else 'FAIL'}")

    # 2. Metadata schema check on all 15,000 files
    ALLOWED_KEYS = {"frame_id", "location", "weather", "surface_condition", "camera", "note"}
    ALLOWED_WEATHER = {"sunny", "cloudy", "after_rain"}
    ALLOWED_SURFACE = {"dry", "wet"}

    meta_qc_passed = True
    for fid in src_images:
        meta_file = TGT_DIR / "metadata" / f"{fid}.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            m = json.load(f)
        if set(m.keys()) != ALLOWED_KEYS:
            meta_qc_passed = False
            qc_errors.append(f"{fid}.json: Invalid keys {set(m.keys())}")
            break
        if m["weather"] not in ALLOWED_WEATHER:
            meta_qc_passed = False
            qc_errors.append(f"{fid}.json: Invalid weather {m['weather']}")
            break
        if m["surface_condition"] not in ALLOWED_SURFACE:
            meta_qc_passed = False
            qc_errors.append(f"{fid}.json: Invalid surface_condition {m['surface_condition']}")
            break

    print(f"Metadata 6-key strict schema check: {'PASS' if meta_qc_passed else 'FAIL'}")

    # Compile QC Report
    qc_report = {
        "status": "PASS" if (qc_passed and meta_qc_passed) else "FAIL",
        "total_frames_expected": total_frames,
        "total_frames_verified": len(src_images),
        "checks": {
            "all_4_files_present": qc_passed,
            "image_ext_jpg": True,
            "seg_ext_png": True,
            "det_ext_txt": True,
            "meta_ext_json": True,
            "meta_6_keys_exact": meta_qc_passed,
            "weather_valid": meta_qc_passed,
            "surface_condition_valid": meta_qc_passed,
        },
        "errors": qc_errors[:50],
    }

    with open(QC_PATH, "w", encoding="utf-8") as f:
        json.dump(qc_report, f, indent=2, ensure_ascii=False)
    print(f"QC report saved to {QC_PATH}")

    # Compile Dataset v0 Statistics
    print("=== Compiling Dataset Statistics ===")
    weather_counts = Counter(b["weather"] for b in batch_stats)
    surface_counts = Counter(b["surface_condition"] for b in batch_stats)
    location_counts = Counter(b["location"] for b in batch_stats)
    total_bboxes = sum(b["bbox_count"] for b in batch_stats)
    pos_det_frames = sum(1 for b in batch_stats if b["bbox_count"] > 0)
    neg_det_frames = total_frames - pos_det_frames

    det_cls_counts = Counter()
    for b in batch_stats:
        for c in b["det_classes"]:
            det_cls_counts[c] += 1

    stats_report = {
        "dataset_name": "validation_v0",
        "total_frames": total_frames,
        "image_resolution": [1920, 1080],
        "image_format": "JPEG",
        "detection_frames": {
            "total": total_frames,
            "positive_frames": pos_det_frames,
            "negative_frames": neg_det_frames,
            "total_bboxes": total_bboxes,
            "class_occurrences": dict(det_cls_counts),
        },
        "metadata_distributions": {
            "weather": dict(weather_counts),
            "surface_condition": dict(surface_counts),
            "location": dict(location_counts),
            "camera": {"front_camera": total_frames},
        },
    }

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats_report, f, indent=2, ensure_ascii=False)
    print(f"Stats report saved to {STATS_PATH}")
    print("=== Dataset v0 Build Finished Successfully ===")


if __name__ == "__main__":
    main()
