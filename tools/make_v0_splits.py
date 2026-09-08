#!/usr/bin/env python3
"""Create sequence-aware Train / Validation / Test splits for Dataset v0.

Prevents data leakage across contiguous camera frames by grouping by sequence ID.
Source metadata: /home/hs/rang/robot_vision/dataset/validation/metadata (READ ONLY)
Target output: /home/hs/rang/robot_vision2_gemini/data/splits/v0_splits.json
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

SRC_META_DIR = Path("/home/hs/rang/robot_vision/dataset/validation/metadata")
OUTPUT_PATH = Path("/home/hs/rang/robot_vision2_gemini/data/splits/v0_splits.json")


def main():
    meta_files = sorted(list(SRC_META_DIR.glob("*.json")))
    print(f"Reading metadata from {len(meta_files)} files...")
    assert len(meta_files) == 15000, f"Expected 15,000 files, got {len(meta_files)}"

    seq_to_frames = defaultdict(list)
    for p in meta_files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        fid = data["frame_id"]
        source_fn = data.get("source_filename", "")
        # Extract sequence prefix, e.g., '1_1_1_08161442_000064.png' -> '1_1_1_08161442'
        parts = source_fn.rsplit("_", 1)
        seq_id = parts[0] if len(parts) == 2 else "unknown_seq"
        seq_to_frames[seq_id].append(fid)

    unique_seqs = sorted(list(seq_to_frames.keys()))
    print(f"Extracted {len(unique_seqs)} unique sequences across {len(meta_files)} frames.")

    # Deterministic sequence split: 80% train, 10% validation, 10% test
    random.seed(42)
    shuffled_seqs = list(unique_seqs)
    random.shuffle(shuffled_seqs)

    train_end = int(len(shuffled_seqs) * 0.8)
    val_end = train_end + int(len(shuffled_seqs) * 0.1)

    train_seqs = set(shuffled_seqs[:train_end])
    val_seqs = set(shuffled_seqs[train_end:val_end])
    test_seqs = set(shuffled_seqs[val_end:])

    train_frames = []
    val_frames = []
    test_frames = []

    for seq in train_seqs:
        train_frames.extend(seq_to_frames[seq])
    for seq in val_seqs:
        val_frames.extend(seq_to_frames[seq])
    for seq in test_seqs:
        test_frames.extend(seq_to_frames[seq])

    train_frames.sort()
    val_frames.sort()
    test_frames.sort()

    print(f"Train: {len(train_seqs)} seqs | {len(train_frames)} frames ({len(train_frames)/len(meta_files)*100:.2f}%)")
    print(f"Val  : {len(val_seqs)} seqs | {len(val_frames)} frames ({len(val_frames)/len(meta_files)*100:.2f}%)")
    print(f"Test : {len(test_seqs)} seqs | {len(test_frames)} frames ({len(test_frames)/len(meta_files)*100:.2f}%)")

    # Verification: Mutual exclusivity
    assert len(set(train_frames) & set(val_frames)) == 0, "Train and Val overlap!"
    assert len(set(train_frames) & set(test_frames)) == 0, "Train and Test overlap!"
    assert len(set(val_frames) & set(test_frames)) == 0, "Val and Test overlap!"
    assert len(train_frames) + len(val_frames) + len(test_frames) == len(meta_files), "Missing frames in split!"

    split_info = {
        "seed": 42,
        "total_frames": len(meta_files),
        "total_sequences": len(unique_seqs),
        "train_seqs_count": len(train_seqs),
        "val_seqs_count": len(val_seqs),
        "test_seqs_count": len(test_seqs),
        "train_frames_count": len(train_frames),
        "val_frames_count": len(val_frames),
        "test_frames_count": len(test_frames),
        "train_frames": train_frames,
        "val_frames": val_frames,
        "test_frames": test_frames,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(f"Successfully saved sequence-aware split to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
