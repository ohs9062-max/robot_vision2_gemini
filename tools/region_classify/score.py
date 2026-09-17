"""Step 4: score an assembled dataset against a hand-labeled reference set.

usage: python3 score.py <pred_dataset_dir> <ref_dataset_dir> [out_json]

Compares every frame present in both. The reference is the hand-labeled
golden set; it is never modified here.

Segmentation: pixel confusion matrix, per-class IoU, mIoU, plus per-frame mIoU
so the worst frames can be looked at first.
Detection: per-class precision / recall with greedy matching at IoU >= 0.5.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SEG_NAMES = ["drivable", "caution", "non_drivable"]
DET_NAMES = ["step", "ditch_hole", "puddle", "obstacle"]
MATCH_IOU = 0.5


def iou_per_class(conf: np.ndarray) -> list:
    out = []
    for c in range(len(SEG_NAMES)):
        tp = conf[c, c]
        denom = conf[c, :].sum() + conf[:, c].sum() - tp
        out.append(float(tp / denom) if denom else None)
    return out


def read_boxes(path: Path) -> list:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, bw, bh = int(float(parts[0])), *map(float, parts[1:])
        boxes.append((c, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return boxes


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0.0, min(a[4], b[4]) - max(a[2], b[2]))
    inter = ix * iy
    union = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def main():
    pred_root, ref_root = Path(sys.argv[1]), Path(sys.argv[2])
    out_json = Path(sys.argv[3]) if len(sys.argv) > 3 else pred_root / "score.json"

    pred_ids = {p.stem for p in (pred_root / "segmentation").glob("*.png")}
    ref_ids = {p.stem for p in (ref_root / "segmentation").glob("*.png")}
    ids = sorted(pred_ids & ref_ids)
    if not ids:
        sys.exit("no frames in common")

    conf = np.zeros((3, 3), dtype=np.int64)
    per_frame = []
    det_tp = {n: 0 for n in DET_NAMES}
    det_fp = {n: 0 for n in DET_NAMES}
    det_fn = {n: 0 for n in DET_NAMES}

    for f in ids:
        pred = cv2.imread(str(pred_root / "segmentation" / f"{f}.png"), cv2.IMREAD_GRAYSCALE).astype(np.int64)
        ref = cv2.imread(str(ref_root / "segmentation" / f"{f}.png"), cv2.IMREAD_GRAYSCALE).astype(np.int64)
        if pred.shape != ref.shape:
            pred = cv2.resize(pred.astype(np.uint8), ref.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.int64)
        frame_conf = np.bincount(3 * ref.ravel() + pred.ravel(), minlength=9).reshape(3, 3)
        conf += frame_conf
        ious = [x for x in iou_per_class(frame_conf) if x is not None]
        per_frame.append({"frame_id": f, "mIoU": round(float(np.mean(ious)), 4) if ious else None,
                          "pixel_acc": round(float(np.trace(frame_conf) / frame_conf.sum()), 4)})

        pb = read_boxes(pred_root / "detection" / f"{f}.txt")
        rb = read_boxes(ref_root / "detection" / f"{f}.txt")
        used = set()
        for p in pb:
            best, best_iou = None, MATCH_IOU
            for i, r in enumerate(rb):
                if i in used or r[0] != p[0]:
                    continue
                iou = box_iou(p, r)
                if iou >= best_iou:
                    best, best_iou = i, iou
            if best is None:
                det_fp[DET_NAMES[p[0]]] += 1
            else:
                used.add(best)
                det_tp[DET_NAMES[p[0]]] += 1
        for i, r in enumerate(rb):
            if i not in used:
                det_fn[DET_NAMES[r[0]]] += 1

    class_iou = iou_per_class(conf)
    valid = [x for x in class_iou if x is not None]
    detection = {}
    for n in DET_NAMES:
        tp, fp, fn = det_tp[n], det_fp[n], det_fn[n]
        detection[n] = {"tp": tp, "fp": fp, "fn": fn,
                        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                        "recall": round(tp / (tp + fn), 4) if tp + fn else None}

    result = {
        "frames": len(ids),
        "segmentation": {
            "mIoU": round(float(np.mean(valid)), 4),
            "pixel_acc": round(float(np.trace(conf) / conf.sum()), 4),
            "class_iou": {n: (round(v, 4) if v is not None else None) for n, v in zip(SEG_NAMES, class_iou)},
            "confusion_rows_ref_cols_pred": conf.tolist(),
        },
        "detection": detection,
        "worst_frames": sorted([p for p in per_frame if p["mIoU"] is not None], key=lambda p: p["mIoU"])[:10],
        "per_frame": per_frame,
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("frames", "segmentation", "detection", "worst_frames")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
