#!/usr/bin/env python3
"""Run the just-trained PIDNet-S + RTMDet-s checkpoints on a handful of
never-before-used frames (from validation_v0, excluding every golden_set
id) and draw side-by-side prediction overlays, so the actual model quality
can be visually judged before deciding whether to scale up training data.
Writes to <PROJECT_ROOT>/model_set/{images,segmentation,detection,draw}.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.pidnet import build_pidnet_s
from models.rtmdet import build_rtmdet_s

IMG_DIR = PROJECT_ROOT / "dataset" / "validation_v0" / "images"
OUT_DIR = PROJECT_ROOT / "model_set"
IDS_FILE = Path("/tmp/model_set_ids2.txt")

SEG_IMG_SIZE = (288, 512)   # H, W -- matches train_segmentation.yaml
DET_IMG_SIZE = (640, 640)   # H, W -- matches train_detection.yaml

SEG_COLORS = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}  # BGR: drivable/caution/non_drivable
DET_CLASS_NAME = {0: "step", 1: "ditch_hole", 2: "puddle", 3: "obstacle"}
DET_COLORS = {0: (255, 0, 255), 1: (255, 0, 0), 2: (255, 255, 0), 3: (0, 165, 255)}

# detection_v0 (validation_v0's ~15k-frame pool, puddle/obstacle only) is
# the current best detector (mAP50 0.245) -- and unlike the old golden_set
# -only model, it actually clears its own real 0.25 threshold now, so we
# can display at the real deployment threshold instead of an artificial
# low one just to see anything at all.
CONF_THRESH_DISPLAY = 0.25


def load_models(device):
    seg_ckpt = torch.load(PROJECT_ROOT / "outputs/checkpoints/segmentation/best.pt", map_location=device)
    seg_model = build_pidnet_s(num_classes=3, m=32, class_weights=[2.0, 10.0, 0.5]).to(device)
    seg_model.load_state_dict(seg_ckpt["model_state"])
    seg_model.eval()

    det_ckpt = torch.load(PROJECT_ROOT / "outputs/checkpoints/detection_v0/best.pt", map_location=device)
    det_model = build_rtmdet_s(num_classes=4, conf_thresh=CONF_THRESH_DISPLAY).to(device)
    det_model.load_state_dict(det_ckpt["model_state"])
    det_model.conf_threshold = CONF_THRESH_DISPLAY
    det_model.eval()
    return seg_model, det_model


def run_segmentation(model, image_bgr, device):
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (SEG_IMG_SIZE[1], SEG_IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    pred_full = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    return pred_full


def run_detection(model, image_bgr, device):
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (DET_IMG_SIZE[1], DET_IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        preds = model(tensor)[0]
    boxes = preds["boxes"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()
    # boxes are in DET_IMG_SIZE pixel space -> normalize -> scale to original size
    # keep only the top-K by score for the visual -- at this training stage
    # conf_thresh=0.03 lets through hundreds of near-noise boxes; showing
    # all of them makes the overlay unreadable and isn't the point of this
    # spot-check (raw count is still printed separately).
    order = np.argsort(-scores)[:15]
    boxes, scores, labels = boxes[order], scores[order], labels[order]

    dets = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        x1n, x2n = x1 / DET_IMG_SIZE[1], x2 / DET_IMG_SIZE[1]
        y1n, y2n = y1 / DET_IMG_SIZE[0], y2 / DET_IMG_SIZE[0]
        dets.append({"class": DET_CLASS_NAME[int(label)], "score": float(score),
                     "box": [x1n * w, y1n * h, x2n * w, y2n * h]})
    return dets


def draw_overlay(image_bgr, seg_mask, dets):
    color_mask = np.zeros_like(image_bgr)
    for cls_id, color in SEG_COLORS.items():
        color_mask[seg_mask == cls_id] = color
    result = cv2.addWeighted(image_bgr, 0.65, color_mask, 0.35, 0)
    for det in dets:
        x1, y1, x2, y2 = map(int, det["box"])
        color = DET_COLORS[[k for k, v in DET_CLASS_NAME.items() if v == det["class"]][0]]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class']} {det['score']:.2f}"
        cv2.putText(result, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return np.hstack([image_bgr, result])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model, det_model = load_models(device)

    for sub in ("images", "segmentation", "detection", "draw"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    frame_ids = IDS_FILE.read_text().split()
    for frame_id in frame_ids:
        img_path = IMG_DIR / f"{frame_id}.jpg"
        image_bgr = cv2.imread(str(img_path))
        seg_pred = run_segmentation(seg_model, image_bgr, device)
        dets = run_detection(det_model, image_bgr, device)

        cv2.imwrite(str(OUT_DIR / "images" / f"{frame_id}.jpg"), image_bgr)
        cv2.imwrite(str(OUT_DIR / "segmentation" / f"{frame_id}.png"), seg_pred)
        det_lines = []
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            h, w = image_bgr.shape[:2]
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            cls_id = [k for k, v in DET_CLASS_NAME.items() if v == d["class"]][0]
            det_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {d['score']:.4f}")
        (OUT_DIR / "detection" / f"{frame_id}.txt").write_text("\n".join(det_lines) + ("\n" if det_lines else ""))

        overlay = draw_overlay(image_bgr, seg_pred, dets)
        cv2.imwrite(str(OUT_DIR / "draw" / f"{frame_id}_PRED_overlay.jpg"), overlay)
        print(f"{frame_id}: seg unique={np.unique(seg_pred).tolist()} dets shown(top15)={len(dets)}")

    print(f"\ndone -> {OUT_DIR}")


if __name__ == "__main__":
    main()
