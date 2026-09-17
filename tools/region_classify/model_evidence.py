"""Create PIDNet and RTMDet evidence for Terra's fixed-boundary review.

This is a labeling aid, not a model evaluation tool: model predictions are
shown as fallible hints.  Terra remains responsible for assigning semantic
classes, and it may only accept/reject RTMDet's existing box candidates.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.pidnet import build_pidnet_s
from models.rtmdet import build_rtmdet_s

SEG_SIZE = (288, 512)
DET_SIZE = (640, 640)
DET_NAMES = ("step", "ditch_hole", "puddle", "obstacle")
SEG_NAMES = ("drivable", "caution", "non_drivable")
SEG_COLORS = ((0, 200, 0), (0, 215, 255), (0, 0, 220))
DET_COLORS = ((148, 0, 211), (0, 0, 255), (0, 191, 255), (255, 69, 0))


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def tensor_for(image: np.ndarray, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0


def load_models(device: torch.device):
    seg_ckpt = torch.load(ROOT / "outputs/checkpoints/segmentation/best.pt", map_location=device)
    seg = build_pidnet_s(num_classes=3, m=32, class_weights=[2.0, 10.0, 0.5]).to(device)
    seg.load_state_dict(seg_ckpt["model_state"])
    seg.eval()

    det_ckpt = torch.load(ROOT / "outputs/checkpoints/detection_v0/best.pt", map_location=device)
    det = build_rtmdet_s(num_classes=4, conf_thresh=0.25).to(device)
    det.load_state_dict(det_ckpt["model_state"])
    det.conf_threshold = 0.25
    det.eval()
    return seg, det


def predict(seg_model, det_model, image: np.ndarray, device: torch.device) -> tuple[np.ndarray, list[dict]]:
    h, w = image.shape[:2]
    with torch.no_grad():
        seg = torch.argmax(seg_model(tensor_for(image, SEG_SIZE, device)), dim=1)[0].cpu().numpy().astype(np.uint8)
        raw = det_model(tensor_for(image, DET_SIZE, device))[0]
    seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
    boxes = raw["boxes"].cpu().numpy()
    scores = raw["scores"].cpu().numpy()
    labels = raw["labels"].cpu().numpy()
    order = np.argsort(-scores)[:15]
    candidates = []
    for index, (box, score, label) in enumerate(zip(boxes[order], scores[order], labels[order])):
        x1, y1, x2, y2 = box
        candidates.append({
            "index": index,
            "class": DET_NAMES[int(label)],
            "score": round(float(score), 4),
            "box": [round(float(x1 / DET_SIZE[1]), 6), round(float(y1 / DET_SIZE[0]), 6),
                    round(float(x2 / DET_SIZE[1]), 6), round(float(y2 / DET_SIZE[0]), 6)],
        })
    return seg, candidates


def draw_evidence(image: np.ndarray, seg: np.ndarray, candidates: list[dict]) -> np.ndarray:
    colored = np.zeros_like(image)
    for class_id, color in enumerate(SEG_COLORS):
        colored[seg == class_id] = color
    result = cv2.addWeighted(image, 0.68, colored, 0.32, 0)
    h, w = image.shape[:2]
    for candidate in candidates:
        x1, y1, x2, y2 = candidate["box"]
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
        class_id = DET_NAMES.index(candidate["class"])
        color = DET_COLORS[class_id]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result, f"D{candidate['index']} {candidate['class']} {candidate['score']:.2f}",
                    (x1, max(22, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return result


def main() -> None:
    ids_path, out_root = Path(sys.argv[1]), Path(sys.argv[2])
    image_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "dataset/golden_set/images"
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model, det_model = load_models(device)
    for frame_id in read_ids(ids_path):
        json_path = out_root / f"{frame_id}.json"
        image_path = image_dir / f"{frame_id}.jpg"
        if json_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        seg, candidates = predict(seg_model, det_model, image, device)
        json_path.write_text(json.dumps({"pidnet_classes": SEG_NAMES, "rtmdet_candidates": candidates}, ensure_ascii=False, indent=2) + "\n")
        cv2.imwrite(str(out_root / f"{frame_id}.jpg"), draw_evidence(image, seg, candidates))
        print(f"{frame_id}: candidates={len(candidates)}", flush=True)


if __name__ == "__main__":
    main()
