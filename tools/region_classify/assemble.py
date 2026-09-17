"""Step 3: region ids + AI labels -> dataset in the AGENTS.md §8 format.

usage: python3 assemble.py <ids_file> <work_dir> <backend> <out_dataset_dir>

writes <out>/images, segmentation, detection, metadata, draw (AGENTS.md §8-§12, §17)
and <out>/qc.json (per-frame mixed regions, hood fallback, class fractions).

Rules applied on top of the AI's per-region answer:
  - the ego-vehicle hood is always non_drivable (AGENTS.md §18), found with the
    same SAM2 point method used by the point pipeline
  - a region labeled obstacle is always non_drivable (AGENTS.md §6)
  - the connectivity filter stays on (AGENTS.md §18): passable ground not
    connected to the area in front of the hood becomes non_drivable
  - each region the AI tagged with a detection class becomes one box, the
    tight bounding box of that region's pixels
Regions the AI did not return stay non_drivable.
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "codex_sam2_pilot"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PILOT))
from common import SEG_CLASSES
from common import V0_IMAGES
from common import V0_METADATA
from common import load_region_map
from common import read_ids
from common import read_json
from common import write_json
from hood_sam2 import detect_hood_sam2
from pipeline_v3 import _keep_components_touching_hood
from pipeline_v3 import hood_mask
from pipeline_v3 import write_detection_txt
from regen_draw_golden import draw_gt

DET_ORDER = ("step", "ditch_hole", "puddle", "obstacle")


def keep_components_touching_hood(passable: np.ndarray, hood: np.ndarray, band_px: int) -> np.ndarray:
    """Keep EVERY passable component that touches the band just above the hood.

    pipeline_v3's version keeps only the single largest touching component.
    That was fine for point-based masks (one SAM2 blob per class), but here a
    road and a reed patch beside it are separate components that both sit
    directly in front of the robot; keeping only the larger one deleted the
    whole road on frame_011379. Components that never touch the hood band are
    still dropped, which is the purpose of the filter (AGENTS.md §18).
    If nothing touches the band at all (hood boundary estimate off), fall back
    to pipeline_v3's single-largest behavior.
    """
    kernel = np.ones((band_px, band_px), np.uint8)
    near_hood = cv2.dilate(hood, kernel).astype(bool) & (hood == 0)
    n, labels = cv2.connectedComponents(passable, connectivity=8)
    touching = set(np.unique(labels[near_hood & (passable == 1)]).tolist()) - {0}
    if not touching:
        return _keep_components_touching_hood(passable, hood, band_px=band_px)
    return np.isin(labels, list(touching)).astype(np.uint8)


def build_frame(image: np.ndarray, region_map: np.ndarray, labels: dict, evidence: dict | None = None):
    h, w = region_map.shape
    seg = np.full((h, w), SEG_CLASSES["non_drivable"], dtype=np.uint8)
    detections = []
    for rid_str, v in labels["regions"].items():
        mask = region_map == int(rid_str)
        if not mask.any():
            continue
        seg_name = "non_drivable" if v.get("det") == "obstacle" else v["seg"]
        seg[mask] = SEG_CLASSES[seg_name]
        if v.get("det"):
            detections.append((v["det"], mask))

    boundary = detect_hood_sam2(image)
    hood = hood_mask(h, w, boundary=boundary)
    seg[hood == 1] = SEG_CLASSES["non_drivable"]

    band_px = 2 * max(15, int(round(min(h, w) * 0.02)))
    passable = ((seg == 0) | (seg == 1)).astype(np.uint8)
    kept = keep_components_touching_hood(passable, hood, band_px)
    seg[(passable == 1) & (kept == 0)] = SEG_CLASSES["non_drivable"]

    boxes = []
    if evidence is not None and "accepted_detections" in labels:
        candidates = {v["index"]: v for v in evidence.get("rtmdet_candidates", [])}
        for accepted in labels["accepted_detections"]:
            candidate = candidates[accepted["index"]]
            boxes.append({"class": accepted["class"], "box": candidate["box"]})
    else:
        for det_class, mask in detections:
            ys, xs = np.nonzero(mask & (hood == 0))
            if len(xs) == 0:
                continue
            boxes.append({"class": det_class, "box": [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]})
    return seg, boxes, boundary is None


def main():
    ids = read_ids(Path(sys.argv[1]))
    work_dir = Path(sys.argv[2]).resolve()
    backend = sys.argv[3]
    out = Path(sys.argv[4]).resolve()
    img_dir = Path(sys.argv[5]).resolve() if len(sys.argv) > 5 else V0_IMAGES
    meta_dir = Path(sys.argv[6]).resolve() if len(sys.argv) > 6 else V0_METADATA
    evidence_dir = Path(sys.argv[7]).resolve() if len(sys.argv) > 7 else None
    draw_ids_path = Path(sys.argv[8]).resolve() if len(sys.argv) > 8 else None
    draw_ids = set(read_ids(draw_ids_path)) if draw_ids_path else None
    for sub in ("images", "segmentation", "detection", "metadata", "draw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    qc_path = out / "qc.json"
    qc = read_json(qc_path) if qc_path.exists() else {}

    built, waiting = 0, []
    for frame_id in ids:
        label_path = work_dir / f"labels_{backend}" / f"{frame_id}.json"
        if not label_path.exists():
            waiting.append(frame_id)
            continue
        labels = read_json(label_path)
        src_img = img_dir / f"{frame_id}.jpg"
        image = cv2.imread(str(src_img))
        region_map = load_region_map(work_dir / "regions" / f"{frame_id}.png")
        evidence = read_json(evidence_dir / f"{frame_id}.json") if evidence_dir else None
        seg, boxes, hood_fallback = build_frame(image, region_map, labels, evidence)

        shutil.copy2(src_img, out / "images" / f"{frame_id}.jpg")
        cv2.imwrite(str(out / "segmentation" / f"{frame_id}.png"), seg)
        det_path = out / "detection" / f"{frame_id}.txt"
        write_detection_txt(boxes, det_path)

        # AGENTS.md §12, §18.1: keep the source metadata (location is a shoot
        # location the AI cannot see); only note is recomputed from detections.
        src_meta = meta_dir / f"{frame_id}.json"
        if src_meta.exists():
            meta = read_json(src_meta)
        else:
            meta = {"frame_id": frame_id, "location": "unknown",
                    "weather": labels.get("weather", "sunny"),
                    "surface_condition": labels.get("surface_condition", "dry"),
                    "camera": "front_camera"}
        meta["frame_id"] = frame_id
        meta["note"] = [c for c in DET_ORDER if any(b["class"] == c for b in boxes)]
        write_json(out / "metadata" / f"{frame_id}.json", meta)
        if draw_ids is None or frame_id in draw_ids:
            draw_gt(frame_id, root=out)

        qc[frame_id] = {
            "backend": backend,
            "mixed_regions": labels.get("mixed_regions", []),
            "hood_fallback": hood_fallback,
            "drivable_frac": round(float((seg == 0).mean()), 4),
            "caution_frac": round(float((seg == 1).mean()), 4),
            "boxes": [b["class"] for b in boxes],
            "reasoning": labels.get("reasoning", ""),
        }
        built += 1
    write_json(qc_path, qc)
    print(f"built {built}, waiting for labels {len(waiting)}")


if __name__ == "__main__":
    main()
