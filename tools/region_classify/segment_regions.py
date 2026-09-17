"""Step 1: split every frame into numbered regions with SAM2 automatic masks.

usage (must run with the X-AnyLabeling-Server venv, which ships sam2):
  /home/hs/rang/X-AnyLabeling-Server/.venv/bin/python segment_regions.py <ids_file> <work_dir>

writes, per frame:
  <work_dir>/regions/<frame_id>.png   16-bit region id map, full resolution
  <work_dir>/marks/<frame_id>.jpg     image with region outlines + numbers (shown to the AI)
  <work_dir>/regions/<frame_id>.json  per-region area/bbox/label position

Problems seen on real frames and how they are handled:
  - SAM2 leaves large areas with no mask at all (half a concrete road, a whole
    grass bank). Leftover pixels are grouped into their own regions, and big
    leftovers are split by color+position so they stay classifiable.
  - Night frames produce almost no masks. Dark frames get a contrast boost
    (CLAHE) for segmentation only; the AI still sees the original image.
  - Many tiny fragments. Regions below MIN_AREA_FRAC merge into the
    neighbor they share the longest border with, so the AI gets a readable
    number of regions.
The GPU is shared with the X-AnyLabeling server, so segmentation runs at half
resolution (SAM2 works at 1024px internally anyway) and the id map is
upscaled back to full resolution.
"""
import glob
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import V0_IMAGES
from common import read_ids
from common import save_region_map
from common import write_json

SAM2_DIR = Path("/home/hs/rang/X-AnyLabeling-Server/app/models")
SAM2_CKPT_GLOB = "/home/hs/.cache/huggingface/hub/models--facebook--sam2.1-hiera-small/snapshots/*/sam2.1_hiera_small.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

WORK_SIZE = (960, 540)
DARK_MEAN_GRAY = 70
MIN_AREA_FRAC = 0.0015      # smaller regions merge into a neighbor
BIG_LEFTOVER_FRAC = 0.04    # leftover areas bigger than this get split
MAX_REGIONS = 60


def build_generator():
    sys.path.insert(0, str(SAM2_DIR))
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    # build_sam2 resolves its config relative to the sam2 package's parent dir
    import os
    os.chdir(SAM2_DIR)
    model = build_sam2(SAM2_CFG, glob.glob(SAM2_CKPT_GLOB)[0], device="cuda")
    return SAM2AutomaticMaskGenerator(
        model,
        points_per_side=32,
        points_per_batch=16,
        pred_iou_thresh=0.75,
        stability_score_thresh=0.88,
        min_mask_region_area=300,
    )


def enhance_if_dark(image_bgr: np.ndarray) -> np.ndarray:
    gray_mean = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).mean()
    if gray_mean >= DARK_MEAN_GRAY:
        return image_bgr
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def paint_masks(masks: list, shape: tuple) -> np.ndarray:
    """Largest first, so smaller (finer) masks overwrite the coarse ones."""
    region_map = np.zeros(shape, dtype=np.int32)
    next_id = 1
    for m in sorted(masks, key=lambda m: -m["area"]):
        region_map[m["segmentation"]] = next_id
        next_id += 1
    return region_map


def fill_leftovers(region_map: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    h, w = region_map.shape
    total = h * w
    next_id = region_map.max() + 1
    unassigned = (region_map == 0).astype(np.uint8)
    n, comp = cv2.connectedComponents(unassigned, connectivity=8)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    for c in range(1, n):
        pix = comp == c
        area = int(pix.sum())
        if area < MIN_AREA_FRAC * total or area <= BIG_LEFTOVER_FRAC * total:
            region_map[pix] = next_id
            next_id += 1
            continue
        k = min(8, math.ceil(area / (BIG_LEFTOVER_FRAC * total)))
        pos_scale = 255.0 / max(h, w)
        feats = np.stack([lab[pix][:, 0], lab[pix][:, 1], lab[pix][:, 2],
                          xs[pix] * pos_scale, ys[pix] * pos_scale], axis=1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, cluster, _ = cv2.kmeans(feats, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
        sub = np.zeros((h, w), dtype=np.int32)
        sub[pix] = cluster.ravel() + 1
        # a color cluster can be scattered; give each connected piece its own id
        for label in range(1, k + 1):
            m, pieces = cv2.connectedComponents((sub == label).astype(np.uint8), connectivity=8)
            for p in range(1, m):
                region_map[pieces == p] = next_id
                next_id += 1
    return region_map


def merge_small(region_map: np.ndarray, min_area: int) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    while True:
        ids, counts = np.unique(region_map, return_counts=True)
        small = [(c, i) for i, c in zip(ids, counts) if i != 0 and c < min_area]
        if not small:
            return region_map
        changed = False
        for _, rid in sorted(small):
            mask = region_map == rid
            if not mask.any():
                continue
            ring = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) & ~mask
            neighbors, n_counts = np.unique(region_map[ring], return_counts=True)
            keep = neighbors != 0
            if not keep.any():
                continue
            region_map[mask] = neighbors[keep][np.argmax(n_counts[keep])]
            changed = True
        if not changed:
            return region_map


def relabel(region_map: np.ndarray) -> np.ndarray:
    out = np.zeros_like(region_map)
    for new_id, old_id in enumerate([i for i in np.unique(region_map) if i != 0], start=1):
        out[region_map == old_id] = new_id
    return out


def segment_frame(generator, image_full: np.ndarray) -> np.ndarray:
    work = cv2.resize(image_full, WORK_SIZE, interpolation=cv2.INTER_AREA)
    seg_input = enhance_if_dark(work)
    rgb = cv2.cvtColor(seg_input, cv2.COLOR_BGR2RGB)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks = generator.generate(rgb)
    region_map = paint_masks(masks, work.shape[:2])
    region_map = fill_leftovers(region_map, seg_input)
    total = region_map.size
    min_area = int(MIN_AREA_FRAC * total)
    region_map = merge_small(region_map, min_area)
    while len(np.unique(region_map)) - 1 > MAX_REGIONS:
        min_area = int(min_area * 1.5)
        region_map = merge_small(region_map, min_area)
    region_map = relabel(region_map)
    h, w = image_full.shape[:2]
    return cv2.resize(region_map.astype(np.uint16), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)


def label_position(mask: np.ndarray) -> tuple[int, int]:
    """Point deepest inside the region, so the number never sits on a border."""
    dist = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return int(x), int(y)


def render_marks(image_full: np.ndarray, region_map: np.ndarray) -> tuple[np.ndarray, list]:
    h, w = region_map.shape
    vis = image_full.copy()
    edges = np.zeros((h, w), dtype=np.uint8)
    edges[:, 1:] |= region_map[:, 1:] != region_map[:, :-1]
    edges[1:, :] |= region_map[1:, :] != region_map[:-1, :]
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8)).astype(bool)
    vis[edges] = (255, 255, 255)
    info = []
    for rid in range(1, region_map.max() + 1):
        mask = region_map == rid
        area = int(mask.sum())
        if area == 0:
            continue
        ys, xs = np.nonzero(mask)
        x, y = label_position(mask)
        text = str(rid)
        scale = 0.9
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x0, y0 = max(0, x - tw // 2 - 4), max(0, y - th // 2 - 4)
        x1, y1 = min(w - 1, x0 + tw + 8), min(h - 1, y0 + th + 8)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.putText(vis, text, (x0 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 2, cv2.LINE_AA)
        info.append({
            "id": rid,
            "area_frac": round(area / (h * w), 5),
            "bbox_norm": [round(xs.min() / w, 4), round(ys.min() / h, 4), round(xs.max() / w, 4), round(ys.max() / h, 4)],
            "label_xy_norm": [round(x / w, 4), round(y / h, 4)],
        })
    return vis, info


def main():
    ids = read_ids(Path(sys.argv[1]))
    work_dir = Path(sys.argv[2]).resolve()
    img_dir = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else V0_IMAGES
    generator = build_generator()
    done = 0
    for frame_id in ids:
        out_png = work_dir / "regions" / f"{frame_id}.png"
        if out_png.exists():
            continue
        image = cv2.imread(str(img_dir / f"{frame_id}.jpg"))
        if image is None:
            print(f"{frame_id}: MISSING image", flush=True)
            continue
        region_map = segment_frame(generator, image)
        save_region_map(out_png, region_map)
        vis, info = render_marks(image, region_map)
        (work_dir / "marks").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(work_dir / "marks" / f"{frame_id}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        write_json(work_dir / "regions" / f"{frame_id}.json", {"frame_id": frame_id, "regions": info})
        done += 1
        print(f"{frame_id}: {len(info)} regions", flush=True)
    print(f"=== segmented {done} ===", flush=True)


if __name__ == "__main__":
    main()
