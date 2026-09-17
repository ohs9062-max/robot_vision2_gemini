"""Shared paths and helpers for the region-classification pipeline (approach A).

Pipeline:
  1. segment_regions.py   SAM2 automatic masks -> every pixel gets a region id
                          (runs in the X-AnyLabeling-Server venv, which has sam2)
  2. classify_regions.py  AI sees the image + numbered regions and assigns each
                          region a segmentation class (+ optional detection class)
  3. assemble.py          region ids + AI labels -> images/segmentation/detection/
                          metadata/draw in the standard dataset format
  4. score.py             compare an assembled set against a hand-labeled set

The AI never decides where a boundary is -- SAM2 already fixed every region's
pixels. The AI only decides what each region is, which is the part it was
shown to do reliably. Boundary/extent errors were most of the point-based
pipeline's mistakes.
"""
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
AGENTS_MD = REPO / "AGENTS.md"
V0_IMAGES = REPO / "dataset" / "validation_v0" / "images"
V0_METADATA = REPO / "dataset" / "validation_v0" / "metadata"

SEG_CLASSES = {"drivable": 0, "caution": 1, "non_drivable": 2}
DET_CLASSES = {"step": 0, "ditch_hole": 1, "puddle": 2, "obstacle": 3}


class RulesSnapshot:
    """AGENTS.md §3-§7 and §18, read exactly once, plus how to tell if it changed.

    The prompt embeds the rules verbatim from AGENTS.md so it can never drift
    from the canonical document (AGENTS.md §0). That used to be done by
    re-reading and re-slicing the file for every single frame, so a 50-frame
    batch read the same unchanged document 50 times (AGENTS.md §1.2 forbids
    this). Now a batch loads one snapshot at start and reuses the text.

    `changed()` only calls stat() -- it never re-reads the content. When it
    reports a change, the caller must ask the user before reloading
    (AGENTS.md §1.2), never reload silently.
    """

    def __init__(self) -> None:
        stat = AGENTS_MD.stat()
        self._mtime_ns = stat.st_mtime_ns
        self._size = stat.st_size
        text = AGENTS_MD.read_text(encoding="utf-8")
        start = text.index("# 3. Segmentation Class")
        end = text.index("# 8. 최종 데이터 저장 구조")
        rules_18 = text[text.index("# 18. 애매한 항목 판단 기준"):]
        self.text = text[start:end] + "\n" + rules_18

    def changed(self) -> bool:
        """True if AGENTS.md was modified after this snapshot was taken."""
        try:
            stat = AGENTS_MD.stat()
        except FileNotFoundError:
            return True
        return stat.st_mtime_ns != self._mtime_ns or stat.st_size != self._size


def agents_rules() -> str:
    """Return AGENTS.md §3-§7 and §18 verbatim (reads the file on every call).

    Kept for one-off use. Batch code must take a single `RulesSnapshot` at
    start instead of calling this per item (AGENTS.md §1.2).
    """
    return RulesSnapshot().text


def save_region_map(path: Path, region_map: np.ndarray) -> None:
    """Region ids as a 16-bit PNG (0 = unassigned, 1..N = regions)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), region_map.astype(np.uint16))


def load_region_map(path: Path) -> np.ndarray:
    region_map = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if region_map is None:
        raise FileNotFoundError(path)
    return region_map.astype(np.int32)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_ids(ids_file: Path) -> list[str]:
    return [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
