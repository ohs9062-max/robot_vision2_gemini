#!/usr/bin/env python3
"""Refine drivable segmentation for 20 representative validation frames.

Follows the specified criteria:
- Connects truncated drivable roads
- Expands overly narrow drivable corridors to robot traversable bounds
- Includes flat dirt, gravel, permeable pavement, and traversable low vegetation
- Removes misclassified steep slopes, berms, and obstacles
- Keeps original images, detections, and metadata unchanged
- Generates 3-panel review overlays: [ ORIGINAL | AI 1차 라벨 | 수정 완료본 ]
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "dataset/validation_v0"
OUTPUT_DIR = PROJECT_ROOT / "dataset/validation_sample"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

# Colors for segmentation (ODT / AGENTS.md specification):
# drivable(0): Green, caution(1): Yellow, non_drivable(2): Red
SEG_COLORS = {
    0: (0, 255, 0),      # drivable     = 초록
    1: (0, 255, 255),    # caution      = 노랑
    2: (0, 0, 255),      # non_drivable = 빨강
}

# 20 selected frames and their refinement definitions
FRAMES_CONFIG = {
    "frame_000480": {
        "category": "평탄 흙길/자갈길 전방 연결",
        "add_polys": [
            [[300, 710], [650, 640], [1450, 640], [1700, 700], [1800, 740], [200, 740]]
        ],
        "remove_polys": [],
        "reason": "전방 T자 교차로의 평탄한 흙·자갈 지면이 상단에서 잘려 있어 전방 주행 가능 지면까지 확장 연결",
        "review_needed": False
    },
    "frame_000750": {
        "category": "경사면/언덕/사면 오라벨 제거",
        "add_polys": [],
        "remove_polys": [
            [[0, 540], [720, 540], [740, 650], [550, 720], [0, 700]]
        ],
        "reason": "좌측 민가 진입부의 높은 경사면(사면) 및 화단 석축이 drivable에 포함된 오류 제거(non_drivable 전환)",
        "review_needed": False
    },
    "frame_000930": {
        "category": "측면 비탈 잡목/수풀 오라벨 제거",
        "add_polys": [],
        "remove_polys": [
            [[1180, 620], [1600, 620], [1600, 900], [1480, 900], [1350, 800]]
        ],
        "reason": "도로 우측 펜스와 도로 사이의 잡목·비탈 수풀로 과확장된 영역을 제거하여 직선 흙길 경계에 밀착",
        "review_needed": False
    },
    "frame_001170": {
        "category": "교각 하부 평탄면 단절 보완",
        "add_polys": [
            [[180, 780], [420, 670], [550, 670], [480, 810], [180, 840]]
        ],
        "remove_polys": [],
        "reason": "교각 아래 좌측 평탄 도로면의 불필요한 단절을 메워 연속 주행 공간 확보",
        "review_needed": False
    },
    "frame_001350": {
        "category": "전방 포장도로 단절 연결",
        "add_polys": [
            [[160, 645], [550, 630], [580, 675], [250, 675]]
        ],
        "remove_polys": [],
        "reason": "전방 좌측으로 명확하게 이어지는 포장도로가 중간에 끊긴 문제를 해결하여 도로 끝까지 연결",
        "review_needed": False
    },
    "frame_001620": {
        "category": "갈림길 도로 연결 및 수풀 정돈",
        "add_polys": [
            [[0, 640], [380, 640], [420, 700], [0, 700]]
        ],
        "remove_polys": [
            [[960, 660], [1120, 660], [1120, 780], [960, 780]]
        ],
        "reason": "좌측 포장 갈림길 가드레일 아래 도로 연결 및 우측 전신주·빽빽한 수풀 침범 마스크 정돈",
        "review_needed": False
    },
    "frame_001641": {
        "category": "갈림길 전방 도로 단절 연결",
        "add_polys": [
            [[750, 835], [1300, 835], [1300, 970], [750, 970]]
        ],
        "remove_polys": [],
        "reason": "가드레일 아래 콘크리트 석축 턱은 보존하고, 차량 바로 앞 아스팔트 단절만 정밀 연결",
        "review_needed": True
    },
    "frame_001941": {
        "category": "갈림길 주행로 단절 연결",
        "add_polys": [
            [[1100, 780], [1380, 780], [1450, 860], [1100, 860]]
        ],
        "remove_polys": [],
        "reason": "우측 닫힌 철문 앞마당 연결 띠의 어색한 분절을 메인 주행선과 조화롭게 정돈",
        "review_needed": False
    },
    "frame_002011": {
        "category": "평탄 잔디블럭 보도면 확장",
        "add_polys": [
            [[0, 850], [600, 570], [1100, 570], [1800, 850]]
        ],
        "remove_polys": [],
        "reason": "하천 둔치 산책로의 평탄한 잔디블럭 전체를 주행 가능 구역으로 정상 통합",
        "review_needed": False
    },
    "frame_002250": {
        "category": "다리 위 및 교차로 도로 단절 연결",
        "add_polys": [
            [[600, 625], [1050, 625], [1080, 700], [580, 700]],
            [[380, 560], [780, 560], [800, 625], [550, 625]]
        ],
        "remove_polys": [],
        "reason": "다리 난간 사이 통과로 및 건너편 아스팔트 교차로 도로 연결 복원",
        "review_needed": True
    },
    "frame_002460": {
        "category": "평탄 잔디블럭/보도면 전체 확장",
        "add_polys": [
            [[0, 850], [650, 580], [1100, 580], [1780, 850]]
        ],
        "remove_polys": [],
        "reason": "중앙 좁은 띠만 남기고 좌우의 평탄한 잔디블럭 지면을 제외했던 오류를 바로잡아 전체 주행면으로 확장",
        "review_needed": False
    },
    "frame_002581": {
        "category": "우측 풀밭 오라벨 제거",
        "add_polys": [],
        "remove_polys": [
            [[1300, 900], [1920, 900], [1920, 1080], [1300, 1080]]
        ],
        "reason": "도로와 완전히 분리되어 우측 풀밭에 고립 생성되었던 불필요한 drivable 조각 제거",
        "review_needed": False
    },
    "frame_003030": {
        "category": "그림자 과삭제 도로면 복원",
        "add_polys": [
            [[520, 950], [680, 650], [750, 950]]
        ],
        "remove_polys": [],
        "reason": "좌측 수풀 그림자로 인해 도로 좌측면이 미세하게 파여 나간 것을 실제 도로선에 맞춰 복원",
        "review_needed": False
    },
    "frame_003150": {
        "category": "전방 사면 벽 차단 및 도로 연결",
        "add_polys": [
            [[60, 780], [280, 740], [330, 820], [60, 850]]
        ],
        "remove_polys": [
            [[400, 650], [1500, 650], [1500, 710], [400, 710]]
        ],
        "reason": "전방 높은 언덕 수풀 벽 밑단 침범을 차단하고 좌측 커브길을 매끄럽게 연결",
        "review_needed": False
    },
    "frame_003540": {
        "category": "작업구역 흙더미 사면 제거",
        "add_polys": [],
        "remove_polys": [
            [[1250, 480], [1920, 480], [1920, 750], [1300, 750]]
        ],
        "reason": "우측 주황 드럼통 너머의 불규칙한 작업 흙더미 사면을 완전히 제거하고 메인 비포장 주행선에 집중",
        "review_needed": False
    },
    "frame_004110": {
        "category": "평탄 자갈 공터 내부 홀 메우기",
        "add_polys": [],
        "remove_polys": [],
        "close_holes": True,
        "reason": "넓고 평탄한 자갈 공터 내부에 자갈 무늬로 인해 발생한 미세 홀을 채워 연속된 평탄면으로 통합",
        "review_needed": False
    },
    "frame_004571": {
        "category": "평탄 흙길과 도로 연결부 복원",
        "add_polys": [
            [[550, 780], [750, 780], [750, 880], [550, 880]]
        ],
        "remove_polys": [],
        "reason": "좌측 평탄 흙길 공터와 우측 콘크리트 도로 사이의 통과 가능한 완만 연결부 연결",
        "review_needed": True
    },
    "frame_005811": {
        "category": "배수로 덮개/이음선 단절 연결",
        "add_polys": [
            [[450, 680], [860, 680], [900, 780], [450, 780]]
        ],
        "remove_polys": [],
        "reason": "콘크리트 임도 중간의 배수로 이음선으로 인해 두 동강 난 도로 마스크를 자연스럽게 결합",
        "review_needed": False
    },
    "frame_007620": {
        "category": "야간 비포장 도로 형상 복원",
        "add_polys": [
            [[800, 680], [1150, 600], [1450, 600], [1450, 850], [1100, 800]]
        ],
        "remove_polys": [],
        "reason": "직선 삼각형으로 뭉툭하게 잘린 야간 비포장 흙길을 실제 주행 가능 노면 형상대로 부드럽게 복원",
        "review_needed": False
    },
    "frame_010860": {
        "category": "갈림길 흙길 및 소실점 연결",
        "add_polys": [
            [[0, 610], [380, 610], [380, 665], [0, 665]],
            [[950, 535], [1010, 535], [1025, 565], [950, 565]]
        ],
        "remove_polys": [],
        "reason": "좌측의 평탄한 비포장 농로 갈림길 및 전방 직진 도로 소실점 부근 연결 누락 복구",
        "review_needed": False
    },
}


def make_color_overlay(base_bgr: np.ndarray, seg_mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Create color overlay: drivable=Green, caution=Yellow, non_drivable=Red."""
    color_mask = np.zeros_like(base_bgr)
    for class_id, color in SEG_COLORS.items():
        color_mask[seg_mask == class_id] = color
    return cv2.addWeighted(base_bgr, 1.0 - alpha, color_mask, alpha, 0)


def create_3panel_overlay(img_orig: np.ndarray, mask_baseline: np.ndarray,
                          mask_refined: np.ndarray, frame_id: str,
                          category: str, tile_w: int = 640, tile_h: int = 360) -> np.ndarray:
    """Generate side-by-side 3-panel review overlay:
    [ ORIGINAL | AI 1차 라벨 | 수정 완료본 ]
    """
    p1 = cv2.resize(img_orig, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    m_base = cv2.resize(mask_baseline, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
    m_ref = cv2.resize(mask_refined, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)

    p2 = make_color_overlay(p1.copy(), m_base, alpha=0.35)
    p3 = make_color_overlay(p1.copy(), m_ref, alpha=0.35)

    # Top banners for each panel
    banner_h = 32
    for p, title, color in [
        (p1, "1. 원본", (255, 255, 255)),
        (p2, "2. AI 1차 라벨", (0, 255, 255)),
        (p3, "3. 수정 완료본", (0, 255, 0))
    ]:
        overlay = p.copy()
        cv2.rectangle(overlay, (0, 0), (tile_w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, p, 0.2, 0, p)
        cv2.putText(p, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

    combined = np.hstack([p1, p2, p3])

    # Bottom summary footer
    footer_h = 32
    footer = np.zeros((footer_h, combined.shape[1], 3), dtype=np.uint8)
    cv2.rectangle(footer, (0, 0), (combined.shape[1], footer_h), (15, 15, 15), -1)
    
    info_text = f"Frame: {frame_id}  |  개선 유형: {category}  |  Segmentation: drivable(초록), caution(노랑), non_drivable(빨강)"
    cv2.putText(footer, info_text, (15, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    final_img = np.vstack([combined, footer])
    return final_img


def main():
    print("=== Starting drivable segmentation refinement for 20 frames ===")

    # Ensure output directories
    for sub in ["images", "segmentation", "detection", "metadata", "overlays"]:
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    summary_records = []

    for fid, cfg in FRAMES_CONFIG.items():
        img_src = SOURCE_DIR / "images" / f"{fid}.jpg"
        seg_src = SOURCE_DIR / "segmentation" / f"{fid}.png"
        det_src = SOURCE_DIR / "detection" / f"{fid}.txt"
        meta_src = SOURCE_DIR / "metadata" / f"{fid}.json"

        # 1. Unchanged files copy (images, detection, metadata)
        shutil.copy2(img_src, OUTPUT_DIR / "images" / f"{fid}.jpg")
        shutil.copy2(det_src, OUTPUT_DIR / "detection" / f"{fid}.txt")
        shutil.copy2(meta_src, OUTPUT_DIR / "metadata" / f"{fid}.json")

        # 2. Read image & baseline mask
        img_orig = cv2.imread(str(img_src))
        mask_base = cv2.imread(str(seg_src), cv2.IMREAD_GRAYSCALE)
        mask_refined = mask_base.copy()

        # 3. Apply refinements
        # A. Fill holes if requested
        if cfg.get("close_holes", False):
            drivable_binary = (mask_refined == 0).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            closed = cv2.morphologyEx(drivable_binary, cv2.MORPH_CLOSE, kernel)
            mask_refined[closed == 1] = 0

        # B. Add drivable polygons
        for poly in cfg.get("add_polys", []):
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask_refined, [pts], 0)

        # C. Remove drivable polygons (set to non_drivable = 2)
        for poly in cfg.get("remove_polys", []):
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask_refined, [pts], 2)

        # 4. Save refined segmentation mask
        seg_out = OUTPUT_DIR / "segmentation" / f"{fid}.png"
        cv2.imwrite(str(seg_out), mask_refined)

        # 5. Compute statistics
        added_px = int(np.sum((mask_base != 0) & (mask_refined == 0)))
        removed_px = int(np.sum((mask_base == 0) & (mask_refined != 0)))
        total_px = mask_base.size

        # 6. Generate 3-panel review overlay
        overlay_img = create_3panel_overlay(
            img_orig=img_orig,
            mask_baseline=mask_base,
            mask_refined=mask_refined,
            frame_id=fid,
            category=cfg["category"]
        )
        overlay_out = OVERLAY_DIR / f"{fid}.jpg"
        cv2.imwrite(str(overlay_out), overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"[{fid}] {cfg['category']:25s} | Added: +{added_px:7d} px | Removed: -{removed_px:7d} px | Saved overlay")

        summary_records.append({
            "frame_id": fid,
            "category": cfg["category"],
            "reason": cfg["reason"],
            "review_needed": cfg.get("review_needed", False),
            "added_pixels": added_px,
            "removed_pixels": removed_px,
            "net_change_pct": round((added_px - removed_px) / total_px * 100, 3),
            "overlay_file": str(overlay_out)
        })

    # Save summary report JSON
    with open(OUTPUT_DIR / "refinement_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully refined all 20 frames.")
    print(f"Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
