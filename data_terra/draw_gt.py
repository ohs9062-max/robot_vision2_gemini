# draw_gt.py

from pathlib import Path

import cv2
import numpy as np


# =========================================================
# 설정
# =========================================================

DATA_DIR = Path(".")

IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "segmentation"
LABEL_DIR = DATA_DIR / "detection"

# draw 폴더 확인 → 없으면 자동 생성
OUTPUT_DIR = DATA_DIR / "draw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 클래스 정의
# =========================================================

# Detection
CLASS_NAMES = {
    0: "step",
    1: "ditch_hole",
    2: "puddle",
    3: "obstacle",
}

# OpenCV는 BGR

# Segmentation
SEG_CLASS_NAMES = {
    0: "drivable",
    1: "caution",
    2: "non_drivable",
}

SEG_COLORS = {
    0: (0, 255, 0),      # drivable = 초록
    1: (0, 255, 255),    # caution = 노랑
    2: (0, 0, 255),      # non_drivable = 빨강
}

# Detection
BOX_COLORS = {
    0: (255, 0, 255),    # step = 보라
    1: (255, 0, 0),      # ditch_hole = 파랑
    2: (255, 255, 0),    # puddle = 청록
    3: (0, 165, 255),    # obstacle = 주황
}


# =========================================================
# 프레임 1개 처리
# =========================================================

def draw_gt(image_path: Path):

    frame_id = image_path.stem

    mask_path = MASK_DIR / f"{frame_id}.png"
    label_path = LABEL_DIR / f"{frame_id}.txt"
    output_path = OUTPUT_DIR / f"{frame_id}_GT_overlay.jpg"

    # -----------------------------------------------------
    # 원본 이미지
    # -----------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"이미지를 읽을 수 없음: {image_path}"
        )

    height, width = image.shape[:2]

    # -----------------------------------------------------
    # GT Segmentation
    # -----------------------------------------------------

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise FileNotFoundError(
            f"마스크를 읽을 수 없음: {mask_path}"
        )

    if mask.shape != (height, width):
        raise ValueError(
            f"{frame_id}: 이미지/마스크 크기 불일치 "
            f"image={width}x{height}, "
            f"mask={mask.shape[::-1]}"
        )

    unique_values = np.unique(mask)

    if not set(unique_values).issubset({0, 1, 2}):
        raise ValueError(
            f"{frame_id}: 잘못된 segmentation 값 "
            f"{unique_values}"
        )

    color_mask = np.zeros_like(image)

    for class_id, color in SEG_COLORS.items():
        color_mask[mask == class_id] = color

    # 원본 + GT mask
    result = cv2.addWeighted(
        image,
        0.65,
        color_mask,
        0.35,
        0,
    )

    # -----------------------------------------------------
    # GT Detection
    # -----------------------------------------------------

    if label_path.exists():

        raw_text = label_path.read_text(
            encoding="utf-8"
        ).strip()

        if raw_text:

            for line in raw_text.splitlines():

                if not line.strip():
                    continue

                parts = line.split()

                if len(parts) != 5:
                    raise ValueError(
                        f"{frame_id}: 잘못된 YOLO 라벨 "
                        f"{line}"
                    )

                class_id, cx, cy, bw, bh = map(
                    float,
                    parts,
                )

                class_id = int(class_id)

                if class_id not in CLASS_NAMES:
                    raise ValueError(
                        f"{frame_id}: 알 수 없는 "
                        f"class_id={class_id}"
                    )

                # YOLO normalized → pixel
                center_x = cx * width
                center_y = cy * height

                box_w = bw * width
                box_h = bh * height

                x1 = int(center_x - box_w / 2)
                y1 = int(center_y - box_h / 2)
                x2 = int(center_x + box_w / 2)
                y2 = int(center_y + box_h / 2)

                # 이미지 범위 보호
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))

                class_name = CLASS_NAMES[class_id]
                color = BOX_COLORS[class_id]

                # Bounding Box
                cv2.rectangle(
                    result,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3,
                )

                # Label
                label = f"GT {class_name}"

                (tw, th), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2,
                )

                label_y1 = max(
                    0,
                    y1 - th - baseline - 8,
                )

                label_y2 = (
                    label_y1
                    + th
                    + baseline
                    + 8
                )

                label_x2 = min(
                    width - 1,
                    x1 + tw + 10,
                )

                cv2.rectangle(
                    result,
                    (x1, label_y1),
                    (label_x2, label_y2),
                    color,
                    -1,
                )

                cv2.putText(
                    result,
                    label,
                    (x1 + 5, label_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

    # -----------------------------------------------------
    # 범례
    # -----------------------------------------------------

    legend_items = [
        ("GT Drivable", SEG_COLORS[0]),
        ("GT Caution", SEG_COLORS[1]),
        ("GT Non-drivable", SEG_COLORS[2]),
        ("GT Step", BOX_COLORS[0]),
        ("GT Ditch Hole", BOX_COLORS[1]),
        ("GT Puddle", BOX_COLORS[2]),
        ("GT Obstacle", BOX_COLORS[3]),
    ]

    overlay = result.copy()

    legend_x = 15
    legend_y = 15
    line_h = 30
    legend_w = 280

    legend_h = (
        15
        + len(legend_items) * line_h
        + 10
    )

    cv2.rectangle(
        overlay,
        (legend_x, legend_y),
        (
            legend_x + legend_w,
            legend_y + legend_h,
        ),
        (0, 0, 0),
        -1,
    )

    result = cv2.addWeighted(
        overlay,
        0.35,
        result,
        0.65,
        0,
    )

    for i, (text, color) in enumerate(
        legend_items
    ):

        y = (
            legend_y
            + 30
            + i * line_h
        )

        cv2.rectangle(
            result,
            (legend_x + 10, y - 15),
            (legend_x + 30, y + 5),
            color,
            -1,
        )

        cv2.putText(
            result,
            text,
            (legend_x + 40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # -----------------------------------------------------
    # 좌우 2분할 (좌: 원본 / 우: GT overlay) 후 저장
    # -----------------------------------------------------

    side_by_side = np.hstack([image, result])

    if not cv2.imwrite(
        str(output_path),
        side_by_side,
    ):
        raise RuntimeError(
            f"이미지 저장 실패: {output_path}"
        )

    return output_path


# =========================================================
# 전체 프레임 실행
# =========================================================

image_files = sorted(
    IMAGE_DIR.glob("frame_*.jpg")
)

if not image_files:
    raise RuntimeError(
        f"이미지가 없습니다: {IMAGE_DIR}"
    )

total = len(image_files)

print("=" * 60)
print("GT Overlay 전체 생성 시작")
print(f"전체 이미지: {total:,}개")
print(f"출력 위치: {OUTPUT_DIR}")
print("=" * 60)


success_count = 0
error_count = 0


for index, image_path in enumerate(
    image_files,
    start=1,
):

    try:

        output_path = draw_gt(image_path)

        success_count += 1

        print(
            f"[{index:05d}/{total:05d}] "
            f"완료: {output_path.name}"
        )

    except Exception as error:

        error_count += 1

        print(
            f"[{index:05d}/{total:05d}] "
            f"ERROR: {image_path.name}"
        )

        print(
            f"    -> {error}"
        )


print()
print("=" * 60)
print("GT Overlay 전체 생성 완료")
print(f"성공: {success_count:,}")
print(f"실패: {error_count:,}")
print(f"총계: {total:,}")
print(f"출력: {OUTPUT_DIR}")
print("=" * 60)
