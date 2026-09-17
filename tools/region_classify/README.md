# region_classify — SAM2 영역 분할 + AI 영역 분류 라벨링

라벨링 규칙은 `AGENTS.md` 하나에만 있습니다. 이 폴더의 프롬프트는 `AGENTS.md` §3~§7, §18
원문을 그대로 넣되, **배치를 시작할 때 한 번만 읽고** 모든 프레임에 재사용합니다 (AGENTS.md §1.2).
실행 도중 `AGENTS.md`가 바뀌면 새 프레임을 시작하지 않고 다시 읽을지 물어봅니다.
이 README에는 규칙을 적지 않습니다.

## 기존 점 방식과의 차이

| | 점 방식 (`tools/codex_sam2_pilot`) | 영역 분류 방식 (이 폴더) |
|---|---|---|
| 경계 결정 | AI가 점을 찍은 위치에 따라 SAM2가 확장 | SAM2가 AI 없이 이미지 전체를 먼저 분할 |
| AI가 하는 일 | 점 위치 선택 (사실상 범위 판단) | 번호 붙은 영역마다 클래스 선택만 |
| Detection 박스 | AI가 좌표를 직접 씀 | AI가 영역에 det 클래스를 붙이면 그 영역의 외곽 상자 |

점 방식은 비교 기준(baseline)으로 그대로 둡니다.

## 실행 순서

`W`는 작업 폴더(영역 맵, 번호 이미지, AI 답변 캐시)이고, 결과 데이터셋은 `dataset/` 아래에 둡니다.

```bash
cd tools/region_classify
W=work/policy50            # 예시
IDS=../codex_sam2_pilot/policy50_ids.txt

# 1. 영역 분할 (sam2가 설치된 X-AnyLabeling-Server venv로 실행, GPU 사용)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/hs/rang/X-AnyLabeling-Server/.venv/bin/python segment_regions.py $IDS $W

# 2. AI 영역 분류 (backend: terra-high | claude-sonnet | agy-sonnet)
python3 classify_regions.py $IDS $W terra-high 3
python3 classify_regions.py $IDS $W claude-sonnet 3

# 3. 데이터셋 조립 (AGENTS.md §8 형식 + draw + qc.json)
python3 assemble.py $IDS $W terra-high    ../../dataset/rc_terra
python3 assemble.py $IDS $W claude-sonnet ../../dataset/rc_sonnet

# 4. 채점 (사람이 직접 라벨링한 golden_set 기준)
python3 score.py ../../dataset/rc_terra  ../../dataset/golden_set
python3 score.py ../../dataset/rc_sonnet ../../dataset/golden_set
```

모든 단계는 이미 처리한 프레임을 건너뛰므로, 중간에 멈춰도 같은 명령으로 이어서 돌리면 됩니다.
다른 이미지 폴더를 쓰려면 1~2단계는 마지막 인자로 이미지 폴더를, 3단계는 이미지 폴더와 메타데이터 폴더를 추가로 넘깁니다.

## 조립 단계에서 AI 답변 위에 적용하는 것

- 촬영 차량 후드는 항상 non_drivable
- obstacle로 분류된 영역은 항상 non_drivable
- 후드 바로 앞 띠에 닿지 않는 주행 가능 영역은 non_drivable (닿는 덩어리는 전부 유지)
- metadata는 원본 값을 유지하고 `note`만 새 detection 기준으로 다시 채움
- `qc.json`에 AI가 "한 영역에 여러 종류가 섞였다"고 표시한 영역 번호(`mixed_regions`)를 남김 — SAM2 분할이 덜 된 곳이라 먼저 확인할 대상

## 알려진 한계 (2026-09-15 테스트 4장 기준)

- 야간 프레임은 대비를 올려 분할하지만, 경계가 지형보다 불빛 경계를 따라가는 경우가 있음
- SAM2가 흙길과 옆 풀을 한 영역으로 묶으면 AI는 그 영역을 한 클래스로만 줄 수 있음 (`mixed_regions`로 표시됨)
- GPU를 X-AnyLabeling 서버와 나눠 쓰므로 분할은 절반 해상도(960×540)로 하고 원본 크기로 되돌림
