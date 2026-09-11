# AGENTS.md — Robot Vision 데이터 라벨링 작업 규칙

이 파일은 `robot_vision` 데이터 라벨링 작업을 수행하는 AI 에이전트의
작업 절차와 준수사항을 정의한다.

라벨 의미와 데이터 형식의 기준 문서는 다음 파일 하나다.

```text
AI모델_데이터라벨링.md
```

이 문서는 기준 ODT의 본문과 삽입 이미지·표·요약 인포그래픽의 내용까지 확인하여 AI 작업용 Markdown으로 옮긴 문서다.
에이전트는 ODT를 다시 해석하거나,
다른 문서의 규칙을 섞거나,
문서에 없는 별도 정책을 임의로 추가하지 않는다.

---

# 1. 작업 시작 시 반드시 수행

1. 이 `AGENTS.md` 전체를 읽는다.
2. `AI모델_데이터라벨링.md` 전체를 처음부터 끝까지 읽는다.
3. 라벨 정의, 판단 기준, 저장 구조, 파일명, 확장자, 메타데이터 형식을 확인한다.
4. 이후 모든 데이터 라벨링 작업은 `AI모델_데이터라벨링.md` 기준으로 수행한다.

사용자 지시와 `AI모델_데이터라벨링.md`가 충돌하지 않는 한
문서 내용을 임의로 바꾸지 않는다.

비슷한 이름의 과거 명세서나 다른 버전의 문서를
임의로 기준 문서로 사용하지 않는다.

## 1.1 ODT 이미지 내용 누락 금지

일반 데이터 라벨링 작업에서는 `AI모델_데이터라벨링.md`를 기준으로 작업한다.

단, ODT와 Markdown의 정합성을 다시 검증하거나
Markdown을 ODT에서 다시 생성하는 작업을 수행할 때는
문단 텍스트만 추출해서 끝내지 않는다.

반드시 ODT 안에 삽입된 모든 이미지·표·요약 인포그래픽을 실제로 열어 확인한다.

특히 ODT 마지막 요약 인포그래픽에 포함된 다음 정보가 누락되면 실패다.

```text
Segmentation 표시색
drivable     = 초록
caution      = 노랑
non_drivable = 빨강

Detection BBox 표시색
step         = 보라
ditch_hole   = 파랑
puddle       = 하늘색
obstacle     = 주황
```

ODT는 HEX/RGB 숫자값을 별도로 정의하지 않으므로
숫자 색상값을 임의로 만들어 규칙으로 추가하지 않는다.

다음 레퍼런스 이미지는 파일럿 및 Overlay 품질 검수에 사용한다.

```text
라벨링_정답_레퍼런스/
├── 00_ODT_요약_인포그래픽.png
├── 01_일반_농로.png
├── 02_얕은_물웅덩이.png
├── 03_15cm_턱.png
├── 04_25cm_턱.png
├── 05_깊은_구덩이.png
└── 06_큰_돌_장애물.png
```

예시의 Class 적용:

```text
일반 농로
→ drivable / Detection 없음

얕은 물웅덩이
→ caution + puddle

15 cm 턱
→ caution + step

25 cm 턱
→ non_drivable + step

깊은 구덩이
→ non_drivable + ditch_hole

큰 돌(장애물)
→ non_drivable + obstacle
```

---

# 2. 작업 목적

과수원 및 노지환경의 카메라 영상에서
로봇의 주행 가능 영역과 주행을 방해하는 위험요소를 라벨링한다.

판단 기준은 사람의 보행 가능 여부가 아니라
실제 로봇의 차폭, 바퀴, 서스펜션 및 주행성능이다.

```text
Detection
= 앞에 무엇이 있는가?

Segmentation
= 그 영역을 로봇이 갈 수 있는가?
```

---

# 3. Segmentation Class

```text
0 = drivable
1 = caution
2 = non_drivable
```

| ID | Class | 판단 기준 |
|---:|---|---|
| 0 | `drivable` | 정상적으로 통과 가능 |
| 1 | `caution` | 통과 가능하지만 감속 또는 주의 필요 |
| 2 | `non_drivable` | 통과 위험 또는 불가능, 회피 필요 |

대표 판단:

```text
평탄한 농로
→ drivable

일반 흙길 또는 자갈길
→ drivable

풀이 있어도 정상 통과 가능
→ drivable

큰 요철
→ caution

얕은 물웅덩이
→ caution

통과 가능한 고랑
→ caution

20 cm 이하의 턱
→ caution

깊은 고랑 또는 구덩이
→ non_drivable

깊이를 판단하기 어려운 물웅덩이
→ non_drivable

20 cm를 초과하는 턱
→ non_drivable

진행을 막는 큰 돌·나무·구조물
→ non_drivable
```

---

# 4. 턱 판단 기준

로봇은 최대 20 cm 높이의 턱을 통과할 수 있는 것으로 가정한다.

```text
20 cm 이하
→ caution
→ 감속하여 통과 가능

20 cm 초과
→ non_drivable
→ 회피 필요
```

단일 영상만으로 실제 턱 높이를 정확하게 확인하기 어려운 경우
임의로 높이를 추정하지 않는다.

---

# 5. Detection Class

```text
0 = step
1 = ditch_hole
2 = puddle
3 = obstacle
```

| ID | Class | 라벨 대상 |
|---:|---|---|
| 0 | `step` | 로봇 진행방향의 턱 |
| 1 | `ditch_hole` | 고랑 및 구덩이 |
| 2 | `puddle` | 물웅덩이 |
| 3 | `obstacle` | 돌, 나무, 구조물 등 일반 장애물 |

고랑과 구덩이는 1차년도에 `ditch_hole` 하나의 Detection Class로 관리한다.

Bounding Box는 실제 위험요소의 외곽에 최대한 밀착하여 생성하고,
불필요한 배경이 포함되지 않도록 한다.

---

# 6. Detection + Segmentation 적용 기준

| 실제 상황 | Detection | Segmentation | 주행 판단 |
|---|---|---|---|
| 15 cm 턱 | `step` | `caution` | 감속 후 통과 |
| 25 cm 턱 | `step` | `non_drivable` | 회피 |
| 얕은 물웅덩이 | `puddle` | `caution` | 감속 후 통과 |
| 깊은 물웅덩이 | `puddle` | `non_drivable` | 회피 |
| 통과 가능한 얕은 고랑 | `ditch_hole` | `caution` | 감속 후 통과 |
| 깊은 구덩이 | `ditch_hole` | `non_drivable` | 회피 |
| 진행을 막는 큰 돌 | `obstacle` | `non_drivable` | 회피 |

Detection Class를
`step_passable`, `step_non_passable`처럼 별도로 나누지 않는다.

---

# 7. 공통 라벨링 규칙

1. 사람이 아니라 실제 로봇의 주행성능을 기준으로 판단한다.
2. 영상에서 실제로 보이는 영역만 라벨링한다.
3. 가려진 지면을 임의로 추정하지 않는다.
4. 그림자는 장애물로 판단하지 않는다.
5. 그림자가 있어도 실제 주행 가능한 지면이면 `drivable`로 처리한다.
6. 흙, 풀, 자갈 등의 종류보다 실제 통과 가능 여부를 우선한다.
7. 통과 가능하지만 감속이 필요하면 `caution`으로 처리한다.
8. 통과 위험성이 높거나 불가능하면 `non_drivable`로 처리한다.
9. Bounding Box는 실제 위험요소 외곽에 최대한 밀착한다.
10. 동일하거나 유사한 상황에는 동일한 기준을 적용한다.
11. 단일 영상으로 정확한 턱 높이를 알 수 없으면 임의 추정하지 않는다.
12. 동일하거나 거의 유사한 연속 프레임을 과도하게 사용하지 않는다.
13. 30 FPS 영상에서 약 1~2 FPS 샘플링은 예시일 뿐 고정 규칙이 아니다.

---

# 8. 최종 데이터 저장 구조 — 반드시 준수

최종 데이터는 반드시 아래 4개 폴더로 분리한다.

```text
dataset/
├── images/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
├── segmentation/
│   ├── frame_000001.png
│   ├── frame_000002.png
│   └── ...
├── detection/
│   ├── frame_000001.txt
│   ├── frame_000002.txt
│   └── ...
└── metadata/
    ├── frame_000001.json
    ├── frame_000002.json
    └── ...
```

최종 파일명과 확장자는 위 형식을 그대로 사용한다.

```text
images          = frame_XXXXXX.jpg
segmentation    = frame_XXXXXX.png
detection       = frame_XXXXXX.txt
metadata        = frame_XXXXXX.json
```

임의로 `.png` 이미지로 저장하거나,
다른 확장자를 유지하거나,
파일명에 `_final`, `_fixed`, `_new` 등의 접미사를 추가하지 않는다.

---

# 9. Frame ID 규칙 — 반드시 1:1 대응

하나의 Frame과 관련된 모든 파일은 동일한 Frame ID를 사용한다.

예:

```text
images/frame_000123.jpg
segmentation/frame_000123.png
detection/frame_000123.txt
metadata/frame_000123.json
```

즉 다음 네 파일은 반드시 같은 Frame이다.

```text
frame_000123.jpg
frame_000123.png
frame_000123.txt
frame_000123.json
```

Frame ID를 임의로 재배치하거나
파일 간 대응 관계를 깨뜨리지 않는다.

---

# 10. Segmentation 저장 형식

Segmentation 결과는 이미지와 동일한 크기의 Mask 이미지로 저장한다.

파일:

```text
segmentation/frame_XXXXXX.png
```

Mask Class ID:

```text
0 = drivable
1 = caution
2 = non_drivable
```

---

# 11. Detection 저장 형식

Detection 파일:

```text
detection/frame_XXXXXX.txt
```

YOLO 형식:

```text
<class_id> <center_x> <center_y> <width> <height>
```

Class ID:

```text
0 = step
1 = ditch_hole
2 = puddle
3 = obstacle
```

---

# 12. Metadata 작성 규칙 — 양식 변경 금지

Metadata 파일:

```text
metadata/frame_XXXXXX.json
```

사용 항목:

```text
frame_id
location
weather
surface_condition
camera
note
```

JSON 구조:

```json
{
  "frame_id": "frame_000123",
  "location": "orchard_01",
  "weather": "after_rain",
  "surface_condition": "wet",
  "camera": "front_camera",
  "note": [
    "puddle",
    "step"
  ]
}
```

`weather` 값:

```text
sunny
cloudy
after_rain
```

의미:

```text
sunny      = 맑음
cloudy     = 흐림
after_rain = 비 온 후
```

`surface_condition` 값:

```text
dry
wet
```

의미:

```text
dry = 건조
wet = 젖음
```

Metadata에 `source_filename` 등
`AI모델_데이터라벨링.md`에 정의되지 않은 필드를
임의로 추가하지 않는다.

Metadata key 이름을 변경하지 않는다.

---

# 13. 데이터 양식 검증

작업 결과는 최소한 다음을 검증한다.

```text
images/frame_XXXXXX.jpg
segmentation/frame_XXXXXX.png
detection/frame_XXXXXX.txt
metadata/frame_XXXXXX.json
```

확인 항목:

- 모든 Frame의 4개 파일 존재 여부
- Frame ID 1:1 대응 여부
- image 확장자가 `.jpg`인지
- segmentation 확장자가 `.png`인지
- detection 확장자가 `.txt`인지
- metadata 확장자가 `.json`인지
- Segmentation Class가 0/1/2인지
- 검수 Overlay의 Segmentation 표시가 drivable=초록, caution=노랑, non_drivable=빨강인지
- Detection Class가 0/1/2/3인지
- 검수 Overlay의 Detection BBox 표시가 step=보라, ditch_hole=파랑, puddle=하늘색, obstacle=주황인지
- Metadata key가 정해진 6개 항목인지
- `weather` 값이 `sunny/cloudy/after_rain` 중 하나인지
- `surface_condition` 값이 `dry/wet` 중 하나인지

양식이 다르면 완료로 처리하지 않는다.

---

# 14. 금지사항

다음은 금지한다.

- `AI모델_데이터라벨링.md`에 없는 별도 라벨 정책 추가
- 다른 과거 명세서의 규칙 혼합
- ODT를 다시 해석하여 Markdown 명세와 다른 규칙 생성
- 사람의 보행 기준으로 주행 가능 여부 판단
- 가려진 지면 추정
- 그림자를 장애물로 판단
- 흙/풀/자갈 종류 자체만으로 Class 결정
- 단일 영상에서 턱 높이를 임의 추정
- Detection Class 임의 추가 또는 분리
- Bounding Box를 실제 대상보다 과도하게 크게 생성
- `images`를 `.jpg` 이외 확장자로 최종 저장
- `segmentation`을 `.png` 이외 확장자로 저장
- `detection`을 `.txt` 이외 확장자로 저장
- `metadata`를 `.json` 이외 확장자로 저장
- Frame ID 불일치
- Metadata key 임의 변경
- Metadata에 명세에 없는 필드 임의 추가
- `weather`에 정해지지 않은 값 사용
- `surface_condition`에 정해지지 않은 값 사용

---

# 15. 작업 완료 조건

다음 조건을 모두 만족해야 데이터 라벨링 작업 완료로 보고한다.

1. Segmentation이 `drivable / caution / non_drivable` 기준에 맞는다.
2. Detection이 `step / ditch_hole / puddle / obstacle` 기준에 맞는다.
3. 실제 로봇의 주행성능 기준으로 판단했다.
4. 가려진 지면을 임의 추정하지 않았다.
5. Bounding Box가 실제 위험요소 외곽에 밀착되어 있다.
6. 최종 파일 구조가 정확하다.
7. `images/frame_XXXXXX.jpg` 형식을 지켰다.
8. `segmentation/frame_XXXXXX.png` 형식을 지켰다.
9. `detection/frame_XXXXXX.txt` 형식을 지켰다.
10. `metadata/frame_XXXXXX.json` 형식을 지켰다.
11. 모든 Frame ID가 1:1 대응한다.
12. Metadata 양식과 값 정의를 지켰다.
13. 동일하거나 거의 유사한 연속 프레임을 과도하게 사용하지 않았다.

---

# 16. AI 응답 규칙

실제로 수행한 작업과 수행하지 않은 작업을 구분해서 보고한다.

문서에 없는 기준을 새로 만들었다고 보고하지 않는다.

양식 검증이 끝나지 않았으면
`완료`라고 보고하지 않는다.

작업 보고 마지막에는 반드시 다음 제목을 포함한다.

```text
해당 코드 작업에서 내가 알아야 할 것 3줄 요약
```

정확히 3줄로 작성한다.

---

# 17. 검수용 샘플/더미 데이터셋 생성 규칙

`dataset/data_harness/` 같은 검수용 샘플 데이터셋을 만들 때 (15,000장 본 데이터셋 전체 파이프라인과 별개로, 소규모 파일럿/골드셋 후보를 만들 때) 아래 4가지를 전부 지켜야 완료로 인정한다. 하나라도 빠지면 "샘플 생성 완료"로 보고하지 않는다.

## 17.1 `segmentation/*.png`는 항상 raw class-id 값이다

`§10`의 원칙(`0=drivable, 1=caution, 2=non_drivable`)은 샘플 데이터에도 그대로 적용된다. 색을 입힌 이미지나 오버레이를 `segmentation/`에 저장하지 않는다. 사람이 눈으로 보면 거의 검은 화면처럼 보이는 게 정상이다 — 이건 시각화용이 아니라 학습용 정답 마스크다.

## 17.2 사람이 눈으로 검수할 수 있는 오버레이(`draw/`)를 항상 함께 생성한다

`images/`, `segmentation/`, `detection/`을 원본 이미지 위에 색으로 겹쳐 그린 시각화를 `draw/frame_XXXXXX_GT_overlay.jpg`로 반드시 함께 만든다. 이 시각화가 없으면 사람이 raw class-id PNG만 보고는 라벨이 맞는지 검수할 수 없다.

- 색상은 `라벨링요약.png` / ODT 기준과 동일하게 맞춘다: Segmentation `drivable=초록, caution=노랑, non_drivable=빨강`, Detection `step=보라, ditch_hole=파랑, puddle=하늘색(청록), obstacle=주황`.
- 매번 새로 코드를 짜지 말고, 이미 검증된 `draw_gt.py`(예: `dataset/validation_v0/gold_samples_6/draw_gt.py`)를 해당 샘플 폴더에 그대로 복사해서 쓴다. 색상표와 렌더링 방식을 임의로 바꾸지 않는다.
- **출력은 좌우 2분할(좌: 원본 이미지, 우: GT overlay)로 한 장에 합쳐서 저장한다** (`np.hstack([원본, overlay])`, 파일명은 동일하게 `frame_XXXXXX_GT_overlay.jpg`). overlay만 단독으로 저장하지 않는다 — 원본과 나란히 있어야 무엇이 새로 라벨링됐는지 바로 비교할 수 있다.
- `draw_gt.py`는 `images/`, `segmentation/`, `detection/`을 읽어 `draw/`에 결과를 쓰는 구조이므로, 세 폴더가 먼저 올바르게 채워져 있어야 정상 동작한다.

## 17.3 `detection/*.txt`는 실제로 생성한 판단 결과여야 한다

원본(1차 데이터 등)의 `detection.txt`를 검증 없이 그대로 복사해서 새 샘플의 정답인 것처럼 두지 않는다. 원본에 detection이 비어 있어도 실제 이미지에 `step/ditch_hole/puddle/obstacle`이 보이면 그건 원본이 놓친 것이므로, 이번에 실제로 판단한 결과를 YOLO 형식(`§11`)으로 새로 써야 한다. 이 판단을 아직 하지 않았다면 해당 프레임은 "샘플 반영 완료"로 보고하지 않고, 어느 프레임의 detection이 미검증 상태인지 명시적으로 밝힌다.

## 17.4 `metadata`의 `location` 값은 일관된 형식으로 분류한다

1차 원본 데이터의 `location` 값(예: 지역명 텍스트)이 `§4`에서 예시로 든 형식(`orchard_01` 등)과 다르거나 신뢰할 수 없는 경우, 실제 촬영 좌표를 임의로 지어내지 않는다. 대신 이미지에 실제로 보이는 환경(농경지/숲길/포장도로 등)을 근거로 분류하되, **형식만은 모든 프레임에서 통일**한다. 예를 들어 `farm`, `forest`, `roadside`, `wetland`처럼 카테고리별로 다른 값이 나오는 것은 괜찮지만, 어떤 프레임은 영어 소문자로 어떤 프레임은 한글로, 또는 어떤 프레임만 번호를 붙이는 식으로 표기 스타일을 섞지 않는다.

## 17.5 식생(풀/나무) 판정은 종류로 구분한다 — "풀숲"을 한 덩어리로 취급하지 않는다

`§3`의 "풀이 있어도 정상 통과 가능 → drivable"과 "진행을 막는 큰 돌·나무·구조물 → non_drivable"은 식생을 두 종류로 더 세분해서 적용한다:

- **주행 가능한 풀숲(벼, 갈대, 억새 등 — 줄기가 얇고 잘 휘어지는 벼과 식물)**: `caution`. 로봇이 밀고 지나갈 수 있지만 시야 차단과 노면 불확실성 때문에 감속이 필요하다. 이런 지역이 보이는데 `caution_points`가 비어 있으면 라벨 누락이다.
- **굵은 가지·관목·나무(목질이라 휘어지지 않는 식생)**: `non_drivable`. 도로에 바로 붙어 있어도 `drivable`이나 `caution`으로 넓게 잡지 않는다. 차체 하부 파손·바퀴 걸림 위험이 있는 대상이므로 obstacle과 동일하게 통과 불가로 취급한다.
- 노면에 낮게 깔린 잔디·잡초(로봇 최저지상고 미만, 지면의 일부에 가까운 것)는 그대로 `drivable`이다.

**도로+흙길, 도로+풀숲처럼 두 지형이 같이 있는 프레임에서는 도로(실제 주행 경로) 영역만 정확히 `drivable`(초록)로 따고, 그 옆 풀숲 영역은 위 기준에 따라 `caution`(노랑) 또는 `non_drivable`(빨강)으로 명확히 분리한다.** "도로 옆에 있으니 같이 통과 가능해 보인다"는 느낌으로 도로와 풀숲 경계를 뭉개서 넓게 `drivable`로 잡지 않는다.

같은 "풀"이라도 화면 좌우에서 종류가 다르면(한쪽은 관목, 반대쪽은 갈대) 각각 다르게 판단한다.

---

# 18. 애매한 항목 판단 기준 — RGB 카메라만 있을 때

아래는 라벨링 중 자주 마주치는 애매한 상황에서, **수치(cm/m)를 임의로 추정하지 않고 육안으로 명확한 것만 라벨링한다**는 원칙을 구체적으로 적용한 기준이다. `§3`(Segmentation Class)와 `§5`(Detection Class)의 기본 원칙을 대체하지 않고, 그 경계 사례에 적용하는 세부 지침이다.

| 애매한 항목 | 기준 |
|---|---|
| 풀밭 | 높이를 cm로 재지 않는다. 낮고 성기며 통과 가능성이 명확하면 `drivable`. 빽빽해서 지면 상태 판단이 안 되면 억지로 정답을 만들지 않는다(해당 프레임을 골든셋에서 제외). |
| 큰 요철 | 깊이를 수치화하지 않는다. 육안으로 명확하게 거친 노면이면 `caution`. 애매하면 골든셋에서 제외한다. |
| 고랑/구덩이 | 존재 자체는 `ditch_hole` detection이 가능하다. 통과 가능 여부가 육안으로 명확할 때만 `caution`/`non_drivable`을 결정한다. |
| 물웅덩이 | 얕음이 명확하면 `caution`. 깊이 판단이 안 되면 `§3` 기준 그대로 `non_drivable`. |
| 장애물 크기 | cm 기준을 만들지 않는다. 로봇 진행을 실제로 막는 게 명확하면 `obstacle`. |
| 로봇 차폭 | 영상에서 실제 폭을 계산하지 않는다. 너무 좁아 통과 불가가 명확한 경우에만 판단한다. 정확한 폭 판정은 센서/캘리브레이션 영역으로 미룬다. |
| Detection 거리 범위 | 몇 m 같은 거리 기준을 쓰지 않는다. 화면에서 식별 가능하고 주행에 영향이 있는 위험요소만 라벨링한다. |
| 여러 장애물 | 가능하면 객체별로 각각 Box를 그린다. 서로 붙어 분리가 불가능할 때만 예외로 하나의 Box를 검토한다. |
| 가려진 장애물 | 안 보이는 부분까지 상상해서 Box를 만들지 않는다. 보이는 부분만 기준으로 Box를 그린다. |
| 높이 모르는 턱 | `step` detection은 가능하다. 20cm 이하/초과를 임의로 추정해서 segmentation 값을 결정하지 않는다. |
| 차량 후드 | **저장값은 여전히 `non_drivable`(2)로 고정한다 — 4번째 클래스는 만들지 않는다.** `segmentation/*.png`는 모든 픽셀이 0/1/2 중 하나여야 하는 고정 포맷이므로(`§10`), "지면이 아니니 셋 중 아무것도 아니다"를 별도 값으로 표현하지 않는다. 개념적으로는 지면이 아니라는 게 맞지만, 실제 라벨은 기존과 동일하게 `non_drivable`로 강제한다. |
| 로봇 시작점 | **무조건 `drivable`로 만들면 안 된다.** 후드 바로 앞이라는 이유만으로 강제로 drivable을 채우지 않는다. 실제 눈앞 지면이 진짜 주행 가능할 때만 `drivable`이다(예: `010659`의 금속판 장애물처럼 후드 바로 앞이 non_drivable인 게 정답인 경우도 있다). |
| 주행영역 연결성 | segmentation은 "보이는 지면이 통과 가능한가"만 라벨링하고, 그 지면이 현재 위치에서 도달 가능한지(연결성/경로)는 원칙적으로 후속 Planner의 문제로 분리한다. **다만 파이프라인 후처리의 연결성 필터(`_keep_components_touching_hood`)는 계속 기본으로 켜둔다** — SAM2가 울타리·나무 등 무관한 대상에 오탐(leak)하는 걸 걸러내는 데 필요하기 때문이다. 그 필터가 SAM2 오탐이 아니라 진짜 지면(예: 도랑 건너편의 실제 도로)까지 지워버리는 게 확인되면, 그 프레임에 한해 사람이 예외로 override한다 — 필터 자체를 전역으로 끄지 않는다. |
| 풀 때문에 지면 안 보임 | 기존 문서의 "가려진 지면 임의 추정 금지" 원칙을 그대로 적용한다. 안 보이면 안 보이는 대로 두고 억지로 지면 상태를 추측하지 않는다. |
| 판단 불가능 영역 | 프레임 전체가 판단 불가면 골든셋에서 제외한다(위 "풀밭"/"큰 요철"과 동일 원칙). 프레임 일부만 판단 불가능한 경우, 새 클래스를 만들지 않고 `§3`의 기존 기본값("잘 안 보이는 영역의 기본값은 non_drivable")을 그대로 적용한다.
