"""Step 2: the AI assigns a class to every numbered region.

usage: python3 classify_regions.py <ids_file> <work_dir> <backend> [workers]

backend:
  terra-high     codex exec, gpt-5.6-terra, model_reasoning_effort=high
  agy-sonnet     agy, claude-sonnet-4-6 (separate quota group from agy's Gemini models)
  claude-sonnet  Claude Code CLI (claude -p), model sonnet

reads  <work_dir>/regions/<frame_id>.json and <work_dir>/marks/<frame_id>.jpg
writes <work_dir>/labels_<backend>/<frame_id>.json

The AI is shown the original image (to judge what things are) and the marked
image (to read region numbers). It never places points or draws boundaries.
The rules it applies are AGENTS.md §3-§7 and §18, taken verbatim from AGENTS.md
rather than restated here -- read ONCE when the batch starts, not per frame
(AGENTS.md §1.2). If AGENTS.md changes mid-run, no new frame is started and the
user is asked before the rules are read again.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import V0_IMAGES
from common import RulesSnapshot
from common import read_ids
from common import read_json
from common import write_json

BACKENDS = ("terra-high", "agy-sonnet", "claude-sonnet")
TIMEOUT_S = 600

TASK = """당신은 노지/농로 자율주행 로봇 학습 데이터의 라벨링 판단을 합니다. 파일을 수정하지 마세요.

이미지 두 장이 주어집니다.
- 원본 이미지: 무엇이 보이는지(흙길, 풀, 갈대, 관목, 논, 물, 장애물 등) 판단하는 데 쓰세요.
- 영역 이미지: 같은 사진을 SAM2가 영역으로 나눠 흰 경계선과 노란 번호를 붙인 것입니다.

영역의 경계는 이미 정해져 있습니다. 당신은 경계를 판단하지 않습니다.
당신이 할 일은 **번호가 붙은 영역 하나하나가 무엇인지 분류하는 것**뿐입니다.

## 분류 기준
아래는 AGENTS.md 원문입니다. 이 기준만 적용하세요.

{rules}

## 영역별로 정할 것
- seg: "drivable" | "caution" | "non_drivable" (위 기준 그대로)
- det: 그 영역 자체가 Detection 대상이면 "step" | "ditch_hole" | "puddle" | "obstacle", 아니면 null
  - 기준(AGENTS.md §5, §18)대로 주행 경로 안에 있거나 경로를 직접 막는 것만 det를 붙입니다.
  - 경로 옆에 나란한 배수로, 논, 밭, 뻘, 먼 배경의 물체에는 det를 붙이지 않습니다.
- 하늘, 먼 산, 건물, 전선, 촬영 차량 보닛/대시보드 반사는 seg="non_drivable", det=null 입니다.

## 한 영역 안에 서로 다른 것이 섞여 있을 때
예를 들어 한 영역의 절반은 흙길이고 절반은 관목이면, 더 넓게 차지하는 쪽으로 분류하고
그 영역 번호를 mixed_regions에 넣으세요. 억지로 쪼개려 하지 마세요.

## 영역 목록 (id, 화면 대비 면적 비율, 경계 상자 [x1,y1,x2,y2], 번호 위치 [x,y] — 모두 0~1 정규화)
{region_list}

## 출력
아래 JSON 하나만 출력하세요. 코드펜스나 다른 설명은 붙이지 마세요.
목록에 있는 모든 id가 regions에 빠짐없이 들어가야 합니다.
{{
  "regions": {{"1": {{"seg": "non_drivable", "det": null}}, "2": {{"seg": "drivable", "det": null}}}},
  "mixed_regions": [],
  "weather": "sunny|cloudy|after_rain",
  "surface_condition": "dry|wet",
  "reasoning": "주행 경로, 좌측, 우측을 각각 어떻게 판단했는지 두세 문장"
}}
"""

NO_SNOOP = "\n\n**중요**: 주어진 두 이미지 파일만 보고 판단하세요. 다른 디렉토리나 파일을 읽거나 탐색하지 마세요."


def build_prompt(regions_info: dict, rules: str, evidence: dict | None = None) -> str:
    region_list = "\n".join(
        f"- {r['id']}: area={r['area_frac']}, bbox={r['bbox_norm']}, label={r['label_xy_norm']}"
        for r in regions_info["regions"]
    )
    model_evidence = ""
    if evidence is not None:
        candidates = evidence.get("rtmdet_candidates", [])
        candidate_list = "\n".join(
            f"- D{v['index']}: class={v['class']}, score={v['score']}, box={v['box']}" for v in candidates
        ) or "- 후보 없음"
        model_evidence = f"""

## 보조 모델 증거 (정답이 아님)
세 번째 이미지는 PIDNet의 3클래스 색상 예측 위에 RTMDet 후보 상자를 D번호로 그린 것입니다.
두 모델 모두 틀릴 수 있으므로 원본 이미지를 우선하세요. PIDNet은 영역 경계를 바꾸는 근거가 아니며,
RTMDet 후보는 아래 목록에서만 수락할 수 있습니다. 상자를 새로 만들거나 위치를 바꾸지 마세요.
RTMDet 후보 목록:
{candidate_list}

출력 JSON에는 아래 accepted_detections도 넣으세요. 실제 위험요소이고 주행 경로에 영향을 주는
후보만 한 번씩 넣고, 나머지는 제외하세요. index와 class는 후보 목록의 값을 그대로 사용하세요.
"accepted_detections": [{{"index": 0, "class": "puddle"}}]
"""
    return TASK.format(rules=rules, region_list=region_list) + model_evidence


def extract_json(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON in output: {text[:300]}")
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    return obj


def run_backend(backend: str, prompt: str, original: Path, marked: Path, evidence: Path | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"rc_{backend}_") as tmp:
        orig = Path(tmp) / "original.jpg"
        mark = Path(tmp) / "regions.jpg"
        shutil.copy2(original, orig)
        shutil.copy2(marked, mark)
        paths = f"\n\n원본 이미지 경로: {orig}\n영역 이미지 경로: {mark}"
        inputs = ["-i", str(orig), "-i", str(mark)]
        if evidence is not None:
            hint = Path(tmp) / "model_hints.jpg"
            shutil.copy2(evidence, hint)
            paths += f"\n보조 모델 증거 이미지 경로: {hint}"
            inputs.extend(["-i", str(hint)])

        if backend == "terra-high":
            # runs from an isolated temp dir, which is not a git repo
            cmd = ["codex", "exec", "--skip-git-repo-check", "-m", "gpt-5.6-terra", "-c", "model_reasoning_effort=high",
                   prompt + paths, *inputs, "--output-last-message", "/dev/stdout"]
        elif backend == "agy-sonnet":
            cmd = ["agy", "-p", prompt + paths + NO_SNOOP, "--model", "claude-sonnet-4-6",
                   "--print-timeout", "9m", "--dangerously-skip-permissions"]
        elif backend == "claude-sonnet":
            cmd = ["claude", "-p", prompt + paths + NO_SNOOP + "\n두 이미지 파일을 Read 도구로 열어서 보세요.",
                   "--model", "sonnet", "--allowedTools", "Read"]
        else:
            raise ValueError(f"unknown backend {backend}")

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, cwd=tmp,
                              stdin=subprocess.DEVNULL)
    try:
        return extract_json(proc.stdout.strip())
    except ValueError:
        raise ValueError(f"{proc.stdout.strip()[-300:]} | stderr: {proc.stderr.strip()[-300:]}")


def validate(result: dict, regions_info: dict, evidence: dict | None = None) -> list[str]:
    """Return problems; an empty list means the answer is usable."""
    problems = []
    expected = {str(r["id"]) for r in regions_info["regions"]}
    got = result.get("regions", {})
    missing = expected - set(got)
    if missing:
        problems.append(f"missing regions {sorted(missing, key=int)}")
    for rid, v in got.items():
        if not isinstance(v, dict) or v.get("seg") not in ("drivable", "caution", "non_drivable"):
            problems.append(f"region {rid} bad seg {v}")
        elif v.get("det") not in (None, "step", "ditch_hole", "puddle", "obstacle"):
            problems.append(f"region {rid} bad det {v.get('det')}")
    if evidence is not None:
        candidates = {v["index"]: v for v in evidence.get("rtmdet_candidates", [])}
        seen = set()
        accepted = result.get("accepted_detections")
        if not isinstance(accepted, list):
            problems.append("accepted_detections missing or not a list")
        else:
            for item in accepted:
                if not isinstance(item, dict) or item.get("index") not in candidates:
                    problems.append(f"bad accepted detection {item}")
                    continue
                index = item["index"]
                if index in seen:
                    problems.append(f"duplicate accepted detection {index}")
                seen.add(index)
                if item.get("class") != candidates[index]["class"]:
                    problems.append(f"accepted detection {index} changed class")
    return problems


def main():
    ids = read_ids(Path(sys.argv[1]))
    work_dir = Path(sys.argv[2]).resolve()
    backend = sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    img_dir = Path(sys.argv[5]).resolve() if len(sys.argv) > 5 else V0_IMAGES
    evidence_dir = Path(sys.argv[6]).resolve() if len(sys.argv) > 6 else None
    if backend not in BACKENDS:
        sys.exit(f"backend must be one of {BACKENDS}")
    out_dir = work_dir / f"labels_{backend}"
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for f in ids:
        if (out_dir / f"{f}.json").exists():
            continue
        if not (work_dir / "regions" / f"{f}.json").exists():
            print(f"{f}: SKIP - not segmented yet", flush=True)
            continue
        todo.append(f)
    print(f"todo {len(todo)} backend={backend} workers={workers}", flush=True)

    # AGENTS.md §1.2: read the rules once for the whole batch.
    rules = RulesSnapshot()

    while todo:
        skipped = run_pass(todo, rules, backend, workers, work_dir, img_dir, evidence_dir, out_dir)
        if not skipped:
            break
        # AGENTS.md changed mid-run. Never reload silently: ask first.
        if not ask_to_reload_rules(len(skipped)):
            print(f"=== stopped: {len(skipped)} frame(s) not started, rules not reloaded ===", flush=True)
            print("not started: " + " ".join(skipped), flush=True)
            sys.exit(2)
        rules = RulesSnapshot()
        print("AGENTS.md re-read once with your permission; continuing.", flush=True)
        todo = skipped
    print("=== pass done ===", flush=True)


def ask_to_reload_rules(remaining: int) -> bool:
    """Ask the user whether to re-read AGENTS.md (AGENTS.md §1.2)."""
    question = (
        f"\nAGENTS.md가 실행 도중 수정되었습니다. 이미 시작된 프레임은 이전 규칙으로 끝났고, "
        f"아직 시작하지 않은 {remaining}장은 멈춰 두었습니다.\n"
        f"AGENTS.md를 한 번 다시 읽고 남은 {remaining}장을 새 규칙으로 이어서 진행할까요?"
    )
    if not sys.stdin.isatty():
        print(question + "\n대화형 터미널이 아니라 허락을 받을 수 없으므로 다시 읽지 않고 중단합니다.", flush=True)
        return False
    try:
        return input(question + " [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def run_pass(todo, rules, backend, workers, work_dir, img_dir, evidence_dir, out_dir) -> list[str]:
    """Label `todo` with one rules snapshot. Returns frames not started because AGENTS.md changed."""
    halt = threading.Event()
    skipped: list[str] = []
    skipped_lock = threading.Lock()

    def one(frame_id: str):
        # A cheap stat() check, not a re-read. Frames already running finish
        # on the snapshot they started with; nothing new starts on stale rules.
        if halt.is_set() or rules.changed():
            if not halt.is_set():
                halt.set()
                print("AGENTS.md changed during the run -- not starting new frames.", flush=True)
            with skipped_lock:
                skipped.append(frame_id)
            return
        regions_info = read_json(work_dir / "regions" / f"{frame_id}.json")
        evidence = read_json(evidence_dir / f"{frame_id}.json") if evidence_dir else None
        prompt = build_prompt(regions_info, rules.text, evidence)
        try:
            evidence_img = evidence_dir / f"{frame_id}.jpg" if evidence_dir else None
            result = run_backend(backend, prompt, img_dir / f"{frame_id}.jpg", work_dir / "marks" / f"{frame_id}.jpg", evidence_img)
        except Exception as e:
            print(f"{frame_id}: FAILED - {str(e)[:300]}", flush=True)
            return
        problems = validate(result, regions_info, evidence)
        if problems:
            print(f"{frame_id}: INVALID - {'; '.join(problems)[:300]}", flush=True)
            write_json(out_dir / f"{frame_id}.invalid.json", {"problems": problems, "result": result})
            return
        write_json(out_dir / f"{frame_id}.json", result)
        segs = [v["seg"] for v in result["regions"].values()]
        dets = [v["det"] for v in result["regions"].values() if v["det"]]
        print(f"{frame_id}: ok drivable={segs.count('drivable')} caution={segs.count('caution')} "
              f"det={dets} mixed={result.get('mixed_regions', [])}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    return sorted(skipped)


if __name__ == "__main__":
    main()
