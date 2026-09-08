#!/usr/bin/env python3
"""Gemma Vision + SAM2 second-pass review for exactly ten validation frames.

The source validation tree is read-only input.  This module contains no
image-content heuristic: Gemma decides semantics and SAM2 supplies geometry.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/hs/rang/robot_vision/dataset/validation")
OUTPUT = ROOT / "dataset" / "validation_gemma10"
MODEL = "@cf/google/gemma-4-26b-a4b-it"
FRAME_COUNT = 10
SEG_NAMES = ("drivable", "caution", "non_drivable")
DET_NAMES = ("step", "ditch_hole", "puddle", "obstacle")
META_KEYS = {"frame_id", "location", "weather", "surface_condition", "camera", "note"}
SEG_ACTION_TO_ID = {
    "add_drivable": 0,
    "remove_drivable": 2,
    "set_drivable": 0,
    "set_caution": 1,
    "set_non_drivable": 2,
}
# BGR visualization colors. The labeling specification defines color names,
# not numeric color values; these constants are visualization-only.
SEG_COLORS = {0: (0, 200, 0), 1: (0, 230, 230), 2: (0, 0, 230)}
DET_COLORS = {0: (211, 0, 148), 1: (255, 0, 0), 2: (255, 220, 0), 3: (0, 140, 255)}

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["frame_id", "summary", "surface_condition", "vegetation_reviews", "segmentation_changes", "detection_changes"],
    "properties": {
        "frame_id": {"type": "string"},
        "summary": {"type": "string"},
        "surface_condition": {"enum": ["dry", "wet"]},
        "vegetation_reviews": {
            "type": "array", "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["vegetation_type", "description", "prompt_box", "confidence", "review_needed"],
                "properties": {
                    "vegetation_type": {"enum": ["soft_vegetation", "rigid_vegetation", "uncertain_vegetation"]},
                    "description": {"type": "string"},
                    "prompt_box": {
                        "type": "array", "minItems": 4, "maxItems": 4,
                        "description": "Normalized [x1,y1,x2,y2], with x1<x2 and y1<y2",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "review_needed": {"type": "boolean",
                                      "description": "Must be true for uncertain_vegetation"},
                },
            },
        },
        "segmentation_changes": {
            "type": "array", "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "vegetation_type", "description", "prompt_box", "confidence"],
                "properties": {
                    "action": {"enum": list(SEG_ACTION_TO_ID)},
                    "vegetation_type": {"type": ["string", "null"],
                                        "enum": ["soft_vegetation", "rigid_vegetation", "uncertain_vegetation", None]},
                    "description": {"type": "string"},
                    "prompt_box": {
                        "type": "array", "minItems": 4, "maxItems": 4,
                        "description": "Normalized [x1,y1,x2,y2], with x1<x2 and y1<y2",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "detection_changes": {
            "type": "array", "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "class", "target_index", "box", "reason", "confidence"],
                "properties": {
                    "action": {"enum": ["keep", "delete", "adjust", "add"]},
                    "class": {"enum": list(DET_NAMES)},
                    "target_index": {"type": ["integer", "null"], "minimum": 0},
                    "box": {
                        "type": ["array", "null"], "minItems": 4, "maxItems": 4,
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "reason": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
}

SYSTEM_PROMPT = """You are reviewing existing robot-vision pre-labels, not relabeling from scratch.
Use the robot's actual traversability, not human walkability. Keep existing drivable by default.
Only return necessary corrections. Do not infer hidden ground, centimetre height, or depth from one RGB.
Never estimate or output vegetation height in cm from RGB. A future LiDAR/depth stage owns the 20 cm rule.
Do not create caution merely for dirt, gravel, roughness, tire tracks, shadow, or night.
Do not call shadow a puddle, simple slope a ditch_hole, or roadside vegetation an obstacle.
Classify relevant vegetation by RGB appearance: soft_vegetation means grass, reeds, thin herbaceous or
visibly flexible plants likely to bend; rigid_vegetation means tree trunks, thick branches, small trees, or
woody shrubs likely to physically block the route; uncertain_vegetation means RGB cannot reliably distinguish
soft from rigid, structure is occluded, or traversability is unclear. Do not remove vegetation as a whole.
Soft vegetation connected to visible traversable ground can remain or become drivable when no other obstacle
is visible. Rigid vegetation blocking the actual route can be remove_drivable and optionally an obstacle.
Uncertain vegetation must preserve the baseline and set review_needed=true in vegetation_reviews.
For every visually distinct vegetation region touching, overlapping, or immediately bordering the current
drivable route, vegetation_reviews is mandatory even when no label change is needed. Do not omit that review
merely because the segmentation stays unchanged.
Review existing drivable bidirectionally: add_drivable for clearly omitted connected traversable ground;
remove_drivable only for a clearly included blocking rigid plant, trunk, thick branch, rock pile, fence, or
other physical obstruction. Never remove merely because grass or reeds are visible.
Detection actions are keep/delete/adjust/add. Boxes are normalized [x1,y1,x2,y2] and tight.
Segmentation actions are add_drivable/remove_drivable/set_drivable/set_caution/set_non_drivable. Their normalized
prompt_box is only a coarse SAM2 prompt around the visible region needing change. Every prompt_box
must be [x1,y1,x2,y2] with x1<x2 and y1<y2; never output cx,cy,width,height there.
Do not inventory every visible object. If evidence is uncertain, keep the baseline and emit no change.
If no correction is needed, both change arrays must be empty. Never emit a placeholder or degenerate box.
If the summary calls an existing detection a misclassification or false positive, emit its delete action;
otherwise do not describe it as a misclassification.
Before adding a segmentation change, compare it to the RIGHT pre-label overlay: if the target area is already
that class, emit no change. A valid full-area example is [0.01,0.01,0.99,0.99], never [1,1,1,1].
For add_drivable/set_drivable, prompt_box must tightly cover only visible missing road pixels outside the current
green mask. It must not include roadside vegetation, vehicles, or a large area already green.
For remove_drivable, prompt_box must tightly cover only the clearly blocking region currently inside green.
Every vegetation-related segmentation change must set vegetation_type; non-vegetation changes use null.
SAM2 only traces boundaries. Do not assume SAM2 can repair an incorrect semantic target.
Inspect once and emit the JSON immediately without extended deliberation.
Return only JSON matching the supplied schema."""


def load_environment() -> None:
    load_dotenv(ROOT / ".env", override=False)


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths(frame_id: str) -> dict[str, Path]:
    return {
        "image": SOURCE / "images" / f"{frame_id}.png",
        "segmentation": SOURCE / "segmentation" / f"{frame_id}.png",
        "detection": SOURCE / "detection" / f"{frame_id}.txt",
        "metadata": SOURCE / "metadata" / f"{frame_id}.json",
    }


def source_inventory(frames: list[str]) -> dict[str, Any]:
    tree = hashlib.sha256()
    counts: dict[str, int] = {}
    for directory, suffix in (("images", ".png"), ("segmentation", ".png"),
                              ("detection", ".txt"), ("metadata", ".json")):
        paths = sorted((SOURCE / directory).glob(f"frame_*{suffix}"))
        counts[directory] = len(paths)
        for path in paths:
            stat = path.stat()
            tree.update(f"{path.relative_to(SOURCE)}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    selected = {
        str(path.relative_to(SOURCE)): sha256(path)
        for frame in frames for path in source_paths(frame).values()
    }
    return {"counts": counts, "tree_stat_sha256": tree.hexdigest(), "selected_sha256": selected}


def parse_yolo(path: Path) -> list[list[float]]:
    result: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise RuntimeError(f"invalid YOLO row: {path}:{line_number}")
        cls = int(fields[0])
        values = [float(value) for value in fields[1:]]
        if cls not in range(4) or not all(0 <= value <= 1 for value in values):
            raise RuntimeError(f"invalid YOLO value: {path}:{line_number}")
        cx, cy, width, height = values
        if cx - width / 2 < 0 or cx + width / 2 > 1 or cy - height / 2 < 0 or cy + height / 2 > 1:
            raise RuntimeError(f"YOLO box crosses image boundary: {path}:{line_number}")
        result.append([cls, *values])
    return result


def candidate_frames() -> list[str]:
    """Select ten frames using only baseline labels/metadata, never RGB semantics."""
    records: list[dict[str, Any]] = []
    image_frames = sorted(path.stem for path in (SOURCE / "images").glob("frame_*.png"))
    if len(image_frames) != 15000:
        raise RuntimeError(f"expected 15000 source images, found {len(image_frames)}")
    for frame_id in image_frames:
        paths = source_paths(frame_id)
        if not all(path.is_file() for path in paths.values()):
            raise RuntimeError(f"source 1:1 files missing for {frame_id}")
        metadata = read_json(paths["metadata"])
        classes = {int(row[0]) for row in parse_yolo(paths["detection"])}
        # The legacy baseline uses Korean weather labels. Only the two values
        # with exact documented equivalents are eligible; ambiguous 비/안개 is
        # not guessed into after_rain.
        if metadata.get("weather") in {"맑음", "흐림"}:
            records.append({"frame": frame_id, "classes": classes,
                            "stratum": (metadata.get("location"), metadata.get("weather"))})

    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    negatives: list[dict[str, Any]] = []
    for record in records:
        if record["classes"]:
            for cls in record["classes"]:
                buckets[cls].append(record)
        else:
            negatives.append(record)

    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()

    def take_spread(pool: list[dict[str, Any]], amount: int, minimum_gap: int = 300) -> None:
        available = [item for item in pool if item["frame"] not in seen]
        if not available:
            return
        targets = np.linspace(0, len(available) - 1, min(amount, len(available)), dtype=int)
        for target in targets:
            ranked = sorted(range(len(available)), key=lambda index: abs(index - int(target)))
            for index in ranked:
                item = available[index]
                number = int(item["frame"].split("_")[1])
                chosen_numbers = [int(entry["frame"].split("_")[1]) for entry in chosen]
                if item["frame"] not in seen and all(abs(number - old) >= minimum_gap for old in chosen_numbers):
                    chosen.append(item); seen.add(item["frame"])
                    break

    for class_id in range(4):
        take_spread(buckets[class_id], 2)
    take_spread(negatives, 3)
    if len(chosen) < FRAME_COUNT:
        take_spread(records, FRAME_COUNT - len(chosen), minimum_gap=100)
    frames = sorted(item["frame"] for item in chosen[:FRAME_COUNT])
    if len(frames) != FRAME_COUNT:
        raise RuntimeError("could not select exactly ten unique frames")
    return frames


def ensure_output_dirs() -> None:
    for directory in ("images", "segmentation", "detection", "metadata", "review_results",
                      "review_results_previous", "geometry_reviews",
                      "overlays/before", "overlays/after", "overlays/comparison"):
        (OUTPUT / directory).mkdir(parents=True, exist_ok=True)


def overlay(image: np.ndarray, mask: np.ndarray, detections: list[list[float]], frame_id: str) -> np.ndarray:
    color = np.zeros_like(image)
    for cls, bgr in SEG_COLORS.items():
        color[mask == cls] = bgr
    result = cv2.addWeighted(image, 0.62, color, 0.38, 0)
    height, width = image.shape[:2]
    for row in detections:
        cls, cx, cy, bw, bh = row
        x1, y1 = int((cx - bw / 2) * width), int((cy - bh / 2) * height)
        x2, y2 = int((cx + bw / 2) * width), int((cy + bh / 2) * height)
        bgr = DET_COLORS[int(cls)]
        cv2.rectangle(result, (x1, y1), (x2, y2), bgr, 3)
        cv2.putText(result, DET_NAMES[int(cls)], (x1, max(22, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, bgr, 2, cv2.LINE_AA)
    cv2.putText(result, frame_id, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                .9, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(result, frame_id, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                .9, (0, 0, 0), 1, cv2.LINE_AA)
    return result


def comparison_image(image: np.ndarray, current_overlay: np.ndarray, frame_id: str) -> np.ndarray:
    left, right = image.copy(), current_overlay.copy()
    cv2.putText(left, "ORIGINAL RGB", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 3)
    cv2.putText(right, "CURRENT PRE-LABEL", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 3)
    result = np.hstack((left, right))
    cv2.putText(result, frame_id, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 3)
    return result


def data_uri(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("comparison JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


class Cloudflare:
    def __init__(self) -> None:
        self.token = require_env("CLOUDFLARE_API_TOKEN")
        self.account = require_env("CLOUDFLARE_ACCOUNT_ID")
        self.base = "https://api.cloudflare.com/client/v4"
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        response = requests.request(method, url, headers=self.headers, timeout=240, **kwargs)
        try:
            body = response.json()
        except ValueError as error:
            raise RuntimeError(f"Cloudflare returned non-JSON HTTP {response.status_code}") from error
        if not response.ok or body.get("success") is False:
            errors = body.get("errors") or body.get("messages") or "unspecified API error"
            raise RuntimeError(f"Cloudflare HTTP {response.status_code}: {errors}")
        return body

    def auth(self) -> dict[str, Any]:
        result = self.request("GET", f"{self.base}/user/tokens/verify")
        return {"success": True, "status": result.get("result", {}).get("status")}

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"{self.base}/accounts/{self.account}/ai/run/{MODEL}", json=payload)

    def text_test(self) -> dict[str, Any]:
        response = self.run({"messages": [{"role": "user", "content": "Return exactly {\"ok\":true} as JSON."}],
                             "response_format": {"type": "json_object"}, "max_tokens": 32})
        return {"success": True, "usage": extract_usage(response)}

    def review(self, frame_id: str, comparison: np.ndarray, detections: list[list[float]]) -> tuple[dict[str, Any], dict[str, Any]]:
        existing = [{"target_index": i, "class": DET_NAMES[int(row[0])],
                     "yolo_cxcywh": row[1:]} for i, row in enumerate(detections)]
        prompt = (f"Frame: {frame_id}\nExisting detection pre-labels:\n{json.dumps(existing)}\n"
                  "Also classify only the visible surface_condition as dry or wet for the required metadata. "
                  "Review only label changes.")
        if frame_id == "frame_007085":
            prompt += ("\nRequired targeted audit: explicitly inspect vegetation touching or included in the "
                       "current green drivable boundary. Classify each relevant region as soft_vegetation, "
                       "rigid_vegetation, or uncertain_vegetation before deciding whether remove_drivable is needed.")
        prior_invalid = OUTPUT / "review_results" / f"{frame_id}.invalid_judgment.json"
        if prior_invalid.exists():
            prompt += ("\nYour prior JSON below failed validation. Correct it rather than repeating it. "
                       "A summary saying no corrections needed requires empty change arrays. Every "
                       "uncertain_vegetation review must set review_needed=true and uncertain vegetation "
                       "must never appear in segmentation_changes. Include all required fields.\n" +
                       prior_invalid.read_text(encoding="utf-8"))
        payload = {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": [
                             {"type": "text", "text": prompt},
                             {"type": "image_url", "image_url": {"url": data_uri(comparison)}},
                         ]}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "robot_vision_second_pass", "strict": True, "schema": REVIEW_SCHEMA}},
            "temperature": 0,
            "reasoning_effort": "low",
            "chat_template_kwargs": {"enable_thinking": False},
            "max_completion_tokens": 3000,
        }
        raw = self.run(payload)
        text = extract_text(raw)
        try:
            judgment = json.loads(text)
        except json.JSONDecodeError as error:
            failure_path = OUTPUT / "review_results" / f"{frame_id}.failure.txt"
            failure_path.write_text(text, encoding="utf-8")
            write_json(OUTPUT / "review_results" / f"{frame_id}.failure_response.json", raw)
            raise RuntimeError("Gemma response was not a direct JSON document") from error
        try:
            validate_judgment(judgment, frame_id, len(detections))
        except RuntimeError:
            write_json(OUTPUT / "review_results" / f"{frame_id}.invalid_judgment.json", judgment)
            raise
        return judgment, extract_usage(raw)

    def geometry_check(self, frame_id: str, original: np.ndarray, before: np.ndarray,
                       candidate: np.ndarray, changes: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        tiles = [cv2.resize(tile, (960, 540), interpolation=cv2.INTER_AREA)
                 for tile in (original, before, candidate)]
        labels = ("ORIGINAL RGB", "BEFORE", "SAM2 CANDIDATE AFTER")
        for tile, label in zip(tiles, labels):
            cv2.putText(tile, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 3)
        triptych = np.hstack(tiles)
        schema = {
            "type": "object", "additionalProperties": False,
            "required": ["frame_id", "accept", "reason"],
            "properties": {"frame_id": {"type": "string"}, "accept": {"type": "boolean"},
                           "reason": {"type": "string"}},
        }
        prompt = (f"Frame: {frame_id}\nProposed changes: {json.dumps(changes)}\n"
                  "Decide whether SAM2 CANDIDATE AFTER accurately changes only the intended visible region. "
                  "Reject if it spills into roadside vegetation, vehicles, hidden ground, or unrelated areas; "
                  "reject if a remove/add action has the wrong semantic direction. Do not estimate cm height. "
                  "SAM2 is geometry-only and cannot repair a wrong target. Return JSON immediately.")
        raw = self.run({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri(triptych)}},
            ]}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "sam2_geometry_qc", "strict": True, "schema": schema}},
            "temperature": 0, "reasoning_effort": "low",
            "chat_template_kwargs": {"enable_thinking": False}, "max_completion_tokens": 512,
        })
        try:
            result = json.loads(extract_text(raw))
        except json.JSONDecodeError as error:
            raise RuntimeError("Gemma geometry QC was not direct JSON") from error
        if (not isinstance(result, dict) or set(result) != {"frame_id", "accept", "reason"}
                or result["frame_id"] != frame_id or not isinstance(result["accept"], bool)
                or not isinstance(result["reason"], str)):
            raise RuntimeError("Gemma geometry QC JSON is invalid")
        return result, extract_usage(raw)


def extract_text(response: dict[str, Any]) -> str:
    result = response.get("result", response)
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        if isinstance(result.get("response"), str):
            return result["response"].strip()
        choices = result.get("choices")
        if choices and isinstance(choices[0].get("message", {}).get("content"), str):
            return choices[0]["message"]["content"].strip()
    raise RuntimeError("Cloudflare response did not contain model text")


def extract_usage(response: dict[str, Any]) -> dict[str, Any]:
    result = response.get("result", response)
    usage = result.get("usage", {}) if isinstance(result, dict) else {}
    return usage if isinstance(usage, dict) else {}


def validate_box(box: Any, label: str) -> None:
    if not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in box):
        raise RuntimeError(f"invalid normalized {label}")
    if not box[0] < box[2] or not box[1] < box[3]:
        raise RuntimeError(f"unordered normalized {label}")


def validate_judgment(value: Any, frame_id: str, detection_count: int) -> None:
    if not isinstance(value, dict) or set(value) != {"frame_id", "summary", "surface_condition", "vegetation_reviews", "segmentation_changes", "detection_changes"}:
        raise RuntimeError("Gemma JSON has unexpected top-level fields")
    if value["frame_id"] != frame_id or not isinstance(value["summary"], str):
        raise RuntimeError("Gemma JSON frame_id/summary is invalid")
    if value["surface_condition"] not in {"dry", "wet"}:
        raise RuntimeError("Gemma JSON surface_condition is invalid")
    if not isinstance(value["vegetation_reviews"], list):
        raise RuntimeError("Gemma JSON vegetation_reviews is invalid")
    if frame_id == "frame_007085" and not value["vegetation_reviews"]:
        raise RuntimeError("frame_007085 requires an explicit vegetation review")
    for review in value["vegetation_reviews"]:
        if set(review) != {"vegetation_type", "description", "prompt_box", "confidence", "review_needed"}:
            raise RuntimeError("invalid vegetation review fields")
        if review["vegetation_type"] not in {"soft_vegetation", "rigid_vegetation", "uncertain_vegetation"}:
            raise RuntimeError("invalid vegetation_type")
        validate_box(review["prompt_box"], "vegetation prompt_box")
        if review["vegetation_type"] == "uncertain_vegetation" and review["review_needed"] is not True:
            raise RuntimeError("uncertain vegetation must be review_needed")
    for change in value["segmentation_changes"]:
        if set(change) != {"action", "vegetation_type", "description", "prompt_box", "confidence"} or change["action"] not in SEG_ACTION_TO_ID:
            raise RuntimeError("invalid segmentation change")
        if change["vegetation_type"] not in {None, "soft_vegetation", "rigid_vegetation", "uncertain_vegetation"}:
            raise RuntimeError("invalid change vegetation_type")
        if change["vegetation_type"] == "uncertain_vegetation":
            raise RuntimeError("uncertain vegetation cannot change segmentation")
        validate_box(change["prompt_box"], "SAM2 prompt_box")
        if not isinstance(change["confidence"], (int, float)) or not 0 <= change["confidence"] <= 1:
            raise RuntimeError("invalid segmentation confidence")
    for change in value["detection_changes"]:
        if set(change) != {"action", "class", "target_index", "box", "reason", "confidence"}:
            raise RuntimeError("invalid detection change fields")
        action, index, box = change["action"], change["target_index"], change["box"]
        if action not in {"keep", "delete", "adjust", "add"} or change["class"] not in DET_NAMES:
            raise RuntimeError("invalid detection action/class")
        if action == "add":
            if index is not None: raise RuntimeError("add target_index must be null")
            validate_box(box, "detection box")
        else:
            if not isinstance(index, int) or not 0 <= index < detection_count:
                raise RuntimeError("detection target_index is out of range")
            if action == "adjust":
                validate_box(box, "detection box")
            elif box is not None:
                # Some schema-constrained model responses echo a reference box
                # for keep/delete. It is validated but never applied.
                validate_box(box, "unused detection reference box")


class SamGeometry:
    def __init__(self) -> None:
        import torch
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = os.environ.get("SAM2_HF_MODEL", "facebook/sam2.1-hiera-small")
        self.predictor = SAM2ImagePredictor(build_sam2_hf(model_id, device=self.device))

    def masks(self, rgb: np.ndarray, changes: list[dict[str, Any]]) -> list[np.ndarray]:
        self.predictor.set_image(rgb)
        height, width = rgb.shape[:2]
        results = []
        for change in changes:
            x1, y1, x2, y2 = change["prompt_box"]
            box = np.asarray([x1 * width, y1 * height, x2 * width, y2 * height], dtype=np.float32)
            masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
            result = masks[int(np.argmax(scores))].astype(bool)
            # SAM2 supplies boundary geometry, but only inside Gemma's semantic
            # coarse location. This prevents mask leakage beyond that prompt.
            within_prompt = np.zeros((height, width), dtype=bool)
            within_prompt[max(0, int(box[1])):min(height, int(np.ceil(box[3]))),
                          max(0, int(box[0])):min(width, int(np.ceil(box[2])))] = True
            results.append(result & within_prompt)
        return results


def apply_detections(baseline: list[list[float]], changes: list[dict[str, Any]]) -> list[list[float]]:
    rows: list[list[float] | None] = [list(row) for row in baseline]
    touched: set[int] = set()
    additions: list[list[float]] = []
    for change in changes:
        action = change["action"]
        if action == "keep": continue
        if action == "add":
            x1, y1, x2, y2 = change["box"]
            additions.append([DET_NAMES.index(change["class"]), (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
            continue
        index = change["target_index"]
        if index in touched: raise RuntimeError("multiple changes target one baseline detection")
        touched.add(index)
        if int(baseline[index][0]) != DET_NAMES.index(change["class"]):
            raise RuntimeError("detection class does not match target_index")
        if action == "delete": rows[index] = None
        elif action == "adjust":
            x1, y1, x2, y2 = change["box"]
            rows[index] = [DET_NAMES.index(change["class"]), (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
    return [row for row in rows if row is not None] + additions


def write_yolo(path: Path, rows: list[list[float]]) -> None:
    text = "".join(f"{int(row[0])} " + " ".join(f"{v:.8f}" for v in row[1:]) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def prepare() -> dict[str, Any]:
    if SOURCE.resolve() == OUTPUT.resolve() or SOURCE.resolve() in OUTPUT.resolve().parents:
        raise RuntimeError("output must not overlap read-only source")
    ensure_output_dirs()
    state_path = OUTPUT / "run_state.json"
    if state_path.exists(): return read_json(state_path)
    frames = candidate_frames()
    inventory = source_inventory(frames)
    state = {"version": 1, "model": MODEL, "frames": frames, "status": "prepared",
             "processed": {}, "failed": {}, "source_inventory_before": inventory,
             "auth": None, "text_test": None, "cvat": None}
    write_json(state_path, state)
    return state


def process_frame(frame_id: str, cloudflare: Cloudflare, sam: SamGeometry | None) -> tuple[dict[str, Any], SamGeometry | None]:
    paths = source_paths(frame_id)
    image = cv2.imread(str(paths["image"]), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(paths["segmentation"]), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None or mask.ndim != 2 or image.shape[:2] != mask.shape or set(np.unique(mask)) - {0, 1, 2}:
        raise RuntimeError("invalid baseline RGB/mask")
    detections = parse_yolo(paths["detection"])
    before = overlay(image, mask, detections, frame_id)
    comparison = comparison_image(image, before, frame_id)
    review_path = OUTPUT / "review_results" / f"{frame_id}.json"
    if review_path.exists():
        judgment = read_json(review_path)
        validate_judgment(judgment, frame_id, len(detections))
        usage: dict[str, Any] = {}
    else:
        judgment, usage = cloudflare.review(frame_id, comparison, detections)
        # Cache a validated original judgment before downstream SAM2 work so a
        # dependency/runtime retry never spends a second Vision call.
        write_json(review_path, judgment)
    for pattern in (f"{frame_id}.failure*", f"{frame_id}.invalid_judgment.json"):
        for failure in (OUTPUT / "review_results").glob(pattern):
            failure.unlink()

    updated_mask = mask.copy()
    sam_used = False
    changes = judgment["segmentation_changes"]
    if changes:
        sam = sam or SamGeometry()
        for change, region in zip(changes, sam.masks(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), changes)):
            updated_mask[region] = SEG_ACTION_TO_ID[change["action"]]
        sam_used = True
    updated_detections = apply_detections(detections, judgment["detection_changes"])
    geometry_qc: dict[str, Any] | None = None
    geometry_usage: dict[str, Any] = {}
    if changes:
        candidate_after = overlay(image, updated_mask, updated_detections, frame_id)
        geometry_qc, geometry_usage = cloudflare.geometry_check(
            frame_id, image, before, candidate_after, changes)
        write_json(OUTPUT / "geometry_reviews" / f"{frame_id}.json",
                   {**geometry_qc, "usage": geometry_usage})
        if not geometry_qc["accept"]:
            updated_mask = mask.copy()

    baseline_metadata = read_json(paths["metadata"])
    weather_map = {"맑음": "sunny", "흐림": "cloudy"}
    if baseline_metadata.get("frame_id") != frame_id or baseline_metadata.get("weather") not in weather_map:
        raise RuntimeError("baseline metadata cannot be converted without guessing")
    metadata = {
        "frame_id": frame_id,
        "location": baseline_metadata.get("location"),
        "weather": weather_map[baseline_metadata["weather"]],
        "surface_condition": judgment["surface_condition"],
        "camera": baseline_metadata.get("camera"),
        "note": baseline_metadata.get("note", []),
    }

    if not cv2.imwrite(str(OUTPUT / "images" / f"{frame_id}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 100]):
        raise RuntimeError("JPG export failed")
    if not cv2.imwrite(str(OUTPUT / "segmentation" / f"{frame_id}.png"), updated_mask):
        raise RuntimeError("mask export failed")
    write_yolo(OUTPUT / "detection" / f"{frame_id}.txt", updated_detections)
    write_json(OUTPUT / "metadata" / f"{frame_id}.json", metadata)
    after = overlay(image, updated_mask, updated_detections, frame_id)
    cv2.imwrite(str(OUTPUT / "overlays" / "before" / f"{frame_id}.jpg"), before)
    cv2.imwrite(str(OUTPUT / "overlays" / "after" / f"{frame_id}.jpg"), after)
    cv2.imwrite(str(OUTPUT / "overlays" / "comparison" / f"{frame_id}.jpg"), comparison)
    final_segmentation_changed = not np.array_equal(updated_mask, mask)
    return {"summary": judgment["summary"], "sam2_used": sam_used,
            "segmentation_changed": final_segmentation_changed,
            "detection_changed": any(c["action"] != "keep" for c in judgment["detection_changes"]),
            "segmentation_actions": [change["action"] for change in changes] if final_segmentation_changed else [],
            "sam2_candidate_rejected": bool(geometry_qc and not geometry_qc["accept"]),
            "geometry_qc": geometry_qc,
            "geometry_usage": geometry_usage,
            "vegetation_types": sorted({review["vegetation_type"] for review in judgment["vegetation_reviews"]}),
            "review_needed": any(review["review_needed"] for review in judgment["vegetation_reviews"]),
            "usage": usage}, sam


def run(pilot_only: bool) -> dict[str, Any]:
    state = prepare(); state_path = OUTPUT / "run_state.json"
    cloudflare = Cloudflare()
    if not state["auth"]:
        state["auth"] = cloudflare.auth(); write_json(state_path, state)
    if not state["text_test"]:
        state["text_test"] = cloudflare.text_test(); write_json(state_path, state)
    limit = 1 if pilot_only else FRAME_COUNT
    if not pilot_only and state.get("status") not in {"pilot_complete", "complete"}:
        raise RuntimeError("pilot must complete before batch")
    sam: SamGeometry | None = None
    for frame_id in state["frames"][:limit]:
        if frame_id in state["processed"]: continue
        try:
            result, sam = process_frame(frame_id, cloudflare, sam)
            state["processed"][frame_id] = result
            state["failed"].pop(frame_id, None)
        except Exception as error:
            state["failed"][frame_id] = f"{type(error).__name__}: {error}"
            write_json(state_path, state)
            raise
        write_json(state_path, state)
    state["status"] = "pilot_complete" if pilot_only else "complete"
    write_json(state_path, state)
    return state


def recheck_segmentation_changes() -> dict[str, Any]:
    """Re-review every changed segmentation after human overlay QC rejection."""
    state_path = OUTPUT / "run_state.json"
    state = read_json(state_path)
    targets = [frame for frame, result in state["processed"].items()
               if result.get("segmentation_changed")]
    cloudflare = Cloudflare()
    sam: SamGeometry | None = None
    for frame_id in targets:
        review_path = OUTPUT / "review_results" / f"{frame_id}.json"
        prior = read_json(review_path)
        write_json(OUTPUT / "review_results" / f"{frame_id}.invalid_judgment.json", prior)
        review_path.unlink()
        for directory, suffix in (("images", ".jpg"), ("segmentation", ".png"),
                                  ("detection", ".txt"), ("metadata", ".json"),
                                  ("overlays/before", ".jpg"), ("overlays/after", ".jpg"),
                                  ("overlays/comparison", ".jpg")):
            path = OUTPUT / directory / f"{frame_id}{suffix}"
            if path.exists(): path.unlink()
        state["processed"].pop(frame_id, None)
        write_json(state_path, state)
        result, sam = process_frame(frame_id, cloudflare, sam)
        state["processed"][frame_id] = result
        state["failed"].pop(frame_id, None)
        write_json(state_path, state)
    return state


def recheck_all_vegetation_policy() -> dict[str, Any]:
    """Re-run only the persisted ten frames under vegetation policy v2."""
    ensure_output_dirs()
    state_path = OUTPUT / "run_state.json"
    state = read_json(state_path)
    if len(state.get("frames", [])) != FRAME_COUNT:
        raise RuntimeError("vegetation recheck requires the existing ten-frame pilot")
    if state.get("policy_version") != 3:
        write_json(OUTPUT / "previous_run_state.json", state)
        for frame_id in state["frames"]:
            review_path = OUTPUT / "review_results" / f"{frame_id}.json"
            if review_path.exists():
                shutil.copy2(review_path, OUTPUT / "review_results_previous" / review_path.name)
                review_path.unlink()
            for directory, suffix in (("images", ".jpg"), ("segmentation", ".png"),
                                      ("detection", ".txt"), ("metadata", ".json"),
                                      ("overlays/before", ".jpg"), ("overlays/after", ".jpg"),
                                      ("overlays/comparison", ".jpg")):
                path = OUTPUT / directory / f"{frame_id}{suffix}"
                if path.exists(): path.unlink()
        state.update(policy_version=3, status="vegetation_recheck", processed={}, failed={}, cvat=None)
        write_json(state_path, state)
    cloudflare = Cloudflare()
    sam: SamGeometry | None = None
    for frame_id in state["frames"]:
        if frame_id in state["processed"]: continue
        try:
            result, sam = process_frame(frame_id, cloudflare, sam)
            state["processed"][frame_id] = result
            state["failed"].pop(frame_id, None)
        except Exception as error:
            state["failed"][frame_id] = f"{type(error).__name__}: {error}"
            write_json(state_path, state)
            raise
        write_json(state_path, state)
    state["status"] = "complete"
    write_json(state_path, state)
    return state


def geometry_qc_existing_changes() -> dict[str, Any]:
    """Run RGB/before/candidate semantic QC for persisted SAM2 changes."""
    state_path = OUTPUT / "run_state.json"
    state = read_json(state_path)
    targets = [frame for frame, result in state["processed"].items()
               if result.get("segmentation_changed")]
    cloudflare = Cloudflare()
    sam: SamGeometry | None = None
    for frame_id in targets:
        result, sam = process_frame(frame_id, cloudflare, sam)
        state["processed"][frame_id] = result
        write_json(state_path, state)
    return state


def qc() -> dict[str, Any]:
    state = read_json(OUTPUT / "run_state.json")
    frames = state["frames"]
    errors: list[str] = []
    for frame_id in state["processed"]:
        for pattern in (f"{frame_id}.failure*", f"{frame_id}.invalid_judgment.json"):
            for diagnostic in (OUTPUT / "review_results").glob(pattern):
                diagnostic.unlink()
    expected = set(frames)
    specs = (("images", ".jpg"), ("segmentation", ".png"), ("detection", ".txt"),
             ("metadata", ".json"), ("review_results", ".json"))
    counts = {}
    for directory, suffix in specs:
        found = {p.stem for p in (OUTPUT / directory).glob(f"frame_*{suffix}")}
        counts[directory] = len(found)
        if found != expected: errors.append(f"{directory} frame stems differ")
    for frame_id in frames:
        image = cv2.imread(str(OUTPUT / "images" / f"{frame_id}.jpg"))
        mask = cv2.imread(str(OUTPUT / "segmentation" / f"{frame_id}.png"), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None: continue
        if mask.ndim != 2 or image.shape[:2] != mask.shape: errors.append(f"{frame_id} mask shape")
        if set(np.unique(mask)) - {0, 1, 2}: errors.append(f"{frame_id} mask classes")
        try: parse_yolo(OUTPUT / "detection" / f"{frame_id}.txt")
        except Exception as error: errors.append(str(error))
        metadata = read_json(OUTPUT / "metadata" / f"{frame_id}.json")
        if set(metadata) != META_KEYS: errors.append(f"{frame_id} metadata keys")
        if metadata.get("frame_id") != frame_id: errors.append(f"{frame_id} metadata frame_id")
        if metadata.get("weather") not in {"sunny", "cloudy", "after_rain"}:
            errors.append(f"{frame_id} metadata weather")
        if metadata.get("surface_condition") not in {"dry", "wet"}:
            errors.append(f"{frame_id} metadata surface_condition")
    after_inventory = source_inventory(frames)
    source_unchanged = after_inventory == state["source_inventory_before"]
    if not source_unchanged: errors.append("source validation inventory changed")
    result = {"passed": not errors, "errors": errors, "counts": counts,
              "source_validation_unchanged": source_unchanged,
              "processed": len(state["processed"]), "failed": state["failed"]}
    write_json(OUTPUT / "qc_report.json", result)
    return result


def cvat_upload() -> dict[str, Any]:
    from cvat_sdk.core.client import AccessTokenCredentials, Client
    from cvat_sdk.core.uploading import DataUploader
    state = read_json(OUTPUT / "run_state.json")
    report = qc()
    if not report["passed"]: raise RuntimeError("QC must pass before CVAT upload")
    token = require_env("CVAT_ACCESS_TOKEN")
    url = os.environ.get("CVAT_URL", "http://localhost:8080")
    project_id = int(os.environ.get("CVAT_PROJECT_ID", "1"))
    client = Client(url, check_server_version=False)
    client.login(AccessTokenCredentials(token))
    try:
        project = client.projects.retrieve(project_id)
        labels = {label.name: label.id for label in project.get_labels()}
        required = {*SEG_NAMES, *DET_NAMES}
        if not required <= set(labels): raise RuntimeError("CVAT project lacks required labels")
        task = client.tasks.create({"name": "validation_gemma10_review", "project_id": project_id,
                                    "segment_size": FRAME_COUNT})
        images = [OUTPUT / "images" / f"{frame}.jpg" for frame in state["frames"]]
        response = DataUploader(client).upload_files(
            f"{url}/api/tasks/{task.id}/data/", images, image_quality=100, sorting_method="lexicographical")
        upload = json.loads(response.data.decode("utf-8"))
        request_id = upload.get("rq_id")
        if not request_id: raise RuntimeError("CVAT upload did not return a request ID")
        client.wait_for_completion(str(request_id), status_check_period=1)

        shapes = []
        for frame_index, frame_id in enumerate(state["frames"]):
            image = cv2.imread(str(OUTPUT / "images" / f"{frame_id}.jpg"))
            mask = cv2.imread(str(OUTPUT / "segmentation" / f"{frame_id}.png"), cv2.IMREAD_UNCHANGED)
            for class_id, name in enumerate(SEG_NAMES):
                contours, _ = cv2.findContours((mask == class_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) < 3: continue
                    perimeter = cv2.arcLength(contour, True)
                    polygon = cv2.approxPolyDP(contour, max(.5, perimeter * .0005), True)
                    if len(polygon) < 3: continue
                    shapes.append({"type": "polygon", "frame": frame_index, "label_id": labels[name],
                                   "points": polygon.reshape(-1, 2).astype(float).reshape(-1).tolist(),
                                   "occluded": False, "outside": False, "z_order": 2-class_id,
                                   "rotation": 0, "source": "auto", "attributes": []})
            height, width = image.shape[:2]
            for row in parse_yolo(OUTPUT / "detection" / f"{frame_id}.txt"):
                cls, cx, cy, bw, bh = row
                shapes.append({"type": "rectangle", "frame": frame_index, "label_id": labels[DET_NAMES[int(cls)]],
                               "points": [(cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height],
                               "occluded": False, "outside": False, "z_order": 10,
                               "rotation": 0, "source": "auto", "attributes": []})
        payload = {"version": 0, "tags": [], "tracks": [], "shapes": shapes}
        response = requests.put(f"{url}/api/tasks/{task.id}/annotations/",
                                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                                json=payload, timeout=240)
        response.raise_for_status()
        result = {"uploaded": True, "task_id": task.id, "task_name": "validation_gemma10_review",
                  "frames": FRAME_COUNT, "shapes": len(shapes), "url": f"{url}/tasks/{task.id}"}
    finally:
        client.close()
    state["cvat"] = result; write_json(OUTPUT / "run_state.json", state)
    return result


def main() -> int:
    load_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "auth", "text-test", "pilot", "batch", "recheck", "recheck-all", "geometry-qc", "qc", "cvat", "all"])
    args = parser.parse_args()
    if args.command == "prepare": result = prepare()
    elif args.command == "auth": result = Cloudflare().auth()
    elif args.command == "text-test": result = Cloudflare().text_test()
    elif args.command == "pilot": result = run(True)
    elif args.command == "batch": result = run(False)
    elif args.command == "recheck": result = recheck_segmentation_changes()
    elif args.command == "recheck-all": result = recheck_all_vegetation_policy()
    elif args.command == "geometry-qc": result = geometry_qc_existing_changes()
    elif args.command == "qc": result = qc()
    elif args.command == "cvat": result = cvat_upload()
    else:
        run(True); run(False); result = qc()
        if result["passed"] and os.environ.get("CVAT_ACCESS_TOKEN"): result["cvat"] = cvat_upload()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
