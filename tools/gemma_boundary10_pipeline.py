#!/usr/bin/env python3
"""Boundary-error focused ten-frame Gemma + SAM2 validation pilot."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

import gemma10_pipeline as common

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/hs/rang/robot_vision/dataset/validation")
OUTPUT = ROOT / os.environ.get("GEMMA_OUTPUT_DIR", "dataset/validation_gemma_sam_boundary10")
FRAME_COUNT = int(os.environ.get("GEMMA_FRAME_COUNT", "10"))
COMBINED_REVIEW_OVERLAY = os.environ.get("GEMMA_COMBINED_REVIEW_OVERLAY", "0") == "1"
BOX_SCHEMA = {"type": "array", "minItems": 4, "maxItems": 4,
              "items": {"type": "number", "minimum": 0, "maximum": 1}}
POINTS_SCHEMA = {"type": "array", "maxItems": 8,
                 "items": {"type": "array", "minItems": 2, "maxItems": 2,
                           "items": {"type": "number", "minimum": 0, "maximum": 1}}}
VEGETATION_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["vegetation_type", "description", "prompt_box", "confidence", "review_needed"],
    "properties": {"vegetation_type": {"enum": ["soft_vegetation", "rigid_vegetation", "uncertain_vegetation"]},
                   "description": {"type": "string"}, "prompt_box": BOX_SCHEMA,
                   "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "review_needed": {"type": "boolean"}}}
DECISION_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["action", "vegetation_type", "description", "coarse_box", "positive_points", "negative_points", "confidence"],
    "properties": {"action": {"enum": ["keep", "add_drivable", "remove_drivable", "review_needed"]},
                   "vegetation_type": {"type": ["string", "null"], "enum": ["soft_vegetation", "rigid_vegetation", "uncertain_vegetation", None]},
                   "description": {"type": "string"}, "coarse_box": {"anyOf": [BOX_SCHEMA, {"type": "null"}]},
                   "positive_points": POINTS_SCHEMA, "negative_points": POINTS_SCHEMA,
                   "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}
DETECTION_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["action", "class", "target_index", "box", "reason", "confidence"],
    "properties": {"action": {"enum": ["keep", "delete", "adjust", "add"]},
                   "class": {"enum": list(common.DET_NAMES)}, "target_index": {"type": ["integer", "null"], "minimum": 0},
                   "box": {"anyOf": [BOX_SCHEMA, {"type": "null"}]}, "reason": {"type": "string"},
                   "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}
REVIEW_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["frame_id", "summary", "surface_condition", "vegetation_reviews", "segmentation_decision", "detection_changes"],
    "properties": {"frame_id": {"type": "string"}, "summary": {"type": "string"}, "surface_condition": {"enum": ["dry", "wet"]},
                   "vegetation_reviews": {"type": "array", "maxItems": 8, "items": VEGETATION_SCHEMA},
                   "segmentation_decision": DECISION_SCHEMA,
                   "detection_changes": {"type": "array", "maxItems": 8, "items": DETECTION_SCHEMA}}}

SYSTEM_PROMPT = """Review an existing robot-vision pre-label, not a label from scratch. The image has ORIGINAL RGB and CURRENT BASELINE OVERLAY. Choose exactly one segmentation_decision action: keep, add_drivable, remove_drivable, or review_needed. Use robot traversability, not human walkability. Do not infer centimetre height/depth, output cm, infer hidden ground, or use a colour as a semantic decision.
Inspect whether the green drivable boundary contains visible road only, omits connected visible road, or includes clearly non-drivable vegetation, rock, fence, trunk, structure, or obstacle. Classify route-adjacent vegetation: soft_vegetation is grass/reeds/thin flexible herbaceous plants; rigid_vegetation is trunk/thick branch/small tree/woody shrub; uncertain_vegetation is ambiguous RGB evidence. Soft is not automatically drivable, and vegetation is not automatically non-drivable. Uncertain evidence must use review_needed, not a forced change. For each vegetation region touching/bordering the route include vegetation_reviews.
For add_drivable/remove_drivable provide a tight normalized coarse_box [x1,y1,x2,y2], at least one positive point in actual road ground to include/preserve, and a negative point in excluded vegetation/rock/fence/non-drivable material. Points are [x,y] normalized. For keep/review_needed set coarse_box null and point arrays empty. SAM2 traces only geometry; it cannot change semantics. Review detections: shadow is not puddle, simple slope not ditch, roadside vegetation alone not obstacle. Return only the strict JSON schema."""


def write_json(path: Path, value: Any) -> None:
    common.write_json(path, value)


def paths(frame: str) -> dict[str, Path]:
    return {"image": SOURCE / "images" / f"{frame}.png", "segmentation": SOURCE / "segmentation" / f"{frame}.png",
            "detection": SOURCE / "detection" / f"{frame}.txt", "metadata": SOURCE / "metadata" / f"{frame}.json"}


def ensure_dirs() -> None:
    names = ["images", "segmentation", "detection", "metadata", "review_results", "overlays"]
    if not COMBINED_REVIEW_OVERLAY:
        names += ["sam_candidates", "geometry_reviews", "candidate_previews", "diagnostics",
                  "overlays/baseline_before", "overlays/gemma_decision", "overlays/sam_candidate", "overlays/final_after"]
    for name in names:
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)


def geometry_score(mask: np.ndarray) -> float:
    """Mask-only candidate ranking, never RGB semantic inference."""
    h, w = mask.shape
    lower = (mask[int(h * .48):] == 0)
    side = np.concatenate((lower[:, :max(1, w // 4)].ravel(), lower[:, -max(1, w // 4):].ravel()))
    return float(side.mean()) + .25 * float((mask == 0).mean())


def select_candidates() -> tuple[list[str], list[dict[str, Any]]]:
    records = []
    for image in sorted((SOURCE / "images").glob("frame_*.png")):
        frame = image.stem
        number = int(frame.split("_")[1])
        # A deterministic 1/30 geometry-only screen avoids an image-model pass over
        # the full 15,000-frame corpus; the mandatory representative frame is added.
        if number % 30 and frame != "frame_007085":
            continue
        source = paths(frame)
        if not all(path.is_file() for path in source.values()):
            raise RuntimeError(f"source 1:1 files missing: {frame}")
        metadata = common.read_json(source["metadata"])
        if metadata.get("weather") not in {"맑음", "흐림"}:
            continue
        mask = cv2.imread(str(source["segmentation"]), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2 or set(np.unique(mask)) - {0, 1, 2}:
            raise RuntimeError(f"invalid source mask: {frame}")
        try:
            detection_rows = common.parse_yolo(source["detection"])
        except RuntimeError:
            # The immutable baseline occasionally has invalid YOLO geometry;
            # it cannot be exported as a valid final frame, so it is excluded.
            continue
        records.append({"frame": frame, "geometry_score": round(geometry_score(mask), 6),
                        "detection_classes": sorted({int(row[0]) for row in detection_rows})})
    required = "frame_007085"
    chosen = [next(item for item in records if item["frame"] == required)]
    # Keep existing detection-bearing scenes in the lightweight pool so the
    # 50-frame review is not only a road-boundary sample. No RGB meaning is inferred here.
    if FRAME_COUNT > 10:
        for class_id in range(4):
            for item in sorted((x for x in records if class_id in x["detection_classes"]), key=lambda x: (-x["geometry_score"], x["frame"]))[:4]:
                if item["frame"] not in {x["frame"] for x in chosen}:
                    chosen.append(item)
    for item in sorted(records, key=lambda x: (-x["geometry_score"], x["frame"])):
        number = int(item["frame"].split("_")[1])
        if item["frame"] not in {x["frame"] for x in chosen} and all(abs(number - int(old["frame"].split("_")[1])) >= 120 for old in chosen):
            chosen.append(item)
        if len(chosen) == FRAME_COUNT:
            break
    if len(chosen) != FRAME_COUNT:
        raise RuntimeError("could not select ten spread mask-geometry candidates")
    return sorted(x["frame"] for x in chosen), sorted(chosen, key=lambda x: x["frame"])


def validate_points(points: Any, label: str) -> None:
    valid = isinstance(points, list) and all(isinstance(p, list) and len(p) == 2 and
                                              all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in p) for p in points)
    if not valid:
        raise RuntimeError(f"invalid normalized {label}")


def validate_judgment(value: Any, frame: str, detection_count: int) -> None:
    required = {"frame_id", "summary", "surface_condition", "vegetation_reviews", "segmentation_decision", "detection_changes"}
    if not isinstance(value, dict) or set(value) != required or value["frame_id"] != frame or value["surface_condition"] not in {"dry", "wet"}:
        raise RuntimeError("invalid Gemma judgment fields")
    decision = value["segmentation_decision"]
    if not isinstance(decision, dict) or set(decision) != set(DECISION_SCHEMA["required"]):
        raise RuntimeError("invalid segmentation decision")
    action = decision["action"]
    if action not in {"keep", "add_drivable", "remove_drivable", "review_needed"}:
        raise RuntimeError("invalid decision action")
    if decision["vegetation_type"] not in {None, "soft_vegetation", "rigid_vegetation", "uncertain_vegetation"}:
        raise RuntimeError("invalid decision vegetation type")
    validate_points(decision["positive_points"], "positive points")
    validate_points(decision["negative_points"], "negative points")
    if not isinstance(decision["confidence"], (int, float)) or not 0 <= decision["confidence"] <= 1:
        raise RuntimeError("invalid decision confidence")
    if action in {"add_drivable", "remove_drivable"}:
        common.validate_box(decision["coarse_box"], "coarse box")
        if not decision["positive_points"] or not decision["negative_points"]:
            raise RuntimeError("a segmentation correction requires positive/negative points")
        if decision["vegetation_type"] == "uncertain_vegetation":
            raise RuntimeError("uncertain vegetation cannot force a segmentation change")
    elif decision["coarse_box"] is not None or decision["positive_points"] or decision["negative_points"]:
        raise RuntimeError("keep/review_needed may not send SAM2 prompts")
    for review in value["vegetation_reviews"]:
        if not isinstance(review, dict) or set(review) != set(VEGETATION_SCHEMA["required"]):
            raise RuntimeError("invalid vegetation review fields")
        if review["vegetation_type"] not in {"soft_vegetation", "rigid_vegetation", "uncertain_vegetation"}:
            raise RuntimeError("invalid vegetation type")
        common.validate_box(review["prompt_box"], "vegetation prompt box")
        if review["vegetation_type"] == "uncertain_vegetation" and review["review_needed"] is not True:
            raise RuntimeError("uncertain vegetation requires review_needed")
    if frame == "frame_007085" and not value["vegetation_reviews"]:
        raise RuntimeError("frame_007085 requires explicit vegetation review")
    # Share the existing strict YOLO-detection validation without accepting its old segmentation schema.
    common.validate_judgment({"frame_id": frame, "summary": value["summary"], "surface_condition": value["surface_condition"],
                             "vegetation_reviews": value["vegetation_reviews"], "segmentation_changes": [],
                             "detection_changes": value["detection_changes"]}, frame, detection_count)


class Cloudflare(common.Cloudflare):
    def review(self, frame: str, image_pair: np.ndarray, detections: list[list[float]]) -> tuple[dict[str, Any], dict[str, Any]]:
        existing = [{"target_index": i, "class": common.DET_NAMES[int(row[0])], "yolo_cxcywh": row[1:]} for i, row in enumerate(detections)]
        prompt = f"Frame: {frame}. Existing detection pre-labels: {json.dumps(existing)}. Also classify visible surface_condition as dry/wet."
        if not existing:
            prompt += " The existing detection list is empty, so detection_changes MUST be an empty array; do not use a target_index."
        else:
            prompt += f" Every delete/adjust target_index must be one of exactly {list(range(len(existing)))}; add uses target_index null."
        if frame == "frame_007085":
            prompt += " Required representative-failure audit: do not keep merely because vegetation is soft; distinguish visible driving surface, soft/rigid/uncertain vegetation, rocks, fence, and other non-drivable material."
        malformed_path = OUTPUT / "review_results" / f"{frame}.malformed_response.txt"
        if malformed_path.exists():
            text, usage = malformed_path.read_text(encoding="utf-8"), {}
        else:
            raw = self.run({"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": [
                                {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": common.data_uri(image_pair)}}]}],
                            "response_format": {"type": "json_schema", "json_schema": {"name": "boundary_review", "strict": True, "schema": REVIEW_SCHEMA}},
                            "temperature": 0, "reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False}, "max_completion_tokens": 2500})
            text, usage = common.extract_text(raw), common.extract_usage(raw)
            malformed_path.write_text(text, encoding="utf-8")
        for attempt in range(3):
            try:
                judgment = json.loads(text)
                validate_judgment(judgment, frame, len(detections))
                malformed_path.unlink(missing_ok=True)
                return judgment, usage
            except (json.JSONDecodeError, RuntimeError) as error:
                if attempt == 2:
                    raise RuntimeError(f"Gemma JSON could not be format-repaired: {error}") from error
                # Formatting/required-field repair only: it receives no image and must preserve semantics.
                repair_prompt = ("Return ONLY one valid JSON object. Preserve the supplied semantic decision; do not newly inspect an image. "
                                 "It must contain exactly frame_id, summary, surface_condition, vegetation_reviews, segmentation_decision, detection_changes. "
                                 "segmentation_decision must contain action, vegetation_type, description, coarse_box, positive_points, negative_points, confidence. "
                                 "If no vegetation review exists use []; if action is keep/review_needed use null and empty point arrays.\nINPUT:\n" + text)
                repair = self.run({"messages": [{"role": "user", "content": repair_prompt}],
                                   "response_format": {"type": "json_schema", "json_schema": {"name": "boundary_review_repair", "strict": True, "schema": REVIEW_SCHEMA}},
                                   "temperature": 0, "reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False}, "max_completion_tokens": 2500})
                text = common.extract_text(repair)
                malformed_path.write_text(text, encoding="utf-8")

    def geometry_qc(self, frame: str, panels: np.ndarray, decision: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = {"type": "object", "additionalProperties": False, "required": ["frame_id", "verdict", "reason"],
                  "properties": {"frame_id": {"type": "string"}, "verdict": {"enum": ["accept", "reject", "review_needed"]}, "reason": {"type": "string"}}}
        prompt = (f"Frame {frame}; Gemma semantic decision: {json.dumps(decision)}. Compare ORIGINAL RGB, BASELINE BEFORE, and SAM2 CANDIDATE AFTER. "
                  "Does the candidate improve only the real road boundary, without vegetation/rock/fence leakage or excessive road removal? Return accept/reject/review_needed. Do not infer cm; SAM2 is geometry only.")
        raw = self.run({"messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": common.data_uri(panels)}}]}],
                        "response_format": {"type": "json_schema", "json_schema": {"name": "boundary_geometry_qc", "strict": True, "schema": schema}},
                        "temperature": 0, "reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False}, "max_completion_tokens": 512})
        result = json.loads(common.extract_text(raw))
        if not isinstance(result, dict) or set(result) != {"frame_id", "verdict", "reason"} or result["frame_id"] != frame or result["verdict"] not in {"accept", "reject", "review_needed"}:
            raise RuntimeError("invalid geometry QC response")
        return result, common.extract_usage(raw)


class Sam2Boundary:
    def __init__(self) -> None:
        import torch
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = SAM2ImagePredictor(build_sam2_hf(os.environ.get("SAM2_HF_MODEL", "facebook/sam2.1-hiera-small"), device=device))

    def road_geometry(self, rgb: np.ndarray, decision: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        self.predictor.set_image(rgb)
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = decision["coarse_box"]
        box = np.asarray([x1 * w, y1 * h, x2 * w, y2 * h], dtype=np.float32)
        points = decision["positive_points"] + decision["negative_points"]
        coordinates = np.asarray([[x * w, y * h] for x, y in points], dtype=np.float32)
        labels = np.asarray([1] * len(decision["positive_points"]) + [0] * len(decision["negative_points"]), dtype=np.int32)
        masks, scores, _ = self.predictor.predict(box=box, point_coords=coordinates, point_labels=labels, multimask_output=True)
        within = np.zeros((h, w), dtype=bool)
        within[max(0, int(y1*h)):min(h, int(np.ceil(y2*h))), max(0, int(x1*w)):min(w, int(np.ceil(x2*w)))] = True
        return masks[int(np.argmax(scores))].astype(bool) & within, within


def apply_decision(baseline: np.ndarray, road: np.ndarray, within: np.ndarray, action: str) -> np.ndarray:
    result = baseline.copy()
    if action == "add_drivable":
        result[road & (baseline != 0)] = 0
    elif action == "remove_drivable":
        # Positive points denote road to retain; SAM's complement inside Gemma's box is explicitly excluded.
        result[within & ~road & (baseline == 0)] = 2
    return result


def draw_decision(image: np.ndarray, baseline: np.ndarray, decision: dict[str, Any], frame: str) -> np.ndarray:
    result = common.overlay(image, baseline, [], frame)
    color = {"keep": (255, 255, 255), "review_needed": (0, 230, 230), "add_drivable": (0, 200, 0), "remove_drivable": (0, 0, 230)}[decision["action"]]
    h, w = image.shape[:2]
    if decision["coarse_box"]:
        x1, y1, x2, y2 = decision["coarse_box"]
        cv2.rectangle(result, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), color, 3)
    for x, y in decision["positive_points"]:
        cv2.drawMarker(result, (int(x*w), int(y*h)), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
    for x, y in decision["negative_points"]:
        cv2.drawMarker(result, (int(x*w), int(y*h)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
    cv2.putText(result, f"GEMMA: {decision['action']}", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 3)
    return result


def triptych(original: np.ndarray, before: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    tiles = [cv2.resize(item, (960, 540), interpolation=cv2.INTER_AREA) for item in (original, before, candidate)]
    for tile, label in zip(tiles, ("ORIGINAL RGB", "BASELINE BEFORE", "SAM2 CANDIDATE AFTER")):
        cv2.putText(tile, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 3)
    return np.hstack(tiles)


def combined_review_overlay(original: np.ndarray, gemma: np.ndarray, sam: np.ndarray, final: np.ndarray,
                            frame: str, action: str, verdict: str | None, review_needed: bool) -> np.ndarray:
    """Single human-review image: ORIGINAL | GEMMA | SAM2 | FINAL."""
    size = (640, 360)
    panels = [cv2.resize(item, size, interpolation=cv2.INTER_AREA) for item in (original, gemma, sam, final)]
    labels = ("ORIGINAL", "GEMMA", "SAM2" if verdict else "SAM2 NOT USED", "FINAL")
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, .82, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(panel, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, .82, (0, 0, 0), 1, cv2.LINE_AA)
    result = np.hstack(panels)
    status = verdict or "NOT_USED"
    footer = f"{frame} | action={action} | SAM2={status} | review_needed={str(review_needed).lower()}"
    cv2.rectangle(result, (0, result.shape[0]-32), (result.shape[1], result.shape[0]), (0, 0, 0), -1)
    cv2.putText(result, footer, (12, result.shape[0]-9), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def prepare() -> dict[str, Any]:
    ensure_dirs()
    state_path = OUTPUT / "run_state.json"
    if state_path.exists():
        return common.read_json(state_path)
    frames, candidates = select_candidates()
    if not COMBINED_REVIEW_OVERLAY:
        for item in candidates:
            frame = item["frame"]
            image = cv2.imread(str(paths(frame)["image"]), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(paths(frame)["segmentation"]), cv2.IMREAD_UNCHANGED)
            preview = common.overlay(image, mask, common.parse_yolo(paths(frame)["detection"]), frame)
            cv2.putText(preview, f"mask geometry score={item['geometry_score']}", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 3)
            cv2.imwrite(str(OUTPUT / "candidate_previews" / f"{frame}.jpg"), preview)
    state = {"version": 1, "pilot": "boundary_error_focus", "model": common.MODEL, "frames": frames, "candidate_records": candidates,
             "status": "prepared", "processed": {}, "failed": {}, "source_inventory_before": common.source_inventory(frames),
             "auth": None, "text_test": None, "cvat": None, "gemma_initial_review_frames": []}
    write_json(state_path, state)
    return state


def process_frame(frame: str, cloudflare: Cloudflare, sam: Sam2Boundary | None) -> tuple[dict[str, Any], Sam2Boundary | None]:
    src = paths(frame)
    image = cv2.imread(str(src["image"]), cv2.IMREAD_COLOR)
    baseline = cv2.imread(str(src["segmentation"]), cv2.IMREAD_UNCHANGED)
    if image is None or baseline is None or baseline.ndim != 2 or image.shape[:2] != baseline.shape or set(np.unique(baseline)) - {0, 1, 2}:
        raise RuntimeError("invalid source image/mask")
    detections = common.parse_yolo(src["detection"])
    before = common.overlay(image, baseline, detections, frame)
    review_path = OUTPUT / "review_results" / f"{frame}.json"
    if review_path.exists():
        stored = common.read_json(review_path)
        judgment = stored.get("gemma_judgment", stored) if isinstance(stored, dict) else stored
        validate_judgment(judgment, frame, len(detections)); usage: dict[str, Any] = {}
    else:
        judgment, usage = cloudflare.review(frame, np.hstack((image, before)), detections)
        write_json(review_path, judgment)
    decision = judgment["segmentation_decision"]
    updated_detections = common.apply_detections(detections, judgment["detection_changes"])
    candidate = baseline.copy()
    geometry: dict[str, Any] | None = None
    geometry_usage: dict[str, Any] = {}
    sam_used = decision["action"] in {"add_drivable", "remove_drivable"}
    if sam_used:
        sam = sam or Sam2Boundary()
        road, within = sam.road_geometry(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), decision)
        candidate = apply_decision(baseline, road, within, decision["action"])
        candidate_view = common.overlay(image, candidate, updated_detections, frame)
        candidate_before_qc = candidate.copy()
        geometry, geometry_usage = cloudflare.geometry_qc(frame, triptych(image, before, candidate_view), decision)
        if not COMBINED_REVIEW_OVERLAY:
            write_json(OUTPUT / "geometry_reviews" / f"{frame}.json", {**geometry, "usage": geometry_usage, "attempts": 1})
        if geometry["verdict"] != "accept":
            candidate = baseline.copy()
    else:
        candidate_before_qc = candidate.copy()
    candidate_view = common.overlay(image, candidate_before_qc, updated_detections, frame)
    if not COMBINED_REVIEW_OVERLAY:
        cv2.imwrite(str(OUTPUT / "sam_candidates" / f"{frame}.png"), candidate_before_qc)
    cv2.imwrite(str(OUTPUT / "images" / f"{frame}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(str(OUTPUT / "segmentation" / f"{frame}.png"), candidate)
    common.write_yolo(OUTPUT / "detection" / f"{frame}.txt", updated_detections)
    meta_in = common.read_json(src["metadata"])
    weather = {"맑음": "sunny", "흐림": "cloudy"}.get(meta_in.get("weather"))
    if meta_in.get("frame_id") != frame or weather is None:
        raise RuntimeError("metadata cannot be converted without guessing")
    write_json(OUTPUT / "metadata" / f"{frame}.json", {"frame_id": frame, "location": meta_in.get("location"), "weather": weather,
               "surface_condition": judgment["surface_condition"], "camera": meta_in.get("camera"), "note": meta_in.get("note", [])})
    final_view = common.overlay(image, candidate, updated_detections, frame)
    gemma_view = draw_decision(image, baseline, decision, frame)
    review_needed = decision["action"] == "review_needed" or any(item["review_needed"] for item in judgment["vegetation_reviews"])
    if COMBINED_REVIEW_OVERLAY:
        cv2.imwrite(str(OUTPUT / "overlays" / f"{frame}_review.jpg"),
                    combined_review_overlay(image, gemma_view, candidate_view, final_view, frame, decision["action"],
                                            geometry["verdict"] if geometry else None, review_needed))
        final_action = decision["action"] if not geometry or geometry["verdict"] == "accept" else "keep"
        write_json(review_path, {"frame_id": frame, "segmentation_action": decision["action"], "detection_changes": judgment["detection_changes"],
                                 "vegetation_type": sorted({item["vegetation_type"] for item in judgment["vegetation_reviews"]}),
                                 "gemma_confidence": decision["confidence"], "sam2_used": sam_used,
                                 "positive_points": decision["positive_points"], "negative_points": decision["negative_points"],
                                 "geometry_qc": {**geometry, "usage": geometry_usage, "retry_count": 0} if geometry else {"verdict": "not_used", "retry_count": 0},
                                 "final_action": final_action, "review_needed": review_needed, "reason": decision["description"], "gemma_judgment": judgment})
    else:
        cv2.imwrite(str(OUTPUT / "overlays" / "baseline_before" / f"{frame}.jpg"), before)
        cv2.imwrite(str(OUTPUT / "overlays" / "gemma_decision" / f"{frame}.jpg"), gemma_view)
        cv2.imwrite(str(OUTPUT / "overlays" / "sam_candidate" / f"{frame}.jpg"), candidate_view)
        cv2.imwrite(str(OUTPUT / "overlays" / "final_after" / f"{frame}.jpg"), final_view)
    return {"action": decision["action"], "summary": judgment["summary"],
            "vegetation_types": sorted({item["vegetation_type"] for item in judgment["vegetation_reviews"]}),
            "review_needed": review_needed,
            "sam2_used": sam_used, "positive_negative_points_used": sam_used,
            "geometry_verdict": geometry["verdict"] if geometry else None,
            "segmentation_changed": not np.array_equal(candidate, baseline),
            "detection_changed": any(item["action"] != "keep" for item in judgment["detection_changes"]),
            "usage": usage, "geometry_usage": geometry_usage}, sam


def run() -> dict[str, Any]:
    state = prepare()
    state_path = OUTPUT / "run_state.json"
    cloudflare = Cloudflare()
    if not state["auth"]:
        state["auth"] = cloudflare.auth(); write_json(state_path, state)
    if not state["text_test"]:
        state["text_test"] = cloudflare.text_test(); write_json(state_path, state)
    sam: Sam2Boundary | None = None
    for frame in state["frames"]:
        if frame in state["processed"]:
            continue
        try:
            result, sam = process_frame(frame, cloudflare, sam)
            state["processed"][frame] = result
            if frame not in state["gemma_initial_review_frames"]:
                state["gemma_initial_review_frames"].append(frame)
            state["failed"].pop(frame, None)
        except Exception as error:
            state["failed"][frame] = f"{type(error).__name__}: {error}"
            write_json(state_path, state)
            raise
        write_json(state_path, state)
    state["status"] = "complete"
    write_json(state_path, state)
    return state


def qc() -> dict[str, Any]:
    state = common.read_json(OUTPUT / "run_state.json")
    frames, expected, errors = state["frames"], set(state["frames"]), []
    counts: dict[str, int] = {}
    specs = [("images", ".jpg"), ("segmentation", ".png"), ("detection", ".txt"), ("metadata", ".json"), ("review_results", ".json")]
    if not COMBINED_REVIEW_OVERLAY:
        specs.append(("sam_candidates", ".png"))
    for directory, suffix in specs:
        found = {path.stem for path in (OUTPUT / directory).glob(f"frame_*{suffix}")}
        counts[directory] = len(found)
        if found != expected:
            errors.append(f"{directory} frame stems differ")
    if COMBINED_REVIEW_OVERLAY:
        found = {path.name.removesuffix("_review.jpg") for path in (OUTPUT / "overlays").glob("frame_*_review.jpg")}
        counts["overlays"] = len(found)
        if found != expected:
            errors.append("combined review overlay frame stems differ")
    else:
        for directory in ("baseline_before", "gemma_decision", "sam_candidate", "final_after"):
            found = {path.stem for path in (OUTPUT / "overlays" / directory).glob("frame_*.jpg")}
            counts[f"overlays/{directory}"] = len(found)
            if found != expected:
                errors.append(f"overlay {directory} frame stems differ")
    for frame in frames:
        image = cv2.imread(str(OUTPUT / "images" / f"{frame}.jpg"))
        mask = cv2.imread(str(OUTPUT / "segmentation" / f"{frame}.png"), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None or mask.ndim != 2 or image.shape[:2] != mask.shape or set(np.unique(mask)) - {0, 1, 2}:
            errors.append(f"{frame} image/mask")
        try:
            common.parse_yolo(OUTPUT / "detection" / f"{frame}.txt")
        except Exception as error:
            errors.append(str(error))
        meta = common.read_json(OUTPUT / "metadata" / f"{frame}.json")
        if set(meta) != common.META_KEYS or meta.get("frame_id") != frame or meta.get("weather") not in {"sunny", "cloudy", "after_rain"} or meta.get("surface_condition") not in {"dry", "wet"}:
            errors.append(f"{frame} metadata")
    unchanged = common.source_inventory(frames) == state["source_inventory_before"]
    if not unchanged:
        errors.append("source validation inventory changed")
    if len(state["gemma_initial_review_frames"]) != FRAME_COUNT or set(state["gemma_initial_review_frames"]) != expected:
        errors.append("initial Gemma review is not exactly ten unique frames")
    actions = [item.get("action") for item in state["processed"].values()]
    vegetation = [kind for item in state["processed"].values() for kind in item.get("vegetation_types", [])]
    geometry = [item.get("geometry_verdict") for item in state["processed"].values()]
    detections_changed = sum(bool(item.get("detection_changed")) for item in state["processed"].values())
    detection_actions: list[str] = []
    for frame in frames:
        record = common.read_json(OUTPUT / "review_results" / f"{frame}.json")
        detection_actions.extend(change.get("action") for change in record.get("detection_changes", []) if isinstance(change, dict))
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "neurons": 0.0}
    for item in state["processed"].values():
        for record in (item.get("usage", {}), item.get("geometry_usage", {})):
            for key in usage:
                if isinstance(record.get(key), (int, float)):
                    usage[key] += record[key]
    statistics = {"keep": actions.count("keep"), "add_drivable": actions.count("add_drivable"), "remove_drivable": actions.count("remove_drivable"),
                  "review_needed": sum(bool(item.get("review_needed")) for item in state["processed"].values()),
                  "soft_vegetation": vegetation.count("soft_vegetation"), "rigid_vegetation": vegetation.count("rigid_vegetation"), "uncertain_vegetation": vegetation.count("uncertain_vegetation"),
                  "sam2_used": sum(bool(item.get("sam2_used")) for item in state["processed"].values()), "sam2_accept": geometry.count("accept"),
                  "sam2_reject": geometry.count("reject"), "sam2_retry": 0, "detection_frames_changed": detections_changed,
                  "detection_add": detection_actions.count("add"), "detection_delete": detection_actions.count("delete"), "detection_adjust": detection_actions.count("adjust"),
                  "gemma_api_failures": len(state["failed"]), "workers_ai_observed_usage": usage}
    result = {"passed": not errors, "errors": errors, "counts": counts, "statistics": statistics, "source_validation_unchanged": unchanged,
              "initial_gemma_unique_frames": len(state["gemma_initial_review_frames"]), "processed": len(state["processed"]), "failed": state["failed"]}
    write_json(OUTPUT / "qc_report.json", result)
    return result


def render_overlays() -> dict[str, Any]:
    """Regenerate only combined review images from stored decisions/final labels; no model call."""
    if not COMBINED_REVIEW_OVERLAY:
        raise RuntimeError("render-overlays is only used by the combined review layout")
    state = common.read_json(OUTPUT / "run_state.json")
    for frame in state["frames"]:
        record = common.read_json(OUTPUT / "review_results" / f"{frame}.json")
        judgment = record["gemma_judgment"]
        image = cv2.imread(str(paths(frame)["image"]), cv2.IMREAD_COLOR)
        baseline = cv2.imread(str(paths(frame)["segmentation"]), cv2.IMREAD_UNCHANGED)
        final_mask = cv2.imread(str(OUTPUT / "segmentation" / f"{frame}.png"), cv2.IMREAD_UNCHANGED)
        final_detections = common.parse_yolo(OUTPUT / "detection" / f"{frame}.txt")
        gemma = draw_decision(image, baseline, judgment["segmentation_decision"], frame)
        final = common.overlay(image, final_mask, final_detections, frame)
        geometry = record.get("geometry_qc", {})
        verdict = geometry.get("verdict")
        if verdict == "not_used":
            verdict = None
        cv2.imwrite(str(OUTPUT / "overlays" / f"{frame}_review.jpg"),
                    combined_review_overlay(image, gemma, final, final, frame, record["segmentation_action"], verdict,
                                            bool(record["review_needed"])))
    return {"rendered": len(state["frames"])}


def cvat() -> dict[str, Any]:
    token = os.environ.get("CVAT_ACCESS_TOKEN", "").strip()
    url = os.environ.get("CVAT_URL", "").rstrip("/")
    project_id = os.environ.get("CVAT_PROJECT_ID", "1")
    if not token or not url:
        raise RuntimeError("CVAT_ACCESS_TOKEN or CVAT_URL is missing; no CVAT write attempted")
    response = requests.get(f"{url}/api/projects/{project_id}", headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if not response.ok:
        result = {"uploaded": False, "http_status": response.status_code, "error": response.text[:500]}
        state_path = OUTPUT / "run_state.json"; state = common.read_json(state_path); state["cvat"] = result; write_json(state_path, state)
        raise RuntimeError(f"CVAT HTTP {response.status_code}: {response.text[:500]}")
    raise RuntimeError("CVAT authentication succeeded but upload is intentionally stopped pending a verified project-label mapping")


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "qc", "cvat", "render-overlays"])
    args = parser.parse_args()
    result = {"prepare": prepare, "run": run, "qc": qc, "cvat": cvat, "render-overlays": render_overlays}[args.command]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
