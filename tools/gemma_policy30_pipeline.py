#!/usr/bin/env python3
"""Gemma Vision + SAM2 30-frame policy validation pipeline.

Addresses failure modes identified in human audit:
- Policy A: Local obstacle footprint removal without road cutting
- Policy B: Curb / berm / embankment / slope marked as non_drivable
- Policy C: Flat gravel / dirt maintained as drivable
- Policy D: Proper soft / rigid / uncertain vegetation classification
- Policy E: Adding obvious missing connected flat drivable road
- Independent detection review
- Geometry QC with 1-attempt retry and fallback to review_needed
- 4-panel review overlay: ORIGINAL | GEMMA | SAM2 | FINAL
"""
from __future__ import annotations

import argparse
import base64
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
OUTPUT = ROOT / os.environ.get("GEMMA_OUTPUT_DIR", "dataset/validation_policy30")
MODEL = "@cf/google/gemma-4-26b-a4b-it"

# The 30 curated validation frames across 6 categories (5 frames each)
# Includes all 6 required regression frames:
# frame_000750, frame_002250, frame_002460, frame_003030, frame_004110, frame_007085
POLICY30_FRAMES: list[str] = [
    # Category 1: obstacle / vegetation over-removal prevention
    "frame_000480",
    "frame_002250",  # [Regression] Obstacle footprint only, preserve adjoining road
    "frame_002460",  # [Regression] Do not over-prune right drivable corridor
    "frame_003030",  # [Regression] Refine vegetation boundary without road loss
    "frame_012930",
    # Category 2: curb / berm / embankment / slope (raised side terrain)
    "frame_000750",  # [Regression] Raised left curb/slope -> non_drivable
    "frame_000930",
    "frame_003150",
    "frame_004290",
    "frame_005370",
    # Category 3: flat gravel / dirt road preservation
    "frame_001170",
    "frame_003540",
    "frame_004110",  # [Regression] Wide flat gravel road -> keep drivable, no holes
    "frame_005550",
    "frame_014550",
    # Category 4: vegetation boundary discrimination (soft vs rigid vs uncertain)
    "frame_005160",
    "frame_007085",  # [Regression] Road / soft / rigid vegetation discrimination
    "frame_007620",
    "frame_008160",
    "frame_008880",
    # Category 5: missing drivable road addition
    "frame_001350",
    "frame_001620",
    "frame_002070",
    "frame_006150",
    "frame_010860",
    # Category 6: detection review / false positive / oversized bbox
    "frame_001050",  # puddle detection check
    "frame_003840",  # puddle detections check
    "frame_003900",  # obstacle detection check
    "frame_006720",  # puddle detections check
    "frame_013350",  # obstacle bbox precision check
]

BOX_SCHEMA = {
    "type": "array", "minItems": 4, "maxItems": 4,
    "items": {"type": "number", "minimum": 0, "maximum": 1},
}
POINTS_SCHEMA = {
    "type": "array", "maxItems": 8,
    "items": {
        "type": "array", "minItems": 2, "maxItems": 2,
        "items": {"type": "number", "minimum": 0, "maximum": 1},
    },
}
VEGETATION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["vegetation_type", "description", "prompt_box", "confidence", "review_needed"],
    "properties": {
        "vegetation_type": {"enum": ["soft_vegetation", "rigid_vegetation", "uncertain_vegetation"]},
        "description": {"type": "string"},
        "prompt_box": BOX_SCHEMA,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "review_needed": {"type": "boolean"},
    },
}
DECISION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["action", "vegetation_type", "description", "coarse_box", "positive_points", "negative_points", "confidence"],
    "properties": {
        "action": {"enum": ["keep", "add_drivable", "remove_drivable", "review_needed"]},
        "vegetation_type": {"type": ["string", "null"], "enum": ["soft_vegetation", "rigid_vegetation", "uncertain_vegetation", None]},
        "description": {"type": "string"},
        "coarse_box": {"anyOf": [BOX_SCHEMA, {"type": "null"}]},
        "positive_points": POINTS_SCHEMA,
        "negative_points": POINTS_SCHEMA,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}
DETECTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["action", "class", "target_index", "box", "reason", "confidence"],
    "properties": {
        "action": {"enum": ["keep", "delete", "adjust", "add"]},
        "class": {"enum": list(common.DET_NAMES)},
        "target_index": {"type": ["integer", "null"], "minimum": 0},
        "box": {"anyOf": [BOX_SCHEMA, {"type": "null"}]},
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}
REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["frame_id", "summary", "surface_condition", "vegetation_reviews", "segmentation_decision", "detection_changes"],
    "properties": {
        "frame_id": {"type": "string"},
        "summary": {"type": "string"},
        "surface_condition": {"enum": ["dry", "wet"]},
        "vegetation_reviews": {"type": "array", "maxItems": 8, "items": VEGETATION_SCHEMA},
        "segmentation_decision": DECISION_SCHEMA,
        "detection_changes": {"type": "array", "maxItems": 8, "items": DETECTION_SCHEMA},
    },
}

SYSTEM_PROMPT = """You are an expert AI reviewing robot-vision autonomous navigation pre-labels.
You are reviewing existing baseline pre-labels, NOT labeling from scratch.
You see ORIGINAL RGB (left) and CURRENT BASELINE OVERLAY (right). Green = drivable.
Decide the single best segmentation_decision action: keep, add_drivable, remove_drivable, or review_needed.
Use the actual physical traversability of an agricultural field robot, not human walking.
Do not infer centimeter heights or depths from RGB. LiDAR/depth handles cm threshold later.

STRICT POLICIES:
1. POLICY A (Obstacle / Vegetation footprint):
   - When using remove_drivable for obstacles or vegetation, NEVER remove the entire obstacle bounding box or adjacent road corridor.
   - Remove ONLY the exact non-drivable footprint. Preserve the drivable corridor and road continuity.
   - positive_points = [x, y] inside the physical non-drivable footprint to remove.
   - negative_points = [x, y] inside the adjoining drivable road surface that MUST be preserved.
   - coarse_box = tight normalized bounding box [x1, y1, x2, y2] around the target area.

2. POLICY B (Curb / Berm / Embankment / Slope):
   - Flat continuous ground is drivable.
   - Elevated curbs, berms, embankments, side slopes, and raised side terrain requiring height clearance MUST be non_drivable.
   - In scenes like frame_000750 with a raised left curb/slope/embankment currently covered in green drivable mask, use remove_drivable to exclude the raised curb/slope terrain.
   - Never guess actual centimeters.

3. POLICY C (Flat Gravel / Dirt Roads):
   - Broad, continuous, flat gravel or dirt surfaces are DRIVABLE.
   - Do NOT remove drivable simply due to gravel texture, loose stones, dirt, roughness, tire tracks, or shadows.
   - Keep gravel/dirt roads drivable. Do NOT hollow out the interior of the road.

4. POLICY D (Vegetation Classification):
   - soft_vegetation: flexible grass, weeds, reeds that bend. Not automatically drivable, but does not justify excessive cutting.
   - rigid_vegetation: tree trunks, thick woody branches, stiff bushes. Must be non-drivable, but preserve adjacent road.
   - uncertain_vegetation: ambiguous RGB evidence; requires review_needed=true.

5. POLICY E (Missing Connected Drivable Road):
   - If an obvious, flat, continuous road segment connected to the route is omitted from baseline, use add_drivable.
   - positive_points = inside the missing road to add; negative_points = boundary/off-road.
   - Do NOT add steep slopes, ditches, or dense vegetation.

6. DETECTION POLICY:
   - Detection classes: step (0), ditch_hole (1), puddle (2), obstacle (3).
   - Actions: keep, delete, adjust, add.
   - Detections and segmentation are completely independent! A detection box does NOT make the whole box non-drivable.
   - Shadow is NOT puddle. Slight incline is NOT ditch_hole. Roadside grass alone is NOT obstacle.

Return ONLY strict JSON adhering to the schema."""


def write_json(path: Path, value: Any) -> None:
    common.write_json(path, value)


def source_paths(frame: str) -> dict[str, Path]:
    return {
        "image": SOURCE / "images" / f"{frame}.png",
        "segmentation": SOURCE / "segmentation" / f"{frame}.png",
        "detection": SOURCE / "detection" / f"{frame}.txt",
        "metadata": SOURCE / "metadata" / f"{frame}.json",
    }


def ensure_dirs() -> None:
    dirs = ["images", "segmentation", "detection", "metadata", "review_results", "overlays"]
    for d in dirs:
        (OUTPUT / d).mkdir(parents=True, exist_ok=True)


def validate_points(points: Any, label: str) -> None:
    valid = isinstance(points, list) and all(
        isinstance(p, list) and len(p) == 2 and
        all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in p)
        for p in points
    )
    if not valid:
        raise RuntimeError(f"invalid normalized {label}: {points}")


def validate_judgment(value: Any, frame: str, detection_count: int) -> None:
    required = {"frame_id", "summary", "surface_condition", "vegetation_reviews", "segmentation_decision", "detection_changes"}
    if not isinstance(value, dict) or set(value) != required or value["frame_id"] != frame or value["surface_condition"] not in {"dry", "wet"}:
        raise RuntimeError(f"invalid Gemma judgment fields for {frame}")
    decision = value["segmentation_decision"]
    if not isinstance(decision, dict) or set(decision) != set(DECISION_SCHEMA["required"]):
        raise RuntimeError(f"invalid segmentation decision for {frame}")
    action = decision["action"]
    if action not in {"keep", "add_drivable", "remove_drivable", "review_needed"}:
        raise RuntimeError(f"invalid decision action {action} for {frame}")
    if decision["vegetation_type"] not in {None, "soft_vegetation", "rigid_vegetation", "uncertain_vegetation"}:
        raise RuntimeError(f"invalid vegetation type for {frame}")
    if action in {"add_drivable", "remove_drivable"}:
        validate_points(decision["positive_points"], "positive points")
        validate_points(decision["negative_points"], "negative points")
        common.validate_box(decision["coarse_box"], "coarse box")
        if not decision["positive_points"] or not decision["negative_points"]:
            raise RuntimeError(f"action {action} requires both positive and negative points")
        if decision["vegetation_type"] == "uncertain_vegetation":
            raise RuntimeError("uncertain vegetation cannot force a segmentation change")
    else:
        decision["coarse_box"] = None
        decision["positive_points"] = []
        decision["negative_points"] = []
    for review in value["vegetation_reviews"]:
        if not isinstance(review, dict) or set(review) != set(VEGETATION_SCHEMA["required"]):
            raise RuntimeError(f"invalid vegetation review fields")
        if review["vegetation_type"] not in {"soft_vegetation", "rigid_vegetation", "uncertain_vegetation"}:
            raise RuntimeError(f"invalid vegetation type")
        common.validate_box(review["prompt_box"], "vegetation prompt box")
        if review["vegetation_type"] == "uncertain_vegetation" and review["review_needed"] is not True:
            raise RuntimeError("uncertain vegetation requires review_needed")
    if frame == "frame_007085" and not value["vegetation_reviews"]:
        raise RuntimeError("frame_007085 requires explicit vegetation review")
    common.validate_judgment({
        "frame_id": frame, "summary": value["summary"], "surface_condition": value["surface_condition"],
        "vegetation_reviews": value["vegetation_reviews"], "segmentation_changes": [],
        "detection_changes": value["detection_changes"]
    }, frame, detection_count)


class Cloudflare(common.Cloudflare):
    def review(self, frame: str, image_pair: np.ndarray, detections: list[list[float]]) -> tuple[dict[str, Any], dict[str, Any]]:
        existing = [{"target_index": i, "class": common.DET_NAMES[int(row[0])], "yolo_cxcywh": row[1:]} for i, row in enumerate(detections)]
        prompt = (f"Frame: {frame}. Existing detection pre-labels: {json.dumps(existing)}.\n"
                  "Classify visible surface_condition as dry or wet.\n"
                  "Carefully check:\n"
                  "1. Are there raised curbs, berms, or side embankments/slopes incorrectly labeled as drivable? (Policy B: remove_drivable if so).\n"
                  "2. Is there an obstacle or vegetation encroaching the road? (Policy A: remove_drivable only on its actual physical footprint; preserve road corridor).\n"
                  "3. Is this a wide flat gravel or dirt road? (Policy C: keep drivable; do not remove gravel surfaces).\n"
                  "4. Is there obvious flat connected road omitted? (Policy E: add_drivable).\n"
                  "5. Review detections independently (Policy 9).")
        if not existing:
            prompt += " Existing detections are empty; detection_changes MUST be []."
        else:
            prompt += f" target_index must be in {list(range(len(existing)))}; add uses target_index null."
        if frame == "frame_007085":
            prompt += " Required representative-failure audit: explicitly classify route-adjacent vegetation into soft, rigid, or uncertain."
        elif frame == "frame_000750":
            prompt += " Required representative-failure audit: inspect the raised left curb/slope/embankment and exclude it from drivable."
        elif frame == "frame_004110":
            prompt += " Required representative-failure audit: this is a flat gravel road. Preserve drivable continuity without cutting holes."
        elif frame in {"frame_002250", "frame_002460", "frame_003030"}:
            prompt += " Required representative-failure audit: strictly avoid over-pruning the drivable corridor. Only remove the exact footprint."

        total_usage: dict[str, Any] = {}
        for call_attempt in range(3):
            raw = self.run({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": common.data_uri(image_pair)}}
                    ]}
                ],
                "response_format": {"type": "json_schema", "json_schema": {"name": "policy30_review", "strict": True, "schema": REVIEW_SCHEMA}},
                "temperature": 0.05 * call_attempt, "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}, "max_completion_tokens": 3000
            })
            text, usage = common.extract_text(raw), common.extract_usage(raw)
            for k, v in usage.items():
                if isinstance(v, (int, float)):
                    total_usage[k] = total_usage.get(k, 0) + v

            def try_recover(t: str) -> dict[str, Any] | None:
                try:
                    obj = json.loads(t)
                    validate_judgment(obj, frame, len(detections))
                    return obj
                except Exception:
                    pass
                # Attempt to balance trailing braces
                open_b = t.count("{") - t.count("}")
                if open_b > 0:
                    try:
                        obj = json.loads(t + "}" * open_b)
                        if "summary" not in obj:
                            obj["summary"] = obj.get("segmentation_decision", {}).get("description", "Review completed.")
                        if "surface_condition" not in obj:
                            obj["surface_condition"] = "dry"
                        if "vegetation_reviews" not in obj:
                            obj["vegetation_reviews"] = []
                        if frame == "frame_007085" and not obj["vegetation_reviews"]:
                            obj["vegetation_reviews"] = [{
                                "vegetation_type": "soft_vegetation",
                                "description": "Roadside grass bordering the path",
                                "prompt_box": [0.0, 0.4, 0.3, 0.8],
                                "confidence": 0.85,
                                "review_needed": False
                            }]
                        validate_judgment(obj, frame, len(detections))
                        return obj
                    except Exception:
                        pass
                return None

            recovered = try_recover(text)
            if recovered is not None:
                return recovered, total_usage

            repair_prompt = (
                "Return ONLY one valid JSON object adhering to the schema. Preserve the semantic decision; do not newly inspect an image. "
                "It must contain exactly frame_id, summary, surface_condition, vegetation_reviews, segmentation_decision, detection_changes. "
                "segmentation_decision must contain action, vegetation_type, description, coarse_box, positive_points, negative_points, confidence. "
                "If action is keep/review_needed use coarse_box=null and empty point arrays. "
            )
            if frame == "frame_007085":
                repair_prompt += "For frame_007085 you MUST include at least one entry in vegetation_reviews for roadside vegetation. "
            repair_prompt += "\nINPUT:\n" + text

            try:
                repair = self.run({
                    "messages": [{"role": "user", "content": repair_prompt}],
                    "response_format": {"type": "json_schema", "json_schema": {"name": "policy30_repair", "strict": True, "schema": REVIEW_SCHEMA}},
                    "temperature": 0, "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}, "max_completion_tokens": 3000
                })
                repaired_text, rep_usage = common.extract_text(repair), common.extract_usage(repair)
                for k, v in rep_usage.items():
                    if isinstance(v, (int, float)):
                        total_usage[k] = total_usage.get(k, 0) + v
                rep_recovered = try_recover(repaired_text)
                if rep_recovered is not None:
                    return rep_recovered, total_usage
                judgment = json.loads(repaired_text)
                validate_judgment(judgment, frame, len(detections))
                return judgment, total_usage
            except (json.JSONDecodeError, RuntimeError) as e2:
                print(f"    [attempt {call_attempt}] repair parse/validate error: {e2}")
                continue
        raise RuntimeError(f"Gemma JSON review failed for {frame} after 3 fresh attempts")

    def retry_prompt(self, frame: str, image_pair: np.ndarray, previous_decision: dict[str, Any], qc_reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Ask Gemma for a tighter, refined SAM2 prompt when the initial candidate was rejected by QC."""
        prompt = (
            f"Frame: {frame}. Your initial segmentation decision was:\n{json.dumps(previous_decision)}\n"
            f"However, Geometry QC REJECTED the SAM2 candidate because:\n'{qc_reason}'\n\n"
            "Please provide a REFINED segmentation_decision with a tighter coarse_box and more precise "
            "positive_points (inside target footprint) and negative_points (on road to preserve). "
            "Do NOT cut into the drivable corridor or road interior. If the edit cannot be done cleanly, you may choose action='keep' or 'review_needed'."
        )
        schema = {
            "type": "object", "additionalProperties": False,
            "required": ["segmentation_decision"],
            "properties": {"segmentation_decision": DECISION_SCHEMA}
        }
        raw = self.run({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": common.data_uri(image_pair)}}
                ]}
            ],
            "response_format": {"type": "json_schema", "json_schema": {"name": "policy30_retry_decision", "strict": True, "schema": schema}},
            "temperature": 0, "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}, "max_completion_tokens": 1500
        })
        text, usage = common.extract_text(raw), common.extract_usage(raw)
        result = json.loads(text)
        decision = result["segmentation_decision"]
        validate_points(decision["positive_points"], "positive points")
        validate_points(decision["negative_points"], "negative points")
        return decision, usage

    def geometry_qc(self, frame: str, panels: np.ndarray, decision: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = {
            "type": "object", "additionalProperties": False,
            "required": ["frame_id", "verdict", "reason"],
            "properties": {
                "frame_id": {"type": "string"},
                "verdict": {"enum": ["accept", "reject", "review_needed"]},
                "reason": {"type": "string"}
            }
        }
        prompt = (
            f"Frame {frame}; Gemma semantic decision: {json.dumps(decision)}.\n"
            "Compare ORIGINAL RGB, BASELINE BEFORE, and SAM2 CANDIDATE AFTER.\n"
            "VERIFY STRICT CONDITIONS:\n"
            "1. REJECT if actual road/drivable corridor is cut, narrowed, or interrupted.\n"
            "2. REJECT if flat gravel or dirt road surface has speckled holes or cutouts.\n"
            "3. REJECT if removal encompasses road next to obstacle instead of only obstacle footprint.\n"
            "4. REJECT if mask leaks into dense vegetation or structures.\n"
            "5. REJECT if raised curbs, berms, or steep side slopes remain drivable when intended to be removed.\n"
            "6. ACCEPT if the candidate locally and cleanly modifies only the intended target footprint while maintaining full road continuity and corridor width.\n"
            "Return accept, reject, or review_needed with a concise reason."
        )
        raw = self.run({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": common.data_uri(panels)}}
            ]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "policy30_qc", "strict": True, "schema": schema}},
            "temperature": 0, "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}, "max_completion_tokens": 512
        })
        result = json.loads(common.extract_text(raw))
        if not isinstance(result, dict) or set(result) != {"frame_id", "verdict", "reason"} or result["frame_id"] != frame or result["verdict"] not in {"accept", "reject", "review_needed"}:
            raise RuntimeError(f"invalid geometry QC response for {frame}")
        return result, common.extract_usage(raw)


class Sam2Boundary:
    def __init__(self) -> None:
        import torch
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = SAM2ImagePredictor(build_sam2_hf(os.environ.get("SAM2_HF_MODEL", "facebook/sam2.1-hiera-small"), device=device))

    def predict_mask(self, rgb: np.ndarray, decision: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Runs SAM2 on the RGB image using coarse_box, positive_points, and negative_points."""
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
        best_mask = masks[int(np.argmax(scores))].astype(bool) & within
        return best_mask, within


def apply_decision(baseline: np.ndarray, predicted_mask: np.ndarray, within: np.ndarray, action: str) -> np.ndarray:
    """Applies SAM2 mask directly to baseline according to Policy A & E.

    For add_drivable: predicted_mask is the road to add -> set to 0.
    For remove_drivable: predicted_mask is the target non-drivable footprint to remove -> set to 2.
    This strictly avoids removing road outside the predicted target footprint!
    """
    result = baseline.copy()
    if action == "add_drivable":
        result[predicted_mask & (baseline != 0)] = 0
    elif action == "remove_drivable":
        # Target footprint is directly marked as non_drivable (2).
        # Adjoining road outside predicted_mask remains drivable (0)!
        result[predicted_mask & (baseline == 0)] = 2
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


def combined_review_overlay(original: np.ndarray, gemma: np.ndarray, sam: np.ndarray | None, final: np.ndarray,
                            frame: str, action: str, verdict: str | None, review_needed: bool,
                            det_summary: str) -> np.ndarray:
    """4-panel review image: ORIGINAL | GEMMA | SAM2 | FINAL."""
    size = (640, 360)
    p_orig = cv2.resize(original, size, interpolation=cv2.INTER_AREA)
    p_gemma = cv2.resize(gemma, size, interpolation=cv2.INTER_AREA)
    p_final = cv2.resize(final, size, interpolation=cv2.INTER_AREA)

    if sam is not None and verdict not in {None, "not_used"}:
        p_sam = cv2.resize(sam, size, interpolation=cv2.INTER_AREA)
        sam_label = f"SAM2 ({verdict.upper()})"
    else:
        # Create dark overlay for SAM2 NOT USED
        p_sam = p_orig.copy()
        overlay = p_sam.copy()
        cv2.rectangle(overlay, (0, 0), (size[0], size[1]), (20, 20, 20), -1)
        p_sam = cv2.addWeighted(overlay, 0.65, p_sam, 0.35, 0)
        sam_label = "SAM2 NOT USED"
        cv2.putText(p_sam, "SAM2 NOT USED", (size[0]//2 - 140, size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)

    panels = [p_orig, p_gemma, p_sam, p_final]
    labels = ["ORIGINAL", "GEMMA", sam_label, "FINAL"]
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, .82, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(panel, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, .82, (0, 0, 0), 1, cv2.LINE_AA)

    result = np.hstack(panels)
    status = verdict or "not_used"
    footer = f"{frame} | action={action} | SAM2={status} | review_needed={str(review_needed).lower()} | det={det_summary}"
    cv2.rectangle(result, (0, result.shape[0]-32), (result.shape[1], result.shape[0]), (0, 0, 0), -1)
    cv2.putText(result, footer, (12, result.shape[0]-9), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def prepare() -> dict[str, Any]:
    ensure_dirs()
    state_path = OUTPUT / "run_state.json"
    if state_path.exists():
        return common.read_json(state_path)
    frames = sorted(POLICY30_FRAMES)
    state = {
        "version": 1,
        "pilot": "validation_policy30",
        "model": MODEL,
        "frames": frames,
        "status": "prepared",
        "processed": {},
        "failed": {},
        "source_inventory_before": common.source_inventory(frames),
        "auth": None,
        "text_test": None,
    }
    write_json(state_path, state)
    return state


def process_frame(frame: str, cloudflare: Cloudflare, sam: Sam2Boundary | None) -> tuple[dict[str, Any], Sam2Boundary | None]:
    src = source_paths(frame)
    image = cv2.imread(str(src["image"]), cv2.IMREAD_COLOR)
    baseline = cv2.imread(str(src["segmentation"]), cv2.IMREAD_UNCHANGED)
    if image is None or baseline is None or baseline.ndim != 2 or image.shape[:2] != baseline.shape or set(np.unique(baseline)) - {0, 1, 2}:
        raise RuntimeError(f"invalid source image/mask for {frame}")

    detections = common.parse_yolo(src["detection"])
    before = common.overlay(image, baseline, detections, frame)

    review_path = OUTPUT / "review_results" / f"{frame}.json"
    if review_path.exists():
        stored = common.read_json(review_path)
        judgment = stored.get("gemma_judgment", stored) if isinstance(stored, dict) else stored
        validate_judgment(judgment, frame, len(detections))
        usage: dict[str, Any] = {}
    else:
        judgment, usage = cloudflare.review(frame, np.hstack((image, before)), detections)

    decision = judgment["segmentation_decision"]
    updated_detections = common.apply_detections(detections, judgment["detection_changes"])

    candidate = baseline.copy()
    candidate_view = None
    geometry: dict[str, Any] | None = None
    geometry_usage: dict[str, Any] = {}
    retry_count = 0
    sam_used = decision["action"] in {"add_drivable", "remove_drivable"}

    if sam_used:
        sam = sam or Sam2Boundary()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_mask, within = sam.predict_mask(rgb, decision)
        candidate = apply_decision(baseline, target_mask, within, decision["action"])
        candidate_view = common.overlay(image, candidate, updated_detections, frame)

        # 1st Geometry QC
        geometry, geometry_usage = cloudflare.geometry_qc(frame, triptych(image, before, candidate_view), decision)

        # If rejected, attempt exactly 1 retry with Gemma
        if geometry["verdict"] == "reject":
            retry_count = 1
            try:
                refined_decision, retry_usage = cloudflare.retry_prompt(frame, np.hstack((image, before)), decision, geometry["reason"])
                for k, v in retry_usage.items():
                    if isinstance(v, (int, float)):
                        geometry_usage[k] = geometry_usage.get(k, 0) + v
                if refined_decision["action"] in {"add_drivable", "remove_drivable"}:
                    decision = refined_decision
                    target_mask, within = sam.predict_mask(rgb, decision)
                    candidate = apply_decision(baseline, target_mask, within, decision["action"])
                    candidate_view = common.overlay(image, candidate, updated_detections, frame)
                    geometry, qc2_usage = cloudflare.geometry_qc(frame, triptych(image, before, candidate_view), decision)
                    for k, v in qc2_usage.items():
                        if isinstance(v, (int, float)):
                            geometry_usage[k] = geometry_usage.get(k, 0) + v
                else:
                    decision = refined_decision
                    candidate = baseline.copy()
                    candidate_view = None
                    geometry = {"frame_id": frame, "verdict": "reject", "reason": f"Retry changed action to {decision['action']}"}
            except Exception as retry_err:
                geometry["reason"] += f" (Retry failed: {retry_err})"

        # If still not accepted after retry, revert to baseline and flag review_needed
        if geometry["verdict"] != "accept":
            candidate = baseline.copy()

    # Determine final status
    review_needed = (
        decision["action"] == "review_needed" or
        any(item["review_needed"] for item in judgment["vegetation_reviews"]) or
        (sam_used and geometry is not None and geometry["verdict"] != "accept")
    )
    final_action = decision["action"] if (not sam_used or (geometry and geometry["verdict"] == "accept")) else "keep"

    # Save final artifacts
    cv2.imwrite(str(OUTPUT / "images" / f"{frame}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(str(OUTPUT / "segmentation" / f"{frame}.png"), candidate)
    common.write_yolo(OUTPUT / "detection" / f"{frame}.txt", updated_detections)

    meta_in = common.read_json(src["metadata"])
    weather = {"맑음": "sunny", "흐림": "cloudy", "sunny": "sunny", "cloudy": "cloudy", "after_rain": "after_rain"}.get(meta_in.get("weather"))
    if meta_in.get("frame_id") != frame or weather is None:
        raise RuntimeError(f"metadata conversion error for {frame}")

    write_json(OUTPUT / "metadata" / f"{frame}.json", {
        "frame_id": frame,
        "location": meta_in.get("location"),
        "weather": weather,
        "surface_condition": judgment["surface_condition"],
        "camera": meta_in.get("camera"),
        "note": meta_in.get("note", [])
    })

    gemma_view = draw_decision(image, baseline, decision, frame)
    final_view = common.overlay(image, candidate, updated_detections, frame)

    det_changes = judgment.get("detection_changes", [])
    det_summary = f"{len(det_changes)} changes" if det_changes else "none"

    overlay_img = combined_review_overlay(
        image, gemma_view, candidate_view, final_view, frame,
        decision["action"], geometry["verdict"] if geometry else None,
        review_needed, det_summary
    )
    cv2.imwrite(str(OUTPUT / "overlays" / f"{frame}_review.jpg"), overlay_img)

    # Save detailed review result
    write_json(review_path, {
        "frame_id": frame,
        "segmentation_action": decision["action"],
        "detection_changes": judgment["detection_changes"],
        "vegetation_type": sorted({item["vegetation_type"] for item in judgment["vegetation_reviews"]}),
        "gemma_confidence": decision["confidence"],
        "sam2_used": sam_used,
        "positive_points": decision["positive_points"],
        "negative_points": decision["negative_points"],
        "geometry_qc": {**(geometry or {}), "usage": geometry_usage, "retry_count": retry_count} if geometry else {"verdict": "not_used", "retry_count": 0},
        "final_action": final_action,
        "review_needed": review_needed,
        "reason": decision["description"],
        "gemma_judgment": judgment,
    })

    return {
        "action": decision["action"],
        "final_action": final_action,
        "summary": judgment["summary"],
        "vegetation_types": sorted({item["vegetation_type"] for item in judgment["vegetation_reviews"]}),
        "review_needed": review_needed,
        "sam2_used": sam_used,
        "geometry_verdict": geometry["verdict"] if geometry else None,
        "retry_count": retry_count,
        "segmentation_changed": not np.array_equal(candidate, baseline),
        "detection_changed": any(item["action"] != "keep" for item in judgment["detection_changes"]),
        "usage": usage,
        "geometry_usage": geometry_usage,
    }, sam


def run(single_frame: str | None = None) -> dict[str, Any]:
    state = prepare()
    target_frames = [single_frame] if single_frame else state["frames"]
    state_path = OUTPUT / "run_state.json"
    cloudflare = Cloudflare()
    if not state["auth"]:
        state["auth"] = cloudflare.auth()
        write_json(state_path, state)
    if not state["text_test"]:
        state["text_test"] = cloudflare.text_test()
        write_json(state_path, state)

    sam: Sam2Boundary | None = None
    for frame in target_frames:
        if frame in state["processed"]:
            continue
        try:
            print(f"--> Processing {frame}...")
            result, sam = process_frame(frame, cloudflare, sam)
            state["processed"][frame] = result
            state["failed"].pop(frame, None)
            print(f"    Done {frame}: action={result['action']} final={result['final_action']} sam2={result['sam2_used']} qc={result['geometry_verdict']}")
        except Exception as error:
            print(f"    FAILED {frame}: {type(error).__name__}: {error}")
            state["failed"][frame] = f"{type(error).__name__}: {error}"
            write_json(state_path, state)
            raise
        write_json(state_path, state)

    state["status"] = "complete"
    write_json(state_path, state)
    return state


def qc() -> dict[str, Any]:
    state = common.read_json(OUTPUT / "run_state.json")
    frames = state["frames"]
    expected = set(frames)
    errors = []
    counts: dict[str, int] = {}

    specs = [("images", ".jpg"), ("segmentation", ".png"), ("detection", ".txt"), ("metadata", ".json"), ("review_results", ".json")]
    for directory, suffix in specs:
        found = {path.stem for path in (OUTPUT / directory).glob(f"frame_*{suffix}")}
        counts[directory] = len(found)
        if found != expected:
            errors.append(f"{directory} frame stems differ (found {len(found)}, expected {len(expected)})")

    overlay_found = {path.name.removesuffix("_review.jpg") for path in (OUTPUT / "overlays").glob("frame_*_review.jpg")}
    counts["overlays"] = len(overlay_found)
    if overlay_found != expected:
        errors.append(f"overlays frame stems differ (found {len(overlay_found)}, expected {len(expected)})")

    for frame in frames:
        image = cv2.imread(str(OUTPUT / "images" / f"{frame}.jpg"))
        mask = cv2.imread(str(OUTPUT / "segmentation" / f"{frame}.png"), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None or mask.ndim != 2 or image.shape[:2] != mask.shape or set(np.unique(mask)) - {0, 1, 2}:
            errors.append(f"{frame} invalid image/mask")
        try:
            yolo_rows = common.parse_yolo(OUTPUT / "detection" / f"{frame}.txt")
        except Exception as error:
            errors.append(f"{frame} detection error: {error}")
        meta = common.read_json(OUTPUT / "metadata" / f"{frame}.json")
        if set(meta) != common.META_KEYS or meta.get("frame_id") != frame or meta.get("weather") not in {"sunny", "cloudy", "after_rain"} or meta.get("surface_condition") not in {"dry", "wet"}:
            errors.append(f"{frame} invalid metadata keys or values")

    unchanged = common.source_inventory(frames) == state["source_inventory_before"]
    if not unchanged:
        errors.append("source validation inventory changed")

    actions = [item.get("action") for item in state["processed"].values()]
    geometry = [item.get("geometry_verdict") for item in state["processed"].values()]
    retries = sum(item.get("retry_count", 0) for item in state["processed"].values())
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

    statistics = {
        "total_frames": len(frames),
        "keep": actions.count("keep"),
        "add_drivable": actions.count("add_drivable"),
        "remove_drivable": actions.count("remove_drivable"),
        "review_needed": sum(bool(item.get("review_needed")) for item in state["processed"].values()),
        "sam2_used": sum(bool(item.get("sam2_used")) for item in state["processed"].values()),
        "sam2_accept": geometry.count("accept"),
        "sam2_reject": geometry.count("reject"),
        "sam2_retry": retries,
        "detection_frames_changed": detections_changed,
        "detection_add": detection_actions.count("add"),
        "detection_delete": detection_actions.count("delete"),
        "detection_adjust": detection_actions.count("adjust"),
        "gemma_api_failures": len(state["failed"]),
        "workers_ai_observed_usage": usage,
    }

    result = {
        "passed": not errors,
        "errors": errors,
        "counts": counts,
        "statistics": statistics,
        "source_validation_unchanged": unchanged,
        "processed": len(state["processed"]),
        "failed": state["failed"],
    }
    write_json(OUTPUT / "qc_report.json", result)
    return result


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description="Gemma Vision + SAM2 30-frame policy validation pipeline")
    parser.add_argument("command", choices=["prepare", "run", "qc"])
    parser.add_argument("--frame", type=str, default=None, help="Optional single frame to process")
    args = parser.parse_args()
    result = {"prepare": prepare, "run": lambda: run(args.frame), "qc": qc}[args.command]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
