import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "tools" / "gemma10_pipeline.py"
SPEC = importlib.util.spec_from_file_location("gemma10_pipeline", MODULE_PATH)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def test_apply_detection_delete_adjust_add():
    baseline = [[2, .2, .3, .1, .2], [3, .7, .6, .2, .1]]
    changes = [
        {"action": "delete", "class": "puddle", "target_index": 0, "box": None,
         "reason": "false positive", "confidence": .99},
        {"action": "adjust", "class": "obstacle", "target_index": 1,
         "box": [.5, .5, .9, .8], "reason": "tighten", "confidence": .9},
        {"action": "add", "class": "step", "target_index": None,
         "box": [.1, .2, .3, .4], "reason": "missing", "confidence": .95},
    ]
    result = pipeline.apply_detections(baseline, changes)
    assert result == [[3, .7, .65, .4, .30000000000000004], [0, .2, .30000000000000004, .19999999999999998, .2]]


def test_validate_rejects_unordered_box():
    try:
        pipeline.validate_box([.8, .1, .2, .9], "box")
    except RuntimeError:
        return
    raise AssertionError("unordered box accepted")


def test_overlay_keeps_dimensions():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=np.uint8)
    result = pipeline.overlay(image, mask, [], "frame_000001")
    assert result.shape == image.shape


def test_remove_drivable_maps_to_non_drivable():
    assert pipeline.SEG_ACTION_TO_ID["remove_drivable"] == 2


def test_uncertain_vegetation_requires_review_needed():
    value = {
        "frame_id": "frame_000001", "summary": "uncertain edge", "surface_condition": "dry",
        "vegetation_reviews": [{
            "vegetation_type": "uncertain_vegetation", "description": "occluded plants",
            "prompt_box": [0.1, 0.2, 0.3, 0.4], "confidence": 0.5, "review_needed": False,
        }],
        "segmentation_changes": [], "detection_changes": [],
    }
    try:
        pipeline.validate_judgment(value, "frame_000001", 0)
    except RuntimeError:
        return
    raise AssertionError("uncertain vegetation without review_needed was accepted")
