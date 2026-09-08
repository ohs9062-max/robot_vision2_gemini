"""Small shared helpers for the vision training and inference commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping without importing PyYAML until it is needed."""
    config_path = require_file(path, "config file")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install requirements.txt first.") from exc

    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    return value


def require_file(path: str | Path, label: str = "file") -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {candidate}")
    return candidate


def require_directory(path: str | Path, label: str = "directory") -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {candidate}")
    return candidate


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def select_device(requested: str = "auto") -> str:
    """Return a validated Torch device string only when training/inference starts."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Install requirements.txt first.") from exc

    requested = requested.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return requested


def setup_logging(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("robot_vision")
