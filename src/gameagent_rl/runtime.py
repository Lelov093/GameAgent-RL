"""Minimal runtime utilities shared by the project bootstrap."""

from __future__ import annotations

import json
import platform
import random
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime.toml"


def load_runtime_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the small WB1 runtime configuration."""
    with path.open("rb") as config_file:
        return tomllib.load(config_file)["runtime"]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime_info() -> dict[str, Any]:
    """Return concise runtime and device information."""
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "numpy": np.__version__,
        "gymnasium": version("gymnasium"),
        "minigrid": version("minigrid"),
        "stable_baselines3": version("stable-baselines3"),
        "sb3_contrib": version("sb3-contrib"),
    }


def write_artifact(relative_path: str, payload: dict[str, Any]) -> Path:
    """Write a JSON artifact below the configured project artifact directory."""
    config = load_runtime_config()
    output_path = PROJECT_ROOT / config["artifact_dir"] / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
