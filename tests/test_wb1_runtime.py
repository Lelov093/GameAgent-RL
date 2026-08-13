"""The six required Phase 1 / WB1 runtime checks."""

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import torch

from gameagent_rl.runtime import load_runtime_config, runtime_info, seed_everything, write_artifact


def test_import_stack() -> None:
    import minigrid  # noqa: F401
    import sb3_contrib  # noqa: F401
    import stable_baselines3  # noqa: F401


def test_pytorch_runtime() -> None:
    seed_everything(1001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value = (torch.tensor([1.0, 2.0], device=device) * 2).cpu()
    assert torch.equal(value, torch.tensor([2.0, 4.0]))


def test_minigrid_reset_and_step() -> None:
    config = load_runtime_config()
    with gym.make(config["environment_id"]) as env:
        observation, _ = env.reset(seed=config["seed"])
        next_observation, reward, terminated, truncated, _ = env.step(0)
    assert observation["image"].shape == (7, 7, 3)
    assert next_observation["image"].shape == (7, 7, 3)
    assert isinstance(reward, (int, float, np.number))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_environment_seed_reproducibility() -> None:
    config = load_runtime_config()
    with gym.make(config["environment_id"]) as first_env:
        first, _ = first_env.reset(seed=config["seed"])
    with gym.make(config["environment_id"]) as second_env:
        second, _ = second_env.reset(seed=config["seed"])
    assert np.array_equal(first["image"], second["image"])
    assert first["direction"] == second["direction"]
    assert first["mission"] == second["mission"]


def test_rgb_array_render() -> None:
    config = load_runtime_config()
    with gym.make(config["environment_id"], render_mode="rgb_array") as env:
        env.reset(seed=config["seed"])
        frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.size > 0


def test_project_artifact_write() -> None:
    info = runtime_info()
    output_path = write_artifact("wb1/runtime_smoke.json", info)
    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8")) == info
