from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TinyChoiceEnv(gym.Env):
    """One-step fixture whose optimal action is unambiguous."""

    observation_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
    action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        success = int(action) == 1
        return (
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            float(success),
            True,
            False,
            {"success": success},
        )
