"""Track A DoorKey environment contract."""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import minigrid  # noqa: F401 - importing registers MiniGrid environments
import numpy as np
from gymnasium import spaces
from minigrid.core.actions import Actions
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX
from minigrid.core.world_object import Door, Goal, Key

from gameagent_rl.envs.preprocessing import OneHotObservationEncoder


TRACK_A_ENV_ID = "MiniGrid-DoorKey-5x5-v0"
TRACK_A_ACTIONS = (
    Actions.left,
    Actions.right,
    Actions.forward,
    Actions.pickup,
    Actions.toggle,
)


def _cell(image: np.ndarray, *, forward: bool) -> tuple[str, str, int]:
    x = image.shape[0] // 2
    y = image.shape[1] - (2 if forward else 1)
    object_idx, color_idx, state_idx = (int(value) for value in image[x, y])
    return IDX_TO_OBJECT[object_idx], IDX_TO_COLOR[color_idx], state_idx


def legality_mask(observation: Mapping[str, Any]) -> np.ndarray:
    """Derive executable DoorKey actions only from the current policy observation."""
    image = np.asarray(observation["image"])
    front_type, front_color, front_state = _cell(image, forward=True)
    carried_type, carried_color, _ = _cell(image, forward=False)

    forward_legal = front_type in {"empty", "floor", "goal", "lava"} or (
        front_type == "door" and front_state == STATE_TO_IDX["open"]
    )
    pickup_legal = carried_type == "empty" and front_type in {"key", "ball", "box"}
    toggle_legal = front_type == "door" and (
        front_state != STATE_TO_IDX["locked"]
        or (carried_type == "key" and carried_color == front_color)
    )

    return np.asarray(
        [True, True, forward_legal, pickup_legal, toggle_legal], dtype=np.bool_
    )


def sample_masked_action(mask: np.ndarray, rng: np.random.Generator) -> int:
    """Sample uniformly from valid policy actions for contract-level validation."""
    valid_actions = np.flatnonzero(mask)
    return int(rng.choice(valid_actions))


class DoorKeyContractEnv(gym.Wrapper):
    """DoorKey-5x5 with the frozen five-action and observation contracts."""

    def __init__(self, render_mode: str | None = None):
        super().__init__(gym.make(TRACK_A_ENV_ID, render_mode=render_mode))
        self.action_space = spaces.Discrete(len(TRACK_A_ACTIONS))
        self.encoder = OneHotObservationEncoder.from_space(self.env.observation_space)
        self.observation_space = self.encoder.observation_space
        self.last_raw_observation: Mapping[str, Any] | None = None
        self._door_pos: tuple[int, int] | None = None
        self._goal_pos: tuple[int, int] | None = None
        self._key_picked = False
        self._door_unlocked = False
        self._door_passed = False
        self._goal_reached = False

    @property
    def native_env(self):
        return self.env.unwrapped

    @staticmethod
    def policy_action_to_native(action: int) -> Actions:
        if action < 0 or action >= len(TRACK_A_ACTIONS):
            raise ValueError(f"Unknown Track A policy action: {action}")
        return TRACK_A_ACTIONS[action]

    def action_masks(self) -> np.ndarray:
        if self.last_raw_observation is None:
            raise RuntimeError("reset() must be called before requesting an action mask")
        return legality_mask(self.last_raw_observation)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.last_raw_observation = observation
        self._door_pos = self._find_position(Door)
        self._goal_pos = self._find_position(Goal)
        self._key_picked = False
        self._door_unlocked = False
        self._door_passed = False
        self._goal_reached = False
        info = dict(info)
        info.update(self._contract_info(selected_action_legal=None))
        return self.encoder.encode(observation), info

    def step(self, action: int):
        current_mask = self.action_masks()
        native_action = self.policy_action_to_native(int(action))
        observation, reward, terminated, truncated, info = self.env.step(native_action)
        self.last_raw_observation = observation
        self._update_diagnostics()
        info = dict(info)
        info.update(self._contract_info(selected_action_legal=bool(current_mask[action])))
        return self.encoder.encode(observation), reward, terminated, truncated, info

    def _find_position(self, object_type: type) -> tuple[int, int]:
        for x in range(self.native_env.width):
            for y in range(self.native_env.height):
                if isinstance(self.native_env.grid.get(x, y), object_type):
                    return (x, y)
        raise RuntimeError(f"Expected {object_type.__name__} in DoorKey grid")

    def _update_diagnostics(self) -> None:
        door = self.native_env.grid.get(*self._door_pos)
        self._key_picked |= isinstance(self.native_env.carrying, Key)
        self._door_unlocked |= isinstance(door, Door) and not door.is_locked
        self._door_passed |= int(self.native_env.agent_pos[0]) > self._door_pos[0]
        self._goal_reached |= tuple(self.native_env.agent_pos) == self._goal_pos

    def _contract_info(self, selected_action_legal: bool | None) -> dict[str, Any]:
        return {
            "action_mask": self.action_masks().copy(),
            "selected_action_legal": selected_action_legal,
            "key_picked": self._key_picked,
            "door_unlocked": self._door_unlocked,
            "door_passed": self._door_passed,
            "goal_reached": self._goal_reached,
            "success": self._goal_reached,
        }
