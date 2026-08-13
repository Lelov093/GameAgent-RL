"""Track B Memory environment contract and evaluator-side signals."""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import minigrid  # noqa: F401 - importing registers MiniGrid environments
import numpy as np
from gymnasium import spaces
from minigrid.core.actions import Actions
from minigrid.core.world_object import Ball, Key

from gameagent_rl.envs.preprocessing import OneHotObservationEncoder


TRACK_B_ENV_ID = "MiniGrid-MemoryS11-v0"
TRACK_B_ACTIONS = (Actions.left, Actions.right, Actions.forward)
TRACK_B_START_POSITION = (1, 5)
TRACK_B_START_DIRECTION = 0


class MemoryContractEnv(gym.Wrapper):
    """MemoryS11 with three navigation actions and memory instrumentation."""

    def __init__(self, render_mode: str | None = None):
        super().__init__(gym.make(TRACK_B_ENV_ID, render_mode=render_mode))
        self.action_space = spaces.Discrete(len(TRACK_B_ACTIONS))
        self.encoder = OneHotObservationEncoder.from_space(self.env.observation_space)
        self.observation_space = self.encoder.observation_space
        self.last_raw_observation: Mapping[str, Any] | None = None
        self._cue_pos: tuple[int, int] | None = None
        self._decision_pos: tuple[int, int] | None = None
        self._cue_seen = False
        self._previous_cue_visible = False
        self._reset_trigger_emitted = False
        self._decision_reached = False

    @property
    def native_env(self):
        return self.env.unwrapped

    @property
    def cue_position(self) -> tuple[int, int]:
        if self._cue_pos is None:
            raise RuntimeError("reset() must be called before reading cue position")
        return self._cue_pos

    @property
    def decision_position(self) -> tuple[int, int]:
        if self._decision_pos is None:
            raise RuntimeError("reset() must be called before reading decision position")
        return self._decision_pos

    @staticmethod
    def policy_action_to_native(action: int) -> Actions:
        if action < 0 or action >= len(TRACK_B_ACTIONS):
            raise ValueError(f"Unknown Track B policy action: {action}")
        return TRACK_B_ACTIONS[action]

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._cue_pos = self._find_cue_position()
        self.native_env.agent_pos = np.asarray(TRACK_B_START_POSITION)
        self.native_env.agent_dir = TRACK_B_START_DIRECTION
        observation = self.native_env.gen_obs()
        self.last_raw_observation = observation
        success_pos = tuple(self.native_env.success_pos)
        failure_pos = tuple(self.native_env.failure_pos)
        self._decision_pos = (
            int(success_pos[0]),
            (int(success_pos[1]) + int(failure_pos[1])) // 2,
        )
        cue_visible = self._cue_visible()
        self._cue_seen = cue_visible
        self._previous_cue_visible = cue_visible
        self._reset_trigger_emitted = False
        self._decision_reached = False
        info = dict(info)
        info.update(self._contract_info(cue_visible=cue_visible, reset_trigger=False))
        return self.encoder.encode(observation), info

    def step(self, action: int):
        native_action = self.policy_action_to_native(int(action))
        observation, reward, terminated, truncated, info = self.env.step(native_action)
        self.last_raw_observation = observation

        cue_visible = self._cue_visible()
        cue_left_view = self._previous_cue_visible and not cue_visible
        self._cue_seen |= cue_visible
        reset_trigger = (
            self._cue_seen and cue_left_view and not self._reset_trigger_emitted
        )
        self._reset_trigger_emitted |= reset_trigger
        self._previous_cue_visible = cue_visible

        choice = self._choice()
        self._decision_reached |= (
            tuple(self.native_env.agent_pos) == self.decision_position or choice is not None
        )
        info = dict(info)
        info.update(
            self._contract_info(
                cue_visible=cue_visible,
                reset_trigger=reset_trigger,
                choice=choice,
            )
        )
        return self.encoder.encode(observation), reward, terminated, truncated, info

    def _find_cue_position(self) -> tuple[int, int]:
        positions: list[tuple[int, int]] = []
        for x in range(self.native_env.width):
            for y in range(self.native_env.height):
                if isinstance(self.native_env.grid.get(x, y), (Key, Ball)):
                    positions.append((x, y))
        if len(positions) != 3:
            raise RuntimeError(f"Expected three Memory objects, found {len(positions)}")
        return min(positions, key=lambda position: position[0])

    def _cue_visible(self) -> bool:
        return bool(self.native_env.agent_sees(*self.cue_position))

    def _choice(self) -> str | None:
        agent_pos = tuple(self.native_env.agent_pos)
        if agent_pos == tuple(self.native_env.success_pos):
            return "correct"
        if agent_pos == tuple(self.native_env.failure_pos):
            return "wrong"
        return None

    def _contract_info(
        self,
        *,
        cue_visible: bool,
        reset_trigger: bool,
        choice: str | None = None,
    ) -> dict[str, Any]:
        return {
            "cue_visible": cue_visible,
            "cue_seen": self._cue_seen,
            "cue_left_view": self._cue_seen and not cue_visible,
            "reset_trigger": reset_trigger,
            "decision_reached": self._decision_reached,
            "correct_choice": choice == "correct",
            "wrong_choice": choice == "wrong",
        }
