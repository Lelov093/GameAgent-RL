from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from minigrid.core.actions import Actions
from minigrid.core.world_object import Ball, Key

from gameagent_rl.envs.memory import (
    TRACK_B_ACTIONS,
    TRACK_B_START_DIRECTION,
    TRACK_B_START_POSITION,
    MemoryContractEnv,
)


def _set_decision_pose(env: MemoryContractEnv, target: tuple[int, int]) -> None:
    env.native_env.agent_pos = np.asarray(env.decision_position)
    env.native_env.agent_dir = 3 if target[1] < env.decision_position[1] else 1
    env.last_raw_observation = env.native_env.gen_obs()


def test_reduced_memory_action_mapping_is_navigation_only() -> None:
    assert TRACK_B_ACTIONS == (Actions.left, Actions.right, Actions.forward)


def test_cue_visibility_leave_event_and_reset_trigger() -> None:
    with MemoryContractEnv() as env:
        _, reset_info = env.reset(seed=1001)
        assert tuple(env.native_env.agent_pos) == TRACK_B_START_POSITION
        assert env.native_env.agent_dir == TRACK_B_START_DIRECTION
        assert reset_info["cue_visible"] is True

        _, _, _, _, step_info = env.step(2)
        assert step_info["cue_visible"] is False
        assert step_info["cue_left_view"] is True
        assert step_info["reset_trigger"] is True

        _, _, _, _, later_info = env.step(2)
        assert later_info["reset_trigger"] is False


def test_reset_trigger_instrumentation_does_not_change_environment_transition() -> None:
    with MemoryContractEnv() as instrumented, gym.make("MiniGrid-MemoryS11-v0") as native:
        instrumented.reset(seed=23)
        native.reset(seed=23)
        native.unwrapped.agent_pos = np.asarray(TRACK_B_START_POSITION)
        native.unwrapped.agent_dir = TRACK_B_START_DIRECTION
        instrumented.step(2)
        native.step(Actions.forward)

        assert tuple(instrumented.native_env.agent_pos) == tuple(native.unwrapped.agent_pos)
        assert instrumented.native_env.agent_dir == native.unwrapped.agent_dir
        assert np.array_equal(
            instrumented.native_env.grid.encode(), native.unwrapped.grid.encode()
        )


def test_final_decision_observation_does_not_contain_cue_identity() -> None:
    with MemoryContractEnv() as env:
        env.reset(seed=23)
        env.native_env.agent_pos = np.asarray(env.decision_position)
        env.native_env.agent_dir = 3
        assert env.native_env.agent_sees(*env.cue_position) is False
        before = env.native_env.gen_obs()["image"].copy()

        cue = env.native_env.grid.get(*env.cue_position)
        replacement = Ball("green") if isinstance(cue, Key) else Key("green")
        env.native_env.grid.set(*env.cue_position, replacement)
        after = env.native_env.gen_obs()["image"].copy()

        assert np.array_equal(before, after)


def test_correct_and_wrong_terminal_choices_are_distinguished() -> None:
    with MemoryContractEnv() as env:
        env.reset(seed=23)
        _set_decision_pose(env, tuple(env.native_env.success_pos))
        _, reward, terminated, truncated, info = env.step(2)
        assert reward > 0
        assert terminated is True and truncated is False
        assert info["decision_reached"] is True
        assert info["correct_choice"] is True
        assert info["wrong_choice"] is False

        env.reset(seed=23)
        _set_decision_pose(env, tuple(env.native_env.failure_pos))
        _, reward, terminated, truncated, info = env.step(2)
        assert reward == 0
        assert terminated is True and truncated is False
        assert info["correct_choice"] is False
        assert info["wrong_choice"] is True


def test_memory_seed_reproducibility_includes_contract_signals() -> None:
    with MemoryContractEnv() as first_env, MemoryContractEnv() as second_env:
        first_observation, first_info = first_env.reset(seed=2001)
        second_observation, second_info = second_env.reset(seed=2001)

    assert np.array_equal(first_observation, second_observation)
    assert first_info == second_info


def test_all_frozen_seed_panels_start_with_visible_cue() -> None:
    with MemoryContractEnv() as env:
        for seed in (*range(1001, 1006), *range(2001, 2021), *range(3001, 3101)):
            _, info = env.reset(seed=seed)
            assert tuple(env.native_env.agent_pos) == TRACK_B_START_POSITION
            assert env.native_env.agent_dir == TRACK_B_START_DIRECTION
            assert info["cue_visible"] is True
