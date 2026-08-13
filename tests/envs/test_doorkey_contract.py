from __future__ import annotations

import numpy as np
from minigrid.core.actions import Actions
from minigrid.core.world_object import Door, Goal, Key

from gameagent_rl.envs.doorkey import (
    TRACK_A_ACTIONS,
    DoorKeyContractEnv,
    legality_mask,
    sample_masked_action,
)


def _find_object(env: DoorKeyContractEnv, object_type: type) -> tuple[int, int]:
    for x in range(env.native_env.width):
        for y in range(env.native_env.height):
            if isinstance(env.native_env.grid.get(x, y), object_type):
                return (x, y)
    raise AssertionError(f"Missing {object_type.__name__}")


def _face_target(env: DoorKeyContractEnv, target: tuple[int, int]) -> None:
    directions = (((-1, 0), 0), ((0, -1), 1), ((1, 0), 2), ((0, 1), 3))
    for (dx, dy), direction in directions:
        candidate = (target[0] + dx, target[1] + dy)
        if env.native_env.grid.get(*candidate) is None:
            env.native_env.agent_pos = np.asarray(candidate)
            env.native_env.agent_dir = direction
            env.last_raw_observation = env.native_env.gen_obs()
            return
    raise AssertionError(f"No empty adjacent cell for target {target}")


def test_reduced_action_mapping_excludes_drop_and_done() -> None:
    assert TRACK_A_ACTIONS == (
        Actions.left,
        Actions.right,
        Actions.forward,
        Actions.pickup,
        Actions.toggle,
    )
    assert Actions.drop not in TRACK_A_ACTIONS
    assert Actions.done not in TRACK_A_ACTIONS


def test_legality_oracle_matches_typical_doorkey_states() -> None:
    with DoorKeyContractEnv() as env:
        env.reset(seed=1001)

        env.native_env.agent_pos = np.asarray((1, 1))
        env.native_env.agent_dir = 3
        env.native_env.carrying = None
        wall_observation = env.native_env.gen_obs()
        assert legality_mask(wall_observation).tolist() == [True, True, False, False, False]

        key_pos = _find_object(env, Key)
        _face_target(env, key_pos)
        env.native_env.carrying = None
        key_mask = legality_mask(env.native_env.gen_obs())
        assert key_mask.tolist() == [True, True, False, True, False]

        door_pos = _find_object(env, Door)
        _face_target(env, door_pos)
        env.native_env.carrying = None
        assert legality_mask(env.native_env.gen_obs()).tolist() == [
            True,
            True,
            False,
            False,
            False,
        ]

        env.native_env.carrying = Key("yellow")
        assert legality_mask(env.native_env.gen_obs()).tolist() == [
            True,
            True,
            False,
            False,
            True,
        ]

        door = env.native_env.grid.get(*door_pos)
        door.is_locked = False
        door.is_open = True
        assert legality_mask(env.native_env.gen_obs()).tolist() == [
            True,
            True,
            True,
            False,
            True,
        ]


def test_mask_always_keeps_turns_and_sampling_never_selects_illegal_action() -> None:
    rng = np.random.default_rng(1001)
    masks = (
        np.asarray([True, True, False, False, False]),
        np.asarray([True, True, True, False, True]),
    )
    for mask in masks:
        assert mask[0] and mask[1]
        for _ in range(200):
            action = sample_masked_action(mask, rng)
            assert mask[action]


def test_native_success_reward_and_goal_signal_are_preserved() -> None:
    with DoorKeyContractEnv() as env:
        env.reset(seed=1001)
        goal_pos = _find_object(env, Goal)
        _face_target(env, goal_pos)
        _, reward, terminated, truncated, info = env.step(2)

        expected_reward = 1 - 0.9 * env.native_env.step_count / env.native_env.max_steps
        assert reward == expected_reward
        assert terminated is True
        assert truncated is False
        assert info["goal_reached"] is True
        assert info["success"] is True


def test_native_timeout_truncation_is_preserved() -> None:
    with DoorKeyContractEnv() as env:
        env.reset(seed=1001)
        env.native_env.step_count = env.native_env.max_steps - 1
        _, reward, terminated, truncated, info = env.step(0)

        assert reward == 0
        assert terminated is False
        assert truncated is True
        assert info["success"] is False


def test_diagnostic_milestones_are_read_from_environment_state() -> None:
    with DoorKeyContractEnv() as env:
        env.reset(seed=1001)
        door_pos = _find_object(env, Door)
        door = env.native_env.grid.get(*door_pos)
        door.is_locked = False
        door.is_open = True
        env.native_env.carrying = Key("yellow")
        env.native_env.agent_pos = np.asarray((door_pos[0] + 1, door_pos[1]))
        env.last_raw_observation = env.native_env.gen_obs()
        _, _, _, _, info = env.step(0)

        assert info["key_picked"] is True
        assert info["door_unlocked"] is True
        assert info["door_passed"] is True


def test_doorkey_seed_reproducibility_includes_vector_and_mask() -> None:
    with DoorKeyContractEnv() as first_env, DoorKeyContractEnv() as second_env:
        first_observation, first_info = first_env.reset(seed=1001)
        second_observation, second_info = second_env.reset(seed=1001)

    assert np.array_equal(first_observation, second_observation)
    assert np.array_equal(first_info["action_mask"], second_info["action_mask"])
