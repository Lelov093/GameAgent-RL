from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

from gameagent_rl.envs.preprocessing import OneHotObservationEncoder


def test_one_hot_dimension_is_derived_from_runtime_vocabularies() -> None:
    with gym.make("MiniGrid-DoorKey-5x5-v0") as env:
        observation, _ = env.reset(seed=1001)
        encoder = OneHotObservationEncoder.from_space(env.observation_space)
        encoded = encoder.encode(observation)

    tile_count = int(np.prod(encoder.image_shape[:2]))
    expected = tile_count * (
        encoder.object_count + encoder.color_count + encoder.state_count
    ) + encoder.direction_count
    assert encoder.vector_dim == expected
    assert encoded.shape == (expected,)
    assert encoded.dtype == np.float32
    assert int(encoded.sum()) == tile_count * 3 + 1


def test_mission_is_not_part_of_policy_vector() -> None:
    with gym.make("MiniGrid-MemoryS11-v0") as env:
        observation, _ = env.reset(seed=23)
        encoder = OneHotObservationEncoder.from_space(env.observation_space)
        original = encoder.encode(observation)
        changed_mission = dict(observation, mission="privileged cue text")
        changed = encoder.encode(changed_mission)

    assert np.array_equal(original, changed)
