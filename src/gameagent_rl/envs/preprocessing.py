"""Deterministic MiniGrid policy-observation preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from gymnasium import spaces
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX


def _vocabulary_size(vocabulary: Mapping[str, int]) -> int:
    return max(vocabulary.values()) + 1


@dataclass(frozen=True)
class OneHotObservationEncoder:
    """Encode MiniGrid image categories and direction as a flat vector."""

    image_shape: tuple[int, int, int]
    direction_count: int
    object_count: int = _vocabulary_size(OBJECT_TO_IDX)
    color_count: int = _vocabulary_size(COLOR_TO_IDX)
    state_count: int = _vocabulary_size(STATE_TO_IDX)

    @classmethod
    def from_space(cls, observation_space: spaces.Dict) -> OneHotObservationEncoder:
        image_space = observation_space["image"]
        direction_space = observation_space["direction"]
        return cls(
            image_shape=tuple(image_space.shape),
            direction_count=int(direction_space.n),
        )

    @property
    def vector_dim(self) -> int:
        tile_count = int(np.prod(self.image_shape[:2]))
        tile_width = self.object_count + self.color_count + self.state_count
        return tile_count * tile_width + self.direction_count

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(self.vector_dim,), dtype=np.float32)

    def encode(self, observation: Mapping[str, Any]) -> np.ndarray:
        image = np.asarray(observation["image"])
        object_one_hot = np.eye(self.object_count, dtype=np.float32)[image[..., 0]]
        color_one_hot = np.eye(self.color_count, dtype=np.float32)[image[..., 1]]
        state_one_hot = np.eye(self.state_count, dtype=np.float32)[image[..., 2]]
        image_vector = np.concatenate(
            (object_one_hot, color_one_hot, state_one_hot), axis=-1
        ).reshape(-1)

        direction = int(observation["direction"])
        direction_vector = np.eye(self.direction_count, dtype=np.float32)[direction]
        return np.concatenate((image_vector, direction_vector)).astype(np.float32, copy=False)
