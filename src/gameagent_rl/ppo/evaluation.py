"""Development evaluation for the Own PPO policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from gameagent_rl.envs.doorkey import DoorKeyContractEnv
from gameagent_rl.ppo.model import ActorCritic


@dataclass(frozen=True)
class PPOEvaluationResult:
    episodes: int
    success_rate: float
    mean_return: float
    mean_length: float
    invalid_action_rate: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_doorkey(
    model: ActorCritic,
    seeds: Sequence[int],
    *,
    masked: bool = False,
) -> PPOEvaluationResult:
    successes: list[float] = []
    returns: list[float] = []
    lengths: list[int] = []
    invalid_actions = 0
    total_actions = 0
    for seed in seeds:
        with DoorKeyContractEnv() as env:
            observation, _ = env.reset(seed=int(seed))
            terminated = truncated = False
            episode_return = 0.0
            length = 0
            while not (terminated or truncated):
                action = model.predict(
                    observation,
                    deterministic=True,
                    action_masks=env.action_masks() if masked else None,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                length += 1
                total_actions += 1
                invalid_actions += int(info["selected_action_legal"] is False)
            successes.append(float(info["success"]))
            returns.append(episode_return)
            lengths.append(length)
    return PPOEvaluationResult(
        episodes=len(seeds),
        success_rate=float(np.mean(successes)),
        mean_return=float(np.mean(returns)),
        mean_length=float(np.mean(lengths)),
        invalid_action_rate=invalid_actions / total_actions,
    )
