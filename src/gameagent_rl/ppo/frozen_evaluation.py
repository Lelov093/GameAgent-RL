"""Read-only Track A Frozen Evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from gameagent_rl.envs.doorkey import DoorKeyContractEnv, sample_masked_action
from gameagent_rl.ppo.model import ActorCritic


Policy = Callable[[np.ndarray, DoorKeyContractEnv], int]


def failure_type(info: dict[str, Any], terminated: bool, truncated: bool) -> str:
    if info["success"]:
        return "success"
    if not info["key_picked"]:
        return "timeout_before_key" if truncated else "terminated_before_key"
    if not info["door_unlocked"]:
        return "key_picked_door_locked"
    if not info["door_passed"]:
        return "door_unlocked_not_passed"
    return "door_passed_goal_not_reached"


def evaluate_episode(
    policy: Policy,
    evaluation_seed: int,
    *,
    algorithm_id: str,
    training_seed: int | None,
    policy_action_seed: int | None = None,
    capture_frames: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    render_mode = "rgb_array" if capture_frames else None
    with DoorKeyContractEnv(render_mode=render_mode) as env:
        observation, info = env.reset(seed=evaluation_seed)
        frames = [env.render()] if capture_frames else []
        actions: list[int] = []
        legal_actions: list[bool] = []
        episode_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy(observation, env)
            observation, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            legal_actions.append(bool(info["selected_action_legal"]))
            episode_return += float(reward)
            if capture_frames:
                frames.append(env.render())
    record = {
        "algorithm_id": algorithm_id,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "policy_action_seed": policy_action_seed,
        "success": bool(info["success"]),
        "return": episode_return,
        "episode_length": len(actions),
        "termination_type": "terminated" if terminated else "truncated",
        "invalid_action_count": legal_actions.count(False),
        "invalid_action_rate": legal_actions.count(False) / len(actions),
        "key_picked": bool(info["key_picked"]),
        "door_unlocked": bool(info["door_unlocked"]),
        "door_passed": bool(info["door_passed"]),
        "goal_reached": bool(info["goal_reached"]),
        "failure_type": failure_type(info, terminated, truncated),
    }
    replay = None
    if capture_frames:
        replay = {
            "frames": np.stack(frames),
            "actions": np.asarray(actions, dtype=np.int64),
            "legal_actions": np.asarray(legal_actions, dtype=np.bool_),
        }
    return record, replay


def learned_policy(model: ActorCritic, masked: bool) -> Policy:
    def policy(observation: np.ndarray, env: DoorKeyContractEnv) -> int:
        return model.predict(
            observation,
            deterministic=True,
            action_masks=env.action_masks() if masked else None,
        )

    return policy


def random_policy(
    evaluation_seed: int, policy_action_seed: int, masked: bool
) -> Policy:
    rng = np.random.default_rng(
        np.random.SeedSequence([evaluation_seed, policy_action_seed])
    )

    def policy(observation: np.ndarray, env: DoorKeyContractEnv) -> int:
        del observation
        if masked:
            return sample_masked_action(env.action_masks(), rng)
        return int(rng.integers(env.action_space.n))

    return policy
