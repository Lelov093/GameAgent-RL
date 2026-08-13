"""Rollout storage and termination-aware GAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE while bootstrapping truncations but stopping episode recursion."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    next_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        bootstrap_mask = 1.0 - terminated[step].astype(np.float32)
        episode_continues = 1.0 - np.logical_or(
            terminated[step], truncated[step]
        ).astype(np.float32)
        delta = (
            rewards[step]
            + gamma * bootstrap_mask * next_values[step]
            - values[step]
        )
        advantages[step] = (
            delta + gamma * gae_lambda * episode_continues * next_advantage
        )
        next_advantage = advantages[step]
    return advantages, advantages + values


@dataclass(frozen=True)
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_masks: torch.Tensor | None


class RolloutBuffer:
    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        observation_dim: int,
        action_dim: int | None = None,
    ):
        shape = (n_steps, n_envs)
        self.observations = np.zeros((*shape, observation_dim), dtype=np.float32)
        self.actions = np.zeros(shape, dtype=np.int64)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.next_values = np.zeros(shape, dtype=np.float32)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.terminated = np.zeros(shape, dtype=np.bool_)
        self.truncated = np.zeros(shape, dtype=np.bool_)
        self.episode_starts = np.zeros(shape, dtype=np.bool_)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.action_masks = (
            None
            if action_dim is None
            else np.zeros((*shape, action_dim), dtype=np.bool_)
        )
        self.position = 0
        self.n_steps = n_steps
        self.n_envs = n_envs

    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        log_probs: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        episode_starts: np.ndarray,
        action_masks: np.ndarray | None = None,
    ) -> None:
        index = self.position
        self.observations[index] = observations
        self.actions[index] = actions
        self.rewards[index] = rewards
        self.values[index] = values
        self.next_values[index] = next_values
        self.log_probs[index] = log_probs
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.episode_starts[index] = episode_starts
        if self.action_masks is not None:
            if action_masks is None:
                raise ValueError("masked rollout requires action_masks")
            self.action_masks[index] = action_masks
        elif action_masks is not None:
            raise ValueError("vanilla rollout does not store action_masks")
        self.position += 1

    def finish(self, gamma: float, gae_lambda: float) -> None:
        if self.position != self.n_steps:
            raise RuntimeError("Cannot finish an incomplete rollout")
        self.advantages, self.returns = compute_gae(
            self.rewards,
            self.values,
            self.next_values,
            self.terminated,
            self.truncated,
            gamma,
            gae_lambda,
        )

    def batches(
        self, batch_size: int, device: torch.device, rng: np.random.Generator
    ) -> Iterator[RolloutBatch]:
        total = self.n_steps * self.n_envs
        indices = rng.permutation(total)
        observations = self.observations.reshape(total, -1)
        actions = self.actions.reshape(total)
        log_probs = self.log_probs.reshape(total)
        values = self.values.reshape(total)
        advantages = self.advantages.reshape(total)
        returns = self.returns.reshape(total)
        action_masks = (
            None
            if self.action_masks is None
            else self.action_masks.reshape(total, -1)
        )
        for start in range(0, total, batch_size):
            batch = indices[start : start + batch_size]
            yield RolloutBatch(
                observations=torch.as_tensor(observations[batch], device=device),
                actions=torch.as_tensor(actions[batch], device=device),
                old_log_probs=torch.as_tensor(log_probs[batch], device=device),
                old_values=torch.as_tensor(values[batch], device=device),
                advantages=torch.as_tensor(advantages[batch], device=device),
                returns=torch.as_tensor(returns[batch], device=device),
                action_masks=(
                    None
                    if action_masks is None
                    else torch.as_tensor(action_masks[batch], device=device)
                ),
            )
