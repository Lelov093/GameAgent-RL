"""Own PPO rollout collection and optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from gameagent_rl.ppo.buffer import RolloutBuffer
from gameagent_rl.ppo.model import ActorCritic
from gameagent_rl.runtime import seed_everything


EnvFactory = Callable[[], gym.Env]


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    n_envs: int = 8
    n_steps: int = 128
    batch_size: int = 256
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    hidden_size: int = 128
    seed: int = 1001
    device: str = "cpu"


def clipped_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    return -torch.minimum(unclipped, clipped).mean(), ratio, clipped


class PPOTrainer:
    def __init__(self, env_factory: EnvFactory, config: PPOConfig):
        self.env_factory = env_factory
        self.config = config
        self.device = torch.device(config.device)
        seed_everything(config.seed)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self.rng = np.random.default_rng(config.seed)
        self.envs = [env_factory() for _ in range(config.n_envs)]
        self.observations = np.stack(
            [env.reset(seed=config.seed + rank)[0] for rank, env in enumerate(self.envs)]
        ).astype(np.float32)
        observation_dim = int(np.prod(self.envs[0].observation_space.shape))
        action_dim = int(self.envs[0].action_space.n)
        self.model = ActorCritic(
            observation_dim, action_dim, config.hidden_size
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.total_steps = 0
        self.episode_starts = np.ones(config.n_envs, dtype=np.bool_)
        self.running_returns = np.zeros(config.n_envs, dtype=np.float32)
        self.running_lengths = np.zeros(config.n_envs, dtype=np.int32)

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def current_action_masks(self) -> np.ndarray | None:
        return None

    def collect_rollout(
        self, n_steps: int | None = None
    ) -> tuple[RolloutBuffer, dict[str, float]]:
        rollout_steps = n_steps or self.config.n_steps
        buffer = RolloutBuffer(
            rollout_steps,
            self.config.n_envs,
            self.model.observation_dim,
            self.model.action_dim if self.current_action_masks() is not None else None,
        )
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        completed_successes: list[float] = []
        invalid_actions = 0
        total_actions = 0

        for _ in range(rollout_steps):
            action_masks = self.current_action_masks()
            observation_tensor = torch.as_tensor(
                self.observations, device=self.device
            )
            mask_tensor = (
                None
                if action_masks is None
                else torch.as_tensor(action_masks, device=self.device)
            )
            with torch.no_grad():
                actions, log_probs, values = self.model.act(
                    observation_tensor, mask_tensor
                )
            action_array = actions.cpu().numpy()
            rewards = np.zeros(self.config.n_envs, dtype=np.float32)
            terminated = np.zeros(self.config.n_envs, dtype=np.bool_)
            truncated = np.zeros(self.config.n_envs, dtype=np.bool_)
            transition_observations = np.zeros_like(self.observations)
            next_observations = np.zeros_like(self.observations)

            for rank, env in enumerate(self.envs):
                next_observation, reward, term, trunc, info = env.step(
                    int(action_array[rank])
                )
                transition_observations[rank] = next_observation
                rewards[rank] = float(reward)
                terminated[rank] = term
                truncated[rank] = trunc
                self.running_returns[rank] += float(reward)
                self.running_lengths[rank] += 1
                total_actions += 1
                invalid_actions += int(info.get("selected_action_legal") is False)
                if term or trunc:
                    completed_returns.append(float(self.running_returns[rank]))
                    completed_lengths.append(int(self.running_lengths[rank]))
                    completed_successes.append(float(info.get("success", False)))
                    self.running_returns[rank] = 0.0
                    self.running_lengths[rank] = 0
                    next_observation, _ = env.reset()
                next_observations[rank] = next_observation

            with torch.no_grad():
                _, transition_values = self.model.distribution_and_value(
                    torch.as_tensor(transition_observations, device=self.device)
                )
            buffer.add(
                self.observations,
                action_array,
                rewards,
                values.cpu().numpy(),
                transition_values.cpu().numpy(),
                log_probs.cpu().numpy(),
                terminated,
                truncated,
                self.episode_starts,
                action_masks,
            )
            self.episode_starts = np.logical_or(terminated, truncated)
            self.observations = next_observations
            self.total_steps += self.config.n_envs

        buffer.finish(self.config.gamma, self.config.gae_lambda)
        return buffer, {
            "rollout_episode_return": float(np.mean(completed_returns))
            if completed_returns
            else 0.0,
            "rollout_success_rate": float(np.mean(completed_successes))
            if completed_successes
            else 0.0,
            "rollout_episode_length": float(np.mean(completed_lengths))
            if completed_lengths
            else 0.0,
            "rollout_invalid_action_rate": invalid_actions / total_actions,
        }

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        advantages = buffer.advantages
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        buffer.advantages = normalized.astype(np.float32)
        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "grad_norm": [],
        }

        for _ in range(self.config.update_epochs):
            for batch in buffer.batches(
                self.config.batch_size, self.device, self.rng
            ):
                new_log_probs, entropy, values = self.model.evaluate_actions(
                    batch.observations, batch.actions, batch.action_masks
                )
                policy_loss, ratio, _ = clipped_policy_loss(
                    new_log_probs,
                    batch.old_log_probs,
                    batch.advantages,
                    self.config.clip_epsilon,
                )
                value_loss = 0.5 * (values - batch.returns).pow(2).mean()
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy_mean
                )
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_probs - batch.old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1) > self.config.clip_epsilon)
                        .float()
                        .mean()
                    )
                for key, value in (
                    ("policy_loss", policy_loss),
                    ("value_loss", value_loss),
                    ("entropy", entropy_mean),
                    ("approx_kl", approx_kl),
                    ("clip_fraction", clip_fraction),
                    ("grad_norm", grad_norm),
                ):
                    metrics[key].append(float(value.detach().cpu()))

        explained_variance = 1.0 - np.var(
            buffer.returns - buffer.values
        ) / max(np.var(buffer.returns), 1e-8)
        result = {key: float(np.mean(values)) for key, values in metrics.items()}
        result["explained_variance"] = float(explained_variance)
        result["learning_rate"] = self.config.learning_rate
        result["total_steps"] = float(self.total_steps)
        return result

    def train(self, total_steps: int) -> list[dict[str, float]]:
        if total_steps % self.config.n_envs:
            raise ValueError("total_steps must be divisible by n_envs")
        history: list[dict[str, float]] = []
        while self.total_steps < total_steps:
            remaining_vector_steps = (total_steps - self.total_steps) // self.config.n_envs
            rollout_steps = min(self.config.n_steps, remaining_vector_steps)
            rollout, rollout_metrics = self.collect_rollout(rollout_steps)
            update_metrics = self.update(rollout)
            history.append({**rollout_metrics, **update_metrics})
        return history

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path), asdict(self.config))


class MaskedPPOTrainer(PPOTrainer):
    """Own PPO with action support supplied by the Track A legality contract."""

    def current_action_masks(self) -> np.ndarray:
        return np.stack([env.action_masks() for env in self.envs])
