"""Shared MLP actor-critic for discrete actions."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from gameagent_rl.ppo.distributions import MaskedCategorical


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def distribution_and_value(
        self,
        observations: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[Categorical, torch.Tensor]:
        features = self.encoder(observations)
        logits = self.actor(features)
        distribution = (
            Categorical(logits=logits)
            if action_masks is None
            else MaskedCategorical(logits, action_masks)
        )
        return distribution, self.critic(features).squeeze(-1)

    def act(
        self,
        observations: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self.distribution_and_value(observations, action_masks)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions), values

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self.distribution_and_value(observations, action_masks)
        return distribution.log_prob(actions), distribution.entropy(), values

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
    ) -> int:
        observations = torch.as_tensor(
            observation, dtype=torch.float32, device=next(self.parameters()).device
        ).unsqueeze(0)
        mask_tensor = (
            None
            if action_masks is None
            else torch.as_tensor(
                action_masks,
                dtype=torch.bool,
                device=observations.device,
            ).unsqueeze(0)
        )
        with torch.no_grad():
            distribution, _ = self.distribution_and_value(observations, mask_tensor)
            action = (
                distribution.probs.argmax(dim=-1)
                if deterministic
                else distribution.sample()
            )
        return int(action.item())

    def save(self, path: str, config: dict) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "hidden_size": self.hidden_size,
                "config": config,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str, device: str | torch.device = "cpu"
    ) -> tuple[ActorCritic, dict]:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            checkpoint["observation_dim"],
            checkpoint["action_dim"],
            checkpoint["hidden_size"],
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model, checkpoint["config"]
