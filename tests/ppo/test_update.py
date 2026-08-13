from __future__ import annotations

import numpy as np
import torch

from gameagent_rl.ppo.trainer import PPOConfig, PPOTrainer, clipped_policy_loss
from tests.ppo.conftest import TinyChoiceEnv


def test_clipped_surrogate_handles_positive_and_negative_advantages():
    positive_loss, ratio, _ = clipped_policy_loss(
        torch.log(torch.tensor([1.5])),
        torch.zeros(1),
        torch.ones(1),
        0.2,
    )
    assert torch.allclose(ratio, torch.tensor([1.5]))
    assert torch.allclose(positive_loss, torch.tensor(-1.2))

    negative_loss, _, _ = clipped_policy_loss(
        torch.log(torch.tensor([0.5])),
        torch.zeros(1),
        -torch.ones(1),
        0.2,
    )
    assert torch.allclose(negative_loss, torch.tensor(0.8))


def test_single_update_has_finite_gradients_and_changes_parameters():
    trainer = PPOTrainer(
        TinyChoiceEnv,
        PPOConfig(
            n_envs=4,
            n_steps=8,
            batch_size=16,
            update_epochs=2,
            hidden_size=16,
            seed=19,
        ),
    )
    try:
        rollout, _ = trainer.collect_rollout()
        before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
        metrics = trainer.update(rollout)
        assert all(np.isfinite(value) for value in metrics.values())
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in trainer.model.parameters()
        )
        assert any(
            not torch.equal(old, new)
            for old, new in zip(before, trainer.model.parameters())
        )
        assert rollout.episode_starts[0].all()
    finally:
        trainer.close()


def test_tiny_fixture_learns_optimal_action():
    trainer = PPOTrainer(
        TinyChoiceEnv,
        PPOConfig(
            learning_rate=3e-3,
            n_envs=8,
            n_steps=16,
            batch_size=64,
            update_epochs=4,
            hidden_size=32,
            seed=23,
        ),
    )
    try:
        trainer.train(2_048)
        observation, _ = TinyChoiceEnv().reset(seed=1)
        distribution, _ = trainer.model.distribution_and_value(
            torch.as_tensor(observation).unsqueeze(0)
        )
        assert trainer.model.predict(observation) == 1
        assert distribution.probs[0, 1].item() > 0.95
    finally:
        trainer.close()


def test_training_stops_at_exact_budget_with_partial_final_rollout():
    trainer = PPOTrainer(
        TinyChoiceEnv,
        PPOConfig(
            n_envs=4,
            n_steps=16,
            batch_size=16,
            update_epochs=1,
            hidden_size=16,
            seed=29,
        ),
    )
    try:
        trainer.train(100)
        assert trainer.total_steps == 100
    finally:
        trainer.close()
