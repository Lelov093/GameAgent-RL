from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from gameagent_rl.envs.doorkey import DoorKeyContractEnv
from gameagent_rl.ppo.distributions import MaskedCategorical
from gameagent_rl.ppo.model import ActorCritic
from gameagent_rl.ppo.trainer import MaskedPPOTrainer, PPOConfig


def test_masked_probability_normalization_and_support():
    logits = torch.tensor([[1.0, 2.0, -1.0, 0.5], [0.0, 0.0, 0.0, 0.0]])
    masks = torch.tensor(
        [[True, False, True, False], [False, True, False, True]]
    )
    distribution = MaskedCategorical(logits, masks)
    assert torch.allclose(distribution.probs.sum(dim=-1), torch.ones(2))
    assert torch.equal(distribution.probs[~masks], torch.zeros(4))
    samples = distribution.sample((10_000,))
    sampled_masks = masks.unsqueeze(0).expand(10_000, -1, -1)
    assert sampled_masks.gather(2, samples.unsqueeze(-1)).all()


def test_masked_log_prob_entropy_and_gradients_are_finite():
    logits = torch.randn(16, 5, requires_grad=True)
    masks = torch.tensor([[True, True, False, False, True]]).expand(16, -1)
    distribution = MaskedCategorical(logits, masks)
    actions = distribution.sample()
    objective = -(distribution.log_prob(actions) + 0.01 * distribution.entropy()).mean()
    objective.backward()
    assert torch.isfinite(distribution.log_prob(actions)).all()
    assert torch.isfinite(distribution.entropy()).all()
    assert torch.isfinite(logits.grad).all()


def test_mask_validation_rejects_wrong_contract():
    logits = torch.zeros(2, 3)
    with pytest.raises(TypeError):
        MaskedCategorical(logits, torch.ones(2, 3))
    with pytest.raises(ValueError):
        MaskedCategorical(logits, torch.ones(2, 2, dtype=torch.bool))
    with pytest.raises(ValueError):
        MaskedCategorical(logits, torch.zeros(2, 3, dtype=torch.bool))


def test_all_actions_valid_is_equivalent_to_vanilla_categorical():
    torch.manual_seed(31)
    logits = torch.randn(12, 5)
    vanilla = Categorical(logits=logits)
    masked = MaskedCategorical(logits, torch.ones_like(logits, dtype=torch.bool))
    actions = torch.arange(12) % 5
    assert torch.allclose(masked.probs, vanilla.probs)
    assert torch.allclose(masked.log_prob(actions), vanilla.log_prob(actions))
    assert torch.allclose(masked.entropy(), vanilla.entropy())


def test_rollout_saves_masks_and_update_reuses_matching_support():
    trainer = MaskedPPOTrainer(
        DoorKeyContractEnv,
        PPOConfig(
            n_envs=2,
            n_steps=8,
            batch_size=8,
            update_epochs=1,
            hidden_size=32,
            seed=37,
        ),
    )
    try:
        rollout, metrics = trainer.collect_rollout()
        assert rollout.action_masks is not None
        selected_masks = np.take_along_axis(
            rollout.action_masks, rollout.actions[..., None], axis=-1
        )
        assert selected_masks.all()
        observations = torch.as_tensor(rollout.observations.reshape(16, -1))
        actions = torch.as_tensor(rollout.actions.reshape(16))
        masks = torch.as_tensor(rollout.action_masks.reshape(16, -1))
        with torch.no_grad():
            restored_log_probs, _, _ = trainer.model.evaluate_actions(
                observations, actions, masks
            )
        assert torch.allclose(
            restored_log_probs, torch.as_tensor(rollout.log_probs.reshape(16))
        )
        update_metrics = trainer.update(rollout)
        assert metrics["rollout_invalid_action_rate"] == 0.0
        assert all(np.isfinite(value) for value in update_metrics.values())
    finally:
        trainer.close()


def test_masked_predict_never_selects_illegal_action():
    model = ActorCritic(4, 5, hidden_size=8)
    with torch.no_grad():
        model.actor.bias.copy_(torch.tensor([0.0, 20.0, 10.0, 5.0, 1.0]))
    observation = np.zeros(4, dtype=np.float32)
    mask = np.asarray([True, False, False, False, True])
    assert model.predict(observation, action_masks=mask) in {0, 4}
