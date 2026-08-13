from __future__ import annotations

import numpy as np
import torch

from gameagent_rl.ppo.model import ActorCritic


def test_distribution_log_prob_and_ratio_identity():
    torch.manual_seed(7)
    model = ActorCritic(6, 3, hidden_size=16)
    observations = torch.randn(8, 6)
    distribution, _ = model.distribution_and_value(observations)
    assert torch.allclose(distribution.probs.sum(dim=-1), torch.ones(8))

    actions, old_log_probs, _ = model.act(observations)
    new_log_probs, _, _ = model.evaluate_actions(observations, actions)
    assert torch.allclose(old_log_probs, new_log_probs)
    assert torch.allclose(torch.exp(new_log_probs - old_log_probs), torch.ones(8))


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(11)
    model = ActorCritic(5, 2, hidden_size=16)
    observation = np.asarray([1, 0, 1, 0, 1], dtype=np.float32)
    expected = model.predict(observation)
    path = tmp_path / "ppo.pt"
    model.save(str(path), {"seed": 11})

    restored, config = ActorCritic.load(str(path))
    assert restored.predict(observation) == expected
    assert config == {"seed": 11}
    for original, loaded in zip(model.parameters(), restored.parameters()):
        assert torch.equal(original, loaded)
