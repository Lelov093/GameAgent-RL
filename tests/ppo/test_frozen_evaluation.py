from __future__ import annotations

import numpy as np

from gameagent_rl.ppo.frozen_evaluation import evaluate_episode, random_policy


def test_random_episode_is_reproducible_for_frozen_seed_pair():
    first, _ = evaluate_episode(
        random_policy(3001, 9001, masked=False),
        3001,
        algorithm_id="a0",
        training_seed=None,
        policy_action_seed=9001,
    )
    second, _ = evaluate_episode(
        random_policy(3001, 9001, masked=False),
        3001,
        algorithm_id="a0",
        training_seed=None,
        policy_action_seed=9001,
    )
    assert first == second


def test_masked_random_has_zero_invalid_actions_and_replay_frames():
    record, replay = evaluate_episode(
        random_policy(3002, 9002, masked=True),
        3002,
        algorithm_id="a0m",
        training_seed=None,
        policy_action_seed=9002,
        capture_frames=True,
    )
    assert record["invalid_action_count"] == 0
    assert replay is not None
    assert replay["frames"].shape[0] == record["episode_length"] + 1
    assert replay["actions"].shape == replay["legal_actions"].shape
    assert np.all(replay["legal_actions"])
