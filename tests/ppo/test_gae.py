from __future__ import annotations

import numpy as np

from gameagent_rl.ppo.buffer import compute_gae


def test_gae_matches_hand_calculation_and_terminal_does_not_bootstrap():
    advantages, returns = compute_gae(
        rewards=np.asarray([[1.0], [2.0]], dtype=np.float32),
        values=np.asarray([[0.5], [0.25]], dtype=np.float32),
        next_values=np.asarray([[0.25], [10.0]], dtype=np.float32),
        terminated=np.asarray([[False], [True]]),
        truncated=np.asarray([[False], [False]]),
        gamma=0.9,
        gae_lambda=0.8,
    )
    np.testing.assert_allclose(advantages[:, 0], [1.985, 1.75], rtol=1e-6)
    np.testing.assert_allclose(returns[:, 0], [2.485, 2.0], rtol=1e-6)


def test_time_limit_truncation_bootstraps_final_observation_without_leaking_episode():
    common = dict(
        rewards=np.asarray([[1.0]], dtype=np.float32),
        values=np.asarray([[0.5]], dtype=np.float32),
        next_values=np.asarray([[2.0]], dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.95,
    )
    truncated_advantage, _ = compute_gae(
        **common,
        terminated=np.asarray([[False]]),
        truncated=np.asarray([[True]]),
    )
    terminal_advantage, _ = compute_gae(
        **common,
        terminated=np.asarray([[True]]),
        truncated=np.asarray([[False]]),
    )
    np.testing.assert_allclose(truncated_advantage, [[2.3]])
    np.testing.assert_allclose(terminal_advantage, [[0.5]])
