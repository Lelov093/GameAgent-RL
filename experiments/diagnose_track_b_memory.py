from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from minigrid.core.world_object import Ball, Key

from gameagent_rl.envs.memory import MemoryContractEnv
from gameagent_rl.runtime import PROJECT_ROOT, runtime_info


DEV_SEEDS = tuple(range(2001, 2021))


def cue_identity(env: MemoryContractEnv) -> str:
    return env.native_env.grid.get(*env.cue_position).type


def branch(position: tuple[int, int]) -> str:
    return "upper" if position[1] < 5 else "lower"


def evaluate(
    model: PPO | RecurrentPPO,
    algorithm: str,
    *,
    reset_recurrent_state: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for seed in DEV_SEEDS:
        with MemoryContractEnv() as env:
            observation, info = env.reset(seed=seed)
            initial_cue = cue_identity(env)
            correct_branch = branch(tuple(env.native_env.success_pos))
            state = None
            episode_start = np.ones((1,), dtype=bool)
            actions: list[int] = []
            reset_count = 0
            state_norm_at_trigger = None
            episode_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                if isinstance(model, RecurrentPPO):
                    action, state = model.predict(
                        observation,
                        state=state,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                else:
                    action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(
                    int(action)
                )
                actions.append(int(action))
                episode_return += float(reward)
                episode_start[:] = terminated or truncated
                if reset_recurrent_state and info["reset_trigger"]:
                    state_norm_at_trigger = float(
                        sum(np.linalg.norm(component) for component in state)
                    )
                    state = tuple(np.zeros_like(component) for component in state)
                    reset_count += 1
            chosen_branch = (
                branch(tuple(env.native_env.agent_pos))
                if info["correct_choice"] or info["wrong_choice"]
                else None
            )
            episodes.append(
                {
                    "algorithm": algorithm,
                    "evaluation_seed": seed,
                    "cue_identity": initial_cue,
                    "correct_branch": correct_branch,
                    "chosen_branch": chosen_branch,
                    "decision_reached": bool(info["decision_reached"]),
                    "correct_choice": bool(info["correct_choice"]),
                    "wrong_choice": bool(info["wrong_choice"]),
                    "success": bool(info["correct_choice"]),
                    "return": episode_return,
                    "episode_length": len(actions),
                    "termination_type": "terminated" if terminated else "truncated",
                    "actions": actions,
                    "reset_count": reset_count,
                    "state_norm_at_trigger": state_norm_at_trigger,
                }
            )
    reached = [row for row in episodes if row["decision_reached"]]
    branch_counts = Counter(row["chosen_branch"] for row in reached)
    cue_branch = Counter(
        (row["cue_identity"], row["chosen_branch"]) for row in reached
    )
    summary = {
        "episodes": len(episodes),
        "decision_reach_rate": len(reached) / len(episodes),
        "memory_decision_accuracy": (
            sum(row["correct_choice"] for row in reached) / len(reached)
            if reached
            else None
        ),
        "task_success_rate": sum(row["success"] for row in episodes) / len(episodes),
        "mean_return": float(np.mean([row["return"] for row in episodes])),
        "mean_episode_length": float(
            np.mean([row["episode_length"] for row in episodes])
        ),
        "chosen_branch_counts": dict(branch_counts),
        "cue_chosen_branch_counts": {
            f"{cue}->{selected}": count
            for (cue, selected), count in cue_branch.items()
        },
        "unique_action_sequences": len(
            {tuple(row["actions"]) for row in episodes}
        ),
        "reset_count_total": sum(row["reset_count"] for row in episodes),
    }
    return episodes, summary


def cue_leakage_check() -> dict[str, Any]:
    checks: list[bool] = []
    for seed in DEV_SEEDS:
        with MemoryContractEnv() as env:
            env.reset(seed=seed)
            env.native_env.agent_pos = np.asarray(env.decision_position)
            env.native_env.agent_dir = 0
            before = env.native_env.gen_obs()["image"].copy()
            cue = env.native_env.grid.get(*env.cue_position)
            replacement = Ball(cue.color) if cue.type == "key" else Key(cue.color)
            env.native_env.grid.set(*env.cue_position, replacement)
            after = env.native_env.gen_obs()["image"].copy()
            checks.append(np.array_equal(before, after))
    return {
        "seeds_checked": len(checks),
        "cue_swap_preserved_final_decision_observation": all(checks),
    }


def main() -> None:
    checkpoint_dir = PROJECT_ROOT / "artifacts" / "reference" / "track_b"
    b1 = PPO.load(checkpoint_dir / "ppo.zip", device="cpu")
    b2 = RecurrentPPO.load(checkpoint_dir / "recurrent_ppo.zip", device="cpu")
    with MemoryContractEnv() as sample_env:
        _, reset_info = sample_env.reset(seed=DEV_SEEDS[0])
        environment_contract = {
            "observation_shape": list(sample_env.observation_space.shape),
            "b1_observation_shape": list(b1.observation_space.shape),
            "b2_observation_shape": list(b2.observation_space.shape),
            "action_count": int(sample_env.action_space.n),
            "b1_action_count": int(b1.action_space.n),
            "b2_action_count": int(b2.action_space.n),
            "native_max_steps": int(sample_env.native_env.max_steps),
            "fixed_start_position": [
                int(value) for value in sample_env.native_env.agent_pos
            ],
            "fixed_start_direction": int(sample_env.native_env.agent_dir),
            "cue_visible_at_reset": reset_info["cue_visible"],
            "shared_environment_wrapper": "MemoryContractEnv",
            "native_reward_termination_and_horizon": True,
        }
    results: dict[str, Any] = {}
    output_dir = PROJECT_ROOT / "artifacts" / "track_b" / "preformal_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    for algorithm, model, reset in (
        ("b1", b1, False),
        ("b2", b2, False),
        ("b2r", b2, True),
    ):
        episodes, summary = evaluate(
            model, algorithm, reset_recurrent_state=reset
        )
        results[algorithm] = summary
        with (output_dir / f"{algorithm}_trajectories.jsonl").open(
            "w", encoding="utf-8"
        ) as output:
            for episode in episodes:
                output.write(json.dumps(episode) + "\n")
    results["cue_leakage_check"] = cue_leakage_check()
    results["contract"] = {
        "development_seed_range": [DEV_SEEDS[0], DEV_SEEDS[-1]],
        "b1_checkpoint": "artifacts/reference/track_b/ppo.zip",
        "b2_checkpoint": "artifacts/reference/track_b/recurrent_ppo.zip",
        "b2r_checkpoint": "same B2 checkpoint; evaluation-time state reset only",
        "deterministic": True,
        "training_performed": False,
        "frozen_seeds_used": False,
        "environment": environment_contract,
        "recurrent_lifecycle": {
            "state_initialized_none_each_episode": True,
            "episode_start_true_on_reset": True,
            "episode_start_false_during_episode": True,
            "b2r_reset_on_environment_trigger": True,
        },
    }
    results["runtime"] = runtime_info()
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
