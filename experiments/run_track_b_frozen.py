from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from gameagent_rl.envs.memory import MemoryContractEnv
from gameagent_rl.runtime import PROJECT_ROOT, runtime_info


FROZEN_SEEDS = tuple(range(3001, 3101))
TRAINING_SEEDS = tuple(range(1001, 1006))
RANDOM_ACTION_SEED = 9001


def failure_type(info: dict[str, Any], truncated: bool) -> str:
    if info["correct_choice"]:
        return "success"
    if info["wrong_choice"]:
        return "wrong_memory_choice"
    return "timeout_before_decision" if truncated else "corridor_navigation_failure"


def run_episode(
    model: PPO | RecurrentPPO | None,
    condition: str,
    evaluation_seed: int,
    training_seed: int | None,
    *,
    reset_recurrent_state: bool = False,
    capture_frames: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    rng = np.random.default_rng(
        np.random.SeedSequence([evaluation_seed, RANDOM_ACTION_SEED])
    )
    render_mode = "rgb_array" if capture_frames else None
    with MemoryContractEnv(render_mode=render_mode) as env:
        observation, info = env.reset(seed=evaluation_seed)
        cue_identity = env.native_env.grid.get(*env.cue_position).type
        correct_branch = "upper" if env.native_env.success_pos[1] < 5 else "lower"
        state = None
        episode_start = np.ones((1,), dtype=bool)
        actions: list[int] = []
        frames = [env.render()] if capture_frames else []
        reset_count = 0
        state_norm_at_reset = None
        episode_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            if model is None:
                action = int(rng.integers(env.action_space.n))
            elif isinstance(model, RecurrentPPO):
                action, state = model.predict(
                    observation,
                    state=state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                action = int(action)
            else:
                action, _ = model.predict(observation, deterministic=True)
                action = int(action)
            observation, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            episode_return += float(reward)
            episode_start[:] = terminated or truncated
            if reset_recurrent_state and info["reset_trigger"]:
                state_norm_at_reset = float(
                    sum(np.linalg.norm(component) for component in state)
                )
                state = tuple(np.zeros_like(component) for component in state)
                reset_count += 1
            if capture_frames:
                frames.append(env.render())
        chosen_branch = (
            "upper" if env.native_env.agent_pos[1] < 5 else "lower"
        ) if info["decision_reached"] else None
    record = {
        "condition": condition,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "random_action_seed": RANDOM_ACTION_SEED if model is None else None,
        "cue_identity": cue_identity,
        "correct_branch": correct_branch,
        "chosen_branch": chosen_branch,
        "success": bool(info["correct_choice"]),
        "decision_reached": bool(info["decision_reached"]),
        "memory_correct": bool(info["correct_choice"]),
        "return": episode_return,
        "episode_length": len(actions),
        "termination_type": "terminated" if terminated else "truncated",
        "failure_type": failure_type(info, truncated),
        "reset_count": reset_count,
        "state_norm_at_reset": state_norm_at_reset,
    }
    replay = None
    if capture_frames:
        replay = {
            "frames": np.stack(frames),
            "actions": np.asarray(actions, dtype=np.int64),
        }
    return record, replay


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    reached = [row for row in records if row["decision_reached"]]
    return {
        "episodes": len(records),
        "success_rate": float(np.mean([row["success"] for row in records])),
        "decision_reach_rate": len(reached) / len(records),
        "memory_accuracy": (
            sum(row["memory_correct"] for row in reached) / len(reached)
            if reached
            else None
        ),
        "mean_return": float(np.mean([row["return"] for row in records])),
        "mean_episode_length": float(
            np.mean([row["episode_length"] for row in records])
        ),
        "chosen_branch_counts": dict(
            Counter(row["chosen_branch"] for row in reached)
        ),
        "failure_counts": dict(Counter(row["failure_type"] for row in records)),
        "reset_count_total": sum(row["reset_count"] for row in records),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")


def first(
    rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]
) -> dict[str, Any] | None:
    return next((row for row in rows if predicate(row)), None)


def save_replay(
    label: str,
    source: dict[str, Any],
    training_root: Path,
    replay_dir: Path,
) -> None:
    condition = source["condition"]
    training_seed = int(source["training_seed"])
    checkpoint = training_root / "b2" / f"seed_{training_seed}" / "final.zip"
    if condition == "b1":
        checkpoint = training_root / "b1" / f"seed_{training_seed}" / "final.zip"
        model: PPO | RecurrentPPO = PPO.load(checkpoint, device="cpu")
    else:
        model = RecurrentPPO.load(checkpoint, device="cpu")
    record, replay = run_episode(
        model,
        condition,
        int(source["evaluation_seed"]),
        training_seed,
        reset_recurrent_state=condition == "b2r",
        capture_frames=True,
    )
    replay_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(replay_dir / f"{label}.npz", **replay)
    metadata = {
        "label": label,
        "checkpoint": "final.zip (500k)",
        "selection_rule": {
            "feedforward_failure": "first B1 failure in sorted seed order",
            "recurrent_success": "first B2 success in sorted seed order",
            "b2r_memory_failure": "first paired B2 success and B2-R failure",
            "representative_recurrent_failure": "first B2 failure in sorted seed order",
        }[label],
        **record,
    }
    (replay_dir / f"{label}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    training_root = PROJECT_ROOT / "artifacts" / "track_b" / "formal_training"
    output_dir = PROJECT_ROOT / "artifacts" / "track_b" / "frozen_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_condition: dict[str, list[dict[str, Any]]] = {}
    aggregate: dict[str, dict[str, Any]] = {}
    per_seed: dict[str, list[dict[str, Any]]] = {"b1": [], "b2": [], "b2r": []}

    b0_records = [
        run_episode(None, "b0", seed, None)[0] for seed in FROZEN_SEEDS
    ]
    records_by_condition["b0"] = b0_records
    aggregate["b0"] = summarize(b0_records)
    write_jsonl(output_dir / "b0_episodes.jsonl", b0_records)

    for condition in ("b1", "b2", "b2r"):
        condition_records: list[dict[str, Any]] = []
        for training_seed in TRAINING_SEEDS:
            checkpoint_condition = "b1" if condition == "b1" else "b2"
            checkpoint = (
                training_root
                / checkpoint_condition
                / f"seed_{training_seed}"
                / "final.zip"
            )
            model = (
                PPO.load(checkpoint, device="cpu")
                if condition == "b1"
                else RecurrentPPO.load(checkpoint, device="cpu")
            )
            seed_records = [
                run_episode(
                    model,
                    condition,
                    evaluation_seed,
                    training_seed,
                    reset_recurrent_state=condition == "b2r",
                )[0]
                for evaluation_seed in FROZEN_SEEDS
            ]
            condition_records.extend(seed_records)
            per_seed[condition].append(
                {"training_seed": training_seed, **summarize(seed_records)}
            )
        records_by_condition[condition] = condition_records
        aggregate[condition] = summarize(condition_records)
        write_jsonl(output_dir / f"{condition}_episodes.jsonl", condition_records)

    b2_pairs = {
        (row["training_seed"], row["evaluation_seed"]): row
        for row in records_by_condition["b2"]
    }
    b2r_pairs = {
        (row["training_seed"], row["evaluation_seed"]): row
        for row in records_by_condition["b2r"]
    }
    reset_effect = []
    for training_seed in TRAINING_SEEDS:
        b2_summary = next(
            row for row in per_seed["b2"] if row["training_seed"] == training_seed
        )
        b2r_summary = next(
            row for row in per_seed["b2r"] if row["training_seed"] == training_seed
        )
        reset_effect.append(
            {
                "training_seed": training_seed,
                "success_effect": b2_summary["success_rate"]
                - b2r_summary["success_rate"],
                "memory_effect": b2_summary["memory_accuracy"]
                - b2r_summary["memory_accuracy"],
            }
        )

    paired_reset_failure = first(
        records_by_condition["b2r"],
        lambda row: b2_pairs[(row["training_seed"], row["evaluation_seed"])][
            "success"
        ]
        and not row["success"],
    )
    selections = {
        "feedforward_failure": first(
            records_by_condition["b1"], lambda row: not row["success"]
        ),
        "recurrent_success": first(
            records_by_condition["b2"], lambda row: row["success"]
        ),
        "b2r_memory_failure": paired_reset_failure,
        "representative_recurrent_failure": first(
            records_by_condition["b2"], lambda row: not row["success"]
        ),
    }
    replay_unavailable = [label for label, row in selections.items() if row is None]
    for label, source in selections.items():
        if source is not None:
            save_replay(label, source, training_root, output_dir / "replays")

    result = {
        "aggregate": aggregate,
        "per_training_seed": per_seed,
        "reset_effect": reset_effect,
        "replay_unavailable": replay_unavailable,
        "protocol": {
            "frozen_seed_range": [FROZEN_SEEDS[0], FROZEN_SEEDS[-1]],
            "training_seeds": list(TRAINING_SEEDS),
            "random_action_seed": RANDOM_ACTION_SEED,
            "checkpoint": "final.zip (500k)",
            "deterministic_learned_policy": True,
            "b2r_uses_same_b2_checkpoint": True,
            "frozen_results_used_for_training_or_tuning": False,
        },
        "runtime": runtime_info(),
    }
    (output_dir / "frozen_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
