from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from gameagent_rl.envs.memory import MemoryContractEnv
from gameagent_rl.reference import (
    build_model,
    evaluate_model,
    load_reference_config,
    make_vector_env,
)
from gameagent_rl.runtime import PROJECT_ROOT, runtime_info, seed_everything


FROZEN_TRAINING_BUDGET = 500_000
DEV_SEEDS = tuple(range(2001, 2021))


def json_metrics(values: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if np.isfinite(numeric):
                metrics[key] = numeric
    return metrics


class FormalTrackBCallback(BaseCallback):
    def __init__(self, output_dir: Path, eval_interval: int):
        super().__init__(verbose=0)
        self.output_dir = output_dir
        self.eval_interval = eval_interval
        self.next_eval = eval_interval
        self.dev_curve: list[dict[str, Any]] = []
        self.training_metrics: list[dict[str, float]] = []

    def _on_training_start(self) -> None:
        initial = evaluate_model(self.model, MemoryContractEnv, DEV_SEEDS)
        self.dev_curve.append({"timesteps": 0, **initial.as_dict()})

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_eval:
            result = evaluate_model(self.model, MemoryContractEnv, DEV_SEEDS)
            self.dev_curve.append(
                {"timesteps": self.next_eval, **result.as_dict()}
            )
            self.next_eval += self.eval_interval
        return self.num_timesteps < FROZEN_TRAINING_BUDGET

    def _on_rollout_end(self) -> None:
        self.training_metrics.append(
            {
                "timesteps": float(self.num_timesteps),
                **json_metrics(self.model.logger.name_to_value),
            }
        )

    def write(self) -> None:
        with (self.output_dir / "dev_curve.jsonl").open(
            "w", encoding="utf-8"
        ) as output:
            for row in self.dev_curve:
                output.write(json.dumps(row) + "\n")
        with (self.output_dir / "training_metrics.jsonl").open(
            "w", encoding="utf-8"
        ) as output:
            for row in self.training_metrics:
                output.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=("b1", "b2"))
    parser.add_argument("--seed", type=int, choices=range(1001, 1006), required=True)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / "reference_track_b.toml"
    config = load_reference_config(config_path)
    if int(config["experiment"]["total_steps"]) != FROZEN_TRAINING_BUDGET:
        raise RuntimeError("Track B training budget differs from frozen contract")
    seed_everything(args.seed)
    output_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "track_b"
        / "formal_training"
        / args.condition
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_env = make_vector_env(
        MemoryContractEnv, int(config["ppo"]["n_envs"]), args.seed
    )
    algorithm = "ppo" if args.condition == "b1" else "recurrent_ppo"
    model = build_model(algorithm, vector_env, config, args.seed)
    callback = FormalTrackBCallback(
        output_dir, int(config["experiment"]["eval_interval"])
    )
    try:
        model.learn(
            total_timesteps=FROZEN_TRAINING_BUDGET,
            callback=callback,
        )
        final = evaluate_model(model, MemoryContractEnv, DEV_SEEDS)
        model.save(output_dir / "final")
    finally:
        vector_env.close()
    callback.write()

    if model.num_timesteps != FROZEN_TRAINING_BUDGET:
        raise RuntimeError(
            f"formal run stopped at {model.num_timesteps}, expected 500000"
        )
    all_values = [
        value
        for row in callback.training_metrics
        for value in row.values()
    ]
    if not all(np.isfinite(value) for value in all_values):
        raise RuntimeError("training metrics contain NaN or Inf")
    if [row["timesteps"] for row in callback.dev_curve] != list(
        range(0, FROZEN_TRAINING_BUDGET + 1, 50_000)
    ):
        raise RuntimeError("Development curve checkpoints are incomplete")

    summary = {
        "condition": args.condition,
        "training_seed": args.seed,
        "actual_training_steps": model.num_timesteps,
        "final_dev": final.as_dict(),
    }
    metadata = {
        "condition": args.condition,
        "implementation": type(model).__name__,
        "training_seed": args.seed,
        "environment": "MiniGrid-MemoryS11-v0 + Research Change 001",
        "development_seed_range": [DEV_SEEDS[0], DEV_SEEDS[-1]],
        "frozen_seeds_used": False,
        "training_budget": FROZEN_TRAINING_BUDGET,
        "final_checkpoint_rule": "final at 500k",
        "config": config,
        "runtime": runtime_info(),
        "recurrent_lifecycle": (
            {
                "episode_starts_managed_by_sb3_contrib": True,
                "evaluation_state_initialized_none": True,
                "evaluation_episode_start_true_on_reset": True,
            }
            if args.condition == "b2"
            else None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
