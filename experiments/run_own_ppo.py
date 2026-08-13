from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np

from gameagent_rl.envs.doorkey import DoorKeyContractEnv
from gameagent_rl.ppo.evaluation import evaluate_doorkey
from gameagent_rl.ppo.trainer import PPOConfig, PPOTrainer
from gameagent_rl.runtime import PROJECT_ROOT, runtime_info


def load_config(path: Path, seed: int) -> tuple[PPOConfig, dict]:
    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    return replace(PPOConfig(**raw["ppo"]), seed=seed), raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / "ppo_v1_canonical.toml"
    config, raw_config = load_config(config_path, args.seed)
    output_dir = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "artifacts" / "phase2" / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = PPOTrainer(DoorKeyContractEnv, config)
    try:
        history = trainer.train(args.steps)
        if not all(
            np.isfinite(value)
            for row in history
            for value in row.values()
        ):
            raise RuntimeError("Training diagnostics contain NaN or Inf")
        result = evaluate_doorkey(trainer.model, range(2001, 2021))
        trainer.save(output_dir / "final.pt")
    finally:
        trainer.close()

    with (output_dir / "learning_curve.jsonl").open("w", encoding="utf-8") as output:
        for row in history:
            output.write(json.dumps(row) + "\n")
    (output_dir / "final.json").write_text(
        json.dumps(result.as_dict(), indent=2), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "algorithm": "own_ppo",
                "training_seed": args.seed,
                "requested_steps": args.steps,
                "actual_steps": trainer.total_steps,
                "config": raw_config,
                "runtime": runtime_info(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
