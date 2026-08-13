from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import replace

import numpy as np

from gameagent_rl.envs.doorkey import DoorKeyContractEnv
from gameagent_rl.ppo.evaluation import evaluate_doorkey
from gameagent_rl.ppo.trainer import MaskedPPOTrainer, PPOConfig
from gameagent_rl.runtime import PROJECT_ROOT, runtime_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / "ppo_v1_canonical.toml"
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    config = replace(PPOConfig(**raw_config["ppo"]), seed=args.seed)
    output_dir = PROJECT_ROOT / "artifacts" / "phase3" / "wb1" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = MaskedPPOTrainer(DoorKeyContractEnv, config)
    try:
        history = trainer.train(args.steps)
        if not all(
            np.isfinite(value) for row in history for value in row.values()
        ):
            raise RuntimeError("Training diagnostics contain NaN or Inf")
        result = evaluate_doorkey(
            trainer.model, range(2001, 2021), masked=True
        )
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
                "algorithm": "own_masked_ppo",
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
