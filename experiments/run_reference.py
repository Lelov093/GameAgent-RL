from __future__ import annotations

import argparse
import json
from pathlib import Path

from gameagent_rl.envs.doorkey import DoorKeyContractEnv
from gameagent_rl.envs.memory import MemoryContractEnv
from gameagent_rl.reference import (
    continue_training,
    evaluate_random,
    load_reference_config,
    run_training,
)
from gameagent_rl.runtime import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "condition",
        choices=("a0", "a0m", "a1r", "a2r", "b0", "b1", "b2", "b2_budget"),
    )
    args = parser.parse_args()

    track = "a" if args.condition.startswith("a") else "b"
    config_path = PROJECT_ROOT / "configs" / f"reference_track_{track}.toml"
    config = load_reference_config(config_path)
    output_dir = PROJECT_ROOT / "artifacts" / "reference" / f"track_{track}"

    if args.condition in {"a0", "a0m", "b0"}:
        env_factory = DoorKeyContractEnv if track == "a" else MemoryContractEnv
        result = evaluate_random(
            env_factory,
            config["experiment"]["eval_seeds"],
            masked=args.condition == "a0m",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{args.condition}_final.json").write_text(
            json.dumps(result.as_dict(), indent=2), encoding="utf-8"
        )
    elif args.condition == "b2_budget":
        result = continue_training(
            "recurrent_ppo",
            config_path,
            output_dir / "recurrent_ppo.zip",
            additional_steps=1_500_000,
            output_dir=output_dir,
        )
    else:
        algorithm = {
            "a1r": "ppo",
            "a2r": "maskable_ppo",
            "b1": "ppo",
            "b2": "recurrent_ppo",
        }[args.condition]
        result = run_training(algorithm, config_path, output_dir)
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
