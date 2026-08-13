from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from gameagent_rl.envs.doorkey import TRACK_A_ENV_ID, DoorKeyContractEnv
from gameagent_rl.ppo.evaluation import evaluate_doorkey
from gameagent_rl.ppo.trainer import MaskedPPOTrainer, PPOConfig, PPOTrainer
from gameagent_rl.runtime import PROJECT_ROOT, runtime_info


CHECKPOINTS = (25_000, 50_000, 75_000, 100_000)
DEV_SEEDS = tuple(range(2001, 2021))


def normalized_auc(curve: list[dict]) -> float:
    steps = np.asarray([point["steps"] for point in curve], dtype=np.float64)
    success = np.asarray(
        [point["success_rate"] for point in curve], dtype=np.float64
    )
    return float(np.trapezoid(success, steps) / CHECKPOINTS[-1])


def steps_to_80(curve: list[dict]) -> int | None:
    return next(
        (int(point["steps"]) for point in curve if point["success_rate"] >= 0.80),
        None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=("a1", "a2"))
    parser.add_argument("--seed", type=int, choices=range(1001, 1006), required=True)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / "ppo_v1_canonical.toml"
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    config = replace(PPOConfig(**raw_config["ppo"]), seed=args.seed)
    masked = args.condition == "a2"
    trainer_type = MaskedPPOTrainer if masked else PPOTrainer
    output_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "track_a"
        / "formal_training"
        / args.condition
        / f"seed_{args.seed}"
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_history: list[dict[str, float]] = []
    curve: list[dict] = []
    trainer = trainer_type(DoorKeyContractEnv, config)
    try:
        initial = evaluate_doorkey(trainer.model, DEV_SEEDS, masked=masked)
        curve.append({"steps": 0, **initial.as_dict()})
        for checkpoint in CHECKPOINTS:
            training_history.extend(trainer.train(checkpoint))
            result = evaluate_doorkey(trainer.model, DEV_SEEDS, masked=masked)
            curve.append({"steps": checkpoint, **result.as_dict()})
            trainer.save(checkpoint_dir / f"step_{checkpoint}.pt")
        trainer.save(output_dir / "final.pt")
    finally:
        trainer.close()

    if trainer.total_steps != CHECKPOINTS[-1]:
        raise RuntimeError("formal run did not stop at the frozen budget")
    if not all(
        np.isfinite(value) for row in training_history for value in row.values()
    ):
        raise RuntimeError("training diagnostics contain NaN or Inf")
    if masked and any(point["invalid_action_rate"] != 0.0 for point in curve):
        raise RuntimeError("masked policy selected an invalid action")

    with (output_dir / "training_metrics.jsonl").open("w", encoding="utf-8") as output:
        for row in training_history:
            output.write(json.dumps(row) + "\n")
    with (output_dir / "dev_curve.jsonl").open("w", encoding="utf-8") as output:
        for point in curve:
            output.write(json.dumps(point) + "\n")
    summary = {
        "condition": args.condition,
        "training_seed": args.seed,
        "final_dev": curve[-1],
        "normalized_auc_success": normalized_auc(curve),
        "steps_to_80": steps_to_80(curve),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    metadata = {
        "algorithm_id": "ppo_v1_frozen",
        "condition": args.condition,
        "implementation": trainer_type.__name__,
        "environment_id": TRACK_A_ENV_ID,
        "training_seed": args.seed,
        "development_evaluation_seeds": [DEV_SEEDS[0], DEV_SEEDS[-1]],
        "deterministic_evaluation": True,
        "training_budget": CHECKPOINTS[-1],
        "checkpoint_steps": list(CHECKPOINTS),
        "final_checkpoint_rule": "step_100000",
        "masked": masked,
        "ppo_config": asdict(config),
        "frozen_config_source": "configs/ppo_v1_frozen.yaml",
        "runtime": runtime_info(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
