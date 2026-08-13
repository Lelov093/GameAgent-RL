from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gameagent_rl.runtime import PROJECT_ROOT


CHECKPOINTS = (25_000, 50_000, 75_000, 100_000)
SEEDS = range(1001, 1006)


def main() -> None:
    root = PROJECT_ROOT / "artifacts" / "track_a" / "formal_training"
    output: dict[str, dict] = {}
    shared_configs: set[str] = set()
    for condition in ("a1", "a2"):
        rows: list[dict] = []
        for seed in SEEDS:
            run_dir = root / condition / f"seed_{seed}"
            summary = json.loads((run_dir / "summary.json").read_text())
            metadata = json.loads((run_dir / "run_metadata.json").read_text())
            config = dict(metadata["ppo_config"])
            config.pop("seed")
            shared_configs.add(json.dumps(config, sort_keys=True))
            rows.append(summary)
        final_success = [row["final_dev"]["success_rate"] for row in rows]
        final_invalid = [row["final_dev"]["invalid_action_rate"] for row in rows]
        auc = [row["normalized_auc_success"] for row in rows]
        steps80 = [row["steps_to_80"] for row in rows]
        output[condition] = {
            "runs_completed": len(rows),
            "final_dev_success_mean": float(np.mean(final_success)),
            "final_dev_success_std": float(np.std(final_success)),
            "final_dev_invalid_action_mean": float(np.mean(final_invalid)),
            "normalized_auc_success_mean": float(np.mean(auc)),
            "normalized_auc_success_std": float(np.std(auc)),
            "steps_to_80": steps80,
            "per_seed": rows,
        }
    if len(shared_configs) != 1:
        raise RuntimeError("A1/A2 PPO configurations differ")
    output["integrity"] = {
        "shared_ppo_config": True,
        "checkpoint_steps": list(CHECKPOINTS),
        "frozen_evaluation_started": False,
    }
    destination = root / "aggregate_summary.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
