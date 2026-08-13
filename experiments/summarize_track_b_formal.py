from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gameagent_rl.runtime import PROJECT_ROOT


def main() -> None:
    root = PROJECT_ROOT / "artifacts" / "track_b" / "formal_training"
    aggregate: dict[str, dict] = {}
    shared_configs: set[str] = set()
    for condition in ("b1", "b2"):
        rows = []
        for seed in range(1001, 1006):
            run_dir = root / condition / f"seed_{seed}"
            summary = json.loads((run_dir / "summary.json").read_text())
            metadata = json.loads((run_dir / "run_metadata.json").read_text())
            config = dict(metadata["config"])
            config["experiment"] = dict(config["experiment"])
            config["experiment"].pop("training_seed", None)
            shared_configs.add(json.dumps(config, sort_keys=True))
            rows.append(summary)
        success = [row["final_dev"]["success_rate"] for row in rows]
        memory = [row["final_dev"]["memory_accuracy"] for row in rows]
        reach = [row["final_dev"]["decision_reach_rate"] for row in rows]
        aggregate[condition] = {
            "runs_completed": len(rows),
            "success_mean": float(np.mean(success)),
            "success_std": float(np.std(success)),
            "memory_accuracy_mean": float(np.mean(memory)),
            "memory_accuracy_std": float(np.std(memory)),
            "decision_reach_mean": float(np.mean(reach)),
            "per_seed": rows,
        }
    if len(shared_configs) != 1:
        raise RuntimeError("B1/B2 frozen shared configurations differ")
    aggregate["integrity"] = {
        "shared_configuration": True,
        "training_budget_per_seed": 500_000,
        "formal_runs_completed": 10,
        "b2r_training_runs": 0,
        "frozen_evaluation_started": False,
    }
    (root / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
