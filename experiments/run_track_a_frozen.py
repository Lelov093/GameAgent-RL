from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from gameagent_rl.ppo.frozen_evaluation import (
    evaluate_episode,
    learned_policy,
    random_policy,
)
from gameagent_rl.ppo.model import ActorCritic
from gameagent_rl.runtime import PROJECT_ROOT, runtime_info


FROZEN_SEEDS = tuple(range(3001, 3101))
ACTION_SEEDS = tuple(range(9001, 9006))
TRAINING_SEEDS = tuple(range(1001, 1006))


def summarize(records: list[dict]) -> dict:
    return {
        "episodes": len(records),
        "success_rate": float(np.mean([row["success"] for row in records])),
        "mean_return": float(np.mean([row["return"] for row in records])),
        "mean_episode_length": float(
            np.mean([row["episode_length"] for row in records])
        ),
        "invalid_action_rate": sum(row["invalid_action_count"] for row in records)
        / sum(row["episode_length"] for row in records),
        "failure_counts": dict(Counter(row["failure_type"] for row in records)),
    }


def write_records(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")


def evaluate_random_condition(condition: str, masked: bool) -> tuple[list[dict], dict]:
    records: list[dict] = []
    for evaluation_seed in FROZEN_SEEDS:
        for action_seed in ACTION_SEEDS:
            record, _ = evaluate_episode(
                random_policy(evaluation_seed, action_seed, masked),
                evaluation_seed,
                algorithm_id=condition,
                training_seed=None,
                policy_action_seed=action_seed,
            )
            records.append(record)
    return records, summarize(records)


def evaluate_learned_condition(
    condition: str, masked: bool, training_root: Path
) -> tuple[list[dict], dict, list[dict]]:
    records: list[dict] = []
    per_seed: list[dict] = []
    for training_seed in TRAINING_SEEDS:
        checkpoint = (
            training_root / condition / f"seed_{training_seed}" / "final.pt"
        )
        model, _ = ActorCritic.load(str(checkpoint))
        seed_records: list[dict] = []
        for evaluation_seed in FROZEN_SEEDS:
            record, _ = evaluate_episode(
                learned_policy(model, masked),
                evaluation_seed,
                algorithm_id=condition,
                training_seed=training_seed,
            )
            seed_records.append(record)
        records.extend(seed_records)
        per_seed.append({"training_seed": training_seed, **summarize(seed_records)})
    return records, summarize(records), per_seed


def save_replay(
    label: str,
    source: dict,
    training_root: Path,
    replay_dir: Path,
) -> None:
    condition = source["algorithm_id"]
    training_seed = source["training_seed"]
    model, _ = ActorCritic.load(
        str(training_root / condition / f"seed_{training_seed}" / "final.pt")
    )
    record, replay = evaluate_episode(
        learned_policy(model, condition == "a2"),
        int(source["evaluation_seed"]),
        algorithm_id=condition,
        training_seed=int(training_seed),
        capture_frames=True,
    )
    replay_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(replay_dir / f"{label}.npz", **replay)
    metadata = {
        "label": label,
        "checkpoint": "final.pt (100k)",
        "selection_rule": {
            "a1_success": "first A1 success in sorted seed order",
            "a1_invalid_action_heavy": "first A1 episode with invalid rate >= 0.50",
            "a2_success": "first A2 success in sorted seed order",
            "representative_failure": "first key-picked learned-policy failure in sorted order",
        }[label],
        **record,
    }
    (replay_dir / f"{label}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def first(records: list[dict], predicate) -> dict | None:
    return next((record for record in records if predicate(record)), None)


def main() -> None:
    training_root = PROJECT_ROOT / "artifacts" / "track_a" / "formal_training"
    output_dir = PROJECT_ROOT / "artifacts" / "track_a" / "frozen_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, dict] = {}
    per_training_seed: dict[str, list[dict]] = {}
    all_records: dict[str, list[dict]] = {}

    for condition, masked in (("a0", False), ("a0m", True)):
        records, summary = evaluate_random_condition(condition, masked)
        all_records[condition] = records
        aggregate[condition] = summary
        write_records(output_dir / f"{condition}_episodes.jsonl", records)

    for condition, masked in (("a1", False), ("a2", True)):
        records, summary, seed_results = evaluate_learned_condition(
            condition, masked, training_root
        )
        all_records[condition] = records
        aggregate[condition] = summary
        per_training_seed[condition] = seed_results
        write_records(output_dir / f"{condition}_episodes.jsonl", records)

    if aggregate["a0m"]["invalid_action_rate"] != 0.0:
        raise RuntimeError("A0-M selected an invalid action")
    if aggregate["a2"]["invalid_action_rate"] != 0.0:
        raise RuntimeError("A2 selected an invalid action")

    selections = {
        "a1_success": first(all_records["a1"], lambda row: row["success"]),
        "a1_invalid_action_heavy": first(
            all_records["a1"], lambda row: row["invalid_action_rate"] >= 0.50
        ),
        "a2_success": first(all_records["a2"], lambda row: row["success"]),
        "representative_failure": first(
            all_records["a1"] + all_records["a2"],
            lambda row: not row["success"] and row["key_picked"],
        ),
    }
    unavailable = [label for label, row in selections.items() if row is None]
    for label, row in selections.items():
        if row is not None:
            save_replay(label, row, training_root, output_dir / "replays")

    result = {
        "aggregate": aggregate,
        "per_training_seed": per_training_seed,
        "replay_unavailable": unavailable,
        "protocol": {
            "frozen_environment_seed_range": [FROZEN_SEEDS[0], FROZEN_SEEDS[-1]],
            "random_policy_action_seeds": list(ACTION_SEEDS),
            "training_seeds": list(TRAINING_SEEDS),
            "checkpoint": "final.pt (100k)",
            "learned_policy_evaluation": "deterministic",
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
