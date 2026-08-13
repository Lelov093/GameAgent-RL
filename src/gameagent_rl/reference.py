"""Minimal development-only reference baselines for Phase 1 / WB3."""

from __future__ import annotations

import json
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from gameagent_rl.envs.doorkey import DoorKeyContractEnv, sample_masked_action
from gameagent_rl.envs.memory import MemoryContractEnv
from gameagent_rl.runtime import PROJECT_ROOT, runtime_info, seed_everything


EnvFactory = Callable[[], DoorKeyContractEnv | MemoryContractEnv]


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    success_rate: float
    mean_return: float
    mean_length: float
    invalid_action_rate: float | None = None
    decision_reach_rate: float | None = None
    memory_accuracy: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return vars(self)


def load_reference_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def make_vector_env(env_factory: EnvFactory, n_envs: int, seed: int) -> DummyVecEnv:
    def factory(rank: int):
        def build():
            env = env_factory()
            env.reset(seed=seed + rank)
            return env

        return build

    return DummyVecEnv([factory(rank) for rank in range(n_envs)])


def _policy_action(
    model: Any,
    observation: np.ndarray,
    env: DoorKeyContractEnv | MemoryContractEnv,
    state: Any,
    episode_start: np.ndarray,
) -> tuple[int, Any]:
    if isinstance(model, MaskablePPO):
        action, next_state = model.predict(
            observation,
            deterministic=True,
            action_masks=env.action_masks(),
        )
    elif isinstance(model, RecurrentPPO):
        action, next_state = model.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=True,
        )
    else:
        action, next_state = model.predict(observation, deterministic=True)
    return int(action), next_state


def evaluate_model(
    model: Any,
    env_factory: EnvFactory,
    seeds: Sequence[int],
) -> EvaluationResult:
    successes: list[float] = []
    returns: list[float] = []
    lengths: list[int] = []
    invalid_actions = 0
    total_actions = 0
    decisions_reached = 0
    decisions_correct = 0

    for seed in seeds:
        with env_factory() as env:
            observation, info = env.reset(seed=int(seed))
            state = None
            episode_start = np.ones((1,), dtype=bool)
            episode_return = 0.0
            length = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action, state = _policy_action(
                    model, observation, env, state, episode_start
                )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_start[:] = terminated or truncated
                episode_return += float(reward)
                length += 1
                total_actions += 1
                if info.get("selected_action_legal") is False:
                    invalid_actions += 1

            success = bool(info.get("success", info.get("correct_choice", False)))
            successes.append(float(success))
            returns.append(episode_return)
            lengths.append(length)
            if info.get("decision_reached", False):
                decisions_reached += 1
                decisions_correct += int(bool(info.get("correct_choice", False)))

    with env_factory() as sample_env:
        is_track_a = isinstance(sample_env, DoorKeyContractEnv)
    episodes = len(seeds)
    return EvaluationResult(
        episodes=episodes,
        success_rate=float(np.mean(successes)),
        mean_return=float(np.mean(returns)),
        mean_length=float(np.mean(lengths)),
        invalid_action_rate=(invalid_actions / total_actions if is_track_a else None),
        decision_reach_rate=(decisions_reached / episodes if not is_track_a else None),
        memory_accuracy=(
            decisions_correct / decisions_reached
            if not is_track_a and decisions_reached
            else None
        ),
    )


def evaluate_random(
    env_factory: EnvFactory,
    seeds: Sequence[int],
    *,
    masked: bool = False,
    action_seed: int = 9001,
) -> EvaluationResult:
    rng = np.random.default_rng(action_seed)
    class RandomModel:
        def predict(self, observation, deterministic=True):
            del observation, deterministic
            return 0, None

    model = RandomModel()

    def random_policy(model, observation, env, state, episode_start):
        del model, observation, state, episode_start
        if masked:
            action = sample_masked_action(env.action_masks(), rng)
        else:
            action = int(rng.integers(env.action_space.n))
        return action, None

    original = globals()["_policy_action"]
    globals()["_policy_action"] = random_policy
    try:
        return evaluate_model(model, env_factory, seeds)
    finally:
        globals()["_policy_action"] = original


class DevelopmentEvalCallback(BaseCallback):
    def __init__(
        self,
        env_factory: EnvFactory,
        eval_seeds: Sequence[int],
        eval_interval: int,
        output_path: Path,
    ):
        super().__init__(verbose=0)
        self.env_factory = env_factory
        self.eval_seeds = eval_seeds
        self.eval_interval = eval_interval
        self.output_path = output_path
        self.next_eval = eval_interval

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_eval:
            return True
        result = evaluate_model(self.model, self.env_factory, self.eval_seeds)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps({"timesteps": self.num_timesteps, **result.as_dict()}) + "\n"
            )
        self.next_eval += self.eval_interval
        return True


def build_model(
    algorithm: str,
    env: DummyVecEnv,
    config: dict[str, Any],
    seed: int,
):
    ppo = config["ppo"]
    common = dict(
        learning_rate=ppo["learning_rate"],
        n_steps=ppo["n_steps"],
        batch_size=ppo["batch_size"],
        n_epochs=ppo["n_epochs"],
        gamma=ppo["gamma"],
        gae_lambda=ppo["gae_lambda"],
        ent_coef=ppo["ent_coef"],
        seed=seed,
        device="cpu",
        verbose=0,
    )
    net_arch = list(ppo["net_arch"])
    if algorithm == "ppo":
        return PPO("MlpPolicy", env, policy_kwargs={"net_arch": net_arch}, **common)
    if algorithm == "maskable_ppo":
        return MaskablePPO(
            "MlpPolicy", env, policy_kwargs={"net_arch": net_arch}, **common
        )
    if algorithm == "recurrent_ppo":
        return RecurrentPPO(
            "MlpLstmPolicy",
            env,
            policy_kwargs={
                "net_arch": net_arch,
                "lstm_hidden_size": ppo["lstm_hidden_size"],
            },
            **common,
        )
    raise ValueError(f"Unknown reference algorithm: {algorithm}")


def run_training(
    algorithm: str,
    config_path: Path,
    output_dir: Path,
) -> EvaluationResult:
    config = load_reference_config(config_path)
    experiment = config["experiment"]
    seed = int(experiment["training_seed"])
    seed_everything(seed)
    env_factory: EnvFactory = (
        DoorKeyContractEnv if experiment["track"] == "A" else MemoryContractEnv
    )
    vector_env = make_vector_env(env_factory, int(config["ppo"]["n_envs"]), seed)
    model = build_model(algorithm, vector_env, config, seed)
    curve_path = output_dir / f"{algorithm}_curve.jsonl"
    curve_path.unlink(missing_ok=True)
    callback = DevelopmentEvalCallback(
        env_factory,
        experiment["eval_seeds"],
        int(experiment["eval_interval"]),
        curve_path,
    )
    model.learn(total_timesteps=int(experiment["total_steps"]), callback=callback)
    result = evaluate_model(model, env_factory, experiment["eval_seeds"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / algorithm)
    (output_dir / f"{algorithm}_final.json").write_text(
        json.dumps(result.as_dict(), indent=2), encoding="utf-8"
    )
    (output_dir / f"{algorithm}_config.json").write_text(
        json.dumps(
            {"algorithm": algorithm, "config": config, "runtime": runtime_info()},
            indent=2,
        ),
        encoding="utf-8",
    )
    vector_env.close()
    return result


def continue_training(
    algorithm: str,
    config_path: Path,
    checkpoint_path: Path,
    additional_steps: int,
    output_dir: Path,
) -> EvaluationResult:
    """Continue one development checkpoint for a single-variable budget trial."""
    config = load_reference_config(config_path)
    experiment = config["experiment"]
    seed = int(experiment["training_seed"])
    seed_everything(seed)
    env_factory: EnvFactory = (
        DoorKeyContractEnv if experiment["track"] == "A" else MemoryContractEnv
    )
    vector_env = make_vector_env(env_factory, int(config["ppo"]["n_envs"]), seed)
    model_type = {
        "ppo": PPO,
        "maskable_ppo": MaskablePPO,
        "recurrent_ppo": RecurrentPPO,
    }[algorithm]
    model = model_type.load(checkpoint_path, env=vector_env, device="cpu")
    curve_path = output_dir / f"{algorithm}_budget_recovery_curve.jsonl"
    curve_path.unlink(missing_ok=True)
    callback = DevelopmentEvalCallback(
        env_factory,
        experiment["eval_seeds"],
        int(experiment["eval_interval"]),
        curve_path,
    )
    model.learn(
        total_timesteps=additional_steps,
        callback=callback,
        reset_num_timesteps=False,
    )
    result = evaluate_model(model, env_factory, experiment["eval_seeds"])
    model.save(output_dir / f"{algorithm}_budget_recovery")
    (output_dir / f"{algorithm}_budget_recovery_final.json").write_text(
        json.dumps(result.as_dict(), indent=2), encoding="utf-8"
    )
    vector_env.close()
    return result
