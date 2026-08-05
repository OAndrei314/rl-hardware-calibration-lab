"""Trains the Q-learning agent across many simulated "units", then evaluates all three
agents fresh on held-out units with the same per-episode step budget.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agents import HillClimbAgent, QLearningAgent, RandomSearchAgent, run_episode
from .env import CalibrationEnv


@dataclass
class ExperimentResult:
    agent_name: str
    best_true_rewards: list[float]  # one per eval episode

    @property
    def mean(self) -> float:
        return float(np.mean(self.best_true_rewards))

    @property
    def std(self) -> float:
        return float(np.std(self.best_true_rewards))


def train_qlearning_agent(
    levels: int,
    train_episodes: int,
    max_steps: int,
    noise_std: float,
    unit_variation: float,
    seed: int,
) -> QLearningAgent:
    env_rng_seed = seed
    agent = QLearningAgent(levels=levels, rng=np.random.default_rng(seed + 1))
    for i in range(train_episodes):
        env = CalibrationEnv(
            levels=levels,
            max_steps=max_steps,
            noise_std=noise_std,
            unit_variation=unit_variation,
            seed=env_rng_seed + i,
        )
        run_episode(env, agent, learn=True)
    return agent


def evaluate_agents(
    trained_qlearning: QLearningAgent,
    levels: int,
    eval_episodes: int,
    max_steps: int,
    noise_std: float,
    unit_variation: float,
    seed: int,
) -> list[ExperimentResult]:
    trained_qlearning.eval_mode()

    results = {
        "random_search": [],
        "hill_climb": [],
        "q_learning": [],
    }

    for i in range(eval_episodes):
        env_seed = seed + 10_000 + i  # disjoint from training seeds

        env = CalibrationEnv(levels, max_steps, noise_std, unit_variation, seed=env_seed)
        agent = RandomSearchAgent(levels, np.random.default_rng(env_seed))
        results["random_search"].append(run_episode(env, agent, learn=False))

        env = CalibrationEnv(levels, max_steps, noise_std, unit_variation, seed=env_seed)
        agent = HillClimbAgent(levels, np.random.default_rng(env_seed))
        results["hill_climb"].append(run_episode(env, agent, learn=False))

        env = CalibrationEnv(levels, max_steps, noise_std, unit_variation, seed=env_seed)
        results["q_learning"].append(run_episode(env, trained_qlearning, learn=False))

    return [ExperimentResult(name, rewards) for name, rewards in results.items()]
