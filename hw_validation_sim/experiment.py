"""Trains the Q-learning agent across many simulated "units", then evaluates all three
agents fresh on held-out units with the same per-episode step budget.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agents import (
    EpisodeMetrics,
    HillClimbAgent,
    QLearningAgent,
    RandomSearchAgent,
    run_episode,
    run_episode_metrics,
)
from .env import CalibrationEnv


@dataclass
class ExperimentResult:
    agent_name: str
    episodes: list[EpisodeMetrics]

    @property
    def best_true_rewards(self) -> list[float]:
        return [episode.best_true_reward for episode in self.episodes]

    @property
    def mean(self) -> float:
        return float(np.mean(self.best_true_rewards))

    @property
    def std(self) -> float:
        return float(np.std(self.best_true_rewards))

    @property
    def success_rate(self) -> float:
        return float(np.mean([episode.steps_to_threshold is not None for episode in self.episodes]))

    @property
    def mean_steps_to_threshold(self) -> float:
        reached = [
            episode.steps_to_threshold
            for episode in self.episodes
            if episode.steps_to_threshold is not None
        ]
        return float(np.mean(reached)) if reached else float("nan")

    @property
    def mean_control_effort(self) -> float:
        return float(np.mean([episode.control_effort for episode in self.episodes]))

    @property
    def mean_boundary_hits(self) -> float:
        return float(np.mean([episode.boundary_hits for episode in self.episodes]))


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
        results["random_search"].append(run_episode_metrics(env, agent, learn=False))

        env = CalibrationEnv(levels, max_steps, noise_std, unit_variation, seed=env_seed)
        agent = HillClimbAgent(levels, np.random.default_rng(env_seed))
        results["hill_climb"].append(run_episode_metrics(env, agent, learn=False))

        env = CalibrationEnv(levels, max_steps, noise_std, unit_variation, seed=env_seed)
        results["q_learning"].append(run_episode_metrics(env, trained_qlearning, learn=False))

    return [ExperimentResult(name, rewards) for name, rewards in results.items()]


def render_markdown_report(results: list[ExperimentResult]) -> str:
    lines = [
        "# RL Hardware Calibration Report",
        "",
        "## Research Question",
        "",
        "Can a learned calibration policy exploit shared structure across simulated units",
        "to reduce validation search time versus per-unit random or hill-climb baselines?",
        "",
        "## Results",
        "",
        "| strategy | mean best reward | success rate | mean steps to threshold | mean control effort | boundary hits |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.agent_name} | "
            f"{result.mean:.3f} | "
            f"{result.success_rate:.1%} | "
            f"{result.mean_steps_to_threshold:.1f} | "
            f"{result.mean_control_effort:.1f} | "
            f"{result.mean_boundary_hits:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Money Signal",
            "",
            "Every measurement in a real bring-up or production calibration loop consumes",
            "instrument time, operator time, thermal settling time, or test-station capacity.",
            "The useful result is not reward alone; it is reward per measurement under safety",
            "constraints.",
            "",
        ]
    )
    return "\n".join(lines)
