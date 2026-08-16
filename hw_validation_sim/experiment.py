"""Trains the Q-learning agent across many simulated "units", then evaluates all three
agents fresh on held-out units with the same per-episode step budget.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agents import (
    EpisodeMetrics,
    HillClimbAgent,
    LinearFAQAgent,
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


def train_function_approx_agent(
    levels: int,
    train_episodes: int,
    max_steps: int,
    noise_std: float,
    unit_variation: float,
    seed: int,
    n_centers_per_dim: int = 6,
    sigma_scale: float = 1.0,
) -> LinearFAQAgent:
    agent = LinearFAQAgent(
        levels=levels,
        rng=np.random.default_rng(seed + 2),
        n_centers_per_dim=n_centers_per_dim,
        sigma_scale=sigma_scale,
    )
    for i in range(train_episodes):
        env = CalibrationEnv(
            levels=levels,
            max_steps=max_steps,
            noise_std=noise_std,
            unit_variation=unit_variation,
            seed=seed + i,
        )
        run_episode(env, agent, learn=True)
    return agent


def run_resolution_comparison(
    levels: int,
    train_episodes: int,
    eval_episodes: int,
    max_steps: int,
    noise_std: float,
    unit_variation: float,
    seed: int,
) -> list[ExperimentResult]:
    """Train tabular Q-learning and linear-FA Q-learning on the *same* fixed training
    budget at the given grid resolution, then evaluate both fresh on held-out units.

    At high `levels`, the tabular Q-table has more cells than the training budget can
    give even one visit to, so it can only have learned about a fraction of the space.
    The FA agent's RBF features let one update generalize to nearby, unvisited cells,
    which is the whole motivation for reaching for function approximation here.
    """
    tabular = train_qlearning_agent(
        levels, train_episodes, max_steps, noise_std, unit_variation, seed
    )
    tabular.eval_mode()
    fa = train_function_approx_agent(
        levels, train_episodes, max_steps, noise_std, unit_variation, seed
    )
    fa.eval_mode()

    results = {"q_learning_tabular": [], "q_learning_linear_fa": []}
    for i in range(eval_episodes):
        env_seed = seed + 20_000 + i  # disjoint from both training and other eval seeds

        env = CalibrationEnv(levels, max_steps, noise_std, unit_variation, seed=env_seed)
        results["q_learning_tabular"].append(run_episode_metrics(env, tabular, learn=False))

        env = CalibrationEnv(levels, max_steps, noise_std, unit_variation, seed=env_seed)
        results["q_learning_linear_fa"].append(run_episode_metrics(env, fa, learn=False))

    return [ExperimentResult(name, rewards) for name, rewards in results.items()]


@dataclass
class CenterSweepPoint:
    """One RBF center-count's results, aggregated across independent training seeds
    (not a single-seed point estimate) at a fixed grid resolution and training
    budget."""

    n_centers_per_dim: int
    seed_means: list[float]

    @property
    def mean(self) -> float:
        return float(np.mean(self.seed_means))

    @property
    def std(self) -> float:
        """Sample std (ddof=1) of the per-seed mean rewards -- how much the center
        count's performance itself varies from one training run to the next."""
        return float(np.std(self.seed_means, ddof=1)) if len(self.seed_means) > 1 else 0.0

    @property
    def ci95_halfwidth(self) -> float:
        """Half-width of a normal-approximation 95% CI on the across-seed mean.
        NaN with fewer than 2 seeds, since a CI is meaningless without variance."""
        n = len(self.seed_means)
        if n < 2:
            return float("nan")
        return float(1.96 * self.std / np.sqrt(n))


def run_center_sweep(
    levels: int,
    center_counts: list[int],
    n_seeds: int,
    train_episodes: int,
    eval_episodes: int,
    max_steps: int,
    noise_std: float,
    unit_variation: float,
    base_seed: int,
) -> list[CenterSweepPoint]:
    """At a fixed grid resolution and training budget, train the linear-FA agent at
    each candidate RBF center count across `n_seeds` independent training seeds, and
    report the distribution of each center count's per-seed mean held-out reward --
    not a single-seed point estimate. The same `n_seeds` training seeds are reused
    across every center count (a paired comparison), so differences between center
    counts aren't confounded by which count happened to get luckier training draws.
    """
    points = []
    for n_centers in center_counts:
        seed_means = []
        for s in range(n_seeds):
            seed = base_seed + s * 1000  # well clear of any one seed's own episode range
            agent = train_function_approx_agent(
                levels,
                train_episodes,
                max_steps,
                noise_std,
                unit_variation,
                seed,
                n_centers_per_dim=n_centers,
            )
            agent.eval_mode()
            rewards = []
            for i in range(eval_episodes):
                env = CalibrationEnv(levels, max_steps, noise_std, unit_variation, seed=seed + 20_000 + i)
                rewards.append(run_episode_metrics(env, agent, learn=False).best_true_reward)
            seed_means.append(float(np.mean(rewards)))
        points.append(CenterSweepPoint(n_centers, seed_means))
    return points


@dataclass
class SigmaSweepPoint:
    """One RBF width's results, aggregated across independent training seeds (not a
    single-seed point estimate), at a fixed grid resolution, center count, and
    training budget."""

    sigma_scale: float
    seed_means: list[float]

    @property
    def mean(self) -> float:
        return float(np.mean(self.seed_means))

    @property
    def std(self) -> float:
        """Sample std (ddof=1) of the per-seed mean rewards."""
        return float(np.std(self.seed_means, ddof=1)) if len(self.seed_means) > 1 else 0.0

    @property
    def ci95_halfwidth(self) -> float:
        """Half-width of a normal-approximation 95% CI on the across-seed mean.
        NaN with fewer than 2 seeds, since a CI is meaningless without variance."""
        n = len(self.seed_means)
        if n < 2:
            return float("nan")
        return float(1.96 * self.std / np.sqrt(n))


def run_sigma_sweep(
    levels: int,
    n_centers_per_dim: int,
    sigma_scales: list[float],
    n_seeds: int,
    train_episodes: int,
    eval_episodes: int,
    max_steps: int,
    noise_std: float,
    unit_variation: float,
    base_seed: int,
) -> list[SigmaSweepPoint]:
    """At a fixed grid resolution, center count, and training budget, train the
    linear-FA agent at each candidate RBF width (`sigma_scale`, a multiplier on the
    default one-grid-spacing-between-centers width) across `n_seeds` independent
    training seeds, and report the distribution of each width's per-seed mean
    held-out reward. Reuses the same `n_seeds` training seeds across every width (a
    paired comparison, mirroring `run_center_sweep`), so differences between widths
    aren't confounded by which one happened to get luckier training draws.
    """
    points = []
    for sigma_scale in sigma_scales:
        seed_means = []
        for s in range(n_seeds):
            seed = base_seed + s * 1000  # same seed stream convention as run_center_sweep
            agent = train_function_approx_agent(
                levels,
                train_episodes,
                max_steps,
                noise_std,
                unit_variation,
                seed,
                n_centers_per_dim=n_centers_per_dim,
                sigma_scale=sigma_scale,
            )
            agent.eval_mode()
            rewards = []
            for i in range(eval_episodes):
                env = CalibrationEnv(levels, max_steps, noise_std, unit_variation, seed=seed + 20_000 + i)
                rewards.append(run_episode_metrics(env, agent, learn=False).best_true_reward)
            seed_means.append(float(np.mean(rewards)))
        points.append(SigmaSweepPoint(sigma_scale, seed_means))
    return points


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
            "## Why these metrics, not just reward",
            "",
            "Every measurement in a real bring-up or production calibration loop consumes",
            "instrument time, operator time, thermal settling time, or test-station capacity.",
            "The useful result is not reward alone; it is reward per measurement under safety",
            "constraints.",
            "",
        ]
    )
    return "\n".join(lines)
