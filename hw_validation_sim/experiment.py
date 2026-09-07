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
    run_episode_shaped,
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


def train_threshold_seeking_agent(
    levels: int,
    train_episodes: int,
    max_steps: int,
    noise_std: float,
    unit_variation: float,
    seed: int,
    spec_threshold: float = 0.85,
    step_cost: float = 0.02,
    bonus: float = 1.0,
    dense_scale: float = 0.0,
) -> QLearningAgent:
    """Trains a tabular Q-learning agent with `run_episode_shaped` instead of
    `run_episode` -- same architecture and hyperparameters as `train_qlearning_agent`,
    but the training signal directly rewards reaching `spec_threshold` in as few
    measurements as possible, rather than maximizing raw measurement reward.

    `dense_scale=0.0` (the default) reproduces the original pure-sparse shaping
    exactly; see `run_episode_shaped` and `run_dense_shaping_sweep` for the
    per-step-proportional-bonus variant this parameter enables.
    """
    agent = QLearningAgent(levels=levels, rng=np.random.default_rng(seed + 1))
    for i in range(train_episodes):
        env = CalibrationEnv(
            levels=levels,
            max_steps=max_steps,
            noise_std=noise_std,
            unit_variation=unit_variation,
            seed=seed + i,
        )
        run_episode_shaped(
            env, agent, spec_threshold=spec_threshold, step_cost=step_cost,
            bonus=bonus, dense_scale=dense_scale, learn=True,
        )
    return agent


def run_threshold_shaping_comparison(
    levels: int,
    train_episodes: int,
    eval_episodes: int,
    max_steps: int,
    noise_std: float,
    unit_variation: float,
    seed: int,
    spec_threshold: float = 0.85,
    step_cost: float = 0.02,
    bonus: float = 1.0,
) -> list[ExperimentResult]:
    """Trains the standard reward-maximizing Q-learning agent and the
    threshold-seeking one on an identical fixed budget (same levels, episodes, noise,
    unit variation, seed), then evaluates BOTH fresh on the same held-out units,
    scored against the same fixed, absolute `spec_threshold` used during shaped
    training. Evaluation uses `run_episode_shaped(..., learn=False)` rather than
    `run_episode_metrics`'s `threshold_fraction`, since that parameter is relative to
    each unit's own hidden optimum, not an absolute spec bar -- the two happen to be
    numerically close in this environment (`optimum_true_reward()` sits within about
    4% of 1.0 for every unit), but conflating them would silently give the wrong
    metric on a landscape where that coincidence didn't hold.

    This directly answers the open question the reward-maximizing baseline could not:
    does training toward "minimize measurements to reach spec" actually reach spec
    faster than training toward "maximize the reward signal," or does the correlation
    between the two objectives already do that job well enough on its own?
    """
    reward_max = train_qlearning_agent(
        levels, train_episodes, max_steps, noise_std, unit_variation, seed
    )
    reward_max.eval_mode()
    threshold_seeking = train_threshold_seeking_agent(
        levels, train_episodes, max_steps, noise_std, unit_variation, seed,
        spec_threshold=spec_threshold, step_cost=step_cost, bonus=bonus,
    )
    threshold_seeking.eval_mode()

    results = {"q_learning_reward_max": [], "q_learning_threshold_seeking": []}
    for i in range(eval_episodes):
        env_seed = seed + 30_000 + i  # disjoint from every other seed range in this module

        env = CalibrationEnv(levels, max_steps, noise_std, unit_variation, seed=env_seed)
        results["q_learning_reward_max"].append(
            run_episode_shaped(env, reward_max, spec_threshold=spec_threshold, learn=False)
        )

        env = CalibrationEnv(levels, max_steps, noise_std, unit_variation, seed=env_seed)
        results["q_learning_threshold_seeking"].append(
            run_episode_shaped(env, threshold_seeking, spec_threshold=spec_threshold, learn=False)
        )

    return [ExperimentResult(name, rewards) for name, rewards in results.items()]


@dataclass
class DenseShapingPoint:
    """One `dense_scale` value's results for the threshold-seeking agent, aggregated
    across independent training seeds (not a single-seed point estimate) -- same
    paired-seed design as `run_center_sweep`/`run_sigma_sweep`/`run_joint_sweep`.

    Unlike those sweeps, this one tracks **success rate** per seed alongside mean
    reward: the open question `run_threshold_shaping_comparison` raised is whether a
    denser reward closes the *success-rate* gap to the reward-maximizing baseline,
    and mean reward alone can look deceptively close while success rate stays far
    apart (a big true-reward improvement on the units that still fail to reach spec
    is not the same thing as more units reaching spec).

    `dense_scale=None` marks the reward-maximizing reference row -- trained with the
    ordinary `train_qlearning_agent`/`run_episode` path, not the shaped runner --
    included so the table this feeds has the baseline it's being compared against
    computed on the exact same seed stream, not pasted in from a different run.
    """

    label: str
    dense_scale: float | None
    seed_success_rates: list[float]
    seed_mean_rewards: list[float]

    @property
    def mean_success_rate(self) -> float:
        return float(np.mean(self.seed_success_rates))

    @property
    def success_std(self) -> float:
        """Sample std (ddof=1) of the per-seed success rates."""
        return float(np.std(self.seed_success_rates, ddof=1)) if len(self.seed_success_rates) > 1 else 0.0

    @property
    def success_ci95_halfwidth(self) -> float:
        """Half-width of a normal-approximation 95% CI on the across-seed mean
        success rate. NaN with fewer than 2 seeds, since a CI is meaningless
        without variance."""
        n = len(self.seed_success_rates)
        if n < 2:
            return float("nan")
        return float(1.96 * self.success_std / np.sqrt(n))

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.seed_mean_rewards))


def _threshold_seeking_seed_stats(
    agent: QLearningAgent,
    levels: int,
    eval_episodes: int,
    max_steps: int,
    noise_std: float,
    unit_variation: float,
    seed: int,
    spec_threshold: float,
) -> tuple[float, float]:
    """Evaluate one trained agent on `eval_episodes` held-out units (the same
    `seed + 20_000 + i` convention every other sweep in this module uses) and
    return (success_rate, mean_best_true_reward)."""
    agent.eval_mode()
    metrics = []
    for i in range(eval_episodes):
        env = CalibrationEnv(levels, max_steps, noise_std, unit_variation, seed=seed + 20_000 + i)
        metrics.append(
            run_episode_shaped(env, agent, spec_threshold=spec_threshold, learn=False)
        )
    success_rate = float(np.mean([m.steps_to_threshold is not None for m in metrics]))
    mean_reward = float(np.mean([m.best_true_reward for m in metrics]))
    return success_rate, mean_reward


def run_dense_shaping_sweep(
    levels: int,
    dense_scales: list[float],
    n_seeds: int,
    train_episodes: int,
    eval_episodes: int,
    max_steps: int,
    noise_std: float,
    unit_variation: float,
    base_seed: int,
    spec_threshold: float = 0.85,
    step_cost: float = 0.02,
    bonus: float = 1.0,
    include_reward_max: bool = True,
) -> list[DenseShapingPoint]:
    """At a fixed grid resolution and training budget, train the threshold-seeking
    agent at each candidate `dense_scale` (a per-step reward term proportional to the
    noisy measurement, on top of the existing sparse spec-crossing bonus) across
    `n_seeds` independent training seeds -- same `base_seed + s * 1000` seed stream
    convention as `run_center_sweep`/`run_sigma_sweep`, so this is a paired
    comparison across `dense_scale` values, not confounded by which value happened
    to draw luckier training units.

    `dense_scale=0.0` reproduces `run_threshold_shaping_comparison`'s pure-sparse
    threshold-seeking agent exactly (same seeds, same shaping, same eval convention)
    -- a built-in sanity check that this is a genuinely new sweep, not a
    reimplementation that happens to disagree with the number already in the README.

    When `include_reward_max` (the default), a `dense_scale=None` reference row is
    prepended: the ordinary reward-maximizing agent, trained and evaluated on the
    exact same seed stream, so the sweep table includes the baseline it's actually
    being compared against rather than requiring the reader to cross-reference a
    different run.
    """
    points = []
    if include_reward_max:
        seed_success, seed_reward = [], []
        for s in range(n_seeds):
            seed = base_seed + s * 1000
            agent = train_qlearning_agent(
                levels, train_episodes, max_steps, noise_std, unit_variation, seed
            )
            success_rate, mean_reward = _threshold_seeking_seed_stats(
                agent, levels, eval_episodes, max_steps, noise_std, unit_variation,
                seed, spec_threshold,
            )
            seed_success.append(success_rate)
            seed_reward.append(mean_reward)
        points.append(DenseShapingPoint("reward_max", None, seed_success, seed_reward))

    for dense_scale in dense_scales:
        seed_success, seed_reward = [], []
        for s in range(n_seeds):
            seed = base_seed + s * 1000
            agent = train_threshold_seeking_agent(
                levels, train_episodes, max_steps, noise_std, unit_variation, seed,
                spec_threshold=spec_threshold, step_cost=step_cost, bonus=bonus,
                dense_scale=dense_scale,
            )
            success_rate, mean_reward = _threshold_seeking_seed_stats(
                agent, levels, eval_episodes, max_steps, noise_std, unit_variation,
                seed, spec_threshold,
            )
            seed_success.append(success_rate)
            seed_reward.append(mean_reward)
        points.append(
            DenseShapingPoint(f"dense_scale={dense_scale}", dense_scale, seed_success, seed_reward)
        )
    return points


def train_threshold_seeking_fa_agent(
    levels: int,
    train_episodes: int,
    max_steps: int,
    noise_std: float,
    unit_variation: float,
    seed: int,
    spec_threshold: float = 0.85,
    step_cost: float = 0.02,
    bonus: float = 1.0,
    dense_scale: float = 0.0,
    n_centers_per_dim: int = 14,
    sigma_scale: float = 0.6,
) -> LinearFAQAgent:
    """Same training loop as `train_threshold_seeking_agent`, but for `LinearFAQAgent`
    instead of tabular `QLearningAgent`. Exists so
    `run_sparse_threshold_sample_efficiency_comparison` can ask whether RBF
    generalization changes the severe sample penalty pure-sparse shaping
    (`dense_scale=0.0`) pays a tabular Q-table, or whether that penalty is a property
    of the sparse reward itself rather than of the tabular representation.
    """
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
        run_episode_shaped(
            env, agent, spec_threshold=spec_threshold, step_cost=step_cost,
            bonus=bonus, dense_scale=dense_scale, learn=True,
        )
    return agent


@dataclass
class SparseThresholdBudgetPoint:
    """One (agent architecture, training-episode budget) combination's results under
    PURE SPARSE threshold-seeking shaping (`dense_scale=0.0` fixed throughout),
    aggregated across independent training seeds -- not a single-seed point estimate.

    Answers the open question this README's "Status / next steps" section raised
    after the dense-shaping sweep: does `LinearFAQAgent`'s cross-cell generalization
    let pure-sparse shaping become viable at a much smaller training-episode budget
    than the tabular agent needs, now that `dense_scale` has already closed the same
    gap a different way?
    """

    agent_type: str  # "tabular" or "linear_fa"
    train_episodes: int
    seed_success_rates: list[float]
    seed_mean_rewards: list[float]

    @property
    def mean_success_rate(self) -> float:
        return float(np.mean(self.seed_success_rates))

    @property
    def success_std(self) -> float:
        """Sample std (ddof=1) of the per-seed success rates."""
        return float(np.std(self.seed_success_rates, ddof=1)) if len(self.seed_success_rates) > 1 else 0.0

    @property
    def success_ci95_halfwidth(self) -> float:
        """Half-width of a normal-approximation 95% CI on the across-seed mean
        success rate. NaN with fewer than 2 seeds, since a CI is meaningless
        without variance."""
        n = len(self.seed_success_rates)
        if n < 2:
            return float("nan")
        return float(1.96 * self.success_std / np.sqrt(n))

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.seed_mean_rewards))


def run_sparse_threshold_sample_efficiency_comparison(
    levels: int,
    train_episode_budgets: list[int],
    n_seeds: int,
    eval_episodes: int,
    max_steps: int,
    noise_std: float,
    unit_variation: float,
    base_seed: int,
    spec_threshold: float = 0.85,
    step_cost: float = 0.02,
    bonus: float = 1.0,
    n_centers_per_dim: int = 14,
    sigma_scale: float = 0.6,
) -> list[SparseThresholdBudgetPoint]:
    """At each candidate training-episode budget, train BOTH a tabular and a
    linear-FA threshold-seeking agent under identical pure-sparse shaping
    (`dense_scale=0.0` -- the exact regime `run_threshold_shaping_comparison` found
    the tabular agent losing badly at), each averaged over `n_seeds` independent
    training seeds using the same `base_seed + s * 1000` seed-stream convention as
    every other sweep in this module, so both architectures see identical
    training/eval units at a given seed and budget -- a paired comparison, not
    confounded by which one got a luckier draw.

    Default `n_centers_per_dim=14, sigma_scale=0.6` reuse this README's own
    joint-sweep result for the best-measured RBF configuration, rather than
    introducing a new hand-picked value here.

    Returns points in `(budget, agent_type)` order for every budget in
    `train_episode_budgets`, `"tabular"` before `"linear_fa"` within each budget.
    """
    points = []
    for train_episodes in train_episode_budgets:
        for agent_type in ("tabular", "linear_fa"):
            seed_success, seed_reward = [], []
            for s in range(n_seeds):
                seed = base_seed + s * 1000
                if agent_type == "tabular":
                    agent = train_threshold_seeking_agent(
                        levels, train_episodes, max_steps, noise_std, unit_variation,
                        seed, spec_threshold=spec_threshold, step_cost=step_cost, bonus=bonus,
                    )
                else:
                    agent = train_threshold_seeking_fa_agent(
                        levels, train_episodes, max_steps, noise_std, unit_variation,
                        seed, spec_threshold=spec_threshold, step_cost=step_cost, bonus=bonus,
                        n_centers_per_dim=n_centers_per_dim, sigma_scale=sigma_scale,
                    )
                success_rate, mean_reward = _threshold_seeking_seed_stats(
                    agent, levels, eval_episodes, max_steps, noise_std, unit_variation,
                    seed, spec_threshold,
                )
                seed_success.append(success_rate)
                seed_reward.append(mean_reward)
            points.append(
                SparseThresholdBudgetPoint(agent_type, train_episodes, seed_success, seed_reward)
            )
    return points


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


@dataclass
class JointSweepPoint:
    """One (center count, sigma_scale) combination's results, aggregated across
    independent training seeds, at a fixed grid resolution and training budget.

    The 1D center-count and sigma-scale sweeps each optimize one hyperparameter at
    the other's already-tuned value, which cannot see an interaction between them --
    e.g. whether the best width depends on how many centers are in play. This joint
    sweep trains at every combination in the given grid instead."""

    n_centers_per_dim: int
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


def run_joint_sweep(
    levels: int,
    center_counts: list[int],
    sigma_scales: list[float],
    n_seeds: int,
    train_episodes: int,
    eval_episodes: int,
    max_steps: int,
    noise_std: float,
    unit_variation: float,
    base_seed: int,
) -> list[JointSweepPoint]:
    """At a fixed grid resolution and training budget, train the linear-FA agent at
    every (center count, sigma_scale) combination in the given grid, each across
    `n_seeds` independent training seeds. Returns one `JointSweepPoint` per
    combination, in row-major (center count, then sigma_scale) order.

    Reuses the same seed stream convention as `run_center_sweep` / `run_sigma_sweep`
    (`base_seed + s * 1000` for seed index `s`), so a given seed trains on the exact
    same sequence of simulated units regardless of which combination it's paired
    with -- a paired comparison across the whole grid, not just within one sweep
    axis.
    """
    points = []
    for n_centers in center_counts:
        for sigma_scale in sigma_scales:
            seed_means = []
            for s in range(n_seeds):
                seed = base_seed + s * 1000
                agent = train_function_approx_agent(
                    levels,
                    train_episodes,
                    max_steps,
                    noise_std,
                    unit_variation,
                    seed,
                    n_centers_per_dim=n_centers,
                    sigma_scale=sigma_scale,
                )
                agent.eval_mode()
                rewards = []
                for i in range(eval_episodes):
                    env = CalibrationEnv(levels, max_steps, noise_std, unit_variation, seed=seed + 20_000 + i)
                    rewards.append(run_episode_metrics(env, agent, learn=False).best_true_reward)
                seed_means.append(float(np.mean(rewards)))
            points.append(JointSweepPoint(n_centers, sigma_scale, seed_means))
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
