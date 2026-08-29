"""Four search strategies compared on the same calibration budget:

- RandomSearchAgent: uniform random moves. The floor.
- HillClimbAgent: measure, nudge in a random direction, keep the move if the noisy
  reading improved, otherwise revert. This is close to how a lot of real manual/
  rule-based calibration procedures actually work.
- QLearningAgent: tabular Q-learning, trained across many simulated "units" so it can
  learn the general shape of the landscape (including the decoy local optimum) instead
  of re-discovering it from scratch on every new unit.
- LinearFAQAgent: Q-learning with linear function approximation over radial basis
  function (RBF) features, for when the calibration grid is too fine for a tabular
  Q-table to get enough visits per cell within a fixed measurement budget (see
  `experiment.run_resolution_comparison`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import ACTIONS, CalibrationEnv


class RandomSearchAgent:
    def __init__(self, levels: int, rng: np.random.Generator):
        self.levels = levels
        self.rng = rng

    def act(self, obs: tuple[int, int]) -> int:
        return int(self.rng.integers(0, len(ACTIONS)))

    def observe(self, obs, action, reward, next_obs, done) -> None:
        pass  # stateless


class HillClimbAgent:
    """Stochastic hill climbing: propose a random move, keep it if the noisy reading
    improved on the best reading seen so far, otherwise step back. Only uses the 4
    movement actions (never "stay") -- restarts a fresh proposal after every accept
    or revert."""

    _OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}

    def __init__(self, levels: int, rng: np.random.Generator):
        self.levels = levels
        self.rng = rng
        self.best_reward_so_far: float | None = None
        self.state = "propose"  # or "revert"
        self.trial_action: int | None = None

    def act(self, obs: tuple[int, int]) -> int:
        if self.state == "revert":
            return self._OPPOSITE[self.trial_action]
        action = int(self.rng.integers(0, 4))
        self.trial_action = action
        return action

    def observe(self, obs, action, reward, next_obs, done) -> None:
        if self.state == "revert":
            self.state = "propose"  # back at baseline, ready for a new proposal
            return
        if self.best_reward_so_far is None or reward > self.best_reward_so_far:
            self.best_reward_so_far = reward  # accept the move
            self.state = "propose"
        else:
            self.state = "revert"  # undo it next step


class QLearningAgent:
    def __init__(
        self,
        levels: int,
        rng: np.random.Generator,
        alpha: float = 0.2,
        gamma: float = 0.9,
        epsilon: float = 0.15,
    ):
        self.levels = levels
        self.rng = rng
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = np.zeros((levels, levels, len(ACTIONS)))
        self.training = True

    def act(self, obs: tuple[int, int]) -> int:
        if self.training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, len(ACTIONS)))
        x, y = obs
        return int(np.argmax(self.q[x, y]))

    def observe(self, obs, action, reward, next_obs, done) -> None:
        if not self.training:
            return
        x, y = obs
        nx, ny = next_obs
        target = reward + (0.0 if done else self.gamma * np.max(self.q[nx, ny]))
        self.q[x, y, action] += self.alpha * (target - self.q[x, y, action])

    def eval_mode(self) -> None:
        self.training = False


class LinearFAQAgent:
    """Q-learning with linear function approximation: Q(s, a) = w[a] . phi(s), where
    phi(s) is a vector of radial basis function (RBF) responses to a fixed grid of
    centers spanning the calibration space, plus a bias term.

    Unlike the tabular agent, this does not need to visit every (x, y) cell to learn
    something useful about it -- an update at one point moves the weights for every
    RBF center whose bump overlaps that point, so nearby, never-visited states inherit
    part of the update. That is the point of function approximation, and exactly what
    a tabular Q-table cannot do. We use a linear model (rather than a small neural
    net, as originally sketched in this repo's next-steps note) because it is a
    semi-gradient update with a closed-form feature map: no optimizer, no
    hyperparameter search over network shape, and the weight vector is directly
    inspectable -- which matters for a calibration controller you'd actually want to
    validate before trusting it near hardware safety limits.
    """

    def __init__(
        self,
        levels: int,
        rng: np.random.Generator,
        n_centers_per_dim: int = 6,
        alpha: float = 0.05,
        gamma: float = 0.9,
        epsilon: float = 0.15,
        sigma_scale: float = 1.0,
    ):
        self.levels = levels
        self.rng = rng
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = True

        centers_1d = np.linspace(0, levels - 1, n_centers_per_dim)
        cx, cy = np.meshgrid(centers_1d, centers_1d)
        self.centers = np.stack([cx.ravel(), cy.ravel()], axis=1)  # (n_centers, 2)
        # `sigma_scale` scales the RBF width away from the hand-picked default of one
        # grid spacing between centers (sigma_scale=1.0 reproduces prior behavior).
        # Narrower (< 1) bumps overlap less between centers; wider (> 1) bumps
        # generalize a training update across more of the grid per visit.
        self.sigma = (levels / n_centers_per_dim) * sigma_scale
        self.n_features = len(self.centers) + 1  # + bias
        self.weights = np.zeros((len(ACTIONS), self.n_features))

    def _features(self, obs: tuple[int, int]) -> np.ndarray:
        xy = np.array(obs, dtype=float)
        d2 = np.sum((self.centers - xy) ** 2, axis=1)
        rbf = np.exp(-d2 / (2 * self.sigma**2))
        return np.concatenate([rbf, [1.0]])

    def act(self, obs: tuple[int, int]) -> int:
        if self.training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, len(ACTIONS)))
        phi = self._features(obs)
        return int(np.argmax(self.weights @ phi))

    def observe(self, obs, action, reward, next_obs, done) -> None:
        if not self.training:
            return
        phi = self._features(obs)
        q_sa = float(self.weights[action] @ phi)
        if done:
            target = reward
        else:
            next_phi = self._features(next_obs)
            target = reward + self.gamma * float(np.max(self.weights @ next_phi))
        td_error = target - q_sa
        self.weights[action] += self.alpha * td_error * phi

    def eval_mode(self) -> None:
        self.training = False


@dataclass(frozen=True)
class EpisodeMetrics:
    best_true_reward: float
    steps_to_threshold: int | None
    control_effort: int
    boundary_hits: int


def run_episode(env: CalibrationEnv, agent, learn: bool = True) -> float:
    """Runs one episode, returns the best TRUE reward observed (agent only sees noisy
    rewards during the episode -- `true_reward` here is for evaluation/reporting only)."""
    return run_episode_metrics(env, agent, learn=learn).best_true_reward


def run_episode_metrics(
    env: CalibrationEnv,
    agent,
    learn: bool = True,
    threshold_fraction: float = 0.85,
) -> EpisodeMetrics:
    """Run one calibration episode and return engineering metrics.

    `steps_to_threshold` is the first measurement step where the true reward reaches a
    fraction of this episode's hidden optimum. `boundary_hits` is a simple safety proxy:
    how often the policy drives calibration parameters to the edge of the valid range.
    """
    obs = env.reset()
    best_true = env.true_reward_at(obs)
    threshold = env.optimum_true_reward() * threshold_fraction
    steps_to_threshold = 0 if best_true >= threshold else None
    control_effort = 0
    boundary_hits = 0
    done = False
    while not done:
        action = agent.act(obs)
        result = env.step(action)
        control_effort += 0 if action == ACTIONS.index("stay") else 1
        x, y = result.obs
        if x in (0, env.levels - 1) or y in (0, env.levels - 1):
            boundary_hits += 1
        if learn:
            agent.observe(obs, action, result.noisy_reward, result.obs, result.done)
        obs = result.obs
        best_true = max(best_true, result.true_reward)
        if steps_to_threshold is None and best_true >= threshold:
            steps_to_threshold = env.steps
        done = result.done
    return EpisodeMetrics(
        best_true_reward=best_true,
        steps_to_threshold=steps_to_threshold,
        control_effort=control_effort,
        boundary_hits=boundary_hits,
    )


def run_episode_shaped(
    env: CalibrationEnv,
    agent,
    spec_threshold: float,
    step_cost: float = 0.02,
    bonus: float = 1.0,
    dense_scale: float = 0.0,
    learn: bool = True,
) -> EpisodeMetrics:
    """Like `run_episode_metrics`, but the reward handed to `agent.observe()` during
    training is SHAPED to directly reward reaching an absolute pass/fail spec bar
    quickly, instead of shaped around the magnitude of the (noisy) measurement itself.

    `q_learning`/`LinearFAQAgent` trained via `run_episode`/`run_episode_metrics`
    maximize cumulative *measurement* reward, which is a proxy: it happens to
    correlate with reaching an acceptable calibration fast, but nothing in that
    training signal actually says "stop wasting measurements once you're good
    enough." This runner instead gives the agent a small constant `-step_cost` per
    measurement (every measurement consumes tester/thermal/operator time, whether or
    not it moves the needle) plus a one-time `+bonus` the first step its *noisy*
    reading crosses `spec_threshold` (the agent only ever sees noisy readings in
    real life, so shaping must be based on what it can observe, not the hidden true
    reward). Because the episode always runs the full `max_steps` regardless of
    when the bar is crossed, the per-step cost is a wash between two training runs
    of equal length -- the only real lever left to increase discounted return is
    reaching the bonus sooner, which is exactly "minimize steps-to-threshold."

    `dense_scale` (0.0 by default, reproducing the original pure-sparse shaping
    exactly) adds `dense_scale * noisy_reward` to every step's shaped reward. This is
    still a sparse-*bonus* runner at its core -- the spec-crossing bonus and its
    "stop wasting measurements" incentive are unchanged -- but a nonzero
    `dense_scale` also gives the agent a per-step signal correlated with how close it
    is to the optimum, the same dense signal the reward-maximizing baseline gets "for
    free," to test whether that's what was missing rather than the direct spec-bar
    framing itself.

    `spec_threshold` is a fixed absolute bar (not `threshold_fraction * this
    episode's hidden optimum`, unlike `run_episode_metrics`) -- realistic for a
    pass/fail engineering spec, which is normally an absolute number (e.g. "insertion
    loss under budget"), not a percentage of what a specific unit could hypothetically
    achieve if searched exhaustively.

    Returns the same `EpisodeMetrics` as `run_episode_metrics` (computed from the
    unshaped true reward, so results are directly comparable to unshaped training),
    with `steps_to_threshold` measured against this same `spec_threshold`.
    """
    obs = env.reset()
    best_true = env.true_reward_at(obs)
    steps_to_threshold = 0 if best_true >= spec_threshold else None
    control_effort = 0
    boundary_hits = 0
    reached = steps_to_threshold is not None
    done = False
    while not done:
        action = agent.act(obs)
        result = env.step(action)
        control_effort += 0 if action == ACTIONS.index("stay") else 1
        x, y = result.obs
        if x in (0, env.levels - 1) or y in (0, env.levels - 1):
            boundary_hits += 1
        just_reached = (not reached) and (result.noisy_reward >= spec_threshold)
        reached = reached or just_reached
        shaped_reward = (bonus if just_reached else 0.0) - step_cost + dense_scale * result.noisy_reward
        if learn:
            agent.observe(obs, action, shaped_reward, result.obs, result.done)
        obs = result.obs
        best_true = max(best_true, result.true_reward)
        if steps_to_threshold is None and best_true >= spec_threshold:
            steps_to_threshold = env.steps
        done = result.done
    return EpisodeMetrics(
        best_true_reward=best_true,
        steps_to_threshold=steps_to_threshold,
        control_effort=control_effort,
        boundary_hits=boundary_hits,
    )
