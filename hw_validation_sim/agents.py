"""Three search strategies compared on the same calibration budget:

- RandomSearchAgent: uniform random moves. The floor.
- HillClimbAgent: measure, nudge in a random direction, keep the move if the noisy
  reading improved, otherwise revert. This is close to how a lot of real manual/
  rule-based calibration procedures actually work.
- QLearningAgent: tabular Q-learning, trained across many simulated "units" so it can
  learn the general shape of the landscape (including the decoy local optimum) instead
  of re-discovering it from scratch on every new unit.
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
