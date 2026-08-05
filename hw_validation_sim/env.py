"""A small, from-scratch calibration environment (deliberately not gymnasium — this is a
~60-line problem and a full RL-framework dependency would obscure more than it clarifies).

Models calibrating two device parameters (think: bias current, TEC setpoint on an optical
module) against a noisy performance measurement. The reward landscape has a global optimum
and a decoy local optimum, so naive greedy search can get stuck. The optimum's exact
location shifts slightly every episode ("unit-to-unit variation") -- a real calibration
landscape doesn't sit in exactly the same place on every unit off the line, which is what
makes a *learned* search policy potentially worth something over blind hill-climbing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTIONS = ["x+", "x-", "y+", "y-", "stay"]  # index 0..4


@dataclass
class StepResult:
    obs: tuple[int, int]
    noisy_reward: float
    true_reward: float
    done: bool


class CalibrationEnv:
    def __init__(
        self,
        levels: int = 20,
        max_steps: int = 40,
        noise_std: float = 0.05,
        unit_variation: float = 1.5,
        seed: int | None = None,
    ):
        self.levels = levels
        self.max_steps = max_steps
        self.noise_std = noise_std
        self.unit_variation = unit_variation
        self.rng = np.random.default_rng(seed)

        self._global_center = np.array([levels * 0.7, levels * 0.6])
        self._local_center = np.array([levels * 0.25, levels * 0.3])
        self._global_amp = 1.0
        self._local_amp = 0.55
        self._sigma = levels * 0.12

        self._center = None
        self.pos = None
        self.steps = 0

    def reset(self) -> tuple[int, int]:
        offset = self.rng.normal(0, self.unit_variation, size=2)
        self._center = np.clip(self._global_center + offset, 0, self.levels - 1)
        self.pos = np.array([self.levels // 2, self.levels // 2], dtype=float)
        self.steps = 0
        return self._obs()

    def _obs(self) -> tuple[int, int]:
        return int(round(self.pos[0])), int(round(self.pos[1]))

    def _true_reward(self, xy: np.ndarray) -> float:
        d_global = np.sum((xy - self._center) ** 2)
        d_local = np.sum((xy - self._local_center) ** 2)
        return float(
            self._global_amp * np.exp(-d_global / (2 * self._sigma**2))
            + self._local_amp * np.exp(-d_local / (2 * self._sigma**2))
        )

    def step(self, action: int) -> StepResult:
        if action == 0:
            self.pos[0] += 1
        elif action == 1:
            self.pos[0] -= 1
        elif action == 2:
            self.pos[1] += 1
        elif action == 3:
            self.pos[1] -= 1
        elif action == 4:
            pass
        else:
            raise ValueError(f"invalid action {action}; expected 0-4")

        self.pos = np.clip(self.pos, 0, self.levels - 1)
        true_r = self._true_reward(self.pos)
        noisy_r = true_r + self.rng.normal(0, self.noise_std)
        self.steps += 1
        done = self.steps >= self.max_steps
        return StepResult(self._obs(), float(noisy_r), true_r, done)

    def optimum_true_reward(self) -> float:
        """The true reward at the (unknown-to-the-agent) global optimum this episode."""
        return self._true_reward(self._center)

    def true_reward_at(self, xy: tuple[float, float]) -> float:
        """Public accessor so callers (e.g. run_episode) don't need env internals."""
        return self._true_reward(np.array(xy, dtype=float))
