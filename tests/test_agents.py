import numpy as np
import pytest

from hw_validation_sim.agents import (
    HillClimbAgent,
    LinearFAQAgent,
    QLearningAgent,
    run_episode,
    run_episode_metrics,
    run_episode_shaped,
)
from hw_validation_sim.env import CalibrationEnv


class _RecordingAgent:
    """Always takes a fixed action; logs every reward `observe()` receives, so tests
    can inspect the exact shaped-reward sequence without depending on Q-learning."""

    def __init__(self, fixed_action: int = 0):
        self.fixed_action = fixed_action
        self.rewards: list[float] = []

    def act(self, obs):
        return self.fixed_action

    def observe(self, obs, action, reward, next_obs, done) -> None:
        self.rewards.append(reward)


def test_hill_climb_reverts_after_a_worse_reading():
    agent = HillClimbAgent(levels=20, rng=np.random.default_rng(0))
    action = agent.act((10, 10))
    assert agent.state == "propose"  # no observation yet, state unchanged until observe()
    # A worse reading than the (unset) baseline is impossible on the first move, since
    # best_reward_so_far starts as None -- so the first move is always accepted.
    agent.observe((10, 10), action, reward=0.1, next_obs=(11, 10), done=False)
    assert agent.best_reward_so_far == 0.1
    assert agent.state == "propose"

    action2 = agent.act((11, 10))
    agent.observe((11, 10), action2, reward=0.05, next_obs=(12, 10), done=False)  # worse
    assert agent.state == "revert"

    revert_action = agent.act((12, 10))
    assert revert_action == HillClimbAgent._OPPOSITE[action2]


def test_qlearning_updates_q_table_toward_higher_reward_action():
    agent = QLearningAgent(levels=5, rng=np.random.default_rng(0), epsilon=0.0)
    # Action 0 leads to a much better reward than action 1 from the same state.
    for _ in range(50):
        agent.observe((2, 2), 0, reward=1.0, next_obs=(3, 2), done=True)
        agent.observe((2, 2), 1, reward=-1.0, next_obs=(1, 2), done=True)
    assert agent.q[2, 2, 0] > agent.q[2, 2, 1]


def test_linear_fa_updates_weights_toward_higher_reward_action():
    agent = LinearFAQAgent(levels=20, rng=np.random.default_rng(0), epsilon=0.0)
    for _ in range(50):
        agent.observe((10, 10), 0, reward=1.0, next_obs=(11, 10), done=True)
        agent.observe((10, 10), 1, reward=-1.0, next_obs=(9, 10), done=True)
    phi = agent._features((10, 10))
    q_values = agent.weights @ phi
    assert q_values[0] > q_values[1]


def test_linear_fa_generalizes_to_a_nearby_unvisited_state():
    """The whole point of function approximation over a tabular Q-table: an update
    at one state should shift the estimated value of a *different*, never-visited,
    nearby state too -- because they share RBF features."""
    agent = LinearFAQAgent(levels=20, rng=np.random.default_rng(0), epsilon=0.0)
    unvisited = (11, 11)
    phi_before = agent._features(unvisited)
    assert np.allclose(agent.weights @ phi_before, 0.0)  # untrained: all zero
    for _ in range(20):
        agent.observe((10, 10), 0, reward=1.0, next_obs=(11, 10), done=True)
    q_unvisited = agent.weights @ agent._features(unvisited)
    assert q_unvisited[0] > 0.0  # action 0's value leaked over to a nearby state


def test_linear_fa_eval_mode_stops_learning_and_exploring():
    agent = LinearFAQAgent(levels=20, rng=np.random.default_rng(0), epsilon=1.0)
    agent.eval_mode()
    weights_before = agent.weights.copy()
    agent.observe((5, 5), 2, reward=1.0, next_obs=(5, 6), done=True)
    assert np.array_equal(agent.weights, weights_before)  # no update while not training
    # epsilon=1.0 would always explore if training; eval_mode must suppress that too.
    action = agent.act((5, 5))
    assert action == int(np.argmax(agent.weights @ agent._features((5, 5))))


def test_run_episode_returns_a_float_and_env_is_reset_internally():
    env = CalibrationEnv(seed=0)
    agent = QLearningAgent(levels=20, rng=np.random.default_rng(0))
    best = run_episode(env, agent, learn=True)
    assert isinstance(best, float)
    assert 0.0 <= best <= 1.0


def test_run_episode_metrics_include_effort_and_threshold():
    env = CalibrationEnv(levels=20, max_steps=20, unit_variation=0.0, seed=0)
    agent = QLearningAgent(levels=20, rng=np.random.default_rng(0), epsilon=0.0)

    metrics = run_episode_metrics(env, agent, learn=False, threshold_fraction=0.1)

    assert 0.0 <= metrics.best_true_reward <= 1.0
    assert metrics.steps_to_threshold is not None
    assert metrics.control_effort <= 20
    assert metrics.boundary_hits >= 0


def test_run_episode_shaped_gives_the_bonus_exactly_once():
    """With noise_std=0 (noisy reward == true reward, deterministic) and a fixed
    "always move +x" agent starting left of the global optimum, the episode must
    cross a low spec_threshold on its very first step and never re-cross it (the
    bonus is one-time by design). Every other step should be pure -step_cost."""
    env = CalibrationEnv(levels=20, max_steps=10, noise_std=0.0, unit_variation=0.0, seed=0)
    agent = _RecordingAgent(fixed_action=0)  # action 0 == "x+"

    metrics = run_episode_shaped(env, agent, spec_threshold=0.2, step_cost=0.02, bonus=1.0, learn=True)

    assert len(agent.rewards) == 10
    bonus_indices = [i for i, r in enumerate(agent.rewards) if r > 0]
    assert len(bonus_indices) == 1
    assert agent.rewards[bonus_indices[0]] == pytest.approx(1.0 - 0.02)
    other_rewards = [r for i, r in enumerate(agent.rewards) if i != bonus_indices[0]]
    assert all(r == pytest.approx(-0.02) for r in other_rewards)
    assert metrics.steps_to_threshold == bonus_indices[0] + 1  # env.steps is 1-indexed


def test_run_episode_shaped_never_reaches_an_impossible_threshold():
    env = CalibrationEnv(levels=20, max_steps=10, noise_std=0.0, unit_variation=0.0, seed=0)
    agent = _RecordingAgent(fixed_action=0)

    metrics = run_episode_shaped(env, agent, spec_threshold=5.0, step_cost=0.02, bonus=1.0, learn=True)

    assert metrics.steps_to_threshold is None
    assert all(r == pytest.approx(-0.02) for r in agent.rewards)


def test_run_episode_shaped_skips_agent_updates_when_learn_is_false():
    env = CalibrationEnv(levels=20, max_steps=10, noise_std=0.0, unit_variation=0.0, seed=0)
    agent = _RecordingAgent(fixed_action=0)

    run_episode_shaped(env, agent, spec_threshold=0.2, step_cost=0.02, bonus=1.0, learn=False)

    assert agent.rewards == []


def test_run_episode_shaped_default_dense_scale_is_unchanged_from_pure_sparse():
    """dense_scale defaults to 0.0, which must add exactly nothing -- every reward
    already established by test_run_episode_shaped_gives_the_bonus_exactly_once
    must be untouched when dense_scale is passed explicitly as 0.0."""
    env_a = CalibrationEnv(levels=20, max_steps=10, noise_std=0.0, unit_variation=0.0, seed=0)
    env_b = CalibrationEnv(levels=20, max_steps=10, noise_std=0.0, unit_variation=0.0, seed=0)
    agent_default = _RecordingAgent(fixed_action=0)
    agent_explicit_zero = _RecordingAgent(fixed_action=0)

    run_episode_shaped(env_a, agent_default, spec_threshold=0.2, step_cost=0.02, bonus=1.0)
    run_episode_shaped(
        env_b, agent_explicit_zero, spec_threshold=0.2, step_cost=0.02, bonus=1.0, dense_scale=0.0
    )

    assert agent_default.rewards == agent_explicit_zero.rewards


def test_run_episode_shaped_dense_scale_adds_a_reward_proportional_to_the_reading():
    """With noise_std=0 (noisy reward == true reward) and dense_scale=0.5, every
    step's shaped reward should equal the pure-sparse reward (bonus-or-nothing minus
    step_cost) plus 0.5 * that step's true reward -- checked directly against a
    dense_scale=0.0 run of the identical trajectory, not just "reward went up"."""
    agent_sparse = _RecordingAgent(fixed_action=0)
    env_sparse = CalibrationEnv(levels=20, max_steps=10, noise_std=0.0, unit_variation=0.0, seed=0)
    run_episode_shaped(env_sparse, agent_sparse, spec_threshold=5.0, step_cost=0.02, bonus=1.0)

    agent_dense = _RecordingAgent(fixed_action=0)
    env_dense = CalibrationEnv(levels=20, max_steps=10, noise_std=0.0, unit_variation=0.0, seed=0)
    run_episode_shaped(
        env_dense, agent_dense, spec_threshold=5.0, step_cost=0.02, bonus=1.0, dense_scale=0.5,
    )

    # spec_threshold=5.0 is unreachable, so both trajectories are identical (same
    # fixed action, same deterministic env) and every sparse reward is exactly
    # -step_cost -- isolating the dense term as the only difference between them.
    dense_minus_sparse = [d - s for d, s in zip(agent_dense.rewards, agent_sparse.rewards)]
    # Recover each step's true reward from the deterministic env by replaying it.
    replay_env = CalibrationEnv(levels=20, max_steps=10, noise_std=0.0, unit_variation=0.0, seed=0)
    replay_env.reset()
    true_rewards = [replay_env.step(0).true_reward for _ in range(10)]
    assert dense_minus_sparse == pytest.approx([0.5 * r for r in true_rewards])
