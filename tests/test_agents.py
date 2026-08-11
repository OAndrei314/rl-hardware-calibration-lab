import numpy as np

from hw_validation_sim.agents import (
    HillClimbAgent,
    LinearFAQAgent,
    QLearningAgent,
    run_episode,
    run_episode_metrics,
)
from hw_validation_sim.env import CalibrationEnv


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
