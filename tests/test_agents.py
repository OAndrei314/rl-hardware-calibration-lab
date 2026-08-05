import numpy as np

from hw_validation_sim.agents import HillClimbAgent, QLearningAgent, run_episode
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


def test_run_episode_returns_a_float_and_env_is_reset_internally():
    env = CalibrationEnv(seed=0)
    agent = QLearningAgent(levels=20, rng=np.random.default_rng(0))
    best = run_episode(env, agent, learn=True)
    assert isinstance(best, float)
    assert 0.0 <= best <= 1.0
