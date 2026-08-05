import numpy as np

from hw_validation_sim.env import CalibrationEnv


def test_reset_is_deterministic_given_seed():
    env_a = CalibrationEnv(seed=42)
    env_b = CalibrationEnv(seed=42)
    obs_a = env_a.reset()
    obs_b = env_b.reset()
    assert obs_a == obs_b


def test_position_stays_within_bounds():
    env = CalibrationEnv(levels=20, max_steps=200, seed=1)
    env.reset()
    for _ in range(200):
        result = env.step(0)  # always push x+
        x, y = result.obs
        assert 0 <= x < 20
        assert 0 <= y < 20
        if result.done:
            break


def test_episode_ends_after_max_steps():
    env = CalibrationEnv(max_steps=10, seed=0)
    env.reset()
    done = False
    steps = 0
    while not done:
        result = env.step(4)  # "stay"
        done = result.done
        steps += 1
    assert steps == 10


def test_reward_is_higher_near_global_optimum_than_far_away():
    env = CalibrationEnv(levels=20, unit_variation=0.0, seed=0)
    env.reset()
    near_optimum = env.true_reward_at((14, 12))  # close to the (0.7, 0.6)*20 center
    far_corner = env.true_reward_at((0, 0))
    assert near_optimum > far_corner
