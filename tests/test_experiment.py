from hw_validation_sim.experiment import evaluate_agents, train_qlearning_agent


def test_trained_qlearning_beats_both_baselines():
    """Regression test pinned to observed behavior at these exact settings/seed:
    random_search ~0.567, hill_climb ~0.572, q_learning ~0.845. Margins below are
    deliberately loose (this is deterministic given the seed, but the margin protects
    against small implementation-detail changes rather than requiring exact equality)."""
    agent = train_qlearning_agent(
        levels=20, train_episodes=300, max_steps=40, noise_std=0.05,
        unit_variation=1.5, seed=0,
    )
    results = {
        r.agent_name: r
        for r in evaluate_agents(
            agent, levels=20, eval_episodes=60, max_steps=40, noise_std=0.05,
            unit_variation=1.5, seed=0,
        )
    }

    assert results["q_learning"].mean > results["random_search"].mean + 0.15
    assert results["q_learning"].mean > results["hill_climb"].mean + 0.15
    # Random search and hill climbing should be roughly comparable at this budget --
    # neither gets to exploit unit-to-unit similarity the way the trained agent does.
    assert abs(results["random_search"].mean - results["hill_climb"].mean) < 0.15
