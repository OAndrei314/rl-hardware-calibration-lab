from hw_validation_sim.experiment import (
    evaluate_agents,
    render_markdown_report,
    run_resolution_comparison,
    train_qlearning_agent,
)


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


def test_markdown_report_contains_engineering_metrics():
    agent = train_qlearning_agent(
        levels=12, train_episodes=40, max_steps=24, noise_std=0.05,
        unit_variation=1.0, seed=3,
    )
    results = evaluate_agents(
        agent, levels=12, eval_episodes=8, max_steps=24, noise_std=0.05,
        unit_variation=1.0, seed=3,
    )

    report = render_markdown_report(results)

    assert "mean steps to threshold" in report
    assert "Why these metrics, not just reward" in report
    assert "q_learning" in report


def test_linear_fa_generalizes_better_than_tabular_at_fine_resolution():
    """At coarse resolution (levels=20, the repo's original default) the tabular
    agent's Q-table gets enough visits per cell in 300 training episodes to beat
    linear function approximation -- matching the existing single-resolution result.
    At a much finer resolution (levels=150) with the *same* 300-episode budget, the
    tabular Q-table has far more cells than the budget can give even one visit to,
    while the FA agent's RBF features let one update generalize to nearby unvisited
    cells -- so FA overtakes it. Margins are loose (pinned regression on this exact
    seed, not exact equality)."""
    coarse = {
        r.agent_name: r
        for r in run_resolution_comparison(
            levels=20, train_episodes=300, eval_episodes=60, max_steps=40,
            noise_std=0.05, unit_variation=1.5, seed=0,
        )
    }
    assert coarse["q_learning_tabular"].mean > coarse["q_learning_linear_fa"].mean + 0.1

    fine = {
        r.agent_name: r
        for r in run_resolution_comparison(
            levels=150, train_episodes=300, eval_episodes=60, max_steps=40,
            noise_std=0.05, unit_variation=1.5, seed=0,
        )
    }
    assert fine["q_learning_linear_fa"].mean > fine["q_learning_tabular"].mean + 0.05
