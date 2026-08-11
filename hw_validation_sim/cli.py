"""`python -m hw_validation_sim.cli run ...`"""
from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import (
    evaluate_agents,
    render_markdown_report,
    run_resolution_comparison,
    train_qlearning_agent,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hw-validation-sim")
    parser.add_argument("--levels", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=40, help="measurement budget per unit")
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--unit-variation", type=float, default=1.5)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", help="optional markdown report path")
    parser.add_argument(
        "--compare-resolutions",
        type=int,
        nargs="+",
        default=None,
        metavar="LEVELS",
        help=(
            "instead of the baseline comparison, train tabular Q-learning and "
            "linear-FA Q-learning at each given grid resolution (same fixed "
            "training/eval budget) and report how each strategy's performance "
            "changes as the grid gets too fine for the tabular table to cover"
        ),
    )
    args = parser.parse_args(argv)

    if args.compare_resolutions:
        print(
            f"Comparing tabular vs. linear-FA Q-learning across resolutions "
            f"{args.compare_resolutions} with a fixed {args.train_episodes}-episode "
            f"training budget..."
        )
        print()
        print(f"{'levels':<8} {'strategy':<22} {'mean best true reward':<25} {'success':<10}")
        for levels in args.compare_resolutions:
            results = run_resolution_comparison(
                levels=levels,
                train_episodes=args.train_episodes,
                eval_episodes=args.eval_episodes,
                max_steps=args.max_steps,
                noise_std=args.noise_std,
                unit_variation=args.unit_variation,
                seed=args.seed,
            )
            for r in results:
                print(f"{levels:<8} {r.agent_name:<22} {r.mean:<25.4f} {r.success_rate:<10.1%}")
        return 0

    print(
        f"Training Q-learning agent on {args.train_episodes} simulated units "
        f"({args.max_steps}-measurement budget each)..."
    )
    agent = train_qlearning_agent(
        levels=args.levels,
        train_episodes=args.train_episodes,
        max_steps=args.max_steps,
        noise_std=args.noise_std,
        unit_variation=args.unit_variation,
        seed=args.seed,
    )

    print(f"Evaluating all 3 strategies on {args.eval_episodes} held-out units...")
    results = evaluate_agents(
        trained_qlearning=agent,
        levels=args.levels,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        noise_std=args.noise_std,
        unit_variation=args.unit_variation,
        seed=args.seed,
    )

    print()
    print(
        f"{'strategy':<15} {'mean best true reward':<25} {'success':<10} "
        f"{'steps<thr':<12} {'effort':<10}"
    )
    for r in results:
        print(
            f"{r.agent_name:<15} {r.mean:<25.4f} {r.success_rate:<10.1%} "
            f"{r.mean_steps_to_threshold:<12.2f} {r.mean_control_effort:<10.2f}"
        )
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown_report(results), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
