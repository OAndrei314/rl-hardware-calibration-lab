"""`python -m hw_validation_sim.cli run ...`"""
from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import (
    evaluate_agents,
    render_markdown_report,
    run_center_sweep,
    run_joint_sweep,
    run_resolution_comparison,
    run_sigma_sweep,
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
    parser.add_argument(
        "--sweep-centers-at-levels",
        type=int,
        default=None,
        metavar="LEVELS",
        help=(
            "instead of the baseline or resolution comparisons, sweep the linear-FA "
            "agent's RBF center count at this single grid resolution, averaging each "
            "count over --sweep-seeds independent training seeds to get a real "
            "confidence interval instead of a single-seed point estimate"
        ),
    )
    parser.add_argument(
        "--center-counts",
        type=int,
        nargs="+",
        default=[4, 6, 8, 10, 14],
        metavar="N",
        help="candidate n_centers_per_dim values for --sweep-centers-at-levels",
    )
    parser.add_argument(
        "--sweep-seeds",
        type=int,
        default=8,
        help=(
            "independent training seeds averaged per point in --sweep-centers-at-levels "
            "or --sweep-sigma-at-levels"
        ),
    )
    parser.add_argument(
        "--sweep-sigma-at-levels",
        type=int,
        default=None,
        metavar="LEVELS",
        help=(
            "instead of the other modes, sweep the linear-FA agent's RBF width "
            "(sigma_scale, a multiplier on the default one-grid-spacing-between-centers "
            "width) at this grid resolution and a fixed center count, averaging each "
            "width over --sweep-seeds independent training seeds"
        ),
    )
    parser.add_argument(
        "--sigma-scales",
        type=float,
        nargs="+",
        default=[0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0],
        metavar="SCALE",
        help="candidate sigma_scale values for --sweep-sigma-at-levels",
    )
    parser.add_argument(
        "--sigma-sweep-centers",
        type=int,
        default=14,
        metavar="N",
        help="fixed n_centers_per_dim to hold constant while sweeping sigma_scale",
    )
    parser.add_argument(
        "--sweep-joint-at-levels",
        type=int,
        default=None,
        metavar="LEVELS",
        help=(
            "instead of the other modes, sweep RBF center count and sigma_scale "
            "together (every combination of --center-counts x --sigma-scales) at "
            "this grid resolution, averaging each combination over --sweep-seeds "
            "independent training seeds -- reveals interactions between the two "
            "hyperparameters that the separate 1D sweeps can't see"
        ),
    )
    args = parser.parse_args(argv)

    if args.sweep_joint_at_levels is not None:
        print(
            f"Sweeping RBF center count {args.center_counts} x sigma_scale "
            f"{args.sigma_scales} jointly at levels={args.sweep_joint_at_levels}, "
            f"{args.sweep_seeds} training seeds each, fixed {args.train_episodes}-"
            "episode training budget..."
        )
        print()
        points = run_joint_sweep(
            levels=args.sweep_joint_at_levels,
            center_counts=args.center_counts,
            sigma_scales=args.sigma_scales,
            n_seeds=args.sweep_seeds,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            max_steps=args.max_steps,
            noise_std=args.noise_std,
            unit_variation=args.unit_variation,
            base_seed=args.seed,
        )
        header = f"{'centers/dim':<14}" + "".join(f"sigma={s:<10}" for s in args.sigma_scales)
        print(header)
        for n_centers in args.center_counts:
            row = [p for p in points if p.n_centers_per_dim == n_centers]
            cells = "".join(f"{p.mean:<15.4f}" for p in row)
            print(f"{n_centers:<14}{cells}")
        best = max(points, key=lambda p: p.mean)
        print()
        print(
            f"Best: centers/dim={best.n_centers_per_dim}, sigma_scale={best.sigma_scale}, "
            f"mean reward={best.mean:.4f} (+/- {best.ci95_halfwidth:.4f})"
        )
        return 0

    if args.sweep_sigma_at_levels is not None:
        print(
            f"Sweeping RBF sigma_scale {args.sigma_scales} at levels="
            f"{args.sweep_sigma_at_levels}, centers/dim={args.sigma_sweep_centers}, "
            f"{args.sweep_seeds} training seeds each, fixed {args.train_episodes}-episode "
            "training budget..."
        )
        print()
        points = run_sigma_sweep(
            levels=args.sweep_sigma_at_levels,
            n_centers_per_dim=args.sigma_sweep_centers,
            sigma_scales=args.sigma_scales,
            n_seeds=args.sweep_seeds,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            max_steps=args.max_steps,
            noise_std=args.noise_std,
            unit_variation=args.unit_variation,
            base_seed=args.seed,
        )
        print(f"{'sigma_scale':<14}{'mean reward':<14}{'std (seeds)':<14}{'95% CI +/-':<12}")
        for p in points:
            print(f"{p.sigma_scale:<14}{p.mean:<14.4f}{p.std:<14.4f}{p.ci95_halfwidth:<12.4f}")
        return 0

    if args.sweep_centers_at_levels is not None:
        print(
            f"Sweeping RBF center count {args.center_counts} at levels="
            f"{args.sweep_centers_at_levels}, {args.sweep_seeds} training seeds each, "
            f"fixed {args.train_episodes}-episode training budget..."
        )
        print()
        points = run_center_sweep(
            levels=args.sweep_centers_at_levels,
            center_counts=args.center_counts,
            n_seeds=args.sweep_seeds,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            max_steps=args.max_steps,
            noise_std=args.noise_std,
            unit_variation=args.unit_variation,
            base_seed=args.seed,
        )
        print(f"{'centers/dim':<14}{'mean reward':<14}{'std (seeds)':<14}{'95% CI +/-':<12}")
        for p in points:
            print(f"{p.n_centers_per_dim:<14}{p.mean:<14.4f}{p.std:<14.4f}{p.ci95_halfwidth:<12.4f}")
        return 0

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
