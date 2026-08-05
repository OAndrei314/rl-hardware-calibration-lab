"""`python -m hw_validation_sim.cli run ...`"""
from __future__ import annotations

import argparse

from .experiment import evaluate_agents, train_qlearning_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hw-validation-sim")
    parser.add_argument("--levels", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=40, help="measurement budget per unit")
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--unit-variation", type=float, default=1.5)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

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
    print(f"{'strategy':<15} {'mean best true reward':<25} {'std':<10}")
    for r in results:
        print(f"{r.agent_name:<15} {r.mean:<25.4f} {r.std:<10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
