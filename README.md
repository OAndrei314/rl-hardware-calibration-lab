# rl-hardware-calibration-lab

*Maintained by: claude-actions-daily-routine · Status: Active*
A small, from-scratch simulation comparing reinforcement learning against classical search
strategies for calibrating hardware — motivated by real hardware bring-up/validation work,
where every measurement is slow and every "unit" (device off the line) starts from a
slightly different optimal calibration point.

## Why this matters

**Research question:** under noisy measurements, unit-to-unit variation, and a fixed
measurement budget, can a learned policy exploit shared structure across devices better
than per-unit search?

**Practical impact:** calibration time is not abstract. Every measurement can consume tester
occupancy, thermal settling time, lab/debug time, and production capacity. In AI
infrastructure and optical hardware, where power, thermals, and link margins are expensive
constraints, a useful controller optimizes reward per measurement and avoids unsafe
regions, not just final score.

**Engineering evidence:** the experiment reports mean best true reward, success rate,
mean steps to threshold, control effort, and boundary hits on held-out simulated units.

## The problem this models

Calibrating a device (think: bias current and TEC setpoint on an optical module) against a
noisy performance measurement, where:
- The performance landscape has a **global optimum and a decoy local optimum** — naive
  greedy search can get stuck.
- Measurements are **noisy**, like a real instrument reading.
- The exact optimum **shifts slightly from unit to unit** (manufacturing variation) — so a
  strategy that starts from scratch on every unit can't get better over time, but one that
  *learns across units* potentially can.

That last point is the actual thesis of this repo: classical per-unit search (random,
hill-climbing) has no way to carry information from unit #1 to unit #500. A policy trained
across many simulated units can learn the general shape of the landscape — including where
the decoy optimum tends to be — and apply that on a brand-new unit it's never seen.

## What's implemented

- `hw_validation_sim/env.py` — a ~90-line calibration environment (not gymnasium; the
  problem is small enough that a full RL-framework dependency would add more overhead than
  clarity). Two discretized parameters, Gaussian-bump reward landscape with a global and a
  decoy local optimum, per-episode random offset for unit-to-unit variation, Gaussian
  measurement noise.
- `hw_validation_sim/agents.py` — three strategies on equal footing (same step budget, same
  `act()`/`observe()` interface):
  - `RandomSearchAgent` — the floor.
  - `HillClimbAgent` — propose a random move, keep it if the noisy reading improved on the
    best seen so far, otherwise step back. This is close to how a lot of real manual/
    rule-based calibration procedures work.
  - `QLearningAgent` — tabular Q-learning, trained across many simulated units before
    evaluation.
- `hw_validation_sim/experiment.py` — trains the Q-learning agent, then evaluates all three
  on **held-out** units using common random numbers per eval episode (all three agents face
  the identical noise sequence and unit offset in a given eval episode, so differences in
  outcome are due to the strategy, not lucky draws).

## Quickstart

```bash
pip install -r requirements.txt
python -m hw_validation_sim.cli --train-episodes 300 --eval-episodes 100 --seed 0 \
  --report reports/seed0.md
```

## Honest results

At the default settings (20×20 discretized parameter space, 40-measurement budget per
unit, 300 training units, 100 held-out eval units, seed 0):

| strategy | mean best true reward | std |
| --- | --- | --- |
| random_search | 0.549 | 0.287 |
| hill_climb | 0.557 | 0.285 |
| q_learning | 0.854 | 0.167 |

The CLI also reports success rate, mean steps to threshold, mean control effort, and
boundary hits. Those metrics matter commercially because they map to test-station time,
control activity, and safety margin.

Two things worth being honest about:
1. **Random search and hill-climbing come out roughly tied.** With a 40-step budget on a
   20×20 grid under measurement noise, greedy accept/revert doesn't have much of an edge
   over blind luck — noisy accept/reject decisions undercut the "greedy" part. That's a
   real result, not a bug.
2. **This only works because the training and eval landscapes come from the same family.**
   Q-learning here is exploiting the fact that all units share a landscape shape and it got
   to see 300 examples of it. It is not claiming to solve calibration in general — it's
   demonstrating the specific, real advantage of learning across units when the underlying
   physics is shared, which is exactly the assumption that holds on a real production line.
3. **Training curve matters**: with only 150 training episodes instead of 300, the trained
   agent actually performs *worse* than both baselines (0.39 vs ~0.57) — 300 was the point
   where the Q-table had converged enough to be useful. Worth remembering before assuming
   "add RL" is automatically a win; here it very much depended on giving it enough data.

## Status / next steps

Tabular Q-learning only works because the state space is small (20×20). A continuous
calibration space would need function approximation (e.g. a small neural Q-network) instead
— a natural next step, along with reporting steps-to-threshold rather than only best reward
found.

## License

MIT — see [LICENSE](LICENSE).
