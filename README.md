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
- `hw_validation_sim/agents.py` — four strategies on equal footing (same step budget, same
  `act()`/`observe()` interface):
  - `RandomSearchAgent` — the floor.
  - `HillClimbAgent` — propose a random move, keep it if the noisy reading improved on the
    best seen so far, otherwise step back. This is close to how a lot of real manual/
    rule-based calibration procedures work.
  - `QLearningAgent` — tabular Q-learning, trained across many simulated units before
    evaluation.
  - `LinearFAQAgent` — Q-learning with linear function approximation over a fixed grid of
    radial basis function (RBF) features, for when the grid is too fine for a tabular
    Q-table to get a useful number of visits per cell within a fixed training budget.
- `hw_validation_sim/experiment.py` — trains the Q-learning agent, then evaluates all three
  on **held-out** units using common random numbers per eval episode (all three agents face
  the identical noise sequence and unit offset in a given eval episode, so differences in
  outcome are due to the strategy, not lucky draws).

## Quickstart

```bash
pip install -r requirements.txt
python -m hw_validation_sim.cli --train-episodes 300 --eval-episodes 100 --seed 0 \
  --report reports/seed0.md

# Compare tabular vs. linear-FA Q-learning as the calibration grid gets finer, under
# the same fixed 300-episode training budget:
python -m hw_validation_sim.cli --compare-resolutions 20 60 100 150 200 --seed 0
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

### Does function approximation actually help at finer resolution?

The tabular agent's edge above only holds because a 20×20 grid (2,000 Q-table cells) can get
a useful number of visits from 300 training units × 40 steps. `--compare-resolutions` trains
both the tabular agent and `LinearFAQAgent` (36 RBF centers) at increasingly fine grids,
**holding the training budget fixed at 300 episodes**, then evaluates both on 100 held-out
units (seed 0):

| grid (levels²) | tabular cells | tabular mean reward | tabular success | FA mean reward | FA success |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20×20 | 2,000 | 0.841 | 67% | 0.600 | 25% |
| 60×60 | 18,000 | 0.843 | 50% | 0.666 | 7% |
| 100×100 | 50,000 | 0.702 | 3% | 0.738 | 10% |
| 150×150 | 112,500 | 0.380 | 0% | **0.501** | 0% |

The story is real but not clean, and it's worth reporting honestly rather than rounding it
off into a tidy narrative:
- The tabular agent's performance degrades as the grid gets finer under the same fixed
  budget — exactly the budget-starvation the "next steps" note below predicted. FA degrades
  too (the underlying problem is genuinely harder at 150×150 than 20×20 — steps are still
  single grid cells, so the same 40-step budget covers proportionally less ground), but more
  gracefully, and crosses over to beat tabular by 100×100–150×150.
- **That crossover does not hold indefinitely.** At 200×200 (seed 0), tabular actually
  recovers relative to FA (tabular 0.325 vs. FA 0.193) — both agents are so budget-starved at
  that resolution that the comparison is dominated by which one got luckier with its training
  trajectory, not by which representation generalizes better. Averaging over 8 training
  seeds at 200×200 shows FA ahead on average (0.49 vs. 0.38) but with high variance in both
  (std ≈ 0.2–0.23) — so "FA wins at extreme resolution" is a tendency, not a guarantee, on
  the noisy single-seed numbers a real run would see.
- **The effect is noisy at the individual-seed level in general.** At 20×20, a spot-check
  across 8 training seeds found tabular ahead of FA about 75% of the time, not 100% — so the
  single-seed numbers above are representative, not definitive. If you need a specific
  resolution's ranking to be reliable rather than "usually true," train several seeds and
  compare means, the same way the 8-seed spot-check above did.

## Status / next steps

Implemented: `LinearFAQAgent`, a linear function approximator over RBF features, as the
next step this README used to call for ("a continuous calibration space would need function
approximation instead"). It's linear rather than a small neural net — a semi-gradient update
with a fixed feature map needs no optimizer or network-shape search, and the weight vector
stays directly inspectable, which matters for a controller you'd want to validate before
trusting it near a hardware safety limit. See "Does function approximation actually help at
finer resolution?" above for the honest, noisy-in-places result.

Remaining open threads: the RBF center grid (6×6) and its width are fixed by hand rather
than tuned per resolution, and steps-to-threshold is now reported but not yet used as an
optimization target (agents are still trained to maximize reward, not to minimize
measurements-to-acceptable). A per-resolution sweep of RBF center count, run across enough
training seeds to get real confidence intervals instead of single-seed point estimates,
would be the next thing to do here.

## License

MIT — see [LICENSE](LICENSE).
