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
  outcome are due to the strategy, not lucky draws). Also: `run_resolution_comparison`
  (tabular vs. linear-FA as the grid gets finer) and `run_center_sweep` (linear-FA's RBF
  center count at a fixed resolution, averaged over multiple training seeds with a 95% CI
  per count — a paired design that reuses the same seed stream across center counts).

## Quickstart

```bash
pip install -r requirements.txt
python -m hw_validation_sim.cli --train-episodes 300 --eval-episodes 100 --seed 0 \
  --report reports/seed0.md

# Compare tabular vs. linear-FA Q-learning as the calibration grid gets finer, under
# the same fixed 300-episode training budget:
python -m hw_validation_sim.cli --compare-resolutions 20 60 100 150 200 --seed 0

# Sweep the linear-FA agent's RBF center count at a fixed resolution, averaged over
# multiple training seeds to get a real confidence interval per center count:
python -m hw_validation_sim.cli --sweep-centers-at-levels 150 \
  --center-counts 3 4 6 8 10 14 20 --sweep-seeds 12 --seed 0
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

### Does RBF center count matter, and is a single seed's answer trustworthy?

The finer-resolution numbers above use a fixed 6×6 RBF center grid picked by hand, with no
attempt to check whether that count is any good. `--sweep-centers-at-levels` trains the FA
agent at a fixed resolution across a range of `n_centers_per_dim` values, each averaged over
multiple independent training seeds, and reports the across-seed mean, sample std, and a 95%
CI half-width — not a single-seed point estimate. At levels=150 (the resolution where FA
already beats tabular above), pooling two independent 12-seed batches (seeds 0 and 7, 300
training episodes, 80 held-out eval units — 24 seeds' worth of training runs total):

| centers/dim | mean reward (24 seeds) |
| ---: | ---: |
| 3 | 0.523 |
| 4 | 0.519 |
| 6 | 0.584 |
| 8 | 0.580 |
| 10 | 0.582 |
| 14 | **0.616** |
| 20 | 0.509 |

Two things worth being honest about here too:
1. **There's a real, replicated U-shape.** Both the seed-0 and seed-7 batches independently
   show the extremes (3-4 and 20 centers) underperforming the 6-14 middle band — consistent
   with the mechanism this repo already argues for: too few centers underfits the two-bump
   landscape, too many makes each RBF too narrow to get useful coverage from a fixed 300-episode
   budget (the same budget-starvation story as the tabular Q-table going too fine, just for a
   different resource).
2. **The exact peak within that middle band is not resolved at 12 seeds per point.** The
   seed-0 batch alone peaks at 6 (0.615); the seed-7 batch alone peaks at 14 (0.645); only the
   pooled 24-seed number lands on 14. The per-point 95% CIs at 12 seeds are typically
   ±0.11-0.15 — wide enough that 6, 8, 10, and 14 are not statistically distinguishable from
   each other on either batch alone. A run that reported just one seed's "optimal" center count
   as a hyperparameter recommendation would have been reporting noise. This is exactly the
   failure mode the "Status / next steps" note below used to flag before this sweep existed.

## Status / next steps

Implemented: `LinearFAQAgent`, a linear function approximator over RBF features, as the
next step this README used to call for ("a continuous calibration space would need function
approximation instead"). It's linear rather than a small neural net — a semi-gradient update
with a fixed feature map needs no optimizer or network-shape search, and the weight vector
stays directly inspectable, which matters for a controller you'd want to validate before
trusting it near a hardware safety limit. See "Does function approximation actually help at
finer resolution?" above for the honest, noisy-in-places result.

Also implemented: `run_center_sweep` / `--sweep-centers-at-levels`, the per-resolution RBF
center-count sweep with multi-seed confidence intervals this README used to call for. See
"Does RBF center count matter, and is a single seed's answer trustworthy?" above — the
honest answer is "somewhat, but not precisely at 12 seeds per point."

Remaining open threads: the RBF center grid's *width* (`sigma`, currently `levels /
n_centers_per_dim`) is still fixed by the same hand-picked formula regardless of resolution,
and isn't swept independently of center count — a wider or narrower RBF at a fixed center
count could plausibly move the U-shape's peak on its own. Steps-to-threshold is now reported
but not yet used as an optimization target (agents are still trained to maximize reward, not
to minimize measurements-to-acceptable). Pinning down the 6-14 middle band's true peak would
need roughly 4x today's seed count per point (variance shrinks with the square root of seed
count, and the CIs above need to roughly halve to separate those points) — a reasonable next
run if the exact center count ever mattered more than "somewhere in a broad, boring middle
range, not at the extremes."

## License

MIT — see [LICENSE](LICENSE).
