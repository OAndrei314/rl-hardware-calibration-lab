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
  (tabular vs. linear-FA as the grid gets finer), `run_center_sweep` / `run_sigma_sweep`
  (linear-FA's RBF center count and width swept independently, each averaged over multiple
  training seeds with a 95% CI — a paired design that reuses the same seed stream across
  values), and `run_joint_sweep` (center count and width swept *together*, over every
  combination in a grid, to check for an interaction the two independent 1D sweeps can't
  see).

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

# Sweep the linear-FA agent's RBF width (sigma_scale, a multiplier on the default
# one-grid-spacing-between-centers width) at a fixed resolution and center count:
python -m hw_validation_sim.cli --sweep-sigma-at-levels 150 --sigma-sweep-centers 14 \
  --sigma-scales 0.2 0.4 0.6 0.8 1.0 1.5 2.0 3.0 --sweep-seeds 12 --seed 0

# Sweep RBF center count and width together (every combination), to check for an
# interaction the two 1D sweeps above can't see:
python -m hw_validation_sim.cli --sweep-joint-at-levels 150 \
  --center-counts 6 10 14 20 --sigma-scales 0.4 0.6 0.8 1.0 --sweep-seeds 12 --seed 0
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

### Does RBF width matter, independently of center count?

The center-count sweep above holds each RBF's width (`sigma`) fixed by a hand-picked
formula (`levels / n_centers_per_dim`) while varying how many centers there are.
`--sweep-sigma-at-levels` does the opposite: it fixes `n_centers_per_dim=14` (the best
center count found above) and `levels=150`, then varies `sigma_scale`, a multiplier on
that same formula (`sigma_scale=1.0` reproduces the center-count sweep's numbers exactly
by construction). Same design as above: two independent 12-seed batches (seeds 0 and 7,
300 training episodes, 80 held-out eval units) pooled to 24 seeds per point:

| sigma_scale | mean reward (24 seeds) | 95% CI half-width |
| ---: | ---: | ---: |
| 0.2 | 0.519 | ±0.098 |
| 0.4 | 0.627 | ±0.105 |
| 0.6 | **0.671** | ±0.069 |
| 0.8 | 0.610 | ±0.075 |
| 1.0 (formula default) | 0.616 | ±0.100 |
| 1.5 | 0.469 | ±0.091 |
| 2.0 | 0.339 | ±0.069 |
| 3.0 | 0.237 | ±0.039 |

Honest reading:
1. **The hand-picked default width is not obviously optimal, but it's also not obviously
   wrong.** `sigma_scale=0.6` (RBF bumps ~40% narrower than one grid-spacing-between-centers)
   pooled ahead of the default `1.0` by about 0.055, but its CI (0.602–0.740) overlaps the
   0.4/0.8/1.0 CIs almost completely — the 0.4-1.0 range is one noisy, statistically
   indistinguishable band, the same "broad middle plateau, not a sharp peak" shape the
   center-count sweep found.
2. **Unlike the center-count sweep, the extremes here are unambiguous and monotonic in one
   direction.** Every step past `1.0` toward wider RBFs (`1.5 -> 2.0 -> 3.0`) is a clear,
   non-overlapping-CI drop — a bump wide enough to blur the global optimum and the decoy
   local optimum together stops being able to tell them apart, which is exactly the failure
   mode a fixed, un-swept width formula risks silently walking into on a differently-shaped
   landscape. The narrow end (`0.2`) is worse too, though not as cleanly separated from the
   0.4-1.0 band, consistent with under-generalizing individual updates the way too many RBF
   centers did in the count sweep.
3. **Net effect of both sweeps together**: at this resolution, `n_centers_per_dim=14,
   sigma_scale≈0.6` measures modestly better (~0.67 vs ~0.50 for the repo's original
   untuned `centers=6, sigma_scale=1.0`), but the gain over the *already-tuned* `centers=14,
   sigma_scale=1.0` default (~0.62) is inside the noise floor at 24 seeds per point. The
   practical takeaway for this landscape is narrower: don't use a wide RBF (`sigma_scale`
   much above 1.0), and a fixed, un-swept width formula is a reasonable engineering default
   as long as it isn't picking a value on the wrong side of that boundary.

### Does center count and width interact, or is tuning them separately good enough?

The two sweeps above each optimize one hyperparameter while holding the other at a fixed
default (`run_center_sweep` fixes `sigma_scale=1.0`; `run_sigma_sweep` fixes
`n_centers_per_dim=14`, the center sweep's winner). That can't detect an interaction between
them — e.g. whether the best width depends on how many centers are in play.
`--sweep-joint-at-levels` trains at every (center count, sigma_scale) combination in a grid
instead. At levels=150, pooling the same two independent 12-seed batches used above (seeds 0
and 7, 300 training episodes, 80 held-out eval units — 24 seeds per cell):

| centers/dim | sigma=0.4 | sigma=0.6 | sigma=0.8 | sigma=1.0 |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 0.688 | **0.697** | 0.643 | 0.584 |
| 10 | 0.596 | 0.625 | 0.625 | 0.582 |
| 14 | 0.627 | 0.671 | 0.610 | 0.616 |
| 20 | 0.674 | 0.619 | 0.534 | 0.509 |

95% CI half-widths across this grid run ±0.06 to ±0.10 per cell (not shown in the table above
to keep it readable — see the CLI output for exact values).

Honest reading:
1. **Sanity check first**: the `sigma_scale=1.0` column reproduces `run_center_sweep`'s
   earlier result almost exactly — centers=14 wins that column at 0.616 here vs. 0.616 in the
   original 24-seed center sweep, and the same relative ordering of 6/10/14/20 holds. That's
   reassuring: this is a genuinely new sweep over new combinations, not a reimplementation
   that happens to disagree with the numbers already reported above.
2. **The nominal joint optimum is centers=6, sigma_scale=0.6 (0.697 ± 0.072), not
   centers=14, sigma_scale=0.6 (0.671 ± 0.069)** — the combination you'd get by pasting
   together each 1D sweep's independently-tuned winner. But their 95% CIs overlap almost
   completely (0.625–0.770 vs. 0.602–0.740), along with two more cells in the same range
   (centers=6/sigma=0.4 at 0.688, centers=20/sigma=0.4 at 0.674) — a broad, noisy plateau
   again, the same shape both 1D sweeps already found on their own axes, not a resolvable
   sharp peak.
3. **There is a real interaction, though, at the high-center-count end.** Centers=20 falls
   monotonically and mostly outside overlapping CIs as sigma_scale grows (0.674 → 0.619 →
   0.534 → 0.509, sigma=0.4 to 1.0), while centers=6 peaks in the *middle* of the same range
   (0.4 and 0.6 are close, then it also declines toward 1.0). That's mechanistically sensible:
   with 20 centers already packed across the grid, a wide RBF (`sigma_scale` near 1.0) makes
   neighboring centers' bumps overlap enough to blur the global optimum and the decoy local
   optimum together — the same failure mode the sigma sweep already flagged for a single
   center count, but here it kicks in at a narrower absolute width because the centers
   themselves are closer together. Six centers spread far apart have more room before that
   happens.
4. **Net conclusion**: for this landscape, decomposing the 2D search into two sequential 1D
   sweeps (tune centers, then tune width) landed close to the true joint optimum — within
   noise, not exactly on it. That is a real result worth having actually checked rather than
   assumed: the interaction that exists (width tolerance shrinking as center count grows) is
   real and explicable, but it wasn't large enough at this landscape's scale to make the
   cheaper sequential-1D approach misleading. A landscape with sharper features (a narrower
   `_sigma` on the Gaussian bumps in `env.py`) is exactly the kind of case where that
   might not hold.

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

Also implemented: `run_sigma_sweep` / `--sweep-sigma-at-levels`, sweeping the RBF grid's
*width* (`sigma_scale`) independently of center count, the exact open thread this README
used to flag. See "Does RBF width matter, independently of center count?" above — the
honest answer is "a narrower-than-default width measures modestly better but isn't
distinguishable from the default at 24 seeds per point, while a wide RBF is a clear and
avoidable mistake."

Also implemented: `run_joint_sweep` / `--sweep-joint-at-levels`, the joint 2D sweep of center
count and sigma_scale this README used to flag as an open thread. See "Does center count and
width interact, or is tuning them separately good enough?" above — the honest answer is
"there's a real, mechanistically explicable interaction at high center counts, but at this
landscape's scale it wasn't large enough to make the cheaper two-sweep approximation
misleading; the joint optimum is within noise of pasting the two 1D optima together, not
meaningfully better."

Remaining open threads: steps-to-threshold is now reported but not yet used as an
optimization target (agents are still trained to maximize reward, not to minimize
measurements-to-acceptable). Pinning down the plateau's true peak precisely (both within a
single sweep axis and across the joint grid) would need roughly 4x today's seed count per
point (variance shrinks with the square root of seed count, and the CIs above need to
roughly halve to separate the top few cells) — a reasonable next run if the exact values
ever mattered more than "somewhere in a broad, boring middle range, not at the extremes."
The interaction the joint sweep did find (wide RBFs hurting more as center count grows) was
only tested at one grid resolution (levels=150); whether it gets stronger at even finer
resolutions, or whether a sharper reward landscape (narrower `_sigma` in `env.py`) makes the
sequential-1D approximation break down for real, are both untested.

## License

MIT — see [LICENSE](LICENSE).
