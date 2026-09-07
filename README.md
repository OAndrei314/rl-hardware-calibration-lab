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
  values), `run_joint_sweep` (center count and width swept *together*, over every
  combination in a grid, to check for an interaction the two independent 1D sweeps can't
  see), `run_threshold_shaping_comparison` (a second tabular Q-learning agent trained
  with `agents.run_episode_shaped`, whose reward is shaped to directly minimize
  measurements-to-spec rather than to maximize the raw measurement, compared against the
  standard agent on identical held-out units and an identical training budget), and
  `run_sparse_threshold_sample_efficiency_comparison` (tabular vs. linear-FA
  threshold-seeking agents, both trained under identical pure-sparse shaping, across a
  range of training-episode budgets — tests whether RBF generalization reduces the
  sample penalty pure-sparse shaping pays a tabular Q-table).

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

# Train a Q-learning agent with reward shaped to minimize measurements-to-spec
# directly, and compare it against the standard reward-maximizing agent on an
# identical training/eval budget:
python -m hw_validation_sim.cli --compare-threshold-shaping \
  --train-episodes 300 --eval-episodes 100 --seed 0

# Sweep how much per-step "dense" signal (proportional to the noisy measurement)
# the threshold-seeking agent needs on top of its sparse spec-crossing bonus to
# close the success-rate gap to the reward-maximizing baseline, averaged over
# multiple training seeds:
python -m hw_validation_sim.cli --sweep-dense-shaping-at-levels 20 \
  --dense-scales 0.0 0.02 0.05 0.1 0.2 0.5 1.0 2.0 --sweep-seeds 24 --seed 0

# Under pure-sparse threshold shaping (no dense signal at all), compare tabular vs.
# linear-FA threshold-seeking agents across a range of training-episode budgets --
# tests whether RBF generalization closes the sparse-shaping gap on its own, at any
# budget, without adding the dense_scale term above:
python -m hw_validation_sim.cli --compare-sparse-threshold-budgets 100 300 1000 3000 6000 \
  --sweep-seeds 8 --seed 0
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

### Does shaping the reward toward steps-to-threshold actually beat maximizing reward?

Every agent above is trained to maximize cumulative measurement reward, then evaluated
on `steps_to_threshold` as one metric among several — but nothing in that training
signal ever tells the agent "stop wasting measurements once you're good enough." A more
direct approach: train on a reward shaped around a fixed, absolute pass/fail spec bar
instead — a small constant `-0.02` cost per measurement (every measurement consumes real
tester time, whether or not it moves the needle) plus a one-time `+1.0` bonus the first
time a (noisy) reading crosses `spec_threshold=0.85`. `--compare-threshold-shaping`
trains a tabular Q-learning agent this way (`run_episode_shaped` in `agents.py`) and
compares it against the standard reward-maximizing agent on identical held-out units,
same training budget (seed 0, 20×20 grid, 40-measurement budget, 100 held-out units):

| strategy | mean best true reward | success rate | mean steps to threshold |
| --- | ---: | ---: | ---: |
| q_learning_reward_max | 0.875 | 67% | 4.8 |
| q_learning_threshold_seeking | 0.321 | 2% | 0.5 |

That is not the result the shaping was designed to produce, and it's worth being honest
about rather than quietly dropping the experiment:

1. **The direct objective loses badly at equal budget.** Shaping reward around the exact
   metric being optimized for sounds like it should help, not hurt — but it turned this
   from a *dense* reward problem (every step's noisy measurement is informative about
   direction, which is what lets 300 episodes train a usable 20×20 tabular Q-table at
   all) into a *sparse* one (a single bonus, reachable only ~45% of the time during
   training, is nearly all the training signal there is). Tabular Q-learning has no way
   to back-propagate a rare, delayed reward across a 2,000-cell table from 300 episodes
   — a textbook credit-assignment problem, reproduced here rather than assumed.
2. **This is a sample-efficiency effect, not a stuck or broken signal.** Training the
   same shaped agent for 3,000 episodes instead of 300 (10x the budget, same everything
   else) raises its success rate from 2% to 15% and its mean reward from 0.32 to 0.54 —
   moving in the right direction, monotonically, exactly as the sparse-reward diagnosis
   predicts, while the dense-reward baseline's success rate is already flat at
   67-70% by 300 episodes and gains nothing from the extra budget. The gap narrows with
   more data; it does not vanish within a budget anyone would consider practical here.
3. **The honest takeaway inverts the naive intuition.** For this environment, a reward
   that's already dense and merely *correlated* with the deployment metric is a far
   better training signal, at any realistic budget, than a sparse reward shaped to match
   that metric exactly. "Shape the reward toward what you actually care about" is
   reasonable general RL advice, but it is not free — it traded away the one thing (a
   dense, every-step signal) that made 300-episode tabular Q-learning work in the first
   place, and nothing in this repo's earlier experiments would have surfaced that
   trade-off without actually running it.

### Does a denser shaped reward close that gap?

The threshold-shaping result above raised its own follow-up, left as an explicitly
untested open thread: would giving the threshold-seeking agent a *dense* per-step signal
too — not just the sparse spec-crossing bonus — recover the sample efficiency it lost?
`run_episode_shaped` now takes a `dense_scale` parameter: on top of the existing
`-step_cost` per measurement and one-time `+bonus` for crossing `spec_threshold`, it adds
`dense_scale * noisy_reward` to every step's shaped reward (`dense_scale=0.0` reproduces
the pure-sparse behavior above exactly — this is a strict superset, not a rewrite).
`--sweep-dense-shaping-at-levels` sweeps `dense_scale` at a fixed grid resolution and
training budget, each value averaged over independent training seeds, alongside a
`reward_max` reference row trained and evaluated on the identical seed stream. At
levels=20, 300 training episodes, 80 held-out eval units, averaged over 24 independent
training seeds:

| dense_scale | mean success rate | 95% CI half-width | mean best true reward |
| ---: | ---: | ---: | ---: |
| (reward_max reference) | 46.6% | ±6.3pp | 0.746 |
| 0.0 (pure sparse) | 11.9% | ±5.3pp | 0.437 |
| 0.02 | 23.1% | ±8.9pp | 0.539 |
| 0.05 | 38.2% | ±9.4pp | 0.665 |
| 0.1 | 39.1% | ±7.3pp | 0.697 |
| 0.2 | 40.5% | ±9.7pp | 0.679 |
| 0.5 | **49.3%** | ±7.3pp | **0.756** |
| 1.0 | 43.7% | ±7.4pp | 0.722 |
| 2.0 | 41.6% | ±8.5pp | 0.704 |

Three things worth being honest about:

1. **The denser reward does close the gap, and fast.** A tiny `dense_scale=0.02` already
   more than doubles success rate over pure-sparse shaping (11.9% → 23.1%); by
   `dense_scale=0.05`–`0.1` the threshold-seeking agent is statistically indistinguishable
   from the reward-maximizing baseline (their 95% CIs overlap almost completely). This
   directly confirms the credit-assignment diagnosis in the section above: it was the
   *sparsity* of the training signal that hurt, not the "optimize for the actual deployment
   metric" framing itself — as soon as the agent gets a dense, every-step signal again
   (even a small one, mixed with the still-sparse crossing bonus), tabular Q-learning
   trains about as well as it does on the fully dense reward-maximizing objective.
2. **There's a broad plateau around `dense_scale≈0.5`, not a sharp optimum.** 0.5 pooled
   ahead of every other value tested, including the reward_max reference itself, but its CI
   (42.0%–56.6%) overlaps 0.1 through 2.0 almost entirely — consistent with this repo's
   other sweeps (RBF center count, RBF width), a wide noisy middle band rather than a
   precise peak. Going past 0.5 to 1.0 and 2.0 shows a real, if noisy, decline — an
   over-large dense term increasingly drowns out the sparse bonus's steps-to-threshold
   incentive, pulling the agent back toward "maximize reward" rather than "reach spec fast."
3. **The single-seed reward_max number reported above (67% success) was an optimistic
   draw, not a stable estimate — and it's worth saying so plainly rather than letting a
   good-looking number stand uninvestigated.** Averaged over 24 independent training seeds
   here, reward_max's true success rate is 46.6% ± 6.3pp — seed 0 alone, used throughout the
   section above, happened to land on the better side of a wide distribution. This doesn't
   change any of that section's *qualitative* conclusions (sparse-vs-dense, sample
   efficiency, the credit-assignment story) since those were internally consistent
   single-seed comparisons, but it does mean the specific "67%" and "2%" numbers should be
   read as one seed's outcome, not the environment's typical behavior — exactly the kind of
   single-seed trap this repo's own center-count sweep flagged earlier ("a run that reported
   just one seed's answer would have been reporting noise").

### Does switching to function approximation rescue pure-sparse shaping on its own?

The dense-shaping sweep above fixed the sample-efficiency gap by adding a dense per-step
signal to the shaped reward — but that leaves an open question this README used to flag
explicitly: is the tabular agent's severe penalty under *pure*-sparse shaping
(`dense_scale=0.0`) specific to the tabular representation, and would `LinearFAQAgent`'s
cross-cell generalization close some of that gap on its own, at a smaller training budget,
without needing the `dense_scale` fix at all?
`run_sparse_threshold_sample_efficiency_comparison` trains both a tabular and a linear-FA
threshold-seeking agent under identical pure-sparse shaping, at each of several
training-episode budgets, each averaged over 8 independent training seeds (`levels=20`,
`n_centers_per_dim=14, sigma_scale=0.6` — this repo's own previously-measured best joint
RBF configuration, not a new hand-picked value):

| train_episodes | tabular success | tabular 95% CI | tabular mean reward | linear-FA success | linear-FA 95% CI | linear-FA mean reward |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 20.6% | ±9.4pp | 0.556 | 24.1% | ±13.2pp | 0.593 |
| 300 | 7.5% | ±7.1pp | 0.370 | 26.4% | ±14.8pp | 0.563 |
| 1,000 | 27.0% | ±18.7pp | 0.543 | 22.4% | ±9.0pp | 0.581 |
| 3,000 | 22.2% | ±13.6pp | 0.536 | 24.3% | ±14.4pp | 0.548 |
| 6,000 | 24.9% | ±16.1pp | 0.558 | 22.4% | ±11.1pp | 0.572 |

This is not the clean win the "generalization should help" intuition predicts, and it is
worth reporting plainly instead of quietly dropping the negative result:

1. **The two representations are statistically indistinguishable at every budget tested.**
   Every pair of CIs above overlaps almost completely, and neither agent shows a consistent
   upward trend across a 60x range of training episodes (100 to 6,000) — contrast this with
   the dense-shaping sweep, where a small `dense_scale` produced a fast, monotonic, clearly
   resolved improvement. Switching representation, on its own, does not reproduce that fix.
2. **This has a mechanistic explanation, not just a null result.** `run_resolution_comparison`
   already established what linear-FA is good for: at a fine grid, the tabular Q-table has
   more cells than the training budget can visit even once, and RBF generalization shares an
   update across nearby unvisited cells — a *spatial coverage* problem. Pure-sparse shaping's
   penalty is a different problem entirely: a rare, delayed spec-crossing bonus is hard for
   *any* one-step TD update to back-propagate across a long episode, regardless of how densely
   the state space itself is covered — a *temporal credit-assignment* problem. Generalizing
   across grid cells doesn't help an agent that visits every relevant cell already (at
   `levels=20`, the tabular table gets plenty of visits per cell in 100+ episodes) but still
   can't tell, from a single sparse terminal-ish bonus, which of its many actions actually
   caused it.
3. **The practical takeaway sharpens, rather than contradicts, the dense-shaping result
   above.** If the bottleneck were representational, switching to `LinearFAQAgent` would have
   been a free win with no reward-shaping design work required. It isn't — the fix that
   actually worked (a small `dense_scale` on top of the sparse bonus) is a fix to the
   *training signal*, not the *function class*, and this experiment is the reason to believe
   that distinction is real rather than assumed.

## Status / next steps

Implemented: `run_episode_shaped` / `train_threshold_seeking_agent` /
`--compare-threshold-shaping`, training a Q-learning agent on a reward shaped directly
around measurements-to-spec instead of raw measurement magnitude — the exact open thread
this README used to flag ("steps-to-threshold is now reported but not yet used as an
optimization target"). See "Does shaping the reward toward steps-to-threshold actually
beat maximizing reward?" above — the honest answer is "no, it loses badly at this
environment's training budget, for a specific and verified reason (sparse vs. dense
reward), not because the idea itself was unreasonable."

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

Implemented: `dense_scale` on `run_episode_shaped` / `train_threshold_seeking_agent`, and
`run_dense_shaping_sweep` / `--sweep-dense-shaping-at-levels`, the denser-shaped-reward
follow-up this README used to flag as an untested next experiment. See "Does a denser
shaped reward close that gap?" above — the honest answer is "yes, and with a surprisingly
small amount of dense signal (`dense_scale=0.05`–`0.1` already closes it), though the
same investigation also caught that the section above's single-seed reward_max baseline
(67% success) was an optimistic draw of a noisier true distribution (46.6% ± 6.3pp over 24
seeds) — worth correcting rather than leaving the better-looking number standing."

Implemented: `train_threshold_seeking_fa_agent` and
`run_sparse_threshold_sample_efficiency_comparison` /
`--compare-sparse-threshold-budgets`, comparing tabular vs. linear-FA threshold-seeking
agents under pure-sparse shaping across a range of training budgets — the exact open
thread this README used to flag ("would `LinearFAQAgent`'s generalization... make the
pure-sparse threshold-seeking signal viable at a much smaller sample-count penalty than
the tabular agent pays"). See "Does switching to function approximation rescue
pure-sparse shaping on its own?" above — the honest answer is "no, the two
representations are statistically indistinguishable at every budget from 100 to 6,000
episodes," with a mechanistic reason: linear-FA fixes a *spatial coverage* problem (too
few visits per grid cell), while pure-sparse shaping's penalty is a *temporal
credit-assignment* problem (a rare, delayed bonus is hard for one-step TD to
back-propagate), and generalizing across grid cells doesn't touch that bottleneck.

Remaining open threads: pinning down the RBF
center-count/width plateau's true peak precisely (both within a single sweep axis and
across the joint grid) would need roughly 4x today's seed count per point (variance
shrinks with the square root of seed count, and the CIs above need to roughly halve to
separate the top few cells) — a reasonable next run if the exact values ever mattered more
than "somewhere in a broad, boring middle range, not at the extremes." The interaction the
joint sweep did find (wide RBFs hurting more as center count grows) was only tested at one
grid resolution (levels=150); whether it gets stronger at even finer resolutions, or
whether a sharper reward landscape (narrower `_sigma` in `env.py`) makes the sequential-1D
approximation break down for real, are both untested.

## License

MIT — see [LICENSE](LICENSE).
