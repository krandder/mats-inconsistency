# Item 1a pre-registration: CostNet+BB seeds on P12 weighted mazes

Written 2026-09-29 ~23:14Z, before any item-1a run. The sha256 is frozen in `PREREG.sha256`. No item-1a test
number exists yet: the only prior number is P12 seed 0. Item 1b (Bansal mazes vs official DT) has its own file,
`PREREG-1b.md`, frozen before any 1b run.

## Question
P12 reported that CostNet+BB solved 27.0% of wide mazes exactly (seed 0, step-9000 checkpoint). CostNet+BB is a
local 3x3 CNN that prices cells, plus an exact shortest-path solver trained by blackbox differentiation. The best
DT arm solved about 3%. Does the 27% hold across many seeds, and does it beat a no-learning floor?

## Arm (P12 recipe verbatim)
- Code is P12's `maze.py flow.py nets.py train.py evaluate.py`; the pre-patch sha256s are in `code/P12_ORIG.sha256`.
- There are three patches, none of which changes the training trajectory:
  - `evaluate.py` writes to `$FU_EVAL_DIR`.
  - `train.py` also saves the step-9000 model.
  - `.part` is written every 500 steps instead of 1000. The RNG is not consumed, so the trajectory is unchanged
    without preemption.
- Training: `train.py --arm CostNet+BB --w 32 --steps 12000 --device cpu --threads 1`. Batch 64, Adam 1e-3,
  λ = 20, Hamming loss. It uses the P12 data (same files, copied to `code/data/`), the same pool, and the trained
  width w = 32.
- Seeds 100–131 (n = 32). They are disjoint from P12's seed 0.
- A known defect is kept verbatim. The blackbox backward pass clamps perturbed costs at 0. A zero-cost target edge
  then makes the solver's walk-back cycle, so that maze's gradient is garbage. On P12's own seed-0 checkpoints I
  measured 7 of 640 backward solves failing at step 9000, 4 of 640 at step 3000, and 0 at init. This was measured
  on training batches only (`/tmp/fu-b2/smoke/diag3.py`). 1a keeps the defect, because 1a replicates the P12
  claim as produced. 1b uses the fixed version.

## Seed count (power, `power.py` / `power.txt`, written before this file)
- One-sided α = 0.025, i.e. a 95% t-CI lower bound.
- Planning mean 17%. This shrinks the single lucky seed by 10 points.
- Bar 8.06%. Per-seed SD 10–15 points.
- At n = 32, power is 0.904 at SD 15 and 0.983 at SD 12. The CI half-width is 3.6–5.4 points.

## Gates (automatic; `code/gate.py 1a`; a failed gate stops all test evaluation)
- **G1, solver/pipeline.** `control_oracle` (the true costs through the same solver and metric code) has
  exact ≥ 0.99 on every evaluation set.
- **G2, training.** The median over the 32 seeds of `gate_W9` exact (final checkpoint) is ≥ 0.60. Failed seeds
  count as 0. At least 90% of seeds must finish training. For reference, P12 seed 0 scored test_W9 78% at step
  9000.
- If either gate fails, I report a Problem, run no test evaluation, and spend nothing more on 1a.

## Primary metric
- Per seed: mean exact over P12's PRIMARY cell, pooled. That cell is `test_W33 ∪ test_W49 ∪ test_W65`, restricted
  to mazes with L > 21 (pool_W9 L_p99 = 21.0). This is the same mask as `P12/analyze.py primary_vec`.
- The final checkpoint (step 12000) is primary.
- Across seeds: the mean, a 95% t-CI, the median, and the min/max.

## Bars (fixed now)
1. **Beats no learning.** The CI lower bound must exceed `control_uniform` primary exact (all costs 1, same
   solver and graph) + 5 points.
2. **Beats DT, labelled.** The CI lower bound must exceed 8.06%, which is the best P12 DT arm (DT+dec, 3.06%) plus 5
   points. P12's DT failed its own validity gate (60% vs the 90% bar at 9x9). This bar is therefore labelled
   "vs an invalid DT baseline" wherever it is quoted. The valid DT comparison is item 1b.
3. **Was 27.0% typical?** Report the fraction of the 32 seeds whose step-9000 primary is ≥ 27.0%, and the seed-0
   value's percentile among the new seeds.

- **Verdict: "replicates"** if bars 1 and 2 hold.
- **Verdict: "does not replicate"** if the CI upper bound is below 8.06%.
- **Otherwise: "inconclusive".**

## Secondaries (reported, no bars)
- Step-9000 primary, with the same statistics.
- Exact by maze size (W9, W15, W25, W33, W49, W65) and on val_W25.
- The shift sets p0.1 and p0.5.
- Valid-path rate.
- Oracle and uniform controls per set.

## Fixed now, not after seeing numbers
- No retuning, no seed exclusion, and no choice of checkpoint by test score.
- A seed that fails to train counts as 0 on the primary. It is reported, not replaced.

## Prior work, named up front
- CostNet+BB is Vlastelica, Paulus, Musil, Martius and Rolínek, "Differentiation of Blackbox Combinatorial
  Solvers", ICLR 2020 (arXiv 1912.02175). It uses the same backward pass, w′ = w + λ·dL/dy with
  ∇ = −(y − y_λ)/λ, and their Warcraft shortest-path task learns vertex costs with a CNN plus Dijkstra and a
  Hamming loss.
- They trained and tested each map size separately, so they did not test size extrapolation.
- The market reading ("local traders price cells, exact clearing aggregates") is a relabelling of this method.

## Compute
- One spot VM `trader-fu-c1-1` (c2d-highcpu-56, ceiling $2.10/h), 33 slots: 32 seeds plus the controls.
- `vm.sh` watchdog; item-1 cap $9, of which 1a may use up to $6.
