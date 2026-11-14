# P17 pre-registration: a breadth-by-depth compute market

Written 2026-09-29 before any bank data existed (no val or test outcome had been seen). Its sha256 is recorded in
`PREREG.sha256` and in RESULTS.md. Changes after freezing are listed at the end, with reasons, and never touch the
kill rule.

## Question

A looped solver can spend compute on depth (more steps on one trajectory) or on breadth (more restarts from fresh
random latents). Does a market that prices each next unit of compute per puzzle, using prices set by traders who
are paid on checkable signals, beat EqR's own selection rules at equal total compute (NFE)?

## Solver and data (fixed)

- Solver: the released EqR (Equilibrium Reasoners, Huang, Geng and Kolter, ICML 2026, arXiv:2605.21488) Sudoku-Extreme
  checkpoint `locuslab/EqR-model/sudoku-extreme/eqr.pth` (EMA weights), code github.com/locuslab/EqR @ 134c754,
  eval settings of the released config: noise_scale 0.5, init std 1.0, lambda 0.95. One NFE = one outer EqR step.
- Data: the released Sudoku-Extreme test set (`locuslab/EqR-data`, sudoku-extreme-1k-aug-1000/test, 422,786 puzzles).
  Fixed sample (seed 17): blocks of 256 puzzles; blocks 0-3 = **val** (1,024 puzzles), blocks 4-11 = **test**
  (2,048 puzzles). Everything tunable is fitted or chosen on val; test is touched once.
- Bank: 128 restarts per puzzle with a staircase of depth caps (restart 0-1: 1024, 2-3: 512, 4-7: 256, 8-15: 128,
  16-31: 64, 32-63: 32, 64-127: 16), per step: residual, violations, correctness, confidences, grid hash
  (`bank.py`). Each policy is replayed from the bank with exact NFE accounting (a trajectory's prefix does not
  depend on how far it is later run).

## Checkable signals (what traders may use)

For a candidate (restart r at depth d): violations of its argmax grid (row/col/box duplicates, non-digits, changed
givens; 0 = valid completion), residual score s = mean over the last 3 steps of the per-cell logit change norm (EqR's
convergence score), residual slope log s(d) - log s(d/2), mean and min max-softmax confidence, log2 depth, the
fraction of the puzzle's other candidates with the same grid, and the number of candidates. Never the label; never
EqR's learned q_halt (recorded, unused).

## Policies (all at average budget b NFE per puzzle)

Budgets b in {16, 32, 64, 128, 256, 512, 1024}. **Primary budget b = 64** (the paper's D64B1 compute point, 93.0%,
with headroom to 99.8%). Pooled policies share a budget of 256*b NFE within each block of 256 puzzles; static
policies spend exactly b per puzzle. Every policy's used NFE is recorded per puzzle; no policy may exceed its budget.

Static (D depth x B restarts, D*B = b, restarts 0..B-1, D in {8, 16, ..., 1024} within the staircase):

- **SR**: EqR smallest-residual selection (argmin s at depth D). Split (D, B) chosen on val per budget.
- **MV**: EqR majority vote over the B final grids (most frequent grid; ties to the first inserted). Split on val.
- **FD**: fixed depth, B = 1, D = b (SR trivially).
- **FR**: fixed restarts at D = 16, B = b/16, SR selection.
- **RS**: random split: per puzzle a uniformly random feasible (D, B) with D*B = b (seed 17), SR selection.
- SR+chk (secondary): as SR, but a restart that was valid at any step up to D is returned first.
- Oracle any-correct at the SR split (information only).

Adaptive simple rules (secondary; they share the checker or the residual with the market):

- **SEQ-chk**: pooled restart-until-valid. Round robin over unsettled puzzles; each turn runs the puzzle's next
  restart to depth D*, stopping at the first valid step (cost = steps run). A puzzle settles when a valid grid is
  found. Answer: the valid grid, else smallest residual. D* in {16, 32, 64, 128} chosen on val per budget.
- **RT**: verifier-free pooled rule: as SEQ-chk, but a puzzle settles when a candidate has s < tau; answer smallest
  residual. D* and tau chosen on val per budget.

Market (primary: **M-chk**):

- Actions per unsettled puzzle: SPAWN (next unused restart, run to depth 16; cost up to 16) or EXTEND a candidate
  (depth d to 2d within its cap; cost up to d). Every run stops at the first valid step (cost = steps run).
- Traders: logistic models of P(action ends with a valid grid) on different signal subsets (violations; residual +
  slope; confidences; agreement + count; all signals; base rate), separate pools for EXTEND and SPAWN.
  Fitted on val: first from an exploratory run (random bids) on val blocks at b in {64, 256, 1024}, then refitted
  with on-policy data from the fitted market on val (b = 64 and 256).
- Price of an action = wealth-weighted mean of the traders' probabilities (Kelly bettors in a parimutuel market,
  Beygelzimer, Langford and Pennock 2012). Wealth starts equal, is updated on val by the Kelly rule
  (w_i *= p_i/p if the action settles valid, (1-p_i)/(1-p) if not), and keeps updating online on test: settlement is
  checkable at run time and needs no labels.
- Clearing: an exact greedy rule. Every unsettled puzzle bids its best action by value density price/cost; the
  highest bid that fits the remaining pooled budget is executed; the puzzle re-bids (stale bids are re-priced when
  popped). Stops when no bid fits or all puzzles settle.
- Answer: the valid grid if found, else the candidate with the smallest residual.
- Variants: **M-res** (verifier-free: settlement = s < tau from val; traders predict that; answer smallest residual),
  **M-chk-local** (same prices, fixed per-puzzle budget b: only the split is decided, no pooling), and two
  information-only ablations: M-chk with one trader (all signals, no pool) and M-chk with random prices.

## Kill rule (as given in the brief; pre-registered)

At the primary budget b = 64 on the 2,048 test puzzles: if the paired bootstrap 95% CI (10,000 resamples over
puzzles) of accuracy(M-chk) - accuracy(SR) or of accuracy(M-chk) - accuracy(MV) has its lower bound <= 0, the status
is **No gain** and the work stops (no mazes).

## Secondary test (decides what a pass means)

M-chk uses the constraint checker; SR and MV do not. So a pass of the kill rule alone could be the checker's gain,
not the market's. Secondary test at b = 64: M-chk against the best verifier-aware simple rule (SEQ-chk or SR+chk,
whichever has the higher val accuracy at b = 64), same paired bootstrap. Status:

- kill rule fails: **No gain**.
- kill rule passes, secondary fails: **Checker gain, not a market gain** (no mazes; the market adds nothing over
  restart-until-valid).
- both pass: **Market gain**; then mazes (EqR maze checkpoint) with the same rules.

M-res against SR and MV (verifier-free comparison) is reported at every budget but decides nothing.
Robustness lines (report only): per-block wins over the 8 test blocks; the kill-rule comparisons against the
best static split in hindsight on test.

## Gates (stop automatically; a failed gate = status Problem)

- G1: D16B1 accuracy (restarts 0-1, val blocks 0-1) in 86.4 +- 5 and D64B1 in 93.0 +- 5 (paper Table 4).
- G2: zero (row, step) cases with a valid grid that is not the label.
- G3: projected full-bank cost < $20 at the $0.71/h ceiling (task cap $30, watchdog $27).

## Changes after freezing

### Amendment 1 (2026-09-29 23:47Z, before any market run and before any test-puzzle data)

What happened. The gate unit at noise_scale 0.5 (the setting in "Solver and data" above) failed G1: D16B1 76.1%
(band 81.4-91.4). D64B1 89.5%, G2 (0) and G3 ($8.93) passed. The status was Problem, as pre-registered. At this
point only val blocks 0-1, restarts 0-1 existed; no test block (4-11) had been run and no policy or market had been
simulated on real data.

Change. The bank's path-noise beta (EqR `noise_scale`) is **0.01** instead of 0.5. Nothing else changes: gate bands,
kill rule, secondary test, budgets, policies, sample, seeds and staircase stay as frozen.

Why this matches the authors' protocol rather than tuning to the gate:
- Paper, Section 5.2: "Empirically, mild damping (λ= 0.05) combined with small path noise (β= 0.01) performs best
  among the tested variants. At test time, one can increase β under breadth scaling to foster exploration, analogous
  to temperature scaling." Beta is raised only under breadth scaling; a single-restart run uses the method's 0.01.
- Paper, Table 12 (noise-scale ablation): β=0.01, Eval 86.4, the same number as the D16B1 headline in Tables 3-4.
- Paper, Table 13: restart scaling reported with "fixed β=0.01" (84.28 at S16 to 96.78 at S2048), so breadth at
  0.01 is also part of the paper's protocol.
- Repo @ 134c754: `models/eqr.py:57 noise_scale: float = 0.01`, `config/arch/eqr.yaml:27 noise_scale: 0.01`. The
  released checkpoint's stored arch config has no noise override, so it was trained at 0.01.

Evidence the other way (recorded, not hidden): every eval config in the repo sets noise 0.5, including
`config/eval/depth_breadth.yaml` (halt_max_steps 16, different_init 1; the `scripts/eval.sh` default) and the
README's D64 B128 reproduction. The frozen line above followed those. The paper's D64B128 99.8% was probably measured
at 0.5, so the paper's breadth headroom may not carry over to a 0.01 bank. The paper also reports the best training
checkpoint (Table 17), which may not be the released one.

Order. This amendment and the code change (`bank.py --noise_scale`, now required and recorded in bank_meta.json;
a resumed bank refuses mixed settings) were written before the rerun. The noise-0.5 gate data is kept in
`results/gate_noise05/` and no policy uses it.

Gate procedure (cost only). Plan A first runs a depth-64 pre-check (group 0, val blocks 0-1; `gate.py --quick`
checks G1 and G2), so a failure costs about $0.2. Then G1-G3 run on group 0 at depth 1024, as frozen. The seeds are
the same, so both checks see the same trajectories up to step 64. If G1 fails at 0.01, the status stays Problem and
nothing more is spent.

Budgets. Depth alone reached 99.0% at depth 1,024 in the noise-0.5 gate data, so restarts have little headroom at
high budgets. The frozen budgets already include 16-128, where restarts could matter, and every budget is reported.
