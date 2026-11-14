# ShipHunt small-network data-efficiency loop: results

eurisko study, 2026-10-10. Written by an Opus 5.5 subagent for eurisko. Spec: `/home/agent/workspaces/mats/outbox/shiphunt-smallnet-spec-2026-10-10.md`. Review: `SPEC-REVIEW.md`. Pre-registration and deviations: `PREREG.md`. Experiment log: `LOOP.md`. All paths are relative to `/home/agent/workspaces/eurisko-study/li-net/smallnet/`. **Status: done.** 31 loop experiments scored on validation; the champion scored once on TEST against both plain arms. Zero smallnet VMs remain (20:33Z). GPU total $34.83.

## 1. TL;DR

- **Champion: reveal-chain TD with 8 levels ("rbtd8", invented in this loop).** For each cell, the forecast is trained toward the network's own forecast for that cell after more of the same board's true cells are revealed, in 8 nested levels; the deepest level trains on the true board. It uses only the realised board and the model's own outputs, so it is **real-world-valid**. It was selected on validation by the pre-registered rule from 31 experiments.
- **On TEST (5 seeds) it beats both plain arms at every board count.** KL is 0.69 / 0.44 / 0.44 / 0.52 / 0.49x k0's at 100 / 300 / 1k / 3k / 10k boards, and every 95% interval is below 0. It matches k0's 10k-board accuracy with **about 2.8k boards (0.28 of the data, CI 0.20-0.37)**. With 10k boards it beats k0 trained on 30k (0.20 vs 0.27e-3). At 30k boards (3 seeds) it reaches 0.16e-3 against k0's 0.27e-3 (0.59x); two of those three runs were still improving at the step cap. The price is 14-22x more training compute per board.
- **Spec decision rule:** the champion passes (rule b: better than k0 at every N; rule a's point estimate is 0.28, its CI bound 0.37). The spec's own Bayes-net arms **fail**: k = 1 and k = 2 (learned or fixed parents) sit at 0.98-1.20x k0 on validation and never reach k0's 10k-board KL.
- **What carries the gain:** lower-variance targets. The same network trained on exact posteriors (evaluation-only) is 3-30x better than every arm, so target noise is the bottleneck. Feeding true cells in as inputs already helps: masked has S_small -0.35, and the full true-value chain -0.62, nearly as good at small N but not at 10k (0.95x). Making the model's own conditional forecasts the target (rbtd) helps most and holds up at 10k. TD across evidence stages alone gives 0.77-0.81x at 300-3k boards. The martingale penalty hurts at large N, and plain regularisation does not help.
- **Scale and cost:** 31 loop experiments (465 runs), 98 calibration runs and 81 final runs. GPU: **$34.83** at the on-demand ceiling rate for 49 rentals and 38 VM-hours (GPU-LOG.md; spot is billed lower), well under the $200 cap. Zero smallnet VMs at the end (gcloud, 20:33Z, checked twice).

## 2. What ran

- **Task.** ShipHunt "hard" from the harness, imported unchanged (`lir-agent/harness/env/shiphunt.py`, commit 47077e7): a 5x5 board, ships of length 3, 2 and 2 (11088 placements, 7 occupied cells each), and 6 random probe cells whose reports flip with probability 0.2. Forecasts are made after 0, 2, 4 and 6 probes.
- **Data.** Each training board comes with ONE observed probe trajectory and its true board. N = 100, 300, 1k, 3k or 10k training boards, nested, with the same boards across arms for a given seed. 3000 validation boards and 5000 test boards, with exact posteriors by enumeration for scoring only.
- **Network.** One fixed transformer for every arm: 4 layers, d = 128, 577k parameters including all heads, with a 2-D relative attention bias. Each arm changes only the training target and the head it trains (`netdef.py`).
- **Training.** AdamW, batch 64 boards, all stages of each board, a random board symmetry per board per step. Per-N lr and EMA were calibrated on k0. Every run early-stops on the validation boards' outcomes (log-loss against the true cells), never on the posterior (`smallnet.py`, PREREG.md).
- **Metric.** Single-cell KL(P* || q) per cell, averaged over 25 cells, 4 stages and the validation (or test) boards. Each arm's forecasts come from its own output. Net arms use exact marginals by elimination, averaged over 4 fixed orders.
- **Loop.** 31 experiments, each at 5 board counts x 3 seeds, scored on validation. Wave 1 (16): the spec's arms, Kelvin's two plain arms, the first inventions and plain-regularisation controls. Wave 2a (9): the rbtd neighbourhood and combinations. Wave 2b (6): TD on the best spec arm, EMA-teacher targets, more chains, dropout. Each hypothesis was written in LOOP.md before its runs. Selection on validation only, by S_small = mean seed-paired log(KL / KL_k0) over N = 100, 300, 1000. TEST was read once, after selection, for the champion against the plain arms, with 5 seeds.

## 3. TEST: the champion against the plain arms (5 seeds, scored once)

The champion was fixed on validation at 19:10Z (LOOP.md, "Selection"). For the final comparison it was retrained from scratch as `F-E17-rbtd8`, seeds 0-4, with the final protocol: early stopping on all 3000 validation boards' outcomes, patience 40 at N >= 3000. It was then scored once on the 5000 TEST boards. The plain arms ran the same protocol: `F-k0` (full map, the spec's k = 0) and `F-k0cell` (cell by cell, also the strict plain baseline). Seeds pair the arms (same training boards). Source: `results/test-summary.json` and `results/test-tables-test.md` (from `final_analysis.py`); raw runs in `results/raw/F/`.

KL x 1e3 on TEST (mean over seeds, 95% t-interval half-width; seeds in brackets):

| arm | N=100 | N=300 | N=1000 | N=3000 | N=10000 |
|---|---|---|---|---|---|
| F-k0 | 7.116 +- 0.642 [5] | 4.222 +- 0.746 [5] | 1.455 +- 0.188 [5] | 0.746 +- 0.051 [5] | 0.409 +- 0.055 [5] |
| F-k0cell | 7.083 +- 0.595 [5] | 3.570 +- 0.835 [5] | 1.261 +- 0.121 [5] | 0.668 +- 0.057 [5] | 0.410 +- 0.098 [5] |
| F-E17-rbtd8 | 4.942 +- 0.929 [5] | 1.864 +- 0.430 [5] | 0.642 +- 0.145 [5] | 0.395 +- 0.103 [5] | 0.202 +- 0.027 [5] |

Paired difference vs F-k0 (KL x 1e3, arm minus base, same seeds = same training boards; 95% t-interval) and geometric-mean ratio:

| arm | N=100 | N=300 | N=1000 | N=3000 | N=10000 |
|---|---|---|---|---|---|
| F-k0cell | -0.032 +- 0.090 (x1.00) | -0.652 +- 0.952 (x0.84) | -0.194 +- 0.240 (x0.87) | -0.078 +- 0.078 (x0.89) | +0.001 +- 0.097 (x0.99) |
| F-E17-rbtd8 | -2.174 +- 0.970 (x0.69) | -2.358 +- 0.541 (x0.44) | -0.814 +- 0.174 (x0.44) | -0.351 +- 0.094 (x0.52) | -0.207 +- 0.066 (x0.49) |

**The spec's decision rule on TEST** (`results/test-decision.json`):
- (a) Boards needed to reach k0's 10k-board KL: **0.28 of 10k (about 2.8k boards)**, bootstrap 95% CI [0.20, 0.37]. The point estimate is under 1/3; the strict CI bound misses it narrowly.
- (b) Beats k0 at equal boards across the sweep: **yes, at every N**. All five paired 95% intervals are below 0.
- Verdict: **more data-efficient, by rule (b).** Against cell by cell it is the same: (b) passes, (a) 0.28 [0.18, 0.40].

**Eventually more accurate:** with 10k boards the champion reaches 0.202e-3, half of k0's 0.409e-3 at 10k. It is also below k0 trained on 3x the data (30k boards: 0.268 +- 0.065e-3, 3 seeds). At 30k boards the champion reaches **0.158 +- 0.010e-3** against k0's 0.268 +- 0.065e-3: 0.59x, paired difference -0.110 +- 0.071e-3, 3 seeds. Two of its three 30k runs were still improving when they hit the 30000-step cap, so 0.158 is an upper bound. The champion stays below k0 over the whole range from 100 to 30k boards (ratio 0.44-0.69). The gap narrows a little at 30k (0.59, from 0.49 at 10k), but that point is capped.

**Compute:** the champion spends 6.1e11 training FLOPs per board at N=1k (22x k0's 2.7e10) and 3.3e11 at N=10k (14x). That compute goes into the reveal chain: 8 network passes per board-stage instead of 1. k0 does not get there with more passes over its data: its validation loss turns up once it starts to overfit, and early stopping ends the run there (for example `results/raw/smoke/conv-k0-n100-s0.json`, where KL goes from 0.010 at step 100 to 0.75 at step 1300).

IMG: results/test-curve_kl_vs_boards.png

Where the gain comes from, and secondary metrics (TEST):

KL x 1e3 per stage at N=1000 (test; stage = probes seen 0 / 2 / 4 / 6):

| arm | 0 | 2 | 4 | 6 | mean |
|---|---|---|---|---|---|
| F-k0 | 0.145 | 0.990 | 1.828 | 2.858 | 1.455 |
| F-k0cell | 0.168 | 0.815 | 1.552 | 2.510 | 1.261 |
| F-E17-rbtd8 | 0.082 | 0.426 | 0.809 | 1.249 | 0.642 |

Secondary metrics at N=1000 (test; seed means): outcome log-loss per cell, ECE, joint log-lik of the true board per board-stage (independent / net / autoregressive), training FLOPs per board to the best checkpoint, steps to the best checkpoint.

| arm | log-loss | ECE | joint LL | joint LL (AR) | FLOPs/board | best step |
|---|---|---|---|---|---|---|
| F-k0 | 0.56359 | 0.0063 | -14.09 |  | 2.72e+10 | 1290 |
| F-k0cell | 0.56342 | 0.0061 | -14.09 |  | 2.88e+11 | 2730 |
| F-E17-rbtd8 | 0.56274 | 0.0048 | -14.07 | -9.38 | 6.08e+11 | 3600 |

Exact posterior joint log-lik of the true board: -8.84 per board-stage (the floor any joint model can reach).

Secondary metrics at N=10000 (test; seed means): outcome log-loss per cell, ECE, joint log-lik of the true board per board-stage (independent / net / autoregressive), training FLOPs per board to the best checkpoint, steps to the best checkpoint.

| arm | log-loss | ECE | joint LL | joint LL (AR) | FLOPs/board | best step |
|---|---|---|---|---|---|---|
| F-k0 | 0.56247 | 0.0029 | -14.06 |  | 2.32e+10 | 11000 |
| F-k0cell | 0.56246 | 0.0026 | -14.06 |  | 2.24e+11 | 21240 |
| F-E17-rbtd8 | 0.56228 | 0.0033 | -14.06 | -8.91 | 3.32e+11 | 19640 |

Exact posterior joint log-lik of the true board: -8.84 per board-stage (the floor any joint model can reach).

## 4. The loop on VALIDATION: all 31 experiments

Three seeds each, at N = 100, 300, 1000, 3000 and 10000 boards. Seeds pair the arms: the same seed means the same training boards. "valid": yes = real-world-valid; cond. = valid only if evidence items are exchangeable, so not eligible; NO = trained on the exact posterior. The ratio is the seed-paired geometric-mean KL ratio against E01-k0. S_small, the selection score, is the mean log-ratio over N = 100, 300 and 1k; S_all is the same over all 5 N. Source: `results/summary.json` (from `analyze.py`), raw runs in `results/raw/W*/`.

| tag | valid | KL x1e3 at N=100 / 300 / 1k / 3k / 10k (ratio vs k0) | S_small | S_all | FLOPs per board at N=1k |
|---|---|---|---|---|---|
| E10-refpost | NO | 0.18 (0.03) / 0.13 (0.03) / 0.14 (0.09) / 0.14 (0.18) / 0.07 (0.13) | -3.204 | -2.669 | 2.3e+10 |
| E22-rbtd+perm | cond. | 4.05 (0.57) / 1.49 (0.33) / 0.63 (0.41) / 0.37 (0.47) / 0.24 (0.51) | -0.854 | -0.797 | 3.4e+11 |
| E17-rbtd8 | yes | 5.00 (0.70) / 1.98 (0.43) / 0.71 (0.47) / 0.42 (0.53) / 0.22 (0.49) | -0.650 | -0.659 | 5.5e+11 |
| E12-rbtd | yes | 5.42 (0.76) / 1.89 (0.42) / 0.70 (0.46) / 0.39 (0.50) / 0.39 (0.87) | -0.642 | -0.552 | 1.9e+11 |
| E21-rbtd+td | yes | 5.88 (0.83) / 1.78 (0.39) / 0.70 (0.46) / 0.43 (0.55) / 0.19 (0.42) | -0.628 | -0.671 | 1.5e+11 |
| E09-chain | yes | 4.90 (0.70) / 2.04 (0.44) / 0.81 (0.52) / 0.46 (0.60) / 0.42 (0.95) | -0.616 | -0.484 | 9.8e+10 |
| E27-rbtd+allprefix | yes | 5.39 (0.76) / 2.03 (0.45) / 0.74 (0.49) / 0.48 (0.61) / 0.22 (0.48) | -0.598 | -0.603 | 3.3e+11 |
| E23-rbtd+masked | yes | 5.25 (0.74) / 2.06 (0.45) / 0.76 (0.49) / 0.43 (0.55) / 0.30 (0.69) | -0.598 | -0.551 | 2e+11 |
| E19-rbtd-lam5 | yes | 5.57 (0.79) / 2.09 (0.46) / 0.71 (0.47) / 0.43 (0.55) / 0.29 (0.66) | -0.592 | -0.559 | 1.6e+11 |
| E29-rbtdR3 | yes | 5.27 (0.74) / 2.06 (0.45) / 0.77 (0.51) / 0.42 (0.54) / 0.18 (0.40) | -0.589 | -0.659 | 3.2e+11 |
| E18-rbtdR2 | yes | 5.29 (0.75) / 2.01 (0.45) / 0.81 (0.53) / 0.40 (0.52) / 0.18 (0.40) | -0.577 | -0.661 | 2.8e+11 |
| E28-masked+td | yes | 5.82 (0.82) / 1.90 (0.42) / 0.79 (0.52) / 0.44 (0.57) / 0.26 (0.57) | -0.574 | -0.569 | 8.9e+10 |
| E26-rbtd-teacher | yes | 5.68 (0.80) / 2.34 (0.52) / 0.75 (0.49) / 0.49 (0.62) / 0.30 (0.68) | -0.532 | -0.490 | 2.5e+11 |
| E30-rbtd+td-teacher | yes | 5.68 (0.79) / 2.35 (0.52) / 0.79 (0.51) / 0.54 (0.70) / 0.26 (0.58) | -0.519 | -0.492 | 2e+11 |
| E20-rbtd+k0 | yes | 6.25 (0.88) / 2.61 (0.57) / 0.83 (0.54) / 0.50 (0.64) / 0.21 (0.48) | -0.435 | -0.497 | 1.3e+11 |
| E31-rbtd-drop3 | yes | 6.62 (0.94) / 2.44 (0.54) / 0.96 (0.63) / 0.65 (0.83) / 0.56 (1.24) | -0.377 | -0.221 | 4.3e+11 |
| E06-masked | yes | 6.82 (0.97) / 2.92 (0.65) / 0.86 (0.56) / 0.44 (0.56) / 0.37 (0.85) | -0.347 | -0.356 | 5.8e+10 |
| E24-masked-R4 | yes | 6.66 (0.94) / 3.26 (0.72) / 1.00 (0.65) / 0.49 (0.64) / 0.26 (0.60) | -0.269 | -0.353 | 9.3e+10 |
| E14-perm | cond. | 6.88 (0.98) / 3.30 (0.73) / 1.20 (0.79) / 0.61 (0.79) / 0.51 (1.15) | -0.192 | -0.136 | 5.8e+10 |
| E07-td | yes | 6.86 (0.98) / 3.50 (0.77) / 1.24 (0.81) / 0.62 (0.80) / 0.40 (0.91) | -0.163 | -0.162 | 2.6e+10 |
| E02-k0cell | yes | 6.89 (0.98) / 3.91 (0.86) / 1.20 (0.79) / 0.72 (0.92) / 0.52 (1.19) | -0.135 | -0.063 | 3.1e+11 |
| E25-perm+td | cond. | 6.78 (0.96) / 3.39 (0.75) / 1.43 (0.93) / 0.66 (0.86) / 0.33 (0.75) | -0.134 | -0.169 | 6.1e+10 |
| E08-mart | yes | 6.76 (0.96) / 3.14 (0.69) / 1.63 (1.07) / 1.15 (1.48) / 0.97 (2.23) | -0.112 | +0.172 | 5.8e+10 |
| E16-k0wd5 | yes | 6.98 (0.99) / 4.51 (1.00) / 1.43 (0.94) / 0.80 (1.03) / 0.55 (1.26) | -0.022 | +0.040 | 3.2e+10 |
| E11-k2marg | yes | 6.93 (0.98) / 4.48 (0.99) / 1.56 (1.02) / 0.80 (1.03) / 0.49 (1.12) | -0.000 | +0.028 | 2.9e+10 |
| E01-k0 | yes | 7.03 (1.00) / 4.50 (1.00) / 1.52 (1.00) / 0.78 (1.00) / 0.44 (1.00) | +0.000 | +0.000 | 2.3e+10 |
| E13-allprefix | yes | 7.11 (1.01) / 4.41 (0.98) / 1.59 (1.04) / 0.79 (1.02) / 0.42 (0.95) | +0.010 | -0.000 | 4.3e+10 |
| E05-k2fixed | yes | 6.12 (0.87) / 4.71 (1.04) / 1.82 (1.19) / 0.77 (1.00) / 0.48 (1.09) | +0.026 | +0.032 | 3.3e+10 |
| E04-k2 | yes | 7.21 (1.02) / 4.57 (1.01) / 1.71 (1.12) / 0.80 (1.03) / 0.53 (1.20) | +0.049 | +0.072 | 2.9e+10 |
| E03-k1 | yes | 6.97 (0.99) / 5.14 (1.13) / 1.59 (1.05) / 0.87 (1.12) / 0.48 (1.09) | +0.052 | +0.071 | 2.5e+10 |
| E15-k0drop3 | yes | 7.32 (1.04) / 4.70 (1.04) / 2.06 (1.35) / 1.14 (1.47) / 0.50 (1.13) | +0.128 | +0.178 | 5e+10 |

Read across the table:
- **Reveal-chain TD (rbtd) variants take the top three eligible places and 8 of the top 10.** At 300 to 3000 boards the top three give 0.39-0.55x k0's KL. They differ by less than their seed noise: E17-rbtd8 (S_small -0.650), E12-rbtd (-0.642), E21-rbtd+td (-0.628).
- **Revealed true cells help even without TD.** E09-chain (the full true-value chain) and E06-masked reach 0.44-0.65x at 300-3000 boards. Adding TD across stages to masked (E28) brings it close to rbtd.
- **TD across evidence stages alone (E07) helps a little**: 0.77-0.81x at 300-3000 boards.
- **The spec's Bayes-net arms do not help.** E03-k1, E04-k2, E05-k2fixed and E11-k2marg sit at 0.98-1.20x of k0's KL at every N (one exception: k2fixed at N=100, 0.87x), so they are the same as or slightly worse than k0.
- **Plain regularisation does not close the gap.** Dropout 0.3 (E15) is worse; weight decay 0.5 (E16) is neutral. The best plain arm is cell by cell (E02, S_small -0.135), which is the strict baseline.
- **The martingale drift penalty (E08)** helps at 300 boards (0.69x) but is 1.5-2.2x worse at 3k-10k, likely because it also shrinks real updates (the penalty pulls q(t+1) and q(t) together).
- **Reordering the observed probes** gives the best score overall when combined with rbtd (E22, S_small -0.854), but it needs exchangeable evidence, so it is reported, not chosen.
- **Training on the exact posterior (E10, reference)** is 3-30x better than everything else. Noise in the 0/1 targets, not capacity, is what limits data efficiency here.

IMG: results/val-curve_kl_vs_boards.png

IMG: results/val-curve_kl_vs_compute.png

In the compute view (same runs, x = training FLOPs to the early-stopping point), rbtd8 buys its data efficiency with compute. To reach k0's 10k-board KL it needs about 2.7k boards but about 6x the FLOPs (about 1e15 against 1.6e14). At 10k boards it goes below anything k0 reaches in the sweep.

Validation tables (KL with 95% t-intervals, paired differences, per stage, secondary metrics): `results/val-tables-val.md`.

## 5. The spec's decision rule for k = 1 and k = 2 (VALIDATION)

The rule, pre-registered: "Structured training is more data-efficient if k=1 or k=2 reaches k=0's 10k-board single-cell KL with <= 1/3 of the boards (CI over seeds), or beats k=0 at equal boards across the sweep." Computed by `analyze.decision` (log-linear interpolation in N, bootstrap over seeds; `results/val-decision.json`):

| arm | boards to reach k0's 10k KL (fraction of 10k) [95% CI] | beats k0 at every N (95% CI below 0)? | verdict |
|---|---|---|---|
| E03-k1 | never within the sweep [0.93, never] | no (paired differences -0.06 to +0.64 x1e-3) | **fails** |
| E04-k2 | never [never, never] | no (+0.02 to +0.19 x1e-3) | **fails** |
| E05-k2fixed (control) | never [0.94, never] | no | fails |
| E11-k2marg (invention: k = 2 loss plus the single-cell head) | never | no | fails |
| for comparison: E17-rbtd8 (champion) | 0.27 [0.20, 0.37] | no at N=100 (-2.0 +- 2.2 x1e-3), yes at 300-10k | point estimate passes; the CI misses 1/3 narrowly with 3 seeds (TEST, 5 seeds: section 3) |
| for comparison: E06-masked | 0.30 [0.24, 0.63] | no at 100 and 10k | point estimate passes, CI does not |

So, as specified (Bayes nets with k parents, single-cell forecasts read from the net), structured training is **not** more data-efficient than k = 0 here. Structure does help when the network is trained to condition on true cells and its own conditional forecasts become the target (rbtd).

## 6. Why it works (and why the Bayes-net arms do not)

- **rbtd** (an invention of this loop, `smallnet.py`, `rbtd`): every arm normally trains the evidence-only forecast q(cell | evidence) on the board's true 0/1 cell, a target with variance p(1-p), about 0.2 per cell. rbtd draws a random cell order and a few cut points. It reveals the first d1 < d2 < ... true cells of the same board as extra inputs, and trains each level's forecast toward the next level's forecast (stop-gradient), with the deepest level trained on the true board. The averages agree (averaging P(cell | evidence, some true cells) over the board's other cells gives P(cell | evidence)), but the next-level forecast is a much less noisy target than 0/1. This is TD learning (Sutton 1988) along a "reveal" axis instead of a time axis. It needs nothing beyond the realised board and the model's own outputs. The real-world analogue: train a question's forecast toward the model's own forecast once related questions have resolved, not only toward the question's own 0/1 outcome.
- **masked / chain** train the same conditionals but keep the noisy 0/1 target for the evidence-only forecast. Structure reaches the network through its inputs, but the target noise stays. masked gets about half of rbtd's gain in log-KL (S_small -0.35 against -0.65). The full chain does nearly as well as rbtd at small N (-0.62) but falls back at 10k (0.95x), likely because its evidence-only forecast is trained only when no cell is revealed (1/25 of its samples).
- **k = 1 / 2 nets** score tables over parent pairs read off a network that never sees true cells. The single-cell forecast comes from inference through the tables. Our reading (not separately tested): the network gets no more evidence per board than k0, and the tables are learned no faster than k0's marginals, so the readout inherits k0's data needs. A measured side fact: even with perfect tables, k = 2 marginals are biased by only 1e-5 nats per cell (SPEC-REVIEW A.3), so the readout is not the problem.
- **Evaluation-only reference:** the same network trained on the exact posterior reaches 0.07-0.18e-3, 3-30x below every arm. So target noise is the bottleneck, and lower-variance targets are what pays.

## 7. Secondary metrics (VALIDATION, N = 1000; `results/val-tables-val.md`)

| arm | log-loss | ECE | joint LL | joint LL (AR) | FLOPs/board | best step |
|---|---|---|---|---|---|---|
| E01-k0 | 0.56526 | 0.0059 | -14.13 |  | 2.25e+10 | 1067 |
| E02-k0cell | 0.56502 | 0.0043 | -14.13 |  | 3.06e+11 | 2900 |
| E03-k1 | 0.56538 | 0.0063 | -14.04 |  | 2.53e+10 | 1150 |
| E04-k2 | 0.56545 | 0.0053 | -14.01 |  | 2.92e+10 | 1067 |
| E05-k2fixed | 0.56570 | 0.0049 | -12.98 |  | 3.28e+10 | 1200 |
| E06-masked | 0.56473 | 0.0042 | -14.12 | -9.92 | 5.84e+10 | 1383 |
| E07-td | 0.56498 | 0.0051 | -14.12 |  | 2.6e+10 | 1233 |
| E28-masked+td | 0.56462 | 0.0038 | -14.12 | -9.79 | 8.94e+10 | 2117 |
| E09-chain | 0.56460 | 0.0042 | -14.11 | -9.57 | 9.82e+10 | 4650 |
| E12-rbtd | 0.56452 | 0.0033 | -14.11 | -9.66 | 1.9e+11 | 2250 |
| E17-rbtd8 | 0.56451 | 0.0037 | -14.11 | -9.41 | 5.46e+11 | 3233 |
| E21-rbtd+td | 0.56451 | 0.0034 | -14.11 | -9.70 | 1.48e+11 | 1750 |
| E10-refpost | 0.56401 | 0.0031 | -14.10 |  | 2.32e+10 | 1100 |

Exact posterior joint log-lik of the true board: -8.86 per board-stage (the floor any joint model can reach).

The rbtd arms are among the best-calibrated learned arms (ECE 0.0033-0.0037 against k0's 0.0059). As joint models of the board they reach autoregressive log-likelihoods of -9.4 to -9.7 nats per board-stage, against -8.9 for the exact posterior and at best -14.1 for any independent-cell model. The k = 2 net with learned parents reaches only -14.0 (-13.0 with fixed nearest parents). The champion's compute per training board at N=1k is 5.5e11 FLOPs, 24x k0's.

## 8. Real-world validity of every arm

| arm | real-world-valid | reason |
|---|---|---|
| k0 (full map), k0cell (cell by cell) | yes | per-cell log score against the realised board only |
| k1, k2, k2fixed, k2marg | yes | tables scored on the realised board's own cells; random cell orders and parent sets of the same board |
| masked, chain | yes | true cells of the same board as inputs, never at test time |
| td | yes | targets are the model's own forecast after the next probes the episode actually showed, plus the realised board |
| mart | yes | label-free penalty on the observed trajectory's forecasts; no noise model, no new evidence |
| rbtd and its variants (levels, R chains, lambda, teacher, +k0, +td, +masked, +allprefix) | yes | targets are the model's own forecasts given more of the SAME board's true cells, ending at the realised board |
| allprefix | yes | all 7 prefixes (0-6 probes) of the observed trajectory, in the order observed |
| perm, rbtd+perm, perm+td | conditional, not eligible | prefixes of a reordering of the observed probes: no new evidence, but valid only if evidence items are exchangeable (true in ShipHunt, false for time-ordered news) |
| refpost | NO | trained on the exact posterior (evaluation-only reference) |

Every run also uses only real-world-available signals for its own decisions. Early stopping uses the validation boards' outcomes (their true cells), never the posterior. The posterior is used only to score.

## 9. Deviations from the plan (full log with times: PREREG.md, "Deviations log")

1. Cell by cell asks 5 random questions per board-stage per training step instead of all 25 (25 ran at 0.89 s/step and would not fit a rental at 10k boards). Every cell is asked equally often in expectation; evaluation asks all 25.
2. Learning rate, weight EMA and evaluation cadence were calibrated per N on k0, picking what is best for k0 (`results/raw/cal`, `results/raw/cal2`): lr 3e-4 at N=100, 1e-3 above; EMA 0.95 / 0.99. Patience 25 evaluations.
3. Arms that reorder the observed probes were made ineligible as champion before any result came in, because they need exchangeable evidence.
4. Wave 2a (rbtd variants) was launched in parallel with wave 1 on a 1-seed smoke signal, to save wall-clock. The experiments are the same either way.
5. The early-stopping signal (outcome log-loss on 1000 validation boards) turned out to be noise-limited at N >= 3000. Loop runs keep that protocol. The final TEST runs use all 3000 validation boards and patience 40 at N >= 3000, for every arm alike (decided before any TEST number was read).
6. The champion was selected with 14 of its 15 validation runs. The missing run (N=10k, seed 0) could not change S_small, nor make S_all >= 0.
7. Infrastructure: 18 of the 27 spot L4 rentals were preempted mid-run (driver logs: remote rc=255); none of the 22 on-demand rentals were. A requeue daemon (`gpu/requeue.py`, at most 2 tries per job) relaunched their missing jobs, and new rentals moved to on-demand VMs via an opt-in `GPU_PROVISIONING=STANDARD` added to `logical-soup/infra/gpu-up.sh` (default unchanged; backup `/tmp/smallnet/gpu-up.sh.bak-1010`). Editing `gpu/launch.sh` while launchers ran broke those launchers' last line, so 6 preempted rentals went unnoticed for about 25 min. Also fixed: the guard's date regex, and phantom reservations left in GPU-LOG.md by failed creates and preemptions (COST-AUDIT lines from `gpu/audit-failed.sh`, `gpu/audit-preempted.sh`). Wall-clock cost of all this: about 1.5 h.

## 10. Cost and clean-up

- **GPU total: $34.83** (smallnet group; per VM the DELETE or COST-AUDIT `cost_usd` in `logical-soup/GPU-LOG.md`, at the on-demand ceiling rate x age; spot is billed lower). Computed by `gpu/cost.py` into `results/gpu-cost.json`; `spend_usd smallnet` gives the same $34.83, with no open reservations.
- By machine: L4 spot, 27 rentals, 14.9 VM-h, $12.92 (18 of them preempted); L4 on-demand, 21 rentals, 22.7 VM-h, $19.69; A100 on-demand, 1 rental (quota fallback), 0.6 VM-h, $2.10; 5 failed creates, $0.12 (audit upper bound). That is 38.1 VM-hours in all.
- Phases: smoke $0.36 and calibration $0.45; waves 1-2b plus requeues, and the final runs, make up the rest (per-VM lines in GPU-LOG.md, VM names `soup-gpu-smallnet-*`).
- Caps: smallnet's $200 group cap was never approached. The project-wide cap (lib.sh `BUDGET_USD=200`, which includes older projects' $95) peaked at about $178 counting reservations, at 18:02Z.
- Clean-up: `gcloud compute instances list --filter='name~soup-gpu-smallnet'` returned 0 at 20:33:37Z and 20:33:54Z. The requeue daemon is stopped. `guard-smallnet.sh` (eurisko's guard, restarted at 15:52Z with the date fix) is still running as pid 641977; it does nothing while no smallnet VM exists (stop it with `kill 641977` when no further smallnet rentals are planned).

## 11. Files

- Plan and log: `SPEC-REVIEW.md` (review, 15 points), `PREREG.md` (pre-registration, pricing, deviations), `LOOP.md` (every experiment's hypothesis before it ran, and the results).
- Code: `data.py` (episodes from the harness env, exact posteriors), `netdef.py` (the network), `smallnet.py` (all arms, training, evaluation), `bninfer.py` (exact Bayes-net marginals, tested against brute force), `analyze.py` (tables, decision rule, plots), `select_champion.py` (the selection rule), `mkjobs.py` / `mkfinal.py` / `missing.py` (job files), `gpu/` (launch, retry, driver, remote runner, requeue daemon, cost audits), `guard-smallnet.sh`.
- Results: `results/runs.csv` (one row per run, validation), `results/summary.json`, `results/val-tables-val.md`, `results/val-decision.json`, `results/test-tables-test.md`, `results/test-decision.json`, PNG curves in `results/`, raw run JSONs in `results/raw/` (copied from `/tmp/smallnet/runs/`).
- Rendered page: `/home/agent/memo-serve/lir-agent/smallnet.html`.
