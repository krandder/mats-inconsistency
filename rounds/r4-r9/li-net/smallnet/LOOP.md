# smallnet autoresearch loop: experiment log

One entry per experiment, written BEFORE it runs (hypothesis, config, what result would change the plan), then
its result (VALIDATION only; TEST is read once at the end). Numbers come from `results/summary.json`
(`python3 analyze.py`), raw runs in `/tmp/smallnet/runs/<wave>/<tag>/*.json`. Selection rule: PREREG.md.

## Wave 1 (written 2026-10-10 16:30Z, before launch): the spec's arms, the two plain arms, first inventions, plain-regularisation controls

All at N = 100, 300, 1000, 3000, 10000 boards, seeds 0-2 (paired boards across arms), training settings from the calibration (PREREG deviations).

| # | tag | hypothesis (what I expect, and why) | what would change the plan |
|---|---|---|---|
| E01 | k0 | plain full map, the reference | - |
| E02 | k0cell | Kelvin's second plain arm; in a small net the query flag only re-routes compute, so I expect it within +-10% of k0, slower | if it beats k0 by >10%, per-question passes are a regime to combine with |
| E03 | k1 | forest net, exact marginals: the per-cell head sees only 1/25 of the cells' loss as a marginal; I expect it to trail k0 at small N | if it wins, structure in the TARGET helps even without a marginal loss |
| E04 | k2 | spec arm; as k1 with richer tables | as E03 |
| E05 | k2fixed | control: if E04 ~ E05, learned parents add nothing | if E05 > E04, the parent-choice head is the bottleneck |
| E06 | masked | revealed true cells teach ship structure from every board: expect a gain at small N (it also regularises) | a clear win makes "conditionals on true cells" the backbone of wave 2 |
| E07 | td | TD(0) across stages lowers target variance (Sutton 1988): expect a gain at small N, and none at 10k | if TD wins, combine TD with the best structured arm (spec) |
| E08 | mart | label-free drift penalty: expect a small effect | drop if ~0 |
| E09 | chain | full true-value chain: the single-cell readout gets 1/25 of the signal: expect it to trail masked | reference only |
| E10 | refpost | exact posterior targets: the floor (smoke: 6x lower KL than k0 at N=1000 after 600 steps) | not a candidate |
| E11 | k2marg | k0 + k = 2 net loss on the same trunk: structure as an auxiliary, single-cell head as readout | if it beats masked, the parent-table head is a better auxiliary than masking |
| E12 | rbtd | invention: train the single-cell forecast toward the network's own forecast given more revealed true cells (Rao-Blackwellised target, TD along a reveal chain): lower variance than the board itself | if it wins, sweep levels and lambda in wave 2 |
| E13 | allprefix | 7 prefixes instead of 4: more evidence states from the same trajectory; small gain expected | combine with the best |
| E14 | perm | reorderings of the observed probes: 64 evidence subsets per board instead of 4; I expect the largest single gain at small N (caveat: needs exchangeable evidence) | if it wins, report the best strictly-valid arm next to it |
| E15 | k0-drop3 | plain k0 with dropout 0.3: is the small-N gap just regularisation? | if it beats the structured arms, they were only regularisers; strict baseline moves |
| E16 | k0-wd5 | plain k0 with weight decay 0.5 | as E15 |

## Wave 2a (written 2026-10-10 16:47Z, launched in parallel with wave 1; driven by the 1-seed smoke signal)

Signal (single seed, N=1000, validation, smoke settings without EMA): rbtd 0.88e-3, masked 1.35e-3, k2 2.46e-3, k0 2.52e-3 (`runs/smoke/conv-*`). Calibration with EMA: k0 1.63e-3 at N=1000, 0.39e-3 at N=10k; masked 0.26e-3 at N=10k (`runs/cal`). Compute is cheap (wave 1 about $8), so wave 2a explores the rbtd neighbourhood now instead of waiting an hour. All at 5 N x 3 seeds.

| # | tag | change vs E12-rbtd | hypothesis | decision it informs |
|---|---|---|---|---|
| E17 | rbtd8 | 8 levels instead of 4 | a finer chain makes each target closer to its level (less bootstrapping bias per hop, lower-variance targets); more compute per board | levels as a compute-for-data knob |
| E18 | rbtdR2 | 2 independent reveal chains per board-stage | averages two Rao-Blackwellised targets: lower gradient variance at 2x compute | the cleanest "compute per sample" test |
| E19 | rbtd-lam5 | lambda 0.5 | mixing the true board into the bootstrap targets reduces bias from a wrong next level, at some variance | pick lambda |
| E20 | rbtd+k0 | + plain per-cell loss on the single-cell head | an outcome anchor for level 0 should help early training | whether rbtd needs an MC anchor |
| E21 | rbtd+td | + plain loss with TD(0) targets across evidence stages (2-D TD: evidence axis and reveal axis) | the spec's "TD applied to the best structured arm" | TD stacks with structure or not |
| E22 | rbtd+perm | evidence prefixes from reorderings of the observed probes | the two best data multipliers combined | strongest exchangeability-dependent candidate |
| E23 | rbtd+masked | + masked-cell auxiliary | more structure signal per board | redundant with rbtd or additive |
| E24 | masked-R4 | masked with 4 masks per board-stage | masked's compute-for-data knob | is masked compute-limited |
| E25 | perm+td | reordered prefixes with TD(0) along each reordered chain (lattice TD) | TD on 7-step chains through 64 evidence subsets | best rbtd-free candidate |

### Observation 17:08Z (validation, wave 1): outcome-based early stopping is noise-limited at large N
At N=10k the validation KL of E06-masked keeps falling to about 0.24-0.28e-3 by step 5000-6000, but the early-stopping signal (outcome log-loss on 1000 validation boards) jitters by about +-1e-4 nats per cell. That is as large as the remaining KL gains, so the runs stopped at best steps 1800 / 2400 / 4200 and report 0.35 / 0.42 / 0.35e-3 (`runs/W1/E06-masked/*n10000*`). k0 is affected less (its curve is still steep there). Effect: large-N numbers in waves 1-2 understate arms that converge slowly in KL. The selection score (N <= 1000) is less affected. Decision: keep the pre-registered protocol for waves 1-2 (comparability). The final TEST runs use, for EVERY arm, early stopping on all 3000 validation boards' outcomes and patience 40 evaluations at N >= 3000 (still outcome-only, real-world-valid). Logged in PREREG before any TEST data is read.

## Wave 2b (written 2026-10-10 17:33Z, before launch; on-demand VMs)

Partial validation picture (1-3 seeds, KL x1e-3, `progress.py`): the rbtd family leads at every N measured so far.
N=100: rbtd 4.34 vs 6.4-7.0 for k0cell, k2, td, perm. N=1000: rbtd 0.65, rbtd+perm 0.70, masked 0.84 vs k0 about 1.6 (calibration runs).
N=3000: refpost 0.14, rbtdR2 0.35, rbtd 0.38, masked 0.46. N=10k: rbtd+k0 0.19, masked-R4 0.24, td 0.34. Among the spec's structured arms (k1, k2, k2fixed, masked), masked is clearly best, so the spec's "TD applied to the best of the above" is masked+td.

| # | tag | change | hypothesis | decision it informs |
|---|---|---|---|---|
| E26 | rbtd-teacher | rbtd targets from the EMA network in eval mode (mean teacher) | the online net's targets carry dropout noise and chase themselves; EMA targets are calmer | whether target noise limits rbtd |
| E27 | rbtd+allprefix | rbtd on all 7 prefixes of the observed trajectory | a strictly valid evidence multiplier stacked on the structure multiplier | strongest strictly valid combination |
| E28 | masked+td | masked + TD(0) across stages | spec item: TD on the best of the spec's structured arms | does TD stack with structure for the spec arms |
| E29 | rbtdR3 | 3 reveal chains per board-stage | more Rao-Blackwellised targets per board: compute for data | the compute-density curve beyond R2 |
| E30 | rbtd+td-teacher | rbtd + TD(0) on the plain head, both with EMA-teacher targets | 2-D TD with stable targets | whether E21's 2-D TD needs stable targets |
| E31 | rbtd-drop3 | rbtd with dropout 0.3 | rbtd's small-N regime may still be variance-limited | regularisation on top of rbtd |

## Final phase, part 1 (launched 2026-10-10 17:53Z): the two plain arms on TEST, 5 seeds
F-k0 (spec k0 = full map) and F-k0cell (cell by cell) are in the final comparison whatever the champion is, and k0cell at N=10k is the slowest run, so they start now to save wall-clock. They use the final protocol (early stopping on all 3000 validation boards' outcomes; patience 40 at N >= 3000) and `--test`. The TEST numbers are not read until the champion is fixed on validation (PREREG). k0 also runs at N=30000 (3 seeds) for the "eventually more accurate" check.

## Selection (2026-10-10 19:10Z, VALIDATION, `select_champion.py results/summary.json`)
Champion by the pre-registered rule: **E17-rbtd8** (reveal-chain TD, 8 levels): S_small -0.650, ratios vs k0 0.70 / 0.43 / 0.47 / 0.53 / 0.52 at N = 100 / 300 / 1k / 3k / 10k. It was selected with 14 of 15 runs. The missing run (N=10k, seed 0, still running) cannot change S_small (N <= 1000), and could make S_all >= 0 only if its KL were more than about 100x k0's. Runner-up E12-rbtd (4 levels, half the compute per step) at S_small -0.642 is statistically indistinguishable. Ineligible but best overall: E22-rbtd+perm (-0.854, reordered evidence). Strict plain baseline: E02-k0cell (S_small -0.135), already in final part 1. The spec's k = 1 / 2 nets: E03-k1 +0.052, E04-k2 +0.049, E05-k2fixed +0.026 (no better than k0).
Final part 2: F-E17-rbtd8, seeds 0-4, N = 100..10k, final protocol, `--test`; N = 30000 seeds 0-2 on a separate rental (for the "eventually more accurate" check against F-k0 at 30k).

## Loop results, all 31 experiments (VALIDATION, 3 seeds each; complete, refreshed 19:25Z from results/summary.json)

valid: yes = real-world-valid; cond. = needs exchangeable evidence (not eligible); NO = uses the exact posterior. Ratio = seed-paired geometric-mean KL ratio vs E01-k0 (same training boards). S_small = mean log-ratio over N=100, 300, 1k (selection score); S_all over all 5 N.

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

Selection confirmed on complete data: E17-rbtd8 S_small -0.650, S_all -0.659 (its N=10k ratio over 3 seeds: 0.49).

## Final TEST result (2026-10-10 20:35Z; read once, after selection; `results/test-tables-test.md`, `results/test-summary.json`)
F-E17-rbtd8 vs F-k0, 5 seeds, paired: KL x1e3 4.94 / 1.86 / 0.64 / 0.40 / 0.20 against 7.12 / 4.22 / 1.46 / 0.75 / 0.41 at N = 100 / 300 / 1k / 3k / 10k (ratios 0.69 / 0.44 / 0.44 / 0.52 / 0.49; every paired 95% interval below 0). It reaches k0's 10k-board KL with 0.28 of the boards (CI 0.20-0.37). Spec rule: passes via (b), (a) passes on the point estimate only. Same picture against cell by cell. At 30k boards (3 seeds): 0.158 vs k0 0.268 (0.59x). Compute per board 14-22x k0's. Report: RESULTS.md, rendered at memo-serve/lir-agent/smallnet.html.
