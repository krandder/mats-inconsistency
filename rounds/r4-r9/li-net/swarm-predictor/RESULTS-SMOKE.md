# CPU check: pool and settlement math on synthetic chains

eurisko study, 2026-10-05. CPU only: no language model, no GPU. Code in this folder: `pool.py`, `settle.py`, `smoke_pool.py`. Every number below is read from `smoke_results.json`, written by the command below. That run also wrote `smoke_tables.md`, which has all 11 pools, every confidence interval, the fitted stakes and the ECE.

## Command

    cd /home/agent/workspaces/eurisko-study/li-net/swarm-predictor && python3 smoke_pool.py

The defaults are `--preset medium --k-shots 4 --k-claims 6 --n-train 400 --n-test 1000 --grad-eps 20`. The run took 642 s on General (`runtime_s`; 16 cores shared with other work). It needs only numpy and the harness file `env/shiphunt.py`, which `pool.py` imports from `lir-agent/harness` without changing it. The harness's exact Bayes oracle imported without problems (`ShipHunt.posterior()`).

## What was simulated

Boards. Harness preset `medium`: 5 x 4 cells, ships of length 3 and 2, and 6 probes that are each wrong with probability 0.15. Each board also gets 4 noiseless shots at random unprobed cells. There are 28 statements: 20 of the form "cell X is occupied" and 8 of the form "the ship of length L lies horizontally in row r". Revealed cells are not statements, which leaves 24 open statements per board. Exact Bayes is `ShipHunt.posterior()`, split over 510 labelled layouts. Those layouts give 482 occupancy patterns, 28 of which come from two layouts, so the split is needed to make the row statements exact.

Synthetic chain (a stand-in for one LM reasoning trace). The chain reads each of the 10 evidence items (6 probes, 4 shots) with probability a, the overlap knob: a = 1 means every chain sees everything, a = 0.25 means chains see small, partly different subsets. It computes exact Bayes from what it read. It claims 6 open statements, picked with probability proportional to |its posterior - prior| + 0.02. Each claim is q = sigmoid(logit(its posterior) + noise), where the noise is Gaussian on the log-odds with sd sigma_i = noise x U(0.5, 1.5). The chain states one confidence, w = clip(exp(N(0, 0.3^2)) / (1 + sigma_i^2), 0.05, 1). Six scenarios: a in {1, 0.5, 0.25} times noise in {0.5, 1.5}. Each board gets 64 chains, and the n-chain pool uses chains 1..n.

Pools (`pool.py`). Every pool gives an unclaimed statement its rules-only prior.
- (a) mean: the average claim.
- (b) vote: the share of claims with q > 0.5, which is self-consistency's agreement rate used as a probability.
- (c) LL: a log-linear pool per statement, logit p = logit p0 + sum of b (logit q - logit p0). Three stake rules: LL-avg uses b = w / sum of w (averaging, fixed); LL-add uses b = w (adding, fixed); LL-fit uses b = kappa w / m^alpha, with m = the number of claims on the statement and (kappa, alpha) learned on 400 separate boards from realised outcomes only.
- W: the same pool computed over whole legal boards in one step. It is the exact PDG minimiser for claim edges at +b plus prior-marginal copies at -b (STUDY-LIR 3c, 3f).
- (d) D: the LL prices on the claimed statements, projected onto legal boards (`pool.project`). The projection is the distribution over legal boards that minimises KL to the prior plus 20 x KL to each pooled price; it is the IDEAS section 1 fair-price projection.

Metric. Brier against exact Bayes: the mean over 1000 test boards of the mean over open statements of (price - Bayes)^2. It equals the expected excess Brier score over the Bayes ceiling. The rules-only prior scores 53.6 x 10^-3 in every scenario.

## Claims fixed before the run, and verdicts

- C1. W-fit beats both the mean and the vote (paired bootstrap 95% CI of the difference below 0) when a < 1, at n = 4, 16, 64: **12 of 12 cells**.
- C1b. The same for D-fit: **12 of 12**. (Added after a 40/60-board trial run, before the full run; see finding 1.)
- C2. Settlement tracks informativeness: the mean within-board Spearman correlation between a chain's settlement s_i and its oracle informativeness info_i has its 95% CI above 0 at n = 4, 16, 64. Result: **18 of 18** cells settling at W-fit prices, and **18 of 18** settling at D-fit prices. info_i (`settle.informativeness`) is the expected log-score gain of the chain's claims over the prior, computed against exact Bayes. It does not depend on the pool or on the realised board.
- C3. The settlement is minus the derivative of the W pool's log loss with respect to the claim's stake. Finite differences on 20 boards per scenario at n = 16 give a largest error of **3.5e-10**.
- C4. With full overlap (a = 1), the arm-(d) pool does not lose to the mean at n = 16 and 64 (lower end of the 95% CI of "pool minus mean" at or below 0). Result: W-fit **2 of 4**, D-fit **4 of 4**. (Added together with C1b.)

## Brier against exact Bayes, x 1000 (lower is better; bold = lowest of the six shown)

| scenario (overlap a, noise) | n | mean | vote | LL-add | LL-fit | W-fit | D-fit |
|---|---|---|---|---|---|---|---|
| a = 1, noise 0.5 | 1 | 30.8 | 44.7 | 31.9 | 31.4 | 24.1 | **22.4** |
| a = 1, noise 0.5 | 4 | 9.9 | 36.4 | 15.6 | 10.4 | 9.8 | **6.8** |
| a = 1, noise 0.5 | 16 | **2.2** | 26.4 | 37.8 | 2.4 | 6.3 | 2.2 |
| a = 1, noise 0.5 | 64 | 0.7 | 21.2 | 78.7 | 0.7 | 5.6 | **0.7** |
| a = 1, noise 1.5 | 1 | 40.0 | 56.1 | 41.5 | 38.9 | **32.4** | 33.6 |
| a = 1, noise 1.5 | 4 | 26.6 | 54.0 | 21.1 | 20.9 | **16.0** | 18.6 |
| a = 1, noise 1.5 | 16 | 15.5 | 33.3 | 21.9 | 8.1 | 8.8 | **7.1** |
| a = 1, noise 1.5 | 64 | 7.1 | 11.1 | 59.7 | 3.3 | 6.7 | **2.6** |
| a = 0.5, noise 0.5 | 1 | 43.1 | 59.4 | 43.4 | 43.4 | 40.0 | **39.7** |
| a = 0.5, noise 0.5 | 4 | 28.8 | 59.0 | 29.0 | 27.4 | **24.1** | 24.5 |
| a = 0.5, noise 0.5 | 16 | 16.9 | 36.8 | 41.5 | 12.7 | 13.1 | **10.8** |
| a = 0.5, noise 0.5 | 64 | 12.5 | 23.0 | 96.3 | 6.6 | 9.0 | **5.4** |
| a = 0.5, noise 1.5 | 1 | 53.5 | 74.0 | 48.4 | 48.2 | **45.3** | 46.5 |
| a = 0.5, noise 1.5 | 4 | 50.9 | 88.8 | 37.5 | 37.5 | **32.2** | 36.1 |
| a = 0.5, noise 1.5 | 16 | 34.5 | 52.4 | 31.5 | 21.3 | **18.3** | 19.3 |
| a = 0.5, noise 1.5 | 64 | 21.5 | 19.0 | 71.2 | 10.7 | 11.1 | **9.1** |
| a = 0.25, noise 0.5 | 1 | 49.4 | 65.1 | 49.2 | 49.2 | **47.6** | 47.7 |
| a = 0.25, noise 0.5 | 4 | 41.2 | 73.6 | 40.6 | 39.5 | **36.5** | 37.8 |
| a = 0.25, noise 0.5 | 16 | 30.0 | 54.3 | 40.0 | 23.7 | 21.8 | **21.5** |
| a = 0.25, noise 0.5 | 64 | 25.8 | 41.9 | 84.6 | 13.3 | 13.3 | **11.4** |
| a = 0.25, noise 1.5 | 1 | 60.4 | 81.5 | 51.9 | 51.8 | **50.8** | 51.5 |
| a = 0.25, noise 1.5 | 4 | 65.4 | 106.8 | 48.1 | 47.3 | **44.2** | 46.6 |
| a = 0.25, noise 1.5 | 16 | 48.7 | 67.4 | 44.6 | 35.8 | **30.6** | 33.8 |
| a = 0.25, noise 1.5 | 64 | 35.2 | 32.8 | 71.8 | 21.5 | **18.7** | 18.8 |


LL-avg, W-avg, W-add and D-avg are in `smoke_tables.md`. The confidence intervals of the differences are there too: at n >= 4 their half-widths are 0.02-2.2 x 10^-3 (`diff` in `smoke_results.json`).

## Settlement against oracle informativeness: mean within-board Spearman [95% CI]

| scenario (overlap a, noise) | n | s_i settled at D-fit | s_i settled at W-fit | raw-score s_i at W-fit | exact leave-one-out (W-fit) |
|---|---|---|---|---|---|
| a = 1, noise 0.5 | 4 | +0.16 [+0.12, +0.20] | +0.15 [+0.11, +0.19] | +0.08 [+0.04, +0.11] | +0.10 [+0.07, +0.14] |
| a = 1, noise 0.5 | 16 | +0.18 [+0.15, +0.21] | +0.15 [+0.13, +0.18] | +0.08 [+0.06, +0.10] | +0.00 [-0.01, +0.02] |
| a = 1, noise 0.5 | 64 | +0.15 [+0.13, +0.18] | +0.14 [+0.12, +0.16] | +0.06 [+0.04, +0.07] | +0.01 [+0.00, +0.02] |
| a = 1, noise 1.5 | 4 | +0.20 [+0.17, +0.24] | +0.17 [+0.14, +0.21] | +0.11 [+0.07, +0.14] | +0.20 [+0.17, +0.24] |
| a = 1, noise 1.5 | 16 | +0.15 [+0.13, +0.18] | +0.12 [+0.10, +0.14] | +0.07 [+0.06, +0.09] | +0.11 [+0.09, +0.13] |
| a = 1, noise 1.5 | 64 | +0.11 [+0.10, +0.13] | +0.10 [+0.08, +0.11] | +0.06 [+0.05, +0.07] | +0.08 [+0.07, +0.10] |
| a = 0.5, noise 0.5 | 4 | +0.36 [+0.32, +0.39] | +0.30 [+0.26, +0.34] | +0.15 [+0.12, +0.19] | +0.32 [+0.28, +0.35] |
| a = 0.5, noise 0.5 | 16 | +0.28 [+0.26, +0.31] | +0.22 [+0.19, +0.25] | +0.12 [+0.10, +0.13] | +0.18 [+0.16, +0.20] |
| a = 0.5, noise 0.5 | 64 | +0.25 [+0.23, +0.28] | +0.19 [+0.16, +0.21] | +0.09 [+0.08, +0.10] | +0.12 [+0.10, +0.13] |
| a = 0.5, noise 1.5 | 4 | +0.35 [+0.31, +0.38] | +0.30 [+0.27, +0.34] | +0.24 [+0.21, +0.28] | +0.34 [+0.30, +0.37] |
| a = 0.5, noise 1.5 | 16 | +0.29 [+0.27, +0.32] | +0.24 [+0.21, +0.26] | +0.18 [+0.16, +0.20] | +0.25 [+0.23, +0.27] |
| a = 0.5, noise 1.5 | 64 | +0.23 [+0.22, +0.25] | +0.18 [+0.16, +0.20] | +0.12 [+0.11, +0.14] | +0.16 [+0.15, +0.18] |
| a = 0.25, noise 0.5 | 4 | +0.41 [+0.37, +0.44] | +0.37 [+0.34, +0.41] | +0.16 [+0.12, +0.19] | +0.39 [+0.35, +0.42] |
| a = 0.25, noise 0.5 | 16 | +0.36 [+0.33, +0.38] | +0.29 [+0.26, +0.32] | +0.13 [+0.11, +0.15] | +0.26 [+0.24, +0.29] |
| a = 0.25, noise 0.5 | 64 | +0.29 [+0.27, +0.32] | +0.21 [+0.19, +0.24] | +0.09 [+0.08, +0.11] | +0.16 [+0.14, +0.18] |
| a = 0.25, noise 1.5 | 4 | +0.41 [+0.38, +0.44] | +0.40 [+0.36, +0.43] | +0.29 [+0.26, +0.32] | +0.40 [+0.37, +0.44] |
| a = 0.25, noise 1.5 | 16 | +0.38 [+0.36, +0.40] | +0.33 [+0.32, +0.35] | +0.24 [+0.23, +0.26] | +0.34 [+0.32, +0.36] |
| a = 0.25, noise 1.5 | 64 | +0.28 [+0.26, +0.29] | +0.21 [+0.19, +0.23] | +0.15 [+0.13, +0.16] | +0.21 [+0.20, +0.23] |

## What this changes in the proposal

1. Arm (d) is D-fit, not W. Adding log-odds shifts over whole boards treats correlated statements as separate evidence, and cells and ship rows are correlated under the rules. The floor check makes this concrete: one noiseless chain stating exact Bayes on every open statement lands 0.0150 from Bayes through W with fixed stakes, but 0.00015 through the projection (`floor_one_exact_chain_all_open_statements`, 1000 boards). The 40/60-board trial (`/tmp/swarm_smoke_quick.json`, scratch) showed this floor first; that is why D was added and why C1b and C4 exist. With full overlap and low noise (a = 1, noise 0.5), W-fit loses to the mean by 4.1 x 10^-3 (n = 16) and 4.9 x 10^-3 (n = 64). D-fit ties at n = 16 (+0.05 x 10^-3, CI [-0.03, +0.13]) and is 8% lower at n = 64. Where chains carry partly independent information (a < 1), D-fit is 8-57% below the better of mean and vote at n = 4 to 64. D-fit beats W-fit at n = 64 in 5 of 6 scenarios (the sixth is a tie). W-fit is better at n = 4 in 4 of 6 scenarios, by 1.2-3.9 x 10^-3.
2. Pay the likelihood-ratio form of settlement, s_ij = (y_j - mu_j)(logit q_ij - logit p0_j) (STUDY-LIR 3c). Its Spearman with informativeness beats the raw log-score form, (y_j - mu_j) logit q_ij, in 18 of 18 cells; for example, 0.29 against 0.13 at a = 0.25, noise 0.5, n = 16. The two forms differ only by the prior's own bet on the statements the chain was asked about, and that bet says nothing about the chain. Settling at D-fit prices gives a higher Spearman than settling at W-fit prices in 18 of 18 cells.
3. Settlement keeps its signal where exact leave-one-out loses it. With fully redundant, low-noise chains (a = 1, noise 0.5, n = 16), leave-one-out has Spearman +0.003 [-0.014, +0.019], while settlement has +0.15 (W-fit) and +0.18 (D-fit). When 15 other chains say the same thing, removing one chain changes almost nothing, but that chain's bet against the price still pays or loses.
4. Learned stakes detect redundancy. As fully overlapping chains are added, the fitted alpha of W-fit rises toward 1 (averaging): 0.75, 1, 1 at n = 4, 16, 64 with noise 0.5, and 0.375, 0.625, 0.875 with noise 1.5. For noisy chains that read little of the board (a = 0.25, noise 1.5), it stays at 0 (adding) at every n. That is the STUDY-LIR 3f point: averaging is right only for redundant signals. Fixed adding stakes are the worst pools at n = 64: LL-add scores 59.7-96.3 and W-add 57.3-67.5, against the prior's 53.6, because they double count everything.
5. As a probability, the vote share is worse than the mean in 22 of 24 cells, and its ECE is 0.03-0.16 against 0.01-0.09 for the mean (`ece`). Self-consistency's agreement rate is not calibrated.

## Limits

- The synthetic chains have independent log-odds noise around an exact posterior on part of the evidence. LM chains will share systematic errors (all of them ignoring the 15% probe error, for example), and no pool removes a shared error. Their stated confidence may also carry no information. This check shows the math works when chains carry partly independent information. Whether LM chains do is what Gate 0 and the experiment measure.
- Claim sets here follow each chain's own evidence. In the experiment, Refocus chooses them.
- One preset and one board seed range; stakes fit per scenario on 400 boards.
- C3 is an identity of the W pool. D-fit's settlement uses the same formula at D-fit's prices, without that identity.
- Not run: the `hard` preset (12,798 labelled layouts), and rounds, Refocus and the martingale check, which need the LM loop.
