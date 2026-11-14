# M2 worker 4 pre-registration: tracr ground truth (written 2026-09-28 ~02:05Z, before any run)

Item 4 of M2 ("tracr ground truth only if time allows"). Question: on networks whose circuit is
known by construction, which unit-importance predictors recover the truth, and where does the
single-unit ablation oracle itself fail (redundancy)? This complements w1-w3 (wealth-net, no
ground truth). Code: `M2/w4/tracr_gt.py`; bulky files under `/tmp/traders-net/M2/w4/`.
tracr: https://github.com/google-deepmind/tracr (Lindner et al. 2023, https://arxiv.org/abs/2301.05062),
installed with pip --target under /tmp.

## Substrate
Four compiled RASP programs from `tracr.compiler.lib` (categorical outputs, no LayerNorm):
reverse (vocab abcd, len 6, all 4096 inputs), sort_unique (vocab 1-6, len 6, all 720 distinct-token
inputs), hist (vocab abcd, len 6, all 4096), shuffle_dyck "()" (len 8, all 256). Labels from
`rasp.evaluate`; the compiled model must reach 100% argmax accuracy (checked, else the program is
dropped). Loss = mean cross-entropy over non-BOS positions of softmax(beta * readout), beta = 5
(primary), beta = 20 (secondary, saturation check). Whole input space, so every oracle is exact.
Units: attention heads and MLP hidden neurons.

## Ground-truth classes (seeded augmentation, seeds 1, 2, 3)
- C: compiled units with nonzero write weights that are active on some input (tracr's circuit).
- Z: compiled padding units (zero write weights or never active): used by nothing.
- R: redundant copies. 25% of live neurons per layer (at least 1) and 50% of live heads (at
  least 1) are split into two identical copies with half the output weights each (function
  unchanged). Both copies are causally used by construction.
- D: distractors, per MLP layer max(4, 5% of hidden) neurons and per layer one head with random
  inputs from real residual dims and writes only into 4 new junk dims that nothing reads: true
  effect exactly 0.
- N: noise units, same counts, random inputs, random writes into real residual dims with norm
  0.15 x the mean live write norm of that layer (reduced if base accuracy drops below 100%):
  harmful by design.

## Predictors (at the fixed weights; no training in stage 1)
taylor = -dL/dgate (signed first-order ablation estimate); abs_ag = sum over inputs of |a.g| per
input; gradnorm = ||dL/da||; magnitude = ||w_out|| (neurons) or ||W_V W_O||_F (heads);
conductance = integrated gradients on the unit gates when all units of that granularity in the
layer are scaled together 0 -> 1 (16-step midpoint); shapley = sampled Shapley value of zero-
ablation dloss, 64 permutations, players = units of one granularity in one layer (heads: all
heads of the net); loo_direct = committed-trader settlement at the output (log p(y) minus log p
with only the unit's write to the readout removed; the CURRENCY.lock rule applied to every unit);
oracle_zero itself (as a predictor of the ground-truth classes); random.

## Metrics (per program x seed x granularity; mean +- sd over seeds)
1. Spearman vs exact zero-ablation dloss (all units, and live units C+R+N only); same vs mean ablation.
2. AUROC of "used by construction" (C+R) vs "unused by construction" (Z+D), per predictor
   including oracle_zero. Main ground-truth metric.
3. R recall: fraction of R copies scored above the 90th percentile of Z+D scores.
4. N sign: fraction of N units given a negative score by signed predictors (taylor,
   conductance, shapley, loo_direct, oracle_zero).
5. Redundancy gap: for R pairs, (pair-ablation dloss - sum of single dlosses) / pair dloss.

Amendment 1 (02:20Z, after a smoke run on hist, before the main runs): tracr MLPs have slope ~100, so the
16-step conductance misses completeness by up to 0.4 nats (beta 5); a 256-step version (`conductance256`) is
added alongside. Liveness of last-layer units is checked on non-BOS positions only (BOS cannot reach the loss).
Noise scale is auto-halved from 0.15 until accuracy is 100% (recorded per run in meta_*.json).

## Stage 2 (optional, if time): perturb compiled weights (Gaussian, 0.05 x weight sd, seeds 1-3),
fine-tune 2000 Adam steps (lr 1e-3) on the full input set to get movement (|theta*(theta-theta0)|
summed per unit) and accumulated |a.g| and signed books; re-run metrics 1-3.

## Stop rule / caps
CPU only, at most 3 concurrent jobs (1 thread each), 3 h wall. No stop rule on outcomes (this is
a measurement). If a program fails to compile or reach 100% accuracy, drop it and say so.
Expectation written now: taylor, conductance and shapley reach AUROC ~1 on Z+D (exact zeros);
magnitude fails on D; oracle_zero and loo_direct miss some R copies at beta = 20 (saturation);
loo_direct scores 0 for every unit that writes only to non-output subspaces (earlier layers), so
its AUROC on C is poor: settlement at the output cannot see intermediate traders.

Amendment 2 (02:40Z, before any reverse/dyck result was seen; cost, not outcome): under box load ~90 a
reverse forward on all 4096 inputs took 0.46 s, making Shapley + 256-step conductance ~3 h per seed.
reverse now uses a fixed random subset of 1024 of the 4096 inputs (rng 0; the oracle is exact on that
subset). The secondary beta = 20 pass computes the cheap predictors, oracles and pairs only
(conductance and Shapley NaN). hist and sort_unique beta = 20 rows already contain them (kept).
Stage 2 runs after stage 1, at most 3 jobs.

Amendment 3 (03:49Z, before any stage-2 result was seen; cost, not outcome): the 3 h wall cap is exceeded
(box load 60-90; dyck stage 1 took 47-53 min per seed). Stage 2 runs only hist, sort_unique and reverse
(dyck skipped). Stage-1 dyck augmented nets reach 0.957-0.992 accuracy, not 100%, even at the noise floor
(noise_frac 0.00059; meta_dyck_s*.json); the compiled dyck is 100%. Per the drop rule, dyck is reported
separately as a deviation, and headline pooled numbers are also given without dyck (POOLED_nodyck).
