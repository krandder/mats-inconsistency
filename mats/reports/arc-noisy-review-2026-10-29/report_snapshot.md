# Can our methods beat ARC on deep random MLPs with small noise? (width 256, depth 8 and 32)

29 October 2026. Brief from opus-mats, relaying Kelvin: beat or improve on ARC's sample-free entries (cumulant
propagation) and the sampling entries that won the 256 x 32 contest phase, with small activation noise for everyone,
using any family from the noisy-channel program. Noise grid {0.001, 0.003, 0.01, 0.03, 0.1}; 0.001 and 0.003 primary.
Code, logs and raw results: `work/arc-noisy/` (PREREG.md hashed before the main runs, two dated amendments; LEDGER.md
lists every idea and why it died; `final_tables.sh` regenerates every table below from raw results).

## Summary

**Depth 32: no win for either entry type at any noise level up to 0.03, and the obstacle is the depth problem ARC calls
open. Depth 8: one modest win for the hybrid entry, built on ARC's own method.** At sigma <= 0.003 the noise is under 0.5%
of the depth-32 variance and moves the depth-32 means by nMSE 4e-7, so the primary results are about ARC's problem.

Results on the 24 evaluation nets at sigma = 0.001 (nMSE = MSE / per-neuron variance, so plain Monte Carlo with N paths
scores 1/N; medians over nets with bootstrap 95% CIs; FLOPs at flopscope rates, ARC's kprop at its own discounted count):

| | depth 8 | depth 32 |
|---|---|---|
| ARC's best sample-free entries (kprop) | K=2 augmented 5.2e-5 [4.8e-5, 5.9e-5] at 4.7e8 FLOPs; factorized K=3 4.9e-6 [4.3e-6, 5.3e-6] at 1.3e10; factorized K=3 augmented 1.9e-6 [1.8e-6, 2.2e-6] at 1.05e11 | K=2 augmented 1.4e-3 [1.1e-3, 1.6e-3] at 1.9e9; factorized K=3 2.2e-4 [1.5e-4, 2.9e-4] at 2.06e11 |
| best sampler: our reproduction of the Kerdock mutually-unbiased-bases design behind the best published Phase-1 entries | 3.7e-6 at 6.8e10 (the warm-up budget), 9.7e-7 at 2.72e11 (0.24/N) | 3.6e-5 at 6.8e10, 6.7e-6 at 2.72e11 (0.40/N, 2.4x better than Monte Carlo) |
| which type leads | sample-free up to 5e10 FLOPs, the sampler from 6.8e10 | the sampler from 5e9 (parity at 2e9) |
| ours | H2, ARC's kprop estimate averaged with the Kerdock mean under error models fitted on development nets: beats the best sampler at every budget from 1e9 to 2.72e11 (median gain 1.19x to 9.9x, all CIs above 1) and the better of the two entries by 1.04x to 1.75x; 0.99x at 3e11 | nothing: H2 0.92-1.02x of the Kerdock sampler; linear controls with ARC's closures 0.03-0.9x; none of our sample-free families beats kprop |

The depth-8 win holds at sigma = 0.001, 0.003 and 0.01, so the lowest noise level at which anything of ours wins is 0.001
(the lowest tested). It is a hybrid-entry win whose sample-free half is ARC's K=2/K=3: the "new" part is only combining a
description and a sampler by their error models. At depth 32 the only win is at sigma = 0.1: antithetic noise on the
Kerdock design beats every plain sampler at every budget (1.4-2.2x), because noise is then 83% of the output variance. At
sigma <= 0.03 that trick loses (0.5-0.9x).

Why depth 32 is blocked (section 5, measured):
1. The depth-32 means depend on the joint non-Gaussian structure of every intermediate layer. Replacing layer 8, 16 or 24
   by a Gaussian with the exact mean and covariance and then running the real remaining layers leaves nMSE 8e-5 to 5e-4
   at depth 32 (3 nets); keeping the exact per-neuron marginals with Gaussian dependence (our copula/neuron code) does
   not help (2.3e-4). That is 12-75x the Kerdock sampler's error at the Phase-1 budget.
2. One layer at a time, the missing information is fourth-order and per neuron: per-neuron cumulants of order 3-4 cut
   the Gaussian one-step error 19-120x, the same cumulants rebuilt from the top 8 collapse modes only 1.9-4.1x. Carrying
   them exactly is ARC's K = 4, about 1e14 FLOPs at 256 x 32.
3. Hybrids need what descriptions cannot supply. With the exact layer-m mean, a linear control on the Kerdock design
   would gain 1.1-1.4x (m = 12), 1.6-2.0x (m = 16) and 4.1-5.5x (m = 24) at 5e10-3e11 FLOPs; with ARC's factorized K=3 mean,
   its bias turns every layer into a loss (0.03-0.9x). Two-level sampling over depth is worse than one level even with a
   perfect coarse estimator.
4. Noise does not help at the primary levels: an error made at layer l survives to depth 32 the same way up to sigma =
   0.03. Only at sigma = 0.1 does noise change the picture (kprop's depth-32 error falls 13-19x, the Kerdock design's
   efficiency halves, and sample-free and sampling meet).

So a description must be right to about 1e-6 at depth 16 or more before it helps either entry type at depth 32. That is
ARC's open problem (matching sampling at large depth); none of our families gets there. The ARC note's prior of about 5%
for beating ARC at depth 16-32 looks right.

## Progress log

- 00:15Z start. Read the ARC note (outbox/arc-deep-networks-2026-10-29.md), the program, phase R and its review.
- 00:50Z Kelvin's amendment via main: sigma grid {0.001, 0.003, 0.01, 0.03, 0.1}, low end primary; GPU cap $50.
- 01:20Z first GPU truth VM (stopped at 01:43Z: kernel memory-bound, 260 s per net-sigma); relaunched 4.4x faster.
- 01:30-02:20Z dev net 100: variance structure; ARC kprop baselines; recursive moment control tested and killed.
- 02:20Z PREREG.md written and hashed (before any main run of our candidates).
- 02:00-02:55Z oracles on dev nets (mode-cumulant gate, meet-in-the-middle and copula seeds, control bias, error survival,
  active subspace); Kerdock design built and verified; evaluation runs on 24 nets (L4 VMs, CPU queue).
- 02:41Z Amendment 1 (antithetic noise; H1 scoring), hashed before its results.
- 02:55Z code review (GPT-6 via codex); 03:02Z back with 11 findings, 2 high (Kerdock setup undercharged at small
  budgets; H1's equal-cost comparison rescaled rather than measured). Fixed: sampler v2 with 48 measured checkpoints and
  exact costs, per-block Kerdock setup, sigma = 0 forwarding bug, least-squares fit cost charged, exact ARC-counter
  closure costs per depth, checksum and noise-scale checks, three interpretations narrowed.
- 03:20Z Amendment 2 (H2 defined and predicted before any evaluation-net result), hashed. Sampler v2 rerun on a fifth VM
  (L4 stockouts in most zones; one found in us-west1-a).
- 04:33Z all truths in; every lane VM deleted, instance list empty. 04:37Z final tables; claims review (section 8).

## 1. Setup

**Object** (fixed by the brief). ARC's family: ReLU, W_l iid N(0, 2/n) in every layer, no bias, no LayerNorm,
x ~ N(0, I), n = 256. After every ReLU, independent noise N(0, (sigma r_l)^2), r_l the RMS of a_l in the noisy net, fixed
per net by a seeded 2^12-path pilot and treated as part of the net (nobody is charged for it). Estimand: the per-neuron
final post-ReLU means E[a_L], L = 8 and 32, depth 8 being the truncation of the depth-32 net. Evaluation nets: 24 seeds
(0-23); development and fitting nets: 100-111; noiseless control nets: 200-204.

**Truth**: plain Monte Carlo on GPU, 2^26 paths per net and sigma (2^25 for development nets), float32 matmuls with TF32
off, float64 accumulation. Its own variance (1/N on the normalized scale, 1.5e-8) is subtracted from every nMSE. The GPU
used the weights rounded to float32 (largest weight difference 1.5e-8); weight checksums and noise scales are verified on
load (for 39 truth files the VM recomputed r_l because a staged table was incomplete; identical to 2e-16).

**Metric**: nMSE = mean_j (estimate_j - truth_j)^2 / mean_j Var(a_L,j), ARC's convention. Medians and means over nets with
bootstrap 95% CIs (10,000 resamples over nets); ratios of two methods are formed per net first.

**Competitors**, all on the same noisy nets:
- Sample-free: ARC's `mlp_kprop` (main at 93d091a; its TODO.md marks older K >= 2 outputs stale, so current code was
  run), K = 1, 2 (simple and augmented), 3 factorized (simple and augmented), with the noise added exactly (it adds
  (sigma r_l)^2 to the diagonal of the second cumulant and changes no other cumulant; the reviewer checked this to 1e-12
  for every variant). Factorized K = 4 costs about 24 n^4 L^2 = 6.6e12 FLOPs at 256 x 8 by ARC's Table 1, 22x the largest
  budget, so it is outside the comparison. The contest's sample-free entries are not public code; their forum numbers are
  quoted as unreproduced.
- Sampling: plain MC (expected nMSE exactly 1/N), MC with the last noise integrated per neuron, scrambled Sobol, and our
  reproduction of the design behind the best published Phase-1 samplers: the 129 mutually unbiased bases of R^256 from the
  Z4 Kerdock code (built here from GR(4,7) and verified exactly: orthonormal, all cross inner products 1/16), antipodal
  pairs, one Haar rotation per 66,048-point block folded into W_1, stratified chi radii so every point is N(0, I) and the
  estimator is unbiased under noise. It omits the contest entries' pruning, FWHT and fast-multiplication tricks, as do all
  samplers here; with them the published entries were about 3x Monte Carlo at 256 x 32, ours 2.4x.

**Accounting**: budgets 1e9 to 3e11 FLOPs. Samplers and all our methods are charged analytically at flopscope 0.12 float32
rates (matmul M N (2K-1); elementwise 1; exp/log 16; normal draw 16; norm.cdf 96, norm.pdf 54), including every setup
cost (Kerdock rotations, pilots, least-squares fits, closures). kprop is charged with ARC's own counter (torch
FlopCounterMode with their symmetric-tensor discounts, exp counted as 1), which is cheaper for kprop than flopscope rates
(about 1.3x for K=2 by our count); H2's depth-8 result was rechecked with kprop charged 1.3x and holds (1.02-1.77x over the
better of the two entries). A sample-free method has one cost; "best entry of a type at budget B" is the lowest median
nMSE among entries of that type affordable at B. Samplers are measured at 48 checkpoints with exact costs, so every
comparison is at a measured sample count.

**Controls** (noiseless nets 200-204, code checks only). ARC's Fig. 2 trend at width 256 reproduces: advantage over MC at
equal FLOPs for factorized K=3 (simple / augmented) 7,600 / 2,100 at 2 hidden layers, 400 / 110 at 4, 21 / 7 at 8, against
ARC's "around 1,000", "10-100" and "underperform at 8 once K reaches 4". Ours are 1-7x higher at depths 2 and 4; two known
differences push that way (our estimand is the last post-ReLU layer rather than an extra linear output layer, and MC is
charged L rather than L + 1 matmuls), and ARC's figure predates the code change in their TODO.md. K=2 against MC at
256 x 32: advantage 1.6 (full-cost matmuls) to 2.1 (ARC's counter), per net 1.1-3.0, against the contest participant's 1.04
(CI 0.75-1.38): within the 2x criterion.

## 2. At the primary noise levels the noisy nets are ARC's problem

| quantity | sigma = 0.003 | 0.01 | 0.03 | 0.1 |
|---|---|---|---|---|
| share of depth-32 output variance due to noise (dev net 100) | 0.6% | 5.4% | 29% | 83% |
| shift of the true means vs sigma = 0.001, depth 8 (nMSE units, median of 18-19 nets) | 0 (within truth noise) | 3.7e-7 | 3.1e-5 | 3.8e-3 |
| same, depth 32 | 4.0e-7 | 6.1e-5 | 4.8e-3 | 0.45 |

The shift is mostly a common rescaling: noise injects energy at each layer and positive homogeneity carries it forward.
So sigma <= 0.003 changes neither the variance structure nor the means at the level of any error discussed here, while
sigma = 0.01 already moves the depth-32 means by more than the best sampler's full-budget error.

Structure at depth 32 (dev net 100): the top covariance mode is the mean direction (cosine 0.998); the relative norm
fluctuation has SD 0.19 and carries 63% of the variance; participation ratio 2.4 (25 at depth 8). A linear function of h_m explains
R^2 = 0.26 (m = 0), 0.71 (8), 0.83 (16), 0.93 (24), 0.99 (31) of the depth-32 output variance.

## 3. Results on the evaluation nets

Median nMSE over 24 nets, sigma = 0.001 (sigma = 0.003 is within a few percent everywhere; all cells, CIs and means are in
`work/arc-noisy/results/tables.md`).

Depth 8, sample-free (kprop): K=1 7.5e-3 at 1.2e6 FLOPs; K=2 2.1e-4 at 4.1e8; K=2 augmented 5.2e-5 at 4.7e8; factorized
K=3 4.9e-6 at 1.26e10; factorized K=3 augmented 1.9e-6 at 1.05e11.

| depth 8 sampler | 1e9 | 5e9 | 1e10 | 2e10 | 5e10 | 6.8e10 | 1e11 | 2e11 | 2.72e11 | 3e11 |
|---|---|---|---|---|---|---|---|---|---|---|
| MC | 1.1e-3 | 2.2e-4 | 1.1e-4 | 5.4e-5 | 2.2e-5 | 1.6e-5 | 1.1e-5 | 5.4e-6 | 4.0e-6 | 3.6e-6 |
| Sobol | 6.6e-4 | 1.1e-4 | 5.0e-5 | 2.8e-5 | 1.1e-5 | 7.3e-6 | 4.9e-6 | 2.7e-6 | 1.8e-6 | 1.5e-6 |
| Kerdock | 4.9e-4 | 9.0e-5 | 4.0e-5 | 1.7e-5 | 6.8e-6 | 3.7e-6 | 2.7e-6 | 1.5e-6 | 9.7e-7 | 7.8e-7 |
| H2 (ours) | 5.0e-5 | 3.3e-5 | 2.3e-5 | 4.5e-6 | 3.1e-6 | 2.5e-6 | 2.0e-6 | 1.1e-6 | 8.1e-7 | 8.1e-7 |

Depth 32, sample-free (kprop): K=1 2.1e-2 at 4.7e6; K=2 1.5e-3 at 1.65e9; K=2 augmented 1.4e-3 at 1.9e9; factorized K=3
2.2e-4 at 2.06e11 (factorized K=3 augmented not run at depth 32: its cost grows like L^2 from 1.05e11 at depth 8).

| depth 32 sampler | 1e9 | 2e9 | 5e9 | 1e10 | 5e10 | 6.8e10 | 1e11 | 2e11 | 2.72e11 | 3e11 |
|---|---|---|---|---|---|---|---|---|---|---|
| MC | 4.4e-3 | 2.2e-3 | 8.7e-4 | 4.3e-4 | 8.7e-5 | 6.4e-5 | 4.3e-5 | 2.2e-5 | 1.6e-5 | 1.5e-5 |
| Sobol | 3.2e-3 | 1.5e-3 | 6.4e-4 | 2.6e-4 | 4.8e-5 | 4.1e-5 | 2.7e-5 | 1.3e-5 | 9.3e-6 | 7.3e-6 |
| Kerdock | 4.1e-3 | 1.3e-3 | 5.1e-4 | 2.1e-4 | 4.1e-5 | 3.6e-5 | 2.1e-5 | 1.0e-5 | 6.7e-6 | 5.4e-6 |

MC with the last noise integrated is 1-3% worse than plain MC at sigma <= 0.01 (its readout costs more than the variance
it removes). Noise trend (median nMSE; Kerdock at 2.72e11 with nMSE x N; C6 = antithetic noise on Kerdock, per-net gain at
2.72e11):

| depth | sigma | K=2 augmented | factorized K=3 | factorized K=3 augmented | Kerdock at 2.72e11 | C6 gain [95% CI] |
|---|---|---|---|---|---|---|
| 8 | 0.001 | 5.2e-5 | 4.9e-6 | 1.9e-6 | 9.7e-7 (0.24) | - |
| 8 | 0.01 | 5.2e-5 | 4.9e-6 | 1.9e-6 | 9.6e-7 (0.23) | 0.52 [0.50, 0.54] |
| 8 | 0.03 | 5.1e-5 | 4.6e-6 | 1.8e-6 | 1.1e-6 (0.26) | 0.58 [0.56, 0.60] |
| 8 | 0.1 | 3.9e-5 | 3.1e-6 | 1.2e-6 | 1.7e-6 (0.41) | 1.29 [1.23, 1.35] |
| 32 | 0.001 | 1.4e-3 | 2.2e-4 | - | 6.7e-6 (0.40) | - |
| 32 | 0.01 | 1.3e-3 | 2.1e-4 | - | 6.7e-6 (0.40) | 0.47 [0.43, 0.53] |
| 32 | 0.03 | 8.3e-4 | 1.4e-4 | - | 8.6e-6 (0.51) | 0.88 [0.82, 0.90] |
| 32 | 0.1 | 1.1e-4 | 1.2e-5 | - | 1.5e-5 (0.91) | 2.11 [2.01, 2.36] |

At sigma = 0.1 and depth 32, factorized K=3 (1.2e-5 at 2.06e11) and the Kerdock sampler (1.5e-5 at 2.72e11) meet, and
antithetic noise beats every plain sampler at every budget from 1e9 to 3e11 (median 1.36-2.21x, CIs above 1). At depth 8 and
sigma = 0.1 it wins at 5e10 and from 2e11 to 3e11 (1.12-1.29x) and loses at the other budgets (0.73-0.97x). PREREG had predicted 2-5x at sigma = 0.1 and depth 32 (lower end met).

## 4. Our candidates and why each died

All tests on development nets unless stated; sigma = 0.001. Full numbers in `work/arc-noisy/LEDGER.md`.

| candidate (program item) | test | outcome |
|---|---|---|
| Exact Gaussian closure (full bivariate-normal ReLU moments instead of ARC's truncated expansion) | nMSE at depth 8, 16, 32 vs kprop K=2 | 1.63e-4 vs 1.63e-4 (depth 8), 7.6e-4 vs 6.2e-4 (32): no gain |
| Recursive moment control: exact Gaussian closure whose non-Gaussian corrections come from particles of the true net | depths 8 and 32; 2^10-2^14 iid particles; Kerdock particles | layer 1 is 30x below MC, as predicted, but correction noise and covariance bias accumulate with depth: 2-3x over MC on the same particles at depth 8 (below the Kerdock sampler once its extra cost is counted), none at depth 32, none on Kerdock particles |
| Homogeneous scale-mixture closure (carry the norm mode as an exact scale, by ReLU's positive homogeneity) | as above | 10% better than K=2; closure errors at depth are 69-84% orthogonal to the mean direction |
| Families that follow the collapse: Edgeworth readout with cumulants from the top-k covariance modes (gate: >= 5x with k <= 8) | one-step readout with true moments at layers 16 and 31, 3 nets | top-8 modes give 1.9-4.1x; per-neuron cumulants of the next pre-activations give 19-120x. Failed the gate |
| Neuron/copula code at depth | layer 24 replaced by exact per-neuron marginals with Gaussian dependence, then the real layers | nMSE 2.3e-4 at depth 32, no better than a Gaussian with exact moments (2.1e-4) |
| Meet in the middle, sample-free side: description up to layer m, exact network after | Gaussian with the true mean and covariance at m = 8, 16, 24, then the real remaining layers (2^20 paths), 3 nets | nMSE 8.0e-5 to 5.0e-4 at depth 32 |
| Noise as a resource, sample-free: run the closure on a noisier copy and extrapolate in sigma^2 | arithmetic on the measured shifts | the shift (6e-5 at 0.01 to 0.45 at 0.1) dwarfs the closure gain (2x at 0.03); extrapolation weights amplify closure errors about 2x |
| Description-guided RQMC | active subspace of the depth-32 output, 2,048 paths | top 16 / 64 input directions carry 30% / 71% of E[J^T J] (isotropic 6% / 25%); not built, untested |
| H1: Kerdock + linear control at layer m with a kprop mean (Team Puffi's idea with ARC's closures), 24 evaluation nets, measured equal budgets | kill test: >= 1.5x with CI above 1 on [5e10, 3e11] at depth 32 | depth 32: 0.03-0.9x for every m (K=3f mean). Killed. Depth 8 (layer-2 control): 1.07-1.13x at 5e10, 2e11 and 2.72e11 (CIs above 1), 0.34-0.95x at the other budgets, below H2 everywhere |
| H2: kprop estimate combined with the Kerdock mean, weights from development nets | 24 evaluation nets | depth 8: win (section 6); depth 32: 0.92-1.02x |
| C6: antithetic noise on the Kerdock design | 24 evaluation nets, sigma = 0.01-0.1 | wins only at sigma = 0.1 (depth 32 at every budget; depth 8 at 5e10 and from 2e11); 0.47-0.88x at sigma <= 0.03 |

Bit-based scoring of the families (waste and coverage per layer, as phase R did) was not run: at sigma <= 0.003 those
costs are dominated by the noise precision (phase R measured waste growing as 1/sigma^2, 81 bits per layer at sigma = 0.1
and width 16), and the estimand error already decides each family.

## 5. Where the depth problem sits

1. **The depth-32 means depend on the joint non-Gaussian structure of every intermediate layer.** Replacing layer 8, 16
   or 24 by a Gaussian with the exact mean and covariance (estimated from 2^25 truth paths) and running the real remaining
   layers on 2^20 paths gives nMSE 2.0e-4 / 5.0e-4 / 2.1e-4 at depth 32 on net 100, 1.2e-4 (m = 8) and 8.0e-5 (m = 24) on
   net 101, 2.3e-4 and 1.6e-4 on net 102 (sampling floor 9e-7). Keeping the exact per-neuron marginals and replacing only
   the dependence by a Gaussian copula does not help (2.3e-4 at m = 24). These errors are 12-75x the Kerdock sampler's
   error at the Phase-1 budget; ARC's factorized K = 3, which carries third cumulants from the start, ends near 2e-4 too.
2. **One layer at a time the missing piece is per-neuron fourth-order structure.** With the true mean and covariance of
   h_31, the Gaussian readout of E[a_32] has nMSE 2.1e-5 to 3.6e-5 (3 nets); adding the third and fourth cumulants of each
   pre-activation (estimated from 2^18 paths) brings it to 4.8e-7 to 1.2e-6. Under a Gaussian closure the deep layers each
   inject about 3e-5 (restarted from exact moments at layer 24, it reaches 2.8e-4 by layer 32). Carrying per-neuron fourth
   cumulants exactly is ARC's K = 4, about 1e14 FLOPs at 256 x 32.
3. **Hybrids inherit the same limit.** The Kerdock design integrates input polynomials up to degree 5 exactly, which is
   where the linear predictability of shallow layers lives; what it leaves is predictable only from deep layers. On the
   24 evaluation nets at sigma = 0.001, a control at layer m with the exact layer-m mean would gain 1.1-1.4x (m = 12),
   1.6-2.0x (m = 16) and 4.1-5.5x (m = 24) over the Kerdock sampler at 5e10-3e11 FLOPs (oracle numbers, not claims; the
   4,096-path pilot it needs costs 1.8e10, so nothing is gained below that). With ARC's factorized K=3 mean the control's
   bias wins: 0.6-0.9x at m = 4, 0.4-0.7x at 8, 0.05-0.5x at 12, 0.04-0.13x at 16, 0.03-0.07x at 24. Two-level sampling over depth does not
   help either: with the oracle gains G(m) (the deep level keeps 1/G of the Kerdock variance, the coarse level at most the
   rest), cost x variance is (sqrt(32/G) + sqrt(m (1 - 1/G)))^2 against 32 for one level: 39.7 at m = 8 (G = 1.15), 46.2
   at 16 (G = 2.1), 47.3 at 24 (G = 5.2), worse in every case even with a perfect coarse estimator (this assumes the design
   reduces both levels' variance alike).
4. **Noise does not help at the primary levels.** A 1% mean shift made at layer l keeps 0.009 / 0.10 / 0.11 / 0.25 / 0.74
   of its energy at depth 32 for l = 4 / 8 / 16 / 24 / 28 at sigma = 0.001 (the participants' law prod(1 - arccos(c)/pi)
   predicts 0.012 / 0.032 / 0.12 / 0.37 / 0.62), unchanged within 15% up to sigma = 0.03 and 2-4x smaller at sigma = 0.1
   (one direction per depth, one net). kprop's depth-32 error is flat to sigma = 0.01 and falls 1.6-1.7x by 0.03 and 13-19x
   by 0.1.

So a description has to be right to about 1e-6 at depth 16 or more before it helps either entry type, which is the open
problem ARC states (matching sampling at large depth). Our families do not get there.

## 6. The depth-8 win, read carefully

H2 is the simplest form of "meet in the middle" at the output: a sample-free description's estimate mu_v (ARC's kprop
variant v, chosen per budget) and the Kerdock sampler's mean on the remaining budget, combined with weight
w = (V/N) / (b2_v + V/N), where b2_v (the description's typical nMSE) and V (the sampler's nMSE x N) are medians over
development nets 100-107 and never touch the evaluation nets. At sigma = 0.001, depth 8, 24 evaluation nets (per-net ratio,
median [95% CI]; means agree):

| budget | variant used | gain over the Kerdock sampler | gain over the better of Kerdock and the best affordable kprop |
|---|---|---|---|
| 1e9 | K=2 augmented | 9.89 [9.11, 10.69] | 1.04 [1.04, 1.05] |
| 5e9 | K=2 augmented | 2.69 [2.57, 2.80] | 1.60 [1.55, 1.64] |
| 1e10 | K=2 augmented | 1.75 [1.70, 1.82] | 1.75 [1.70, 1.82] |
| 2e10 | factorized K=3 | 3.79 [3.39, 4.04] | 1.09 [1.07, 1.09] |
| 6.8e10 | factorized K=3 | 1.52 [1.41, 1.58] | 1.52 [1.41, 1.58] |
| 1e11 | factorized K=3 | 1.39 [1.33, 1.47] | 1.39 [1.33, 1.47] |
| 2e11 | factorized K=3 augmented | 1.24 [1.20, 1.37] | 1.24 [1.20, 1.37] |
| 2.72e11 | factorized K=3 | 1.19 [1.16, 1.20] | 1.19 [1.16, 1.20] |
| 3e11 | factorized K=3 | 0.99 [0.93, 1.00] | 0.99 [0.95, 1.00] |

(2e9 and 5e10: 5.09x / 1.15x and 2.20x / 1.58x.) At sigma = 0.003: 1.18-9.6x over the sampler and 1.04-1.79x over the better
of the two, 0.98x at 3e11; sigma = 0.01 is the same. Reading:
- Against the best sampling entry, H2 wins at every budget from 1e9 to 2.72e11, which includes the warm-up (6.8e10) and
  Phase-1 (2.72e11) budgets. Most of that margin is ARC's kprop being better than any sampler at depth 8 below about 5e10.
- Against the better of the two entry types at the same budget, H2 still gains 1.04-1.75x, most where they are comparable
  (5e9-1e11). That part is the combination.
- The small loss at 3e11 comes from checkpoint rounding: with K=3 charged, the sampler part falls one checkpoint (about 20%
  of its paths) below the plain sampler's.
- With kprop charged 1.3x (flopscope-style instead of ARC's discounted count) the result holds: 1.17-9.6x and 1.02-1.77x.
- It is not one of the program's new families, and it does not touch ARC's open problem. Its use is practical: at
  shallow depth, adding a well-weighted sampled correction to ARC's K=3 is worth 1.2-1.8x.
- PREREG amendment 2 predicted >= 1.2x with CIs above 1 from 1e9 to 2e11 and 1.0-1.2x at 2.72e11-3e11 at depth 8 (held), and a
  gain above 1 at depth 32 for 1e9-5e9 (did not appear: 0.94-1.00x).

## 7. Predictions (PREREG) against results

| item | prediction | result |
|---|---|---|
| C1 recursive moment control | (killed before PREREG) | killed: see section 4 |
| C2 scale-mixture closure | killed by the >= 2x rule (seen before PREREG) | killed |
| C3 mode-cumulant closure | gate >= 5x with k <= 8; then fails at depth 8, 30% chance at depth 32 | failed the gate (1.9-4.1x on 3 nets); not built |
| C4 / H1 Kerdock + layer-m control | best m gives 1.1-1.4x at depth 32; kill unless >= 1.5x with CI above 1 on [5e10, 3e11] | 0.03-0.9x at every m with ARC's K=3f mean; oracle 1.6-2.0x (m = 16), 4.1-5.5x (m = 24). Killed |
| C5 noise | kprop at depth 32 within 1.5x of sigma = 0.001 for sigma <= 0.01, >= 2x lower at 0.1; last-noise integration gains < 2% at sigma <= 0.01 | held: within 1.1x at 0.01, 13-19x lower at 0.1; integration is 1-3% worse after its cost |
| C6 antithetic noise | depth 32: 2-5x at 0.1, 0.9-1.2x at 0.03, about 0.55x at 0.01; depth 8 about 1.0x at 0.1 | 2.11x at 0.1 (lower end); 0.88x at 0.03; 0.47x at 0.01; depth 8: 1.29x at 0.1 (at 2.72e11) |
| H2 combination | depth 8: >= 1.2x from 1e9 to 2e11 with CIs above 1, 1.0-1.2x at 2.72e11-3e11; depth 32: > 1 only at 1e9-5e9 | depth 8 held; depth 32 did not (0.94-1.00x at 1e9-5e9) |

## 8. Review

Code review (GPT-6 via codex, read-only, its checks in `/tmp/arcn_review/`): 11 findings. High: (1) Kerdock rotation setup
undercharged at small budgets; (2) H1's equal-cost comparison used a 1/N rescaling instead of a measured comparison.
Medium: OLS fit cost missing; the active-subspace "1.4x cap" was unjustified (withdrawn); H2 left budget unused; sigma = 0
forwarding bug in the sampler (no effect on the noisy runs); oracle floors not calibrated for shared truth (small at the
levels reported); closure costs approximated. Low: oracle wording, float32 truth weights, cost-convention labels. All
fixed or narrowed as listed in the progress log; the reviewer confirmed the noise injection into kprop for every variant,
the Kerdock geometry, the checkpoint logic and the Rao-Blackwell readout.

[CLAIMS REVIEW]

## 9. Compute and files

GPU: 7 on-demand L4 VMs (g2-standard-8) plus 2 creations that failed on L4 stockouts; total 7.5 VM-hours, about $6.5 at
the $0.87/h ceiling, against the $50 cap. Every VM was deleted by the launcher's cleanup; the instance list for this lane
was empty at 04:33Z. Log: `work/arc-noisy/GPU-LOG.md`. CPU: up to 8 processes under nice on this host.

Files (`work/arc-noisy/`): `anet.py` (object, FLOP conventions), `truth_gpu.py`, `truthlib.py`, `kprop_noisy.py`,
`run_kprop.py`, `kprop_costs.py`, `kerdock.py`, `samplers.py`, `gpu_samplers.py` (v2; v1 kept), `hybrid.py` (H1),
`combine.py` (H2), `analyze.py`, `trend.py`, `controls.py`, `gauss_relu.py` and `rmc.py` (killed candidates),
`scratch/*.py` (oracles), `final_tables.sh`, `PREREG.md`, `prereg_hashes.txt`, `LEDGER.md`, `GPU-LOG.md`, `results/`
(kprop outputs, tables.md, h1.md, h2.md, trend.md), `truth/`, `samp/`, `samp_v2/`, `sampanti/`.
