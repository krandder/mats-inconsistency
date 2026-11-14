# Can our methods beat ARC on deep random MLPs with small noise? (width 256, depth 8 and 32)

29 October 2026. Brief from opus-mats, relaying Kelvin: beat or improve on ARC's sample-free entries (cumulant
propagation) and the sampling entries that won the 256 x 32 contest phase, with small activation noise for everyone,
using any family from the noisy-channel program. Noise grid {0.001, 0.003, 0.01, 0.03, 0.1}; 0.001 and 0.003 primary.
Code, logs and raw results: `work/arc-noisy/` (PREREG.md hashed before the main runs, two dated amendments; LEDGER.md
lists every idea and why it died; `final_tables.sh` regenerates every table below from raw results).

## Summary

**Depth 32: no win for either entry type at the primary noise levels (sigma = 0.001, 0.003), and the obstacle is the
depth problem ARC calls open. Depth 8: one modest win for the hybrid entry, built on ARC's own method.** At sigma <= 0.003
the noise is under 1% of the depth-32 variance and moves the depth-32 means by nMSE 3e-7, so the primary results are about
ARC's problem.

Results on the 24 evaluation nets at sigma = 0.001 (nMSE = MSE / per-neuron variance, so plain Monte Carlo with N paths
scores 1/N; medians over nets with bootstrap 95% CIs; FLOPs at flopscope rates, ARC's kprop at its own discounted count):

| | depth 8 | depth 32 |
|---|---|---|
| ARC's best sample-free entries (kprop) | K=2 augmented 5.2e-5 [4.8e-5, 5.9e-5] at 4.7e8 FLOPs; factorized K=3 4.9e-6 [4.3e-6, 5.3e-6] at 1.3e10; factorized K=3 augmented 1.9e-6 [1.8e-6, 2.2e-6] at 1.05e11 | K=2 augmented 1.4e-3 [1.1e-3, 1.6e-3] at 1.9e9; factorized K=3 2.2e-4 [1.5e-4, 2.9e-4] at 2.06e11 |
| best sampler: our reproduction of the Kerdock mutually-unbiased-bases design behind the best published Phase-1 entries | 3.7e-6 at 6.8e10 (the warm-up budget), 9.7e-7 at 2.72e11 (0.24/N) | 3.6e-5 at 6.8e10, 6.7e-6 at 2.72e11 (0.40/N, 2.4x better than Monte Carlo) |
| which type leads | sample-free up to 5e10 FLOPs, the sampler from 6.8e10 | the sampler from 5e9 (parity at 2e9) |
| ours | H2, ARC's kprop estimate averaged with the Kerdock mean under error models fitted on development nets: beats the best sampler at every budget from 1e9 to 2.72e11 (median gain 1.19x to 9.9x, all CIs above 1) and the better of the two entries by 1.04x to 1.76x; 0.99x at 3e11 | nothing at sigma = 0.001 or 0.003: H2 at 0.84-1.06x of the Kerdock sampler (its 1-2% point gains at 6.8e10-1e11 pass the median CI test but not the mean one); linear controls with ARC's closures 0.03-0.9x; none of our sample-free families beats kprop |

The depth-8 win holds at sigma = 0.001, 0.003 and 0.01, so the lowest noise level at which anything of ours wins is 0.001
(the lowest tested). It is a hybrid-entry win whose sample-free half is ARC's K=2/K=3: the "new" part is only combining a
description and a sampler by their error models. At depth 32 the wins start above the primary levels and are small until
sigma = 0.1: H2 beats the Kerdock sampler by 1-6% over [6.8e10, 2.72e11] at sigma = 0.03 (and by 2% at 6.8e10 only at
sigma = 0.01), found by the claims review; at sigma = 0.1, antithetic noise on the Kerdock design beats every plain sampler at
every budget (1.4-2.2x), because noise is then 83% of the output variance. At sigma <= 0.03 that trick loses (0.5-0.9x).

Why depth 32 is blocked (section 5, measured; the claims review narrowed the wording):
1. The depth-32 means depend on non-Gaussian joint structure at intermediate layers. Replacing layer 8, 16 or 24 by a
   Gaussian with the (estimated) mean and covariance and then running the real remaining layers leaves nMSE 8e-5 to 5e-4 at
   depth 32 (3 nets); keeping the empirical per-neuron marginals (2^18 paths) with Gaussian dependence (our
   copula/neuron code) does not help (2.3e-4). That is 12-75x the Kerdock sampler's error at the Phase-1 budget. This rules
   out these replacements, not every possible description.
2. One layer at a time, the missing information is per-neuron and fourth-order in the next layer's pre-activations:
   their cumulants of order 3-4 cut the Gaussian one-step error 19-120x, the same cumulants rebuilt from the top 8 collapse
   modes only 1.9-4.1x. Carrying them by ARC's K = 4 costs about 1e14 FLOPs at 256 x 32 (not a lower bound for other ways).
3. Hybrids need what our descriptions cannot supply. With the exact layer-m mean, a linear control on the Kerdock design
   would gain 1.1-1.4x (m = 12), 1.6-2.0x (m = 16) and 4.1-5.5x (m = 24) at 5e10-3e11 FLOPs; with ARC's factorized K=3 mean,
   its bias turns every layer into a loss (0.03-0.9x). When the coarse level has to be sampled itself, the three two-level
   splits we modelled are worse than one level.
4. Noise does not help at the primary levels: an error made at layer l survives to depth 32 about the same way (within 16%)
   up to sigma = 0.03 (one direction per depth, one net). Only around sigma = 0.1 does noise change the picture (kprop's
   depth-32 error falls 13-19x, the Kerdock design's efficiency halves, and sample-free and sampling meet).

So, as a control mean, a description must be right to about 1e-6 at depth 16 or more before it helps at depth 32. That is
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
- 04:33Z all truths in; every lane VM deleted, instance list empty. 04:37Z final tables.
- 04:39-04:50Z claims review (GPT-6, high effort): confirmed the depth-8 H2 and sigma = 0.1 antithetic wins; found that H2
  also wins narrowly at depth 32 at sigma = 0.03 (headline narrowed to the primary levels), refuted my arithmetic rejection
  of noise extrapolation (its exploratory check: +4% at depth 8), and corrected several numbers and wordings (section 8).
- 04:56Z fixes applied; tables regenerated with the complete development set (H2 moved by at most 0.04); report final.
  Open, untested: noise extrapolation of kprop (sample-free) and description-guided RQMC.

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
  estimator is unbiased under noise. (The full direction set is a spherical design of strength 5; with randomized radii and
  partial blocks our version is unbiased but not exact for degree-5 polynomials.) It omits the contest entries' pruning, FWHT and fast-multiplication tricks, as do all
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
| shift of the true means vs sigma = 0.001, depth 8 (nMSE units with the sigma = 0.001 variance, median of 24 nets) | 0 (within truth noise) | 3.9e-7 | 3.3e-5 | 4.1e-3 |
| same, depth 32 | 3.1e-7 | 5.1e-5 | 3.9e-3 | 0.39 |

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
| Homogeneous scale-mixture closure (carry the norm mode as an exact scale, by ReLU's positive homogeneity) | as above | 9% better than our Gaussian closure but 11% worse than K=2 at depth 32; closure errors at depth are 69-84% orthogonal to the mean direction |
| Families that follow the collapse: Edgeworth readout with cumulants from the top-k covariance modes (gate: >= 5x with k <= 8) | one-step readout with true moments at layers 16 and 31, 3 nets | top-8 modes give 1.9-4.1x; per-neuron cumulants of the next pre-activations give 19-120x. Failed the gate |
| Neuron/copula code at depth | layer 24 replaced by empirical per-neuron marginals (2^18 paths) with Gaussian dependence, then the real layers | nMSE 2.3e-4 at depth 32, no better than a Gaussian with the true (estimated) moments (2.1e-4) |
| Meet in the middle, sample-free side: description up to layer m, exact network after | Gaussian with the true mean and covariance at m = 8, 16, 24, then the real remaining layers (2^20 paths), 3 nets | nMSE 8.0e-5 to 5.0e-4 at depth 32 |
| Noise as a resource, sample-free: run the closure on noisier copies and extrapolate in sigma^2 | I rejected it on arithmetic; the reviewer showed that argument was wrong | untested by me. The reviewer's exploratory check (not preregistered, on evaluation nets): K=3f-augmented at sigma = 0.03 and 0.1 extrapolated to 0.001 gives 1.83e-6 vs 1.92e-6 at depth 8 (+4%, CI above 1, cost 2.1e11); no tested depth-32 pair passed. Open |
| Description-guided RQMC | active subspace of the depth-32 output, 2,048 paths | top 16 / 64 input directions carry 30% / 71% of E[J^T J] (isotropic 6% / 25%); not built, untested |
| H1: Kerdock + linear control at layer m with a kprop mean (Team Puffi's idea with ARC's closures), 24 evaluation nets, measured equal budgets | kill test: >= 1.5x with CI above 1 on [5e10, 3e11] at depth 32 | depth 32: 0.03-0.9x for every m (K=3f mean). Killed. Depth 8 (layer-2 control): 1.07-1.13x at 5e10, 2e11 and 2.72e11 (CIs above 1), 0.34-0.95x at the other budgets, below H2 everywhere |
| H2: kprop estimate combined with the Kerdock mean, weights from development nets | 24 evaluation nets | depth 8: win at sigma <= 0.01 (section 6); depth 32: no win at sigma 0.001-0.003 (1-2% point gains, mean CIs include 1); +2% at 6.8e10 at sigma = 0.01; +1-6% over [6.8e10, 2.72e11] at sigma = 0.03 |
| C6: antithetic noise on the Kerdock design | 24 evaluation nets, sigma = 0.01-0.1 | wins only at sigma = 0.1 (depth 32 at every budget; depth 8 at 5e10 and from 2e11); 0.47-0.88x at sigma <= 0.03 |

Bit-based scoring of the families (waste and coverage per layer, as phase R did) was not run: at sigma <= 0.003 those
costs are dominated by the noise precision (phase R measured waste growing as 1/sigma^2, 81 bits per layer at sigma = 0.1
and width 16), and the estimand error already decides each family.

## 5. Where the depth problem sits

1. **The depth-32 means depend on non-Gaussian joint structure at the intermediate layers we tested.** Replacing layer 8,
   16 or 24 by a Gaussian with the true mean and covariance (estimated from 2^25 truth paths) and running the real remaining
   layers on 2^20 paths gives nMSE 2.0e-4 / 5.0e-4 / 2.1e-4 at depth 32 on net 100, 1.2e-4 (m = 8) and 8.0e-5 (m = 24) on
   net 101, 2.3e-4 and 1.6e-4 on net 102 (sampling floor 9e-7). Keeping the empirical per-neuron marginals (2^18 paths) and
   replacing only the dependence by a Gaussian copula does not help (2.3e-4 at m = 24). These errors are 12-75x the Kerdock sampler's
   error at the Phase-1 budget; ARC's factorized K = 3, which carries third cumulants from the start, ends near 2e-4 too.
2. **One layer at a time the missing piece is per-neuron fourth-order structure.** With the true mean and covariance of
   h_31, the Gaussian readout of E[a_32] has nMSE 2.1e-5 to 3.6e-5 (3 nets); adding the third and fourth cumulants of each
   pre-activation (estimated from 2^18 paths) brings it to 4.8e-7 to 1.2e-6. Under a Gaussian closure the deep layers each
   inject about 3e-5 (restarted from exact moments at layer 24, it reaches 2.8e-4 by layer 32). Carrying fourth cumulants
   with ARC's K = 4 costs about 1e14 FLOPs at 256 x 32 (a cost for that algorithm, not a lower bound).
3. **Hybrids inherit the same limit.** The Kerdock design removes most of the low-order input dependence, which is where
   the linear predictability of shallow layers lives; the oracle controls below show that what it leaves is predictable
   mainly from deep layers. On the
   24 evaluation nets at sigma = 0.001, a control at layer m with the exact layer-m mean would gain 1.1-1.4x (m = 12),
   1.6-2.0x (m = 16) and 4.1-5.5x (m = 24) over the Kerdock sampler at 5e10-3e11 FLOPs (oracle numbers, not claims; the
   4,096-path pilot it needs costs 1.8e10, so nothing is gained below that). With ARC's factorized K=3 mean the control's
   bias wins: 0.6-0.9x at m = 4, 0.4-0.7x at 8, 0.05-0.5x at 12, 0.04-0.13x at 16, 0.03-0.07x at 24 (the oracle numbers use the
   truth's 1/N correction although oracle and target share truth paths, which makes them slightly optimistic; the effect
   is about 1-2% at these levels). Two-level sampling over depth, with the coarse expectation itself sampled: if the deep
   level keeps 1/G(m) of the Kerdock variance and the coarse level carries the rest at m/32 of the path cost, cost x variance
   is (sqrt(32/G) + sqrt(m (1 - 1/G)))^2 against 32 for one level: 39.6 at m = 8 (G = 1.15), 46.2 at 16 (G = 2.1), 47.4 at 24
   (G = 5.2). These three modelled splits are worse than one level; with the coarse expectation given for free the gain would
   be G, which is the oracle control above, i.e. ARC's open problem again.
4. **Noise does not help at the primary levels.** A 1% mean shift made at layer l keeps 0.009 / 0.10 / 0.11 / 0.25 / 0.74
   of its energy at depth 32 for l = 4 / 8 / 16 / 24 / 28 at sigma = 0.001 (the participants' law prod(1 - arccos(c)/pi)
   predicts 0.012 / 0.032 / 0.12 / 0.37 / 0.62), unchanged within 16% up to sigma = 0.03 and 1.4-3.9x smaller at sigma = 0.1
   (one direction per depth, one net). kprop's depth-32 error is flat to sigma = 0.01 and falls 1.6-1.7x by 0.03 and 13-19x
   by 0.1.

So a description used this way has to be right to about 1e-6 at depth 16 or more before it helps at depth 32, which is the
open problem ARC states (matching sampling at large depth). Our families do not get there.

## 6. The depth-8 win, read carefully

H2 is the simplest form of "meet in the middle" at the output: a sample-free description's estimate mu_v (ARC's kprop
variant v, chosen per budget) and the Kerdock sampler's mean on the remaining budget, combined with weight
w = (V/N) / (b2_v + V/N), where b2_v (the description's typical nMSE) and V (the sampler's nMSE x N) are medians over
development nets 100-107 and never touch the evaluation nets. At sigma = 0.001, depth 8, 24 evaluation nets (per-net ratio,
median [95% CI]; means agree):

| budget | variant used | gain over the Kerdock sampler | gain over the better of Kerdock and the best affordable kprop |
|---|---|---|---|
| 1e9 | K=2 augmented | 9.89 [9.11, 10.71] | 1.04 [1.04, 1.05] |
| 5e9 | K=2 augmented | 2.69 [2.57, 2.80] | 1.60 [1.55, 1.64] |
| 1e10 | K=2 augmented | 1.76 [1.71, 1.82] | 1.76 [1.71, 1.82] |
| 2e10 | factorized K=3 | 3.78 [3.39, 4.03] | 1.09 [1.07, 1.09] |
| 6.8e10 | factorized K=3 | 1.52 [1.40, 1.58] | 1.52 [1.40, 1.58] |
| 1e11 | factorized K=3 | 1.39 [1.32, 1.47] | 1.39 [1.32, 1.47] |
| 2e11 | factorized K=3 | 1.28 [1.24, 1.30] | 1.28 [1.25, 1.29] |
| 2.72e11 | factorized K=3 | 1.19 [1.16, 1.20] | 1.19 [1.16, 1.20] |
| 3e11 | factorized K=3 | 0.99 [0.93, 1.00] | 0.99 [0.94, 1.00] |

(2e9 and 5e10: 5.09x / 1.15x and 2.19x / 1.58x.) At sigma = 0.003: 1.18-9.6x over the sampler and 1.04-1.80x over the better
of the two, 0.97x at 3e11; at sigma = 0.01: 1.19-10.1x and 1.05-1.75x, 0.96x at 3e11. (Final tables use development nets
100-107; an earlier run with fewer development kprop files gave the same picture within 0.04.) Reading:
- Against the best sampling entry, H2 wins at every budget from 1e9 to 2.72e11, which includes the warm-up (6.8e10) and
  Phase-1 (2.72e11) budgets. Most of that margin is ARC's kprop being better than any sampler at depth 8 below about 5e10.
- Against the better of the two entry types at the same budget, H2 still gains 1.04-1.76x, most where they are comparable
  (5e9-1e11). That part is the combination.
- The small loss at 3e11 comes from checkpoint rounding: with K=3 charged, the sampler part falls one checkpoint (about 20%
  of its paths) below the plain sampler's.
- With kprop charged 1.3x (flopscope-style instead of ARC's discounted count) the result holds: 1.17-9.6x and 1.02-1.77x.
- It is not one of the program's new families, and it does not touch ARC's open problem. Its use is practical: at
  shallow depth, combining ARC's K=2/K=3 with a well-weighted sampled estimate is worth 1.04-1.76x over the better of the
  two at the same budget.
- PREREG amendment 2 predicted >= 1.2x with CIs above 1 from 1e9 to 2e11 at depth 8 (held) and 1.0-1.2x at 2.72e11-3e11
  (held at 2.72e11, 1.18-1.19x; just missed at 3e11, 0.97-0.99x), and a gain above 1 at depth 32 for 1e9-5e9 (not
  significant: point estimates 0.94-1.00x at sigma = 0.001 and up to 1.06x at sigma = 0.003, CIs including 1).

## 7. Predictions (PREREG) against results

| item | prediction | result |
|---|---|---|
| C1 recursive moment control | (killed before PREREG) | killed: see section 4 |
| C2 scale-mixture closure | killed by the >= 2x rule (seen before PREREG) | killed |
| C3 mode-cumulant closure | gate >= 5x with k <= 8; then fails at depth 8, 30% chance at depth 32 | failed the gate (1.9-4.1x on 3 nets); not built |
| C4 / H1 Kerdock + layer-m control | best m gives 1.1-1.4x at depth 32; kill unless >= 1.5x with CI above 1 on [5e10, 3e11] | 0.03-0.9x at every m with ARC's K=3f mean; oracle 1.6-2.0x (m = 16), 4.1-5.5x (m = 24). Killed |
| C5 noise | kprop at depth 32 within 1.5x of sigma = 0.001 for sigma <= 0.01, >= 2x lower at 0.1; last-noise integration gains < 2% at sigma <= 0.01 | held: within 1.1x at 0.01, 13-19x lower at 0.1; integration is 1-3% worse after its cost |
| C6 antithetic noise | depth 32: 2-5x at 0.1, 0.9-1.2x at 0.03, about 0.55x at 0.01; depth 8 about 1.0x at 0.1 | 2.11x at 0.1 (lower end); 0.88x at 0.03; 0.47x at 0.01; depth 8: 1.29x at 0.1 (at 2.72e11) |
| H2 combination | depth 8: >= 1.2x from 1e9 to 2e11 with CIs above 1, 1.0-1.2x at 2.72e11-3e11; depth 32: > 1 only at 1e9-5e9 | depth 8 held except 3e11 (0.97-0.99x); depth 32 not significant at 1e9-5e9; unpredicted small wins at sigma = 0.03 (+1-6%, 6.8e10-2.72e11) |

## 8. Review

Code review (GPT-6 via codex, read-only, its checks in `/tmp/arcn_review/`): 11 findings. High: (1) Kerdock rotation setup
undercharged at small budgets; (2) H1's equal-cost comparison used a 1/N rescaling instead of a measured comparison.
Medium: OLS fit cost missing; the active-subspace "1.4x cap" was unjustified (withdrawn); H2 left budget unused; sigma = 0
forwarding bug in the sampler (no effect on the noisy runs); oracle floors not calibrated for shared truth (small at the
levels reported); closure costs approximated. Low: oracle wording, float32 truth weights, cost-convention labels. All
fixed or narrowed as listed in the progress log; the reviewer confirmed the noise injection into kprop for every variant,
the Kerdock geometry, the checkpoint logic and the Rao-Blackwell readout.

Claims review (GPT-6 via codex, high effort, 04:39-04:50Z; it regenerated every table and rebuilt H2 independently):
- Confirmed: the depth-8 H2 range 1e9-2.72e11 at sigma 0.001, 0.003 and 0.01 with both CI tests; no dev/eval leak; Kerdock is
  the best plain sampler across those ranges; C6 at depth 32 and sigma 0.1 passes at every budget; H1's numbers; the main
  competitor tables; ARC's counter is about 1.3x cheaper than full-cost matmuls for K=2 (1.32-1.34x) but not for factorized
  K=3 (1.01x), so the 1.3x sensitivity is a check, not a repricing.
- High: my depth-32 headline ("no win up to sigma 0.03") was false: H2 wins by 1-6% at sigma 0.03 over 6.8e10-2.72e11 and by
  2% at 6.8e10 at sigma 0.01. Fixed: headline restricted to the primary levels, wins reported.
- Medium: my rejection of noise extrapolation was not a test; its exploratory check found +4% at depth 8 (section 4).
  The multilevel argument assumed a sampled coarse level; reworded. The oracle numbers share truth paths with the target
  (slightly optimistic; stated). "Exact" marginals in the copula oracle are empirical quantiles from 2^18 paths (stated).
  The sampler is unbiased but not exact for degree-5 polynomials once radii are randomized and blocks are partial (stated).
- Low: HSC is 9% better than the Gaussian closure but 11% worse than K=2 (fixed); survival is within 16%, and 1.4-3.9x at
  sigma 0.1 (fixed); the mean-shift table now uses all 24 nets; H2's prediction at 3e11 just missed (0.97-0.99x; fixed);
  Sobol and chi-quantile costs are stipulated allowances (100 FLOPs per coordinate / per point), H1's final control matvec
  is uncharged (2n^2 per estimate, negligible), and H2's checkpoint grid leaves up to about 13% of the budget unused at 3e11.
- Noted here: the PREREG's "what was seen" noise-share line came from an early exploration net with a pre-final weight
  generator; the final weights give the same numbers (section 2).

## 9. Compute and files

GPU: 7 on-demand L4 VMs (g2-standard-8) plus 2 creations that failed on L4 stockouts; total 7.5 VM-hours, about $6.5 at
the $0.87/h ceiling, against the $50 cap. Every VM was deleted by the launcher's cleanup; the instance list for this lane
was empty at 04:33Z. Log: `work/arc-noisy/GPU-LOG.md`. CPU: up to 8 processes under nice on this host.

Files (`work/arc-noisy/`): `anet.py` (object, FLOP conventions), `truth_gpu.py`, `truthlib.py`, `kprop_noisy.py`,
`run_kprop.py`, `kprop_costs.py`, `kerdock.py`, `samplers.py`, `gpu_samplers.py` (v2; v1 kept), `hybrid.py` (H1),
`combine.py` (H2), `analyze.py`, `trend.py`, `controls.py`, `gauss_relu.py` and `rmc.py` (killed candidates),
`scratch/*.py` (oracles), `final_tables.sh`, `PREREG.md`, `prereg_hashes.txt`, `LEDGER.md`, `GPU-LOG.md`, `results/`
(kprop outputs, tables.md, h1.md, h2.md, trend.md), `truth/`, `samp/`, `samp_v2/`, `sampanti/`.

---

# Part 2: rare output events at depth (width 256, depth 4, 8 and 32)

Brief (opus-mats, 29 October ~05:00Z): ARC's other benchmark, tail probabilities of output neurons, at ARC's depth-4
shape and at depths 8 and 32; sigma 0.001 and 0.003 primary, 0.01 secondary; thresholds set so that p spans 1e-4 to
1e-12; competitors ARC's tail method, plain Monte Carlo, splitting without psi and the Kerdock design; ours: splitting and
importance sampling over x with backward functions from the forward description, layer-wise twisted SMC where the noise
gives leverage, deductive tail estimates; metric FLOPs to 50% relative error per event and amortised over the 256
outputs. Code: `work/arc-noisy/tails/`.

## Summary (Part 2)

**One pre-registered win, and it is a readout, not a sampler: Monte Carlo moments read out with a saddlepoint. With its
cost shared over the 256 outputs and the five thresholds per output, it is cheaper than every sampler at ARC's depth 4
at 1e-8 and 1e-10 (37-38x and 55-58x cheaper than splitting per event) and than Monte Carlo at depth 8 at 1e-6
(9.4x), at every noise level tested; within 50% for 69-86% of those events across the three sigma (70-84% at 0.001).
None of the tested backward-function candidates gives a qualifying win, and at depth 32 nothing of ours wins.** A depth-4 win at 1e-12 (83x) holds only under the median
convention the scorer uses (the lower of the two middle values); with the usual midpoint median it is 19x [0, 80], not
a win (P2.3).

The numbers (sigma = 0.001, 20 evaluation nets, outputs 0-7, thresholds set so that p is 1e-4 to 1e-12; FLOPs at part
1's flopscope rates, ARC's kprop at its own counter; cells give FLOPs to 50% relative error per event, amortised where
one run serves all outputs, and the fraction of events within 50%):

| | depth 4 (ARC's shape) | depth 8 | depth 32 |
|---|---|---|---|
| ARC's tail method (factorized K = 3, sample-free; 3.4e9 / 1.3e10 / 2.1e11 FLOPs for all outputs at one threshold) | within 50% for 96% / 55% / 28% of events at 1e-4 / 1e-6 / 1e-8, 9% at 1e-12; beats zero to 1e-8 | 56% at 1e-4, 17% at 1e-6; beats zero only at 1e-4 | within 50% for at most 1% of events at every level; never beats zero |
| splitting without psi, per event (unbiased; B50 projected from the measured run variance) | 5.4e9 (1e-4) to 4.0e10 (1e-12), finite for every event | 1.3e10 to 1.5e11 | 6.4e10 to 4.2e12 |
| Monte Carlo, amortised | 8.2e7 at 1e-4; 4.2e9 at 1e-6 | 1.7e8; 8.6e9 | 6.6e8; 3.2e10 |
| **ours: Monte Carlo moments (smallest successful budget, 2^20 or 2^22 paths) + saddlepoint, amortised (biased)** | 4.5e8 at every level; within 50% for 99 / 93 / 84 / 70 / 53% (1e-4 ... 1e-12); wins at 1e-8 and 1e-10 | 8.9e8; 89 / 71 / 40 / 23 / 16%; wins at 1e-6 | 56% at 1e-4, under 20% below; no win |
| ours: splitting on the top mode of the last hidden layer (one run for all outputs) | - | - | no win under both pre-registered reach rules |

- The same readout applied to ARC's own description (sample-free; 1.7-2.3x ARC's cost for one threshold vector,
  0.84-0.88x when five thresholds share the propagation) failed its pre-registered kill test
  at depth 8 (+0.12 more events within 50% than ARC's readout; the bar was +0.20), so it makes no claim. Post hoc, on
  the evaluation nets, it is within 50% more often than ARC's readout at every level from 1e-6 to 1e-12 at depths 4 and
  8 (+0.09 to +0.34; P2.5), and it beats zero down to 1e-12 at depth 4 where ARC's readout stops at 1e-8. A
  pre-registered confirmatory test on 20 fresh nets (P2.10) failed its joint primary test: depth 4 met the +0.20 bar
  at 1e-8 (+0.24), depth 8 did not (+0.14). At every level from 1e-6 to 1e-12 the readout did raise the fraction within
  50% at both depths, with positive pointwise 95% intervals (+0.19 to +0.25 at depth 4, +0.11 to +0.19 at depth 8), but
  none of the joint +0.20 tests passed.
- Killed by their pre-registered tests: importance sampling with a Gaussian fitted to the event (the chain's
  conditional pulled back to x, tested at depth 4 and 1e-8; relative variance 650-3,700 per sample) and layer-wise
  twisted SMC (the noise-leverage diagnostic r had per-net medians 0.46-1.1 on two development nets, suggesting limited
  typical gains for estimators that keep prior input draws; no uniform bound, single events went as low as 0.10). Mode
  splitting passed its kill test only through near-zero estimates and has no win on the evaluation nets.
- Smallest p at which each method still beats zero, at the pre-registered budgets (depth 4 / 8 / 32): ARC's tail method
  (at its own cost) 1e-8 / 1e-4 / none; Monte Carlo at ARC's cost none / 1e-4 / 1e-4; splitting at 3e11 FLOPs per event
  1e-12 / 1e-12 / 1e-8. Ours at its larger tested budget (5.7e11 / 1.1e12 / 4.6e12 FLOPs per run, before sharing):
  1e-12 / 1e-12 / 1e-6 (P2.3).
- Caveats: our winner is biased; it misses 50% for 16%, 30% and 47% of the depth-4 events at 1e-8, 1e-10, 1e-12 (18%,
  34%, 50% if the budget is fixed at 2^20 paths); its advantage needs the cost shared over 256 outputs and five
  thresholds, and per event it loses to splitting everywhere; accuracy was measured on 8 outputs per net. At depth 32 a
  third to a half of the 1e-10 and 1e-12 truths are provisional (the conclusions hold without them).

## Progress log (Part 2)

- 06:00Z. Exploration on development nets 100-101 only (before the pre-registration): splitting (generalised splitting,
  pCN moves, frozen tuning) works at depths 4 and 32 down to 1e-12; importance sampling with Gaussian proposals fitted
  to the event fails at depth 4 (relative variance per sample 1e3-3e4 even at 1e-6); ARC's factorized K = 3 tail
  estimate is close at 1e-4 to 1e-6 for some neurons at depth 4, off by up to 3 decades at 1e-12, and fails at depth 32
  even at 1e-4. Splitting on the top mode of the last hidden layer (one run for all outputs) works only for the few
  outputs strongly aligned with that mode. One GPU dev VM (on-demand L4, job loop) is running; logged in GPU-LOG.md.
- 06:27Z. Pre-registered (`work/arc-noisy/tails/PREREG-TAILS.md`, hash in `prereg_hashes.txt`; amendment 1 at 06:39Z
  fixes scoring details only). The exploration also showed the candidate that matters most: reading ARC's own
  factorized K = 3 description out with a saddlepoint (from its per-output moments up to order 4) instead of its
  heaviside expansion; on the exploration nets it was within 50% far more often than ARC's readout at depths 4 and 8.
  Kill tests on development nets 102-105 are running; a second L4 VM computes the evaluation-net thresholds and truths
  (20 nets x 3 depths x 3 sigma x 8 outputs x 5 thresholds).
- 07:10Z. Kill tests (development nets 102-105, 32 events per depth): the saddlepoint readout of ARC's description
  (U2) passed at depth 4 (+0.47 within-50% fraction over ARC's readout at 1e-8) but missed the +0.20 bar at depth 8
  (+0.12), so it is killed; its evaluation-net numbers come out of the same kprop runs and will be shown labelled post
  hoc, with no claim. Alive: Monte Carlo moments + saddlepoint (U3, 0.81 within 50% at depth 4, 1e-8) and, after a
  correction found by the code review, splitting on the top mode at depth 32 (U1, 17/32 under the written rule; it
  passes only through events whose estimates are essentially zero, so it is scored under a second, stricter rule too).
  Killed: Gaussian importance sampling (relative variance 650-3,700 per sample), layer-wise twisted SMC (noise leverage
  per-net median r of 0.46-1.1 on two development nets, suggesting limited typical gains). GPT-6 code review done (13 findings: 1 critical, 7 major); every fix is in
  the scorer before any evaluation-net result was looked at (amendment 3 records the kill-test correction).
- 08:45Z. All truths done (20 nets x 3 depths x 3 sigma; cross-checks broadly consistent, failures flagged; at depth 32 about a third of
  the 1e-10 and half of the 1e-12 truths are provisional). Both VMs deleted (empty instance list 08:39Z; part 2 GPU
  $4.23). sigma = 0.001 scored: the Monte Carlo moment + saddlepoint candidate wins amortised at depth 4 from 1e-8 to
  1e-12 and at depth 8 at 1e-6; nothing of ours wins at depth 32. sigma = 0.003 and 0.01 are being scored; the claims
  review follows.
- 09:56Z. All three noise levels scored; the headline wins persist at sigma = 0.001, 0.003 and 0.01. GPT-6 claims
  review done (7 major, 5 minor findings, all taken: chiefly, the depth-4 1e-12 win holds only under the scorer's
  median convention and is out of the headline; the moment budgets are labelled as selected; the better-than-zero table
  separates pre-registered from tested budgets; the backward-function and noise statements are narrowed). Confirmatory
  test of the saddlepoint readout of ARC's description on 20 fresh nets (pre-registered 09:27Z): better than ARC's
  readout at every level 1e-6 to 1e-12 at depths 4 and 8, but its +0.20 bar met only at depth 4. All VMs deleted.
- 10:07Z. Follow-up GPT-6 review of the fixes and the confirmatory section done; its points are applied (the
  confirmatory verdict is now stated as a failed joint primary test; remaining reporting gaps are listed in P2.8).
  Part 2 is complete. No VMs or jobs left running (instance list empty).

## P2.1 Setup

- **Nets**: part 1's (ReLU, W iid N(0, 2/n), no bias, no LayerNorm, x ~ N(0, I_256), noise sd sigma r_l after every
  ReLU). 20 evaluation nets (seeds 0-19); development nets 100-105 (exploration on 100-101, kill tests on 102-105).
- **Events**: the last pre-activation z_L,i = W_L h_{L-1} of output i above a threshold, at L = 4 (ARC's section 6.6
  shape: three ReLU layers, a linear layer, a threshold), 8 and 32 (truncations of the same depth-32 net). Outputs 0-7 of
  every net; five thresholds per output, set once at sigma = 0.001 by an adaptive splitting run so that p is about 1e-4,
  1e-6, 1e-8, 1e-10 and 1e-12 (medians of the true p: within 0.05 decades of those targets at L = 4 and 8, within 0.25 at
  L = 32), the same thresholds at sigma = 0.003 and 0.01. 40 events per (net, depth, sigma), 160 per cell, 7,200 in all.
  ARC used one fixed threshold (3 in their scale, 3 sqrt 2 in ours, since our first layer has twice their variance) and
  read the error off across outputs.
- **Truth**: generalised splitting (Botev and Kroese) over the Gaussian latent (x and the L - 1 noise vectors) with pCN
  moves; levels and move scales frozen from an independent pilot, so each run is unbiased; 32 runs of 2,000 particles at
  L = 32, 16 at L = 4 and 8. Held-out checks: split halves, plain Monte Carlo with 2^23 paths at 1e-4, and a second
  splitting estimator with other tuning (rho = 0.3, own pilot and seeds) for two outputs per net. An event is provisional
  if its relative SE exceeds 0.25 or a check fails. The aggregate cross-checks are broadly consistent and individual
  failures are flagged provisional: against Monte Carlo at 1e-4 (the registered check) the mean log ratio is -0.013 to
  +0.007 at every depth and sigma, z-scores have SD 0.91-1.18 and 0-5 of 160 fall beyond 3 SE; the second splitting
  estimator gives z SD 0.98-1.32 with 0-6 of 200 beyond 3 SE, the wider ones at L = 32. (An exploratory Monte Carlo
  comparison at 1e-6 is consistent too, apart from the small-count bias of a log; zero-count cases were dropped.) At L = 32 the deep levels are
  noisy: at sigma = 0.001, 48 of 160 events at 1e-10 and 82 of 160 at 1e-12 are provisional (sigma = 0.003: 54 and 83;
  0.01: 29 and 65); L = 4 has at most 2 of 800 and L = 8 at most 10 of 800 per sigma.
- **Metric**: FLOPs to 50% relative error (B50). Deterministic methods: their cost if within 50%, else never. Unbiased
  samplers: setup + run cost x relative MSE / 0.25, counted only if the method beat zero at its tested budget (some
  replicate nonzero, pooled relative MSE below 1). Biased moment-based samplers: the cheapest tested budget (2^20 or 2^22
  paths, 4 replicates) whose relative MSE is at most 0.25. Per event, and amortised over the 256 outputs (a method whose
  one run serves every output, and sometimes every threshold, divides its cost by the events it serves; splitting per
  event cannot share). Costs at part 1's flopscope float32 rates including pilots and readouts, ARC's kprop at its own
  counter. Medians over the 160 events of a cell, bootstrap CIs from resampling the 20 nets with their 8 outputs kept
  together; the bootstrap treats each truth and each replicate error as fixed, and the intervals are pointwise, not
  simultaneous across cells.
- **Better than zero**: median relative error (or relative RMSE) at most 0.9 with the CI's upper end below 1.

## P2.2 Kill tests (development nets 102-105, sigma = 0.001, 32 events per depth)

| candidate (ours) | what it is | kill-test result | verdict |
|---|---|---|---|
| U1 | splitting on the top mode of h_{L-1} (the description's dominant direction at depth 32), one run for all outputs, read out for every output's events | amortised B50 below splitting's per-event B50 for 17/32 events at 1e-8 (bar: half) under the written rule, but only because events whose eight estimates were essentially zero count as "beating zero"; 13/32 under the prereg's better-than-zero margin | alive by the written rule (after the review corrected my code's 15/32); scored under both rules |
| U2 | ARC's own factorized K = 3 description read out by a saddlepoint (Lugannani-Rice on its per-output moments up to order 4) instead of its heaviside expansion | within 50% at 1e-8: 0.66 vs ARC's 0.19 at L = 4 (+0.47), 0.31 vs 0.19 at L = 8 (+0.12; bar +0.20) | killed; its evaluation-net numbers are shown below as post hoc, with no claim |
| U3 | Monte Carlo moments (2^20-2^22 paths) read out by the same saddlepoint | 0.81 of events within 50% at L = 4, 1e-8 (bar 0.5) | alive |
| U4 | importance sampling over x from a Gaussian fitted to splitting particles in the event (the chain's conditional pulled back to x) | median relative variance per sample 656-3,650 (bar 100) | killed |
| U5 | layer-wise twisted SMC (x from its prior, noise twisted) | noise leverage r = P(A and A') / P(A): per-net medians 0.46-1.1 on two development nets (bar 0.1), suggesting limited typical gains (single events as low as 0.10); no uniform bound | killed |

## P2.3 Results on the evaluation nets (sigma = 0.001, 20 nets, 160 events per cell)

Each cell: FLOPs to 50% relative error per event / amortised (shared work divided by the 256 outputs, and by the five
thresholds where it serves them all) (fraction of the 160 events that reach 50%). Medians over events; "never" when
fewer than half the events reach 50%. For randomised rows "reach" means relative RMSE <= 0.5 over the replicates; for
splitting it is projected from the measured variance of the truth's frozen runs. The moment-based rows use, per event,
the smallest tested budget that reaches 50% (2^20 or 2^22 paths, 4 replicates each; a retrospective selection using the
truth, as amendment 1 specifies, not a stopping rule). Full tables with 95% CIs for B50, the validated-events-only
versions and every variant: `work/arc-noisy/tails/results/tables_s0.001.txt` and `report_tables_s0.001.md`; CIs for the
fractions and the fixed-budget fractions: `results/sensitivity.txt`. Deductive (no sampling of the network): kprop rows
and the post-hoc row. Biased: the moment-based rows. Unbiased: Monte Carlo, Kerdock, splitting, mode splitting.

**Depth 4 (ARC's shape)**

| method | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
|---|---|---|---|---|---|
| Monte Carlo (analytic) | 2.1e10 / 8.2e7 (1.00) | 2.1e12 / 4.2e9 | 2.1e14 / 2.7e11 | 2.1e16 / 2.0e13 | 2.0e18 / 1.6e15 |
| Kerdock design | 1.7e10 / 6.7e7 (1.00) | never (0.23) | - | - | - |
| splitting without psi, per event | 5.4e9 (1.00) | 1.0e10 (1.00) | 1.7e10 (1.00) | 2.6e10 (1.00) | 4.0e10 (1.00) |
| ARC's tail method (factorized K = 3) | 3.4e9 / 7.3e6 (0.96) | 3.4e9 / 7.3e6 (0.55) | never (0.28) | never (0.19) | never (0.09) |
| factorized K = 3, augmented | 2.9e10 / 6.3e7 (0.97) | 2.9e10 / 6.3e7 (0.55) | never (0.28) | never (0.16) | never (0.08) |
| Monte Carlo moments + Edgeworth (ARC's baseline) | 5.7e11 / 4.5e8 (0.98) | 5.7e11 / 4.5e8 (0.57) | never (0.22) | never (0.14) | never (0.07) |
| **ours (U3): Monte Carlo moments + saddlepoint** | 5.7e11 / 4.5e8 (0.99) | 5.7e11 / 4.5e8 (0.93) | 5.7e11 / 4.5e8 (0.84) | 5.7e11 / 4.5e8 (0.70) | 5.7e11 / 4.5e8 (0.53) |
| post hoc (U2, killed): saddlepoint on ARC's K3f description | 7.9e9 / 6.2e6 (1.00) | 7.9e9 / 6.2e6 (0.87) | 7.9e9 / 6.2e6 (0.62) | never (0.42) | never (0.34) |

**Depth 8**

| method | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
|---|---|---|---|---|---|
| Monte Carlo (analytic) | 4.3e10 / 1.7e8 (1.00) | 4.4e12 / 8.6e9 | 4.3e14 / 5.6e11 | 4.2e16 / 4.1e13 | 3.9e18 / 3.0e15 |
| Kerdock design | 3.8e10 / 1.5e8 (1.00) | never (0.26) | - | - | - |
| splitting without psi, per event | 1.3e10 (1.00) | 2.6e10 (1.00) | 4.9e10 (1.00) | 9.1e10 (1.00) | 1.5e11 (1.00) |
| ARC's tail method (factorized K = 3) | 1.3e10 / 2.0e7 (0.56) | never (0.17) | never (0.04) | never (0.03) | never (0.02) |
| factorized K = 3, augmented | 1.2e11 / 1.8e8 (0.68) | never (0.26) | never (0.08) | never (0.04) | never (0.03) |
| Monte Carlo moments + Edgeworth | 1.1e12 / 8.9e8 (0.84) | never (0.17) | never (0.07) | never (0.04) | never (0.01) |
| **ours (U3): Monte Carlo moments + saddlepoint** | 1.1e12 / 8.9e8 (0.89) | 1.1e12 / 8.9e8 (0.71) | never (0.40) | never (0.23) | never (0.16) |
| ours, pre-registered variant (U3r): with the chi mixture over \|x\| | 1.1e12 / 8.9e8 (0.92) | 1.1e12 / 8.9e8 (0.82) | 1.1e12 / 8.9e8 (0.61) | never (0.46) | never (0.34) |
| post hoc (U2, killed): saddlepoint on ARC's K3f description | 2.3e10 / 1.8e7 (0.82) | 2.3e10 / 1.8e7 (0.51) | never (0.28) | never (0.19) | never (0.11) |

**Depth 32** (at 1e-10 and 1e-12, 48 and 82 of the 160 truths are provisional; the conclusions below hold on the
validated events alone, where splitting's B50 is lower, 8.6e11 and 1.4e12)

| method | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
|---|---|---|---|---|---|
| Monte Carlo (analytic) | 1.7e11 / 6.6e8 (1.00) | 1.6e13 / 3.2e10 | 1.5e15 / 1.9e12 | 1.3e17 / 1.2e14 | 1.0e19 / 8.0e15 |
| Kerdock design | 1.6e11 / 6.1e8 (1.00) | never (0.24) | - | - | - |
| splitting without psi, per event | 6.4e10 (1.00) | 1.7e11 (1.00) | 4.8e11 (1.00) | 1.6e12 (1.00) | 4.2e12 (1.00) |
| ARC's tail method (factorized K = 3) | never (0.01) | never (0.01) | never (0.01) | never (0.00) | never (0.01) |
| ours (U3): Monte Carlo moments + saddlepoint | 4.6e12 / 3.6e9 (0.56) | never (0.17) | never (0.09) | never (0.06) | never (0.06) |
| ours (U1): mode splitting, reach rule "relative MSE < 1" | 4.7e12 / 3.7e9 (0.64) | 9.9e12 / 7.7e9 (0.60) | 1.2e13 / 9.0e9 (0.52) | never (0.48) | never (0.47) |
| ours (U1): same, rule "relative RMSE <= 0.9" | 4.7e12 / 3.7e9 (0.62) | never (0.49) | never (0.38) | never (0.25) | never (0.19) |

Wins (pre-registered rule: median over nets of the per-net ratio competitor B50 / our B50, bootstrap CI above 1; the
competitor is the category's best at that cell). The prereg did not say how to take a median of eight events when some
never reach 50% (infinite B50); the scorer takes the lower of the two middle values. The usual midpoint median makes a
net infinite as soon as one of its two middle events is, and I report both (`results/sensitivity.txt`):
- **Sampling/hybrid, amortised, U3**: depth 4 at 1e-8 and 1e-10, over splitting: 37x [36, 38] and 55x [51, 58] (lower
  median), 38x [37, 39] and 58x [53, 60] (midpoint median); depth 8 at 1e-6, over Monte Carlo: 9.4x [8.8, 9.7]
  (lower median) and 9.3x [8.8, 10.0] (midpoint). At depth 4, 1e-12 the lower median gives 83x [70, 85] and the midpoint median 19x [0, 80]: not a robust win.
  No win at depth 4 for 1e-4 (the Kerdock design is 6.7x and Monte Carlo 5.5x cheaper) or 1e-6 (a tie with ARC's Monte
  Carlo + Edgeworth baseline, which uses the same moments; U3 reaches 50% for 93% of events against its 57%), and none
  at depth 32. The pre-registered variant U3r also wins at depth 8, 1e-8 (47x [44, 51]; midpoint CI about [11, 52]); I
  treat U3 as the primary and do not pick the better of the two.
- These wins are for biased estimates that miss 50% for some events: at depth 4, 16%, 30% and 47% of events at 1e-8,
  1e-10 and 1e-12 (fractions within 50%: 0.84 [0.78, 0.89], 0.70 [0.63, 0.76], 0.53 [0.45, 0.61]; with the budget fixed
  at 2^20 paths, 0.82, 0.66, 0.50). With 2^20 paths the misses at 1e-6 and 1e-8 are mostly overestimates, median 1.9x;
  at 1e-10 and 1e-12 they go both ways (10th-90th percentiles of estimate / truth among the misses: 0.4-5.5x at 1e-10,
  0.3-8.1x at 1e-12). Per event (no sharing across
  outputs and thresholds) U3 loses to splitting everywhere (0.009-0.065x), because 2^20 paths cost more than one
  splitting run.
- **U1 (mode splitting)**: under the rule "relative MSE < 1" it would win at depth 32, 1e-6 over Monte Carlo (5.0x
  [3.2, 7.6]), and on validated events also at 1e-10 and 1e-12 over splitting, but every one of these rests on B50
  extrapolated from runs whose estimates were mostly near zero; under the stricter rule none holds, so by amendment 3
  there is no U1 win.
- **Sample-free**: no pre-registered candidate of ours survived to the evaluation nets, so there is no sample-free
  claim.

Smallest p at which each method still beats zero (median relative error or relative RMSE <= 0.9, CI below 1), sigma =
0.001. Pre-registered comparisons (deterministic methods at their own cost; randomised methods at ARC's cost for one
threshold vector, 3.4e9 / 1.3e10 / 2.1e11 FLOPs, and at 3e11 FLOPs):

| method | depth 4 | depth 8 | depth 32 |
|---|---|---|---|
| ARC's tail method (factorized K = 3) | 1e-8 | 1e-4 | none (not at 1e-4) |
| kprop K = 2 | 1e-6 | 1e-4 | none |
| factorized K = 3, augmented | 1e-8 | 1e-6 | (not run) |
| Monte Carlo at ARC's cost | none | 1e-4 | 1e-4 |
| Monte Carlo at 3e11 FLOPs | 1e-4 | 1e-4 | 1e-4 |
| splitting at 3e11 FLOPs per event | 1e-12 | 1e-12 | 1e-8 |
| post hoc (U2): saddlepoint on ARC's K3f description, at its own cost | 1e-12 | 1e-10 | none |

Supplementary, at the moment methods' tested budgets (2^20 paths: 5.7e11 / 1.1e12 / 4.6e12 FLOPs per run before any
sharing, above both pre-registered budgets; not evaluated at those budgets, since a biased readout's error does not
scale as 1/budget): Monte Carlo moments + Edgeworth (ARC's baseline) 1e-8 / 1e-6 / 1e-4; ours (U3) 1e-12 / 1e-12 /
1e-6. Mode splitting at its tested budget (8 runs in each direction): 1e-4 at depth 32. Splitting per event and
Monte Carlo have finite B50 at every level by construction, so "beats zero" for them depends only on the budget.

Our better-than-zero cutoff for ARC's method at depth 4, 1e-8, is of the same order as ARC's reported range ("beats
every baseline down to nearly 10,000x below 1/N", about 1.8e-8); the criteria and the events differ.

## P2.4 Noise levels

The thresholds are fixed, so the noise level changes the events only through their probabilities. At depths 4 and 8
the median event probability moves by about 0.006 decades at most between sigma = 0.001 and 0.01 (5-95% range at 1e-12
within about 0.16 decades). At depth 32 sigma = 0.003 raises it by 0.01-0.04 decades in the median (at 1e-4, +0.007 [0.004,
0.012], a small but nonzero shift) and sigma = 0.01 by 0.07-0.14 decades; at 1e-12 the ratio of the sigma = 0.01 and
0.001 reference probabilities has a 95th percentile of about 16x and a maximum of about 170x (one validated event). The
noise starts to matter in the deepest tails, as the twisted-SMC leverage diagnostic also suggested (per-net median r of
0.46-0.81 at depth 32, sigma = 0.01, on development nets 102-103; 0.26 on exploration net 100).

| result | sigma = 0.001 | sigma = 0.003 | sigma = 0.01 (secondary) |
|---|---|---|---|
| U3 amortised win, depth 4 (ratio to splitting, 1e-8 / 1e-10 / 1e-12) | 37x / 55x / 83x | 38x / 57x / 83x | 38x / 56x / 76x, the last CI [0, 82] (no win at 1e-12) |
| U3 amortised win, depth 8, 1e-6 (ratio to Monte Carlo) | 9.4x | 9.5x | 9.5x |
| U3, fraction of events within 50%, depth 4 (1e-6 ... 1e-12) | 0.93 / 0.84 / 0.70 / 0.53 | 0.93 / 0.86 / 0.69 / 0.56 | 0.93 / 0.83 / 0.69 / 0.54 |
| ARC's tail method within 50%, depth 4 (1e-4 / 1e-6 / 1e-8) | 0.96 / 0.55 / 0.28 | 0.96 / 0.55 / 0.27 | 0.96 / 0.53 / 0.30 |
| splitting per event at 1e-12 (depth 4 / 8 / 32) | 4.0e10 / 1.5e11 / 4.2e12 | 3.9e10 / 1.5e11 / 4.4e12 | 4.0e10 / 1.5e11 / 2.9e12 |
| mode splitting (depth 32) | no win | no win | wins at 1e-6 and 1e-8 under the looser reach rule only, so no win |

The win verdicts do not change from sigma = 0.001 to 0.003 (deep depth-32 probabilities do move a little). The lowest
noise level at which something of ours wins is 0.001, the lowest tested; there is no sigma = 0 comparison. The headline
wins (depth 4 at 1e-8 and 1e-10, depth 8 at 1e-6) persist at all three sigma; the convention-sensitive depth-4 win at
1e-12 fails at 0.01. (Reviewer's paired check on validated events: at depth 32, 1e-4, the 0.003 / 0.001 shift is
+0.0066 [0.0041, 0.0096] decades over 152 events.)

## P2.5 The saddlepoint readout of ARC's own description (post hoc; killed by its pre-registered test)

U2 reads the tail out of exactly the description ARC's tail method uses (factorized K = 3, simple kind, ARC's run_lpe
settings, our weights and noise): four polynomial readouts give that description's per-output moments E[z^p], p <= 4
(the K = 3 budget is ARC's harmonic approximation budget, not a cut at the third cumulant, so the description does carry
a fourth cumulant), and the Lugannani-Rice saddlepoint turns them into a tail probability. ARC's heaviside step instead
expands the indicator in Hermite polynomials around the description's Gaussian, which behaves like an Edgeworth series
in our tests: close at 1e-4, mostly off by 1e-8. The saddlepoint tilts a truncated cumulant generating function towards
the threshold and improves the empirical errors in these cells; no relative-error guarantee is established here. It
missed its kill bar on the development nets at depth 8 (+0.12 against +0.20), so it makes no claim here.
Post hoc, on the evaluation nets at sigma = 0.001, the per-net fraction of events within 50% is higher than ARC's by
+0.32 [0.23, 0.41], +0.34 [0.25, 0.44], +0.23 [0.11, 0.34], +0.25 [0.15, 0.36] at depth 4 (1e-6 to 1e-12) and by +0.34,
+0.23 [0.15, 0.31], +0.16, +0.09 at depth 8, at 1.7-2.3x ARC's cost for one threshold vector and 0.84-0.88x of it when
the five thresholds share the propagation. At depth 32 both fail. On ARC's augmented description the gap is larger
(+0.38 to +0.42 at depth 4).

## P2.6 What the results say

- **Readout, not sampler.** Every gain of ours comes from reading moments out with a saddlepoint, which tilts a
  truncated cumulant generating function towards the threshold. It improves the empirical errors in these cells; no
  relative-error guarantee is established here (a truncated CGF need not be a valid one). With 2^20-2^22 Monte Carlo
  paths of moments the median event stays within 50% to 1e-12 at depth 4 and to 1e-6 at depth 8 (1e-8 with the chi
  mixture). The deterioration with depth is consistent with information the four-moment approximation loses (at depth
  32 some outputs reach 1e-12 at 13 SD, others are lighter than Gaussian); this experiment does not isolate that cause.
- **The tested backward-function candidates failed.** At sigma <= 0.01 the event is decided mostly by x through a
  nonlinear map; our working hypothesis is that its conditional law given the event spreads over many input directions
  (not established here). The Gaussian input proposals
  we tried (depth 4, 1e-8) had 650-3,700 relative variance per importance sample, and the noise-only leverage diagnostic
  (per-net median r 0.46-1.1 on two development nets) failed its bar, which suggests limited typical gains for any
  estimator that keeps prior input draws. We did not test every backward function or every layer-wise SMC scheme.
  Splitting needs no psi and has finite projected cost for every event, about 1e4-1e6 network evaluations per event
  (1e-4 at depth 4 to 1e-12 at depth 32).
- **Depth 32.** None of our candidates has a qualifying win. The collapse onto the top mode helps only the outputs
  aligned with it: splitting on that mode serves all outputs at once, but one of its runs has relative RMSE <= 0.5 for
  49% of the events at 1e-4, 13% at 1e-8 and 3% at 1e-12. Per-event splitting has a finite projected B50 for every
  event (4.2e12 FLOPs at 1e-12); Monte Carlo has finite analytic B50 but is far more expensive; the Kerdock design
  reaches every event at 1e-4; beyond 1e-4 the moment readouts miss 50% for most events (U3 still beats zero at 1e-6 at
  its tested budget), and kprop and every deductive readout fail.
- **ARC's tail method** beats zero to about 1e-8 at depth 4 (the same order as ARC's reported range; the criteria and
  events differ) and degrades fast with depth:
  better than zero only at 1e-4 at depth 8, and never at depth 32, where most of its estimates are negative.

## P2.7 Predictions (PREREG-TAILS section 6) against results (sigma = 0.001)

| prediction | result |
|---|---|
| C-K3f within 50% for most events at 1e-4 (L = 4, 8), about half at 1e-6 and a quarter at 1e-8 (L = 4), under a fifth beyond 1e-6 at L = 8, wrong at L = 32 | right: 0.96 / 0.56 at 1e-4; 0.55 and 0.28; 0.17 and below; 0.01 |
| C-K3f better than zero to about 1e-6 (L = 8) and 1e-8 (L = 4) | half right: 1e-8 at L = 4, only 1e-4 at L = 8 |
| U2 passes its kill test; sample-free win at L = 4 and 8 | wrong: failed at L = 8 (+0.12 vs +0.20), so no claim; post hoc +0.09 to +0.34 at k >= 6 |
| U2 within 50% for most events to 1e-6 (L = 4, 8) and to 1e-8 (L = 4); better than zero to 1e-12 | mostly right post hoc: 0.87 / 0.51 at 1e-6, 0.62 at 1e-8; better than zero to 1e-12 (L = 4), 1e-10 (L = 8) |
| C-GS reaches 50% for every event; B50 at 1e-12 about 1e10-1e11 (L = 4), 1e11 (L = 8), 1e12 (L = 32) | right (projected from the measured run variance), except L = 32 is 4.2e12 |
| C-MC (amortised) cheapest sampler at 1e-4 only; C-Kd within 1.5x of it | close: the Kerdock design is cheaper still at 1e-4 (1.2x), within 1.5x as predicted |
| U1 fails its kill test | wrong by the written rule (17/32), but it has no win on the evaluation nets under both rules |
| U4 and U5 fail their kill tests | right |
| U3 passes at L = 4; about half within 50% at 1e-6 at L = 8; fails at L = 32 | right, and better than predicted at L = 8 (0.71) |
| No change from sigma = 0.001 to 0.003; small changes at 0.01, most at L = 32 | right for the headline verdicts; deep depth-32 probabilities shift a little even at 0.003 (P2.4) |

## P2.8 Review

**Code review** (GPT-6 via `codex-any exec`, `/tmp/arcn_review2/review_code.md`, done before any evaluation-net result was
looked at). 13 findings; what changed:
1. (critical) My kill-test code used relative MSE < 0.999, not the "< 1" amendment 2 states; under the written rule
   mode splitting passes (17/32, not 15/32). Fixed in amendment 3: U1 was run on the evaluation nets and scored under
   both the written rule and the prereg's better-than-zero margin, a win to count only under both (none did).
2. The strict reach rule was not applied to Kerdock, simulated Monte Carlo or splitting. Fixed: an unbiased sampler
   reaches 50% only if some replicate is nonzero and its pooled relative MSE is below 1; residuals normalised so all-zero
   replicates give exactly 1. For splitting the tested budget is the pooled frozen runs.
3. Provisional truths were counted but never filtered; a missing Monte Carlo check passed silently; zero counts were
   mishandled. Fixed: every table exists for all events and for validated events only; missing checks make events
   provisional; zero counts use the 95% bound 3/N.
4. Kerdock's first layer was charged twice, and its per-replicate rotations were treated as reusable setup. Fixed.
5. Summaries used the median of per-net medians. Fixed: median over the pooled events, bootstrap over nets (per-net
   medians only in the win ratio, as pre-registered).
6. The method selection for moment budgets, the category win test, the budget better-than-zero test and the post-hoc
   labels were missing from the reviewed version. Added.
7. Mode splitting's readout recomputes the hidden layers; that cost was missing. Fixed (charged as implemented, in
   the scorer and, after the claims review, in the kill-test script too; the kill-test numbers did not change).
8. Infinite B50 values made bootstrap bounds NaN that printed as "inf". Fixed (order-statistic quantiles).
9. The saddlepoint root search could miss an admissible root. Fixed (exact roots of the cubic); no kill-test number
   changed.
10. Readout and moment-accumulation FLOPs were omitted. Fixed (counted analytically; they matter only for the chi
    mixture, about 1e5 FLOPs per event). Splitting's resampling and Kerdock's radius draws are still not counted (small).
11. All-dead truth runs would crash the loader. Fixed (infinite variance, provisional).
12. The docstring said K = 3 has no fourth cumulant, and amendment 2 called U2's moments free. Both wrong: corrected
    here (the four moment readouts cost 5.9e9 FLOPs at depth 4, more than the shared propagation).
13. Seeds of development truths and of some evaluation cross-checks coincide (different nets and depths). Not changed;
    no event shares a stream with its own check.

**Claims review** (GPT-6, `/tmp/arcn_review2/review_claims.md`, on the report as of 09:28Z): no estimator failure; 7
major and 5 minor findings, taken as follows (a follow-up review, `review_claims2.md`, checked the fixes; its remaining
points are fixed in the text, except the reporting gaps listed after the list):
1. The depth-4 win at 1e-12 rests on the scorer's lower-median convention (83x); with the midpoint median it is 19x
   [0, 80]. Removed from the headline; both conventions reported (the 1e-8 and 1e-10 wins and depth 8 at 1e-6 hold
   under both).
2. The moment-based rows select the smallest successful budget per event (2^20 or 2^22 paths), not a fixed 2^20.
   Relabelled; fixed-budget fractions added (depth 4: 0.82 / 0.66 / 0.50 at 1e-8 / 1e-10 / 1e-12); the sharing over
   five thresholds is now stated in the headline.
3. The smallest-p table mixed the pre-registered budgets with the moment methods' larger tested budgets. Split into
   pre-registered and supplementary parts, with the tested budgets' FLOPs.
4. The failed U4/U5 tests were generalised to "backward functions have no leverage" and r >= 0.46 was read as a bound.
   Narrowed to the candidates tested; r reported as per-net medians, with the single-event minimum 0.10.
5. "Its error is relative" and "what limits it is the four-term CGF" were asserted, not shown. Reworded.
6. "Nothing changes from 0.001 to 0.003" overstated (deep depth-32 probabilities shift a little), and "up to 16x" was a
   95th percentile (the maximum is about 170x). Reworded.
7. "Splitting is the only accurate method at depth 32" conflicted with the tables (Monte Carlo and Kerdock reach 1e-4;
   splitting's reach is projected, not run). Reworded.
8. ARC's depth-32 cost for one threshold vector is 2.1e11, not 2.6e11 (that was five vectors). Fixed in text and
   scorer; no better-than-zero result changes.
9. Bootstrap CIs were given for B50 only. Added for the fractions within 50% (`results/sensitivity.txt`) and the
   bootstrap's scope stated.
10. "Checks pass at 1e-4 and 1e-6" overstated the truth validation (1e-6 was exploratory, and individual checks fail).
    Reworded.
11. ARC's "beats every baseline" is not our "beats zero"; the post-hoc U2 gap range is +0.09 to +0.34; the Monte Carlo
    prediction is "close" (Kerdock is cheaper at 1e-4). Fixed.
12. The kill-test script still used the older U1 cost model. Aligned with the scorer; verdict unchanged (27 / 17 / 13
    of 32 under the three readings).
Still incomplete after the follow-up review: CIs for the fractions within 50% exist only for U3, U2 and ARC's method at
depths 4 and 8 (4,000 resamples, all events); median-error CIs and the numeric intervals behind the better-than-zero
calls are not printed; the Kerdock design was not evaluated at the pre-registered fixed budgets.

## P2.9 Compute and files

- GPU: two on-demand L4 VMs with a job loop, 4.86 VM-hours, $4.23 at the $0.87/h ceiling, plus one for the
  confirmatory test (P2.10), 0.37 h, $0.32 (lane total, parts 1 and 2, $11.08 of the $50 cap). All deleted;
  `gcloud compute instances list --filter="name~arcn"` was empty at 08:39Z and again at 09:55Z. Log: `GPU-LOG.md`.
- CPU: 460 kprop runs (K = 2, factorized K = 3 simple and augmented) on this host under nice, at most 6 processes; the
  augmented K = 3 runs at depth 8, sigma = 0.01 were stopped for CPU time (that row is missing from the sigma = 0.01
  tables).
- Code and results in `work/arc-noisy/tails/`: `PREREG-TAILS.md` (three amendments; hashes in `../prereg_hashes.txt`),
  `tl.py` (net, latent, costs), `bgs.py` and `gs.py` (splitting), `thresholds.py`, `truth.py`, `truthcheck.py`,
  `mcx.py`, `kerdock_tail.py`, `kprop_tail.py` and `run_kprop_tail.py` (ARC's tail method and the description's
  moments), `deduce.py` (Gaussian, Edgeworth, saddlepoint, chi mixture), `mode.py`, `x1.py`-`x4.py` (exploration and
  kill-test runs), `killtests.py`, `analyze_tails.py` (every table), `report_tables.py`; raw outputs in `runs/`, tables
  in `results/` (`tables_s*.txt`, `report_tables_s*.md`, `killtests.txt`).

## P2.10 Confirmatory test of the saddlepoint readout of ARC's description (fresh nets)

Pre-registered at 09:27Z (`tails/PREREG-TAILS-confirm.md`, hash in `prereg_hashes.txt`) after the post-hoc result of
P2.5 and before any run on the nets used: 20 fresh nets (seeds 24-43), sigma = 0.001, depths 4 and 8, the same
thresholds, truth, code and scoring. Primary: at 1e-8, U2's per-net fraction of events within 50% exceeds ARC's by at
least +0.20 with the CI above 0, at both depths.

| depth | 1e-6 | 1e-8 (primary) | 1e-10 | 1e-12 |
|---|---|---|---|---|
| 4 | +0.25 [0.17, 0.33] | +0.24 [0.16, 0.33] | +0.21 [0.14, 0.29] | +0.19 [0.10, 0.29] |
| 8 | +0.19 [0.12, 0.26] | +0.14 [0.07, 0.21] | +0.12 [0.04, 0.21] | +0.11 [0.06, 0.16] |

(Mean over nets of U2 minus ARC's fraction of events within 50%, bootstrap 95% CI, 10,000 resamples; with the 10 of
1,600 provisional events removed the verdicts are the same.) **The registered joint primary test failed: depth 4 met
the criterion, depth 8 did not** (+0.14 at 1e-8, below the +0.20 bar set before the test). At every level from 1e-6 to
1e-12 U2 increased the fraction of events within 50% at both depths, with positive pointwise intervals, but none of the
secondary joint +0.20 tests passed either, since every depth-8 gain is below 0.20. Better than zero: U2 to 1e-12 at
depth 4 and to 1e-8 at depth 8 (predicted 1e-10); ARC's readout to 1e-8 and 1e-4 (as predicted). So, on fresh nets,
reading ARC's own description out with a saddlepoint put about a fifth more of the events within 50% at depth 4 and a
tenth to a fifth more at depth 8, at 1.7-2.3x ARC's cost for one threshold vector (0.84-0.88x when five thresholds
share the propagation); the effect is smaller than the bar we set at depth 8. It modifies ARC's own method's readout,
not the description; at depth 32 (evaluation nets, P2.3) neither readout works. GPU:
one L4 VM, 0.37 h, $0.32 (deleted; empty instance list at 09:55Z). Tables: `tails/results/tables_confirm.txt`.

---

# Part 3: a better sample-free readout of ARC's own descriptions (depths 4 and 8)

Brief (opus-mats, ~10:08Z, under Kelvin's delegation of experiment choice): turn part 2's measured depth-8 improvement
of the saddlepoint readout over ARC's readout (+0.14 against a +0.20 bar) into a claim with a better readout of ARC's
own description, no sampling; second priority, a beyond-four-moment readout of Monte Carlo moments; kill bar on
development nets, confirmatory test on fresh nets; CPU only (at most 4 processes), about 3 h. Code:
`work/arc-noisy/tails/` (`PREREG-TAILS-p3.md`, `p3_explore.py`, `p3_score.py`).

## Summary (Part 3)

**Pre-registered confirmatory win for a sample-free readout of ARC's own description, at depth 8. On 20 fresh nets, the
chi-mixture saddlepoint on ARC's augmented factorized K = 3 description puts +0.32 [0.23, 0.41] more of the 1e-8 events
within 50% than ARC's own (heaviside) readout of the same description: 0.45 against 0.13.** The pre-registered bar was
+0.20 with the CI above 0. The gain holds at every level from 1e-4 to 1e-12 (+0.19 to +0.45, all CIs above 0), it beats
zero down to 1e-12 where ARC's readouts stop at 1e-6, and it is the same on validated events alone. No sampling of the
network is involved: it uses only the four moments the description assigns to each output and the chi_256 law of |x|.

| depth 8, sigma = 0.001, 20 fresh nets (160 events per level) | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 | FLOPs (ARC's counter) |
|---|---|---|---|---|---|---|
| ARC's tail method, factorized K = 3 (heaviside) | 0.59 | 0.21 | 0.10 | 0.06 | 0.01 | 1.3e10 per threshold vector |
| ARC's augmented factorized K = 3 (heaviside) | 0.74 | 0.26 | 0.13 | 0.08 | 0.04 | 1.2e11 per threshold vector |
| **ours: chi-mixture saddlepoint on the augmented description** | **0.93** | **0.71** | **0.45** | **0.37** | **0.24** | 2.0e11 for all thresholds |

(Fraction of events within 50%.) The cost is 1.7x ARC's augmented readout for one threshold vector and 0.88x when five
thresholds share the propagation; against ARC's cheaper K = 3 tail method it is 15x per threshold vector. Why it works:
ARC's plain K = 3 description gets each output's mean, variance and skewness right but gives nearly every output the same
fourth cumulant; the augmented description individualises it (correlation with the true fourth cumulant 0.50 against 0.21
at depth 8), and the chi-mixture saddlepoint uses it far deeper into the tail than ARC's Hermite-type expansion. It is
still biased: more than half the 1e-8 events at depth 8 are not within 50%. At depth 4 (secondary) the gain over ARC's
augmented readout is the same size, +0.33 [0.21, 0.43] at 1e-8 (0.63 against 0.31 within 50%), at 2.3x its cost per
threshold vector. Priority 2 (beyond-four-moment Monte Carlo readouts) was not run (CPU budget).

## Progress log (Part 3)

- 10:34Z. Exploration on development nets 100-101 found why the part 2 readout stalls at depth 8: ARC's K3f description
  gets each output's mean, variance and skewness right but its fourth cumulant is nearly the same for every output
  (correlation 0.21 with the true one at depth 8), and it carries nothing beyond order four. ARC's augmented K3
  description has a better fourth cumulant (correlation 0.50), and the chi-mixture saddlepoint on it puts 0.55 of the
  1e-8 depth-8 events within 50% there, against 0.09 for ARC's own readout of the same description. Pre-registered
  (`tails/PREREG-TAILS-p3.md`, 10:33Z): kill bar on nets 102-105, confirmatory test on 20 fresh nets (44-63) whose
  truths are running on the CPU. Priority 2 is not run (CPU budget); stated in the prereg.
- 11:13Z. Kill test passed on development nets 102-105, barely: at depth 8, 1e-8, 0.375 of events within 50% against
  0.156 for ARC's readout of the same augmented description (+0.219, bar +0.20; `tails/results/p3_killtest.txt`).
  Fresh-net truths done on the CPU (20 nets, depths 4 and 8; at depth 8, 1e-8, median relative SE 0.13 and 8 of 160
  events provisional). ARC's augmented kprop runs on the fresh nets are under way (4 processes); scoring follows.
- 11:38Z. Confirmatory primary passed at depth 8, 1e-8: +0.32 [0.23, 0.41] more events within 50% than ARC's readout
  of the same description (bar +0.20, CI above 0); positive at every level 1e-4 to 1e-12. Depth 4 (secondary) runs are
  finishing; the GPT-6 claims review follows.

## P3.1 What was tried, and why the augmented description

Exploration on development nets 100-101 (64 events per level, sigma = 0.001, before the prereg), against the part 2
splitting references and a 2^26-path Monte Carlo estimate of each output's moments:
- ARC's plain factorized K = 3 description reproduces each output's mean, variance and skewness (skewness correlation
  0.996-0.998 with the Monte Carlo value), but its excess kurtosis is nearly the same for every output (correlation
  0.26 at depth 4, 0.21 at depth 8), and it carries nothing beyond order four (its implied fifth cumulant is exactly 0,
  its sixth an artifact). So no readout of it can individualise the tail beyond what three cumulants allow.
- ARC's augmented description (same K, AUGMENT kind, about 9x the cost) individualises the fourth cumulant (correlation
  0.63 at depth 4, 0.50 at depth 8; half the median error).
- Readouts compared at depth 8, 1e-8 (fraction within 50%): ARC's heaviside 0.11 (K3f) / 0.09 (K3fa); order-4
  saddlepoint 0.28 / 0.38; chi-mixture saddlepoint 0.33 / 0.55; a normal-inverse-Gaussian fitted to the four cumulants,
  and its chi mixture, 0.25-0.30; dropping the fourth cumulant, under 0.05. With Monte Carlo moments the chi-mixture
  saddlepoint reaches 0.61, so on the augmented description the readout gets close to what four exact moments allow.

## P3.2 Kill test and confirmatory test

- Kill test (development nets 102-105, 32 events per level, part 2 truths): at depth 8, 1e-8, 0.375 within 50% against
  0.156 for ARC's readout of the same description, +0.219 (bar +0.20): passed, narrowly. Other levels: +0.22, +0.41,
  +0.13, +0.13 at 1e-4, 1e-6, 1e-10, 1e-12.
- Confirmatory test (fresh nets 44-63, never used before): thresholds and truths computed on the CPU at 10:12-11:12Z
  with the part 2 protocol at smaller sizes (8 frozen splitting runs of 1,000 particles per output; no Monte Carlo
  cross-check). At depth 8, 1e-8 the truths' median relative SE is 0.13 and 8 of 160 events are provisional (31 and 59
  at 1e-10 and 1e-12).

| depth 8: mean per-net difference in the fraction within 50% [95% CI] | 1e-4 | 1e-6 | 1e-8 (primary) | 1e-10 | 1e-12 |
|---|---|---|---|---|---|
| ours minus ARC's augmented readout | +0.19 [0.14, 0.24] | +0.45 [0.35, 0.54] | **+0.32 [0.23, 0.41]** | +0.29 [0.22, 0.38] | +0.21 [0.15, 0.26] |
| ours minus ARC's K = 3 tail method | +0.34 [0.28, 0.39] | +0.50 [0.41, 0.59] | +0.35 [0.26, 0.43] | +0.31 [0.24, 0.39] | +0.23 [0.17, 0.29] |
| same as the first row, validated events only | +0.19 | +0.45 | +0.32 [0.23, 0.41] | +0.33 | +0.27 |

Primary (pre-registered: at least +0.20 with the CI above 0, depth 8, 1e-8): passed, +0.32 [0.23, 0.41]. The saddlepoint
never fell back to a lower order on these 800 events. Better than zero (median relative error <= 0.9, CI below 1): ours
at every level to 1e-12; both ARC readouts only to 1e-6.

Depth 4 (secondary; same fresh nets): ours minus ARC's augmented readout +0.03 [0.01, 0.06], +0.29 [0.22, 0.36],
+0.33 [0.21, 0.43], +0.26 [0.15, 0.38], +0.19 [0.12, 0.26] at 1e-4 ... 1e-12 (minus ARC's K = 3 tail method: +0.08,
+0.33, +0.32, +0.27, +0.20); fraction within 50% 1.00 / 0.91 / 0.63 / 0.43 / 0.29 against 0.97 / 0.62 / 0.31 / 0.17 /
0.10; better than zero to 1e-10 (ARC's readouts to 1e-8). Costs at depth 4: 6.8e10 (ours, all thresholds), 2.9e10 (ARC
augmented, per threshold vector), 3.5e9 (ARC K = 3). Raw output: `tails/results/p3_confirm.txt`.

## P3.3 Caveats

- It is an improvement of ARC's own method's readout on ARC's more expensive description, not a new description. The
  matched comparison is with ARC's readout of that same description (+0.32). ARC's cheaper K = 3 tail method is beaten
  by more (+0.35) but costs 15x less per threshold vector.
- Biased and deductive: there is no error bar, and 55% of the depth-8 1e-8 events are still outside 50% (76% at 1e-12).
- The radial mixture assumes noiseless homogeneity (z = |x| g); at sigma = 0.001 that is an approximation.
- The fresh-net truths are smaller than part 2's (8 runs of 1,000 particles): about 13% relative SE at 1e-8 at depth 8.
  The verdict is the same on validated events alone.
- The readout was chosen on development nets 100-101 after part 2's evaluation-net results were known (part 2 showed this
  variant post hoc at +0.38 at depth 8, 1e-8); the confirmatory nets 44-63 were untouched until scoring.

## P3.4 Compute and files

CPU only (this host, nice, at most 4 processes): fresh-net thresholds and truths about 60 minutes on 3 processes; ARC's
augmented kprop about 4 minutes per net and depth. No GPU. Files in `work/arc-noisy/tails/`: `PREREG-TAILS-p3.md`
(hashes in `../prereg_hashes.txt`), `p3_explore.py`, `p3_score.py`, `deduce.py` (unchanged readouts plus the NIG ones
tried in exploration), `results/p3_killtest.txt`, `results/p3_confirm.txt`; raw outputs in `runs/fresh3/`,
`runs/kp_fresh3/`, `runs/kp_x1/` and `runs/kp/`.
