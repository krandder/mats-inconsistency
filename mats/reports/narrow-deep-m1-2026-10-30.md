# M1 at width 8: can a two-sided, sample-free estimator beat random sampling on plain ReLU nets of width 8 and depth 256? (30-31 October 2026)

Lane: narrow-deep-m1. Work directory: `work/narrow-deep-m1/`. Runs from 22:25Z on 30 October to about 01:55Z on 31
October, CPU only (at most 4 processes under `nice -n 19 ionice -c3`).
- Pre-registration: `PREREG.md`, Part 1 (hashed 22:57:03Z) plus three amendments (23:27:43Z, 23:37:23Z, and 00:18:28Z
  on 31 October), each hashed before the evaluation-net results it affects. The hashes and times in
  `prereg_hashes.txt` are authoritative: two amendment headers carry a written time that ran ahead of the hash. The
  reviewer reproduced all four hashes.
- Review: one GPT-6 review (gpt-6-astra, high effort). Its verdict was "major revision": keep the negative M1 result,
  narrow the causal claims, repair the scoring and the accounting. Section 12 lists its findings and the changes; one
  oracle was rerun with a fix.
- Status labels: PROVEN, MEASURED, HEURISTIC. Every number below is MEASURED unless labelled otherwise.

Ask (Kelvin, via main, 22:21Z): "Let us try to beat random sampling with width=8, and 256 layers. (a) we can use ARC's
method until layer 5-8 (right? they can do it until layer 8 right?) (b) we can use our own methods with few dimensions
after. Maybe discuss how to handle the middle." Later, via main: "we can try go backwards from there?", that is, a
backward-only arm that meets the known Gaussian input at layer 0. Bar: a major advance on M1, a sample-free estimator
that matches or beats sampling at equal compute on deep random MLPs.

## Summary for Kelvin

**Verdict: no major advance. The sample-free estimators built here do not reach M1 at width 8, depth 256, sigma =
1e-3, and the registered M1 tests (P5) fail.** Two estimators were built: the two-sided one (TS) meets at layer 2, 4 or
8, and the backward-only one (BO) meets at the input.
- Each beats RQMC by 3x at some point of its ladder on 2 of 6 healthy nets. Elsewhere RQMC is ahead, by up to 5,000x.
  Above 2^13 paths only one ladder point is ahead (on 217, a cancellation).
- Your (b) holds.
- What limited these methods: the forward descriptions beyond layer 2, how many directions the value function needs
  near the input, and the noise's effect on E[a_L], which none of our deterministic noise treatments captures. These
  are floors of the methods we built, not proven lower bounds.

1. **(a) ARC's method to layer 5-8: true at width 256 (ARC's result), not at width 8.**
   - ARC's cumulant propagation (K = 2, 3, 4; simple, augmented, factorized) is exact at layer 1.
   - Its error on E[a_c] is 3e-5 to 2.3e-2 at layer 2. By layers 5-8 it is 4e-3 to 2.2e4 (4 nets).
   - Its law at layer 2 is still good on 3 of 4 nets. With an exact (oracle) downstream it would beat RQMC at about 30
     paths' cost, under a nominal 25-path downstream: the unregistered Gaussian-law meet does so on 3 of 4 nets, and the
     registered polynomial meet also on 3 (216, 217, 220; 2-33x).
   - A gate-pattern mixture (2k-60k Gaussian pieces) is steadier, but floors at 1e-3 to 1e-2. The floor comes from this
     implementation's approximations: Gaussian pieces, Stein cross-covariances, forced gates.
2. **(b) Few dimensions after: true.**
   - At layer 8, 10-77 downstream-distinct gate patterns carry 99% of the output mass, against 9,600-18,000 distinct
     upstream histories among 2^15-2^16 inputs (most seen once). By layer 24 there is one.
   - There the noiseless map to the output is one linear map; for states in its class it is exact to 4e-14.
   - The value function needs 1-4 directions at layer 8 on 5 of 6 nets.
3. **The middle (layers 8-32) is easy.** Deterministic noiseless passes are exact there. A grid over a few directions is
   worse: multilinear interpolation gave errors of 2.5e-5 to 1.9, depending on the grid.
4. **Backward to the input (your later idea).**
   - Under our projection rule, the value function needs 6-7 of 8 input directions for nMSE 1e-5 (3-7 for 1e-4). The
     count falls to 4-8 at layer 2 and 1-4 at layer 8 (one net: 7).
   - BO (radius exact, a deterministic product rule over directions) beats RQMC by 3x on 2 of 6 nets: 10x at 335
     paths on 220, 14x at 228 on 221. Both are at coarse rules: refining the rule raises its error on 221 from 9e-5 to
     6.4e-3 and on 220 from 8e-5 to 3e-4.
5. **The noise.**
   - sigma = 1e-3 per layer moves E[a_L] by nMSE 5e-5 to 5e-3 (above 1e-4 on 4 of 6 healthy nets), almost all of it
     through layers 33-256.
   - It is steep in sigma: on net 216, 4e-6 at 1e-4, 5e-3 at 1e-3, 5e-2 at 3e-3.
   - Our deterministic treatments miss it. K2 tables are 3-5x off for small-radius paths and make it worse on 3 of 4
     dev nets. A Gaussian-sum filter (the coordinator's suggestion) gets typical radii right but leaves 1.3e-4, at a
     cost of 7e4-5e5 paths.
   - Every deterministic arm here is therefore noiseless downstream and inherits this bias. RQMC reaches about 1e-5 at
     2^13 paths and 3e-7 to 1e-6 at 2^17.
6. **The isolated wins depend on the net and are unstable.**
   - Some are cancellations between two biases of opposite sign. On net 217 a forward bias of about +4.8% of SD and the
     omitted noise effect of about -4.8% give nMSE 9e-8, below the truth's resolution, so that ratio is not claimed.
     BO on 221 is the other case; refining the rule undoes it.
   - Others are a fixed closure that happens to land close on that net at tiny budgets: the exact-moment Gaussian
     closure K2X wins 54x at 28 paths on 217 (itself helped by a partial cancellation: +3.2% against -4.8% of SD) and
     10x at 32 paths on 221 (same-sign terms).
   - At c = 2 the same closure is 10-31x worse than RQMC on 216, 220 and 223, and 2.3x better on 203.
   - No selection rule picks the winning configuration in advance.
7. **Tails (P = 1e-4).**
   - TS's tail is off 2-1,000x, or zero.
   - BO's tail integrates the radius exactly on its slice of directions. It is within 2x on 6-8 of 11 configurations per
     net. Its median relative-MSE advantage over the radius-integrated sampler (RB) at equal cost is 2.2-4.7x on 5 of 6
     nets, and 0.8x (a loss) on 203. It is biased low and unconverged.
   - It is the one place where a deterministic arm is modestly ahead, and it is not a certified bound.
8. **Sampling layers 1-8 and doing the rest deterministically (not sample-free).** This beats RQMC 4.6-6x at 512 paths
   on the two nets with small noise effects. It carries the noise bias: at most 2x from 2^11 paths, and behind from
   2^13.
9. **Scaling with width (coordinator's arm), one net per width at n = 16 and 32, L = 32 n.**
   - The closures improve at shallow cuts: K2's layer-2 error halves from n = 16 to n = 32.
   - The middle lengthens: at layer 32 there are 1, 14 and 121 downstream classes at n = 8, 16 and 32.
   - The input needs about 3n/4 directions under our rule.
   - The estimator scaling arm was registered to run only if width 8 succeeded, so it was not run. The cost-vs-width
     reading (section 8) is HEURISTIC.

## 1. Setup

- **Nets** (`work/narrow-deep/lib/nd.py`, unchanged from the previous lane): plain He-ReLU, width 8, depth 256,
  W_l iid N(0, 2/8), no bias, x ~ N(0, I_8); h_l = ReLU(W_l h_{l-1}) + s_l eps_l for l < 256, output
  a_256 = ReLU(W_256 h_255).
- **Noise**: s_l = sigma r_l, with r_l the RMS of ReLU(W_l h_{l-1}) over a seeded 2^14-path pilot of the noisy net (the
  activation before layer l's own noise, on noisy trajectories). The pilot is part of the net; nobody is charged for
  it. sigma = 1e-3 primary, 1e-2 secondary. Estimators and samplers use the same s_l.
- **Census** (`scratch/census.py`, `results/census_table.md`): 128 nets, seeds 100-163 (dev) and 200-263 (evaluation).
  Classes were fixed before any estimator result:
  - healthy: at most 5% of noiseless inputs dead by depth 256, and the input explains at least 90% of the output
    variance at sigma = 1e-3;
  - dead-noise: every sampled noiseless input dead by 256;
  - dead-exact: noisy output identically 0;
  - intermediate: the rest.
  The evaluation pool has 15 healthy, 16 intermediate, 32 dead-noise and 1 dead-exact net. With noise, a "dead" net is
  not zero: its output is driven by noise injected after the death layer.
- **Evaluation set** (registered): the first 6 healthy seeds (203, 216, 217, 220, 221, 223). Secondary arms after
  Amendment 3's trims: 200 (dead-noise) and 202 (intermediate) at sigma = 1e-3, and 203 and 216 at sigma = 1e-2.
- **Estimand**: E[a_L] in R^8, scored by nMSE = |est - truth|^2 / tr Var(a_L); plain MC with N paths scores 1/N. Tail:
  p = P(|a_L| > t), t the 1e-4 quantile from a threshold set disjoint from the paths that estimate p.
- **Truth**: plain MC, 2^24 paths for the 6 healthy nets at sigma = 1e-3 (truth nMSE 6e-8), 2^23 otherwise. Noiseless
  (sigma = 0) truths at 2^22 for the healthy nets give the total noise effect.
  - Ratios are claimed only where the method's nMSE is at least 10/P (6e-7 here).
  - RQMC at 2^20 sits at the truth's resolution and is not used.
  - All replicates share one truth. Subtracting its expected error 1/P is right in expectation, but the shared
    realised error is not in the quoted replicate SEs.
- **FLOPs**: flopscope conventions (matmul M N (2K - 1), elementwise 1, normal draw 16, exp/log 16, cdf 96, pdf 54).
  One MC path costs 69,616 FLOPs; budgets are in MC paths.
  - The counters charge the large items: forward descriptions, probe Jacobians, reference-class passes, and every
    downstream evaluation. ARC's kprop is charged by ARC's own counter.
  - They are approximate. The review lists omissions: BO's rule construction and eigensolve, reductions, TS's tail,
    Sobol generation for RQMC, 144 noise FLOPs per sample at SH's cut, and a 480-for-720 undercount in meet_direct. It
    also lists overcharges: zero-weight nodes, and SH's unused grid.
  - The reviewer's sensitivity check: halving every deterministic cost gives 3 winning nets per arm, doubling gives 2.
    M1's 4-net criterion fails either way.
- **Baselines**:
  - MC: nMSE exactly 1/N; simulated to 2^13 and consistent with the truth.
  - RQMC: scrambled Sobol on the input, iid noise. Replicates R = 32 to 2^13, 16 at 2^15, 8 at 2^17 and 2 at 2^20
    (8 and 2 instead of the registered 16 and 4: an undeclared deviation), and 256 at 2^3-2^6 (post hoc, section 10).
  - HYB: the previous lane's hybrid (MC to l0 = 16 or 32, then 8 log-radius bins with a K2-Stein deep part).
  - Between measured budgets, RQMC is interpolated log-log. That is a convention, not a measurement at every N.
- **Sample-free**: no pseudo-random numbers, and the estimate is not an average of the net's output over a point set in
  input space. BO is the exception noted in Amendment 1: its nodes are input points on a k-dimensional slice, so it sits
  close to QMC.

## 2. Structure of the problem: what collapses and what does not

**Downstream-distinct classes** (Kelvin's "not all combinations matter downstream"; `scratch/oracle_k_by_cut.py`).
The count is of noiseless future gate patterns covering 99% of the noiseless |a_L| mass, against distinct upstream
histories among 2^15 inputs (2^16 for net 203).

| net | cut 1 | 2 | 4 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| 203 | 5,480 vs 239 | 2,429 vs 2,289 | 418 vs 10,814 | 18 vs 18,233 | 9 vs 20,380 | 2 vs 22,406 | 1 vs 22,787 |
| 216 | 2,441 vs 241 | 592 vs 2,283 | 85 vs 8,127 | 17 vs 11,842 | 10 vs 12,327 | 7 vs 12,492 | 1 vs 13,074 |
| 217 | 2,722 vs 255 | 1,029 vs 2,566 | 199 vs 8,255 | 33 vs 15,711 | 5 vs 17,722 | 2 vs 18,229 | 1 vs 18,424 |
| 220 | 3,040 vs 234 | 1,395 vs 2,206 | 370 vs 6,944 | 33 vs 9,652 | 17 vs 10,976 | 5 vs 12,259 | 1 vs 12,364 |
| 221 | 4,367 vs 240 | 1,456 vs 1,853 | 574 vs 4,788 | 77 vs 11,417 | 20 vs 14,258 | 7 vs 15,317 | 1 vs 15,998 |
| 223 | 2,666 vs 236 | 800 vs 2,115 | 200 vs 6,490 | 10 vs 11,109 | 4 vs 11,260 | 3 vs 11,641 | 1 vs 11,721 |

- Kelvin's point holds strongly. At layer 8, 10-77 downstream classes carry 99% of the mass, against at least 9,600
  realised histories (most seen once, so the true count is larger). From layer 24 on there is one class.
- The two counts cross between layers 2 and 4.

**Directions the value function needs at each cut** (Amendment 1, item 1; corrected after review,
`runs/kbycut2_*.json`).
- The measure is the registered statistic: the smallest k whose projected meet has nMSE at most 1e-5, with 1e-4 in
  brackets. In square brackets, for the 1e-5 threshold only and where it differs, is the post-hoc "stable" k, from
  which every larger k also passes. At 1e-4, smallest and stable k differ in 12 of 54 cells (for example, net 220 at the
  input: 3 against 6).
- The sample-free probe-direction results are in the same JSON files and are not tabulated.
- The projected meet evaluates the true noisy downstream on the coordinates of the top-k eigenvectors of S_c =
  C^1/2 M C^1/2. Following the review, the observation directions are C^-1/2 V. The dropped part goes to its best
  affine predictor; at the input it is 0, which there is the exact conditional mean.
- The first version used C^1/2 V and gave 1-3 more directions at cuts 1-5. At the input the transformation is close
  to the identity, because the covariance there is estimated near I, but 217 moved from 8 to 7.
- The rerun used 2^15 paths for every net (the first version: 2^16 for 203 and 217).
- The counts describe this projection rule, not the minimum any representation could need.

| net | c = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| 203 | 7 (6) | 6 (5) | 4 (4) [6] | 5 (3) | 6 (3) | 4 (4) [6] | 4 (2) | 4 (2) | 3 (3) |
| 216 | 7 (6) | 8 (7) | 5 (1) | 5 (5) | 4 (1) | 4 (3) | 2 (1) | 3 (1) | 1 (1) |
| 217 | 7 (7) | 6 (6) | 6 (6) | 5 (3) | 5 (1) | 5 (5) | 5 (5) | 5 (4) | 4 (4) |
| 220 | 6 (3) | 7 (5) | 8 (2) | 6 (5) | 2 (2) [5] | 5 (4) | 1 (1) [4] | 1 (1) | 1 (1) |
| 221 | 7 (7) | 7 (7) | 6 (5) | 5 (3) | 4 (4) | 6 (5) | 4 (1) | 7 (5) | 7 (4) |
| 223 | 7 (6) | 8 (5) | 4 (4) | 2 (2) | 2 (2) [5] | 6 (3) | 3 (2) | 2 (2) | 2 (1) |

- At the input every net needs 6-7 of the 8 directions for 1e-5.
- The count falls to 4-8 at layer 2, 2-6 at layer 4 and 1-4 at layer 8. Net 221, which synchronises later (77 classes
  at layer 8), still needs 7 there.
- The projected-meet nMSE is not monotone in k: the threshold calls ignore its standard errors, and several cells sit
  near a threshold.
- **Collapse.** By layer 8, 3-22% of inputs (by net) have passed a layer with exactly one active unit. Their future is
  then one deterministic ray.

## 3. The forward part (Kelvin's (a)): ARC's method and gate-pattern mixtures at width 8

`scratch/fwd_study.py` (PREREG P1, P4). For each method the table gives three numbers at cuts c = 2 / 4 / 8:
- the nMSE of E[a_c] against the truth's layer moments;
- the Gaussian meet: the method's mean and covariance at c as a Gaussian, pushed through the TRUE noisy downstream
  (2^18 draws; an oracle isolating the forward part, added in code, not registered for KP);
- the registered meet: the degree-K polynomial fit of V_c (oracle fit on 2^19 true paths) integrated against kprop's
  moments; for GHM, draws from the mixture.
Costs to cut 2 are in MC paths.

| method (cost to c = 2) | net 203 | net 216 | net 217 | net 220 |
|---|---|---|---|---|
| KP-2 (0.1) | 7e-3 / 1e-2 / 0.6; Gauss 0.035 / 0.35 / 5.7; poly 0.47 / 190 / 7e3 | 2e-3 / 0.1 / 0.3; 0.049 / 0.19 / 0.15; 5.7e-4 / 0.11 / 8.5 | 4e-3 / 0.04 / 0.04; 2e-3 / 1.6e-3 / 0.07; 7e-3 / 0.94 / 0.78 | 8e-3 / 0.08 / 0.25; 0.12 / 0.14 / 0.16; 0.038 / 0.17 / 0.08 |
| KP-2 augmented (0.5) | 7e-3 / 1.6 / 0.9; 0.015 / 2.2 / 180; 0.17 / 7e3 / 7e5 | 1e-3 / 0.8 / 41; 0.037 / 0.52 / 7.2; 0.016 / 0.45 / 2e7 | 9e-4 / 2 / 8.6; 2.3e-5 / 2.4 / 48; 3.8e-3 / 260 / 20 | 0.02 / 1e3 / 8.5e3; 0.14 / 650 / 1.7e3; 0.01 / 4e6 / 1e11 |
| KP-3 factorized (1.5) | 4e-3 / 0.05 / 440; 3.8e-4 / 0.39 / 280; 290 / 2e5 / 6e9 | 4e-4 / 190 / 1.1e3; 2.8e-3 / 520 / 440; 0.076 / 5e6 / 5e18 | 3e-4 / 0.08 / 14; 6e-5 / 0.12 / 72; 2.7 / 3.7e3 / 2e11 | 9e-3 / 20 / 2.2e4; 0.14 / 21 / 1.4e4; 0.041 / 1.4e4 / 6e17 |
| KP-3 factorized augmented (6) | 5e-4 / 0.02 / 78; 6.4e-3 / 0.15 / 830; 200 / 1e6 / 6e13 | 1.5e-4 / 0.03 / 0.3; 7e-4 / 0.05 / 0.29; 0.095 / 0.12 / 1.6e10 | 3.5e-4 / 0.03 / 0.6; 4.7e-4 / 0.03 / 5.6; 0.93 / 24 / 1.8e7 | 1.4e-3 / 0.04 / 2.2; 0.07 / 0.48 / 1.5; 0.71 / 4.5 / 4e9 |
| KP-4 factorized (13) | 3e-4 / 1e-3 / 2.8; 7.5e-3 / 0.03 / 560; 1.1e4 / 3e9 / 2e19 | 3e-5 / 0.02 / 0.2; 4.1e-3 / 0.30 / 0.03; 4.3 / 5e4 / 1e9 | 2.6e-4 / 0.03 / 0.3; 5.1e-4 / 0.03 / 1.8; 69 / 2.5e6 / 4e14 | 5e-4 / 0.013 / 0.22; 0.044 / 0.011 / 0.017; 6.4 / 10 / 3e15 |
| K2X, exact-moment Gaussian closure (5) | 7e-3 / 7e-3 / 0.4; Gauss 0.035 / 0.31 / 3.9 | 2e-3 / 0.1 / 0.2; 0.048 / 0.16 / 0.11 | 2e-3 / 0.02 / 4e-3; 3.5e-4 / 6e-3 / 0.01 | 7e-3 / 0.07 / 0.26; 0.11 / 0.15 / 0.16 |
| GHM, tau = 1e-3 (76-81; 1.3k-1.4k to c = 8) | 1e-3 / 2e-3 / 1e-2; mixture 2.4e-3 / 7.3e-3 / 0.043 | 4e-4 / 2e-3 / 4e-3; 1.3e-3 / 3.1e-3 / 2e-3 | 5e-4 / 2e-3 / 3e-3; 1e-3 / 1.7e-3 / 5.9e-3 | 1e-3 / 6e-3 / 0.014; 6.8e-3 / 0.010 / 0.013 |
| GHM, tau = 1e-4 (254-286; 10k-12k to c = 8) | 1e-3 / 2e-3 / 5e-3; 2.3e-3 / 3.5e-3 / 9.1e-3 | 4e-4 / 1e-3 / 2e-3; 1.4e-3 / 1.9e-3 / 1.9e-3 | 5e-4 / 1e-3 / 2e-3; 9.4e-4 / 1.5e-3 / 2.9e-3 | 1e-3 / 5e-3 / 0.01; 6.7e-3 / 8.4e-3 / 8.8e-3 |

RQMC at about 30 paths has nMSE 0.012-0.015, at 2^9 paths 2.6e-4 to 5.8e-4, and at 2^13 8.8e-6 to 1.9e-5.

- **ARC's kprop does not carry the law to layers 5-8 at width 8.**
  - All five variants are exact at layer 1 (8.5e-7 to 3e-8, at the truth's resolution).
  - At layer 2 they are at 3e-5 to 2.3e-2. The augmented variants and K = 3 diverge from layers 3-5, up to 2.2e4 by
    layer 8. K = 4 (factorized) is the most stable, at 0.2-2.8 by layer 8.
  - ARC's power counting is in 1/n, which is not small at n = 8. At n = 16 and 32 (section 8) K = 2's layer-2 error is
    2-20x smaller.
- **The two meets disagree by orders of magnitude.** The registered polynomial meet extrapolates badly whenever kprop's
  higher moments are off: KP-3 factorized at c = 2 on 203 scores 3.8e-4 as a Gaussian and 290 as a polynomial. Both
  are oracles with the true downstream; neither is a deployable estimator.
- **With an oracle downstream, the law at layer 2 is close.** The best Gaussian meet at c = 2 is 2.3e-5 (217), 3.8e-4
  (203) and 7e-4 (216), against RQMC's 1.2e-2 at 30 paths; on 220 it is 0.044. The best registered polynomial meet at
  c = 2 is 5.7e-4 (216), 3.8e-3 (217), 0.01 (220) and 0.17 (203).
  - So at cut 2 the forward description alone is not always the bottleneck.
  - What a deterministic method must still do there is the meet: integrate V_2, which needs 4-8 directions under our
    rule, against that law, and treat the noise.
  - Beyond cut 2, kprop degrades by one to five orders of magnitude.
- **The gate-pattern mixture (GHM)** is our computable stand-in for the brief's "exact mixture over the realised gate
  patterns, pruned by mass". It is not exact:
  - Gaussian components are split one unit at a time into halfspace-conditioned halves, each re-Gaussianised;
  - light components stop splitting;
  - straddling units get Stein moments.
  It is the most stable forward description, but it floors at 1e-3 to 1e-2 on E[a_c], and 1.9k to 60k components
  (tau = 1e-3 to 1e-4) change its error by at most 2x.
- **Why it floors (review-narrowed).** Three approximations, besides the Gaussian piece shapes:
  - straddling free units get Stein cross-covariances (Phi_i C_ij Phi_j), not the exact Gaussian ReLU covariances the
    registration's wording promised. For a correlation of 0.9 that is 0.225 against 0.296;
  - forced gates can leave negative post-ReLU means: 15 components with 0.27% of the weight on dev 103's first layer;
  - light pieces stop splitting rather than being dropped.
  Starting from the empirical first two moments of every layer-1 piece (2^22 paths, `scratch/ghm_shape_test.py`) still
  leaves 5e-4 at layer 2 and 7e-3 to 1e-2 at layer 8, against 8e-4 and 1e-2 from the input. So better layer-1 moments
  do not cure this implementation. That test does not separate the three causes.
- **Exact pieces, a naive cost estimate** (not a lower bound; HEURISTIC). One 8-D orthant probability by a deterministic
  lattice (Genz's transformation, no random shift) moves by about 1% between 2^14 and 2^18 points (410 to 6,500 MC
  paths of FLOPs) and agrees with MC to 1%.
  - A piece also needs about 36 lower-dimensional terms for its first two moments.
  - Layer 1 has 255 pieces; layer 2 has 2.3k-2.6k sampled ones with up to 16 facets.
  - With noise, exact history regions also involve the injected noise, so the 8-D picture is itself an approximation.
  - Shared computation, adaptive integration and other representations were not examined.

## 4. The downstream (Kelvin's (b)) and the middle

- **The deep map is exact on its class (PROVEN for the noiseless net; class shares MEASURED).**
  - The deep part takes a reference ray at c_d = 32 (or 64) from the majority future class of deterministic probes, and
    J_ref = the noiseless Jacobian to the output along it.
  - Every state in the same noiseless future class maps exactly to J_ref h (4,068 of 4,096 states checked to 4e-14;
    the reviewer reproduced J(h) h = f(h) to 9e-16).
  - The implementation applies J_ref to every state without checking its class. The reference class carries 83-100%
    of the probes' mass at c_d = 32 and 100% at 64. Outside it, J_ref is an approximation with no bound.
- **The middle: direct deterministic passes beat a grid.**
  - A grid of V_8 on the top-k directions, with the radius exact and multilinear interpolation in gnomonic coordinates,
    was erratic on dev net 103: 2.5e-5 to 1.9 across k = 2-6, 8-32 points per dimension and box sizes. V_8's linear
    pieces have kinks that defeat the interpolation.
  - Evaluating V_c exactly where the meet needs it, by noiseless passes c -> c_d, costs 128 FLOPs per state per layer
    and has no interpolation error. Every arm uses it (Amendment 2).
  - The meet's integration rule still drops covariance directions: it uses 2k points along k directions for spread
    components.
- **The noise's effect on E[a_L] (P3, P10).**

| net | total noise effect, nMSE of (sigma = 1e-3 truth - sigma = 0 truth) | signed, along the truth's direction (% of SD) | deep part only, noiseless vs MC from the sampled law at 32 (normalisation inflated, see note) | noiseless downstream on the exact law at c = 8 (P3) |
|---|---|---|---|---|
| 203 | 7.2e-5 | +0.8 | 6.4e-5 | 5.5e-5 |
| 216 | 4.6e-3 | +6.8 | 4.9e-3 | 5.1e-3 |
| 217 | 2.1e-3 | +4.6 | 2.1e-3 | 2.2e-3 |
| 220 | 2.1e-4 | +1.3 | 1.6e-4 | 2.3e-4 |
| 221 | 4.9e-5 | -0.7 | 4.6e-5 | 9.4e-5 |
| 223 | 4.2e-4 | -2.0 | 4.4e-4 | 4.7e-4 |

Note: `deep_noise_effect.py` normalises by the variance of 4-draw averages, which overstates its nMSEs somewhat. The
truth-based first column is the clean measure.
- The effect is almost all in the deep part. On 4 of 6 nets it exceeds RQMC's error at 2^11 paths, and every
  deterministic arm here omits it (Amendment 2).
- **It is steep in sigma**: on net 216 the deep-part effect is 4.2e-6 at sigma = 1e-4, 4.9e-3 at 1e-3 and 5.4e-2 at
  3e-3 (same diagnostic).
- Mechanism (HEURISTIC, from dev net 103). Along the shared trajectory, the noise's accumulated angular spread (about
  8 sigma) straddles the thresholds of a few attractor units. Rectification there shifts the mean coherently. For
  small-radius paths the noise dominates outright: along the reference ray the relative effect is 64 at 0.15% of the
  median radius, 2.0 at 3%, 0.046 at the median and below 1e-4 at 16x the median (MC).
- **Deterministic noise treatments tried** (dev net 103, Amendments 2 and 3; the reviewer re-ran the K2 radii):
  - K2-Stein from point masses gives relative effects of 204 and 9.2 where MC gives 64 and 2.0. Exact bivariate ReLU
    moments are worse, at 255 and 11.5. As tables, these made the deep part 1,000-5,000x worse on three of four dev
    nets.
  - A Gaussian-sum filter (`lib/gsf.py`; it preserves first and second moments, per the reviewer) gets the median
    radius within 0.1-5%, but is 1.8-3.6x off at small radii. Its table costs 6.9e4-4.9e5 paths and leaves 2.4e-4 and
    1.3e-4 on the exact law (noiseless: 1.7e-3). Not used.
  - These tables are proxies: they start at the point mass with layer 32's noise and are applied to states that
    already carry it. Their residuals mix several approximations.
  - The previous lane's binned K2-Stein deep part (inside HYB) diverges on 4 of 6 evaluation nets (nMSE 5 to 1e4).

## 5. The estimators against RQMC at equal FLOPs

For each registered ladder, the table gives the best ratio RQMC/estimator at the estimator's own cost. RQMC is
interpolated log-log between measured budgets, including 2^3-2^6 and excluding 2^20. SH is the practical arm.

| net | RQMC nMSE at 2^5 / 2^9 / 2^13 / 2^17 | noise effect | TS: best admissible ratio (config, cost) | BO: best ratio (config, cost) | SH at 2^9 / 2^11 / 2^13 |
|---|---|---|---|---|---|
| 203 | 0.015 / 5.8e-4 / 1.5e-5 / 9.6e-7 | 7.2e-5 | 2.3x (K2X, c = 2, 28 paths) | 1.7x (k 5, m 4, 274 paths) | 6.2x / 1.4x / 0.25x |
| 216 | 0.012 / 2.6e-4 / 1.5e-5 / 2.8e-7 | 4.6e-3 | 0.85x (GHM 1e-3, c = 4, 2,375) | 0.16x (k 5, m 4, 274) | 0.06x / 0.01x / 0.003x |
| 217 | 0.012 / 3.2e-4 / 8.8e-6 / 6.1e-7 | 2.1e-3 | 54x (K2X, c = 2, 28); also GHM 1e-3: 7.4x (c = 2, 1,144), 5.1x (c = 4, 2,461); GHM 1e-4: 3.1x (c = 4, 15,391) | 0.46x (k 6, m 4, 459) | 0.16x / 0.05x / 0.004x |
| 220 | 0.015 / 5.3e-4 / 9.2e-6 / 1.0e-6 | 2.1e-4 | 0.09x (K2X, c = 8, 40) | 10x (k 4, m 8, 335); 4.3x (k 5, m 8, 1,199) | 1.9x / 0.9x / 0.04x |
| 221 | 0.013 / 4.2e-4 / 1.9e-5 / 8.0e-7 | 4.9e-5 | 10x (K2X, c = 4, 32); about 5x (K2X, c = 2, 28) | 14x (k 4, m 4, 228); 7.5x (k 3, m 4, 216); 6x (k 6, m 4, 459) | 4.6x / 2.0x / 0.35x |
| 223 | 0.012 / 3.5e-4 / 1.1e-5 / 4.9e-7 | 4.2e-4 | 0.03x (K2X, c = 4, 32) | 0.98x (k 5, m 8, 1,199) | 0.75x / 0.15x / 0.02x |

All rows are in `results/tables.md`. On net 217, TS's GHM tau = 1e-4 at c = 2 scores nMSE 9e-8, below 10/P = 6e-7, so
its ratio is unresolved and not claimed.
- **RQMC** is strong here: 3-27x MC at 2^9-2^17 paths, with nMSE x N = 0.07-0.16 at 2^13.
- **HYB** floors at 1.4e-3 on 203 and diverges on 4 nets. On 221 its K2-Stein deep part works: it beats RQMC at 2^9
  and 2^13, and floors at about 1e-5.
- **TS** (two-sided; ladder of Amendment 2: GHM tau in {1e-3, 1e-4} and K2X at cuts 2, 4, 8). Its typical error is
  1e-3 to 1e-2. It beats RQMC by 3x or more at some ladder point on 2 nets, 217 and 221, and never beyond the ranges
  shown. On the other 4 nets TS's best point is 2.3x (203, K2X at 28 paths) or below, and RQMC is 1.2-5,000x better
  at most of TS's ladder points.
- **BO** (backward-only, meeting the input; k = 2-6 directions, m = 4-16 points per angle) beats RQMC 3x or more on 2
  nets, 220 and 221. Elsewhere it is at 0.16-1.7x.
- **SH** (practical arm, not sample-free: RQMC on layers 1-8 with noise, then the deterministic downstream; 1/13 of a
  path per sample) wins 4.6-6.2x at 2^9 on 203 and 221. It plateaus at the noise effect: at most 2.0x at 2^11, and
  behind on every net by 2^13.

**What the wins are (post-hoc diagnostics `scratch/bias_decomp.py` and the reviewer's refinement runs).**
- On 5 of 6 nets the errors lie mostly along the output direction (the truth's direction). On net 220 the
  perpendicular part is 47-113% of the along part, so the full vector stays in the nMSE.
- For TS at cut 2 the along-direction error splits into a downstream part and an upstream part:
  - downstream: the noiseless downstream on the exact law minus the truth, which is almost all noise;
  - upstream: TS's estimate minus the noiseless downstream on the exact law. This mixes the forward law's error with
    the meet rule's error and any change of reference class; it does not isolate the forward law.
- In % of SD(a_L), each MC-estimated term has an SE of about 0.27 points:

| net | downstream + noise | upstream, GHM 1e-3 | upstream, GHM 1e-4 | upstream, K2X | TS total, GHM 1e-4 |
|---|---|---|---|---|---|
| 203 | -1.0 | +5.7 | +5.7 | -7.5 | +4.6 |
| 216 | -6.8 | +4.5 | +4.0 | +45 | -2.8 |
| 217 | -4.8 | +5.3 | +4.75 | +3.2 | -0.03 |
| 220 | -1.4 | +9.8 | +9.2 | +55 | +7.8 |
| 221 | +0.5 | +1.7 | +1.5 | +5.2 | +2.0 |
| 223 | +2.5 | +11.2 | +10.9 | +65 | +13.4 |

- The upstream part for GHM is positive on every net (+1.5% to +11%) and nearly independent of tau. The noise part
  ranges from -6.8% to +2.5%.
- On 217 they nearly cancel for GHM. That cancellation is real (about +4.75% and -4.78%, each about 0.3 points
  uncertain), but its precision is partly algebraic, since both terms share one diagnostic estimate.
- K2X's wins sit in between. At 28-32 paths, on 217 and 221, the single exact-moment Gaussian closure is within 2-4% of
  SD. On 217 (c = 2) that is helped by a partial cancellation (+3.2% upstream, -4.8% downstream). At c = 4 on 221 both
  terms are positive (+3.1% and +0.5%).
  - These are observed small-budget advantages on those two nets, not explained away.
  - The same closure is off 9-68% on 203, 216, 220 and 223, and no rule picks it in advance.
- **BO's wins come from coarse rules.**
  - Against the sigma = 0 truth (projection plus rule error, no noise), BO is off 0.3-4% of SD at k = 5-6, m = 8-16.
    The rule's error at m = 4-8 is of the same size and changes sign with m.
  - The reviewer refined the k = 4 rule. On 221, nMSE 9.1e-5 at m = 4 becomes 6.4e-3 at m = 32 and 64. On 220, 7.8e-5
    at m = 8 becomes 3.2e-4. There the refined noiseless projected estimate itself is good (1.45e-5) and the noise
    effect (2.1e-4) dominates.
- None of these wins comes with a rule that would pick the configuration without the truth. Across the ladder they
  are an oracle envelope.

**Best meeting point by measured total cost** (Kelvin's request).
- Smallest nMSE any registered configuration reached:

| arm | 203 | 216 | 217 | 220 | 221 | 223 |
|---|---|---|---|---|---|---|
| TS (cut 2/4/8) | 2.2e-3 | 6.4e-5 | 1.9e-6 (resolved; 9e-8 unresolved) | 6.1e-3 | 3.8e-4 | 1.8e-2 |
| BO (cut 0) | 2.5e-5 | 2.8e-3 | 7.6e-4 | 4.9e-5 | 8.1e-5 | 1.3e-4 |

- No meeting point gives a stable advantage. The input (BO) has the lower envelope on 4 of 6 nets, because the forward
  descriptions are poor beyond cut 2 and the cut-2 meet has no accurate deterministic integration rule.

## 6. Tails (P = 1e-4)

| net | p | MC relative MSE at 2^9 / 2^13 (exact) | RB at 2^9 / 2^13 | RQMC at 2^13 | BO tail: p / p_true over its 11 k >= 3 configs | median RB/BO relative-MSE ratio at BO's cost | TS tail p / p_true |
|---|---|---|---|---|---|---|---|
| 203 | 1.16e-4 | 17 / 1.1 | 0.27 / 0.025 | 0.79 | 0.23-1.21 | 0.8 (a loss) | 0-4.4 (one of 9 within 2x) |
| 216 | 1.06e-4 | 19 / 1.2 | 1.6 / 0.073 | 0.68 | 0.10-0.97 | 2.5 | below 0.003 |
| 217 | 9.7e-5 | 20 / 1.3 | 0.38 / 0.14 | 1.3 | 0.43-0.82 | 3.0 | below 0.05 |
| 220 | 9.8e-5 | 20 / 1.2 | 1.0 / 0.041 | 1.1 | 0.10-1.51 | 2.2 | below 0.07 |
| 221 | 8.8e-5 | 22 / 1.4 | 1.3 / 0.093 | 1.6 | 0.17-1.20 | 4.7 | below 0.08 |
| 223 | 1.1e-4 | 18 / 1.1 | 0.39 / 0.20 | 1.1 | 0.06-1.34 | 2.3 | below 0.02 |

- The truth's relative SE is 2-3%.
- RB (radius integrated exactly, noise rescaled to a typical radius; a sampler, slightly biased by the rescaling) beats
  MC 5-44x at 2^13 on every net. RQMC barely helps tails.
- TS's tail is mostly far too small, by 2-1,000x or exactly zero; one configuration on 203 overestimates it 4.4x.
  - Its tail is computed with an approximate output variance: the input-radial spread of each component scaled by its
    gain, rather than g^T C g.
  - So the failure is not attributable to the Gaussian pieces alone.
- BO's tail (Q_chi_k of t over |V_0| at each node) is within 2x on 6-8 of 11 configurations per net.
  - Its median advantage over RB at equal cost is 2.2-4.7x on 5 of 6 nets, and 0.8x on 203.
  - It is biased low (most ratios below 1), plausibly because the k-slice misses some high-gain directions, and it does
    not converge cleanly in m.
  - A modest lead, not a certified bound.

## 7. Secondary arms

| arm | RQMC nMSE at 2^7 / 2^9 / 2^13 / 2^17 | noise effect (deep part) | TS best ratio | BO best ratio |
|---|---|---|---|---|
| dead-noise 200, sigma = 1e-3 | 1.8e-2 / 2.2e-3 / 1.1e-4 / 8.5e-6 (at MC's level) | 1.6e-2 | 0 is its estimate: nMSE 4.3e-3 = MC at about 230 paths | same (0) |
| healthy 203, sigma = 1e-2 | 3.2e-3 / 5.4e-4 / 6.2e-5 / 1.6e-6 | 7.3e-3 | 1.2x (K2X, c = 2, 28 paths); GHM 0.2x | 0.46x |
| healthy 216, sigma = 1e-2 | 5.4e-3 / 6.2e-4 / 4.5e-5 / 1.8e-6 | 0.56 | 0.85x (K2X, c = 2); GHM 0.001x | 0.008x |
| intermediate 202, sigma = 1e-3 | 3.7e-3 / 5.5e-4 / 2.6e-5 / 1.0e-6 | 3.6e-3 | 2x (K2X, c = 2, 28 paths); GHM 0.2x | 66x at k 4, m 4 (228 paths), but 8.6e-4 at m 8 and about 4e-3 at k 5-6, m 8, i.e. the coarse-rule cancellation again |

- **Dead-noise nets.**
  - The noiseless net is dead, so every deterministic arm returns exactly 0. That scores |E a_L|^2 / tr Var = 4.3e-3,
    because the noise-driven mean is small relative to its spread.
  - RQMC gains nothing there: the input is irrelevant.
  - These nets are "trivial" only in the sense that 0 is a fair answer at small budgets. No method here estimates them
    better.
- **At sigma = 1e-2** the noise effect dominates: 7e-3 on 203, and 0.56 on 216, where the noise carries most of the
  output variance. Every deterministic arm here loses.
- **Intermediate net 202** (24% of noiseless inputs dead by layer 64): BO's best point is 66x at a coarse rule, and that
  advantage disappears when the rule is refined, as on 221.

## 8. Scaling with width (coordinator's arm)

Registered: the estimator scaling arm runs only if P5(a) holds at width 8. It does not, so only the structure is
measured (`scratch/scale_oracle.py`, one net per width, L = 32 n, sigma = 1e-3).

| quantity | n = 8 (6 nets) | n = 16 (L = 512, seed 300) | n = 32 (L = 1024, seed 300) |
|---|---|---|---|
| projected meet at the input with k = n/2 / 3n/4 | 5.3e-5 to 0.069 (k = 4) / 9.1e-6 to 1.0e-3 (k = 6) | 2e-3 (k = 8) / 3.3e-5 (k = 12) | 1.6e-3 (k = 16) / not run (k = 24) |
| first cut where 4 directions give a projected meet <= 1e-4 | 2-3 on 5 nets; on 220 already at the input (next positive cut 5) | 4 | 16 |
| downstream classes (99%) at cut 8 / 16 / 32 / 64 | 10-77 / 2-7 / 1 / 1 | 3,087 / 526 / 14 / 3 | 15,739 / 8,280 / 121 / 2 |
| K2 (kprop) error of E[a_c], c = 2 / 4 / 8 | 2e-3 to 8e-3 / 0.01-0.11 / 0.04-0.6 | 9e-4 / 4.8e-3 / 2.9e-3 | 4.5e-4 / 2.4e-3 / 1.2e-2 |
| GHM(1e-3) to c = 4: components, cost, error of E[a_4] | 3.5k-4k, 380-410 paths, 1.4e-3 to 6e-3 | 1.6k, 250 paths, 6.3e-4 | 1.4k, 220 paths, 9.1e-4 |

The n = 16 and 32 rows use the first oracle version (observation directions C^1/2 V); at the input, where C = I, the two
agree.
- **The middle lengthens with width**: downstream classes fall to 3 or fewer by layer 16-24 at n = 8 and between layers
  32 and 64 at n = 16 and 32. At layer 32 there are 1, 14 and 121.
- **The relevant input directions grow with n.** Under our rule it takes 6-7 of 8 for 1e-5 at n = 8 and 12 of 16 for
  3.3e-5 at n = 16; at n = 32, 16 directions still leave 1.6e-3, and 24 were not tested. The thresholds differ, so
  "about 3n/4" is a rough reading, not a fixed-error law.
- **The closures improve at shallow cuts.**
  - K2's layer-2 error halves from n = 16 to n = 32 (9.2e-4 to 4.5e-4), and is 2e-3 to 8e-3 on the n = 8 nets.
  - At n = 32 it is 1.2e-2 by layer 8, while the downstream becomes simple only around layer 16-32.
- **Cost against width (HEURISTIC; one net per width, so no scaling law is established).**
  - BO-style quadrature at the input needs about 3n/4 directions, so m^(3n/4) nodes: exponential in n.
  - At n = 32, almost every one of 2^14 sampled inputs has its own downstream class at layers 2-8 (16,295 distinct at
    layer 8). That suggests exact gate-pattern pieces grow exponentially in n at fixed depth.
  - Closures cost n^(K+1) per layer, but in these runs their error at the depth where the downstream becomes simple
    does not shrink with n.
  - None of the routes here shows a polynomial cost for fixed error. That is a reading of one net per width, not a
    bound.

## 9. Error budget

| part | where | error | how it scales with FLOPs |
|---|---|---|---|
| forward description | cut 2 | Gaussian meets 2.3e-5 to 0.14 (best 2.3e-5 to 0.044); polynomial 5.7e-4 to 1e4; GHM 9e-4 to 6.8e-3 (oracle downstream) | kprop: fixed cost (0.1-13 paths), no refinement; GHM flat from 76 to 286 paths |
| forward description | cuts 4-8 | **kprop 1.6e-3 to 1e4 (Gaussian), >= 0.11 (polynomial); GHM 1.5e-3 to 4e-2** | **GHM flat from 400 to 12k paths**; exact pieces: a naive estimate of >= 1e5 paths per layer |
| meet (value function at the cut) | cut 2 | needs 4-8 directions under our rule; the 2k-point rule drops covariance directions | **no accurate deterministic integration rule found** |
| meet | cut 8 | 1-4 directions on 5 of 6 nets; direct evaluation exact at its nodes | about 0.05 paths per evaluated state |
| deep part (noiseless) | layers 24-256 | exact on its class (4e-14) | one matvec per state |
| **noise** | layers 33-256 | **5e-5 to 5e-3 (above 1e-4 on 4 of 6)** | **omitted; GSF lowers it 7-13x on one dev net at 7e4-5e5 paths, still 100x above RQMC there** |
| BO projection | input | 0.3-4% of SD at k = 5-6 and m = 8-16, including rule error (nMSE 1e-5 to 1.4e-3) | 6-7 directions needed for 1e-5 under our rule |
| BO rule | input | about as large as the projection error; the sign changes with m | slow: midpoint rule on S^(k-1) for a function with kinks |

Each arm is held back by at least one floor that does not fall with FLOPs: the forward description beyond cut 2, the
cut-2 meet, the noise, or BO's projection and rule. Removing one alone would not bring these arms to RQMC beyond about
2^11 paths on most nets.

## 10. Pre-registered predictions: scorecard

| prediction | outcome |
|---|---|
| P1: kprop K <= 4 has nMSE of E[a_c] > 1e-3 at every c in 5..8 on all healthy nets; its meet loses to RQMC at the same total FLOPs at c >= 4 on >= 5 of 6 | **First clause: hit** on the 4 nets with the forward study (minimum 4.3e-3, KP-2 on 217 at c = 5). **Second clause**: diagnostic support with the registered polynomial meet. Every meet at c >= 4 is at least 0.077,
above RQMC at a nominal 25-path total cost; the downstream cost is nominal, so this is not a completed equal-compute
test. With the unregistered Gaussian meet, one exception: KP-2 on 217 at c = 4 (1.6e-3 against RQMC's 1.5e-2 at about 25 paths). The refutation clause (e_8 <= 1e-4) is not met |
| P2: < 1,000 downstream classes at cut 8, > 5,000 upstream histories, < 30 at cut 16 | hit on all 6 (10-77 against 9,652-18,233; 2-7 at cut 16) |
| P3: the downstream given the exact law at c = 8 reaches 1e-5 with k <= 5 at <= 1,000 paths on >= 5 of 6 | **miss** for the tested configurations: the noiseless D0 downstream is off 5.5e-5 to 5.1e-3 on all 6 nets, which is the omitted noise effect. This establishes failure of the tested downstream, not the registered refutation condition (> 10,000 paths) |
| P4: the best forward description at its best c is >= 10x worse than RQMC at the same total FLOPs at every budget on >= 5 of 6 | **diagnostics oppose the prediction; the formal equal-compute condition is not established.** At cut 2, with oracle downstreams and a nominal 25-path downstream cost, the best forward description is within 10x of RQMC or better on all 4 nets: Gaussian meet 2.3e-5 to 7e-4 on three; registered polynomial meet better than RQMC on 216, 217 and 220 (2-33x) |
| P5(a): TS beats RQMC >= 3x at some budget on >= 4 of 6 | **not met** (2 of 6: 217 and 221). Predicted: not met |
| P5(b): TS never loses by more than 2x up to 1e6 paths | **not met** (loses 10-1,000x). Predicted: not met |
| P6: SH beats RQMC >= 3x at 2^11-2^17 on >= 4 of 6 | **miss** (at most 1.97x in that range, on 221; 4.6-6.2x only at 2^9) |
| P7: TS's tail off by > 2x on >= 4 of 6; RB beats MC >= 3x on >= 4 of 6 | hit / hit (RB with its noise-rescaling bias) |
| P8: secondary arms reported, no pass/fail | done: 200 (dead-noise), 202 (intermediate), and 203 and 216 at sigma = 1e-2 |
| P9: input needs k >= 5 for 1e-5 on >= 4 of 6 (>= 4 for 1e-4); k rises toward the input on >= 4 of 6; BO wins 3x on at most 2 of 6 | hit (6-7 on all 6; for 1e-4, 3-7, with 5 of 6 at >= 4). Rise toward the input: hit as an endpoint comparison (cut 8 against the input, 5 of 6; not on 221); the counts are not monotone through the intermediate cuts. BO wins: hit (2 of 6); the reviewer notes that halving deterministic costs gives 3 |
| P10: deep noise effect > 1e-4 on >= 2 of 6; on those nets no deterministic arm stays within 2x of RQMC beyond 2^13 paths | first clause hit (4 of 6). **Second clause missed on 217** under a pointwise reading of "stays within 2x beyond 2^13 paths": TS (GHM 1e-4, c = 4) wins 3.1x at 15,391 paths, a cancellation-assisted win |
| Scaling (registered: the advantage stays constant or grows with width) | not tested: the estimator arm was conditional on P5(a) |

**Deviations.** Registered before the affected evaluation results (Amendments 1-3):
- the deep part became noiseless;
- the grid was replaced by direct evaluation;
- tails were added for TS and BO;
- the forward study ran on 4 nets;
- the secondary arms and scaling were trimmed.

Not registered, found or made later:
- RQMC replicates 8 and 2 at 2^17 and 2^20 (registered: 16 and 4);
- meet draws 2^18 and polynomial-fit draws 2^19 (registered: 2^20);
- kprop's x1.3 cost sensitivity not run;
- the Gaussian meet for kprop added beside the registered polynomial meet;
- GHM's straddling units use Stein cross-covariances, though the registration says exact K2 moments;
- the post-hoc "stable" direction count and the oracle's corrected direction construction (rerun after review);
- the small-budget RQMC runs and the signed decomposition (both post hoc).

## 11. What is proven, measured, heuristic

- **PROVEN**:
  - for the noiseless net, every state in the reference ray's future gate class maps to the output exactly by J_ref
    (piecewise linearity); checked numerically;
  - positive homogeneity makes the radius integrals in BO and RB exact for the noiseless projected problem; the
    reviewer confirmed BO's sphere weights and the E[chi_k] factor;
  - at layer 1 the law is an exact mixture of 256 orthant pieces; every K >= 2 kprop variant reproduces E[a_1].
- **MEASURED**: everything in sections 2-8. Forward study on 4 nets; directions, classes, estimators and decomposition on
  6; scaling on one net per width.
- **HEURISTIC**:
  - the noise mechanism;
  - the exact-piece cost estimate;
  - the cost-against-width reading;
  - "what limited these methods" as a general statement beyond the methods built here.

## 12. Review

One GPT-6 review (gpt-6-astra, high effort; `review/review_gpt6.raw`, prompt `review/prompt_gpt6.md`). Its verdict was
"major revision. Retain the narrow negative M1 result; revise the causal claims, scoring, and accounting before treating
this as evidence of fundamental blockers."

It reproduced all four registration hashes, the sampler 1/P corrections, the admissible 54x TS ratio on 217, the 10.3x
and 14.5x BO ratios on 220 and 221, and GHM's covariance discrepancy and negative means. It also reproduced the deep K2
effects (204 and 9.2; 255 and 11.5 with exact pair moments) and ran the BO refinement quoted in section 5.

Findings and changes:
1. **The 330x TS ratio on 217 violated the registered truth-resolution rule** (9e-8 < 6e-7), and "its neighbours lose"
   was false. Now: unresolved, and the admissible wins on 217 are listed (K2X 54x; GHM 7.4x, 5.1x, 3.1x).
2. **The cancellation reading went too far.**
   - The "forward" term mixes the forward law with the meet rule.
   - K2X's win on 221 is not a cancellation.
   - Now: "upstream" term; the wins are split into cancellations (GHM on 217, BO on 221) and closures that happen to
     be accurate on that net (K2X on 217 and 221).
3. **"Every error is one signed scalar" was false** (perpendicular 47-113% on net 220). Now: mostly along the output on
   5 of 6 nets; the vector stays in the nMSE.
4. **The S-based directions used C^1/2 V as observation directions; they should be C^-1/2 V.** Fixed, and the per-cut
   oracle was rerun on all 6 nets. Counts at cuts 1-5 dropped by 1-3; the input is unchanged at 6-7. The counts are now
   labelled as describing this projection rule.
5. **The direction count had been switched to a "stable" criterion after registration.** Now the registered smallest k
   is reported, with the stable k beside it where it differs.
6. **GHM has further approximations**: Stein cross-covariances, negative forced-gate means, and pruning that stops
   splitting. Now they are named, and the floor is attributed to this implementation, not to piece shape alone.
7. **"Every operation is charged" was false.** Now the counters are called approximate, the omissions and overcharges
   are listed, and the reviewer's factor-2 sensitivity is quoted: M1 still fails.
8. **RQMC reporting.** The shared truth, ddof = 0, the undeclared replicate counts and censored 2^20 points were
   flagged. Now 2^20 is excluded from interpolation, and the replicate deviation and shared-truth caveat are stated.
   compare.py is superseded by analyze.py.
9. **P1 and P4 were scored with the unregistered Gaussian meet.** Now both meets are reported. P1's second clause is a
   hit with the registered metric. "P4 holds at cuts 4-8" was removed.
10. **The deep-noise diagnostic's labels and normalisation.** "K2-table residual" in the evaluation logs is D0, and the
    nMSE is inflated by 4-draw averaging. Now the truth-based total noise effect is primary and the proxies are stated.
11. **The exact-piece cost is not a bound.** Now it is called a naive estimate, with the noise-dimension caveat.
12. **Exact evaluation is not exact integration.** J_ref is unchecked outside its class, the meet rule drops
    directions, and the TS tail uses an approximate variance. All stated.
13. **Scorecard repairs**: P1, P3, P4 and P6 (1.97x); P10's second clause is missed on 217; the full deviation list.
14. **Narrower wording** on: input counts, BO's "converged" bias, HYB on 221, TS tails, BO's 0.8x on 203, single-net
    scaling, "as hard as the original integral", and the noise-scale definition. All changed.

Round 2 (`review/review_gpt6_r2.raw`, same model). Verdict: **minor revision**. It confirmed:
- the winning-net counts (TS 2 of 6, 54.0x and 10.2x; BO 2 of 6, 10.3x and 14.5x), so the negative M1 result survives;
- all 108 threshold entries of the corrected per-cut table;
- the noise-effect table;
- the BO tail medians.

It asked for, and this version makes:
- removal of the universal "RQMC wins above 2^13" line;
- P1 (0.077 is the minimum polynomial meet at c >= 4), P3 and P4 scored as diagnostics rather than completed tests,
  P9 as an endpoint comparison, and P10 under a stated pointwise reading;
- the polynomial meet beating the nominal comparator on 3 nets, not 1;
- the K2X win on 217 noted as cancellation-assisted;
- K2X's c = 2 losses (10-31x);
- the per-cut table's provenance, and the square brackets applying to 1e-5 only;
- the n = 8 scaling entries corrected, and the scaling wording narrowed;
- GHM described as the implemented approximation.

It also noted:
- TS's own direction construction (`bvf.relevant_directions`) still uses C^1/2 V. It sets only the axes of TS's 2k-point
  meet rule for spread components. BO is unaffected: its directions come from the input sensitivity, with no
  covariance.
- analyze.py prints the unresolved TS ratio without enforcing the 10/P rule, and samplers.py's SE uses ddof = 0. The
  report applies the rule by hand, and the SEs are not used in any decision.

## 13. Files

- `PREREG.md`, `prereg_hashes.txt`; `notes/questions.md` (assumptions), `notes/dev_findings.txt`.
- Code:
  - `lib/ghm.py`: gate-history mixture.
  - `lib/bvf.py`: noiseless passes, the deep part, probes, the direct meet and the TS tail; the grid, which is not used.
  - `lib/bo.py`: hyperspherical rule, the BO estimate.
  - `lib/gsf.py`: Gaussian-sum filter.
  - `scratch/*.py`: census, truth, samplers, rqmc_small, oracle_structure, oracle_k_by_cut (v2; v1 kept as
    oracle_k_by_cut_v1.py), fwd_study, ts_run, bo_run, sh_run, p3_test, deep_noise_effect, bias_decomp, scale_oracle,
    and the dev diagnostics (ds_diag, deep_diag3, gsf_test, ghm_shape_test, test_lib).
- Results: `runs/*.json` (kbycut2_* are the corrected per-cut runs); tables in `results/tables.md` and
  `results/census_table.md`; logs in `logs/`.
- Review: `review/review_gpt6.raw`, `review/prompt_gpt6.md`.
