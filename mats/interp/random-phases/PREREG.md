# Random-net phases: pre-registration (written 2026-10-29 ~14:25Z, before any main run)

Question (Kelvin, 2026-10-29): in very deep chaotic nets the inputs should not matter, so a backward description from
the weights should give the output distribution; in ordered nets inputs matter more; the critical case is interesting;
should the right representations differ by phase? Target this feeds (program M1): a sample-free estimator that matches
sampling on deep random MLPs at depths that grow with width.

## Seen before writing this (declared)
- Mean-field theory only (mf.py, mf_predict.py -> mf_predictions.json): q-, c-maps with noise; numbers below.
- Smoke tests on development seed 100, width 64, tanh-chaos, sigma = 0.01 (never used in results): branching
  R(0 -> L) = 0.997 / 0.965 / 0.49 / 0.017 at L = 8 / 16 / 32 / 64, top Lyapunov exponent 0.124 per layer;
  backward: MF stand-in at depth 32 nMSE 0.15 (k = 1) to 0.002 (k = 29), MM stand-in <= 6e-4, kprop from the input
  0.005 at L = 64, input-law swap 'aniso' <= 2e-3, 'point' 1.04 (L = 1) to 0.004 (L = 64), Gaussian control ~0.
  Saturated fraction |a| > 0.9 at layer 64: 0.33. Timing tests of ARC's kprop at width 256 (no accuracy looked at).
- Prior lanes: work/arc-noisy (ARC's family, depth 32: noise share of output variance 0.6 / 5.4 / 29 / 83% at
  sigma = 0.003 / 0.01 / 0.03 / 0.1 on dev net 100; Gaussian with true moments at layer 8-24 then real layers: nMSE
  8e-5 to 5e-4; participation ratio 2.4 at depth 32, 25 at depth 8), work/cm-interp-R (LN nets).

## Nets (pnets.py)
x ~ N(0, I_n), n in {64, 256}, depth up to 64, every layer n x n; noise after every activation with sd sigma r_l,
r_l = activation RMS of layer l in the noisy net (seeded 2^12-path pilot), sigma in {0, 0.001, 0.003, 0.01, 0.03, 0.1}.
Output of the depth-L net: a_L (post-activation, before its noise). Weight seeds 0, 1, 2 (dev seed 100 for smoke only).
- tanh, no LN, W iid N(0, sw2/n), b iid N(0, 0.05) (Schoenholz et al.'s sb2), same standard normals for every gain:
  tanh-ord2 sw2 = 0.8 (chi1 0.647, lambda_mf = 0.5 ln chi1 = -0.218), tanh-ord 1.0 (0.759, -0.138),
  tanh-crit 1.761 (1.000), tanh-chaos 4.0 (1.342, +0.147; chi_c 0.867, c* 0.165), tanh-chaos2 9.0 (1.843, +0.306;
  chi_c 0.800, c* 0.039). chi_c = slope of the cross-covariance map at the different-input fixed point.
- relu-he: ARC's family as in work/arc-noisy (W iid N(0, 2/n) every layer, no bias, no LN; same weights for the
  same seed; the first 32 layers equal arc-noisy's nets). Critical (chi1 = 1).
- relu-ln: ReLU then LN (no affine), no bias; chaotic, chi1 = pi/(pi-1) = 1.467 (lambda_mf 0.192).
Why not cm-interp-R's LN-tanh: under LN without bias the c-map slope at c = 1 is >= 1 for any nonlinearity, so its
"ordered" gain 0.25 was ordered only because of sigma = 0.3 (chi = f'(1)/(1 + sigma^2)); at sigma <= 0.1 it is
critical-to-chaotic. The classic no-LN tanh with bias has all three phases at small noise.

## Measures
1. Forgetting (branch.py): M base paths, K continuations from h_m at m in {0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 56};
   R(m -> L) = tr Var(E[a_L | h_m]) / tr Var(a_L) (one-way ANOVA, debiased; jackknife SE over 8 blocks). m = 0 is the
   input share. L*(sigma) = first depth with R(0 -> L) < 0.5 (linear interpolation; seed mean). Also S(L) =
   tr(Sigma_w^-1 Sigma_b) and the Gaussian-proxy total variation between the output laws of two typical inputs,
   TV = 2 Phi(sqrt(S/2)) - 1. Lyapunov: QR on 16 tangent vectors along 16 noisy trajectories, layers 9-64.
2. Backward (backward.py): truth by plain MC (n = 256: 2^17 paths; n = 64: 2^18). Stand-ins at depth d pushed by
   sampling through the real layers d+1..L: MF = N(m 1, v I) with mean-field per-neuron moments (input-independent);
   MM = N(true mean, true covariance) (reference, not input-independent); PT = one real state (expected nMSE =
   R(d -> L), from measure 1). Input-law swaps through the whole net: 'aniso' (N(0, C), tr C = n, eigenvalues ~ 1/i),
   'point' (x0 + 0.1 N(0, I)). nMSE = mean_j (est_j - truth_j)^2 / mean_j Var(a_L,j), both MC variances subtracted.
   k*(tau) = smallest k (in the grid) with nMSE <= tau there and at every larger k of the grid; tau in {1e-1, 1e-2,
   1e-3, 1e-4}; "matches the truth to sampling accuracy" = tau = 1e-4 (n = 256 resolution about 2e-5).
   Deductive: ARC's linear_kprop + nonlin_kprop, K = 2 simple, from MF or MM at depth d (and from the input: ARC's
   standard), noise added to K2's diagonal; FLOPs by ARC's counter. Plain sampling at equal FLOPs: nMSE = 1/N with
   N = FLOPs / (flopscope cost of one depth-L path). A win = deductive nMSE below 1/N.
3. Critical-case structure (relu-he, also relu-ln): per layer, fraction of neurons with mixed gates over inputs and
   noise (min(p, 1-p) > 0, 0.01, 0.1), margins |E z_j| / SD(z_j) (total) and |E[z|x]| / SD(z|x) (noise only),
   noise-flip fraction, participation ratios and PCs for 90 / 99% of Var(a_l), share of E|a|^2 in the mean,
   cos(PC1, mean). Gate entropies are left to the gate-entropy lane.

## Predictions (each with its test)
P1 Chaotic tanh (both gains, n = 256): R(0 -> L) stays near 1 and drops through 0.5 at L*(sigma) = a + ln(1/sigma) /
   lambda_eff; the least-squares slope dL*/d ln(1/sigma) over the five sigmas is within 25% of 1/lambda_1 (QR, same
   width) and within 25% of 1/lambda_mf (6.8 and 3.3). Mean-field L*: chaos 44.8 / 37.4 / 29.2 / 21.6 / 12.9, chaos2
   23.7 / 20.2 / 16.3 / 12.7 / 8.5 (sigma = 0.001 ... 0.1); measured within 20% or 4 layers. Same for relu-ln
   (mean-field 36.6 / 30.8 / 24.5 / 18.4 / 11.4, slope 1/0.192 = 5.2). n = 64: L* later (smaller lambda_1).
P2 Ordered tanh: the same logarithmic law with |lambda|: slope within 25% of 1/|lambda_mf| (7.2 ord, 4.6 ord2);
   mean-field L*: ord 48.2 / 40.2 / 31.5 / 23.5 / 14.6, ord2 32.8 / 27.8 / 22.3 / 17.2 / 11.6. Mechanism differs
   (the input's deviation decays below the noise floor instead of the noise growing above the signal) but the law
   is the same.
P3 Critical nets (tanh-crit, relu-he): no logarithmic law. Where two or more L* are measurable, successive
   increments of L* per sigma step grow by >= 1.5x (constant within 30% in P1-P2). relu-he: R(0 -> L) at n = 256 is
   above its mean-field value at every L >= 16 for sigma >= 0.01 (finite-width norm mode, positive homogeneity), e.g.
   R(0 -> 32) at sigma = 0.1 about 0.15-0.25 against mean-field 0.027; at sigma <= 0.003, R(0 -> 64) >= 0.9.
   tanh-crit (mean-field R(0 -> 64) = 0.999 / 0.995 / 0.945 / 0.628 / 0.042) within 0.1 of mean-field.
P4 TV versus variance share (n = 256): the inputs keep mattering for the full output law well after their variance
   share is small: wherever R(0 -> L) >= 0.01, TV >= 0.5; the depth where TV < 0.5 is at least 3 layers past L*
   in P1-P2 nets.
P5 Backward, chaotic tanh: the MF stand-in works without noise. nMSE_MF(k) decays exponentially at about
   ln(1/chi_c) per layer (0.14 chaos, 0.22 chaos2; fitted rate within 40%), for every sigma including 0;
   k*(1e-3) <= 40 (chaos) and <= 25 (chaos2) at L = 64; k* at sigma = 0 within 1.5x of k* at sigma = 0.1.
   Single-state stand-in (PT) needs noise: R(d -> L) -> 0 only for sigma > 0, after about ln(1/sigma)/lambda layers
   (same law as P1). Input-law swaps: 'aniso' nMSE <= 1e-3 by L = 40 (chaos) / 25 (chaos2) for every sigma incl. 0;
   'point' <= 1e-2 by L = 64 for both chaotic gains for every sigma incl. 0.
P6 Backward, ordered tanh: at sigma = 0 every stand-in fails in nMSE terms (nMSE_MF > 0.1 for all k <= L - 8: the
   output variance collapses at the same rate as the stand-in's bias); with noise, k*(1e-3) grows with ln(1/sigma),
   roughly [ln(1/tau) + 2 ln(1/sigma)] / ln(1/chi1): >= 40 at sigma = 0.1, beyond 64 at sigma <= 0.01.
P7 Backward, critical: the MF stand-in fails for relu-he at every k <= 63 and every sigma (nMSE >= 1e-2; the overall
   gain of a finite-width He-ReLU net is a random factor remembered by homogeneity); tanh-crit reaches 1e-2 only at
   sigma = 0.1 if at all. 'aniso' swap: relu-he nMSE >= 1e-3 at every depth <= 64 for sigma <= 0.03. MM (true
   Gaussian) at relu-he depth 64 from d = 32: 1e-5 to 1e-3 (arc-noisy's depth-32 range).
P8 Deductive (kprop K = 2 from MF at L - k) vs sampling at equal FLOPs: ordered tanh at sigma >= 0.01 wins by >= 10x
   at some k; chaotic tanh within a factor 3 either way at its best k (n = 256); relu-he loses by >= 10x at every k.
   ARC's full K = 2 from the input: nMSE <= 1e-4 in the ordered nets at every depth, >= 1e-3 at depth 64 in the
   chaotic and critical nets.
P9 relu-he structure (n = 256, sigma <= 0.01): fraction of neurons with total margin < 1 falls roughly as c / l with
   depth (fitted exponent between -0.6 and -1.4), about 0.1-0.25 at l = 32, and the same at n = 64 (width-free
   fraction); participation ratio of Var(a_l) 2-4 at l = 32 and <= 3 at l = 64, cos(PC1, mean) >= 0.95 from l = 32;
   at sigma = 0.1 the PR at l = 32 is >= 10. Noise-only near-boundary fraction (noise margin < 1) at l = 32 below 2%
   at sigma = 0.001 and above 20% at sigma = 0.1.
P10 Chaotic tanh saturation: fraction of |a| > 0.9 at depth >= 16 = mean-field 0.56 (chaos2), 0.32 (chaos) +- 0.05.
   Lyapunov: |lambda_1 - lambda_mf| <= 0.02 per layer at n = 256 for tanh; relu-he lambda_1 in [-0.02, 0.02].

## What would change the picture for M1
- If P5 holds (chaotic E[out] fixed by the last O(log(1/tau)) layers, width-free), a sample-free estimator in the
  chaotic phase needs an accurate description of only a few final layers: depth stops being the problem, and the
  bottleneck is the closure error of the stationary law. If P7 holds, ARC's family (critical) has no such shortcut:
  the input's scale and direction persist polynomially, and any backward-only description must carry them.
- P9's c / l law would mean about c n / l near-boundary gates per layer, i.e. a constant number at depth
  proportional to width: the quantity a boundary-aware family would pay for.

## Runs and budget
CPU only, at most 3 processes, nice 15, one thread each. branch.py: n = 256 seeds 0-1 (M = 256, K = 32), n = 64
seeds 0-2 (M = 512, K = 32), sigma > 0. backward.py: n = 256 seed 0, sigma in {0, 0.001, 0.01, 0.1}, five nets
(tanh-ord, -crit, -chaos, -chaos2, relu-he); n = 64 seeds 0-1, all six sigmas, same nets. Changes after this file
is hashed go into dated amendments below.

## Amendment 1 (2026-10-29 14:29Z, before any backward result; branch results seen: 2 runs, tanh-ord2 sigma = 0.001)
Added to backward.py for relu-he only, after all other estimates (earlier random streams unchanged): MFr, the MF
stand-in rescaled so its activation RMS equals the net's true per-layer RMS (a single scalar per layer), at depths
8-63. Purpose: separate the overall-gain error (homogeneity) from the direction/shape error at criticality.
Prediction P7b: relu-he MFr still fails, nMSE >= 1e-2 for every k <= 48 at L = 64, for every sigma; removing the
gain error lowers the error by >= 3x relative to MF at k >= 32 for sigma <= 0.01.
Analysis addition: relative error of the mean vector, nMSE x mean_j Var(a_L,j) / mean_j E[a_L,j]^2, to show absolute
accuracy where nMSE (relative to a collapsing variance) is large.

## Amendment 2 (2026-10-29 14:35Z; runs only, no prediction changed)
Time budget: width-64 branch runs use seeds 0-1 (not 0-2); width-64 backward runs use seed 0 at all six sigmas and
seed 1 only at sigma = 0, 0.01, 0.1. Results seen so far: 6 branch runs (n = 256, sigma = 0.001) and one backward run
(tanh-ord, n = 256, sigma = 0.01).

## Amendment 3 (analysis addition, no prediction changed)
backward.py now also stores, for every stand-in and input-law swap, the relative RMS error of the per-neuron output
variances (floor about sqrt(2/NS), 1.1% at NS = 2^14), to speak to the output law beyond its mean. Backward runs
started before this amendment (tanh-ord, tanh-crit, tanh-chaos at n = 256, sigma = 0.01) lack it.

## Amendment 4 (2026-10-29 15:10Z; runs only, no prediction changed)
Host load average 55-66, so the queue was trimmed further: width-256 branch seed 1 only at sigma = 0.001, 0.01, 0.1;
no width-64 backward seed 1; width-64 backward at sigma = 0.003 and 0.03 only for relu-he. Dropped job lines are in
jobs_dropped.txt.

## Note (2026-10-29 15:47Z): review received, no prediction changed
GPT-6 review of code, numbers and the draft (review/review_gpt6.md, 15:43Z). Amendment 3 above was appended at
14:36Z (hash in prereg_hashes.txt). Analysis changes after the review, all reporting only: relative error of the mean
vector reported as relative L2 error (square root), explicit seed selection, a resolution table from the Gaussian
controls, a raw-count column for the equal-cost comparison, the TV diagnostic renamed. Predictions are scored clause
by clause in the report, including k = 63 (a stand-in at depth 1).

## Amendment 5 (2026-10-29 16:03Z; runs only, no prediction changed)
Width-64 backward runs take about 7 minutes each at nice 19 on this host, so width 64 is cut to: backward at
sigma = 0 and 0.01 (seed 0), branch at all five sigmas (seed 0). Dropped lines are appended to jobs_dropped.txt.
