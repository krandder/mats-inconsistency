# Random nets by phase: what is forgotten, and when does a backward description work? (29 October 2026)

Answer to Kelvin's question of 29 October ("On Random Nets. I am not convinced at all ..."), aimed at the program's
target M1 (a sample-free estimator that matches sampling on deep random MLPs at depths that grow with width). Code,
logs, raw results and regenerated tables: `work/random-phases/` (`python analyze.py > results/tables.md`; every number
below is in that file). Predictions were written and hashed before the main runs (`PREREG.md`, `prereg_hashes.txt`).
Amendments: 1 adds a scale-matched stand-in, 3 adds an output-variance metric, 2, 4 and 5 trim runs because the host
load was 50-66 (width 64 ended up with one seed and, for the backward runs, sigma = 0 and 0.01 only). One GPT-6
review (Codex, high effort) checked code, numbers and claims; section 7 lists what it found and what changed.

## Summary

In one paragraph: Kelvin's intuition is right for the chaotic phase and for E[output]: there the last few dozen
layers fix the output's mean and variances, whatever the input law and with or without noise, so a backward
description from the weights works (a Gaussian placed 24-40 layers before the output gets E[output] to nMSE 1e-3).
It is wrong for the other two phases in different ways: ordered nets forget their input just as fast (by the same
ln(1/sigma)/|lambda| clock) but the output variance collapses with it, and critical nets, which is where ARC's He-ReLU
family sits, barely forget at all by depth 64 unless the noise is large. The representations do differ: a Gaussian
around one trajectory (ordered), a stationary law set by the last layers (chaotic), a collapse direction plus a
remembered scale plus near-boundary gates (critical). For M1 this locates the depth problem at the critical point,
where closure errors are not contracted away.

Nets: x ~ N(0, I), every layer n x n, depth up to 64, noise after every activation with standard deviation sigma
times that layer's activation RMS. tanh with bias (sb2 = 0.05) at five gains: ordered (sw2 = 0.8 and 1.0; mean-field
Lyapunov exponent lambda = -0.22 and -0.14 per layer), critical (1.761), chaotic (4.0 and 9.0; lambda = +0.15 and
+0.31); ARC's He-ReLU family exactly as in work/arc-noisy (critical); ReLU with LayerNorm (chaotic, lambda = +0.19).
Width 256 unless stated; forgetting runs on two weight seeds, backward runs on one. These are measurements on single
random nets at widths 64 and 256 and depth <= 64, not proofs or scaling laws.

1. "The input does not matter" means two different things, and Kelvin's backward idea needs only the second.
   (a) One input: does the output's distribution given that input still depend on it? Only noise can erase that.
   (b) The input law: do E[output] (ARC's estimand) and other low-order statistics of the output depend on which
   input law was fed in? In the chaotic phase they stop depending on it, with or without noise. The full output law
   does not forget without noise (a noiseless tanh net with square Gaussian weights is injective, so it preserves the
   total-variation distance between input laws exactly); chaos moves that information into fine structure that means
   and variances no longer see.
2. Chaotic phase. (a) With noise sigma, the input's share of the output variance halves at depth
   L* = ln(1/sigma)/lambda + const: the fitted slope dL*/d ln(1/sigma) over sigma = 0.001-0.1 is 7.2 and 3.4 layers
   against 1/lambda_1 = 7.1 and 3.4 measured from Jacobian products (tanh), 5.6 against 5.4 (ReLU+LN), and every L* is
   within 1.5 layers of mean-field theory. (b) E[a_64] forgets the input law at the contraction rate of the
   decorrelated fixed point, ln(1/chi_c) per layer (theory 0.143 and 0.223; fitted 0.150 and 0.20-0.21), not at
   lambda, and noise plays no role: a Gaussian that knows only the mean-field per-neuron moments, placed k layers
   before the output and pushed through the real last k layers, gives E[a_64] to nMSE 1e-3 from k = 34-40
   (lambda = 0.15) and k = 24 (lambda = 0.31) at sigma = 0, 0.001, 0.01 and 0.1 alike, and the per-neuron output
   variances to within 0.2 points of the 0.5-0.8% sampling floor by k = 48 and 32. So for E[output] and the output
   variances, yes: in the chaotic phase the last few dozen layers decide them, and a backward description from the
   weights works.
3. Ordered phase. Single inputs are forgotten by the same law, L* = ln(1/sigma)/|lambda| + const (slopes 4.5 and 7.3
   against 1/|lambda_1| = 4.5 and 7.0). All inputs converge onto one deterministic trajectory with near-Gaussian noise
   around it: ARC's K = 2 propagation matches the truth to our resolution (|nMSE| <= 1e-5 at every depth to 64, all
   sigma; resolution about 2-3e-5). A backward stand-in gets the mean vector right to 1.4% with 32 layers to go
   (sigma = 0), but the output's own variance has shrunk to 4e-8 of its mean square, so its nMSE is 4.8e3. Only noise
   sets a floor: nMSE 1e-3 at k = 36 for sigma = 0.1 and k = 56 for sigma = 0.01; at sigma = 0.001 only from depth 1
   (k = 63). "Inputs matter more" holds for the first ln(1/sigma)/|lambda| layers only.
4. Critical point (ARC's He-ReLU and critical tanh). Little is forgotten by depth 64 unless sigma is large: at
   sigma <= 0.01 the input explains 89-100% of the output variance at depth 64 (He-ReLU 0.89 at sigma = 0.01 against
   0.60 in mean-field theory: the finite-width overall gain is remembered through positive homogeneity); at sigma =
   0.1, L* = 13 (He-ReLU) and 23 (tanh). Swapping the input law moves E[a_64] by nMSE 2.5e-3 to 6.8e-3 at sigma <=
   0.01. The mean-field stand-in fails for He-ReLU wherever it is placed from depth 2 to 63 (nMSE 0.07-57 at sigma <=
   0.01; rescaled to the net's true layer RMS, 0.09-8.1). An input-independent backward description does not work
   here; whether a weight-dependent one could is open (the oracle Gaussian with the true moments reaches nMSE <=
   1.3e-3).
5. The right representations do differ by phase. Ordered: a Gaussian around one trajectory. Chaotic: a stationary,
   high-dimensional law set by the last O(log(1/tolerance)) layers (participation ratio 63-91 of 256; 32% and 56% of
   units saturated, |a| > 0.9). Critical: one collapse direction (He-ReLU participation ratio 4.0-4.2 at depth 32 and
   1.8-2.0 at depth 64, 94% of E|a|^2 in the mean) plus a remembered overall scale plus a band of near-boundary gates
   (19% of neurons within one standard deviation of their boundary at depth 32, 12% at depth 64; noise alone flips
   1.3% of gates at sigma = 0.001 and 59% at sigma = 0.1, depth 32).
6. For M1 (hypotheses with their evidence). In the chaotic phase depth looks harmless: ARC's K = 2 from the input
   beats plain Monte Carlo at equal cost at 256 x 64, sigma = 0.01, by 8.1x (lambda = 0.15) and 59x (lambda = 0.31)
   by ARC's FLOP count (6.0x and 44x at raw count), and its error does not grow from depth 32 to 64 (3.3e-4 and
   3.3e-4; 4.6e-5 and 4.5e-5). At the critical point, where ARC's family sits, K = 2's error grows with depth at
   sigma <= 0.01 (He-ReLU 8.3e-4 at depth 32, 2.1e-3 at 64; critical tanh 9.6e-4 and 7.0e-3), and at depth 64 it gains
   1.3x (He-ReLU; 0.9x at raw count) or loses 2.6x (tanh). With large noise (sigma = 0.1) the He-ReLU error stops
   growing (1.9e-4 at 32, 1.7e-4 at 64; 16x over plain sampling by ARC's count, 12x raw), but there the noise is most
   of the output variance. Our reading: the depth problem in ARC's setting belongs to the critical point, where
   closure errors are not contracted away. A backward-only, input-independent description is not a route there; in
   the chaotic phase it needs a fixed number of final layers whatever the total depth. Width (64 against 256, one
   seed): at fixed depth 64, K = 2's error falls about n^-1.5 in both phases, faster than sampling's n^-1, but along
   depth = width / 4 He-ReLU's K = 2 only keeps pace with plain sampling (1.6x at 64 x 16, 1.3x at 256 x 64).

## 1. Setup

- Nets (`pnets.py`). tanh: z_l = W_l h_{l-1} + b_l, W iid N(0, sw2/n), b iid N(0, 0.05), the same standard normals for
  every gain, so the tanh regimes differ only by sw2. He-ReLU: ARC's family as in work/arc-noisy (W iid N(0, 2/n) in
  every layer, no bias, no LayerNorm; the first 32 layers are arc-noisy's nets for the same seed). ReLU+LN: LN after
  ReLU, no affine, no bias. Noise h_l = a_l + sigma r_l eps, r_l the RMS of a_l in the noisy net from a seeded
  2^12-path pilot (part of the object; every "sample-free" statement below is conditional on these noise scales).
  Output of the depth-L net: a_L, before its noise. Development seed 100 was used only for smoke tests (declared in
  PREREG.md).
- Why not the LayerNorm tanh nets of work/cm-interp-R: under LN without bias the slope of the correlation map at c = 1
  is at least 1 for any nonlinearity, so those nets' "ordered" gain was ordered only because of their noise (sigma =
  0.3); at sigma <= 0.1 it is critical to chaotic. The no-LN tanh with bias has all three phases.
- Mean-field theory (`mf.py`, 96-node quadrature; chi_1 accurate to about 0.5%): q- and c-maps with the noise on the
  diagonal; lambda_mf = 0.5 ln chi_1 with chi_1 = sw2 E[phi'(z)^2]; chi_c = slope of the cross-covariance map at the
  different-input fixed point.
- Forgetting (`branch.py`): M base paths (256 at width 256, 512 at width 64), K = 32 noise continuations from the
  state at depth m; R(m -> L) = tr Var(E[a_L | h_m]) / tr Var(a_L) by one-way ANOVA (debiased; jackknife SE over 8
  blocks). m = 0 is the input's share; L* = first depth with R(0 -> L) < 0.5. Lyapunov exponents: QR on 16 tangent
  vectors along 16 noisy trajectories, layers 9-64. Distinguishability: S = tr(Sigma_w^+ Sigma_b) and the projected
  equal-covariance Gaussian diagnostic TV_G = 2 Phi(sqrt(S/2)) - 1 (a diagnostic, not a measured total variation;
  resolution about 0.1-0.2).
- Backward (`backward.py`): truth by plain Monte Carlo (2^17 paths at width 256, 2^18 at width 64). Stand-ins for
  the state at depth d = L - k, pushed by 2^14 (width 64: 2^15) sampled paths through the real noisy layers d+1..L:
  MF = N(m 1, v I) with the mean-field per-neuron moments (input-independent; knows the input scale); MFr = MF rescaled
  to the net's true layer RMS (He-ReLU); MM = N(true mean, true covariance), an oracle reference whose moments come
  from the truth paths (its debiasing is approximate, section 7). Input-law swaps through the whole net: 'aniso' =
  N(0, C), tr C = n, eigenvalues ~ 1/i; 'point' = a cloud x0 + 0.1 N(0, I). Metric (ARC's): nMSE = mean_j (est_j -
  truth_j)^2 / mean_j Var(a_L,j), both Monte Carlo variances subtracted. Resolution, from an independent Gaussian-input
  control scored the same way: RMS 1e-5 to 3.5e-5 for the tanh nets, 3e-5 (sigma = 0) to 9e-5 (sigma = 0.01) for
  He-ReLU. Deductive version: ARC's linear_kprop + nonlin_kprop (K = 2 simple, bias, noise added to K2's diagonal)
  started from MF at depth d or from the input; cost by ARC's counter (the raw count is 1.31-1.36x higher; both are
  given); plain sampling at equal cost has nMSE 1/N with N = cost / (flopscope cost of one depth-L path).

## 2. Forgetting of a single input (needs noise)

### 2.1 The logarithmic law in the ordered and chaotic phases
Depth L* at which the input's share of the output variance falls to one half (width 256; mean of seeds 0-1 at
sigma = 0.001, 0.01, 0.1, seed 0 elsewhere; mean-field value in brackets), slope of L* against ln(1/sigma), and the
measured top Lyapunov exponent:

| net | lambda_1 measured (mean-field) | L* at sigma = 0.001 / 0.003 / 0.01 / 0.03 / 0.1 | slope | 1/abs(lambda_1) |
|---|---|---|---|---|
| tanh ordered sw2 = 0.8 | -0.223 (-0.218) | 32.5 / 27.1 / 22.0 / 17.0 / 11.7 (32.8 / 27.8 / 22.3 / 17.2 / 11.6) | 4.49 | 4.47 |
| tanh ordered sw2 = 1.0 | -0.143 (-0.138) | 48.5 / 39.9 / 30.8 / 22.6 / 14.8 (48.2 / 40.2 / 31.5 / 23.5 / 14.6) | 7.34 | 6.99 |
| tanh chaotic sw2 = 4 | 0.141 (0.147) | 46.2 / 38.4 / 29.8 / 21.9 / 13.1 (44.8 / 37.4 / 29.2 / 21.6 / 12.9) | 7.20 | 7.08 |
| tanh chaotic sw2 = 9 | 0.299 (0.306) | 24.5 / 20.9 / 16.8 / 13.0 / 8.7 (23.7 / 20.2 / 16.3 / 12.7 / 8.5) | 3.44 | 3.35 |
| ReLU + LN | 0.187 (0.192) | 37.1 / 30.9 / 24.7 / 18.3 / 11.4 (36.6 / 30.8 / 24.5 / 18.4 / 11.4) | 5.55 | 5.36 |
| tanh critical | -0.004 (0) | > 64 / > 64 / > 64 / > 64 / 23.3 (- / - / 235 / 80 / 23.6) | - | - |
| He-ReLU (ARC) | -0.005 (0) | > 64 / > 64 / > 64 / 54.9 / 12.9 (370 / 173 / 74 / 32 / 11.7) | - | - |

- Ordered and chaotic nets follow the same logarithmic law with 1/|lambda|, within 5% in the slope. The mechanisms
  differ (ordered: the input's deviation from the common trajectory decays below the noise floor; chaotic: the noise
  grows until it fills the attractor), but the clock is the same. Increments of L* per factor 3.3 in sigma are
  constant within 20% in all five nets. Seed-to-seed differences in L* are at most 1.7 layers.

### 2.2 The critical point
- Input share of Var(a_64), He-ReLU (mean of two seeds at sigma = 0.001, 0.01, 0.1): 0.998 / 0.978 / 0.887 / 0.454 /
  0.060 at sigma = 0.001 / 0.003 / 0.01 / 0.03 / 0.1, against mean-field 0.993 / 0.941 / 0.600 / 0.082 / 0.0002.
  Critical tanh: 0.999 / 0.994 / 0.938 / 0.605 / 0.040 against 0.999 / 0.995 / 0.945 / 0.628 / 0.042. The two He-ReLU
  seeds differ a lot at sigma = 0.1 (input share at depth 32: 0.105 and 0.210), as a net-specific gain would.
- So critical tanh follows mean-field theory, while He-ReLU remembers its input much longer than mean-field theory
  says. The difference is the overall scale: by positive homogeneity a random finite-width gain per input is carried
  to the output unchanged (the top principal component of Var(a_l) is the mean direction, cosine 0.99 from depth 32).
- From sigma = 0.1 to 0.03, He-ReLU's L* grows 4.3x (12.9 to 54.9) against 2.7x in mean-field theory and 1.6-1.7x in
  the logarithmic nets. That is consistent with a power law in 1/sigma, but with one finite increment it is not a test.

### 2.3 Variance share is not distinguishability
At width 256 the Gaussian diagnostic TV_G between two typical inputs' output laws stays above 0.5 until the input's
share of the variance is down to 0.5-1.6% in every ordered and chaotic cell, which takes another 11-28 layers after L*
(tanh sw2 = 9, sigma = 0.001: L* = 24.5, TV_G < 0.5 from depth 43; tanh sw2 = 4, sigma = 0.01: L* = 29.8, from depth
58). A small per-direction signal adds up over 256 directions. For He-ReLU at sigma = 0.1 the threshold is 6.7%: its
input-driven variance sits in a few directions.

### 2.4 Width 64 against width 256 (one seed each, sigma = 0.01 unless stated)
- ARC's K = 2 from the input, nMSE at depth 64: tanh sw2 = 4: 3.1e-3 (width 64) and 3.3e-4 (256); sw2 = 9: 4.0e-4 and
  4.5e-5; He-ReLU: 1.7e-2 and 2.1e-3. Each is a factor 8-9 for a factor 4 in width, about n^-1.5, at fixed depth.
  Plain sampling at equal cost improves only like n^-1 (K = 2's cost per layer is about n^3, a path's n^2), so K = 2's
  gain grows with width at fixed depth: sw2 = 4: 3.4x to 8.1x; sw2 = 9: 26x to 59x; He-ReLU: 0.65x to 1.26x (ARC's
  count; raw count 0.48x to 0.94x).
- Depth at width 64: in the chaotic nets K = 2's error is again flat beyond about depth 16-32 (sw2 = 9: 4.1e-4,
  6.2e-4, 5.1e-4, 4.0e-4 at depths 16, 32, 48, 64); for He-ReLU it grows (2.2e-3, 6.7e-3, 5.4e-3, 1.3e-2, 1.7e-2 at
  depths 8, 16, 32, 48, 64).
- Along depth proportional to width (depth = width / 4): He-ReLU K = 2 has nMSE 6.7e-3 at 64 x 16 and 2.1e-3 at
  256 x 64, i.e. a gain over plain sampling of 1.6x and 1.3x by ARC's count (1.2x and 0.94x raw). Over this one
  factor of 4 it neither pulls ahead nor falls behind. Two points, one seed: an observation, not a scaling law.
- The backward stand-in in the chaotic nets needs a similar number of final layers at both widths: k*(1e-3) = 48-56
  (width 64, sigma = 0.01 and 0) against 36-40 (256) for sw2 = 4, and 24 at both widths for sw2 = 9; fitted decay
  rates 0.12 and 0.19 per layer at width 64 (0.150 and 0.20 at 256; mean-field 0.143 and 0.223). The Gaussian with
  the true moments (MM) is less exact at width 64 in the sw2 = 4 net (3e-4 to 7.5e-4 for k <= 16, against <= 6e-5 at
  256): the per-layer laws are less Gaussian at small width.
- Critical tanh at width 64: K = 2 from the input reaches nMSE 5.5e-2 at depth 64 (7.0e-3 at width 256), a gain of
  0.17-0.19x over plain sampling.
- He-ReLU structure at width 64: the near-boundary fraction is 0.27 / 0.20 / 0.08 at depths 16 / 32 / 64 (width 256:
  0.22 / 0.19 / 0.12), i.e. 17 / 13 / 5 neurons; the collapse is stronger (participation ratio 1.5 at depth 32 and
  1.0 at depth 64, against 4.2 and 2.0 at width 256). Along depth = width / 4 the near-boundary count is 17 (64 x 16)
  and 31 (256 x 64).
- Forgetting at width 64 (seed 0): the logarithmic law holds with the width-64 Lyapunov exponents, which are smaller
  in the chaotic nets (lambda_1 = 0.119, 0.271, 0.172 for sw2 = 4, sw2 = 9, ReLU+LN, against 0.141, 0.299, 0.186 at
  width 256) and larger in magnitude in the ordered ones (-0.157, -0.235). Slopes against 1/|lambda_1|: 8.29 vs 8.42,
  3.77 vs 3.69, 5.96 vs 5.80 (chaotic), 7.06 vs 6.35 and 4.53 vs 4.25 (ordered). L* is accordingly later in the
  chaotic nets (sigma = 0.001: 51.0, 26.2, 38.7 against 46.2, 24.5, 37.1) and earlier in the ordered ones. Critical
  tanh: L* = 63.9 at sigma = 0.03 and 21.1 at 0.1; He-ReLU: 13.7 at 0.1, beyond 64 below.

## 3. Backward-only estimation of E[a_L]

nMSE of E[a_64] from a stand-in placed k layers before the output (width 256, seed 0):

| net, sigma | stand-in | k = 1 | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 62 | 63 | k* at 1e-3 / 1e-4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tanh ordered sw2 = 1, 0 | MF | 1.9e7 | 2.9e6 | 4.6e5 | 1.8e4 | 4.8e3 | 330 | 42 | 1.3 | 0.18 | 0.044 | none / none |
| tanh ordered sw2 = 1, 0.01 | MF | 2.1e3 | 317 | 50 | 1.9 | 0.53 | 0.036 | 4.7e-3 | 1.7e-4 | 2.5e-6 | 1.0e-6 | 56 / 60 |
| tanh ordered sw2 = 1, 0.1 | MF | 21 | 3.0 | 0.45 | 0.017 | 4.5e-3 | 2.7e-4 | 5.6e-5 | 4.9e-6 | < res | < res | 36 / 48 |
| tanh critical, 0 | MF | 8.4 | 5.7 | 3.4 | 1.3 | 0.88 | 0.52 | 0.20 | 0.089 | 0.010 | 2.4e-3 | none / none |
| tanh critical, 0.1 | MF | 2.8 | 1.6 | 0.83 | 0.22 | 0.13 | 0.060 | 0.021 | 5.5e-3 | 7.7e-4 | 1.8e-4 | 62 / none |
| tanh chaotic sw2 = 4, 0 | MF | 0.16 | 0.066 | 0.024 | 4.4e-3 | 1.8e-3 | 2.5e-4 | 1.3e-4 | 2.4e-5 | 1.8e-5 | 1.3e-5 | 36 / 56 |
| tanh chaotic sw2 = 4, 0.01 | MF | 0.16 | 0.066 | 0.025 | 4.2e-3 | 2.0e-3 | 3.5e-4 | 1.6e-4 | 2.5e-5 | 5.0e-6 | 7.7e-7 | 40 / 56 |
| tanh chaotic sw2 = 4, 0.1 | MF | 0.14 | 0.053 | 0.019 | 2.9e-3 | 1.3e-3 | 1.4e-4 | 6.7e-5 | 5.9e-6 | < res | < res | 34 / 48 |
| tanh chaotic sw2 = 9, 0 | MF | 0.027 | 5.7e-3 | 1.1e-3 | 1.9e-4 | 1.8e-5 | < res | < res | < res | < res | 1.5e-5 | 24 / 32 |
| tanh chaotic sw2 = 9, 0.01 | MF | 0.028 | 6.0e-3 | 1.3e-3 | 1.5e-4 | 4.0e-5 | 1.2e-5 | < res | 2.4e-5 | 1.7e-5 | < res | 24 / 32 |
| tanh chaotic sw2 = 9, 0.1 | MF | 0.026 | 5.1e-3 | 1.0e-3 | 7.6e-5 | 6.1e-5 | 3.3e-6 | < res | < res | < res | < res | 24 / 24 |
| He-ReLU, 0 | MF | 24 | 49 | 57 | 56 | 27 | 14 | 0.72 | 0.22 | 0.076 | 1.0e-4 | 63 / none |
| He-ReLU, 0.01 | MF | 23 | 47 | 55 | 53 | 26 | 13 | 0.68 | 0.19 | 0.072 | 8.9e-5 | 63 / 63 |
| He-ReLU, 0.01 | MFr | 7.7 | 3.3 | 0.85 | 0.92 | 1.9 | 2.5 | 0.73 | 0.086 | - | - | none |
| He-ReLU, 0.1 | MF | 6.5 | 12 | 13 | 11 | 4.2 | 1.8 | 0.098 | 3.6e-3 | - | - | - |
| He-ReLU, 0.1 | MFr | 2.6 | 0.85 | 0.12 | 0.075 | 0.15 | 0.24 | 0.096 | 2.2e-4 | - | - | - |

"< res" = within the resolution of that run (Gaussian-control RMS about 1e-5). k = 63 is a stand-in one layer after the
input, close to the input law itself; it is reported, and it counts against prediction P7 (section 5). At
sigma = 0.001 (arc-noisy's primary level) the picture is the same: k*(1e-3) = 40 (sw2 = 4) and 24 (sw2 = 9); ordered
63; critical tanh and He-ReLU none below k = 63 (He-ReLU MF 0.079-57, MFr 0.099-8.1, MM <= 1.2e-3).

- Chaotic: the MF error decays exponentially in k at 0.150 per layer (sw2 = 4) and 0.20-0.21 (sw2 = 9), against
  ln(1/chi_c) = 0.143 and 0.223, the same at sigma = 0, 0.01 and 0.1. The rate is ln(1/chi_c) rather than lambda (0.14
  and 0.30) because two spread-out laws converge toward the decorrelated fixed point, while lambda governs how a small
  perturbation grows. In absolute terms the mean vector is right to 10% (sw2 = 4) and 2.4% (sw2 = 9) of its size with
  32 layers to go.
- Input-law swaps, without any stand-in: under 'aniso' input, E[a_L] is within nMSE 6e-5 of the truth by depth 32
  (sw2 = 4) and 24 (sw2 = 9), at sigma = 0, 0.01 and 0.1. Under 'point' (a cloud of relative width 0.1 around one
  input) it is within 6.6e-4 (sw2 = 4) and 2e-6 (sw2 = 9) at depth 64 with no noise at all: chaos spreads the cloud.
- Ordered: the MF stand-in's mean vector is off by 1.4% of its size with 32 layers to go (sigma = 0), but the output
  variance is 3.8e-8 of the mean square, so the nMSE is 4.8e3; noise raises the variance floor and the nMSE falls
  accordingly. The rate in k is ln(1/chi_1): fitted 0.28-0.31 against 0.276.
- Critical: MF and both input-law swaps fail at every depth at sigma <= 0.01 (critical tanh 'aniso' 1.1e-3 to 6.8e-3,
  He-ReLU 2.5e-3 to 1.4e-2; 'point' 0.48-1.06). Rescaling MF to the true layer RMS (MFr) lowers He-ReLU's error 13x
  at k = 32 but not at k = 48 (0.73 against 0.68), so the overall scale is only part of what is missing. At sigma = 0.1
  MFr reaches 2.2e-4 at k = 56. The oracle MM stays at <= 1.3e-3 (He-ReLU) and <= 5.7e-4 (critical tanh), in line with
  work/arc-noisy's 8e-5 to 5e-4 at depth 32.
- Beyond the mean (amendment 3): the per-neuron output variances under MF are off by 11% / 2.5% / 0.9% at k = 8 / 32 /
  48 (sw2 = 4, sigma = 0) and 1.9% / 0.5% at k = 8 / 32 (sw2 = 9), against a sampling floor of 0.5-0.8%; for He-ReLU
  at sigma = 0.01 by 1,400% / 500% / 58%.

### Deductive version against sampling at equal cost (width 256, depth 64)

Gain = plain Monte Carlo's nMSE at equal cost / K = 2's nMSE, by ARC's count (raw count in brackets). MF-started rows
are the largest k run at width 256 (k = 40), not a selected best; the full curves over k = 1, 4, 8, 16, 32, 40 are in
the tables (sigma = 0.01: in the sw2 = 4 net K = 2 from MF wins at k = 32 and 40, 2.4x and 6.0x, ties at k = 1 and
loses at k = 4-16; in the sw2 = 9 net it wins at every k, 3.1-87x).

| net | sigma | K = 2 from the input: nMSE, gain | K = 2 from MF 40 layers before the output: nMSE, gain | K = 2 at depth 32 / 64 |
|---|---|---|---|---|
| tanh ordered sw2 = 1 | 0.01 | < 2e-5, > 176x (> 131x) | 0.036, 0.12x | 2.1e-6 / 2e-8 |
| tanh ordered sw2 = 1 | 0.1 | < 2e-5, > 176x (> 131x) | 2.7e-4, 16x (12x) | < res |
| tanh critical | 0.01 | 7.0e-3, 0.38x (0.29x) | 0.51, 0.008x | 9.6e-4 / 7.0e-3 |
| tanh critical | 0.1 | 9.4e-4, 2.9x (2.1x) | 0.062, 0.07x | 4.1e-4 / 9.4e-4 |
| tanh chaotic sw2 = 4 | 0.01 | 3.3e-4, 8.1x (6.0x) | 7.2e-4, 6.0x (4.5x) | 3.3e-4 / 3.3e-4 |
| tanh chaotic sw2 = 4 | 0.1 | 2.1e-4, 13x (9.7x) | 4.1e-4, 10.5x (7.8x) | 2.4e-4 / 2.1e-4 |
| tanh chaotic sw2 = 9 | 0.01 | 4.5e-5, 59x (44x) | 5.0e-5, 87x (65x) | 4.6e-5 / 4.5e-5 |
| tanh chaotic sw2 = 9 | 0.1 | 3.6e-5, 74x (55x) | 4.0e-5, 108x (81x) | 2.8e-5 / 3.6e-5 |
| He-ReLU | 0 | 2.4e-3, 1.06x (0.79x) | 13.9 | 9.5e-4 / 2.4e-3 |
| He-ReLU | 0.01 | 2.1e-3, 1.26x (0.94x) | 13.8 | 8.3e-4 / 2.1e-3 |
| He-ReLU | 0.1 | 1.7e-4, 16x (12x) | (MF fails; not run with MFr) | 1.9e-4 / 1.7e-4 |

K = 2 started from the true moments (oracle) at depth 64 - k shows how closure errors propagate: in the sw2 = 4 net
the error saturates after about 16 layers (2.0e-4, 3.0e-4, 3.3e-4, 3.4e-4 for k = 8, 16, 32, 40; sigma = 0.01); in
He-ReLU it keeps growing (1.7e-4, 7.7e-4, 2.1e-3, 1.8e-3) and in critical tanh too (5.4e-4, 1.6e-3, 4.7e-3, 5.1e-3).
Caveats: one weight seed; plain Monte Carlo, not the best sampler (the Kerdock design is about 2.4x better at
256 x 32 for He-ReLU, work/arc-noisy); costs omit setup.

## 4. Critical-case structure for the boundary-aware family (He-ReLU, width 256)

Per-neuron statistics over inputs and noise from the truth runs (sigma = 0.01; sigma = 0 is within 0.02 everywhere):

| depth | gates mixed (min(p, 1-p) > 0.01) | dead | margin < 1 SD | < 2 | < 3 | participation ratio | PCs for 90% / 99% | mean share of E|a|^2 |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.68 | 0.01 | 0.36 | 0.65 | 0.90 | 27.3 | 60 / 160 | 0.82 |
| 16 | 0.49 | 0.10 | 0.22 | 0.46 | 0.69 | 11.1 | 31 / 116 | 0.90 |
| 32 | 0.35 | 0.18 | 0.19 | 0.35 | 0.55 | 4.2 | 16 / 81 | 0.94 |
| 48 | 0.32 | 0.23 | 0.15 | 0.29 | 0.48 | 2.8 | 10 / 62 | 0.93 |
| 64 | 0.26 | 0.27 | 0.12 | 0.25 | 0.35 | 2.0 | 7 / 47 | 0.95 |

- Margin = |E z_j| / SD(z_j) over inputs and noise. The near-boundary fraction (margin < 1) falls slowly, about
  1.2-1.3 l^-0.6 over depths 4-64 at sigma <= 0.01: 49 neurons at depth 32 and 31 at depth 64 (of 256). At
  sigma = 0.1 it stays at 0.29-0.34 (73-87 neurons).
- Noise alone (K = 32 draws per input): the fraction of (input, neuron) pairs whose gate the noise flips at depth 32 is
  1.3% / 3.8% / 12% / 28% / 59% at sigma = 0.001 / 0.003 / 0.01 / 0.03 / 0.1; noise-only margin < 1: 0.6% / 1.8% /
  5.7% / 15% / 34%.
- Collapse: the participation ratio falls from 27 (depth 8) to 1.8-2.0 (depth 64) at sigma <= 0.01; the input-driven
  part of the covariance is as collapsed as the total, the noise-driven part less (participation ratio 27 at depth
  32); at sigma = 0.1 the noise keeps the total at 19 (depth 32).
- Width 64: section 2.4.

## 5. Predictions against results

Clause by clause against PREREG.md (width 256 unless stated).

| prediction | result |
|---|---|
| P1 chaotic: slope within 25% of 1/lambda_1 and 1/lambda_mf | held: 7.20 vs 7.08 / 6.8; 3.44 vs 3.35 / 3.27; ReLU+LN 5.55 vs 5.36 / 5.21 |
| P1 L* within 20% or 4 layers of mean-field | held (largest gap 1.4 layers) |
| P1 width 64: later L* (smaller lambda_1) | held (sigma = 0.001: 51.0, 26.2, 38.7 against 46.2, 24.5, 37.1; slopes still within 3% of 1/lambda_1) |
| P2 ordered: slope within 25% of 1/abs(lambda_mf) | held: 7.34 vs 7.25; 4.49 vs 4.59 |
| P3 increments of L* grow >= 1.5x per sigma step (critical) | not testable: one finite increment for He-ReLU, none for critical tanh |
| P3 He-ReLU input share above mean-field at L >= 16, sigma >= 0.01 | held at depths 16, 32, 48, 64 |
| P3 He-ReLU R(0 -> 32) at sigma = 0.1 about 0.15-0.25 | mixed: two-seed mean 0.157 inside, seed 0 (0.105) outside, seed 1 (0.210) inside (mean-field 0.027) |
| P3 He-ReLU R(0 -> 64) >= 0.9 at sigma <= 0.003; critical tanh within 0.1 of mean-field | held (0.998, 0.978; largest tanh gap 0.023) |
| P4 TV_G >= 0.5 wherever R >= 0.01 | roughly: the threshold is 0.5-1.6% in ordered and chaotic nets; missed for He-ReLU at sigma = 0.1 (6.7%) |
| P4 TV_G < 0.5 at least 3 layers after L* | held (11-28 layers) |
| P5 MF decay rate within 40% of ln(1/chi_c) | held (0.150 vs 0.143; 0.20-0.21 vs 0.223) |
| P5 k*(1e-3) <= 40 (sw2 = 4) and <= 25 (sw2 = 9) at L = 64 | held (34-40; 24) |
| P5 k* at sigma = 0 within 1.5x of sigma = 0.1 | held (36 vs 34; 24 vs 24) |
| P5 'aniso' <= 1e-3 by depth 40 / 25 and 'point' <= 1e-2 by 64, every sigma incl. 0 | held |
| P6 ordered, sigma = 0: MF nMSE > 0.1 for every k <= L - 8 | held (1.3 at k = 56) |
| P6 ordered with noise: k*(1e-3) >= 40 at sigma = 0.1, beyond 64 at sigma <= 0.01 | missed: 36 and 56. The written formula [ln(1/tau) + 2 ln(1/sigma)] / ln(1/chi_1) gives 42 and 58 |
| P7 He-ReLU MF nMSE >= 1e-2 for every k <= 63, every sigma | missed at k = 63 (8.9e-5 at sigma = 0.01, 1.0e-4 at 0; a stand-in at depth 1); held for k <= 62 at sigma <= 0.01; at sigma = 0.1 missed from k = 56 (3.6e-3) |
| P7 critical tanh reaches 1e-2 only at sigma = 0.1 if at all | missed: at sigma = 0 and 0.01 it reaches 2.4e-3 at k = 63 (depth-1 stand-in); at sigma = 0.1 from k = 56 |
| P7 He-ReLU 'aniso' >= 1e-3 at every depth, sigma <= 0.03 | held at sigma = 0, 0.01 (min 2.5e-3); other sigma not run at width 256 |
| P7 MM from depth 32 to 64: 1e-5 to 1e-3 | held (6.9e-4, 8.6e-4) |
| P7b MFr still >= 1e-2 for every k <= 48, every sigma | held (smallest 0.075 at sigma = 0.1, k = 24; MFr reaches 2.2e-4 only at k = 56, sigma = 0.1) |
| P7b MFr lowers MF's error >= 3x at k >= 32, sigma <= 0.01 | missed: 13x at k = 32 and 5x at 40, but 0.93x at 48 and 2.2x at 56 |
| P8 ordered, sigma >= 0.01: K = 2 from MF wins >= 10x at some k | held at sigma = 0.1 (16x at k = 40); missed at sigma = 0.01 (0.12x at k = 40, larger k not run at width 256) |
| P8 chaotic: within a factor 3 either way | missed, in kprop's favour: 6.0x and 87x at k = 40 (sigma = 0.01) |
| P8 He-ReLU loses >= 10x at every k | held (MF-started) |
| P8 full K = 2: <= 1e-4 ordered; >= 1e-3 at depth 64 chaotic and critical | ordered held; chaotic missed (3.3e-4, 4.5e-5); critical held at sigma <= 0.01, missed at 0.1 (tanh 9.4e-4, He-ReLU 1.7e-4) |
| P9 near-boundary fraction ~ c l^a, a in [-1.4, -0.6], 0.1-0.25 at depth 32 | held at the edge (a = -0.60 to -0.62; 0.19) |
| P9 same fraction at width 64 | held at depth 32 (0.20 against 0.19), not at depth 64 (0.08 against 0.12) |
| P9 participation ratio 2-4 at depth 32, <= 3 at 64; cos(PC1, mean) >= 0.95 from 32 | ratio at depth 32 narrowly missed (4.04-4.17); the rest held |
| P9 sigma = 0.1: PR >= 10 at depth 32; noise-only margin < 1 below 2% at sigma = 0.001, above 20% at 0.1 | held (18.8; 0.6%; 34%) |
| P10 saturated fraction 0.56 / 0.32 +- 0.05; abs(lambda_1 - lambda_mf) <= 0.02 (tanh); He-ReLU lambda_1 in [-0.02, 0.02] | held |

Net: the forgetting laws and the chaotic backward predictions held; the ordered and critical backward predictions
mostly held in direction but missed several thresholds; the deductive predictions were too pessimistic for chaos.

## 6. What each finding implies for M1

M1 asks for a sample-free estimator of E[output] that matches sampling on deep random MLPs at depths that grow with
width. What these measurements say about where such an estimator is easy and where it is hard:

1. The forgetting laws (section 2). In the ordered and chaotic phases noise limits memory to about
   ln(1/sigma)/|lambda| layers; at the critical point it does not at the noise levels arc-noisy used (He-ReLU keeps
   98-100% of its input share to depth 64 at sigma <= 0.003). Implication: small noise does not shorten ARC's problem;
   only noise around 0.1, where noise is most of the output variance, does.
2. Backward forgetting in the chaotic phase (section 3). E[a_L] is fixed by the last k ~ ln(A/tau)/ln(1/chi_c) layers
   (k = 24-40 for tau = 1e-3 here), with or without noise, and k does not depend on the total depth. Implication: in
   the chaotic phase a sample-free estimator only has to describe a fixed-length suffix, so "depth proportional to
   width" costs nothing extra there; the suffix length was about the same at widths 64 and 256 (24-48 layers). This
   is a structural observation, not a major advance: ARC's existing K = 2 already wins there.
3. Whether closure errors contract (sections 3, deductive). This is the quantity that separates the phases for M1. In
   the chaotic nets K = 2's error saturates after about 16 layers and does not grow from depth 32 to 64; at the
   critical point it keeps growing (He-ReLU: 9.5e-4 at depth 32, 2.4e-3 at 64, sigma = 0; roughly depth^1.4 from
   depth 8 to 64 at width 256). With width (section 2.4) K = 2's error at fixed depth 64 falls about n^-1.5 in both
   phases, faster than sampling's n^-1 at equal cost, so at fixed depth K = 2 gains with width everywhere. Along
   depth proportional to width the phases separate: in the chaotic nets the depth costs nothing, so the gain keeps
   growing; for He-ReLU the depth growth eats the width gain, and K = 2 stays at 1.3-1.6x of plain sampling from
   64 x 16 to 256 x 64 (below 1 at raw count at 256 x 64). Implication: at the critical point an M1-grade estimator
   must have a per-layer error that falls with width clearly faster than its accumulation grows with depth; K = 2
   only keeps pace over this range. Two widths and one seed cannot fix exponents.
4. Noise as a contraction. At sigma = 0.1 He-ReLU's K = 2 error stops growing (1.9e-4 at depth 32, 1.7e-4 at 64;
   16x over plain sampling) and the scale-matched backward stand-in reaches 2.2e-4 from 56 layers back. Implication:
   noise can supply the contraction the critical point lacks, but only at a level that changes the problem.
5. Critical-case structure (section 4), for the boundary-aware family. At width 256 the critical state is a collapse
   direction (participation ratio 2-4 from depth 32), a remembered overall scale (rescaling the stand-in to the true
   RMS removes up to 13x of its error, not all of it), and 31-49 neurons per layer within one standard deviation of
   their gate boundary at depths 32-64, of which the noise alone moves 1-6% per input at sigma <= 0.003. Implication:
   a family that carries the collapse direction and the scale explicitly and treats on the order of 30-50
   near-boundary gates per layer would address what the Gaussian misses. Along depth = width / 4 the count went from
   17 (64 x 16) to 31 (256 x 64), so it is not bounded over this range, though it grows more slowly than width.
6. What would count as a major advance, given this picture: an estimator that matches sampling for He-ReLU at
   sigma <= 0.01 at depth 64 or more and keeps doing so as width grows with depth. Nothing here does that; the
   measurements locate the difficulty (no contraction at the critical point) and rule out one shortcut (an
   input-independent backward description).

## 7. Review (GPT-6 via codex-any exec, high effort) and what changed

The review (`work/random-phases/review/review_gpt6.md`, 15:43Z, model gpt-6-astra) found no implementation defect;
its own checks passed (ANOVA and jackknife, branching, mean-field noise handling, stand-in conventions, QR transport
including the LayerNorm Jacobian, and ARC's bias, starting cumulants and noise handling, to 1e-8 or better). It raised
nine issues; what changed:
1. Full-law forgetting was overstated: without noise, a tanh net with square Gaussian weights is injective, so total
   variation between input laws is preserved exactly. Claims now say "E[output] and the output variances", and point 1
   of the summary states the distinction.
2. The M1 conclusions outran the evidence (one width at the time, one seed, depth <= 64; failure of an
   input-independent stand-in does not exclude a weight-dependent one). Rewritten as hypotheses with their evidence;
   the oracle result is kept as evidence that more information can help; sigma qualifiers added.
3. The k = 63 point was excluded after the fact although P7 covers it. It is now reported and scored as a miss, and
   P7b's 3x clause as a miss.
4. MM's moments and MFr's scale come from the truth paths, so their debiasing assumes an independence that is absent;
   uncertainty was understated (the He-ReLU Gaussian control is 8e-5, not 1e-5). MM is labelled an oracle, the
   resolution table from the Gaussian controls is added, and values below it are marked "< res". MM numbers below about
   1e-4 should not be read as exact.
5. Prediction scoring now covers every clause (section 5).
6. "Relative error" was the squared relative error; now the relative L2 error (1.4%, not 0.02%).
7. The TV diagnostic is renamed and qualified; "11-19 layers" corrected to 11-28.
8. Costs: raw-count gains are given next to ARC-count gains; "sample-free" is conditioned on the given noise scales.
9. Cell corrections: MM was not <= 3e-5 everywhere in the ordered and chaotic nets (1.4e-4 at one point); L* now uses
   seed means; the amendment summary was wrong (fixed in the header).
Not done: cross-fitted MM (independent paths for the moments), which would cost another truth run per cell.
