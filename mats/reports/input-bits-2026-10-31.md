# Input bits: what the input contributes below the memory time, whether a structured sampler's gain is set by the low-order share, and samplers for the input regime (31 October 2026)

Lane: input bits (target M1), bead pool-qxszj1.22 under Kelvin's ask pool-qxszj1 ("expand to the case where some of the
bits come from the input"). Status: complete; GPT-6 review in section 10.
Work directory `work/input-bits/`: `FORMAL.md` (theory v2; v1 frozen 18:07:44Z before the co-prover's answer is
`coprover/FORMAL_v1_as_written.md`), `PREREG.md` (Parts 1, 2, 3 and Amendments 2A-2C, hashes and times in
`prereg_hashes.txt`, each before its runs), `coprover/` (gpt-6-astra rounds, prompts, raw logs, `PROVENANCE.md`),
`lib/ib.py` with `tests/test_ib.py` (all pass, `logs/test_ib.log`), `scratch/` (run and scoring scripts), `runs/` (JSON),
`results/` (scorecards and tables), `review/`.

Labels: PROVEN, PROVEN UNDER <hypothesis>, MEASURED, HEURISTIC, CONJECTURE. Provenance: [C] Claude, [A] gpt-6-astra,
[C+A] both independently.

## Summary for Kelvin

**Short answer.** Below the memory time the input reaches the output through two channels: the output's direction and
its amplitude (the input-dependent gain of the whole stack). The input radius |x| carries only 3-9%. The direction
dominates for the first ~2 sqrt(n) layers (half the variance at ARC's 256 x 32) and is a very high-order function of the
input; beyond that the amplitude carries nearly everything. On your link: interactions do grow with width (this lane
confirms the numerics lane's result on its own integrand), but the growth is not visible in the low-order part of the
variance (it is consistent with a high-order tail, whose size is not pinned down). The low-order shares, which predict a
structured method's gain, show no detected decline with width in either regime; what declines is the sampler's ability
to use them (in the calibration on synthetic pair functions, RQMC removes pairwise variance well only when its points
far outnumber the coordinates). At the budgets measured, structured sampling in the input regime buys constants of
2-7x with no robust rate improvement; at L = 2n only the dyadic multilevel sampler's gain grows with width, and slowly.

1. **What carries the input** (exact split PROVEN [C+A]; MEASURED on 4 nets per cell, n = 32-256). tr Var_x(a) =
   radius + amplitude + direction exactly. At 256 x 32: amplitude 0.49, direction 0.47, |x| 0.034 (medians; the
   weight-averaged chain gives 0.60/0.36/0.04, astra's independent chain the same). The direction share grows with width at
   fixed depth (0.04 -> 0.47 at L = 32, n = 32 -> 256) and vanishes at fixed L/n; the kernel + 1/n theory gets the
   direction/amplitude ratio within 2x at every L <= n/2. So "the radius carries the input" is right for |h_l|, not for
   |x|, and only beyond ~2 sqrt(n) layers. Correction to the stationary lane: at sigma = 1e-3 the noise already reaches
   1-20% of the input variance at L ~ n/2-n on many nets, long before T_mem ~ 4n, because the input's variance is only
   ~11/n of |h|^2 (HEURISTIC reason; MEASURED onset).
2. **Input bits** (identity PROVEN; MEASURED). No births without noise. The gates that differ between two inputs decay
   like 3n/l (the kernel's finite-angle contraction, within 25% on 113 of 116 layers l <= n/4), not at the Lyapunov rate;
   marginal entropy = 2.3 x that count. The 38 bits at layer 31 of 256 x 32 are the kernel's 44.6, inside the
   net-to-net scatter (fresh nets 39-56). Each layer's bits feed the amplitude with weight ~theta^2, so the amplitude
   collects input variance over ~100 layers; its annealed expected variance tends to C_inf = 10.65 in the joint limit
   n, L -> inf (PROVEN [A]).
3. **Interaction order of the input map** (identity PROVEN [A]; MEASURED). Its variance-weighted mean polynomial degree,
   averaged over weights, is exactly n / E_W[V] (about 19 at ARC, growing like n at fixed L/n); its mean ANOVA dimension
   is 9-12 at ARC and grows about 3x from n = 64 to 256 at L = n/2. Yet ~30% of its variance is additive and ~20%
   pairwise at every width (astra predicted 0.32 additive from a finite-width linear term).
4. **Age by Lyapunov direction** (MEASURED). Finite input differences lose their angle as the kernel says (within 10%
   at n = 256 for l <= 64, on the power-of-two layers evaluated); infinitesimal ones keep it (growth factor ~1 up to l ~ n/4); their ratio grows like l^1.5 (a sensitivity
   diagnostic, akin to a mean polynomial degree): the same mechanism that makes the noise interactions. At 256 x 32 the
   input's directional information lives in ~10 effective directions; by l ~ n the input cloud is >= 90% radial
   (amplitude) on 14 of 16 net-depths.
5. **The link you asked about** (pre-registered; MEASURED).
   - RQMC: the calibrated surrogate G = 1/(1 - S1 - (1 - rho_2(N/d)) W2_off), with the pair efficiency rho_2 measured on
     synthetic functions only, predicts 44 of 48 input-regime cells within 1.5x (median ratio 1.16). In the noise regime
     no change is detected at the measured values of the ANOVA-order generating function C(t <= 0.75) at n = 8-128 (read
     as ~0.6-0.7 first order and ~0.25 second-third order) and the MC floor from older noise is
     small (median 1.06% at n = 128, up to 3.95% on one net), yet the gain falls 5.5 -> 2.5 as the coordinates grow 512 -> 21,000 at N = 1024. Moving the n = 8
     integrand, unchanged, to Sobol' dimensions 8192+ drops its gain from ~8 to ~3. Registered verdict on "a growing
     interaction order explains the shrinking gains" (operationally: a declining additive share): REFUTED in both regimes.
   - Reconciliation (section 6.7, registered): on the same noise integrand, with the numerics lane's conditioning, the
     mean interaction order does grow (blocks 2.3 -> 3.5, coordinates 3.8 -> 7.3 from n = 16 to 128, matching the
     numerics lane), and the growth sits in the non-additive part (a proxy for its mean order rises from 6.3-8.0 at n = 16
     to 15.5-20.3 at n = 128, conservative bounds from the point estimates). Mean interaction order grows with width,
     while no shift is detected at the measured C(t) values; movement
     within a high-order tail is a possible reconciliation, but its mass and its effect on RQMC gains are not established.
   - Antithetics: parity sets the gain exactly; the even share is flat in width (noise 0.23-0.27, gain ~2; input ~0.47,
     no gain).
   - Multilevel: depth locality sets the gain; its level variances follow the amplitude's depth profile (86% of cells
     within 1.5x), and its pilot-estimated two-level gain grows with width at L = n (1.3 -> 1.9, n = 64 -> 256, Part 2)
     but not at fixed L = 32.
6. **Samplers for the input regime** (pre-registered; MEASURED on 2 fresh nets per cell; gains at N <= 4,096 with the
   one-off pilot excluded). All exponents ~1 (fitted 0.92-1.38, medians 1.05-1.16): no robust rate improvement
   established. Best: multilevel over depth with RQMC inside each level, 3-7x at L = n-2n (n = 64-128) and 2.7-3.4x at
   n = 256 (best of three allocations; one pre-set allocation gives 2-7x). RQMC alone 1.6-5x, falling with width. At
   L = 2n only the dyadic multilevel sampler's predicted gain grows with width (1.6 -> 2.5), consistent with the
   unrestricted lane's n/log^2 n law with a small constant. A quadratic control fails at n >= 128 and a learned rotation
   helps in only 3 of 9 cells (e.g. 4.3x against 2.9x at 128 x 64, little at n = 256); noisy n^2-parameter pilots are the
   likely reason (HEURISTIC; a mismatched proxy is another). With the registered pilots charged, every pilot method is
   unaffordable at a 4,096-pass budget; at larger budgets the multilevel-RQMC sampler wins at L >= n for n <= 128. The
   registered rule declared "progress" on two exponent cells; I attribute them to noise (section 7.3).
7. **Process.** gpt-6-astra co-proved (round 1 independent; theory v1 frozen first) and audited (round 2): it found three
   real bugs (a normalisation in the Lyapunov-direction run, a missing multilevel cell, an unimplemented Part 3 scorer),
   all fixed and reported both ways. Scorecard: Part 1 12 of 20 hit, Part 2 5 of 5 plus three REFUTED verdicts,
   amendments 10 of 11, Part 3 2 of 5 (section 8). **Both GPT-6 review rounds asked for major revision**: round 1 for
   overstatement (10 findings, in the summary, the order claim and Part 3's wording), round 2 for the reconciliation's
   mismatched conditioning and an unproven "10% tail"; a registered measurement (2E) settled the conditioning, and a
   third round asked for minor revision (label precision), applied (section 10).

Figures: `work/input-bits/figs/fig1_split_bits.png` (split and bits), `fig2_synthesis.png` (RQMC against the model; noise
regime), `fig3_samplers.png` (Part 3 constants against width).

## 1. Setting
- Nets as the stationary lane's Part C: x ~ N(0, I_n), h_l = relu(W_l h_{l-1}) + sigma xi_l for l < L, output a = relu(W_L h_{L-1}),
  W_l iid N(0, 2/n) from the stationary lane's weight stream. sigma = 0 unless stated (then a(x) = |x| g(x/|x|) exactly);
  ARC's point is n = 256, L = 32, sigma = 1e-3.
- Widths n = 32, 64, 128, 256; depths n/8 to 2n (4n at n <= 64) and the fixed depth L = 32. Evaluation nets: seeds
  4000-4003 (Parts 1-2), 4004-4005 (Part 3); dev nets (seeds >= 9000) only calibrated the predictions.
- Cost unit: one n x n matrix-vector product (mv); a forward pass costs L mv. Errors: nMSE = |error|^2 / tr Var(a), per net.
- Annealed (weight-averaged; the exact two-replica Gram chain, weights integrated out) and quenched (one fixed net)
  numbers are kept apart. "Median" is over the 4 nets of a cell.

## 2. What carries the input (Part 1a)

### 2.1 An exact three-term split (PROVEN [C+A])
For a fixed noiseless net, with A(u) = |g(u)| (the amplitude: the input-dependent gain of the whole stack) and
d(u) = g/|g| (the output direction),
  tr Var_x(a) = V_rad + V_amp + V_dir,  V_rad = (n - m_n^2) E A^2,  V_amp = m_n^2 Var A,  V_dir = m_n^2 E[A A'(1 - d.d')],
with m_n = E|x| and u, u' independent directions. All three are >= 0; radius-exact sampling removes exactly V_rad
(n - m_n^2 = 1/2 - 1/(8n) + ...). The split is a variance identity, not three orthogonal ANOVA components [A].

### 2.2 Theory (PROVEN for fixed L as n -> inf [A]; HEURISTIC beyond)
- Direction: V_dir ~ n (1 - c_L), c_L = kappa^(L)(0) the iterated arc-cosine kernel (1 - c_L = 0.166, 0.071, 0.025, 0.008,
  0.0023 at L = 8, 16, 32, 64, 128; its 44.4/L^2 asymptote is reached slowly).
- Amplitude: two inputs at angle theta get per-layer squared gains whose log-difference has variance
  (2/n)(5 - B(cos theta)), B the degree-2 arc-cosine covariance [C+A]. Summed along the kernel angles,
  E_W V_amp -> C_L = (1/4) sum_{j<L} [5 - B(c_j)] = 5.5, 7.2, 8.6, 9.5, 10.0 at L = 8, 16, 32, 64, 128 and C_inf = 10.65,
  n-independent. [A] proved that E_W V_amp is nondecreasing in depth with a summable increment, so E_W V_amp -> C_inf
  whenever n, L -> inf, and E_W V_dir = O(n/L^2).
- So at fixed depth the direction term grows like n and takes over; at fixed L/n it vanishes like 1/n and the amplitude
  and radius carry everything. Crossover V_amp = V_dir near L ~ 2.0 sqrt(n) [A] (about 25 layers at n = 256 [A, chain];
  24 by my kernel equation [C]).

### 2.3 Measured (MEASURED; `results/part1_tables.md` Table 1; PREREG P1.1-P1.6)
| n | L | s_rad | s_amp | s_dir (median over 4 nets; range) | chain (annealed) s_amp / s_dir | V_dir/V_amp: nets / chain / kernel |
|---|---|---|---|---|---|---|
| 32 | 8 | 0.07 | 0.49 | 0.45 (0.24-0.51) | 0.57 / 0.37 | 0.95 / 0.64 / 0.97 |
| 64 | 16 | 0.06 | 0.63 | 0.31 (0.18-0.52) | 0.67 / 0.28 | 0.52 / 0.42 / 0.63 |
| 64 | 64 | 0.07 | 0.89 | 0.04 (0.01-0.08) | 0.99 / 0.007 | 0.042 / 0.007 / 0.054 |
| 128 | 32 | 0.05 | 0.67 | 0.28 (0.18-0.33) | 0.76 / 0.18 | 0.42 / 0.24 / 0.38 |
| 128 | 128 | 0.06 | 0.93 | 0.007 (0.00-0.07) | 0.91 / 0.004 | 0.007 / 0.004 / 0.029 |
| 256 | 32 (ARC) | 0.034 | 0.49 (0.43-0.64) | 0.47 (0.33-0.54) | 0.60 / 0.36 | 0.96 / 0.61 / 0.76 |
| 256 | 64 | 0.046 | 0.76 | 0.19 (0.08-0.32) | 0.84 / 0.12 | 0.28 / 0.14 / 0.21 |
| 256 | 128 | 0.050 | 0.88 | 0.07 (0.02-0.10) | 0.93 / 0.02 | 0.084 / 0.022 / 0.058 |
| 256 | 256 | 0.052 | 0.94 | 0.006 (0.00-0.03) | 0.93 / 0.002 | 0.0065 / 0.0021 / 0.015 |

- At ARC's 256 x 32 the input variance is carried about half by the amplitude and half by the output direction (medians
  0.49 and 0.47; the annealed chain gives 0.60 and 0.36, astra's independent chain 0.59 and 0.37); the input radius |x|
  carries 3.4% (P1.1 hit). The stationary lane's "the radius carries the input" is right only for the amplitude
  radius |h_l| and only at depths beyond about 2 sqrt(n): the input radius |x| never carries more than 4-9%.
- The kernel + 1/n prediction of V_dir/V_amp is within a factor 2 in all 9 cells with L <= n/2 (factor 0.7-1.45), and at
  L >= n the nets' directions are erased faster than the kernel says (6 of 7 cells; the 7th, 256 x 512, is at a floor of
  0.9%) (P1.2: missed on that cell). The trends are as predicted: at L = 32 the direction share rises with width
  (0.04, 0.11, 0.28, 0.47 at n = 32..256), at L = n/2 it falls (0.11, 0.11, 0.08, 0.07) (P1.3 hit).
- The amplitude's log-variance is 0.70-0.96 times the theory C_L/n in all 12 cells with n >= 64, L <= n (P1.4 hit).
- Heavy tails. Per-net V varies 3-10x between nets of the same cell (e.g. 31.9, 7.9, 13.3 at ARC's point), and at L >= n
  the median net's V collapses (1e-4 to 1e-2) while the annealed mean stays near C_inf + 1/2: the typical net's radius
  decays like exp(-5L/(4n)) and the weight-average is carried by rare nets. The annealed chain therefore matched the
  4-net pooled shares within 1.5x in only 6 of 12 cells (P1.5 missed): annealed means are the wrong summary at L >= n/2
  with 4 nets; per-net medians are used in this report.
- Noise. At sigma = 1e-3 the noise is below 1% of the input variance on every net up to L = n/4, but it reaches 1-20% at
  L = n/2 to n on a quarter to all of the nets (15 of 72 cells over 1%; P1.6, assumption A1, missed). The reason
  (HEURISTIC): the input variance across inputs is only about (C_inf + 1/2)/n of |h|^2, so the noise's accumulated
  n sigma^2 L takes over the VARIANCE long before the radius reaches the noise floor that defines T_mem. So the input
  regime, in the sense "noise < 1% of the variance", ends near L ~ n/2 to n at sigma = 1e-3, not at T_mem ~ 4n.

## 3. Input gate bits (Part 1a; PREREG P1.7-P1.10)
- No births without noise; each layer's input-decided gates are the units whose gate differs between inputs. The
  expected number that differ between two independent inputs is exactly n theta_{l-1}/pi given the angle (PROVEN), and
  the kernel angle gives about 3n/l (algebraic decay, set by the kernel's finite-angle contraction c delta^{3/2} per
  layer, not by the Lyapunov gaps, which act only on infinitesimal differences) [C+A].
- MEASURED (Table 2 of `results/part1_tables.md`): the median Hamming count follows the kernel within 25% in 113 of 116
  layers l <= n/4 (misses: n = 64 at l = 13, 16, 25-38% low; n = 256 at l = 34, 26% low), and the exact chain within 25% in
  341 of 472 layers with count >= 2 (the misses are deep layers where the 4-net median of 2-5 bits is noisy) (P1.7 missed
  on both clauses). At n = 256: 128, 100, 74, 51, 33, 18, 11, 5 bits differ at layers 1, 2, 4, 8, 16, 32, 64, 128 (chain
  128, 101, 75, 51, 31, 18, 9.5, 4.6). Log-log slope over l in [8, n/4]: -0.78 at n = 128 and 256 (P1.8 hit), against a
  slope near 0 for a Lyapunov-rate decay.
- The sum of marginal gate entropies is 2.0 x Hamming at layer 1 (every unit a fair coin over inputs) and 2.24-2.39 x at
  depth in all 202 scored layers (P1.9 hit): the Gaussian-cloud closure gives 2.31 [C] = 0.735 theta n / (n theta/pi) [A].
- Where the 38 bits at 256 x 32 come from (P1.10 hit): at layer 31 the four fresh nets have 39, 45, 46, 56 bits of marginal
  entropy and 17-25 differing gates; the kernel predicts 19.3 differing gates and 2.31 x 19.3 = 44.6 bits [C+A]. The
  stationary lane's 38 is a net at the low end of this scatter. Each layer's input bits are the units whose
  pre-activation for the mean input lies within the input cloud's angular spread theta_{l-1} of zero: about
  2.3 n theta/pi bits, falling like 7n/l.
- How the bits feed the amplitude (the "combination with the radius"): layer l adds (5 - B(c_{l-1}))/4 ~ theta_{l-1}^2 ~
  9 pi^2/l^2 to the annealed V_amp (PROVEN for fixed L [A]). The disagreements fall like 1/l but their amplitude effect like
  1/l^2, so the amplitude collects input variance over O(100) layers whatever n is, then freezes. This is not a variance
  per bit: correlated gains fluctuate even when two inputs share their gates [A].

## 4. Interaction structure of the input-to-output map (Part 1b; PREREG P1.11-P1.16)

### 4.1 Theory
- PROVEN [A], checked [C]: a critical net preserves E_W|J v|^2 = |v|^2 for every input and direction, so the
  variance-weighted Hermite degrees satisfy E_W sum_k k H_k(W) = E_W E_x |J|_F^2 = n exactly. The annealed mean Hermite
  degree of the map is therefore n / E_W V. With E_W V -> C_inf + 1/2 = 11.15 at depth, the map's mean polynomial degree in
  its input grows linearly with width (chain: 4.7, 8.4, 14.3, 26.7 at L = n/2 for n = 32, 64, 128, 256; 19 at ARC's point).
- The ANOVA mean dimension (distinct coordinates) is smaller [A]: about 1/(1 - c_L) for the direction part at fixed L
  (39.6 at L = 32 in the infinite-width limit), about 2L/C_L for the amplitude when L << sqrt(n) and about sqrt(n) when
  L >> sqrt(n) (HEURISTIC [A]); my v1 had the same direction formula with a finite-width factor Gamma_L [C].
- A finite-width additive part survives (HEURISTIC [A]): the amplitude kernel C_L(0) - C_L(t) has a linear term with
  coefficient 2 sum_j c_{j+1} K_j'(0) = 2.98 (L = 32) and 3.10 (L -> inf), so the first-order share of the full map tends to
  (3.10 + 0.5)/(10.65 + 0.5) = 0.32 at fixed L/n. Its origin is a preferred input direction of each fixed net,
  b ~ (2 pi n)^(-1/2) sum_i |w_i| w_i, carried forward by later layers [A]. (My v1 had only a CONJECTURE here.)

### 4.2 Measured (MEASURED; Table 3 of `results/part1_tables.md`)
| n | L | W1 (degree 1) | S1 (first order, degrees <= 4) | W2 (off-diagonal) | s_even | mean dimension D (range) | chain D_ann |
|---|---|---|---|---|---|---|---|
| 32 | 8 | 0.37 | 0.43 | 0.32 (0.26) | 0.44 | 2.7 (2.0-3.0) | 2.4 |
| 64 | 32 | 0.23 | 0.30 | 0.29 (0.22) | 0.50 | 4.9 (3.5-7.1) | 3.7 |
| 128 | 64 | 0.26 | 0.32 | 0.26 (0.20) | 0.47 | 7.5 (4.9-10.5) | 5.4 |
| 256 | 32 (ARC) | 0.23 | 0.27 | 0.22 (0.18) | 0.46 | 10.3 (8.7-12.5) | 9.6 |
| 256 | 128 | 0.27 | 0.32 | 0.26 (0.20) | 0.46 | 14.6 (8.4-21.0) | 8.8 |
| 256 | 256 | 0.28 | 0.33 | 0.26 (0.21) | 0.46 | 12.2 (10.8-20.9) | 9.5 |
- The mean dimension over input coordinates is 9-12 at ARC's point (P1.11 hit) and grows with width at fixed L/n: at
  L = n/2 it is 4.9 at n = 64 and 14.6 at n = 256 (ratio 3.0; P1.12 hit; astra's sqrt(n) law predicts 2, the data are
  steeper but noisy, 8-21 across nets).
- Yet a first-order share survives at every width: 0.27-0.33 at n >= 64 once L >= n/4 (0.34-0.43 at n = 32), with a
  degree-1 share of 0.20-0.28 (0.25-0.37 at n = 32) (P1.13's band hit in all 12 cells; its "more than twice the
  infinite-width part" clause failed on 2 of 48 nets, both at n = 32), matching astra's 0.32. The univariate excess S1 - W1 equals the input-radius share within the registered band in 64 of 64 cells
  (P1.14 hit). Degree 2 holds another 0.22-0.29 (0.18-0.22 of it off-diagonal). So the input map is high-order on average
  (mean dimension ~10, mean degree ~20 at ARC) with about 30% additive and 20% pairwise variance.
- Antithetic x -> -x: s_even = 0.38-0.51 in every cell (P1.15 hit), gain 1/(2 s_even) = 1.0-1.3: flipping the input flips
  every first-layer gate, and the two inputs then behave almost independently.
- Rotation (P1.16 missed): rotating the input to the eigenbasis of the pilot's degree-2 matrix of |a|^2 raises the
  first-order share from 0.39 to 0.62 (n = 32, L = 16), 0.30 to 0.47 (n = 64, L = 32), 0.32 to 0.41 (n = 128) but only
  0.32 to 0.35 at n = 256, where the off-diagonal share falls by just 11-13% (registered: by 40%). The pilot (16,384
  inputs) cannot estimate the n^2/2 = 32,768 entries of the matrix well at n = 256: the rotation's benefit is pilot-limited
  and its pilot must grow like n^2 (HEURISTIC).

## 5. Age and decay of input information by Lyapunov direction (Part 1c; PREREG P1.17-P1.20)
Method: along a reference input, the backward Lyapunov (QR) basis of the tangent maps; two clouds at each layer l:
finite input differences (4,096 independent inputs) and infinitesimal ones (images of random tangent perturbations).
| n | l | finite angular fraction / kernel 1 - c_l | finite angular PR | tangent angular PR (n/(2l)+1) | finite radial share | Gamma: infinitesimal angle growth | Gamma / finite fraction (ratio of the medians) |
|---|---|---|---|---|---|---|---|
| 256 | 1 | 0.68 / 0.68 | 159 | 73 (129) | 0.005 | 0.96 | 1.4 |
| 256 | 4 | 0.31 / 0.32 | 57 | 27 (33) | 0.03 | 0.91 | 2.9 |
| 256 | 16 | 0.074 / 0.071 | 17 | 8.0 (9) | 0.25 | 0.97 | 13 |
| 256 | 32 | 0.026 / 0.025 | 9.6 | 4.2 (5) | 0.49 | 1.1 | 42 |
| 256 | 64 | 0.0088 / 0.0079 | 6.1 | 2.4 (3) | 0.79 | 1.1 | 128 |
| 256 | 128 | 0.0029 / 0.0023 | 3.6 | 1.5 (2) | 0.92 | 3.0 | 1,000 |
| 256 | 256 | 0.00025 / 0.00061 | 1.9 | 1.2 (1.5) | 0.99 | 1.9 | 7,800 |
| 64 | 16 | 0.047 / 0.071 | 5.5 | 2.1 (3) | 0.64 | 0.82 | 18 |
| 64 | 64 | 0.0064 / 0.0079 | 1.6 | 1.2 (1.5) | 0.94 | 1.6 | 250 |
(Gamma at n = 256 is corrected for a normalisation bug, section 9; the rerun is `runs/p1cfix_*`.)
- Finite input differences lose their angle algebraically, exactly as the kernel says: the angular fraction of the input
  cloud is within 10% of 1 - c_l at n = 256 up to l = 64 and within a factor 2 at every scored layer (P1.17 hit on the
  power-of-two layers that the registered runner records; partially evaluated).
- Infinitesimal input changes keep their angle: Gamma, the mean-square angle at layer l per unit input angle, stays near
  1 up to l ~ n/4 (then wanders, heavy-tailed). Gamma here is a 16-reference sample mean; its population mean is infinite
  at any finite width (two active units occur with positive probability and E[1/chi^2_2] = infinity, PROVEN [A]), so read
  it as a typical value. Their ratio, the infinitesimal-to-finite sensitivity, grows like l^1.4-1.6 (P1.19's slope clause
  hit; its Gamma band clause missed: 6 of 14 as registered with the bug, 13 of 14 on the fixed rerun, where l = 128 at
  n = 256 gives 3.04 against the band's 3). This ratio is a sensitivity diagnostic, not an ANOVA dimension (it reaches
  thousands, beyond the 256 coordinates; the per-net median ratio is 4,753 at 256 x 256 against the 7,875 of the ratio of
  medians in the table, and 180 against 248 at 64 x 64): it compares the mean-square derivative of the direction with its
  finite variance, i.e. it behaves like the direction's mean Hermite degree (HEURISTIC). A small input change is not
  erased while whole-input differences are, which is why the input map is so high-order. It mirrors the stationary lane's
  finding for noise (a lone injection decays slowly while inside the full cloud it is erased fast).
- Both clouds concentrate on a few directions: the tangent cloud's angular participation ratio is at most n/(2l) + 1
  (P1.18 hit; the rank-conditioned exponent gaps (k-1)/n), and the finite cloud spans about twice as many (P1.18 hit):
  at 256 x 32 the input's directional information lives in about 10 effective directions of R^256. The finite cloud's
  shares along the QR directions are not concentrated on the top Lyapunov vectors until l ~ n (Q1-Q3 shares 0.04-0.14 at
  l = 32-64): finite input differences are not carried by the Lyapunov directions.
- By l = n the cloud is radial: radial share >= 0.9 on 14 of 16 net-depths at l = n, 2n (P1.20 missed on 2: 0.87 and 0.89).
  This is the amplitude: the input's age profile is "direction first, gone by about 2 sqrt(n)-n layers; amplitude
  forever" (at sigma = 0).

## 6. Synthesis: is a structured method's gain set by the low-order share? (Part 2; PREREG H2.1-H2.6, Amendment 2A)

### 6.1 The model and its calibration (registered before the data)
- RQMC (scrambled Sobol', scipy LMS + digital shift): 1/G = sum over ANOVA orders k of rho_k(N, d) s_k, with s_k the
  order-k share in the coordinates the Sobol' points feed and rho_k the order-k efficiency (HEURISTIC [C+A]; [A] adds that
  pair efficiencies depend on the specific projections and that MC on older noise leaves a hybrid floor).
- Calibration on synthetic functions (no net data; `results/rho_calibration.json`, R = 64): order 1 is removed; an
  exchangeable order-2 form keeps rho_2 = 0.18, 0.05, 0.01 of its MC variance at N = 2^8, 2^10, 2^12 in d = 32 dimensions,
  0.43, 0.09, 0.03 in d = 64, 0.68, 0.19, 0.06 in d = 128 and 1.04, 0.44, 0.13 in d = 256: rho_2 is set by N/d,
  so at fixed N a larger dimension alone shrinks the gain; order 3 keeps 0.25-1.0.
- Antithetics: gain 1/(2 s_even) exactly (parity, not order). Multilevel: (sum_j sqrt(V_j C_j))^2 / budget (depth
  locality of the amplitude, not an ANOVA share).

### 6.2 Noise regime: no detected change in the low-order shares with width; the gain falls anyway (MEASURED)
Stationary lane's nets (seeds 1000, 1001; their RQMC gains unchanged) plus fresh nets 4000, 4001 (RQMC rerun with the
stationary lane's unchanged code). Shares in the RQMC window coordinates (the noise of the d most recent layers, d n <=
21,201 coordinates), medians over 4 nets:
| n | window coordinates | S1 (first order) | s_even | odd degree >= 3 share | G(2^10) | E = G (1 - S1): removal beyond first order |
|---|---|---|---|---|---|---|
| 8 | 512 | 0.66 | 0.27 | 0.11 | 5.5 | 1.87 |
| 16 | 2,048 | 0.59 | 0.27 | 0.14 | 3.8 | 1.56 |
| 32 | 8,192 | 0.65 | 0.24 | 0.11 | 3.8 | 1.34 |
| 64 | 21,184 | 0.66 | 0.23 | 0.12 | 2.8 | 0.97 |
| 128 | 21,120 | 0.62 | 0.23 | 0.15 | 2.5 | 0.89 |
- The first-order share is flat at 0.59-0.66 from n = 8 to 128 (H2.3a hit), as astra's frozen-radius closure predicted
  (0.67 -> 0.65 -> 2/3) [A]; the even share is flat at 0.23-0.27 (antithetic gain 1.9-2.2) and so is the odd high-degree
  share (0.11-0.15; H2.4 hit, and 2 s_even reproduces the stationary lane's antithetic pair ratios within 0.02 on all 10
  nets). No change with width is detected in the low-order shares of the noise-to-output map (its mean interaction order
  does grow; section 6.7).
- RQMC's gain still falls (5.5 -> 2.5) because what it removes beyond the first-order share fades: E = 1.9 at n = 8 and
  below 1 at n >= 64 (H2.3b hit), where the order >= 2 part is integrated slightly worse than by MC, as the calibration
  predicts for N < d. Every cell keeps the first-order floor (H2.1 hit for the noise regime).
- Registered verdict (H2.6) for the noise regime: REFUTED. "A growing interaction order explains the shrinking RQMC gain"
  fails: S1 changes by +0.03 (0.59 -> 0.62) from n = 16 to 128 while G(2^10) falls 1.54x.
- Direct test of the alternative (Amendment 2A, registered before its runs; MEASURED): the same integrand, unchanged
  (same function, same orders, same N), with its coordinates read from Sobol' dimensions 2048+ or 8192+ instead of 0+.
  At N = 2^12 the gain falls from 8.1, 6.9, 8.9, 3.2 to 3.4, 3.7, 3.0, 2.8 at n = 8 and from 6.2, 4.7, 4.1, 5.6 to 1.8,
  2.5, 2.6, 1.6 at n = 16 (median ratios 0.48 and 0.41; A2A.1 hit; off = 0 reproduces the stationary lane's gains, A2A.2
  hit). Placed in poor Sobol' dimensions, RQMC removes about the first-order share only (1/(1 - S1) = 2.4-2.9). So most of
  what RQMC removes beyond the first-order share at small width depends on the projection quality of the leading Sobol'
  dimensions, and the window integrand at large width (8,192-21,184 coordinates) necessarily occupies poor ones.
- Is the order shifting inside the non-additive part? Amendment 2B (the order generating function C(t), suggested by
  astra) answers it below (section 6.4).

### 6.3 Input regime: the calibrated low-order model predicts the gains (MEASURED; H2.1, H2.2)
RQMC over the n Gaussian input coordinates on the Part 1 nets (R = 32 scrambles, 16 at n = 256), N = 2^8, 2^10, 2^12, at
L = n/8 to n and L = 32; S1 and W2_off from Part 1b on independent samples; rho_2(N, n) from the synthetic calibration.
| n | L | S1 | W2_off | G(2^8) / G_pred | G(2^10) / G_pred | G(2^12) / G_pred |
|---|---|---|---|---|---|---|
| 32 | 8 | 0.43 | 0.26 | 3.22 / 2.76 | 3.41 / 3.02 | 4.13 / 3.12 |
| 64 | 32 | 0.30 | 0.22 | 2.16 / 1.75 | 1.92 / 2.00 | 2.67 / 2.06 |
| 128 | 64 | 0.32 | 0.20 | 1.55 / 1.63 | 2.32 / 1.94 | 2.35 / 2.05 |
| 256 | 32 (ARC) | 0.27 | 0.18 | 1.63 / 1.36 | 1.25 / 1.59 | 1.72 / 1.74 |
| 256 | 128 | 0.32 | 0.20 | 2.13 / 1.45 | 1.13 / 1.76 | 1.67 / 1.98 |
(all 48 cells in `results/part2_scores.json`.)
- Every cell keeps the first-order floor (H2.1 hit, 32 of 32 cells at N = 2^10, 2^12).
- G_pred = 1/(1 - S1 - (1 - rho_2) W2_off), with no parameter fitted to the nets, is within a factor 1.5 of the measured
  gain in 44 of 48 cells (92%), median ratio 1.16 (H2.2 hit as registered, cell medians); with per-net ratios (astra's
  reading of the default aggregation) 47 of 48 cells, median 1.11. The model is slightly low at small n (RQMC also
  removes some order >= 3 variance there: rho_3 = 0.25-0.8 in the calibration) and the n = 256 cells scatter 0.64-1.51
  (R = 16 scrambles; the three N values are not monotone, sampling noise of about 35% per variance).
- So in the input regime the low-order shares predict the gain: a surrogate in which first-order variance is removed,
  pairwise variance is removed at the exchangeable-calibration efficiency rho_2(N/d), and higher orders are left at MC
  efficiency gets 92% of the cells within 1.5x. This is a predictive test of that surrogate, not a proof of its mechanism:
  the calibration itself shows some order-3 reduction, and a net's own pair projections need not behave like an
  exchangeable quadratic form [review].
- Registered verdicts (H2.6): REFUTED along L = 32 (S1 0.30, 0.31, 0.27 at n = 64, 128, 256 while G(2^10) falls 1.9 -> 1.25)
  and along L = n/2 (S1 0.30, 0.32, 0.32 while G falls 1.9 -> 1.1). The mean interaction order does grow (D 4.9 -> 10.3 at
  L = 32, 4.9 -> 14.6 at L = n/2), but the low-order shares that the model uses barely move; in the model the gain falls
  because the dimension n grows at fixed N, which lowers the pair efficiency rho_2(N/n) as the calibration measured.
  (astra's caution, adopted: H2.6 tests a declining additive share; it cannot see variance moving between high orders.)
- As compute grows at fixed width, rho_2 -> 0, so this model predicts RQMC gains rising toward 1/(1 - S1 - W2_off) ~ 2 in
  the input regime at depth (about 2.0-2.2 at n = 64-256), independent of width; in our range they are 1.1-2.7.

### 6.4 No detected shift in the low-order part of the noise map's ANOVA-order distribution (MEASURED; Amendment 2B)
Registered before its runs (suggested by astra): C(t) = sum_S t^|S| V_S / V for the RQMC window integrand, measured by
sharing each scalar window coordinate between two histories with probability t (older noise independent), 4 nets x 8,192
pairs per width. C(t)/C(1) is the probability-generating function of the ANOVA order; C(1) = 1 - eta, eta the MC floor
from the older noise.
| n | eta (hybrid floor) | C(0.25)/C(1) | C(0.5)/C(1) | C(0.75)/C(1) |
|---|---|---|---|---|
| 8 | 0 (window = whole history) | 0.171 | 0.384 | 0.639 |
| 16 | 0 | 0.165 | 0.357 | 0.598 |
| 32 | 0 | 0.171 | 0.371 | 0.617 |
| 64 | 6e-6 | 0.163 | 0.373 | 0.616 |
| 128 | 0.011 | 0.179 | 0.371 | 0.607 |
(standard errors 0.003-0.014 per net.)
- The hybrid floor is small: median eta = 0, 0, 0, 6e-6 and 0.0106 at n = 8-128 (A2B.1 hit), so the older-noise MC part
  does not explain the falling gain.
- No shift is detected at the measured t: the generating function agrees across widths within +-0.02-0.04 (A2B.2, which
  predicted a decrease of >= 0.04 at t = 0.75 from n = 16 to 128, missed: 0.598 -> 0.607; from n = 8 to 128 it falls by
  0.032, inside the net-to-net scatter). C(t) at t <= 0.75 resolves only the low orders: a tail at order k contributes
  t^k of its mass, so moving 10% of the variance from order 10 to order 100 changes C(0.75) by 0.0056 while raising the
  mean order by 9 [review]. Read as an order distribution (non-negative least squares with orders 1, 2, 3 and a tail
  represented by order 12, `scratch/fit_orders.py`; the split between orders 2 and 3 is ill-conditioned): about
  0.60-0.70 first order, 0.20-0.33 second and third order, and 0.07-0.11 in a tail whose order these data cannot fix.
  Small-t slopes match Part 2's S1 (A2B.3 hit).
- So the low-order part of the noise map's structure in its scalar coordinates, the part a sampler like RQMC can use,
  shows no width dependence. With it, the surrogate 1/G ~ 0.1 + (pair efficiency) x 0.25 runs from about 1/0.17 ~ 6 when
  the pairs are removed (n = 8: N = 2^10 points for 512 coordinates) to about 1/0.35 ~ 3 when they are not (21,000
  coordinates), close to the measured 5.5 -> 2.5. How this squares with the growing interaction order measured by the
  stationary-numerics lane is settled in section 6.7.

### 6.5 Multilevel shared trajectory: its gain follows the depth profile of the amplitude (MEASURED; H2.5)
Two-level version of the unrestricted lane's sampler (`lib/mlmc.py`: one reference direction, prefix depth r, ideal
allocation, setup and pilot excluded, gain over radius-exact MC), level variances on 8,192 directions per net.
- Theory (FORMAL 5.4): the final level's share V2(r)/V_sph ~ s'_dir + s'_amp (C_L - C_r)/C_L, where C_r is the amplitude
  collected by depth r (section 2.2). Measured/predicted within a factor 1.5 in 31 of 36 cells (86%) with n >= 64 and
  r = L/8, L/4, L/2 (H2.5a hit); the misses are at L = 2n with n = 64, 128 (measured 0.4-0.6 of the prediction: deep
  nets' amplitude is collected faster than the kernel profile says).
- Best two-level gain (median over 4 nets): at L = n it grows with width, 1.29, 1.60, 1.87 at n = 64, 128, 256; at L = 2n
  2.30, 2.47, 2.52; at the fixed depth L = 32 it does not (1.08, 1.02, 0.93) (H2.5b hit, both adjacent differences;
  the registered scorer first dropped the n = 256, L = 32 cell by the same L >= n/4 filter as the runner; fixed, section 9).
- So the multilevel sampler's gain is set neither by an ANOVA share nor by parity but by depth locality: how much of
  the input variance is collected in a cheap prefix (the amplitude's C_r profile) against what still needs a full pass
  (the direction residual, ~ n/L^2). At fixed L/n the direction residual vanishes and the gain grows with width; at fixed
  L the direction dominates and it does not.

### 6.6 Answer to "does a growing interaction order explain why gains shrink with width?"
- The interaction order does grow with width (stationary-numerics lane; confirmed on this lane's own noise integrand in
  section 6.7 and for the input map in section 4). But for RQMC, in both regimes, the growth is not what the gains
  track. The low-order shares that predict RQMC's gain are flat in width (noise: S1 0.59-0.66, even 0.23-0.27, no shift of
  C(t <= 0.75); input: S1 ~0.30, off-diagonal degree 2 ~0.2) while the gains fall; what changes is how much of the
  pairwise (and higher) part the point set removes, which needs N >> the number of coordinates. The calibrated surrogate
  G_pred = 1/(1 - S1 - (1 - rho_2(N/d)) W2_off) predicts the input regime (92% of cells within 1.5x), and moving one fixed
  integrand to worse Sobol' dimensions reproduces the loss (8x -> 3x). The surrogate assumes that RQMC leaves the
  high-order part at MC efficiency; under that assumption the growth of the mean order, if it sits in high orders, does
  not change the gain (an assumption, not a measurement [review]). Caveat [A]: H2.6
  tests a declining additive share and cannot see variance moving between high orders; sections 6.4 and 6.7 measure
  that directly.
- Antithetics: no; parity sets the gain exactly, and the even share is flat in width (noise 0.23-0.27, gain 1.9-2.2;
  input 0.45-0.51, gain about 1).
- Multilevel: no; depth locality sets it. Its gain grows with width at L = n (two-level, Part 2: 1.3 -> 1.9 from n = 64 to
  256) and, for the dyadic version, at L = 2n (Part 3: 1.6 -> 2.5); the two-level version is flat at L = 2n (2.3-2.9 in
  Parts 2 and 3), and nothing grows at fixed L = 32.

### 6.7 Reconciliation with the stationary-numerics lane's growing interaction order (MEASURED; Amendments 2D, 2E)
The numerics lane reported that the mean interaction order of the noise map grows with width (state h_0: 1.41 -> 3.87,
output a: 1.51 -> 4.11 from n = 8 to 256, slope 0.26). Section 6.4 found no shift in C(t). Both are true; they measure
different statistics of the order distribution, and with the conditioning matched (Amendment 2E, below) both describe the
same conditional window function of the same nets.
| | numerics lane (item 6 of its report) | this lane, Amendment 2B (section 6.4) | Amendment 2E | Amendment 2D |
|---|---|---|---|---|
| function | h_0, log\|h_0\|, direction, output a, gates | output a, as the window function g(U) = E_Z a(U, Z) (older noise Z independent in the two copies) | output a, older noise shared (conditional, variance-weighted) | output a, older noise shared (conditional, variance-weighted) |
| variable groups | one layer's n-vector of noise per group (blocks) | scalar noise coordinates | scalar noise coordinates | blocks (D_block) and scalar coordinates (D_coord) |
| window | last 4n + 1 layers after a 12n deep past | the RQMC window: d = min(8n, 21201/n) layers after an 8n burn-in | as 2B | as 2B |
| estimand | mean dimension sum_S \|S\| V_S/V (first moment of the order distribution) | generating function sum_S t^\|S\| V_S/V at t <= 0.75 (dominated by low orders) | the same at matched conditioning | first moment, both groupings |
| estimator | single-block refresh ratio (consistent, bias O(1/P)) | Bernoulli coordinate sharing | Bernoulli coordinate sharing | single-block and single-coordinate refresh ratios |
The conditioning matters only when the window does not cover the whole history (n >= 64 here). Amendment 2E (registered
20:27:41Z after the second review pointed this out [review r2]) repeated 2B with the older noise shared: the normalised
conditional curve agrees with 2B's within about 0.01 (largest median difference 0.010 at n = 64, t = 0.25) at every t
and width (A2E.1 hit; e.g. t = 0.75: 0.617 vs 0.616 at n = 64, 0.602 vs 0.607 at n = 128), and the older noise's own main
effect has medians below 0.01 (single nets up to 0.014; A2E.2 hit). So at the measured t values the two conventions
agree; this is consistency at three points, not a proof that they are the same distribution [review r3]. Medians over 4 nets (4,096 paths per net in 2D, 8,192 in 2B and 2E):
| n | window layers d | D_block (numerics lane's estimand) | D_coord | additive proxy A = Cc(0.25)/0.25 (an upper proxy: A = s1 + 0.25 s2 + ...) | non-additive order proxy (D_coord - A)/(1 - A) | Cc(0.75) (2E) | C(0.75)/C(1) (2B) |
|---|---|---|---|---|---|---|---|
| 16 | 128 | 2.28 | 3.75 | 0.63 | 8.0 | 0.595 | 0.598 |
| 32 | 256 | 2.19 | 3.39 | - | 7.8 (with 2B's S1) | - | 0.617 |
| 64 | 331 | 2.65 | 5.11 | 0.69 | 15.0 | 0.617 | 0.616 |
| 128 | 165 | 3.49 | 7.32 | 0.68 | 20.3 | 0.602 | 0.607 |
- On this lane's own integrand the block-level mean order grows 1.53x from n = 16 to 128 (A2D.1 hit), in line with the
  numerics lane's output-a values (1.98, 2.22, 2.99, 3.31 at n = 16, 32, 64, 128); the coordinate-level mean order is
  larger and grows 1.95x (A2D.2 hit). With the additive proxy from 2E (the slope Cc(0.25)/0.25, which overstates the
  first-order share because it includes 0.25 x the order-2 share, so the proxy below overstates the non-additive mean
  order), the non-additive order proxy rises from 8.0 to 20.3 (A2E.3 hit; 7.3 to 18.6 with 2B's S1, A2D.3 hit);
  conservative algebraic bounds from the same point estimates are 6.3-8.0 at n = 16, 11.5-15.0 at 64 and 15.5-20.3 at 128
  [review r3], so the growth survives, while the generating function at t <= 0.75 shows no detected change.
- How both can hold: the first moment of the order distribution (the mean order) can grow without any change in the
  generating function at t <= 0.75 if the growth happens at high orders, which contribute only t^k of their mass there
  (moving 10% of the variance from order 10 to 100 changes C(0.75) by 0.006 and the mean order by 9). Movement within a
  high-order tail is therefore a possible reconciliation and fits every number above; the tail's mass is not established
  (the order fits give 10-22% depending on the order assumed for it, `results/order_fits.json`) [review r2]. Neither
  lane is wrong: the numerics lane's "interactions grow with width" is confirmed on this integrand; what this lane adds
  is that the measured low-order part (first order ~0.6-0.7, orders 2-3 ~0.2-0.3) shows no width dependence.
- For Kelvin's question: interactions grow with width, and the growth is not visible in the low-order shares that
  predict the RQMC and antithetic gains. (In the input regime likewise: mean dimension 4.9 -> 14.6 at L = n/2 with S1
  ~0.3 and W2 ~0.2 flat.)

## 7. Structured sampling for the input regime (Part 3; PREREG P3.1-P3.6)
Fresh nets 4004, 4005 (not used in Parts 1-2), n = 64, 128, 256, L = 32, n/2, n, 2n, sigma = 0. All estimators unbiased,
MSE = variance; R = 10 scrambles per N = 2^6..2^12; cost in mv; gains against plain MC at C* = 2^12 L mv. Methods:
radius-exact MC (RBMC); RQMC in the Gaussian coordinates (RQMC), radius exact (RQMC-RB), in the coordinates rotated to the
pilot's degree-2 eigenbasis of |a|^2 (RQMC-ROT); a quadratic control with exactly zero Gaussian mean, c(x) = G x +
qhat (x^T H_k x - tr H_k)/2 from a pilot (CVQ, with MC or RQMC); the unrestricted lane's shared-trajectory multilevel
sampler, two-level (ML2) and dyadic (MLD) with ideal allocation from a pilot (plug-in risk, not a measured MSE [A]), and
two-level with RQMC on both levels (ML2-RQMC, measured). Pilots: 8,192 inputs (16,384 at n = 256).
`results/part3_tables.md`; medians over the 2 nets (single-cell gains carry about +-45% noise from R = 10).

### 7.1 Finite-budget gains with setup and pilot excluded (N <= 2^12), gain over MC
These are gains at the measured budget with the one-off pilot left out, i.e. the constants a method would keep if its
pilot were amortised and its curve kept its measured slope; measurements to 4,096 points cannot establish a limiting
gain or rule out a better eventual rate [review]. ML2 and MLD are pilot-estimated ideal-allocation risks; ML2-RQMC is the
best of three fixed allocations per budget (an oracle envelope [review]); with one pre-set allocation (N1 = 4 N2) it gives
2.3, 3.9, 6.8 (n = 64), 2.1, 2.1, 6.4, 5.1 (n = 128) and 1.9, 2.2, 2.7, 3.1 (n = 256) in the order of the rows below.
| n | L | RBMC | RQMC-RB | RQMC-ROT | CVQ | ML2 | MLD | ML2-RQMC |
|---|---|---|---|---|---|---|---|---|
| 64 | 32 (n/2) | 1.06 | 5.0 | 4.7 | 1.6 | 1.3 | 1.0 | 2.9 |
| 64 | 64 (n) | 1.06 | 4.0 | 3.5 | 1.7 | 1.8 | 1.1 | 3.6 |
| 64 | 128 (2n) | 1.05 | 4.4 | 2.5 | 1.7 | 2.6 | 1.6 | 7.1 |
| 128 | 32 | 1.07 | 2.9 | 2.3 | 0.9 | 1.2 | 0.5 | 2.2 |
| 128 | 64 (n/2) | 1.07 | 2.9 | 4.3 | 0.9 | 1.3 | 0.7 | 3.1 |
| 128 | 128 (n) | 1.08 | 1.9 | 2.3 | 0.9 | 1.9 | 1.2 | 6.3 |
| 128 | 256 (2n) | 1.07 | 2.8 | 2.8 | 0.9 | 2.9 | 1.9 | 5.9 |
| 256 | 32 (ARC) | 1.03 | 1.6 | 2.0 | 0.8 | 1.0 | 0.5 | 2.1 |
| 256 | 128 (n/2) | 1.03 | 1.7 | 2.0 | 0.7 | 1.4 | 1.1 | 2.2 |
| 256 | 256 (n) | 1.04 | 4.6 | 3.2 | 0.7 | 1.9 | 1.3 | 2.7 |
| 256 | 512 (2n) | 1.03 | 2.3 | 2.9 | 0.7 | 2.7 | 2.5 | 3.4 |

- Rates (MEASURED): every RQMC-type method's MSE falls like compute^-p with fitted p = 0.92-1.38 (medians per method
  1.05-1.16) over N = 2^6..2^12, in the registered band in 55 of 55 method-cells (P3.1 hit); the MC-type methods have
  p = 1 by construction. No method shows a robustly better rate (section 7.3).
- Fixed depth L = 32: every gain falls with width (RQMC-RB 5.0, 2.9, 1.6 at n = 64, 128, 256; P3.2 hit), as Part 2 predicts.
- Fixed L/n: the RQMC family falls with width (RQMC-RB at L = 2n: 4.4, 2.8, 2.3), the two-level multilevel is flat (ML2
  at L = 2n: 2.6, 2.9, 2.7; P3.5's "increasing" clause missed), and only the dyadic multilevel grows (MLD at L = 2n:
  1.6, 1.9, 2.5; at L = n: 1.07, 1.16, 1.32), consistent with the Omega(n/log^2 n) law of the unrestricted lane [A] but,
  as a pilot-based risk prediction on two nets per cell, not a validation of it.
- The best constants come from combining the two structures: multilevel over depth (the amplitude is collected in a cheap
  prefix) with RQMC inside each level (each level is a lower-order function than the whole map): ML2-RQMC gives 3-7x at
  L = n-2n for n = 64-128 and 2.7-3.4x at n = 256, at least as good as both of its parts in 4 of 6 cells (P3.5's second
  clause hit), but its gain falls with width at fixed L/n (7.1, 5.9, 3.4 at L = 2n).
- Coordinate rotation (RQMC-ROT) beat RQMC-RB by >= 1.2x in only 3 of 9 cells with L >= n/2 (P3.4 missed): with a pilot of
  8-16 thousand inputs the n x n degree-2 matrix is too noisy at n >= 128 (Part 1b, P1.16).
- The quadratic control fails at n >= 128 (residual variance 1.0-1.45 of V; it removes 37-47% at n = 64): its 32-term
  quadratic is fitted from the pilot and adds more fitting noise than it removes, as astra predicted from a linear test
  function (P3.3 missed: 6 of 18 net-depths, 3 of 9 median cells, in band).

### 7.2 With setup and pilot charged
With the registered pilots (2-4 times the measured budget of 4,096 passes), every pilot-based method is UNAFFORDABLE at
C* = 4,096 L mv. At each method's own largest measured budget (pilot included, 3-5 times C*) the picture is mixed (post
hoc accounting): ML2-RQMC exceeds 1 in 5 of 11 cells (2.6 and 2.9 at 64 x 64 and 64 x 128; 1.8, 2.1, 2.5 at 128 x 64,
128 x 128, 128 x 256) and is at or below 1 at L = 32 and at n = 256 (0.4-1.0); RQMC-ROT exceeds 1 at 64 x 32, 64 x 64 and
128 x 64 (1.2-1.6) only; RQMC-CVQ at n = 64 (1.2-2.1) only; ML2, MLD and CVQ never (0.1-1.0). The pilot-free RQMC and
RQMC-RB win in every cell (1.6-5.0).
The charged costs omit the pilot's matrix work (an O(P n^2) product, about P mv for the rotation and 2P mv for CVQ, i.e.
1/L-2/L of the pilot's passes) [review]; including it changes these numbers by under 7%.

### 7.3 Registered verdict and my reading
- P3.6 as registered: PROGRESS, through its rate clause: two RQMC-type method-cells at n = 256, L >= n have fitted and
  last-step exponents above 1.2 (RQMC-CVQ at 256 x 256: 1.21 and 1.51; RQMC-ROT at 256 x 512: 1.24 and 1.44); the
  width-growth clause found nothing (pilot methods unaffordable with charged pilots; RQMC falls).
- My reading: this is not a better rate. Astra warned before the data that R = 10 gives each variance a ~47% relative error
  and a single-cell exponent an error of about 0.5; the same cells' other RQMC methods have 1.09 (RQMC-RB at 256 x 512) and
  the CVQ curve steepens because RQMC removes the control's own fitting noise (its gain is below RQMC-RB's in that cell:
  2.9 against 4.6). Across all 55 method-cells the median exponent is 1.05-1.16, in line with the order-2 removal that
  the Part 2 model predicts (rho_2 falls with N/d) and with Owen's o(1/N), not a new power.
- So the honest summary of Part 3 is: constants, no robust rate improvement established. At the measured budgets, with
  pilots amortised, structured sampling in the input regime buys 2-7x (best: multilevel over depth with RQMC inside the
  levels); the gain grows with width only for the dyadic multilevel sampler at L proportional to n, and slowly (1.6 ->
  2.5 from n = 64 to 256 at L = 2n, a pilot-based risk prediction).

## 8. Pre-registration scorecard
Hash times in `work/input-bits/prereg_hashes.txt`; scorers `scratch/score_part{1,2,3}.py` (results in `results/`). Rules that
were implemented or fixed after a hash are marked (section 9).
| part | hit | missed |
|---|---|---|
| 1 (18:16:53Z) | P1.1 (ARC split), P1.3 (trends), P1.4 (amplitude variance), P1.8 (algebraic bit decay), P1.9 (entropy factor), P1.10 (38 bits), P1.11 (D at ARC), P1.12 (D grows with width), P1.14 (univariate excess = radius), P1.15 (antithetic x), P1.17 (finite angle = kernel), P1.18 (PRs) (both partially evaluated: the registered runner records only power-of-two layers) | P1.2 (1 of 16 cells: 256 x 512 at a floor), P1.5 (chain vs 4 nets: 6 of 12 cells, heavy tails), P1.6 (noise < 1%: 57 of 72; the input regime ends near n/2-n), P1.7 (kernel 113 of 116, chain 341 of 472), P1.13 (2 of 48 nets on the GP clause), P1.16 (rotation at n >= 128), P1.19 (Gamma band: registered run 6 of 14 because of the bug; fixed rerun 13 of 14, l = 128 at n = 256 is 3.04 > 3; slopes 1.44, 1.64 pass both ways), P1.20 (radial share 14 of 16) |
| 2 (18:23:45Z) | H2.1 (first-order floor, 37 of 37 cells), H2.2 (calibrated model, 92% of 48 cells, median 1.16), H2.3 (noise shares flat; excess fades), H2.4 (parity), H2.5 (multilevel depth profile; after the scorer fix) | none; H2.6 is a verdict rule: REFUTED in the noise regime and along L = 32 and L = n/2 in the input regime |
| 2A (18:33:33Z) | A2A.1 (placement effect: median ratios 0.48, 0.41), A2A.2 (control) | |
| 2B (19:12:18Z) | A2B.1 (hybrid floor: medians <= 0.011), A2B.3 (consistency with S1) | A2B.2 (the predicted decrease of C(0.75)/C(1) was not detected: 0.598 -> 0.607 from n = 16 to 128) |
| 2D (20:20:50Z) | A2D.1 (block mean order grows 1.53x, n = 16 -> 128), A2D.2 (coordinate mean order grows 1.95x), A2D.3 (non-additive order proxy grows 2.5x) | |
| 2E (20:27:41Z) | A2E.1 (conditional C(t) = unconditional within about 0.01), A2E.2 (older-noise main effect, medians < 0.01), A2E.3 (non-additive order proxy 8.0 -> 20.3 with the conditional additive proxy) | |
| 3 (18:30:47Z) | P3.1 (exponents in band, 55 of 55), P3.2 (gains fall with width at L = 32) | P3.3 (CVQ fails at n >= 128: 6 of 18 nets, 3 of 9 median cells in band), P3.4 (rotation 3 of 9), P3.5 (ML2 at L = 2n not monotone in n: 2.6, 2.9, 2.7); P3.6 verdict: PROGRESS as registered (rate clause, 2 cells), which I attribute to noise (section 7.3) |
Most Part 1 misses are bands set too tight for 4 heavy-tailed nets (P1.5, P1.7 chain clause, P1.13, P1.20) or the input
regime's noise boundary (P1.6); the substantive ones are P1.6 (the input regime ends far earlier than T_mem), P1.16 (the
rotation is likely pilot-limited at large n) and A2B.2 (the decrease in C(0.75) that I predicted was not detected).

## 9. Process, deviations, bugs (all reported both ways where they mattered)
- Co-prover gpt-6-astra (`coprover/PROVENANCE.md`): round 1 (18:01-18:27Z, xhigh, read-only, independent of my directory)
  while my theory v1 was frozen at 18:07:44Z (disclosure: the hash command printed two context-free lines of astra's
  running log at 18:07:33Z; v1's content was complete). Both found the three-term split, the amplitude coefficient C_L,
  the crossover near 25 layers, the 3n/l bit decay and the 2.31 entropy factor. New from astra: the proofs (monotone
  E V_amp -> C_inf, the replica formulas, mean Hermite degree n/V), the finite-width linear coefficient (S1 -> 0.32), the
  sqrt(n) mean-dimension law, the noise-regime predictions (flat linear share 2/3; antithetic 2.42), the exact-mean
  first-layer control. Round 2 (18:50-19:10Z) audited scorers and estimators (below) and proposed Amendment 2B.
- Registrations: Part 1 18:16:53Z, Part 2 18:23:45Z, Part 3 18:30:47Z, Amendment 2A 18:33:33Z, Amendments 2B and 2C
  19:12:18Z, Amendment 2D 20:20:50Z (added at the coordinator's request to reconcile with the numerics lane, after the
  first review), Amendment 2E 20:27:41Z (matched conditioning, after the second review), each before the runs it covers (`prereg_hashes.txt`; copies as hashed in `coprover/PREREG_part*_as_hashed.md`).
- Bugs and fixes:
  1. Part 1c (registered `part1c.py`): the multi-reference tangent vectors were normalised along the wrong array axis, so
     their mean squared norm was n/64 instead of 1 (4 at n = 256, 1 at n = 64). Gamma at n = 256 was 4x too large. Fixed
     copy `part1c.py` (registered copy kept as `part1c_as_registered.py`) and rerun on the same nets (`runs/p1cfix_*`); the
     tables use the fixed values; P1.19 is reported both ways (section 10).
  2. `part2_mlmc.py` never generated the n = 256, L = 32 cell that H2.5(b) needs; `part3.py`'s ML2-RQMC rows lacked the
     charged cost. Both fixed before any run of those scripts (Amendment 2C; registered copies kept).
  3. Scorers: Part 3's registered scorer computed tables but not P3.1-P3.6, and it extrapolated pilot-charged gains below
     the pilot's own cost; the rules were implemented after the hash (labelled), and such gains are UNAFFORDABLE.
     H2.2 is reported with the registered cell-median aggregation and with per-net ratios. P1.17-P1.18 were evaluated on the
     recorded power-of-two layers only.
  4. Estimators: the Hermite two-halves estimators centred with the full-sample mean (halves not independent; relative bias
     O(1/h) ~ 1e-4 at h = 8,192) and `part2_noise.py` divided centred sums by h instead of h - 1 (factor 1 - 2/h); fixed in
     the library after all runs (`lib/ib_as_used_in_runs.py` is the version the runs used); negligible at these sizes.
     At n = 256 the degree-2 shares use 64 of the 256 outputs, normalised by those outputs' variance.
- Statistical caveats: 4 nets per Part 1-2 cell, 2 per Part 3 cell; per-net variances are heavy-tailed (the amplitude's
  moments grow like (1 + 5/n)^L [A]; one n = 128 net carries 97% of a 4-net amplitude sum), so medians describe typical
  nets and annealed chain means describe weight averages; RQMC variances from R = 10-32 scrambles have relative SE
  about sqrt(2/(R-1)) = 25-47%.
- Compute: CPU only, nice 19, idle IO, at most 3 heavy processes (one brief 4th dev process was stopped at 18:36Z), single
  BLAS thread; no GPU.

## 10. GPT-6 review
Round 1 (fresh session, high effort, read-only; 20:14-20:19Z; `review/review_gpt6.md`): verdict **major revision**,
10 findings. It reproduced the ARC split (0.494/0.470/0.034), the layer-31 Hamming count (19.7), the ARC mean dimension
(10.30), the n = 128 noise first-order share (0.622), the ARC RQMC gain at N = 1024 (1.247), the 64 x 128 multilevel-RQMC
gain (7.12), the registration hashes, scorecards, tables and tests. What changed:
1. "The whole ANOVA-order distribution is the same" overstated what three values of C(t) at t <= 0.75 identify: now "no
   detected shift at the measured t", with the tail sensitivity (moving 10% from order 10 to 100 changes C(0.75) by
   0.0056) and the fitting code `scratch/fit_orders.py` (section 6.4). Together with the coordinator's request this led to
   Amendment 2D and section 6.7.
2. The RQMC surrogate was presented as a mechanism: now a predictive test of a stated surrogate; "the rest is not
   touched", "pair removal impossible" and "no sampler removes the tail" removed or qualified (sections 6.3, 6.6).
3. Finite-budget gains were called asymptotic constants: now "finite-budget gains with pilots excluded"; "no robust rate
   improvement established"; MLD's growth labelled a pilot-based risk prediction (section 7).
4. The multilevel-RQMC envelope selects among three allocations on the same data: labelled an oracle envelope, with
   one pre-set allocation reported alongside (2-7x survives; section 7.1).
5. The sensitivity ratio Gamma/(finite fraction) was called the direction's mean ANOVA dimension (it exceeds n): now a
   sensitivity diagnostic, closer to a mean Hermite degree (section 5).
6. "Every pilot method still loses at its largest budget" was wrong: corrected with the cells where they win (section
   7.2).
7. Pilot matrix work omitted from charged costs: stated, at most 7% (section 7.2).
8. Scorecard coverage: P1.17-P1.18 marked partially evaluated; P3.3 reported per net and per median cell (section 8).
9. Summary qualifications restored (eta 1.06% at n = 128; radial share 14 of 16; n/V is the annealed ratio).
10. The sensitivity column uses a ratio of medians; per-net medians given (section 5).
Round 2 (fresh session, 20:25-20:27Z; `review/review_gpt6_r2.md`): verdict **major revision** again. It judged
findings 3-6 and 8, 10 resolved and 1, 2, 7, 9 partly resolved, and added: (11) section 6.7 compared a window function
with independent older noise (2B) against conditional quantities with shared older noise (2D); (12) "growth confined to
a ~10% tail" is only consistent with the data, not measured; (13) small numeric and label errors. What changed: Amendment
2E (registered before its runs) measured the conditional generating function: it equals 2B's within 0.01, so the
comparison stands with matched conventions, and the non-additive mean order is recomputed with the conditional additive
share (8 -> 20); the tail statement now reads "a possible reconciliation; its mass and effect on RQMC gains are not
established" in the summary and section 6.7; the remaining wording fixes (the surrogate's assumption about high orders,
the hybrid floor's per-net maximum 3.95%, S1's sign of change, 7.3, net-depths vs cells, partial coverage in section 5,
unaffordable extrapolations marked as history in `results/part3_scores.json`) are applied.
Round 3 (fresh session, 20:31-20:34Z; `review/review_gpt6_r3.md`): verdict **minor revision**. It recomputed every 2E
number and found the conditioning fixed; it asked for narrower labels: "no detected change at the measured C(t)" instead
of a width-independent distribution, the non-additive orders as proxies with conservative bounds (6.3-8.0, 11.5-15.0,
15.5-20.3), "within about 0.01", the calibration scope of the pair-removal sentence, the joint-limit scope of C_inf, the
evaluated-layer scope of the 10% kernel agreement, the rotation's 128 x 64 win, and the Part 3 scope of "only the dyadic
sampler grows". All applied. Summary of the review history: major, major, minor.

## 11. Files
All under `work/input-bits/`: `FORMAL.md` (theory v2), `PREREG.md` + `prereg_hashes.txt`, `notes/questions.md`,
`lib/ib.py` (+ `lib/ib_as_used_in_runs.py`), `tests/test_ib.py` (`logs/test_ib.log`, `logs/test_ib_after_fix.log`: all pass),
`scratch/` (runs: part1a/b/c, part1chain, part2_noise/input/mlmc, amend2a_offset, amend2b_stability, amend2d_meandim,
amend2e_condstab, calib_rho2, part3; scorers score_part1/2/3, score_a2d, score_a2e; fit_orders; tables_part1/3; figs;
registered copies `*_as_registered.py`), `runs/` (JSON), `results/` (part1_scores.json, part1_tables.md,
part2_scores.json, part3_scores.json, part3_tables.md, a2d_scores.json, a2e_scores.json, order_fits.json,
rho_calibration.json),
`figs/`, `coprover/` (prompts, raw logs, answers, PROVENANCE.md), `review/`. Python: /home/agent/.venvs/arc-noisy/bin/python.

