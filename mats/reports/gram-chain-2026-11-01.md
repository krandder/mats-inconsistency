# The replica Gram chain as a theory of the bits that matter at a layer (1-2 November 2026)

Status: complete (2 November, 03:00Z). All registered runs finished except two dropped widths (PA7 at n = 512, 1024; PA8 third moment at n = 256). Revised after two GPT-6 reviews (section 10).

Lane: gram-chain (target M1), bead pool-qxszj1.23 under Kelvin's stationary ask pool-qxszj1. Work directory
`work/gram-chain/`. First deliverable, the explainer: `outbox/gram-chain-explainer-2026-11-01.md` (checked by
gpt-6-astra, revised). Model S throughout: h_l = relu(W_l h_{l-1}) + sigma xi_l, W_l iid N(0, 2/n), output
a = relu(W_L h_{L-1}). Labels: PROVEN, NUMERICAL (exact chain, sampled, bootstrap errors over chains), MEASURED (real
nets), HEURISTIC (mean field or leading order, tested), CONJECTURE. Provenance: [C] Claude, [A] gpt-6-astra, [C+A] both
independently. "Annealed" = averaged over the weights; "per net" = fixed weights. t = layers to go counting the output
layer, so layer l = L - t. Q = |h|^2/n, q = Q/n, delta = 1 - cos(angle between two replicas), A = 2.2309,
c_k = 2 sqrt2/(3 pi), theta* = arccos(1 - delta*), tau = pi/theta*.

## Summary for Kelvin

**Short answer.** The chain works as a theory tool, and with it the bits that matter at a layer have a quantitative,
weight-averaged theory: the exact chain confirms its closed-relevance curves to within 0.003-0.014 (absolute) at
n = 256-1024 and its other laws to within 10-30%, and real nets follow it once each net's radius is used. At layer L - t of a stationary net, the state
decides a fraction 1 - Y((t - 1)/T*) of the output direction's variance (Y' = 1 - Y^{3/2}; T* = 2.23 Q^{1/3}, about
2 n^{1/3} layers), and that part lives in about n/(n/PR_C + c t) directions (c ~ 1.1-1.6; PR_C = n/tau, tau = T*/1.5;
order n^{2/3} directions near the output); the radius carries a further, net-dependent share rho(q) of the output variance
for order n layers. In the input regime the input's
directional information at layer l lives in about 1.5 n/l directions (a 1/n four-point effect that the kernel cannot see),
and the radius carries the kernel share (1/2 + C_l)/(1/2 + C_L) of the log-amplitude. Your "Lyapunov covariance times
what the future amplifies, direction by direction" becomes a free product: the cloud's principal axes and the future's
amplified directions sit in random relative position (PROVEN for the frozen Jacobian given the past, HEURISTIC for
the full nonlinear relevance operator), so dimensions combine harmonically,
1/PR = 1/PR_cloud + 1/PR_future - 1/n (plus a gate-switching excess beyond a few layers). What the chain cannot give is
which directions: it is invariant under rotations of the layer that fix the replicas, so it carries no orientation
information; describing a k-dimensional relevant subspace to precision eps takes about k(n - k) log2(1/eps) bits, which
annealed theory does not supply (this is a description length, not a proven cost for estimating the mean). Per net, the
scale of all of this is set by the net's radius, which does not self-average; the radius law predicts a net's cloud size
with R^2 = 0.88-0.93 at n = 64-256.

1. **The tool** (PROVEN; tested). The replica Gram chain integrates the weights out exactly at any width, for any noise
   sharing, tangent vectors, perturbations and inputs (`lib/gc2.py`, 19 checks against explicit nets). It computes the
   weight average of any noise expectation of a replica Gram path: closed and total relevance, Sobol and noise-stability
   curves, participation ratios, and the across-net scatter of each. In the stationary regime raw-state annealed moments
   are infinite (heavy radius tail), so everything here uses directions and log radii.
2. **The stationary law of two to four replicas** (HEURISTIC limit [C+A]; NUMERICAL, n = 8-1024). The angle is slaved to
   the noisy-kernel fixed point delta*(Q) = 2.231 Q^{-2/3}(1 - 1.56 Q^{-2/3} + ...) with relative fluctuations of about
   1.8 n^{-1/3} (the rescaled law collapses); its median ratio to delta*(Q) rises to 0.989 and its mean is 1.000 at
   n = 1024. The radius ratio of two replicas is a slow diffusion jointly with the radius, with a closed form for
   E[ell^2 | q] [A]: the chain's SD(ell) n^{1/3} rises to 1.875 (0.025) at n = 1024 against the limit 1.887 and matches the
   closed form within 1-8% in every radius bin. The radius mixes at rate 5/(8n) per layer (spectral gap of the limit
   diffusion, PROVEN there); autocorrelations fall to 1/e in about 0.8n (radius), 1.0-1.4 Q (radius ratio) and
   1.2-1.4 x 1.49 Q^{1/3} layers (angle).
3. **How many directions the stationary cloud spans** (HEURISTIC four-point balance, [A] first, re-derived [C];
   NUMERICAL; MEASURED). PR_C = n theta*/pi = 0.67 n Q^{-1/3}, of order n^{2/3}, not the (log n)^2 of the numerics lane's
   assumed 1/k spectrum. In the chain, binned by radius, PR follows n theta*(Q)/pi over the whole radius range (a 3.5x
   fall from q = 0.2 to 10 at n = 256; measured/predicted 1.14-1.29 at n = 256 and 1.04-1.16 at n = 1024); on real nets
   the per-net PR falls
   with the net's radius with log-log slope -0.22 to -0.32 at n >= 32 (predicted -1/3; Spearman -0.73 to -0.87). A
   radius-independent 1/k shape cannot do this. The width test was registered against both laws (Amendment A1): between n = 256 and 1024 the slope of PR_w is 0.614 (90% interval 0.599-0.629) against 0.34 for 1/k, and PR_w = 114.7 at n = 1024 against 110.4 from 1.0865 n^{2/3} and 65.8 from 1/k (the ratio to the n^{2/3} law falls 1.47, 1.36, 1.26, 1.17, 1.12, 1.09, 1.04 from n = 16 to 1024).
   With the relevance horizon below: PR_C x T* = 3n/2 at
   every radius to leading order (1.28n, 1.40n, 1.46n at the median radius for n = 64, 256, 1024).
4. **How much the state at layer L - t matters, and for how long, stationary** (HEURISTIC [C+A]; NUMERICAL; PB2 hit, PB1
   hit only under the revised analysis).
   The closed relevance of the state at layer L - t for the output direction is 0.877, 0.666, 0.377, 0.126 at
   t = 2, 4, 8, 16 for n = 256 against the mean field's 0.877, 0.665, 0.374, 0.123, and 0.928, 0.795, 0.566, 0.294 at n = 1024 against 0.928, 0.792, 0.567, 0.283; the
   curves collapse in (t - 1)/T*. The total relevance decays algebraically with a slower, radius-driven tail [A]. The
   log-amplitude's closed relevance collapses in t/n and halves after about n/4 layers. The radial share of a net's
   output variance has the closed form rho(q) = m(q)/(m(q) + 2A q^{-2/3}) [C, with A's m] (0.35 at the median radius,
   n-independent; HEURISTIC, a small-fluctuation variance proxy, not a Sobol share): the chain's ratio of means is 0.375,
   0.349, 0.354 at n = 64, 256, 1024, and on real nets the per-net radial share follows rho of the net's own radius
   (median deviation 12%, 11%, 5% at n = 64, 128, 256; PC7 hit).
5. **How many directions are relevant at layer L - t, stationary** (HEURISTIC free product [C+A; relevance lane];
   NUMERICAL; PB3 missed on 1 of 12 clauses). The linear free-product value n/(n/PR_C + 2t) is 9-16% below the chain's
   exact participation ratio at t = 1 (and t = 2 at n = 256) and 22-58% below it at t = 2-4; beyond a couple of layers,
   gates switching across the cloud spread relevance over more directions, and empirically PR_rel ~ n/(n/PR_C + c t) with
   c ~ 1.1-1.6 (closed version) instead of the frozen-gate 2. At n = 256 the closed relevance at t = 1, 2, 4, 8, 16 has
   variance-weighted participation ratios 39, 30, 24, 16, 9 (annealed moment ratios; the median net's is about 0.7 of
   these, section 5). The future's second spectral moment matches free probability (tangent probes within 3-5% of
   n/(2t) at t = 2-16).
6. **Input regime** (HEURISTIC vertex recursion [C+A]; NUMERICAL; PB6, PB7, PB8 hit). The input cloud's angular PR at
   layer l follows a 1/n vertex recursion that tends to 1.5 n/l: chain 166, 62, 18.8, 12.1, 6.1 at l = 1, 4, 16, 32, 64
   (n = 256) against the recursion's 166, 61, 18.7, 10.2, 5.4 and the input-bits nets' 159, 57, 17, 9.6, 6.1. The radius
   share of the log-amplitude matches the kernel within 4% at every layer of 256 x 32. At 8 x 256 the input is one direction plus the
   radius after about 8 layers (PR 1.1-1.4 at l = 16-128, among chains whose four inputs all survive) and 63% of inputs
   die in all-off layers. Small noise: at 256 x 32, sigma = 1e-3 leaves 99.94-100% of the variance decided by the layer-l state at every l; sigma = 1e-2 leaves 95-100% of the raw output and
   85-99% of its direction; sigma = 0.1 hands most of it to the noise (20-59% of the raw output, 2-38% of the direction
   decided at l <= 24).
7. **Annealed versus quenched** (MEASURED on 168 nets, n = 8-256; PC1, PC2, PC3, PC5, PC6, PC7 hit, PC4 missed). The
   per-net cloud size is not self-averaging: its across-net CV is 0.39-0.75 against the chain's 0.42-0.62 (rising toward
   0.68); divided by delta*(Qbar) the CV falls to 0.14-0.20 (squared CV cut by 87-97%; the radius law alone predicts a
   net's cloud size with R^2 = 0.88, 0.92, 0.93 at n = 64, 128, 256). Per-net closed relevance follows the mean field at the net's own radius within 0.02-0.08. Where the annealed count misleads:
   the chain's variance-weighted PR overstates the median net's dimension by 1.4-1.5x at n = 128-256 (small-radius nets
   have bigger clouds and dominate the weights; 1.35x in the limit), and the annealed relevance curve sits below the
   median net's (0.67 against 0.74 at t = 4, n = 256).
8. **What the chain explains of earlier measurements** (section 6): the cloud's size per net (A q^{-2/3} collapses 168
   nets), births, the interaction order's mechanism and growth (chain's finite-eps block order 1.90, 2.60, 3.34 at
   n = 16, 64, 256, slope 0.20, against the nets' 0.17-0.26), the flat low-order shares (chain C(t) = 0.156-0.186,
   0.359-0.389, 0.591-0.632 at t = 0.25, 0.5, 0.75 for n = 16-1024, per-coordinate sharing, against the closure's 0.175, 0.370, 0.603 and the
   input-bits nets' 0.16-0.18, 0.36-0.38, 0.60-0.64), the antithetic even share (0.198-0.209 against 0.23-0.27 on nets),
   the 3n/l gate decay, the "about 10 directions" of the input at 256 x 32, and the amplitude/direction split gap (an
   averaging effect). Not explained: the top-k shape of the cloud spectrum and per-net residuals beyond the radius.
9. **Scorecard and review.** Of the 24 registered predictions, 16 hit (counting PA9, the verdict for the n^{2/3} law), 7 missed and 1 is incomplete (PA7: in band at every width run, two widths dropped), with the revised PB1 analysis (15 hits, 8 misses with PB1 as first analysed); section 9 lists each. Two GPT-6 review rounds (major revision, then residuals), all findings applied (section 10).

## 1. The tool

The replica Gram chain (explainer section 2; numerics lane P-G1) is exact at every width: given the past, the rows of
W_l see the replicas only through their Gram matrix, so G_l = (1/n) sum_i v_i v_i^T with v_i = relu(z_i) + xi_i,
z_i ~ N(0, 2 G_{l-1}). This lane extended it (FORMAL 1.2, `lib/gc2.py`) to correlated noise, per-coordinate and per-layer
Bernoulli noise sharing, tangent vectors, Gram surgery (perturbations, radius rescaling, direction swaps), inputs at
sigma = 0 and the noiseless output layer, all exact by the same argument and tested against explicit networks (19 KS and
mean checks pass, `logs/test_gc2.log`). Three facts frame what follows.
- Response invariance (PROVEN, FORMAL 1.3; the relevance lane's Theorem 1 is the sharper form): given the past, the
  future's response to a perturbation of layer l depends on it only through its Gram relations with the replicas, so
  annealed theory can count and weigh relevant directions but cannot name them.
- Annealed tangent conservation (PROVEN, FORMAL 1.4): E_W |J v|^2 = |v|^2 at every depth while the typical value decays,
  so annealed linear-response means are carried by rare nets and are not used as relevance measures.
- Integrability (FORMAL 1.5; NUMERICAL check): E Q = infinity and the radius ratio between replicas does not shrink at
  large Q, so the annealed second moments of raw-state differences are infinite (n = 64: the top 1% of chains by radius
  carry 82% of the sample mean of |h1 - h2|^2, 94% of it radial). The annealed quantities here use the direction
  u = h/|h| or a/|a| and log radii. Both need a convention on the all-off event (a = 0, probability 2^-n per layer for a
  noiseless output layer): the code floors |a|^2 at 1e-300 (Part B) or |a| at 1e-300 (Part C), so log|a| moments are
  regularised estimands; likewise Part B's direction distance 2(1 - cos) is not meaningful when an output is zero, while
  Part C sets u(0) = 0. At n >= 64 the event never occurred in these runs; at n = 16 it did and the log channel is not
  used. This is also why the numerics lane's annealed D_h was "too noisy".

## 2. The stationary law of the normalised Gram (Part A)

2.1 Structure (HEURISTIC, leading order in n^{-1/3}; [C+A] unless marked; FORMAL 2 and 7). Two replicas with independent
noise have three coordinates on three time scales.
- The radius. Each replica's marginal is the exact one-input law. On the scale s = t/n it is the diffusion
  dq = ds + sqrt5 q dW with stationary density (2/5) q^{-2} e^{-2/(5q)}; in u = log q its generator is a Morse problem with
  a single bound state, so the spectral gap is 5/8, the bottom of a continuous spectrum (PROVEN for the limit diffusion).
- The angle. Fast: it relaxes at rate theta*/pi per layer (tau ~ 1.49 Q^{1/3} layers) to the fixed point of the
  mean-field closure rho' = Q kappa(rho)/(Q + 1), delta*(Q) = eps(1 - 0.7 eps + O(eps^2)), eps = A Q^{-2/3} (two-term form
  within 0.3% at Q >= 148) [C]. One exact step has n Var(delta')/delta^2 = 6.0, 6.8, 7.3 at n = 64, 256, 1024, tending to
  8 [C measured; A limit], so delta fluctuates by order n^{-1/3} relative.
- The radius ratio y = n^{1/3}(log|h1| - log|h2|). Slow: pulled back at rate 1/(Q + 1) per layer and driven at 4 delta/n
  per layer (one exact step: 4.05-4.09 delta/n). With q it forms the limit diffusion dy = -(y/q) ds + 2 sqrt(A) q^{-1/3} dW',
  and E[y^2 | q] has the closed form m(q) = (24A/25) beta^{-2/3} e^{beta/q} Gamma(2/3, beta/q), beta = 2/5 [A]; E y^2 = 3.561,
  so SD(log|h1| - log|h2|) -> 1.887 n^{-1/3} [C+A].

2.2 The exact chain (NUMERICAL; PREREG Part A; P = 8000 chains at n <= 64 down to 500 at n = 1024, 8n burn-in from the
exact radius law, 64 samples per chain over 8n layers; bootstrap SE over chains in parentheses;
`results/partA_scores.json`, `figs/fig1_stationary_law.png`).

| n | median delta/delta*(Q) | mean | SD(delta/delta*) n^{1/3} | SD(ell) n^{1/3} (limit 1.887) | 1/e lag: angle / ratio / log q (layers) |
|---|---|---|---|---|---|
| 8 | 0.980 (0.002) | 1.0841 (0.0019) | 1.37 | 1.421 (0.005) | 2 / 5 / 5 |
| 16 | 0.966 (0.002) | 1.0507 (0.0015) | 1.47 | 1.542 (0.005) | 3 / 10 / 10 |
| 32 | 0.956 (0.001) | 1.0246 (0.0011) | 1.56 | 1.635 (0.005) | 4 / 22 / 41 |
| 64 | 0.962 (0.001) | 1.0125 (0.0010) | 1.65 | 1.720 (0.005) | 6 / 39 / 51 |
| 128 | 0.969 (0.001) | 1.0051 (0.0009) | 1.73 | 1.798 (0.008) | 8 / 94 / 158 |
| 256 | 0.974 (0.001) | 1.0011 (0.0010) | 1.78 | 1.819 (0.010) | 10 / 204 / 229 |
| 512 | 0.984 (0.001) | 1.0012 (0.0010) | 1.79 | 1.843 (0.016) | 13 / 325 / 410 |
| 1024 | 0.989 (0.001) | 1.0000 (0.0010) | 1.83 | 1.875 (0.025) | 17 / 663 / 797 |

- Scaling: the law of (delta/delta*(Q) - 1) n^{1/3} collapses across n (fig 1, left); SD(delta/delta*) n^{1/3} rises
  slowly, 1.65 to 1.83 between n = 64 and 1024 (PA4 hit), short of the 2.2 that the one-step limit 8 would give.
- The radius ratio reaches its limit: SD(ell) n^{1/3} = 1.875 (0.025) at n = 1024 against 1.887 (PA1 hit), and
  E[ell^2 | q] n^{2/3} = 2.66, 3.24, 3.79, 4.16, 4.11, 4.66, 4.91 in the radius bins [0, .3), ..., [10, inf) at n = 1024
  against the closed form's 2.74, 3.30, 3.73, 4.14, 4.49, 4.75, 5.13 (PA2 hit; fig 1, right).
- The angle's level: the median of delta/delta*(Q) approaches 1 from below and the mean is 1.000 at n = 1024. PA3 missed:
  the median is below 0.97 at n = 64 and 128, and n^{2/3}(mean - 1) is not constant (0.20, 0.13, 0.05, 0.08, 0.00 at
  n = 64-1024). Given the radius there is a tilt: E[delta/delta*(Q) | q] is 1.030 for q < 0.3 and 0.920 for q > 10 at
  n = 1024 (1.063 and 0.835 at n = 256), shrinking roughly like n^{-1/2}. Correlated angle-radius increments
  (a_{Q delta} ~ -4 Q delta/n [A]) push the angle below its fixed point when the radius is high; gpt-6-astra's
  second-order coefficient for this conditioning (+0.077 n^{-2/3} relative at the median) is far smaller, so the tilt is a
  finite-n effect beyond that order (not derived).
- Mixing: autocorrelations of 200 contiguous traces (4n layers) fall to 1/e in 1.2-1.4 x the mean-field 1.49 Q^{1/3}
  layers (angle), 1.0-1.4 x (Q + 1) layers (radius ratio) and about 0.8n layers (log radius). PA6 missed on its
  log-radius clause by under 3% (0.78n at n = 1024 against the band's 0.8n): the gap's 1.6n is the slowest rate of a
  continuous spectrum, not the 1/e time.

2.3 Noise sharing (A-share; NUMERICAL). The mean field depends only on E S_12 = t, so Gaussian correlation t (Hermite
degrees), per-coordinate sharing (ANOVA over scalar coordinates) and per-layer sharing (ANOVA over layer blocks) share
C(t) = 1 - E delta_t / E delta_0 -> 1 - (1 - t)^{2/3} = 0.175, 0.370, 0.603 at t = 0.25, 0.5, 0.75 [A, input-bits round 1].
Values below use the registered baseline (E delta_0 from the A-r2 run at the same n). Deviations from the registration:
the sharing runs used a 2n burn-in and 4n recorded layers (registered: 6n and 8n), n = 1024 omits t = 0.9 and 0.99, and
the first scorer used each sharing run's own independent-noise baseline (at n = 256 that gave coordinate C(0.25) = 0.164
and an antithetic ratio 1.605 instead of 0.173 and 1.587; every PA5 clause passes either way).

| n | correlation | per coordinate | per layer | finite-eps mean order G(0.01)/0.01: corr / coord / layer | antithetic E delta(-1)/E delta(0) (2^{2/3} = 1.587) |
|---|---|---|---|---|---|
| 16 | 0.182, 0.386, 0.630 | 0.186, 0.389, 0.632 | 0.193, 0.406, 0.654 | 2.79 / 2.54 / 1.90 | 1.603 |
| 64 | 0.179, 0.383, 0.616 | 0.183, 0.383, 0.617 | 0.187, 0.389, 0.629 | 3.45 / 3.44 / 2.60 | 1.583 |
| 256 | 0.186, 0.368, 0.606 | 0.173, 0.374, 0.610 | 0.177, 0.380, 0.613 | 4.06 / 4.06 / 3.34 | 1.587 |
| 1024 | 0.158, 0.345, 0.588 | 0.156, 0.359, 0.591 | 0.172, 0.371, 0.606 | t = 0.99 not run; G(0.03)/0.03 = 3.15 / 3.12 / 3.00 | 1.603 |

The three kinds of sharing agree within 0.03 at t <= 0.75 and match the closure within 0.03 at n >= 64 (within 0.015 for
correlation and per-coordinate sharing at n = 64-256; at n = 1024 they sit 0.01-0.025 below it); the
input-bits lane measured 0.16-0.18, 0.36-0.38, 0.60-0.64 for the output under per-coordinate sharing on nets at n = 8-128.
They separate only at small refresh rates (t >= 0.9), where single-layer injections stop overlapping: the per-layer
(block) mean order is the smallest and grows like n^{0.21} (1.90 to 3.34 from n = 16 to 256), the slope of the numerics
lane's nets (0.17-0.26). Antithetic noise multiplies the stationary angle by 2^{2/3} to within 0.0-1.0% (1.583-1.603;
the even share of the direction is then 0.198-0.209, gain 2.4-2.5; nets 0.23-0.27). PA5 (registered baseline) hits: every clause holds at n = 64, 256 and 1024; at n = 1024 per-coordinate sharing gives 0.156, 0.359, 0.591 (closure 0.175, 0.370, 0.603) and antithetic noise 1.603 x the independent angle.

2.4 Three and four replicas: how many directions the stationary cloud spans.
- Four-point law (HEURISTIC [A], re-derived [C], FORMAL 7.5). For four independent replicas the double difference
  D = (u1 - u2).(u3 - u4) relaxes at the angle's own rate theta*/pi and is driven by 8 delta^2/n per layer (nearly aligned
  replicas share their gates; z1 - z2 and z3 - z4 are independent). Hence, per net, PR_C = (2 delta*)^2 / Var D =
  n theta*/pi = 0.672 n Q^{-1/3}, of order n^{2/3}; with the chain's variance weighting
  PR_w = E_W (tr C)^2 / E_W tr C^2 = 1.0865 n^{2/3}. A 1/k spectrum would give (log n)^2 with no radius dependence.

| n | PR_w (exact chain, 4 replicas) | 1/k spectrum | 1.0865 n^{2/3} | across-net CV of tr C (chain) |
|---|---|---|---|---|
| 16 | 10.04 (0.03) | 9.7 | 6.9 | 0.425 |
| 32 | 14.77 (0.04) | 15.2 | 11.0 | 0.491 |
| 64 | 21.73 (0.07) | 22.3 | 17.4 | 0.555 |
| 128 | 32.12 (0.14) | 30.8 | 27.6 | 0.594 |
| 256 | 48.96 (0.30) | 41.0 | 43.8 | 0.619 |
| 512 | 75.79 (0.68) | 52.6 | 69.5 | 0.650 |
| 1024 | 114.74 (1.38) | 65.8 | 110.4 | 0.665 |

- Binned by radius (post hoc, `results/pr_by_radius.json`, fig 3 right), the chain's PR follows n theta*(Q)/pi across the
  whole radius range at each n: at n = 256 it falls from 56.8 to 16.1 between q = 0.2 and q = 10 against 44.1 to 12.5
  predicted, measured/predicted is flat in q (1.14-1.29) and falls with n (median over bins about 1.9, 1.6, 1.4, 1.27,
  1.18, 1.13, 1.08 at n = 16 ... 1024; at n = 1024 every bin is within 4-16%). On real nets the per-net PR falls with the radius too (section 5). A radius-independent 1/k shape
  cannot do this. Amendment A1 (hashed at 21:33:29Z, before the n = 256 result existed but about six minutes after that run had started; n = 1024 prospective): the slope of PR_w between n = 256 and 1024 is 0.614 (bootstrap 90% interval 0.599-0.629), against 0.34 for a 1/k spectrum, so the n^{2/3} law is favoured; at n = 1024 PR_w = 114.7 (1.3) against 110.4 from 1.0865 n^{2/3} and 65.8 from 1/k, and the ratio to the n^{2/3} law falls to 1.04. This favours the four-point law over this range; it does not prove the exponent or determine the spectral shape. PA8 (1/k moments) missed: the radius-normalised PR is already 27% below the 1/k value at
  n = 64. The third moment (six replicas, n = 16-128; n = 256 dropped): the variance-weighted ratio tr C^3 tr C/(tr C^2)^2 is 1.49, 1.69, 1.87, 1.95, rising more slowly than a 1/k spectrum's 1.40, 1.64, 1.91, 2.21; the radius-normalised version registered in PA8 is too noisy to use (1.72, 1.96, 3.50, 2.31). The spectrum's shape beyond its participation ratio remains open.
- Three replicas: the mean (max - min)/mean of the three pairwise angles is 0.751, 0.656, 0.568, 0.482, 0.404 at n = 16, 32, 64, 128, 256, 6-16% above the Gaussian-cloud estimate 1.46 sqrt(2/PR) = 0.708, 0.598, 0.507, 0.424, 0.347 (PA7: in band at every width run; n = 512, 1024 were dropped, so the registered prediction is incomplete). With PR ~ n^{2/3} the triangle becomes equilateral only like n^{-1/3}.
- One time scale (HEURISTIC, FORMAL 7.10 [C]): T* = (Q + 1) delta* = 1.5 tau and PR_C = n/tau at leading order, so
  PR_C x T* = 3n/2 at every radius to leading order (with the finite-Q definitions 1.28n, 1.40n, 1.46n at the median radius
  for n = 64, 256, 1024): a large-radius net has a smaller, longer-lived cloud and a small-radius net a larger,
  shorter-lived one.

## 3. The bits that matter at layer l: stationary regime (Part B)

Estimands (annealed, scale-free, exact replica constructions; `scratch/partB.py`; `results/partB_scores.json`).
- Closed relevance V_P(t) = E_W Var(E[u | noise up to l]) / E_W Var(u), u = a/|a|: two replicas identical at layer l with
  independent noise afterwards. It is what any description of h_l can explain about the conditional mean of the output
  direction u (for E[a] itself see 3.2: the radius carries the rest) (the relevance lane's memory
  share M_l, here weight-averaged). The same for log|a|.
- Total relevance T_P(t) = E|u - u'|^2 / (2 E_W Var u) with independent pasts and a shared future: everything before l,
  including its interaction with the future noise.
- Relevant dimension: participation ratios of the output's dependence on the past, total (four pasts under one future)
  and closed (four pasts, four independent futures; 10 replicas).

3.1 How much and for how long: the direction (HEURISTIC mean field [C+A]; NUMERICAL; PB2 hit, PB1 hit only under the
revised analysis). Two replicas that
start together and get independent noise open an angle that grows to delta* on the cloud's own time scale, so
  V_P(t) = 1 - Y((t - 1)/T*),   Y' = 1 - Y^{3/2},  Y(0) = 0,   T* = (Q + 1) delta*(Q) ~ 2.23 Q^{1/3} = 1.5 tau,
with area 2(1 - pi/(3 sqrt3)) T* = 0.79 T* and an exponential tail at rate 1.5/T*. With the future shared, the cloud angle
decays algebraically, T_P ~ (1 + (t - 1)/(2T*))^{-2}, then exponentially at rate 1/Q.

| n | closed V_P at t = 2, 4, 8, 16, 32: chain | mean field (averaged over radii) | V_P half-life / area: chain vs MF | total T_P at t = 2, 4, 8, 16, 32: chain | MF | T_P half-life, T_P(4T*): chain vs MF |
|---|---|---|---|---|---|---|
| 16 | 0.574, 0.219, 0.054, 0.012, 0.006 | 0.632, 0.268, 0.064, 0.008, 0.001 | 2.3 / 2.91 vs 3 / 2.98 | 0.725, 0.446, 0.231, 0.087, 0.024 | 0.700, 0.396, 0.172, 0.056, 0.014 | 3.5, 0.096 vs 4, 0.063 |
| 64 | 0.780, 0.469, 0.166, 0.031, 0.011 | 0.788, 0.482, 0.187, 0.037, 0.003 | 3.8 / 4.76 vs 4 / 4.85 | 0.839, 0.616, 0.393, 0.208, 0.093 | 0.823, 0.588, 0.344, 0.155, 0.053 | 5.7, 0.115 vs 6, 0.071 |
| 256 | 0.877, 0.666, 0.377, 0.126, 0.016 | 0.877, 0.665, 0.374, 0.123, 0.019 | 6.1 / 8.04 vs 6 / 7.87 | 0.905, 0.745, 0.534, 0.324, 0.168 | 0.893, 0.726, 0.511, 0.294, 0.133 | 9.0, 0.115 vs 9, 0.082 |
| 1024 | 0.928, 0.795, 0.566, 0.294, 0.089 | 0.928, 0.792, 0.567, 0.283, 0.075 | 9.6 / 14.3 vs 10 / 12.7 | 0.937, 0.817, 0.655, 0.446, 0.256 | 0.934, 0.821, 0.651, 0.442, 0.245 | 13.6, 0.107 vs 14, 0.089 |

The closed relevance agrees with the mean field averaged over the radius law to within 0.003-0.014 at n = 256-1024 and
the curves collapse in (t - 1)/T* (fig 2); the total relevance's tail is 1.2-1.6 times the mean field at t = 4T*, the
slower tail gpt-6-astra attributes to unequal radii under shared noise. V_P copies the whole state, so it is the closed
relevance of the full layer-l state for the output direction (its angular part dominates; the radial part is 3.2).
PB1 scoring, both ways: the registered analysis missed (the first recorded t below 0.5 is 8 at n = 256, against 6,
because t = 7 is not recorded; the full-grid area integrates noise out to t = 4n and gives -1.28 at n = 256 and 66.7 at
n = 1024); the revised analysis (linear interpolation between recorded t, area truncated at 8 T*, where the mean field is
below 1e-3) hits; the revision was made after seeing the n = 256 curve (`logs/deviations.log`). Scaling: the layer-l state
matters for the output direction's mean for about 0.8 T* ~ 1.6 n^{1/3} layers in total, and its relevance falls below 10%
after about 2-3 n^{1/3} layers.

3.2 The amplitude remembers far longer (NUMERICAL). The closed relevance of log|a| is 0.96, 0.87, 0.71, 0.51, 0.30, 0.13
at t = 8, 16, 32, 64, 128, 256 for n = 256 and 0.99, 0.97, 0.93, 0.83, 0.71, 0.56, 0.31, 0.14 at t = 8, 16, 32, 64, 128,
256, 512, 1024 for n = 1024: it collapses in t/n and halves after about n/4 (fig 2, right; at n = 16, dead outputs, 2^-16 per
sample, make the log channel unusable). PB4 missed as registered: the strict clause "log above direction at every
t >= 8" fails where both relevances are near zero (n = 64 at t = 160-192, n = 256 at t = 1024; log-channel standard errors
were not saved); the
substantive part (0.34, 0.30, 0.31 at t = n/2 for n = 64, 256, 1024, against about 0.01 for the direction) holds. The radial share of a net's
(scale-normalised) output variance, measured by the small-fluctuation proxy Var(log|a|)/(Var(log|a|) + tr Cov(u)) (a
variance-component ratio, not a Sobol share), is rho(q) = m(q)/(m(q) + 2A q^{-2/3}) (FORMAL 7.11 [C], with A's m;
HEURISTIC): 0.10, 0.24, 0.35,
0.46, 0.60, 0.75, 0.84 at q = 0.1, 0.3, 0.577, 1, 2, 5, 10, independent of n at leading order; the chain's ratio of means is
0.375, 0.349, 0.354 at n = 64, 256, 1024. For a net's raw output the closed relevance at layer l is then
M_l ~ (1 - rho) V_P^dir(t) + rho V_P^log(t): small-radius nets forget in about 2 n^{1/3} layers, large-radius nets keep their
radius for order n layers (the stationary lane's long-memory net, Q = 9 x median, is an instance). The relevance lane's
per-net memory shares for raw a at n = 64 (their `results/report_tables.md`: 0.58, 0.40, 0.15 at t = 8, 16, 32, radial
share 0.5-0.7) fit this two-channel form with rho ~ 0.6 (0.54, 0.36, 0.21).

3.3 How many directions are relevant (HEURISTIC free product [C+A; relevance lane Corollary 1c]; NUMERICAL; PB3 missed on
1 of 12 clauses). Given the past, the frozen future Jacobian's transverse singular frame is Haar (section 7, PROVEN);
assuming the same for the full relevance operator (HEURISTIC), in the linear regime the relevant covariance has the free
multiplicative convolution of the cloud's spectrum and the future's, and
participation ratios combine as 1/PR = 1/PR_C + 1/PR_M - 1/n, with PR_M = n/(2t + 1) for t frozen-gate ReLU layers (one
layer has phi(M^2)/phi(M)^2 = 3). With PR_C = n/tau, PR_rel(t) = n/(tau + 2t). This replaces "Lyapunov covariance times
future amplification, direction by direction": the cloud's axes are past-measurable while the future's amplified
directions (covariant Lyapunov vectors k >= 2 included) are future-measurable and Haar relative to them.
- The future's second spectral moment matches free probability: two orthonormal tangent probes read, up to
  finite-dimensional corrections (the probes live in the n - 1 transverse dimensions) and a ratio-of-means weighting,
  (tr M)^2/(tr M^2 - (tr M)^2/n) = n/(p - 1); the exact tangent chain gives 125.8, 63.6, 32.7, 15.2, 7.9 at t = 1, 2, 4, 8,
  16 (n = 256) against n/(2t) = 128, 64, 32, 16, 8 (`results/prM_tangent.json`, post hoc). This checks one moment ratio,
  not the whole spectral law.
- The output's dependence on the past spans more directions than the linear formula once gates switch across the cloud:

| n | t | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| 64 | total PR (chain) | 15.0 | 12.5 | 9.3 | 5.7 | 3.9 | 2.4 |
| 64 | closed PR (chain) | 15.0 | 12.3 | 8.3 | 4.9 | 2.9 | 1.4 |
| 64 | n/(n/PR_C + 2t) | 12.9 | 9.2 | 5.9 | 3.4 | 1.8 | 1.0 |
| 256 | total PR (chain) | 38.8 | 30.2 | 25.0 | 18.2 | 12.6 | 7.4 |
| 256 | closed PR (chain) | 38.8 | 29.9 | 23.7 | 16.3 | 8.7 | 4.5 |
| 256 | n/(n/PR_C + 2t) | 35.4 | 27.7 | 19.4 | 12.1 | 6.9 | 3.7 |

  The formula is 9-16% low at t = 1 (and t = 2 at n = 256) and 22-58% low at t = 2-4 (PB3's single failing clause:
  total PR at n = 64, t = 4, 1.58 times). Writing PR = n/(n/PR_C + c t), the chain gives c ~ 1.1-1.6 for the closed
  version and c ~ 1.0-1.6 for the total version at t <= 16, between the soft-gate value 1 (each layer acting like a
  Gaussian matrix on differences whose gates differ) and the frozen-gate 2. So at n = 256 the closed relevance of layer
  L - t has variance-weighted participation ratios of about 39, 30, 24, 16 and 9 at t = 1, 2, 4, 8, 16. These are annealed
  moment ratios, not per-net dimensions: Part C finds the median net's cloud PR at about 0.7 of the variance-weighted
  value. For t of order n the formula fails altogether (the future saturates at one direction).

3.4 Survival of a layer-l difference (PB5 missed narrowly: 98 of 100 clauses; eps = 0.1 at n = 64 runs 32-33% above the mean field at t = 20-24 against the 30% band). A replica perturbed by a vector of relative size eps orthogonal
to the state, all noise shared afterwards: in the finite-angle regime the angle decays algebraically, (1 + eps t/(3 pi))^{-2}
before the dilution e^{-t/Q} takes over [A]; for eps t << 3 pi it decays only through the dilution, and the exact annealed
mean decays even more slowly (annealed tangent means are carried by rare nets, section 1). Chain against the mean field
(hidden direction): eps = 0.3 at n = 256 survives 0.937, 0.780, 0.406, 0.207 at t = 1, 4, 16, 32 against 0.929, 0.756, 0.386,
0.198; eps = 0.1: 0.916, 0.703, 0.505 at t = 4, 16, 32 against 0.885, 0.635, 0.434; eps = 1e-3: 0.995, 0.977, 0.864 at t = 4,
16, 64 against 0.961, 0.861, 0.606. At n = 1024 the finite-eps survival matches the mean field within 0-3% (eps = 0.3:
0.778, 0.427, 0.230 at t = 4, 16, 32 against 0.777, 0.423, 0.231). Small differences outlive large ones (10x at t = n/4,
n = 256), the same mechanism as
the interaction order (a lone injection decays slowly, the full cloud fast). Two orthogonal finite perturbations' responses
spread over more directions than tangent ones (n = 64, t = 8: 4.7 against 3.6).

3.5 The stationary answer. At layer L - t, the state decides a fraction 1 - Y((t - 1)/T*) of the output direction's
variance (T* = 2.23 Q^{1/3}), spread over about n/(n/PR_C + c t) directions (c ~ 1.1-1.6; order n^{2/3} near the output, as
variance-weighted participation ratios); the radius carries a share rho(q) of the output variance (small-fluctuation
variance proxy, HEURISTIC) for order n layers. Each net's radius sets T*,
PR_C and rho; the directions themselves are net-specific and invisible to the chain (section 7).

## 4. The bits that matter at layer l: input regime (sigma = 0, replicas = inputs; Part B)

4.1 How many directions: the input cloud's dimension is a four-point-vertex effect (HEURISTIC theory [C+A]; NUMERICAL;
PB6 hit). At infinite width independent inputs are equicorrelated at every layer (correlation c_l, the kernel iterate), so
the infinite-width cloud is isotropic and the kernel says nothing about its dimension. Its anisotropy is a 1/n effect: for
four inputs the scaled double difference D = sqrt(n) (u1 - u2).(u3 - u4) has no radial term at linear order and obeys
D' = kappa'(c_l) D + zeta_l, Var zeta_l = E[(relu z1 - relu z2)^2 (relu z3 - relu z4)^2] (four equicorrelated Gaussians of
unit radius), Var D_0 = 4. The angular participation ratio is PR(l) = 4n(1 - c_l)^2 / Var D_l, and at depth
(theta_l ~ 3 pi/l, Var zeta ~ 2 theta^4, kappa' ~ 1 - theta/pi) PR(l) ~ 1.5 n/l. Correction after review: the first
numerical recursion interpolated a Monte Carlo source on a 201-point grid in c, which near c = 1 turns its quadratic
vanishing into linear and wrongly made the recursion collapse at depth; with the source by quadrature
(`scratch/vertex_exact.py`) the recursion tends to 1.5 n/l (PR/n = 0.650, 0.240, 0.0732, 0.0397, 0.0211, 0.0110, 0.0057 at
l = 1, 4, 16, 32, 64, 128, 256). The registered PB6 targets came from the first version; they differ from the corrected
values by under 2% at l <= 64.

| l | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 127 |
|---|---|---|---|---|---|---|---|---|
| exact chain, n = 256 (4 inputs, PR_w) | 166 | 110 | 62 | 36 | 18.8 | 12.1 | 6.1 | 3.6 |
| vertex recursion (quadrature), n = 256 | 166 | 107 | 61 | 34 | 18.7 | 10.2 | 5.4 | 2.8 |
| input-bits nets, n = 256 (MEASURED there) | 159 | - | 57 | - | 17 | 9.6 | 6.1 | 3.6 (l = 128) |
| exact chain, n = 64 | 40.9 | 26.3 | 15.5 | 9.1 | 5.2 | 3.0 | - | - |
| vertex recursion (quadrature), n = 64 | 41.6 | 26.7 | 15.3 | 8.5 | 4.7 | 2.5 | - | - |

The leading-order vertex is within 1-3% of the exact chain up to l = 16 (n = 256) and 11-27% below it at l = 32-127
(n = 256) and l = 16-32 (n = 64): the finite-width corrections grow as the PR falls toward O(1). (`figs/fig4_input.png`.) So at ARC's 256 x 32 the input's directional information lives
in about 12 effective directions at the last layer (the input-bits lane found "about 10"). The input-regime answer to "how
many bits matter at layer l": at sigma = 0 all of the input's variance is relevant at every layer (the future is
deterministic), and it lives in about 1.5 n/l angular directions plus the radius. The same structure as the stationary
cloud's (width x angular contraction rate theta/pi, halved for a decaying transient). The free-product formula does not
apply here (finite input differences are far outside the future's coherence radius).

4.2 What carries it (identity PROVEN, kernel HEURISTIC, NUMERICAL; PB7 hit). At sigma = 0, a = |h_l| F_l(u_l) exactly, so
log|a| = log|h_l| + log A_l(u_l): the radius at layer l carries the amplitude collected so far, the direction carries the
output direction and the amplitude still to come. With the input-bits lane's per-layer gain covariance,
E_W Var_x(log|h_l|) = (1/2 + C_l)/n, C_l = (1/4) sum_{j<l} (5 - B(c_j)). Exact chain at 256 x 32: the share of
E_W Var_x(log|a|) carried by log|h_l| is 0.187, 0.288, 0.455, 0.643, 0.823, 0.925, 0.990 at l = 1, 2, 4, 8, 16, 24, 31, against
the kernel's 0.19, 0.30, 0.46, 0.66, 0.85, 0.94, 0.99 (within 4%); the absolute value agrees too (0.0357 against 0.0354).
gpt-6-astra computed the raw-a version by an exact change of measure (radius surgery removes 0.04, 0.31, 0.56, 0.63 of
Var_x(a) at l = 0, 4, 16, 31): for raw a the radius never carries the output direction's share.

4.3 Narrow-deep 8 x 256 (NUMERICAL; PB8 hit). Extinction: an all-off layer (probability 2^-8 for any live state) kills the
state; the chain's alive fraction is 0.377 at l = 255 against the exact (1 - 2^-8)^255 = 0.369, and 35% of input pairs are
both alive at the output. Collapse: among chains whose four inputs all survive, the input cloud's angular PR is 5.2, 2.3, 1.5, 1.3 at
l = 1, 4, 8, 16, about 1.1-1.4 for l = 16-128 and 2.0 at l = 255, so after about 8 layers the input is about one direction
plus the radius (the relevance lane's N = 1 there).

4.4 Small noise (descriptive runs, added after registration; `runs/B_inputVP_*`). Closed relevance of the layer-l state
(input and noise up to l shared, noise after l independent), raw output / output direction:

| net | sigma | l = 0 | L/8 | L/4 | L/2 | 3L/4 | L - 2 |
|---|---|---|---|---|---|---|---|
| 256 x 32 | 1e-3 | 0.9994 / 0.998 | 0.9995 / 0.998 | 0.9996 / 0.999 | 0.9997 / 0.999 | 0.9999 / 0.999 | 1.000 / 1.000 |
| 256 x 32 | 1e-2 | 0.952 / 0.852 | 0.956 / 0.865 | 0.962 / 0.878 | 0.974 / 0.910 | 0.987 / 0.952 | 0.998 / 0.993 |
| 256 x 32 | 0.1 | 0.20 / 0.02 | 0.22 / 0.04 | 0.26 / 0.05 | 0.36 / 0.13 | 0.59 / 0.38 | 0.93 / 0.88 |
| 64 x 64 | 1e-3 | 0.9994 / 0.76 | 0.9995 / 0.76 | 0.9995 / 0.76 | 0.9997 / 0.81 | 0.9999 / 0.84 | 1.000 / 0.98 |
| 64 x 64 | 1e-2 | 0.965 / 0.08 | 0.971 / 0.09 | 0.975 / 0.11 | 0.983 / 0.12 | 0.990 / 0.20 | 0.999 / 0.86 |

At ARC's 256 x 32 with sigma = 1e-3 the layer-l state decides 99.94-100% of the variance at every layer: the input
regime holds. At 64 x 64 the output direction is already noise-dominated at sigma = 1e-2 (the input's directional
variance has decayed to ~n/L^2 of the amplitude's), while the raw output stays 96-100% input-decided. At 8 x 256 any
sigma >= 1e-3 hands the output to the noise within about 40 layers (T_mem = 40 at sigma = 1e-3): the cloud's PR rises to
about 6 of 8 and extinction disappears (99% of pairs alive); the closed-relevance estimates there are noise-dominated
(the input's variance at the output is tiny) and are not reported.

## 5. Annealed versus quenched (Part C: real nets)

168 fresh nets (seeds 5000+: 32 at each of n = 8, 16, 32, 64, 24 at n = 128, 16 at n = 256), each with 256 independent noise replicas run from
h = 0 for 8n layers; per net the median radius Qbar, the direction cloud's trace and participation ratio, the output
direction's and log-amplitude's variances, and the closed and total relevance at t = 1-64 (`scratch/partC.py`,
`results/partC_scores.json`, `figs/fig5_per_net.png`). Chain comparators are Part A and B outputs at the same n.

| n | nets | median Qbar/n | CV of cloud size: nets / chain | CV after dividing by delta*(Qbar) | mean cloud size: nets (SE) / chain | median per-net PR / chain PR_w | slope of log PR on log Qbar (Spearman) | median radial share: nets / rho(q) |
|---|---|---|---|---|---|---|---|---|
| 8 | 32 | 0.50 | 0.32 / - | 0.09 | 0.567 (0.032) / - | 6.7 / - | -0.13 (-0.52) | - |
| 16 | 32 | 0.65 | 0.39 / 0.42 | 0.15 | 0.408 (0.028) / 0.403 | 9.0 / 10.0 | -0.13 (-0.61) | - |
| 32 | 32 | 0.73 | 0.45 / 0.49 | 0.17 | 0.276 (0.022) / 0.288 | 13.4 / 14.8 | -0.30 (-0.73) | - |
| 64 | 32 | 0.73 | 0.57 / 0.56 | 0.20 | 0.177 (0.018) / 0.196 | 17.8 / 21.7 | -0.22 (-0.79) | 0.37 / 0.40 |
| 128 | 24 | 0.86 | 0.47 / 0.59 | 0.16 | 0.112 (0.011) / 0.132 | 21.5 / 32.1 | -0.32 (-0.79) | 0.43 / 0.43 |
| 256 | 16 | 0.99 | 0.75 / 0.62 | 0.14 | 0.065 (0.012) / 0.087 | 34.5 / 49.0 | -0.26 (-0.87) | 0.41 / 0.46 |

- The scale does not self-average (PC1, PC2 hit). The per-net cloud size tr Var(u | W) has an across-net CV of 0.39-0.75,
  matching the chain's 4-replica prediction 0.42-0.62 within 30%, and the chain's prediction rises with n toward the
  inverse-gamma limit 0.68 (FORMAL 5.1). The annealed mean is within 2.5 SE of the nets' mean at every n.
- The radius explains the scatter (PC3 hit; hypothesis H-rad). Dividing each net's cloud size by delta*(Qbar(W)), the
  noisy-kernel fixed point at that net's radius, cuts the CV to 0.14-0.20 at n = 64-256 (the squared CV falls by 87%,
  89%, 97%); used directly as a predictor, delta*(Qbar(W)) has R^2 = 0.88, 0.92, 0.93 for the per-net cloud size at
  n = 64, 128, 256 (0.92-0.95 with one fitted scale factor 1.06-1.11). The normalised cloud size is 1.01-1.16 (median) at
  every n: per net, the cloud is the fixed point at the net's own radius (fig 5, left: 168 nets collapse on
  A q^{-2/3} n^{-2/3}).
- The shape self-averages, but the annealed dimension is biased (PC4 missed). Per-net PRs scatter by 0.15-0.35 (CV,
  including sampling noise from 256 replicas) and fall with the net's radius (slope -0.22 to -0.32 at n >= 32, predicted
  -1/3); the median net's PR is 0.89, 0.91, 0.82, 0.67, 0.71 times the chain's variance-weighted PR_w at n = 16-256, which
  fails the registered 25% band at n = 128 and 256. The bias is the radius again: PR_w weights nets by (tr C)^2 ~ Q^{-4/3},
  i.e. small-radius nets, whose clouds are bigger and span more directions; with PR_C ~ Q^{-1/3} the limit ratio is 1.35.
  This is the clearest case of an annealed count misleading about the typical net.
- Relevance per net follows the net's radius (PC5, PC6 hit). The Spearman correlation of V_P(t = 8; W) with Qbar(W) is
  0.67, 0.70, 0.94 at n = 64, 128, 256, and the per-net closed relevance is within 0.02-0.08 (median) of the mean field
  evaluated at the net's own radius at t = 2-16. The annealed curve (a ratio of means dominated by small-radius,
  fast-forgetting nets) sits below the median net's: 0.469 against 0.526 and 0.166 against 0.207 at t = 4, 8 (n = 64);
  0.666 against 0.743 and 0.377 against 0.484 (n = 256).
- The radial share per net follows rho(q) (PC7 hit): median deviation 12%, 11%, 5% at n = 64, 128, 256, Spearman with the
  radius 0.89-0.99 (fig 5, right).
- So per net, the chain's annealed theory plus the net's radius predicts the cloud's size, its relevance horizon, its
  radial share and (with a 1.35x weighting correction) its dimension. What it does not predict is any net-specific
  direction (section 7).

## 6. What the chain explains of what we measured, and what it does not (Part D)

| measured (lane) | chain / theory | status |
|---|---|---|
| cloud size: two noise replicas at angle 2.23 X^{-2/3}, nets measured/law 0.87 (0.72-1.24) (stationary lane, A3) | exact chain: median delta/delta*(Q) 0.96-0.99 at n = 64-1024; per net, cloud size / delta*(Qbar) = 1.01-1.16 (median), R^2 0.88-0.93 | explained: the noisy-kernel fixed point at each net's own radius; nets scatter because their radii do |
| cloud 5-8x above the Lyapunov formula, with an assumed 1/k shape (stationary and numerics lanes) | trace explained by delta*(Q); dimension n theta*/pi ~ n^{2/3} (four-point law), radius dependent in chain and nets | size explained; the 1/k shape is refuted as a global law; the top-k eigenvalue ratios (k = 2-8) were not computed |
| births: 0.48-0.64 sqrt(n) noise-decided gate flips per layer (stationary lane) | exact chain 0.62-0.63 sqrt(n); mean field n arccos(Q/(Q+1))/pi = 0.59 sqrt(n) (numerics lane) | explained in law; per-net spread follows the radius (births ~ (n/Q)^{1/2}) |
| interaction order: direction 1.72 -> 3.97, h_0 1.41 -> 3.87 at n = 8 -> 256 (numerics lane) | mechanism: a lone injection decays algebraically while the cloud relaxes on tau, D_u -> 2.99 Q^{1/6} (numerics lane); exact chain's finite-eps block order 1.90, 2.60, 3.34 at n = 16, 64, 256 (slope 0.20; a lower bound) | explained in mechanism and growth rate; level within about 25% |
| flat low-order shares: C(t) = 0.16-0.18, 0.36-0.38, 0.60-0.64 at t = 0.25, 0.5, 0.75 and S1 0.59-0.66 for n = 8-128 (input-bits) | closure 1 - (1 - t)^{2/3} [A]: 0.175, 0.370, 0.603, first coefficient 2/3; exact chain 0.164-0.192, 0.361-0.406, 0.601-0.654 at n = 16-256 for three kinds of sharing | explained: one fixed point at every width; the growing mean order sits in a tail that C(t <= 0.75) cannot see |
| antithetic noise: even share 0.23-0.27, gain 1.9-2.2 (stationary, input-bits) | E delta(-1)/E delta(0) = 1.58-1.61 (2^{2/3} = 1.587); even share of the direction 0.198-0.209 | explained to within 0.03-0.07 in the even share |
| input regime 256 x 32: amplitude/direction 0.49/0.47 (per-net medians) vs 0.60/0.36 (annealed chain) (input-bits) | the gap is annealed vs quenched (ratio of means weights large-gain nets); in log form the radius share matches the kernel within 4% (4.2) | explained as an averaging effect; the per-net median split is not a replica quantity |
| 3n/l decay of differing gates (input-bits: 128, 100, 74, 51, 33, 18, 11, 5 at layers 1-128, n = 256) | exact chain E[n arccos(rho)/pi] = 128, 101.6, 85.8, 66.7, 47.1, 30.1, 17.5, 9.3, 4.6 for the gates of layers 1, 2, 3, 5, 9, 17, 33, 65, 129; kernel 128, 101.6, 85.9, 67.0, 47.6, 30.8, 18.4, 10.3, 5.5 (`results/partD_gates.json`) | explained (kernel; the chain adds the finite-width deficit seen in the nets at depth) |
| the input's direction lives in about 10 effective directions at 256 x 32 (input-bits) | vertex law PR(l) ~ 1.5 n/l: 12.1 at l = 32 (exact chain), 10.3 (recursion) | explained (new) |

What it does not explain: which directions carry the relevance in a given net (section 7); per-net deviations beyond
what the radius fixes (the 0.14-0.20 residual CV of the cloud size, the 0.15-0.35 scatter of per-net PR); the top-k shape
of the stationary cloud's spectrum (only its participation ratio is pinned down); and anything about estimators, which is
the relevance and estimator lanes' work.

## 7. What the chain cannot account for, and how much that is

- Which directions. PROVEN (FORMAL 1.3; relevance lane Theorem 1 is the sharper form): given the past, the joint law of
  the future responses to perturbations of layer l depends on the perturbations only through their Gram relations with
  the replicas (rotations of layer l fixing the replicas leave it unchanged). So the chain, which
  only sees Gram matrices, carries no information about which directions of layer l are relevant; and the frozen
  Jacobian's transverse singular frame is Haar given the past (relevance lane Corollary 1b, PROVEN). That the frame of the
  full nonlinear, cloud-averaged relevance operator is Haar relative to the cloud is HEURISTIC (it is the free-product
  assumption of 3.3). None of this applies to output coordinates (E[h' | h] is along the all-ones vector).
  Size of what is missing (interpretation, not a bound): describing a generic k-dimensional subspace of R^n to precision
  eps takes about k(n - k) log2(1/eps) bits, while the chain supplies O(k) numbers (spectral moments). With k taken of the
  order of the participation ratio, n^{2/3} near the output in the stationary regime, that is of order n^{5/3}
  log2(1/eps) bits per layer. A participation ratio is not the rank needed to capture 1 - eps of the variance (the
  spectrum's tail is not measured here), and nothing here shows that estimating a mean requires encoding that subspace;
  the kstar-frames lane found a polynomial-size learned map recovering 93-99% of the sensitivity weight.
- Per-net ratios. A fixed-weight Sobol index, participation ratio or relevance share is a ratio of noise
  expectations; the chain gives weight averages of numerator and denominator and their joint moments (2r replicas for
  the across-net second moment of an r-replica quantity), not the average of the ratio. Part C shows how far ratios of
  annealed means sit from per-net medians.
- Raw-state second moments: in the stationary regime the annealed second moments of raw-state differences are infinite
  (section 1), so the chain gives no annealed raw-variance quantity; it gives the direction and log-radius versions.
- Rare nets. Annealed tangent means are carried by rare nets (E_W |J v|^2 = |v|^2 while the typical value decays), and
  heavy radius tails make ratio-of-means estimands weight large-cloud nets.
- Conditioning on a replica's radius approximates conditioning on the net's radius (the radius varies by n^{-1/3}
  within a net and by O(1) across nets) but does not keep replicas iid given the weights; radius-binned chain
  statistics in this report are labelled approximate.
- What this means for M1 (interpretation only). The chain says how much of the output's variance each layer's state
  decides, for how long, and with what participation ratio. It says nothing about which directions, and for the frozen
  Jacobian those are Haar given the past apart from the state's own direction (PROVEN). If a method had to encode a
  generic rank-k relevant subspace per layer, that would cost about k(n - k) log2(1/eps) bits; the measured moments do
  not establish the required rank k (a participation ratio is not a rank), and nothing here shows that estimating the
  mean requires encoding any subspace. Whether it does is the bits-bounds lane's question.

## 8. Coordination with the sibling lanes

- Relevance accounting (work/relevance/, quenched forward-backward theory). Their Theorem 1 (rotation invariance about
  the mean direction) is the sharper form of my FORMAL 1.3, and their free-product Corollary 1c is the same formula as
  FORMAL 3.4 / 7.6 (found independently by them, by me and by gpt-6-astra). Two points I sent them
  (`work/gram-chain/notes/for-siblings.md`): their Gaussian-convention quantities (gamma_l = N(m_l, Sigma_l), K_l(rho))
  are not replica quantities, so the chain cannot give their annealed counterparts exactly (it gives the true-law,
  noise-coordinate versions); and the stationary memory share they need a "separate derivation" for is FORMAL 3.2:
  M(t) = 1 - Y((t - 1)/T*), checked here to within 0.003-0.014 at n = 256-1024 (section 3). Their per-net memory shares
  for the raw output at n = 32-64 fit my two-channel form (1 - rho) V_P^dir + rho V_P^log (section 3.2), and their
  quenched counts N(0.1) (12, 12, 8, 6, 2 at t = 1, 2, 4, 8, 16, n = 64) are of the size of my participation ratios
  (15, 12, 8, 5, 3). One nuance: their Corollary 1c says the degree-1 count follows the backward spectrum; with
  PR_rel = n/(n/PR_C + c t) that holds once c t exceeds n/PR_C (about t > 3 at n = 256), while over the last couple of
  layers the cloud's own dimension n^{2/3} is the binding term. Their dev observation that the stationary
  direction channel is degree-1 dominated supports using the linear free-product formula in the stationary regime. No
  disagreement; no duplicated runs (they measure N(l, eps) and Hermite shares on nets; I measure annealed quantities and
  per-net scatter of cloud size, PR and relevance).
- Bits-bounds (work/bits-bounds/, theory). Their forward-only cell count N_fwd is "an exact finite-width chain
  function"; the chain library (`lib/gc2.py`) computes such functions. No overlap in runs.
- Disagreement reported to main: the numerics lane's 1/k cloud spectrum (and with it the (log n)^2 dimension) is
  contradicted by the four-point law PR_C = n theta*/pi and by its radius dependence in the exact chain (section 2.4).

## 9. Pre-registration scorecard

Registrations: `PREREG.md` (Part A 21:15:54Z, Part B 21:21:14Z, Part C 21:24:21Z, Amendment A1 21:33:29Z, Amendment C1
21:51:59Z; hashes in `prereg_hashes.txt`, copies `coprover/PREREG_*_as_hashed.md`). Scorers `scratch/score_A.py`,
`score_B.py`, `score_C.py`; results `results/part{A,B,C}_scores.json`.

| id | prediction (short) | result |
|---|---|---|
| PA1 | SD(ell) n^{1/3} in [1.70, 2.05] at n = 1024 (limit 1.887), non-decreasing from n = 64 | hit (1.875) |
| PA2 | E[ell^2 | q] n^{2/3} within 25% of the limit in every bin, n = 1024 | hit (within 1-8%) |
| PA3 | median delta/delta* in [0.97, 1.03] at n >= 64; n^{2/3}(mean - 1) constant within 2x | miss (median 0.962, 0.969 at n = 64, 128; mean - 1 falls faster than n^{-2/3}) |
| PA4 | SD(delta/delta*) n^{1/3} in [1.5, 2.4], max/min <= 1.4 over n = 64-1024 | hit (1.65-1.83) |
| PA5 | sharing curves vs closure and each other; antithetic within 10% of 2^{2/3} | hit (all 29 clauses at n = 64, 256, 1024 with the registered baseline) |
| PA6 | 1/e lags: log q in [0.8n, 3.2n], ratio and angle within 2x of their rates | miss (log q at 0.78-0.80 n: n = 64 and 1024 just below 0.8n) |
| PA7 | triangle spread within 25% of 1.46 sqrt(2/PR) | incomplete: within the band at all 5 widths run (6-16% above 1.46 sqrt(2/PR) at n = 16-256); n = 512, 1024 dropped |
| PA8 | cloud moments of a 1/k spectrum (r3 and PR within 25%) | miss (radius-normalised PR 27% below 1/k at n = 64; the r3 clause also fails at n = 64 (registered radius-normalised estimator 3.50 against 1.91), n = 256 not run) |
| PA9 (A1) | PR_w slope between n = 256 and 1024: n^{2/3} (> 0.5) or 1/k (< 0.5) | n^{2/3} favoured (slope 0.614, 90% interval 0.599-0.629; 1/k predicts 0.34) |
| PB1 | closed relevance half-life and area within 30% of the mean field, n = 64-1024 | registered analysis missed (coarse grid; full-grid area integrates noise: -1.28 at n = 256, 66.7 at n = 1024); revised analysis hit (interpolated half-life, area truncated at 8 T*; chosen after seeing n = 256) |
| PB2 | total relevance half-life within 30%, value at 4 T* within 2x | hit |
| PB3 | closed and total PR at t = 1, 2, 4 within 1.5x of the free product, n = 64, 256 | miss (11 of 12 clauses; total PR at n = 64, t = 4 is 1.58x) |
| PB4 | log|a| relevance above the direction's at every t >= 8, > 0.2 at t = n/2 | miss (both near zero at n = 64, t = 160-192, and n = 256, t = 1024; holds at n = 1024) |
| PB5 | perturbation survival within 30% of the mean field (eps = 0.1, 0.3); tangent >= 2x finite at t = n/4 | miss (98 of 100 clauses; eps = 0.1 at n = 64, t = 20-24, 1.32-1.33x) |
| PB6 | input-cloud PR within 20% of the vertex recursion, n = 64, 256 | hit (8 of 8, within 1-17%) |
| PB7 | radius share of log-amplitude within 25% of the kernel, 256 x 32 | hit (7 of 7, within 4%) |
| PB8 | extinction (1 - 2^-8)^l within 3 SE at 8 x 256 | hit (code check) |
| PC1 | across-net CV of the cloud size within 30% of the chain's, which rises toward 0.68 | hit |
| PC2 | chain's annealed cloud size within 2.5 SE of the nets' mean | hit |
| PC3 | dividing by delta*(Qbar) at least halves the across-net CV at n = 64-256 | hit (0.57 -> 0.20, 0.47 -> 0.16, 0.75 -> 0.14) |
| PC4 | median per-net PR within 25% of the chain's PR_w; per-net PR CV < 0.35 | miss (0.67, 0.71 at n = 128, 256: variance weighting) |
| PC5 | per-net V_P(8) tracks the radius (Spearman > 0.5); within 0.12 of the per-radius mean field | hit |
| PC6 | annealed V_P below the nets' median at t = 4, 8 | hit |
| PC7 (C1) | per-net radial share within 30% of rho(q); Spearman > 0.6 | hit (median deviation 5-12%, Spearman 0.89-0.99) |

Totals: 16 hits (PA9 counted as a hit for the four-point law), 7 misses, 1 incomplete with PB1 as revised; 15 hits, 8 misses, 1 incomplete with PB1 as first analysed. The misses are of three kinds: bands that did not allow for noise or grid resolution (PA6 by under 3%,
PB4 at t >= 160, PB5 by 2-3% on 2 of 100 clauses); a finite-n effect larger than predicted (PA3, the angle's level converging from below);
and two substantive findings, both about the cloud's dimension: the 1/k spectrum fails (PA8; the four-point law replaces
it), and the variance-weighted annealed dimension overstates the typical net's (PC4), with the linear free product
undercounting it beyond a couple of layers (PB3).

## 10. Process

- Co-prover gpt-6-astra (`coprover/PROVENANCE.md`; every raw log shows "model: gpt-6-astra"). Explainer check
  (20:51-20:56Z, high effort): 27 findings, 13 errors, all applied to the explainer. Round 1 (21:07-21:31Z, xhigh,
  independent; told not to read this lane's directory): it agreed independently on the scaling limit, the 1.887 n^{-1/3}
  radius ratio, the 5/8 spectral gap, the relevance curves, the free-position formula and the input vertex recursion; new
  from it: the closed form m(q), the second-order angle coefficient, the n^{2/3} cloud dimension (the most consequential
  addition; I re-derived it and registered it against 1/k in Amendment A1 before the decisive runs), the radius-ratio
  effect under shared noise, and the tau^{-3} vs tau^{-5} annealed/quenched contrast. My theory v1 was frozen at
  21:14:08Z, before its answer existed.
- Pre-registration as in section 9. Disclosed before Part A: exploratory runs at n <= 256 informed PA3, PA4, PA7.
  Amendment A1 (21:33:29Z) was hashed before the n = 256 four-replica result existed but about six minutes after that run
  had started (21:27:34Z); I had not seen its output. Its n = 512 and 1024 tests are prospective.
- Deviations (`logs/deviations.log`), none hidden: (1) partB.py's input-mode alive flag was always true (fixed 21:30Z;
  affected only the n = 8 extinction field and n = 8 direction statistics; the buggy output is kept); (2) the registered
  8-replica closed-PR design shared futures across the two past pairs, which biases the closed estimator; it was
  replaced by a 10-replica four-future design before any such run (registered copy kept); (3) PB1's scoring
  (interpolated half-life, area truncated at 8 T*) was changed after seeing the n = 256 curve and is reported both ways;
  (4) score_C.py first treated the per-net radius Q as q (fixed before any Part C result was used); (5) queues were
  restructured to respect the two-process limit at load 20-35: r6 at n = 256 and r3 at n = 512, 1024 were dropped, share
  at n = 1024 ran last, r4 at n = 1024 before 512, and one early third process was stopped after a minute when the load
  rose above 16; (6) an inputVP mode (small-sigma closed relevance) was added after registration and is descriptive;
  (7) Part C's header time was typed one minute ahead of its hash; (8) the sharing runs (A-share) used a 2n burn-in and
  4n recorded layers instead of the registered 6n and 8n, omit t = 0.9, 0.99 at n = 1024, and were first scored against
  each run's own baseline instead of the registered A-r2 baseline (rescored with the registered one; every clause
  passes either way at n = 64, 256); the six-replica runs used half the registered chain budget; (9) the input vertex
  recursion first interpolated its Monte Carlo source on a coarse grid in c, which made it collapse at depth; recomputed
  by quadrature (`scratch/vertex_exact.py`), which changes the deep-layer comparison (section 4.1) but not the registered
  targets at l <= 64 by more than 2%; (10) log|a| needs a convention on the all-off output event (floors at 1e-300, which
  differ between Parts B and C); immaterial at n >= 64, fatal for the log channel at n = 16, which is not used.
- Compute: CPU only, nice 19, ionice -c3, at most two heavy processes (the machine's load was 15-35 throughout), one BLAS
  thread; no GPU.
- Review: GPT-6 (gpt-6-astra, fresh session, high effort, read-only; `review/review_gpt6_r1.md`, 00:10-00:15Z) on the
  near-final draft: verdict major revision, 14 findings (13 major or minor problems, one confirming that the central
  measurements and all 19 tests reproduce). It reproduced the n = 1024 stationary numbers, the four-replica PRs and PB5's
  98 of 100 clauses. Changes made: (1) the log|a| convention is stated and the "annealed raw quantities do not exist"
  claim narrowed to the divergent moments (section 1); (2) the vertex recursion's source is now computed by quadrature
  and section 4.1 no longer attributes its deep-layer gap to a breakdown of the 1/n expansion; (3) the scorecard tallies
  corrected (at that time 14 hits, 7 misses, 3 pending; final tallies in section 9) and PB1 reported both ways; (4) Amendment
  A1's timing stated exactly; (5) the sharing-run deviations disclosed and PA5 rescored with the registered baseline;
  (6) the direction theorem restricted to perturbations of layer l with matched Gram relations, the Haar frame of the
  nonlinear relevance operator labelled HEURISTIC, output coordinates excluded (section 7); (7) the orientation bit
  count presented as a description length, with PR not equal to the rank and no claim that an estimator must encode it;
  (8) the tangent check described as one moment ratio, with the probe corrections; (9) V_P described as the full state's
  closed relevance for the output direction, PRs as annealed moment ratios, the radial share as a variance-component
  proxy; (10) "within a few percent" limited to the relevance curves, PR_C T* given at finite Q; (11) the cloud-size
  scatter statistic stated as a squared-CV reduction and R^2 (0.88-0.93); (12) 168 nets, c ranges, PB4's failing
  points, n = 8 PR conditioning, and the vertex normalisation corrected; (13) completion criteria for the pending items
  stated (section 2.4, 9). Round 2 (fresh session, 01:42-01:46Z; `review/review_gpt6_r2.md`): verdict major revision
  on residuals; it judged findings 2, 4, 5, 8, 11, 12, 14 resolved, reproduced PA9 exactly (slope 0.6143, 90% interval
  0.599-0.629; an independent 10,000-resample bootstrap gives 0.599-0.630) and the available PA5 clauses at n = 1024,
  and asked for: the M1 paragraph to state the bit count only conditionally (if a generic rank-k subspace had to be
  encoded) with no implication for estimators; the frozen-Jacobian / nonlinear-operator distinction in the summary and
  3.3; "or the mean direction" removed from the invariance statement; V_P described as explaining the conditional mean
  of the output direction, not E[a]; "within a few percent" replaced by absolute errors; PB1 marked as revised-analysis
  wherever it says hit; PA7 shown as pending, not failed, in the scores; the direction convention on extinction stated;
  the n = 256 mean-order column recomputed with the registered baseline (4.06 / 4.06 / 3.34). All applied. After round 2
  the remaining registered runs landed (PA5 at n = 1024: hit; PA7: in band at every width run; the third-moment part of
  PA8); they were scored by the unchanged registered scorers and were not re-reviewed.

## 11. Files

All under `work/gram-chain/`:
- `FORMAL.md` (theory v2: sections 1-6 are v1 as frozen at 21:14:08Z, kept also as `coprover/FORMAL_v1_as_written.md`;
  section 7 holds the v2 additions); `notes/questions.md` (premises checked, open questions with their checks);
  `notes/for-siblings.md` (what I sent the relevance and bits-bounds lanes); `HANDOFF.md`.
- `lib/gc2.py` (the chain: any R, correlated / label / coordinate noise sharing, tangents, surgery, inputs, output layer,
  mean-field fixed point); `tests/test_gc2.py` (`logs/test_gc2.log`, 19 checks pass).
- `PREREG.md` (Parts A, B, C; Amendments A1, C1), `prereg_hashes.txt`, `coprover/PREREG_*_as_hashed.md`;
  `logs/deviations.log`.
- `scratch/`: `partA.py`, `partB.py` (`partB_as_registered.py`), `partC.py` (runs); `score_A.py`, `score_B.py`,
  `score_C.py` (scorers); `theory_numbers.py`, `predB.py`, `onestep.py`, `pr_by_radius.py`, `partD_gates.py`
  (theory and post hoc numbers); `figs.py`, `tables.py`; `quick_*.py` (exploratory, disclosed); queue scripts.
- `runs/` (one JSON or NPZ per run), `results/` (scores and tables), `figs/` (fig1_stationary_law, fig2_relevance,
  fig3_dimension, fig4_input), `coprover/` (astra prompts, raw logs, answers, `PROVENANCE.md`), `review/`.
Python: /home/agent/.venvs/arc-noisy/bin/python.

