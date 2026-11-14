# PREREG, stationary-regime lane, Part A (written 13:21Z 31 Oct, before any evaluation-seed run)

Theory: `FORMAL.md` v1 (frozen 13:15Z, sha in `formal_v1.sha256`). Dev data seen before this registration (dev seeds
1-5 only): `scratch/dev_probe.py` (n = 8, 32), `scratch/dev_chain.py` (seed 5, n = 8, 32, 64), and
`scratch/partA.py` on seed 3 (cloud at n = 16, 64; births at n = 16; age, lyap, couple at n = 64; JSON in
`runs/dev_*`). What dev showed is stated next to each prediction; the predictions below are therefore dev-informed
and are tested on fresh evaluation seeds and on widths not seen in dev (128, 256).

## Evaluation set and runs (Model S, sigma = 1 unless stated; Model S is scale-free, FORMAL item 2)
- Nets: seeds 1000-1003 at n in {8, 16, 32, 64, 128, 256}. Expensive modes use the first 2 seeds at n >= 128.
- `partA.py cloud`: P = max(2048, 8n) samples, burn-in B = 12n. `births`: P = 512, K = 64 layers. `age`: P = 512,
  ages 0..4n. `lyap`: 1 trajectory, 30n layers, k = min(n, 64), first 2n discarded. `couple`: P = 256, B = 30n.
- Radius: `scratch/radius_law.py` (to be written): the exact scalar chain of FORMAL item 3 (>= 4e6 draws per n)
  against long chains of the evaluation nets (n <= 64: 4 nets x 8 paths x 40,000 layers; n = 128, 256: 2 nets x 4 paths
  x 20,000 layers; first 20n layers discarded).
- Model R sub-study (`scratch/partR.py`, to be written): n in {32, 128}, sigma in {1e-3, 1e-2, 1e-1}, 2 nets each,
  one fixed input per net, P = 1024 noise samples, measured at depth L = 4n and 16n after the input.
- Compute: CPU only, nice 19 + idle IO, at most 3 heavy processes, single-threaded BLAS.

## Predictions and refutation conditions
PA1 (exact radius chain; a code and ergodicity check of a theorem). Pooled over layers and nets, the 10/50/90%
quantiles of X = |h|^2/(n sigma^2) agree with the scalar chain's within 10% at every n. Refuted if any differs by more
than 20% at n <= 64. (Dev, seed 5: medians within 2% at n = 8, 32, 64.)
PA2 (Dufresne bulk and Kesten tail). Scalar-chain median of X/n within 10% of 2/(5 ln 2) = 0.577 at n = 256 and
approaching it monotonically in n; Hill index of X (top 1% of the pooled net samples) in [0.7, 1.4] for n >= 16.
PA3 (Lyapunov). n lambda_1 in [-1.6, -1.0] for n >= 16 (prediction -1.25); mean gap n(lambda_k - lambda_{k+1}) over
k = 1..8 in [0.7, 2.0] for n >= 32 (dev n = 64: lambda_1 n = -1.29, gaps 0.8-1.6).
PA4 (synchronisation). Coupling-from-the-past rate n d(log dist)/dT in [-1.6, -1.0] for n >= 16 (dev: -1.29).
PA5 (cloud shape). Log-log slope of the covariance eigenvalues of h_0 over k in [2, n/8]: in [-1.3, -0.7] for
n >= 32 (coordinator's harmonic shape; dev -0.92, -0.96). The frozen-gate Lyapunov recursions (along one path's gates
and along the attractor's gates) give slopes <= -1.4 (dev -1.5 to -2.2), i.e. they get the shape wrong.
PA6 (cloud size against the coordinator's sigma^2 / (1 - exp(-2(k + 1/4)/n))). The median over k = 2..8 of
measured/coordinator increases with n (Spearman over the six widths >= 0.8) and is >= 2 at n = 128 and 256
(dev: about 1.2 at n = 16, about 2.9 at n = 64). Refuted if <= 1.5 at n = 256.
PA7 (radial dominance). Median over nets of the radial share E(h.u*)^2 / E|h|^2 increases with n and is >= 0.85 at
n = 256 (dev: 0.41 at n = 16 on a low-radius layer, 0.79 at n = 64).
PA8 (births, exact formula). In every X_pre decile bin with median X_pre >= 1, mean flips under resampling one
layer's noise are within 15% of n arccos(X/(X+1))/pi, at every n (dev n = 16: within 3-5%). The coordinator's
near-threshold count is within 20% of n sqrt(2/pi)/sqrt(X) in bins with X >= 10.
PA9 (births scale like sqrt(n)). Mean flips per layer divided by sqrt(n) in [0.4, 1.2] at every n >= 16, and its
max/min ratio across n = 16..256 below 1.6 (FORMAL item 6: 0.71 E sqrt(Z) ~ 0.63).
PA10 (age profile). (a) Radial share of the age-j response at j in [n/2, n] >= 0.4 for n >= 32 (dev 0.53-0.56).
(b) Participation ratio of Cov(Delta_j) at j = 4 and 8 within a factor 2 of the coordinator's n/(2j) + 1.
(c) Beyond j = n the total effect decays with an e-folding rate between 1/n and 4/n per layer.
(d) Interaction excess: sum over all ages of the total-effect shares exceeds 1.5 at every n >= 16 (an additive,
linear-Gaussian state would give exactly 1; dev n = 64 about 2.6). (e) Ages j >= n carry < 5% of the summed effects.
PA11 (where near-Gaussianity fails). At n >= 64, the median |skewness| of the projections on covariance eigenvectors
2-8 is < 0.3 while the top (radial) projection has |skew| > 0.5 or |excess kurtosis| > 0.8 on >= half the nets.
PA12 (scale invariance, code check). cloud at sigma = 1e-3, 1e-2, 1e-1 (n = 32, seed 1000): X quantiles and
evC/sigma^2 equal to the sigma = 1 run within 1e-8 relative.
PR1 (Model R births). Flips per layer within 25% of the Gaussian-unit formula with the measured per-path ratio
(about 0.45 n sigma for small sigma) at sigma = 1e-2 and 1e-1.
PR2 (Model R cloud). With the input fixed, the angular variance of the noise cloud relative to |h|^2 scales as
sigma^2: the ratio between sigma = 1e-2 and 1e-3 is in [50, 200]; its eigenvalue slope over k in [2, n/8] at L = 16n
is in [-1.3, -0.7].
Descriptive (no pass/fail): output-layer gate entropy and uncertain-gate fraction against n; within-net spread of
log R; cosine of the state to u*.

## Analysis rules fixed now
Medians over nets; slopes by least squares in log-log; "within x%" on the ratio measured/predicted. Sums over ages
in PA10 use the trapezoid rule over the sampled ages (j treated as continuous). Any rerun after
a bug fix is reported with both versions.

# PREREG Part B (written 13:40Z 31 Oct, after the Part A scorecard, before any Part B evaluation run)

Dev data seen (seed 3, n = 16 only; `runs/dev_*_n16_s3.json`): burn-in MLMC variance x cost 156 trVar against 65 for
MC with burn-in 64; RQMC 3-6x at N = 64-1024; soft-gate Gaussian-surrogate CV residual 0.29 (0.25 with final-noise
Rao-Blackwellisation), hard-gate surrogate 0.47; final-noise RB alone 0.55; antithetic pair ratio 0.47; K2 closure
nMSE 2.0e-3; tail at p = 1e-3: radial tilt biased low (heavy weights), gradient tilt s = 2 unbiased with 11x.
Post-hoc Part A check (`runs/cloudhist_*`): a radius-history linear model explains only part of the cloud excess.

Target: E[a] and P(|a| > t), a = relu(W_1 h_0), h_0 ~ Model S's stationary sample measure (sigma = 1, fixed weights).
t = the 0.99 quantile of |a| (p = 1e-2 at every n; the 1e-3 level is too thin for the n = 256 truth).
Code: `scratch/partB.py` (sha in prereg_hashes.txt). Cost unit: one n x n matrix-vector product (mv).
Variance x cost constant K (in mv, per unit of nMSE): per-sample cost x per-sample variance / tr Var(a). Plain MC with
burn-in B has K = B + 1. All samplers start from h = 0 at layer -B + 1 with B = 8n (bias^2 measured by `mlmc`).

Runs: seeds 1000, 1001 at n in {8, 16, 32, 64, 128, 256}.
- truth: B = 8n; M = 2^17 (n <= 32), 2^16 (64), 2^15 (128), 2^13 (256).
- mlmc: M = 4096 (n <= 64), 2048 (128), 512 (256). cv: M = 8192 (n <= 64), 4096 (128), 2048 (256).
- anti: M = 4096 pairs (n <= 64), 2048 (128), 1024 (256). rqmc: n <= 64: R = 16, N = 2^6..2^12; n = 128: R = 8,
  N = 2^6..2^10; not run at n = 256 (cost). tail: M = 8192 (n <= 64), 4096 (128), 2048 (256) per variant; radial and
  gradient tilts at s = 1.5 and 2.5.

Predictions (refutation in brackets):
PB1 (burn-in MLMC). K_ML / n in [4, 25] at every n, and K_ML > B_6 + 1 on at least half the nets, where B_6 is the
smallest dyadic burn-in with bias^2/trVar <= 1e-6 (no practical gain; only the asymptotic log factor).
PB2 (burn-in bias). The decay rate of bias^2/trVar over T in [2n, 8n], times n, is in [1.5, 3.5].
PB3 (Gaussian-surrogate CV with final-noise RB). Residual variance fraction decreases with n (Spearman over the 12
nets <= -0.6) and is <= 0.2 at n = 256 on both nets. [Refuted if Spearman > -0.3 or > 0.35 at n = 256.]
PB4 (K2 closure as the CLT test). nMSE of the deterministic K2-Stein closure decreases with n (Spearman <= -0.6) and
is <= 5e-4 at n = 256 on both nets.
PB5 (RQMC). MC/RQMC variance ratio at N = 2^10 is >= 2 at n <= 16 and < 2 at n = 128; the fitted exponent of RQMC
variance in N over 2^6..2^12 (2^10 at n = 128) is in [1.0, 1.3] at n >= 32.
PB6 (final-noise RB). Variance ratio in [0.4, 0.9], increasing with n (Spearman >= 0.6).
PB7 (antithetic). Pair variance / (trVar/2) <= 0.7 at every n.
PB8 (tail). The better gradient tilt has relative variance per sample <= (1-p)/p / 5 on at least half the nets at
every n, with |p_IS / p_truth - 1| within 3 combined SE; the radial tilt is worse than the gradient tilt on at least
half the nets.
HB (main hypothesis: no width-growing gain). Let G(n) be the best gain over MC with burn-in B_6 among the unbiased
methods and their combinations measured here (CV+RB, antithetic, RQMC at its largest N, and products of separately
measured gains are NOT claimed as measured combinations). HB: G(256) < 3 G(16) (median over nets). [Refuted if
G(256) >= 3 G(16).] Every unbiased method has MSE-vs-cost exponent 1 except RQMC, whose fitted exponent is <= 1.3;
the K2 closure has exponent 0 (a floor).

# PREREG Part C (written 13:43Z 31 Oct, before any Part C evaluation run; dev: two smoke runs at n = 32, seed 3)

Model S with an input: x ~ N(0, I_n) at layer 0, absolute noise sigma after each hidden layer, layers 1..L, output
a_L = relu(W_L h_{L-1}); weights st.W_layer(seed, n, l), l >= 1. Code `scratch/partC.py`. Heuristic memory time
T_mem = (4n/5) ln(1/(sigma sqrt(0.4 n))) (FORMAL v1 item 7; astra P16).
Grid: n in {32, 128, 256}; sigma in {1e-3, 1e-2, 1e-1}; L in {n/8, n/2, 2n, 8n} (n = 256: {32, 128, 512});
seed 1000 everywhere, also 1001 at n = 32 and 128. K x M = 128 x 8 (n <= 128), 64 x 8 (n = 256).
Dev seen: n = 32, sigma = 1e-3, seed 3: at L = 4 input share 1.0, noise effect 3e-8, anti-both 0.73, RQMC over x 3.6x;
at L = 512 input share 0.04, anti 0.45, births 1.9 per layer.

PC1 (memory time). On every (n, sigma) with T_mem >= n/8, the input share S_x is >= 0.9 at grid depths L <= T_mem/4
and <= 0.2 at grid depths L >= 2 T_mem.
PC2 (relative noise grows from sigma). At L <= T_mem/2 the median relative noise eps(L) at layer L-1 is within a
factor 2.5 of sigma exp(5L/(4n)).
PC3 (noise births track eps). At L <= T_mem/2, births per layer are within a factor 1.5 of (sqrt 2 / pi) n eps_med.
PC4 (ARC regime, n = 256, L = 32, sigma = 1e-3). S_x >= 0.999; noise effect on the mean <= 1e-5 (nMSE); births per
layer <= 0.3; input gate entropy at layer L-1 >= n/2 bits; the coupled noise correction has variance <= 1e-3 trVar.
PC5 (which structured gains survive the input regime). (a) Antithetic in the noise alone gives no gain (pair ratio
>= 1.5) wherever S_x >= 0.9, and the stationary-regime gain (ratio <= 0.7) wherever S_x <= 0.2. (b) RQMC over x alone
gains >= 2x wherever S_x >= 0.9 and L <= n/2, and < 1.5x wherever S_x <= 0.2. (c) The coupled two-level estimator
(noiseless sampler + noise correction) has correction variance <= 0.01 trVar wherever L <= T_mem/4: there the noise
is a cheap correction and the noiseless samplers' gains (RQMC, the multilevel reference-trajectory sampler) carry over.
PC6 (one budget). Input gate entropy (bits over inputs at layer L-1, noiseless) falls with depth and noise births
rise; on every (n, sigma) the depth where births x n/2 first exceed the input gate entropy lies within [T_mem/8, 4 T_mem]
of the grid (coarse: the grid is factor-4 spaced).

## Amendment C1 (13:51Z, before any Part C evaluation run; dev: n = 32, seed 3, L = 4 and 64)
partC.py also runs a crossed design (32 inputs x 32 shared noise sequences) and reports the two-way random-effects
ANOVA shares: input main, noise main, input x noise interaction (of their sum). Dev at L = 64, sigma = 1e-3: 0.51 /
0.36 / 0.13. PC7: (a) wherever S_x >= 0.9 the interaction share is <= 0.05; (b) it never exceeds 0.35 on the grid
(once the noise matters, inputs have mostly merged, so the noise acts as a main effect); (c) wherever L >= 2 T_mem
the input main share is <= 0.1.

## Amendment A2 (13:56Z, after the Part A scorecard, before any evaluation run of `partA.py bitlife`)
Dev seen: n = 16, seed 3 only. Runs: n in {16, 64, 256}, seeds 1000, 1001, P = 512, ages j in {2, 8, n/4, n/2, n, 2n}.
PA13 (bit birth and death after resampling ONE layer's noise). (a) Gate flips at the first layer after the resampled
one are in [0.35, 0.8] sqrt(n) (the births law, PA9). (b) The flip count dies out: at n >= 64, flips at t = n/2
layers after the perturbation (age j >= n) are below half the flips at t = 1, and the total cascade summed over the
2n layers after the perturbation is between 0.3 n and 3 n flips. (c) The frozen-gate linear response (along the
baseline path's gates) to a resampling at age 8 correlates with the true change of h_0 more at larger n (Spearman over
the 6 nets >= 0.6) and at >= 0.8 at n = 256 (dev n = 16: 0.40).

## Amendment A3 (13:58Z; tests astra's round-2 finite-angle closure on fresh nets, before any of these runs)
Astra (round 2, after reading the Part A data) derived, HEURISTIC: two noise replicas at fixed weights balance the
noise injection 1/X per layer against the ReLU kernel's finite-angle contraction (2 sqrt 2/(3 pi)) delta^{3/2}, so
delta = E[1 - u.u'] ~ 2.23 X^{-2/3} (X = |h|^2/(n sigma^2) at that layer), and the quenched radial CV^2 ~
2.23 X^{1/3}/n + 1/(2n). Post hoc on the Part A nets (n = 32-256, 14 nets, X = 12-1182) the angular proxy
(ang_total_rel) / (2.23 X^{-2/3}) has median 1.06 (0.70-1.52) and R_cv / law has median 0.84 (0.65-0.97).
Fresh test: `partA.py cloud` (now also reporting delta_pairs, the direct two-replica dispersion) on seeds 2000-2003 at
n in {16, 32, 64, 128} and 2000-2001 at n = 256, P = max(2048, 8n), sigma = 1.
PA14: delta_pairs / (2.23 X_med^{-2/3}) has median in [0.75, 1.33] and every value in [0.5, 2] at n >= 32. At n = 16
(not predicted) reported only.
PA15: R_cv / sqrt(2.23 X_med^{1/3}/n + 1/(2n)) has median in [0.65, 1.05] and every value in [0.5, 1.25] at n >= 32.

# PREREG Part I: the interaction structure (written 14:10Z, before any evaluation run of `scratch/partI.py`)
Kelvin (via the coordinator, 14:05Z): "Yes, we need to understand the structure of this" (Part A's summed total
effects 2-4x the variance, growing with n). Dev seen: n = 16, seed 3, P = 256 (all four modes, smoke test).
Definitions. Model S stationary, window = last J = 4n layers before layer 0, conditional on each sample's deep past.
Blocks = one layer's noise in causal radial/angular coordinates (eta_l, zeta_l) (partI.py docstring). Observables:
h = h_0, logR = log|h_0|, u = h_0/|h_0|, a = relu(W_1 h_0), g = output gates. G_f(eps) = E|f - f_eps|^2/(2 V_window)
with each block refreshed with probability eps; mean dimension D = G(0.01)/0.01 (= sum of total Sobol indices over
blocks = the variance-weighted mean interaction order); bulk order D_0.1 = G(0.1)/0.1. Superset importance of an age
pair Y2_jk (Liu-Owen) = sum of the ANOVA variances of all terms containing both ages. Truncation: C(d) = closed index
of the most recent d layers; sharing floor S_D(d) = variance fraction explained by everything older than d.
Mechanism controls: (a) radius only: the annealed scalar Kesten chain (per-layer blocks (M, U, V)); (b) gate switching
only: the radius-clamped model (h <- Rbar h/|h| after every layer, Rbar = median stationary radius).
Runs: seeds 1000, 1001 at n in {16, 64, 256}; P = 1024 (n <= 64), 512 (n = 256); modes curves (Model S and clamp),
pairs, trunc; kesten at n = 16, 64, 256.

PI1 (consistency). D for h is within a factor 1.5 of the Part A Efron-Stein sums (medians 2.1, 2.55, 3.9 at
n = 16, 64, 256).
PI2 (heavy-tailed dimension distribution). For h and a at every n, D_0.1 < D/2: most variance sits in low-order terms
while a small variance fraction in high-order interactions dominates the mean dimension (dev n = 16: 1.44 vs 3.3).
PI3 (where the variance lives). Total effect of ALL angular coordinates >= 0.8 and of ALL radial coordinates <= 0.3,
for h and a at every n: the radial noise components are not the important coordinates.
PI4 (mechanisms). (i) The Kesten chain's D for log Q is in [1.0, 1.6] at every n. (ii) Model S's D for logR exceeds
the Kesten chain's D for log Q at every n. (iii) The clamped model's D for u is >= 0.7 x Model S's D for u at every n.
PI5 (growth with n, 16 -> 256). D for h grows by >= 1.2x; the Kesten D for log Q changes by a factor in [0.8, 1.25];
the clamped model's D for u grows by >= 1.2x. Reading: if all three hold, gate switching (b), not the radius (a),
drives the growth.
PI6 (pairs). For every pair with j <= 8, Y2 of the output gates g is >= 2x Y2 of a. Y2 of h at fixed j in {0, 2}
decreases monotonically with the lag over the grid.
PI7 (truncation, sharing floor). The 99% truncation dimension of a (smallest grid d with C_a(d) >= 0.99) is between
n/2 and 4n layers; S_D(n) for a is <= 0.1 at n >= 64.
PI8 (link to QMC). D_0.1 for a is <= 2 at every n.

## Amendment B1 (14:19Z, after the Part B scorecard, before these runs): does the antithetic gain keep growing?
Part B: antithetic pair ratio (1 + corr(a(xi), a(-xi))) fell from 0.50-0.54 (n = 8) to 0.41-0.42 (n = 256). Runs:
`partB.py anti` at n = 512 (B = 8n, M = 256 pairs) and n = 1024 (B = 4n, M = 128 pairs), seed 1000.
PB9: the ratio is in [0.30, 0.45] at both widths (a slow decline, log-linear extrapolation 0.37 at n = 1024).
Refuted if below 0.25 (the even part vanishing quickly, which would make the free 2x gain grow with width) or above 0.5.

## Amendment B2 (14:19Z, before these runs): is RQMC's exponent above 1 asymptotic at small width?
Part B fitted RQMC variance exponents 1.19-1.22 at n = 8, 1.13 at 16, 1.10-1.12 at 32 over N = 2^6..2^12. Runs:
`partB.py rqmc` with N up to 2^16 at n = 8, 16 and 2^14 at n = 32 (seeds 1000, 1001, R = 16; new output files).
PB10: the exponent fitted over N = 2^10..2^16 (2^10..2^14 at n = 32) is in [1.0, 1.15] at every width, i.e. lower than
over 2^6..2^12 (a pre-asymptotic gain). Refuted if >= 1.2 at n = 8 (a persistent rate improvement at small width).
