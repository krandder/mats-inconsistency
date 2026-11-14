# PREREG, stationary-numerics lane

## Part R: the exact radius law at finite n (written 15:07Z 31 Oct, before the batch below)

Theory used (no batch data): `lib/radius_chain.py` (exact reduction Q = (S/n) r, r' = 1 + A r, Nystrom solver);
`scratch/km_first_order.py`, `scratch/km_second_order.py` (closed-form Kramers-Moyal terms, `results/km_*.txt`):
  F_n(xn) = P(Q <= xn) = F0 + F1/n + F2/n^2 + ..., F0 = exp(-2/(5x)),
  F1 = -(65x - 47) exp(-2/(5x)) / (375 x^2),
  F2 = -(198375x^3 - 18825x^2 + 63640x - 6627) exp(-2/(5x)) / (843750 x^4);
  median/n = 0.5770780164 - 0.0632661929/n + 0.5192608/n^2; C_n = 2n/5 + 13/75 + 529/(2250 n) + O(n^-2);
`scratch/radius_ldp.py` (left-tail large-deviation rates, theory): P(r <= rho) ~ exp(-n J(rho)),
J(2) = 0.21424, J(3) = 0.13850, J(5) = 0.08164; P(Q <= q) ~ exp(-n K(q)), K(1) = 0.31071, K(2) = 0.17412.
Dev data seen before this registration: `logs/test_radius_chain.log` (solver quantiles at n = 2, 3, 4, 8, 64, 256;
x P(Q > x) at x = 1e7 n; MC at n = 4, 8, 32 for the 10/50/90% points), `scratch/dev_radius1.py` (n = 8, 64), and the
stationary lane's `runs/radius_n{8,64}.json` quantiles (compared by eye with the solver at n = 8, 64).

Batch: `scratch/radius_corrections.py` at n in {2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
768, 1024}; `scratch/radius_mc.py` at n in {8, 16, 32} (P = 1e5 chains, T = 4000 steps); comparison with the
stationary lane's `runs/radius_n{8..256}.json` (scalar chain 4.1e6 draws; nets pooled over layers).

Predictions (refutation condition in brackets):
R1 (KM expansion is the asymptotic series). Over x in {0.1, 0.15, 0.2, 0.3, 0.4, 0.577, 0.8, 1, 1.5, 2, 3, 5, 10, 30,
100}: max_x |n(F_n - F0) - F1| <= 1.5 max_x |F2| / n at every n >= 64; and the two-term remainder
R2 = n^2(F_n - F0 - F1/n) - F2 shrinks like 1/n: |R2(512)| / |R2(1024)| in [1.6, 2.6] at x = 0.577, 1, 2.
[Refuted if either fails at any listed n or x.]
R2 (median). |median/n - (0.5770780164 - 0.0632661929/n + 0.5192608/n^2)| < 1e-7 at n >= 256 and < 5e-3 at n = 16.
R3 (Kesten tail of the solved law). x P(Q > x) at x = 1e8 n equals C_n = 1/E[A log A] within 2e-6 (relative) at every
n; at x = 1e4 n the deficit 1 - xP(Q>x)/C_n is in [0.1, 0.5] x n/x for every n >= 32.
R4 (C_n series, exact arithmetic). n^2 (C_n - 2n/5 - 13/75 - 529/(2250n)) agrees between n = 512 and 1024 within 5%;
n^3 (E[A log A] - 5/(2n) + 13/(12 n^2)) is within 1% of -1 at n = 1024.
R5 (left tail). -(1/n) ln P(r <= 2) is within 10% of J(2) = 0.2142 at n = 1024 (direction of approach not
predicted); same for -(1/n) ln P(Q <= 1) against K(1) = 0.3107. The inverse-gamma extrapolation exp(-0.4 n) for
"near-floor" layers (Q <= 1) is refuted at large n: -(1/n) ln P(Q <= 1) < 0.36 at n = 1024.
R6 (where the expansion fails). At n = 2 the two-term median formula is off by more than 1% (relative).
R7 (moments). n (E sqrt(Q/n) - sqrt(2 pi/5)) and n (E log(Q/n) - log(2/5) - 0.5772) each change by less than 10%
between n = 512 and 1024 (both errors are O(1/n)).
R8 (simulation). (a) Direct chain MC (`radius_mc.py`, n = 8, 16, 32): P(Q <= exact q-quantile) within 4 SE of q for
q in {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}, and P(r <= 2), P(Q <= 1), P(Q <= 2) within 4 SE of the solver.
(b) The stationary lane's chain quantiles (q10, q50, q90) are within 1% of the solver at every n (8-256), q99 within 5%.
(c) Its net quantiles (pooled over layers and nets) are within 3% at q10 and q50 for n <= 64, within 10% at q90.

## Part G: exact two- and four-replica Gram chain against the effective theory and the stationary lane (written 15:15Z, before the Gram batch)

Theory (no batch data): `lib/gram_chain.py` (exact replica Gram chain; validated against full-network replicas,
`logs/test_gram_chain.log`); level-0 effective theory `scratch/l0_theory.py` -> `results/l0_theory.json`: infinite-width
noisy kernel recursion with the radius held at Q = 0.577n, rho' = (Q kappa(rho) + s)/(Q + 1) (s = 1 shared noise,
0 independent; HEURISTIC adiabatic model). L0 numbers: delta* = 0.369, 0.176, 0.0756, 0.0310 at n = 16, 64, 256, 1024;
angular interaction order D_u = 1.73, 2.68, 4.19, 6.27; output-gate order D_g = 6.8, 16.0, 36.4, 78.7; births/sqrt(n)
= 0.57-0.59; gate cascade over the 2n layers after one refresh = 1.69n, 2.75n, 4.15n, 5.83n.
Dev seen: one smoke run each of `scratch/gram_experiments.py window|pr 16 --P 300` (runs/dev_gram_*_n16.json).
Batch: `gram_experiments.py window n` at n in {8, 16, 32, 64, 128, 256, 512, 1024}, P = 8000 (n <= 64), 4000 (128,
256), 2000 (512), 1200 (1024); `gram_experiments.py pr n` at n in {16, 32, 64, 128, 256, 512}, P = 4000 (n <= 128),
2000 (256, 512). Comparison data from the stationary lane: `runs/cloudA3_*` (delta_pairs, R_cv, X), `runs/partI_*`
and `results/partI_scores.json` (mean order D per net), `runs/bitlife_*` (cascade), `runs/births_*`.
"Median bin" = the bin of Q_final/n containing 0.577 ([0.5, 1)).

G2 (cloud angle). In the median bin, the exact mean delta of the independent-noise pair is within [0.8, 1.2] x L0's
delta*(Q) at the bin's median Q for every n >= 64, and |exact/L0 - 1| decreases from n = 64 to n = 1024.
G3 (radial asymmetry). In the median bin, CV^2 := E[(log|h1| - log|h2|)^2]/2 is within a factor 1.5 of astra's
2.23 X^{1/3}/n + 1/(2n) at X = the bin's median Q, for n >= 64.
G4 (interaction order, annealed ratio of means over the 4n window). (a) In the median bin, D_u is within [0.7, 1.3] x
L0's D_u at n >= 32. (b) D_h and D_u at n = 1024 are >= 1.3 x their values at n = 64 (the order grows with width).
(c) The log-log slope of D_g over n = 64..1024 is in [0.40, 0.75]. (d) At n = 16, 64, 256 the exact D_h, D_u, D_a
are within a factor 1.5 of the stationary lane's per-net Part I medians (h 2.5, 2.1, 3.1 at n = 16, 64, 256).
G5 (births, cascade). births/sqrt(n) within 5% of L0 at every n >= 16; cascade total over 2n layers within 25% of L0
at every n >= 32, and within 30% of the stationary lane's bitlife medians at n = 64 and 256.
G6 (angular participation ratio, annealed within the median bin). PR_u within a factor 2 of the harmonic value
(H_n - 1)^2/(pi^2/6 - 1) at every n in 16..512 (a 1/k angular spectrum from k = 2 to n).

## Part I: interaction order across width and causal gate-freezing tests (written 15:18Z, before the Part I batch)

Code: `scratch/interact_net.py` (sha below). Nets: the stationary lane's st.W_layer(seed, n, l). Estimator (one for all
n): per net D_f = (J+1) mean_s |f_s - f_s^(j_s)|^2 / mean_s |f_s - f'_s|^2 with j_s uniform over the J+1 = 4n+1 window
layers (unbiased for the sum of total Sobol indices over layer blocks, conditional on the deep past), bootstrap SE.
Dev seen: one net (seed 3, n = 16, P = 512, all variants; runs/dev_interact_n16_s3.json): D_h 1.45 +- 0.22 (true),
0.87 +- 0.15 (frozen); D_u 1.46, D_g 4.6 (true).
Batch I1: variants true,frozen; seeds 1000-1007; n = 8, 16, 32 (P = 4096), 64 (2048), 128 (1024), 256 (1024).
Batch I2: variants true,frozen,free_0.5,free_1,free_2,free_4,recent_q,recent_n,clamp; seeds 1000-1003;
n = 32 (P = 4096), 64 (2048), 128 (1024), 256 (1024; seeds 1000-1001 only).
Excess fraction E_v = (D_v - D_frozen)/(D_true - D_frozen) per net (f = u unless stated), medians over nets.

I1a (width trend, true model). Median D_u over nets increases with n: Spearman over the six widths >= 0.8 and
median D_u(256) / median D_u(16) >= 1.5. Same for D_h with ratio >= 1.3. Median D_g(256)/D_g(16) >= 3.
I1b (L0 comparison). Median D_u within [0.6, 1.4] x the L0 value (results/l0_theory.json) at every n >= 32.
I2a (code check, PROVEN: h_0 affine in the window noise). frozen: |D_h - 1| < 3 SE on at least 46 of the 48 I1 nets.
I2b (who carries the interactions). E(free_1) < 0.5 and E(free_4) > 0.7 at n = 64; E(free_4) at n = 256 is below
E(free_4) at n = 32 (the gates that matter sit at a margin growing like Q^{1/6} noise SDs).
I2c (age). E(recent_n) >= 0.8 at every n; E(recent_q) in [0.3, 0.9] at every n.
I2d (radius not needed). D_u(clamp) >= 0.7 D_u(true) (median over nets) at every n.

## Part S: exact quenched densities at n = 2, 3, 4 (written 15:46Z, before the Part S evaluation runs)

Code: `lib/grid_transfer.py` (tests: `tests/test_grid_transfer.py`, dev net seed 3), `scratch/smalln_density.py`.
Dev seen: test log (n = 2: refinement TV 5e-4, pullback TV 9e-13 at T = 40, MC mean/cov/orthants pass) and one dev
analysis (n = 2, seed 3: TV to the Gaussian 0.133, KL 0.065; to the gate-pattern mixture TV 0.035, KL 0.007).
Runs: n = 2: seeds 1000-1005, T = 40, dx 0.1 / 0.05; n = 3: seeds 1000-1003, T = 30, dx 0.2 / 0.14;
n = 4: seeds 1000-1001, T = 24, dx 0.5 / 0.4 (coarse; reported with its refinement TV).
S1 (accuracy). Refinement TV < 5e-3 (n = 2, 3) and < 3e-2 (n = 4); two-start pullback TV < 1e-6 (n = 2), < 1e-4 (n = 3).
S2 (non-Gaussian). TV(exact, moment-matched Gaussian) >= 0.05 on every net at n = 2 and 3; the gate-pattern mixture
has at most half the Gaussian's TV on every net at n = 2, 3.
S3 (falls with width). Median TV to the Gaussian over nets decreases from n = 2 to n = 3 to n = 4.
S4 (boundaries). The Gaussian approximation misstates the next layer's boundary density p_{z_1i}(0) by more than 20%
on at least half of the units at n = 2; the mixture is within 10% on at least half of the units at n = 2.

## Part D: response kernels, flip response and data collapse (written 15:50Z, before the Part D evaluation runs)

Code: `scratch/response_net.py` (CLVs by the Ginelli method, K = min(24, max(2, n/4)) directions; copies along CLV k
with relative size eps in {1e-7 (linear), 0.3/sqrt(n), 3/sqrt(n)}; flip of one gate decided by one noise SD; control
flip at 3-6 noise SDs; same noise and weights after the injection).
Dev seen: seed 3 at n = 16, 32, 64 (runs/dev_resp_*): e.g. n = 64: CLV exponents n lambda_k = -1.58, -2.61, -3.54,
...; at k = 8 the eps = 0.0375 response at t = 64 is 0.0029 against 0 (linear); noise-decided flip: cascade ~11 gate
differences in 2n layers, median gate lifetime 94 layers (censored at 2n).
Runs: n = 8, 16, 32, 64, 128, 256; seeds 1000-1003; P = 128 (n <= 64), 64 (n = 128, 256).
Response R(t) = r_t^2 / eps^2, r_t = |h'_t - h_t| / |h_t|; "median log response" = median over paths of log10 r_t^2.
D1 (linear collapse in t/n). For the linear copies at fixed k <= 4, the median log10 response curves of n = 64, 128,
256, plotted against t/n, differ pairwise by less than 0.5 (log10 units) for t/n in [0.1, 1] (seed-pooled).
D2 (linear rates are the CLV exponents). The slope of the median log response over t in [n/4, n], times n/2,
matches n lambda_k (the same run's QR exponent) within 25% for k <= 8, n >= 64.
D3 (nonlinear floor). For k >= 4 and eps = 3/sqrt(n), the mean response at t = n exceeds the linear one by more than
10x at every n >= 32; between the two finite amplitudes the mean response at t = n scales as eps^beta with beta in
[0.5, 1.2] (median over k >= 4).
D4 (flip cascade). The total gate cascade over 2n layers after one noise-decided flip, divided by n, is in
[0.08, 0.4] at every n >= 16 (median over nets of the per-net mean).
D5 (lifetimes collapse in t/n). The gate lifetime (last layer with a differing gate, censored at 2n) divided by n has
a distribution whose median varies by less than a factor 1.5 across n = 32..256.

### Amendment S1 (15:53Z, before any Part S evaluation run)
Cost: at n = 4 a grid with an 8-SD margin reaches 2e7-7e7 nodes (about 100 s per layer). n = 4 runs therefore use
`--marg 6 --skip-two-starts`, T = 16, dx 0.5 / 0.4 (one pullback per step; pullback convergence is taken from n = 2, 3).
`smalln_density.py` gained these two options (defaults unchanged); n = 2, 3 runs are as registered. S1 for n = 4 now
reads: refinement TV < 3e-2 (no two-start test).

### Amendment G1: the "5-8x" cloud excess from the finite-angle fixed point (15:54Z, before reading any evC value of
the stationary lane's fresh-net runs `runs/cloudA3_*`; the Part A table values at seeds 1000-1003 were seen)
Prediction (HEURISTIC model): angular cloud trace tr C_ang = n Q delta*(Q) (delta* = L0 exact-kernel fixed point,
Q = the net's median X at layer 0), spread harmonically, lambda_k = tr C_ang / ((H_n - 1) k) for k >= 2; against the
coordinator's sigma^2/(1 - exp(-2(k + 1/4)/n)) the median ratio over k = 2..8 is predicted as
2.1 Q delta*(Q) / (H_n - 1). Radial (k = 1) ratio predicted as 2.5 Q (2.23 Q^{1/3}/n + 1/(2n)).
G7: on every cloudA3 net with n >= 64, measured/predicted is in [0.6, 1.6] for the k = 2..8 ratio and in [0.4, 1.5]
for the radial ratio.

### Amendment S2 (16:11Z, before any Part S evaluation run)
A dev timing (seed 3) at n = 4, dx 0.4, margin 6 did not finish 6 layers in 280 s, so Amendment S1's n = 4 plan is
infeasible. n = 4 now: one net (seed 1000), T = 12, dx 0.6 / 0.5, `--marg 5 --skip-two-starts`. S1 for n = 4 reads:
refinement TV < 5e-2; S3's n = 4 "median" is that single net. n = 2, 3 unchanged.

### Amendment G2: prospective test of the cloud-excess prediction (16:12Z, before these runs exist)
The reviewer noted that the cloudA3 files predate Amendment G1. New nets: the stationary lane's `scratch/partA.py cloud`
(unchanged, sha in prereg_hashes.txt) at n = 64, 128, 256, seeds 3000-3003, P = 2048, outputs `runs/cloudG8_*`.
G8: same predictions and bands as G7 (k = 2..8 ratio measured/predicted in [0.6, 1.6], radial in [0.4, 1.5] on every
net), scored by `scratch/score_G7.py` logic.
