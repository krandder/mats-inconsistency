# Stationary regime, numerically: exact radius law, effective theory with noise, small-width densities, response kernels (31 Oct 2026)

Lane: stationary-numerics (bead pool-qxszj1.21), sibling of the stationary-regime lane
(`outbox/stationary-regime-2026-10-31.md`). Work directory: `work/stationary-numerics/`. Model S throughout: absolute
noise, h_l = relu(W_l h_{l-1}) + sigma xi_l, W_l iid N(0, 2/n); sigma = 1 without loss (sigma is a pure scale).
Labels: PROVEN (proof in `notes/proofs.md` or cited), NUMERICAL (converged computation, error check stated),
MEASURED (simulation with error bars), HEURISTIC (a model or closure that is tested, not derived).
Revised after a GPT-6 (gpt-6-astra) review that asked for major revision (13 findings, `review/review_gpt6.md`);
section 9 lists the changes.

## Summary for Kelvin

1. The radius law is solved exactly at every width (NUMERICAL). |h|^2 is an exact one-dimensional chain. I reduce it to
   a perpetuity r' = 1 + A r (PROVEN) and solve it with a spectral transfer operator; halving the grid changes tested
   quantiles by less than 5e-12. The corrections to the inverse-gamma limit have closed forms to order 1/n^2 (PROVEN as
   formal coefficients; their asymptotic accuracy is NUMERICAL): the median of |h|^2/(n^2 sigma^2) is
   0.5770780164 - 0.0632661929/n + 0.5192608/n^2 (error 6e-10 at n = 1024, 0.6% at n = 2), and the Kesten tail constant
   is C_n = 2n/5 + 13/75 + 529/(2250 n) + O(n^-2).
2. I can compute how fast near-floor layers become rare (the rate is HEURISTIC; the exact solver confirms it). At large
   width P(|h|^2 <= n sigma^2) ~ exp(-0.3107 n); the exact solver gives 0.3109 at n = 1024. These layers are exp(0.09 n)
   times more frequent than the inverse-gamma extrapolation exp(-0.4 n). The most likely route down is gradual, about one
   noise unit of |h|^2/n lost per layer.
3. The Roberts-Yaida finite-width recursions with a noise source have no fixed point for one input (PROVEN): the kernel
   grows by one noise unit per layer and the normalised four-point vertex diverges (the stationary mean of |h|^2 is
   infinite). The Gram matrix of a few noise replicas that share the weights does have a fixed point. That matrix is an
   exact Markov chain with the weights integrated out (PROVEN). Its 1/n expansion is the kernel + vertex recursion with
   noise, and it costs O(n) per layer.
4. The angle between two noise replicas sits at a fixed point of the noisy kernel recursion (HEURISTIC adiabatic model,
   tested): one layer's noise (about 1/Q, Q = |h|^2/(n sigma^2)) balances the ReLU kernel's finite-angle contraction,
   delta*(Q) ~ 2.23 Q^{-2/3}. The exact replica chain's mean angle is 1.12x this at n = 8, 1.04x at 64 and 1.00x at 1024.
5. With a 1/k spread over directions this matches the 5-8x on 21 of 22 nets (HEURISTIC). It predicts the cloud's
   eigenvalues (k = 2..8) over the Lyapunov formula as 2.1 Q delta*(Q)/(H_n - 1), about 4.7 Q^{1/3}/ln n. The model
   uses each net's measured radius and assumes a 1/k shape; the 4-replica participation ratio (0.79-0.99 of the 1/k
   value) is consistent with that shape but does not establish it, and I have not derived it. On 12 new nets registered
   before the runs (n = 64-256), measured/predicted is 0.86-1.41 on 11 and 0.56 on one large-radius net at n = 64, so
   the registered test misses. The stationary lane's
   10 fresh nets give 0.90-1.42. The radial eigenvalue is off by up to 3x (0.31-1.70).
6. The interaction order grows with width (MEASURED, 8 nets per width, one estimator). The mean interaction order of
   h_0 (a variance-weighted sum of total Sobol indices over layers) is 1.41, 1.94, 2.02, 2.75, 3.06, 3.87 at
   n = 8, 16, 32, 64, 128, 256; the log-log slope over 16-256 is 0.26 (90% CI 0.23-0.30). For the output gate pattern
   it goes from 3.4 to 31 (slope 0.51). The stationary lane's flatter numbers used a different estimator in different
   noise coordinates, so the two are not directly comparable.
7. The same fixed point accounts for most of the growth (HEURISTIC). One layer's noise opens an angle of about 1/Q, and
   the kernel contracts an angle at a rate proportional to the square root of the angle. A lone injection therefore
   decays slowly, while inside the full cloud (angle delta*, much larger) it is erased fast, so the single-layer total
   effects add up to more than the variance. For the direction of h_0 this model is within 8-15% of the networks at
   n = 32-256 and 16-28% below the exact replica chain. Its asymptote is 2.99 Q^{1/6} (about n^{1/6}).
8. The interactions come from gate switching, and from more gates than the noise-decided ones (MEASURED intervention).
   Freezing all gates makes the order exactly 1 (PROVEN; checked on 48 nets, a code check). In a post hoc analysis on h_0,
   letting only units within k noise SDs of the boundary switch (margin measured on a noiseless reference path at the
   typical radius) recovers about 0.2, 0.3, 0.6 and 1.0 of the excess for k = 0.5, 1, 2, 4. These are statements about
   the intervened models, not an allocation of the network's interactions, and the registered version on the direction
   observable missed its bands (section 3.2).
9. Response kernels at n >= 64 (MEASURED; covariant Lyapunov vectors from a finite window, not validated for
   convergence: they shift with the warm-up length). The tangent response along vector k decays like
   exp(-2 c (k + 1/4) t/n): across 256 per-net fits c has median 1.14 (10-90% range 1.01-1.30), and the
   second-order rank-conditioned exponents (k + 1/4)/n + (k^2 + k + 1/24)/n^2 predict it with median ratio 1.04 (10-90%
   range 0.94-1.15; single nets range more widely). A finite perturbation (relative size 0.3 or 3 over sqrt(n)) along a
   fast direction first falls like the tangent one, then levels off (median R ~ 0.004-0.01 near t = n, in units of its
   own size squared) and decays slowly; along vector 8 it is about 1e9 above the tangent median near t = 1.5 n.
10. One noise-decided gate flip causes 0.17-0.25 n further gate differences over the next 2n layers at every width.
   Its effect takes 1.0-1.3 n (lower quartile) and 1.4-1.9 n layers (median) to fall 100x, and more than 2n for 27-44%
   of paths at n >= 16 (20% at n = 8), so lifetimes scale with n. The exact cascade after resampling one layer's noise
   grows like log n (0.8 n at n = 8 to 5.4 n at n = 1024) and is 4-20% below the networks.
11. Exact small-width densities (NUMERICAL, no sampling; refinement diagnostics in total variation 5e-4 to 2e-3 at
   n = 2, 3; n = 4 is an exploratory 12-layer density with refinement 5e-2). At n = 2 the quenched stationary law is a
   mixture over the last layer's gate pattern. 13-69% of the mass comes from layers where
   every gate was off (pure noise centred on all of the next layer's boundaries). Total variation to the
   moment-matched Gaussian is 0.005-0.23, depending on the layer's radius. The gate-pattern Gaussian mixture is 1.8-6.5x
   closer in total variation and gets the density at the next layer's gate boundaries within 13% (the Gaussian is off
   by up to 38%). At n = 3 (4 nets) the total variation to the Gaussian is 0.09-0.19 and to the mixture 0.04-0.09; one
   n = 4 net on a coarse grid gives 0.20 and 0.09. Across all 11 nets the non-Gaussianity tracks the layer's radius
   rather than the width: nearly Gaussian at Q ~ 1, TV 0.09-0.23 from Q ~ 2 on.

What this means: in the stationary regime the cloud is a finite-angle object. One balance in the noisy kernel recursion
predicts its size, its excess over the Lyapunov picture (given an assumed 1/k shape) and most of its interaction
order, and the exact replica Gram chain computes every annealed replica quantity at O(n) cost. Open: the 1/k shape, the
radial variance at the fixed point (closures are 1.1-3.7x the exact value at n = 8-1024), and per-net versus annealed
statistics.

Pre-registered scorecard: 17 of 35 predictions hit, 17 missed, 1 unscorable (section 8, every miss listed).

## 1. Exact stationary radius law at finite n

1.1 Reduction (PROVEN, `notes/proofs.md` P-R1). Q = |h|^2/n satisfies Q = (S/n) r with S ~ chi2_n independent of r, and
r_l = 1 + A_l r_{l-1}, A = (2/n) chi2_M, M ~ Bin(n, 1/2): a perpetuity with B = 1, with the same tail constant
C_n = 1/E[A log A]. With u = log(r - 1) a layer is a fixed smooth map followed by a translation,
u' = log A + softplus(u).

1.2 Solver (NUMERICAL). Nystrom discretisation with the trapezoid rule on a uniform u-grid (`lib/radius_chain.py`);
stationary vector by inverse iteration, then positive power steps (they keep relative accuracy in the tails). Checks
(`tests/test_radius_chain.py`, 42 pass, at n = 2, 3, 4, 8, 64, 256): halving the step changes the 10/50/90/99%
quantiles of Q by at most 5e-12; widening the domain by at most 1e-13; Monte Carlo of the chain (4e8 steps at n = 8, 16,
32) agrees within 3.2 SE at seven quantiles and three left-tail probabilities; simulating r x S/n reproduces the direct
chain. A solve takes 1-400 s for n = 2-1024.

1.3 Finite-n corrections (closed forms PROVEN as the coefficients of the formal Kramers-Moyal series, P-R2; that the
series is asymptotic is NUMERICAL): with x = Q/n,

  P(Q <= x n) = e^{-2/(5x)} [ 1 - (65x - 47)/(375 x^2 n) - (198375 x^3 - 18825 x^2 + 63640 x - 6627)/(843750 x^4 n^2) + O(n^-3) ].

The two-term remainder halves when n doubles (ratio 2.00 at x = 0.58, 1, 2 between n = 512 and 1024).
`figs/radius_density.png` shows the exact densities from n = 2 and the collapse of n (F_n - F0) onto F1.

| n | median of Q/n | error of the 2-term formula | C_n | n^2 (C_n - 2n/5 - 13/75 - 529/(2250n)) | -ln P(Q <= 1)/n |
|---|---|---|---|---|---|
| 2 | 0.679204 | +3.9e-3 | 1.083677 | -0.029 | 0.442 |
| 8 | 0.578762 | +1.5e-3 | 3.408421 | 0.365 | 0.338 |
| 32 | 0.5756301 | +2.2e-5 | 12.981018 | 0.345 | 0.316 |
| 128 | 0.57661578 | +3.3e-7 | 51.375190 | 0.334 | 0.312 |
| 512 | 0.576956436 | +5.1e-9 | 204.973794 | 0.331 | 0.3110 |
| 1024 | 0.577016729 | +6.0e-10 | 409.773563 | 0.330 | 0.3109 |

(The formula is 0.5770780164 - 0.0632661929/n + 0.5192608/n^2; rounding these coefficients changes the error.)
Other limits (NUMERICAL): E|h|/(sigma n) = sqrt(2 pi/5) + 0.0238/n and E log(Q/n) = log(2/5) + gamma - 0.349/n to
leading order; E[A log A] = 5/(2n) - 13/(12 n^2) - 1/n^3 + ... (the n^-3 coefficient is -1.001 at n = 1024).

1.4 Kesten tail (NUMERICAL). x P(Q > x) at x = 1e8 n equals C_n = 1/E[A log A] within 1.5e-7 at every n; at finite x the
deficit is (0.2 + 0.4/n) n/x, the inverse-gamma correction.

1.5 Left tail (rate HEURISTIC, a Freidlin-Wentzell quasi-potential, P-R3; NUMERICAL agreement). P(r <= rho) ~
exp(-n J(rho)) with J(2) = 0.2142, J(3) = 0.1385, J(5) = 0.0816 and J(rho) = 0.4/rho + 0.0347/rho^2 + ... at large rho;
P(Q <= 1) ~ exp(-0.3107 n), P(Q <= 2) ~ exp(-0.1741 n). The exact rates at n = 1024 are 0.2143 and 0.3109
(`figs/radius_left_tail.png`). The optimal path loses about one unit of r per layer (1.6 near r = 2). The stationary
lane's conjecture "near-floor layers ~ exp(-0.4 n)" is refuted.

1.6 Against the stationary lane's data (MEASURED): its scalar-chain quantiles (4.1e6 correlated draws) agree within
0.4% at q10/q50 at every n, within 0.8% at q90 for n <= 64 and 1.6-2.4% for n = 128, 256 (5-6% at q99), consistent with
that sampler's correlation; its network quantiles (pooled layers) agree within 2% at q10/q50 for n <= 64.

## 2. Finite-width effective theory with noise

2.1 Literature (`notes/literature.md`). Roberts-Yaida (with Hanin) give for ReLU at criticality V^(l) = 5 (l - 1) K^2
(eq. 5.120; V/K^2 grows with the aspect ratio l/n) and the two-input angle recursion cos psi' = cos psi +
(1/pi)(sin psi - psi cos psi). Neither the book nor Hanin's finite-width correlation functions, Li-Nica-Roy or Hanin-Nica
include per-layer noise; Pretorius et al. (2018) show at mean-field level that additive noise leaves no critical fixed
point. The exact Gram recursion without noise is in the book (eq. 4.69-4.71). I found no finite-n corrections to the
inverse-gamma limit and no left-tail result for this perpetuity.

2.2 No fixed point for one input (PROVEN, P-G2). E[Q' | Q] = Q + 1, and Var(Q)/E(Q)^2 grows without bound; at
stationarity E Q = infinity. The stationary radius lives at aspect ratio l/n = O(1), where the 1/n expansion is not
uniform; section 1 replaces it by the exact chain.

2.3 Replica Gram chain (PROVEN, P-G1; `lib/gram_chain.py`, validated against full-network replicas in 22 KS/mean
checks, `logs/test_gram_chain.log`). R replicas share the weights and share noise according to a pattern S_l. Their
Gram matrix G_ab = h^(a).h^(b)/n is an exact Markov chain, G' = (1/n) sum_i v_i v_i^T with v_i = relu(z_i) + xi_i,
z_i ~ N(0, 2G), xi_i ~ N(0, S). The conditional mean is the kernel map with noise source,
E[G'_ab | G] = sqrt(G_aa G_bb) kappa(rho_ab) + S_ab; the conditional covariance is the (1/n) vertex source. Annealed
(weight-averaged) versions of quenched quantities are replica quantities: the cloud trace (2 replicas), tr C^2 (4), the
total Sobol effect of one layer (2 replicas sharing all noise but that layer's), births and the gate cascade (closed
forms in the final Gram matrix). Per-net ratios are not. Conditioning on one replica's own final radius breaks the
replica structure, so the comparisons below are unconditional unless stated.

2.4 The fixed point (L0: infinite-width noisy kernel at the current radius; HEURISTIC adiabatic model). Two replicas
with independent noise: rho' = Q kappa(rho)/(Q + 1), fixed point delta*(Q) = 1 - rho*: 0.369, 0.176, 0.0756, 0.0310 at
Q = 0.577 n for n = 16, 64, 256, 1024 (tending to 2.23 Q^{-2/3}; the exact kernel gives 2-27% less than the
delta^{3/2} truncation at these Q). Exact chain against L0, unconditional (L0 evaluated at each chain's own final
radius and averaged the same way; `results/posthoc_review_fixes.json`):

| n | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| E delta, exact / L0 | 1.123 | 1.088 | 1.060 | 1.044 | 1.023 | 1.023 | 1.018 | 0.997 |
| direction order D_u, exact / L0 (same averaging) | 1.22 | 1.17 | 1.22 | 1.28 | 1.19 | 1.17 | 1.16 | 0.93 |

The mean angle converges to the fixed point. The interaction order of the direction is 16-28% above L0 at n <= 512
(0.93 at 1024, where the estimate is noisier).

The radial asymmetry between replicas (the replica four-point vertex) is the weak point. The closure law
2.23 X^{1/3}/n + 1/(2n) is 3.7, 2.5, 2.0, 1.7, 1.4, 1.2, 1.14, 1.10 times the exact value at n = 8 ... 1024 (median-
radius bin of the final radius, so this comparison carries the selection caveat of 2.3). A post hoc AR(1) closure
(HEURISTIC, `scratch/vertex_fixed_point.py`) uses the exact one-step injection, which I measured at 4.07-4.15 delta*/n
per layer (astra's 4 delta/n), with an imposed restoring rate 1/(Q + 1) and frozen Q. It overshoots the exact value by
1.99, 1.70, 1.47, 1.30, 1.16, 1.10 at n = 16 ... 512. Both closures freeze Q and delta during a radial relaxation that
takes about Q/2 layers, so their large-n agreement is suggestive, not shown.

2.5 The 5-8x (HEURISTIC model; tests G7, G8; `results/partG7_scores.json`). The angular trace is n Q delta*(Q). With a
1/k spectrum over k = 2..n its eigenvalues are tr/((H_n - 1) k), and against the coordinator's
sigma^2/(1 - exp(-2(k + 1/4)/n)) the median ratio over k = 2..8 is 2.1 Q delta*(Q)/(H_n - 1) (2.1 = the median over
k = 2..8 of 2(k + 1/4)/k; nothing is fitted). Q is each net's measured median radius at layer 0; the 1/k shape is
assumed. Measured/predicted:
- G8, prospective: 12 new nets (seeds 3000-3003 at n = 64, 128, 256; the stationary lane's unchanged `partA.py cloud`),
  registered before the runs: 0.92, 1.08, 0.56, 0.86 (n = 64), 1.11, 1.41, 1.03, 1.20 (128), 1.01, 1.20, 0.88, 1.02
  (256). The 0.56 is a net with Q/n = 5.65 at layer 0, 9.8 times the stationary median. Radial: 0.38-1.19. G8 missed (one net on
  each clause).
- G7, on the stationary lane's fresh nets (seeds 2000-2003). Their files predate my registration; I state that I had not
  read their eigenvalues, which timestamps cannot verify. 0.90, 1.42, 0.95, 1.01 (n = 64), 1.11, 1.01, 0.95, 0.96 (128),
  1.12, 1.18 (256); 0.61-1.14 at n = 32 and 0.50-1.04 at n = 16. Radial 0.31-1.70. G7 missed on 2 of 10 nets (radial).
The ratio grows like Q^{1/3}/ln n and with the layer's radius, matching the stationary lane's observation that the
excess is larger on layers with larger radius.

2.6 The 1/k shape. Exact 4-replica annealed participation ratio of the angular cloud (all chains): 8.7, 11.8, 17.4,
24.2, 37.7, 50.9 at n = 16 ... 512 (bootstrap SE 3-5%), 0.79-0.99 of the 1/k value (H_n - 1)^2/(pi^2/6 - 1). A
participation ratio cannot establish a spectrum; it is consistent with 1/k and with the stationary lane's measured
slopes near -1. I have no derivation of the shape.

2.7 Interactions in the L0 model. A single layer's noise opens delta_0 = 1/(Q + 1); with the rest of the noise shared
it decays as rho' = (Q kappa(rho) + 1)/(Q + 1), i.e. delta_t ~ delta_0/(1 + (c/2) sqrt(delta_0) t)^2 with
c = 2 sqrt 2/(3 pi). The decay is algebraic, at a rate set by its own small angle. The cloud's angle delta* is much larger
than delta_0, so in the full variance each injection is contracted much faster. Hence the sum of single-layer total
effects exceeds the variance: D_u = sum_t delta_t/delta*, with asymptote 2 c^{-1/3} Q^{1/6} = 2.99 Q^{1/6}. L0 at
Q = 0.577 n: 1.43, 1.73, 2.14, 2.68, 3.37, 4.19, 5.16, 6.27 at n = 8 ... 1024. Networks (section 3.1, median over 8 nets):
1.72, 2.53, 2.27, 3.07, 3.09, 3.97 at n = 8 ... 256, i.e. 0.92-1.15x L0 at n >= 32 (1.20-1.46x at n = 8, 16). For the
output gate pattern L0 gives 6.8, 16.0, 36.4, 78.7 at n = 16, 64, 256, 1024; networks 7.6, 14.6, 30.6.

2.8 Births and cascade (exact replica chain; unconditional). Gate flips at the next layer after resampling one layer's
noise: 0.62-0.63 sqrt(n) at n = 8-1024 (L0 0.59; the stationary lane measured 0.48-0.64). Flips over the next 2n
layers: 0.82, 1.23, 1.70, 2.26, 2.90, 3.63, 4.43, 5.43 n at n = 8 ... 1024. The networks give 2.38-2.56 n (n = 64) and
3.78-4.52 n (n = 256), 4-20% above. For sqrt(n) << t << n the finite-angle law gives 3n/t flips per layer, but that window
needs n >> 1e4, so at reachable widths the cascade collapses neither in t/sqrt(n) nor in t/n
(`figs/flip_cascade_exact.png`).

## 3. Interaction order across width, and what drives it (the coordinator's two items)

3.1 Width trend (MEASURED, PREREG I1; `results/partI_scores.json`, `results/fits.json`, `figs/interaction_order.png`).
Per net, D_f = (J + 1) mean_s |f_s - f_s^(j_s)|^2 / mean_s |f_s - f'_s|^2 over P samples. Here j_s is uniform over the
J + 1 = 4n + 1 window layers (single-layer refresh), f' refreshes the whole window, and each sample has its own deep
past. Numerator and denominator are unbiased; the ratio is a consistent estimator (bias O(1/P), P = 1024-4096) of the
variance-weighted mean dimension E[V_Z D_Z]/E[V_Z] over deep pasts Z. Bias diagnostic (post hoc, `runs/I1small_*`):
rerunning 4 nets per width at a quarter of the sample size changes D_h by -0.05 +- 0.13 (n = 16), -0.27 +- 0.20 (64)
and +1.4 +- 0.6 (256, where one net jumps from 4.3 to 9.1 at P = 256; the 4-net median moves from 4.11 to 4.42). If the
bias scales as 1/P, at the full P it is below about 0.1 at n <= 64 and could reach 0.3-0.5 at n = 256. That does not
change the trend. Nets: the stationary lane's st.W_layer seeds 1000-1007. Medians over 8 nets (bootstrap SE):

| n | 8 | 16 | 32 | 64 | 128 | 256 | log-log slope 16-256 (90% CI) |
|---|---|---|---|---|---|---|---|
| h_0 | 1.41 (0.07) | 1.94 (0.11) | 2.02 (0.07) | 2.75 (0.15) | 3.06 (0.09) | 3.87 (0.14) | 0.26 (0.23-0.30) |
| direction | 1.72 (0.18) | 2.53 (0.21) | 2.27 (0.11) | 3.07 (0.23) | 3.09 (0.12) | 3.97 (0.32) | 0.17 (0.12-0.25) |
| output a | 1.51 (0.04) | 1.98 (0.16) | 2.22 (0.11) | 2.99 (0.18) | 3.31 (0.09) | 4.11 (0.16) | 0.27 (0.21-0.31) |
| output gates | 3.40 (0.34) | 7.61 (0.64) | 8.68 (0.49) | 14.6 (1.5) | 18.8 (0.9) | 30.6 (0.9) | 0.51 (0.49-0.59) |

The stationary lane's Part I (2.5, 2.1, 3.1 for h_0 at n = 16, 64, 256) refreshed each layer with probability eps in
causal radial/angular coordinates. In those coordinates "keeping the other layers fixed" means something different
(the noise basis of later layers follows the state), so the two decompositions are of different functions and the
numbers are not directly comparable. In my coordinates G(eps)/eps is a lower bound on D (P-I1), but I did not run that
estimator on the same nets, so I do not attribute the difference to it.

3.2 Causal interventions (MEASURED; registered as I2 on the direction observable; `figs/interaction_causal.png`). All
variants share every noise with the network. "Frozen": every gate is set by the noiseless continuation of the deep past,
so h_0 is affine in the window noise and D_h = 1 exactly (PROVEN, P-I2); measured on all 48 nets within 3 SE (max |z|
2.1). This is a code check, since freezing all gates removes every nonlinearity at once. "free_k" lets a unit switch
only where the noiseless reference's normalised margin |z_i|/rms(z) is below k/sqrt(0.577 n). That is a proxy for
"within k noise SDs at the typical radius", not the unit's actual margin. "recent" lets only the last n/4 or n layers
switch. "clamp" fixes the radius after every layer, which itself adds a nonlinearity.
Registered (direction observable; its frozen baseline is not 1 because normalising h_0 is nonlinear): I2a hit; I2b,
I2c, I2d missed (E(free_4) at n = 256 not below n = 32; recent_q shares 1.5-3.1 instead of 0.3-0.9; clamp/true 0.68 at
n = 64). Post hoc, on h_0 (frozen baseline exactly 1), share of the excess D_h - 1 recovered, median over 4 nets
(2 at n = 256); per-net SEs on D are 0.1-0.5, so differences across n are not resolved:

| units allowed to switch | n = 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| reference margin below 0.5 SD (3-9% of units) | 0.16 | 0.18 | 0.13 | 0.28 |
| below 1 SD (7-18%) | 0.32 | 0.27 | 0.31 | 0.44 |
| below 2 SD (13-35%) | 0.64 | 0.59 | 0.50 | 0.95 |
| below 4 SD (26-64%) | 0.92 | 1.05 | 0.84 | 1.11 |
| only layers of age < n/4 | 0.90 | 1.11 | 0.86 | 0.87 |
| only layers of age < n | 1.12 | 1.07 | 0.97 | 1.22 |
| radius clamped (all gates free) | 1.47 | 0.74 | 0.81 | 0.86 |

Within these interventions, the excess order of h_0 is reproduced when units within about 4 reference-noise SDs may
switch in the last n/4 to n layers, and is largely kept when the radius is clamped. That fits the finite-angle picture:
two noise realisations of one net differ by an angle of order Q^{-1/3}, which moves preactivations by about 2 Q^{1/6}
noise SDs (3.4 at n = 32, 4.8 at n = 256). These are statements about the intervened models, not an allocation of the
original network's interactions.

## 4. Exact quenched densities at n = 2, 3, 4 (NUMERICAL; PREREG S; `results/partS_scores.json`)

Method (`lib/grid_transfer.py`): for one fixed net, the density of h_l is pushed through x -> relu(W_l x) cell by cell,
deposited on a new grid with linear weights and convolved with the unit Gaussian, separably; no sampling. The box
follows the mass. Diagnostics (`tests/test_grid_transfer.py`, dev net): total variation between two grid steps 5e-4
(n = 2, dx 0.1 vs 0.05) and 2.4e-3 (n = 3, 0.2 vs 0.14), a refinement diagnostic rather than a certified error bound;
the pullback forgets its start (TV between two initial laws 9e-13 after 40 layers at n = 2, 7e-10 after 30 at n = 3); a
4e5-path Monte Carlo of the same net matches the mean, covariance and orthant masses within 4 SE. The analysis at
layer 0 compares the exact density with (i) the moment-matched Gaussian and (ii) the Gaussian mixture over the gate
pattern of layer 0, with each component's exact mean and covariance plus the noise. Boundary densities of the next
layer's preactivations are computed exactly from the last layer's atoms, p_i(0) = sum_j m_j phi(w_i.A_j/|w_i|)/|w_i|
(`scratch/boundary_density_exact.py`, post hoc; identical to 3 digits at grid steps 0.05 and 0.1 for n = 2).

n = 2, six nets (`figs/smalln_density_n2.png`, `figs/smalln_summary.png`):

| net | mean Q | TV to Gaussian | TV to mixture | KL Gaussian / mixture | all off / one on / both on at layer 0 | gate entropy layer 1 (bits) |
|---|---|---|---|---|---|---|
| 1000 | 5.10 | 0.146 | 0.063 | 0.077 / 0.017 | 0.30 / 0.04 / 0.66 | 1.35 |
| 1001 | 1.05 | 0.005 | 0.001 | 0.0001 / 0.0000 | 0.54 / 0.19 / 0.27 | 1.78 |
| 1002 | 4.81 | 0.193 | 0.083 | 0.155 / 0.030 | 0.13 / 0.66 / 0.21 | 1.35 |
| 1003 | 3.72 | 0.206 | 0.114 | 0.156 / 0.057 | 0.29 / 0.09 / 0.62 | 1.21 |
| 1004 | 1.25 | 0.040 | 0.008 | 0.011 / 0.001 | 0.63 / 0.32 / 0.06 | 1.75 |
| 1005 | 2.78 | 0.234 | 0.036 | 0.198 / 0.010 | 0.69 / 0.03 / 0.29 | 1.92 |

Shape: before the last noise, all mass lies in the closed positive quadrant: its interior (both units on), the two
half-axes (one on) and the origin (both off). The noise turns these into a smeared 2-D lobe, two ridges and a Gaussian
blob at the origin. When the layer's radius is small (Q ~ 1) the blob dominates and the law is Gaussian to 1%; at
Q ~ 3-5 it is skewed (|skewness| up to 1.7, excess kurtosis up to 4.1), with ridges along the half-axes (the noise
smooths the edges, so there are no literal kinks). The all-off component is centred exactly on every boundary of the
next layer, which is why the exact law has more mass within one noise SD of those boundaries than its Gaussian
approximation: 0.37 against 0.27, 0.49 against 0.37 and 0.56 against 0.39 on the three worst units. Exact boundary
densities: the Gaussian is off by 0.1-37% (more than 20% on 4 of 12 units), the mixture by 0-13% (within 10% on 10 of 12).

n = 3 (four nets; grid steps 0.2 and 0.14, refinement TV 2.0-2.3e-3, two-start TV 2e-8 to 3e-7 after 30 layers) and
n = 4 (one net; Amendment S2 grid steps 0.6 and 0.5, refinement TV 0.051; only 12 layers and no two-start test, so
this is an exploratory finite-depth density, not a validated stationary one):

| n, net | mean Q | TV to Gaussian | TV to mixture | KL Gaussian / mixture | all off / all on at layer 0 | patterns above 1% | gate entropy layer 1 (bits) |
|---|---|---|---|---|---|---|---|
| 3, 1000 | 2.04 | 0.112 | 0.046 | 0.060 / 0.012 | 0.065 / 0.042 | 8 of 8 | 2.51 |
| 3, 1001 | 7.90 | 0.194 | 0.085 | 0.159 / 0.033 | 0.038 / 0.004 | 7 | 2.03 |
| 3, 1002 | 2.67 | 0.122 | 0.044 | 0.064 / 0.011 | 0.093 / 0.220 | 7 | 2.26 |
| 3, 1003 | 2.36 | 0.086 | 0.049 | 0.040 / 0.012 | 0.081 / 0.059 | 6 | 1.98 |
| 4, 1000 | 4.23 | 0.198 | 0.088 | 0.176 / 0.045 | 0.002 / 0.000 | 10 of 16 | 3.31 |

The all-off component shrinks with width (13-69% at n = 2, 4-9% at n = 3, 0.2% on the n = 4 net) and the mass spreads
over more gate patterns, but the distance to the Gaussian does not fall with n on these nets: it follows the layer's
radius (`figs/smalln_summary.png`, right). The mixture is 1.8-2.6x closer than the Gaussian at n = 3 and 2.2x at n = 4,
and gets the exact next-layer boundary densities within 7% at n = 3 (the Gaussian is off by up to 38%; grid steps 0.2 and
0.14 give the same errors to three digits on net 1000) and within 10% at n = 4 (Gaussian up to 14%; grid steps 0.6 and
0.5 agree to 0.5%). n = 6 is out of reach for dense grids (the n = 4 grids already hold 1e7-1e8 nodes).

## 5. Response kernels (MEASURED, PREREG D; `results/partD_scores.json`, `results/tangent_fits.json`)

Setup: 4 nets per width (n = 8-256), 64-128 stationary paths per net. Covariant Lyapunov vectors (CLV) at the injection
layer by the Ginelli method (K = min(24, n/4)). Copies are perturbed along CLV k with relative size eps = 1e-7,
0.3/sqrt(n) or 3/sqrt(n) and propagated with the same noise and weights. Response R(t) = |h'_t - h_t|^2/(|h_t|^2 eps^2).
Validity: rank loss (layers with fewer active units than directions) hit 63-79% of paths at n = 8, 46-70% at n = 16 and
3-24% at n = 32; there the regularised Ginelli step gives no covariant directions (and NaN means at n = 8, 16). It is
0-2% at n >= 64, so CLV results are reported for n >= 64 only. The eps = 1e-7 copies are finite differences. For fast
directions they reach a double-precision floor near R ~ 1e-15 (the reviewer found 1.3e-15 where tangent propagation
gives 7e-26), so the linear kernel comes from exact tangent propagation (`scratch/tangent_response.py`, post hoc).

5.1 Linear kernel (tangent propagation, post hoc; `figs/response_tangent.png`). The CLVs here are finite-window,
survival-conditioned directions: at finite n every infinite-time tangent exponent is -infinity (an all-off layer kills
all tangents), and the directions depend on the 2n-layer QR warm-up (the reviewer found that a 4n warm-up moves one
net's fitted c at k = 4 from 0.69 to 0.75). I did not validate them against a second horizon, and rank loss was
checked only in the window after the injection, so these are unvalidated finite-window directions. The median-over-paths ln R(t) is close to linear in (k + 1/4) t/n for all
k = 1-24 and n = 64, 128, 256, with slope -2c. Fitting each net's median curve over t in [n/8, n] (256 fits):
c has median 1.14, 10-90% range 1.01-1.30 and extremes 0.07-2.1. The second-order rank-conditioned exponents of astra's
round 1, (k + 1/4)/n + (k^2 + k + 1/24)/n^2, predict c_k = 1 + (k^2 + k + 1/24)/((k + 1/4) n). Fitted/predicted has
median 1.04, 10-90% range 0.94-1.15 and extremes 0.07-2.1; averaged over nets it is 0.79-1.22. Beyond (k + 1/4) t/n ~ 30
the CLVs' double-precision contamination by faster directions sets a floor at ln R ~ -73. Registered D1 (collapse of the
finite-difference curves within 0.5 decades) missed (0.57-0.68 decades for k = 2-4, computed from the mean of per-net
medians; per-path pooling is not possible from the saved outputs). Registered D2 (slopes within 25% of the QR exponents)
missed: per run, 16 of 72 (net, k <= 8) pairs fail, ratios 0.01-1.71 (k = 5, 7 were not simulated).

5.2 Finite perturbations (empirical, n >= 64; `figs/response_collapse.png`, left). Populations: the same nets; the same
64 paths at n = 128 and 256; at n = 64 the tangent runs used 64 paths with other noise than the 128 finite-size paths.
"Near t = n" means the sampled layer closest to n, t/n = 0.95, 0.98, 1.05 at n = 64, 128, 256 (and 1.56, 1.51, 1.45 for
"near 1.5n"). Along fast directions the response first falls like the tangent one, then levels off. Near t = n the
median over nets of the per-net median R over paths, over k >= 4, is 0.004 for eps = 0.3/sqrt(n) and 0.010 for
3/sqrt(n) (the means over paths are 0.039 and 0.063: the distribution is skewed). This is 1.4-23 decades (median 5.7)
above the tangent median for the same net and k. For CLV 8 near t = 1.5n the gap is 8.6-9.6 decades. (The figure shows
the mean over nets of the per-net log-medians.) Between the two sizes the mean R scales as eps^beta with median beta = 0.12
(individual values -0.32 to 0.62), so on this grid the retained part of r_t^2 is roughly proportional to eps^2. Whether
the plateau is the radial mode is not established.

5.3 Flip response (`figs/response_collapse.png`, right). One gate is flipped at the first layer after the injection: a
unit whose margin is below one noise SD (the registered population). When no unit qualified, the code fell back to the
nearest-margin unit; that happened on 1-19% of paths (19% at n = 16, fallback margins up to 9 SDs; 0-4% at n >= 32), and
those paths are excluded here. Further gate differences over the next 2n layers: 0.25, 0.17, 0.24, 0.20, 0.23, 0.19 n
(median over nets, n = 8 ... 256). The squared difference falls 100x after 1.0-1.3n (lower quartile) and 1.4-1.9n
(median); the upper quartile lies beyond the 2n window for every n >= 16 (27-44% of paths are censored at 2n; 20% at
n = 8, upper quartile 1.9n). Lifetimes scale with n, consistent with decay at the top Lyapunov rate of about 2.5/n. A control flip 3-6 SDs from the boundary (a 40-140x larger squared
perturbation) causes 1.1-1.7n differences.

## 6. Where this leaves the structure

- Radius: exact at every n (section 1). The effective theory never needs a radius closure.
- Angle: a fixed point of the noisy kernel recursion, delta* ~ 2.23 Q^{-2/3}; the exact replica chain is within 0-12% of it
  at n = 8-1024, converging.
- Cloud size and its excess over the Lyapunov formula: delta* spread as 1/k (2.5); consistent with 21 of 22 nets, with
  the 1/k shape taken from data.
- Non-additivity: the same fixed point gives the direction's interaction order within 8-28%; the order grows slowly
  (n^0.26 for h_0 at n = 16-256, n^{1/6} asymptotically in the model); interventions point to gate switching within
  about 4 noise SDs in the recent n layers.
- Small width: the stationary law is a gate-pattern mixture, Gaussian only when the radius is small.
- Linear response along Lyapunov directions: the rank-conditioned exponents describe it within about 20%; finite
  perturbations keep a slowly decaying part that the linear theory misses entirely.
- Open: (a) a derivation of the 1/k angular spectrum; (b) the radial variance at the fixed point (closures 1.1-3.7x the
  exact value); (c) per-net versus annealed statistics (the annealed D_h of the replica chain is too noisy to use:
  6.7 +- 3.9 at n = 256); (d) whether the response plateau is the radial mode.

## 7. Reproduction

Every number above comes from a script in `scratch/` writing to `runs/` or `results/`: radius (`radius_corrections.py`,
`radius_mc.py`, `radius_ldp.py`, `km_*.py`, `score_R.py`), replica chain (`gram_experiments.py`, `l0_theory.py`,
`score_G.py`, `score_G7.py`, `vertex_fixed_point.py`, `posthoc_review_fixes.py`), interaction order (`interact_net.py`,
`score_I.py`, `fits.py`), densities (`smalln_density.py`, `boundary_density_exact.py`, `score_S.py`,
`fig_smalln.py`), responses (`response_net.py`, `tangent_response.py`, `score_D.py`, `score_T.py`, `fig_response.py`),
review re-analyses (`posthoc_review_fixes.py`, `posthoc_review2.py`). Python: `/home/agent/.venvs/arc-noisy/bin/python`.

## 8. Pre-registration scorecard

PREREG and hashes: `work/stationary-numerics/PREREG.md`, `prereg_hashes.txt`. Every part was hashed before its batch;
header-time typos were corrected by rehashing before any data, each logged. Deviations: one bug fix in
`response_net.py` (all-off layers made the rank-loss regulariser divide by zero; only the crashed runs were rerun;
registered version kept as `scratch/response_net_as_registered.py`); the nearest-margin fallback in the flip protocol
(5.3); Amendments S1, S2 (cheaper n = 4 grids, before any Part S run), G1 (G7) and G2 (G8).

| id | prediction (short) | result |
|---|---|---|
| R1 | KM series is asymptotic (remainders) | hit |
| R2 | two-term median formula, 1e-7 at n >= 256 | hit |
| R3 | Kesten constant of the solved law | hit |
| R4 | C_n series and E[A log A] coefficient | hit |
| R5 | left-tail rates J(2), K(1) at n = 1024; not 0.4 | hit |
| R6 | two-term median off by > 1% at n = 2 | miss (0.6%) |
| R7 | moment corrections are O(1/n) | hit |
| R8a | direct Monte Carlo within 4 SE | hit |
| R8b | stationary lane's chain quantiles within 1% (q99 5%) | miss (q90 1.6-2.4% at n = 128, 256; q99 5-6%) |
| R8c | stationary lane's network quantiles | hit |
| G2 | mean replica angle within 20% of L0, converging | hit (binned by final radius; unconditional re-analysis agrees) |
| G3 | radial CV^2 within 1.5x of the closure at n >= 64 | miss (0.60 at n = 64) |
| G4a | direction order within 30% of L0 (median bin) | hit (binned; unconditional 16-28%) |
| G4b | annealed D_h, D_u grow 1.3x from n = 64 to 1024 | miss (annealed D_h too noisy) |
| G4c | output-gate order slope 0.40-0.75 | hit |
| G4d | annealed D within 1.5x of the stationary lane | miss (annealed D_h 6.7 at n = 256) |
| G5 | births within 5% of L0; cascade within 25-30% | miss (births up to 10.2% above L0, n = 16) |
| G6 | angular participation ratio within 2x of 1/k | hit |
| G7 | cloud excess on the stationary lane's fresh nets | miss (radial clause, 2 of 10 nets) |
| G8 | same, prospective, 12 new nets | miss (one net per clause) |
| I1a | interaction order grows with n | hit |
| I1b | networks within 0.6-1.4x of L0 | hit |
| I2a | frozen gates give D_h = 1 | hit |
| I2b | margin shares (direction observable) | miss |
| I2c | age shares (direction observable) | miss |
| I2d | clamp keeps 70% (direction observable) | miss (0.68 at n = 64) |
| S1 | grid accuracy | miss (n = 2, 3 pass; the n = 4 refinement TV is 0.051 against 0.05) |
| S2 | TV to Gaussian >= 0.05 on every net, mixture <= half | miss (two n = 2 nets at 0.005, 0.040; mixture ratios 0.55, 0.57 on two nets) |
| S3 | TV to Gaussian falls from n = 2 to 3 to 4 | miss (n = 4 net 0.198) |
| S4 | Gaussian boundary density off > 20% on half the units | miss (4 of 12 with exact boundary densities; the mixture clause holds, 10 of 12) |
| D1 | finite-difference linear curves collapse within 0.5 decades | unscorable as registered (path pooling not computable); the mean of per-net medians misses (0.57-0.68 decades) |
| D2 | slopes within 25% of QR exponents | miss (per run: 16 of 72 pairs outside; k = 5, 7 not simulated) |
| D3 | nonlinear floor > 10x; amplitude exponent 0.5-1.2 | miss (exponent 0.12) |
| D4 | flip cascade 0.08-0.4 n | hit (also on the registered population only) |
| D5 | flip lifetimes scale with n (median within 1.5x) | hit (also on the registered population only) |

Totals: 17 hits, 17 misses and 1 unscorable (D1: its registered path-pooled statistic cannot be computed from the
saved outputs) of 35. Most misses are bands set too tight (R6, R8b, G5, D2, S1) or registered on an observable or estimator that turned out unsuitable (G4b, G4d, I2b-d, D1). The substantive misses are the radial variance (G3), the radial and one angular net in the cloud-excess tests (G7, G8), the plateau exponent (D3) and the small-width non-Gaussianity claims (S2-S4).

## 9. Changes after the GPT-6 review

The review (`review/review_gpt6.md`) found the radius mathematics sound, reproduced the section 3 medians and shares,
the G7 ratios and the radius table, and confirmed that the batches followed their registrations. Its findings and what
changed:
1. Binning replica statistics by one replica's own final radius breaks the replica estimands: all replica-chain
   comparisons are now unconditional (2.4, 2.6), with the remaining binned closure comparison labelled; "establishes
   1/k" became "consistent with 1/k".
2. The interaction-order estimator is a consistent ratio estimator of a variance-weighted mean dimension, not unbiased:
   reworded (3.1).
3. CLVs are destroyed by rank loss at n = 8, 16 (and partly 32): CLV results are restricted to n >= 64, with the
   rank-loss fractions stated (5).
4. The finite-difference "linear" responses hit a double-precision floor and the registered collapse tests failed: the
   linear kernel was redone by tangent propagation (5.1) and D1, D2 are reported as missed.
5. The response plateau was overinterpreted: kept as an empirical observation with its spread of exponents; the
   universal and radial readings were withdrawn (5.2).
6. The lower-bound argument did not explain the sibling lane's numbers, which use different noise coordinates: removed
   (3.1).
7. The causal variants were overinterpreted and the registered misses were not shown: the margin proxy is defined,
   I2b-d are reported as missed, and the conclusions are limited to the intervened models (3.2).
8. The flip protocol's fallback to the nearest-margin unit was undisclosed: disclosed, and eligible paths are scored
   separately (5.3).
9. G7 was not prospective: reframed, and a prospective test on 12 new nets was registered and run (G8, 2.5).
10. The vertex closure imposes its dynamics and a ratio was reversed: labelled HEURISTIC, wording corrected (2.4, 6).
11. Agreement claims were stronger than the ranges: every agreement now states its range; the annealed D_h is marked
   unusable (2.7, 2.8, 6).
12. The printed median coefficients were rounded: full coefficients printed and the tested scope stated (1.2, 1.3).
13. The summary dropped evidence labels: restored.

A second-round review (`review/review_gpt6_r2.md`, 17:20Z) confirmed findings 6, 9, 10 and 12 as resolved and
reproduced the unconditional replica ratios and the G8 failures. It asked for nine more corrections, now applied:
per-net spread of the tangent fits and the finite-window nature of the CLVs (5.1, Summary 9); medians compared with
medians, n >= 64 only, for the plateau (5.2, revised figure); censored lifetimes reported as beyond 2n (5.3, Summary
10); registered aggregations stated for D1, D2 per run (16 of 72 fail), D4 and D5 rescored on the registered
population (both still hit); "refinement diagnostics" and an exploratory label for the 12-layer n = 4 density (4);
exact boundary densities from the last layer's atoms instead of a narrow histogram, with S4 rescored (same verdict);
a sample-size bias diagnostic for the interaction order (3.1); and corrected numbers for G5 (10.2%), S2 (two and two
failures) and the G8 outlier (Q/n = 5.65).

A third round (`review/review_gpt6_r3.md`, 17:35Z) recomputed the tangent median c (1.137), the per-run D2 failures
(16 of 72) and the exact S4 counts, and found round-2 items 1 and 6-9 addressed. It asked to: label the CLVs as
unvalidated finite-window directions (5.1, Summary 9); state the path populations, aggregations and the actual sampled
times of the plateau comparison (5.2); stop the lifetime curves at the 2n window with the censored mass kept (revised
figure) and qualify Summary 10 by width; and count D1 as unscorable rather than missed (17 hits, 17 misses,
1 unscorable). All applied.

## 10. Files

All under `work/stationary-numerics/`: `lib/` (radius_chain.py, gram_chain.py, grid_transfer.py, plotstyle.py),
`tests/` (three test files; logs in `logs/`), `scratch/` (batch, scoring and figure scripts), `runs/` (one JSON per run),
`results/` (scores and tables), `figs/` (PNG), `notes/` (questions.md, literature.md, proofs.md), `review/`.
