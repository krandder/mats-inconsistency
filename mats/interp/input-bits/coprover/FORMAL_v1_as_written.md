# Input-bits lane: theory v1 (Claude, written 17:55-18:07Z on 31 Oct 2026, before reading gpt-6-astra's round-1 answer)

Status: v1, frozen (sha in `formal_v1.sha256`; a copy will be kept as `coprover/FORMAL_v1_as_written.md`). Labels: PROVEN,
HEURISTIC, CONJECTURE; DEV = a number from the dev nets (seeds 9001, 9002; `logs/dev_probe*.log`), used to calibrate,
never scored. Provenance: everything here is [C] (Claude); astra's round 1 (`coprover/prompt_r1.md`) runs in parallel.

## 0. Setting
Width n, depth L, W_l iid N(0, 2/n), no biases, x ~ N(0, I_n). h_l = relu(W_l h_{l-1}) + sigma xi_l (l = 1..L-1), a =
relu(W_L h_{L-1}). For sigma = 0: a(x) = |x| g(u), u = x/|x|; A(u) = |g(u)| (amplitude), d(u) = g/|g| (direction),
m_n = E|x| (n - m_n^2 = 1/2 - 1/(8n) + O(n^-2)). kappa(c) = (sqrt(1 - c^2) + (pi - arccos c) c)/pi; rho_l = kappa^(l)(0),
theta_l = arccos rho_l, delta_l = 1 - rho_l. J_2(t) = 3 sin t cos t + (pi - t)(1 + 2 cos^2 t) (Cho-Saul degree-2 arc-cosine:
E relu(X)^2 relu(Y)^2 = J_2(t)/(2 pi) for unit-variance X, Y at angle t).

## 1. Exact identities (PROVEN)
1.1 Three-term split. For a fixed noiseless net, with expectations over u uniform on the sphere:
  tr Var_x(a) = V_rad + V_amp + V_dir,  V_rad = (n - m_n^2) E A^2,  V_amp = m_n^2 Var A,  V_dir = m_n^2 [(E A)^2 - |E(A d)|^2],
all three >= 0 (Cauchy-Schwarz for V_dir). Proof: E|a|^2 = n E A^2 and E a = m_n E(A d), |x| independent of u.
Equivalently V_dir = m_n^2 E_{u,u'}[A A' (1 - d.d')] over independent u, u'. Each term is a two-input (two-replica)
quantity, so its weight average is computed exactly by the replica Gram chain started from the input Gram matrix.
1.2 Gate disagreements. Given layer-(l-1) states h, h' of two inputs with cosine rho, E_{W_l} #{i : sign(w_i.h) != sign(w_i.h')}
= n arccos(rho)/pi exactly (a fresh W_l). With sigma = 0 there are no births: every gate is a function of x.
1.3 Antithetic identity: for a pair (x, -x) the pair mean is f_even(x) = (f(x) + f(-x))/2, so its variance gain over two
independent samples is 1/(2 s_even), s_even = Var f_even / Var f (sum over Hermite degrees 0 mod 2 of the variance).
1.4 Stein: the Hermite degree-1 share W_1 = sum_c |E[a_c x]|^2 / V and the degree-2 share W_2 = sum_c |E[a_c (x x^T - I)]|_F^2/(2V);
the first-order (additive) Sobol share S_1 >= W_1 + (diagonal part of W_2); the mean dimension D = sum_j T_j = sum_S |S| V_S/V.
1.5 Owen scrambling: the scrambled-net variance is a sum over ANOVA components with per-component gain coefficients, so
1/G_RQMC = sum_k rho_k s_k with s_k the order-k share and rho_k in [0, Gamma_max] the order-k average gain coefficient
at (N, d). (Standard; for scipy's LMS + digital shift scrambling I assume the same structure. Calibration: section 4.)

## 2. Leading-order theory of the input regime (HEURISTIC; large n, sigma = 0)
2.1 Directions. Two independent inputs start at cosine 0 +- n^-1/2; the infinite-width kernel gives rho_l = kappa^(l)(0):
delta_l = 0.682, 0.319, 0.166, 0.0705, 0.0253, 0.0079, 0.0023 at l = 1, 4, 8, 16, 32, 64, 128 (asymptote 44.4/l^2 is
reached slowly; delta_l^-1/2 ~ c l/2 + 1.5, c = 2 sqrt 2/(3 pi)). Prediction: V_dir ~ m_n^2 (E A)^2 delta_L, i.e. about
n delta_L in units where E|a|^2 = n. Finite-width correction: DEV nets give 1.6x at 256 x 32 (one net) and 0.5x at
64 x 32: expect the kernel formula to be off by up to 2x at L >= n/4, and the exact Gram chain to fix it.
2.2 Amplitude. For two unit directions at angle t, one layer's squared gains g = |relu(Wq)|^2 obey
Var(log g - log g') = (2/n)(6 - (2/pi) J_2(t)) + O(n^-2) (from E relu^4 = 3/2 and the degree-2 arc-cosine kernel).
Summing independent layers along the kernel angles: Var_u log A = Lambda(L)/(4n), Lambda(L) = sum_{l=1}^{L} [6 - (2/pi)
J_2(theta_{l-1})] = 5.0, 14.8, 21.9, 28.8, 34.2, 37.9, 40.1 at L = 1, 4, 8, 16, 32, 64, 128; Lambda(inf) = 42.5. The tail
Lambda(inf) - Lambda(r) ~ 355/r: the amplitude keeps collecting input variance over O(100) layers whatever n is.
So V_amp ~ m_n^2 (E A)^2 Lambda(L)/(4n) = O(1) (n-independent at fixed L), while V_dir ~ n delta_L.
2.3 What carries the input (the stationary lane's open item). Crossover V_amp = V_dir where 4 n delta_L = Lambda(L):
L_x ~ 7, 13, 24, 45 at n = 64, 128, 256, 1024 (kernel values). ARC's 256 x 32 sits just past it. Predictions:
  - 256 x 32: s_amp in [0.40, 0.75], s_dir in [0.20, 0.55], s_rad <= 0.06 (DEV: 0.55/0.43/0.02 and 0.59/0.38/0.04).
    So the "radius" that carries the input is the amplitude |h_l| (input-dependent gains), not |x|; the direction still
    carries a large part. The input radius |x| carries 2-5%.
  - Fixed L, n -> inf: s_dir -> 1 (direction, high order). Fixed L/n = alpha, n -> inf: s_dir ~ 4.2/(alpha^2 Lambda n) -> 0,
    the amplitude and radius carry everything (DEV n = 64: s_dir 0.10 at L = n/2, 0.003 at L = n).
2.4 Bit budget (sigma = 0, no births). Hamming_l = n E arccos(rho_{l-1})/pi with the kernel: n theta_{l-1}/pi = 0.5n,
0.397n, ... ~ 3n/l at large l (algebraic decay, set by the finite-angle kernel contraction c delta^{3/2} per layer, not by
the Lyapunov gaps k/n, which act on infinitesimal differences only). Sum of marginal gate entropies: 2.00 x Hamming at
l = 1 (all p_i = 1/2), tending to 2.31 x Hamming at depth (Gaussian-cap model: p_i = Phi(m_i/s) with m_i spread on a
scale >> s; ratio int H(Phi(t)) dt / int 2 Phi(t) Phi(-t) dt = 2.606/1.128 = 2.31). DEV: 2.26-2.41 at l >= 8.
Prediction for layer 31 of 256 x 32: Hamming ~ 19 (kernel; nets 1.0-1.2x), marginal entropy ~ 2.3 x 19-23 = 44-53 bits
(the stationary lane's 38 on one net is inside the net-to-net scatter I expect, about +-25%).
2.5 How the bits feed the amplitude: layer l adds (1/(4n))(6 - (2/pi) J_2(theta_{l-1})) ~ (1/n) theta_{l-1}^2 ~
pi^2 Ham_l^2 / n^3 to Var log A for small angles: the amplitude is a quadratic read-out of the per-layer input bits.
2.6 Noise (sigma = 1e-3 below T_mem/2): births ~ n arccos(Q/(Q+1))/pi per layer with Q = |h|^2/(n sigma^2) ~ 1e6 at
256 x 32, i.e. ~0.1-0.3 per layer: negligible against 19+ input bits; the split and bit budget change by < 1%.

## 3. Interaction structure over input coordinates (HEURISTIC)
3.1 Mean dimension. One coordinate refresh moves u by |du|^2 ~ 2/n, carried to depth with a mean-square factor Gamma_L
(1 at infinite width since kappa'(1) = 1; < 1 at finite L/sqrt(n) through the finite-angle contraction, roughly
(1 + 0.15 L/sqrt(n))^-2). Direction part: D_dir ~ Gamma_L/delta_L (~ L^2/44 at infinite width). Amplitude part: each
layer's log-gain has tangent gradient |grad|^2 ~ 8, so sum_j E(d log A)^2 ~ 4 L Gamma/n and D_amp ~ 8 L Gamma / Lambda(L).
Total D ~ s_rad + s_amp D_amp + s_dir D_dir. DEV: D = 1.4, 2.6, 4.3, 7.2, 8.8 at n = 256, L = 1, 4, 8, 16, 32 and 13-14
at L = 64-128; at n = 64 it saturates at 4.5-5 for L >= 16. Prediction: D grows with n at fixed L/n (DEV: 4.7 at
64 x 32 vs 13.3 at 256 x 128).
3.2 Low-order share. The degree-1 share is NOT small at finite width: DEV W_1 = 0.73 (L = 1), 0.47 (4), 0.27-0.35 (8-16),
0.21-0.33 (L >= 32). The infinite-width GP gives only n b_1(L)/V (b_1 = prod kappa'(rho_{l-1}) = 2.8e-3 at L = 32: about 3%
of V at 256 x 32), so most of W_1 is a finite-width effect carried by the amplitude (|a| has W_1 ~ 0.4 at n = 256 for
L <= 128; the first layer's squared gain sum_i relu(w_i.x)^2 is 51% degree 1, 40% degree 2 in its Hermite expansion).
Mechanism: CONJECTURE (the low-rank deep Jacobian product reads a fixed projection of the early states).
The univariate share S_1 adds the radius (s_rad) to W_1: DEV S_1 = W_1 + 0.03-0.06.
3.3 Coordinate change. The off-diagonal degree-2 part is order 2 in random coordinates; in the eigenbasis of its matrix
it becomes order 1. Prediction: rotating to the eigenbasis of E[|a|^2 (x x^T - I)] raises the first-order share by about
the off-diagonal degree-2 share of the amplitude (a few to 15% of V).

## 4. Synthesis: structured gain vs low-order share (HEURISTIC; to be registered in Part 2 with its failure condition)
4.1 RQMC. 1/G = sum_k rho_k(N, d) s_k (1.5). Calibration on synthetic functions (`logs/calib_rqmc_d*.log`, R = 32):
order-1 terms are removed (rho_1 <= 0.02), order-2 terms (exchangeable quadratic form) have rho_2 = 1.5, 0.47, 0.10 at
N = d, 4d, 16d (d = 256) and 0.35, 0.14, 0.03 at N = 4d, 16d, 64d (d = 64), order-3 terms rho_3 = 0.4-1.3. So at fixed N
an RQMC gain shrinks with width for two separate reasons: (i) the order spectrum moves up (a growing interaction order),
(ii) the dimension d grows at fixed N, so the pair projections that remove order-2 terms get worse (rho_2 depends on N/d).
DEV: G(1024) = 7.5, 4.3, 2.7, 2.0, 1.8, 2.2 at n = 256, L = 1..32 against 1/(1 - S_1) = 3.8, 2.6, 1.9, 1.6, 1.4, 1.5: RQMC
removes more than the first-order share (the order-2 part).
4.2 Noise regime. If the saturated gains are first-order dominated, S_1 ~ 1 - 1/G_sat: about 0.87, 0.82, 0.73, 0.65, 0.57 at
n = 8, 16, 32, 64, 128 (from the stationary lane's gains). The antithetic even share is flat (0.21-0.27) while S_1 would fall,
so the growth of the interaction order with width would sit in odd degrees >= 3 (W_odd - W_1 rising from ~0 to ~0.2).
4.3 Antithetics in the input regime: x -> -x flips every first-layer gate, h_1(x) and h_1(-x) are orthogonal, and below
depth 1 the pair behaves like two independent inputs at cosine 0, except for the odd part (W_1 and higher odd degrees).
Prediction: s_even = 0.5 - (W_odd - 1/2)... i.e. gain 1/(2 s_even) between 1.0 and 1.4 for L >= 8 (stationary lane: pair
ratio 0.85-1.06).
4.4 Multilevel shared trajectory. Its level variances follow the depth profile of the amplitude: V(Y - G_r)/V_sph ~
s_amp (Lambda(L) - Lambda(r))/Lambda(L) + s_dir. Its gain is set by depth locality (Lambda's 355/r tail), not by the ANOVA
order: a different "low-dimensional share" from RQMC's.

## 5. Structured sampling in the input regime (HEURISTIC predictions for Part 3)
- At fixed L as n grows: every method's gain tends to 1 (direction part, high order, dominates).
- At fixed L/n >= 1/2: the amplitude dominates; its order profile is n-independent to leading order (Lambda profile), so
  RQMC gains at fixed N/d stay roughly flat and at fixed N fall with n through rho_2(N/d); rotated RQMC adds up to the
  off-diagonal degree-2 share; none of these grows with n.
- Multilevel over depth: gain grows with L at fixed L/n (two-level ~ sqrt(L), dyadic ~ L/log^2 L asymptotically) because
  the amplitude is set within O(100) layers; constants are small at L <= 1024 (about 1-3).
- Exponents: every unbiased method has MSE ~ 1/compute asymptotically except RQMC, whose variance is o(1/N) for L2
  integrands; I expect RQMC's fitted exponents to exceed 1 in the input regime as N/d grows (order-2 removal), unlike the
  noise regime where they saturated.
