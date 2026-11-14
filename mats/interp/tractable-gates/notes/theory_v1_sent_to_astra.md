# Tractable gates: theory note (2026-10-31, opus-mats-gates; for the gpt-6-astra check)

Labels: PROVEN, PROVEN UNDER (stated assumptions), HEURISTIC, CONJECTURE, COMPUTED (exact numerics of a stated model).
Setting and notation are the cut-rank lane's (outbox/cut-rank-lower-bound-2026-10-30.md, section 1) with a general
activation phi: z_l = W_l h_{l-1}, h_l = phi(z_l), front f = layers 1..c, back b = layers c+1..L, P_f = law of h_c,
g_{b,j}(h) = output j given h_c = h, M_j(f,b) = E_{P_f} g_{b,j}, Vbar = mean per-net variance, r(eps) = smallest s with
sum_{i>s} sigma_i(T)^2 <= eps^2 Vbar (one-way linear summaries; Fact 1 of the lane). Hermite convention:
h_k = He_k/sqrt(k!), c_k(phi; s) = E[phi(sZ) h_k(Z)], T_D(s) = sum_{k>D} c_k(s)^2.

## 1. A summary bound that needs nothing about the fronts

**Lemma P (PROVEN).** Let Pbar = E_f P_f (the cut state's law over fronts and inputs). For any s-dimensional space
S of functions in L^2(Pbar),
  Err_opt(s) <= E_{b,j} dist_{L^2(Pbar)}(g_{b,j}, S)^2.
Proof. Let e_1..e_s be an orthonormal basis of S in L^2(Pbar) and Pi the projection. Protocol: the front sends
phi_k(f) = integral of e_k dP_f; the back sends psi_{j,k}(b) = <g_{b,j}, e_k>; the estimate is integral of
(Pi g_{b,j}) dP_f. By Jensen and the independence of f and b,
E_{f,b,j} (integral of (g - Pi g) dP_f)^2 <= E_{b,j} E_f integral of (g - Pi g)^2 dP_f = E_{b,j} ||g - Pi g||^2_{Pbar}.
(The map u -> integral of u dP_f is a contraction L^2(Pbar) -> L^2(fronts), well defined on classes; P_f need not be
absolutely continuous, so ReLU's faces and atoms are allowed.)

**Corollary P1 (PROVEN).** With K(h,h') = E_{b,j} g_{b,j}(h) g_{b,j}(h') on L^2(Pbar) and lambda_i its Mercer
eigenvalues, T = S R with ||S|| <= 1, so sigma_i(T) <= sqrt(lambda_i) termwise and r_T(eps) <= r_K(eps), the eps-rank of
the back's covariance kernel on the cut state's marginal law. Fronts can only help (front averaging), never hurt.
For a single trained back the same bound holds with E_b removed: the summary is the front's moments of a basis.

**Corollary P2 (polynomial summary, PROVEN).** With S = polynomials of degree <= D in h (dimension C(n+D, n)),
r(eps) <= C(n + D, n) as soon as E_{b,j} dist_{L^2(Pbar)}(g_{b,j}, Poly_D)^2 <= eps^2 Vbar.

**Lemma R (one back layer reduces to one dimension, PROVEN).** If the output is phi(w.h) with w ~ N(0, g^2 I_n / n)
independent of the front, then E_b dist_{L^2(Pbar)}(phi(w.h), Poly_D)^2 <= dist_{L^2(mu)}(phi, Poly_D)^2, where mu is
the law of w.h for h ~ Pbar: a Gaussian scale mixture N(0, S^2), S = g |h| / sqrt(n).
Proof. A 1-D polynomial in w.h is a polynomial in h; take the best p for mu and average over w:
E_w ||phi(w.) - p(w.)||^2_{Pbar} = ||phi - p||^2_mu.
So for one back layer the summary size is governed by the 1-D polynomial approximability of the gate on its own
pre-activation law, for ANY front law (Gaussian, faces, rays).

**Corollary R1 (PROVEN UNDER bounded scale).** If S <= S_max a.s. and E[S_max/S; S > 0] = kappa < infinity, then
dist^2_mu(phi, Poly_D) <= kappa T_D(S_max) + P(S = 0) (phi(0) - (Pi_D phi)(0))^2 with Pi_D the Hermite truncation at
scale S_max (density ratio dN(0,S^2)/dN(0,S_max^2) <= S_max/S). The gate's sharpness at the largest pre-activation
scale sets D(eps). For unbounded, Gaussian-tailed S (Gaussian inputs) truncate at S_max ~ sqrt(log 1/eps): HEURISTIC
(the extra tail term is not bounded here).

## 2. Hermite classes of the gate family (exact formulas, lib/acts.py; tests/test_acts.py, 38 checks)

| gate | c_k(phi; s), large k | class |
|---|---|---|
| ReLU | closed form, c_k^2 = s^2 ((k-3)!!)^2 / (2 pi k!) ~ k^{-5/2} (even k), 0 for odd k >= 3 | power, alpha = 5/2 (PROVEN) |
| ReLU^2 | c_k^2 ~ k^{-7/2} | power, alpha = 7/2 (PROVEN order) |
| step / saturated tanh (s -> infinity) | c_k^2 ~ k^{-3/2} | power, alpha = 3/2 |
| ReLU + Gaussian pre-activation noise tau (gsrelu) | c_k = r^k S' c_k^relu(1), S' = sqrt(s^2 + tau^2), r^2 = s^2/(s^2+tau^2) | geometric, rho = 1/(1 + tau^2/s^2) (PROVEN) |
| GELU = gsrelu(tau=1) - npdf | both pieces geometric with rho = s^2/(1+s^2) | geometric (PROVEN) |
| softplus_beta = E relu(z + L/beta), L logistic | c_k = c_k^relu(1) E_sigma[S'_sigma r_sigma^k], tau = sigma/beta, sigma = 2K (Kolmogorov) | log c_k^2 ~ -2 pi sqrt(k)/(beta s) (HEURISTIC, Laplace on the Kolmogorov density's exp(-pi^2/(2 sigma^2)) edge) |
| SiLU = z P(L < z) = E_sigma sigma GELU(z/sigma) | mixture of geometric | exp(-c sqrt k) class (HEURISTIC) |
| tanh = 2 P(L < 2z) - 1 = E_sigma [2 Phi(2z/sigma) - 1] | mixture of geometric | exp(-c sqrt k) (HEURISTIC) |
| smoothstep (z S(z), S C-infinity with exp(-1/x) flanks, exact zeros) | quadrature | faster than any power, slower than exp(-c sqrt k): local log-slope -1.2 at D in [32,128], -3 at [64,1024] (COMPUTED) |

The logistic law is a Gaussian scale mixture with scale 2K, K Kolmogorov-distributed (Andrews-Mallows 1974;
Stefanski 1991); checked to 1e-12 (tests). Hille (1940): Hermite coefficients of functions analytic in a strip of
half-width tau decay like exp(-tau sqrt(2k)) (physicists' normalisation); softplus_beta(s x) has poles at
x = i pi (2m+1)/(beta s), and in the probabilists' normalisation the rate exp(-pi sqrt(k)/(beta s)) per coefficient
matches the Laplace computation above.

**Effective sharpness (PROVEN for the noise case).** What matters is the gate's smoothing width relative to the
pre-activation scale s: softplus beta s; GELU s (its Phi has unit width); ReLU with Gaussian pre-activation noise of
std tau: rho = 1/(1 + tau^2/s^2). Post-activation noise sigma (h = relu(z) + sigma xi, the noisy-net lanes) becomes
pre-activation noise of std g sigma at the next layer (W xi, W ~ N(0, g^2/n)), so tau/s = sigma/sqrt(E h^2) for He
nets, and the degree needed is D(eps) ~ log(1/eps^2) / log(1 + sigma^2) ~ log(1/eps^2)/sigma^2 for small sigma
(sigma = 1e-3: D ~ 1e7; sigma = 0.3: D ~ 100). Realistic noise levels do not smooth enough.

Numbers at s = sqrt 2 (logs/hermite_classes.log): the nonlinear tail share T_8/T_1 is 1.9e-2 (ReLU), 1.7e-2
(softplus16), 6.4e-3 (softplus4), 2.8e-5 (softplus1), 3.1e-3 (GELU), 2.9e-4 (SiLU), 5.1e-2 (tanh, all-odd), 4.6e-3
(gsrelu0.5), 2.5e-2 (smoothstep1); T_64/T_1: 9.0e-4, 4.4e-4, 3.5e-6, ~0, 1.1e-13, ~0, 6.2e-7, 8.6e-8, 5.5e-4.

## 3. Fixed width, eps -> 0: what smoothness buys

**Theorem A (one back layer).** (i) PROVEN UNDER Corollary R1's assumptions: if T_D(S_max) <= C D^{1-alpha}, then
r(eps) = O(eps^{-2n/(alpha-1)}); if T_D(S_max) <= C exp(-c D^gamma), then r(eps) = O((log 1/eps)^{n/gamma}) = o(eps^{-2})
at every fixed n and any front law. (ii) COMPUTED (exact, ray fronts as in the lane's Theorem E: h = s R0 u_f, u_f uniform
on the sphere, R0 ~ Gamma(1, 1), one back unit; logs/ray_family.log): the cut spectrum is the Funk-Hecke spectrum of
kt(t) = sum_k (E_R c_k(s R0))^2 t^k; ranks:

| n = 8; eps -> | 0.3 | 0.1 | 0.03 | 0.01 | 1e-3 | 1e-4 | 1e-6 |
|---|---|---|---|---|---|---|---|
| sampling 1/eps^2 | 11 | 100 | 1,111 | 1e4 | 1e6 | 1e8 | 1e12 |
| ReLU | 27 | 184 | 1.4e4 | 1.6e6 | 7.1e10 | 3.3e15 | 7.1e24 |
| softplus16 (beta s = 23) | 26 | 162 | 7.1e3 | 5.2e5 | 1.4e9 | 5.1e11 | 1.7e14 |
| softplus4 (5.7) | 24 | 45 | 1.7e3 | 5.0e4 | 2.2e7 | 2.4e9 | 2.3e12 |
| softplus1 (1.4) | 11 | 42 | 281 | 1.7e3 | 1.4e5 | 5.7e6 | 1.75e9 |
| GELU | 26 | 110 | 2.4e3 | 4.5e4 | 6.5e6 | 2.8e8 | 6.2e10 |
| SiLU | 23 | 44 | 1.1e3 | 1.2e4 | 1.1e6 | 4.2e7 | 1.1e10 |
| gsrelu0.5 | 23 | 44 | 1.6e3 | 3.9e4 | 9.9e6 | 7.8e8 | 3.9e11 |

At n = 16 every arm loses to sampling at every eps down to 1e-6 (e.g. softplus1: 144 / 3.6e3 / 3e5 / 9e8 / 1.6e12 /
2.2e17 at eps 0.1 .. 1e-6; sampling 100 .. 1e12). Local exponents d log r / d log(1/eps) at n = 8, between eps = 1e-3
and 1e-4, then between 1e-4 and 1e-6: ReLU 4.66 / 4.66 (= 2(n-1)/3); softplus16 2.58 / 1.26; softplus4 2.04 / 1.49;
GELU 1.64 / 1.17; SiLU 1.59 / 1.21; gsrelu0.5 1.89 / 1.35; softplus1 1.60 / 1.24. The smooth arms' exponents fall
below sampling's 2 and keep falling (polylog growth). Against sampling at n = 8: softplus1 wins at every eps <= 0.1;
SiLU and gsrelu0.5 win at eps = 0.1, lose or tie at 0.03-1e-3 and win again at 1e-4 (SiLU) or 1e-6; GELU wins only at
1e-6; softplus4 wins only at 0.1; softplus16 and ReLU never win in the range (softplus16 would cross near eps ~ 1e-9 by
extrapolation). The win-loss pattern is not monotone in eps: the smooth cut-off acts only beyond the gate's crossover
degree, while the low degrees are ReLU-like.
(iii) HEURISTIC (Laplace): for ray fronts, a power class b_k ~ k^{-alpha} gives degree energies E_l ~ l^{1-2alpha} and
r ~ eps^{-(n-1)/(alpha-1)} (alpha = 5/2 recovers the lane's 2(n-1)/3, PROVEN there); this outgrows 1/eps^2 iff
n > 2 alpha - 1: n >= 5 for ReLU, n >= 7 for ReLU^2. A class b_k ~ exp(-c k^gamma) gives E_l ~ exp(-c' l^{2gamma/(1+gamma)})
and r ~ (log 1/eps)^{(n-1)(1+gamma)/(2 gamma)}.

**Reading.** Any finite smoothness loses at large enough width; only super-polynomial Hermite decay (C-infinity gates)
wins at every fixed width as eps -> 0. But in the worst case (concentrated fronts) the win starts at astronomically
small eps unless n is small or the gate is soft (beta s ~ 1). The constant is the issue, not the exponent.

## 4. Fixed precision, growing width (PROVEN upper bound, HEURISTIC reading)
Corollary P2 gives C(n + D(eps), n) ~ n^D / D! for D << n: polynomial in width at fixed eps for every gate (the lane's
Theorem 3 for ReLU). With D(eps) from section 2 (logs/hermite_classes.log, n = 64): log10 of the bound at
eps = 0.1/0.01/1e-4 is 8.1/49.7/- (ReLU), 5.9/20.2/52.6 (softplus4), 3.3/8.1/17.8 (softplus1), 5.9/15.0/26.4 (GELU),
against log10(1/eps^2) = 2/4/8. The polynomial bound never certifies a win at width >= 16 for practical eps; whether
actual fronts do better is the measurement (grid C).

## 5. Depth (infinite width; PROVEN for positive power series, COMPUTED numbers; logs/depth_radius.log)
With per-layer normalisation the normalised correlation map f(t) = sum_k b_k t^k / sum b_k is the same at every layer,
with f(1) = 1 and f'(1) = chi_1. The d-fold composition has radius of convergence R_d = min(R_1, f^{-1}(R_{d-1})).
- ordered (chi_1 < 1, e.g. softplus at small beta s, gsrelu): f(t) < t just above 1, so R_d = R_1 for all d: depth
  does not erode analyticity (but correlations collapse to 1: the net forgets its input).
- chaotic (chi_1 > 1, GELU at q0 = 2 has chi_1 = 1.06): R_d - 1 shrinks geometrically: 0.50, 0.37, 0.25, 0.14,
  0.063, 0.019, 0.0025 at d = 1, 2, 4, 8, 16, 32, 64; the Taylor degree for a relative tail 1e-4 grows 23 -> 3,673.
- critical (chi_1 = 1): R_d - 1 ~ 2/(f''(1) d) (from f(1+delta) ~ 1 + delta + f''(1) delta^2/2), degree linear in depth.
So a smooth gate keeps its smoothness with depth only in the ordered phase, which is the phase where deep nets lose
their input. This is the smooth counterpart of the lane's growing ReLU kink coefficient (0.21 d).
Finite width: HEURISTIC that the same ordering holds (not proven; finite-width kernels differ by O(L/n)).

## 6. Multiplicative gating
- **GLU kernel (PROVEN, one layer).** For phi(w1.h) (w2.h) with independent Gaussian w1, w2, E over the back of the
  product at (h, h') is (h.h') E[phi(w1.h) phi(w1.h')], so the Taylor coefficients are shifted by one: SwiGLU has SiLU's
  class, GEGLU GELU's, ReGLU ReLU's. The product does not change the decay class (ray model: glu-silu at n = 8 needs
  122 / 8.1e3 / 1.9e10 at eps 0.1 / 0.01 / 1e-6, against SiLU's 44 / 1.2e4 / 1.1e10).
- **Bilinear GLU (PROVEN).** With phi = identity a depth-d back is a polynomial of degree 2^d in h, so
  r(eps) <= C(n + 2^d, n) for every eps (exactly finite). But plain GLU stacks are scale-unstable: the infinite-width
  fixed point q0 of the q-map is repelling (h' is degree 2 in the scale), and finite nets diverge even with per-layer
  pilot normalisation (measured in the smoke check: RMS pre-activation 1.4 -> 37 by layer 8 at n = 8). Real models add
  a per-input normalisation (RMSNorm/LayerNorm), which is analytic away from 0 but not polynomial, so the exact
  finiteness is lost and the class becomes that of the normalisation composed with the polynomial.

## 7. Smooth fronts
- **Densities (PROVEN).** If phi is C^1 with phi' > 0 everywhere (softplus, tanh, erf, gsrelu) and every W_l is
  invertible (probability 1 for Gaussian weights), the front map x -> h_c(x) is a C^1 diffeomorphism onto its image,
  so P_f has a density (change of variables). No faces, no ray atoms: the ReLU mechanisms behind the lane's exponent
  (one-active-unit cones, the exact merge event of Theorem 4(c)) do not occur.
- **Approximate merging (HEURISTIC).** The lane's rare merge event (one positive row, all other rows negative) gives,
  for softplus, h_{c-1} = a(x) e_1 + delta(x) with delta_i = softplus_beta(negative z) ~ exp(-beta |z|)/beta, a
  "fuzzy ray"; the ray lower bound then transfers (the lane's C2 route) only down to eps of order the energy-weighted
  spread, which vanishes as beta -> infinity (ReLU recovered).
- **Folds (PROVEN, 1-D; HEURISTIC in n dimensions).** GELU and SiLU are not monotone (phi' = 0 at z = -0.75 and
  -1.28), so their pushforwards have fold singularities: in 1-D the density of phi(Z) behaves like |y - y_c|^{-1/2} at
  the minimum value, whose square is not integrable (log divergence). So GELU/SiLU fronts are absolutely continuous
  but fail the lane's common-L^2-density condition (smooth-front theorem) by a logarithm; strictly monotone gates pass it
  in 1-D. Whether E_f ||q_f||^2 < infinity holds in n dimensions for softplus/tanh fronts is not proven here
  (CONJECTURE C-front: it holds whenever phi' is bounded below on compacts and the Gaussian tails control saturation).
- **Consequence (PROVEN UNDER C-front).** Then the lane's smooth-front theorem gives r(eps) = o(eps^{-2}) whatever the
  back, even a ReLU back. Two independent routes to "summaries eventually beat sampling": smooth fronts (densities) or
  smooth backs (fast Hermite decay). ReLU fails both.

## 8. What this predicts for practical precision (feeds PREREG.md P1-P4)
At eps >= 0.03 the measured ReLU cut ranks (lane: 14-139 scale-normalised) sit far below the ray model's, so front
averaging, not the gate's high-degree tail, sets them. Smoothing removes energy only at high Hermite degree (beyond
D ~ (beta s / 2 pi)^2 for softplus), which practical eps does not resolve unless beta s ~ 1-2. At beta s ~ 1-2 the gate is
nearly quadratic on its input range and bias-free nets are strongly ordered (chi_1 = 0.46 for softplus1), so fronts
are tight, Vbar is small and the relative budget eps^2 Vbar shrinks: the soft end can raise, not lower, the practical
cut rank. Hence P1 (no practical gain at beta s >= 5) and P2 (softplus1 not below ReLU).

## Open (conjecture ledger)
- C-front (above).
- C-depth: finite-width backs follow the phase ordering of section 5 with O(L/n) corrections.
- C-window: there is a sharpness window beta s in [2, 4] where (a) training reaches ReLU's loss and (b) the summary
  beats sampling at eps = 0.01 on genuine fronts at n = 8; at n >= 16 no such window (by section 4) unless fronts spread.
