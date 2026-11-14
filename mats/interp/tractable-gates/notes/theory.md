# Tractable gates: theory note v2.1 (2026-10-31; gpt-6-astra round 1 and the final GPT-6 review applied)

Author: opus-mats-gates (Claude Opus 5.5). Co-prover: gpt-6-astra (coprover/astra_r1_answer.md; v1 of this note is
notes/theory_v1_sent_to_astra.md). Labels: PROVEN, PROVEN UNDER (stated), HEURISTIC, CONJECTURE, COMPUTED (numerics of a
stated model, with its convergence check), WRONG-in-v1 (kept visible where a v1 claim was withdrawn).
Setting is the cut-rank lane's (outbox/cut-rank-lower-bound-2026-10-30.md, section 1) with a general activation phi:
front f = layers 1..c, back b = layers c+1..L, P_f = law of the cut state h_c, g_{b,j}(h) = output j given h_c = h,
M_j(f,b) = E_{P_f} g_{b,j}, Vbar = mean per-net variance, r(eps) = smallest s with sum_{i>s} sigma_i(T)^2 <= eps^2 Vbar.
Hermite convention: h_k = He_k/sqrt(k!), c_k(phi; s) = E[phi(sZ) h_k(Z)], T_D(s) = sum_{k>D} c_k(s)^2.

## 1. A summary bound that needs nothing about the fronts

**Lemma P (PROVEN UNDER measurability: f -> P_f a measurable probability kernel, g_{b,j}(h) jointly measurable, f and b
independent, E_{b,j} integral of |g|^2 dPbar < infinity).** With Pbar = E_f P_f and any s-dimensional S in L^2(Pbar),
Err_opt(s) <= E_{b,j} dist_{L^2(Pbar)}(g_{b,j}, S)^2.
Proof: front sends the integrals of an orthonormal basis of S against P_f, back sends the coefficients of its projection;
Jensen and independence give the bound. The map u -> integral of u dP_f is a contraction L^2(Pbar) -> L^2(fronts),
defined almost everywhere in f; individual P_f need not be absolutely continuous (ReLU faces and atoms allowed).
**P1 (PROVEN, same conditions).** T = A R with ||A|| <= 1, so sigma_i(T) <= sqrt(lambda_i(K)) termwise, with K the
uncentred back covariance operator on L^2(Pbar) (trace class): r_T(eps) <= r_K(eps) at the same budget eps^2 Vbar.
**P2 (PROVEN UNDER E|h|^{2D} < infinity).** Polynomials of degree <= D give r(eps) <= C(n + D, n) once
E_{b,j} dist(g, Poly_D)^2 <= eps^2 Vbar. For a single fixed (trained) back, drop E_b.
**Lemma R (PROVEN; an upper bound, not an exact reduction).** For a one-unit back phi(w.h), w ~ N(0, g^2 I/n)
independent of the front:
E_b dist_{L^2(Pbar)}(phi(w.h), Poly_D)^2 <= dist_{L^2(mu)}(phi, Poly_D)^2, mu the Gaussian scale mixture N(0, S^2),
S = g|h|/sqrt n, h ~ Pbar. One common scalar polynomial, averaged over w.
**R1 (PROVEN UNDER S <= S_max and E[S_max/S; S > 0] < infinity).** dist^2 <= E[S_max/S; S>0] T_D(S_max), plus one extra
feature 1{h = 0} for an atom at 0 (the Gaussian L^2 error does not control the value at 0). Accumulation of mass near
S = 0 can make the inverse moment infinite.
**Unbounded scale (PROVEN UNDER, astra's repair).** If phi grows polynomially, P(S > A) <= C exp(-a A^2) and
H_D(2A) = sum_{k>D} (k+1) c_k(phi; 2A)^2 <= C (1+A)^p exp(-c D^gamma / A^q) uniformly in A >= 1, then truncated
polynomial features p(h) 1{S(h) <= A} with A ~ sqrt(log 1/eps) give r(eps) = O(log(1/eps)^{n(1+q/2)/gamma}).
Caveat: Gaussian inputs do not make S Gaussian-tailed after random front weights (a product of two Gaussians already has
exponential tails); with exponential tails A ~ log(1/eps) and the polylog exponent grows.

## 2. Hermite classes of the gate family (lib/acts.py; tests/test_acts.py, 38 checks pass)

| gate | c_k(phi; s), large k | class, label |
|---|---|---|
| ReLU | c_k^2 = s^2 ((k-3)!!)^2/(2 pi k!) ~ k^{-5/2} (even k), 0 odd k >= 3 | power alpha = 5/2, PROVEN |
| ReLU^2 | ~ k^{-7/2} | power alpha = 7/2, PROVEN (order) |
| step (tanh at s -> infinity) | ~ k^{-3/2} | power alpha = 3/2 |
| ReLU + Gaussian pre-activation noise tau (gsrelu) | c_k = r^k S' c_k^relu(1), S'^2 = s^2 + tau^2, r^2 = s^2/(s^2 + tau^2) | geometric rho = 1/(1 + tau^2/s^2), PROVEN |
| GELU = gsrelu(tau = 1) - npdf | both geometric with rho = s^2/(1 + s^2) | geometric, PROVEN |
| softplus_beta = E relu(z + L/beta), L logistic = 2K Z | c_k = c_k^relu(1) E_sigma[S'_sigma r_sigma^k], tau = sigma/beta | log c_k^2 = -2 pi sqrt(k)/(beta s) + O(log k) along even k, PROVEN (Laplace on exp(-pi^2/(2 sigma^2)); no cancellation, all terms one sign) |
| SiLU = E_sigma sigma GELU(z/sigma); tanh = E_sigma[2 Phi(2z/sigma) - 1] | mixtures of geometric | root-exponential on their nonzero parity subsequences, PROVEN |
| smoothstep (z S(z/delta), C-infinity, exact zeros) | quadrature only | faster than any power in the computed range; no asymptotic rate established (COMPUTED fits only) |

Hille's strip theorem agrees with the softplus constant (probabilists' strip width tau -> physicists' tau/sqrt 2). Softplus
has logarithmic branch points at z = i pi(2m+1)/beta; its derivative has poles.
**Effective sharpness.** What sets the decay is the smoothing width relative to the pre-activation scale s: beta s for
softplus, s for GELU, s/tau for noise. Post-activation noise sigma becomes pre-activation noise of std sigma ||w|| at the
next layer (conditional on the row w; about g sigma under concentration), so tau/s ~ sigma/sqrt(E h^2) in He nets and
D(eps) ~ log(1/eps^2)/log(1 + tau^2/s^2): realistic noise levels (sigma = 1e-3 to 1e-2 of the activation scale) do not
smooth enough to matter (HEURISTIC under concentration).
At s = sqrt 2 the nonlinear tail share T_8/T_1 is 1.9e-2 (ReLU), 1.7e-2 (softplus16), 6.4e-3 (softplus4), 2.8e-5
(softplus1), 3.1e-3 (GELU), 2.9e-4 (SiLU), 5.1e-2 (tanh), 4.6e-3 (gsrelu0.5), 2.5e-2 (smoothstep1)
(logs/hermite_classes.log). A C-infinity gate with exact zeros (smoothstep) is no smoother than ReLU at low degrees.

## 3. Fixed width, eps -> 0

**Theorem A (one back layer).** (i) PROVEN UNDER section 1's conditions (bounded scale with the inverse moment, or the
unbounded-scale repair's tail and uniform-approximation bounds; these are essential: final review, finding 1, gives a
counterexample with a heavy-tailed radius, P(R > r) = r^{-5/2}, where radial averaging turns the geometric decay of
Gaussian-smoothed ReLU into b_k ~ k^{-4} and the width-8 ray rank into eps^{-7/3}): power-class tails T_D <= C D^{1-alpha} give
r(eps) = O(eps^{-2n/(alpha-1)}); stretched-exponential tails T_D <= C exp(-c D^gamma) give r(eps) = O((log 1/eps)^{n/gamma}) = o(eps^{-2}) at every
fixed n and any front law in the bounded-scale case (R1); under the unbounded-scale repair of section 1 the bound is
r(eps) = O((log 1/eps)^{n(1+q/2)/gamma}), still polylogarithmic (second-round review, finding 6). WRONG-in-v1: "super-polynomial decay implies polylog rank";
merely super-polynomial decay (e.g. exp(-(log k)^2)) gives only r(eps) = o(eps^{-eta}) for every eta > 0.
(ii) Ray fronts (the lane's Theorem E model: h = s R0 u_f, u_f uniform on the sphere, R0 independent, one back unit).
PROVEN: the cut spectrum is the Funk-Hecke spectrum of kt(t) = sum_k (E_R c_k(s R0))^2 t^k, Vbar =
sum_k [E_R c_k^2 - (E_R c_k)^2], and the Gegenbauer energy shares and counting in scratch/ray_family.py are right.
WRONG-in-v1: calling the v1 table "exact": truncation at K = 6000 and quadrature moments leave unchecked tails (softplus16:
coefficients beyond K carry more than the eps = 1e-6 budget). The convergence-checked table (K vs 2K, 24 vs 48 radial
nodes, logs/ray_cert.log) replaces it in the report; ReLU uses the lane's closed form.
(iii) PROVEN UNDER regular variation b_k ~ B k^{-alpha} (alpha > 1, fixed n >= 3): E_l ~ l^{1-2 alpha} and
r(eps) ~ eps^{-(n-1)/(alpha-1)} (Stirling on the exact shares, e_{k,l} ~ l^{n-2} k^{-(n-1)/2} exp(-l^2/(2k)) on k ~ l^2);
outgrows 1/eps^2 iff n > 2 alpha - 1: n >= 5 for ReLU (the lane), n >= 7 for ReLU^2. For log b_k ~ -c k^gamma,
0 < gamma < 1: log E_l ~ -c' l^{2 gamma/(1+gamma)} (PROVEN); gamma = 1: exponential in l (constant from the Bernstein
ellipse, not the quadratic saddle); gamma > 1: log E_l ~ -c l^gamma.
HEURISTIC (astra): unbounded radial averaging weakens classes (exponential R: Gaussian smoothing to a k^{1/3} exponent,
softplus to k^{1/4}).
**Reading.** Finite smoothness loses at large enough width; stretched-exponential Hermite decay wins at every fixed
width as eps -> 0. In the worst case (concentrated fronts) the constants decide where.

## 4. Fixed precision, growing width
P2 gives C(n + D(eps), n) ~ n^D/D! (polynomial in width; the lane's Theorem 3 for ReLU). With D(eps) from the 1-D tails
(and Var phi(sZ) standing in for Vbar, so these are illustrations, not certificates for any front law), the bound at
n = 64 is 10^8.1/10^49.7 (ReLU) and 10^3.3/10^8.1 (softplus1) at eps = 0.1/0.01, against sampling's 10^2/10^4. An upper
bound above sampling does not show a loss: a useful window at n >= 16 is not ruled out by this section.

## 5. Depth (infinite width, normalised layers; logs/depth_radius.log)
The normalised correlation map f(t) = sum_k b_k t^k / sum b_k is the same at every layer (f(1) = 1, f'(1) = chi_1).
PROVEN UNDER a nonconstant positive-coefficient series with 1 < R_1 < infinity (generalised inverse when the target is
not reached): the d-fold composition has radius R_d = min(R_1, f^{-1}(R_{d-1})).
These radius statements need 1 < R_1 < infinity, i.e. geometric-class gates (GELU, gsrelu). They do not apply to
root-exponential kernels (softplus, SiLU, tanh), whose radius is already 1.
- chaotic (chi_1 > 1): R_d - 1 contracts geometrically. GELU at q0 = 2 (chi_1 = 1.06): R_d - 1 = 0.50, 0.37, 0.25, 0.14,
  0.063, 0.019, 0.0025 at d = 1, 2, 4, 8, 16, 32, 64 (COMPUTED).
- critical (chi_1 = 1): R_d - 1 ~ 2/(f''(1) d) (PROVEN UNDER f''(1) > 0).
- ordered (chi_1 < 1): the radii stay uniformly above 1; R_d = R_1 holds when also f(R_1^-) <= R_1 (true for gsrelu0.5
  and gsrelu1 at q0 = 2, COMPUTED). WRONG-in-v1: "chi_1 < 1 implies R_d = R_1" (astra's counterexample
  f(t) = 0.8 + 0.2t/(2 - t): R_2 = 12/7 < R_1 = 2).
- WRONG-in-v1: "smoothness survives depth only in the ordered phase". Every finite composition stays C-infinity (entire
  stays entire), and root-exponential classes already have radius 1. The radius alone does not fix the degree needed at a
  finite eps (prefactors matter), so no fixed-eps degree growth is claimed; the degree column of the log is indicative.
- Ordering is a property of the particular gate and scale, not of smoothness or non-negativity: phi(z) = z^2 is smooth and
  non-negative with normalised chi_1 = 4/3 (final review, finding 15).
- Finite width: per-net compositions stay smooth; O(L/n) corrections to radii are CONJECTURE.

## 6. Multiplicative gating
- PROVEN: for phi(w1.h)(w2.h), K_GLU(h, h') = (g_2^2/n)(h.h') K_phi(h, h'): Taylor coefficients shift by one, so SwiGLU
  has SiLU's class, GEGLU GELU's, ReGLU ReLU's. On rays the shifted coefficients are (E_R[s R c_k(s R)])^2 (as coded).
- PROVEN UNDER moments: a depth-d bilinear back is a degree-2^d polynomial, r(eps) <= C(n + 2^d, n) for every eps.
  Its variance map q -> A q^2 has derivative 2 at the nonzero fixed point, so plain bilinear stacks are scale-unstable.
  This argument covers bilinear (homogeneous) gates only; for a general GLU the map contains q E[phi(g sqrt(q) Z)^2] and
  is not universally unstable. The SwiGLU divergence seen here (smoke check; sorting at initialisation) is an observed
  failure of this initialisation and architecture, not a theorem. Real models add RMSNorm/LayerNorm, which removes
  polynomial finiteness (LayerNorm is singular at 0 and on constant vectors).

## 7. Smooth fronts
- PROVEN UNDER square invertible layers and phi' > 0 everywhere: the front map is a C^1 diffeomorphism onto its image,
  so P_f has a density; no faces, no ray atoms (the ReLU mechanisms behind the lane's exponent do not occur).
- WRONG-in-v1: "strictly monotone gates pass the L^2-density condition". In 1-D the Lebesgue norm is
  integral of p(z)^2/phi'(z) dz: for erf with Z ~ N(0, s^2) it is finite only when s^2 < 1. Softplus and tanh pass at each
  fixed Gaussian scale; ensemble integrability is a separate question (even random scalar linear maps diverge near zero
  weight).
- GELU/SiLU folds give |y - y_c|^{-1/2} densities (also locally in n dimensions where a fold is reached with positive
  density): this fails Lebesgue L^2 but not every common reference measure (WRONG-in-v1 as a failure of the theorem).
- The lane's smooth-front theorem is stated for homogeneous backs (radially weighted angular densities). For
  non-homogeneous smooth backs: PROVEN UNDER a common measure making both the front-density factor and the back-function
  factor Hilbert-Schmidt, T is trace class and r(eps) = o(eps^{-2}).
- **Perturbation lower bound (PROVEN, astra).** Couple ReLU and softplus_beta nets with the same weights.
  0 <= softplus_beta(z) - z_+ <= log 2/beta and Lipschitz propagation give ||T_beta - T_ReLU||_HS <= C_{n,L}/beta =: Delta,
  hence r_beta(eps) >= r_ReLU((eps sqrt(Vbar_beta) + Delta)/sqrt(Vbar_ReLU)). The ReLU lower bounds transfer to softplus
  at tolerances above O(1/beta); nothing asymptotic as eps -> 0 at fixed beta.

## 8. Learnability side (PROVEN approximation; necessity and training unresolved)
Replacing each comparator's ReLU in a sorting network of comparator depth d_sort by softplus_beta gives uniform
coordinate error at most d_sort log 2/beta with the same connectivity (astra). So smooth gates can approximate sorting
through increasingly sharp comparators. This is a sufficient construction; it does not show that every accurate smooth
network must be sharp (final review, finding 4), and the perturbation bound of section 7 is a stability statement for
coupled networks, not a learnability lower bound.

## 9. Practical prediction (HEURISTIC; feeds PREREG P1-P4)
Measured ReLU cut ranks at eps >= 0.03 (lane: 14-139) sit far below the ray model's, so front averaging, not the gate's
high-degree tail, sets them; smoothing removes energy only beyond the gate's crossover degree. WRONG-in-v1 as a
generalisation: "soft gates are ordered". Only small-beta softplus and gsrelu are ordered at q0 = 2 (chi_1 = 0.46 for
softplus1); SiLU (1.07), GELU (1.06) and tanh (1.34) are chaotic. For the ordered softplus1, tight fronts shrink Vbar;
whether that raises the relative cut rank is not determined by the theory (HEURISTIC) and is P2's measurement.

## Open (conjecture ledger)
- C-front: ensemble L^2 integrability of softplus/tanh fronts against a common reference measure.
- C-depth: finite-width backs follow section 5's phase ordering with O(L/n) corrections.
- C-window: a sharpness window (beta s about 2-4) where training reaches ReLU's loss and summaries beat sampling at
  eps = 0.01 on genuine fronts at n = 8; not ruled out at n >= 16 by section 4.
