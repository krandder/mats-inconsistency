# Cut-rank extension: nonlinear readouts and bounded back-and-forth. Claude's statement and lemmas

Lane: cut-rank-ext (theory), bead pool-4fm6vg. Author: opus-mats-cutrank-ext (Claude Opus 5.5). Written 2026-10-31
~10:50-11:20Z, after gpt-6-astra round 1 was launched on the lemma-free `coprover/STATEMENT_r1.md` and before its answer
was read. Labels: PROVEN, PROVEN UNDER X, HEURISTIC, CONJECTURE.

Setting, criterion and the known results (K1)-(K7) are as in `coprover/STATEMENT_r1.md`. Extra notation:
- H := L^2(backs x [n]) with ||u||_H^2 = E_b (1/n) sum_j u_j(b)^2, X(f) := M(f, .) in H, so Err = E_f ||X(f) - est(f, .)||_H^2.
- E_lin(s) := sum_{i > s} sigma_i^2 (the optimal rank-s linear error), r(eps) = min{ s : E_lin(s) <= eps^2 Vbar }.
- a := 2(n-1)/3. For the genuine ensemble at fixed (n, c >= 3, d), and for ray fronts at fixed (n, d),
  r(eps) >= C eps^{-a} for eps <= eps_0 (K4, K5), C = C_{n,c,d} > 0. For the genuine ensemble C is about
  P(E)^{(n-1)/3}, P(E) ~ 2^{-n^2}.
- p_f = n^2 c, p_b = n^2 d: the parameter counts of the two halves.

## 0. Plain answer (my reading before the co-prover)

Kelvin's statement, "nonlinear readouts and bounded back-and-forth also need (1/eps)^{Theta(n)} numbers", is FALSE for
every regularity class of finite order (Lipschitz, Holder, C^k), and for back-and-forth with real messages. It is TRUE
in three forms: bounded-degree polynomial readouts (exponent divided by the degree), bit messages with any number of
rounds (as the alphabet size 2^b >= r(eps), i.e. b >= a log2(1/eps) - O(1)), and readouts that are analytic with a fixed
radius (as s >~ a log(1/eps)/log log(1/eps)). The precision exponent is a property of the linear span of a curved,
low-dimensional family (the kink functions u -> g_b(u)); nonlinear readouts see the family's dimension instead. On the
very instances that carry the genuine-ensemble theorem (K5: the cut state is a ray), the mean E h_c (n numbers) read
out by g_b itself (naive mean propagation through the back) is exact.

## 1. Bits, any number of rounds (Theorem A). PROVEN

**Theorem A.** Let Pi be any protocol between the halves, with any number of rounds, public and private randomness,
in which at most b bits are exchanged on every run, and the back outputs est_j = R_j(tau, b, omega_B), any function of
the transcript tau, its own input and its own coins. Then Err(Pi) >= E_lin(2^b). Hence Err <= eps^2 Vbar needs
2^b >= r(eps), i.e. b >= log2 r(eps).

Proof. Fix the public coins (Err is their average, so a bound for each fixed value suffices). The protocol tree has at
most 2^b leaves. The front's messages are functions of (f, omega_F, transcript so far) and the back's of (b, omega_B,
transcript so far), so P(tau | f, b, omega_B) = alpha_tau(f) beta_tau(b, omega_B), with alpha_tau(f) the probability over
omega_F that the front's messages agree with tau and beta_tau in {0, 1}. By Jensen, E_coins (M_j - est_j)^2 >=
(M_j - E_coins est_j)^2, and E_coins est_j = sum_tau alpha_tau(f) E_{omega_B}[beta_tau(b, omega_B) R_j(tau, b, omega_B)],
a linear protocol of rank <= 2^b (front features alpha_tau). Its error is >= E_lin(2^b). The same holds if the front
outputs (swap roles). This is the Krause / Lee-Shraibman leaf decomposition, with the output allowed to depend on the
outputting party's full input. []

**Corollary A1.** Genuine ensemble, fixed (n, c >= 3, d): b >= a log2(1/eps) + log2 C, with log2 C ~ -n^2 (n-1)/3 for
the rare-event constant; ray fronts at fixed (n, d): b >= a log2(1/eps) - O_{n,d}(1). PROVEN.

**Proposition A2 (matching order).** One-way, front to back: b <= (p_f + o(1)) log2(1/eps) bits suffice (quantise
theta_f on a box of side T ~ sqrt(log(1/eps)), mesh delta = eps / poly(n, c, T); M is locally Lipschitz in theta_f with
a polynomial constant; the overflow event costs E[|M|^2; overflow] <= (E|M|^4)^{1/2} P(overflow)^{1/2}). Two-way: back
sends quantised theta_b, front returns quantised M: (p_b + 1 + o(1)) log2(1/eps). So the bit complexity is
Theta_{n,c,d}(log(1/eps)) with constant in [a, min(p_f, p_b + 1)]. PROVEN (routine; constants not optimised).

## 2. Bounded-degree, analytic and C^k readouts (Theorem B and corollaries). PROVEN

**Theorem B (degree p, with backward bits).** Suppose the back first sends a B-bit message beta(b) (a function of b),
the front replies phi_k(f) in R^s (any function of f and the received k), and the back outputs
est_j = P_{j,k,b}(phi_k(f)) with P a polynomial of total degree <= p in s variables whose coefficients depend arbitrarily
on (j, k, b). Then est is a linear protocol of rank <= 2^B C(s+p, p). Hence 2^B C(s+p, p) >= r(eps); for B = 0,
s >= (p! r(eps))^{1/p} - p >= (p/e) r(eps)^{1/p} - p, i.e. s = Omega_p(eps^{-a/p}).
Proof: front features Phi_{k,alpha}(f) = phi_k(f)^alpha (all k, |alpha| <= p), back coefficients 1[beta(b) = k] x
(coefficient of x^alpha in P_{j,k,b}). []
Several rounds keep the bound (with B the total backward bits and s the total forward reals) as long as the back's
messages do not depend on the front's real replies. If they do (a threshold of a real reply), rank is not preserved in
general: that case belongs to Section 3 (nonlinear readouts) or, after quantisation, to Theorem A.

**Corollary B1 (analytic readouts with a fixed radius).** Let |phi_i(f)| <= rho for all i and f, and let each
R_j(., b) extend holomorphically to the closed polydisc of radius rho' > rho with |R_j| <= A sqrt(Vbar) there (A, q =
rho/rho' independent of b, j, eps). If Err <= eps^2 Vbar then C(s+p, s) >= r(2 eps) for every p with
A q^{p+1} C(p+s, s) (1-q)^{-s} <= eps. Consequently s >= (a - o(1)) log(1/eps) / log log(1/eps) as eps -> 0.
Proof: Cauchy estimates |c_alpha| <= A sqrt(Vbar) rho'^{-|alpha|}; on |phi_i| <= rho the Taylor tail beyond total degree p is
<= A sqrt(Vbar) sum_{k>p} C(k+s-1, s-1) q^k <= A sqrt(Vbar) q^{p+1} C(p+s, s) (1-q)^{-s} (split multisets: C(m1+m2+s-1, s-1)
<= C(m1+s-1, s-1) C(m2+s-1, s-1)). Triangle inequality in L^2, then Theorem B at error 2 eps. Asymptotics: with
s = sigma log(1/eps)/log log(1/eps), the smallest admissible p is (1 + sigma + o(1)) log(1/eps)/log(1/q), and
log C(s+p, s) = (sigma + o(1)) log(1/eps); r(2 eps) >= C (2eps)^{-a} forces sigma >= a. []

**Corollary B2 (C^k readouts).** Let phi take values in [-rho, rho]^s and each R_j(., b) be C^k there with
max_{|alpha| = k} sup |d^alpha R_j| <= Lambda sqrt(Vbar) rho^{-k} (Lambda independent of b, j, eps). If protocols with a fixed s
reach Err <= eps^2 Vbar along a sequence eps -> 0, then s >= k a = 2k(n-1)/3.
Proof: tensor-product Jackson operators (degree m per coordinate, total degree p = s m) give a polynomial P_j with
sup |R_j - P_j| <= J_{s,k} Lambda sqrt(Vbar) p^{-k}. Take p = ceil((J Lambda / eps)^{1/k}); Theorem B at error 2 eps gives
C(s+p, s) >= C (2 eps)^{-a}; the left side is O(eps^{-s/k}) at fixed s, so s/k >= a. []

## 3. Where it breaks: finite-order regularity (Theorem C). PROVEN

**C1 (raw parameters).** For every front law and every eps: phi(f) = theta_f in R^{p_f} with R(phi, b) = M(phi, b) is
exact, and R(., b) is Lipschitz on every ball |theta_f| <= T with a constant polynomial in T (product of operator norms
times E|x| times Lip(g_b)). Two-way: the back sends theta_b (p_b reals), the front returns y = E_{P_f} g_b (one real),
est = y: exact, two rounds, and the final readout is LINEAR in the front's reply.

**C2 (the hard instances are easy).** If h_c(x) = a(x) v with a(x) >= 0 and a fixed vector v (a ray front; exactly the
event E of K5, and the ray fronts of K4), then phi(f) = E h_c = (E a) v in R^n and R(phi, b) = g_b(phi) give est = M
exactly, by positive homogeneity; R(., b) is globally Lipschitz (constant <= prod_l ||W_l||_op). The same estimate is
naive mean propagation m_{l+1} = relu(W_{l+1} m_l) from m_c = E h_c: a layer-by-layer state of size n with Lipschitz
updates. Hence, for any class containing this readout, the restriction argument of K5 (Err >= P(E) Err(. | E)) gives 0.

**C3 (continuity of the decoder alone is vacuous).** Let Gamma: [0,1] -> {|theta_f| <= T} be a continuous surjection
(space-filling curve). Then phi(f) = any t with Gamma(t) = theta_f (measurable selection) and R(t, b) = M(Gamma(t), b)
give s = 1 with R(., b) continuous (Holder of order 1/p_f), exact on the ball.

**C4 (Lipschitz and Holder decoders: the dimension, not the span; average-case Carl-type lemma).** Let the decoder
y -> R(y, .) be gamma-Lipschitz from (R^s, |.|_2) to H on the ball B_rho, and P(|phi(f)| <= rho) >= 1 - eta. Define
N_mu(r, delta) = the least number of H-balls of radius r whose union has mass >= 1 - delta under the law mu of X(f).
Then for all t, lambda > 0:
  N_mu(t + lambda, eta + Err/lambda^2) <= (1 + 2 rho gamma / t)^s.
Proof: B_rho is covered by (1 + 2 rho gamma/t)^s Euclidean balls of radius t/gamma; their images are covered by H-balls of
radius t; Markov gives P(||X - R(phi, .)||_H > lambda) <= Err/lambda^2. []
For gamma'-Holder-alpha decoders the same argument gives N_mu(t + lambda, .) <= (1 + 2 rho (gamma'/t)^{1/alpha})^s.

**C5 (lower bound for Lipschitz readouts, genuine ensemble and rays).** At fixed (n, c >= 3, d), for Lipschitz decoders
with log(rho gamma) = o(log(1/eps)): s >= (n - 1)(1 - o(1)) as eps -> 0. With C1: (n-1)(1 - o(1)) <= s*_Lip <= p_f.
Holder-alpha: s >= alpha (n-1)(1 - o(1)).
Proof. On E (K5), X(f) = S G(u), S > 0 a scalar independent of u, u uniform on the positive-orthant directions,
G(u) = g_.(u) in H. By (K2), ||S G(u) - S' G(u')||_H^2 = [(S - S')^2 + 2 S S' (1 - kappa(u.u'))]/n, and
1 - kappa(t) = sum_l a_l (1 - P_l(t)) >= a_1 (1 - t) with a_1 > 0 (one back layer: the odd part of the ReLU kernel is
t/2; depth d: a_1^{(d)} >= 2^{-(d-1)} a_1^{(1)}, astra B2 of the cut-rank lane). So with S, S' >= s_0:
||S G(u) - S' G(u')||_H >= s_0 sqrt(a_1/n) |u - u'|, and an H-ball of radius r captures directions in a cap of radius
2 r sqrt(n/a_1)/s_0, of orthant measure <= C_n r^{n-1}. Choose s_0 with P(S < s_0 | E) <= 1/4. Then
N_mu(r, delta) >= c' r^{-(n-1)} for delta < P(E)/4. Apply C4 with lambda = t = 4 eps sqrt(Vbar/P(E)) and eta <= P(E)/16:
s >= [(n-1) log(1/(2 lambda)) + log c'] / log(1 + 2 rho gamma/lambda) -> n - 1. []

**C6 (where exactly the linear argument fails).** The linear proof bounds the error of the class of estimate
operators {(f, b) -> <phi(f), psi(b)>} = {operators of rank <= s} by Eckart-Young. With a nonlinear readout the class is
{(f, b) -> R(phi(f), b)}: for Lipschitz R it contains the exact operator (u, b) -> S g_b(u) on rays, which has infinite
rank but is the image of an (n-1)-dimensional manifold. The step "rank <= s" fails; what survives is "the image of an
s-dimensional set", which bounds dimension (C4), not span. Polynomial approximation of the readout (Theorem B and its
corollaries) converts readout regularity back into rank, which is why degree p, analyticity and C^k each keep part of the
exponent, and Lipschitz keeps only dimension.

## 4. Bounded back-and-forth (Theorem D). PROVEN

**Theorem D (feature span).** Suppose every real number the front sends is y_t = E_{P_f}[e_t(h)] with e_t in a fixed
linear space F of functions, dim F = N; the back's messages and choices of e_t and its output coefficients may depend
arbitrarily on b but not on the front's replies; and est_j = sum_t c_{t,j}(b) y_t. Then est is a linear protocol of rank
<= N, so N >= r(eps), however many rounds and whatever the back sends.
Proof: est_j = E_{P_f}[sum_t c_{t,j} e_t] = sum_k w_{jk}(b) E_{P_f} e^{(k)} for a basis e^{(1..N)} of F. []

**Corollary D1 (naming frames).** If the back names any frame V (any dimension, from any set, real or discrete) and the
front replies with moments of degree <= K of the projected law (Pi_V)_# P_f, then F is inside the polynomials of degree
<= K in h, N <= C(n+K, K), and Err >= E_lin(C(n+K, K)): naming frames never beats the one-way protocol that sends all
moments of degree <= K. It shortens the front's message, not the error.

**Corollary D2 (atlas size).** With B bits of naming and a degree-p readout of s forward reals (Theorem B):
2^B >= r(eps)/C(s+p, p). This is the rigorous form of "even the backward message is long": the number of distinct namings
(the atlas) must be at least r(eps)/C(s+p,p), in bits a log2(1/eps) - log2 C(s+p,p) - O(1).

**D3 (where back-and-forth breaks it).** If the backward message may parametrise nonlinear features (any family not
contained in a fixed finite-dimensional space), two rounds suffice: C1's two-way protocol (theta_b back, E_{P_f} g_b
forward, p_b + 1 reals). For one back layer the ridge family e(h) = relu(w.h) already suffices: the back names its n rows,
the front returns n numbers, est is linear in them and exact (n^2 + n reals).

**On (K7).** Forward measurability shows a one-way front cannot compute the back's frame. It does not constrain two-way
protocols (the back can send it), and by D1 a named frame does not help linear moment protocols anyway. It is not needed
for any bound above.

## 5. Layer-by-layer fixed-size states (streaming). PROVEN

A layer-by-layer estimator keeps S_l in R^s, S_l = U_l(S_{l-1}, W_l), and outputs est = Out(S_L). At any cut c, S_c is a
front-measurable message and Out o U_L o ... o U_{c+1} is a decoder. Hence, at the last cut c = L-1 (one back layer;
needs L - 1 >= 3 for the genuine ensemble, or rays):
- updates of polynomial degree <= q in the state, linear readout: s >= (q/e) r_{L-1}(eps)^{1/q} - q (Theorem B);
- uniformly analytic last update: Corollary B1; uniformly C^k: s >= k a (B2); Lipschitz: s >= (n-1)(1 - o(1)) (C5);
- merely continuous: nothing (C3).

## 6. Quantifiers and window

- Every lower bound above is an ensemble statement per protocol (for each protocol in the class, the ensemble error is
  at least the bound), at fixed (n, c >= 3, d) as eps -> 0 (or rays at fixed (n, d)), with constants inherited from K5
  (astronomically small). Nothing is uniform in n or depth; at fixed eps as n grows, K3 makes r(eps) polynomial in n and
  every derived bound at most logarithmic or polynomial.
- Because the genuine-ensemble bounds are carried by the event E (probability ~ 2^{-n^2}), "most nets defeat each
  protocol" does not follow and would need a small-ball statement (cut-rank C4) for the ray problem; "on most nets every
  protocol fails" is false (a constant answer fitted to one net is exact).
- The constructions C1-C3 and D3 hold for every net and every eps.

## 7. What stays outside

- Unrestricted joint computation: communication is cheap (p_f or p_b + 1 reals; O(min(p_f, p_b) log(1/eps)) bits), so a
  separation from sampling must be a running-time statement. In fixed dimension even black-box randomised cubature beats
  sampling's exponent for Lipschitz integrands (Bakhvalov: error N^{-1/2 - 1/m} in m dimensions), so "nothing beats
  sampling" is false as a black-box time statement at fixed n; for white-box methods a time lower bound needs a
  complexity assumption.
- Adaptive case splitting with unbounded case counts (mixtures over gate patterns): discontinuous updates and growing
  state; only Theorem A applies, and only to the final message length.

## 8. Conjectures (mine, before the co-prover)

- I1 (identifiability). The map theta_f -> X(f) in H has generic differential rank p_f - n(n-1)/2 - n(c-1) (input
  rotations W_1 -> W_1 Q and positive diagonal rescalings of hidden units are the only continuous symmetries). Then C4
  gives s*_Lip >= (p_f - n(n-1)/2 - n(c-1))(1 - o(1)): Lipschitz summaries cannot compress below the parameter count
  modulo symmetries. Cheapest test: the restricted Jacobian's smallest singular value against Monte Carlo noise at small
  n, c.
- I2. C^k readouts: s*_{C^k} = Theta(k n) (upper bound open).
- I3. Analytic readouts: s* = Theta(n log(1/eps)) up to log log factors (upper bound open).

## Addendum 1 (Claude, appended 11:22Z, after round 2 was launched at 11:16Z and before reading its answer; round 2 may or may not have read it)

**C7 (finite-order smoothness is eps-independent: snowflake encoders). Claimed PROVEN, to be checked.** Let the front
family be parametrised by t in a compact set S of dimension m (rays: t = z = alpha v in a cube, m = n; genuine fronts
after clipping: t = theta_f, m = p), with t -> X(t) in H L-Lipschitz and every t -> M(t, b) L_b-Lipschitz. Let
psi: (S, |.|^{1/k}) -> R^s be a bi-Lipschitz embedding of the 1/k-snowflake of S, c |t - t'|^{1/k} <= |psi(t) - psi(t')|
(Assouad's theorem gives one for some finite s = N(k, m)). Send phi = psi(t). On E = psi(S), F_b(y) = M(psi^{-1}(y), b)
satisfies |F_b(y) - F_b(y')| <= (L_b / c^k) |y - y'|^k, so the Whitney jet (F_b, 0, ..., 0) is a C^{k-1,1} Whitney jet
and extends to R^s with C^{k-1,1} norm <= C_{s,k} L_b / c^k (Whitney / Stein extension), exact on E. Hence decoders with an
eps-INDEPENDENT C^{k-1,1} bound reach error 0 with s = N(k, m) numbers, and with Corollary B2:
  2k(n-1)/3 <= s*_{C^{k-1,1}} <= N(k, m), independent of eps.
So the precision exponent is lost at every finite order of smoothness; only infinite-order (uniformly analytic,
Corollary B1) or algebraic (polynomial degree D, Theorem B) restrictions keep an eps-dependence. This supersedes
conjecture I2 (Theta(k n)): proven up to the snowflake embedding dimension N(k, m) >= k m.
- Caveat to check: Whitney's C^{k-1,1} extension needs only the order-k Lipschitz condition on the jet with zero
  derivatives; uniformity over b needs clipped back weights (as in astra A2).

## Addendum 2 (Claude, appended 11:31Z, before reading round 2's answer)

**B1' (analytic class: matching order up to log log).** Corollary B1 holds with the bound A allowed to grow polynomially,
|R| <= eps^{-kappa} sqrt(Vbar) on the polydisc (then the coordinate-degree cutoff is m ~ (1 + kappa) log(1/eps)/log(1/q)
and the conclusion s >= (a - o(1)) log(1/eps)/log log(1/eps) is unchanged). Upper bound: quantise theta_f to
B = (p + o(1)) log2(1/eps) bits (Proposition A2) and send the bits as B reals in {-1, 1}. For each back, the decoder is the
multilinear extension P_b(z) = sum_S fhat_b(S) prod_{i in S} z_i of the clipped dequantised M(., b); it equals the target on
the cube, is entire, and on the polydisc of radius 2 satisfies |P_b| <= 3^B max|f_b| = eps^{-(p log2 3 + o(1))} max|f_b|.
So the analytic class (fixed neighbourhood q = 1/2, polynomially bounded) has
  (a - o(1)) log(1/eps)/log log(1/eps) <= s*_An <= (p + o(1)) log2(1/eps).
This settles conjecture I3 up to the log log factor and the constants (a = 2(n-1)/3 against p = n^2 c).

**Measured so far (NC3, (4,2,2)).** The restricted Jacobian spectrum decays smoothly over 3-4 decades with no gap; only
7-8 of 22 directions are resolved above 10x Monte Carlo noise at N = 65,536. I1 is therefore untested at this precision.
