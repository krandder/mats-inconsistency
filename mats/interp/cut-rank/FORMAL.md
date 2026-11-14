# Cut rank of critical random ReLU nets: the one-way statement (fixed 2026-10-30, ~23:40Z)

Lane: cut-rank (theory). Bead pool-ttdzrk. Author: opus-mats-cutrank (Claude Opus 5.5). Co-prover: gpt-6-astra.
Status labels: PROVEN, PROVEN UNDER X, HEURISTIC, CONJECTURE, MEASURED.

## 0. Kelvin's premise, and why it needs a cut

Kelvin: on any deep enough net the state collapses to k* effective dimensions; they interact fully and pass through
random gates, like k* bits (several bits each) going through layers of arbitrary reshuffling and flipping of about
half the bits; so there is no way to estimate better than sampling. Premise to prove: after random gating, the joint
structure on the ~k* effective dimensions cannot be described with fewer than about exp(k*) numbers at the needed
precision; each gate cuts the joint law and later layers read all of it. "Prove it for a deep enough network at a
given width, almost always (due to the randomness of the weights)."

As a statement about description size with arbitrary decoders it is false: the front weights (n^2 c numbers), or N
samples of the cut state (N n numbers, error Var/N), are short descriptions. So the statement must restrict how a
description is read. The one-way, linear version below is the first target (Kelvin: "simple proof first").

## 1. Setting

- Width n, depth L, cut after layer c, L' = L - c layers to go. x ~ N(0, I_n). z_l = W_l h_{l-1}, h_l = relu(z_l),
  h_0 = x. W_l iid entries N(0, 2/n), independent over l, no biases. Main case noiseless (sigma = 0); variant: absolute
  activation noise h_l = relu(z_l) + sigma xi_l.
- Front theta_f = (W_1..W_c), back theta_b = (W_{c+1}..W_L), independent. P_f = law of the cut state h_c(x).
- Back function g_b: R^n -> R^n, g_b(h) = a_L given h_c = h (noise integrated). Output neuron j: g_{b,j}.
- Estimand M_j(f, b) = E_{h ~ P_f} g_{b,j}(h) = E_x a_{L,j}. Per-net variance V_j(f, b) = Var_x a_{L,j}.
  Vbar = E_{f,b} (1/n) sum_j V_j(f,b).

## 2. Protocol class and the eps-rank (one-way, linear)

- A rank-s one-way linear protocol: phi: fronts -> R^s (computed from theta_f alone), psi_j: backs -> R^s (from theta_b
  alone); estimate <phi(f), psi_j(b)>. Error Err = E_{f,b} (1/n) sum_j (M_j(f,b) - <phi(f), psi_j(b)>)^2.
- eps-rank: r(eps) = min{ s : some rank-s protocol has Err <= eps^2 Vbar }.
  Err <= eps^2 Vbar is ARC's averaged matching-sampling criterion (E_W MSE <= eps^2 E_W Var) at N = 1/eps^2 samples.
- Fact 1 (PROVEN, Schmidt/Eckart-Young for Hilbert-Schmidt operators). Let T: L2(backs x [n]) -> L2(fronts),
  (T u)(f) = E_{b,j} M_j(f,b) u(b,j), singular values sigma_1 >= sigma_2 >= ... Then min Err over rank s =
  sum_{i>s} sigma_i^2, so r(eps) = min{ s : sum_{i>s} sigma_i^2 <= eps^2 Vbar }.
- Who is covered: any method whose answer is linear in a summary of the front computed without the back (moment or
  feature summaries with a linear readout, polynomial-chaos/Hermite truncations, kernel quadrature with front-only
  features, one-cut meet-in-the-middle with a linear join). Lifting: a decoder that is a polynomial of degree p in an
  s-number message is a linear protocol of rank C(s+p, p), so r(eps) also bounds such decoders. Not covered: nonlinear
  decoders (ARC's closures, particles/sampling), two-way or adaptive protocols, unrestricted joint computation (that is
  a time bound, out of reach unconditionally).

## 3. Target statement (one-way)

(T1) There are constants c0 > 0 and eps0 such that, for n large and (c, L') inside the window
1 << c, L' and c + L' << n ln(1/tol), for eps <= eps0:  r(eps) >= exp(c0 k*),
where k* = k*(n, L'; delta) = E_path N_surv(J_x; delta), N_surv(J; delta) = #{i : s_i(J) >= delta s_1(J)},
J_x = d a_L / d z_c along one path (delta = 0.01). Known (lyapunov lane, Theorem C, PROVEN for sigma = 0 on the
no-dead-layer event): J_x^T J_x = Jhat^T Jhat + r r^T with Jhat a product of independent Gaussian matrices of size
~ n/2 with N(0, 2/n) entries (critical at half width); Fuss-Catalan gives N_surv ~ n G(delta) / (2(L'+1)), e.g. 23-25
at n = 256, L' = 32, delta = 0.01 (measured 24). The surviving directions rotate with the input: one fixed block for all
inputs needs 185 of 195 directions (measured; averaging provably only hurts).
(T2, window) Past synchronisation (c or L' >> n ln(1/tol), where all inputs of a net merge onto one trajectory and one
direction per path survives) the premise should fail: r(eps) should collapse.
(T3, quantifier) From the ensemble bound to "almost every fixed net": for each fixed protocol, the per-net error
exceeds eps^2 V(f,b) on most nets.

## 4. Lemmas offered for cross-check (mine, ~23:30Z)

**Lemma B (exact harmonic structure). Claimed PROVEN; please check.** Assume sigma = 0. (i) Rotation invariance: W_{c+1}
has iid Gaussian entries, so W_{c+1} Q =d W_{c+1} for every orthogonal Q; hence the joint law of (g_b(Q h), g_b(Q h'))
equals that of (g_b(h), g_b(h')), and K(h, h') := E_{b,j} g_{b,j}(h) g_{b,j}(h') depends only on (|h|, |h'|, h.h').
(ii) Positive homogeneity: g_b(t h) = t g_b(h) for t > 0. So K(h, h') = |h||h'| kappa(u.u'), u = h/|h|, for a
continuous positive-definite zonal kernel kappa on S^{n-1} (depends on n, L'; exact at finite width).
(iii) Schoenberg/Funk-Hecke: kappa(u.u') = sum_d mu_d sum_{j <= N(n,d)} Y_dj(u) Y_dj(u'), mu_d >= 0, Y_dj orthonormal
spherical harmonics of degree d in L2(normalised uniform measure), N(n,d) = their number.
(iv) Let nu_f(du) = E_x[ |h_c(x)| ; h_c(x)/|h_c(x)| in du ] (radially weighted angular law of the front) and
a_dj(f) = integral Y_dj d nu_f. Then G(f, f') := E_{b,j} M_j(f,b) M_j(f',b) = sum_{d,j} mu_d a_dj(f) a_dj(f'),
so the sigma_i^2 of T are the eigenvalues of C = D_mu^{1/2} A D_mu^{1/2}, A = E_f a(f) a(f)^T, D_mu = diag(mu_d
repeated N(n,d) times). Also Vbar = kappa(1) E_{f,x}|h_c|^2 - tr C.
Consequences: (a) the back enters only through the 1-D function kappa (its Funk-Hecke spectrum mu_d), the front only
through the harmonic second moments A of its angular laws. (b) The per-path anisotropy that defines k* is averaged out
by the rotation invariance of the back ensemble: k* can enter r(eps) only through the shape of kappa.

**Lemma C (cap toy: an exact exponential). Claimed PROVEN; please check.** Fronts nu_f = a zonal law of angular
radius r centred at u_f (e.g. von Mises-Fisher), u_f uniform on S^{k-1}; back = one random ReLU layer with random
readout (kappa = arc-cosine kernel of degree 1). By isotropy A is diagonal: C has eigenvalue mu_d lambda_d(r)^2 with
multiplicity N(k, d), where lambda_d(r) is the Funk-Hecke multiplier of the cap law; Vbar = sum_d N(k,d) mu_d
(1 - lambda_d(r)^2). Hence r(eps) >= sum_{d <= D} N(k, d) >= C(D + k - 2, k - 1), where D = D(eps, r) is the largest
degree with sum_{d >= D} N(k,d) mu_d lambda_d^2 > eps^2 Vbar. With the ReLU decay N(k,d) mu_d ~ d^-4 (even d) and a
Gaussian-like cap lambda_d ~ exp(-d^2 r^2 / 2): D ~ sqrt(2 ln(1/eps)) / r for r >> eps^2, so r(eps) >= exp(k ln(1 + D/k))
for k <= D: exponential in the dimension k of the front variation, with rate ln(1 + D/k).

**Heuristic H (kernel limit; my current reading, may be wrong).** In the large-n limit at fixed c, L', the cut
operator behaves like rank one plus a "nugget" (independent fronts' states are at cosine ~ 1/pi, so the back's
responses to two fronts are nearly independent beyond the common part); the nugget's energy is spread over harmonics
of degree d with multiplicity ~ n^d / d!, with degree-d energy ~ the d-th Taylor coefficient of kappa at 1/pi. Then
r(eps) ~ n^{D(eps)} with D(eps) bounded in n at fixed eps: polynomial in n, not exp(c k*) with k* ~ n G / (2 L').
If right, (T1) fails at fixed eps for n >> L' D(eps) ln n, and the true exponential is exp(min(n, D(eps)) log(...)):
in the dimension of the fronts' variation, at a rate set by the back kernel's roughness (gate kinks: kappa has a
(1 - t)^{3/2} singularity at t = 1) relative to Vbar. Also: past synchronisation each front's law is nearly a point mass
at a random direction u_f, so r(eps) is the eps-rank of the evaluation matrix g_b(u_f), which need not collapse, while
the per-net problem's deep part becomes easy (one path). (T2) may then be false for the cut rank.

## 5. The crux lemma (Kelvin's sharpening, via main)

k* is already backward-selected (the survivors of the back's Jacobian, including its early layers), so a two-way
protocol reduces to one-way. The crux: the back half's value function v restricted to the k* relevant directions has at
least exp(c k*) independent components with respect to the fronts' distribution, through the random gates of the back's
early layers. Linear survival alone does not give this (a function of a k*-dim vector could depend only on its norm).
Formal candidate: for the fronts' laws projected on the relevant k-dim subspace, the eps-rank of the operator
(front law, back) -> E g_b is >= exp(c k) for k <= k*.

## 6. Questions for the co-prover

1. Give an independent proof strategy for (T1), or for the strongest version you think is provable, in the simplest
   setting where the exponential is real. Say which steps are routine and which are the bottleneck.
2. Check Lemma B and Lemma C line by line (statements, hypotheses, constants).
3. Is Heuristic H right? If so, (T1) is false at fixed eps as n grows; give the correct form (what replaces k*).
4. The crux lemma: formalise it so that it is true, provable, and implies a lower bound on r(eps).
5. For each step you cannot prove, the weakest precise conjecture that would close it, and whether it is a known open
   problem.
6. Quantifier: what the ensemble bound gives for a typical fixed net (per protocol), and what more is needed for
   "almost always".

## Erratum (appended 2026-10-31 ~00:35Z; the text above is the hashed original of 23:16Z, sha a79866...)
- Lemma C's displayed rank bound is superseded: it counted the null odd degrees (l >= 3) and the whole boundary
  degree. Exact count (gpt-6-astra, round 2): r = N_{<D} + ceil((T_{>=D} - delta)/e_D), N_{<D} = k + C(k+D-3, D-2) for
  even D >= 2. The asymptotic "D ~ sqrt(2 ln(1/eps))/s" is only a leading-log approximation; fixed-s, fixed-eps caps give
  no exponential in k. scratch/toy_cap.py now uses the closed-form spectrum (logs/toy_cap.log regenerated).
- Lemma B's normalisation: with He scaling and output averaging, K(h,h') = |h||h'| kappa(u.u') / n with kappa(1) = 1,
  and the front moments are E[(|h|/sqrt(n)) Y_l(h/|h|)] (final GPT-6 review, finding 3).
- Consequence (b) of Lemma B is too strong as worded: rotation invariance removes pathwise ORIENTATIONS, not singular
  values; k* alone does not determine r(eps), but k* is not "averaged out" (final review, finding 6).
- Heuristic H: its upper-bound half is proven (gpt-6-astra A3); the nugget reading is not established.
