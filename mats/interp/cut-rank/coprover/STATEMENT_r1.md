# Cut rank of critical random ReLU nets: the one-way statement (round-1 copy for the co-prover)

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

## 5. The crux lemma (Kelvin's sharpening, via main)

k* is already backward-selected (the survivors of the back's Jacobian, including its early layers), so a two-way
protocol reduces to one-way. The crux: the back half's value function v restricted to the k* relevant directions has at
least exp(c k*) independent components with respect to the fronts' distribution, through the random gates of the back's
early layers. Linear survival alone does not give this (a function of a k*-dim vector could depend only on its norm).
Formal candidate: for the fronts' laws projected on the relevant k-dim subspace, the eps-rank of the operator
(front law, back) -> E g_b is >= exp(c k) for k <= k*.
