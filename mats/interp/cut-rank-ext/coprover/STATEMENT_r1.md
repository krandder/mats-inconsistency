# Cut-rank extension: nonlinear readouts and bounded back-and-forth (statement for the co-prover, round 1)

Lane: cut-rank-ext (theory). Fixed 2026-10-31 ~10:55Z. This file states the setting, what is already proven (by the
earlier cut-rank and kstar-frames lanes) and the questions. It contains no new lemmas.

## 1. Setting (unchanged from the cut-rank lane)

- Width n, depth L = c + d, cut after layer c (front = layers 1..c, back = layers c+1..L, d = L - c).
- x ~ N(0, I_n). z_l = W_l h_{l-1}, h_l = relu(z_l), h_0 = x, a_L = h_L. W_l iid entries N(0, 2/n), independent over
  l, no biases, noiseless.
- Front f = (W_1, ..., W_c), back b = (W_{c+1}, ..., W_L), independent. P_f = law of the cut state h_c(x).
- Back map g_b: R^n -> R^n, g_b(h) = a_L given h_c = h (positively homogeneous, piecewise linear).
- Estimand M_j(f, b) = E_{h ~ P_f} g_{b,j}(h) = E_x a_{L,j}. Per-net variance V_j(f, b) = Var_x a_{L,j}.
  Vbar = E_{f,b} (1/n) sum_j V_j.
- Criterion (ARC's averaged matching-sampling criterion at N = 1/eps^2 samples): an estimator est_j(f, b) succeeds if
  Err = E_{f,b} (1/n) sum_j (M_j(f,b) - est_j(f,b))^2 <= eps^2 Vbar.

## 2. Already proven (cut-rank lane, 30-31 Oct; kstar-frames lane)

- (K1) One-way LINEAR protocols: the front computes phi(f) in R^s from the front alone, the back computes psi_j(b) in
  R^s from the back alone, est_j = <phi(f), psi_j(b)>. The minimal s is r(eps) := min{ s : sum_{i>s} sigma_i^2 <=
  eps^2 Vbar }, sigma_i the singular values of the cut operator (T u)(f) = E_{b,j} M_j(f,b) u(b,j) (Eckart-Young).
- (K2) Harmonic structure: E_{b,j} g_{b,j}(h) g_{b,j}(h') = |h||h'| kappa(u.u')/n, u = h/|h|, kappa = sum_l a_l P_l
  (normalised Gegenbauer), a_l >= 0, sum a_l = 1, at every finite width and depth; the squared singular values are the
  eigenvalues of Lambda^{1/2} A Lambda^{1/2}, Lambda = diag(a_l / N(n,l)), A = E_f m(f) m(f)^T,
  m_{l,a}(f) = E_x[(|h_c|/sqrt n) Y_{l,a}(h_c/|h_c|)].
- (K3) r(eps) <= C(n+m, m) with m ~ 1/(eps^2 Vbar): polynomial in n at fixed depth and eps.
- (K4) Ray fronts h = R u (u uniform on the sphere or on the positive orthant, R >= 0 independent, Var R > 0), any
  fixed finite back depth d: r(eps) = Theta_{n,d}(eps^{-2(n-1)/3}) as eps -> 0. Mechanism: the gate kinks of g_b give
  kappa a (1-t)^{3/2} endpoint singularity; harmonic tail beyond degree D ~ D^{-3}; harmonic count ~ D^{n-1}.
- (K5) Genuine Gaussian ensemble, fixed (n, c >= 3, d): r(eps) = Theta(eps^{-2(n-1)/3}). Lower bound through a
  positive-probability event E (probability about 2^{-n^2}): W_{c-1} has one entrywise-positive row and all other rows
  entrywise negative, so h_{c-1}(x) = a(x) e_1; the first column of W_c is positive, so h_c(x) = a(x) v with
  v = W_c e_1 a ray whose direction is uniform on the positive orthant and independent of its norm. Then
  Err >= P(E) Err(conditional ray problem). Constants are astronomically small.
- (K6) If the radially weighted front laws have densities q_f in L^2 of a common angular measure with
  E_f ||q_f||^2 < infinity, then r(eps) = o(eps^{-2}).
- (K7) kstar-frames, Proposition 2 (sigma = 0): let J = d a_L / d z_c along an input's path and F its top-k right
  singular frame (k* ~ n G(delta)/(2(d+1)) survivors within factor delta of the top). Given a spectral gap, any
  d-dimensional subspace S computed from the input and W_1..W_c only satisfies
  E[tr(P_F P_S) | past] <= |P_S e|^2 + k (d - |P_S e|^2)/(m_c - 1) (e = state direction, m_c = active units): a
  forward computation sees the back's frame no better than chance plus one direction. Measured: two inputs' frames
  agree only while their states are within about 1.4/d radians.

## 3. Questions (Kelvin: "Of course, you should definitely seek the lane to prove that", about nonlinear readouts of a
## summary and bounded back-and-forth methods; "I am very happy with proving only the asymptotics")

Asymptotic in eps -> 0 is the deliverable (fixed n, c, d allowed; tiny constants acceptable if stated).

**Target 1, nonlinear readouts.** The front sends phi(f) in R^s (computed from the front alone) and the back outputs
est_j = R_j(phi(f), b) with R arbitrary in b. With arbitrary R and real phi one real number can encode everything, so a
regularity class is needed. Candidate classes:
- (a) R(., b) continuous (and/or phi continuous in the front weights or in the cut law);
- (b) R(., b) Lipschitz with constant Lambda (possibly allowed to grow as eps -> 0), phi bounded or Lipschitz;
- (c) bounded-precision messages: the front sends b bits;
- (d) the shape of ARC's K-th order closures: phi = the moments (or cumulants) of P_f up to order K, and R a fixed
  continuous map (a closure such as Gaussian/Wick-type) propagated through the back's layers;
- any other natural class you think matters (e.g. polynomial or analytic readouts).
For each class: does the precision scaling r ~ (1/eps)^{Theta(n)} survive? Prove the lower bound where it does; where
it breaks, give the construction and say exactly which step fails. Natural tools: nonlinear/manifold widths with
continuity (DeVore-Howard-Micchelli), Bernstein widths, stable widths and Carl-type inequalities
(Cohen-DeVore-Petrova-Wojtaszczyk), metric entropy and covering numbers, communication complexity. Check what is
known before deriving.

**Target 2, bounded back-and-forth.** Protocols with r rounds, each message s numbers (or bits), alternating between
the halves, under the same regularity classes; the back produces the estimate. Kelvin's point: k* is already the
back-selected set, so a backward message can at best name those directions; they rotate with the state, so even that
message is long. Use (K7) where it helps. Aim: a lower bound on total communication, or a construction showing where
bounded back-and-forth breaks the scaling.

**Target 3.** State the fixed-network quantifier ("almost every net": what the ensemble bound gives per protocol, what
more is needed) and the depth window for each result. Say exactly what remains outside (unrestricted joint
computation over both halves, which would need computational lower bounds).

## 4. Deliverables

Numbered sections, each claim marked PROVEN / PROVEN UNDER (stated hypothesis) / HEURISTIC / CONJECTURE. For each step
you cannot prove: the weakest precise conjecture that would close it, the conditional proof, its status (known open
problem or new; "unsure" if unsure), evidence for and against, and the cheapest refutation. Mark the bottleneck.
Literature: only references you are confident of; mark uncertain ones.
