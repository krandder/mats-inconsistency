# Proofs and derivations used by the numerics lane (31 Oct 2026)

Model S, sigma = 1: h_l = relu(W_l h_{l-1}) + xi_l, W_l iid N(0, 2/n), xi_l iid N(0, I_n). Q_l = |h_l|^2 / n.

## P-R1. Radius reduction (PROVEN)
Given h_{l-1} with Q_{l-1} = q, each new coordinate is relu(sqrt(2q) Z_i) + xi_i with Z_i, xi_i iid N(0, 1)
(rotation invariance of the fresh row). Its square is distributed as c_i chi2_1 with c_i in {1, 1 + 2q} a fair coin
(astra r1 P4: for Z < 0 it is xi^2; for Z > 0, (sqrt(2q) Z + xi)^2 has the law of (1 + 2q) chi2_1 by the symmetry
(Z, xi) -> (-Z, -xi)). Group the M coordinates with c_i = 1 + 2q: U = their chi-square sum ~ chi2_M, V the rest ~ chi2_{n-M}.
Then n Q_l = (1 + 2q) U + V = S (1 + 2 q beta), S = U + V ~ chi2_n, beta = U / S. Given M = m, S and beta are
independent with beta ~ Beta(m/2, (n-m)/2) (Gamma-Beta independence); since S's law does not depend on m, S is
independent of (M, beta). Hence Q_l = (S_l / n) r_l with r_l := 1 + 2 Q_{l-1} beta_l, and
  r_l = 1 + (2 S_{l-1} beta_l / n) r_{l-1} = 1 + A_l r_{l-1},  A_l := 2 S_{l-1} beta_l / n.
Given M_l = m, S_{l-1} beta_l ~ chi2_m (Gamma(n/2) x Beta(m/2, (n-m)/2) = Gamma(m/2), independent factors), so A_l has
the law of A = (2/n) chi2_M; the A_l are iid (disjoint inputs) and A_l is independent of r_{l-1}; S_l is independent of
r_l. So the stationary law is Q = (S/n) r with r = 1 + A_1 + A_1 A_2 + ... (a perpetuity with B = 1), S ~ chi2_n
independent of r. With u = log(r - 1): u_l = log A_l + softplus(u_{l-1}) (u = -inf when A = 0, probability 2^-n).
Tail: Goldie/Kesten with E A = 1 gives P(r > x) ~ (1/E[A log A]) / x, and Breiman (E S/n = 1) transfers it to Q: the same
C_n = 1/E[A log A] as astra's form.

## P-R2. Kramers-Moyal expansion of the radius law (derivation; asymptotic validity checked numerically)
X = r/n obeys X' = A X + 1/n. Exact moments D_k = n E[(X' - X)^k | X = x] from the cumulants of A,
kappa_k(A) = (2/n)^k n kappa_k(Y), Y = Bern(1/2) Z^2 (kappa_2..4(Y) = 5/4, 11/2, 279/8):
D_1 = 1; D_2 = 5x^2 + 1/n; D_3 = (44x^3 + 15x^2)/n + 1/n^2; D_4 = 75x^4/n + (558x^4 + 176x^3 + 30x^2)/n^2; D_5 =
(2200x^5 + 375x^4)/n^2; D_6 = 1875x^6/n^2 (+ higher orders). The stationary Kramers-Moyal equation (zero flux)
sum_k (-1)^k/k! d^{k-1}(D_k P) = 0, solved order by order (P = P0 + P1/n + P2/n^2; at each order (5/2)(x^2 P_j)' - P_j
= source, solved by P_j = P0 f_j with f_j' = 2 source / (5 x^2 P0), normalised to zero mass) gives
  P0 = (2/5) x^-2 exp(-2/(5x)),  P1 = 13 (25x^2 - 2) exp(-2/(5x)) / (1875 x^4),
  P2 = (991875x^4 + 146250x^2 - 40220x + 1014) exp(-2/(5x)) / (4218750 x^6).
Mixing with S/n (cumulants 2^{k-1}(k-1)!/n^{k-1}) gives the CDF of Q/n stated in PREREG Part R (scratch/km_*.py,
sympy, residuals of each order checked to vanish identically). The large-x limits reproduce C_n/n:
F1 ~ -(13/75)/x (matches astra's 13/75), F2 ~ -(529/2250)/x, i.e. C_n = 2n/5 + 13/75 + 529/(2250n) + O(n^-2),
consistent with E[A log A] = 5/(2n) - 13/(12n^2) - 1/n^3 + ... (numerically: n^3 x residual -> -1.0010 at n = 1024).

## P-R3. Left-tail rate (derivation; HEURISTIC at the level of a Freidlin-Wentzell argument, checked numerically)
At large n, A has the large-deviation rate n I_A(a), I_A = Legendre transform of Lambda(t) = log((1 + (1-4t)^-1/2)/2)
(per-coordinate generating function of 2Y). For r = O(1) the stationary probability is exp(-n J(r)) with the
quasi-potential J(rho') = min(ln 2, min_rho [J(rho) + I_A((rho' - 1)/rho)]) and J(rho) -> 0.4/rho + b/rho^2,
b = 4 alpha3 - 1/5 = 0.0347 (alpha3 = 44/750 the cubic coefficient of I_A(1 - e) = e^2/10 + alpha3 e^3 + ...), from
expanding the Bellman equation to order rho^-3 (the optimal step at large rho is rho_prev - rho = 1). For Q:
K(q) = min_s [J(q/s) + (s - 1 - log s)/2].

## P-G1. Replica Gram chain (PROVEN)
R replicas share every W_l and have their own noise (shared between replicas a, b at layer l iff S_ab = 1). Condition on
the past (all replicas at layer l-1, all earlier weights and noises). The rows w_i of W_l are iid N(0, (2/n) I) and
independent of the past, so the vectors z_i = (w_i . h^(a))_{a=1..R} are iid over i with law N(0, 2G) where
G_ab = h^(a) . h^(b) / n: their law depends on the past only through G. The noises are iid over i with covariance S.
Hence G' = (1/n) sum_i v_i v_i^T, v_i = relu(z_i) + xi_i, is a function of G and of fresh randomness: G_l is a Markov
chain. Every rotation-invariant function of the replicas at layer 0 (norms, overlaps, and, after integrating the
output weights W_1, the closed forms in lib/gram_chain.py) therefore has the same joint law under the Gram chain and
under the network; Var and Sobol quantities of quenched (fixed-weight) laws are such functions of 2-4 replicas:
  tr Cov_xi(h|W) = E_xi |h1 - h2|^2 / 2,  tr Cov^2 = E[((h1 - h2).(h3 - h4))^2] / 4,
  total effect of layer j: E |f(xi) - f(xi with layer j resampled)|^2 / (2V) (replicas sharing all but layer j).
Conditional mean: E[G'_ab | G] = sqrt(G_aa G_bb) kappa(rho_ab) + S_ab, kappa(rho) = (sqrt(1-rho^2) + (pi - arccos rho) rho)/pi;
conditional covariance = (1/n) [E v_a v_b v_c v_d - E v_a v_b E v_c v_d]: the PDLT kernel and four-point-vertex
recursions with a noise source (Roberts-Yaida eq. 4.69-4.71 without noise).
Annealed quantities (averages over W) are computed exactly; per-net (quenched) ratios are not replica quantities and
can differ from ratios of annealed means when they vary across nets.

## P-G2. No PDLT fixed point for one replica (PROVEN)
E[Q' | Q] = Q + 1 and E[Q'^2] = (1 + 5/n) E[Q^2] + (2 + 4/n) E[Q] + 1 + 2/n exactly, so the kernel grows by one noise
unit per layer and the normalised vertex Var(Q)/E(Q)^2 grows without bound; at stationarity E Q = infinity (index-1
tail). The single-input recursions have no fixed point; normalised replica quantities (overlaps, radius ratios) do.

## P-I1. The eps-curve estimator is a lower bound on the mean dimension (PROVEN)
G(eps) = sum_S (1 - (1-eps)^|S|) V_S / V and 1 - (1-eps)^s <= s eps for s >= 1, eps in [0, 1], so G(eps)/eps <= sum_S |S| V_S/V = D,
with a deficit of order eps sum_S |S|(|S|-1)/2 V_S/V that grows with the spread of interaction orders.

## P-I2. Frozen gates make h_0 affine in the window noise (PROVEN)
If every gate in the window is a fixed function of the deep past (here the noiseless continuation), each layer is
h -> D_l W_l h + xi_l with D_l independent of the window noise, so h_0 = L_0 + sum_l L_l xi_l and its Sobol
decomposition over layer blocks has only first-order terms: D_h = 1 exactly. Nonlinear observables of h_0 (direction,
log-radius, output relu, gate pattern) keep interactions from the observation map alone.
