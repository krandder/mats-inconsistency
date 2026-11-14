# How many directions must a description carry jointly? Lyapunov theory on ARC's critical ReLU nets (29 October 2026)

Lane: lyapunov-dimension (theory). Work directory: `work/lyapunov-dimension/`. 18:40Z to about 21:20Z. Revised after
two adversarial reviews and two re-checks (section 12); addendum with a pre-registered oracle run (section 13).

Kelvin's ask (via main): "On the non-gaussian dependence: I think we can prove, based on the lyapunov theory, how many
dimensions that are fully dependent are needed per layer. This would show maybe what is the best we can hope for?"

## Summary for Kelvin

Yes for one input path, and that count is small. But it is not what limits a description that uses one block for all
inputs. And "the best we can hope for" does not follow as a theorem; what follows is a statement about the families
tested.

1. **Per input path: small, and stable across widths (proven in part, measured in part).** Along one noiseless
   path on which no layer dies, the downstream Jacobian of a He-ReLU net is, apart from one radial row, exactly a
   product of independent Gaussian matrices of about half the width (Theorem C, proven; "half" because about half the
   units are active at the cut and in every layer). So, in the limit of large width at fixed depth, its angular
   spectrum follows free probability (the Fuss-Catalan law at width n/2). That limit puts 23 directions within 1% of
   the top at 256 x 64 with 32 layers to go (the large-depth form n G(delta) / (2 L') of section 3 gives 25); measured on ARC's nets,
   about 24, including the radial row. The Lyapunov heuristic said 75. The counts hold within 1-2 directions at widths
   128-512 at fixed L'/n (measured; at depth proportional to width this is not proven).
2. **One block for all inputs cannot use this (proven direction, measured size).** Different inputs switch 6-10% of
   the gates differently in every layer, so each input amplifies different directions. Averaging over inputs can only
   spread the sensitivity (Theorem D, proven). Measured, it is spread over nearly every direction that some input
   activates: 185 of 195 are needed to leave 1e-4 of the energy outside, against 25 for one path. Separate blocks for up
   to 64 input classes lowered the averaged tail by 14-17% when scored on held-out paths (up to 30% scored in-sample,
   which flatters them), against a gap of three orders of magnitude to one path.
3. **What one fixed block needs is set mostly by the state's collapse, weighted by sensitivity.** To first order the
   error of a block is the tail of S = Sigma^(1/2) M Sigma^(1/2), and the best block conditions on whitened spread x
   sensitivity directions (Theorem B: the algebra is proven, the first-order step is an approximation). Proxy tails
   computed this way track the nongauss lane's measured errors across its families (R^2 = 0.96 on family means, 0.77
   per net; a descriptive association). Read through it, about 43 PCs bring the error from the L/2 cut to 1e-5 at
   widths 128, 256 and 512 alike (roughly stable, not a proven scaling). Run afterwards as a pre-registered oracle
   family (section 13), the optimal block reached 1e-5 with 16-24 directions from either cut at 256 x 64, where PCs
   needed 32-64 on seven of eight net-cut pairs; at equal size its error was lower than the PC block's by a geometric
   mean of 1.9-4.4 across nets, and the predicted errors held within a factor 2.5 (cut 32). Re-projected at every layer
   in finite-particle chains, a 64-direction optimal block still ended at 2-4e-5.
4. **The best we can hope for: not established.** In the families tested (an exact linear block, everything else
   approximated), bringing one cut's error to 1e-5 took a joint non-Gaussian law of tens of directions, and the
   composition runs suggest more once a description re-projects at every layer. None of the representations tried for
   such a law (particles, cells, moments) beats sampling. Whether another algorithm does is open. Routes this analysis
   does not cover include nonlinear latent codes, backward or observable-specific approximations, control variates and
   bias correction.

Status labels for every statement are in section 11. Section 12 lists what the two adversarial reviews (GPT-6,
Gemini) found and what changed.

## 1. Setting and what "k directions carried jointly" means

- Nets: ARC's family as in `work/arc-noisy`: x ~ N(0, I_n), z_l = W_l h_{l-1}, a_l = relu(z_l), h_l = a_l + s_l eps_l,
  W_l iid N(0, 2/n), no bias, s_l = sigma r_l with sigma = 0.001 unless stated. Cut at layer c, L' = L - c layers to go.
- Downstream map F(z) = E[a_L | z_c = z] (downstream noise integrated). Along one path x, with its own downstream
  noise draw xi, the pathwise Jacobian is J_x = d a_L / d z_c = D_L W_L ... D_{c+1} W_{c+1} D_c, with D_l the 0/1 gate
  masks of that path. M = E_{x, xi} J_x^T J_x is the pathwise sensitivity used throughout; it exceeds E J_F^T J_F
  (J_F = E_xi J_x, the Jacobian of F) by a positive-semidefinite noise term. M is the right operator for couplings that
  share the downstream noise between truth and family (Theorem B), and an upper surrogate for F itself.
- A family "carries the block B jointly" (B an n x k matrix of linear functionals) if its law Q of z_c has the exact
  joint law of B^T z_c; everything else may be wrong. The output error is e = E_Q F - E_P F (a vector over output
  neurons), scored like ARC as nMSE = |e|^2 / tr Var(a_L).
- Counting directions. For a matrix J with singular values s_1 >= s_2 >= ...: N_surv(J; delta) = #{k : s_k >= delta s_1}
  (directions whose amplification is within a factor delta of the top). For a PSD matrix A:
  k*_F(A; eta) = min{k : sum_{j>k} lambda_j(A) <= eta tr A} (directions needed to leave at most a fraction eta of the
  energy outside the block). The coordinator's heuristic is N_surv with Newman's exponents:
  N_surv ~ 1 + (2n/L') ln(1/delta), about 75 at n = 256, L' = 32, delta = 0.01.

## 2. Exact facts about errors confined outside a block

**Theorem A (exact error identity). PROVEN.** Let P be the true law of z_c and Q any law with the same law of B^T z_c.
Put F_B(z) = E_P[F(z) | B^T z]. Then e = E_Q[F - F_B]. If moreover chi2_B(Q, P) := E_P[chi^2(Q(. | B^T z), P(. | B^T z))]
is finite, then |u . e|^2 <= chi2_B(Q, P) * u^T E_P[Cov_P(F | B^T z)] u for every unit vector u. The bound is sharp
only locally: as the chi-square radius tends to 0, truncated, conditionally recentred perturbations along u.(F - F_B)
reach it to leading order; exact attainment at a fixed radius needs u.(F - F_B) bounded below on the support, which
fails for the piecewise-linear F here. So, locally, the worst-case error of families exact on the block is governed by
E_P Cov_P(F(z_c) | B^T z_c): the part of the downstream map that the block does not determine. The theorem is abstract:
the fitted empirical families of the nongauss lane need not have a finite chi-square distance to the noisy state law,
and no such distance is measured here, so it characterises the question rather than bounding those runs.
*Proof.* E_Q F_B = E_P F_B = E_P F because F_B depends on z only through B^T z, whose law is the same under Q and P;
subtract. For the bound write e = E_P[(F - F_B)(dQ/dP - 1)] (E_P[F - F_B | B^T z] = 0 lets one subtract 1; since the
block marginals agree, dQ/dP(z) = dQ_b/dP_b(z) with b = B^T z) and apply Cauchy-Schwarz in L2(P), using
E_P (dQ/dP - 1)^2 = chi2_B(Q, P). Equality needs dQ/dP - 1 = c u.(F - F_B) P-a.e. for one global constant c, a
probability density only if u.(F - F_B) is bounded below (for P = N(0, 1), no block and F(z) = z, 1 + cz < 0 somewhere
for every c != 0). Truncating at level T, recentring within each fibre and letting the radius go to 0 before T to
infinity gives the local sharpness. QED

So "how many directions must be carried jointly" has an exact answer in principle: the smallest k for which some k
linear features of the state make the downstream mean map nearly a function of them (a sufficient-dimension-reduction
dimension). Everything below estimates this through linear response, where the Lyapunov structure enters.

**Theorem B (first-order error and the optimal block). The bound and the optimisation are PROVEN; replacing
E[Delta^T Jbar^T Jbar Delta] by tr(M Cov Delta) is an approximation (A1) that is NOT tested here (section 8 tests
only the linearisation of single paths).** Couple P and Q so that z_Q = z_P + Delta with B^T Delta = 0 and the same
downstream noise, and let f be the downstream map at that noise, Lipschitz with Jacobian J (piecewise linear). Then
|e| <= E|f(z_P + Delta) - f(z_P)| <= (E[Delta^T Jbar^T Jbar Delta])^(1/2), Jbar = integral_0^1 J(z_P + t Delta) dt.
(A1): E[Delta^T Jbar^T Jbar Delta] ~ tr(M Cov Delta) with M = E_x J_x^T J_x (exact if Jbar were independent of Delta
with second moment M; Jbar does depend on Delta through the gates it flips). For the natural family that keeps the
block exact and replaces the residual r = z - Sigma B (B^T Sigma B)^-1 B^T z of the linear regression on the block by
an independent copy, B^T Delta = 0 and Cov Delta = 2 Schur_B with Schur_B = Sigma - Sigma B (B^T Sigma B)^-1 B^T Sigma
(Sigma = Cov z_c, centred variables, so E Delta = 0; Sigma is singular when neurons are inactive at the cut on every
path, and then
inverses are pseudo-inverses on range(Sigma)). Then
  min over k-dimensional blocks of tr(M Schur_B) = sum_{j>k} lambda_j(S),   S = Sigma^(1/2) M Sigma^(1/2),
attained by the block functionals B = Sigma^(-1/2) V_k (V_k = top-k eigenvectors of S).
*Proof.* With C = Sigma^(1/2) B, Schur_B = Sigma^(1/2) (I - Pi_C) Sigma^(1/2) (Pi_C the projector onto range C), so
tr(M Schur_B) = tr S - tr(Pi_C S) >= tr S - sum_{j<=k} lambda_j(S) (Ky Fan), with equality for range C = span V_k. QED
Remark: the "bidirectional" family PBW of the nongauss lane conditions on projections onto orth(Sigma^(1/2) V_k), i.e.
C = Sigma V_k. When S has a gap after its k-th eigenvalue this is optimal only if span V_k is Sigma-invariant (with ties,
e.g. S = I, other choices are optimal too). Section 6 measures the loss on proxies of these families (2-4x).

## 3. Linear Gaussian products: Lyapunov or free probability?

For J = W_N ... W_1 with W iid n x n, entries N(0, 1/n), and a fixed linear map, the block question is exact
(Eckart-Young-Mirsky): the best k-dimensional block is the top-k right singular subspace, its worst leftover
amplification is s_{k+1}, and the minimal block for relative operator error delta is N_surv(J; delta). Everything is in
the singular values. Two limits are known:

- **Lyapunov limit (N -> infinity at fixed n). PROVEN (Newman 1986).** (1/N) log s_k -> lambda_k =
  (1/2)[log(2/n) + psi((n-k+1)/2)], so lambda_1 - lambda_k ~ (k-1)/(2n) and N_surv(delta) ~ 1 + (2n/N) ln(1/delta).
  This is the coordinator's formula.
- **Free limit (n -> infinity at fixed N). PROVEN.** The squared singular values follow the Fuss-Catalan law FC_N
  (Penson and Zyczkowski 2011; density in Haagerup and Moller 2013), and s_1^2 tends to its edge
  x_+ = (N+1)^(N+1)/N^N (Nait Saada and Naderi 2024). Hence N_surv(delta)/n -> nu_N(delta) := FC_N((delta^2 x_+, x_+]).
- **Edge law (N large inside the free limit). PROVEN (asymptotics of the exact parametrisation; checked
  numerically).** (N+1) nu_N(delta) -> G(delta) = integral_0^{theta_delta} (t/pi) |(log f)'(t)| dt, with
  f(t) = (sin t / t) exp(t cot t) and log f(theta_delta) = 1 + 2 ln delta. So N_surv(delta) ~ n G(delta) / (N + 1).
  G = 0.79, 2.11, 6.34, 10.46 at delta = e^-1, e^-2, 0.01, 0.001, against 2 ln(1/delta) = 2, 4, 9.21, 13.82 for
  Lyapunov. (N+1) nu_N at N = 32: 0.81, 2.09, 5.88, 9.11. *Derivation:* put p = theta/(N+1) in the parametrisation
  x(p) = sin((N+1)p)^(N+1) / (sin p sin(Np)^N), rho = sin(p)^2 sin(Np)^(N-1) / (pi sin((N+1)p)^N); then
  x / (N+1) -> f(theta) and the mass element -> (theta/pi) |(log f)'(theta)| d theta / (N+1), uniformly on
  [0, theta_delta] for theta_delta < pi.

In our regime (L' between n/8 and n/2) the measured counts follow the free-probability formula, not Newman's. This is
an empirical statement: the theorems cover fixed N as n grows (or fixed n as N grows), not N proportional to n.
Numbers (`logs/n1_gauss_a.log` for the measured and Newman columns; its Fuss-Catalan column was computed with a wrong
density and is superseded by `logs/fc_counts.log`; 4 products each):

| n, N (= L') | N_surv at delta = e^-1 / e^-2 / 0.01 / 0.001: measured | Fuss-Catalan | Newman | coordinator 1 + (2n/N) ln(1/delta) |
|---|---|---|---|---|
| 128, 16 | 7.5 / 17.2 / 42.8 / 61.8 | 6.2 / 15.7 / 41.2 / 60.4 | 15 / 29 / 56 / 74 | 17 / 33 / 75 / 112 |
| 256, 32 | 7.8 / 17.8 / 46.2 / 71.8 | 6.3 / 16.2 / 45.6 / 70.6 | 16 / 30 / 64 / 90 | 17 / 33 / 75 / 112 |
| 256, 64 | 3.2 / 8.5 / 24.5 / 38.5 | 3.2 / 8.3 / 24.0 / 38.4 | 8 / 16 / 35 / 50 | 9 / 17 / 38 / 56 |
| 256, 128 | 2.0 / 4.2 / 12.2 / 20.5 | 1.6 / 4.2 / 12.3 / 20.0 | 4 / 8 / 18 / 27 | 5 / 9 / 19 / 29 |

- Why Newman over-counts here: at N < n the top singular value sits at the free edge, log s_1 ~ (1 + ln(N+1))/2
  (measured +2.10 at n = 256, N = 32; edge +2.24; Newman's N lambda_1 = -0.06). The top is boosted relative to the
  bulk, so fewer directions lie within delta of it. The measured profile log(s_k/s_1) at k = 2, 4, 8, 16, 32, 64
  (n = 256, N = 32) is -0.17, -0.44, -0.98, -1.82, -3.23, -6.15; Fuss-Catalan gives -0.19, -0.49, -0.96, -1.75, -3.19,
  -6.06; Newman -0.06, -0.19, -0.45, -0.97, -2.07, -4.54.
- What survives of the heuristic: in the iterated limit the count is G(delta) n / L', so O(1) at fixed L'/n, and
  logarithmic in 1/delta for small delta (G(delta) = 2 ln(1/delta) - O(ln ln(1/delta))), with a smaller constant than
  Lyapunov's. At L' proportional to n the counts are O(1) in the measurements (n = 128-512), not by a theorem; the
  predicted count is a few extreme singular values, for which convergence of the spectral distribution says nothing.
- The bulk law of Hanin and Jiang (2025; normalized squared singular values uniform on [0, 1] for any joint growth of
  n and N) does not resolve this: the surviving directions are a vanishing fraction n G / N of the spectrum.

## 4. ReLU along one path: an exact reduction to Gaussian products of half width

**Theorem C (single path). PROVEN for the noiseless net, on the event that no layer dies (every m_l >= 2; it fails
with probability at most N (n+1) 2^-n).** Fix a cut state z with a = relu(z) != 0 and no coordinate exactly 0, and let
the downstream weights W_1, ..., W_N be iid N(0, 2/n), independent of z (B_l is chosen by a fixed measurable rule
from g_l). Let h_0 = a, h_l = relu(W_l h_{l-1}),
D_l = diag(1[W_l h_{l-1} > 0]), D_0 = diag(1[z > 0]), m_l = rank D_l, e_l = h_l / |h_l|, and
J = D_N W_N ... D_1 W_1 D_0 = d h_N / d z. Then
1. J a = h_N (Euler's identity for the positively homogeneous map);
2. J^T J = Jhat^T Jhat + r r^T for a vector r, where Jhat = P_{e_N-perp} J = B_N H_N ... H_1 B_0^T, B_l an orthonormal
   basis of range(D_l) inside e_l-perp, and H_l = B_l^T W_l B_{l-1} of size (m_l - 1) x (m_{l-1} - 1);
3. conditionally on (m_0, ..., m_N), the H_l are independent with iid N(0, 2/n) entries; each m_l is Bin(n, 1/2) given
   the past (so on the survival event the m_l are iid truncated binomials). The half-width picture also needs the cut's
   active fraction m_0 / n near 1/2, which holds at ARC's cuts (measured 0.51-0.53).
So the singular values of J interlace with those of a product of independent rectangular Gaussian matrices:
s_k(Jhat) <= s_k(J) <= s_{k-1}(Jhat). Effectively a Gaussian product of width n/2 with entries of variance
2/n = 1/(n/2): critical, at half the width.

*Proof.* Write g_l = W_l e_{l-1}. Then W_l h_{l-1} = |h_{l-1}| g_l, so D_l and e_l = relu(g_l)/|relu(g_l)| are functions
of g_l, and g_l ~ N(0, (2/n) I) whatever the past, so m_l ~ Bin(n, 1/2) independently of the past. For v orthogonal to
e_0, D_0 v = B_0 B_0^T v (because e_0 lies in range D_0). Inductively, a tangent vector of the form beta e_{l-1} +
B_{l-1} c is mapped by D_l W_l to beta |h_l|/|h_{l-1}| e_l + (e_l . D_l W_l B_{l-1} c) e_l + B_l H_l c: the radial part
stays radial (D_l W_l e_{l-1} = relu(g_l)), and the angular part is B_l H_l c plus a radial leak. Hence
P_{e_N-perp} J = B_N H_N ... H_1 B_0^T, and J = e_N r^T + Jhat with e_N^T Jhat = 0, which gives the identity for J^T J
and, by rank-one interlacing, the singular-value sandwich. Given the past F_{l-1} (which fixes e_{l-1}, B_{l-1}),
W_l [e_{l-1}, B_{l-1}] has independent Gaussian columns, so W_l B_{l-1} is independent of g_l and hence of B_l; then
H_l = B_l^T (W_l B_{l-1}) has iid N(0, 2/n) entries given (F_{l-1}, g_l). Induct over l. QED

**Corollary C1 (exponents of the truncated chain). PROVEN.** Fix k and condition on every m_l > k (a chain whose
layers all keep at least k + 1 active units; at fixed n the physical network eventually violates this, so these are not
its infinite-depth exponents, and the conditioning changes the law of every m_l to Bin(n, 1/2) given m > k). For an
orthonormal k-frame Q, exactly for every N,
E log vol_k(H_N ... H_1 Q) = (N/2) sum_{i<=k} [log(4/n) + E(psi((m - i)/2) | m > k)],   m ~ Bin(n, 1/2)
(Bartlett: H_l Q_{l-1} is an (m_l - 1) x k Gaussian matrix whose squared k-volume is (2/n)^k prod_{i<=k} chi^2_{m_l - i},
as in Newman's proof). So within this chain the i-th angular exponent is (1/2)[log(4/n) + E(psi((m - i)/2) | m > k)]
for i <= k, the radial growth rate (1/2)[log(4/n) + E(psi(m/2) | m > k)] lies above them (the i-th angular exponent is
the (i+1)-th exponent of J), and the gap between the 1st and the i-th angular exponent is ~ (i - 1)/n, twice that of a
Gaussian product of width n. The conditioning event has
probability at least 1 - N (k+1) n^k 2^-n, so the statement is relevant for N much smaller than 2^n.

**Corollary C2 (counts at finite depth). For the angular part Jhat: PROVEN UNDER two assumptions, (i) the fixed-N free
limit for products of independent rectangular Gaussians with aspect ratios -> 1 (standard; bulk law FC_N at width n/2)
and (ii) convergence of the top singular value to the edge, proven for square Ginibre products (Nait Saada and Naderi)
and assumed here for the random sizes m_l. For the full J (with the radial row): HEURISTIC, see below. The use at L'
proportional to n is outside the fixed-N theorems and is supported numerically.**
N_surv(Jhat; delta) ~ (n/2) nu_{L'}(delta), whose large-L' form is n G(delta) / (2 (L' + 1)).
N_surv normalises by s_1(J), not s_1(Jhat). The radial row r = J^T e_N collects one O(1) leak from the angular part
per layer (e_l^T W_l B_{l-1} is a Gaussian vector of squared norm about 1), so |r|^2 is of order L' + 1, about 1/e of
the edge value e (L' + 1) of s_1(Jhat)^2 in the same units. Then s_1(J)^2 <= s_1(Jhat)^2 + |r|^2 moves the
normalisation by at most about 18%, which changes the count by about one direction (HEURISTIC); numerically the counts
of J and Jhat agree within one direction (table).

Checks (`logs/n2_256_64_c32_s113_sig0.001.log`, `..._sig0.log`, `logs/a2_theory.log`; 64 paths per row; "real" = the
Jacobian of ARC's net 113, "sim" = an independent simulated product with the same active counts m_l):

| 256 x 64, cut 32, L' | N_surv(0.01): real sigma = 1e-3 / sigma = 0 / sim / Fuss-Catalan at n/2 / Lyapunov at n/2 / coordinator | k*_F(J^T J; 1e-4): real / sim |
|---|---|---|
| 8 | 69.5 / 69.5 / 69.6 / 68.1 / 148 / 296 | 69.0 / 69.0 |
| 16 | 42.4 / 42.3 / 42.2 / 41.2 / 75 / 148 | 43.1 / 43.0 |
| 32 | 24.5 / 24.4 / 24.3 / 22.8 / 38 / 75 | 25.2 / 24.9 |

- Profiles agree too: log(s_k/s_1) at k = 2, 4, 8, 16, 32, 64 for L' = 32: real -0.32, -0.79, -1.56, -3.04, -5.92,
  -12.86; sim -0.32, -0.77, -1.63, -3.08, -5.95, -13.06. Nets 114 and cut 48 give the same within 1-2 directions.
- Width 512 (nongauss lane, T5b on CPU, cut 32, nets 113-114): N_surv(e^-1) = 13.3, 8.0, 4.3 at L' = 16, 32, 64
  against Fuss-Catalan at n/2: 12.4, 6.3, 3.2. The measured counts sit about 1.5 above Fuss-Catalan at every width
  (plausibly the finite-n fluctuation of s_1 below the edge; not checked); their prediction "N_surv(e^-1) of 3-8, constant across widths at fixed
  L'/n" is what Corollary C2 gives (n G / (2 L') = G / (2 tau)).
- Noise at sigma = 1e-3 changes nothing visible (columns 1 and 2). The proof covers sigma = 0 only: with noise the
  direction that sets the next masks, h_l / |h_l|, leaves range(D_l), so the exact structure breaks; we have no bound on
  the effect, only this measurement.

Relation to known results. Pennington, Schoenholz and Ganguli (2017, 2018) compute the spectrum of one input's
input-output Jacobian by free probability, treating masks and weights as free. For ReLU with Gaussian weights, Theorem C
shows that this treatment is exact at finite width apart from one radial direction, and identifies the answer: the
Fuss-Catalan law at width n/2. Hanin and Nica (2020) prove that log |J u| is asymptotically Gaussian for a fixed vector
u; the radial factor here is the product of iid (2/n) chi^2_{m_l} terms of Lemma 6.1 in
`outbox/hardness-mean-estimation-2026-10-29.md`.

So the Lyapunov answer for one path is: about n G(delta) / (2 L') directions, e.g. **24 at 256 x 64 with 32 layers to
go and delta = 0.01**, a third of the 75 of the heuristic (a factor 2 from the ReLU halving the effective width and a
factor 1.5 from the free-probability edge). It is O(1) in n at fixed L'/n in the iterated limit and in the
measurements at widths 128-512.

## 5. Averaging over inputs: why a fixed block cannot use the per-path structure

A family has one block for all inputs. Its first-order error involves M = E_x J_x^T J_x (Theorem B), not one J_x.

**Theorem D (averaging can only hurt). PROVEN.** For every k-dimensional U,
E_x ||J_x P_{U-perp}||_F^2 >= min_U' E_x ||J_x P_{U'-perp}||_F^2 = sum_{j>k} lambda_j(M) >= E_x sum_{j>k} s_j(J_x)^2,
and for any partition of the inputs into classes with their own blocks U_c the class-averaged optimum lies between the
last two quantities. Equality on the right holds if and only if one k-dimensional subspace maximises
tr(U^T J_x^T J_x U) for almost every path (a common maximiser; with a gap s_k(J_x) > s_{k+1}(J_x) this means a common
top-k right singular subspace).
*Proof.* Ky Fan: sum_{j<=k} lambda_j(M) = max_U tr(U^T M U) = max_U E tr(U^T J^T J U) <= E max_U tr(U^T J^T J U). The
class statement applies the same inequality inside each class and once more across classes. QED

How much is lost is set by how fast the paths' Jacobians decorrelate. **Two-replica formula. HEURISTIC (annealed:
each layer's W is treated as fresh given the masks of the two paths).** tr(M^2) = E_{x,x'} ||J_x J_{x'}^T||_F^2. With
Y_l = A_l ... A_1 A'_1^T ... A'_l^T (A = D W of path x, A' of path x'), Wick's formula gives, for one layer with
p = |D|, p' = |D'|, o = |D and D'|, v = 2/n:
  E||Y_l||_F^2 = v^2 [p p' ||Y||_F^2 + o ((tr Y)^2 + tr Y^2)],
  E(tr Y_l)^2 = v^2 [o^2 (tr Y)^2 + o (||Y||_F^2 + tr Y^2)],  E tr Y_l^2 = v^2 [o^2 tr Y^2 + o ((tr Y)^2 + ||Y||_F^2)].
The shared ("coherent") part decays by the factor r_l^2 per layer, r_l = 2 o_l / n ~ 1 - theta_l / pi with theta_l the
angle between the two paths' states. Starting from Y_0 = D_c D'_c (o_0 ~ r_0 n / 2) and keeping the leading terms,
PR(M) = (tr M)^2 / tr M^2 ~ n / (2 r_0 [1 + r_0 sum_{l=1}^{L'} r_l prod_{m<l} r_m^2]), against n / (2 + 2 L') for one
path (the Fuss-Catalan second moment 1 + L' on n/2 live directions). The table uses the full recursion. At depth 32 of
ARC's nets r ~ 0.92 per layer: the paths share their Jacobian structure for about 6 layers, whatever L'. The annealed
step treats each layer's W as independent of the two paths' masks, which the real masks are not (they are functions of
W); the table below is the check. This is the same mask-overlap factor that governs the survival of mean shifts in the
hardness report (its Proposition 6.8, and Proposition 6.9 / Heuristic 6.9' for the angle recursion).

Measured (`logs/a2_theory.log`; 2048 paths, 256 x 64, net 113 unless stated; overlaps measured per layer):

| cut, L' | PR of one path: measured / n/(2+2L') | PR(M): measured / two-replica | k*_F(M) at 1e-1 / 1e-2 / 1e-4 (live dims) | k*_F(J^T J; 1e-4), one path |
|---|---|---|---|---|
| 32, 8 | 14.0 / 14.2 | 28.3 / 29.5 | 66 / 144 / 185 (195) | 69 |
| 32, 16 | 7.7 / 7.5 | 22.3 / 22.3 | 63 / 143 / 185 (195) | 43 |
| 32, 32 | 4.4 / 3.9 | 20.6 / 18.7 | 63 / 142 / 186 (195) | 25 |
| 48, 16 | 8.2 / 7.5 | 17.1 / 14.1 | 35 / 97 / 159 (165) | 44 |
| 32, 32 (net 114) | 3.9 / 3.9 | 13.7 / 14.2 | 44 / 114 / 172 (182) | 25 |

- The input-averaged sensitivity has a moderately concentrated top (PR 14-28) and a heavy tail: 185 of its ~195
  directions with nonzero sensitivity (the rest are neurons inactive at the cut on every sampled path) are needed to
  leave 1e-4 of its trace outside, against 25 for one path. Its tail barely changes with L' (8 to 32): beyond the ~6
  coherent layers, more depth adds little that a fixed block can use.
- So a fixed block cannot use the Lyapunov concentration of single paths. What concentrates the operator that matters
  for a family, S = Sigma^(1/2) M Sigma^(1/2), is the state's own collapse (section 6).

## 6. The state's spread, the optimal block, and the measured PB ladder

At depth the state is collapsed: with the mean (scale) direction removed, the participation ratio of Sigma is 11.7-12.5
at layer 32 and 9.4 at layer 48 (256 x 64). The first-order error of a block family is tr(M Schur_B) (Theorem B), so it
weighs the unblocked part of the state by the downstream sensitivity. Its share of the k = 0 value ("tail") for
**proxies** of the families the nongauss lane ran (`logs/a1_tails_a.log`; net 113, cut 32, L' = 32). The proxies live
in the z chart with the mean direction projected out and use bases computed here; the actual families use
radius-weighted PCs of the direction z/|z| with an independent radius, and their PBW/PBS/PBA bases came from that
lane's own estimates. The proxy numbers are therefore indicative, not the families' own first-order errors.

| k | optimal block (Sigma^(-1/2) V_k) | PB_k proxy (top-k PCs) | PBW_k proxy (orth Sigma^(1/2) V_k) | PBS_k proxy (top-k of M) | state PCs alone |
|---|---|---|---|---|---|
| 8 | 0.23 | 0.38 | 0.29 | 0.31 | 0.31 |
| 16 | 0.093 | 0.18 | 0.15 | 0.13 | 0.17 |
| 32 | 0.028 | 0.080 | 0.058 | 0.045 | 0.073 |
| 64 | 0.0050 | 0.024 | 0.017 | 0.0089 | 0.021 |
| 128 | 1.4e-4 | 1.9e-3 | 1.5e-3 | 3.4e-4 | 1.7e-3 |

Net 114 is similar with a steeper fall (optimal: 0.042, 0.011, 0.0013 at k = 16, 32, 64; PB: 0.11, 0.039, 0.0099).
The tails change by up to about 20% between L' = 8 and 32 (net 114, optimal block at k = 16: 0.052 and 0.042): the
block is set mostly at the cut, by the state and the first few coherent downstream layers. The PB tail follows the
state's own PC tail; the sensitivity reweights it only mildly. Spread x sensitivity directions help most when used as
whitened functionals; the PBW projection keeps only part of the gain.

Why the heuristic's 75 came out close to the measured 64: coincidence. The heuristic over-counts one path's
directions about 3x (section 4), and a fixed block needs a few times more than one path (section 5) at this tolerance;
the two errors roughly cancel at 256 x 64.

**Association between the proxy tails and the measured errors. MEASURED, descriptive.** Pairing each nongauss family
with its proxy tail (`logs/a4_fit.log`, `logs/a4_robust.log`):
- cut 32, family means over the four nets (10 points with nMSE > 2e-6; PB_0, PB_8, PB_16, PB_32 are rounded means
  taken from that lane's summary, the rest from its Amendment 1 files): nMSE = C tail^beta with beta = 1.00,
  C = 2.7e-4, R^2 = 0.96. Dropping single points moves beta between 0.96 and 1.19.
- cut 32, per net (25 points): beta = 1.05, C = 2.9e-4, R^2 = 0.77, scatter a factor 1.6 rms.
- cut 48: family means (6 points) beta = 1.12, R^2 = 0.90; per net R^2 = 0.51 (21 points, nMSE > 2e-6) and 0.59 (20
  points, nMSE > 4e-6). Weak.
- PB64 lies a factor 3 below the cut-32 line, near the fit-set floor. The bootstrap in `a4_robust.log` resamples family
  means, not nets, so it does not measure between-net uncertainty; no held-out validation was done.
So the proxy tail orders the families, and at cut 32 the fitted line leaves a residual scatter of a factor 1.6 per net
(in-sample: no held-out nets, no between-net bootstrap); at cut 48 the relation is weaker. We have no derivation of the exponent: changing k changes the geometry of the perturbation, not
only its size, so beta near 1 does not identify a mechanism.

If the cut-32 association is taken at face value, the block needed for a target error follows from
C tail(k)^beta <= epsilon (`a1_tails` spectra, C = 2.7e-4, beta = 1):

| target nMSE (cut 32, 256 x 64) | optimal block, nets 113 / 114 (a prediction: not run) | PB (PCs), nets 113 / 114 |
|---|---|---|
| 1e-5 (tail 0.037) | 28 / 18 | 51 / 33 |
| 2e-6 (tail 0.0074) | 57 / 38 | 98 / 71 (measured: PB64 already reaches the floor) |

## 7. Input-dependent blocks (coordinator's question)

Would a block that depends on a coarse class of the input (a cell of the state, or a gate pattern), each class with its
own ~24 surviving directions, recover the per-path structure? Not with the classings tried, up to 64 classes, and the
measurements suggest it would need very fine classes.

- **PROVEN (Theorem D).** With class-wise blocks the optimum lies between the global tail and the single-path tail
  (ratio of expectations); it reaches the single-path tail only if each class has a common maximising subspace.
- **Measured** (`logs/n4_*`, `logs/n4b_*`; 256 x 64, cut 32, L' = 32, nets 113 and 114, 8192 Jacobians, ~64 per class
  and half at K = 64; all classes kept except 1-2 sparse gate classes, 99.6-100% of the weight). Sensitivity-only tail of
  M_c (energy outside the class's top-k), weighted over classes. The block is chosen on one half of each class and
  scored on the other ("cross-fitted"), which is biased upward relative to the classing's population optimum; scoring
  on the same half ("in-sample", net 113 only) is biased downward. The two suggest a range but do not guarantee one:

| classing (K classes) | k = 16 | k = 24 | k = 32 | k = 64 |
|---|---|---|---|---|
| one block for all (K = 1), cross-fitted | 0.33 / 0.26 | 0.25 / 0.19 | 0.20 / 0.14 | 0.098 / 0.056 |
| k-means on the top-8 PCs (64), cross-fitted | 0.28 / 0.21 | 0.22 / 0.15 | 0.18 / 0.12 | 0.086 / 0.046 |
| same, in-sample (net 113) | 0.25 | 0.18 | 0.13 | 0.048 |
| 4 x 4 x 4 cells on the top-3 PCs (64), cross-fitted | 0.30 / 0.23 | 0.23 / 0.17 | 0.18 / 0.13 | 0.092 / 0.051 |
| sign pattern of the 6 most uncertain gates of layer c+1 (64), cross-fitted | 0.31 / 0.23 | 0.24 / 0.17 | 0.19 / 0.13 | 0.098 / 0.052 |
| k-means on the masks of layers c+1..c+8 (64), cross-fitted | 0.28 / 0.21 | 0.22 / 0.15 | 0.18 / 0.12 | 0.088 / 0.047 |
| same, in-sample (net 113) | 0.25 | 0.18 | 0.13 | 0.049 |
| one path (net 113 trace-weighted, net 114 mean of per-path ratios) | 2.4e-3 / 1.9e-3 | 1.3e-4 / 1.1e-4 | 7.1e-6 / 5.7e-6 | 4e-12 / 4e-12 |

  (first number net 113, second net 114.) At k = 24 the best 64-class blocks give 0.22 / 0.15 cross-fitted against
  0.25 / 0.19 for one block (a gain of 14-17%), and 0.18 in-sample on net 113 (30%, biased favourably), against 1.3e-4
  for one path: a gap of three orders of magnitude. Classes do gain about 2x in the
  state-weighted tail (0.028 -> 0.013-0.014 at k = 32, net 113), through the smaller spread inside each class: the cells
  effect the boundary-family lane found, not recovered Lyapunov structure.
- **Why (two-replica formula, measured inputs; `logs/n5_256_64_c32_s113.log`).** Two paths' Jacobians stay coherent
  for about sum_{l<L'} r^(2l) layers, r = 1 - (fraction of gates that differ between them). Globally the per-layer gate
  disagreement at layers 33-40 is 6-10% (rms angle between states 0.285 rad): about 6.7 coherent layers of 32. Inside
  the best 64 classes it is 4-7% (rms angle 0.21 rad): about 9 layers. Keeping half of the 32 layers coherent needs a
  disagreement of about 2.5% per layer (angle ~0.08 rad); 64% needs 1.6% (0.05 rad); 90% needs about 0.3%.
- **How many classes: an illustrative extrapolation, not a claim.** Over K = 1-64 the within-class angle fell like
  K^(-1/d) with d ~ 13.6 (close to the participation ratio 12 of the angular state, though that is not a quantisation
  dimension). If that rate continued, half-coherence over 32 layers would take ~4e7 classes and 64% ~2e10. Nothing
  measured here supports extrapolating over six or more orders of magnitude; the point is only that the needed
  resolution is far beyond the classings tried.
- Downstream noise also flips gates of the same state (random-net-phases: 1.3% of (input, neuron) pairs at depth 32,
  sigma = 0.001, over 32 noise draws), so even exact knowledge of the cut state leaves some decoherence at this sigma.

Kelvin's bidirectional measure made input-conditional is the right object in principle; in the measurements the
conditioning needed to make it sharp has to resolve the gate patterns of many downstream layers.

## 8. Gate flips and distributional errors beyond linear response

- **Gate flips along one path. MEASURED** (`logs/n6_256_64_c32_s113.log`; 256 paths; a perturbation of size
  eps |z - mean| along a random direction drawn from the residual PCs beyond the 16th, weighted by their standard
  deviations; same downstream noise. These are not the families' actual scrambles):

| eps | gates flipped per layer | median pathwise nonlinearity abs(dF - J D) / abs(J D) | E abs(dF)^2 / E abs(J D)^2 |
|---|---|---|---|
| 1e-3 | 0.01% | 0.000 | 1.00 |
| 1e-2 | 0.08% | 0.17 | 0.98 |
| 3e-2 | 0.23% | 0.30 | 0.94 |
| 1e-1 | 0.75% | 0.53 | 0.80 |
| 3e-1 | 2.1% | 0.76 | 0.60 |

  Pathwise linear response fails from eps ~ 1e-2, while the scramble of a PB_k family moves paths by
  eps ~ (2 x state tail)^(1/2) ~ 0.2-0.6. The mean-square size of the response stays within 20-40% of its linearisation
  E|J(z) Delta|^2 up to eps = 0.3: flips re-route each path's response more than they change its size. This tests only
  the linearisation step; the second step of (A1), replacing E|J(z) Delta|^2 by tr(M Cov Delta), assumes Delta is
  independent of the path's Jacobian and is not tested here.
- **Where gate flips matter for k*.** Not as a correction to one path's response, but through the input average: the
  masks differ between inputs (6-10% of gates per layer at depth 32), and that is what turns 25 directions per path into
  ~185 for the input-averaged sensitivity (section 5). In that sense gate flips dominate k* for any fixed block.
- **Distributional errors. PROVEN + MEASURED.** For the mean the response is exactly linear in the law (Theorem A):
  e = integral of F d(Q - P). What is not linear is F. Theorem A gives the local worst case (conditional variance of F
  given the block, times the family's conditional chi-square distance); the association of section 6 describes these
  particular families. Errors confined to the unblocked directions grow 4-8x by the output (nongauss lane); in the
  telescoping identity of the hardness report (its Lemma 6.6) each later layer adds a signed local term, so this growth
  can come from new conversions at later gates as well as from transport. The tails above use M of the whole remaining
  depth, so they include both, to first order.

## 9. Width at fixed L/n

L/n = 1/4: 128 x 32, 256 x 64, 512 x 128, nets 113-116, cuts at L/2 and 3L/4 (`logs/a5_width.log`; proxy PC tails
from vector-Jacobian products, `scratch/n7_pctail.py`, which agrees with the full-Jacobian tails within 0.5% at
256 x 64, e.g. 0.0801 against 0.0804 for PB32 on net 113).

- **Per path.** Corollary C2 makes N_surv a function of L'/n only (n G / (2 L') = G / (2 tau)). The nongauss lane's T5b
  measured 13 / 8 / 4 directions within e^-1 at L'/n = 1/32, 1/16, 1/8, the same at all three widths; Fuss-Catalan at
  half width gives 12.4 / 6.3 / 3.2 (the measured counts sit about 1.5 higher at every width).
- **One block for all inputs.** Net means of the PC tail and of the conversion factor C = nMSE(PB_k) / tail_PC(k):

| size, cut | PR of the state (mean direction removed) | tail_PC at k = 8 / 16 / 32 / 64 / 128 | C from PB8 / PB16 / PB32 / PB64 |
|---|---|---|---|
| 128 x 32, 16 | 9.7 | 0.26 / 0.11 / 0.033 / 0.0034 / - | 5.8e-4 / 7.7e-4 / 6.9e-4 / 7.9e-4 |
| 128 x 32, 24 | 6.7 | 0.18 / 0.073 / 0.020 / 0.0014 / - | 6.6e-4 / 6.3e-4 / 4.8e-4 / (floor) |
| 256 x 64, 32 | 11.3 | 0.30 / 0.14 / 0.057 / 0.015 / 0.0010 | 3.7e-4 / 3.2e-4 / 2.8e-4 / 0.9e-4 (floor) |
| 256 x 64, 48 | 7.7 | 0.20 / 0.086 / 0.029 / 0.0063 / 0.0002 | (PB8-PB32 not in the files read) / 1.2e-4 (floor) |
| 512 x 128, 64 | 10.7 | 0.30 / 0.15 / 0.066 / 0.025 / 0.0059 | - / - / 2.1e-4 / 1.3e-4 (near floor) |
| 512 x 128, 96 | 9.2 | 0.22 / 0.12 / 0.048 / 0.017 / 0.0036 | - / - / 1.8e-4 / 2.2e-4 |

- C is roughly constant across k at each size (the association of section 6 holds at 128 and 512 too, where measured
  above the floor) and falls with width: 6.9e-4, 2.8e-4, 2.1e-4 from PB32 at the L/2 cut. The tail at fixed k grows
  with width, more slowly from 256 to 512 (k = 32: 0.033, 0.057, 0.066): the state spreads over somewhat more
  directions. The two compensate: **interpolating the tails log-linearly through the association, the block that brings
  the error from the L/2 cut to 1e-5 is about 44, 43 and 42 PCs at widths 128, 256 and 512.** At 2e-6 the same
  arithmetic gives about 66 (extrapolated beyond k = 64), 82 and 106, but there C rests on points near the fit-set floor
  (PB64 at 256 lies 3x below the line), and the nongauss lane's direct reading, the block that reaches the floor, is
  64 / 64 / 64-128. Its PB32 at the L/2 cut is 2.3e-5 / 1.6e-5 / 1.4e-5 at 128 / 256 / 512.
- So the needed joint dimension is roughly stable over widths 128-512 at fixed L/n: interpolated through the
  association of section 6 (factor-1.6 scatter) at 1e-5, within noise at 2e-6. Three widths cannot separate a bounded
  count from slow growth. Why C falls with width is not derived (CONJECTURE: the mean error collects gate effects of
  individual neurons, each of relative size about 1/n, and C levels off as the share of near-boundary neurons does).

## 10. The best we can hope for

What follows is about the class of descriptions studied here: carry an exact joint law on a linear block of k
directions, approximate everything else, and (for an estimator) re-project layer by layer. It is not a lower bound on
algorithms.

**10.1 How many directions, per layer, in that class.**
- One path (Corollary C2; iterated-limit theorem, measured at fixed L'/n): about n G(delta) / (2 L') directions keep a
  share delta of the top amplification, e.g. 23-25 (delta = 0.01) at L' = 32, n = 256; 41-42 at L' = 16; 68-70 at
  L' = 8; nearly all n/2 active ones for L' <= 2. Measured to be the same at widths 128-512 at fixed L'/n; grows like
  ln(1/delta).
- One block for all inputs, one cut: measured with the optimal block as an oracle family (section 13), 16-24
  directions reach nMSE 1e-5 from layer 32 or 48 at 256 x 64, against 32-64 PCs on seven of eight net-cut pairs. The association of section 6 had
  forecast 28 / 18 (nets 113 / 114) for the optimal block and 51 / 33 PCs; for 2e-6 it forecasts 57 / 38 or 98 / 71. About 43 PCs for 1e-5 at all three
  widths (interpolated). The measured PB ladder itself: PB32 gives 1.4-2.3e-5 at the L/2 cut, and the floor is reached
  at PB64 for widths 128 and 256 and at PB64-PB128 for 512 (within noise).
- One block per layer, re-projected at every layer (the deductive version of the same class). The nongauss lane's
  composition runs: PB64 composed from layer 32 reached 3-6e-5 against 1.4e-6 from one cut; PB128 reached 1.4e-6 and
  1.2e-5 on the two nets run. The output error is the signed sum of the layers' local errors (hardness report, Lemma
  6.6), so these need not add in nMSE; a forecast that assumes they add roughly incoherently, with per-layer errors
  following the cut-32 association, puts the block for a composed 1e-5 at roughly 90-130 PCs (65-95 optimal
  directions) at 256 x 64. This is a forecast, not a measured requirement.

Where the error budget should go: the proxy tails change little with L' (section 6) while the measured growth after a
cut (4-8x) is larger for cuts with more layers to go, so on this reading blocks should be larger where more layers
remain, the opposite of the per-path picture. The last layer is cheap for a different reason: E[a_L] needs only the n
one-dimensional marginals of z_L. One layer earlier it needs n one-dimensional projections of the joint law of h_{L-1}.

**10.2 What carrying a k-dimensional joint law costs, for the representations tried.**
- Particles: error ~1/N at N pushes, i.e. sampling in k dimensions; each push still needs the whole layer, because the
  block is not closed (the next layer's block depends on the unblocked part through the gates).
- Grids and tensor-product rules for a generic law: cost exponential in k, out of reach for k = 20-130. This is the
  cost of those schemes, not a lower bound.
- Moments or cumulants of order m: poly(k^m) numbers, but pushing them through a ReLU layer needs a closure, and closure
  errors are what accumulate at the critical point (ARC's K = 2 and K3f; the random-net-phases and hardness lanes).
- Cells with Gaussian or independent residuals and fitted blocks: the boundary-family and nongauss lanes (errors above
  the sampler at matched budgets, or oracle-only).
- Quasi-Monte Carlo and smoothing: left open. Whether the downstream map lies in a weighted function space with
  summable weights is not established (a bounded participation ratio does not imply it), and kinks do not by themselves
  exclude useful rates.

**10.3 Plain statement.** For the families studied, bringing one cut's error to 1e-5 took tens of jointly carried
directions at 256 x 64 (roughly stable over widths 128-512 at fixed L/n); the composition runs suggest more when a
description re-projects at every layer; and the law in the block is non-Gaussian and changes with depth (nongauss and
boundary-family lanes). None of the representations tried
for such a law beats sampling. One route to a positive result inside this class would be a structure in the block's
law that the layer maps preserve in closed form (for example Gaussian conditionals on a few discrete variables, or a
tensor train of bounded rank); we see no sign of one (gates carry 28-40 bits per deep layer and are nearly independent pairwise; within-cell
non-Gaussianity grows with depth). Whether any algorithm beats sampling at depth proportional to width remains open.
Routes outside this analysis: nonlinear latent codes (Theorem A allows any sufficient statistic; the nongauss lane's
16-dimensional learned code is one, though its law must still be carried and pushed), approximate latent closures,
backward or observable-specific approximations, conditional integration, control variates, adaptive cubature and bias
correction.

## 11. Status of every statement

| statement | status |
|---|---|
| Theorem A: exact error identity; chi-square bound, sharp only locally (radius -> 0) | PROVEN; not applied quantitatively to the fitted families (no finite chi-square measured) |
| Theorem B: the coupling bound and the optimal block (whitened top eigenvectors of S; PBW suboptimal given a gap) | PROVEN |
| Theorem B step (A1): E[Delta^T Jbar^T Jbar Delta] ~ tr(M Cov Delta) | APPROXIMATION, untested; its linearisation half MEASURED (section 8) |
| Gaussian products: Newman's exponents (N -> infinity); Fuss-Catalan law and edge (n -> infinity, N fixed) | PROVEN (cited) |
| Edge law G(delta) | PROVEN in the iterated limit (n, then N), checked numerically |
| Fuss-Catalan counts at L'/n = 1/8-1/2 | MEASURED (n = 128-512; Gaussian products and ARC's nets) |
| Theorem C: one noiseless ReLU path = radial row + product of independent Gaussian matrices of half width | PROVEN on the survival event; MEASURED for sigma = 1e-3 |
| Corollary C1: angular exponents with gaps (i-1)/n, radial rate on top | PROVEN for the truncated chain (all m_l > k) |
| Corollary C2: counts (n/2) nu_{L'}(delta) ~ n G(delta) / (2 L') | for Jhat PROVEN UNDER (i) the fixed-L' free limit and (ii) edge convergence for the random sizes; for J (radial row) HEURISTIC; at L' proportional to n MEASURED only |
| Theorem D: input averaging and classes can only increase the tail | PROVEN |
| Two-replica formula for PR(M) | HEURISTIC (annealed); within 0-20% of measurements |
| Association nMSE ~ C x proxy tail | MEASURED, descriptive (R^2 0.96 on means, 0.77 per net at cut 32; 0.51-0.59 per net at cut 48) |
| Needed blocks for 1e-5 / 2e-6 (section 6), optimal-block counts | FORECAST from the association; optimal block not run |
| Classes: up to 64 classes lower the averaged sensitivity tail by 14-17% cross-fitted (30% in-sample, biased) | MEASURED (two nets; no confidence bounds) |
| Class counts for coherence over 32 layers | ILLUSTRATIVE extrapolation, not a claim |
| Pathwise linear response fails from eps ~ 1e-2; mean square within 20-40% of its linearisation to eps = 0.3 | MEASURED (random residual directions) |
| Needed block roughly stable over widths 128-512 | INTERPOLATED through the association (three widths) |
| Composed-block forecast (90-130 PCs) | FORECAST; with the optimal block, 64 directions composed ended at 1.8-3.6e-5 in finite-particle chains (MEASURED, two nets) |
| Section 13: optimal block reaches 1e-5 with 16-24 directions vs 16-64 PCs (32-64 on 7 of 8 net-cut pairs); geometric-mean error ratios 1.9-4.4 vs PCs and 1.6-2.8 vs PBW at equal k | MEASURED, pre-registered (T-A to T-D supported at both cuts, four nets) |
| Section 13: composed errors, PBO vs PB | T-E REFUTED by the written rule, on bias estimates that look noise-dominated; raw composed errors lower for PBO in 8 of 8 comparisons; UNRESOLVED |
| Section 10.3 | a statement about the class studied; algorithmic complexity OPEN |

## 12. Reviews and what changed

Two adversarial reviews of the 19:46-19:53Z draft: Gemini 4 Argon (math audit, `review/review_gemini.md`) and GPT-6
via codex (high effort, read the code and logs, recomputed numbers; `review/review_gpt6.md`, raw `review_gpt6.raw`).
Both confirmed Theorem C's argument, Theorem D, the Schur/Ky Fan optimisation of Theorem B, the edge law G and the
three Wick formulas (GPT-6 checked them numerically), and every number they spot-checked (GPT-6: 14 checks).
Gemini: 8 findings; GPT-6: 4 critical, 14 major, 3 minor. Applied:
- Proportional depth (both, critical): the counts at L' proportional to n are now MEASURED, not proven; the theorem is
  the iterated limit. Summary, sections 3-4 and the status table say so.
- Radial row (GPT-6, critical): my first fix claimed |r|^2 = O(1); it is about L' + 1. The correction to the count is
  now argued heuristically (a shift of the normalisation by at most ~18%, about one direction) and checked numerically.
- Proxies (GPT-6, critical): the tails are computed for z-chart proxies, not for the nongauss lane's polar families;
  section 6 now says so and calls the tail-error relation a descriptive association, reports the per-net fit
  (R^2 0.77 at cut 32, 0.51-0.59 at cut 48) next to the family-mean fit, and drops the mechanism reading of beta ~ 1.
- Sampling lower bound (GPT-6, critical; Gemini): section 10 and the summary no longer say that nothing beats
  sampling; they describe the class studied, list the routes outside it, and leave algorithmic complexity open. The QMC
  argument was removed.
- Corollary C1 (both): restated for a truncated chain with one common conditioning (all m_l > k), not as infinite-depth
  exponents of the physical net.
- Theorem A (both): sharpness is local (radius -> 0); attainment needs one-sided boundedness; not applied to the
  fitted families, whose chi-square distance is unmeasured.
- Theorem B (both): (A1) is an untested approximation (section 8 tests only its linearisation half); coupling with
  shared noise; pathwise M distinguished from the Jacobian of the noise-integrated F; E Delta = 0 and pseudo-inverses
  stated; PBW's suboptimality needs a spectral gap.
- PR denominator (Gemini): n / (2 + 2 L') per path, and the leading-order PR(M) formula corrected; table values were
  already from the full recursion.
- Classes (GPT-6): cross-fitted numbers are upper bounds on each classing's population optimum; in-sample values (a
  lower estimate) and trace-weighted single-path tails added; the class-count extrapolation is labelled illustrative and
  its coherence fractions corrected (64%, not "near-full", at disagreement 1/(2L')).
- Composition (GPT-6): the 90-130 PC number is a forecast; errors add as signed vectors, not necessarily in nMSE.
- Width (GPT-6): "O(1)" replaced by "roughly stable over 128-512".
- Precision (GPT-6, minor): horizon dependence quantified (up to 20%), VJP vs full-Jacobian agreement stated as 0.5%,
  PC counts at 2e-6 corrected to 98 / 71, "dead" replaced by "inactive on every sampled path", logs added
  (`a4_fit.log`, `a5_width.log`).
A focused GPT-6 re-check of the revision (`review/review_gpt6_r2.md`, ~20:18Z) marked 12 of its findings and 5 of
Gemini's fixed and the rest partly fixed, and found no wrong number. Its remaining points were then applied: the class
gain is reported as 14-17% held-out (30% in-sample, biased), the in-sample and cross-fitted values no longer called a
bracket; the width counts are labelled interpolations; Corollary C2 is proven only for the angular part, the radial
row heuristic; the summary's 23 is identified as the finite-depth Fuss-Catalan value (25 from the large-depth form);
"needs" language for the class studied replaced by what the measured families took; the leave-one-out slope range is
0.96-1.19; cut-48 point counts corrected.
The addendum (section 13) had its own focused GPT-6 check (`review/review_gpt6_r3.md`, ~21:13Z): implementation of
PBO, application of the pre-registered rules, and 88 table entries verified; it corrected the composition counts (6 of
12 negative bias estimates; PBO lower in 8 of 8, not 7), removed an overclaim (composition "needs more than 64"
directions), and qualified the PC counts (16-64) and the error ratios (geometric means). All applied.
Not done: several composition chains per family (to settle T-E); an amplitude sweep for a fixed perturbation family; per-net bootstrap and held-out validation of the
association; a trace-weighted single-path tail for net 114; a joint-limit theorem for counts at L' proportional to n;
a bound on s_1(J) / s_1(Jhat).

## 13. Addendum: the optimal block as an oracle family (pre-registered; run 20:25-21:10Z)

Approved by the coordinator at ~20:20Z: run the whitened optimal block as an oracle family at 256 x 64 from the L/2 and
3L/4 cuts against PCA blocks, and check its composition. Pre-registration: `work/lyapunov-dimension/pbo/PREREG-PBO.md`
(hashes in `prereg_hashes.txt`, 20:24Z; every predicted number in `PREREG-PBO-predictions.txt`). Harness: a copy of the
nongauss lane's code (its `ng.py` and `ng2.py` byte-identical) plus `lib/ng3.py`.

- **The family.** PBO_k conditions on U = orth(Pi M R V_k): the optimal whitened functionals R^+ V_k of Theorem B,
  rewritten without an inverse, since R^+ v = Pi M R v / mu for an eigenvector v of S = R M R. In words: block on the
  sensitivity applied to the spread directions (PBW blocks on the spread directions R V_k themselves). Everything else as
  PBW: block drawn jointly, rest regressed on it and carried by independent exact marginals.
- **The proxy question settled first** (`pbo/logs/chart_tails.log`). Recomputed in the families' own sphere chart with
  the harness's sensitivity estimate, the first-order tails agree with the z-chart proxies of section 6 to about two
  digits (net 113, cut 32, k = 8 / 16 / 32 / 64: PB 0.38 / 0.18 / 0.081 / 0.024, PBO 0.23 / 0.094 / 0.028 / 0.005;
  section 6: 0.38 / 0.18 / 0.080 / 0.024 and 0.23 / 0.093 / 0.028 / 0.0050). GPT-6's finding 3 therefore does not change
  the numbers.

**Single cuts** (one L4, nets 113-116, fit 2^20 true paths, 2^21 family samples, truth 2^24; nMSE of E[a_64]; the
nongauss lane's floor rule; `pbo/logs/gpu_lyapdim1_stream*.log`):

| cut 32 (L' = 32) | net 113 | 114 | 115 | 116 | mean | pre-registered prediction, mean |
|---|---|---|---|---|---|---|
| PB16 / PB32 / PB48 / PB64 (PCs) | 4.9e-5 / 2.1e-5 / 1.4e-5 / 3.6e-6 | 7.2e-5 / 1.7e-5 / 1.8e-6 / 4.8e-7 | 3.2e-5 / 5.7e-6 / 1.4e-6 / 1.0e-6 | 2.6e-5 / 1.9e-5 / 1.6e-6 / 2.8e-7 | 4.5e-5 / 1.6e-5 / 4.7e-6 / 1.4e-6 | 3.6e-5 / 1.6e-5 / 7.5e-6 / 4.1e-6 |
| PBW16 / PBW32 | 4.7e-5 / 1.2e-5 | 7.2e-5 / 9.6e-6 | 2.2e-5 / 8.4e-6 | 2.5e-5 / 4.2e-6 | 4.1e-5 / 8.6e-6 | 3.0e-5 / 1.1e-5 |
| PBO8 / PBO16 / PBO24 / PBO32 / PBO48 | 8.0e-5 / 3.4e-5 / 9.8e-6 / 3.6e-6 / 1.7e-6 | 7.4e-5 / 1.3e-5 / 4.1e-6 / 1.3e-6 / 4.8e-7 | 2.5e-5 / 6.9e-6 / 2.7e-6 / 2.6e-6 / ~0 | 2.7e-5 / 1.1e-5 / 6.6e-6 / 5.2e-6 / 2.5e-6 | 5.1e-5 / 1.6e-5 / 5.8e-6 / 3.2e-6 / 1.2e-6 | 4.4e-5 / 1.6e-5 / 7.6e-6 / 4.3e-6 / 1.6e-6 |

| cut 48 (L' = 16) | net 113 | 114 | 115 | 116 | mean |
|---|---|---|---|---|---|
| PB16 / PB32 / PB48 / PB64 | 2.5e-5 / 7.5e-6 / 2.5e-6 / 1.3e-6 | 4.5e-5 / 8.4e-6 / 2.8e-6 / 5.3e-7 | 1.0e-5 / 2.0e-6 / 7.5e-7 / 2.8e-7 | 8.1e-6 / 6.2e-6 / 3.8e-6 / 8.6e-7 | 2.2e-5 / 6.0e-6 / 2.5e-6 / 7.5e-7 |
| PBW16 / PBW32 | 2.1e-5 / 4.9e-6 | 2.7e-5 / 5.7e-6 | 8.9e-6 / 2.7e-6 | 1.2e-5 / 2.5e-6 | 1.7e-5 / 4.0e-6 |
| PBO8 / PBO16 / PBO24 / PBO32 / PBO48 | 3.9e-5 / 1.1e-5 / 3.3e-6 / 2.8e-6 / 7.5e-7 | 6.5e-5 / 1.4e-5 / 7.6e-6 / 2.2e-6 / 3.4e-6 | 4.1e-5 / 8.9e-6 / 8.2e-7 / 8.0e-7 / 3.1e-7 | 1.7e-5 / 5.4e-6 / 5.6e-6 / 2.2e-6 / ~0 | 4.1e-5 / 9.7e-6 / 4.3e-6 / 2.0e-6 / 1.0e-6 |

Fit-set floor (EMP): 3e-8 (cut 32) and 2e-7 (cut 48) on average.

Pre-registered tests (all four rules met at both cuts):
- **T-A, the count. SUPPORTED.** Smallest block reaching 1e-5: PBO 24 / 24 / 16 / 24 directions at cut 32 (predicted
  32 / 24 / 24 / 24) against PB 64 / 48 / 32 / 48 PCs (predicted 64 / 48 / 48 / 48); at cut 48, PBO 24 / 24 / 16 / 16
  against PB 32 / 32 / 32 / 16.
- **T-B, against PCs at equal k. SUPPORTED.** PB_k / PBO_k error ratio, geometric mean over nets: cut 32: 3.1 at k = 16,
  4.5 at k = 32; cut 48: 1.9 and 2.3.
- **T-C, against PBW at equal k. SUPPORTED.** PBW_k / PBO_k: cut 32: 2.8 and 2.6; cut 48: 1.7 and 1.6 (one point
  below 1: net 116, cut 32, k = 32, 0.8).
- **T-D, calibration out of sample. SUPPORTED.** The association of section 6, fitted on other families, predicted every
  PBO point above the floor within a factor 2.5 at cut 32 (16 points: mean log10(measured / predicted) -0.03, range
  -0.34 to +0.39) and 14 of 15 within a factor 3.2 at cut 48 (mean +0.10, range -0.23 to +0.77).

So the first-order theory picks better directions than PCA or PBW, by about the predicted factors, and the count it
gives is about half the PCA count: **16-24 jointly carried directions reach 1e-5 from either cut at 256 x 64**, where
the PB ladder needed 32-64 on seven of the eight net-cut pairs (16 on net 116 at cut 48, where PB16 and PBO16 tie).
The chart check above compares fresh fits and a fresh sensitivity estimate (all 144 tail pairs within 4%): it shows the
proxies are numerically the right tails, not that the first-order approximation itself is exact.

**Composition** (re-projection at every layer from 32 to 63, nets 113 and 114, one chain per family and particle
count NC = 2^20 or 2^21; `pbo/logs/analyze_pbo.log`):
- By the written rule **T-E is REFUTED** (net 113, k = 32: B_PBO32 = 2.5e-4 against B_PB32 = -6.9e-5). The bias
  estimate B(64) = 2 C(2^21) - C(2^20) looks dominated by chain-to-chain noise: the EMP control, whose bias should be
  near zero, gives -7.2e-5 and -1.4e-5, and 6 of the 12 values (3 of the 8 family values) are negative. That the
  verdict reflects noise rather than projection bias is likely but not verified; one chain per setting cannot decide.
- Read directly, the composed errors at L = 64 at equal particle count are lower for PBO in all 8 comparisons (by
  1.08-6.5x). k = 32: 6.3e-5 vs 4.1e-4 and 1.6e-4 vs 1.7e-4 (net 113, NC = 2^20 and 2^21), 8.1e-5 vs 1.9e-4
  and 4.1e-5 vs 1.2e-4 (net 114). k = 64: 3.3e-5 vs 4.7e-5 and 2.0e-5 vs 5.7e-5 (113), 3.6e-5 vs 8.6e-5 and
  1.8e-5 vs 3.2e-5 (114). MC on the same paths: below 3e-6.
- PBO64 composed sits at 1.8-3.6e-5 in these finite-particle chains, far above the single-cut errors of comparable
  blocks (PBO48 at cut 32: 0.5-1.7e-6; PB64: 0.5-3.6e-6 on these nets) and above 1e-5. How much of that is projection
  bias and how much particle noise, and whether a larger block would reach 1e-5, is unresolved.
- Settling the composition comparison needs several chains per family; not done.

Cost: one on-demand L4, 0.72 h, $0.63 at the list-price ceiling (`pbo/GPU-LOG.md`; VM deleted, `gcloud compute
instances list` empty). A CPU replicate of the cut-32 runs was stopped after two families (about 50x slower than the
L4); its PB16 value on net 113 (5.1e-5) agrees with the GPU run (4.9e-5).

## Handoff

Code (`work/lyapunov-dimension/`; ARC's net code copied from `work/nongauss-family/lib/anet.py`, nothing there edited):
- `scratch/n1_gauss.py`, `lib/fc.py`, `scratch/fc_counts.py`, `scratch/a2_theory.py`: Gaussian products, Fuss-Catalan
  counts, the edge law G, two-replica predictions.
- `scratch/n2_relu.py`: per-path Jacobians of ARC's nets, simulated half-width products, M, Sigma, S (saves
  `runs/n2mat_*.npz`).
- `scratch/a1_tails.py`, `scratch/a4_fit.py`: first-order tails of the block families and the fit against the nongauss
  lane's errors (read-only from `work/nongauss-family/runs/gpu/`).
- `scratch/n4_classes.py`, `scratch/n5_within.py`: input-dependent blocks (cross-fitted) and within-class coherence.
- `scratch/n6_flips.py`: gate flips and pathwise nonlinearity.
- `scratch/n7_pctail.py`, `scratch/a5_width.py`: PC tails at any width from vector-Jacobian products; width table.
Logs in `logs/`, results in `runs/`, references checked in `notes/refs.md`, reviews in `review/`.
Nets: seeds 113-116 at 128 x 32, 256 x 64, 512 x 128 (the nongauss lane's pre-registered nets, used here only
descriptively; no fitting), sigma = 0.001 with that lane's noise scales. CPU only, nice 19, at most 2 processes.
