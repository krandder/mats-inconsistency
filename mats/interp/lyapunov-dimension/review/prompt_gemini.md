You are an adversarial referee in random matrix theory and probability. Audit the mathematics of the report below
(no tools needed). For each numbered result (Theorems A, B, C, D, Corollaries C1, C2, the edge law G(delta), the
two-replica Wick recursion), say CORRECT, FIXABLE GAP (state the missing hypothesis or the error and the fix), or WRONG
(give a counterexample). Then check that each status label (PROVEN, PROVEN UNDER, HEURISTIC, MEASURED, CONJECTURE) is
no stronger than its support, and whether the conclusions in the summary and in section 10 follow. Be concrete and
terse; order findings by severity. End with a 3-line verdict.

Points that deserve particular scrutiny:
- Theorem C: masks depend on W_l only through g_l = W_l e_{l-1}; the claimed conditional independence of
  H_l = B_l^T W_l B_{l-1} from (g_l, past); the rank-one decomposition J = e_N r^T + Jhat with e_N^T Jhat = 0 and the
  resulting interlacing; whether D_0 (the cut mask) is handled right.
- Corollary C1: the Bartlett-decomposition degrees of freedom (m_l - i) for an (m_l - 1) x k Gaussian matrix, and
  random dimensions m_l.
- Corollary C2: using the fixed-N free-probability limit when L' is proportional to n.
- Theorem A: the chi-square bound and its claimed attainment.
- The edge law: the limit x/(N+1) -> f(theta) = (sin theta / theta) exp(theta cot theta) and the mass element
  (theta/pi) |(log f)'| d theta / (N+1).
- The Wick formulas: E||D W Y W^T D'||_F^2 = v^2 [p p' ||Y||_F^2 + o((tr Y)^2 + tr Y^2)] etc., with W iid N(0, v).

REPORT:

# How many directions must a description carry jointly? Lyapunov theory on ARC's critical ReLU nets (29 October 2026)

Lane: lyapunov-dimension (theory). Work directory: `work/lyapunov-dimension/`. Started 18:40Z. DRAFT, being written.

Kelvin's ask (via main): "On the non-gaussian dependence: I think we can prove, based on the lyapunov theory, how many
dimensions that are fully dependent are needed per layer. This would show maybe what is the best we can hope for?"

## Summary for Kelvin

Yes, the count can be proven, but it answers a narrower question than the one that limits the families.

1. **Per input path the count is small and does not grow with width.** Along one path, the downstream Jacobian of a
   He-ReLU net is exactly a product of independent Gaussian matrices of half the width (Theorem C, proven). At these
   depths its spectrum follows free probability, not Lyapunov exponents. About n G(delta) / (2 L') directions stay
   within a factor delta of the top: **24 at 256 x 64 with 32 layers to go, at delta = 1%** (the Lyapunov heuristic
   said 75). Measured on ARC's nets to within 1-2 directions, and the same at widths 128-512 at fixed L'/n.
2. **A description with one block for all inputs cannot use this.** Different inputs switch 6-10% of the gates
   differently in every layer, so each input amplifies different directions. Averaged over inputs, the sensitivity is
   spread over nearly every live direction: 185 of 195 are needed at 1e-4 (25 for one path). Theorem D (proven): averaging
   can only make it worse. Giving each of up to 64 input classes its own block recovers at most about 20% of the gap.
   Recovering it fully would take about 1e7-1e10 classes.
3. **What a block must carry is set by the state's collapse, weighted by sensitivity.** The first-order error of a
   block is the tail of S = Sigma^(1/2) M Sigma^(1/2); the best block conditions on whitened spread x sensitivity
   directions (Theorem B). This tail predicts the nongauss lane's measured errors across all its families
   (error = 2.7e-4 x tail, R^2 = 0.96). From it, at 256 x 64:
   - from one cut, 20-30 jointly carried directions reach 1e-5 and 40-60 reach 2e-6;
   - re-projecting at every layer, as an estimator must, the errors add up and about 100-130 directions are needed.
   The count is about the same at widths 128 and 256 (512 pending) at fixed L/n.
4. **Best we can hope for.** A non-Gaussian joint law in tens of dimensions per layer, changing with depth. Carrying it
   by particles is sampling; carrying it deterministically costs exponentially in its dimension. On this route nothing
   beats sampling at depth proportional to width unless the law in the block has a closed-form structure that the
   layers preserve. We see no sign of one (gates: 28-40 bits per deep layer, nearly independent pairwise).

Status labels per statement in section 11. Two adversarial reviews (GPT-6, Gemini) in section 12.

## 1. Setting and what "k directions carried jointly" means

- Nets: ARC's family as in `work/arc-noisy`: x ~ N(0, I_n), z_l = W_l h_{l-1}, a_l = relu(z_l), h_l = a_l + s_l eps_l,
  W_l iid N(0, 2/n), no bias, s_l = sigma r_l with sigma = 0.001 unless stated. Cut at layer c, L' = L - c layers to go.
- Downstream map F(z) = E[a_L | z_c = z] (downstream noise integrated). Along one path x the downstream Jacobian is
  J_x = d a_L / d z_c = D_L W_L ... D_{c+1} W_{c+1} D_c, with D_l the 0/1 gate masks of that path.
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
is finite, then |u . e|^2 <= chi2_B(Q, P) * u^T E_P[Cov_P(F | B^T z)] u for every unit vector u, and for small
chi2_B the bound is attained by some Q. So the worst-case error over families that are exact on the block and within
a chi-square ball is exactly governed by E_P Cov_P(F(z_c) | B^T z_c): the part of the downstream map that the block does
not determine.
*Proof.* E_Q F_B = E_P F_B = E_P F because F_B depends on z only through B^T z, whose law is the same under Q and P;
subtract. For the bound write e = E_P[(F - F_B)(dQ/dP - 1)] and apply Cauchy-Schwarz fibre by fibre (E_P[F - F_B | B^T z]
= 0 lets one subtract 1 from the density ratio). Equality: take dQ_b/dP_b = 1 + t u.(F - F_B) / ||u.(F - F_B)||, small t.
QED

So "how many directions must be carried jointly" has an exact answer in principle: the smallest k for which some k
linear features of the state make the downstream mean map nearly a function of them (a sufficient-dimension-reduction
dimension). Everything below estimates this through linear response, where the Lyapunov structure enters.

**Theorem B (linear response and the optimal block). PROVEN (linear algebra); the bound is PROVEN UNDER the
independence condition (I).** Couple P and Q so that z_Q = z_P + Delta with B^T Delta = 0, and let F be Lipschitz with
Jacobian J (piecewise linear without noise). Then |e| <= E|F(z_P + Delta) - F(z_P)| <= (E[Delta^T Jbar^T Jbar Delta])^(1/2),
Jbar = integral_0^1 J(z_P + t Delta) dt. (I): if Delta is independent of Jbar, this is (tr(Mbar Cov Delta))^(1/2) with
Mbar = E Jbar^T Jbar, close to M = E_x J_x^T J_x for small Delta. For the natural family that keeps the block exact and
replaces the residual of the linear regression on the block by an independent copy, Cov Delta = 2 Schur_B with
Schur_B = Sigma - Sigma B (B^T Sigma B)^-1 B^T Sigma (Sigma = Cov z_c). Then
  min over k-dimensional blocks of tr(M Schur_B) = sum_{j>k} lambda_j(S),   S = Sigma^(1/2) M Sigma^(1/2),
attained by the block functionals B = Sigma^(-1/2) V_k (V_k = top-k eigenvectors of S).
*Proof.* With C = Sigma^(1/2) B, Schur_B = Sigma^(1/2) (I - Pi_C) Sigma^(1/2) (Pi_C the projector onto range C), so
tr(M Schur_B) = tr S - tr(Pi_C S) >= tr S - sum_{j<=k} lambda_j(S) (Ky Fan), with equality for range C = span V_k. QED
Remark: the "bidirectional" family PBW of the nongauss lane conditions on projections onto orth(Sigma^(1/2) V_k), i.e.
C = Sigma V_k, which is not the optimum unless span V_k is Sigma-invariant. Section 6 measures the loss (2-4x).

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

Our regime (L' at most n/2) is the free one, not the Lyapunov one. Numbers (`logs/n1_gauss_a.log`,
`logs/fc_counts.log`; 4 products each):

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
- What survives of the heuristic: at fixed L'/n the count is O(1) in n (G(delta) n / L'), and logarithmic in 1/delta
  for small delta (G(delta) = 2 ln(1/delta) - O(ln ln(1/delta))). The constant is smaller than the Lyapunov one.
- The bulk law of Hanin and Jiang (2025; normalized squared singular values uniform on [0, 1] for any joint growth of
  n and N) does not resolve this: the surviving directions are a vanishing fraction n G / N of the spectrum.

## 4. ReLU along one path: an exact reduction to Gaussian products of half width

**Theorem C (single path). PROVEN for the noiseless net.** Fix a cut state z with a = relu(z) != 0 and let the
downstream weights W_1, ..., W_N be iid N(0, 2/n), independent of z. Let h_0 = a, h_l = relu(W_l h_{l-1}),
D_l = diag(1[W_l h_{l-1} > 0]), D_0 = diag(1[z > 0]), m_l = rank D_l, e_l = h_l / |h_l|, and
J = D_N W_N ... D_1 W_1 D_0 = d h_N / d z. Then
1. J a = h_N (Euler's identity for the positively homogeneous map);
2. J^T J = Jhat^T Jhat + r r^T for a vector r, where Jhat = P_{e_N-perp} J = B_N H_N ... H_1 B_0^T, B_l an orthonormal
   basis of range(D_l) inside e_l-perp, and H_l = B_l^T W_l B_{l-1} of size (m_l - 1) x (m_{l-1} - 1);
3. conditionally on (m_0, ..., m_N), the H_l are independent with iid N(0, 2/n) entries, and m_1, ..., m_N are iid
   Bin(n, 1/2).
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

**Corollary C1 (exponents). PROVEN.** For N -> infinity the angular exponents are
lambda_j = (1/2)[log(4/n) + E psi((m - j)/2)], m ~ Bin(n, 1/2) (Bartlett decomposition of H_l Q for an orthonormal
k-frame Q, as in Newman's proof), and the radial exponent (1/2)[log(4/n) + E psi(m/2)] lies above all of them: the
gaps are lambda_1 - lambda_j ~ (j - 1)/n, twice the gaps of a Gaussian product of width n.

**Corollary C2 (counts at finite depth). PROVEN UNDER the free limit for the half-width product (standard for
rectangular Gaussian products with aspect ratios -> 1; the edge convergence is the cited square case).**
N_surv(J; delta) ~ (n/2) nu_{L'}(delta) ~ n G(delta) / (2 (L' + 1)).

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
  (finite-n fluctuation of s_1 below the edge); their prediction "N_surv(e^-1) of 3-8, constant across widths at fixed
  L'/n" is what Corollary C2 gives (n G / (2 L') = G / (2 tau)).
- Noise at sigma = 1e-3 changes nothing visible (columns 1 and 2). The proof covers sigma = 0 only: with noise the
  radial direction leaks into the angular part by O(sigma) per layer.

So the Lyapunov answer for one path is: about n G(delta) / (2 L') directions, e.g. **24 at 256 x 64 with 32 layers to
go and delta = 0.01**, a third of the 75 of the heuristic (a factor 2 from the ReLU halving the effective width and a
factor 1.5 from the free-probability edge). It is O(1) in n at fixed L'/n.

## 5. Averaging over inputs: why a fixed block cannot use the per-path structure

A family has one block for all inputs. Its first-order error involves M = E_x J_x^T J_x (Theorem B), not one J_x.

**Theorem D (averaging can only hurt). PROVEN.** For every k-dimensional U,
E_x ||J_x P_{U-perp}||_F^2 >= min_U' E_x ||J_x P_{U'-perp}||_F^2 = sum_{j>k} lambda_j(M) >= E_x sum_{j>k} s_j(J_x)^2,
and for any partition of the inputs into classes with their own blocks U_c the class-averaged optimum lies between the
last two quantities. Equality on the right holds only if almost every path has the same top-k right singular subspace.
*Proof.* Ky Fan: sum_{j<=k} lambda_j(M) = max_U tr(U^T M U) = max_U E tr(U^T J^T J U) <= E max_U tr(U^T J^T J U). The
class statement applies the same inequality inside each class and once more across classes. QED

How much is lost is set by how fast the paths' Jacobians decorrelate. **Two-replica formula. HEURISTIC (annealed:
each layer's W is treated as fresh given the masks of the two paths).** tr(M^2) = E_{x,x'} ||J_x J_{x'}^T||_F^2. With
Y_l = A_l ... A_1 A'_1^T ... A'_l^T (A = D W of path x, A' of path x'), Wick's formula gives, for one layer with
p = |D|, p' = |D'|, o = |D and D'|, v = 2/n:
  E||Y_l||_F^2 = v^2 [p p' ||Y||_F^2 + o ((tr Y)^2 + tr Y^2)],
  E(tr Y_l)^2 = v^2 [o^2 (tr Y)^2 + o (||Y||_F^2 + tr Y^2)],  E tr Y_l^2 = v^2 [o^2 tr Y^2 + o ((tr Y)^2 + ||Y||_F^2)].
The shared ("coherent") part decays by the factor r_l^2 per layer, r_l = 2 o_l / n ~ 1 - theta_l / pi with theta_l the
angle between the two paths' states, so PR(M) = (tr M)^2 / tr M^2 ~ n / (1 + 2 sum_l r_l prod_{m<l} r_m^2), against
n / (1 + 2 L') for one path. At depth 32 of ARC's nets r ~ 0.92 per layer: the paths share their Jacobian structure for
about 6 layers, whatever L'. This is the same mask-overlap factor that governs the survival of mean shifts in the
hardness lane (its Proposition 6.8 and Corollary 6.9).

Measured (`logs/a2_theory.log`; 2048 paths, 256 x 64, net 113 unless stated; overlaps measured per layer):

| cut, L' | PR of one path: measured / n/(1+2L') | PR(M): measured / two-replica | k*_F(M) at 1e-1 / 1e-2 / 1e-4 (live dims) | k*_F(J^T J; 1e-4), one path |
|---|---|---|---|---|
| 32, 8 | 14.0 / 15.1 | 28.3 / 29.5 | 66 / 144 / 185 (195) | 69 |
| 32, 16 | 7.7 / 7.8 | 22.3 / 22.3 | 63 / 143 / 185 (195) | 43 |
| 32, 32 | 4.4 / 3.9 | 20.6 / 18.7 | 63 / 142 / 186 (195) | 25 |
| 48, 16 | 8.2 / 7.8 | 17.1 / 14.1 | 35 / 97 / 159 (165) | 44 |
| 32, 32 (net 114) | 3.9 / 3.9 | 13.7 / 14.2 | 44 / 114 / 172 (182) | 25 |

- The input-averaged sensitivity has a moderately concentrated top (PR 14-28) and a heavy tail: 185 of its ~195 live
  directions (the rest are dead neurons at the cut) are needed to leave 1e-4 of its trace outside, against 25 for one
  path. Its tail does not shrink with L' (8 to 32): beyond the ~6 coherent layers, more depth adds nothing a fixed
  block can use.
- So a fixed block cannot use the Lyapunov concentration of single paths. What concentrates the operator that matters
  for a family, S = Sigma^(1/2) M Sigma^(1/2), is the state's own collapse (section 6).

## 6. The state's spread, the optimal block, and the measured PB ladder

At depth the state is collapsed: with the mean (scale) direction removed, the participation ratio of Sigma is 11.7-12.5
at layer 32 and 9.4 at layer 48 (256 x 64). The first-order error of a block family is tr(M Schur_B) (Theorem B), so it
weighs the unblocked part of the state by the downstream sensitivity. Its share of the k = 0 value ("tail") for the
families the nongauss lane ran (`logs/a1_tails_a.log`; net 113, cut 32, L' = 32; the mean direction is carried exactly,
as in the polar families):

| k | optimal block (Sigma^(-1/2) V_k) | PB_k (top-k PCs) | PBW_k (orth Sigma^(1/2) V_k) | PBS_k (top-k of M) | state PCs alone |
|---|---|---|---|---|---|
| 8 | 0.23 | 0.38 | 0.29 | 0.31 | 0.31 |
| 16 | 0.093 | 0.18 | 0.15 | 0.13 | 0.17 |
| 32 | 0.028 | 0.080 | 0.058 | 0.045 | 0.073 |
| 64 | 0.0050 | 0.024 | 0.017 | 0.0089 | 0.021 |
| 128 | 1.4e-4 | 1.9e-3 | 1.5e-3 | 3.4e-4 | 1.7e-3 |

Net 114 is similar with a steeper fall (optimal: 0.042, 0.011, 0.0013 at k = 16, 32, 64; PB: 0.11, 0.039, 0.0099). The
tails hardly depend on L' (8, 16 or 32 give the same numbers to two digits): the block is set at the cut, by the state
and the first few coherent downstream layers. The PB tail follows the state's own PC tail; the sensitivity reweights it
only mildly. Kelvin's spread x sensitivity directions help when they are used as whitened functionals; PBW's
projection onto orth(Sigma^(1/2) V_k) keeps only part of the gain.

**Empirical law linking the tail to the measured error. MEASURED (not a theorem).** Across the nongauss lane's families
at cut 32 (four nets, net means; PB_0, PB_8, PB_16, PB_32 from its R1 summary, PBA_8, PBA_16, PBS_16, PBW_8, PBW_16,
PBW_32 from its Amendment 1 files), the output error at L = 64 is proportional to the first-order tail:
nMSE = C tail^beta with beta = 1.00, C = 2.7e-4, R^2 = 0.96, residuals within a factor 1.24 rms (10 points with
nMSE > 2e-6; `logs/a4_fit.log`). At cut 48 (two nets so far): beta = 1.09, C = 5.7e-4, R^2 = 0.92. PB64 (1.4e-6) lies a
factor 3 below the line, at the fit-set floor. So the first-order tail is the right currency for comparing blocks, and
the needed block follows from tail(k) <= epsilon / C:

| target nMSE (cut 32, 256 x 64) | optimal block, nets 113 / 114 | PB (PCs), nets 113 / 114 |
|---|---|---|
| 1e-5 (tail 0.037) | ~28 / ~20 | ~50 / ~33 |
| 2e-6 (tail 0.0074) | ~57 / ~40 | ~90 / ~70 (measured: PB64 is already at the floor) |

Reading of beta = 1 (HEURISTIC): the mean error behaves like a fixed fraction sqrt(C), about 2%, of the root-mean-square
pathwise error of the scramble. That is what a first-order term gives: the true residual is correlated with the
downstream gate pattern, the scrambled one is not, and E[J(z) Delta] picks up that correlation linearly in the residual's
amplitude. A purely second-order (curvature) mechanism would give beta = 2, which the data reject.

## 7. Input-dependent blocks (coordinator's question)

Would a block that depends on a coarse class of the input (a cell of the state, or a gate pattern), each class with its
own ~24 surviving directions, recover the per-path structure? No, not at any practical number of classes.

- **PROVEN (Theorem D).** With class-wise blocks the optimum lies between the global tail and the single-path tail; it
  reaches the single-path tail only if the members of each class share their top singular subspace.
- **Measured** (`logs/n4_*`; 256 x 64, cut 32, L' = 32, nets 113 and 114, 8192 Jacobians, blocks chosen on one half of
  each class and scored on the other). Sensitivity-only tail of M_c (energy outside the class's top-k), averaged over
  classes:

| classing (K classes) | k = 16 | k = 24 | k = 32 | k = 64 |
|---|---|---|---|---|
| one block for all (K = 1) | 0.33 / 0.26 | 0.25 / 0.19 | 0.20 / 0.14 | 0.098 / 0.056 |
| k-means on the top-8 PCs (64) | 0.28 / 0.21 | 0.22 / 0.15 | 0.18 / 0.12 | 0.086 / 0.046 |
| 4 x 4 x 4 cells on the top-3 PCs (64) | 0.30 / 0.23 | 0.23 / 0.17 | 0.18 / 0.13 | 0.092 / 0.051 |
| sign pattern of the 6 most uncertain gates of layer c+1 (64) | 0.31 / 0.23 | 0.24 / 0.17 | 0.19 / 0.13 | 0.098 / 0.052 |
| k-means on the masks of layers c+1..c+8 (64) | 0.28 / 0.21 | 0.22 / 0.15 | 0.18 / 0.12 | 0.088 / 0.047 |
| one path (no averaging) | 2.4e-3 / 1.9e-3 | 1.4e-4 / 1.1e-4 | 7.3e-6 / 5.7e-6 | 4e-12 / 4e-12 |

  (first number net 113, second net 114.) Classes gain at most about 20% in sensitivity. They do gain about 2x in the
  state-weighted tail (0.028 -> 0.014 at k = 32, net 113), through the smaller spread inside each class: the cells
  effect the boundary-family lane found, not recovered Lyapunov structure.
- **Why (two-replica formula, measured inputs; `logs/n5_256_64_c32_s113.log`).** Two paths' Jacobians stay coherent
  for about sum_l r^(2l) layers, r = 1 - (fraction of gates that differ between them). Globally the per-layer gate
  disagreement at layers 33-40 is 6-10% (rms angle between states 0.285 rad): about 6.7 coherent layers of 32. Inside
  the best 64 classes it is 4-7% (rms angle 0.21 rad): about 9 layers. Coherence over L' layers needs a disagreement
  below about 0.8 / L' (2.5% for L' = 32), i.e. an angle below about 0.08 rad.
- **How many classes (HEURISTIC).** The within-class angle fell from 0.285 to 0.21 rad for K = 64, i.e. like
  K^(-1/d) with d = 13.6, close to the participation ratio of the angular state (12). Extrapolating with
  K ~ (0.285 / theta_target)^13.6: half-coherence over 32 layers (theta_target = 0.8 pi / L' = 0.079) needs ~4e7
  classes, near-full coherence (theta_target = pi / (2 L') = 0.049) ~2e10; over 16 layers 3e3 to 2e6. The count is
  exponential in the effective dimension of the state's angular spread, and even half-coherence leaves the deep tail far
  from the single-path one (at L' = 8, where the global block is already half-coherent, k*_F(1e-4) is 185 against 69
  for one path).
- Downstream noise also flips gates of the same state (random-net-phases: 1.3% of (input, neuron) pairs at depth 32,
  sigma = 0.001, over 32 noise draws), so even exact knowledge of the cut state leaves some decoherence at this sigma.

Kelvin's bidirectional measure made input-conditional is the right object in principle; in practice the conditioning
needed to make it sharp is as fine as the gate patterns of the next ~L' layers.

## 8. Gate flips and distributional errors beyond linear response

- **Gate flips along one path. MEASURED** (`logs/n6_256_64_c32_s113.log`; 256 paths, perturbation of relative size eps
  along the residual (beyond the top-16 PCs) directions, same downstream noise):

| eps | gates flipped per layer | median pathwise nonlinearity abs(dF - J D) / abs(J D) | E abs(dF)^2 / E abs(J D)^2 |
|---|---|---|---|
| 1e-3 | 0.01% | 0.000 | 1.00 |
| 1e-2 | 0.08% | 0.17 | 0.98 |
| 3e-2 | 0.23% | 0.30 | 0.94 |
| 1e-1 | 0.75% | 0.53 | 0.80 |
| 3e-1 | 2.1% | 0.76 | 0.60 |

  Pathwise linear response fails from eps ~ 1e-2, while the scramble of a PB_k family moves paths by
  eps ~ (2 x state tail)^(1/2) ~ 0.2-0.6. But the mean-square response stays within 20-40% of linear response even
  there: flips re-route each path's response without changing its size much. This is why tr(M Schur_B) still works
  as a currency (section 6) although no single path is linear.
- **Where gate flips matter for k*.** Not as a correction to one path's response, but through the input average: the
  masks differ between inputs (6-10% of gates per layer at depth 32), and that is what turns 25 directions per path into
  ~185 for the input-averaged sensitivity (section 5). In that sense gate flips dominate k* for any fixed block.
- **Distributional errors. PROVEN + MEASURED.** For the mean the response is exactly linear in the law (Theorem A):
  e = integral of F d(Q - P). What is not linear is F. Theorem A gives the exact worst case (conditional variance of F
  given the block, times the family's conditional chi-square distance); the measured law of section 6 (nMSE = C tail,
  beta = 1) gives the typical case for these families. Errors confined to the unblocked directions grow 4-8x by the
  output (nongauss lane) because each later layer converts more of them into mean error through its own gates; the
  tails above already include this, since they use M of the whole remaining depth.

## 9. Width at fixed L/n

L/n = 1/4: 128 x 32, 256 x 64, 512 x 128, nets 113-116, cuts at L/2 and 3L/4 (`logs/a5_width.log`; tails from
vector-Jacobian products, `scratch/n7_pctail.py`, which reproduces the full-Jacobian tails to 3 digits at 256 x 64).

- **Per path.** Corollary C2 makes N_surv a function of L'/n only (n G / (2 L') = G / (2 tau)). The nongauss lane's T5b
  measured 13 / 8 / 4 directions within e^-1 at L'/n = 1/32, 1/16, 1/8, the same at all three widths; Fuss-Catalan at
  half width gives 12.4 / 6.3 / 3.2 (+1.5 from the finite-n edge).
- **One block for all inputs.** Net means of the PC tail and of the conversion factor C = nMSE(PB_k) / tail_PC(k):

| size, cut | PR of the state (mean direction removed) | tail_PC at k = 8 / 16 / 32 / 64 | C from PB8 / PB16 / PB32 |
|---|---|---|---|
| 128 x 32, 16 | 9.7 | 0.26 / 0.11 / 0.033 / 0.0034 | 5.8e-4 / 7.7e-4 / 6.9e-4 (PB64: 7.9e-4) |
| 128 x 32, 24 | 6.7 | 0.18 / 0.073 / 0.020 / 0.0014 | 6.6e-4 / 6.3e-4 / 4.8e-4 |
| 256 x 64, 32 | 11.3 | 0.30 / 0.14 / 0.057 / 0.015 | 3.7e-4 / 3.2e-4 / 2.8e-4 (PB64: 0.9e-4, at the floor) |
| 256 x 64, 48 | 7.7 | 0.20 / 0.086 / 0.029 / 0.0063 | (PB8-PB32 not in the files read) |
| 512 x 128, 64 | (pending) | (pending) | (pending) |
| 512 x 128, 96 | (pending) | (pending) | (pending) |

- C is constant across k at each size (the linear law of section 6 holds at 128 too) and falls by 2.2x from 128 to 256,
  while the tail at fixed k grows (the state spreads over more directions). The two compensate: the block that brings
  the error from the L/2 cut to 1e-5 is about 43 PCs at 128 and 45 at 256. The nongauss PB ladder says the same
  directly: PB32 at the L/2 cut gives 2.3e-5 / 1.6e-5 / 1.4e-5 at 128 / 256 / 512, and the block that reaches the
  floor is 64 / 64 / 64-128.
- So the needed joint dimension is O(1) in width at fixed L/n over this range. MEASURED. Why C falls like 1/n is not
  derived here (CONJECTURE: the mean error collects gate effects of individual neurons, each of relative size 1/n).

## 10. The best we can hope for

**10.1 How many directions, per layer.** Three answers, from the most to the least optimistic:
- one path, any width (Corollary C2): about n G(delta) / (2 L') directions keep a share delta of the top amplification,
  e.g. 24 (delta = 0.01) at L' = 32, n = 256; 42 at L' = 16; 69 at L' = 8; nearly all n/2 live ones for L' <= 2.
  O(1) in n at fixed L'/n, and growing like ln(1/delta).
- one block for all inputs, single cut (section 6, measured law): 20-30 (optimal whitened block) or 33-50 (PCs) for
  nMSE 1e-5 at 256 x 64 from layer 32; 40-57 or 64-90 for 2e-6.
- one block per layer, re-projected at every layer (what a deductive estimator does): the single-cut errors add up
  over the layers, as the nongauss lane's composition runs show (PB64 composed: 3-6e-5 against 1.4e-6 from one cut,
  about the sum of 32 single-cut errors weighted by their 4-8x growth). Holding the total at 1e-5 needs per-layer tails
  ~30x smaller, i.e. about 100-130 directions at 256 x 64 (PB128 composed reached 1.4e-6 and 1.2e-5 on two nets).

Where the budget should go: the tails hardly depend on L' (section 6), but the conversion factor C grows with L' (the
4-8x growth). So blocks should be larger where more layers remain, the opposite of the per-path picture (where L' small
means more directions). The last layer is cheap for a different reason: E[a_L] needs only the n one-dimensional
marginals of z_L. Already one layer earlier it needs n one-dimensional projections of the joint law of h_{L-1}, so
"exact last layers" does not extend cheaply beyond one layer.

**10.2 What carrying a k-dimensional joint law costs.**
- Particles: error ~1/N at N pushes. That is sampling, in k dimensions instead of n, but each push still needs the
  whole layer (the block is not closed: the next layer's block depends on the unblocked part through the gates, which
  is exactly the error measured above).
- Grids, tensor products, sparse grids for a generic law in k dimensions: cost exponential in k. With k = 20-130 this is
  out of reach. Even in the best analytic case, k* ~ c ln(1/epsilon) and spectral accuracy give cost
  (ln 1/epsilon)^(c ln 1/epsilon) = (1/epsilon)^(c ln ln(1/epsilon)), worse than sampling's 1/epsilon.
- Weighted quasi-Monte Carlo (Sloan and Wozniakowski 1998): the importance weights of the block directions are summable
  (participation ratio of S about 5-10), which is the condition for dimension-free QMC error bounds in smooth weighted
  spaces. But the downstream map is piecewise linear with gate kinks in every layer, outside those smoothness classes;
  on ARC's nets structured input designs gain constant factors (Kerdock 2.4x, work/arc-noisy), which is what we would
  expect here too.
- Moments or cumulants of order m: poly(k^m) numbers, but their push through a ReLU layer needs a closure, and the
  closure errors are what accumulate at the critical point (ARC's K = 2 and K3f, the random-net-phases lane).

**10.3 Plain statement.** On this route, nothing beats sampling at depth proportional to width unless the joint law
inside the block has a closed-form structure that the layer maps preserve (for example Gaussian conditionals on a few
discrete variables, or a tensor train whose rank stays bounded through ReLU layers). The theory makes the needed
dimension O(1) in width at fixed L/n, which is the good news; but it is tens of directions per layer for one cut and
about half the width at 256 x 64 once errors add over layers, and the law there is non-Gaussian and changing with
depth (the nongauss and boundary-family lanes). We see no sign of such a structure: the gate structure carries 28-40
bits per deep layer and is nearly independent pairwise (boundary-family), within-cell non-Gaussianity grows with depth,
and a 16-dimensional learned code reproduces paths only through a nonlinear decoder (nongauss T6). A positive result
would need either such a structure, found and proven to be preserved, or a non-local estimator (one that does not
re-project layer by layer), which this analysis does not cover.

## 11. Status of every statement

| statement | status |
|---|---|
| Theorem A: exact error identity; chi-square worst case | PROVEN |
| Theorem B: optimal block under the first-order bound = whitened top eigenvectors of S | PROVEN (algebra); the bound PROVEN UNDER independence (I) |
| Gaussian products: Newman's exponents; Fuss-Catalan limit; edge convergence | PROVEN (cited) |
| Edge law G(delta) for large N | PROVEN (asymptotics of the exact parametrisation), checked numerically |
| Free-probability counts match at L'/n = 1/8-1/2 | MEASURED (n = 128-512) |
| Theorem C: one ReLU path = product of independent Gaussian matrices of half width (interlacing) | PROVEN for sigma = 0; MEASURED for sigma = 1e-3 |
| Corollary C1: exponents with gaps (j-1)/n; radial exponent on top | PROVEN |
| Corollary C2: counts n G(delta) / (2 L') | PROVEN UNDER the free limit for the half-width product |
| Theorem D: input averaging and classes can only increase the tail | PROVEN |
| Two-replica formula for PR(M) | HEURISTIC (annealed); within 0-20% of measurements |
| nMSE = C x first-order tail, beta = 1 | MEASURED (4 nets, 10 points, one size, two cuts) |
| Class counts 1e7-1e10 for coherence over 32 layers | HEURISTIC (extrapolated from K <= 64) |
| Pathwise linear response fails from eps ~ 1e-2, mean square holds to 20-40% | MEASURED |
| Width scaling of the needed block (section 9) | MEASURED (PB ladder, three widths) + CONJECTURE for the mechanism |
| "Nothing on this route beats sampling without closed-form structure in the block" | CONJECTURE, argued in 10.2 |

## 12. Reviews

(pending)
