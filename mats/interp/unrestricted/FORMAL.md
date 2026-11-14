# FORMAL.md v2.1: unrestricted estimators of the mean of random critical ReLU nets (31 Oct 2026; v2 11:40-11:48Z, v2.1 after astra round 2, 12:00Z)

> **Superseded in parts.** Sections 1-8 are v2.1 (12:00Z). Where they conflict with the errata in sections 9-10
> (GPT-6 review A; astra round 3), the errata win, and the final report
> (outbox/unrestricted-estimators-2026-10-31.md, after GPT-6 review B) supersedes both. In particular: C0 in section
> 8 is refuted (section 10); the threshold-circuit match in 1c is withdrawn; the spin-glass condition fails from
> depth 9; the 4.1x design cap is a plug-in estimate, not a bound.


Lane: unrestricted estimators (bead pool-uwmcvo). v1 (written by 11:16Z, before gpt-6-astra's round-1 answer) is
frozen in coprover/FORMAL_v1_as_written.md. v2 merges astra's round 1 (coprover/astra_r1_answer.md, "a-r1 n" = its
section n) and the measurements so far. Provenance tags: [C] Claude, [A] astra, [C+A] both independently.
Labels: PROVEN, PROVEN UNDER X, COMPUTED (numerics of a stated model), MEASURED, HEURISTIC, CONJECTURE.

## 0. Setting
- W_l iid N(0, 2/n), no bias, x ~ N(0, I_n), h_l = relu(W_l h_{l-1}), output f_W(x) = a_L in R^n; sigma = 0 unless
  stated. mu(W) = E_x f_W(x); V = E_W tr Var_x f_W. R(A) = E_W |A(W) - mu(W)|^2. m_n = E chi_n. By positive
  homogeneity f_W(x) = |x| g_W(x/|x|), so mu = m_n E_u g_W(u); V_sph = m_n^2 E_W tr Var_u g_W is the per-sample error of
  radius-exact ("RB") MC, and V = (n - m_n^2) + V_sph with n - m_n^2 -> 1/2 [A, a-r1 2.7].
- Cost unit: one forward pass of one input with schoolbook products (2 n^2 L).

## 1. Determinism is free on average, uniformly and explicitly (T1)
**1a [A], PROVEN (real model with unit-cost special functions).** QR-reduce three rows of W_1: B_1 = W_1 Q. Then
mu(B) = mu(W), and the 3n - 6 angular coordinates removed are uniform and independent of B (Bartlett decomposition
of Gaussian rows). Their conditional cdfs give independent uniforms U, V in [0,1)^n; X_i = Phi^{-1}((U + i V) mod 1)
(coordinatewise) are pairwise independent N(0, I_n), because (u, v) -> (u + i v, u + j v) mod 1 preserves Haar
measure on the torus for i != j. A(W) = (1/N) sum_i f_B(X_i) has R(A) = V/N exactly. Cost N passes + O(n^2 + n N).
**1b [C], PROVEN (any weight law with a density; floor and inverse-cdf as primitives; fixed n >= 2, L, N).** Seeds
U_c = frac(2^k w_c) from 2n weight entries: (W, U) -> (W, uniform independent of W) in distribution as k -> infinity
(Riemann-Lebesgue), so the same pairwise-independent construction has risk -> V/N; uniform integrability holds because
frac(2^k(w_c + i w_{n+c})) has a wrapped density bounded uniformly in k [A r2]. This is convergence at fixed size, not a
uniform rate; with b-bit inputs a precision schedule is needed.
**1c [C], PROVEN.** The same holds for random threshold circuits on uniform Boolean inputs whose first-layer weights are
iid symmetric: flipping input i and column i of W_1 preserves the acceptance probability, and canonicalising the signs
of one row frees n uniform bits independent of the canonical instance (|r_i| and sign(r_i) are independent; the other
rows stay iid symmetric given the signs). With the input-permutation symmetry (sort the columns by a statistic) about
log2(n!) more bits are free; 2n bits seed 2^n pairwise-independent points x_i = a + i c over GF(2^n).
**Consequences (PROVEN, logical) [C+A].**
1. In ARC's random-MLP instance, "deterministic" excludes nothing, even uniformly. This is typically-correct
   derandomisation (Goldreich-Wigderson 2002; Hoza 2019) made exact by the ensemble's symmetry.
2. Neither determinism, nor unrestricted algebraic degree (Chebyshev doubling T_{2m} = 2 T_m^2 - 1 on a weight mapped
   into (-1, 1) gives rapidly oscillating values with an arcsine limit law, which a further cdf makes uniform; my v1
   said "uniform" with plain arithmetic, which was wrong [A r2]), nor continuity or stability in W [A: fixed-design
   evaluation is itself continuous] separates "mechanistic" from sampling. Any useful restriction must be representational (an
   auditable expression language, e.g. instance-independent nodes plus specified moment operations, with no
   weight-derived seeds [A]) or a bounded interface (the sibling lane).
3. Hence "no deterministic estimator beats sampling on average" is the same question as "is MC optimal among all
   algorithms on average". This needs a simulation argument for an arbitrary randomized estimator A(W; R) [A r2];
   PROVEN [C] in the real model with digit-splitting primitives: from the 3n - 6 free directions of 1a draw a Haar
   Q''' and, from disjoint digits, R; set W' = (B_1 Q''', W_2, ..., W_L). Then W' has the law of W, R is independent of
   W', and mu(W') = mu(W) (input rotation), so A(W'; R) has A's average error, at an extra O(n^3).

## 2. The baseline must be the best sampler [A, a-r1 5], PROVEN
A batch of n inputs through an n x n layer is an n x n matrix product; with Strassen it costs O(n^{log2 7}) instead
of 2 n^3. So batched MC beats serial MC by n^{3 - omega} at equal arithmetic cost: "no algorithm beats serial MC by
more than a constant" is false. Conjectures must compare with the best implementation of sampling (B(T) in C0
below), or fix a cost model that charges a matrix-vector product n^2 (what every experiment here does).

## 3. Fixed point sets (deterministic QMC, designs) (T2)
**3a [C+A], PROVEN (exact).** For a rule A = sum_i c_i f_W(u_i) on unit vectors (radius exact: c_i = m_n w_i),
  R(A) = sum_{ij} c_i c_j k(u_i . u_j) - 2 m_n q sum_i c_i + m_n^2 q,   k(t) = E_W <f_W(u), f_W(v)>,  q = E k(U . V),
and in spherical harmonics R = q (sum c_i - m_n)^2 + sum_{l >= 1} lambda_l sum_m (sum_i c_i Y_lm(u_i))^2.
A deterministic rule and its random rotation have the same average error. (This does not equate a deterministic
Sobol set with scrambled Sobol: scrambling changes the pairwise geometry [A r2].)
**3b [A], PROVEN.** k(t) = sum_d b_d t^d with b_d >= 0, sum_d b_d = 1 (Mehler expansion of the downstream function of
the first-layer pre-activations). For positive weights summing to one, the degree-d (even) term of R is at least
m_n^2 b_d (1/N - tau_d), tau_{2s} = E (U . V)^{2s} = (2s-1)!! / [n (n+2) ... (n+2s-2)], so
  R(A) >= (m_n^2 / (2N)) sum_{s : tau_{2s} <= 1/(2N)} b_{2s},   gain over RB-MC <= 2 V_sph / (m_n^2 sum_{...} b_{2s}).
For N <= n^p only even degrees above about 2p count. So the gain of every positive-weight design is bounded by the
inverse of the high even-degree power-series mass of the kernel, relative to the angular variance.
**3c [C], HEURISTIC ceiling, MEASURED inputs.** A t-design that is MC-like above degree t gains about
1/(1 - f_1 - ... - f_t), f_l the fraction of angular variance at harmonic degree l. Measured f_1 + f_2 = 0.75 (8 x 32)
and 0.79 (32 x 8) on dev nets: a 2-design ceiling of about 4-5x; the registered M2 sweep measures it up to 256.
**3d [C], PROVEN UNDER the GP limit, fixed n and L, two-sided spectral bounds and the cited cubature theorem
[unverified constants].** With Gegenbauer eigenvalues ~ l^{-(n+2)}, the best N-point rule has squared error of order
N^{-1-3/(n-1)}: a gain of order N^{3/(n-1)} (not uniform at L = alpha n)
(N^0.43 at n = 8, N^0.012 at n = 256). [A] adds: for Gaussian priors adaptivity cannot help (posterior covariance is
data-independent), but the finite-width prior is not Gaussian, so this does not settle adaptive evaluation.

## 4. Restricted classes (T4)
**4a [C+A], PROVEN.** Degree-D polynomials in the standardised weights with instance-independent coefficients have
optimal error sum_{k>D} E_k (Hermite tail of mu). With S(rho) = Cov(mu(W), mu(W_rho)) = sum_{k>=1} rho^k E_k,
  max(0, S1 - rho^{-D} S(rho)) <= sum_{k>D} E_k <= (S1 - S(rho)) / (1 - rho^{D+1})   [two-sided form: A].
**4b [A r2], PROVEN in the infinite-width mean-field model.** S_L(rho) = m_n^2 (E_T K_{rho,L}(T) - 1/pi), with
K_{rho,0}(t) = t, K_{rho,l+1} = kappa(rho K_{rho,l}), T = U . V for independent uniform U, V. At fixed depth the
small-delta law is linear with normalised slope ~ L/(4(1 - 1/pi)) (mean Hermite degree of order L); for delta -> 0
with L delta^{1/3} -> infinity, 1 - Corr(mu(W), mu(W_rho)) ~ 3.27 delta^{2/3} (crossover at delta ~ L^{-3}), and the
two-scale bound T_D >= [(1 + rho^{-D}) F(rho) - rho^{-D} F(rho^2)]_+ (F = S(1) - S) gives a Hermite tail
T_D = Theta(S(1) D^{-2/3}) for 1 << D << L^3. Against RB-MC a degree-D estimator is worth at most about
(V_sph/S(1)) D^{2/3} samples, and V_sph/S(1) = Theta(L^{-2}) in that limit. My v1 heuristic (attractor direction) had
the right exponent but the wrong crossover (I placed the 2/3 window at [1e-4, 1e-2] for L <= 32; it starts later).
**4c MEASURED (registered M1, 48 net pairs, 2^14 Sobol points per net).** At n = 32 the measured D(delta) =
1 - S(1 - delta)/S1 matches the mean-field formula within about 5% at L = 4 for every delta (2.2e-5 .. 0.48), and at
L = 16-32 for delta <= 1e-2 once normalised by the measured D(1) (logs/nstab_mf_compare.log). The mean Hermite
degree is about 2.2 (L = 4), 7 (L = 16), 14 (L = 32). Class bounds (registered estimator): degree-1 estimators are
worth at most 3.6 (n = 32, L = 4) and about 0.5 (L = 16-32) MC samples; degree-4 at most 2.3 (L = 16) and 1.2 (L = 32);
degree-8 at most 14.5 and 5.7 (normalised by D(1): 6.5 and 3.9 for degree 4, 17 for degree 8 at L = 32). At n = 16 and
L >= 16 the second moments are dominated by rare high-energy nets (D(1) = 1.6-12 instead of 1) and are unresolved.
**4d [C+A], PROVEN.** For any A and rho: 2 ||A - mu|| >= ||mu - mu_rho|| - ||A - A_rho||. Overlap-gap arguments do not
apply as such: the acceptable estimates form a Euclidean ball (connected, convex) [A]; mu is continuous in W.
**4e [C+A], PROVEN.** Statistical queries: the mean (and a tail) is itself one query; SQ lower bounds count queries,
so they are vacuous unless queries are restricted or charged.
**4f [A], PROVEN.** Cell selection: kappa_n = E_W sum_s p_s(W_1)^2 = E_T (1/2 + arcsin(T)/pi)^n is exponentially
small, so any K first-layer gate cells chosen after reading W cover expected mass <= sqrt(K kappa_n); a method that
integrates selected cells exactly and drops the rest has R >= c_alpha n for polynomial K at L = alpha n.
**4g [A], PROVEN UNDER a mesh hypothesis.** A cubature exact on cells inside one linear region, with M cells of
diameter h ~ M^{-1/(n-1)}, has error <= C_W M^{-2/(n-1)} (coherent signs, MSE M^{-4/(n-1)}); with centred uncorrelated
cell errors (a HEURISTIC hypothesis for our algorithm) MSE ~ M^{-1-3/(n-1)} [A r2, correcting my v1 exponent, which
used an O(h^2) error on crossing cells; it is O(h)]. Both are upper bounds; the constants grow with depth.
**4h [C], PROVEN UNDER a region-count bound.** Exact enumeration of linear cones costs R(W) poly(nL) polylog(1/eps);
if E R <= (C n L)^{n-1} (Hanin-Rolnick order; proven for nets with random biases, not bias-free ones), exact
deterministic integration at fixed width is polynomial in L and beats MC for eps below about (C n L)^{-(n-1)/2}.

## 5. Necessary conditions for any estimator [A, a-r1 8], PROVEN
- R(A) >= S (1 - r^2) + |E A - m|^2 (r = correlation of A with mu across nets, S = E|mu - E mu|^2); gain G over N-sample
  MC needs r^2 >= 1 - V/(N G S), and S is of order n at L = alpha n.
- Final-row identity: mu_j = (1/2) w_j . m_h + (1/2) E_x |w_j . h_{L-1}| (relu(z) = (z + |z|)/2); the odd part in w_j is
  exactly linear. Hence R(A) >= (1/(2n)) sum_j E |mhat_j - m_h|^2 with mhat_j = n E_{w_j}[w_j A_j]: any accurate
  estimator implicitly encodes the full penultimate mean vector in its final-row linear response.
- Depth recursion [A r2]: with B_{r,L} = W_L ... W_{r+1}, E[B_{r,L}^T mu_L | W_{1:r}] = mu_r, so any A induces an estimator
  muhat_r = E[B_{r,L}^T A | W_{1:r}] of the depth-r mean with R(A) >= 2^{-(L-r)} E|muhat_r - mu_r|^2 (sharp for this
  extraction; informative only O(log n) layers back at polynomially small error; muhat_r need not be cheap).
- Control variates: A = E g_W + mean(f - g) has R = E tr Var(f - g)/M; a growing gain needs a shrinking residual
  variance after charging g's construction and integration.

## 6. Barriers and reductions (T3) [C+A]
- PROVEN (logical): an unconditional "no algorithm beats the best sampler by more than C" at budgets N >= M^delta
  passes would be a super-linear time lower bound for an explicit average-case problem in a general model (once
  precision, accuracy and model are fixed). HEURISTIC (state of the art, not a theorem [A r2]): no current technique
  gives such bounds. It would not by itself imply P != NP or circuit lower bounds [A].
- IKW / Chen-Tell are worst-case; on random weights derandomisation is free (T1), so they do not transfer. No
  worst-to-average reduction is available: random self-reducibility needs a low-degree polynomial estimand, and
  mu is not one (positively 1-homogeneous in each layer; already mu = |w|/sqrt(2 pi) at L = 1). That exact circuit
  encodings have probability zero proves little by itself (every continuous instance does); robust-neighbourhood
  reductions are not excluded, just unknown [A r2].
- Natural proofs: would need a large constructive property of truth tables; Gaussian measure on weights is not that
  largeness condition; not applicable as such [A].

## 7. Analogies (coordinator's request; notes/analogies.md)
- TC0 derandomisation: close in the worst case (ReLU nets of depth O(d) contain depth-d TC0; deterministic CAPP for
  poly-size depth-2 TC0 is open and tied to lower bounds). On random weights the determinism question disappears
  (T1, 1c). Even at depth 1, PRG-based deterministic counting costs more than sampling as eps -> 0, so ARC's bar is
  stronger than derandomisation. The positive depth-1 theory stops at depth 2 for ReLU nets of width n.
- Convex-body volume: the oracle separation (DFK vs Elekes / Barany-Furedi) has a proven ReLU analogue in the black-box
  model (hardness lane Thms 2.1, 2.3). White box: mu is a sum of Gaussian first moments of explicit polyhedral cones;
  deterministic volume of explicit H-polytopes is open in general and solved only for structured polytopes via
  correlation decay (Gamarnik-Smedira), whose ReLU counterpart (contraction of backward value functions) fails at
  criticality. A white-box separation is believed false at polynomial scale and false on average.
- Spin glasses: the mean is the infinite-temperature end. COMPUTED: for the infinite-width kernel kappa_L of L ReLU
  layers, (kappa_L'')^{-1/2} is concave on (0, 1) at L <= 8; astra proved non-concavity at L = 9 (q = 0.4) by a
  rigorous interval enclosure; my numerics show it at every L >= 9 tested. With the external field h^2 = kappa_L'(0)
  (post hoc), concave on [qhat, 1) (h^2 + xi'(qhat) = qhat xi''(qhat)) for L <= 8, not for L >= 10. HEURISTIC reading
  only: the spherical spin-glass theorems (Subag; Huang-Sellke, recalled, unverified) would place an overlap gap for
  maximising the output over input directions at depth >= 10, but (i) the actual final-ReLU field is not Gaussian
  (the GP surrogate is a different field) [A r2], (ii) xi''(1) = infinity violates the theorems' smoothness
  assumptions, (iii) concavity alone is not the theorem. ARC's 1e-12 tails sit far below the maximum (about 7-13 SD
  against sqrt(n) OPT), so even the reading concerns near-maximal outputs, not ARC's tails, and nothing about means.

## 8. Conjectures (merged ledger with astra round 2)
- **C0 / M-whole (THE BOTTLENECK).** Fix a cost model; let B(T) = V_sph / m(T), m(T) the largest radius-exact MC batch
  executable at cost T with the best allowed batching. For fixed alpha, every uniform estimator family of cost T_n in
  a stated polynomial window has R >= c B(T_n) at L = alpha n. Status: CONJECTURE. Evidence: every tested deductive
  family loses or saturates (limited evidence about unrestricted algorithms). Cheapest refutation: a charged estimator
  family (explicit design, analytic control variate, global approximation) whose gain grows with n. The missing
  theorem would connect the arithmetic cost of an arbitrary white-box computation to an unavoidable residual error.
- **C1 (low degree at finite width).** The 2/3 window of 4b persists at width n with L = alpha n. CONJECTURE; PROVEN
  in mean field; MEASURED agreement at n = 32, L <= 32 in the linear regime. Refutation: unbiased two-scale stability
  measurements showing a different window.
- **C3 / M-design.** A fixed high-even-power tail keeps a constant fraction of the angular variance at L = alpha n, so
  positive designs with N <= n^p have bounded gains. CONJECTURE at finite width; PROVEN in mean field (fraction -> 1/2,
  so the gain bound tends to 4) [A r2]. Test: kernel values k(-1), k(0), b_2 across n (scratch/kernel_even.py).
- **C2 (partition methods).** PROVEN for covering a fixed mass with fully gate-resolved cells (4f); CONJECTURE for
  downstream-informed certificates. Refutation: a polynomial cell count with enclosed integration error at fixed
  normalised precision.
- **Region count.** E R <= (C n L)^{n-1} for bias-free nets: an extra hypothesis (sampled distinct patterns are lower
  bounds only).
- **T-relative (tails).** A computable threshold with p in [e^{-bn}, e^{-an}] on which no polynomial-time algorithm
  gets relative error 1/2. CONJECTURE; no distribution-preserving reduction. Refutation: radius integration plus
  importance sampling with polynomial cost and bounded relative variance.
- **Refuted:** my v1 C4 (concavity at every depth) is false from L = 9.

## 9. Errata after GPT-6 review A (12:10Z; review/review_gpt6_A.md), applied in the report
- 1b holds for Gaussian weights (bounded wrapped density); for a general density the risk convergence needs an
  integrability assumption (the review gave a counterexample density).
- 1c: n fair bits are free, but 2n are needed for a pairwise-independent sample of n-bit inputs and a free
  permutation does not give fair bits exactly; a full deterministic match for threshold circuits is not established.
- Consequence 3 is PROVEN UNDER digit-splitting and exact-transform primitives; with finitely many bits per weight an
  exact simulation is impossible (H(W', R) > H(W)).
- 4c: mean-field agreement is within 5% at L = 4 only; at L = 16-32 (normalised by D(1)) within -21% to +29% for
  delta <= 1e-2. Degree proxies are finite differences D(1e-5)/1e-5: 2.2, 7, 17 (27 unnormalised) at L = 4, 16, 32. The
  degree-D numbers are plug-in estimates of the ceiling against plain MC, with input standard errors of 20-60%.
- 6: "no technique proves super-linear lower bounds" is literally false (time hierarchies); the claim is that no
  technique is known for natural problems like this one. Random self-reducibility does not in general need a
  polynomial estimand (discrete log); only the interpolation route does. IKW needs time 2^{E^{o(1)}}.
- 7 (spin glasses): the field-adjusted condition also fails at L = 9 (review's interval computation at q = 0.5 >
  qhat = 0.4926); "concave for L <= 8" means no violation on the tested grids. The statement is about a Gaussian
  surrogate only; Huang-Sellke's hardness theorem is for even mixtures (abstract verified), and this mixture has odd
  terms. Tails "far below" the threshold was unsupported (ALG not evaluated); removed.
- 7 (analogies): verdicts changed from "not plausible" to "no transfer established"; PRG runtimes are upper bounds on
  those algorithms, not lower bounds against sampling; "deterministic volume needs correlation decay" narrowed to "the
  known structured-polytope algorithms use it".
- Provenance: the explicit degree-D lower bound is astra's (I wrote mine after its progress line); the identity and
  the noise-stability idea were in my plan before astra's run.

## 10. Round 3 (astra, 12:32Z): C0 is false
- PROVEN UNDER layerwise batching costs and m(cT) = O(m(T)) [A r3]: the multilevel reference-trajectory estimator
  (control G_r(U) = m_n s_r |h_r(U)| q_L from one reference input's normalised path) has risk O(log^2 L / (L N)) at the
  cost of N passes while V_sph = Theta(1) at L = alpha n, so its gain over radius-exact MC is Omega(n / log^2 n).
  Residual bound E|Y - G_r|^2 <= (4 m_n^2/n) sum_{l>=r} delta_l + 2 m_n^2 delta_L, delta_l = 1 - kappa^{(l)}(0).
- MEASURED post hoc [C] (logs/mlmc_check.log): two-level gains 0.91-1.29 (ratio; below 1 is a gain) at L = n/4,
  0.66-1.04 at L = n, 0.33-0.60 at L = 4n, 0.14 at L = 16n.
- C0 is refuted. Revised bottleneck C0*: no uniform estimator has a polynomially growing gain over the best
  multilevel reference-trajectory sampler at L = alpha n (CONJECTURE, little evidence).
- Partition certificates [A r3]: K >~ (n^3/eps)^{(n-2)/2} PROVEN UNDER nonalignment, isolation and capture hypotheses;
  bounded aspect ratio alone gives nothing; expected regular kink area Theta(n^{3/2} L), mean l2 jump Theta(1).
