# FORMAL.md: unrestricted deterministic estimators of the mean of random critical ReLU nets (v1, 31 Oct 2026, 11:35Z)

Lane: unrestricted estimators (bead pool-uwmcvo). Labels: PROVEN, PROVEN UNDER X, HEURISTIC, CONJECTURE, MEASURED.
This is my own draft, written before reading gpt-6-astra's round-1 answer (coprover/astra_r1.raw, launched 11:01Z,
from a prompt that states the questions but none of the statements below).

## 0. Setting and notation
- Net: W = (W_1..W_L), W_l iid N(0, 2/n) entries, no bias; x ~ N(0, I_n); h_l = relu(W_l h_{l-1}) (l < L, plus noise
  s_l xi_l in the noisy variant), output f_W(x) = a_L = relu(W_L h_{L-1}) in R^n. M = n^2 L weights.
- mu(W) = E_x f_W(x); V(W) = tr Var_x f_W(x). Average-case error of an estimator A: Err(A) = E_W |A(W) - mu(W)|^2.
  Monte Carlo with N samples: Err = E_W V / N. For sigma = 0 the net is positively homogeneous:
  f_W(x) = |x| g_W(x/|x|), so mu = E|x| E_u g_W(u) (u uniform on S^{n-1}); "RB" samplers integrate the radius exactly.
- Cost: arithmetic operations. A forward pass of one input costs about 2 n^2 L with schoolbook matrix-vector
  products. (Fast matrix multiplication changes the per-sample cost of batched sampling; see 3.4.)

## 1. Determinism is free on average, uniformly (T1)

**Proposition 1 (PROVEN; real-RAM with floor, or b-bit inputs as b -> infinity).** There is one explicit
deterministic algorithm D_N (for all n, L, N) such that Err(D_N) -> E_W V / N, the error of N-sample MC, at cost
N forward passes + O(N n).
*Construction.* Pick 2n weight entries w_1..w_{2n} (say the first 2n entries of W_1) and an integer k. Put
U_c = frac(2^k w_c). For coordinate c = 1..n and sample i = 1..N let V_{i,c} = frac(U_c + i U_{n+c}) and
x_{i,c} = Phi^{-1}(V_{i,c}). Output (1/N) sum_i f_W(x_i).
*Proof.* (i) Pairwise independence: for fixed i != j the map (u, v) -> (frac(u + i v), frac(u + j v)) is a surjective
endomorphism of the torus T^2 (integer matrix of determinant j - i != 0), hence preserves Haar measure. So if
(U_c, U_{n+c}) are iid uniform and independent across c, the points x_1..x_N are pairwise independent N(0, I_n),
and the MSE of their average is exactly V(W)/N for every fixed W (pairwise independence suffices for the variance of a
mean). (ii) Asymptotic independence from W: W has a density, so for every integrable phi and every nonzero integer
vector j, E[phi(W) exp(2 pi i j . 2^k w_{1..2n})] -> 0 as k -> infinity (Riemann-Lebesgue). Hence
(W, U) -> (W, U') in distribution with U' uniform on T^{2n} independent of W. (iii) The squared error
e(W, U) = |(1/N) sum_i f_W(x_i(U)) - mu(W)|^2 is continuous in U off a null set and uniformly integrable (Gaussian
tails of Phi^{-1}, moments of W), so E e(W, U) -> E e(W, U') = E_W V(W)/N. QED.
(With b-bit inputs the same holds with explicit rates when the seed bits are the low-order bits of entries whose
perturbation moves f by o(1); I do not need the rates.)

**Consequences (PROVEN as logical relations).**
1. In ARC's random-MLP instance (average over W, no advice) "deterministic" excludes nothing, even uniformly: the
   nonuniform statement of the hardness note (Prop 3.5) becomes explicit. Typically-correct derandomisation
   (Goldreich-Wigderson 2002) uses the same idea (the input as the randomness source).
2. No syntactic restriction short of forbidding high-precision digit operations stops this: Chebyshev iteration
   T_{2m} = 2 T_m^2 - 1 (k squarings) computes cos(2^k arccos w), an equidistributed seed, with plain arithmetic.
   So "mechanistic" must be semantic (a stability / regularity bound on W -> A(W), section 4) or representational
   (a bounded interface: the sibling lane), not "deterministic" or "algebraic".
3. Therefore "no deterministic estimator beats sampling on average" is equivalent (up to o(1)) to "no estimator at
   all beats sampling on average": the question of this lane is whether MC is optimal among all algorithms.

## 2. Fixed point sets: error is a kernel energy (T2)

**Proposition 2 (PROVEN, exact).** Let sigma = 0, U = (u_1..u_N) fixed unit vectors, weights c with sum c_i = 1, and
A_U(W) = E|x| sum_i c_i g_W(u_i) (radius exact). Let kappa(u.v) = E_W [g_W(u) . g_W(v)] (zonal by rotation invariance
of the first layer, which holds jointly for He nets) with Gegenbauer expansion kappa(t) = sum_l a_l P_l(t), a_l >= 0,
P_l(1) = 1. Then
  Err(A_U) = (E|x|)^2 sum_{l >= 1} a_l S_l(U, c),   S_l = sum_{ij} c_i c_j P_l(u_i . u_j) = |sum_i c_i Y_l(u_i)|^2 / N(n, l) >= 0,
and RB-MC (iid uniform u_i, c_i = 1/N) has E S_l = 1/N for every l >= 1.
*Proof.* E_W[g(u_i) . g(u_j)] = kappa(u_i . u_j); E_W[g(u_i) . gbar] = a_0 and E_W |gbar|^2 = a_0 (gbar = E_u g),
by Funk-Hecke; expand and use sum c_i = 1; S_l >= 0 by the addition theorem. QED
**Consequences.**
- A spherical t-design with equal weights has S_l = 0 for 1 <= l <= t, so it removes the degree-1..t part of the
  angular variance exactly; its gain over RB-MC is (sum_{l>=1} a_l) / (N sum_{l>t} a_l S_l). If the design is
  MC-like beyond t (S_l about 1/N), the gain is the inverse of the fraction of angular variance above degree t.
  (HEURISTIC as a ceiling: nothing forces S_l >= 1/N for l > t at finite l; PROVEN in the limit: for distinct points
  P_l(u_i . u_j) -> 0 as l -> infinity, so S_l -> sum c_i^2 >= 1/N.)
- The deterministic rule and its random rotation have the same average error (rotation invariance), so
  deterministic QMC is no better or worse than RQMC on average over nets; any gain is in the harmonic content.
- In the Gaussian-process limit (n fixed, width of hidden layers -> infinity, fixed L) with spectrum
  a_l / N(n, l) ~ l^{-(n+2)} (Bietti-Bach; cut-rank lane), the best N-point rule has error of order
  N^{-1-3/(n-1)} (Sobolev-space QMC-design rates, s = (n+2)/2 on S^{n-1}): a gain of order N^{3/(n-1)} over MC
  (N^0.43 at n = 8, N^0.012 at n = 256). For Gaussian measures adaptive or nonlinear use of function values does not
  improve on the best linear rule (average-case IBC; Ritter 2000, ch. "Nonlinear methods for linear problems").
  PROVEN UNDER the GP limit and the cited spectrum and IBC results [unverified constants].

## 3. Barriers (T3)

**3.1 (PROVEN, logical).** By Proposition 1, any statement "deterministic algorithms cannot beat sampling by more
than C on average" is a statement about all algorithms. Its unconditional proof at budgets T = N T_fwd with
N >= M^delta would be a time lower bound of order M^{1+delta} for an explicit average-case problem in the general
(RAM or circuit) model. No technique gives super-linear lower bounds for explicit problems in general models, so
an unrestricted impossibility proof is out of reach; only restricted classes or conditional statements are possible.
**3.2 (PROVEN, logical).** IKW-type barriers (subexponential derandomisation of CAPP implies NEXP not in P/poly)
are worst-case statements. On average over continuous weights derandomisation is free (Prop. 1), so they say
nothing here. Worst-case hardness (the circuit reduction plus #NSETH) does not transfer: random critical nets
embed a given circuit with probability 0, and the standard worst-to-average tool (random self-reducibility of
low-degree polynomials: Lipton; Ball-Rosen-Sabin-Vasudevan; Gamarnik-Kizildag for SK) needs an estimand that is a
low-degree polynomial in the instance; mu(W) is not a polynomial in W (it is positively homogeneous of degree 1 in
each layer, so a polynomial would be multilinear in the layers; already for L = 1, mu = |w|/sqrt(2 pi)).
**3.3 (natural proofs).** Razborov-Rudich concerns circuit lower bounds proven via large constructive properties;
an unrestricted fine-grained lower bound here would be stronger than any known explicit circuit lower bound, so it
is blocked at least by whatever blocks super-linear lower bounds. I claim nothing more specific.
**3.4 (the baseline must be the best sampler).** A batch of N inputs through an n x n layer is a matrix product;
with fast matrix multiplication it costs O(N n^{omega-1}) instead of 2 N n^2. So "beats serial MC by a growing
factor" is achievable trivially by batched MC; statements must be made against the best sampling implementation or
in a cost model that charges matrix-vector products at n^2. (I adopt the latter: the unit is a forward pass of one
input; every arm here uses schoolbook products.)

## 4. Restricted classes (T4)

**4.1 Low-degree polynomials in W (PROVEN identity; MEASURED and HEURISTIC rate).** Let mu = sum_k mu^{=k} be the
Hermite expansion in the M Gaussian weights (normalised), w_k = ||mu^{=k}||^2. A degree-D estimator with
instance-independent coefficients has Err >= sum_{k > D} w_k (orthogonality), with equality for the projection.
Noise stability S(rho) = Cov(mu(W), mu(W_rho)), W_rho = rho W + sqrt(1 - rho^2) W', equals sum_{k>=1} rho^k w_k, so
for every rho in (0, 1): sum_{1 <= k <= D} w_k <= S(rho) / rho^D, i.e.
  Err(any degree-D estimator) >= Var mu - min_rho S(rho) rho^{-D}   (+ the k = 0 part is free).
This turns a measurement of S(rho) into a lower bound for the whole class.
*Heuristic rate.* At criticality the cosine of two trajectories through weights correlated by rho obeys
c' = kappa(rho c) (kappa the arc-cosine map). Its fixed point near 1 satisfies 1 - c* ~ (3 pi/(2 sqrt 2))^{2/3}
delta^{2/3} (delta = 1 - rho) and is reached after about delta^{-1/3} layers (PROVEN for the mean-field map by the
expansion 1 - kappa(cos t) = t^2/2 - t^3/(3 pi) + O(t^4)). If the per-net mean follows the collapsed state's
direction, 1 - S(1 - delta)/S(1) ~ C delta^{2/3} for L^{-3} << delta << 1, which corresponds to w_k ~ k^{-5/3} up to
k ~ L^3: degree-D estimators are worth at most about D^{2/3} samples. HEURISTIC; the measurement in section 6 tests it.

**4.2 Stability (PROVEN, trivial; content is MEASURED).** For any estimator A and any rho,
  2 ||A - mu||_2 >= ||mu - mu_rho||_2 - ||A - A_rho||_2
(triangle inequality; W_rho has the law of W). So an accurate estimator must be at least as rough in W as mu at
every scale delta where ||mu - mu_rho|| exceeds twice the target error. Overlap-gap arguments do not apply as such:
they exploit an unstable solution geometry, while mu is a single real-valued, continuous function of W; the class of
stable algorithms contains mu itself whenever mu's own modulus is admitted.

**4.3 Statistical queries (PROVEN, trivial).** The estimand is itself one statistical query (E_x f_W(x)) to
tolerance eps; SQ lower bounds count queries, not the cost of answering one, so they are vacuous here.

## 5. Where known average-case hardness lives (T5)
With Z(beta) = E_x exp(beta f), mu = (log Z)'(0): the mean is the infinite-temperature end. The known average-case
hard problems (exact SK partition functions, Gamarnik-Kizildag; random-circuit output probabilities and Gaussian
permanents, Bouland-Fefferman-Nirkhe-Vazirani, Aaronson-Arkhipov; deterministic volume in the oracle model,
Elekes / Barany-Furedi) are exact or relative-precision problems on exponentially small quantities, where sampling
fails. The mean with SD-normalised error is additive precision at the scale MC achieves in polynomial time (hardness
note Prop 3.1), so all of them are silent (PROVEN as a logical relation). The tail problem P(a_{L,j} > t) at tiny p is
a relative-precision, low-temperature problem: there they could bite (CONJECTURE). In the infinite-width GP limit the
output field on S^{n-1} is a mixed spherical spin glass with mixture xi(q) = kappa_L(q). For one layer
kappa'' = 1/(pi sqrt(1 - q^2)), so xi''^{-1/2} = sqrt(pi) (1 - q^2)^{1/4} is concave on (0, 1): the full-RSB
condition under which Lipschitz algorithms reach the maximum (Subag; Huang-Sellke) holds, so there is no overlap-gap
obstruction to the extreme tail of a one-layer infinite-width ReLU field (PROVEN for the concavity; the transfer of
the spin-glass theorem, with the external-field term, is HEURISTIC). Depth: to compute (section 6).

## 6. Exact and partition methods (T6)
**6.1 (PROVEN UNDER a region-count bound).** For sigma = 0 the net is linear on each of finitely many polyhedral
cones. Enumerating them by adjacency and integrating each exactly (fixed dimension: polylog(1/eps) per cone) costs
R(W) poly(n L) polylog(1/eps), R the number of cones of positive measure. If E R <= (C n L)^{n-1} (the Hanin-Rolnick
order, proven for nets with random biases; not for bias-free nets), the cost is polynomial in L at fixed width, so at
fixed width exact deterministic integration beats MC for every eps below about (C n L)^{-(n-1)/2}: ARC's literal
reading (fixed n, eps -> 0) is easy at every depth, conditionally. At L = alpha n the count is exp(Theta(n log n)).
**6.2 (HEURISTIC).** A partition method that is exact on cells inside one region and first-order elsewhere, with K
cells of diameter r ~ K^{-1/(n-1)} on S^{n-1}, makes errors of order r^2 on the fraction ~ r (kink area) of cells that
cross kinks; with signs that do not cancel coherently, its squared error is ~ K^{-(n+4)/(n-1)} (times the kink
density, which grows with L), against MC's 1/N with N ~ K / c_cell. It beats MC asymptotically at every fixed n,
with an exponent advantage 5/(n-1) that vanishes with width, and constants that grow with depth.

## 7. Conjectures (T7); the bottleneck is C0
- **C0 (master).** For each alpha > 0 and every algorithm (white-box, any randomness), the advantage over
  RB-MC at equal cost (unit: one-input forward pass) is bounded by C(alpha) at L = alpha n, uniformly over budgets
  in [n^a, n^b] passes. Status: open; not provable unconditionally (3.1); consistent with all measurements so far.
  Cheapest refutation: any family with advantage growing in n at fixed alpha.
- **C1 (low degree).** w_k ~ k^{-5/3} up to k ~ L^3 (4.1). Refuted if S(1 - delta) is linear in delta for
  delta >> L^{-3}.
- **C2 (partition methods).** Certified-cell methods need exp(c n) cells at fixed precision at L = alpha n.
  Refuted by a polynomial-in-n cell count at fixed precision.
- **C3 (designs).** At L = alpha n the fraction of angular variance in degrees <= t stays bounded away from 1 as
  n grows (designs give a bounded gain). Refuted by that fraction -> 1.
- **C4 (tails).** Stable algorithms fail for tails at large thresholds only through relative precision, not through
  an overlap gap (the GP-limit mixture satisfies the full-RSB condition at every depth). Refuted by a depth at which
  xi''^{-1/2} loses concavity.
