# Stationary regime of noisy critical He-ReLU stacks: theory v1 (Claude, 13:15Z 31 Oct, frozen before astra's answer)

Labels: PROVEN, PROVEN UNDER <hypothesis>, HEURISTIC, CONJECTURE. Dev data used so far (dev seeds 1-5 only):
`scratch/dev_probe.py`, `scratch/dev_chain.py` (stdout in the session; rerun to reproduce). Disclosure: at about
13:13Z I saw the last lines of astra's round-1 log (a table "tail_C/n ~ 0.40" for n = 8..256); item 3 below was
derived and checked on dev data before that.

## 0. Setting
W_l iid N(0, 2/n) entries (n x n), xi_l iid N(0, I_n), all independent, l in Z.
F_l(h) = relu(W_l h) + s_l xi_l. Model S: s_l = sigma (absolute). Model R: s_l = sigma r_l (relative to the pilot RMS
of the clean activation). Phi_{m -> l} = F_l o ... o F_{m+1}. Output a = relu(W_1 h_0).
Write X_l = |h_l|^2 / (n sigma^2) (Model S), u_l = h_l / |h_l|, g_l = |relu(W_l u_{l-1})|^2 (one-layer gain of the
state's own direction), e_l = relu(W_l u_{l-1}) / |relu(W_l u_{l-1})|.

## 1. Existence, uniqueness, synchronisation (Model S)

**Lemma 1 (exact one-step pair contraction). PROVEN.** For fixed h, h' with |h| = a, |h'| = b, angle theta:
E_W |relu(W h) - relu(W h')|^2 = |h - h'|^2 - (2 a b / pi)(sin theta - theta cos theta) <= |h - h'|^2,
with equality iff theta = 0 or ab = 0.
*Proof.* Per row, (w.h, w.h') is centred Gaussian with covariance (2/n)[[a^2, ab c], [ab c, b^2]], c = cos theta.
E relu(X)^2 = a^2/n, and E relu(X) relu(Y) = (ab/(n pi))(sin theta + (pi - theta) c) (arc-cosine kernel). Sum over
n rows: a^2 + b^2 - (2ab/pi)(sin theta + (pi - theta) c). Subtract from a^2 + b^2 - 2ab c. sin theta - theta cos theta
>= 0 on [0, pi]. QED

**Lemma 2 (fractional-moment contraction). PROVEN (compactness step sketched).** For s in (0, 1],
rho_s := sup_{h != h'} E_W [ (|relu(Wh) - relu(Wh')| / |h - h'|)^s ] < 1.
*Proof sketch.* The law of the ratio R depends on (h, h') only through their configuration modulo rotations and
scaling (W -> W O preserves the law of W; relu is 1-homogeneous): a compact parameter set after adding the limits
h' -> h along a direction v (where R -> |D_h W v| / |v|) and h' -> 0. At every point of the closure E R^2 <= 1 by
Lemma 1 (or by E|D W v|^2 = |v|^2 in the limits) and R is not a.s. constant, so E R^s < (E R^2)^{s/2} <= 1 by strict
Jensen. E R^s is continuous on the closure (dominated by ||W||_op^s, gate boundaries have measure zero), so its sup
is attained and is < 1. QED (the compactification is the step to check).

**Theorem 1 (random point attractor and stationary sample measure). PROVEN, given Lemma 2.**
(i) For all h, h', T: E |Phi_{-T -> 0}(h) - Phi_{-T -> 0}(h')|^s <= rho_s^T |h - h'|^s (the noise cancels in
differences; condition layer by layer).
(ii) For every deterministic start h, Y_T := Phi_{-T -> 0}(h) converges a.s. and in L^s as T -> infinity, to a limit
Y_0 = Y(W, xi) that does not depend on h. *Proof:* E|Y_{T+1} - Y_T|^s <= rho_s^T E|F_{-T}(h) - h|^s = C_h rho_s^T,
summable; with s <= 1, sum |Y_{T+1} - Y_T| < infinity a.s.
(iii) Y_l := Y(shifted (W, xi)) satisfies Y_l = F_l(Y_{l-1}): with the weights AND the noise fixed, all trajectories
synchronise onto one random trajectory (synchronisation by noise; the noiseless system has the trivial attractor 0).
(iv) The quenched sample measure mu_l^W := Law(Y_l | W) (noise integrated) satisfies mu_l^W = mu_{l-1}^W P_{W_l}, where
P_W(h, .) = Law(relu(W h) + sigma xi). It is the unique such family among tempered families (sup_l E int |h|^s dnu_l^W
< infinity): for such nu, the coupling in (i) gives W_s(nu_0, mu_0) -> 0 in expectation over W.
(v) Rate. Nearby pairs separate at the top exponent lambda_1 of the tangent cocycle D_l W_l, so the pullback error
decays like exp(lambda_1 T) with lambda_1 = -5/(4n) + O(n^-2) (item 4); far pairs contract at least as fast in L^s.
Memory time of the coupling: 4n/5 layers per e-fold of distance (2n/5 for squared distances).

**Model R.** The same coupling argument works for any deterministic scale sequence s_l, so the pullback limit exists.
But Model R's scales are tied to the input layer (r_l tracks the typical size of the input-driven state), so there is
no shift-invariant stationary regime in absolute units; in units of r_l the radius ratio of two inputs is not
contracted (both decay at lambda_1). Model R has a stationary law for the direction only (HEURISTIC; section 6).

## 2. Scale invariance (Model S). PROVEN.
Phi^sigma_{m -> l}(sigma h) = sigma Phi^1_{m -> l}(h) pathwise (same W, xi), because relu is positively 1-homogeneous
and there are no biases. Hence Y^sigma = sigma Y^1 and mu^{W, sigma} = (h -> sigma h)_# mu^{W, 1}.
Consequences: in the stationary regime sigma is not a parameter. E[a] scales as sigma, tail events P(|a| > t) depend
on t/sigma only, and nMSE-type ratios, gate statistics, bit counts and every estimator's relative efficiency are
independent of sigma. A "small-noise" expansion in sigma has nothing to expand in. sigma only matters for the transient
from an input of fixed size (section 7).

## 3. The radius is an exact one-dimensional Markov chain (annealed). PROVEN.
X_l = |h_l|^2 / (n sigma^2) satisfies
  n X_l = ( sqrt(g_l n X_{l-1}) + eta_l )^2 + C_l,
with (g_l, eta_l, C_l) iid over l and independent of the past: g ~ (2/n) chi^2_N with N ~ Bin(n, 1/2) (equivalently
(2/n) sum_i relu(gamma_i)^2), eta ~ N(0, 1), C ~ chi^2_{n-1}, all independent.
*Proof.* Given the past F_{l-1}, W_l is independent of u_{l-1} and rotation invariant, so W_l u_{l-1} ~ N(0, (2/n) I)
and g_l has the stated law, independent of F_{l-1}. Decompose xi_l = eta_l e_l + xi_perp (e_l is F_{l-1} v W_l
measurable, xi_l independent of both): eta_l ~ N(0,1), |xi_perp|^2 ~ chi^2_{n-1}, independent. Expand
|relu(W_l h_{l-1}) + sigma xi_l|^2. QED
So the radius forgets everything except itself: its law (over weights and noise) is that of a scalar chain, which
can be computed to any precision, and by ergodicity of the iid environment it is also the distribution over layers
of one net.

**Corollaries.**
(a) E[n X_l | X_{l-1}] = n X_{l-1} + n: the annealed second moment grows linearly forever (E g = 1 is He
criticality); the stationary law has E X = infinity. PROVEN.
(b) Tail: writing R_l = sqrt(n X_l), R_l = sqrt(g_l) R_{l-1} + D_l with |D_l| <= |eta_l| + sqrt(C_l), an integrable
perturbation independent of R_{l-1}. With E (sqrt g)^kappa = 1 at kappa = 2 (E g = 1), Goldie's implicit renewal
theorem gives P(R > t) ~ c t^{-2}, i.e. P(X > x) ~ c'/x: tail index 1 for X. PROVEN UNDER the hypotheses of Goldie
(1991, Thm 2.3; non-lattice log g, E g log+ g < infinity; c' > 0 needs a separate argument; citation from memory).
(c) Bulk (diffusion limit, n large). log g has mean -5/(2n) + O(n^-2) and variance 5/n + O(n^-2) (Var g = 5/n exactly).
Dufresne's identity for the exponential functional of Brownian motion with drift gives
X ~ 2n/(5 Z), Z ~ Exp(1): median 0.577 n, quantile q at 0.4 n / ln(1/q). HEURISTIC (diffusion approximation; the
scalar chain gives the exact finite-n law). Dev check (seed 5, 40,000 layers): medians 4.65 / 18.2 / 37.7 against
4.62 / 18.5 / 36.9 at n = 8 / 32 / 64; Hill index of X near 1.
(d) The stationary noise-to-signal ratio is set by n, not sigma: per coordinate, one layer's noise is
1/sqrt(X) ~ sqrt(Z / (0.4 n)) of the state, i.e. O(n^{-1/2}) with an Exp(1)-distributed Z. Layers where the radius
sits near the noise floor (X = O(1)) have probability about exp(-0.4 n): common at n = 8, negligible at n >= 64.

## 4. Lyapunov spectrum of the tangent cocycle (both models)
For any tangent vector v independent of W_l, |D_l W_l v|^2 / |v|^2 has mean 1 and variance 5/n whatever the angle
between v and the state (E[gamma'^2 1{gamma > 0}] = 1/2 and E[gamma'^4 1{gamma > 0}] = 3/2 for any correlation). PROVEN.
Hence lambda_1 = -5/(4n) + O(n^-2) along noisy and noiseless trajectories alike (HEURISTIC for the identification of
lambda_1 with this one-vector rate; it is exact for rotation-invariant iid products, and here the gates are
correlated with the state). Lower exponents: half-width Ginibre heuristic lambda_k - lambda_1 ~ -(k-1)/n
(narrow-deep lane, measured at n = 16 for k = 2). HEURISTIC.

## 5. The attractor and the cloud (Model S)
**Picture (HEURISTIC, to be tested).** The noiseless directional dynamics u -> e(u) synchronises (narrow-deep lane):
there is a random attractor direction u*_l. In Model S the state is h_l ~ R_l u*_l + delta_l with R_l the radius (the
chain of item 3, heavy-tailed over layers) and delta_l a cloud of angular deviations.
- Radial part. NOT sigma^2 / (2 |lambda_1|) = 0.4 n sigma^2 (the coordinator's formula), but about 0.4 n^2 sigma^2 / Z:
  n times larger. Reason: the top singular value of the Jacobian product J_j collects the Frobenius mass of all n input
  directions (||J_j||_F^2 has mean n at every j, and for j >> n it sits in s_1), so s_1(J_j)^2 ~ n exp(2 S_j), not
  exp(2 S_j). Equivalently, item 3: every layer adds n sigma^2 to |h|^2 and the total decays at 5/(2n).
- Angular part. Linear response along the trajectory's own gates: Cov(delta) ~ sigma^2 sum_j P_perp J_j J_j^T P_perp.
  Its size relative to R^2 is O(Z ln n / n) if the angular singular values of J_j decay like exp(lambda_k j) after a
  transient, so the cloud is thin at large n (angular RMS ~ sqrt(Z ln n / n)) and fat at n = 8.
- Where the near-Gaussian picture must fail: (1) the radius (Pareto tail over layers/weights; over the noise at fixed
  weights its law is whatever the last ~n layers make it); (2) near-floor layers (X = O(1)), where the noise
  re-randomises the direction; (3) small n, where one layer's noise is ~1/sqrt(X) ~ 0.3-0.5 of the state.

## 6. Bits (both models)
Pre-activation of unit i at layer l+1: z = w_i.a_l + sigma w_i.xi_l, a deterministic part with spread
tau_det^2 = (2/n)|a_l|^2 across units and a noise part with variance s_z^2 = sigma^2 |w_i|^2 ~ 2 sigma^2.
- Flip probability when only the last layer's noise is resampled (Gaussian-unit model, PROVEN for that model):
  p_flip = arccos(r)/pi, r = tau_det^2 / (tau_det^2 + s_z^2); for s_z << tau_det, p_flip ~ (sqrt 2 / pi) s_z / tau_det.
- The coordinator's count "within sigma of zero" is P(|z| < s_z) ~ sqrt(2/pi) s_z / tau_det; per layer
  n sqrt(2/pi) s_z / tau_det. The two differ by the constant 0.45 against 0.80.
- Model S: s_z / tau_det = 1 / sqrt(X_pre), X_pre = |a_l|^2 / (n sigma^2). So births per layer are about
  0.45 n / sqrt(X) ~ 0.71 sqrt(n Z): of order sqrt(n), fluctuating with the radius. Model R: about 0.45 n sigma.
- Deaths: a flipped gate perturbs the state by O(s_z) in one coordinate; its radial part decays at |lambda_1| (memory
  4n/5 in amplitude) and its angular parts faster (HEURISTIC).
- Live-bit budget (HEURISTIC): births x radial lifetime ~ 0.7 sqrt(nZ) x 0.4 n ~ 0.3 n^{3/2}, each carrying a relative
  radial effect ~ 1/(n sqrt(X)): many small effects at large n (CLT end), few large ones at n = 8.

## 7. Input bits (Model S with an input)
From an input of radius about sqrt(n) at layer 0, log R drifts down at 5/(4n) per layer with diffusion 5/(4n), so the
relative noise grows like sigma exp(5L/(4n) +- sqrt(5L/(4n))), and the stationary scale (R^2 ~ 0.4 n^2 sigma^2) is
reached after T_mem ~ (4n/5) ln(1/(sigma sqrt(0.4 n)))... layers (HEURISTIC). Below T_mem the state is the
input-driven trajectory with a small noise cloud: the coordinator's small-sigma picture applies there, with effective
relative noise sigma_eff(L) = sigma sqrt(n) / R_L growing with depth. At ARC's 256 x 32 with sigma = 1e-3,
T_mem ~ 200 x 5 = 1,000 layers, so L = 32 is deep in the input regime.

## 8. Open
- Uniqueness for non-tempered families; the compactification in Lemma 2.
- The angular spectrum of the cloud and its prefactors (measured in part A).
- Whether the quenched (fixed-weight, fixed-layer) law of R over the noise is near-Gaussian.
