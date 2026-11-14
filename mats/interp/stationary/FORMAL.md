# Stationary regime of noisy critical He-ReLU stacks: theory v2.2 (31 Oct 2026)

v1 (Claude, frozen 13:15Z, before astra's round-1 answer) is kept in `coprover/FORMAL_v1_as_written.md`. v2 merges
astra's round 1 (`coprover/astra_r1_answer.md`) and the Part A evaluation (`results/partA_tables.md`); v2.1 applies
astra's round-2 audit (`coprover/astra_r2_answer.md`, 13:55Z) and its finite-angle closure (section 5a); v2.2 applies
the GPT-6 review (`review/review_gpt6.md`, 14:25Z). Provenance tags:
[C] Claude, [A] gpt-6-astra, [C+A] both independently. Labels: PROVEN, PROVEN UNDER <hypothesis>, MEASURED,
HEURISTIC, CONJECTURE. Corrections of v1 are marked "(v1 corrected)".

## 0. Setting
W_l iid N(0, 2/n) (n x n), xi_l iid N(0, I_n), l in Z. F_l(h) = relu(W_l h) + s_l xi_l. Model S: s_l = sigma
(absolute). Model R: s_l = sigma r_l (pilot RMS of the clean activation; sibling-lane convention). Output
a = relu(W_1 h_0). Q_l = |h_l|^2 / (n sigma^2) (Model S). "Fixed weights" = conditioning on the weight sequence.

## 1. Existence, uniqueness and synchronisation (Model S)
1.1 PROVEN [A]. Exact global reset: if every entry of W_{2j} is negative and xi_{2j-1} is coordinatewise nonnegative,
the two-layer block maps all of R^n to sigma xi_{2j}. Such blocks occur a.s. (probability 2^(-n^2-n) each), so all
long pullbacks from all initial states coincide exactly: a random point attractor H_l = F_l(H_{l-1}) with the noise
fixed, and a unique quenched sample measure mu_l^W = Law_xi(H_l | W) among ALL entrance laws (no moment condition).
The reset is a correctness device (mean waiting time ~ 2^(n^2+n)), not the practical memory. (v1 corrected: v1 proved
uniqueness only among tempered families.)
1.2 PROVEN [C identity, A uniform bound]. One-step pair identity (v1 Lemma 1):
E_W |relu(Wh) - relu(Wh')|^2 = |h - h'|^2 - (2|h||h'|/pi)(sin t - t cos t). Uniform fractional contraction (astra's
sign-symmetrisation proof replaces v1's compactness sketch): E_W |relu(Wx) - relu(Wy)|^p <= rho_{n,p} |x - y|^p with
rho_{n,p} = E[A^{p/2}] < 1, A = (2/n) chi^2_M, M ~ Bin(n, 1/2); -log rho_{n,1} = 5/(8n) + 3/(8n^2) + O(n^-3).
Synchronous coupling (noise cancels in differences) gives E|h_T^x - h_T^y|^p <= rho_{n,p}^T E|x - y|^p. The displayed
constant is for 1 <= p < 2 (it fails in general for p < 1) [GPT-6 review]. Statements about fixed weights hold for
almost every iid weight sequence, not for every fixed sequence.
1.3 PROVEN [A]. Every quenched sample measure has all moments and exponentially decaying squared-radius tails
(E_xi exp(c_W |H_0|^2/sigma^2) < infinity), while the annealed law has a power tail (item 3). Zero-start pullbacks
converge in quenched mean square at any rate beta < 5/(8n), with weight-dependent constants.
1.4 MEASURED (PA4). The practical coupling-from-the-past rate is the top typical exponent: n d log(dist)/dT in
[-1.54, -1.15] at n = 16..256 (prediction -5/4; the fit uses the relative distance |hA - hB|/|hA| [A r2]).
(v1 corrected [A r2]: v1's Theorem 1(v) conflated this typical, survival-conditioned rate with moment contraction; the
proven first-moment contraction rate is about 5/(8n) per layer.)

## 2. Scale invariance (Model S). PROVEN [C+A].
H^sigma = sigma H^1 pathwise (relu positively homogeneous, no biases). sigma is not a shape parameter of the
stationary regime; the noise-to-signal ratio sigma/tau = 1/sqrt(2 Q) is set by width. MEASURED (PA12): identical
laws to 5e-15 at sigma = 1e-3, 1e-2, 1e-1. For a tail probability normalised by tr Var(a), the score is not invariant
under output units; rescale t with sigma [A].

## 3. The radius (Model S)
3.1 PROVEN [C+A]. Annealed exact scalar chain. Q_l = A_l Q_{l-1} + B_l with (A_l, B_l) iid:
M ~ Bin(n, 1/2), U ~ chi^2_M, V ~ chi^2_{n-M}, A = 2U/n, B = (U + V)/n [A's form; v1's form
n Q' = (sqrt(g n Q) + eta)^2 + C with g = (2/n) chi^2_M is the same law]. E A = E B = 1, E[Q_l | Q_{l-1} = q] = q + 1,
Var = (5q^2 + 4q + 2)/n [A]. By ergodicity of the iid environment this is also the distribution over layers of one net.
3.2 PROVEN [C+A] (A gave the constant). Tail: P(Q > x) ~ C_n / x, C_n = 1/E[A log A] = 2n/5 + 13/75 + O(1/n)
(Goldie's implicit renewal theorem; citation details unverified). E Q = infinity; from a zero start E|H^(T)|^2 =
n sigma^2 T.
3.3 PROVEN [A] (v1: HEURISTIC via Dufresne). Q/n => X with P(X <= x) = exp(-2/(5x)) (inverse gamma; X = 2/(5Z),
Z ~ Exp(1)); median 0.5771 n; E|H| ~ sqrt(2 pi/5) sigma n. This is a large-width limit, not the exact finite-n law
[A r2]. (v1 corrected [A r2]: unconditionally E log g = -infinity since P(g = 0) = 2^-n; the log-gain expansions
-5/(2n), 5/n are conditional on g > 0. v1's "near-floor layers have probability ~ exp(-0.4 n)" is a CONJECTURE needing
a large-deviation argument. v1's bounded-perturbation form of the recursion depends on R; use the affine form 3.1.)
3.4 MEASURED (PA1, PA2). Net chains over layers match the scalar chain: 10/50/90% quantile ratios 0.94-1.05 at
n <= 128 (0.98-1.19 at n = 256, 2 nets); the chain's median/n is 0.575-0.579 at every n from 8 to 256 (the limit is
reached already at n = 8 for the median); Hill index 0.8-1.2 (tail index 1).

## 4. Tangent exponents
4.1 PROVEN [A] (v1 corrected). At fixed finite n, every infinite-time tangent exponent is -infinity: an all-off layer
(probability 2^-n per layer) makes J = 0. Finite exponents describe a survival-conditioned regime.
4.2 PROVEN UNDER rank conditioning [A]. Conditioning each layer on >= r active gates, for k <= r:
gamma_k = (1/2)[log(4/n) + E psi((M-k+1)/2 | M >= r)] = -(k + 1/4)/n - (k^2 + k + 1/24)/n^2 + O(n^-3);
gamma_1 - gamma_k = (k-1)/n + (k-1)(k+2)/n^2 + .... The proof needs no noiseless trajectory (gate signs are fair coins
independent of the row outer products). v1's one-vector argument (mean 1, variance 5/n for every tangent direction)
is the k = 1 case.
4.3 MEASURED (PA3; hit under the registered median-over-nets rule, as astra's audit pointed out). Single-trajectory QR
over 28n layers: medians of n lambda_1 are -1.42, -1.16, -1.06, -1.25, -1.28 at n = 16..256 (individual nets -0.8 to
-1.7, SE about 0.2 per trajectory); median gaps over k <= 8 are 1.93, 1.23, 1.06, 1.06 at n = 32..256. At n = 16
rank collapse (layers with <= 3 active units) makes the exponents beyond k = 3 diverge, as 4.1 predicts. Caveats
[A r2]: the QR warm-up started from h = 0 (only the first 2n layers discarded), and clipping after rank loss reports
finite numbers; at n = 32 about 3 layers with fewer than 9 active units are expected in 28n layers, so the 8-gap
statistic there does not describe 9 surviving directions.

## 5. The sample measure: attractor and cloud (MEASURED, PA5-PA7, PA11; post hoc `runs/cloudhist_*`)
- The "attractor" is a point, not the noiseless direction: for fixed weights the stationary law at layer l is
  concentrated around its mean m_l (|m|^2 / E|h|^2 = 0.4-0.9, larger at larger n), and the mean direction is close to
  the noiseless directional attractor u* (radial share E(h.u*)^2/E|h|^2 = 0.67, 0.75, 0.77, 0.88, 0.88, 0.95 at
  n = 8..256). |m_l| moves over layers with the inverse-gamma radius law of item 3.
- Within a net the radius is light-tailed (1.3) with coefficient of variation 0.36-0.66 (n = 8), 0.32-0.34 (64),
  0.23-0.28 (128), 0.18-0.23 (256).
- Near-Gaussian at large n: |skew| of the radial projection 1.1-1.3 (n = 16), 0.56-0.66 (64), 0.45-0.55 (128),
  0.29-0.42 (256); angular projections |skew| <= 0.25 at n >= 32 (median). PA11's "radial stays non-Gaussian at
  n = 256" is refuted: the radial skew falls roughly like n^-0.4.
- Covariance spectrum. Log-log slope over k = 2..n/8 scatters around -1 (-0.5 to -1.5; PA5's band missed), i.e. the
  coordinator's harmonic shape on average, but the size is NOT sigma^2/(2|lambda_k|): measured/coordinator (median over
  k = 2..8) is 1.2 (n = 8), 2.1 (16), 2.7 (32), 6.7 (64), 5.7 (128), 7.7 (256) (PA6 hit), larger on layers with larger
  radius; the radial eigenvalue is 14-160x. The frozen-gate Lyapunov recursions get the shape wrong (slopes -1.1 to
  -3.0) and overstate the radial term. PROVEN [A] that a Lyapunov covariance formula requires coefficients fixed
  independently of the noise; here gates depend on it, and E[J^T J | past] = I so mean-square influence never decays.
- Post hoc (HEURISTIC model, tested): relative angular deviations injected at relative size sigma/R_{-j} and decaying at
  the gaps; weighting by the radius history (R_0/R_{-j})^2 explains part of the excess (median measured/model
  1.9-2.5 at n = 32, 2.4-11.6 at n = 256) but not all; on the n = 256 large-radius net it does worse than its
  constant-radius version [A r2]. Rejected as the explanation.
- (v1 corrected [A r2]) v1's "radial covariance ~ 0.4 n^2 sigma^2 / Z" is a squared-radius scale, not a centred
  variance; and E||J_j||_F^2 = n does not establish that s_1(J_j)^2 ~ n exp(2 S_j) typically.
- The "attractor direction" u* is a finite-window, survival-conditioned reference: at fixed n the noiseless state
  eventually dies [A r2]. The radial share is an uncentred second-moment share.

## 5a. The cloud is set by a finite-angle balance, not by the Lyapunov spectrum (HEURISTIC [A r2]; tested)
- Exact facts used [A r2, PROVEN]: the He-ReLU kernel for unit vectors at 1 - delta is
  kappa(1 - delta) = 1 - delta + (2 sqrt 2/(3 pi)) delta^{3/2} + O(delta^{5/2}); for a finite window of noise the
  two-trajectory covariance identity Cov_xi(h_0 | W) = sigma^2 sum_j int_0^1 E[J_j(Z) J_j(Z_t)^T] dt
  (Z_t = tZ + sqrt(1-t^2) Z') holds, with two correlated trajectories and different gates; its stationary
  (infinite-past) version needs a limit interchange that is not justified here; Var_xi(|h_l|^2 | W) = Var(|b|^2) +
  4 sigma^2 E|b|^2 + 2 n sigma^4.
- Closure (HEURISTIC): two noise replicas at the same weights gain angular separation 1/X per layer from the noise and
  lose (2 sqrt 2/(3 pi)) delta^{3/2} to the kernel's finite-angle contraction, so delta = E[1 - u.u'] ~ 2.23 X^{-2/3}
  (X = Q at that layer), with relaxation time O(X^{1/3}) layers, much shorter than the infinitesimal scale n. Gain
  differences between replicas then give a quenched radial CV^2 ~ 2.23 X^{1/3}/n + 1/(2n), i.e. CV ~ 1.36 n^{-1/3} at the
  median radius.
- Test on the Part A nets (post hoc, 14 nets, n = 32-256, X = 12-1182): the angular proxy over 2.23 X^{-2/3} has
  median 1.06 (range 0.70-1.52); the radial CV over its law has median 0.84 (0.65-0.97).
- 5b. Pre-registered fresh test (Amendment A3, seeds 2000-2003, n = 16-256; MEASURED): the direct two-replica
  dispersion delta over 2.23 X^{-2/3} has median 0.87 (range 0.72-1.24) at n >= 32 over X = 8-766 (PA14 hit); the radial
  CV over its law has median 0.80 (0.54-1.16) (PA15 hit). At n = 16 both laws over-predict by 20-40%.
- Reading (HEURISTIC): the cloud's size, its short and nearly k-independent memories (section 7) and its excess over
  sigma^2/(2|lambda_k|) are consistent with the cloud living at finite angles, where the ReLU kernel contracts much
  faster than the infinitesimal Lyapunov gaps. The closure passed its numerical bands (5b); it does not by itself
  derive the memories or the spectrum.

## 6. Bits (both models)
6.1 PROVEN. Flip probabilities of one unit under the most recent noise: deletion (actual gate vs gate without the last
noise) Phi(-|w.a|/(s|w|)) [A]; resampling (two independent draws) arccos(r)/pi in the independent Gaussian-unit model,
r = tau^2/(tau^2 + s_z^2) [C]; the coordinator's sqrt(2/pi) counts a one-noise-SD band, not flips [A]. The
pre-activation noise scale is s|w| ~ sqrt(2) s, not s [A].
6.2 MEASURED (PA8, PA9). Resampling flips per layer match n arccos(Q/(Q+1))/pi with the path's own Q within 2-9% at
n >= 64 (median deviation -6% to 0%; worst bin 18-24% at n <= 32). Flips per layer / sqrt(n) = 0.48-0.64 (median per
width, n = 16..256): births scale as sqrt(n) (astra's deletion constant sqrt(5/(8 pi)) = 0.446 times sqrt 2 = 0.63).
Output-layer gate entropy at fixed weights: 5.5-7 bits (n = 8), 12 (16), 15-20 (32), 13-34 (64), 32-53 (128), 31-73
(256): about 3-4.5 sqrt(n) noise-decided bits per layer.
6.3 Consequence. Model S's stationary regime has many bits at every width (no enumeration regime at n >= 8).

## 7. Age profile and interactions (MEASURED, PA10; `results/age_by_direction.json`)
- Radial information is the oldest: the radial share of the response to noise of age j rises from 1-5% (j = 0) to
  0.5-0.95 at ages n/2..n; median age of the radial information 0.3-1.1 n (prediction 0.4 n ln 2 = 0.28 n).
- The angular eigendirections of the cloud do NOT have memories n/(2(k-1)): their median information ages are nearly
  flat in k (n = 256: 32, 28, 28, 29, 24 layers for k = 2, 3, 5, 9, 16 on one net; 78, 73, 77, 61, 57 on the other,
  against 89, 44, 22, 11, 6). The number of directions excited by noise of age j (participation ratio of
  Cov(Delta_j)) does fall like n/j: 1.3-1.9 x (n/(2j) + 1) at j = 4, 8 (PA10b missed by a little).
- Strong interactions: summed total-effect shares (Sobol total indices by age; 1 for an additive state) are
  1.1-1.4 (n = 8), 1.9-2.2 (16), 1.9-2.4 (32), 2.5-2.6 (64), 3.0-3.1 (128), 3.6-4.3 (256), growing with n (PA10d
  hit); ages >= n carry 4-52% of the sum (PA10e missed).
- Bit birth and death (Amendment A2, MEASURED): resampling one layer's noise flips gates at every later layer;
  the flips per layer fall from the first layer's value to about 1-2 per layer by n/2 layers later and stay there,
  and the cascade over the next 2n layers totals 2.3-2.6 n (n = 64) and 3.8-4.5 n (n = 256) flips (PA13b's upper bound
  3n exceeded at n = 256). The frozen-gate linear response to the same resampling correlates 0.94-0.98 with the true
  change at age 2, 0.73-0.93 at age 8 (n >= 64; 0.44-0.66 at n = 16; PA13c hit), and 0.2-0.5 at age n. First-layer
  flips track the local radius, so they scatter (0.14-1.5 sqrt(n); PA13a missed).

## 7a. The interaction structure (Part I; MEASURED unless labelled)
- Identity [A r2, PROVEN]: sum_j T_j = sum_S |S| V_S / V, the variance-weighted mean interaction order (Owen's mean
  dimension); refreshing each block with probability eps gives E|f - f_eps|^2/(2V) = sum_S (1 - (1-eps)^|S|) V_S/V.
- Mean order over layer blocks within the last 4n layers: h_0 2.1-3.1, a 2.2-3.2, log|h_0| 1.9-3.3, direction 2.4-3.0,
  output gates 5-15 (growing roughly like n^0.35); bulk order (eps = 0.1) 1.7-2.1. Consistent with the Part A per-age
  sums within 30% (PI1).
- Coordinates: the angular noise carries >= 96% of the total effect of h_0 and a; the radial noise components carry
  3-12% (4-27% for log|h_0|). The radius is driven by angular noise through direction-dependent gains (PI3).
- Mechanisms (HEURISTIC reading): the annealed Kesten chain alone has mean order 1.22-1.25 for log Q at n = 16-256
  (flat; a different randomness from the quenched net); the radius-clamped net keeps 0.66-1.13 of the direction's order
  (clamping is itself nonlinear and changes the relative noise); log-radius plus direction coordinates do not make the
  map additive. The reading is that gate switching is the main source and the radius feedback adds.
- The reported mean orders are finite-eps statistics G(0.01)/0.01, lower bounds of sum_S |S| V_S/V; for the output
  gates the finite-eps effect is large (13.2 at eps = 0.01, 7.5 at 0.03). Part I's blocks are causal radial/angular
  coordinates; Part A's are Cartesian; the two ANOVA decompositions differ [GPT-6 review].
- Width: unresolved for the continuous state and output (2.5, 2.1, 3.1 for h_0 at n = 16, 64, 256 in radial/angular
  blocks; Part A's Cartesian per-age sums 2.1 -> 3.9; PI5 missed); neither control grows; the gate pattern's order
  grows.
- Effective dimension: 99% truncation dimension of a is 2n-4n layers (2-4 n^2 coordinates); the deep past older than n
  layers explains <= 5% of Var(a) on 5 of 6 nets and 33% on one large-radius net (long memory of a large radius).

## 8. Model R (relative noise) (PROVEN [A] / MEASURED)
- PROVEN [A]: exact scale/shape decomposition q_l = (1 + sigma^2) g_l^2 q_{l-1} with normalised state Z_l; the global
  input scale persists exactly; there is no restoring term for the absolute log scale. For an individual input with
  the pilot schedule fixed, synchronous coupling still contracts absolute differences (1.2).
- MEASURED (PR1 hit): flips per layer match the Gaussian-unit formula with each path's own ratio (0.9-1.1x at
  sigma = 1e-2, 1e-1). PR2 missed: the angular cloud is not proportional to sigma^2 (ratio 13-133 between 1e-2 and
  1e-3), its spectrum is steep at small sigma (slopes -2 to -5.6) and harmonic only at sigma = 0.1; the spread of
  log|h| over the noise at a fixed input grows with depth even at sigma = 1e-3 (0.01-0.04 at L = 4n, 0.08-0.33 at 16n).

## 9. The coordinator's model, item by item
| claim | verdict | evidence |
|---|---|---|
| pullback / random attractor exists | PROVEN, and with the noise fixed too (exact synchronisation) | 1.1-1.3 |
| state = past noise pushed through Jacobians | exact identity along the path's own gates (piecewise linear), but gates depend on the noise | 5, [A] P9 |
| direction k remembers n/(2(k-1)) layers | number of excited directions falls like n/j (yes); cloud eigendirections all have similar memory (no) | 7 |
| radius remembers longest, ~1/(2|lambda_1|) | yes, and longer (median radial age 0.3-1.1 n) | 7 |
| radius walks down to the noise floor | yes: exact scalar chain, inverse-gamma bulk, Pareto tail index 1 over layers | 3 |
| cloud near-Gaussian | within a net, increasingly so with n (radial skew 0.3-0.4 at n = 256); not over layers | 5 |
| variance along k ~ sigma^2/(2|lambda_k|) | shape ~k^-1 roughly; size 2-8x larger, growing with n and with the radius | 5 |
| births n sqrt(2/pi) sigma/tau | right scaling; exact flips n arccos(Q/(Q+1))/pi; sigma/tau ~ n^-1/2 fixed by width, not by sigma | 6 |
| sigma 1e-3..1e-1 matters | not in Model S (pure scale); it matters only in the transient from an input and in Model R | 2, 8 |
| easy when bits many (CLT) or few | Model S is at the many-bits end at every n; whether it is easy is Part B | 6.3 |

## 10. Estimators (theory, PROVEN UNDER stated hypotheses [A]; Part B measures)
- Fixed-weight MC: nMSE = 1/N; indicator MSE p(1-p)/N [A].
- Burn-in: plain MC needs T ~ log(1/eps)/beta, so nMSE = O(log C / C); burn-in multilevel (shared recent noise)
  removes the log: nMSE = O(1/C) with constant (sum_j sqrt(v_j c_j))^2, and a random-level (Glynn-Rhee type) version
  is unbiased [A]. Exact CFTP via resets has exponent one with an astronomically large constant [A].
- Nested sharing of a deep past: MSE = v_between/M + v_within/(MK); one shared past leaves a floor v_between [A].
- RQMC on a fixed finite noise vector: o(1/C) for square-integrable integrands; no extra power follows without
  dimension/regularity control; increasing stratification at fixed finite depth beats exponent one, but in the
  stationary regime the truncation dimension grows with the accuracy [A].
- Control variates, near-boundary stratification/IS: exponent one, constant = residual variance x cost [A].
- Fixed-order deterministic expansions: error floor; small sigma does not help in Model S (scale invariance) [A].
- Exact final-noise Rao-Blackwellisation: E relu(m + nu G) = nu phi(m/nu) + m Phi(m/nu) [A; also C in partB.py].
- Hinge residual of a linearisation: E R ~ f(0) nu^2/2, E R^2 ~ (2 sqrt 2/(3 sqrt pi)) f(0) nu^3 [A].
- No stationary analogue of the noiseless Omega(n/log^2 n) multilevel gain is established [A, C25].
