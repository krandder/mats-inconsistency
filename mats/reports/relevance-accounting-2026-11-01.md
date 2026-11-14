# What matters at a layer: an accounting of the relevant bits in deep random ReLU nets (1-2 November 2026)

Status: final. Revised after three GPT-6 review rounds (round 1 major revision, 15 findings; round 2 minor revision, 19
items; round 3 minor revision, 13 items; all applied, section 12). All registered cells are complete (stationary
n = 128, 256 finished 01:20Z; final scoring 01:28-01:33Z, 2 November). Added after round 2: Amendment 3A (two more
prospective ARC nets); a per-net gain factor in the mean degree, found from 3A's misses (theory v3.3, mine; v3.4, the
co-prover's independent version); their prospective tests on four fresh nets (Amendments 3B, 3C, scored 02:36-02:38Z).

Lane: relevance accounting (target M1), bead pool-qxszj1.24 under Kelvin's ask pool-qxszj1. Work directory
`work/relevance/`: `DEFINITIONS.md` (v1); `FORMAL.md` (theory v1 frozen 21:17:35Z before the co-prover's answer; v2
appended and frozen 21:48:41Z; v3 errata E1-E20 and additions after the co-prover's audit; v3.3 frozen 01:26:28Z; v3.5 the prospective outcome;
v3.4 [A], the co-prover's round 3, registered 02:03:18Z); `PREREG.md` (Part 1 21:23:36Z, Part 2 21:36:43Z, Part 3
21:59:37Z, Amendment 2A 22:08:11Z, Amendment 3A 00:35:48Z, Amendment 3B 01:26:28Z, Amendment 3C 02:03:18Z; hashes in
`prereg_hashes.txt`, frozen code in `frozen/`; deviations with evidence in
`logs/deviations.txt`); `coprover/` (gpt-6-astra rounds 1-3, `PROVENANCE.md`);
`review/` (GPT-6 review); `lib/`, `tests/test_rel.py` (25 checks pass), `scratch/`, `runs/`, `results/`, `figs/`.
Labels: PROVEN, PROVEN UNDER <hypothesis>, NUMERICAL, MEASURED, HEURISTIC, CONJECTURE. Provenance: [C] Claude,
[A] gpt-6-astra, [C+A] both independently.

## Summary for Kelvin

**Short answer.** A heuristic accounting, built on one theorem about how the future sees the past, predicts from a
layer's own cloud the counts of the tested subspaces (within 2x on 55 and 56 of 56 layer cells of eight ARC 256 x 32
nets run after it was frozen, at eps = 0.1 and 0.01) and, less reliably, the variance budget (45 of 56 within 20%), the
linear share (49 of 56) and the mean Hermite degree (44 of 56). The mean degree is exactly a per-net gain factor R
(the future's gain across the cloud over its gain along it) times the cloud's relative spread over the output's
relative variance (R as scored is a reference-normalised version, section 3a). The annealed theory sets R = 1;
across ten nets R is 0.61-2.4 at l <= 8, and forward-only heuristic estimates of it (gate counts and radial gains along
sampled trajectories) matched it on four fresh nets to a median 3.5% (max 19%; the co-prover's rule, mine 5% and 25%).
The registered mean-degree tests, which paired R with the annealed budget, both missed their joint criteria: the
co-prover's rule met the all-cell part (22 of 28) but not the early-layer part (7 of 12); mine got 21 of 28 and 6 of
12. The budget carries its own per-net error, correlated with R. It also splits the variance into degree
1, an even-degree surrogate and a remainder; that comparison is exploratory. It is not a route to
beating sampling at 256 x 32, and it does not prove that none exists: the tested descriptions need many coordinates at
every layer, better projections or integration methods remain unresolved, and a certified lower bound for these
networks is the missing piece.

1. **The future does not see the past cloud's axes (PROVEN [C+A], for the angular part and the frozen Jacobian at a
   nonzero mean).** Given the past, the next layer acts on the state's angular part through a Gaussian matrix
   independent of everything before it. So every past-measurable angular direction (the cloud's principal axes, the
   past's Lyapunov vectors) gets the same expected backward amplification, and "cloud variance along covariant Lyapunov
   vector k times amplification along k" is the wrong product (covariant vectors k >= 2 are chosen by the future). For
   the linear part the natural object is a free product of the cloud's spectrum with the future's singular spectrum
   (PROVEN for an independently Haar-oriented operator; HEURISTIC for the cloud-averaged Jacobian the accounting uses).
   On the networks its dimension matches the gram-chain lane's annealed formula within 20% on 21 of 24 non-input ARC
   cells (all four nets; median ratio 1.03; NUMERICAL, post hoc).

2. **The variance split at a layer (HEURISTIC model; three prospective tests on eight nets).** Three non-orthogonal channels: the
   radial feature e.X (along the cloud's mean direction: for a thin cloud, the input radius times the gains collected so
   far), the gains still to come, and the output's direction. The model takes the layer's measured cloud and the kernel
   plus a 1/n amplitude term. Frozen before ARC nets 4002-4009 were run (Part 3, Amendments 3A and 3B), it held on them
   for the variance budget on 45 of 56 layer cells (within 20%; 26 of 28 on nets 4002-4005, 19 of 28 on 4006-4009, a
   miss), the linear share on 49 of 56 (within 0.08) and the mean Hermite degree on 44 of 56 (within [0.75, 1.33];
   Amendment 3A missed both on its two nets). Over the first four nets the worst cell errors are 22%, 0.12 and 46%; on
   nets 4004-4005 the mean degree is up to 2.2 times the model's (30 against 16 at the input), and on net 4008 the budget
   is 0.70-0.72 of it at l <= 16.
   - Per-net gain factor (found after 3A's results; derived independently by me and by gpt-6-astra [C+A]). At
     sigma = 0 the mean degree is exactly D = R (tr Sigma/E|X|^2)/(V/E|a|^2), where R is the future's mean squared gain
     on the cloud's covariance directions divided by its gain along the samples themselves (as scored, a
     reference-normalised version; section 3a). The annealed theory has R = 1; over ten nets R is 0.61-2.4 at l <= 8 (at
     the input mean log R = 0.17, sample SD 0.43: a factor of about 1.5 either way). Gate counts and radial gains give
     forward-only heuristic estimates of R (Theorem 1 makes the transverse block independent of the gates); the
     co-prover's trajectory-conditional version also includes radial transfer. Prospectively, on four fresh nets, it
     erred by a median 3.5% (max 19%) and mine by 5% (max 25%). The registered mean-degree tests paired R with v2's
     budget and both missed their joint criteria: the co-prover's rule met the all-cell part (22 of 28 in band) but not
     the early-layer part (7 of 12); mine got 21 of 28 and 6 of 12. The budget has its own per-net error, associated
     with R (correlation 0.67 across 70 dependent layer cells of ten nets, post hoc; my earlier claim that it does not
     respond to R is withdrawn). v2's own mean-degree prediction passed on 25 of 28 fresh cells, including 21 of 21
     cells from the three nets whose early-layer factors were below one, where R and v2's budget error cancel. With the
     measured budget the identity gives D within the error of R (all 28 fresh cells within [0.91, 1.19]; a consequence of
     the R accuracy, not a separate test). On these tested cells, then, the mean degree was predictable per net from
     forward-only estimates and the measured budget, but not from (n, l, L, sigma) alone.
   Medians over the four nets (degree 1 / even-degree surrogate s_2 + s_4/4 + ... / remainder):
   0.22 / 0.21 / 0.57 at the input, 0.61 / 0.13 / 0.26 at l = 8, 0.77 / 0.10 / 0.13 at l = 16, 0.97 / 0.02 / 0.01 at
   l = 31; the model gives 0.29 / 0.19 / 0.52, 0.63 / 0.13 / 0.24, 0.79 / 0.10 / 0.11, 0.99 / 0.01 / 0.00. The late-layer
   surrogates spread by about +-0.1 between nets, and their sampling uncertainty was not quantified, so this comparison is
   descriptive. The radial feature alone explains 0.32 of the variance at l = 8 and 0.45 at l = 16.

3. **How many directions (point estimates for exhibited subspaces).** For ARC's output vector the best tested subspace
   needs N(l, 0.01) = 256, 192, 128, 80, 48, 48, 48 coordinates at l = 0, 4, 8, 16, 24, 28, 31 (median of 4 nets); the
   v2 rule was within a factor 2 on 55 and 56 of the 56 prospective cells (nets 4002-4009) at eps = 0.1 and 0.01. The
   linear part alone needs about 7-30
   (uncertified first-chaos estimates, tolerance 0.01 V). So the count is driven by the nonlinear part early and by the
   output vector's own spread late. A scalar readout needs few only in the last layers (N(l, 0.01) = 36, 18, 2 at
   l = 24, 28, 31).
   - Gates (your question via main): the tested gate orderings need thousands of gates. At l = 8-16 the smallest tested
     prefix of the importance ordering that brings the nonlinear residual within 1% of V is 1,950-4,900 gates (8-19 times
     the width; about the number within two cloud spreads of their boundary), while the nonlinear residual falls below
     10% of V with 24-56 directions. Median per-sample flip fractions (relative to the mean path) range from 5.2% to 8.9%
     across the reported layers.
   - Stationary regime (sigma = 1; n = 32-256, all registered cells): on cells with t >= 2 n^{1/3} ln(1/eps) (the
     gram-chain lane's scale) the top active-subspace direction leaves at most 10% of V on 25 of 29 (13 of 16 seen at
     n <= 64, 12 of 13 prospective at n = 128, 256) and at most 1% on 1 of 15. It is not the radial feature: that alone
     is within 10% on 0 of 16 cells at n <= 64 and 5 of 13 at n >= 128 (per net it leaves 3-85% at t = 16-128). At
     n = 256 the state 128 layers back still holds 36% of the output's variance.
   - Narrow-deep 8 x 256: the radial feature alone leaves at most 2.1e-2 at l = 16 and 9e-6 at l = 32 on all six nets
     (synchronisation). At n = 16 and 32 the residual stays below 1e-3 from l = 2n to 6.25n depending on the net.

4. **Estimation (M1).**
   - Where one direction suffices it pays, by the measured factors at the reported budgets (MEASURED, exploratory, not
     registered): on narrow-deep
     8 x 256 at sigma = 0, the radius at layer 32 times one shared tail pass beats plain MC 9.7x at equal nominal sampling
     cost (median of 6 nets, 4-17x), and 113x (50-313x) with scrambled Sobol on the 8 inputs, which is 15x over RQMC. Setup
     (about 2,300 mv) is excluded; the gains are measured at budgets up to 2^20 mv. This is structured sampling, not a
     sample-free estimator, and it needs L >> n and sigma = 0.
   - At ARC's 256 x 32 the tested subspaces need many coordinates for 99% predictive capture under fitted Gaussian
     clouds, and an exact conditional-mean control on the top 16, 64 or 128 active-subspace features tested at the input
     would gain only 1.33x, 1.80x, 3.11x (this does not bound what optimal features could gain). Whether better future-informed projections, or integration methods that do not resolve the state, beat
     sampling remains unresolved: N bounds predictive dimension, not integration cost (one relu layer under a Gaussian law
     has a closed-form mean in any dimension [A]). The missing result is a certified nonlinear lower bound for these
     networks (v3.2 gives one only under concentration assumptions), followed by a computational-cost argument.

## 1. What is counted (DEFINITIONS.md)

- Model S, W iid N(0, 2/n), no biases; h_k = relu(W_k h_{k-1}) + sigma xi_k; output a = relu(W_L h_{L-1}). Layer l,
  t = L - l layers to go. mu_l = law of h_l (inputs at sigma = 0; past noise in the stationary regime).
- Value function g_l(x) = E[a | h_l = x] (future noise averaged). Relevant variance V_l = tr Var g_l(h_l). At sigma = 0
  the true-law V_l = tr Var(a) at every l; in the stationary regime the memory share M_l = V_l / tr Var(a) < 1.
- A description T explains S(T) = tr Var E[g_l | T] / V_l. N(l, eps) = least dimension of a linear projection with
  S >= 1 - eps, computed with h_l replaced by the Gaussian with its fitted mean and covariance (the complement is then
  explicit and S is measured by pick-freeze). The replacement's fidelity is reported; it fails for narrow-deep nets
  (true law used there). Measured counts are point-estimated residuals on the grid {0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32,
  48, 64, 96, 128, ...} for the best of three exhibited bases (active subspace, degree-1 basis, cloud axes): upper bounds
  on the optimum up to sampling error, not certified.
- Hermite (Wiener chaos) accounting on the layer-l Gaussian cloud: degree 1 has coefficient the cloud-averaged Jacobian
  (Stein), not the Jacobian at the mean. Degree 2 is a sum over gate boundaries of rank-one terms (cloud density at the
  boundary x backward sensitivity x normal outer product), including cross-gate contributions; higher degrees contain
  single-gate tails as well as gate-gate effects, so degrees do not identify "single gates" and "interactions" exactly.
  Measured with Cov(g(X), g(X_rho)) = sum_d rho^d V_d; the shares are normalised by the Gaussianised V_l.

## 2. The future does not see the past cloud's axes

**Theorem 1 (PROVEN [C+A]; FORMAL section 1 and errata E1-E3).** Let e = m/|m| (m != 0; at the exact input layer
m = 0 and the invariance is the full rotation group) and B an orthonormal basis of e^perp, both past-measurable. Write the
first future layer W_{l+1} = g e^T + M B^T. Given the past and given g and everything after layer l + 1, M is an iid
N(0, 2/n) matrix independent of both. Rotating the state about e is the same as replacing M by M O, so the future map's
law is unchanged when the cloud is rotated about its mean direction.
- Let J = J^m_{l->L} be the Jacobian at the mean with its gates frozen and K = J^m_{l+1->L} D_{l+1}(m), so that
  J B = K M. For every past-measurable unit u in e^perp, E[|J u|^2 | past, g, later layers] = (2/n) |K|_F^2, the same
  for every u (corrected statement E1). Covariant Lyapunov vectors k >= 2 are future-measurable, so their decay rates
  exp(-2c(k + 1/4)t/n) do not price the cloud's axes.
- For an independently Haar-oriented backward operator the linear relevance spectrum is that of Lam^{1/2} O^T S O
  Lam^{1/2}, whose bulk law is a free multiplicative convolution (PROVEN UNDER standard asymptotic-freeness hypotheses).
  Top-eigenvalue statements and the extension to the cloud-averaged Jacobian are HEURISTIC (E3).
- NUMERICAL (post hoc): with the rank-one radial/amplitude eigenvalue removed, the angular degree-1 participation ratio
  agrees with the gram-chain lane's annealed formula 1/PR = 1/PR_C + 1/PR_M - 1/n, PR_M = n/(2t + 1), within 20% on 21 of
  24 non-input ARC cells (all four nets; median ratio 1.03, range 0.86-1.31), and within 1.5x on 37 of 53 cells including
  the stationary ones. A forward-only proxy, the number of cloud axes holding 99% of the cloud's covariance, is 163-253 at
  l <= 8, against uncertified first-chaos estimates of 6-16 directions for the linear part.
- The Jacobian at the mean is the wrong linear part once the cloud is wider than the future's coherence radius (about
  3 pi/t): the frozen linear map's variance is 11.9 V at ARC's input (computed at the empirical mean) and 3.4 V at l = 8,
  close to the mean degree there (16.1, 3.3), with a large negative cross term against the gate flips (P5: within
  [D/2, 2D] on 12 of 13 cells, >= 1.5 V on 13 of 13).

## 3. The variance split at a layer (input regime)

Model (FORMAL v2.2; HEURISTIC, E13): in units of E|a|^2,
  V/E|a|^2 ~ [1 - kappa^(L)(0)] (output direction) + (C_L - C_l)/n (gains still to come) + (1/2 + C_l)/n (radial),
  C_l = (1/4) sum_{j<l} [5 - B(c_j)], c_j = kappa^(j)(0), B(c) = (2/pi) J_2(arccos c) - 1 (the input-bits lane's per-layer
  gain covariance; the sum runs over j = 0..t-1 from the layer's cloud in the evaluated form). The channels are not
  orthogonal and the additive form neglects cross terms. The radial channel refers to the linear feature e.X; |X| itself
  is nonlinear (at the centred Gaussian input its Hermite expansion starts at degree two and contains higher even
  degrees), and the two agree only for thin clouds.
Evaluated with each layer's measured cloud (cosine between independent points, radial variance). ARC 256 x 32, medians
over the four nets (seeds 4002-4003 prospective for v2), measured / model:
| l | t | V/E|a|^2 | radial-feature share of V | degree-1 share | mean degree D |
|---|---|---|---|---|---|
| 0 | 32 | 0.0571 / 0.0626 | 0.01 | 0.22 / 0.29 | 21.2 / 16.0 |
| 4 | 28 | 0.0514 / 0.0565 | 0.15 | 0.48 / 0.52 | 7.7 / 5.7 |
| 8 | 24 | 0.0525 / 0.0555 | 0.32 | 0.61 / 0.63 | 4.1 / 3.3 |
| 16 | 16 | 0.0528 / 0.0562 | 0.45 | 0.77 / 0.79 | 1.9 / 1.7 |
| 24 | 8 | 0.0550 / 0.0539 | 0.51 | 0.92 / 0.91 | 1.26 / 1.18 |
| 28 | 4 | 0.0558 / 0.0531 | 0.52 | 0.93 / 0.96 | 1.10 / 1.05 |
| 31 | 1 | 0.0567 / 0.0590 | 0.52 | 0.97 / 0.99 | 1.04 / 1.02 |
The Gaussianised total is approximately flat in l empirically (0.051-0.057); only the true-law total must be.
Worst cell errors over these 28 ARC cells: 22% (budget), 0.12 (degree-1 share), 46% (mean degree, at l = 0-4, where the
model is low by about a quarter in the median). Two more nets (4004-4005, Amendment 3A, prospective) keep the budget
within 20% on 14 of 14 cells, but their mean degree at l <= 8 is 1.6-2.2 times the model's (D = 31.6, 11.7, 6.2 and
29.6, 10.9, 5.4 at l = 0, 4, 8 against 16.0, 5.4-5.9, 3.2-3.3) and their degree-1 share is 0.07-0.15 below it at
l = 4-8 (and 0.06-0.09 below at l = 0, where five of the six nets measure 0.20-0.235 against the model's 0.29). With all
six nets the medians move little (degree-1 share 0.22, 0.45, 0.57, 0.77, 0.91, 0.94, 0.98; `results/report_tables_arc6.md`,
`results/arc_accounting_arc6.json`). On four fresh nets (4006-4009, Amendment 3B) the budget held on only 19 of 28 cells
(net 4008: 0.70-0.72 of the model at l = 0-16), while the degree-1 share held on 26 and the mean degree on 25 (section
3a explains why the mean degree did better than the budget there).

Cross-lane check: the gram-chain lane's annealed fraction of the log-amplitude carried by the radius, (1/2 + C_l)/
(1/2 + C_L) = 0.46, 0.64, 0.82 at l = 4, 8, 16, with the input-bits split (amplitude 0.49, |x| 0.034), predicts radial
shares 0.26, 0.35, 0.44; measured medians 0.15, 0.32, 0.45.

### 3a. The per-net gain factor (theory v3.3 [C] and v3.4 [A]; FORMAL v3.3, v3.4)

- Exact rewriting (PROVEN [C+A]; positive homogeneity a(x) = J(x) x at sigma = 0): D = J(1)/V =
  R (tr Sigma/E|X|^2)/(V/E|a|^2) with R = (J(1)/tr Sigma)/(E|a|^2/E|X|^2), the future's mean squared gain on the cloud's
  covariance directions over its mean squared gain along the samples. The v2 identity is the annealed value R = 1;
  criticality fixes E_W J(1) and E_W E|a|^2 but neither R = 1 for a fixed net nor E_W R = 1 [A]. As scored, R is the
  reference-normalised R_ref = (J(1) under the fitted Gaussian/tr Sigma)/(E|a|^2 under the true law/E|X|^2), with
  b_ref = V (Gaussian)/E|a|^2 (true law); the identity D = R_ref (tr Sigma/E|X|^2)/b_ref is exact, and reading R_ref
  as the same-cloud gain ratio is an approximation (the two output energies differ by at most about 1.3% [A]).
- Forward-only heuristic estimates, no Jacobian [C+A]. By Theorem 1 the block of each future weight matrix transverse to the path
  is independent of that layer's gate pattern, so tangent energy across the path grows by about twice the active
  fraction per layer while the path direction grows by the radial gain. C (v3.3, HEURISTIC):
  R_q = r + (1 - r) E_x[prod_k 2 a_k(x)]/(E|a|^2/E|X|^2), r the cloud's radial variance fraction. A (v3.4; PROVEN UNDER
  conditioning on each forward trajectory, used per net as HEURISTIC): the conditional expectation
  (q_t/q_0)[r(X) + (tr Sigma - r(X)) H_t], where H_t, built from the active counts and radial gains, also carries the
  transfer of transverse energy into the next radial direction (FORMAL v3.4(b)). Cost: 1,024 sampled forward
  trajectories per cell (v3.3); 16,384 pilot inputs for the Gaussian fit plus 4,096 Gaussian trajectories (v3.4; no
  pilot at the input).
- Spread over nets [C+A]: the single-ray proxy G_t = prod_j K_j/S_j (K_j active units, S_j | K_j ~ chi^2_{K_j},
  conditioned on K_j > 0) has log mean 2t/n and log variance 4t/n to leading order (A: PROVEN UNDER that proxy; 0.25
  and 0.51 at ARC's input); a thin-cloud average gives about 0.20 and 0.35 [A, HEURISTIC]. These are scales, not
  established moments of the cloud-averaged R. The data-implied R at l = 0 over ten nets: mean log 0.17, sample SD
  0.43, a factor of about 1.5 either way; the first six nets alone gave 0.38 and 0.36.
- Budget against degree [A, PROVEN]: with K(rho) = sum_d rho^d V_d, V = K(1), J(1) = K'(1) and V_1 = K'(0), so the
  budget, the mean degree and the linear share test the integral, the end slope and the initial slope of one curve;
  a budget that holds says nothing about D. The budget does respond to R, more weakly: across the 70 layer cells of
  nets 4000-4009 (dependent within a net; a post hoc description), corr(log R, log(measured budget/v2 budget)) = 0.67
  with slope 0.30, so D/D_v2 ~ R^0.7 (my v3.3 claim that the budget does not respond to R is withdrawn, FORMAL E19; A
  had said there is no general invariance).
- Linear share [A, PROVEN]: R does not determine s_1 (s_1 = C D with the Jacobian coherence
  C = |E[J] Sigma^{1/2}|_F^2/E|J Sigma^{1/2}|_F^2). The input radius has zero first chaos (PROVEN [A]), while v2 put its
  radial channel at degree one; A's correction of that channel gives 0.238 at the input instead of 0.29. C's v3.3 rule
  (s_1 = D^{-1/2} on the direction channel) holds only inside the one-parameter small-angle model (HEURISTIC).
- Post hoc on nets 4000-4005 (NUMERICAL; `results/part3B_seen_check.json`, `results/part3C_seen_check.json`; the rules
  were written after seeing these nets):
| | v2 (annealed) | v3.3 [C] | v3.4 [A] |
|---|---|---|---|
| gain-ratio error against the data-implied R, median / max | (R = 1) | 3.6% / 27% | 2.8% / 18% |
| mean degree within [0.75, 1.33], 42 cells | 30 | 41 | 42 |
| the same at l <= 8 (18 cells) | 6 | 17 | 18 |
| degree-1 share within 0.08, 42 cells | 34 | 38 | 37 |
  The two s_1 corrections fix different cells (A's the common input bias, from 0.076 to 0.020 on nets 4001-4005 with
  net 4000 then 0.057 low; C's the high-gain nets at l = 4-8); neither is a full channel theory of s_1. C's radial term
  in R_q was added after seeing that the transverse product alone overshoots at l >= 16 (37 of 42 without it).
- Prospective test on four fresh nets 4006-4009 (Amendment 3B for v3.3, registered 01:26:28Z before the nets were run;
  Amendment 3C for v3.4, registered 02:03:18Z during their execution, before output inspection according to the author
  (reads were not logged); `results/part3B_scores.json`, `results/part3C_scores.json`). The data-implied R was
  0.61-1.36 at l <= 8, below 1 on three of the four nets. B1 and C1 each require both of their rows:
| fresh nets, 28 cells (registered target) | v2 | v3.3 [C] | v3.4 [A] |
|---|---|---|---|
| gain-ratio error against the data-implied R, median / max (reported) | (R = 1) | 5.1% / 25% | 3.5% / 19% |
| mean degree within [0.75, 1.33] (>= 22) | 25 | 21 | 22 (met) |
| the same at l <= 8 (>= 9 of 12) | 9 | 6 | 7 |
| registered joint mean-degree criterion | - | missed | missed |
| degree-1 share within 0.08 (>= 22) | 26 | 26, hit | 22, hit |
| budget within 20% (v2 Q1, >= 24) | 19, miss | - | - |
  Every early miss of the corrected mean degree is a budget miss: on net 4008, R is predicted within 1-5% but the
  measured budget is 0.70-0.72 of v2's, and the corrected D inherits that. With the measured budget the identity puts
  the co-prover's D within [0.91, 1.19] of the measured D on all 28 cells (a consequence of the R accuracy, not a
  separate test). v2's own D did better on these nets (25 of 28, including 21 of 21 on the three nets with early
  factors below one) because R < 1 and v2's budget error cancel there. Limitations of the frozen predictions (not
  clipped; scores unchanged): predicted D < 1 on 6 of 28 cells for each rule (minimum 0.897 and 0.890) and predicted
  s_1 > 1 on 4 of 28 (v3.3, maximum 1.047) and 1 of 28 (v3.4, 1.010), although the population values satisfy D >= 1
  and 0 <= s_1 <= 1.

## 4. The degree split

- Kernel (infinite width) [C+A]: s_d = Taylor coefficients of kappa^(t)(c_l + (1 - c_l) rho), normalised by
  1 - kappa^(t)(c_l); deep asymptotics s_1 ~ l/L, s_2 ~ 3 l t/(4 L^2), mean degree D ~ (L/l)^2 (PROVEN UNDER the
  concentrating-kernel limit, E5). The cloud is coherent for the future when theta_l t << 3 pi; in the input regime
  theta_l ~ 3 pi/(l + 6), so the non-linear part is small only in the last fraction of the layers.
- Finite width: the amplitude channel is leading order at ARC (kernel alone: degree-1 share 0.11 at l = 0 [A]; with the
  amplitude term 0.29 [C]; measured median 0.22, range 0.21-0.29; astra's exact Gram chain 0.26 [A]).
- Mean degree: criticality keeps E_W |J v|^2 = |v|^2, so the variance-weighted annealed mean degree is
  (tr Sigma / E|X|^2)/(V/E|a|^2) (PROVEN, annealed; used per net as HEURISTIC, E14). Per net it is multiplied by the
  gain factor R of section 3a.
- Measured (ARC, median over 4 nets [per-net range]) and model (post hoc comparison of the same surrogates):
| l | degree 1 | even-degree surrogate s_2 + s_4/4 + ... | remainder 1 - s_1 - surrogate | model (same three) |
|---|---|---|---|---|
| 0 | 0.22 [0.21, 0.29] | 0.21 [0.16, 0.25] | 0.57 [0.45, 0.62] | 0.29 / 0.19 / 0.52 |
| 4 | 0.48 [0.41, 0.59] | 0.12 [0.05, 0.18] | 0.39 [0.34, 0.45] | 0.52 / 0.14 / 0.34 |
| 8 | 0.61 [0.54, 0.72] | 0.13 [0.03, 0.17] | 0.26 [0.23, 0.30] | 0.63 / 0.13 / 0.24 |
| 16 | 0.77 [0.68, 0.81] | 0.10 [-0.04, 0.19] | 0.13 [0.09, 0.25] | 0.79 / 0.10 / 0.11 |
| 24 | 0.92 [0.83, 0.92] | 0.06 [-0.06, 0.11] | 0.07 [-0.03, 0.14] | 0.91 / 0.06 / 0.03 |
| 28 | 0.93 [0.92, 0.96] | 0.04 [-0.02, 0.12] | 0.03 [-0.05, 0.06] | 0.96 / 0.03 / 0.01 |
| 31 | 0.97 [0.96, 1.00] | 0.02 [-0.07, 0.14] | 0.01 [-0.14, 0.11] | 0.99 / 0.01 / 0.00 |
  The surrogate is the Mehler combination (2V - D(0.5) - D(-0.5))/(0.5 V), an estimate (not a confidence bound) that
  over-weights degree 2 relative to degree 4 and higher; the remainder is a lower surrogate for degree >= 3 when the
  surrogate is accurate. Columns are separate medians, so rows need not sum to 1. Sampling uncertainty of these
  combinations was not quantified (the Mehler draws are shared across correlations and their covariance was not kept);
  the comparison is descriptive. At l >= 16 the per-net values are not resolved (negative values occur); at l <= 8 the
  model matches all three medians within 0.072.

## 5. How many directions

- Bracket. Lower: the first-chaos eigenvalue tail sum_{j>k} lambda_j(C_1) lower-bounds the residual of every k-dim
  projection (PROVEN [A]); the numbers here are heuristically noise-corrected, uncertified estimates with tolerance
  sum_{j>k} lambda_j <= eps V (not eps V_1), not bounds (E12). Upper: every exhibited basis (point estimates).
- Rule v2.7 (HEURISTIC, E17): one direction for the radial and future-amplitude linear parts, the free product for the
  output-direction channel's linear part, generic capture of the angular cloud spectrum for degrees >= 2 (a model, not a
  consequence of Theorem 1, E9-E10). Known defects, disclosed: it drops chaos above degree 60 (0.055 of the predicted
  variance at ARC's input, above the 1% target) and allocates subspaces to different degrees without exhibiting a common
  one. The frozen v2 scores are kept as registered.
- Conditional lower bound for the nonlinear part (PROVEN UNDER iid Haar gate normals with diffuse nonnegative weights and
  an isotropic r-dimensional cloud [A], v3.2): every k-dim projection, future-informed or not, leaves at least about
  (1 - k/r) of the degree-2 part. Whether the networks' gate tensors concentrate enough is open (CONJECTURE).
ARC 256 x 32, output vector, medians of 4 nets: measured count (best of three bases) / v2 rule / first-chaos estimate;
scalar readout measured / v2:
| l | N(0.1) | N(0.01) | N(0.001) | scalar N(0.1, 0.01, 0.001) |
|---|---|---|---|---|
| 0 | 208 / 248 / 1 | 256 / 255 / 7 | 256 / 255 / 14 | 208, 256, 256 / 248, 255, 255 |
| 4 | 96 / 143 / 2 | 192 / 213 / 9 | 224 / 238 / 16 | 112, 176, 224 / 143, 213, 238 |
| 8 | 56 / 75 / 4 | 128 / 174 / 10 | 192 / 205 / 17 | 48, 128, 176 / 75, 174, 205 |
| 16 | 20 / 22 / 4 | 80 / 110 / 13 | 128 / 166 / 20 | 14, 80, 128 / 21, 110, 166 |
| 24 | 12 / 10 / 6 | 48 / 44 / 18 | 96 / 118 / 28 | 2, 36, 96 / 1, 43, 118 |
| 28 | 12 / 9 / 8 | 48 / 34 / 22 | 96 / 88 / 34 | 1, 18, 56 / 1, 20, 86 |
| 31 | 12 / 11 / 9 | 48 / 48 / 30 | 96 / 91 / 48 | 1, 2, 20 / 1, 2, 44 |
Per-net ranges at eps = 0.01: 256 (l = 0), 160-192 (4), 128 (8), 64-96 (16), 48-64 (24), 48 (28), 48-64 (31). Figure 1
(`figs/fig1_residual_curves_arc.png`): the full residual curves at l = 0, 8, 16, 31 of the first net; the v2 curve
follows the measured active-subspace curve, and at l = 31 the measured curve meets the first-chaos estimate. Figure 2
(`figs/fig2_arc_shares.png`): shares, mean degree and N against l.

## 6. Gates (MEASURED; PREREG Part 2 with Amendment 2A, registered before any Part 2 measurement)

G(l, eps) is the smallest tested prefix, on a 40-point geometric grid, of a fixed gate ordering such that switching those
gates (all others frozen at the mean point's pattern) brings the nonlinear residual V_>=2(g - g_G) + 2 SE below eps V.
Orderings: first-order importance |u|^2 s^2 psi(tau) (backward sensitivity x spread^2 x hinge moment at standardised
margin tau), or margin alone. G^(1) is where the sorted importances themselves leave eps of their total (the
no-interaction prediction). G is a tested count for the stated ordering, not a minimum over all gate sets. ARC 256 x 32,
medians of 4 nets (medians of grid values need not be grid values):
| l | t | future gates | within 1 / 2 / 3 cloud spreads of boundary | flipped per sample vs the mean path | G(0.1): importance / margin / first-order | G(0.01): importance / margin / first-order | N_>=2(0.1) |
|---|---|---|---|---|---|---|---|
| 8 | 24 | 6144 | 2102 / 4037 / 5391 | 547 (8.9%) | 3141 / 2826 / 2260 | 4913 / 4420 / 3928 | 56 |
| 16 | 16 | 4096 | 871 / 1795 / 2734 | 286 (7.0%) | 920 / 920 / 1030 | 1952 / 1745 / 1952 | 24 |
| 24 | 8 | 2048 | 300 / 626 / 1015 | 116 (5.7%) | 2 / 8 / 353 | 521 / 521 / 702 | 12 |
| 31 | 1 | 256 | 33 / 72 / 116 | 13 (5.2%) | 0 / 0 / 40 | 10 / 12 / 76 | 8 |
N_>=2(0.1) = number of nonlinear-active-subspace directions after which the nonlinear residual (+ 2 SE) is below 0.1 V
(10% of the total relevant variance, not of the nonlinear part).
- At l = 8-16 the tested orderings need thousands of gates to bring the nonlinear residual within 1% of V (8-19 times the
  width), close to the number of gates within about two cloud spreads of their boundary, while the nonlinear residual
  falls below 10% of V with 24-56 directions. The layerwise medians of per-sample flip fractions range from 5.18% to 8.90%; which gates flip depends on the
  sample.
- PG1 (G(0.01) >= n through l = 24, >= n t/4 at l <= 16): held on all four nets. PG2 (first-order count within 2x):
  9 of 16 ARC cells, 9 of 24 overall; it fails where the nonlinear share is already below eps (no gate needed). PG3
  (G >= 3N at eps = 0.1): 9 of 16, failing at the same late layers, where the count is linear.
- Stationary regime (n = 64, 128; t = 4, 16; PROVISIONAL: the jackknife SE does not include the outer future-noise
  dimension, only two noise realisations were drawn; re-estimation with independent outer noise pairs is outstanding):
  the nonlinear part is 4-14% of V; at eps = 0.1 the importance ordering needs no gate in 4 of 8 cells and 1, 13, 23, 25
  gates in the others; at eps = 0.01, 109-634 (importance order; 126-771 in margin order).
- Scaling (HEURISTIC [A]): gates within T spreads of their boundary over the future ~ 3 sqrt(pi) n T ln((L + 6)/(l + 6))
  (2,700 at l = 8, 1,500 at l = 16 for T = 2; measured 4,037 and 1,795).

## 7. Stationary regime (sigma = 1)

Medians over nets per (n, t), resolved cells only (V > 3 SE and memory share >= 0.02); all registered cells complete
(n = 128, 256 prospective for v2; the two n = 256, t = 256 cells dropped, section 11):
| n | t | nets resolved | memory share | degree-1 share | top AS direction leaves | radial feature leaves | N_AS(0.1) | N_AS(0.01) |
|---|---|---|---|---|---|---|---|---|
| 32 | 1 | 4 | 1.00 | 0.91 | 0.54 | 0.57 | 12 | 24 |
| 32 | 4 | 4 | 0.62 | 0.82 | 0.42 | 0.60 | 6 | 20 |
| 32 | 16 | 4 | 0.13 | 0.83 | 0.06 | 0.20 | 1 | 12 |
| 32 | 32 | 3 of 4 | 0.10 | 0.83 | 0.09 | 0.38 | 1 | 8 |
| 64 | 1 | 4 | 1.00 | 0.96 | 0.44 | 0.47 | 10 | 32 |
| 64 | 4 | 4 | 0.85 | 0.90 | 0.36 | 0.42 | 7 | 24 |
| 64 | 16 | 4 | 0.47 | 0.88 | 0.12 | 0.22 | 2 | 14 |
| 64 | 32 | 4 | 0.26 | 0.97 | 0.04 | 0.19 | 1 | 15 |
| 128 | 1 | 4 | 1.00 | 0.95 | 0.51 | 0.52 | 24 | 64 |
| 128 | 4 | 4 | 0.84 | 0.89 | 0.44 | 0.48 | 14 | 48 |
| 128 | 16 | 4 | 0.43 | 0.93 | 0.13 | 0.31 | 2.5 | 36 |
| 128 | 32 | 4 | 0.24 | 1.00 | 0.06 | 0.28 | 1 | 70 |
| 128 | 64 | 2 of 4 | 0.33 | 0.95 | 0.02 | 0.09 | 1 | 7 |
| 128 | 128 | 1 of 4 | 0.06 | 0.78 | 0.04 | 0.17 | 1 | 128 |
| 256 | 1 | 2 | 1.00 | 1.00 | 0.41 | 0.41 | 30 | 96 |
| 256 | 4 | 2 | 0.89 | 0.94 | 0.35 | 0.37 | 19 | 72 |
| 256 | 16 | 2 | 0.63 | 0.93 | 0.13 | 0.16 | 3.5 | 56 |
| 256 | 32 | 2 | 0.52 | 0.98 | 0.04 | 0.10 | 1 | 132 |
| 256 | 64 | 2 | 0.43 | 0.99 | 0.01 | 0.07 | 1 | 17 |
| 256 | 128 | 2 | 0.36 | 1.09 | 0.01 | 0.08 | 1 | 6.5 |
Degree-1 shares above 1 are sampling noise (V_1 and V estimated separately). N_AS is the registered count along the
active-subspace basis; section 5's ARC counts are the best of three bases (at n = 256, t = 32 the median best-of-three
N(0.01) is 100 against N_AS 132). The N_AS(0.01) entries at t >= 32 swing between 6.5 and 132 because the residual of
a few directions sits near 0.01 V within its standard error. The
first-chaos estimates (`results/report_tables.md`) are noise-dominated at small memory share and exceed the measured
counts at n = 256, t >= 64 (7-8 against 1 at eps = 0.1): there they are not lower bounds (E12).
- Per-net ranges at t >= 16: the top AS direction leaves 0.006-0.37, the radial feature 0.027-0.85.
- PS1 (Part 2; cells with t >= 2 n^{1/3} ln(1/eps), M >= 0.02): one direction within 10% of V on 13 of 16 seen cells
  (n <= 64) and 12 of 13 prospective cells (n = 128, 256): (a) holds at eps = 0.1. Within 1% on 0 of 8 and 1 of 7:
  (a) fails at eps = 0.01. The radial feature within 10% on 0 of 16 and 5 of 13, within 1% on none: (b) fails, as
  written in the registration. v2.8's own threshold t_1(0.1) = ln 9/(0.672 Q^{-1/3} + 1/Q - 5/n) (Part 3 Q6): one
  direction within 10% on 10 of 14 prospective cells beyond t_1 (target 75%: a miss), the radial feature not enough on
  11 of 14 (target 50%: a hit).
- The degree-1 share is 0.78-1.09 (median per (n, t); above 1 is noise); it is not >= 0.85 everywhere (P4 missed, 80 of 95; the v1 claim
  retracted, E6); the angular sector's long-time shape is s_1 = 0.715 under the frozen-radius continuum model [A]. The v2
  degree-1 share is within 0.12 on 40 of 43 prospective cells (Q2) and the v2 count within 2x on 40 of 43 at eps = 0.1
  (Q5).
- Memory: V_l decays far more slowly than 5/n at late t. Registered window t in [n/4, n] (P11b), median rate per layer
  0.047 at n = 64 (5/n: 0.078), 0.017 at n = 128 (0.039), 0.005 at n = 256 (0.020): missed at every width. Per net
  (Q7, window [n/8, n]) the rates at n = 128 are 0.071, 0.013, 0.022, 0.139 against 5/(2n) + 1/Q = 0.055, 0.024, 0.025,
  0.057, and at n = 256 0.012 and 0.002 against 0.013 and 0.010: a 10x spread between nets that no tested model (5/n,
  5/(2n) + 1/Q, B(c*)/n) follows per net. The slow coordinate is the top active-subspace direction, not proven optimal
  and not identified theoretically (astra: the leading nonconstant response of the backward Markov operator,
  CONJECTURE). The gram-chain lane's statement that the radius matters for about n layers is consistent with the n = 256
  memory share (0.36 at t = 128) but was not tested as a rate.

## 8. Narrow-deep and synchronisation (input regime, sigma = 0)

- Narrow-deep 8 x 256, six evaluation nets (true law): the radial feature alone leaves at most 2.1e-2 at l = 16 and
  8.6e-6 at l = 32, 1e-27 or less from l = 128 on. The Gaussian replacement fails there: V_gamma/V_mu = 0.51-362 at
  l >= 32.
- n = 16 (L = 256) and n = 32 (L = 512), two nets each (Part 2, PS2): the residual of the radial feature stays below 1e-3
  from l = 100 (6.25n) and 32 (2n) at n = 16, and from l = 176 (5.5n) and 80 (2.5n) at n = 32; at l = 4n it is 3.1e-3,
  1.9e-7, 1.7e-3, 3.1e-5; the final values on the grid are 9.7e-9, 1.1e-22, 4.4e-15, 1.6e-20. PS2(a) (<= 0.05 from 4n)
  held on all four nets; PS2(b) (decay at the Lyapunov-gap rate 2/n) failed or was unresolved: the residual falls
  irregularly while 0.2-2 gates per layer still differ between independent inputs. The first part of the v2.6 mechanism
  (kernel contraction to angle ~pi/n near l ~ 3n) is consistent with these depths; the second is not supported.
- Before collapse a near-ray cloud can still matter: on the n = 32 net with seed 300 at l = 64-96, the mean pair cosine
  of independent inputs' states is 0.997435-0.999966 (arccos of the mean pair cosine 0.008-0.072 rad), yet the radial
  feature leaves 11-15% (the shared future map amplifies the small angular spread).

## 9. Estimation where one direction suffices, and what the counts mean for M1

Narrow-deep 8 x 256 at sigma = 0 (MEASURED; exploratory, not registered; `scratch/est_nd.py`, `runs/est_nd.json`;
R = 40 repetitions per budget; reference from 2^22 MC samples, its own nMSE about 2.4e-7). Relevance says the layer-32
state is nearly a ray, so the truncated estimator uses the approximation a(x) ~ |h_32(x)| g(u*) with one tail map g.
Only repetition averages were stored, so confidence intervals cannot be reconstructed. Costs are nominal sampling costs in mv; the setup (64 pilot
paths for u* and one 224-layer tail pass, about 2,300 mv) is excluded, which matters at 2^12 mv and not at 2^20.
| at 2^20 mv | median nMSE | gain over plain MC (median of per-net ratios, range) |
|---|---|---|
| plain MC (256 mv per sample) | 2.4e-4 | 1 |
| RQMC on the 8 input coordinates | 3.1e-5 | 10.6 |
| radius at l = 32 x one shared tail pass (32 mv per sample) | 2.5e-5 | 9.7 (4.2-17.3) |
| the same with scrambled Sobol inputs | 2.6e-6 | 113 (50-313); 15 over RQMC (6.5-25) |
| unbiased two-level version (full-depth correction on 1/64 of the samples) | 3.3e-5 | 7.3 (4.9-11.3) |
The squared discrepancy between the truncated estimator (2^20 inputs) and the reference is at most a few 1e-6 in nMSE,
at the resolution of the two finite samples; it does not isolate a deterministic bias. Finite-budget slopes of nMSE
against compute (2^12-2^20): MC 1.03, RQMC 1.27, radius 1.03, radius with Sobol 1.19, two-level 0.99; no asymptotic rate
is claimed. The method is the shared-trajectory idea with the sharing depth set by measured synchronisation; it needs
L >> n and sigma = 0 (the narrow-deep lane found that sigma = 1e-3 moves E a by 5e-5 to 5e-3 nMSE through the deep
layers).

ARC 256 x 32 (MEASURED, 4 nets): a control variate equal to the exact conditional mean of g on the top-k tested
active-subspace directions has variance V x residual(k) (astra's formula), so its gain is at most 1/residual(k) for
those features (optimal features could do better): at the
input, the only layer whose law is known exactly, 1.05, 1.33, 1.80, 3.11 at k = 1, 16, 64, 128, before the cost of
computing the k-dimensional conditional mean. Deeper layers give larger ratios under the Gaussian replacement (16
features: 2.4x at l = 4, 4.1x at l = 8, 10x at l = 16), but using them needs the layer's own law, the forward problem the
narrow-deep and unrestricted lanes found to be the bottleneck; the Gaussian replacement already moves the mean output by
1.6e-4 to 2.7e-3 nMSE (a positively biased estimate with a sampling floor near 5e-4).

What the counts imply for M1: the tested subspaces need many coordinates for 99% predictive capture under fitted Gaussian
clouds at every layer of ARC's 256 x 32, and the tested gate orderings need thousands of gates. Whether better
future-informed projections, or integration methods that do not resolve the state, beat sampling is unresolved. N is a
predictive dimension, not an integration cost; closure-type sample-free methods are governed instead by the nonlinear
residual times the forward discrepancy (the bits-bounds lane's bound). The missing results are a certified nonlinear
lower bound for these networks and a computational-cost argument on top of it.

## 10. Scorecards (final; results/part1_scores.json 01:28:55Z, part1_scores_v2 01:29Z, part2 01:29Z, part3 01:30Z, part3A 01:08Z, part3B 02:36:43Z, part3C 02:38:06Z)

Part 1 (registered 21:23:36Z; the registered scorer `scratch/score_part1.py`, unchanged, run through
`score_part1_registered.py`, which hides the later Amendment 3A/3B nets from its file glob; post hoc corrected scorer
`score_part1_v2.py`). All registered cells are complete: ARC 4 nets, narrow-deep 6, stationary n = 32, 64, 128 (4 nets
each) and 256 (2 nets; t = 256 dropped). Every Part 1 cell was run after the registration (all are prospective for
Part 1; ARC 4002-4003 are also prospective for v2); where it matters the counts are given as group splits: ARC with
stationary n <= 64, and stationary n >= 128.
| prediction (registered threshold) | result |
|---|---|
| P1 free-product N_1 within 2x (>= 75%, each eps) | eps 0.1: miss, 87/123 (70.7%; group splits 64/80 and 23/43); eps 0.01: miss, 52/123 (42%; 38/80 and 14/43), with the registered eigenvalue count, which sampling noise biases low; a post hoc alternative count (degree-1 capture along the AS basis) gives 101/123 (82%) and 84/123 (68%). Forward-only proxy >= 3x measured: hit, 91/91 |
| P2 degree-1 share within 0.12 (>= 75%, each regime) | hit: ARC 27/28; stationary 86/95 |
| P3 mean degree within 1.5x (v1; >= 60%) | hit, 19/28 (68%); v1 summed a truncated series |
| P4 stationary degree-1 share >= 0.85 (>= 90%) | miss: 80/95 (84%; 39/52 at n <= 64, 41/43 at n >= 128); claim retracted (E6) |
| P5 frozen-at-mean variance in [D/2, 2D] (>= 75%; >= 1.5 V on all) | hit, 12/13; 13/13 |
| P6 generic capture of the cloud-axes basis (>= 75%) | miss, 49/119 (41%) |
| P7 vector N within 2x (>= 70%), eps 0.1 / 0.01 (0.001 reported) | hit: 107/123 (87%) / 103/123 (84%); 0.001: 106/123 |
| P8 scalar N within 2x (>= 70%) | eps 0.1: hit, 92/123 (74.8%; group splits 58/80 and 34/43); eps 0.01: miss, 81/123 (65.9%; 57/80 and 24/43); 0.001: 98/123 |
| P9 (a) N(0.01) >= 64 at l <= 8; (b) >= 16 everywhere; (c) scalar N(31, 0.01) <= 4; (d) narrow-deep radial residual <= 0.01 at l >= 16; (e) stationary N(0.1) <= 2 at t = 16, 32 | (a) hit 4/4; (b) hit 4/4; (c) hit 3/4; (d) hit 5/6; (e) hit 15/20 (75%, at the threshold) |
| P10 (a) stationary Gaussian fidelity at t <= 8 (>= 75%); (b) narrow-deep failure (>= 4 of 6) | hit 40/40; hit 6/6 |
| P11 (a) V non-increasing in t (>= 90%); (b) late rate within 1.6x of 5/n, n = 64, 128, 256 | (a) hit 94/94; (b) miss at all three (0.047, 0.017, 0.005 against 0.078, 0.039, 0.020) |
Part 2 (21:36:43Z; Amendment 2A 22:08:11Z, before any Part 2 measurement): PG1a hit 4/4; PG1b hit 4/4; PG2 miss 9/24;
PG3 miss 9/16; PS1(a) at eps 0.1 hit (13/16 seen, 12/13 prospective); PS1(a) at eps 0.01 miss (0/8, 1/7); PS1(b)
"radius suffices" miss at both (0/16 and 5/13 at eps 0.1; none at 0.01), as written in the registration; PS2(a) hit 4/4;
PS2(b) miss (two fits outside, two unresolved).
Part 3 (prospective test of theory v2, 21:59:37Z). ARC seeds 4002-4003: Q1 budget hit 12/14; Q2 degree-1 share hit
13/14; Q3 mean degree hit 11/14; Q4 N within 2x hit 14/14 at eps 0.1 and 14/14 at 0.01; first-chaos estimates below the
measured counts on every cell. Amendment 3A (registered 00:35:48Z before seeds 4004-4005 were run; same criteria): Q1
hit 14/14; Q2 miss 10/14; Q3 miss 8/14; Q4 hit 14/14 at both eps; first-chaos estimates below the measured counts
everywhere. The misses are at l = 0-8 (degree-1 share 0.06-0.15 below the model; model mean degree 0.46-0.62 of the
measured). Pooled over the four nets with the per-14 thresholds doubled (the pooled thresholds were not registered
separately): Q1 26/28 (>= 24) hit; Q2 23/28 (>= 22) hit; Q3 19/28 (>= 22) miss; Q4 28/28 at both eps hit. Stationary
(n = 128, 256, final): Q2 hit 40/43 (>= 75%); Q5 hit 40/43 (>= 60%); Q6 one direction beyond t_1 miss 10/14 (target
75%), radius not enough hit 11/14 (target 50%); Q7 per-net rates in section 7 (no band).
Amendment 3B (theory v3.3 and a third v2 test; registered 01:26:28Z before nets 4006-4009 were run; 28 cells): B1
(mean degree in band on >= 22 of 28 and on >= 9 of 12 at l <= 8) missed, 21/28 and 6/12; B2 degree-1 share hit 26/28 (>= 22); v2 on the same
cells: Q1 budget miss 19/28 (>= 24), Q2 hit 26/28, Q3 hit 25/28, Q4 hit 27/28 and 28/28; reported: R_q within 1.33x of
the data-implied R on 28/28. Amendment 3C (theory v3.4 [A]; registered 02:03:18Z during the runs, before output
inspection according to the author; reads were not logged): C1 missed its joint criterion: it met the all-cell part
(22/28, >= 22) and missed the early part (7/12, >= 9); C2 degree-1 share hit 22/28 (>= 22);
reported: gain-ratio error median 3.5%, max 19%. Theory v2 over all eight prospective ARC nets (4002-4009, 56 cells;
the per-14 thresholds scaled, not separately registered): budget 45/56 (>= 48) miss; degree-1 share 49/56 (>= 44)
hit; mean degree 44/56 (>= 44) hit at the threshold; N 55/56 and 56/56 hit.

## 11. Process

- Co-prover gpt-6-astra. Round 1 (21:10-21:34Z, independent; my v1 frozen 21:17:35Z before reading it) agreed on
  Theorem 1, the kernel degree spectrum and its asymptotics, the leading-order amplitude correction and the stationary
  angular rate, and added the first-chaos lower bound, the stationary angular limit, a radial-rate model and the
  estimation caveats. Round 2 (21:48-22:04Z; audit of v1-v2, code, scorers): errata E2-E17 in FORMAL.md (two theory
  steps withdrawn: generic capture as a consequence of Theorem 1, and the v2.4 lemma), Amendment 2A, the corrected Part 1
  scorer. The correction to Corollary 1b (E1) came from the bits-bounds lane's round 2. Round 3 (from 01:28Z; an
  independent derivation of the per-net mean-degree factor, given the raw per-net data but not v3.3, which was frozen at
  01:26:28Z before the round started; finished 01:42:53Z, model line gpt-6-astra) found the same exact rewriting, the
  same kind of forward-only predictor and the same single-ray law [C+A], and added a more accurate trajectory-conditional
  predictor, the proof that R does not determine s_1, the proof that the input radius has zero first chaos (with the
  radial-channel correction of v2's s_1) and a correction of my budget-invariance claim [A]; its rules were implemented
  from its text and registered as Amendment 3C before any output of the fresh nets was read. Details:
  `coprover/PROVENANCE.md`.
- Deviations (`logs/deviations.txt`, with evidence pointers): (1) the stationary design was cut for load at about 21:38Z
  (`scratch/queue_B2.sh`), restored at main's request at 21:50:32Z, and the two registered n = 256, t = 256 cells were
  dropped at main's request at 22:53:08Z (compute goes to the bits-accounting lane; theory predicts their memory share
  below the 0.02 resolution threshold); (2) one stored field (cloud_evals_perp) added after the Part 1 hash, registered
  fields unchanged; earlier runs get it from a fresh forward sample; (3) the registered N_1 measure is biased low by
  eigenvalue noise (about n/(P s_1) of V_1; 21% at ARC's input), reported both ways; (4) Amendment 2A before any Part 2
  measurement; (5) a light third process (n = 8 estimator test, about 13 minutes) ran alongside the two queues at load
  > 16; (6) the Part 2 text names "main's request, 21:40Z" and "main 21:45Z": these were clock times I guessed and typed;
  both messages were read before the Part 2 hash at 21:36:43Z, which is authoritative; (7) post hoc analyses labelled as
  such (corrected Part 1 scorer, free-position check, ARC degree surrogates, control-gain bound, slow-mode candidates,
  the narrow-deep estimator test); (8) the registered Part 1 scorer, unchanged, was run through a wrapper that hides the
  later Amendment 3A/3B nets from its file glob (they are not Part 1 cells); (9) theory v3.3 was written after reading
  Amendment 3A's results, so its counts on nets 4000-4005 are post hoc; Amendment 3B tests it on fresh nets; (10) the
  final scorers ran as short single-thread processes next to the two Amendment 3B runs at load about 28; (11) the
  co-prover's v3.4 rules were registered (Amendment 3C, 02:03:18Z) while the fresh nets were running: none had
  finished (the first finished at 02:04:07Z), and their partial output files had not been read (my statement; file
  reads are not logged).
- Amendments after the second review: 3A (00:35:48Z; ARC nets 4004-4005, same criteria as Part 3), 3B (01:26:28Z;
  theory v3.3 and a third v2 test on nets 4006-4009) and 3C (02:03:18Z; the co-prover's v3.4 on the same nets,
  registered during their execution, before output inspection according to me; reads were not logged), each hashed
  with its scorer. Amendment 3A's registered PREREG is kept as `frozen/amend3A/PREREG_as_hashed.md` (the first 12,911
  bytes of the later frozen PREREG; sha a00177f9, as in `prereg_hashes.txt`).
- Compute: CPU only, nice 19, ionice -c3, one BLAS thread per process, two heavy processes (load 20-47, mostly other
  jobs); no GPU.

## 12. Review

GPT-6 (gpt-6-astra, fresh session, high effort, read-only; 23:11-23:16Z; `review/review_gpt6.md`): verdict major
revision, 15 findings; it reproduced 12 reported numbers and all 18 preregistration hashes. Changes:
1. M1 headline replaced by the reviewer's wording: many coordinates for the tested subspaces; better projections or
   integration methods unresolved; a certified nonlinear lower bound is the missing result (Summary, section 9).
2. Theorem 1 restricted to the angular part and the frozen Jacobian at a nonzero mean in the Summary; the free-product
   model for the averaged Jacobian labelled HEURISTIC; the participation-ratio agreement rescoped and extended to all four
   nets (21 of 24 non-input cells within 20%).
3. Radius and the linear radial feature e.X distinguished throughout; "flat as it must be" replaced by "approximately
   flat empirically under Gaussian replacement"; channels called non-orthogonal and heuristic.
4. The degree rows renamed (degree 1, even-degree surrogate, remainder), per-net ranges added, negative estimates shown,
   "interactions dominate" removed, separate-median caveat stated.
5. Accuracy claims replaced by the registered prospective results (12/14, 13/14, 11/14) and the worst-cell errors over
   all 28 cells (22%, 0.12, 46%).
6. Counts labelled point estimates for exhibited subspaces, first-chaos numbers labelled uncertified estimates (figures
   relabelled), the eps V normalisation stated, v2's omitted chaos mass (0.055) and incompatible allocations disclosed.
7. Stationary gate counts marked provisional (outer future-noise uncertainty missing).
8. Gate statements corrected: zero gates at eps 0.1 in 4 of 8 stationary cells (others 1, 13, 23, 25); N_>=2 threshold
   stated as 10% of V; G described as the smallest tested prefix for an ordering; necessity and normal-span claims removed.
9. Stationary statements: medians labelled, per-net ranges given, "floor" replaced by "observed residual at these
   depths", "best single direction" by "top active-subspace direction", PS1 counts given with seen/prospective split.
10. Synchronisation corrected: the four persistent 1e-3 crossing depths (2n-6.25n), final residuals 1e-8 to 1e-22, the
    4n entry for n = 32 seed 301 (3.1e-5), and the n = 8 statement (below 1e-5 by l = 32).
11. Narrow-deep estimator: equal nominal sampling cost with the setup cost stated, "bias" relabelled as a finite-sample
    discrepancy, finite-budget slopes reported, labelled exploratory and unregistered.
12. Numbers corrected: RQMC gain 10.6x (median of per-net ratios), stationary rate 0.047, gate flips 5-9% per sample,
    control gains 1.33/1.80/3.11, the narrow-deep Gaussian-fidelity ratio orientation and range.
13. Scorecard rebuilt with numerators and denominators, completion status and seen/prospective status per test; the post
    hoc P1 count reported as a miss (59/86); Part 3 interim stationary results (Q5-Q7) listed.
14. Chronology corrected from the logs (21:50:32Z, 22:53:08Z), missing deviations added to the log with evidence, the
    Part 2 embedded-time inconsistency disclosed; "before any Part 2 measurement".
15. Technical fixes: J and K defined; the empirical mean at l = 0 noted; kernel-share normalisation stated; the
    cloud-axes number described as a forward-covariance proxy; P5 reported as 12/13.

Round 2 (GPT-6, fresh session, 23:21-23:27Z; `review/review_gpt6_r2.md`): verdict minor revision; six first-round
findings resolved, nine partly resolved, four new minor items; it recomputed the free-position ratios on all four nets
(21/24), the synchronisation crossings, the estimator gains and slopes, the degree-table ranges, the prospective scores
(12/14, 13/14, 11/14, 14/14) and the 18 hashes. Changes: the opening now names degree 1, an even-degree surrogate and a
remainder (no gate attribution) and says the surrogate comparison is exploratory with unquantified sampling uncertainty;
"best input features" became "top tested active-subspace features" (not a bound for optimal features); the radius
expansion is described as starting at degree two with higher even degrees; first-chaos numbers are "heuristically
noise-corrected, uncertified estimates" in the text, the table generator and the figures; stationary gate counts stay
provisional with re-estimation outstanding, and the counts name their ordering (margin order 126-771); flip fractions are
stated as medians 5.2-8.9%; the stationary figure legend says "top active-subspace direction"; the truncated estimator's
ray relation is written as an approximation and the missing repetition-level uncertainty is stated; the near-ray example
names its net and reports the arccos of the mean pair cosine (0.008-0.072 rad); the scorecard gives the snapshot times and
splits pooled rows into completed and interim cells (P8 at eps 0.1 is a hit on completed cells, 58/80, and an interim
miss pooled); 0.78 -> 0.77 and "within 0.07" -> "within 0.072".

Round 3 (GPT-6, fresh session, 02:43-02:53Z, on the document with sha 686152e3; `review/review_gpt6_r3.md`): verdict
minor revision, 13 minor items. It matched all 34 preregistration hashes; recomputed both forward predictors on the 28
fresh cells with the registered seeds (exact match), the B1/B2/C1/C2 and v2 counts, the ten-net R statistics, the
budget correlation and slope, two stationary rows, the P1/P8 group splits and the pooled Part 3 counts; and ran the
Part 1 wrapper in memory, reproducing every stored score. Changes:
1. Section 6's flip sentence now gives the layerwise medians of per-sample flip fractions, 5.18% to 8.90%.
2. Amendment 3C is described everywhere as registered during the runs, before output inspection according to me, with
   reads not logged; 3A's registered PREREG is kept as `frozen/amend3A/PREREG_as_hashed.md` (sha a00177f9).
3. The scored factor is named the reference-normalised R_ref, with b_ref; the identity is exact for them, and reading
   R_ref as the same-cloud gain ratio is stated as an approximation (section 3a, FORMAL E20).
4. The forward rules are called heuristic estimates of R (gate counts and radial gains; the co-prover's adds radial
   transfer), with their sampling costs; the general claim that D is forward-predictable became a statement about the
   tested cells (Summary, section 3a, FORMAL E20 replacing v3.3(g) and v3.5(d)).
5. The single-ray moments belong to the proxy G_t conditioned on survival; the concentration rate uses the effective
   rank (FORMAL E20).
6. B1 and C1 are reported as missing their registered joint criteria (C1 met the all-cell part, 22/28, and missed the
   early part, 7/12) in the Summary, section 3a, section 10 and FORMAL v3.5; B2 and C2 stay hits.
7. The frozen predictors' out-of-range outputs are disclosed: predicted D < 1 on 6 of 28 fresh cells per rule, predicted
   s_1 > 1 on 4 (v3.3) and 1 (v3.4).
8. Sample SD is named; the budget correlation is called a post hoc description across 70 dependent cells; the budget
   error is stated as measured/model (down to 0.70, a 42.6% model error), not "up to 30%".
9. The stationary counts are labelled N_AS, with the best-of-three value where it differs (100 against 132 at n = 256,
   t = 32).
10. All four P1/P8 group splits are given and called group splits (all Part 1 cells are prospective for Part 1).
11. "25 of 28 there" now reads 25 of 28 fresh cells, including 21 of 21 on the three nets with early factors below one.
12. FORMAL E20 corrects E3 (21 of 24 cells), E13 (the radius expansion starts at degree two, with higher even degrees),
    E14 (superseded by the scorecards); the Summary's narrow-deep bound is 2.1e-2; the 3B and 3C scoring times are given
    separately (02:36:43Z, 02:38:06Z).
13. The Summary carries items 3-6, 11 and 12.
