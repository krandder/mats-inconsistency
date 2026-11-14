# Future-aware layer-wise estimators for ARC's deep random MLPs: what a backward pass buys at 256 x 32, and why it did not change the verdict here (2 November 2026)

Status: final for the registered Part 1 and the post hoc analyses after GPT-6 review rounds 1-4 (round 4: minor
revision, applied; section 8). Added after round 4: Amendment 1 (n = 512, nets 500-502, scored; 4.5), the n = 512
relevance spectra, and the mixture results corrected for a sampling bug (section 5); verification review round 5
(minor revision, applied) is in section 8.

Lane: future-aware (target M1), bead pool-ttdzrk.14 under pool-ttdzrk. Work directory `work/future-aware/`:
`FORMAL.md` (v1 frozen 10:39:14Z before the co-prover's round 1; v2 11:28:22Z before round 2; v3 before reading round 2;
v4 before round 3; v5 = E13-E14 after it),
`PREREG.md` (Part 1, hashed 11:21:08Z with all code; `prereg_hashes.txt`, frozen copies in `frozen/part1/`),
`coprover/` (gpt-6-astra rounds, `PROVENANCE.md`), `lib/` (fa.py, fa_kprop.py; fa_kprop4.py exploratory),
`scratch/` (run_cut, run_rec, run_k3p, run_hyb, score_part1, tables_part1; queue scripts), `runs/`, `results/`,
`logs/` (deviations.txt). Labels: PROVEN, PROVEN UNDER <hypothesis>, NUMERICAL, MEASURED, HEURISTIC, CONJECTURE.
Provenance: [C] me, [A] gpt-6-astra, [prior: lane] an earlier lane's report.

## Summary for Kelvin

**Verdict: the negative, for the future-aware extensions tested (at n = 64-256 along L = n/8, and at n = 512 for the
deductive member, ARC's ladder and the relevance spectrum),
with a measured mechanism (HEURISTIC as a general explanation), pricing theorems for the classes that can be priced, and
a precise statement of what is open. No major positive was found; this is not a theorem about every future-aware
method.** A backward pass does find where the future will read a closure's error, and that information is real, but in
these nets it is spread over a number of directions that grew with the width over n = 64-512 (about linearly in the
first layers, where the closures' errors are made), and keeping the law right in that many directions was, for every
representation tried, either sampling or ARC's ladder again.

1. **What the backward pass finds (benchmarks, registered).** Keeping the cloud's law exact in the k directions the
   future reads most (from vector-Jacobian products), Gaussian elsewhere, cuts the Gaussian closure's full-recursion
   error at 256 x 32 by 19x at k = 64 and about 200x at k = 128 (finite-particle benchmark, 5 nets); the cloud's own
   principal axes do worse at the same k = n/4 by a median 2.0x, 2.7x, 2.9x over nets (n = 64, 128, 256). At fixed
   k = 16 the gain shrank with width (5.2, 2.3, 1.4x): the future's whitened nonlinear relevance has a participation
   ratio that grew with n (17.5, 33.3, 67 at the first layer, and 127.5 on three n = 512 nets; slower at fixed l/L); at the
   first layer its top n/4 directions hold a constant ~80% of it at every width while the top 16 hold 0.80, 0.59, 0.39,
   0.25 (n = 64-512), and the improvement any k-dim choice
   gives (future, PCA or random) tracks the share s of it that the k directions capture, about (1 - s)^-1.2 (post hoc,
   379 cells at n = 64, correlation 0.96; 0.90 at n = 128).
2. **Carrying the law there.** By particles it is a sampler costing ~6 passes per particle and, at its flop-counted
   cost, its error is 45-250x RQMC's (an extrapolated RQMC comparison; it is also worse than plain MC). Most of what the
   future reads in the kept span is dependence between the kept directions: making them independent with exact
   marginals leaves 0.72 of the Gaussian's single-cut error, against 0.003-0.19 for the joint law (exploratory, n = 64);
   whether a few mixed moments or oblique features would do is open. Deductively, ARC's own order-3 propagation with
   its third cumulant kept only in the k future directions keeps most of k3's error factor at k = n/4 (median per-net
   retained factor 0.86, 0.82, 0.80, individual nets 0.26-2.0; medians 0.78-0.91 with a deductive mean-field backward
   sweep of ~2n passes), much less at fixed k = 16 (0.86, 0.46, 0.28); at n/2 most runs failed numerically at 256 (1 of
   5 succeeded: ARC's K = 3 state covariance becomes indefinite; with a post hoc rank guard they run, but 3 of the 8
   rescued runs diverge or lose badly). Order-4 precision needed the third cumulant in about half the directions or more
   (exploratory, n = 32 and 64, few nets; on some nets half was not enough).
3. **Windows: "beats RQMC by X up to precision Y".** The deductive member as implemented loses by 6-200x at n = 64-256
   (medians per variant; 150-400x at 512). Under the registered hypothetical cost of a dedicated implementation it would win by 2.6, 3.0, 2.4x up
   to 5.9e-4, 3.1e-4, 2.2e-4 (n = 64, 128, 256; windows end at 0.6, 1.6, 2.8 thousand passes). That cost omits work the
   method needs: charging the separate K = 2 pilot that supplies its gate probabilities gives 1.8, 2.2, 1.75x, and
   charging also the per-layer extraction of the k future directions gives 0.5, 0.7, 0.5x with the dense eigensolvers
   used here (approximate flop count), or 1.3, 1.5, 1.3x under an assumed ~2n-pass extraction whose feasibility is not
   established (ARC's states are indefinite, so plain Cholesky whitening fails, and approximate directions are
   untested; post hoc ledger, 4.4). At best a small constant factor, not growing over n = 64-256, for an unbuilt
   implementation; at n = 512 the n/4 version fails numerically and the n/8 version's hypothetical gain falls to 1.05x
   (0.74x with the pilot), while ARC's own K = 2 drops below RQMC (0.84x; 4.5). Hybrids (future-built control variates with closure means, RQMC on the residual), scored directly
   against the truth, are near unity against RQMC on the same scrambles: 1.00-1.03x at 6.5e4 passes and 0.86-0.95x
   (losses of about 5-15%) at 1e4 passes, with the controls applied to sums (post hoc; statistical uncertainty not
   quantified; the registered score, a sum that omitted a cross term, gave 0.8-1.0x): controls with exactly known means
   remove what scrambled Sobol already integrates, and deeper controls inherit the closure's bias.
4. **Theory.** Conditional pricing holds for any future interface; ARC's L^2 sketching is the annealed case (Theorem A's
   kernel); an additive correction that sees the next layer through k fixed directions removes at most the radius term
   plus a fraction k/r of the positive-degree price of a fixed defect (sharp); for future-chosen spans the tested
   projection removes exactly the future's Hermite modes inside the span when the future's reading is polynomial (in
   general an exact series-free identity holds; post-ReLU clouds have no L^2 density against the Gaussian, so for the
   actual nets the mode statement is a formal description), and under a moment model or in a Haar linear model at most
   the future's top-k spectral share. What fails unconditionally is the domination of realised by annealed price
   (future-adapted defects can have zero error); a lower bound for every future-aware layer-wise estimator would be a
   super-linear time lower bound for an explicit problem, which remains open.
5. **What ARC should update on.** Using the realised future is the natural step beyond ARC's law-based sketching, and
   here it bought constant factors set by the captured share of the future's nonlinear reading. Two concrete tests for
   any proposal at L = alpha n (each under stated hypotheses, section 2.5): with cost Theta(n) passes and a sampler of
   bounded gain, its nMSE must fall faster than 1/n; a truncated hierarchy must keep its retained dimension per order
   below its measured per-order error ratio (measured e_k2/e_k3: 4.7, 6.7, 11.9, then 2.95 at n = 512). Nothing tested
   here or in arc-update was observed to pass either. The
   cheapest positive left is a better order-3 variant (ARC's k3 with its third cumulant projected on n/4 future
   directions), worth at most a small constant factor if built: 1.75-2.2x over RQMC at n = 64-256 before its per-layer
   direction extraction is charged, about 1.3-1.5x if that extraction cost an assumed ~2n passes (not yet a feasible
   method), below 1 with the dense extraction used here; the open route is a cheap deterministic representation of the
   joint law in ~n/4 future directions (mixed moments, oblique features, implicit descriptions), which nothing here
   provides.

## 1. The question
arc-update (outbox/arc-update-2026-11-02.md) proved that a FORWARD layer-wise estimator pays for its model defect at
layer l through the future's rotation-invariant reading (Theorem A) and measured that every tested forward closure
gives at most constant factors over sampling along L = n/8. Its largest named gap: future-aware representations. A
future-aware estimator may use the weights after layer l when it decides what to keep at layer l. The relevance lane's
decoupling theorem says the future's relevant directions are uniformly oriented relative to everything the past
computes, so only a backward pass can find them. This lane asks whether that information changes the verdict, on
ARC's 256 x 32 and along L = n/8 (n = 64, 128, 256), at equal compute with the backward pass counted, against RQMC,
the multilevel shared-trajectory sampler and ARC's own K = 2. Three natural extensions were registered:
(a) a backward-informed closure: the law kept exactly in the k future-relevant directions, Gaussian elsewhere;
(b) future-weighted cumulant truncation: ARC's third cumulant kept only in the k directions the future reads;
(c) a hybrid: a future-built control variate whose mean comes from a closure, RQMC on the residual.

## 2. Theory

### 2.1 What "future-aware" means, and the pricing that survives
A layer-wise deduction-projection estimator (ARC's "expectation machine" form; arc-update 2.1) keeps a model Q_l of
the layer-l law; its error telescopes, est - Y = sum_l <g_l, Delta_l>, Delta_l = Q_l - Phi_l Q_{l-1}, with g_l the map
from layer l to the output. It has FUTURE INTERFACE S_l if Q_l depends on the future weights only through a statistic
S_l (forward-only: S_l trivial; fully future-aware: S_l = all future weights).
- Conditional pricing (PROVEN [C+A], FORMAL F-2): for a defect measurable from (past, S_l),
  E[ |<g_l, Delta>|^2 | past, S_l ] = iint K_S dDelta dDelta, K_S(h, h') = E[g_l(h) . g_l(h') | past, S_l].
  K_S splits into the part of the reading predictable from S_l and a residual part; only the residual part is a lower
  bound for defects the method cannot correct.
- ARC's "mechanistic L^2 sketching" is the annealed case (PROVEN identification [C], exact eigenvalues [A]): ARC keep
  the top eigenfunctions of the tail kernel built from the tail's LAW. In the MLP setting that kernel is Theorem A's,
  |h||h'| kappa_t(u . u'); on a rotation-invariant reference law its eigenfunctions are r Y_{j,a}(u) (radius times
  spherical harmonics), eigenvalue lambda_j = m_2 sum_{d >= j, d - j even} b_d d! Gamma(n/2) / (2^d ((d - j)/2)!
  Gamma((n + d + j)/2)) [A]. ARC's own rule therefore prescribes the radius first (arc-update's radial closure is its
  first step), then radially weighted angular harmonics; on the actual anisotropic cloud the operator must be
  diagonalised directly [A]. The realised kernel (S_l = everything) has rank <= n and its optimal sketch is the answer
  itself (PROVEN): realised sketching is trivial in information and the whole problem in computation.

### 2.2 The k-sketch theorem (PROVEN [C+A])
If the layer-l projection sees the first future layer only on k past-measurable directions V (S = W_{l+1} V and all
later layers), then given S the rest of W_{l+1} is a fresh Gaussian block, and Mehler's formula in the complement gives
K_S(h, h') = sum_alpha C_alpha(a, r) . C_alpha(a', r') (v . v')^{|alpha|} with a = V^T h, r and v the complement's
radius and direction (FORMAL section 3; astra r1 Q2). For a FIXED past-measurable defect, the best ADDITIVE correction
any S-measurable computation can make leaves exactly (this does not cover interface-dependent reconstructions, which
can change the defect itself [A, E13.5])
  sum_d b_d ( ||M_d||_F^2 - ||P^{tensor d} M_d||_F^2 ),   P = V V^T,   M_d = int |h| (h/|h|)^{tensor d} dDelta,
and ||P^{tensor d} M_d||^2 <= min(1, k / r_inf(M_d)) ||M_d||^2 for d >= 1, with r_inf the stable rank of the
flattening [A]. The d = 0 (radius) term b_0 |M_0|^2 is fully predictable from S (S reveals all later layers), so a
k-sketch correction can remove at most b_0 |M_0|^2 + sum_{d >= 1} b_d min(1, k / r_d) ||M_d||^2: the radius term (which
mean-and-covariance matching does not cancel; radius matching does) plus a fraction k / r of the positive-degree price
(review finding 1). k = o(n) cannot remove a non-vanishing diffuse positive-degree share [A, PROVEN UNDER the
fixed-defect, sketch-only model]. High participation ratio alone does not give (k/n)^d (counterexamples [A]).

### 2.3 Future-chosen subspaces (PROVEN in a linear model; HEURISTIC for the ReLU future)
When the k directions are chosen from the future (the top active-subspace directions of g_l), Euclidean overlap of the
defect with a uniformly oriented subspace does NOT bound what is removed (a rank-one future removes the whole price with
overlap 1/n [A]). For an independently Haar-oriented LINEAR future with singular values s_i, keeping the top k
directions removes exactly the share E sum_{i <= k} s_i^2 / E sum_i s_i^2 of a fixed defect's expected price [A]: what
matters is the spectral concentration of the future's reading, not the defect's rank. For the ReLU future this is a
model (the anisotropic cloud breaks the needed equivariance [A]). Two further statements for the projection actually
tested ("k features exact, linear-Gaussian complement") [C]:
- Exact identity (PROVEN under integrability [C+A], FORMAL E10, E13.1): in whitened coordinates z the projection keeps
  Pi z (Pi the projector on the kept span) and replaces the rest by fresh Gaussian noise eta, so its single-cut defect is
  int g d(Q - P) = E_P[(T_Pi - I) g], T_Pi g(z) = E_eta g(Pi z + eta). For polynomial readings (or when the cloud has an
  L^2 density against the Gaussian, which post-ReLU laws do not) this says the projection removes exactly the future's
  Hermite modes lying inside the span. Removing modes need not lower the squared error (cancellations [A]).
- Pricing under a moment model (PROVEN UNDER a conditional covariance model for the cloud's higher moment tensors [A],
  FORMAL E13.3, replacing my withdrawn E12): if, given the future and the chosen span, the cloud's normalised degree-d
  moment tensors are centred, uncorrelated across degrees and have isotropic covariance sigma_d^2 I, then any k-dim
  span leaves E|D_V|^2 >= B_0 - sum_{i <= k} lambda_i(A_*), A_* = sum_d sigma_d^2 G_{d,(1)} G_{d,(1)}^T (the future's
  coefficient flattenings), B_0 = tr A_* the expected Gaussian defect: a future-chosen span removes at most the top-k
  spectral share of A_*. Whether this moment model describes the actual clouds is a CONJECTURE.
The measured relevance spectra and improvements are in section 4.1.

### 2.4 What does and does not extend unconditionally
(i) PROVEN [A]: even for one ReLU layer there are future-adapted defects that preserve any finite list of moments, have
zero realised error and positive annealed price, so no inequality "realised price >= c x annealed price" holds once the
defect may depend on the future: Theorem A's annealed pricing does not extend to future-aware defects. (ii) The
conditional pricing identity of 2.1 does hold unconditionally (given measurability and integrability). (iii) Any
algorithm that outputs an estimate from the weights is a fully future-aware layer-wise estimator of the same cost (the
state can carry the answer), so a lower bound for all future-aware layer-wise estimators would be a super-linear time
lower bound for an explicit problem: not shown impossible, but beyond every known technique, and open. The theorems
here therefore restrict the interface (2.2), the projection family (2.3) or the representation (2.5).

### 2.5 The future-weighted ladder and the width criterion (PROVEN arithmetic under stated laws [C]; E3, E7)
- If an order-K hierarchy whose tensors of order >= 3 are kept in k* dimensions has errors e_K >= a q^{-K} and costs
  C_K >= c (k*)^{K-1}, uniformly over the orders used (both are hypotheses, not established by finite-order data), then
  e_best(C) >= (a/q) (C/c)^{-gamma*} with gamma* = log q / log k* (with bounded adjacent cost spacing [A]). The dense
  hierarchy has k* = n (arc-update 2.5). Under these laws truncation reaches sampling's rate only if k* <= q. The
  per-order ratio q is measured, not bounded: e_2/e_3 has median 4.7, 6.7, 11.9 at n = 64, 128, 256 and 2.95 at 512 here (ARC 'simple'
  k2 to k3), and arc-update measured 2-4 from order 3 to 4 at n <= 128; ARC's conjectured law has q ~ n/L, which is an
  upper-bound conjecture, not a bound on q (review finding 3).
- Along L = alpha n, the multiplicative window of a deterministic method with nMSE e(n) at cost C(n) passes against a
  sampler of gain G(n) grows only if G(n) C(n) e(n) -> 0 (PROVEN arithmetic; E7, E13.7; astra r1 Q6). For an
  implementation charged C(n) = Theta(n) passes (every method here that carries an n x n state) and G bounded (RQMC's
  measured G ~ 1.2-2.2), this means e(n) = o(1/n) in nMSE. No universal cost floor is claimed for other algorithms.
  ARC's fixed orders have errors that change little with n at fixed L/n (gain falls like 1/n); the radius-corrected
  closure has ~alpha/n (gain constant) [prior: arc-update].

## 3. What was run (PREREG Part 1, hashed 11:21:08Z with all code; Amendment 1 at 12:17:16Z)
Nets: 256 x 32 seeds 200-204 (arc-noisy's noiseless 2^26-path truths); along L = n/8, n = 64 (L = 8) and n = 128
(L = 16) seeds 300-307 (arc-update's 2^24-path truths); Amendment 1: n = 512, L = 64, seeds 500-502 (2^22-path truths
computed here). Cost unit: one forward pass = 2 n^2 L flops; every cost includes the backward work used to find the
future directions (VJPs 3 x 2 n^2 t flops per sample through t layers; basis VJPs for E[J]; mean-field backward sweeps
2-4 n^3 per layer); linear algebra is charged by the formulas in each script, and the ledger at the end of this section
lists what is not charged. Future directions: "futnl" = whitened top-k active subspace of the
NONLINEAR part of the future map g_l (E[J^T J] - E[J]^T E[J], from VJPs on the cloud); "futv" = the whitened active
subspace from VJPs at samples of a method's own Gaussian state; "futmf2" = the deductive mean-field E[J^T J] with
independent gates (one n x n backward sweep). Past-measurable comparators: "pca" (top cloud covariance axes), "rand".
- B1, single-cut BENCHMARK (not an estimator: true samples carry the kept law). The true layer-l law is replaced by
  "k features exact, linear-Gaussian conditional complement (mean and covariance exact)" ("cond"), optionally with
  arc-update's radial transport ("radg"), or by the oracle Gaussian times an order-3 Gram-Charlier weight with the k
  features' true third cumulant ("edge", (b) as a benchmark), and pushed through the exact rest of the net; b2 is the
  unbiased squared single-cut defect (two halves, common random numbers across variants; 2^18 pushes).
- B2, full-recursion BENCHMARK of the closure (a): the same projection at every layer, the k-dim law carried by 2^18
  particles, the complement regenerated from the push's own mean and covariance; two replicates, b2 = cross product.
  It is also scored as an estimator at its flop-counted cost (about 6 passes per particle per run plus the backward
  work), because carrying a k-dim law by particles is a hybrid sampler.
- D1, deductive member of (b) and (a): ARC's own K = 3 'simple' propagation (factorised, numerically identical to the
  dense one) with the third cumulant replaced at every hidden layer by its restriction to k directions,
  kappa_3 <- kappa_3 x Pi x Pi x Pi, Pi = S A (A^T S A)^-1 A^T: "third cumulant only in the future-relevant subspace,
  Gaussian elsewhere". Cost of this implementation: ARC's counter + the projection + the backward work. A
  hypothetical cost, fixed in the PREREG, for a dedicated implementation that never forms the full tensor:
  C_ded = C(k2) + k + 1.5 k^4/n^2 + C_back (C_back = 2n passes for futmf2); it omits the separate K = 2 pilot that
  supplies the mean-field gate probabilities (charged in an extra row of 4.4). The co-prover's sufficient implementation
  bound for contractible diagrams, C_K = O(n + k^{K-1}) passes (astra r2), is used only to define cost scenarios with
  assumed constants (k^2/2 passes per diagram at K = 3), not as a lower bound.
- H1, hybrid (c): control c(h_l) = E[J] (h_l - m) (basis VJPs), the mean-field Jacobian (deductive), a least-squares
  linear fit on a 2^15-sample pilot, or that plus all quadratics in 16 future directions; its mean from the exact
  Gaussian recursion (exact at l = 1) or ARC k2; RQMC (8 scrambles, N = 2^8..2^14, the same points for h_L and every
  residual) on the residual. Registered accounting: error(C) = beta + the residual's RQMC error against a reference
  control mean, at N = (C - B)/a. That sum is not the hybrid's error: it omits the cross term between the residual's
  error and the closure-mean error and counts the reference error (GPT-6 review round 2). So the hybrid estimates are
  also scored directly (post hoc, scratch/run_hyb_direct.py: the same controls, seeds and scrambles, reproduced to
  3e-8 relative, plus N = 2^15 and 2^16): error = |E_Q[c] + mean_RQMC(h_L - c) - Y|^2 / V against the truth Y.
- Comparators: RQMC curves of arc-update (8 scrambles, N = 2^8..2^16 at 256, to 2^14 below; log-log interpolation,
  slope -1 outside the measured range, labelled); ARC k2/k2a/k3 (arc-update and this lane's runs); the multilevel
  shared-trajectory sampler is ~1.0x MC at 256 x 32 [prior: unrestricted, bits-bounds] and is dominated there by RQMC.
- Ledger (operation level; review rounds 1-2, finding 10). B2: per particle and layer, regeneration, push, covariance
  and features (about 6 passes per particle), the VJPs and basis VJPs, and 20 n^3 per layer of dense linear algebra;
  not charged: random-number generation and the relevance Gram matrices (under 1% of the total). D1 as implemented:
  ARC's flop counter, the projection (10 n^3 + 2 n^2 k + 6 n^2 r per layer, r the factor count), the backward work (VJPs,
  or the mean-field recursion at 4 n^3 per layer plus its K = 2 run); not charged: the feature extraction's eigensolves
  beyond the 10 n^3 allowance (~16 n^3 per layer, 2-3% of the implemented cost) and, for the VJP kind, the Gaussian
  sample construction and relevance Gram (~5% of its backward cost). C_ded (hypothetical) leaves out the K = 2 pilot and
  the feature extraction; both are charged in extra rows of 4.4 (scratch/d1_ledger.py). H1, registered ledger: closure
  to layer l, pilot paths, VJPs and basis VJPs, per-sample control evaluation; not charged there: the fits' Gram,
  covariance and pseudo-inverse work and the quadratic control's 2^16-sample cloud; the direct scores of 4.3 are also
  given with an expanded ledger that charges them and with the sum-based evaluation the rerun performs
  (scratch/score_hyb_direct.py).

## 4. Results

<!-- scorecard start -->
Scorecard (results/scorecard.md from the hashed scorers; wording abbreviated, PREREG.md has the exact text):

| id | prediction (abbreviated) | measured | result |
|---|---|---|---|
| P1 | n = 256: median b2(cond_futnl64)/b2(cond_pca64) <= 0.5 at >= 3 of l = 4, 8, 16, 24 | l=4: 0.284 (5 nets); l=8: 0.0897 (3 nets); l=16: 0.552 (4 nets); l=24: 0.909 (3 nets) | FAILS |
| P2 | I_16(256) < I_16(64), I_16 = gauss/futnl16 in the recursion | 64: 5.23; 128: 2.26; 256: 1.36 | HOLDS |
| P3 | gauss/futnl{n/4} >= 2 and gauss/futnl{n/2} >= 10 at every width | 64: 5.23, 75.2; 128: 9.12, 121; 256: 18.8, 204 | HOLDS |
| P4 | median b2 futnl{n/4} < pca{n/4} at every width | 64: 0.000501 vs 0.000795; 128: 0.000204 vs 0.000578; 256: 6.44e-05 vs 0.000308 | HOLDS |
| P5 | particle recursion futnl{n/2} at its honest cost: G_RQMC < 0.5 on every net | 64: max 0.00724; 128: max 0.0273; 256: max 0.0289 | HOLDS |
| P6 | (a) R(futv, n/4) >= 0.7; (b) R(pca, n/8) < R(futv, n/8), every width | 64: R(futv,n/4) 0.984, R(futv,n/8) 0.787 vs R(pca,n/8) 0.651; 128: R(futv,n/4) 0.949, R(futv,n/8) 0.763 vs R(pca,n/8) 0.676; 256: R(futv,n/4) 0.977, R(futv,n/8) 0.867 vs R(pca,n/8) 0.782 | HOLDS |
| P7 | n = 256: every D1 variant at its implemented cost has G_RQMC < 0.2 (5/5 nets) | hashed scorer: fails (NaN runs included); corrected (NaN runs excluded, deviations item 3): max G 0.181 over 57 runs, 8 failed runs | HOLDS |
| P8 | n = 256, dedicated best-case cost: median G_RQMC of k3p(futmf2, 64) and k3p(futv, 64) <= 3 | hashed scorer: futmf2: nan; futv: 0.0236 (FAILS: it looked up "futmf264", the runs are "futmf2_64"; deviations item 7); post hoc with the actual names: futmf2: 2.37; futv: 0.0236 -> holds | FAILS |
| P9 | (a) l = 1 best eta >= 0.25; (b) l >= 8 every control beta_eg >= 1e-4; (c) best equal-cost gain at 6.5e4 passes <= 2 (n = 256) | (a) median eta 0.546: True; (b) min median beta 0.000277: True; (c) median gain 0.83: True | HOLDS |
| P10 (Amendment 1) | median k3f nMSE at 512 x 64 in [1e-4, 4e-4] | k3_512 0.000348 | HOLDS |
| P11 (Amendment 1) | q_proj(futmf2, n/4)/q_3 >= 0.6 at 512 and q_proj(futmf2, 16) at 512 < at 256 | qp_over_q3_512 nan; qp16_512 2.06; qp16_256 2.34 | FAILS |
| P12 (Amendment 1) | best-case G_RQMC of k3p(futmf2, n/4) at 512 <= at 256 | Gded_512 nan; Gded_256 2.37 | FAILS |
| P13 (Amendment 1) | ARC k2 G_RQMC at 512 < at 256 | k2G_512 0.836; k2G_256 1.4 | HOLDS |
<!-- scorecard end -->

Reading the scorecard (final run 14:38Z; original, amended and corrected outcomes kept apart, review 15). As scored by
the hashed scorer, six of nine predictions hold (P2-P6, P9). P7 as originally worded fails because eight k = n/2 runs
returned NaN (ARC's K = 3 state covariance becomes indefinite and the whitening divides by ~0; 4.2); under the NaN rule
written after those failures were seen but before any scoring (deviations item 3) it holds (57 finite runs, maximum
gain 0.18); with the eight runs completed by a post hoc rank guard (a changed algorithm), all 65 runs have gain below 0.2 (the rescued runs'
maximum is 0.12), so the original wording also holds post hoc. P8 fails as scored
because the hashed scorer looked up a misspelled variant name and found no data for half of it; with the runs' actual
names (post hoc, deviations item 7) the intended prediction holds (median hypothetical-cost gain 2.37 <= 3). P1 fails:
the future-vs-PCA ratio at k = n/4 holds at l = 4 and 8 and not at l = 16 and 24, where both values are within a few
noise units of zero. The hashed scorer also floors recursion b2 at max(b2, noise, 1e-7), a rule not in the PREREG wording;
applying the wording literally (raw b2, the truth-floor exclusion) changes P1's l = 16 median to 0.75 and P3's n/2
medians to 139 and 236, and no outcome (deviations item 8). P9(c) with the cheapest setup per control kind, as worded:
0.832 against the scorer's 0.830, same outcome. P9(c) was scored with a sum that omits a cross term (review round 2);
scored directly (post hoc, 4.3) the best gain at 6.5e4 passes is 0.98-1.00 against RQMC on the same scrambles and
1.00-1.02 against arc-update's curve (registered to sum-based ledger), so P9 holds either way.

### 4.1 What the backward pass finds (B1, B2; registered P1-P4)
**Full recursions (B2; the benchmark that includes accumulation).** Median over nets of the raw squared error b2 of the
finite-particle recursion (2^18 particles; its target is the bias of that finite algorithm against the stored truth,
not of an infinite-particle closure [review 8]), and in brackets the median over nets of the per-net improvement over
the Gaussian closure (cells with b2 below 10x the truth floor excluded, as the PREREG says; the hashed scorer instead
floored b2 at max(b2, replicate noise, 1e-7), a rule not in the PREREG wording, which changes no outcome; deviations
item 8):
| n x L (nets) | Gaussian closure | radius-corrected | future k = 16 | future k = n/4 | future k = n/2 | PCA k = n/4 | random k = n/4 |
|---|---|---|---|---|---|---|---|
| 64 x 8 (8) | 2.4e-3 | 2.0e-3 (1.4x) | 5.0e-4 (5.2x) | 5.0e-4 (5.2x) | 4.5e-5 (75x) | 8.0e-4 (3.1x) | 1.8e-3 (1.4x) |
| 128 x 16 (8) | 1.8e-3 | 9.2e-4 (2.3x) | 6.9e-4 (2.3x) | 2.0e-4 (9.1x) | 1.3e-5 (139x) | 5.8e-4 (3.3x) | 9.9e-4 (1.6x) |
| 256 x 32 (5) | 1.55e-3 | 4.9e-4 (2.8x) | 1.17e-3 (1.4x) | 6.4e-5 (19x) | 4.6e-6 (236x; 4 of 5 nets) | 3.1e-4 (4.3x) | - |
- At the same k = n/4 the future directions beat the cloud's principal axes by a median per-net factor of 2.0, 2.7, 2.9
  (P4 holds; registered as a comparison of medians) and random directions by more. At fixed k = 16 the improvement
  shrinks with width (5.2, 2.3, 1.4; P2 holds), while at k = n/4 and n/2 it grows (P3 holds): over n = 64-256, what a
  fixed number of directions buys shrank and what a fixed fraction buys grew (three widths; descriptive, not a limit
  statement). The benchmark's n e at k = n/4 is 0.032,
  0.026, 0.016 and at k = n/2 2.9e-3, 1.7e-3, 1.2e-3 (raw medians; 1.7e-3 at 256 with the scorer's floor); with two
  replicates per net and no particle-convergence study at 256, these are descriptive observations, not a rate.
- Spread over nets (min-max of the per-net literal ratios; with 5-8 nets a crude uncertainty statement, not a standard
  error; review 8): improvement at k = 16: 3.5-12.4, 1.9-3.8, 1.15-1.7 (the ranges at n = 64 and 256 do not overlap);
  at k = n/4: 3.5-12.4, 5.2-14, 9.3-37; PCA/future at k = n/4: 1.2-2.8, 1.9-4.4, 2.5-6.6 (the future directions are
  better on every net at every width).
- A finite-particle check at n = 64 (2^20 against 2^18 particles, dev net): 1.50e-3 vs 1.41e-3 (Gaussian), 3.2e-4 vs
  2.9e-4 (k = 16), 3.3e-5 vs 2.9e-5 (k = 32): within ~15%; no such check was made at 256 (review 8).
- The particle recursion is an estimator only at its full cost; there it loses to RQMC by 45-250x and to plain MC as well
  (P5; 4.2).

**Single cuts (B1) and the future's reading.** Paired comparisons at the same cut and k (scratch/paired_cut.py, post
hoc; common random numbers): at n = 64 the future directions beat PCA at k = 16 on 29 of 32 net-cut cells; at 256 x 32,
at k = 64 they beat PCA on 5 of 5 nets at l = 1, 4, 8, 16 and 4 of 5 at l = 24 (median b2 7.4e-6, 9.4e-6, 1.0e-6,
2.1e-6, 3.9e-6 against 4.8e-5, 1.9e-5, 1.1e-5, 9.9e-6, 5.5e-6; the noise scale is 1-4e-6, so the future values at
l >= 8 are at the floor), while at k = 16 PCA is better at l = 1-4 and the future at l >= 8. The registered P1 (median
ratio future/PCA at k = n/4 <= 0.5 at >= 3 of the cuts l = 4, 8, 16, 24, cells with PCA b2 < 5e-6 excluded) FAILS as
scored: it holds at l = 4 (0.28) and 8 (0.09) and not at 16 (0.55) or 24 (0.91); with the PREREG's truth-floor rule
applied literally (review 5) the l = 16 median becomes 0.75 (3 nets), same outcome. Both values sit within a few noise
units of zero there, so the ratio test is noise-dominated; the paired differences favour the future directions on 4-5
of 5 nets.
The improvement any k-dim choice gives follows the share s of the future's whitened nonlinear relevance (E[J^T J] minus
E[J]^T E[J], whitened by the cloud covariance) that the k directions capture: log improvement = 1.19 (-log(1 - s)) - 0.10,
correlation 0.96 over 379 cells at n = 64 (futnl, fut, pca and random features alike; post hoc; cells share nets and
baselines); 0.90 and slope 0.98 over 580 cells at n = 128; at 256 the relation is weaker (0.77, slope 0.75 over 241
cells), the large improvements being capped by the noise floor. The participation ratio of that relevance grew
about linearly with n at a fixed early layer over these widths (17.5, 33.3, 67 at l = 1 for n = 64, 128, 256; at 256
computed from the top 128 eigenvalues, which changes it by at most 1.2%) and slowly at fixed l/L (17.5,
24.1, 31.8 at l/L = 1/8; 13.2, 16.4, 19.3 at 1/4; 9.2, 11.1, 11.8 at 1/2), so the share 16 directions capture falls
(0.80, 0.59, 0.39 at l = 1; 0.80, 0.69, 0.61 at l/L = 1/8). At l = 1 the spectrum scales with the width: the top n/4
directions capture 0.80, 0.80, 0.81 of the trace at every width while the top 16 capture less and less. Three n = 512
nets (exploratory, scratch/relevance_spectrum.py, validated against the registered n = 64 values) continue all of
these: participation ratio 127.5 (126-135) at l = 1 (36, 22, 12 at l/L = 1/8, 1/4, 1/2), top-16 share 0.25 and top-n/4
share 0.82 at l = 1. The closures' errors are made early [prior: arc-update P6], where this rank grows fastest. Paired single cuts at n = 128: the future directions beat PCA on 6-8 of 8 nets at every
cut, at k = 16 and at k = n/4.

### 4.2 Carrying the law in the future directions (B2 as an estimator, P5; D1, P6-P8)
**By particles (B2 scored as an estimator, P5).** One replicate of the recursion at k = n/2 costs 1.7e6, 1.9e6, 2.6e6
passes (2^18 particles, ~6 passes each, plus 2^12 VJPs and the basis VJPs for E[J] at every layer) and reaches a
per-replicate nMSE of 5.0e-5, 1.7e-5, 7.9e-6. Plain MC at that cost would reach 1/C = 6.0e-7, 5.2e-7, 3.8e-7, so the particle
closure is 83x, 33x, 21x worse than plain MC; RQMC, extrapolated with slope -1 about 40-120x beyond its largest measured point,
would be 45-250x better (G_RQMC 0.004, 0.012, 0.022; P5 holds on every net). Carrying the kept law by particles is
sampling with extra work per sample (its replicate variance per particle is 0.3-1x MC's, at about six times MC's cost
per particle).

**Deductively, at low order (D1, P6-P8).** ARC's k3 with its third cumulant kept only in k directions. The retained
factor is e_k3 / e_projected per net (1 = as accurate as full k3; >1 = better), median over nets, with successful runs
over attempted ones (a failed run is a rank-deficient whitening of ARC's state covariance at large k; deviations items
3 and review 6); R = (e_k2 - e_proj)/(e_k2 - e_k3), the registered statistic, retains the error DIFFERENCE and is not
the same quantity [A]:
| n | ARC k2 / k3 nMSE (median) | e_k2/e_k3 (median) | future (VJP) k = n/8, n/4, n/2 | mean-field future n/8, n/4, n/2 | PCA n/8, n/4, n/2 | fixed k = 16: future / PCA |
|---|---|---|---|---|---|---|
| 64 | 2.4e-3 / 5.2e-4 | 4.7 | 0.46, 0.86, 1.18 (8/8 each) | 0.47, 0.78, 1.13 | 0.40, 0.58, 1.02 | 0.86 / 0.58 |
| 128 | 1.6e-3 / 2.65e-4 | 6.7 | 0.46, 0.82, 1.06 (6/8 at n/2) | 0.45, 0.86, 1.04 (6/8 at n/2) | 0.39, 0.68, 0.96 | 0.46 / 0.39 |
| 256 | 1.2e-3 / 2.0e-4 | 11.9 | 0.50, 0.80, 1.23 (1/5 at n/2) | 0.58, 0.91, 1.19 (1/5 at n/2) | 0.36, 0.67, 1.12 | 0.28 / 0.18 |
- On n/4 future directions the projected k3 keeps a median 80-86% of k3's accuracy (78-91% with the mean-field
  directions; P6a holds: R = 0.98, 0.95, 0.98), with wide per-net variation (VJP directions: 0.38-2.0, 0.26-1.38,
  0.61-0.97 at n = 64, 128, 256), and the future directions beat PCA at k = n/8 and n/4 at every width (P6b holds at
  n/8: R 0.79 vs 0.65, 0.76 vs 0.68, 0.87 vs 0.78). On n/2 directions the successful runs' median retained factor is
  above 1 (VJP 1.18, 1.06, 1.23; mean-field 1.13, 1.04, 1.19; the 256 values are single nets), but per net the
  projection beat full k3 on only 7 of 8, 3 of 6 and 1 of 1 successful runs (VJP) and 6 of 8, 5 of 6, 1 of 1
  (mean-field); the worst is 0.56 (VJP, n = 128, seed 302: 1.8x worse than k3). PCA on n/2 directions: medians 1.02,
  0.96, 1.12, better than k3 on 4 of 8, 3 of 8 and 3 of 5 nets (range 0.40-1.68). So "better than k3" is a statement
  about medians, not about every net; that it happens at all suggests the projection can remove error components of
  ARC's propagated kappa_3 on some nets.
- Why runs failed, and what they give when they run (post hoc; lib/fa_kprop_guard.py, scratch/score_guard.py): ARC's
  K = 3 'simple' state covariance becomes indefinite after a few layers (n = 128, seed 304: 5 non-positive eigenvalues
  of 128 at layer 4, 36 at layer 13), and whitening a wide feature block then divides by ~0. A rank guard that keeps
  only the S-positive part of the kept span reproduces the six registered successes it was checked on (relative
  difference <= 5e-14) and makes all 12 failed runs finite. At n = 128 the rescues give 0.60, 1.21 (VJP) and 0.89, 1.09
  (mean-field); over all 8 nets the medians are 1.06 (4 of 8 beat k3) and 1.04 (6 of 8). At n = 256 they give 0.40,
  1.07, 1.05 and one divergence (nMSE 45) for VJP, and 0.92, 1.07, 0.02 (nMSE 5e-3) and one divergence (nMSE 38) for
  mean-field. At half width the projected k3 is numerically fragile on ARC's indefinite states.
- The deductive mean-field backward sweep (~2n passes) finds directions about as good as 2^11 VJPs per layer (which
  cost ~1e5 passes at 256 x 32).
- At fixed k = 16 the retained factor decays with width (0.86, 0.46, 0.28): the dimension the third cumulant needs
  grows with n.
- The errors are k3-level and fall slowly with width at fixed L/n (k3: 5.2e-4, 2.65e-4, 2.0e-4; future n/4: 5.3e-4,
  3.1e-4, 2.7e-4), so n e(n) does not fall (0.034, 0.040, 0.068): the o(1/n) of 2.5 is not seen in these data. As
  implemented (full k3f plus projection) every variant loses to RQMC by 6-200x at its cost at n = 64-256 (medians over nets per variant; single runs 2.3-340x; P7 under the amendment of
  deviations item 3; as originally worded P7 fails on the NaN runs). Under the hypothetical dedicated cost, see 4.4 (P8).

### 4.3 The hybrid (H1, P9)
Scored directly (post hoc; review round 2): each hybrid estimate E_Q[c] + mean_RQMC(h_L - c), with the closure mean
from the exact Gaussian recursion, against the truth (scratch/run_hyb_direct.py, scratch/score_hyb_direct.py,
results/hyb_direct.md). The rerun rebuilds the registered controls with the same seeds and the same scrambles
(eta, beta and the registered residual values reproduced to 3e-8 relative) and adds N = 2^15, 2^16, so no hybrid curve
is extrapolated (equal-cost values are log-log interpolations between measured N). The registered numbers (a sum that
omits a cross term) stay in the scorecard and in results/tables_part1.md T4. Three cost ledgers (review rounds 2-3):
the registered one (historical); an expanded one that charges the closure actually used for the mean (once), the
fits' linear algebra and the quadratic control's 2^16-sample cloud; and the sum-based one, which is the expanded ledger
with the evaluation the rerun actually performs (the controls are linear in h_l and the quadratic features, so they are
applied once to the sums instead of to every sample). Medians over nets (8, 8, 5 nets at n = 64, 128, 256); "same
scrambles" = plain RQMC on the same points in the same run:
| n | eta at l = 1 (exact means): E[J] / mean-field / least-squares / +16 quadratics | hybrid / plain RQMC error at 2^16 points, same scrambles (l = 1, same order) | smallest closure bias beta at l >= L/4 | best gain over RQMC at 1e4 / 6.5e4 passes, same scrambles: registered / expanded / sum-based ledger | sum-based, against arc-update's curve (the registered comparator) |
|---|---|---|---|---|---|
| 64 | 0.52 / 0.53 / 0.47 / 0.32 | 0.95 / 0.96 / 0.95 / 0.81 | 6.8e-5 | 0.85 / 0.85 / 0.95 and 0.96 / 0.96 / 1.03 | 0.90 / 1.10 (extrap.) |
| 128 | 0.62 / 0.61 / 0.55 / 0.45 | 0.98 / 0.98 / 0.98 / 0.88 | 8.9e-5 | 0.87 / 0.86 / 0.91 and 0.98 / 0.98 / 1.02 | 0.98 / 0.75 (extrap.) |
| 256 | 0.76 / 0.70 / 0.61 / 0.55 | 0.99 / 0.98 / 0.98 / 0.97 | 2.6e-4 | 0.85 / 0.83 / 0.86 and 0.98 / 0.97 / 1.00 | 0.92 / 1.02 |
- Controls whose mean is exactly known (functions of h_1, whose law is explicit) remove 25-70% of the plain-MC variance,
  less as the width grows over 64-256 (the input map's mean Hermite degree is n / E V [prior: input-bits]); but the
  part they remove is the part scrambled Sobol already integrates: on the same scrambles the hybrid's error at 2^16
  points is 0.81-0.99 of plain RQMC's (the quadratic control at n = 64 is the best case), i.e. the residual's gain over
  MC is G_R = eta G / rho ~ eta G, not G (E13.9). Owen's gain-coefficient identity for nested uniformly scrambled nets
  (astra r2 Q4) explains this: gains multiply only if the residual keeps the integrand's average scrambling gain; here
  it loses about a factor 1/eta (the implemented scramble is scipy's LMS plus digital shift, for which the identity is
  used as an explanation only).
- Controls built deeper (l >= L/4) explain 75-95% of the variance, but their mean must come from a closure, and the
  closure's layer-l bias (beta >= 7e-5 to 3e-4) already exceeds RQMC's error at 2^14 points (2-3e-5).
- At equal total cost the best fixed control is the mean-field linear control at l = 1 (deductive setup ~1.5e3 passes
  at 256 x 32); per net the winner is occasionally another control (e.g. the same control at l = 2). With the
  sum-based ledger the best control per net gains 0.86-0.95x at 1e4 passes and 1.00-1.03x at 6.5e4 passes against RQMC
  on the same scrambles (medians over nets; per net 0.95-1.15 at 6.5e4); the fixed l = 1 mean-field control alone gives
  0.85-0.95x and 1.00-1.03x. The registered and expanded ledgers, which charge a matrix-vector product per sample, give
  0.83-0.87x and 0.96-0.98x. Against arc-update's curves (other scrambles) the sum-based medians are 0.90-0.98x and
  0.75-1.10x; two independent 8-scramble estimates of plain RQMC differ by factors 0.47-1.98 per point (4.4), so the
  same-scramble comparison is the more reliable one. So the hybrid is near unity at 6.5e4 passes and loses about 5-15%
  at 1e4 passes; its statistical uncertainty was not quantified. The sum-based ledger is an approximate cost of one
  selected control: it charges the per-sample accumulation and one final control application, not the separate
  computation of the control's closure mean (another ~2 n^2 flops unless fused), the full outer products the quadratic
  features are built from, the diagnostics, or the construction of the competing controls.
- What the registered sum missed: the omitted cross term changes the error of individual (control, N, net) cells by
  -22% to +13% (5-95% range; median 0). P9(c) (gain <= 2 at 6.5e4 passes) holds under every scoring and ledger:
  registered 0.83; direct 0.98 (registered ledger) to 1.00 (sum-based) against the same scrambles, 1.00-1.02 against
  arc-update's curve at 256 x 32. The exploratory run_cvk.py (section 5), which uses the analytic arc-cosine moments,
  reaches the same conclusion.

### 4.4 Windows per width: "beats RQMC by X up to precision Y"
X = median over nets of a method's gain over RQMC at its own cost (RQMC's error at that cost over the method's error);
Y = its median nMSE; the window ends where RQMC reaches Y. RQMC errors outside the measured range (256 points to 2^14
at n = 64, 128; to 2^16 at 256) are extrapolated with slope -1 at either end ("extrap."); the multilevel
shared-trajectory sampler is about MC at 256 x 32 [prior] and is not a stronger comparator.
| method (cost charged) | n = 64 (L = 8) | n = 128 (L = 16) | n = 256 (L = 32) |
|---|---|---|---|
| ARC k2a, reference [prior: arc-update T1] | 1.9x up to 1.3e-3 at 164 passes (extrap.) | 2.0x up to 1.0e-3 at 260 | 2.0x up to 8.4e-4 at 452 |
| ARC k2 'simple' (this lane's runs) | 1.6x up to 2.4e-3 at 106 (extrap.) | 2.0x up to 1.6e-3 at 201 (extrap.) | 1.4x up to 1.2e-3 at 393 |
| D1, mean-field future k = n/4, as implemented | 0.12x at 5.0e3 | 0.056x at 1.9e4 (extrap.) | 0.036x at 7.5e4 (extrap.) |
| D1, same, registered hypothetical dedicated cost C_ded | 2.6x up to 5.9e-4 at 274 (to 636) | 3.0x up to 3.1e-4 at 585 (to 1556) | 2.4x up to 2.2e-4 at 1353 (to 2790) |
| D1, same, C_ded plus the separate K = 2 pilot it needs | 1.8x | 2.2x | 1.75x |
| D1, same, C_ded + pilot + per-layer extraction of the k directions: assumed ~2n-pass extraction (feasibility not established) / dense, as implemented (approximate count; post hoc) | 1.3x / 0.49x (nets 0.77-2.3 / 0.31-0.98) | 1.5x / 0.67x (0.73-2.0 / 0.38-0.95) | 1.3x / 0.53x (1.2-2.3 / 0.50-0.75) |
| D1, same, cost scenario C_k2 + D k^2/2 + 2n with D = 1 / 3 (pilot and extraction not charged) | 1.9x / 1.1x | 1.6x / 0.78x | 0.93x / 0.36x |
| D1, PCA k = n/8, the C_ded formula (C_back = 0; its eigensolves not charged; descriptive) | 3.1x up to 1.2e-3 at 115 (extrap.) | 5.2x up to 6.5e-4 at 223 (extrap.) | 3.5x up to 5.5e-4 at 449 |
| B2, future k = n/2, particles (flop count) | 0.004x at 1.7e6 (extrap.) | 0.012x at 1.9e6 (extrap.) | 0.022x at 2.6e6 (extrap.) |
| H1, best control, scored directly, sum-based ledger, at 1e4 / 6.5e4 passes, RQMC on the same scrambles (post hoc; registered sum and ledger: 0.78 / 0.98, 0.93 / 0.93, 0.91 / 0.83) | 0.95x / 1.03x | 0.91x / 1.02x | 0.86x / 1.00x |
- The deductive member gives a constant factor only under a cost model no implementation has achieved (C_ded, a
  registered hypothesis); the factor does not grow with n (2.6, 3.0, 2.4; 1.8, 2.2, 1.75 with the K = 2 pilot charged).
  C_ded also leaves out finding the k directions at every layer (the top-k eigenvectors of the whitened relevance): the
  dense eigensolvers used here cost ~13n passes over the net (about 26 n^3 flops per layer, an approximate count) and
  close the window (0.5-0.7x). An assumed extraction at ~2n passes (the matrix-product arithmetic of a randomised range
  finder with Cholesky whitening) would leave 1.3-1.5x, but it is not a specified feasible method: ARC's K = 3 states
  are indefinite, so plain Cholesky fails at most layers even at k = n/4 (layers 5-15 of a quarter-width n = 128 run,
  checked in review round 3), and the extra work of a fix and the accuracy of approximate directions are not established
  (scratch/d1_ledger.py; reusing directions across layers is not excluded either).
  The "cost scenario" rows charge k^2/2 passes per diagram, the K = 3 form of the co-prover's sufficient implementation
  bound O(n + k^{K-1}) for contractible diagrams (astra r2) with assumed constants D = 1 or 3; they are scenarios, not
  lower bounds, and a cheaper contraction is not excluded (review 9).
- The precision improves with width (k3's error falls slowly at fixed L/n: 5.2e-4, 2.65e-4, 2.0e-4), but the cost
  grows with n at least as fast (the K = 2 base alone is ~1.5n passes), so the multiplicative window does not grow over
  these widths (2.5's criterion; descriptive over three widths).
- At ARC's contest budget (6.5e4 passes at 256 x 32) every method in this table except the H1 hybrid loses to RQMC,
  which is at ~1e-5 there; the hybrid is near unity (1.00 against RQMC on the same scrambles, 1.02 against
  arc-update's curve).
- Sensitivity of the extrapolated cells (scratch/extrap_sensitivity.py, post hoc; review 13): replacing slope -1 by each
  net's local slope at the measured end, or by a slope chosen against the stated conclusion (-0.5 beyond the top end,
  RQMC flattening; -0.8 below the bottom end), B2's gain stays 0.004-0.14 (a loss by 7x or more in every case; it also
  loses to plain MC, whose 1/C needs no extrapolation), D1 as implemented stays 0.035-0.061 at n = 128, 256 (0.12 at n = 64, which needs no extrapolation), ARC k2 at n = 64 moves
  within 1.3-2.2x and the PCA n/8 row at n = 64 within 2.6-3.6x (n = 128: 1.9-2.1x and 5.0-5.3x). No conclusion
  depends on the extrapolation slope.
- Comparator noise (scratch/comparator_noise.py, post hoc): arc-update's RQMC curves average 8 scrambles per N; the H1
  rerun measured plain RQMC on 8 other scrambles of the same nets, and the two estimates differ by factors 0.47-1.98
  per point (median 0.97-1.07 per width), so a single net's gain can be off by up to a factor of about 2 from the comparator alone (a range, not a
  confidence interval). Against the pooled
  16-scramble curve the medians become: ARC k2 1.6, 1.8, 1.2x (from 1.6, 2.0, 1.4); D1 at C_ded 2.9, 2.9, 2.1x (from
  2.6, 3.0, 2.4); D1 as implemented essentially unchanged. No conclusion changes.

### 4.5 A fourth width: n = 512, L = 64 (Amendment 1, P10-P13; nets 500-502)
Registered at 12:17:16Z before any run at that width (PREREG.md Amendment 1; frozen/amend1/), run by scratch/run_w512.py
(truths 2^22 paths, RQMC 8 scrambles to 2^14) and scored by the hashed scratch/score_amend1.py (scorecard above).
Medians over the three nets (per-net values in results/amend1_posthoc.json, scratch/amend1_posthoc.py, post hoc):
| quantity | n = 64 | n = 128 | n = 256 | n = 512 |
|---|---|---|---|---|
| ARC k2 'simple': nMSE; gain over RQMC at its cost | 2.4e-3; 1.6x | 1.6e-3; 2.0x | 1.2e-3; 1.4x | 1.0e-3; 0.84x (1.09, 0.84, 0.63) |
| ARC k3 nMSE; per-order ratio q_3 = e_k2/e_k3 | 5.2e-4; 4.7 | 2.65e-4; 6.7 | 2.0e-4; 11.9 | 3.5e-4; 2.95 (2.9-3.3) |
| D1 mean-field, retained factor e_k3/e at k = n/8 / n/4 | 0.47 / 0.78 | 0.45 / 0.86 | 0.58 / 0.91 | 1.22 / failed (below) |
| D1 mean-field k = n/8, gain at C_ded / + K = 2 pilot / + dense extraction (post hoc) | - | - | 1.95 / 1.31 / 0.37 | 1.05 / 0.74 / 0.21 |
- P10 holds (k3 3.5e-4 at 512, inside [1e-4, 4e-4]) and P13 holds: ARC's K = 2 now loses to RQMC at its own cost
  (0.84x, against 1.40x at 256). P11 and P12 FAIL as scored because every registered k = n/4 = 128 run at 512 returned
  NaN (the same indefinite-state failure as the half-width runs, 4.2), so their k = n/4 statistics have no data; P11's
  other half holds (fixed k = 16 retains less at 512: q_proj 2.06 against 2.34). With the rank guard (post hoc, a
  changed algorithm) the k = n/4 runs give a retained factor 1.15 on net 501 and diverge on nets 500 and 502 (nMSE 1.6
  and 15.8): at 512 the projected k3 on n/4 directions is not a usable method as implemented.
- Post hoc substitute for P12 at k = n/8, where all runs succeed: the hypothetical dedicated-cost gain falls from 1.95 at
  256 to 1.05 at 512 (0.74 with the K = 2 pilot, 0.21 with the dense extraction used here); the window does not grow.
  As implemented every D1 variant at 512 loses to RQMC by 150-400x at its cost (medians over nets of the variants with
  finite runs, failed runs excluded; single runs 124-406x) (RQMC extrapolated 18x beyond 2^14).
- ARC's own ladder weakens at 512 x 64: k3 improves on k2 only 2.95x (median; 4.7, 6.7, 11.9 at 64-256), so the ladder
  criterion of 2.5 (retained dimension per order below q) would need k* <= 3 there. The k = n/8 projection beats full
  k3 on all three nets (1.30, 1.02, 1.22), i.e. it removes error that ARC's k3 propagates at this depth; it is still a
  constant factor at k3's accuracy level.
- The relevance spectrum at 512 (three nets, exploratory; 4.1): participation ratio 127.5 at l = 1, top-16 share 0.25,
  top-n/4 share 0.82: the first-layer relevance keeps scaling with n.

## 5. Exploratory (not registered): what must be kept, and does truncation carry the ladder to order 4?
**The joint law, not k one-dimensional laws.** If the future read the kept span through one-dimensional (ridge) modes,
the k future directions' marginal laws would suffice, and k 1-D laws are cheap to carry deterministically. Test
(scratch/run_ridge.py; eight n = 64 evaluation nets; single cut, 2^18 pushes): keep each future direction's 1-D marginal
exact but make the directions independent (each whitened feature coordinate permuted independently across the batch),
with and without arc-update's radial transport. Medians at l = 1, 2, 4, 6:
| projection | l = 1 | l = 2 | l = 4 | l = 6 |
|---|---|---|---|---|
| Gaussian | 4.1e-4 | 7.3e-4 | 6.9e-4 | 5.2e-4 |
| radius transport | 1.0e-4 | 2.6e-4 | 4.2e-4 | 4.5e-4 |
| joint law exact, k = 16 / 32 | 7.7e-5 / 1.5e-5 | 9.2e-5 / 8.0e-6 | 5.5e-5 / 2.5e-6 | 2.9e-5 / 1.6e-6 |
| independent exact marginals, k = 16 / 32 | 3.2e-4 / 3.3e-4 | 6.0e-4 / 6.1e-4 | 5.3e-4 / 5.4e-4 | 3.8e-4 / 3.8e-4 |
| the same + radius transport, k = 16 / 32 | 1.2e-4 / 1.1e-4 | 2.2e-4 / 2.2e-4 | 3.5e-4 / 3.4e-4 | 3.0e-4 / 3.0e-4 |
The joint law beats the independent marginals in 32 of 32 net-cut cells at both k; the marginals keep a measurable
part (median 0.72 of the Gaussian's error at k = 16) and adding k from 16 to 32 does not help them. So, in the tested
coordinates, most of what the future reads in the kept span is dependence between the kept directions, and the
coordinate-additive ("ridge") reading I first gave (FORMAL E10) is withdrawn (E11, narrowed in E13.2 [A]). This does
not prove that the whole k-dim law is needed: an oblique ridge or a few mixed moments might suffice. (The rows with
radius transport rescale whole samples and change the marginals, so they do not isolate "beyond the radius" [A].)

**Is the joint law cheap in another form? A Gaussian mixture in the k directions** (scratch/run_mix.py, single cut;
numbers corrected after a sampling bug, deviations item 16). Development net 9500, n = 64, k = 16: fitting an
M-component full-covariance Gaussian mixture to the 16 future features of 2^15 true samples (complement as before)
gives b2 = 6.2e-4, 2.4e-4, 3.0e-4, 2.4e-4, 1.2e-4, 1.2e-4 at l = 1 for M = 1, 2, 4, 8, 16, 32 (Gaussian 5.0e-4, exact
joint law 6.2e-5), and 7.8e-5 / 6.1e-5 at l = 4 for M = 16 / 32 (exact 3.2e-5): about k components recover 85-95% of
the joint law's gain at this width. Across widths (l = 1, k = n/4, one net per width: 300, 300, 200; mixtures fitted by
EM to 2^16 samples, M = 1, 4, 16, 64), the best fraction of the exact joint law's gain recovered is 0.78, 0.69, 0.54
at n = 64, 128, 256, and more than 4-16 components did not help at 128-256 (n = 256: Gaussian 2.2e-4, exact 1.5e-5;
M = 4, 16, 64: 1.1e-4, 1.5e-4, 1.7e-4). EM's fit noise in 32-64 dimensions (visible at n = 64, 128 as the one-component
fit's excess over the exact Gaussian) confounds the larger M, so this bounds what FITTED mixtures do here, not what a
propagated mixture could do; but it gives no sign that a fixed number of components suffices as n grows. A
deterministic closure that propagates M Gaussian components pays at least M Gaussian-closure steps per layer (~1.5 n
passes each), i.e. ~k n ~ n^2/4 passes at k = n/4 IF M must grow like k, against benchmark errors (B2, k = n/4) that
fall like n^-1.3 to n^-1.7 between n = 64 and 256: then C e would still grow with n. How M must scale is not
established, so a growing window for this route is not excluded, only not seen (review 12).

**Priced as estimators: the mixture closure and exact-mean controls of growing complexity** (asked by main; exploratory).
- Mixture closure, full recursion (scratch/run_mixrec.py; n = 64, nets 300-303, k = 16 future directions, the kept law
  refitted as an M-component Gaussian mixture at every layer, particles otherwise): b2 per net for M = 8 / 16 / 32
  against exact particles: 300: 1.3e-3 / 4.9e-4 / 2.8e-4 vs 3.3e-4; 301: 1.7e-3 / 1.6e-3 / 2.0e-3 vs 6.7e-4; 302: 6.0e-4 /
  4.5e-4 / 4.7e-4 vs 1.4e-4; 303: 2.1e-3 / - / - vs 5.7e-4 (Gaussian closure 1.8-3.9e-3). In the recursion the mixture
  recovers part of the gain on some nets and little on others. These are EM-fitted particle recursions, not a specific
  deterministic mixture algorithm; a deterministic M-component closure would cost at least M K = 2 steps (106 passes
  each at n = 64) plus the backward sweep: ~1.8e3 passes at M = 16, ~3.5e3 at M = 32, where RQMC's error is ~1.7e-4 and
  ~1.0e-4 (net 300), 3-20x below these recursions' 2.8e-4-2e-3. On this evidence it would lose to RQMC at equal cost at
  n = 64; scaling with width is not measured.
- Exact-mean controls of growing complexity (Kelvin's exponent question via main: if Var(f - g) fell like
  (construction cost)^-delta with g's mean exact, a fixed budget fraction spent on g would give MSE ~ C^-(1 + delta)).
  Control at l = 1, whose law is explicit: linear in h_1 plus all quadratics in the top-k future directions, mean exact
  (arc-cosine moments), fitted by least squares (scratch/run_cvk.py; net 300). Residual share eta(k): n = 64: 0.38
  (linear only), 0.32, 0.29, 0.23, 0.19 at k = 4, 8, 16, 32 (eta ~ k^-0.25); n = 128: 0.59, 0.52, 0.49, 0.46, 0.43,
  0.41 at k = 4, ..., 64 (eta ~ k^-0.09). Against the saved construction costs directly (pilot, VJPs and fit), the
  descriptive slopes are -0.34 (n = 64) and -0.10 (n = 128) [review 12]. The residual gains little on top of scrambled Sobol (RQMC error of the residual / of h_L at 2^14
  points: 0.71-0.85 at n = 64, 0.91-0.98 at n = 128), and at n = 256 (registered H1, k = 16) eta is 0.55 with the
  residual's RQMC error equal to h_L's. So a delta > 0 holding uniformly in width is not established by these runs (one
  net per width, two exploratory widths); in this family it looks small and shrinking with width.

**Order 4 under truncation.** ARC's dense K = 4 'simple' with its third- and fourth-cumulant cores projected each hidden layer on k future directions
(lib/fa_kprop4.py; runs/k4p_*_x*.json). n = 32 (L = 4), five nets (9500, 300-303): full k4 2.6e-5, 3.3e-5, 3.5e-5,
2.5e-4, 1.8e-5 against k3 9.5e-4, 4.9e-4, 3.9e-4, 5.1e-4, 3.3e-4; both cores on k = n/4 = 8 directions: 5.6e-4, 7.7e-4,
4.7e-4, 7.2e-4, 6.1e-4 (1.7x better than k3 on net 9500, worse than k3 on the other four); on k = n/2 = 16: 1.0e-4,
8.6e-5, 1.5e-4, 5.2e-4, 1.4e-4 (4.0, 2.6, 4.3, 2.1, 7.8x worse than k4).
Both retention metrics (astra r3): on net 9500, k = 8 keeps 0.43 of the absolute improvement e_3 - e_4 but a factor
e_3/e = 1.7 of the full 36.8; k = 16 keeps 0.92 of the improvement but a factor 9.2. Separating the orders (nets 9500,
300, 303): the FOURTH cumulant alone on 16 directions keeps k4 (2.6e-5, 3.6e-5, 2.8e-5) and on 8 loses 3-4x; the THIRD
cumulant alone on 16 directions inside k4 loses most of k4's factor (1.2e-4, 7.2e-5, 1.4e-4) and on 8 all of it (5.5e-4,
6.3e-4, 6.6e-4), although inside k3 a third cumulant on n/4 directions was enough for order-3 precision (D1). So the
number of directions in which the lower-order structure must be kept grows with the target precision (HEURISTIC from a
few k values at small width; FORMAL E8, E13.8). n = 64 (L = 8, nets 300-302, queue X): k2 3.9e-3, 3.6e-3, 1.9e-3;
k3 2.7e-4, 3.1e-4, 4.5e-4; k4 1.6e-4, 6.2e-5, 1.2e-4 (k3/k4 = 1.65, 5.1, 3.7). Errors as multiples of k4's: both cores on
n/4 = 16 future directions 2.0, 9.4, 1.1 (worse than k3 on nets 300 and 301); on n/2 = 32: 0.85, 2.0, 0.76; PCA 5.1,
19.7, 3.5 (n/4) and 1.6, 6.6, 0.69 (n/2). Separating the orders: the fourth cumulant alone on 16 directions 0.29, 0.85,
0.55 (on 32: 0.76, 0.78, 0.74), never worse than full k4 (projection again removes error); the third cumulant alone on
16 directions inside k4 3.0, 10.3, 1.3 (on 32: 1.0, 2.2, 1.0). So at n = 64 the bottleneck is again the third
cumulant: order-4 precision needed it on about half the directions on two of three nets, and on net 301 half still
left 2x; the fourth cumulant tolerated n/4. At n = 32 half the directions left 2.1-7.8x.

## 6. The answer to the brief, and what ARC should update on
**Does any future-aware extension beat RQMC, and does its window grow with n?** Per width (4.4): the deductive member
(D1) as implemented loses to RQMC by 6-200x at n = 64-256 and 150-400x at 512 (medians over nets of each variant;
single runs 2.3-340x at 64-256); under the
registered hypothetical cost of a dedicated implementation it would beat RQMC by 2.4-3.0x up to precision 2-6e-4
(window to 0.6-2.8e3 passes), 1.75-2.2x with its K = 2 pilot charged, 0.5-0.7x with the per-layer direction extraction
used here, and 1.3-1.5x if that extraction cost an assumed ~2n passes (feasibility not established): a small constant
factor at best, not growing with n over 64-256, and smaller at 512 (4.5); under the contraction-cost scenarios it falls
below 1 at 256. The
particle-carried closure loses to RQMC by 45-250x (extrapolated; also to plain MC by 21-83x) and the hybrid, scored
directly, is near unity against RQMC (1.00-1.03x at 6.5e4 passes, 0.86-0.95x at 1e4). None of these windows grows with width over the measured range, consistent with 2.5
(for a method charged Theta(n) passes against a bounded-gain sampler, a growing window needs nMSE o(1/n); the measured
errors of the implementable members change little with n at fixed L/n).

**Extending Theorem A: what extends, and what does not (section 2).**
1. Conditional pricing holds for every future interface (PROVEN under integrability). ARC's own L^2 sketching is the
   annealed case, whose kernel is Theorem A's; the realised future has a kernel of rank at most n whose optimal sketch
   is the answer.
2. For additive corrections that see the next layer through k past-chosen directions, the k-sketch theorem is sharp
   (PROVEN [C+A], for a fixed past-measurable defect): beyond the predictable radius term, they remove at most a
   fraction k / r of the positive-degree price of a defect whose radially normalised moments have stable rank r.
3. For future-chosen subspaces the removed share is the future's spectral share in a Haar linear model, and at most the
   top-k spectral share of the future's coefficient flattenings under a conditional moment model (PROVEN under those
   models [A]); for the tested projection it is exactly the future's modes inside the span (PROVEN for polynomial
   readings [C+A]; for the post-ReLU clouds only the series-free identity applies). Measured for the actual ReLU future:
   the single-cut improvement follows about (1 - s)^{-1.2} in the
   captured share s of the whitened nonlinear relevance, for future, PCA and random choices alike (post hoc; 379 cells,
   correlation 0.96 at n = 64; 0.90 at 128). In the independent-gate mean-field model that relevance has participation
   ratio Theta_P(n) (PROVEN [A]); measured, its participation ratio at the first layer is 17.5, 33.3, 67 (127.5 on three
   n = 512 nets) and the share 16 future directions capture there falls 0.80, 0.59, 0.39 (0.25), while the top n/4
   directions keep ~0.80 at every width. The bridge from this derivative energy to integration error is empirical (the
   share law), not proved.
4. Theorem A's annealed pricing does not extend to future-adapted defects (PROVEN [A]: zero realised error with any
   finite list of moments matched and positive annealed price). A lower bound for every future-aware layer-wise
   estimator would be a super-linear time lower bound for an explicit problem (every algorithm is one): open, and beyond
   known techniques. With the interface restricted (2), the projection restricted (3) or the representation restricted
   to truncated cumulants under uniform laws (2.5), the statements above are what can be proved.

**The mechanism (HEURISTIC as a general explanation; measured for the tested methods).** A backward pass tells a
layer-wise method WHERE the future will read its error, and this information is real (future directions beat the
cloud's own axes by a median per-net factor 2.0-2.9 at k = n/4 in the recursion), but in these nets the reading is
spread over a number of directions that grows with the width, fastest in the first layers where the closures' errors
are made; most of what is read in the kept span is dependence between the kept directions (section 5). For every
representation tried, keeping that law right was either sampling (particles: worse than plain MC at equal cost) or one
rung of ARC's ladder (the third cumulant on ~n/4 directions gives order-3 accuracy; order-4 precision needed about half
the directions or more, exploratory). Mixed moments, oblique features and implicit descriptions were not tested.

**What ARC should update on (our reading).**
1. Their L^2 sketching rule, applied with the tail's LAW, keeps the radius and then radially weighted harmonics (2.1);
   arc-update showed this family gives constant factors at L = n/8. Using the realised tail (a backward pass) is the
   natural next step and it does help, here by constant factors set by the captured share of the future's nonlinear
   relevance, which needed a growing number of directions at fixed share.
2. Two concrete tests for any layer-wise proposal at L = alpha n, each under its stated hypotheses (2.5): charged
   Theta(n) passes against a bounded-gain sampler, its nMSE must fall faster than 1/n; as a truncated hierarchy under
   uniform error and cost laws, its retained dimension per order must stay below its measured per-order error ratio.
   None of the methods tested here or by arc-update was observed to pass either.
3. The one structural positive, stated as a condition: the finite-particle benchmark that keeps the law exact in a
   fixed fraction of future directions has n e falling over 64-256 (k = n/2: 2.9e-3, 1.7e-3, 1.2e-3; k = n/4: 0.032,
   0.026, 0.016; descriptive). A deterministic representation of the cloud's JOINT law in the top ~n/4 future
   directions, costing O(n) passes and keeping that accuracy, would therefore be a candidate for a window that grows
   with width. Nothing tested provides it: independent one-dimensional laws do not suffice (section 5), the third
   cumulant there gives order-3 accuracy (4.2), a ~k-component Gaussian mixture helps at a single cut and less in the
   recursion (section 5). This is the precise form of the "implicit representation" route that arc-update left open.
4. The cheapest positive found is a better order-3 variant: ARC's k3 with its third cumulant projected on n/4 future
   directions keeps a median 78-91% of k3's accuracy (per net 0.26-2.0); on n/2 directions, with the numerically failed
   runs completed by the post hoc rank guard, the medians are 1.18, 1.06, 1.05 (VJP) and 1.13, 1.04, 0.92 (mean-field)
   at n = 64, 128, 256, with divergences at 256, so half width is not a reliable improvement over k3; a deductive
   mean-field backward sweep finds the directions. A dedicated implementation could make order-3 accuracy cost little
   more than order 2: at 256 x 32 a 2.4x window over RQMC up to ~2e-4 under the registered hypothetical cost, 1.75x with
   the K = 2 pilot charged, 0.5x with the dense direction extraction used here and about 1.3x if an extraction at an
   assumed ~2n passes proved feasible. This is arc-update's steelman item 2, now measured, and it is a small constant
   factor at best.

## 7. What is missing
- An unconditional lower bound for future-aware layer-wise estimators: impossible without a super-linear time lower bound
  (2.4). Within restricted classes: a pricing theorem for future-CHOSEN subspaces of the actual ReLU future (proved only
  in a Haar-oriented linear model [A]); a proof that the future's nonlinear relevance has rank Theta(n) for the actual
  gates and whitened cloud (proved in the independent-gate mean-field model [A]); and the bridge from derivative energy
  to integration error (the post hoc share-improvement relation of 4.1 is empirical; a full-rank nonlinear spectrum does
  not by itself imply hardness [A]).
- A dedicated implementation of the projected k3 and its real, operation-level cost: the window in 4.4 uses a
  registered hypothetical cost model (and cost scenarios); ARC's 'simple' diagrams have not been checked for
  contractibility, the K = 2 pilot and the per-layer direction extraction are charged only as scenarios (whether
  approximate or reused directions keep the measured accuracy is untested), and the large-k runs need a stable
  formulation: ARC's K = 3 state covariance becomes indefinite, and with a rank guard 3 of the 8 rescued runs at 256
  diverge or lose badly (4.2).
- Measurement uncertainty: the recursion benchmark's target is the finite-particle algorithm's bias; particle and
  pilot convergence was checked only at n = 64; two replicates per net do not give standard errors for the tiny
  cross products; the share-law cells are clustered within nets. Width trends are descriptive over three widths.
- Widths beyond 512 along L = n/8, and more nets per width (5-8 here; trends are descriptive slopes over 3-4 widths).
- Representations other than "exact on a subspace", cumulants in a subspace, and Gaussian-mixture splits: implicit
  descriptions, non-layer-wise mechanisms, and the participant "scale separation" estimator (forum topic 18298;
  UNVERIFIED; its README reports gains over plain MC of 2.31x at 64 x 16 shrinking to 1.21x at 1024 x 256, width
  exponent -0.20, no RQMC comparison), which belongs to the radius-corrected family arc-update already measured.

## 8. Process, review, deviations, compute
- Co-prover gpt-6-astra via codex-any (model line checked in every raw log; coprover/PROVENANCE.md). Round 1
  (10:36-10:53Z) independent from a lemma-free statement, with my FORMAL v1 frozen at 10:39:14Z before its answer
  existed; round 2 (11:28-11:50Z) on new questions with the dev measurements, my v2 frozen at 11:28:22Z before launch and
  v3 frozen at 11:50:27Z before I read the answer. Accepted items are FORMAL E1 and E9 with labels; my E4 (Theta(n)
  nonlinear rank) and E5 (worst-case integration) were proved by astra in the stated models; my H-overlap was proved in
  a Haar linear model; my reading of R as a retained error factor was corrected.
- Pre-registration (prereg_hashes.txt, frozen/part1/, frozen/amend1/): Part 1 (P1-P9, all code, the imported
  arc-update library) hashed 11:21:08Z, before any run of this lane on the evaluation nets (queue A started 11:21:30Z);
  Amendment 1 (P10-P13, n = 512) hashed 12:17:16Z before any run at that width; the NaN-exclusion correction and the
  table script hashed 11:46:17Z before final scoring.
- Deviations (logs/deviations.txt): header times typed ahead of the clock in four files (the hashes establish which
  artifact was registered and the file times record when it was written; neither is an independently certified
  ordering); a frozen library file modified for about a minute and restored byte-identically (no run used the
  modified version); NaN runs at k = n/2 (rank-deficient whitening of ARC's state covariance) excluded from P7 by a rule
  written before scoring; an interim look at partial results while testing the table script (nothing registered
  changed afterwards). Post hoc analyses are labelled: the share-improvement relation (scratch/share_vs_improvement.py),
  the paired single-cut comparison (scratch/paired_cut.py), the q factors, the window tables, the order-4 runs.
- Compute: CPU only, nice 19, ionice -c3, at most three heavy queue processes (two BLAS threads each, except the post
  hoc H1 rerun, which used default threading), plus short
  single-thread runs at n <= 64 (dev checks, the 2^20-particle check, the order-4 and ridge runs) and, at main's
  request, two exploratory pricing runs (about 15 and 5 minutes, one at n = 128) that briefly made a fourth heavy
  process (deviations item 6); no GPU.
- Review: GPT-6 (gpt-6-astra, fresh session, model line checked; review/review_gpt6.raw and review_gpt6.md, 14:40-
  14:55Z, on the draft with sha 5c4d541d) gave a major revision with 15 findings (12 major, 3 minor); it reproduced
  the scorer outputs, all four tables byte-for-byte and the 20 manifest hashes. All findings were accepted and applied:
  the negative is restricted to the tested methods and widths and the mechanism labelled HEURISTIC (4); the k-sketch
  bound now carries the predictable radius term (1); the unconditional-extension claim is split into what is proved
  and what is open (2); the per-order ratio q is a measured quantity, not bounded by n/L, and the width criterion keeps
  its hypotheses (3); the b2 flooring is disclosed with literal recomputations (5); D1 success counts are shown and the
  n/2 claims restricted (6); paired per-net ratios replace ratios of medians (7); finite-particle and uncertainty
  limits are stated and n e trends called descriptive (8); the co-prover's big-O bound is presented as cost scenarios,
  not lower bounds (9); the K = 2 pilot is charged in an extra window row and ledger gaps are stated (10); the hybrid's
  reference-mean limitation is stated (11); the exploratory exclusions are softened to "not established" (12);
  extrapolations are marked at both ends (13); numbers corrected (14); original, amended and corrected outcomes are
  kept apart (15).
- Review round 2 (review/review_gpt6_r2.md, 15:01-15:08Z, on sha c461da6c, model line checked): major revision. It
  reproduced the corrected paired ratios, retained factors, success counts, window rows and literal P1/P3; it judged
  findings 1-3, 7, 9, 12 and 14 resolved and 4-6, 8, 10, 11, 13 and 15 partly resolved, and raised six new issues.
  Applied: the H1 hybrid estimates are scored directly against the truth (a post hoc rerun of the same controls and
  scrambles, extended to 2^16 points so the hybrid curve is not extrapolated, with expanded and sum-based ledgers; 4.3) [new 1;
  11; 13]; the post hoc scripts use side-effect-free helpers and the scorer's output was re-derived byte for byte
  [new 2]; T2 shows raw, literal and floored statistics, labelled [new 3; 5]; the half-width D1 claim is replaced by
  per-net counts and ranges [new 4; 6]; FORMAL E15 supersedes the old "no unconditional extension" wording (FORMAL
  section 5, E1) and E3's "q <= n/L" [new 5]; the order-4 ratios (2.1-7.8x), the retained range (78-91%) and the
  "smaller still" sentence are corrected [new 6]; the summary carries the polynomial-reading qualification and says
  "over n = 64-256" where a trend is descriptive [4; 8]; an operation-level ledger (section 3) and rows charging the D1
  pilot and direction extraction (4.4) [10]; an extrapolation-sensitivity analysis (4.4) [13]; per-net ranges (4.1)
  [8]; the chronology wording [15]. Not done: additional particle and pilot replicates at n = 256 (the limits are stated
  in section 7). Also added after round 2 (post hoc): the half-width D1 runs completed with a rank guard (4.2) and the
  RQMC comparator-noise check (4.4). Deviations items 10-13 record these changes.
- Review round 3 (review/review_gpt6_r3.md, 17:24-17:34Z, on sha 65b06768, model line checked): major revision,
  concentrated in cost accounting. It verified the direct H1 scoring (the stored error equals residual + beta + cross
  term to 1e-17; v2 algebraically equivalent to v1; reproduction to 3e-8), regenerated every post hoc artifact byte for
  byte in memory, and reproduced the guard, order-4, sensitivity and ledger arithmetic. Applied: the optimistic D1
  extraction row is now labelled assumed-cost arithmetic whose feasibility is not established (ARC's indefinite states
  defeat plain Cholesky), and "all its work charged" is withdrawn; H1 is reported under three ledgers (registered,
  expanded, sum-based; the sum-based one prices the evaluation the rerun performs, and with it the hybrid is near unity,
  1.00-1.03x at 6.5e4 passes, instead of losing); the closing half-width recommendation uses the completed guarded
  results (mean-field median 0.92 at 256); "best fixed control" replaces "best hybrid"; the D1 loss range is given as
  variant medians (6-200x; single runs 2.3-340x); smaller numbers (1.13, 1.65, the guard check "on six reruns", the
  sensitivity range at n = 64, the comparator-noise wording) are corrected. Deviations item 14.
- Review round 4 (review/review_gpt6_r4.md, 17:38-17:43Z, on sha e309100d, model line checked): minor revision. It
  regenerated hyb_direct and d1_ledger byte for byte, reproduced the n = 256 gains under all three ledgers, and judged
  the round-3 items resolved except two partly resolved: the sum-based ledger is now labelled an approximate cost of
  one selected control (what it leaves out is listed in 4.3), and best-per-net and fixed-control gains are reported
  separately. "Tie" is replaced by "near unity at 6.5e4 passes, about 5-15% losses at 1e4; statistical uncertainty not
  quantified", and the contest-budget sentence of 4.4 excepts the hybrid. The measurement-uncertainty item stays partly
  resolved (no additional particle or pilot replicates; section 7).
- Review round 5 (review/review_gpt6_r5.md, 19:59-20:03Z, on sha f65ad331, model line checked), a verification of what
  was added after round 4 (4.5 at n = 512, the n = 512 relevance spectra, the corrected mixture paragraph): minor
  revision. It reproduced the Amendment 1 scorer and post hoc results byte for byte, confirmed P10-P13 as reported and
  the post hoc labels, confirmed the run_mix.py fix (independent draws per batch and half; the corrected development
  run replays exactly) and the mixture recovery fractions (0.78, 0.69, 0.54). Applied: the n = 512 D1 loss range
  stated as medians over finite variants (150-400x; single runs 124-406x), the 6-200x range restricted to n = 64-256,
  and the first-layer participation ratio at 512 given as 127.5.
