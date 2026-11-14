# Can anything estimate the output means of ARC's random 256 x 32 ReLU MLP better than sampling? A checked synthesis (DRAFT)

Draft, 2 November 2026, written 14:15-14:40Z from the lanes' files as they stood at ~14:25Z. Read-only synthesis for
main's review; not reviewed. Several inputs are interim and are marked.

Labels: PROVEN (hypotheses named when it holds only under them); MEASURED-reg = MEASURED (registered); MEASURED-exp =
MEASURED (exploratory, dev or post hoc); HEURISTIC; CITED (external source quoted by a lane); CONJECTURE. Pending:
[pending: lane, ETA].

Sources, all under /home/agent/workspaces/mats/: AU = outbox/arc-update-2026-11-02.md; SM =
outbox/scale-mode-2026-11-02.md; PI = work/preint/FORMAL.md (PI-notes = work/preint/notes/QUESTIONS.md, READING.md);
RN = work/randomness/FORMAL.md (RN-notes = work/randomness/notes/READING.md); FA = work/future-aware/future-aware.md;
BK = work/backward-bits/FORMAL.md; BAA = work/bit-accounting-arc/results/P1_tables_partial.md; BAA-upd =
work/bit-accounting-arc/results/update_vs_1133Z.md; BA = work/bit-accounting/bit-accounting.md; REL =
outbox/relevance-accounting-2026-11-01.md; BB = outbox/bits-bounds-2026-11-01.md; ND = outbox/nd-exact-2026-11-02.md;
UE = outbox/unrestricted-estimators-2026-10-31.md; LOG hh:mm = the 2 November entry at that time in
outbox/overnight-report-2026-10-29.md; "Sum." = a report's Summary section; A# and C# = rows of Appendix A and of the
conflicts list.

**TL;DR.** No positive. Every deductive method tested at 256 x 32 wins at most a constant factor, in a budget window
that closes within ~5% of the contest budget; at the budget RQMC beats those that fit by 7-65x, and hybrids built on
RQMC reach 0.8-1.0x of it. The negative is restricted: measured for those methods, proven under hypotheses for
black-box sampling rules (interim, unreviewed) and for ARC's hierarchy if ARC's own law is tight. The sources tie high
angular degree to the sampling bound; the tie to the deductive ladder is partial, and the ladder part is a CONJECTURE.

## 1. The problem and the bar

- Estimand: the mean of every final-layer neuron of one fixed bias-free He-ReLU net, x ~ N(0, I); the MSE is
  normalised so that N Monte Carlo (MC) samples score 1/N (CITED, AU §1; metrics differ in detail, C17). ARC conjecture
  an order-K error c_K (L/n)^K at ~n^(K-1) passes and concede: "This depth scaling is worse than Monte Carlo sampling"
  (CITED, AU §1).
- Phase 1 is exactly 256 x 32 (L/n = 1/8), budget 64,850 passes. Nets 200-204 match its setting; the contest scores raw
  MSE x max(0.1, C/B) (SM §2; CITED). Organisers: Phase 1's best methods "tended to be based on variations of sampling"
  (AU §1). Phase 2 (1024 x 16, L/n = 1/64) is a different regime (section 2).
- The comparator is the best sampler (UE §3.2). At 256 x 32 RQMC gains 1.2-2.2x over MC and a Kerdock design 2.4x (AU
  §3; MEASURED). The public board's best Phase-1 entry is roughly 10-50x below plain MC at full budget (AU §1; CITED,
  approximate, method not public; C10).
- Kelvin's bar: a major positive for 256 x 32, or a major argument that it is hopeless; since ~13:09Z, a route to a
  better error-vs-compute asymptote, preferably from the sampling side (LOG 13:00, 13:10).

## 2. Deduction side

Measured (MEASURED-reg unless marked):
- ARC's own code at 256 x 32 (AU Sum. 1, §4.2): order 2 wins a ~2x window near 450 passes at precision ~1e-3 on 4 of 5
  nets, and RQMC catches up by 400-1,800 passes. Order 3 loses 5-30x at its own cost. At the budget RQMC is 7-40x more
  accurate than any ARC variant that fits.
- Along L = n/8, n = 32-256 (AU §4.1): each order costs ~n/4 to ~3n times more; the error falls 4-8x from order 2 to 3
  and 2-4x from 3 to 4. The error-cost exponent between orders 2 and 3 is 0.23-0.52, against sampling's 1. Order 2's
  gain over MC falls from 8.1 to 2.6.
- Order 2 makes 44-64% of its final error by layer 8, by accumulation: a single moment-matched Gaussian cut of the true
  law costs only 1.3-2.4e-4 of its 1.2e-3 (AU §4.3).
- Stronger variants are constant factors in windows. Scale mode (forum topic 18298): 2.5x RQMC at 688 passes, caught at
  ~2.3x that cost, 33-65x behind at the budget; its two-level exponent equals ARC's at 256 x 32 (SM §5, §8). Order 3
  with its third cumulant kept on n/4 future directions: 2.4-3.0x under a best-case cost no implementation has reached,
  below 1x at 256 x 32 under the co-prover's cost bound (FA §4.4; interim). The radius-corrected closure errs near
  L/n^2 (4.9e-4 at 256 x 32; match post hoc) but as run costs at least 2^19 passes; a deductive version would need
  fewer than ~5 order-2 steps per layer to beat MC (AU §4.4c). None exists.
- At Phase 2's 1024 x 16 the exponent is 0.83 and order 3 is ~19x more accurate than full-budget RQMC at 1.5x the
  budget (SM §7). The negative concerns depth growing with width.

Proven:
- Pricing identity (AU §2.2, A2; PROVEN at every width): a forward layer-wise estimator's layer-l defect costs
  sum_d b_d ||M_d||^2 over its radially normalised moment tensors, b_d >= 0, sum b_d = 1. It covers ARC's order-2
  'simple' and the exact Gaussian recursion (AU §2.1). It prices a given defect and bounds no class of representations.
- Ladder (AU §2.5; PROVEN under the hypothesis that ARC's law is a lower bound uniformly over orders): the best
  affordable error falls no faster than C^(-gamma), gamma = log(n/L)/log n: 0.375 at 256 x 32, tending to 0 at
  L = alpha n. The law's measured form covers orders <= 4 only.
- Gaussian closures have bounded gain at L = alpha n (AU §2.4, BB Sum. 10; PROVEN under the radius-sign hypothesis
  H_R, a CONJECTURE that one exploratory net did not contradict).
- An unconditional lower bound would be a super-linear time lower bound for an explicit problem (AU §8, UE §3.3); no
  technique for that is known (HEURISTIC).

Also tested and failed (MEASURED): certified codes at 8 x 256, 23-106x behind RQMC (ND Sum. 5); deterministic
branch-and-bound at widths 3-32 (UE Sum.); code propagation, 60-5,000x behind MC at 16 bits (BA §5.7); particle-carried
future-aware closures, 40-250x behind RQMC (LOG 13:35; interim).

## 3. Sampling side

- Random bits are free on average: some fixed design matches radius-exact MC's weight-averaged risk (RN T3; PROVEN).
- Theorem F7 (PI F7; PROVEN; unreviewed, co-prover answer not yet in; C14). Assumptions: the rule is linear in f (N
  weighted point values or line + radius integrals, signed weights allowed); nodes and weights fixed or randomised
  independently of the weights W; the error averaged over W. Then the averaged MSE is a discrepancy in the kernel
  |x||x'| kappa(<xhat, xhat'>), exact at every width. Under H_even(p, c) (the energy at even degrees 2j whose
  degree-j harmonics have dimension N_j >= 2N holds a fraction >= c of the variance, for all n):
  E_W nMSE >= (c/2 - o(1))/N. So no such rule beats MC by more than 2/c, and none keeps an exponent above 1 as n grows
  with N <= n^p. Not covered: odd degrees, per-net error, N far above the harmonic dimensions at fixed width.
- F9 (PI E1, 14:14Z): under a Gaussian surrogate prior the bound also covers adaptive and nonlinear algorithms (PROVEN
  from CITED average-case results; abstracts checked only). Under the true prior ~nL = 8,192 evaluations at 256 x 32
  determine W, so beyond that no information bound can hold and the question is computational (PI E1c).
- Its size at 256 x 32 is uncertain. The infinite-width proxy gives N x nMSE >= 0.34-0.18 (PI F7g; HEURISTIC, proxy off
  ~2x in low degrees); RQMC sits at 0.81-0.46 (AU §3). H_even is unverified at finite width (C6).
- Positive designs (RN T4, E1.1; PROVEN): at n = 256 every design of <= 8.6e4 points pays the even tail from degree 6
  in full. Unions of independently rotated copies of a design have exponent exactly 1 (RN T1; PROVEN).
- Measured: RQMC flattens to c(n)/N, with c ~0.19 at n = 32 rising to ~0.47 at 256 (PI-notes A4, C8; read as
  asymptote 1, HEURISTIC). Dev at 256 x 32: antipodal Kerdock 2.1-4.0x MC; scrambled Sobol in the polar frame of W_1
  (white-box) 1.8-3.1x against 1.6-1.7x for standard RQMC (LOG 13:48; MEASURED-exp). [pending: randomness, registered
  results and exponents, ETA ~15:15Z]
- Multilevel: the shared-trajectory sampler gains Omega(n / log^2 n) at L = alpha n (UE §3.7; PROVEN under layerwise
  cost hypotheses), at exponent 1. Measured ideal gain at 256 x 32: ~1 (AU §3, BB Sum. 8; MEASURED-reg).
- Pre-integration: exact per line (PI F1-F2; PROVEN) at ~nL/2 passes per line, 3,469 measured at 256 x 32 (PI F3);
  expected exponent gain ~1/d (PI F5; HEURISTIC). Dev: steeper only at input dimensions 4-6, and 25-500x worse at equal
  cost for n >= 32 (LOG 14:03; MEASURED-exp). [pending: preint, registered runs, no ETA given]
- At fixed width better asymptotes exist. Exact polyhedral integration of one net costs polylog(1/eps), with
  astronomical constants (AU §6 item 5; PROVEN), and QMC-type rates exceed 1 with a crossover cost exponential in d
  (PI F5, F8; HEURISTIC). Measured, exact integration wins only at widths 2-3 (AU §2.6, ND Sum. 6).

## 4. Hybrids

- Criterion (LOG 13:52): an exact-mean control whose residual variance falls like (construction cost)^(-delta) gives
  MSE ~ C^(-(1 + delta)) (PROVEN, arithmetic under that scaling).
- Measured delta (FA §5; MEASURED-exp, one net per width): layer-1 controls with all quadratics in the top-k future
  directions give delta <= 0.25 at n = 64 and <= 0.09 at n = 128. At 256 x 32 the registered control leaves 55% of the
  variance and gains nothing over RQMC. delta shrinks with width.
- Why: the controls remove what scrambled Sobol already integrates (RQMC on the residual over RQMC on h_L: 0.71-0.85,
  0.91-0.98, ~1 at n = 64, 128, 256; FA §5, §4.3). Best registered hybrid: 0.8-1.0x RQMC (FA §4.3; interim).
- Mixtures (FA §5; MEASURED-exp, n = 64): about k Gaussian components in k = 16 future directions recover much of the
  joint law at a single cut. As a recursion (M = 16-32, ~1.8-3.5e3 passes) they lose to RQMC by 3-20x, and M must grow
  with k ~ n/4.

## 5. Mechanism

- Between layers, compressed descriptions move almost linearly. At 256 x 32 consecutive output-relevant coordinates
  map with linear R^2 0.82-0.99 (BA §5.3, AU §5; MEASURED, registered test BA.1 missed narrowly); in small nets a
  threshold map captures a median 96% of each next code bit's predictable part (BA §5.1; MEASURED). Scrambled controls
  have not been through these tests (AU §8), so this reads "the tests found low degree". [pending: bit-accounting
  phase 2, no ETA]
- Relative to coordinates where the law is computable, dependence is high-degree and high-dimensional: mean Hermite
  degree 21 at the input (AU §3, from REL; MEASURED-reg); annealed input degree n / E_W V, linear in n at fixed L/n (AU
  §5; PROVEN); third-cumulant participation ratio ~115 at layer 1, where closures make their error (AU §4.3;
  MEASURED-reg, descriptive); the future's frame uniformly random relative to the past cloud (REL Sum. 1; PROVEN).
- Bits price representation only: with decoder E[a | c] a code sum is unbiased at any rate (AU §5; PROVEN). Local codes
  at 256 x 32 need ~629 / 1,138 bits at the input and ~19 / 95 at layer 31 for 10% / 1% nMSE (BAA; MEASURED-reg;
  uncertified upper bounds; C9).
- Backward bits (BK §1-3, §6; PROVEN): Kelvin's target T_l = I(h_l; c_(l+1)) is the relevance ceiling, no code with
  countably many values carries all of it (under a checkable condition), and a chain's targets describe its own codes.
  Dev, 16 x 16: the rate rule keeps 69-77% of T per step and the target collapses from 4.85 to 0.44 bit by l = 10,
  truth ~3 (LOG 14:19; MEASURED-exp). [pending: backward-bits, ~3-4 h from 14:19Z]
- AU §5's answer to Kelvin's question: codes are not scrambled layer to layer, as far as the tests see; the bits are
  scrambled relative to the coordinates in which the law is computable.

Does "depth pushes the output's dependence into high angular degree" (PI F8) explain both sides?
- Sampling half: supported. The kernel prices every rule linear in f, and energy beyond the budget's degree reach
  cannot be integrated by them (PI F7; PROVEN under H_even; RN T4 for positive designs without H_even).
- Deductive half: partly supported. A kernel of the same form, |h||h'| kappa_t, with t = L - l layers to go, prices a
  closure's defect (AU §2.2, FA §2.1; compare PI F7a). Order-K matching does not in general cancel degrees above K or
  the radius (AU §2.2, A3; PROVEN). At infinite width a 31-layer future reads the non-radial, non-mean part at mean
  degree 42 (AU §2.3; PROVEN, exact for the infinite-width kernel). Three gaps remain. The finite-width spectrum beyond
  b_0 is unmeasured and differs at t proportional to n. The deep future's weight is mostly radial (b_0 = 0.95 at
  t = 31, finite width; MEASURED-exp), and the degree >= 2 weight shrinks with depth (0.18 to 0.024, infinite width).
  A2 does not force affordable representations to have large high-degree defects (AU §2.2-2.3).
- The ladder: no source derives the per-order factor (~n/L) or gamma from the degree spectrum; AU §2.5 takes both from
  ARC's law. CONJECTURE: the ladder's exponent decay along L = alpha n and the sampling bound are two consequences of one
  degree spectrum. A related HEURISTIC: order-4 precision appears to need the lower orders in nearly all directions (FA
  §5, n = 32).
- F8's premise ("most of the variance sits in even angular degrees") is infinite-width; at finite width degrees <= 2
  hold 0.4-0.75 of the angular variance on the ladder (RN E1.5d; C6).

## 6. Status

The negative is restricted, in three parts.
- Proven in restricted classes: rules linear in f have exponent 1 for N <= poly(n), on average over nets, under H_even
  (PI F7); all black-box rules obey the same bound under a Gaussian surrogate prior (PI E1); positive designs at n = 256
  are capped by the degree-6 even tail (RN T4); ARC's hierarchy has exponent tending to 0 if its law is a uniform lower
  law (AU §2.5); Gaussian closures have bounded gain under H_R (AU §2.4); future-aware sketches through k past-chosen
  directions remove at most k/r of a fixed defect's price (FA §2.2).
- Measured only: every method in sections 2-4 (ARC's code to order 3 at 256 x 32, order 4 below); pre-integration
  and structured designs at dev level so far.
- Not covered: nonlinear black-box estimators under the true prior (PI E1c); white-box sampling beyond polar alignment,
  exact tails and the multilevel sampler; implicit representations of the early structure; scale-law orders >= 4 (SM
  §8); a deductive radius-corrected closure; non-layer-wise methods; per-net versions of averaged bounds; widths above
  256-512.

What would count as a positive (criteria the sources state):
- Deduction at L = alpha n: error o(1/n) at Theta(n) passes, or a retained dimension per order below the per-order
  error ratio, at most n/L (FA §2.5). The scale-mode lane's registered escape criteria: a two-level exponent >= 0.9 at
  n = 128 and 256 with the higher level beating RQMC, or a gain over RQMC above 1 at every width and growing (SM §4).
- Sampling or hybrid: an error-vs-compute exponent above 1 that survives n growing along L = alpha n (LOG 13:10;
  PI-notes Q4), or delta > 0 uniformly in width (LOG 13:52).
- The one structural lead: a deterministic O(n)-pass representation of the cloud's joint law in its top ~n/4 future
  directions, at the benchmark's accuracy, would open a window that grows with width (FA §6 item 3). None is known.
  [pending: future-aware, the 256 x 32 benchmark behind it, ETA ~14:50Z per LOG 13:35]

## 7. What ARC could reasonably update

ARC already concede that their hierarchy scales worse than sampling with depth (AU §1), and an adversarial adjudication
judged the evidence short of "hopeless, ARC should update" (LOG 10:24). The evidence supports a quantitative update of
that concession. At the Phase-1 shape their own code wins a ~2x window near 450 passes on 4 of 5 nets and is 7-40x
behind RQMC at the budget (AU §4.2). Along L = n/8 the measured exponent between orders 2 and 3 is 0.23-0.52 against
sampling's 1 (AU §4.1). Every stronger deductive variant tested, the public scale-mode estimator included, is a
constant factor in a window that closes within ~5% of the budget (SM §5, FA §4.4, AU §4.4). A proposal for depth
proportional to width should state its error-cost law along L = alpha n, compare with the best sampler (~2-2.5x MC
here), and say how it represents the non-Gaussian structure of the first ~8 layers, where the tested closures make most
of their error (AU §4.3, §7). The sampling-side theorems (PI F7, F9) are unreviewed and should not yet enter any update.

## Appendix A. Numbers and their sources

| # | Quantity | Value | Label | Source |
|---|---|---|---|---|
| A1 | Phase-1 shape and budget | 256 x 32; 2.72e11 flops = 64,850 passes | CITED | SM §2; AU §1 |
| A2 | Public Phase-1 board, best entry | raw MSE 3.63e-8 at 51% of budget; ~10-50x below plain MC at full budget | CITED (approximate; method not public) | AU §1; SM §2 |
| A3 | Phase-2 engineered order-3 entries | 1.10e-8 at 12% of budget; ~100x below full-budget MC | CITED (external) | SM §2 |
| A4 | RQMC gain over MC, 256 x 32, nets 200-204 | 1.24, 1.48, 1.74, 2.19, 2.10 at N = 2^8, 2^10, 2^12, 2^14, 2^16 (N x nMSE 0.81, 0.68, 0.57, 0.46, 0.48) | MEASURED-reg | AU §3; reproduced from work/arc-update/runs/rqmc_n256_L32_s20[0-4].json |
| A5 | Kerdock design | 0.40/N (2.4x MC), sigma = 1e-3, 24 nets | MEASURED | AU §3 (arc-noisy) |
| A6 | ARC variants at 256 x 32 (medians) | k1 2.6e-2 at 1 pass; k2 1.2e-3 at 393; k2a 8.4e-4 at 452; k3f 2.0e-4 at 4.9e4 | MEASURED-reg | AU §3, §4.1 |
| A7 | k2a against RQMC at own cost | median 2.0x (0.85-4.6x), wins on 4 of 5 nets; RQMC catches up at 400-1,800 passes | MEASURED-reg | AU Sum. 1, §4.2 |
| A8 | k3f against RQMC at own cost | 5-30x worse | MEASURED-reg | AU §4.2 |
| A9 | RQMC against best affordable ARC variant at 6.5e4 passes | 7-40x (median 22x); P4 (>= 10x on 5 of 5) fails: 6.7x on net 200 | MEASURED-reg | AU §4.2 |
| A10 | Exponent gamma between orders 2 and 3, L = n/8 | 0.52, 0.39, 0.23, 0.30 at n = 32, 64, 128, 256; ARC's law 0.60, 0.50, 0.43, 0.375 | MEASURED-reg (n = 32, 64 descriptive) | AU §4.1 |
| A11 | Per-order factors | error / 4-8 (order 2 to 3), / 2-4 (3 to 4); cost x ~n/4 to ~3n | MEASURED-reg | AU §4.1 |
| A12 | Order-2 gain over MC against width | 8.1 at n = 32 to 2.6 at n = 256 | MEASURED-reg | AU §4.1, §7 |
| A13 | Share of order-2 error made by layer 8 | 44-64% (median 59%) | MEASURED-reg | AU §4.3 |
| A14 | Single-cut Gaussian defect against the full recursion | 1.3-2.4e-4 (l = 2-24) against 1.2e-3 | MEASURED-reg | AU §4.3 |
| A15 | Third-cumulant participation ratio of the true cloud | 114-116 (l = 1), 23-28 (l = 4), 4-11 (l = 8), 1-4 (l >= 12) | MEASURED-reg (descriptive) | AU §4.3 |
| A16 | Radius-corrected closure, finite-sample recursion | 1.9e-3, 9.2e-4, 4.9e-4 at n = 64, 128, 256; within 6% of L/n^2 | MEASURED-reg (L/n^2 match post hoc) | AU §4.4(c) |
| A17 | Its cost and break-even | >= 2^19 passes per replicate (>= ~255x what MC needs); deductive break-even < ~5 (MC) or 2-3.5 (RQMC) order-2 steps per layer | MEASURED-reg; arithmetic | AU §4.4(c) |
| A18 | Scale mode (topic 18298) at 256 x 32 | 3.6e-4 at 688 passes; 2.5x RQMC at own cost (CI 2.0-4.7); RQMC catches up at ~2.3x; 33-65x behind RQMC at budget | MEASURED-reg | SM §5 |
| A19 | Scale-law two-level exponent against ARC's, same nets | 1.0, 0.72, 0.52, 0.30 against 0.54, 0.37, 0.28, 0.31 (n = 32-256) | MEASURED-reg | SM §5 |
| A20 | Scale mode on its own diagonal L = n/4 | median gain over RQMC 0.79, 0.73, 1.2, 0.43, 2.0 (64 x 16 to 1024 x 256) | MEASURED-reg | SM §6 |
| A21 | Scale mode with flopscope prices for its special functions | gain over RQMC 2.5 to 1.16 at 256 x 32 | MEASURED-exp | SM §5 |
| A22 | Phase 2, 1024 x 16 | exponent (k2a to k3f) 0.83; k3f ~19x below full-budget RQMC at 1.5x the budget | MEASURED-reg | SM §7 |
| A23 | Projected order 3 (k = n/4 future directions), best-case cost | 2.6x, 3.0x, 2.4x over RQMC up to 5.9e-4, 3.1e-4, 2.2e-4 (n = 64, 128, 256); co-prover bound 0.93x / 0.36x at 256 | MEASURED (interim draft) | FA §4.4 |
| A24 | Future-aware benchmark gains over the Gaussian closure | k = n/4: 5.2, 9.1, 19x; k = n/2: 75, 121, 270x; k = 16: 5.2, 2.3, 1.4x (n = 64, 128, 256; 256 on 3 of 5 nets) | MEASURED-reg benchmark (interim) [pending: future-aware, ETA ~14:50Z per LOG 13:35] | LOG 13:35; FA §4.1 |
| A25 | n x error at order-3 level | 0.034, 0.040, 0.068 at n = 64, 128, 256 (not falling) | MEASURED (interim) | FA §4.2 |
| A26 | Price weights at infinite width | b_0 = 0.318, 0.681, 0.834, 0.930, 0.973; b_>=2 = 0.182, 0.178, 0.116, 0.057, 0.024; mean degree of b_>=2 = 2.75, 4.84, 8.22, 17.2, 42.3 at t = 1, 4, 8, 16, 31 | PROVEN (exact for the infinite-width kernel) | AU §2.3 |
| A27 | Radius weight at finite width, t = 31 | 0.948 (infinite width 0.973) | MEASURED-exp | AU §3 |
| A28 | F7 bound with the NNGP proxy, 256 x 32 | N x nMSE >= 0.34 to 0.18 for N = 2^6 to 2^22 | HEURISTIC | PI F7(g) |
| A29 | Evaluations that identify W | ~nL = 8,192 at 256 x 32 | PROVEN (information count) | PI E1(c) |
| A30 | Non-Gaussianity, 32 x 4 smoke test | excess kurtosis of unit outputs 4.7; Var log abs(h_L)^2 = 0.54 against 5L/n = 0.625 | MEASURED-exp (32 nets) | PI E1(d) |
| A31 | Positive-design degree threshold, n = 256 | full even tail from degree 6 for m <= 8.6e4 points (from degree 8 for m <= 8.1e5) | PROVEN | RN T4, E1.1 |
| A32 | Even tail from degree 6, share of radius-exact variance | 0.13-0.20 (2 dev nets) | MEASURED-exp | RN §5 |
| A33 | Degrees <= 2, share of angular variance at finite width | 0.4-0.75 on the ladder | MEASURED-exp (co-prover note) | RN E1.5(d) |
| A34 | RQMC N x nMSE at the largest budgets, L = n/8 | ~0.19 (n = 32), ~0.33 (64), ~0.38 (128), ~0.47 (256); local exponents 0.99-1.03 | MEASURED (medians include a dev net; C8) | PI-notes QUESTIONS A4 |
| A35 | Dev designs at 256 x 32 | antipodal Kerdock 2.1-4.0x MC; polar-frame RQMC 1.8-3.1x against standard 1.6-1.7x at 2^14 | MEASURED-exp | LOG 13:48 |
| A36 | Multilevel shared-trajectory sampler | ideal 0.99-1.06 at 256 x 32; 1.21, 1.41, 1.58, 2.43 at L = n (n = 32-256); Omega(n / log^2 n) at L = alpha n | MEASURED-reg (optimistic); PROVEN under cost hypotheses | AU §3; BB Sum. 8; UE §3.7 |
| A37 | Pre-integration cost | ~3 + nL vertices per half circle (0.76-1.39x of that on 12 nets); 6,686 at 256 x 32; exact line 3,469 passes, windowed 752 | PROVEN (Kac-Rice); MEASURED | PI F3 |
| A38 | Pre-integration exponents, dev | n = 4: 2.12 vs 1.79; n = 6: 1.70 vs 1.40; none at n >= 64; 25-500x worse at equal cost for n >= 32 | MEASURED-exp | LOG 14:03 |
| A39 | Value of one line | variance ratio 0.87 (random u), 0.69 (active-subspace u) at 32 x 4; 0.55 at 8 x 256 | MEASURED-exp | PI F6 |
| A40 | Gate-partition surface area | ~nL / sqrt(2 pi), ~3,300 at 256 x 32 | PROVEN (annealed) | RN T7, E1.2 |
| A41 | Exact-mean control residual share, layer 1 | n = 64: 0.38 to 0.19 (k <= 32), ~k^-0.25; n = 128: 0.59 to 0.41 (k <= 64), ~k^-0.09; n = 256: 0.55 (k = 16) | MEASURED-exp (net 300; 256 registered) | FA §5, §4.3 |
| A42 | RQMC on residual over RQMC on h_L, 2^14 points | 0.71-0.85 (n = 64), 0.91-0.98 (128), ~1 (256) | MEASURED | FA §5, §4.3; LOG 14:09 |
| A43 | Best hybrid gain over RQMC at 1e4 / 6.5e4 passes | 0.78 / 0.98 (n = 64), 0.93 / 0.93 (128), 0.91 / 0.83 (256) | MEASURED (registered design; interim) | FA §4.3 |
| A44 | Mixture closure, n = 64 | 2.8e-4 to 2e-3 at ~1.8-3.5e3 passes, where RQMC is at 1.7e-4 / 1.0e-4: 3-20x worse | MEASURED-exp | FA §5 |
| A45 | Exact conditional-mean controls at the input | at most 1.33, 1.80, 3.11x for 16, 64, 128 features, before cost | MEASURED | REL §9 |
| A46 | Value function's mean Hermite degree | 21, 7.7, 4.1, 1.9, 1.3, 1.04 at l = 0, 4, 8, 16, 24, 31 | MEASURED-reg | AU §3 (REL) |
| A47 | Directions for 99% predictive capture | 256, 192, 128, 80, 48 at l = 0, 4, 8, 16, 24 | MEASURED (point estimates) | AU §3; REL Sum. 3 |
| A48 | Consecutive-layer relevant maps, 256 x 32 | linear R^2 0.82-0.99; sign bits within 0.03 of an MLP from l = 16 | MEASURED-reg | BA §5.3; AU §5 |
| A49 | Local code bits at 256 x 32, 10% / 1% nMSE (4-net medians) | l = 0: 629 / 1,138; l = 8: 111 / 410; l = 12: 74 / 289; l = 16: 39 / 205; l = 20: 27 / 148; l = 31: 18.9 / 95; output 18.7 / 94 | MEASURED-reg (best-found upper bounds; not certified) | BAA per-layer profile; BAA-upd item 2 |
| A50 | Small nets, local bits per halving of error | ~n/2 at the input (9.3 at n = 16, 18.0 at n = 32); 1.1-1.2 at the output | MEASURED | BA Sum. 4 |
| A51 | Code propagation as an estimator | 60-5,000x worse than MC at b = 16 | MEASURED | BA §5.7 |
| A52 | Narrow-deep 8 x 256 | 10-76 classes at l = 8, each up to 113,000 input regions; certified code 23-106x worse than RQMC; RQMC with exact tail 3-15x better than RQMC | MEASURED-reg | ND Sum. 1, 3, 5 |
| A53 | Backward bits, dev | 16 x 16: rate rule keeps 69-77% of T per step, target 4.85 to 0.44 bit by l = 10 (truth ~3); 8 x 8: rate-capped chain keeps 38% of I(h_0; c_L) = 0.92 bit at rate 0.66 | MEASURED-exp | LOG 14:19; LOG 13:43 |

## Conflicts and uncertain points

Where a later log entry corrects a report or an earlier entry, the later version is used above.

- C1. arc-update scorecard. LOG 10:16 gives 28 hold, 3 fail, 4 partial, 4 undetermined; LOG 10:24 corrects to 29
  holds; the final AU (13:50Z update) and LOG 13:29 give 31 hold, 4 fail (P4, P7, P9b, P12), 1 partial (P19), 4
  undetermined (P20-P23). Final version used.
- C2. Radius-corrected lead. LOG 10:16 claims a "constant ~5/J gain" and "astra r4 proves L/n^2 order". LOG 10:24, LOG
  10:34 and AU §9 correct this: the gain was mis-costed (393 passes charged to a recursion using 2^19 samples per layer)
  and withdrawn, and E12.2 is relabelled a consistency decomposition. LOG 08:29's "neither grows polynomially with n" is
  withdrawn (AU §9, finding 9); the mid-layer slope 0.67 cannot be told from 1 at these widths (AU §4.4b). LOG 06:54's
  "n^-2 first-cut floor" is an asymptotic lower bound only (AU §2.4), and "floor" was dropped for L/n^2 (no convergence
  study). AU used.
- C3. Order-2 window. LOG 07:39 says K = 2 beats RQMC "~2x (1.9-4.6x)". AU §4.2: k2a wins on 4 of 5 nets (median 2.0x,
  0.85x on net 203); 1.9-4.6x takes the better of k2 and k2a per net, a post hoc choice. AU used.
- C4. Degree of the future's reading. LOG 06:16 says "degrees up to ~2t". AU §2.3 gives the exact mean degree of the
  b_>=2 mass, 42.3 at t = 31 (the co-prover corrected an understated FFT), growing like 2t^2/(9 pi^2). AU used.
- C5. "Mean degree" names different objects. PI F8: the input map's degree "grows like L" (consistent with the
  finite-width annealed identity n / E_W V at L = alpha n, AU §5). AU §2.3: the infinite-width b_>=2 spectrum, growing
  like t^2; finite-width b_d differ from it at t proportional to n. Relevance and input-bits count Hermite degree; PI
  and RN count spherical-harmonic degree, and the two differ (RN E1.4). No contradiction, but they must not be mixed.
- C6. How much variance sits in high even degrees at 256 x 32. UE (via RN-notes READING): plug-in even tail above
  degree 2 = 0.48 of the angular variance (post hoc, 3 nets, a_2 estimate negative), ceiling "about 4x". RN §5, dev:
  even tail from degree 6 = 0.13-0.20. PI F7(g)'s NNGP proxy implies far more high-degree energy, and PI F8 says most
  variance sits in even degrees, while RN E1.5(d) says degrees <= 2 hold 0.4-0.75 at finite width on the ladder. This
  sets c in H_even and the design ceiling. [pending: randomness, ETA ~15:15Z]
- C7. RN §5 says "no positive design of that size can gain more than ~5-8 (T4 with these numbers)". T4's own formula,
  2 Lambda / Lambda_even>=6 with 0.13-0.20, gives ~10-15. Either a factor-2 slip or a different quantity (an ideal
  5-design's gain, Lambda / Lambda_>=6, would give ~5-8). Unresolved; no ceiling value is quoted in the main text.
- C8. RQMC N x nMSE medians. PI-notes QUESTIONS A4 and READING quote 0.80, 0.63, 0.52, 0.51 (0.48) at 256 x 32; AU §3's
  gains imply 0.81, 0.68, 0.57, 0.46, 0.48. My check of work/arc-update/runs/rqmc_*.json: the PI-notes medians include
  dev net 100 (and dev net 9500 on the ladder); on the registered nets the AU values reproduce exactly. The plateau
  reading is unchanged.
- C9. Bit counts at 256 x 32. AU §3 and §5 quote net 4000 (l = 31: 15 / 90 bits; l = 16: 32 / 186; l = 0: 575-642 /
  1,084-1,126); BAA gives 4-net medians (A49), and net 4000 is the cheapest net (BAA-upd item 2). BAA (12:37Z) has
  layer 12 at 44 / 244 bits from one net; BAA-upd (13:45Z) supersedes it with 74 / 289 (4 nets); layer 20 moved from
  19.6 / 125 to 27 / 148. No ARC cell is certified, and early-layer counts are best-found upper bounds (LOG 11:36,
  12:44). Latest medians used.
- C10. The Phase-1 bar. The public board's best entry is ~10-50x below plain MC at full budget (AU §1; approximate,
  different nets, method not public). The best samplers measured here give ~2-2.5x (RQMC, Kerdock); the arc-noisy
  report calls its Kerdock reproduction (2.4x) "the design behind the best published Phase-1 entries", and RN-notes
  READING says "published Phase-1 entries ~3x". If the board entry is a black-box rule linear in f on nets like ours,
  it would exceed the F7 proxy numbers (HEURISTIC). The sources cannot explain the gap.
- C11. Future-aware at 256 x 32. FA §4.1 still has placeholders; FA results/tables_part1.md (12:24Z, 2 nets) gives
  9.3x (k = 64), 169x (k = 128), 1.4x (k = 16); LOG 13:35 (3 of 5 nets) gives 19x, 270x, 1.4x. The log is used,
  marked interim.
- C12. FA §7 still calls the scale-mode estimator "UNVERIFIED" with no RQMC comparison; SM (final 13:24Z) supersedes
  it. LOG 12:15 gives the best hybrid as 0.8-1.1x; FA §4.3 gives 0.8-1.0x by median (0.65-1.65x per net at 256). SM and
  FA §4.3 used.
- C13. PI F8 (14:00Z) names "estimators that are non-linear in f" as an escape; PI E1 (14:14Z) shows that under a
  Gaussian surrogate prior nonlinearity and adaption do not help, while under the true prior the information bound
  fails beyond ~8,192 evaluations. E1 used. The cited average-case theorems were checked from abstracts only.
- C14. F7 status. FORMAL v1 was frozen before the co-prover's round-1 answer (started 13:47Z); at ~14:20Z that answer
  was still being produced (work/preint/coprover/astra_r1.raw) and is not incorporated. Main checked the proof (LOG
  14:03).
- C15. LOG 14:03 summarises pre-integration as steeper "only at input dim 4-6"; work/preint/PREREG.md P3c notes that dev
  also showed a steeper slope at n = 32 over a short window, while losing at equal cost. Registered test pending.
- C16. Kerdock 2.4x (A5) was measured on noisy nets (sigma = 1e-3); the dev 2.1-4.0x (A35) is on noiseless dev nets.
  Registered noiseless numbers are pending (randomness).
- C17. Metrics differ: ARC's paper normalises by per-neuron variance, the contest scores raw MSE x max(0.1, C/B) under
  flopscope, and the lanes use |est - Y|^2 / tr Var with ARC's flop counter (AU §1, SM §2). All board comparisons are
  approximate.
