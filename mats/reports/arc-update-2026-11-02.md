# ARC's cumulant hierarchy on deep random MLPs: a registered benchmark at the Phase-1 shape, the error-cost ladder along L = n/8, and the ladder an estimator must escape (2 November 2026)

Status (10:30Z): final for this lane's run; reframed and corrected after an independent adjudication relayed by main
(section 9), then checked by gpt-6-astra. Update 13:50Z (main): the queued runs have landed (the P9 rank runs on nets
203-204, k3fa on net 200) and the scorecard is regenerated: 31 hold, 4 fail, 1 partial, 4 undetermined; k3fa changes
nothing (section 4.1). The scale-mode counter-claim (forum topic 18298) has since been tested by a separate lane
(outbox/scale-mode-2026-11-02.md); the places below that called it untested now say what it found.

Lane: arc-update (target M1), bead pool-ttdzrk.13 under pool-ttdzrk. Work directory `work/arc-update/`:
`FORMAL.md` (theory v1 frozen 04:54:16Z before the co-prover's answer; additions E1-E12 appended), `PREREG.md` (Part 1
hashed 05:29:26Z, Amendments 1-6 by 08:25:38Z; `prereg_hashes.txt`), `coprover/` (gpt-6-astra rounds 1-4, `PROVENANCE.md`),
`review/` (the GPT-6 review),
`notes/` (READING.md: prior-lane facts with labels; ARC_SOURCES.md: 93 verbatim ARC quotes re-fetched today;
CHECK1.md; QUESTIONS.md), `lib/au.py` with `tests/`, `scratch/` (runs and scorers), `runs/`, `results/`, `logs/`
(deviations.txt). Labels: PROVEN, PROVEN UNDER <hypothesis>, NUMERICAL, MEASURED, HEURISTIC, CONJECTURE.
Provenance: [C] me, [A] gpt-6-astra, [prior: lane] an earlier lane's report.

## Summary for Kelvin

**What this is: a registered benchmark of ARC's own code on the contest's Phase-1 shape and along depth growing with
width, plus one framing: an estimator must escape the error-cost ladder below, or it loses to sampling. It is a
restricted negative (the methods tested), not a proof, and not by itself a reason for ARC to update beyond what they
already concede ("This depth scaling is worse than Monte Carlo sampling", their App. D; methods "beyond basic cumulant
propagation" are expected, Hilton). What is new is quantitative: how large the gap is at 256 x 32 and how it scales.**

1. **ARC's own code at 256 x 32** (mlp_kprop, ARC's flop counter; five fresh nets; registered). The best order-2 variant
   (k2a, 452 passes) beats RQMC at its own cost on 4 of 5 nets (median 2.0x; net 203: 0.85x), at precision ~1e-3, and
   RQMC catches up by 400-1800 passes. Order 3 (k3f, 4.9e4 passes) loses to RQMC by 5-30x at its own cost. At the
   contest budget (6.5e4 passes) RQMC is 7-40x (median 22x) more accurate than the best affordable ARC variant.
2. **The ladder along L = n/8** (n = 32, 64, 128, 256; registered). Each order's error stays within a factor ~2-3 as n
   grows 8-fold (order 2: ~1e-3 throughout); going from order 2 to 3 divides it by 4-8 (n/L = 8) and multiplies the
   cost by ~n/4 to ~3n. The measured error-cost exponent between orders 2 and 3 is gamma = 0.52, 0.39, 0.23, 0.30
   (ARC's own law predicts log(n/L)/log n = 0.60, 0.50, 0.43, 0.375; sampling has 1). The framing (section 2.5): if the
   law holds as a lower bound, uniformly over orders, climbing the ladder cannot reduce the error faster than C^-gamma
   with gamma -> 0 at L = alpha n. A method must escape this ladder, with a different error-cost law, or lose to
   sampling at depth growing with width; the measurements say ARC's hierarchy is on it for K <= 4.
3. **Where the error is made** (registered attribution). About 60% of order 2's final error (44-64% per net) is already
   in its state at layer 8, and it is accumulated error: the moment-matched Gaussian of the true layer law, put in at a
   single cut, loses only 1.3-2.4e-4, against the recursion's 1.2e-3.
4. **The radius-corrected closure, our strongest lead, errs by about L/n^2.** A closure that gets the amplitude
   (radius) law exactly right, run as a finite-sample recursion (2^19 samples of its own state per layer), errs by
   1.9e-3, 9.2e-4, 4.9e-4 at n = 64, 128, 256, within 6% of L/n^2 (post hoc observation; convergence in sample and
   batch size untested), 2-3x below the Gaussian closure at n >= 128. As run, it costs >= 2^19 passes per replicate,
   at least ~255x what MC needs to reach its error. No deductive version was built: one would beat MC at that error
   only if it cost less than ~5 order-2 steps per layer
   (break-even 2.1e3 passes at 256), RQMC only below ~3.5. The L/n^2 law is consistent with incoherent one-step
   angular defects of order n^-2 (co-prover round 4); that is a decomposition whose hypothesis is close to its
   conclusion, not a proof.
5. **Where the tested methods' errors are priced** (a pricing identity, A2, exact at every width): the future reads a
   model defect through nonnegative weights on the cloud's radially normalised moments in random orientations; at
   256 x 32 a 31-layer future puts weight 0.95 on the radius (measured), and at infinite width reads the rest at mean
   degree ~40. Your hypothesis: frames scrambled (PROVEN); compressed codes not scrambled layer to layer (MEASURED,
   positive controls pending); bits scrambled relative to where the law is computable (section 5).
6. **Untested and open.** A public counter-claim (AIcrowd forum topic 18298) says a "scale-mode" correction restores
   convergence with depth growing with width. A separate lane has since tested it at orders 2 and 3
   (outbox/scale-mode-2026-11-02.md): a constant factor in a window (2.5x RQMC at its own cost at 256 x 32, closing at
   ~2.3x that cost), a two-level exponent equal to ARC's at 256 x 32, and neither registered escape criterion met. Its
   higher orders, implicit representations, future-aware projections and other couplings are untested.

## 1. What ARC claims, and what "matching sampling" means (sources re-verified today, notes/ARC_SOURCES.md)

- The estimand is E_x M_theta(x) for every output neuron of one fixed random He-ReLU net (not a scalar readout); the
  error is the MSE over all output neurons, averaged over weights and divided by the variance of one forward pass, so
  N Monte Carlo samples score exactly 1/N (Wu et al. 2026, arXiv:2605.05179 v2, Sec. 2.2, 6.1). Our nMSE,
  |est - Y|^2 / tr Var, is the same quantity up to the per-neuron variance weighting.
- Theorems 5.1-5.2 (polynomial activations, fixed depth): error O(eps^2) in time O(n^2/eps^2) (basic) and O(n/eps^2)
  (factorised, eps^2 = O(n^-2)). Conjecture (Sec. 5): "mean squared error at most c_K (L/n)^K"; runtime c'_K L n^{K+1}
  (basic), c''_K L^2 n^K (factorised).
- Depth (App. D): "This depth scaling is worse than Monte Carlo sampling, whose error does not increase with depth. We
  believe that a sample-free algorithm can be developed whose error also does not increase with depth". Open problems
  (Sec. 7.3): matching sampling "at large depth" and "for narrow networks"; Hilton's blog: "with the depth growing
  linearly in the width". The paper's experiments stop at 12 hidden layers.
- The matching sampling principle ("Competing with sampling", Nov 2025): an estimator in time O((1/eps^2) Time(M)) with
  error <= eps^2 E Var, for all eps, that is "mechanistic" (no formal definition; heuristic: "avoids any random or
  pseudorandom sampling").
- The contest's Phase 1 is exactly 256 x 32 with 2.72e11 flops (6.5e4 forward passes). Organiser: "The best-performing
  methods in Phase 1 tended to be based on variations of sampling." Hilton (23 June): "Our existing algorithms scale
  poorly with depth". Christiano (3 June): "We think there probably do exist adequate algorithms for analyzing random
  networks and the reason we haven't found them is because so little research has addressed the question head on."
  No official Phase 1 or 2 results are public (Phase 2 winners ~15 November). The contest scores the raw MSE over all
  final-layer neurons, not a variance-normalised one. The public Phase 1 board's best entry has final-layer MSE
  3.6e-8 at about half the budget; with per-neuron variance V/n = 0.02-0.12 on our five nets that is nMSE ~3e-7-2e-6,
  ~10-50x below plain MC at the full budget (methods not public; public board, not the prize ranking). A participant's
  Phase 2 post (1024 x 16, L/n = 1/64) reports a never-sampling estimator 84x below full-budget MC (forum topic 18282; UNVERIFIED).
  These board numbers are approximate comparisons only: different nets, and the official score also multiplies by a
  compute-utilisation factor. Both fit the picture below: deduction wins when L/n is small and loses to (enhanced)
  sampling at L/n = 1/8. ARC also expect improvements beyond basic cumulant propagation, and a participant (forum topic
  18298) claims a "scale-mode" correction that restores convergence with depth growing with width, tested from
  64 x 16 to 1024 x 256. A separate lane has since tested it at orders 2 and 3 (outbox/scale-mode-2026-11-02.md): a
  constant factor in a window, no escape from the ladder; its higher orders belong to the open routes of section 7.

## 2. Part A: pricing identities, in a computational model that contains ARC's estimators

### 2.1 The model
A layer-wise estimator keeps a state S_l = U_l(S_{l-1}, W_l); it is forward-only at a cut if its state there depends
only on the weights before it. A deduction-projection estimator (ARC's "expectation machine" form) also has a
reconstruction of each state as a signed measure Q_l (or a linear functional on a class containing the future maps),
with Q_0 the input law and output int h dQ_L. An order-K moment closure is one whose defects
Delta_l = Q_l - Phi_l Q_{l-1} annihilate polynomials of degree <= K.

Contained in the state-update model: ARC's cumulant propagation of every order and variant, exact Gaussian closures,
Gaussian mixtures and gate splits, particle methods (so sampling is in the class too), layer-wise code propagation,
cell and region merging, breadth-first branch-and-bound. Covered by the signed-measure pricing results below (A2-A5),
which need a reconstruction Q_l: ARC's K = 2 'simple' (its last step is the exact Gaussian ReLU mean of its state,
checked to 2.2e-16, notes/CHECK1.md, up to its 1e-10 variance floor), the exact Gaussian recursion, Gram-Charlier
closures run with the exact metric, and the radial closures of section 4.4. NOT covered: ARC's default averaged-metric
'augment' and K >= 3 implementations, whose corrections break positive homogeneity, so no fixed signed measure
reproduces them [A r2]; for those only astra's last-row certificate applies (FORMAL E7(c)), and it does not by itself
give a large error. Not contained at all: computations that are not layer-wise (whole-network symbolic or analytic
methods), and estimators fed advice.
Why a restriction is unavoidable [prior: cut-rank-ext, bits-bounds; A r1]: the weights are an exact description of
n^2 L numbers, a future-aware state can be the answer itself (n numbers), and an unrestricted running-time lower bound
for this explicit problem would be a super-linear time lower bound, for which no technique exists.

### 2.2 Pricing identities A1-A6: what a forward layer-wise estimator pays (FORMAL A1-A6 as corrected in E7)
Setting: width n, depth L, cut at layer l with t = L - l layers to go; P_l the true law of h_l; g_l the map from layer l
to the output; the future's weights are independent of everything an estimator has computed at the cut ("forward").

A1, telescoping (PROVEN [prior: bits-bounds]). est - Y = sum_l <g_l, Delta_l>.

A2, exact pricing (PROVEN [C], every finite width; proof step corrected and strengthened by [A]). For any past-measurable
defect Delta at layer l,
  E_future |<g_l, Delta>|^2 = sum_{d >= 0} b_d^{(n,t)} || int |h| (h/|h|)^{tensor d} dDelta(h) ||_F^2,
with b_d >= 0, sum_d b_d = 1 exactly, and b_0 >= 1/pi, nondecreasing in t [A]. The proof is Mehler's formula for the
fresh Gaussian layer plus positive homogeneity. So the future reads a defect only through the radially normalised
moment tensors M_d, in a rotation-invariant way, with nonnegative weights. No forward method can know which directions
of its cloud will matter (the same statement as the relevance lane's Theorem 1, in a quantitative form). A2 prices a
given model defect; it does not say that every affordable representation has a large one.

A3, what moment matching buys (PROVEN, corrected by [A]). Matching moments through order K cancels M_1, and on a common
sphere every M_d with d <= K; it does not in general cancel the radius feature M_0 or any M_d with d > K. (It can: a
signed measure exists that is invisible to every homogeneous query [A]; that is not what closures produce.)

A4, random probes (PROVEN [C]; scope narrowed by [A]). The next layer's neurons read the cloud's order-r cumulant tensor
T through independent Gaussian rank-one probes: kappa_r(z_i) = <w_i^{tensor r}, T>. For any past-measurable
reconstruction T^, E_w <w^{tensor r}, T - T^>^2 >= r! (2/n)^r ||T - T^||_F^2, and if T^ has mode-1 flattening rank
<= k (any CP-rank-k or Tucker-rank-k compression), ||T - T^||_F^2 >= the tail of T's flattening spectrum beyond k.
This prices approximation, not storage: a tensor with a cheap implicit description is not excluded [A].

A5, terminal-model bound (PROVEN [C]; needs no assumption about cancellation between layers [A]). If the last step is
exact for the method's model Q_{L-1} (est = int relu(W_L h) dQ_{L-1}), then
  E_{W_L} |est - Y|^2 = sum_d b_d^{(n,1)} || M_d(Q_{L-1} - P_{L-1}) ||^2,
b^{(1)} = (1/pi, 1/2, 1/(2 pi), 0, 1/(24 pi), 0, 1/(80 pi), ...). This covers ARC's K = 2 'simple' exactly (checked to
2.2e-16 up to its 1e-10 variance floor) and the exact Gaussian recursion. It does not cover ARC's 'augment' or K >= 3 as
run (the averaged metric breaks the reconstruction [A]); for those, astra's hypothesis-free last-row certificate
applies (FORMAL E7(c)). More generally, at any cut, (E|est - Y|^2)^{1/2} >= ||Q_l - P_l||_{k_t} - (suffix error)^{1/2}.
A5 says where the error is; it does not by itself say it is large (on a ray-like cloud the Gaussian bottleneck can be
zero [A]). That is a measurement (section 4, P6-P8).

### 2.3 What a deep future reads (FORMAL E6 as corrected in E10.8; NUMERICAL, INFINITE WIDTH; exact b_0, b_1, b_{>=2})
| layers to go t | b_0 (radius) | b_1 (mean) | b_{>=2} (higher M_d) | degree holding 50% / 90% of b_{>=2} | mean degree of b_{>=2} (exact) |
|---|---|---|---|---|---|
| 1 | 0.318 | 0.500 | 0.182 | 2 / 4 | 2.75 |
| 4 | 0.681 | 0.142 | 0.178 | 3 / 8 | 4.84 |
| 8 | 0.834 | 0.050 | 0.116 | 4 / 14 | 8.22 |
| 16 | 0.930 | 0.013 | 0.057 | 6 / 30 | 17.2 |
| 31 | 0.973 | 0.003 | 0.024 | 10 / 74 | 42.3 |
The mean-degree column is exact: sum_d d b_d = kappa_t'(1) = 1, so the mean degree of the b_{>=2} mass is
(1 - b_1)/(1 - b_0 - b_1) [A r3; my truncated FFT had understated it]. At infinite width a deep future barely reads
the cloud's mean, reads its radius feature with weight near one, and reads the rest of its shape through radially
normalised moments of high degree: half the b_{>=2} mass sits above degree 10 and the mean degree is 42 at t = 31.
There is no cutoff: as t grows, 1 - b_0 ~ 9 pi^2/(2 t^2) and b_1 = Theta(t^{-3}), so the mean degree grows like
2 t^2/(9 pi^2) (PROVEN asymptotics [A r3], checked). A cumulant method of order K matches moments up to K = 2-4.
At finite width only b_0 was measured (section 3: 0.948 at t = 31 against 0.973), and the finite-width b_d are known
not to follow the infinite-width table uniformly at t proportional to n (PROVEN [A r3]); so the degree numbers are
infinite-width statements about ARC's instance, not measurements of it. These are the weights of the features; how
much each feature contributes to an actual error, b_d |M_d(Delta)|^2, depends on the defect.

### 2.4 Separation at depth proportional to width (conditional results; hypotheses as stated)
- [A, astra eq. 12/34] An order-K method with annealed normalised error >= b n^{-p} at cost Theta(n^{K-1}) passes has
  gain over plain MC O(n^{p-K+1}) and over the multilevel shared-trajectory sampler O(log^2 n n^{p-K}); a polynomial
  advantage over the latter needs p > K.
- [prior: bits-bounds r3; A r1-r2] Exact Gaussian recursion: p <= 1 under (H_R); ARC's K = 2 'simple': liminf n R >=
  0.0069 e^{-5 alpha/4} under (H_{R,s,tau}) (both radius-sign hypotheses are CONJECTURES, checkable layer by layer).
  So at L = alpha n both lose to the multilevel sampler by a factor growing like n / log^2 n
  (PROVEN UNDER those hypotheses and the sampler's layerwise cost hypotheses [prior: unrestricted]).
- Unconditional: one exact Gaussianisation immediately after the first layer, followed by the exact network to any
  depth, leaves liminf n R >= 0.0064, with no attenuation in depth (PROVEN [A r2, eq. 26]; asymptotic, annealed
  normalisation). This is a statement about that cut, not about a replacement at a later layer.
- Higher orders: only conditional radial exponents (p_rad = K - 1 for 'simple' even K, K for odd K with the extra trace,
  K + 1 for augmented even K) [A r2]; three gaps (projection error, suffix propagation, accumulation) block a proof for
  the full recursion. Augmented even K is not ruled out by the radius mechanism alone.
- Radius-corrected closures, the strongest lead (FORMAL E10, E12 [A r3, r4]; constants checked). Three different
  reconstructions are involved and must not be mixed: astra's implementable one (Gaussian direction, independent
  lognormal radius; update cost O(J n^3) per layer with J quadrature nodes, at fixed accuracy only), and the monotone-
  transport one that was measured (section 4.4). Matching the true radius law removes the radius feature M_0 exactly;
  the lognormal fit leaves an O(n^{-3/2}) radius defect (PROVEN UNDER radial extensivity and Taylor-remainder control).
  What remains is angular, and the mean term b_1 |M_1|^2 (transports move the mean). At the first layer, radius
  matching cannot remove the coordinates' skewness gamma_3 = 1/sqrt(pi) + 2/pi^{3/2} = 0.92: PROVEN UNDER a
  first-layer shell hypothesis, b_3^{(n,t)} >= (e^{-5 beta/4} + o(1))/(3 pi^2 n) at t = beta n, and the
  radius-corrected first cut has liminf n^2 E|e|^2 >= e^{-5 beta/4} gamma_3^2/(3 pi^2) in absolute squared error
  (normalised error needs division by E V). This is an asymptotic lower bound, not a Theta(n^-2) law.
- The full radial recursion (FORMAL E12-E13 [A r4]). Its error is the sum of its OWN one-step defects (not the oracle
  single-cut defects), each priced by A2 (PROVEN). Under cap, remainder and local-discrepancy hypotheses a one-step
  defect is O(n^-2) when the priced tangent discrepancy is O(n^-1/2) (PROVEN UNDER those hypotheses). The step from
  there to a total of Theta(L/n^2) assumes sum_l D_l = Theta(L/n^2) plus two-sided Gram bounds, which is close to
  assuming the conclusion; so it is a consistency decomposition of the measured L/n^2 law (section 4.4 (c)), not a
  proof of it. If it holds, a deductive version at O(J n^3) per layer has gain Theta(1/J) over MC and O(log^2 n/(J n))
  over the multilevel sampler; J is not known.
- The next order in the radial setting (PROVEN [A r4]): tr M_3 = M_1, so the trace sector adds nothing once the mean is
  matched; the trace sector holds only 3/(n+2) of the diagonal-skewness energy; and mixed cubic cumulants carry Theta(n)
  energy at the first layer, so diagonal skewness is not the whole correction. A matrix-cost correction with o(n^-2)
  one-step defect would need a low-rank innovation representation maintained through gating (CONJECTURE that one
  exists). If further orders follow ARC's ladder (cost x n, error / q), the precision exponent is log q/log n = 0.25-
  0.375 at 256 for q = 4-8 (PROVEN UNDER that transfer): the radius correction improves the starting error, not the
  exponent.
- Why splitting off the radius cannot finish the job (PROVEN [A r3]): the next radius is R' = R |relu(W U)|, a
  gate-dependent function of the direction U, so the radius law cannot be propagated from its marginal; the exact
  object every positively homogeneous query needs is the radius-weighted direction measure nu(du) = E[R | U = u] P(du),
  which propagates by a gain-weighted push-forward on the sphere: the same law-propagation problem, one dimension down.
- At 256 x 32 (alpha = 1/8) the multilevel sampler's gain is ~1 [prior]; the comparators there are RQMC (~2x MC) and
  the Kerdock design (2.4x MC, [prior: arc-noisy]).

### 2.5 ARC's scaling law, if tight, as an error-cost exponent (FORMAL E9 with astra r3's correction; P13-P14)
ARC conjecture an order-K error of at most c_K (L/n)^K at cost ~L n^{K+1} flops (basic), i.e. ~n^{K-1} forward passes.
Suppose the law is tight, uniformly over the orders used: e_K >= a c_* (L/n)^K and C_K >= d_* n^{K-1}, with a, c_*,
d_* independent of K. At budget C the affordable orders satisfy K <= 1 + log(C/d_*)/log n, so the best affordable
error obeys, for n > 1 and 0 < L/n < 1,
  e_best(C) >= a c_* (L/n) (C/d_*)^{-gamma},   gamma = log(n/L)/log n = 1 - log L/log n
(PROVEN UNDER the uniform law [A r3, eq. 60]; the same exponent as my E9 arithmetic). This is a lower bound: the error
cannot fall faster than C^-gamma. At fixed depth gamma -> 1 as the width grows: that is ARC's theorem regime, where the
hierarchy matches sampling. At L = alpha n, gamma = log(1/alpha)/log n -> 0: each extra order costs a factor n and buys
only a factor 1/alpha, so the error can fall with compute only ever more slowly than sampling's 1/C; at 256 x 32,
gamma = 0.375. For every L > 1 at finite width gamma < 1,
so the ratio of the hierarchy's error to MC's grows at least like C^{1 - gamma}: the hierarchy cannot match sampling at
all precisions, even if its error over all orders has no positive floor. (My earlier wording, "a precision floor
min_K c_K alpha^K", was wrong: c_K >= c_* > 0 does not give a positive infimum [A r3]. The conclusion does not need
one.) The caution matters: ARC state the law as an upper bound with order-dependent constants; the argument needs a
lower bound on error and cost, uniformly in K (and, for statements about growing width, uniformly in n). Measurements
at orders 2-4 cannot establish that uniformity. So this is not "ARC's law implies the negative" but "if ARC's law is
tight, uniformly, the negative follows". At fixed order the gain over MC falls like n^{-(K-1)} under the same law. What
we can check is the law's form along L = n/8 for K <= 4: the per-order ratio and the exponent between orders 2 and 3
(P13-P14, section 4); 0.40 on ARC's own 256 x 32 medians [prior: arc-noisy].

### 2.6 Resolving and future-aware estimators
- Resolving (cells, regions, codes, branch-and-bound): PROVEN in class [prior: bits-bounds Thm 7.1/7.3] that a
  forward-only cell sweep needs at least the rate-distortion function of the cloud under the future-averaged merge
  distortion; model-based estimates at 256 x 32: 224, 112, 45 bits at l = 4, 8, 16 for 10% distortion. These are
  representation diagnostics, not a mean-estimation separation: the bits-bounds report itself says the geometric
  hypotheses are not established globally for these nets and that its theorem separates representations, and with
  exact conditional means every partition, even a one-cell one, gives the exact mean (the tower property of section 5).
  So no comparison of cell counts with sample counts is drawn here. Width 8 x 256 [prior: nd-exact draft]: a certified
  deterministic code loses to RQMC by 23-106x (108-910x against RQMC with the exact tail); certifiable cells
  ~1e16-1e18; exact integration wins only at widths 2-3 (measured comparisons of implemented methods).
- Future-aware: no information obstruction exists (the state can be the quantised answer) [prior: bit-accounting; A];
  the limit is computation. A future-aware projection that preserves a set S of test statistics pays
  sum_l <g_l - proj_S g_l, Delta_l> (PROVEN UNDER exact preservation [A]); the value function is far from any small
  linear span (mean Hermite degree 21, 7.7, 4.1, 1.9 at l = 0, 4, 8, 16 [prior: relevance]). I do not have a lower bound
  for future-aware layer-wise methods; my v1 claim that future-awareness "changes which statistics are carried, not how
  many" is withdrawn [A].

## 3. Part B: the parameters at 256 x 32, with their status

"Certified" here means: PROVEN, or NUMERICAL with an error check, or MEASURED with a registered design and stated
resolution. Items that are only point estimates or uncertified learned codes are marked so.

| parameter (used where) | value at 256 x 32 | status | source |
|---|---|---|---|
| future price spectrum b_d^{(t)} (A2, section 2.3) | infinite width: b_0 = 0.32, 0.68, 0.83, 0.93, 0.97; b_1 = 0.50, 0.14, 0.05, 0.013, 0.003; mean degree of b_{>=2} 2.75, 4.84, 8.22, 17.2, 42.3 at t = 1, 4, 8, 16, 31 | NUMERICAL (exact b_0, b_1 and mean degree; FFT quantiles to 1e-4); b_0 >= 1/pi, nondecreasing in t, PROVEN at every width [A] | this lane, scratch/kernel_spectrum.py |
| finite-width radius weight b_0^{(256,t)} | 0.318, 0.677, 0.824, 0.911, 0.938, 0.948 at t = 1, 4, 8, 16, 24, 31 (SE <= 0.0004): the non-radial weight at t = 31 is 0.052, twice the infinite-width 0.027 | NUMERICAL (4000 random futures; exploratory) | this lane, scratch/finite_b0.py, FORMAL E11 |
| E V (normalisation) | 0.4995 < E V < 20.14 | PROVEN [A] | astra r1 |
| angular cloud participation ratio | 1.04 n/l + 1.38 (33, 18, 15, 13 at l = 8, 16, 24, 28) | MEASURED (post hoc fit, 64 nets) | bits-bounds |
| input-cloud PR, vertex law | 166, 62, 18.8, 12.1 at l = 1, 4, 16, 32 | NUMERICAL (exact replica chain) | gram-chain |
| full-cloud covariance PR and third-cumulant PR (A4, P9) | net 200: PR3 = 115, 66, 23, 4.0, 4.1, 1.6, 1.1 and covariance PR = 165, 106, 57, 21, 11.5, 4.1, 1.9 at l = 1, 2, 4, 8, 16, 24, 31; held-out top-32 capture 0.08 at l = 1, 0.95 at l = 16; nets 201-204: PR3 114-116, 61-67, 23-28 at l = 1, 2, 4 and 4.0-10.6 at l = 8 | MEASURED (registered P9; U-statistics, N = 2^16; planted-rank controls recover PR 7.7 and 32.0) | this lane |
| future's linearity scale; roughness ratio rho_l | 2.45/t rad; rho = 7.2, 3.0, 1.1, 0.46 at l = 8, 16, 24, 28 | PROVEN (mean field) / MEASURED | bits-bounds |
| variance made after layer l | share ~8.3/(l + 9) | mean field + PROVEN martingale identity | bits-bounds, input-bits |
| input-variance split amplitude / direction / radius | 0.49 / 0.47 / 0.034 | MEASURED (4 nets, medians) | input-bits |
| mean Hermite degree of the value function | 21, 7.7, 4.1, 1.9, 1.3, 1.04 at l = 0, 4, 8, 16, 24, 31 | MEASURED (registered; per-net factor R 0.6-2.4) | relevance |
| annealed mean degree of the input map | n / E_W V (~19) | PROVEN [prior: input-bits, A] | input-bits |
| projections for 99% of the value function's variance | N(l, 0.01) = 256, 192, 128, 80, 48 at l = 0, 4, 8, 16, 24 | MEASURED point estimates (upper bounds on the optimum for exhibited bases) | relevance |
| gates switched per sample vs the mean path | 5-9% | MEASURED | relevance |
| local-code bits for 10% output error (earlier fixed-budget codes; superseded by the next row) | > 256 (l <= 4), 135 (l = 8), 43 (l = 16), ~25 (l >= 24); output 19-39 | MEASURED, NOT CERTIFIED | bit-accounting v1 |
| bits-vs-accuracy curves, whole 256-dim output vector (ARC's target) | net 4000, local code, 10% / 1% nMSE: l = 31: 15 / 90 bits; l = 16: 32 / 186; l = 0: 575-642 / 1084-1126 | MEASURED; per main 07:00Z the early (l = 0) counts are not yet certified; results file not yet in bit-accounting-arc/results | bit-accounting-arc |
| consecutive-layer relevant-coordinate maps | linear R^2 0.82-0.99 | MEASURED (registered BA.1 missed narrowly) | bit-accounting B-ARC |
| single-cut Gaussian defect Gamma*_2(l) (A5, P7) | 6.7e-5, 1.6e-4, 2.4e-4, 2.2e-4, 1.8e-4, 1.3e-4, 3.3e-5 at l = 1, 2, 4, 8, 16, 24, 31 (medians, 5 nets); with the true radius law 8e-6, 3.5e-5, 3.2e-5, 7.9e-5, 1.4e-4, 1.4e-4, 5.7e-5 | MEASURED (registered; 2^19 pushes; push noise debiased by independent halves, projection-estimation error not removed; small values poorly resolved, section 4.3) | this lane |
| error attribution of ARC K = 2 and the exact Gaussian recursion (A6, P6) | past defect b_l^2 (k2) = 1.0e-4, 1.8e-4, 5.0e-4, 7.8e-4, 9.8e-4, 1.2e-3 at l = 1, 2, 4, 8, 16, 24 against a total of 1.2e-3: 44-64% of the error is made by layer 8; no strong cancellation | MEASURED (registered, 5 of 5 nets) | this lane |
| ARC hierarchy errors, costs, gains (C1, P1-P5, P11-P12) | 256 x 32 medians: k1 2.6e-2 at 1 pass; k2 1.2e-3 at 393; k2a 8.4e-4 at 452; k3f 2.0e-4 at 4.9e4 (gains over MC 34, 2.1, 2.6, 0.10); ladder in section 4.1 | MEASURED (registered; ARC's flop counter) | this lane |
| RQMC gain over MC at 256 x 32 | 1.24, 1.48, 1.74, 2.19, 2.10 (medians over 5 nets) at N = 2^8, 2^10, 2^12, 2^14, 2^16; range 0.74-2.64 | MEASURED (8 scrambles per size) | this lane |
| Kerdock design | 0.40/N (2.4x MC) | MEASURED (sigma = 1e-3, 24 nets) | arc-noisy |
| multilevel shared-trajectory sampler, ideal | 0.99-1.06 of radius-exact MC | MEASURED (ideal, setup excluded) | unrestricted, bits-bounds |
| radius-sign hypotheses H_R, H_{R,s,tau} (2.4) | fresh-layer mean radius defect on the states ARC k2 reaches (net 200): +0.036, +0.020, +0.015, +0.006, +0.002 at l = 0, 1, 2, 4, 8 (SE 0.001-0.003), indistinguishable from 0 at l = 16-24; l = 0 matches the proven 9/(16 sqrt n) | CONJECTURE; one exploratory test on one net (FORMAL E8) did not contradict it; not a proof | this lane, scratch/hr_test.py |

## 4. Part C: pre-registered predictions and results

Design (PREREG.md; every prediction was hashed before the evaluation runs it scores and code was frozen by sha256;
several amendments were written after part of the evaluation data had been seen, as each header states: P13 at n = 32
and 64 and everything about net 200 in Amendments 2-4 are therefore descriptive, not fresh predictions): ARC's mlp_kprop
(k1, k2 = 'simple', k2a = 'augment', k3 basic, k3f factorised, k3fa, k4, k4f; ARC's own flop counter) along L = n/8 at
n = 32, 64, 128 (eight fresh nets each, 2^24-path truths) and at 256 x 32 on the five nets with arc-noisy's
noiseless 2^26-path truths; scrambled-Sobol RQMC (8 replicates per size); the exact Gaussian recursion; per-cut
attribution (FORMAL A6) with the oracle Gaussian and oracle radial-Gaussian single cuts; third-cumulant structure
(U-statistics); and two later amendments on the strongest lead: the width scaling of its single-cut floor
(Amendment 4) and its full recursion, run as a finite-sample stochastic recursion (Amendments 5-6). Costs in forward passes
(2 n^2 L flops each); nMSE = |est - Y|^2 / tr Var, so plain MC with N samples scores 1/N. Results are medians of per-net
nMSE; the theory's annealed ratio E_W|est - Y|^2 / E_W V differs in principle. For the recursions' totals at
256 x 32 they agree closely (ARC k2: annealed 1.23e-3, median 1.22e-3; exact Gaussian 1.22e-3 vs 1.25e-3), but not for
every statistic (the radius-corrected first cut's width exponent is 2.5 by medians and 1.8 annealed; section 4.4).

<!-- scorecard start -->
Scorecard (results/scorecard.md, generated by scratch/scorecard.py from the hashed scorers' outputs; prediction wording abbreviated, PREREG.md has the exact text):

| id | prediction (PREREG wording abbreviated) | measured | result |
|---|---|---|---|
| P1a | G_3/G_2 <= 2/L at n = 32, 64, 128, 256 | 32: 0.35; 64: 0.131; 128: 0.0786; 256: 0.0384 | HOLDS |
| P1b | G_4/G_3 <= 2/L at n = 64, 128 | 64: 0.0709; 128: 0.0166 | HOLDS |
| P2 | G_3 < 1 (n = 128, 256), G_4 < 1 (128); best order >= 3 G_RQMC < 0.5 (n >= 64) | G3_128: 0.31; G3_256: 0.101; G4_128: 0.00515; best_order_ge3_G_RQMC: 64: 0.182; 128: 0.116; 256: 0.0571 | HOLDS |
| P3 | RQMC reaches the best K = 2 error at <= 4x its cost, >= 4 of 5 nets (per-net G_RQMC) | [2.1, 1.87, 4.58, 1.88, 1.96] | HOLDS |
| P4 | at 6.5e4 passes RQMC >= 10x below the best affordable ARC variant, 5 of 5 nets | [6.71, 39.8, 17.8, 34.1, 21.6] | FAILS |
| P5a | G_3 strictly decreases over n = 32..256 | [2.82, 0.629, 0.31, 0.101] | HOLDS |
| P5b | G_2(256) < G_2(32) | G2_32: 8.06; G2_256: 2.63 | HOLDS |
| P6a k2 | b_8^2 >= 0.3 total (ARC k2), >= 4 of 5 | 5 | HOLDS |
| P6a exact | b_8^2 >= 0.3 total (exact Gaussian), >= 4 of 5 | 5 | HOLDS |
| P6b k2 | total >= 0.4 (a^2 + b^2) at l = 4, 8, 16 (k2), >= 4 of 5 | 5 | HOLDS |
| P6b exact | same for the exact Gaussian recursion | 5 | HOLDS |
| P6c | abs(a_31^2) < 1e-5 (code check of CHECK-1), k2 | 2.24e-06 | HOLDS |
| P7 | median Gamma*_2 >= 5e-5 at l = 4, 8, 16 and >= 0.2 x k2 total at l = 16 | 4: 0.000238; 8: 0.000221; 16: 0.000182 | FAILS |
| P8 | median Gamma_rad/Gamma*_2 >= 0.5 at >= 2 of l = 8, 16, 24 (rule applied here to score_attr ratios) | 4: 0.135; 8: 0.492; 16: 0.762; 24: 1.01 | HOLDS |
| P9a | PR3 <= covariance PR at every layer, every net | None | HOLDS |
| P9b | PR3(1) >= 40 and PR3 <= 4 at l >= 16, every net (nets with rank runs) | pr_l1: [115, 116, 115, 114, 115]; pr_deep_max: 4.05 | FAILS |
| P9c | held-out top-32 capture <= 0.2 at l = 1 and >= 0.9 at l = 16, every net | cap32_l1: [0.0759, 0.074, 0.0748, 0.0782, 0.0789]; cap32_l16: [0.95, 0.979, 0.981, 0.987, 0.959] | HOLDS |
| P9d | planted rank 32 control gives PR3 in [24, 40] | 32 | HOLDS |
| P10 | exact Gaussian at 256 x 32: median nMSE in [2e-4, 2e-3], median G_RQMC <= 1.5 | nmse: [0.00112, 0.001, 0.00164, 0.00124, 0.00202]; G_RQMC: [0.357, 0.504, 0.616, 0.458, 0.307] | HOLDS |
| P11 | e*(256) >= 3 e*(32) and e*(128) >= e*(64) | 32: 3.41e-05; 64: 8.69e-05; 128: 9.46e-05; 256: 0.000201 | HOLDS |
| P12 | e_3/e_2 >= 0.2 at n = 64, 128, 256 | 64: 0.143; 128: 0.251; 256: 0.239 | FAILS |
| P13 | gamma_{2->3} < 1 at every n | 32: 0.522; 64: 0.385; 128: 0.232; 256: 0.305 | HOLDS |
| P14 | gamma_{2->3} in [0.5, 1.5] x log 8/log n at n = 128, 256 | 32: (0.52207562324953, 0.6); 64: (0.3850541926561814, 0.5); 128: (0.23198703532647683, 0.42857142857142855); 256: (0.30493802989412316, 0.375) | HOLDS |
| R1 | median Gamma_rad/Gamma*_2 <= 0.33 at >= 2 of l = 1, 2, 4 (nets 201-204) | 1: 0.076; 2: 0.188; 4: 0.136 | HOLDS |
| R2 | median Z = max_{l = 8, 16, 24} Gamma_rad >= 5e-5 (201-204) | 0.000205 | HOLDS |
| R3 | RQMC reaches Z at <= 2e4 passes on >= 3 of 4 nets | [2.88e+03, 2.76e+03, 4.68e+03, 3.06e+03] | HOLDS |
| P15 | p(G1) in [0.6, 1.4] | 0.949 | HOLDS |
| P16 | p(F1) >= p(G1) + 0.4 | p_F1: 2.48; p_G1: 0.949 | HOLDS |
| P17 | p(Zm) in [0.5, 1.5] | 0.673 | HOLDS |
| P18 | Wr(256)/Wr(64) <= 2 | 1.4 | HOLDS |
| P19 | Gaussian IPR within 1.5x of the deductive exact Gaussian recursion, every 256 net | [True] | PARTIAL: 1 of 5 nets (run stopped); the same Gaussian path in Amendment 6 is within 1.5x on 5 of 5 |
| P20 | median radial IPR >= 0.5 Zm(256) | median_radial_ipr: -0.000191; Zm256: 0.00014 | UNDETERMINED (run stopped, noise-dominated) |
| P21 | median radial IPR / Gaussian IPR <= 0.5 at 256 | -0.195 | UNDETERMINED (run stopped, noise-dominated) |
| P22 | p_full in [0.5, 1.5] over n = 64, 128, 256 | nan | UNDETERMINED (run stopped) |
| P23 | RQMC reaches the radial IPR error at <= 2e4 passes on >= 4 of 5 nets | [] | UNDETERMINED (run stopped) |
| P24 | rank-transport radial IPR: median replicate noise var at 256 <= 1e-4 (validity) | 2.15e-05 | HOLDS |
| P25 | median radial IPR2 >= 0.5 Zm(256) | median_radial_ipr: 0.000486; Zm256: 0.00014 | HOLDS |
| P26 | median radial IPR2 / Gaussian IPR2 <= 0.5 at 256 | 0.353 | HOLDS |
| P27 | RQMC reaches the radial IPR2 error at <= 2e4 passes on >= 4 of 5 nets | [1.05e+03, 933, 1.99e+03, 1.37e+03, 1.45e+03] | HOLDS |
| P28 | p_full (IPR2) in [0.5, 1.5] over n = 64, 128, 256 | 0.991 | HOLDS |
<!-- scorecard end -->

Reading the scorecard (final run 10:10Z; counts corrected 10:30Z; regenerated 13:45Z with the last rank runs). Of the
40 rows, 31 hold and 4 fail: P4 (the
contest-budget margin is 7-40x, not >= 10x on every net: 6.7x on net 200), P7 (second clause: the single-cut Gaussian
defect at l = 16 is 0.15 of k2's total, not >= 0.2), P9b (narrowly, on net 200 only: PR3 = 4.05 at l = 16 against
<= 4) and P12 (order 3 divides the error by 7 at n = 64, more than the >= 0.2 ratio allowed); P9a and P9c hold on all
five nets. Partial: P19 holds on the one net Amendment 5 ran at 256 x 32, and the
same check holds on 5 of 5 nets with Amendment 6's runs. Undetermined: P20-P23 (Amendment 5's radial full recursion at
256 x 32, stopped after one net because replicate noise swamped the signal; section 4.4 (c)). Two scoring corrections
made after the review (section 4.1) change no outcome. The registered statistics are the hashed scorers' outputs; the
state labels (partial, undetermined) were added after the review and change no number.

### 4.1 The ladder along L = n/8 (P1, P2, P5, P11-P14)
Best variant of each order (median over nets; error in nMSE, cost in passes, gain over MC and over RQMC at own cost;
results/tables.md T1 has every variant; RQMC gains at costs below 256 passes, K = 2 at n <= 128, use the RQMC curve
extrapolated below its measured range):
| n x L | K = 2 (k2a) | K = 3 | K = 4 | e_3/e_2 | gamma_{2->3} (ARC's law: log 8/log n) |
|---|---|---|---|---|---|
| 32 x 4 | 1.1e-3 at 117; G 7.6 / 1.7 | 1.4e-4 at 6.8e3 (k3fa); best G 2.8 / 0.64 (k3f) | 3.4e-5 at 3.4e4; G 0.86 / 0.18 | 0.12 | 0.52 (0.60) |
| 64 x 8 | 1.3e-3 at 164; G 4.8 / 1.9 | 1.8e-4 at 2.6e4 (k3fa); best G 0.63 / 0.18 | 8.7e-5 at 2.6e5; G 0.045 / 0.016 | 0.14 | 0.39 (0.50) |
| 128 x 16 | 1.0e-3 at 260; G 3.9 / 2.0 | 2.5e-4 at 1.0e5 (k3fa); best G 0.31 / 0.12 | 9.5e-5 at 2.1e6; G 0.005 / 0.002 | 0.25 | 0.23 (0.43) |
| 256 x 32 | 8.4e-4 at 452; G 2.6 / 2.0 | 2.0e-4 at 4.9e4 (k3f); G 0.10 / 0.06 | not run | 0.24 | 0.30 (0.375) |
- The ladder has the form of ARC's law for K <= 4 (P13, P14 hold; P13 at n = 32, 64 is descriptive, those results
  having been seen before Amendment 3). Along L = n/8 the error of each order stays within a factor ~2-3
  as the width grows 8-fold (K = 2: 1.1e-3 to 8.4e-4; K = 3: 1.4e-4 to 2.5e-4; K = 4: 3.4e-5 to 9.5e-5), as
  c_K (L/n)^K says at fixed L/n, while each order's cost grows like n^{K-1}. Going from order 2 to 3 divides the
  error by 4-8 (n/L = 8), from 3 to 4 by 2-4, and each step multiplies the cost by ~n/4 (factorised) to ~3n
  (factorised augmented).
  The measured exponent between orders 2 and 3 is 0.23-0.52, below 1 at every width and inside P14's registered band
  [0.5, 1.5] x log(n/L)/log n at n = 128 and 256 (a factor 1.85 below the law's value at n = 128, 1.2 below at 256). At fixed order the gain over MC falls with width like n^-0.54 (K = 2),
  n^-1.6 (K = 3) and n^-3.7 (K = 4, n <= 128), against the law's n^-1, n^-2, n^-3 (FORMAL E10.7).
- Consequences (P1, P2, P5 hold): the gain of order 3 over order 2 falls below 2/L at every width; order 3 loses to MC
  at n >= 128 and to RQMC by 5-20x at n >= 64; order 4 loses to MC by 20-200x at n >= 64; the best order-3 gain falls
  monotonically with width (2.8, 0.63, 0.31, 0.10), and K = 2's gain over MC falls from 8.1 to 2.6.
- The smallest error among the completed variants gets worse with width (P11 holds as registered): 3.4e-5, 8.7e-5,
  9.5e-5, 2.0e-4 at n = 32, 64, 128, 256. The comparison mixes order caps (K <= 4 below 256, K <= 3 at 256, where k4 was
  not run and k3fa ran on net 200 only, after the scoring); with a common cap K <= 3 it is 1.4e-4, 1.8e-4, 2.5e-4, 2.0e-4, not monotone.
- P12 fails: at n = 64 order 3 divides the error by 7 (e_3/e_2 = 0.14 < 0.2). My dev calibration had used k3f, not the
  best order-3 variant (k3fa), which is the error that P12 scores (noted in this report's draft before P12 was scored;
  now also in logs/deviations.txt).
- k3fa at 256 x 32 (registered for nets 200-201; it finished on net 200 at 12:54Z, after the scoring; net 201 was not
  run). At n <= 128 it had the lowest order-3 error at 4-8x k3f's cost. On net 200 it is worse than k3f in both error
  and cost: nMSE 9.5e-5 at 4.0e5 passes against k3f's 6.2e-5 at 4.9e4 (scratch/score_lad.py, run by main at 13:45Z;
  gain over RQMC at its own cost 0.016). So e_3, the measured exponent and the order-3 gains stand as scored.
- Scoring corrections (scratch/score_corrections.py, results/score_corrections.json; the hashed outputs are kept): the
  PREREG wording of G_K reads most naturally as the median over nets of each net's best variant, while the hashed
  scorer took the best variant's median. The literal reading gives G_2 = 8.8, 5.1, 4.6, 3.0 (instead of 8.1, 4.8, 3.9,
  2.6) and leaves P1a, P1b, P2, P5a and P5b holding. P3's wording is a crossover-cost ratio; the literal ratio is
  1.9, 1.7, 3.9, 1.6, 1.9 (5 of 5 <= 4; the hashed proxy gave 4 of 5).

### 4.2 256 x 32: who wins at which budget (P3, P4, P10)
A deterministic method with error e at cost C beats a sampler exactly on the budget window from C to the cost at which
the sampler reaches e (FORMAL E4). Per net (results/tables.md T2), with RQMC as the sampler:
| method | cost (passes) | nMSE (5 nets) | gain over RQMC at own cost | RQMC reaches its error at |
|---|---|---|---|---|
| ARC k1 (means only) | 1 | 0.8-4.3e-2 | 16-90 (extrapolated) | 18-102 passes (extrapolated) |
| ARC k2 ('simple') | 393 | 0.85-2.4e-3 | 0.83-2.2 | 330-740 |
| ARC k2a ('augment') | 452 | 0.48-2.3e-3 | 0.85-4.6 (median 2.0) | 400-1800 |
| exact Gaussian recursion | 1184 | 1.0-2.0e-3 | 0.31-0.62 | 390-730 |
| ARC k3f | 49097 | 0.6-2.8e-4 | 0.035-0.18 | 2300-6400 |
- The K = 2 window is real and narrow (P3 holds): RQMC reaches each net's best K = 2 error at 1.6-3.9x its cost (the
  gain at own cost is 1.9-4.6; the scorer used the gain, the stricter proxy), so deduction wins by about 2x below
  ~10^3 passes at precision ~1e-3. That uses the better of k2 and k2a on each net, a post hoc choice; a fixed variant
  does not win everywhere (k2a: median 2.0x, but 0.85x on net 203; k2: 0.83-2.2x). Means-only propagation (k1) wins by 16-90x, at precision 1-4e-2, but this
  comparison is an extrapolation: RQMC was measured from 256 points up, and the scorer extends its curve below that
  with slope -1 (PREREG registered extrapolation only above the largest size). Every order-3 run loses to RQMC by
  5-30x at its own cost (within the measured range).
- The exact Gaussian recursion does not beat RQMC at its own cost (P10 holds: nMSE 1.0-2.0e-3, gain 0.31-0.62), so
  exact Gaussian moments are no better than ARC's cheaper approximations (k2 has the same median error, 1.2e-3).
- At the contest budget (6.5e4 passes) RQMC's error is 7-40x (median 22x) below the best ARC variant affordable within
  it. P4 fails as registered (it needed >= 10x on all 5 nets): on net 200, k3f reached 6.2e-5 and RQMC 9e-6 (6.7x).
  The public Phase 1 board (section 1) suggests the best sampling-based entries do ~10-50x better than plain MC (an
  approximate comparison on different nets).

### 4.3 Where the error is made, and the structure there (P6-P9)
Per-cut attribution (A6: est - Y = a_l + b_l; b_l^2 = the method's layer-l model pushed through the true rest of the
net, a_l^2 = what the method's own suffix adds), medians over the five 256 x 32 nets (results/tables.md T3):
| cut l | 1 | 2 | 4 | 8 | 16 | 24 | 31 |
|---|---|---|---|---|---|---|---|
| ARC k2 past defect b_l^2 | 1.0e-4 | 1.8e-4 | 5.0e-4 | 7.8e-4 | 9.8e-4 | 1.2e-3 | 1.25e-3 |
| oracle Gaussian of the true law, single cut | 6.7e-5 | 1.6e-4 | 2.4e-4 | 2.2e-4 | 1.8e-4 | 1.3e-4 | 3.3e-5 |
| oracle radial-Gaussian, single cut | 8.1e-6 | 3.5e-5 | 3.2e-5 | 7.9e-5 | 1.4e-4 | 1.4e-4 | 5.7e-5 |
(ARC k2 total: median 1.2e-3; the exact Gaussian recursion's profile is the same within 10%.)
- The closure error is made early (P6 holds for both recursions on 5 of 5 nets): by layer 8 the k2 state already
  carries 44-64% of the final error (median 59%), and there is no strong cancellation between the past defect and the
  suffix error.
  At layer 4 the k2 state's defect is twice that of the moment-matched Gaussian of the true layer-4 law: the excess is
  accumulated error, not a limit of the Gaussian family. CHECK-1 holds on every net (|a_31|^2 <= 2.2e-6: ARC's last
  step is exact for its Gaussian state).
- The single-cut Gaussian defect is 1.3-2.4e-4 at l = 2-24 (P7's first clause holds); P7 fails on its second clause
  (at l = 16 it is 0.15 of k2's total, not >= 0.2): the moment-matched Gaussian of the true law at a cut loses only
  ~2e-4, while the recursion loses 1.2e-3 by accumulating.
- In the middle layers the defect is mostly angular (P8 holds): replacing the radius law by the true one (an
  intervention that also moves the mean and covariance, so not an orthogonal "radial share") lowers the single-cut
  Gaussian defect at l = 1, 2, 4 by a median 92%, 81%, 86% on the registered nets 201-204 (91%, 85%, 86% with net 200;
  per-net reductions 50-100%, one net's l = 1 value is negative within noise; by the ratio of medians at l = 2, 74% on
  the registered nets and 78% on all five), by
  half at l = 8, by a quarter at l = 16 and not at l = 24.
- Resolution: these single-cut numbers debias push noise by independent halves, but run_attr.py shares the estimated
  covariance, radius quantiles and pilot between halves, so projection-estimation error (of order 1/2^17 of the
  relevant variance) is not removed; and some small values are poorly resolved (the radial first-cut values at 256 x 32
  are 2.0e-5, 9.5e-6, 3.3e-6, -1.9e-6, 1.1e-6 per net, with half-difference noise 0.7-8.9e-6). Maxima over noisy cuts
  (R2, R3) are biased upward.
- Third-cumulant structure (P9; rank runs take ~50 min per net at one thread; nets 200-202 were scored first, 203-204
  at 13:45Z). The structure is high-dimensional where the error is made and collapses afterwards: the third-cumulant
  participation ratio is 114-116 at l = 1, 61-67 at l = 2, 23-28 at l = 4, 4-11 at l = 8 and 1-4 from l = 12 on
  (covariance PR: 165-166, 104-106, 56-60, 21-31, then 11-17 at l = 12 falling to 2-4 at l = 31); the held-out top-32
  multilinear capture is 0.07-0.08 at l = 1 and 0.95-0.99 at l = 16. P9a, P9c and the planted-rank control (P9d,
  PR 32.0 for a planted rank 32) hold on all five nets; P9b fails narrowly on net 200 only (PR3 = 4.05 at l = 16 against
  the registered <= 4; no uncertainty was attached to the threshold; at most 2.2 at l >= 16 on the other four). These are descriptive statistics of the true
  cloud (the PR numerator and capture ratios are biased ratios of unbiased estimates): a high flattening PR does not
  imply an expensive implicit representation, and a low one does not make accurate propagation cheap.

### 4.4 The strongest lead, measured the way ARC would judge it (R1-R3, P15-P28)
The lead: keep a Gaussian model of the cloud's direction but get its amplitude (radius) law exactly right. No
deductive version was built. What was measured are specified projections (oracle single cuts) and a specified
stochastic recursion; they are benchmarks for this family, not ceilings for every deductive closure: other members of
the family, other radius-direction couplings and other estimators are not covered. The measured reconstruction couples
the radius to the Gaussian norm by monotone transport; astra's implementable one (eq. 40, O(J n^3) per layer) uses an
independent lognormal radius, a different model, so its cost is not simply paired with these errors (FORMAL E12.5).
No deductive implementation cost is measured here. Where costs appear they are break-even costs: the compute at which MC or
RQMC reaches the same error, in units of ARC's order-2 step; a deductive version would have to cost less to win. The
number of quadrature nodes J it would need is not known (J = O(log^2(1/accuracy)) under analytic-strip hypotheses
[A r4]).

(a) Single cut at 256 x 32 (Amendment 2, R1-R3 hold; nets 201-204 scored, 200 seen before registering). Putting the
true radius law into the moment-matched Gaussian at one cut lowers its defect by a median 81-92% at l = 1-4, but not
in the middle layers: Z = max over l = 8, 16, 24 is 1.4-2.3e-4 (median 2.1e-4 on 201-204; a maximum over noisy cuts,
biased upward). If a closure's whole error were this single-cut value, RQMC would reach it at 2.8-4.7e3 passes (7-12
order-2 steps): a closure with this error at the cost of one order-2 step would beat RQMC by 8-17x there, and the
window would close at ~5% of the contest budget.

(b) Width scaling of the single cuts along L = n/8 (Amendment 4, P15-P18 hold; cuts at fixed l/n; medians over 8, 8, 8,
5 nets):
| n x L | G1: Gaussian, l = 1 | F1: radial, l = 1 | Zm: radial, mid cuts | Zm in sample-equivalents (1/Zm) | k2 cost (passes) | MC / RQMC cost to reach Zm, in units of the k2 cost |
|---|---|---|---|---|---|---|
| 32 x 4 | 6.3e-4 | 4.5e-4 | 5.8e-4 | 1.7e3 | 58 | 30 / 6.8 |
| 64 x 8 | 4.0e-4 | 1.0e-4 | 3.6e-4 | 2.8e3 | 106 | 27 / 8.6 |
| 128 x 16 | 1.8e-4 | 3.1e-5 | 2.6e-4 | 3.8e3 | 201 | 19 / 10.7 |
| 256 x 32 | 1.1e-4 | 3.3e-6 | 1.4e-4 | 7.1e3 | 393 | 18 / 12.1 |
Descriptive slopes over n = 64-256 (nMSE ~ n^-p; three widths, so the endpoints dominate; 95% bootstrap intervals over
nets, scratch/width_boot.py; with the annealed normalisation in brackets): p(G1) = 0.95 [0.70-1.08] (0.94): the plain
Gaussian's first-cut defect decays like the radius defect, ~1/n. p(F1) = 2.5 [1.2-3.5] (1.8): after radius
correction the first cut decays much faster; astra's asymptotic n^-2 lower bound has a constant ~200x below these
values at 256, so there is no conflict. p(Zm) = 0.67 [0.60-1.24] (0.77): the mid-layer value decays at most about
like 1/n; these widths cannot tell 2/3 from 1. The RQMC window at the mid-layer value grows 1.4x from n = 64 to 256
(P18's threshold of 2 allows growth up to about n^0.5); the MC window shrinks from 30x to 18x.

(c) The full recursion (Amendments 5 and 6). A single-cut value is not a bound on a recursion (its error is the sum of
its own one-step defects, which can accumulate or cancel; FORMAL E2, E12.1), so the radial closure itself was run
through all L layers, each layer's mean, covariance and radius law computed from 2^19 samples of the model's own state
(rank transport in batches of 2^14; two replicates). The replicate cross product removes replicate variance, not the
finite-sample bias of fitting and transport: these are finite-sample stochastic-recursion measurements, and their
convergence in the sample size and the batch size has not been established. The Gaussian projection run the same way
agrees with the deductive exact Gaussian recursion within 1.5x on 5 of 5 nets at 256 x 32 (P19's check, here on
Amendment 6's runs; P19 itself is partial, Amendment 5 having been stopped).
| n x L | Gaussian closure | radial closure | radial / Gaussian (median of per-net ratios) | ARC k2a | L/n^2 | cost as run, per replicate | break-even deductive cost vs MC / vs RQMC, in order-2 steps per layer |
|---|---|---|---|---|---|---|---|
| 64 x 8 | 2.1e-3 | 1.9e-3 | 0.80 (0.41-2.3) | 1.3e-3 | 2.0e-3 | >= 2^19 = 5.2e5 passes | 4.9 / 2.1 |
| 128 x 16 | 1.6e-3 | 9.2e-4 | 0.47 (0.18-2.9) | 1.0e-3 | 9.8e-4 | >= 5.2e5 passes | 5.4 / 3.5 |
| 256 x 32 | 1.4e-3 | 4.9e-4 | 0.35 (0.22-0.76) | 8.4e-4 | 4.9e-4 | >= 5.2e5 passes | 5.2 / 3.5 |
(Amendment 6, rank transport, all nets: 8, 8, 5; replicate noise 4e-6-4e-5. P24-P28 hold. Amendment 5's pilot-transport
runs at n = 64, 128 agree: radial 1.8e-3 and 9.6e-4. Break-even = (cost at which MC, or RQMC, reaches the radial
closure's error) / (ARC's order-2 cost); RQMC at n = 64 is extrapolated below 256 points.)
What the full recursion shows. The radius correction helps more as the width grows (radial/Gaussian 0.80, 0.47, 0.35),
the radial closure's error decays with p = 0.99 [0.75-1.19] (annealed 1.04) and sits within 6% of L/n^2 at every width
(MEASURED; the L/n^2 match is a post hoc observation, not registered), while the Gaussian closure decays only with
p = 0.30 [0.03-0.81] and ARC's k2a with ~0.3. ARC's k2a, which already carries a radial third-cumulant tensor and an
isotropic fourth cumulant (FORMAL E7), a cheap radius correction, is better than the radial closure at n = 64, equal
at 128 and 1.7x worse at 256. Whether L/n^2 is the closure's limit or partly finite-sample bias is untested (no
convergence study in the sample or batch size).
Cost, honestly: as run, the recursion pushes 2^19 samples through every layer, at least 5.2e5 passes per replicate,
so it costs at least ~255x what plain MC needs to reach its error at 256 x 32 (1/error = 2.1e3 passes; counting
network pushes only, per replicate). It is a measurement of the closure's error, not a method. A deductive version
(none was built) would beat MC at this error
only if it cost less than ~5 order-2 steps per layer at every width, and RQMC only below 2-3.5; astra's quadrature
update needs J = O(log^2(1/accuracy)) nodes with unknown constants (FORMAL E12.4), so whether that budget is met is
open. Consistency, not proof (FORMAL E12-E13 [A r4]): the error is the sum of the closure's own one-step defects; if
each is O(n^-2) after the radius is fixed and they add incoherently, the total is ~L/n^2, as measured, but that
assumption is close to the conclusion (a direct test: measure the recursion's own one-step defects and their Gram
matrix). CONJECTURE: going below L/n^2 needs higher angular structure. What is known: the trace sector of the cubic
adds nothing once the mean is matched, diagonal skewness is not the whole cubic correction (PROVEN [A r4]), and if
further orders follow ARC's ladder the precision exponent is again log q/log n ~ 0.25-0.375 at 256 (PROVEN UNDER that
transfer [A r4]).
The registered Amendment 5 run at 256 x 32 was stopped after one net: its radius transport mapped samples through a
fresh 2^16-sample pilot CDF at every layer, whose error shifts all radii coherently by ~0.1% per layer; over 32 layers
this random walk reached a replicate noise variance of 1.8e-3 (the Gaussian closure's: 1.7e-5), so P20-P23 are
undetermined (logs/deviations.txt). Amendment 6 (hashed before its runs) replaced it by a rank transport within each
batch; the radial noise fell to the Gaussian closure's level. The episode is itself informative: long suffixes read
the radius with weight 0.82-0.95 (t = 8-31; section 3) and the output scales with it (|Y|^2/V = 15-28 on these nets),
so to reach nMSE 1e-4 a radius closure must keep its amplitude law accurate to ~0.04% per layer if its errors are
independent across layers, ~0.007% if they are systematic (PROVEN UNDER multiplicative scale errors [A r4]).

## 5. Kelvin's hypothesis: "whether the bits get scrambled when correctly compressed"

Three precise readings, each checked against definitions:

- **Correctly compressed = the minimal description that still determines the output.** Then every layer carries
  exactly the output's own bits, and the code transition can be taken to be the identity (PROVEN [prior:
  bit-accounting; A]). Nothing is scrambled, and nothing is gained: computing that code is running the whole future.
  Under this reading the hypothesis is false and the compression is the estimation problem itself.
- **Compressed by a layer-local encoder.** At 256 x 32 (net 4000, whole 256-dim output vector, which is ARC's target;
  bit-accounting-arc via main 07:00Z, early counts not yet certified) a local code needs 15 / 90 bits at l = 31,
  32 / 186 at l = 16 and ~600 / ~1100 at l = 0 for 10% / 1% output error, and consecutive codes are related by
  near-linear threshold maps (96% of the predictable part [prior: bit-accounting B-ARC, MEASURED]). So the transitions
  are not scrambled, as far as those tests can see. Three cautions. The scrambled positive controls (chaotic tanh,
  Feistel) have not been run through the same transition tests, so this is "the tests found low degree", not "no
  scrambling". Cheap transitions do not make the code's law cheap to propagate (PROVEN separation example [A]: a
  weighted threshold is evaluated in linear time, while its exact probability under uniform inputs counts knapsack
  solutions). And per-sample description lengths say nothing about estimation cost: with the decoder G(c) = E[a | c],
  the code sum sum_c P(c) G(c) equals E a exactly at ANY rate (tower property), so the rate needed "at bias level" is
  zero bits (PROVEN, one line; sharpened by main). The whole cost of a code-based estimator sits in computing P(c) and
  E[a | c], the computability question of section 2; the curves are cited only as the structure a forward code must
  carry.
- **The frame is scrambled.** PROVEN: given the past, the next layer is an independent Gaussian matrix, and its
  neurons read the cloud's order-r cumulants through independent Gaussian rank-one probes (A4); a deep future reads the
  cloud through rotation-invariant radially normalised moments of high degree (A2, section 2.3). This is the version of
  "scrambling" that holds, and it is about orientation, not about code dynamics.
- **Correctly compressed for ARC's query, forward-only.** For the mean of a positively homogeneous network, the exact
  object a forward method must carry is the radius-weighted direction measure nu(du) = E[|h| | h/|h| = u] P(du) on the
  sphere (PROVEN [A r3]): the output mean is a linear functional of nu, and nu propagates linearly, by a gain-weighted
  push-forward. A deep future puts positive weight on radially normalised moments of every degree (b_d > 0 for every d
  once t >= 2 at infinite width, since kappa_1 has positive coefficients at degrees 0, 1, 2, 4, 6, ...; at finite
  width b_2, b_3 > 0 are PROVEN [A r3]), so no moment of nu can be dropped without a future that reads it. Under
  this reading "correctly compressed" means nu itself, and the question becomes how complex nu is in the layers where
  the future reads it at high degree: high-dimensional non-Gaussian structure early (P9), read at degree ~40 at
  infinite width (section 2.3; the finite-width spectrum beyond b_0 was not measured).

**Where the scrambling sits (main's question from the nd-exact result, checked at 256 x 32).** At 8 x 256 the few gate
classes the future needs are unions of up to 113,000 input regions each, while nothing is lost from one layer's class
to the next [prior: nd-exact]: the bits are scrambled relative to the input coordinates, not between layers. The same
split holds at 256 x 32, in measured form and with a theoretical reason:
- between layers: consecutive layers' output-relevant coordinates are related almost linearly (R^2 0.82 at 8 -> 12
  rising to 0.99 at 30 -> 31 on a dev net, 0.87-0.91 on net 4000) and learned code transitions are linear-threshold
  maps [prior: bit-accounting B-ARC, MEASURED];
- relative to where the law is computable: the value function's mean Hermite degree on the Gaussianised layer-l cloud
  is 21, 7.7, 4.1, 1.9, 1.3, 1.04 at l = 0, 4, 8, 16, 24, 31 [prior: relevance, MEASURED], and the input map's annealed
  mean degree is exactly n / E_W V, growing linearly with width at fixed L/n [prior: input-bits, PROVEN];
- the reason (A2 and section 2.3, PROVEN identity plus an infinite-width NUMERICAL spectrum): a future of depth t
  reads the layer-(L - t) cloud through radially normalised moments that, at infinite width, have mean degree 17 at
  t = 16 and 42 at t = 31 (half the mass above degree 6 and 10). Near the input the cloud is nearly Gaussian (its law is
  computable) but is read at high degree; near the output it is read at degree <= 4, but it is the push-forward through
  ~L random layers, which none of the tested families tracks (single-cut Gaussian defects, P7).
So the precise answer to "whether the bits get scrambled when correctly compressed": the layer-to-layer dynamics of
compressed codes are not scrambled (as far as the tests can see); the bits are scrambled relative to the coordinates
in which the law is computable. The tested forward methods would need both a computable law and a low read-out degree
at the same layer, and no layer of 256 x 32 offers both; whether an untested representation (implicit, future-aware)
could avoid needing that is open.

What blocks the tested layer-wise methods is this combination: the non-Gaussian structure of the true cloud is
high-dimensional in the first layers (P9; a descriptive statistic), where the closures' errors are made (P6), and at
infinite width a deep future reads it at degrees far beyond the orders a cumulant method can afford (section 2.3). So
"hash" (k* bits mixed by random gates) is the wrong picture for these methods; "random-orientation reads of a
high-dimensional, high-degree structure that is made early" is the better one, and it is a statement about
representation and computation, not about information.

## 6. The positive side, steelmanned and quantified

Every structure we know of that could still give a deductive win, with the best case I can argue for:

1. **The K = 2 budget window (real, small; MEASURED, section 4.2).** A deterministic estimator with error e and cost
   C beats a sampler exactly on the window from C to the cost at which the sampler reaches e (E4). At 256 x 32 ARC's
   k2a wins against RQMC on 4 of 5 nets (median 2.0x, 0.85-4.6x; the better of k2/k2a per net wins on all 5, a post hoc
   choice) at ~400-450 passes (precision 0.5-2e-3), and the window closes at ~400-1800 passes; means-only propagation (k1) wins by 16-90x but only at precision 1-4e-2 (an extrapolated
   comparison). The contest budget is 6.5e4 passes.
2. **Cheap higher orders through low rank.** The true cloud's third cumulant has participation ratio ~1-4 after layer
   ~8 (P9), so a low-rank representation there is plausible (a low PR alone does not make accurate propagation
   cheap). But the attribution (P6) puts most of the closure error in layers 1-8, where it is
   high-rank (PR3 ~115 at l = 1, 23 at l = 4). Best case: order-3 accuracy (k3f: 0.6-2.8e-4) at order-2 cost (~450
   passes), which would beat RQMC by 6-21x at that cost, up to the 2.3-6.4e3 passes at which RQMC reaches that error:
   a wider window, still closed far inside the contest budget (RQMC reaches 0.6-0.9e-5 at 6.5e4 passes). No method we
   know reaches it (k3f costs 4.9e4 passes).
3. **Low-order ANOVA and analytic controls** (astra's radial control for the multilevel sampler's first level;
   input-map first-order share ~0.30, pairwise ~0.20 [prior: input-bits]). These are control variates inside sampling.
   Acceptance criterion (astra, PROVEN arithmetic): a control with exact mean, residual share eta, evaluation cost a
   passes and setup B beats a 0.40/N design iff eta < (0.40/a)(1 - B/N); with eta ~0.5-0.7 for the shared-trajectory
   controls at 256 x 32 [prior] none qualifies. Best measured: ~2-2.5x over MC (RQMC, Kerdock).
4. **Depth synchronisation used deductively.** It pays when L >> n: at 8 x 256 the deep part past layer 20-48 is one
   linear map, and RQMC with the exact tail is 3-15x better than RQMC [prior: nd-exact]; the multilevel sampler's gain
   grows like n / log^2 n at L = alpha n (PROVEN UNDER cost hypotheses [prior]). At 256 x 32 the future is rough at every
   cut before ~0.8 L and the multilevel gain is ~1 [prior]: nothing to synchronise on.
5. **Exact integration at fixed weights.** For one fixed net, polyhedral region integration has cost polylog(1/eps)
   (PROVEN [A]): a deterministic method does eventually beat sampling as eps -> 0. The constants are astronomical
   (certifiable cells ~1e16-1e18 already at width 8 [prior: nd-exact]; exact integration wins only at widths 2-3).
6. **Future-aware probe tracking (a design, HEURISTIC).** A method that knows the next layer needs only the n per-neuron
   probes of each cumulant order, but each pulls back through every earlier layer; at tree level this costs ~L^2 n^3 per
   order, ARC's factorised cost. It changes the constant, not the ladder.
7. **The radius-corrected closure (our strongest lead; sections 2.4 and 4.4).** Measured as a finite-sample full
   recursion, its error is close to L/n^2 (1.9e-3, 9.2e-4, 4.9e-4 at n = 64, 128, 256 along L = n/8), 2-3x below the
   Gaussian closure at n >= 128 and 1.7x below ARC's k2a at 256 x 32 (though not at n = 64). As run it costs at least
   5.2e5 passes per replicate, at least ~255x what MC needs at its error. Best case for a deductive version, which was
   not built: it beats MC at that error if it costs less than ~5 order-2 steps per layer (2-3.5 against RQMC). Its single-cut value
   (the error if nothing accumulated) is 3-4x lower still, and RQMC reaches even that at 2.8-4.7e3 passes. It does not
   go below L/n^2 without higher angular orders (CONJECTURE), whose cost is not known.
8. **Untested routes.** Implicit representations of the early high-dimensional structure (a cheap description of a
   dense tensor is not excluded by A4), future-aware projections (no information obstruction; no lower bound here),
   other radius-direction couplings and projections, and participant proposals beyond the "scale-mode" correction
   of forum topic 18298. That correction has since been tested at orders 2 and 3 (outbox/scale-mode-2026-11-02.md):
   a constant factor in a window; two-level exponent equal to ARC's at 256 x 32 and ~0.1 at widths 128 and 256 on the
   L = n/4 diagonal; its higher orders are untested. Nothing here measures the other routes; the
   round-4 analysis says what a matrix-cost angular correction would need (a low-rank innovation representation
   maintained through gating; CONJECTURE).
9. **Shallow nets (small L/n).** ARC's theorems and their law say deduction wins when L/n is small; a participant
   reports a never-sampling Phase 2 estimator (1024 x 16, L/n = 1/64) 84x below full-budget MC (UNVERIFIED). Nothing
   here contradicts that: gamma = log(n/L)/log n is 0.6 there against 0.375 at 256 x 32, and each order buys a factor
   ~64 instead of ~8. The negative is about depth growing with width.
Bottom line of the steelman: the measured deductive wins at 256 x 32 are constant factors inside budget windows that
close below ~2e3 passes: about 2x for ARC's k2a on 4 of 5 nets at precision ~1e-3, and 16-90x for means-only
propagation at precision ~3e-2 (an extrapolated comparison). The radius-corrected closure has a lower error but
no measured cost-competitive implementation. Along L = n/8 the order-2 gains decay with width. At the contest budget
RQMC beats the best affordable ARC variant by 7-40x.

## 7. What this means for ARC

ARC already concede the direction: "This depth scaling is worse than Monte Carlo sampling" (App. D), and they expect
methods "beyond basic cumulant propagation" (Hilton). So the update this work supports is quantitative and specific,
not a change of view:
1. **The size of the gap at the contest shape** (sections 4.1-4.2; ARC's own code). At 256 x 32 the best order-2
   variant (k2a, ~450 passes) wins against RQMC on 4 of 5 nets and loses on net 203 (median ~2x), order 3 loses by
   5-30x at its own cost, and at the Phase-1 budget
   RQMC is 7-40x more accurate than anything in the hierarchy that fits the budget. The best sampling-based public
   entries are better still (approximately 10-50x below plain MC).
2. **The ladder to escape** (section 2.5). Along L = n/8 each cumulant order costs ~n times more and buys a factor of
   4-8 (about n/L); the measured exponent between orders 2 and 3 is 0.23-0.52 against sampling's 1, and ARC's own law
   gives log(n/L)/log n -> 0 at depth proportional to width. If that law holds as a lower bound, uniformly over orders
   (an assumption; the measurements cover K <= 4), any method that only climbs this ladder eventually loses to
   sampling at sufficiently high precision (its constants can preserve an early winning window), and that window
   narrows as depth grows with width (order 2's gain over MC fell from 8.1 to 2.6 between n = 32 and 256). A
   depth-proof method has to change the error-cost law, not its constant: "escape this ladder or lose".
3. **Where the error is, for the methods tested** (P6, P9, A2). The closures we tested make most of their error in the
   first ~8 layers, by accumulation, where the true cloud's non-Gaussian structure is high-dimensional and, at infinite
   width, is read by a deep future at high degree. A candidate method should say how it represents that early
   structure.
4. **The comparator at depth is the best sampler, not plain MC.** RQMC is ~1.5-2.5x better than MC at the relevant
   sizes and a Kerdock design 2.4x [prior: arc-noisy]. Asymptotically the multilevel shared-trajectory sampler's gain
   grows like n / log^2 n at L = alpha n (PROVEN UNDER cost hypotheses), while Gaussian closures have bounded gain there
   (PROVEN UNDER radius-sign hypotheses, which one exploratory test on one net did not contradict).
5. **What is not settled.** This is a restricted negative: the tested methods, on nets up to 256 x 32. Untested:
   implicit representations of the early structure, future-aware projections, other projections or couplings, a
   deductive radius-corrected closure (it would need to cost less than ~5 order-2 steps per layer to beat MC at its
   observed error near L/n^2; the related scale-mode estimator of forum topic 18298, since tested at orders 2 and 3,
   fits this: a constant factor in a window, outbox/scale-mode-2026-11-02.md), and that estimator's higher orders. A proof
   that no layer-wise method works would be a super-linear time lower bound, which no one can prove for any problem
   like this.

## 8. What is missing
- An unconditional lower bound for all deductive algorithms. It would be a super-linear time lower bound for an
  explicit distributional problem; no technique is known [prior: unrestricted 3.3]. This is the precise blocker for the
  strongest negative statement, and it is not specific to this problem.
- Within layer-wise estimators: a lower bound on the family defect inf_{Q in F} ||Q - P_l||_{k_t} over a whole model
  family (Gaussians, order-K Gram-Charlier, mixtures). We measure it for the moment-matched member and for the methods'
  own models, not the infimum.
- The radius-sign hypotheses (H_R, H_{R,s,tau}) behind the asymptotic separation: checkable layer by layer, not proved.
- A transfer from the order-K radial exponents to ARC's full recursion (projection, suffix propagation,
  accumulation [A]); for augmented even K even the radial exponent allows a polynomial advantage.
- For the radius-corrected closure: the hypotheses of the Theta(L/n^2) result (one-step defects O(n^-2), Gram bounds),
  testable by measuring the recursion's own one-step defects and their Gram matrix; a deductive implementation and its
  J; convergence of the finite-sample recursion in sample size and transport batch size; and variants that also
  preserve the mean and covariance while matching the radius law, or use other couplings (untested).
- Widths beyond 256 along L = n/8: every exponent here is a descriptive slope over a factor of 4 in width.
- The finite-width price spectrum beyond b_0 (the degree statements are infinite-width).
- Uncertainty for the single-cut values beyond push noise (independent cloud and pilot replicates), and k3fa on a
  second net at 256 x 32 (net 201 was registered but not run).
- The scrambled positive controls (chaotic tanh, Feistel) for the layer-to-layer transition tests (pending in the
  bit-accounting lanes), without which "not scrambled between layers" means "the tests found low degree". The
  certified per-layer bit curves are not needed for the estimation argument (tower property, section 5); they
  describe representation only.

## 9. Process, review, deviations, compute
- Co-prover gpt-6-astra (model line checked in every raw log; coprover/PROVENANCE.md): round 1 independent from a
  lemma-free statement (my FORMAL v1 frozen at 04:54:16Z before its answer existed, which I read at ~05:27Z); round 2 a
  claim-by-claim audit of v1 (applied as FORMAL E7: A3, B1, B3, C2, F3 corrected, CHECK-1 narrowed); round 3 on the
  radial lead and ARC's law (E10: the radius-corrected closure and its first-layer asymptotic bound, the minimal object
  nu, the H_cap conjecture, the correction that ARC's law gives no all-order "floor" but gives gamma_n -> 0 under
  uniform laws, the exact mean-degree identity, an audit narrowing E1, E5, E8); round 4 on the radial closure as a full
  recursion (E12: its own one-step defects, Theta(alpha/n) under Gram hypotheses, the next cubic order, amplitude
  tolerances, quadrature). The astra constants I used were rechecked as listed in PROVENANCE.md; some remainders and
  constants were not (also listed there).
- Pre-registration (prereg_hashes.txt; frozen code in frozen/): Part 1 hashed 05:29:26Z before any evaluation run;
  Amendment 1 (P11-P12) 05:31:54Z before the ladder queue; Amendment 2 (R1-R3) 06:17:21Z before any attribution run on
  nets 201-204 (net 200 seen); Amendment 3 (P13-P14) 06:20:00Z before any n = 128 ladder result was read (n = 32, 64
  seen, so P13 there is descriptive); Amendment 4 (P15-P18) 06:50:38Z before any width-scaling run (net 200's single
  cuts seen); Amendment 5 (P19-P23) 07:07Z before any run of its code; Amendment 6 (P24-P28) 08:25:38Z before any run of
  its code (Amendment 5's n = 64, 128 results and net 200 seen). Scorers hashed before the results they score
  (score_prereg at 05:55Z after I had seen the n = 32 ladder, disclosed). The GPT-6 reviewer verified all 25
  manifest-listed hashes and the frozen copies (content hashes; they do not by themselves prove chronology).
  Results (results/scorecard.md, final run 10:10Z, regenerated 13:45Z after the last rank runs): 31 rows hold (P24-P28
  among them), 4 fail (P4, P7, P9b, P12), 1 is partial (P19 holding on 1 of 5 nets, its check holding 5 of 5 on
  Amendment 6's runs) and 4 undetermined (P20-P23, stopped run).
- Deviations (logs/deviations.txt, all disclosed): queue X restarted after 1 min to pin BLAS to one thread (no output
  existed); the rank U-statistic rewritten in row tiles after the hash (identical outputs, tests/test_tiled.py); a dev
  rank job stopped; queue Y paused 06:05-06:15Z for the exploratory radius-sign test; queue X replanned (rank
  statistics take ~50 min per net at one thread, so P9 is scored on the nets that have them) and its stopped script
  killed by PID after its last job; queue V (Amendment 5) stopped by PID at 08:25Z when its 256 x 32 radial run proved
  noise-dominated, and replaced by Amendment 6; one 'pkill -f' that matched only my own shell (rule: kill by PID; not
  repeated); PREREG header times typed ahead of the clock four times and corrected (hash times authoritative); my dev
  calibration of P12 used k3f instead of the best order-3 variant (noted in this report's draft before P12 was scored;
  the earlier reference to logs/deviations.txt was wrong and the entry has been added there); exploratory measurements
  outside the registration (the fresh-layer radius-sign test, the finite-width b_0, the bootstrap slopes, the L/n^2
  observation) are labelled as such; scoring corrections after the review are in a separate script with the hashed
  outputs kept.
- Review: a fresh GPT-6 session (gpt-6-astra, model line checked; review/review_gpt6.raw, 09:29-09:40Z) gave a major
  revision with 17 findings; all were accepted in substance and applied: the negative is now restricted to the tested
  methods, with untested classes named as open (1, 16); the ARC-law argument is stated as conditional on a uniform
  lower law, as a lower bound, with its domain (2); degree statements are labelled infinite-width (3); section 2.4's
  results carry their cuts, limits, normalisations and hypotheses (4); the pricing identity's coverage is separated from
  the state-update model (5); the cell-count inference is removed (6); the oracle projections are benchmarks, not
  ceilings, and the measured and implementable reconstructions are separated (7); the full recursions are labelled
  finite-sample stochastic recursions with convergence open (8); width exponents are descriptive slopes with bootstrap
  intervals and both normalisations, and "neither grows polynomially" is withdrawn (9); the scorecard has explicit
  partial, pending and undetermined states and validity gates, and the full-recursion trend uses one transport at all
  widths (10); literal scoring corrections for P1/P5 and P3 (11); single-cut uncertainty and selection bias stated
  (12); rank statistics labelled descriptive and partial (13); P11 restated and the k3fa claim withdrawn (14); RQMC
  comparisons below 256 points labelled as extrapolations (15); provenance claims corrected (17). Co-prover round 4 ran
  in parallel and independently reached most of the same corrections to the radial sections. The reviewer read a
  version from 09:29Z; the revisions above were made afterwards and have not been re-reviewed.
- Independent adjudication (relayed by main, 10:22Z): honest and well registered, but a restricted negative, not "ARC
  should update"; ARC's App. D already concedes the depth direction, so section 2.5 is arithmetic on a conjecture they
  partly concede. Applied 10:25-10:30Z: the report is reframed as a benchmark of ARC's own code plus the ladder framing
  ("escape this ladder or lose"), with the measured parts first and section 7 renamed "What this means for ARC"; the
  scorecard count corrected to 29 holds (P28 had been missed) and P9 to 3 of 5 nets; P14's band stated as registered
  (a factor 1.85 below the law's value at n = 128, not "within 1.5"); the early radial reductions stated as 81-92% by
  median on the registered nets (74% at l = 2 by ratio of medians there, 78% on all five nets); E12.2 relabelled a
  consistency decomposition (it assumes the total defect energy Theta(L/n^2), with Gram bounds), not
  PROVEN; the radius-sign evidence called one exploratory net; "Theorem A" renamed a pricing identity; the order-2 win
  stated per variant (k2a loses on net 203); the radial closure's gain claim withdrawn (it charged 393 passes to a
  measurement that used >= 5.2e5) and replaced by its measured cost and a break-even cost; the public "scale-mode"
  counter-claim (forum topic 18298) named as untested (FORMAL E13). A short gpt-6-astra check of the revised text
  (review/check2_gpt6.raw, 10:30-10:32Z, model line checked) recomputed the corrected numbers (29/3/4/4; 1.847; the
  break-even costs; 255x) and found six leftovers, all fixed: L/n^2 called a "floor" without a convergence study, the
  ladder's "loses at every precision" (now: eventually, at high enough precision), the 74%/78% cohorts, "within 5%"
  (5.4% at n = 128), the ~255x as a per-replicate lower bound, and two wording leftovers.
- Compute: CPU only, nice 19, ionice -c3, at most two heavy processes (one briefly a third: the 2-minute dev BLAS run
  before the thread fix), one BLAS thread after 05:31Z; no GPU.
