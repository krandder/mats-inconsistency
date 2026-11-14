# Adaptive regions for ARC-style random ReLU nets: which regions can be understood exactly, how far merging them goes, and whether learned experts beat sampling (2 November 2026)

Status: final for the cone-tree round; the 4 x 4 A4/A5 runs on nets 9702-9703 are still running (noted where used).

Lane adaptive-regions (target M1), bead pool-ttdzrk.18 under pool-ttdzrk. Work directory `work/adaptive-regions/`.
Labels: PROVEN, PROVEN UNDER <hypothesis>, NUMERICAL, MEASURED (registered / exploratory), HEURISTIC, CONJECTURE.
Nothing here was preregistered, so every measurement is exploratory. [A] = gpt-6-astra, [C] = me.
Conventions: f = h_L (all n outputs), x ~ N(0, I_n), W_l iid N(0, 2/n), no biases, depth L. The error of an
estimate of E f is nMSE = |est - mu|^2 / tr Var f (plain MC with N paths: 1/N). Truths for the small nets are the
mean of 4 scrambled-Sobol runs of 2^24 points (2^26 in all); their own noise is reported next to each error.
"Advice paths" are the input samples a method uses before estimation (to list cells, to train). The headline
comparison, set by Kelvin, is against MC and RQMC that get the same number of samples as the advice.

## Summary for Kelvin

The net has no biases, so f(t x) = t f(x): every linear region is a cone, and E f = E|x| times the mean of f over
directions. The radial factor is exact. "Understanding a region" means knowing its map A_C and the region's first
moment E[x 1_C], and the facet formula (PROVEN, divergence theorem; E1 [A]) computes that moment from the solid angles
of the cone's facets. All results below are MEASURED (exploratory) on 3 nets per width, unless marked.

**1. Cone tree (merging similar regions), depth 4, widths 4, 5, 6.** How each property changes with width:

| property | 4 x 4 | 5 x 4 | 6 x 4 |
|---|---|---|---|
| cells (distinct gate patterns among 2^20 paths) | 103-200 | 355-734 | 1,110-2,524 |
| distinct maps among them | 103-200 | 320-715 | 1,110-2,481 |
| exact-cell error (every seen cell, exact moments) | 6e-15 to 5e-13 (at the truth's own noise) | 3e-10 to 2e-9 (limited by unseen cells) | 1.4e-8 to 1.5e-7 (limited by unseen cells; RQMC now wins) |
| MC / RQMC with the same 2^20 paths | 9.5e-7 / 1e-11 to 1.4e-10 | 9.5e-7 / 2e-10 to 1.1e-9 | 9.5e-7 / 5.6e-10 to 4.1e-9 |
| tree leaves to reach MC's error | 37-76 | 160-170 | 429-1,088 |
| tree leaves to reach 2x the exact-cell error | 102-200 | 272-462 | 736-1,504 |
| units fixed per leaf at MC's error (mean, of n L) | 4.1-6.5 of 16 | 7.7-8.0 of 20 | 9.2-10.3 of 24 |
| within-leaf map dispersion at MC's error (share of map energy) | 5e-5 to 2e-3 | 3e-4 to 1.5e-3 | 4e-4 to 1.5e-3 |
| layer of the first split | L3, L1, L4 | L3, L1, L1 | L1, L1, L2 |

Reading: merging buys a constant factor and the factor does not improve with width. The tree needs about 32%, 36% and 40% of the cells to reach MC's error (geometric means 51, 167, 737 leaves against 157, 468, 1,827 cells), and 56-100% to reach the exact-cell error. Leaves fix 35-45% of all units at every width. Merged maps are very similar (dispersion 3e-4 to 2e-3), but the number of groups grows as fast as the number of cells. Layer-1 units are the most frequent splits at every width. The exact-cell error itself degrades with width (unseen cells), and at 6 x 4 RQMC with the same paths beats it.

**2. Exact cells stop being usable between widths 8 and 12 at depth 4 (MEASURED).** With 2^20 advice paths the
number of cells grows about 3-4x per unit of width: 103-200 (4 x 4), 1.1k-2.5k (6 x 4), 17k-26k (8 x 4), 340k-410k
(12 x 4, covering only 74-81% of held-out paths), 880k-920k (16 x 4, nearly one cell per path, covering 18-25%). At
4 x 4 the exact-cell estimate is at the truth's noise floor (1e-14 to 5e-13), 6-8 orders below MC and 2-3 orders below
RQMC with the same paths. So "understand every region" wins outright where the regions are few, and the number of
regions is what kills it as width grows; this matches step 1 at 256 x 32 (patterns nearly unique per input).

**3. Learned experts (A4 bumps, A5 product of experts) do not beat averaging their own training samples, even at
4 x 4 (MEASURED).** N(0, I) error with the same advice count: A4 isotropic bumps 2.2e-5 / 3.1e-7 / 1.3e-6 / 8.2e-7 at
K = 4 / 16 / 64 / 256 (net 9701; MC 6.4e-7, importance-reweighted advice 1.3e-6, RQMC 1.05e-8); A5 2.3e-3 / 2.1e-3 /
7.0e-4 at K = 4 / 16 / 64 (MC 1.0e-6). The doubling check fails (the error moves 16-190% when training doubles), so
these are not converged to the 3% rule. At 32 x 4 the A4 error rose with K above 64 at a fixed budget, and A5 K = 64
(76.8k advice paths) gave 0.11-0.46 on held-out shells. Why: f is exactly linear on polyhedral cones; soft Gaussian
gates and bumps blur the cone boundaries, and A5 adds a Gaussian re-fit at every layer; the resulting bias (1e-7 to
1e-3) sits far above what 1M samples achieve. The exact-cell construction shows that the structure is there; the
smooth representation is what loses it.

**4. The certified-linear backward probe (steps 1-3): certification with our checks reaches 0 layers back at every
width (MEASURED).** Exactly linear pieces and eps-certified pieces (own IBP and CROWN, PCA boxes, slices, pins) cover
>= 50% of an output's cases only at the last hidden layer, on 256 x 32 and on the 32 x 4, 64 x 8 and 128 x 16 ladder.
These are coverages of particular sufficient checks, not limits on certification (review item 1).

**5. Theory (PROVEN under stated hypotheses, [C] and [A]).** Plain stratification by K regions chosen without the
weights, averaged over a random rotation, explains at most (K-1)/N_d of the variance in angular degree d; trees using
the weights have an asymptotic exponent gain beta <= 8/n for these nets. These bound stratified sampling only, not
learned control variates or exact understanding of regions.

## A. Cone tree: merging similar regions (Kelvin, ~21:50Z), depth 4, widths 4, 5, 6, 3 nets each

**Method (scratch/cone_tree.py; MEASURED, exploratory; moments NUMERICAL).** Cells are the gate patterns seen among
2^20 advice paths, each with its exact map A_C and exact first moment m_C = E[x 1_C] (section B: on the sphere,
E[x 1_C] = E|x| E_theta[theta 1_C(theta)], facet formula; exact solid angles at n = 4, Plackett orthant quadrature at
n = 5, 6). The tree starts with one leaf holding every cell. A leaf's merged map is the advice-mass-weighted mean of its
cells' maps (this uses the advice samples), and its contribution is that map times the sum of its cells' moments. A
leaf splits on the unit (layer, index) whose gate changes the map most, mass-weighted: score = p_on p_off / (p_on +
p_off) |Abar_on - Abar_off|_F^2, which is the drop in the leaf's within-leaf map dispersion. For two cells that differ
in one gate, the map difference is rank one (that unit's input weights times its downstream influence); after averaging
it need not be. The leaf with the largest score splits first. Caveat: a leaf is the set of cells agreeing on its split
units. When a split unit's upstream gates are not all fixed, this set is a cone but not a polyhedral one, and its moment
is the sum of its cells' exact moments. So the tree reduces the number of maps, not the number of moment computations.
Checks: the exact moments agree with the advice paths' own cell moments within 5 standard errors for 95-99% of cells
(the rest are mostly cells seen 1-3 times, where the standard error is itself unreliable). One 6 x 4 cell (advice mass
2e-6) used its sample moment after the orthant budget ran out.

**Per net (results/cone_tree.md; errors are nMSE of the full-input mean).**

| n x 4 | net | cells | distinct maps | exact-cell err | MC, 2^20 | RQMC, 2^20 | leaves to MC's err | leaves to 2x exact err | units fixed per leaf at MC's err (mean / max, of 4n) | within-leaf map dispersion at MC's err |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 9701 | 103 | 103 | 1.7e-14 | 9.5e-7 | 2.0e-11 | 76 | 102 | 6.5 / 11 | 5.1e-5 |
| 4 | 9702 | 200 | 200 | 5.9e-15 | 9.5e-7 | 1.3e-11 | 37 | 200 | 5.5 / 8 | 2.3e-3 |
| 4 | 9703 | 189 | 149 | 5.4e-13 | 9.5e-7 | 1.4e-10 | 46 | 123 | 4.1 / 8 | 6.1e-4 |
| 5 | 9701 | 734 | 715 | 3.4e-10 | 9.5e-7 | 2.1e-10 | 160 | 462 | 8.0 / 11 | 1.5e-3 |
| 5 | 9702 | 355 | 320 | 2.2e-9 | 9.5e-7 | 7.0e-10 | 170 | 272 | 8.0 / 11 | 3.0e-4 |
| 5 | 9703 | 394 | 371 | 8.0e-10 | 9.5e-7 | 1.1e-9 | 170 | 320 | 7.7 / 11 | 2.8e-4 |
| 6 | 9701 | 2,524 | 2,481 | 1.5e-7 | 9.5e-7 | 4.1e-9 | 858 | 1,408 | 10.3 / 16 | 5.9e-4 |
| 6 | 9702 | 1,110 | 1,110 | 1.4e-8 | 9.5e-7 | 5.6e-10 | 429 | 736 | 9.2 / 13 | 1.5e-3 |
| 6 | 9703 | 2,176 | 2,144 | 1.5e-8 | 9.5e-7 | 3.8e-9 | 1,088 | 1,504 | 9.6 / 17 | 4.2e-4 |

"Leaves to" is the first point on the greedy curve where the error falls below the target. The curve is not monotone:
a split can move the estimate away from the truth before later splits bring it back.

**What happens from 4 to 5 to 6.**
- Cells grow about 3-4x per unit of width (geometric means 157, 468, 1,827). Cells with identical maps are rare (0-21%).
- The tree does not grow slower than the cells. Leaves needed for MC's error: geometric means 51, 167, 737, i.e. 32%,
  36% and 40% of the cells. Leaves needed for 2x the exact-cell error: 65-100%, 63-81% and 56-69% of the cells.
  Merging buys a constant factor of 2-3 at MC's accuracy and almost nothing at exact accuracy, and that factor does not
  improve with width.
- Units fixed per leaf at MC's error: 4-6.5 of 16, 8 of 20, 9-10 of 24, so a steady 35-45% of all units. The leaves are
  not defined by a few units.
- Merged maps are very similar where the tree stops for MC's accuracy: the mass-weighted within-leaf dispersion is
  3e-4 to 2e-3 of the map energy at every width. Regions that differ only in the remaining units really are almost
  the same linear map, but there are too many such groups for the merge to change the scaling.
- Which units split first: the very first split is a layer-3 or layer-4 unit on 3 of 6 nets at widths 4-5 and a
  layer-1 or layer-2 unit on all 3 nets at width 6. Among the first 40 splits, layer-1 units are the most frequent at every width
  (11-23 of 40), then layers 2 and 4. The first layer's 2^n orthants are the coarse structure, and the late units
  separate the large-mass regions.
- Exact-cell error grows with width: from 1e-14 (4 x 4) to 3e-10-2e-9 (5 x 4) to 1.4e-8-1.5e-7 (6 x 4). The cause is
  the mass of cells never seen among the advice paths (held-out coverage 0.9996-0.9999 at 6 x 4), not the moment
  arithmetic. By 6 x 4 RQMC with the same paths (5.6e-10 to 4.1e-9) beats the exact-cell estimate. The hybrid
  (section B), which samples the unseen mass, is the fix, and the next step if Kelvin wants it.

## B. Exact cells, and how their number grows with width (depth 4)

**Method (scratch/exact_cells.py at 4 x 4, lib/cones.py in general; NUMERICAL where marked).** On a cell C (fixed gate
pattern of every unit) f(x) = A_C x exactly, with A_C the product of the masked weight matrices. So
E f = sum_C A_C E[x 1_C], and E[x 1_C] = E|x| E_theta[theta 1_C(theta)] = (2 pi)^{-1/2} sum_j g_j P_{n-1}(F_j), the
sum over the cone's facets F_j (unit normals g_j) of the Gaussian measure of the facet inside its hyperplane (PROVEN,
divergence theorem; FORMAL E1 [A]). At n = 4 a facet is a 3-dimensional cone and its measure is a spherical polygon's
area / 4 pi (exact). For n >= 5, lib/cones.py removes redundant and parallel constraints, triangulates the facet cone
(halfspace intersection of a cross-section, convex hull, Delaunay) and sums orthant probabilities of the simplicial
pieces, computed by Plackett's identity with an arcsine substitution and 32-node quadrature (agrees with Genz to about
1e-9; tests/test_cones.py checks the moments against 2^22-sample MC at n = 4, 5, 6, logs/test_cones.log). Cells are
the ones seen among the 2^20 advice paths.

**Result at 4 x 4 (3 nets; MEASURED, exploratory).**

| net | cells | held-out coverage of the cells | exact-cell error | truth noise | MC, same 2^20 paths | RQMC, same paths |
|---|---|---|---|---|---|---|
| 9701 | 103 | 1.000000 | 1.7e-14 | 7.5e-14 | 9.5e-7 | 2.0e-11 |
| 9702 | 200 | 1.000000 | 5.9e-15 | 3.4e-14 | 9.5e-7 | 1.3e-11 |
| 9703 | 189 | 0.999994 | 5.4e-13 | 3.0e-13 | 9.5e-7 | 1.4e-10 |

The exact-cell estimate costs arithmetic only (no network passes beyond listing the cells) and is at the truth's own
noise: at least 6-8 orders of magnitude below MC and 2-3 below RQMC with the same samples.

**How many cells (scratch/cells_scaling.py, runs/cells_scaling.json; 2^20 training and 2^20 held-out paths).**

| n x 4 | cells seen (3 nets) | cells for 99% held-out coverage | cells for 99.99% | held-out coverage of all seen cells | per-layer patterns, net 9701 | non-redundant facets per top cell |
|---|---|---|---|---|---|---|
| 4 | 103-200 | 58-115 | 90-185 | 0.99999-1 | 16, 5, 5, 6 | 4.8-4.9 |
| 6 | 1,110-2,524 | 500-951 | not reached | 0.9996-0.9999 | 64, 23, 21, 14 | 7.7-8.4 |
| 8 | 17,523-26,017 | 12,207-23,060 | not reached | 0.992-0.996 | 242, 75, 36, 38 | 10.8-12.7 |
| 12 | 336,575-406,109 | not reached | not reached | 0.74-0.81 | 3,674, 840, 316, 128 | 20.4-23.7 |
| 16 | 877,847-923,327 | not reached | not reached | 0.18-0.25 | 40,435, 7,737, 2,220, 1,430 | 19.1-31.9 |

The cell count grows about 3-4x per unit of width (read off these points, not fitted). The first layer alone has 2^n
cells (PROVEN: W_1 is square and invertible, so its gates are the orthants of W_1 x), and among 2^20 paths we see
16 / 64 / 242-256 / 3,641-3,697 / 37,767-41,019 of the 16 / 64 / 256 / 4,096 / 65,536 that exist. By 12 x 4 a million advice paths no longer
list the cells that carry the mass, and at 16 x 4 almost every path is its own cell. The cost of the exact part also
grows: about 0.05 s per cell at n = 5 and about 1.4 s per cell at n = 6 (facet triangulations with 10-40 simplices,
Plackett orthant probabilities in 4-5 dimensions), and the orthant dimension is n - 1 - (constraints dropped), so for
n >= 8 the recursion needs quadrature over 7-dimensional orthants, which I did not attempt.

**Hybrid (held by main).** E f = (exact over the listed cells) + (MC over the rest) was built (scratch/exact_hybrid.py)
and checked at 4 x 4 net 9701 with the top 58 cells (99% of advice): 2.5e-5 / 7.2e-7 / 1.4e-7 / 1.3e-9 at 2^8 / 2^12 /
2^16 / 2^20 residual paths against MC 5.5e-3 / 1.8e-4 / 9.3e-6 / 1.9e-6, so 100-1000x below MC, but RQMC on the whole
input (5.7e-5 / 5.0e-7 / 4.2e-9 / 9.3e-11) wins from 2^12 paths on, because the residual indicator breaks QMC's
smoothness. The 6 x 4 runs were stopped for the cone tree.

## C. Learned experts against their own training samples (Kelvin's headline)

The headline asks whether a learned model adds anything beyond averaging the samples it was trained on: N(0, I) error
of the model against MC and RQMC with the same total number of advice paths, and against the self-normalised
importance-reweighted mean of exactly those paths (the advice mixture as proposal). Held-out populations are secondary.
Training to convergence: early stop on held-out training populations, learning rate x0.3 on plateaus, then a doubling
check (as many steps again; Kelvin's rule: the error should move by less than 3%). scratch/step4m.py (A4) and
scratch/step5m.py (A5).

**4 x 4 net 9701 (MEASURED, exploratory).**

| model | K | N(0, I) error | MC, same paths | IS of the advice | RQMC, same paths | doubling check |
|---|---|---|---|---|---|---|
| A4 isotropic bumps (1.56M advice paths) | 4 | 2.2e-5 | 6.4e-7 | 1.3e-6 | 1.05e-8 | 73% |
| | 16 | 3.1e-7 | | | | 192% |
| | 64 | 1.3e-6 | | | | 91% |
| | 256 | 8.2e-7 | | | | 27% |
| A5 plain PoE, learned variances (1.02M advice paths) | 4 | 2.3e-3 | 1.0e-6 | 4.1e-6 | 1.0e-8 | 11% |
| | 16 | 2.1e-3 | | | | 19% |
| | 64 | 7.0e-4 | | | | 16% |

A4 at K = 16 lands between MC and RQMC once, but no A4 or A5 run passes the doubling check, and A4's error does not fall
with K. A5 is 3 orders of magnitude worse than MC. Coverage of A5's error bars on held-out populations (95% nominal):
0.82 / 0.59 / 0.53 at K = 4 / 16 / 64; with one global variance scale fitted on separate calibration populations
(c = 0.45 / 3.6 / 2.8): 0.59 / 1.00 / 0.71. The scale is not stable across K, so the variances are mis-shaped, not only
mis-scaled. Nets 9702 and 9703 were paused for the cone tree and are running again; they are not in this table.

**32 x 4 seed 9611 (MEASURED, exploratory).** A4 isotropic bumps, 1M advice paths, 1,200 steps at every K: held-out
shell error rises with K above 64 (s = 0.3: 0.060 / 0.11 / 0.27 at K = 64 / 256 / 1,024; s = 0.9: 0.0053 / 0.0099 /
0.029); main judged this undertraining, and the converged rerun was not done at 32 x 4. A5 on the L4 (1.02M advice
paths, 300 steps, all review fixes): deductive held-out error at s = 0.1 / 0.3 / 0.6 / 0.9 and N(0, I): K = 16:
0.75 / 0.46 / 0.17 / 0.042 / 0.013; K = 64: 0.52 / 0.32 / 0.11 / 0.037 / 0.022; MC at the same estimation cost
(1e5-9e5 passes) is 3e-8 to 9e-6. Full K curve (same settings): s = 0.1 / 0.3 / 0.6 / 0.9 / N(0, I): K = 256: 0.68 / 0.39 / 0.12 / 0.058 / 0.046; K = 1,024: 0.52 / 0.38 / 0.19 / 0.091 / 0.066, at 1e7 and 4e8 estimation passes; 95% coverage 0.71 / 0.82 / 0.76 / 0.82 at K = 16 / 64 / 256 / 1,024. The held-out error does not fall with K. These GPU runs used 300 fixed steps, not the convergence protocol of the 4 x 4 runs.

**Why they lose (HEURISTIC, supported by B).** f is exactly linear on polyhedral cones. A4's Gaussian bumps and A5's
Gaussian gates are smooth, so near every cone boundary the model is wrong by an amount set by the bump width, and the
mean inherits that bias (A4 leftover 0.003-0.04 at 4 x 4 integrates to 1e-7 to 1e-5). A5 also re-fits a Gaussian to
the propagated population at every layer, which costs about 1e-3 at 4 x 4. 2^20 samples already give 1e-6, so a
surrogate must get the mean right to about 1e-3 of its standard deviation, which needs the exact boundaries; the
exact-cell estimator, which has them, is at 1e-14.

## 1. The original question and the pivot
- Kelvin (2 Nov ~15:00Z, after the MCTS/AlphaZero analogy): "Here instead of 'looking for good moves' what we are
  looking for are the regions that we need to understand and sample more to better estimate the mean"; earlier:
  "some kind of monte-carlo thing, where we simultaneously 'learn' structure and similarity about different 'branches'".
  Target: the mean of ARC's random 256 x 32 ReLU net at a better error-vs-compute asymptote than sampling, or a clear
  argument why not. Brief from main: theory first (FORMAL.md), then a diagnostic of the best achievable gain over MC as
  the number of regions K grows, (sum_k p_k sigma_k)^2 / Var f ~ K^-beta, at n = 32, 64, 128, 256 along L = n/8, and a
  full test against RQMC only if beta > 0 held as width grows.
- Follow-ups relayed by main: Kelvin asked "what would be needed for this algorithm to be able to really basically stop
  visiting some regions?" (settle-ability, MCTS-Solver style, own IBP and CROWN). Then the pivot (~15:20-15:35Z): "I'm
  fine with looking for f=c (or particularly f=0) regions. I agree with calling it certified linear, and working
  backwards. We'll not seek to estimate probabilities of regions"; "Any 'certified linear' region needs to specify
  what, exactly, it is linear over ... so it composes"; and a step 4, a learned layered mixture of linear experts,
  revised twice ("experts at layer l-1 to land at multiple experts at layer l ... validity boundary ... accuracy ...
  and coverage"), with variant (a) chosen: zones and landing shares learned from samples of the same net; headline:
  leftover variance Var(f - g)/Var f against the number of input zones at widths 32-256.
- What was done: the theory of the original question (FORMAL v1, checked independently by gpt-6-astra); the scored
  beta diagnostic was cancelled by the pivot and never run; step 1 of the certified-linear probe on the five 256 x 32
  nets (200-204) and step 2 on nets 200-201; step 3 on the ladder (32 x 4 and 64 x 8 nets 300-302, 128 x 16 nets
  300-301); step 4 in its successive versions (section 6 and C); then, at Kelvin's request, exact cells, the 4 x 4
  headline and the cone tree (sections A-C).
- Conventions (as arc-update): f = h_L, all n outputs; x ~ N(0, I); W_l iid N(0, 2/n), no bias; nMSE =
  |est - mu|^2 / tr Var f (MC: 1/N); cost in forward passes (2 n^2 L flops). Truths: arc-noisy's 2^26-path truths for
  nets 200-204, arc-update's 2^24-path truths for the ladder. "back" = number of layers back from the last hidden
  state: back = L - 1 - l for the state h_l (back 0 is h_31 at 256 x 32, the input of the output layer).

## 2. Theory (FORMAL.md v1, appendix E1, pivot section P1-P2)
**Bookkeeping (PROVEN, elementary).** With K regions of exact probability p_k and within-region trace variance
sigma_k^2, Neyman allocation gives MSE (sum_k p_k sigma_k)^2 / N; proportional allocation (free: post-stratify plain MC
with exact p_k) gives sum_k p_k sigma_k^2 / N. If (sum p_k sigma_k)^2 / Var f ~ K^-beta and K ~ N^a, the MSE falls
like N^-(1 + a beta). To beat RQMC's fitted exponents (1.05-1.12 on these nets, arc-update runs) one needs a beta of
order 0.1-0.3 that holds as n grows. Allocation rules (MISER, VEGAS+, MC-UCB) can only lose against oracle Neyman on the
same partition; with a pilot the guaranteed exponent is 1 + beta/(1 + 3 beta) [A].

**Theorem 1: regions chosen without the net (PROVEN [C]; found independently by [A]).** Rotate any fixed partition into
K cells by a Haar-random rotation. Then, for every net, the expected between-cell variance is at most
sum_d min(1, (K-1)/N_d) E_d, where E_d is the net's energy in angular degree d and N_d = dim of degree-d spherical
harmonics (N_1 = n, N_2 = n(n+1)/2 - 1, N_d ~ n^d/d!). Proof: the averaged projection onto cellwise constants commutes
with rotations, so it is a scalar on each harmonic space, and its trace is at most K - 1. So a weight-blind tree needs
K ~ n cells to capture degree 1, ~n^2/2 for degree 2, ~n^3/6 for degree 3, whatever its cells look like. Along
L = alpha n about half of the angular variance sits at degree >= 3 (0.50 in the limit; degrees <= 2 carry 0.47 at
256 x 32 [prior: randomness E2.4]), so the proportional gain stays below about 2 for K up to n^2/2 at every width.
The same partitions with fixed allocation are linear rules, so the ensemble barriers of the preint and randomness
lanes apply as well.

**Regions chosen with the net.** No ceiling depending on K alone exists: [A] gives a 4-unit ReLU net whose 2-cell
partition gains 1/p for any p (a rare wedge). Three ceilings do apply:
- Proposition 2 (PROVEN [C], [A]): a tree whose splits use only directions in a subspace S of dimension s gains at
  most 1/(1 - Sbar_S) (proportional) and V/(E sqrt(tau_S))^2 (Neyman), Sbar_S the closed Sobol index of S. For a
  linear target, any depth-D hyperplane tree is capped at V / sum_{j>D} lambda_j [A].
- Proposition 3 (PROVEN [C], [A]): for any K-cell partition, (sum p_k sigma_k)^2 >= r N(P_r f) K^{-2/r} (entropy power
  of an r-dimensional output projection); for an output that is Gaussian over r dimensions, beta <= 2/r. Weak here:
  the deep output is nearly one-dimensional (covariance participation ratio 1.9 at 256 x 32).
- For these random nets specifically (PROVEN [A], checked): a full gate cell of positive mass whose linear map has
  rank r forces A(K) >= c K^{-2/r} for every partition; with every layer keeping >= n/4 active units (probability
  >= 1 - L e^{-n/8}), beta_asymptotic <= 8/n, about 4/n along L = n/8. This is asymptotic in K (c can be tiny).
So the beta test fails as a statement uniform in width: weight-blind trees by Theorem 1, every tree asymptotically by
the gate-cell bound. Finite-K constant gains remain possible, and [A] forecasts beta_eff of 0.001-0.03 for sign-split
trees over K = 16-4,096 (HEURISTIC). The measurement that would have tested this was cancelled.

**Understanding a region (PROVEN facts [C], [A]).** On a full activation region C, f(x) = A_C x exactly, so
E[f 1_C] = A_C p_C c_C: the linear map is cheap and everything sits in the region's Gaussian mass and centroid
(p_C c_C = (1/sqrt(2 pi)) sum_j u_j p_j^facet, facet probabilities [A]). Any K full regions cover expected mass at most
sqrt(K C_n), with C_n the first-layer collision probability: 1.0e-2, 1.7e-6, 4.3e-14, 2.9e-29 at K = 4,096 and
n = 32, 64, 128, 256 [A]. Cheap-mass families (boxes in a frame, cones of a few first-layer signs) contain many gate
boundaries, so f is far from linear inside them.

**Settling a region (PROVEN [C], [A]).** A region can be dropped when its certified interval times its mass is within
its share of the error budget. An unbounded cone has infinite output range (homogeneity), so bounds must work on the
unit-sphere section; a cap of angular radius delta at radius sqrt n meets about 0.8 n^{3/2} L delta gate boundaries
(HEURISTIC), so linearity-exact caps need delta << 1/(n^{3/2} L) and astronomically many caps. Settling at a deep layer
does not remove the probability problem: if the deep regions' probabilities come from M sampled paths, the error is
Var(m_cell)/M, the MC rate (PROVEN, [C] S5(d) and [A]).

**The certified-linear objects (pivot, FORMAL P1-P2).** A certificate at layer l is (D_l, A_l, c_l, eps_l) with D_l an
explicit set of layer-l states and |f_j - A_l[j].h - c_l[j]| <= eps_l[j] on it. Backward step: D_{l-1} must map into
D_l under relu(W_l .), and the functional pulls back with each unpinned unit costing its Chebyshev gap
-lo u/(2(u - lo)) (PROVEN: best uniform linear fit of relu on [lo, u]). Coverage can only fall going back (PROVEN).
Exact pieces are suffix gate-pattern cones (PROVEN). Boxes in the next layer's pre-activation coordinates make the
composition check exact interval arithmetic (PROVEN), but a box's worst corner exceeds the cloud's spread of a dense
row by about sqrt(n) (HEURISTIC). Domains defined by running the net trivialise the zero class (PROVEN), which is why D
must be explicit in layer-l coordinates.

## 3. Step 1: exact pieces (eps = 0)
Method (scratch/step1.py; MEASURED, exploratory). 2^17 training and 2^17 held-out Gaussian inputs per net; every gate
recorded. At each state h_l the exact linear pieces of the remaining net are suffix gate-pattern cones: for one output
j, the gates of layers l+1..L-1 plus j's own gate (cones where j is off all carry the functional 0 and are merged into
one zero piece); for the whole output vector, the gates of layers l+1..L. Pieces are ranked by training frequency and
their coverage is measured on held-out inputs, so a list of pieces never covers its own sample. Where the held-out
coverage cannot be resolved, the number of pieces needed for coverage q is bounded below by q^2 / sum_s p_s^2
(Cauchy-Schwarz, PROVEN), with sum_s p_s^2 estimated by held-out collisions. Where there are no collisions the code
uses a Poisson zero-count rule; the review showed that rule is not a distribution-free 95% bound (pair collisions are
dependent; a cell of mass 3/N gives zero collisions about 20% of the time). Where collisions occur the printed ">="
uses the point estimate. So the piece counts and bits below are estimates, not certified lower bounds.

Full per-net tables: results/tables.md, "Step 1" (generated by scratch/summarize.py from runs/step1_*.json).

Reading.
- The exact composition holds >= 50% coverage for 0 layers back on all five nets: only at back 0, where each output
  is a halfspace pair (100%). One layer back the 1,000 most common pieces cover 0.1-7.6% of an always-on output's
  cases (median 0.3%), with the same variance share; two layers back, 0 (at most 5e-5).
- Outputs that are zero on >= 99.9% of inputs (78-96 per net, ~0 variance) keep >= 99.9% only through the merged zero
  piece, which is the union of all their cones and is the zero set itself; as single cones they fall like the rest.
- The reason is visible in the gate statistics: 57-111 output units and 63-113 layer-31 units switch (are not constant
  on 99.9% of inputs), 137-170 at layer 16 and all 256 at layers 1-2; the suffix pattern's marginal-entropy upper
  bound is 18-45 bits at back 0, 323-421 at back 8, 2,601-2,984 at the input. Held-out patterns are all distinct from
  two layers back (2^17 of 2^17).
- Bits (the convenience metric main asked for): whole-vector exact pieces need 10.1 bits for 50% at back 0 on net 200
  and >= 18-20 bits on the other four; >= 20-29 bits one layer back; >= 28-29 bits (the resolution of 2^17 paths)
  further back; at the input, >= 200 bits in ensemble expectation (sqrt(K C_n) bound [A]). The local codes of
  bit-accounting-arc need 19 bits at layer 31 for 10% nMSE and 630 at layer 0. The two are different objects (a code
  quantises the state; a piece only names the linear map, which still needs the exact state), but exact pieces are
  not cheaper anywhere.

## 4. Step 2: eps-certified pieces (own IBP and CROWN)
Method (scratch/step2a.py, step2b.py; lib/ar.py; MEASURED, exploratory). The CROWN implementation is a full backward
linear relaxation with CROWN intermediate bounds at every layer, intersected with IBP, and adaptive lower slopes;
tests/test_ar.py checks soundness against 2,000-3,000 sampled points per box at several layers and sizes, tightness
against IBP, and exactness on a linear piece (50 checks pass). Certificates bound the output pre-activation
z_j = w_j . h_31 through the remaining layers (so the last ReLU is never relaxed): zero certificate f_j <= eps where the
upper bound is <= eps; linear certificate |f_j - a.h - c| <= eps where the lower bound is >= -eps/2 and the bound gap
is <= eps (a, c = the midpoint of the CROWN bounds). Domains, all explicit in layer-l coordinates:
(i) PCA boxes around the training cloud (scaled by tau = 1 ... 1e-4; tau = 1 holds 99% of held-out states);
(ii) a box plus the CROWN halfspaces themselves ("slice"); (iii) a slice plus pinned gates of layer l+1 (the units
constant on 99.9% of the cloud always pinned, plus 0, 4, 16, 64 or all switching units chosen by first-order effect).
One-shot bounds from layer l over a set are at least as tight as a layer-by-layer CROWN composition over the same set.
That makes these coverages upper bounds only for that composition with the same relaxation; they are not bounds on
certification in general (review: CROWN's adaptive slopes are not the best uniform affine fits, e.g. maximum error 0.5
against Chebyshev's 0.25 for ReLU on [-1, 1]; step2b tries one vector of majority pin signs; passing a one-element
intersection list turns off the IBP intersection at later layers, which is sound but can loosen the bounds). eps is relative to
each output's RMS: 0 (a 1e-9 tolerance in float64 without outward rounding, so "exact" here is numerical, not
machine-verified), 1e-3, 1e-2, 1e-1. Output selection, RMS normalisation and the tau / domain choices used test data,
so these coverages are exploratory validation statistics (review item 18).

Full per-net tables: results/tables.md, "Step 2a" and "Step 2b" (from runs/step2a_*.json, runs/step2b_*.json).

Reading.
- Certified coverage stays >= 50% for 0 layers back in every class and at every eps up to 0.1. At back 0 the slices
  (halfspaces) certify 96-99% (the rest is the tau = 1 box's own loss); whole boxes certify ~0 even there, because a
  box's corners reach about sqrt(n) beyond the cloud: over the box 251 of 256 layer-31 units straddle 0, while only 62
  switch on the cloud (net 200).
- One layer back: with the 189 cloud-fixed units pinned, zero outputs keep 0-46% (median about 6% on net 200), and the
  certificate is exactly zero, the same at every eps; always-on and switching outputs keep 0% at every eps and every
  number of pins short of the full pattern (which is step 1's piece, 0.05% coverage). Two and three layers back: 0.
- Units that must be pinned: every switching unit with a nonzero downstream coefficient; at eps <= 0.1 a single
  unpinned switching unit costs a Chebyshev gap comparable to the output's RMS in the worst case. So the slack only
  helps where effects are tiny, which is not where the switching units are.
- Compared with the certified branch-and-bound of the unrestricted lane (input cones; certified nothing from width 6
  on, 1e3-1e6x behind RQMC from width 4 to 32): certifying from the output end does not change that picture; it fails
  one layer from the output instead of at the input. The deep state's low dimension (third-cumulant PR 1-4) does not
  rescue it: the gaps come from the switching units, of which there are 57-113 at the last two layers.

## 5. Step 3: the ladder
The ladder runs of steps 1-2 (scratch/step3 chain, paused and resumed around the step-4 runs) cover 32 x 4 (nets
300-302), 64 x 8 (300-302) and 128 x 16 (300-301); results/tables.md has every row. At 64 x 8 and 128 x 16 (MEASURED):
the number of layers back with >= 50% exact coverage of always-on outputs is 0 on every net; the slices certify
0.89-0.997 at back 0 and 0 one layer back (at most 1.2e-4 for one switching output, net 128 x 16 / 300) at every eps up
to 0.1, with pins. At 32 x 4 (MEASURED): almost every
output switches (0-1 always-zero or always-on outputs per net), the switching outputs' halfspace pieces certify
76-88% at back 0 (one of the two halfspaces per output) and 0% one layer back at every eps up to 0.1, as at
256 x 32; exact suffix pieces cover at most 24% one layer back (top 1,000, net 302) and 0.2% two layers back. So the
"0 layers back" result holds at the smallest width too; width does not rescue certification.

## 6. Step 4 history: learned and oracle linear experts (SPEC-learned-experts.md, eight versions)
The specification changed during the session with Kelvin's decisions (relayed by main): gate-pattern composed experts,
then a Gaussian mixture of experts, a population-trained version, then the consolidated spec v1 with amendments A1-A5
(hashed versions in notes/formal_hashes.txt; chronology in notes/STEP4-LOG.md). Everything here is MEASURED
(exploratory). Section C has the converged, headline-accounted runs; this table keeps the earlier stages, corrected
after the review.

| version | width, nets | measured | result |
|---|---|---|---|
| gate-pattern composed experts (scratch/step4b.py) | 32x4 ... 256x32, 2 nets each | residual mean-square error E|f - g|^2 / tr Var f at the input vs 1-1,024 input zones (review: this is MSE, variance plus squared mean; e.g. 0.651 = 0.555 + 0.096 at 32x4 net 300) | 0.50-0.71, 0.64-0.80, 0.93-1.29, 0.96-1.28 by width, flat in zones |
| centroid zones, free regression (step4.py) | 256x32 net 200 | pointwise nMSE of f from h_l | 0.013 (l = 31) ... 0.92 (input) |
| oracle experts (A3, step4c.py) | 32x4 seed 9611 | N(0, I) error vs K | 1.9 ... 0.20 at K = 1 ... 256; a particular construction (local Jacobians plus fitted gates), not a ceiling (review item 13; Kelvin ruled the old-fusion oracle results not decisive) |
| A4 full bumps, K = 16 (step4f.py) | 32x4 seed 9611 | held-out error vs MC at equal estimation compute | model-only 0.28 / 0.16 / 4.8e-2 / 6.0e-3 at s = 0.1 / 0.3 / 0.6 / 0.9 vs MC 4.6e-5 ... 1.8e-3 |
| A4 isotropic bumps, K = 16 / 64 | 32x4 seed 9611 | same | at 8.8 / 35 passes: s = 0.6: 4.1e-2 / 2.1e-2 (MC 6.5e-2 / 1.7e-2); s = 0.9: 2.7e-3 / 2.0e-3 (MC 0.11 / 2.7e-2); so at K = 16 the model beat MC at equal estimation compute on these two held-out widths (review item 15), not RQMC |
| A4 isotropic, K = 64-1,024, 1M advice paths, 2,000 shells (step4k.py) | 32x4 seed 9611 | held-out error vs K, fixed 1,200 steps | rises above K = 64 (see C); undertrained per main |
| A5 plain PoE with learned variances (step5.py, GPU copy gpu_src/) | 32x4 seed 9611 | deductive error vs K | K = 16 / 64 / 256 (1.02M advice paths): s = 0.3: 0.46 / 0.32 / 0.39; N(0, I): 0.013 / 0.022 / 0.046; 95% coverage 0.71 / 0.82 / 0.76 |

Review fixes applied before the A5 K curve (logs/deviations.txt): triangular gate factor (item 7), every part kept in
the corrections (item 8), a defensive 50/50 importance proposal (item 9), the sampling covariance of the empirical
target in the calibration loss (item 10); R_err is not used by plain PoE (confirmed to main). Still open: the split
mass bug in LMLE.propagate (item 5; A5 does not split), the fused Jacobian without gate derivatives (item 6), and
operation accounting of the input-partition step (item 12).

## 7. Process, compute, review
- Order: QUESTIONS and READING; FORMAL v1 frozen at 15:16:02Z, before any run and before the co-prover's answer; the
  co-prover round (gpt-6-astra, 15:07-15:45Z); the pivot to the certified probe (steps 1-3); step 4 through its
  specifications; the independent review; the 4 x 4 headline, the exact cells and the cone tree (2-3 Nov). Deviations,
  kills (by PID) and pauses are in logs/deviations.txt.
- CPU: nice 19 and ionice idle. At most two heavy processes, except two short smoke tests and the 4 x 4 exact-cell runs
  (each under 4 minutes), which ran as a third process. To stay at two, I paused the A4 and A5 4 x 4 chains (SIGSTOP)
  while the cone tree ran.
- GPU (GPU-LOG.md): one L4 at a time through the shared inconsistency-search infra, budget group adaptreg, cap $5 at
  list price. Six attempts:
  - two spot VMs were preempted ($0.44, $0.63);
  - one on-demand VM ran out of GPU memory in training ($0.50);
  - one died on a CUDA generator bug ($0.07);
  - two completed, giving A5 at K = 16, 64 ($0.36) and K = 256, 1,024 ($0.84);
  - several creates failed on stockouts (no cost).
  Total at the rate ceiling: $2.84. No VM is left (gcloud instance list for adaptreg empty at 01:50Z on 3 Nov).
- Tests: tests/test_ar.py 50/50; tests/test_cones.py (moments vs MC at n = 4, 5, 6) passes; tests/test_lmle.py now
  exits nonzero, and its non-spec forced-split case fails (0.07), which is the split mass bug (review item 5, still
  open; A5 does not split).
- Review: gpt-6-astra (model line checked, review/astra_review.raw), 18 findings, verdict "substantial revision".
  - Fixed in code: items 7-10 (A5 gates, dropped parts, defensive importance proposal, calibration sampling
    covariance) and item 17 (test exit status).
  - Fixed in the text: items 1 (certification coverages are those of particular checks), 2 ("leftover variance" is
    MSE), 3 (theorem scope), 4 (collision rule not a bound), 13 (oracle and hard cells are constructions, not
    ceilings), 15 (A4 iso K = 16 beat recorded MC at s = 0.6 / 0.9), 16 (which nets ran which steps) and 18.
  - Open: items 5, 6, 11 (A4 population run without confidence-weighted importance correction), 12 (operation
    accounting) and 14 (P2's domain hypotheses).
  - Not yet reviewed: the cone-tree, exact-cell and headline work (sections A-C) was done after the review and has
    no independent review.
