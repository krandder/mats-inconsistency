# Can narrow-deep nets be solved exactly? Regions, classes and a certified deterministic code at width 8, depth 256 (2 November 2026)

Status: final. Three gpt-6-astra review rounds (major revision, major revision, minor revision); every finding
applied (section 12).

Lane nd-exact (target M1), bead pool-yode25 under Kelvin's ask pool-e0w4ba. Work directory `work/nd-exact/`:
`PREREG.md` (Part 1 hashed 04:14:52Z, Amendment 1 hashed 04:20:55Z; hashes and frozen code in `prereg_hashes.txt`,
`frozen/`; deviations and post-hoc additions in `logs/deviations.txt`), `lib/` (`ndx.py` nets and patterns,
`cubesphere.py` cone moments, `cs2.py` the certified code, `cs2tree.py` with its refinement tree), `tests/`
(`test_ndx.py` 3 checks, `test_cubesphere.py` 6 checks, all pass), `scratch/` (runs and scorers), `runs/`,
`results/` (`census_table.md`, `tables.md`, `scores.json`, `scores_a1.json`, `fig_curves.png`), `review/`. CPU only,
at most two processes at `nice -n 19 ionice -c3`. Labels: PROVEN, MEASURED, HEURISTIC; every number is MEASURED unless
labelled otherwise.

Ask (Kelvin, 2 November about 03:15Z): "Maybe we can use that to 'solve' things exactly for the narrow deep networks
that we can afford to do (much less than 40 bits)?" Target: a sample-free estimate of E[a] at width 8, depth 256 that
beats sampling at equal compute, or a precise account of where that cannot work. Framing relayed by main at 04:10Z:
regions are gate patterns, each contributing its exact Gaussian mass times the linear map at its conditional mean;
merge what the future treats alike; carry mass and moments per class; compose linear maps where the gates do not
depend on within-class differences; report error against the number of classes; optionally an unbiased version with
an MC correction.

## Summary for Kelvin

**Short answer: not with the exact methods I could build. What stops them is the input: 7 angular dimensions cut
into about 10^5 thin regions.** The deep part is easy: on every sampled input the problem past layer 20-48 is linear
and needs zero bits. The map from the 8 inputs to that layer has 1.3-2.2 x 10^5
linear regions in 4 million inputs. Their entropy is only 13-14 bits, but the tail is long: 1-2% of inputs land in
regions never seen. The certified code built in your framing lost to RQMC by 23-106 times (registered success test:
0 of 6 nets). A region code integrating regions exactly and dropping the rest lost to the best sampler even at an
assumed 9 mv per region. The reviewer's deterministic baseline, a fixed Sobol rule with the best sampler's tail map,
is competitive with that sampler but inconsistent across nets and budgets; it averages the net over a fixed point set,
which is sampling without random numbers, so it does not answer your question either. All of this concerns the
methods tested; it does not prove that no exact method can work.

1. **The reduction you expected holds on every sampled input** (PROVEN as an identity under its premise; the premise,
   one downstream class, is MEASURED on samples). E[a] = E|x| x (mean over the sphere). Once every live state shares
   one downstream gate pattern, E[a] = J_c E[h_c], with J_c of rank 1. That held on every sampled input from layer 20
   on nets 203, 220, 223, 24 on 217, 32 on 216 and 48 on 221. Past that layer the relevant state is one number entering
   linearly. The best sampler uses this: RQMCJ (Sobol points plus the sampled-class tail map, 55 mv a point instead of
   262) is 3-15 times more accurate than RQMC at equal compute.
2. **Regions (step 2).** At depth 32 there are 125k-221k distinct regions in 2^22 inputs (Chao1 1.9-3.5 x 10^5).
   90% of the output mass needs 6.5k-33k of them; 99% needs 50k-70k on four nets and more than 1.2 x 10^5 on the other
   two; 99.9% is unresolved on every net. The count rises steeply to depth 8 and saturates by depth 16-24:
   synchronisation stops new splits.
3. **Few bits per layer, many regions behind each bit.** The downstream classes (the gate distinctions the future
   needs) number 10-76 at layer 8, 2-8 at layer 16 and one from layer 20-48. The largest of them are unions of
   thousands to 113,000 input regions each, and a class's mass and mean are integrals over all of them. That is where
   the "scrambling" is: in the map from the input to the class. From one class to the next nothing is lost, because a
   class's gate pattern fixes its class at the next layer.
4. **Integrating regions exactly and dropping the rest loses** (T4 holds, 6 of 6). This is one strategy (rank regions
   on a sample of 2^22 inputs, integrate the top R exactly, drop the rest) at an assumed cost of 9 mv per region, which
   is generous: true Gaussian moments of a polyhedral cone in 8 dimensions need orthant integrals. The dropped part
   leaves 7-1,222 times RQMCJ's error at equal cost, at every resolved R. Under the same generous cost it beats plain
   MC by 2.5-14 times at about 2^17 regions on four nets (on 223, at all 125,293 regions the sample found).
5. **The certified code in your framing does not converge in time** (T1, T2 fail 0 of 6; T3 holds 6 of 6). Classes
   are input cones whose Gaussian mass and conditional mean come from one-dimensional integrals (relative error about
   3e-9). Gates are certified constant cell by cell (numerically, in double precision), and the linear maps are
   composed exactly there. Certified cells merge into one class at layer 32 through a certified tail, and refinement
   goes to the largest surplus. Up to 445,000 classes (open leaf cells), the certified cells carried a negligible mass:
   9 cells with 6e-6 of the probability on net 220, none on the other five. At its best point the code is 23-106 times
   worse than RQMC and 108-910 times worse than RQMCJ. Corrected for a root-cell defect (CS2c, post hoc) it needs 3.3-4.3
   bits per halving of the error, against 2 for MC and 1.3-2.1 for RQMC and RQMCJ.
6. **Why these methods fail: the regions are thin in 7 dimensions** (MEASURED distances; the cell counts are a rough
   HEURISTIC for isotropic cells). A random input direction lies a median 0.041 (sine of the angle) from the nearest
   region boundary at depth 1, and 0.0034-0.0063 at depth 32. If cells must resolve that scale in all 7 directions,
   covering the sphere takes about 10^10.5 cells at depth 1 and 10^16.2 to 10^18.2 at depth 32. Cells shaped to the
   regions could be far larger; finding and integrating those shapes is the region problem of item 4. On one ReLU layer
   (whose mean is closed form at every width) the same code is exact with about 400 cells at width 2 and beats RQMC
   92-170 times at width 3. At width 4 it is about 270 times worse than RQMC; at width 8, about 50,000 times worse.
7. **A deterministic point set is competitive but inconsistent** (post hoc; the reviewer's baseline). A fixed Sobol
   rule with RQMCJ's tail map beats RQMCJ's mean error in 42 of 66 (net, N) cells and its median in 32. Against 64
   scrambled runs per cell, the share of runs better than it averages 0.51 (by net 0.25 to 0.69). Between neighbouring
   budgets its error moves by up to 1,100 times, and it carries no error estimate. This establishes neither a
   systematic gain nor equivalence. It is a fixed point set, sample-free only in the literal sense that no random
   numbers are drawn.
8. **The unbiased version** (Amendment 1): D (the code at 2^20 mv) plus an RQMC correction of the within-class
   residual. It about matches RQMCJ (ratio 0.21-1.35; T5 fails, 0 of 6) and beats full-depth RQMC 3-14 times (T6 holds,
   5 of 6, against my prediction). A post-hoc paired ablation on the same inputs traces that gain to the 32-layer net
   with the certified tail and shows no benefit from the cells. Without the cells the estimated error is lower in all
   18 (net, points) comparisons (1.6-3.3 times at 2^16 points), with paired 95% intervals excluding zero in 10 of the
   18, and the 2^20 mv of cells is saved. My guess at the mechanism, untested: the cells' piecewise-constant values add
   jumps that RQMC integrates worse than the kinks of f itself.
9. **Plain statement (step 5).** For the methods tested, exact solution is not feasible at 8 x 256 at any precision
   measured (nMSE 0.3 down to 1e-7). The certified cell code fails already on one layer at width 8. The
   integrate-and-drop region code loses even at an assumed optimistic cost, because 10^5 or more thin regions carry the
   mass by depth 8 and the tail of rare regions is long. What could change this: a smaller angular dimension (widths
   2-3), or a way to integrate a union of regions without resolving it, which is the original problem.

## 1. Setup

- **Nets**: `ndx.make_weights(seed)`, identical to `nd.make_weights('plain', seed, 8, 256)` (checked in
  `tests/test_ndx.py`); W_l iid N(0, 2/8), no biases, h_l = relu(W_l h_{l-1}), a = h_256, x ~ N(0, I_8), sigma = 0.
  Evaluation seeds 203, 216, 217, 220, 221, 223 (the healthy seeds of the relevance and narrow-deep-m1 lanes); dev
  seeds 103, 111 for all choices.
- **Score**: nMSE = |est - E[a]|^2 / tr Var(a); plain MC with N paths scores 1/N.
- **Truth** (`scratch/truth.py`): E[a] = J_64 E[h_64] (section 2) with E[h_64] from 16 replicates of 2^22
  scrambled-Sobol points mapped to the sphere (radius exact): replicate standard error 3e-10 to 9e-10 in nMSE,
  conditional on the single-class premise. Cross-check by plain full-depth MC (2^21 inputs): differences 1.8e-10 to
  2.3e-6 against an expected 4.8e-7 (216 is the largest, 4.8 times its expectation for a one-dimensional error,
  p about 0.03); no input in 2^21 left J_64's downstream class. nMSE values below 1e-8 are not used in any test.
- **Cost** (a nominal FLOP model, not an exact ledger): 1 mv = 128 FLOPs (one 8 x 8 matrix-vector product with
  ReLU), flopscope conventions (matmul M N (2K - 1), elementwise 1, normal cdf and inverse normal 96, pdf 54, normal
  draw 16). Charged: forward passes, inverse normals, tail Jacobians, the tail-cone certificate, pilots, cell moments,
  certification, lookups, nominal heap bookkeeping. Known gaps (review finding 9): quadrature setup is not charged;
  the moment counter overcharges about 5 mv per cell; the tail-Jacobian setup undercharges its forward matvecs; Sobol
  generation, some reductions and RADQ's pilot draws are not charged. The largest known item, the moment counter,
  overcharges about 5 of the about 75 mv CS2 spends per constructed cell; the others are smaller per unit of work.
  None comes near the factors behind the verdicts (23 times and more). Offline structural information is not charged
  either: the census that chose RQMCJ's cut c = 48 cost about 2^30 mv per net.
- **Samplers** (`scratch/baselines.py`): MC (257 mv per path); RQMC (scrambled Sobol on the 8 inputs, 262 mv per
  point); RQMCJ, the strongest sampler and the fair comparator for the deterministic code, since it uses the same exact
  structure: J_48 E|x| mean f_48(x/|x|) over scrambled Sobol points (55 mv per point plus a 1,664 mv setup); RADQ, the
  relevance lane's estimator g(u*) E|h_32| with Sobol inputs, now with its 2,272 mv setup charged. Replicates: 40 up
  to 2^18 mv, 16 up to 2^22, 8 above. RQMCJ and the truth apply J to every point without a per-point class check
  (section 2).

## 2. The reduction (step 1 of the brief)

- **Radius** (PROVEN). The net is positively homogeneous, f(r u) = r f(u), and for x ~ N(0, I_8) the radius |x| and
  the direction u = x/|x| are independent. So E[a] = E|x| x (mean of f over the sphere S^7), with
  E|x| = sqrt(2) Gamma(4.5)/Gamma(4), about 2.741625. All cells below are cones, so this is used throughout.
- **Tail** (the identity is PROVEN; its premise is MEASURED, not proven). If every live state h_c at layer c has the
  same activation pattern in layers c+1..256 (one downstream class), the tail is one linear map on all of them and
  E[a] = J_c E[h_c] exactly, J_c = D_256 W_256 ... D_{c+1} W_{c+1} on that class. In the census (2^22 inputs per net)
  one live class carries all of the sampled output mass from c = 20 on nets 203, 220, 223, from 24 on 217, from 32 on
  216 (at 24 the rest is 7e-6 of the mass) and from 48 on 221 (at 32 the rest is 0.3%). J_c has rank 1 numerically
  (second singular value 1e-16 to 1.2e-15 of the first), so past that cut the relevant state is one number,
  w . h_c, entering linearly: zero bits are needed there. Kelvin's picture is right about this part.
- **Exceptions** (a sensitivity calculation, not a bound). At cut 48 (RQMCJ) the evidence is the census alone: no
  exception in 2^22 inputs per net. For a fixed cut that zero-event count would put the exceptional probability below
  about 7e-7 at 95%, but the cut was chosen with the same census, so no coverage is guaranteed.
  At cut 64 (the truth) the census and the truth's own check of 2^21 full-depth inputs both found none. What matters
  for bias is the exceptional contribution of a - J_c h_c, not the probability alone: if exceptional outputs are of
  typical size, the effect on E[a] is below about 1e-12 in nMSE, far under the 1e-8 floor of the scored comparisons.
  The cs2 code does not rely on this: it checks every state against a certified tail cone and falls back to a full
  pass.
- **The radius estimator's tail factor** (relevance lane, a(x) ~ |h_32| g(u*)) approximates the same identity: for
  in-class states its bias is J (E[h_c] - u* E|h_c|), second order in the angular spread.

So the whole difficulty is E[h_c] for c = 20-48: the mean of a piecewise linear map from the 8 inputs (7 angles once
the radius is integrated) to R^8.

## 3. How the sphere's mass splits over the activation regions (step 2; `results/census_table.md`)

A region at depth d is an activation pattern of layers 1..d: a polyhedral cone on which the depth-d net is linear.
Counts from 2^22 Gaussian inputs per net, hashed patterns (`scratch/census.py`); output mass = the region's share of
E|a|; GT = the Good-Turing estimate of the mass in regions never seen.

| depth d | distinct regions seen (6 nets) | Chao-Shen entropy (bits) | GT unseen probability |
|---|---|---|---|
| 1 | 248-256 (of 256) | 6.7-7.0 | below 1e-6 |
| 2 | 4.0k-6.0k | 9.0-9.7 | 1e-4 to 2e-4 |
| 4 | 20k-48k | 10.6-12.2 | 0.1-0.3% |
| 8 | 104k-121k | 12.3-13.3 | 0.8-1.0% |
| 16 | 121k-195k | 13.1-14.1 | 1.0-2.1% |
| 32 | 125k-221k | 13.1-14.3 | 1.1-2.2% |

At depth 32 (by net: 203, 216, 217, 220, 221, 223):
- distinct regions seen 202k, 139k, 152k, 199k, 221k, 125k; Chao1 richness (a lower bound on the total) 1.85e5 to
  3.54e5;
- regions covering 90% of the output mass: 25k, 11k, 6.5k, 6.9k, 33k, 12k; 99%: more than 119k (unresolved), 51k, 50k,
  70k, more than 128k (unresolved), 61k; 99.9%: unresolved on every net (above 1.0e5 to 1.9e5), because 0.5-1.4% of
  the output mass lies in regions never seen in 4 million inputs.
- **Growth with depth** (step 2's question): the count rises steeply to d = 8 and saturates by d = 16-24; d = 24 and
  d = 32 are identical on four nets and differ by at most 1% on the other two. Synchronisation stops new splits: the
  later layers cut almost nothing. The exact problem is therefore the first 16-24 layers.
- **Rank**: by d = 24 every live input on five nets has passed a layer with at most two active units (on 223, with
  at most three by d = 24 and two by d = 32); 98-100% of the output mass has passed a single-unit layer by d = 32 on
  216, 217, 221.

The entropy is small, as Kelvin guessed (13-14 bits, far below 40). But for a code that integrates regions exactly
and drops the rest (section 5), the error is governed by coverage, not entropy: the region masses have a long tail,
and such a code must find and integrate enough regions to leave less than the target error.

## 4. Few bits per layer, many regions behind each bit (where the "scrambling" is; `scratch/preimage.py`)

Downstream classes (the pattern of layers k+1..256) are the gate distinctions the future needs. Counted on 2^20
inputs per net: the classes covering 99% of the output mass, and the number of distinct activation regions of layers
1..k inside the largest three.

| net | k = 4 | k = 8 | k = 16 | k = 24 |
|---|---|---|---|---|
| 203 | 446 classes; 2,414, 5,982, 1,584 | 18; 10,644, 10,761, 13,776 | 2; 84,948, 23,580 | 1; 108,532 |
| 216 | 87; 4,046, 4,268, 5,748 | 17; 30,161, 9,599, 4,644 | 7; 43,395, 6,925, 3,145 | 1; 80,503 |
| 217 | 202; 7,773, 2,672, 2,819 | 33; 25,575, 12,180, 3,638 | 2; 78,748, 1,025 | 1; 82,380 |
| 220 | 396; 5,179, 3,512, 449 | 32; 17,535, 2,361, 4,234 | 5; 49,293, 27,334, 5,241 | 1; 99,993 |
| 221 | 628; 1,935, 2,232, 1,012 | 76; 7,881, 10,992, 4,711 | 8; 59,524, 29,586, 9,301 | 1; 113,024 |
| 223 | 203; 5,516, 2,710, 1,105 | 10; 24,247, 18,431, 9,640 | 2; 42,268, 26,164 | 1; 72,177 |

So there is no scrambling from one layer's classes to the next: a class at layer k fixes its class at layer k+1 (its
pattern is a suffix). The difficulty is the map from the input to the class. The preimage of a one-bit class at layer
16 is a union of up to 85,000 polyhedral cones, and its Gaussian mass and first moment (what the later layers need)
are integrals over all of them. Carrying a class forward exactly needs those integrals; carrying it forward
approximately is what the deterministic code below does, refining where the approximation is poor.

## 5. Integrating regions exactly and dropping the rest (`scratch/region_code.py`; T4)

One strategy for Kelvin's plan: integrate the activation regions of layers 1..32 exactly, one by one, largest
contribution first, and drop the rest. Two independent samples of 2^22 inputs: A finds and ranks the regions; B
measures the contribution of everything not in the top R (held out, so regions A never saw count as dropped). Each
region is charged an assumed 9 mv (8 for its linear map, 1 for its moments). This is meant as a generous cost, not a
proven lower bound: the true Gaussian mass and mean of a polyhedral cone in R^8 have no closed form and need 7- and
8-dimensional orthant integrals (the narrow-deep-m1 lane measured one 8-dimensional orthant probability to 1% at
410-6,500 MC paths of FLOPs, about 1e5 to 1.7e6 mv), while a cleverer method might share work between regions or need
only a row vector per region (J has rank 1). Finding the regions (sample A, 2^22 x 32 mv) is not charged.

nMSE of the dropped part, held-out / in-sample:

| net | regions in A | B inputs outside A's regions | R = 2^10 | 2^14 | 2^16 | 2^17 | all of A's |
|---|---|---|---|---|---|---|---|
| 203 | 201,877 | 2.0% | 0.51 / 0.51 | 3.7e-2 / 3.5e-2 | 3.0e-3 / 1.8e-3 | 6.7e-4 / 8.7e-5 | 3.2e-4 / 0 |
| 216 | 139,111 | 1.2% | 0.12 / 0.12 | 2.5e-3 / 2.3e-3 | 5.7e-5 / 1.6e-5 | 1.5e-5 / 1.6e-9 | 1.5e-5 / 0 |
| 217 | 152,101 | 1.5% | 8.8e-2 / 8.8e-2 | 1.9e-3 / 1.7e-3 | 8.9e-5 / 3.0e-5 | 2.4e-5 / 3.6e-8 | 2.2e-5 / 0 |
| 220 | 199,130 | 2.3% | 3.3e-2 / 3.3e-2 | 1.2e-3 / 1.1e-3 | 1.1e-4 / 5.1e-5 | 3.9e-5 / 2.2e-6 | 2.6e-5 / 0 |
| 221 | 220,775 | 2.2% | 0.85 / 0.85 | 7.5e-2 / 7.2e-2 | 4.9e-3 / 3.2e-3 | 8.8e-4 / 1.6e-4 | 3.7e-4 / 0 |
| 223 | 125,293 | 1.1% | 0.46 / 0.46 | 1.3e-2 / 1.3e-2 | 4.6e-4 / 1.8e-4 | - | 9.0e-5 / 0 |

- **T4 (registered): holds on 6 of 6 nets.** At every resolved R (held-out nMSE at least 10 times its noise; R from
  2^11 to all of A's regions) the region code's error exceeds RQMCJ's at cost 9R mv: by 143-600 times on 203, 13-211
  on 216, 7-93 on 217, 11-25 on 220, 80-1,222 on 221, 30-622 on 223. The gap narrows with R, and the held-out curve
  floors where A's regions run out: 1-2% of fresh inputs land in regions that 4 million inputs never visited.
- **Against plain MC** the same accounting gives a gain: at R = 2^17 (1.2e6 mv) the region code beats MC 14 times on
  216, 9 on 217 and 5.6 on 220, and 2.5 times on 223 at all 125,293 of A's regions (1.1e6 mv; 223 has fewer than
  2^17); it loses on 203 and 221.
- **Scope.** This tests one strategy (sample-ranked, integrate and drop) under an assumed cost. The in-sample curve is
  optimistic (fitted ranking, unseen regions ignored) and the held-out one is a specific region set; neither bounds the
  best possible choice of R regions, and the strategy does not cover approximating the remainder, integrating unions
  of regions, or sharing work between regions.

## 6. The certified deterministic code (Kelvin's framing; `lib/cs2.py`; T1-T3)

Cells are cones over dyadic boxes on the 16 faces of the cube [-1, 1]^8: C = {s x_i > 0, x_j / |x_i| in
[alpha_j, beta_j]}. Such a cone is convex and polyhedral, and given t = |x_i| the other coordinates are independent,
so its Gaussian mass and conditional mean are one-dimensional integrals, computed numerically (Gauss-Legendre, 32
nodes on [0, 9]; relative error 3.2e-9 against 160 nodes on 200 test cells, `tests/test_cubesphere.py`; the test
asserts the tolerance only for 64 nodes). In Kelvin's terms:
- **Classes and the exact limit.** Each cell is a class. It contributes its Gaussian mass times the net at its
  conditional mean. In exact arithmetic that is exact whenever the gates are constant on the cell (the conditional mean
  lies in the cell, and the net is linear there), so the estimate becomes exact as cells shrink; numerically it is
  exact up to the quadrature error.
- **Skips.** Gates are certified constant on a cell by interval arithmetic in its gnomonic coordinates, layer by
  layer, composing the linear maps exactly (kept in factored form once fewer than 8 units are active). The interval
  bounds are exact in real arithmetic; the implementation uses double precision without outward rounding, so the
  certificate is numerical (review finding 4). Certified cells are checked against brute-force sampling in the tests
  (one activation pattern; f = J A x through all 256 layers).
- **Compression.** At layer c0 = 32 the certified cells whose image lies in the tail cone K are merged into one class
  and sent to the output by one exact map J. K is a box cone around the synchronised direction on which the gates of
  layers 33..256 are certified constant (half-width 2^-12 to 2^-6 in gnomonic units; 3,200-3,800 mv of setup,
  charged). States outside K get a full forward pass.
- **Refinement.** Split the open cell with the largest surplus first (half the change its last split made to the
  estimate), along the coordinate in which its straddling gates vary most. An a priori rule (the forward-propagated
  distortion of the straddling gates) and the maximum of the two were worse on the dev nets. CS2b (Amendment 1) adds a
  mass floor for children; CS2c (post hoc) is CS2b with every root face floored as well.
- No random numbers anywhere; the output is a deterministic function of the net.

Results (`results/tables.md`; `results/fig_curves.png` shows every method on every net; classes = open leaves, about
half of the constructed nodes):

| net | CS2 nMSE at 2^20 / 2^23 / 2^26 mv | classes at 2^26 | certified cells (probability mass) | RQMC at 2^24 | RQMCJ at 2^24 |
|---|---|---|---|---|---|
| 203 | 1.6e-2 / 1.0e-2 / 6.6e-3 | 445,137 | 0 | 1.3e-6 | 1.9e-7 |
| 216 | 5.9e-3 / 2.4e-3 / 7.7e-4 | 444,103 | 0 | 2.2e-7 | 4.1e-8 |
| 217 | 8.9e-3 / 4.4e-3 / 2.7e-3 | 444,682 | 0 | 5.8e-7 | 1.3e-7 |
| 220 | 6.9e-3 / 2.4e-3 / 8.1e-4 | 445,258 | 9 (5.8e-6) | 1.1e-6 | 2.0e-7 |
| 221 | 1.5e-2 / 6.2e-3 / 2.3e-3 | 421,563 | 0 | 2.0e-6 | 4.1e-7 |
| 223 | 1.5e-2 / 5.3e-3 / 1.6e-3 | 444,220 | 0 | 4.9e-7 | 7.4e-8 |

- **T1 (registered success criterion; CS2 at most a third of RQMC's nMSE at equal cost, at some point in 2^16-2^24
  mv, on 4 of 6 nets): fails, 0 of 6.** The best ratio RQMC/CS2 is 0.0095-0.044: CS2 is 23-106 times worse at its
  best point.
- **T2 (the same against RQMCJ): fails, 0 of 6.** Best ratio 0.0011-0.0093: 108-910 times worse.
- **T3 (bits per halving above 2, the MC value): holds, 6 of 6.** The frozen scorer gives 3.6-8.5 bits per halving
  over 2^18-2^24 mv (CS2b 3.1-7.5); its window drops the 2^24 checkpoint, whose cost lands just above 2^24. Refitted
  descriptively over the nominal checkpoints 2^18-2^24 for every method: CS2 3.7-8.8, CS2b 3.3-7.6, against 2 for MC,
  1.44-2.06 for RQMC, 1.31-1.86 for RQMCJ and 1.35-2.04 for RADQ.
- **Error against the number of classes.** The certified mass is negligible, so the classes are the open leaves. From
  about 6,900 classes (2^20 mv) to 445,000 (2^26 mv), 6 bits more, the nMSE falls 2.4-9.4 times, which is 0.6-1.6
  halvings of the error. Six more bits of samples would buy MC three halvings.
- **A defect, found after the runs** (`logs/deviations.txt`, review finding 6). A root face cell whose centroid output
  is zero got priority zero and was never split. Amendment 1's mass floor applied only to children, so CS2b has the
  same defect. It hit one face on 203 and one on 217 (each a sixteenth of the sphere) and no other net. CS2c (post hoc)
  is CS2b plus a floor on every root face, run on all six nets. On the four unaffected nets its curves equal CS2b's;
  the root fix alone improves the final error 6.7 times on 203 and 3.7 times on 217 (against CS2b). At 2^26 mv CS2c
  reaches 4.2e-4 to 1.6e-3 on the six nets, with 3.3-4.3 bits per halving: still 6,000-30,000 times RQMCJ's error. The
  frozen results stay as the registered outcomes; conclusions about convergence use CS2c.

## 7. Why these methods fail: thin regions in 7 dimensions (`scratch/kink_distance.py`, `scratch/depth1_rate4.py`)

**Distance to the nearest region boundary** (2^14 random directions per net). For a direction u, d(u) is the distance
to the boundary of its activation region of layers 1..k: the minimum over gates of |w . u| / |w|, with w the gate's
row in the region's linear map, which equals the distance to the nearest facet.

| depth k | median d (6 nets) | 10% quantile | 1% quantile |
|---|---|---|---|
| 1 | 0.041 | 6.3e-3 to 6.6e-3 | 5.3e-4 to 6.6e-4 |
| 8 | 6.4e-3 to 7.3e-3 | 6.7e-4 to 8.8e-4 | 5.2e-5 to 8.6e-5 |
| 32 | 3.4e-3 to 6.3e-3 | 3.6e-4 to 7.7e-4 | 2.6e-5 to 7.6e-5 |

A rough isotropic-resolution model (HEURISTIC; review finding 5): suppose cells must resolve the scale d in all 7
directions to be certified. A cap of radius d covers a fraction 0.146 d^7 of the 7-sphere, so covering the sphere at
the median distance would take 10^10.5 cells for one layer and 10^16.2 to 10^18.2 at depth 32. This is not a
containment requirement or a lower bound: d is the radius of the largest ball around a point inside its region, and
a contained cell, even a box, can extend farther along the region's facets. It is consistent with what we saw: our
largest runs had 10^5.6 open cells and certified a negligible mass. Building cells shaped to the regions means finding
the regions' geometry, the problem of section 5. Without certification, the centroid
rule's error in a cell of size h crossed by gates falls like h^2. With N ~ h^-7 cells (h^-6 when refinement follows
the kinks), the error falls like N^-2/7 to N^-1/3, that is 3 to 3.5 bits per halving (HEURISTIC); CS2c measured
3.3-4.3.

**One ReLU layer by width** (`runs/depth1_rate4.json`, post hoc with the root-cell fix; the registered-rule versions
`depth1_rate2.json` and `depth1_rate3.json` stall at width 2 because of the defect and agree elsewhere). One layer
has a closed-form mean at every width, E[relu(W x)]_j = |w_j| / sqrt(2 pi), so this measures the cell code's
efficiency against RQMC on a known answer, not whether exact integration is possible. Compared at the code's last
stored checkpoint within RQMC's measured cost range (RQMC interpolated log-log there):

| width | angular dimensions | code: nMSE, certified mass | RQMC at the same cost | RQMC / code |
|---|---|---|---|---|
| 2 | 1 | 1.5e-30 (exact), 1.000, about 400 cells | 2.3e-8 to 2.5e-8 | 10^22 |
| 3 | 2 | 9e-13 to 5e-12, 0.99 | 1.6e-10 to 4.8e-10 | 92-170 |
| 4 | 3 | 5.8e-7 to 6.2e-7, 0.66-0.67 | 2.1e-9 to 2.2e-9 | 0.0034-0.0038 |
| 6 | 5 | 5.2e-4 to 6.0e-4, 0.00 | 3.4e-8 to 3.7e-8 | 6e-5 |
| 8 | 7 | 2.9e-3 to 3.1e-3, 0.00 | 5.7e-8 to 6.7e-8 | 2e-5 |

For this cell code the crossover sits between widths 3 and 4, between 2 and 3 angular dimensions. Width 8 has 7.

## 8. A deterministic point set (the reviewer's baseline; post hoc; `scratch/posthoc_dsob.py`)

The review's critical finding: a fixed cubature with no random numbers, the unscrambled Sobol net of N = 2^m points
shifted by 1/(2N), mapped to the sphere, through layers 1..48 and the tail map J_48 (exactly RQMCJ's integrand), beat
RQMCJ's mean error at N = 4096 on four nets (by 109, 18, 4.9 and 6.3 times). It is deterministic and so qualifies
literally. Checked at every N = 2^6 to 2^16 on all six nets against 64 independent scramblings of the same rule:

- It beats RQMCJ's mean nMSE in 42 of 66 (net, N) cells and its median in 32 of 66.
- The share of the 64 scrambled runs that are better than it averages 0.506 over the 66 cells (quartiles 0.21, 0.54,
  0.79). By net the mean share is 0.25 (203), 0.45 (216), 0.67 (217), 0.51 (220), 0.45 (221) and 0.69 (223); the
  standard error of the overall mean across the six nets is about 0.07. The 11 budgets on one net are not independent
  draws, so these numbers are descriptive.
- RQMCJ's nMSE is skewed: its mean is 1.3-4.4 times its median, so beating the mean is easier than beating a typical
  run. The fixed rule's error scatters around RQMCJ's median by a factor of about 7.6 (one standard deviation of
  log10, 0.88), and between neighbouring N it moves by up to 1,100 times.

So the fixed Sobol rule is competitive but inconsistent across nets and budgets: better than most scrambled runs on
203, worse than most on 217 and 223. This experiment establishes neither a systematic gain nor equivalence, and the
rule gives no error estimate. It answers "deterministic" but not "exact"; in the earlier lanes' definition (an
estimate that is not an average of the net's output over a point set) it is not sample-free. Exact one-dimensional
integration along lines (pre-integration, the usual QMC remedy for kinks) was not tried: a random line crosses a median
of 103 regions of layers 1..48 (dev net 103, 200 lines, `scratch/line_crossings.py`), so an exact line costs about
10^4 mv, as much as 180 RQMCJ points.

## 9. The unbiased version (Amendment 1; `scratch/run_unbiased.py`; T5, T6)

E[a] = D + E[f(x) - q(x)], with D the code's estimate at 2^20 mv and q(x) the value of x's cell (its centroid
output). The correction is averaged over scrambled-Sobol inputs (residual zero on certified cells), with f evaluated
through 32 layers and the certified tail. It is unbiased up to the cells' quadrature error (relative 3e-9).

| net | Var(f - q) / Var(f) (plain MC) | nMSE at M = 2^12 / 2^14 / 2^16 points (cost 1.2e6 / 1.8e6 / 3.9e6 mv) | RQMCJ at the same costs |
|---|---|---|---|
| 203 | 0.59 | 8.8e-5 / 1.4e-5 / 1.6e-6 | 3.5e-6 / 2.3e-6 / 9.3e-7 |
| 216 | 0.41 | 4.9e-5 / 8.9e-6 / 1.1e-6 | 1.1e-6 / 6.9e-7 / 2.3e-7 |
| 217 | 0.43 | 5.5e-5 / 8.4e-6 / 2.2e-6 | 3.4e-6 / 2.8e-6 / 1.7e-6 |
| 220 | 0.51 | 6.7e-5 / 2.8e-6 / 7.1e-7 | 3.3e-6 / 2.0e-6 / 7.1e-7 |
| 221 | 0.64 | 8.3e-5 / 1.3e-5 / 3.8e-6 | 7.5e-6 / 5.0e-6 / 2.1e-6 |
| 223 | 0.56 | 6.1e-5 / 1.2e-5 / 5.2e-7 | 2.7e-6 / 1.8e-6 / 7.0e-7 |

- **T5 (UNB at most a third of RQMCJ's nMSE): fails, 0 of 6** (best ratios RQMCJ/UNB 0.21-1.35). With 8 replicates
  the largest-budget estimates have relative standard errors of 30-65%.
- **T6 (the same against full-depth RQMC): holds, 5 of 6 (ratios 2.75-13.7; 221 at 3.05).** I predicted a failure.
- **Ablation** (post hoc, 32 paired replicates per (net, M), `scratch/posthoc_unb_ablation.py`, per-replicate errors
  in `runs/posthoc_unb_ablation_s*.json`): on the same scrambled points, the same estimator without cells (D = 0,
  q = 0: plain RQMC on the 32-layer net with the certified tail, charged its tail-cone certificate, 1,100-1,700 mv;
  the 16 direction probes behind the cone, about 512 mv more, are not charged, under 0.3% of its cost) has lower
  estimated nMSE in all 18 (net, M) comparisons; at 2^16 points 1.6-3.3 times lower (203: 1.7, 216: 3.3, 217: 2.1,
  220: 3.0, 221: 1.6, 223: 3.2). The paired 95% t-intervals for the difference exclude zero in 10 of the 18 (including
  2^16 points on 216, 217, 220 and 223; not on 203 and 221). It also saves D's 2^20 mv. So the ablation supports the
  truncated net with the certified tail as sufficient for T6's gain and shows no benefit from the cells. Under plain MC the
  cells do halve the variance (the ratios above). My guess at why they do not help under RQMC (a hypothesis, not
  tested): the residual f - q jumps at every cell boundary, and RQMC integrates jumps worse than the kinks of f itself.

## 10. Plain statement (step 5 of the brief)

- **Feasible at 8 x 256?** Not with the exact methods built here. Over nMSE 0.3 to 1e-7 (compute 2^12 to 2^26 mv),
  the certified code and the integrate-and-drop region code lose to RQMCJ at equal compute on all six nets, the code
  by 10^2-10^4 and the region code by 7-1,222 times even at an assumed 9 mv per region. A deterministic point set is
  competitive with RQMCJ but inconsistent across nets and budgets (section 8). These are results about the methods
  tested, not an impossibility proof.
- **Where it breaks.** Depth: at depth 1 the mean is closed form unit by unit (each output is one ReLU of a Gaussian),
  although region by region the 255 orthant cones already need 8-dimensional orthant probabilities. From depth 2 no
  unit-wise shortcut was found here: region by region, each of the 4,000-6,000 cones needs 7- and 8-dimensional
  orthant integrals, and by depth 8 there are 10^5 or more. The certified cell code fails already on one layer at
  width 8, a problem with a closed-form answer: at 8,192 cells (its largest checkpoint within RQMC's measured cost
  range) it is 4.6-5.0 x 10^4 times worse than RQMC, and at 131,072 cells it still certifies a negligible mass.
  Precision: at every precision measured; the gap grows with precision, because the corrected code's slope (3.3-4.3
  bits per halving) is worse than sampling's (1.3-2).
- **Why.** On every sampled input the future of a width-8 net is simple: zero bits past layer 20-48, a handful of
  classes from layer 8. The input is not: the map from the 8 inputs to a layer-20 class has 10^5 or more linear
  pieces, and a typical input direction lies 0.003-0.006 from the nearest piece boundary, in 7 angular dimensions. The
  exact methods tested must resolve the pieces. Sampling only has to land in them: plain MC's normalised error rate
  (nMSE = 1/N) does not depend on the dimension, and RQMC measured 1.3-2.1 bits per halving here.
- **What could change it.** Width 2-3: on one layer the code is exact, or 100 times better than RQMC (deep nets at
  width 2-3 were not tested; most of them die). A method that integrates a union of regions (a downstream class)
  without resolving it, which is the original problem. Or inputs that are themselves low-dimensional.
- **What the bits insight does buy at width 8.** The zero-bit tail, exact on every sampled input (and certified
  inside the cs2 code's tail cone). Used inside a sampler (RQMCJ, or the relevance lane's RADQ) it costs 55 mv a point
  instead of 262 and gains 3-15 times over RQMC at equal compute. That is the working form of "solve the deep part
  exactly".

## 11. Scorecard (registered; `results/scores.json`, `results/scores_a1.json`)

| test | prediction | outcome |
|---|---|---|
| T1: CS2 beats RQMC 3 times, 4 of 6 nets | fails | fails, 0 of 6 (best ratio 0.044) |
| T2: CS2 beats RQMCJ 3 times, 4 of 6 | fails | fails, 0 of 6 (best 0.0093) |
| T3: CS2 bits per halving above 2, 4 of 6 | holds | holds, 6 of 6 (3.6-8.5) |
| T4: integrate-and-drop region code loses to RQMCJ at every resolved R, 5 of 6 | holds | holds, 6 of 6 |
| T1b, T2b (CS2b) | fail | fail, 0 of 6 each |
| T3b (CS2b) | holds | holds, 6 of 6 (3.1-7.5) |
| T5: UNB beats RQMCJ 3 times, 4 of 6 | fails | fails, 0 of 6 (best 1.35) |
| T6: UNB beats RQMC 3 times, 4 of 6 | fails | holds, 5 of 6 (prediction wrong; section 9's ablation traces the gain to the truncated tail) |

## 12. Process and review

- Registration: Part 1 at 04:14:52Z, before any estimator ran on an evaluation net (the census and truths, which are
  descriptive and references, ran before it; first scored log 04:15:57Z); Amendment 1 at 04:20:55Z, before its runs
  (first log 04:24:56Z). Code frozen by sha256; the reviewer re-verified the hashes and the chronology.
- Development on dev nets 103 and 111 only (rule choice, cost per cell, the unbiased version's variance ratio).
- Post hoc and labelled so: the root-cell fix (CS2c), the one-layer study with that fix, the distance-to-boundary
  measurement, the preimage counts, the fixed-Sobol analysis, the UNB ablation, the line-crossing count, the figure.
- Compute: CPU only, at most two processes at nice 19, about 4 process-hours in total.
- **Review**, two rounds, gpt-6-astra via codex-any (high effort, read-only, fresh sessions; the "model:" line was
  checked both times; `review/review_gpt6.md`, `review/review_gpt6_r2.raw`). Both rounds reproduced the registered
  scores, the frozen hashes, the chronology and all nine tests; round 2 also reproduced all 66 fixed-Sobol errors and
  all 18 ablation means. Both verdicts were major revision. The findings and what changed:
  1. (critical) A deterministic baseline was missing. Round 1 found a fixed Sobol rule beating RQMCJ at N = 4096 on
     four nets. Added as section 8, checked against 64 scramblings in 66 (net, N) cells. Round 2 rejected my first
     reading ("one RQMC run with the randomness fixed", "ties sampling"): a mean rank near one half establishes
     neither equivalence nor a systematic gain. Now stated as competitive but inconsistent, with descriptive
     statistics only.
  2. T4 is not a lower bound: rephrased as one strategy (sample-ranked, integrate and drop) under an assumed cost; the
     oracle bracket was removed, and section 3's coverage sentence is limited to this strategy.
  3. The tail identity rests on a sampled premise: stated so in the summary and in sections 2 and 10. The evidence is
     separated by cut (48: census only; 64: census and the truth's check), the bias figure is labelled a sensitivity
     calculation under an amplitude assumption, and the census cost behind c = 48 is named as uncharged offline
     information.
  4. "Certified" and "exact" are numerical in the implementation: now "numerically certified" (double precision,
     no outward rounding), moments by quadrature (3.2e-9), UNB unbiased up to quadrature error; E|x| corrected to
     2.741625 (the code was right).
  5. The distance argument was a necessary condition it is not: now a rough isotropic-resolution model, with the
     inscribed-ball caveat. The one-layer study (closed-form mean at every width) is scoped to this code's efficiency.
  6. Amendment 1 did not fix root cells: stated. CS2c is identified as CS2b plus the root floor (root fix alone:
     6.7 times on 203, 3.7 on 217), and conclusions about convergence use its 3.3-4.3 bits per halving.
  7. Classes are open leaves, not constructed nodes. Certification covered a negligible mass, not none: 9 cells with
     5.8e-6 of the probability on 220 (CS2; 4 cells, 3.2e-6, for CS2b and CS2c).
  8. Slope windows: frozen values kept; `results/tables.md` now refits every method over the same nominal
     checkpoints. Also fixed: 223's RQMC and RQMCJ values at 2^24, the largest jump of the fixed rule's error (1,130
     times, not 2,000), and the comparison convention of the one-layer table.
  9. The cost ledger is now a nominal model, with known gaps listed and the unsupported "under 10%" removed; the
     ablation's no-cell arm is now charged its tail setup.
  10. Overclaims: the region code's gain over MC is reported (223's at 125,293 regions); T5 and T6 are scored outcomes
      with their uncertainty; the ablation now reports paired intervals (10 of 18 exclude zero) and supports, rather
      than settles, that the cells add nothing; the boundary-jump mechanism is labelled a hypothesis.
- **Round 3** (`review/review_gpt6_r3.raw`): minor revision. Round-2 findings 1, 2, 4-7 and 10 were confirmed fixed;
  the Sobol statistics, CS2c numbers, certification counts and all 18 paired intervals reproduced. Four minor items,
  all applied: the no-cell arm's setup charge is now stated as implemented (1,100-1,700 mv; the 512 mv of probes is
  uncharged); section 10 separates the 8,192-cell comparison (4.6-5.0 x 10^4) from the 131,072-cell certification
  count; the 7e-7 figure is a fixed-cut benchmark without guaranteed coverage, since the cut was chosen with the same
  census; "no unit-wise shortcut" from depth 2 is now "none found here", and the dimension-free rate is scoped to
  plain MC.
