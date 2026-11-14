# M1 at width 8: a two-sided sample-free estimator for plain ReLU nets of width 8, depth 256 (pre-registration, Part 1)

Written 2026-10-30 ~23:00Z, before any estimator result on an evaluation net. The sha256 and time in `prereg_hashes.txt`
are authoritative. Amendments are appended below with dates and their own hashes.

Ask (Kelvin, via main, 22:21Z): "Let us try to beat random sampling with width=8, and 256 layers. (a) we can use ARC's
method until layer 5-8 (right? they can do it until layer 8 right?) (b) we can use our own methods with few dimensions
after. Maybe discuss how to handle the middle." Bar: a major advance; M1 = a sample-free estimator that matches or beats
sampling at equal compute on deep random MLPs, at depths that grow with width.

## Seen before writing this (declared)
- Census of 128 nets (`results/census_table.md`, `scratch/census.py`): dev pool seeds 100-163 (12 healthy, 6
  intermediate, 46 dead-noise), evaluation pool 200-263 (15 healthy, 16 intermediate, 32 dead-noise, 1 dead-exact).
  With noise a "dead" net is not zero: its output is noise-driven (the input explains 0-11% of its variance).
- Oracle structure on dev nets 103 (2^14 and 2^16 paths), 110, 111, 129 (2^16, cuts up to 24 at writing time;
  `scratch/oracle_structure.py`):
  - downstream-distinct classes (noiseless future gate patterns) on dev 103: 4,910 / 2,388 / 991 / 259 / 37 / 6 / 2 at
    cuts 1 / 2 / 4 / 8 / 12 / 16 / 24 (100 classes cover 99% of the |a_L| mass at cut 8), against 242 / 2,408 /
    5,397 / 8,900 / 9,829 / 10,062 / 10,074 distinct upstream histories among 2^14 inputs (most of them singletons);
  - projected meet (value function evaluated on the top-k directions of the sensitivity M_c, rest by regression):
    nMSE at cut 8 with k = 3 / 4 / 5: 4e-3 / 3.6e-6 / 3e-8 (dev 103), 1.1e-7 / 1.4e-7 / 2e-8 (110), 1e-8 / 0 / 0
    (111), 2.6e-5 / 2.4e-6 / 1.2e-6 (129); at cut 16, k = 2 gives 2e-8 to 1.5e-5;
  - by cut 8, 31% of dev 103's inputs have passed a layer with exactly one active unit; 99% by cut 64.
- ARC's kprop on dev net 103 (sigma = 1e-3), nMSE of E[a_c] at c = 1..8 (`scratch/try_kprop.py`):
  K = 2: 5e-7, 5.4e-3, 0.024, 0.044, 0.069, 0.11, 0.13, 0.30; K = 3 (factorized): 5e-7, 5.1e-4, 8.6e-3, 0.11, 1.4,
  4.2, 53, 280; K = 4 (factorized): 5e-7, 1.8e-4, 1.2e-3, 3.4e-3, 0.02, 0.14, 0.43, 1.7; augmented K = 2 and 3 diverge
  from c = 5 (2e2 to 1.5e5). Costs 2.7e4 (K = 2) to 5.7e6 (K = 4) FLOPs by ARC's counter.
- One timing run of the truth script on evaluation net 203 (2^18 paths): tr Var 1.2e-45, |E a_L|^2 / tr Var = 1.69,
  p(1e-4 quantile) = 1.07e-4. No estimator has been run on any evaluation net.
- Coordinator addition (22:41Z): a scaling arm (below).

## Objects
- Nets: `nd.Net('plain', seed, 8, 256, sigma)` (`work/narrow-deep/lib/nd.py`, unchanged): W_l iid N(0, 2/n), no bias,
  x ~ N(0, I_8), h_l = ReLU(W_l h_{l-1}) + s_l eps_l for l < L, output a_L = ReLU(W_L h_{L-1}); s_l = sigma r_l, r_l the
  RMS of a_l in the noisy net from the seeded 2^14-path pilot (part of the net; nobody is charged). sigma = 1e-3
  primary, 1e-2 secondary; estimators and samplers use the same s_l.
- Classes (census rule, fixed): dead-exact = noisy output identically 0; dead-noise = every sampled noiseless input
  has passed an all-off layer by depth 256; healthy = at most 5% of noiseless inputs dead by 256 and the input
  explains >= 90% of the sigma = 1e-3 output variance; intermediate = the rest.
- Evaluation set: the first 6 healthy seeds of the evaluation pool in seed order (203, 216, 217, 220, 221, 223); the
  first 2 dead-noise (200, 201) and the first 2 intermediate (202, 206) as secondary arms (10 nets). Dev nets (100-163)
  are used for building and tuning only.
- Estimands: E[a_L] in R^8, scored by nMSE = |est - truth|^2 / tr Var(a_L); tail p = P(|a_L| > t) with t the
  (1 - 1e-4) quantile of |a_L| from a threshold set disjoint from the paths that estimate p; scored by relative MSE.
- Truth: plain MC, 2^24 paths for the 6 healthy nets at sigma = 1e-3, 2^23 for the other runs; its expected nMSE 1/P
  is subtracted from measured sampler MSEs and reported; ratios are claimed only where the method's nMSE is >= 10/P.
- FLOPs: flopscope conventions (matmul M N (2K-1), elementwise 1, normal draw 16, exp/log 16, cdf 96, pdf 54). The
  deductive method is charged for everything: forward, backward/value function, grids, interpolation, merging
  (compare/index = 1). ARC's kprop is charged by ARC's counter and, as a check, x1.3. One MC path costs
  `net.flops_path()` (about 7.0e4 FLOPs at n = 8, L = 256). Budgets are quoted in MC paths.
- Sample-free: no pseudo-random numbers in the estimator, and the estimate is not an average of the network's output
  over a point set in input space (that would be QMC). Deterministic grids for a low-dimensional value function,
  evaluated by deterministic passes from a cut, are allowed, and so are fixed 1D quadrature tables.

## Baselines (equal FLOPs; budgets N_eq = 2^7, 2^9, 2^11, 2^13, 2^15, 2^17, 2^20 paths)
- MC; RQMC (scrambled Sobol on the 8 input coordinates through the normal inverse CDF, iid noise); HYB, the previous
  lane's sample-the-shallow hybrid (MC to l0 in {16, 32}, 8 log-radius bins, K2-Stein deep part). Replicates R = 32
  up to 2^13, 16 at 2^15 and 2^17, 4 at 2^20 (the measured MSE's own standard error is reported). The best of MC and
  RQMC at each budget is "the sampler"; HYB is reported beside it.

## Methods
Forward part (layers 1..c, c in 1..8), each producing a description of the law of h_c:
- KP-K: ARC's kprop (`work/arc-noisy/kprop_noisy.py`, noise added exactly), K = 2 (simple, augmented), 3 (factorized
  simple, augmented), 4 (factorized simple).
- K2X: our exact-moment Gaussian closure (`work/narrow-deep/lib/k2.py`, exact bivariate ReLU moments).
- GHM(tau): gate-history mixture, the computable form of "an exact mixture over the realised gate patterns, pruned by
  mass": Gaussian components; at each layer every unit whose pre-activation straddles 0 (|mean| < 4 sd) splits the
  component into its exact on and off halves (single-halfspace truncated moments, closed form), re-Gaussianised;
  units are taken in order of straddle; a child with weight below tau is not split further and gets the exact K2
  ReLU moments instead; noise is added exactly to each component's covariance; tau in {1e-3, 1e-4, 1e-5}. (Exact
  pieces would need 8-dimensional polyhedral Gaussian integrals per piece; the moment-matching is the approximation.)
- Scoring of the forward part at cut c: (i) nMSE of E[a_c] against the truth's layer moments; (ii) the meet error
  e_c = |E_{law_c}[V_c] - E[a_L]|^2 / tr Var(a_L) with the true downstream V_c: for mixtures, 2^20 draws of the
  method's law pushed through the true noisy layers c+1..L (an oracle evaluation, not an estimator); for KP-K, the
  degree-K polynomial fit of V_c (oracle fit on 2^20 true paths) integrated against KP-K's moments of order <= K;
  (iii) FLOPs. "Accuracy of the layer-c law per FLOP" is (ii) against (iii), set beside RQMC's error on E[a_L] at the
  same FLOPs plus the downstream's cost.
Downstream (Kelvin's (b) and the middle):
- Deep part from c_d = 32: V_{c_d}(h) = E[a_L | h_{c_d} = h] on the attractor's relevant directions, radius exact by
  positive homogeneity: for h = rho v with v a unit vector, V(rho v) = rho G(v) + noise correction; G from the
  noiseless map (one deterministic pass), the noise correction from K2 (exact moments, noise included) started at
  point masses on a log-radius grid of 8 values along the reference ray.
- Middle (c to c_d), BVF: backward recursion of V_l on the top-k_l directions at each cut l = c_d - 1 .. c, k_l = the
  number of eigen-directions of a sample-free sensitivity estimate (Jacobians along deterministic probe paths from the
  forward description's component means, weighted by mass) needed for 99.9% of its trace, capped at 5; V_l stored on a
  grid over the (k_l - 1)-sphere of directions (radius exact), multilinear interpolation, dropped directions set to the
  forward description's mean; plus the direct variant (V_c evaluated at grid points by deterministic passes c -> c_d,
  no recursion). Grid resolutions 4 to 24 points per angular dimension.
- Meet at c: estimate = sum over forward components of their weight times E_{N(mu_j, C_j)}[V_c], the Gaussian
  expectation taken by a 3^k-point Gauss-Hermite rule in the k relevant directions on the interpolated grid. Variants:
  c in {2, 4, 6, 8}; merging forward components that fall in the same downstream class.
- Full sample-free estimator TS(c, F): forward method F to c, BVF downstream. The scored configuration per net is fixed
  on dev nets before the evaluation run (Amendment, hashed, if it changes).
- Downstream-distinct classes: counts of distinct noiseless future gate patterns at cuts 1..64 (2^16 inputs), against
  distinct upstream histories, with the counts covering 90% / 99% of the |a_L| mass.
- Practical arm, NOT sample-free (labelled as such): SH(c_h) = RQMC on layers 1..c_h, then the deterministic
  two-sided downstream (BVF at c_h). It answers "beat random sampling" in Kelvin's words, not M1.
- Tails: (i) TS's tail: P(|a_L| > t) from the forward mixture at c and the downstream (|a_L| = |V_c| along the
  relevant direction), each component's contribution by a 1D Gaussian integral; (ii) RB (radius integrated exactly,
  noise rescaled to a typical radius; a sampler) and SH-RB as sampling arms; MC and RQMC as baselines.

## Predictions and refutation conditions (healthy nets, sigma = 1e-3 unless stated)
P1 (Kelvin's (a)). ARC's kprop at width 8 does not carry the law to layers 5-8: for every K <= 4 the nMSE of E[a_c]
   exceeds 1e-3 at every c in 5..8 on all 6 healthy nets, and the meet error e_c exceeds RQMC's error on E[a_L] at the
   same total FLOPs at c >= 4 on >= 5 of 6. Refuted if some K <= 4 has e_8 <= 1e-4 on >= 4 of 6.
P2 (downstream classes). At cut 8 the classes covering 99% of the |a_L| mass number < 1,000 on all 6 healthy nets,
   against > 5,000 distinct upstream histories among 2^16 inputs; at cut 16, < 30 classes.
P3 (middle + deep). Given the exact (sampled) law at c = 8, BVF reaches nMSE <= 1e-5 with k <= 5 at a downstream cost
   <= 1,000 paths on >= 5 of 6 healthy nets. Refuted if it needs > 10,000 paths' FLOPs for 1e-5 on >= 3.
P4 (the forward part is the bottleneck). At every budget up to 1e6 paths, the best sample-free forward description
   (KP-K, K2X, GHM) at its best c has a meet error >= 10x RQMC's error on E[a_L] at the same total FLOPs, on >= 5 of 6
   healthy nets.
P5 (M1, the brief's criteria). (a) TS beats RQMC by >= 3x on E[a_L] at some budget on >= 4 of 6 healthy nets;
   (b) TS never loses by more than 2x at any budget up to 1e6 paths. Predicted: neither holds. If (a) fails, the
   two-sided approach fails at width 8 for E[a_L], and the error budget (forward at c, middle, deep, each against
   FLOPs) names the failing part. If (a) holds, the scaling arm runs.
P6 (practical arm, not sample-free). SH beats plain RQMC by >= 3x at equal FLOPs at budgets 2^11-2^17 on >= 4 of 6
   healthy nets (its saving is about L / c_h in cost per path).
P7 (tails). TS's tail estimate is off by more than 2x (relative) on >= 4 of 6 healthy nets; RB beats MC at equal FLOPs
   by >= 3x in relative MSE on >= 4 of 6.
P8 (secondary arms). sigma = 1e-2, dead-noise and intermediate nets: no pass/fail; the same tables are reported. We
   expect TS to be worse there (noise-driven states are wider).
Scaling (coordinator, 22:41Z). If P5(a) holds at width 8: rerun TS at widths 16 and 32 with L = 32 n (64 if affordable
   on CPU), 2 healthy nets each; prediction registered as requested: the advantage over RQMC at equal FLOPs stays
   constant or grows with width. Regardless of P5, the structure is measured at n = 8, 16, 32 (L = 32 n, 2 healthy nets
   each, oracle only): the middle's length (first cut with <= 3 downstream classes for 99% of the mass, and first cut
   with k*_F(M; 1e-2) <= 2), the relevant directions at cuts 4 and 8, kprop's error at layers 1-8, and the forward
   description's cost for a fixed meet error; we report whether that cost grows polynomially or exponentially in n.
   Prediction: the middle's length grows about linearly in n, k*_F at cut 4 grows with n (12 of 16 at n = 16), and the
   forward cost for fixed error grows exponentially in n.

## Amendment 1 (2026-10-30 ~23:35Z, before any estimator result on an evaluation net): backward-only arm (Kelvin, via main)
Kelvin, on the milestone note: "Ok, this is good news in a way; we can try go backwards from there?" Main asked for a
backward-only arm with the meeting point at layer 0, registered before results. Seen since Part 1 (dev nets only):
- dev 103, cut 8, c_d = 32: the deep part (J_ref + K2-Stein noise table from point masses at c_d) applied to the exact
  law at c_d leaves nMSE 5e-4; the fully noiseless downstream leaves 1.5e-3 (the noise moves E[a_L] that much);
  binned K2-Stein from the sampled law at c_d leaves 5e-3 (16 bins) to 1.3e-2 (8 bins); single-Gaussian K2 from the
  sampled moments at c_d is off by 15-140. Multilinear interpolation of V_8 on gnomonic grids is erratic (1e-4 to 2 by
  grid choice). So the downstream's noise treatment needs a better deep part; this is being fixed on dev nets.
- dev oracles, projected meet at cut 1 with k = 5: 7e-3 (103), 4e-7 (110), 3e-5 (111), ~0 (129).
Arm BO (backward only, meeting point 0):
1. Measurement (oracle, the 6 healthy evaluation nets): for every cut c = 0..8, the smallest k whose projected meet
   (value function evaluated on the top-k directions of S_c = Sigma_c^1/2 M_c Sigma_c^1/2, dropped part at its
   conditional mean; at c = 0 the dropped input directions are set to 0, their conditional mean) has nMSE <= 1e-5,
   and <= 1e-4. Both oracle directions (sampled Jacobians) and sample-free directions (probe Jacobians) are reported.
2. Estimator BO(k, m): relevant input directions U_0 from deterministic probe Jacobians (charged); E[a_L] ~ E_t[V_0(U_0 t)],
   t ~ N(0, I_k): radius exact by positive homogeneity (E chi_k for the noiseless part; a fixed radial Gauss rule over
   chi_k for the noise correction), the direction by a product midpoint rule in hyperspherical angles on S^(k-1) (m
   points per angle), each node evaluated by a deterministic noiseless pass to c_d and the deep part (the same deep part
   as TS and SH). No random numbers. Its nodes are input points of a k-dimensional slice, so it sits close to QMC; what
   separates it is the projection to k < 8 directions and the exact radial integral. It is reported as sample-free
   under that caveat.
3. Comparison at equal FLOPs (whole backward pass and probes counted) against RQMC and TS; and the best meeting point
   c in {0, 2, 3, 4, 8} by measured total cost for a fixed error, using the per-cut projected-meet floors and the
   forward descriptions' measured costs.
P9 (prediction). At the input, k for nMSE <= 1e-5 is >= 5 on >= 4 of 6 healthy nets (for <= 1e-4: >= 4); the needed k
   rises from cut 8 toward the input on >= 4 of 6. BO beats RQMC by >= 3x at some budget on at most 2 of 6 healthy nets.
   Refutation: BO beats RQMC by >= 3x at some budget on >= 4 of 6 healthy nets. If that happens, P4/P5's "the forward
   part is the bottleneck" is refuted in the sense that going backward to the input avoids it, and the scaling arm runs
   for BO. The pre-registered P1-P8 are kept unchanged.

## Amendment 2 (2026-10-30, written ~23:58Z, before any estimator result on an evaluation net): deep part and meet
Seen since Amendment 1 (dev nets only; `scratch/ds_diag.py`, `deep_diag3.py`, `deep_noise_effect.py`, `fwd_study.py`):
- The noise moves E[a_L] through the deep part (layers 33-256) by nMSE 1.5e-3 (dev 103), 3.9e-5 (110), 1.1e-3 (111) and
  1.9e-5 (129): noiseless deep part against MC of the noisy deep part from the same sampled states at c_d = 32.
- The K2-Stein noise table from point masses (the registered deep part) leaves 1.9e-4 (103) but 0.21 (110), 0.059 (111)
  and 0.056 (129): it overestimates the noise's effect 3-5x for small radii (MC check along the reference ray of dev
  103: relative effect 64 vs K2 204 at 0.15% of the median radius, 2.0 vs 9.2 at 3%, 0.046 vs 0.074 at the median);
  GHM(1e-3) from the point masses is no better (203, 9.1, 0.062). Binned K2-Stein from the sampled law at 32 leaves
  5e-3 to 1.3e-2 on dev 103.
- Multilinear interpolation of V_8 on gnomonic grids (k = 2-6, G = 8-32) is erratic: 2.5e-5 to 1.9 by grid choice.
- On dev 103, KP-2/3 and K2X meets: Gaussian-law meets 0.06-0.12 at c = 2 and 0.34-5,000 beyond; polynomial meets
  1.6e-3 at best (KP-3f, degree 2, c = 2) and 1e2 to 1e20 at c >= 4. KP-4f failed in our tower code (autograd), fixed.
Changes (all arms, evaluation nets):
1. Deep part: noiseless J_ref from the majority future class at c_d (D0), no noise table. Every deterministic arm (TS,
   BO, SH) therefore leaves out the noise's effect on E[a_L]; per evaluation net the deep noise effect is measured by
   the paired MC diagnostic above (an oracle measurement, not part of any estimator) and reported as their floor.
2. Meet: V_c is evaluated exactly (noiseless pass c -> c_d, then D0) at each forward component's mean, and at 2k points
   along the k relevant directions for components whose spread there exceeds 5% of their norm (`bvf.meet_direct`).
   The grid (BVF recursion) is not run on evaluation nets; its dev result above is reported.
3. BO: nodes are passed noiselessly to the output (BO-full, exact noiseless V_0 at each node); also BO with D0 at
   c_d = 64. Directions: per-probe-normalised probe sensitivity ('norm'), with the oracle directions as a reference.
4. Ladders, fixed now: TS: F in {GHM 1e-3, GHM 1e-4, K2X} x c in {2, 4, 8}, c_d = 32, k = 4; BO: k in {2..6} x m in
   {4, 8, 16} (at most 3e5 nodes); SH: c_h = 8, c_d = 32, direct variant.
5. Secondary arms (sigma = 1e-2; dead and intermediate nets): samplers, BO, TS (GHM 1e-3) and SH only, as compute
   allows; no forward study or oracles there.
P10 (new). The deep noise effect exceeds 1e-4 nMSE on >= 2 of the 6 healthy evaluation nets; on those nets no
   deterministic arm here stays within 2x of RQMC beyond 2^13 paths. P1-P9 are unchanged.

## Amendment 3 (2026-10-31 ~00:20Z, before any deterministic-arm result on an evaluation net): coordinator's two suggestions, tails, compute trims
Seen since Amendment 2: RQMC, MC and HYB on evaluation nets 203 and 216 (sampler baselines only; HYB's binned
K2-Stein deep part fails there: nMSE 1.4e-3 to 1e-2 on 203 and about 5 on 216); dev-net tests below.
1. Exact pieces (coordinator's suggestion 1), dev 103 (`scratch/ghm_shape_test.py`): GHM started from the exact
   first two moments of every layer-1 gate-pattern piece (oracle, 2^22 paths) still has E[h_c] error 5e-4 at c = 2 and
   7e-3 to 1e-2 at c = 8 (against 8e-4 and 1e-2 from the input). So the floor is the Gaussian shape of each piece at
   the next layers' boundaries, as the coordinator suggested, not the layer-1 moments. Price of exact pieces: one 8-D
   orthant probability by a deterministic rank-1 lattice (Genz's transformation, no random shift) reaches about 1% at
   2^14 points (410 MC paths' FLOPs) and 0.1% at 2^18 (6,500 paths); a piece needs the mass and about 36 lower-
   dimensional orthant terms for its first two moments, and layer 1 alone has 255 pieces (layer 2: 2.4k-5k pieces, up
   to 16 facets each). Exact pieces therefore cost >= 1e5 MC paths per layer before any accuracy gain; they are not run
   on evaluation nets.
2. Deep-part noise (coordinator's suggestion 2), dev 103 (`lib/gsf.py`, `scratch/gsf_test.py`): a deterministic
   Gaussian-sum filter (narrowing splits, merging, exact Gaussian noise each layer) gets the noise effect at the median
   radius within 0.1-5% of MC (K2: 60% off), but at 3% and 0.15% of the median radius it is still 1.8-3.6x too large
   (noise-driven regime), and its table costs 6.9e4 (kmax 16) to 4.9e5 (kmax 48) paths. Applied to the exact law at
   c_d = 32 it leaves nMSE 2.4e-4 and 1.3e-4 (noiseless deep part: 1.7e-3). Decision: D0 (noiseless deep part) stays
   for the evaluation runs; the GSF result is reported as tested.
3. Tails added (registered P7 needed a TS tail; none was coded): TS tail = sum over components of a Gaussian in the
   output norm (mean |V(mu_j)|, radial spread from C_j scaled by the gain |V(mu_j)|/|mu_j|); BO tail = sum over nodes of
   Q_chi_k(t / |V_0(U theta_j)|) (radius exact in the projected problem).
4. Compute trims (CPU is shared and slower than planned): the forward study on 4 healthy nets (203, 216, 217, 220); the
   per-cut direction oracle on all 6 (2^16 paths for 203 and 217, 2^15 for the rest); sigma = 1e-2 on 203 and 216; the
   secondary arms on 200 (dead-noise) and 202 (intermediate); truths at sigma = 1e-2 only for 203 and 216; the scaling
   structure at n = 16 and n = 32 on one net each.
P1-P10 unchanged; P1 and P4 are scored on the 4 nets with the forward study.
