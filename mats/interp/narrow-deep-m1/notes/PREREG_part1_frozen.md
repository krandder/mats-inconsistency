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
