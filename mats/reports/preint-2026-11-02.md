# Pre-integration and the sampling-side asymptote for deep ReLU nets (2 November 2026)

Status: final after three gpt-6-astra review rounds (major, major, minor revision; 16 + 9 + 9 findings; responses in review/RESPONSE.md, section 8). All scored runs complete under PREREG Part 1 and Amendments 1-2.

Lane preint (target M1), bead pool-ttdzrk.17 under pool-ttdzrk. Work directory `work/preint/`: `lib/line.py` (exact
line + radius integration), `lib/est.py` (estimators), `lib/nets.py`, `tests/test_line.py` (9 checks, all pass,
`logs/test_line.log`), `scratch/` (runs, scorers, tables, report assembly), `runs/`, `results/` (`scorecard.md`,
`scores.json`, `tables.md`, `pieces.md`, `degree_table.md`, `kernel_bound.json`, `posthoc_matched.json`,
`cost_sensitivity.json`), `FORMAL.md` (theory v1 frozen 14:00:39Z before the co-prover's first answer; additions
E1-E5), `PREREG.md` (Part 1 hashed 14:03:47Z), `PREREG_A1.md` (15:08:43Z) and `PREREG_A2.md` (15:34:02Z),
`prereg_hashes.txt`, frozen code in `frozen/`, `coprover/` (gpt-6-astra rounds 1-2, `PROVENANCE.md`), `review/`,
`notes/` (QUESTIONS.md, READING.md, this report's template), `logs/deviations.txt`. CPU only, at most two heavy
processes at nice 19, ionice idle. Labels: PROVEN, PROVEN UNDER <hypothesis>, NUMERICAL, MEASURED, HEURISTIC,
CONJECTURE; [A] = gpt-6-astra, [C] = me (Claude).

Ask (Kelvin, 2 November about 13:15Z): "If we can find something that we can at least hope will lead to estimating the
mean at better assymptote per precision, that is what we should seek." Brief from main: test pre-integration
(integrate one direction exactly, then RQMC over the rest) as a route to an MSE-vs-compute exponent above RQMC's that
holds as width grows; replace it if something better turns up.

## Summary for Kelvin

**Short answer: no for pre-integration as built; the wider question stays open. Exact pre-integration raised the
fitted exponent clearly where the input dimension was tiny (n = 4-8: by 0.1-0.3), but it lost at equal compute at
every common budget we ran: the best PI variant was worse than the reference by factors of 1.2-110 on those nets,
15-716 at 8 x 256 (against RQMCJ) and 21-2,956 along the L = n/8 ladder up to 256 x 32 (against RQMC). On the wider
nets its registered fits were also steeper, but over much smaller sample counts; at matched sample counts our finite
windows show no consistent exponent difference (inconclusive about equivalence and about asymptotes). What we can
prove about sampling in general is narrower and differs by information type: averaged over random nets, rules that
average point values with net-independent nodes and scalar weights have error at least (c/2)/N, relative to the
averaged variance, up to N ~ n^r/r!, when a spectral share c holds (measured on two nets per width, growing with
width, not proven); rules built from exact line integrals have only a degree-by-degree certificate and an absolute
bound for N <= n/4; adaptive or nonlinear methods are not covered, and I found no candidate among them.**

1. **Cost of one exact line** (this algorithm; expectation PROVEN [A]; MEASURED). Following a line exactly means
   following every gate flip along it, and each unit flips on average once per layer along a half great circle: about
   nL vertices and nL/2 forward passes per line (256 x 32: 3,469 passes, 752 for a window holding all but 1e-8 of the
   weight, 836-1,563 in the scored runs). Narrow-deep 8 x 256 saturates at 100-170 vertices because inputs
   synchronise. Pruning redundant vertices could lower this; I did not try.
2. **Value of one line** (MEASURED at equal sample counts, post hoc). In equivalent RQMC-RB points per line: 0.9-1.7
   at n = 32-128 and 0.84-1.29 at 256 x 32, for 41-1,563 passes; 0.9-7.9 RQMCJ points per PIJ line at 8 x 256 for 15-36
   passes; 1.6-7.7 points on the n = 4-8 nets for 46-90 passes.
3. **Registered results** (PREREG hashed before any scored run; two amendments cut the 256 x 32 budgets; scorecard in
   section 7). Equal-cost losses (P3a, P5a hold; per-net minima over common budgets 21-1,410x on the ladder and at
   256 x 32, 15-91x at 8 x 256; 1.8-13x for PI_as at the largest common budget of the small nets). Exponents: on the small nets PI's exceeds RQMC-RB's by more
   than 0.1 on 8 of 9 (P4 holds); at 8 x 256 they agree within 0.3 (P5b holds); on the ladder and at 256 x 32 the
   registered exponent tests compare fits over different sample ranges (P3b fails (4 of 7), P3c fails); at equal sample counts
   the slopes show no consistent difference (no equivalence test).
4. **What pre-integration does** (theory with gpt-6-astra). It smooths: with the line transverse to every kink the
   integrand becomes C^{1,1}, with no monotonicity condition (piecewise-linear folds are harmless). For a fixed net an
   interpolation-plus-residual algorithm then attains MSE N^{-1-5/(d-2)} instead of N^{-1-3/(d-1)} [A]: an upper bound
   for that algorithm (not a rate established for our Sobol rules), a gain of about 2/d; under power-law models the
   break-even compute is about c^{d/2} for a line costing c passes.
5. **The restricted barriers** (PROVEN, scope as stated). Averaged over random nets, for net-independent nodes or
   lines and scalar weights (signed allowed): (a) absolute squared error >= 1/(4 pi^2 N) for N <= n/2 points and
   >= 1/(8 pi^2 N) for N <= n/4 lines, at every depth [A]; (b) for point rules only, error relative to the averaged
   variance >= (c/2 - o(1))/N up to N ~ n^r/r! when the even angular energy above degree 2r carries a fraction c of
   that variance [C, repaired by A]; (c) for line rules, a degree-by-degree LP certificate that needs individual degree
   energies [C, A]. Measured on two nets per width, the share of even degrees >= 4 grows from 0.08 (n = 32) to 0.26
   (n = 256): descriptive, not the ensemble certificate (b) needs.
6. **Beyond.** Under a Gaussian surrogate prior, adaptive and nonlinear algorithms cannot beat the best nonadaptive
   linear rule (classical, re-proved [A]); a quantitative bound for the nets' vector outputs needs per-channel kernels
   that I did not derive. The real prior is far from Gaussian (excess kurtosis 7-9 per output unit), so for it
   adaptive and nonlinear methods are neither excluded nor shown to help; my earlier claim that about nL queries
   identify the net is withdrawn [A]. The one constructive route I can state is a white-box surrogate family with
   exact means whose residual variance falls with its cost uniformly in width [A]; none is known, and controls built
   from squared hinges (such as the exact first-layer norm) provably cannot touch the even degrees >= 4 [A].

## 1. Setup
- Nets: the ARC family (W_l iid N(0, 2/n), square, no biases, input dim d = n, output h_L, last layer a ReLU) with
  arc-noisy's weight stream: ARC eval nets 201-204 at 256 x 32 (truths 2^26 paths; net 200 was used for vertex
  counts), the arc-update ladder at L = n/8 (seeds 302-304; truths 2^24 paths); narrow-deep 8 x 256 (nd-exact stream,
  seeds 203, 216, 217, 220, 221, 223; truths J_64 E[h_64]); small nets n = 4, 6, 8 at L = n/2 (ARC stream, seeds
  11-13; no truths). Dev nets (choices only): ARC stream seeds 1 and 9500, narrow-deep 103.
- Score: nMSE = |est - E f|^2 / tr Var f (MC with N paths: 1/N). All estimators are unbiased (RQMCJ and PIJ under
  nd-exact's empirical tail premise), so the scored MSE is the across-replicate variance (8-16 independent
  randomisations; with 8 replicates its relative standard deviation is about 50%). Each run also reports a bias ratio
  |mean - truth|^2 / (replicate variance / R); it is not a calibrated z-score (truth noise omitted, few replicates).
  Values above 16 occurred in 3 of 888 registered cells; for a rank-one Gaussian toy the rate would be 0.5% per cell
  (t_7 tails), an illustration, not a calibration of this vector statistic.
- Compute: counted FLOPs over one forward pass (2 n^2 L) under a nominal model: every matvec, sign test, new-vertex
  interpolation, two Student-t cdfs (200 FLOPs each) and one power per final vertex, inverse normals, pilots and basis
  maps. The code as written evaluates four cdfs and two powers per piece, and the ray arithmetic and crossing sort are
  not charged; PI_as's pilot SVD is charged 4 d P^2, more than a thin SVD needs when P = 64 > d; MLMC's import charges
  its prefix passes but not its draws (not adjusted below). An allowance-based scenario approximating what the line
  code does (scenario A: four cdfs and two powers per piece plus 20 FLOPs per vertex for sorting and ray work;
  results/cost_sensitivity.json, all 56 PI records with their registered comparators) raises PI's cost by 1.54-1.83x on
  the n = 4-8 nets, 1.07x at 8 x 256, 1.10x at n = 32, 1.013-1.014x at 64 and at most 1.002x at n >= 128; fitted
  exponents move by at most 0.017 (about 0.026 with the cdf priced at 400 FLOPs, 0.033 when PI_as's SVD is charged the
  thin-matrix cost); PI stays worse than its comparator at every common cost in all 56 records and all scenarios. Sobol generation is not
  charged for any method. Same nets, truths and pass unit as the randomness and arc-update lanes.
- Estimators: MC; RQMC (scrambled Sobol, inverse normal; arc-update's); RQMCRB (the same on the sphere, radius exact);
  RQMCJ (8 x 256: RQMCRB through 48 layers then nd-exact's tail map, its 64-point pilot charged); MLMC (two-level
  shared-trajectory sampler, unrestricted lane, allocation pilot charged); PI (this lane): exact line + radius integrals
  on 2^p lines with bases from scrambled Sobol in d - 1 dimensions; u random per replicate (PI_rand) or the top
  eigenvector of a 64-point pilot estimate of E[J^T J] (PI_as, Liu-Owen's active subspace; pilot charged; this u depends
  on the net); _w = window mode on the ladder and at 256 x 32 (exact on the window holding all but 1e-8 of the weight,
  one-pass unbiased tail estimate; the tail adds an MC term of order eps^2 times the tail's relative output variance per
  line, about 1e-16 by that heuristic (not measured), and an eventual C_eps/N floor); PIJ = PI through 48 layers then the tail map (applied per vertex; hoisting it after the
  integral would save about 1% and is not done; PIJ_as's pilot targets the 48-layer prefix, not the output).
- Scored runs execute frozen/curves.py, which imports the live lib/; the current lib/line.py and lib/nets.py match
  frozen/ byte for byte and lib/est.py differs only by a post-hoc direction option never used by scored runs, checked
  retrospectively (the modules each run loaded were not recorded; logs/deviations.txt).

## 2. The exact line + radius integral and what it costs

**Identity (F1; PROVEN [C], checked [A]).** With x = rho (yhat + s u), yhat uniform on the unit sphere of u-perp,
positive homogeneity gives E[f(x) | yhat] = int f(yhat + s u) w(s) ds with w(s) = ((d-1)/sqrt(2 pi)) (1 + s^2)^{-(d+1)/2}
(a scaled Student-t density). One exact line therefore also integrates the radius: a two-dimensional half-plane for
the price of a line; the outer rule runs on the sphere S^{d-2}. Conditioning on yhat rather than y never increases the
variance.

**Algorithm (F2; PROVEN [C], checked [A]; tests).** The net restricted to the half-plane is piecewise linear on cones.
Carry the vertices of the half great circle from -u to +u through the layers: one matvec per vertex per layer; a
strict sign change of a unit inside a segment adds a vertex at the chord point where its pre-activation vanishes
(interpolated, no matvec); after the last layer the integral is a sum of closed forms in the Student-t cdf. Exact in
real arithmetic (window mode: exact inside, unbiased outside). Tests (9, all pass): fine-grid quadrature on single
lines (relative error 1.5e-10, the quadrature's own), constant activation pattern inside all 203 checked pieces,
depth-1 closed form (|z| = 1.25), a 6 x 10 net against 2^21-path MC (|z| = 1.82), window against full mode (1.3e-10),
the window's tail estimate unbiased (|z| = 0.51), and, added after review, chunk invariance (exact), linearity in a
supplied tail map (4e-16; this does not test nd-exact's tail premise), coincident roots (identical, scaled and opposite
rows; 6.5e-12 against quadrature) and the FLOP counter against the nominal formula (exact; it checks the counter, not
the code's actual operation count).

**Cost law (F3 + E2(c); PROVEN in expectation).** Conditional on the previous layer, each unit's pre-activation along a
weight-independent arc is a Gaussian process, and Kac-Rice gives E[# new strict sign changes | past] = (length of the
normalised hidden curve) / pi [C]. A fresh He-ReLU layer preserves the expected length exactly whenever at least two
units are active [A]: the expected number of new vertices at layer l on a half circle is n q_n^{l-1},
q_n = 1 - (n+1)/2^n. So this algorithm keeps about 3 + nL vertices and costs about nL/2 passes per line (expected
3,971 at 256 x 32; 205 at 8 x 256). Measured (results/pieces.md; 12 nets, n = 8-256, L <= 64, 6-64 lines each, random
u): 0.76-1.39 times 3 + nl (a heavy-tailed product), exactly 3 + n after layer 1; at 256 x 32 a full line costs 3,469
passes and a window line 752 (836-984 in the scored PI_rand_w runs, 1,370-1,563 with the active-subspace u, which is
net-dependent and crosses more gates); 8 x 256 saturates at 103-172 vertices and a PIJ line (48 layers) costs 15-36
passes, against 0.2 per RQMCJ point. Inherited vertices can become redundant downstream, so the minimal fan of f along
a line may be smaller: this is the cost of this algorithm, not a lower bound for exact integration. Registered P1:
window vertex counts within [0.6, 1.5] of the law on fresh nets (section 7).

## 3. What pre-integration does to smoothness, and the rate it predicts
- **Smoothness (F4; PROVEN [C], sharpened [A]).** If u is transverse to every facet, the pre-integrated function is
  C^{1,1} (W^{2,inf} locally; it grows linearly). Its Hessian is sum_j v_j (P a_j)(P a_j)^T p(t_j(y)) / |a_j . u| over
  the facets the line crosses (v_j a_j^T the Jacobian jump across facet j). No monotonicity is needed:
  Gilbert-Kuo-Sloan's failure mode is a smooth fold, where two crossings merge like a square root; at a piecewise-
  linear fold they merge linearly and contribute O(distance^2). C^{1,1} is sharp [A]. For a fixed finite fan f is
  H^{3/2-} and the pre-integrated function H^{5/2-} [A]; in an orientation-averaged Poisson-kink model only half a
  derivative is gained (H^{3/2-} to H^{2-}) [C, A].
- **Rates (E2(b), upper bounds).** For a fixed net an interpolation-plus-residual cubature (a different algorithm from
  ours) attains MSE N^{-1-3/(d-1)} on the sphere and N^{-1-5/(d-2)} after line + radius [A]; the Poisson model suggests
  N^{-1-4/(d-2)} [C, HEURISTIC]. Neither is established for our rules: astra's o(1/N) argument assumes ideal
  nested-uniform scrambling with bounded gains, not shown here for SciPy's LMS scrambling plus digital shift, and the
  window estimator has an eventual C_eps/N term; our lower measured slopes do not by themselves show that the runs are
  pre-asymptotic. The exponent gains are about 2/d (constructive) and 1/d (Poisson model); under power-law models with comparable prefactors the break-even compute for a line costing c passes is
  about c^{d/2} [A].
- **What one line is worth (F6, HEURISTIC).** A line removes the variance of angular components that vary along u,
  roughly D/d of it for mean angular degree D.

## 4. Measurements (PREREG Part 1 and Amendments 1-2; scorecard in section 7)

Exponent = minus the slope of log nMSE on log cost over the top 60% of each curve's budgets, at least four points
(frozen fit protocol). "Matched N" columns are post hoc (after co-prover round 2): slopes per sample count over the
same N values, and the equivalent number of points per line r^{1/alpha}, with r the ratio of the point rule's nMSE at N
points to PI's at N lines and alpha the point rule's per-N slope there.

### 4.1 Small nets (n = 4, 6, 8; L = n/2; 16 replicates): where pre-integration should help

| net (n x L, seed) | RQMC | RQMCRB | PI_as | PI_rand | constructive upper-bound exponents RB / PI (another algorithm) [A] | PI_as / RQMCRB at equal cost (largest common budget) | matched N: slope per N, PI_as vs RQMCRB | equivalent points per line (matched N) | passes per line |
|---|---|---|---|---|---|---|---|---|---|
| 4x2 s11 | 1.73 | 1.71 | 1.98 | 1.84 | 2.00 / 3.50 | 2.2 | 1.96 vs 1.60 (N 64-16384) | 6.1 | 89 |
| 4x2 s12 | 1.75 | 1.68 | 1.82 | 1.97 | 2.00 / 3.50 | 1.8 | 1.89 vs 1.76 (N 64-16384) | 7.7 | 82 |
| 4x2 s13 | 1.65 | 1.67 | 1.93 | 1.92 | 2.00 / 3.50 | 2.0 | 1.86 vs 1.69 (N 64-16384) | 6.3 | 89 |
| 6x3 s11 | 1.42 | 1.51 | 1.63 | 1.55 | 1.60 / 2.25 | 3.2 | 1.63 vs 1.44 (N 64-16384) | 5.7 | 61 |
| 6x3 s12 | 1.49 | 1.50 | 1.55 | 1.50 | 1.60 / 2.25 | 7.0 | 1.55 vs 1.38 (N 64-16384) | 4.8 | 61 |
| 6x3 s13 | 1.56 | 1.52 | 1.68 | 1.71 | 1.60 / 2.25 | 4.2 | 1.66 vs 1.50 (N 64-16384) | 3.7 | 59 |
| 8x4 s11 | 1.23 | 1.28 | 1.42 | 1.36 | 1.43 / 1.83 | 10.9 | 1.40 vs 1.37 (N 64-8192) | 2.3 | 46 |
| 8x4 s12 | 1.26 | 1.27 | 1.54 | 1.42 | 1.43 / 1.83 | 13.4 | 1.51 vs 1.44 (N 64-8192) | 2.0 | 52 |
| 8x4 s13 | 1.43 | 1.37 | 1.58 | 1.48 | 1.43 / 1.83 | 7.0 | 1.55 vs 1.31 (N 64-8192) | 3.6 | 46 |

- Registered P4 holds, 8 of 9: PI_as's exponent exceeds RQMC-RB's by more than 0.1 on 8 nets (0.27, 0.14, 0.26 at
  n = 4; 0.12, 0.05, 0.16 at n = 6; 0.14, 0.27, 0.21 at n = 8). At matched N the per-sample gain is 0.13-0.36 at n = 4,
  0.16-0.19 at n = 6 and 0.03-0.24 at n = 8: it shrinks with dimension, as section 3 suggests.
- At equal compute PI still loses at every budget tested: by 1.8-2.2x at n = 4, 3-7x at n = 6 and 7-13x at n = 8 at the
  largest common budget (10^5-10^6 passes), with the gap closing. Extrapolating the fitted exponents (HEURISTIC) puts
  the crossover near 10^7 passes at n = 4 and beyond 10^10 at n = 8; charging what the code does (section 1) pushes
  both further. A line costs 46-89 passes here, mostly cdf and bookkeeping (a pass of these nets is 64-512 FLOPs).

### 4.2 Narrow-deep 8 x 256 (6 nets, 8 replicates; lines through 48 layers then nd-exact's tail map, exact under its empirical one-class premise)

| net | RQMC | RQMCRB | RQMCJ | PIJ_as | PIJ_rand | MLMC_32 | nMSE at 8192 passes: RQMCJ / PIJ best / MLMC | min PIJ / RQMCJ at equal cost | matched N slope PIJ_as vs RQMCJ | equivalent points per line (matched N) | passes per PIJ line |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 203 | 1.23 | 1.08 | 1.22 | 1.23 | 1.07 | 1.00 | 1.4e-06 / 3.3e-05 / 9.8e-06 | 14.6 | 1.21 vs 1.40 | 7.5 | 20 |
| 216 | 1.20 | 1.21 | 1.31 | 1.36 | 1.22 | 0.84 | 7.8e-07 / 7.0e-05 / 2.2e-05 | 19.9 | 1.35 vs 1.00 | 3.9 | 20 |
| 217 | 1.26 | 1.25 | 1.18 | 1.06 | 1.16 | 0.48 | 1.6e-06 / 1.0e-04 / 2.4e-05 | 15.5 | 1.04 vs 0.82 | 7.9 | 25 |
| 220 | 1.03 | 1.43 | 1.37 | 1.31 | 1.14 | 1.40 | 7.9e-07 / 2.5e-04 / 7.5e-06 | 91.1 | 1.29 vs 0.88 | 2.3 | 35 |
| 221 | 1.37 | 1.36 | 1.24 | 1.11 | 1.05 | 1.28 | 1.5e-06 / 2.2e-04 / 1.6e-05 | 31.5 | 1.11 vs 1.09 | 4.4 | 32 |
| 223 | 1.26 | 1.42 | 1.05 | 1.23 | 1.21 | 1.06 | 1.5e-06 / 1.2e-04 / 1.3e-05 | 19.3 | 1.21 vs 1.40 | 3.1 | 15 |

- Registered P5a holds, 6 of 6: PIJ is worse than RQMCJ at every common cost, by at least 15-91x. P5b holds, 6 of 6:
  fitted exponents differ by less than 0.3 (RQMCJ 1.05-1.37, best PIJ 1.11-1.36). With about 10^5 activation regions
  over 7 angles every quadrature cell holds many kinks, and the slopes are noisy (standard errors 0.1-0.25). At matched
  N, PIJ_as's per-sample slope is above RQMCJ's on four nets (one by 0.02) and below on two.
- MLMC (two-level, r = 32) sits between: 7-28x worse than RQMCJ at 8,192 passes, 3-33x better than PIJ.

### 4.3 The ladder at L = n/8 and ARC 256 x 32 (8 replicates; 256 x 32 PI budgets cut by Amendments 1-2)

| net | RQMC | RQMCRB | PI_as_w | PI_rand_w | MLMC | N x nMSE of RQMC at its last 3 budgets | min PI / RQMC at equal cost | matched N slope PI_as_w vs RQMCRB | equivalent points per line (matched N) | passes per line (as / rand) | window vertices / law | line variance ratio rand / as |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32x4 s302 | 1.02 | 1.12 | 1.36 | 1.32 | 1.16 | 0.14, 0.15, 0.17 | 26 | 1.34 vs 1.30 | 1.7 | 49 / 41 | 0.94 | 0.88 / 0.61 |
| 32x4 s303 | 0.95 | 1.01 | 1.18 | 1.21 | 1.20 | 0.19, 0.23, 0.26 | 21 | 1.16 vs 1.26 | 1.4 | 50 / 42 | 1.02 | 0.90 / 0.74 |
| 32x4 s304 | 0.99 | 1.08 | 1.15 | 1.13 | 1.22 | 0.21, 0.31, 0.23 | 22 | 1.14 vs 1.12 | 1.5 | 54 / 42 | 1.07 | 0.90 / 0.71 |
| 64x8 s302 | 1.07 | 1.03 | 1.07 | 1.45 | 1.15 | 0.36, 0.23, 0.21 | 128 | 1.09 vs 1.13 | 1.0 | 143 / 99 | 0.89 | 0.91 / 0.63 |
| 64x8 s303 | 0.99 | 1.02 | 1.25 | 1.34 | 1.32 | 0.35, 0.34, 0.41 | 108 | 1.31 vs 1.17 | 1.3 | 143 / 97 | 0.89 | 0.93 / 0.77 |
| 64x8 s304 | 1.09 | 1.10 | 1.32 | 1.06 | 1.17 | 0.29, 0.22, 0.34 | 109 | 1.44 vs 1.27 | 1.1 | 147 / 99 | 0.89 | 0.91 / 0.74 |
| 128x16 s302 | 0.97 | 1.07 | 1.15 | 1.01 | 1.15 | 0.35, 0.43, 0.41 | 386 | 0.94 vs 1.38 | 1.4 | 419 / 271 | 0.81 | 0.92 / 0.83 |
| 128x16 s303 | 1.07 | 1.07 | 1.08 | 0.98 | 1.23 | 0.47, 0.65, 0.42 | 308 | 0.68 vs 1.12 | 1.4 | 552 / 345 | 1.20 | 0.90 / 0.79 |
| 128x16 s304 | 1.05 | 1.02 | 1.21 | 0.91 | 1.36 | 0.49, 0.46, 0.54 | 361 | 1.19 vs 1.05 | 1.3 | 485 / 317 | 1.01 | 0.91 / 0.83 |
| 256x32 s201 | 1.11 | 1.04 | 0.95 | 0.62 | 0.99 | 0.62, 0.42, 0.42 | 647 | 0.95 vs 0.99 | 1.0 | 1370 / 836 | 0.88 | 0.98 / 0.85 |
| 256x32 s202 | 1.13 | 1.10 | 1.11 | 0.93 | 1.13 | 0.44, 0.63, 0.32 | 1067 | 1.12 vs 1.27 | 1.1 | 1541 / 972 | 1.07 | 0.80 / 0.75 |
| 256x32 s203 | 1.19 | 1.08 | 0.76 | 1.23 | 1.30 | 0.44, 0.56, 0.39 | 1410 | 0.76 vs 0.77 | 1.1 | 1504 / 932 | 1.02 | 0.95 / 0.88 |
| 256x32 s204 | 1.03 | 0.98 | 1.05 | 1.28 | 1.05 | 0.59, 0.52, 0.71 | 766 | 1.04 vs 0.95 | 1.3 | 1563 / 984 | 1.11 | 0.81 / 0.87 |

- Registered P3a holds: PI is worse than RQMC at every common cost on every net, by at least 21-26x at n = 32,
  108-128x at 64, 308-386x at 128 and 647-1410x at 256 (the cost per line grows like L sqrt(n) for windows and like
  nL for full lines, while a line stays worth about one point). P1 holds (13 of 13) (window vertex counts 0.81-1.20 of the law).
  P2 holds (line variance ratio, as registered, 0.80-0.98 with a random u and 0.63-0.88 with the active-subspace u; the registered statistic omits the variance of the line
  average, and adding it back raises these by up to 0.035, most at the amended 256 x 32 nets' 32-line budgets).
- Registered P3b fails (4 of 7) and P3c fails: PI's exponents, fitted over its own top-60% window, exceed RQMC's by more than
  the registered margins at n = 32 on 3 of 3 nets (predicted), at n = 64 on 2 of 3 (predicted at most 1) and at
  n >= 128 on 3 of 7. The registered comparison is flawed: PI's window sits at much smaller N (its lines cost
  40-1,600 passes), where curves tend to be steeper, while RQMC's sits on its plateau. Post hoc, at matched N, slopes are PI_as 1.14-1.34 vs RQMC-RB 1.12-1.30 at n = 32;
  1.09-1.44 vs 1.13-1.27 at 64; 0.68-1.19 vs 1.05-1.38 at 128 (three sample counts only; regression standard errors up
  to 0.36); PI_as 0.76-1.12 vs RQMC-RB 0.77-1.27 at 256 (point references at N = 2-64 from post-hoc runs with 16 replicates). No consistent
  sign; this is not an equivalence test (regression standard errors only; replicate vectors were not saved). The
  follow-up PREREG named for a P3b failure, larger N at n = 128 (post hoc; one extra budget, 2^9 lines, 8 replicates,
  on net 302 only; stopped because of machine load, net 304 not run): net 302: PI_as_w exponent 1.03 over 2^2..2^9 lines (registered window: 1.15), RQMC 0.97. With the fit window shifted from
  16-256 to 32-512 lines the gap to RQMC's exponent shrinks from 0.18 to 0.06, consistent with the window explanation
  but not establishing it; the new budget lies beyond RQMC's cost range, so it adds no equal-cost comparison.
- RQMC sits on a plateau: N x nMSE over its last three budgets is 0.14-0.31 at n = 32, 0.21-0.41 at 64, 0.35-0.65 at
  128 and 0.32-0.71 at 256 (P7 predicted a local exponent <= 1.3 on at least 10 of 13 nets: it holds (12 of 13)). MLMC's fitted exponents are partly inflated by its allocation pilot at
  small budgets (P8 fails as scored on the registered budgets, 11 of 19; registered-budget exponents 0.99-1.36 on the
  ladder and at 256 x 32, 0.48-1.40 at 8 x 256).

## 5. What can and cannot be proved about the sampling side at large width

**One line is worth about one point** (section 4): at matched N, 0.9-1.7 RQMC-RB points at n = 32-128 and 0.84-1.29 at
256 x 32, while costing 41-1,563 passes. As registered, a line removes 2-20% of the variance with a random u and
12-37% with the active-subspace u at n >= 64 (the largest removals are at the 32-line budgets of the amended 256 x 32
runs, where the registered statistic is biased low).

**A rank barrier for rules that average sampled values (F7, E5; PROVEN [C], audited and repaired [A]).** Average over
random nets, with nodes, line directions and scalar weights chosen independently of the net (PI_as and RQMCJ, whose
choices depend on the net, are outside; RQMC, RQMC-RB, lattices, designs and PI with a random u are inside). The
weight-averaged second moment K(x, x') = E_W f(x) . f(x') equals |x||x'| kappa(<xhat, xhat'>) at every width (rotation
invariance of W_1; homogeneity), and a rule's expected squared error is its kernel discrepancy to the Gaussian. With
kappa = sum_k a_k P_k (a_k >= 0, the energy of angular degree k):
- Rank lemma: for N points with any real weights and any j, the degree-j harmonic feature matrix of the rule has rank
  <= N, and (tr A)^2 <= rank(A) ||A||_F^2 holds for every real symmetric A, so sum_k pi^{(j)}_k Q_k >= s^2 (1/N - 1/N_j)
  (Q_k >= 0 the rule's degree-k discrepancy, pi^{(j)} the Gegenbauer coefficients of P_j^2, s the rule's radial mass,
  N_j ~ d^j/j! the dimension of degree-j harmonics). For N line + radius integrals the rank is <= N (j + 1), or
  <= 1 + N j with a common u [A].
- LP certificate: E MSE >= m^2 a_0 B / (a_0 + B) with B = sum_j alpha_j (1/r_j - 1/N_j)_+ for any alpha >= 0 with
  sum_j alpha_j pi^{(j)} <= a coefficientwise (r_j = N for points, N (j + 1) for lines). It needs individual degree
  energies: a signed rule on 3d - 2 nodes integrates every degree-4 harmonic exactly [A] (checked numerically), so an
  aggregate even energy does not suffice.
- Point-rule corollary for the ensemble [A, repairing mine]: the finite-width power-series coefficients of kappa are
  nonnegative, which gives E_W ||error||^2 / E_W tr Var f >= (c/2 - O(1/d) - O(1/N)) / N for N <= d^p and
  r = min{j : N_j >= 2N}, whenever the spherical even energy above degree 2r is at least a fraction c of the averaged
  variance. A ratio of ensemble expectations, not per-net nMSE; for line rules the j + 1 factor does not drop out of an
  unrestricted tail, so only the LP certificate and the low-degree bound below apply to them.
- Proxy-free low-degree bound [A]: >= 1/(4 pi^2 N) for N <= n/2 points and >= 1/(8 pi^2 N) for N <= n/4 lines, in
  absolute squared error, at every depth.

The corollary's hypothesis concerns the finite-width kernel, and the NNGP kernel is not a safe proxy for it along
L = alpha n [A]; registered P6, a check of RQMC against my NNGP-proxy numbers, fails on 1 of 13 nets (256x32 s202). Descriptive per-net
energies (scratch/degrees.py v2, post hoc; 2^14 directions per net; degree 1 and 2 from split-half products of centred
moment tensors, unbiased up to a factor 1 + 1.5e-8; odd and even parts from antipodal pairs; shares of the angular
variance; no uncertainty attached):

| net | radial share of plain variance | degree 1 | odd >= 3 | degree 2 | even >= 4 | all odd | barrier c ~ (1 - radial) x even>=4, for N << N(d,2) | RQMC N x nMSE at N = 2^12, 2^13, 2^14 |
|---|---|---|---|---|---|---|---|---|
| 32x4 s302 | 0.045 | 0.52 | 0.12 | 0.28 | 0.08 | 0.64 | 0.07 (N << 527) | 0.14, 0.15, 0.17 |
| 32x4 s303 | 0.032 | 0.51 | 0.13 | 0.27 | 0.09 | 0.64 | 0.09 (N << 527) | 0.19, 0.23, 0.26 |
| 64x8 s302 | 0.048 | 0.48 | 0.16 | 0.25 | 0.11 | 0.64 | 0.10 (N << 2079) | 0.36, 0.23, 0.21 |
| 64x8 s303 | 0.036 | 0.36 | 0.23 | 0.24 | 0.17 | 0.59 | 0.17 (N << 2079) | 0.35, 0.34, 0.41 |
| 128x16 s302 | 0.039 | 0.36 | 0.23 | 0.23 | 0.17 | 0.59 | 0.17 (N << 8255) | 0.35, 0.43, 0.41 |
| 128x16 s303 | 0.031 | 0.21 | 0.33 | 0.19 | 0.26 | 0.54 | 0.26 (N << 8255) | 0.47, 0.65, 0.42 |
| 256x32 s201 | 0.034 | 0.25 | 0.31 | 0.17 | 0.27 | 0.56 | 0.26 (N << 32895) | 0.62, 0.42, 0.42 |
| 256x32 s202 | 0.039 | 0.26 | 0.30 | 0.18 | 0.25 | 0.57 | 0.24 (N << 32895) | 0.44, 0.63, 0.32 |

- The even energy above degree 2 grows with width along L = n/8 (0.08 at n = 32 to 0.26 at 256), as do the odd degrees
  above 1 (0.12 to 0.31), while degrees <= 2 fall from 0.79 to 0.43. Two nets per size give a descriptive check of
  the ensemble hypothesis, not an estimate of the ensemble ratio with controlled precision and not a proof that the
  share stays bounded below. The "barrier" column treats the aggregate as if it sat at degree 4; it is not a
  certificate (that needs degree 4 and 6 individually, beyond our sampling budgets at d >= 64), and it would apply
  only to rules randomised by a Haar rotation, which our Sobol runs are not.

**Beyond rules that average sampled values (E1, E5).** Under a Gaussian surrogate prior with the nets' covariance,
optimal algorithms are affine and adaption cannot beat the best nonadaptive design, also for scale mixtures of
Gaussians (Traub, Wasilkowski, Wozniakowski 1984; Wasilkowski, Wozniakowski 1984; re-proved directly [A]). Turning that
into a quantitative bound needs the nets' output channels separately (their covariance is exchangeable, not
separable), which I did not do. The real prior is far from Gaussian (scratch/nongauss.py, fresh nets 10000+,
n = 32-256 at L = n/8, descriptive): excess kurtosis of a unit's pre-activation at a fixed input 1.8-2.4 (close to the
lognormal scale mixture of Hanin-Nica, Var log |h_L|^2 = 0.67-0.73 against 5L/n = 0.625), of a unit's output 6.9-9.1
(half the units are zero at a given input). So for the true prior, adaptive and nonlinear algorithms are outside every
result here. My earlier claim that about nL evaluations would identify the net is withdrawn: a finite query set can
miss arbitrarily narrow activation cones, so open families of nets agree on all queries and differ in their
integrals [A].

## 6. What could still give a better asymptote
- Not pre-integration as built here: in fixed small input dimension it raises the fitted exponent (4.1) but loses on
  constants at every budget tested; at 8 x 256 no gain is seen (4.2); along the ladder it loses 21x to 1,410x and
  our finite windows show no consistent exponent difference (4.3; inconclusive about asymptotes). Window lines would
  also need a budget-dependent window to escape their eventual 1/N tail floor.
- A white-box surrogate family [A; PROVEN UNDER its hypothesis]: surrogates p_q with exactly computable means, costing
  q^a to build and q^gamma to evaluate (gamma <= a), with residual variance C q^-b uniformly in width, give MSE
  T^{-1-(b-gamma)/a} by surrogate plus independent residual sampling, an exponent above 1 when b > gamma. This is the one route I can state to an exponent
  above 1 that holds as width grows. Constraint [A]: controls in the span of squared hinges (the first-layer norm
  |relu(W_1 U)|^2, whose mean ||W_1||_F^2/(2d) is exact, and sums of such) have no even harmonic component of degree
  >= 4. Candidate [A]: biased Gaussian ridge controls a + B x + sum_r v_r (a_r . x - t_r)_+, with normals from
  first-layer rows and sampled gate normals and analytic Gaussian integrals; cheapest falsification is a
  nested-dictionary residual test across the ladder (held-out residual variance against dictionary size, all
  construction, fitting and evaluation charged). Not run; no evidence yet for a width-uniform b > 0.
- Combination with the randomness lane's designs (main's request): not run. At n >= 32 PI loses by factors of 21 to
  1,410 at equal cost, and at 8 x 256 RQMCJ already beats PIJ by 15-91x; I did not test whether an outer design
  could close such factors. (An earlier draft claimed a ceiling of about 4x on what a design can gain; withdrawn, it
  does not follow from the uncertified per-net energies.)
- Multilevel combination: the shared-trajectory controls depend on |h_r(U)|, which is not piecewise linear along a line,
  so a level difference has no closed-form line integral; not built.
- Direction choice: u from the cloud-averaged Jacobian (main's suggestion; post hoc, one 6 x 3 dev net) behaves like the
  active-subspace u (exponent 1.79 vs 1.70); the active-subspace u beats a random u on the small nets.

## 7. Scorecard (registered; results/scorecard.md, results/scores.json)

| test | prediction | outcome |
|---|---|---|
| P1 cost law (window vertices / Rice) | in [0.6, 1.5] on 13 of 13 | holds, 13 of 13 (ratios 0.94, 1.02, 1.07, 0.89, 0.89, 0.89, 0.81, 1.20, 1.01, 0.88, 1.07, 1.02, 1.11) |
| P2 line variance ratio, n >= 64 | rand >= 0.8, active >= 0.6 on all | holds (rand 0.91, 0.93, 0.91, 0.92, 0.90, 0.91, 0.98, 0.80, 0.95, 0.81; active 0.63, 0.77, 0.74, 0.83, 0.79, 0.83, 0.85, 0.75, 0.88, 0.87) |
| P3a PI >= 10x worse than RQMC at equal cost (S3) | all 13 | holds, 13 of 13 (min ratios 26, 21, 22, 128, 108, 109, 386, 308, 361, 647, 1067, 1410, 766) |
| P3b alpha(PI best) <= alpha(RQMC) + 0.15, n >= 128 | >= 5 of 7 | fails, 4 of 7 |
| P3c PI_as steeper by > 0.1: n = 32 (>= 2 of 3), n = 64 (<= 1 of 3) | both | fails (n=32: 3, n=64: 2) |
| P4 small nets: alpha(PI_as) > alpha(RQMCRB) + 0.1 | >= 6 of 9 | holds, 8 of 9 |
| P5a 8 x 256: PIJ worse than RQMCJ at every common cost | >= 5 of 6 | holds, 6 of 6 |
| P5b 8 x 256: abs(alpha(PIJ best) - alpha(RQMCJ)) < 0.3 | >= 4 of 6 | holds, 6 of 6 |
| P6 RQMC N x nMSE >= NNGP-proxy bound | all 13 | fails, 12 of 13 (min measured/bound 2.45, 2.90, 3.07, 1.77, 2.12, 1.77, 1.39, 1.79, 1.94, 1.63, 0.96, 1.52, 1.90) |
| P7 RQMC local exponent (last 3 budgets) <= 1.3 | >= 10 of 13 | holds, 12 of 13 |
| P8 MLMC alpha in [0.8, 1.2] | all | fails, 11 of 19 |

Validity: bias-flag cells by method: {'RQMC': 1, 'PIJ_as': 1, 'MLMC_32': 1}; suspect methods: []

Notes: P3b and P3c compare exponents fitted over different N windows (section 4.3); P6 is scored against the NNGP proxy
that the co-prover showed to be unreliable at L = alpha n; P8's MLMC fits include the allocation pilot, which dominates
the cost at the smallest budgets. All three are reported as scored.

## 8. Process and review
- Order: questions and reading (notes/QUESTIONS.md, notes/READING.md); the exact line library with its tests; piece
  counts on count-only nets; dev curves on dev nets; FORMAL v1 frozen 14:00:39Z; PREREG Part 1 hashed 14:03:47Z with the
  code frozen; scored runs from 14:04:09Z; Amendment 1 (15:08:43Z) cut the PI budgets of 256 x 32 nets 202-204 and
  Amendment 2 (15:34:02Z) those of net 201, because window lines took about 2 s each on the loaded machine; post-hoc
  analyses labelled as such (logs/deviations.txt). The PREREG header misstates its writing time (says 14:15-14:30Z;
  hashed 14:03:47Z).
- Co-prover (gpt-6-astra via codex-any, read-only, xhigh; "model: gpt-6-astra" checked in each raw log): round 1
  independent from a lemma-free statement (13:47-14:37Z); round 2 an audit (14:41-15:16Z). What each side found is in
  coprover/PROVENANCE.md and FORMAL E2, E5.
- Review: gpt-6-astra via codex-any (read-only, xhigh; model line checked in all three raw logs). Round 1 (15:22-15:45Z): major revision, 16 findings; round 2 (16:39-16:54Z): major revision, 9 findings; round 3 (17:20-17:25Z): minor revision, 9 items, all applied; rounds 2 and 3 reproduced every generated file byte for byte and all hashes. Resolved: scope of the summary, point/line separation, registered MLMC budgets, upper-bound labels, matched-N references at 256 x 32, ledger sensitivity (allowance-based), scorer completeness checks, wording. Disclosed limitations, not fixed: no slope-difference intervals or equivalence test (replicate vectors not saved), the P3b follow-up stopped after one extra budget on one net, no per-net degree certificate (degree 4 and 6 not measured), no instrumented operation counts, frozen runner imports the live library (file equality checked retrospectively), MLMC's uncharged draws.
- Compute: CPU only, at most two heavy processes at nice 19 / ionice idle; about 6 process-hours.
