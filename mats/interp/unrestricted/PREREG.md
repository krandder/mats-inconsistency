# PREREG: unrestricted deterministic estimators lane (31 October 2026)

Written before any evaluation-net result. Dev runs on seeds 900-1009 (census_dev.json) informed the predictions;
those seeds are excluded from evaluation. Hashes of this file and of the code at registration go in
prereg_hashes.txt. Labels in the report will mark anything not registered here as post hoc.

## Nets and selection
- Plain He-ReLU, sigma = 0 (lib/nets.py): W_l iid N(0, 2/n), no bias, x ~ N(0, I_n), output a_L in R^n.
- Census rule (scratch/census.py, fixed): a seed is healthy if, on 2^14 fixed sphere directions, at most half give
  a_L = 0 and the angular variance is positive. Evaluation seeds start at 5000; the first 3 healthy seeds per
  configuration are used. Healthy rates are reported.
- Configurations: n in {3, 4, 6, 8} x L in {4, 8, 16, 32}; (6, 64), (8, 64); n in {16, 32} x L in {8, 16, 32}.

## Estimand, truth, metric
- mu = E a_L; nMSE = |est - truth|^2 / V, V = tr Var_x a_L (so plain MC with N paths scores 1/N).
- Truth: mean of 8 scrambled-Sobol radius-exact replicates (seeds 1000-1007) at 2^23 points (n <= 8) or 2^21
  (n = 16, 32). Truth nMSE (its own squared SE / V) is reported. A method's nMSE counts as resolved only if it is at
  least 10x the truth nMSE; ratios with unresolved values are reported as bounds, not claimed.
- Cost unit: one noiseless forward pass of one input, 2 n^2 L FLOPs (schoolbook products; flopscope conventions:
  cdf 96, pdf 54). Every B&B operation of order n or more is charged (bounds, cell moments with q-node quadrature,
  centre passes, bookkeeping). Baseline samplers are charged only their forward passes (generation free), which
  favours the baselines.

## Arms
- B&B primary: lib/bnb.py CertifiedBnB, priority 'gap', split 'mid' (bisect the smear coordinate), batch 128.
- B&B secondary: priority 'instab' (mass x summed unstable range fractions), split 'mid'.
- Budget per B&B run: checkpoints at 2^4 .. 2^23 passes, stopping at 2^23 passes, 400,000 leaves or 900 s wall.
- Baselines: plain MC (1/N exactly); RB-MC (radius exact, nMSE = rho_rb / N analytic); RQMC-RB (scrambled Sobol,
  radius exact; 32 replicates for N <= 2^12, 16 to 2^16, 8 above; N = 2^4 .. 2^22 for n <= 8, .. 2^20 for n >= 16);
  QMC-det (unscrambled Sobol, centred by 2^-(m+1), radius exact; one deterministic run per N).
- Comparison at equal cost: RQMC interpolated log-log between measured N; never extrapolated beyond its largest N.
  Scored budgets: 2^k passes, k = 10, 12, ..., 22.

## Predictions (B&B; scored on the primary arm unless stated)
- P1. n = 3, L in {4, 8}: beats RQMC by >= 3x at some scored budget in [2^16, 2^22] on >= 4 of 6 nets.
- P2. n = 4 (all L): >= 10x worse than RQMC at every scored budget in [2^12, 2^22] on >= 9 of 12 nets.
- P3. n in {6, 8} (all L): >= 100x worse than RQMC at every scored budget in [2^14, 2^22] on all 30 nets.
- P4. n in {16, 32}: >= 1000x worse than RQMC at the largest scored budget reached (>= 2^16) on all 18 nets, and
  certified nMSE (sum of squared half-widths / V) >= 1 at every checkpoint.
- P5. Certified nMSE <= 1e-6 within 2^23 passes on >= 4 of 6 nets at n = 3, L <= 8; on no net at n >= 6.
- P6. Median B&B nMSE at 2^20 passes increases monotonically in n (3, 4, 6, 8) at L = 8 and at L = 16, by >= 100x
  from n = 3 to n = 8.
- P7. Deterministic designs: the geometric mean over the n <= 8 nets of nMSE(QMC-det) / nMSE(RQMC) lies in
  [1/3, 3] at N = 2^10, 2^14 and 2^18 (rotation-equivalence prediction, FORMAL Prop. 2).
- P8. n in {6, 8}: the primary B&B also loses to RB-MC at every scored budget on >= 2/3 of nets.
- P9. Secondary vs primary: no prediction (descriptive).

## Theory measurements (registered here; scripts fixed)
- M1 noise stability (scratch/noise_stability.py, K = 48 net pairs, 2^14 Sobol points per net, deltas
  1e-5 .. 1): n in {16, 32} x L in {4, 16, 32}.
  - P10a. At L in {16, 32} the least-squares slope of log D(delta) on log delta over delta in [1e-4, 1e-2] is in
    [0.5, 0.8] (heuristic delta^{2/3}).
  - P10b. At L = 4 the slope over delta in [1e-5, 1e-3] is >= 0.8.
  - P10c. At L in {16, 32}: D(0.1) >= 0.25, and the degree-4 tail lower bound (S1 - min_rho S(rho) rho^-4) / S1
    is >= 0.1.
- M2 harmonic profile (scratch/harmonic_profile.py, 2^18 directions per net, 3 nets): L = n/4 for
  n in {16, 32, 64, 128, 256}, and 256 x 32.
  - P11. f1 + f2 in [0.3, 0.95] on every net, and the median t = 2 ceiling 1/(1 - f1 - f2) changes by less than 2x
    between n = 16 and n = 256 (designs give a bounded gain).
- M3 spin-glass check (scratch/spin_glass.py): infinite-width kernel kappa_L, L = 1, 2, 4, ..., 64.
  - P12. (kappa_L'')^{-1/2} is concave on a fine grid of (0, 1) for every L.
- M4 region counts (scratch/regions.py, n = 3, census-healthy seeds from 5000, 3 per L in {4, 8, 16, 32}): distinct
  gate patterns met by 2^22 Fibonacci-sphere directions.
  - P13. log-log slope of the median count against L in [1, 3].
- M5 noise effect at sigma = 1e-3 (paired MC, 2^20 paths) on the B&B evaluation nets with n >= 6: descriptive.

## Deviations
Any change after this point is listed in the report as a deviation, with its time.

## Amendment 1 (11:29Z, before any evaluation-net result; compute)
- The secondary B&B arm runs only at n <= 8; at n in {16, 32} only the primary arm runs (900 s cap each).
- Jobs run 3 at a time (xargs -P 3) under nice 19 / ionice idle, single-threaded BLAS.

## Post-hoc addition (12:40Z, after gpt-6-astra round 3; not registered)
Astra's round 3 (coprover/astra_r3_answer.md) proposes a multilevel reference-trajectory estimator with a proven
growing gain over radius-exact MC at L = alpha n. scratch/mlmc_check.py measures its ideal variance x cost against
RB-MC (setup excluded) on fresh seeds 9100-9103 at (n, L) in {16, 32, 64, 128, 256} x {n/4, n, 4n} where feasible. It
is a short (about 2 CPU-minutes) single-threaded job run beside the queue. No prediction is registered; it is
descriptive and labelled post hoc.
