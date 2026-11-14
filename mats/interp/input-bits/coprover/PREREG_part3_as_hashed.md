# PREREG, input-bits lane (pool-qxszj1.22)

Each part is hashed (sha256 in `prereg_hashes.txt`) before any of its evaluation runs. Dev seeds >= 9000 informed the
predictions (`logs/dev_probe*.log`, `logs/calib_rqmc_d*.log`); evaluation seeds are fresh: 4000-4003.
"Median" = median over the 4 evaluation nets of a cell (n, L) unless stated; "pooled" = sum over nets of a variance
term divided by the sum of V (the quenched analogue of an annealed ratio). Kernel numbers: rho_l = kappa^(l)(0),
delta_L = 1 - rho_L, theta_l = arccos rho_l, Lambda(L) = sum_{l=1}^{L} [6 - (2/pi) J_2(theta_{l-1})] (FORMAL v1, 2.1-2.2).

## Part 1: input structure (registered before any evaluation run of part1a/part1b/part1c/part1chain)
Runs (scripts as committed at hash time): `scratch/part1a.py n seed`, `scratch/part1b.py n seed`, `scratch/part1c.py n seed`
for n in {32, 64, 128, 256}, seeds 4000-4003 (part1c only n in {64, 256}); `scratch/part1chain.py n` for the same n.
Depth grid D = {n/8, n/4, n/2, n, 2n, 32} (+ 4n for n <= 64); part1b uses {n/8, n/4, n/2, n, 32}. sigma = 0 unless stated.
Scorer: `scratch/score_part1.py` (written after this hash, implementing exactly these rules).

### 1a. What carries the input (three-term split, FORMAL 1.1, 2.1-2.3)
- P1.1 ARC's point (n = 256, L = 32): median s_amp in [0.40, 0.75], median s_dir in [0.20, 0.55], median s_rad <= 0.06.
- P1.2 Kernel + 1/n ratio: for n >= 64 and L <= n/2, the median of V_dir/V_amp is within a factor 2 of 4 n delta_L / Lambda(L)
  in every cell; for L >= n it is below that value in every cell (one-sided: finite width erases directions faster).
- P1.3 Trends: at fixed L = 32 the median s_dir increases with n over n = 32, 64, 128, 256 (monotone); at fixed L/n = 1/2
  it decreases with n (monotone over the four widths).
- P1.4 Amplitude: for n >= 64, L <= n, the median of Var_u log A is within [0.6, 2.0] x Lambda(L)/(4n) in every cell.
- P1.5 Chain vs nets: for n >= 64 and L <= n, the exact chain's annealed s_dir is within a factor 1.5 of the pooled s_dir of
  the 4 nets in every cell, and its annealed V within a factor 1.5 of the nets' mean V.
- P1.6 Noise (assumption A1): at sigma = 1e-3 and L <= T_mem/2, every cell has coupled noise variance <= 0.01 of V and
  |change of s_amp|, |change of s_dir| <= 0.01 (all nets).

### 1a. Input gate bits (FORMAL 1.2, 2.4, 2.5)
- P1.7 Kernel bits: for every n and every layer l <= n/4 (l >= 2), the median Hamming count is within 25% of
  n theta_{l-1}/pi; the exact chain's annealed Hamming is within 25% of the nets' median at every layer with median >= 2.
- P1.8 Algebraic decay: at n = 128 and 256 the least-squares slope of log(median Hamming) on log l over l in [8, n/4] lies
  in [-1.25, -0.6] (a Lyapunov-rate decay would give a slope near 0 there).
- P1.9 Entropy factor: H_marg / Hamming in [2.1, 2.5] at every layer 8 <= l <= n/2 with median Hamming >= 4 (median over
  nets, every n).
- P1.10 The 38 bits: at 256 x 32, layer 31, every net has H_marg in [30, 65] bits and Hamming in [14, 28].

### 1b. Interaction structure over input coordinates (FORMAL 3)
- P1.11 Mean dimension at 256 x 32: median D in [6, 18].
- P1.12 Width growth at fixed L/n = 1/2: median D(256, 128) / median D(64, 32) >= 1.5.
- P1.13 Linear share: for every cell with L >= n/4, median W1 in [0.12, 0.45], and W1 exceeds twice the infinite-width
  kernel part n b_1(L) (E A)^2 m_n^2/(n V) (b_1 = prod_{l<=L} kappa'(rho_{l-1})) on every net.
- P1.14 Univariate excess is the radius: S1 - W1 in [s_rad - 0.03, s_rad + 0.06] (per net, s_rad from part1a at the same
  depth) in every cell.
- P1.15 Antithetic x -> -x: s_even in [0.35, 0.65] in every cell with L >= n/8 (gain 1/(2 s_even) in [0.77, 1.43]).
- P1.16 Rotation: in cells with L >= n/2, the rotated off-diagonal degree-2 share is <= 0.6 x the unrotated one (median) and
  the rotated first-order share exceeds the unrotated one by >= 0.5 x (W2_off - rot_W2_off) (median).

### 1c. Age and decay of input information by Lyapunov direction (n = 64, 256)
- P1.17 Finite angular fraction (tr of the input cloud's covariance orthogonal to the reference state over E|h|^2) within a
  factor 2 of delta_l for 2 <= l <= n/4 (median over nets).
- P1.18 Infinitesimal input changes concentrate: tangent angular PR <= n/(2l) + 2 for 4 <= l <= n/2; the finite angular PR is
  >= the tangent one at every l <= n/2 (medians).
- P1.19 Finite vs infinitesimal decay: the ratio Gamma_mean / (finite angular fraction) grows with l: log-log slope over
  l in [4, n/2] >= 1.0 (median over nets); Gamma_mean stays within [0.3, 3] for l <= n/2.
- P1.20 Depth >= n: the finite cloud's radial share >= 0.9 at l = n and 2n on every net.

Failure of any band is reported as a miss with its numbers; no band is changed after the data.

## Part 2: synthesis, structured gain vs low-order share (registered before any Part 2 evaluation run)
Runs: `scratch/calib_rho2.py --R 64` (instrument, no net data) -> `results/rho_calibration.json`;
`scratch/part2_noise.py n seed` for n in {8, 16, 32, 64, 128} and seeds {1000, 1001} (the stationary lane's nets, whose
RQMC gains are in `work/stationary/runs/rqmc_n*_s*.json`, unchanged) and {4000, 4001} (fresh; RQMC by the stationary lane's
unchanged `partB.py rqmc n seed --kmax 10 --R 16`, gains against part2_noise's V); `scratch/part2_input.py n seed`
(R = 32 for n <= 128, R = 16 for n = 256; N = 2^8, 2^10, 2^12) and `scratch/part2_mlmc.py n seed` for n in {32, 64, 128, 256},
seeds 4000-4003 (the Part 1 nets; S1, W2_off come from part1b on independent samples). Scorer `scratch/score_part2.py`
(written after this hash). Medians over the nets of a cell; noise-regime cells pool the 4 nets (1000, 1001, 4000, 4001).
Notation: S1 = first-order (univariate, degrees 1-4) share in the Sobol' coordinates; W1 = degree-1 share; W2_off =
off-diagonal degree-2 share; s_even = even share (antithetic pair); G(N) = RQMC variance gain over MC at N points;
rho_2(N, d) = calibrated order-2 RQMC variance ratio (exchangeable quadratic forms, d = n input coordinates).
Model (FORMAL 1.5, 4.1): 1/G = sum_k rho_k s_k, order-1 terms removed, order >= 3 terms not reduced:
  G_pred(N) = 1 / (1 - S1 - (1 - rho_2(N, n)) W2_off).

- H2.1 First-order floor: in every cell of both regimes, median G(N) >= 0.75 / (1 - median S1) at N = 2^10 and 2^12 (noise
  regime: N = 2^10 only, the largest N run at every n).
- H2.2 Input regime, quantitative: median G / G_pred in [0.67, 1.5] in >= 80% of the (n, L, N) cells, and the median over
  cells of that ratio in [0.8, 1.25]. Implied residual check (reported, not scored separately): the RQMC gain on f - f1
  (first-order projection removed) ~ 1/(1 - (1 - rho_2) W2_off/(1 - S1)).
- H2.3 Noise regime, shares vs width: (a) the median S1_win at n = 64 and at n = 128 is within 0.15 of the median at n = 16, and
  the median s_even lies in [0.15, 0.35] at every n; (b) the excess factor E = G(2^10) (1 - S1_win) has median >= 1.5 at n = 8
  and <= 1.3 at n = 64 and at n = 128 (what RQMC removes beyond the first-order share fades as the window dimension grows).
- H2.4 Parity: in the noise regime 2 s_even agrees with the stationary lane's antithetic pair ratio within 0.08 on seeds
  1000, 1001 (consistency of two estimators of the same identity), and the odd high-degree share 1 - s_even - W1_all lies in
  [0.03, 0.25] at every n and its median changes by <= 0.10 from n = 8 to n = 128.
- H2.5 Multilevel shared trajectory: (a) V2(r)/V_sph within a factor 1.5 of s'_dir + s'_amp (Lambda(L) - Lambda(r))/Lambda(L)
  for r in {L/8, L/4, L/2} in >= 75% of the (n >= 64, L, r) cells (medians); (b) the best two-level ideal gain at L = n
  increases with n over n = 64, 128, 256, and at L = 32 it does not increase over the same widths (medians).
- H2.6 Verdict rule for Kelvin's question ("does a growing interaction order explain why gains shrink with width?"), per
  regime (noise: n = 16 -> 128; input: along L = 32 and along L = n/2, n = 64 -> 256): SUPPORTED if the median S1 falls by
  >= 0.10, H2.1 holds there, and median G/G_pred (input) or E (noise) shows no monotone trend in n; REFUTED if the median S1
  changes by < 0.10 while the median G(2^10) falls by >= 1.5x; otherwise UNRESOLVED.

## Part 3: structured samplers for the input regime (registered before any Part 3 evaluation run)
Runs: `scratch/part3.py n seed L --R 10 --mmax 12 --Pp P` with P = 8192 (n <= 128) or 16384 (n = 256), fresh seeds 4004 and
4005 (not used by Parts 1-2), n in {64, 128, 256}, L in {32, n/2, n, 2n}. Scorer `scratch/score_part3.py` (committed at this
hash): per method, the exponent of nMSE in compute (least squares over N = 2^6..2^12, and the last step 2^10 -> 2^12) and the
gain over plain MC at C* = 2^12 L mv, with and without the pilot/setup charged. Methods are defined in part3.py's docstring
(MC, RBMC, RQMC, RQMC-RB, RQMC-ROT, CVQ, RQMC-CVQ, ML2, MLD, ML2-RQMC). Predictions come from FORMAL v1, astra round 1 and
the dev nets only; medians over the two nets.
- P3.1 Rates: every RQMC-type method (RQMC, RQMC-RB, RQMC-ROT, RQMC-CVQ, ML2-RQMC envelope) has a fitted exponent in [0.9, 1.6]
  in every cell; at n = 256 and L >= 128 its fitted exponent is in [0.9, 1.3].
- P3.2 Fixed depth L = 32: the RQMC-RB gain (no pilot) is lower at n = 256 than at n = 64.
- P3.3 The quadratic control: CVQ's residual variance share is in [0.35, 0.75] in every cell with L >= n/2.
- P3.4 Rotation: RQMC-ROT beats RQMC-RB at C* (no pilot) by >= 1.2x in at least half of the cells with L >= n/2.
- P3.5 Multilevel: the ML2 gain (no pilot) at L = 2n increases with n over 64, 128, 256, and is >= 1.5 at n = 256; the
  ML2-RQMC envelope gain is >= max(RQMC-RB, ML2) in at least half of the cells with L >= n.
- P3.6 Verdict rule for "progress": PROGRESS if some method's median gain at C* with the pilot charged, at L = 2n, grows by
  >= 1.5x from n = 64 to n = 256, or some RQMC-type method at n = 256, L >= n has fitted and last-step exponents both >= 1.2;
  otherwise "constants only".
