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
