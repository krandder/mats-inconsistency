# PREREG: cut rank of critical random ReLU nets (written 2026-10-30 ~23:35Z, before any data)

Lane cut-rank, bead pool-ttdzrk. Definitions as in FORMAL.md (sha a79866...). Hash of this file in prereg_hashes.txt.
Only a timing pilot on a configuration excluded from every test (n = 8, c = 2, L' = 2, F = 16, B = 4) may run before
the hash; its numbers are not analysed.

## 1. Objects

- Nets: noiseless bias-free He-ReLU (W iid N(0, 2/n)), Gaussian input, numpy float32 forward, float64 means. Fronts
  W_1..W_c and backs W_{c+1}..W_L from disjoint seeded streams. Inputs: u = x/|x| on the sphere (radius integrated
  exactly by homogeneity: E_x a = E|x| E_u a); per front, 3 independent splits of N' directions each.
- Ensemble cut matrix: rows = F fronts, columns = (back b, output neuron j), C = B n columns. M_s(f, c) = split-s mean.
  Vbar = mean over (f, b, j) of Var_x a_{L,j} (radial part included: n E_u a^2 - (E|x|)^2 (E_u a)^2).
- Per-path k*: mean over 64 paths (8 fronts x 8 backs) of N_surv(J; 0.01), J = d a_L / d z_c. Also N_surv(J; 0.1),
  the cut state's participation ratio (mean direction removed) and synchronisation diagnostics (median angle between
  two inputs' cut states; gate disagreement at the cut layer).

## 2. Estimators (fixed now)

- In-sample tail: eigenvalues of the symmetrised cross-split Gram (G12 + G13 + G23)/3, G_st = sym(M_s M_t^T)/(F C);
  tail_in(s) = sum_{i>s} lambda_i. Biased low (Ky Fan/Jensen), so r_in tends to underestimate r.
- Cross-fitted tail: subspace = top-s left singular vectors of M_1; captured energy from tr(P M_2 M_3^T)/(F C), with the
  roles of the splits rotated and averaged; tail_cf(s) = total - captured. Biased high (suboptimal subspace).
- r_in(eps), r_cf(eps): smallest s with the tail <= eps^2 Vbar, for eps in {0.3, 0.1, 0.03, 0.01}.
- Noise floor: identical-front controls (one front repeated F times with independent inputs; true rank 1) at
  (n, c, L') = (16, 8, 8) and (8, 64, 8). A tail value is resolved only if it exceeds 3x the control's tail_in(1) at the
  same n, N', F, C; eps with eps^2 Vbar below that floor are reported as noise-limited and excluded from tests.
- Saturation: r recomputed on the first F' fronts, F' in {64, 128, 256 (, 512)}; a rank with r_in(eps) >= F'/2 is
  "saturated" (population rank above what F' fronts can show).
- Fixed-back formulation (main's 23:2xZ note): for backs b = 0..3 of each configuration, readouts per front
  R2 = {1[a_j > t_jk]}: every output neuron, 8 thresholds at the 1/9..8/9 quantiles of a_j under that back (quantiles
  from a separate pilot of 8 extra fronts), and R3 = {1[v_m . a_L > t_mk]}: 64 random unit directions v_m, 8 quantile
  thresholds each. Per back: F x R matrix, split-sample tails as above, Vbar_b = mean p(1-p); r_b(eps) per back.
- Moment order: per front, moments of the cut state h_c(u) of order K <= 3 (n = 8), K <= 2 (n = 16), K <= 1 (n >= 32)
  from 2^16 separate samples; multi-output least squares from moments of order <= K to the cut matrix (response split 1),
  fitted on half the fronts, scored on the other half with the cross-split product of splits 2 and 3; held-out residual
  energy against eps^2 Vbar gives D(eps) = smallest K that reaches eps.
- Crux check (local rank on the relevant directions): n in {16, 32}, (c, L') = (8, 8) and (16, 16); 256 backs; for each
  back a reference cut pre-activation z_0 (a random front, random input), J = d a_L / d z_c at z_0, V = its right
  singular vectors; for k in {1, 2, 3, 4, 6, 8, 12, 16}, points z_0 + rho |z_0| V_k R_b w_i, w_i ~ N(0, I_k)/sqrt(k)
  (1024 common points), R_b a Haar rotation of R^k per back, rho in {0.05, 0.2}; matrix rows = points, columns = (b, j);
  r_loc(eps; k) = smallest s with uncentred tail <= eps^2 x (mean over columns of the variance over points). Control:
  the same with V_k replaced by a random k-dim subspace of the active coordinates.

## 3. Configurations (n, c, L')

- Window, width at fixed ratio c/n = L'/n = 1/2: (8,4,4), (16,8,8), (32,16,16), (64,32,32).
- Window, width at fixed depth: (8,8,8), (16,8,8), (32,8,8), (64,8,8).
- Window, back depth at n = 16, c = 8: L' in {2, 4, 8, 16, 32}.
- Front depth at n = 8, L' = 8: c in {2, 8, 64}.
- Past synchronisation (n ln 100 = 37 at n = 8, 74 at n = 16): (8,64,8), (8,8,64), (8,64,64), (16,128,16), (16,16,128).
- F = 512 fronts for n <= 16, 256 for n >= 32; B with C = B n >= 1024; N' = 8192 (n <= 16), 4096 (n = 32),
  2048 (n = 64). Seeds: fronts 0..F-1, backs 0..B-1, per configuration.

## 4. Predictions and decision rules (mine, from the kernel analysis in FORMAL.md section 4; the premise's in brackets)

- P1 (k* dependence, ensemble). n = 16, c = 8, L' from 4 to 32 (k* falls about 8x). [Premise: r(0.1) falls with L',
  r(L'=4)/r(L'=32) >= 4.] Mine: ratio <= 2. Supported if ratio <= 2, premise supported if >= 4, else inconclusive.
- P2 (width at fixed k*). c/n = L'/n = 1/2. [Premise: r(0.1) roughly flat in n.] Mine: r(0.1) at n = 32 >= 4 x that at
  n = 8, or n = 32 saturated while n = 8 is not. Premise supported if the ratio <= 2.
- P3 (synchronisation). [Coordinator: r_cf(0.1) <= 5 past synchronisation.] Mine: no collapse: r_in(0.1) >= 10 in at
  least 3 of the 5 configurations past synchronisation. Collapse supported if r_cf(0.1) <= 5 in at least 4 of 5.
- P4 (front depth). Mine: at n = 8, L' = 8, r_in(0.1) is non-decreasing in c over {2, 8, 64}.
- P5 (fixed back, rich readouts). Mine: if k* shows up anywhere it is here: median over backs of r_b(0.1; R3) at
  n = 16, c = 8 is >= 2x larger at L' = 4 than at L' = 32. Refuted if <= 1.
- P6 (moment order, n = 8 window configs (8,4,4), (8,8,8)): D(0.1) >= 2 in both (first moments are not enough).
- P7 (crux local rank, rho = 0.2): r_loc(0.1; k) increases in k, and r_loc(0.1; 8) / r_loc(0.1; 4) >= 2 at n = 16,
  (c, L') = (8, 8). Refuted if the ratio <= 9/5 (the count for linear functions, k + 1).
- P8 (typical fixed net). For the best ensemble protocol of rank s = floor(r_in(0.1)/2), the fraction of (f, b) pairs
  whose own error exceeds 0.01 x V(f, b) (eps = 0.1) is >= 0.5 at (16, 8, 8).

## 5. What would change the plan

- If P1 or P2 support the premise's k* dependence, the proof route switches from harmonic counting to gate combinatorics
  along the relevant directions. If P3 shows collapse, the harmonic analysis is wrong somewhere and is rechecked first.
- All runs: CPU only, nice -n 19 ionice -c3, at most 4 heavy processes.

## Amendment 1 (2026-10-30 ~23:55Z, before any main or control data; only the timing pilot has run)

Reason: measured forward throughput at nice 19 under load ~14 is ~1 GFLOP/s for n <= 16 (2 for column-major at n = 16),
6 at n = 32, 10 at n = 64, so the planned sample sizes would take 4-15 h per deep configuration.
- Splits: S = 8 splits of N' = 1024 directions per front (8192 per front) in every configuration, replacing 3 splits of
  N' = 8192/4096/2048. In-sample Gram = mean over the 28 split pairs of sym(M_s M_t^T)/(F C); cross-fitted tail =
  subspace from the mean of 4 splits, energy from the 6 pairs of the other 4, halves swapped and averaged. Probe
  protocol and per-net errors: protocol from split 0, unbiased scoring by the pairs of the other 7 splits.
- Sizes: F = 512 for n <= 16 and 256 for n >= 32; B = 64 (n = 8), 32 (16), 16 (32), 8 (64), so C = B n = 512.
  Configurations with c or L' >= 64 at n = 8, and (16, 16, 128): F = 256 and B halved (C = 256).
- Everything else (configurations, tests, decision rules, noise-floor rule) unchanged.

## Amendment 2 (2026-10-30 ~23:50Z, AFTER the first 8 runs were analysed: exploratory, not confirmatory)

Coordinator steer (~23:48Z): make the growth of r(eps) as eps -> 0, against sampling's 1/eps^2, a headline. Added:
- A held-out-front tail (lib/spec.py tails_heldout): column subspace from the pair Gram of half the fronts, energy of
  the other half's rows outside it. An upper estimate of the population tail for the sampled columns, to bracket the
  in-sample (lower) estimate. Reported as r_ho.
- Three runs at n = 8 with F = 2048 fronts (8 splits of 1024): (8, 8, 8), (8, 64, 8) and the identical-front control
  of (8, 8, 8), to see ranks up to ~1000 and compare spread fronts (c = 8) with synchronised ones (c = 64).
Tests P1-P8 are unchanged and scored only on the registered estimators.
