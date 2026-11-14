# Provenance: Claude (C) and gpt-6-astra (A), input-bits lane

## Round 1 (A: 18:01:33Z-18:26:47Z, xhigh, read-only, told not to read work/input-bits/)
Prompt `prompt_r1.md` (sha 9220f403), raw `astra_r1.raw`, answer `astra_r1_answer.md` (extracted from the raw log's last
"codex" block). My FORMAL v1 (`FORMAL_v1_as_written.md`, sha 6972fef3, frozen 18:07:44Z) was written before A's answer.
Disclosure: at 18:07:33Z the hash command printed the last lines of A's running log: two computed lines,
"total-effect sums vector,norm [129.67, 39.54]" and "n kernel derivative vector,norm [3.285, 2.715]", without context.
FORMAL v1's content was complete at that point; afterwards only its header time was corrected (typo) and rehashed.
PREREG Part 1 (18:16:53Z) and Part 2 (18:23:45Z) were hashed before I read A's answer (read 18:27Z onward).

Found independently by both [C+A]:
- the exact three-term split V = V_rad + V_amp + V_dir with V_dir = m_n^2 E[A A'(1 - d.d')] (identical formulas);
- the amplitude coefficient: my Lambda(L)/4 = A's C_L = (1/4) sum_j [5 - B(c_j)] (the same sum; B(c) = (2/pi)J_2 - 1);
  C_inf = 10.65; V_dir ~ n(1 - c_L); the crossover depth near 24-28 at n = 256 (A: ~25 with the finite-width chain);
- the shares at ARC's 256 x 32: amplitude first (C dev nets 0.55-0.59, A annealed chain 0.594), direction 0.37-0.43,
  input radius 0.02-0.04; "the radius that carries the input is the amplitude A(u), not |x|";
- gate disagreements n arccos(rho)/pi ~ 3n/l (algebraic); the Gaussian-cap entropy factor (my 2.31 x Hamming equals A's
  0.735 theta n, since 2.31/pi = 0.735); 38 bits consistent (A: 8 nets 28.6-57.0, mean 39.5; kernel 44.5);
- direction mean dimension ~ 1/delta_L at infinite width (A: annealed d = 1/(1 - K_L(0)) for fixed L);
- antithetic gain = 1/(2 s_even), parity not order; RQMC gain needs more than the first-order share (pairs).

New from A (adopted in FORMAL v2):
- proofs: E_W V_A -> C_inf whenever n, L -> inf (uniform summable bound 0 <= E V_A(l+1) - E V_A(l) <= (2 m_n^2/n) D_l);
  the annealed replica formulas S_1^ann = n(F_1 - F_0)/V, d^ann = n(F_n - F_{n-1})/V; the fixed-depth limits
  S_1 -> K_L'(0)/(1 - K_L(0)), d -> 1/(1 - K_L(0));
- E_W sum_k k H_k(W) = n exactly (critical weights preserve E|J|_F^2), so the annealed mean Hermite degree is n/V
  (about 19.5 at ARC);
- the finite-width linear (degree-1) share of the amplitude: kernel C_L(t), linear coefficient a_1 = -C_L'(0) =
  2 sum_j c_{j+1} K_j'(0) (2.98 at L = 32, 3.095 at inf), so the amplitude's first-order share tends to 0.29 and the full
  map's to 0.32 at L = alpha n; the first-layer preferred direction b ~ (2 pi n)^(-1/2) sum_i |w_i| w_i. This replaces my
  FORMAL v1 3.2 CONJECTURE (low-rank deep Jacobian).
- amplitude mean dimension ~ 2L/C_L for L << sqrt(n) and ~ 0.999 sqrt(n) for L >> sqrt(n): D grows like sqrt(n) at fixed L/n;
- stationary noise: frozen-radius prediction of the output linear share 0.674 (n = 8) -> 0.651 (n = 128) -> 2/3, i.e. flat;
  antithetic s_even ~ 1 - 2^(-1/3) = 0.206 (gain 2.42) from a (1 - t)^(2/3) stability closure;
- the hybrid MC floor eta = E_U Var_Z f / V for RQMC with MC on older noise;
- the exact-mean first-layer control C_1(U) = m_n s_1 q_L [sqrt(q_1) + (Q_1(U) - q_1)/(2 sqrt q_1)], q_1 = |W_1|_F^2/(2n),
  with first-level residual variance O(1/n); quadratic-Hermite controls with exactly known Gaussian mean;
- framing: fixed sigma with n -> inf at fixed L/n leaves the input regime unless sigma sqrt(n) is controlled.
New from C (not in A's round 1): the dev measurements (`logs/dev_probe*.log`), the RQMC order-k calibration
(`logs/calib_rqmc_d*.log`: rho_2 depends on N/d, order-2 terms worse than MC at N = d), the registered predictions.

## Round 2 (A: 18:50:32Z-19:09:32Z, xhigh, read-only, could read the lane directory)
Prompt `prompt_r2.md` (sha ff2c3b91), raw `astra_r2.raw`, answer `astra_r2_answer.md`. Tasks: scorer and estimator audit,
theory follow-up, synthesis red team, Part 3 forecasts. Findings and what I did:
- Registration integrity OK (hashes match; the three PREREG parts are prefixes of the current file).
- Scorer deviations: P1.17/P1.18 test only the recorded power-of-two layers (the registered runner records only those):
  reported as "sampled layers"; P1.19 slope aggregation (both conventions pass); P1.16 aggregation ambiguity; H2.2 built
  G_pred from cell medians instead of per-net ratios (both now reported); H2.5 needed an n = 256, L = 32 cell that the
  registered runner never generated (fixed before any run, Amendment 2C); Part 3's scorer did not implement P3.1-P3.6 and
  extrapolated charged-pilot gains below the pilot's own cost (now UNAFFORDABLE; implemented after the hash, reported as
  such); ML2-RQMC lacked charged cost (fixed before any run, Amendment 2C). Missing data could pass as `all([])`
  (completeness now checked).
- Estimator bias: the Hermite two-halves estimators centre with the full-sample mean (halves dependent; relative bias
  O(1/h) ~ 1e-4 at h = 8192) and part2_noise divides centred sums by h, not h - 1 (factor 1 - 2/h); both negligible here,
  disclosed. deg2_share at n = 256 normalises a 64-output subset by its own variance (a subset share).
- Gamma: the registered normalisation bug confirmed (mean squared norm n/Mref); A also proved that the population mean
  angular sensitivity is infinite at finite n (P(two active units) > 0 and E[1/chi^2_2] = inf), so Gamma is reported as a
  finite-sample pilot statistic with medians, not as a population mean.
- Heavy tails: chain amplitude terms at L ~ n have effective sample sizes 20-312 of 20,000; one n = 128 net carries 97% of
  the 4-net amplitude sum; exact moments E Q_L^2 = (1 + 5/n)^L, E Q_L^3 = (1 + 15/n + 44/n^2)^L explain it. Medians are used
  for typical-net statements, pooled/annealed ratios only for ensemble statements.
- Theory: a first finite-width correction to V_dir at fixed depth (expansion) and chain factors E V_dir/(n delta_L) of
  0.56-0.57 at L = n/4 and 0.32-0.33 at L = n/2 for n = 64-256.
- Synthesis: H2.6 tests "declining additive share", not interaction order in general (fixed S1 with variance moving from
  order 2 to order 10 would be classified REFUTED); proposed the coordinate-sharing generating function C(t) and the hybrid
  floor eta (adopted as Amendment 2B, registered before its runs).
- Part 3: forecasts (gains RQMC ~2.0-2.6, ML2-RQMC up to 3.4-4.5 at L = 2n, CVQ below 1 at n >= 128 from quadratic fitting
  noise of 0.24-0.80 of the variance on a linear test function), exponent forecasts 1.10-1.28; P3.6 can declare progress from
  variance-estimation noise (R = 10: relative SE ~47% per variance).
