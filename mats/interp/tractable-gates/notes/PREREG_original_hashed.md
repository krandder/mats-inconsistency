# Tractable gates: pre-registration (written 2026-10-31 ~01:40Z, before any network measurement)

Lane: tractable-gates, bead pool-2wj2bh. Author: opus-mats-gates (Claude Opus 5.5). Ask (Kelvin): "study whether
interesting functions that can serve as gates (and help networks learn) can still be more tractable in this sense",
"this sense" being the cut-rank lane's (outbox/cut-rank-lower-bound-2026-10-30.md). His bar: "I am only interested in
things that could scale to large (10M+ parameter) models".

## Seen before writing this (declared)
- Theory numbers only, no network: `logs/ray_family.log` (exact eps-ranks for ray fronts and one back layer, 18 arms,
  n = 4-64, eps = 0.3-1e-6), `logs/hermite_classes.log` (Hermite tails at s = sqrt 2, polynomial-summary bound),
  `logs/meanfield.log` (infinite-width correlation maps and chi_1 at q0 = 2), `logs/test_acts.log` (library tests).
  Headlines: in the ray model at n = 8 ReLU needs 184 numbers at eps = 0.1 (sampling 100) and 7e24 at 1e-6
  (sampling 1e12); softplus with beta*s = 1.4 needs 42 and 1.75e9 (wins at every eps); GELU 110 and 6e10; SiLU 44 and
  1.1e10; at n = 16 every arm loses to sampling down to eps = 1e-6 in the ray model. Mean field at q0 = 2: chi_1 = 1.00
  (ReLU), 0.97/0.86/0.46 (softplus beta = 16/4/1), 1.06 (GELU), 1.07 (SiLU), 1.34 (tanh), 0.84 (gsrelu 0.5),
  1.08 (smoothstep 1). Bias-free smooth non-negative gates at a fixed pre-activation scale are ordered (chi < 1).
- Code smoke tests only (timing and "does it run", outputs trashed unread): run_cut_act.py at (8,2,2), F = 16;
  pilot-normalisation scale check (pre-activation RMS per layer, which showed that plain GLU stacks blow up in scale
  even with per-layer pilot normalisation); one PFN CPU timing (74 ms/step).
- Prior lanes: cut-rank (code reused read-only: work/cut-rank/lib/spec.py), narrow-deep (sorting MLP recipe, K2 code),
  arc-deep-networks (ARC App. F: GELU and tanh "qualitatively similar" for cumulant propagation).

## Objects
Activation arms (lib/acts.py): relu; softplus_beta(z) = log(1 + e^{beta z})/beta, beta in {1, 2, 4, 8, 16}; gelu;
silu; tanh; gsrelu_0.5 (E relu(z + 0.5 xi), ReLU with integrated Gaussian pre-activation noise of std 0.5);
smoothstep_1 (z S(z), S a C-infinity step from 0 at z <= -1 to 1 at z >= 1: exact zeros below -1); swiglu
(silu(W1 h) * (W2 h)) and the bilinear GLU (W1 h * W2 h) in training only (see "GLU" below).
Effective sharpness: softplus beta*s with s the RMS pre-activation; s = sqrt 2 at init by construction.

Random nets (lib/gnet.py): plain, bias-free, x ~ N(0, I_n), weights iid Gaussian, then per-net per-layer pilot
normalisation (4096 Gaussian inputs) so the RMS pre-activation of every layer is sqrt(q0) = sqrt 2. This fixes the
gate's effective sharpness in every net and layer; for ReLU it changes nothing but a global scale per net.
GLU: plain GLU stacks are polynomial-like and their scale diverges with depth even when normalised (smoke check), so
GLU is not in the random-net grids; its kernel class is covered by the ray model and the theory note, and it is
trained on the sorting task.

## Grids
C (cut rank, run_cut_act.py + analyze_cut.py, the cut-rank lane's estimators):
- configs A = (n, c, L') = (8, 4, 4): F = 512 fronts, B = 32 backs (C = 256 columns), Np = 512, S = 8 splits;
  B' = (16, 8, 8): F = 512, B = 16 (C = 256), Np = 512, S = 8.
- arms: relu, softplus16, softplus4, softplus1, gelu, silu, tanh, gsrelu0.5, smoothstep1 (9 x 2 runs).
- controls: identical fronts (true rank 1) for relu and softplus4 at A and B'.
- optional if time: (32, 8, 8), F = 256, B = 8, Np = 512 for relu, softplus4, softplus1, gelu.
- estimators: r_in (in-sample pair-Gram tail; primary) and r_ho (held-out), raw and scale-normalised (each front row
  by its RMS output, each back by its RMS gain). eps in {0.3, 0.1, 0.03}; eps = 0.01 is reported as unresolved unless
  both controls' |tail_in(1)| < 0.01^2 Vbar. Ranks are capped by min(F, C) = 256 ("sat").
K (deductive vs sampling, scratch/k2_grid.py):
- nets: n in {8, 16, 32, 64}, depth L in {2, 4, 8, 16} (truncations of one depth-16 net), nets (seeds) 0-3 per (arm, n).
- arms: relu, softplus1, softplus2, softplus4, softplus8, softplus16, gelu, silu, tanh, gsrelu0.5, smoothstep1.
- truth: plain MC with 2^21 inputs (float64); its own standard error reported.
- K2: Gaussian closure from the exact input law, mean + full covariance; per-unit 1-D Gaussian moments by
  Gauss-Hermite (64 nodes; smoothstep 200); off-diagonal E phi(z_i) phi(z_j) by Mehler's series with each unit's
  shifted Hermite coefficients to order 32; ReLU in closed form (narrow-deep's gauss_relu). FLOPs counted with the
  program's conventions (matmul M N (2K - 1); exp/log/erf 16; elementwise 1).
- MC at equal FLOPs: N_eq = FLOPs(K2) / FLOPs(one sample); MC's nMSE is 1/N_eq exactly (ARC's metric,
  nMSE = |est - truth|^2 / tr Var).
- RQMC: scrambled Sobol (Owen) inputs through the inverse normal CDF, N = largest power of 2 <= N_eq (and N = 2^10,
  2^14 for the gain curve), 16 replicates; nMSE from replicates against the truth.
- win rule: K2 beats sampling in a cell (arm, n, L) if its nMSE is below both 1/N_eq and RQMC's at N_eq on >= 3 of
  4 nets.
T (learnability):
- Sorting MLP (narrow-deep's residual, bias-free recipe: h_l = h_{l-1} + V_l phi(W_l h_{l-1}), W ~ N(0, 2/n),
  V ~ N(0, 1/(n L)), target sort(x), x ~ N(0, I_16), Adam lr 2e-3, warm-up 1/32, cosine to 0.01x, batch 1024,
  grad-clip 1.0, noise sigma = 0), width 16, L = 16, 2^14 steps (CPU), seeds 0-2.
  Arms: relu, softplus1, softplus4, softplus16, gelu, silu, tanh, smoothstep1, swiglu, bilinear. GLU arms use hidden
  width 11 (3*16*11 = 528 parameters per layer against 512).
- PFN Bayes task (tiny-pfn recipe as in cm-interp-B: pfn2 set transformer, 2 blocks, width 32, 2 heads, mixture head
  K = 4, fixed 90,000-step schedule) with the MLP activation swapped: gelu (the recipe's own), relu, softplus1,
  softplus4, silu, swiglu (hidden width matched: 2/3 of 64 -> 43). Seeds 0-2, on one L4 if main approves (asked
  01:38Z); otherwise relu, gelu, softplus1, swiglu x seeds 0-1 x a fixed 15,000-step schedule on CPU, labelled reduced.
  Metric: test KL to Bayes (pfn2 evaluate, 16,000 test episodes).
- Effective sharpness after training: per layer, the RMS pre-activation s_l on fresh inputs (softplus: beta*s_l);
  per unit, the Hermite tail share T_D/T_1 at D = 4, 8, 16 of phi on N(m_u, s_u^2) fitted to the unit's trained
  pre-activation, energy-weighted over units.
R (tractability of trained nets): on the trained sorting nets (single-gate arms), K2 (residual closure, Stein's
lemma for Cov(h, phi(u))) against MC and RQMC at equal FLOPs for E[output] under x ~ N(0, I_16), the same scoring as K.

## Predictions (P#) and refutation conditions
Theory-derived; "within a factor f" means the ratio lies in [1/f, f].
- P1 (smoothness does not shrink practical cut ranks). In grid C, no arm with effective sharpness >= 5 at init
  (softplus16, softplus4, gelu, silu, gsrelu0.5, smoothstep1) has raw r_in(0.1) below 2/3 of ReLU's at either config.
  Refuted if any does.
- P2 (ordered fronts are tight). softplus1 (chi_1 = 0.46) has raw r_in(0.1) >= ReLU's at config A. Refuted if lower.
- P3 (saturation). tanh has raw r_in(0.1) >= ReLU's at both configs. Refuted if lower at either.
- P4 (no visible asymptotics). The secant exponent log(r(0.03)/r(0.1))/log(0.1/0.03) is < 2 for every arm and config,
  and the analytic arms' exponents differ from ReLU's by < 0.4 at the same config. Refuted by an exponent >= 2, or by an
  analytic arm 0.4 or more below ReLU's at both configs.
- P5 (deductive estimation: depth problem is activation-independent). In grid K, K2 wins (rule above) in no cell with
  L = 16 at n <= 16, for every arm; and at each (n, L) the median over nets of nMSE_K2 * N_eq differs across the
  arms with sharpness >= 5 by less than 3x. Refuted by a win at L = 16, n <= 16, or by a 3x or larger spread among
  those arms in a majority of (n, L) cells.
- P6 (smoothness helps the integrator more). RQMC's gain over MC (1/N_eq over nMSE_RQMC) at N_eq is at least 1.5x
  ReLU's for softplus1, gelu and silu at n = 8 (median over nets and depths); at n >= 32 the gain is < 2 for every arm.
- P7 (softplus ladder). At fixed (n, L), median nMSE_K2 decreases from beta = 16 to beta = 1 (Spearman rho <= -0.6
  over the five betas) in >= 12 of the 16 (n, L) cells.
- P8 (sorting learnability). Mean final nMSE over seeds: gelu, silu, softplus4, softplus16 and swiglu within 20% of
  ReLU's (or better); softplus1 at least 20% worse than ReLU.
- P9 (training sharpens on a kinked target). Median over layers of the trained RMS pre-activation divided by its
  init value (sqrt 2) is >= 1.5 for softplus1 and softplus4 on sorting.
- P10 (PFN). Mean test KL of relu, softplus4, silu and swiglu within 20% of gelu's; softplus1 within 30%; the trained
  sharpness ratio (P9's statistic) is < 1.5 for every arm (no sharpening on a smooth target).
- P11 (trained tractability). On trained sorting nets K2 loses to sampling (rule above) for every arm; the trained
  smooth arms' energy-weighted tail share at D = 8 is >= 0.5x ReLU's.
- P12 (frontier). No arm both (a) is within 20% of ReLU's sorting nMSE and (b) wins grid R or has random-net raw
  r_in(0.1) at most half of ReLU's at both configs. Refuted by any such arm.

## Rules
- Primary statistics are the ones named above; anything else is post hoc and labelled so.
- Amendments are appended with a timestamp and hashed before the data they concern.
- CPU at nice -n 19 ionice -c3, at most 3 heavy processes; GPU only with main's OK (cap about $3), logged in GPU-LOG.md.
- Review: gpt-6-astra checks the theory note; a fresh GPT-6 session reviews the report, code and numbers before any
  claim leaves the lane.
