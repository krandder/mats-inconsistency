# Pre-registration: numerical checks for the cut-rank extension (nonlinear readouts, bits, identifiability)

Lane cut-rank-ext, bead pool-4fm6vg. Written 2026-10-31 ~11:15Z, before any evaluation run. Hash in `prereg_hashes.txt`.
Theory: `FORMAL-ext.md` (hashed 11:0xZ). CPU only, `nice -n 19 ionice -c3`, at most 3 heavy processes.
Pilot (allowed, declared): n = 8, c = 2, d = 2, F = 16 fronts, for timing and correctness only; never scored.

## NC1. Linear against nonlinear readouts of the same front numbers (genuine nets)

Purpose: the theory says the precision exponent constrains linear readouts only, and that on ray-like fronts a
Lipschitz readout of the mean is exact (FORMAL-ext C2). NC1 measures, on the cut-rank lane's ensemble, how much a
nonlinear readout of n (or 1 + n + n(n+1)/2) front numbers gains over the best linear readouts of the same numbers and
over the optimal linear protocol of that size, on window and merged-front configurations.

- Nets: the cut-rank lane's ensemble and seeds (`work/cut-rank/lib/crnet.py`, cfg_tag(n, c, Lp), front_weights,
  back_weights): He-ReLU, no bias, noiseless. Configurations (n, c, d): (8,4,4), (8,8,8), (16,8,8) [window] and (8,64,8)
  [merged front].
- F = 512 fronts (first 256 = train, last 256 = test), B = 32 backs, all n output neurons: 32 n columns.
- Inputs: unit directions from new streams (keys (tag, 21, f, split)), radius integrated exactly (E|x| for means, n
  for second moments). Three splits of Np = 8192 directions per front: split A gives the front numbers (mean
  mu = E h_c, covariance Sigma = Cov h_c, third cumulant for ARC K = 3); splits B and C give two independent estimates
  of M(f, b) and of V(f, b); the target is (M_B + M_C)/2; the Monte Carlo floor is mean((M_B - M_C)^2)/4.
- Primary criterion: scale-normalised (as `work/cut-rank/scratch/scale_norm.py`: front rows divided by s_f, back
  columns by t_b, V rescaled the same way; fronts with s_f = 0 dropped). Every estimator below is positively
  homogeneous of degree 1 in the front's scale, so the normalisation commutes with it. Secondary: raw criterion.
  Score: Err_X / Vbar on the test fronts (Vbar from the test fronts), Err_X = mean over (f, b, j) of (target - X)^2.
- Estimators (all from split A):
  - MF: g_b(mu) (naive mean propagation; s = n numbers; Lipschitz readout).
  - K2X: Gaussian closure through the back from N(mu, Sigma), exact Gaussian ReLU means and covariances
    (`work/arc-noisy/gauss_relu.py`); s = n + n(n+1)/2.
  - ARC2: ARC's cumulant propagation (`work/arc-noisy/ext/mlp_cumulant_propagation`, kprop_harmonic: coerce_input,
    linear_kprop with set_metric 2.0, nonlin_kprop with the ReLU Wick coefficients, kind SIMPLE, k_max = 2) started
    from the cut cumulants {1: mu, 2: Sigma}. ARC3 (k_max = 3, adding the third cumulant) at n = 8 only, if it runs in
    under 30 min per configuration; otherwise dropped and reported as not run.
  - LIN1: least squares on [1, mu] fitted on train fronts per column, scored on test fronts (ridge 1e-8 x trace).
  - LIN2: the same on [1, mu, vech(E h h^T)] (1 + n + n(n+1)/2 features).
  - SVD(s): the optimal rank-s linear protocol on the test fronts, in-sample tail of the pair-averaged Gram
    (`work/cut-rank/lib/spec.py` tails_in with [M_B, M_C]); s = n + 1 and s = 1 + n + n(n+1)/2. Biased low: it is an
    optimistic floor for every linear protocol of that size.
- Predictions:
  - P1 (merged, (8,64,8), primary): Err_MF / Vbar <= 0.1 x SVD(n+1) / Vbar.
  - P2 (window, each of the three configs, primary): Err_MF / Vbar >= 0.1.
  - P3 (window, primary): Err_ARC2 < Err_LIN2 in at least 2 of the 3 window configurations.
  - P4 (all, primary): Err_LIN1 >= 0.8 SVD(n+1) and Err_LIN2 >= 0.8 SVD(1+n+n(n+1)/2) (consistency of the framework).
  - Reported without prediction: K2X against ARC2; ARC3; raw-criterion versions; MC floors.

## NC2. Bits on the mean (all NC1 configurations; the decoder is MF)

- Code: the norm |mu| quantised log-uniformly with 16 bits over [1e-6, 1e3] x s_f; the direction mu/|mu| (nonnegative)
  by uniform scalar quantisation of each coordinate on [0, 1] with q bits, then renormalised; B = n q + 16,
  q = 1..12. Decoder: g_b(quantised mu). Excess(B) = Err_MF(B) - Err_MF(unquantised).
- Predictions (primary criterion):
  - P5 ((8,64,8)): over the q where Excess > 2 x max(MC floor, Err_MF(unquantised)), the least-squares slope of
    log2 Excess against q lies in [-3, -1] (2^{-2q} expected for scalar quantisation).
  - P6 ((8,64,8)): if Err_MF(unquantised)/Vbar <= 0.005, the smallest q with Err_MF(B)/Vbar <= 0.01 has
    n q + 16 >= log2 r_in(0.1) - 1 (Theorem A, with r_in the in-sample eps-rank of the same normalised operator at
    eps = 0.1) and n q + 16 <= 12 n + 16.

## NC3. Identifiability: the differential rank of theta_f -> X(f) (conjecture I1)

- Configurations (n, c, d) = (4,2,2), (4,3,2), (6,2,2), (6,3,2); K = 64 backs (keys as cut-rank back_weights with a new
  tag 'nc3'), output coordinates (b, j): 64 n. Three random fronts per configuration, drawn in order from seeds 0, 1, ...
  and kept only if no unit at layers 1..c is identically zero on the input sample (dead units create extra exact
  degeneracies); the number rejected is reported.
- M_N(theta_f, b) = E|x| mean over N = 65536 unit directions of g_b(h_c(u)); two independent direction sets A, B.
  Jacobian J = d M_N / d theta_f (exact backpropagation per sample, float64), p_f = n^2 c columns.
- Known symmetry directions at theta_f: input rotations dW_1 = W_1 A (A antisymmetric, n(n-1)/2) and hidden-unit
  rescalings (dW_l = e_i e_i^T W_l, dW_{l+1} = -W_{l+1} e_i e_i^T, l = 1..c-1: n(c-1)). Q = orthonormal basis of
  their orthogonal complement (dimension m = p_f - n(n-1)/2 - n(c-1)).
- Noise level nu = ||(J_A - J_B) Q||_op / 2; restricted spectrum = singular values of ((J_A + J_B)/2) Q.
- Predictions:
  - P7: for every configuration and every accepted front, all m restricted singular values are >= 10 nu.
  - P8: the rescaling directions give singular values <= 1e-8 x the top singular value (exact symmetry); the rotation
    directions give values <= 3 nu (symmetry in expectation only).

## Rules
- Each prediction is scored as written; misses are reported. Anything added after the first evaluation result is
  labelled post hoc. Seeds, sizes and estimators above are fixed; a crash is fixed and rerun with a note.

## Amendment 1 (11:29Z, after the first NC1 results for (8,4,4) and (8,64,8) were seen)
- Fix, per the "crash is fixed and rerun" rule: ARC's released code floors variances at 1e-10 in absolute units, and the
  deep (8,64,8) fronts have cut scales down to 1e-7, so ARC2/ARC3 as run were numerically corrupted on low-scale fronts
  (32% / 62% of pairs above 100 Vbar under the primary criterion). The registered estimator is recomputed on unit-scaled
  cut cumulants and rescaled (`scratch/arc_unit_rerun.py`; exact by positive homogeneity apart from the floors). Both
  versions are reported; the unit-scaled one is used for P3. K2X is recomputed the same way as a control.
- Post hoc diagnostics, not registered: `scratch/arc3_diag.py` (per-pair medians and blow-up fractions) and
  `scratch/arc_zero_cov_check.py` (ARC's K = 2 code fed (E h_c, 0) equals mean propagation).

## Amendment 2 (after the final GPT-6 review, 12:10Z; post hoc corrections, all reported as such)
- NC2: the registered norm code decoded with a front-dependent scale s_ref that was never transmitted (review finding 9).
  Rerun with a global norm codebook (16 bits, log-uniform on [1e-12, 1e4] absolute, clipped, overflows counted);
  `scratch/post_review_fixes.py nc2g`. P5 and P6 are rescored on the corrected code; the original numbers are kept.
- ARC3: the registered runs supplied cumulants 1-3 only, while Wu et al.'s basic K = 3 also uses the scalar harmonic
  component of cumulant 4 (review finding 10). Rerun with the fourth cumulant on unit-scaled inputs
  (`post_review_fixes.py arc3k4`); the earlier ARC3 runs are relabelled as an incomplete-input variant.
- Moment-sample convergence of ARC2 and K2X (review finding 11): 64 test fronts, cut moments from 8,192 against 65,536
  directions (`post_review_fixes.py conv`), on (8,8,8) and (8,64,8).
