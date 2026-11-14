# Ledger of ideas (arc-noisy)

Each idea: what it is, the test, the outcome, why it died (or status). Nets: dev net 100 unless stated; sigma = 0.001.

1. Exact Gaussian closure (full bivariate-normal ReLU moments, Genz BVN) instead of ARC's truncated Wick/Mehler K=2.
   Test: nMSE at L = 8, 16, 32 vs kprop K=2. Outcome: 1.63e-4 vs 1.63e-4 at L = 8; 7.6e-4 vs 6.2e-4 at L = 32.
   Died: no gain; ARC's first-order Mehler truncation is not what limits K=2.
2. Recursive moment control (RMC; exact Gaussian closure + particle control-variate corrections of mean and covariance).
   Test: L = 8 and 32, N = 2^10..2^14 particles, 4 replicates, against the 2^26 truth.
   Outcome: layer 1 is 30x below MC on the same particles (Hermite tail of ReLU, as predicted), but the error grows with
   depth. Mean-only corrections: bias floor 4.8e-5 at L = 8 (nMSE 2.1e-4 / 8.5e-5 / 5.7e-5 at N = 2^10 / 2^12 / 2^14;
   MC 9.4e-4 / 2.5e-4 / 6.3e-5). Full covariance corrections: noisier (9.6e-5 at 2^12, L = 8; 3.5e-4 vs MC 2.6e-4 at
   L = 32). Top-k mode-block covariance corrections (k = 1, 4, 16) and particle covariance: no better.
   Died: injected correction noise and the covariance bias both accumulate over depth (mean errors are only weakly
   forgotten, covariance errors not at all), so at L = 32 it is no better than MC; at L = 8 it is 2-3x over MC on the same
   particles, i.e. below the Kerdock-MUB sampler (3x) once its extra cost (1.4-2.4x per particle) is counted, and far
   behind kprop K=3f (3.1e-6 at 1.26e10 FLOPs).
3. Homogeneous scale-mixture closure (HSC). Motivation: the top covariance mode at depth is the mean direction (relative
   norm SD 0.19 at L = 32, 63% of variance); ReLU is positively homogeneous, so a scale passes every later layer exactly,
   while a Gaussian closure damps it at kink neurons. Outcome: 1.41e-4 (L = 8), 6.9e-4 (L = 32) vs GC 1.63e-4 / 7.6e-4.
   Died: 9% better than the exact Gaussian closure but 11% worse than kprop K=2 at depth 32 (claims-review correction).
   Closure errors at depth are mostly orthogonal to the mean direction (69% for K=2, 84% for K=3f).
4. RMC on Kerdock-MUB particles (does a 5-design shrink the correction noise?). L = 8, N = 4096, 4 reps.
   Outcome: full 8.4e-5 vs Kerdock mean 8.6e-5 on the same particles; mean-only 7.4e-5 / 1.0e-4 / 1.17e-4 vs 8.6e-5 /
   1.06e-4 / 9.6e-5. Died: no gain over the Kerdock mean (the design already integrates the low-order part RMC removes).
5. Mode-cumulant closure (C3, PREREG gate). Readout oracle at layers 16 and 31 (true mean and covariance, cumulants from
   2^18 samples): Gaussian readout 2.7e-5 / 2.9e-5; per-neuron Edgeworth k3 5.0e-6 / 5.9e-6; k3+k4 2.3e-7 / 6.2e-7; the
   same cumulants rebuilt from the top-k covariance modes (k3+k4): k=1 2.6e-5 / 2.8e-5, k=8 1.4e-5 / 1.2e-5, k=16 7.0e-6 /
   6.6e-6. Replicated at layer 31 on nets 101 and 102: Gaussian 2.1e-5 / 3.6e-5; per-neuron k3+k4 1.2e-6 / 4.8e-7;
   top-8 modes 7.6e-6 / 8.8e-6. Died at the gate (needed >= 5x with k <= 8; got 1.9-4.1x). Precise reading (review
   finding 9): the readout needs the marginal cumulants of the NEXT pre-activations z_i = W_i h, which are projections of the
   full joint cumulants of h; cumulants built from the top-k covariance modes of h recover only a part of them. ("k3 only"
   in the logs means the skewness terms, including the gamma3^2 He4 term.)
6. Closure on a noisier copy of the net + sigma^2 extrapolation ("noise as a resource" for the sample-free entry).
   Noise-induced shift of the true L = 32 means (nMSE units, nets 100-102, 0, 1): 3e-7 to 9e-7 (0.003), 5e-5 to 1.2e-4
   (0.01), 4e-3 to 9e-3 (0.03), 0.35 to 0.83 (0.1), mostly a scale change from injected energy. kprop gains only 2x
   (K=3f at 0.03) to 3.5x (K=2 at 0.1). I rejected it "on arithmetic" (the shift is 10-1000x the closure error; extrapolation weights amplify closure errors).
   WITHDRAWN after the claims review: a large shift is what extrapolation is meant to cancel, and correlated closure
   biases need not be amplified. The reviewer's exploratory check (not preregistered, evaluation nets): K=3f-augmented at
   sigma 0.03 and 0.1, weights 1.0988 / -0.0988, gives depth-8 nMSE 1.83e-6 vs 1.92e-6 for the best single entry (+4%, CI
   above 1, cost 2.1e11); no tested depth-32 pair passed. Status: open, untested by me.
7. Linear control at layer m with a closure mean (H1 bias). Net 100, sigma 0.001, L = 32, OLS response from 2^14 paths:
   control bias nMSE with K=3f means 1.2e-7 (m = 4, R^2 0.58), 4.3e-6 (8, 0.68), 9.4e-6 (12, 0.77), 6.1e-5 (16, 0.75),
   6.0e-5 (20, 0.86), 3.6e-4 (24, 0.92); with K=2 or K=2-augment means 1e-4 or more from m = 8. The OLS response
   amplifies closure errors beyond the (m/L)^3 attenuation law. Status: scored on Kerdock replicates (hybrid.py).
8. Meet-in-the-middle oracle: h_m ~ N(true mean, true covariance), then the real layers m+1..32 (2^20 paths).
   m = 24: nMSE 2.1e-4 at layer 32 (sampling floor 9e-7). A two-moment description at depth fails even with exact
   moments and exact propagation afterwards; the output mean depends on the higher-order structure of p_24.
   m = 16: 5.0e-4; m = 8: 2.0e-4 (exact moments at layer 8, where K=3f's own error is 3e-6, still give 2e-4 at layer 32).
9. Copula seed (the program's neuron/copula code at depth): layer 24 replaced by exact per-neuron marginals (2^18 paths)
   with Gaussian dependence (normal-score correlation), then the real layers: nMSE 2.3e-4 at layer 32 (Gaussian seed 2.1e-4).
   Died: per-neuron marginals of the NEXT layer's pre-activations are enough for a one-step readout (item 5), but the
   empirical per-neuron marginals of h_24 (2^18 paths) with Gaussian dependence are not enough for 8 more layers; the
   depth-32 means depend on non-Gaussian dependence at the intermediate layers (for the replacements tested).
10. Description-guided RQMC. Active subspace of the depth-32 output (E[J^T J], 2,048 paths): top 1/4/16/64 input
   directions carry 3% / 10% / 30% / 71% (isotropic 0.4% / 1.6% / 6% / 25%). Not built and not tested. (An earlier note
   here claimed this caps the gain at 1.4x; the GPT-6 review showed derivative-energy shares do not bound variance shares,
   so that cap is withdrawn: description-guided RQMC is untested, not refuted.)
11. Error survival with noise (net 100, mean shift of 1% of |E h_l| in a random direction, coupled paths): energy kept at
   layer 32 from l = 4 / 8 / 16 / 24 / 28: 0.009 / 0.10 / 0.11 / 0.25 / 0.74 at sigma = 0.001 (predicted by
   prod(1 - arccos(c_k)/pi): 0.012 / 0.032 / 0.12 / 0.37 / 0.62); unchanged within 16% up to sigma = 0.03; 1.4-3.9x smaller
   at 0.1. One random direction per depth on one net: no sign of noise-driven forgetting at the primary noise levels in this
   measurement; it does not exclude direction-dependent effects.
12. H1 on the 24 evaluation nets (sampler v2, measured equal budgets, pilot + fit + closure charged). Depth 32, sigma
   0.001, median gain over Kerdock at 5e10-3e11: K=3f mean 0.64-0.92 (m = 4), 0.36-0.69 (8), 0.05-0.53 (12), 0.04-0.13
   (16), 0.03-0.07 (24); oracle (true layer-m mean, not a claim) 0.73-0.95 (4), 0.78-1.09 (8), 1.14-1.38 (12), 1.63-2.03
   (16), 4.09-5.47 (24). Killed by its preregistered test. Depth 8 (layer-2 control, K=3f mean): 1.13 at 5e10, 1.07 and
   1.08 at 2e11 and 2.72e11 (CIs above 1), 0.34-0.95 elsewhere; dominated by H2.
13. H2 (kprop estimate combined with the Kerdock mean, error models from dev nets 100-107). Depth 8, sigma 0.001 / 0.003
   / 0.01: beats the Kerdock sampler at every budget from 1e9 to 2.72e11 (1.18-10x, CIs above 1) and the better of the two
   entries by 1.04-1.79x; 0.96-0.99x at 3e11 (checkpoint rounding). Robust to charging kprop 1.3x. Depth 32: 0.92-1.02x.
   The win rests on ARC's K=2/K=3; it is the program's "meet in the middle" only in its simplest form.
14. C6 antithetic noise on Kerdock. Depth 32: 2.11x [2.01, 2.36] at 2.72e11 and above 1 at every budget at sigma = 0.1;
   0.88x at 0.03; 0.47x at 0.01. Depth 8: 1.29x at 2.72e11 at sigma = 0.1 (wins at 5e10 and 2e11-3e11 only), 0.5-0.6x
   below. A noise-regime result only.
15. Claims review (GPT-6, 04:39-04:50Z): H2 also wins narrowly at depth 32 at sigma = 0.03 (+1-6% over 6.8e10-2.72e11,
   both CIs above 1) and at sigma = 0.01 at 6.8e10 only (+2%); report headline narrowed to the primary noise levels.

## Part 2: rare output events (tails/PREREG-TAILS.md, 29 October)

16. U4, Gaussian importance sampling over x fitted to splitting particles in the event (the chain's conditional pulled
    back to the input). Exploration (dev 100) and kill test (dev 102-105, depth 4, 1e-8): relative variance per sample
    656-3,650 for 0, 3, 10 expanded directions. Killed.
17. U5, layer-wise twisted SMC (prior x, noise twisted). Diagnostic r = P(A and A') / P(A): per-net medians 0.46-1.1 on
    dev 102-103 at every (L, sigma), 0.26 at depth 32, sigma = 0.01 on dev 100. Killed (bar r <= 0.1).
18. U1, splitting on the top mode of h_{L-1} with a Feynman-Kac readout of all outputs. Works only for outputs aligned
    with the mode (one run within 50% for 13% of events at 1e-8, depth 32). Passed its kill test (17/32) only through
    near-zero estimates under the written rule; no evaluation-net win under both reach rules.
19. U2, saddlepoint (Lugannani-Rice, order-4 truncated CGF) on ARC's K3f description's moments. Kill test: +0.47 at
    depth 4, +0.12 at depth 8 (bar +0.20). Killed. Post hoc on evaluation nets +0.09 to +0.34 over ARC's readout.
    Confirmatory test on fresh nets 24-43: +0.24 (depth 4) and +0.14 (depth 8) at 1e-8; joint bar failed; positive
    pointwise CIs at every level 1e-6 to 1e-12 at both depths.
20. U3, Monte Carlo moments (2^20-2^22 paths) + the same saddlepoint. Alive; amortised wins at depth 4 for 1e-8 and
    1e-10 (37-38x, 55-58x over per-event splitting) and depth 8 at 1e-6 (9.4x over Monte Carlo), all sigma; 1e-12 at
    depth 4 is median-convention sensitive. Biased; loses per event.
21. Explored and dropped before the prereg: elliptical-slice moves in splitting (worse per evaluation than tuned pCN);
    directional (radial) Rao-Blackwellisation (the radial share of a 1e-12 event is only about e^2 in 256 dimensions).

## Part 3: sample-free readout of ARC's descriptions (tails/PREREG-TAILS-p3.md, 29 October)

22. P3A, chi-mixture order-4 saddlepoint on ARC's augmented factorized K = 3 description's moments. Kill test (dev
    102-105, depth 8, 1e-8): +0.219 over ARC's heaviside readout of the same description (bar +0.20). Confirmatory test
    on fresh nets 44-63 (CPU truths): +0.32 [0.23, 0.41] at depth 8, 1e-8 (bar +0.20, CI above 0): passed; +0.19 to
    +0.45 at every level 1e-4 to 1e-12; depth 4 +0.33 at 1e-8. Cost 1.7x ARC's augmented readout per threshold vector.
23. Tried in exploration (dev 100-101) and dropped: normal-inverse-Gaussian fitted to four cumulants (and its chi
    mixture), worse than the saddlepoint beyond 1e-6; RAD3 (no fourth cumulant), useless; higher moments from ARC's
    K3f (it implies k5 = 0 and an artifact k6). Priority 2 (Monte Carlo moments beyond order four) not run (CPU).
24. Part 3b (tails/PREREG-TAILS-p3b.md): the part 3 readout at depth 32 on part 2's 20 evaluation nets (sigma = 0.001,
    part 2 truths; ARC's augmented K = 3 run at depth 32 on the CPU, 12-24 min per net). Primary 1 (beats zero at 1e-4)
    failed: median relative error 0.79 [0.67, 1.00]. Primary 2 (gain over ARC's readout of the same description at
    1e-4) passed: +0.29 [0.21, 0.36]; positive at every tested level. ARC's readouts never beat zero at depth 32 either.
