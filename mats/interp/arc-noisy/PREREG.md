# PREREG — beating ARC on noisy deep random MLPs (work/arc-noisy)

Written 2026-10-29 02:20Z, before any main run of our candidates on the evaluation nets. Brief: Kelvin's goal relayed by
opus-mats ("We dont need to use K=2 ... beat or improve on ARC, for width=256, deep (8 or 32 layers), both their normal and
sampling entries, using our new methods"; small noise for everyone). Noise grid amended by Kelvin before this file was
written (2026-10-29 ~01:00Z): sigma in {0.001, 0.003, 0.01, 0.03, 0.1}, with 0.001 and 0.003 primary.

## 1. Object and estimand (fixed)
- ARC's family: ReLU, W_l iid N(0, 2/n) (every layer, contest form), no bias, no LN, x ~ N(0, I_n), n = 256.
- Noise after every ReLU: h_l = a_l + sigma r_l eps_l, r_l = RMS of a_l in the noisy net, fixed per net by a seeded
  2^12-path pilot (rtable.json). Part of the object; nobody is charged for it.
- Estimand: E[a_L] per neuron (final post-ReLU means), L = 8 and 32; depth 8 = truncation of the depth-32 net.
- Nets: evaluation seeds 0-23 (24 nets per cell); development/fitting seeds 100-111 (never scored in the main tables).
  Noiseless control nets 200-204 (code checks only, no claims).
- Truth: plain Monte Carlo on GPU, 2^26 paths per (eval net, sigma) (2^25 for dev nets), float32 matmuls with TF32 off,
  float64 accumulation. Truth MC variance (1/N on the normalized scale) is subtracted from every reported nMSE and stated.

## 2. Metric, budgets, accounting
- nMSE = mean_j (est_j - truth_j)^2 / mean_j Var(a_L,j) (ARC's convention; plain MC with N paths has expectation 1/N).
- Budgets 1e9 to 3e11 FLOPs (grid 1e9, 2e9, 5e9, 1e10, 2e10, 5e10, 6.8e10, 1e11, 2e11, 2.72e11, 3e11).
- One convention: flopscope 0.12 float32 rates (matmul M N (2K-1), elementwise 1, exp/log/erf 16, normal draw 16,
  norm.cdf 96, norm.pdf 54), applied analytically to every method we implement, including all setup (pilots, fits,
  closures). ARC's kprop is charged with ARC's own counter (torch FlopCounterMode with their symmetric-tensor discounts),
  which favours kprop; stated wherever kprop appears. Implementation tricks of the contest entries (pruning dead units,
  Strassen, FWHT) are not used by any sampler, ours included.
- A deterministic method has one cost C and one error; at budget B >= C its error is its error. "Best entry of a type at
  budget B" = lowest nMSE among entries of that type with cost <= B.
- Per net: nMSE of randomized methods = mean over R independent replicates (R = 8-16). Across nets: median and mean with
  bootstrap 95% CIs (10,000 resamples over nets); ratios of two methods are computed per net and then summarized.

## 3. Competitors (all on the same noisy nets)
- Sample-free: ARC mlp_kprop at main 93d091a (TODO.md notes older K>=2 outputs are stale; we run current code), K = 1,
  2 (simple, augment), 3 factorized (simple, augment), 4 factorized where affordable (L = 8), average metric on; noise
  injected exactly (adds s_l^2 to the second cumulant's diagonal after each nonlinear step). Contest sample-free entries
  (Puffi SSC etc.) are not public code; reported from the forum as unreproduced numbers only.
- Sampling: plain MC (expected nMSE 1/N analytically; checked by simulation on 8 nets), MC with the last noise integrated
  per neuron, scrambled Sobol RQMC, and our reproduction of the best published Phase-1 sampler class: the Kerdock
  mutually-unbiased-bases design (129 MUBs of R^256 built from the Z4 Kerdock code, verified exactly; antipodal pairs,
  Haar rotation per block, stratified chi radius per pair so every point is N(0, I) and the estimator is unbiased under
  noise). Labelled "Kerdock-MUB sampler (our reproduction of the class used by Team Puffi, kaileh57, ely2sh, baltsat;
  without their pruning or fast-multiply tricks)".

## 4. Win definition
A win on an entry type: a method of that type with lower nMSE than the best entry of that type at every budget in a stated
contiguous range of the grid, with the bootstrap 95% CI of the per-net ratio (competitor / ours, median and mean) above 1
at every such budget, on the 24 evaluation nets, separately for L = 8 and L = 32. For every result, the smallest sigma at
which it holds is stated; sigma = 0.001 and 0.003 are the primary settings.

## 5. What was seen before this file (dev net 100, sigma = 0.001 only)
- Noise share of output variance at L = 32: 0.1% (0.001), 0.5% (0.003), 5% (0.01), 30% (0.03), 85% (0.1).
- Collapse: at L = 32 the top covariance mode is the mean direction (cos 0.998), the relative norm fluctuation has SD 0.19
  and carries 63% of the variance; participation ratio 2.4.
- kprop on net 100: K=2 1.6e-4 (L=8, 4.1e8 FLOPs) / 6.2e-4 (L=32, 1.65e9); K=2 augment 4.9e-5 / 1.8e-3; K=3 factorized
  3.1e-6 (L=8, 1.26e10) / 3.2e-4 (L=32, 2.06e11). Our exact Gaussian closure equals K=2 (1.63e-4 at L=8).
- The Gaussian closure restarted from exact moments at layer 24 already has nMSE 2.8e-4 at layer 32; from layer 31, 2.9e-5
  (the two-moment readout floor): deep layers inject about 3e-5 per layer.
- Linear predictability of a_32 from h_m (R^2): 0.26 (m = 0), 0.71 (8), 0.83 (16), 0.93 (24), 0.99 (31).
- Kerdock-MUB sampler at L = 8: nMSE x N = 0.32-0.35 (3 replicates), Sobol 0.42-0.63, MC 0.9-1.1.

## 6. Candidates, predictions and kill tests
C1 Recursive moment control (RMC): exact Gaussian closure whose non-Gaussian mean/covariance corrections come from
   particles of the true net (control variates on the first two sample moments). Already tested on net 100 before this
   file: killed (see ledger). Predictions not applicable.
C2 Homogeneous scale-mixture closure (HSC): Gaussian closure that carries the norm mode as an exact scale (ReLU positive
   homogeneity). Seen on net 100: 6.9e-4 vs K=2 6.2e-4 at L=32 -> killed by the >= 2x rule below.
C3 Mode-cumulant closure (MCC, sample-free): exact Gaussian closure for mean/covariance plus per-neuron Edgeworth
   corrections (orders 3-4) whose cumulants come from the joint cumulants of the top-k covariance modes, propagated layer to
   layer. Built only if the readout oracle (true moments + mode cumulants) removes >= 5x of the Gaussian readout error with
   k <= 8 at layers 16 and 31.
   Kill test (dev nets 100-103, sigma = 0.001): at L = 32 must beat both K=2 and K=3 factorized by >= 2x (median) at <= the
   FLOPs of K=3 factorized; at L = 8 must beat K=3 factorized at <= its FLOPs. Prediction: fails at L = 8 (K=3f is strong
   there), 30% chance of passing at L = 32.
C4 Hybrid H1: Kerdock-MUB carrier + linear control at layer m (a_L ~ A_m a_m fitted on a separate 4,096-path pilot) with the
   control mean from a sample-free closure (kprop K=3f or C3) at layer m; m in {4, 8, 12, 16, 24} at L = 32, {2, 4} at L = 8.
   Prediction: best m gives a gain over the plain Kerdock-MUB sampler of 1.1-1.4x at L = 32 (Team Puffi reported 17.6%
   from a layer-9 control); with the closure bias the gain shrinks above m = 12. Kill test: gain >= 1.5x with CI above 1 at
   all budgets in [5e10, 3e11] at L = 32, sigma <= 0.003, else reported as a minor result.
C5 Noise as a resource (object effect, not a method claim): kprop's nMSE at L = 32 vs sigma; prediction: within 1.5x of
   sigma = 0.001 for sigma <= 0.01, lower by >= 2x at sigma = 0.1. Samplers with noise integrated at the last layer gain
   < 2% at sigma <= 0.01.

## 7. Ledger of ideas (each with why it died) is kept in LEDGER.md.

## Amendments
(none yet)

### Amendment 1 (2026-10-29 02:42Z, before any result of the methods it adds)
- C6 Antithetic noise on the Kerdock-MUB design ("noise as a resource" for the sampling entry): each design point runs two
  paths with noise +e and -e, averaged; cost counted as two full paths. Predictions (L = 32, 24 eval nets, median gain over
  the plain Kerdock-MUB sampler at equal cost, budgets >= 1e10): sigma = 0.1: 2-5x; sigma = 0.03: 0.9-1.2x; sigma = 0.01:
  below 1 (about 0.55x). At L = 8: sigma = 0.1 about 1.0x. A win at sigma = 0.1 would be a noise-regime result, not a
  result about ARC's (near-noiseless) problem.
- H1 scoring detail: the control's response A_m is fitted on a separate 4,096-path plain-MC pilot whose cost is charged,
  which makes H1 lose at budgets below about 2e10 by construction; H1 is judged on budgets >= 5e10. The same scorer is
  also run with the true layer-m mean in place of the closure mean ("oracle"), to report the gain a perfect description
  at layer m would give; oracle numbers are never claims.
- C3 failed its gate (ledger item 5) and was not built.

### Amendment 2 (2026-10-29 03:22Z, before any evaluation-net result of H2; after the GPT-6 code review)
- H2 (new): est = w * kprop_v + (1 - w) * Kerdock mean on the rest of the budget, w = (V_j/N_j)/(b2_v + V_j/N_j), with
  b2_v and V_j fitted on development nets 100-107 only; at each budget the variant (or plain Kerdock) with the lowest
  predicted error is used. Its sample-free part is ARC's kprop, so any H2 win is a hybrid-entry win built on ARC's own
  method, reported as such. Predictions (sigma <= 0.003, 24 eval nets, gain = Kerdock / H2 per net): L = 8: median gain
  >= 1.2 with CI above 1 at every budget from 1e9 to 2e11, about 1.0-1.2 at 2.72e11-3e11; L = 32: gain above 1 only at
  1e9-5e9 (where K=2 and Kerdock are comparable), about 1.0 from 1e10 up.
- Review fixes that change scoring (not predictions): samplers re-run as v2 (48 measured checkpoints with exact costs,
  Kerdock setup charged per 66,048-point block); H1 and H2 compared with plain Kerdock at measured equal budgets on the
  same replicates (no 1/N rescaling); H1 charged pilot paths + one least-squares fit + the closure's ARC-counter cost at
  depth m (results/kprop_costs.json). Version-1 sampler files are used only where no v2 file exists (Sobol at sigma >= 0.01).
