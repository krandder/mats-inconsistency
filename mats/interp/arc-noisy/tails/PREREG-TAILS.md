# PREREG-TAILS — rare output events on the noisy deep ARC nets (work/arc-noisy/tails, part 2)

Written 2026-10-29 06:25Z, before any run on the evaluation nets and before the kill tests below. Brief (opus-mats,
~05:00Z): ARC's rare-event benchmark (their section 6.6) at depth 4 (ARC's shape) and at depths 8 and 32; sigma 0.001 and
0.003 primary, 0.01 secondary; thresholds so that p spans 1e-4 to 1e-12; competitors ARC's tail method, plain Monte
Carlo, splitting without psi, the Kerdock design where it applies; ours: importance sampling and splitting over x with
backward functions from the forward description, layer-wise twisted SMC where the noise gives leverage, deductive tail
estimates from the best description; metric FLOPs to 50% relative error per event and amortised over the 256 outputs,
all setup counted; the smallest p at which each method still beats zero.

## 1. Object and events (fixed)
- Nets: exactly part 1's (anet.py): ReLU, W_l iid N(0, 2/n), no bias, no LN, x ~ N(0, I_256), noise
  h_l = a_l + sigma r_l eps_l after every ReLU, r_l from rtable.json (depth-32 table; depth L is the truncation).
- Depth L in {4, 8, 32}. Event variable: the last pre-activation z_L = W_L h_{L-1} (L - 1 noisy ReLU layers, then a
  linear layer). L = 4 is ARC's section 6.6 shape (three ReLU layers, a linear layer, a threshold). Our first layer has
  variance 2/n where ARC's has 1/n, so ARC's threshold 3 corresponds to 3 sqrt 2 here; it is not used for scoring.
- sigma in {0.001, 0.003} (primary) and 0.01 (secondary).
- Evaluation nets: seeds 0-19 (20 per cell). Development nets: 100-107 (100-101 already used for exploration, kill tests
  on 102-105, any fitted constant from 100-105 only).
- Output neurons i = 0..7 of every net (a fixed rule; the weights are iid, so any fixed 8 are a random sample).
- Thresholds t_(i,k), k in {4, 6, 8, 10, 12}: set once per (net, L, i) at sigma = 0.001 by one adaptive splitting run
  (N = 2000, rho = 0.1, 5 pCN moves per level, pCN scale adapted towards acceptance 0.3, 13 levels, its own seed): t_(i,k)
  is its k-th level, so P(z_L,i > t_(i,k)) is about 10^-k. The same thresholds are used at sigma = 0.003 and 0.01.
- Events A = {z_L,i > t_(i,k)}: 40 per (net, L, sigma), 800 per (L, sigma) cell, 7,200 in all.

## 2. Truth
- Per (net, L, sigma, i): frozen-tuning generalised splitting (fixed levels, pCN scales frozen from an independent
  pilot; unbiased). Pilot: N = 1000, adaptive (1 - rho) levels with the five thresholds as stops, rho = 0.1, scales
  adapted on its own particles, own seed. Then R independent frozen runs of N = 2000 particles, 5 pCN moves per level
  (R = 32 at L = 32, R = 16 at L = 4 and 8). Reference = mean of the R estimates; relative SE from their spread.
  Float32 on GPU with TF32 off.
- Held-out cross-checks (rules fixed now):
  (a) split halves of the R runs: |log2(half 1 / half 2)| <= 1;
  (b) plain Monte Carlo, 2^23 paths per (net, sigma) (one depth-32 pass gives all three depths), at k = 4:
      |log(GS / MC)| <= 3 sqrt(se_GS^2 + se_MC^2) (relative SEs);
  (c) an independent splitting estimator with other tuning (rho = 0.3, N = 1000, R = 8, own pilot and seeds) for
      neurons 0 and 1 of every net at every (L, sigma) and all five thresholds: same 3-SE rule.
  An event is provisional if its relative SE exceeds 0.25 or a check that covers it fails. Every claim is stated with
  and without provisional events.

## 3. Methods and costs
Costs: flopscope 0.12 float32 rates (part 1: matmul M N (2K - 1), elementwise 1, normal draw 16, exp/log 16, cdf 96,
pdf 54), counted analytically for everything we run, including pilots, fits and readouts. ARC's kprop at its own
counter (NamedFlopCounter, its symmetric-tensor discounts), as in part 1. One path at depth L: c_path(L) = n 16 +
(L - 1)(n (2n - 1) + 3n) + n (2n - 1) (all 256 outputs); one pCN move of one particle: L n (16 + 3 + 1) + (L - 1)(n (2n - 1)
+ 3n) + (2n - 1) + 2.

Sample-free (deterministic; one run serves all 256 outputs):
- C-K2: kprop K = 2 with the heaviside readout (Gaussian tail).
- C-K3f, ARC's tail method: kprop factorized K = 3, simple, exact metric (ARC's run_lpe settings: use_avg_metric off,
  factor on), noise added to the second cumulant after every ReLU, last linear layer with bias -t and the heaviside step.
  One shared propagation plus one last step per threshold vector.
- C-K3fa: the same with the augmented kind, at L = 4 and 8 (too expensive at 32).
- U2 (ours, deductive): ARC's same K3f description, read out by the saddlepoint instead of the heaviside expansion. Four
  polynomial readouts give the description's E[z^p], p = 1..4, per output; cumulants k1..k4; Lugannani-Rice with the
  cumulant generating function truncated at order 4 (smallest saddlepoint theta > 0 with K'' > 0; if none, order 3, then
  Gaussian; fallbacks counted and reported). Variant U2r: chi_256 mixture over |x| of the conditional saddlepoint
  (noiseless homogeneity). Variant U2a: the K3f augmented description at L = 4 and 8. Cost: shared propagation + 4 last
  steps, for any number of thresholds.

Sampling and hybrid:
- C-MC: plain Monte Carlo, analytic: relative MSE (1 - p)/(p N), N paths at c_path(L).
- C-MCx (ARC's baselines): moments from 2^20 Monte Carlo paths, Gaussian and Edgeworth tails.
- C-GS, splitting without psi (per event): adaptive pilot (N = 500) to the event's threshold, then frozen runs. Its
  relative variance per frozen run is measured from the truth's frozen runs (sample variance of the R runs, N = 2000,
  levels up to the event); its pilot cost is that of a 500-particle pilot to the event's level.
- C-Kd: our Kerdock-MUB design (part 1's sampler: 129 MUBs, antipodal pairs, Haar rotation per block, stratified chi
  radii; unbiased under noise) in place of Monte Carlo inputs, at k = 4 and 6 only (where an input design can matter),
  2^16 points per replicate, 16 replicates.
- U1 (ours, psi from the description, amortised): splitting on the score S = v . h_(L-1), v the top eigenvector of
  E[h h^T] at the last hidden layer (from 2^14 pilot paths), run up (S) and down (-S); every output's events are read out
  from the particles at every level (Feynman-Kac readout, unbiased with fixed levels and frozen scales); output i uses
  the up run if v . W_L[i] > 0, else the down run. L = 32 and 8.
- U3 (ours, hybrid, amortised, biased): moments from M Monte Carlo paths (M = 2^20, 2^22, 2^24), read out by U2's
  saddlepoint (and the chi mixture).
- U4 (ours, psi = Gaussian description of x given the event): importance sampling over x from a Gaussian fitted to
  splitting particles in the event (mean and the top-k covariance directions, k in {0, 3, 10}).
- U5 (ours): layer-wise twisted SMC, i.e. any scheme that draws x from its prior and twists or resamples only the noise.
  Its relative variance is at least r/p - 1 with r = P(A and A') / P(A), A' the same event with the same x and fresh
  noise, so its gain over plain Monte Carlo is at most about 1/r; r is measured by splitting on min(z(x, eps), z(x, eps')).

## 4. Metric and scoring
- B50 = FLOPs to 50% relative error. Deterministic: its cost if |p_hat / p - 1| <= 0.5, else infinite. Randomised, with
  R replicates at setup C_s and run cost C_r: rMSE = mean_r (p_hat_r - p)^2 / p^2; B50 = C_s + C_r rMSE / 0.25 if
  rMSE <= 1 at the tested budget, else not reached there (infinite in medians).
- Per event: the cost of everything that produced that event's estimate. Amortised over the 256 outputs: the cost of a
  run that gives every output's estimate at one threshold each, divided by 256; for methods that share work across the
  five threshold levels (kprop propagations, U2's moments, U3's moments, U1's runs, Monte Carlo paths), the shared work is
  divided by 5 x 256 and per-threshold work by 256. Splitting per event is not amortised.
- Summaries per (L, sigma, k): median B50 over the 160 events (8 outputs x 20 nets), fraction of events reached, median
  relative error; bootstrap 95% CIs from 10,000 resamples of nets.
- Win: within a category (sample-free; sampling/hybrid), ours beats the best competitor of the category at (L, sigma, k)
  if the bootstrap CI of the median over nets of the per-net ratio [median B50 of the competitor over the net's 8
  events] / [same for ours] lies above 1. For two sample-free methods with ours at no higher cost, equivalently: the
  per-net fraction of events within 50% is higher for ours with the CI of the per-net difference above 0. Stated per
  sigma; the lowest sigma at which a win holds is named.
- Better than zero at k: median relative error (deterministic) or relative RMSE (randomised, at a stated budget) <= 0.9
  with the bootstrap CI upper end below 1. Budgets for randomised methods: ARC's K3f cost at that depth, and 3e11.
- Deductive = no sampling of the network at all (C-K2, C-K3f, C-K3fa, U2 and variants). Every other method samples.

## 5. Kill tests (development nets 102-105, sigma = 0.001, run after this file is hashed)
- U1: alive if at L = 32, k = 8 its amortised B50 is below C-GS's per-event B50 for at least half of the events.
- U2: alive if at L = 4 and at L = 8, k = 8, its fraction of events within 50% exceeds C-K3f's by at least 0.2.
- U3: alive if at L = 4, k = 8, with M = 2^22, at least half of the events are within 50%.
- U4: alive if at L = 4, k = 8, the relative variance per importance sample of the median event is <= 100.
- U5: alive if r <= 0.1 for the median event at k = 8 at some (L, sigma).
A killed candidate is reported with its kill-test numbers and not run on the evaluation nets.

## 6. Predictions (from the exploration in section 7)
- C-K3f: within 50% for most events at 1e-4 (L = 4, 8), for about half at 1e-6 and a quarter at 1e-8 (L = 4), under a
  fifth beyond 1e-6 at L = 8; at L = 32 wrong even at 1e-4 (many negative estimates). Better than zero to about 1e-6
  (L = 8) and 1e-8 (L = 4).
- U2: within 50% for most events to 1e-6 at L = 4 and 8 and to 1e-8 at L = 4; median relative error <= 0.9 (better than
  zero) to 1e-12 at L = 4 and 8; no better than C-K3f at L = 32 (both fail). Kill test passes; sample-free win over C-K3f
  at L = 4 and 8 for k >= 6 at every sigma.
- C-GS reaches 50% for every event; per-event B50 at 1e-12 about 1e10-1e11 (L = 4), 1e11 (L = 8), 1e12 (L = 32).
- C-MC (amortised) is the cheapest sampler at 1e-4 only; C-Kd within 1.5x of it.
- U1 fails its kill test (works only for the few outputs aligned with the top mode). U4 and U5 fail theirs.
- U3: passes at L = 4; at L = 8 within 50% for about half the events at 1e-6; fails at L = 32.
- sigma: no method's result changes beyond noise between 0.001 and 0.003; small changes at 0.01 (L = 32 most).

## 7. What was seen before this file (development nets 100-101, sigma = 0.001, 32 outputs per net)
- Splitting (pCN, frozen tuning): median relative variance of one frozen run with N = 1000 at 1e-4 / 1e-8 / 1e-12:
  0.04 / 0.10-0.14 / 0.21-0.24 (L = 4), 0.05 / 0.15-0.17 / 0.38-0.49 (L = 8), 0.07 / 0.31-0.47 / 1.1-2.6 (L = 32).
  Elliptical-slice moves were worse per evaluation than tuned pCN (L = 4, one output).
- Gaussian importance sampling fitted to splitting particles (L = 4, one output): relative variance per sample 7e2-3e4 at
  1e-6 to 1e-12 for every k tried.
- Tails at L = 32 are far from Gaussian (1e-12 at up to 13.5 sd for some outputs, lighter than Gaussian for others).
- C-K3f within 50% at 1e-4 / 1e-6 / 1e-8 / 1e-10 / 1e-12: 0.97 / 0.53-0.62 / 0.28 / 0.16-0.19 / 0.09-0.16 (L = 4);
  0.47-0.59 / 0.12-0.19 / 0.09-0.12 / 0.06 / 0.06 (L = 8); 0 (L = 32, 69-78% of estimates <= 0).
- U2 (SP4 on the K3f moments): 0.97 / 0.91-0.94 / 0.62-0.72 / 0.41-0.50 / 0.34-0.41 (L = 4); 0.84-0.88 / 0.44-0.56 /
  0.28 / 0.19-0.25 / 0.12-0.25 (L = 8); fails at L = 32 (no saddlepoint for most outputs). Median log10 error at 1e-12
  +0.13 / -0.12 (L = 4), -0.66 / -0.48 (L = 8). The radial variant was similar.
- Oracle moments (2^26 Monte Carlo paths) with SP4: 1.0 / 0.94-0.97 / 0.84-0.91 / 0.62-0.66 / 0.50-0.62 (L = 4);
  0.91-0.94 / 0.62-0.66 / 0.31-0.41 / 0.19 / 0.09-0.12 (L = 8); the radial variant at L = 8: 0.97 / 0.78 / 0.59-0.62 /
  0.44-0.47 / 0.34-0.41.
- U1: at L = 32 rMSE <= 0.25 at 1e-8 for 17% of the outputs with v . W_L[i] > 0 and for 5% of the others; at L = 8 for
  none.
- U5 (net 100, k = 8): r = 1.0 / 1.0 / 0.92 (L = 4, sigma = 0.001 / 0.003 / 0.01), 0.99 / 0.98 / 0.81 (L = 8),
  0.72 / 0.60 / 0.26 (L = 32).

## 8. Compute
One on-demand L4 VM with a job loop (tails/gpu/launch_dev.sh), logged in ../GPU-LOG.md, deleted at the end with an
empty instance list verified. Expected: truth about 3 GPU-hours, everything else about 1; within the remaining cap
(about $43 of the lane's $50). CPU (nice, at most 8 processes): kprop.

## Amendment 1 (06:39Z, before any evaluation-net result was looked at; scoring details only)
1. Path cost: the formula in section 3 left out the noise draws; scoring uses c_path(L) = 16 n + (L - 1)(n (2n - 1) +
   n + 18 n) + n (2n - 1) (about 3% more for Monte Carlo).
2. Biased sampling estimators (C-MCx, U3): their error does not shrink as 1/budget, so B50 is the cost of the smallest
   tested moment budget (M = 2^20, 2^22 Monte Carlo paths, 4 replicates each) whose relative MSE over the replicates is
   at most 0.25; infinite if none.
3. Amortisation of Monte Carlo: a run sized for level k (N = 4(1 - p)/p paths) serves every output's events at levels
   4..k, so its amortised cost per event is N c_path / (256 (number of levels served)).
4. Kerdock design: per point no input draw and a first layer of n^2 + n (one matvec per antipodal pair), plus part 1's
   per-block setup; unbiased, so B50 = setup + run cost x rMSE / 0.25.
5. C-GS pilot cost: a 500-particle pilot with the same number of levels as the truth's tuning pilot up to the event's
   threshold (computed, not run).

## Amendment 2 (06:45Z, after the kill tests, before any evaluation-net result was looked at)
Kill-test outcomes (development nets 102-105, sigma = 0.001, 32 events per depth; tails/killtests.py):
- U2 (saddlepoint on ARC's K3f description): fraction within 50% at k = 8 is 0.66 vs C-K3f's 0.19 at L = 4 (+0.47) but
  0.31 vs 0.19 at L = 8 (+0.12, below the +0.20 bar). Killed. U2r: +0.47 and +0.15, killed.
- U3 (Monte Carlo moments, 2^22 paths, + saddlepoint): 0.81 of events within 50% at L = 4, k = 8. Alive.
- U1 (mode splitting): amortised B50 below C-GS's at L = 32, k = 8 for 0.84 of events under the literal rule, but that
  rule counted runs whose estimates were all zero (relative MSE exactly 1) as reaching 50% at four times the cost. With
  the strict reading (relative MSE below 1 at the tested budget, so all-zero replicates do not count), 0.47. Killed
  (the correction goes against our candidate).
- U4 (Gaussian importance sampling): median relative variance per sample 656-3,650 (k = 0, 3, 10 directions). Killed.
- U5 (layer-wise twisted SMC): median r between 0.46 and 1.1 at every (L, sigma) on nets 102-103 (gain over Monte Carlo
  at most about 2x). Killed.
Scoring change (applies to every randomised method): a method reaches 50% at a tested budget only if its relative MSE
there is strictly below 1 (the B50 formula of section 4 otherwise credits all-zero replicates).
Deviation, stated now: U2's moments come out of the same kprop propagation as C-K3f at no extra cost, so U2, U2r and U2a
are scored on the evaluation nets and reported, labelled post hoc. They are excluded from every win claim; the verdict
on U2 stays "killed by the pre-registered kill test at L = 8".

## Amendment 3 (06:55Z, correction found by the code review; before any evaluation-net result was looked at)
Amendment 2 misreported U1's kill test: the code used relative MSE < 0.999, not the "strictly below 1" written there.
Under the rule as written, U1's amortised B50 beats C-GS's per-event B50 for 17/32 = 0.53 of the events at L = 32,
k = 8, so U1 is alive and is run on the evaluation nets at L = 32. (Readings: <= 1, 0.84; < 1, 0.53; < 0.999, 0.47;
relative RMSE <= 0.9, the prereg's better-than-zero margin, 0.41.) The pass rests on events whose eight replicate
estimates are essentially zero (relative MSE 0.99999), where B50 extrapolates from no evidence of convergence. So U1 is
scored on the evaluation nets under both reach rules (relative MSE < 1, and relative RMSE <= 0.9), and a U1 win is claimed
only if it holds under both. The prereg-conforming saddlepoint root search (smallest root with K'' > 0, skipping regions
where K'' <= 0; the earlier code stopped at the first such region) changes none of the kill-test numbers.
