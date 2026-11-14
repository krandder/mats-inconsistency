# PREREG-TAILS-p3 — a better sample-free readout of ARC's own descriptions (part 3)

Written 2026-10-29 10:33Z, before the kill-test scoring below and before any method was scored on nets 44-63.
Brief (opus-mats, under Kelvin's delegation of experiment choice): priority 1, a better readout of ARC's own description
(no sampling) that turns the depth-8 improvement of part 2 (+0.14 against a +0.20 bar) into a claim; priority 2, a
beyond-four-moment readout of Monte Carlo moments; pre-register a kill bar on development nets and a confirmatory test
on fresh nets; CPU only, at most 4 processes, about 3 h.

## What was seen before this file (development nets 100-101 only, 64 events per level, sigma = 0.001)
- ARC's K3f description gets mean, variance and skewness of every output essentially right (skewness correlation with a
  2^26-path Monte Carlo value 0.996-0.998), but its excess kurtosis correlates only 0.21-0.26 with the true one: it is
  roughly the same for every output. The fifth cumulant it implies is exactly 0 and the sixth is an artifact (-10 k3^2),
  so it carries nothing beyond order four. ARC's augmented K3 description (K3fa) has excess-kurtosis correlation 0.63
  (depth 4) and 0.50 (depth 8), with half the median absolute error.
- Readouts tried, fraction of events within 50% at 1e-8, depth 8 (depth 4): ARC's heaviside on K3f 0.11 (0.28), on
  K3fa 0.09 (0.31); saddlepoint SP4 on K3f 0.28 (0.67), on K3fa 0.38 (0.80); chi-mixture saddlepoint RAD4 on K3f 0.33
  (0.58), on K3fa 0.55 (0.73); a normal-inverse-Gaussian fitted to the four cumulants and its chi mixture 0.25-0.30
  (0.47-0.56); RAD3 (dropping the fourth cumulant) under 0.05. With 2^26-path Monte Carlo moments: SP4 0.36, RAD4 0.61.
- Parts 1-2 also saw nets 0-23 (evaluation, part 2) and 24-43 (part 2b confirmation); none of them is used here.

## Candidate (priority 1, deductive, no sampling of the network)
P3A: RAD4 on ARC's augmented description. ARC's factorized K = 3, AUGMENT kind, exact metric (run_lpe settings: factor
on, use_avg_metric off), our weights and noise (added to the second cumulant after every ReLU), propagated to the last
pre-activation; four polynomial readouts give the description's E[z^p], p <= 4 (kprop_tail.kprop_all); the readout is
deduce.radial(t, moments, order = 4): the chi_256 mixture over |x| of the order-4 Lugannani-Rice saddlepoint for
g = z / |x| (noiseless homogeneity), unchanged code. If it returns no value (no admissible saddlepoint at some radius),
fall back to SP4 on the same moments, then order 3, then Gaussian; fallbacks are counted.
Competitors: ARC's heaviside readout of the same K3fa description (C-K3fa, the matched comparison) and of the cheaper
K3f description (C-K3f, ARC's tail method as run in part 2). Costs (ARC's counter): P3A = K3fa shared propagation + 4
moment readouts (about 2.0e11 FLOPs at depth 8 for all outputs and any number of thresholds) + the readout (about
1e5 FLOPs per event); C-K3fa = shared + one heaviside step per threshold vector (about 1.2e11 at depth 8).

## Kill test (development nets 102-105, part 2 truths, 32 events per depth, sigma = 0.001)
P3A is alive if at depth 8, 1e-8, its fraction of events within 50% exceeds C-K3fa's by at least +0.20. If it fails,
stop (no confirmatory scoring).

## Confirmatory test (fresh nets 44-63, never used before)
- Truth: the part 2 protocol on CPU with smaller sizes, fixed at 10:12Z when the runs started: thresholds.py at
  depths 4 and 8 (adaptive splitting, N = 2000, 13 levels), truth.py with R = 8 frozen runs of N = 1000, pilot N = 500,
  cross-check (c) with R = 4, N = 500 on outputs 0-1; no Monte Carlo cross-check (CPU). Provisional rule as in part 2
  (relative SE > 0.25, split halves > 1 bit, or a failed cross-check).
- Primary: at depth 8, 1e-8, the mean over the 20 nets of the per-net difference (P3A minus C-K3fa) in the fraction of
  its 8 events within 50% is at least +0.20 and its bootstrap 95% CI (10,000 resamples of nets) lies above 0. The
  claim, if it passes: on fresh nets, this sample-free readout of ARC's augmented description puts at least a fifth more
  of the depth-8 1e-8 events within 50% than ARC's own readout of that description.
- Secondary (reported, not part of the claim): the same at depth 8 for 1e-6, 1e-10, 1e-12; at depth 4 if its K3fa runs
  finish within the time box; P3A minus C-K3f; P3A's fraction within 50% and better-than-zero cutoffs; everything also on
  validated events only.

## Priority 2 (Monte Carlo moments beyond order four)
Not run: new Monte Carlo moment runs with higher moments on 20 + development nets do not fit the CPU budget next to the
truth runs. Stated here so its absence is not a silent omission.

## Predictions
Kill test passes (development 100-101 showed +0.46). Confirmatory primary passes with a difference of about +0.3 to
+0.45; P3A's fraction within 50% at depth 8, 1e-8, about 0.5.
