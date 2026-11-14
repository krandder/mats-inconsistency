# PREREG-TAILS-confirm — confirmatory test of the saddlepoint readout of ARC's description (part 2b)

Written 2026-10-29 09:30Z, before any run on the nets below. Background: in part 2 (PREREG-TAILS.md) the candidate U2,
ARC's own factorized K = 3 description (simple kind, exact metric, run_lpe settings, our weights and noise) read out
by a Lugannani-Rice saddlepoint on its per-output moments E[z^p], p <= 4, instead of ARC's heaviside expansion, failed
its kill test at depth 8 on development nets 102-105 (+0.12 more events within 50% than ARC at 1e-8; bar +0.20) after
passing at depth 4 (+0.47). On the 20 evaluation nets, scored post hoc, the difference at 1e-8 was +0.34 [0.25, 0.44]
(depth 4) and +0.23 [0.15, 0.31] (depth 8) at sigma = 0.001. This file tests that post-hoc observation on fresh nets.
Nothing here changes any part 2 verdict.

## Object, events, truth (exactly as PREREG-TAILS sections 1-2, except where stated)
- Nets: seeds 24-43 (20 nets never used in parts 1 or 2; their noise scales r_l come from the same seeded pilot
  definition, computed at run time). sigma = 0.001 only. Depths L = 4 and 8 only.
- Outputs 0-7; thresholds t_(i,k), k = 4, 6, 8, 10, 12, set by the same adaptive splitting run (thresholds.py).
- Truth: truth.py (frozen-tuning splitting, R = 16 runs of N = 2000 at L = 4 and 8, cross-check (c) on outputs 0-1) and
  mcx.py (2^23 Monte Carlo paths, cross-check (b) at 1e-4). Same provisional rule.

## Methods (unchanged code: kprop_tail.kprop_all, deduce.saddle with the exact-root search and order-3 / Gaussian
fallbacks, deduce.radial)
- C-K3f: ARC's tail method (heaviside readout of the factorized K = 3 description).
- U2: the saddlepoint (order 4) on the same description's moments. U2r: the chi-mixture variant (secondary).

## Test
- Statistic: per net, the fraction of its 8 events within 50% (|p_hat / p - 1| <= 0.5) for U2 minus that for C-K3f, at
  each (L, k); mean over the 20 nets with a bootstrap 95% CI (10,000 resamples of nets).
- Primary (confirms the post-hoc observation): at k = 8 (1e-8), at both L = 4 and L = 8, the mean difference is at least
  +0.20 (the original kill bar) and its CI lower end is above 0.
- Secondary: the same at k = 6, 10, 12; the smallest p at which each method beats zero (median relative error <= 0.9,
  CI upper end below 1); everything also on validated events only.
- Costs (stated, not part of the test): per threshold vector U2 costs about 2.3x (L = 4) and 1.7x (L = 8) of C-K3f; with
  five thresholds sharing the propagation, about 0.85x.

## Predictions
- Primary passes at L = 4 (difference about +0.3) and is borderline at L = 8 (about +0.2, CI above 0).
- U2 beats zero to 1e-12 at L = 4 and to 1e-10 at L = 8; C-K3f to 1e-8 and 1e-4.

## Compute
One on-demand L4 VM (job loop, logged in ../GPU-LOG.md, deleted with an empty instance list verified), about 0.5 h;
kprop on this host's CPU (40 runs).
