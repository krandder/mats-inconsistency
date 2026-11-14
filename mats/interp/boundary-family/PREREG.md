# PREREG (boundary-family lane), 29 October 2026, written 15:47Z

Scope: the early kill test K1' of DESIGN.md v2 (section 6) for Candidate A (cell families that follow the collapse
modes, the question's relevance axes, or the gate hyperplanes), on width-256 ARC development nets at depth 32,
sigma = 0.001. Exploration so far used dev net 100 only (diag4, diag5, diag6; seen before this file for G and PCA1 at
cut 24, not for PCA3, AS3 or GATE6). This test uses dev nets 101 and 102, never used by this lane.

## Test K1' (representation at a cut)
- Cut m = 24. Fit set: 2^18 Monte Carlo paths of z_24 per net (seed 70000 + net), as in `scratch/diag5.py`.
- Families (each cell a Gaussian with the fitted within-cell mean and covariance, untruncated):
  G (one Gaussian); PCA3 (4 x 4 x 4 quantile bins of the top-3 PCA modes of z_24); AS3 (the same along the top-3
  eigenvectors of E[g g^T], g = d(r . a_32)/d z_24, r ~ N(0, I), 2^13 paths); GATE6 (sign patterns of the 6 most
  relevant gates of layer 24, relevance = gate entropy x sqrt(E[g_i^2]); patterns with < 600 fit paths pooled).
  (GATET6, the truncated variant, is excluded: its construction truncates an already-truncated fit; found 15:47Z.)
- Representation error: 2^20 paths drawn from the fitted mixture, propagated through the exact layers 25-32 with noise;
  nMSE of E[a_32] against the independent ARC truth (2^25 paths), minus the truth's 1/N.
- Kill rule: Candidate A (these families at <= 64 cells) is dead at 256 x 32 unless, on BOTH nets, the best family's
  representation error is <= 1/10 of G's AND <= 2e-5. (A family cannot be propagated more accurately than it
  represents the state; these errors are of the family, not of any push rule.)
- Also recorded (no rule): the push error of each family (per-cell exact Gaussian closure, layers 25-32), and the
  axes statistics (principal cosines PCA3 vs AS3, variance in AS3, gradient energy in PCA3).

## Prediction (written before any result on nets 101-102)
G about 2-3e-4; the best family 1.2-2e-4 (PCA3); AS3 no better than G; GATE6 within 30% of G. Kill expected.

## Descriptive measurement D (gate compression, no rule)
On the same nets at layers 16, 24, 31: D_raw = gates with entropy > 0.1 bit; D_eff by co-switching (single linkage on
|phi| > c) and by downstream effect (cosine of the finite-difference effect vectors on a_32 > c), c = 0.8, 0.9, 0.95;
the median |phi| among near-boundary pairs; the fraction of near-boundary gates whose sign the top-3 PCA latent predicts
in >= 90% of paths (sign of mu_i + U_i . y). Compared with the gate-entropy census when it lands.

## Amendment 1 (written 16:05Z, before any run of scratch/diag8.py): test K-F, the factored discrete part
Asked by main (relaying the gate-entropy census: no merging at width >= 128, about 2^25 gate classes at layer 32, so the
discrete part must be factored). Question: does a factored model of z_m reproduce the joint gate statistics that matter
for the next layer's mean, i.e. pairwise and triple co-switching along W (per-neuron variance and third cumulant of
z_{m+1}), and the next layer's mean itself?
- Nets 101 and 102 (width 256), layers m = 16, 24, 31. Two independent Monte Carlo sets of 2^17 paths of z_m (fit set A,
  seeds 100000 + net; reference set B, seeds 110000 + net). Script `scratch/diag8.py` as written at 16:05Z.
- Factored models (fitted on A): F3 and F8 (exact top-k PCA latent y, residual coordinates independent Gaussian given y,
  i.e. gates conditionally independent given a 3- or 8-dim latent); F3C (same with per-cell residual means and
  variances on 4^3 cells of y). Comparators: G (full Gaussian, all pairwise structure, nothing beyond), IND (product of
  exact marginals, no latent), C3 (unfactored: full Gaussian per cell on the same 64 cells), and A itself (sampling
  floor).
- Statistics after one exact push: var_err (median |Var ratio - 1| of z_{m+1}), k3_err (median |skew difference|),
  mean_err (nMSE of E[a_{m+1}]).
- Kill rule: the factored discrete part (conditional independence given <= 8 latent dims) is dead for the mean at
  256 x 32 unless at least one of F3, F8, F3C has, at m = 24 AND m = 31 on BOTH nets, mean_err <= 1/3 of G's and
  var_err <= 0.02.
- Prediction: all three fail; the diagonal residual misses the W-projected off-diagonal covariance (random-matrix
  spectrum of W D W^T), so var_err is 0.05-0.2 and mean_err exceeds G's.
