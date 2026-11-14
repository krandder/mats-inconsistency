# PREREG: the first-order optimal block as an oracle family (lyapunov-dimension addendum, 29 Oct 2026, ~20:40Z)

Written before any PBO run at 256 x 64. Seen before: the 64 x 16 CPU smoke test of the code (net 113 at width 64, a
different net); the nongauss lane's measured PB, PBW, PBA, PBS errors at 256 x 64 (the report's section 6).

## Question
Theorem B of the report: to first order, the best k-dimensional block conditions on the whitened functionals R^+ V_k
(R = (P C P)^1/2, V_k the top-k eigenvectors of S = R M R), equivalently on U = orth(Pi M R V_k). The report predicts,
through its proxy-tail association, that this block reaches nMSE 1e-5 from the L/2 cut with 18-28 directions at
256 x 64, against 33-51 PCs. Approved by the coordinator 20:2xZ.

## Families (lib/ng3.py; harness copied from work/nongauss-family, ng.py / ng2.py byte-identical)
- PBO{k}: block U = orth(Pi M R V_k), rest regressed on the block and carried by independent exact weighted marginals
  (ng2.BlockU), sphere chart. PBW{k}: the same with U = orth(R V_k). PB{k}: the nongauss lane's top-k weighted-PC block.
- PBW and PBO of one (net, cut) share one sensitivity estimate M (2^14 fit paths, one random output projection each).

## Runs (one L4 job, scratch/gpu_job_pbo.sh)
- R-A single cuts: nets 113-116, 256 x 64, cuts 32 and 48, fit 2^20 true paths, 2^21 family samples, truth 2^24 (the
  nongauss lane's truth files, copied). Families EMP, PB16, PB32, PB48, PB64, PBW16, PBW32, PBO8, PBO16, PBO24, PBO32,
  PBO48. Output at L = 64.
- R-B composition: nets 113 and 114, re-projection at every layer 32..63 on fresh true paths, NC = 2^20 and 2^21;
  families MC, EMP, PB32, PBO32, PB64, PBO64. B_F(64) = 2 C_F(2^21) - C_F(2^20) as in the nongauss lane's T3.

## Predictions (PREREG-PBO-predictions.txt has every number; proxy tails x C, beta = 1)
- k*_O (smallest k in {8, 16, 24, 32, 48} with nMSE(PBO_k) <= 1e-5 at L = 64): cut 32: 32 / 24 / 24 / 24 (nets
  113 / 114 / 115 / 116); cut 48: 24 / 16 / 24 / 16. k*_P for PB in {16, 32, 48, 64}: cut 32: 64 / 48 / 48 / 48;
  cut 48: 48 / 32 / 32 / 32.
- PB_k / PBO_k error ratio at k = 16 and 32: cut 32: 1.9-2.9 (net 113) to 2.5-4.7 (others); cut 48: 2.7-6.2.
- PBW_k / PBO_k at k = 16 and 32: 1.6-4.4.
- Composition (forecast): each family's composed error about 20-40x its own single-cut error at cut 32 (the ratio
  measured for PB64 by the nongauss lane); PBO below PB at equal k.

## Tests and rules (per cut; "nets" = 113-116)
- T-A (the count): SUPPORTED if k*_O <= 32 on at least 3 nets and k*_O < k*_P on at least 3 nets; REFUTED if
  k*_O >= 48 or not reached on at least 3 nets, or k*_O >= k*_P on at least 3 nets; otherwise MIXED.
- T-B (against PCs at equal k): geometric mean over nets of nMSE(PB_k) / nMSE(PBO_k), at k = 16 and at k = 32:
  SUPPORTED if both >= 1.5; REFUTED if either <= 1; otherwise MIXED. Values below the floor (nMSE <= 2e-6) are
  replaced by 2e-6 in ratios.
- T-C (against PBW at equal k): the same with PBW, thresholds 1.3 / 1.
- T-D (calibration of the association out of sample): points (net, cut, k) with measured nMSE(PBO_k) > 2e-6:
  SUPPORTED if at least 75% have |log10(measured / predicted)| <= 0.5; REFUTED if fewer than 50%.
- T-E (composition, nets 113 and 114): SUPPORTED if B_PBO32 < B_PB32 / 1.5 and B_PBO64 < B_PB64 / 1.5 on both nets;
  REFUTED if PBO is not below PB at either k on either net; otherwise MIXED. Reported with MC and EMP controls.

## What the outcomes mean
- T-A and T-B supported: the whitened spread x sensitivity block is the better block, and the count it needs is
  about half the PCA count, as Theorem B predicts; the PB ladder's 64 overstates what a well-chosen block needs.
- T-A refuted: the first-order theory picks the wrong directions for the mean error; the association of section 6 does
  not transfer to a family it was not fitted on.
- T-E: whether the better single-cut block also composes better (re-projection every layer).
