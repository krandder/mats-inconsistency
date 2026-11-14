p = '/home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md'
s = open(p).read()
rep = {
# (2) 3.5 kernel test relabel
"""    256) over RB-MC at 64, 19, 10, 15, 8.4 and 4.1. The fraction drifts toward the mean-field 1/2 (noisily), so the
    cap at ARC's setting is already near its limiting value of about 4. b_2 is bounded via the degree-2 harmonic
    energy (b_2 <= n a_2/(n-1) [A]), which makes the even-tail estimate a lower estimate and the cap conservative.""":
"""    256) over RB-MC at 64, 19, 10, 15, 8.4 and 4.1. These are plug-in estimates, not bounds: they rest on 3 nets,
    sampled outputs and noisy integrals, and at ARC's 256 x 32 the estimated degree-2 energy a_2 came out negative
    (-0.0023), so the step b_2 <= n a_2/(n-1) [A] cannot be applied to it as a bound. The fraction drifts toward the
    mean-field 1/2, so the plug-in cap near ARC's setting is near the limiting value of about 4.""",
# (5) 3.7 variance wording and measured-gain labels
"""  Input directions merge with depth (the cosine deficit 1 - kappa^{(l)}(0) decays like 9 pi^2 / (2 l^2)), while the
  per-net variance V_sph stays of order 1 at L = alpha n because of amplitude variation set in the first layers. Astra""":
"""  Input directions merge with depth (the cosine deficit 1 - kappa^{(l)}(0) decays like 9 pi^2 / (2 l^2)), while the
  ensemble-average angular variance V_sph stays of order 1 at L = alpha n because of amplitude variation set in the
  first layers. Astra""",
"""- **MEASURED** (post hoc, `scratch/mlmc_check.py`, `logs/mlmc_check.log`; 4 fresh nets per cell, seeds 9100-9103,""":
"""- **Estimated ideal gains, not end-to-end speedups** (post hoc, `scratch/mlmc_check.py`, `logs/mlmc_check.log`, run
  12:40-12:50Z; 4 fresh nets per cell, seeds 9100-9103,""",
"""  variance x cost relative to RB-MC, setup excluded, so below 1 is a gain). Two-level version with the best
  dyadic prefix depth:""":
"""  variance x cost relative to RB-MC, so below 1 is a gain). Two-level version; the prefix depth is chosen as the best
  dyadic one on the same 4,096 directions that estimate the variances, so the figures are optimistic by that
  selection, and they exclude the setup pass, the pilot, integer allocation and lower-order work:""",
"""- **Consequences.** C0 as I stated it is false, asymptotically. With the symmetric derandomisation of 3.1 the
  estimator becomes deterministic, so an unrestricted deterministic estimator does beat plain sampling by a factor
  that grows at depth proportional to width; it does so by better sampling, not by deduction, and only where the net
  is several times deeper than wide at the sizes we can test (no gain at ARC's 256 x 32, L = n/8). The baseline for
  ARC's question at depth should be this structured sampler, and matching it without sampling is the open question.
  At width 8 and depth 256, the narrow-deep-m1 setting, the two-level sampler is about 15x better than radius-exact
  MC (noiseless; median over 12 live nets), comparable with the 3-27x that lane measured for RQMC over plain MC, and
  the two can be combined. That lane's sampled-prefix arm beat RQMC 4.6-6x at small budgets but carried a bias, which
  the multilevel corrections remove. All of this is for sigma = 0.""":
"""- **Consequences.** C0 as I stated it is false, asymptotically, under astra's layerwise cost hypotheses and against
  radius-exact MC with the best allowed batching. With the symmetric derandomisation of 3.1 (idealised arithmetic)
  the estimator becomes deterministic, so an unrestricted deterministic estimator does beat plain sampling by a factor
  that grows at depth proportional to width; it does so by better sampling, not by deduction. At the sizes tested
  the estimated gain is about break-even at L <= n/4 (0.91-1.29, including ARC's 256 x 32 at 1.01), up to 1.5x at
  L = n, and larger at L >= 4n. Nothing here shows it beats RQMC, and the finite-precision version of the
  derandomisation is not analysed. The baseline for ARC's question at depth should be this structured sampler, and
  matching it without sampling is the open question. At width 8 and depth 256, the narrow-deep-m1 setting, the
  estimated two-level gain over radius-exact MC is about 15x (noiseless; median over the 12 of 32 seeds whose nets had
  positive empirical angular variance). That lane's sampled-prefix arm beat RQMC 4.6-6x at small budgets but carried a
  bias, which multilevel corrections remove; the two lanes' baselines and nets differ, so no ranking between this
  sampler and RQMC follows. All of this is for sigma = 0.""",
# (7) partition theorem norm
"""- **Partition certificates** [A, round 3]. Bounded aspect ratio alone gives no lower bound (a hinge can be aligned
  with the cells). Under explicit nonalignment, isolation and capture hypotheses, a certificate of width eps sqrt(V)
  needs K >~ (n^3 / eps)^{(n-2)/2} cells, using the exact expected kink area Theta(n^{3/2} L) of the regular kink set. Not
  checked numerically; the hypotheses are unproved for coarse cells.""":
"""- **Partition certificates** [A, round 3]. Bounded aspect ratio alone gives no lower bound (a hinge can be aligned
  with the cells). Under explicit nonalignment, isolation and capture hypotheses, holding uniformly in dimension, a
  certificate of expected summed-coordinate width eps sqrt(V) needs K >~ (n^3 / eps)^{(n-2)/2} cells, using the exact
  expected kink area Theta(n^{3/2} L) of the regular kink set; for Euclidean width (the metric of our certificates)
  n^3 becomes n^{5/2}. Not checked numerically; the hypotheses are unproved for coarse cells.""",
# (11) spin glass wording
"""  computation found one at L = 9, q = 0.5, inside the interval (qhat = 0.4926). So the first failure is at depth 9
  in both forms. My registered P12 (concave at every depth) is refuted.""":
"""  computation found one at L = 9, q = 0.5, inside the interval (qhat = 0.4926). So the condition fails at depth 9 in
  both forms and at every depth tested above it; depths 3, 5 and 7 were not checked. My registered P12 (concave at
  every depth) is refuted.""",
"""- The mean is the infinite-temperature end (3.4); tails at tiny p are the low-temperature end.""":
"""- The mean is the infinite-temperature end (3.4). Tail probabilities at large thresholds are governed by large
  deviations, i.e. by log Z at large beta, which is where the low-temperature results live.""",
"""   infinite-width kernel fails the full-RSB concavity test from depth 9, which suggests, for a Gaussian surrogate""":
"""   infinite-width kernel fails the full-RSB concavity test at depth 9 and above (depths 3, 5, 7 unchecked), which
   suggests, for a Gaussian surrogate""",
}
for a, b in rep.items():
    assert a in s, a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s)
print('ok')
