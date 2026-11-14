Model: GPT-6 (Codex)

Read-only review, checked 2026-10-30 against source and saved results; no Python or recomputation.

1. **High: causal ordering is unsupported.** Draft:7,19,51–53 calls c a consequence of routing. Earlier strong discrimination does not establish causal precedence; c already discriminates in reverse at 2,000 (results/e57.json:auc.2000, 0.30 [0.14,0.44]). Fix: “Strong discrimination in the final direction appears later for c”; retain downstream causation only as an untested hypothesis.

2. **High: c is a geometric proxy, not measured behavioural saturation.** lngeom.py:25–32 correctly centres the error-free embedding (including bias), replaces o with y, and centres W[:,32]; labelled indices match world.py’s layout. But the even one-parameter approximation requires negligible rho and narrow A (PREREG-B.md:3107–3110). Initial exploratory |rho|=0.35 and A CV=0.27 violate those approximations; e57_run.py:33 saves neither diagnostic. Changes in A alone can move c; attention, value projections and LayerNorm affine parameters also affect behaviour. Fix draft:3,19,52: describe a crossover-scale proxy, disclose assumptions, and reserve “un-capping” for a hypothesis pending diagnostics/behaviour.

3. **High: the claimed behavioural peak is wrong.** Draft:5,18 places the largest departure at 2,000–3,000. Update 26 in cm-interp-phaseB-2026-10-28.md reports onset at 2,500/2,750 but peaks at 4,500/9,000, in only two nets. Fix: distinguish onset from peak; E55’s later anticorrelation does not locate the peak in E57’s cohort.

4. **Medium: universal monotonicity fails.** results/e57_c.json:e37_ctrl_s8 falls from 1.266 at 3,000 to 1.224 at 5,000. Draft:6,52 overgeneralizes group medians, whose membership changes from 149 to 215. Fix: report sampled group medians; require paired trajectories for individual growth-rate claims.

5. **Medium: inference exceeds the tests.** P222–P224 correctly pass registered point-estimate thresholds; P222’s interval extends outside [0.4,0.6], and P223’s crosses 0.75. Neither establishes “no information.” Eleven clusters, shared hybrid donors across recipient clusters, and disclosed related-lineage knowledge limit precision and independence; dropping E37 is no fresh replication. Fix: qualify inference accordingly.

6. **Low: numeric overstatement.** e57.json:auc_without_e37.10000 minus auc.10000 is approximately 0.0123, exceeding draft:17’s 0.01. Fix: “under 0.013”; initial B median rounds to 2.06, not 2.07.

**Survive:** source-level checkpoint/token selection, E47 filtering/types/duplicates/clustering/bootstrap, reversed AUC direction, registered scores, and table/JSON agreement. **Reword:** late strong marker and descriptive two-phase median trajectory. **Fail:** established causal origin, universal monotonic rise, peak-at-acquisition, and information-free initialisation.
