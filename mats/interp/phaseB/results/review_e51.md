Model: GPT-6 (Codex)

Checked 2026-10-30 by static file inspection and SHA-256 comparison; no Python or numerical rerun.

1. **High: mean correlations do not identify the source of KL sign cost.** `e51_summ.py:42-49` measures cross-net agreement only in predictive means; shared sign-dependent variance or shape errors remain possible. `update60_draft.md:6,19,62,65` consequently overclaims “not ... shared bias,” explains earlier prediction failures causally, and bounds architectural predictability. P204 actually fails. **Fix:** restrict the finding to predictive-mean shifts in these nets; call the ensemble explanation compatible with the results, not established. Architecture could also predict the distribution of seed-specific errors.

2. **High: correlation intervals lack a valid bootstrap calibration.** `e51_summ.py:27,53-58` converts each resample to a set, discarding multiplicities. This is random subsampling of roughly 63% of nets, with unadjusted percentile intervals, not an ordinary net bootstrap. The synthetic check validates point correlations, not interval coverage. **Fix:** use a calibrated net-level jackknife/U-statistic bootstrap; retain the current intervals only as descriptive sensitivity ranges. The c_sign bootstrap retains multiplicities and is reasonable conditional on these episodes.

3. **Medium: the post hoc split is treated as an observed decomposition.** `e51_post.py:1-5,16-18` fits two quantities with two parameters without testing the 1/M approximation or uncertainty. `update60_draft.md:3-4,18,64-65` turns this into claims about each network and no D=1 format difference. JSON `d1_prod.specific` differs from ctrl, with no equivalence test. **Fix:** label all splits conditional estimates; say “mean KL,” not “each network's distance”; remove the equality claim pending uncertainty and additional ensemble sizes.

4. **Medium: sys_h includes finite-net noise.** `e51_summ.py:47` includes diagonal variances; even independent equal-variance shifts give 1/N (explicit in `e51_plumb.py`). **Fix:** call it the observed ensemble-mean variance ratio, or estimate shared covariance excluding diagonals.

5. **Medium: equal member counts establish only a benchmark.** `e51_run.py:41-43,77-78,104-107` compares five overlapping seed subsets with correlated sign variants, using different baselines and training costs. **Fix:** retain the measured comparison; replace “any averaging” (`draft:6`) with this specific geometric-averaging result.

**Survive:** registered families, nets, grids, estimators, point-statistic formulas and P204=false/P205=true match code/JSON; hashes match and timestamps support the disclosed plumbing order. **Reword:** mostly net-specific mean shifts, ensemble analogy, model-based split. **Fail:** causal explanation, exclusion of shared density bias, and verified 95% correlation coverage.
