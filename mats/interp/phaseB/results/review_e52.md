Model: GPT-6 (Codex)

Checked 2026-10-30, read-only; no Python or numerical rerun. Source hashes match the registration. Metadata, selection code, logs, grid and estimator support the registered design. JSON and table agree; K1, K9, c_seed, rho_g and the split formulas are implemented correctly. P206–P207 correctly fail their registered point-estimate bars. NPZ numerical contents were not independently recomputed.

1. **High: depth does not increase absolute variation.** `results/e52.json:{w32,d3}.{ctrl,mag}.K_specific` falls from 0.00877 to 0.00807 (ctrl), and 0.00343 to 0.00313 (mag). Draft lines 3, 6, 16–17 and 47 confuse a larger removable fraction/lower correlation with greater variation. Fix: “Depth reduces the estimated shared component more strongly; the varying component falls slightly and occupies a larger share.”

2. **High: rejecting two thresholds does not establish a capacity-only explanation or exclude a floor.** Draft lines 17, 46–48; `e51_post.py:1–5`. Two ensemble sizes exactly fit an untested second-order model; K9 retains finite-ensemble variation. Ten–twelve nets, overlapping subsets and one training recipe cannot separate architectural constraints from optimization or rule out a smaller fixed component. Fix: report cohort estimates conditional on the model; describe architecture-level scaling as a hypothesis. Subset ranges are not uncertainty intervals for the population claims.

3. **Medium: equal shrinkage and monotonic improvement are overstated across formats.** Draft lines 3–4, 17, 48; JSON `{w32,w128}.mag`: shared retains about 57%, specific 37%, and c_seed drops 0.084. Width-64 mag worsens. Fix: restrict approximate proportional shrinkage to ctrl and explicitly state mag’s exception.

4. **Medium: distinguish denominators and E51’s rho_g.** `e52_summ.py:34–38` uses K1_subsets for the split, whereas the table displays all-net K1 (width-32 ctrl: 0.013848 versus 0.013347). E51 rho_g uses sign-averaged means; E52 uses base means. Fix: label both distinctions explicitly; use K1_subsets in the displayed formula.

5. **Low: provenance timestamp.** Draft line 22 says 13:06:55Z; `prereg_hashes.txt` records 13:06:53Z. Correct it; retain the disclosed pre-hash plumbing decision exposure.

Survive: registered failures, ctrl’s descriptive width trend, lower depth K9 and mean-error correlation. Reword: shared/varying and architecture interpretations, including plain words. Fail: increased absolute depth variation, universal equal shrinkage, exclusion of a fixed floor.
