Model: GPT-6 (Codex)

Read-only review, checked 2026-10-31 16:23 UTC; source inspection and jq checks, no Python.

1. **High: nocons is loaded as standard PFN.** `e63_run.py:35,90` dispatches through `pfn_own.build_any`, whose fallback chain ends at `pfn_pre.build_any`, constructing standard PFN; neither E72 runner installs `pfn_e64`. Thus the claimed nocons mask is absent in Part 1 and detq. Near-equal rom does not establish mask correctness (`results/e72_detq/nocons_s0.json`: base differs from registered). **Fix:** install the correct variant builder, verify masks, rerun nocons and affected aggregates before accepting cross-architecture claims.

2. **High: “ideal reasoner” and “information loss” exceed this estimator.** `e72_run.py:79,101-110,141-157` renormalises own-key attention, estimates marginal p(v|T), multiplies source factors, and adds the query marginal. T itself depends on query readings; shared labels/codes also prevent assuming conditional independence. This is not established as Bayes conditional on all observed statistics/readings. PCA and classifier error add further approximation; scalar calibration does not bound M3 error. **Fix:** call M3 a factorised classifier-based benchmark; remove optimality/information-loss claims from both versions pending conditional-reader and calibration checks.

3. **High: detq does not identify the proposed three mechanisms.** `e72_detq.py:38-43,54-62` detaches summary queries in every head, including attention to non-own tokens; keys, values, residuals and other routes remain live. M3 contains only selected, renormalised own pools. Consequently detq−M3 includes information/route differences, not solely reader excess. **Fix:** retain the arithmetic identity and unchanged-forward claim, but describe operational differences; isolate matching information and routes before calling this “like-for-like.”

4. **Medium: route absence/unpatched-route claims are too strong.** `e72_patch.py:74-85,118-119` measures single-route interventions against detach controls, not additive mediation. Wide-A detach_iii medians reach +0.324; fixed-B contributions cannot establish where the remainder travels. Sign-free B has no selected pooling heads. **Fix:** report intervention effects and unexplained cases, replacing “carries none/whole” and “most … unpatched.”

5. **Medium: mixed denominators obscure decomposition.** Draft line 43 uses eight fix-B nets for net=0.191 but three for decomposition, whose base=0.144 (`results/e72_tables.md`, addendum). **Fix:** use matched subsets throughout.

Survive: stored numerical verdicts match `results/e72.json`; thresholds, exclusions and positive-rom B filtering agree with registered scoring (all current B roms are positive). Dev-tuned thresholds and post-Part-2 addendum timing are disclosed. Reword mechanistic/plain-language claims; nocons-dependent results need rerunning. P283/P284/P287 fail; other stored passes remain provisional where affected above.
