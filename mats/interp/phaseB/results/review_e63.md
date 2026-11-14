GPT-6 (Codex)

Read-only review, checked 2026-10-31 against the supplied code, registration and JSON; no Python or model execution.

1. **High: “exact” omits epsilon.** `e63_run.py:134-143` uses n²=|a+ru|², but `pfn2.py:38` constructs default LayerNorm: the exact denominator is sqrt(|a+ru|²+d·eps). Query bias is included; LayerNorm offset and key bias correctly contribute a key-independent constant, so the scaling zero point is sound. Fix the denominator and reference normalization, or label these registered quantities epsilon-free approximations and quantify the discrepancy before asserting exactness.

2. **High: coefficients are not complete weight effects or sign symmetry.** `e63_run.py:143,203` computes only the alpha-term change. With nonzero rho, n(r) is not even; K_odd is the registered signed-numerator coefficient, not the full logit derivative. Opposite mean coefficients do not establish cancellation through unequal value/output maps. Fix draft lines 3, 7, 96-99: say “norm-term contribution,” retain the three-cohort departure association and sign-pair mechanism as hypotheses, and replace “30 times less weight” with an unnormalised-score factor. Neither routing necessity nor departure causation follows.

3. **Medium: B allocation is overstated.** `results/e63_post.json:{ctrl,fix,ext}.type_by_npool` contains 1+5+9 B nets with zero pooling heads; fix also has three mixed non-pooling heads. The 137/149 counts heads, not networks. Fix: “A has two; B has fewer than two,” explicitly post hoc; C measures mass on summaries, not equal averaging or functional consensus.

4. **Medium: geometry remainder is unidentified.** Draft line 98 subtracts unlike interventions: orthonormalisation and code-redraw averaging. A 14% versus 28% gain does not show the remaining dependence excludes norms/overlaps. Delete that remainder claim; retain the evaluation-time improvement in the 11 ctrl nets.

5. **Medium: training timing overgeneralises medians.** `results/e63/traj.json`: s1 already has two pooling heads at 5k; s6 still has one at 10k; s8 has zero at 20k. Fix plain-words timing to observed per-run intervals and qualify “one pooling head by 5k.”

6. **Low: descriptive labels/counts.** `results/e63.json:cohorts` gives 200 leakage correlations, not 196. Tables’ “KL” is stored NLL gap (`e30_eval.py:66`); A-CV aggregates are medians. Correct labels/counts.

Survive: registered cohort/filter/exclusion/cluster logic, Bayes-invariant Loewdin transformations, attention definitions, P237 failure and P238-P241 scores under registered formulas. Reword mechanisms and summaries above. Fail: exactness as written, universal one-pooling-head B account, and inferred geometry remainder.
