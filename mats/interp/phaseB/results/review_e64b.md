Model: GPT-6 (Codex)

Read-only review, checked 2026-10-31; no Python or recomputation. Source inspection supports the registered split formula, code-free content normaliser, correct 1/sqrt(2) code scaling, unchanged parameter initialisation/episode construction, and evaluator builders. Step-zero equality is supported by construction and disclosed plumbing, not independently rerun. Registered verdicts and E64b table entries match results/e64b.json.

1. **High — cap predictions silently clamp outside the table.** `e64b_post.py:41` uses `np.interp`; `results/e55_cbtable.json:grid` starts at 0.8. Eight split and two splitnc nets have c below 0.8 and receive exactly the 0.8 predictions (e.g. `results/e64b_post.json:e64_split_s2`). Thus “at those c” and “corrects about half” are unsupported numerically. Fix: flag out-of-range values; extend the constrained-Bayes table before quoting predictions. Even corrected uniform-pool predictions do not identify learned weighting as the compensating mechanism.

2. **High — causal isolation and M2 overclaim.** `update78_draft.md:52,56–57` acknowledges that split also fixes code length and changes cap scale, yet identifies shared normalisation as the main cost and treats failed predictions as locating other causes. Codes also remain in the residual stream (`pfn_e64b.py:57`). Fix: report a successful bundled intervention consistent with the mechanism; reserve attribution for matched ablations. Failed predictions do not locate their causes. Apply this qualification to the plain-words addition too.

3. **High — interaction inference exceeds the design.** `update78_draft.md:53`: arithmetic is correct (−0.0030), but there is no membership-given/sign-free fourth cell, common acquired-seed analysis, or uncertainty for the contrast. The KL floor can invalidate additive extrapolation without identifying a shared-normaliser interaction. Fix: call this suggestive nonadditivity; measure the fourth cell and paired interaction before asserting “only when.”

4. **Medium — equivalence is not established.** `results/e64b.json:arms.split.vs_own_kl` gives 1.08 [0.91, 1.25]; P253 tests against ctrl, not equivalence to own. Fix “equals/as close” to “similar observed mean KL,” including the plain-words headline.

5. **Medium — head claims exceed diagnostics.** `e64b_heads.py:99` measures one norm-logit component, not total attention-weight change. `results/e64b_tables.md`, seed 3, is splitnc A with one pooling head, contradicting “forces the A nets” (`draft:55`). Fix: “positive norm contribution” and “10/11 A nets”; summary-information mediation remains hypothetical.

Survive: P251a, P253, P255 and descriptive allocations. Reword: mechanism, equivalence, head usefulness, M2. Fail: P251b, P252, joint P254 (splitnc individually passes), and the current quantitative cap explanation.
