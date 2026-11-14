Model: GPT-6

Read-only review, checked 2026-10-31 against source, registration/hash log, metadata, head JSON and saved aggregates; no Python or numerical rerun. “Draft” means update_e64_draft.md; “report” means cm-interp-phaseB-2026-10-28.md.

1. **High — synthesis turns approximations into mechanisms.** Report:954 calls the norm “even, capped” and says learned pools build an even statistic. The denominator contains `2*r*A*U*rho` (e64_heads.py:80–89); the normalised feature saturates, not the norm. E66 establishes sensitivity, while its variance-statistic explanation remains a candidate (report:7047–7050). **Fix:** restrict this decomposition to block-1 raw signed inputs; say “nearly even normaliser, saturating feature” and “can construct an approximately even statistic”; uniform-pool departures approximately match constrained Bayes.

2. **High — nocons does not eliminate B or force pooling.** Draft:4,50 overstate the mask and outcome. pfn_e64.py:54–58 bars only other summaries; self and other-source labelled keys remain. `results/e64_heads/e64_nocons_s3.json` has self_h≈0.97, olab_h≈0.69, n_pool=0. **Fix:** say “11/12 become A”; explain this surviving B allocation. “Consensus” here includes self-attention, not necessarily cross-summary aggregation. E69/E70 support the allocation account but still confound head count and width.

3. **High — confounding is not causal exoneration.** Draft:5,20,51 says these arms measure weak codes, not coupling, whereas PREREG-B.md:3704–3706 explicitly describes both changes after observing seed-0 plateaus. E64b also changes code length and cap scale (report:7213). **Fix:** say neither experiment isolates coupling; preserve failed predictions, distinguish pre-final-analysis caveat from wholly unseen outcomes, and call delayed routing consistent with the weakness hypothesis.

4. **Medium — intervals and “halving” disagree with saved evidence.** `results/e64.json`: codeln.vs_ctrl_kl CI=[1.56,3.44], codeln.vs_ctrl_rom=[1.05,1.64], both.vs_ctrl_kl=[1.04,3.50], both.vs_own_kl=[4.08,9.06]. Draft:15–17,37 differ. **Fix:** copy saved intervals; label 0.51 as the 11-seed paired ratio. Overall means give ≈0.68; medians ≈0.64, not half.

5. **Medium — blanket claims contradict measurements.** Draft:20’s “worse ... every respect” conflicts with improved sign shares (0.054/0.088 versus 0.200) and P243a. Draft:50’s “does not touch” code dependence lacks an equivalence test. **Fix:** name worsened metrics; say no demonstrated code-share reduction.

**Survive:** static architecture/evaluator construction and registered scoring; tables match JSON; P242, P243a and nocons’s P245 component hold. **Reword:** synthesis, causal attribution and plain-language comparisons above. **Fail:** joint P243–P246; B-elimination, universal worsening and exact-evenness claims.
