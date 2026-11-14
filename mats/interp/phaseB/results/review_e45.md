Model: GPT-6 (Codex)

Checked 2026-10-30 07:45Z; read-only inspection, SHA-256 and lightweight jq checks; no Python or bootstrap rerun.

1. **High: Sonnet’s invariance evidence is fragile.** `e45_summ.py:39-49` averages available samples but estimates retest variance only from doubles. Under invariance, singleton versions have greater shift variance, so the baseline is not necessarily 1. `e45_calls.jsonl`, S_sonnet46/pair=49: singleton answers 3.49 versus 0.34 contribute about 65% of squared shifts, with neither version contributing retest variance. Numerous cap-triggered retries also condition on compliance. Fix: retain registered results, add complete-four-answer and leave-one-pair-out sensitivities; qualify “beyond noise” pending them.

2. **High: causal conclusions exceed the experiment.** Draft:53-54 identifies learned text priors and says reasoning improves invariance. Update 31 used different model generations and big-miss/spread manipulations, not E45’s negations. E28 rejected a particular proportional correction, not every bias-correction mechanism. Fix: describe compatibility with a bias-correction heuristic; remove the claimed causal separation and reasoning effect. Require matched-model controls to establish them.

3. **Medium: “full correction” is an operational scale.** `e18_data.py:32-37`, `qweights.py:22-28` define weights as local derivatives, not fixed aggregation coefficients. X’s positive sign is correct for subtracting mean error, but c=1 describes the registered linearized/fixed-sensitivity correction, not general finite Bayesian debiasing. Fix: define this explicitly; soften “about the full amount” (draft:5,52).

4. **Medium: nulls and pooled results become universal claims.** Draft:4,10,14,18,52 says GPT does not change and correction “vanishes.” `e45.json:pooled.S` spans −0.31 to 0.32; stated Haiku is positive. Unstated pooling masks strong heterogeneity; P188 passes its point-estimate rule despite its interval extending below 0.5. Fix: “not detected,” identify the Claude pool, and call per-arm intervals descriptive/unadjusted. “No consistent direction” only means no detected proportional relationship.

5. **Low: presentation errors.** Draft:3 says twice for every model; Qwen has one sample. “Without reasoning” is requested, not verified for GPT; numeric generation is autoregressive. Draft:17’s stated MAD range should be 0.12–0.22. Fix these and limit “correct answer unchanged” to the stipulated zero-mean benchmark.

Survive: all 60 selected-source negations and stored Bayes equalities; code/items hashes; E18b settings, leading-number parser, retries and last-ok filtering; registered formulas, JSON/table agreement, P187 failure/P188 success; logged Claude spend $4.139875 including failures. Need rewording: population-wide invariance and full-correction claims. Fail: identified mechanisms and a causal reasoning benefit.
