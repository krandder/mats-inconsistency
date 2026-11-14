Model: GPT-6 (gpt-6-astra)

Checked 2026-11-01 by reading code, JSON, hashes and logs; no Python or model execution.

1. **High: convergence and attribution overclaims.** `update87_draft.md:7,20,84-87` turns one fresh reader per source and one continuation schedule into “any reader,” “attainable distance,” and “close to its optimum.” Replication under one optimiser/architecture cannot distinguish representation limits from shared optimisation limits; the embedding is frozen too. P310's failure is not evidence of convergence. **Fix:** say this recipe approximately reproduces the measured gaps and A departures; describe block-1 limitation as a hypothesis, including in the plain-words addition.

2. **High: checkpoint sensitivity undermines the headline.** `results/e76x_post.json:medians.A_fresh_best.ratio` is 1.233 versus 0.968 for final weights; registered s0/s2 ratios are 1.381/1.336. `per_source.1.continue_best.ratio` is 1 because selection retains the source, whereas `continue_final.ratio` is 0.825. **Fix:** foreground registered A failures and distinguish selected-checkpoint gains from gains after 90,000 steps. Keep final weights as sensitivity analysis. Call validation noise a possible explanation, not an established cause.

3. **Medium: “without a KL cost” exceeds the measurement.** `e76_eval.py:29-30` computes sampled test-NLL differences, without uncertainty; `update87_draft.md:86` calls these unchanged KL and attributes about an eighth of B's departure to training path. **Fix:** report similar measured test-NLL gaps and a descriptive B shift, without exact costlessness or a general attributable fraction. Preserve E72's operational decomposition: detq includes other live routes, not a pure reader component.

4. **Low: provenance needs explicit evidence limits.** `update87_draft.md:19,25-26`: queue logs support completed trainings and later evaluation, but the exact 503 interval comes from the handoff; no saved 40-file frozen-check output was located. **Fix:** attribute the outage account and link the frozen-check output. Current scoring matches registration; the bool conversion preserves its tests, but hashes alone cannot prove the entire historical edit.

Survive: training implementation, E72 measure reuse/source agreement, tables, P307/P309 passes and P308/P310 failures (8/7/8/5), and descriptive A/B contrast. Reword: checkpoint, outage and cost claims. Fail: universal-reader, attained-optimum and established block-1-limit conclusions.
