Model: GPT-6

Checked at 2026-10-30T23:09Z, read-only; no Python. Registered hashes match. All 20 initial checkpoints are byte-identical to E38 controls; metadata and train_sw.py:96 support matched schedules, seeds and episodes. E61 NPZ KL/rom scalars match JSON; displayed numbers match JSON rounding. Evaluator dimensions, A/U geometry and in-range interpolation are sound by inspection, not recomputation.

1. **High — depth-3 uniform bottleneck is not enforced.** pfn_uni.py:73 masks only the query; :76 fixes summary rows only in block 1. Block-1 labelled states can reach block-2 summaries through learned attention, then the block-3 query, bypassing uniform pooling. This matches the registered code but defeats the structural interpretation. **Fix:** disclose this route; describe depth-3 agreement as empirical, and test blocking the route before claiming architectural necessity.

2. **High — invariance and causation exceed the evidence.** update69_draft.md:5, :17, :44–45 turn two additional configurations into “does not shrink” and “fixed by” c. Means actually fall from 0.177 to 0.165/0.159; the base used CPU and a different cohort. The pooled vector retains signed information; e53_cb.py:35 uses a simulated, smoothed even-only reader. Given membership also changes the architecture. **Fix:** say “similar departures, approximately predicted at measured c under this recipe”; remove extrapolation, isolated-weighting causation and the unmeasured “rest of KL” decomposition.

3. **Medium — standard depth comparison is wrong.** Draft :17/:45 says c grows with size. Context Update 67:6397–6400 gives ctrl depth-3 c=3.63 versus base 3.79, despite smaller departure. **Fix:** restrict c-growth to width; retain association, not causal explanation.

4. **Medium — distinguish scoring from certainty.** results/e61.json:predictions correctly passes P232/P233, but both ratio intervals extend above 0.6 (0.659/0.604). P233 bounds arm-mean differences, not individual nets. **Fix:** label point-estimate passes and mean agreement throughout, including item 50.

5. **Low — cohort caption.** Draft :25 says ten per cell; results/e53.json:arms has 12 unif and 11 acquired ctrl, with 11 pairs. **Fix:** annotate reference counts.

Survive: matched-run advantage and registered passes. Reword: approximate cap agreement and observed scaling contrast. Fail: universal size invariance, enforced depth-3 bottleneck, and c-growth with depth.
