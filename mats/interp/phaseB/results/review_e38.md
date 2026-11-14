Model: GPT-6 (Codex)

Read-only review, checked 2026-10-30 06:43 UTC; no Python or model execution.

1. **High: the capacity conclusion outruns the registered result.** Draft:15,21,60 declares persistence across sizes and an architectural inductive bias, dismissing width 64's closing gap. `results/e38.json:configs.w64.rho` is 0.18 [-0.39,0.60]; P171 fails. Amendment 46 (`PREREG-B.md:2294`) specified the opposite interpretation for failure. The three B outcomes belong to the estimand; post-hoc medians cannot rescue it. **Fix:** report heterogeneous outcomes, uncertainty at width 64, and persistence at width 128 under this budget; explicitly identify the revised interpretation as post hoc. Neither architectural inevitability nor learned invariance is established.

2. **Medium: “both formats gain from size” is false for the reported means.** Draft:4,19; JSON `configs.w64.mag.K` is 0.00796 versus width 32's 0.00550. **Fix:** say medians decline, while mean mag KL worsens at width 64. Carry that qualification into the plain-words addition.

3. **Medium: prediction failure becomes an overly broad depth claim.** Draft:18,61; `predictions.P172.holds=false` correctly scores point estimates, but no uncertainty for rho(d3) minus half rho(w32) is reported. Ten seeds, different configuration streams, extra parameters and one schedule cannot establish that depth is “not the missing ingredient.” **Fix:** “The registered halving prediction failed in this cohort”; keep the E17 comparison descriptive and pooling mechanism hypothetical.

4. **Medium: median and population labels need precision.** Draft:17,43 calls 0.36/0.40/0.33/0.47 “paired medians”; these are ratios of medians on common seeds, not median paired ratios (0.34/0.45/0.47/0.54; JSON `configs.*.per_seed_ratio`). The first table uses independently acquired arms, including 12 mag versus 11 ctrl reference nets (`e38_summ.py:40`). **Fix:** define both statistics and denominators explicitly; avoid presenting either as the registered estimand.

5. **Low: metric wording obscures what was measured.** Draft:20 calls depth-3's marker “Block-2” and says dependence does not shrink. `e37_eval.py:92` uses block 3; `e38_summ.py:49` reports a relative nine-draw geometric-averaging reduction on separate data. **Fix:** say “last block” and “sign-cost share,” including in plain words.

**Survive:** registered configs, formats, pairing/schedule, width/depth evaluation, rom/xq/sign-cost implementation, paired bootstrap logic, rho/P171–P172 scoring, and numerical tables. Hashes match; 1,510 numeric fields across 84 NPZs match JSON. **Reword:** capacity, depth, typical-gain and sign-dependence claims. **Fail:** universal mean improvement and demonstrated architecture-level explanation.
