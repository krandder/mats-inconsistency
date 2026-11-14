Model: GPT-6 (Codex)

Checked 2026-10-30: static source, hash, metadata, JSON/table/log review; no Python or recomputation of NPZ results.

1. **High: sgnmag initialization is not paired.** `e30_fmt.py:21-22` increases fan-in; `../tiny-pfn-interp/heads/pfn2.py:63-66,215-220` initializes that layer before all others. The extra random draws shift subsequent initialization; matching seeds does not match weights. Metadata confirms 18,764 versus 18,732 parameters. Fix: disclose this registration breach and qualify sgnmag comparisons; a future matched initialization must explicitly copy shared parameters. ctrl/mag/fix retain matching initialization. Episode streams share prefixes, but stopping at 50k/90k means unequal exposure.

2. **High: two different estimands are called K.** Amendment 38 defines each arm's acquired mean; `e30_summ.py:44-65` instead computes ratios on jointly acquired seeds. Both choices were present in the hashed materials, but disagree. `results/e30.json:arms` gives 0.006105/0.012889 ≈ 0.474, not 0.361; P151's arm-mean ratio is ≈1.42, not 1.31. Fix: report both, identify paired sample sizes, and label “one third” as conditional on 11 jointly acquired seeds. P150/P151 fail and P152 holds under either point-estimate definition. Bootstrap pairing is implemented correctly for its selected subset.

3. **High: mechanism and generalization exceed evidence.** Draft lines 79-84 and plain-words item 27 claim signs steer training into a worse solution and that format predicts departure size, although P150 misses. A/B is only a middle-range derivative-sign classification (`e30_eval.py:69-75`); fix's eight B nets have low KL. Fix: say format changes attained error and this diagnostic in this recipe; retain sign-driven basin selection and easier binding as hypotheses. P151's interval includes 1.1, and excluded stalled seeds cannot explain its acquired-only ratio.

4. **Medium: “three times” needs a precise comparator.** `e30_summ.py:82-83` scales test NLL gaps by nine-draw geometric-averaging shares measured on separate datasets, not actual averaged-control test KL. Fix: call 0.639/0.199 ≈3.2 a descriptive ratio to that finite-ensemble benchmark, not an identified sign cost or causal decomposition; fix also removes question-code variability.

5. **Medium: fold-change range is wrong.** `results/e30.json:per_net` gives B-seed improvements from ≈1.79× (s9) to 5.39× (s8), not 3–4×. Fix both passages. Per-seed values and A/B counts otherwise match.

Survive: format implementations, evaluation routing, every ctrl–mag seed improving, recorded P150–P153 outcomes, and estimated $1.97/$3.07 accounting (`GPU-LOG.md:26,38-45`; rate explicitly unverified). Eight code hashes match. Reword: one-third, three-times, solution-selection and general-transformer claims. Fail: identical initialization across all arms, 3–4× per-B-seed claim, and demonstrated causal mechanism.
