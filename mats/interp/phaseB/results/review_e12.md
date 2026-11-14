Model: GPT-6 (Codex)

Read-only review, checked 2026-10-29 12:55Z; no Python or model reruns.

1. **Medium: undisclosed bootstrap conditioning.** `e12_summ.py:19–20` resamples only `unique(base)` after range filtering (`e12_data.py:22`). Reading `results/e12_pairs.npz:base` gives 2,875 represented episodes, not all 3,000 generated episodes: 125 have no eligible rows. Amendment 19 (`PREREG-B.md:1083–1084`) specifies base-episode resampling without this conditioning. Clustering is preserved, but this is not literally the full-base bootstrap. **Fix:** disclose the conditioning and rerun the registered bootstrap over all 3,000 IDs, retaining empty clusters, before asserting exact compliance. The direction is strongly supported; revised interval endpoints remain unchecked.

2. **Medium: headline implies a per-pair result.** Draft lines 3, 24, 38 and 73 say every net gives a large-error source more influence. The test establishes each net's *mean* contrast, not positivity for every source/pair (`e12_summ.py:23,29,33`). **Fix:** say “On average across the tested large-error pairs, all 12 nets give the source more influence…” Define influence as sensitivity to its query reading; the percentages are ratios of summed weight differences to summed Bayes weights, not typical individual percentage changes.

3. **Medium: analogy loses its strongest caveat.** Draft lines 6 and 75 retain “analogy,” but omit that E11's absolute-error match used an extra post-hoc objective matching three group means; its per-row R² at p=1 was −0.29 (Update 21:2283–2284,2326–2335). E12 does not validate that statistic. **Fix:** restore “post-hoc match to three averages, with poor per-pair fit”; identify the statistic as a calibrated squared sum of absolute errors. Robustness itself remains untested.

4. **Low: timestamp error.** Draft line 39 gives 12:42:09Z; pair-file birth/mtime are 12:42:08.236/.336Z. **Fix:** use 12:42:08Z. Both follow the logged 12:41:40Z hash, which matches current PREREG-B.md (`84f00872…647a76`). Filesystem chronology supports the sequence, without independently attesting generation history.

**Survive:** JSON numbers, rounding, sign counts, and saved P93/P94 verdicts; registered ranges, pair construction, K=8/K=64 measures, shared draws, invariance guard and decision bars. Draft line 51 honestly conditions intervals on the fixed nets. All 12 run files/log completions exist.
**Reword:** average-effect headline, analogy, bootstrap scope, timestamp.
**Fail:** exact bootstrap compliance; universal per-pair or confirmed robust-mechanism interpretations. Neither core directional finding fails this review.
