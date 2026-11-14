Model: GPT-6

Checked 2026-10-29 18:37 UTC; read-only inspection, hashes, metadata and JSON; no Python or numerical reruns.

1. **High: failure of P126 is not evidence that width cannot shrink the effect.** `results/e21.json:widths,predictions.P126` gives means .127/.185/.065/.134 and rho −.122. Five seeds per new width and episode-only intervals (`e21_summ.py:57-73`) cannot establish equivalence or that seed effects dominate width. **Fix:** replace “does not shrink with width” and draft lines 69-72 with “persists in all 15 tested nets; no monotone decline in these sample means.” Keep pooling order as a compatible hypothesis, not an established explanation.

2. **Medium: convergence limitations omitted.** `runs/e21_w64_s{0,1,3}/meta.json:converged` and `runs/e21_w128_s1/meta.json:converged` are false. All four reached 90,000 steps. This follows the registered cap, so is not a protocol violation; acquisition does not imply convergence. **Fix:** disclose 4/15 failed the stopping criterion and restrict conclusions to this training budget.

3. **Medium: group language exceeds the measured signature.** `results/e23.json:_P130` supports 13/15 agreement, but always predicting B gets 12/15. Negative rho identifies A in only 3/5. `e23_run.py:27-28` pools relative attention shares across episodes, sources and heads; it does not establish a causal correction or the original wiring-defined groups (Update 25). **Fix:** say “a correlational layer-1 marker predicts the middle-range sign”; include the baseline and exceptions. In plain words, qualify “both groups still form” as “both middle-range signatures occur.”

4. **Medium: timing language hides prior exposure.** Amendment 30 explicitly admits training-log reads (`PREREG-B.md:1601-1602`); those contain validation gaps and best checkpoints (`pfn2.py:262-263`). Registration prefixes and code reproduce the hashes at `prereg_hashes.txt:203-219`; earliest saved E21 measurement is 18:32:44, after 17:51:46 registration. That timestamp is completion, not measurement start, and cannot prove absence of earlier analysis. **Fix:** state “registered before saved departure/attention measurements; training progress was visible.” No outcome leakage was found.

5. **Low: GPU accounting needs qualification.** `GPU-LOG.md:5-29` records one rejected ID, two stockouts and one successful VM; line 29 switches the cap from $10 to $40. Deletion is corroborated by `/tmp/l4runs/cmb-e21-w3/gpu-run.out:1122`; the empty-list check is only narrated. **Fix:** reconcile caps, retain that distinction, label $1.10 an estimate.

Survive: reported numbers, P125/P127/P130 passes and P126 failure; registered recipe, widths/seeds/measures/bars. `pfn2.py:38-65` with default overrides implements the same architecture family at width/2 head dimension; staged hashes match. Reword: width generality, group identity and blinding. Fail: seed-dominance inference and any causal interpretation of E23. No implementation deviation found.
