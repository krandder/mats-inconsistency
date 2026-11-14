Model: GPT-6 (Codex)

Checked 2026-10-30 06:40 UTC, read-only shell/JSON inspection; no Python or training.

1. **High — Determinism and negligible-data claims exceed the evidence.** `update50_draft.md:3,24,55–58`; `results/e43.json:per_net.br0_s2`: changing the stream flips A→B and raises KL from 0.00643 to 0.01450. There is only one replacement stream per parent, 11 selected acquired parents, one recipe and one sign diagnostic. Correlation 0.76 does not identify the fraction of deviation attributable to initialisation. Fix: say “strong initialisation-associated persistence under these tested stream replacements”; retain the exception, avoid “decides,” “fixed,” and “most variation,” and treat the E6 group identification as a hypothesis. These are fresh episodes, not merely a permutation of fixed examples (`train_br.py:70–73`).

2. **High — KL difference claims are wrong or misleading.** Draft lines 4,20–21,56; JSON `per_net`, `arms.br0.kl_ratio_median`. Median(br0/parent)=0.974 does not mean “median 3% apart.” Median absolute relative difference, median(|KL/parent−1|), is **9.04%** for br0 and **27.52%**, not 24%, for acquired newinit. Fix both numbers, name the denominator, and keep “median” beside the 9% claim.

3. **Medium — Chance/equivalence language ignores conditioning.** Draft lines 5,21,45; `e43_summ.py:44–58`. The registered 0.537 chance reference is correctly computed, but newinit uses eight survivors; its 0.344 similarity mixes same/different-type pairs, whereas 0.408 uses only same-type parents. Neither comparison establishes “no more than unrelated” or equivalence to chance. Fix: report descriptive comparisons, eight-survivor conditioning and three stalls; reserve equivalence claims for matched analyses with uncertainty.

4. **Medium — Blanket rank statement is false.** Draft line 22; JSON `per_net`: br3k Spearman is **0.9636**; br2k/br5k/br10k/br20k are 0.9909. br0 0.6455 and acquired newinit 0.2381 are correct. Fix: report 0.96–0.99.

5. **Medium — Endpoint/commitment wording needs precision.** `e30_eval.py:31` loads best-validation `model.pt`, not step-90,000 weights; metadata selects steps 85,000–90,000. `PREREG-B.md:2493` defines first K with agreement≥0.9: **K=0** already qualifies. Fix endpoint wording and report that registered descriptive result; 11/11 does not establish population certainty.

Survive: P183 passes/P184 fails as registered; agreement, per-seed table, similarity and chance match JSON/code. Seeds/streams/schedule match registration; 231 saved pre-branch checkpoint comparisons are byte-identical; parent links/acquisition check out. Reword: initialisation predominance and commitment. Fail: deterministic/negligible-data conclusions, 3%/24% differences, blanket 0.99.
