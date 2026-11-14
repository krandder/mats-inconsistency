Model: gpt-6-astra

Read-only review, checked 2026-11-01 against source, stored JSON, tables and validation curves; no Python or training/evaluation reruns.

1. **High: KL non-growth is not established.** `results/e74.json:predictions.P297.ratio` allows 84% growth; `results/e74_desc.json:P297_stat_without_s8_s9` still allows 30%. Removing the two late nets is post hoc conditioning on an outcome. Fix draft lines 9, 72, 75: retain “P297 fails its registered criterion”; replace “same cost,” “cost moved into acquisition,” and “therefore scale differently” with descriptive point estimates and uncertainty. The bootstrap covers seeds, conditional on fixed test episodes.

2. **Medium: acquisition is overstated and misattributed.** Both ctrl cohorts have one stall among twelve; “less reliable” is unsupported. The curves support later first crossings of gap <0.10, not calibrated task mastery across S (`train_n.py:51-55,75-82`). `e74_desc.py:14-64` never reads curves, contrary to draft line 32. Fix: report threshold-crossing times, equal final acquisition counts, and the actual extraction procedure; keep type changes descriptive because episode streams also change.

3. **Medium: leakage comparisons hide selection.** `e74_summ.py:49-52` excludes S3 s8 because it has no pooling head; its `leak_h[0]` is 0.584 (`results/e74/S3/e37_ctrl_s8.json`). P299 compares 11 versus 10 eligible nets, and `e74_desc.py:50-54` restricts B3 to six of seven B nets. Fix: disclose these denominators; say “B nets with a pooling head.” Preserve P299's registered pass.

4. **Medium: mechanistic language exceeds measurement.** Draft lines 6, 71, 73: constant total leak implies lower average mass per competing token, not demonstrated sharper code discrimination. L_own is not an isolated code contribution; the norm term is not the whole score. Fix: describe co-movement, label mechanisms hypotheses, and say A rises about 1.6-fold, B about 1.9-fold.

5. **Low: “exact” decomposition omits LayerNorm epsilon.** `e74_eval.py:83-89` omits it; `pfn2.py:38` uses LayerNorm. Fix: qualify the approximation; E63's bound was not checked here.

Survive: S-aware loaders/masks, gap, matched-pair rom, head classifications/sign pairs, ten-seed paired bootstrap, tables and P298–P301 passes. Reword: causal scaling, sharpening and acquisition claims. Fail: P297's registered criterion; stronger equivalence/reliability conclusions.
