Model: GPT-6 (Codex)

1. **High: the reference statistic is not a measured large-value attention effect.** `e67_run.py:113-118` evaluates +3 only, against zero raw columns, using mean a and mean L. In sgnprod, mean products remain fixed; this is not a valid x=0,y=3 token (which requires y²=9). Mean L also differs from alpha(mean a)/||mean a||. Norm cross-terms distinguish ±3. **Fix:** call this the registered synthetic +3 diagnostic, specify its zero/reference, and remove “keeps or gains weight” (`update76_draft.md:4,48`). A component of a logit does not establish softmax weight changes.

2. **High: routing specificity and the M2 explanation are unidentified.** `update76_draft.md:48,51` compares different tasks, layouts, sufficient statistics, and selected world-1 summary heads versus all world-2 query heads (`e67_summ.py:21-25`). E54 measures matched residual-recipe departures, not this diagnostic (Update 63:6235-6240). **Fix:** describe compatibility with the routing hypothesis; retain E54 as contextual agreement, not a demonstrated mechanism.

3. **High: “no negative norm term” is false.** `results/e67.json:cohorts.d1sgnprod.dlogit_y3` is -0.314; `results/e67/d1sgnprod.json:e48_d1sgnprod_s8.dlogit_y3` has both heads below -1. **Fix:** state “136/138 head averages exceed -1”; P261 never tested nonnegativity.

4. **Medium: replacement is overstated.** `update76_draft.md:5,17,49` turns near-unit median KL factors into “do not use…at all.” Surgery preserves signed dependence through norms, values, other rows and later blocks (`e67_run.py:55-68`). **Fix:** report negligible median sensitivity of this numerator contribution; replacement by products remains a mechanism hypothesis. M2 must also include the product-dependent numerator a_j.

5. **Medium: canon attribution is wrong.** Draft line 18 attributes 4.7–5.3× to y removal; `e67_run.py:56-63` removes x and y together. **Fix:** say “folded raw-column terms.”

6. **Low: “exact” needs qualification.** `e67_run.py:106-118` omits LayerNorm epsilon; surgery includes it. **Fix:** qualify the decomposition or include epsilon.

Survive: matching hashes, cohorts/exclusions/acquisition, layout split, example-averaged L, K normalised by mean ||a_j||, surgery, KL direction, tables and registered P260–P263 scoring. Reword the plain-language and M2 claims. Universal nonnegativity, y-only attribution and established causal replacement fail. Read-only review; no Python executed.
