GPT-6 (Codex)

Checked 2026-10-31 04:30 UTC: code, registration, hashes, JSON, tables and prior reports; no Python/model execution or independent NPZ decoding.

1. **High: necessity and mechanism overclaimed.** Draft lines 6, 20, 59, 61 turn opposite mean coefficients into “needs two heads” and a freed-head explanation. `e63_run.py:140-143` measures numerator coefficients and one denominator contribution; nonzero rho makes the denominator sign-dependent. Opposite means do not establish even pooling through unequal value/output maps. E65 measures neither E37's switched heads nor E39's heads. Fix: restrict the algebra to block-1 logits; describe sign pairing and E30/E37/E39 explanations as compatible hypotheses, not demonstrated necessity or sufficiency.

2. **High: positive K is not increasing total score/weight.** Draft lines 5, 17, 59 imply this. `e65_run.py:71-72` separates dlogit from K; the epsilon-free total change is L(A/n−1)+KrA/n. Positive K alone cannot determine that change, its derivative, or softmax weight. Fix: “positive mean normalized numerator coefficient,” with a positive linear contribution; retain the negative norm contribution separately.

3. **Medium: allocation/count prose contradicts tables.** `results/e65.json:cohorts.d3ctrl.B.n_pool=[2,4,0]`: B has fewer than two, not always one (draft 4,58). `w128mag.A.n_pool=[1,9,0]`, `w32mag.A.consensus=10`: mag A exceptions must remain explicit. The 36 mag pooling heads comprise 35 A heads and one B head; 36 allocation exceptions include one zero-head A. Cohort counts total **72**, not 70. Fix lines 4-5,17,26,54 accordingly.

4. **Medium: size ordering changes population.** Draft line 7 generalizes A-only summaries to all mag nets. `e65_summ.py:35-44` conditions on type and summarizes heads versus nets. Fix: explicitly say “A nets, four cohorts, post hoc association”; replace “orders/explains” (61) with “co-orders descriptively.”

5. **Low: inherited approximation omitted.** `e63_run.py:134-143`, `e65_run.py:35` omit d·eps used by `pfn2.py:38`. Fix: label epsilon-free; E63's error bound does not verify new cohorts.

Survive: dynamic head dimensions, sqrt(d/dh), block-1-only extraction, mag y+|r| encoding/+3 term, acquisition/types/exclusion, tables and registered P247/P248/P250 passes and P249 failure. Consensus attention is demonstrably insufficient for B. Reword mechanisms and size association; universal allocation and head-necessity claims fail.
