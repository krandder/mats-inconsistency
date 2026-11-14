Model: GPT-6 (Codex)

Read-only review, checked 2026-10-30 against source, hashes, logs and JSON; no Python or numerical reruns. “Draft” means update68_draft.md.

1. **High: the budget equality is false.** Draft:17 equates ctrl's all-net KL 0.013347 with shared+varying = 0.012574. Those components sum to the sampled subsets' member mean, not the all-net mean (`results/e59.json:arms.ctrl.{kl,members7,shared7,varying7}`; `e59_summ.py:52-54`). Unif likewise sums to 0.005111, not 0.005186. **Fix:** report members7 alongside the split; retain all-net KL separately. The approximately 85% varying reduction and 21% shared reduction are correct comparisons of these fitted components.

2. **High: nuisance causation is not identified.** Draft:7,18,44,46 assigns most KL to learned pooling's nuisance-tracking errors. E51 measured sign-shift correlations in predictive means, not a partition of density KL; E59 does not jointly attribute seed variation to codes/signs. Fixed pooling also supplies membership (`pfn_uni.py:27-33`), and training device differs (Draft:47). **Fix:** say the three reductions co-occur in these cohorts; keep weighting, membership and device explanations unresolved. The largest *fitted varying component* survives; its proposed causal identity does not.

3. **High: additive floor is unestablished.** Draft:45-46: 0.002844 + 0.000968 = 0.003812 is close to unif shared7 = 0.003995, but E51's second-order seed model supplies no additivity theorem across independently trained architectures and an approximate even-feature Bayes reader. **Fix:** call this numerical compatibility, and “floor” an observed residual under this recipe. Cap cost is smaller in magnitude, not an identified removable component. Five overlapping subsets and eight sq nets give no population uncertainty or demonstrated equivalence.

4. **Medium: ctrl cap cells lack matching provenance.** Draft:30 adds 3.79/~0.00004 absent from `e59.json` and generated tables (`e59_summ.py:55-57` restricts costs to uni/unif). Cost at median c differs from mean per-net cost. **Fix:** label this external illustrative estimate and source, or remove it.

5. **Medium: plain words overstate precision.** Draft:3-5,44 says “removes,” “neither,” and “closer to the correct answer.” **Fix:** say nearly removes measured dependence and reduces KL to Bayes using normalized geometric averaging.

Survive: registered SID, shares/bootstrap, ensembles and P227–P228; hashes match. `set_stats` returns member-mean KL, reduction, mass (`e26_run.py:94-97`); ensemble KL correctly subtracts reduction. Other table values match JSON. Reword: 85%, cap comparison, shared residual. Fail: displayed equality, established floor/additive causal budget.
