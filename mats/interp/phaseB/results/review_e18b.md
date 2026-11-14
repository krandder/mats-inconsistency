Model: GPT-6

Read-only review, checked 2026-10-29 against code, hashes, JSON and call logs; no Python or experimental reruns.

1. **High: causal comparison is unidentified.** Draft:7,29,69–73 attributes poorer accuracy to removing reasoning and calls E18's shift “most likely bias handling.” E18 used different model generations; histories also change (kappa 0.85→0.75, signs and means: e18_data.py:58; e18b_data.py:23–27). Fix: report a descriptive cross-experiment difference; establish causality only with the same models and items crossed with reasoning and mean-error controls. “Not a general property” also exceeds a failed directional test.

2. **High: the plain-language null contradicts the registered result.** Draft:9 says “taken together ... no clear effect either way,” but e18b.json:predictions.U gives +3.113 [1.153,5.358]; U_qwen17 also has a positive interval. P132 failing means the predicted negative direction failed, not no effect. Fix: distinguish the positive, Qwen-driven registered pool from the inconclusive post-hoc API-only pool; replace “essentially noise” with “large errors relative to Bayes.”

3. **Medium: retry rates are wrong and selection matters.** Draft:40–41's first-attempt failures are actually 53/240=22.1% stated and 32/120=26.7% unstated (e18b_calls.jsonl:attempt=0, model=sonnet46). The quoted 28%/41% describe all attempts. Registered retries are implemented, but conditioning on compliant answers can select easier items/answers. e18b_summ.py:26–31 accepts partially sampled pairs; :57 pools only common pairs (59/51), affecting every model. Fix rates; disclose these denominators and add first-attempt/completeness sensitivity analyses.

4. **Medium: “single-pass” needs an operational definition.** e18b_call.py:45–51 requests GPT effort none but saves no reasoning-token telemetry: absence of observable reasoning is not verified zero internal tokens. Qwen :63–67 uses non-thinking, greedy autoregressive generation, not one forward pass. All accepted texts are bare numbers; all 1,064 accepted Claude calls explicitly log think_tok=0. Fix: say “direct-answer generation with reasoning disabled/requested off”; distinguish calls from retries.

5. **Low: exact equality and budget wording.** Items' maximum displayed mean magnitude is 0.0025, SS mismatch 0.0627, Bayes-mean mismatch 0.001287. Fix: qualify zero-mean/SS equality as pre-rounding (e18b_data.py:27). The cap counts Claude cost only, not GPT (e18b_cap.sh:9); label it accordingly.

**Survive:** code/items hashes match; arms, parser, measures and bars match registration; reported arm statistics and P131/P132 failures match JSON; all five models qualify in each framing. **Reword:** single-pass, pooled conclusions, equality. **Fail:** first-attempt percentages, causal reasoning/bias attribution, and pooled “no clear effect either way.”
