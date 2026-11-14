# Eliciting which answers should carry weight in the projection: results (2026-09-27)

Kelvin's question (27 Sep): when the model's 12 answers are projected onto a coherent joint, which answers should
get the most weight, and can we elicit that so a weighted projection beats equal weights? He added a priority idea:
"how much might you change your mind on this if you think for longer?" The hypotheses (12) and their literature
are in docs/eliciting-weights-ideas-20260927.md.

**Answer.** Letting the model think is what helps. Asking it to predict how much it would change its mind does not.

- **Measured revision works; predicted revision does not.** On Qwen3-32B, each of the 12 quick answers
  (thinking off) was compared with the same prompt answered with thinking on (budget 1,024 tokens).
  - Down-weighting the answers that moved most beats equal weights: −0.015 nats/answer [−0.026, −0.003].
  - Replacing the quick answer by the thinking answer before projecting gains twice as much: −0.030 [−0.047, −0.012].
    Averaging the two answers in log odds gives a tighter CI: −0.026 [−0.035, −0.017].
  - With the thinking answer as the point, the weights add almost nothing (−0.031 vs −0.030).
  - So the gain is the better answer, not a better weight.
- **The model cannot predict its own revision.** The predicted change ("Change: 0.xx") is uncorrelated with the
  measured change once extremeness is controlled (partial ρ 0.009 on 32B, 0.05 on 235B). As a weight, it gains
  nothing that is significant.
- **No other elicited or label-free signal beats equal weights significantly.** The per-answer signals are:
  - the trust score from the 12-answer self-critique;
  - comparative evidence directness;
  - truth discovery.

  The only other significant gain is stacking on the form alone, which learns the per-form weights from labels on
  the training folds: −0.008 [−0.014, −0.001].
- **235B.** The thinking answer does not help the 235B model, whose quick answers are already better (0.550 vs
  0.619 raw). This was tested on 30 pairs only, because 235B thinking tokens cost $8.4/M. No weight is
  significant on 235B.
- **Structured pairs replicate the 32B result** (300 sampled pairs with a known relation). With the thinking answer
  as the point, recovery of the relation's gain rises from 0.40 to 0.64. The hybrid joint log loss improves by
  −0.008 [−0.015, −0.001].
- **Headroom is large but not found.** The best of 25 weight vectors per pair (an oracle that uses labels) would
  reach 0.443 against 0.538 for equal weights on 32B. No elicited signal gets close.

Setup. Qwen3 on DashScope, question only, v6_direct prompts, one form per call.
- Main test: the 300-pair subset of the 1,066-pair test, the same pairs as docs/confidence-weighting-20260927.md.
  The quick point is the T=0 logprob answer.
- Structured pairs: see below.
- Projection: `project(T, W, "kl")` with W = 1/(V + τ).
- Cross-fitting: τ (or the stacking θ) is cross-fitted over 5 folds by pair (seed 20260924) and picked on the
  training folds by all-12 log loss. τ = ∞ means equal weights.
- CIs: paired bootstrap 95%, 2,000 resamples.
- Log loss: nats per answer over all 12 forms. Lower is better.

## Main table: Qwen3-32B, 300 pairs (3,600 answers)

Equal weights: 0.5375. Per-form weights (label-free sampling variance): 0.5330.

| Arm | Log loss | vs equal | vs per-form |
|---|---|---|---|
| per-form weights | 0.5330 | −0.0046 [−0.0094, +0.0005] | |
| **predicted revision** (Change²) | 0.5359 | −0.0017 [−0.0055, +0.0021] | +0.0029 [−0.0041, +0.0097] |
| predicted revision × form | 0.5342 | −0.0033 [−0.0084, +0.0020] | +0.0012 [−0.0018, +0.0042] |
| **measured revision** (Δlogit² quick→think) | 0.5223 | **−0.0152 [−0.0262, −0.0030]** | −0.0107 [−0.0213, +0.0012] |
| measured revision × form | 0.5240 | −0.0135 [−0.0267, +0.0009] | −0.0089 [−0.0212, +0.0039] |
| critique trust (5 − E[trust]) | 0.5384 | +0.0009 [−0.0004, +0.0024] | +0.0055 [+0.0001, +0.0104] |
| critique revision distance | 0.5347 | −0.0029 [−0.0089, +0.0033] | +0.0017 [−0.0050, +0.0088] |
| evidence directness (τ always picked equal) | 0.5375 | 0 | +0.0046 [−0.0005, +0.0094] |
| truth discovery, per-form residual (5 iterations) | 0.5361 | −0.0014 [−0.0034, +0.0007] | +0.0032 [−0.0003, +0.0065] |
| truth discovery, Huber per answer | 0.5379 | +0.0004 [−0.0043, +0.0057] | +0.0049 [−0.0013, +0.0117] |
| stacking: form only | 0.5297 | **−0.0079 [−0.0141, −0.0014]** | −0.0033 [−0.0101, +0.0034] |
| stacking: form + free features | 0.5373 | −0.0002 [−0.0110, +0.0114] | +0.0043 [−0.0066, +0.0160] |
| stacking: form + measured revision | 0.5207 | **−0.0168 [−0.0279, −0.0044]** | **−0.0122 [−0.0230, −0.0002]** |
| stacking: form + free + elicited | 0.5287 | −0.0088 [−0.0205, +0.0048] | −0.0043 [−0.0164, +0.0093] |
| *point:* thinking answer, equal weights | 0.5079 | **−0.0296 [−0.0466, −0.0124]** | **−0.0251 [−0.0409, −0.0088]** |
| *point:* thinking answer, measured-revision weights | 0.5067 | **−0.0308 [−0.0478, −0.0136]** | **−0.0263 [−0.0420, −0.0104]** |
| *point:* mean(quick, thinking), equal | 0.5117 | **−0.0258 [−0.0345, −0.0170]** | **−0.0213 [−0.0303, −0.0129]** |
| *point:* mean(quick, thinking), per-form | 0.5103 | **−0.0273 [−0.0366, −0.0173]** | **−0.0227 [−0.0311, −0.0147]** |
| *point:* critique revised answers, equal | 0.5408 | +0.0033 [−0.0097, +0.0165] | +0.0078 [−0.0051, +0.0207] |
| *point:* the rev prompt's own answer, equal | 0.5499 | +0.0124 [+0.0029, +0.0220] | +0.0170 [+0.0067, +0.0272] |
| *oracle* (labels): best of 25 vectors per pair | 0.4432 | −0.094 | |

Raw answers before projection, log loss vs the quick answer (0.6193):
- thinking answer: 0.5425 (−0.077 [−0.097, −0.057]);
- mean(quick, thinking): 0.5465;
- critique revised: 0.6001 (−0.019 [−0.034, −0.004]), but that gain disappears after projection;
- the rev prompt's own answer: 0.6336 (+0.014). Asking for a "Change" line makes the probability worse.

Thinking details (32B):
- 72% of the thinking calls hit the 1,024-token budget.
- Mean |Δp| between the quick and thinking answers is 0.144; 68% of answers moved more than 0.05.
- Thinking runs at T=0.6 (the API requires sampling with thinking on), while the quick answer is T=0.

## Does a signal predict per-answer error beyond extremeness?

The table gives the Spearman correlation with the answer's own log loss, partial on |logit p|. Positive means a
higher signal goes with a worse answer, which is what a noise weight needs.

| Signal | 32B | 235B |
|---|---|---|
| predicted revision (Change²) | +0.011 | +0.100 |
| measured revision (Δlogit² quick→think) | **+0.135** | −0.052 (360 answers) |
| critique trust | +0.032 | +0.012 |
| critique revision distance | +0.115 | +0.006 |
| evidence directness | +0.010 | not run |

Does (a) predict (b)? Predicted change vs measured |Δp| on 32B (3,595 answers):
- Spearman 0.039 pooled; 0.009 partial on extremeness;
- −0.02 on per-pair means.

On 235B (360 answers): 0.117 pooled, 0.05 partial, −0.18 per pair.

The model's stated uncertainty about its own revision mostly restates how extreme the answer is (ρ with
extremeness −0.26 on 32B, −0.47 on 235B).

## Qwen3-235B, 300 pairs

Equal weights: 0.5142. Per-form weights: τ picked equal in every fold. All arms are within ±0.003:

| Arm | vs equal |
|---|---|
| predicted revision | −0.0010 [−0.0027, +0.0009] |
| critique trust | −0.0018 [−0.0062, +0.0028] |
| critique revision distance | −0.0011 [−0.0056, +0.0039] |
| point: mean(quick, critique revised) | −0.0024 [−0.0065, +0.0020] |
| truth discovery (5 iterations) | +0.0002 [−0.0002, +0.0007] |

Other 235B results:
- Raw critique-revised answers beat the quick answers: 0.5378 vs 0.5497 (−0.012 [−0.020, −0.004]). The gain does
  not survive projection.
- The thinking run was stopped at 30 pairs (360 answers, $3.11). On these pairs:
  - raw thinking answer vs quick: −0.001 [−0.062, +0.063];
  - measured-revision weights: +0.013 [−0.003, +0.032];
  - thinking answer as the point: +0.005 [−0.051, +0.061].

  The test is underpowered, but it gives no sign of the 32B gain.
- Oracle headroom: 0.442 vs 0.514.

## Structured pairs, Qwen3-32B (known relation; recovery = the hybrid's share of the joint-LL gain)

Setup:
- A random sample of 300 of the 1,077 structured pairs (seed 20260927) got thinking answers for all 12 forms.
- The point is the 32B T=0 logprob rerun.
- Clustered bootstrap CIs.
- "Joint" is the joint log loss of the projection, or of the hybrid (market marginals + the projection's log odds
  ratio).
- τ is picked by the model's all-12 log loss (_m) or by the hybrid's joint log loss (_h).
- Recovery with equal weights is 0.395.

| Arm | Recovery | Projection joint vs equal | Hybrid joint vs equal |
|---|---|---|---|
| per-form weights _m | 0.402 | | −0.0002 [−0.0018, +0.0013] |
| critique trust (τ always picked equal) | 0.395 | 0 | 0 |
| measured revision _m | 0.470 | −0.0134 [−0.0229, −0.0023] | −0.0025 [−0.0054, +0.0004] |
| form × measured revision _m | 0.505 | −0.0210 [−0.0319, −0.0090] | −0.0037 [−0.0071, −0.0003] |
| *point:* thinking answer | **0.644** | −0.0301 [−0.0519, −0.0072] | **−0.0084 [−0.0154, −0.0014]** |
| *point:* mean(quick, thinking) | 0.617 | −0.0290 [−0.0416, −0.0171] | **−0.0075 [−0.0115, −0.0040]** |
| *point:* critique revised | 0.297 | +0.0343 [+0.0184, +0.0510] | +0.0033 [−0.0013, +0.0085] |

On all 1,077 structured pairs, the critique-only run agrees:
- critique trust: τ picked equal in every fold;
- critique revised as the point hurts: projection joint +0.029 [+0.021, +0.037];
- recovery: 0.342 for critique revised, against 0.453 for equal weights.

The structured pairs replicate the 32B result. Thinking raises recovery of the known relation from 0.40 to 0.64,
and the measured-revision weight helps on its own, but less.

## What works, what doesn't

- **Works (32B):** thinking. Use the thinking answer, or its log-odds mean with the quick answer, as the point.
  - It costs about 960 reasoning tokens per answer, about $0.0007 on 32B, or 7× a quick call.
  - It gains −0.026 to −0.031 nats/answer after projection.
  - The measured revision (quick→think) is the only per-answer signal found that predicts error beyond
    extremeness (partial ρ +0.135). It helps as a weight (−0.015) or stacked with form (−0.017). Once the thinking
    answer is the point, the weight is nearly redundant.
- **Doesn't work:**
  - predicting one's own revision;
  - trust scores from the self-critique;
  - comparative evidence directness;
  - label-free truth discovery;
  - stacking many cheap features (it overfits);
  - using the critique's revised answers as the point (better raw, not better after projection).
- **Not tested** (ideas 7–12): betting against the consensus, base-rate decomposition, knowledge quizzes,
  meta-prediction, framing preference, conditional anchors. The oracle gap says per-pair weights could matter a
  lot, but none of the signals here finds them.

## Process notes, cost, files

Cost: $12.06 of the $15 DashScope cap (checked 27 Sep after the last run), summed from the `cost` fields of the logs. By run:

| Run | Variants | Calls | Cost |
|---|---|---|---|
| 32B, 300 pairs | rev, critique, evidence | 4,200 | $0.49 |
| 32B, 300 pairs | thinking | 3,600 | $2.48 |
| 235B, 300 pairs | lp | 3,600 | $1.30 |
| 235B, 300 pairs | rev, critique | 3,900 | $1.89 |
| 235B, 30 pairs | thinking | 360 | $3.11 |
| 32B, structured | critique | 1,077 | $0.29 |
| 32B, structured | thinking | 3,600 on a 300-pair sample | $2.49 |

Parallelism overran the ≤8 cap once. While the structured runs were switched over, a wrapper PID was killed
instead of its python child, so for about 15 minutes 11 workers ran concurrently (5 + 3 + 3). It was fixed by
killing the child. Because of the switch, the structured predicted-revision run stopped at 75 calls, so predicted
revision is not tested on the structured pairs.

Data in /home/agent/data/eliciting-weights-20260927/:
- old32/, old235/ and struct32/: `log.jsonl`, one row per call, with reasoning text;
- `*-analysis.json` and `struct32-*.json`.

Rerun:
```
python3 scripts/run_eliciting_weights.py --model qwen3-32b --out $D/old32 --variants rev,critique,evidence,think   # paid
python3 scripts/eliciting_weights.py --log $D/old32/log.jsonl --out $D/old32-analysis.json          # CPU, ~40 min with stacking
python3 scripts/eliciting_weights.py --log $D/old235/log.jsonl --out $D/old235-analysis.json --quick-log $D/old235/log.jsonl
python3 scripts/eliciting_weights.py --log $D/old235/log.jsonl --out $D/old235-think-analysis.json --quick-log $D/old235/log.jsonl --think-subset --no-stack
python3 scripts/eliciting_weights_structured.py --log $D/struct32/log.jsonl --out $D/struct32-think.json --think-subset
```

Next step, if wanted: thinking at a larger budget (72% of calls hit 1,024), and two thinking samples per answer,
on the full 1,066 pairs. That costs about $9 on 32B.
