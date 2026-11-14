# Qwen3-32B zero-shot verbalized baseline, 23 September 2026: results

Frozen Qwen3-32B (bf16, revision `9216db57`), no training. For each of the 12
Boolean forms of every pair the model writes a probability (`Probability: 0.xx`),
which is the forecast. Data: `jev-disjoint-v1-filtered-20260919`. Method (reviewed
before rental): `docs/qwen32b-verbalized-methodology-20260923.md`. Code:
`training/verbalized_baseline.py`, `scripts/run_verbalized_baseline.py`,
`scripts/score_verbalized_baseline.py`, `scripts/select_verbalized_variant.py`,
Vast lifecycle in `ops/vast-qwen32b-20260923/`. Outputs:
`/home/agent/data/qwen32b-baseline-20260923/` (predictions per query, scores,
`manifest.json`).

## Setup

- Prompts render each proposition with the option name (`answer_text`), creator
  edits stripped from the criteria (validation 378/844 and test 489/2,132
  propositions affected), criteria capped at 1,500 characters, and an as-of date
  (the pair's earlier closing week minus 7 days).
- Non-thinking mode (`enable_thinking=False` and `/no_think`), greedy decoding,
  vLLM 0.29.0 offline on one H100 SXM, `max_model_len` 4096, prefix caching.
- Six prompt variants searched on validation only (422 pairs, 5,064 queries each).
  Rule fixed in advance: lowest all-12-form log loss with missing imputed at 0.5,
  among variants with at most 5% missing. Test (1,066 pairs, 12,792 queries) was
  scored once with the chosen variant.
- Missing answers: 0 in every non-thinking run. v4 needed 7 format retries, and all
  7 succeeded, so the missing-excluded and 0.5-imputed scores are identical.

## Validation prompt search

| Variant | LL all 12 | Brier | Acc | ECE | LL observed | Consistency |
|---|---:|---:|---:|---:|---:|---:|
| v1 plain, symbolic forms | 0.6782 | 0.2318 | 0.651 | 0.095 | 0.7180 | 0.0281 |
| v2 plain language forms | 0.6613 | 0.2258 | 0.660 | 0.075 | 0.7199 | 0.0203 |
| v3 superforecaster + train base rate | 0.6506 | 0.2223 | 0.669 | 0.069 | 0.6902 | 0.0205 |
| v4 logic steps, both questions | 0.6857 | 0.2306 | 0.659 | 0.119 | 0.7422 | **0.0122** |
| v5 both questions as context | 0.6519 | 0.2222 | 0.666 | 0.068 | 0.6866 | 0.0181 |
| **v6 direct answer (chosen)** | **0.6467** | 0.2201 | 0.676 | 0.098 | 0.6973 | 0.0160 |
| Boolean prior (validation) | 0.6059 | | | | 0.6159 | 0 |
| Per-form train rate (validation) | 0.6021 | | | | 0.6160 | 0 |

The plain-language forms beat the symbolic ones (0.661 vs 0.678). Reasoning did not
help: the direct one-line answer scored best. The logic-steps prompt gave the most
consistent answers but the worst log loss.

**Thinking ablation** (v6 on the first 50 validation pairs, 600 queries, thinking
on, 4,096-token budget): log loss 0.631 vs 0.633 without thinking on the same
queries. Consistency improved from 0.0156 to 0.0115. The run took 43 min for 600
queries, and 33 replies (5.5%) hit the token limit. The format retry recovered all
33, so none is missing. Thinking gives no gain worth its cost.

## Test (chosen v6, scored once)

| Predictor | LL all 12 | Brier | Acc | ECE | LL observed (1,066) | Consistency |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-32B zero-shot, raw | 0.6114 | 0.2063 | 0.703 | 0.062 | 0.6155 | 0.0217 |
| Qwen3-32B + Platt (fit on validation) | 0.5946 | 0.2034 | 0.703 | 0.037 | 0.5971 | 0.0086 |
| Constant 0.5 | 0.6931 | 0.2500 | 0.500 | 0 | 0.6931 | 0 |
| Boolean prior 0.5/0.25/0.75 | 0.6059 | 0.2083 | 0.667 | 0 | 0.6067 | 0 |
| Per-form train rate (one-label targets) | 0.5721 | 0.1938 | 0.723 | 0.067 | 0.5780 | 0.0004 |
| Per-form train rate (full labels) | 0.5700 | 0.1927 | 0.723 | 0.067 | 0.5736 | 0 |

Platt: a=0.554, b=-0.028. The model is overconfident, and calibration mostly
shrinks its probabilities toward 0.5. Missing: 0%, before and after the retry.
Reliability bins: `scores/reliability-test.json`.

The fair baselines are constant 0.5, the Boolean prior and the per-form base rate.
**Zero-shot Qwen3-32B beats constant 0.5. It is slightly worse than the Boolean
prior on raw scores and about equal to it after Platt scaling. It is clearly
worse than the per-form train base rate (0.6114 vs 0.5700 on all 12 forms).**

The 20 September Qwen3-0.6B supervised arms are **not comparable** and are not
baselines to beat. Their prompts dropped `answer_text`, which leaves about 79% of
test propositions undefined. For reference only (observed forms, test): A4 0.666
(epoch-3 rescore 0.634), A5 λ=1 0.642 (0.614), A2 0.598, A3 0.595.

### Per form (test, raw)

| Form | LL | Brier | Acc | ECE |
|---|---:|---:|---:|---:|
| P | 0.619 | 0.213 | 0.675 | 0.091 |
| NOT P | 0.848 | 0.300 | 0.530 | 0.230 |
| Q | 0.671 | 0.233 | 0.644 | 0.109 |
| NOT Q | 0.861 | 0.307 | 0.520 | 0.221 |
| (P) AND (Q) | 0.400 | 0.121 | 0.847 | 0.062 |
| (P) OR (Q) | 0.731 | 0.253 | 0.611 | 0.116 |
| (P) AND (NOT Q) | 0.469 | 0.152 | 0.794 | 0.037 |
| (P) OR (NOT Q) | 0.540 | 0.175 | 0.789 | 0.103 |
| (NOT P) AND (Q) | 0.530 | 0.172 | 0.774 | 0.077 |
| (NOT P) OR (Q) | 0.502 | 0.162 | 0.804 | 0.085 |
| (NOT P) AND (NOT Q) | 0.733 | 0.260 | 0.596 | 0.148 |
| (NOT P) OR (NOT Q) | 0.433 | 0.129 | 0.848 | 0.079 |

Negation is the weak point. The model gives "resolves NO" a mean of 0.56 when the
true rate is 0.68, and P + NOT P is off from 1 by more than 0.2 in 35% of pairs.
The literal complement terms dominate the consistency violation (0.109 for P and
0.113 for Q, against a total of 0.022).

### Strata (test, raw log loss; all 0% missing)

| Stratum | Queries | Qwen3-32B | Per-form rate |
|---|---:|---:|---:|
| Both propositions binary | 2,108 | 0.609 | 0.521 |
| Both multiple-choice options | 9,452 | 0.624 | 0.588 |
| Mixed | 1,232 | 0.517 | 0.515 |
| Some proposition had creator edits stripped | 4,370 | 0.592 | 0.554 |
| No edits | 8,422 | 0.621 | 0.578 |

## Cost

Vast instance 52194743 (1×H100 SXM, Montana, $2.33/h billed) ran 1.51 h and cost
$3.79 (credit went from $68.88 to $65.09). All files were synced to General and
verified by sha256 (27 files, 0 mismatches). The instance was then destroyed, and
the API no longer lists it.
