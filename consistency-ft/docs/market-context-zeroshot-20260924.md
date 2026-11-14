# Market-context zero-shot test, one form per call (24 Sep 2026)

Design: `docs/conditional-joint-design-20260923.md`. The model states its own
probability for each of the 12 forms of a pair, one form per call. Market prices are
evidence in the prompt, not constraints. Consistency is internal. No training and no tuning.

## Result in one paragraph

On the 723 test pairs, **no model × arm beats "market prices + independence" on the joint
forms**:
- **Raw:** 0.13 to 0.25 log loss worse.
- **Projected to coherent:** 0.04 to 0.18 worse.
- **Hybrid:** market marginals joined with the model's own odds ratio. It ties the market
  to within ±0.003 on every arm and stratum (for 235B L, −0.0001 [−0.003, +0.002]).

So the model's dependence knowledge adds nothing that shows up here.

Other findings:
- **Context helps.** It helps a lot once the answers are made coherent: 32B projected
  joints go from Q 0.488 to P 0.363 and L 0.386. P beats L by 0.022 [0.013, 0.032].
- **235B beats 32B** with the same Luna context (projected −0.036 [−0.046, −0.026]).
- **Singles:** every arm is worse than the price at t. The best is 235B L raw, +0.024
  [+0.013, +0.037].
- **Projection is the largest gain** on joints (−0.05 to −0.18). With context, it slightly
  hurts the singles.

## Setup

- **Data:** `/home/agent/data/jev-market-context-open-20260923`, 723 test pairs (320
  same-topic, 403 other) and 332 validation pairs. Both questions are open at t, where t =
  00:00 UTC of (earlier closing week − 7 days). Outcomes (`pair-outcomes.jsonl`,
  `full-label-targets.jsonl`) were used only for scoring.
- **Arms.** The input is identical across the 12 forms of a pair and across models.
  - **Q:** question only.
  - **P:** a deterministic evidence pack, `training/asof.py build_pack(pair, t, "small")`.
  - **L:** the Luna context-agent output,
    `/home/agent/data/manifold-asof-20260923/context-agent/{split}.jsonl`, field `context`.
    Its t matches the pack's t on all 1,055 pairs (asserted).
- **Pack size: the 2,000-token "small" budget (Qwen3 tokenizer), for every model.** Reasons:
  - It is the layout that `build_pack` was designed and leak-tested at.
  - It matches the size of the Luna context (median 1,454 tokens, p90 2,364 on test), so
    P and L differ in selection, not in volume.
  - It keeps the 32B arms at about $3 per 8,676 test calls and leaves room for the 235B arm.
  - Packs measured 1,550 to 2,000 tokens.
- **Prompt:** v6_direct wording (`training/verbalized_baseline.py`):
  - system prompt `BASE_SYSTEM`;
  - "Assume today is t …";
  - the context block (P and L only);
  - `Question A` and `Question B` blocks;
  - "Statement to assess: …";
  - `ANSWER_DIRECT` and `/no_think`.

  Two deliberate differences from v6_direct:
  1. Both question blocks are shown for every form, so that the input is the same across
     the 12 forms. v6_direct showed only the relevant question for P, NOT P, Q and NOT Q.
  2. The question blocks use the as-of-t question and criteria text
     (`question_block(context="asof_text")`), not the export text, so nothing written
     after t enters the prompt.

  Median prompt tokens on test: Q 485, P 2,250, L 1,969.
- **Models:** DashScope intl. `qwen3-32b` on Q, P and L; `qwen3-235b-a22b` on L. Both ran
  with thinking off, temperature 0, `max_tokens` 20 and one retry with `RETRY_REMINDER`.
  Of 46,896 calls, 46,841 were answered on the first try, and every one reported the
  requested model. The other 55 were content-filter refusals, and 13 of them were answered
  on retry. No other model was used.
- **235B Q was not run.** It would have cost about $3.2, and the total would then have
  passed $25.
- **Scoring:** `scripts/score_market_context_zeroshot.py`.
  - Probabilities are clipped to [0.01, 0.99], and a missing answer counts as 0.5.
  - **Projected:** the KL fit over the 4 atoms (`scripts/coherence_projection.py`).
  - **Hybrid:** market p(P) and p(Q) at t, joined with the log odds ratio of the model's
    projected atoms (solved by bisection). Its singles are the market.
  - **Baseline:** market independence, the price at t clipped to 0.01.
  - **CIs:** paired bootstrap over pairs, 2,000 resamples.
  - **Joint forms:** the 8 AND/OR forms including the signed variants.

## Missing answers (test)

| model | arm | calls | unparsed at first try | missing after retry | cost |
|---|---|---|---|---|---|
| qwen3-32b | Q | 8,676 | 0 | 0 | $0.80 |
| qwen3-32b | P | 8,676 | 0 | 0 | $3.12 |
| qwen3-32b | L | 8,676 | 28 | 22 (0.25%) | $2.91 |
| qwen3-235b-a22b | L | 8,676 | 27 | 20 (0.23%) | $12.73 |

Every missing answer is a DashScope content-filter refusal (HTTP 400
`data_inspection_failed`) on the same 3 pairs' Luna context. There were no parse failures.

Validation: 0 missing answers out of 3 × 3,984 calls for 32B and 0 out of 240 for 235B L.
For 235B, validation was a 20-pair parse check to save budget.

## Main result: joint forms, test (723 pairs)

Market independence on these forms scores log loss 0.313 and Brier 0.098. Δ is model minus
market, so negative means the model is better.

| model | arm | raw LL | Δ raw [95% CI] | projected LL | Δ projected [95% CI] | hybrid LL | Δ hybrid [95% CI] |
|---|---|---|---|---|---|---|---|
| 32B | Q | 0.537 | +0.225 [+0.204, +0.244] | 0.488 | +0.175 [+0.156, +0.194] | 0.313 | +0.000 [−0.002, +0.002] |
| 32B | P | 0.528 | +0.215 [+0.191, +0.238] | 0.363 | +0.050 [+0.040, +0.060] | 0.313 | +0.000 [−0.002, +0.003] |
| 32B | L | 0.565 | +0.252 [+0.227, +0.278] | 0.386 | +0.073 [+0.061, +0.085] | 0.315 | +0.002 [−0.001, +0.005] |
| 235B | L | 0.447 | +0.135 [+0.117, +0.152] | 0.349 | +0.037 [+0.027, +0.045] | 0.313 | −0.000 [−0.003, +0.002] |

The Brier differences point the same way, for example 235B L projected +0.010 [+0.007, +0.014].

**Why the hybrid is a tie.** The hybrid moves the joint forms by only 0.014 on average
(p90 0.04) away from independence, for two reasons:
- The models' odds ratios are modest.
- 38% of pairs have a price at or below 5% (or at or above 95%) on one of the questions,
  which leaves little room for dependence to matter.

The models also barely separate related pairs from unrelated ones. The mean log odds ratio
of the projected atoms is:

| model and arm | same topic | other |
|---|---|---|
| 235B L | +0.84 | +0.72 |
| 32B L | +0.36 | +0.26 |
| 32B P | +0.22 | +0.20 |
| 32B Q | +0.17 | +0.12 |

The pooled outcome log odds ratio on test is +0.83 for same-topic pairs and +0.21 for
other pairs. It includes base-rate heterogeneity, so it is only a rough reference.

## Single forms vs the market price at t (test)

The market scores log loss 0.359 on these forms.

| model | arm | raw Δ [95% CI] | projected Δ [95% CI] |
|---|---|---|---|
| 32B | Q | +0.358 [+0.325, +0.390] | +0.229 [+0.204, +0.254] |
| 32B | P | +0.047 [+0.032, +0.063] | +0.063 [+0.050, +0.075] |
| 32B | L | +0.093 [+0.071, +0.116] | +0.090 [+0.075, +0.105] |
| 235B | L | +0.024 [+0.013, +0.037] | +0.047 [+0.036, +0.058] |

With context the models mostly read the price, but they do not add to it. Projection pulls
the singles toward the noisier joint answers, so with context it hurts them slightly.

On all 12 forms the market scores 0.328. The best result is 235B L projected, at 0.368.

## Strata (test, joint forms, Δ vs market)

| model | arm | same topic: raw / projected / hybrid | other: raw / projected / hybrid |
|---|---|---|---|
| 32B | Q | +0.238 / +0.196 / −0.000 | +0.214 / +0.159 / +0.000 |
| 32B | P | +0.219 / +0.052 / +0.002 | +0.211 / +0.049 / −0.001 |
| 32B | L | +0.228 / +0.063 / +0.001 | +0.272 / +0.081 / +0.003 |
| 235B | L | +0.113 / +0.025 / +0.001 | +0.152 / +0.046 / −0.001 |

The market joint log loss is 0.313 in both strata. Every hybrid CI contains 0, and every raw
and projected CI is above 0. The full CIs are in `tables.md`.

## Paired comparisons (test, joint forms, x − y)

| comparison | raw | projected | hybrid |
|---|---|---|---|
| projected − raw, 32B Q | | −0.050 [−0.057, −0.042] | |
| projected − raw, 32B P | | −0.164 [−0.180, −0.149] | |
| projected − raw, 32B L | | −0.180 [−0.197, −0.163] | |
| projected − raw, 235B L | | −0.098 [−0.108, −0.087] | |
| 32B L − P | +0.038 [+0.019, +0.058] | +0.022 [+0.013, +0.032] | +0.002 [−0.002, +0.005] |
| 32B L − Q | +0.028 [+0.002, +0.052] | −0.102 [−0.119, −0.085] | +0.002 [−0.001, +0.005] |
| 32B P − Q | −0.010 [−0.032, +0.012] | −0.125 [−0.140, −0.108] | +0.000 [−0.002, +0.003] |
| 235B L − 32B L | −0.118 [−0.139, −0.096] | −0.036 [−0.046, −0.026] | −0.002 [−0.006, +0.001] |

- **Raw:** context does not help. P and Q are equal, and L is worse than Q. The one-form
  answers with context are individually noisy.
- **Projected:** context is worth 0.10 to 0.13, and the deterministic pack beats the Luna
  context by 0.022 on joints. The pack beats Luna in the "other" stratum (+0.032
  [+0.018, +0.046]) but not in the same-topic stratum (+0.010 [−0.002, +0.023]).

## Validation (reference only; nothing was tuned on it)

Joint Δ vs market (0.450) on 332 pairs:

| model | arm | raw | projected | hybrid |
|---|---|---|---|---|
| 32B | Q | +0.164 | +0.109 | +0.004 [−0.000, +0.009] |
| 32B | P | +0.187 | +0.049 | +0.007 [+0.001, +0.013] |
| 32B | L | +0.224 | +0.071 | +0.003 [−0.003, +0.008] |

235B L on the 20-pair check: +0.118, +0.037 and +0.004. The pattern is the same as on test.

## Caveats

- **Test started early.** The first test launch fired while one 32B validation job had
  stopped on a transient DashScope 429 `insufficient_quota`. It was stopped after 104 32B Q
  and 200 235B L test calls. Those records were kept: they use the same prompts at
  temperature 0 and informed no decision. Validation was then completed and gated (every
  job complete, missing ≤ 5%) before test resumed. The runner now waits and retries on that
  refusal.
- **Missing answers count as 0.5 in raw scoring.** This covers 3 pairs in the L arms. The
  projection is fitted on the forms that were answered.
- **The hybrid uses the model's projected odds ratio exactly**, with no shrinkage and no
  fitting on outcomes.

## Cost

All costs are at the Model Studio Singapore list price:
- qwen3-32b: $0.16 per million input tokens and $0.64 per million output tokens;
- qwen3-235b-a22b: $0.70 per million input tokens and $2.80 per million output tokens.

The total was **$23.86**, under the $25 cap:

| split | model and arms | cost |
|---|---|---|
| validation | 32B, three arms | $3.93 |
| validation | 235B L, 20 pairs | $0.37 |
| test | 32B Q, P and L | $6.83 |
| test | 235B L | $12.73 |

The runner projects the cost of each job before it starts and refuses the job if the ledger
plus the projection would pass the cap.

## Files

- Code:
  - `scripts/market_context_zeroshot.py` builds the packs and prompts, projects the cost and
    runs the jobs.
  - `scripts/score_market_context_zeroshot.py` scores the runs and writes the tables and
    chart JSON.
- Data in `/home/agent/data/market-context-zeroshot-20260924/`:
  - `packs-{split}.jsonl`;
  - `prompts-{split}-{Q,P,L}.jsonl`;
  - `logs/{model}-{arm}-{split}.jsonl` (raw replies, usage, cost);
  - `scores.json`;
  - `tables.md` (every table with CIs, including validation);
  - `run-*.log`.
- Chart JSON: `docs/market-context-zeroshot-20260924-charts.json`, with a row for each split
  × model × arm × method × form group × stratum, and the paired comparisons.
