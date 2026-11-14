# Weekly world model of Qwen3-32B over the test link graph (2026-09-27)

Question: when Qwen3-32B is asked about linked events week by week, do pairwise ("A and B") and
triple ("A and B and C") answers add anything to a world model built from its answers? We fit its
answers with a log-linear joint and compare three models: M1 (marginals only), M2 (plus a pairwise term
per link) and M3 (plus a triple term per triangle).

Short answer: no. Pairwise and triple terms do not improve held-out answer prediction or accuracy on
real outcomes. The joint answers carry one large, systematic signal: an AND bias. Stated "A and B" sits
about 0.07 above the product of the stated singles, while the true YES rate of those conjunctions is 6.6%.
Any model fitted to those answers pulls the marginals up and loses accuracy.

Code: `scripts/weekly_worldmodel.py` (plan, build, run), `scripts/weekly_worldmodel_fit.py` (tune, fit),
`scripts/weekly_worldmodel_score.py`.
Data: `/home/agent/data/weekly-worldmodel-20260927/` (queries, answer log, per-week fits,
`summary.json`, `charts.json`). Copies are in `docs/weekly-worldmodel-20260927-summary.json` and
`docs/weekly-worldmodel-20260927-charts.json`.

## Setup

- **Graph.** Luna test graph, `luna-linkgraph-20260927/test`: `edges.jsonl` plus the 2,170 second-pass
  edges in `edges-pass2.jsonl`. Every relation type is used, including related-topic. There are no
  embedding bridges, and the graph files were not modified.
- **Weeks.** All 35 Mondays from 2025-05-05 to 2025-12-29.
  - The events for a week are the test events open and unresolved at t, checked again against the
    as-of store and using the as-of question text.
  - That gives 34 to 998 events per week (mean 639) and 2,112 distinct events. 1,496 of them were asked
    in 3 or more weeks.
- **Queries.** Every open event alone, "A and B" for every open link, and "A and B and C" for every open
  triangle. In total 22,366 singles, 59,589 and2 and 48,563 and3: 130,518 calls.
- **Prompts.** v6_direct question-only, one statement per call, with the as-of line "Assume today is t".
  The model is qwen3-32b with thinking off, called through `scripts/precutoff_client.py`. Answers use
  the parser with one retry.
  - 0 answers are missing, and none are imputed.
  - Every served model id matched qwen3-32b.
- **Reuse.** 2,301 answers for week 2025-06-23 were reused from the example-week run. Their prompt
  SHA-256, model and settings are identical. Every other week was asked fresh.
- **Cost.** $52.10 in total: $7.73 for the first 6 weeks and $44.37 for the remaining 29, against Kelvin's
  $60 cap for the full run. Prices are conservative list prices ($0.7/$2.8 per M tokens), so the real
  bill is likely lower.
  - The original $10 cap would have allowed only every 6th week, so the first 6 weeks were run that way:
    05-12, 06-23, 08-04, 09-15, 10-27 and 12-08. Weekly was projected at $54.7 and biweekly at $26–28.
  - Kelvin then approved all 35 weeks.
- **Hold-out.** A random 20% of and2 and 20% of and3 queries are held out, stratified per week (seed
  20260927): 21,632 queries in total.
- **Fit.** For each week, minimise the sum of KL(stated‖implied) plus λ(ΣJ²+ΣK²) over the training
  answers, with stated values clipped to [0.01, 0.99].
  - The objective separates over the connected components of the week graph. Components with min-fill
    treewidth ≤ 12 are solved exactly with a junction tree (502 components). Larger ones use damped loopy
    BP on a factor graph (51 components, up to 685 nodes and treewidth 43).
  - Every BP fit was re-run from cold messages at the fitted parameters. All 51 converged to a residual
    below 1e-9, and none needed the heavier-damping refit.
  - A Gibbs sampler (1,000 chains) re-estimated every BP value. The mean gap on triangle P111 is about
    0.001. Two M3 components are worse: 10-13 and 10-27 have a mean gap of 0.02 and a maximum of 0.27.
    The "M3g" columns show the Gibbs values, and none of the conclusions change.
- **λ.** Chosen on an inner 20% split of the training joints for weeks 06-23 and 09-15. The rule, fixed
  in advance, was the largest λ within 5% of the best.
  - M2 and M3 got λ = 1.0, the top of the grid. Inner KL was flat from 0.3 to 1.0, and smaller λ was
    worse.
  - The fitted |J| averages 0.009 and |K| averages 0.017, so the data prefer near-independence once the
    marginals are free.
  - A sensitivity run at λ = 0.1 on the 6 sampled weeks gives the same picture. M2−M1 held-out KL is
    −0.0001, and M3−M2 is +0.0006 [−0.0002, 0.0014].
- **Extra variants M2b and M3b ("singles-anchored").** These are M2 and M3 with each stated single
  weighted 50× in the loss. The marginals stay at the model's own singles, and J/K must explain the
  joints. This tests whether the AND bias is what spoils the singles. λ is tuned the same way (0.1 and
  0.3).
- **M0.** The product of the stated singles, with no fit. It is the independence reference.
- **CIs.** Paired event-cluster bootstrap with 2,000 draws. Each draw resamples events, and a query is
  weighted by the mean multiplicity of its events, so an event's rows across all weeks move together.
  A week-block bootstrap is reported as well.

## Main table (35 weeks)

(a) Held-out answer prediction: mean KL(stated ‖ predicted) on 21,632 held-out joints.

| | M0 | M1 | M2 | M3 | M2b | M3b |
|---|---|---|---|---|---|---|
| all | 0.134 | 0.0448 | 0.0448 | 0.0452 | 0.105 | 0.106 |
| and2 | 0.122 | 0.0451 | 0.0451 | 0.0452 | 0.097 | 0.097 |
| and3 | 0.149 | 0.0444 | 0.0444 | 0.0452 | 0.116 | 0.117 |

- M2−M1 = −0.00001 [−0.00002, −0.00001].
- M3−M2 = +0.0004 [0.0002, 0.0006]; the week-block CI is [−0.0000, 0.0012]. The triple terms are slightly
  worse.
- M1−M0 = −0.090 [−0.097, −0.082].

The fitted marginals explain the joint answers much better than the stated singles do. Links and
triangles add nothing on top.

(b) Outcome log loss, with raw = the stated answer:

| | raw | M0 | M1 | M2 | M3 | M2b | M3b |
|---|---|---|---|---|---|---|---|
| singles (22,366) | 0.611 | 0.611 | 0.687 | 0.687 | 0.685 | 0.612 | 0.612 |
| all joints (108,152) | 0.383 | **0.295** | 0.339 | 0.339 | 0.339 | 0.301 | 0.301 |
| held-out joints (21,632) | 0.381 | **0.298** | 0.340 | 0.340 | 0.340 | 0.304 | 0.304 |
| and2 (59,589) | 0.421 | **0.333** | 0.388 | 0.388 | 0.387 | 0.337 | 0.337 |
| and3 (48,563) | 0.336 | **0.248** | 0.279 | 0.279 | 0.280 | 0.257 | 0.258 |

- Singles:
  - M1−raw = +0.076 [0.061, 0.091]
  - M2−M1 = −0.0004 [−0.0005, −0.0004]
  - M3−M2 = −0.0019 [−0.0026, −0.0012]
  - M2b−raw = +0.0013 [−0.0014, 0.0037]
- Held-out joints:
  - M1−raw = −0.041 [−0.047, −0.036]
  - M2−M1 = −0.0001 [−0.0001, −0.0001]
  - M3−M2 = −0.0001 [−0.0007, 0.0007]
  - M0−raw = −0.083 [−0.093, −0.072]
  - M3b−M0 = +0.006 [0.003, 0.010]

Brier gives the same ordering: singles 0.204 raw vs 0.237 M1; joints 0.113 raw, 0.087 M0 and 0.100 M1.

(c) Relation type of the link. Rows are the and2 queries; held-out KL is on the held-out and2 answers.

| relation | links asked | held-out KL M1 | M2−M1 | M3−M2 [CI] | outcome LL raw / M0 / M1 / M2b | YES rate / mean stated |
|---|---|---|---|---|---|---|
| related-topic | 34,095 | 0.039 | 0.0000 | +0.0004 [0.0003, 0.0006] | 0.453 / 0.379 / 0.433 / 0.385 | 0.079 / 0.267 |
| alternatives | 9,947 | 0.049 | 0.0000 | +0.0004 [0.0002, 0.0007] | 0.327 / 0.236 / 0.295 / 0.243 | 0.044 / 0.205 |
| cause/effect | 7,507 | 0.030 | 0.0000 | +0.0005 [0.0002, 0.0008] | 0.291 / 0.193 / 0.280 / 0.202 | 0.046 / 0.209 |
| same-quantity | 7,061 | 0.085 | 0.0000 | −0.0022 [−0.0042, −0.0005] | 0.527 / 0.388 / 0.405 / 0.374 | 0.052 / 0.290 |
| same-event-different-date | 979 | 0.043 | 0.0000 | +0.0006 [0.0002, 0.0010] | 0.476 / 0.400 / 0.466 / 0.412 | 0.073 / 0.304 |

Only same-quantity links show any structure: triple terms help a little, and there M2b beats M0.
Alternatives, which are mutually exclusive, are the clearest failure. The model says 0.21 for
"A and B", where the truth is close to 0.

## Why fitting hurts the singles: the AND bias

- Stated "A and B" averages 0.253, and the product of the two stated singles averages 0.184. The true YES
  rate is 0.066.
  - For and3 the numbers are 0.199 vs 0.122, against a true rate of 0.032.
  - 19% of and2 answers are above the smaller of the two stated singles, which is incoherent.
- M1, M2 and M3 match the joints by lifting the marginals: the mean single rises by +0.074. The singles
  are already too high (mean stated 0.40 against a YES rate of 0.18), so single log loss gets worse
  by 0.076.
- Anchoring the singles (M2b/M3b) removes that damage: singles log loss is 0.612 vs 0.611 raw. The link
  terms then absorb the AND bias as positive J, so the joints still lose to M0 by 0.006.
- The example-week findings reproduce on all 35 weeks: the fitted joint beats the raw joints, the product
  of the model's own singles beats both, and fitting worsens the singles.

## Trajectories

Log loss of single events by weeks left until the event closes, raw / M1 / M3 / M2b:

| weeks left | n | YES rate | raw | M1 | M3 | M2b |
|---|---|---|---|---|---|---|
| 0–2 | 3,795 | 0.29 | 0.733 | 0.758 | 0.758 | 0.724 |
| 2–4 | 2,839 | 0.24 | 0.761 | 0.813 | 0.814 | 0.754 |
| 4–8 | 3,975 | 0.20 | 0.620 | 0.699 | 0.698 | 0.623 |
| 8–16 | 6,038 | 0.16 | 0.574 | 0.665 | 0.658 | 0.580 |
| 16+ | 5,719 | 0.09 | 0.487 | 0.592 | 0.591 | 0.493 |

Neither the raw answers nor the fitted marginals get better as close approaches. Near close, the YES rate
is higher and the model's errors are larger. From one week to the next, 17.0% of raw answer changes move
toward the outcome and 16.5% move away; the rest are unchanged. The fitted marginals move almost every
week, 48.5% toward and 48.5% away, which is noise.

## Per week

`summary.json` → `per_week` gives, for each week: events, links, triangles, largest component,
treewidth, held-out KL and single log loss for every model, the BP residual and iteration counts, and the
Gibbs gaps. `charts.json` has the series for held-out KL by week, single log loss by week, trajectory, the
main table and the relation breakdown. M1−M0 held-out KL is negative in every week. M2−M1 is never
larger than 0.0001 in absolute value.

## Conclusion

Qwen3-32B's pairwise and triple answers add no usable joint structure beyond marginals. Their one
systematic content is an AND bias that pushes the answers toward "too likely". The best predictor of
conjunction outcomes is the independence product of its own single answers.

## Follow-up arms on the 6 sampled weeks: thinking mode and an AND-correction prompt (2026-09-30)

### PREREG (written and committed 2026-09-30 before any arm call)

Question: does Qwen3-32B in thinking mode, or a prompt that warns about the too-high "A and B" rating,
give joint (and, for thinking, single) answers with lower outcome log loss than the product-of-singles
baseline M0 (product of the base run's stated singles, same week)?

Population: the 17,531 "A and B" / "A and B and C" queries of the six sampled weeks
(`sixweek/queries.jsonl`, all resolved). One seeded permutation (seed `arms-20260930`) of these joints;
each arm takes a prefix of it, so each arm's sample is a uniform random sample and the thinking sample is
a subset of the correction sample.

Arms (code: `scripts/weekly_worldmodel_arms.py`; data: `<data>/arms/`):
- **C, AND-correction.** Same model and settings as the base run (qwen3-32b, thinking off, temperature 0,
  max_tokens 20, one retry). Prompt: the base joint prompt with this paragraph inserted before the answer
  line: "Note: forecasters tend to rate statements like this one too high. The statement needs every
  question above to resolve YES, so its probability cannot be higher than that of the least likely
  question, and unless the questions are closely linked it is close to the product of their separate
  probabilities, which is usually much lower than any one of them." Sample: the first 6,000 joints
  (about $3.5 at list price). Singles are not re-asked; C's singles are the base singles.
- **T, thinking.** qwen3-32b with `enable_thinking: true` (streamed), the base prompts with " /no_think"
  removed, temperature 0.6 and top_p 0.95 (Qwen's recommended thinking settings; it advises against greedy
  decoding in thinking mode), `thinking_budget` 4000 and max_tokens 6000 as runaway guards, one retry.
  Sample: the first N_T joints plus the single question of every event in them (same week), all in
  thinking mode.
- **Model and price.** Any served model other than `qwen3-32b` stops the run (no fallback). List price
  is $0.7/M input, $2.8/M output without thinking, and $8.4/M output with thinking.

Sizing (cost only; no outcomes are read before the analysis):
1. The pilot is the first 20 joints of the permutation and their singles in T. It stays in the sample.
2. Then C runs in full.
3. N_T is the largest prefix whose projected total spend is at most $14.0. The projection covers all
   arm calls, pilot included, at the pilot's mean cost per joint call and per single call, over the
   prefix's actual count of distinct singles.
4. N_T is cut further if T's measured throughput would not finish by 17:15Z.
5. A hard stop at $15 total covers the arms together.

Primary comparisons. Outcome log loss is clipped to [0.01, 0.99]. Each comparison is paired on the same
queries with a 95% paired event-cluster bootstrap CI (B = 2,000, the same method as above: resample
events, weight each query by the mean multiplicity of its events). A week-block CI is also reported.
- P1: joints, LL(C stated) − LL(M0) over C's sample.
- P2: joints, LL(T stated) − LL(M0) over T's sample.
- P3: singles, LL(T stated single) − LL(base stated single) over T's singles. For singles, the
  product-of-singles baseline is the base single itself. C's singles equal the base singles by
  construction, so C gets no singles test.

Decision rule: an arm beats the baseline when the CI of its difference lies wholly below 0. It is worse
when the CI lies wholly above 0, and no different otherwise.

Secondary (descriptive, same CIs):
- Each arm vs base raw joints: the effect of the intervention on the direct answers.
- T joints vs M0_T, the product of T's own singles, and M0_T − M0.
- C vs T on T's sample.
- Splits by kind (and2, and3).
- The AND bias: mean stated joint vs product of singles, the share of joints above the smaller single,
  and the YES rate.

A query with no parsed answer after the retry is dropped from every column of its comparison. The count
is reported.

Sizing record (13:52Z, before any outcome was read):
- A smoke call ($0.007) checked that DashScope accepts `thinking_budget` and `top_p`. Its answer was not
  logged or used.
- The pilot was 68 calls for $0.916, with 0 missing and served model qwen3-32b only. Mean completion
  tokens were 1,494 per single, 1,306 per and2 and 2,193 per and3. That is $0.0128, $0.0114 and $0.0189
  per call.
- C finished its 6,000 calls for $2.945.
- The rule gives N_T = 262 joints: 820 T calls in all, 558 of them singles, and a projected arms total of
  $13.97.

### Results (14:08Z)

Both arms finished with 0 missing answers; only qwen3-32b was served. The spend was $12.98 of the $15 cap
at list price:
- C: 6,000 calls for $2.95.
- T: 820 calls for $10.03, pilot included. Median completion was 1,093 tokens, no call hit the length
  cap, and 1 call needed a retry.
- Smoke call: $0.007.

The summary is in `<data>/arms/summary.json`, with a copy in `docs/weekly-worldmodel-20260927-arms-summary.json`.

Outcome log loss is paired, with the 95% event-cluster bootstrap CI and the week-block CI in brackets.

| comparison | n | arm | baseline | difference [event CI] | week-block CI | verdict |
|---|---|---|---|---|---|---|
| P1 C joints vs M0 | 6,000 | 0.229 | 0.275 | −0.046 [−0.058, −0.035] | [−0.071, −0.023] | beats |
| P2 T joints vs M0 | 262 | 0.203 | 0.226 | −0.024 [−0.045, −0.002] | [−0.041, −0.011] | beats |
| P3 T singles vs base singles | 558 | 0.484 | 0.594 | −0.110 [−0.155, −0.062] | [−0.175, −0.038] | beats |

Secondary results:
- **Against the base direct answers.** Both arms cut joint log loss sharply: C by −0.136
  [−0.149, −0.124] and T by −0.134 [−0.167, −0.103]. On T's 262 joints, C and T are level (C − T −0.003
  [−0.020, 0.015]).
- **Thinking still has the AND bias against its own singles.** The product of T's own singles (M0_T,
  0.161) beats T's stated joints by 0.042 [0.031, 0.054]. It also beats base M0 by 0.066
  [0.046, 0.087]. T's stated joints average 0.138 (and2) and 0.104 (and3), against own-single products of
  0.122 and 0.060. Of T's joints, 15% (and2) and 14% (and3) exceed T's smaller single.
- **By kind.** T − M0 is −0.033 [−0.068, 0.001] for and2 and −0.013 [−0.036, 0.008] for and3; each kind
  alone is not significant. C − M0 is −0.045 for and2 and −0.046 for and3, both significant.
- **The correction prompt over-shoots the product.** C's stated joints average 0.156 (and2) and 0.107
  (and3), slightly below the base-single product (0.179, 0.116). Only 5.0% and 1.6% of C's joints exceed
  the smaller base single. YES rates are 5.8% and 2.9%.
- **Thinking lowers the singles.** On T's sample, T's singles average 0.297 against 0.385 for the base
  singles, with a YES rate of 19.7%.

Reading:
- Both interventions beat the product-of-singles baseline on joints. The one-paragraph correction prompt
  does so at about 1/25 of the per-call cost of thinking ($0.0005 against $0.012).
- Thinking also improves the singles by 0.11 nats. The best joint forecast on T's sample is still the
  product of the thinking singles, 0.161, against 0.200 for C and 0.203 for T. So what thinking adds to
  joints comes through better singles, not through better joint reasoning.
- T's sample is small (262 joints, 448 events), so its joint CIs are wide.

## Confirmatory run of the AND-correction prompt on held-out events (2026-09-30)

### PREREG (written and committed 2026-09-30 before any confirmatory call)

The correction prompt was designed from the AND bias seen in the 35-week test run, and it was first
scored on six of those weeks. This run checks it on events that were never used to see or design it.

**Held-out data.** The data is the train split of the same Luna link graph
(`/home/agent/data/luna-linkgraph-20260927/train`, 5,413 events). No weekly world-model query has ever
been asked on it. It is disjoint from the test split:
- 0 shared event ids;
- 0 shared parent markets;
- 0 shared question texts;
- 0 edges into the test split.

The validation split is excluded, because earlier prompt variants (v7b, v7c) were iterated on it.

**No newer weeks.** Our data has no weeks newer than the 35. The graph ends in December 2025; the last
train event ends on 2025-12-14. Newer weeks would need a new market snapshot and a new Luna graph.

**Calendar overlap caveat.** The train events come from the same calendar period (May–Dec 2025) as the
test weeks. They are different markets, so this is a held-out-events check, not a held-out-period check.
Every train event in these weeks closes after the Qwen3 cutoff (week of 2025-05-05 or later), the same
rule as for the test split.

**Weeks.** Five Mondays chosen by rule before any answer: stride 6, offset 4 of the train split's 32
Mondays. That gives 2025-06-02, 07-14, 08-25, 10-06 and 11-17, which fall between the weeks of the first
arms run. They hold 3,743 events and about 17,100 joints.

The build uses the same code and design as the test run: `weekly_worldmodel.py build --graph .../train`.

**Sample.** One seeded permutation (seed `confirm-train-20260930`) of those weeks' joints; the sample is
its first 6,000.

**Calls.** All calls use qwen3-32b, thinking off, temperature 0, max_tokens 20, one retry. There is no
fallback: any other served model stops the run. List price is $0.7/M in and $2.8/M out.
- **C:** the frozen correction prompt, byte-identical to the first run (CORRECTION sha256 `31e15164…4875c`).
- **R:** the base prompt on the same 6,000 joints.
- **S:** the base single question of every event in those joints.

**Budget.** Estimate about $6.5. Cap $15, with a hard stop in the runner. Rows go in
BUDGET-24H-20260929.md.

**Power.** The first run's C − M0 difference had an event-cluster bootstrap SE of 0.0060 at 6,000
joints. With that SE and two-sided α = 0.05, power is:
- 97% for half the first effect (−0.023);
- 70% for a third of it (−0.015).

The train weeks have about 1.2 times as many events per week as the test weeks (749 vs 608), so the SE
should not be larger. Subsampling joints within the same events (1,000 to 6,000) shows the SE flattening, from 0.0084
to 0.0060, because events, not joints, limit it. So 6,000 joints over the five weeks is the chosen size.

**Primary.** Joint outcome log loss (clipped to [0.01, 0.99]), LL(C) − LL(M0), where M0 is the product of
the S singles. The CI is the paired 95% event-cluster bootstrap (B = 2,000), the same method as before.
- Confirmed: the CI lies wholly below 0.
- Reversed: the CI lies wholly above 0.
- Not confirmed: otherwise.

A week-block CI (5 weeks) is also reported; it is descriptive only.

**Secondary.**
- C − R, R − M0 (does the AND bias replicate?).
- Splits by kind.
- Mean stated joint vs product of singles, the share of joints above the smaller single, and the YES rate.

A query with no parsed answer after the retry is dropped from every column. The count is reported.

### Results (14:48Z)

The run finished with 0 missing answers and served qwen3-32b only. It made 15,059 calls:
- S: 3,059 calls, $0.75.
- C: 6,000 calls, $3.13.
- R: 6,000 calls, $2.85.

Total spend was $6.73 of the $15 cap. The sample covers 2,088 events. The summary is in
`<data>/confirm-train/arms/summary.json`, with a copy in `docs/weekly-worldmodel-20260927-confirm-summary.json`.

| joint log loss | n | C | M0 | R | C − M0 [event CI] | week-block CI |
|---|---|---|---|---|---|---|
| all joints | 6,000 | 0.434 | 0.496 | 0.542 | **−0.062 [−0.078, −0.046]** | [−0.081, −0.042] |
| and2 | 3,071 | 0.495 | 0.572 | 0.643 | −0.077 [−0.098, −0.057] | |
| and3 | 2,929 | 0.370 | 0.416 | 0.437 | −0.046 [−0.062, −0.030] | |

**Primary: confirmed.** On held-out events, the frozen correction prompt beats the product of singles.

Secondary results:
- **Correction vs base prompt.** C − R is −0.108 [−0.126, −0.090].
- **The AND bias replicates.** R − M0 is +0.046 [+0.027, +0.064], and 23% of base and2 answers exceed
  the smaller single.
- **Base rates differ from the test split.** The train-split conjunctions resolve YES far more often:
  20.0% (and2) and 13.7% (and3), against 5.8% and 2.9% on the test sample.
- **Stated values vs outcomes.** For and2, C states 0.189, the product of singles is 0.266 and R states
  0.311. For and3, the figures are 0.112, 0.157 and 0.196. C's stated values sit below both the
  product and the YES rate, yet C still has the lowest log loss.

Caveat: these are held-out events from the same calendar period as the test run, not a later period.
