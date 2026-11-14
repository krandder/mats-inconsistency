# Does the coherence gain grow with a larger world model? (2026-09-24)

Question (Kelvin, approved 24 Sep): the "adjust to be consistent" projection gained about
0.07 log loss on pairs (docs/coherence-projection-20260924.md, Qwen3-32B, 12 forms).
Does that gain grow when the answers cover clusters of n = 2, 3, 5 or 8 related events?

**Answer: no.** Per query, the gain falls from −0.092 (n=2) to −0.058 (n=8) log loss.
On the single-event forecasts, which are the same target at every n, it stays flat at
−0.06 to −0.07. If negations are left out, the gain moves only from −0.041 to −0.051 and
the CIs overlap. The gain comes from real logical structure: when the formulas are
permuted within a cluster, projection makes the forecasts worse than raw.

## Setup

- Model: Qwen3-32B, served by DashScope (`qwen3-32b`, thinking off, temperature 0,
  max_tokens 20). Every call was served by `qwen3-32b`, with no substitution. 8,100 calls,
  0 retries, 0 missing.
- Prompt: v6_direct style (`training/verbalized_baseline.py`). It has the system prompt
  BASE_SYSTEM and the line "Assume today is t". For each question in the query it shows
  the question, the option and the resolution criteria, with creator edits stripped and
  a 1,500-character cap. Then comes the statement, ANSWER_DIRECT and " /no_think". One
  query per call, parsed from the "Probability: 0.xx" line (one retry on a parse
  failure). No market data. The question text and criteria are the **as-of-t** versions
  from the Manifold store (`training/asof.py`).
- Leakage: only test-split parents of jev-disjoint-v1-filtered-20260919 (the
  train/validation disjointness is asserted). The 178 excluded parents are removed.
  Every event closes in the week of 2025-05-05 or later. Every event is open at a common
  t = (earliest closing week − 7 days) 00:00 UTC: created before t, not resolved and no
  resolution logged at or before t. Every event resolved YES or NO.
- Clusters: 50 per type and n, 400 in all.
  - `mc`: n options of one multiple-choice market (80 parents). None of the test
    multiple-choice parents is sum-to-one, and none says only/exactly/at most one in its
    text. So no exclusivity structure was imposed.
  - `group`: n binary markets that share a Manifold group (368 markets). Specific groups
    come first, then "-default" topics. The earliest-closing market sets t, and the other
    members are those with the nearest closing weeks. A market is used in at most 2
    clusters per n.
  - YES rate: mc 0.31–0.37; group 0.07–0.14.
- Queries per cluster (the formula of each is recorded):

  | kind | n=2 | n=3 | n=5 | n=8 |
  |---|---|---|---|---|
  | singles | 2 | 3 | 5 | 8 |
  | pairwise ANDs (n=8: random 12 of 28) | 1 | 3 | 10 | 12 |
  | conditionals "Suppose A resolves YES. What is the probability that B resolves YES?" | 2 | 3 | 5 | 8 |
  | 3-way AND + OR (n=2: the pairwise OR) | 1 | 2 | 4 | 4 |
  | negations | 2 | 2 | 2 | 2 |
  | total | 8 | 13 | 26 | 34 |

  The query budget is the same for each cluster type: 400, 650, 1,300 and 1,700 calls for
  n = 2, 3, 5 and 8.
- Projection (`scripts/worldmodel_projection.py`): fit q over the 2^n atoms, with q =
  softmax(z), L-BFGS with analytic gradients, two starts, and a 1e-4 max-entropy
  tie-break. A conditional is implied as a ratio. Two fits: the sum of Bernoulli
  KL(stated‖implied) (`kl`) and least squares in log-odds (`ls_logit`). The fit never
  sees labels.
- Scoring: log loss and Brier (clip 0.01) on every query whose truth value is known.
  Conditionals count only where the condition resolved YES. CIs are 95% paired bootstraps
  over clusters (2,000 resamples).

## Raw vs projected (log loss per query, all queries; gain = projected − raw, negative = better)

| type | n | raw | kl | gain kl (95% CI) | ls_logit gain (95% CI) | Brier raw → kl |
|---|---|---|---|---|---|---|
| all | 2 | 0.679 | 0.587 | −0.092 [−0.124, −0.067] | −0.087 [−0.119, −0.058] | 0.231 → 0.200 |
| all | 3 | 0.654 | 0.580 | −0.073 [−0.106, −0.049] | −0.074 [−0.105, −0.049] | 0.219 → 0.196 |
| all | 5 | 0.627 | 0.559 | −0.068 [−0.096, −0.046] | −0.061 [−0.084, −0.043] | 0.203 → 0.184 |
| all | 8 | 0.629 | 0.570 | −0.058 [−0.084, −0.039] | −0.052 [−0.075, −0.034] | 0.204 → 0.189 |
| mc | 2 | 0.709 | 0.614 | −0.096 [−0.146, −0.052] | −0.080 [−0.139, −0.029] | 0.236 → 0.209 |
| mc | 3 | 0.750 | 0.670 | −0.081 [−0.139, −0.038] | −0.080 [−0.143, −0.035] | 0.251 → 0.228 |
| mc | 5 | 0.756 | 0.665 | −0.091 [−0.142, −0.050] | −0.077 [−0.123, −0.043] | 0.243 → 0.222 |
| mc | 8 | 0.754 | 0.683 | −0.071 [−0.118, −0.036] | −0.060 [−0.103, −0.027] | 0.244 → 0.229 |
| group | 2 | 0.646 | 0.558 | −0.088 [−0.113, −0.066] | −0.096 [−0.122, −0.071] | 0.225 → 0.190 |
| group | 3 | 0.551 | 0.485 | −0.066 [−0.086, −0.047] | −0.066 [−0.086, −0.049] | 0.186 → 0.162 |
| group | 5 | 0.493 | 0.449 | −0.044 [−0.053, −0.035] | −0.043 [−0.053, −0.035] | 0.161 → 0.145 |
| group | 8 | 0.494 | 0.450 | −0.044 [−0.055, −0.034] | −0.042 [−0.054, −0.034] | 0.161 → 0.145 |

"Gain per query" in this table is the loss reduction per query, averaged over all queries.
The query mix changes with n: negations are 25% of the queries at n=2 and 6% at n=8.
Negations are also where projection gains most (kl gain −0.15 to −0.26 per negation
query at every n). Leaving negations out, the kl gain is −0.041 [−0.085, −0.001], −0.048,
−0.049 and −0.051 [−0.078, −0.029] for n = 2, 3, 5 and 8. By type, mc is −0.094 to −0.077
with no trend, and group is +0.019 to −0.023, noisy.

Gain by query kind (all types, kl), for n = 2 / 3 / 5 / 8:

| kind | gain by n |
|---|---|
| single | −0.074 / −0.062 / −0.056 / −0.073 |
| pairwise AND | −0.116 / −0.082 / −0.066 / −0.063 |
| conditional | +0.015 / +0.010 / +0.035 / −0.013 (all CIs cross 0) |
| negation | −0.21 / −0.18 / −0.26 / −0.15 |
| OR (n=2 pair, else 3-way) | +0.075 / +0.084 / +0.007 / +0.026 |
| 3-way AND | — / −0.072 / −0.044 / −0.005 |

## Query-count curve (kl; fit on nested query sets, gain on the singles, which are the same target at every n)

| n | singles only | + pairwise ANDs | + conditionals | + 3-way/OR | + negations (all) | singles + negations only |
|---|---|---|---|---|---|---|
| 2 | 0 (2 q) | −0.031 (3) | −0.030 (5) | −0.088 (6) | −0.074 (8) | −0.009 (4) |
| 3 | 0 (3) | −0.014 (6) | −0.009 (9) | −0.059 (11) | −0.062 (13) | −0.000 (5) |
| 5 | 0 (5) | −0.030 (15) | −0.037 (20) | −0.068 (24) | −0.056 (26) | +0.015 (7) |
| 8 | 0 (8) | −0.043 (20) | −0.044 (28) | −0.065 (32) | −0.073 (34) | −0.014 (10) |

(q) = queries in the fit per cluster. A singles-only fit reproduces the stated singles, so
its gain is 0. The ORs add the largest step. The mean single forecast does not move
(0.431 raw, 0.434 projected, against a YES rate of 0.234), so the gain comes from
ranking the events within a cluster, not from a shift in level. Conditionals add almost
nothing. Negations
alone barely move the singles. More queries per event, from a larger n, do not raise the
final gain on the singles. Scored on all queries (queries outside the fit are read from
the fitted q), the final gains are the table above. Every stage, target and type is in
`docs/worldmodel-clusters-20260924-charts.json` (`query_count_curve`).

## Shuffled-structure control (formulas permuted within the cluster, 5 permutations, all queries)

| n | kl gain, all | kl gain, mc | kl gain, group |
|---|---|---|---|
| 2 | +0.045 [−0.009, +0.099] | +0.015 | +0.079 [+0.011, +0.146] |
| 3 | +0.052 [−0.007, +0.107] | −0.010 | +0.119 [+0.064, +0.169] |
| 5 | +0.023 [−0.023, +0.065] | −0.035 | +0.084 [+0.057, +0.111] |
| 8 | +0.012 [−0.023, +0.043] | −0.029 | +0.056 [+0.033, +0.079] |

With the wrong structure, projection is neutral to harmful. The mc control is near 0,
because options of one market get similar answers, so any shrinkage among them helps a
little. The real-structure gain is 0.04–0.18 better than the control in every type and n. The
real formulas carry the gain.

## Caveats

- The group clusters are mostly NO (7–14% YES), so their raw loss is low, and their
  "relatedness" is shared group membership only.
- mc parents repeat across n (80 parents for 200 clusters; each is used at most once per
  n). CIs are over clusters, not parents.
- Per query the cap price (0.7/2.8 $/M) is conservative. Third-party listings give
  0.16/0.64 $/M for qwen3-32b in the international region, which would make the cost
  about $0.65.

## Cost and files

8,100 calls; 3.83M prompt tokens and 57k output tokens; **$2.84** at the conservative cap
price (cap $10); 8 workers.
Code: `scripts/worldmodel_clusters.py` (build and run), `scripts/worldmodel_projection.py`
(fits, scores, bootstrap) and `scripts/worldmodel_charts.py` (chart JSON).
Data: `/home/agent/data/worldmodel-clusters-20260924/` (clusters.jsonl, queries.jsonl
with prompts and formulas, log.jsonl with raw replies, projections.jsonl, and copies of
both JSONs).
Numbers: `docs/worldmodel-clusters-20260924.json` (full) and
`docs/worldmodel-clusters-20260924-charts.json` (chart-ready).
