# Structured pairs: do models capture BUILT-IN dependence? (24 Sep 2026)

Follow-up to `docs/market-context-zeroshot-20260924.md` (on pairs matched by similar wording, the
models add nothing over market prices + independence) and `docs/worldmodel-global-20260924.md`
(Qwen3-32B's global world model has almost no real pairwise dependence). Here the dependence is
built into the pair: ladders (A implies B) and options of one single-answer market (exclusive).

## Result in one paragraph

**Yes, partly.** On structured pairs the model's dependence is real and useful:
- **Hybrid.** Market marginals + the model's projected odds ratio beats market prices +
  independence on the joint forms for every structured type, on every full-set run (32B Q,
  32B P, 235B Q): −0.010 to −0.038 log loss, all CIs below 0. It ties on the unrelated controls.
- **Only part of the known relation's value.** The hybrid recovers about 20% to 65% of what the
  **known** relation is worth. Market prices + the known relation score −0.039 on threshold,
  −0.056 on date and −0.059 on exclusive pairs.
- **The models' own answers still lose to the market.** Raw loses everywhere (+0.03 to +0.38).
  Projected loses or at best ties (−0.02 to +0.22).
- **Projection is the bottleneck.** The raw one-form answers respect the relations well: the
  order is right on 87–99% of implication pairs, and P(A and B) ≤ 0.05 on 94–96% of exclusive
  pairs. The KL coherence projection washes most of that structure out.
- **The market is almost always coherent.** At t it violates an implication on 7 of 300 threshold
  and 10 of 180 date pairs, and only 3 + 3 by more than 0.02. So there is little room for a
  coherent model to beat the market by fixing its incoherence.

## Pair set

`scripts/structured_pairs.py build` (rules in `scripts/structured_pairs_rules.py`, LLM-free).
Universe: public Manifold binary markets and multiple-choice options in the as-of store
(`/home/agent/data/manifold-asof-20260923/asof.sqlite`), created on or after 2025-04-28 (the
JEV pool's first week, after the Qwen3 release), resolved YES/NO. Parents: never a
train/validation parent of `jev-pairings-v1`, `jev-disjoint-v1(-filtered)` or the open
market-context set, never one of the 178 excluded parents. 987 of 1,077 pairs use only unused
parents; the rest touch test parents. t = 00:00 UTC of (earlier closing week − 7 days), closing
week from min(close, resolution time). Both questions open at t (created before t, resolution
after t, nothing logged before t), a price at t, ≥ 5 traders. Text shown to the model is the
as-of-t text; every ladder pair is re-checked on the as-of-t text with the same rule and dropped
if the relation changes. Selection never reads outcomes. P/Q order is a seeded coin flip.

| type | rule | relation | pairs | groups | available |
|---|---|---|---|---|---|
| threshold | same template, one number, comparator up (above/≥/+/at least/…) or down (below/≤/or lower/…) | stricter rung implies looser | 300 | 168 | 454 candidate pairs, capped at 300 |
| date | same template, one date after by/before (or a bare-date option under "… by …?") | earlier implies later | 180 | 72 | 180 (all) |
| mc_excl | the two highest-priced options at t of a single-answer MC market resolved to one option | mutually exclusive | 300 | 300 markets | 3,842 markets |
| control | random pairs, different parents, no shared group at t | none | 297 | 297 | — |
| explicit (win vs reach final) | not built | — | 0 | — | no cheap rule with a checked direction |

- At most 3 pairs per ladder group, a proposition in at most 2 pairs of a type. 437 of the 480
  ladder pairs are options of one independent-answer MC market; 43 are separate binary markets.
  Rejected: negations and persistence words (not, still, remain, until, …), ranges,
  self-referential markets ("this market"), and "hit/reach $X" price ladders (direction ambiguous).
- **Rule validity on outcomes** (checked after the build, not used for selection): 0 of 780
  structured pairs violate their relation (no A-YES/B-NO for implications, no double YES for
  exclusives).
- Base rates: both YES in 35% of threshold, 47% of date and 8% of control pairs.
- Median t: threshold 2026-05-18, date 2026-04-13, mc_excl and control 2026-02-16.

## Market check at t

| type | pairs | violations | > 0.02 | > 0.05 | mean excess when violated |
|---|---|---|---|---|---|
| threshold: price(A) > price(B) | 300 | 7 (2.3%) | 3 | 2 | 0.049 |
| date: price(A) > price(B) | 180 | 10 (5.6%) | 3 | 2 | 0.045 |
| mc_excl: price(A) + price(B) > 1 | 300 | 75 raw (25%) | 53 > 1.02 | | |

The mc_excl "violations" are a **store artifact**, not market incoherence. The store has no
arbitrage fills for sum-to-one markets, so each option's last-bet price is stale (the open
options' prices sum to a median of 1.05). Real Manifold sum-to-one prices cannot violate
exclusivity. The mc_excl market baselines therefore use prices normalized over the open options
at t. With the raw prices, market independence scores 0.368 and market + known relation 0.323,
against 0.377 and 0.318 normalized. The evidence packs show the raw stale prices, as in the
main test.

## Runs

Same prompt layout as the market-context zero-shot test (v6_direct, both question blocks with
the as-of-t text, one form per call, 12 forms). DashScope, thinking off, temperature 0,
`max_tokens` 20, one retry. Arms: Q (question only) and P (`build_pack(pair, t, "small")`, the
main test's 2,000-token pack). Every call reported the requested model and none was missing.

| model | arm | pairs | calls | cost |
|---|---|---|---|---|
| qwen3-32b | Q | 1,077 | 12,924 | $1.20 |
| qwen3-32b | P | 1,077 | 12,924 | $4.50 |
| qwen3-235b-a22b | Q | 1,077 | 12,924 | $5.24 |
| qwen3-235b-a22b | P | 170 (seeded random subset) | 2,040 | $3.16 |

The total was **$14.09**, under the $15 cap. The full 235B P run was dropped because it would have
cost about $21: DashScope counted prompt tokens about 20% above the local projection. The
leftover budget bought the 170-pair subset (50 threshold, 27 date, 49 mc_excl, 44 control).

## Main table: joint forms, log loss, Δ vs market + independence

Δ is method minus market + independence (negative = better). CIs are a paired cluster bootstrap
over groups (ladder group, MC market, control pair), 2,000 resamples.
"Known" = market prices + the known relation: P(A and B) = min(a, b) for implication (upper
Fréchet bound), max(0, a + b − 1) for exclusivity, independence for controls.

| type | market indep | market + known | model | raw Δ | projected Δ | hybrid Δ [95% CI] | hybrid − known |
|---|---|---|---|---|---|---|---|
| threshold (300) | 0.291 | 0.252 (−0.039 [−0.053, −0.024]) | 32B Q | +0.378 | +0.177 | −0.015 [−0.022, −0.007] | +0.024 |
| | | | 32B P | +0.202 | +0.042 | −0.011 [−0.017, −0.005] | +0.028 |
| | | | 235B Q | +0.239 | +0.149 | −0.024 [−0.031, −0.018] | +0.015 |
| date (180) | 0.338 | 0.282 (−0.056 [−0.074, −0.037]) | 32B Q | +0.345 | +0.220 | −0.018 [−0.027, −0.010] | +0.038 |
| | | | 32B P | +0.110 | +0.017 | −0.010 [−0.018, −0.002] | +0.046 |
| | | | 235B Q | +0.265 | +0.176 | −0.032 [−0.044, −0.022] | +0.023 |
| mc_excl (300) | 0.377 | 0.318 (−0.059 [−0.071, −0.045]) | 32B Q | +0.296 | +0.136 | −0.038 [−0.047, −0.028] | +0.021 |
| | | | 32B P | +0.156 | −0.000 | −0.035 [−0.042, −0.027] | +0.024 |
| | | | 235B Q | +0.240 | +0.133 | −0.030 [−0.037, −0.023] | +0.029 |
| control (297) | 0.304 | 0.304 | 32B Q | +0.237 | +0.185 | −0.000 [−0.002, +0.001] | −0.000 |
| | | | 32B P | +0.174 | +0.043 | −0.001 [−0.004, +0.001] | −0.001 |
| | | | 235B Q | +0.210 | +0.171 | −0.001 [−0.003, +0.000] | −0.001 |

235B P, 170-pair subset. On these pairs market independence scores 0.307 and market + known
relation 0.264:

| | raw Δ | projected Δ | hybrid Δ | hybrid − known |
|---|---|---|---|---|
| all | +0.087 [+0.061, +0.117] | +0.010 [−0.005, +0.024] | −0.020 [−0.027, −0.013] | +0.023 [+0.014, +0.033] |
| threshold (50) | +0.030 | −0.018 [−0.045, +0.004] | −0.037 [−0.056, −0.023] | +0.026 |

Every "hybrid − known" CI on the structured types excludes 0, except 235B P date (27 pairs,
[−0.005, +0.044]). The CIs are in the appendix.

**Singles.** Every model and arm is worse than the price at t: Q arms +0.26 to +0.47, P arms
+0.02 to +0.07 (32B P control raw +0.024 [−0.001, +0.049]). Hybrid singles are the market by
construction.

## How well the answers respect the known relation

A is the antecedent. "Zero cell" means the stated P(A and not B) ≤ 0.05. D is the share of the
attainable dependence captured with the model's own marginals: D = 0 is independence and D = 1
is the Fréchet bound.

| type | model | raw: order right | raw: zero cell | raw: D ≥ 0.5 | projected: order | projected: zero cell | projected: D ≥ 0.5 (median D) |
|---|---|---|---|---|---|---|---|
| threshold | 32B Q | 0.96 | 0.66 | 0.62 | 0.90 | 0.40 | 0.33 (0.31) |
| threshold | 32B P | 0.98 | 0.76 | 0.77 | 0.97 | 0.44 | 0.19 (0.26) |
| threshold | 235B Q | 0.99 | 0.60 | 0.83 | 0.98 | 0.38 | 0.45 (0.46) |
| date | 32B Q | 0.87 | 0.27 | 0.74 | 0.80 | 0.07 | 0.15 (0.21) |
| date | 32B P | 0.97 | 0.41 | 0.83 | 0.92 | 0.25 | 0.06 (0.20) |
| date | 235B Q | 0.89 | 0.39 | 0.84 | 0.92 | 0.26 | 0.41 (0.46) |

| type | model | raw: P(A and B) ≤ 0.05 | raw: p(A) + p(B) ≤ 1.01 | raw D ≥ 0.5 | projected: P(A and B) ≤ 0.05 | projected D ≥ 0.5 |
|---|---|---|---|---|---|---|
| mc_excl | 32B Q | 0.94 | 0.87 | 0.92 | 0.64 | 0.68 |
| mc_excl | 32B P | 0.95 | 0.85 | 0.94 | 0.44 | 0.43 |
| mc_excl | 235B Q | 0.96 | 0.92 | 0.87 | 0.55 | 0.47 |

For comparison, on the controls the median projected log odds ratio is +0.08 (32B Q), +0.21
(32B P) and +0.37 (235B Q). |log OR| > 1 on 8%, 36% and 6% of control pairs.

What this shows:
- The stated AND answers carry the structure: for implications, P(A and B) is close to P(A), and
  for exclusives, P(A and B) is close to 0.
- The KL projection fits one table to all 12 noisy answers, including the OR and negated forms.
  That pulls the implied zero cell up and cuts D from about 0.8 to about 0.3.
- The hybrid inherits the diluted odds ratio.

## Exploratory: odds ratio from the raw AND answer (post hoc)

**hybrid_and3** uses the market marginals plus the odds ratio of the model's raw stated P, Q and
"P and Q". The AND answer is clipped into its Fréchet range, there is no projection and the known
relation is not used. I defined it **after seeing the 32B results**. The 235B runs were scored with
it unchanged, so they are the cleaner test.

Joint-form Δ vs market + independence:

| type | 32B Q | 32B P | 235B Q | 235B P (subset) |
|---|---|---|---|---|
| threshold | −0.008 [−0.022, +0.007] | −0.023 [−0.038, −0.009] | −0.032 [−0.046, −0.018] | −0.058 [−0.079, −0.039] |
| date | −0.038 [−0.058, −0.017] | −0.040 [−0.056, −0.024] | −0.044 [−0.064, −0.022] | −0.035 [−0.072, +0.003] |
| mc_excl | −0.053 [−0.064, −0.039] | −0.057 [−0.069, −0.044] | −0.042 [−0.053, −0.031] | −0.037 [−0.055, −0.017] |
| control | +0.019 [+0.005, +0.034] | +0.005 [−0.001, +0.014] | +0.011 [−0.002, +0.026] | −0.000 |

- It closes most of the gap to the known-relation oracle. For 235B Q the remaining gap is +0.006
  [−0.001, +0.014] on threshold, +0.012 on date and +0.017 on mc_excl.
- It pays for this on unrelated pairs, where it invents dependence: +0.011 to +0.019 on controls.
- A structure-preserving projection, for example one that weights the AND forms or fits the
  4 atoms on literals + AND only, is the obvious next step. It is untested here.

## Context and model size (joint forms, paired, x − y)

| comparison | projected | hybrid |
|---|---|---|
| 32B P − 32B Q, all | −0.149 [−0.167, −0.130] | +0.003 [−0.000, +0.006] |
| 235B Q − 32B Q, all | −0.020 [−0.032, −0.007] | −0.003 [−0.006, +0.000] |
| 235B Q − 32B Q, threshold / date | −0.028 / −0.044 | −0.009 [−0.016, −0.003] / −0.015 [−0.023, −0.007] |
| 235B Q − 32B Q, mc_excl | −0.002 | +0.008 [+0.001, +0.014] |
| 235B P − 235B Q (170 pairs) | −0.163 [−0.200, −0.125] | +0.003 [−0.003, +0.009] |

- The pack improves the model's own answers a lot but adds nothing to the dependence: the hybrid
  ties across arms. The structure knowledge comes from the question text, not from the evidence.
- 235B captures ladder structure better than 32B, but not exclusivity.

## Where the market violates the relation

This subset is small, so these numbers are indicative only. Joint log loss:

| subset | pairs | market indep | market + known | 32B P projected | 235B Q projected | 32B P hybrid |
|---|---|---|---|---|---|---|
| threshold, price(A) > price(B) | 7 | 0.220 | 0.146 | 0.189 | 0.418 | 0.179 |
| date, price(A) > price(B) | 10 | 0.069 | 0.051 | 0.135 | 0.389 | 0.063 |

- Only 32B P projected beats the market there, on 7 pairs: −0.032 [−0.085, +0.023].
- Imposing the relation on the market prices (market + known) always helps.
- The mc_excl "violation" subset is the stale-price artifact described above.

## Caveats

- **Ladders are clustered.** 480 ladder pairs come from 240 groups, mostly MC markets whose rungs
  share one event. The CIs resample whole groups.
- **The oracle uses the Fréchet bound, which is exact only when both marginals are right.** With a
  market violation (price(A) > price(B)), the oracle keeps a positive A-and-not-B cell.
- **Selection leaves many ladders out.** The open-at-t rule drops many rungs that had already
  resolved (4,133 rejected candidate pairs), so ladders whose first rung resolves early are
  under-represented.
- **Label-dependent closing week.** The t rule inherits the JEV design: the closing week comes
  from min(close, resolution), so an early YES moves t. The market and the models see the same t.
- **mc_excl uses the top-2 options at t.** Both are priced, so these pairs carry the most
  exclusivity signal.

## Files

- Code:
  - `scripts/structured_pairs.py` (build and run);
  - `scripts/structured_pairs_rules.py` (ladder rules);
  - `scripts/score_structured_pairs.py` (scoring, chart JSON);
  - `scripts/tables_structured_pairs.py` (tables).
- Data in `/home/agent/data/structured-pairs-20260924/`:
  - `pairs.jsonl` (relation, prices at t, as-of text; no outcomes);
  - `labels.jsonl` (scoring only);
  - `packs-structured.jsonl` and `prompts-structured-{Q,P}.jsonl`;
  - `logs/{model}-{arm}-structured.jsonl` (raw replies, usage, cost);
  - `scores.json` and `tables.md`;
  - `build-stats.json`, `build.log` and `run-*.log`.
- Chart JSON: `docs/structured-pairs-20260924-charts.json`, with a row for each model × arm ×
  type × form group × method (Δ vs market independence and vs market + known relation, with CIs),
  plus the market check, the outcome check and the respect rates.
- Progress log: `/home/agent/workspaces/calibrated-ai/structured-pairs-progress.log`.

## Appendix: all tables (generated)

### joint forms: log loss (Δ vs market + independence, cluster-bootstrap 95% CI)

| type | model | arm | pairs | market indep | market + known relation | raw Δ | projected Δ | hybrid Δ | hybrid_and3 Δ |
|---|---|---|---|---|---|---|---|---|---|
| threshold | 235B | P | 50 | 0.251 | 0.189 | 0.282 (+0.030 [-0.014, +0.075]) | 0.233 (-0.018 [-0.045, +0.004]) | 0.214 (-0.037 [-0.056, -0.023]) | 0.193 (-0.058 [-0.079, -0.039]) |
| threshold | 235B | Q | 300 | 0.291 | 0.252 | 0.529 (+0.239 [+0.195, +0.282]) | 0.440 (+0.149 [+0.109, +0.187]) | 0.267 (-0.024 [-0.031, -0.018]) | 0.258 (-0.032 [-0.046, -0.018]) |
| threshold | 32B | P | 300 | 0.291 | 0.252 | 0.493 (+0.202 [+0.161, +0.247]) | 0.333 (+0.042 [+0.022, +0.063]) | 0.280 (-0.011 [-0.017, -0.005]) | 0.267 (-0.023 [-0.038, -0.009]) |
| threshold | 32B | Q | 300 | 0.291 | 0.252 | 0.669 (+0.378 [+0.318, +0.433]) | 0.468 (+0.177 [+0.131, +0.224]) | 0.276 (-0.015 [-0.022, -0.007]) | 0.282 (-0.008 [-0.022, +0.007]) |
| date | 235B | P | 27 | 0.371 | 0.335 | 0.426 (+0.055 [-0.000, +0.113]) | 0.390 (+0.018 [-0.032, +0.064]) | 0.355 (-0.016 [-0.034, +0.003]) | 0.337 (-0.035 [-0.072, +0.003]) |
| date | 235B | Q | 180 | 0.338 | 0.282 | 0.603 (+0.265 [+0.198, +0.328]) | 0.514 (+0.176 [+0.112, +0.237]) | 0.306 (-0.032 [-0.044, -0.022]) | 0.294 (-0.044 [-0.064, -0.022]) |
| date | 32B | P | 180 | 0.338 | 0.282 | 0.448 (+0.110 [+0.070, +0.150]) | 0.355 (+0.017 [-0.004, +0.038]) | 0.328 (-0.010 [-0.018, -0.002]) | 0.298 (-0.040 [-0.056, -0.024]) |
| date | 32B | Q | 180 | 0.338 | 0.282 | 0.682 (+0.345 [+0.270, +0.418]) | 0.558 (+0.220 [+0.153, +0.284]) | 0.320 (-0.018 [-0.027, -0.010]) | 0.300 (-0.038 [-0.058, -0.017]) |
| mc_excl | 235B | P | 49 | 0.357 | 0.292 | 0.486 (+0.129 [+0.075, +0.189]) | 0.358 (+0.001 [-0.025, +0.028]) | 0.336 (-0.021 [-0.032, -0.010]) | 0.320 (-0.037 [-0.055, -0.017]) |
| mc_excl | 235B | Q | 300 | 0.377 | 0.318 | 0.617 (+0.240 [+0.203, +0.279]) | 0.510 (+0.133 [+0.098, +0.170]) | 0.347 (-0.030 [-0.037, -0.023]) | 0.335 (-0.042 [-0.053, -0.031]) |
| mc_excl | 32B | P | 300 | 0.377 | 0.318 | 0.533 (+0.156 [+0.121, +0.193]) | 0.377 (-0.000 [-0.020, +0.021]) | 0.342 (-0.035 [-0.042, -0.027]) | 0.320 (-0.057 [-0.069, -0.044]) |
| mc_excl | 32B | Q | 300 | 0.377 | 0.318 | 0.673 (+0.296 [+0.254, +0.339]) | 0.513 (+0.136 [+0.099, +0.171]) | 0.339 (-0.038 [-0.047, -0.028]) | 0.324 (-0.053 [-0.064, -0.039]) |
| control | 235B | P | 44 | 0.276 | 0.276 | 0.403 (+0.126 [+0.073, +0.181]) | 0.321 (+0.044 [+0.020, +0.071]) | 0.275 (-0.001 [-0.004, +0.004]) | 0.276 (-0.000 [-0.002, +0.002]) |
| control | 235B | Q | 297 | 0.304 | 0.304 | 0.514 (+0.210 [+0.179, +0.242]) | 0.475 (+0.171 [+0.141, +0.201]) | 0.302 (-0.001 [-0.003, +0.000]) | 0.314 (+0.011 [-0.002, +0.026]) |
| control | 32B | P | 297 | 0.304 | 0.304 | 0.477 (+0.174 [+0.142, +0.205]) | 0.347 (+0.043 [+0.023, +0.062]) | 0.302 (-0.001 [-0.004, +0.001]) | 0.309 (+0.005 [-0.001, +0.014]) |
| control | 32B | Q | 297 | 0.304 | 0.304 | 0.541 (+0.237 [+0.205, +0.269]) | 0.489 (+0.185 [+0.155, +0.214]) | 0.303 (-0.000 [-0.002, +0.001]) | 0.323 (+0.019 [+0.005, +0.034]) |
| all | 235B | P | 170 | 0.307 | 0.264 | 0.395 (+0.087 [+0.061, +0.117]) | 0.317 (+0.010 [-0.005, +0.024]) | 0.288 (-0.020 [-0.027, -0.013]) | 0.274 (-0.033 [-0.043, -0.023]) |
| all | 235B | Q | 1077 | 0.326 | 0.290 | 0.562 (+0.236 [+0.215, +0.258]) | 0.482 (+0.155 [+0.136, +0.176]) | 0.305 (-0.021 [-0.024, -0.017]) | 0.301 (-0.025 [-0.032, -0.018]) |
| all | 32B | P | 1077 | 0.326 | 0.290 | 0.492 (+0.166 [+0.147, +0.185]) | 0.353 (+0.026 [+0.016, +0.037]) | 0.311 (-0.015 [-0.018, -0.012]) | 0.298 (-0.028 [-0.034, -0.021]) |
| all | 32B | Q | 1077 | 0.326 | 0.290 | 0.637 (+0.311 [+0.286, +0.337]) | 0.501 (+0.175 [+0.154, +0.197]) | 0.308 (-0.018 [-0.022, -0.014]) | 0.308 (-0.018 [-0.026, -0.010]) |

### single forms: log loss (Δ vs market + independence, cluster-bootstrap 95% CI)

| type | model | arm | pairs | market indep | market + known relation | raw Δ | projected Δ | hybrid Δ | hybrid_and3 Δ |
|---|---|---|---|---|---|---|---|---|---|
| threshold | 235B | P | 50 | 0.292 | 0.292 | 0.299 (+0.007 [-0.001, +0.020]) | 0.318 (+0.026 [+0.006, +0.044]) | 0.292 (+0.000 [-0.000, +0.000]) | 0.292 (+0.000 [-0.000, +0.000]) |
| threshold | 235B | Q | 300 | 0.336 | 0.336 | 0.643 (+0.308 [+0.243, +0.372]) | 0.591 (+0.255 [+0.197, +0.312]) | 0.336 (-0.000 [-0.000, +0.000]) | 0.336 (-0.000 [-0.000, +0.000]) |
| threshold | 32B | P | 300 | 0.336 | 0.336 | 0.385 (+0.049 [+0.022, +0.085]) | 0.400 (+0.064 [+0.041, +0.091]) | 0.336 (-0.000 [-0.000, +0.000]) | 0.336 (-0.000 [-0.000, +0.000]) |
| threshold | 32B | Q | 300 | 0.336 | 0.336 | 0.726 (+0.390 [+0.313, +0.468]) | 0.602 (+0.266 [+0.201, +0.333]) | 0.336 (-0.000 [-0.000, +0.000]) | 0.336 (-0.000 [-0.000, +0.000]) |
| date | 235B | P | 27 | 0.426 | 0.426 | 0.639 (+0.213 [+0.045, +0.445]) | 0.477 (+0.051 [+0.000, +0.099]) | 0.426 (-0.000 [-0.000, +0.000]) | 0.426 (-0.000 [-0.000, +0.000]) |
| date | 235B | Q | 180 | 0.401 | 0.401 | 0.862 (+0.462 [+0.362, +0.555]) | 0.731 (+0.330 [+0.240, +0.418]) | 0.401 (+0.000 [-0.000, +0.000]) | 0.401 (-0.000 [-0.000, +0.000]) |
| date | 32B | P | 180 | 0.401 | 0.401 | 0.432 (+0.031 [+0.011, +0.056]) | 0.433 (+0.032 [+0.008, +0.054]) | 0.401 (+0.000 [-0.000, +0.000]) | 0.401 (-0.000 [-0.000, +0.000]) |
| date | 32B | Q | 180 | 0.401 | 0.401 | 0.873 (+0.472 [+0.363, +0.586]) | 0.744 (+0.343 [+0.247, +0.437]) | 0.401 (-0.000 [-0.000, +0.000]) | 0.401 (+0.000 [-0.000, +0.000]) |
| mc_excl | 235B | P | 49 | 0.419 | 0.419 | 0.510 (+0.091 [+0.017, +0.174]) | 0.450 (+0.030 [-0.002, +0.064]) | 0.419 (-0.000 [-0.000, +0.000]) | 0.419 (-0.000 [-0.000, +0.000]) |
| mc_excl | 235B | Q | 300 | 0.450 | 0.450 | 0.711 (+0.260 [+0.213, +0.310]) | 0.666 (+0.215 [+0.169, +0.263]) | 0.450 (-0.000 [-0.000, +0.000]) | 0.450 (+0.000 [-0.000, +0.000]) |
| mc_excl | 32B | P | 300 | 0.450 | 0.450 | 0.502 (+0.052 [+0.026, +0.079]) | 0.485 (+0.035 [+0.015, +0.056]) | 0.450 (-0.000 [-0.000, +0.000]) | 0.450 (+0.000 [-0.000, +0.000]) |
| mc_excl | 32B | Q | 300 | 0.450 | 0.450 | 0.727 (+0.277 [+0.227, +0.328]) | 0.672 (+0.221 [+0.176, +0.270]) | 0.450 (-0.000 [-0.000, +0.000]) | 0.450 (+0.000 [-0.000, +0.000]) |
| control | 235B | P | 44 | 0.300 | 0.300 | 0.300 (+0.000 [-0.001, +0.002]) | 0.354 (+0.054 [+0.025, +0.082]) | 0.300 (-0.000 [-0.000, +0.000]) | 0.300 (+0.000 [-0.000, +0.000]) |
| control | 235B | Q | 297 | 0.338 | 0.338 | 0.587 (+0.249 [+0.206, +0.293]) | 0.568 (+0.230 [+0.192, +0.268]) | 0.338 (-0.000 [-0.000, +0.000]) | 0.338 (+0.000 [-0.000, +0.000]) |
| control | 32B | P | 297 | 0.338 | 0.338 | 0.362 (+0.024 [-0.001, +0.049]) | 0.393 (+0.055 [+0.031, +0.076]) | 0.338 (+0.000 [-0.000, +0.000]) | 0.338 (-0.000 [-0.000, +0.000]) |
| control | 32B | Q | 297 | 0.338 | 0.338 | 0.708 (+0.370 [+0.323, +0.416]) | 0.581 (+0.244 [+0.206, +0.279]) | 0.338 (+0.000 [-0.000, +0.000]) | 0.338 (-0.000 [-0.000, +0.000]) |
| all | 235B | P | 170 | 0.352 | 0.352 | 0.414 (+0.062 [+0.025, +0.104]) | 0.390 (+0.039 [+0.023, +0.054]) | 0.352 (-0.000 [-0.000, +0.000]) | 0.352 (-0.000 [-0.000, +0.000]) |
| all | 235B | Q | 1077 | 0.379 | 0.379 | 0.683 (+0.304 [+0.274, +0.333]) | 0.629 (+0.250 [+0.224, +0.278]) | 0.379 (-0.000 [-0.000, +0.000]) | 0.379 (+0.000 [-0.000, +0.000]) |
| all | 32B | P | 1077 | 0.379 | 0.379 | 0.419 (+0.040 [+0.027, +0.055]) | 0.427 (+0.048 [+0.037, +0.060]) | 0.379 (-0.000 [-0.000, +0.000]) | 0.379 (-0.000 [-0.000, +0.000]) |
| all | 32B | Q | 1077 | 0.379 | 0.379 | 0.746 (+0.367 [+0.335, +0.402]) | 0.639 (+0.260 [+0.232, +0.291]) | 0.379 (-0.000 [-0.000, +0.000]) | 0.379 (-0.000 [-0.000, +0.000]) |

### joint forms: Δ vs market + KNOWN relation (oracle structure)

| type | model | arm | raw | projected | hybrid | hybrid_and3 |
|---|---|---|---|---|---|---|
| threshold | 235B | P | +0.093 [+0.060, +0.130] | +0.045 [+0.031, +0.059] | +0.026 [+0.017, +0.035] | +0.005 [+0.000, +0.013] |
| threshold | 235B | Q | +0.278 [+0.234, +0.318] | +0.188 [+0.149, +0.224] | +0.015 [+0.004, +0.024] | +0.006 [-0.001, +0.014] |
| threshold | 32B | P | +0.241 [+0.199, +0.286] | +0.081 [+0.058, +0.105] | +0.028 [+0.014, +0.041] | +0.015 [+0.008, +0.025] |
| threshold | 32B | Q | +0.417 [+0.358, +0.472] | +0.216 [+0.170, +0.262] | +0.024 [+0.014, +0.034] | +0.030 [+0.013, +0.050] |
| date | 235B | P | +0.091 [+0.037, +0.144] | +0.055 [+0.009, +0.095] | +0.020 [-0.005, +0.044] | +0.002 [-0.009, +0.011] |
| date | 235B | Q | +0.321 [+0.260, +0.377] | +0.232 [+0.176, +0.284] | +0.023 [+0.013, +0.034] | +0.012 [+0.003, +0.025] |
| date | 32B | P | +0.166 [+0.134, +0.198] | +0.073 [+0.056, +0.090] | +0.046 [+0.031, +0.060] | +0.016 [+0.004, +0.028] |
| date | 32B | Q | +0.400 [+0.334, +0.467] | +0.276 [+0.217, +0.332] | +0.038 [+0.024, +0.053] | +0.018 [+0.004, +0.036] |
| mc_excl | 235B | P | +0.194 [+0.131, +0.261] | +0.067 [+0.031, +0.099] | +0.044 [+0.017, +0.069] | +0.028 [+0.010, +0.049] |
| mc_excl | 235B | Q | +0.299 [+0.261, +0.336] | +0.192 [+0.157, +0.228] | +0.029 [+0.019, +0.038] | +0.017 [+0.008, +0.026] |
| mc_excl | 32B | P | +0.215 [+0.182, +0.250] | +0.058 [+0.042, +0.076] | +0.024 [+0.014, +0.032] | +0.002 [-0.004, +0.006] |
| mc_excl | 32B | Q | +0.355 [+0.313, +0.397] | +0.194 [+0.161, +0.227] | +0.021 [+0.012, +0.029] | +0.006 [+0.001, +0.012] |
| control | 235B | P | +0.126 [+0.073, +0.181] | +0.044 [+0.020, +0.071] | -0.001 [-0.004, +0.004] | -0.000 [-0.002, +0.002] |
| control | 235B | Q | +0.210 [+0.179, +0.242] | +0.171 [+0.141, +0.201] | -0.001 [-0.003, +0.000] | +0.011 [-0.002, +0.026] |
| control | 32B | P | +0.174 [+0.142, +0.205] | +0.043 [+0.023, +0.062] | -0.001 [-0.004, +0.001] | +0.005 [-0.001, +0.014] |
| control | 32B | Q | +0.237 [+0.205, +0.269] | +0.185 [+0.155, +0.214] | -0.000 [-0.002, +0.001] | +0.019 [+0.005, +0.034] |
| all | 235B | P | +0.131 [+0.104, +0.160] | +0.053 [+0.038, +0.067] | +0.023 [+0.014, +0.033] | +0.010 [+0.004, +0.016] |
| all | 235B | Q | +0.272 [+0.252, +0.294] | +0.192 [+0.173, +0.211] | +0.016 [+0.011, +0.020] | +0.011 [+0.006, +0.017] |
| all | 32B | P | +0.203 [+0.183, +0.222] | +0.063 [+0.053, +0.074] | +0.022 [+0.016, +0.027] | +0.009 [+0.005, +0.013] |
| all | 32B | Q | +0.347 [+0.323, +0.372] | +0.211 [+0.192, +0.233] | +0.019 [+0.014, +0.023] | +0.019 [+0.011, +0.026] |

### oracle structure vs independence, market prices (joint / all 12)

| type | market indep joint | oracle joint | Δ joint [CI] | Δ all12 [CI] |
|---|---|---|---|---|
| threshold | 0.291 | 0.252 | -0.039 [-0.053, -0.024] | -0.026 [-0.035, -0.016] |
| date | 0.338 | 0.282 | -0.056 [-0.074, -0.037] | -0.037 [-0.049, -0.025] |
| mc_excl | 0.377 | 0.318 | -0.059 [-0.071, -0.045] | -0.039 [-0.047, -0.030] |
| control | 0.304 | 0.304 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### Does the model respect the known relation? (share of pairs; D = dependence captured, 1 = full)

| type | model | arm | form | order p(A)≤p(B) | zero cell ≤0.05 | AND≈p(A) | D median | D≥0.5 (n) |
|---|---|---|---|---|---|---|---|---|
| threshold | 235B | P | raw | 0.98 | 0.56 | 0.96 | 1.00 | 0.95 (43) |
| threshold | 235B | P | projected | 0.98 | 0.56 | 0.56 | 0.45 | 0.37 (46) |
| threshold | 235B | Q | raw | 0.99 | 0.60 | 0.42 | 1.00 | 0.83 (294) |
| threshold | 235B | Q | projected | 0.98 | 0.38 | 0.38 | 0.46 | 0.45 (300) |
| threshold | 32B | P | raw | 0.98 | 0.76 | 0.77 | 1.00 | 0.77 (265) |
| threshold | 32B | P | projected | 0.97 | 0.44 | 0.44 | 0.26 | 0.19 (293) |
| threshold | 32B | Q | raw | 0.96 | 0.66 | 0.29 | 1.00 | 0.62 (264) |
| threshold | 32B | Q | projected | 0.90 | 0.40 | 0.40 | 0.31 | 0.33 (293) |
| date | 235B | P | raw | 0.96 | 0.37 | 0.89 | 1.00 | 0.77 (22) |
| date | 235B | P | projected | 0.96 | 0.44 | 0.44 | 0.39 | 0.30 (27) |
| date | 235B | Q | raw | 0.89 | 0.39 | 0.29 | 1.89 | 0.84 (178) |
| date | 235B | Q | projected | 0.92 | 0.26 | 0.26 | 0.46 | 0.41 (180) |
| date | 32B | P | raw | 0.97 | 0.41 | 0.77 | 1.00 | 0.83 (151) |
| date | 32B | P | projected | 0.92 | 0.25 | 0.25 | 0.20 | 0.06 (174) |
| date | 32B | Q | raw | 0.87 | 0.27 | 0.28 | 1.00 | 0.74 (179) |
| date | 32B | Q | projected | 0.80 | 0.07 | 0.07 | 0.21 | 0.15 (180) |

| type | model | arm | form | p(A and B)≤0.05 | p(A)+p(B)≤1.01 | D median | D≥0.5 (n) |
|---|---|---|---|---|---|---|---|
| mc_excl | 235B | P | raw | 0.63 | 0.78 | 0.29 | 0.48 (44) |
| mc_excl | 235B | P | projected | 0.24 | 0.69 | 0.15 | 0.21 (47) |
| mc_excl | 235B | Q | raw | 0.96 | 0.92 | 0.85 | 0.87 (282) |
| mc_excl | 235B | Q | projected | 0.55 | 0.98 | 0.49 | 0.47 (294) |
| mc_excl | 32B | P | raw | 0.95 | 0.85 | 1.00 | 0.94 (277) |
| mc_excl | 32B | P | projected | 0.44 | 0.81 | 0.42 | 0.43 (295) |
| mc_excl | 32B | Q | raw | 0.94 | 0.87 | 1.00 | 0.92 (295) |
| mc_excl | 32B | Q | projected | 0.64 | 0.94 | 0.67 | 0.68 (295) |

### control pairs: projected log odds ratio

| model | arm | median | share abs > 1 |
|---|---|---|---|
| 235B | P | +0.35 | 0.30 |
| 235B | Q | +0.37 | 0.06 |
| 32B | P | +0.21 | 0.36 |
| 32B | Q | +0.08 | 0.08 |

### pairs where the market violates the relation at t (joint log loss)

| type | model | arm | pairs | market indep | market oracle | raw | projected | hybrid | hybrid_and3 | projected − indep [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| threshold | 235B | P | 1 | 0.389 | 0.231 | 0.199 | 0.245 | 0.306 | 0.231 | -0.144 [-0.144, -0.144] |
| threshold | 235B | Q | 7 | 0.220 | 0.146 | 0.474 | 0.418 | 0.177 | 0.146 | +0.198 [-0.031, +0.399] |
| threshold | 32B | P | 7 | 0.220 | 0.146 | 0.260 | 0.189 | 0.179 | 0.155 | -0.032 [-0.085, +0.023] |
| threshold | 32B | Q | 7 | 0.220 | 0.146 | 0.488 | 0.413 | 0.169 | 0.146 | +0.192 [-0.034, +0.385] |
| date | 235B | P | 3 | 0.050 | 0.033 | 0.135 | 0.124 | 0.042 | 0.042 | +0.074 [-0.015, +0.154] |
| date | 235B | Q | 10 | 0.069 | 0.051 | 0.424 | 0.389 | 0.062 | 0.056 | +0.319 [+0.176, +0.473] |
| date | 32B | P | 10 | 0.069 | 0.051 | 0.285 | 0.135 | 0.063 | 0.053 | +0.066 [+0.025, +0.109] |
| date | 32B | Q | 10 | 0.069 | 0.051 | 0.624 | 0.509 | 0.064 | 0.057 | +0.440 [+0.308, +0.598] |
| mc_excl | 235B | P | 15 | 0.280 | 0.192 | 0.504 | 0.289 | 0.241 | 0.243 | +0.009 [-0.049, +0.060] |
| mc_excl | 235B | Q | 75 | 0.299 | 0.233 | 0.660 | 0.526 | 0.259 | 0.242 | +0.226 [+0.147, +0.307] |
| mc_excl | 32B | P | 75 | 0.299 | 0.233 | 0.619 | 0.336 | 0.262 | 0.237 | +0.037 [-0.003, +0.080] |
| mc_excl | 32B | Q | 75 | 0.299 | 0.233 | 0.678 | 0.512 | 0.250 | 0.241 | +0.213 [+0.138, +0.292] |

### projected − raw (joint)

| type | model | arm | Δ [CI] |
|---|---|---|---|
| threshold | 235B | P | -0.048 [-0.076, -0.026] |
| threshold | 235B | Q | -0.089 [-0.105, -0.074] |
| threshold | 32B | P | -0.160 [-0.188, -0.135] |
| threshold | 32B | Q | -0.201 [-0.229, -0.176] |
| date | 235B | P | -0.036 [-0.069, -0.004] |
| date | 235B | Q | -0.089 [-0.114, -0.065] |
| date | 32B | P | -0.093 [-0.115, -0.073] |
| date | 32B | Q | -0.125 [-0.160, -0.093] |
| mc_excl | 235B | P | -0.128 [-0.172, -0.089] |
| mc_excl | 235B | Q | -0.107 [-0.120, -0.094] |
| mc_excl | 32B | P | -0.157 [-0.180, -0.135] |
| mc_excl | 32B | Q | -0.161 [-0.180, -0.141] |
| control | 235B | P | -0.082 [-0.112, -0.052] |
| control | 235B | Q | -0.039 [-0.045, -0.033] |
| control | 32B | P | -0.131 [-0.150, -0.112] |
| control | 32B | Q | -0.052 [-0.064, -0.041] |
| all | 235B | P | -0.078 [-0.097, -0.061] |
| all | 235B | Q | -0.080 [-0.088, -0.073] |
| all | 32B | P | -0.140 [-0.152, -0.128] |
| all | 32B | Q | -0.136 [-0.148, -0.124] |

### paired comparisons, joint forms (x − y on common pairs; negative = x better)

| x | y | type | pairs | raw | projected | hybrid | hybrid_and3 |
|---|---|---|---|---|---|---|---|
| qwen3-32b|P | qwen3-32b|Q | threshold | 300 | -0.176 [-0.230, -0.121] | -0.135 [-0.176, -0.095] | +0.004 [-0.004, +0.011] | -0.015 [-0.033, +0.001] |
| qwen3-32b|P | qwen3-32b|Q | date | 180 | -0.235 [-0.300, -0.170] | -0.202 [-0.258, -0.145] | +0.008 [-0.001, +0.017] | -0.003 [-0.018, +0.012] |
| qwen3-32b|P | qwen3-32b|Q | mc_excl | 300 | -0.140 [-0.181, -0.094] | -0.136 [-0.167, -0.104] | +0.003 [-0.004, +0.009] | -0.005 [-0.009, -0.001] |
| qwen3-32b|P | qwen3-32b|Q | control | 297 | -0.063 [-0.093, -0.034] | -0.142 [-0.164, -0.119] | -0.001 [-0.004, +0.002] | -0.014 [-0.029, -0.000] |
| qwen3-32b|P | qwen3-32b|Q | all | 1077 | -0.145 [-0.168, -0.120] | -0.149 [-0.167, -0.130] | +0.003 [-0.000, +0.006] | -0.010 [-0.017, -0.003] |
| qwen3-235b-a22b|Q | qwen3-32b|Q | threshold | 300 | -0.140 [-0.176, -0.104] | -0.028 [-0.052, -0.003] | -0.009 [-0.016, -0.003] | -0.024 [-0.044, -0.007] |
| qwen3-235b-a22b|Q | qwen3-32b|Q | date | 180 | -0.079 [-0.138, -0.021] | -0.044 [-0.085, -0.005] | -0.015 [-0.023, -0.007] | -0.006 [-0.027, +0.013] |
| qwen3-235b-a22b|Q | qwen3-32b|Q | mc_excl | 300 | -0.056 [-0.089, -0.023] | -0.002 [-0.025, +0.021] | +0.008 [+0.001, +0.014] | +0.010 [+0.002, +0.019] |
| qwen3-235b-a22b|Q | qwen3-32b|Q | control | 297 | -0.027 [-0.044, -0.010] | -0.014 [-0.029, +0.003] | -0.001 [-0.003, +0.001] | -0.009 [-0.022, +0.005] |
| qwen3-235b-a22b|Q | qwen3-32b|Q | all | 1077 | -0.075 [-0.093, -0.057] | -0.020 [-0.032, -0.007] | -0.003 [-0.006, +0.000] | -0.007 [-0.015, +0.001] |
| qwen3-235b-a22b|P | qwen3-235b-a22b|Q | threshold | 50 | -0.201 [-0.289, -0.116] | -0.175 [-0.244, -0.105] | -0.001 [-0.007, +0.005] | -0.005 [-0.014, +0.002] |
| qwen3-235b-a22b|P | qwen3-235b-a22b|Q | date | 27 | -0.175 [-0.297, -0.047] | -0.139 [-0.260, -0.017] | +0.010 [-0.003, +0.025] | +0.001 [-0.015, +0.015] |
| qwen3-235b-a22b|P | qwen3-235b-a22b|Q | mc_excl | 49 | -0.120 [-0.191, -0.035] | -0.155 [-0.227, -0.071] | +0.006 [-0.010, +0.026] | +0.007 [-0.017, +0.035] |
| qwen3-235b-a22b|P | qwen3-235b-a22b|Q | control | 44 | -0.135 [-0.190, -0.083] | -0.173 [-0.229, -0.115] | -0.001 [-0.006, +0.004] | -0.016 [-0.063, +0.015] |
| qwen3-235b-a22b|P | qwen3-235b-a22b|Q | all | 170 | -0.157 [-0.197, -0.115] | -0.163 [-0.200, -0.125] | +0.003 [-0.003, +0.009] | -0.003 [-0.018, +0.009] |
| qwen3-235b-a22b|P | qwen3-32b|P | threshold | 50 | -0.167 [-0.247, -0.094] | -0.060 [-0.095, -0.031] | -0.019 [-0.034, -0.007] | -0.005 [-0.020, +0.007] |
| qwen3-235b-a22b|P | qwen3-32b|P | date | 27 | -0.048 [-0.125, +0.030] | -0.007 [-0.046, +0.029] | -0.012 [-0.029, +0.003] | -0.015 [-0.046, +0.016] |
| qwen3-235b-a22b|P | qwen3-32b|P | mc_excl | 49 | -0.003 [-0.095, +0.087] | +0.003 [-0.048, +0.049] | +0.018 [+0.001, +0.038] | +0.024 [+0.000, +0.050] |
| qwen3-235b-a22b|P | qwen3-32b|P | control | 44 | -0.054 [-0.123, +0.013] | -0.003 [-0.036, +0.030] | +0.005 [-0.001, +0.013] | -0.009 [-0.034, +0.006] |
| qwen3-235b-a22b|P | qwen3-32b|P | all | 170 | -0.072 [-0.115, -0.029] | -0.019 [-0.041, +0.001] | -0.001 [-0.008, +0.007] | +0.001 [-0.011, +0.012] |

### jobs

| model | arm | calls | missing | cost | served |
|---|---|---|---|---|---|
| qwen3-235b-a22b | P | 2040 | 0 | $3.16 | {'qwen3-235b-a22b': 2040} |
| qwen3-235b-a22b | Q | 12924 | 0 | $5.24 | {'qwen3-235b-a22b': 12924} |
| qwen3-32b | P | 12924 | 0 | $4.50 | {'qwen3-32b': 12924} |
| qwen3-32b | Q | 12924 | 0 | $1.20 | {'qwen3-32b': 12924} |
