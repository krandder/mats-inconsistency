# Bias-balanced triangle questions (2026-09-27)

Kelvin's approved design (27 Sep): the weekly world-model run asks only "A and B" and "A and B and C". The
32B states those about +0.07 above the product of its singles, so fits that match the joints drag the singles
up and never beat independence on joints. This run asks every triangle in a sign-balanced way, on the same
triangles and weeks, to check two things: does a balanced design beat independence on joints, and does it
remove the damage to the singles?

**Answer: yes to both, locally per triangle.** The KL projection of the 16 balanced answers gives these
paired differences (95% CI over 839 triangles):
- Sign-balanced joints: −0.087 nats [−0.106, −0.068] against the independence product of the model's own singles.
- The weekly run's 7 AND forms: −0.070 [−0.091, −0.049] against the AND-only fit of the same triangles.
- YES singles: 0.602 against the raw 0.668, which is −0.066 [−0.090, −0.043].
  - The weekly run's global M3 fit scores 0.736 on these same singles, a gain of +0.069 for raw over M3. That gap is the singles damage.
  - The singles-anchored M3b scores 0.664.

The one exception is the 8-outcome joint log loss, where the fit only ties independence (−0.03 [−0.13, +0.06]).
Correcting each question type for its bias, estimated on the other weeks, adds nothing.

Model: Qwen3-32B on DashScope, thinking off, temperature 0, max_tokens 20, one statement per call, v6_direct
question-only prompts with the as-of line (`worldmodel_clusters.messages`, the weekly run's builder).
There was no model substitution: every served model was `qwen3-32b`. All answers parsed and none needed a retry.
Scripts: `scripts/balanced_triangles.py` (build/run) and `scripts/balanced_triangles_analyze.py`.
Data: `/home/agent/data/balanced-triangles-20260927/`, with `triangles.jsonl`, `queries.jsonl`, `log.jsonl`,
`summary.json`, `charts.json` and `analyze.out`. The chart JSON is copied to `docs/balanced-triangles-20260927-charts.json`.

## Design

The triangles come from the test-split Luna graph, with each week's open events, links and triangles taken
from the weekly run (`weekly-worldmodel-20260927/weeks/<t>`). The weeks are 2025-05-12, 06-23, 08-04, 09-15,
10-27 and 12-08. Sampling was seeded, with up to 150 triangles per week. It was stratified by the sorted tuple
of the Luna relation types of the triangle's three links, using largest-remainder allocation. 05-12 has only
89 triangles, so the total is 839.

Each triangle gets a uniformly random sign pattern s. The literal L_k is "Question k resolves YES" or
"... resolves NO". Each triangle has 16 statements:

| Group | Statements |
|---|---|
| chain (10) | T = L_A ∧ L_B ∧ L_C; 3 siblings (one literal flipped); 3 pairs (one literal dropped); 3 singles |
| count (4) | "Of the following three statements, none of them is true / exactly one of them is true / exactly two of them are true / all three of them are true: (1) …; (2) …; (3) …" |
| mirror (2) | the OR form of ¬T ("At least one of the following is true: A resolves NO; …") and of ¬pair_k for a random k |

Prompts reuse the weekly run's wording and orders:
- Singles use the same "Question A" naming as the weekly run.
- Pairs take the weekly and2 order of their link, and triangle-level forms take the weekly and3 order.
- So every all-positive single, pair, triple or sibling is the byte-identical prompt the weekly run answered, and it was reused by sha256.
- A signed statement that appears in several triangles was asked once.

| | count |
|---|---|
| statements | 13,424 (839 × 16) |
| unique prompts | 12,172 |
| reused from the weekly run ($0) | 1,764 (1,233 singles, 599 pairs, 93 T, 320 siblings, counted per statement; 1,764 unique prompts) |
| new calls | 10,408 |
| cost (conservative 0.7/2.8 $/M) | **$4.63** (projected $4.78; cap $9.5 of the $10 budget) |

## Bias: mean (stated − outcome)

Stated answers are clipped to [0.01, 0.99]. CIs come from a bootstrap over statements. The last column breaks
the bias down by how many literals the statement negates, with the yes-rate in brackets.

| type | n | mean stated | yes-rate | bias [95% CI] | by #NOT in statement (yes-rate) |
|---|---|---|---|---|---|
| single | 2,517 | 0.514 | 0.513 | +0.001 [−0.018, +0.021] | 0: +0.198 (0.22) · 1: −0.189 (0.80) |
| pair | 2,517 | 0.347 | 0.259 | +0.089 [+0.072, +0.105] | 0: +0.219 (0.08) · 1: +0.158 (0.12) · 2: −0.168 (0.69) |
| T (triple) | 839 | 0.256 | 0.123 | +0.133 [+0.112, +0.154] | 0: +0.166 (0.04) · 1: +0.144 · 2: +0.205 · 3: −0.125 (0.61) |
| sibling | 2,517 | 0.255 | 0.136 | +0.119 [+0.106, +0.132] | 0: +0.183 (0.03) · 1: +0.133 · 2: +0.168 · 3: −0.137 (0.64) |
| none of three | 839 | 0.279 | 0.116 | +0.163 [+0.140, +0.185] | |
| exactly one | 839 | 0.498 | 0.353 | +0.145 [+0.108, +0.183] | |
| exactly two | 839 | 0.498 | 0.409 | +0.090 [+0.052, +0.124] | |
| all three (count words) | 839 | 0.273 | 0.123 | +0.150 [+0.128, +0.173] | |
| ¬T as OR | 839 | 0.807 | 0.877 | −0.070 [−0.092, −0.049] | |
| ¬pair as OR | 839 | 0.727 | 0.733 | −0.006 [−0.034, +0.021] | |

The bias follows how rare the event is. Rare statements are stated too high and common ones too low, whatever
the wording. A YES single (yes-rate 0.22) is +0.20 too high, and a NO single (yes-rate 0.80) is 0.19 too low.
An all-NO conjunction (yes-rate ≈ 0.6–0.7) is stated too low, and every other conjunction (yes-rate 3–12%) too
high. So the AND bias in the weekly run is mostly the model's pull toward 0.5 on rare events, not something
about the word "and". The same pattern appears in the counting forms: rare cells are over-stated by +0.2 to +0.56.

## Coherence checks on the raw answers

| check | should be | mean [95% CI] | mean \|·\| | share \|·\| > 0.05 |
|---|---|---|---|---|
| ¬T(OR) + T − 1 | 0 | +0.063 [+0.049, +0.077] | 0.139 | 69% |
| ¬pair(OR) + pair − 1 | 0 | +0.080 [+0.064, +0.095] | 0.158 | 72% |
| NO single + weekly YES single − 1 (1,284) | 0 | +0.005 [−0.012, +0.023] | 0.220 | 79% |
| none + one + two + all − 1 | 0 | **+0.548** [+0.506, +0.589] | 0.646 | 96% |
| pair_k − T − sibling_k | 0 | −0.164 [−0.174, −0.153] | 0.222 | 79% |
| single_i − pair_ij − pair_ik + T | ≥ 0 | +0.074; negative in 32% | 0.252 | 77% |
| all-three (count words) − T (and-words) | 0 | +0.017 [+0.009, +0.026] | 0.064 | 38% |

Order checks (T ≤ pair, sibling ≤ pair, pair ≤ each of its singles): 17.8% [16.8, 18.8] of the 12
comparisons are violated, and 79% of triangles have at least one violation. The count partition is the least
coherent: the four counts sum to 1.55 on average. The two wordings of "all three" agree on average (+0.017)
but differ by more than 0.05 in 38% of triangles.

## Log loss (nats per statement, mean over triangles; paired bootstrap 95% CIs, 2,000 resamples)

The variants:
- **raw**: each stated answer as is.
- **indep**: the product of this run's stated signed singles.
- **fitAND**: the KL projection, over the 8 outcomes, of the weekly run's 7 AND-only answers on the same triangle (3 YES singles, 3 "A and B", "A and B and C").
- **fit16**: the KL projection of all 16 balanced answers, with equal weights.
- **fit16bc**: the same after subtracting each type's mean bias from the other 5 weeks.
- **M3 / M3b global**: the weekly run's own week-wide fits (`sixweek/fits`), which exist only for YES and AND forms.

For scoring, every fitted q is mixed with 1% uniform, and every binary prediction is clipped to [0.01, 0.99].

| set | raw | indep | fitAND | fit16 | fit16bc | M3 global | M3b global |
|---|---|---|---|---|---|---|---|
| signed singles (3/tri) | 0.714 | 0.709 | 0.645 | **0.602** | 0.608 | | |
| YES singles (raw = weekly) | 0.668 | 0.709 | 0.645 | **0.602** | 0.608 | 0.736 | 0.664 |
| all 16 forms | 0.558 | 0.535 | 0.496 | **0.454** | 0.454 | | |
| sign-balanced joints (T, 3 sib, 3 pairs) | 0.458 | 0.480 | 0.442 | **0.393** | 0.390 | | |
| weekly AND-only 7 forms (raw = weekly) | 0.539 | 0.512 | 0.506 | **0.436** | 0.447 | 0.548 | 0.499 |
| 8-outcome joint (−log q(outcome)) | – | 2.093 | 2.486 | **2.061** | 2.075 | | |

Key paired differences:

| comparison | Δ [95% CI] |
|---|---|
| fit16 − indep, sign-balanced joints | **−0.087 [−0.106, −0.068]** |
| fit16 − indep, weekly AND 7 forms | −0.076 [−0.093, −0.061] |
| fit16 − indep, 8-outcome joint | −0.032 [−0.127, +0.062] (tie) |
| fit16 − fitAND, sign-balanced joints | −0.049 [−0.069, −0.030] |
| fit16 − fitAND, weekly AND 7 forms | −0.070 [−0.091, −0.049] |
| fit16 − fitAND, 8-outcome joint | −0.425 [−0.584, −0.263] |
| fit16 − raw, YES singles | **−0.066 [−0.090, −0.043]** |
| fit16 − fitAND, YES singles | −0.043 [−0.064, −0.023] |
| M3 global − raw, YES singles (the weekly damage) | +0.069 [+0.049, +0.087] |
| fitAND − indep, sign-balanced joints | +0.038 [+0.014, +0.062] (AND-only fit is worse than independence) |
| fit16bc − fit16, all 16 forms | −0.001 [−0.006, +0.005] |

Mean fitted minus stated single (signed): fit16 +0.020, fit16bc −0.010, fitAND −0.005. The balanced fit
moves the singles in both directions, not upward.

## Reading

- **Joints.** The AND-only design fitted per triangle is worse than independence on sign-balanced joints.
  The balanced design beats independence clearly on every binary joint set. The one exception is the full
  8-outcome log loss, where the fit only ties independence (the CI straddles 0). Its gain is in the low- and
  mid-probability cells that the binary forms score.
- **Singles.** The damage goes away. The balanced fit's YES singles (0.602) beat the raw stated singles
  (0.668), the local AND-only fit (0.645), the weekly global M3 (0.736) and the singles-anchored M3b (0.664).
  Balanced questions make a coherence fit improve the singles instead of dragging them up.
- **Bias correction** by type (leave-one-week-out) adds nothing on top of the balanced fit. With both signs
  asked, the rarity bias partly cancels inside the projection.
- **NO-worded singles** are answered worse than 1 − the YES answer (signed singles raw 0.714 against 0.668 for YES).
  The two are coherent only on average (mean +0.005, mean |·| 0.22).
- **Caveats.** The fits here are local, one triangle at a time, while the weekly fits are week-wide
  (M3 = global, with pairwise and triple terms). The comparison with M3/M3b uses their predictions on the same
  events. Everything is scored on the same weeks the answers came from; outcomes are never used in fitting,
  except that fit16bc uses outcome-based biases from the other weeks. Singles and pairs are shared across
  triangles, so the triangle-level CIs are somewhat optimistic.
