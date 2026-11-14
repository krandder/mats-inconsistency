# Which answers improve which predictions: the order matrix (2026-09-28)

Kelvin's questions (28 Sep):
- Does making answers consistent with pairs improve the singles? With triples? Do triples improve the pairs?
- (1) Do higher-order answers carry information about a single event at all?
- (2) Does the joint structure *connect* information about Q to the prediction of P?

The baseline throughout is **A agreeing with its own negation** ("base"). For each event, only its own
single-event answers (A and NOT A) are reconciled into one probability. There are no joints and no other events,
and the unit's joint is the independence product of those. Every fit is reported as Δ log loss vs base, with 95% CIs.
Negative means better. Raw and independence-of-stated-singles rows are kept for reference.

**Short answers:**
- **(1) Yes, but the information is about the same event.** Where both signs are asked, joint statements that mention P restate P (for example P∧Q + P∧¬Q = P).
  Fitted with no single answers at all, the joints predict P about as well as P's own answers or better.
- **(2) No connection worth the name.** Adding Q's and R's single answers on top of P's own answers and the joints
  changes P's log loss by +0.000 (pairs) and −0.004 (balanced triangles). Replacing them with a random other event's
  answers gives the same result. In the AND-only weekly run it is −0.018 [−0.021, −0.015], but random events'
  singles do even better (−0.006 more), so it is a generic effect, not Q's content.
- **Joints alone** (the weekly run has only positive ANDs) recover nothing there: +0.12 worse than P's own answer.
- **The one exception** is alternatives and cause/effect triangles when P's own answers are left out: there,
  Q's real answer helps by about 0.025 nats.
- **In the 12-form pair data, the singles did gain,** by −0.136 against all four raw single forms. 64% of that is
  P agreeing with NOT P. The other 36% comes from the joint forms that mention P. The cross-event part is 0.

No API calls. Code: `scripts/order_matrix.py`. Outputs: `/home/agent/data/order-matrix-20260928/`, with
`balanced.json`, `weekly.json`, `pairs12.json` and heatmap-ready `charts.json`. The chart JSON is copied to
`docs/order-matrix-20260928-charts.json`.

## Method

- **Fit.** For each unit (a triangle has 8 outcomes, a pair 4), q minimises the sum over forms of KL(stated‖implied), with equal weights.
  Stated values are clipped to [0.01, 0.99], as in the balanced doc.
  - A 1e-3 entropy term breaks ties in directions the chosen answers leave free, which is a max-entropy completion.
  - Singles-only fits are exact: the independence product of the reconciled singles.
- **Scoring.** q is mixed with 1% uniform, and each binary statement is clipped to [0.01, 0.99]. Scores are the mean
  within a unit, then the mean over units. Brier is in the JSON and gives the same signs, except where noted.
- **Target levels.**
  - singles: the YES forms. Log loss is symmetric in sign, so this also covers the NOT forms.
  - pairs asked: the pair statements actually asked.
  - pairs, all 4 signs: all 12 signed pair statements, read from the fit even when they were not asked.
  - triples asked: T + 3 siblings in the balanced run, the positive triple in the weekly run.
  - triples, all 8: the 8 atoms.
  - full outcome: −log q(outcome).
- **CIs.** Event-cluster bootstrap with 2,000 draws, as in the weekly run. Each event (global id, across weeks) gets a
  multinomial count, and a unit gets the mean count of its events. In pairs12 no event repeats, so pairs are resampled.
- **Leave-P-out and swap tests.** These run per unit and per focal event P, scored on P's single:
  - own+J: P's own answers + all joint answers, without the other events' singles.
  - own+oth+J: everything.
  - oth+J: drops every answer about P alone (leave-P-out).
  - J: joints only.
  - oth~+J and own+oth~+J: the other events' singles are replaced by those of random other events of the same week
    (pairs12: a random other pair's event).
  - base rate: the YES rate in the other weeks (pairs12: the other folds).

## 1. The 12-form pairs (Qwen3-32B, 1,066 test pairs, one form per call)

These are the data behind the earlier 0.611 → 0.541 all-12 gain. S = P, NOT P, Q, NOT Q; A = the 4 signed ANDs;
O = the 4 signed ORs. The ORs are complements of the ANDs, so the fitted AND and OR log losses are identical.

| answers used | singles | ANDs (= ORs) | all 12 forms | full outcome |
|---|---|---|---|---|
| base (P with NOT P) | 0.663 | 0.541 | 0.582 | 1.326 |
| S+A | −0.040 [−0.047, −0.032] | −0.030 [−0.036, −0.023] | −0.033 [−0.040, −0.026] | −0.083 [−0.102, −0.063] |
| S+O | −0.037 [−0.045, −0.029] | −0.021 [−0.029, −0.014] | −0.027 [−0.034, −0.020] | −0.059 [−0.081, −0.038] |
| S+A+O (all 12) | **−0.049 [−0.059, −0.039]** | −0.037 [−0.044, −0.029] | −0.041 [−0.049, −0.033] | −0.104 [−0.125, −0.082] |
| A only (no singles) | −0.047 [−0.064, −0.030] | −0.035 [−0.047, −0.022] | −0.039 [−0.053, −0.025] | −0.099 [−0.135, −0.065] |
| A+O only (no singles) | −0.053 [−0.069, −0.037] | −0.040 [−0.052, −0.028] | −0.044 [−0.057, −0.031] | −0.111 [−0.145, −0.078] |
| indep of stated P, Q | −0.020 [−0.037, −0.002] | −0.016 [−0.028, −0.004] | −0.018 [−0.031, −0.004] | −0.042 [−0.075, −0.008] |
| raw (as asked) | 0.750 (4 single forms); YES form 0.645 | 0.533 / ORs 0.551 | 0.611 | – |

Where the singles gain comes from (P's YES single; all 4 single forms in brackets):

| step | Δ log loss [95% CI] |
|---|---|
| raw → base (P reconciled with NOT P: same event) | **+0.019 [+0.000, +0.036]** on the YES form; **−0.087** on all 4 single forms (0.750 → 0.663) |
| base → own + joints (the joint forms that mention P) | **−0.050 [−0.060, −0.040]** |
| + Q's single answers (cross-event) | **+0.000 [−0.001, +0.001]** |
| swap Q's answers for a random event's | +0.000 [−0.001, +0.001] |
| total, fit − raw YES form | −0.031 [−0.049, −0.014] |
| leave-P-out: joints + Q only, − raw YES form | −0.035 [−0.055, −0.016] (joints alone: −0.035) |
| leave-P-out vs the base rate (0.636) | −0.026 [−0.042, −0.011] |

- **Where the gain comes from.** The singles gain is P agreeing with NOT P, plus the ANDs and ORs that mention P
  acting as eight more readings of P. None of it is Q's answer reaching P: adding Q's answer, or swapping it for a
  random event's, changes nothing.
- **P's own two forms vs the joint forms.** Reconciling P with NOT P alone is slightly worse than the YES form alone.
  The NOT answers are poor. The joint forms, fitted with no single forms at all, beat P's own two answers.
- **The all-12 gain.** Reconciling only the singles gives −0.029 of the −0.070 all-12 gain. The joints give the other −0.041.

## 2. Balanced triangles (839 triangles, 16 signed statements, 6 weeks)

S = every single-event answer on the triangle's events that week: the weekly YES single, plus the balanced NO single
when the event was negated in any triangle that week. 76% of events have a NO answer. P = 3 signed pairs,
R = T + 3 siblings, C = 4 count cells, M = 2 mirrors.

| answers used | singles | pairs asked | pairs, all 4 signs | triples asked | triples, all 8 | full outcome |
|---|---|---|---|---|---|---|
| base (A with NOT A) | 0.620 | 0.512 | 0.504 | 0.349 | 0.334 | 1.841 |
| S+P | **−0.017 [−0.023, −0.011]** | −0.019 [−0.032, −0.006] | +0.026 [+0.018, +0.034] | **+0.027 [+0.015, +0.039]** | +0.033 [+0.025, +0.042] | +0.629 [+0.531, +0.724] |
| S+R | **−0.027 [−0.036, −0.018]** | **−0.033 [−0.045, −0.023]** | −0.014 [−0.021, −0.007] | −0.026 [−0.034, −0.019] | +0.015 [+0.009, +0.022] | +0.413 [+0.336, +0.484] |
| S+P+R | −0.029 [−0.039, −0.020] | −0.034 [−0.047, −0.021] | −0.018 [−0.025, −0.010] | −0.026 [−0.034, −0.018] | +0.011 [+0.004, +0.018] | +0.374 [+0.298, +0.446] |
| S+P+R+C | −0.030 [−0.043, −0.017] | −0.032 [−0.046, −0.019] | −0.018 [−0.027, −0.009] | −0.024 [−0.033, −0.016] | +0.000 [−0.006, +0.006] | +0.122 [+0.071, +0.173] |
| S+P+R+M | −0.031 [−0.040, −0.021] | −0.036 [−0.049, −0.023] | −0.015 [−0.023, −0.008] | −0.026 [−0.034, −0.018] | +0.012 [+0.006, +0.019] | +0.357 [+0.285, +0.426] |
| all (S+P+R+C+M) | **−0.032 [−0.045, −0.019]** | −0.034 [−0.048, −0.021] | −0.017 [−0.027, −0.009] | −0.025 [−0.033, −0.016] | +0.001 [−0.005, +0.007] | +0.131 [+0.079, +0.181] |
| S+C | −0.013 [−0.023, −0.002] | −0.004 [−0.015, +0.005] | +0.009 [+0.002, +0.016] | +0.013 [+0.008, +0.018] | +0.015 [+0.010, +0.019] | +0.163 [+0.129, +0.199] |
| S+M | −0.008 [−0.011, −0.004] | −0.022 [−0.031, −0.012] | +0.023 [+0.018, +0.028] | +0.023 [+0.015, +0.031] | +0.032 [+0.026, +0.037] | +0.456 [+0.398, +0.518] |
| all but S | −0.017 [−0.040, +0.005] | −0.030 [−0.047, −0.015] | −0.004 [−0.020, +0.013] | −0.022 [−0.032, −0.013] | +0.010 [−0.001, +0.020] | +0.034 [−0.039, +0.105] |
| all but P | −0.031 [−0.044, −0.018] | −0.033 [−0.046, −0.020] | −0.015 [−0.025, −0.006] | −0.024 [−0.032, −0.016] | +0.004 [−0.002, +0.010] | +0.158 [+0.107, +0.206] |
| all but R | −0.025 [−0.036, −0.015] | −0.021 [−0.034, −0.009] | +0.004 [−0.004, +0.011] | +0.006 [−0.001, +0.014] | +0.017 [+0.011, +0.022] | +0.289 [+0.233, +0.345] |
| P+R only | +0.018 [−0.003, +0.038] | −0.013 [−0.031, +0.004] | +0.025 [+0.009, +0.041] | −0.016 [−0.026, −0.005] | +0.029 [+0.019, +0.040] | +0.400 [+0.317, +0.490] |
| R+C only | −0.004 [−0.028, +0.020] | −0.017 [−0.034, −0.001] | +0.005 [−0.012, +0.023] | −0.018 [−0.029, −0.008] | +0.014 [+0.003, +0.026] | +0.055 [−0.021, +0.131] |
| P only | +0.072 [+0.051, +0.093] | +0.019 [+0.001, +0.038] | +0.049 [+0.033, +0.063] | +0.017 [+0.005, +0.029] | +0.035 [+0.025, +0.045] | +0.315 [+0.239, +0.391] |
| R only | +0.042 [+0.017, +0.066] | +0.017 [−0.003, +0.037] | +0.052 [+0.033, +0.070] | −0.006 [−0.017, +0.005] | +0.048 [+0.037, +0.060] | +0.647 [+0.545, +0.751] |
| fit16 of the balanced doc (signed singles only) | −0.018 [−0.033, −0.004] | −0.030 [−0.045, −0.016] | −0.007 [−0.018, +0.003] | −0.022 [−0.031, −0.014] | +0.010 [+0.003, +0.017] | +0.218 [+0.158, +0.275] |
| indep of stated signed singles | +0.089 [+0.079, +0.099] | +0.087 [+0.074, +0.099] | +0.062 [+0.055, +0.069] | +0.043 [+0.037, +0.048] | +0.037 [+0.033, +0.042] | +0.252 [+0.223, +0.279] |
| raw (as asked) | 0.714 signed / 0.668 YES | 0.549 | – | 0.390 | – | – |

Findings:
- **Pairs → singles** helps a little: −0.017 [−0.023, −0.011] on top of base, and −0.111 against the raw signed single.
- **Triples → singles** helps more: −0.027 [−0.036, −0.018]. Everything together gives −0.032 [−0.045, −0.019].
- **Triples → pairs** helps: −0.033 [−0.045, −0.023] on the asked pairs and −0.014 on all 12 signed pairs.
- **Pairs → triples** hurts: +0.027 [+0.015, +0.039]. Pairs alone also make the unasked sign combinations worse (+0.026).
- **Counts and mirrors** add little on top of S+P+R. Counts are the only level that stops the damage to the 8 atoms
  (+0.000) and to the full outcome (+0.12 instead of +0.37).
- **Full outcome log loss.** Every fit is worse than base. Reconciled independence (1.841) is the best
  8-outcome predictor here, better than independence of the stated signed singles (2.093). In Brier the full
  fit ties base (−0.007 [−0.021, +0.008]).
- **Higher-order answers alone, with no singles.** R+C alone ties base on the singles (−0.004 [−0.028, +0.020]).
  All but S is −0.017 [−0.040, +0.005]. Either is far better than the raw signed single (0.714).

Same event vs cross event, per focal event (all joints P+R+C+M; the P+R-only numbers are in the JSON):

| step | Δ log loss on P's single [95% CI] |
|---|---|
| raw signed → base (A with NOT A) | −0.093 [−0.103, −0.082]; vs raw YES 0.668: −0.046 [−0.066, −0.027] |
| base → own + joints | **−0.030 [−0.042, −0.019]** |
| + Q's and R's singles (cross-event) | **−0.004 [−0.008, −0.000]** |
| swap Q, R for random same-week events (full fit) | −0.001 [−0.005, +0.002] (with P+R joints: −0.006 [−0.008, −0.003]) |
| leave-P-out, joints + Q, R: vs base | −0.002 [−0.024, +0.020] (joints alone: −0.019 [−0.042, +0.004]) |
| leave-P-out: vs raw YES single | −0.065 (joints alone) [−0.097, −0.034] |
| swap Q, R in leave-P-out | +0.007 [−0.002, +0.015]; with P+R joints −0.010 [−0.017, −0.003] |

- **Where the singles gain comes from.** The cross-event share is about 0.004 nats, and it is not specific to Q's
  content: random events' singles do as well.
- **The structure does connect Q to P for some relation types.** For alternatives the swap in leave-P-out costs
  +0.025 [+0.013, +0.038], and for cause/effect +0.027 [+0.002, +0.051]. For related-topic it costs −0.004
  [−0.017, +0.010] and for same-quantity −0.001.
- **A caveat on base rates.** The model's singles are worse than a constant base rate on these events (0.520
  leave-week-out, vs base 0.620). No fit reaches it, so a "recovery share" is undefined here.

Splits by week and relation type (`splits` in `balanced.json`):
- Pairs → singles and triples → singles are negative in every week. Their CIs cross 0 in 05-12 (and 08-04 for pairs).
- By relation type, the gains come from related-topic and same-quantity (all → singles −0.052 and −0.058).
  They are about 0 for alternatives (+0.012) and cause/effect (+0.004).

## 3. The weekly AND-only run (48,563 triangles, 35 weeks)

Refit per triangle (not the global M1/M3 fits); the global M1/M3/M3b predictions appear as reference rows on the levels they
cover. Only YES singles were asked here, so base = raw singles = independence of the stated singles. P = the 3 positive
ANDs, R = the positive triple.

| answers used | singles | ANDs asked | pairs, all 4 signs | triple asked | triples, all 8 | full outcome |
|---|---|---|---|---|---|---|
| base (= raw singles = independence) | 0.653 | 0.390 | 0.522 | 0.246 | 0.344 | 1.920 |
| S+P | **−0.018 [−0.025, −0.011]** | +0.042 [+0.030, +0.054] | +0.011 [+0.004, +0.018] | +0.079 [+0.067, +0.092] | +0.014 [+0.007, +0.021] | +0.493 [+0.414, +0.572] |
| S+R | **−0.006 [−0.009, −0.003]** | **+0.032 [+0.024, +0.039]** | −0.005 [−0.011, −0.000] | +0.063 [+0.054, +0.073] | −0.008 [−0.012, −0.004] | +0.089 [+0.044, +0.132] |
| S+P+R | −0.017 [−0.025, −0.010] | +0.035 [+0.023, +0.047] | +0.002 [−0.004, +0.008] | +0.064 [+0.054, +0.075] | +0.012 [+0.006, +0.019] | +0.482 [+0.401, +0.563] |
| P only | +0.149 [+0.131, +0.168] | +0.071 | +0.112 | +0.029 | +0.071 | +0.493 |
| R only | +0.124 [+0.100, +0.147] | +0.090 | +0.088 | +0.085 | +0.059 | +0.296 |
| P+R only | +0.123 [+0.105, +0.141] | +0.067 | +0.095 | +0.056 | +0.068 | +0.558 |
| global M1 (weekly doc) | – | 0.444 (+0.054) | – | 0.279 (+0.033) | – | – |
| global M3 | – | 0.444 (+0.053) | – | 0.280 (+0.034) | – | – |
| global M3b | – | 0.396 (+0.005) | – | 0.258 (+0.012) | – | – |
| raw ANDs as asked | – | 0.481 (+0.091) | – | 0.336 (+0.090) | – | – |

Brier agrees: singles S+P −0.0049 [−0.0071, −0.0028], S+R −0.0010; ANDs asked S+R +0.0101; triple asked S+P +0.0212.

- **Pairs → singles** helps: −0.018 [−0.025, −0.011]. **Triples → singles** barely: −0.006 [−0.009, −0.003].
- **Triples → pairs hurts** here: +0.032 [+0.024, +0.039] on the asked ANDs, unlike the balanced run. This AND-only
  model's joints are worse than independence of its own singles (raw ANDs 0.481 vs 0.390), so pulling toward them
  costs; it is the same finding as the weekly doc (independence beats every structural fit on the joints).
- **Joints alone** are far worse on the singles (+0.12 to +0.15): with only positive ANDs, the joints say little about P
  by themselves, unlike the 12-form and balanced data where joints of both signs restate P.

Same event vs cross event (P+R joints, focal event's single):

| step | Δ log loss [95% CI] |
|---|---|
| raw → base | 0 (only one answer per event) |
| base → own + joints | −0.003 [−0.012, +0.005] |
| + Q's and R's singles (cross-event) | **−0.018 [−0.021, −0.015]** |
| swap Q, R for random same-week events (full fit) | **−0.006 [−0.008, −0.004]** (random singles do better) |
| leave-P-out, joints + Q, R vs raw P | +0.179 [+0.151, +0.206]; joints alone +0.120; vs base rate (0.490) +0.345 |
| swap Q, R in leave-P-out | −0.116 [−0.131, −0.104] (random singles do much better) |

- Here there *is* a cross-event gain (−0.018), in 33 of 35 weeks and in every relation type, but it is **not Q's
  content**: random other events' singles give a slightly larger gain. The other singles only fix how much of each
  low AND answer is blamed on Q rather than on P.
- Leave-P-out does not recover P at all (worse than the base rate by 0.35).
- By week: S+P and S+R → singles are significantly better in 20 of 35 weeks (worse in 4 and 1); S+R → ANDs is
  significantly worse in 25 weeks. By relation: same-quantity gains most (S+P −0.080 [−0.126, −0.040]); cause/effect
  S+P is worse (+0.009 [+0.004, +0.013]); the swap test is negative (random better) in every relation type.

## Reading

- **What consistency buys the singles.** In every data set, it is the model's other readings of the same event:
  the NOT form, and joint statements that restate the event. Q's answer does not reach P through the fitted joint.
  The one exception is small: the leave-P-out gain in alternatives and cause/effect triangles.
- **Joints vs full outcomes.** Joints help the singles and the asked joints. They do not help the full outcome.
  Triples help the pairs, but pairs hurt the triples.
- **Caveats.**
  - Everything is scored on the weeks the answers came from. Outcomes are used only for scoring (and for the
    leave-week-out base rate).
  - Units share events, and the event-cluster bootstrap handles that.
  - Balanced "base" uses up to two answers per event (YES from the weekly run, NO from the balanced run). The weekly
    run has only the YES answer, so there base = raw.
