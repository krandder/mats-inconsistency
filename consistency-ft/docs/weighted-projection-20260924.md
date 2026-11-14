# Better coherence projections: bias-corrected, reliability-weighted and dependence-preserving (24 Sep 2026)

This follows up `docs/structured-pairs-20260924.md`. On structured pairs the models' raw one-form
answers respect the known relations: the implication order is right on 87–99% of pairs, and the
exclusive AND is ≤ 0.05 on 94–96%. But the equal-weight KL coherence projection
(`scripts/coherence_projection.py`) washes most of that dependence out.

The hypothesis was that two kinds of forms overwhelm the informative ones in an equal-weight fit:
- forms with a systematic bias (AND too high, OR too low; `docs/averaging-vs-structure-20260924.md`);
- noisy forms.

No API calls were made; everything here is CPU work on existing answers.

## Result in one paragraph

**The hypothesis is only half right.** The dependence is lost because the projection pools all 12
forms. Fixing the bias or the weights does not fix this. Reading the dependence from the AND
answers alone does.

- **Bias correction and inverse-variance weights do not help (1, 2).** Joint log loss on the
  ladder types moves by −0.010 to +0.025 against the equal-weight projection. It is mostly
  worse, and the captured dependence falls: for 32B Q threshold pairs the median D goes from
  0.31 to 0.13. The reason is that the bias is estimated as each form's mean residual from the
  coherent fit. On related pairs, the AND answer really is high (P(A and B) ≈ P(A)), so the
  "bias" absorbs the real dependence, and subtracting it removes that dependence. These two
  methods help only on exclusive pairs (−0.005 to −0.015 in the hybrid).
- **Taking the dependence from the AND answers works (3, 4).** `h_and4` joins the market
  marginals with the odds ratio of the model's 4 AND answers, bias-corrected and normalized.
  It has no label-tuned parameters. On three of the four runs (32B P, 235B Q, 235B P) it
  recovers 60–68% of the known relation's gain, where the old hybrid recovers 41–57%. Against
  the old hybrid, paired: −0.003 to −0.009 on all pairs, CIs below 0. It costs nothing on the
  unrelated controls (−0.002 to +0.002).
- **The raw 3-form odds ratio (`h_and3`) recovers about as much** (49–77%), and more on dates.
  But it invents dependence: up to +0.019 on the controls, and +0.006 to +0.020 on the
  market-context pairs.
- **32B question-only gains nothing.** Its AND answers on ladders are too noisy, and no variant
  beats the old hybrid there (41–52% against 49%).
- **A label-tuned scale on the odds ratio (λ) learns the relation mix, not the model.** On
  structured pairs, which are 72% related by construction, the tuned λ is 1.2–3.0, which
  amplifies. That gets 81–93% for 235B. On the market-context pairs the tuned λ is 0–0.7,
  which shrinks. A single λ does not transfer between the two, so the tuned variants (`_s`) are
  upper bounds for this pair mix, not a method to deploy.
- **Market-context zero-shot pairs: still a tie.** No variant beats market prices +
  independence; every hybrid is within ±0.003, except h_and3, which loses.
- **Old 1,066-pair test (question only): no variant beats the equal-weight KL projection.**
  Every difference is within +0.005 / −0.004; bias correction is +0.002 to +0.005 worse. So
  equal-weight KL stays the right model-only projection when there are no market prices.
- **Model-only with evidence packs.** Weighting (γ tuned) and the dependence-preserving fill
  improve projected joints by 0.02–0.03 on structured pairs and about 0.01 on market-context.
  On the question-only arms they are neutral to 0.015 worse. The one exception is 235B Q
  structured, where fill_and4 is −0.007.

## Headline table: structured pairs, joint forms, Δ vs market + independence

All 1,077 pairs (235B P: its 170-pair subset). Log loss difference with 95% cluster-bootstrap
CIs. "Rec." is the share of the known relation's gain recovered:
(indep − method) / (indep − known), with the controls included in the pool.

| run | market + known | old hybrid h_eq (rec.) | h_rw (rec.) | **h_and4** (rec.) | h_and3 (rec.) | h_and4_s, λ tuned (rec.) | h_and4 − h_eq | h_and4 on controls |
|---|---|---|---|---|---|---|---|---|
| 32B Q | −0.037 [−0.043, −0.030] | −0.018 (49%) | −0.015 (41%) | −0.016 [−0.021, −0.011] (44%) | −0.018 (49%) | −0.016 (45%) | +0.002 [−0.001, +0.005] | +0.002 [−0.002, +0.005] |
| 32B P | −0.037 | −0.015 (41%) | −0.015 (42%) | **−0.022 [−0.026, −0.017] (60%)** | −0.028 (76%) | −0.024 (66%) | −0.007 [−0.010, −0.004] | −0.001 [−0.004, +0.002] |
| 235B Q | −0.037 | −0.021 (57%) | −0.019 (52%) | **−0.024 [−0.028, −0.020] (66%)** | −0.025 (69%) | −0.029 (81%) | −0.003 [−0.005, −0.001] | −0.001 [−0.003, +0.000] |
| 235B P (170) | −0.043 [−0.056, −0.030] | −0.020 (46%) | −0.024 (55%) | **−0.029 [−0.037, −0.021] (68%)** | −0.033 (77%) | −0.040 (93%) | −0.009 [−0.015, −0.005] | −0.002 [−0.010, +0.005] |
| h_and3 on controls | | | | | | | | +0.019, +0.005, +0.011, −0.000 (32B Q, 32B P, 235B Q, 235B P) |

Market-context zero-shot (723 test pairs, joint Δ vs market independence 0.313): h_eq −0.000 to
+0.002, h_and4 +0.001 to +0.003, h_and4_s −0.000 to +0.001, h_and3 +0.006 to +0.020.

## Recovery by relation type (joint; old hybrid → h_and4 → h_and3)

| run | threshold | date | mc_excl |
|---|---|---|---|
| 32B Q | 38% → 17% → 22% | 32% → 20% → 67% | 65% → 79% → 89% |
| 32B P | 29% → 47% → 60% | 18% → 33% → 72% | 60% → 83% → 97% |
| 235B Q | 62% → 57% → 83% | 58% → 46% → 78% | 51% → 81% → 72% |
| 235B P (170) | 59% → 66% → 92% | 45% → 35% → 95% | 32% → 76% → 56% |

- **Exclusive pairs:** the 4 normalized AND answers are the best source. The model's
  P(A and B) ≈ 0 is kept exactly, and the AND-based sources beat the old hybrid on every run
  (paired, h_and4 − h_eq: −0.008 to −0.029).
- **Date ladders:** the raw 3-form ratio (P, Q, P-and-Q) is the best source. The 4-AND source
  also uses the answers to (A and not B) and (not A and not B), and those break the implication
  on date pairs: the zero cell ≤ 0.05 in only 27–41% of raw answers.
- **Threshold ladders:** in between.

## What was built (all cross-fitted)

`scripts/weighted_projection.py` holds the methods, the cross-fitting and the scoring.
`scripts/batch_projection.py` is a vectorized damped-Newton projector. Its equal-weight KL
answers equal `coherence_projection.fit` to 1e-13 in loss and 2e-7 in atoms on 400 pairs.

1. **Bias-corrected (`bc_kl`).** Targets are logit(stated) − b_f, then equal-weight KL. The
   per-form log-odds bias b is label-free: iterate b ← b + (mean residual of the log-odds LS
   projection of logit − b), 4 rounds. On the 32B old test it reproduces the
   averaging-vs-structure pattern: AND +0.09 to +0.55 and OR −0.08 to −0.17 in log-odds.
   **`bcx_kl`** uses b estimated once on that external data set (32B t0, 1,066 old test pairs)
   and applies it everywhere else.
2. **Reliability-weighted (`rw_kl`, `rw_lsl`, `rwg_kl`).** w_f = 1/σ²_f, where σ²_f is the mean
   squared residual after bias correction (incoherence plus sampling noise, the c + a of the
   variance decomposition). The weights are used in KL (`rw_kl`) or log-odds least squares
   (`rw_lsl`). `rwg_kl` uses w^γ, with γ ∈ {0, .5, 1, 2, 3} tuned on the training folds.
   `rwx_kl` uses the external b and σ².
3. **Dependence-preserving (`fill_*`).** The marginals come from `rw_kl`. The log odds ratio
   comes from the most informative forms, scaled by λ (tuned on the training folds, grid 0–3):
   - `and3`: raw P, Q and P-and-Q, with the AND clipped into its Fréchet range;
   - `and4`: the 4 AND answers, bias-corrected and normalized to sum 1;
   - `litand`: `rw_kl` fitted on the 4 literals + 4 ANDs only.

   No conditional forms were asked, so a conditional-based odds ratio is not available.
4. **Market hybrids (`h_*`).** Market marginals (price at t; normalized for mc_excl) joined with
   the log odds ratio from each of 1–3:
   - `h_eq` is the old "hybrid";
   - `h_and3` is the old "hybrid_and3";
   - `h_and3/4/litand` use λ = 1 (no labels); the `_s` variants use a tuned λ.

**Protocol.** There are 5 folds, split by cluster: ladder group, MC market or control pair for
structured pairs, and pair elsewhere. b and σ² come from the other 4 folds, even though they are
label-free. γ and λ are also tuned on the other 4 folds, with joint log loss as the target. No
parameter is fitted on the pairs it scores. No method uses the pair's relation type.
Metrics: mean per-pair log loss on the 8 joint forms and on all 12, clip 0.01. CIs: paired
cluster bootstrap, 2,000 resamples, seed 20260924.

## Model-only projections (no market): Δ vs equal-weight KL, joint forms, all pairs

| data set | run | eq_kl joint | eq_lsl | bc_kl | rw_kl | rw_lsl | rwg_kl | bcx_kl / rwx_kl | fill_and3 | fill_and4 |
|---|---|---|---|---|---|---|---|---|---|---|
| structured | 32B Q | 0.501 | +0.005 | +0.010 | +0.009 | +0.015 | +0.009 | +0.005 / +0.003 | +0.002 | +0.009 |
| structured | 32B P | 0.353 | −0.007 | +0.003 | −0.011 | −0.016 | **−0.024 [−0.029, −0.018]** | +0.003 / +0.006 | −0.022 | −0.018 |
| structured | 235B Q | 0.482 | +0.001 | +0.003 | +0.003 | +0.002 | +0.004 | +0.001 / +0.000 | +0.010 | **−0.007 [−0.012, −0.002]** |
| structured | 235B P (170) | 0.317 | −0.010 | −0.001 | −0.015 | −0.026 | −0.031 | +0.000 / −0.001 | −0.027 | **−0.032 [−0.043, −0.021]** |
| market ctx | 32B Q | 0.488 | −0.001 | +0.009 | +0.007 | +0.010 | +0.008 | +0.007 / +0.008 | +0.011 | +0.007 |
| market ctx | 32B P | 0.363 | −0.002 | +0.006 | −0.003 | −0.008 | **−0.009 [−0.017, −0.001]** | +0.003 / +0.008 | −0.006 | −0.002 |
| market ctx | 32B L | 0.386 | +0.002 | +0.006 | +0.000 | −0.003 | −0.002 | +0.003 / +0.003 | −0.006 | −0.003 |
| market ctx | 235B L | 0.349 | **−0.009 [−0.012, −0.006]** | +0.003 | −0.000 | −0.009 | −0.002 | +0.002 / +0.009 | −0.006 | +0.000 |
| old 1,066 | 32B Q (t0) | 0.504 | +0.001 | +0.005 | +0.001 | +0.002 | +0.001 | (in-sample, not run) | −0.004 [−0.009, +0.001] | +0.001 |
| old 1,066 | 235B Q | 0.483 | +0.001 | +0.002 | +0.002 | +0.002 | +0.002 | +0.002 / +0.004 | +0.000 | +0.001 |

All-12 moves the same way, within 0.006 of the joint numbers (appendix). Against raw, the
equal-weight projection is 0.04–0.18 better on joints, as before.

- **Where the weighting helps (P, L and 235B P arms).** It downweights the negated-OR forms,
  and on P arms it upweights the literals. The tuned γ goes to 3 there. With context, the
  literals copy the market price, so they are the most reliable forms.
- **Question-only arms and the old test.** There the ANDs are the least noisy forms. Weighting
  and bias correction change little or hurt.
- **The gauge problem.** The bias is identified only up to the coherent subspace, and on
  structured pairs it is confounded with dependence. That is why `bc_kl` is never better than
  equal weights on ladders.

## Dependence captured (D, median / share ≥ 0.5; own marginals; 1 = the known relation)

| run | type | raw | eq_kl | bc_kl | rw_kl | fill_and4 (λ tuned) |
|---|---|---|---|---|---|---|
| 32B Q | threshold | 1.00 / 0.62 | 0.31 / 0.33 | 0.13 / 0.18 | 0.15 / 0.19 | 0.18 / 0.15 |
| 32B Q | date | 1.00 / 0.74 | 0.21 / 0.15 | 0.04 / 0.07 | 0.07 / 0.08 | 0.10 / 0.07 |
| 32B Q | mc_excl | 1.00 / 0.92 | 0.67 / 0.68 | 0.73 / 0.77 | 0.76 / 0.77 | 0.69 / 0.70 |
| 32B P | threshold | 1.00 / 0.77 | 0.26 / 0.19 | 0.14 / 0.11 | 0.21 / 0.13 | 0.54 / 0.51 |
| 32B P | date | 1.00 / 0.83 | 0.20 / 0.06 | 0.12 / 0.03 | 0.17 / 0.05 | 0.46 / 0.47 |
| 32B P | mc_excl | 1.00 / 0.94 | 0.42 / 0.43 | 0.55 / 0.55 | 0.58 / 0.57 | 0.98 / 0.88 |
| 235B Q | threshold | 1.00 / 0.83 | 0.46 / 0.45 | 0.29 / 0.23 | 0.34 / 0.31 | 0.80 / 0.67 |
| 235B Q | date | 1.89 / 0.84 | 0.46 / 0.41 | 0.33 / 0.27 | 0.30 / 0.27 | 0.78 / 0.71 |
| 235B Q | mc_excl | 0.85 / 0.87 | 0.49 / 0.47 | 0.53 / 0.58 | 0.53 / 0.54 | 0.99 / 0.93 |

The pattern is the same everywhere:
- Bias correction and weighting cut ladder dependence further, and raise exclusive dependence
  a little.
- Reading the odds ratio from the ANDs restores most of it: to 0.5–1.0 when λ is tuned, except
  for 32B Q, whose ladder ANDs are too noisy.
- The fill (D) numbers use the tuned λ, which is inflated on this pair mix (see below). The
  λ = 1 hybrid numbers in the headline table are the clean ones.

## Fitted parameters (mean over training folds)

Log-odds bias b and weights 1/σ², normalized to mean 1. The full table is in the appendix and the
chart JSON.

- **External (32B, old test).**
  - b: AND forms +0.41, +0.22, +0.09, +0.55; OR forms −0.17, −0.10, −0.10, −0.08; literals
    within ±0.08.
  - Weights: P∧Q 1.75 (the most reliable form); ¬P and ¬Q 0.42–0.46 (the least).
- **Structured.** The bias estimates differ a lot between runs. For example, ¬P∨¬Q is −1.13
  (235B P) and P∨¬Q is +0.88. That instability is itself a sign that b absorbs pair-specific
  structure rather than a fixed per-form error.
- **Tuned γ.**
  - 3.0 on 32B P, 235B P, market ctx 32B P and 235B L;
  - 0–1 on the question-only arms, the 235B old test (0) and the 32B old test (1).
- **Tuned λ.**

  | data set | and3 | and4 | litand |
  |---|---|---|---|
  | structured (hybrid) | 0.3–1.4; 3.0 on 235B P | 1.2–3.0 (235B P at the grid edge, 3.0) | same range as and4 |
  | market ctx (hybrid) | 0–0.2 | 0–0.6 | 0–0.7 |
  | old test (fill) | 0.1 | 0.0–0.9 | 0.1–0.9 |

## Caveats

- **The λ-tuned variants depend on the pair mix.** Cross-fitting stops them from fitting the
  pairs they score, but it does not stop them from learning "most pairs in this set are
  related". The structured set is 72% related by construction. Deploying a λ would need a
  relatedness gate, or a λ fitted on a representative pair mix. The headline uses λ = 1.
- **h_and4 uses the bias b.** b is label-free, but it is estimated on the same run's training
  folds. `h_and4` with the external b was not run separately. The correction shifts the log
  odds ratio by about −(b_PQ + b_¬P¬Q − b_P¬Q − b_¬PQ). That sum is +0.65 for the external b and
  −0.68 to +0.88 across the structured runs, so b matters. The largest value, +0.88, is on
  32B Q, the run where h_and4 does worst; there b probably absorbs real dependence again.
- **235B P is a 170-pair subset.** Its λ and γ hit the grid edges (3.0), and its CIs are wide.
- **The market check is unchanged.** The market prices are almost always coherent, so every
  gain here is about dependence, not about fixing market incoherence.
- **No conditional forms were asked,** so "project with the conditional" could not be tested.
- **The old 1,066-pair test has no market prices at t,** so only model-only methods are scored
  there.

## Next steps suggested by this

- **Use `h_and4` (λ = 1) as the default market hybrid.** It is never worse than the old hybrid
  on controls or market-context pairs (+0.001 to +0.003), and it is better on structured pairs
  for three of the four runs.
- **For date ladders, prefer the 3-form odds ratio,** or ask a conditional form P(B | A)
  directly.
- **Gate the odds ratio by a relatedness score** before trusting λ > 1. One candidate is the
  model's own |log OR| from the ANDs, calibrated on a representative validation mix.

## Files

- Code:
  - `scripts/batch_projection.py` (vectorized weighted projector);
  - `scripts/weighted_projection.py` (methods, cross-fitting, scoring);
  - `scripts/tables_weighted_projection.py` (tables and chart JSON).
- Data: `/home/agent/data/weighted-projection-20260924/`:
  - `full.json` (every stratum × method × reference, with CIs);
  - `tables.md`;
  - `run.log`.
- Chart JSON: `docs/weighted-projection-20260924-charts.json`. It has a flat `rows` list
  (dataset, job, stratum, method, joint and all-12 LL, Δ vs raw, eq_kl, market_indep,
  market_known and h_eq with CIs, and recovery), plus the fitted parameters, the dependence
  captured and the control log odds ratios for each job.
- Inputs:
  - `/home/agent/data/structured-pairs-20260924/logs`;
  - `/home/agent/data/market-context-zeroshot-20260924/logs/*-test.jsonl`;
  - `/home/agent/data/decomposition-32b-20260924/log.jsonl` (sample t0);
  - `/home/agent/data/precutoff-baseline-20260923/per-form/predictions-test.jsonl`.
- Progress log: `/home/agent/workspaces/calibrated-ai/structured-pairs-progress.log`.

## Appendix: all tables (generated by `scripts/tables_weighted_projection.py`)

### structured: model-only methods, Δ vs equal-weight KL projection (joint / all-12)

| job | stratum | eq_kl joint | eq_lsl | bc_kl | rw_kl | rw_lsl | rwg_kl | bcx_kl | rwx_kl | fill_and3 | fill_and4 | fill_litand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 235B P | all (170) | 0.317 | -0.010 [-0.016, -0.004] / -0.011 | -0.001 [-0.005, +0.003] / -0.001 | -0.015 [-0.021, -0.010] / -0.014 | -0.026 [-0.033, -0.019] / -0.026 | -0.031 [-0.041, -0.021] / -0.030 | +0.000 [-0.002, +0.003] / +0.001 | -0.001 [-0.004, +0.003] / -0.000 | -0.027 [-0.038, -0.015] / -0.022 | -0.032 [-0.043, -0.021] / -0.026 | -0.032 [-0.042, -0.021] / -0.026 |
| 235B P | control (44) | 0.321 | -0.010 [-0.019, -0.000] / -0.010 | -0.009 [-0.017, -0.000] / -0.008 | -0.014 [-0.025, -0.004] / -0.014 | -0.027 [-0.039, -0.015] / -0.028 | -0.025 [-0.044, -0.007] / -0.026 | -0.003 [-0.009, +0.002] / -0.003 | +0.002 [-0.004, +0.009] / +0.004 | -0.010 [-0.027, +0.007] / -0.011 | -0.014 [-0.029, +0.003] / -0.014 | -0.013 [-0.029, +0.003] / -0.013 |
| 235B P | date (27) | 0.390 | -0.006 [-0.021, +0.011] / -0.008 | +0.005 [-0.005, +0.013] / +0.004 | -0.010 [-0.024, +0.004] / -0.012 | -0.016 [-0.033, +0.001] / -0.020 | -0.023 [-0.052, +0.011] / -0.024 | +0.000 [-0.007, +0.008] / -0.001 | -0.007 [-0.019, +0.005] / -0.011 | -0.020 [-0.061, +0.028] / -0.019 | -0.013 [-0.049, +0.032] / -0.014 | -0.016 [-0.051, +0.029] / -0.016 |
| 235B P | mc_excl (49) | 0.358 | -0.015 [-0.030, +0.002] / -0.017 | -0.007 [-0.013, -0.002] / -0.004 | -0.030 [-0.042, -0.019] / -0.025 | -0.044 [-0.062, -0.025] / -0.039 | -0.051 [-0.072, -0.030] / -0.047 | -0.003 [-0.006, +0.001] / -0.001 | -0.011 [-0.017, -0.004] / -0.008 | -0.036 [-0.059, -0.012] / -0.029 | -0.062 [-0.086, -0.037] / -0.046 | -0.063 [-0.086, -0.038] / -0.047 |
| 235B P | threshold (50) | 0.233 | -0.008 [-0.013, -0.004] / -0.008 | +0.010 [+0.004, +0.016] / +0.007 | -0.004 [-0.010, +0.004] / -0.006 | -0.013 [-0.021, -0.006] / -0.015 | -0.021 [-0.032, -0.009] / -0.022 | +0.007 [+0.004, +0.010] / +0.006 | +0.010 [+0.006, +0.014] / +0.009 | -0.036 [-0.045, -0.026] / -0.027 | -0.030 [-0.041, -0.019] / -0.023 | -0.027 [-0.038, -0.016] / -0.021 |
| 235B Q | all (1077) | 0.482 | +0.001 [-0.001, +0.003] / +0.001 | +0.003 [+0.001, +0.005] / +0.002 | +0.003 [+0.001, +0.006] / +0.002 | +0.002 [-0.001, +0.005] / +0.002 | +0.004 [+0.002, +0.006] / +0.003 | +0.001 [-0.000, +0.002] / +0.000 | +0.000 [-0.003, +0.003] / +0.001 | +0.010 [+0.003, +0.017] / +0.006 | -0.007 [-0.012, -0.002] / -0.005 | -0.004 [-0.008, +0.001] / -0.003 |
| 235B Q | control (297) | 0.475 | +0.001 [-0.001, +0.002] / +0.001 | +0.001 [-0.004, +0.006] / +0.002 | -0.000 [-0.004, +0.004] / -0.000 | -0.001 [-0.005, +0.003] / -0.001 | +0.001 [-0.004, +0.005] / +0.001 | +0.000 [-0.002, +0.003] / +0.001 | +0.004 [-0.000, +0.008] / +0.005 | +0.034 [+0.016, +0.053] / +0.023 | +0.005 [-0.001, +0.012] / +0.004 | +0.007 [+0.001, +0.014] / +0.005 |
| 235B Q | date (180) | 0.514 | -0.003 [-0.009, +0.003] / -0.001 | +0.012 [+0.008, +0.017] / +0.007 | +0.018 [+0.012, +0.023] / +0.013 | +0.020 [+0.013, +0.027] / +0.015 | +0.016 [+0.011, +0.022] / +0.011 | +0.005 [+0.002, +0.008] / +0.002 | -0.006 [-0.018, +0.006] / -0.010 | -0.007 [-0.022, +0.011] / -0.004 | -0.004 [-0.015, +0.007] / -0.002 | -0.001 [-0.011, +0.010] / -0.000 |
| 235B Q | mc_excl (300) | 0.510 | +0.002 [-0.002, +0.006] / +0.002 | -0.011 [-0.015, -0.008] / -0.009 | -0.012 [-0.016, -0.008] / -0.010 | -0.020 [-0.025, -0.015] / -0.015 | -0.011 [-0.014, -0.007] / -0.009 | -0.007 [-0.010, -0.005] / -0.006 | -0.011 [-0.014, -0.007] / -0.008 | +0.011 [+0.005, +0.018] / +0.006 | -0.026 [-0.036, -0.014] / -0.019 | -0.022 [-0.030, -0.014] / -0.016 |
| 235B Q | threshold (300) | 0.440 | +0.002 [-0.003, +0.006] / +0.003 | +0.014 [+0.011, +0.018] / +0.010 | +0.014 [+0.009, +0.019] / +0.010 | +0.017 [+0.010, +0.026] / +0.012 | +0.015 [+0.011, +0.019] / +0.011 | +0.007 [+0.005, +0.009] / +0.005 | +0.011 [+0.005, +0.018] / +0.012 | -0.005 [-0.015, +0.005] / -0.003 | -0.003 [-0.011, +0.006] / -0.002 | +0.001 [-0.007, +0.010] / +0.001 |
| 32B P | all (1077) | 0.353 | -0.007 [-0.011, -0.002] / -0.006 | +0.003 [+0.002, +0.004] / +0.003 | -0.011 [-0.013, -0.008] / -0.012 | -0.016 [-0.021, -0.012] / -0.017 | -0.024 [-0.029, -0.018] / -0.025 | +0.003 [+0.002, +0.004] / +0.002 | +0.006 [+0.004, +0.008] / +0.006 | -0.022 [-0.028, -0.016] / -0.019 | -0.018 [-0.023, -0.012] / -0.016 | -0.018 [-0.023, -0.012] / -0.016 |
| 32B P | control (297) | 0.347 | -0.010 [-0.016, -0.001] / -0.011 | +0.001 [-0.002, +0.004] / +0.000 | -0.017 [-0.023, -0.011] / -0.019 | -0.027 [-0.036, -0.017] / -0.029 | -0.032 [-0.042, -0.021] / -0.034 | +0.001 [-0.000, +0.003] / +0.001 | +0.008 [+0.004, +0.011] / +0.009 | -0.016 [-0.025, -0.007] / -0.018 | -0.017 [-0.025, -0.010] / -0.019 | -0.017 [-0.024, -0.009] / -0.018 |
| 32B P | date (180) | 0.355 | -0.009 [-0.015, -0.003] / -0.007 | +0.009 [+0.007, +0.012] / +0.007 | -0.007 [-0.012, -0.001] / -0.008 | -0.008 [-0.018, +0.001] / -0.010 | -0.025 [-0.035, -0.011] / -0.024 | +0.006 [+0.003, +0.008] / +0.003 | +0.008 [+0.003, +0.012] / +0.005 | -0.027 [-0.040, -0.012] / -0.022 | -0.017 [-0.029, -0.005] / -0.015 | -0.016 [-0.030, -0.003] / -0.015 |
| 32B P | mc_excl (300) | 0.377 | -0.016 [-0.022, -0.009] / -0.014 | -0.005 [-0.007, -0.002] / -0.003 | -0.014 [-0.019, -0.010] / -0.013 | -0.029 [-0.037, -0.020] / -0.026 | -0.021 [-0.030, -0.012] / -0.020 | -0.000 [-0.002, +0.002] / +0.001 | -0.002 [-0.005, +0.001] / -0.000 | -0.023 [-0.033, -0.012] / -0.019 | -0.018 [-0.029, -0.004] / -0.015 | -0.021 [-0.032, -0.008] / -0.017 |
| 32B P | threshold (300) | 0.333 | +0.007 [-0.004, +0.020] / +0.007 | +0.009 [+0.007, +0.012] / +0.007 | -0.004 [-0.009, +0.002] / -0.006 | +0.002 [-0.008, +0.014] / -0.000 | -0.019 [-0.031, -0.007] / -0.020 | +0.006 [+0.004, +0.008] / +0.004 | +0.012 [+0.009, +0.016] / +0.011 | -0.024 [-0.036, -0.009] / -0.019 | -0.019 [-0.030, -0.008] / -0.016 | -0.017 [-0.029, -0.006] / -0.015 |
| 32B Q | all (1077) | 0.501 | +0.005 [+0.001, +0.009] / +0.007 | +0.010 [+0.007, +0.012] / +0.007 | +0.009 [+0.006, +0.012] / +0.007 | +0.015 [+0.010, +0.021] / +0.013 | +0.009 [+0.006, +0.012] / +0.007 | +0.005 [+0.003, +0.006] / +0.004 | +0.003 [-0.000, +0.006] / +0.003 | +0.002 [-0.006, +0.009] / +0.002 | +0.009 [+0.005, +0.013] / +0.006 | +0.009 [+0.005, +0.013] / +0.006 |
| 32B Q | control (297) | 0.489 | -0.002 [-0.006, +0.001] / -0.002 | +0.007 [+0.002, +0.013] / +0.007 | +0.005 [-0.001, +0.010] / +0.004 | +0.003 [-0.003, +0.010] / +0.003 | +0.006 [+0.001, +0.011] / +0.006 | +0.004 [+0.001, +0.006] / +0.004 | +0.006 [+0.001, +0.011] / +0.005 | +0.023 [+0.007, +0.038] / +0.016 | +0.003 [-0.002, +0.009] / +0.003 | +0.004 [-0.002, +0.009] / +0.003 |
| 32B Q | date (180) | 0.558 | +0.006 [-0.002, +0.014] / +0.006 | +0.025 [+0.019, +0.030] / +0.015 | +0.024 [+0.017, +0.032] / +0.016 | +0.026 [+0.014, +0.038] / +0.017 | +0.024 [+0.018, +0.030] / +0.015 | +0.010 [+0.007, +0.013] / +0.006 | -0.003 [-0.012, +0.007] / -0.006 | -0.036 [-0.054, -0.019] / -0.025 | +0.018 [+0.006, +0.031] / +0.012 | +0.013 [+0.003, +0.025] / +0.008 |
| 32B Q | mc_excl (300) | 0.513 | +0.008 [+0.001, +0.016] / +0.012 | -0.008 [-0.012, -0.004] / -0.006 | -0.009 [-0.014, -0.004] / -0.007 | -0.006 [-0.015, +0.003] / -0.001 | -0.008 [-0.013, -0.004] / -0.006 | -0.003 [-0.005, -0.001] / -0.002 | -0.004 [-0.008, +0.000] / -0.001 | -0.012 [-0.021, -0.002] / -0.009 | -0.008 [-0.014, -0.001] / -0.006 | -0.001 [-0.008, +0.006] / -0.002 |
| 32B Q | threshold (300) | 0.468 | +0.009 [-0.002, +0.021] / +0.010 | +0.020 [+0.016, +0.025] / +0.014 | +0.023 [+0.016, +0.030] / +0.018 | +0.042 [+0.027, +0.059] / +0.035 | +0.021 [+0.015, +0.027] / +0.016 | +0.010 [+0.007, +0.013] / +0.008 | +0.010 [+0.004, +0.017] / +0.011 | +0.017 [+0.005, +0.030] / +0.014 | +0.025 [+0.016, +0.034] / +0.019 | +0.021 [+0.013, +0.029] / +0.016 |

### structured: Δ vs raw (joint / all-12)

| job | stratum | raw joint | raw all12 | eq_kl | eq_lsl | bc_kl | rw_kl | rw_lsl | rwg_kl | bcx_kl | rwx_kl | fill_and3 | fill_and4 | fill_litand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 235B P | all (170) | 0.395 | 0.401 | -0.078 / -0.060 | -0.088 / -0.071 | -0.079 / -0.060 | -0.093 / -0.074 | -0.104 / -0.086 | -0.109 / -0.090 | -0.077 / -0.059 | -0.079 / -0.060 | -0.105 / -0.082 | -0.110 / -0.086 | -0.110 / -0.085 |
| 235B Q | all (1077) | 0.562 | 0.602 | -0.080 / -0.072 | -0.080 / -0.070 | -0.077 / -0.070 | -0.077 / -0.070 | -0.078 / -0.070 | -0.076 / -0.069 | -0.080 / -0.071 | -0.080 / -0.071 | -0.070 / -0.065 | -0.087 / -0.077 | -0.084 / -0.075 |
| 32B P | all (1077) | 0.492 | 0.468 | -0.140 / -0.091 | -0.147 / -0.097 | -0.137 / -0.088 | -0.151 / -0.102 | -0.156 / -0.107 | -0.164 / -0.115 | -0.137 / -0.088 | -0.134 / -0.084 | -0.162 / -0.110 | -0.158 / -0.107 | -0.158 / -0.107 |
| 32B Q | all (1077) | 0.637 | 0.673 | -0.136 / -0.126 | -0.131 / -0.120 | -0.126 / -0.119 | -0.127 / -0.119 | -0.121 / -0.113 | -0.127 / -0.119 | -0.132 / -0.123 | -0.133 / -0.123 | -0.134 / -0.124 | -0.127 / -0.120 | -0.127 / -0.120 |

### structured: market variants, joint Δ vs market + independence [95% CI]

| job | stratum | market indep | market_known | h_eq | h_bc | h_rw | h_rwg | h_bcx | h_rwx | h_and3 | h_and4 | h_litand | h_and3_s | h_and4_s | h_litand_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 235B P | all (170) | 0.307 | -0.043 [-0.056, -0.030] | -0.020 [-0.027, -0.013] | -0.020 [-0.026, -0.014] | -0.024 [-0.031, -0.017] | -0.027 [-0.036, -0.019] | -0.019 [-0.026, -0.013] | -0.021 [-0.027, -0.014] | -0.033 [-0.043, -0.023] | -0.029 [-0.037, -0.021] | -0.029 [-0.037, -0.020] | -0.036 [-0.047, -0.025] | -0.040 [-0.051, -0.028] | -0.040 [-0.051, -0.027] |
| 235B P | control (44) | 0.276 | +0.000 [+0.000, +0.000] | -0.001 [-0.004, +0.004] | -0.002 [-0.008, +0.002] | -0.002 [-0.006, +0.002] | -0.000 [-0.004, +0.004] | -0.002 [-0.005, +0.002] | -0.002 [-0.006, +0.002] | -0.000 [-0.002, +0.002] | -0.002 [-0.010, +0.005] | -0.002 [-0.009, +0.005] | +0.000 [-0.003, +0.003] | -0.005 [-0.022, +0.012] | -0.004 [-0.020, +0.012] |
| 235B P | date (27) | 0.371 | -0.037 [-0.075, +0.004] | -0.016 [-0.034, +0.003] | -0.013 [-0.029, +0.001] | -0.017 [-0.036, +0.003] | -0.021 [-0.046, +0.006] | -0.013 [-0.030, +0.004] | -0.011 [-0.028, +0.006] | -0.035 [-0.072, +0.003] | -0.013 [-0.035, +0.014] | -0.014 [-0.037, +0.014] | -0.037 [-0.073, +0.001] | -0.024 [-0.057, +0.011] | -0.027 [-0.060, +0.009] |
| 235B P | mc_excl (49) | 0.357 | -0.065 [-0.090, -0.037] | -0.021 [-0.032, -0.010] | -0.028 [-0.039, -0.017] | -0.036 [-0.048, -0.024] | -0.041 [-0.058, -0.025] | -0.025 [-0.036, -0.014] | -0.030 [-0.042, -0.018] | -0.037 [-0.055, -0.017] | -0.050 [-0.065, -0.034] | -0.049 [-0.064, -0.034] | -0.042 [-0.064, -0.018] | -0.062 [-0.084, -0.037] | -0.063 [-0.085, -0.039] |
| 235B P | threshold (50) | 0.251 | -0.063 [-0.088, -0.042] | -0.037 [-0.056, -0.023] | -0.029 [-0.046, -0.017] | -0.034 [-0.053, -0.021] | -0.040 [-0.061, -0.026] | -0.033 [-0.050, -0.020] | -0.032 [-0.049, -0.020] | -0.058 [-0.079, -0.039] | -0.042 [-0.060, -0.027] | -0.040 [-0.059, -0.025] | -0.062 [-0.087, -0.042] | -0.057 [-0.082, -0.036] | -0.055 [-0.080, -0.034] |
| 235B Q | all (1077) | 0.326 | -0.037 [-0.043, -0.030] | -0.021 [-0.024, -0.017] | -0.019 [-0.022, -0.016] | -0.019 [-0.022, -0.016] | -0.019 [-0.022, -0.016] | -0.020 [-0.024, -0.017] | -0.021 [-0.025, -0.018] | -0.025 [-0.032, -0.018] | -0.024 [-0.028, -0.020] | -0.022 [-0.026, -0.018] | -0.025 [-0.032, -0.017] | -0.029 [-0.035, -0.024] | -0.027 [-0.032, -0.021] |
| 235B Q | control (297) | 0.304 | +0.000 [+0.000, +0.000] | -0.001 [-0.003, +0.000] | -0.000 [-0.002, +0.002] | -0.001 [-0.002, +0.001] | -0.001 [-0.003, +0.001] | -0.001 [-0.003, +0.000] | -0.001 [-0.003, +0.000] | +0.011 [-0.002, +0.026] | -0.001 [-0.003, +0.000] | -0.001 [-0.003, +0.001] | +0.011 [-0.001, +0.027] | -0.002 [-0.006, +0.002] | +0.000 [-0.005, +0.005] |
| 235B Q | date (180) | 0.338 | -0.056 [-0.074, -0.037] | -0.032 [-0.044, -0.022] | -0.025 [-0.034, -0.016] | -0.024 [-0.034, -0.016] | -0.024 [-0.034, -0.016] | -0.030 [-0.041, -0.020] | -0.030 [-0.041, -0.020] | -0.044 [-0.064, -0.022] | -0.026 [-0.037, -0.016] | -0.026 [-0.037, -0.016] | -0.043 [-0.064, -0.021] | -0.035 [-0.052, -0.016] | -0.034 [-0.051, -0.016] |
| 235B Q | mc_excl (300) | 0.377 | -0.059 [-0.071, -0.045] | -0.030 [-0.037, -0.023] | -0.035 [-0.042, -0.027] | -0.035 [-0.042, -0.027] | -0.034 [-0.042, -0.027] | -0.033 [-0.040, -0.025] | -0.036 [-0.044, -0.028] | -0.042 [-0.053, -0.031] | -0.048 [-0.058, -0.038] | -0.041 [-0.050, -0.032] | -0.043 [-0.055, -0.031] | -0.053 [-0.065, -0.039] | -0.047 [-0.059, -0.035] |
| 235B Q | threshold (300) | 0.291 | -0.039 [-0.053, -0.024] | -0.024 [-0.031, -0.018] | -0.018 [-0.023, -0.013] | -0.019 [-0.024, -0.013] | -0.018 [-0.024, -0.013] | -0.022 [-0.028, -0.016] | -0.022 [-0.028, -0.016] | -0.032 [-0.046, -0.018] | -0.022 [-0.029, -0.016] | -0.022 [-0.029, -0.016] | -0.032 [-0.046, -0.017] | -0.030 [-0.041, -0.020] | -0.029 [-0.039, -0.018] |
| 32B P | all (1077) | 0.326 | -0.037 [-0.043, -0.030] | -0.015 [-0.018, -0.012] | -0.014 [-0.017, -0.010] | -0.015 [-0.019, -0.012] | -0.018 [-0.022, -0.015] | -0.013 [-0.016, -0.010] | -0.014 [-0.017, -0.010] | -0.028 [-0.034, -0.021] | -0.022 [-0.026, -0.017] | -0.021 [-0.026, -0.017] | -0.028 [-0.034, -0.022] | -0.024 [-0.030, -0.018] | -0.024 [-0.030, -0.018] |
| 32B P | control (297) | 0.304 | +0.000 [+0.000, +0.000] | -0.001 [-0.004, +0.001] | -0.000 [-0.003, +0.002] | -0.001 [-0.004, +0.001] | -0.003 [-0.005, -0.000] | -0.001 [-0.004, +0.001] | -0.001 [-0.003, +0.001] | +0.005 [-0.001, +0.014] | -0.001 [-0.004, +0.002] | -0.000 [-0.003, +0.003] | +0.003 [-0.003, +0.010] | +0.000 [-0.005, +0.007] | +0.002 [-0.004, +0.008] |
| 32B P | date (180) | 0.338 | -0.056 [-0.074, -0.037] | -0.010 [-0.018, -0.002] | -0.003 [-0.010, +0.004] | -0.008 [-0.015, -0.001] | -0.017 [-0.025, -0.008] | -0.004 [-0.011, +0.004] | -0.004 [-0.011, +0.004] | -0.040 [-0.056, -0.024] | -0.018 [-0.029, -0.008] | -0.018 [-0.029, -0.008] | -0.036 [-0.049, -0.021] | -0.023 [-0.038, -0.006] | -0.023 [-0.040, -0.006] |
| 32B P | mc_excl (300) | 0.377 | -0.059 [-0.071, -0.045] | -0.035 [-0.042, -0.027] | -0.041 [-0.049, -0.032] | -0.041 [-0.049, -0.032] | -0.039 [-0.047, -0.030] | -0.037 [-0.044, -0.029] | -0.040 [-0.047, -0.031] | -0.057 [-0.069, -0.044] | -0.049 [-0.059, -0.038] | -0.050 [-0.060, -0.038] | -0.057 [-0.068, -0.044] | -0.052 [-0.064, -0.039] | -0.055 [-0.066, -0.042] |
| 32B P | threshold (300) | 0.291 | -0.039 [-0.053, -0.024] | -0.011 [-0.017, -0.005] | -0.005 [-0.011, +0.000] | -0.008 [-0.013, -0.003] | -0.013 [-0.019, -0.007] | -0.007 [-0.012, -0.001] | -0.007 [-0.012, -0.001] | -0.023 [-0.038, -0.009] | -0.018 [-0.026, -0.010] | -0.016 [-0.023, -0.009] | -0.026 [-0.039, -0.013] | -0.021 [-0.032, -0.009] | -0.019 [-0.030, -0.007] |
| 32B Q | all (1077) | 0.326 | -0.037 [-0.043, -0.030] | -0.018 [-0.022, -0.014] | -0.015 [-0.018, -0.011] | -0.015 [-0.019, -0.011] | -0.015 [-0.019, -0.011] | -0.017 [-0.021, -0.013] | -0.018 [-0.022, -0.014] | -0.018 [-0.026, -0.010] | -0.016 [-0.021, -0.011] | -0.016 [-0.021, -0.011] | -0.019 [-0.025, -0.013] | -0.016 [-0.022, -0.010] | -0.016 [-0.021, -0.010] |
| 32B Q | control (297) | 0.304 | +0.000 [+0.000, +0.000] | -0.000 [-0.002, +0.001] | +0.001 [-0.002, +0.003] | +0.001 [-0.002, +0.003] | +0.001 [-0.002, +0.003] | +0.000 [-0.002, +0.002] | +0.000 [-0.002, +0.002] | +0.019 [+0.005, +0.034] | +0.002 [-0.002, +0.005] | +0.002 [-0.002, +0.005] | +0.014 [+0.003, +0.026] | +0.003 [-0.002, +0.008] | +0.002 [-0.002, +0.007] |
| 32B Q | date (180) | 0.338 | -0.056 [-0.074, -0.037] | -0.018 [-0.027, -0.010] | -0.006 [-0.013, +0.002] | -0.007 [-0.015, +0.001] | -0.007 [-0.015, +0.002] | -0.014 [-0.023, -0.006] | -0.015 [-0.024, -0.007] | -0.038 [-0.058, -0.017] | -0.011 [-0.024, +0.001] | -0.014 [-0.028, -0.001] | -0.036 [-0.053, -0.020] | -0.010 [-0.025, +0.006] | -0.014 [-0.029, +0.001] |
| 32B Q | mc_excl (300) | 0.377 | -0.059 [-0.071, -0.045] | -0.038 [-0.047, -0.028] | -0.042 [-0.052, -0.032] | -0.043 [-0.052, -0.033] | -0.042 [-0.052, -0.032] | -0.040 [-0.049, -0.030] | -0.042 [-0.051, -0.032] | -0.053 [-0.064, -0.039] | -0.046 [-0.057, -0.033] | -0.042 [-0.054, -0.029] | -0.047 [-0.058, -0.036] | -0.049 [-0.061, -0.035] | -0.044 [-0.056, -0.030] |
| 32B Q | threshold (300) | 0.291 | -0.039 [-0.053, -0.024] | -0.015 [-0.022, -0.007] | -0.007 [-0.014, -0.000] | -0.008 [-0.015, -0.000] | -0.008 [-0.014, -0.001] | -0.012 [-0.019, -0.004] | -0.013 [-0.020, -0.005] | -0.008 [-0.022, +0.007] | -0.007 [-0.015, +0.002] | -0.007 [-0.016, +0.002] | -0.013 [-0.024, -0.001] | -0.007 [-0.017, +0.004] | -0.006 [-0.017, +0.005] |

### structured: market variants, joint Δ vs the old hybrid h_eq (paired) [95% CI]

| job | stratum | h_bc | h_rw | h_rwg | h_bcx | h_rwx | h_and3 | h_and4 | h_litand | h_and3_s | h_and4_s | h_litand_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 235B P | all (170) | +0.000 [-0.002, +0.002] | -0.004 [-0.007, -0.001] | -0.007 [-0.013, -0.003] | +0.000 [-0.001, +0.001] | -0.001 [-0.003, +0.001] | -0.014 [-0.020, -0.007] | -0.009 [-0.015, -0.005] | -0.009 [-0.014, -0.004] | -0.016 [-0.024, -0.009] | -0.020 [-0.028, -0.012] | -0.020 [-0.028, -0.011] |
| 235B P | control (44) | -0.002 [-0.007, +0.003] | -0.001 [-0.005, +0.002] | +0.001 [-0.002, +0.004] | -0.001 [-0.003, +0.001] | -0.001 [-0.005, +0.001] | +0.000 [-0.004, +0.004] | -0.001 [-0.008, +0.003] | -0.001 [-0.007, +0.004] | +0.001 [-0.004, +0.006] | -0.004 [-0.019, +0.010] | -0.003 [-0.018, +0.011] |
| 235B P | date (27) | +0.003 [-0.002, +0.008] | -0.000 [-0.003, +0.002] | -0.005 [-0.013, +0.005] | +0.003 [+0.000, +0.006] | +0.005 [+0.001, +0.009] | -0.018 [-0.040, +0.004] | +0.003 [-0.005, +0.014] | +0.003 [-0.005, +0.015] | -0.020 [-0.042, +0.003] | -0.008 [-0.025, +0.010] | -0.010 [-0.028, +0.008] |
| 235B P | mc_excl (49) | -0.008 [-0.013, -0.003] | -0.015 [-0.023, -0.007] | -0.021 [-0.035, -0.007] | -0.004 [-0.006, -0.002] | -0.009 [-0.013, -0.005] | -0.016 [-0.033, +0.002] | -0.029 [-0.043, -0.015] | -0.028 [-0.042, -0.015] | -0.021 [-0.041, +0.001] | -0.041 [-0.062, -0.018] | -0.042 [-0.063, -0.020] |
| 235B P | threshold (50) | +0.008 [+0.005, +0.010] | +0.003 [+0.001, +0.005] | -0.003 [-0.008, +0.002] | +0.004 [+0.003, +0.006] | +0.005 [+0.003, +0.007] | -0.021 [-0.030, -0.012] | -0.005 [-0.009, -0.000] | -0.003 [-0.007, +0.002] | -0.025 [-0.034, -0.017] | -0.020 [-0.029, -0.011] | -0.018 [-0.028, -0.008] |
| 235B Q | all (1077) | +0.002 [+0.001, +0.003] | +0.002 [+0.001, +0.003] | +0.002 [+0.001, +0.003] | +0.000 [+0.000, +0.001] | -0.001 [-0.001, +0.000] | -0.004 [-0.010, +0.001] | -0.003 [-0.005, -0.001] | -0.001 [-0.003, +0.001] | -0.004 [-0.010, +0.002] | -0.009 [-0.012, -0.005] | -0.006 [-0.009, -0.002] |
| 235B Q | control (297) | +0.001 [-0.001, +0.003] | +0.001 [-0.001, +0.003] | +0.001 [-0.002, +0.003] | +0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.001] | +0.012 [+0.000, +0.026] | -0.000 [-0.002, +0.001] | +0.000 [-0.001, +0.002] | +0.013 [+0.000, +0.027] | -0.001 [-0.004, +0.003] | +0.001 [-0.003, +0.006] |
| 235B Q | date (180) | +0.007 [+0.004, +0.011] | +0.008 [+0.005, +0.011] | +0.008 [+0.005, +0.012] | +0.003 [+0.002, +0.004] | +0.003 [+0.001, +0.004] | -0.011 [-0.024, +0.004] | +0.007 [+0.002, +0.011] | +0.007 [+0.002, +0.012] | -0.011 [-0.024, +0.004] | -0.002 [-0.013, +0.011] | -0.001 [-0.012, +0.012] |
| 235B Q | mc_excl (300) | -0.005 [-0.006, -0.003] | -0.005 [-0.006, -0.003] | -0.004 [-0.006, -0.003] | -0.003 [-0.003, -0.002] | -0.006 [-0.007, -0.004] | -0.012 [-0.019, -0.005] | -0.018 [-0.023, -0.013] | -0.011 [-0.015, -0.008] | -0.014 [-0.021, -0.006] | -0.023 [-0.031, -0.014] | -0.017 [-0.024, -0.010] |
| 235B Q | threshold (300) | +0.006 [+0.004, +0.008] | +0.005 [+0.004, +0.007] | +0.006 [+0.004, +0.008] | +0.003 [+0.002, +0.003] | +0.002 [+0.001, +0.003] | -0.008 [-0.019, +0.003] | +0.002 [-0.001, +0.005] | +0.002 [-0.000, +0.005] | -0.008 [-0.019, +0.005] | -0.006 [-0.011, -0.001] | -0.005 [-0.010, +0.001] |
| 32B P | all (1077) | +0.001 [+0.000, +0.002] | -0.001 [-0.001, +0.000] | -0.003 [-0.005, -0.002] | +0.002 [+0.001, +0.002] | +0.001 [+0.000, +0.002] | -0.013 [-0.018, -0.007] | -0.007 [-0.010, -0.004] | -0.007 [-0.010, -0.004] | -0.013 [-0.018, -0.008] | -0.009 [-0.013, -0.005] | -0.009 [-0.013, -0.004] |
| 32B P | control (297) | +0.001 [-0.001, +0.003] | -0.000 [-0.001, +0.001] | -0.001 [-0.003, +0.000] | +0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.001] | +0.007 [-0.000, +0.015] | +0.000 [-0.002, +0.003] | +0.001 [-0.002, +0.003] | +0.004 [-0.002, +0.012] | +0.002 [-0.003, +0.007] | +0.003 [-0.002, +0.009] |
| 32B P | date (180) | +0.007 [+0.004, +0.010] | +0.002 [+0.000, +0.004] | -0.006 [-0.010, -0.003] | +0.007 [+0.004, +0.009] | +0.007 [+0.005, +0.009] | -0.030 [-0.043, -0.018] | -0.008 [-0.016, -0.001] | -0.008 [-0.015, -0.002] | -0.025 [-0.037, -0.015] | -0.012 [-0.024, -0.002] | -0.013 [-0.024, -0.002] |
| 32B P | mc_excl (300) | -0.006 [-0.008, -0.004] | -0.006 [-0.008, -0.004] | -0.004 [-0.007, -0.001] | -0.002 [-0.002, -0.001] | -0.005 [-0.006, -0.003] | -0.022 [-0.030, -0.014] | -0.014 [-0.020, -0.007] | -0.015 [-0.020, -0.009] | -0.022 [-0.029, -0.013] | -0.017 [-0.025, -0.008] | -0.020 [-0.027, -0.012] |
| 32B P | threshold (300) | +0.006 [+0.004, +0.007] | +0.003 [+0.001, +0.004] | -0.002 [-0.006, +0.002] | +0.004 [+0.003, +0.005] | +0.004 [+0.003, +0.005] | -0.012 [-0.026, +0.001] | -0.007 [-0.014, -0.001] | -0.005 [-0.012, +0.001] | -0.015 [-0.027, -0.002] | -0.010 [-0.020, -0.001] | -0.008 [-0.018, +0.002] |
| 32B Q | all (1077) | +0.003 [+0.002, +0.005] | +0.003 [+0.002, +0.004] | +0.003 [+0.002, +0.004] | +0.001 [+0.001, +0.002] | +0.000 [-0.001, +0.001] | -0.000 [-0.006, +0.007] | +0.002 [-0.001, +0.005] | +0.002 [-0.001, +0.005] | -0.001 [-0.006, +0.004] | +0.001 [-0.002, +0.005] | +0.002 [-0.001, +0.006] |
| 32B Q | control (297) | +0.001 [-0.001, +0.003] | +0.001 [-0.001, +0.003] | +0.001 [-0.001, +0.003] | +0.000 [-0.000, +0.001] | +0.000 [-0.001, +0.001] | +0.020 [+0.005, +0.035] | +0.002 [-0.001, +0.005] | +0.002 [-0.001, +0.005] | +0.014 [+0.003, +0.027] | +0.003 [-0.001, +0.007] | +0.002 [-0.001, +0.006] |
| 32B Q | date (180) | +0.012 [+0.008, +0.017] | +0.011 [+0.007, +0.016] | +0.011 [+0.007, +0.016] | +0.004 [+0.003, +0.006] | +0.003 [+0.001, +0.004] | -0.020 [-0.036, -0.002] | +0.007 [-0.001, +0.014] | +0.004 [-0.004, +0.012] | -0.018 [-0.031, -0.006] | +0.008 [-0.003, +0.020] | +0.004 [-0.006, +0.014] |
| 32B Q | mc_excl (300) | -0.004 [-0.006, -0.003] | -0.005 [-0.006, -0.003] | -0.004 [-0.006, -0.003] | -0.002 [-0.002, -0.001] | -0.004 [-0.005, -0.002] | -0.014 [-0.022, -0.006] | -0.008 [-0.013, -0.003] | -0.004 [-0.009, +0.002] | -0.009 [-0.015, -0.003] | -0.011 [-0.016, -0.004] | -0.005 [-0.011, +0.002] |
| 32B Q | threshold (300) | +0.008 [+0.005, +0.010] | +0.007 [+0.005, +0.009] | +0.007 [+0.005, +0.009] | +0.003 [+0.002, +0.004] | +0.002 [+0.001, +0.003] | +0.007 [-0.007, +0.021] | +0.008 [+0.002, +0.014] | +0.007 [+0.002, +0.013] | +0.002 [-0.009, +0.013] | +0.008 [+0.001, +0.016] | +0.008 [+0.002, +0.015] |

### structured: market variants, all-12 Δ vs market + independence

| job | stratum | market_known | h_eq | h_bc | h_rw | h_rwg | h_bcx | h_rwx | h_and3 | h_and4 | h_litand | h_and3_s | h_and4_s | h_litand_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 235B P | all (170) | -0.029 [-0.037, -0.020] | -0.013 [-0.018, -0.009] | -0.013 [-0.017, -0.009] | -0.016 [-0.021, -0.011] | -0.018 [-0.024, -0.013] | -0.013 [-0.017, -0.009] | -0.014 [-0.018, -0.010] | -0.022 [-0.029, -0.015] | -0.019 [-0.025, -0.014] | -0.019 [-0.025, -0.013] | -0.024 [-0.032, -0.016] | -0.027 [-0.034, -0.019] | -0.026 [-0.034, -0.018] |
| 235B P | control (44) | +0.000 [+0.000, +0.000] | -0.001 [-0.003, +0.002] | -0.002 [-0.005, +0.001] | -0.001 [-0.004, +0.001] | -0.000 [-0.003, +0.003] | -0.001 [-0.003, +0.001] | -0.002 [-0.004, +0.001] | -0.000 [-0.002, +0.001] | -0.001 [-0.006, +0.003] | -0.001 [-0.006, +0.003] | +0.000 [-0.002, +0.002] | -0.003 [-0.015, +0.008] | -0.003 [-0.014, +0.008] |
| 235B P | date (27) | -0.024 [-0.050, +0.002] | -0.011 [-0.023, +0.002] | -0.009 [-0.019, +0.001] | -0.011 [-0.024, +0.002] | -0.014 [-0.031, +0.004] | -0.009 [-0.020, +0.003] | -0.008 [-0.018, +0.004] | -0.023 [-0.048, +0.002] | -0.009 [-0.024, +0.010] | -0.009 [-0.025, +0.009] | -0.025 [-0.049, +0.001] | -0.016 [-0.038, +0.007] | -0.018 [-0.040, +0.006] |
| 235B P | mc_excl (49) | -0.043 [-0.060, -0.025] | -0.014 [-0.021, -0.006] | -0.019 [-0.026, -0.011] | -0.024 [-0.032, -0.016] | -0.028 [-0.039, -0.017] | -0.016 [-0.024, -0.009] | -0.020 [-0.028, -0.012] | -0.024 [-0.036, -0.011] | -0.033 [-0.043, -0.022] | -0.033 [-0.043, -0.023] | -0.028 [-0.042, -0.012] | -0.041 [-0.056, -0.025] | -0.042 [-0.057, -0.026] |
| 235B P | threshold (50) | -0.042 [-0.059, -0.028] | -0.025 [-0.037, -0.015] | -0.020 [-0.031, -0.011] | -0.023 [-0.035, -0.014] | -0.027 [-0.041, -0.017] | -0.022 [-0.034, -0.013] | -0.022 [-0.033, -0.013] | -0.039 [-0.053, -0.026] | -0.028 [-0.040, -0.018] | -0.027 [-0.039, -0.017] | -0.042 [-0.058, -0.028] | -0.038 [-0.054, -0.024] | -0.037 [-0.053, -0.023] |
| 235B Q | all (1077) | -0.024 [-0.029, -0.020] | -0.014 [-0.016, -0.011] | -0.013 [-0.015, -0.010] | -0.013 [-0.015, -0.010] | -0.013 [-0.015, -0.010] | -0.014 [-0.016, -0.011] | -0.014 [-0.017, -0.012] | -0.017 [-0.022, -0.012] | -0.016 [-0.019, -0.013] | -0.015 [-0.017, -0.012] | -0.017 [-0.022, -0.012] | -0.020 [-0.023, -0.016] | -0.018 [-0.021, -0.014] |
| 235B Q | control (297) | +0.000 [+0.000, +0.000] | -0.001 [-0.002, +0.000] | -0.000 [-0.002, +0.001] | -0.000 [-0.002, +0.001] | -0.000 [-0.002, +0.001] | -0.001 [-0.002, +0.000] | -0.001 [-0.002, +0.000] | +0.007 [-0.001, +0.017] | -0.001 [-0.002, +0.000] | -0.001 [-0.002, +0.001] | +0.008 [-0.001, +0.018] | -0.001 [-0.004, +0.001] | +0.000 [-0.003, +0.003] |
| 235B Q | date (180) | -0.037 [-0.049, -0.025] | -0.022 [-0.029, -0.015] | -0.017 [-0.023, -0.011] | -0.016 [-0.023, -0.011] | -0.016 [-0.023, -0.010] | -0.020 [-0.027, -0.013] | -0.020 [-0.027, -0.013] | -0.029 [-0.043, -0.015] | -0.017 [-0.025, -0.011] | -0.017 [-0.025, -0.010] | -0.029 [-0.042, -0.014] | -0.023 [-0.035, -0.011] | -0.022 [-0.034, -0.011] |
| 235B Q | mc_excl (300) | -0.039 [-0.047, -0.030] | -0.020 [-0.025, -0.015] | -0.023 [-0.028, -0.018] | -0.023 [-0.028, -0.018] | -0.023 [-0.028, -0.018] | -0.022 [-0.027, -0.017] | -0.024 [-0.029, -0.019] | -0.028 [-0.036, -0.021] | -0.032 [-0.038, -0.025] | -0.027 [-0.033, -0.021] | -0.029 [-0.037, -0.021] | -0.035 [-0.043, -0.026] | -0.031 [-0.039, -0.023] |
| 235B Q | threshold (300) | -0.026 [-0.035, -0.016] | -0.016 [-0.021, -0.012] | -0.012 [-0.016, -0.009] | -0.012 [-0.016, -0.009] | -0.012 [-0.016, -0.009] | -0.014 [-0.018, -0.011] | -0.015 [-0.019, -0.011] | -0.022 [-0.031, -0.012] | -0.015 [-0.019, -0.011] | -0.015 [-0.019, -0.011] | -0.021 [-0.031, -0.011] | -0.020 [-0.027, -0.013] | -0.019 [-0.026, -0.012] |
| 32B P | all (1077) | -0.024 [-0.029, -0.020] | -0.010 [-0.012, -0.008] | -0.009 [-0.011, -0.007] | -0.010 [-0.012, -0.008] | -0.012 [-0.014, -0.010] | -0.009 [-0.011, -0.007] | -0.009 [-0.011, -0.007] | -0.018 [-0.023, -0.014] | -0.015 [-0.018, -0.011] | -0.014 [-0.017, -0.011] | -0.019 [-0.023, -0.015] | -0.016 [-0.020, -0.012] | -0.016 [-0.020, -0.012] |
| 32B P | control (297) | +0.000 [+0.000, +0.000] | -0.001 [-0.002, +0.001] | -0.000 [-0.002, +0.002] | -0.001 [-0.002, +0.001] | -0.002 [-0.003, -0.000] | -0.001 [-0.002, +0.001] | -0.001 [-0.002, +0.001] | +0.004 [-0.001, +0.010] | -0.001 [-0.003, +0.002] | -0.000 [-0.002, +0.002] | +0.002 [-0.002, +0.007] | +0.000 [-0.004, +0.005] | +0.001 [-0.003, +0.006] |
| 32B P | date (180) | -0.037 [-0.049, -0.025] | -0.007 [-0.012, -0.002] | -0.002 [-0.007, +0.003] | -0.006 [-0.010, -0.001] | -0.011 [-0.016, -0.006] | -0.003 [-0.007, +0.003] | -0.002 [-0.007, +0.003] | -0.027 [-0.038, -0.016] | -0.012 [-0.019, -0.005] | -0.012 [-0.019, -0.005] | -0.024 [-0.033, -0.014] | -0.015 [-0.026, -0.004] | -0.015 [-0.026, -0.004] |
| 32B P | mc_excl (300) | -0.039 [-0.047, -0.030] | -0.023 [-0.028, -0.018] | -0.027 [-0.033, -0.021] | -0.027 [-0.033, -0.021] | -0.026 [-0.032, -0.020] | -0.024 [-0.029, -0.019] | -0.026 [-0.031, -0.021] | -0.038 [-0.046, -0.029] | -0.033 [-0.039, -0.025] | -0.033 [-0.040, -0.026] | -0.038 [-0.045, -0.029] | -0.035 [-0.042, -0.026] | -0.036 [-0.044, -0.028] |
| 32B P | threshold (300) | -0.026 [-0.035, -0.016] | -0.007 [-0.011, -0.004] | -0.004 [-0.007, +0.000] | -0.006 [-0.009, -0.002] | -0.009 [-0.013, -0.005] | -0.005 [-0.008, -0.001] | -0.005 [-0.008, -0.001] | -0.016 [-0.025, -0.006] | -0.012 [-0.018, -0.007] | -0.011 [-0.016, -0.006] | -0.017 [-0.026, -0.009] | -0.014 [-0.022, -0.006] | -0.013 [-0.020, -0.005] |
| 32B Q | all (1077) | -0.024 [-0.029, -0.020] | -0.012 [-0.014, -0.009] | -0.010 [-0.012, -0.007] | -0.010 [-0.013, -0.008] | -0.010 [-0.013, -0.007] | -0.011 [-0.014, -0.009] | -0.012 [-0.014, -0.009] | -0.012 [-0.017, -0.007] | -0.011 [-0.014, -0.007] | -0.011 [-0.014, -0.007] | -0.013 [-0.017, -0.008] | -0.011 [-0.015, -0.007] | -0.010 [-0.014, -0.007] |
| 32B Q | control (297) | +0.000 [+0.000, +0.000] | -0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.002] | +0.000 [-0.001, +0.002] | +0.000 [-0.001, +0.002] | +0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.001] | +0.013 [+0.004, +0.023] | +0.001 [-0.001, +0.004] | +0.001 [-0.001, +0.003] | +0.009 [+0.002, +0.017] | +0.002 [-0.001, +0.005] | +0.001 [-0.001, +0.004] |
| 32B Q | date (180) | -0.037 [-0.049, -0.025] | -0.012 [-0.018, -0.006] | -0.004 [-0.009, +0.001] | -0.005 [-0.010, +0.001] | -0.004 [-0.010, +0.001] | -0.009 [-0.015, -0.004] | -0.010 [-0.016, -0.005] | -0.025 [-0.039, -0.012] | -0.007 [-0.016, +0.001] | -0.009 [-0.018, -0.001] | -0.024 [-0.035, -0.013] | -0.006 [-0.017, +0.004] | -0.009 [-0.020, +0.001] |
| 32B Q | mc_excl (300) | -0.039 [-0.047, -0.030] | -0.025 [-0.032, -0.019] | -0.028 [-0.035, -0.022] | -0.028 [-0.035, -0.022] | -0.028 [-0.035, -0.021] | -0.027 [-0.033, -0.020] | -0.028 [-0.034, -0.021] | -0.035 [-0.043, -0.026] | -0.031 [-0.038, -0.022] | -0.028 [-0.036, -0.019] | -0.032 [-0.039, -0.024] | -0.033 [-0.041, -0.023] | -0.029 [-0.037, -0.020] |
| 32B Q | threshold (300) | -0.026 [-0.035, -0.016] | -0.010 [-0.015, -0.005] | -0.005 [-0.009, -0.000] | -0.005 [-0.010, -0.000] | -0.005 [-0.010, -0.000] | -0.008 [-0.013, -0.003] | -0.008 [-0.013, -0.003] | -0.006 [-0.015, +0.005] | -0.004 [-0.010, +0.001] | -0.005 [-0.011, +0.001] | -0.009 [-0.016, -0.001] | -0.004 [-0.011, +0.003] | -0.004 [-0.011, +0.003] |

### structured: share of the known-relation gain recovered (joint; (indep − method) / (indep − known))

| job | stratum | known gain | raw | h_eq | h_bc | h_rw | h_rwg | h_bcx | h_rwx | h_and3 | h_and4 | h_litand | h_and3_s | h_and4_s | h_litand_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 235B P | all | 0.043 | -203% | 46% | 46% | 55% | 63% | 45% | 48% | 77% | 68% | 66% | 84% | 93% | 92% |
| 235B P | date | 0.037 | -149% | 45% | 36% | 46% | 57% | 36% | 31% | 95% | 35% | 37% | 100% | 67% | 73% |
| 235B P | mc_excl | 0.065 | -198% | 32% | 44% | 56% | 64% | 38% | 45% | 56% | 76% | 75% | 64% | 95% | 96% |
| 235B P | threshold | 0.063 | -48% | 59% | 47% | 55% | 64% | 52% | 52% | 92% | 66% | 64% | 99% | 91% | 88% |
| 235B Q | all | 0.037 | -645% | 57% | 52% | 52% | 52% | 56% | 59% | 69% | 66% | 60% | 69% | 81% | 73% |
| 235B Q | date | 0.056 | -475% | 58% | 45% | 44% | 43% | 53% | 53% | 78% | 46% | 46% | 77% | 62% | 60% |
| 235B Q | mc_excl | 0.059 | -408% | 51% | 59% | 59% | 58% | 55% | 61% | 72% | 81% | 70% | 74% | 89% | 80% |
| 235B Q | threshold | 0.039 | -615% | 62% | 47% | 48% | 48% | 56% | 57% | 83% | 57% | 57% | 82% | 78% | 74% |
| 32B P | all | 0.037 | -455% | 41% | 37% | 42% | 49% | 36% | 38% | 76% | 60% | 59% | 77% | 66% | 65% |
| 32B P | date | 0.056 | -197% | 18% | 6% | 15% | 30% | 7% | 6% | 72% | 33% | 33% | 64% | 40% | 41% |
| 32B P | mc_excl | 0.059 | -266% | 60% | 70% | 69% | 67% | 62% | 67% | 97% | 83% | 84% | 96% | 88% | 93% |
| 32B P | threshold | 0.039 | -521% | 29% | 14% | 21% | 34% | 18% | 18% | 60% | 47% | 42% | 67% | 54% | 48% |
| 32B Q | all | 0.037 | -851% | 49% | 40% | 41% | 41% | 46% | 48% | 49% | 44% | 43% | 52% | 45% | 43% |
| 32B Q | date | 0.056 | -617% | 32% | 10% | 12% | 12% | 24% | 27% | 67% | 20% | 25% | 65% | 17% | 25% |
| 32B Q | mc_excl | 0.059 | -504% | 65% | 72% | 73% | 72% | 68% | 71% | 89% | 79% | 72% | 80% | 83% | 74% |
| 32B Q | threshold | 0.039 | -974% | 38% | 19% | 20% | 20% | 31% | 33% | 22% | 17% | 19% | 33% | 17% | 17% |

### structured: joint Δ vs market + known relation [95% CI]

| job | stratum | eq_kl | rw_kl | fill_and3 | fill_and4 | h_eq | h_bc | h_rw | h_rwg | h_bcx | h_rwx | h_and3 | h_and4 | h_litand | h_and3_s | h_and4_s | h_litand_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 235B P | all | +0.053 [+0.038, +0.067] | +0.038 [+0.025, +0.050] | +0.026 [+0.013, +0.040] | +0.020 [+0.007, +0.034] | +0.023 [+0.014, +0.033] | +0.023 [+0.015, +0.033] | +0.019 [+0.012, +0.027] | +0.016 [+0.010, +0.022] | +0.024 [+0.015, +0.033] | +0.023 [+0.014, +0.031] | +0.010 [+0.004, +0.016] | +0.014 [+0.007, +0.021] | +0.015 [+0.008, +0.021] | +0.007 [+0.002, +0.013] | +0.003 [-0.003, +0.009] | +0.003 [-0.003, +0.010] |
| 235B P | control | +0.044 [+0.020, +0.071] | +0.030 [+0.011, +0.050] | +0.035 [+0.014, +0.056] | +0.031 [+0.000, +0.060] | -0.001 [-0.004, +0.004] | -0.002 [-0.008, +0.002] | -0.002 [-0.006, +0.002] | -0.000 [-0.004, +0.004] | -0.002 [-0.005, +0.002] | -0.002 [-0.006, +0.002] | -0.000 [-0.002, +0.002] | -0.002 [-0.010, +0.005] | -0.002 [-0.009, +0.005] | +0.000 [-0.003, +0.003] | -0.005 [-0.022, +0.012] | -0.004 [-0.020, +0.012] |
| 235B P | date | +0.055 [+0.009, +0.095] | +0.045 [+0.003, +0.082] | +0.035 [-0.012, +0.082] | +0.042 [-0.003, +0.087] | +0.020 [-0.005, +0.044] | +0.023 [-0.007, +0.052] | +0.020 [-0.005, +0.044] | +0.016 [-0.004, +0.034] | +0.023 [-0.005, +0.050] | +0.025 [-0.003, +0.052] | +0.002 [-0.009, +0.011] | +0.024 [+0.002, +0.046] | +0.023 [+0.001, +0.045] | -0.000 [-0.010, +0.007] | +0.012 [-0.003, +0.032] | +0.010 [-0.006, +0.029] |
| 235B P | mc_excl | +0.067 [+0.031, +0.099] | +0.036 [+0.006, +0.067] | +0.031 [-0.000, +0.066] | +0.004 [-0.023, +0.033] | +0.044 [+0.017, +0.069] | +0.037 [+0.014, +0.058] | +0.029 [+0.009, +0.047] | +0.024 [+0.008, +0.040] | +0.040 [+0.015, +0.064] | +0.036 [+0.012, +0.057] | +0.028 [+0.010, +0.049] | +0.016 [-0.000, +0.031] | +0.016 [-0.001, +0.031] | +0.023 [+0.006, +0.042] | +0.003 [-0.003, +0.010] | +0.002 [-0.004, +0.009] |
| 235B P | threshold | +0.045 [+0.031, +0.059] | +0.041 [+0.028, +0.056] | +0.009 [-0.002, +0.020] | +0.015 [+0.002, +0.029] | +0.026 [+0.017, +0.035] | +0.033 [+0.023, +0.045] | +0.028 [+0.019, +0.038] | +0.022 [+0.014, +0.032] | +0.030 [+0.020, +0.040] | +0.030 [+0.020, +0.041] | +0.005 [+0.000, +0.013] | +0.021 [+0.012, +0.031] | +0.023 [+0.013, +0.033] | +0.000 [-0.001, +0.002] | +0.005 [-0.001, +0.016] | +0.007 [-0.000, +0.019] |
| 235B Q | all | +0.192 [+0.173, +0.211] | +0.195 [+0.177, +0.215] | +0.202 [+0.182, +0.222] | +0.185 [+0.166, +0.204] | +0.016 [+0.011, +0.020] | +0.018 [+0.013, +0.022] | +0.017 [+0.013, +0.022] | +0.018 [+0.013, +0.022] | +0.016 [+0.012, +0.021] | +0.015 [+0.011, +0.019] | +0.011 [+0.006, +0.017] | +0.012 [+0.008, +0.017] | +0.014 [+0.010, +0.019] | +0.011 [+0.006, +0.017] | +0.007 [+0.003, +0.011] | +0.010 [+0.005, +0.015] |
| 235B Q | control | +0.171 [+0.141, +0.201] | +0.171 [+0.141, +0.200] | +0.205 [+0.167, +0.244] | +0.176 [+0.145, +0.208] | -0.001 [-0.003, +0.000] | -0.000 [-0.002, +0.002] | -0.001 [-0.002, +0.001] | -0.001 [-0.003, +0.001] | -0.001 [-0.003, +0.000] | -0.001 [-0.003, +0.000] | +0.011 [-0.002, +0.026] | -0.001 [-0.003, +0.000] | -0.001 [-0.003, +0.001] | +0.011 [-0.001, +0.027] | -0.002 [-0.006, +0.002] | +0.000 [-0.005, +0.005] |
| 235B Q | date | +0.232 [+0.176, +0.284] | +0.250 [+0.193, +0.302] | +0.225 [+0.168, +0.282] | +0.228 [+0.172, +0.281] | +0.023 [+0.013, +0.034] | +0.031 [+0.018, +0.045] | +0.031 [+0.018, +0.045] | +0.032 [+0.018, +0.045] | +0.026 [+0.015, +0.038] | +0.026 [+0.015, +0.037] | +0.012 [+0.003, +0.025] | +0.030 [+0.018, +0.043] | +0.030 [+0.017, +0.044] | +0.013 [+0.004, +0.026] | +0.021 [+0.007, +0.038] | +0.022 [+0.007, +0.041] |
| 235B Q | mc_excl | +0.192 [+0.157, +0.228] | +0.180 [+0.146, +0.214] | +0.203 [+0.170, +0.239] | +0.167 [+0.133, +0.201] | +0.029 [+0.019, +0.038] | +0.024 [+0.015, +0.032] | +0.024 [+0.015, +0.033] | +0.025 [+0.015, +0.033] | +0.026 [+0.017, +0.035] | +0.023 [+0.014, +0.031] | +0.017 [+0.008, +0.026] | +0.011 [+0.004, +0.017] | +0.018 [+0.008, +0.026] | +0.015 [+0.007, +0.024] | +0.006 [-0.001, +0.014] | +0.012 [+0.002, +0.021] |
| 235B Q | threshold | +0.188 [+0.149, +0.224] | +0.202 [+0.162, +0.241] | +0.183 [+0.143, +0.220] | +0.185 [+0.146, +0.222] | +0.015 [+0.004, +0.024] | +0.021 [+0.009, +0.031] | +0.020 [+0.008, +0.030] | +0.020 [+0.009, +0.031] | +0.017 [+0.006, +0.027] | +0.017 [+0.006, +0.026] | +0.006 [-0.001, +0.014] | +0.017 [+0.005, +0.026] | +0.017 [+0.005, +0.027] | +0.007 [-0.001, +0.016] | +0.008 [+0.000, +0.016] | +0.010 [+0.001, +0.018] |
| 32B P | all | +0.063 [+0.053, +0.074] | +0.052 [+0.042, +0.062] | +0.041 [+0.031, +0.052] | +0.045 [+0.035, +0.056] | +0.022 [+0.016, +0.027] | +0.023 [+0.017, +0.029] | +0.021 [+0.016, +0.026] | +0.018 [+0.014, +0.023] | +0.023 [+0.018, +0.029] | +0.023 [+0.017, +0.029] | +0.009 [+0.005, +0.013] | +0.015 [+0.010, +0.019] | +0.015 [+0.011, +0.019] | +0.008 [+0.005, +0.012] | +0.013 [+0.008, +0.017] | +0.013 [+0.008, +0.017] |
| 32B P | control | +0.043 [+0.023, +0.062] | +0.026 [+0.008, +0.042] | +0.027 [+0.009, +0.044] | +0.026 [+0.007, +0.044] | -0.001 [-0.004, +0.001] | -0.000 [-0.003, +0.002] | -0.001 [-0.004, +0.001] | -0.003 [-0.005, -0.000] | -0.001 [-0.004, +0.001] | -0.001 [-0.003, +0.001] | +0.005 [-0.001, +0.014] | -0.001 [-0.004, +0.002] | -0.000 [-0.003, +0.003] | +0.003 [-0.003, +0.010] | +0.000 [-0.005, +0.007] | +0.002 [-0.004, +0.008] |
| 32B P | date | +0.073 [+0.056, +0.090] | +0.067 [+0.050, +0.083] | +0.047 [+0.030, +0.064] | +0.056 [+0.040, +0.073] | +0.046 [+0.031, +0.060] | +0.052 [+0.035, +0.069] | +0.047 [+0.032, +0.063] | +0.039 [+0.026, +0.053] | +0.052 [+0.035, +0.069] | +0.052 [+0.035, +0.069] | +0.016 [+0.004, +0.028] | +0.037 [+0.025, +0.050] | +0.038 [+0.025, +0.050] | +0.020 [+0.009, +0.032] | +0.033 [+0.020, +0.047] | +0.033 [+0.020, +0.046] |
| 32B P | mc_excl | +0.058 [+0.042, +0.076] | +0.044 [+0.029, +0.060] | +0.035 [+0.017, +0.057] | +0.041 [+0.023, +0.062] | +0.024 [+0.014, +0.032] | +0.018 [+0.010, +0.025] | +0.018 [+0.010, +0.025] | +0.020 [+0.010, +0.028] | +0.022 [+0.013, +0.030] | +0.019 [+0.011, +0.027] | +0.002 [-0.004, +0.006] | +0.010 [+0.004, +0.016] | +0.009 [+0.004, +0.015] | +0.002 [-0.003, +0.007] | +0.007 [+0.001, +0.013] | +0.004 [-0.001, +0.009] |
| 32B P | threshold | +0.081 [+0.058, +0.105] | +0.077 [+0.054, +0.102] | +0.057 [+0.035, +0.085] | +0.062 [+0.041, +0.085] | +0.028 [+0.014, +0.041] | +0.034 [+0.019, +0.048] | +0.031 [+0.017, +0.044] | +0.026 [+0.015, +0.037] | +0.032 [+0.018, +0.046] | +0.032 [+0.018, +0.046] | +0.015 [+0.008, +0.025] | +0.021 [+0.011, +0.031] | +0.023 [+0.012, +0.033] | +0.013 [+0.007, +0.019] | +0.018 [+0.009, +0.029] | +0.020 [+0.011, +0.030] |
| 32B Q | all | +0.211 [+0.192, +0.233] | +0.221 [+0.200, +0.243] | +0.213 [+0.194, +0.234] | +0.220 [+0.200, +0.241] | +0.019 [+0.014, +0.023] | +0.022 [+0.017, +0.027] | +0.021 [+0.016, +0.027] | +0.022 [+0.016, +0.027] | +0.020 [+0.015, +0.025] | +0.019 [+0.014, +0.024] | +0.019 [+0.011, +0.026] | +0.020 [+0.015, +0.026] | +0.021 [+0.015, +0.026] | +0.018 [+0.011, +0.024] | +0.020 [+0.014, +0.026] | +0.021 [+0.015, +0.027] |
| 32B Q | control | +0.185 [+0.155, +0.214] | +0.190 [+0.160, +0.219] | +0.208 [+0.175, +0.240] | +0.189 [+0.159, +0.218] | -0.000 [-0.002, +0.001] | +0.001 [-0.002, +0.003] | +0.001 [-0.002, +0.003] | +0.001 [-0.002, +0.003] | +0.000 [-0.002, +0.002] | +0.000 [-0.002, +0.002] | +0.019 [+0.005, +0.034] | +0.002 [-0.002, +0.005] | +0.002 [-0.002, +0.005] | +0.014 [+0.003, +0.026] | +0.003 [-0.002, +0.008] | +0.002 [-0.002, +0.007] |
| 32B Q | date | +0.276 [+0.217, +0.332] | +0.300 [+0.240, +0.357] | +0.239 [+0.180, +0.295] | +0.294 [+0.234, +0.351] | +0.038 [+0.024, +0.053] | +0.050 [+0.033, +0.069] | +0.049 [+0.032, +0.067] | +0.049 [+0.032, +0.068] | +0.042 [+0.027, +0.058] | +0.041 [+0.026, +0.056] | +0.018 [+0.004, +0.036] | +0.045 [+0.030, +0.061] | +0.042 [+0.028, +0.056] | +0.020 [+0.008, +0.032] | +0.046 [+0.030, +0.065] | +0.042 [+0.027, +0.058] |
| 32B Q | mc_excl | +0.194 [+0.161, +0.227] | +0.186 [+0.153, +0.218] | +0.183 [+0.151, +0.214] | +0.186 [+0.154, +0.219] | +0.021 [+0.012, +0.029] | +0.016 [+0.009, +0.023] | +0.016 [+0.008, +0.023] | +0.016 [+0.009, +0.023] | +0.019 [+0.011, +0.027] | +0.017 [+0.009, +0.024] | +0.006 [+0.001, +0.012] | +0.013 [+0.004, +0.023] | +0.017 [+0.006, +0.029] | +0.012 [+0.005, +0.017] | +0.010 [+0.000, +0.020] | +0.015 [+0.005, +0.027] |
| 32B Q | threshold | +0.216 [+0.170, +0.262] | +0.238 [+0.188, +0.289] | +0.233 [+0.184, +0.281] | +0.240 [+0.191, +0.290] | +0.024 [+0.014, +0.034] | +0.032 [+0.020, +0.044] | +0.031 [+0.019, +0.043] | +0.031 [+0.019, +0.043] | +0.027 [+0.016, +0.038] | +0.026 [+0.015, +0.037] | +0.030 [+0.013, +0.050] | +0.032 [+0.019, +0.046] | +0.031 [+0.019, +0.044] | +0.026 [+0.010, +0.042] | +0.032 [+0.019, +0.047] | +0.032 [+0.020, +0.046] |

### structured: dependence captured D (median / share ≥ 0.5), own marginals

| job | type | raw | eq_kl | bc_kl | rw_kl | rwg_kl | fill_and3 | fill_and4 | fill_litand |
|---|---|---|---|---|---|---|---|---|---|
| 235B P | threshold | 1.00 / 0.95 | 0.45 / 0.37 | 0.35 / 0.23 | 0.42 / 0.35 | 0.51 / 0.52 | 1.00 / 0.98 | 0.98 / 0.87 | 0.98 / 0.87 |
| 235B P | date | 1.00 / 0.77 | 0.39 / 0.30 | 0.31 / 0.22 | 0.38 / 0.22 | 0.54 / 0.52 | 1.00 / 0.81 | 0.95 / 0.81 | 0.95 / 0.81 |
| 235B P | mc_excl | 0.29 / 0.48 | 0.15 / 0.21 | 0.31 / 0.30 | 0.33 / 0.37 | 0.50 / 0.52 | 0.71 / 0.54 | 0.99 / 0.87 | 0.99 / 0.89 |
| 235B Q | threshold | 1.00 / 0.83 | 0.46 / 0.45 | 0.29 / 0.23 | 0.34 / 0.31 | 0.32 / 0.27 | 0.73 / 0.80 | 0.80 / 0.67 | 0.74 / 0.64 |
| 235B Q | date | 1.89 / 0.84 | 0.46 / 0.41 | 0.33 / 0.27 | 0.30 / 0.27 | 0.31 / 0.27 | 0.72 / 0.83 | 0.78 / 0.71 | 0.68 / 0.67 |
| 235B Q | mc_excl | 0.85 / 0.87 | 0.49 / 0.47 | 0.53 / 0.58 | 0.53 / 0.54 | 0.54 / 0.56 | 0.27 / 0.12 | 0.99 / 0.93 | 0.93 / 0.78 |
| 32B P | threshold | 1.00 / 0.77 | 0.26 / 0.19 | 0.14 / 0.11 | 0.21 / 0.13 | 0.32 / 0.29 | 0.87 / 0.72 | 0.54 / 0.51 | 0.63 / 0.57 |
| 32B P | date | 1.00 / 0.83 | 0.20 / 0.06 | 0.12 / 0.03 | 0.17 / 0.05 | 0.30 / 0.15 | 0.85 / 0.71 | 0.46 / 0.47 | 0.52 / 0.51 |
| 32B P | mc_excl | 1.00 / 0.94 | 0.42 / 0.43 | 0.55 / 0.55 | 0.58 / 0.57 | 0.57 / 0.58 | 0.91 / 0.90 | 0.98 / 0.88 | 0.99 / 0.90 |
| 32B Q | threshold | 1.00 / 0.62 | 0.31 / 0.33 | 0.13 / 0.18 | 0.15 / 0.19 | 0.13 / 0.18 | 0.63 / 0.59 | 0.18 / 0.15 | 0.17 / 0.20 |
| 32B Q | date | 1.00 / 0.74 | 0.21 / 0.15 | 0.04 / 0.07 | 0.07 / 0.08 | 0.05 / 0.08 | 0.65 / 0.73 | 0.10 / 0.07 | 0.14 / 0.09 |
| 32B Q | mc_excl | 1.00 / 0.92 | 0.67 / 0.68 | 0.73 / 0.77 | 0.76 / 0.77 | 0.75 / 0.77 | 0.72 / 0.73 | 0.69 / 0.70 | 0.67 / 0.66 |

### structured controls: log odds ratio (median / share |log OR| > 1)

| job | eq_kl | eq_lsl | bc_kl | rw_kl | rw_lsl | rwg_kl | bcx_kl | rwx_kl | fill_and3 | fill_and4 | fill_litand |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 235B P | +0.35 / 0.30 | +0.37 / 0.30 | -0.15 / 0.36 | -0.02 / 0.36 | -0.09 / 0.34 | +0.09 / 0.39 | +0.13 / 0.32 | +0.04 / 0.36 | +0.36 / 0.48 | +1.14 / 0.84 | +1.39 / 0.89 |
| 235B Q | +0.37 / 0.06 | +0.44 / 0.11 | -0.31 / 0.09 | -0.16 / 0.07 | -0.09 / 0.05 | -0.24 / 0.08 | +0.13 / 0.06 | +0.06 / 0.05 | +2.23 / 0.66 | +0.12 / 0.44 | +0.06 / 0.39 |
| 32B P | +0.21 / 0.36 | +0.17 / 0.39 | -0.24 / 0.37 | -0.14 / 0.37 | -0.12 / 0.40 | +0.07 / 0.39 | +0.05 / 0.35 | +0.01 / 0.36 | +0.02 / 0.42 | +0.35 / 0.61 | +0.29 / 0.70 |
| 32B Q | +0.08 / 0.08 | +0.10 / 0.10 | -0.44 / 0.17 | -0.45 / 0.16 | -0.42 / 0.13 | -0.45 / 0.16 | -0.14 / 0.11 | -0.19 / 0.12 | +0.10 / 0.51 | -0.38 / 0.10 | -0.33 / 0.14 |

### market_context: model-only methods, Δ vs equal-weight KL projection (joint / all-12)

| job | stratum | eq_kl joint | eq_lsl | bc_kl | rw_kl | rw_lsl | rwg_kl | bcx_kl | rwx_kl | fill_and3 | fill_and4 | fill_litand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 235B L | all (723) | 0.349 | -0.009 [-0.012, -0.006] / -0.009 | +0.003 [+0.001, +0.005] / +0.003 | -0.000 [-0.003, +0.002] / -0.000 | -0.009 [-0.013, -0.005] / -0.010 | -0.002 [-0.007, +0.002] / -0.002 | +0.002 [+0.000, +0.003] / +0.002 | +0.009 [+0.007, +0.011] / +0.009 | -0.006 [-0.011, -0.001] / -0.004 | +0.000 [-0.003, +0.003] / +0.000 | +0.001 [-0.003, +0.004] / +0.000 |
| 235B L | other (403) | 0.358 | -0.009 [-0.013, -0.005] / -0.010 | +0.002 [-0.001, +0.005] / +0.002 | -0.001 [-0.005, +0.002] / -0.001 | -0.010 [-0.016, -0.003] / -0.011 | -0.003 [-0.009, +0.003] / -0.003 | +0.001 [-0.001, +0.003] / +0.001 | +0.010 [+0.007, +0.013] / +0.011 | -0.006 [-0.012, +0.001] / -0.004 | -0.000 [-0.004, +0.004] / -0.001 | +0.000 [-0.004, +0.005] / -0.000 |
| 235B L | same_topic (320) | 0.338 | -0.009 [-0.013, -0.003] / -0.009 | +0.004 [+0.002, +0.007] / +0.004 | +0.001 [-0.002, +0.005] / +0.002 | -0.008 [-0.013, -0.002] / -0.008 | -0.001 [-0.008, +0.004] / -0.000 | +0.002 [+0.000, +0.004] / +0.002 | +0.007 [+0.004, +0.010] / +0.007 | -0.006 [-0.013, +0.001] / -0.003 | +0.001 [-0.003, +0.005] / +0.002 | +0.001 [-0.003, +0.006] / +0.002 |
| 32B L | all (723) | 0.386 | +0.002 [-0.004, +0.009] / +0.001 | +0.006 [+0.004, +0.008] / +0.006 | +0.000 [-0.003, +0.003] / -0.001 | -0.003 [-0.009, +0.003] / -0.005 | -0.002 [-0.008, +0.003] / -0.004 | +0.003 [+0.002, +0.005] / +0.003 | +0.003 [+0.000, +0.006] / +0.003 | -0.006 [-0.011, -0.001] / -0.005 | -0.003 [-0.007, +0.000] / -0.003 | -0.003 [-0.007, +0.001] / -0.003 |
| 32B L | other (403) | 0.394 | +0.005 [-0.004, +0.015] / +0.003 | +0.007 [+0.004, +0.010] / +0.006 | -0.001 [-0.005, +0.003] / -0.002 | -0.005 [-0.012, +0.004] / -0.006 | -0.005 [-0.012, +0.002] / -0.007 | +0.002 [+0.000, +0.005] / +0.002 | +0.003 [-0.001, +0.006] / +0.003 | -0.007 [-0.015, +0.001] / -0.006 | -0.004 [-0.010, +0.001] / -0.004 | -0.004 [-0.009, +0.001] / -0.004 |
| 32B L | same_topic (320) | 0.376 | -0.002 [-0.009, +0.006] / -0.003 | +0.005 [+0.002, +0.008] / +0.005 | +0.001 [-0.003, +0.006] / +0.000 | -0.002 [-0.010, +0.007] / -0.004 | +0.001 [-0.006, +0.009] / -0.001 | +0.004 [+0.002, +0.006] / +0.003 | +0.004 [-0.000, +0.009] / +0.004 | -0.005 [-0.012, +0.001] / -0.004 | -0.002 [-0.007, +0.003] / -0.002 | -0.002 [-0.007, +0.003] / -0.002 |
| 32B P | all (723) | 0.363 | -0.002 [-0.008, +0.004] / -0.003 | +0.006 [+0.005, +0.008] / +0.006 | -0.003 [-0.007, +0.000] / -0.004 | -0.008 [-0.014, -0.002] / -0.010 | -0.009 [-0.017, -0.001] / -0.011 | +0.003 [+0.002, +0.005] / +0.003 | +0.008 [+0.006, +0.011] / +0.009 | -0.006 [-0.010, -0.002] / -0.006 | -0.002 [-0.006, +0.001] / -0.003 | -0.002 [-0.006, +0.001] / -0.003 |
| 32B P | other (403) | 0.361 | -0.003 [-0.010, +0.005] / -0.004 | +0.006 [+0.004, +0.008] / +0.006 | -0.005 [-0.009, -0.001] / -0.006 | -0.011 [-0.018, -0.003] / -0.012 | -0.013 [-0.023, -0.003] / -0.015 | +0.003 [+0.001, +0.005] / +0.003 | +0.010 [+0.006, +0.013] / +0.010 | -0.007 [-0.012, -0.002] / -0.007 | -0.003 [-0.007, +0.002] / -0.004 | -0.003 [-0.007, +0.002] / -0.004 |
| 32B P | same_topic (320) | 0.366 | -0.001 [-0.010, +0.009] / -0.003 | +0.007 [+0.004, +0.010] / +0.007 | -0.001 [-0.005, +0.004] / -0.002 | -0.005 [-0.014, +0.004] / -0.007 | -0.004 [-0.015, +0.007] / -0.006 | +0.004 [+0.002, +0.006] / +0.003 | +0.007 [+0.003, +0.010] / +0.007 | -0.005 [-0.012, +0.002] / -0.004 | -0.002 [-0.008, +0.004] / -0.003 | -0.002 [-0.008, +0.004] / -0.002 |
| 32B Q | all (723) | 0.488 | -0.001 [-0.003, +0.001] / -0.001 | +0.009 [+0.007, +0.011] / +0.009 | +0.007 [+0.004, +0.010] / +0.008 | +0.010 [+0.005, +0.014] / +0.009 | +0.008 [+0.005, +0.010] / +0.008 | +0.007 [+0.005, +0.008] / +0.006 | +0.008 [+0.005, +0.012] / +0.007 | +0.011 [+0.006, +0.016] / +0.010 | +0.007 [+0.003, +0.012] / +0.008 | +0.007 [+0.003, +0.012] / +0.008 |
| 32B Q | other (403) | 0.471 | +0.000 [-0.002, +0.003] / -0.000 | +0.007 [+0.005, +0.010] / +0.008 | +0.007 [+0.003, +0.011] / +0.007 | +0.009 [+0.004, +0.015] / +0.009 | +0.007 [+0.004, +0.010] / +0.007 | +0.005 [+0.003, +0.007] / +0.005 | +0.007 [+0.003, +0.012] / +0.007 | +0.010 [+0.004, +0.016] / +0.009 | +0.005 [-0.000, +0.011] / +0.006 | +0.006 [+0.001, +0.011] / +0.006 |
| 32B Q | same_topic (320) | 0.509 | -0.002 [-0.005, +0.001] / -0.003 | +0.011 [+0.008, +0.013] / +0.011 | +0.008 [+0.003, +0.012] / +0.008 | +0.011 [+0.004, +0.018] / +0.010 | +0.009 [+0.005, +0.013] / +0.009 | +0.009 [+0.006, +0.011] / +0.008 | +0.009 [+0.005, +0.014] / +0.008 | +0.012 [+0.005, +0.020] / +0.011 | +0.010 [+0.003, +0.018] / +0.010 | +0.010 [+0.003, +0.017] / +0.009 |

### market_context: Δ vs raw (joint / all-12)

| job | stratum | raw joint | raw all12 | eq_kl | eq_lsl | bc_kl | rw_kl | rw_lsl | rwg_kl | bcx_kl | rwx_kl | fill_and3 | fill_and4 | fill_litand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 235B L | all (723) | 0.447 | 0.426 | -0.098 / -0.058 | -0.107 / -0.067 | -0.095 / -0.055 | -0.098 / -0.058 | -0.107 / -0.067 | -0.100 / -0.060 | -0.096 / -0.056 | -0.089 / -0.048 | -0.104 / -0.062 | -0.098 / -0.057 | -0.097 / -0.057 |
| 32B L | all (723) | 0.565 | 0.528 | -0.179 / -0.120 | -0.177 / -0.120 | -0.173 / -0.115 | -0.179 / -0.122 | -0.182 / -0.126 | -0.182 / -0.124 | -0.176 / -0.118 | -0.176 / -0.117 | -0.185 / -0.126 | -0.183 / -0.124 | -0.182 / -0.124 |
| 32B P | all (723) | 0.528 | 0.487 | -0.164 / -0.104 | -0.167 / -0.108 | -0.158 / -0.098 | -0.167 / -0.108 | -0.172 / -0.114 | -0.173 / -0.115 | -0.161 / -0.101 | -0.156 / -0.096 | -0.170 / -0.110 | -0.167 / -0.108 | -0.167 / -0.108 |
| 32B Q | all (723) | 0.537 | 0.597 | -0.050 / -0.076 | -0.050 / -0.078 | -0.041 / -0.067 | -0.042 / -0.068 | -0.040 / -0.067 | -0.042 / -0.068 | -0.043 / -0.070 | -0.041 / -0.069 | -0.039 / -0.066 | -0.042 / -0.068 | -0.042 / -0.068 |

### market_context: market variants, joint Δ vs market + independence [95% CI]

| job | stratum | market indep | h_eq | h_bc | h_rw | h_rwg | h_bcx | h_rwx | h_and3 | h_and4 | h_litand | h_and3_s | h_and4_s | h_litand_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 235B L | all (723) | 0.313 | -0.000 [-0.003, +0.002] | +0.000 [-0.002, +0.003] | +0.000 [-0.002, +0.003] | +0.000 [-0.003, +0.003] | -0.000 [-0.002, +0.002] | -0.000 [-0.002, +0.002] | +0.007 [-0.002, +0.016] | +0.001 [-0.003, +0.004] | +0.000 [-0.003, +0.004] | -0.001 [-0.004, +0.003] | -0.000 [-0.002, +0.001] | +0.000 [-0.001, +0.002] |
| 235B L | other (403) | 0.312 | -0.001 [-0.004, +0.003] | -0.000 [-0.003, +0.003] | -0.000 [-0.003, +0.003] | +0.000 [-0.003, +0.004] | -0.001 [-0.004, +0.002] | -0.001 [-0.004, +0.002] | +0.008 [-0.003, +0.021] | +0.001 [-0.003, +0.006] | +0.001 [-0.003, +0.006] | -0.000 [-0.005, +0.005] | -0.000 [-0.002, +0.002] | +0.000 [-0.002, +0.002] |
| 235B L | same_topic (320) | 0.313 | +0.001 [-0.003, +0.005] | +0.001 [-0.002, +0.005] | +0.001 [-0.003, +0.004] | -0.000 [-0.004, +0.003] | +0.001 [-0.002, +0.004] | +0.000 [-0.003, +0.003] | +0.005 [-0.007, +0.019] | +0.000 [-0.005, +0.005] | -0.001 [-0.005, +0.004] | -0.002 [-0.006, +0.003] | -0.000 [-0.002, +0.002] | -0.000 [-0.002, +0.002] |
| 32B L | all (723) | 0.313 | +0.002 [-0.000, +0.005] | +0.003 [+0.001, +0.006] | +0.003 [+0.000, +0.006] | +0.003 [-0.000, +0.006] | +0.003 [+0.000, +0.006] | +0.002 [-0.000, +0.005] | +0.007 [+0.000, +0.016] | +0.001 [-0.002, +0.004] | +0.001 [-0.002, +0.004] | +0.001 [-0.001, +0.003] | +0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.002] |
| 32B L | other (403) | 0.312 | +0.003 [-0.001, +0.006] | +0.004 [-0.000, +0.008] | +0.003 [-0.000, +0.007] | +0.003 [-0.001, +0.007] | +0.003 [-0.001, +0.007] | +0.003 [-0.001, +0.006] | +0.008 [-0.003, +0.019] | +0.001 [-0.003, +0.006] | +0.001 [-0.002, +0.005] | +0.002 [-0.001, +0.005] | +0.001 [-0.001, +0.002] | +0.001 [-0.001, +0.002] |
| 32B L | same_topic (320) | 0.313 | +0.001 [-0.002, +0.005] | +0.003 [-0.001, +0.007] | +0.003 [-0.001, +0.007] | +0.003 [-0.001, +0.008] | +0.003 [-0.001, +0.007] | +0.002 [-0.001, +0.006] | +0.007 [-0.002, +0.018] | +0.001 [-0.004, +0.005] | +0.001 [-0.004, +0.006] | -0.000 [-0.003, +0.002] | +0.000 [-0.001, +0.002] | +0.000 [-0.002, +0.002] |
| 32B P | all (723) | 0.313 | +0.000 [-0.002, +0.003] | +0.001 [-0.002, +0.004] | +0.001 [-0.002, +0.004] | +0.002 [-0.001, +0.006] | +0.001 [-0.001, +0.004] | +0.001 [-0.001, +0.004] | +0.006 [-0.001, +0.014] | +0.002 [-0.001, +0.005] | +0.002 [-0.001, +0.005] | +0.000 [-0.002, +0.002] | +0.001 [+0.000, +0.002] | +0.002 [+0.001, +0.003] |
| 32B P | other (403) | 0.312 | -0.001 [-0.004, +0.003] | -0.000 [-0.004, +0.003] | -0.000 [-0.004, +0.003] | +0.002 [-0.003, +0.007] | -0.000 [-0.004, +0.003] | +0.000 [-0.003, +0.004] | +0.005 [-0.004, +0.015] | +0.001 [-0.003, +0.005] | +0.001 [-0.003, +0.005] | -0.000 [-0.003, +0.002] | +0.001 [+0.000, +0.002] | +0.002 [+0.000, +0.003] |
| 32B P | same_topic (320) | 0.313 | +0.002 [-0.002, +0.006] | +0.003 [-0.001, +0.008] | +0.003 [-0.001, +0.008] | +0.004 [-0.002, +0.009] | +0.003 [-0.001, +0.008] | +0.003 [-0.001, +0.007] | +0.008 [-0.003, +0.020] | +0.004 [-0.001, +0.008] | +0.003 [-0.001, +0.007] | +0.000 [-0.002, +0.003] | +0.001 [+0.000, +0.002] | +0.002 [+0.001, +0.003] |
| 32B Q | all (723) | 0.313 | +0.000 [-0.002, +0.002] | +0.001 [-0.001, +0.003] | +0.000 [-0.002, +0.003] | +0.001 [-0.001, +0.003] | +0.001 [-0.001, +0.003] | +0.001 [-0.001, +0.003] | +0.020 [+0.009, +0.031] | +0.003 [+0.000, +0.005] | +0.002 [-0.000, +0.005] | +0.002 [+0.000, +0.003] | -0.000 [-0.000, +0.000] | -0.000 [-0.000, +0.000] |
| 32B Q | other (403) | 0.312 | +0.000 [-0.002, +0.003] | +0.001 [-0.002, +0.003] | +0.001 [-0.002, +0.003] | +0.001 [-0.002, +0.003] | +0.001 [-0.002, +0.004] | +0.001 [-0.001, +0.004] | +0.017 [+0.004, +0.032] | +0.002 [-0.002, +0.006] | +0.003 [-0.001, +0.007] | +0.002 [+0.000, +0.004] | -0.000 [-0.000, +0.000] | -0.000 [-0.000, +0.000] |
| 32B Q | same_topic (320) | 0.313 | -0.000 [-0.003, +0.003] | +0.001 [-0.003, +0.004] | +0.000 [-0.003, +0.004] | +0.000 [-0.003, +0.004] | +0.001 [-0.003, +0.005] | +0.001 [-0.003, +0.005] | +0.022 [+0.005, +0.041] | +0.003 [-0.000, +0.007] | +0.002 [-0.001, +0.006] | +0.001 [-0.000, +0.003] | +0.000 [-0.000, +0.000] | +0.000 [-0.000, +0.000] |

### market_context: market variants, joint Δ vs the old hybrid h_eq (paired) [95% CI]

| job | stratum | h_bc | h_rw | h_rwg | h_bcx | h_rwx | h_and3 | h_and4 | h_litand | h_and3_s | h_and4_s | h_litand_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 235B L | all (723) | +0.000 [-0.000, +0.001] | +0.000 [-0.001, +0.001] | +0.000 [-0.002, +0.002] | -0.000 [-0.001, +0.001] | -0.000 [-0.001, +0.001] | +0.007 [-0.001, +0.015] | +0.001 [-0.001, +0.003] | +0.000 [-0.001, +0.002] | -0.001 [-0.004, +0.003] | -0.000 [-0.002, +0.002] | +0.000 [-0.002, +0.002] |
| 235B L | other (403) | +0.001 [-0.001, +0.002] | +0.001 [-0.001, +0.002] | +0.001 [-0.001, +0.003] | -0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.001] | +0.008 [-0.002, +0.020] | +0.002 [-0.000, +0.005] | +0.002 [-0.000, +0.005] | +0.001 [-0.004, +0.006] | +0.001 [-0.002, +0.003] | +0.001 [-0.001, +0.004] |
| 235B L | same_topic (320) | +0.000 [-0.001, +0.001] | -0.000 [-0.002, +0.001] | -0.001 [-0.004, +0.001] | -0.000 [-0.001, +0.001] | -0.000 [-0.002, +0.001] | +0.004 [-0.007, +0.017] | -0.001 [-0.003, +0.002] | -0.002 [-0.005, +0.001] | -0.003 [-0.007, +0.002] | -0.001 [-0.004, +0.002] | -0.001 [-0.004, +0.002] |
| 32B L | all (723) | +0.001 [+0.000, +0.002] | +0.001 [-0.000, +0.002] | +0.001 [-0.001, +0.003] | +0.001 [-0.000, +0.001] | +0.000 [-0.000, +0.001] | +0.005 [-0.002, +0.013] | -0.001 [-0.003, +0.001] | -0.001 [-0.004, +0.001] | -0.001 [-0.005, +0.002] | -0.002 [-0.004, +0.000] | -0.002 [-0.004, +0.000] |
| 32B L | other (403) | +0.001 [-0.000, +0.002] | +0.001 [-0.001, +0.002] | +0.000 [-0.002, +0.003] | +0.000 [-0.001, +0.001] | -0.000 [-0.001, +0.001] | +0.005 [-0.005, +0.017] | -0.001 [-0.004, +0.002] | -0.001 [-0.005, +0.002] | -0.001 [-0.006, +0.004] | -0.002 [-0.005, +0.001] | -0.002 [-0.005, +0.001] |
| 32B L | same_topic (320) | +0.001 [-0.000, +0.003] | +0.001 [-0.000, +0.003] | +0.002 [-0.001, +0.005] | +0.001 [+0.000, +0.003] | +0.001 [-0.000, +0.002] | +0.006 [-0.003, +0.016] | -0.001 [-0.004, +0.003] | -0.001 [-0.004, +0.003] | -0.002 [-0.005, +0.002] | -0.001 [-0.004, +0.002] | -0.001 [-0.004, +0.002] |
| 32B P | all (723) | +0.001 [-0.000, +0.002] | +0.001 [-0.000, +0.002] | +0.002 [-0.001, +0.005] | +0.001 [+0.000, +0.002] | +0.001 [+0.000, +0.002] | +0.006 [-0.001, +0.014] | +0.002 [-0.001, +0.004] | +0.001 [-0.001, +0.004] | -0.000 [-0.003, +0.002] | +0.001 [-0.002, +0.003] | +0.001 [-0.001, +0.003] |
| 32B P | other (403) | +0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.002] | +0.002 [-0.001, +0.007] | +0.000 [-0.000, +0.001] | +0.001 [+0.000, +0.002] | +0.005 [-0.003, +0.016] | +0.002 [-0.001, +0.004] | +0.002 [-0.001, +0.005] | +0.001 [-0.003, +0.004] | +0.002 [-0.001, +0.005] | +0.002 [-0.000, +0.005] |
| 32B P | same_topic (320) | +0.001 [-0.000, +0.003] | +0.001 [-0.001, +0.003] | +0.002 [-0.002, +0.005] | +0.001 [-0.000, +0.003] | +0.001 [-0.000, +0.002] | +0.006 [-0.005, +0.018] | +0.002 [-0.002, +0.005] | +0.001 [-0.003, +0.004] | -0.002 [-0.006, +0.003] | -0.001 [-0.005, +0.002] | -0.000 [-0.004, +0.003] |
| 32B Q | all (723) | +0.001 [+0.000, +0.001] | +0.000 [-0.000, +0.001] | +0.000 [-0.000, +0.001] | +0.001 [+0.000, +0.002] | +0.001 [+0.000, +0.002] | +0.019 [+0.009, +0.030] | +0.002 [+0.001, +0.004] | +0.002 [+0.001, +0.004] | +0.002 [-0.000, +0.004] | -0.000 [-0.002, +0.002] | -0.000 [-0.002, +0.002] |
| 32B Q | other (403) | +0.001 [-0.000, +0.001] | +0.000 [-0.000, +0.001] | +0.000 [-0.000, +0.001] | +0.001 [-0.000, +0.002] | +0.001 [-0.000, +0.002] | +0.017 [+0.005, +0.031] | +0.002 [-0.001, +0.004] | +0.002 [-0.000, +0.005] | +0.002 [-0.001, +0.005] | -0.000 [-0.003, +0.002] | -0.000 [-0.003, +0.002] |
| 32B Q | same_topic (320) | +0.001 [-0.000, +0.001] | +0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.001] | +0.001 [-0.000, +0.002] | +0.001 [-0.000, +0.002] | +0.022 [+0.005, +0.040] | +0.003 [+0.001, +0.006] | +0.002 [+0.000, +0.005] | +0.001 [-0.002, +0.004] | +0.000 [-0.003, +0.003] | +0.000 [-0.003, +0.003] |

### market_context: market variants, all-12 Δ vs market + independence

| job | stratum | h_eq | h_bc | h_rw | h_rwg | h_bcx | h_rwx | h_and3 | h_and4 | h_litand | h_and3_s | h_and4_s | h_litand_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 235B L | all (723) | -0.000 [-0.002, +0.002] | +0.000 [-0.001, +0.002] | +0.000 [-0.001, +0.002] | +0.000 [-0.002, +0.002] | -0.000 [-0.001, +0.001] | -0.000 [-0.002, +0.001] | +0.004 [-0.001, +0.010] | +0.001 [-0.002, +0.003] | +0.000 [-0.002, +0.002] | -0.001 [-0.003, +0.002] | -0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.001] |
| 235B L | other (403) | -0.001 [-0.003, +0.002] | -0.000 [-0.002, +0.002] | -0.000 [-0.002, +0.002] | +0.000 [-0.002, +0.003] | -0.001 [-0.002, +0.001] | -0.001 [-0.002, +0.001] | +0.005 [-0.002, +0.014] | +0.001 [-0.002, +0.004] | +0.001 [-0.002, +0.004] | -0.000 [-0.003, +0.003] | -0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.001] |
| 235B L | same_topic (320) | +0.001 [-0.002, +0.003] | +0.001 [-0.002, +0.003] | +0.000 [-0.002, +0.003] | -0.000 [-0.003, +0.002] | +0.001 [-0.002, +0.003] | +0.000 [-0.002, +0.002] | +0.004 [-0.005, +0.012] | +0.000 [-0.003, +0.003] | -0.000 [-0.003, +0.003] | -0.001 [-0.004, +0.002] | -0.000 [-0.001, +0.001] | -0.000 [-0.001, +0.001] |
| 32B L | all (723) | +0.001 [-0.000, +0.003] | +0.002 [+0.000, +0.004] | +0.002 [+0.000, +0.004] | +0.002 [-0.000, +0.004] | +0.002 [+0.000, +0.004] | +0.002 [-0.000, +0.004] | +0.005 [+0.000, +0.010] | +0.001 [-0.001, +0.003] | +0.001 [-0.001, +0.003] | +0.000 [-0.001, +0.002] | +0.000 [-0.001, +0.001] | +0.000 [-0.000, +0.001] |
| 32B L | other (403) | +0.002 [-0.001, +0.004] | +0.002 [-0.000, +0.005] | +0.002 [-0.000, +0.005] | +0.002 [-0.000, +0.004] | +0.002 [-0.001, +0.004] | +0.002 [-0.001, +0.004] | +0.005 [-0.002, +0.013] | +0.001 [-0.002, +0.004] | +0.001 [-0.002, +0.004] | +0.001 [-0.001, +0.003] | +0.000 [-0.001, +0.001] | +0.000 [-0.000, +0.001] |
| 32B L | same_topic (320) | +0.001 [-0.001, +0.003] | +0.002 [-0.001, +0.004] | +0.002 [-0.001, +0.004] | +0.002 [-0.001, +0.005] | +0.002 [-0.001, +0.004] | +0.002 [-0.001, +0.004] | +0.005 [-0.001, +0.012] | +0.000 [-0.003, +0.004] | +0.001 [-0.003, +0.004] | -0.000 [-0.002, +0.001] | +0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.001] |
| 32B P | all (723) | +0.000 [-0.001, +0.002] | +0.001 [-0.001, +0.003] | +0.001 [-0.001, +0.003] | +0.002 [-0.001, +0.004] | +0.001 [-0.001, +0.003] | +0.001 [-0.001, +0.003] | +0.004 [-0.001, +0.009] | +0.001 [-0.001, +0.003] | +0.001 [-0.001, +0.003] | +0.000 [-0.001, +0.001] | +0.001 [+0.000, +0.001] | +0.001 [+0.000, +0.002] |
| 32B P | other (403) | -0.001 [-0.003, +0.002] | -0.000 [-0.003, +0.002] | -0.000 [-0.003, +0.002] | +0.001 [-0.002, +0.005] | -0.000 [-0.003, +0.002] | +0.000 [-0.002, +0.003] | +0.003 [-0.003, +0.010] | +0.001 [-0.002, +0.003] | +0.001 [-0.002, +0.003] | -0.000 [-0.002, +0.001] | +0.001 [+0.000, +0.001] | +0.001 [+0.000, +0.002] |
| 32B P | same_topic (320) | +0.001 [-0.001, +0.004] | +0.002 [-0.001, +0.005] | +0.002 [-0.001, +0.005] | +0.002 [-0.001, +0.006] | +0.002 [-0.001, +0.005] | +0.002 [-0.001, +0.005] | +0.005 [-0.002, +0.013] | +0.002 [-0.000, +0.005] | +0.002 [-0.001, +0.005] | +0.000 [-0.001, +0.002] | +0.001 [+0.000, +0.001] | +0.001 [+0.000, +0.002] |
| 32B Q | all (723) | +0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.002] | +0.000 [-0.001, +0.002] | +0.000 [-0.001, +0.002] | +0.001 [-0.001, +0.002] | +0.001 [-0.001, +0.002] | +0.013 [+0.006, +0.021] | +0.002 [+0.000, +0.004] | +0.002 [-0.000, +0.004] | +0.001 [+0.000, +0.002] | -0.000 [-0.000, +0.000] | -0.000 [-0.000, +0.000] |
| 32B Q | other (403) | +0.000 [-0.001, +0.002] | +0.001 [-0.001, +0.002] | +0.000 [-0.001, +0.002] | +0.000 [-0.001, +0.002] | +0.001 [-0.001, +0.003] | +0.001 [-0.001, +0.003] | +0.012 [+0.003, +0.021] | +0.001 [-0.001, +0.004] | +0.002 [-0.001, +0.004] | +0.001 [+0.000, +0.003] | -0.000 [-0.000, +0.000] | -0.000 [-0.000, +0.000] |
| 32B Q | same_topic (320) | -0.000 [-0.002, +0.002] | +0.000 [-0.002, +0.003] | +0.000 [-0.002, +0.002] | +0.000 [-0.002, +0.002] | +0.001 [-0.002, +0.003] | +0.001 [-0.002, +0.003] | +0.015 [+0.003, +0.027] | +0.002 [-0.000, +0.005] | +0.002 [-0.001, +0.004] | +0.001 [-0.000, +0.002] | -0.000 [-0.000, +0.000] | -0.000 [-0.000, +0.000] |

### old_1066: model-only methods, Δ vs equal-weight KL projection (joint / all-12)

| job | stratum | eq_kl joint | eq_lsl | bc_kl | rw_kl | rw_lsl | rwg_kl | bcx_kl | rwx_kl | fill_and3 | fill_and4 | fill_litand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32B Q (DashScope t0) | all (1066) | 0.504 | +0.001 [-0.001, +0.003] / +0.000 | +0.005 [+0.004, +0.007] / +0.005 | +0.001 [-0.002, +0.003] / -0.001 | +0.002 [-0.001, +0.006] / +0.001 | +0.001 [-0.002, +0.003] / -0.001 | — | — | -0.004 [-0.009, +0.001] / -0.004 | +0.001 [-0.002, +0.005] / -0.000 | +0.001 [-0.002, +0.004] / -0.000 |
| 235B Q | all (1066) | 0.483 | +0.001 [-0.001, +0.002] / +0.001 | +0.002 [+0.001, +0.003] / +0.002 | +0.002 [+0.001, +0.003] / +0.003 | +0.002 [+0.001, +0.004] / +0.002 | +0.002 [+0.001, +0.003] / +0.002 | +0.002 [+0.001, +0.003] / +0.002 | +0.004 [+0.002, +0.007] / +0.005 | +0.000 [-0.003, +0.004] / +0.001 | +0.001 [-0.001, +0.003] / +0.002 | +0.000 [-0.002, +0.002] / +0.001 |

### old_1066: Δ vs raw (joint / all-12)

| job | stratum | raw joint | raw all12 | eq_kl | eq_lsl | bc_kl | rw_kl | rw_lsl | rwg_kl | bcx_kl | rwx_kl | fill_and3 | fill_and4 | fill_litand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32B Q (DashScope t0) | all (1066) | 0.558 | 0.622 | -0.055 / -0.081 | -0.054 / -0.081 | -0.049 / -0.076 | -0.054 / -0.082 | -0.052 / -0.080 | -0.054 / -0.082 | — | — | -0.058 / -0.085 | -0.053 / -0.081 | -0.054 / -0.082 |
| 235B Q | all (1066) | 0.523 | 0.553 | -0.040 / -0.037 | -0.040 / -0.036 | -0.039 / -0.035 | -0.038 / -0.034 | -0.038 / -0.035 | -0.039 / -0.035 | -0.038 / -0.035 | -0.036 / -0.032 | -0.040 / -0.036 | -0.040 / -0.036 | -0.040 / -0.036 |

### fitted parameters (mean over the 5 training folds)

| dataset | job | param | P | NOT P | Q | NOT Q | (P) AND (Q) | (P) OR (Q) | (P) AND (NOT Q) | (P) OR (NOT Q) | (NOT P) AND (Q) | (NOT P) OR (Q) | (NOT P) AND (NOT Q) | (NOT P) OR (NOT Q) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| external | 32B old test | bias | +0.06 | +0.08 | -0.08 | +0.03 | +0.41 | -0.17 | +0.22 | -0.10 | +0.09 | -0.10 | +0.55 | -0.08 |
| external | 32B old test | σ² | 0.75 | 1.55 | 0.74 | 1.70 | 0.41 | 0.96 | 0.57 | 0.70 | 0.58 | 0.59 | 0.84 | 0.63 |
| structured | 235B P | bias | +0.01 | +0.28 | -0.04 | +0.30 | +0.12 | +0.08 | +0.45 | +0.88 | +0.55 | +0.07 | +0.20 | -1.13 |
| structured | 235B P | weight | 1.53 | 0.49 | 1.21 | 0.60 | 0.97 | 2.41 | 1.01 | 0.51 | 0.94 | 0.78 | 1.13 | 0.41 |
| structured | 235B Q | bias | +0.03 | +0.54 | +0.15 | +0.38 | +0.55 | -0.01 | +0.42 | +0.97 | +0.50 | +0.44 | +0.26 | -0.40 |
| structured | 235B Q | weight | 1.04 | 1.17 | 1.31 | 1.07 | 0.77 | 1.13 | 1.26 | 0.55 | 1.42 | 0.69 | 0.92 | 0.68 |
| structured | 32B P | bias | -0.02 | +0.06 | -0.13 | +0.29 | -0.05 | +0.17 | +0.23 | +0.46 | +0.16 | +0.16 | +0.22 | -0.76 |
| structured | 32B P | weight | 1.68 | 1.53 | 1.66 | 1.37 | 0.98 | 0.89 | 0.64 | 0.42 | 0.66 | 0.62 | 0.68 | 0.88 |
| structured | 32B Q | bias | +0.27 | +0.26 | +0.17 | +0.37 | +0.52 | +0.04 | -0.10 | +0.62 | +0.30 | +0.38 | +0.56 | +0.22 |
| structured | 32B Q | weight | 1.39 | 1.04 | 1.73 | 0.82 | 0.91 | 1.06 | 0.86 | 0.67 | 0.90 | 1.08 | 0.92 | 0.62 |
| market_context | 235B L | bias | -0.31 | +0.43 | -0.40 | +0.50 | +0.17 | +0.07 | +0.69 | +0.70 | +0.64 | +0.11 | +0.35 | -0.75 |
| market_context | 235B L | weight | 1.33 | 1.21 | 0.91 | 0.87 | 1.69 | 1.02 | 0.86 | 0.67 | 0.91 | 0.80 | 0.79 | 0.94 |
| market_context | 32B L | bias | -0.36 | +0.43 | -0.41 | +0.39 | -0.03 | +0.80 | +0.32 | +0.53 | -0.18 | +0.33 | +0.12 | -0.53 |
| market_context | 32B L | weight | 1.33 | 1.37 | 1.09 | 0.96 | 1.76 | 0.49 | 0.84 | 0.62 | 0.95 | 0.80 | 0.74 | 1.04 |
| market_context | 32B P | bias | -0.28 | +0.36 | -0.23 | +0.37 | +0.01 | +0.69 | +0.18 | +0.30 | -0.06 | +0.46 | +0.32 | -0.31 |
| market_context | 32B P | weight | 1.42 | 1.46 | 1.09 | 1.20 | 1.77 | 0.51 | 0.61 | 0.60 | 0.74 | 0.77 | 0.70 | 1.12 |
| market_context | 32B Q | bias | +0.23 | +0.29 | +0.15 | +0.28 | +0.31 | -0.28 | +0.32 | -0.18 | +0.22 | -0.18 | +0.69 | -0.06 |
| market_context | 32B Q | weight | 0.81 | 0.42 | 0.90 | 0.39 | 1.86 | 1.11 | 0.95 | 0.88 | 1.04 | 1.51 | 0.86 | 1.26 |
| old_1066 | 32B Q (DashScope t0) | bias | +0.06 | +0.08 | -0.08 | +0.03 | +0.41 | -0.17 | +0.22 | -0.10 | +0.09 | -0.10 | +0.55 | -0.08 |
| old_1066 | 32B Q (DashScope t0) | weight | 0.95 | 0.46 | 0.96 | 0.42 | 1.75 | 0.74 | 1.26 | 1.02 | 1.23 | 1.22 | 0.85 | 1.13 |
| old_1066 | 235B Q | bias | -0.14 | +0.35 | -0.04 | +0.17 | +0.28 | -0.35 | +0.68 | +0.06 | +0.58 | -0.12 | +0.39 | -0.47 |
| old_1066 | 235B Q | weight | 0.59 | 0.76 | 0.75 | 0.76 | 1.44 | 1.22 | 1.35 | 0.97 | 1.19 | 1.10 | 0.86 | 1.01 |

| dataset | job | γ by fold | λ fill (and3 / and4 / litand) | λ hybrid (and3 / and4 / litand) |
|---|---|---|---|---|
| structured | 235B P | 3,3,3,3,3 | 3,3,3,3,3 / 3,3,3,3,3 / 3,3,3,3,3 | 3,3,3,3,3 / 3,3,3,3,3 / 3,3,3,3,3 |
| structured | 235B Q | 1,0.5,0,0.5,1 | 0.3,0.2,0.2,0.2,0.2 / 2.3,2,1.9,1.9,2 / 1.8,1.7,1.7,1.6,1.8 | 1.1,1.2,1.3,1.4,1.2 / 2.8,2.7,2.7,2.5,3 / 2.7,3,2.5,2.3,3 |
| structured | 32B P | 3,3,3,3,3 | 0.3,0.3,0.3,0.3,0.3 / 1.8,1.6,1.5,1.5,1.9 / 2.5,1.7,1.7,1.7,2.5 | 0.4,0.5,0.5,0.4,0.5 / 2,1.7,1.8,1.9,2 / 2.3,1.8,1.9,1.9,2.4 |
| structured | 32B Q | 0.5,1,0.5,0.5,0.5 | 0.2,0.2,0.2,0.2,0.2 / 0.6,0.6,0.6,0.6,0.7 / 0.6,0.6,0.7,0.6,0.7 | 0.3,0.3,0.3,0.3,0.4 / 1.4,1.5,1.4,1.2,1.4 / 1.2,1.5,1.3,1.1,1.3 |
| market_context | 235B L | 3,3,3,3,3 | 0.2,0.2,0.2,0.2,0.2 / 0.6,0.5,0.9,0.6,0.5 / 0.6,0.4,0.9,0.6,0.4 | 0.2,0.1,0.2,0.1,0.1 / 0.4,0.3,0.6,0.4,0.3 / 0.4,0.2,0.7,0.4,0.3 |
| market_context | 32B L | 2,2,2,2,3 | 0.2,0.2,0.1,0.1,0.1 / 0,0.3,0,0.1,0.1 / 0,0.5,0.1,0.1,0.2 | 0.1,0.1,0,0,0.1 / 0,0.5,0.3,0.2,0.3 / 0,0.6,0.3,0.2,0.3 |
| market_context | 32B P | 3,3,3,3,3 | 0.1,0.1,0.1,0.1,0.1 / 0,0,0,0.2,0.5 / 0,0,0.1,0.1,0.6 | 0.1,0.1,0,0.1,0.1 / 0,0,0,0.1,0.5 / 0,0,0,0.2,0.7 |
| market_context | 32B Q | 0.5,0.5,0.5,1,0.5 | 0,0.1,0,0,0 / 0,0.2,0,0.1,0 / 0,0.3,0.1,0.2,0.1 | 0,0.1,0,0,0 / 0,0,0,0,0 / 0,0,0,0,0 |
| old_1066 | 32B Q (DashScope t0) | 1,1,1,1,1 | 0.1,0.1,0.1,0.1,0.1 / 0.2,0.2,0.3,0.2,0 / 0.2,0.2,0.3,0.3,0.1 | — |
| old_1066 | 235B Q | 0,0,0,0,0 | 0.1,0.1,0.1,0.1,0.1 / 0.7,0.7,0.6,0.8,0.9 / 0.7,0.7,0.6,0.9,0.9 | — |
