# bt-scale results: consistency (C) vs plain (B) on 347 resolved MiniBench questions

Status as of 2026-09-30 06:40Z (numbers corrected for a scorer edge case; see the Corrected note). The run is complete. The analysis follows PREREG.md exactly (committed as 7abd43d
before any forecast; the analysis and audit scripts were committed before scoring). The headline numbers were recomputed by an independent
verifier; see Verification below. Nothing was submitted to Metaculus: MINIBENCH_LIVE was unset and no LIVE_APPROVED
file exists.

## Verdict: **No gain** (the pre-registered sub-label is *inconclusive*)

C does not beat B. The primary difference is **C − B = −0.30 baseline points, 95% CI [−1.92, +1.41]**, over
n = 347 questions (paired bootstrap, 10,000 resamples). The data rule out a C gain larger than about +1.4
points on all types, and larger than +0.6 points on binary questions alone. The CI does not reach below the
pre-registered MDE (1.1 points), so the label is "inconclusive" rather than "gain ruled out". That happened
because the numeric and discrete questions were noisier than the pilot suggested (see Power).

**What it means for live MiniBench:** keep B as the submit arm, as PREREG says for No gain. Consistency
projection did not improve short-horizon forecasts here. (A later fix to C's continuous blend gives C′ − B =
+0.69 [−0.45, +2.14]; that is exploratory, and B stays. See "After the continuous-blend fix".)

None of the Problem checks fired:

| check | result |
|---|---|
| P1: 2 or more audited questions with confirmed post-cut evidence | 1 of 20 |
| P2: B does not beat uniform | B − U = +31.3 [+24.6, +37.7] |
| P3: primary n below 250 | 347 |
| P4: model, guard or ledger chain failure | all 348 scored forecasts on claude-opus-5-5 (`model` and `models_used`), no guard errors, the 3 ledger chains verify |
| P5: scorer leak suspects above 5% | 0 of 348 |

## Sample

- 348 resolved questions: every scorable question in the six MiniBench rounds that opened after the model's
  June 2026 cutoff (33064, 33067, 33069, 33074, 33075, 33122).
- 318 were forecast fresh in bt-scale. The other 30 are the byte-identical bt-clean (22) and bt-clean-8 (8)
  runs, made with the same code.
- 0 questions were skipped. One question (45000) was dropped by the leak audit, so the primary n is 347:
  178 binary, 80 discrete, 76 numeric, 13 MC.

## Primary and breakdowns

C − B in Metaculus baseline points (higher means C is better), with 95% bootstrap CIs:

| slice | n | C − B | 95% CI |
|---|---|---|---|
| **all (primary)** | 347 | **−0.30** | [−1.92, +1.41] |
| all, round-clustered bootstrap | 347 (6 rounds) | −0.30 | [−2.37, +1.84] |
| binary | 178 | −0.44 | [−1.43, +0.61] |
| numeric | 76 | +0.46 | [−4.06, +4.84] |
| discrete | 80 | −0.69 | [−5.88, +4.97] |
| multiple choice | 13 | −0.37 | [−2.08, +0.80] |
| round 33064 (opens 06-29) | 58 | +1.49 | [−1.22, +4.75] |
| round 33067 (07-13) | 59 | −2.83 | [−8.07, +0.72] |
| round 33069 (07-27) | 58 | +0.80 | [−5.08, +7.99] |
| round 33074 (08-10) | 55 | −1.36 | [−3.72, +1.00] |
| round 33075 (08-24) | 59 | +1.27 | [−2.04, +5.15] |
| round 33122 (09-07) | 58 | −1.21 | [−3.18, +0.67] |
| sensitivity: without 33064 | 289 | −0.66 | [−2.51, +1.26] |
| sensitivity: new questions only | 317 | −0.35 | [−2.07, +1.47] |
| sensitivity: including the audit-dropped question | 348 | −0.28 | [−1.95, +1.41] |

No round and no type shows a CI that excludes 0.

The other metrics agree:

- **Log loss:** C − B = +0.003 nats [−0.030, +0.034] on all types and +0.003 [−0.004, +0.010] on binary.
  Lower is better, so both lean slightly against C.
- **Brier (binary + MC):** C − B = +0.001 [−0.002, +0.003]. Lower is better.

**Where C's effect comes from (binary).** C adds almost nothing beyond the P/NOT-P average (PN):

- C equals PN on 89% of binary questions.
- C − PN = +0.01 [−0.00, +0.03].
- PN − B = −0.45 [−1.45, +0.60].

So on binary questions, "consistency" in practice means averaging P with 1 − P(NOT); the chain projection
rarely binds on MiniBench's single-event questions. The averaging is neutral to slightly harmful.

## Against the bot crowd and uniform

- **B vs uniform:** B averages +32.2 baseline points (C +31.9). B − U = +31.3 [+24.6, +37.7]. It is above chance on every
  type: binary +37.1, numeric +21.9, discrete +27.1, MC +33.2, each with a CI above 0.
- **B vs the bot crowd** (binary + MC, 100·(ln p − ln cp), where cp is the crowd's final unweighted aggregate):
  B scores **+4.2 [+0.9, +7.4]** (n 191) and C +3.9 [+0.8, +6.9]; C − B = −0.31 [−0.96, +0.38].
- The crowd comparison favours B, with three caveats:
  - The crowd's aggregate is taken at close, so the crowd had up to 3 h more information than our as-of.
  - The crowd is bots, not Metaculus pros.
  - The one leak the audit found (below) means a small fraction of our research bundles may carry
    post-cutoff text. That could inflate B and C against the crowd and uniform. It barely affects C − B,
    because both arms saw the same bundle.
- **80% interval coverage** (numeric + discrete, n 156): B 0.80 [0.74, 0.86], C 0.81 [0.74, 0.87].

## Leak audit (random 20, pre-registered)

The 20 questions were drawn with `random.Random(20260931).sample(sorted(ids), 20)`: 17 bt-scale and 3 bt-clean.

| step | result |
|---|---|
| (a) scorer's strict filter + date re-check | 0 of 20 flagged; 0 of all 348 flagged |
| (b) regex scan for dates after the cutoff | 42 flags in 11 questions, all reviewed by hand: 34 were forward-looking schedules stated before the cutoff (court deadlines, CPI/FOMC calendars, earnings dates, tournament dates) and 8 were false alarms (dates from 2012, 2013, 2023 or 2025 in older articles) |
| (c) model auditor | flagged 1 line, in question 45000 |
| (d) manual review | confirmed post-cut evidence in **1 of 20** (question 45000) |

What was found in question 45000 (the question asked whether a Morawiecki-aligned club would be registered
in the Sejm before 08-07; it resolved yes; cutoff 07-27):

- The research included infopolityka.pl's page for the 63rd Sejm sitting.
- Every date field on the page says 2026-07-21: published, modified, og:updated_time and ld:dateModified.
- But the crawled text reports the 29–31 July sitting as already held, with 7 votes.
- So the page was updated after the cutoff without changing its date metadata. This is a class the date
  verification cannot catch.
- It gave little away: B gave the outcome 0.55.

The question is dropped from all analyses. One confirmed question is below the Problem bar (2 or more).

- **Residual risk:** 1 of 20 is about 5% (95% CI 0.1% to 25%), so perhaps about 17 of the 347 carry some
  post-cutoff text.
- This matters for absolute scores (B against the crowd and uniform). It matters much less for C − B, because
  both arms use the same bundle.
- A cheap follow-up fix: drop pages whose URL is an index or "sitting" (live) page, or re-check the text
  against the crawl date.

## Power: what the pre-registration assumed and what happened

| type | pre-registered SD of C − B (30-question pilot) | observed SD (n) |
|---|---|---|
| binary | 7.2 | 6.8 (178) |
| numeric | 10.1 | 19.5 (76) |
| discrete | 3.7 | 24.3 (80) |
| MC | 1.3 | 2.8 (13) |
| all types | 7.2 | 15.6 (347) |

- The overall SD was 2.2 times the planned value. The achieved MDE at 80% power is 2.3 points, not the
  planned 1.1, and the CI half-width is ±1.6.
- The extra noise comes from a few continuous questions where C's threshold-anchor blend moved the CDF a lot:
  the largest single differences were +157, −127, −102, +79 and +58 points.
- The binary estimate held: its CI half-width is ±1.0, and it gives the tightest conclusion. C is not better
  on binary questions, and a gain above +0.6 points is excluded.

## Cost and time

| item | cost |
|---|---|
| forecast phase, 318 questions (Opus at list price via Prime + Exa; about $0.32 per question) | $100.70 |
| leak-audit model calls | $0.99 |
| **total** | **$101.69 of the $120 cap** |

Both rows are logged in BUDGET-24H-20260929.md. The run took 05:10Z to 06:24Z.

## Files

All paths are in backtest/bt-scale/:

- `PREREG.md`
- `power.json`
- `report.json` (every scored row, from `run.py backtest score --run bt-scale --base bt-clean bt-clean-8`)
- `analysis.json` (every number above, from tools/analyze.py)
- `chart.json`
- `leak_audit/flags.json` (steps a to c) and `leak_audit/audit.json` (manual classifications)
- `ledger/`, `bundles/`, `runs/`, `cost.jsonl`, `forecast.log`

## Corrected (2026-09-30 06:40Z)

The first version of this file used the pipeline scorer before the bucket-edge fix. Its numbers were:

- primary C − B −0.27 [−1.89, +1.45]
- numeric +0.59
- round 33067 −2.66
- clustered [−2.31, +1.86]

The verdict is unchanged: No gain, inconclusive.

## After the continuous-blend fix (exploratory, 2026-09-30 09:35Z)

Not pre-registered. The primary and the verdict above stand.

The gpt-6-luna follow-up (`../bt-scale-luna/RESULTS.md`, "Why C loses") found a defect in C's continuous blend
(`consistency.numeric_blend`). Where the threshold answers disagreed with the quantile CDF, the blend emptied
buckets that B gave real mass. That hit 57 of Opus's 156 numeric and discrete questions. The blend now rescales
B's CDF between thresholds instead, and a regression test covers it. C′ was rebuilt from the saved `arms.json`
with no model calls (`../bt-scale-luna/tools/rescore_blendfix.py`, `../bt-scale-luna/blendfix.json`). The old
blend rebuilds every saved C exactly, and the script reproduces the primary above.

| slice | n | C − B (pre-registered) | C′ − B (fixed blend) |
|---|---|---|---|
| all | 347 | −0.30 [−1.92, +1.41] | +0.69 [−0.45, +2.14] |
| numeric | 76 | +0.46 [−4.06, +4.84] | +1.76 [−0.82, +5.16] |
| discrete | 80 | −0.69 [−5.88, +4.97] | +2.37 [−0.87, +7.30] |
| binary, MC | 191 | unchanged | unchanged |

- Questions where C empties a bucket: 57 → 0 of 156.
- Log loss C′ − B on numeric + discrete: −0.04 [−0.10, +0.00]. 80% coverage of C′: 0.80.
- Mean C′ is +32.85 (B +32.16).

The fix removes a loss but shows no significant gain, and this comparison was chosen after the fact. **B stays
the submit arm.** Live forecasts now compute arms C and D with the fixed blend and still submit B.

## After the tied-quantile CDF fix (exploratory, 2026-10-02)

Not pre-registered. The primary and the verdict above stand.

When the model was near-sure of one value, several of its quantiles tied (Q45981: 1% to 90% all = 3). The old
`cdf.cdf_from_knots` merged the tied knots into one at the top level and ran the linear lower tail from there,
so B said the opposite of the model (P(X ≤ 2) = 0.64). `cdf._spread_ties` now spreads a tied run over the
value's bucket (wider on numeric questions, to respect the PMF cap). B and C were rebuilt from the saved quantile
samples and threshold answers, with no model calls (`tools/rescore_tiefix.py`, `tiefix.json`). The pre-fix
code rebuilds every saved B exactly, and the "before" C reproduces C′ above.

| model | B changed | B′ − B, all (n 347) | numeric (76) | discrete (80) | C − B before → after |
|---|---|---|---|---|---|
| Opus | 45 of 156 | +0.11 [−0.38, +0.62] | −0.00 | +0.47 [−1.62, +2.67] | +0.69 → +0.62 |
| gpt-6-luna | 47 of 156 | +0.31 [−0.07, +0.69] | −0.09 | +1.44 [−0.17, +3.07] | +0.82 → +0.78 |

- B's CDF disagreed with the model's own quantiles by more than 0.05 on 33 of 329 logged numeric and discrete
  forecasts (all runs); after the fix, on 0.
- Mean B (Opus) +32.16 → +32.26; B − U +31.32 → +31.43; 80% coverage 0.80 → 0.80.
- The score effect is small because a confident model is sometimes confidently wrong, and the old spread-out
  mass then helped by accident. The fix is about submitting what the model said.
- Two other ways to place the tie were tried and rejected. Putting the tie's lowest level exactly on the bucket
  edge gave Opus −0.08 (it gives the tied value only its minimum mass). Putting knots on every discrete bucket
  edge gave −9.2, because it emptied the buckets between quantile values. The shipped version uses the midpoint
  where a tie meets a neighbouring value. That choice was made after seeing these numbers.

## Chart data (chart.json)

C − B with 95% CI, in baseline points:

```json
{
 "title": "MiniBench backtest: consistency arm C minus plain arm B (Metaculus baseline points)",
 "note": "mean paired difference with 95% bootstrap CI; >0 = C better",
 "series": [
  {
   "label": "all (primary)",
   "mean": -0.3,
   "lo": -1.92,
   "hi": 1.41,
   "n": 347
  },
  {
   "label": "type binary",
   "mean": -0.44,
   "lo": -1.43,
   "hi": 0.61,
   "n": 178
  },
  {
   "label": "type numeric",
   "mean": 0.46,
   "lo": -4.06,
   "hi": 4.84,
   "n": 76
  },
  {
   "label": "type discrete",
   "mean": -0.69,
   "lo": -5.88,
   "hi": 4.97,
   "n": 80
  },
  {
   "label": "type multiple_choice",
   "mean": -0.37,
   "lo": -2.08,
   "hi": 0.8,
   "n": 13
  },
  {
   "label": "round 33064",
   "mean": 1.49,
   "lo": -1.22,
   "hi": 4.75,
   "n": 58
  },
  {
   "label": "round 33067",
   "mean": -2.83,
   "lo": -8.07,
   "hi": 0.72,
   "n": 59
  },
  {
   "label": "round 33069",
   "mean": 0.8,
   "lo": -5.08,
   "hi": 7.99,
   "n": 58
  },
  {
   "label": "round 33074",
   "mean": -1.36,
   "lo": -3.72,
   "hi": 1.0,
   "n": 55
  },
  {
   "label": "round 33075",
   "mean": 1.27,
   "lo": -2.04,
   "hi": 5.15,
   "n": 59
  },
  {
   "label": "round 33122",
   "mean": -1.21,
   "lo": -3.18,
   "hi": 0.67,
   "n": 58
  }
 ],
 "context": {
  "B_minus_uniform": {
   "mean": 31.32,
   "lo": 24.56,
   "hi": 37.71,
   "n": 347
  },
  "B_vs_crowd": {
   "mean": 4.18,
   "lo": 0.93,
   "hi": 7.42,
   "n": 191
  }
 }
}
```

## Verification

An independent verifier (a separate agent that wrote its own scoring and bootstrap code and did not use
score.py, backtest.py or analyze.py) recomputed the headline numbers from the raw ledgers, snapshots, truth
and cost files, at about 06:40Z.

**Confirmed:**

- n = 348 scored and 347 primary, with no duplicates, no ids outside the manifest and 0 skips.
- B − U = +31.3 [+24.6, +37.7] and a mean B of +32.2.
- By type: binary −0.44, discrete −0.69 and MC −0.37.
- On binary questions: PN − B = −0.45, C − PN = +0.01, and C = PN on 88.8% of them.
- Against the crowd: B +4.18 (n 191) and C − B −0.31.
- Four of the six rounds (33064, 33069, 33074, 33075).
- Every forecast was made on claude-opus-5-5, and all three hash chains are intact.
- Cost: $100.70 for the forecasts (LLM $91.80 + Exa $8.90) and $0.985 for the audit.
- The audit sample reproduces from its seed. Truth resolutions are the ones used. The 30 reused files are
  equal to bt-scale's, and every as-of equals the question's open time.

**One discrepancy, now fixed: a scorer edge case.**

- Two numeric questions resolved exactly on a CDF grid edge: 44712 (3.64 on the range 3–4.6) and 45321
  (2.97 on the range 1.5–4.5).
- `cdf.bucket_index` searched a float grid, so rounding put those outcomes one bucket up, against its own rule
  that an edge value goes in the lower bucket.
- Fixed in `bot/mb/cdf.py`: the index is now computed in cdf location with a 1e-9 tolerance. A regression test,
  `bot/tests/test_cdf_bucket.py`, fails on the old code, and all 133 bot tests pass.
- Everything was rescored from the saved forecasts, with no new model calls. Only those two questions changed.
- The corrected numbers above equal the verifier's independent values: primary −0.30 [−1.92, +1.41],
  numeric +0.46 and round 33067 −2.83.

Minor: PREREG.md says "~05:15Z", but it was committed at 05:09:47Z (7abd43d). The first forecast started at
05:10:21Z, so the pre-registration came first, and it has not changed since.
