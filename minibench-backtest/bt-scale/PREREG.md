# bt-scale pre-registration: does the consistency arm (C) beat the plain arm (B) on MiniBench?

Written 2026-09-30 ~05:15Z, before any bt-scale forecast was made. The file is committed to the minibench repo
before the forecast phase starts; any later change to this file is a deviation and is listed in RESULTS.md.

## Question

Does logical consistency help forecasting on real short-horizon questions? Operationally: on resolved MiniBench
questions, does arm C (the consistency projection) score better than arm B (median of 3 plain samples, the
pre-registered would-submit arm)? This decides whether live MiniBench should submit C instead of B. The current
recommendation is B. bt-clean (n = 30) found C − B = +0.2 [−2.0, +2.6] baseline points, which is too wide to
decide anything.

## Pipeline (unchanged)

- Code: minibench repo at commit 04feaa2 (bot/mb/*). There are no prompt, arm, parameter, effort or model changes.
  The only edit is to the harness: `backtest.ROUNDS` also lists 33064 and 33067, so the S1 base-rate history
  covers them. It does not touch B or C.
- Model `claude-opus-5-5` via Prime (tag pool-minibench), pinned with no fallback. A mismatch skips the question.
- Arms: A, B, PN, C, D (= C, identity policy), U, S1/S2, exactly as in forecaster.py. The comparison is C vs B.
- The 30 questions already forecast by this code in bt-clean (22) and bt-clean-8 (8) are reused as they are. Their
  snapshots and truth files were checked byte-equal to the bt-scale prep (ignoring `snapshot_at`). The other
  318 questions are forecast fresh in bt-scale.
- Nothing is submitted. MINIBENCH_LIVE stays unset and no LIVE_APPROVED file is created. Metaculus reads are GET
  only (REST post lists), and Radiant is used only through its read tool `get_metaculus_posts`.

## Sample

- Pool: every resolved, scorable question (binary yes/no, MC, numeric, discrete; annulled and unresolved
  excluded) in the six MiniBench rounds that opened after the model's June 2026 knowledge cutoff:
  33064 (opens 06-29), 33067 (07-13), 33069 (07-27), 33074 (08-10), 33075 (08-24), 33122 (09-07).
  n = 348: 179 binary, 80 discrete, 76 numeric, 13 MC; 55 to 59 per round.
- Rounds before 33064 are excluded because their outcomes (June 2026 and earlier) may be in the model's
  training data. That leak is one the research cutoff cannot remove.
- Order: prep shuffles the pool with seed 20260930 (`run.py backtest prep --seed 20260930`). The 318 new
  questions are forecast in that manifest order. If the budget or time cap stops the run, the scored set is the
  30 reused questions plus a seeded-random prefix of the rest. Nothing is chosen by outcome.

## Leak controls

1. Question view: a whitelisted snapshot as of each question's open time, with no aggregations, resolution,
   actual close/resolve times or forecaster counts. Truth lives in truth/, and the forecast process runs with an
   audit-hook denylist on truth/, Radiant, resolution, aggregation and leaderboard files (guard.py).
2. Research cutoff at the question's open time: Exa `endPublishedDate` = open time − 36 h. The pipeline drops
   results with no date, dated after the cutoff, or carrying a date-only stamp within 36 h of it; results on
   excluded domains (encyclopedias, prediction markets, quote/tracker pages) or matching live-page URL
   patterns; and results whose text names a calendar date after the cutoff. Every kept result also needs an
   independent date (page metadata, JSON-LD, PDF info or URL): published or modified after the cutoff, or no
   date at all, means it is dropped. Resolution sources are never fetched in a backtest.
3. Model knowledge: only rounds that opened after the stated June 2026 cutoff (see Sample). The sensitivity
   analysis drops 33064, the round closest to the cutoff.
4. Scorer re-audit: `backtest.score` re-runs today's strict filter and date check on every stored bundle
   (`leak_suspect`). Rows it flags are left out of the primary set.
5. Leak audit on a random 20 (below).

### Leak audit (random 20)

- Selection: `random.Random(20260931).sample(sorted(scored question ids), 20)`, drawn after the forecast phase.
  It depends only on which questions got scored.
- For each audited question:
  (a) the scorer's strict filter plus date re-check on its bundle (must be empty);
  (b) a regex scan of the exact model-visible research text for any date after the cutoff day, including the
  "Month YYYY" and "Month D" forms that the pipeline filter does not parse;
  (c) an LLM auditor (the same model, via the same Prime tag, costed in the budget) reads the cutoff, the
  question, its actual resolution and the numbered research lines, and lists every line that reports an event
  after the cutoff or reveals the outcome;
  (d) I review every line that (b) or (c) flags by hand, opening the source page where needed, and classify
  it: **confirmed post-cut evidence**, **forward-looking schedule** (known before the cutoff, not a leak) or
  **false alarm**.
- Any audited question with at least one confirmed post-cut line is dropped from every analysis and reported.

## Metrics

- **Primary:** Δ = mean over questions of (baseline score of C − baseline score of B). This is the Metaculus
  baseline score, which is log-score based and comparable across types: binary 100·log2(2p), MC
  100·ln(N·p)/ln N, continuous 50·ln(pmf / uniform pmf), as in score.py. Δ > 0 means C is better.
  The 95% CI is a paired percentile bootstrap over questions: 10,000 resamples, seed 20260930.
- Primary set: all scored questions, minus scorer leak suspects, minus audit-confirmed drops.
- Secondary (reported; they do not change the verdict):
  - Δ by type and by round; a round-clustered bootstrap CI (resample rounds, then questions within rounds);
  - log-loss Δ (all types; binary only); Brier Δ (binary + MC);
  - binary decomposition: PN − B (P/NOT-P averaging alone) and C − PN (the chain projection on top);
  - B and C vs the bot crowd: 100·(ln p(o) − ln cp(o)) against the unweighted bot aggregate at close, for
    binary + MC, as in the dry run. MiniBench questions are open for 3 h, so the crowd saw up to 3 h more
    than our as-of;
  - B and C vs uniform: baseline score (arm U), all types;
  - 80% interval coverage for numeric and discrete;
  - sensitivity: without round 33064; new questions only (without the 30 reused ones).

## Power

Source: power.json (tools/power.py). The per-question SD of C − B comes from the two earlier runs of the same 30
questions (bt-clean and bt-warm: 30 questions × 2 runs, so a rough estimate), reweighted to the pool's type mix.

| metric | SD of C − B | n | 95% CI half-width | MDE at 80% power (α = 0.05, two-sided) |
|---|---|---|---|---|
| baseline points (all types) | 7.2 | 348 | ±0.76 | 1.08 |
| baseline points (all types) | 7.2 | 300 | ±0.82 | 1.17 |
| baseline points (all types) | 7.2 | 250 | ±0.89 | 1.28 |
| baseline points (binary only) | 7.2 | 179 | ±1.05 | 1.51 |
| log loss, nats (all types) | 0.107 | 348 | ±0.011 | 0.016 |
| Brier (binary + MC) | 0.020 | 192 | ±0.0028 | 0.0040 |

For scale: B beat uniform by +32 baseline points in bt-clean. One baseline point on a binary question is 0.69
peer points. If the true SD is 1.5 times the estimate, the MDE at n = 348 becomes about 1.6 points.

## Verdict bars (decided now)

- **Gain:** the lower bound of the primary 95% CI is above 0.
- **No gain:** the CI includes 0 or lies below 0. The sub-case is reported: *C worse* (upper bound < 0),
  *gain of ≥ 1.1 ruled out* (upper bound < +1.1, the MDE), or *inconclusive* (otherwise).
- **Problem** (overrides both) if any of these hold:
  - P1: 2 or more of the 20 audited questions have confirmed post-cut evidence;
  - P2: B does not beat uniform (lower bound of the B − U baseline CI ≤ 0);
  - P3: fewer than 250 questions are in the primary set;
  - P4: any scored forecast used a model other than claude-opus-5-5, a contamination-guard error occurred, or
    the ledger hash chain fails;
  - P5: more than 5% of scored rows are scorer leak suspects.

What each verdict means for the live decision: **Gain** → propose C as the submit arm (Kelvin decides).
**No gain** → keep B. **Problem** → no switch; fix first.

## Budget and stop rules

- Cap: $120 total, LLM at list price plus Exa, logged in
  calibrated-ai/BUDGET-24H-20260929.md. The forecast phase is hard-capped with MB_BUDGET_USD = the bot's
  7-day spend at launch ($22.54) + $112. The audit gets up to $6 and there is a $2 margin. At the cap, stop and
  report.
- Time: finish by 18:30Z. If the forecast phase has not finished by 15:30Z, it is stopped and the questions done
  by then are analysed (the prefix rule above keeps them a random subset).
- A verifier subagent recomputes the headline numbers from the raw files (ledgers, truth, snapshots) before
  anything is reported.
