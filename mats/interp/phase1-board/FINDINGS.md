# Does the Phase 1 cost rule explain ednacob's 3.63e-8? (FINDINGS)

Read-only research note, 2026-11-02, fetches 14:41-15:25Z. Written by a research subagent of opus-mats; saved by main
(the subagent could not write report files). No code from the dataset or any repository was run (files read with
`git show`/`cat`). Raw downloads with per-directory fetch.log: `downloads/` (forum/ lb/ subs/ sublist/ gh/ hf/ rules/
ednacob/); parsers: `scripts/`; ednacob's 83 graded runs since 30 Jul: `ednacob_runs.tsv`.

## Verdict
1. The cost rule does NOT explain #1. Run #327208 is 93% metered: F = 1.283e11 FLOPs per MLP and R = 0.096 s, so
   lambda R = 9.6e9 (7% of charged cost). The final board was graded on the post-3-Aug grader (participant code pinned
   to one physical core), where lambda = 1e11 FLOP/s is at or above what a core delivers. ednacob's own off-meter runs
   on that grader scale as MSE ~ 5.8e-7 / R(seconds): reaching 3.63e-8 that way needs ~16 s of residual time, 5.9x the
   budget. (Main's hypothesis, the wall-time loophole, was viable before 3 Aug only; no final-board row was graded then.)
2. #1's accuracy comes from fitting the 50 public leaderboard MLPs using the per-MLP errors every submission page
   publishes (seeds fixed across submissions). VERIFIED for 5-6 Aug: entries are exact per-MLP copies of whichever
   earlier run scored lower. INFERRED for 10 Aug (#327208): 46 same-cost probe runs in 100 minutes, then a run 6.3x below
   the base on all 50 MLPs and below every earlier run on every MLP, at unchanged compute.
3. No evidence of a genuine algorithmic gain beyond the field's Kerdock-class level: ednacob's unfitted runs score raw
   2.1-2.3e-7, ~6.1x better than plain MC at equal charged cost, matching the documented 5-7x frontier.

## Key numbers
| Quantity | Value | Source |
|---|---|---|
| #1 ednacob #327208 raw MSE, utilisation | 3.633e-8 at 0.508 (adjusted 1.845e-8) | submission page |
| Grader full-budget MC reference, public 50 / holdout 50 | 6.5005e-7 / 8.6967e-7 | submission page; leaderboard JS |
| Implied mean avg_variance, public / holdout | 0.04179 / 0.05590 (N = 64,280.96) | whestbench mc_at_bm |
| HF v1-phase1 full (1000 nets), full-budget MC raw MSE | mean 8.12e-7, median 6.74e-7, range 1.46e-7..6.44e-6 | avg_variance column |
| HF v1-phase1 mini (100 nets) | mean 7.70e-7, range 2.38e-7..3.15e-6 | same |
| #1 vs plain MC | 17.9x below full-budget reference; 35.2x better at equal charged cost | computed |
| Plain-MC samples for 3.63e-8 (public 50) | 1.15e6 = 17.9 budgets | computed |
| One core, whole budget as residual (post 3 Aug) | <= ~5.4e11 FLOPs, ~130k samples: MC >= 3.2e-7 | 1.5-2e11 FLOP/s/core (forum) |
| Same before 3 Aug (16 vCPUs, ~8 cores) | ~0.8-1.0e6 samples: 4-5e-8 at full budget | computed (viable then) |
| ednacob unfitted base runs (same cost as #1) | 2.11e-7 (#324395), 2.29e-7 (#327098); 12 reseeds 2.11-2.55e-7 | submission pages |

## Board (round 1428)
| # | Entry | Adjusted | Raw MSE | Utilisation | Metered share F/C |
|---|---|---|---|---|---|
| 1 | ednacob #327208 | 1.84e-8 | 3.63e-8 | 0.508 | 0.93 |
| 2 | J2W #319049 | 5.39e-8 | 2.78e-7 | 0.205 | 0.95 |
| 3 | dstepanov #324219 | 6.22e-8 | 2.20e-7 | 0.284 | 0.0002 (residual channel) |
| 4 | dpskv5 #328020 | 7.17e-8 | 1.50e-7 | 0.478 | 0.93 |
| 5 | oabuod #327687 | 7.23e-8 | 2.36e-7 | 0.309 | 0.91 |
At equal charged cost #2-#5 are 8.9-11.4x better than plain MC (not tested for public-set fitting); documented
Kerdock-cubature write-ups (kaileh57, Puffi, pranay212) 5.3-7.1x.

## Rules (VERIFIED)
- Organiser 18125: "Residual wall time is converted into FLOPs at a fixed rate, roughly equivalent to one x86 core:
  C_m = flops_used + lambda * residual_wall_time_s, lambda = 1e11 FLOP/s"; whestbench v0.14.0 budget.py
  LAMBDA_FLOPS_PER_SECOND = 1e11. s_m = MSE_final,m x max(0.1, C_m/B_m), B = 2.72e11; 60 s wall cap per MLP.
- R = predict() wall time - flopscope backend - flopscope overhead (telemetry on #327208 MLP 0 reproduces exactly).
- Hardware: 16 vCPUs until 3 Aug; then participant code pinned to one physical core (2 vCPUs), backend 7 cores (18125;
  rules 5.6). CPU model not published. Top-5 rows graded 5-10 Aug on whestbench 0.14.0 / flopscope 0.10.0 (one core).
- Off-meter compute allowed in Phase 1 but priced at lambda and policed (rules 5.2, 5.6; organisers 18132, 18108: int8
  routes to be disqualified); prohibited in Phase 2 (18197). Organisers 18125: "Some participants exploited this
  asymmetry ... to obtain artificially low FLOP counts while effectively running efficient Monte Carlo sampling outside
  the metered path"; "Some submissions were clearly overfitting to the public MLPs, including their IDs and names. This
  is allowed, but the resulting performance is unlikely to generalize to the post-Phase 1 private re-evaluation."

## Accounting for #1's 35x (INFERRED split)
- ~6x shared with honest Kerdock-cubature entries: 2.4-2.9x from the design (MUB-129 vs iid at equal support; Puffi
  ladder, 18157) and ~2x from cheaper points (fast Walsh-Hadamard first layer, antithetic sharing, dropping dead units,
  Strassen/Winograd). Consistent with our measured ~2-3x design gain.
- The remaining ~5.8x is public-set fitting (verified mechanism 6 Aug, inferred 10 Aug). bulzan's truth-aware oracle
  (reweighting 129 basis blocks with the truth) removes ~5.5x after cross-fitting; weights from lawful observables
  remove 1.1%.
- Not ruled out: a genuinely new method written in the 55 minutes after the probe burst at exactly unchanged compute.
- Settles it: the Phase 1 private re-evaluation on fresh MLPs (not public as of 2026-11-02; winners ~15 Nov); expect
  #327208 at ~2-3e-7 raw.

## Not obtained
Grader CPU model; ednacob's code; holdout per-MLP scores; census PDF attachments; ednacob's AIcrowd profile.

## Sources (fetched 2026-11-02 UTC)
Forum topics 18025, 18026, 18099, 18105, 18108, 18118, 18122, 18125, 18132, 18141, 18143, 18157, 18164, 18170, 18179,
18184, 18197, 18199, 18251 (https://discourse.aicrowd.com/t/<id>, 14:41-14:47Z); leaderboard round 1428 (14:48:45Z);
submission pages of the top 5 and 83 ednacob runs (14:49-15:20Z); AIcrowd GraphQL listing (15:11-15:25Z); rules
(current, 15:06:53Z; Wayback 20260818152915); whest-starterkit main 96e308c and 5b7a347; whestbench v0.14.0 and
ef8416b; HF aicrowd/arc-whestbench-public-2026 at v1-phase1 (7c3aa28b), avg_variance columns via range requests
(scripts/hf_avgvar.py, run with python -I).
