# X1 worker 3 PREREG: skeptic + fresh-seed reproduction for M5, M6, M7

Written 2026-09-28 12:04:18Z (file birth time), before any rerun (first job started 12:05:00Z, queue.log). Corrected at 12:45Z from a wrong "~12:15Z". Claim: `../STATUS.md` (worker3: M5, M6, M7).

## A. Static checks (no new runs)
- Every number in the orchestrator's summary lines for M5, M6, M7 is re-read from the file it cites
  (`check_w3.py`, output `results/static_check.json`). Status per number: match (to the quoted precision),
  mismatch, or no-file.
- Each lead check reruns on a clean copy under `/tmp/traders-net/X1w3/clean/`: `M5/exploit.py check`,
  `M6/lead/merge.py`, `M7/merge/merge_m7.py`, `M7/w1/check_stage0.py`; outputs compared byte for byte
  with the committed files.
- Each result is read against its PREREG (worker and lead): the stop rule as written, deviations.

## B. Fresh-seed reproduction (seed 4; the originals used seeds 1-3, M6 w3 used 0-2)
Code is copied into `/tmp/traders-net/X1w3/clean/study/` (same relative layout), unchanged except output
and bulk paths redirected to `/tmp/traders-net/X1w3/`. CPU only, 1 torch thread, nice 10, at most 3
concurrent jobs (fleet cap 12), no GPU.

| id | experiment (headline) | runs (seed 4) | metric | claimed (seeds 1-3) |
|---|---|---|---|---|
| R-M5a | M5 probe on prequential net P4 | S0 w1 `run_w1.py P4 4`, then M5 w4 `probes.py --source preq --ckpt P4 --seed 4` | EC_phi (probe L, last-half slope, nats/q-step) | 0.092 ± 0.0486 |
| R-M5b | same, P16 | `run_w1.py P16 4`, `probes.py ... P16` | EC_phi | 0.118 ± 6.63e-4 |
| R-M5c | calibration leak via price-only probe | from R-M5a/b | phiP | 0.00164 ± 1.91e-4 / 0.00174 ± 7.9e-5 |
| R-M6a | (b)-(a), P regime | M6 w1 `run_w1.py P A 4`, `P B 4` | learn_bce diff | +0.0074 ± 0.0107 |
| R-M6b | (b)-(a), ME; BOOST noise; ES | `run_w1.py ME A 4`, `ME B 4`, `ME BOOST 4` | learn_bce diff; BOOST humility; ES learn diff | +0.0008 ± 0.0012; 0.0017 ± 0.0008; -0.0050 ± 0.0004 |
| R-M6c | cg noise protection vs BOOST (w2 harness) | w2 `pool.py --arm cg --seed 4 --T 60000`, lead `pool_kc.py --arm boost --seed 4 --T 60000` | noise excess cg vs boost (paired sign) | cg 0.0019-0.0023 < boost 0.0033-0.0040 in 3/3 |
| R-M6d | fixed point vs 2-pass | w4 `fp_layer.py base fp 4`, `base pass2 4` | fp_vs_pass2 violation rel (rule: <= -50% to keep) | +0.17%/+0.12%/-0.08% |
| R-M6e | committed heads Cm vs U | w5 `commit.py --arm Cm --seed 4`, `--arm U --seed 4` | rel learn BCE Cm vs U; head_corr_mean | -0.3/-1.3/-0.9%; 0.9934 ± 0.0015 |
| R-M7a | dsfw vs DeepSeek ds3, E=8 | w2 `moe_w2.py --arm dsfw/ds3 --E 8 --seed 4 --T 24000 --B 8` | paired bce_all rel diff | -1.30% (abs diff -0.0051 ± 0.0009) |
| R-M7b | same, E=64 | `--E 64` | paired bce_all rel diff | -1.87% (paired sd 0.0049) |

## Reproduction rule
- **verified**: the fresh-seed value lies within mean ± 2 sd of the claimed 3-seed values (sample sd; for
  paired quantities, the paired-difference sd), AND the qualitative claim (sign / rule outcome) holds on seed 4.
- **refuted**: outside 2 sd and the qualitative claim fails on seed 4, or a static check finds the number wrong.
- **within-2sd, qualitative differs** or **outside 2 sd, qualitative holds**: reported as such (status
  "partly verified" with the reason), not rounded to verified.
- Where sd is tiny (e.g. P16 EC_phi sd 6.6e-4), 2 sd is applied literally, and the relative gap is also reported.
- Numbers with no rerun (checkpoint-only arms; wealth-net checkpoints exist only for seeds 1-3) are
  "verified (static)" at most: the file says what the text says, no fresh seed.

## Stop rule
Wall cap 4 h for B. If the box load (1-min) stays > 60 for 20 min, run at 2 concurrent jobs. Any run over
3x its original wall time is reported as not finished and its row stays unverified.
Bulk outputs stay under `/tmp/traders-net/X1w3/`; only JSON/CSV summaries are copied to `results/`.

## Deviations (appended 2026-09-28T12:12:56Z, before any seed-4 result was read)
- R-M6d: the claimed fp-vs-2-pass number is M6 w4's `fp_vs_pass2_M3_rel`, which summarize.py computes on the
  **li** stream (M3_D_coherence), not the base stream. Added `fp_layer.py li fp 4` and `li pass2 4`
  (originals 356 s / 190 s). The base fp run stays for the Newton-iteration claim (2.05 ± 0.08).
- R-M6b: ES is not a separate run; M6 w1 summarize.py derives ES from the A run's `score_es`. No extra job.
- Seed-4 derived metrics are computed with the workers' own formulas (M5 w4 aggregate.py, M6 lead/merge.py
  w2rec, M6 w4 summarize.py, M6 w5 aggregate.py, M7 merge_m7.py), each checked to reproduce seeds 1-3 first.
- Stop rule for M7 w2: the originals ran on GPU (265 s), so the 3x-wall rule has no CPU baseline; the 4 h
  wall cap for part B applies to them instead. A CPU/GPU numeric difference is expected to be small but
  nonzero; the reproduction rule (mean +- 2 sd) is unchanged.

- 2026-09-28T12:32:11Z deviation (post hoc, disclosed): added rows R-M5d.P4/P16 (E_C log-log exponent b, qualitative b >= 1) AFTER reading the seed-4 M5 probe JSONs. The claim quotes b, so it needed a row; the threshold b >= 1 was chosen after seeing seed 4 (P16 b=1.03), so treat R-M5d as exploratory.
