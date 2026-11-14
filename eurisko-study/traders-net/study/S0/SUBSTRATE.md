# SUBSTRATE (S0 deliverable): streams, prequential loop, LI families, metric panel, run log

S0 lead merge, 2026-09-27 23:10Z. Parts by four workers: `SUBSTRATE-w1.md` .. `SUBSTRATE-w4.md`
(details), pre-registrations `PREREG-w1.md` .. `PREREG-w4.md` (index `PREREG.md`).
All paths below are relative to `traders-net/study/S0/`.

**Check (re-runs itself):**
`PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib python3 check_substrate.py`
runs the 39 tests and re-reads every number quoted in this file from its source CSV/JSON.
Checked at 2026-09-27 23:10Z: `pytest: 39 passed`, `PASS (0 problems)`.

## 1. What to import (every later workstream)

Add `study/S0/substrate` to `sys.path`; torch/numpy/scipy from `wealth-net/.pylib`;
**call `torch.set_num_threads(1)`** (the default spawns ~23 threads and thrashes this box).

| need | use | notes |
|---|---|---|
| base streams | `streams_ext.make_streams(Q, T, seed, heldout=8, release=T//2, regime_every=10000, regime_frac=0.25)` then `streams_ext.to_array_stream(s)` | families `periodic, lfsr16, ca30, ca110, prime, prng, and, or, periodic_ho` (n_fam 9). `prng` replaces wealth-net's pi. Set `regime_every` > T for a static stream. |
| wealth-net streams (comparison only) | `streams.py`, `model.py` | verbatim copies of `../wealth-net/`; do not edit. `streams_ext.make_questions` is a drop-in with prng for pi. |
| LI families | `li_families.make_li_stream(seed, T)` then `.as_positions()` | D-delay (D 0/64/512, twins and solos), D-prove (A∨¬A, A∧¬A, (A∧B)→A, exclusive-exhaustive triples, never sampled, revealed late), D-expensive (CA110/LFSR bit k∈{64,256,1024} ahead). `DelayBook` books at trade price, settles at t+delay; `log_score_gain` = LMSR settlement vs a prior. |
| training loop | `prequential.run_prequential(model, stream, opt, B, baseline=, on_step=)` | test-then-train single pass; asserts each label is scored before it is trained on, and trained once. Delayed labels train at reveal. |
| stress mode | `prequential.run_multiepoch(..., epochs, B, frac=0.8)` | `assert_disjoint` enforces train targets precede eval targets; `overlap_audit` checks content-level leaks. |
| metrics | `panel.panel_row(...)`, `panel.panel_from_prequential(res, fam, names, stream)` | BCE/Brier/ECE per family; AND/OR/NOT/EQUAL/TAUT/CONTRA/EXHAUSTIVE violations counted only while unresolved (`constraints_from_li` for LI streams); `ec_slopes` (summarises M5's E_C); `gini` + `gini_null` (shuffled-label z); `dead_bankrupt`; `FlopCounter`/`net_forward_flops`; `conservation_residual`. |
| logging for V1 | `runlog.RunLog(out_dir, run_id, meta)` | one long CSV per run `run_id,step,wall_s,metric,group,value` + `.meta.json` with `code_hash`; `PrequentialGuard` for loops not using `run_prequential`. Example: `w4/demo/demo-s{0,1,2}.csv` (schema only, not a result). |

Plan coverage (STUDY-PLAN S0): all bullets are built. Wealth Gini, dead/bankrupt fraction and the
conservation residual take arrays from books; S0 computes no wealth (M1's `books_v2.py` does).

Tests (`substrate/tests/`, 39): seeded determinism (`test_prequential`, `test_streams_ext`,
`test_li_families`, `test_integration`); train/eval never overlap
(`test_multiepoch_disjoint_and_raises`, `test_predict_before_train_once`, `test_no_leak`,
`test_prequential_no_overlap_and_determinism`, `test_prequential_delay_no_overlap`,
`test_guard_single_pass`); known panel values on toys (`test_panel.py`, exact to 1e-12).

## 2. Validation results (3 seeds, mean ± sd, sample sd)

**Prequential vs multi-epoch** (`w1/SUMMARY-w1.csv`; wealth-net streams, Q=64, n_pos=64000,
BCE nats on positions 51200-64000):
- pi: prequential P4 0.6933 ± 0.0004, P16 0.6957 ± 0.0004 (≈ ln 2); multi-epoch (15 epochs)
  held-out 0.830 ± 0.021, in-sample 0.562 ± 0.022, gap 0.267 ± 0.043. Multi-epoch training
  memorises pi; this is what in-sample books pay for.
- CA30: no memorisation gap (0.004 ± 0.004); P4 0.6945 ± 0.0003. Prereg expected a gap > 0.05:
  not met.
- LFSR16 is bimodal under prequential training: P4 0.46 ± 0.40 (learnt in 1/3 seeds), P16
  0.10 ± 0.17 (2/3; per-seed values in `w1/results/P*-s*.json`). Report learnt counts, not
  mean ± sd alone.
- Leak: at T ≈ 64k, held-out LFSR16 and prime contexts appear in the train region
  (overlap49 1.00 ± 0.00 each). Keep T < 65535 or treat those held-out numbers as in-sample.
- The per-question logistic baseline (Adam 1e-2) sits above ln 2 on noise: pi P4 0.714 ± 0.0001.

**Stream changes** (`results/w2/audit_family_bce.csv`, `results/w2/audit-s{1,2,3}.json`):
- Shifted-copy share: wealth-net pi 1.0 ± 0.0 (every pi question is a shifted copy of another);
  prng 0.0 ± 0.0; prng max lag-0 |corr| 0.0096 ± 0.0030.
- Held-out periodic family: BCE 0.0960 ± 0.0003 in the first 1000 steps after release (learnable).
- Regime shift: exactly 0.25 of questions switch at every boundary (`A2_switch_frac` in the
  audit JSONs); switched-question BCE rises by 0.193 ± 0.057 (1000 steps after minus before,
  online logistic).
- Prereg M3 failed: prng BCE for the online SGD logistic learner is 0.8299 ± 0.0004, not ≈ ln 2
  (learner noise; pi is the same). The fix that matters is no shared offsets, not the BCE.

**LI families** (`w3/baseline-summary.csv`, T=20000, prequential B=4; net d=32 L=1):
- Net BCE: delay0 0.3765 ± 0.006, delay64 0.327 ± 0.003, delay512 0.442 ± 0.013. Prereg
  "delay0 < delay64" not met: family BCE confounds delay with question mix; compare a twin with its
  own base.
- Coherence before resolution (mean abs violation; 0.5 = uninformed): net A∨¬A 0.480 ± 0.184,
  A∧¬A 0.535 ± 0.067, (A∧B)→A 0.554 ± 0.083, triples 0.388 ± 0.148; logistic 0.500 ± 0.000 on all.
  Over all live steps the triples stay at 0.132 ± 0.023 for the net. This is the floor for M5/M6:
  "coherent before resolution" must push the `pre:` numbers well below 0.5.

**Panel** (`w4/null_calib.json`): Gini shuffled-label null z mean -0.19 ± 0.19 (calibrated);
positive control (5/50 traders with a 60% edge) z mean 2.03 ± 0.18, below the prereg's z > 3.
The Gini test has low power for an edge held by a few traders: report z and the null sd, and do
not read a low z as "no skill".

## 3. Fixes and inconsistencies resolved at merge

1. **Window padding unified to 0.5.** `prequential.ArrayStream` padded unknown pre-time-0 window
   entries with 0, while `streams_ext` and `li_families` used 0.5. Now 0.5 everywhere
   (`substrate/prequential.py`). No reported number changes: w1 runs use delay 0 (no padding
   read), w3 uses its own adapter.
2. **Held-out leak in the V1 demo.** `w4/demo_runlog.py` built a raw `ArrayStream` over a
   `streams_ext` stream, so held-out questions were visible as window context before release.
   It now uses `streams_ext.to_array_stream`; `w4/demo/` was regenerated (schema unchanged).
   **Rule for everyone: never wrap `make_streams` output in a raw `ArrayStream`.**
3. Test count: status lines said 38; the suite is 39 (worker 1's last test landed after worker 4's
   count).
4. Violation tolerance defaults differ: `panel.violations` tol 0.01, `li_families.violations`
   tol 0.05. Means agree to 1e-9; the `frac>tol` columns do not. Pass `tol` explicitly.
5. Worker 3's "net after history" column is the mean over all live steps (`viol:<kind>:mean`),
   which includes pre-resolution steps; the values in section 2 name the column they come from.

## 4. Caveats for later workstreams
- D-expensive is not yet compute-graded for the net (exp64/256/1024 BCE 0.520/0.537/0.533 in
  `w3/baseline-summary.csv`, flat in k); CA110 is biased and partly periodic. Use the LFSR rows
  or report CA110 and LFSR separately.
- Regime-shift "steps" are stream time steps; convert if a training step is not one stream step.
- Compare arms at matched steps × batch; ME's extra compute helps the AND/OR compounds.
- Bulky trade-time logits for w1 live at `/tmp/traders-net/S0/w1/P*-s*.npz` (root fs, not synced).
- ECE uses fixed bins (biased upward at small n); compare arms at equal n.
