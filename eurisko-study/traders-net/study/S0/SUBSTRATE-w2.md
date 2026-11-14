# SUBSTRATE, worker 2 part: PRNG-parity, held-out periodic family, regime shift

Code: `substrate/streams_ext.py`; tests: `substrate/tests/test_streams_ext.py` (6 pass, ~8 s; adapter checks need torch via PYTHONPATH=wealth-net/.pylib:
determinism, no shifted copies among PRNG questions, held-out hidden then released, a quarter
switched per boundary and arms paired, compounds track current operands, drop-in API);
audit: `w2/audit.py`; numbers: `results/w2/audit-s{1,2,3}.json`, `results/w2/audit_family_bce.csv`.
Prereg: `PREREG-w2.md`. Seeds 1, 2, 3; mean ± sd (ddof 1).

## What changed vs wealth-net streams
- Family 5 is `prng`: each question has its own PCG64 stream (SeedSequence.spawn), bit = top bit of
  the raw output. wealth-net's pi questions are windows of one digit string at offsets 0..2000.
- Family 8 `periodic_ho`: 8 extra questions (Q = 64 + 8), periods 17..31 (training periodic uses
  2..16; all < W = 32), hidden before `release` (default T/2): label mask False and window entries
  0.5 for times before release, so no question sees them as context. Never regime-shifted.
- Regime shift: every `regime_every` (default 10000) stream steps, round(0.25 x 64) = 16 random
  non-held-out questions switch to a different base family (0..5) with fresh parameters from that
  step on; unchosen AND/OR questions keep combining their operands' current bits (logged as
  `indirect` events). Independent seed children for base, held-out and regime draws, so arms are
  paired: v2-shift equals v2-static bit-for-bit before the first boundary (asserted in the audit).
- Interface for `prequential.py` (worker 1): `to_array_stream(make_streams(...))` returns an
  `ArrayStream` (position convention, live = active, held-out windows read 0.5 before release,
  n_fam 9); smoke-tested through `run_prequential` (300 steps, Q 72). Raw arrays: `bits`, `fam_t`
  (family per step), `active`, `events`, `spec`.
  `make_questions(Q, T, seed)` is a drop-in for wealth-net's with prng in place of pi.
- "Steps" for the shift are stream time steps; if a run's training step is not one stream step,
  the caller must convert.

## Audit results (T = 24000 for A0, 60000 for A1/A2; online per-question logistic, W 32, lr 0.05)
| check | result | prereg criterion |
|---|---|---|
| M1 shifted-copy share, wealth-net pi | 1.000 ± 0.000 | > 0.5: pass (every pi question is a shifted copy of another) |
| M1 shifted-copy share, prng | 0.000 ± 0.000 | = 0: pass |
| M2 max lag-0 abs corr, prng (8 questions) | 0.0096 ± 0.0030 | < 0.03: pass (pi 0.0165 ± 0.0031) |
| M3 prequential BCE, prng | 0.8299 ± 0.0004 nats | within 0.01 of ln 2: **FAIL** (see below) |
| M3 held-out BCE, first 1000 steps after release | 0.0960 ± 0.0003 | < prng - 0.3: pass |
| M3 held-out BCE, last 10000 steps | 0.0008 ± 0.0000 | learnable |
| M4 switched share per boundary | 0.25 in all 5 boundaries x 3 seeds | = 0.25: pass |
| M4 BCE of switched questions, 1000 after minus 1000 before | +0.193 ± 0.057 | > 0: pass (3 of 15 boundaries negative) |
| whole-run BCE, v2-shift vs v2-static | 0.490 ± 0.026 vs 0.481 ± 0.022 | not prereg'd |

M3 failure reading: the criterion was mis-set. The online SGD logistic learner (lr 0.05) pays a
noise cost on any unpredictable bit: pi 0.8280 ± 0.0001, prng 0.8299 ± 0.0004, lfsr16
0.8314 ± 0.0002 (all above ln 2 = 0.693, the constant-0.5 loss). PRNG and pi are equally
unpredictable to a windowed learner; the difference that matters is M1 (pi lets a net with
cross-question attention copy one question from another; prng does not). Per the stop rule
the audit was not rerun with another learning rate.

Per-family BCE for all arms: `results/w2/audit_family_bce.csv`.

## Caveats
- M4 is small: the logistic learner re-adapts in hundreds of steps; a net's shift response is
  for M3/M8 to measure.
- Compound questions are not re-drawn; which ones change indirectly depends on the seed
  (21 ± 10 indirect events per run).
- The PCG64 raw-output stability across numpy versions is numpy's documented policy for bit
  generators (unverified for this exact version pair; tests pin determinism within numpy 2.5.3).
