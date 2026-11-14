# PREREG, S0 worker 2: stream changes (written 2026-09-27 before any run)

Scope: `substrate/streams_ext.py` = seeded PRNG-parity replacing pi (no shared offsets), a held-out
periodic family released mid-run, and a regime shift (every `regime_every` stream steps a random
quarter of the non-held-out questions switch to a different base family).

Arms (stream audits, no neural net; all CPU, numpy only):
- A0 `wealth-net` streams (`../wealth-net/streams.py`, family 5 = pi-parity), T=24000, Q=64.
- A1 `v2-static`: streams_ext with regime shift off, held-out released at T/2, T=60000, Q=64+8.
- A2 `v2-shift`: A1 plus regime shift every 10000 steps, fraction 0.25.

Metrics (per seed, then mean ± sd over seeds 1, 2, 3):
- M1 shift-copy fraction: share of questions in the pi (A0) / prng (A1) family whose sequence equals
  another question of the same family up to a time shift (exact match over >= 2000 overlapping steps).
- M2 max |lag-0 Pearson corr| over pairs within the pi/prng family.
- M3 prequential BCE (nats/bit) of an online per-question logistic learner on the last W=32 outcomes
  (test then train, SGD lr 0.05), per family; for the held-out family measured after release only.
- M4 regime shift (A2): mean prequential BCE of switched questions over the 1000 steps after each
  boundary minus the 1000 steps before; the realised switched fraction per boundary.

Predictions / pass criteria: M1 pi > 0.5 and prng = 0; M2 prng < 0.03; M3 prng within 0.01 of ln 2
= 0.6931; held-out BCE after release lower than prng by > 0.3 (learnable); M4 realised fraction
= 0.25 exactly (rounded to questions) and switched-question BCE rises after boundaries (> 0).
Stop rule: one audit pass per seed per arm; no reruns with changed parameters unless a test fails,
in which case the fix and the rerun are logged in STATUS.md.
