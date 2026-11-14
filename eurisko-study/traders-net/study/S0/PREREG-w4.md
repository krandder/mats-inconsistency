# PREREG, S0 worker 4 (panel.py, runlog.py, tests), written 2026-09-27 before any run

Scope: measurement code, not an experiment. The "runs" are validation checks of the panel.

Arms (validation checks):
1. Toy fixtures with hand-computed values: BCE, Brier, ECE, per-family split, AND/OR/D-prove
   violations with a resolution mask, signed-wealth Gini, dead/bankrupt fractions, FLOP
   counter, conservation residual, E_C slope on a synthetic power law.
2. Null calibration of the wealth-Gini test: wealth = sum over examples of (label-driven P&L)
   for random traders whose signals are independent of labels; the shuffled-label z-score
   should be ~N(0,1) (|mean z| < 0.5, sd z in [0.6, 1.5] over 30 replicates per seed).
   Positive control: traders with a real edge on the labels; z > 3 expected.
3. runlog round trip: a CSV written by RunLog reads back to the same values.

Metric: absolute error vs hand values (tolerance 1e-9, 1e-6 for ECE bins); null z mean/sd.
Seeds: 0, 1, 2 (report mean ± sd over seeds for arm 2).
Stop rule: all tests green, or after 3 fix rounds report what is still failing. No GPU.
