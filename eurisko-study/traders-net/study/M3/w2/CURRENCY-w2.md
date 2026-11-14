# M3-w2: which currency survives a regime shift, for committed traders

Worker 2 of M3, 2026-09-28. This complements worker 1's note on wealth-net's decay results; it does not repeat it.
Under CURRENCY.lock only committed traders get paid, so this tests lock-in for that currency alone. Pre-registered
in `PREREG.md`. Code: `lockin.py` (one run per seed, ~40 s on 1 CPU) and `aggregate.py`. Data: `results/`.
Rerun: `PYTHONPATH=<wealth-net>/.pylib python3 lockin.py {1,2,3} && python3 aggregate.py`.

**Setup.** S0 shift stream (Q=64 plus 8 held-out, T=40000; 16 questions switch family at 10k, 20k and 30k).
Each question has 35 committed traders: the running base-rate prior, lag-k copy traders (k = 1..32, bets of
0.95/0.05), an LFSR16 parity trader and an online logistic learner. Price = the wealth-weighted linear pool.
Settlement = log p_i(y) − log p(y), the exact LMSR/Kelly log-score gain at the trade price.
There were 48 learnable switches (new family periodic or LFSR16) over 3 seeds: 18, 14 and 16
(`results/seed*.json` meta).

**Result (mean ± sd over 3 seeds; `results/summary.csv`):**

| currency | loss in 2k steps after the shift (nats) | median lag (steps) | censored at 10k (s1/s2/s3) | stationary loss |
|---|---|---|---|---|
| kelly (cumulative) | 0.690 ± 0.050 | 9007 | 11/7/9 | 0.0363 |
| EMA, half-life 100 | 0.074 ± 0.003 | 188 | 0/0/0 | 0.0375 |
| EMA, half-life 1k | 0.334 ± 0.012 | 881 | 0/0/0 | 0.0363 |
| EMA, half-life 10k | 0.662 ± 0.035 | 5793 | 2/2/0 | 0.0363 |
| fixed share 1e-3 | **0.047 ± 0.002** | **120** | 0/0/0 | 0.0373 |
| fixed share 1e-2 | 0.054 ± 0.001 | 118 | 0/0/0 | 0.0461 |
| disjoint 500-step window | 0.225 ± 0.014 | 592 | 0/0/0 | 0.0363 |
| gradient gate, lr 0.1 | 0.397 ± 0.035 | 1019 | 0/0/0 | 0.0371 |
| gradient gate, lr 1 | 0.336 ± 0.014 | 597 | 0/0/0 | 0.0443 |
| uniform (control) | 0.556 ± 0.013 | 10000 | 18/14/16 | 0.5814 |
| shuffled-label kelly (control) | 0.713 ± 0.073 | 10000 | 17/14/16 | 0.5975 |

**Decision** (the pre-registered rule; `results/decision.json`): six currencies beat kelly by ≥ 0.05 nats in 3/3 seeds
and keep the stationary cost within 0.01. The best is **fixed share α = 1e-3**: it gains 0.59/0.65/0.69 nats after
the shift and costs 0.001 nats on questions that never switch.

1. **Expected. Cumulative settlement locks in, and the lock-in grows with time in the old regime.** For kelly the
   median lag is 5486 steps after the 10k shift, and more than 10000 after the 20k and 30k shifts
   (`results/switches_s*.csv`, arm kelly). Kelly is worse than the uniform pool after a shift (0.690 vs 0.556),
   because the old winner keeps betting 0.95 on a question that has changed. This is the committed-trader form of
   wealth-net's "wealth is locked in".
2. **Expected. Fixed share fixes it at almost no cost.** It recovers in about 120 steps, whatever the time spent in
   the old regime (median 119, 122 and 119 steps at the three boundaries). EMA with half-life 100 comes close, but a
   decay long enough to be safe on stationary questions (10k) barely helps (0.662). This is Herbster and Warmuth's
   tracking result (Machine Learning 32, 1998, https://doi.org/10.1023/A:1007424614876, unverified), not something new.
3. **Surprising (the prediction failed). Gradient gating recovers faster than cumulative settlement, not slower.** Its
   update lr·w_i·(r_i − 1) is bounded, so the gradient never drives a loser's log-weight to −10⁴ the way cumulative log
   wealth does. It still loses to fixed share by 0.29-0.35 nats. It has the lowest overall loss (lr 1: 0.364 against
   0.374 for fixed share, `summary.csv` overall_mean), but that comes from questions other than the learnable switches.
   It was not the primary metric, so it is not a claim.

**For Q6:** for committed traders, the currency that follows current value is settlement with a floor on wealth (fixed
share), not cumulative wealth and not a decayed average. Every gated or pooled trader arm (M6, M7, M8) should add
α ≈ 1e-3 as a floor. That is plain online learning; the market framing adds nothing beyond it here.

**Limits.** Designed traders on synthetic bits; a linear pool, not the LMSR fixed point; the traders do not learn,
except the logistic one; one stream configuration; 14-18 learnable switches per seed.
