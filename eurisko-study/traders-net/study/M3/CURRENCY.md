# M3 Currency and lock-in (2026-09-28, lead merge)

Scope: the 00:25Z RETARGET cut M3 to a short note built on wealth-net's decay results. Two parts:
(A) worker 1, generic MLP neurons, read-only re-aggregation of wealth-net (`analyze.py`, prereg
`PREREG.md`, data `results/`); (B) worker 2, committed traders under `../M1/CURRENCY.lock` on the
S0 regime-shift stream (`w2/lockin.py`, `w2/aggregate.py`, prereg `w2/PREREG.md`, data `w2/results/`,
detail `w2/CURRENCY-w2.md`). Seeds 1-3, mean ± sd (ddof 1). Checks: `python3 check.py` asserts every
number below against its file. Rerun: `python3 analyze.py`;
`cd w2 && PYTHONPATH=../../../../wealth-net/.pylib python3 lockin.py {1,2,3} && python3 aggregate.py`.

**TL;DR.** (1) For generic neurons, no gradient-wealth currency, decayed or not, carries signal
(all within about one null sd of 0); gradient norm beats all of them, and exact output settlement
beats gradient norm. The plan's stop rule fires: gate by gradient magnitude, wealth is accounting only.
(2) For committed traders, cumulative (Kelly) settlement locks in after a regime shift; a fixed-share
floor α = 1e-3 fixes it (post-shift loss 0.047 vs 0.690 nats, lag about 120 vs 9007 steps) at 0.001 nats
stationary cost, and beats a gradient gate by 0.29-0.35 nats. That is Herbster-Warmuth tracking, not new.

## A. Generic neurons (wealth-net checkpoints, 512 MLP neurons per seed)
Oracle: single-neuron zero ablation at the final checkpoint (wealth-net `results/decay_oracle_s*.npy`).
Spearman with the oracle (`results/currency_summary.csv`; shuffled-oracle null sd 0.044):

| currency | Spearman |
|---|---|
| all-time gradient wealth | 0.043 ± 0.058 |
| decayed, half-life 100 / 1k / 10k | -0.017 ± 0.030 / 0.001 ± 0.048 / 0.012 ± 0.056 |
| income, last 100 / 500 / 2000 steps | -0.012 ± 0.044 / -0.019 ± 0.029 / 0.010 ± 0.045 |
| gradient norm, all-time (control) | 0.168 ± 0.070 |
| gradient norm, last 500 steps (proxy for recent \|a·g\|) | 0.127 ± 0.064 |
| exact output settlement, LOO, training data | **0.417 ± 0.034** |
| exact output settlement, LOO, held-out data (upper bound) | 0.576 ± 0.032 |

1. **Decay does not rescue gradient wealth.** The best half-life (10k) is the one closest to no decay.
2. **Stop rule fires (`results/stop_rule.json`):** no decayed/windowed currency beats the gradient-norm
   control by ≥ 0.05 in 3/3 seeds (all-time or last-500 control). Consistent with `../M1/CURRENCY.lock`.
3. **A sign question, not a memory question.** Exact output settlement beats the gradient-norm control
   by +0.21/+0.25/+0.29 per seed (`results/stop_rule.json`). M2 agrees (`../M2/BAKEOFF.md` §2:
   settle_loo_train 0.458 ± 0.043 at 20k, 0.464 ± 0.041 at final; accumulated abs(a·g) 0.313 ± 0.115
   at final, above every gradient-wealth currency, below settlement).
4. **Pruning (`results/prune_final.csv`, held-out dBCE, lowest 1% / 20% removed):** all-time wealth
   +0.245 ± 0.382 / +0.821 ± 0.393; half-life 100 +0.156 ± 0.169 / +1.105 ± 0.233; random
   +0.023 ± 0.014 / +0.760 ± 0.120; gradient norm +0.009 ± 0.004 / +0.392 ± 0.135. At 20% every
   gradient-wealth currency prunes worse than random.
5. **Lock-in signal.** Rank correlation of neuron income across consecutive disjoint 1k-step windows
   falls from 0.564 ± 0.066 (first five) to 0.355 ± 0.110 (last five) (`results/currency_summary.csv`),
   so all-time wealth is increasingly set by early history. At step 20k, pruning the lowest 1% by
   last-100 income or half-life 100 costs +0.69 to +0.75 BCE vs +0.20 for all-time wealth (quoted from
   `../../../wealth-net/RESULTS.md` §Decayed wealth, obs. 3; not re-aggregated here).

## B. Committed traders after a regime shift (Q6 under CURRENCY.lock)
S0 stream `streams_ext.make_streams(Q=64, T=40000, heldout=8, regime_every=10000, regime_frac=0.25)`;
35 committed traders per question (base-rate prior, lag-1..32 copy, LFSR16 parity, online logistic),
price = wealth-weighted linear pool, settlement = log p_i(y) − log p(y). 48 learnable switches
(18/14/16 per seed). Table from `w2/results/summary.csv`; lag = mean of per-seed medians, censored at 10k.

| currency | post-shift loss, 2k steps (nats) | lag (steps) | censored s1/s2/s3 | stationary loss |
|---|---|---|---|---|
| kelly (cumulative) | 0.690 ± 0.050 | 9007 | 11/7/9 | 0.0363 |
| EMA half-life 100 / 1k / 10k | 0.074 ± 0.003 / 0.334 ± 0.012 / 0.662 ± 0.035 | 188 / 881 / 5793 | 0/0/0, 0/0/0, 2/2/0 | 0.0375 / 0.0363 / 0.0363 |
| fixed share α 1e-3 | **0.047 ± 0.002** | **120** | 0/0/0 | 0.0373 |
| fixed share α 1e-2 | 0.054 ± 0.001 | 118 | 0/0/0 | 0.0461 |
| disjoint 500-step window | 0.225 ± 0.014 | 592 | 0/0/0 | 0.0363 |
| gradient gate lr 0.1 / 1 | 0.397 ± 0.035 / 0.336 ± 0.014 | 1019 / 597 | 0/0/0 | 0.0371 / 0.0443 |
| uniform (control) | 0.556 ± 0.013 | 10000 | 18/14/16 | 0.5814 |
| shuffled-label kelly (control) | 0.713 ± 0.073 | 10000 | 17/14/16 | 0.5975 |

6. **Decision (w2 prereg rule, `w2/results/decision.json`):** six currencies beat kelly by ≥ 0.05 nats
   in 3/3 seeds with stationary cost ≤ 0.01 (ema100, ema1k, fs1e-3, fs1e-2, win500, grad0.1); best is
   fixed share 1e-3 (gain 0.59/0.65/0.69, stationary cost 0.001). grad1 fails the stationary bound.
7. **Kelly lock-in grows with time in the old regime:** median lag 5486 steps after the 10k shift and
   censored (10000) after 20k and 30k; fixed share 119 / 121.5 / 119 at the three boundaries (pooled
   over seeds, `w2/results/switches_s*.csv`). Kelly is worse than the uniform pool after a shift.
8. **Gradient gating recovers faster than kelly (prediction failed)** because its update is bounded, but
   loses to fixed share by 0.29 (lr 1) to 0.35 (lr 0.1) nats on the mean, ≥ 0.27 in every seed. Its lower overall loss (0.364 vs 0.374,
   `summary.csv` overall_mean) is off the primary metric and not a claim.
9. Prior art: Herbster and Warmuth, "Tracking the best expert", Machine Learning 32, 1998,
   https://doi.org/10.1023/A:1007424614876 (unverified). Recommendation for M6/M7/M8 pooled or gated
   trader arms: add a fixed-share floor α ≈ 1e-3.

## Gate B input
The plan's M3 stop rule asks whether a currency beats recent |a·g| by ≥ 0.05 in 3/3 seeds. For generic
neurons (what M8 gates) it does not (A.2): **Gate B fails on M3**, M8 stays on its reduced arm (already so
under the RETARGET). For committed traders fixed share beats both gradient gates by ≥ 0.27 nats in 3/3 seeds (B.8),
but the gradient gate there is a pool-weight analog, not |a·g| on neurons, so it does not open M8's grid.

## Not done (unverified)
- Recent |a·g| is not in wealth-net snapshots; windowed gradient norm is the proxy in part A.
- Part A has no regime shift (no shift run on wealth-net); part B has no generic neurons.
- Part B: designed, mostly non-learning traders; linear pool, not the LMSR fixed point; one stream config.
- Decayed exact settlement not tested (settlement saved only at 20k/final).
- The lead did not rerun `w2/lockin.py` (box load 44 on 16 CPU at 06:27Z); `aggregate.py` and `analyze.py`
  were rerun and reproduce their outputs byte for byte.
