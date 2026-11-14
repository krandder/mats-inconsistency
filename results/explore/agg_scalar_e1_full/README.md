# EXPLORATORY: Scalar-Report Clearing on `E1` (`visibility=full`)

Serves: auditable belief clearing where confiability (theta_trust in params_real) and Key-Gram de-duplication recover oracle_clearing from scalar reports across overlapping, Sybil, colluding, and biased sources.

**Status:** `● Gain` · **Track:** `EXPLORATORY` (`splits`: `train` $n=2000$, `dev` $n=2000$, `test` untouched; `harness.git_dirty = False`)

## Why
We tested why standard token attention (`naive_mha`) fails on `E1` (`full` keys) and whether restoring coordinate/piece permutation equivariance (`key_gram_clear`) plus a cross-agent Confiability Head $\beta_i \in \text{params}_{\text{real}}$ (`key_gram_confiability_clear`) trained strictly on `tr.truth` recovers the reports-level `oracle_clearing` ceiling.

## What we did
On `E1` (`visibility=full`), we trained `naive_mha`, `key_gram_clear`, and `key_gram_confiability_clear` on $n=2000$ `train` episodes using `tr.truth` labels only (never touching `tr.bayes` or `tr.oracle`), and evaluated all learned arms alongside `untrained_init` and `random_network` controls (`run_with_controls`) and non-learned pooling baselines on $n=2000$ `dev` episodes.

## Result
**Result in one line:** `key_gram_confiability_clear:trained` achieved **0.0057 [0.0004, 0.0108]** regret to Bayes (vs. `0.8293 [0.7716, 0.8859]` for `extremized_log_pool`, `0.0027 [0.0002, 0.0052]` for `key_gram_clear:trained`, `2.6119 [2.5011, 2.7124]` for `naive_mha:trained`, and `-0.0000 [-0.0000, 0.0000]` for `oracle_clearing`), with paired regret difference **-0.8236 [-0.8801, -0.7671]** against `extremized_log_pool` and **+0.0030 [-0.0025, +0.0082]** against `key_gram_clear:trained`.

## Dev Split Metrics ($n=2000$ episodes, 95% bootstrap CIs)

| Method | Regret to `bayes` (nats) | KL to `bayes` | Log loss | Brier |
|---|---:|---:|---:|---:|
| `bayes` | 0.0000 [0.0000, 0.0000] | -0.0000 [-0.0000, -0.0000] | 1.1930 [1.1227, 1.2707] | 0.7208 [0.6867, 0.7544] |
| `oracle_clearing` | -0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, -0.0000] | 1.1930 [1.1227, 1.2707] | 0.7208 [0.6867, 0.7544] |
| `linear_pool` | 3.3977 [3.3239, 3.4667] | 3.4140 [3.3685, 3.4566] | 4.5906 [4.5496, 4.6310] | 2.2794 [2.2040, 2.3513] |
| `log_pool` | 2.0475 [1.9883, 2.1020] | 2.0609 [2.0326, 2.0902] | 3.2404 [3.1916, 3.2853] | 1.4143 [1.3588, 1.4652] |
| `sum_logodds` | 0.8273 [0.7713, 0.8826] | 0.8289 [0.7993, 0.8605] | 2.0202 [1.9516, 2.1028] | 0.9464 [0.9067, 0.9866] |
| `extremized_log_pool` | 0.8293 [0.7716, 0.8859] | 0.8304 [0.8000, 0.8625] | 2.0222 [1.9519, 2.1060] | 0.9470 [0.9072, 0.9875] |
| `naive_mha:trained` | 2.6119 [2.5011, 2.7124] | 2.6162 [2.5311, 2.7035] | 3.8048 [3.7053, 3.9063] | 1.4911 [1.4375, 1.5447] |
| `naive_mha:control:untrained_init` | 4.5316 [4.4419, 4.6172] | 4.5485 [4.4824, 4.6123] | 5.7246 [5.6629, 5.7887] | 4.0755 [3.9531, 4.1970] |
| `naive_mha:control:random_network` | 4.6997 [4.5904, 4.7981] | 4.7264 [4.6441, 4.8047] | 5.8926 [5.8104, 5.9713] | 4.2754 [4.1394, 4.4082] |
| `key_gram_clear:trained` | 0.0027 [0.0002, 0.0052] | 0.0015 [0.0013, 0.0017] | 1.1956 [1.1253, 1.2740] | 0.7210 [0.6869, 0.7544] |
| `key_gram_clear:control:untrained_init` | 64.0374 [60.6559, 67.4329] | 64.3548 [61.0960, 67.7146] | 65.2303 [61.8695, 68.6103] | 4.0329 [3.9171, 4.1507] |
| `key_gram_clear:control:random_network` | 473.4415 [444.2550, 504.1298] | 473.2391 [443.9382, 503.9520] | 474.6345 [445.5013, 505.2531] | 233.7990 [223.4556, 244.1590] |
| `key_gram_confiability_clear:trained` | 0.0057 [0.0004, 0.0108] | 0.0073 [0.0069, 0.0077] | 1.1986 [1.1276, 1.2777] | 0.7221 [0.6888, 0.7554] |
| `key_gram_confiability_clear:control:untrained_init` | 52.0467 [49.2190, 54.9944] | 52.2652 [49.4601, 55.0707] | 53.2397 [50.4247, 56.1857] | 3.8942 [3.7803, 4.0104] |
| `key_gram_confiability_clear:control:random_network` | 3.3411 [3.1497, 3.5483] | 3.3148 [3.1297, 3.5017] | 4.5341 [4.3531, 4.7315] | 1.3829 [1.3357, 1.4356] |

## Paired Regret Differences (95% Bootstrap CIs)

| Comparison ($A - B$, negative favors $A$) | Mean $\Delta$ Regret [95% CI] |
|---|---:|
| `key_gram_confiability_clear:trained` $-$ `extremized_log_pool` | `-0.8236 [-0.8801, -0.7671]` |
| `key_gram_confiability_clear:trained` $-$ `oracle_clearing` | `+0.0057 [+0.0004, +0.0108]` |
| `key_gram_confiability_clear:trained` $-$ `key_gram_clear:trained` | `+0.0030 [-0.0025, +0.0082]` |
| `key_gram_clear:trained` $-$ `extremized_log_pool` | `-0.8266 [-0.8833, -0.7699]` |
| `key_gram_clear:trained` $-$ `oracle_clearing` | `+0.0027 [+0.0002, +0.0052]` |
| `naive_mha:trained` $-$ `extremized_log_pool` | `+1.7826 [+1.6857, +1.8696]` |
