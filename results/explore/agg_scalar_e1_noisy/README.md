# EXPLORATORY: Scalar-Report Clearing on `E1` (`visibility=noisy`)

Serves: auditable belief clearing where confiability (theta_trust in params_real) and Key-Gram de-duplication recover oracle_clearing from scalar reports across overlapping, Sybil, colluding, and biased sources.

**Status:** `● Gain` · **Track:** `EXPLORATORY` (`splits`: `train` $n=2000$, `dev` $n=2000$, `test` untouched; `harness.git_dirty = False`)

## Why
We tested why standard token attention (`naive_mha`) fails on `E1` (`noisy` keys) and whether restoring coordinate/piece permutation equivariance (`key_gram_clear`) plus a cross-agent Confiability Head $\beta_i \in \text{params}_{\text{real}}$ (`key_gram_confiability_clear`) trained strictly on `tr.truth` recovers the reports-level `oracle_clearing` ceiling.

## What we did
On `E1` (`visibility=noisy`), we trained `naive_mha`, `key_gram_clear`, and `key_gram_confiability_clear` on $n=2000$ `train` episodes using `tr.truth` labels only (never touching `tr.bayes` or `tr.oracle`), and evaluated all learned arms alongside `untrained_init` and `random_network` controls (`run_with_controls`) and non-learned pooling baselines on $n=2000$ `dev` episodes.

## Result
**Result in one line:** `key_gram_confiability_clear:trained` achieved **0.5928 [0.5458, 0.6413]** regret to Bayes (vs. `0.8293 [0.7716, 0.8859]` for `extremized_log_pool`, `0.8209 [0.7635, 0.8768]` for `key_gram_clear:trained`, `2.6188 [2.5036, 2.7198]` for `naive_mha:trained`, and `-0.0000 [-0.0000, 0.0000]` for `oracle_clearing`), with paired regret difference **-0.2364 [-0.2715, -0.2004]** against `extremized_log_pool` and **-0.2280 [-0.2617, -0.1937]** against `key_gram_clear:trained`.

## Dev Split Metrics ($n=2000$ episodes, 95% bootstrap CIs)

| Method | Regret to `bayes` (nats) | KL to `bayes` | Log loss | Brier |
|---|---:|---:|---:|---:|
| `bayes` | 0.0000 [0.0000, 0.0000] | -0.0000 [-0.0000, -0.0000] | 1.1930 [1.1227, 1.2707] | 0.7208 [0.6867, 0.7544] |
| `oracle_clearing` | -0.0000 [-0.0000, 0.0000] | -0.0000 [-0.0000, -0.0000] | 1.1930 [1.1227, 1.2707] | 0.7208 [0.6867, 0.7544] |
| `linear_pool` | 3.3977 [3.3239, 3.4667] | 3.4140 [3.3685, 3.4566] | 4.5906 [4.5496, 4.6310] | 2.2794 [2.2040, 2.3513] |
| `log_pool` | 2.0475 [1.9883, 2.1020] | 2.0609 [2.0326, 2.0902] | 3.2404 [3.1916, 3.2853] | 1.4143 [1.3588, 1.4652] |
| `sum_logodds` | 0.8273 [0.7713, 0.8826] | 0.8289 [0.7993, 0.8605] | 2.0202 [1.9516, 2.1028] | 0.9464 [0.9067, 0.9866] |
| `extremized_log_pool` | 0.8293 [0.7716, 0.8859] | 0.8304 [0.8000, 0.8625] | 2.0222 [1.9519, 2.1060] | 0.9470 [0.9072, 0.9875] |
| `naive_mha:trained` | 2.6188 [2.5036, 2.7198] | 2.6215 [2.5418, 2.7053] | 3.8118 [3.7081, 3.9134] | 1.4918 [1.4374, 1.5456] |
| `naive_mha:control:untrained_init` | 4.5350 [4.4459, 4.6204] | 4.5511 [4.4841, 4.6151] | 5.7280 [5.6655, 5.7924] | 4.0815 [3.9596, 4.2044] |
| `naive_mha:control:random_network` | 4.7194 [4.6102, 4.8197] | 4.7437 [4.6590, 4.8249] | 5.9123 [5.8299, 5.9957] | 4.2933 [4.1597, 4.4272] |
| `key_gram_clear:trained` | 0.8209 [0.7635, 0.8768] | 0.8228 [0.7949, 0.8544] | 2.0138 [1.9442, 2.0955] | 0.9457 [0.9064, 0.9866] |
| `key_gram_clear:control:untrained_init` | 80.0304 [75.2788, 84.7260] | 80.0167 [75.5076, 84.5300] | 81.2233 [76.5194, 85.9419] | 4.2688 [4.1211, 4.4126] |
| `key_gram_clear:control:random_network` | 965.2805 [830.7437, 1110.8364] | 965.1856 [830.9388, 1110.5043] | 966.4735 [831.9335, 1112.0321] | 669.1696 [578.2592, 779.8910] |
| `key_gram_confiability_clear:trained` | 0.5928 [0.5458, 0.6413] | 0.5832 [0.5606, 0.6073] | 1.7858 [1.7131, 1.8632] | 0.8626 [0.8239, 0.9024] |
| `key_gram_confiability_clear:control:untrained_init` | 53.4389 [50.4191, 56.5737] | 53.6463 [50.7415, 56.7328] | 54.6318 [51.5842, 57.7595] | 3.9083 [3.7936, 4.0254] |
| `key_gram_confiability_clear:control:random_network` | 2.7052 [2.5300, 2.8958] | 2.7169 [2.5623, 2.8755] | 3.8981 [3.7199, 4.0967] | 1.1047 [1.0627, 1.1457] |

## Paired Regret Differences (95% Bootstrap CIs)

| Comparison ($A - B$, negative favors $A$) | Mean $\Delta$ Regret [95% CI] |
|---|---:|
| `key_gram_confiability_clear:trained` $-$ `extremized_log_pool` | `-0.2364 [-0.2715, -0.2004]` |
| `key_gram_confiability_clear:trained` $-$ `oracle_clearing` | `+0.5928 [+0.5458, +0.6413]` |
| `key_gram_confiability_clear:trained` $-$ `key_gram_clear:trained` | `-0.2280 [-0.2617, -0.1937]` |
| `key_gram_clear:trained` $-$ `extremized_log_pool` | `-0.0084 [-0.0142, -0.0022]` |
| `key_gram_clear:trained` $-$ `oracle_clearing` | `+0.8209 [+0.7635, +0.8768]` |
| `naive_mha:trained` $-$ `extremized_log_pool` | `+1.7896 [+1.6876, +1.8810]` |
