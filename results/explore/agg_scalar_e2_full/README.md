# EXPLORATORY: Scalar-Report Clearing on `E2` (`visibility=full`)

Serves: auditable belief clearing where confiability (theta_trust in params_real) and Key-Gram de-duplication recover oracle_clearing from scalar reports across overlapping, Sybil, colluding, and biased sources.

**Status:** `● Gain` · **Track:** `EXPLORATORY` (`splits`: `train` $n=2000$, `dev` $n=2000$, `test` untouched; `harness.git_dirty = False`)

## Why
We tested why standard token attention (`naive_mha`) fails on `E2` (`full` keys) and whether restoring coordinate/piece permutation equivariance (`key_gram_clear`) plus a cross-agent Confiability Head $\beta_i \in \text{params}_{\text{real}}$ (`key_gram_confiability_clear`) trained strictly on `tr.truth` recovers the reports-level `oracle_clearing` ceiling.

## What we did
On `E2` (`visibility=full`), we trained `naive_mha`, `key_gram_clear`, and `key_gram_confiability_clear` on $n=2000$ `train` episodes using `tr.truth` labels only (never touching `tr.bayes` or `tr.oracle`), and evaluated all learned arms alongside `untrained_init` and `random_network` controls (`run_with_controls`) and non-learned pooling baselines on $n=2000$ `dev` episodes.

## Result
**Result in one line:** `key_gram_confiability_clear:trained` achieved **0.0178 [0.0107, 0.0245]** regret to Bayes (vs. `0.0448 [0.0326, 0.0564]` for `extremized_log_pool`, `0.0178 [0.0107, 0.0245]` for `key_gram_clear:trained`, `0.0563 [0.0419, 0.0706]` for `naive_mha:trained`, and `0.0166 [0.0095, 0.0235]` for `oracle_clearing`), with paired regret difference **-0.0270 [-0.0356, -0.0180]** against `extremized_log_pool` and **-0.0001 [-0.0002, +0.0001]** against `key_gram_clear:trained`.

## Dev Split Metrics ($n=2000$ episodes, 95% bootstrap CIs)

| Method | Regret to `bayes` (nats) | KL to `bayes` | Log loss | Brier |
|---|---:|---:|---:|---:|
| `bayes` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.2601 [0.2400, 0.2822] | 0.0808 [0.0738, 0.0887] |
| `oracle_clearing` | 0.0166 [0.0095, 0.0235] | 0.0158 [0.0142, 0.0175] | 0.2768 [0.2564, 0.2992] | 0.0862 [0.0788, 0.0944] |
| `linear_pool` | 0.1517 [0.1378, 0.1648] | 0.1407 [0.1364, 0.1451] | 0.4119 [0.4010, 0.4239] | 0.1247 [0.1199, 0.1300] |
| `log_pool` | 0.1124 [0.0991, 0.1244] | 0.1026 [0.0992, 0.1060] | 0.3725 [0.3600, 0.3865] | 0.1121 [0.1069, 0.1178] |
| `sum_logodds` | 0.1324 [0.0994, 0.1660] | 0.1419 [0.1308, 0.1524] | 0.3925 [0.3463, 0.4413] | 0.1044 [0.0940, 0.1159] |
| `extremized_log_pool` | 0.0448 [0.0326, 0.0564] | 0.0407 [0.0373, 0.0441] | 0.3050 [0.2826, 0.3278] | 0.0953 [0.0877, 0.1036] |
| `naive_mha:trained` | 0.0563 [0.0419, 0.0706] | 0.0568 [0.0516, 0.0619] | 0.3164 [0.2932, 0.3422] | 0.1003 [0.0920, 0.1093] |
| `naive_mha:control:untrained_init` | 0.4520 [0.4288, 0.4724] | 0.4377 [0.4261, 0.4484] | 0.7122 [0.7108, 0.7134] | 0.2595 [0.2588, 0.2601] |
| `naive_mha:control:random_network` | 0.4540 [0.4292, 0.4755] | 0.4400 [0.4266, 0.4517] | 0.7142 [0.7062, 0.7218] | 0.2603 [0.2563, 0.2640] |
| `key_gram_clear:trained` | 0.0178 [0.0107, 0.0245] | 0.0161 [0.0144, 0.0178] | 0.2780 [0.2582, 0.2993] | 0.0865 [0.0792, 0.0945] |
| `key_gram_clear:control:untrained_init` | 0.2054 [0.1894, 0.2207] | 0.1934 [0.1889, 0.1980] | 0.4656 [0.4579, 0.4746] | 0.1438 [0.1403, 0.1478] |
| `key_gram_clear:control:random_network` | 0.3888 [0.3677, 0.4081] | 0.3748 [0.3656, 0.3838] | 0.6490 [0.6469, 0.6511] | 0.2280 [0.2270, 0.2291] |
| `key_gram_confiability_clear:trained` | 0.0178 [0.0107, 0.0245] | 0.0161 [0.0144, 0.0178] | 0.2779 [0.2581, 0.2993] | 0.0864 [0.0792, 0.0945] |
| `key_gram_confiability_clear:control:untrained_init` | 0.2261 [0.2090, 0.2419] | 0.2147 [0.2095, 0.2199] | 0.4862 [0.4782, 0.4949] | 0.1525 [0.1488, 0.1564] |
| `key_gram_confiability_clear:control:random_network` | 0.1068 [0.0940, 0.1187] | 0.0971 [0.0943, 0.1000] | 0.3670 [0.3554, 0.3802] | 0.1090 [0.1042, 0.1146] |

## Paired Regret Differences (95% Bootstrap CIs)

| Comparison ($A - B$, negative favors $A$) | Mean $\Delta$ Regret [95% CI] |
|---|---:|
| `key_gram_confiability_clear:trained` $-$ `extremized_log_pool` | `-0.0270 [-0.0356, -0.0180]` |
| `key_gram_confiability_clear:trained` $-$ `oracle_clearing` | `+0.0012 [+0.0001, +0.0022]` |
| `key_gram_confiability_clear:trained` $-$ `key_gram_clear:trained` | `-0.0001 [-0.0002, +0.0001]` |
| `key_gram_clear:trained` $-$ `extremized_log_pool` | `-0.0270 [-0.0355, -0.0180]` |
| `key_gram_clear:trained` $-$ `oracle_clearing` | `+0.0012 [+0.0000, +0.0023]` |
| `naive_mha:trained` $-$ `extremized_log_pool` | `+0.0115 [+0.0013, +0.0223]` |
