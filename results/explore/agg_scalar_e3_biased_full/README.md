# EXPLORATORY: Scalar-Report Clearing on `E3-biased` (`visibility=full`)

Serves: auditable belief clearing where confiability (theta_trust in params_real) and Key-Gram de-duplication recover oracle_clearing from scalar reports across overlapping, Sybil, colluding, and biased sources.

**Status:** `● Gain` · **Track:** `EXPLORATORY` (`splits`: `train` $n=2000$, `dev` $n=2000$, `test` untouched; `harness.git_dirty = False`)

## Why
We tested why standard token attention (`naive_mha`) fails on `E3-biased` (`full` keys) and whether restoring coordinate/piece permutation equivariance (`key_gram_clear`) plus a cross-agent Confiability Head $\beta_i \in \text{params}_{\text{real}}$ (`key_gram_confiability_clear`) trained strictly on `tr.truth` recovers the reports-level `oracle_clearing` ceiling.

## What we did
On `E3-biased` (`visibility=full`), we trained `naive_mha`, `key_gram_clear`, and `key_gram_confiability_clear` on $n=2000$ `train` episodes using `tr.truth` labels only (never touching `tr.bayes` or `tr.oracle`), and evaluated all learned arms alongside `untrained_init` and `random_network` controls (`run_with_controls`) and non-learned pooling baselines on $n=2000$ `dev` episodes.

## Result
**Result in one line:** `key_gram_confiability_clear:trained` achieved **0.0309 [0.0213, 0.0409]** regret to Bayes (vs. `0.0763 [0.0602, 0.0915]` for `extremized_log_pool`, `0.0626 [0.0488, 0.0761]` for `key_gram_clear:trained`, `0.0725 [0.0573, 0.0882]` for `naive_mha:trained`, and `0.0166 [0.0095, 0.0235]` for `oracle_clearing`), with paired regret difference **-0.0453 [-0.0552, -0.0355]** against `extremized_log_pool` and **-0.0317 [-0.0385, -0.0249]** against `key_gram_clear:trained`.

## Dev Split Metrics ($n=2000$ episodes, 95% bootstrap CIs)

| Method | Regret to `bayes` (nats) | KL to `bayes` | Log loss | Brier |
|---|---:|---:|---:|---:|
| `bayes` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.2601 [0.2400, 0.2822] | 0.0808 [0.0738, 0.0887] |
| `oracle_clearing` | 0.0166 [0.0095, 0.0235] | 0.0158 [0.0142, 0.0175] | 0.2768 [0.2564, 0.2992] | 0.0862 [0.0788, 0.0944] |
| `linear_pool` | 0.1425 [0.1286, 0.1551] | 0.1319 [0.1273, 0.1363] | 0.4026 [0.3905, 0.4160] | 0.1222 [0.1172, 0.1279] |
| `log_pool` | 0.0913 [0.0773, 0.1044] | 0.0839 [0.0791, 0.0891] | 0.3514 [0.3354, 0.3708] | 0.1088 [0.1022, 0.1163] |
| `sum_logodds` | 0.3410 [0.2779, 0.4046] | 0.3626 [0.3342, 0.3910] | 0.6012 [0.5264, 0.6801] | 0.1222 [0.1101, 0.1353] |
| `extremized_log_pool` | 0.0763 [0.0602, 0.0915] | 0.0718 [0.0657, 0.0780] | 0.3364 [0.3150, 0.3613] | 0.1052 [0.0974, 0.1140] |
| `naive_mha:trained` | 0.0725 [0.0573, 0.0882] | 0.0714 [0.0650, 0.0788] | 0.3326 [0.3075, 0.3598] | 0.1041 [0.0957, 0.1137] |
| `naive_mha:control:untrained_init` | 0.4593 [0.4359, 0.4799] | 0.4449 [0.4331, 0.4559] | 0.7195 [0.7178, 0.7209] | 0.2631 [0.2623, 0.2639] |
| `naive_mha:control:random_network` | 0.4545 [0.4298, 0.4759] | 0.4403 [0.4269, 0.4520] | 0.7147 [0.7073, 0.7217] | 0.2606 [0.2569, 0.2640] |
| `key_gram_clear:trained` | 0.0626 [0.0488, 0.0761] | 0.0587 [0.0534, 0.0640] | 0.3228 [0.3011, 0.3464] | 0.0998 [0.0923, 0.1081] |
| `key_gram_clear:control:untrained_init` | 0.1708 [0.1549, 0.1860] | 0.1594 [0.1536, 0.1647] | 0.4309 [0.4194, 0.4432] | 0.1334 [0.1285, 0.1387] |
| `key_gram_clear:control:random_network` | 0.4257 [0.4037, 0.4457] | 0.4112 [0.4006, 0.4213] | 0.6859 [0.6855, 0.6863] | 0.2464 [0.2462, 0.2466] |
| `key_gram_confiability_clear:trained` | 0.0309 [0.0213, 0.0409] | 0.0297 [0.0269, 0.0326] | 0.2911 [0.2697, 0.3130] | 0.0900 [0.0823, 0.0980] |
| `key_gram_confiability_clear:control:untrained_init` | 0.1943 [0.1776, 0.2096] | 0.1833 [0.1773, 0.1892] | 0.4545 [0.4444, 0.4651] | 0.1407 [0.1363, 0.1455] |
| `key_gram_confiability_clear:control:random_network` | 0.2334 [0.2166, 0.2492] | 0.2209 [0.2155, 0.2269] | 0.4936 [0.4859, 0.5023] | 0.1561 [0.1526, 0.1601] |

## Paired Regret Differences (95% Bootstrap CIs)

| Comparison ($A - B$, negative favors $A$) | Mean $\Delta$ Regret [95% CI] |
|---|---:|
| `key_gram_confiability_clear:trained` $-$ `extremized_log_pool` | `-0.0453 [-0.0552, -0.0355]` |
| `key_gram_confiability_clear:trained` $-$ `oracle_clearing` | `+0.0143 [+0.0076, +0.0213]` |
| `key_gram_confiability_clear:trained` $-$ `key_gram_clear:trained` | `-0.0317 [-0.0385, -0.0249]` |
| `key_gram_clear:trained` $-$ `extremized_log_pool` | `-0.0136 [-0.0214, -0.0059]` |
| `key_gram_clear:trained` $-$ `oracle_clearing` | `+0.0460 [+0.0343, +0.0579]` |
| `naive_mha:trained` $-$ `extremized_log_pool` | `-0.0038 [-0.0169, +0.0092]` |
