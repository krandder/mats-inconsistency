# EXPLORATORY: Scalar-Report Clearing on `E3-biased` (`visibility=noisy`)

Serves: auditable belief clearing where confiability (theta_trust in params_real) and Key-Gram de-duplication recover oracle_clearing from scalar reports across overlapping, Sybil, colluding, and biased sources.

**Status:** `● Gain` · **Track:** `EXPLORATORY` (`splits`: `train` $n=2000$, `dev` $n=2000$, `test` untouched; `harness.git_dirty = False`)

## Why
We tested why standard token attention (`naive_mha`) fails on `E3-biased` (`noisy` keys) and whether restoring coordinate/piece permutation equivariance (`key_gram_clear`) plus a cross-agent Confiability Head $\beta_i \in \text{params}_{\text{real}}$ (`key_gram_confiability_clear`) trained strictly on `tr.truth` recovers the reports-level `oracle_clearing` ceiling.

## What we did
On `E3-biased` (`visibility=noisy`), we trained `naive_mha`, `key_gram_clear`, and `key_gram_confiability_clear` on $n=2000$ `train` episodes using `tr.truth` labels only (never touching `tr.bayes` or `tr.oracle`), and evaluated all learned arms alongside `untrained_init` and `random_network` controls (`run_with_controls`) and non-learned pooling baselines on $n=2000$ `dev` episodes.

## Result
**Result in one line:** `key_gram_confiability_clear:trained` achieved **0.0514 [0.0390, 0.0633]** regret to Bayes (vs. `0.0763 [0.0602, 0.0915]` for `extremized_log_pool`, `0.0753 [0.0604, 0.0890]` for `key_gram_clear:trained`, `0.0702 [0.0549, 0.0864]` for `naive_mha:trained`, and `0.0166 [0.0095, 0.0235]` for `oracle_clearing`), with paired regret difference **-0.0249 [-0.0344, -0.0160]** against `extremized_log_pool` and **-0.0240 [-0.0301, -0.0174]** against `key_gram_clear:trained`.

## Dev Split Metrics ($n=2000$ episodes, 95% bootstrap CIs)

| Method | Regret to `bayes` (nats) | KL to `bayes` | Log loss | Brier |
|---|---:|---:|---:|---:|
| `bayes` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.2601 [0.2400, 0.2822] | 0.0808 [0.0738, 0.0887] |
| `oracle_clearing` | 0.0166 [0.0095, 0.0235] | 0.0158 [0.0142, 0.0175] | 0.2768 [0.2564, 0.2992] | 0.0862 [0.0788, 0.0944] |
| `linear_pool` | 0.1425 [0.1286, 0.1551] | 0.1319 [0.1273, 0.1363] | 0.4026 [0.3905, 0.4160] | 0.1222 [0.1172, 0.1279] |
| `log_pool` | 0.0913 [0.0773, 0.1044] | 0.0839 [0.0791, 0.0891] | 0.3514 [0.3354, 0.3708] | 0.1088 [0.1022, 0.1163] |
| `sum_logodds` | 0.3410 [0.2779, 0.4046] | 0.3626 [0.3342, 0.3910] | 0.6012 [0.5264, 0.6801] | 0.1222 [0.1101, 0.1353] |
| `extremized_log_pool` | 0.0763 [0.0602, 0.0915] | 0.0718 [0.0657, 0.0780] | 0.3364 [0.3150, 0.3613] | 0.1052 [0.0974, 0.1140] |
| `naive_mha:trained` | 0.0702 [0.0549, 0.0864] | 0.0694 [0.0632, 0.0766] | 0.3303 [0.3067, 0.3579] | 0.1032 [0.0950, 0.1126] |
| `naive_mha:control:untrained_init` | 0.4583 [0.4350, 0.4797] | 0.4441 [0.4324, 0.4551] | 0.7185 [0.7168, 0.7201] | 0.2627 [0.2618, 0.2635] |
| `naive_mha:control:random_network` | 0.4550 [0.4307, 0.4764] | 0.4402 [0.4266, 0.4519] | 0.7151 [0.7074, 0.7220] | 0.2608 [0.2569, 0.2642] |
| `key_gram_clear:trained` | 0.0753 [0.0604, 0.0890] | 0.0691 [0.0630, 0.0747] | 0.3355 [0.3146, 0.3594] | 0.1047 [0.0971, 0.1132] |
| `key_gram_clear:control:untrained_init` | 0.2037 [0.1871, 0.2199] | 0.1899 [0.1830, 0.1965] | 0.4639 [0.4527, 0.4764] | 0.1470 [0.1422, 0.1525] |
| `key_gram_clear:control:random_network` | 0.4324 [0.4104, 0.4526] | 0.4179 [0.4070, 0.4280] | 0.6926 [0.6926, 0.6926] | 0.2497 [0.2497, 0.2497] |
| `key_gram_confiability_clear:trained` | 0.0514 [0.0390, 0.0633] | 0.0478 [0.0434, 0.0522] | 0.3115 [0.2895, 0.3363] | 0.0974 [0.0898, 0.1060] |
| `key_gram_confiability_clear:control:untrained_init` | 0.2234 [0.2065, 0.2395] | 0.2101 [0.2031, 0.2169] | 0.4836 [0.4731, 0.4945] | 0.1536 [0.1489, 0.1585] |
| `key_gram_confiability_clear:control:random_network` | 0.0919 [0.0764, 0.1056] | 0.0820 [0.0762, 0.0873] | 0.3520 [0.3334, 0.3735] | 0.1094 [0.1022, 0.1174] |

## Paired Regret Differences (95% Bootstrap CIs)

| Comparison ($A - B$, negative favors $A$) | Mean $\Delta$ Regret [95% CI] |
|---|---:|
| `key_gram_confiability_clear:trained` $-$ `extremized_log_pool` | `-0.0249 [-0.0344, -0.0160]` |
| `key_gram_confiability_clear:trained` $-$ `oracle_clearing` | `+0.0348 [+0.0243, +0.0452]` |
| `key_gram_confiability_clear:trained` $-$ `key_gram_clear:trained` | `-0.0240 [-0.0301, -0.0174]` |
| `key_gram_clear:trained` $-$ `extremized_log_pool` | `-0.0009 [-0.0076, +0.0061]` |
| `key_gram_clear:trained` $-$ `oracle_clearing` | `+0.0587 [+0.0461, +0.0716]` |
| `naive_mha:trained` $-$ `extremized_log_pool` | `-0.0061 [-0.0188, +0.0062]` |
