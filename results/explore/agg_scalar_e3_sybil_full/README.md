# EXPLORATORY: Scalar-Report Clearing on `E3-sybil` (`visibility=full`)

Serves: auditable belief clearing where confiability (theta_trust in params_real) and Key-Gram de-duplication recover oracle_clearing from scalar reports across overlapping, Sybil, colluding, and biased sources.

**Status:** `● Gain` · **Track:** `EXPLORATORY` (`splits`: `train` $n=2000$, `dev` $n=2000$, `test` untouched; `harness.git_dirty = False`)

## Why
We tested why standard token attention (`naive_mha`) fails on `E3-sybil` (`full` keys) and whether restoring coordinate/piece permutation equivariance (`key_gram_clear`) plus a cross-agent Confiability Head $\beta_i \in \text{params}_{\text{real}}$ (`key_gram_confiability_clear`) trained strictly on `tr.truth` recovers the reports-level `oracle_clearing` ceiling.

## What we did
On `E3-sybil` (`visibility=full`), we trained `naive_mha`, `key_gram_clear`, and `key_gram_confiability_clear` on $n=2000$ `train` episodes using `tr.truth` labels only (never touching `tr.bayes` or `tr.oracle`), and evaluated all learned arms alongside `untrained_init` and `random_network` controls (`run_with_controls`) and non-learned pooling baselines on $n=2000$ `dev` episodes.

## Result
**Result in one line:** `key_gram_confiability_clear:trained` achieved **0.0178 [0.0106, 0.0245]** regret to Bayes (vs. `0.0778 [0.0623, 0.0919]` for `extremized_log_pool`, `0.0179 [0.0107, 0.0245]` for `key_gram_clear:trained`, `0.0674 [0.0506, 0.0834]` for `naive_mha:trained`, and `0.0166 [0.0095, 0.0235]` for `oracle_clearing`), with paired regret difference **-0.0600 [-0.0723, -0.0475]** against `extremized_log_pool` and **-0.0001 [-0.0002, +0.0001]** against `key_gram_clear:trained`.

## Dev Split Metrics ($n=2000$ episodes, 95% bootstrap CIs)

| Method | Regret to `bayes` (nats) | KL to `bayes` | Log loss | Brier |
|---|---:|---:|---:|---:|
| `bayes` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.2601 [0.2400, 0.2822] | 0.0808 [0.0738, 0.0887] |
| `oracle_clearing` | 0.0166 [0.0095, 0.0235] | 0.0158 [0.0142, 0.0175] | 0.2768 [0.2564, 0.2992] | 0.0862 [0.0788, 0.0944] |
| `linear_pool` | 0.1587 [0.1435, 0.1723] | 0.1484 [0.1428, 0.1541] | 0.4189 [0.4061, 0.4325] | 0.1293 [0.1239, 0.1352] |
| `log_pool` | 0.1261 [0.1112, 0.1392] | 0.1176 [0.1127, 0.1227] | 0.3863 [0.3725, 0.4013] | 0.1188 [0.1129, 0.1251] |
| `sum_logodds` | 0.4170 [0.3450, 0.4936] | 0.4509 [0.4178, 0.4829] | 0.6772 [0.5911, 0.7704] | 0.1278 [0.1151, 0.1420] |
| `extremized_log_pool` | 0.0778 [0.0623, 0.0919] | 0.0747 [0.0687, 0.0808] | 0.3380 [0.3156, 0.3617] | 0.1069 [0.0988, 0.1159] |
| `naive_mha:trained` | 0.0674 [0.0506, 0.0834] | 0.0683 [0.0620, 0.0746] | 0.3275 [0.3031, 0.3548] | 0.1027 [0.0944, 0.1121] |
| `naive_mha:control:untrained_init` | 0.4525 [0.4295, 0.4728] | 0.4381 [0.4265, 0.4487] | 0.7127 [0.7111, 0.7142] | 0.2598 [0.2589, 0.2605] |
| `naive_mha:control:random_network` | 0.4544 [0.4292, 0.4762] | 0.4406 [0.4269, 0.4528] | 0.7145 [0.7057, 0.7226] | 0.2604 [0.2560, 0.2643] |
| `key_gram_clear:trained` | 0.0179 [0.0107, 0.0245] | 0.0161 [0.0145, 0.0178] | 0.2780 [0.2582, 0.2994] | 0.0865 [0.0792, 0.0945] |
| `key_gram_clear:control:untrained_init` | 0.2143 [0.1979, 0.2296] | 0.2020 [0.1973, 0.2068] | 0.4745 [0.4670, 0.4830] | 0.1475 [0.1440, 0.1515] |
| `key_gram_clear:control:random_network` | 0.3762 [0.3552, 0.3952] | 0.3624 [0.3539, 0.3711] | 0.6364 [0.6336, 0.6394] | 0.2219 [0.2205, 0.2233] |
| `key_gram_confiability_clear:trained` | 0.0178 [0.0106, 0.0245] | 0.0161 [0.0144, 0.0178] | 0.2780 [0.2580, 0.2994] | 0.0864 [0.0791, 0.0945] |
| `key_gram_confiability_clear:control:untrained_init` | 0.2404 [0.2232, 0.2565] | 0.2287 [0.2225, 0.2348] | 0.5006 [0.4923, 0.5091] | 0.1590 [0.1551, 0.1630] |
| `key_gram_confiability_clear:control:random_network` | 0.1035 [0.0908, 0.1150] | 0.0941 [0.0913, 0.0970] | 0.3637 [0.3517, 0.3767] | 0.1080 [0.1030, 0.1137] |

## Paired Regret Differences (95% Bootstrap CIs)

| Comparison ($A - B$, negative favors $A$) | Mean $\Delta$ Regret [95% CI] |
|---|---:|
| `key_gram_confiability_clear:trained` $-$ `extremized_log_pool` | `-0.0600 [-0.0723, -0.0475]` |
| `key_gram_confiability_clear:trained` $-$ `oracle_clearing` | `+0.0012 [+0.0001, +0.0022]` |
| `key_gram_confiability_clear:trained` $-$ `key_gram_clear:trained` | `-0.0001 [-0.0002, +0.0001]` |
| `key_gram_clear:trained` $-$ `extremized_log_pool` | `-0.0599 [-0.0721, -0.0475]` |
| `key_gram_clear:trained` $-$ `oracle_clearing` | `+0.0013 [+0.0001, +0.0024]` |
| `naive_mha:trained` $-$ `extremized_log_pool` | `-0.0104 [-0.0231, +0.0024]` |
