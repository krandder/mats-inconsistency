# EXPLORATORY: Scalar-Report Clearing on `E3-sybil` (`visibility=noisy`)

Serves: auditable belief clearing where confiability (theta_trust in params_real) and Key-Gram de-duplication recover oracle_clearing from scalar reports across overlapping, Sybil, colluding, and biased sources.

**Status:** `● Gain` · **Track:** `EXPLORATORY` (`splits`: `train` $n=2000$, `dev` $n=2000$, `test` untouched; `harness.git_dirty = False`)

## Why
We tested why standard token attention (`naive_mha`) fails on `E3-sybil` (`noisy` keys) and whether restoring coordinate/piece permutation equivariance (`key_gram_clear`) plus a cross-agent Confiability Head $\beta_i \in \text{params}_{\text{real}}$ (`key_gram_confiability_clear`) trained strictly on `tr.truth` recovers the reports-level `oracle_clearing` ceiling.

## What we did
On `E3-sybil` (`visibility=noisy`), we trained `naive_mha`, `key_gram_clear`, and `key_gram_confiability_clear` on $n=2000$ `train` episodes using `tr.truth` labels only (never touching `tr.bayes` or `tr.oracle`), and evaluated all learned arms alongside `untrained_init` and `random_network` controls (`run_with_controls`) and non-learned pooling baselines on $n=2000$ `dev` episodes.

## Result
**Result in one line:** `key_gram_confiability_clear:trained` achieved **0.0421 [0.0303, 0.0538]** regret to Bayes (vs. `0.0778 [0.0623, 0.0919]` for `extremized_log_pool`, `0.0495 [0.0366, 0.0614]` for `key_gram_clear:trained`, `0.0727 [0.0558, 0.0876]` for `naive_mha:trained`, and `0.0166 [0.0095, 0.0235]` for `oracle_clearing`), with paired regret difference **-0.0357 [-0.0470, -0.0239]** against `extremized_log_pool` and **-0.0074 [-0.0125, -0.0022]** against `key_gram_clear:trained`.

## Dev Split Metrics ($n=2000$ episodes, 95% bootstrap CIs)

| Method | Regret to `bayes` (nats) | KL to `bayes` | Log loss | Brier |
|---|---:|---:|---:|---:|
| `bayes` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.2601 [0.2400, 0.2822] | 0.0808 [0.0738, 0.0887] |
| `oracle_clearing` | 0.0166 [0.0095, 0.0235] | 0.0158 [0.0142, 0.0175] | 0.2768 [0.2564, 0.2992] | 0.0862 [0.0788, 0.0944] |
| `linear_pool` | 0.1587 [0.1435, 0.1723] | 0.1484 [0.1428, 0.1541] | 0.4189 [0.4061, 0.4325] | 0.1293 [0.1239, 0.1352] |
| `log_pool` | 0.1261 [0.1112, 0.1392] | 0.1176 [0.1127, 0.1227] | 0.3863 [0.3725, 0.4013] | 0.1188 [0.1129, 0.1251] |
| `sum_logodds` | 0.4170 [0.3450, 0.4936] | 0.4509 [0.4178, 0.4829] | 0.6772 [0.5911, 0.7704] | 0.1278 [0.1151, 0.1420] |
| `extremized_log_pool` | 0.0778 [0.0623, 0.0919] | 0.0747 [0.0687, 0.0808] | 0.3380 [0.3156, 0.3617] | 0.1069 [0.0988, 0.1159] |
| `naive_mha:trained` | 0.0727 [0.0558, 0.0876] | 0.0718 [0.0658, 0.0779] | 0.3328 [0.3090, 0.3580] | 0.1063 [0.0981, 0.1150] |
| `naive_mha:control:untrained_init` | 0.4525 [0.4298, 0.4725] | 0.4377 [0.4262, 0.4483] | 0.7126 [0.7111, 0.7141] | 0.2597 [0.2590, 0.2605] |
| `naive_mha:control:random_network` | 0.4525 [0.4282, 0.4740] | 0.4383 [0.4253, 0.4501] | 0.7126 [0.7052, 0.7197] | 0.2595 [0.2559, 0.2630] |
| `key_gram_clear:trained` | 0.0495 [0.0366, 0.0614] | 0.0468 [0.0427, 0.0509] | 0.3096 [0.2878, 0.3323] | 0.0976 [0.0897, 0.1060] |
| `key_gram_clear:control:untrained_init` | 0.2465 [0.2282, 0.2636] | 0.2351 [0.2289, 0.2418] | 0.5067 [0.4983, 0.5158] | 0.1623 [0.1586, 0.1665] |
| `key_gram_clear:control:random_network` | 0.4236 [0.4017, 0.4435] | 0.4091 [0.3987, 0.4191] | 0.6837 [0.6833, 0.6842] | 0.2453 [0.2451, 0.2455] |
| `key_gram_confiability_clear:trained` | 0.0421 [0.0303, 0.0538] | 0.0393 [0.0354, 0.0429] | 0.3022 [0.2806, 0.3270] | 0.0952 [0.0873, 0.1038] |
| `key_gram_confiability_clear:control:untrained_init` | 0.3040 [0.2827, 0.3239] | 0.2903 [0.2784, 0.3031] | 0.5642 [0.5515, 0.5779] | 0.1874 [0.1824, 0.1926] |
| `key_gram_confiability_clear:control:random_network` | 0.0656 [0.0535, 0.0767] | 0.0594 [0.0563, 0.0626] | 0.3258 [0.3100, 0.3422] | 0.1000 [0.0936, 0.1069] |

## Paired Regret Differences (95% Bootstrap CIs)

| Comparison ($A - B$, negative favors $A$) | Mean $\Delta$ Regret [95% CI] |
|---|---:|
| `key_gram_confiability_clear:trained` $-$ `extremized_log_pool` | `-0.0357 [-0.0470, -0.0239]` |
| `key_gram_confiability_clear:trained` $-$ `oracle_clearing` | `+0.0255 [+0.0157, +0.0353]` |
| `key_gram_confiability_clear:trained` $-$ `key_gram_clear:trained` | `-0.0074 [-0.0125, -0.0022]` |
| `key_gram_clear:trained` $-$ `extremized_log_pool` | `-0.0283 [-0.0374, -0.0196]` |
| `key_gram_clear:trained` $-$ `oracle_clearing` | `+0.0329 [+0.0224, +0.0432]` |
| `naive_mha:trained` $-$ `extremized_log_pool` | `-0.0051 [-0.0164, +0.0060]` |
