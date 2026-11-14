# EXPLORATORY: Scalar-Report Clearing on `E2` (`visibility=noisy`)

Serves: auditable belief clearing where confiability (theta_trust in params_real) and Key-Gram de-duplication recover oracle_clearing from scalar reports across overlapping, Sybil, colluding, and biased sources.

**Status:** `● Informative` · **Track:** `EXPLORATORY` (`splits`: `train` $n=2000$, `dev` $n=2000$, `test` untouched; `harness.git_dirty = False`)

## Why
We tested why standard token attention (`naive_mha`) fails on `E2` (`noisy` keys) and whether restoring coordinate/piece permutation equivariance (`key_gram_clear`) plus a cross-agent Confiability Head $\beta_i \in \text{params}_{\text{real}}$ (`key_gram_confiability_clear`) trained strictly on `tr.truth` recovers the reports-level `oracle_clearing` ceiling.

## What we did
On `E2` (`visibility=noisy`), we trained `naive_mha`, `key_gram_clear`, and `key_gram_confiability_clear` on $n=2000$ `train` episodes using `tr.truth` labels only (never touching `tr.bayes` or `tr.oracle`), and evaluated all learned arms alongside `untrained_init` and `random_network` controls (`run_with_controls`) and non-learned pooling baselines on $n=2000$ `dev` episodes.

## Result
**Result in one line:** `key_gram_confiability_clear:trained` achieved **0.0409 [0.0304, 0.0525]** regret to Bayes (vs. `0.0448 [0.0326, 0.0564]` for `extremized_log_pool`, `0.0406 [0.0302, 0.0515]` for `key_gram_clear:trained`, `0.0561 [0.0420, 0.0713]` for `naive_mha:trained`, and `0.0166 [0.0095, 0.0235]` for `oracle_clearing`), with paired regret difference **-0.0039 [-0.0110, +0.0037]** against `extremized_log_pool` and **+0.0004 [-0.0010, +0.0018]** against `key_gram_clear:trained`.

## Dev Split Metrics ($n=2000$ episodes, 95% bootstrap CIs)

| Method | Regret to `bayes` (nats) | KL to `bayes` | Log loss | Brier |
|---|---:|---:|---:|---:|
| `bayes` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.2601 [0.2400, 0.2822] | 0.0808 [0.0738, 0.0887] |
| `oracle_clearing` | 0.0166 [0.0095, 0.0235] | 0.0158 [0.0142, 0.0175] | 0.2768 [0.2564, 0.2992] | 0.0862 [0.0788, 0.0944] |
| `linear_pool` | 0.1517 [0.1378, 0.1648] | 0.1407 [0.1364, 0.1451] | 0.4119 [0.4010, 0.4239] | 0.1247 [0.1199, 0.1300] |
| `log_pool` | 0.1124 [0.0991, 0.1244] | 0.1026 [0.0992, 0.1060] | 0.3725 [0.3600, 0.3865] | 0.1121 [0.1069, 0.1178] |
| `sum_logodds` | 0.1324 [0.0994, 0.1660] | 0.1419 [0.1308, 0.1524] | 0.3925 [0.3463, 0.4413] | 0.1044 [0.0940, 0.1159] |
| `extremized_log_pool` | 0.0448 [0.0326, 0.0564] | 0.0407 [0.0373, 0.0441] | 0.3050 [0.2826, 0.3278] | 0.0953 [0.0877, 0.1036] |
| `naive_mha:trained` | 0.0561 [0.0420, 0.0713] | 0.0556 [0.0510, 0.0598] | 0.3162 [0.2938, 0.3412] | 0.1004 [0.0923, 0.1091] |
| `naive_mha:control:untrained_init` | 0.4510 [0.4279, 0.4722] | 0.4368 [0.4252, 0.4478] | 0.7112 [0.7098, 0.7125] | 0.2590 [0.2583, 0.2597] |
| `naive_mha:control:random_network` | 0.4549 [0.4304, 0.4762] | 0.4403 [0.4266, 0.4523] | 0.7151 [0.7067, 0.7228] | 0.2607 [0.2565, 0.2645] |
| `key_gram_clear:trained` | 0.0406 [0.0302, 0.0515] | 0.0376 [0.0341, 0.0410] | 0.3007 [0.2804, 0.3237] | 0.0941 [0.0870, 0.1023] |
| `key_gram_clear:control:untrained_init` | 0.2319 [0.2153, 0.2482] | 0.2203 [0.2144, 0.2261] | 0.4921 [0.4843, 0.5008] | 0.1556 [0.1521, 0.1596] |
| `key_gram_clear:control:random_network` | 0.4316 [0.4096, 0.4517] | 0.4170 [0.4062, 0.4271] | 0.6917 [0.6917, 0.6918] | 0.2493 [0.2493, 0.2493] |
| `key_gram_confiability_clear:trained` | 0.0409 [0.0304, 0.0525] | 0.0386 [0.0349, 0.0422] | 0.3011 [0.2803, 0.3244] | 0.0943 [0.0872, 0.1030] |
| `key_gram_confiability_clear:control:untrained_init` | 0.2566 [0.2383, 0.2734] | 0.2459 [0.2387, 0.2533] | 0.5167 [0.5084, 0.5260] | 0.1663 [0.1625, 0.1704] |
| `key_gram_confiability_clear:control:random_network` | 0.0834 [0.0712, 0.0953] | 0.0752 [0.0721, 0.0787] | 0.3435 [0.3298, 0.3586] | 0.1036 [0.0981, 0.1100] |

## Paired Regret Differences (95% Bootstrap CIs)

| Comparison ($A - B$, negative favors $A$) | Mean $\Delta$ Regret [95% CI] |
|---|---:|
| `key_gram_confiability_clear:trained` $-$ `extremized_log_pool` | `-0.0039 [-0.0110, +0.0037]` |
| `key_gram_confiability_clear:trained` $-$ `oracle_clearing` | `+0.0243 [+0.0154, +0.0337]` |
| `key_gram_confiability_clear:trained` $-$ `key_gram_clear:trained` | `+0.0004 [-0.0010, +0.0018]` |
| `key_gram_clear:trained` $-$ `extremized_log_pool` | `-0.0043 [-0.0110, +0.0027]` |
| `key_gram_clear:trained` $-$ `oracle_clearing` | `+0.0240 [+0.0149, +0.0329]` |
| `naive_mha:trained` $-$ `extremized_log_pool` | `+0.0113 [+0.0027, +0.0202]` |
