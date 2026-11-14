# EXPLORATORY: Scalar-Report Clearing on `E3-collude` (`visibility=noisy`)

Serves: auditable belief clearing where confiability (theta_trust in params_real) and Key-Gram de-duplication recover oracle_clearing from scalar reports across overlapping, Sybil, colluding, and biased sources.

**Status:** `● Informative` · **Track:** `EXPLORATORY` (`splits`: `train` $n=2000$, `dev` $n=2000$, `test` untouched; `harness.git_dirty = False`)

## Why
We tested why standard token attention (`naive_mha`) fails on `E3-collude` (`noisy` keys) and whether restoring coordinate/piece permutation equivariance (`key_gram_clear`) plus a cross-agent Confiability Head $\beta_i \in \text{params}_{\text{real}}$ (`key_gram_confiability_clear`) trained strictly on `tr.truth` recovers the reports-level `oracle_clearing` ceiling.

## What we did
On `E3-collude` (`visibility=noisy`), we trained `naive_mha`, `key_gram_clear`, and `key_gram_confiability_clear` on $n=2000$ `train` episodes using `tr.truth` labels only (never touching `tr.bayes` or `tr.oracle`), and evaluated all learned arms alongside `untrained_init` and `random_network` controls (`run_with_controls`) and non-learned pooling baselines on $n=2000$ `dev` episodes.

## Result
**Result in one line:** `key_gram_confiability_clear:trained` achieved **0.0283 [0.0193, 0.0370]** regret to Bayes (vs. `0.0302 [0.0210, 0.0391]` for `extremized_log_pool`, `0.0279 [0.0189, 0.0367]` for `key_gram_clear:trained`, `0.0437 [0.0323, 0.0564]` for `naive_mha:trained`, and `0.0175 [0.0095, 0.0244]` for `oracle_clearing`), with paired regret difference **-0.0020 [-0.0065, +0.0024]** against `extremized_log_pool` and **+0.0003 [-0.0001, +0.0008]** against `key_gram_clear:trained`.

## Dev Split Metrics ($n=2000$ episodes, 95% bootstrap CIs)

| Method | Regret to `bayes` (nats) | KL to `bayes` | Log loss | Brier |
|---|---:|---:|---:|---:|
| `bayes` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.2601 [0.2400, 0.2822] | 0.0808 [0.0738, 0.0887] |
| `oracle_clearing` | 0.0175 [0.0095, 0.0244] | 0.0144 [0.0127, 0.0160] | 0.2777 [0.2561, 0.3009] | 0.0870 [0.0793, 0.0956] |
| `linear_pool` | 0.0744 [0.0634, 0.0847] | 0.0669 [0.0641, 0.0701] | 0.3345 [0.3196, 0.3510] | 0.1001 [0.0940, 0.1070] |
| `log_pool` | 0.0463 [0.0364, 0.0556] | 0.0410 [0.0385, 0.0437] | 0.3064 [0.2894, 0.3251] | 0.0944 [0.0879, 0.1016] |
| `sum_logodds` | 0.2589 [0.2077, 0.3136] | 0.2912 [0.2751, 0.3074] | 0.5190 [0.4513, 0.5938] | 0.1114 [0.1000, 0.1240] |
| `extremized_log_pool` | 0.0302 [0.0210, 0.0391] | 0.0286 [0.0259, 0.0315] | 0.2904 [0.2693, 0.3141] | 0.0920 [0.0843, 0.1004] |
| `naive_mha:trained` | 0.0437 [0.0323, 0.0564] | 0.0441 [0.0400, 0.0486] | 0.3039 [0.2806, 0.3294] | 0.0958 [0.0877, 0.1044] |
| `naive_mha:control:untrained_init` | 0.4625 [0.4394, 0.4836] | 0.4487 [0.4365, 0.4601] | 0.7227 [0.7201, 0.7252] | 0.2647 [0.2634, 0.2659] |
| `naive_mha:control:random_network` | 0.4624 [0.4368, 0.4843] | 0.4473 [0.4336, 0.4596] | 0.7225 [0.7134, 0.7311] | 0.2642 [0.2598, 0.2684] |
| `key_gram_clear:trained` | 0.0279 [0.0189, 0.0367] | 0.0254 [0.0228, 0.0280] | 0.2881 [0.2670, 0.3122] | 0.0907 [0.0829, 0.0994] |
| `key_gram_clear:control:untrained_init` | 0.2299 [0.2132, 0.2456] | 0.2182 [0.2128, 0.2235] | 0.4900 [0.4827, 0.4983] | 0.1543 [0.1509, 0.1580] |
| `key_gram_clear:control:random_network` | 0.4291 [0.4072, 0.4491] | 0.4146 [0.4039, 0.4247] | 0.6893 [0.6891, 0.6895] | 0.2481 [0.2480, 0.2482] |
| `key_gram_confiability_clear:trained` | 0.0283 [0.0193, 0.0370] | 0.0254 [0.0229, 0.0280] | 0.2884 [0.2674, 0.3126] | 0.0908 [0.0830, 0.0995] |
| `key_gram_confiability_clear:control:untrained_init` | 0.2595 [0.2415, 0.2761] | 0.2466 [0.2401, 0.2530] | 0.5196 [0.5120, 0.5277] | 0.1675 [0.1639, 0.1712] |
| `key_gram_confiability_clear:control:random_network` | 0.0885 [0.0765, 0.0997] | 0.0795 [0.0766, 0.0823] | 0.3487 [0.3355, 0.3637] | 0.1042 [0.0986, 0.1103] |

## Paired Regret Differences (95% Bootstrap CIs)

| Comparison ($A - B$, negative favors $A$) | Mean $\Delta$ Regret [95% CI] |
|---|---:|
| `key_gram_confiability_clear:trained` $-$ `extremized_log_pool` | `-0.0020 [-0.0065, +0.0024]` |
| `key_gram_confiability_clear:trained` $-$ `oracle_clearing` | `+0.0108 [+0.0042, +0.0167]` |
| `key_gram_confiability_clear:trained` $-$ `key_gram_clear:trained` | `+0.0003 [-0.0001, +0.0008]` |
| `key_gram_clear:trained` $-$ `extremized_log_pool` | `-0.0023 [-0.0070, +0.0022]` |
| `key_gram_clear:trained` $-$ `oracle_clearing` | `+0.0104 [+0.0039, +0.0163]` |
| `naive_mha:trained` $-$ `extremized_log_pool` | `+0.0135 [+0.0048, +0.0216]` |
