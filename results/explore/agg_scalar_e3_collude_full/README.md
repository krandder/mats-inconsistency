# EXPLORATORY: Scalar-Report Clearing on `E3-collude` (`visibility=full`)

Serves: auditable belief clearing where confiability (theta_trust in params_real) and Key-Gram de-duplication recover oracle_clearing from scalar reports across overlapping, Sybil, colluding, and biased sources.

**Status:** `● Gain` · **Track:** `EXPLORATORY` (`splits`: `train` $n=2000$, `dev` $n=2000$, `test` untouched; `harness.git_dirty = False`)

## Why
We tested why standard token attention (`naive_mha`) fails on `E3-collude` (`full` keys) and whether restoring coordinate/piece permutation equivariance (`key_gram_clear`) plus a cross-agent Confiability Head $\beta_i \in \text{params}_{\text{real}}$ (`key_gram_confiability_clear`) trained strictly on `tr.truth` recovers the reports-level `oracle_clearing` ceiling.

## What we did
On `E3-collude` (`visibility=full`), we trained `naive_mha`, `key_gram_clear`, and `key_gram_confiability_clear` on $n=2000$ `train` episodes using `tr.truth` labels only (never touching `tr.bayes` or `tr.oracle`), and evaluated all learned arms alongside `untrained_init` and `random_network` controls (`run_with_controls`) and non-learned pooling baselines on $n=2000$ `dev` episodes.

## Result
**Result in one line:** `key_gram_confiability_clear:trained` achieved **0.0186 [0.0107, 0.0256]** regret to Bayes (vs. `0.0302 [0.0210, 0.0391]` for `extremized_log_pool`, `0.0185 [0.0107, 0.0256]` for `key_gram_clear:trained`, `0.0478 [0.0351, 0.0612]` for `naive_mha:trained`, and `0.0175 [0.0095, 0.0244]` for `oracle_clearing`), with paired regret difference **-0.0117 [-0.0175, -0.0058]** against `extremized_log_pool` and **+0.0000 [-0.0000, +0.0001]** against `key_gram_clear:trained`.

## Dev Split Metrics ($n=2000$ episodes, 95% bootstrap CIs)

| Method | Regret to `bayes` (nats) | KL to `bayes` | Log loss | Brier |
|---|---:|---:|---:|---:|
| `bayes` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.2601 [0.2400, 0.2822] | 0.0808 [0.0738, 0.0887] |
| `oracle_clearing` | 0.0175 [0.0095, 0.0244] | 0.0144 [0.0127, 0.0160] | 0.2777 [0.2561, 0.3009] | 0.0870 [0.0793, 0.0956] |
| `linear_pool` | 0.0744 [0.0634, 0.0847] | 0.0669 [0.0641, 0.0701] | 0.3345 [0.3196, 0.3510] | 0.1001 [0.0940, 0.1070] |
| `log_pool` | 0.0463 [0.0364, 0.0556] | 0.0410 [0.0385, 0.0437] | 0.3064 [0.2894, 0.3251] | 0.0944 [0.0879, 0.1016] |
| `sum_logodds` | 0.2589 [0.2077, 0.3136] | 0.2912 [0.2751, 0.3074] | 0.5190 [0.4513, 0.5938] | 0.1114 [0.1000, 0.1240] |
| `extremized_log_pool` | 0.0302 [0.0210, 0.0391] | 0.0286 [0.0259, 0.0315] | 0.2904 [0.2693, 0.3141] | 0.0920 [0.0843, 0.1004] |
| `naive_mha:trained` | 0.0478 [0.0351, 0.0612] | 0.0513 [0.0462, 0.0567] | 0.3080 [0.2845, 0.3344] | 0.0979 [0.0895, 0.1069] |
| `naive_mha:control:untrained_init` | 0.4639 [0.4407, 0.4851] | 0.4501 [0.4381, 0.4615] | 0.7240 [0.7212, 0.7267] | 0.2654 [0.2640, 0.2667] |
| `naive_mha:control:random_network` | 0.4634 [0.4374, 0.4857] | 0.4492 [0.4350, 0.4619] | 0.7235 [0.7136, 0.7327] | 0.2647 [0.2598, 0.2691] |
| `key_gram_clear:trained` | 0.0185 [0.0107, 0.0256] | 0.0154 [0.0136, 0.0171] | 0.2787 [0.2577, 0.3017] | 0.0873 [0.0798, 0.0958] |
| `key_gram_clear:control:untrained_init` | 0.2291 [0.2122, 0.2448] | 0.2168 [0.2119, 0.2218] | 0.4893 [0.4827, 0.4970] | 0.1534 [0.1503, 0.1571] |
| `key_gram_clear:control:random_network` | 0.4101 [0.3882, 0.4296] | 0.3957 [0.3858, 0.4054] | 0.6702 [0.6692, 0.6713] | 0.2386 [0.2380, 0.2391] |
| `key_gram_confiability_clear:trained` | 0.0186 [0.0107, 0.0256] | 0.0154 [0.0136, 0.0171] | 0.2787 [0.2577, 0.3017] | 0.0873 [0.0798, 0.0958] |
| `key_gram_confiability_clear:control:untrained_init` | 0.2376 [0.2208, 0.2533] | 0.2252 [0.2197, 0.2303] | 0.4978 [0.4910, 0.5055] | 0.1572 [0.1540, 0.1608] |
| `key_gram_confiability_clear:control:random_network` | 0.0882 [0.0764, 0.0996] | 0.0791 [0.0764, 0.0817] | 0.3484 [0.3357, 0.3627] | 0.1035 [0.0981, 0.1096] |

## Paired Regret Differences (95% Bootstrap CIs)

| Comparison ($A - B$, negative favors $A$) | Mean $\Delta$ Regret [95% CI] |
|---|---:|
| `key_gram_confiability_clear:trained` $-$ `extremized_log_pool` | `-0.0117 [-0.0175, -0.0058]` |
| `key_gram_confiability_clear:trained` $-$ `oracle_clearing` | `+0.0010 [-0.0008, +0.0029]` |
| `key_gram_confiability_clear:trained` $-$ `key_gram_clear:trained` | `+0.0000 [-0.0000, +0.0001]` |
| `key_gram_clear:trained` $-$ `extremized_log_pool` | `-0.0117 [-0.0175, -0.0059]` |
| `key_gram_clear:trained` $-$ `oracle_clearing` | `+0.0010 [-0.0008, +0.0028]` |
| `naive_mha:trained` $-$ `extremized_log_pool` | `+0.0176 [+0.0080, +0.0274]` |
