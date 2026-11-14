Serves: aggregation that is robust to duplicated and colluding sources, as a market should be

# L5: Sybil copies, colluders and a biased source with hidden or noisy provenance (EXPLORATORY)

**Track E (EXPLORATORY, docs/PROTOCOL.md 10.1).** These results guide direction only. They are not findings, and they have not been audited.
Dev split only; the test split was never sampled. Harness `aggbench` imported unmodified (every JSON records `harness.git_dirty: false`).

**One-line result:** With provenance hidden (`none`), a set network trained with adversaries (matched or pooled) keeps regret at 0.042-0.043 nats for 1, 3 and 7 Sybil copies, the same as with no Sybils (0.042). Over the same range, extremized_log_pool rises from 0.051 to 0.119. It reaches 0.026 with colluders and 0.050 with a biased source, 0.007-0.034 above oracle_clearing. Trained without adversaries, it discounts Sybils only partly and fails on the biased source (0.228). The wealth market is no better than extremized_log_pool without persistent identities. Even with oracle identities it does not discount Sybil copies, because a copy earns exactly the same wealth as its source.

## Question

Provenance is hidden (visibility `none`: reports only) or noisy (`noisy`: coverage keys with 15% bit flips). Can a learned aggregator find and discount Sybil copies, colluders and a biased source from the content of the reports alone? How much regret remains compared with `oracle_clearing`? We vary the number of Sybil copies (0, 1, 3, 7) and whether training includes adversaries. A market-style baseline weights each source by a wealth earned from its log score.

## Environments (all `aggbench.make_env`, dev split, 2000 episodes each)

| label | preset + overrides | agents |
|---|---|---|
| sybil0 | `E2` | 5 honest |
| sybil1 / sybil3 / sybil7 | `E3-sybil`, `n_sybil` = 1 / 3 / 7 | 5 + k exact copies of one honest agent |
| collude3 | `E3-collude` | 3 colluders who pool their pieces and all report the pooled posterior, 2 honest |
| biased1 | `E3-biased` | 1 agent reports l0 + 3 (l - l0) + 1.5, 4 honest |
| mixed | `E3-sybil`, `n_sybil=3, n_colluders=3, n_biased=1` | 3 colluders, 1 biased, 1 honest + 3 copies of it |

## Arms

- **set (learned).** A pairwise-relation set network over agent tokens [(l_i - l0)/4, key_i]. Its pair features are the report difference d_ij as (d/4, |d|/4, log(|d| + 1e-3)/7) and, when keys exist, the mean absolute key difference. Two relation-attention blocks, width 64, 4 heads. Output: l0 + sum_i w_i (l_i - l0) + sum_i c_i, with w_i and c_i read out per agent. It is trained on truth labels (no distillation): AdamW, lr 2e-3, one-cycle schedule, batch 512, 4000 steps, 40,000 train episodes, 3 training seeds. There are three training regimes:
  - **clean**: E2 with the same total number of agents and no adversaries. This tests whether training needs to include adversaries.
  - **matched**: the evaluation environment's own train split.
  - **pooled**: all seven environments mixed, 7 x 40,000 episodes from disjoint train seed ranges, 8000 steps.
- **market_wealth:slot (the traders analogy).** Each source's wealth is exp(eta x its cumulative log score over the training episodes). This is a Hedge update; eta = 1 is the Kelly/Bayes wealth update. The weights are normalised wealth, and the aggregate is l0 + alpha sum_i w_i (l_i - l0). eta comes from {0, 1e-4, ..., 1}, chosen on the last 20% of train; alpha is fitted on train. A "source" is an agent slot of the view, and the harness shuffles slots every episode. So this is the market that can actually be built from the view.
- **REF market_wealth:identity** and **REF logpool_fit:identity**. These references are not fair arms. They use the same market (or per-source weights and offsets fitted by L-BFGS on train log loss), but over persistent trader identities rebuilt from the oracle roles: colluders, biased, honest, Sybil source and copies. The view has no identities. These arms show what a market would do if traders kept their names across episodes.
- **Harness baselines.** `oracle_clearing` (the reports-level ceiling), `extremized_log_pool`, `log_pool`, `linear_pool` and `sum_logodds`, all from `aggbench.baselines`.
- **Controls.** Every learned arm, the markets included, goes through `aggbench.controls.run_with_controls`, which adds an untrained-init and a scale-matched random-weight version through the same `predict`.

**Follow-up arms (`set:clean-280k`, `set:matched-280k`, visibility `noisy`, seed 0).** In the main run, clean and matched training at `noisy` overfit on 40,000 episodes. Their dev regret (3-seed means 0.12-1.16) is often worse than their own untrained control: sybil0 clean scores 0.476, against 0.24 for the untrained control. Their final training-minibatch losses go as low as 0.08, against 0.25-0.33 at `none` and a Bayes log loss of 0.26. The pooled arm differs from them in two ways: it sees adversaries, and it trains on 7x the data. To separate the two, a second VM run retrained clean and matched at `noisy` with the pooled arm's budget: 280,000 episodes and 8000 steps. The 40k `noisy` clean and matched rows below are kept, but they measure overfitting, not robustness.

Metric: mean regret to the global Bayes posterior in nats per episode, with a 95% percentile bootstrap CI over the 2000 dev episodes (`aggbench.metrics.summarize`). Paired differences come from `aggbench.metrics.paired_difference`.

## Results

All numbers below are generated by `python3 run_l5.py tables` from `json/*.json` (one harness-validated `aggbench.result/v1` file per environment and visibility; `extra` holds the paired differences, seed spreads, weight diagnostics and market fits).

<!-- TABLES:BEGIN -->

**Regret to Bayes (nats/episode), visibility `none`**, mean [95% bootstrap CI] over 2000 dev episodes; set arms are training seed 0.

| env | oracle_clearing | extremized | market: slot wealth | set: clean-trained | set: matched-trained | set: pooled-trained | REF market: identity wealth | REF identity log pool |
|---|---|---|---|---|---|---|---|---|
| sybil0 | 0.017 [0.009, 0.024] | 0.045 [0.033, 0.056] | 0.045 [0.033, 0.056] | 0.042 [0.031, 0.053] | = clean | 0.046 [0.034, 0.057] | 0.045 [0.033, 0.056] | 0.045 [0.033, 0.056] |
| sybil1 | 0.017 [0.009, 0.024] | 0.051 [0.038, 0.063] | 0.051 [0.038, 0.063] | 0.045 [0.032, 0.057] | 0.042 [0.030, 0.053] | 0.042 [0.031, 0.054] | 0.051 [0.038, 0.063] | 0.045 [0.033, 0.057] |
| sybil3 | 0.017 [0.009, 0.024] | 0.078 [0.062, 0.092] | 0.078 [0.063, 0.092] | 0.060 [0.045, 0.074] | 0.042 [0.030, 0.054] | 0.044 [0.032, 0.055] | 0.078 [0.062, 0.092] | 0.045 [0.033, 0.057] |
| sybil7 | 0.017 [0.009, 0.024] | 0.119 [0.101, 0.136] | 0.119 [0.101, 0.136] | 0.099 [0.078, 0.119] | 0.043 [0.031, 0.054] | 0.043 [0.031, 0.054] | 0.111 [0.093, 0.127] | 0.045 [0.033, 0.057] |
| collude3 | 0.018 [0.010, 0.025] | 0.030 [0.020, 0.039] | 0.030 [0.020, 0.039] | 0.040 [0.026, 0.055] | 0.026 [0.017, 0.035] | 0.028 [0.018, 0.037] | 0.030 [0.020, 0.039] | 0.028 [0.018, 0.037] |
| biased1 | 0.017 [0.009, 0.024] | 0.076 [0.061, 0.091] | 0.076 [0.061, 0.091] | 0.221 [0.178, 0.265] | 0.050 [0.038, 0.062] | 0.055 [0.041, 0.069] | 0.048 [0.036, 0.060] | 0.045 [0.033, 0.056] |
| mixed | 0.018 [0.010, 0.025] | 0.042 [0.030, 0.053] | 0.042 [0.030, 0.053] | 0.138 [0.106, 0.172] | 0.025 [0.015, 0.034] | 0.029 [0.019, 0.038] | 0.034 [0.024, 0.044] | 0.028 [0.018, 0.037] |

**Regret to Bayes (nats/episode), visibility `noisy`**, mean [95% bootstrap CI] over 2000 dev episodes; set arms are training seed 0. The clean- and matched-trained columns here are the 40k runs, which overfit; see the equal-data follow-up table.

| env | oracle_clearing | extremized | market: slot wealth | set: clean-trained | set: matched-trained | set: pooled-trained | REF market: identity wealth | REF identity log pool |
|---|---|---|---|---|---|---|---|---|
| sybil0 | 0.017 [0.009, 0.024] | 0.045 [0.033, 0.056] | 0.045 [0.033, 0.056] | 0.441 [0.374, 0.509] | = clean | 0.040 [0.029, 0.050] | 0.045 [0.033, 0.056] | 0.045 [0.033, 0.056] |
| sybil1 | 0.017 [0.009, 0.024] | 0.051 [0.038, 0.063] | 0.051 [0.038, 0.063] | 0.381 [0.315, 0.452] | 0.179 [0.143, 0.217] | 0.040 [0.029, 0.051] | 0.051 [0.038, 0.063] | 0.045 [0.033, 0.057] |
| sybil3 | 0.017 [0.009, 0.024] | 0.078 [0.062, 0.092] | 0.078 [0.063, 0.092] | 0.261 [0.213, 0.309] | 0.128 [0.098, 0.158] | 0.043 [0.032, 0.054] | 0.078 [0.062, 0.092] | 0.045 [0.033, 0.057] |
| sybil7 | 0.017 [0.009, 0.024] | 0.119 [0.101, 0.136] | 0.119 [0.101, 0.136] | 0.185 [0.149, 0.226] | 0.107 [0.086, 0.129] | 0.041 [0.030, 0.052] | 0.111 [0.093, 0.127] | 0.045 [0.033, 0.057] |
| collude3 | 0.018 [0.010, 0.025] | 0.030 [0.020, 0.039] | 0.030 [0.020, 0.039] | 0.499 [0.416, 0.583] | 0.226 [0.179, 0.275] | 0.027 [0.018, 0.036] | 0.030 [0.020, 0.039] | 0.028 [0.018, 0.037] |
| biased1 | 0.017 [0.009, 0.024] | 0.076 [0.061, 0.091] | 0.076 [0.061, 0.091] | 0.997 [0.834, 1.162] | 0.443 [0.371, 0.517] | 0.054 [0.041, 0.067] | 0.048 [0.036, 0.060] | 0.045 [0.033, 0.056] |
| mixed | 0.018 [0.010, 0.025] | 0.042 [0.030, 0.053] | 0.042 [0.030, 0.053] | 0.637 [0.515, 0.761] | 0.224 [0.175, 0.274] | 0.026 [0.016, 0.035] | 0.034 [0.024, 0.044] | 0.028 [0.018, 0.037] |

**Set arms over 3 training seeds, visibility `none`**: mean of seed means (min, max); paired regret difference of seed 0 against oracle_clearing and extremized [95% CI].

| env | train | regret 3 seeds | minus oracle_clearing | minus extremized | untrained ctl | random-net ctl |
|---|---|---|---|---|---|---|
| sybil0 | clean | 0.042 (0.042, 0.042) | 0.026 [0.016, 0.035] | -0.003 [-0.006, 0.001] | 0.45 | 0.28 |
| sybil0 | pooled | 0.046 (0.045, 0.046) | 0.029 [0.020, 0.038] | 0.001 [-0.002, 0.004] | 0.45 | 0.40 |
| sybil1 | clean | 0.045 (0.045, 0.046) | 0.028 [0.018, 0.038] | -0.006 [-0.010, -0.002] | 0.54 | 0.32 |
| sybil1 | matched | 0.042 (0.042, 0.042) | 0.025 [0.016, 0.035] | -0.009 [-0.015, -0.003] | 0.54 | 0.40 |
| sybil1 | pooled | 0.042 (0.042, 0.043) | 0.026 [0.016, 0.035] | -0.009 [-0.014, -0.003] | 0.54 | 0.41 |
| sybil3 | clean | 0.060 (0.060, 0.060) | 0.044 [0.031, 0.056] | -0.018 [-0.024, -0.011] | 0.79 | 0.33 |
| sybil3 | matched | 0.042 (0.042, 0.042) | 0.026 [0.016, 0.035] | -0.036 [-0.046, -0.025] | 0.79 | 0.39 |
| sybil3 | pooled | 0.043 (0.043, 0.044) | 0.027 [0.018, 0.036] | -0.034 [-0.045, -0.023] | 0.79 | 0.47 |
| sybil7 | clean | 0.101 (0.099, 0.104) | 0.082 [0.063, 0.101] | -0.021 [-0.031, -0.009] | 1.35 | 0.38 |
| sybil7 | matched | 0.042 (0.042, 0.043) | 0.026 [0.016, 0.036] | -0.077 [-0.092, -0.061] | 1.35 | 0.47 |
| sybil7 | pooled | 0.043 (0.042, 0.043) | 0.026 [0.017, 0.036] | -0.076 [-0.091, -0.061] | 1.35 | 0.65 |
| collude3 | clean | 0.041 (0.040, 0.041) | 0.023 [0.012, 0.034] | 0.010 [0.000, 0.020] | 0.34 | 0.18 |
| collude3 | matched | 0.026 (0.026, 0.026) | 0.009 [0.003, 0.015] | -0.004 [-0.007, -0.001] | 0.34 | 0.24 |
| collude3 | pooled | 0.028 (0.027, 0.028) | 0.010 [0.004, 0.017] | -0.002 [-0.005, 0.000] | 0.34 | 0.33 |
| biased1 | clean | 0.228 (0.214, 0.247) | 0.204 [0.163, 0.248] | 0.144 [0.112, 0.179] | 0.60 | 0.30 |
| biased1 | matched | 0.050 (0.050, 0.051) | 0.034 [0.024, 0.044] | -0.026 [-0.035, -0.017] | 0.60 | 0.34 |
| biased1 | pooled | 0.056 (0.055, 0.057) | 0.039 [0.027, 0.050] | -0.021 [-0.030, -0.012] | 0.60 | 0.36 |
| mixed | clean | 0.143 (0.136, 0.153) | 0.121 [0.090, 0.153] | 0.096 [0.070, 0.125] | 0.76 | 0.27 |
| mixed | matched | 0.026 (0.025, 0.026) | 0.007 [0.001, 0.014] | -0.017 [-0.024, -0.009] | 0.76 | 0.39 |
| mixed | pooled | 0.028 (0.027, 0.029) | 0.012 [0.005, 0.018] | -0.013 [-0.020, -0.005] | 0.76 | 0.39 |

**Set arms over 3 training seeds, visibility `noisy`**: mean of seed means (min, max); paired regret difference of seed 0 against oracle_clearing and extremized [95% CI].

| env | train | regret 3 seeds | minus oracle_clearing | minus extremized | untrained ctl | random-net ctl |
|---|---|---|---|---|---|---|
| sybil0 | clean | 0.476 (0.441, 0.513) | 0.424 [0.358, 0.491] | 0.396 [0.330, 0.463] | 0.24 | 0.32 |
| sybil0 | pooled | 0.039 (0.039, 0.040) | 0.024 [0.015, 0.032] | -0.005 [-0.010, 0.001] | 0.24 | 0.19 |
| sybil1 | clean | 0.388 (0.244, 0.537) | 0.365 [0.302, 0.434] | 0.330 [0.267, 0.400] | 0.28 | 0.34 |
| sybil1 | matched | 0.215 (0.154, 0.311) | 0.163 [0.127, 0.200] | 0.128 [0.095, 0.165] | 0.28 | 0.36 |
| sybil1 | pooled | 0.040 (0.039, 0.040) | 0.024 [0.015, 0.032] | -0.011 [-0.017, -0.004] | 0.28 | 0.18 |
| sybil3 | clean | 0.350 (0.261, 0.494) | 0.244 [0.197, 0.293] | 0.183 [0.135, 0.231] | 0.37 | 0.42 |
| sybil3 | matched | 0.143 (0.116, 0.184) | 0.111 [0.083, 0.141] | 0.050 [0.021, 0.081] | 0.37 | 0.50 |
| sybil3 | pooled | 0.041 (0.040, 0.043) | 0.026 [0.017, 0.035] | -0.035 [-0.046, -0.023] | 0.37 | 0.19 |
| sybil7 | clean | 0.195 (0.156, 0.244) | 0.169 [0.133, 0.208] | 0.066 [0.028, 0.107] | 0.66 | 0.69 |
| sybil7 | matched | 0.118 (0.088, 0.159) | 0.091 [0.070, 0.112] | -0.012 [-0.036, 0.013] | 0.66 | 0.78 |
| sybil7 | pooled | 0.041 (0.039, 0.041) | 0.025 [0.015, 0.033] | -0.078 [-0.093, -0.063] | 0.66 | 0.27 |
| collude3 | clean | 0.598 (0.499, 0.704) | 0.481 [0.400, 0.563] | 0.469 [0.388, 0.550] | 0.31 | 0.35 |
| collude3 | matched | 0.261 (0.226, 0.298) | 0.209 [0.162, 0.257] | 0.196 [0.148, 0.245] | 0.31 | 0.35 |
| collude3 | pooled | 0.026 (0.026, 0.027) | 0.010 [0.004, 0.015] | -0.003 [-0.007, 0.001] | 0.31 | 0.13 |
| biased1 | clean | 1.161 (0.997, 1.253) | 0.980 [0.821, 1.144] | 0.921 [0.766, 1.076] | 0.24 | 0.55 |
| biased1 | matched | 0.466 (0.443, 0.483) | 0.427 [0.355, 0.500] | 0.367 [0.296, 0.437] | 0.24 | 0.51 |
| biased1 | pooled | 0.054 (0.052, 0.056) | 0.038 [0.027, 0.049] | -0.022 [-0.032, -0.011] | 0.24 | 0.19 |
| mixed | clean | 0.719 (0.589, 0.930) | 0.619 [0.499, 0.739] | 0.595 [0.477, 0.717] | 0.34 | 0.54 |
| mixed | matched | 0.214 (0.176, 0.242) | 0.206 [0.158, 0.256] | 0.182 [0.135, 0.233] | 0.34 | 0.48 |
| mixed | pooled | 0.025 (0.025, 0.026) | 0.009 [0.002, 0.015] | -0.016 [-0.024, -0.007] | 0.34 | 0.16 |

**Follow-up, equal data, visibility `noisy`**: clean and matched training with the pooled arm's data size (280,000 train episodes, 8000 steps; seed 0), next to the pooled arm (seed 0). Regret [95% CI]; paired differences [95% CI]; control regrets.

| env | train | regret | minus oracle_clearing | minus extremized | untrained ctl | random-net ctl |
|---|---|---|---|---|---|---|
| sybil0 | clean-280k | 0.039 [0.029, 0.051] | 0.023 [0.014, 0.032] | -0.005 [-0.010, 0.000] | 0.24 | 0.22 |
| sybil0 | pooled | 0.040 [0.029, 0.050] | 0.024 [0.015, 0.032] | -0.005 [-0.010, 0.001] | 0.24 | 0.19 |
| sybil1 | clean-280k | 0.042 [0.031, 0.053] | 0.025 [0.016, 0.034] | -0.009 [-0.015, -0.003] | 0.28 | 0.22 |
| sybil1 | matched-280k | 0.039 [0.028, 0.050] | 0.023 [0.014, 0.032] | -0.012 [-0.018, -0.005] | 0.28 | 0.22 |
| sybil1 | pooled | 0.040 [0.029, 0.051] | 0.024 [0.015, 0.032] | -0.011 [-0.017, -0.004] | 0.28 | 0.18 |
| sybil3 | clean-280k | 0.043 [0.030, 0.056] | 0.026 [0.016, 0.036] | -0.035 [-0.044, -0.025] | 0.37 | 0.23 |
| sybil3 | matched-280k | 0.039 [0.028, 0.050] | 0.023 [0.013, 0.031] | -0.039 [-0.050, -0.027] | 0.37 | 0.24 |
| sybil3 | pooled | 0.043 [0.032, 0.054] | 0.026 [0.017, 0.035] | -0.035 [-0.046, -0.023] | 0.37 | 0.19 |
| sybil7 | clean-280k | 0.048 [0.035, 0.061] | 0.031 [0.020, 0.043] | -0.071 [-0.086, -0.056] | 0.66 | 0.31 |
| sybil7 | matched-280k | 0.039 [0.027, 0.049] | 0.022 [0.013, 0.031] | -0.081 [-0.096, -0.065] | 0.66 | 0.34 |
| sybil7 | pooled | 0.041 [0.030, 0.052] | 0.025 [0.015, 0.033] | -0.078 [-0.093, -0.063] | 0.66 | 0.27 |
| collude3 | clean-280k | 0.031 [0.020, 0.042] | 0.013 [0.005, 0.021] | 0.000 [-0.007, 0.008] | 0.31 | 0.16 |
| collude3 | matched-280k | 0.025 [0.016, 0.034] | 0.008 [0.002, 0.013] | -0.005 [-0.009, -0.000] | 0.31 | 0.18 |
| collude3 | pooled | 0.027 [0.018, 0.036] | 0.010 [0.004, 0.015] | -0.003 [-0.007, 0.001] | 0.31 | 0.13 |
| biased1 | clean-280k | 0.165 [0.132, 0.200] | 0.148 [0.118, 0.181] | 0.089 [0.066, 0.113] | 0.24 | 0.24 |
| biased1 | matched-280k | 0.051 [0.039, 0.063] | 0.034 [0.025, 0.044] | -0.026 [-0.036, -0.015] | 0.24 | 0.24 |
| biased1 | pooled | 0.054 [0.041, 0.067] | 0.038 [0.027, 0.049] | -0.022 [-0.032, -0.011] | 0.24 | 0.19 |
| mixed | clean-280k | 0.129 [0.097, 0.163] | 0.111 [0.081, 0.144] | 0.087 [0.060, 0.116] | 0.34 | 0.20 |
| mixed | matched-280k | 0.024 [0.015, 0.034] | 0.007 [0.001, 0.013] | -0.017 [-0.026, -0.009] | 0.34 | 0.21 |
| mixed | pooled | 0.026 [0.016, 0.035] | 0.009 [0.002, 0.015] | -0.016 [-0.024, -0.007] | 0.34 | 0.16 |

**Market and reference arms**: paired regret difference against extremized [95% CI], visibility `none` (these arms ignore keys); controls are the regret of the untrained and random-weight versions.

| env | slot wealth (eta) | REF identity wealth (eta) | REF identity log pool | ctl (untrained / random), slot | ctl, identity log pool |
|---|---|---|---|---|---|
| sybil0 | -0.000 [-0.000, 0.000] (0) | -0.000 [-0.000, 0.000] (0) | 0.000 [-0.000, 0.001] | 0.112 / 0.131 | 0.112 / 0.045 |
| sybil1 | 0.000 [-0.000, 0.000] (0) | 0.000 [-0.001, 0.001] (0.001) | -0.006 [-0.011, -0.001] | 0.115 / 0.281 | 0.115 / 0.047 |
| sybil3 | 0.000 [-0.000, 0.001] (0.001) | -0.000 [-0.001, 0.001] (0.001) | -0.033 [-0.043, -0.023] | 0.126 / 0.219 | 0.126 / 0.055 |
| sybil7 | 0.000 [-0.000, 0.001] (0.001) | -0.008 [-0.011, -0.006] (0.01) | -0.075 [-0.089, -0.060] | 0.146 / 0.156 | 0.146 / 0.065 |
| collude3 | -0.000 [-0.000, 0.000] (0) | -0.000 [-0.000, 0.000] (0) | -0.003 [-0.005, -0.000] | 0.046 / 0.051 | 0.046 / 0.031 |
| biased1 | 0.000 [-0.000, 0.000] (0) | -0.028 [-0.037, -0.019] (0.0001) | -0.032 [-0.041, -0.022] | 0.091 / 0.096 | 0.091 / 0.102 |
| mixed | 0.000 [-0.000, 0.000] (0) | -0.007 [-0.015, 0.001] (0.0001) | -0.014 [-0.021, -0.007] | 0.054 / 0.081 | 0.054 / 0.035 |

**Does the set network count duplicated content once?** Total weight w on the duplicated group (Sybil source + copies) divided by the mean weight per honest agent, seed 0 (1 = counted once, 1 + n_sybil = counted per copy); and the identity log pool's fitted weights for comparison. Diagnostic only: roles come from the oracle after prediction.

| env | vis | set clean | set matched | set pooled | REF identity log pool |
|---|---|---|---|---|---|
| sybil1 | none | 1.92 | 1.43 | 1.21 | 0.98 |
| sybil1 | noisy | 1.53 | 1.22 | 1.27 | 0.98 |
| sybil3 | none | 3.69 | 1.73 | 1.19 | 0.98 |
| sybil3 | noisy | 1.62 | 1.45 | 1.41 | 0.98 |
| sybil7 | none | 7.41 | 3.09 | 1.46 | 0.98 |
| sybil7 | noisy | 2.23 | 1.78 | 1.82 | 0.98 |
<!-- TABLES:END -->

## Reading

- **Sybil copies, provenance hidden.** Exact copies send identical reports, and the network sees the pairwise report differences. Trained with adversaries, it counts the duplicated group about once: total weight on the source and its copies is 1.2-1.5 honest-agent weights (pooled), against 1.9-7.4 for the clean-trained network. Regret is then flat in the number of copies, at 0.042-0.043 for 1, 3 and 7 copies. That equals the no-adversary level (0.042), so Sybils cost nothing extra. The remaining 0.026 nats to oracle_clearing is the ordinary E2 gap (also 0.026 at sybil0), not an adversary cost. Trained without adversaries, the network gets 0.045 / 0.060 / 0.101 for 1 / 3 / 7 copies (3-seed means). That beats extremized_log_pool but grows with the number of copies.
- **Colluders.** Colluders also send identical reports, of the pooled posterior. The matched network reaches 0.026, which is 0.009 [0.003, 0.015] above oracle_clearing and 0.004 below extremized. Here the adversary costs little in any arm, because the colluders' pooled report is well informed.
- **Biased source.** It is found only partly from content: matched 0.050 and pooled 0.056 (3-seed means), 0.034-0.039 above oracle_clearing. The pooled network gives the biased agent weight 0.33 against 0.39 for an honest agent, while the identity reference gives it about 1/3 of an honest agent's weight. A network trained without adversaries is badly hurt by the biased source (0.228 `none`, 0.165 `noisy` at 280k), worse than extremized_log_pool (0.076).
- **All three at once (mixed).** Matched 0.025 and pooled 0.029 (`none`), against extremized 0.042 and oracle_clearing 0.018.
- **Noisy provenance.** With enough data (280k follow-up), noisy coverage keys help slightly: matched 0.039 against 0.042 at `none`. With keys, even clean training handles Sybils well (0.042 / 0.043 / 0.048), but not the biased source (0.165) or the mixed case (0.129). So adversarial training matters most for the biased source. At 40k episodes, clean and matched training on noisy keys overfit; see the follow-up note under Arms.
- **Market ("traders") baseline.** Without persistent identities, slot wealth cannot accumulate: the chosen eta is 0 or 0.001, and the arm equals extremized_log_pool to within 0.001 nats. With oracle identities (REF), wealth discounts the biased source (-0.028 against extremized) but not Sybil copies (0.000 at 1 and 3 copies, -0.008 at 7). Each copy has the same log score as its source, so a wealth market pays every copy in full. Per-source weights fitted for the aggregate's log loss (REF identity log pool) do discount the copies: the duplicated group's weight is 0.98 honest-agent weights, and regret is 0.045 on every Sybil environment. The content-only network matches or beats this identity reference (0.042-0.043 against 0.045 on Sybils) with no identities at all. The biased source is the exception (0.050 against 0.045).
- **Controls.** All untrained-init controls of the set arms score 0.24 or worse. The random-weight control (`aggbench.controls`, which keeps each tensor's mean and sd) is a weaker bar here. The readout l0 + sum_i w_i (l_i - l0) turns a random network into a rough log pool, so some seeds' random controls land at 0.035-0.10 (seed 1 at `none`, for example). The Sybil claims survive this control: on sybil3 and sybil7 every random control scores 0.092 or worse, against 0.042 for the trained arm. The biased-source claim is weaker. Pooled seed 1 at `none` scores 0.056, but its random control (0.071) also beats extremized_log_pool (0.076). For the REF identity log pool, the random control is close to the trained arm wherever the fitted weights are near-uniform (sybil0 0.045 against 0.045, collude3 0.031 against 0.028), and clearly worse where it discounts (sybil7 0.065, biased1 0.102, against 0.045). Clean-trained arms on biased1 and mixed are worse than some of their own random controls, which is further evidence that clean training fails there. All control values are in the JSONs.
- **Limits.** Exploratory, dev split only, one harness configuration. The Sybils are exact copies and the colluders report identical values, so "detection" here means finding identical reports. Near-duplicates with noise were not tested. Set arms use 3 training seeds (the 280k follow-up uses 1). CIs are bootstrap over episodes and do not include seed variance.

## Reproduce

```bash
# VM stage (run on a spot L4 via run_on_l4.sh at commit 12b73d7; 6 processes = 2 visibilities x 3 seeds, plus the markets)
python run_l5.py train --out /root/runs/l5-sybil-collusion-20261020 --vis {none,noisy} --seed {0,1,2} --device cuda
python run_l5.py market --out /root/runs/l5-sybil-collusion-20261020
# follow-up (run_on_l4 at commit 33c8e92): 4 processes, configs split across them
python run_l5.py train --out /root/runs/l5-sybil-big-20261020/g<i> --vis noisy --seed 0 --device cuda --n-train 280000 --steps 8000 --only <configs>
# local stage (numpy + jsonschema; regenerates train/dev batches, baselines, oracle check, JSON)
PYTHONPATH=. python3 results/explore/l5-sybil-collusion-20261020/run_l5.py assemble --inp <synced runs dir> \
  --vm-commit 12b73d7 --vm-run-id l5-sybil-collusion-20261020 --big-inp <merged follow-up npz dir>
PYTHONPATH=. python3 results/explore/l5-sybil-collusion-20261020/run_l5.py tables
```

Harness versions: the VM runs used `aggbench` at 12b73d7 and 33c8e92, and the local assembly used it at 22de117 (recorded in the JSONs). The harness commits between them add E4, the structure zoo, `agent_inputs` alignment and a GPU fix in `controls`. On 300 dev episodes of each adversarial environment, sybil7, collude3, biased1 and mixed give bit-identical reports, noisy keys, Bayes posteriors and roles at both commits.

Compute: spot L4 (g2-standard-8). Main run 0.82 h; follow-up 0.47 h; a preempted follow-up attempt about 0.08 h. Total about 1.4 L4-hours, $1.19 at the list-price ceiling (GPU-LOG). Both VMs were launched through a copy of `run_on_l4.sh` that adds `--run-id`: the stock script fails with 'Use --run-id when other soup VMs exist' while other lanes are running. The per-episode forecasts (npz) are archived at `/home/agent/data/archive/l5-sybil-collusion-20261020/` and are not in git.
