Serves: one mechanism that decides what to look at next (compute, data or memory), by value of information

# EXPLORATORY: L4, budgeted consultation by learned queries vs exact-Bayes greedy VoI (`l4-query-voi-20261020`)

**Track E (PROTOCOL 10.1). Exploratory: dev split only, test split never touched, no audit. These are not findings.**

## Question

Each agent's KEY (a description of what it saw) is free to read, but its REPORT costs one consultation. With a budget of b < n consultations, can an attention aggregator choose whom to consult next and then aggregate what it bought, and get close to the exact-Bayes greedy value-of-information (VoI) policy? Its tool is a QUERY computed from the consulted set ("what the consensus still needs") that attends over the keys of the agents not yet consulted. Does it beat random or fixed-order consultation? Regret is measured to Bayes (full information) and to the best-b-subset ceiling as b varies, in E1, E2 and E3-sybil. In E3-sybil a good selector should skip duplicates.

## Setup

aggbench unmodified (harness commit and sha256 are in every JSON), default env configs, visibility `full`, train split 40,000 episodes, dev split 2,000 episodes (seeds 10,000,000+). E1 and E2 have N = 5 agents; E3-sybil has 5 + 3 exact copies of one agent, so N = 8. Each b from 1 to N-1 is evaluated.

**Arms.** Each arm is a selection policy plus an aggregation rule.
- **Exact aggregation** is the exact posterior given the consulted reports and the structure read from the full-visibility keys: coverage and piece strengths in E2/E3, and A, B, sigma in E1. It is a function of the view. With all agents consulted it matches the harness `oracle_clearing` to 0 (E2/E3) and 3e-15 (E1); the check is `view_exact_vs_oracle_clearing_max_abs` in each JSON.
- **info(S).** In E2/E3, q_S = m^T C_S^T (C_S M C_S^T)^+ C_S m. Given Y, the posterior log-odds from S's reports is N(+-q_S, 2 q_S), so a higher q_S is a Blackwell-more-informative experiment. In E1, info(S) is log det of the posterior precision. In both families the expected regret after consulting S depends only on info(S) and is decreasing in it. The best fixed b-subset, `best_subset` (enumerated), is therefore the Bayes-optimal b-step policy in expectation: adaptivity cannot help, because the best completion of any partial set does not depend on the values observed so far. `greedy_voi` takes argmax_j info(S + j) one step at a time.
- **Non-learned selectors:** `random` (a random order, seed 7); `fixed` (index order; the env shuffles agents, so this is also random in distribution); `static_voi` (ranks agents by their stand-alone info and ignores overlap; a zero-parameter readout of the keys); `greedy_voi`; `best_subset` (the ceiling); `hindsight_best` (per episode, the b-subset closest to the realized Bayes posterior). `hindsight_best` uses `bayes`, is unachievable, and is shown for scale only.
- **Learned (QueryNet, `l4_query_voi.py`):**
  - The encoder maps each agent's key to e_j. The consensus state is an MLP of the sum over consulted agents of phi(e_j), plus the step and the remaining budget. The query is W_q(state); each candidate's score is a 4-head dot product with W_k e_j, and agents already consulted are masked.
  - The aggregator runs masked attention among the consulted agents and gives each agent a positive gate. In E2/E3 the output is l0 + sum_j g_j (r_j - l0). In E1 it is precision I + sum_j g_j (P_j - I) with information sum_j g_j P_j m_j.
  - Selection during training is straight-through Gumbel-softmax (tau 1 → 0.3), with budget b uniform on 1..N per minibatch. One network serves every budget. Settings: 6,000 Adam steps, batch 512, lr 2e-3 with cosine decay, width 64, seeds 0/1/2. These constants were set by hand before any run; none was tuned on results.
  - Three evaluation arms come from each network: `learned_query+learned` (the main arm), `learned_query+exact` (its selections with exact aggregation, so selection quality alone) and `random+learned` (random selections with its aggregator).
- **Training signal**, in three separate runs:
  - `truth`: log loss of the truth (main run, no distillation).
  - `distill`: KL from the train-split Bayes posterior. This is distillation and is declared as such (aggbench README rule 2).
  - `imitate`: truth loss plus the cross-entropy of each query choice against the greedy-VoI order. The teacher is computed from the train view's keys, with no oracle field, and is teacher-forced. Its sets have the same info(S) as `greedy_voi` on dev to within 2e-14 (`teacher_dev_max_info_diff`).
- **Controls (aggbench.controls).** Every learned arm in every JSON has `control:random_network` (`random_weights(trained, seed+1)`) and `control:untrained_init` (`untrained_init(make, seed)`). Both run through the identical `predict`, including the selection and the exact-aggregation variant. `run_with_controls` itself is not called, because one fitted network serves all budgets. Its two control functions are called with the same seeds instead.

**Metric.** Mean regret to the global Bayes posterior, in nats per episode (aggbench `summarize`, KL-equivalent), with 95% bootstrap CIs over 2,000 dev episodes. "Regret to the best-b-subset ceiling" is the paired difference arm − `best_subset+exact` on the same episodes (`metrics.paired_difference`). The "share of the gap closed" row is (random − arm) / (random − best_subset), computed from the JSON means.

## Tables

The tables below are generated from the JSONs by `make_tables.py`, which re-validates each JSON against `aggbench/result_schema.json`. No number was typed by hand.

### Training signal `truth` (main run, `truth/`)

### E1 (N = 5 consultable agents; 2000 dev episodes; `result-E1.json`, validated)

Mean regret to the global Bayes posterior (nats/episode, 95% bootstrap CI). Learned rows: seed 0.

| arm | b=1 | b=2 | b=3 | b=4 |
|---|---|---|---|---|
| random order, exact aggregation | 3.399 [3.316, 3.483] | 2.382 [2.308, 2.461] | 1.425 [1.360, 1.486] | 0.640 [0.599, 0.683] |
| index order, exact aggregation | 3.439 [3.358, 3.520] | 2.406 [2.332, 2.478] | 1.468 [1.409, 1.528] | 0.697 [0.654, 0.740] |
| stand-alone VoI ranking (ignores overlap), exact | 2.734 [2.658, 2.808] | 1.473 [1.413, 1.534] | 0.694 [0.650, 0.737] | 0.235 [0.205, 0.262] |
| exact-Bayes greedy VoI, exact | 2.734 [2.658, 2.808] | 1.412 [1.350, 1.471] | 0.556 [0.515, 0.599] | 0.179 [0.154, 0.204] |
| best b-subset in expectation (ceiling), exact | 2.734 [2.658, 2.808] | 1.400 [1.339, 1.458] | 0.543 [0.502, 0.587] | 0.180 [0.155, 0.205] |
| hindsight-best b-subset (unachievable), exact | 2.402 [2.334, 2.466] | 1.099 [1.050, 1.148] | 0.399 [0.367, 0.431] | 0.099 [0.081, 0.116] |
| learned query selection, exact aggregation | 2.789 [2.709, 2.865] | 1.490 [1.426, 1.550] | 0.668 [0.622, 0.713] | 0.220 [0.193, 0.247] |
| **learned query + learned aggregation (main arm)** | 2.791 [2.711, 2.868] | 1.840 [1.769, 1.911] | 1.289 [1.224, 1.358] | 0.979 [0.920, 1.039] |
| random selection + learned aggregation | 3.400 [3.317, 3.485] | 2.579 [2.500, 2.658] | 1.870 [1.797, 1.942] | 1.296 [1.233, 1.359] |
| control: random network (main arm pipeline) | 3.402 [3.322, 3.481] | 2.564 [2.491, 2.634] | 1.918 [1.849, 1.983] | 1.387 [1.326, 1.445] |
| control: untrained init (main arm pipeline) | 3.291 [3.207, 3.368] | 2.453 [2.378, 2.527] | 1.841 [1.773, 1.906] | 1.343 [1.287, 1.400] |
| learned query + learned, all seeds (means) | 2.791 / 2.802 / 2.793 | 1.840 / 1.829 / 1.821 | 1.289 / 1.280 / 1.286 | 0.979 / 0.987 / 0.990 |
| learned query + exact, all seeds (means) | 2.789 / 2.800 / 2.792 | 1.490 / 1.486 / 1.480 | 0.668 / 0.655 / 0.664 | 0.220 / 0.227 / 0.231 |
| share of the random → best-subset gap closed by learned selection (exact aggregation), per seed | 0.92 / 0.90 / 0.91 | 0.91 / 0.91 / 0.92 | 0.86 / 0.87 / 0.86 | 0.91 / 0.90 / 0.89 |
| same share for static VoI / greedy VoI | 1.00 / 1.00 | 0.92 / 0.99 | 0.83 / 0.99 | 0.88 / 1.00 |

Full information (b = N = 5): oracle_clearing -0.000 [-0.000, 0.000], extremized_log_pool 0.828 [0.770, 0.884], learned aggregator on all agents (s0) 0.830 [0.772, 0.886], its random-network control 0.909 [0.858, 0.960].

Regret to the best-b-subset ceiling: paired difference arm - best_subset+exact (95% CI).

| arm | b=1 | b=2 | b=3 | b=4 |
|---|---|---|---|---|
| random+exact | +0.665 [+0.605, +0.727] | +0.982 [+0.912, +1.053] | +0.881 [+0.814, +0.946] | +0.461 [+0.414, +0.509] |
| static_voi+exact | +0.000 [+0.000, +0.000] | +0.074 [+0.038, +0.106] | +0.151 [+0.114, +0.184] | +0.055 [+0.026, +0.081] |
| greedy_voi+exact | +0.000 [+0.000, +0.000] | +0.012 [-0.004, +0.028] | +0.013 [+0.000, +0.026] | -0.001 [-0.008, +0.005] |
| learned_query+exact:s0 | +0.055 [+0.027, +0.084] | +0.090 [+0.053, +0.129] | +0.124 [+0.088, +0.156] | +0.040 [+0.014, +0.063] |
| learned_query+learned:s0 | +0.057 [+0.029, +0.086] | +0.440 [+0.393, +0.488] | +0.746 [+0.688, +0.801] | +0.799 [+0.741, +0.857] |
| random+learned:s0 | +0.666 [+0.607, +0.728] | +1.179 [+1.106, +1.247] | +1.327 [+1.251, +1.398] | +1.116 [+1.050, +1.181] |
| learned_query+learned:s0:control:random_network | +0.668 [+0.611, +0.727] | +1.165 [+1.097, +1.229] | +1.374 [+1.309, +1.440] | +1.207 [+1.146, +1.268] |

Value of learned selection inside the learned aggregator: learned_query+learned - random+learned (paired, 95% CI), per seed.

| seed | b=1 | b=2 | b=3 | b=4 |
|---|---|---|---|---|
| s0 | -0.609 [-0.675, -0.546] | -0.738 [-0.805, -0.671] | -0.581 [-0.638, -0.523] | -0.317 [-0.363, -0.275] |
| s0:control:random_network | -0.033 [-0.085, +0.021] | -0.064 [-0.121, -0.007] | -0.017 [-0.073, +0.037] | +0.012 [-0.029, +0.053] |
| s1 | -0.598 [-0.664, -0.533] | -0.751 [-0.818, -0.685] | -0.591 [-0.647, -0.533] | -0.311 [-0.355, -0.269] |
| s1:control:random_network | -0.106 [-0.156, -0.055] | -0.156 [-0.212, -0.100] | -0.114 [-0.164, -0.062] | -0.039 [-0.076, -0.002] |
| s2 | -0.608 [-0.672, -0.546] | -0.758 [-0.825, -0.692] | -0.585 [-0.643, -0.526] | -0.307 [-0.351, -0.265] |
| s2:control:random_network | +0.035 [-0.016, +0.083] | +0.058 [+0.003, +0.112] | +0.081 [+0.028, +0.133] | +0.065 [+0.028, +0.103] |

Selection diagnostics: share of episodes whose chosen set equals the best b-subset; share holding two agents with identical keys (a Sybil duplicate).

| policy | b=1 same / dup | b=2 same / dup | b=3 same / dup | b=4 same / dup |
|---|---|---|---|---|
| random | 0.21 / 0.00 | 0.10 / 0.00 | 0.10 / 0.00 | 0.20 / 0.00 |
| static_voi | 1.00 / 0.00 | 0.64 / 0.00 | 0.53 / 0.00 | 0.59 / 0.00 |
| greedy_voi | 1.00 / 0.00 | 0.94 / 0.00 | 0.94 / 0.00 | 0.98 / 0.00 |
| best_subset | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| learned_query:s0 | 0.77 / 0.00 | 0.64 / 0.00 | 0.57 / 0.00 | 0.63 / 0.00 |
| learned_query:s0:control:random_network | 0.24 / 0.00 | 0.12 / 0.00 | 0.12 / 0.00 | 0.21 / 0.00 |
| learned_query:s0:control:untrained_init | 0.29 / 0.00 | 0.19 / 0.00 | 0.17 / 0.00 | 0.28 / 0.00 |

### E2 (N = 5 consultable agents; 2000 dev episodes; `result-E2.json`, validated)

Mean regret to the global Bayes posterior (nats/episode, 95% bootstrap CI). Learned rows: seed 0.

| arm | b=1 | b=2 | b=3 | b=4 |
|---|---|---|---|---|
| random order, exact aggregation | 0.224 [0.203, 0.245] | 0.132 [0.115, 0.151] | 0.078 [0.064, 0.092] | 0.038 [0.027, 0.049] |
| index order, exact aggregation | 0.214 [0.193, 0.234] | 0.117 [0.099, 0.134] | 0.069 [0.055, 0.084] | 0.039 [0.028, 0.049] |
| stand-alone VoI ranking (ignores overlap), exact | 0.110 [0.092, 0.128] | 0.062 [0.047, 0.076] | 0.044 [0.033, 0.054] | 0.026 [0.018, 0.035] |
| exact-Bayes greedy VoI, exact | 0.110 [0.092, 0.128] | 0.049 [0.036, 0.061] | 0.025 [0.016, 0.034] | 0.021 [0.013, 0.029] |
| best b-subset in expectation (ceiling), exact | 0.110 [0.092, 0.128] | 0.044 [0.032, 0.055] | 0.024 [0.015, 0.033] | 0.020 [0.012, 0.027] |
| hindsight-best b-subset (unachievable), exact | 0.017 [0.012, 0.023] | 0.005 [0.002, 0.007] | 0.002 [-0.001, 0.005] | 0.004 [0.000, 0.008] |
| learned query selection, exact aggregation | 0.144 [0.125, 0.163] | 0.074 [0.060, 0.088] | 0.049 [0.037, 0.059] | 0.035 [0.025, 0.044] |
| **learned query + learned aggregation (main arm)** | 0.160 [0.142, 0.177] | 0.084 [0.069, 0.098] | 0.061 [0.048, 0.074] | 0.049 [0.036, 0.062] |
| random selection + learned aggregation | 0.236 [0.217, 0.255] | 0.141 [0.124, 0.159] | 0.094 [0.079, 0.110] | 0.062 [0.048, 0.076] |
| control: random network (main arm pipeline) | 0.181 [0.161, 0.199] | 0.107 [0.089, 0.125] | 0.082 [0.064, 0.101] | 0.071 [0.052, 0.091] |
| control: untrained init (main arm pipeline) | 0.201 [0.182, 0.219] | 0.129 [0.113, 0.146] | 0.091 [0.073, 0.107] | 0.063 [0.045, 0.081] |
| learned query + learned, all seeds (means) | 0.160 / 0.153 / 0.170 | 0.084 / 0.075 / 0.090 | 0.061 / 0.056 / 0.069 | 0.049 / 0.050 / 0.062 |
| learned query + exact, all seeds (means) | 0.144 / 0.131 / 0.152 | 0.074 / 0.067 / 0.083 | 0.049 / 0.048 / 0.056 | 0.035 / 0.032 / 0.034 |
| share of the random → best-subset gap closed by learned selection (exact aggregation), per seed | 0.70 / 0.81 / 0.63 | 0.66 / 0.74 / 0.56 | 0.54 / 0.55 / 0.40 | 0.20 / 0.34 / 0.21 |
| same share for static VoI / greedy VoI | 1.00 / 1.00 | 0.80 / 0.94 | 0.63 / 0.98 | 0.64 / 0.90 |

Full information (b = N = 5): oracle_clearing 0.017 [0.009, 0.024], extremized_log_pool 0.045 [0.033, 0.056], learned aggregator on all agents (s0) 0.044 [0.031, 0.058], its random-network control 0.077 [0.055, 0.100].

Regret to the best-b-subset ceiling: paired difference arm - best_subset+exact (95% CI).

| arm | b=1 | b=2 | b=3 | b=4 |
|---|---|---|---|---|
| random+exact | +0.114 [+0.092, +0.137] | +0.088 [+0.070, +0.107] | +0.055 [+0.042, +0.068] | +0.019 [+0.011, +0.027] |
| static_voi+exact | +0.000 [+0.000, +0.000] | +0.018 [+0.006, +0.030] | +0.020 [+0.011, +0.029] | +0.007 [+0.001, +0.013] |
| greedy_voi+exact | +0.000 [+0.000, +0.000] | +0.005 [-0.003, +0.013] | +0.001 [-0.004, +0.006] | +0.002 [-0.001, +0.005] |
| learned_query+exact:s0 | +0.034 [+0.016, +0.053] | +0.030 [+0.016, +0.045] | +0.025 [+0.015, +0.034] | +0.015 [+0.008, +0.022] |
| learned_query+learned:s0 | +0.049 [+0.032, +0.066] | +0.040 [+0.026, +0.055] | +0.038 [+0.026, +0.050] | +0.030 [+0.019, +0.042] |
| random+learned:s0 | +0.126 [+0.106, +0.146] | +0.097 [+0.079, +0.115] | +0.071 [+0.057, +0.087] | +0.043 [+0.031, +0.055] |
| learned_query+learned:s0:control:random_network | +0.070 [+0.051, +0.090] | +0.063 [+0.046, +0.081] | +0.059 [+0.042, +0.076] | +0.052 [+0.035, +0.070] |

Value of learned selection inside the learned aggregator: learned_query+learned - random+learned (paired, 95% CI), per seed.

| seed | b=1 | b=2 | b=3 | b=4 |
|---|---|---|---|---|
| s0 | -0.077 [-0.094, -0.060] | -0.058 [-0.075, -0.041] | -0.033 [-0.048, -0.020] | -0.013 [-0.023, -0.003] |
| s0:control:random_network | -0.051 [-0.069, -0.033] | -0.030 [-0.051, -0.009] | -0.019 [-0.039, +0.001] | -0.012 [-0.026, +0.002] |
| s1 | -0.096 [-0.111, -0.081] | -0.076 [-0.091, -0.060] | -0.046 [-0.060, -0.032] | -0.020 [-0.030, -0.009] |
| s1:control:random_network | -0.052 [-0.067, -0.036] | -0.028 [-0.046, -0.010] | -0.015 [-0.032, +0.002] | -0.005 [-0.017, +0.006] |
| s2 | -0.076 [-0.092, -0.061] | -0.059 [-0.075, -0.042] | -0.032 [-0.048, -0.015] | -0.005 [-0.017, +0.007] |
| s2:control:random_network | -0.002 [-0.018, +0.014] | +0.008 [-0.011, +0.027] | +0.002 [-0.015, +0.018] | -0.000 [-0.013, +0.012] |

Selection diagnostics: share of episodes whose chosen set equals the best b-subset; share holding two agents with identical keys (a Sybil duplicate).

| policy | b=1 same / dup | b=2 same / dup | b=3 same / dup | b=4 same / dup |
|---|---|---|---|---|
| random | 0.20 / 0.00 | 0.10 / 0.00 | 0.10 / 0.00 | 0.21 / 0.00 |
| static_voi | 1.00 / 0.00 | 0.31 / 0.00 | 0.22 / 0.00 | 0.30 / 0.00 |
| greedy_voi | 1.00 / 0.00 | 0.79 / 0.00 | 0.76 / 0.00 | 0.84 / 0.00 |
| best_subset | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| learned_query:s0 | 0.56 / 0.00 | 0.26 / 0.00 | 0.21 / 0.00 | 0.28 / 0.01 |
| learned_query:s0:control:random_network | 0.36 / 0.00 | 0.16 / 0.00 | 0.11 / 0.00 | 0.23 / 0.01 |
| learned_query:s0:control:untrained_init | 0.32 / 0.00 | 0.11 / 0.00 | 0.10 / 0.00 | 0.21 / 0.01 |

### E3-sybil (N = 8 consultable agents; 2000 dev episodes; `result-E3-sybil.json`, validated)

Mean regret to the global Bayes posterior (nats/episode, 95% bootstrap CI). Learned rows: seed 0.

| arm | b=1 | b=2 | b=3 | b=4 | b=5 | b=6 | b=7 |
|---|---|---|---|---|---|---|---|
| random order, exact aggregation | 0.207 [0.187, 0.226] | 0.130 [0.111, 0.148] | 0.094 [0.078, 0.110] | 0.068 [0.054, 0.081] | 0.048 [0.036, 0.060] | 0.038 [0.028, 0.049] | 0.026 [0.017, 0.035] |
| index order, exact aggregation | 0.220 [0.198, 0.240] | 0.146 [0.126, 0.163] | 0.110 [0.093, 0.126] | 0.082 [0.067, 0.097] | 0.054 [0.042, 0.067] | 0.038 [0.027, 0.049] | 0.025 [0.016, 0.034] |
| stand-alone VoI ranking (ignores overlap), exact | 0.110 [0.092, 0.128] | 0.069 [0.053, 0.084] | 0.058 [0.046, 0.071] | 0.047 [0.034, 0.058] | 0.038 [0.027, 0.049] | 0.031 [0.021, 0.040] | 0.025 [0.016, 0.034] |
| exact-Bayes greedy VoI, exact | 0.110 [0.092, 0.128] | 0.049 [0.036, 0.061] | 0.025 [0.016, 0.034] | 0.021 [0.013, 0.029] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.024] |
| best b-subset in expectation (ceiling), exact | 0.110 [0.092, 0.128] | 0.044 [0.032, 0.055] | 0.024 [0.015, 0.033] | 0.020 [0.012, 0.027] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.024] |
| hindsight-best b-subset (unachievable), exact | 0.017 [0.012, 0.023] | 0.004 [0.001, 0.007] | 0.002 [-0.000, 0.005] | 0.002 [-0.001, 0.005] | 0.003 [-0.000, 0.006] | 0.003 [-0.000, 0.006] | 0.005 [0.001, 0.010] |
| learned query selection, exact aggregation | 0.142 [0.123, 0.161] | 0.098 [0.083, 0.113] | 0.076 [0.063, 0.090] | 0.067 [0.054, 0.081] | 0.051 [0.040, 0.063] | 0.040 [0.029, 0.050] | 0.027 [0.018, 0.035] |
| **learned query + learned aggregation (main arm)** | 0.214 [0.197, 0.231] | 0.131 [0.116, 0.146] | 0.101 [0.087, 0.116] | 0.096 [0.080, 0.112] | 0.096 [0.079, 0.114] | 0.093 [0.074, 0.111] | 0.090 [0.071, 0.109] |
| random selection + learned aggregation | 0.288 [0.269, 0.305] | 0.201 [0.184, 0.218] | 0.152 [0.137, 0.168] | 0.122 [0.106, 0.136] | 0.101 [0.085, 0.116] | 0.090 [0.074, 0.105] | 0.080 [0.064, 0.096] |
| control: random network (main arm pipeline) | 0.186 [0.166, 0.206] | 0.130 [0.110, 0.150] | 0.132 [0.108, 0.157] | 0.167 [0.135, 0.200] | 0.192 [0.154, 0.230] | 0.243 [0.197, 0.287] | 0.280 [0.229, 0.334] |
| control: untrained init (main arm pipeline) | 0.201 [0.181, 0.219] | 0.142 [0.123, 0.160] | 0.139 [0.115, 0.160] | 0.159 [0.130, 0.188] | 0.179 [0.146, 0.213] | 0.198 [0.159, 0.236] | 0.212 [0.170, 0.255] |
| learned query + learned, all seeds (means) | 0.214 / 0.263 / 0.215 | 0.131 / 0.185 / 0.133 | 0.101 / 0.151 / 0.102 | 0.096 / 0.146 / 0.098 | 0.096 / 0.137 / 0.101 | 0.093 / 0.126 / 0.098 | 0.090 / 0.111 / 0.094 |
| learned query + exact, all seeds (means) | 0.142 / 0.212 / 0.139 | 0.098 / 0.146 / 0.097 | 0.076 / 0.120 / 0.077 | 0.067 / 0.103 / 0.067 | 0.051 / 0.072 / 0.051 | 0.040 / 0.054 / 0.039 | 0.027 / 0.034 / 0.028 |
| share of the random → best-subset gap closed by learned selection (exact aggregation), per seed | 0.67 / -0.05 / 0.70 | 0.37 / -0.19 / 0.38 | 0.25 / -0.38 / 0.24 | 0.01 / -0.72 / 0.02 | -0.10 / -0.74 / -0.10 | -0.05 / -0.69 / -0.04 | -0.04 / -0.82 / -0.16 |
| same share for static VoI / greedy VoI | 1.00 / 1.00 | 0.71 / 0.94 | 0.50 / 0.98 | 0.44 / 0.96 | 0.32 / 1.00 | 0.35 / 1.00 | 0.11 / 1.00 |

Full information (b = N = 8): oracle_clearing 0.017 [0.009, 0.024], extremized_log_pool 0.078 [0.062, 0.092], learned aggregator on all agents (s0) 0.077 [0.060, 0.093], its random-network control 0.308 [0.252, 0.369].

Regret to the best-b-subset ceiling: paired difference arm - best_subset+exact (95% CI).

| arm | b=1 | b=2 | b=3 | b=4 | b=5 | b=6 | b=7 |
|---|---|---|---|---|---|---|---|
| random+exact | +0.097 [+0.074, +0.119] | +0.086 [+0.068, +0.104] | +0.070 [+0.056, +0.086] | +0.048 [+0.036, +0.061] | +0.032 [+0.022, +0.042] | +0.022 [+0.014, +0.030] | +0.010 [+0.004, +0.016] |
| static_voi+exact | +0.000 [+0.000, +0.000] | +0.025 [+0.012, +0.039] | +0.035 [+0.023, +0.047] | +0.027 [+0.017, +0.037] | +0.022 [+0.012, +0.030] | +0.014 [+0.007, +0.021] | +0.009 [+0.003, +0.014] |
| greedy_voi+exact | +0.000 [+0.000, +0.000] | +0.005 [-0.003, +0.013] | +0.001 [-0.004, +0.006] | +0.002 [-0.001, +0.005] | +0.000 [-0.000, +0.000] | +0.000 [-0.000, +0.000] | -0.000 [-0.000, +0.000] |
| learned_query+exact:s0 | +0.032 [+0.014, +0.051] | +0.054 [+0.039, +0.070] | +0.053 [+0.040, +0.066] | +0.048 [+0.036, +0.060] | +0.035 [+0.025, +0.044] | +0.023 [+0.015, +0.031] | +0.010 [+0.004, +0.016] |
| learned_query+learned:s0 | +0.104 [+0.087, +0.120] | +0.087 [+0.072, +0.102] | +0.078 [+0.063, +0.092] | +0.076 [+0.061, +0.092] | +0.080 [+0.064, +0.096] | +0.076 [+0.059, +0.093] | +0.074 [+0.056, +0.091] |
| random+learned:s0 | +0.178 [+0.158, +0.196] | +0.157 [+0.141, +0.174] | +0.129 [+0.114, +0.144] | +0.102 [+0.088, +0.116] | +0.084 [+0.070, +0.098] | +0.074 [+0.059, +0.087] | +0.064 [+0.049, +0.077] |
| learned_query+learned:s0:control:random_network | +0.076 [+0.056, +0.095] | +0.086 [+0.067, +0.106] | +0.109 [+0.085, +0.132] | +0.148 [+0.118, +0.178] | +0.175 [+0.140, +0.211] | +0.226 [+0.183, +0.269] | +0.263 [+0.214, +0.315] |

Value of learned selection inside the learned aggregator: learned_query+learned - random+learned (paired, 95% CI), per seed.

| seed | b=1 | b=2 | b=3 | b=4 | b=5 | b=6 | b=7 |
|---|---|---|---|---|---|---|---|
| s0 | -0.073 [-0.083, -0.063] | -0.070 [-0.082, -0.058] | -0.051 [-0.063, -0.039] | -0.026 [-0.038, -0.012] | -0.004 [-0.016, +0.008] | +0.003 [-0.008, +0.014] | +0.010 [+0.002, +0.018] |
| s0:control:random_network | -0.027 [-0.046, -0.007] | -0.010 [-0.033, +0.014] | -0.005 [-0.033, +0.022] | +0.009 [-0.023, +0.038] | +0.011 [-0.016, +0.038] | +0.018 [-0.006, +0.040] | +0.020 [+0.003, +0.038] |
| s1 | +0.011 [-0.003, +0.025] | +0.012 [-0.005, +0.029] | +0.009 [-0.010, +0.029] | +0.025 [+0.006, +0.045] | +0.026 [+0.006, +0.047] | +0.017 [-0.002, +0.036] | +0.018 [+0.003, +0.033] |
| s1:control:random_network | -0.036 [-0.050, -0.022] | -0.024 [-0.041, -0.006] | -0.013 [-0.031, +0.005] | -0.004 [-0.023, +0.014] | +0.016 [-0.001, +0.034] | +0.019 [+0.004, +0.034] | +0.018 [+0.006, +0.030] |
| s2 | -0.067 [-0.077, -0.056] | -0.061 [-0.073, -0.049] | -0.046 [-0.059, -0.034] | -0.023 [-0.036, -0.009] | -0.001 [-0.013, +0.012] | +0.007 [-0.004, +0.019] | +0.014 [+0.005, +0.023] |
| s2:control:random_network | +0.020 [+0.007, +0.035] | +0.033 [+0.015, +0.050] | +0.028 [+0.009, +0.046] | +0.024 [+0.004, +0.042] | +0.032 [+0.014, +0.050] | +0.028 [+0.011, +0.045] | +0.022 [+0.009, +0.034] |

Selection diagnostics: share of episodes whose chosen set equals the best b-subset; share holding two agents with identical keys (a Sybil duplicate).

| policy | b=1 same / dup | b=2 same / dup | b=3 same / dup | b=4 same / dup | b=5 same / dup | b=6 same / dup | b=7 same / dup |
|---|---|---|---|---|---|---|---|
| random | 0.12 / 0.00 | 0.03 / 0.23 | 0.02 / 0.52 | 0.02 / 0.77 | 0.01 / 0.93 | 0.04 / 1.00 | 0.13 / 1.00 |
| static_voi | 1.00 / 0.00 | 0.24 / 0.20 | 0.12 / 0.40 | 0.09 / 0.59 | 0.09 / 0.80 | 0.09 / 1.00 | 0.10 / 1.00 |
| greedy_voi | 1.00 / 0.00 | 0.77 / 0.00 | 0.66 / 0.00 | 0.61 / 0.01 | 0.58 / 0.03 | 0.65 / 1.00 | 0.62 / 1.00 |
| best_subset | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.03 | 1.00 / 1.00 | 1.00 / 1.00 |
| learned_query:s0 | 0.53 / 0.00 | 0.18 / 0.20 | 0.10 / 0.41 | 0.08 / 0.61 | 0.10 / 0.81 | 0.08 / 1.00 | 0.10 / 1.00 |
| learned_query:s0:control:random_network | 0.34 / 0.00 | 0.11 / 0.17 | 0.06 / 0.33 | 0.07 / 0.52 | 0.13 / 0.75 | 0.12 / 1.00 | 0.16 / 1.00 |
| learned_query:s0:control:untrained_init | 0.32 / 0.00 | 0.09 / 0.19 | 0.06 / 0.40 | 0.06 / 0.60 | 0.10 / 0.80 | 0.08 / 1.00 | 0.10 / 1.00 |

### Training signal `distill` (declared distillation, `distill/`): regret and selection only

### E1 (N = 5 consultable agents; 2000 dev episodes; `result-E1.json`, validated)

Mean regret to the global Bayes posterior (nats/episode, 95% bootstrap CI). Learned rows: seed 0.

| arm | b=1 | b=2 | b=3 | b=4 |
|---|---|---|---|---|
| random order, exact aggregation | 3.399 [3.316, 3.483] | 2.382 [2.308, 2.461] | 1.425 [1.360, 1.486] | 0.640 [0.599, 0.683] |
| index order, exact aggregation | 3.439 [3.358, 3.520] | 2.406 [2.332, 2.478] | 1.468 [1.409, 1.528] | 0.697 [0.654, 0.740] |
| stand-alone VoI ranking (ignores overlap), exact | 2.734 [2.658, 2.808] | 1.473 [1.413, 1.534] | 0.694 [0.650, 0.737] | 0.235 [0.205, 0.262] |
| exact-Bayes greedy VoI, exact | 2.734 [2.658, 2.808] | 1.412 [1.350, 1.471] | 0.556 [0.515, 0.599] | 0.179 [0.154, 0.204] |
| best b-subset in expectation (ceiling), exact | 2.734 [2.658, 2.808] | 1.400 [1.339, 1.458] | 0.543 [0.502, 0.587] | 0.180 [0.155, 0.205] |
| hindsight-best b-subset (unachievable), exact | 2.402 [2.334, 2.466] | 1.099 [1.050, 1.148] | 0.399 [0.367, 0.431] | 0.099 [0.081, 0.116] |
| learned query selection, exact aggregation | 2.789 [2.710, 2.865] | 1.489 [1.426, 1.553] | 0.672 [0.626, 0.720] | 0.219 [0.192, 0.246] |
| **learned query + learned aggregation (main arm)** | 2.788 [2.709, 2.865] | 1.824 [1.752, 1.895] | 1.291 [1.225, 1.359] | 0.980 [0.921, 1.040] |
| random selection + learned aggregation | 3.400 [3.316, 3.484] | 2.578 [2.500, 2.658] | 1.869 [1.794, 1.940] | 1.295 [1.230, 1.357] |
| control: random network (main arm pipeline) | 3.416 [3.333, 3.494] | 2.594 [2.520, 2.665] | 1.939 [1.871, 2.007] | 1.413 [1.354, 1.470] |
| control: untrained init (main arm pipeline) | 3.291 [3.207, 3.368] | 2.453 [2.378, 2.527] | 1.841 [1.773, 1.906] | 1.343 [1.287, 1.400] |
| learned query + learned, all seeds (means) | 2.788 / 2.800 / 2.800 | 1.824 / 1.822 / 1.832 | 1.291 / 1.285 / 1.295 | 0.980 / 0.989 / 0.984 |
| learned query + exact, all seeds (means) | 2.789 / 2.801 / 2.800 | 1.489 / 1.475 / 1.498 | 0.672 / 0.670 / 0.676 | 0.219 / 0.234 / 0.230 |
| share of the random → best-subset gap closed by learned selection (exact aggregation), per seed | 0.92 / 0.90 / 0.90 | 0.91 / 0.92 / 0.90 | 0.85 / 0.86 / 0.85 | 0.92 / 0.88 / 0.89 |
| same share for static VoI / greedy VoI | 1.00 / 1.00 | 0.92 / 0.99 | 0.83 / 0.99 | 0.88 / 1.00 |

Full information (b = N = 5): oracle_clearing -0.000 [-0.000, 0.000], extremized_log_pool 0.828 [0.770, 0.884], learned aggregator on all agents (s0) 0.827 [0.770, 0.884], its random-network control 0.918 [0.867, 0.969].


Selection diagnostics: share of episodes whose chosen set equals the best b-subset; share holding two agents with identical keys (a Sybil duplicate).

| policy | b=1 same / dup | b=2 same / dup | b=3 same / dup | b=4 same / dup |
|---|---|---|---|---|
| random | 0.21 / 0.00 | 0.10 / 0.00 | 0.10 / 0.00 | 0.20 / 0.00 |
| static_voi | 1.00 / 0.00 | 0.64 / 0.00 | 0.53 / 0.00 | 0.59 / 0.00 |
| greedy_voi | 1.00 / 0.00 | 0.94 / 0.00 | 0.94 / 0.00 | 0.98 / 0.00 |
| best_subset | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| learned_query:s0 | 0.76 / 0.00 | 0.63 / 0.00 | 0.58 / 0.00 | 0.63 / 0.00 |
| learned_query:s0:control:random_network | 0.23 / 0.00 | 0.12 / 0.00 | 0.11 / 0.00 | 0.20 / 0.00 |
| learned_query:s0:control:untrained_init | 0.29 / 0.00 | 0.19 / 0.00 | 0.17 / 0.00 | 0.28 / 0.00 |

### E2 (N = 5 consultable agents; 2000 dev episodes; `result-E2.json`, validated)

Mean regret to the global Bayes posterior (nats/episode, 95% bootstrap CI). Learned rows: seed 0.

| arm | b=1 | b=2 | b=3 | b=4 |
|---|---|---|---|---|
| random order, exact aggregation | 0.224 [0.203, 0.245] | 0.132 [0.115, 0.151] | 0.078 [0.064, 0.092] | 0.038 [0.027, 0.049] |
| index order, exact aggregation | 0.214 [0.193, 0.234] | 0.117 [0.099, 0.134] | 0.069 [0.055, 0.084] | 0.039 [0.028, 0.049] |
| stand-alone VoI ranking (ignores overlap), exact | 0.110 [0.092, 0.128] | 0.062 [0.047, 0.076] | 0.044 [0.033, 0.054] | 0.026 [0.018, 0.035] |
| exact-Bayes greedy VoI, exact | 0.110 [0.092, 0.128] | 0.049 [0.036, 0.061] | 0.025 [0.016, 0.034] | 0.021 [0.013, 0.029] |
| best b-subset in expectation (ceiling), exact | 0.110 [0.092, 0.128] | 0.044 [0.032, 0.055] | 0.024 [0.015, 0.033] | 0.020 [0.012, 0.027] |
| hindsight-best b-subset (unachievable), exact | 0.017 [0.012, 0.023] | 0.005 [0.002, 0.007] | 0.002 [-0.001, 0.005] | 0.004 [0.000, 0.008] |
| learned query selection, exact aggregation | 0.141 [0.122, 0.159] | 0.095 [0.079, 0.112] | 0.062 [0.047, 0.075] | 0.036 [0.024, 0.047] |
| **learned query + learned aggregation (main arm)** | 0.140 [0.122, 0.158] | 0.117 [0.101, 0.132] | 0.082 [0.066, 0.096] | 0.060 [0.045, 0.074] |
| random selection + learned aggregation | 0.232 [0.212, 0.253] | 0.146 [0.127, 0.166] | 0.109 [0.092, 0.128] | 0.082 [0.063, 0.101] |
| control: random network (main arm pipeline) | 0.174 [0.154, 0.193] | 0.099 [0.082, 0.117] | 0.081 [0.062, 0.100] | 0.078 [0.058, 0.099] |
| control: untrained init (main arm pipeline) | 0.201 [0.182, 0.219] | 0.129 [0.113, 0.146] | 0.091 [0.073, 0.107] | 0.063 [0.045, 0.081] |
| learned query + learned, all seeds (means) | 0.140 / 0.220 / 0.258 | 0.117 / 0.124 / 0.165 | 0.082 / 0.081 / 0.109 | 0.060 / 0.060 / 0.075 |
| learned query + exact, all seeds (means) | 0.141 / 0.218 / 0.247 | 0.095 / 0.109 / 0.151 | 0.062 / 0.064 / 0.089 | 0.036 / 0.033 / 0.046 |
| share of the random → best-subset gap closed by learned selection (exact aggregation), per seed | 0.73 / 0.06 / -0.20 | 0.42 / 0.27 / -0.21 | 0.30 / 0.26 / -0.19 | 0.12 / 0.27 / -0.42 |
| same share for static VoI / greedy VoI | 1.00 / 1.00 | 0.80 / 0.94 | 0.63 / 0.98 | 0.64 / 0.90 |

Full information (b = N = 5): oracle_clearing 0.017 [0.009, 0.024], extremized_log_pool 0.045 [0.033, 0.056], learned aggregator on all agents (s0) 0.050 [0.035, 0.065], its random-network control 0.083 [0.060, 0.108].


Selection diagnostics: share of episodes whose chosen set equals the best b-subset; share holding two agents with identical keys (a Sybil duplicate).

| policy | b=1 same / dup | b=2 same / dup | b=3 same / dup | b=4 same / dup |
|---|---|---|---|---|
| random | 0.20 / 0.00 | 0.10 / 0.00 | 0.10 / 0.00 | 0.21 / 0.00 |
| static_voi | 1.00 / 0.00 | 0.31 / 0.00 | 0.22 / 0.00 | 0.30 / 0.00 |
| greedy_voi | 1.00 / 0.00 | 0.79 / 0.00 | 0.76 / 0.00 | 0.84 / 0.00 |
| best_subset | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| learned_query:s0 | 0.53 / 0.00 | 0.12 / 0.00 | 0.09 / 0.00 | 0.16 / 0.01 |
| learned_query:s0:control:random_network | 0.38 / 0.00 | 0.17 / 0.00 | 0.12 / 0.00 | 0.22 / 0.01 |
| learned_query:s0:control:untrained_init | 0.32 / 0.00 | 0.11 / 0.00 | 0.10 / 0.00 | 0.21 / 0.01 |

### E3-sybil (N = 8 consultable agents; 2000 dev episodes; `result-E3-sybil.json`, validated)

Mean regret to the global Bayes posterior (nats/episode, 95% bootstrap CI). Learned rows: seed 0.

| arm | b=1 | b=2 | b=3 | b=4 | b=5 | b=6 | b=7 |
|---|---|---|---|---|---|---|---|
| random order, exact aggregation | 0.207 [0.187, 0.226] | 0.130 [0.111, 0.148] | 0.094 [0.078, 0.110] | 0.068 [0.054, 0.081] | 0.048 [0.036, 0.060] | 0.038 [0.028, 0.049] | 0.026 [0.017, 0.035] |
| index order, exact aggregation | 0.220 [0.198, 0.240] | 0.146 [0.126, 0.163] | 0.110 [0.093, 0.126] | 0.082 [0.067, 0.097] | 0.054 [0.042, 0.067] | 0.038 [0.027, 0.049] | 0.025 [0.016, 0.034] |
| stand-alone VoI ranking (ignores overlap), exact | 0.110 [0.092, 0.128] | 0.069 [0.053, 0.084] | 0.058 [0.046, 0.071] | 0.047 [0.034, 0.058] | 0.038 [0.027, 0.049] | 0.031 [0.021, 0.040] | 0.025 [0.016, 0.034] |
| exact-Bayes greedy VoI, exact | 0.110 [0.092, 0.128] | 0.049 [0.036, 0.061] | 0.025 [0.016, 0.034] | 0.021 [0.013, 0.029] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.024] |
| best b-subset in expectation (ceiling), exact | 0.110 [0.092, 0.128] | 0.044 [0.032, 0.055] | 0.024 [0.015, 0.033] | 0.020 [0.012, 0.027] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.024] |
| hindsight-best b-subset (unachievable), exact | 0.017 [0.012, 0.023] | 0.004 [0.001, 0.007] | 0.002 [-0.000, 0.005] | 0.002 [-0.001, 0.005] | 0.003 [-0.000, 0.006] | 0.003 [-0.000, 0.006] | 0.005 [0.001, 0.010] |
| learned query selection, exact aggregation | 0.179 [0.160, 0.199] | 0.098 [0.083, 0.113] | 0.069 [0.055, 0.083] | 0.059 [0.045, 0.071] | 0.047 [0.034, 0.059] | 0.040 [0.029, 0.051] | 0.030 [0.020, 0.039] |
| **learned query + learned aggregation (main arm)** | 0.194 [0.175, 0.213] | 0.117 [0.101, 0.132] | 0.095 [0.080, 0.109] | 0.082 [0.067, 0.095] | 0.073 [0.057, 0.087] | 0.070 [0.054, 0.086] | 0.072 [0.053, 0.090] |
| random selection + learned aggregation | 0.280 [0.260, 0.298] | 0.179 [0.161, 0.196] | 0.145 [0.127, 0.164] | 0.111 [0.094, 0.129] | 0.084 [0.067, 0.102] | 0.079 [0.060, 0.097] | 0.067 [0.049, 0.085] |
| control: random network (main arm pipeline) | 0.172 [0.152, 0.192] | 0.130 [0.109, 0.153] | 0.143 [0.116, 0.171] | 0.177 [0.142, 0.213] | 0.219 [0.178, 0.259] | 0.266 [0.217, 0.315] | 0.311 [0.255, 0.370] |
| control: untrained init (main arm pipeline) | 0.201 [0.181, 0.219] | 0.142 [0.123, 0.160] | 0.139 [0.115, 0.160] | 0.159 [0.130, 0.188] | 0.179 [0.146, 0.213] | 0.198 [0.159, 0.236] | 0.212 [0.170, 0.255] |
| learned query + learned, all seeds (means) | 0.194 / 0.220 / 0.164 | 0.117 / 0.135 / 0.100 | 0.095 / 0.108 / 0.078 | 0.082 / 0.105 / 0.063 | 0.073 / 0.106 / 0.062 | 0.070 / 0.103 / 0.065 | 0.072 / 0.100 / 0.069 |
| learned query + exact, all seeds (means) | 0.179 / 0.149 / 0.151 | 0.098 / 0.096 / 0.076 | 0.069 / 0.078 / 0.054 | 0.059 / 0.067 / 0.042 | 0.047 / 0.049 / 0.039 | 0.040 / 0.038 / 0.036 | 0.030 / 0.032 / 0.034 |
| share of the random → best-subset gap closed by learned selection (exact aggregation), per seed | 0.29 / 0.60 / 0.57 | 0.37 / 0.40 / 0.63 | 0.35 / 0.23 / 0.57 | 0.19 / 0.02 / 0.54 | 0.04 / -0.01 / 0.30 | -0.08 / 0.02 / 0.10 | -0.37 / -0.55 / -0.77 |
| same share for static VoI / greedy VoI | 1.00 / 1.00 | 0.71 / 0.94 | 0.50 / 0.98 | 0.44 / 0.96 | 0.32 / 1.00 | 0.35 / 1.00 | 0.11 / 1.00 |

Full information (b = N = 8): oracle_clearing 0.017 [0.009, 0.024], extremized_log_pool 0.078 [0.062, 0.092], learned aggregator on all agents (s0) 0.069 [0.050, 0.088], its random-network control 0.340 [0.279, 0.405].


Selection diagnostics: share of episodes whose chosen set equals the best b-subset; share holding two agents with identical keys (a Sybil duplicate).

| policy | b=1 same / dup | b=2 same / dup | b=3 same / dup | b=4 same / dup | b=5 same / dup | b=6 same / dup | b=7 same / dup |
|---|---|---|---|---|---|---|---|
| random | 0.12 / 0.00 | 0.03 / 0.23 | 0.02 / 0.52 | 0.02 / 0.77 | 0.01 / 0.93 | 0.04 / 1.00 | 0.13 / 1.00 |
| static_voi | 1.00 / 0.00 | 0.24 / 0.20 | 0.12 / 0.40 | 0.09 / 0.59 | 0.09 / 0.80 | 0.09 / 1.00 | 0.10 / 1.00 |
| greedy_voi | 1.00 / 0.00 | 0.77 / 0.00 | 0.66 / 0.00 | 0.61 / 0.01 | 0.58 / 0.03 | 0.65 / 1.00 | 0.62 / 1.00 |
| best_subset | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.03 | 1.00 / 1.00 | 1.00 / 1.00 |
| learned_query:s0 | 0.32 / 0.00 | 0.20 / 0.00 | 0.09 / 0.21 | 0.09 / 0.41 | 0.18 / 0.61 | 0.16 / 1.00 | 0.19 / 1.00 |
| learned_query:s0:control:random_network | 0.36 / 0.00 | 0.12 / 0.18 | 0.06 / 0.36 | 0.06 / 0.54 | 0.12 / 0.76 | 0.12 / 1.00 | 0.16 / 1.00 |
| learned_query:s0:control:untrained_init | 0.32 / 0.00 | 0.09 / 0.19 | 0.06 / 0.40 | 0.06 / 0.60 | 0.10 / 0.80 | 0.08 / 1.00 | 0.10 / 1.00 |

### Training signal `imitate` (greedy-VoI teacher from the view, `imitate/`): regret and selection only

### E1 (N = 5 consultable agents; 2000 dev episodes; `result-E1.json`, validated)

Mean regret to the global Bayes posterior (nats/episode, 95% bootstrap CI). Learned rows: seed 0.

| arm | b=1 | b=2 | b=3 | b=4 |
|---|---|---|---|---|
| random order, exact aggregation | 3.399 [3.316, 3.483] | 2.382 [2.308, 2.461] | 1.425 [1.360, 1.486] | 0.640 [0.599, 0.683] |
| index order, exact aggregation | 3.439 [3.358, 3.520] | 2.406 [2.332, 2.478] | 1.468 [1.409, 1.528] | 0.697 [0.654, 0.740] |
| stand-alone VoI ranking (ignores overlap), exact | 2.734 [2.658, 2.808] | 1.473 [1.413, 1.534] | 0.694 [0.650, 0.737] | 0.235 [0.205, 0.262] |
| exact-Bayes greedy VoI, exact | 2.734 [2.658, 2.808] | 1.412 [1.350, 1.471] | 0.556 [0.515, 0.599] | 0.179 [0.154, 0.204] |
| best b-subset in expectation (ceiling), exact | 2.734 [2.658, 2.808] | 1.400 [1.339, 1.458] | 0.543 [0.502, 0.587] | 0.180 [0.155, 0.205] |
| hindsight-best b-subset (unachievable), exact | 2.402 [2.334, 2.466] | 1.099 [1.050, 1.148] | 0.399 [0.367, 0.431] | 0.099 [0.081, 0.116] |
| learned query selection, exact aggregation | 2.753 [2.674, 2.831] | 1.474 [1.411, 1.534] | 0.642 [0.599, 0.684] | 0.202 [0.176, 0.229] |
| **learned query + learned aggregation (main arm)** | 2.754 [2.675, 2.833] | 1.818 [1.747, 1.888] | 1.275 [1.209, 1.342] | 0.975 [0.914, 1.035] |
| random selection + learned aggregation | 3.399 [3.316, 3.484] | 2.578 [2.499, 2.657] | 1.869 [1.795, 1.940] | 1.295 [1.232, 1.358] |
| control: random network (main arm pipeline) | 3.447 [3.367, 3.525] | 2.636 [2.560, 2.708] | 1.987 [1.921, 2.052] | 1.428 [1.368, 1.487] |
| control: untrained init (main arm pipeline) | 3.291 [3.207, 3.368] | 2.453 [2.378, 2.527] | 1.841 [1.773, 1.906] | 1.343 [1.287, 1.400] |
| learned query + learned, all seeds (means) | 2.754 / 2.782 / 2.771 | 1.818 / 1.823 / 1.805 | 1.275 / 1.288 / 1.289 | 0.975 / 0.995 / 0.982 |
| learned query + exact, all seeds (means) | 2.753 / 2.781 / 2.771 | 1.474 / 1.474 / 1.469 | 0.642 / 0.639 / 0.651 | 0.202 / 0.224 / 0.207 |
| share of the random → best-subset gap closed by learned selection (exact aggregation), per seed | 0.97 / 0.93 / 0.95 | 0.92 / 0.92 / 0.93 | 0.89 / 0.89 / 0.88 | 0.95 / 0.90 / 0.94 |
| same share for static VoI / greedy VoI | 1.00 / 1.00 | 0.92 / 0.99 | 0.83 / 0.99 | 0.88 / 1.00 |

Full information (b = N = 5): oracle_clearing -0.000 [-0.000, 0.000], extremized_log_pool 0.828 [0.770, 0.884], learned aggregator on all agents (s0) 0.828 [0.770, 0.885], its random-network control 0.915 [0.864, 0.966].


Selection diagnostics: share of episodes whose chosen set equals the best b-subset; share holding two agents with identical keys (a Sybil duplicate).

| policy | b=1 same / dup | b=2 same / dup | b=3 same / dup | b=4 same / dup |
|---|---|---|---|---|
| random | 0.21 / 0.00 | 0.10 / 0.00 | 0.10 / 0.00 | 0.20 / 0.00 |
| static_voi | 1.00 / 0.00 | 0.64 / 0.00 | 0.53 / 0.00 | 0.59 / 0.00 |
| greedy_voi | 1.00 / 0.00 | 0.94 / 0.00 | 0.94 / 0.00 | 0.98 / 0.00 |
| best_subset | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| learned_query:s0 | 0.81 / 0.00 | 0.67 / 0.00 | 0.61 / 0.00 | 0.68 / 0.00 |
| learned_query:s0:control:random_network | 0.21 / 0.00 | 0.10 / 0.00 | 0.10 / 0.00 | 0.18 / 0.00 |
| learned_query:s0:control:untrained_init | 0.29 / 0.00 | 0.19 / 0.00 | 0.17 / 0.00 | 0.28 / 0.00 |

### E2 (N = 5 consultable agents; 2000 dev episodes; `result-E2.json`, validated)

Mean regret to the global Bayes posterior (nats/episode, 95% bootstrap CI). Learned rows: seed 0.

| arm | b=1 | b=2 | b=3 | b=4 |
|---|---|---|---|---|
| random order, exact aggregation | 0.224 [0.203, 0.245] | 0.132 [0.115, 0.151] | 0.078 [0.064, 0.092] | 0.038 [0.027, 0.049] |
| index order, exact aggregation | 0.214 [0.193, 0.234] | 0.117 [0.099, 0.134] | 0.069 [0.055, 0.084] | 0.039 [0.028, 0.049] |
| stand-alone VoI ranking (ignores overlap), exact | 0.110 [0.092, 0.128] | 0.062 [0.047, 0.076] | 0.044 [0.033, 0.054] | 0.026 [0.018, 0.035] |
| exact-Bayes greedy VoI, exact | 0.110 [0.092, 0.128] | 0.049 [0.036, 0.061] | 0.025 [0.016, 0.034] | 0.021 [0.013, 0.029] |
| best b-subset in expectation (ceiling), exact | 0.110 [0.092, 0.128] | 0.044 [0.032, 0.055] | 0.024 [0.015, 0.033] | 0.020 [0.012, 0.027] |
| hindsight-best b-subset (unachievable), exact | 0.017 [0.012, 0.023] | 0.005 [0.002, 0.007] | 0.002 [-0.001, 0.005] | 0.004 [0.000, 0.008] |
| learned query selection, exact aggregation | 0.111 [0.093, 0.128] | 0.075 [0.060, 0.090] | 0.052 [0.039, 0.065] | 0.032 [0.023, 0.042] |
| **learned query + learned aggregation (main arm)** | 0.116 [0.098, 0.133] | 0.109 [0.093, 0.123] | 0.072 [0.057, 0.085] | 0.052 [0.040, 0.065] |
| random selection + learned aggregation | 0.242 [0.222, 0.261] | 0.156 [0.140, 0.174] | 0.108 [0.092, 0.124] | 0.070 [0.056, 0.084] |
| control: random network (main arm pipeline) | 0.192 [0.173, 0.212] | 0.124 [0.105, 0.142] | 0.091 [0.073, 0.109] | 0.078 [0.058, 0.098] |
| control: untrained init (main arm pipeline) | 0.201 [0.182, 0.219] | 0.129 [0.113, 0.146] | 0.091 [0.073, 0.107] | 0.063 [0.045, 0.081] |
| learned query + learned, all seeds (means) | 0.116 / 0.116 / 0.118 | 0.109 / 0.073 / 0.095 | 0.072 / 0.069 / 0.080 | 0.052 / 0.055 / 0.058 |
| learned query + exact, all seeds (means) | 0.111 / 0.109 / 0.110 | 0.075 / 0.059 / 0.070 | 0.052 / 0.045 / 0.060 | 0.032 / 0.029 / 0.033 |
| share of the random → best-subset gap closed by learned selection (exact aggregation), per seed | 0.99 / 1.01 / 1.00 | 0.64 / 0.83 / 0.71 | 0.48 / 0.61 / 0.33 | 0.31 / 0.52 / 0.26 |
| same share for static VoI / greedy VoI | 1.00 / 1.00 | 0.80 / 0.94 | 0.63 / 0.98 | 0.64 / 0.90 |

Full information (b = N = 5): oracle_clearing 0.017 [0.009, 0.024], extremized_log_pool 0.045 [0.033, 0.056], learned aggregator on all agents (s0) 0.047 [0.034, 0.062], its random-network control 0.083 [0.060, 0.108].


Selection diagnostics: share of episodes whose chosen set equals the best b-subset; share holding two agents with identical keys (a Sybil duplicate).

| policy | b=1 same / dup | b=2 same / dup | b=3 same / dup | b=4 same / dup |
|---|---|---|---|---|
| random | 0.20 / 0.00 | 0.10 / 0.00 | 0.10 / 0.00 | 0.21 / 0.00 |
| static_voi | 1.00 / 0.00 | 0.31 / 0.00 | 0.22 / 0.00 | 0.30 / 0.00 |
| greedy_voi | 1.00 / 0.00 | 0.79 / 0.00 | 0.76 / 0.00 | 0.84 / 0.00 |
| best_subset | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| learned_query:s0 | 0.93 / 0.00 | 0.20 / 0.00 | 0.13 / 0.00 | 0.25 / 0.01 |
| learned_query:s0:control:random_network | 0.29 / 0.00 | 0.12 / 0.00 | 0.09 / 0.00 | 0.18 / 0.01 |
| learned_query:s0:control:untrained_init | 0.32 / 0.00 | 0.11 / 0.00 | 0.10 / 0.00 | 0.21 / 0.01 |

### E3-sybil (N = 8 consultable agents; 2000 dev episodes; `result-E3-sybil.json`, validated)

Mean regret to the global Bayes posterior (nats/episode, 95% bootstrap CI). Learned rows: seed 0.

| arm | b=1 | b=2 | b=3 | b=4 | b=5 | b=6 | b=7 |
|---|---|---|---|---|---|---|---|
| random order, exact aggregation | 0.207 [0.187, 0.226] | 0.130 [0.111, 0.148] | 0.094 [0.078, 0.110] | 0.068 [0.054, 0.081] | 0.048 [0.036, 0.060] | 0.038 [0.028, 0.049] | 0.026 [0.017, 0.035] |
| index order, exact aggregation | 0.220 [0.198, 0.240] | 0.146 [0.126, 0.163] | 0.110 [0.093, 0.126] | 0.082 [0.067, 0.097] | 0.054 [0.042, 0.067] | 0.038 [0.027, 0.049] | 0.025 [0.016, 0.034] |
| stand-alone VoI ranking (ignores overlap), exact | 0.110 [0.092, 0.128] | 0.069 [0.053, 0.084] | 0.058 [0.046, 0.071] | 0.047 [0.034, 0.058] | 0.038 [0.027, 0.049] | 0.031 [0.021, 0.040] | 0.025 [0.016, 0.034] |
| exact-Bayes greedy VoI, exact | 0.110 [0.092, 0.128] | 0.049 [0.036, 0.061] | 0.025 [0.016, 0.034] | 0.021 [0.013, 0.029] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.024] |
| best b-subset in expectation (ceiling), exact | 0.110 [0.092, 0.128] | 0.044 [0.032, 0.055] | 0.024 [0.015, 0.033] | 0.020 [0.012, 0.027] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.024] | 0.017 [0.009, 0.024] |
| hindsight-best b-subset (unachievable), exact | 0.017 [0.012, 0.023] | 0.004 [0.001, 0.007] | 0.002 [-0.000, 0.005] | 0.002 [-0.001, 0.005] | 0.003 [-0.000, 0.006] | 0.003 [-0.000, 0.006] | 0.005 [0.001, 0.010] |
| learned query selection, exact aggregation | 0.111 [0.093, 0.128] | 0.069 [0.054, 0.082] | 0.054 [0.040, 0.066] | 0.053 [0.041, 0.064] | 0.049 [0.037, 0.060] | 0.045 [0.032, 0.056] | 0.033 [0.023, 0.044] |
| **learned query + learned aggregation (main arm)** | 0.125 [0.109, 0.142] | 0.086 [0.068, 0.103] | 0.105 [0.085, 0.123] | 0.111 [0.092, 0.131] | 0.118 [0.096, 0.138] | 0.113 [0.090, 0.133] | 0.106 [0.083, 0.127] |
| random selection + learned aggregation | 0.248 [0.230, 0.265] | 0.197 [0.178, 0.214] | 0.170 [0.150, 0.190] | 0.144 [0.124, 0.163] | 0.123 [0.103, 0.144] | 0.110 [0.089, 0.129] | 0.098 [0.076, 0.119] |
| control: random network (main arm pipeline) | 0.170 [0.151, 0.188] | 0.113 [0.093, 0.131] | 0.115 [0.093, 0.138] | 0.151 [0.120, 0.180] | 0.185 [0.147, 0.221] | 0.220 [0.177, 0.264] | 0.266 [0.216, 0.317] |
| control: untrained init (main arm pipeline) | 0.201 [0.181, 0.219] | 0.142 [0.123, 0.160] | 0.139 [0.115, 0.160] | 0.159 [0.130, 0.188] | 0.179 [0.146, 0.213] | 0.198 [0.159, 0.236] | 0.212 [0.170, 0.255] |
| learned query + learned, all seeds (means) | 0.125 / 0.129 / 0.126 | 0.086 / 0.173 / 0.089 | 0.105 / 0.137 / 0.077 | 0.111 / 0.110 / 0.058 | 0.118 / 0.098 / 0.051 | 0.113 / 0.090 / 0.054 | 0.106 / 0.087 / 0.062 |
| learned query + exact, all seeds (means) | 0.111 / 0.111 / 0.109 | 0.069 / 0.079 / 0.064 | 0.054 / 0.064 / 0.053 | 0.053 / 0.049 / 0.037 | 0.049 / 0.045 / 0.026 | 0.045 / 0.039 / 0.025 | 0.033 / 0.031 / 0.022 |
| share of the random → best-subset gap closed by learned selection (exact aggregation), per seed | 0.99 / 0.99 / 1.01 | 0.71 / 0.60 / 0.77 | 0.57 / 0.43 / 0.58 | 0.32 / 0.39 / 0.64 | -0.02 / 0.09 / 0.71 | -0.28 / -0.04 / 0.62 | -0.74 / -0.45 / 0.47 |
| same share for static VoI / greedy VoI | 1.00 / 1.00 | 0.71 / 0.94 | 0.50 / 0.98 | 0.44 / 0.96 | 0.32 / 1.00 | 0.35 / 1.00 | 0.11 / 1.00 |

Full information (b = N = 8): oracle_clearing 0.017 [0.009, 0.024], extremized_log_pool 0.078 [0.062, 0.092], learned aggregator on all agents (s0) 0.084 [0.065, 0.101], its random-network control 0.290 [0.237, 0.347].


Selection diagnostics: share of episodes whose chosen set equals the best b-subset; share holding two agents with identical keys (a Sybil duplicate).

| policy | b=1 same / dup | b=2 same / dup | b=3 same / dup | b=4 same / dup | b=5 same / dup | b=6 same / dup | b=7 same / dup |
|---|---|---|---|---|---|---|---|
| random | 0.12 / 0.00 | 0.03 / 0.23 | 0.02 / 0.52 | 0.02 / 0.77 | 0.01 / 0.93 | 0.04 / 1.00 | 0.13 / 1.00 |
| static_voi | 1.00 / 0.00 | 0.24 / 0.20 | 0.12 / 0.40 | 0.09 / 0.59 | 0.09 / 0.80 | 0.09 / 1.00 | 0.10 / 1.00 |
| greedy_voi | 1.00 / 0.00 | 0.77 / 0.00 | 0.66 / 0.00 | 0.61 / 0.01 | 0.58 / 0.03 | 0.65 / 1.00 | 0.62 / 1.00 |
| best_subset | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.03 | 1.00 / 1.00 | 1.00 / 1.00 |
| learned_query:s0 | 0.89 / 0.00 | 0.23 / 0.19 | 0.11 / 0.32 | 0.09 / 0.52 | 0.11 / 0.77 | 0.10 / 1.00 | 0.12 / 1.00 |
| learned_query:s0:control:random_network | 0.39 / 0.00 | 0.14 / 0.17 | 0.07 / 0.37 | 0.08 / 0.55 | 0.12 / 0.76 | 0.13 / 1.00 | 0.16 / 1.00 |
| learned_query:s0:control:untrained_init | 0.32 / 0.00 | 0.09 / 0.19 | 0.06 / 0.40 | 0.06 / 0.60 | 0.10 / 0.80 | 0.08 / 1.00 | 0.10 / 1.00 |

## Result (EXPLORATORY, one line)

In E1 and E2 the learned query beats random and fixed-order consultation and both controls, but in no environment does it approach exact-Bayes greedy VoI:
- It behaves like a static "most informative agent first" ranking that ignores what has already been consulted.
- In E3-sybil it consults a Sybil duplicate exactly as often as the static ranking (truth s0: 0.20 / 0.41 / 0.61 of episodes at b = 2 / 3 / 4, against at most 0.01 for greedy VoI). From b = 4 on it does no better than random consultation.
- Training it to imitate the greedy-VoI order does not remove the duplicates either. Imitation only fixes the first pick (b = 1).

## What this does and does not show

- **Selection and aggregation separately.** `learned_query+exact` isolates selection.
  - In E1 the learned query closes 0.86–0.92 of the random → best-subset gap (truth run, all seeds and budgets), about as much as the static VoI ranking (0.83–1.00) and short of greedy VoI (≥ 0.99).
  - In E2 it closes 0.20–0.81 (truth), declining as b grows.
  - In E3-sybil it closes 0.67/0.70 at b = 1 for seeds 0 and 2 (seed 1: −0.05) and ≤ 0.02 from b = 4 on.
  - Its learned aggregator adds further regret, worst in E3-sybil: the main arm stays near 0.09 nats for every b ≥ 4, because it does not discount duplicates either. That aggregation problem is lane L5's topic and is not studied here.
- **Controls (truth run, paired CI against the same network's random-network control).**
  - In E1 and E2 the main arm beats that control at every b for all three seeds, except E2 seed 2 at b = 4, where the CI includes 0. It has lower regret than the untrained-init control everywhere.
  - In E3-sybil, seed 2 beats the control at every b. Seed 0 loses at b = 1 (main 0.214 against 0.186) and ties at b = 2. Seed 1 loses at b = 1–4.
  - Random networks over keys also beat random consultation at b = 1: a random readout of the keys is correlated with how much an agent covers. The claim that "learned selection beats random" is therefore only as strong as its margin over the random-network row, not over `random+exact`.
- **Training signal.** Distillation to the train-split Bayes posterior does not help (E2 seed 2 is worse than random at every b). Imitation helps the first pick in E2 and E3-sybil (share ≈ 1.0 at b = 1) and a little in E1. Later picks stay far from greedy VoI.
  - An open candidate, not tested here: the query state is a sum of consulted-agent embeddings, read by a dot product with each candidate's key. Representing "this candidate duplicates a consulted one" may need a direct candidate-to-consulted similarity term.
- **Adaptivity.** In these environments, with known structure, the Bayes-optimal b-step policy is the non-adaptive `best_subset` (see Setup), so "queries that depend on the reports" have nothing to gain over key-only queries here. `hindsight_best` is far lower (E2 b = 2: 0.005 against 0.044 for the best subset), but it uses realized values that no policy can see before consulting.
- **Greedy vs. best subset.** `greedy_voi` is close to `best_subset`: the paired difference is at most +0.013 nats (E1 b = 3, CI [0.000, 0.026]), and every other CI includes 0.

## Provenance, cost, broken things

- **Code.** `l4_query_voi.py` (runner) and `make_tables.py` (tables) in this directory.
  - The truth run used commit `b94f9fe`. The distill and imitate runs used `1d48a0a`, which only adds the two training modes and the teacher; the truth path is unchanged.
  - Each JSON records `harness.git_commit`, `harness.sha256`, the runner's sha256 and `extra.harness_provenance`.
- **Harness provenance on the VM.** The VM copy has no `.git`, so `harness_info()` returned `git_commit = unknown`. The runner refused to write unless the aggbench sha256 matched the sha256 computed locally at the launch commit (`e508819…`). It then recorded the commit and set `git_dirty = false`; the note is in `extra.harness_provenance`.
- **Oracle check.** `oracle_check` is verified in every JSON: 24 brute-force smoke-split episodes per env.
- **Splits.** Train starts at seed 0 and dev at 10,000,000; the test split was never sampled.
- **Compute.** Spot L4 (g2-standard-8), list-price ceilings from GPU-LOG, 1.12 L4-hours and $0.98 in total:
  - attempt 1: 0.18 h, $0.16, crashed (below), `truth/failed-attempt1-*`;
  - truth run: 0.41 h, $0.36;
  - distill + imitate in parallel on one VM: 0.53 h, $0.46.
- **Broken 1, harness.** `aggbench.controls.random_weights` fails on a CUDA module: it draws on the CPU and adds the draw to CUDA tensors. The runner works around it by building the control from a CPU copy and moving it back. This is not fixed in aggbench.
- **Broken 2, launcher.** `work/run_on_l4.sh` does not pass `--run-id` to `gpu-run.sh`, so it exits ("Use --run-id when other soup VMs exist") whenever another lane has a VM up. It also deletes its worktree directory without `git worktree prune`, so reusing a run id fails ("missing but already registered worktree"). Runs 2 and 3 used a copy of the script with `--run-id` added.
- **Settings.** Hand-set constants are listed in `extra.hyperparameters`. None was tuned on results; one setting (6,000 steps) was used for every run after the CPU smoke.
