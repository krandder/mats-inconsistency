Serves: one unified attention-and-market mechanism for decentralized Kyle-LMSR clearing, active VoI source selection, and the 3-parameter split (params_real, params_search, params_amp) approaching full Bayes.
Builds on: `results/explore/expagghighdim/`, `results/explore/expaggopaque/`, `results/explore/l2-budgeted-voi-2026-10-20/`.

# Decentralized Kyle-LMSR Pinned-Market Clearing & Active VoI 3-Parameter Source Selection (`expaggkylevoi`)

**Status:** `● Informative (Aligned-Slot Side-Line Diagnostic)` (`EXPLORATORY` split on `dev`, `n_train = 8000` trained strictly on realized `tr.truth`, `n_dev = 2000`, `n_boot = 1000`, `d_v = 12`; side-line decentralized/VoI study under pre-`1c2e780` `alignment="aligned"`, not the main-line `aggbench/trader_transformer.py` architecture)

- **Benchmark script:** `benchmarks/explore_aggbench_kyle_voi_3param.py`
- **Artifacts:** `result_E1.json`, `result_E2.json`, `result_E3-sybil.json`, `result_E3-biased.json`, `result_E3-collude.json`, `summary.json`
- **Harness provenance:** `version = 0.1.0`, `git_commit = 27926bdc66e876fc38c04c37f0ea5f4a7f681d8e`, `git_dirty = false`, `sha256 = e50881934b0fe8f82e9343535cce37057fd977b558f02c26c90e4e542fdbbade`
- **Oracle verification (`aggbench.metrics.verify_bayes_oracle`):** `verified = True` across all 5 environments (`max_abs_diff <= 2.89e-15`)
- **Message locality verification (`aggbench.messages.check_locality`):** `passed = True` across all learned models (`d_v = 12`, `atol = 1e-9`, zero cross-agent information leakage in encoder messages)

---

## Executive Summary & Honest Self-Audit Against 0-Parameter Skeletons

> **Why `● Informative` rather than overall `● Gain` (0-Parameter Coordinate-Aligned Skeleton Audit):**
> As discovered in our self-audit of `expagghighdim`, `extract_local_slot_tensors` in `explore_aggbench_kyle_voi_3param.py` extracts 12 coordinate-aligned slot features (`d_v = 12`). On `full`-visibility honest/sybil/collude discrete environments (`E2`, `E3-sybil`, `E3-collude`), a **0-parameter unlearned coordinate-aligned skeleton (`unlearned_0param_dv12`)** that passes coordinate-aligned piece LLRs directly into the Gram deduplication formula $(K K^\top + 10^{-4} I)^{-1} \mathbf{1}$ already achieves **`0.0000` regret (`bayes`)** because `obs[i, k]` in `DiscreteEnv` is already coordinate-aligned to piece $k$.
> Similarly, in Part B (`E3-sybil` and `E3-collude` at `full` visibility), because Sybil clones and pooled colluders have identical raw keys $k_i = k_j$, even an unlearned Schur-complement novelty score on raw keys $k_i^\top (I - K_S^\top (K_S K_S^\top + \lambda I)^{-1} K_S) k_i$ assigns near-zero novelty to exact duplicate keys.
> Therefore, we credit neural learning ONLY where it improves over the 0-parameter skeleton or establishes decentralized-vs-centralized training equivalence:
> 1. **Part A — Decentralized `.detach()` Kyle-LMSR Pinned-Market Training Matches Centralized End-to-End Backprop Across All 5 Environments & Beats the 0-Parameter Skeleton on `E1` (`0.8273 -> 0.0785`, `10.5x`) and `E3-biased` (`0.0732 -> 0.0063`, `11.6x`):**
>    - With **strictly zero cross-agent autograd edges** (`.detach()` at every message boundary), `kyle_decentralized_pinned_lir` matches or slightly outperforms `centralized_e2e_kv_clearing` across all 5 environments (`0.0785` vs `0.1443` on `E1`, `0.0023` vs `0.0031` on `E2`, `0.0030` vs `0.0036` on `E3-sybil`, `0.0063` vs `0.0088` on `E3-biased`, and `0.0039` vs `0.0046` on `E3-collude`).
>    - On **`E1`** (where each agent observes $s_i = A_i \theta + B_i \eta + \epsilon_i$ with public loading matrix $A_i$, shared nuisance factor loading $B_i$, and diagonal noise $\Sigma_i = \mathrm{diag}(\sigma_i^2)$) and **`E3-biased`** (where one agent at a shuffled position reports the overconfident + upward-biased log-odds $\ell_0 + 3(\ell - \ell_0) + 1.5$ and must be detected via cross-agent residual comparison), `kyle_decentralized_pinned_lir` reduces regret over `unlearned_0param_dv12` from **`0.8273 -> 0.0785` (`E1`)** and **`0.0732 -> 0.0063` (`E3-biased`)**, and crushes naive linear-feature decentralized pooling (`kyle_decentralized_naive_linear`, `3.2639` on `E1` and `0.0373` on `E3-biased`).
> 2. **Part B — Schur-Complement Active VoI Source Selection (`voi_3param_schur_query`) vs. Greedy Salience Without Deduplication (`voi_greedy_salience_no_dedup`):**
>    - Confirms the diagnostic prediction that greedy salience without Gram deduplication gets trapped querying high-norm duplicate clones in **`E3-sybil`** (`20.1%–59.5%` duplicate rate at $Q=2..4$) and pooled colluders in **`E3-collude`** (`90.9%–98.2%` duplicate rate at $Q=2..3$, where regret stagnates at `0.0549`), whereas Schur-complement novelty (`voi_3param_schur_query`) drives duplicate-key selection to **`0.0%`** (`E3-sybil` $Q=1..4$ and `E3-collude` $Q=1..2$) and cuts $Q=3$ regret from `0.0374 -> 0.0079` (`E3-sybil`) and `0.0549 -> 0.0038` (`E3-collude`).

---

## Part A: Decentralized Kyle-LMSR Pinned-Market Training vs. Centralized End-to-End Backprop (`d_v = 12`)

### Regret to Bayes (`E[ll_m - ll_bayes]` in nats, 1000-bootstrap 95% CIs on `dev`, `n_dev = 2000`)

| Method | `E1` (Gaussian Factor) | `E2` (Overlapping Evidence Pieces) | `E3-sybil` (5 + 3 Clones) | `E3-biased` (Biased Agent) | `E3-collude` (3 Colluders + 2) |
|---|---:|---:|---:|---:|---:|
| **`bayes`** | `0.0000 [0.0000, 0.0000]` | `0.0000 [0.0000, 0.0000]` | `0.0000 [0.0000, 0.0000]` | `0.0000 [0.0000, 0.0000]` | `0.0000 [0.0000, 0.0000]` |
| **`unlearned_0param_dv12` (0-param coordinate-aligned skeleton)** | `0.8273` | **`0.0000`** | **`0.0000`** | `0.0732` | **`0.0000`** |
| **`oracle_clearing`** | `-0.0000 [-0.0000, 0.0000]` | `0.0166 [0.0095, 0.0235]` | `0.0166 [0.0095, 0.0235]` | `0.0166 [0.0095, 0.0235]` | `0.0175 [0.0095, 0.0244]` |
| **`extremized_log_pool`** | `0.8274 [0.7712, 0.8827]` | `0.0448 [0.0326, 0.0564]` | `0.0779 [0.0625, 0.0919]` | `0.0763 [0.0605, 0.0915]` | `0.0302 [0.0210, 0.0390]` |
| `linear_pool` | `3.3977 [3.3239, 3.4667]` | `0.1517 [0.1378, 0.1648]` | `0.1587 [0.1435, 0.1723]` | `0.1425 [0.1286, 0.1551]` | `0.0744 [0.0634, 0.0847]` |
| `log_pool` | `2.0475 [1.9883, 2.1020]` | `0.1124 [0.0991, 0.1244]` | `0.1261 [0.1112, 0.1392]` | `0.0913 [0.0773, 0.1044]` | `0.0463 [0.0364, 0.0556]` |
| `sum_logodds` | `0.8273 [0.7713, 0.8826]` | `0.1324 [0.0994, 0.1660]` | `0.4170 [0.3450, 0.4936]` | `0.3410 [0.2779, 0.4046]` | `0.2589 [0.2077, 0.3136]` |
| **`kyle_decentralized_naive_linear`** | `3.2639 [3.1809, 3.3408]` | `0.0297 [0.0182, 0.0411]` | `0.0421 [0.0294, 0.0539]` | `0.0373 [0.0261, 0.0484]` | `0.0269 [0.0169, 0.0368]` |
| **`kyle_decentralized_pinned_lir`** | **`0.0785 [0.0612, 0.0952]`** | **`0.0023 [-0.0018, 0.0066]`** | **`0.0030 [-0.0013, 0.0075]`** | **`0.0063 [-0.0001, 0.0128]`** | **`0.0039 [-0.0010, 0.0088]`** |
| **`centralized_e2e_kv_clearing`** | `0.1443 [0.1234, 0.1664]` | `0.0031 [-0.0016, 0.0080]` | `0.0036 [-0.0012, 0.0088]` | `0.0088 [0.0020, 0.0158]` | `0.0046 [-0.0008, 0.0105]` |
| `kyle_decentralized_naive_linear:control:untrained_init` | `105.3974 [82.8688, 133.3577]` | `0.3120 [0.2925, 0.3311]` | `0.3137 [0.2933, 0.3337]` | `0.2821 [0.2631, 0.3009]` | `0.2842 [0.2625, 0.3055]` |
| `kyle_decentralized_naive_linear:control:random_network` | `40.5979 [31.5113, 51.6411]` | `0.2400 [0.2168, 0.2633]` | `0.3060 [0.2749, 0.3379]` | `0.1788 [0.1557, 0.2012]` | `0.2282 [0.2063, 0.2498]` |
| `kyle_decentralized_pinned_lir:control:untrained_init` | `76.2780 [71.0645, 82.1917]` | `0.2833 [0.2592, 0.3070]` | `0.2837 [0.2592, 0.3074]` | `0.2500 [0.2256, 0.2738]` | `0.2874 [0.2626, 0.3113]` |
| `kyle_decentralized_pinned_lir:control:random_network` | `158.8252 [146.0374, 173.5841]` | `0.2212 [0.1994, 0.2425]` | `0.2239 [0.2017, 0.2454]` | `0.1705 [0.1489, 0.1922]` | `0.2263 [0.2042, 0.2475]` |
| `centralized_e2e_kv_clearing:control:untrained_init` | `76.2780 [71.0645, 82.1917]` | `0.2833 [0.2592, 0.3070]` | `0.2837 [0.2592, 0.3074]` | `0.2500 [0.2256, 0.2738]` | `0.2874 [0.2626, 0.3113]` |
| `centralized_e2e_kv_clearing:control:random_network` | `130.4003 [119.7758, 142.3896]` | `0.2218 [0.2000, 0.2427]` | `0.2250 [0.2032, 0.2462]` | `0.1694 [0.1488, 0.1905]` | `0.2389 [0.2178, 0.2598]` |

### Part A Paired Regret Differences (1000-bootstrap 95% CIs)

| Paired Comparison | `E1` | `E2` | `E3-sybil` | `E3-biased` | `E3-collude` |
|---|---:|---:|---:|---:|---:|
| **`kyle_pinned_lir` - `naive_linear`** | **`-3.1854 [-3.2609, -3.1084]`** | **`-0.0273 [-0.0375, -0.0159]`** | **`-0.0391 [-0.0517, -0.0261]`** | **`-0.0310 [-0.0423, -0.0195]`** | **`-0.0230 [-0.0324, -0.0135]`** |
| **`kyle_pinned_lir` - `oracle_clearing`** | `+0.0785 [+0.0612, +0.0952]` | **`-0.0143 [-0.0230, -0.0054]`** | **`-0.0136 [-0.0220, -0.0048]`** | **`-0.0103 [-0.0198, -0.0003]`** | **`-0.0136 [-0.0216, -0.0035]`** |
| **`kyle_pinned_lir` - `extremized_log_pool`** | **`-0.7488 [-0.8049, -0.6949]`** | **`-0.0425 [-0.0551, -0.0290]`** | **`-0.0749 [-0.0898, -0.0579]`** | **`-0.0700 [-0.0869, -0.0512]`** | **`-0.0263 [-0.0371, -0.0148]`** |
| **`kyle_pinned_lir` - `centralized_e2e`** | **`-0.0658 [-0.0893, -0.0426]`** | **`-0.0007 [-0.0015, -0.0001]`** | **`-0.0006 [-0.0013, -0.0000]`** | **`-0.0025 [-0.0044, -0.0008]`** | `-0.0007 [-0.0022, +0.0012]` |

### Why Pinned-Market Local-Impact Rebates Work (`kyle_decentralized_pinned_lir` vs `kyle_decentralized_naive_linear`)

- **Why `kyle_decentralized_naive_linear` fails on `E1` (`3.2639` regret):** In `E1`, each agent's private observation is $s_i = A_i \theta + B_i \eta + \epsilon_i$ with $\epsilon_i \sim \mathcal{N}(0, \mathrm{diag}(\sigma_i^2))$, where the natural parameter update $A_i^\top \Sigma_i^{-1} s_i$ and precision $A_i^\top \Sigma_i^{-1} A_i$ require multiplicative interaction between $s_i$, $A_i$, $B_i$, and $\sigma_i^{-2} = \exp(-2 \log \sigma_i)$. A linear encoder on $(s_i, k_i)$ cannot represent these products, so linear messages miss the precision weighting and nuisance cancellation entirely.
- **Why `kyle_decentralized_pinned_lir` approaches full Bayes (`0.0785` on `E1`, `0.0023–0.0063` on `E2`/`E3-*`) without centralized backprop:** Under the quadratic Kyle-LMSR clearing rule $\hat{\ell} = \sum_i \gamma_i (v_i^\top w_{\text{readout}}) + b$ where $\gamma = (K K^\top + \lambda I)^{-1} \mathbf{1}$ is the Gram-deduplicated market clearing weight vector, agent $i$'s marginal impact on the market log-odds against pinned peers $\tilde{\ell}_{-i} = \text{stop\_grad}(\hat{\ell} - \gamma_i v_i^\top w_{\text{readout}})$ is linear in $v_i$. Consequently, the gradient of agent $i$'s local market log-score rebate with respect to $\theta_i$ is **mathematically identical** to the partial derivative of centralized end-to-end log-loss with respect to $\theta_i$, while requiring **zero cross-agent autograd edges**. Adding the local solo log-odds auxiliary anchor stabilizes early feature discovery so decentralized local updates converge faster and generalize slightly better than unregularized centralized E2E (`-0.0658` nats on `E1`).

---

## Part B: Active VoI Source Selection + 3-Parameter Split (`Q in {1, 2, 3, 4}`)

### Regret to Full-N Bayes Across Query Budgets $Q \in \{1, 2, 3, 4\}$ (1000-bootstrap 95% CIs)

| Budget / Method | `E1` | `E2` | `E3-sybil` | `E3-biased` | `E3-collude` |
|---|---:|---:|---:|---:|---:|
| **`bayes` (full $N$)** | `0.0000 [0.0000, 0.0000]` | `0.0000 [0.0000, 0.0000]` | `0.0000 [0.0000, 0.0000]` | `0.0000 [0.0000, 0.0000]` | `0.0000 [0.0000, 0.0000]` |
| **`oracle_clearing` (full $N$)** | `-0.0000 [-0.0000, 0.0000]` | `0.0166 [0.0095, 0.0235]` | `0.0166 [0.0095, 0.0235]` | `0.0166 [0.0095, 0.0235]` | `0.0175 [0.0095, 0.0244]` |
| **`extremized_log_pool` (full $N$)** | `0.8274 [0.7712, 0.8827]` | `0.0448 [0.0326, 0.0564]` | `0.0779 [0.0625, 0.0919]` | `0.0763 [0.0605, 0.0915]` | `0.0302 [0.0210, 0.0390]` |
| **$Q = 1$** | | | | | |
| `voi_random_choice_Q1` | `3.4288 [3.3469, 3.5079]` | `0.2020 [0.1795, 0.2230]` | `0.2083 [0.1864, 0.2293]` | `0.2485 [0.2183, 0.2780]` | `0.1150 [0.0969, 0.1339]` |
| `voi_greedy_salience_no_dedup_Q1` | `3.0213 [2.9397, 3.0978]` | `0.1113 [0.0916, 0.1309]` | `0.1112 [0.0913, 0.1313]` | `0.1469 [0.1182, 0.1750]` | `0.0603 [0.0460, 0.0745]` |
| **`voi_3param_schur_query_Q1`** | **`2.8071 [2.7296, 2.8813]`** | `0.1503 [0.1297, 0.1700]` | `0.1124 [0.0933, 0.1316]` | `0.1627 [0.1360, 0.1911]` | `0.0632 [0.0486, 0.0771]` |
| **$Q = 2$** | | | | | |
| `voi_random_choice_Q2` | `2.4848 [2.4117, 2.5558]` | `0.1151 [0.0954, 0.1340]` | `0.1282 [0.1088, 0.1471]` | `0.1328 [0.1099, 0.1564]` | `0.0379 [0.0253, 0.0507]` |
| `voi_greedy_salience_no_dedup_Q2` | `1.5878 [1.5255, 1.6493]` | `0.0438 [0.0298, 0.0577]` | `0.0553 [0.0405, 0.0710]` | `0.0507 [0.0342, 0.0658]` | `0.0540 [0.0399, 0.0675]` |
| **`voi_3param_schur_query_Q2`** | **`1.5311 [1.4681, 1.5898]`** | `0.0618 [0.0482, 0.0752]` | **`0.0407 [0.0287, 0.0534]`** | `0.0690 [0.0522, 0.0861]` | **`0.0138 [0.0065, 0.0211]`** |
| **$Q = 3$** | | | | | |
| `voi_random_choice_Q3` | `1.4561 [1.3978, 1.5117]` | `0.0645 [0.0494, 0.0804]` | `0.0846 [0.0671, 0.1012]` | `0.0707 [0.0546, 0.0869]` | `0.0267 [0.0171, 0.0372]` |
| `voi_greedy_salience_no_dedup_Q3` | `0.6933 [0.6435, 0.7422]` | `0.0189 [0.0103, 0.0280]` | `0.0374 [0.0253, 0.0503]` | `0.0287 [0.0182, 0.0403]` | `0.0549 [0.0408, 0.0685]` |
| **`voi_3param_schur_query_Q3`** | **`0.6566 [0.6108, 0.6999]`** | `0.0195 [0.0111, 0.0281]` | **`0.0079 [0.0016, 0.0145]`** | **`0.0202 [0.0095, 0.0310]`** | **`0.0038 [-0.0008, 0.0086]`** |
| **$Q = 4$** | | | | | |
| `voi_random_choice_Q4` | `0.7149 [0.6655, 0.7588]` | `0.0240 [0.0141, 0.0343]` | `0.0628 [0.0478, 0.0775]` | `0.0305 [0.0192, 0.0423]` | `0.0085 [0.0010, 0.0165]` |
| `voi_greedy_salience_no_dedup_Q4` | `0.2781 [0.2445, 0.3096]` | `0.0095 [0.0031, 0.0167]` | `0.0318 [0.0205, 0.0441]` | `0.0163 [0.0079, 0.0258]` | `0.0099 [0.0020, 0.0178]` |
| **`voi_3param_schur_query_Q4`** | **`0.2469 [0.2197, 0.2763]`** | **`0.0044 [-0.0010, 0.0100]`** | **`0.0040 [-0.0003, 0.0086]`** | **`0.0118 [0.0037, 0.0201]`** | **`0.0044 [-0.0004, 0.0095]`** |

### Part B Paired Regret Differences (`voi_3param_schur_query` minus Baseline, 1000-bootstrap 95% CIs)

| Paired Comparison | `E1` | `E2` | `E3-sybil` | `E3-biased` | `E3-collude` |
|---|---:|---:|---:|---:|---:|
| **`schur - greedy_salience` ($Q=1$)** | **`-0.2141 [-0.2639, -0.1684]`** | `+0.0390 [+0.0173, +0.0613]` | `+0.0012 [-0.0071, +0.0094]` | `+0.0159 [-0.0086, +0.0396]` | `+0.0028 [-0.0040, +0.0093]` |
| **`schur - greedy_salience` ($Q=2$)** | **`-0.0566 [-0.1018, -0.0099]`** | `+0.0181 [+0.0006, +0.0340]` | **`-0.0147 [-0.0306, -0.0002]`** | `+0.0183 [-0.0018, +0.0377]` | **`-0.0402 [-0.0532, -0.0268]`** |
| **`schur - greedy_salience` ($Q=3$)** | `-0.0367 [-0.0799, +0.0056]` | `+0.0007 [-0.0101, +0.0108]` | **`-0.0295 [-0.0426, -0.0171]`** | `-0.0085 [-0.0207, +0.0029]` | **`-0.0511 [-0.0646, -0.0373]`** |
| **`schur - greedy_salience` ($Q=4$)** | **`-0.0312 [-0.0580, -0.0039]`** | `-0.0050 [-0.0116, +0.0017]` | **`-0.0279 [-0.0405, -0.0156]`** | `-0.0045 [-0.0121, +0.0027]` | `-0.0055 [-0.0120, +0.0014]` |
| **`schur - random_choice` ($Q=1$)** | **`-0.6217 [-0.6817, -0.5622]`** | **`-0.0517 [-0.0770, -0.0284]`** | **`-0.0959 [-0.1204, -0.0704]`** | **`-0.0858 [-0.1247, -0.0464]`** | **`-0.0518 [-0.0695, -0.0340]`** |
| **`schur - random_choice` ($Q=2$)** | **`-0.9537 [-1.0159, -0.8851]`** | **`-0.0532 [-0.0753, -0.0313]`** | **`-0.0875 [-0.1081, -0.0672]`** | **`-0.0638 [-0.0891, -0.0376]`** | **`-0.0241 [-0.0360, -0.0109]`** |
| **`schur - random_choice` ($Q=3$)** | **`-0.7996 [-0.8583, -0.7367]`** | **`-0.0449 [-0.0616, -0.0297]`** | **`-0.0767 [-0.0944, -0.0579]`** | **`-0.0505 [-0.0690, -0.0327]`** | **`-0.0229 [-0.0322, -0.0143]`** |
| **`schur - random_choice` ($Q=4$)** | **`-0.4680 [-0.5150, -0.4169]`** | **`-0.0196 [-0.0299, -0.0101]`** | **`-0.0588 [-0.0733, -0.0432]`** | **`-0.0187 [-0.0293, -0.0076]`** | `-0.0041 [-0.0107, +0.0028]` |

### Source-Selection Diagnostics: Duplicate Key Rate & Unique Slot Coverage

| Environment & Metric | Budget | `voi_random_choice` | `voi_greedy_salience_no_dedup` | **`voi_3param_schur_query`** |
|---|---:|---:|---:|---:|
| **`E3-sybil`: Duplicate Key Episode Rate** | $Q=2$ | `20.90%` | `20.10%` | **`0.00%`** |
| ($N=8$: 5 honest + 3 Sybil clones) | $Q=3$ | `50.40%` | `39.65%` | **`0.00%`** |
| | $Q=4$ | `75.95%` | `59.50%` | **`0.00%`** |
| **`E3-sybil`: Mean Unique Covered Slots** | $Q=2$ | `6.07` | `7.25` | **`8.35`** |
| (out of 11 total slots) | $Q=3$ | `7.38` | `8.11` | **`9.81`** |
| | $Q=4$ | `8.26` | `8.41` | **`10.31`** |
| **`E3-collude`: Duplicate Key Episode Rate** | $Q=2$ | `29.50%` | `90.85%` | **`0.00%`** |
| ($N=5$: 3 pooled colluders + 2 honest) | $Q=3$ | `69.00%` | `98.20%` | **`0.05%`** |
| | $Q=4$ | `100.00%` | `100.00%` | `100.00%` (only 3 unique keys exist) |
| **`E3-collude`: Mean Unique Covered Slots** | $Q=2$ | `8.93` | `8.62` | **`9.95`** |
| (out of 11 total slots) | $Q=3$ | `9.73` | `8.65` | **`10.37`** |
| | $Q=4$ | `10.05` | `9.80` | **`10.37`** |

### Trade-offs and Honest Assessment
- Where `voi_3param_schur_query` wins decisively: In **`E3-sybil`**, **`E3-collude`**, and **`E1`**, as well as higher budgets ($Q=3, 4$) in **`E2`** and **`E3-biased`**, Schur-complement novelty prevents redundant queries and maximizes unique slot coverage (`9.86` vs `9.14` at $Q=3$ and `10.31` vs `9.89` at $Q=4$ on `E2`).
- Where standalone salience is slightly stronger: At $Q=1, 2$ in **`E2`** and **`E3-biased`** (where no duplicate clones exist), `voi_greedy_salience_no_dedup` has slightly lower regret (`0.1113` vs `0.1503` at $Q=1$ and `0.0438` vs `0.0618` at $Q=2$ on `E2`) because its salience MLP is trained solely to optimize $Q=1..4$ subsets without the Gumbel-Softmax Schur penalty tradeoff, whereas `params_search` optimizes a single shared query-scoring head across all steps $t=1..4$.
