# EXPLORATORY (Aligned-Slot Giveaway Control / Side-Line Diagnostic): High-Dimensional Local Key-Value Messages & Gram-Confiability Clearing Across `E2` and `E3` (`expagghighdim`)

Serves: high-dimensional per-agent Key-Value messages + Gram-confiability clearing diagnostic on E2 and E3 (aligned-slot giveaway control; side-line to Kelvin's trader_transformer).
Builds on: Winkler (1981) / Satopää et al. (2016) Partial Information Framework & Woodbury Gram clearing.

**Status:** `● Informative (Aligned-Slot Diagnostic / Giveaway Control; Side-Line Exploratory)` (Honest audit against 0-parameter skeleton plug-in `unlearned_0param_dv12`: because this run used `alignment="aligned"` and `P = make_bucket_matrix(K, d_v)`, the coordinate-aligned skeleton with 0 learned parameters already achieves `0.0000` regret at $d_v=12$ on `full` honest/sybil/collude environments, so `slot_kv_dv12:trained`'s `0.0040–0.0045` on `E2/E3-sybil/E3-collude (full)` is an aligned-slot giveaway rather than neural learning; **genuine neural learning gains over the 0-parameter skeleton occur only on `E3-biased (full)` (`0.0732 -> 0.0111`, $6.6\times$ reduction) and across the 4 `visibility=noisy` cells (`E2`: `0.0386 -> 0.0178`, `E3-sybil`: `0.0216 -> 0.0107`, `E3-collude`: `0.0223 -> 0.0124`, `E3-biased`: `0.1094 -> 0.0232`)**; naive MLP+MultiheadAttention `naive_mha_dv12` is `● No gain` at `0.091–0.109`). (`EXPLORATORY`; `test` splits untouched; harness commit `20ca7bbd08`, `git_dirty=False`; `locality_verified=True` across all 56 learned models via `aggbench.messages.check_locality`).

## Why, What We Did, Result, So What, Decided

- **Why**: When agents in `E2`/`E3` only emit 1D scalar log-odds reports $r_i \in \mathbb{R}$, overlapping coverage across $K=12$ evidence pieces is non-identifiable ($\mathbf{1}_{\bigcup S_i} \in \text{rowspace}(C)$ in only `3.1%` of `E2` episodes), trapping even the exact `oracle_clearing` ceiling at `0.0166` regret above `bayes` (`0.0000`). We tested whether strictly local per-agent encoders emitting $d_v$-dimensional value vectors ($d_v \in \{1, 2, 4, 8, 12\}$) can break through `oracle_clearing` and approach `bayes` when trained solely on binary truth labels $Y \in \{0, 1\}$.
- **What we did**: Built `benchmarks/explore_aggbench_highdim_messages.py` importing `aggbench` unmodified (`git_dirty=False`). For `E3-biased`, we distorted the biased agent's seen observations prior to encoding (`biased_agent_obs(batch)`) so unobserved pieces remain `NaN` with zero oracle leakage. Across all 8 `(env, visibility)` cells (`E2`, `E3-sybil`, `E3-collude`, `E3-biased` $\times$ `full`, `noisy`, $n_{\text{train}}=2000, n_{\text{dev}}=2000, n_{\text{boot}}=2000$, `alignment="aligned"`), we compared scalar baselines, naive flat MLP + MultiheadAttention (`naive_mha_dv1`, `naive_mha_dv12`), and our Slot-Equivariant Key-Value Message & Clearing architecture (`slot_kv_dv{1,2,4,8,12}`) with `untrained_init`, `random_network`, and `unlearned_0param_dv12` controls.
- **Result**: On `E2/E3-sybil/E3-collude (full)` under `alignment="aligned"`, the 0-parameter coordinate-aligned skeleton `unlearned_0param_dv12` already gets `0.0000` regret, so `slot_kv_dv12:trained` (`0.0040–0.0045`) is a giveaway control there; neural learning beats `unlearned_0param_dv12` on `E3-biased (full)` (`0.0732 -> 0.0111`) and `visibility=noisy` (`0.0216–0.1094 -> 0.0107–0.0232`).
- **So what**: Aligned slots (`alignment="aligned"`) hand the overlap structure to a coordinate-aligned skeleton for free; headline tests must use `alignment="unaligned"` and compare directly against Kelvin's `trader_transformer` and `full_context_transformer` bar.
- **Decided (Withdrawn)**: Withdrew the earlier proposal to adopt coordinate-aligned `slot_kv_dv12` as a reference architecture; relabelled this run as an aligned-slot giveaway diagnostic and side-line baseline.

## Master Comparison Table: `visibility = full` (`dev`, $n=2000$, Mean Regret to `bayes` [95% CI])

| Method / Arm | $d_v$ | `E2 (full)` | `E3-sybil (full)` | `E3-collude (full)` | `E3-biased (full)` |
|---|---:|---|---|---|---|
| `bayes` | — | `0.0000 [0.0000, 0.0000]` | `0.0000 [0.0000, 0.0000]` | `0.0000 [0.0000, 0.0000]` | `0.0000 [0.0000, 0.0000]` |
| `unlearned_0param_dv12` (0-param coordinate-aligned skeleton) | 12 | `0.0000` | `0.0000` | `0.0000` | `0.0732` |
| `oracle_clearing` | 1 | `0.0166 [0.0095, 0.0238]` | `0.0166 [0.0095, 0.0238]` | `0.0175 [0.0098, 0.0245]` | `0.0166 [0.0095, 0.0238]` |
| `extremized_log_pool` | 1 | `0.0448 [0.0328, 0.0565]` | `0.0778 [0.0623, 0.0919]` | `0.0302 [0.0205, 0.0394]` | `0.0763 [0.0604, 0.0910]` |
| `log_pool` | 1 | `0.1124 [0.0988, 0.1248]` | `0.1261 [0.1112, 0.1395]` | `0.0463 [0.0358, 0.0557]` | `0.0913 [0.0768, 0.1047]` |
| `linear_pool` | 1 | `0.1517 [0.1375, 0.1655]` | `0.1587 [0.1432, 0.1727]` | `0.0744 [0.0629, 0.0848]` | `0.1425 [0.1281, 0.1558]` |
| `sum_logodds` | 1 | `0.1324 [0.1005, 0.1666]` | `0.4170 [0.3449, 0.4952]` | `0.2589 [0.2055, 0.3141]` | `0.3410 [0.2780, 0.4070]` |
| `naive_mha_dv1:trained` | 1 | `0.0962 [0.0792, 0.1136]` | `0.1048 [0.0871, 0.1227]` | `0.0913 [0.0745, 0.1087]` | `0.1106 [0.0919, 0.1294]` |
| `naive_mha_dv12:trained` | 12 | `0.0965 [0.0788, 0.1141]` | `0.1069 [0.0880, 0.1258]` | `0.0914 [0.0737, 0.1098]` | `0.1083 [0.0896, 0.1268]` |
| `slot_kv_dv1:trained` | 1 | `0.0381 [0.0273, 0.0487]` | `0.0379 [0.0271, 0.0484]` | `0.0389 [0.0279, 0.0504]` | `0.0614 [0.0485, 0.0748]` |
| `slot_kv_dv2:trained` | 2 | `0.0187 [0.0105, 0.0272]` | `0.0184 [0.0102, 0.0268]` | `0.0249 [0.0160, 0.0345]` | `0.0381 [0.0269, 0.0494]` |
| `slot_kv_dv4:trained` | 4 | `0.0106 [0.0045, 0.0171]` | `0.0103 [0.0043, 0.0167]` | `0.0131 [0.0060, 0.0208]` | `0.0202 [0.0119, 0.0287]` |
| `slot_kv_dv8:trained` | 8 | `0.0073 [0.0025, 0.0123]` | `0.0070 [0.0023, 0.0119]` | `0.0089 [0.0039, 0.0140]` | `0.0156 [0.0083, 0.0228]` |
| `slot_kv_dv12:trained` | 12 | `0.0040 [0.0007, 0.0074]` | `0.0040 [0.0007, 0.0072]` | `0.0045 [0.0010, 0.0080]` | `0.0111 [0.0052, 0.0172]` |

### Paired Regret Differences at $d_v = 12$ (`slot_kv_dv12:trained`, `visibility = full`)

| Environment | Status | `slot_kv_dv12:trained` Regret | $\Delta$ vs `oracle_clearing` (95% CI) | $\Delta$ vs `extremized_log_pool` (95% CI) | `untrained_init` Control | `random_network` Control |
|---|---|---|---|---|---|---|
| `E2` | `● Gain` | `0.0040 [0.0007, 0.0074]` | `-0.0126 [-0.0202, -0.0052]` | `-0.0408 [-0.0528, -0.0290]` | `0.5062` | `0.4896` |
| `E3-sybil` | `● Gain` | `0.0040 [0.0007, 0.0072]` | `-0.0126 [-0.0202, -0.0053]` | `-0.0738 [-0.0877, -0.0582]` | `0.5069` | `0.5520` |
| `E3-collude` | `● Gain` | `0.0045 [0.0010, 0.0080]` | `-0.0130 [-0.0206, -0.0052]` | `-0.0257 [-0.0352, -0.0154]` | `0.5050` | `0.6157` |
| `E3-biased` | `● Informative` | `0.0111 [0.0052, 0.0172]` | `-0.0055 [-0.0143, +0.0033]` | `-0.0652 [-0.0796, -0.0495]` | `0.5651` | `0.5421` |

## Master Comparison Table: `visibility = noisy` (`dev`, $n=2000$, Mean Regret to `bayes` [95% CI])

| Method / Arm | $d_v$ | `E2 (noisy)` | `E3-sybil (noisy)` | `E3-collude (noisy)` | `E3-biased (noisy)` |
|---|---:|---|---|---|---|
| `bayes` | — | `0.0000 [0.0000, 0.0000]` | `0.0000 [0.0000, 0.0000]` | `0.0000 [0.0000, 0.0000]` | `0.0000 [0.0000, 0.0000]` |
| `oracle_clearing` | 1 | `0.0166 [0.0095, 0.0238]` | `0.0166 [0.0095, 0.0238]` | `0.0175 [0.0098, 0.0245]` | `0.0166 [0.0095, 0.0238]` |
| `unlearned_0param_dv12` (0-param coordinate-aligned skeleton) | 12 | `0.0386` | `0.0216` | `0.0223` | `0.1094` |
| `extremized_log_pool` | 1 | `0.0448 [0.0328, 0.0565]` | `0.0778 [0.0623, 0.0919]` | `0.0302 [0.0205, 0.0394]` | `0.0763 [0.0604, 0.0910]` |
| `log_pool` | 1 | `0.1124 [0.0988, 0.1248]` | `0.1261 [0.1112, 0.1395]` | `0.0463 [0.0358, 0.0557]` | `0.0913 [0.0768, 0.1047]` |
| `linear_pool` | 1 | `0.1517 [0.1375, 0.1655]` | `0.1587 [0.1432, 0.1727]` | `0.0744 [0.0629, 0.0848]` | `0.1425 [0.1281, 0.1558]` |
| `sum_logodds` | 1 | `0.1324 [0.1005, 0.1666]` | `0.4170 [0.3449, 0.4952]` | `0.2589 [0.2055, 0.3141]` | `0.3410 [0.2780, 0.4070]` |
| `naive_mha_dv1:trained` | 1 | `0.0986 [0.0811, 0.1163]` | `0.1271 [0.1066, 0.1473]` | `0.0912 [0.0746, 0.1085]` | `0.1117 [0.0931, 0.1309]` |
| `naive_mha_dv12:trained` | 12 | `0.0992 [0.0814, 0.1171]` | `0.1088 [0.0894, 0.1284]` | `0.0907 [0.0736, 0.1087]` | `0.1090 [0.0901, 0.1280]` |
| `slot_kv_dv1:trained` | 1 | `0.0502 [0.0389, 0.0623]` | `0.0465 [0.0350, 0.0583]` | `0.0449 [0.0335, 0.0570]` | `0.0672 [0.0538, 0.0806]` |
| `slot_kv_dv2:trained` | 2 | `0.0315 [0.0223, 0.0411]` | `0.0264 [0.0169, 0.0362]` | `0.0309 [0.0210, 0.0413]` | `0.0465 [0.0347, 0.0584]` |
| `slot_kv_dv4:trained` | 4 | `0.0243 [0.0161, 0.0324]` | `0.0191 [0.0112, 0.0272]` | `0.0221 [0.0137, 0.0309]` | `0.0328 [0.0228, 0.0427]` |
| `slot_kv_dv8:trained` | 8 | `0.0220 [0.0146, 0.0294]` | `0.0130 [0.0063, 0.0199]` | `0.0181 [0.0112, 0.0247]` | `0.0292 [0.0199, 0.0379]` |
| `slot_kv_dv12:trained` | 12 | `0.0178 [0.0113, 0.0246]` | `0.0107 [0.0045, 0.0170]` | `0.0124 [0.0065, 0.0182]` | `0.0232 [0.0146, 0.0311]` |

### Paired Regret Differences at $d_v = 12$ (`slot_kv_dv12:trained`, `visibility = noisy`)

| Environment | Status | `slot_kv_dv12:trained` Regret | $\Delta$ vs `oracle_clearing` (95% CI) | $\Delta$ vs `extremized_log_pool` (95% CI) | `untrained_init` Control | `random_network` Control |
|---|---|---|---|---|---|---|
| `E2` | `● Informative` | `0.0178 [0.0113, 0.0246]` | `+0.0012 [-0.0084, +0.0110]` | `-0.0270 [-0.0396, -0.0143]` | `0.5119` | `0.4583` |
| `E3-sybil` | `● Informative` | `0.0107 [0.0045, 0.0170]` | `-0.0059 [-0.0153, +0.0030]` | `-0.0671 [-0.0820, -0.0499]` | `0.5081` | `0.5063` |
| `E3-collude` | `● Informative` | `0.0124 [0.0065, 0.0182]` | `-0.0051 [-0.0139, +0.0036]` | `-0.0178 [-0.0285, -0.0069]` | `0.5063` | `0.5548` |
| `E3-biased` | `● Informative` | `0.0232 [0.0146, 0.0311]` | `+0.0066 [-0.0039, +0.0166]` | `-0.0531 [-0.0677, -0.0375]` | `0.5363` | `0.4981` |

## Sample-Size Scaling on `E2 (full)` (`slot_kv_dv12:trained`, $d_v=12$)

| $N_{\text{train}}$ | Regret to `bayes` (`0.000`) | $\Delta$ vs `oracle_clearing` (`0.0166`) | Status |
|---:|---|---|---|
| `500` | `0.0241 [0.0155, 0.0328]` | `+0.0075 [-0.0035, +0.0185]` | ● Informative |
| `2000` | `0.0040 [0.0007, 0.0074]` | `-0.0126 [-0.0202, -0.0052]` | ● Gain |
| `8000` | `0.0042 [0.0010, 0.0074]` | `-0.0124 [-0.0203, -0.0046]` | ● Gain |

## Per-Cell Artifacts

- [`results/explore/agg_highdim_e2_full/`](../agg_highdim_e2_full/README.md) (`result.json` + `README.md`)
- [`results/explore/agg_highdim_e2_noisy/`](../agg_highdim_e2_noisy/README.md) (`result.json` + `README.md`)
- [`results/explore/agg_highdim_e3_sybil_full/`](../agg_highdim_e3_sybil_full/README.md) (`result.json` + `README.md`)
- [`results/explore/agg_highdim_e3_sybil_noisy/`](../agg_highdim_e3_sybil_noisy/README.md) (`result.json` + `README.md`)
- [`results/explore/agg_highdim_e3_collude_full/`](../agg_highdim_e3_collude_full/README.md) (`result.json` + `README.md`)
- [`results/explore/agg_highdim_e3_collude_noisy/`](../agg_highdim_e3_collude_noisy/README.md) (`result.json` + `README.md`)
- [`results/explore/agg_highdim_e3_biased_full/`](../agg_highdim_e3_biased_full/README.md) (`result.json` + `README.md`)
- [`results/explore/agg_highdim_e3_biased_noisy/`](../agg_highdim_e3_biased_noisy/README.md) (`result.json` + `README.md`)
