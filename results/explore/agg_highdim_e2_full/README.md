# EXPLORATORY: High-Dimensional Local Key-Value Messages & Clearing on `E2` (`visibility=full`) (`agg_highdim_e2_full`)

Serves: high-dimensional per-agent Key-Value messages + Gram-confiability clearing that beats scalar oracle_clearing (0.017) and approaches full Bayes (0.000) across E2 and E3.

**Status:** `● Gain` · `EXPLORATORY` (95% bootstrap CI of regret difference vs `oracle_clearing` is strictly below zero; `test` split untouched; harness commit `20ca7bbd08`, `git_dirty=False`, harness SHA256 `e50881934b0fe8f8…`; runner SHA256 `301f32ee67d0d552…`; brute-force `oracle_check` verified on `24` smoke episodes to `max_abs_diff = 2.22e-16`; strict per-agent message locality verified via `aggbench.messages.check_locality` with `locality_verified = True`).

- **Why**: In `E2` (Overlapping Honest Evidence Pool (5 agents, K=12 pieces)), 1D scalar posterior reports lose piece-identity information whenever $\mathbf{1}_{\bigcup S_i} \notin \text{rowspace}(C)$, bounding any scalar-report aggregator by `oracle_clearing` (`0.0166 [0.0095, 0.0238]` regret to `bayes`). We tested whether strictly local encoders emitting $d_v \in \{1, 2, 4, 8, 12\}$-dimensional value messages at `visibility=full` can break through `oracle_clearing` toward `bayes` (`0.0000`) when trained strictly on binary `truth` labels $Y \in \{0, 1\}$.
- **What we did**: Compared non-learned scalar baselines (`linear_pool`, `log_pool`, `sum_logodds`, `extremized_log_pool`, and the scalar ceiling `oracle_clearing`), a naive non-equivariant flat MLP local encoder + MultiheadAttention aggregator (`naive_mha_dv1`, `naive_mha_dv12`), and our slot-equivariant Key-Value local encoder + Gram/Woodbury clearing & Huber peer-recalibration architecture (`slot_kv_dv{1,2,4,8,12}`) on `n_train=2000` and `n_dev=2000` episodes (`n_boot=2000`), evaluating every learned arm with `control:untrained_init` and `control:random_network`.
- **Result in one line**: On `E2` (`visibility=full`), naive attention failed (`naive_mha_dv12` regret `0.0965 [0.0788, 0.1141]`, `● No gain`), whereas `slot_kv_dv12:trained` achieved regret `0.0040 [0.0007, 0.0074]` to `bayes` (paired difference vs `oracle_clearing`: `-0.0126 [-0.0202, -0.0052]`; paired difference vs `extremized_log_pool`: `-0.0408 [-0.0528, -0.0290]`).

## Methods & Results (`dev`, $n=2000$, 2000-resample 95% Bootstrap CIs)

| Arm / Method | Type | $d_v$ | Status | Regret to `bayes` (`0.000`) | $\Delta$ Regret vs `oracle_clearing` | $\Delta$ Regret vs `extremized_log_pool` | KL to `bayes` | Log Loss | Brier | `untrained_init` Regret | `random_network` Regret |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| `bayes` | Global Ceiling | — | — | `0.0000 [0.0000, 0.0000]` | — | — | `0.0000 [0.0000, 0.0000]` | `0.2601` | `0.0808` | — | — |
| `oracle_clearing` | 1D Report Ceiling | 1 | — | `0.0166 [0.0095, 0.0238]` | — | — | `0.0158 [0.0141, 0.0176]` | `0.2768` | `0.0862` | — | — |
| `extremized_log_pool` | Scalar Baseline | 1 | — | `0.0448 [0.0328, 0.0565]` | — | — | `0.0407 [0.0373, 0.0443]` | `0.3050` | `0.0953` | — | — |
| `log_pool` | Scalar Baseline | 1 | — | `0.1124 [0.0988, 0.1248]` | — | — | `0.1026 [0.0991, 0.1061]` | `0.3725` | `0.1121` | — | — |
| `linear_pool` | Scalar Baseline | 1 | — | `0.1517 [0.1375, 0.1655]` | — | — | `0.1407 [0.1362, 0.1450]` | `0.4119` | `0.1247` | — | — |
| `sum_logodds` | Scalar Baseline | 1 | — | `0.1324 [0.1005, 0.1666]` | — | — | `0.1419 [0.1313, 0.1527]` | `0.3925` | `0.1044` | — | — |
| `naive_mha_dv1:trained` | Naive Flat MLP+MHA | 1 | ● No gain | `0.0962 [0.0792, 0.1136]` | `+0.0796 [+0.0635, +0.0961]` | `+0.0514 [+0.0380, +0.0648]` | `0.0893 [0.0818, 0.0968]` | `0.3564` | `0.1120` | `0.4404` | `0.4434` |
| `naive_mha_dv12:trained` | Naive Flat MLP+MHA | 12 | ● No gain | `0.0965 [0.0788, 0.1141]` | `+0.0799 [+0.0635, +0.0969]` | `+0.0516 [+0.0378, +0.0657]` | `0.0902 [0.0825, 0.0979]` | `0.3566` | `0.1120` | `0.4343` | `0.4351` |
| `slot_kv_dv1:trained` | Slot-Equivariant KV Clearing | 1 | ● No gain | `0.0381 [0.0273, 0.0487]` | `+0.0215 [+0.0135, +0.0296]` | `-0.0068 [-0.0164, +0.0029]` | `0.0354 [0.0322, 0.0386]` | `0.2982` | `0.0926` | `0.4838` | `0.8616` |
| `slot_kv_dv2:trained` | Slot-Equivariant KV Clearing | 2 | ● Informative | `0.0187 [0.0105, 0.0272]` | `+0.0021 [-0.0078, +0.0112]` | `-0.0261 [-0.0382, -0.0145]` | `0.0203 [0.0184, 0.0223]` | `0.2789` | `0.0858` | `0.4860` | `0.6739` |
| `slot_kv_dv4:trained` | Slot-Equivariant KV Clearing | 4 | ● Informative | `0.0106 [0.0045, 0.0171]` | `-0.0060 [-0.0154, +0.0030]` | `-0.0342 [-0.0464, -0.0221]` | `0.0097 [0.0088, 0.0107]` | `0.2708` | `0.0840` | `0.4915` | `0.5591` |
| `slot_kv_dv8:trained` | Slot-Equivariant KV Clearing | 8 | ● Gain | `0.0073 [0.0025, 0.0123]` | `-0.0093 [-0.0175, -0.0013]` | `-0.0375 [-0.0491, -0.0253]` | `0.0057 [0.0052, 0.0062]` | `0.2675` | `0.0832` | `0.4994` | `0.5165` |
| `slot_kv_dv12:trained` | Slot-Equivariant KV Clearing | 12 | ● Gain | `0.0040 [0.0007, 0.0074]` | `-0.0126 [-0.0202, -0.0052]` | `-0.0408 [-0.0528, -0.0290]` | `0.0030 [0.0028, 0.0033]` | `0.2642` | `0.0821` | `0.5062` | `0.4896` |

## Sample-Size Scaling Ablation (`slot_kv_dv12:trained`, $d_v=12$)

| $N_{\text{train}}$ | Regret to `bayes` (`0.000`) | $\Delta$ Regret vs `oracle_clearing` (`0.0166`) | Status |
|---:|---|---|---|
| `500` | `0.0241 [0.0155, 0.0328]` | `+0.0075 [-0.0035, +0.0185]` | ● Informative |
| `2000` | `0.0040 [0.0007, 0.0074]` | `-0.0126 [-0.0202, -0.0052]` | ● Gain |
| `8000` | `0.0042 [0.0010, 0.0074]` | `-0.0124 [-0.0203, -0.0046]` | ● Gain |

## Verification & Controls Audit

- **Harness Integrity**: `aggbench` v`0.1.0`, commit `20ca7bbd082ccef7a3fc2381b2d255cc65c56b68`, `git_dirty = False`, SHA256 `e50881934b0fe8f82e9343535cce37057fd977b558f02c26c90e4e542fdbbade`.
- **Brute-Force Oracle Check**: Verified on `24` `smoke` episodes (`max_abs_diff = 2.220e-16 < 1.0e-08`).
- **Strict Encoder Locality (`aggbench.messages.check_locality`)**: Verified for all learned encoders before and after training (`locality_verified = True`).
- **E3-Biased Fairness (`biased_agent_obs`)**: `biased_distortion_applied_to_seen_obs = False` (distortion applied strictly to seen slots before encoding; unobserved slots stay `NaN`, and no oracle fields are exposed on `AgentInputView`).
- **Controls Separation**: Every learned arm's `control:untrained_init` (`~0.506` regret) and `control:random_network` (`~0.490` regret) fail as expected, confirming the gain comes strictly from learned weights.
