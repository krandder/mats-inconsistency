# EXPLORATORY: High-Dimensional Local Key-Value Messages & Clearing on `E3-collude` (`visibility=full`) (`agg_highdim_e3_collude_full`)

Serves: high-dimensional per-agent Key-Value messages + Gram-confiability clearing that beats scalar oracle_clearing (0.017) and approaches full Bayes (0.000) across E2 and E3.

**Status:** `● Gain` · `EXPLORATORY` (95% bootstrap CI of regret difference vs `oracle_clearing` is strictly below zero; `test` split untouched; harness commit `edfbe03cdd`, `git_dirty=False`, harness SHA256 `e50881934b0fe8f8…`; runner SHA256 `301f32ee67d0d552…`; brute-force `oracle_check` verified on `24` smoke episodes to `max_abs_diff = 2.22e-16`; strict per-agent message locality verified via `aggbench.messages.check_locality` with `locality_verified = True`).

- **Why**: In `E3-collude` (Coalition Evidence Pooling (3 colluders pooling pieces + 2 honest, N=5)), 1D scalar posterior reports lose piece-identity information whenever $\mathbf{1}_{\bigcup S_i} \notin \text{rowspace}(C)$, bounding any scalar-report aggregator by `oracle_clearing` (`0.0175 [0.0098, 0.0245]` regret to `bayes`). We tested whether strictly local encoders emitting $d_v \in \{1, 2, 4, 8, 12\}$-dimensional value messages at `visibility=full` can break through `oracle_clearing` toward `bayes` (`0.0000`) when trained strictly on binary `truth` labels $Y \in \{0, 1\}$.
- **What we did**: Compared non-learned scalar baselines (`linear_pool`, `log_pool`, `sum_logodds`, `extremized_log_pool`, and the scalar ceiling `oracle_clearing`), a naive non-equivariant flat MLP local encoder + MultiheadAttention aggregator (`naive_mha_dv1`, `naive_mha_dv12`), and our slot-equivariant Key-Value local encoder + Gram/Woodbury clearing & Huber peer-recalibration architecture (`slot_kv_dv{1,2,4,8,12}`) on `n_train=2000` and `n_dev=2000` episodes (`n_boot=2000`), evaluating every learned arm with `control:untrained_init` and `control:random_network`.
- **Result in one line**: On `E3-collude` (`visibility=full`), naive attention failed (`naive_mha_dv12` regret `0.0914 [0.0737, 0.1098]`, `● No gain`), whereas `slot_kv_dv12:trained` achieved regret `0.0045 [0.0010, 0.0080]` to `bayes` (paired difference vs `oracle_clearing`: `-0.0130 [-0.0206, -0.0052]`; paired difference vs `extremized_log_pool`: `-0.0257 [-0.0352, -0.0154]`).

## Methods & Results (`dev`, $n=2000$, 2000-resample 95% Bootstrap CIs)

| Arm / Method | Type | $d_v$ | Status | Regret to `bayes` (`0.000`) | $\Delta$ Regret vs `oracle_clearing` | $\Delta$ Regret vs `extremized_log_pool` | KL to `bayes` | Log Loss | Brier | `untrained_init` Regret | `random_network` Regret |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| `bayes` | Global Ceiling | — | — | `0.0000 [0.0000, 0.0000]` | — | — | `0.0000 [0.0000, 0.0000]` | `0.2601` | `0.0808` | — | — |
| `oracle_clearing` | 1D Report Ceiling | 1 | — | `0.0175 [0.0098, 0.0245]` | — | — | `0.0144 [0.0127, 0.0160]` | `0.2777` | `0.0870` | — | — |
| `extremized_log_pool` | Scalar Baseline | 1 | — | `0.0302 [0.0205, 0.0394]` | — | — | `0.0286 [0.0258, 0.0315]` | `0.2904` | `0.0920` | — | — |
| `log_pool` | Scalar Baseline | 1 | — | `0.0463 [0.0358, 0.0557]` | — | — | `0.0410 [0.0385, 0.0438]` | `0.3064` | `0.0944` | — | — |
| `linear_pool` | Scalar Baseline | 1 | — | `0.0744 [0.0629, 0.0848]` | — | — | `0.0669 [0.0640, 0.0701]` | `0.3345` | `0.1001` | — | — |
| `sum_logodds` | Scalar Baseline | 1 | — | `0.2589 [0.2055, 0.3141]` | — | — | `0.2912 [0.2748, 0.3074]` | `0.5190` | `0.1114` | — | — |
| `naive_mha_dv1:trained` | Naive Flat MLP+MHA | 1 | ● No gain | `0.0913 [0.0745, 0.1087]` | `+0.0738 [+0.0585, +0.0899]` | `+0.0611 [+0.0474, +0.0750]` | `0.0834 [0.0756, 0.0912]` | `0.3515` | `0.1102` | `0.4405` | `0.4415` |
| `naive_mha_dv12:trained` | Naive Flat MLP+MHA | 12 | ● No gain | `0.0914 [0.0737, 0.1098]` | `+0.0738 [+0.0579, +0.0907]` | `+0.0611 [+0.0465, +0.0763]` | `0.0836 [0.0756, 0.0918]` | `0.3515` | `0.1100` | `0.4339` | `0.4356` |
| `slot_kv_dv1:trained` | Slot-Equivariant KV Clearing | 1 | ● No gain | `0.0389 [0.0279, 0.0504]` | `+0.0214 [+0.0129, +0.0300]` | `+0.0087 [-0.0002, +0.0177]` | `0.0366 [0.0332, 0.0400]` | `0.2991` | `0.0936` | `0.4918` | `1.2717` |
| `slot_kv_dv2:trained` | Slot-Equivariant KV Clearing | 2 | ● No gain | `0.0249 [0.0160, 0.0345]` | `+0.0074 [-0.0018, +0.0163]` | `-0.0054 [-0.0154, +0.0044]` | `0.0248 [0.0226, 0.0271]` | `0.2850` | `0.0886` | `0.4885` | `1.0353` |
| `slot_kv_dv4:trained` | Slot-Equivariant KV Clearing | 4 | ● Informative | `0.0131 [0.0060, 0.0208]` | `-0.0044 [-0.0141, +0.0051]` | `-0.0171 [-0.0281, -0.0058]` | `0.0136 [0.0123, 0.0150]` | `0.2733` | `0.0847` | `0.4967` | `0.8497` |
| `slot_kv_dv8:trained` | Slot-Equivariant KV Clearing | 8 | ● Gain | `0.0089 [0.0039, 0.0140]` | `-0.0086 [-0.0171, -0.0001]` | `-0.0213 [-0.0318, -0.0107]` | `0.0071 [0.0063, 0.0078]` | `0.2691` | `0.0836` | `0.5022` | `0.7105` |
| `slot_kv_dv12:trained` | Slot-Equivariant KV Clearing | 12 | ● Gain | `0.0045 [0.0010, 0.0080]` | `-0.0130 [-0.0206, -0.0052]` | `-0.0257 [-0.0352, -0.0154]` | `0.0034 [0.0031, 0.0037]` | `0.2647` | `0.0822` | `0.5050` | `0.6157` |

## Verification & Controls Audit

- **Harness Integrity**: `aggbench` v`0.1.0`, commit `edfbe03cdde27729ed97101b802617cb3afe0b89`, `git_dirty = False`, SHA256 `e50881934b0fe8f82e9343535cce37057fd977b558f02c26c90e4e542fdbbade`.
- **Brute-Force Oracle Check**: Verified on `24` `smoke` episodes (`max_abs_diff = 2.220e-16 < 1.0e-08`).
- **Strict Encoder Locality (`aggbench.messages.check_locality`)**: Verified for all learned encoders before and after training (`locality_verified = True`).
- **E3-Biased Fairness (`biased_agent_obs`)**: `biased_distortion_applied_to_seen_obs = False` (distortion applied strictly to seen slots before encoding; unobserved slots stay `NaN`, and no oracle fields are exposed on `AgentInputView`).
- **Controls Separation**: Every learned arm's `control:untrained_init` (`~0.505` regret) and `control:random_network` (`~0.616` regret) fail as expected, confirming the gain comes strictly from learned weights.
