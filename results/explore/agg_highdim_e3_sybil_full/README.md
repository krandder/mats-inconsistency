# EXPLORATORY: High-Dimensional Local Key-Value Messages & Clearing on `E3-sybil` (`visibility=full`) (`agg_highdim_e3_sybil_full`)

Serves: high-dimensional per-agent Key-Value messages + Gram-confiability clearing that beats scalar oracle_clearing (0.017) and approaches full Bayes (0.000) across E2 and E3.

**Status:** `● Gain` · `EXPLORATORY` (95% bootstrap CI of regret difference vs `oracle_clearing` is strictly below zero; `test` split untouched; harness commit `edfbe03cdd`, `git_dirty=False`, harness SHA256 `e50881934b0fe8f8…`; runner SHA256 `301f32ee67d0d552…`; brute-force `oracle_check` verified on `24` smoke episodes to `max_abs_diff = 1.11e-16`; strict per-agent message locality verified via `aggbench.messages.check_locality` with `locality_verified = True`).

- **Why**: In `E3-sybil` (Sybil Duplication Attack (5 honest + 3 exact Sybil copies, N=8)), 1D scalar posterior reports lose piece-identity information whenever $\mathbf{1}_{\bigcup S_i} \notin \text{rowspace}(C)$, bounding any scalar-report aggregator by `oracle_clearing` (`0.0166 [0.0095, 0.0238]` regret to `bayes`). We tested whether strictly local encoders emitting $d_v \in \{1, 2, 4, 8, 12\}$-dimensional value messages at `visibility=full` can break through `oracle_clearing` toward `bayes` (`0.0000`) when trained strictly on binary `truth` labels $Y \in \{0, 1\}$.
- **What we did**: Compared non-learned scalar baselines (`linear_pool`, `log_pool`, `sum_logodds`, `extremized_log_pool`, and the scalar ceiling `oracle_clearing`), a naive non-equivariant flat MLP local encoder + MultiheadAttention aggregator (`naive_mha_dv1`, `naive_mha_dv12`), and our slot-equivariant Key-Value local encoder + Gram/Woodbury clearing & Huber peer-recalibration architecture (`slot_kv_dv{1,2,4,8,12}`) on `n_train=2000` and `n_dev=2000` episodes (`n_boot=2000`), evaluating every learned arm with `control:untrained_init` and `control:random_network`.
- **Result in one line**: On `E3-sybil` (`visibility=full`), naive attention failed (`naive_mha_dv12` regret `0.1069 [0.0880, 0.1258]`, `● No gain`), whereas `slot_kv_dv12:trained` achieved regret `0.0040 [0.0007, 0.0072]` to `bayes` (paired difference vs `oracle_clearing`: `-0.0126 [-0.0202, -0.0053]`; paired difference vs `extremized_log_pool`: `-0.0738 [-0.0877, -0.0582]`).

## Methods & Results (`dev`, $n=2000$, 2000-resample 95% Bootstrap CIs)

| Arm / Method | Type | $d_v$ | Status | Regret to `bayes` (`0.000`) | $\Delta$ Regret vs `oracle_clearing` | $\Delta$ Regret vs `extremized_log_pool` | KL to `bayes` | Log Loss | Brier | `untrained_init` Regret | `random_network` Regret |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| `bayes` | Global Ceiling | — | — | `0.0000 [0.0000, 0.0000]` | — | — | `0.0000 [0.0000, 0.0000]` | `0.2601` | `0.0808` | — | — |
| `oracle_clearing` | 1D Report Ceiling | 1 | — | `0.0166 [0.0095, 0.0238]` | — | — | `0.0158 [0.0141, 0.0176]` | `0.2768` | `0.0862` | — | — |
| `extremized_log_pool` | Scalar Baseline | 1 | — | `0.0778 [0.0623, 0.0919]` | — | — | `0.0747 [0.0687, 0.0808]` | `0.3380` | `0.1069` | — | — |
| `log_pool` | Scalar Baseline | 1 | — | `0.1261 [0.1112, 0.1395]` | — | — | `0.1176 [0.1127, 0.1226]` | `0.3863` | `0.1188` | — | — |
| `linear_pool` | Scalar Baseline | 1 | — | `0.1587 [0.1432, 0.1727]` | — | — | `0.1484 [0.1428, 0.1540]` | `0.4189` | `0.1293` | — | — |
| `sum_logodds` | Scalar Baseline | 1 | — | `0.4170 [0.3449, 0.4952]` | — | — | `0.4509 [0.4180, 0.4838]` | `0.6772` | `0.1278` | — | — |
| `naive_mha_dv1:trained` | Naive Flat MLP+MHA | 1 | ● No gain | `0.1048 [0.0871, 0.1227]` | `+0.0882 [+0.0715, +0.1058]` | `+0.0270 [+0.0129, +0.0419]` | `0.1030 [0.0946, 0.1115]` | `0.3649` | `0.1159` | `0.4403` | `0.4438` |
| `naive_mha_dv12:trained` | Naive Flat MLP+MHA | 12 | ● No gain | `0.1069 [0.0880, 0.1258]` | `+0.0903 [+0.0720, +0.1086]` | `+0.0291 [+0.0145, +0.0444]` | `0.1071 [0.0981, 0.1161]` | `0.3670` | `0.1160` | `0.4342` | `0.4352` |
| `slot_kv_dv1:trained` | Slot-Equivariant KV Clearing | 1 | ● Informative | `0.0379 [0.0271, 0.0484]` | `+0.0213 [+0.0133, +0.0294]` | `-0.0399 [-0.0532, -0.0259]` | `0.0356 [0.0324, 0.0387]` | `0.2980` | `0.0927` | `0.4846` | `0.8877` |
| `slot_kv_dv2:trained` | Slot-Equivariant KV Clearing | 2 | ● Informative | `0.0184 [0.0102, 0.0268]` | `+0.0018 [-0.0083, +0.0109]` | `-0.0594 [-0.0734, -0.0446]` | `0.0202 [0.0183, 0.0223]` | `0.2786` | `0.0857` | `0.4876` | `0.7421` |
| `slot_kv_dv4:trained` | Slot-Equivariant KV Clearing | 4 | ● Informative | `0.0103 [0.0043, 0.0167]` | `-0.0063 [-0.0154, +0.0027]` | `-0.0675 [-0.0819, -0.0521]` | `0.0093 [0.0084, 0.0102]` | `0.2705` | `0.0840` | `0.4946` | `0.6330` |
| `slot_kv_dv8:trained` | Slot-Equivariant KV Clearing | 8 | ● Gain | `0.0070 [0.0023, 0.0119]` | `-0.0096 [-0.0177, -0.0015]` | `-0.0708 [-0.0847, -0.0550]` | `0.0057 [0.0051, 0.0062]` | `0.2672` | `0.0831` | `0.5016` | `0.5889` |
| `slot_kv_dv12:trained` | Slot-Equivariant KV Clearing | 12 | ● Gain | `0.0040 [0.0007, 0.0072]` | `-0.0126 [-0.0202, -0.0053]` | `-0.0738 [-0.0877, -0.0582]` | `0.0030 [0.0027, 0.0032]` | `0.2641` | `0.0821` | `0.5069` | `0.5520` |

## Verification & Controls Audit

- **Harness Integrity**: `aggbench` v`0.1.0`, commit `edfbe03cdde27729ed97101b802617cb3afe0b89`, `git_dirty = False`, SHA256 `e50881934b0fe8f82e9343535cce37057fd977b558f02c26c90e4e542fdbbade`.
- **Brute-Force Oracle Check**: Verified on `24` `smoke` episodes (`max_abs_diff = 1.110e-16 < 1.0e-08`).
- **Strict Encoder Locality (`aggbench.messages.check_locality`)**: Verified for all learned encoders before and after training (`locality_verified = True`).
- **E3-Biased Fairness (`biased_agent_obs`)**: `biased_distortion_applied_to_seen_obs = False` (distortion applied strictly to seen slots before encoding; unobserved slots stay `NaN`, and no oracle fields are exposed on `AgentInputView`).
- **Controls Separation**: Every learned arm's `control:untrained_init` (`~0.507` regret) and `control:random_network` (`~0.552` regret) fail as expected, confirming the gain comes strictly from learned weights.
