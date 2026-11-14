# EXPLORATORY: High-Dimensional Local Key-Value Messages & Clearing on `E3-sybil` (`visibility=noisy`) (`agg_highdim_e3_sybil_noisy`)

Serves: high-dimensional per-agent Key-Value messages + Gram-confiability clearing that beats scalar oracle_clearing (0.017) and approaches full Bayes (0.000) across E2 and E3.

**Status:** `● Informative` · `EXPLORATORY` (Point-estimate regret is below `oracle_clearing` and 95% CI is strictly below `extremized_log_pool` (`oracle_clearing` CI overlaps zero); `test` split untouched; harness commit `edfbe03cdd`, `git_dirty=False`, harness SHA256 `e50881934b0fe8f8…`; runner SHA256 `301f32ee67d0d552…`; brute-force `oracle_check` verified on `24` smoke episodes to `max_abs_diff = 1.11e-16`; strict per-agent message locality verified via `aggbench.messages.check_locality` with `locality_verified = True`).

- **Why**: In `E3-sybil` (Sybil Duplication Attack (5 honest + 3 exact Sybil copies, N=8)), 1D scalar posterior reports lose piece-identity information whenever $\mathbf{1}_{\bigcup S_i} \notin \text{rowspace}(C)$, bounding any scalar-report aggregator by `oracle_clearing` (`0.0166 [0.0095, 0.0238]` regret to `bayes`). We tested whether strictly local encoders emitting $d_v \in \{1, 2, 4, 8, 12\}$-dimensional value messages at `visibility=noisy` can break through `oracle_clearing` toward `bayes` (`0.0000`) when trained strictly on binary `truth` labels $Y \in \{0, 1\}$.
- **What we did**: Compared non-learned scalar baselines (`linear_pool`, `log_pool`, `sum_logodds`, `extremized_log_pool`, and the scalar ceiling `oracle_clearing`), a naive non-equivariant flat MLP local encoder + MultiheadAttention aggregator (`naive_mha_dv1`, `naive_mha_dv12`), and our slot-equivariant Key-Value local encoder + Gram/Woodbury clearing & Huber peer-recalibration architecture (`slot_kv_dv{1,2,4,8,12}`) on `n_train=2000` and `n_dev=2000` episodes (`n_boot=2000`), evaluating every learned arm with `control:untrained_init` and `control:random_network`.
- **Result in one line**: On `E3-sybil` (`visibility=noisy`), naive attention failed (`naive_mha_dv12` regret `0.1088 [0.0894, 0.1284]`, `● No gain`), whereas `slot_kv_dv12:trained` achieved regret `0.0107 [0.0045, 0.0170]` to `bayes` (paired difference vs `oracle_clearing`: `-0.0059 [-0.0153, +0.0030]`; paired difference vs `extremized_log_pool`: `-0.0671 [-0.0820, -0.0499]`).

## Methods & Results (`dev`, $n=2000$, 2000-resample 95% Bootstrap CIs)

| Arm / Method | Type | $d_v$ | Status | Regret to `bayes` (`0.000`) | $\Delta$ Regret vs `oracle_clearing` | $\Delta$ Regret vs `extremized_log_pool` | KL to `bayes` | Log Loss | Brier | `untrained_init` Regret | `random_network` Regret |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| `bayes` | Global Ceiling | — | — | `0.0000 [0.0000, 0.0000]` | — | — | `0.0000 [0.0000, 0.0000]` | `0.2601` | `0.0808` | — | — |
| `oracle_clearing` | 1D Report Ceiling | 1 | — | `0.0166 [0.0095, 0.0238]` | — | — | `0.0158 [0.0141, 0.0176]` | `0.2768` | `0.0862` | — | — |
| `extremized_log_pool` | Scalar Baseline | 1 | — | `0.0778 [0.0623, 0.0919]` | — | — | `0.0747 [0.0687, 0.0808]` | `0.3380` | `0.1069` | — | — |
| `log_pool` | Scalar Baseline | 1 | — | `0.1261 [0.1112, 0.1395]` | — | — | `0.1176 [0.1127, 0.1226]` | `0.3863` | `0.1188` | — | — |
| `linear_pool` | Scalar Baseline | 1 | — | `0.1587 [0.1432, 0.1727]` | — | — | `0.1484 [0.1428, 0.1540]` | `0.4189` | `0.1293` | — | — |
| `sum_logodds` | Scalar Baseline | 1 | — | `0.4170 [0.3449, 0.4952]` | — | — | `0.4509 [0.4180, 0.4838]` | `0.6772` | `0.1278` | — | — |
| `naive_mha_dv1:trained` | Naive Flat MLP+MHA | 1 | ● No gain | `0.1271 [0.1066, 0.1473]` | `+0.1105 [+0.0905, +0.1300]` | `+0.0493 [+0.0336, +0.0650]` | `0.1263 [0.1156, 0.1372]` | `0.3872` | `0.1230` | `0.4403` | `0.4442` |
| `naive_mha_dv12:trained` | Naive Flat MLP+MHA | 12 | ● No gain | `0.1088 [0.0894, 0.1284]` | `+0.0922 [+0.0735, +0.1114]` | `+0.0310 [+0.0154, +0.0477]` | `0.1078 [0.0985, 0.1170]` | `0.3690` | `0.1164` | `0.4342` | `0.4353` |
| `slot_kv_dv1:trained` | Slot-Equivariant KV Clearing | 1 | ● Informative | `0.0465 [0.0350, 0.0583]` | `+0.0299 [+0.0206, +0.0399]` | `-0.0313 [-0.0456, -0.0157]` | `0.0431 [0.0394, 0.0468]` | `0.3067` | `0.0954` | `0.5042` | `0.7167` |
| `slot_kv_dv2:trained` | Slot-Equivariant KV Clearing | 2 | ● Informative | `0.0264 [0.0169, 0.0362]` | `+0.0098 [-0.0011, +0.0201]` | `-0.0514 [-0.0664, -0.0355]` | `0.0282 [0.0257, 0.0308]` | `0.2866` | `0.0884` | `0.5057` | `0.6367` |
| `slot_kv_dv4:trained` | Slot-Equivariant KV Clearing | 4 | ● Informative | `0.0191 [0.0112, 0.0272]` | `+0.0025 [-0.0077, +0.0126]` | `-0.0587 [-0.0739, -0.0420]` | `0.0172 [0.0156, 0.0188]` | `0.2793` | `0.0868` | `0.5058` | `0.5566` |
| `slot_kv_dv8:trained` | Slot-Equivariant KV Clearing | 8 | ● Informative | `0.0130 [0.0063, 0.0199]` | `-0.0036 [-0.0134, +0.0060]` | `-0.0648 [-0.0799, -0.0478]` | `0.0128 [0.0117, 0.0141]` | `0.2732` | `0.0849` | `0.5067` | `0.5230` |
| `slot_kv_dv12:trained` | Slot-Equivariant KV Clearing | 12 | ● Informative | `0.0107 [0.0045, 0.0170]` | `-0.0059 [-0.0153, +0.0030]` | `-0.0671 [-0.0820, -0.0499]` | `0.0096 [0.0085, 0.0108]` | `0.2708` | `0.0838` | `0.5081` | `0.5063` |

## Verification & Controls Audit

- **Harness Integrity**: `aggbench` v`0.1.0`, commit `edfbe03cdde27729ed97101b802617cb3afe0b89`, `git_dirty = False`, SHA256 `e50881934b0fe8f82e9343535cce37057fd977b558f02c26c90e4e542fdbbade`.
- **Brute-Force Oracle Check**: Verified on `24` `smoke` episodes (`max_abs_diff = 1.110e-16 < 1.0e-08`).
- **Strict Encoder Locality (`aggbench.messages.check_locality`)**: Verified for all learned encoders before and after training (`locality_verified = True`).
- **E3-Biased Fairness (`biased_agent_obs`)**: `biased_distortion_applied_to_seen_obs = False` (distortion applied strictly to seen slots before encoding; unobserved slots stay `NaN`, and no oracle fields are exposed on `AgentInputView`).
- **Controls Separation**: Every learned arm's `control:untrained_init` (`~0.508` regret) and `control:random_network` (`~0.506` regret) fail as expected, confirming the gain comes strictly from learned weights.
