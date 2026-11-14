# EXPLORATORY: High-Dimensional Local Key-Value Messages & Clearing on `E3-biased` (`visibility=full`) (`agg_highdim_e3_biased_full`)

Serves: high-dimensional per-agent Key-Value messages + Gram-confiability clearing that beats scalar oracle_clearing (0.017) and approaches full Bayes (0.000) across E2 and E3.

**Status:** `● Informative` · `EXPLORATORY` (Point-estimate regret is below `oracle_clearing` and 95% CI is strictly below `extremized_log_pool` (`oracle_clearing` CI overlaps zero); `test` split untouched; harness commit `2be5d77aea`, `git_dirty=False`, harness SHA256 `e50881934b0fe8f8…`; runner SHA256 `301f32ee67d0d552…`; brute-force `oracle_check` verified on `24` smoke episodes to `max_abs_diff = 2.22e-16`; strict per-agent message locality verified via `aggbench.messages.check_locality` with `locality_verified = True`).

- **Why**: In `E3-biased` (Adversarial Affine Distortion (1 agent with 3x scale + 1.5 shift on seen obs, N=5)), 1D scalar posterior reports lose piece-identity information whenever $\mathbf{1}_{\bigcup S_i} \notin \text{rowspace}(C)$, bounding any scalar-report aggregator by `oracle_clearing` (`0.0166 [0.0095, 0.0238]` regret to `bayes`). We tested whether strictly local encoders emitting $d_v \in \{1, 2, 4, 8, 12\}$-dimensional value messages at `visibility=full` can break through `oracle_clearing` toward `bayes` (`0.0000`) when trained strictly on binary `truth` labels $Y \in \{0, 1\}$.
- **What we did**: Compared non-learned scalar baselines (`linear_pool`, `log_pool`, `sum_logodds`, `extremized_log_pool`, and the scalar ceiling `oracle_clearing`), a naive non-equivariant flat MLP local encoder + MultiheadAttention aggregator (`naive_mha_dv1`, `naive_mha_dv12`), and our slot-equivariant Key-Value local encoder + Gram/Woodbury clearing & Huber peer-recalibration architecture (`slot_kv_dv{1,2,4,8,12}`) on `n_train=2000` and `n_dev=2000` episodes (`n_boot=2000`), evaluating every learned arm with `control:untrained_init` and `control:random_network`.
- **Result in one line**: On `E3-biased` (`visibility=full`), naive attention failed (`naive_mha_dv12` regret `0.1083 [0.0896, 0.1268]`, `● No gain`), whereas `slot_kv_dv12:trained` achieved regret `0.0111 [0.0052, 0.0172]` to `bayes` (paired difference vs `oracle_clearing`: `-0.0055 [-0.0143, +0.0033]`; paired difference vs `extremized_log_pool`: `-0.0652 [-0.0796, -0.0495]`).

## Methods & Results (`dev`, $n=2000$, 2000-resample 95% Bootstrap CIs)

| Arm / Method | Type | $d_v$ | Status | Regret to `bayes` (`0.000`) | $\Delta$ Regret vs `oracle_clearing` | $\Delta$ Regret vs `extremized_log_pool` | KL to `bayes` | Log Loss | Brier | `untrained_init` Regret | `random_network` Regret |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| `bayes` | Global Ceiling | — | — | `0.0000 [0.0000, 0.0000]` | — | — | `0.0000 [0.0000, 0.0000]` | `0.2601` | `0.0808` | — | — |
| `oracle_clearing` | 1D Report Ceiling | 1 | — | `0.0166 [0.0095, 0.0238]` | — | — | `0.0158 [0.0141, 0.0176]` | `0.2768` | `0.0862` | — | — |
| `extremized_log_pool` | Scalar Baseline | 1 | — | `0.0763 [0.0604, 0.0910]` | — | — | `0.0718 [0.0659, 0.0780]` | `0.3364` | `0.1052` | — | — |
| `log_pool` | Scalar Baseline | 1 | — | `0.0913 [0.0768, 0.1047]` | — | — | `0.0839 [0.0791, 0.0889]` | `0.3514` | `0.1088` | — | — |
| `linear_pool` | Scalar Baseline | 1 | — | `0.1425 [0.1281, 0.1558]` | — | — | `0.1319 [0.1273, 0.1363]` | `0.4026` | `0.1222` | — | — |
| `sum_logodds` | Scalar Baseline | 1 | — | `0.3410 [0.2780, 0.4070]` | — | — | `0.3626 [0.3351, 0.3911]` | `0.6012` | `0.1222` | — | — |
| `naive_mha_dv1:trained` | Naive Flat MLP+MHA | 1 | ● No gain | `0.1106 [0.0919, 0.1294]` | `+0.0940 [+0.0764, +0.1120]` | `+0.0343 [+0.0181, +0.0505]` | `0.1026 [0.0937, 0.1126]` | `0.3707` | `0.1157` | `0.4403` | `0.4405` |
| `naive_mha_dv12:trained` | Naive Flat MLP+MHA | 12 | ● No gain | `0.1083 [0.0896, 0.1268]` | `+0.0916 [+0.0743, +0.1096]` | `+0.0320 [+0.0159, +0.0486]` | `0.1018 [0.0928, 0.1111]` | `0.3684` | `0.1148` | `0.4342` | `0.4352` |
| `slot_kv_dv1:trained` | Slot-Equivariant KV Clearing | 1 | ● Informative | `0.0614 [0.0485, 0.0748]` | `+0.0448 [+0.0337, +0.0565]` | `-0.0148 [-0.0279, -0.0013]` | `0.0565 [0.0517, 0.0612]` | `0.3216` | `0.0992` | `0.5528` | `1.1707` |
| `slot_kv_dv2:trained` | Slot-Equivariant KV Clearing | 2 | ● Informative | `0.0381 [0.0269, 0.0494]` | `+0.0215 [+0.0100, +0.0330]` | `-0.0381 [-0.0527, -0.0228]` | `0.0389 [0.0354, 0.0428]` | `0.2983` | `0.0916` | `0.5630` | `0.8061` |
| `slot_kv_dv4:trained` | Slot-Equivariant KV Clearing | 4 | ● Informative | `0.0202 [0.0119, 0.0287]` | `+0.0036 [-0.0065, +0.0137]` | `-0.0561 [-0.0703, -0.0410]` | `0.0200 [0.0180, 0.0220]` | `0.2803` | `0.0868` | `0.5586` | `0.6768` |
| `slot_kv_dv8:trained` | Slot-Equivariant KV Clearing | 8 | ● Informative | `0.0156 [0.0083, 0.0228]` | `-0.0011 [-0.0102, +0.0081]` | `-0.0607 [-0.0750, -0.0454]` | `0.0141 [0.0126, 0.0156]` | `0.2757` | `0.0855` | `0.5614` | `0.5888` |
| `slot_kv_dv12:trained` | Slot-Equivariant KV Clearing | 12 | ● Informative | `0.0111 [0.0052, 0.0172]` | `-0.0055 [-0.0143, +0.0033]` | `-0.0652 [-0.0796, -0.0495]` | `0.0102 [0.0091, 0.0115]` | `0.2713` | `0.0842` | `0.5651` | `0.5421` |

## Verification & Controls Audit

- **Harness Integrity**: `aggbench` v`0.1.0`, commit `2be5d77aea09538efefbaf373ed6591b51c1e075`, `git_dirty = False`, SHA256 `e50881934b0fe8f82e9343535cce37057fd977b558f02c26c90e4e542fdbbade`.
- **Brute-Force Oracle Check**: Verified on `24` `smoke` episodes (`max_abs_diff = 2.220e-16 < 1.0e-08`).
- **Strict Encoder Locality (`aggbench.messages.check_locality`)**: Verified for all learned encoders before and after training (`locality_verified = True`).
- **E3-Biased Fairness (`biased_agent_obs`)**: `biased_distortion_applied_to_seen_obs = True` (distortion applied strictly to seen slots before encoding; unobserved slots stay `NaN`, and no oracle fields are exposed on `AgentInputView`).
- **Controls Separation**: Every learned arm's `control:untrained_init` (`~0.565` regret) and `control:random_network` (`~0.542` regret) fail as expected, confirming the gain comes strictly from learned weights.
