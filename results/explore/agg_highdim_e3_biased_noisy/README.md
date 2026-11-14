# EXPLORATORY: High-Dimensional Local Key-Value Messages & Clearing on `E3-biased` (`visibility=noisy`) (`agg_highdim_e3_biased_noisy`)

Serves: high-dimensional per-agent Key-Value messages + Gram-confiability clearing that beats scalar oracle_clearing (0.017) and approaches full Bayes (0.000) across E2 and E3.

**Status:** `● Informative` · `EXPLORATORY` (95% bootstrap CI is strictly below `extremized_log_pool`, while noisy keys prevent beating noise-free `oracle_clearing`; `test` split untouched; harness commit `2be5d77aea`, `git_dirty=False`, harness SHA256 `e50881934b0fe8f8…`; runner SHA256 `301f32ee67d0d552…`; brute-force `oracle_check` verified on `24` smoke episodes to `max_abs_diff = 2.22e-16`; strict per-agent message locality verified via `aggbench.messages.check_locality` with `locality_verified = True`).

- **Why**: In `E3-biased` (Adversarial Affine Distortion (1 agent with 3x scale + 1.5 shift on seen obs, N=5)), 1D scalar posterior reports lose piece-identity information whenever $\mathbf{1}_{\bigcup S_i} \notin \text{rowspace}(C)$, bounding any scalar-report aggregator by `oracle_clearing` (`0.0166 [0.0095, 0.0238]` regret to `bayes`). We tested whether strictly local encoders emitting $d_v \in \{1, 2, 4, 8, 12\}$-dimensional value messages at `visibility=noisy` can break through `oracle_clearing` toward `bayes` (`0.0000`) when trained strictly on binary `truth` labels $Y \in \{0, 1\}$.
- **What we did**: Compared non-learned scalar baselines (`linear_pool`, `log_pool`, `sum_logodds`, `extremized_log_pool`, and the scalar ceiling `oracle_clearing`), a naive non-equivariant flat MLP local encoder + MultiheadAttention aggregator (`naive_mha_dv1`, `naive_mha_dv12`), and our slot-equivariant Key-Value local encoder + Gram/Woodbury clearing & Huber peer-recalibration architecture (`slot_kv_dv{1,2,4,8,12}`) on `n_train=2000` and `n_dev=2000` episodes (`n_boot=2000`), evaluating every learned arm with `control:untrained_init` and `control:random_network`.
- **Result in one line**: On `E3-biased` (`visibility=noisy`), naive attention failed (`naive_mha_dv12` regret `0.1090 [0.0901, 0.1280]`, `● No gain`), whereas `slot_kv_dv12:trained` achieved regret `0.0232 [0.0146, 0.0311]` to `bayes` (paired difference vs `oracle_clearing`: `+0.0066 [-0.0039, +0.0166]`; paired difference vs `extremized_log_pool`: `-0.0531 [-0.0677, -0.0375]`).

## Methods & Results (`dev`, $n=2000$, 2000-resample 95% Bootstrap CIs)

| Arm / Method | Type | $d_v$ | Status | Regret to `bayes` (`0.000`) | $\Delta$ Regret vs `oracle_clearing` | $\Delta$ Regret vs `extremized_log_pool` | KL to `bayes` | Log Loss | Brier | `untrained_init` Regret | `random_network` Regret |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| `bayes` | Global Ceiling | — | — | `0.0000 [0.0000, 0.0000]` | — | — | `0.0000 [0.0000, 0.0000]` | `0.2601` | `0.0808` | — | — |
| `oracle_clearing` | 1D Report Ceiling | 1 | — | `0.0166 [0.0095, 0.0238]` | — | — | `0.0158 [0.0141, 0.0176]` | `0.2768` | `0.0862` | — | — |
| `extremized_log_pool` | Scalar Baseline | 1 | — | `0.0763 [0.0604, 0.0910]` | — | — | `0.0718 [0.0659, 0.0780]` | `0.3364` | `0.1052` | — | — |
| `log_pool` | Scalar Baseline | 1 | — | `0.0913 [0.0768, 0.1047]` | — | — | `0.0839 [0.0791, 0.0889]` | `0.3514` | `0.1088` | — | — |
| `linear_pool` | Scalar Baseline | 1 | — | `0.1425 [0.1281, 0.1558]` | — | — | `0.1319 [0.1273, 0.1363]` | `0.4026` | `0.1222` | — | — |
| `sum_logodds` | Scalar Baseline | 1 | — | `0.3410 [0.2780, 0.4070]` | — | — | `0.3626 [0.3351, 0.3911]` | `0.6012` | `0.1222` | — | — |
| `naive_mha_dv1:trained` | Naive Flat MLP+MHA | 1 | ● No gain | `0.1117 [0.0931, 0.1309]` | `+0.0951 [+0.0780, +0.1127]` | `+0.0354 [+0.0191, +0.0519]` | `0.1038 [0.0949, 0.1135]` | `0.3718` | `0.1164` | `0.4403` | `0.4394` |
| `naive_mha_dv12:trained` | Naive Flat MLP+MHA | 12 | ● No gain | `0.1090 [0.0901, 0.1280]` | `+0.0923 [+0.0754, +0.1097]` | `+0.0327 [+0.0166, +0.0492]` | `0.1012 [0.0924, 0.1105]` | `0.3691` | `0.1153` | `0.4341` | `0.4353` |
| `slot_kv_dv1:trained` | Slot-Equivariant KV Clearing | 1 | ● No gain | `0.0672 [0.0538, 0.0806]` | `+0.0506 [+0.0389, +0.0629]` | `-0.0091 [-0.0224, +0.0054]` | `0.0608 [0.0558, 0.0660]` | `0.3273` | `0.1012` | `0.5382` | `1.0823` |
| `slot_kv_dv2:trained` | Slot-Equivariant KV Clearing | 2 | ● Informative | `0.0465 [0.0347, 0.0584]` | `+0.0299 [+0.0173, +0.0420]` | `-0.0298 [-0.0439, -0.0140]` | `0.0448 [0.0412, 0.0492]` | `0.3066` | `0.0945` | `0.5476` | `0.7280` |
| `slot_kv_dv4:trained` | Slot-Equivariant KV Clearing | 4 | ● Informative | `0.0328 [0.0228, 0.0427]` | `+0.0162 [+0.0044, +0.0280]` | `-0.0434 [-0.0576, -0.0278]` | `0.0289 [0.0263, 0.0317]` | `0.2930` | `0.0906` | `0.5377` | `0.6011` |
| `slot_kv_dv8:trained` | Slot-Equivariant KV Clearing | 8 | ● Informative | `0.0292 [0.0199, 0.0379]` | `+0.0125 [+0.0014, +0.0233]` | `-0.0471 [-0.0625, -0.0315]` | `0.0235 [0.0213, 0.0258]` | `0.2893` | `0.0895` | `0.5355` | `0.5382` |
| `slot_kv_dv12:trained` | Slot-Equivariant KV Clearing | 12 | ● Informative | `0.0232 [0.0146, 0.0311]` | `+0.0066 [-0.0039, +0.0166]` | `-0.0531 [-0.0677, -0.0375]` | `0.0193 [0.0174, 0.0213]` | `0.2833` | `0.0873` | `0.5363` | `0.4981` |

## Verification & Controls Audit

- **Harness Integrity**: `aggbench` v`0.1.0`, commit `2be5d77aea09538efefbaf373ed6591b51c1e075`, `git_dirty = False`, SHA256 `e50881934b0fe8f82e9343535cce37057fd977b558f02c26c90e4e542fdbbade`.
- **Brute-Force Oracle Check**: Verified on `24` `smoke` episodes (`max_abs_diff = 2.220e-16 < 1.0e-08`).
- **Strict Encoder Locality (`aggbench.messages.check_locality`)**: Verified for all learned encoders before and after training (`locality_verified = True`).
- **E3-Biased Fairness (`biased_agent_obs`)**: `biased_distortion_applied_to_seen_obs = True` (distortion applied strictly to seen slots before encoding; unobserved slots stay `NaN`, and no oracle fields are exposed on `AgentInputView`).
- **Controls Separation**: Every learned arm's `control:untrained_init` (`~0.536` regret) and `control:random_network` (`~0.498` regret) fail as expected, confirming the gain comes strictly from learned weights.
