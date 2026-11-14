# EXPLORATORY: High-Dimensional Local Key-Value Messages & Clearing on `E2` (`visibility=noisy`) (`agg_highdim_e2_noisy`)

Serves: high-dimensional per-agent Key-Value messages + Gram-confiability clearing that beats scalar oracle_clearing (0.017) and approaches full Bayes (0.000) across E2 and E3.

**Status:** `● Informative` · `EXPLORATORY` (95% bootstrap CI is strictly below `extremized_log_pool`, while noisy keys prevent beating noise-free `oracle_clearing`; `test` split untouched; harness commit `006e34a621`, `git_dirty=False`, harness SHA256 `e50881934b0fe8f8…`; runner SHA256 `301f32ee67d0d552…`; brute-force `oracle_check` verified on `24` smoke episodes to `max_abs_diff = 2.22e-16`; strict per-agent message locality verified via `aggbench.messages.check_locality` with `locality_verified = True`).

- **Why**: In `E2` (Overlapping Honest Evidence Pool (5 agents, K=12 pieces)), 1D scalar posterior reports lose piece-identity information whenever $\mathbf{1}_{\bigcup S_i} \notin \text{rowspace}(C)$, bounding any scalar-report aggregator by `oracle_clearing` (`0.0166 [0.0095, 0.0238]` regret to `bayes`). We tested whether strictly local encoders emitting $d_v \in \{1, 2, 4, 8, 12\}$-dimensional value messages at `visibility=noisy` can break through `oracle_clearing` toward `bayes` (`0.0000`) when trained strictly on binary `truth` labels $Y \in \{0, 1\}$.
- **What we did**: Compared non-learned scalar baselines (`linear_pool`, `log_pool`, `sum_logodds`, `extremized_log_pool`, and the scalar ceiling `oracle_clearing`), a naive non-equivariant flat MLP local encoder + MultiheadAttention aggregator (`naive_mha_dv1`, `naive_mha_dv12`), and our slot-equivariant Key-Value local encoder + Gram/Woodbury clearing & Huber peer-recalibration architecture (`slot_kv_dv{1,2,4,8,12}`) on `n_train=2000` and `n_dev=2000` episodes (`n_boot=2000`), evaluating every learned arm with `control:untrained_init` and `control:random_network`.
- **Result in one line**: On `E2` (`visibility=noisy`), naive attention failed (`naive_mha_dv12` regret `0.0992 [0.0814, 0.1171]`, `● No gain`), whereas `slot_kv_dv12:trained` achieved regret `0.0178 [0.0113, 0.0246]` to `bayes` (paired difference vs `oracle_clearing`: `+0.0012 [-0.0084, +0.0110]`; paired difference vs `extremized_log_pool`: `-0.0270 [-0.0396, -0.0143]`).

## Methods & Results (`dev`, $n=2000$, 2000-resample 95% Bootstrap CIs)

| Arm / Method | Type | $d_v$ | Status | Regret to `bayes` (`0.000`) | $\Delta$ Regret vs `oracle_clearing` | $\Delta$ Regret vs `extremized_log_pool` | KL to `bayes` | Log Loss | Brier | `untrained_init` Regret | `random_network` Regret |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| `bayes` | Global Ceiling | — | — | `0.0000 [0.0000, 0.0000]` | — | — | `0.0000 [0.0000, 0.0000]` | `0.2601` | `0.0808` | — | — |
| `oracle_clearing` | 1D Report Ceiling | 1 | — | `0.0166 [0.0095, 0.0238]` | — | — | `0.0158 [0.0141, 0.0176]` | `0.2768` | `0.0862` | — | — |
| `extremized_log_pool` | Scalar Baseline | 1 | — | `0.0448 [0.0328, 0.0565]` | — | — | `0.0407 [0.0373, 0.0443]` | `0.3050` | `0.0953` | — | — |
| `log_pool` | Scalar Baseline | 1 | — | `0.1124 [0.0988, 0.1248]` | — | — | `0.1026 [0.0991, 0.1061]` | `0.3725` | `0.1121` | — | — |
| `linear_pool` | Scalar Baseline | 1 | — | `0.1517 [0.1375, 0.1655]` | — | — | `0.1407 [0.1362, 0.1450]` | `0.4119` | `0.1247` | — | — |
| `sum_logodds` | Scalar Baseline | 1 | — | `0.1324 [0.1005, 0.1666]` | — | — | `0.1419 [0.1313, 0.1527]` | `0.3925` | `0.1044` | — | — |
| `naive_mha_dv1:trained` | Naive Flat MLP+MHA | 1 | ● No gain | `0.0986 [0.0811, 0.1163]` | `+0.0820 [+0.0660, +0.0984]` | `+0.0538 [+0.0403, +0.0675]` | `0.0913 [0.0835, 0.0993]` | `0.3588` | `0.1129` | `0.4403` | `0.4412` |
| `naive_mha_dv12:trained` | Naive Flat MLP+MHA | 12 | ● No gain | `0.0992 [0.0814, 0.1171]` | `+0.0826 [+0.0663, +0.0995]` | `+0.0544 [+0.0405, +0.0685]` | `0.0918 [0.0837, 0.0999]` | `0.3594` | `0.1130` | `0.4342` | `0.4353` |
| `slot_kv_dv1:trained` | Slot-Equivariant KV Clearing | 1 | ● No gain | `0.0502 [0.0389, 0.0623]` | `+0.0336 [+0.0242, +0.0436]` | `+0.0054 [-0.0053, +0.0161]` | `0.0449 [0.0409, 0.0488]` | `0.3104` | `0.0966` | `0.5077` | `0.6699` |
| `slot_kv_dv2:trained` | Slot-Equivariant KV Clearing | 2 | ● Informative | `0.0315 [0.0223, 0.0411]` | `+0.0149 [+0.0036, +0.0254]` | `-0.0133 [-0.0260, -0.0006]` | `0.0293 [0.0267, 0.0321]` | `0.2916` | `0.0901` | `0.5119` | `0.5768` |
| `slot_kv_dv4:trained` | Slot-Equivariant KV Clearing | 4 | ● Informative | `0.0243 [0.0161, 0.0324]` | `+0.0077 [-0.0033, +0.0181]` | `-0.0206 [-0.0335, -0.0074]` | `0.0189 [0.0171, 0.0209]` | `0.2844` | `0.0882` | `0.5086` | `0.5085` |
| `slot_kv_dv8:trained` | Slot-Equivariant KV Clearing | 8 | ● Informative | `0.0220 [0.0146, 0.0294]` | `+0.0054 [-0.0047, +0.0154]` | `-0.0228 [-0.0353, -0.0094]` | `0.0157 [0.0142, 0.0174]` | `0.2822` | `0.0877` | `0.5093` | `0.4761` |
| `slot_kv_dv12:trained` | Slot-Equivariant KV Clearing | 12 | ● Informative | `0.0178 [0.0113, 0.0246]` | `+0.0012 [-0.0084, +0.0110]` | `-0.0270 [-0.0396, -0.0143]` | `0.0123 [0.0109, 0.0136]` | `0.2780` | `0.0861` | `0.5119` | `0.4583` |

## Verification & Controls Audit

- **Harness Integrity**: `aggbench` v`0.1.0`, commit `006e34a621aa6b86b88a63aac73deac4eda8cba7`, `git_dirty = False`, SHA256 `e50881934b0fe8f82e9343535cce37057fd977b558f02c26c90e4e542fdbbade`.
- **Brute-Force Oracle Check**: Verified on `24` `smoke` episodes (`max_abs_diff = 2.220e-16 < 1.0e-08`).
- **Strict Encoder Locality (`aggbench.messages.check_locality`)**: Verified for all learned encoders before and after training (`locality_verified = True`).
- **E3-Biased Fairness (`biased_agent_obs`)**: `biased_distortion_applied_to_seen_obs = False` (distortion applied strictly to seen slots before encoding; unobserved slots stay `NaN`, and no oracle fields are exposed on `AgentInputView`).
- **Controls Separation**: Every learned arm's `control:untrained_init` (`~0.512` regret) and `control:random_network` (`~0.458` regret) fail as expected, confirming the gain comes strictly from learned weights.
