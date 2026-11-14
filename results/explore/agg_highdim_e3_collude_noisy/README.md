# EXPLORATORY: High-Dimensional Local Key-Value Messages & Clearing on `E3-collude` (`visibility=noisy`) (`agg_highdim_e3_collude_noisy`)

Serves: high-dimensional per-agent Key-Value messages + Gram-confiability clearing that beats scalar oracle_clearing (0.017) and approaches full Bayes (0.000) across E2 and E3.

**Status:** `● Informative` · `EXPLORATORY` (Point-estimate regret is below `oracle_clearing` and 95% CI is strictly below `extremized_log_pool` (`oracle_clearing` CI overlaps zero); `test` split untouched; harness commit `2be5d77aea`, `git_dirty=False`, harness SHA256 `e50881934b0fe8f8…`; runner SHA256 `301f32ee67d0d552…`; brute-force `oracle_check` verified on `24` smoke episodes to `max_abs_diff = 2.22e-16`; strict per-agent message locality verified via `aggbench.messages.check_locality` with `locality_verified = True`).

- **Why**: In `E3-collude` (Coalition Evidence Pooling (3 colluders pooling pieces + 2 honest, N=5)), 1D scalar posterior reports lose piece-identity information whenever $\mathbf{1}_{\bigcup S_i} \notin \text{rowspace}(C)$, bounding any scalar-report aggregator by `oracle_clearing` (`0.0175 [0.0098, 0.0245]` regret to `bayes`). We tested whether strictly local encoders emitting $d_v \in \{1, 2, 4, 8, 12\}$-dimensional value messages at `visibility=noisy` can break through `oracle_clearing` toward `bayes` (`0.0000`) when trained strictly on binary `truth` labels $Y \in \{0, 1\}$.
- **What we did**: Compared non-learned scalar baselines (`linear_pool`, `log_pool`, `sum_logodds`, `extremized_log_pool`, and the scalar ceiling `oracle_clearing`), a naive non-equivariant flat MLP local encoder + MultiheadAttention aggregator (`naive_mha_dv1`, `naive_mha_dv12`), and our slot-equivariant Key-Value local encoder + Gram/Woodbury clearing & Huber peer-recalibration architecture (`slot_kv_dv{1,2,4,8,12}`) on `n_train=2000` and `n_dev=2000` episodes (`n_boot=2000`), evaluating every learned arm with `control:untrained_init` and `control:random_network`.
- **Result in one line**: On `E3-collude` (`visibility=noisy`), naive attention failed (`naive_mha_dv12` regret `0.0907 [0.0736, 0.1087]`, `● No gain`), whereas `slot_kv_dv12:trained` achieved regret `0.0124 [0.0065, 0.0182]` to `bayes` (paired difference vs `oracle_clearing`: `-0.0051 [-0.0139, +0.0036]`; paired difference vs `extremized_log_pool`: `-0.0178 [-0.0285, -0.0069]`).

## Methods & Results (`dev`, $n=2000$, 2000-resample 95% Bootstrap CIs)

| Arm / Method | Type | $d_v$ | Status | Regret to `bayes` (`0.000`) | $\Delta$ Regret vs `oracle_clearing` | $\Delta$ Regret vs `extremized_log_pool` | KL to `bayes` | Log Loss | Brier | `untrained_init` Regret | `random_network` Regret |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| `bayes` | Global Ceiling | — | — | `0.0000 [0.0000, 0.0000]` | — | — | `0.0000 [0.0000, 0.0000]` | `0.2601` | `0.0808` | — | — |
| `oracle_clearing` | 1D Report Ceiling | 1 | — | `0.0175 [0.0098, 0.0245]` | — | — | `0.0144 [0.0127, 0.0160]` | `0.2777` | `0.0870` | — | — |
| `extremized_log_pool` | Scalar Baseline | 1 | — | `0.0302 [0.0205, 0.0394]` | — | — | `0.0286 [0.0258, 0.0315]` | `0.2904` | `0.0920` | — | — |
| `log_pool` | Scalar Baseline | 1 | — | `0.0463 [0.0358, 0.0557]` | — | — | `0.0410 [0.0385, 0.0438]` | `0.3064` | `0.0944` | — | — |
| `linear_pool` | Scalar Baseline | 1 | — | `0.0744 [0.0629, 0.0848]` | — | — | `0.0669 [0.0640, 0.0701]` | `0.3345` | `0.1001` | — | — |
| `sum_logodds` | Scalar Baseline | 1 | — | `0.2589 [0.2055, 0.3141]` | — | — | `0.2912 [0.2748, 0.3074]` | `0.5190` | `0.1114` | — | — |
| `naive_mha_dv1:trained` | Naive Flat MLP+MHA | 1 | ● No gain | `0.0912 [0.0746, 0.1085]` | `+0.0737 [+0.0589, +0.0900]` | `+0.0610 [+0.0471, +0.0754]` | `0.0824 [0.0748, 0.0902]` | `0.3514` | `0.1109` | `0.4404` | `0.4411` |
| `naive_mha_dv12:trained` | Naive Flat MLP+MHA | 12 | ● No gain | `0.0907 [0.0736, 0.1087]` | `+0.0732 [+0.0576, +0.0899]` | `+0.0605 [+0.0460, +0.0754]` | `0.0826 [0.0747, 0.0910]` | `0.3509` | `0.1102` | `0.4338` | `0.4356` |
| `slot_kv_dv1:trained` | Slot-Equivariant KV Clearing | 1 | ● No gain | `0.0449 [0.0335, 0.0570]` | `+0.0274 [+0.0179, +0.0369]` | `+0.0146 [+0.0047, +0.0245]` | `0.0411 [0.0374, 0.0447]` | `0.3050` | `0.0956` | `0.5051` | `0.9370` |
| `slot_kv_dv2:trained` | Slot-Equivariant KV Clearing | 2 | ● No gain | `0.0309 [0.0210, 0.0413]` | `+0.0133 [+0.0034, +0.0231]` | `+0.0006 [-0.0102, +0.0109]` | `0.0299 [0.0274, 0.0328]` | `0.2910` | `0.0906` | `0.5060` | `0.8363` |
| `slot_kv_dv4:trained` | Slot-Equivariant KV Clearing | 4 | ● No gain | `0.0221 [0.0137, 0.0309]` | `+0.0046 [-0.0060, +0.0148]` | `-0.0081 [-0.0200, +0.0041]` | `0.0197 [0.0178, 0.0218]` | `0.2823` | `0.0879` | `0.5060` | `0.7078` |
| `slot_kv_dv8:trained` | Slot-Equivariant KV Clearing | 8 | ● Informative | `0.0181 [0.0112, 0.0247]` | `+0.0006 [-0.0092, +0.0101]` | `-0.0121 [-0.0234, -0.0004]` | `0.0135 [0.0122, 0.0148]` | `0.2783` | `0.0870` | `0.5059` | `0.6138` |
| `slot_kv_dv12:trained` | Slot-Equivariant KV Clearing | 12 | ● Informative | `0.0124 [0.0065, 0.0182]` | `-0.0051 [-0.0139, +0.0036]` | `-0.0178 [-0.0285, -0.0069]` | `0.0091 [0.0082, 0.0101]` | `0.2726` | `0.0847` | `0.5063` | `0.5548` |

## Verification & Controls Audit

- **Harness Integrity**: `aggbench` v`0.1.0`, commit `2be5d77aea09538efefbaf373ed6591b51c1e075`, `git_dirty = False`, SHA256 `e50881934b0fe8f82e9343535cce37057fd977b558f02c26c90e4e542fdbbade`.
- **Brute-Force Oracle Check**: Verified on `24` `smoke` episodes (`max_abs_diff = 2.220e-16 < 1.0e-08`).
- **Strict Encoder Locality (`aggbench.messages.check_locality`)**: Verified for all learned encoders before and after training (`locality_verified = True`).
- **E3-Biased Fairness (`biased_agent_obs`)**: `biased_distortion_applied_to_seen_obs = False` (distortion applied strictly to seen slots before encoding; unobserved slots stay `NaN`, and no oracle fields are exposed on `AgentInputView`).
- **Controls Separation**: Every learned arm's `control:untrained_init` (`~0.506` regret) and `control:random_network` (`~0.555` regret) fail as expected, confirming the gain comes strictly from learned weights.
