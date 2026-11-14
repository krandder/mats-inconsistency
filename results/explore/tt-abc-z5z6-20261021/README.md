# EXPLORATORY Baseline: Kelvin's Trader-Transformer Axes (A), (B), (C) on Complementary/XOR Environments `Z5` and `Z6` (`tt-abc-z5z6-20261021`)

Serves: Exploratory 20k-pool baseline for Kelvin's 1-token-per-trader causal Transformer architecture (`aggbench/trader_transformer.py`) across Axes `(A)` cross-trader attention schedule, `(B)` supervision target (`final`, `every_prefix`, and oracle `distill`), and `(C)` trader-direction residual stream (`kv`, `recurrent`, `feedback`) on the two open complementary/XOR zoo environments `Z5` (`MultiQuestionEnv`) and `Z6` (`ParityEnv`, `complementary-xor`) (`3` seeds `0, 1, 2` on NVIDIA L4 GPU).
Builds on: Kelvin's Trader-Transformer specification (`research-orientation-2026-10-20-addendum.md` lines 38–53, `aggbench/trader_transformer.py`) and `gate-gemini-tt-2026-10-21.md` (`#3859`, Must-Fix 7).

**Status:** `● Informative (EXPLORATORY baseline)` (`test` splits untouched; `alignment="unaligned"`; fixed `n_train=20000` pool shared across seeds `0, 1, 2` with 10% held-out train-split early stopping + `A_attn_all_no_es` un-early-stopped control; `n_eval=2000` dev episodes; GPU: `NVIDIA L4`; runner commit `e5039a60da52fbffc5208c549cd866cacceba307`).

> [!IMPORTANT]
> **Gate Audit Disclosures (`gate-gemini-tt-2026-10-21.md`, `#3859`):**
> 1. **Matched Encoder Across All Arms:** On `Z5` and `Z6` ($N=6$ agents), there is no piece-matrix `deepsets` encoder; every learned arm (`A_*`, `B_*`, `C_*`, `V_*`, and `full_context_transformer_*`) uses the **exact same flat linear token representation** (`full_context`, `in_dim = 11` at `visibility="none", own_model=True`).
> 2. **Regret Relative to `0.01` Nats Target:**
>    - On **`Z6` (`ParityEnv`, complementary XOR)**, `A_attn_all_reperm` reaches **`0.0055 ± 0.0015` regret (`kl = 0.0080 ± 0.0012`)** and `B_loss_every_prefix_reperm` reaches **`0.0091 ± 0.0013` regret (`mean_pref_kl = 0.0051`)** — both below `0.01` nats to Bayes (`0.0000`) on the 20k pool, whereas all scalar report-pooling baselines (`oracle_clearing`, `extremized_log_pool`, `log_pool`, `linear_pool`, `sum_logodds`) and `A_attn_none` remain at the uninformative prior floor (`0.2464–0.2466`).
>    - On **`Z5` (`MultiQuestionEnv`)**, the 20k-pool non-oracle learned arms reach `0.0255 ± 0.0039` (`B_loss_every_prefix_reperm`) and `0.0262 ± 0.0033` (`A_attn_all_reperm`), matching 20k-pool `full_context_transformer_none` (`0.0265 ± 0.0015`) but remaining `2.5x` above `0.01` nats until trained in the `3000 x 512` fresh regime (`tt-gate-xor-fresh`).
> 3. **Oracle Ceiling Disclosure:** `B_loss_distill` (`0.0049 ± 0.0008` on `Z5`, `0.0000 ± 0.0004` on `Z6`) uses exact prefix Bayes supervision (`uses_oracle=True`) and is graded strictly as an **Oracle Ceiling**.
> 4. **Fixed 20k Train Pool Across Seeds:** In commit `e5039a6`, `env.sample("train", 20000)` was sampled once per environment and shared across seeds `0, 1, 2`; `± SD` reflects weight-initialization and trader-order resampling variance on that fixed 20k pool. The companion `3000 x 512` fresh-episode run (`tt-gate-xor-fresh`) resamples `1.536M` fresh episodes per seed.

---

## Mandatory `(a)–(d)` Gate Disclosure (`#3843` / `#3847`)

- **`(a)` What a token is:**
  - One flat vector per trader $r \in \{0, \dots, 5\}$ in arrival order $\sigma_e$, produced by `aggbench.transformer.full_context` (`[obs_i, own_sensor_model_i, provenance_key_i at visibility, report_logit_i / 4]`, dimension `in_dim = 11` at `visibility="none", own_model=True` on both `Z5` and `Z6`).
- **`(b)` What a layer is:**
  - Pre-norm Transformer block (`TraderNet` in `aggbench/trader_transformer.py`, $L=4$ layers, $H=4$ heads, $d_{\text{model}}=128$, MLP hidden width $512$).
- **`(c)` What crosses between traders and at what width:**
  - Width $d_{\text{model}} = 128$ vector representations cross between traders $r' \le r$ under a strict causal trader mask via `token_state="kv"` (causal K/V self-attention), `token_state="recurrent"` (GRU CEO-summary state $s_r^{(l)} \in \mathbb{R}^{128}$), or `token_state="feedback"` (depth-pooled top-down memory $m_{r'} \in \mathbb{R}^{128}$).
- **`(d)` Regret to Bayes (`0.0000`) and Comparison Bars (`alignment="unaligned"`):**
  - Evaluated on `n_eval = 2000` unaligned `dev` episodes against `bayes` (`0.0000`), `floor_none` (`0.0000` on `Z5` and `Z6`), `oracle_clearing` (`0.0000` on `Z5`, `0.2466` on `Z6`), `extremized_log_pool` (`0.1341` on `Z5`, `0.2466` on `Z6`), and `full_context_transformer_none` (`0.0265 ± 0.0015` on `Z5`, `0.0127 ± 0.0045` on `Z6`), with `untrained_init` and `random_network` controls in [`tt_abc_z5/result.json`](./tt_abc_z5/result.json) and [`tt_abc_z6/result.json`](./tt_abc_z6/result.json).

---

## Why, What We Did, Result, So What, Next Step

- **Why:** In `Z5` (`MultiQuestionEnv`, where agents observe different sub-questions $A, B, C$ of a joint target) and `Z6` (`ParityEnv`, where the target is the XOR parity of latent bits and each individual agent sees only one bit, making every scalar report marginally uninformative with `report_logit = 0`), aggregation requires combining complementary information across traders rather than pooling scalar log-odds.
- **What we did:** Ran `benchmarks/explore_trader_transformer_abc_grid.py --suite abc --envs Z5,Z6 --seeds 0,1,2` (`n_train=20000, n_eval=2000, steps=2000, batch=512, width=128, layers=4`, commit `e5039a6`) on an NVIDIA L4 GPU (`29.3` min wall-clock).
- **Result (`● Informative — Exploratory Baseline`):**
  1. **`Z6` (`ParityEnv`, Complementary XOR):**
     - Every scalar report baseline (`oracle_clearing`, `extremized_log_pool`, `log_pool`, `linear_pool`, `sum_logodds`) and the no-cross-attention ablation (`A_attn_none`) fail completely (`regret = 0.2464–0.2466`, `kl = 0.2619`), because single-agent marginal reports carry zero information about parity.
     - With cross-trader attention and trader-order resampling, `A_attn_all_reperm` reaches **`0.0055 ± 0.0015` regret (`kl = 0.0080 ± 0.0012`)** and `B_loss_every_prefix_reperm` reaches **`0.0091 ± 0.0013` regret (`mean_pref_kl = 0.0051`)**, outperforming 20k-pool `full_context_transformer_none` (`0.0127 ± 0.0045`).
     - A single cross-trader attention layer at the final layer (`A_attn_last`) reaches **`0.0122 ± 0.0034`** regret (`kl = 0.0124 ± 0.0012`), and a single cross-trader attention layer at the first layer (`A_attn_first`) reaches **`0.0131 ± 0.0038`**, matching 4-layer `full_context_transformer_none` (`0.0127 ± 0.0045`).
     - In Axis `(C)`, the $O(1)$-state GRU summary (`C_state_recurrent_prefix` `0.0189 ± 0.0062`, `C_state_recurrent_final` `0.0278 ± 0.0070`) and Feedback Transformer (`C_state_feedback_prefix` `0.0214 ± 0.0060`) also learn the XOR parity composition, though trailing full K/V attention (`0.0055–0.0091`).
  2. **`Z5` (`MultiQuestionEnv`):**
     - Removing cross-trader attention (`A_attn_none`) yields `0.2052 ± 0.0105` regret, and `extremized_log_pool` gets `0.1341`.
     - `B_loss_every_prefix_reperm` (`0.0255 ± 0.0039`, `mean_pref_kl = 0.0182`), `A_attn_all_reperm` (`0.0262 ± 0.0033`), and `C_state_recurrent_prefix` (`0.0274 ± 0.0022`, `mean_pref_kl = 0.0210`) match 20k-pool `full_context_transformer_none` (`0.0265 ± 0.0015`), while the **Oracle Distillation Ceiling** (`B_loss_distill`, `uses_oracle=True`) reaches `0.0049 ± 0.0008` (`mean_pref_kl = 0.0025`).
  3. **Own-Model Necessity (`V_vis_none_no_own`):** Dropping the agent's own sensor model (`own_model=False` at `visibility="none"`) degrades regret to `0.0908 ± 0.0007` on `Z5` and `0.1498 ± 0.0013` on `Z6` because agents cannot identify which sub-question or bit they observed.
- **So what:** On complementary/XOR structures (`Z5`, `Z6`), cross-trader attention (even a single final-layer attention step `A_attn_last` or recurrent state `C_state_recurrent_prefix`) resolves the parity/multi-question composition where scalar pooling fails (`0.2466` on `Z6`, `0.1341` on `Z5`), and per-minibatch trader-order resampling (`_reperm`) cuts 20k-pool regret from `0.0302` to `0.0055` on `Z6`.
- **Next step:** Compare against the full `3000 x 512` (`1.536M` fresh episodes) run (`tt-gate-xor-fresh`) on `Z5` and `Z6`.

---

## Master Table 1: `Z6` (`ParityEnv`, Complementary XOR, `alignment="unaligned"`, `dev` $n=2000$, 3 Seeds `0, 1, 2`, Fixed 20k Pool)

| Arm / Method | Axis / Category | Vis / `own` | Regret (3-Seed Mean ± SD) | KL to Bayes (Mean ± SD) | Mean Prefix KL ($t=1..6$) | `untrained_init` | `random_network` | Status / Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| `bayes` | Ceiling | oracle | `0.0000 ± 0.0000` | `0.0000 ± 0.0000` | `0.0000` | — | — | Exact Bayes Ceiling (`uses_oracle=True`) |
| `floor_none` | Info Floor (`none`) | oracle | `0.0000 ± 0.0000` | `0.0000 ± 0.0000` | — | — | — | Exact `none` Floor (`own_model` identifies bit) |
| `B_loss_distill` | `(B)` Oracle Distill Ceiling | `none` (`True`) | `0.0000 ± 0.0004` | `0.0005 ± 0.0000` | `0.0001` | `0.2546` | `0.3110` | Oracle Ceiling (`uses_oracle=True`, Distill) |
| **`A_attn_all_reperm`** | **`(A)` All + Order Resample** | `none` (`True`) | **`0.0055 ± 0.0015`** | **`0.0080 ± 0.0012`** | `0.0480` | `0.2546` | `0.2988` | **`● Informative` (`< 0.01` nats; matched linear encoder)** |
| **`B_loss_every_prefix_reperm`** | **`(B)` Prefix + Resample** | `none` (`True`) | **`0.0091 ± 0.0013`** | **`0.0109 ± 0.0004`** | **`0.0051`** | `0.2546` | `0.2856` | **`● Informative` (`< 0.01` nats final & mean prefix KL)** |
| `A_attn_last` | `(A)` Last-Layer Only | `none` (`True`) | `0.0122 ± 0.0034` | `0.0124 ± 0.0012` | `0.0457` | `0.2745` | `0.2872` | `● Informative` (1 Attn layer matches `FCT_none`) |
| **`full_context_transformer_none`** | **Reference Bar (20k pool)** | `none` (`True`) | **`0.0127 ± 0.0045`** | **`0.0102 ± 0.0025`** | — | `0.2774` | `0.2592` | **Reference Bar (20k pool)** |
| `full_context_transformer_full` | Reference Bar (`full`) | `full` (`True`) | `0.0130 ± 0.0048` | `0.0114 ± 0.0030` | — | `0.2689` | `0.2729` | Reference Bar (`full`) |
| `A_attn_first` | `(A)` Layer `(0,)` Only | `none` (`True`) | `0.0131 ± 0.0038` | `0.0139 ± 0.0018` | `0.0532` | `0.2720` | `0.2845` | `● Informative` |
| `A_attn_every_other` | `(A)` Layers `(1, 3)` | `none` (`True`) | `0.0177 ± 0.0029` | `0.0197 ± 0.0011` | `0.0547` | `0.2644` | `0.2599` | `● Informative` |
| `C_state_recurrent_prefix` | `(C)` GRU + Prefix | `none` (`True`) | `0.0189 ± 0.0062` | `0.0169 ± 0.0030` | `0.0086` | `0.2587` | `0.2851` | `● Informative` (`O(1)` Recurrent State) |
| `C_state_feedback_prefix` | `(C)` Feedback + Prefix | `none` (`True`) | `0.0214 ± 0.0060` | `0.0250 ± 0.0056` | `0.0122` | `0.2545` | `0.2949` | `● Informative` |
| `V_vis_full` | `Vis` Full (`own=True`) | `full` (`True`) | `0.0261 ± 0.0036` | `0.0270 ± 0.0030` | `0.0482` | `0.2722` | `0.2894` | `● Informative` |
| `B_loss_every_prefix_pos` | `(B)` Prefix + Learned Pos | `none` (`True`) | `0.0270 ± 0.0027` | `0.0283 ± 0.0012` | `0.0132` | `0.2537` | `0.2647` | `● Informative` |
| `C_state_recurrent_final` | `(C)` GRU (`final`) | `none` (`True`) | `0.0278 ± 0.0070` | `0.0219 ± 0.0032` | `0.0339` | `0.2587` | `0.2900` | `● Informative` |
| `A_attn_first_last` | `(A)` Layers `(0, 3)` | `none` (`True`) | `0.0279 ± 0.0043` | `0.0247 ± 0.0035` | `0.0559` | `0.2806` | `0.2622` | `● Informative` |
| `C_state_feedback_final` | `(C)` Feedback (`final`) | `none` (`True`) | `0.0287 ± 0.0091` | `0.0299 ± 0.0074` | `0.0480` | `0.2545` | `0.3090` | `● Informative` |
| `B_loss_every_prefix` | `(B)` Every-Prefix | `none` (`True`) | `0.0298 ± 0.0029` | `0.0335 ± 0.0018` | `0.0167` | `0.2546` | `0.2944` | `● Informative` |
| `A_attn_all` | `(A)` All Layers `(0,1,2,3)` | `none` (`True`) | `0.0302 ± 0.0054` | `0.0272 ± 0.0011` | `0.0570` | `0.2546` | `0.3027` | `● Informative` |
| `V_vis_noisy` | `Vis` Noisy (`own=True`) | `noisy` (`True`) | `0.0388 ± 0.0150` | `0.0383 ± 0.0111` | `0.0558` | `0.2728` | `0.2852` | `● Informative` |
| `V_vis_noisy_no_own` | `V2` Diagnostic | `noisy` (`False`) | `0.0921 ± 0.0019` | `0.0868 ± 0.0009` | `0.0823` | `0.2563` | `0.3047` | `● Informative` |
| `V_vis_none_no_own` | `V2` Diagnostic | `none` (`False`) | `0.1498 ± 0.0013` | `0.1425 ± 0.0002` | `0.1987` | `0.2889` | `0.2644` | `● Informative` |
| `A_attn_none` | `(A)` No Cross-Trader Attn | `none` (`True`) | `0.2464 ± 0.0008` | `0.2619 ± 0.0002` | `0.1266` | `0.2863` | `0.2755` | `● No gain` (Single-trader XOR floor) |
| `oracle_clearing` | Scalar Clearing Ceiling | oracle | `0.2466 ± 0.0000` | `0.2619 ± 0.0000` | — | — | — | Scalar Reports Uninformative on XOR (`0.2466`) |
| `extremized_log_pool` | Baseline (Scalar) | reports | `0.2466 ± 0.0000` | `0.2619 ± 0.0000` | — | — | — | Baseline |
| `log_pool` | Baseline (Scalar) | reports | `0.2466 ± 0.0000` | `0.2619 ± 0.0000` | — | — | — | Baseline |
| `linear_pool` | Baseline (Scalar) | reports | `0.2466 ± 0.0000` | `0.2619 ± 0.0000` | — | — | — | Baseline |
| `sum_logodds` | Baseline (Scalar) | reports | `0.2466 ± 0.0000` | `0.2619 ± 0.0000` | — | — | — | Baseline |
| `A_attn_all_no_es` | `#3856` No Early-Stop | `none` (`True`) | `0.5920 ± 0.0566` | `0.5746 ± 0.0302` | `0.4472` | `0.2546` | `0.3142` | `● Informative` (Memorizes 20k Pool) |

---

## Master Table 2: `Z5` (`MultiQuestionEnv`, `alignment="unaligned"`, `dev` $n=2000$, 3 Seeds `0, 1, 2`, Fixed 20k Pool)

| Arm / Method | Axis / Category | Vis / `own` | Regret (3-Seed Mean ± SD) | KL to Bayes (Mean ± SD) | Mean Prefix KL ($t=1..6$) | `untrained_init` | `random_network` | Status / Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| `bayes` | Ceiling | oracle | `0.0000 ± 0.0000` | `0.0000 ± 0.0000` | `0.0000` | — | — | Exact Bayes Ceiling (`uses_oracle=True`) |
| `floor_none` | Info Floor (`none`) | oracle | `0.0000 ± 0.0000` | `0.0000 ± 0.0000` | — | — | — | Exact `none` Floor |
| `oracle_clearing` | Scalar Clearing Ceiling | oracle | `0.0000 ± 0.0000` | `0.0000 ± 0.0000` | — | — | — | Scalar Ceiling (`uses_oracle=True`) |
| `B_loss_distill` | `(B)` Oracle Distill Ceiling | `none` (`True`) | `0.0049 ± 0.0008` | `0.0041 ± 0.0000` | `0.0025` | `0.4945` | `0.5230` | Oracle Ceiling (`uses_oracle=True`, Distill) |
| `B_loss_every_prefix_reperm` | `(B)` Prefix + Resample | `none` (`True`) | `0.0255 ± 0.0039` | `0.0287 ± 0.0021` | `0.0182` | `0.4945` | `0.4924` | `● Informative` (Ties `FCT_none` on 20k pool) |
| `A_attn_all_reperm` | `(A)` All + Order Resample | `none` (`True`) | `0.0262 ± 0.0033` | `0.0276 ± 0.0021` | `0.0862` | `0.4945` | `0.4961` | `● Informative` (Ties `FCT_none` on 20k pool) |
| **`full_context_transformer_none`** | **Reference Bar (20k pool)** | `none` (`True`) | **`0.0265 ± 0.0015`** | **`0.0279 ± 0.0039`** | — | `0.4628` | `0.4778` | **Reference Bar (20k pool)** |
| `C_state_recurrent_prefix` | `(C)` GRU + Prefix | `none` (`True`) | `0.0274 ± 0.0022` | `0.0317 ± 0.0009` | `0.0210` | `0.4529` | `0.4227` | `● Informative` (`O(1)` Recurrent State) |
| `full_context_transformer_full` | Reference Bar (`full`) | `full` (`True`) | `0.0279 ± 0.0077` | `0.0323 ± 0.0020` | — | `0.4183` | `0.4501` | Reference Bar (`full`) |
| `C_state_feedback_prefix` | `(C)` Feedback + Prefix | `none` (`True`) | `0.0311 ± 0.0051` | `0.0334 ± 0.0050` | `0.0231` | `0.4960` | `0.4147` | `● Informative` |
| `C_state_recurrent_final` | `(C)` GRU (`final`) | `none` (`True`) | `0.0315 ± 0.0038` | `0.0353 ± 0.0016` | `0.0493` | `0.4529` | `0.4255` | `● Informative` |
| `C_state_feedback_final` | `(C)` Feedback (`final`) | `none` (`True`) | `0.0350 ± 0.0079` | `0.0392 ± 0.0031` | `0.0800` | `0.4960` | `0.4160` | `● Informative` |
| `A_attn_all` | `(A)` All Layers `(0,1,2,3)` | `none` (`True`) | `0.0351 ± 0.0024` | `0.0384 ± 0.0029` | `0.0909` | `0.4945` | `0.4998` | `● Informative` |
| `A_attn_every_other` | `(A)` Layers `(1, 3)` | `none` (`True`) | `0.0359 ± 0.0047` | `0.0394 ± 0.0042` | `0.1044` | `0.4540` | `0.4323` | `● Informative` |
| `V_vis_full` | `Vis` Full (`own=True`) | `full` (`True`) | `0.0392 ± 0.0066` | `0.0414 ± 0.0037` | `0.0879` | `0.4534` | `0.4665` | `● Informative` |
| `B_loss_every_prefix` | `(B)` Every-Prefix | `none` (`True`) | `0.0393 ± 0.0010` | `0.0383 ± 0.0045` | `0.0256` | `0.4945` | `0.4952` | `● Informative` |
| `A_attn_last` | `(A)` Last-Layer Only | `none` (`True`) | `0.0399 ± 0.0033` | `0.0382 ± 0.0021` | `0.0970` | `0.4334` | `0.4986` | `● Informative` |
| `A_attn_first_last` | `(A)` Layers `(0, 3)` | `none` (`True`) | `0.0404 ± 0.0059` | `0.0396 ± 0.0019` | `0.0950` | `0.4539` | `0.4311` | `● Informative` |
| `A_attn_first` | `(A)` Layer `(0,)` Only | `none` (`True`) | `0.0419 ± 0.0055` | `0.0428 ± 0.0008` | `0.0995` | `0.4340` | `0.4932` | `● Informative` |
| `V_vis_noisy` | `Vis` Noisy (`own=True`) | `noisy` (`True`) | `0.0431 ± 0.0035` | `0.0468 ± 0.0046` | `0.0951` | `0.4529` | `0.4673` | `● Informative` |
| `B_loss_every_prefix_pos` | `(B)` Prefix + Learned Pos | `none` (`True`) | `0.0448 ± 0.0036` | `0.0474 ± 0.0054` | `0.0271` | `0.5054` | `0.4213` | `● Informative` |
| `V_vis_noisy_no_own` | `V2` Diagnostic | `noisy` (`False`) | `0.0556 ± 0.0017` | `0.0568 ± 0.0028` | `0.1053` | `0.4941` | `0.4975` | `● Informative` |
| `V_vis_none_no_own` | `V2` Diagnostic | `none` (`False`) | `0.0908 ± 0.0007` | `0.0838 ± 0.0012` | `0.0983` | `0.4594` | `0.4327` | `● Informative` |
| `extremized_log_pool` | Baseline (Scalar) | reports | `0.1341 ± 0.0000` | `0.1286 ± 0.0000` | — | — | — | Baseline |
| `log_pool` | Baseline (Scalar) | reports | `0.1495 ± 0.0000` | `0.1415 ± 0.0000` | — | — | — | Baseline |
| `linear_pool` | Baseline (Scalar) | reports | `0.1591 ± 0.0000` | `0.1507 ± 0.0000` | — | — | — | Baseline |
| `A_attn_none` | `(A)` No Cross-Trader Attn | `none` (`True`) | `0.2052 ± 0.0105` | `0.1977 ± 0.0032` | `0.1136` | `0.4231` | `0.4183` | `● No gain` (Ablation Floor) |
| `sum_logodds` | Baseline (Scalar) | reports | `0.5177 ± 0.0000` | `0.5437 ± 0.0000` | — | — | — | Baseline |
| `A_attn_all_no_es` | `#3856` No Early-Stop | `none` (`True`) | `0.6049 ± 0.0318` | `0.6295 ± 0.0048` | `0.7164` | `0.4945` | `0.5432` | `● Informative` (Memorizes 20k Pool) |

---

## Axis `(B)` Prefix KL Curves Across Trader Positions $t = 1 \dots 6$ (`3`-Seed Mean KL to Exact Prefix Bayes)

| Environment | Arm | Supervision Loss | $t=1$ | $t=2$ | $t=3$ | $t=4$ | $t=5$ | $t=6$ (Final) | Mean Prefix KL |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Z6` | `A_attn_none` | `final` (no cross-attn) | `0.0005` | `0.0415` | `0.0949` | `0.1513` | `0.2094` | `0.2619` | `0.1266` |
| `Z6` | `A_attn_all_reperm` | `final` (`reperm`) | `0.0138` | `0.1429` | `0.0748` | `0.0344` | `0.0143` | `0.0080` | `0.0480` |
| `Z6` | `B_loss_every_prefix_reperm` | `every_prefix` (`reperm`) | `0.0001` | `0.0024` | `0.0039` | `0.0055` | `0.0079` | `0.0109` | `0.0051` |
| `Z6` | `C_state_recurrent_prefix` | `every_prefix` (GRU) | `0.0011` | `0.0038` | `0.0068` | `0.0100` | `0.0131` | `0.0169` | `0.0086` |
| `Z6` | `B_loss_distill` | `distill` (`uses_oracle=True`) | `0.0000` | `0.0000` | `0.0000` | `0.0001` | `0.0002` | `0.0005` | `0.0001` |
| `Z5` | `A_attn_none` | `final` (no cross-attn) | `0.0096` | `0.0608` | `0.1053` | `0.1384` | `0.1699` | `0.1977` | `0.1136` |
| `Z5` | `A_attn_all_reperm` | `final` (`reperm`) | `0.2398` | `0.1142` | `0.0632` | `0.0414` | `0.0311` | `0.0276` | `0.0862` |
| `Z5` | `B_loss_every_prefix_reperm` | `every_prefix` (`reperm`) | `0.0064` | `0.0122` | `0.0161` | `0.0201` | `0.0255` | `0.0287` | `0.0182` |
| `Z5` | `C_state_recurrent_prefix` | `every_prefix` (GRU) | `0.0096` | `0.0135` | `0.0190` | `0.0239` | `0.0282` | `0.0317` | `0.0210` |
| `Z5` | `B_loss_distill` | `distill` (`uses_oracle=True`) | `0.0003` | `0.0020` | `0.0026` | `0.0030` | `0.0033` | `0.0041` | `0.0025` |

---

## Artifacts in This Directory

- [`summary.json`](./summary.json): Full 3-seed aggregated metrics, prefix KL curves, training logs, and per-seed bootstrap CIs (`commit e5039a60da52fbffc5208c549cd866cacceba307`).
- [`tt_abc_z5/result.json`](./tt_abc_z5/result.json): Schema-validated `aggbench.result/v1` artifact for `Z5`.
- [`tt_abc_z6/result.json`](./tt_abc_z6/result.json): Schema-validated `aggbench.result/v1` artifact for `Z6`.
