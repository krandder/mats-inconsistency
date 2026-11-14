# EXPLORATORY Baseline: Kelvin's Trader-Transformer Axes (A), (B), (C) & Intra-Trader DeepSets on Unaligned `E2` and `Z8` (`tt-abc-e2z8-20261021`)

Serves: Exploratory 20k-pool baseline for Kelvin's 1-token-per-trader causal Transformer architecture (`aggbench/trader_transformer.py`) across Axes `(A)` cross-trader attention schedule, `(B)` supervision target (`final`, `every_prefix`, `distill`), `(C)` trader-direction residual stream (`kv`, `recurrent`, `feedback`), and `V4` intra-trader `trader_encoder="deepsets"` on `alignment="unaligned"` `E2` and `Z8` (`PrivateReadEnv`, `3` seeds `0, 1, 2` on NVIDIA L4 GPU).
Builds on: Kelvin's Trader-Transformer specification (`research-orientation-2026-10-20-addendum.md` lines 38–53, `aggbench/trader_transformer.py` commit `eb3cd7d`) and `gate-gemini-tt-2026-10-21.md` (`#3859`).

**Status:** `● Informative (EXPLORATORY baseline)` (`test` splits untouched; `alignment="unaligned"`; fixed `n_train=20000` pool shared across seeds `0, 1, 2` with 10% held-out train-split early stopping + `A_attn_all_no_es` un-early-stopped control; `n_eval=2000` dev episodes; GPU: `NVIDIA L4`; runner commit `605faec`).

> [!IMPORTANT]
> **Gate Audit Disclosures (`gate-gemini-tt-2026-10-21.md`, `#3859`):**
> 1. **No environment is below `0.01` nats regret to Bayes (`0.0000`) yet:** The best non-oracle learned arm in this 20k-pool exploratory run reaches `0.0365 ± 0.0053` regret (`kl = 0.0322`) on `E2` and `0.0391 ± 0.0018` (`kl = 0.0352`) on `Z8` — still `3.6x–3.9x` above the `0.01` nats near-Bayes target.
> 2. **On `Z8`, `extremized_log_pool` (`0.0344`, `kl = 0.0302`) beats every non-oracle learned arm (`0.0391–0.0957`).**
> 3. **Encoder difference vs. `full_context_transformer_none`:** In this exploratory run, `full_context_transformer_none` (`0.0615` on `E2`, `0.0733` on `Z8`) used the flat 49-D token (`full_context`, which concatenates 12 unaligned slots and appends `report_logit / 4`), whereas the `*_deepsets*` trader arms used a 2-layer slot-permutation-invariant `DeepSets` encoder over the 12 piece cells with no scalar report. When both use the flat linear token (`A_attn_all` `0.0841`, `A_attn_all_reperm` `0.0557` vs `full_context_transformer_none` `0.0615` on `E2`), the 1-token-per-trader causal transformer does not beat `full_context_transformer_none`. Unconfounded matched-encoder bars (`bar_fct_deepsets_bidir`, `bar_fct_no_report`) and the `3000 x 512` fresh-stream headline bar (`0d94ef1`) are evaluated in the follow-up fresh-regime run.
> 4. **Pre-`7379c5d` `Z8` Prefix Bayes Bug:** Commit `605faec` ran prior to commit `7379c5d`, when `aggbench/sequential.py:prefix_bayes_logits` computed `Z8` prefix Bayes using `E2`'s noiseless formula (averaging noisy reads instead of dividing their sum by $n_k + \tau^2$, off by up to `3.02` log-odds; hence `final_prefix_kl = 0.0421` differed from `kl = 0.0303` on `Z8`). All `Z8` `Mean Prefix KL` values and `Z8` `every_prefix` / `distill` rows in this exploratory table are marked **`[Pre-7379c5d Z8 Prefix Bug]`** and superseded by the post-`7379c5d` run.
> 5. **Fixed 20k Train Pool Across Seeds:** In commit `605faec`, `env.sample("train", 20000)` was sampled once per environment and shared across seeds `0, 1, 2`; `± SD` therefore reflects weight-initialization and trader-order resampling variance on a fixed 20k pool.

---

## Mandatory `(a)–(d)` Gate Disclosure (`#3843` / `#3847`)

- **`(a)` What a token is:**
  - In `layout="trader"` with default `trader_encoder="linear"`: one flat vector per trader $r \in \{0, \dots, N-1\}$ in arrival order $\sigma_e$, produced by `aggbench.transformer.full_context` (`[obs_i (NaN->0), obs_present_mask_i, own_sensor_model_i, provenance_key_i at visibility, report_logit_i / 4]`, dimension `in_dim = 49` at `visibility="none"` on `E2` and `Z8`, `73` on `E2 (full/noisy)`, `85` on `Z8 (full/noisy)`).
  - In `layout="trader"` with `trader_encoder="deepsets"` (`Finding V4`): each trader $r$ holds up to $K=12$ piece cells `[present, obs_k, own_cov_k, own_mu_k, (key_cov_k, key_mu_k)]` (`in_dim = 4` at `visibility="none"`, `6` on `E2 (full)`, `7` on `Z8 (full)`, with no scalar `report_logit`). A shared 2-layer DeepSets encoder $\phi: \mathbb{R}^{d_{\text{cell}}} \to \mathbb{R}^{128}$ maps each present cell and mean-pools across present slots to produce one slot-permutation-invariant trader token $h_r^{(0)} \in \mathbb{R}^{128}$ before layer 0.
- **`(b)` What a layer is:**
  - Pre-norm Transformer block (`TraderNet` in `aggbench/trader_transformer.py`, $L=4$ layers, $H=4$ heads, $d_{\text{model}}=128$, MLP hidden width $4 d_{\text{model}} = 512$):
    $$\tilde{h}^{(l)} = h^{(l-1)} + \text{CrossTraderChannel}_l\!\left(\text{LN}(h^{(l-1)})\right), \qquad h^{(l)} = \tilde{h}^{(l)} + \text{MLP}_l\!\left(\text{LN}(\tilde{h}^{(l)})\right)$$
    where $\text{CrossTraderChannel}_l$ is active only when $l \in \text{attn\_layers}$.
- **`(c)` What crosses between traders and at what width:**
  - Width $d_{\text{model}} = 128$ vector representations cross between traders $r' \le r$ under a strict causal trader mask:
    * `token_state="kv"`: standard multi-head causal self-attention over $(K_{1:r}^{(l)}, V_{1:r}^{(l)}) \in \mathbb{R}^{r \times 128}$.
    * `token_state="recurrent"`: a learned GRU CEO-summary state $s_r^{(l)} = \text{GRUCell}_l(\text{LN}(h_r^{(l-1)}), s_{r-1}^{(l)}) \in \mathbb{R}^{128}$ (`O(1)` state width per trader, zero KV history table).
    * `token_state="feedback"`: Fan et al. (2020) Feedback Transformer where keys and values at every active layer come from the depth-pooled top-down memory $m_{r'} = \sum_{l=0}^L w_l h_{r'}^{(l)} \in \mathbb{R}^{128}$ of strictly earlier traders $r' < r$.
- **`(d)` Regret to Bayes (`0.0000`) and Comparison Bars (`alignment="unaligned"`):**
  - Every learned arm is evaluated on `n_eval = 2000` unaligned `dev` episodes against `bayes` (`0.0000`), `oracle_clearing` (`0.0166` on `E2`, `0.0236` on `Z8`), `extremized_log_pool` (`0.0448` on `E2`, `0.0344` on `Z8`), and flat-token `full_context_transformer_none` (`0.0615 ± 0.0066` on `E2`, `0.0733 ± 0.0123` on `Z8`), with `untrained_init` and `random_network` controls in [`tt_abc_e2/result.json`](./tt_abc_e2/result.json) and [`tt_abc_z8/result.json`](./tt_abc_z8/result.json).

---

## Why, What We Did, Result, So What, Next Step

- **Why:** Under `alignment="unaligned"`, each trader $i$ applies an independent secret permutation $T_{e,i} \in S_K$ to its $K=12$ slot coordinates. Flattening those $K$ unaligned slots into a 49-D vector (`full_context`) forces a linear input projection $W_{\text{in}} \in \mathbb{R}^{49 \times 128}$ to learn $K!$-permutation invariance from scratch, causing severe sample complexity and fixed-pool memorization (`#3856`). We ran an exploratory 20k-pool sweep to diagnose:
  1. Whether 10% train-split early stopping + per-minibatch trader-order resampling (`permute_traders_train=True`) prevents the fixed-pool memorization pathology (`A_attn_all_no_es`),
  2. How much of the gap on unaligned `E2` and `Z8` comes from the intra-trader token encoder (`trader_encoder="linear"` vs `"deepsets"`), and
  3. How Axes `(A)` cross-trader attention schedule, `(B)` prefix supervision (`final`, `every_prefix`, and oracle `distill`), and `(C)` recurrent/feedback state channels behave within this 20k-pool regime.
- **What we did:** Ran `benchmarks/explore_trader_transformer_abc_grid.py --suite abc --envs E2,Z8 --seeds 0,1,2` (`n_train=20000, n_eval=2000, steps=2000, batch=512, width=128, layers=4`, runner commit `605faec`) on an NVIDIA L4 GPU (`50.8` min wall-clock), evaluating 29 learned arms + 6 baselines + 58 control arms per seed (`279` model evaluations per environment).
- **Result (`● Informative — Exploratory Baseline`):**
  1. **Fixed-Pool Memorization Diagnostic (`#3856`):** Without early stopping (`A_attn_all_no_es`), the 800k-parameter transformer memorizes the 20k training pool: dev regret explodes to **`0.9423 ± 0.0602`** on `E2` and **`0.8677 ± 0.0840`** on `Z8` (worse than `untrained_init` `0.4833`). Adding 10% held-out train-split early stopping (`A_attn_all`) lowers dev regret to `0.0841 ± 0.0074` on `E2` and `0.0902 ± 0.0185` on `Z8`, and resampling trader arrival order per minibatch (`A_attn_all_reperm`) lowers it to `0.0557 ± 0.0078` on `E2` and `0.0718 ± 0.0117` on `Z8` (comparable to flat-token `full_context_transformer_none` `0.0615` and `0.0733`).
  2. **Intra-Trader DeepSets (`trader_encoder="deepsets"`, `V4`) vs. Flat Linear Tokens:**
     - Replacing the flat linear projection with `trader_encoder="deepsets"` lowers `E2` regret from `0.0557 ± 0.0078` (`A_attn_all_reperm`) to `0.0365 ± 0.0053` (`A_attn_all_deepsets_reperm`, `kl = 0.0322 ± 0.0023`, vs `extremized_log_pool` `0.0448`).
     - On `Z8`, `A_attn_last_deepsets` (`0.0393 ± 0.0043`) and `A_attn_all_deepsets_reperm` (`0.0395 ± 0.0018`, `kl = 0.0303 ± 0.0006`) improve over flat-linear `A_attn_all_reperm` (`0.0718`), **but still trail `extremized_log_pool` (`0.0344`, `kl = 0.0302`)**, and remain far above `0.01` nats regret.
     - Because `full_context_transformer_none` in this run used the flat linear token (`0.0615` on `E2`, `0.0733` on `Z8`), the comparison between `*_deepsets*` and `full_context_transformer_none` is confounded by the input encoder (`Must-Fix 2`).
  3. **Axis `(A)` Cross-Trader Attention Schedule:** Removing cross-trader attention (`A_attn_none`) leaves regret at the single-trader floor (`0.2460 ± 0.0010` on `E2`, `0.2476 ± 0.0051` on `Z8`). With `deepsets` tokens, a single cross-trader attention layer at the final layer (`A_attn_last_deepsets`, `629k` params) reaches `0.0430 ± 0.0062` on `E2` and `0.0393 ± 0.0043` on `Z8` (similar to 4-layer `A_attn_all_deepsets` `0.0447` on `E2`, `0.0446` on `Z8`).
  4. **Axis `(B)` Prefix Supervision Target & Oracle Distillation Ceiling:** On `E2` (where `prefix_bayes_logits` was exact), `B_loss_every_prefix_deepsets_reperm` lowers mean prefix KL ($t=1..5$) from `0.0407` to `0.0240` (`t=1` KL: `0.0792 -> 0.0062`), while the **Oracle Distillation Ceiling** (`B_loss_distill_deepsets`, `uses_oracle=True`) reaches `0.0250 ± 0.0032` regret (`mean_pref_kl = 0.0150`). On `Z8`, `every_prefix` and `distill` rows in this table were trained against the pre-`7379c5d` buggy `Z8` prefix Bayes target and are re-run with exact `Z8` prefix Bayes in the follow-up run.
  5. **Axis `(C)` Trader-Direction Residual Stream:** With `deepsets` tokens, `C_state_recurrent_deepsets` (GRU summary, no KV history) reaches `0.0508 ± 0.0052` on `E2` and `0.0422 ± 0.0029` on `Z8`.
- **So what:** In unaligned piece-structured environments (`E2`, `Z8`), the flat 49-D token projection is a major bottleneck, and fixed 20k pools require early stopping or fresh episodes (`--fresh-episodes`) to avoid memorization. However, to know whether causal trader architectures or attention schedules provide any gain beyond the `deepsets` input encoder itself, we must compare against a bidirectional `FullContextTransformer` equipped with the exact same `deepsets` encoder (`bar_fct_deepsets_bidir`) and against the `3000 x 512` fresh-regime headline bar (`0d94ef1`).
- **Next step:** Run the unconfounded matched-encoder bars (`bar_fct_deepsets_bidir`, `bar_fct_no_report`, `bar_fct_fresh_headline_w64_l3`), the post-`7379c5d` exact `Z8` prefix Bayes target, and the XOR environments (`Z5`, `Z6`, `Z6-parity3`) at the full `3000 x 512` fresh-episode budget.

---

## Master Table 1: `E2` (`alignment="unaligned"`, `dev` $n=2000$, 3 Seeds `0, 1, 2`, Fixed 20k Train Pool)

| Arm / Method | Axis / Category | Token Encoder | Vis / `own` | Regret (3-Seed Mean ± SD) | KL to Bayes (Mean ± SD) | Mean Prefix KL ($t=1..5$) | `untrained_init` | `random_network` | Status / Note |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `bayes` | Ceiling | — | oracle | `0.0000 ± 0.0000` | `0.0000 ± 0.0000` | `0.0000` | — | — | Exact Bayes Ceiling (`uses_oracle=True`) |
| `oracle_clearing` | Ceiling (Scalar) | — | oracle | `0.0166 ± 0.0000` | `0.0158 ± 0.0000` | — | — | — | Scalar Clearing Ceiling (`uses_oracle=True`) |
| `B_loss_distill_deepsets` | `(B)` Oracle Distill Ceiling | `deepsets` | `none` (`True`) | `0.0250 ± 0.0032` | `0.0244 ± 0.0009` | `0.0150` | `0.4627` | `0.3937` | Oracle Ceiling (`uses_oracle=True`, Distill) |
| `A_attn_all_deepsets_reperm` | `(A)` + `V4` DeepSets | `deepsets` | `none` (`True`) | `0.0365 ± 0.0053` | `0.0322 ± 0.0023` | `0.0407` | `0.4627` | `0.3946` | `● Informative` (Encoder confound vs linear `FCT_none`) |
| `A_attn_last_deepsets` | `(A)` Last-Layer Only | `deepsets` | `none` (`True`) | `0.0430 ± 0.0062` | `0.0356 ± 0.0036` | `0.0379` | `0.5115` | `0.4946` | `● Informative` (1 Attn layer + `deepsets`) |
| `V_vis_full_deepsets` | `Vis` Full + DeepSets | `deepsets` | `full` (`True`) | `0.0435 ± 0.0060` | `0.0365 ± 0.0027` | `0.0359` | `0.5200` | `0.5098` | `● Informative` |
| `B_loss_every_prefix_deepsets_reperm` | `(B)` Every-Prefix | `deepsets` | `none` (`True`) | `0.0439 ± 0.0058` | `0.0358 ± 0.0034` | `0.0240` | `0.4627` | `0.3952` | `● Informative` (`-41%` Prefix KL vs `final`) |
| `A_attn_all_deepsets` | `(A)` All + DeepSets | `deepsets` | `none` (`True`) | `0.0447 ± 0.0009` | `0.0367 ± 0.0010` | `0.0410` | `0.4627` | `0.3956` | `● Informative` |
| `extremized_log_pool` | Baseline (Scalar) | — | reports | `0.0448 ± 0.0000` | `0.0407 ± 0.0000` | — | — | — | Scalar Heuristic Baseline |
| `V_vis_full_deepsets_prefix` | `Vis` Full + Prefix | `deepsets` | `full` (`True`) | `0.0456 ± 0.0028` | `0.0379 ± 0.0008` | `0.0248` | `0.5200` | `0.5030` | `● Informative` |
| `B_loss_every_prefix_deepsets` | `(B)` Every-Prefix | `deepsets` | `none` (`True`) | `0.0488 ± 0.0026` | `0.0397 ± 0.0028` | `0.0270` | `0.4627` | `0.3959` | `● Informative` |
| `B_loss_distill` | `(B)` Oracle Distill Ceiling | `linear` | `none` (`True`) | `0.0506 ± 0.0030` | `0.0433 ± 0.0015` | `0.0280` | `0.4833` | `0.4520` | Oracle Ceiling (`uses_oracle=True`, Distill) |
| `C_state_recurrent_deepsets` | `(C)` GRU Summary | `deepsets` | `none` (`True`) | `0.0508 ± 0.0052` | `0.0434 ± 0.0029` | `0.0456` | `0.4581` | `0.4237` | `● Informative` (`O(1)` Recurrent State) |
| `B_loss_every_prefix_pos` | `(B)` Prefix + Learned Pos | `linear` | `none` (`True`) | `0.0550 ± 0.0044` | `0.0517 ± 0.0048` | `0.0368` | `0.4815` | `0.4606` | `● Informative` |
| `A_attn_all_reperm` | `(A)` All + Order Resample | `linear` | `none` (`True`) | `0.0557 ± 0.0078` | `0.0563 ± 0.0027` | `0.0616` | `0.4833` | `0.4483` | `● Informative` (Matched linear encoder vs `FCT_none`) |
| `B_loss_every_prefix_reperm` | `(B)` Prefix + Resample | `linear` | `none` (`True`) | `0.0573 ± 0.0090` | `0.0561 ± 0.0054` | `0.0412` | `0.4833` | `0.4478` | `● Informative` |
| **`full_context_transformer_none`** | **Reference Bar (Linear)** | `linear` | `none` (`True`) | **`0.0615 ± 0.0066`** | **`0.0605 ± 0.0049`** | — | `0.5933` | `0.4935` | **Flat-Linear Reference Bar (20k pool)** |
| `C_state_feedback_prefix` | `(C)` Feedback + Prefix | `linear` | `none` (`True`) | `0.0659 ± 0.0043` | `0.0633 ± 0.0035` | `0.0478` | `0.4614` | `0.4595` | `● Informative` |
| `C_state_recurrent_prefix` | `(C)` GRU + Prefix | `linear` | `none` (`True`) | `0.0669 ± 0.0100` | `0.0619 ± 0.0094` | `0.0445` | `0.4975` | `0.4907` | `● Informative` |
| `B_loss_every_prefix` | `(B)` Every-Prefix | `linear` | `none` (`True`) | `0.0689 ± 0.0049` | `0.0650 ± 0.0017` | `0.0518` | `0.4833` | `0.4480` | `● Informative` |
| `full_context_transformer_full` | Reference Bar (`full`) | `linear` | `full` (`True`) | `0.0698 ± 0.0043` | `0.0642 ± 0.0036` | — | `0.4990` | `0.4883` | Flat-Linear Reference Bar (`full`) |
| `V_vis_none_no_own` | `V2` Diagnostic | `linear` | `none` (`False`) | `0.0764 ± 0.0046` | `0.0778 ± 0.0080` | `0.0791` | `0.4585` | `0.4304` | `● Informative` |
| `A_attn_last` | `(A)` Last-Layer Only | `linear` | `none` (`True`) | `0.0768 ± 0.0077` | `0.0735 ± 0.0071` | `0.0892` | `0.4530` | `0.4621` | `● Informative` |
| `C_state_feedback_final` | `(C)` Feedback (`final`) | `linear` | `none` (`True`) | `0.0779 ± 0.0058` | `0.0758 ± 0.0025` | `0.0933` | `0.4614` | `0.4603` | `● Informative` |
| `C_state_recurrent_final` | `(C)` GRU (`final`) | `linear` | `none` (`True`) | `0.0795 ± 0.0082` | `0.0753 ± 0.0048` | `0.0626` | `0.4975` | `0.4925` | `● Informative` |
| `A_attn_every_other` | `(A)` Layers `(1, 3)` | `linear` | `none` (`True`) | `0.0795 ± 0.0084` | `0.0782 ± 0.0037` | `0.0910` | `0.4565` | `0.4348` | `● Informative` |
| `A_attn_first_last` | `(A)` Layers `(0, 3)` | `linear` | `none` (`True`) | `0.0806 ± 0.0101` | `0.0787 ± 0.0063` | `0.0870` | `0.4607` | `0.4339` | `● Informative` |
| `A_attn_first` | `(A)` Layer `(0,)` Only | `linear` | `none` (`True`) | `0.0820 ± 0.0113` | `0.0816 ± 0.0082` | `0.0892` | `0.4666` | `0.4666` | `● Informative` |
| `A_attn_all` | `(A)` All Layers `(0,1,2,3)` | `linear` | `none` (`True`) | `0.0841 ± 0.0074` | `0.0828 ± 0.0055` | `0.0842` | `0.4833` | `0.4490` | `● Informative` |
| `V_vis_noisy_no_own` | `V2` Diagnostic | `linear` | `noisy` (`False`) | `0.0852 ± 0.0142` | `0.0836 ± 0.0123` | `0.0886` | `0.4828` | `0.4504` | `● Informative` |
| `V_vis_noisy` | `Vis` Noisy (`own=True`) | `linear` | `noisy` (`True`) | `0.0858 ± 0.0098` | `0.0835 ± 0.0107` | `0.0815` | `0.4563` | `0.5034` | `● Informative` |
| `V_vis_full` | `Vis` Full (`own=True`) | `linear` | `full` (`True`) | `0.0893 ± 0.0091` | `0.0854 ± 0.0102` | `0.0829` | `0.4534` | `0.5003` | `● Informative` |
| `log_pool` | Baseline (Scalar) | — | reports | `0.1124 ± 0.0000` | `0.1026 ± 0.0000` | — | — | — | Baseline |
| `sum_logodds` | Baseline (Scalar) | — | reports | `0.1324 ± 0.0000` | `0.1419 ± 0.0000` | — | — | — | Baseline |
| `linear_pool` | Baseline (Scalar) | — | reports | `0.1517 ± 0.0000` | `0.1407 ± 0.0000` | — | — | — | Baseline |
| `A_attn_none` | `(A)` No Cross-Trader Attn | `linear` | `none` (`True`) | `0.2460 ± 0.0010` | `0.2333 ± 0.0007` | `0.1529` | `0.4626` | `0.4740` | `● No gain` (Ablation Floor) |
| **`A_attn_all_no_es`** | **`#3856` No Early-Stop** | `linear` | `none` (`True`) | **`0.9423 ± 0.0602`** | **`0.9622 ± 0.0161`** | `0.8944` | `0.4833` | `0.4861` | **`● Informative` (Memorizes 20k Pool)** |

---

## Master Table 2: `Z8` (`PrivateReadEnv`, `alignment="unaligned"`, `dev` $n=2000$, 3 Seeds `0, 1, 2`, Fixed 20k Train Pool)

> [!WARNING]
> On `Z8`, `Mean Prefix KL` and all `every_prefix` / `distill` rows below were computed before commit `7379c5d` fixed `aggbench/sequential.py:prefix_bayes_logits` for `Z8` (`sum / (n_k + tau^2)` instead of `mean`). Final-step `Regret` and `KL to Bayes` for `loss="final"` arms are unaffected because `batch.bayes["logit"]` used `Z8`'s exact `PrivateReadEnv._episode` posterior.

| Arm / Method | Axis / Category | Token Encoder | Vis / `own` | Regret (3-Seed Mean ± SD) | KL to Bayes (Mean ± SD) | Mean Prefix KL ($t=1..5$) | `untrained_init` | `random_network` | Status / Note |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `bayes` | Ceiling | — | oracle | `0.0000 ± 0.0000` | `0.0000 ± 0.0000` | `0.0000` | — | — | Exact Bayes Ceiling (`uses_oracle=True`) |
| `oracle_clearing` | Ceiling (Scalar) | — | oracle | `0.0236 ± 0.0000` | `0.0168 ± 0.0000` | — | — | — | Scalar Clearing Ceiling (`uses_oracle=True`) |
| **`extremized_log_pool`** | **Baseline (Scalar)** | — | reports | **`0.0344 ± 0.0000`** | **`0.0302 ± 0.0000`** | — | — | — | **Best Non-Oracle Method on `Z8`** |
| `B_loss_distill_deepsets` | `(B)` Oracle Distill Ceiling | `deepsets` | `none` (`True`) | `0.0348 ± 0.0022` | `0.0286 ± 0.0007` | `0.0254*` | `0.4889` | `0.4153` | Oracle Ceiling (`uses_oracle=True`; `[Pre-7379c5d Z8 Prefix Bug]`) |
| `V_vis_full_deepsets` | `Vis` Full + DeepSets | `deepsets` | `full` (`True`) | `0.0391 ± 0.0018` | `0.0352 ± 0.0021` | `0.0499*` | `0.4449` | `0.4963` | `● Informative` (Trails `extremized_log_pool` `0.0344`) |
| `A_attn_last_deepsets` | `(A)` Last-Layer Only | `deepsets` | `none` (`True`) | `0.0393 ± 0.0043` | `0.0330 ± 0.0015` | `0.0512*` | `0.5351` | `0.5263` | `● Informative` (Trails `extremized_log_pool` `0.0344`) |
| `A_attn_all_deepsets_reperm` | `(A)` + `V4` DeepSets | `deepsets` | `none` (`True`) | `0.0395 ± 0.0018` | `0.0303 ± 0.0006` | `0.0491*` | `0.4889` | `0.4137` | `● Informative` (Trails `extremized_log_pool` `0.0344`) |
| `B_loss_every_prefix_deepsets_reperm` | `(B)` Every-Prefix | `deepsets` | `none` (`True`) | `0.0413 ± 0.0011` | `0.0329 ± 0.0005` | `0.0316*` | `0.4889` | `0.4143` | `● Informative` (`[Pre-7379c5d Z8 Prefix Bug]`) |
| `C_state_recurrent_deepsets` | `(C)` GRU Summary | `deepsets` | `none` (`True`) | `0.0422 ± 0.0029` | `0.0366 ± 0.0036` | `0.0392*` | `0.4823` | `0.4459` | `● Informative` (`O(1)` Recurrent State) |
| `V_vis_full_deepsets_prefix` | `Vis` Full + Prefix | `deepsets` | `full` (`True`) | `0.0443 ± 0.0027` | `0.0392 ± 0.0020` | `0.0367*` | `0.4449` | `0.4941` | `● Informative` (`[Pre-7379c5d Z8 Prefix Bug]`) |
| `A_attn_all_deepsets` | `(A)` All + DeepSets | `deepsets` | `none` (`True`) | `0.0446 ± 0.0037` | `0.0372 ± 0.0039` | `0.0522*` | `0.4889` | `0.4178` | `● Informative` |
| `B_loss_every_prefix_deepsets` | `(B)` Every-Prefix | `deepsets` | `none` (`True`) | `0.0448 ± 0.0036` | `0.0380 ± 0.0012` | `0.0342*` | `0.4889` | `0.4160` | `● Informative` (`[Pre-7379c5d Z8 Prefix Bug]`) |
| `B_loss_distill` | `(B)` Oracle Distill Ceiling | `linear` | `none` (`True`) | `0.0449 ± 0.0018` | `0.0390 ± 0.0011` | `0.0349*` | `0.5095` | `0.4766` | Oracle Ceiling (`uses_oracle=True`; `[Pre-7379c5d Z8 Prefix Bug]`) |
| `B_loss_every_prefix_pos` | `(B)` Prefix + Learned Pos | `linear` | `none` (`True`) | `0.0573 ± 0.0016` | `0.0438 ± 0.0008` | `0.0425*` | `0.5020` | `0.4866` | `● Informative` (`[Pre-7379c5d Z8 Prefix Bug]`) |
| `B_loss_every_prefix_reperm` | `(B)` Prefix + Resample | `linear` | `none` (`True`) | `0.0641 ± 0.0027` | `0.0524 ± 0.0039` | `0.0488*` | `0.5095` | `0.4725` | `● Informative` (`[Pre-7379c5d Z8 Prefix Bug]`) |
| `C_state_recurrent_prefix` | `(C)` GRU + Prefix | `linear` | `none` (`True`) | `0.0690 ± 0.0059` | `0.0549 ± 0.0032` | `0.0512*` | `0.5186` | `0.5149` | `● Informative` (`[Pre-7379c5d Z8 Prefix Bug]`) |
| `A_attn_all_reperm` | `(A)` All + Order Resample | `linear` | `none` (`True`) | `0.0718 ± 0.0117` | `0.0592 ± 0.0049` | `0.0676*` | `0.5095` | `0.4726` | `● Informative` (Matched linear encoder vs `FCT_none`) |
| `C_state_feedback_prefix` | `(C)` Feedback + Prefix | `linear` | `none` (`True`) | `0.0730 ± 0.0095` | `0.0638 ± 0.0090` | `0.0635*` | `0.4857` | `0.4913` | `● Informative` (`[Pre-7379c5d Z8 Prefix Bug]`) |
| **`full_context_transformer_none`** | **Reference Bar (Linear)** | `linear` | `none` (`True`) | **`0.0733 ± 0.0123`** | **`0.0587 ± 0.0110`** | — | `0.6126` | `0.5139` | **Flat-Linear Reference Bar (20k pool)** |
| `full_context_transformer_full` | Reference Bar (`full`) | `linear` | `full` (`True`) | `0.0747 ± 0.0100` | `0.0629 ± 0.0091` | — | `0.4812` | `0.5395` | Flat-Linear Reference Bar (`full`) |
| `A_attn_first` | `(A)` Layer `(0,)` Only | `linear` | `none` (`True`) | `0.0786 ± 0.0106` | `0.0686 ± 0.0052` | `0.0845*` | `0.4854` | `0.4902` | `● Informative` |
| `C_state_recurrent_final` | `(C)` GRU (`final`) | `linear` | `none` (`True`) | `0.0794 ± 0.0141` | `0.0657 ± 0.0086` | `0.0784*` | `0.5186` | `0.5165` | `● Informative` |
| `V_vis_none_no_own` | `V2` Diagnostic | `linear` | `none` (`False`) | `0.0799 ± 0.0131` | `0.0724 ± 0.0142` | `0.0806*` | `0.4814` | `0.4454` | `● Informative` |
| `C_state_feedback_final` | `(C)` Feedback (`final`) | `linear` | `none` (`True`) | `0.0801 ± 0.0093` | `0.0707 ± 0.0094` | `0.0923*` | `0.4857` | `0.4911` | `● Informative` |
| `B_loss_every_prefix` | `(B)` Every-Prefix | `linear` | `none` (`True`) | `0.0804 ± 0.0103` | `0.0714 ± 0.0050` | `0.0651*` | `0.5095` | `0.4736` | `● Informative` (`[Pre-7379c5d Z8 Prefix Bug]`) |
| `A_attn_first_last` | `(A)` Layers `(0, 3)` | `linear` | `none` (`True`) | `0.0831 ± 0.0149` | `0.0751 ± 0.0164` | `0.0888*` | `0.4828` | `0.4617` | `● Informative` |
| `V_vis_full` | `Vis` Full (`own=True`) | `linear` | `full` (`True`) | `0.0856 ± 0.0051` | `0.0760 ± 0.0044` | `0.0838*` | `0.5043` | `0.4773` | `● Informative` |
| `V_vis_noisy_no_own` | `V2` Diagnostic | `linear` | `noisy` (`False`) | `0.0872 ± 0.0069` | `0.0747 ± 0.0009` | `0.0846*` | `0.5019` | `0.5160` | `● Informative` |
| `A_attn_last` | `(A)` Last-Layer Only | `linear` | `none` (`True`) | `0.0891 ± 0.0154` | `0.0773 ± 0.0090` | `0.0956*` | `0.4672` | `0.4868` | `● Informative` |
| `A_attn_all` | `(A)` All Layers `(0,1,2,3)` | `linear` | `none` (`True`) | `0.0902 ± 0.0185` | `0.0805 ± 0.0135` | `0.0919*` | `0.5095` | `0.4733` | `● Informative` |
| `V_vis_noisy` | `Vis` Noisy (`own=True`) | `linear` | `noisy` (`True`) | `0.0920 ± 0.0029` | `0.0824 ± 0.0041` | `0.0922*` | `0.5043` | `0.4780` | `● Informative` |
| `sum_logodds` | Baseline (Scalar) | — | reports | `0.0952 ± 0.0000` | `0.0939 ± 0.0000` | — | — | — | Baseline |
| `A_attn_every_other` | `(A)` Layers `(1, 3)` | `linear` | `none` (`True`) | `0.0957 ± 0.0246` | `0.0837 ± 0.0188` | `0.1015*` | `0.4809` | `0.4629` | `● Informative` |
| `log_pool` | Baseline (Scalar) | — | reports | `0.1178 ± 0.0000` | `0.1118 ± 0.0000` | — | — | — | Baseline |
| `linear_pool` | Baseline (Scalar) | — | reports | `0.1636 ± 0.0000` | `0.1568 ± 0.0000` | — | — | — | Baseline |
| `A_attn_none` | `(A)` No Cross-Trader Attn | `linear` | `none` (`True`) | `0.2476 ± 0.0051` | `0.2434 ± 0.0018` | `0.1835*` | `0.4894` | `0.4956` | `● No gain` (Ablation Floor) |
| **`A_attn_all_no_es`** | **`#3856` No Early-Stop** | `linear` | `none` (`True`) | **`0.8677 ± 0.0840`** | **`0.8393 ± 0.0283`** | `0.7572*` | `0.5095` | `0.5085` | **`● Informative` (Memorizes 20k Pool)** |

---

## Axis `(B)` Prefix KL Curves Across Trader Positions $t = 1 \dots 5$ (`E2` Exact Prefix Bayes; `Z8` Marked `*` Pre-`7379c5d`)

| Environment | Arm | Supervision Loss | $t=1$ | $t=2$ | $t=3$ | $t=4$ | $t=5$ (Final) | Mean Prefix KL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `E2` | `A_attn_none` | `final` (no cross-attn) | `0.0228` | `0.1175` | `0.1804` | `0.2106` | `0.2333` | `0.1529` |
| `E2` | `A_attn_all_reperm` | `final` (`linear`) | `0.0833` | `0.0565` | `0.0557` | `0.0563` | `0.0563` | `0.0616` |
| `E2` | `B_loss_every_prefix_reperm` | `every_prefix` (`linear`) | `0.0157` | `0.0343` | `0.0462` | `0.0536` | `0.0561` | `0.0412` |
| `E2` | `A_attn_all_deepsets_reperm` | `final` (`deepsets`) | `0.0792` | `0.0321` | `0.0287` | `0.0313` | `0.0322` | `0.0407` |
| `E2` | `B_loss_every_prefix_deepsets_reperm` | `every_prefix` (`deepsets`) | `0.0062` | `0.0179` | `0.0268` | `0.0335` | `0.0358` | `0.0240` |
| `E2` | `B_loss_distill_deepsets` | `distill` (`deepsets`, `uses_oracle=True`) | `0.0006` | `0.0108` | `0.0173` | `0.0219` | `0.0244` | `0.0150` |
| `Z8*` | `A_attn_none` | `final` (no cross-attn, pre-`7379c5d` prefix eval) | `0.0277*` | `0.1527*` | `0.2213*` | `0.2480*` | `0.2681*` | `0.1835*` |
| `Z8*` | `A_attn_all_reperm` | `final` (`linear`, pre-`7379c5d` prefix eval) | `0.0614*` | `0.0620*` | `0.0681*` | `0.0722*` | `0.0741*` | `0.0676*` |
| `Z8*` | `B_loss_every_prefix_reperm` | `every_prefix` (`linear`, pre-`7379c5d` bug) | `0.0117*` | `0.0424*` | `0.0582*` | `0.0658*` | `0.0662*` | `0.0488*` |
| `Z8*` | `A_attn_all_deepsets_reperm` | `final` (`deepsets`, pre-`7379c5d` prefix eval) | `0.0877*` | `0.0370*` | `0.0372*` | `0.0412*` | `0.0421*` | `0.0491*` |
| `Z8*` | `B_loss_every_prefix_deepsets_reperm` | `every_prefix` (`deepsets`, pre-`7379c5d` bug) | `0.0041*` | `0.0250*` | `0.0378*` | `0.0449*` | `0.0465*` | `0.0316*` |
| `Z8*` | `B_loss_distill_deepsets` | `distill` (`uses_oracle=True`, pre-`7379c5d` bug) | `0.0013*` | `0.0187*` | `0.0307*` | `0.0370*` | `0.0391*` | `0.0254*` |

---

## Artifacts in This Directory

- [`summary.json`](./summary.json): Full 3-seed aggregated metrics, prefix KL curves, training logs, and per-seed bootstrap CIs (`commit 605faec`).
- [`tt_abc_e2/result.json`](./tt_abc_e2/result.json): Schema-validated `aggbench.result/v1` artifact for `E2`.
- [`tt_abc_z8/result.json`](./tt_abc_z8/result.json): Schema-validated `aggbench.result/v1` artifact for `Z8`.
