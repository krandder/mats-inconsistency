# EXPLORATORY Baseline: Kelvin's Axis (D) 3-D `(depth, token-index, trader-index)` Piece-by-Trader Grid Transformer Layouts on Unaligned `E2`, `Z8`, and `E3-sybil` (`tt-grid-e2z8-20261021`)

Serves: Exploratory 20k-pool baseline for Kelvin's Axis `(D)` 3-D `(depth, token-index, trader-index)` grid architecture (`aggbench/trader_transformer.py` layouts `"trader"`, `"grid"`, `"axial"`, `"grid_full"`, `"grid_rankfree"`, and `"trader_deepsets"`) on `alignment="unaligned"` `E2`, `Z8` (`PrivateReadEnv`), and `E3-sybil` (`3` seeds `0, 1, 2` on NVIDIA L4 GPU).
Builds on: Kelvin's 3-D grid transformer design (`research-orientation-2026-10-20-addendum.md` lines 38–53, `aggbench/trader_transformer.py` commit `eb3cd7d`) and `gate-gemini-tt-2026-10-21.md` (`#3859`).

**Status:** `● Informative (EXPLORATORY baseline)` (`test` splits untouched; `alignment="unaligned"`; fixed `n_train=20000` pool shared across seeds `0, 1, 2` with 10% held-out train-split early stopping; `n_eval=2000` dev episodes; GPU: `NVIDIA L4`; runner commit `f6ba763`).

> [!IMPORTANT]
> **Gate Audit Disclosures (`gate-gemini-tt-2026-10-21.md`, `#3859`):**
> 1. **No environment is below `0.01` nats regret to Bayes (`0.0000`) yet:** The best learned arms in this 20k-pool exploratory run reach `0.0373 ± 0.0059` (`kl = 0.0315`) on `E2`, `0.0382 ± 0.0010` (`kl = 0.0305`) on `Z8`, and `0.0403 ± 0.0025` (`kl = 0.0389`) on `E3-sybil` — still `3.7x–4.0x` above `0.01` nats.
> 2. **On `Z8`, `extremized_log_pool` (`0.0344`, `kl = 0.0302`) beats every learned arm (`0.0382–0.0902`).**
> 3. **Encoder difference vs. `full_context_transformer_none` (`Must-Fix 2`):** In this exploratory run, `full_context_transformer_none` (`0.0615` on `E2`, `0.0733` on `Z8`, `0.0728` on `E3-sybil`) used flat linear 49-D trader tokens (`full_context`, including `report_logit / 4`), whereas the 3-D grid arms used 12 piece tokens per trader (`[present, obs_k, own_cov_k, own_mu_k]`, no scalar report). The unconfounded piece-token bidirectional bar (`bar_fct_piece_bidir`) and `deepsets` bidirectional bar (`bar_fct_deepsets_bidir`) are evaluated in the follow-up run.
> 4. **Mask vs. Readout Confound in `grid_rankfree` (`Must-Fix 6`):** In commit `f6ba763`, `D_layout_grid_rankfree_*` used pooled readout (`0.5 * last_cell + 0.5 * mean_present_cells`), whereas `D_layout_grid_full_final`, `D_layout_grid_final`, and `D_layout_axial_final` read cell `T-1` (`h[:, T-1::T]`, which is often an empty padding slot when a trader holds `< 12` pieces). Thus the difference between `D_layout_grid_rankfree_final` (`0.0373`) and `D_layout_grid_full_final` (`0.0443`) in this table confounds attention mask ($r' \le r$ vs $t' \le t \wedge r' \le r$) with readout (`pooled` vs `cell T-1`). The clean `2x2` factorial (`grid_full` vs `grid_rankfree` $\times$ `cellT1` vs `pooled`) is run in the follow-up grid experiment.
> 5. **`E3-biased` Dropped (`Must-Fix 3`):** With identical seeds, `E3-biased` and `E2` share byte-identical `obs`, `own_model`, `truth`, and `bayes` (only `reports` differ, which piece-token grids and `deepsets` never read). `E3-biased` is therefore excluded from piece/grid claims.
> 6. **Pre-`7379c5d` `Z8` Prefix Bayes Bug (`Must-Fix 4`) & Fixed 20k Pool (`Must-Fix 5`):** `Z8` `every_prefix` rows (`D_layout_grid_rankfree_prefix`, `D_layout_grid_prefix`, `D_layout_trader_deepsets_prefix`, `D_layout_trader_prefix`) were trained before commit `7379c5d` fixed `Z8` `prefix_bayes_logits` in `aggbench/sequential.py`. Also, `env.sample("train", 20000)` was sampled once per environment and shared across the 3 seeds (`± SD` reflects init and trader-order variance on that fixed 20k pool).

---

## Mandatory `(a)–(d)` Gate Disclosure (`#3843` / `#3847`)

- **`(a)` What a token is:**
  - In `layout="trader"` (`D_layout_trader_*`): one flat vector per trader $r \in \{0, \dots, N-1\}$ (`in_dim = 49`).
  - In `layout="trader"` with `trader_encoder="deepsets"` (`D_layout_trader_deepsets_*`): a 2-layer DeepSets encoder over trader $r$'s $K=12$ piece cells `[present, obs_k, own_cov_k, own_mu_k]` (`in_dim = 4`), mean-pooled across present pieces into 1 token per trader ($d_{\text{model}}=128$).
  - In 3-D grid layouts (`"grid"`, `"axial"`, `"grid_full"`, `"grid_rankfree"`): a 2-D grid of $N \times T = 5 \times 12 = 60$ cell tokens per episode, where cell $(r, t)$ represents trader $r \in \{0, \dots, N-1\}$ (in arrival order $\sigma_e$) and compacted present-piece rank $t = \text{cumsum}(\text{present}) - 1 \in \{0, \dots, K-1\}$, with cell feature vector `[present, obs_k, own_cov_k, own_mu_k, (key_cov_k, key_mu_k)]` (`in_dim = 4` at `visibility="none", own_model=True`; `in_dim = 4` at `visibility="noisy", own_model=False`).
- **`(b)` What a layer is:**
  - Pre-norm Transformer block ($L=4$ layers, $H=4$ heads, $d_{\text{model}}=64$ for grid layouts [`200,449` params] and $d_{\text{model}}=128$ for 1-token-per-trader layouts [`799,873–827,137` params]):
    * `layout="grid"` (`D_layout_grid_*`): **per-head axial attention** in every layer — half the heads ($H/2 = 2$) attend within-trader across piece ranks $(r' = r, t' \le t)$ and half ($H/2 = 2$) attend across traders at the same piece rank $(t' = t, r' \le r)$.
    * `layout="axial"` (`D_layout_axial_final`): **alternating axial layers** — even layers ($l \in \{0, 2\}$) attend within-trader $(r' = r, t' \le t)$; odd layers ($l \in \{1, 3\}$) attend across traders $(t' = t, r' \le r)$.
    * `layout="grid_full"` (`D_layout_grid_full_*`): **2-D causal cone attention** over $(t' \le t \wedge r' \le r)$ (with readout at cell $t = T-1$ in this run).
    * `layout="grid_rankfree"` (`D_layout_grid_rankfree_*`): **rank-free trader-causal cone attention** over $(r' \le r, \forall t' \in \{0, \dots, T-1\})$ (with pooled `0.5 * last_cell + 0.5 * mean_present_cells` readout in this run; see `Must-Fix 6` note above).
- **`(c)` What crosses between traders and at what width:**
  - In `layout="trader*"`: one $d_{\text{model}}=128$ vector per trader crosses via causal self-attention.
  - In `layout="grid*"`: $T=12$ piece-cell vectors of width $d_{\text{model}}=64$ per trader cross between traders $r' \le r$ via axial or 2-D cone K/V attention.
- **`(d)` Regret to Bayes (`0.0000`) and Comparison Bars (`alignment="unaligned"`):**
  - Evaluated on `n_eval = 2000` unaligned `dev` episodes against `bayes` (`0.0000`), `oracle_clearing` (`0.0166` on `E2`, `0.0236` on `Z8`), `extremized_log_pool` (`0.0448` on `E2`, `0.0344` on `Z8`, `0.0778` on `E3-sybil`), and flat-token `full_context_transformer_none` (`0.0615 ± 0.0066` on `E2`, `0.0733 ± 0.0123` on `Z8`, `0.0728 ± 0.0062` on `E3-sybil`), with `untrained_init` and `random_network` controls.

---

## Why, What We Did, Result, So What, Next Step

- **Why:** Kelvin proposed a 3-D `(depth, token-index, trader-index)` grid where each trader contributes $T=K$ piece tokens, and asked how axial/cone attention across the 2-D `(piece, trader)` grid compares against 1-token-per-trader compression (`layout="trader"` and `trader_encoder="deepsets"`).
- **What we did:** Ran `benchmarks/explore_trader_transformer_abc_grid.py --suite grid --envs E2,Z8,E3-sybil --seeds 0,1,2` (`n_train=20000, n_eval=2000, steps=2000, batch=512, width=128, grid_width=64, layers=4`, commit `f6ba763`) on an NVIDIA L4 GPU.
- **Result (`● Informative — Exploratory Baseline`):**
  1. **Piece-Token 3-D Grids and `trader_deepsets` Both Avoid the Flat-Token Unaligned Bottleneck:**
     - On `E2` (`3` seeds), piece-token grid arms (`0.0373–0.0443`, `200k` params) and 1-token-per-trader `D_layout_trader_deepsets_final` (`0.0447 ± 0.0009`, `827k` params) all substantially outperform flat-token `D_layout_trader_final` (`0.0841 ± 0.0074`).
     - On `Z8` (`3` seeds), piece-token grid arms reach `0.0382–0.0434` vs flat-token `D_layout_trader_final` (`0.0902 ± 0.0185`), **but all learned arms still trail `extremized_log_pool` (`0.0344`)**.
     - On `E3-sybil` (`3` seeds), where `extremized_log_pool` degrades to `0.0778` because it cannot de-duplicate Sybil reports, piece-token grid arms reach `0.0403–0.0502` and `D_layout_trader_deepsets_final` reaches `0.0450 ± 0.0041`.
  2. **Confound Between Mask and Readout in `grid_rankfree` vs. `grid_full` (`Must-Fix 6`):**
     - While `D_layout_grid_rankfree_final` (`0.0373` on `E2`, `0.0382` on `Z8`, `0.0403` on `E3-sybil`) improved over `D_layout_grid_full_final` (`0.0443` on `E2`, `0.0402` on `Z8`, `0.0502` on `E3-sybil`), `D_layout_grid_rankfree_*` also used present-cell pooled readout whereas `D_layout_grid_full_final` read cell `T-1` (`h[:, T-1::T]`). Isolating the mask effect from the readout effect requires a `2x2` factorial (`grid_full` vs `grid_rankfree` $\times$ `cellT1` vs `pooled`).
- **So what:** Piece-level tokenization (either via 3-D grids at `200k` params or intra-trader `deepsets` pooling at `827k` params) removes the flat 49-D unaligned slot bottleneck, but isolating the architectural contribution of the 3-D grid mask requires (1) the `2x2` mask $\times$ readout factorial and (2) an unconfounded bidirectional piece-token `FullContextTransformer` bar (`bar_fct_piece_bidir`).
- **Next step:** Run the clean `2x2` (`grid_full` vs `grid_rankfree` $\times$ `cellT1` vs `pooled`) factorial with `bar_fct_piece_bidir`, per-seed resampled train pools, and exact post-`7379c5d` `Z8` prefix Bayes.

---

## Master Table: Axis `(D)` 3-D Grid Layouts on Unaligned `E2`, `Z8`, and `E3-sybil` (`dev` $n=2000$, 3 Seeds `0, 1, 2`, Fixed 20k Train Pool)

| Arm / Method | Layout, Mask & Readout | Loss | Params | `E2` Regret (`KL`) | `Z8` Regret (`KL`) | `E3-sybil` Regret (`KL`) | Status / Note |
|---|---|---|---:|---:|---:|---:|---|
| `bayes` | Exact Oracle | — | — | `0.0000 ± 0.0000` (`0.0000`) | `0.0000 ± 0.0000` (`0.0000`) | `0.0000 ± 0.0000` (`0.0000`) | Exact Bayes Ceiling (`uses_oracle=True`) |
| `oracle_clearing` | Scalar Clearing Ceiling | — | — | `0.0166 ± 0.0000` (`0.0158`) | `0.0236 ± 0.0000` (`0.0168`) | `0.0166 ± 0.0000` (`0.0158`) | Scalar Ceiling (`uses_oracle=True`) |
| `D_layout_grid_rankfree_final` | 3-D Rank-Free ($r' \le r, \forall t'$), `pooled` readout | `final` | `200,449` | `0.0373 ± 0.0059` (`0.0315`) | `0.0382 ± 0.0010` (`0.0305`) | `0.0403 ± 0.0025` (`0.0389`) | `● Informative` (Mask+readout confounded; see `Must-Fix 6`) |
| `D_layout_grid_rankfree_prefix` | 3-D Rank-Free ($r' \le r, \forall t'$), `pooled` readout | `every_prefix` | `200,449` | `0.0378 ± 0.0016` (`0.0309`) | `0.0405 ± 0.0011*` (`0.0336*`) | `0.0429 ± 0.0022` (`0.0387`) | `● Informative` (`*Z8` pre-`7379c5d` prefix bug) |
| `D_layout_grid_prefix` | 3-D Per-Head Axial Grid, `cell T-1` readout | `every_prefix` | `200,449` | `0.0404 ± 0.0054` (`0.0342`) | `0.0447 ± 0.0023*` (`0.0370*`) | `0.0463 ± 0.0040` (`0.0391`) | `● Informative` (`*Z8` pre-`7379c5d` prefix bug) |
| `D_layout_axial_final` | 3-D Alternating-Layer Axial, `cell T-1` readout | `final` | `200,449` | `0.0415 ± 0.0052` (`0.0351`) | `0.0400 ± 0.0067` (`0.0334`) | `0.0473 ± 0.0041` (`0.0405`) | `● Informative` |
| `D_layout_grid_final` | 3-D Per-Head Axial Grid, `cell T-1` readout | `final` | `200,449` | `0.0426 ± 0.0041` (`0.0361`) | `0.0434 ± 0.0064` (`0.0362`) | `0.0460 ± 0.0067` (`0.0402`) | `● Informative` |
| `D_layout_grid_full_final` | 3-D Cone ($t' \le t \wedge r' \le r$), `cell T-1` readout | `final` | `200,449` | `0.0443 ± 0.0049` (`0.0373`) | `0.0402 ± 0.0007` (`0.0308`) | `0.0502 ± 0.0072` (`0.0472`) | `● Informative` |
| `D_layout_trader_deepsets_final` | 1-Token/Trader (`deepsets`) | `final` | `827,137` | `0.0447 ± 0.0009` (`0.0367`) | `0.0446 ± 0.0037` (`0.0372`) | `0.0450 ± 0.0041` (`0.0390`) | `● Informative` |
| `extremized_log_pool` | Scalar Heuristic Baseline | — | — | `0.0448 ± 0.0000` (`0.0407`) | **`0.0344 ± 0.0000` (`0.0302`)** | `0.0778 ± 0.0000` (`0.0747`) | Scalar Baseline (**#1 on `Z8`**) |
| `D_layout_grid_full_pos` | 3-D Cone + Learned Rank Pos, `cell T-1` readout | `final` | `208,641` | `0.0468 ± 0.0013` (`0.0415`) | `0.0426 ± 0.0004` (`0.0349`) | `0.0638 ± 0.0024` (`0.0582`) | `● Informative` |
| `D_layout_trader_deepsets_prefix` | 1-Token/Trader (`deepsets`) | `every_prefix` | `827,137` | `0.0488 ± 0.0026` (`0.0397`) | `0.0448 ± 0.0036*` (`0.0380*`) | `0.0491 ± 0.0006` (`0.0426`) | `● Informative` (`*Z8` pre-`7379c5d` prefix bug) |
| `D_layout_grid_rankfree_noisy_no_own` | 3-D Rank-Free (`noisy, own=False`), `pooled` | `final` | `200,449` | `0.0502 ± 0.0082` (`0.0440`) | `0.0527 ± 0.0037` (`0.0399`) | `0.0634 ± 0.0087` (`0.0570`) | `● Informative` |
| `D_layout_grid_full_noisy_no_own` | 3-D Cone (`noisy, own=False`), `cell T-1` | `final` | `200,449` | `0.0542 ± 0.0018` (`0.0452`) | `0.0524 ± 0.0024` (`0.0397`) | `0.0609 ± 0.0074` (`0.0558`) | `● Informative` |
| **`full_context_transformer_none`** | **Flat-Linear Reference Bar (`none`)** | `final` | `~550,000` | **`0.0615 ± 0.0066` (`0.0605`)** | **`0.0733 ± 0.0123` (`0.0587`)** | **`0.0728 ± 0.0062` (`0.0700`)** | **Flat-Linear Reference Bar (20k pool)** |
| `D_layout_trader_prefix` | 1-Token/Trader (`linear`) | `every_prefix` | `799,873` | `0.0689 ± 0.0049` (`0.0650`) | `0.0804 ± 0.0103*` (`0.0714*`) | `0.0899 ± 0.0073` (`0.0854`) | `● Informative` |
| `full_context_transformer_full` | Flat-Linear Reference Bar (`full`) | `final` | `~555,000` | `0.0698 ± 0.0043` (`0.0642`) | `0.0747 ± 0.0100` (`0.0629`) | `0.0729 ± 0.0076` (`0.0686`) | Flat-Linear Reference Bar (`full`) |
| `D_layout_trader_final` | 1-Token/Trader (`linear`) | `final` | `799,873` | `0.0841 ± 0.0074` (`0.0828`) | `0.0902 ± 0.0185` (`0.0805`) | `0.0968 ± 0.0094` (`0.0953`) | `● Informative` |

---

## Artifacts in This Directory

- [`summary.json`](./summary.json): Multi-seed summary and prefix KL curves for `E2`, `Z8`, and `E3-sybil` (`commit f6ba763`).
- [`tt_grid_e2/result.json`](./tt_grid_e2/result.json): Schema-validated `aggbench.result/v1` artifact for `E2`.
- [`tt_grid_z8/result.json`](./tt_grid_z8/result.json): Schema-validated `aggbench.result/v1` artifact for `Z8`.
- [`tt_grid_e3-sybil/result.json`](./tt_grid_e3-sybil/result.json): Schema-validated `aggbench.result/v1` artifact for `E3-sybil`.
