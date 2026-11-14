# EXPLORATORY Baseline: Kelvin's Trader-Transformer Axes (A), (B), (C) in 1-Pass Fresh-Episode Regime (`800 x 128 = 102.4k` Episodes) on `E2`, `Z8`, `Z1-blind`, and `Z5` (`tt-abc-fresh102k-20261021`)

Serves: Exploratory 1-pass fresh-episode run (`--fresh-episodes --steps 800 --batch-size 128`, `102,400` fresh episodes per environment with zero episode reuse and no early stopping) across `E2`, `Z8` (`PrivateReadEnv`), `Z1-blind` (`CommonCauseEnv`), and `Z5` (`MultiQuestionEnv`) (`3` seeds `0, 1, 2` on NVIDIA L4 GPU).
Builds on: `#3856` (`PrecomputedFreshStream` / 1-pass fresh training diagnostic) and `gate-gemini-tt-2026-10-21.md` (`#3859`, Must-Fixes 1, 2, 4, 5).

**Status:** `● Informative (EXPLORATORY baseline)` (`test` splits untouched; `alignment="unaligned"`; `steps=800, batch=128` -> `n_train=102400` fresh episodes; `n_eval=2000` dev episodes; `width=128, layers=4, heads=4`; GPU: `NVIDIA L4`; runner commit `e5039a60da52fbffc5208c549cd866cacceba307`).

> [!IMPORTANT]
> **Gate Audit Disclosures (`gate-gemini-tt-2026-10-21.md`, `#3859`):**
> 1. **Exact CLI Args (`Must-Fix 5`):** `PYTHONPATH=. python3 benchmarks/explore_trader_transformer_abc_grid.py --suite abc --envs E2,Z8,Z1-blind,Z5 --seeds 0,1,2 --fresh-episodes --steps 800 --batch-size 128 --n-eval 2000 --width 128 --layers 4 --out /root/runs/tt-abc-fresh-0d9/summary.json`. Notice this intermediate diagnostic used `800 x 128 = 102,400` fresh episodes (`15x` smaller than the `3000 x 512 = 1,536,000` fresh-episode headline budget in `0d94ef1`, which is run in `tt-gate-xor-fresh`).
> 2. **Fresh-Mode Alias Note (`Must-Fix 5`):** In commit `e5039a6`, `CudaTraderTransformer.__init__` set `val_frac=0.0, patience=None, permute_traders_train=True` whenever `fresh_episodes=True`, making `A_attn_all`, `A_attn_all_reperm`, and `A_attn_all_no_es` identical in fresh mode (and likewise `B_loss_every_prefix` and `B_loss_every_prefix_reperm`). This was fixed in commit `204e78a`.
> 3. **Pre-`2546bb2` `Z8` Prefix Bayes Bug (`Must-Fix 4`):** Commit `e5039a6` ran before commit `2546bb2` fixed `Z8` `prefix_bayes_logits` in `aggbench/sequential.py`; `Z8` `every_prefix` / `distill` and `mean_pref_kl` rows in this table are marked `*` and superseded by the post-`204e78a` runs (`tt-gate-e2z8syb` and `tt-gate-xor-fresh`).
> 4. **No environment is below `0.01` nats at `800 x 128` (`102.4k` episodes):** At `102.4k` fresh episodes, best non-oracle regrets are `0.0351` on `E2`, `0.0345` on `Z8` (vs `extremized_log_pool` `0.0344`), `0.0188` on `Z1-blind` (where `floor_none = 0.0124`), and `0.0152` on `Z5` (which drops to `0.0031–0.0043` at the full `3000 x 512` fresh budget in `tt-gate-xor-fresh`).

---

## Mandatory `(a)–(d)` Gate Disclosure (`#3843` / `#3847`)

- **`(a)` What a token is:** Flat linear trader token (`full_context`) on all 4 environments, plus 2-layer `trader_encoder="deepsets"` over 12 piece cells (`no scalar report`) on `E2` and `Z8`.
- **`(b)` What a layer is:** Pre-norm Transformer block (`TraderNet`, $L=4$ layers, $H=4$ heads, $d_{\text{model}}=128$).
- **`(c)` What crosses between traders and at what width:** Width $d_{\text{model}}=128$ vectors cross causally ($r' \le r$) via `kv`, `recurrent` (GRU), or `feedback` memory.
- **`(d)` Regret to Bayes (`0.0000`) and Comparison Bars (`alignment="unaligned"`):** Evaluated on `n_eval = 2000` unaligned `dev` episodes against `bayes` (`0.0000`), `floor_none` (`0.0000` on `E2`/`Z5`, `0.0124` on `Z1-blind`), `oracle_clearing`, `extremized_log_pool`, and `full_context_transformer_none` (`PrecomputedFreshStream` at `800 x 128`), with `untrained_init` and `random_network` controls in [`tt_abc_e2/result.json`](./tt_abc_e2/result.json), [`tt_abc_z8/result.json`](./tt_abc_z8/result.json), [`tt_abc_z1-blind/result.json`](./tt_abc_z1-blind/result.json), and [`tt_abc_z5/result.json`](./tt_abc_z5/result.json).

---

## Key Diagnostic Finding: `102.4k` Fresh Episodes Eliminate Fixed-Pool Memorization & Close the Flat-Linear Gap

| Environment | `A_attn_all_no_es` (20k Pool) | `A_attn_all_no_es` (102.4k Fresh) | `full_context_transformer_none` (20k Pool) | `full_context_transformer_none` (102.4k Fresh) | `A_attn_last_deepsets` (102.4k Fresh) | `C_state_recurrent` Best (102.4k Fresh) |
|---|---:|---:|---:|---:|---:|---:|
| **`E2`** | `0.9423 ± 0.0602` | **`0.0446 ± 0.0008`** | `0.0615 ± 0.0066` | **`0.0403 ± 0.0006`** | `0.0354 ± 0.0007` | `0.0379 ± 0.0008` (`deepsets`) |
| **`Z8`** | `0.8677 ± 0.0840` | **`0.0422 ± 0.0010`** | `0.0733 ± 0.0123` | **`0.0415 ± 0.0014`** | `0.0345 ± 0.0001` | `0.0347 ± 0.0003` (`deepsets`) |
| **`Z1-blind`** | — | **`0.0222 ± 0.0002`** | — | **`0.0188 ± 0.0015`** | — | `0.0229 ± 0.0011` (`linear`) |
| **`Z5`** | `0.6049 ± 0.0318` | **`0.0226 ± 0.0015`** | `0.0265 ± 0.0015` | **`0.0152 ± 0.0016`** | — | `0.0190 ± 0.0013` (`linear`) |

1. **Memorization Eliminated (`#3856`):** Drawing fresh episodes per gradient step without early stopping (`A_attn_all_no_es`) lowers dev regret from `0.60–0.94` (on the 20k pool) to `0.022–0.045` across all 4 environments.
2. **Flat-Linear `FullContextTransformer` Improves Sharply with Fresh Unaligned Permutations:** On `E2` and `Z8`, `full_context_transformer_none` improves from `0.0615` / `0.0733` (20k pool) to `0.0403 ± 0.0006` / `0.0415 ± 0.0014` (`102.4k` fresh), confirming `opus-mats`'s diagnosis (`#3859`) that the 20k pool starved the flat 49-D linear projection of slot-permutation diversity.
3. **`Z1-blind` Near the `floor_none = 0.0124` Information Floor:** On `Z1-blind` (where hidden common-cause group assignments give an exact `visibility="none"` information floor of `floor_none = 0.0124` regret to full-information Bayes), `full_context_transformer_none` reaches `0.0188 ± 0.0015` (`0.0064` nats above `floor_none`), `C_state_feedback_final` reaches `0.0209 ± 0.0018`, and `A_attn_last` reaches `0.0219 ± 0.0013`.

---

## Master Table: `E2`, `Z8`, `Z1-blind`, and `Z5` at `800 x 128` (`102.4k` Fresh Episodes, 3 Seeds `0, 1, 2`)

| Arm / Method | Encoder | `E2` Regret (`KL`) | `Z8` Regret (`KL`) | `Z1-blind` Regret (`KL`) | `Z5` Regret (`KL`) | Status / Note |
|---|---|---:|---:|---:|---:|---|
| `bayes` | — | `0.0000` (`0.0000`) | `0.0000` (`0.0000`) | `0.0000` (`0.0000`) | `0.0000` (`0.0000`) | Exact Bayes Ceiling (`uses_oracle=True`) |
| `floor_none` | — | — | — | `0.0124` (`0.0093`) | `0.0000` (`0.0000`) | Exact `none` Floor (`uses_oracle=True`) |
| `oracle_clearing` | — | `0.0166` (`0.0158`) | `0.0236` (`0.0168`) | `0.0000` (`0.0000`) | `0.0000` (`0.0000`) | Scalar Ceiling (`uses_oracle=True`) |
| `B_loss_distill_deepsets` | `deepsets` | `0.0348 ± 0.0003` (`0.0283`) | `0.0338 ± 0.0010*` (`0.0281*`) | — | — | Oracle Ceiling (`uses_oracle=True`) |
| `A_attn_all_deepsets` | `deepsets` | `0.0351 ± 0.0011` (`0.0301`) | `0.0347 ± 0.0006` (`0.0288`) | — | — | `● Informative` |
| `A_attn_last_deepsets` | `deepsets` | `0.0354 ± 0.0007` (`0.0304`) | `0.0345 ± 0.0001` (`0.0291`) | — | — | `● Informative` (1 Attn layer) |
| `V_vis_full_deepsets` | `deepsets` | `0.0351 ± 0.0013` (`0.0301`) | `0.0352 ± 0.0003` (`0.0291`) | — | — | `● Informative` |
| `C_state_recurrent_deepsets` | `deepsets` | `0.0379 ± 0.0008` (`0.0326`) | `0.0347 ± 0.0003` (`0.0296`) | — | — | `● Informative` (`O(1)` GRU state) |
| `B_loss_every_prefix_deepsets` | `deepsets` | `0.0422 ± 0.0016` (`0.0358`) | `0.0416 ± 0.0012*` (`0.0359*`) | — | — | `● Informative` |
| `extremized_log_pool` | — | `0.0448` (`0.0407`) | **`0.0344` (`0.0302`)** | `0.0729` (`0.0680`) | `0.1341` (`0.1286`) | Scalar Baseline |
| `B_loss_distill` | `linear` | `0.0393 ± 0.0009` (`0.0345`) | `0.0395 ± 0.0012*` (`0.0323*`) | `0.0183 ± 0.0016` (`0.0143`) | `0.0123 ± 0.0007` (`0.0099`) | Oracle Ceiling (`uses_oracle=True`) |
| **`full_context_transformer_none`** | `linear` | **`0.0403 ± 0.0006` (`0.0375`)** | **`0.0415 ± 0.0014` (`0.0333`)** | **`0.0188 ± 0.0015` (`0.0149`)** | **`0.0152 ± 0.0016` (`0.0161`)** | **Flat-Linear Bar (102.4k fresh)** |
| `full_context_transformer_full` | `linear` | `0.0416 ± 0.0011` (`0.0382`) | `0.0434 ± 0.0016` (`0.0350`) | `0.0198 ± 0.0013` (`0.0158`) | `0.0183 ± 0.0011` (`0.0184`) | Flat-Linear Bar (`full`) |
| `B_loss_every_prefix_pos` | `linear` | `0.0421 ± 0.0002` (`0.0387`) | `0.0391 ± 0.0007*` (`0.0324*`) | `0.0257 ± 0.0012` (`0.0209`) | `0.0324 ± 0.0003` (`0.0324`) | `● Informative` |
| `C_state_recurrent_prefix` | `linear` | `0.0428 ± 0.0006` (`0.0409`) | `0.0410 ± 0.0013*` (`0.0337*`) | `0.0235 ± 0.0019` (`0.0180`) | `0.0192 ± 0.0016` (`0.0189`) | `● Informative` |
| `C_state_recurrent_final` | `linear` | `0.0439 ± 0.0007` (`0.0415`) | `0.0419 ± 0.0008` (`0.0352`) | `0.0229 ± 0.0011` (`0.0181`) | `0.0190 ± 0.0013` (`0.0161`) | `● Informative` |
| `C_state_feedback_prefix` | `linear` | `0.0433 ± 0.0038` (`0.0397`) | `0.0420 ± 0.0010*` (`0.0356*`) | `0.0216 ± 0.0037` (`0.0163`) | `0.0178 ± 0.0026` (`0.0182`) | `● Informative` |
| `C_state_feedback_final` | `linear` | `0.0445 ± 0.0015` (`0.0394`) | `0.0423 ± 0.0013` (`0.0365`) | `0.0209 ± 0.0018` (`0.0165`) | `0.0210 ± 0.0014` (`0.0176`) | `● Informative` |
| `A_attn_last` | `linear` | `0.0436 ± 0.0006` (`0.0389`) | `0.0405 ± 0.0008` (`0.0342`) | `0.0219 ± 0.0013` (`0.0175`) | `0.0273 ± 0.0005` (`0.0213`) | `● Informative` |
| `A_attn_every_other` | `linear` | `0.0436 ± 0.0011` (`0.0384`) | `0.0420 ± 0.0003` (`0.0346`) | `0.0221 ± 0.0016` (`0.0171`) | `0.0227 ± 0.0015` (`0.0186`) | `● Informative` |
| `A_attn_first_last` | `linear` | `0.0441 ± 0.0012` (`0.0402`) | `0.0422 ± 0.0006` (`0.0351`) | `0.0224 ± 0.0007` (`0.0182`) | `0.0228 ± 0.0025` (`0.0215`) | `● Informative` |
| `A_attn_all` (`= reperm = no_es`) | `linear` | `0.0446 ± 0.0008` (`0.0399`) | `0.0422 ± 0.0010` (`0.0351`) | `0.0222 ± 0.0001` (`0.0179`) | `0.0226 ± 0.0015` (`0.0204`) | `● Informative` |
| `A_attn_first` | `linear` | `0.0451 ± 0.0009` (`0.0412`) | `0.0424 ± 0.0013` (`0.0355`) | `0.0224 ± 0.0012` (`0.0179`) | `0.0278 ± 0.0015` (`0.0223`) | `● Informative` |
| `B_loss_every_prefix` | `linear` | `0.0441 ± 0.0038` (`0.0415`) | `0.0422 ± 0.0011*` (`0.0355*`) | `0.0235 ± 0.0011` (`0.0182`) | `0.0309 ± 0.0023` (`0.0276`) | `● Informative` |
| `A_attn_none` | `linear` | `0.2282 ± 0.0056` (`0.2131`) | `0.2282 ± 0.0056` (`0.2194`) | `0.3401 ± 0.0017` (`0.3352`) | `0.1886 ± 0.0032` (`0.1806`) | `● No gain` (Ablation Floor) |

---

## Artifacts in This Directory

- [`summary.json`](./summary.json): Full 3-seed aggregated metrics, prefix KL curves, training logs, and per-seed bootstrap CIs (`commit e5039a60da52fbffc5208c549cd866cacceba307`).
- [`tt_abc_e2/result.json`](./tt_abc_e2/result.json): Schema-validated `aggbench.result/v1` artifact for `E2`.
- [`tt_abc_z8/result.json`](./tt_abc_z8/result.json): Schema-validated `aggbench.result/v1` artifact for `Z8`.
- [`tt_abc_z1-blind/result.json`](./tt_abc_z1-blind/result.json): Schema-validated `aggbench.result/v1` artifact for `Z1-blind`.
- [`tt_abc_z5/result.json`](./tt_abc_z5/result.json): Schema-validated `aggbench.result/v1` artifact for `Z5`.
