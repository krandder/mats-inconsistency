# EXPLORATORY Baseline: Kelvin's Trader-Transformer Axes (A), (B), (C) & Intra-Trader DeepSets on Adversarial Environments `E3-collude`, `E3-sybil`, and `E3-biased` (`tt-abc-e3-20261021`)

Serves: Exploratory 20k-pool baseline for Kelvin's 1-token-per-trader causal Transformer architecture (`aggbench/trader_transformer.py`) across Axes `(A)` cross-trader attention schedule, `(B)` supervision target (`final`, `every_prefix`, and oracle `distill`), `(C)` trader-direction residual stream (`kv`, `recurrent`, `feedback`), and `trader_encoder="deepsets"` on `alignment="unaligned"` `E3-collude` ($N=5$ with 3 colluders), `E3-sybil` ($N=8$ with 3 Sybils), and `E3-biased` ($N=5$ with 1 biased report) (`3` seeds `0, 1, 2` on NVIDIA L4 GPU).
Builds on: `aggbench/trader_transformer.py` (`commit eb3cd7d`) and `gate-gemini-tt-2026-10-21.md` (`#3859`, Must-Fixes 1, 2, 3, 5).

**Status:** `● Informative (EXPLORATORY baseline)` (`test` splits untouched; `alignment="unaligned"`; fixed `n_train=20000` pool shared across seeds `0, 1, 2` with 10% held-out train-split early stopping + `A_attn_all_no_es` un-early-stopped control; `n_eval=2000` dev episodes; GPU: `NVIDIA L4`; runner commit `e5039a60da52fbffc5208c549cd866cacceba307`).

> [!IMPORTANT]
> **Gate Audit Disclosures (`gate-gemini-tt-2026-10-21.md`, `#3859`):**
> 1. **No environment is below `0.01` nats regret to Bayes (`0.0000`) yet:** Best non-oracle learned arms on this 20k pool reach `0.0275 ± 0.0011` (`A_attn_last_deepsets`) on `E3-collude` and `0.0421 ± 0.0077` (`B_loss_every_prefix_deepsets_reperm`) on `E3-sybil`.
> 2. **Encoder Confound vs. `full_context_transformer_none` (`Must-Fix 2`):** In this 20k-pool run (`commit e5039a6`), `full_context_transformer_none` (`0.0514` on `E3-collude`, `0.0728` on `E3-sybil`, `0.0729` on `E3-biased`) used flat linear 49-D tokens (`full_context`, including `report_logit / 4`), whereas `*_deepsets*` arms used a 2-layer `DeepSets` piece encoder with no scalar report. When both use the flat linear token (`A_attn_all_reperm`: `0.0500` on `E3-collude`, `0.0694` on `E3-sybil`, `0.0678` on `E3-biased`), the 1-token-per-trader causal transformer is tied with `full_context_transformer_none`. Unconfounded matched-encoder bars (`bar_fct_deepsets_bidir`, `bar_fct_piece_bidir`, `bar_fct_no_report`, and `bar_fct_fresh_headline_w64_l3`) are evaluated in the companion `tt-gate-e2z8syb` run.
> 3. **`E3-biased` vs. `E2` Equivalence on `deepsets` (`Must-Fix 3`):** In `E3-biased`, the biased agent's raw piece observations `obs` and `own_model` are honest and byte-identical to `E2` under the same episode seeds; only `reports["logit"]` is shifted. Because `trader_encoder="deepsets"` does not read `reports["logit"]`, all `*_deepsets*` rows on `E3-biased` are numerically identical to `E2` and are not an adversarial test of report bias. By contrast, `trader_encoder="linear"` (`full_context`) appends `report_logit / 4`, so flat-linear arms (`A_attn_all_reperm` `0.0678` vs `0.0557` on `E2`; `full_context_transformer_none` `0.0729` vs `0.0615` on `E2`) reflect the effect of the biased report feature.
> 4. **Oracle Ceiling & Fixed 20k Pool (`Must-Fix 1, 5`):** `B_loss_distill` and `B_loss_distill_deepsets` use `prefix_bayes_logits` (`uses_oracle=True`) and are graded as **Oracle Ceilings**. `env.sample("train", 20000)` was sampled once per environment and shared across seeds `0, 1, 2`.

---

## Mandatory `(a)–(d)` Gate Disclosure (`#3843` / `#3847`)

- **`(a)` What a token is:**
  - `trader_encoder="linear"`: one flat 49-D vector per trader $r \in \{0, \dots, N-1\}$ (`full_context`, including `report_logit / 4`).
  - `trader_encoder="deepsets"`: shared 2-layer DeepSets encoder over trader $r$'s $K=12$ piece cells `[present, obs_k, own_cov_k, own_mu_k]` (no scalar report), summed over present pieces into 1 token per trader ($d_{\text{model}}=128$).
- **`(b)` What a layer is:**
  - Pre-norm Transformer block (`TraderNet`, $L=4$ layers, $H=4$ heads, $d_{\text{model}}=128$).
- **`(c)` What crosses between traders and at what width:**
  - Width $d_{\text{model}}=128$ vectors cross causally ($r' \le r$) via `kv`, `recurrent` (GRU), or `feedback` memory.
- **`(d)` Regret to Bayes (`0.0000`) and Comparison Bars (`alignment="unaligned"`):**
  - Evaluated on `n_eval = 2000` unaligned `dev` episodes against `bayes` (`0.0000`), `oracle_clearing` (`0.0175` on `E3-collude`, `0.0166` on `E3-sybil`/`E3-biased`), `extremized_log_pool` (`0.0302` on `E3-collude`, `0.0778` on `E3-sybil`, `0.0764` on `E3-biased`), and flat-linear `full_context_transformer_none` (`0.0514` on `E3-collude`, `0.0728` on `E3-sybil`, `0.0729` on `E3-biased`), with `untrained_init` and `random_network` controls in [`tt_abc_e3-collude/result.json`](./tt_abc_e3-collude/result.json), [`tt_abc_e3-sybil/result.json`](./tt_abc_e3-sybil/result.json), and [`tt_abc_e3-biased/result.json`](./tt_abc_e3-biased/result.json).

---

## Master Table: `E3-collude`, `E3-sybil`, and `E3-biased` (`dev` $n=2000$, 3 Seeds `0, 1, 2`, Fixed 20k Pool)

| Arm / Method | Encoder | Loss | `E3-collude` Regret (`KL`) | `E3-sybil` Regret (`KL`) | `E3-biased` Regret (`KL`) | Status / Note |
|---|---|---|---:|---:|---:|---|
| `bayes` | — | — | `0.0000 ± 0.0000` (`0.0000`) | `0.0000 ± 0.0000` (`0.0000`) | `0.0000 ± 0.0000` (`0.0000`) | Exact Bayes Ceiling (`uses_oracle=True`) |
| `oracle_clearing` | — | — | `0.0175 ± 0.0000` (`0.0144`) | `0.0166 ± 0.0000` (`0.0158`) | `0.0166 ± 0.0000` (`0.0158`) | Scalar Ceiling (`uses_oracle=True`) |
| `B_loss_distill_deepsets` | `deepsets` | `distill` | `0.0192 ± 0.0023` (`0.0159`) | `0.0293 ± 0.0005` (`0.0263`) | `0.0250 ± 0.0032*` (`0.0244*`) | Oracle Ceiling (`uses_oracle=True`) |
| `A_attn_last_deepsets` | `deepsets` | `final` | `0.0275 ± 0.0011` (`0.0255`) | `0.0455 ± 0.0026` (`0.0385`) | `0.0430 ± 0.0062*` (`0.0356*`) | `● Informative` (`*`identical to `E2`) |
| `B_loss_every_prefix_deepsets_reperm` | `deepsets` | `every_prefix` | `0.0291 ± 0.0033` (`0.0252`) | `0.0421 ± 0.0077` (`0.0356`) | `0.0439 ± 0.0058*` (`0.0358*`) | `● Informative` (`*`identical to `E2`) |
| `V_vis_full_deepsets_prefix` | `deepsets` | `every_prefix` | `0.0294 ± 0.0026` (`0.0250`) | `0.0446 ± 0.0046` (`0.0387`) | `0.0456 ± 0.0028*` (`0.0379*`) | `● Informative` (`*`identical to `E2`) |
| `V_vis_full_deepsets` | `deepsets` | `final` | `0.0297 ± 0.0030` (`0.0263`) | `0.0468 ± 0.0065` (`0.0419`) | `0.0435 ± 0.0060*` (`0.0365*`) | `● Informative` (`*`identical to `E2`) |
| `A_attn_all_deepsets_reperm` | `deepsets` | `final` | `0.0298 ± 0.0056` (`0.0259`) | `0.0434 ± 0.0042` (`0.0373`) | `0.0365 ± 0.0053*` (`0.0322*`) | `● Informative` (`*`identical to `E2`) |
| `extremized_log_pool` | — | — | `0.0302 ± 0.0000` (`0.0286`) | `0.0778 ± 0.0000` (`0.0747`) | `0.0764 ± 0.0000` (`0.0718`) | Scalar Baseline |
| `B_loss_every_prefix_deepsets` | `deepsets` | `every_prefix` | `0.0305 ± 0.0026` (`0.0250`) | `0.0491 ± 0.0006` (`0.0426`) | `0.0488 ± 0.0026*` (`0.0397*`) | `● Informative` (`*`identical to `E2`) |
| `A_attn_all_deepsets` | `deepsets` | `final` | `0.0309 ± 0.0023` (`0.0269`) | `0.0450 ± 0.0041` (`0.0390`) | `0.0447 ± 0.0009*` (`0.0367*`) | `● Informative` (`*`identical to `E2`) |
| `B_loss_distill` | `linear` | `distill` | `0.0328 ± 0.0008` (`0.0306`) | `0.0460 ± 0.0013` (`0.0436`) | `0.0525 ± 0.0029` (`0.0459`) | Oracle Ceiling (`uses_oracle=True`) |
| `C_state_recurrent_deepsets` | `deepsets` | `final` | `0.0330 ± 0.0034` (`0.0310`) | `0.0542 ± 0.0087` (`0.0476`) | `0.0508 ± 0.0052*` (`0.0434*`) | `● Informative` (`O(1)` Recurrent State) |
| `B_loss_every_prefix_pos` | `linear` | `every_prefix` | `0.0427 ± 0.0059` (`0.0393`) | `0.0652 ± 0.0016` (`0.0662`) | `0.0721 ± 0.0043` (`0.0662`) | `● Informative` |
| `B_loss_every_prefix_reperm` | `linear` | `every_prefix` | `0.0473 ± 0.0039` (`0.0456`) | `0.0636 ± 0.0008` (`0.0586`) | `0.0667 ± 0.0082` (`0.0655`) | `● Informative` (Matched linear encoder) |
| `C_state_recurrent_prefix` | `linear` | `every_prefix` | `0.0485 ± 0.0043` (`0.0474`) | `0.0816 ± 0.0076` (`0.0832`) | `0.0774 ± 0.0040` (`0.0725`) | `● Informative` |
| `A_attn_all_reperm` | `linear` | `final` | `0.0500 ± 0.0079` (`0.0488`) | `0.0694 ± 0.0065` (`0.0649`) | `0.0678 ± 0.0065` (`0.0680`) | `● Informative` (Matched linear encoder) |
| **`full_context_transformer_none`** | `linear` | `final` | **`0.0514 ± 0.0043` (`0.0486`)** | **`0.0728 ± 0.0062` (`0.0700`)** | **`0.0729 ± 0.0049` (`0.0688`)** | **Flat-Linear Reference Bar (20k pool)** |
| `full_context_transformer_full` | `linear` | `final` | `0.0500 ± 0.0005` (`0.0483`) | `0.0729 ± 0.0076` (`0.0686`) | `0.0756 ± 0.0029` (`0.0699`) | Flat-Linear Reference Bar (`full`) |
| `C_state_feedback_prefix` | `linear` | `every_prefix` | `0.0566 ± 0.0073` (`0.0539`) | `0.0831 ± 0.0087` (`0.0817`) | `0.0792 ± 0.0077` (`0.0752`) | `● Informative` |
| `B_loss_every_prefix` | `linear` | `every_prefix` | `0.0579 ± 0.0067` (`0.0529`) | `0.0899 ± 0.0073` (`0.0854`) | `0.0805 ± 0.0054` (`0.0758`) | `● Informative` |
| `C_state_recurrent_final` | `linear` | `final` | `0.0596 ± 0.0069` (`0.0572`) | `0.1015 ± 0.0080` (`0.1015`) | `0.0886 ± 0.0083` (`0.0858`) | `● Informative` |
| `A_attn_first_last` | `linear` | `final` | `0.0599 ± 0.0034` (`0.0595`) | `0.0931 ± 0.0150` (`0.0906`) | `0.0894 ± 0.0107` (`0.0876`) | `● Informative` |
| `V_vis_none_no_own` | `linear` | `final` | `0.0601 ± 0.0132` (`0.0630`) | `0.0857 ± 0.0106` (`0.0848`) | `0.0849 ± 0.0025` (`0.0870`) | `● Informative` |
| `V_vis_full` | `linear` | `final` | `0.0610 ± 0.0073` (`0.0627`) | `0.0991 ± 0.0134` (`0.0966`) | `0.0954 ± 0.0081` (`0.0921`) | `● Informative` |
| `A_attn_first` | `linear` | `final` | `0.0619 ± 0.0032` (`0.0590`) | `0.0963 ± 0.0045` (`0.0974`) | `0.0917 ± 0.0114` (`0.0912`) | `● Informative` |
| `A_attn_every_other` | `linear` | `final` | `0.0624 ± 0.0048` (`0.0601`) | `0.0920 ± 0.0143` (`0.0894`) | `0.0875 ± 0.0074` (`0.0856`) | `● Informative` |
| `A_attn_all` | `linear` | `final` | `0.0625 ± 0.0087` (`0.0614`) | `0.0968 ± 0.0094` (`0.0953`) | `0.0936 ± 0.0069` (`0.0926`) | `● Informative` |
| `A_attn_last` | `linear` | `final` | `0.0634 ± 0.0084` (`0.0622`) | `0.0797 ± 0.0014` (`0.0756`) | `0.0868 ± 0.0062` (`0.0845`) | `● Informative` |
| `V_vis_noisy` | `linear` | `final` | `0.0641 ± 0.0097` (`0.0634`) | `0.1032 ± 0.0101` (`0.1024`) | `0.0939 ± 0.0062` (`0.0921`) | `● Informative` |
| `C_state_feedback_final` | `linear` | `final` | `0.0650 ± 0.0073` (`0.0615`) | `0.0923 ± 0.0111` (`0.0904`) | `0.0877 ± 0.0051` (`0.0855`) | `● Informative` |
| `V_vis_noisy_no_own` | `linear` | `final` | `0.0674 ± 0.0118` (`0.0622`) | `0.1032 ± 0.0077` (`0.1034`) | `0.0959 ± 0.0140` (`0.0937`) | `● Informative` |
| `log_pool` | — | — | `0.0463 ± 0.0000` (`0.0410`) | `0.1261 ± 0.0000` (`0.1176`) | `0.0913 ± 0.0000` (`0.0839`) | Baseline |
| `linear_pool` | — | — | `0.0744 ± 0.0000` (`0.0669`) | `0.1587 ± 0.0000` (`0.1484`) | `0.1425 ± 0.0000` (`0.1319`) | Baseline |
| `A_attn_none` | `linear` | `final` | `0.1482 ± 0.0042` (`0.1372`) | `0.2268 ± 0.0065` (`0.2187`) | `0.2547 ± 0.0029` (`0.2422`) | `● No gain` (Ablation Floor) |
| `sum_logodds` | — | — | `0.2589 ± 0.0000` (`0.2912`) | `0.4170 ± 0.0000` (`0.4509`) | `0.3410 ± 0.0000` (`0.3626`) | Baseline |
| `A_attn_all_no_es` | `linear` | `final` | `0.9515 ± 0.0168` (`0.9386`) | `1.0066 ± 0.0731` (`1.0183`) | `0.9857 ± 0.0411` (`0.9923`) | `● Informative` (Memorizes 20k Pool) |

---

## Artifacts in This Directory

- [`summary.json`](./summary.json): Full 3-seed aggregated metrics, prefix KL curves, training logs, and per-seed bootstrap CIs (`commit e5039a60da52fbffc5208c549cd866cacceba307`).
- [`tt_abc_e3-collude/result.json`](./tt_abc_e3-collude/result.json): Schema-validated `aggbench.result/v1` artifact for `E3-collude`.
- [`tt_abc_e3-sybil/result.json`](./tt_abc_e3-sybil/result.json): Schema-validated `aggbench.result/v1` artifact for `E3-sybil`.
- [`tt_abc_e3-biased/result.json`](./tt_abc_e3-biased/result.json): Schema-validated `aggbench.result/v1` artifact for `E3-biased`.
