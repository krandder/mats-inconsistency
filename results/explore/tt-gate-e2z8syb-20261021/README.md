# Trader-Transformer Gate Fixes Sweep: `E2`, `Z8`, and `E3-sybil` (3 Seeds)

- **Status:** `EXPLORATORY` baseline (no confirmatory gates; `B_loss_distill*` arms use privileged exact prefix-Bayes targets and are graded `Oracle Ceiling`).
- **Commit:** `204e78a5f6222a0fdb81dd48f5a4e2e32dbb4e1b` (`204e78a`, `git_dirty: false`)
- **Generator script:** `python3 scripts/render_tt_gate_readmes.py` (all tables and inline numbers rendered directly from `summary.json`).
- **Runner command:** `PYTHONPATH=. python3 benchmarks/explore_trader_transformer_gate_fixes.py --envs E2,Z8,E3-sybil --seeds 0,1,2 --steps 2000 --batch-size 512 --n-train 20000 --n-eval 2000 --width 128 --grid-width 64 --layers 4 --commit 204e78a5f6222a0fdb81dd48f5a4e2e32dbb4e1b --out results/explore/tt-gate-e2z8syb-20261021/summary.json`
- **Per-seed train pools:** Seed `s` samples `n_train = 20000` training episodes starting at offset `start = s * n_train` (`0`, `20000`, `40000`), disjoint from the `n_eval = 2000` evaluation episodes.
- **Prefix Bayes verification (`2546bb2`):** Runtime assertion `max |prefix_bayes[:, -1] - bayes_logits| < 1e-8` passed on all three environments (`E2`: `3.55e-15`, `Z8`: `3.55e-15`, `E3-sybil`: `3.55e-15`).

## 1. Non-Neural Baselines (`n_eval = 2000`, rendered from `summary.json`)

| Baseline | `E2` Regret | `E2` KL | `Z8` Regret | `Z8` KL | `E3-sybil` Regret | `E3-sybil` KL |
|---|---:|---:|---:|---:|---:|---:|
| `bayes` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `oracle_clearing` | `0.0166` | `0.0158` | `0.0236` | `0.0168` | `0.0166` | `0.0158` |
| `extremized_log_pool` | `0.0448` | `0.0407` | `0.0344` | `0.0302` | `0.0777` | `0.0747` |
| `log_pool` | `0.1124` | `0.1026` | `0.1178` | `0.1118` | `0.1261` | `0.1176` |
| `sum_logodds` | `0.1324` | `0.1419` | `0.0952` | `0.0939` | `0.4170` | `0.4509` |
| `linear_pool` | `0.1517` | `0.1407` | `0.1636` | `0.1568` | `0.1587` | `0.1484` |

**Comparison to `extremized_log_pool` on `Z8`:** On `Z8`, `extremized_log_pool` achieves regret `0.0344` (`kl = 0.0302`), which beats every causal and 2D grid neural arm in this sweep (the best non-oracle causal/grid arm on `Z8` is `D_layout_grid_rankfree_cellT1_final` at `0.0347 ± 0.0026`). Only the bidirectional piece bar `bar_fct_piece_bidir` (`0.0331 ± 0.0008`) and the `[ORACLE CEILING]` arm `B_loss_distill_deepsets` (`0.0318 ± 0.0019`) achieve lower mean regret on `Z8`.

## 2. Matched Bidirectional Bars vs. Causal and Grid Arms

1. **Input dimension (`in_dim=49`) of linear trader arms vs. bidirectional bars:**
   - The linear trader arms (`A_attn_*`, `B_loss_*`, `C_state_*`) take `in_dim = 49` (`24` values + `24` mask bits + `1` own `report_logit / 4`), which is the exact same per-trader input as `full_context_transformer_none` (`in_dim = 49`). Only `bar_fct_no_report` drops `report_logit / 4` (`in_dim = 48`).
   - With matched `in_dim = 49` inputs on the `20k` pool (`steps = 2000, width = 128, layers = 4`), bidirectional `full_context_transformer_none` beats single-order causal `A_attn_all` on all 3 environments: `E2` `0.0609` vs `0.0843`, `Z8` `0.0665` vs `0.0809`, and `E3-sybil` `0.0742` vs `0.0948`.
   - Removing `report_logit / 4` from the bidirectional bar (`bar_fct_no_report`, `in_dim = 48`) raises its regret to `0.0847` (`E2`), `0.0915` (`Z8`), and `0.0963` (`E3-sybil`). Re-permuting trader order each step (`A_attn_all_reperm`: `0.0656`, `0.0641`, `0.0730`) and `B_loss_every_prefix_reperm` (`0.0583`, `0.0599`, `0.0643`) narrow the gap to `full_context_transformer_none`.
2. **Matched encoder comparison (`deepsets` and `24x2` piece tokens):**
   - With `deepsets` tokens (`in_dim = 65`), bidirectional `bar_fct_deepsets_bidir` (`E2`: `0.0346`, `Z8`: `0.0359`, `E3-sybil`: `0.0403`) beats causal `A_attn_all_deepsets` (`0.0401`, `0.0390`, `0.0470`) and is slightly below `A_attn_all_deepsets_reperm` (`0.0366`, `0.0382`, `0.0426`).
   - With `24x2` piece tokens (`in_dim = 4`), bidirectional `bar_fct_piece_bidir` (`E2`: `0.0309 ± 0.0025`, `Z8`: `0.0331 ± 0.0008`, `E3-sybil`: `0.0354 ± 0.0078`) is lower than `D_layout_grid_rankfree_cellT1_final` (`E2`: `0.0320 ± 0.0006`, `Z8`: `0.0347 ± 0.0026`, `E3-sybil`: `0.0392 ± 0.0022`). Across all 6 matched representation pairs (`{flat 49-d, deepsets, piece}` $\times$ `{E2, Z8}`), the bidirectional bar has lower mean regret than the matched causal arm.
3. **`2x2` Grid Readout Factorial (`grid_full` vs `grid_rankfree` $\times$ `cell_T1` vs `pooled`):**
   - Across `{E2, Z8, E3-sybil} x {cell_T1, pooled} x {final, every_prefix}` (12 cells), `grid_rankfree` has lower mean regret than `grid_full` in 12 of 12 cells. Note that `grid_rankfree` both removes causal masking along the slot axis and omits slot embeddings, so those two effects are combined in this comparison.

## 3. Full 3-Seed Results Table (`E2`, `Z8`, `E3-sybil`, rendered from `summary.json`)

| Arm | Input (`in_dim`) | Role | `E2` Regret | `E2` Prefix KL | `Z8` Regret | `Z8` Prefix KL | `E3-sybil` Regret | `E3-sybil` Prefix KL |
|---|---|---|---|---:|---:|---:|---:|---:|
| `full_context_transformer_none` | Flat + own `report/4` (`in_dim=49`) | Bidirectional FCT bar | `0.0609 ± 0.0047` | — | `0.0665 ± 0.0034` | — | `0.0742 ± 0.0097` | — |
| `full_context_transformer_full` | Flat + all reports (`in_dim=73`) | Bidirectional FCT bar | `0.0670 ± 0.0036` | — | `0.0672 ± 0.0025` | — | `0.0761 ± 0.0111` | — |
| `bar_fct_no_report` | Flat, no `report/4` (`in_dim=48`) | Bidirectional FCT bar | `0.0847 ± 0.0034` | — | `0.0915 ± 0.0099` | — | `0.0963 ± 0.0121` | — |
| `bar_fct_fresh_headline_w64_l3` | Flat + own `report/4` (`in_dim=49`) | Bidirectional FCT bar | `0.0364 ± 0.0016` | — | `0.0362 ± 0.0005` | — | `0.0428 ± 0.0008` | — |
| `bar_fct_deepsets_bidir` | `deepsets` (`in_dim=4`) | Bidirectional FCT bar | `0.0346 ± 0.0036` | — | `0.0359 ± 0.0009` | — | `0.0403 ± 0.0043` | — |
| `bar_fct_piece_bidir` | `24x2` piece tokens (`in_dim=4`) | Bidirectional FCT bar | `0.0309 ± 0.0025` | — | `0.0331 ± 0.0008` | — | `0.0354 ± 0.0078` | — |
| `A_attn_all` | Flat + own `report/4` (`in_dim=49`) | Axis (A) causal attention schedule | `0.0843 ± 0.0055` | `0.0829 ± 0.0116` | `0.0809 ± 0.0151` | `0.0814 ± 0.0086` | `0.0948 ± 0.0078` | `0.0813 ± 0.0071` |
| `A_attn_all_reperm` | Flat + own `report/4` (`in_dim=49`) | Axis (A) causal attention schedule | `0.0656 ± 0.0039` | `0.0707 ± 0.0073` | `0.0641 ± 0.0065` | `0.0726 ± 0.0085` | `0.0730 ± 0.0032` | `0.0671 ± 0.0104` |
| `A_attn_all_no_es` | Flat + own `report/4` (`in_dim=49`) | Axis (A) causal attention schedule | `0.9840 ± 0.0162` | `0.8996 ± 0.0040` | `0.8213 ± 0.0426` | `0.8368 ± 0.0174` | `0.9795 ± 0.0303` | `0.9473 ± 0.0262` |
| `A_attn_first_last` | Flat + own `report/4` (`in_dim=49`) | Axis (A) causal attention schedule | `0.0842 ± 0.0056` | `0.0856 ± 0.0108` | `0.0833 ± 0.0114` | `0.0901 ± 0.0082` | `0.0898 ± 0.0064` | `0.0816 ± 0.0043` |
| `A_attn_every_other` | Flat + own `report/4` (`in_dim=49`) | Axis (A) causal attention schedule | `0.0816 ± 0.0058` | `0.0849 ± 0.0103` | `0.0896 ± 0.0228` | `0.0986 ± 0.0174` | `0.0901 ± 0.0051` | `0.0827 ± 0.0041` |
| `A_attn_last` | Flat + own `report/4` (`in_dim=49`) | Axis (A) causal attention schedule | `0.0772 ± 0.0036` | `0.0880 ± 0.0085` | `0.0786 ± 0.0066` | `0.0975 ± 0.0093` | `0.0825 ± 0.0053` | `0.0758 ± 0.0041` |
| `A_attn_first` | Flat + own `report/4` (`in_dim=49`) | Axis (A) causal attention schedule | `0.0837 ± 0.0017` | `0.0827 ± 0.0063` | `0.0734 ± 0.0079` | `0.0885 ± 0.0060` | `0.0995 ± 0.0058` | `0.0834 ± 0.0035` |
| `A_attn_none` | Flat + own `report/4` (`in_dim=49`) | Axis (A) causal attention schedule | `0.2473 ± 0.0033` | `0.1497 ± 0.0015` | `0.2488 ± 0.0064` | `0.1606 ± 0.0032` | `0.2273 ± 0.0073` | `0.1561 ± 0.0016` |
| `B_loss_every_prefix` | Flat + own `report/4` (`in_dim=49`) | Axis (B) loss schedule | `0.0709 ± 0.0038` | `0.0513 ± 0.0044` | `0.0698 ± 0.0099` | `0.0495 ± 0.0078` | `0.0886 ± 0.0056` | `0.0627 ± 0.0010` |
| `B_loss_every_prefix_reperm` | Flat + own `report/4` (`in_dim=49`) | Axis (B) loss schedule | `0.0583 ± 0.0030` | `0.0397 ± 0.0019` | `0.0599 ± 0.0081` | `0.0407 ± 0.0023` | `0.0643 ± 0.0025` | `0.0446 ± 0.0002` |
| `B_loss_distill` | Flat + own `report/4` (`in_dim=49`) | `[ORACLE CEILING]` (privileged prefix Bayes target) | `0.0448 ± 0.0035` | `0.0277 ± 0.0010` | `0.0442 ± 0.0009` | `0.0242 ± 0.0002` | `0.0489 ± 0.0015` | `0.0311 ± 0.0006` |
| `B_loss_every_prefix_pos` | Flat + own `report/4` (`in_dim=49`) | Axis (B) loss schedule | `0.0613 ± 0.0038` | `0.0383 ± 0.0033` | `0.0533 ± 0.0031` | `0.0312 ± 0.0017` | `0.0705 ± 0.0061` | `0.0465 ± 0.0011` |
| `C_state_recurrent_final` | Flat + own `report/4` (`in_dim=49`) | Axis (C) token state | `0.0806 ± 0.0052` | `0.0635 ± 0.0043` | `0.0749 ± 0.0158` | `0.0638 ± 0.0093` | `0.0979 ± 0.0051` | `0.0945 ± 0.0027` |
| `C_state_recurrent_prefix` | Flat + own `report/4` (`in_dim=49`) | Axis (C) token state | `0.0631 ± 0.0062` | `0.0413 ± 0.0020` | `0.0625 ± 0.0057` | `0.0397 ± 0.0014` | `0.0839 ± 0.0083` | `0.0574 ± 0.0024` |
| `C_state_feedback_final` | Flat + own `report/4` (`in_dim=49`) | Axis (C) token state | `0.0826 ± 0.0016` | `0.0953 ± 0.0055` | `0.0802 ± 0.0051` | `0.0944 ± 0.0060` | `0.0870 ± 0.0051` | `0.0871 ± 0.0049` |
| `C_state_feedback_prefix` | Flat + own `report/4` (`in_dim=49`) | Axis (C) token state | `0.0704 ± 0.0030` | `0.0499 ± 0.0053` | `0.0689 ± 0.0114` | `0.0458 ± 0.0106` | `0.0810 ± 0.0058` | `0.0600 ± 0.0033` |
| `V_vis_full` | Flat + own `report/4` (`in_dim=73`) | TraderTransformer | `0.0893 ± 0.0080` | `0.0875 ± 0.0119` | `0.0777 ± 0.0050` | `0.0846 ± 0.0034` | `0.0994 ± 0.0060` | `0.0813 ± 0.0039` |
| `A_attn_all_deepsets` | `deepsets` (`in_dim=4`) | Axis (A) causal attention schedule | `0.0401 ± 0.0040` | `0.0496 ± 0.0020` | `0.0390 ± 0.0036` | `0.0520 ± 0.0067` | `0.0470 ± 0.0015` | `0.0360 ± 0.0031` |
| `A_attn_all_deepsets_reperm` | `deepsets` (`in_dim=4`) | Axis (A) causal attention schedule | `0.0366 ± 0.0018` | `0.0365 ± 0.0036` | `0.0382 ± 0.0019` | `0.0520 ± 0.0069` | `0.0426 ± 0.0009` | `0.0344 ± 0.0042` |
| `A_attn_last_deepsets` | `deepsets` (`in_dim=4`) | Axis (A) causal attention schedule | `0.0373 ± 0.0037` | `0.0404 ± 0.0034` | `0.0408 ± 0.0021` | `0.0522 ± 0.0067` | `0.0435 ± 0.0026` | `0.0399 ± 0.0026` |
| `B_loss_every_prefix_deepsets` | `deepsets` (`in_dim=4`) | Axis (B) loss schedule | `0.0466 ± 0.0009` | `0.0251 ± 0.0011` | `0.0433 ± 0.0030` | `0.0259 ± 0.0006` | `0.0463 ± 0.0021` | `0.0266 ± 0.0005` |
| `B_loss_every_prefix_deepsets_reperm` | `deepsets` (`in_dim=4`) | Axis (B) loss schedule | `0.0385 ± 0.0027` | `0.0213 ± 0.0020` | `0.0388 ± 0.0012` | `0.0207 ± 0.0005` | `0.0421 ± 0.0007` | `0.0239 ± 0.0001` |
| `B_loss_distill_deepsets` | `deepsets` (`in_dim=4`) | `[ORACLE CEILING]` (privileged prefix Bayes target) | `0.0256 ± 0.0007` | `0.0151 ± 0.0007` | `0.0318 ± 0.0019` | `0.0173 ± 0.0001` | `0.0284 ± 0.0017` | `0.0173 ± 0.0004` |
| `C_state_recurrent_deepsets` | `deepsets` (`in_dim=4`) | Axis (C) token state | `0.0481 ± 0.0060` | `0.0432 ± 0.0105` | `0.0395 ± 0.0020` | `0.0378 ± 0.0024` | `0.0546 ± 0.0042` | `0.0538 ± 0.0059` |
| `V_vis_full_deepsets` | `deepsets` (`in_dim=6`) | TraderTransformer | `0.0417 ± 0.0034` | `0.0407 ± 0.0026` | `0.0406 ± 0.0042` | `0.0519 ± 0.0111` | `0.0438 ± 0.0039` | `0.0372 ± 0.0020` |
| `V_vis_full_deepsets_prefix` | `deepsets` (`in_dim=6`) | TraderTransformer | `0.0470 ± 0.0020` | `0.0247 ± 0.0021` | `0.0410 ± 0.0045` | `0.0259 ± 0.0001` | `0.0467 ± 0.0023` | `0.0276 ± 0.0011` |
| `D_layout_grid_full_cellT1_final` | `24x2` piece tokens (`in_dim=4`) | 2D grid TraderTransformer | `0.0365 ± 0.0044` | `0.0474 ± 0.0061` | `0.0376 ± 0.0020` | `0.0576 ± 0.0026` | `0.0450 ± 0.0008` | `0.0431 ± 0.0012` |
| `D_layout_grid_full_pooled_final` | `24x2` piece tokens (`in_dim=4`) | 2D grid TraderTransformer | `0.0387 ± 0.0022` | `0.0434 ± 0.0066` | `0.0425 ± 0.0030` | `0.0538 ± 0.0049` | `0.0479 ± 0.0058` | `0.0496 ± 0.0047` |
| `D_layout_grid_rankfree_cellT1_final` | `24x2` piece tokens (`in_dim=4`) | 2D grid TraderTransformer | `0.0320 ± 0.0006` | `0.0422 ± 0.0032` | `0.0347 ± 0.0026` | `0.0577 ± 0.0005` | `0.0392 ± 0.0022` | `0.0451 ± 0.0023` |
| `D_layout_grid_rankfree_pooled_final` | `24x2` piece tokens (`in_dim=4`) | 2D grid TraderTransformer | `0.0353 ± 0.0006` | `0.0439 ± 0.0043` | `0.0355 ± 0.0033` | `0.0592 ± 0.0024` | `0.0414 ± 0.0045` | `0.0459 ± 0.0013` |
| `D_layout_grid_full_cellT1_prefix` | `24x2` piece tokens (`in_dim=4`) | 2D grid TraderTransformer | `0.0395 ± 0.0037` | `0.0256 ± 0.0023` | `0.0448 ± 0.0073` | `0.0275 ± 0.0023` | `0.0461 ± 0.0020` | `0.0312 ± 0.0012` |
| `D_layout_grid_full_pooled_prefix` | `24x2` piece tokens (`in_dim=4`) | 2D grid TraderTransformer | `0.0429 ± 0.0009` | `0.0270 ± 0.0008` | `0.0462 ± 0.0005` | `0.0269 ± 0.0006` | `0.0466 ± 0.0015` | `0.0322 ± 0.0021` |
| `D_layout_grid_rankfree_cellT1_prefix` | `24x2` piece tokens (`in_dim=4`) | 2D grid TraderTransformer | `0.0357 ± 0.0021` | `0.0237 ± 0.0015` | `0.0394 ± 0.0010` | `0.0250 ± 0.0011` | `0.0433 ± 0.0037` | `0.0283 ± 0.0016` |
| `D_layout_grid_rankfree_pooled_prefix` | `24x2` piece tokens (`in_dim=4`) | 2D grid TraderTransformer | `0.0380 ± 0.0012` | `0.0243 ± 0.0012` | `0.0388 ± 0.0005` | `0.0252 ± 0.0007` | `0.0422 ± 0.0039` | `0.0283 ± 0.0020` |
| `D_layout_grid_cellT1_final` | `24x2` piece tokens (`in_dim=4`) | 2D grid TraderTransformer | `0.0404 ± 0.0013` | `0.0333 ± 0.0033` | `0.0429 ± 0.0022` | `0.0517 ± 0.0057` | `0.0436 ± 0.0036` | `0.0424 ± 0.0086` |
| `D_layout_grid_pooled_final` | `24x2` piece tokens (`in_dim=4`) | 2D grid TraderTransformer | `0.0427 ± 0.0008` | `0.0418 ± 0.0025` | `0.0427 ± 0.0036` | `0.0492 ± 0.0023` | `0.0424 ± 0.0006` | `0.0471 ± 0.0055` |
| `D_layout_grid_cellT1_prefix` | `24x2` piece tokens (`in_dim=4`) | 2D grid TraderTransformer | `0.0378 ± 0.0027` | `0.0219 ± 0.0013` | `0.0408 ± 0.0025` | `0.0240 ± 0.0012` | `0.0405 ± 0.0064` | `0.0257 ± 0.0014` |
| `D_layout_axial_cellT1_final` | Flat + own `report/4` (`in_dim=4`) | TraderTransformer | `0.0408 ± 0.0040` | `0.0385 ± 0.0035` | `0.0409 ± 0.0009` | `0.0507 ± 0.0022` | `0.0478 ± 0.0038` | `0.0442 ± 0.0059` |
| `D_layout_axial_pooled_final` | Flat + own `report/4` (`in_dim=4`) | TraderTransformer | `0.0418 ± 0.0017` | `0.0410 ± 0.0040` | `0.0423 ± 0.0044` | `0.0514 ± 0.0006` | `0.0446 ± 0.0010` | `0.0409 ± 0.0008` |
| `D_layout_grid_full_pos` | `24x2` piece tokens (`in_dim=4`) | 2D grid TraderTransformer | `0.0494 ± 0.0034` | `0.0628 ± 0.0108` | `0.0468 ± 0.0029` | `0.0592 ± 0.0093` | `0.0639 ± 0.0028` | `0.0628 ± 0.0119` |
