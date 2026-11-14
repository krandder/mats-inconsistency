# Trader-Transformer Gate Fixes Sweep: `Z5` (`MultiQuestionEnv`) and `Z6` (`ParityEnv`, 2-bit XOR) at `3000 x 512` Fresh Episodes

- **Status:** `EXPLORATORY` baseline (no confirmatory gates; `B_loss_distill` uses privileged exact prefix-Bayes targets and is graded `Oracle Ceiling`).
- **Commit:** `204e78a5f6222a0fdb81dd48f5a4e2e32dbb4e1b` (`204e78a`, `git_dirty: false`)
- **Generator script:** `python3 scripts/render_tt_gate_readmes.py` (all tables and inline numbers rendered directly from `summary.json`).
- **Runner command:** `PYTHONPATH=. python3 benchmarks/explore_trader_transformer_gate_fixes.py --envs Z5,Z6 --seeds 0,1,2 --fresh-episodes --steps 3000 --batch-size 512 --n-eval 2000 --width 64 --grid-width 64 --layers 3 --commit 204e78a5f6222a0fdb81dd48f5a4e2e32dbb4e1b --out results/explore/tt-gate-xor-fresh-20261021/summary.json`
- **Training regime:** `fresh_episodes = true` (`3000 x 512 = 1536000` fresh episodes per seed, `seeds = [0, 1, 2]`, `width = 64`, `layers = 3`). `Z6-parity3` is a held-out benchmark environment (`src/benchmarks/holdout.py`) and is excluded.
- **Prefix Bayes verification (`2546bb2`):** Runtime assertion `max |prefix_bayes[:, -1] - bayes_logits| < 1e-8` passed (`Z5`: `7.11e-15`, `Z6`: `3.55e-15`).

## 1. Non-Neural Baselines (`n_eval = 2000`, rendered from `summary.json`)

| Baseline | `Z5` Regret | `Z5` KL | `Z6` Regret | `Z6` KL |
|---|---:|---:|---:|---:|
| `bayes` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `oracle_clearing` | `0.0000` | `0.0000` | `0.2466` | `0.2619` |
| `extremized_log_pool` | `0.1341` | `0.1286` | `0.2466` | `0.2619` |
| `log_pool` | `0.1495` | `0.1415` | `0.2466` | `0.2619` |
| `linear_pool` | `0.1591` | `0.1507` | `0.2466` | `0.2619` |
| `sum_logodds` | `0.5177` | `0.5437` | `0.2466` | `0.2619` |

**Baseline notes:** On `Z5`, `oracle_clearing` achieves `0.0000` regret (`kl = 0.0000`), while report-pooling rules (`extremized_log_pool`: `0.1341`, `log_pool`: `0.1495`, `linear_pool`: `0.1591`) leave substantial regret because individual reports on a single sub-question are uninformative about the global XOR without cross-question composition. On `Z6` (2-bit parity), `oracle_clearing` (`0.2466`) and all report-pooling rules (`0.2466` to `0.2466`) fail because individual readings have zero marginal log-odds.

## 2. Summary of `3000 x 512` Fresh-Episode Results on `Z5` and `Z6`

1. **Bidirectional bars and non-`none` causal arms both reach low regret on `Z5` and `Z6`, so `Z5` and `Z6` (2-bit XOR) do not separate the neural architectures:**
   - On `Z6` (2-bit XOR), `A_attn_none` stays near the prior at `0.2466 ± 0.0000`, while every non-`none` arm reaches mean regret at or below `0.0069` (`bar_fct_no_report`: `0.0031 ± 0.0002`, `full_context_transformer_none`: `0.0030 ± 0.0002`, `A_attn_last`: `0.0012 ± 0.0004`, `C_state_recurrent_final`: `0.0007 ± 0.0013`, and `[ORACLE CEILING]` `B_loss_distill`: `0.0003 ± 0.0003`).
   - On `Z5`, `A_attn_none` is `0.1981 ± 0.0061`. The bidirectional bars reach `0.0031 ± 0.0006` (`bar_fct_no_report`) and `0.0039 ± 0.0010` (`full_context_transformer_none`). Final-loss causal arms with last-layer attention reach `0.0051` (`C_state_feedback_final`) to `0.0060` (`A_attn_first_last`), while `A_attn_first` (`0.0120 ± 0.0025`), `B_loss_every_prefix` (`0.0135 ± 0.0007`), and `B_loss_every_prefix_reperm` (`0.0147 ± 0.0009`) remain above `0.0100`. The `[ORACLE CEILING]` arm `B_loss_distill` reaches `0.0045 ± 0.0014` (`mean_prefix_kl = 0.0025 ± 0.0002`).

## 3. Full 3-Seed Results Table (`Z5` and `Z6`, rendered from `summary.json`)

| Arm | `Z5` `in_dim` | `Z6` `in_dim` | Role | `Z5` Regret | `Z5` KL | `Z5` Prefix KL | `Z6` Regret | `Z6` KL | `Z6` Prefix KL |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `full_context_transformer_none` | `14` | `5` | Bidirectional FCT bar | `0.0039 ± 0.0010` | `0.0041 ± 0.0001` | — | `0.0030 ± 0.0002` | `0.0014 ± 0.0002` | — |
| `full_context_transformer_full` | `26` | `8` | Bidirectional FCT bar | `0.0042 ± 0.0006` | `0.0044 ± 0.0001` | — | `0.0031 ± 0.0002` | `0.0015 ± 0.0001` | — |
| `bar_fct_no_report` | `13` | `4` | Bidirectional FCT bar | `0.0031 ± 0.0006` | `0.0037 ± 0.0001` | — | `0.0031 ± 0.0002` | `0.0014 ± 0.0001` | — |
| `bar_fct_fresh_headline_w64_l3` | `14` | `5` | Bidirectional FCT bar | `0.0039 ± 0.0010` | `0.0041 ± 0.0001` | — | `0.0030 ± 0.0002` | `0.0014 ± 0.0002` | — |
| `A_attn_all` | `14` | `5` | Axis (A) causal attention schedule | `0.0055 ± 0.0007` | `0.0048 ± 0.0003` | `0.0589 ± 0.0008` | `0.0026 ± 0.0002` | `0.0015 ± 0.0001` | `0.0449 ± 0.0023` |
| `A_attn_all_reperm` | `14` | `5` | Axis (A) causal attention schedule | `0.0058 ± 0.0014` | `0.0049 ± 0.0002` | `0.0587 ± 0.0015` | `0.0025 ± 0.0006` | `0.0015 ± 0.0002` | `0.0452 ± 0.0024` |
| `A_attn_first_last` | `14` | `5` | Axis (A) causal attention schedule | `0.0060 ± 0.0005` | `0.0054 ± 0.0002` | `0.0637 ± 0.0051` | `0.0023 ± 0.0005` | `0.0015 ± 0.0001` | `0.0441 ± 0.0014` |
| `A_attn_every_other` | `14` | `5` | Axis (A) causal attention schedule | `0.0060 ± 0.0005` | `0.0054 ± 0.0002` | `0.0637 ± 0.0051` | `0.0023 ± 0.0005` | `0.0015 ± 0.0001` | `0.0441 ± 0.0014` |
| `A_attn_last` | `14` | `5` | Axis (A) causal attention schedule | `0.0061 ± 0.0003` | `0.0058 ± 0.0004` | `0.0709 ± 0.0024` | `0.0012 ± 0.0004` | `0.0007 ± 0.0000` | `0.0461 ± 0.0006` |
| `A_attn_first` | `14` | `5` | Axis (A) causal attention schedule | `0.0120 ± 0.0025` | `0.0089 ± 0.0009` | `0.0571 ± 0.0009` | `0.0032 ± 0.0005` | `0.0019 ± 0.0002` | `0.0469 ± 0.0002` |
| `A_attn_none` | `14` | `5` | Axis (A) causal attention schedule | `0.1981 ± 0.0061` | `0.1876 ± 0.0012` | `0.1037 ± 0.0030` | `0.2466 ± 0.0000` | `0.2619 ± 0.0000` | `0.1263 ± 0.0008` |
| `B_loss_every_prefix` | `14` | `5` | Axis (B) loss schedule | `0.0135 ± 0.0007` | `0.0122 ± 0.0001` | `0.0072 ± 0.0001` | `0.0066 ± 0.0004` | `0.0078 ± 0.0008` | `0.0034 ± 0.0005` |
| `B_loss_every_prefix_reperm` | `14` | `5` | Axis (B) loss schedule | `0.0147 ± 0.0009` | `0.0124 ± 0.0001` | `0.0073 ± 0.0001` | `0.0069 ± 0.0010` | `0.0075 ± 0.0008` | `0.0033 ± 0.0005` |
| `B_loss_distill` | `14` | `5` | `[ORACLE CEILING]` (privileged prefix Bayes target) | `0.0045 ± 0.0014` | `0.0039 ± 0.0005` | `0.0025 ± 0.0002` | `0.0003 ± 0.0003` | `0.0005 ± 0.0000` | `0.0002 ± 0.0000` |
| `B_loss_every_prefix_pos` | `14` | `5` | Axis (B) loss schedule | `0.0099 ± 0.0011` | `0.0077 ± 0.0003` | `0.0044 ± 0.0001` | `0.0031 ± 0.0012` | `0.0020 ± 0.0000` | `0.0007 ± 0.0000` |
| `C_state_recurrent_final` | `14` | `5` | Axis (C) token state | `0.0057 ± 0.0014` | `0.0046 ± 0.0001` | `0.0227 ± 0.0009` | `0.0007 ± 0.0013` | `0.0008 ± 0.0000` | `0.0106 ± 0.0030` |
| `C_state_recurrent_prefix` | `14` | `5` | Axis (C) token state | `0.0061 ± 0.0022` | `0.0052 ± 0.0001` | `0.0028 ± 0.0000` | `0.0011 ± 0.0008` | `0.0013 ± 0.0000` | `0.0004 ± 0.0000` |
| `C_state_feedback_final` | `14` | `5` | Axis (C) token state | `0.0051 ± 0.0010` | `0.0044 ± 0.0002` | `0.0453 ± 0.0018` | `0.0009 ± 0.0008` | `0.0011 ± 0.0000` | `0.0076 ± 0.0012` |
| `C_state_feedback_prefix` | `14` | `5` | Axis (C) token state | `0.0055 ± 0.0015` | `0.0058 ± 0.0003` | `0.0033 ± 0.0001` | `0.0023 ± 0.0008` | `0.0017 ± 0.0000` | `0.0006 ± 0.0000` |
| `V_vis_full` | `26` | `8` | TraderTransformer | `0.0050 ± 0.0014` | `0.0050 ± 0.0001` | `0.0585 ± 0.0021` | `0.0014 ± 0.0005` | `0.0013 ± 0.0001` | `0.0454 ± 0.0004` |
