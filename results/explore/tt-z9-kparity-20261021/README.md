# Trader-Transformer $k$-Parity Depth & Architecture Sweep: `Z9-k3`, `Z9-k4`, `Z9-k6`, and `Z9-hard-k4` (3 Seeds)

- **Status:** `EXPLORATORY` (`track = explore`; `B_loss_distill` uses privileged exact prefix-Bayes targets and is tagged `[ORACLE CEILING]`).
- **Commit:** `ec4bfd1e80dfbbd2309c669df55c1677db48f401` (`ec4bfd1`, `git_dirty: false`, `partial: false`)
- **Generator script:** `python3 scripts/render_tt_z9_e1_readmes.py` (all tables and inline numbers rendered directly from `summary.json`).
- **Runner command:** `PYTHONPATH=. python3 benchmarks/explore_trader_transformer_z9_kparity.py --envs Z9-k3,Z9-k4,Z9-k6,Z9-hard-k4 --seeds 0,1,2 --fresh-episodes --steps 3000 --batch-size 512 --n-eval 2000 --width 128 --layers 4 --commit ec4bfd1e80dfbbd2309c669df55c1677db48f401`
- **Training regime:** `fresh_episodes = true` (`3000 x 512 = 1536000` fresh episodes per seed, `seeds = [0, 1, 2]`, `width = 128`, `layers = 4`, `alignment = unaligned`).
- **Prefix Bayes verification:** Exact tabular prefix-Bayes recursion verified against `dv.bayes['logit']` at the final trader (`Z9-k3`: `2.38e-07`, `Z9-k4`: `2.37e-07`, `Z9-k6`: `2.38e-07`, `Z9-hard-k4`: `1.96e-07`).

## 1. Environment Structure & Non-Neural Baselines (`n_eval = 2000`, rendered from `summary.json`)

| Metric / Baseline | `Z9-k3` (`k=3`, `N=11`) | `Z9-k4` (`k=4`, `N=14`) | `Z9-k6` (`k=6`, `N=20`) | `Z9-hard-k4` (`k=4`, `N=14`) |
|---|---:|---:|---:|---:|
| `bayes` Regret | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `floor_none` Regret | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `oracle_clearing` Regret | `0.5053` | `0.4468` | `0.3656` | `0.0820` |
| `extremized_log_pool` Regret | `0.5053` | `0.4468` | `0.3656` | `0.0820` |
| `log_pool` Regret | `0.5053` | `0.4468` | `0.3656` | `0.0820` |
| `sum_logodds` Regret | `0.5053` | `0.4468` | `0.3656` | `0.0820` |
| `linear_pool` Regret | `0.5053` | `0.4468` | `0.3656` | `0.0820` |

## 2. Key Findings (Rendered from `summary.json`)

1. **Sharp attention-depth threshold on `Z9-k4` ($k=4$ parity, `N=14`, prior regret `0.4468`):**
   - **0 or 1 attention layers fail at or near the prior:** `A_attn_none` (`0` attn layers) stays at `0.4468 ± 0.0001`, `A_attn_last` (`1` attn layer at `l=3`) stays at `0.3160 ± 0.1850`, `A_attn_last_w64_l3` stays at `0.3357 ± 0.1572`, and `A_attn_first` (`1` attn layer at `l=0`) stays at `0.1844 ± 0.1764`. Unlike 2-bit parity (`Z6`), a single attention layer cannot compose 4-way parity across 14 unaligned traders.
   - **2 attention layers partially solve `Z9-k4` across seeds:** `A_attn_every_other` (`2` attn layers at `[1, 3]`) drops regret to `0.0429 ± 0.0077` and `A_attn_first_last` (`2` attn layers at `[0, 3]`) reaches `0.0238 ± 0.0120`.
   - **3 to 4 attention layers solve `Z9-k4` to near-zero regret:** `A_attn_all_w64_l3` (`3` attn layers) reaches `0.0091 ± 0.0067`, `A_attn_all` (`4` attn layers) reaches `0.0120 ± 0.0075`, `A_attn_all_no_report` reaches `0.0123 ± 0.0060`, bidirectional `bar_fct_no_report` reaches `0.0034 ± 0.0020`, `full_context_transformer_none` reaches `0.0017 ± 0.0013`, and `[ORACLE CEILING]` `B_loss_distill` reaches `0.0026 ± 0.0006` (`mean_prefix_kl = 0.0014 ± 0.0002`).
2. **Progression across $k \in \{3, 4, 6\}$ (`Z9-k3` $\to$ `Z9-k4` $\to$ `Z9-k6`):**
   - On `Z9-k3` (`k=3`, prior regret `0.5053`), `A_attn_none` (`0.5054`) fails at the prior, whereas 1-layer attention (`A_attn_last`: `0.0068 ± 0.0018`, `A_attn_last_w64_l3`: `0.0305 ± 0.0233`) and recurrent (`C_state_recurrent_final`: `0.0026 ± 0.0009`) still succeed, and 4-layer causal `A_attn_all` reaches `0.0023 ± 0.0007` (vs bidirectional `full_context_transformer_none`: `0.0007 ± 0.0003`).
   - On `Z9-k6` (`k=6`, `N=20`, prior regret `0.3656`), at `layers = 4, steps = 3000`, **every causal arm stays stuck at the prior** (`A_attn_none`: `0.3656`, `A_attn_last`: `0.3656`, `A_attn_all`: `0.3656 ± 0.0000`, `A_attn_all_reperm`: `0.3656 ± 0.0000`, and `B_loss_distill`: `0.3656 ± 0.0000`), whereas **only the bidirectional `FullContextTransformer` bars break through** (`bar_fct_no_report`: `0.0166 ± 0.0120`, `full_context_transformer_none`: `0.0304 ± 0.0182`, `bar_fct_fresh_headline_w64_l3`: `0.0625 ± 0.0428`).
3. **State Mechanism (`C_state_*`) and `Z9-hard-k4`:**
   - Recurrent GRU state without prefix supervision (`C_state_recurrent_final`) fails at the prior on `Z9-k4` (`0.3490`) and `Z9-k6` (`0.3656`), whereas prefix supervision (`C_state_recurrent_prefix`) partially learns on `Z9-k3` (`0.0031`) and `Z9-k4` (`0.2001`). Similarly, unnormalized feedback with final-only loss (`C_state_feedback_final`) diverges to `NaN` on 2/3 seeds of `Z9-k4` (finite seed regret `0.4467`) and 1/3 seeds of `Z9-k6` (finite mean `0.3656`), while prefix-supervised feedback (`C_state_feedback_prefix`) is stable across all seeds (`Z9-k3`: `0.0034 ± 0.0016`, `Z9-k4`: `0.0070 ± 0.0040`).
   - On `Z9-hard-k4` (random coverage with missing-bit episodes and distractors, `floor_none = 0.0000`), bidirectional bars (`full_context_transformer_none`: `0.0039 ± 0.0028`, `bar_fct_no_report`: `0.0045 ± 0.0017`) and `[ORACLE CEILING]` `B_loss_distill` (`0.0046 ± 0.0007`) reach low regret, while non-oracle causal arms plateau near `0.0820` (`A_attn_all`).

## 3. Full 3-Seed Regret Table Across All 4 `Z9` Environments (Rendered from `summary.json`)

| Arm | Role | `Z9-k3` Regret | `Z9-k4` Regret | `Z9-k6` Regret | `Z9-hard-k4` Regret |
|---|---|---:|---:|---:|---:|
| `bar_fct_fresh_headline_w64_l3` | Bidirectional FCT bar | `0.0021 ± 0.0008` | `0.0025 ± 0.0002` | `0.0625 ± 0.0428` | `0.0308 ± 0.0361` |
| `full_context_transformer_none` | Bidirectional FCT bar | `0.0007 ± 0.0003` | `0.0017 ± 0.0013` | `0.0304 ± 0.0182` | `0.0039 ± 0.0028` |
| `full_context_transformer_full` | Bidirectional FCT bar | `0.0016 ± 0.0004` | `0.0019 ± 0.0001` | `0.1469 ± 0.1549` | `0.0076 ± 0.0007` |
| `bar_fct_no_report` | Bidirectional FCT bar | `0.0005 ± 0.0008` | `0.0034 ± 0.0020` | `0.0166 ± 0.0120` | `0.0045 ± 0.0017` |
| `A_attn_all_w64_l3` | Causal TraderTransformer | `0.0043 ± 0.0015` | `0.0091 ± 0.0067` | `0.3656 ± 0.0000` | `0.0820 ± 0.0000` |
| `A_attn_last_w64_l3` | Causal TraderTransformer | `0.0305 ± 0.0233` | `0.3357 ± 0.1572` | `0.3655 ± 0.0001` | `0.0820 ± 0.0000` |
| `A_attn_all` | Causal TraderTransformer | `0.0023 ± 0.0007` | `0.0120 ± 0.0075` | `0.3656 ± 0.0000` | `0.0820 ± 0.0000` |
| `A_attn_all_reperm` | Causal TraderTransformer | `0.0027 ± 0.0008` | `0.0095 ± 0.0026` | `0.3656 ± 0.0000` | `0.0820 ± 0.0000` |
| `A_attn_all_no_report` | Causal TraderTransformer | `0.0026 ± 0.0006` | `0.0123 ± 0.0060` | `0.3656 ± 0.0000` | `0.0820 ± 0.0000` |
| `A_attn_first_last` | Causal TraderTransformer | `0.0041 ± 0.0016` | `0.0238 ± 0.0120` | `0.3656 ± 0.0000` | `0.0798 ± 0.0031` |
| `A_attn_every_other` | Causal TraderTransformer | `0.0048 ± 0.0007` | `0.0429 ± 0.0077` | `0.3656 ± 0.0000` | `0.0817 ± 0.0006` |
| `A_attn_last` | Causal TraderTransformer | `0.0068 ± 0.0018` | `0.3160 ± 0.1850` | `0.3656 ± 0.0000` | `0.0818 ± 0.0003` |
| `A_attn_first` | Causal TraderTransformer | `0.0101 ± 0.0094` | `0.1844 ± 0.1764` | `0.3656 ± 0.0000` | `0.0810 ± 0.0011` |
| `A_attn_none` | Causal TraderTransformer | `0.5054 ± 0.0001` | `0.4468 ± 0.0001` | `0.3656 ± 0.0000` | `0.0820 ± 0.0000` |
| `B_loss_every_prefix` | Causal TraderTransformer | `0.0114 ± 0.0014` | `0.0198 ± 0.0022` | `0.3656 ± 0.0000` | `0.0820 ± 0.0000` |
| `B_loss_every_prefix_reperm` | Causal TraderTransformer | `0.0117 ± 0.0010` | `0.0217 ± 0.0054` | `0.3657 ± 0.0000` | `0.0820 ± 0.0000` |
| `B_loss_every_prefix_pos` | Causal TraderTransformer | `0.0048 ± 0.0002` | `0.4468 ± 0.0001` | `0.3656 ± 0.0000` | `0.0820 ± 0.0000` |
| `B_loss_distill` | `[ORACLE CEILING]` | `0.0017 ± 0.0007` | `0.0026 ± 0.0006` | `0.3656 ± 0.0000` | `0.0046 ± 0.0007` |
| `C_state_recurrent_final` | Causal TraderTransformer | `0.0026 ± 0.0009` | `0.3490 ± 0.1383` | `0.3656 ± 0.0000` | `0.0820 ± 0.0000` |
| `C_state_recurrent_prefix` | Causal TraderTransformer | `0.0031 ± 0.0010` | `0.2001 ± 0.1842` | `0.3656 ± 0.0000` | `0.0819 ± 0.0001` |
| `C_state_feedback_final` | Causal TraderTransformer | `0.0017 ± 0.0013` | `0.4467 ± 0.0000` (1/3 fin) | `0.3656 ± 0.0000` (2/3 fin) | `0.0820 ± 0.0000` |
| `C_state_feedback_prefix` | Causal TraderTransformer | `0.0034 ± 0.0016` | `0.0070 ± 0.0040` | `0.3656 ± 0.0000` | `0.0820 ± 0.0000` |
| `V_vis_full` | Causal TraderTransformer | `0.0032 ± 0.0002` | `0.1693 ± 0.1969` | `0.3656 ± 0.0000` | `0.0820 ± 0.0000` |

## 4. Full 3-Seed Mean Prefix KL Table (`mean_t KL(P_t^Bayes || P_t^hat)`, Rendered from `summary.json`)

| Arm | `Z9-k3` Prefix KL | `Z9-k4` Prefix KL | `Z9-k6` Prefix KL | `Z9-hard-k4` Prefix KL |
|---|---:|---:|---:|---:|
| `A_attn_all_w64_l3` | `0.6367 ± 0.0056` | `0.5998 ± 0.0881` | `0.1096 ± 0.0000` | `0.0225 ± 0.0000` |
| `A_attn_last_w64_l3` | `0.5285 ± 0.0747` | `0.2918 ± 0.1718` | `0.1096 ± 0.0000` | `0.0225 ± 0.0000` |
| `A_attn_all` | `0.6465 ± 0.0213` | `0.3815 ± 0.0705` | `0.1096 ± 0.0000` | `0.0225 ± 0.0000` |
| `A_attn_all_reperm` | `0.6431 ± 0.0470` | `0.4680 ± 0.0245` | `0.1096 ± 0.0000` | `0.0225 ± 0.0000` |
| `A_attn_all_no_report` | `0.6981 ± 0.0367` | `0.3855 ± 0.0557` | `0.1096 ± 0.0000` | `0.0225 ± 0.0000` |
| `A_attn_first_last` | `0.6376 ± 0.0449` | `0.3674 ± 0.0249` | `0.1096 ± 0.0000` | `0.0296 ± 0.0101` |
| `A_attn_every_other` | `0.5689 ± 0.0374` | `0.4989 ± 0.0169` | `0.1096 ± 0.0000` | `0.0226 ± 0.0002` |
| `A_attn_last` | `0.2248 ± 0.0349` | `0.2838 ± 0.1605` | `0.1096 ± 0.0000` | `0.0288 ± 0.0089` |
| `A_attn_first` | `0.5926 ± 0.0108` | `0.4450 ± 0.1905` | `0.1096 ± 0.0000` | `0.0238 ± 0.0018` |
| `A_attn_none` | `0.2211 ± 0.0000` | `0.1703 ± 0.0000` | `0.1096 ± 0.0000` | `0.0225 ± 0.0000` |
| `B_loss_every_prefix` | `0.0057 ± 0.0003` | `0.0088 ± 0.0008` | `0.1096 ± 0.0000` | `0.0225 ± 0.0000` |
| `B_loss_every_prefix_reperm` | `0.0057 ± 0.0003` | `0.0098 ± 0.0016` | `0.1096 ± 0.0000` | `0.0225 ± 0.0000` |
| `B_loss_every_prefix_pos` | `0.0020 ± 0.0003` | `0.1703 ± 0.0000` | `0.1096 ± 0.0000` | `0.0225 ± 0.0000` |
| `B_loss_distill` | `0.0017 ± 0.0008` | `0.0014 ± 0.0002` | `0.1096 ± 0.0000` | `0.0013 ± 0.0001` |
| `C_state_recurrent_final` | `0.1189 ± 0.0065` | `0.2721 ± 0.1439` | `0.1096 ± 0.0000` | `0.0225 ± 0.0000` |
| `C_state_recurrent_prefix` | `0.0008 ± 0.0000` | `0.0781 ± 0.0695` | `0.1096 ± 0.0000` | `0.0225 ± 0.0000` |
| `C_state_feedback_final` | `0.4089 ± 0.0717` | `0.1703 ± 0.0000` (1/3 fin) | `0.1096 ± 0.0000` (2/3 fin) | `0.0225 ± 0.0000` |
| `C_state_feedback_prefix` | `0.0014 ± 0.0001` | `0.0030 ± 0.0010` | `0.1096 ± 0.0000` | `0.0225 ± 0.0000` |
| `V_vis_full` | `0.6993 ± 0.0530` | `0.3925 ± 0.1607` | `0.1096 ± 0.0000` | `0.0225 ± 0.0000` |
