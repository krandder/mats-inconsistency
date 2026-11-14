# Trader-Transformer `E1` (`LinearGaussianEnv`) Information-Form & Designed GLS Solver Sweep Across `visibility in {none, noisy, full}` (3 Seeds)

- **Status:** `EXPLORATORY` (`track = explore`; `own_info_gls` arms are tagged **`designed solver (GLS given the model class)`**; `tt_own_info_distill` uses privileged exact prefix-Bayes targets and is tagged `[ORACLE CEILING]`).
- **Commit:** `647f1ac69ba7cc685872ed7e4db48bf0e6e6553e` (`647f1ac`, `git_dirty: false`, `partial: false`)
- **Generator script:** `python3 scripts/render_tt_z9_e1_readmes.py` (all tables and inline numbers rendered directly from `summary.json`).
- **Runner command:** `PYTHONPATH=. python3 benchmarks/explore_trader_transformer_e1_infogls.py --visibilities none,noisy,full --seeds 0,1,2 --fresh-episodes --steps 3000 --scaled-steps 6000 --batch-size 512 --n-eval 2000 --width 128 --layers 4 --commit 647f1ac69ba7cc685872ed7e4db48bf0e6e6553e`
- **Training regime:** `fresh_episodes = true` (`3000 x 512` standard arms and `6000 x 512 = 3072000` scaled arms per seed, `seeds = [0, 1, 2]`, `alignment = unaligned`).
- **Prefix Bayes verification:** Exact Woodbury prefix-Bayes recursion verified against `dv.bayes` at the final trader (`max_err = 7.02e-06`).

## 1. Non-Neural Baselines & Information-Theoretic Floors at `visibility=none` (`n_eval = 2000`, rendered from `summary.json`)

| Baseline / Floor | Regret (nats) | KL (nats) | Notes |
|---|---:|---:|---|
| `bayes` | `0.0000` | `-0.0000` | Standard `aggbench` baseline |
| `oracle_clearing` | `-0.0000` | `-0.0000` | Standard `aggbench` baseline |
| `sum_logodds` | `0.8273` | `0.8289` | Standard `aggbench` baseline |
| `extremized_log_pool` | `0.8293` | `0.8304` | Standard `aggbench` baseline |
| `log_pool` | `2.0475` | `2.0609` | Standard `aggbench` baseline |
| `linear_pool` | `3.3977` | `3.4140` | Standard `aggbench` baseline |
| `floor_none (full marginal)` | `0.6854` | `—` | Exact `visibility=none` floor marginalizing $(B_i, \sigma_i)$ given $R_i$ (`e1floor`) |
| `floor_none (Gaussian head)` | `0.7085` | `—` | Best Gaussian output-head projection of `floor_none` (`e1floor`) |
| `floor_rot (rotation-only)` | `0.6436` | `0.6434` | Marginalizing $Q_i \in O(2)$ only with $\sigma_i$ known |

**Why `visibility=none` has a non-zero floor (`floor_none = 0.6854`, Gaussian output floor `0.7085`):** Under `visibility="none"`, each trader observes only its marginal noise covariance $R_i = B_i B_i^\top + \sigma_i^2 I_2$, which determines neither the orthogonal orientation $Q_i \in O(2)$ nor $\sigma_i^2$. Marginalizing over the full conditional prior given $R_i$ yields an achievable floor regret of `0.6854` (`0.7085` for a Gaussian output head, and `0.6436` if $\sigma_i$ were known), whereas naive independent fusion (`sum_logodds`) gets `0.8273`.

## 2. Key Findings Across the 3 Diagnostic Deltas (`#3865` & `#3881`, rendered from `summary.json`)

1. **Delta 1 (`moment` vs `pure_info` without explicit own-sensor $(A_i, R_i)$ path):**
   - Generic moment-form transformers (`bar_fct_fresh_headline_w64_l3` at `64x3`: `1.0866` at `full`; size-matched `128x4` `bar_fct_moment_w128_l4`: `0.9556`; causal `tt_moment_all`: `0.9372`) and pure neural information-form heads without explicit $(A_i, R_i)$ transport (`tt_pure_info_all`: `0.9938`, `fct_pure_info_bidir`: `0.9556`) fail to reach `sum_logodds` (`0.8273`), because an MLP over a flattened unaligned token struggles to invert per-episode random $2\times 2$ covariance matrices $R_i$ and multiply $A_i^\top R_i^-1 A_i$ from scratch.
2. **Delta 2 (`OwnSensorInfoHead` — carrying each trader's own $(A_i, R_i)$ into natural parameters):**
   - Giving each trader its own analytic natural contribution $\Delta P_i^{\text{own}} = A_i^\top R_i^{-1} A_i, \Delta h_i^{\text{own}} = A_i^\top R_i^{-1} s_i$ plus learned reliability/correlation corrections drops regret across all visibilities to `0.8253 ± 0.0068` (`none`), `0.8292 ± 0.0015` (`noisy`), and `0.8303 ± 0.0004` (`full`) — matching `sum_logodds` (`0.8273`), but showing no gain beyond independent own-sensor pooling because additive natural-parameter pooling over $y$ alone does not eliminate the shared latent noise factor $u$.
3. **Delta 3 (`OwnSensorInfoGLSHead` — Designed Solver: GLS Given the Model Class):**
   - **Important distinction (Designed Solver vs. Learned Aggregation, `#3881`):** `OwnSensorInfoGLSHead` hard-codes the exact E1 linear-Gaussian GLS posterior solve ($\Sigma = \hat{B}\hat{B}^\top + \text{diag}(\hat{\sigma}^2) + \gamma R_k$, $P = I + A^\top \Sigma^{-1} A$, $h = A^\top \Sigma^{-1} s$). Therefore, the neural network only has to extract $B_i$ and $\sigma_i$ from the provenance key; all cross-agent coupling happens in the hand-written $\Sigma^{-1}$ solve, not in attention.
   - **Proof that attention is not doing the aggregation (`tt_own_info_gls_none` with `attn_layers="none"`):** At `visibility="full"`, `tt_own_info_gls_none` (zero attention layers!) achieves **`0.0037 ± 0.0008`**, virtually identical to 4-layer causal `tt_own_info_gls_all` (**`0.0057 ± 0.0018`**), bidirectional `fct_own_info_gls_bidir` (**`0.0052 ± 0.0011`**), and 6000-step `tt_own_info_gls_all_scaled6k` (**`0.0031 ± 0.0008`**). Likewise at `visibility="noisy"`, `tt_own_info_gls_none` achieves **`0.3298 ± 0.0020`** vs `tt_own_info_gls_all` **`0.3416 ± 0.0039`** (`tt_own_info_gls_all_scaled6k`: `0.3131 ± 0.0025`).
   - **At `visibility="none"` (with `has_prov_key=False` fix):** `tt_own_info_gls_none` (`0.8166 ± 0.0014`), `tt_own_info_gls_all` (`0.8180 ± 0.0004`), and `tt_own_info_gls_all_scaled6k` (`0.8064 ± 0.0023`) sit at `sum_logodds` (`0.8273`, about `0.11–0.13` nats above the Gaussian floor `0.7085` / full floor `0.6854`), offering **no gain** at `visibility="none"`.

## 3. Full 3-Seed Regret Table Across `visibility in {none, noisy, full}` (Rendered from `summary.json`)

| Arm | Head Mode | Role | `vis=none` Regret | `vis=noisy` Regret | `vis=full` Regret |
|---|---|---|---:|---:|---:|
| `bar_fct_fresh_headline_w64_l3` | `moment` | Bidirectional FCT | `1.0891 ± 0.0061` | `1.0893 ± 0.0051` | `1.0866 ± 0.0086` |
| `bar_fct_moment_w128_l4` | `moment` | Bidirectional FCT | `0.9555 ± 0.0014` | `0.9582 ± 0.0071` | `0.9556 ± 0.0033` |
| `tt_moment_all` | `moment` | Causal TraderTransformer | `0.9310 ± 0.0013` | `0.9426 ± 0.0047` | `0.9372 ± 0.0026` |
| `tt_pure_info_all` | `pure_info` | Causal TraderTransformer | `0.9952 ± 0.0036` | `0.9921 ± 0.0003` | `0.9938 ± 0.0004` |
| `fct_pure_info_bidir` | `pure_info` | Bidirectional FCT | `0.9561 ± 0.0043` | `0.9568 ± 0.0073` | `0.9556 ± 0.0075` |
| `tt_own_info_all_w64_l3` | `own_info` | Causal TraderTransformer | `0.8167 ± 0.0012` | `0.8293 ± 0.0013` | `0.8201 ± 0.0073` |
| `tt_own_info_all` | `own_info` | Causal TraderTransformer | `0.8253 ± 0.0068` | `0.8292 ± 0.0015` | `0.8303 ± 0.0004` |
| `tt_own_info_all_no_report` | `own_info` | Causal TraderTransformer | `0.8190 ± 0.0023` | `0.8266 ± 0.0055` | `0.8263 ± 0.0061` |
| `fct_own_info_bidir` | `own_info` | Bidirectional FCT | `0.8007 ± 0.0015` | `0.8073 ± 0.0116` | `0.8100 ± 0.0145` |
| `tt_own_info_last` | `own_info` | Causal TraderTransformer | `0.8185 ± 0.0014` | `0.8262 ± 0.0054` | `0.8265 ± 0.0047` |
| `tt_own_info_first` | `own_info` | Causal TraderTransformer | `0.8211 ± 0.0061` | `0.8193 ± 0.0060` | `0.8253 ± 0.0068` |
| `tt_own_info_every_other` | `own_info` | Causal TraderTransformer | `0.8208 ± 0.0031` | `0.8204 ± 0.0064` | `0.8242 ± 0.0085` |
| `tt_own_info_none` | `own_info` | Causal TraderTransformer | `0.8289 ± 0.0002` | `0.8294 ± 0.0006` | `0.8292 ± 0.0001` |
| `tt_own_info_every_prefix` | `own_info` | Causal TraderTransformer | `0.8185 ± 0.0002` | `0.8235 ± 0.0042` | `0.8260 ± 0.0074` |
| `tt_own_info_distill` | `own_info` | `[ORACLE CEILING]` | `0.8072 ± 0.0024` | `0.8065 ± 0.0028` | `0.8062 ± 0.0016` |
| `tt_own_info_recurrent` | `own_info` | Causal TraderTransformer | `0.8129 ± 0.0016` | `0.8118 ± 0.0006` | `0.8110 ± 0.0007` |
| `tt_own_info_feedback` | `own_info` | Causal TraderTransformer | `0.8261 ± 0.0059` | `0.8305 ± 0.0000` | `0.8262 ± 0.0062` |
| `tt_own_info_gls_none` | `own_info_gls` | Designed solver (GLS given model class) | `0.8166 ± 0.0014` | `0.3298 ± 0.0020` | `0.0037 ± 0.0008` |
| `tt_own_info_gls_all` | `own_info_gls` | Designed solver (GLS given model class) | `0.8180 ± 0.0004` | `0.3416 ± 0.0039` | `0.0057 ± 0.0018` |
| `fct_own_info_gls_bidir` | `own_info_gls` | Designed solver (GLS given model class) | `0.8045 ± 0.0078` | `0.3354 ± 0.0032` | `0.0052 ± 0.0011` |
| `tt_own_info_all_scaled6k` | `own_info` | Causal TraderTransformer | `0.8202 ± 0.0071` | `0.8205 ± 0.0067` | `0.8157 ± 0.0006` |
| `tt_own_info_gls_all_scaled6k` | `own_info_gls` | Designed solver (GLS given model class) | `0.8064 ± 0.0023` | `0.3131 ± 0.0025` | `0.0031 ± 0.0008` |

## 4. Full 3-Seed Mean Prefix KL Table (`mean_t KL(P_t^Bayes || P_t^hat)`, Rendered from `summary.json`)

| Arm | Head Mode | `vis=none` Prefix KL | `vis=noisy` Prefix KL | `vis=full` Prefix KL |
|---|---|---:|---:|---:|
| `tt_moment_all` | `moment` | `1.1036 ± 0.0030` | `1.1107 ± 0.0097` | `1.1293 ± 0.0096` |
| `tt_pure_info_all` | `pure_info` | `0.5491 ± 0.0006` | `0.5503 ± 0.0006` | `0.5501 ± 0.0006` |
| `fct_pure_info_bidir` | `pure_info` | `1.3758 ± 0.0224` | `1.3138 ± 0.0294` | `1.3111 ± 0.0300` |
| `tt_own_info_all_w64_l3` | `own_info` | `0.4312 ± 0.0004` | `0.4341 ± 0.0009` | `0.4313 ± 0.0026` |
| `tt_own_info_all` | `own_info` | `0.4340 ± 0.0010` | `0.4346 ± 0.0002` | `0.4343 ± 0.0008` |
| `tt_own_info_all_no_report` | `own_info` | `0.4325 ± 0.0007` | `0.4344 ± 0.0010` | `0.4340 ± 0.0010` |
| `fct_own_info_bidir` | `own_info` | `0.4250 ± 0.0007` | `0.4282 ± 0.0037` | `0.4292 ± 0.0042` |
| `tt_own_info_last` | `own_info` | `0.4305 ± 0.0002` | `0.4330 ± 0.0016` | `0.4331 ± 0.0014` |
| `tt_own_info_first` | `own_info` | `0.4328 ± 0.0010` | `0.4322 ± 0.0012` | `0.4345 ± 0.0004` |
| `tt_own_info_every_other` | `own_info` | `0.4322 ± 0.0004` | `0.4320 ± 0.0022` | `0.4327 ± 0.0028` |
| `tt_own_info_none` | `own_info` | `0.4330 ± 0.0001` | `0.4336 ± 0.0011` | `0.4330 ± 0.0002` |
| `tt_own_info_every_prefix` | `own_info` | `0.4198 ± 0.0003` | `0.4222 ± 0.0020` | `0.4235 ± 0.0034` |
| `tt_own_info_distill` | `own_info` | `0.4098 ± 0.0010` | `0.4097 ± 0.0009` | `0.4089 ± 0.0013` |
| `tt_own_info_recurrent` | `own_info` | `0.4295 ± 0.0007` | `0.4287 ± 0.0003` | `0.4283 ± 0.0007` |
| `tt_own_info_feedback` | `own_info` | `0.4336 ± 0.0027` | `0.4354 ± 0.0003` | `0.4357 ± 0.0045` |
| `tt_own_info_gls_none` | `own_info_gls` | `0.4286 ± 0.0010` | `0.2121 ± 0.0032` | `0.0217 ± 0.0013` |
| `tt_own_info_gls_all` | `own_info_gls` | `0.4298 ± 0.0002` | `0.2208 ± 0.0015` | `0.0240 ± 0.0022` |
| `fct_own_info_gls_bidir` | `own_info_gls` | `0.4258 ± 0.0021` | `0.2181 ± 0.0038` | `0.0241 ± 0.0007` |
| `tt_own_info_all_scaled6k` | `own_info` | `0.4331 ± 0.0011` | `0.4321 ± 0.0020` | `0.4317 ± 0.0002` |
| `tt_own_info_gls_all_scaled6k` | `own_info_gls` | `0.4263 ± 0.0003` | `0.2008 ± 0.0026` | `0.0198 ± 0.0002` |
