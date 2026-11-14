# PREREG-R27C: Neural Infoset-Policy Distillation of Self-Referential Opponent Rationality + Proximity + Prior Inconsistency Minimisation (`MMD`) & Policy Commitments on Public OpenSpiel Games

- **Round:** `R27C` (`R27C-OPPONENT-INCONSISTENCY-MMD`)
- **Date:** `2026-10-20`
- **Status:** Pre-registered (`docs/PROTOCOL.md` §§1–9, including Rules 9.1–9.9)
- **Frozen:** tag `freeze-R27C` by opus-mats, 2026-10-20 ~11:00Z (after four review rounds, `outbox/r27-freeze-review-2026-10-20.md`); runner opus-mats on General CPU (OpenSpiel 2.0.2 by path, torch from the trader-cat venv).
- **Runner:** `benchmarks/run_r27c_opponent_inconsistency_mmd.py`
- **Scorer:** `scripts/score_r27.py`

---

## 1. Explicit Mechanism Statement & Hypotheses

### 1.1 Exact Mechanism of `Inconsistency_ProxPrior_Neural`
To ensure complete transparency (`outbox/r27-freeze-review-2026-10-20.md`, `R27C` Blocking Item 1):
- **What is computed over the game tree:** At each outer stage $m \in \{1, \dots, S_{\text{outer}}\}$, `ExtensiveFormInconsistencyOracle` (`lines 127–336`) evaluates the exact extensive-form counterfactual action values $Q^{\pi_\theta}(I, a)$ under the current neural policy $\pi_\theta$ across all reachable infosets $I \in \mathcal{I}$ of the game tree (`12` infosets on Kuhn Poker, `936` on Leduc Poker, `1024` on Liar's Dice `1d4`), and forms the 2-step Mirror-Prox midpoint target distribution:
  $$p_{\text{stage}}(a \mid I) = \text{softmax}\!\left(\frac{\beta_{\text{eff}}(p)\log \pi_{\text{anc}}(a \mid I) + Q^{\pi_{\text{mid}}}(I, a)}{\tau_{\text{eff}}(p) + \beta_{\text{eff}}(p)}\right)$$
  with per-player root-infoset scaling $\tau_{\text{eff}}(p) = \tau_{\text{base}} / |\mathcal{I}_p^{\text{root}}|$ and $\beta_{\text{eff}}(p) = \beta_{\text{prox}} / |\mathcal{I}_p^{\text{root}}|$.
- **What is neural:** `PureNeuralInfosetPolicy` (`lines 343–376`) is a shared-parameter residual MLP (`in_dim -> 64 -> 64 -> 64 -> max_actions`) taking OpenSpiel's native C++ `information_state_tensor(p)` with **zero per-infoset lookup tables** (`zero info_logit`, `zero info_emb`). At each outer stage, `PureNeuralInfosetPolicy` is trained for $K_{\text{inner}}$ AdamW steps (`lr = 8e-3`, `weight_decay = 1e-4`, `grad_clip = 1.0`) to distill $p_{\text{stage}}(\cdot \mid I)$ across the batch of all game infosets via masked cross-entropy (`lines 502–508`).
- **Tabular Same-Stage Reference (`Ceiling_Tabular_MMD_SameStages`):** Because `Inconsistency_ProxPrior_Neural` distills exact-tree 2-step Mirror-Prox `MMD` steps into a shared neural network, we also report `Ceiling_Tabular_MMD_SameStages` (`lines 746–757`, `813`) — the exact 2-step Mirror-Prox tabular `MMD` algorithm run for the identical number of midpoint stages (`outer_stages`, equal to `2 * outer_stages` Q evaluations) at the identical $\beta_{\text{prox}} = 2.4$ without neural function approximation.

### 1.2 Falsifiable Claims Tested (`R27C_D0`–`R27C_D2`)
1. **Claim `R27C-C1` (Role of the Proximity Edge $\beta_{\text{prox}} > 0$ in Neural Infoset-Policy Distillation):**
   In 2-player zero-sum imperfect-information games (`kuhn_poker`, `leduc_poker`, `liars_dice(numdice=1,dice_sides=4)`), removing the proximity edge (`Ablation_Neural_NoProx`, $\beta_{\text{prox}} = 0.0$, which differs from `Inconsistency_ProxPrior_Neural` **only** in setting `beta_prox = 0.0` vs `2.4`) causes best-response cycling and higher exploitability (`NashConv`). Including the proximity edge ($\beta_{\text{prox}} = 2.4$) significantly reduces `NashConv` (`ci95_low > 0` across all 3 games) relative to `Ablation_Neural_NoProx` and all 4 non-learned/random/untrained controls (`Control_Random_Network`, `Control_Untrained_Init`, `Control_Uniform`, `Control_ZeroParam_PotOdds`), while testing the restored `928ddec` absolute `NashConv` thresholds (`<= 0.05` on Kuhn, `<= 0.95` on Leduc, `<= 0.12` on Liar's Dice). Note that on `liars_dice(numdice=1,dice_sides=4)`, the exact tabular QRE target at $\tau_{\text{base}} = 0.06$ (`|\mathcal{I}_0^{\text{root}}| = 4, |\mathcal{I}_1^{\text{root}}| = 32$) itself has `NashConv = 0.485 > 0.12`, so the restored `<= 0.12` bar fails by construction and is recorded as an honest `FAIL`.
2. **Claim `R27C-C2` (Objective-Aligned Neural Policy-as-Commitment Under Opponent Readability $\alpha_{\text{read}} \in \{0.0, 0.5, 1.0\}$):**
   When Player 0 trains `PureNeuralInfosetPolicy` against a mixture of `(1 - alpha_read) * pi_biased + alpha_read * SoftBR_P1(pi_theta)` (`lines 532–635`), training with $\alpha_{\text{read}} = 1.0$ (commitment-aware objective) achieves significantly higher expected payoff **when evaluated against the reading opponent $\text{SoftBR}_{P_1}(\pi_\theta)$** (`payoff_vs_reader`, `ci95_low > 0`, `mean_diff >= 0.05`) and lower `NashConv` (`<= 0.05`, restored `928ddec` bar) with fixed-point L1 drift `<= 0.01` (restored `928ddec` bar), whereas $\alpha_{\text{read}} = 0.0$ achieves higher payoff against the fixed non-reading opponent `pi_biased` (`payoff_vs_biased`, `ci95_low > 0`, `mean_diff >= 0.03`). The `R27C` claim fails iff any of `R27C_D0`, `R27C_D1`, or `R27C_D2` is `FAIL`.

---

## 2. Exact Game Structural Counts & Per-Player Root Infosets (`Blocking Item 2`)

In `ExtensiveFormInconsistencyOracle` (`lines 158–161`), `num_infosets_per_player[p]` counts the root infosets of player $p$ (`self.infos[k] == p and self.parent[k] is None`, i.e., infosets of player $p$ with no prior action by player $p$). The exact structural dimensions on OpenSpiel `2.0.2` (`pyspiel.__file__ = /tmp/r4e3a/pylib/pyspiel.so`) are:

| OpenSpiel Game String | Total Infosets $|\mathcal{I}|$ | Player 0 Total / Root Infosets $|\mathcal{I}_0^{\text{root}}|$ | Player 1 Total / Root Infosets $|\mathcal{I}_1^{\text{root}}|$ | Native `information_state_tensor` Dim (`in_dim`) | Max Legal Actions | $\tau_{\text{eff}}(0) = \frac{0.06}{|\mathcal{I}_0^{\text{root}}|}$ | $\tau_{\text{eff}}(1) = \frac{0.06}{|\mathcal{I}_1^{\text{root}}|}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `kuhn_poker` | `12` | `6` / `3` | `6` / `6` | `11` | `2` | `0.02` | `0.01` |
| `leduc_poker` | `936` | `468` / `6` | `468` / `12` | `30` | `3` | `0.01` | `0.005` |
| `liars_dice(numdice=1,dice_sides=4)` | `1024` | `512` / `4` | `512` / `32` | `15` | `8` | `0.015` | `0.001875` |

---

## 3. Controls, Baselines, and Rule 9.9 Bar Restoration Table

### 3.1 Controls & Comparators (`PROTOCOL.md` §§4, 9.1, 9.5)
1. **`Control_Uniform` (`lines 271–272`, `723–726`):** Non-learned uniform random policy $\pi(a \mid I) = 1 / |A(I)|$.
2. **`Control_ZeroParam_PotOdds` (`lines 274–280`, `727–730`):** Non-learned zero-parameter action-index bias baseline ($w_0 = 1.35$, $w_{j>0} = 0.85$, normalized over legal actions).
3. **`Control_Random_Network` (`lines 413–430`, `767`, `797–800`, Rule 9.1):** Instantiates the exact `PureNeuralInfosetPolicy` class with a distinct random seed (`seed + 9000`) and identical head scaling (`mul_(0.05)`) and runs through the identical `extract_policy_dict` evaluation pipeline with zero training steps.
4. **`Control_Untrained_Init` (`lines 457–461`, `777–785`, `801–804`, Rule 9.1):** Extracts `pi_init` from `PureNeuralInfosetPolicy` at step 0 (immediately after `mul_(0.05)`) on the training seed `seed`.
5. **`Ablation_Neural_NoProx` (`lines 768–776`, `805–808`):** Identical `train_pure_neural_inconsistency` call, network initialization, stages, inner steps, and learning rate as `Inconsistency_ProxPrior_Neural`, differing **solely** in `beta_prox = 0.0` (vs `2.4`).
6. **`Ceiling_Tabular_MMD_SameStages` (`lines 746–757`, `813`):** Deterministic 2-step Mirror-Prox tabular `MMD` run for the exact same number of midpoint stages (`outer_stages`, equal to `2 * outer_stages` Q evaluations) at `beta_prox = 2.4`.
7. **`OpenSpiel_Upstream_MCCFR_OurRun` (`lines 676–684`, `787–794`, `809–812`, Rule 9.5):** Upstream `OutcomeSamplingSolver` run for `800` episodes (`150` in `--smoke-cpu`), labelled with `"training_regime"` and **ungated** per Rule 9.5.

### 3.2 Complete Rule 9.9 Bar Provenance & Justification Table (Comparison Against Initial `928ddec` and `33c4d10` / `35858f7`)
Per Rule 9.9, every sub-bar from the initial `928ddec` pre-registration (`git show 928ddec:benchmarks/PREREG-R27C-OPPONENT-INCONSISTENCY-MMD.md` §4) is accounted for below with its status and result-independent justification:

| Gate | Bar in `928ddec` | Bar in `33c4d10` / `35858f7` | Current Bar in `b6c137f` | Rule 9.9 Status & Result-Independent Justification |
| :--- | :--- | :--- | :--- | :--- |
| `R27C_D0` `OpenSpiel_Upstream_CFRPlus` curve check | `OpenSpiel_Upstream_CFRPlus` reaches `nash_conv <= 0.02` on `kuhn_poker` and `<= 0.25` on `leduc_poker` at $T = 2000$ iterations | `max_curve_err <= 1e-3` across $T \in \{10, 40, 120\}$ for both `CFR` and `CFR+` | `max_curve_err <= 1e-3` across $T \in \{10, 40, 120\}$ for both `CFR` and `CFR+` | **Replaced by strictly tighter multi-checkpoint trajectory verification:** Checking exact agreement (`<= 1e-3`) with upstream `OpenSpiel 2.0.2` `CFRSolver` and `CFRPlusSolver` across 3 checkpoints (`10, 40, 120`, where `CFR+` already reaches `0.00201` on Kuhn and `0.02192` on Leduc at $T=120$, well below `0.02` and `0.25`) is a strictly tighter deterministic regression test than a single coarse threshold at $T=2000$. |
| `R27C_D0` `OpenSpiel_Upstream_DeepCFR` check | `OpenSpiel_Upstream_DeepCFR` beats `Control_Uniform` (`4.7472`) on `leduc_poker` by `>= 2.0 NashConv` (reaching `<= 2.65` at $T \ge 40$; coded in `928ddec` runner as `leduc_unif > 4.50`) | Removed | Removed | **Removed per Rule 9.5 & Review 1 Item 4 / Review 2 Item 2:** Upstream `DeepCFRSolver` required missing `dm-tree`, called `set_seed(42)` internally, and running it at `24 x 24` traversals rather than the published `10,000 x 4,000` regime violated Rule 9.5. |
| `R27C_D0` Tabular MMD Inconsistency | `Tabular_MMD_Inconsistency_Ceiling` drives $\mathcal{L}_{\mathrm{inc}}(\pi; \tau) \le 10^{-8}$ across all 3 games (`kuhn_poker`, `leduc_poker`, `liars_dice`) | `<= 1e-4` (Leduc omitted in `33c4d10`) | `kuhn_mmd_inc <= 1e-8 and leduc_mmd_inc <= 1e-8 and liars_mmd_inc <= 1e-8` (plus `min_oracle_eps >= 20 and max_oracle_val_err <= 1e-9 and seed_overlap == 0`) | **Restored in full to `928ddec` (`<= 1e-8` on all 3 games)**, plus Rule 9.6 oracle root-value verification (`<= 1e-9`) added per Review 3. |
| `R27C_D1` `Simultaneous_SoftBR_NoProx` failure-to-converge floor | `Simultaneous_SoftBR_NoProx` ($\beta_{\mathrm{prox}} = 0$) fails to converge ($\mathcal{L}_{\mathrm{inc}} \ge 0.05$ and `nash_conv >= 0.25` on `kuhn_poker`) | `kuhn_noprox_nc >= 0.30` in `33c4d10`; removed in `35858f7` | Replaced by paired difference gate `ci_vs_noprox["ci95_low"] > 0.0` across all 3 games | **Replaced per Rule 9.4 & Review 1 Item 2 / Review 3:** In `928ddec`, `Simultaneous_SoftBR_NoProx` was a non-neural tabular Jacobi step without iterate averaging (`Review 1 Item 2`), whereas Rule 9.4 requires the ablation (`Ablation_Neural_NoProx`) to use the identical neural architecture and training pipeline differing solely in `beta_prox = 0.0` vs `2.4`. Under neural SGD distillation, gating the paired difference `ci_vs_noprox["ci95_low"] > 0.0` across all 3 games directly tests the causal contribution of $\beta_{\mathrm{prox}}$ without imposing an absolute floor derived from a non-neural tabular recurrence. |
| `R27C_D1` Monotonicity across budgets | `Inconsistency_ProxPrior_Neural` reduces both $\mathcal{L}_{\mathrm{inc}}$ and `nash_conv` monotonically across budgets | Removed | Removed (`Inconsistency_ProxPrior_Neural` evaluated at fixed `outer_stages` alongside `Ceiling_Tabular_MMD_SameStages`) | **Removed per Review 1 Item 1:** In `928ddec`, multi-budget monotonicity came from evaluating a tabular policy blended with a tiny neural residual across stage checkpoints; once `Inconsistency_ProxPrior_Neural` was made 100% pure neural (`PureNeuralInfosetPolicy` with zero tabular lookup parameters, Review 1 Item 1), each run trains for a fixed stage budget (`outer_stages = 48` on Kuhn/Liar's, `32` on Leduc) and evaluates the final neural network. |
| `R27C_D1` Paired 5-seed CI > 0 vs controls & `NoProx` across all 3 games | Beats `Control_Uniform`, `Control_Untrained_Init`, `Control_ZeroParam_PotOdds`, and `Simultaneous_SoftBR_NoProx` on all 3 games with 5-seed paired 95% CI lower bound `> 0.0` | Kept (plus `Control_Random_Network` added per Rule 9.1) | `ci_vs_rand["ci95_low"] > 0.0 and ci_vs_untr["ci95_low"] > 0.0 and ci_vs_noprox["ci95_low"] > 0.0 and ci_vs_unif["ci95_low"] > 0.0 and ci_vs_pot["ci95_low"] > 0.0` across all 3 games | **Restored in full to `928ddec` (plus `Control_Random_Network` added per Rule 9.1).** |
| `R27C_D1` Kuhn `NashConv` | `<= 0.05` | `<= 0.12` | `kuhn_ours_nc <= 0.05` | **Restored in full to `928ddec` (`<= 0.05`).** |
| `R27C_D1` Leduc `NashConv` | `<= 0.95` | Removed | `leduc_ours_nc <= 0.95` | **Restored in full to `928ddec` (`<= 0.95`).** Records an honest `FAIL` when neural distillation on 936 infosets exceeds `0.95`. |
| `R27C_D1` Liar's Dice `NashConv` | `<= 0.12` | Removed | `liars_ours_nc <= 0.12` | **Restored in full to `928ddec` (`<= 0.12`).** Note: the exact tabular QRE at $\tau_{\text{base}} = 0.06$ on Liar's Dice (`1d4`) itself has `NashConv = 0.485 > 0.12`, so this bar fails by construction and records an honest `FAIL`. |
| `R27C_D2` Payoff at $\alpha_{\text{read}} = 0.0$ vs $\alpha_{\text{read}} = 1.0$ | At $\alpha_{\text{read}} = 0.0$, achieves `>= 90%` of best-response payoff | `ci_a0_vs_a1["mean_diff"] >= 0.03 and ci_a0_vs_a1["ci95_low"] > 0.0` | `ci_reader_a1_vs_a0["mean_diff"] >= 0.05 and ci_reader_a1_vs_a0["ci95_low"] > 0.0 and ci_biased_a0_vs_a1["mean_diff"] >= 0.03 and ci_biased_a0_vs_a1["ci95_low"] > 0.0 and ci_nc_a0_vs_a1["mean_diff"] >= 0.05 and ci_nc_a0_vs_a1["ci95_low"] > 0.0` | **Replaced per Review 1 Item 3 & Review 3 Item 4:** In `928ddec`, `D2` blended two precomputed tabular vectors `(1 - alpha_read) * pi_br + alpha_read * pi_qre` without training the neural network (`Review 1 Item 3`). Replacing that shortcut with genuine neural training (`PureNeuralInfosetPolicy` trained at $\alpha_{\text{read}} \in \{0.0, 0.5, 1.0\}$) and evaluating every trained policy against both `pi_biased` (`payoff_vs_biased`) and `SoftBR(pi_commit)` (`payoff_vs_reader`, `Review 3 Item 4`) tests objective alignment via two-way paired CIs. |
| `R27C_D2` General-sum commitment game bars | Cooperative mutual payoff `>= 0.90` on general-sum commitment game, beating readable greedy best-response by `>= +0.25` payoff per hand with paired 95% CI `> 0` | Removed | Removed | **Removed per Review 1 Item 3:** In `928ddec`, the general-sum commitment game was two hardcoded scalar constants (`0.10` and `sigmoid(18)`) rather than an OpenSpiel game (`Review 1 Item 3`); it was removed rather than fabricated. |
| `R27C_D2` `NashConv` at $\alpha_{\text{read}}=1.0$ | `<= 0.05` | `<= 0.12` | `float(np.mean(nc_a1)) <= 0.05` | **Restored in full to `928ddec` (`<= 0.05`).** |
| `R27C_D2` Fixed-Point L1 Drift at $\alpha_{\text{read}}=1.0$ | `<= 0.01` | `<= 0.08` | `float(np.max(drift_a1)) <= 0.01` | **Restored in full to `928ddec` (`<= 0.01`).** |

---

## 4. Exhaustive Hand-Set Constants & Hyperparameters Provenance Table (`Rule 9.3`)

| Constant / Hyperparameter | Value | Location in `run_r27c_opponent_inconsistency_mmd.py` | Provenance / Justification |
| :--- | :--- | :--- | :--- |
| `T_CRIT_DF4` / `T_CRIT_DF2` | `2.7764451051977987` / `4.302652729911275` | `lines 77–78` | Two-sided 95% Student-$t$ critical values for $\text{df}=4$ (`5` full seeds) and $\text{df}=2$ (`3` smoke seeds). |
| `FULL_SEEDS` | `[101, 102, 103, 104, 105]` | `line 79` | Standard 5-seed protocol across R26–R27 (`df = 4`, $t_{0.975, 4} = 2.7764$). |
| `SMOKE_SEEDS` | `[201, 202, 203]` | `line 80` | Strictly disjoint smoke seed set (`seed_overlap_count == 0`, Rule 9.9). |
| `Control_Random_Network` seed offset | `seed + 9000` | `line 424` | Deterministic independent seed offset for `Control_Random_Network` (`Rule 9.1`). |
| `tau_base` | `0.06` | `lines 546, 694` | Base entropic QRE temperature before per-player root-infoset division $\tau_{\text{eff}}(p) = \tau_{\text{base}} / |\mathcal{I}_p^{\text{root}}|$. |
| `beta_prox` | `2.4` | `lines 547, 695` | Proximity weight for `Inconsistency_ProxPrior_Neural`, `Ceiling_Tabular_MMD_SameStages`, and `D2` (`beta_prox / tau_base = 40`). |
| `beta_prox_tabular` (`D0`) | `1.2` (`0.6` on Leduc), `mmd_iters = 800` on Kuhn/Liar's (`400` smoke), `600` on Leduc (`80` smoke) | `lines 696, 734–735` | 2-step Mirror-Prox tabular MMD step weight and iteration counts for `D0` QRE convergence check (`<= 1e-8`). |
| `hidden_size` | `64` | `lines 350, 425, 457, 566` | 2-block residual MLP width (`PureNeuralInfosetPolicy`). |
| `head` init scaling | `0.05` | `lines 427–428, 459–460, 568–569` | Standard near-uniform softmax policy initialization (`mul_(0.05)`). |
| `lr` / `weight_decay` / `grad_clip` | `8e-3` / `1e-4` / `1.0` | `lines 464, 507, 571, 608` | Standard AdamW optimization hyperparameters across all games and arms (`D1` and `D2`). |
| `outer_stages` (`D1`) | `48` (Kuhn/Liar's), `32` (Leduc); smoke: `36` / `18` | `lines 749, 764` | Number of outer 2-step Mirror-Prox stages (`2 * outer_stages` Q evaluations). |
| `inner_grad_steps` (`D1`) | `28` (full), `20` (smoke) | `line 765` | Inner AdamW distillation steps per outer stage in `D1`. |
| `D2` stages / `inner_steps` | `32` stages / `18` inner steps | `lines 574, 603` | Outer stage and inner step budget for `D2` readability sweep $\alpha_{\text{read}} \in \{0.0, 0.5, 1.0\}$ (`seed + int(alpha_read * 100)` at `line 565`). |
| `pi_biased` (`D2`) | Player 0 uniform `[0.5, 0.5]`; Player 1 `[0.25, 0.75]` (`k.startswith("0")`), `[0.15, 0.85]` (`k.startswith("1")`), `[0.30, 0.70]` (`k.startswith("2")`) | `lines 550–558` | Fixed non-equilibrium card-conditioned Player 1 bias toward action 1 (bet/call) on Kuhn Poker (`35858f7`). |
| `Control_ZeroParam_PotOdds` weights | `1.35` (`j == 0`), `0.85` (`j > 0`) | `line 278` | Fixed non-learned cautious action-index prior. |
| `verify_oracle_vs_openspiel_value` | `n_episodes = 20`, `seed = 2700`, `tau_base = 1e-12` | `lines 311, 320, 719` | Rule 9.6 verification of unregularized root value against `expected_game_score.policy_value`. |
| Upstream `CFR` / `CFR+` / `MCCFR` budgets | `CFR`/`CFR+`: `[10, 40, 120, 400]` (`[10, 40, 120]` smoke); `MCCFR`: `800` episodes (`150` smoke) | `lines 647, 677, 791` | Checkpoints for `D0` curve check and `MCCFR_OurRun` diagnostic. |

---

## 5. Exact Gate–Text Table (`Rule 9.2` — Character-for-Character Match with `evaluate_r27c_gates`)

All gates return `"SMOKE_NO_VERDICT"` when `smoke_cpu=True` (`lines 1004–1007`), and `scripts/score_r27.py` refuses to score any JSON where `smoke_cpu` is not strictly `False` or where `runner_sha256` / `prereg_sha256` / `seeds` do not match the frozen values (`Rule 9.8`).

| Gate ID | Line(s) in `benchmarks/run_r27c_opponent_inconsistency_mmd.py` | Exact Code Condition (Character-for-Character) |
| :--- | :--- | :--- |
| `R27C_D0_upstream_openspiel_exact_curve_reproduction` | `lines 893–901` | `max_curve_err <= 1e-3 and kuhn_mmd_inc <= 1e-8 and leduc_mmd_inc <= 1e-8 and liars_mmd_inc <= 1e-8 and min_oracle_eps >= 20 and max_oracle_val_err <= 1e-9 and seed_overlap == 0` |
| `R27C_D1_proximity_edge_prevents_cycling_and_beats_baselines` (per-game loop) | `lines 928–935` | `ci_vs_rand["ci95_low"] > 0.0 and ci_vs_untr["ci95_low"] > 0.0 and ci_vs_noprox["ci95_low"] > 0.0 and ci_vs_unif["ci95_low"] > 0.0 and ci_vs_pot["ci95_low"] > 0.0` |
| `R27C_D1_proximity_edge_prevents_cycling_and_beats_baselines` (overall verdict) | `lines 958–963` | `all_games_beat and kuhn_ours_nc <= 0.05 and leduc_ours_nc <= 0.95 and liars_ours_nc <= 0.12` |
| `R27C_D2_neural_policy_commitment_readability_and_stability` | `lines 993–1002` | `ci_reader_a1_vs_a0["mean_diff"] >= 0.05 and ci_reader_a1_vs_a0["ci95_low"] > 0.0 and ci_biased_a0_vs_a1["mean_diff"] >= 0.03 and ci_biased_a0_vs_a1["ci95_low"] > 0.0 and ci_nc_a0_vs_a1["mean_diff"] >= 0.05 and ci_nc_a0_vs_a1["ci95_low"] > 0.0 and float(np.mean(nc_a1)) <= 0.05 and float(np.max(drift_a1)) <= 0.01` |

---

## 6. Exact Execution & Scoring Commands

```bash
# 1. Local Dev Smoke Check (uses disjoint SMOKE_SEEDS = [201, 202, 203]; emits SMOKE_NO_VERDICT):
/home/agent/.venvs/trader-cat/bin/python3 benchmarks/run_r27c_opponent_inconsistency_mmd.py --smoke-cpu --out-dir /tmp/r27c_smoke

# 2. Frozen Full Evaluation (uses FULL_SEEDS = [101, 102, 103, 104, 105] on General CPU with OpenSpiel 2.0.2):
/home/agent/.venvs/trader-cat/bin/python3 benchmarks/run_r27c_opponent_inconsistency_mmd.py --out-dir results/r27

# 3. Deterministic Scorer (refuses smoke_cpu != False, verifies runner/prereg SHA256 and frozen seeds):
/home/agent/.venvs/trader-cat/bin/python3 scripts/score_r27.py --results-dir results/r27 --subrounds r27c --out results/r27/r27c_verdicts.json
```
