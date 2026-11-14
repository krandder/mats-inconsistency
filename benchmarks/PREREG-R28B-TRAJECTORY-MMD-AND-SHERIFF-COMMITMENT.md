# PREREG-R28B: Annealed & Learned World-Model Trajectory Search ($B \in \{1, 4, 16\}$) for Self-Referential Inconsistency Minimisation (`MMD`) and General-Sum `Sheriff` Stackelberg Policy Commitment on Public OpenSpiel Games

- **Round:** `R28B` (`R28B-TRAJECTORY-MMD-AND-SHERIFF-COMMITMENT`)
- **Date:** `2026-10-20`
- **Status:** Pre-registration draft by `gemini` (`agy`), ready for freeze review (`freeze-R28B`) by `opus-mats` under `docs/PROTOCOL.md` §§1–9 (including Standing Rules 9.1–9.9)
- **Runner:** `benchmarks/run_r28b_trajectory_mmd_and_sheriff_commitment.py`
- **Scorer:** `scripts/score_r28.py` (refuses `smoke_cpu=true` by default per `docs/PROTOCOL.md` §9.8)
- **Seeds:** `101, 102, 103, 104, 105` (`df = 4`, two-sided $t_{0.975,4} = 2.7764451051977987$; smoke runs offset RNG seeds by `SMOKE_RNG_OFFSET = 1000` -> `[1101, 1102, 1103, 1104, 1105]`, `smoke_full_rng_seed_overlap == 0`)
- **Compute Plan:** General CPU or 1x `NVIDIA L4` (`<= 10 min`, OpenSpiel 2.0.2, dispatched by `opus-mats` after freeze)

---

## 1. Research Question & Direct Resolution of `docs/audits/R27C-audit.md`

The independent read-only audit of Round 27C (`docs/audits/R27C-audit.md`) verified that `R27C`'s provenance, bit-identical seed-101 reproducibility, and proximity-vs-no-proximity effect (`Ablation_Neural_NoProx`) held cleanly, but identified three substantive scientific limitations:
1. **`R27C-D1` Finding 1 (Fixed-$\tau$ QRE Floor & Unweighted Descendant Entropy in Counterfactual Backup):** `R27C` used a fixed temperature $\tau_{\text{base}} = 0.08$ across all outer stages and added unweighted descendant entropy inside `evaluate_q_and_reach`, creating a structural Quantal Response Equilibrium (QRE) floor that prevented convergence below `0.4546` on `liars_dice(numdice=1,dice_sides=4)` (bar `<= 0.12`) and `1.0809` on `leduc_poker` (bar `<= 0.95`).
2. **`R27C-D1` Finding 2 (Exact-Tree Only / No Learned World Model or Test-Time Searcher):** `R27C` only distilled exact-tree tabular MMD iterates into an MLP without testing a learned game world model trained from sampled trajectory rollouts or test-time branching search across search budgets $B \in \{1, 4, 16\}$.
3. **`R27C-D2` Finding 3 (Zero-Sum Game Used to Test Policy Commitment):** `R27C-D2` tested policy commitment on zero-sum `kuhn_poker`, where Von Neumann's Minimax Theorem guarantees that committing to a readable policy cannot outperform a simultaneous Nash equilibrium against a rational best-responder.

`R28B` directly resolves all three findings on public OpenSpiel 2.0.2 games:
1. **Exact Counterfactual Backup + Cosine-Annealed Prior Temperature (`Inconsistency_Annealed_MMD` vs `Ablation_FixedTau_R27C` and `Ablation_Annealed_NoProx` in `D1`):**
   - `ExtensiveFormInconsistencyOracle.evaluate_q_and_reach` (`lines 217–268`) computes exact counterfactual action values $Q^\pi(I, a)$ (`V[k] = float(p @ qs)`), verified against `open_spiel.python.algorithms.expected_game_score.policy_value` on **both Player 0 and Player 1** across 20 random policies on all 5 games (`max_oracle_vs_openspiel_value_err <= 1e-9` in `D0`).
   - `train_neural_annealed_mmd` (`lines 504–568`) runs 2-step Mirror-Prox MMD with cosine temperature annealing $\tau_{\text{base}}(m) = \tau_{\text{end}} + \frac{1}{2}(\tau_{\text{start}} - \tau_{\text{end}})(1 + \cos(\pi m / (S_{\text{outer}} - 1)))$ and proximal anchor weight $\beta_{\text{prox}} = 0.80$, distilling the converged stage policy into `PureNeuralInfosetPolicy` (`hidden_size = 64`, zero per-infoset lookup parameters).
   - Directly ablated against `Ablation_FixedTau_R27C` ($\tau_{\text{end}} = \tau_{\text{start}}$, holding every other hyperparameter identical) and `Ablation_Annealed_NoProx` ($\beta_{\text{prox}} = 0.0$).
2. **Learned Neural Game World Model & Test-Time Branching Search ($B \in \{1, 4, 16\}$) Strictly from Sampled Trajectories (`Inconsistency_Sampled_Trajectory_MMD` in `D1`):**
   - `SampledTrajectoryWorldModel` (`lines 574–847`) rolls out $N_{\text{ep}}$ Monte-Carlo trajectory episodes directly on `oracle.game.new_initial_state()` using per-node stratified sampling (`lines 606–640`) and trains a neural payoff world model `LearnedGeneralSumWorldModelNet` (`lines 399–420`) on the observed terminal infoset-action pairs (**zero** calls to `oracle.evaluate_q_and_reach`, verified by hard counter `exact_oracle_calls_in_sampled_mmd == 0` in `D0`).
   - `run_sampled_world_model_search` (`lines 850–908`) executes test-time branching search across search budgets $B \in \{1, 4, 16\}$ (`Ablation_NoSearch_B1` at $B=1$, `Inconsistency_WorldModel_Search_B4` at $B=4$, `Inconsistency_Sampled_Trajectory_MMD` at $B=16$, and `Ablation_Sampled_NoProx` at $B=16, \beta_{\text{prox}}=0.0$) strictly inside `wm.eval_wm_q`, and is compared at matched sample episodes against upstream OpenSpiel `ExternalSamplingSolver` (`External_Sampled_MCCFR`).
3. **General-Sum `Sheriff` Bilevel Stackelberg Commitment & Blind Zero-Sum `Kuhn` Negative Control (`Inconsistency_Stackelberg_Commitment_LIR` in `D2`):**
   - Evaluates Leader (Player 1, Sheriff) policy commitments against a Pessimistic (Weak Stackelberg) Follower (Player 0, Smuggler) Best Response (`evaluate_leader_commitment`, `lines 914–983`, indifference tolerance $\varepsilon_{\text{indiff}} = 0.02$) on OpenSpiel's general-sum `sheriff(max_bribe=2,max_items=2,num_rounds=1)` and `sheriff(max_bribe=2,max_items=2,num_rounds=2)`, plus zero-sum `kuhn_poker` as a **blind negative control** (run through the exact same `zero_sum_mode=False` world-model and search pipeline with zero game-name branching).
   - `train_stackelberg_commitment_lir` (`lines 1020–1123`) runs $B \in \{1, 4, 16\}$ bilevel commitment search stages strictly inside the learned world model (`exact_oracle_calls_in_stackelberg_search == 0` in `D0`), advancing a 3-step Mirror-Prox MMD trajectory and accepting non-MMD Stackelberg candidate commitments only when they generate positive-sum cooperative social welfare ($\widehat{\text{SW}}(\pi_{\text{cand}}) > 0.25$) inside the learned world model.
   - In general-sum `sheriff`, cooperative bribery commitments achieve $\widehat{\text{SW}} \approx 1.63 > 0.25$ and lift Leader payoff from $V_1^{\text{Nash}} \approx -0.67$ to $V_1^{\text{Ours}} \approx +0.93$. In zero-sum `kuhn_poker`, $\widehat{\text{SW}} \approx 0.0 \le 0.25$ everywhere, so the search remains on the minimax MMD trajectory and converges to $V_1^{\text{Ours}} \approx +0.051 \le V_1^{\text{Nash}} \approx +0.055$, empirically confirming Von Neumann's Minimax Theorem.

---

## 2. Public Benchmarks, Baselines, Controls & Ablations

### 2.1 Public OpenSpiel 2.0.2 Games (`5` Games Total)
- **Zero-sum imperfect-information games (`D1`, plus `kuhn_poker` in `D2`):**
  1. `kuhn_poker` (`12` infosets, `30` terminals, `information_state_tensor` dim `11`, `2` actions)
  2. `leduc_poker` (`936` infosets, `5520` terminals, `information_state_tensor` dim `30`, `3` actions)
  3. `liars_dice(numdice=1,dice_sides=4)` (`1024` infosets, `4080` terminals, `information_state_tensor` dim `15`, `8` actions)
- **General-sum extensive-form bargaining/inspection games (`D2`):**
  4. `sheriff(max_bribe=2,max_items=2,num_rounds=1)` (`7` infosets, `18` terminals, `information_state_tensor` dim `13`, `3` actions)
  5. `sheriff(max_bribe=2,max_items=2,num_rounds=2)` (`43` infosets, `108` terminals, `information_state_tensor` dim `18`, `3` actions)

### 2.2 Arms Evaluated in `D1` (`13` Arms per Game per Seed)
1. `Control_Uniform`: Uniform random policy over legal actions at every infoset.
2. `Control_PotOdds`: Zero-parameter action-index prior (`w = [1.35, 0.85, ...]`).
3. `Control_Random_Network`: `PureNeuralInfosetPolicy` with random weights (`rng_seed + 9000`) and identical `0.05` head weight scaling evaluated through `extract_policy_dict` (Rule 9.1).
4. `Control_Untrained_Init`: Initial `PureNeuralInfosetPolicy` (`rng_seed`) extracted before optimization.
5. `External_Sampled_MCCFR`: Upstream `open_spiel.python.algorithms.external_sampling_mccfr.ExternalSamplingSolver` run for the exact same number of episodes `sample_eps` as `SampledTrajectoryWorldModel`.
6. `Ablation_FixedTau_R27C`: Exact-tree Mirror-Prox MMD with fixed temperature $\tau_{\text{end}} = \tau_{\text{start}}$ and $\beta_{\text{prox}} = 0.80$ distilled into `PureNeuralInfosetPolicy`.
7. `Ablation_Annealed_NoProx`: Exact-tree Mirror-Prox MMD with cosine-annealed $\tau_{\text{start}} \to \tau_{\text{end}}$ and $\beta_{\text{prox}} = 0.0$ (undamped softmax best-response iteration) distilled into `PureNeuralInfosetPolicy`.
8. `Inconsistency_Annealed_MMD` (**Ours — Exact-Tree Annealed**): Exact-tree Mirror-Prox MMD with cosine-annealed $\tau_{\text{start}} \to \tau_{\text{end}}$ and $\beta_{\text{prox}} = 0.80$ distilled into `PureNeuralInfosetPolicy`.
9. `Ablation_NoSearch_B1`: Learned trajectory world model `SampledTrajectoryWorldModel` (`0` exact oracle calls) with search budget $B = 1$ distilled into `PureNeuralInfosetPolicy`.
10. `Inconsistency_WorldModel_Search_B4`: Learned trajectory world model with search budget $B = 4$ distilled into `PureNeuralInfosetPolicy`.
11. `Ablation_Sampled_NoProx`: Learned trajectory world model with search budget $B = 16$ and $\beta_{\text{prox}} = 0.0$ distilled into `PureNeuralInfosetPolicy`.
12. `Inconsistency_Sampled_Trajectory_MMD` (**Ours — Learned World Model + Search $B=16$**): Learned trajectory world model (`LearnedGeneralSumWorldModelNet`, `0` exact oracle calls) with search budget $B = 16$ and proximal weight $\beta_{\text{prox,sampled}}$ distilled into `PureNeuralInfosetPolicy`.
13. `Ceiling_Tabular_CFRPlus`: Upstream `open_spiel.python.algorithms.cfr.CFRPlusSolver` tabular reference ceiling.

### 2.3 Arms Evaluated in `D2` (`10` Arms per Game per Seed)
1. `Control_Uniform`: Uniform Leader policy evaluated against pessimistic Follower best response (`eps_indiff = 0.02`).
2. `Control_FirstAction`: Deterministic first-legal-action (`will_inspect=False` / `Pass`) Leader policy.
3. `Control_Random_Network`: Random-weight `PureNeuralInfosetPolicy` (`rng_seed + 9000`) evaluated through `evaluate_leader_commitment` (Rule 9.1).
4. `Control_Untrained_Init`: Untrained initial `PureNeuralInfosetPolicy` (`rng_seed`) evaluated through `evaluate_leader_commitment`.
5. `Baseline_Simultaneous_Nash_CFRPlus`: Simultaneous Nash/CFR+ strategy profile from upstream `cfr.CFRPlusSolver` evaluated under `evaluate_leader_commitment`.
6. `Ablation_Local_Myopic_NoFollowerBR`: 24 local myopic P1 steps against fixed small follower temperature $\tau_f = 0.02$ inside `wm_d2` without multi-candidate follower-BR search, distilled into `PureNeuralInfosetPolicy`.
7. `Ablation_Commitment_NoSearch_B1`: Bilevel commitment search inside `wm_d2` (`0` exact oracle calls) at search budget $B = 1$, distilled into `PureNeuralInfosetPolicy`.
8. `Inconsistency_Stackelberg_Commitment_B4`: Bilevel commitment search inside `wm_d2` (`0` exact oracle calls) at search budget $B = 4$, distilled into `PureNeuralInfosetPolicy`.
9. `Inconsistency_Stackelberg_Commitment_LIR` (**Ours — Bilevel World-Model Commitment Search $B=16$**): Bilevel commitment search inside `wm_d2` (`0` exact oracle calls) at search budget $B = 16$ with cooperative welfare filter $\widehat{\text{SW}} > 0.25$, distilled into `PureNeuralInfosetPolicy`.
10. `Ceiling_Tabular_Grid_Stackelberg`: Deterministic tabular grid ceiling over P1 action-probability thresholds (`compute_tabular_grid_stackelberg_ceiling`).

---

## 3. Pre-Registered Gates (`R28B_D0`, `R28B_D1`, `R28B_D2`)

1. **`R28B_D0_exact_oracle_and_controls_verified`:**
   - All 5 games verify `ExtensiveFormInconsistencyOracle.verify_oracle_vs_openspiel_value` against `expected_game_score.policy_value` on both Player 0 and Player 1 across `>= 20` random policies with `max_oracle_vs_openspiel_value_err <= 1e-9`.
   - Upstream OpenSpiel 2.0.2 `CFRSolver` and `CFRPlusSolver` curves at `[10, 40, 120]` iterations on `kuhn_poker` and `leduc_poker` match `KNOWN_CFR_CURVES` within `max_cfr_cfrplus_curve_abs_err <= 1e-3`.
   - Hard leakage counters verify `exact_oracle_calls_in_sampled_mmd == 0`, `exact_oracle_calls_in_stackelberg_search == 0`, and `smoke_full_rng_seed_overlap == 0`.
   - All 13 `D1` arms and 10 `D2` arms are present for every seed and game (`controls_ok == True`).
2. **`R28B_D1_annealed_and_sampled_mmd_beats_fixed_tau_and_noprox_per_game`:**
   - Tightened absolute `NashConv` bars on `Inconsistency_Annealed_MMD`: `kuhn_ann_nc <= 0.04` (tightened from `0.05` in `R27C`), `leduc_ann_nc <= 0.75` (tightened from `0.95` in `R27C`), and `liars_ann_nc <= 0.12` (maintained from `R27C`).
   - On **every one** of the 3 zero-sum games (`kuhn_poker`, `leduc_poker`, `liars_dice(numdice=1,dice_sides=4)`) individually:
     * `Inconsistency_Annealed_MMD` beats `Ablation_FixedTau_R27C`, `Ablation_Annealed_NoProx`, `Control_Uniform`, `Control_PotOdds`, `Control_Random_Network`, and `Control_Untrained_Init` with 5-seed paired 95% CI lower bound `ci95_low > 0.0`.
     * `Inconsistency_Sampled_Trajectory_MMD` ($B=16$, `0` exact oracle calls) beats `Ablation_NoSearch_B1` ($B=1$), `Inconsistency_WorldModel_Search_B4` ($B=4$), `Ablation_Sampled_NoProx`, `External_Sampled_MCCFR`, `Control_Uniform`, `Control_PotOdds`, `Control_Random_Network`, and `Control_Untrained_Init` with 5-seed paired 95% CI lower bound `ci95_low > 0.0`.
3. **`R28B_D2_general_sum_sheriff_stackelberg_commitment_and_zerosum_control`:**
   - On `sheriff(max_bribe=2,max_items=2,num_rounds=1)`: `v1_ours_mean >= 0.80`, `sw_ours_mean >= 1.50`, and `ci95_low > 0.0` vs `Baseline_Simultaneous_Nash_CFRPlus`, `Ablation_Local_Myopic_NoFollowerBR`, `Ablation_Commitment_NoSearch_B1`, `Control_Uniform`, `Control_FirstAction`, `Control_Random_Network`, and `Control_Untrained_Init`.
   - On `sheriff(max_bribe=2,max_items=2,num_rounds=2)`: `v1_ours_mean >= 0.50`, `sw_ours_mean >= 1.20`, and `ci95_low > 0.0` vs `Baseline_Simultaneous_Nash_CFRPlus`, `Ablation_Local_Myopic_NoFollowerBR`, `Ablation_Commitment_NoSearch_B1`, `Control_Uniform`, `Control_FirstAction`, `Control_Random_Network`, and `Control_Untrained_Init`.
   - On blind zero-sum negative control `kuhn_poker`: `v1_ours_mean - v1_nash_cfrplus_mean <= 0.0001` and `abs(v1_ours_mean - v1_nash_cfrplus_mean) <= 0.02`, while beating `Control_Uniform`, `Control_FirstAction`, `Control_Random_Network`, `Control_Untrained_Init`, `Ablation_Local_Myopic_NoFollowerBR`, and `Ablation_Commitment_NoSearch_B1` with `ci95_low > 0.0`.

---

## 4. Gate–Text Table (`docs/PROTOCOL.md` Rule 9.2 — Verified via `ast.unparse` with `0` Mismatches)

| Gate ID / Variable | Code Location in `benchmarks/run_r28b_trajectory_mmd_and_sheriff_commitment.py` | Exact AST-Unparsed Boolean Expression (`ast.unparse`) |
|---|---|---|
| `controls_ok` (`D0` helper) | `lines 1481–1484` | `bool(all((all((a in per_seed[s]['d1_games'][g] for a in req_d1_arms)) for s in seeds for g in d1_games)) and all((all((a in per_seed[s]['d2_games'][g] for a in req_d2_arms)) for s in seeds for g in d2_games)))` |
| `R28B_D0_exact_oracle_and_controls_verified` (`d0_pass`) | `lines 1489–1498` | `bool(len(ov_checks) == 5 and min_oracle_eps >= 20 and (max_oracle_val_err <= 1e-09) and (max_curve_err <= 0.001) and (oracle_calls_sampled == 0) and (oracle_calls_stackelberg == 0) and (rng_overlap == 0) and controls_ok)` |
| `R28B_D1_annealed_and_sampled_mmd_beats_fixed_tau_and_noprox_per_game` (`d1_pass`) | `lines 1554–1575` | `bool(kuhn_ann_nc <= 0.04 and leduc_ann_nc <= 0.75 and (liars_ann_nc <= 0.12) and all((d1_by_game[g]['ci_annealed_vs_fixed_tau']['ci95_low'] > 0.0 and d1_by_game[g]['ci_annealed_vs_noprox']['ci95_low'] > 0.0 and (d1_by_game[g]['ci_annealed_vs_uniform']['ci95_low'] > 0.0) and (d1_by_game[g]['ci_annealed_vs_pot_odds']['ci95_low'] > 0.0) and (d1_by_game[g]['ci_annealed_vs_random_net']['ci95_low'] > 0.0) and (d1_by_game[g]['ci_annealed_vs_untrained']['ci95_low'] > 0.0) and (d1_by_game[g]['ci_sampled_b16_vs_b1']['ci95_low'] > 0.0) and (d1_by_game[g]['ci_sampled_b16_vs_b4']['ci95_low'] > 0.0) and (d1_by_game[g]['ci_sampled_b16_vs_sampled_noprox']['ci95_low'] > 0.0) and (d1_by_game[g]['ci_sampled_b16_vs_external_mccfr']['ci95_low'] > 0.0) and (d1_by_game[g]['ci_sampled_b16_vs_uniform']['ci95_low'] > 0.0) and (d1_by_game[g]['ci_sampled_b16_vs_pot_odds']['ci95_low'] > 0.0) and (d1_by_game[g]['ci_sampled_b16_vs_random_net']['ci95_low'] > 0.0) and (d1_by_game[g]['ci_sampled_b16_vs_untrained']['ci95_low'] > 0.0) for g in d1_games)))` |
| `R28B_D2_general_sum_sheriff_stackelberg_commitment_and_zerosum_control` (`d2_pass`) | `lines 1622–1649` | `bool(sr1['v1_ours_mean'] >= 0.8 and sr1['sw_ours_mean'] >= 1.5 and (sr1['ci_ours_vs_nash']['ci95_low'] > 0.0) and (sr1['ci_ours_vs_myopic']['ci95_low'] > 0.0) and (sr1['ci_ours_vs_b1']['ci95_low'] > 0.0) and (sr1['ci_ours_vs_uniform']['ci95_low'] > 0.0) and (sr1['ci_ours_vs_first_action']['ci95_low'] > 0.0) and (sr1['ci_ours_vs_random_net']['ci95_low'] > 0.0) and (sr1['ci_ours_vs_untrained']['ci95_low'] > 0.0) and (sr2['v1_ours_mean'] >= 0.5) and (sr2['sw_ours_mean'] >= 1.2) and (sr2['ci_ours_vs_nash']['ci95_low'] > 0.0) and (sr2['ci_ours_vs_myopic']['ci95_low'] > 0.0) and (sr2['ci_ours_vs_b1']['ci95_low'] > 0.0) and (sr2['ci_ours_vs_uniform']['ci95_low'] > 0.0) and (sr2['ci_ours_vs_first_action']['ci95_low'] > 0.0) and (sr2['ci_ours_vs_random_net']['ci95_low'] > 0.0) and (sr2['ci_ours_vs_untrained']['ci95_low'] > 0.0) and (kp['v1_ours_mean'] - kp['v1_nash_cfrplus_mean'] <= 0.0001) and (abs(kp['v1_ours_mean'] - kp['v1_nash_cfrplus_mean']) <= 0.02) and (kp['ci_ours_vs_uniform']['ci95_low'] > 0.0) and (kp['ci_ours_vs_first_action']['ci95_low'] > 0.0) and (kp['ci_ours_vs_random_net']['ci95_low'] > 0.0) and (kp['ci_ours_vs_untrained']['ci95_low'] > 0.0) and (kp['ci_ours_vs_myopic']['ci95_low'] > 0.0) and (kp['ci_ours_vs_b1']['ci95_low'] > 0.0))` |

---

## 5. Constants & Hyperparameters Provenance Table (`docs/PROTOCOL.md` Rule 9.3)

| Constant / Hyperparameter | Value | Code Location (`benchmarks/run_r28b_trajectory_mmd_and_sheriff_commitment.py`) | Provenance / Source Justification |
|---|---|---|---|
| `T_CRIT_DF4` | `2.7764451051977987` | `line 66` | Exact two-sided Student-$t_{0.975, 4}$ critical value for 5 seeds (`df = 4`), inherited from `run_r27c_opponent_inconsistency_mmd.py:55`. |
| `FULL_SEEDS` / `SMOKE_RNG_OFFSET` | `[101, 102, 103, 104, 105]` / `1000` | `lines 67–68` | Standard 5-seed evaluation list (`score_r28.py:35`); smoke mode offsets all RNG seeds by `+1000` (`[1101..1105]`) so smoke and full RNG streams have zero overlap (Rule 9.9). |
| `KNOWN_CFR_CURVES` | OpenSpiel 2.0.2 `CFRSolver` & `CFRPlusSolver` NashConv at `[10, 40, 120]` | `lines 71–80` | Deterministic upstream OpenSpiel 2.0.2 reference values inherited from `run_r27c_opponent_inconsistency_mmd.py:60–81`. |
| `D1_GAME_CONFIGS["kuhn_poker"]` | `tau_start=0.06, tau_end=0.008, beta_prox_annealed=0.80, outer_stages=48 (36 smoke), sample_episodes=36 (30 smoke), tau_end_sampled=0.012, beta_prox_sampled=2.40` | `lines 83–93` | `beta_prox_sampled=2.40` inherited directly from `R27C` (`run_r27c_opponent_inconsistency_mmd.py:725`); `sample_episodes` (`30` smoke / `36` full) are multiples of the `6` Kuhn card deals; cosine schedule $\tau: 0.06 \to 0.008$ clears the fixed-$\tau$ QRE floor (`docs/audits/R27C-audit.md` §3). |
| `D1_GAME_CONFIGS["leduc_poker"]` | `tau_start=0.20, tau_end=0.025, beta_prox_annealed=0.80, outer_stages=56 (44 smoke), sample_episodes=48 (36 smoke), tau_end_sampled=0.025, beta_prox_sampled=1.60` | `lines 94–104` | Cosine annealing from $\tau_{\text{start}}=0.20$ (high exploration over Leduc's 2-round betting tree) to $\tau_{\text{end}}=0.025$ (low-QRE floor) resolves `R27C`'s fixed-$\tau$ stagnation on Leduc (`docs/audits/R27C-audit.md` §7). |
| `D1_GAME_CONFIGS["liars_dice(numdice=1,dice_sides=4)"]` | `tau_start=0.08, tau_end=0.008, beta_prox_annealed=0.80, outer_stages=56 (48 smoke), sample_episodes=48 (36 smoke), tau_end_sampled=0.015, beta_prox_sampled=1.60` | `lines 105–115` | Starts at `R27C`'s exact $\tau_{\text{base}} = 0.08$ (`run_r27c_opponent_inconsistency_mmd.py:724`) and cosine-anneals $10\times$ lower to $\tau_{\text{end}} = 0.008$ to eliminate the `0.4546` QRE floor identified in `docs/audits/R27C-audit.md` §3. |
| `zero_param_pot_odds_policy` weights | `[1.35, 0.85, ...]` | `lines 300–306` | Unchanged zero-parameter baseline inherited from `run_r27c_opponent_inconsistency_mmd.py:303–309`. |
| `PureNeuralInfosetPolicy` & `LearnedGeneralSumWorldModelNet` | `hidden_size = 64`, `head.weight.mul_(0.05)`, `lr = 1.5e-2`, `weight_decay = 1e-4`, `distill_steps = 240 (200 smoke)` | `lines 364–497, 1250` | Architecture and `0.05` head initialization inherited from `run_r27c_opponent_inconsistency_mmd.py:343–398`; `200–240` cosine-annealed AdamW steps ensure full convergence on 936–1024 infoset trees (`leduc_poker`, `liars_dice`). |
| `SampledTrajectoryWorldModel` blend & training | `80` AdamW steps at `lr=1.5e-2`, `0.92` empirical + `0.08` `wm_net` terminal payoff regularizer | `lines 659–683` | Fits `LearnedGeneralSumWorldModelNet` on observed terminal pairs and blends `0.92` empirical Monte-Carlo payoff with `0.08` shared neural world-model prediction across sequence pairs. |
| Search budgets $B \in \{1, 4, 16\}$ | `B = 1` (`NoSearch_B1`), `B = 4` (`Search_B4`), `B = 16` (`Ours`) | `lines 850–908, 1020–1123, 1284–1295, 1368–1376` | Pre-registered geometric search-budget ladder $B \in \{1, 4, 16\}$ testing test-time compute scaling inside the learned world model in both `D1` and `D2`. |
| `evaluate_leader_commitment` & world-model indifference margins | `eps_indiff = 0.02` (exact evaluation), `eps_indiff = 0.035` (world-model search), `sw_c > 0.25` (cooperative welfare threshold) | `lines 759, 917, 1054, 1085, 1113–1114` | `eps_indiff = 0.02` defines pessimistic (Weak Stackelberg) Follower tie-breaking; `eps_indiff = 0.035` inside the world model provides a `0.015` safety margin against neural distillation shift; `sw_c > 0.25` accepts non-MMD Stackelberg commitments only when they generate positive-sum cooperative welfare. |

---

## 6. Pre-Freeze 5-Seed CPU Smoke Summary (`--smoke-cpu`, `SMOKE_RNG_OFFSET = 1000`, `SMOKE_NO_VERDICT`)

Executed via:
```bash
/home/agent/.venvs/trader-cat/bin/python3 benchmarks/run_r28b_trajectory_mmd_and_sheriff_commitment.py --smoke-cpu --out /tmp/r28b_dev/smoke_r28b.json
```
- **All 3 gates pass (`R28B_D0: pass=True`, `R28B_D1: pass=True`, `R28B_D2: pass=True`, all with `verdict="SMOKE_NO_VERDICT"` per Rule 9.8).**
- **D0 Verification:** `num_games_verified = 5`, `min_oracle_check_episodes = 20`, `max_oracle_vs_openspiel_value_err = 4.44e-16 <= 1e-9`, `max_cfr_cfrplus_curve_abs_err = 0.0 <= 1e-3`, `exact_oracle_calls_in_sampled_mmd = 0`, `exact_oracle_calls_in_stackelberg_search = 0`, `smoke_full_rng_seed_overlap = 0`, `controls_ok = True`.

### 6.1 `D1` 5-Seed Smoke Summary (`NashConv` $\downarrow$, Mean over 5 Disjoint Smoke Seeds `1101..1105`)

| Arm | `kuhn_poker` | `leduc_poker` | `liars_dice(1d4)` | Role / Description |
|---|---:|---:|---:|---|
| `Control_Uniform` | `0.9167` | `4.7472` | `1.3101` | Uniform random policy |
| `Control_PotOdds` | `1.1190` | `4.5550` | `1.3661` | Zero-param pot-odds heuristic |
| `Control_Random_Network` | `0.9130` | `4.7666` | `1.3118` | Random-weight `PureNeuralInfosetPolicy` (Rule 9.1) |
| `Control_Untrained_Init` | `0.9158` | `4.7435` | `1.3114` | Untrained initial `PureNeuralInfosetPolicy` |
| `External_Sampled_MCCFR` | `0.4572` | `4.7534` | `1.2584` | Upstream `ExternalSamplingSolver` at matched episodes |
| `Ablation_FixedTau_R27C` | `0.0186` | `1.5822` | `0.3091` | Fixed-$\tau$ exact-tree MMD (`R27C` baseline) |
| `Ablation_Annealed_NoProx` | `0.9167` | `5.6581` | `1.4353` | Annealed $\tau$, $\beta_{\text{prox}} = 0$ (no proximal anchor) |
| **`Inconsistency_Annealed_MMD` (Ours)** | **`0.0030`** | **`0.5642`** | **`0.0934`** | Annealed $\tau_{\text{start}} \to \tau_{\text{end}}$, $\beta_{\text{prox}} = 0.80$ (bars: `<=0.04, <=0.75, <=0.12`) |
| `Ablation_NoSearch_B1` ($B=1$) | `0.4825` | `4.3008` | `1.2226` | Learned world model (`0` oracle calls), $B=1$ |
| `Inconsistency_WorldModel_Search_B4` ($B=4$) | `0.3129` | `3.3933` | `0.9260` | Learned world model (`0` oracle calls), $B=4$ |
| `Ablation_Sampled_NoProx` ($B=16, \beta=0$) | `1.0238` | `5.1103` | `1.4758` | Learned world model (`0` oracle calls), $B=16, \beta_{\text{prox}}=0$ |
| **`Inconsistency_Sampled_Trajectory_MMD` (Ours, $B=16$)** | **`0.1555`** | **`2.6272`** | **`0.5135`** | Learned world model (`0` oracle calls), $B=16$ (`ci95_low > 0` vs $B=4, B=1$, `NoProx`, `MCCFR`) |
| `Ceiling_Tabular_CFRPlus` | `0.0016` | `0.0135` | `0.0020` | Upstream `CFRPlusSolver` tabular ceiling |

### 6.2 `D2` 5-Seed Smoke Summary (Leader Payoff $V_1$ $\uparrow$ and Social Welfare $\text{SW}$ $\uparrow$, Mean over 5 Disjoint Smoke Seeds `1101..1105`)

| Arm | `sheriff(rounds=1)` $V_1$ ($\text{SW}$) | `sheriff(rounds=2)` $V_1$ ($\text{SW}$) | `kuhn_poker` $V_1$ ($\text{SW}$) |
|---|---:|---:|---:|
| `Control_Uniform` | `-1.5000` (`0.0000`) | `-1.5000` (`0.0000`) | `-0.5000` (`0.0000`) |
| `Control_FirstAction` | `0.0000` (`2.0000`) | `0.0000` (`2.0000`) | `-1.0000` (`0.0000`) |
| `Control_Random_Network` | `-1.5003` (`0.0000`) | `-1.5020` (`-0.0101`) | `-0.4962` (`0.0000`) |
| `Control_Untrained_Init` | `-1.5008` (`0.0000`) | `-1.5205` (`-0.1025`) | `-0.4992` (`0.0000`) |
| `Baseline_Simultaneous_Nash_CFRPlus` | `-0.6726` (`0.0000`) | `-0.6726` (`0.0000`) | `0.0547` (`0.0000`) |
| `Ablation_Local_Myopic_NoFollowerBR` | `0.0505` (`2.0168`) | `-0.5247` (`1.2699`) | `0.0024` (`0.0000`) |
| `Ablation_Commitment_NoSearch_B1` ($B=1$) | `0.0005` (`2.0002`) | `0.4133` (`1.6085`) | `-0.0751` (`0.0000`) |
| `Inconsistency_Stackelberg_Commitment_B4` ($B=4$) | `0.5947` (`1.6061`) | `0.5916` (`1.6069`) | `0.0205` (`0.0000`) |
| **`Inconsistency_Stackelberg_Commitment_LIR` (Ours, $B=16$)** | **`0.9302` (`1.6292`)** | **`0.9227` (`1.6288`)** | **`0.0510` (`0.0000`)** |
| `Ceiling_Tabular_Grid_Stackelberg` | `1.0600` (`1.8000`) | `1.0600` (`1.8000`) | `0.0547` (`0.0000`) |

---

## 7. Exact Execution & Scoring Commands

```bash
# 1. Pre-Freeze 5-Seed CPU Smoke Check (uses SMOKE_RNG_OFFSET = 1000; emits SMOKE_NO_VERDICT):
/home/agent/.venvs/trader-cat/bin/python3 benchmarks/run_r28b_trajectory_mmd_and_sheriff_commitment.py --smoke-cpu --out /tmp/r28b_dev/smoke_r28b.json

# 2. Dev Scorer Verification on Smoke JSON (--allow-smoke required by Rule 9.8):
/home/agent/.venvs/trader-cat/bin/python3 scripts/score_r28.py --json /tmp/r28b_dev/smoke_r28b.json --allow-smoke

# 3. Frozen Full Evaluation (dispatched by runner agent after opus-mats freeze):
/home/agent/.venvs/trader-cat/bin/python3 benchmarks/run_r28b_trajectory_mmd_and_sheriff_commitment.py --out results/r28/r28b_results.json

# 4. Deterministic Scorer (refuses smoke_cpu != False, verifies runner/prereg SHA256 and frozen seeds):
/home/agent/.venvs/trader-cat/bin/python3 scripts/score_r28.py --results-dir results/r28 --subrounds r28b --out results/r28/r28b_verdicts.json
```
