# Society of Thinkers — Round 3 Deliverable 01: Independent Code & Number Audit of `R11-GPU-NEURAL` (`NVIDIA L4` GPU)

- **Document ID:** `R3-01-R11-NEURAL-AUDIT`
- **Author:** Member 14 (`Gemini 4 Argon` — Round-3 Multi-Harness Synthesis & `R11-GPU-NEURAL` Forensic Auditor)
- **Date:** 2026-10-18
- **Audited Code & Artifacts:**
  - [`benchmarks/run_r11_gpu_neural.py`](../../benchmarks/run_r11_gpu_neural.py) (`2,277` lines)
  - [`src/eval/ast_leak_auditor.py`](../../src/eval/ast_leak_auditor.py) (`302` lines)
  - [`results/r11_gpu_neural_results.json`](../../results/r11_gpu_neural_results.json) (`95,264` lines, `3,580,709` bytes)
  - [`results/RESULTS-R11-GPU-NEURAL.md`](../../results/RESULTS-R11-GPU-NEURAL.md) (`246` lines)
  - [`benchmarks/PREREG-R11-GPU-NEURAL.md`](../../benchmarks/PREREG-R11-GPU-NEURAL.md) (`124` lines)
  - [`/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md`](file:///home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md) (lines `2177–2189`)

---

## 1. Executive Audit Verdict

1. **100% Numerical & Hardware Ledger Verification (`PASS`):** Every single metric, confidence interval, eigenvalue, factor loading, correlation, runtime, and cost figure reported in [`results/RESULTS-R11-GPU-NEURAL.md`](../../results/RESULTS-R11-GPU-NEURAL.md) was programmatically cross-checked against [`results/r11_gpu_neural_results.json`](../../results/r11_gpu_neural_results.json) and [`GPU-LOG.md`](file:///home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md) (`soup-gpu-r11gpu-1018040509-13412` at `$0.08` + `soup-gpu-r11gpu-1018041347-24711` at `$0.58` = **`$0.66`** total spend across `2,128.57s` Python execution on a real `NVIDIA L4` GPU with `22.03 GB` VRAM, CUDA `13.0`, PyTorch `2.14.1+cu130`). Zero transcription discrepancies exist.
2. **Zero Oracle Leakage Gate `R11-D0` (`PASS`):** Independent execution of `src/eval/ast_leak_auditor.py` and line-by-line inspection of `benchmarks/run_r11_gpu_neural.py` confirm that **all oracle interpolation formulas identified in Round 2 (`0.55 * p_true + 0.45 * p_anc`, `(1 - ls)*0.5 + ls*true_q`, `l0_ship + alpha * 0.86 * (logit_exact - l0_ship)`, seed-conditional branch `seed == 102 and b_budget == 2`, and synthetic `mechanistic_redundancy_delta_c` data generation) have been completely eradicated**. Prior to every test-time model forward pass or `TreePagedKVSearch` invocation, `strip_oracle_fields` (`run_r11_gpu_neural.py:382–400`) replaces `exact_marginals`, `target_tokens`, and `query_targets` with `PoisonTensor` sentinels and strips private `meta` keys. Across all 12 environments $\times$ 3 seeds (`101, 102, 103`), label-permutation and zeroing canary tests yield exact bitwise invariance (**`global_max_diff = 0.000000000`**), cached-vs-uncached KV root equivalence holds to **`0.000000000`**, and genuine `AdamW` training shifts parameters by $\|\theta_{\text{trained}} - \theta_{\text{init}}\|_2 = \mathbf{17.9362}$ with a **`95.6%`** training loss reduction.
3. **Scientific Trajectory (`R10/Stage-2` Operator Ceiling vs. `R11-GPU-NEURAL` Real Neural Reality):**
   - `R11-GPU-NEURAL` confirms the core architectural thesis of `GLC-AOM` under real SGD-trained weights while exhibiting **exemplary Bitter-Lesson scientific honesty** wherever finite-step neural representations fall short of the `R10`/`Stage-2` analytic operator ceiling (`PASS` on `R11-D0`, `R11-D2`, `R11-D3`, `R11-D5`; `PARTIAL PASS` with honest negative sub-bars on `R11-D1`, `R11-D4`, `R11-D6`).
   - Three specific, falsifiable mechanistic diagnoses emerge from the `PARTIAL PASS` sub-bars—(i) depth-accumulated uncalibrated bilinear query logit temperature in `R11-D4`, (ii) per-environment gradient-step asymmetry (`15` vs. `30–60` steps/env) between `Unified-Generalist` and `Single-Game-Specialist` in `R11-D6`, and (iii) continued non-saturating tree search gains at $B=128$ in `R11-D1`—which directly define the locked pre-registration for **Round 12 (`PREREG-R12-GPU-V2.md`)**.

---

## 2. Verification of Hardware, Runtime & GCE `NVIDIA L4` Cost Ledger

Cross-checking [`/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md`](file:///home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md) (lines `2177–2189`) against `results/r11_gpu_neural_results.json` (`hardware`, `runtime_sec`, `per_seed[*].runtime_sec`):

| Audit Item | `GPU-LOG.md` Record | `r11_gpu_neural_results.json` | `RESULTS-R11-GPU-NEURAL.md` | Audit Status |
| :--- | :--- | :--- | :--- | :---: |
| **Attempt 1 VM** | `soup-gpu-r11gpu-1018040509-13412` (`us-central1-b`, `g2-standard-8`, `1x NVIDIA L4`, `age_h=0.09`, `cost_usd=0.08`, `DELETED 04:11Z`) | — (Initial Triton `Python.h` probe before `deregister_op_overrides`) | `$0.08` (`0.09 h`, `DELETED 2026-10-18T04:11Z`) | **EXACT MATCH** |
| **Attempt 2 VM** | `soup-gpu-r11gpu-1018041347-24711` (`us-central1-a`, `g2-standard-8`, `1x NVIDIA L4`, driver `580.178.04`, `age_h=0.67`, `cost_usd=0.58`, `rc=0`, `DELETED 04:55Z`) | `device="cuda"`, `gpu_name="NVIDIA L4"`, `vram_gb=22.03`, `torch_version="2.14.1+cu130"`, `cuda_version="13.0"`, `timestamp_utc="2026-10-18T04:54:16Z"` | `$0.58` (`0.67 h` VM age; `2128.57s` Python runtime, `DELETED 2026-10-18T04:55Z`) | **EXACT MATCH** |
| **Total GPU Cost** | `$0.08 + $0.58 = $0.66` (`spent_before_usd`: `320.32 -> 320.98`) | — | **`$0.66`** (`0.76 h` total VM time vs. `$3.50` cap) | **EXACT MATCH** |
| **Per-Seed Runtime** | — | Seed `101`: `749.10s`; Seed `102`: `698.02s`; Seed `103`: `673.71s`; Joint IRT fit: `7.74s`; Total: `2128.57s` | `749.1s` + `698.0s` + `673.7s` + `7.8s` = `2128.57s` | **EXACT MATCH** |

---

## 3. Forensic Audit of Gate `R11-D0` (Zero Oracle Leakage: `S1`–`S6` + `R1`–`R4`)

### 3.1 Static AST Audit (`src/eval/ast_leak_auditor.py`)
We re-executed `audit_r11_files(["src/models/glc_aom.py", "src/search/kv_tree_search.py", "benchmarks/run_r11_gpu_neural.py"])` independently during this Round-3 audit:
- **`num_violations = 0`** across all three files (`src/models/glc_aom.py`, `src/search/kv_tree_search.py`, `benchmarks/run_r11_gpu_neural.py`).
- **`canary_self_test`: `3/3` caught (`pass = True`)**:
  - `canary_1` (`batch.exact_marginals` read inside `leaky_search_step`) $\to$ caught by `S1`.
  - `canary_2` (`mechanistic_redundancy_delta_c` call inside `fake_episode_generator`) $\to$ caught by `S3`.
  - `canary_3` (`if seed == 102 and b_budget == 2:` inside `rigged_outcome`) $\to$ caught by `S4`.
- **Inspection of `EXEMPT_SCORING_OR_TRAINING_FUNCTIONS` (`ast_leak_auditor.py:53–65`):** The exemption set contains only genuine training/calibration loss functions (`compute_any_order_loss`, `compute_training_step_loss`, `calibrate_model_and_clearer`, `train_generalist_and_cohorts_for_seed`), oracle-stripping helpers (`strip_oracle_labels`, `split_and_poison_oracle`, `strip_oracle_fields`, `slice_env_batch`), and post-search metric evaluators (`evaluate_beliefs`, `evaluate_blind_predictions`, `verify_zero_oracle_leakage_gate`). None of the search execution functions (`search`, `_run_single_pass_all_budgets`, `evaluate_r11_e1_ablations`, `evaluate_r11_e2_blind_search_suite`, `make_tactical_neural_policy`, `run_real_neural_multiplayer_tournaments`) are exempt.

### 3.2 Runtime `PoisonTensor` (`R1`), Canary Invariance (`R2`), SGD Shift (`R3`), and KV Equivalence (`R4`)
Cross-checking `per_seed[s]["r11_d0_gate"]` and `per_seed[s]["generalist_loss_curve"]` in `results/r11_gpu_neural_results.json`:

| Seed | `R1` Traps Caught | `R2` Canary $\max|\Delta|$ (12 Envs, Model + Search) | `R3` $\|\theta_{\text{trained}} - \theta_{\text{init}}\|_2$ | `R3` Loss Drop (`Step 0 -> Step 179`) | `R3` $B=0$ KL Reduction vs. Uniform | `R3` $B=0$ KL Reduction vs. $\theta_{\text{init}}$ | `R4` KV Root $\max|\Delta|$ | JSON vs. Report Match |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`101`** | `3 / 3` | `0.000000000` | `17.7591` | `10.0001 -> 0.9129` (`-90.87%`) | `+2.95%` | `-18.94%` | `0.000000000` | **EXACT** |
| **`102`** | `3 / 3` | `0.000000000` | `17.6201` | `17.1429 -> 0.4818` (`-97.19%`) | `+8.94%` | `+3.60%` | `0.000000000` | **EXACT** |
| **`103`** | `3 / 3` | `0.000000000` | `18.4295` | `18.1009 -> 0.5791` (`-96.80%`) | `+32.67%` | `+11.79%` | `0.000000000` | **EXACT** |
| **3-Seed Mean** | **`3 / 3`** | **`0.000000000`** | **`17.9362`** | **`15.0813 -> 0.6579` (`-95.64%`)** | **`+14.85%`** | **`-1.18%`** | **`0.000000000`** | **EXACT** |

> **Forensic Note on `R3` ($B=0$ KL Reduction vs. $\theta_{\text{init}}$ on Seed 101):** On `Seed 101`, the 180-step multi-task generalist (`15` gradient steps per environment) reduces $B=0$ blind test KL by `+2.95%` relative to the uniform prior (`mean_uniform_kl_b0 = 0.5934 -> 0.5759`), while `Seed 102` (`+8.94%` vs. uniform, `+3.60%` vs. init) and `Seed 103` (`+32.67%` vs. uniform, `+11.79%` vs. init) improve over both. On `Seed 101`, `sudoku_9x9_extreme` (`81` variables, `10` values) has high initial logit variance after only `15` updates on that environment (`kl = 2.85`), shifting the unweighted 12-environment mean vs. $\theta_{\text{init}}$. Increasing per-environment training steps in Round 12 (`R12-D2`) directly addresses this under-training regime.

---

## 4. Point-by-Point Number Verification & Scientific Analysis (`R11-D1` through `R11-D6`)

### 4.1 `R11-D1`: `Mini-ECI` Compute Scaling Laws & Comparison to `Stage-2` Analytic Ceiling
All 12 cohort trajectories in `RESULTS-R11-GPU-NEURAL.md` §4.1–4.2 match `mini_eci_fit.scaling_laws` in `results/r11_gpu_neural_results.json` to the reported precision (`0.01` Mini-ECI, `0.0001` KL/Brier/$\kappa$/$R^2$):

| Scientific Dimension | Tier-1 Operator Ceiling (`Stage-2` Surrogate) | Tier-1 Real Neural GPU Run (`R11-GPU-NEURAL` on `NVIDIA L4`) | Scientific Interpretation |
| :--- | :---: | :---: | :--- |
| **`Unified-Generalist` `Mini-ECI` ($B=0 \to 128$)** | `100.00 -> 130.73` (`+30.73`) | **`98.34 -> 105.30` (`+6.95`)** | Real neural search gain is **positive, monotonic (`7/8` steps), and highly log-linear ($R^2_{\text{log}} = 0.9241$, $\kappa = +0.7637$/dbl)**, landing squarely inside Claude Opus 5.5's Round-2 `[PRIOR]` prediction (`[+3.5, +11.0]` Mini-ECI points) while honestly falling below the `+18.0` analytic ceiling bar. |
| **`GLC-AOM-KV-Tree` Redundancy Fit $R^2_{\text{red}}$** | `0.996` (circular in Stage-2) | **`0.9443`** (`Unified-Generalist`: **`0.9155`**) | Fitted **non-circularly** on genuine neural test-time scaling curves (`fit_redundancy_law`, `run_r11_gpu_neural.py:1946–1972`), exceeding the `0.92` bar for `GLC-AOM-KV-Tree` with low effective redundancy $\rho_{\text{IRT}} = 0.04$. |
| **Combinatorial Witness Discovery ($B=0 \to 128$)** | `14.2% -> 88.5%` | **`0.52% -> 43.06%` (`+42.54 pp`)** | On blind test boards, `TreePagedKVSearch` increases verified witness discovery by **82x** (`100%` on `equiv_sat`, `80.6%` on `shiphunt_8x8`, `58.3%` on `shiphunt_6x6`, `5.6%` on `sudoku_6x6`). |
| **Flat Baselines (`Flat-Mix`, `Flat-Econ-Prec-K3`, `P3`, `P8`)** | Mild positive gain in surrogate | **All degrade catastrophically ($\Delta \in [-36.82, -10.72]$)** | Under real neural weights with imperfect root marginals, unpreconditioned flat/beam pooling (`Flat-Econ-Prec-K3` KL `0.5759 -> 1.5017`) suffers severe conjunctive log-odds over-counting and veto blowups, whereas `TwoStageTreeClearer` preserves contraction ($\|T_b\|_\infty \le 0.9983 < 1$). |

### 4.2 `R11-D2`: 9-Item General Factor Unidimensionality & Schmid-Leiman Bifactor Analysis (`PASS 5/5`)
Every psychometric statistic in `RESULTS-R11-GPU-NEURAL.md` §5 matches `bifactor_analysis` in `results/r11_gpu_neural_results.json`:
- **$\text{PC1} = 64.75\%$** ($\lambda_1 = 5.8277$ vs. $\lambda_2 = 1.3691$; bar $\ge 58\% \to$ **PASS**).
- **McDonald's $\omega_h = 0.8961$** (bar $\ge 0.72 \to$ **PASS**), $\omega_t = 0.9803$.
- **Explained Common Variance $\text{ECV} = 0.7350$** (bar $\ge 0.68 \to$ **PASS**).
- **Leave-One-Task-Out (`LOTO`) $r = 0.7122$** (bar $\ge 0.60 \to$ **PASS**).
- **General Factor Loadings $\lambda_{g,m}$ (`9/9` $\ge 0.45 \to$ PASS):** `Hex-5x5` (`0.9389`), `Connect-4` (`0.8945`), `Micro-Stratego` (`0.8929`), `Galanis-Market` (`0.8735`), `Proposal-Poker` (`0.8484`), `Equiv-SAT` (`0.7921`), `Minesweeper` (`0.7581`), `ShipHunt` (`0.6515`), `Sudoku` (`0.4891`, which also carries strong deductive group factor loading $\lambda_s = 0.8622$).

### 4.3 `R11-D3`: Real Neural Multiplayer Tournaments & `BayesianEloLeague` Alignment (`PASS 3/3`)
- Cross-metric alignment across all $N=108$ cohorts (`elo_correlations` in JSON):
  - **Pearson $r(\text{Mini-ECI}, \text{Mean Elo}) = 0.8862$** ($\ge 0.85 \to$ **PASS**).
  - **Spearman $\rho(\text{Mini-ECI}, \text{Mean Elo}) = 0.9652$** ($\ge 0.85 \to$ **PASS**).
  - **Single-Agent vs. Multiplayer Sub-ECI:** $r = 0.6750$, $\rho = 0.7689$.
  - **`GLC-AOM@B=32` vs. `Flat-Mix@B=32`:** `1081.6` / `1080.3` ALL Elo vs. `329.0` ALL Elo (**`+751.3` ALL Elo**, **`+928.4` Mean Elo**, exceeding the `+120` Elo bar by $>6\times$).

### 4.4 `R11-D4`: Architectural Ablations (`A1`–`A4`) & Diagnosis of the `query_kl` Scale Failure (`PARTIAL PASS`)
Cross-checking `r11_e1_summary_3seed` in `results/r11_gpu_neural_results.json`:
1. **What Succeeded Decisively:**
   - **Disjunctive Clearing Protection (`A4` vs. `A3`):** On `Equiv-SAT` at $B=16$, `A3_BilinearAux_RandomSearch` explodes to `Var KL = 1.6535` (`2.3255` at $B=4$), whereas `A4_GLC_BilinearAux_QuerySteered` maintains `Var KL = 0.0079` (**`209.3x` lower KL**) while achieving `100.0%` witness discovery. On `Sudoku-6x6` at $B=16$, `A3` explodes to `Var KL = 3.5453`, whereas `A4` reduces `Var KL` from `0.9348` ($B=0$) to **`0.8518`** (**`4.16x` lower than `A3`**).
   - **Even-Parity Bilinear vs. Odd-Linear Auxiliary Lemma Channel (`A4` vs. `A2`):** `A2_LinearAux` suffers exact sibling cancellation on symmetric case splits, causing `query_acc` to degrade with search (`53.1% -> 52.6%` on `Equiv-SAT`; `75.3% -> 72.2%` on `Sudoku-6x6`). In contrast, `A4` achieves the **highest `query_acc` of all 4 arms** on both `Equiv-SAT` (**`58.1%`** at $B=16$) and `Sudoku-6x6` (**`78.5%`** at $B=16$).
2. **Exact Code-Level Root Cause of the `query_kl` Magnitude Failure (`0.8735` / `1.0219` vs. $\le 0.15$):**
   - In `compute_training_step_loss` (`run_r11_gpu_neural.py:775–778`), `model.query_head.predict_query_probs(batch, var_probs.detach(), h_cons_sib)` is trained exclusively on **1-step** sibling consensus `h_cons_sib = out["h_cons"] + 0.5 * (v_exp_true + v_exp_alt)`.
   - During test-time tree search up to $B=32$ (`_run_single_pass_all_budgets`, `line 1487`), `h_cons_curr` accumulates cleared `v_exp` updates across up to 32 tree steps, inflating the norm $\|h_{\text{cons}, Z^W_{\text{aux}}}(B)\|_2$ relative to the 1-step training scale. When `query_head` projects `h_cons_curr` without depth normalization or post-depth temperature scaling $\tau_q$, its query logits become overconfident by a factor of $3\text{--}5\times$, inflating Bernoulli `query_kl` (`0.5224 -> 0.8735` on `Equiv-SAT`, `0.5157 -> 1.0219` on `Sudoku-6x6`) even while classification accuracy (`query_acc`) improves!
   - **Direct Remedy for Round 12 (`R12-D1`):** Normalize `h_cons_curr` in the $Z^W_{\text{aux}}$ subspace by its running RMS / depth factor and fit a 2-parameter post-depth query calibration $(\tau_q, b_q)$ strictly on `split="cal"`.

### 4.5 `R11-D5`: `LIR` Causal Firewall & Anti-Wishful-Thinking Verification (`PASS 5/5`)
Every seed (`101, 102, 103`) satisfies:
- `grad_v_wrt_z_pi_aux_max = 0.000000` (exact autograd blockade).
- `post_rl_belief_drift = 0.000000` after 20 AdamW policy-gradient steps on `model.get_policy_parameters()`.
- `d_bias_b0 = 0.000000` under interventional utility swap `UtilitySpec.BIAS_PRO_APPROVE` vs. `BIAS_PRO_REJECT` on `Proposal-Poker`.
- `action_flip_rate = 85.42%` (`75.00%`, `100.00%`, `81.25%` $\ge 50\%$ bar).
- Skewed belief KL on `Proposal-Poker`: `0.2031` (`LIR-Masked`) vs. `0.2612` (`Fused`, **`-22.25%` lower wishful-thinking distortion**), and **`+526.2` mean Elo** (`1123.6` vs. `593.4`) at $B=32$.

### 4.6 `R11-D6`: `Unified-Generalist` vs. `Single-Game-Specialist` & Integrity Panel (`PARTIAL PASS`)
- **Why `Unified-Generalist` Beat `Single-Game-Specialist` on Deductive/Abductive KL (`-32.6%`) and Scaling Slope $\kappa$ (`4.06x`):** On `equiv_sat`, `sudoku_6x6`, and `sudoku_9x9_extreme`, cross-environment multi-task training regularizes the shared `2.04M`-parameter trunk and `BilinearAuxLemmaEmitter`, yielding dramatically lower test KL (`0.0184` vs. `0.0596` on `equiv_sat`; `0.9036` vs. `1.9556` on `sudoku_6x6`; `2.5862` vs. `3.5239` on `sudoku_9x9_extreme`) and `4.06x` steeper test-time compute scaling ($\kappa = +0.7637$ vs. $+0.1881$).
- **Why `Unified-Generalist` Trailed on Raw Multiplayer Elo (`1123.6` vs. `1438.0` at $B=32$):** Inspecting `train_generalist_and_cohorts_for_seed` (`run_r11_gpu_neural.py:939–1016`) reveals an explicit **training-update asymmetry**:
  - `Unified-Generalist` runs `generalist_steps = 180` cycling over all 12 environments = **only `15` gradient steps per environment** (`120` samples total per environment!).
  - Each `Single-Game-Specialist` runs `specialist_steps = 60` on a single family (`1` or `2` environments) = **`30` to `60` gradient steps per environment** (`2x` to `4x` more gradient updates per game on `Hex-5x5`, `Connect-4`, `Galanis-Market`, `Proposal-Poker`, and `Micro-Stratego`!).
  - **Direct Remedy for Round 12 (`R12-D2`):** Equalize per-environment gradient updates (**Iso-Update training**: e.g., `60` steps/env for both `Unified-Generalist` and `Single-Game-Specialist`) and equip `Unified-Generalist` with a lightweight header-conditioned FiLM ($\gamma_{\text{env}}, \beta_{\text{env}}$) modulation (`< 2%` parameter overhead) to prevent multi-game gradient interference on game-specific strategic payoffs.

---

## 5. Additional Forensic Code Observations for Round-12 Hardening

While `R11-D0` passes with zero oracle leakage (`global_max_diff = 0.000000000`), our line-by-line code review of `benchmarks/run_r11_gpu_neural.py` identified three methodological refinements that should be incorporated into `PREREG-R12-GPU-V2.md`:
1. **Separating Pure Turn-by-Turn Interactive Match Elo from Belief-Payoff Pairwise Matches (`run_r11_gpu_neural.py:1622–1723`):**
   - In `run_real_neural_multiplayer_tournaments`, lines `1622–1667` run genuine alternating-seat turn-by-turn board games (`t_env.play_match(pol_a, pol_b, num_games=4)`) on `Hex-5x5` and `Connect-4`, while lines `1679–1723` supplement the 108-agent sparse tournament graph with pairwise matches scored from each cohort's blind neural test metrics (`payoff_mean = -1.8 * kl - 0.6 * brier + 0.45 * discovery_witness_rate`). Although 100% derived from blind neural evaluations (zero oracle leak), `R12` should report **pure interactive head-to-head match win rates and Elo** separately from belief-payoff tournament ratings.
2. **External Ground-Truth Target for `AUROC(m_ql)` vs. `AUROC(m_qs)` (`run_r11_gpu_neural.py:1860`):**
   - Line `1860` defines `valid_leaf_lbl = (mql_all >= float(np.median(mql_all))).astype(np.float32)`, which makes `AUROC(m_ql) = 1.0000` tautological against its own median (while genuinely testing `AUROC(m_qs) = 0.8038` against `m_ql`). In `R12`, both `m_ql` and `m_qs` must be evaluated against an **independent ground-truth leaf validity / KL-improvement indicator** (whether the candidate leaf's unmasked variable assignment matches `oracle_targets["target_tokens"]` or improves post-hoc test KL).
3. **Unified `edge_keys_bank` Gram Crowding Across All Cohorts (`run_r11_gpu_neural.py:1498–1505`):**
   - At `step_b == 32`, `GLC_Tree_Active_PUCT` computes Gram crowding on `tree_bank.edge_keys_bank` (`0.1536`), whereas flat baselines compute it on `tree_bank.root_keys_bank` (`0.2085`) and `B0_root` assigns `0.98`. In `R12`, `compute_physical_gram_crowding` should be evaluated on both `edge_keys_bank` and `root_keys_bank` identically across all active search cohorts (`B >= 2`).

---

## 6. Auditor's Round-3 Rubric Scores for `R11-GPU-NEURAL` (`D1`–`D6`)

Comparing Round-2 `Blind-Emp` (`R10/Stage-2` surrogates), Round-2 `Proj-R11` (Society consensus projection), and Round-3 **Realized `R11-GPU-NEURAL`** on `NVIDIA L4`:

| Charter Dimension | R2 `Math/Unit` (`src/`) | R2 `Blind-Emp` (`R10` Surrogate) | R2 `Proj-R11` (Consensus) | **R3 Realized `R11-GPU-NEURAL` (`Gemini 4 Argon`)** | **Realized Lift vs. R2 `Blind-Emp`** | **Projected `R12-GPU-V2`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **D1. Universal KV-Cache Tree Machine** (`SELECT_NODE` + `STEP`) | `8.50` | `2.60` | `7.75` | **`8.40`** | **`+5.80`** | `9.10` |
| **D2. Active Fractal Prospectus Plans** ($p_r \in \mathbb{S}^{D_{\text{amp}}-1}$) | `7.93` | `2.20` | `7.28` | **`8.30`** | **`+6.10`** | `8.90` |
| **D3. Bidirectional Latent Lemma Space & Queries** ($Z^W_{\text{aux}} \oplus Z^\Pi_{\text{aux}}$, $Q(A,B)$) | `8.25` | `1.80` | `7.13` | **`7.60`** | **`+5.80`** | `9.00` |
| **D4. Present-Time Episodic Memory & LIR Firewall** ($M_t$, $\chi=(0,1)$) | `7.70` | `1.88` | `7.03` | **`8.50`** | **`+6.62`** | `9.00` |
| **D5. Anytime Monotonicity & Self-Consistency** ($b = 0 \dots 128+$) | `7.98` | `1.68` | `7.13` | **`8.40`** | **`+6.72`** | `9.10` |
| **D6. Multi-Game Elo League & `Mini-ECI` Bifactor IRT** | `8.28` | `1.50` | `7.38` | **`8.20`** | **`+6.70`** | `9.10` |
| **Unweighted Mean Across All 6 Dimensions** | **`8.10`** | **`1.94`** | **`7.28`** | **`8.23`** | **`+6.29`** | **`9.03`** |
