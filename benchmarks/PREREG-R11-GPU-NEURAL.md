# Pre-Registration Protocol: Round 11 (`R11-GPU-NEURAL`) — End-to-End SGD-Trained Neural `GLC-AOM` & Zero-Oracle-Leakage `NVIDIA L4` GPU Benchmark

- **Document ID:** `PREREG-R11-GPU-NEURAL`
- **Status:** **LOCKED PRIOR TO ROUND-11 GPU EXECUTION**
- **Date:** 2026-10-18
- **Repository:** `/home/agent/workspaces/inconsistency-search` (`futarchy-fi/inconsistency-search`)
- **Governing Audit & Critiques (`society/round2/`):**
  - [`society/round2/01-round2-code-and-results-audit.md`](../society/round2/01-round2-code-and-results-audit.md) (Member 12 Forensic Code Audit)
  - [`society/round2/02-glm53-round2-critique.md`](../society/round2/02-glm53-round2-critique.md) (Member 4a `GLM-5.3` Critique & DAgger/P9 Spec)
  - [`society/round2/03-gpt6-astra-round2-critique.md`](../society/round2/03-gpt6-astra-round2-critique.md) (Member 4b `GPT-6-Astra` Critique & `R11-D0`/LSO Spec)
  - [`society/round2/04-opus-round2-critique.md`](../society/round2/04-opus-round2-critique.md) (Member 4c `Claude Opus 5.5` Critique, `[PRIOR]` Ledger & `PoisonTensor` Spec)
  - [`society/round2/ROUND2-SCOREBOARD.md`](../society/round2/ROUND2-SCOREBOARD.md) (Round-2 Multi-Model Consensus Scoreboard)

---

## 0. Purpose, Retroactive Reclassification & Hardware Specification

### 0.1 Retroactive Reclassification of Stage-1 (`R10`) and Stage-2 (`Mini-ECI`) Results
Per the unanimous Round-2 Society of Thinkers audit (`ROUND2-SCOREBOARD.md` §2.1):
1. All architectural, algebraic, autograd-firewall, spectral-contraction, and statistical/IRT estimator verifications in `src/models/glc_aom.py`, `src/search/kv_tree_search.py`, `src/eval/elo_league.py`, and `src/eval/mini_eci.py` stand as **`UNIT-MATH-VERIFIED` (`A-EXACT` / `A-LIVE-UNTRAINED`)**.
2. All multi-budget KL, Brier, Witness, Elo, and `Mini-ECI` scaling curves in `benchmarks/run_r10_e1_e2.py`, `benchmarks/run_r10_e3_e4.py`, and `benchmarks/run_stage2_mini_eci.py` that relied on untrained world-model trunks, oracle-interpolated log-odds (`0.55 * p_true + 0.45 * p_anc`, `(1 - ls)*0.5 + ls*true_q`, `l0_ship + alpha * 0.86 * (logit_exact - l0_ship)`), seed-conditional branches (`run_r10_e1_e2.py:1351`), algorithm-dependent kernel smoothing (`run_stage2_mini_eci.py:608–640`), circular `mechanistic_redundancy_delta_c` generation (`run_stage2_mini_eci.py:913–922`), or latent BTD match sampling (`run_stage2_mini_eci.py:1177–1190`) are reclassified as **`CONTROLLED-SURROGATE-SIMULATION` (`B-SURROGATE`)**.
3. **Round 11 (`R11-GPU-NEURAL`)** is the definitive, zero-surrogate, end-to-end SGD-trained neural evaluation on an **`NVIDIA L4` GPU** (`24 GB VRAM`, `bf16` / `fp32`), where `exact_marginals`, `target_tokens`, and `query_targets` are physically stripped and poisoned before any test-time model forward pass or search call.

### 0.2 Locked Split, Seed & Compute Accounting Ledger
- **Seeds:** Minimum $N_{\text{seeds}} = 3$ independent end-to-end training & evaluation seeds (`Seeds 1101, 1102, 1103`; paired two-sided 95% Student-$t$ $t_{0.975,2} = 4.303$, expanded to $N_{\text{seeds}} = 5$ `1101..1105`, $t_{0.975,4} = 2.776$ where budget permits), plus hierarchical bootstrap (`10,000` replicates over seeds and episodes).
- **Four-Way Disjoint Data Splits (Zero Puzzle/Generator Ancestry Overlap):**
  1. `train` (`OFFSET = 0`): Used exclusively for AdamW gradient updates on `GLCAOMWorldModel` (`theta_real`, `BilinearAuxLemmaEmitter`, `CombinatorialQueryHead`, `MoPSFractalPlanHead`, `action_policy_mlp`).
  2. `cal_fit` (`OFFSET = 100_000`): Used exclusively to fit `TwoStageTreeClearer` (`<= 8,192` parameters), `DepthDebiasedTrustHead` horizon-bias table $\hat{b}_{\text{hor}}(d)$, Leave-Subtree-Out (`LSO`) innovation predictors, and candidate certificate step sizes $\alpha_b$.
  3. `cal_val` (`OFFSET = 200_000`): Used exclusively for early stopping, hyperparameter selection, and empirical-Bernstein / bootstrap-$t$ lower confidence bound ($\mathrm{LCB}_{1-\delta}$) validation of $\Gamma_b(\alpha)$.
  4. `test` (`OFFSET = 300_000`): Sealed held-out evaluation split. Every test `EnvBatch` is split into `(batch_blind, oracle_labels) = batch.split_and_poison_oracle()` **before** entering any model or search function.
- **Compute Budgets Evaluated (WTEs):** $B \in \{0, 1, 2, 4, 8, 16, 32, 64, 128\}$ (with out-of-sample scaling extrapolation tested from fit budgets $B \in \{0, 1, 2, 4, 8, 16, 32\}$ to held-out budgets $B \in \{64, 128\}$).
- **Three-Ledger Hardware Accounting on `NVIDIA L4`:**
  1. *Training Ledger:* Total optimizer steps, tokens processed, estimated training FLOPs, L4 GPU wall-clock seconds, and initial-to-final training/validation loss curves.
  2. *Inference Search Ledger:* Exact `GLCAOMWorldModel` forward-pass calls (WTEs), `TreePagedKVBank` bytes read/written and KV-reuse ratio, and L4 wall-clock milliseconds per decision.
  3. *Parameter Ledger:* Exact parameter counts for trunk, `BilinearAuxLemmaEmitter`, `CombinatorialQueryHead`, `MoPSFractalPlanHead`, `DepthDebiasedTrustHead`, and `TwoStageTreeClearer` ($\le 8,192$ calibration parameters).

---

## 1. Gate `R11-D0` — Zero-Oracle-Leakage Code Audit & Runtime Firewall (Hard Prerequisite)

> **Rule `R11-D0` (`PASS` required before any `R11-D1`–`R11-D6` metric can be unblinded or reported):**
> Any violation of `S1`–`S5` or `R1`–`R4` immediately voids the entire benchmark run (`VERDICT = VOID_ORACLE_LEAK`).

### 1.1 Static AST Audit Rules (`S1`–`S5`)
An automated `ast.NodeVisitor` auditor (`src/eval/ast_leak_auditor.py`) inspects every function in `benchmarks/run_r11_*.py` and `src/` reachable from the test-time prediction and tournament execution call graph:
- **`S1` (Banned Oracle Attributes & Identifiers):** Zero reads of `.exact_marginals`, `.target_tokens`, `.query_targets`, or banned oracle variable names (`p_true*`, `p_exact*`, `logit_exact*`, `p_a_given_b_true`, `p_b_given_a_true`, `cond_b_given_a*`, `cond_a_given_b*`, `true_q`, `q_star*`, `true_llr_shift`) inside any model, search, memory, or match-play function.
- **`S2` (Banned Synthetic Metric/Gate Samplers):** Zero calls to `rng.beta`, `rng.uniform`, `rng.normal`, or `torch.randn` that write directly into reported evaluation metrics, `m_ql`, `m_qs`, `delta_rel`, or `auroc` buffers outside genuine `GLCAOMWorldModel` / `TreePagedKVSearch` tensor operations.
- **`S3` (Banned Law-Under-Test in Data/Match Generators):** Zero calls to `mechanistic_redundancy_delta_c` or `btd_probabilities` inside episode evaluation or tournament match generation (`mechanistic_redundancy_delta_c` may appear only inside the post-hoc curve fitter `fit_mechanistic_redundancy_law`, and `btd_probabilities` may appear only inside `BayesianEloLeague.fit`).
- **`S4` (Banned Seed/Budget/Arm Conditional Outcome Branches):** Zero `if` statements conditioned on literals of `seed`, `b_budget`, `m0`, or `arm_name` that modify noise scales, leak terms, or probability shifts (permanently banning constructs like `run_r10_e1_e2.py:1351`).
- **`S5` (Banned Arm-Specific Kernel/Offset Constants in Diagnostics):** `compute_empirical_gram_crowding` and all diagnostic functions must apply a **single, identical, label-blind formula** directly to `bank.edge_keys_bank` across all algorithms without `if algorithm == ...` kernel widths or background offsets.
- **`S6` (Canary Self-Test):** The AST auditor must be tested against 3 synthetic canary functions containing (i) an `.exact_marginals` read, (ii) a `mechanistic_redundancy_delta_c` generator call, and (iii) a `seed == 102 and b_budget == 2` branch, catching `3/3` canaries while reporting `0` violations on the R11 runner.

### 1.2 Runtime `PoisonTensor` & Canary Invariance Rules (`R1`–`R4`)
- **`R1` (`PoisonTensor` Oracle Stripping):** Before any test-time call to `model(...)` or `search_engine.run_search(...)`, the test batch is sanitized via `batch_blind = batch.strip_oracle_labels()`, which sets `exact_marginals`, `target_tokens`, and `query_targets` to `PoisonTensor` instances (subclasses/wrappers that raise `OracleLeakError` on any `__torch_function__`, indexing, `.item()`, `.cpu()`, or `.numpy()` access) and strips all non-public keys from `batch_blind.meta`.
- **`R2` (Label-Permutation Canary Invariance Test):** On a canary subset of $N = 64$ test boards per environment, `search_engine.run_search(env, batch_blind, budget=32)` is executed twice under identical RNG seeds: once derived from the true `EnvBatch`, and once derived from a corrupted `EnvBatch` whose `exact_marginals`, `target_tokens`, and `query_targets` were replaced by random permutations prior to calling `strip_oracle_labels()`. The resulting `var_probs`, `query_probs`, `selected_nodes`, and `step_var` tensors must agree to $\max |\Delta| = 0.000000$ (exact bitwise identity under deterministic CUDA).
- **`R3` (Non-Trivial SGD Training Verification):** Every evaluated `GLCAOMWorldModel` checkpoint must log its complete AdamW training trajectory on the `train` split (`>= 3,000` gradient updates across environments on L4 GPU), satisfy $\|\theta_{\text{trained}} - \theta_{\text{init}}\|_2 \ge 5.0$, and reduce held-out $B=0$ validation KL / NLL by $\ge 15\%$ relative to both its random initialization $\theta_{\text{init}}$ and the uniform prior $1/K$.
- **`R4` (Cached vs. Uncached KV Equivalence Check):** On 32 test boards, `TreePagedKVSearch` incremental KV-bank updates must match full sequence re-evaluation within $\max |\Delta p| \le 10^{-5}$ in `float32` reference mode.

---

## 2. Falsifiable Empirical Decision Rules (`R11-D1` through `R11-D6`)

All six empirical rules below are evaluated **strictly on the blind held-out `test` split (`R11-D0` verified)** across all trained seeds. Whenever a metric fails to meet its pre-registered threshold, the verdict **must** be reported verbatim as **`FAIL`** (or **`INCONCLUSIVE`** if the 95% CI straddles zero/threshold) — never patched with a surrogate.

### 2.1 Rule `R11-D1` — End-to-End Trained Universal KV-Cache Tree Machine (`D1`)
Evaluated on `sudoku_9x9_extreme`, `sudoku_6x6`, `equiv_sat`, and `shiphunt_8x8` across $B \in \{0, 1, 2, 4, 8, 16, 32, 64, 128\}$ using the SGD-trained `GLCAOMWorldModel` + live `TreePagedKVSearch`:
1. **Trained $B=0$ Root Competence:** On every environment, the SGD-trained root belief $q_{\text{real}}(V \mid \mathcal{C}_0)$ (`B=0`) achieves $\ge 15\%$ lower test $\mathrm{KL}(P^* \Vert \mu_0)$ (and lower normalized Brier loss) than the untrained initialization $\theta_{\text{init}}$, with paired 95% CI upper bound $< 0$.
2. **Blind Test-Time Search Gain ($B=128$ vs. $B=0$):** Live `GLC_Tree_Active_PUCT` search at $B=128$ reduces test belief KL (and Brier loss) relative to $B=0$ by $\ge 12\%$ across the deductive/inference suite, and increases verified combinatorial witness discovery rate on `sudoku_9x9_extreme` / `sudoku_6x6` by $\ge +8.0$ percentage points over $B=0$ (95% CI lower bound $> 0$).
3. **Tree Search Beats Flat Sampling at Matched Budget ($B=32, 128$):** At matched WTE compute ($B=32$ and $B=128$), `GLC_Tree_Active_PUCT` achieves strictly lower test belief KL and higher witness rate than flat uniform rollouts (`flat_mix` / `P3_uniform`) and legacy greedy-locked beam (`P8_lead_hybrid` with $m_{\text{qs}} \ge 0.50$), with paired 95% CI excluding zero.
4. **Live Leaf Self-Consistency $m_{\text{ql}}$ vs. Chain-Time $m_{\text{qs}}$ (Measured from Real `TreePagedKVBank`):** Extracted directly from `bank.m_ql_bank` and `bank.m_qs_bank` during live `run_search` on `sudoku_9x9_extreme` (zero `rng.beta` draws), $\mathrm{AUROC}(m_{\text{ql}}) \ge 0.85$ and $\mathrm{AUROC}(m_{\text{ql}}) - \mathrm{AUROC}(m_{\text{qs}}) \ge +0.15$ for discriminating verified-consistent vs. clashing/starved candidate leaves.

### 2.2 Rule `R11-D2` — Active Fractal Prospectus Plans & Leave-Subtree-Out (`LSO`) Credit (`D2`)
Evaluated on `shiphunt_8x8`, `sudoku_9x9_extreme`, and `equiv_sat` using the trained `MoPSFractalPlanHead` and `TreePagedKVSearch`:
1. **Non-Degenerate Trained Tree Topology:** At $B=128$, live tree selections on the **trained** trunk satisfy $f_{\text{root}}(128) \in [0.02, 0.45]$ and $f_{\text{tip}}(128) \in [0.20, 0.88]$ with fork fraction $f_{\text{fork}}(128) \ge 0.02$ and zero hardcoded depth/wave/stall thresholds.
2. **Beneficial Causal Plan-Swap & Ablation:**
   - Swapping the active prospectus plan $p_r \leftrightarrow p_r'$ between orthogonal sector/query modes flips the selected search step variable on $\ge 25\%$ of test states;
   - More importantly (**outcome benefit**, `GPT-6-Astra §5 R11-D2` & `Opus §5 R11-D2.c`), active Gram-orthogonalized plans (`GLC_Tree_Active_PUCT`) achieve strictly higher 24-sector coverage at $B=24$ on `shiphunt_8x8` (or higher endpoint hit rate on `equiv_sat`) and strictly lower test belief KL / Brier loss at $B=32$ than the no-plan ablation (`p_r = 0`) and randomly swapped plans (`95% CI < 0`).
3. **Leave-Subtree-Out (`LSO`) Innovation Prediction:** On `cal_val`, predicted branch priority correlates positively with realized Leave-Subtree-Out proper-score gain $I_{i,s} = S(q_i^{-s}, y_i) - S(q_i^{\text{full}}, y_i)$ (top quintile mean LSO gain $> 0$ with 95% CI lower bound $> 0$).

### 2.3 Rule `R11-D3` — Bidirectional Latent Lemma Space ($Z^W_{\text{aux}} \oplus Z^\Pi_{\text{aux}}$) & Combinatorial Queries (`D3`)
Evaluated on `equiv_sat` and `sudoku_6x6` across $B \in \{0, 2, 4, 8, 16, 32\}$ after end-to-end SGD training of `GLCAOMWorldModel`, `BilinearAuxLemmaEmitter`, and `CombinatorialQueryHead` with VICReg regularization on $Z^W_{\text{aux}}$:
1. **Train-Time & Post-Train Exact Firewall (`A-EXACT` Preserved Under SGD):** On the fully SGD-trained checkpoints:
   - $\max |\mu_0(V) - q_{\text{real}}(V \mid \mathcal{C}_0)| = 0.000000$;
   - $\|\partial \ell_V / \partial Z^\Pi_{\text{aux}}\|_\infty = 0.000000$;
   - Even-parity error $\max |\mathrm{emit}(+\Delta \ell_V) - \mathrm{emit}(-\Delta \ell_V)| \le 10^{-6}$;
   - Post-RL belief drift after 300 policy-gradient steps on `get_policy_parameters()` is $\Delta_{\text{belief}} = 0.000000$.
2. **Real Rollout `Z^W_aux` Spectrum (Zero Synthetic `iso_reg` Blending):** Computed directly on the cleared $Z^W_{\text{aux}}$ activations emitted by live test-time search rollouts at $B=16$ (with **zero** additive synthetic Gaussian `iso_reg`), the VICReg-trained partitioned arm achieves Participation Ratio $\mathrm{PR}(Z^W_{\text{aux}}) \ge 0.45$ (and stable rank $\ge 50\%$ of initialization) with dead-slot fraction $\le 15\%$, exceeding the unregularized/unpartitioned ablation by $\ge +0.20$.
3. **Blind Bidirectional Lemma Transfer & Direct Query Calibration:** Measured from live neural rollouts on `batch_blind` at $B=16$ (with **zero** `(1 - ls)*0.5 + ls*true_q` oracle interpolation):
   - Full `A5_GLC_Bilinear_QuerySteered` reduces relational query Brier loss $\mathrm{Brier}_Q(16)$ and bidirectional conditional $\mathrm{KL}_{\text{bidir}}(16)$ by $\ge 15\%$ relative to `A1_Marginal_Only` ($Z_{\text{aux}} = 0$) and strictly beats `A2_Linear_Aux` (odd linear emitter) and `A4_GLC_Bilinear_RandomRollout` (`95% CI < 0`);
   - Dual-readout self-consistency discrepancy $\delta_{\text{rel}}(A, B) = |P_{\text{direct}}(A=B) - P_{\text{AR}}(A=B)|$ computed directly from the neural heads achieves $\mathrm{AUROC} \ge 0.75$ for predicting high-error query reads ($|P_{\text{direct}} - Q^*| > 0.25$).

### 2.4 Rule `R11-D4` — Present-Time Episodic Memory ($M_t$), Trust Value Pooling & Belief–Policy Separation (`D4`)
Evaluated on `micro_stratego_6x6`, `Galanis-Market` (all 4 structures), and `Proposal-Poker` using SGD-trained checkpoints and real multi-turn episodes:
1. **Depth-Debiased `trust_shrunk` Value Pooling on Real Rollout Bags:** On mixed-depth ($d \in \{1, 2, 4, 6\}$) rollout bags generated by the trained model on `micro_stratego_6x6`, `DepthDebiasedTrustHead.pool_values_trust_shrunk` (`T5`) achieves strictly lower value MSE against $Q^*(s, a)$ than stiffness pooling (`T1_Stiff_Pool`, $\ge 25\%$ MSE reduction) and uniform pooling (`T2_Unif_Pool`, paired 95% CI $< 0$), with Spearman $\rho(\beta_{\text{trust}}, -\text{MSE}) > 0 > \rho(\text{stiffness}, -\text{MSE})$.
2. **Real-Episode Present-Time Memory $M_t$ & Surprise Contraction:** On multi-turn `micro_stratego_6x6` trajectories where $M_t = h_{\text{cons}}^{(t-1)}$ is transported from the model's own prior-turn consensus (zero `torch.log(p_exact_test) + 0.36 * hist_shift` surrogate):
   - Enabling $M_t$ reduces turn $4\text{--}12$ belief KL by $\ge 8\%$ relative to the memoryless (`M_t = 0`) baseline (`95% CI < 0`), with larger gains in the top $G_{\text{hist}}$ tercile than the bottom tercile;
   - When contradicting opponent observations are encountered (or injected from a mismatched episode at turn $t=6$), the `EpisodicMemoryTransport` surprise discount reduces KL blow-up by $\ge 80\%$ relative to an ungated memory baseline ($\lambda_{\text{surp}} = 0$), with multi-turn contraction ratio $\hat{\kappa} \le 0.75$.
3. **Trained `Galanis-Market` & `Proposal-Poker` Separation (`LIR-Masked` vs. `Token-Masked` vs. `Fused`):**
   - Across all 4 `Galanis-Market` structures, live search at $B=16$ reduces posterior KL below $B=0$ (`95% CI < 0`) and preserves the separation between agreement (`t3s111y2` inter-agent gap $\le 0.04$) and persistent rational disagreement (`t3s111o2ye2` gap within $\pm 0.06$ of $1/3$);
   - On `Proposal-Poker`, `LIR-Masked` maintains exact $D_{\text{bias}}(B=0) = 0.000000$, post-RL $\Delta_{\text{belief}} = 0.000000$, and zero-shot utility-swap action flip rate $\ge 0.90$, while strictly beating SGD-trained `Fused` and `Token-Masked` baselines on belief KL under utility skew (`95% CI < 0`).

### 2.5 Rule `R11-D5` — Certified Anytime Monotonicity, Two-Stage Clearing & Non-Circular Scaling Law (`D5`)
Evaluated on live neural search trajectories across $B \in \{0, 1, 2, 4, 8, 16, 32, 64, 128\}$:
1. **Two-Stage Disjunctive-then-Conjunctive Clearing on Learned Branches:**
   - On `micro_stratego_6x6` case-splits generated by the trained model, `TwoStageTreeClearer` (with Stage-1 Disjunctive Bayes Marginalization) reduces the piece-count exclusion veto rate by $\ge 70\%$ relative to conjunctive-only log-odds pooling (`C3/C5`) and achieves strictly lower test KL (`95% CI < 0`);
   - Every executed Stage-2 preconditioned clearing operator satisfies $\|T_b\|_\infty < 1.0$, and small-sample calibration (`calibrate_soft_teacher` on $N_{\text{cal}} \in \{500, 2000\}$ learned branch shifts) achieves generalization gap $|\mathrm{KL}_{\text{test}} - \mathrm{KL}_{\text{val}}| \le 0.015$ nats.
2. **Certified Anytime Monotonicity & Non-Vacuous Gain Capture ($\mathrm{GCR}(128)$):**
   - Across all $(e, B \to B')$ budget transitions on the 15 environments, $\ge 90\%$ of transitions satisfy $\Delta\mathrm{KL} \le +0.002$ nats (with 0 catastrophic episode explosions $\mathrm{KL}(128) > 2\,\mathrm{KL}(0)$);
   - At strong roots ($M_0 \in \{32, 34\}$ on `shiphunt_6x6`) as well as moderate roots ($M_0 \in \{16, 24\}$), the calibrated certificate gate $\alpha_b$ prevents strong-root degradation ($\Delta\mathrm{KL}(128 - 0) \le +0.001$ at $M_0 \in \{32, 34\}$) while achieving non-vacuous Gain Capture Ratio $\mathrm{GCR}(128) \ge 0.50$ at $M_0 \in \{16, 24\}$ with abstention rate $P(\alpha_{128} = 0) \le 30\%$.
3. **Out-of-Sample Non-Circular Mechanistic Scaling Law & Prediction P9 Test:**
   - Fit (F1) the Mechanistic Redundancy Law $\Delta c(B) = \alpha \log(1 + \frac{B}{1 + \rho(B-1)})$, (F2) a 2-parameter log law $\alpha \log(1 + B)$, and (F3) a saturating exponential $c_\infty (1 - e^{-B/\tau})$ on **measured** neural search gains at fit budgets $B \in \{0, 1, 2, 4, 8, 16, 32\}$, and evaluate out-of-sample prediction error on held-out budgets $B \in \{64, 128\}$;
   - Compute physical Gram crowding $\bar{\rho}_{\text{Gram}}(A, e)$ at $B=32$ using a **single, algorithm-blind functional** on raw `bank.edge_keys_bank` (Rule `S5`), and test Prediction P9 ($\mathrm{corr}(\hat{\rho}_{\text{IRT}}, \bar{\rho}_{\text{Gram}}) \ge 0.50$, 95% CI $> 0$). If the empirical correlation falls below $0.50$, report P9 as **empirically refuted / qualified** under neural weights.

### 2.6 Rule `R11-D6` — Interactive Multiplayer Elo/`mElo` League, Generalist Transfer & Bifactor `Mini-ECI` (`D6`)
Evaluated across all 15 environment configurations (8 families) and all 7 core search families + 5 cohorts:
1. **100% Interactive Turn-by-Turn Multiplayer Tournaments (`Zero btd_probabilities Simulation`):**
   - Every match in `Hex-5x5`, `Connect-4`, `Micro-Stratego`, `Galanis-Market`, and `Proposal-Poker` is played turn-by-turn by the competing agents invoking their trained models and search engines on the live game state until terminal payoff, using 2-game seat-mirrored duplicate deals;
   - At $B=64$, `GLC-AOM-KV-Tree` achieves mean 5-environment Elo strictly above `1-pass@B=0` (`1000.0`) by $\ge +120.0$ Elo (95% CI $> +80.0$) and beats `Flat-Econ-Prec-K3@B=64` by $\ge +35.0$ Elo (95% CI $> 0.0$);
   - Both `BayesianEloLeague` (BTD) and antisymmetric `mElo` (with cyclic variance share $\Omega_{\text{cyc}, e}$ and Nash mixture $p^*_e$) are evaluated for held-out match log-loss improvement over a seat-only baseline.
2. **Compute-Matched `Unified-Generalist` vs. `Single-Game-Specialist` (`GEN` vs. `SPEC-TOT` / `SPEC-PER`):**
   - At matched total GPU training compute (`SPEC-TOT`), `Unified-Generalist` is non-inferior (or superior) to the specialist mean within $[-3.5, +\infty)$ `Mini-ECI` points across the 8 families, and outperforms specialists on zero-shot OOD header/rule perturbations by $\ge +4.0$ `Mini-ECI` points (95% CI $> 0$).
3. **Empirical Bifactor `Mini-ECI` Psychometric Fit:**
   - Fitted exclusively to the genuinely observed neural episode likelihoods, verified witnesses, and interactive match outcomes across all `(family, B)` population cells, the 3-factor orthogonal Bifactor IRT model (`Ded`, `Inf`, `Str`) and 1D `MiniECIFitter` achieve leave-one-task-out correlation $\mathrm{LOTO}\;r \ge 0.80$ and cross-channel Spearman $r_S \ge 0.85$, with `GLC-AOM-KV-Tree` leading `Flat-Econ-Prec-K3` and `P3/P8-Heuristic-Beam` at $B=32$ and $B=128$ across all 3 domain sub-scores.
