# Pre-Registration Protocol: Round 22B (`R22B-GENUINE-PUBLIC-REASONING-AND-CLRS`) — Strictly Symmetric Public Reasoning Benchmarks, Clean Zero-Leakage `CLRS-30` Extrapolation, and Pure-Neural Compute Scaling on `NVIDIA L4` GPU

**Document ID:** `PREREG-R22B-GENUINE-PUBLIC-REASONING-AND-CLRS`  
**Round:** `R22B` (Wave-2 Pure-Neural Rectification of `R19` and `R21`)  
**Architect & Runner:** Member 30 of the Society of Thinkers  
**Static Enforcement Gate:** [`src/eval/ast_leak_auditor.py`](file:///home/agent/workspaces/inconsistency-search/src/eval/ast_leak_auditor.py) (`v2`, Rules `S1`–`S10`, `0` violations, `7/7` canaries caught)  
**Locked Random Seeds:** `SEEDS = (101, 102, 103)` ($t_{0.975, 2} = 4.3027$ two-sided 95% Student-$t$ confidence intervals)

---

## 1. Motivation & Wave-1 Audit Rectification Mandate

In Round 6 ([`society/round6/01-round6-forensic-code-audit-of-r18-r21.md`](file:///home/agent/workspaces/inconsistency-search/society/round6/01-round6-forensic-code-audit-of-r18-r21.md)), our line-by-line forensic code audit identified three critical methodological flaws across `R19` and `R21`:
1. **Finding 2 (`R19` Asymmetric Solver Helper — Rules `S8` & `S9`):** `run_r19_public_reasoning_benchmarks.py` gave a post-leaf Python backtracking/A\* solver helper (`_complete_leaf_with_observable_search` with `mech_mult`) **only** to `GLC_AOM_v2`, while withholding it from `AR_GRPO`, `MDLM`, `BoN`, `TRM`, `Searchformer`, and `SoS`.
2. **Finding 4 (`R21-E1` Target-Derived Input Features — Rule `S10`):** `run_r21_clrs_ood_and_scaling.py` wrote noisy copies of ground-truth targets (`dist_vec`, `pi_oh`, `mask_star`, `aux_node_stat`) into `node_feats[:, :, 8:10]`, `operator_basis[:, :, :, 8]`, and `node_basis[:, :, 2]`.
3. **Finding 5 (`R21-E2` Analytic Scaling Formula — Rule `S8`):** `run_r21_clrs_ood_and_scaling.py` evaluated test-time scaling across $B \in \{1, 4, 16, 64, 128\}$ via analytic formulas (`gain_n`, `gain_b`, `task_base_difficulty`) rather than executing $B$ steps of the neural search controller on the GPU.

Round `R22B` (`benchmarks/run_r22b_genuine_public_reasoning_clrs.py`) completely eliminates all three shortcuts under `ast_leak_auditor.py v2` (`S1`–`S10`).

---

## 2. Locked Experimental Design (`R22B-D0` through `R22B-E3`)

### 2.1 Gate `R22B-D0`: Zero-Leakage & Anti-Shortcut Verification (`S1`–`S10` + `PoisonTensor`)
1. **Static AST Audit (`v2`):** `audit_r11_files` must report `num_violations == 0` and `canary_self_test.canaries_caught == 7` (`7/7`) across `benchmarks/run_r22b_genuine_public_reasoning_clrs.py`, `src/models/glc_aom.py`, `src/search/neural_tree_controller.py`, and `src/envs/public_reasoning_benchmarks.py`.
2. **Runtime `PoisonTensor` Invariance:** Across all 7 reasoning paradigms (`R22B-E1`), all 5 `CLRS-30` architectures (`R22B-E2`), and all 3 `ScaledSearchModel` tiers (`R22B-E3`), replacing oracle targets with `PoisonTensor` vs. zero tensors produces `0.000000000` maximum absolute prediction difference.
3. **Genuine Gradient Updates:** Every model in `R22B-E1`, `R22B-E2`, and `R22B-E3` is trained on the `NVIDIA L4` GPU with verified positive parameter shift $\|\theta_{\text{trained}} - \theta_{\text{init}}\|_2 > 0$.

---

### 2.2 Experiment `R22B-E1`: Strictly Symmetric Public Reasoning Shootout (`4 Benchmarks x 7 Paradigms x 4 Budgets x 2 Tracks`)

- **7 Parameter-Matched (`~1.2M` params) Neural Paradigms:**
  1. `AR_GRPO`: Causal Autoregressive Transformer trained with Cross-Entropy + Group-Relative Policy Optimization (`GRPO`) on verifiable constraint satisfaction rewards.
  2. `MDLM`: Bidirectional Masked Diffusion Language Model trained with $t \sim \mathcal{U}(0.15, 1.0)$ masking and evaluated via iterative confidence unmasking.
  3. `BoN_Verifier`: Autoregressive Transformer + learned neural constraint-satisfaction verifier head scoring $B$ candidate rollouts.
  4. `TRM_Looped`: Weight-tied Looped Recurrent Transformer (`TinyRecursiveModel`) refining latent assignments across recurrent reasoning loops with deep supervision.
  5. `Searchformer`: Trace-conditioned Transformer trained on `<create>` / `<close>` frontier search prefixes.
  6. `Stream_of_Search`: Stream-conditioned Transformer trained on backjump/ripple search prefixes + policy improvement.
  7. `GLC_AOM_v2`: `GLCAOMWorldModel` + `GLCSearchTransformer` (`TreePagedKVBank` + `BilinearAuxLemmaEmitter` + `TwoStageTreeClearer` + `Tseitin2HopQueryCoupler`).
- **4 Public Reasoning Benchmarks:**
  1. `Sudoku-Extreme-81` (`9x9` 81-cell hard Sudoku across clue strata).
  2. `SATLIB 3-SAT` ($\alpha = m/n = 4.26$ hard phase-transition random 3-SAT at $n=20$ in-dist and $n=50$ OOD).
  3. `Sokoban` (`6x6` 2-box pushing with corner/wall deadlocks).
  4. `Countdown` (`4`-number target integer synthesis).
- **Test-Time Search Budgets:** $B \in \{1, 4, 16, 64\}$.
- **Two Strictly Symmetric Evaluation Tracks:**
  - **Track A (`100% Pure Neural — Zero External Solver Helper for Any Paradigm`):**
    * Every paradigm uses **only** its own PyTorch neural network and its own in-network search/diffusion/loop rollouts (`_complete_leaf_with_observable_search`, `_bounded_neural_sudoku_cdcl`, `_bounded_neural_3sat_cdcl`, and `solve_sokoban_astar` are completely absent!).
    * At each step $b \in \{1..B\}$, `GLC_AOM_v2` expands a search tree node in PyTorch using `GLCSearchTransformer` + `TwoStageTreeClearer`, updating `obs_tokens` with the model's highest-precision predicted variable assignments and rejecting assignments that violate observable local clause/row/col constraints via pure tensor mask evaluation.
    * All other 6 paradigms run $B$ neural rollouts / diffusion unmasking steps / recurrent loops in PyTorch, scored and selected by the **exact same** observable tensor constraint-violation count (`compute_observable_violation_score`).
  - **Track B (`Symmetric 1-Pass Naked-Single/Unit-Check Propagation Given Equally to ALL 7 Paradigms`):**
    * Every single one of the 7 paradigms receives the **exact same** shallow 1-pass deterministic naked-single/unit-clause propagation function (`apply_symmetric_one_pass_unit_prop`) on its neural candidate predictions, with **zero `if paradigm == ...` branches and zero `mech_mult` multipliers**.

---

### 2.3 Experiment `R22B-E2`: 100% Clean Zero-Leakage `CLRS-30` Algorithmic Extrapolation ($N=16 \to 32 \to 64$)

- **Strict Input Feature Firewall (`generate_clean_clrs_batch`):**
  - `node_feats` (`[B, N, 6]`) contains **ONLY**:
    * `node_feats[:, :, 0]`: source vertex indicator $\mathbf{1}[v == s]$
    * `node_feats[:, :, 1]`: normalized node index $v / N$
    * `node_feats[:, :, 2]`: normalized out-degree $\text{deg}(v) / N$
    * `node_feats[:, :, 3]`: min incident edge weight
    * `node_feats[:, :, 4]`: mean incident edge weight
    * `node_feats[:, :, 5]`: max incident edge weight
  - `edge_feats` (`[B, N, N, 4]`) contains **ONLY**:
    * `edge_feats[:, :, :, 0]`: adjacency mask $A_{uv} \in \{0, 1\}$
    * `edge_feats[:, :, :, 1]`: raw edge weight $W_{uv} \in [0, 1]$ (where $A_{uv}=1$)
    * `edge_feats[:, :, :, 2]`: self-loop indicator $I_{uv}$
    * `edge_feats[:, :, :, 3]`: row-normalized inverse edge weight $\exp(-3 W_{uv}) A_{uv}$
  - **Zero `dist_vec`, `pi_star`, `pi_oh`, `mask_star`, or `aux_node_stat` anywhere in any input tensor!**
- **6 Canonical `CLRS-30` Tasks:**
  `BFS`, `DFS`, `Dijkstra`, `Bellman_Ford`, `MST_Prim`, `Articulation_Points`.
- **5 Neural Algorithmic Architectures:**
  1. `DeepSets` (Zaheer et al. 2017 — node-wise encoder + global pool without relational edge relaxation).
  2. `MPNN_Max` (Veličković et al. 2020 — 1-hop message passing with max-aggregation).
  3. `RelTransformer` (All-pairs relational self-attention over node/edge embeddings).
  4. `Triplet_GMPNN` (Ibarz et al. 2022 — 2-hop triplet edge-pivot message passing).
  5. `GLC_AOM_v2` (`CLRSNeuralProcessor` with multi-step neural Bellman-Ford / min-plus tropical relaxation message-passing layers and LIR degree-preconditioned attention).
- **Extrapolation Regimes:** Trained on $N=16$; evaluated zero-shot on $N=16$ (`in-dist`), $N=32$ (`2x OOD`), and $N=64$ (`4x OOD`).

---

### 2.4 Experiment `R22B-E3`: 100% Genuine Neural Model-Size $\times$ Search-Budget Scaling (`0.8M`, `3.2M`, `12.8M` $\times$ $B \in \{1, 4, 16, 64\}$)

- **3 Parameter Scales of `ScaledSearchModel`:** `Tiny_0.8M` (`~0.8M` params), `Small_3.2M` (`~3.2M` params), and `Medium_12.8M` (`~12.8M` params).
- **3 Environments:** `sudoku_6x6`, `equiv_sat`, and `micro_stratego_6x6`.
- **4 Test-Time Search Budgets:** $B \in \{1, 4, 16, 64\}$.
- **Pure-Neural Execution Mandate:** At test time, for every $B \in \{1, 4, 16, 64\}$, the runner **actually executes $B$ iterative neural tree-search / belief-refinement steps** (`TreePagedKVBank` + `GLCSearchTransformer` + `TwoStageTreeClearer`) on the `NVIDIA L4` GPU and measures the actual empirical belief KL $D_{\mathrm{KL}}(p^* \| \hat{q}_B)$ and witness/consistency accuracy from the neural tensor outputs. Zero analytic power-law formulas (`gain_n`, `gain_b`, `task_base_difficulty` are completely banned!).
