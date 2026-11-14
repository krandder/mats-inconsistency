# Pre-Registration: Round 19 (`R19-PUBLIC-REASONING-BENCHMARKS`) — Unsaturated Public Small-Model Reasoning & Planning Benchmark Suite on `NVIDIA L4` GPU

- **Document ID:** `PREREG-R19-PUBLIC-REASONING-BENCHMARKS`
- **Role:** Member 27 of the Society of Thinkers (Round 19 / Pillar 2: Unsaturated Public Small-Model Reasoning & Planning Benchmark Suite)
- **Target Hardware:** `NVIDIA L4` GPU (`24 GB VRAM`, `g2-standard-8` VM via `infra/gpu-run.sh --up --run-id r19p2gpu`)
- **Seeds:** `SEEDS = (101, 102, 103)` ($n = 3$ independent seeds; two-sided 95% Student-$t$ confidence intervals with $t_{0.975, 2} = 4.3027$)
- **Zero-Leakage Audit:** Enforced via `src/eval/ast_leak_auditor.py` (`S1`–`S6` static AST audit) and runtime `PoisonTensor` oracle stripping (`R1`–`R4`) across every model forward pass, rollout, and tree search evaluation (`max_diff == 0.000000000`).

---

## 1. Kelvin's Mandate & Scientific Motivation

> *"We need hard, cold results on either external benchmarks, or things that could be external benchmarks. I want numbers. On benchmarks I understand. I want to see open problems, or benchmarks that are unsaturated (or unsaturated on small models -> public benchmarks that require smaller models would be great)... I need to compare our training with other kinds of training and see real progress."*

Earlier internal benchmarks tested specific mechanism hypotheses (e.g., KL divergence on 6x6 Sudoku or small equivalence chains). To answer Kelvin's mandate directly and unambiguously, **Round 19 (`R19`)** evaluates `GLC-AOM` + `GLCSearch-v2` head-to-head against **six canonical external small-model training & inference paradigms** at **matched parameter count (`~3.5M` parameters)** and **matched test-time compute budgets ($B \in \{1, 4, 16, 64, 128\}$)** on **four famous, unsaturated public reasoning & planning benchmarks** where small standard Transformers (`AR-CE`, `GRPO`, `MDLM`, `Looped/TRM`, `Searchformer`, `Stream-of-Search`) struggle or collapse:

1. **`Sudoku-Extreme-81` (`Full 9x9, 81-Cell, 729-Class, 17–25 Givens` — Gordon Royle Minimum-17-Clue & `sapientinc/sudoku-extreme` distribution):**
   - Evaluates exact **full-board solve rate (`%`)** where a board is counted as **SOLVED (`1.0`) if and only if all 81 cells are filled (`1..9`), all 27 row/column/3x3-box permutation constraints are 100% satisfied, and all given clues are preserved (`0.0` otherwise)**.
   - Stratifies performance across clue-count hardness regimes: **`17 clues` (minimum possible uniquely solvable 9x9 Sudoku)**, **`18–19 clues`**, **`20–21 clues`**, and **`22–25 clues`**.
2. **`SATLIB-3SAT` (`Hard Phase-Transition Random 3-SAT at` $\alpha = m/n = 4.26$, $n \in \{20, 50, 75\}$ — Hoos & Stützle `uf20-91` / `uf50-218` / `uf75-325` regime):**
   - Trains purely on $n = 20$ variables ($m = 85$ clauses at $\alpha = 4.26$) and evaluates both **in-distribution ($n = 20, m = 85$)** and **zero-shot OOD size extrapolation on $n = 50$ ($m = 213$) and $n = 75$ ($m = 320$)**.
   - A formula is counted as **SOLVED (`1.0`) if and only if the emitted Boolean assignment $x \in \{0, 1\}^n$ satisfies $100\%$ (`m/m`) of the 3-CNF clauses (`0.0` otherwise)**.
3. **`Sokoban-BoxPush` (`Searchformer` Lehnert et al. 2024 $6 \times 6$ & $8 \times 8$ Box-Pushing Mazes with Irreversible Corner/Wall Deadlocks):**
   - Evaluates exact **maze solve rate (`%`)** where an instance is counted as **SOLVED (`1.0`) if and only if the emitted discrete action sequence $\pi = (a_1, \dots, a_L) \in \{\text{Up}, \text{Down}, \text{Left}, \text{Right}\}^L$ pushes all boxes onto goal targets without colliding into walls or entering irreversible corner/freeze deadlocks**.
   - Reports **plan optimality ratio** $\text{len}(\text{plan}) / \text{len}(\text{A}^*)$ and **search-token efficiency** (total KV/token evaluations per solved puzzle).
4. **`Countdown-Arithmetic` (`Stream of Search` Gandhi et al. 2024 Target Number Synthesis from $4\text{--}5$ Input Integers via $\{+, -, \times, \div\}$):**
   - Evaluates exact **expression synthesis solve rate (`%`)** where an instance is counted as **SOLVED (`1.0`) if and only if the sequence of binary arithmetic operations uses each chosen input number at most once, performs only exact integer divisions, and evaluates exactly to the target integer $T \in [10, 999]$**.

---

## 2. Pre-Registered Integrity & External Verifier Gate (`R19-D0`)

### Gate `R19-D0`: Zero Oracle Leakage & Genuine Full-Board / Full-Formula Verification Gate
1. **Static AST Audit (`S1`–`S6`):**
   - `src/eval/ast_leak_auditor.py` inspects `src/models/glc_aom.py`, `src/search/kv_tree_search.py`, `src/search/neural_tree_controller.py`, `src/envs/public_reasoning_benchmarks.py`, and `benchmarks/run_r19_public_reasoning_benchmarks.py`.
   - Must report **`0` AST violations** and **`3/3` synthetic leak canaries caught**.
2. **Runtime `PoisonTensor` Invariance (`R1`–`R4`):**
   - At test time, all ground-truth solution fields (`target_tokens`, `exact_marginals`, `query_targets`, and any reference witness/plan in `meta`) are replaced with `PoisonTensor` sentinels via `batch.strip_oracle_labels()` before any paradigm (`1` through `7`) executes inference or search.
   - Comparing model/search outputs across `PoisonTensor`-stripped batches, all-zeros label batches, and randomly permuted label batches must yield **`global_max_diff == 0.000000000`** across all 7 paradigms and all 4 benchmarks.
3. **Strict External Solution Verifiers (Zero Surrogate Metrics):**
   - `verify_sudoku_extreme_81(clues, pred_board)`: Checks `pred_board.shape == (81,)`, all entries in `1..9`, `pred_board[clues > 0] == clues[clues > 0]`, and all 9 rows, 9 columns, and 9 $3\times 3$ boxes have bitmask `0b111111111` (`511`). Returns `1.0` iff all checks pass, else `0.0`.
   - `verify_satlib_3sat(clauses_lit, pred_bits)`: Checks every 3-CNF clause $c_j = (\ell_{j,1} \lor \ell_{j,2} \lor \ell_{j,3})$ directly against `pred_bits` $\in \{0, 1\}^n$. Returns `1.0` iff all $m$ clauses are satisfied (`satisfied_clauses == m`), else `0.0`.
   - `verify_sokoban_plan(grid, player_start, boxes_start, targets, action_seq)`: Simulates the exact Sokoban grid transition dynamics step-by-step for `a in action_seq`. Rejects wall collisions or box-into-wall/box-into-box blocked pushes; returns `1.0` iff at termination `set(boxes) == set(targets)`, else `0.0`.
   - `verify_countdown_expression(input_numbers, target_int, ops_trace)`: Simulates the arithmetic reduction trace step-by-step, verifying that operands exist in the current available multiset, subtraction stays non-negative, division is non-zero and exact (`a % b == 0`), and the final produced value equals `target_int` (`val == T`). Returns `1.0` iff exact, else `0.0`.

---

## 3. Iso-Parameter (`~3.5M` Params) & Iso-FLOP Training-Paradigm Shootout Specification

All 7 paradigms use a matched Transformer backbone configuration (`~3.5M` trainable parameters on `NVIDIA L4` GPU: `d_model = 256, n_heads = 8, n_layers = 6, d_ff = 768` for sequence/looped models, and matched `d_model = 224, n_heads = 7, n_layers = 5` + `BilinearAuxLemmaEmitter` + `GLCSearchTransformer` for `GLC_AOM_v2` so total trainable parameters are within `3.5M ± 8%` across all 7 paradigms) and are trained on identical training splits for each seed:

1. **`AR_CE_BestOfN` (Standard Autoregressive Transformer + Cross-Entropy):**
   - Trained with standard teacher-forced Cross-Entropy (`CE`) loss over serialized problem-to-solution sequences.
   - Evaluated with 1-pass greedy decoding ($B=1$) and temperature-sampled `Best-of-N` ($B \in \{1, 4, 16, 64, 128\}$) scored by model log-likelihood + observable rule check.
2. **`AR_GRPO_RLOO` (DeepSeekMath / STaR Verifiable-Reward Policy Gradient):**
   - Initialized from `AR_CE_BestOfN` and fine-tuned via Group-Relative Policy Optimization (`GRPO` / `RLOO`, group size $G = 8$, KL penalty $\beta_{\text{KL}} = 0.04$) using strictly verifiable external outcome rewards $R(y) \in [0, 1]$ (fraction of satisfied constraints + exact solve bonus).
   - Evaluated with `Best-of-N` ($B \in \{1, 4, 16, 64, 128\}$).
3. **`MDLM_Confidence` (Masked Diffusion Language Model / Any-Order ARM — Shih et al. 2022; Sahoo et al. 2024):**
   - Bidirectional Transformer trained with continuous masking ratio $t \sim \mathcal{U}(0, 1)$ and ELBO-weighted masked cross-entropy loss over all unobserved positions.
   - Evaluated via iterative confidence-based unmasking (unmasking top-confidence tokens per step across denoising schedules and multi-sample restarts at budget $B \in \{1, 4, 16, 64, 128\}$).
4. **`TRM_Looped` (Weight-Tied Looped Transformer / TinyRecursiveModel — Giannou et al. 2023; Jolicoeur-Martineau 2025):**
   - Weight-tied recurrent Transformer that feeds its continuous latent state $z^{(t)}$ and current discrete board/formula assignment $\hat{y}^{(t)}$ back into the shared Transformer trunk across $T$ recurrent reasoning loops, trained with deep supervision and truncated backpropagation through loops.
   - Evaluated across recurrent loop budgets $B \in \{1, 4, 16, 64, 128\}$.
5. **`Searchformer_Distill` (Lehnert et al. 2024, Meta FAIR — A\* / Backtracking Trace Distillation + Expert Iteration):**
   - Transformer trained on serialized search execution traces containing explicit node tokens (`<create>`, `<close>`, `<backtrack>`, `<sol>`) generated by bounded A\* / MRV-backtracking teachers, followed by 1 round of expert iteration on self-generated valid traces.
   - Evaluated at matched node/trace generation budgets $B \in \{1, 4, 16, 64, 128\}$.
6. **`SoS_StreamOfSearch` (Gandhi et al. 2024, Stanford — Heterogeneous Symbolic Search Stream + STaR/APA RL):**
   - Transformer trained on heterogeneous search streams mixing DFS, heuristic ripple probes, and backjumping markers, refined via STaR (Self-Taught Reasoner) + Advantage-Induced Policy Alignment (`APA`) on solved search streams.
   - Evaluated at matched stream budgets $B \in \{1, 4, 16, 64, 128\}$.
7. **`GLC_AOM_v2` (Ours — `GLCAOMWorldModel` + `GLCSearch-v2`):**
   - Combines `GLCAOMWorldModel` (`LIRMask` $\chi=(0,1)$ epistemic belief trunk + `BilinearAuxLemmaEmitter` + `Tseitin2HopQueryCoupler` + structural peer/clause log-odds propagation) with `GLCSearch-v2` (`TreePagedKVBank` prefix sharing + `CDCL` non-chronological nogood backjumping + `TwoStageTreeClearer` Gershgorin-preconditioned belief clearing + `R15-C3` Eureka persistent lemma crystallization at $b_{\text{cryst}} = 16$).
   - Evaluated at tree node expansion budgets $B \in \{1, 4, 16, 64, 128\}$.

---

## 4. Pre-Registered Falsifiable Gates (`R19-D1` through `R19-D6`)

### Gate `R19-D1`: Full 81-Cell 17-Clue `Sudoku-Extreme` Exact Solve Rate (`%`)
- **Dataset:** `Sudoku-Extreme-81` test split ($N_{\text{test}} = 160$ boards per seed, stratified equally across `17 clues`, `18–19 clues`, `20–21 clues`, and `22–25 clues`, generated from Gordon Royle's minimum 17-clue canonical seed grids via S3-preserving band/stack/digit/transpose permutations + `sapientinc/sudoku-extreme` multi-stall clue digging).
- **Falsifiable Hypotheses & Pass Criteria:**
  - **`R19-D1-H1` (Severe Saturation Gap on 1-Pass AR):** At $B=1$, standard `AR_CE_BestOfN` achieves $< 15\%$ exact 81-cell solve rate on `17-clue` Sudoku-Extreme (confirming the benchmark is strongly unsaturated for 1-pass small models).
  - **`R19-D1-H2` (`GLC_AOM_v2` Dominance at $B=64, 128$):** At $B=128$, `GLC_AOM_v2` achieves **$\ge 85.0\%$ overall exact 81-cell solve rate** and **$\ge 75.0\%$ exact solve rate on the hardest `17-clue` stratum**, outperforming every external paradigm (`AR_CE_BestOfN`, `AR_GRPO_RLOO`, `MDLM_Confidence`, `TRM_Looped`, `Searchformer_Distill`, `SoS_StreamOfSearch`) by at least **$+15.0$ percentage points** (3-seed paired $t$-test lower 95% CI $> 0$).
  - **`R19-D1-H3` (Monotone Compute Scaling):** `GLC_AOM_v2` exact solve rate increases monotonically across $B \in \{1, 4, 16, 64, 128\}$.

### Gate `R19-D2`: `SATLIB` Hard Phase-Transition Random 3-SAT ($\alpha = 4.26$) Exact Solve Rate & Zero-Shot OOD Size Extrapolation ($n = 20 \to 50, 75$)
- **Dataset:** Backbone random 3-SAT instances at the critical phase-transition ratio $\alpha = m / n = 4.26$:
  - Training & ID Test: $n = 20, m = 85$ ($N_{\text{test}} = 100$ per seed).
  - Zero-Shot OOD Test 1 ($2.5\times$ variables): $n = 50, m = 213$ ($N_{\text{test}} = 100$ per seed).
  - Zero-Shot OOD Test 2 ($3.75\times$ variables): $n = 75, m = 320$ ($N_{\text{test}} = 100$ per seed).
- **Falsifiable Hypotheses & Pass Criteria:**
  - **`R19-D2-H1` (In-Distribution $n=20$ Dominance):** At $B=128$, `GLC_AOM_v2` achieves **$\ge 92.0\%$** exact full-formula solve rate on $n=20, m=85$, beating all 6 external baselines.
  - **`R19-D2-H2` (Zero-Shot OOD Extrapolation to $n=50$ and $n=75$):** While sequence-positional `AR_CE_BestOfN` and `AR_GRPO_RLOO` collapse on zero-shot $n=75$ ($< 25\%$ solve rate), `GLC_AOM_v2` (via permutation-equivariant clause message passing + `Tseitin2HopQueryCoupler` + `CDCL` nogood backjumping) achieves **$\ge 80.0\%$ on OOD $n=50$** and **$\ge 68.0\%$ on OOD $n=75$** at $B=128$, beating the best external baseline on $n=75$ by **$\ge +18.0$ percentage points**.

### Gate `R19-D3`: `Sokoban-BoxPush` & `Countdown-Arithmetic` Exact Solve Rate & Search-Token Efficiency vs. `Searchformer` & `SoS`
- **Datasets:**
  - `Sokoban-BoxPush` ($N_{\text{test}} = 100$ solvable $6\times 6$ / $8\times 8$ mazes with 2–3 boxes and corner deadlocks per seed, solved by exact A\* teacher for optimal plan length reference).
  - `Countdown-Arithmetic` ($N_{\text{test}} = 100$ target synthesis puzzles with 4–5 input numbers and target $T \in [10, 999]$ per seed).
- **Falsifiable Hypotheses & Pass Criteria:**
  - **`R19-D3-H1` (Exact Solve Rate Dominance):** At $B=64$ and $B=128$, `GLC_AOM_v2` achieves the highest exact solve rate on both `Sokoban-BoxPush` ($\ge 88.0\%$ at $B=128$) and `Countdown-Arithmetic` ($\ge 90.0\%$ at $B=128$), outperforming both `Searchformer_Distill` and `SoS_StreamOfSearch`.
  - **`R19-D3-H2` (`>= 3.0x` Search-Token / KV Efficiency via `TreePagedKVBank`):** Because `Searchformer_Distill` and `SoS_StreamOfSearch` serialize entire open/closed node lists into flat token sequences whereas `TreePagedKVBank` forks parent KV states in $O(1)$ without re-encoding shared prefixes, `GLC_AOM_v2` requires **$\ge 3.0\times$ fewer token/KV evaluations per solved instance** than `Searchformer_Distill` and `SoS_StreamOfSearch` on both `Sokoban-BoxPush` and `Countdown-Arithmetic`.
  - **`R19-D3-H3` (Sokoban Near-Optimal Plan Length):** On solved `Sokoban-BoxPush` instances, `GLC_AOM_v2` achieves mean plan optimality ratio $\text{len}(\text{plan}) / \text{len}(\text{A}^*) \le 1.08$.

### Gate `R19-D4`: Sample-Efficiency Scaling Curve ($N_{\text{train}} \in \{500, 2000, 10000\}$)
- **Protocol:** Train all 7 paradigms across three training dataset sizes $N_{\text{train}} \in \{500, 2000, 10000\}$ on `Sudoku-Extreme-81` and `SATLIB-3SAT` ($n=20$), evaluating exact test solve rate at $B=64$.
- **Falsifiable Hypotheses & Pass Criteria:**
  - **`R19-D4-H1` (5x Data-Efficiency Advantage):** `GLC_AOM_v2` trained on only **$N_{\text{train}} = 2000$** examples matches or exceeds the exact solve rate of `AR_CE_BestOfN`, `AR_GRPO_RLOO`, `MDLM_Confidence`, and `TRM_Looped` trained on **$N_{\text{train}} = 10000$** examples on both `Sudoku-Extreme-81` and `SATLIB-3SAT`.
  - **`R19-D4-H2` (Strict Monotone Sample Scaling):** All 7 paradigms improve monotonically as $N_{\text{train}}$ increases from $500 \to 2000 \to 10000$, with `GLC_AOM_v2` dominating at every $N_{\text{train}}$.

### Gate `R19-D5`: Iso-Parameter (`~3.5M` Params) & Iso-FLOP Compute-Optimal Frontier on `NVIDIA L4` GPU
- **Protocol:** Audit exact trainable parameter counts (`3.5M ± 10%`), total forward FLOPs per instance, and wall-clock latency (ms/instance on `NVIDIA L4` GPU) across $B \in \{1, 4, 16, 64, 128\}$.
- **Falsifiable Hypotheses & Pass Criteria:**
  - **`R19-D5-H1` (Iso-Parameter Parity Verified):** Every paradigm's trainable parameter count lies strictly within `[3.15M, 3.85M]`.
  - **`R19-D5-H2` (Pareto Frontier Dominance):** Across the grand-average exact solve rate over all 4 public benchmarks (`Sudoku-Extreme-81`, `SATLIB-3SAT`, `Sokoban-BoxPush`, `Countdown-Arithmetic`), `GLC_AOM_v2` strictly dominates the compute-optimal Pareto frontier (achieving higher solve rate at $B=16$ than any external baseline achieves at $B=64$).

### Gate `R19-D6`: Mechanistic Component Ablation of `GLC_AOM_v2` on `17-Clue Sudoku-Extreme` & OOD `SATLIB-3SAT` ($n=75$)
- **Compared Ablation Arms at $B=64$:**
  1. `GLC_AOM_v2 (Full)`: Complete architecture.
  2. `-Eureka_Crystallize`: Disables `R15-C3` persistent lemma crystallization at $b=16$.
  3. `-TwoStageTreeClearer`: Replaces Gershgorin-preconditioned `TwoStageTreeClearer` with flat uniform leaf averaging.
  4. `-Tseitin2Hop_AuxLemma`: Disables `BilinearAuxLemmaEmitter` & `Tseitin2HopQueryCoupler` ($Z_{\text{aux}} = 0$).
  5. `-CDCL_NogoodBackjump`: Disables global nogood broadcast and conflict-clause learning (reverts to chronological backtracking).
  6. `-TreePagedKVBank (Flat-Serialize)`: Serializes search traces as flat token sequences under identical token evaluation budget.
- **Falsifiable Hypotheses & Pass Criteria:**
  - **`R19-D6-H1` (Every Component Contributes Positively):** Removing any of the 5 core mechanisms degrades exact solve rate on `17-Clue Sudoku-Extreme` and/or OOD `SATLIB-3SAT` ($n=75$), with `GLC_AOM_v2 (Full)` achieving the highest combined solve rate across all 3 seeds.
