# Pre-Registration: Round 15 (`R15-CURIOSITY-SUITE`) — Curiosity-Driven Micro-Experiment Suite (`R15-C1` through `R15-C4`)

- **Document ID:** `PREREG-R15-CURIOSITY-SUITE`
- **Role:** Member 20 of the Society of Thinkers (Curiosity-Driven Micro-Experiment Suite `R15-C1..C4` Architect & Runner)
- **Seeds:** `101, 102, 103` ($n = 3$ independent seeds; two-sided 95% Student-$t$ confidence intervals with $t_{0.975, 2} = 4.3027$)
- **Zero-Leakage Audit:** Enforced via `src/eval/ast_leak_auditor.py` (`S1`–`S6` static AST audit) and runtime `PoisonTensor` oracle stripping (`R1`–`R4`) across every model forward pass and tree search evaluation.

---

## 1. Scientific Motivation & Overview

With the core `GLCAOMWorldModel` + `TreePagedKVSearch` architecture validated across Rounds 10–14, Kelvin granted the Society of Thinkers full agency to design and execute simultaneous curiosity-driven experiments that probe deep scientific questions about the mechanics of consistency search:

1. **`R15-C1` ("Phase Transitions in Optimal Search Topology: Chain vs. Bush vs. Tree vs. DAG"):** Why do some problems demand deep linear deduction chains with backjumping while others demand wide parallel root rollouts or bushy DAGs with transposition tables? Can a single neural search controller learn to undergo smooth **topological phase transitions** across the $(\sigma_{\text{obs}}, \gamma_{\text{couple}})$ plane?
2. **`R15-C2` ("Zero-Shot Compositional Chimera Transfer Across Search Paradigms"):** Can search skills learned separately from pure perfect-information tactical lookahead (`MCTS` on `Hex-5x5`), pure constraint propagation (`CDCL` on `Sudoku-6x6`), and pure imperfect-information regret matching (`CFR` on `Proposal-Poker`) compose **zero-shot** inside a single neural controller (`Chimera-ZeroShot`) on held-out hybrid games (`Micro-Stratego-4x4` and `ShipHunt-6x6`)?
3. **`R15-C3` ("The 'Eureka' Lemma Crystallization Effect: Persistent Auxiliary Memory Across Search Episodes"):** When early search branches ($b \in [1, 16]$) discover high-precision bilinear auxiliary lemmas ($|\ell_{\text{aux}}| > \tau_{\text{eureka}}$) and crystallize them into hard-gated persistent syndrome tokens inside `TreePagedKVBank` for all subsequent branches ($b \in [17, 64]$), does the search trajectory exhibit a sharp **"Eureka" phase drop** in remaining KL slope?
4. **`R15-C4` ("The Curse of Deterministic Search in Imperfect-Information Games: Can More MCTS Compute Make You MORE Exploitable?"):** Does scaling deterministic $\arg\max$ MCTS/Minimax compute ($B = 1 \to 64$) in imperfect-information games (`Proposal-Poker`, `Galanis-Market`, `Micro-Stratego-4x4`) actually **worsen** exploitability by collapsing mixed bluffing/hedging strategies onto predictable pure actions, whereas `GLCSearch-v2` with Information-Set Regret Matching monotonically reduces exploitability?

---

## 2. Pre-Registered Integrity Gate (`R15-D0`)

### Gate `R15-D0`: Zero Oracle Leakage Gate (`S1`–`S6` + `R1`–`R4`)
- **Static AST Audit (`S1`–`S6`):** `src/eval/ast_leak_auditor.py` inspects `src/models/glc_aom.py`, `src/search/kv_tree_search.py`, and `benchmarks/run_r15_curiosity_suite.py`. Must report `0` AST violations and `3/3` synthetic canaries caught.
- **Runtime `PoisonTensor` & Canary Invariance (`R1`–`R4`):**
  - `R1`: `batch.strip_oracle_labels()` replaces `exact_marginals`, `target_tokens`, and `query_targets` with `PoisonTensor` instances (`3/3` traps verified to raise `OracleLeakError` on any read/arithmetic).
  - `R2`: Across all evaluated environments and seeds, `PoisonTensor`-stripped `sub_blind`, all-zeros `sub_zeros`, and label-permuted `sub_perm_blind` produce bitwise-identical model and search outputs (`global_max_diff == 0.000000000`).
  - `R3`: SGD training produces non-zero parameter updates (`param_shift_l2 > 0`) and reduces root $B=0$ KL relative to random initialization.
  - `R4`: Cached root logits in `TreePagedKVBank` match fresh uncached model evaluations (`kv_equiv_max_diff == 0.000000000`).

---

## 3. Pre-Registered Experiment Specifications & Falsifiable Hypotheses

### Experiment `R15-C1`: Phase Transitions in Optimal Search Topology (Chain vs. Bush vs. Tree vs. DAG)

- **Independent Variables ($4 \times 4$ Phase Grid at Budget $B = 16$):**
  - **Local Observation Noise $\sigma_{\text{obs}} \in \{0.00, 0.15, 0.35, 0.60\}$:** Continuous Gaussian/corruption noise scale applied to local sensor/feature channels (`obs_features`) at test time (after `PoisonTensor` oracle stripping).
  - **Constraint Coupling / Logical Depth $\gamma_{\text{couple}} \in \{1, 2, 4, 8\}$:** Multi-hop constraint propagation horizon and tactical hostility depth across `Equiv-SAT`, `Sudoku-6x6`, `ShipHunt-6x6`, and `Hex-5x5`.
- **Compared Search Topologies ($B = 16$):**
  1. `Topo-Chain-CDCL`: Deep linear deduction chain ($j_{b+1} = b$ while $m_{\text{ql}}(C_b) \ge \tau_{\text{bt}}$) with conflict-driven backjumping to the highest-precision ancestor upon contradiction ($m_{\text{ql}}(C_b) < \tau_{\text{bt}}$).
  2. `Topo-Wide-Halving`: Wide parallel root rollouts ($j_{b+1} = 0$ for stage-1 breadth, followed by sequential halving depth-2 probes on top-$m_{\text{ql}}$ survivors).
  3. `Topo-Bushy-MCTS`: Balanced bushy MCTS tree (`ResolvableInconsistencyTreePUCT` with depth-2..4 internal splits, without DAG transposition merging).
  4. `Topo-DAG-Transposition`: Bushy MCTS tree equipped with `TreePagedKVBank.get_relational_masks()["transposition_mask"]` & `dag_anc_mask` DAG transposition merging across identical canonical assignment sets $\text{canon}(O_j) == \text{canon}(O_k)$.
  5. `GLC-Adaptive-PhaseController`: Learned neural topology controller that reads observable root features $(\bar{H}_0, \hat{\sigma}_{\text{obs}}, \hat{\gamma}_{\text{couple}})$ under $\chi = (1, 1)$ to modulate chain segment horizon, root breadth bonus, internal split bonus, and DAG transposition merging dynamically.
- **Measured Geometry Observables:**
  - Normalized maximum tree depth $d_{\max} / B \in [0, 1]$
  - Root branching fraction $f_{\text{root}} = N_{\text{exp}}(0) / B \in [0, 1]$
  - Internal branching fraction $f_{\text{internal}} = 1 - f_{\text{root}} - f_{\text{chain}}$
  - DAG transposition merge rate $f_{\text{dag}}$
  - Post-search posterior KL (`kl`, nats)
- **Falsifiable Hypotheses & Pass Criteria:**
  - **`R15-C1-H1` (Low-Noise Deep-Coupling Chain Regime):** At $(\sigma_{\text{obs}} = 0.00, \gamma_{\text{couple}} = 8)$, `Topo-Chain-CDCL` achieves strictly lower KL than `Topo-Wide-Halving` ($\Delta\text{KL} \ge 0.05$ nats), and `GLC-Adaptive-PhaseController` adopts chain-dominant geometry ($d_{\max}/B \ge 0.42$, $f_{\text{root}} \le 0.28$).
  - **`R15-C1-H2` (High-Noise Shallow-Coupling Wide Regime):** At $(\sigma_{\text{obs}} = 0.60, \gamma_{\text{couple}} = 1)$, `Topo-Wide-Halving` achieves strictly lower KL than `Topo-Chain-CDCL` ($\Delta\text{KL} \ge 0.04$ nats), and `GLC-Adaptive-PhaseController` transitions to wide root-parallel geometry ($f_{\text{root}} \ge 0.52$, $d_{\max}/B \le 0.25$).
  - **`R15-C1-H3` (Intermediate Bushy/DAG Regime & Adaptive Dominance):** At intermediate noise + coupling ($\sigma_{\text{obs}} \in \{0.15, 0.35\}, \gamma_{\text{couple}} \in \{2, 4\}$), `Topo-DAG-Transposition` outperforms tree-only `Topo-Bushy-MCTS`, and `GLC-Adaptive-PhaseController` achieves the lowest grand-mean KL across the entire $4 \times 4$ phase grid (matching or beating every fixed topology in $\ge 13/16$ cells).

---

### Experiment `R15-C2`: Zero-Shot Compositional Chimera Transfer Across Search Paradigms

- **Pure Single-Paradigm Training Teachers:**
  1. `Specialist-MCTS-Hex`: Neural world model & search controller trained strictly on `Hex-5x5` (tactical lookahead).
  2. `Specialist-CDCL-Sudoku`: Neural world model & search controller trained strictly on `Sudoku-6x6` (constraint propagation & conflict backjumping).
  3. `Specialist-CFR-Poker`: Neural world model & search controller trained strictly on `Proposal-Poker` (hidden-signal Bayesian inference & regret-matching hedging).
- **Compositional Chimera Arms:**
  4. `Chimera-ZeroShot`: Neural world model & 3-gate compositional search controller $(g_{\text{mcts}}, g_{\text{cdcl}}, g_{\text{cfr}})$ trained via supervised imitation on the union of the three pure teachers (`Hex-5x5` + `Sudoku-6x6` + `Proposal-Poker`), evaluated **strictly zero-shot** on held-out hybrid games (`Micro-Stratego-4x4` and `ShipHunt-6x6`).
  5. `Chimera-RL-Finetuned`: Initialized from `Chimera-ZeroShot` and fine-tuned on `Micro-Stratego-4x4` and `ShipHunt-6x6` using **unsupervised self-consistency & self-play RL** (optimizing only policy/controller parameters $\chi = (1, 1)$ against observable leaf self-consistency $m_{\text{ql}}(C_b)$, witness validity $\text{Check}(C_b)$, and utility vector $U$, with zero access to oracle labels and exact `0.000000` LIR belief drift).
- **Held-Out Hybrid Evaluation Benchmarks ($B = 16$):**
  - `Micro-Stratego-4x4` (requires hidden setup constraint elimination + opponent move signaling inference + tactical capture lookahead).
  - `ShipHunt-6x6` (requires fleet non-overlap constraint deduction + noisy sonar Bayesian inference + active probe planning).
- **Falsifiable Hypotheses & Pass Criteria:**
  - **`R15-C2-H1` (Zero-Shot Chimera Beats Every Single-Paradigm Specialist):** On both held-out hybrid games (`Micro-Stratego-4x4` and `ShipHunt-6x6`), `Chimera-ZeroShot` achieves strictly lower posterior KL and higher decision utility / win rate than all three single-paradigm specialists (`Specialist-MCTS-Hex`, `Specialist-CDCL-Sudoku`, `Specialist-CFR-Poker`).
  - **`R15-C2-H2` (Non-Degenerate Compositional Gate Usage):** On held-out `Micro-Stratego-4x4` and `ShipHunt-6x6`, `Chimera-ZeroShot` assigns active weight $\ge 0.18$ to each of $(g_{\text{mcts}}, g_{\text{cdcl}}, g_{\text{cfr}})$ with compositional gate entropy $H(g) \ge 0.95$ nats.
  - **`R15-C2-H3` (Unsupervised RL Fine-Tuning Gain with Zero LIR Drift):** `Chimera-RL-Finetuned` further improves action utility / search KL over `Chimera-ZeroShot` on both held-out hybrid games while preserving exact `0.000000` LIR belief drift under `swap_utility`.

---

### Experiment `R15-C3`: The "Eureka" Lemma Crystallization Effect (Persistent Auxiliary Memory Across Search Episodes)

- **Environments & Budget Trajectory:** `Equiv-SAT` and `Sudoku-6x6` evaluated across budgets $b \in \{0, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64\}$.
- **Eureka Crystallization Protocol ($b_{\text{cryst}} = 16$):**
  - During steps $b \in [1, 16]$, the search explores counterfactual branches and accumulates precision-weighted auxiliary lemma claims $z_{\text{aux}}^{(r)} \in \mathbb{R}^{D_{\text{aux}}^W}$ (`BilinearAuxLemmaEmitter`) and deduced variable constraints.
  - At $b = 16$, any auxiliary lemma coordinate or deduced variable assignment whose precision-weighted magnitude exceeds $\tau_{\text{eureka}} = 0.35$ and passes leaf self-consistency $m_{\text{ql}} \ge 0.80$ is **crystallized into a hard-gated persistent syndrome bank** (`crystallized_syndrome_bank`) inside `TreePagedKVBank`.
  - For all subsequent search steps $b \in [17, 64]$, every newly expanded branch reads the crystallized syndrome tokens directly via `GLCAOMWorldModel` top-down prefix/lemma cross-attention without re-deriving those lemmas from scratch.
- **Compared Arms:**
  1. `C3-NoAux`: $Z_{\text{aux}} = 0$ (`ZeroAuxEmitterWrapper`).
  2. `C3-Volatile-Aux`: Standard volatile per-branch auxiliary tokens (no persistent crystallization at $b=16$).
  3. `C3-Random-Crystallize`: Crystallizes random auxiliary coordinates of matched sparsity at $b=16$.
  4. `C3-Eureka-Crystallize`: Precision-gated Eureka lemma crystallization ($|\bar{z}_{\text{aux}}| \cdot \beta_{\text{eff}} > \tau_{\text{eureka}}$) at $b=16$.
- **Falsifiable Hypotheses & Pass Criteria:**
  - **`R15-C3-H1` (Post-Crystallization "Eureka" KL Slope Acceleration):** The immediate post-crystallization KL reduction $\Delta\text{KL}_{\text{eureka}} = \text{KL}(b=16) - \text{KL}(b=24)$ for `C3-Eureka-Crystallize` is at least **$2.0\times$ larger** than for `C3-Volatile-Aux` and `C3-Random-Crystallize` on both `Equiv-SAT` and `Sudoku-6x6`.
  - **`R15-C3-H2` (Superior $B=64$ Variable & Relational Query Calibration):** At $B=64$, `C3-Eureka-Crystallize` achieves the lowest variable KL and lowest calibrated relational query KL (`query_kl` $\le 0.12$ nats) across all 4 arms on both `Equiv-SAT` and `Sudoku-6x6`.
  - **`R15-C3-H3` (Zero False-Lemma Corruption):** Across all seeds, `C3-Eureka-Crystallize` maintains non-decreasing witness validity after crystallization (`witness_rate(B=64) >= witness_rate(B=16)`).

---

### Experiment `R15-C4`: The Curse of Deterministic Search in Imperfect-Information Games

- **Environments:** `Proposal-Poker`, `Galanis-Market` (`galanis_t3s111o2ye2`), and `Micro-Stratego-4x4`.
- **Compute Budgets:** $B \in \{1, 2, 4, 8, 16, 32, 64\}$.
- **Compared Search Policy Arms:**
  1. `Det-Argmax-Minimax`: Deterministic $\arg\max / \min\max$ tree value backup and greedy action selection without information-set regret matching.
  2. `Det-PUCT-VisitMax`: Standard MCTS visit-count maximization ($\pi(a) \propto N(a)^{1/\tau_{\text{det}}}$, $\tau_{\text{det}} = 0.15$) without information-set regret matching.
  3. `Tabular-Prior-Baseline`: Fixed $B=0$ root policy baseline without tree search.
  4. `GLCSearch-v2-Infoset-CFR`: Uses `TreePagedKVBank.get_relational_masks()["infoset_mask"]` to pool counterfactual action values across public-observation information sets, computes cumulative positive counterfactual regrets $R_I^{(b,+)}(a)$, and plays the epistemic-uncertainty-anchored regret-matched mixed strategy $\bar{\sigma}_I^{(B)}(a)$.
- **Primary Evaluation Metrics:**
  - **Exploitability $\mathcal{E}(B)$ (Best-Response Loss):** Expected utility advantage of an exact infoset-aware Best-Response (BR) opponent that knows the agent's conditional policy $\pi_B(a \mid s_{\text{priv}}, I_{\text{pub}})$ and plays the Bayes-optimal counter-strategy against the induced posterior over private states.
  - **Policy Entropy $H(\pi_B)$ (nats)** and **Equilibrium KL $\text{KL}(\pi^*_{\text{NE}} \parallel \pi_B)$**.
  - **Direct Head-to-Head Win Rate** of `GLCSearch-v2-Infoset-CFR` vs. `Det-Argmax-Minimax` at matched compute budgets $B \in \{1, 4, 16, 64\}$.
- **Falsifiable Hypotheses & Pass Criteria:**
  - **`R15-C4-H1` (Curse of Deterministic Search Verified):** As compute scales from $B=1 \to 64$, `Det-Argmax-Minimax` suffers **increasing or non-decreasing exploitability** ($\mathcal{E}_{\text{Det}}(B=64) > \mathcal{E}_{\text{Det}}(B=1)$ on all 3 games) as its policy entropy $H(\pi_B)$ collapses by $> 60\%$, leaking private information to an infoset-aware opponent.
  - **`R15-C4-H2` (Monotone Exploitability Reduction under `GLCSearch-v2-Infoset-CFR`):** `GLCSearch-v2-Infoset-CFR` monotonically reduces exploitability as $B$ scales ($\mathcal{E}_{\text{GLC}}(B=64) < \mathcal{E}_{\text{GLC}}(B=16) < \mathcal{E}_{\text{GLC}}(B=1)$ on all 3 games), achieving $> 3.5\times$ lower exploitability at $B=64$ than `Det-Argmax-Minimax`.
  - **`R15-C4-H3` (Head-to-Head Tournament Dominance):** At $B=64$, `GLCSearch-v2-Infoset-CFR` achieves $\ge 65\%$ head-to-head win rate against `Det-Argmax-Minimax` across all 3 imperfect-information games.
