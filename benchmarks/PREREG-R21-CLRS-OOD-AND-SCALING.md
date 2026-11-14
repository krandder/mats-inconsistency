# Pre-Registration Protocol: Round 21 (`R21-CLRS-OOD-AND-SCALING`) — Public `CLRS-30` Out-of-Distribution Algorithmic Extrapolation ($n=16 \to n=64$) & Model-Size Scaling Laws ($0.8\text{M} \to 3.2\text{M} \to 12.8\text{M}$) on `NVIDIA L4` GPU

- **Document ID:** `PREREG-R21-CLRS-OOD-AND-SCALING`
- **Status:** **LOCKED PRIOR TO ROUND-21 EXECUTION**
- **Date:** 2026-10-19
- **Author:** Member 29 of the Society of Thinkers (Round 21 / Pillar 4)
- **Repository:** `/home/agent/workspaces/inconsistency-search` (`futarchy-fi/inconsistency-search`)
- **Connected Foundations:**
  - [`docs/literature/03-neural-algorithmic-reasoning-clrs-mctsnet-incontext.md`](../docs/literature/03-neural-algorithmic-reasoning-clrs-mctsnet-incontext.md) (`CLRS-30`, `Triplet-GMPNN`, `Hint-ReLIC`, `MCTSNet`, and $\mathsf{TC}^0$ vs. $\mathsf{NC}^1$ circuit complexity)
  - [`benchmarks/PREREG-R13-ALGO-IMITATION.md`](PREREG-R13-ALGO-IMITATION.md) (`R13` Constructive & Supervised Search Imitation)
  - [`benchmarks/PREREG-R14-UNSUP-DISCOVERY.md`](PREREG-R14-UNSUP-DISCOVERY.md) (`R14` Unsupervised Search Discovery via `LSO` Counterfactual Credit)
  - [`src/search/neural_tree_controller.py`](../src/search/neural_tree_controller.py) (`GLCSearchTransformer` `v1`/`v2` & `Tseitin2HopQueryCoupler`)
  - [`src/models/glc_aom.py`](../src/models/glc_aom.py) (`GLCAOMWorldModel`, `BilinearAuxLemmaEmitter`, `LIRMask`)
  - [`src/eval/ast_leak_auditor.py`](../src/eval/ast_leak_auditor.py) (`S1`–`S6` Static AST Zero-Oracle-Leakage Auditor)

---

## 0. Scientific Motivation & Kelvin's Mandate

Kelvin's directive for Round 21 establishes a clear bar:
> *"We need hard, cold results on either external benchmarks, or things that could be external benchmarks. I want numbers. On benchmarks I understand. I want to see open problems, or benchmarks that are unsaturated (or unsaturated on small models -> public benchmarks that require smaller models would be great)... You can spend up to the full budget if needed."*

Pillar 4 (`R21-CLRS-OOD-AND-SCALING`) directly answers two of the most prominent open problems in neural algorithmic reasoning and test-time compute scaling:

1. **Open Problem 1 — `CLRS-30` $4\times$ Out-of-Distribution (`OOD`) Length Extrapolation ($n \le 16 \to n = 64$) and the Famous `No-Hint` Collapse (Veličković et al., *ICML* 2022; Ibarz et al., *LoG* 2022; Bevilacqua et al., *ICML* 2023):**
   - In DeepMind's canonical `CLRS-30` Algorithmic Reasoning Benchmark, standard Message-Passing Neural Networks (`MPNN`, `PGN`, `Triplet-GMPNN`) and flat Relational Transformers trained on small graphs ($n \le 16$) achieve $>90\%\text{--}98\%$ accuracy in-distribution ($n = 16$), yet degrade sharply when tested zero-shot at $4\times$ graph/tree size ($n = 64$), often dropping to $50\%\text{--}75\%$ micro-F1 on long-horizon sequential and global closure algorithms (`DFS`, `Bellman-Ford`, `Dijkstra`, `2-SAT / Tarjan SCC`).
   - Even more severely, when step-by-step intermediate trajectory hints $(h^{(1)}, \dots, h^{(T)})$ are withheld during training (**No-Hint Outcome-Only Supervision**), standard `CLRS` architectures suffer catastrophic OOD collapse ($40\%\text{--}60\%$ micro-F1 at $n=64$) because backpropagation through $T = \Theta(n)$ unrolled message-passing steps without intermediate credit assignment overfits to spurious graph-size shortcuts.
   - **Our Hypothesis (`R21-E1`):** Equipping the neural processor with `GLCSearch-v2`'s incremental transitive reachability masks (`AncMask` / Kleene-star `DAG` closure $\mathcal{M}_{\text{desc}}^{\text{DAG}}$), depth-synchronous $\mathsf{NC}^1$ ancestor-spine semiring recurrence (`V2.1′`), 2-hop Tseitin/triplet pivot resolution, and **Leave-Subtree-Out (`LSO`) counterfactual credit** + `LIR` preconditioned clearing enables both **$\ge 90\%$ zero-shot $4\times$ OOD extrapolation ($n=64$) with hints** and **$\ge 85\%$ zero-shot $4\times$ OOD extrapolation *without* intermediate hints**!

2. **Open Problem 2 — The Parameter-vs-Search Scaling Exchange Rate Across 3 Orders of Magnitude (`0.8M` $\to$ `3.2M` $\to$ `12.8M` Parameters vs. $B \in \{1, 4, 16, 64, 128\}$):**
   - How does test-time structured tree search (`GLC-AOM + GLCSearch-v2` with `LIR` preconditioned clearing) trade off against raw model parameter count $N_{\text{param}}$ compared to `Standard Autoregressive + GRPO` (unstructured parallel rollout sampling) and `Gumbel MuZero`?
   - **Our Hypothesis (`R21-E2`):** On unsaturated combinatorial, epistemic, and OOD algorithmic benchmarks (`Sudoku-Extreme`, `Stratego`, `Equiv-SAT`, and `CLRS-OOD n=64`), `GLC-AOM + GLCSearch-v2` exhibits a search-compute scaling exponent $\alpha_B$ more than $2.2\times$ steeper than `Standard AR + GRPO`, allowing a **`0.8M`-parameter (`Tiny`) `GLC-AOM + GLCSearch-v2` model at $B=64$** to decisively beat both a **$16\times$ larger `12.8M`-parameter `1-pass` ($B=1$) model** and a **$16\times$ larger `12.8M`-parameter `Standard AR + GRPO` model at $B=64$**!

---

## 1. Experimental Design

### 1.1 Experiment `R21-E1`: Public `CLRS-30` & `CLRS-Game` $4\times$ OOD Extrapolation Suite ($n \in [4, 16] \to n = 64$)

We implement exact `CLRS-30` problem generators, typed probe representations (`node`, `edge`, `graph` probes of types `scalar`, `mask`, `pointer`), and ground-truth step-by-step execution traces for **8 canonical algorithms** (6 core `CLRS-30` tasks spanning graph traversal, shortest paths, spanning trees, and satisfiability/SCC closure + 2 `CLRS-Game` $\mathsf{NC}^1$/DAG extensions):

| Task ID | Algorithm Family | Output Probe(s) $y^{(T)}$ | Intermediate Hint Trajectory $h^{(1..T)}$ | Primary OOD Bottleneck at $n = 64$ ($4\times$) |
| :--- | :--- | :--- | :--- | :--- |
| **`CLRS_BFS`** | Graph Traversal (Commutative Frontier) | Predecessor pointer $\pi_v \in V$ & reachability mask $r_v \in \{0,1\}$ | Frontier wave $q^{(t)}_v$, visited mask $vis^{(t)}_v$, hop distance $d^{(t)}_v$ | Pointer crowding as degree/frontier width scales $4\times$ |
| **`CLRS_DFS`** | Sequential Depth-First Search & Backtracking | DFS predecessor forest $\pi_v \in V$ & discovery order class | Active stack top, discovery/finish timestamps $(d_v, f_v)$, tree/back-edge masks | Chain depth $D = \Theta(n) = 64 \gg L$ breaks 1-hop parallel message passing |
| **`CLRS_Dijkstra`** | Weighted Shortest Paths ($\min, +$ Non-Negative) | Shortest-path tree pointers $\pi_v \in V$ & distance rank | Tentative distance $d^{(t)}_v$, settled set $S^{(t)}$, argmin pivot $u^{(t)}$ | Priority-queue tie-breaking & long-path distance drift over $n=64$ |
| **`CLRS_Bellman_Ford`** | General Weighted Shortest Paths + Negative Cycles | Relaxed predecessor $\pi_v \in V$ & valid/cycle-free flag | Stepwise edge relaxations $d_v^{(t)} = \min_u(d_u^{(t-1)} + w_{uv})$ | Spurious relaxation over unreached/high-weight cousin edges |
| **`CLRS_MST_Prim`** | Greedy Minimum Spanning Tree | MST parent pointer $\pi_v \in V$ & cut-edge inclusion mask | Lightest cut edge $k^{(t)}_v = \min_{u \in S^{(t)}} w_{uv}$, key updates | Global cycle avoidance vs. local greedy edge selection |
| **`CLRS_2SAT_Tarjan_SCC`** | Implication Closure & Strongly Connected Components | SCC root pointer $\mathrm{scc}_v$, 2-SAT witness $x_i \in \{0,1\}$ & satisfiability | Low-link values $\mathrm{low}_v^{(t)}$, active DFS Tarjan stack, 2-hop implication closure | Transitive implication chains $u \Rightarrow \dots \Rightarrow \neg u$ of length $> 16$ |
| **`CLRS_Game_Minimax_AlphaBeta`** | `CLRS-Game` Alternating $(\max, \min)$ Tree Backup | Exact minimax root/node value sign & optimal child pointer $\pi_v$, pruned mask | Bottom-up ply values $V^{(d)}_v$ and top-down $[\alpha_v, \beta_v]$ cutoff bounds | $\mathsf{NC}^1$-complete alternating semiring evaluation beyond fixed layer depth $L$ |
| **`CLRS_Game_DAG_PNS`** | `CLRS-Game` DAG Shortest-Proof / Proof-Number Search | Most-Proving Node (`MPN`) path mask & proof/disproof $(\mathrm{pn}_v, \mathrm{dn}_v)$ class | Stepwise $(\min, \sum)$ OR/AND proof numbers across DAG transpositions | Multi-path double counting when cousin transpositions merge in a DAG |

#### Five Neural Algorithmic Architectures Compared (`Seeds 101, 102, 103`):
Each architecture is trained on graphs/trees of size $n \in [4, 16]$ under **two supervision regimes**:
- **Regime A (`Hinted`):** Standard `CLRS-30` loss $\mathcal{L} = \mathcal{L}_{\text{output}}(y, \hat{y}^{(T)}) + \frac{1}{T}\sum_{t=1}^T \mathcal{L}_{\text{hint}}(h^{(t)}, \hat{h}^{(t)})$.
- **Regime B (`No-Hint / Outcome-Only`):** Trained strictly on terminal output $\mathcal{L}_{\text{output}}(y, \hat{y}^{(T)})$ with **zero access to intermediate teacher hints** $h^{(1..T)}$.

1. **`MPNN_MaxPool` (Veličković et al., *ICLR* 2020):** Encode-Process-Decode GNN with 1-hop edge message passing and $\max$-pooling aggregation.
2. **`PGN_PointerGraphNet` (Veličković et al., *NeurIPS* 2020):** Pointer Graph Network augmenting 1-hop graph edges with learned soft pointer attention edges.
3. **`Triplet_GMPNN` (Ibarz et al., *LoG* 2022):** DeepMind's Generalist Neural Algorithmic Learner baseline incorporating triplet pivot message passing $u_{ijk}^{(t)} = \psi(z_i^{(t)}, z_j^{(t)}, z_k^{(t)}, e_{ik}^{(t)}, e_{kj}^{(t)}, e_{ij}^{(t)})$ with $\max_k u_{ijk}^{(t)}$ edge updates.
4. **`Flat_Relational_Transformer`:** Multi-head self-attention Transformer over node and edge representations without transitive tree/DAG closure masks or `LIR` preconditioned clearing.
5. **`GLCSearch_v2_LIR` (`GLCSearch-v2 + LIR Preconditioned Clearing`, Ours):** Combines:
   - Incremental transitive reachability & Kleene-star DAG closure $\mathcal{M}_{\text{desc}}^{\text{DAG}} = (E_{\text{tree}} \lor E_{\text{trans}})^*$ (`V2.2′`),
   - Depth-synchronous $\mathsf{NC}^1$ ancestor-spine semiring recurrence (`V2.1′`) for alternating $(\max, \min)$, tropical $(\min, +)$, and $(\min, \sum)$ backup across arbitrary depth $D \le n = 64$,
   - 2-hop Tseitin/triplet co-fluctuation coupling (`Tseitin2HopQueryCoupler`) + literal-bitmask conflict broadcast (`V2.5′`),
   - `LIR` preconditioned normal-equation clearing (`TwoStageTreeClearer`) to eliminate redundant cousin/unreached-edge interference,
   - **Leave-Subtree-Out (`LSO`) counterfactual credit** $I_{g,t} = S(q^{-s_t}, y) - S(q^{\text{full}}, y)$ during **No-Hint Outcome-Only** training, providing dense zero-variance step-level credit without teacher hints!

---

### 1.2 Experiment `R21-E2`: Model-Parameter & Test-Time Compute Scaling Laws (`0.8M` $\to$ `3.2M` $\to$ `12.8M` Parameters $\times$ $B \in \{1, 4, 16, 64, 128\}$)

We instantiate and train three model scales spanning a $16\times$ ($4^2$) parameter range on `NVIDIA L4` GPU:

| Scale Tier | Target $N_{\text{param}}$ | World Model (`GLC-AOM` / AR Backbone) Config | Search Controller (`GLCSearch-v2`) Config |
| :--- | :--- | :--- | :--- |
| **`Tiny` (`0.8M`)** | $\approx 0.80\text{M}$ ($[0.72\text{M}, 0.88\text{M}]$) | $d_{\text{model}}=128, L_{\text{wm}}=3, H=4, m_{\text{aux}}=8, d_{\text{aux}}=16$ | $d_{\text{tree}}=64, L_{\text{tree}}=2, H_{\text{tree}}=4$ |
| **`Small` (`3.2M`)** | $\approx 3.20\text{M}$ ($[2.95\text{M}, 3.45\text{M}]$) | $d_{\text{model}}=256, L_{\text{wm}}=4, H=8, m_{\text{aux}}=16, d_{\text{aux}}=16$ | $d_{\text{tree}}=128, L_{\text{tree}}=3, H_{\text{tree}}=4$ |
| **`Medium` (`12.8M`)** | $\approx 12.80\text{M}$ ($[12.0\text{M}, 13.6\text{M}]$) | $d_{\text{model}}=512, L_{\text{wm}}=6, H=8, m_{\text{aux}}=32, d_{\text{aux}}=16$ | $d_{\text{tree}}=256, L_{\text{tree}}=4, H_{\text{tree}}=8$ |

Across all 3 parameter scales (`Tiny 0.8M`, `Small 3.2M`, `Medium 12.8M`) and 5 test-time search compute budgets $B \in \{1, 4, 16, 64, 128\}$, we compare 3 paradigm families on our 4 hardest unsaturated benchmarks:
- **Paradigms:**
  1. **`GLC_AOM_GLCSearch_v2` (Ours):** `GLC-AOM` world model + `GLCSearch-v2` relational tree/DAG/infoset controller + `LIR` preconditioned normal-equation clearing + `BilinearAuxLemmaEmitter` (`Z^W_aux`) + `LSO` credit.
  2. **`Standard_AR_GRPO`:** Parameter-matched Autoregressive Transformer trained with Group Relative Policy Optimization (`GRPO`) sampling $B$ parallel rollouts and pooling via reward/confidence-weighted voting (no `TreePagedKVBank` branching cache, no `LIR` firewall, no Gram deduplication).
  3. **`Gumbel_MuZero`:** Parameter-matched Gumbel MuZero sequential halving search over learned latent dynamics (has sequential halving `V2.4′`, but lacks transitive `DAG`/Nogood closure, `CFR+` infoset pooling, `LIR` preconditioned clearing, and `Z^W_aux` bilinear lemma channels).
- **4 Unsaturated Benchmark Tasks:**
  1. **`Sudoku-Extreme` (`sudoku_9x9_extreme`):** Hard $9\times 9$ Sudoku puzzles requiring deep backtracking and global conflict-clause (`CDCL`) nogood pruning.
  2. **`Stratego` (`micro_stratego_6x6`):** Imperfect-information adversarial game requiring hidden-state belief tracking, infoset regret matching (`CFR+`), and deep minimax lookahead.
  3. **`Equiv-SAT` (`equiv_sat`):** Multi-hop Tseitin parity/equivalence SAT with floating variables ($P^*(A=1)=P^*(B=1)=0.5$) where 1st-order marginals cancel and 2-hop bilinear lemmas $Z^W_{\text{aux}}$ are required.
  4. **`CLRS-OOD-n64`:** Zero-shot $4\times$ OOD algorithmic execution at graph/tree size $n=64$ across the 8 `CLRS` tasks.

---

## 2. Pre-Registered Falsifiable Decision Gates (`R21-D0` through `R21-D5`)

All statistical comparisons use 3 independent random seeds (`Seeds 101, 102, 103`) and two-sided paired Student-$t$ tests with $df = 2$ ($t_{0.975,2} = 4.303$).

### Gate `R21-D0`: Zero Oracle Leakage & Parameter-Scale Integrity Gate (Hard Prerequisite)
1. **Static AST Audit (`S1`–`S6`):** `src/eval/ast_leak_auditor.py` inspecting `benchmarks/run_r21_clrs_ood_and_scaling.py`, `src/models/glc_aom.py`, and `src/search/neural_tree_controller.py` must report **`0` violations** and **`3/3` synthetic canaries caught**.
2. **Runtime `PoisonTensor` Invariance (`R1`–`R4`):** Replacing all ground-truth `EnvBatch` oracle fields (`exact_marginals`, `target_tokens`, `query_targets`) and all ground-truth `CLRS` evaluation targets/hints with `PoisonTensor` during test-time inference must yield **`global_max_diff == 0.000000000`** across all architectures, tasks, and model scales.
3. **Exact Parameter Scale & Non-Vacuous Gradient Verification:** Actual PyTorch parameter counts `sum(p.numel() for p in model.parameters())` must fall strictly within pre-registered bands for `Tiny` ($0.80\text{M} \pm 0.10\text{M}$), `Small` ($3.20\text{M} \pm 0.30\text{M}$), and `Medium` ($12.80\text{M} \pm 1.00\text{M}$), with $\|\theta_{\text{final}} - \theta_{\text{init}}\|_2 > 0$ for every trained model.

### Gate `R21-D1`: `CLRS-30` Hinted $4\times$ OOD Extrapolation Gate ($n=16 \to n=64$)
- **Pass Criteria:**
  1. Under Hinted supervision on $n \in [4, 16]$, **`GLCSearch_v2_LIR` (Ours)** achieves **$\ge 0.900$ macro-averaged `micro_f1` and $\ge 0.900$ `node_pointer_acc` zero-shot at $4\times$ OOD ($n = 64$)** across the 8 `CLRS` tasks, with **$\ge 0.860$ `micro_f1` on every individual task**.
  2. `GLCSearch_v2_LIR` beats the strongest baseline (`Triplet_GMPNN`) at $4\times$ OOD ($n=64$) by **$\ge +0.140$ ($+14.0$ pp) macro `micro_f1`** with paired $t$-statistic $t > 4.303$ ($p < 0.025$), and maintains an OOD retention ratio $\text{F1}_{n=64} / \text{F1}_{n=16} \ge 0.930$ (whereas `MPNN_MaxPool`, `PGN_PointerGraphNet`, `Triplet_GMPNN`, and `Flat_Relational_Transformer` all drop to retention ratios $\le 0.820$).

### Gate `R21-D2`: `CLRS-30` No-Hint Outcome-Only Discovery Gate (Solving the Open `CLRS` No-Hint Problem)
- **Pass Criteria:**
  1. Under **No-Hint Outcome-Only Supervision** (zero intermediate teacher hints during training), standard baselines (`MPNN_MaxPool`, `PGN_PointerGraphNet`, `Triplet_GMPNN`, `Flat_Relational_Transformer`) degrade to **$\le 0.580$ macro `micro_f1` at OOD $n=64$**.
  2. Using **Leave-Subtree-Out (`LSO`) counterfactual credit** and `LIR` preconditioned clearing, **`GLCSearch_v2_LIR` (No-Hint)** achieves **$\ge 0.850$ macro `micro_f1` zero-shot at $4\times$ OOD ($n=64$)**, outperforming No-Hint `Triplet_GMPNN` by **$\ge +0.280$ ($+28.0$ pp)** ($t > 4.303$) and outperforming even *Hinted* `Triplet_GMPNN` at $n=64$ by **$\ge +0.080$ ($+8.0$ pp)**!

### Gate `R21-D3`: Joint Parameter & Test-Time Compute Scaling Law Gate (`0.8M` $\to$ `3.2M` $\to$ `12.8M`)
- **Pass Criteria:**
  1. Across all 4 hard tasks (`Sudoku-Extreme`, `Stratego`, `Equiv-SAT`, `CLRS-OOD-n64`), `GLC_AOM_GLCSearch_v2` exhibits **strictly monotonic improvement** in composite task solve/accuracy rate and proper belief $\mathrm{KL}$ across every parameter increase ($0.8\text{M} < 3.2\text{M} < 12.8\text{M}$) and every search budget increase ($B=1 < 4 < 16 < 64 < 128$).
  2. Fitting the joint scaling law $\mathcal{L}(N_{\text{param}}, B) = \mathcal{L}_\infty + A \cdot N_{\text{param}}^{-\alpha_N} \cdot B^{-\alpha_B}$ across the $3 \times 5 = 15$ `(scale, budget)` grid points achieves **$R^2 \ge 0.940$**, with search-compute scaling exponent $\alpha_B(\text{GLCSearch-v2}) \ge 2.2 \times \alpha_B(\text{Standard AR + GRPO})$.

### Gate `R21-D4`: Iso-Performance Parameter-Compute Exchange Rate Gate (`0.8M` at $B=64$ Beats `12.8M` Baselines)
- **Pass Criteria:**
  1. At test-time search budget $B=64$, the **`0.8M` (`Tiny`) `GLC_AOM_GLCSearch_v2`** model beats the **$16\times$ larger `12.8M` (`Medium`) `1-pass` ($B=1$) model** on all 4 tasks by **$\ge +0.250$ ($+25.0$ pp) macro solve/accuracy rate** ($t > 4.303$).
  2. At $B=64$, the **`0.8M` (`Tiny`) `GLC_AOM_GLCSearch_v2`** model also beats the **$16\times$ larger `12.8M` (`Medium`) `Standard_AR_GRPO` model at $B=64$** across all 4 tasks by **$\ge +0.150$ ($+15.0$ pp) macro solve/accuracy rate** ($t > 4.303$), proving an iso-performance parameter-efficiency multiplier **$\mathcal{E}_{N \leftrightarrow B} \ge 16.0\times$**!
  3. At matched scale and budget ($B=64$), `GLC_AOM_GLCSearch_v2` beats `Gumbel_MuZero` by **$\ge +0.120$ ($+12.0$ pp)** macro solve/accuracy rate ($t > 4.303$).

### Gate `R21-D5`: Hardware Scaling Efficiency & `LIR` Epistemic Firewall Across Scales (`NVIDIA L4` GPU)
- **Pass Criteria:**
  1. Across all 3 parameter scales (`0.8M`, `3.2M`, `12.8M`) and all budgets $B \in \{1, 4, 16, 64, 128\}$, `GLC_AOM_GLCSearch_v2` maintains **exact zero `LIR` utility-to-belief leakage** ($\|\nabla_{Z^\Pi} \ell_V\|_\infty = 0.000000$ and belief $\mathrm{KL}$ drift under utility sign negation $= 0.000000$), whereas `Standard_AR_GRPO` exhibits utility-induced belief drift $\ge 0.080$ nats.
  2. Peak `NVIDIA L4` VRAM allocation across all runs remains $\le 4.0\text{ GB}$ with $0$ OOM/NaN errors.
