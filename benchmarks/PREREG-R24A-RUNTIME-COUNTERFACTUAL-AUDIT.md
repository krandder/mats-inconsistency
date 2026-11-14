# Pre-Registration Protocol: Round `R24A` — Stage-3 Runtime Name-Agnostic Counterfactual Audit (`Rules S11–S14`)

**Round ID:** `R24A` (`Runtime Counterfactual Audit Suite S11–S14`)  
**Author:** Member 33 (`Runtime Counterfactual Auditor S11–S14` Builder & Verifier, Society of Thinkers)  
**Pre-Registration Timestamp:** `2026-10-19T05:00:00Z`  
**Resolves Mandate:** Society of Thinkers Round 6 Scoreboard §4.4 & Critique `Opus-5` / `Fable-4` (*"Static AST Auditing Can Be Bypassed via Identifier Renaming"*)  
**Target Codebases:**
- **Wave-1 (Forensically Retracted Runners):**
  - [benchmarks/run_r18_openspiel_stratego.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r18_openspiel_stratego.py)
  - [benchmarks/run_r19_public_reasoning_benchmarks.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r19_public_reasoning_benchmarks.py)
  - [benchmarks/run_r20_kyle_lmsr_futarchy.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r20_kyle_lmsr_futarchy.py)
  - [benchmarks/run_r21_clrs_ood_and_scaling.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r21_clrs_ood_and_scaling.py)
- **Wave-2 (Genuine Re-Executed Runners):**
  - [benchmarks/run_r22a_genuine_openspiel_stratego.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r22a_genuine_openspiel_stratego.py)
  - [benchmarks/run_r22b_genuine_public_reasoning_clrs.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r22b_genuine_public_reasoning_clrs.py)
  - [benchmarks/run_r23_genuine_kyle_lmsr_futarchy.py](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r23_genuine_kyle_lmsr_futarchy.py)

---

## 1. Scientific Rationale & Threat Model (`Opus-5` / `Fable-4` Resolution)

In Round 6 (`04-opus-round6-audit.md` §5 and `05-fable-round6-audit.md` §3), the Society of Thinkers demonstrated that while `src/eval/ast_leak_auditor.py` (`v2`, Rules `S1`–`S10`) catches static oracle-field accesses and known banned variable names, any purely syntactic AST scanner is vulnerable to **identifier-renaming evasion**:
1. A graph model can rename `idx_norm` or `dist_vec` to a benign name (`canonical_slot` or `global_edge_summary`), passing `S1`–`S10` while still breaking node permutation equivariance or 1-hop message-passing locality.
2. A test-time search loop can rename `base_kl` or `b_pow` to benign identifiers, passing `S1`–`S10` while executing an inert rollout that ignores intermediate search states at step $t=2$ and scales $B=1$ loss by a deterministic curve.
3. A reasoning benchmark runner can rename `_complete_leaf_with_observable_search` to a benign helper name, passing `S1`–`S10` while running a multi-node classical solver that solves $>50\%$ of instances with $0$ neural forward calls or applies helper calls asymmetrically across paradigms.
4. A decentralized market runner can rename `warm_w` or `sym_anchor_1` to benign identifiers, passing `S1`–`S10` while ignoring the broadcast market price $P_t$ (`0.0` price sensitivity) or leaking live cross-module autograd gradients.

To make shortcut evasion impossible regardless of identifier naming, `R24A` introduces **Stage-3 Runtime Name-Agnostic Counterfactual Auditing (`Rules S11–S14`)** in [src/eval/runtime_counterfactual_auditor.py](file:///home/agent/workspaces/inconsistency-search/src/eval/runtime_counterfactual_auditor.py). Every rule treats the target model/runner as an opaque black-box function and probes its input-output Jacobian, equivariance, and counterfactual intervention response at runtime.

---

## 2. Pre-Registered Mathematical Definitions & Pass/Fail Gates (`Rules S11–S14`)

### 2.1 Rule `S11`: Graph Permutation Equivariance (`S11-E`) & 1-Hop Message-Passing Locality (`S11-L`)

- **Gate `S11-E` (`Node Permutation Equivariance`):**
  Let $G = (X, E, A)$ be a graph batch with node features $X \in \mathbb{R}^{B \times N \times d_V}$, edge weights $E \in \mathbb{R}^{B \times N \times N}$, and adjacency mask $A \in \{0,1\}^{B \times N \times N}$. For any random non-identity permutation matrix $P \in \{0,1\}^{N \times N}$ ($P \neq I_N$) and permuted graph $\pi(G) = (P X, P E P^\top, P A P^\top)$, the processor's pointer probability matrix $p(G) = \operatorname{softmax}(\ell_{\text{ptr}}(G))$ and node mask probabilities $m(G) = \sigma(\ell_{\text{mask}}(G))$ must satisfy:
  $$\|p(\pi(G)) - P\, p(G)\, P^\top\|_\infty < 10^{-5} \quad \text{and} \quad \|m(\pi(G)) - P\, m(G)\|_\infty < 10^{-5}.$$

- **Gate `S11-L` (`1-Hop Message-Passing Locality vs. Multi-Hop Propagation`):**
  On an $N=16$ weighted graph with source vertex $s = 0$, perturb an edge weight $W_{uv} \leftarrow W_{uv} + \delta$ ($\delta = 5.0$) at shortest-path hop distance $\operatorname{dist}_G(s, u) \ge 3$ and $\operatorname{dist}_G(s, v) \ge 3$ (specifically $(u,v) = (4,5)$ with $\operatorname{dist}_G(0,4)=4$).
  - At 1-hop message-passing step $t=1$, source vertex $s$'s hidden representation $h_s^{(1)}$ must exhibit **exact zero sensitivity**:
    $$\Delta_{t=1}(s) = \|h_s^{(1)}(G) - h_s^{(1)}(\tilde{G})\|_\infty < 10^{-7}.$$
  - At multi-hop step $t=L$ ($L = 6 \ge \operatorname{dist}_G(s, u) + 1$), the perturbation reaches $s$ along the graph and must produce **strictly positive sensitivity**:
    $$\Delta_{t=L}(s) = \|h_s^{(L)}(G) - h_s^{(L)}(\tilde{G})\|_\infty > 10^{-5}.$$

### 2.2 Rule `S12`: Step-Wise Search State Intervention (`S12-2`) & Loss-Ratio Dispersion (`S12-4`)

- **Gate `S12-2` (`Mid-Rollout Step-$t=2$ Neural State Intervention`):**
  During a $B$-step neural search rollout ($B \ge 4$), inject an $\epsilon = 0.05$ perturbation into the internal neural search state (tree node logits / consensus state) at step $t=2$ (`step_b == 1`, strictly after the $B=1$ root pass has completed).
  - Step $t=1$ (`B=1`) root probabilities must remain bitwise identical:
    $$\Delta_{t=1} = \|p^{(1)} - \tilde{p}^{(1)}\|_\infty \le 10^{-8}.$$
  - Final rollout probabilities ($t' = B > 2$) must change strictly:
    $$\Delta_{t' > 2} = \|p^{(B)} - \tilde{p}^{(B)}\|_\infty > 10^{-5}.$$

- **Gate `S12-4` (`Cross-Instance Loss-Ratio Dispersion`):**
  Across $M \ge 8$ distinct test instances $i \in \{1, \dots, M\}$, let $L_i(B)$ denote the per-instance search loss (KL divergence or constraint violation) at budget $B$. For every tested search budget $B > 1$, the per-instance improvement ratio $r_i(B) = L_i(B) / \max(L_i(1), 10^{-12})$ must exhibit strictly positive cross-instance standard deviation:
  $$\operatorname{Std}_{i=1..M}\!\left(\frac{L_i(B)}{L_i(1)}\right) > 10^{-4}.$$

### 2.3 Rule `S13`: Track-0 Helper-Only Baseline & Symmetric Evaluator Ledger Gate

- **Gate `S13-Track0` (`0-Neural-Call Helper-Only Baseline`):**
  Run the test-time constraint propagation helper on an uninformative uniform prior ($p_{b,v,k} = 1/K$, $0$ neural forward calls) across all test instances (`Sudoku-Extreme-81`, `SATLIB 3-SAT`, `Sokoban`, `Countdown`).
  - Total neural forward calls must equal $0$.
  - Aggregate helper-only solve rate across all tasks must satisfy:
    $$\text{SolveRate}(\text{Track } 0) \le 5.0\%.$$

- **Gate `S13-Ledger` (`Symmetric Evaluator Helper & Node-Expansion Ledger`):**
  Across all compared paradigms $p \in \{1, \dots, P\}$ at equal budget $B$:
  - In **Track A (`Pure Neural`)**: helper call count $C_p^{\text{A}} = 0$ for every paradigm $p$.
  - In **Track B (`Symmetric 1-Pass Helper`)**: helper call count $C_p^{\text{B}}$ and external node expansions $E_p^{\text{B}}$ must have zero cross-paradigm spread:
    $$\max_p C_p^{\text{B}} - \min_p C_p^{\text{B}} = 0 \quad \text{and} \quad \max_p E_p^{\text{B}} - \min_p E_p^{\text{B}} = 0.$$

### 2.4 Rule `S14`: Paired Counterfactual Broadcast Price Replay Gate

- **Gate `S14-Detach` (`Zero Cross-Module Autograd & Peer-Perturbation Leakage`):**
  In decentralized market learning (`M4`/`M5` in `R23-E2`), module $m$'s local update receives broadcast pre-entry market price $P_t$.
  - $P_t$ must be a detached scalar tensor (`P_t.grad_fn is None`).
  - Perturbing peer modules' parameters/outputs $\theta_{-m} \leftarrow \theta_{-m} + \Delta\theta_{-m}$ while holding the broadcast pre-entry market price $P_t$ fixed must yield **zero gradient drift** on module $m$:
    $$\|g_m(P_t; \theta_{-m}) - g_m(P_t; \theta_{-m} + \Delta\theta_{-m})\|_2 \le 10^{-8}.$$

- **Gate `S14-PriceReplay` (`Counterfactual Broadcast Price Sensitivity`):**
  Replaying module $m$'s local update with identical local batch, weights $\theta_m$, and RNG seed, but replacing the broadcast pre-entry market price $P_t$ with a counterfactual price $P'_t = \operatorname{clip}(P_t + \delta_P, 0.05, 0.95)$ ($\delta_P = 0.20$), must strictly alter module $m$'s parameter gradient:
  $$\|g_m(P_t) - g_m(P'_t)\|_2 > 10^{-4}.$$

---

## 3. Pre-Registered Canary Matrix (`11/11` Total: `7/7` AST + `4/4` Runtime)

| Canary ID | Layer | Evasion Strategy Tested | Target Rule | Required Outcome |
| :--- | :--- | :--- | :--- | :--- |
| `AST-Canary-1..7` | Stage-1 Static AST | Direct oracle attribute/name/branch cheats (`S1`–`S10`) | `S1`–`S10` | `7/7 CAUGHT` |
| `S11-Canary` | Stage-3 Runtime | Benignly named canonical slot bias + global edge mean at $t=1$ (0 AST violations) | `S11` (`S11-E` + `S11-L`) | `CAUGHT` (`passed == False`) |
| `S12-Canary` | Stage-3 Runtime | Benignly named inert search loop + deterministic power-law multiplier (0 AST violations) | `S12` (`S12-2` + `S12-4`) | `CAUGHT` (`passed == False`) |
| `S13-Canary` | Stage-3 Runtime | Benignly named Track-0 classical solver ($64.58\%$ solve rate at $0$ neural calls) + asymmetric Track-B calls | `S13` (`Track0` + `Ledger`) | `CAUGHT` (`passed == False`) |
| `S14-Canary` | Stage-3 Runtime | Benignly named epiphenomenal price update ($0.0$ price gradient shift) + live peer gradient leak | `S14` (`Detach` + `PriceReplay`) | `CAUGHT` (`passed == False`) |

---

## 4. Pre-Registered Wave-1 vs. Wave-2 Falsification Predictions

1. **Wave-1 (`R18`–`R21`) Must Fail Both Stage-1 AST (`S1`–`S10`) and Stage-3 Runtime (`S11`–`S14`):**
   - `R21-E1` (`run_r21_clrs_ood_and_scaling.py`): Fails `S11-E` (embeds `idx_norm = torch.arange(n)/n` and DFS order in node features) and fails `S11-L` (precomputes global shortest paths `dist_vec` and `reach` at $t=0$ so $\Delta_{t=1}(s) > 10^{-2}$).
   - `R21-E2` (`run_r21_clrs_ood_and_scaling.py`): Fails `S12-2` ($\Delta_{t' > 2} = 0.0$ because $B > 1$ does not run neural search steps) and fails `S12-4` ($\operatorname{Std}_i(L_i(B)/L_i(1)) = 0.0$ due to analytic power-law scaling).
   - `R19` (`run_r19_public_reasoning_benchmarks.py`): Fails `S13-Track0` (`_complete_leaf_with_observable_search` solves $31/48 = 64.58\%$ of instances on a uniform prior with $0$ neural calls) and fails `S13-Ledger` (asymmetric helper calls and `mech_mult` scaling).
   - `R20` (`run_r20_kyle_lmsr_futarchy.py`): Fails `S14` (`P1_Local_Greedy` and `P5` at `epoch=0` with `warm_w=1.0` have $0.0$ counterfactual broadcast price sensitivity on market return terms, while `P5`/`P6` leak cross-module private anchor/gradient signals).
2. **Wave-2 (`R22A`, `R22B`, `R23`) Must Pass 100% of Stage-1 AST (`S1`–`S10`) and Stage-3 Runtime (`S11`–`S14`):**
   - `0` AST violations across all Wave-2 scripts (`R22A`, `R22B`, `R23`) and `R24A` audit modules.
   - `S11 PASS` on `R22B-E2` `CLRSNeuralProcessor` across all 5 architectures (`GLC_AOM_v2`, `MPNN_Max`, `Relational_Transformer`, `Pointer_Graph_Net`, `NAR_Triplet_GNN`) and all 6 CLRS tasks (`bfs`, `dfs`, `dijkstra`, `bellman_ford`, `mst_prim`, `floyd_warshall`).
   - `S12 PASS` on `R22B-E3` `ScaledSearchModel` (`150K`, `600K`, `2.4M`) and `R22A` `PureNeuralGameBackbone150K` tree search rollout.
   - `S13 PASS` on `R22B-E1` (`Track 0` solve rate $= 0/48 = 0.00\% \le 5.0\%$, Track A helper calls $= 0$, Track B helper call spread $= 0$ across all 7 paradigms).
   - `S14 PASS` on `R23-E2` decentralized market learners (`M4_Decentralized_Kyle_LMSR_Pinned_LIR` and `M5_Decentralized_Kyle_LMSR_Unpinned`: `grad_fn is None`, peer perturbation drift $= 0.000000000$, counterfactual price gradient shift $> 10^{-4}$).
