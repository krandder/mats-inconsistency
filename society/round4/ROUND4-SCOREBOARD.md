# Society of Thinkers — Round 4 Multi-Harness Synthesis: Search-Algorithm Expressivity (`Theorem R4.1`–`R4.3`), Supervised Algorithmic Imitation (`PREREG-R13`), and Unsupervised Search Discovery (`PREREG-R14`)

**Repository:** `/home/agent/workspaces/inconsistency-search` (`futarchy-fi/inconsistency-search`)  
**Date:** 2026-10-18  
**Coordinator:** Member 18 (`Gemini 4 Argon` — Round-4 Search-Algorithm Expressivity & Imitation Architect)  
**Participating Model Families & Round-4 Deliverables (`society/round4/` & `benchmarks/`):**
1. [`01-expressivity-theorems-and-mcts-gap.md`](01-expressivity-theorems-and-mcts-gap.md) — *Member 18 (`Gemini 4 Argon`)*: Line-by-line audit of `src/search/kv_tree_search.py`, proofs of `Theorem R4.1` (1-hop depth barrier & flat-attention topology blindness), `Theorem R4.2` (constant-depth MCTS-PUCT in `GLC-Search-Transformer v1`), and `Theorem R4.3` (the 5 expressivity barriers of `v1` and their resolution in `GLC-Search-Transformer v2`).
2. [`02-glm53-round4-expressivity.md`](02-glm53-round4-expressivity.md) — *Member 19 / 4a (`GLM-5.3` / `zai-coding-plan/glm-5.3` via `/home/agent/bin/glm`)*: Verification of `R4.1`–`R4.3`, 12 edge-case refinements (`E1`–`E12`), Kleene-star Boolean DAG closure + visit-weighted MCGS fix (`V2.2′`), exact literal-bitmask nogood subsumption (`V2.5′`), unified per-step decision-record imitation specification (`R13`), and Mechanistic Algorithmic Alignment Probes (`MAAP`, `R14`).
3. [`03-gpt6-astra-round4-expressivity.md`](03-gpt6-astra-round4-expressivity.md) — *Member 4b (`GPT-6-Astra` via `codex-any` on `127.0.0.1:7810`)*: Distinction between occurrence nodes, directed action edges, and simulation events (`A_{e,t}`), separation of **Constructive Capability (`CW`)**, **Teacher-Forced Learnability**, and **Free-Running Execution Fidelity** in `R13`, perfect-recall infoset equivalence and hidden-state permutation invariance in `CFR+`, and causal ablation probes in `R14`.
4. [`04-opus-round4-expressivity.md`](04-opus-round4-expressivity.md) — *Member 4c (`Claude Opus 5.5 [1m]` via `claude-any` on `127.0.0.1:7800`)*: Proof of `Theorem R4.1(b′)` (radius-$L_1$ ego-neighborhood tail-swap lower bound) and `Theorem R4.2′` ($\varepsilon$-greedy PUCT equivalence with conservative $\mathrm{LSE}_\beta$ bounds), frontier action tokens (`E3`), complete catalog of residual `v3` search families (`RAVE/GRAVE`, `Iterative Deepening/IDA*`, `CDCL Restarts`, `NRPA`, `DeepStack/ReBeL` inner loops), and Geiger-style Causal Interchange Intervention (`IIA`) alignment probes for `R14`.
5. [`benchmarks/PREREG-R13-ALGO-IMITATION.md`](../../benchmarks/PREREG-R13-ALGO-IMITATION.md) — *Locked Round-13 Pre-Registration Protocol (`R13-ALGO-IMITATION`)*: Constructive + Supervised Imitation across all 6 canonical search algorithm families (`R13-D0` through `R13-D6`).
6. [`benchmarks/PREREG-R14-UNSUP-DISCOVERY.md`](../../benchmarks/PREREG-R14-UNSUP-DISCOVERY.md) — *Locked Round-14 Pre-Registration Protocol (`R14-UNSUP-DISCOVERY`)*: Unsupervised Task-Driven Search Discovery & Mechanistic Algorithmic Alignment (`R14-D0` through `R14-D6`).

---

## 0. Fleet & Proxy Health Verification Log (Policies 3, 6 & 15)

Prior to launching the bounded headless Round-4 consultations, all provider gateways were verified read-only:
- **Check Command:** `/home/agent/bin/tokens accounts | grep -v "pin:"` (checked at `2026-10-18T14:17:00Z`).
- **Claude / Prime Proxy (`127.0.0.1:7800`):** Healthy (`serial 464`; `ai-canary` `5h 11% used, 7d 8% used`; `robin` `5h 24% used, 7d 19% used`). Zero restarts or profile mutations performed (Fleet Policy 6). Consultation [`04-opus-round4-expressivity.md`](04-opus-round4-expressivity.md) ran headlessly (`CLAUDE_CODE_NO_MODEL_FALLBACK=1 claude-any --model "claude-opus-5-5[1m]" --print --bare --tools "" --no-session-persistence`) with exit code `0` (`58,209` bytes, `735` lines).
- **Codex Proxy (`127.0.0.1:7810`):** Healthy (`serial 464`; `kas-codex` `13%`, `adriana-codex` `16%`, `atlas` `16%`, `vega` `16%`, `nova` `22%`, `daniel` `24%`). Consultation [`03-gpt6-astra-round4-expressivity.md`](03-gpt6-astra-round4-expressivity.md) ran headlessly via `codex-any exec -m gpt-6-astra --ephemeral -s read-only` with exit code `0` (`55,903` bytes, `1,176` lines).
- **GLM-5.3 (`/home/agent/bin/glm`):** Healthy (`opencode`, model `zai-coding-plan/glm-5.3`). Consultation [`02-glm53-round4-expressivity.md`](02-glm53-round4-expressivity.md) ran headlessly with exit code `0` (`41,344` bytes, `230` lines).

---

## 1. Consensus Answers to Kelvin's Two Core Questions

### 1.1 Question 1: Can Our `R11`/`R12` Algorithms Learn MCTS Subtree Aggregation?
**Unanimous 4-Model Answer:**
1. **As shipped in `R11`/`R12` (`src/search/kv_tree_search.py`): NO (`Theorem R4.1` & `R4.1(b′)`).**
   - `TreePagedKVBank.append_node` (`kv_tree_search.py:238, 290–295`) updated `n_exp` only on the 1-hop immediate parent $\mathrm{pa}(b)$ (counting immediate children $|\mathrm{Ch}(j)|$ rather than subtree visits $N(j) = |\mathrm{Desc}(j)|$) and wrote `q_tree_bank[:, b_next]` only once at node creation with **zero backpropagation to ancestors**.
   - `ResolvableInconsistencyTreePUCT.select_node` (`kv_tree_search.py:691–756`) is a 0-parameter hardcoded Python formula (`target_seg_len = 6`, `b // 6`).
   - `TwoStageTreeClearer` (`kv_tree_search.py:895–1099`) combines a 1-hop Stage-1 parent-child mixture ($L_1 = 1$) with an unmasked Stage-2 flat edge-key attention layer. By `Theorem R4.1(b′)`, swapping the opposite-valued leaf tails ($\pm 1$) of two chains of length $\ell > 2L_1 + 1 = 3$ beneath symmetric depth-1 siblings preserves every radius-$L_1$ ego-neighborhood and edge-key multiset identically while flipping $Q^{\text{MCTS}}(j_1)$ from $+1$ to $-1$ (worst-case error $\ge 1/2$, branch selection accuracy $= 50\%$).
2. **With `GLC-Search-Transformer v1` (`Theorem R4.2` & `R4.2′`): YES, in $O(1)$ layers ($L=3$) at any tree depth $D \le B$.**
   - `TreePagedKVBank` (`kv_tree_search.py:241–242`) **already maintains** the exact reflexive ancestor bitmask $M_{\text{tree}}[k, j] = \mathbf{1}[j \in \mathrm{Anc}(k)]$ in $O(b)$ bit operations per step. Its transpose $\mathcal{M}_{\text{desc}}[j, k] = M_{\text{tree}}[k, j] = \mathbf{1}[k \in \mathrm{Desc}(j)]$ is the exact transitive descendant mask.
   - Because MCTS subtree value/visit aggregation is a **commutative monoid** over $\mathrm{Desc}(j)$, a single layer with $\mathcal{M}_{\text{desc}}$ ($\beta=0$ uniform mean with two-player parity sign $s_j s_k v_k$), exact integer/fp32 count $\log N(j) = \log \sum_k \mathcal{M}_{\text{desc}}[j, k]$, and parallel $\mathcal{M}_{\text{parent}}$ fetch computes $(Q^{\text{MCTS}}(j), \log N(j), \log N(\mathrm{pa}(j)))$ at **Layer 1**; Layer 2 computes $U(j)$ and conservative LSE sibling margin $\hat{\mu}_{\text{edge}}(j) = \hat{U}(j) - \mathrm{LSE}_\beta(\{\hat{U}(s)\}_{s \in \mathrm{Sib}(j)})$; and Layer 3 computes the ancestor path bottleneck $\hat{S}_{\text{path}}(j) = -\mathrm{LSE}_\beta(\{-\hat{\mu}_{\text{edge}}(a)\}_{a \in \mathrm{Anc}^\circ(j)})$, which is provably $\varepsilon$-greedy PUCT with **zero false positives**!

---

### 1.2 Question 2: What Known Multi-Game Search Algorithms Can Even `GLC-Search v1` NOT Represent — and How Does `GLC-Search v2` (and `v3`) Address Them?

Across all four Society members, we established **two tiers of expressivity boundaries**:
- **Tier A (`Theorem R4.3` — The 5 Expressivity Barriers of `GLC-Search v1` Fixed in `GLC-Search v2` via `V2.1′`–`V2.5′`):** Algorithms that fail in `v1` (fixed $L=3$ layers, syntactic tree masks only, symmetric siblings, no persistent spine/infoset state) and are **resolved in `GLC-Search-Transformer v2`**.
- **Tier B (Residual `v3` Horizon — Multi-Game Algorithms Requiring Non-Append-Only / Action-Hash / Inner-Loop Primitives):** Algorithms that remain outside even `GLC-Search v2` and define the exact minimal `v3` extension roadmap.

#### Table 1.1: Complete Synthesis of `Theorem R4.3` (`GLC-Search v1` Barriers & Society-Hardened `GLC-Search v2` Fixes `V2.1′`–`V2.5′`)

| Barrier & Canonical Search Family | Why `GLC-Search v1` Fails (`Theorem R4.3`) | Initial `v2` Proposal (`01`) | Society-Hardened `v2` Specification (`V2.1′`–`V2.5′` from `GLM-5.3`, `GPT-6-Astra`, `Opus 5.5`) |
| :--- | :--- | :--- | :--- |
| **Barrier 1 (`Algo 2`):** Arbitrary-Depth Alternating Minimax / Alpha-Beta & Proof-Number Search (`PNS` / `df-pn`) at $D > L$ | Alternating $(\max, \min)$ and $(\min, \sum)$ tree evaluation is $\mathsf{NC}^1$-complete whereas fixed-$L$ log-precision Transformers lie in $\mathsf{TC}^0$; symmetric $\mathcal{M}_{\text{sib}}$ cannot represent left-to-right $\alpha\text{--}\beta$ windows. | `V2.1`: Incremental ancestor-spine backup + directed $\mathcal{M}_{\text{sib}<}$. | **`V2.1′`:** Recompute $\mathrm{backup}[a] = \mathrm{Agg}_\theta(\{\mathrm{backup}[c]\}_{c \in \mathrm{Ch}(a)}, z_a)$ over **all children** along the active spine $a \in \mathrm{Anc}(b)$ in $O(\sum_{a \in \mathrm{Anc}(b)} m_a) \le O(b)$ (since $\max/\min$ is non-invertible when a child value drops); saturating $\log(1+\mathrm{pn})$ arithmetic; 2-hop $\mathcal{M}_{\text{anc}} \circ \mathcal{M}_{\text{sib}<}$ for $(\alpha, \beta)$ bounds. |
| **Barrier 2 (`Algo 4`):** Transposition Tables & Monte Carlo Graph Search (`KataGo MCGS`) | Syntactic tree $\mathcal{M}_{\text{desc}}$ misses commuted move sequences $(a_1, b_1, a_2) \equiv (a_2, b_1, a_1)$ reaching identical world states $\mathrm{canon}(O_j) = \mathrm{canon}(O_k)$ in cousin branches. | `V2.2`: $\mathcal{M}_{\text{desc}}^{\text{DAG}} = E_{\text{trans}} \cdot \mathcal{M}_{\text{desc}}$. | **`V2.2′`:** Raw matrix product $E_{\text{trans}} \mathcal{M}_{\text{desc}}$ double-counts multi-path nodes and takes only 1 hop! Fix: (i) **Boolean Kleene-star closure** $\mathcal{M}_{\text{desc}}^{\text{DAG}} = (E_{\text{tree}} \lor E_{\text{trans}})^*$ using **exact** ply/history-augmented canonical equality for value/visit backup (soft $\exp(-\|h_j-h_k\|^2/\tau)$ restricted strictly to prior/selection channel); (ii) **edge-visit-weighted MCGS backup** $Q(n) = \frac{U(n) + \sum_a N(n,a)Q(c(n,a))}{1 + \sum_a N(n,a)}$ via `V2.1′`. |
| **Barrier 3 (`Algo 5`):** Imperfect-Info `CFR+` Regret Matching (`Proposal-Poker`, `Micro-Stratego`, `Galanis-Market`) | `v1` lacks information-set pooling across indistinguishable histories, opponent counterfactual reach $\pi_{-i}(h)$, and mixed-strategy Nash output. | `V2.3`: $\mathcal{M}_{\text{infoset}}$ + ancestor log-reach + $\mathrm{ReLU}(R_+)$ head. | **`V2.3′`:** One forward pass is only 1-step regret matching, not `CFR+`! Fix: (i) Key $\mathcal{M}_{\text{infoset}}$ on the acting player's **perfect-recall observation-action history hash**; (ii) maintain persistent per-infoset table $(R_+^{(t)}(I,a), \Sigma^{(t)}(I,a))$ updated at each step with linear-weighted **average strategy** readout $\bar{\sigma}^{(T)}(I,a) \propto \sum_{t=1}^T t \pi_i^{(t)}(I)\sigma^{(t)}(I,a)$; (iii) split signed utility $u = u^+ - u^-$ in log-reach LSE; (iv) enforce **Infoset Hidden-Permutation Invariance** $\le 10^{-6}$. |
| **Barrier 4 (`Algo 6`):** Gumbel MuZero with Sequential Halving (`SH`) at Budgets $B \in [2, 32]$ | Softmax sibling attention depends on score magnitudes and cannot compute ordinal rank $\mathrm{rank}_{\text{sib}}(j) \le \lceil m/2^p \rceil$ for halving elimination. | `V2.4`: Pairwise sibling rank-percentile head $\widehat{\mathrm{pct}}_{\text{sib}}(j)$. | **`V2.4′`:** Certified temperature bound $\tau_{\text{rank}} < \frac{\Delta}{2\log(2m)}$, lexicographic tie-break via $\mathcal{M}_{\text{sib}<}$, frontier tokens for unvisited actions with completed-$Q$ transform $(c_{\text{visit}} + \max_b N(b)) c_{\text{scale}} \hat{q}$, and recorded teacher Gumbel noise $g_j$ in `R13`. |
| **Barrier 5 (`Algo 3`):** Global Conflict-Clause / Nogood Broadcast in `CDCL` (`Sudoku`, `Equiv-SAT`) | `v1`'s $\mathcal{M}_{\text{anc}}$ only backjumps to ancestors of conflict leaf $b$, failing to prune **disjoint** subtrees $j \notin \mathrm{Anc}(b)$ that repeat the falsified core. | `V2.5`: Global conflict cross-attention $\mathcal{M}_{\text{nogood}}[j, k]$. | **`V2.5′`:** Attending to "conflict happened at $k$" is not a clause subsumption check! Fix: maintain incremental path literal bitmask $L[j, \ell] = L[\mathrm{pa}(j), \ell] \lor \mathbf{e}_{\ell(j)}$ and exact subset test $\langle e(C_j), e(\mathrm{core}_k) \rangle - |\mathrm{core}_k| = 0 \iff \mathrm{core}_k \subseteq C_j$ masked by $\mathcal{M}_{\text{nogood}}$, guaranteeing **0 unsound prunes**. |

#### Table 1.2: Tier B — Multi-Game Search Algorithms That Even `GLC-Search-Transformer v2` Cannot Represent (Deferred to `v3`)

| Search Family & Canonical Implementations | Games Where Used | Why Even `GLC-Search v2` Cannot Represent It | Minimal `GLC-Search v3` Primitive |
| :--- | :--- | :--- | :--- |
| **1. Action-Identity Cross-Branch Heuristics (`RAVE` / `GRAVE`, `MAST`, Killer & History Heuristic)** | General Game Playing (`Ludii`, `GGP`), Go, Alpha-Beta Chess/Shogi | Pools returns of move $a$ played *anywhere later* in simulations (`AMAF`) or across disjoint branches at the same ply (`killer/history`). `v2` masks are keyed by tree/DAG node identity or state equivalence, not **action identity** $a(j) = a(k)$. | **`V3.1` Action-Identity Masks:** $\mathcal{M}_{\text{act}}[j, k] = \mathbf{1}[a(j) = a(k)]$ and $\mathcal{M}_{\text{amaf}}[j, k] = \mathbf{1}[k \in \mathrm{Desc}(\mathrm{pa}(j)) \land a(j) \in \text{path}(\mathrm{pa}(j)\to k)]$. |
| **2. Non-Append-Only / Iterative Re-Search & Restarts (`Iterative Deepening`, `MTD(f)`, `IDA*`, `CDCL Restarts + Phase Saving`)** | Chess (`Stockfish`), Sokoban, 15-Puzzle, Tsume-Shogi, SAT (`Kissat`) | `TreePagedKVBank` is strictly append-only ($b = 0 \to B$); it cannot discard the search tree while retaining a compressed transposition/clause/phase table to restart from root with a tighter threshold. | **`V3.2` Tree-Reset & Persistent Summary Op:** Allows search controller to emit `RESET_TREE` while preserving `backup_state` / `nogood_bank` / `infoset_bank`. |
| **3. Unbounded Inner Fixed-Point / Continual Re-Solving (`DeepStack`, `ReBeL` PBS Gadget, Retrograde Tablebases)** | Heads-Up No-Limit Poker, Liar's Dice, Cyclic Endgames | Requires solving an inner equilibrium fixed-point ($T \gg 1$ CFR iterations on a public-belief-state gadget) *within a single search step*, whereas `v2` performs one $L$-layer pass per node expansion. | **`V3.3` Adaptive-Iteration Inner Looped Block:** Weight-tied inner equilibrium solver loop with halting gate. |
| **4. In-Search Fast-Weight Policy Adaptation (`NRPA` — Nested Rollout Policy Adaptation)** | Morpion Solitaire, SameGame, Combinatorial Optimization | Updates rollout policy weights $\theta_{\text{playout}}$ via gradient ascent *inside* nested search levels. | **`V3.4` Fast-Weight Episodic Policy Slot:** Low-rank rollout adapter updated online per rollout trajectory. |

---

## 2. Multi-Model Society Rubric Evaluation (`D1`–`D6`) & Pre-Registered `[PRIOR]` Forecasts for `R13` and `R14`

### Table 2.1: 4-Model Society Rubric Trajectory (`R3 Realized-R12` $\to$ `R4 Expressivity-Risk` $\to$ `Proj-R13` $\to$ `Proj-R14`)

| Charter Dimension | Model Family | `R3 Realized-R12` (`NVIDIA L4`, $B \le 256$) | `R4 Expressivity-Risk` (`R12` without `M_desc` / `v2`) | **`Proj-R13` (`GLCSearch v2` Supervised Imitation)** | **`Proj-R14` (`GLCSearch v2` Unsupervised Discovery)** | Key Rationale Across Models |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **D1. Universal KV-Cache Tree Machine & Integrity** | `GLM-5.3` (`02`) | `6.50` | `7.50` | `8.50` | `8.50` | `R12` had 0-param selector & 1-hop `n_exp`; `v2` makes tree search genuinely neural + `CW` verifiable. |
| | `GPT-6-Astra` (`03`) | `8.00` | `8.00` | `8.00` | `8.70` | Strong `R12` outcomes, but `R13` tests representation/learning before `R14` lifts task outcomes. |
| | `Claude Opus 5.5` (`04`) | `9.00` | `8.50` | `9.00` | `8.50` | `0.000000000` leak/KV bars passed; new `v2` surfaces (`E_trans`, `infoset` permutation) gated in `D0`. |
| | `Gemini 4 Argon` (`01`) | `8.85` | `6.80` | `9.10` | `9.30` | `Theorem R4.1` exposes `R12`'s 0-backup gap; `Theorem R4.2′/R4.3` closes it in $O(b)$ incremental state. |
| | **D1 Consensus Mean** | **`8.09`** | **`7.70`** | **`8.65`** | **`8.75`** | **`+0.95` (`R13`) / `+1.05` (`R14`) lift over `R4 Risk`** |
| **D2. Algorithmic Expressivity & Active Plans (`Algo 1`–`Algo 6`)** | `GLM-5.3` (`02`) | `3.00` | `4.00` | `8.50` | `8.50` | `R12` could not represent MCTS, Minimax, CDCL, MCGS, CFR+, or SH; `v2` represents all 6. |
| | `GPT-6-Astra` (`03`) | `4.00` | `2.50` | `7.50` | `8.00` | Sharpest penalty on `R12`'s hardcoded selector (`2.50`); `+5.00` jump once `R13` verifies `Algo 1`–`6`. |
| | `Claude Opus 5.5` (`04`) | `8.00` | `7.00` | `8.00` | `8.50` | Frontier tokens + `V2.1′`–`V2.5′` turn readout-driven deduction into true search-driven deduction. |
| | `Gemini 4 Argon` (`01`) | `8.60` | `5.20` | `9.10` | `9.25` | `C3_GLCSearch_v2` unifies all 6 canonical search families in a single `<64K`-param module. |
| | **D2 Consensus Mean** | **`5.90`** | **`4.68`** | **`8.28`** | **`8.56`** | **`+3.60` (`R13`) / `+3.88` (`R14`) expressivity jump** |
| **D3. Learnability, Sample Efficiency & Lemma Calibration** | `GLM-5.3` (`02`) | `9.00` | `9.00` | `9.00` | `9.00` | `N_train in {128, 512, 2048}` sample-efficiency ladder + deterministic decision records. |
| | `GPT-6-Astra` (`03`) | `8.00` | `7.00` | `7.00` | `7.00` | Distinguishes `CW` (constructive weights), teacher-forced imitation, and free-running rollouts. |
| | `Claude Opus 5.5` (`04`) | `5.50` | `4.00` | `6.00` | `7.50` | `LSO` counterfactual proper-score credit directly targets multi-witness partial observability. |
| | `Gemini 4 Argon` (`01`) | `8.65` | `7.50` | `9.00` | `9.15` | `R13 -> R14` warm-start vs. cold-start ablation quantifies the exact value of imitation pretraining. |
| | **D3 Consensus Mean** | **`7.79`** | **`6.88`** | **`7.75`** | **`8.16`** | **`+0.87` (`R13`) / `+1.28` (`R14`) lift** |
| **D4. Present-Time Memory, Infoset `CFR+` & `LIR` Firewall** | `GLM-5.3` (`02`) | `7.50` | `7.50` | `7.00` | `7.00` | Hard infoset equality + opponent-card probe `AUC <= 0.55` prevents strategy fusion. |
| | `GPT-6-Astra` (`03`) | `6.50` | `4.50` | `7.00` | `7.50` | Perfect-recall observation-action history keys + OOD depth/budget generalization. |
| | `Claude Opus 5.5` (`04`) | `7.50` | `6.00` | `7.50` | `8.00` | Persistent per-infoset `(R_+, \bar{\sigma})` memory + hidden-state permutation canary `<= 1e-6`. |
| | `Gemini 4 Argon` (`01`) | `8.80` | `7.20` | `9.00` | `9.20` | Preserves `0.000000` `LIR` belief firewall while enabling mixed-strategy Nash play on Poker/Stratego. |
| | **D4 Consensus Mean** | **`7.58`** | **`6.30`** | **`7.63`** | **`7.93`** | **`+1.33` (`R13`) / `+1.63` (`R14`) lift** |
| **D5. Anytime Monotonicity & $B=128 \to 256$ Multi-Witness Fix** | `GLM-5.3` (`02`) | `5.50` | `5.00` | `7.50` | `8.00` | `90%` prior probability that `V2.2′` + `LSO` fixes the `R12` $B=128 \to 256$ KL uptick (`0.1294 -> 0.1387`). |
| | `GPT-6-Astra` (`03`) | `8.00` | `6.50` | `8.00` | `8.50` | Separates pure tree-topology reachability from edge-visit-weighted MCGS and multi-witness Bayes. |
| | `Claude Opus 5.5` (`04`) | `8.00` | `6.50` | `7.50` | `8.00` | `LSO` assigns $I > 0$ to a second distinct witness and $I \approx 0$ to redundant duplicates of witness 1. |
| | `Gemini 4 Argon` (`01`) | `8.45` | `6.90` | `8.90` | `9.20` | Subtree visit/witness normalization stops single-branch over-reinforcement at $B=256$. |
| | **D5 Consensus Mean** | **`7.49`** | **`6.23`** | **`7.98`** | **`8.43`** | **`+1.75` (`R13`) / `+2.20` (`R14`) lift** |
| **D6. Mechanistic Discovery, `Mini-ECI` & Multiplayer Elo** | `GLM-5.3` (`02`) | `2.00` | `2.00` | `3.00` | `7.00` | `MAAP` alignment probes make both emergent alignment and novel non-classical search falsifiable. |
| | `GPT-6-Astra` (`03`) | `7.50` | `6.50` | `8.50` | `8.50` | Decision table cleanly classifies capability vs. optimization vs. execution vs. discovery. |
| | `Claude Opus 5.5` (`04`) | `3.50` | `2.50` | `7.50` | `6.50` | Constructive weights (`CW`) + learned imitation (`R13`) + causal interchange (`IIA`) in `R14`. |
| | `Gemini 4 Argon` (`01`) | `8.55` | `6.50` | `8.90` | `9.20` | Tests whether task-trained `GLCSearch v2` beats every individual fixed teacher algorithm on `Mini-ECI`. |
| | **D6 Consensus Mean** | **`5.39`** | **`4.38`** | **`6.98`** | **`7.80`** | **`+2.60` (`R13`) / `+3.42` (`R14`) lift** |
| **Overall 6-Dimension Unweighted Mean** | **4-Model Consensus** | **`7.04`** | **`6.03`** | **`7.88`** | **`8.27`** | **`+1.85` (`R13`) / `+2.24` (`R14`) net lift over `R4 Expressivity-Risk`** |

---

### Table 2.2: Pre-Registered 4-Model `[PRIOR]` Forecasts for `R13-ALGO-IMITATION` and `R14-UNSUP-DISCOVERY`

| Benchmark & Target Metric | Pre-Registered Bar | `GLM-5.3` `[PRIOR]` (80% CI) | `GPT-6-Astra` `[PRIOR]` (80% CI) | `Claude Opus 5.5` `[PRIOR]` (80% CI) | `Gemini 4 Argon` `[PRIOR]` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`R13-D0f`:** Constructive Weights (`CW`) Match on Margin Trees (`Algo 1`–`6`) | `>= 99.5%` | `99.8% – 100.0%` | `99.5% – 100.0%` | `99.5% – 100.0%` (after `CW` debug) | `99.9% – 100.0%` |
| **`R13-D1`:** `Algo 1 (MCTS-PUCT)` `Full-v2` ID / OOD Selection Acc & $Q$ $R^2$ | `>= 95%` ID / `>= 88%` OOD, $R^2 \ge 0.98$ | `99.5–100%` / $R^2$ `0.999+` | `98–99.9%` ID / `93–99%` OOD | `95–99%` ID / `87–97%` OOD ($R^2$ `0.97–0.995`) | `97–99.5%` ID / `92–97%` OOD |
| **`R13-D1`:** `Abl-R12-1Hop-Flat` OOD Selection Acc & $Q$ $R^2$ (`Theorem R4.1`) | `<= 65%` Acc, $R^2 \le 0.60$ | `55–75%` Acc / $R^2$ `0.60–0.85` | `35–65%` Acc | `35–60%` OOD Acc / $R^2$ `0.40–0.70` | `48–62%` OOD Acc / $R^2$ `0.35–0.55` |
| **`R13-D2`:** `Algo 2 (Deep Minimax / α-β / PNS)` `Full-v2` vs. `Abl-v1` at $D \ge 10$ | `v2 - v1 >= +10 pp` / `+0.15` $R^2$ | $R^2$ `0.99+` (`v2`) vs `0.75–0.90` (`v1`) | `93–99%` ID / `75–95%` OOD | `90–98%` (`v2`) vs `50–70%` (`v1` OOD) | `94–98%` (`v2`) vs `60–75%` (`v1`) |
| **`R13-D3`:** `Algo 3 (CDCL Global Nogood)` `Full-v2` Backjump & Nogood F1 | `>= 92%` F1, `0` unsound prunes | `93–99%` Backjump / `85–95%` F1 | `88–98%` ID / `65–91%` OOD | `86–96%` ID / `78–92%` OOD (`0` unsound) | `92–97%` F1 (`0` unsound via `V2.5′`) |
| **`R13-D4`:** `Algo 4 (MCGS DAG + Disjunctive BP)` `Full-v2` $Q$ $R^2$ | $R^2 \ge 0.95$ ID / $\ge 0.90$ OOD | `96–99.5%` SelAcc / $E_{\text{trans}}$ `0.93–0.99` | `95–99.5%` ID / $R^2$ `0.95–0.998` | $R^2$ `0.90–0.98` ID / `0.84–0.95` OOD | $R^2$ `0.96–0.99` ID / `0.91–0.97` OOD |
| **`R13-D5`:** `Algo 5 (CFR+ Infoset Regret Matching)` Strategy KL & Exploitability | $\mathrm{KL} \le 0.02$, Expl $\le 1.5\times$, Perm $\le 10^{-6}$ | $\mathrm{KL}$ `0.010–0.030`, Expl `1.0–2.0x` | `94–99.5%` ID / $R^2$ `0.94–0.995` | Expl `1.05–1.6x` ID (`v1`: `2–5x`) | $\mathrm{KL}$ `0.008–0.018`, Perm `0.000000` |
| **`R13-D6`:** `Algo 6 (Gumbel Sequential Halving)` Phase Survivor Exact Match | `>= 93%` ID / `>= 88%` OOD | `88–97%` SetEM / `99.5%+` final action | `98–99.9%` ID / `94–99.5%` OOD | `90–98%` ID / `83–95%` OOD (`v1`: `55–80%`) | `94–98%` ID / `90–96%` OOD |
| **`R14-D1/D2`:** 12-Env Mean KL at $B=256$ (Fixing `R12`'s `0.1294 -> 0.1387` Regression) | `KL(256) < KL(128) <= 0.1294` | `0.112 – 0.126` (`90%` prob fix) | `0.102 – 0.145` (median `0.118`, $\Delta = -0.004$) | `0.115 – 0.136` (`60%` prob $\le 0.1294$) | `0.114 – 0.124` (monotonic $B=0\to 256$) |
| **`R14-D1/D6`:** `GLCSearch-v2-Unsup` `Mini-ECI` at $B=256$ (vs. `R12` `106.53`) | `>= 107.53` (`+1.0` over `R12`) | `104.5 – 106.2` (at $B=128$) | `104.0 – 114.0` (median `109.5`) | `106.8 – 109.3` (median `107.9`) | `108.0 – 110.5` |
| **`R14-D3/D4`:** Multiplayer Elo Gain at $B=256$ over `R12` (`1205.1` Elo) | `>= +45` to `+75` Elo | `+50` to `+250` Elo | `-60` to `+240` Elo (median `+90`) | `+40` to `+220` Elo (median `+110`) | `+55` to `+140` Elo |
| **`R14-D5`:** Spontaneous Mechanistic Alignment (`MAAP` / `IIA`) Across Families | Passes in $\ge 3/4$ domain families | `30–45%` full strong / `80%+` partial | `50–75%` in $\ge 2$ groups | `45%` full D5 (`IIA` `0.55–0.85` on MCTS) | `65–80%` domain-family dissociation |
