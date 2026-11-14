# Pre-Registration Protocol: Round 13 (`R13-ALGO-IMITATION`) — Constructive & Supervised Search-Algorithm Imitation Benchmark for `GLC-Search-Transformer v2` Across Six Canonical Search Families

- **Document ID:** `PREREG-R13-ALGO-IMITATION`
- **Status:** **LOCKED PRIOR TO ROUND-13 EXECUTION**
- **Date:** 2026-10-18
- **Repository:** `/home/agent/workspaces/inconsistency-search` (`futarchy-fi/inconsistency-search`)
- **Governing Theory & Multi-Model Society Audits (`society/round4/`):**
  - [`society/round4/01-expressivity-theorems-and-mcts-gap.md`](../society/round4/01-expressivity-theorems-and-mcts-gap.md) (Member 18 `Gemini 4 Argon`: `Theorem R4.1`, `Theorem R4.2`, `Theorem R4.3`)
  - [`society/round4/02-glm53-round4-expressivity.md`](../society/round4/02-glm53-round4-expressivity.md) (Member 19/4a `GLM-5.3`: `E1`–`E12` Refinements, Kleene-Star `V2.2′`, Literal-Bitmask `V2.5′`, Decision-Record Imitation)
  - [`society/round4/03-gpt6-astra-round4-expressivity.md`](../society/round4/03-gpt6-astra-round4-expressivity.md) (Member 4b `GPT-6-Astra`: Constructive vs. Teacher-Forced vs. Free-Running Separation, Perfect-Recall `CFR+` Firewall)
  - [`society/round4/04-opus-round4-expressivity.md`](../society/round4/04-opus-round4-expressivity.md) (Member 4c `Claude Opus 5.5`: `Theorem R4.1(b′)`, `Theorem R4.2′` $\varepsilon$-PUCT, Frontier Tokens `E3`, `CW` Representability Gate, Residual `v3` Table)
  - [`society/round4/ROUND4-SCOREBOARD.md`](../society/round4/ROUND4-SCOREBOARD.md) (Round-4 4-Model Synthesis & `[PRIOR]` Ledger)
- **Companion Unsupervised Discovery Protocol:** [`benchmarks/PREREG-R14-UNSUP-DISCOVERY.md`](PREREG-R14-UNSUP-DISCOVERY.md)

---

## 0. Scientific Motivation & Kelvin's Two Core Questions

Kelvin formulated the two foundational questions governing Round 13 (`R13-ALGO-IMITATION`) and Round 14 (`R14-UNSUP-DISCOVERY`):

1. **Question 1 (MCTS Subtree Aggregation & Supervised Imitation vs. Unsupervised Discovery):**
   *"One thing that I am unsure about is to what extent our general algorithms are really capable of implementing and learning things like MCTS. E.g. in MCTS, the values of all descended nodes get aggregated/summed on parents. I dont see how our current algorithms can learn that? Similarly for other search algorithms. I think we should have of course the general algorithm learn the best algorithm for each case without supervision on specific algorithmic implementations. But I think it would be a good test and intermediate benchmark, to also train specific supervision on specific search algorithms! This will tell and iron/out whether the algorithm is capable (can represent known search algorithms), but also whether they can learn it efficiently!"*
2. **Question 2 (What Search Algorithms Can Even `GLCSearch v1` NOT Represent?):**
   *"Are there other search algorithms that even this new GLCSearch can't represent? Check known implementations that are useful for several games and tell me"*

### 0.1 Society Consensus on `Theorem R4.1`–`R4.3` & the Hardened `GLC-Search-Transformer v2` Specification (`V2.1′`–`V2.5′`)
Per the unanimous 4-model Round-4 audit ([`society/round4/ROUND4-SCOREBOARD.md`](../society/round4/ROUND4-SCOREBOARD.md)):
1. **`Theorem R4.1` & `R4.1(b′)` (Why `R11`/`R12` Cannot Represent MCTS):** In `src/search/kv_tree_search.py`, `n_exp` (`line 238`) counts only 1-hop immediate children $|\mathrm{Ch}(j)|$, `q_tree_bank` (`lines 290–295`) is written once at creation with zero ancestor backup, `ResolvableInconsistencyTreePUCT.select_node` (`lines 691–756`) has `0` trainable parameters, and Stage-2 clearing (`lines 895–965`) attends over flat unmasked edge keys. Any $L$-layer 1-hop network has zero gradient beyond relative depth $\Delta d > L$, and swapping the opposite-valued leaf tails ($\pm 1$) of chains of length $\ell > 2L_1 + 1 = 3$ beneath symmetric depth-1 siblings leaves every radius-$L_1$ ego-neighborhood and edge-key multiset identical while flipping $Q^{\text{MCTS}}(j_1)$ from $+1$ to $-1$ (worst-case error $\ge 1/2$, branch selection accuracy $= 50.0\%$).
2. **`Theorem R4.2` & `R4.2′` ($\varepsilon$-Exact MCTS-PUCT in $L=3$ Layers in `GLC-Search v1`):** Exposing the transpose of `TreePagedKVBank.tree_mask` (`kv_tree_search.py:241–242`) as the reflexive descendant mask $\mathcal{M}_{\text{desc}}[j, k] = \mathbf{1}[k \in \mathrm{Desc}(j)]$ with two-player parity sign $s_j s_k v_k$ ($s_k = (-1)^{d(k)}$, `E1`), multiplicity counter $c_k$ (`E2`), **frontier candidate-action tokens** (`E3`), null-key fallback for empty masks (`E5`), directed sibling tie-breaking $\mathcal{M}_{\text{sib}<}$ (`E6`), and conservative log-sum-exp sibling/ancestor bounds $\mathrm{LSE}_\beta$ (`E7`) computes exact MCTS subtree means $Q^{\text{MCTS}}(j)$, exact log visit counts $\log N(j) = \log \sum_k \mathcal{M}_{\text{desc}}[j, k] c_k$, and $\varepsilon$-greedy PUCT root-to-leaf selection with **zero false positives** at any tree depth $D \le B$ in $L=3$ layers!
3. **`Theorem R4.3` & Society-Hardened `V2.1′`–`V2.5′` (Closing the 5 Expressivity Barriers of `GLC-Search v1`):**
   - **`V2.1′` (Barrier 1 — Deep Alternating Minimax / Alpha-Beta & Proof-Number Search `PNS`/`df-pn` at $D > L$):** Recomputes $\mathrm{backup}[a] = \mathrm{Agg}_\theta(\{\mathrm{backup}[c]\}_{c \in \mathrm{Ch}(a)}, z_a)$ over all children along the active ancestor spine $a \in \mathrm{Anc}(b)$ in $O(\sum_{a \in \mathrm{Anc}(b)} m_a) \le O(b)$ operations per step using a learned gated semiring cell (`mean`, `LSE_{+beta}`, `LSE_{-beta}`, saturating `log-sum`), plus 2-hop $\mathcal{M}_{\text{anc}} \circ \mathcal{M}_{\text{sib}<}$ for sequential $(\alpha, \beta)$ cutoffs.
   - **`V2.2′` (Barrier 2 — Transposition Tables & `KataGo MCGS` DAGs):** Uses Boolean Kleene-star reachability $\mathcal{M}_{\text{desc}}^{\text{DAG}} = (E_{\text{tree}} \lor E_{\text{trans}})^*$ with **exact ply/history-augmented canonical equality** $\mathbf{1}[\mathrm{canon}(O_j) = \mathrm{canon}(O_k)]$ for value/visit backup (restricting soft kernel $\exp(-\|h_j-h_k\|^2/\tau)$ strictly to the prior/proposal channel so distinct states never leak values) and edge-visit-weighted MCGS backups $Q(n) = \frac{U(n) + \sum_a N(n,a)Q(c(n,a))}{1 + \sum_a N(n,a)}$ via `V2.1′`.
   - **`V2.3′` (Barrier 3 — Imperfect-Info `CFR+` Infoset Regret Matching):** Keys $\mathcal{M}_{\text{infoset}}$ on the acting player's **perfect-recall observation-action history hash**, maintains persistent per-infoset cumulative clipped regret $R_+^{(t)}(I, a)$ and linear-weighted **average strategy** $\bar{\sigma}^{(T)}(I, a) \propto \sum_{t=1}^T t \, \pi_i^{(t)}(I) \sigma^{(t)}(I, a)$ updated at each step with signed log-reach LSE ($u = u^+ - u^-$), and enforces **Infoset Hidden-State Permutation Invariance** ($\max |\Delta \hat{\sigma}| \le 10^{-6}$).
   - **`V2.4′` (Barrier 4 — Gumbel MuZero Sequential Halving `SH`):** Pairwise sibling rank-percentile head $\widehat{\mathrm{pct}}_{\text{sib}}(j) = \frac{1}{|\mathrm{Sib}(j)|} \sum_{s \in \mathrm{Sib}(j)} \sigma((S(j) - S(s))/\tau_{\text{rank}})$ with temperature certificate $\tau_{\text{rank}} < \frac{\Delta}{2\log(2m)}$, frontier tokens with completed-$Q$ transform, and recorded teacher Gumbel noise $g_j$.
   - **`V2.5′` (Barrier 5 — Global Conflict-Clause / Nogood Broadcast in `CDCL`):** Incremental ancestor clamped-literal bitmask $L[j, \ell] = L[\mathrm{pa}(j), \ell] \lor \mathbf{e}_{\ell(j)}$ paired with an exact subset-test head $\langle e(C_j), e(\mathrm{core}_k) \rangle - |\mathrm{core}_k| = 0 \iff \mathrm{core}_k \subseteq C_j$ masked by $\mathcal{M}_{\text{nogood}}[j, k] = \mathbf{1}[m_{\text{ql}}(C_k) < \tau_{\text{clash}}]$, guaranteeing **zero unsound prunes**.

---

## 1. Three Separable Questions Tested in `R13` & Four Architectural Cohorts

Following `GPT-6-Astra` (`03 §2`) and `Claude Opus 5.5` (`04 §2.1`), `R13-ALGO-IMITATION` separates three distinct scientific questions for each of the 6 canonical search algorithms (`Algo 1`–`Algo 6`):
1. **`(R)` Constructive Representability (`CW`, Zero Training):** Do analytical/hand-set weights $\theta_{\text{CW}}$ in `GLCSearchTransformerV2` reproduce the teacher algorithm to $\ge 99.5\%$ accuracy on margin-filtered trees? (Isolates architectural expressivity from SGD optimization.)
2. **`(L-TF)` Teacher-Forced Learnability & Sample Efficiency ($N_{\text{train}} \in \{128, 512, 2048\}$):** Does AdamW supervised training on deterministic per-step teacher decision records $\mathcal{D}_t$ learn the algorithm sample-efficiently in-distribution ($B \le 32$, $D \le 8$) and generalize zero-shot out-of-distribution to deeper budgets ($B \in \{64, 128\}$, $D \in [10, 20]$)?
3. **`(L-FR)` Free-Running Rollout Fidelity (DAgger / Closed-Loop Execution):** When the learned student drives its own tree expansions from $b = 0 \to B$, does its resulting search tree and root output match the teacher without compounding trajectory drift?

### Four Architectural Cohorts Compared (`d_tree = 64`, 3 Seeds `1301, 1302, 1303`):
- **`C0_R12_1Hop_Flat` (`Theorem R4.1` Baseline):** 1-hop parent/child masks + unmasked flat content-key attention (`TwoStageTreeClearer` topology).
- **`C1_GLCSearch_v1_NoLSE`:** Adds transitive $\mathcal{M}_{\text{anc}}$ and $\mathcal{M}_{\text{desc}}$ masks (`Theorem R4.2`) with softmax/mean attention, but omits `LSE` count pooling and `V2.1′`–`V2.5′`.
- **`C2_GLCSearch_v1_Full` (`Theorem R4.2′` Architecture):** Full 7-mask `GLC-Search-Transformer v1` + frontier tokens + `LSE` count/soft-extrema pooling, without `V2.1′`–`V2.5′`.
- **`C3_GLCSearch_v2_Universal` (`Theorem R4.3` Architecture):** Full `GLC-Search-Transformer v2` (`C2` + `V2.1′`–`V2.5′`).

---

## 2. Pre-Registered Falsifiable Decision Gates (`R13-D0` through `R13-D6`)

### Gate `R13-D0`: Integrity, Theorem-Falsification Counterexample & Constructive Representability (`CW`) Gate (Hard Prerequisite)
1. **Static AST & Call-Graph Audit (`D0a`–`D0b`):** `0` static AST violations across `src/search/glc_search_transformer.py` and `benchmarks/run_r13_algo_imitation.py`; `3/3` synthetic canaries caught; teacher modules importable only by the trace generator, never by the student model; `PoisonTensor` `global_max_diff == 0.000000000`; `r4_kv_equiv_max_diff == 0.000000000`; zero absolute-depth embeddings (`E8`).
2. **Bitwise-Deterministic Teacher Replay & Incremental-vs-Recompute Equivalence (`D0c`):** Replaying any teacher seed reproduces decision records $\mathcal{D}_t$ bitwise (`max_diff == 0.0`), and incremental `V2.1′`–`V2.5′` updates match full from-scratch recomputation within $\le 10^{-6}$ across $1,000$ random traces.
3. **Empirical Verification of `Theorem R4.1(b′)` Tail-Swap Counterexample (`D0d`):** On $512$ adversarial tail-swapped tree pairs $(\mathcal{T}^{(1)}, \mathcal{T}^{(2)})$ of chain length $\ell > 3$ beneath symmetric siblings, `C0_R12_1Hop_Flat` must score **$\le 60.0\%$** branch discrimination (chance $= 50.0\%$, $R^2(Q^{\text{MCTS}}) \le 0.05$), whereas `C2_GLCSearch_v1_Full` and `C3_GLCSearch_v2_Universal` must achieve **$\ge 99.5\%$ accuracy and $R^2 \ge 0.999$**.
4. **Constructive Weights (`CW`) Representability Verification (`D0f`):** Before running SGD, analytical constructive weights $\theta_{\text{CW}}$ must achieve **$\ge 99.5\%$ selection match** on margin-filtered test trees for `Algo 1` in both `C2` (`v1`) and `C3` (`v2`), and **$\ge 99.5\%$** for `Algo 2`–`Algo 6` in `C3` (`v2`).

### Gate `R13-D1`: Supervised Imitation of `Algo 1` (Exact MCTS / AlphaZero PUCT Subtree Aggregation & Root-to-Leaf Selection)
- **Pass Criteria (`C3_GLCSearch_v2_Universal` & `C2_GLCSearch_v1_Full`, $N_{\text{train}}=2048$):**
  1. **In-Distribution ($B \le 32$) & OOD ($B \in \{64, 128\}$, depth $D \le 20$):** Active frontier selection top-1 accuracy **$\ge 95.0\%$ ID** and **$\ge 88.0\%$ OOD**; subtree value regression **$R^2(Q^{\text{MCTS}}) \ge 0.980$ ID** ($\ge 0.930$ OOD); log-visit count **$R^2(\log N) \ge 0.990$**; root visit distribution $\mathrm{KL}(\pi^{\text{MCTS}}_{\text{root}} \Vert \hat{\pi}_{\text{root}}) \le 0.030$ nats.
  2. **Pre-Registered Ablation Pattern (`Theorem R4.1` & `R4.2′`):** `C2 (v1)` and `C3 (v2)` agree within $|\Delta\text{Acc}| \le 3.0$ pp (confirming `v1` already suffices for commutative MCTS!), while both beat `C0_R12_1Hop_Flat` at OOD $B \in \{64, 128\}$ by **$\ge +25.0$ pp selection accuracy** and **$\ge +0.35$ $R^2(Q^{\text{MCTS}})$**.

### Gate `R13-D2`: Supervised Imitation of `Algo 2` (Arbitrary-Depth Alternating Minimax / Alpha-Beta & Proof-Number Search `PNS`/`df-pn`)
- **Pass Criteria (`Theorem R4.3` Barrier 1 Verification):**
  1. `C3_GLCSearch_v2_Universal` (`V2.1′`) achieves minimax value sign accuracy **$\ge 95.0\%$ ID ($D \le 6$)** and **$\ge 88.0\%$ OOD ($D \in [10, 14]$)**, $R^2(V^{\text{MM}}) \ge 0.970$, $\alpha\text{--}\beta$ pruning F1 **$\ge 0.920$**, PNS $\log(1+\mathrm{pn})/\log(1+\mathrm{dn})$ **$R^2 \ge 0.950$**, and Most-Proving Node top-1 accuracy **$\ge 92.0\%$ ID / $\ge 82.0\%$ OOD**.
  2. **Pre-Registered Ablation Pattern (`v2` vs. `v1`):** At OOD depths $D \in [10, 14] > L=3$, `C3_GLCSearch_v2_Universal` outperforms non-recurrent `C2_GLCSearch_v1_Full` by **$\ge +12.0$ pp** on minimax value sign / MPN accuracy and **$\ge +0.18$ $R^2$**, and the learned `V2.1′` gate places $\ge 80\%$ weight on $\mathrm{LSE}_{+\beta}$ at MAX nodes and $\mathrm{LSE}_{-\beta}$ at MIN nodes (`R13-X4`).

### Gate `R13-D3`: Supervised Imitation of `Algo 3` (`CDCL` Non-Chronological Backjumping + Global Nogood Pruning)
- **Pass Criteria (`Theorem R4.3` Barrier 5 Verification):**
  1. `C3_GLCSearch_v2_Universal` (`V2.5′`) achieves asserting backjump ancestor accuracy **$\ge 94.0\%$ ID / $\ge 86.0\%$ OOD**, cross-subtree nogood prune F1 **$\ge 0.920$ ID / $\ge 0.850$ OOD**, and **hard soundness bar: `0` unsound prunes** across all test instances.
  2. **Pre-Registered Ablation Pattern:** On disjoint-subtree repeat conflicts ($k \notin \mathrm{Anc}(b)$), `C3` (`V2.5′`) beats `C2` (`v1`, ancestor-only) by **$\ge +15.0$ pp prune F1**.

### Gate `R13-D4`: Supervised Imitation of `Algo 4` (`KataGo MCGS` Transposition DAG Backup + Multi-Witness Disjunctive BP)
- **Pass Criteria (`Theorem R4.3` Barrier 2 Verification):**
  1. `C3_GLCSearch_v2_Universal` (`V2.2′`) achieves edge-visit-weighted $Q^{\text{MCGS}}$ **$R^2 \ge 0.950$ ID / $\ge 0.900$ OOD**, transposition merge precision **$\ge 0.995$**, multi-witness Disjunctive Bayes root posterior $\mathrm{KL}(\mu^*_{\text{BP}} \Vert \hat{\mu}) \le 0.008$ nats, **`0` soft-merge value leaks** between non-equivalent states, and **`0` cycle NaNs**.
  2. **Double-Counting Falsification & Ablation (`E10`):** On DAGs with $\ge 3$ engineered transposition paths, `V2.2′` edge-visit-weighted backup beats unnormalized matrix-product pooling $E_{\text{trans}} \mathcal{M}_{\text{desc}}$ by $\ge 0.10$ $R^2$ and beats tree-only `C2` (`v1`) by **$\ge +0.12$ $R^2$** on transposed cousin nodes.

### Gate `R13-D5`: Supervised Imitation of `Algo 5` (`CFR+` Perfect-Recall Infoset Regret Matching)
- **Pass Criteria (`Theorem R4.3` Barrier 3 Verification):**
  1. On `Proposal-Poker`, `Micro-Stratego`, and Kuhn/Leduc benchmarks, `C3_GLCSearch_v2_Universal` (`V2.3′`) achieves average-strategy $\mathrm{KL}(\bar{\sigma}^*_T \Vert \hat{\bar{\sigma}}_T) \le 0.020$ nats, regret-sign agreement **$\ge 90.0\%$**, and Nash exploitability **$\le 1.25\times$ teacher exploitability ID** ($\le 1.50\times$ OOD at $T=128$).
  2. **Hard Firewall & Strategy-Fusion Bars (`C3d` & `E11c`):** Permuting private opponent states across histories within each information set yields **$\max |\Delta \hat{\bar{\sigma}}| \le 10^{-6}$** (`PoisonTensor`-style canary), linear probe AUC from pooled infoset representation to opponent private card is **$\le 0.55$**, and `LIR` belief gradient leak is **`0.000000`**. `C2` (`v1` without $\mathcal{M}_{\text{infoset}}$) exhibits $\ge 2.0\times$ higher exploitability.

### Gate `R13-D6`: Supervised Imitation of `Algo 6` (`Gumbel MuZero` Sequential Halving) & Multi-Algorithm Generalist Trunk
1. **Gumbel Sequential Halving (`Algo 6`):** With recorded teacher Gumbel noise $g_j$, `C3_GLCSearch_v2_Universal` (`V2.4′`) achieves per-phase survivor-set exact match **$\mathrm{SetEM} \ge 93.0\%$ ID / $\ge 88.0\%$ OOD**, rank-percentile MAE **$\le 0.030$**, and final root action match **$\ge 95.0\%$**, beating softmax-only `C2` (`v1`) by **$\ge +10.0$ pp** on survivor-set exact match.
2. **Sample Efficiency ($N_{\text{train}} = 512$ vs. $2048$) & Multi-Algorithm Shared Trunk (`R13-X1`):** At $N_{\text{train}} = 512$, `C3` retains **$\ge 85.0\%$** of its $N_{\text{train}} = 2048$ performance across all 6 algorithms ($\ge 90.0\%$ on `Algo 1, 4, 6`), and a **single shared `C3_GLCSearch_v2_Universal` trunk** conditioned on an algorithm-ID token passes all 6 gates within $\le 0.05$ of single-algorithm specialists!
