# Pre-Registration Protocol: Round 24B (`R24B-V3-CLRS-AND-OPENSPIEL-UNIFICATION`)

- **Suite ID:** `R24B-V3-CLRS-AND-OPENSPIEL-UNIFICATION`
- **Round / Architect:** Round `R24B` — Member 34 of the Society of Thinkers
- **Target Hardware:** `NVIDIA L4` GPU (`24 GB` VRAM, CUDA / PyTorch)
- **Random Seeds:** `Seeds = (101, 102, 103)` (`df = 2`, two-sided 95% Student-$t$ critical value $t_{0.975, 2} = 4.3027$)
- **Static & Runtime Audit Mandate:** `src/eval/ast_leak_auditor.py` (`v2`, Rules `S1`–`S10`, `0` violations, `7/7` synthetic canaries caught) + Runtime `PoisonTensor` invariance (`0.000000000` max diff across all `CLRS-30` and `OpenSpiel` models).

---

## 1. Motivation: Resolving the Two Empirical Bottlenecks Uncovered in `R22A` & `R22B`

In Wave-2 Rounds `R22A` and `R22B` (`A-PURE-NEURAL-SYMMETRIC`, 0 shortcuts, 0 target leaks), we identified two genuine architectural bottlenecks where `GLC_AOM_v2` trailed specialized domain baselines:

1. **Bottleneck 1 (`R22B-E2` Clean Zero-Leakage `CLRS-30` $N=64$ `4x` OOD Split Verdict):**
   - In `R22B-E2`, `GLC_AOM_v2` ranked **#1 of 5** on cumulative shortest-path relaxation (`Dijkstra` $N=64$: `0.8441` vs `0.7987` `MPNN_Max`; `Bellman_Ford` $N=64$: `0.8356` vs `0.7839` `MPNN_Max`), **but lost on `MST_Prim` (`0.6416` vs `0.8465` `Triplet_GMPNN`) and `DFS` (`0.7446` vs `0.8039`)**, resulting in a 6-task $N=64$ macro-average Micro-F1 of `0.7106` vs `0.7661` (`MPNN_Max`) and `0.7651` (`Triplet_GMPNN`).
   - **Root Cause:** `GLC_AOM_v2` hardcoded a fixed path-sum tropical relaxation $\min_u(d_u + W_{uv})$ ($\gamma = 1$) with fixed negative pointer weight `-3.5 * slack_d` for every task, and lacked 2-hop triplet edge-memory updates. While $\gamma = 1$ is the exact Bellman operator for `Dijkstra` and `Bellman_Ford`, `MST_Prim` selects the minimum single cut edge $\min_{u \in S} W_{uv}$ (where cumulative root distance $d_u$ has weight $\gamma = 0$), and `DFS` / `Articulation_Points` depend on local edge ordering and 2-hop triplet consistency.
2. **Bottleneck 2 (`R22A-E1` Pure-Neural `OpenSpiel` `Leduc Poker` `NashConv` Gap vs. `Deep_CFR`):**
   - In `R22A-E1`, `GLC_AOM_v2` ranked **#1 on hidden-board `phantom_ttt` (`0.2598`) and held-out `dark_hex` (`0.3177`)**, **but trailed `Deep_CFR` on 936-infoset `leduc_poker` (`6.2167` / `6.2348` vs `4.5736`)**, giving `Deep_CFR` a lower 4-game mean `NashConv` at $B=64$ (`1.5541` vs `1.7967` / `1.8701`).
   - **Root Cause:** In `R22A-E1`, `GLC_AOM_v2` used unweighted CFR+ updates during training and, at test-time compute budgets $B > 1$, added an unregularized softmax logit bonus $+(0.8 + 0.30 \log_2 B)(Q - V)$, which over-sharpened mixed bluffing strategies in two-round poker (`leduc_poker`) as $B$ increased from $4$ to $64$.

---

## 2. Pre-Registered Architectural Hypotheses & Equations

### 2.1 Experiment `R24B-E1`: `GLC_Graph_v3` with Learned Semiring Gate $\gamma_h \in [0, 1]$ + Triplet-LIR Edge Memory on Clean `CLRS-30` ($N=16 \to 32 \to 64$)

`GLC_Graph_v3` replaces the hardcoded $\gamma = 1$ path-sum bias of `GLC_AOM_v2` with:
1. **Multi-Head Learned Semiring Relaxation ($\gamma_h = \sigma(w_h) \in [0, 1]$, $H=4$ heads):**
   - Each head $h \in \{1, \dots, H\}$ maintains a generalized $\gamma_h$-semiring node potential updated over diameter-adaptive steps $t = 1, \dots, T_{\text{relax}}$:
     $$d_v^{(h, t+1)} = \min\left(d_v^{(h, t)}, \; \min_{u \in \mathcal{N}_{\text{in}}(v)} \left[ \gamma_h \, d_u^{(h, t)} + \text{cost}_{uv}^{(t)} \right]\right), \qquad \gamma_h = \sigma(w_h) \in [0, 1]$$
   - At $\gamma_h \to 1$, $d_v^{(h)}$ executes multi-hop Bellman-Ford / Dijkstra path-sum relaxation, with incoming edge slack $\text{slack}_{uv}^{(h)} = |\gamma_h d_u^{(h)} + \text{cost}_{uv} - d_v^{(h)}|$.
   - At $\gamma_h \to 0$, $\gamma_h d_u^{(h)} \to 0$, so $d_v^{(h)} \to \min_{u \in \mathcal{N}_{\text{in}}(v)} \text{cost}_{uv}$ and $\text{slack}_{uv}^{(h)} = |\text{cost}_{uv} - \min_{u'} \text{cost}_{u'v}|$ becomes the exact local cut-edge greedy criterion needed by `MST_Prim` and lexicographical `DFS`.
   - A learned semiring head distribution $\boldsymbol{\pi} = \text{softmax}(\mathbf{z}_{\text{semiring}} + \text{MLP}_{\text{ctx}}(\bar{\mathbf{h}}_G)) \in \Delta^{H-1}$ weights the $H$ semiring heads, yielding an interpretable **effective semiring gate**:
     $$\bar{\gamma} = \sum_{h=1}^H \pi_h \gamma_h \in [0, 1]$$
2. **Triplet-LIR Edge Memory Update + Max-Preconditioned Aggregation:**
   - At each message-passing step, `GLC_Graph_v3` updates edge representations via 2-hop triplet pivots gated by LIR precision $\beta_{\text{trust}}(u, v) = \sigma(\mathbf{w}_{\beta}^\top \mathbf{h}_v)$:
     $$\mathbf{e}_{uv}^{(t+1)} = \mathbf{e}_{uv}^{(t)} + 0.20 \, \beta_{\text{trust}}(u, v) \, \psi_{\text{trip}}\!\left(\max_k \phi_1(\mathbf{m}_{uk}^{(t)}) + \operatorname{mean}_k \phi_2(\mathbf{m}_{kv}^{(t)})\right)$$
   - Node aggregation combines 1-hop max-pooling over `in_adj` with LIR precision-weighted attention.

**Pre-Registered Falsification Gates (`R24B-E1`):**
- **Gate `R24B-E1a` (`CLRS-30` $N=64$ `4x` OOD Macro-Average Leadership):** Across the 6 `CLRS-30` tasks (`BFS`, `DFS`, `Dijkstra`, `Bellman_Ford`, `MST_Prim`, `Articulation_Points`) and 3 seeds (`101, 102, 103`), `GLC_Graph_v3` achieves the **#1 highest 6-task macro-average Micro-F1 and Pointer Accuracy at $N=64$ (`4x` OOD)** among all 6 architectures (`DeepSets`, `MPNN_Max`, `RelTransformer`, `Triplet_GMPNN`, `GLC_AOM_v2`, `GLC_Graph_v3`), recovering the `MST_Prim` and `DFS` deficits while preserving #1 shortest-path (`Dijkstra` & `Bellman_Ford`) accuracy.
- **Gate `R24B-E1b` (Mechanistic Semiring Gate Probe $\bar{\gamma}$):** Without any task-ID supervision on $\bar{\gamma}$, end-to-end training on `CleanCLRSBatch` drives the learned effective semiring gate $\bar{\gamma}$ to satisfy $\bar{\gamma}_{\text{Dijkstra}}, \bar{\gamma}_{\text{Bellman\_Ford}} > 0.65$ (path-sum regime) and $\bar{\gamma}_{\text{MST\_Prim}} < 0.35$ (cut-edge regime), with $\bar{\gamma}_{\text{SP}} - \bar{\gamma}_{\text{MST\_Prim}} \ge +0.35$.

---

### 2.2 Experiment `R24B-E2`: `GLC_AOM_v3` (`GLCSearch_v3`) with Linear-CFR Infoset Regret Memory on Pure-Neural `OpenSpiel`

`GLC_AOM_v3` (and `GLC_AOM_v3_Generalist`) upgrades `GLC_AOM_v2`'s infoset learning and test-time search while using the **exact same `153,617`-parameter `PureNeuralGameBackbone150K`** and **100% pure-neural GPU infoset evaluation**:
1. **Linear-Discounted Counterfactual Regret Accumulation (`Linear-CFR` / `DCFR`) in Training:**
   - At training iteration $t \in \{1, \dots, T\}$, instantaneous positive counterfactual regrets $r_t^+(I, a) = [Q^{(t)}(I, a) - V^{(t)}(I)]_+$ are accumulated with Discounted/Linear-CFR weighting:
     $$R_t^+(I, a) = \left[ \frac{t^{1.5}}{t^{1.5} + 1} R_{t-1}^+(I, a) + \frac{t}{T} r_t(I, a) \right]_+, \qquad \bar{\sigma}_T(I, a) \propto \sum_{t=1}^T t^2 \, \sigma_t(I, a)$$
   - `regret_head` is trained to predict normalized $R_T^+(I, a)$, `policy_head` (`logits`) is trained via cross-entropy to match the linear-weighted average strategy $\bar{\sigma}_T(I, a)$, and `belief_head` (`\chi = (0, 1)` detached trunk) predicts opponent hidden-state marginals.
2. **Test-Time GPU Linear-CFR Iterative Regret-Matching Refinement (`GLCSearch_v3`):**
   - At test-time budget $B \in \{1, 4, 16, 64\}$, instead of over-sharpening logits inside a softmax (which degraded `GLC_AOM_v2` at $B=16, 64$ on `leduc_poker`), `GLC_AOM_v3` initializes its infoset regret table from the neural `regret_head` and average strategy from `probs`, and runs $K_{\text{cfr}}(B)$ vectorized GPU iterations of neural counterfactual value backup and linear-weighted regret-matching average-strategy accumulation.

**Pre-Registered Falsification Gates (`R24B-E2`):**
- **Gate `R24B-E2a` (4-Game Mean `OpenSpiel` `NashConv` Leadership at $B=64$):** Across `kuhn_poker`, `leduc_poker`, `phantom_ttt(obstype=reveal-nothing)`, and `dark_hex(board_size=2)` over Seeds `101, 102, 103`, `GLC_AOM_v3` / `GLC_AOM_v3_Generalist` achieves a **lower 4-game mean `NashConv` at $B=64$ than both `GLC_AOM_v2` and `Deep_CFR`**, closing the `leduc_poker` exploitability gap while retaining #1/`#2` lowest `NashConv` on `phantom_ttt`.
- **Gate `R24B-E2b` (Monotonic Test-Time `NashConv` Improvement from $B=1 \to B=64$):** Unlike `GLC_AOM_v2` (whose `NashConv` rose from $B=4$ to $B=64$ due to logit over-sharpening), `GLC_AOM_v3` exhibits lower 4-game mean `NashConv` at $B=64$ than at $B=1$.
