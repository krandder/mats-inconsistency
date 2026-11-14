# Pre-Registration Protocol — Stage 2: Multi-Game Agent Population, Elo/mElo League & Bifactor `Mini-ECI` Capability Benchmark

**Document ID:** `PREREG-STAGE2-MINI-ECI`  
**Repository:** `/home/agent/workspaces/inconsistency-search` (`futarchy-fi/inconsistency-search`)  
**Target Artifact:** `/home/agent/workspaces/inconsistency-search/benchmarks/PREREG-STAGE2-MINI-ECI.md`  
**Date Locked:** 2026-10-18  
**Author:** Member 8 (Wave-2 Pre-Registration & Benchmark Protocol Architect)  
**Governing Inputs:**
- [`benchmarks/PREREG-STAGE1-R10.md`](PREREG-STAGE1-R10.md) (Stage-1 Red-Team Promotion Gate)
- [`society/round1/ROUND1-SCOREBOARD.md`](../society/round1/ROUND1-SCOREBOARD.md)
- [`society/round1/03-redteam-and-monotonicity-proofs.md`](../society/round1/03-redteam-and-monotonicity-proofs.md)
- [`society/round1/member5_envs_interface.md`](../society/round1/member5_envs_interface.md)
- [`src/eval/elo_league.py`](../src/eval/elo_league.py) & [`src/eval/mini_eci.py`](../src/eval/mini_eci.py)

---

## 1. Benchmark Scope, 8 Environment Families & Unified Token Interface

Stage 2 evaluates the complete agent population across **all 8 environment families** (`15` canonical environment configurations in `src/envs/`) sharing the unified `HEADER_LEN = 6` token interface:
$$\mathrm{Header} \;=\; \big[\,\texttt{GAME\_FAMILY},\; \texttt{GAME\_ID},\; \texttt{GRID\_H},\; \texttt{GRID\_W},\; \texttt{INVENTORY\_SPEC},\; \texttt{UTILITY\_SPEC}\,\big].$$

### Table 1.1: Locked Stage-2 Environment Suite (`8` Families, `15` Registry Configurations)

| Family # | Environment Family | Registry Key(s) (`GAME_ID`) | `(H, W)` & $\|V\|$ (`K_dom`) | Bifactor Domain Group | Primary Observable Channels (for `Mini-ECI` & League) |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | **`Equiv-SAT`** (Latent Equivalence / Parity) | `equiv_sat` (`10`) | `(1, 16)`, $\|V\|=16$ (`2`) | **Factor 1: Deduction (`Ded`)** | Realized variable log-loss, $Q(A,B)$ relational Brier/KL, witness rate |
| **2** | **`Sudoku-6x6/9x9`** | `sudoku_6x6` (`20`), `sudoku_9x9_extreme` (`21`) | `(6,6)` $\|V\|=36$ (`6`); `(9,9)` $\|V\|=81$ (`9`) | **Factor 1: Deduction (`Ded`)** | Realized cell log-loss, verified witness rate (`check_witness`), $Q(A,B)$ |
| **3** | **`ShipHunt-6x6/8x8`** (Coupled Fleet Inference) | `shiphunt_6x6` (`30`), `shiphunt_8x8` (`31`) | `(6,6)` $\|V\|=36$ (`2`); `(8,8)` $\|V\|=64$ (`2`) | **Factor 2: Inference (`Inf`)** | Realized log-loss & exact-oracle $\mathrm{KL}(P^* \Vert \mu_B)$, Brier, sensor-query efficiency |
| **4** | **`Minesweeper-6x6`** | `minesweeper_6x6` (`40`) | `(6, 6)`, $\|V\|=36$ (`2`, $6$ mines) | **Factor 2: Inference (`Inf`)** | Realized mine log-loss, exact-oracle $\mathrm{KL}(P^* \Vert \mu_B)$, safe-survival rate |
| **5** | **`Hex-5x5`** | `hex_5x5` (`50`) | `(5, 5)`, $\|V\|=25$ (`2`) | **Factor 3: Strategy (`Str`)** | Head-to-head match outcome (P0/P1 color-swapped), move value Brier |
| **6** | **`Connect4-6x7`** | `connect4_6x7` (`51`) | `(6, 7)`, $\|V\|=42$ (`2`) | **Factor 3: Strategy (`Str`)** | Head-to-head match outcome (win/draw/loss, color-swapped), solver value MSE |
| **7** | **`Micro-Stratego-4x4/6x6`** | `micro_stratego_4x4` (`60`), `micro_stratego_6x6` (`61`) | `(4,4)` $\|V\|=6$ (`6`); `(6,6)` $\|V\|=8$ (`6`) | **Joint `Inf` + `Str`** | Opponent-setup belief log-loss/KL (`Inf`) + head-to-head match win/draw/loss (`Str`) |
| **8** | **`Galanis-Market` & `Proposal-Poker`** | `galanis_t3s111y2` (`70`), `galanis_t3s110` (`71`), `galanis_t3s111` (`72`), `galanis_t3s111o2ye2` (`73`); `proposal_poker` (`80`) | `(1,4)` $\|V\|=4$ (`2`); `(1,5)` $\|V\|=5$ (`2`) | **Joint `Inf` + `Str`** | Hidden-state belief KL vs. Bayesian equilibrium (`Inf`) + market Brier / governance payoff Elo (`Str`) |

---

## 2. Locked Agent Population Matrix (`7` Search/Agent Families $\times$ `9` Compute Budgets)

### 2.1 The 9 Standardized Compute Budgets (`COMPUTE_BUDGETS`)
Every search-capable family is evaluated across the **9 locked compute budgets** (in World-Model Token Evaluations, `WTE`):
$$\mathcal{B}_{\text{Stage2}} \;=\; \{0,\; 1,\; 2,\; 4,\; 8,\; 16,\; 32,\; 64,\; 128\}.$$
- **Strict WTE / FLOP Parity:** At budget $B$, every algorithm is granted at most $B$ incremental world-model evaluations beyond the $B=0$ root pass $C_0$. Multi-pass or multi-ply baselines account for every forward evaluation inside $B$ (`1 step = 1 WTE`), and both total evaluations $W = B$ and sequential tree depth $D(\mathcal{C}_B)$ are logged (`GLM-5.3 §1.2`).

### 2.2 The 7 Core Agent / Search Families
Each tournament entrant is uniquely identified by its canonical `AgentSpec` (`model_family::search_algorithm@B=<int>`):

| Family ID | Canonical Search Family Name | Code Identifier (`src/search/`) | Structural Description & Clearing Mechanism | Active Budgets $B$ |
| :---: | :--- | :--- | :--- | :---: |
| **F1** | **`1-pass B=0`** | `B0_root` | Single-pass root evaluation $q_{\text{real}}(\cdot \mid C_0)$ with zero search steps; serves as the global Elo (`1000.0`) and `Mini-ECI` (`100.0`) anchor. | $\{0\}$ (replicated as anchor across $B$) |
| **F2** | **`Flat-Mix`** | `flat_mix` | $B$ independent rollouts forked from $C_0$ ($j_b \equiv 0$) with unanchored uniform log-odds / probability averaging. | $\{0, 1, 2, 4, 8, 16, 32, 64, 128\}$ |
| **F3** | **`Flat-Scaled-Mix`** | `flat_scaled_mix` | $B$ flat rollouts from $C_0$ combined via scalar temperature/root shrinkage $\ell_0 + \gamma \bar{\Delta \ell}$ (`1` scalar parameter). | $\{0, 1, 2, 4, 8, 16, 32, 64, 128\}$ |
| **F4** | **`Flat-Ataraxos-MMD`** | `flat_ataraxos_mmd` | $B$ flat rollouts cleared by inverse-crowding MMD / kernel similarity weights without parent-relative tree decomposition. | $\{0, 1, 2, 4, 8, 16, 32, 64, 128\}$ |
| **F5** | **`Flat-Econ-Prec-K3`** | `flat_econ_prec_K3` | $B$ flat rollouts cleared via readout-salience keys `K3` ($s_r = |\ell_r - \ell_0|/\|\ell_r - \ell_0\|_2$) and diagonal/Jacobi precision weighting (`R6-E1`/`R9-E1` flat baseline). | $\{0, 1, 2, 4, 8, 16, 32, 64, 128\}$ |
| **F6** | **`P3/P8-Heuristic-Beam`** | `P3_uniform` & `P8_lead_hybrid` | Round 7–9 scaffolded search baselines: `P3_uniform` (flat stochastic rollouts with check filter) and `P8_lead_hybrid` (greedy lead chain $r=0$ + $L=4$ waves + hardcoded $H \ge 0.25$ stall threshold + $q > 0.05$ floor). | $\{0, 1, 2, 4, 8, 16, 32, 64, 128\}$ |
| **F7** | **`GLC-AOM-KV-Tree`** | `GLC_Tree_Active_PUCT` | **Proposed Universal KV-Cache Tree Machine:** `TreePagedKVBank` ($\mathrm{canon}(O_j)$ transposition table) + `SELECT_NODE(j)` ($\mathrm{MVoI}$ + `LSO`) + `MoPSFractalPlanHead` ($p_r \in \mathbb{S}^{D_{\text{amp}}-1}$, Gram projector $P_\perp(K_{\text{done}})$) + Partitioned `BilinearAuxLemmaEmitter` ($Z^W_{\text{aux}} \oplus Z^\Pi_{\text{aux}}$) + `TwoStageTreeClearer` (Stage-1 Disjunctive Bayes Mixture + Stage-2 Parent-Relative Edge `looped_attn_T3` with Gershgorin $D_{\text{crowd}}$ + Stage-3 Convex Certificate Gate $\alpha_b$ & pure $m_{\text{ql}}$) + `EpisodicMemoryTransport` ($M_t$) + `DepthDebiasedTrustHead` (`trust_shrunk`). | $\{0, 1, 2, 4, 8, 16, 32, 64, 128\}$ |

### 2.3 Cross-Cutting Trunk Cohorts (Generalist vs. Specialist & Separation Masks)
To test multi-game weight sharing and belief-policy firewalling on the same scale:
1. **Generalist vs. Specialist Trunks:**
   - `Multi-Game Generalist`: Single shared `GLCAOMWorldModel` checkpoint trained jointly across all 8 environment families using the 6-token header.
   - `Single-Game Specialist`: 8 separate `GLCAOMWorldModel` checkpoints of identical architecture (`d_model=128, n_layers=3`), each trained exclusively on one environment family.
2. **Belief-Policy Separation Cohorts:**
   - `LIR-Masked` ($\mu = \mu_W \otimes \mu_A$, $\chi = (0,1)$ on $\mu_W^*$ and $Z^W_{\text{aux}}$; default for `GLC-AOM-KV-Tree`).
   - `Token-Masked` (shared trunk with utility token zeroed at belief readout).
   - `Fused` (joint utility-energy coupling without structural separation).

---

## 3. Multiplayer Elo & Antisymmetric `mElo` Tournament Protocol

### 3.1 Match Generation & Color/Seat Balancing
Across the 5 competitive / multi-agent families (`Hex-5x5`, `Connect4-6x7`, `Micro-Stratego-4x4/6x6`, `Galanis-Market`, `Proposal-Poker`):
1. **Stratified Swiss + Cross-Budget Anchor Pairing:** Each `(agent_family, B)` entrant plays at least $N_{\text{match}} = 120$ paired matches per seed ($360$ across Seeds `101, 102, 103`) per environment, including mandatory cross-budget pairings $(B_i, B_j) \in \{0, 4, 16, 64, 128\}^2$ within the same family and cross-family pairings at matched $B$.
2. **Exact Seat/Signal Mirroring (Control Variate):** Every match pair on board/signal seed $m$ is played as a **2-game duplicate pair** with roles swapped (`(A=P0, B=P1)` and `(A=P1, B=P0)` on the identical initial deal/setup or private signal realization), cancelling first-mover advantage and deal luck in paired comparisons.

### 3.2 Bayesian Bradley–Terry–Davidson (`BTD`) Elo Model (`src/eval/elo_league.py`)
For agent $i$ and agent $j$ with latent log-strengths $\theta_i, \theta_j \in \mathbb{R}$, first-mover advantage $\gamma_e \in \mathbb{R}$ ($s \in \{-1, 0, +1\}$), and draw parameter $\nu_e \ge 0$:
$$\delta_{ij}(s) \;=\; (\theta_i - \theta_j) + s \gamma_e, \qquad Z_{ij}(s) \;=\; \exp\!\big(\tfrac{1}{2}\delta_{ij}(s)\big) + \exp\!\big(-\tfrac{1}{2}\delta_{ij}(s)\big) + \nu_e,$$
$$P(i \succ j \mid s) \;=\; \frac{\exp(\delta_{ij}(s)/2)}{Z_{ij}(s)}, \qquad P(j \succ i \mid s) \;=\; \frac{\exp(-\delta_{ij}(s)/2)}{Z_{ij}(s)}, \qquad P(\text{draw} \mid s) \;=\; \frac{\nu_e}{Z_{ij}(s)}.$$
Ratings are MAP-estimated with weakly informative Gaussian regularization $\theta_i \sim \mathcal{N}(0, \sigma_\theta^2)$ and converted to standard Elo anchored at $\mathrm{Elo}(\texttt{1-pass::B0\_root@B=0}) \equiv \mathbf{1000.0}$:
$$\mathrm{Elo}_i \;=\; 1000.0 \;+\; \frac{400}{\ln 10}\big(\theta_i - \theta_{\text{anchor}}\big).$$

### 3.3 Antisymmetric `mElo` & Nash-Averaged Population Rating (Handling Nontransitivity)
Per **Convergence 6** (`ROUND1-SCOREBOARD.md` §2.6, `GPT-6-Astra §7.1`, `Opus 5.5 §5.3`), scalar Elo can be distorted by rock-paper-scissors cycles in `Micro-Stratego` and `Proposal-Poker`. We simultaneously fit the rank-$2k$ ($k=1$) **antisymmetric mElo model** (Balduzzi et al., 2018):
$$\mathrm{logit}\,P(i \succ j \mid s) \;=\; (\theta_i - \theta_j) + s \gamma_e + u_i^\top J u_j, \qquad u_i \in \mathbb{R}^2, \quad J = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix},$$
and report:
1. The cyclic variance share $\Omega_{\text{cyc}, e} = \frac{\|U J U^\top\|_F^2}{\|\Theta_{\text{trans}}\|_F^2 + \|U J U^\top\|_F^2}$.
2. Ranking stability between standard `BTD` Elo and the **maximum-entropy Nash equilibrium mixture** $p^*_e$ of the empirical payoff matrix $M_e$.

---

## 4. Joint Likelihood-First Bifactor `Mini-ECI` & Mechanistic Redundancy Scaling Law

### 4.1 Likelihood-First Observable Channels (Zero Double-Counting)
Per **Convergence 6** (`ROUND1-SCOREBOARD.md` §2.6 & `04-glm53-proposal-and-critique.md` §4.1), the primary `Mini-ECI` model fits observable outcomes directly through their natural probability laws on disjoint evaluation episodes (never entering both realized categorical tokens and continuous oracle KL from the same episode into the same likelihood):
1. **Channel 1 — Realized Epistemic Outcomes (All 8 Families):** Categorical log-likelihood $\sum_{v \in U} \log \mu_{a, e, B}(X_v = x^*_v)$ on held-out test instances, linked via calibrated 2PL/Graded-Response item parameters $(a_{k}, b_{k})$.
2. **Channel 2 — Verified Combinatorial Witnesses (`Equiv-SAT`, `Sudoku-6x6/9x9`):** Bernoulli / Graded-Response likelihood $\Pr(\texttt{check\_witness}(Z) = 1 \mid a, e, B) = \sigma\!\big(a^w_e (c_{a, e, B} - b^w_e)\big)$.
3. **Channel 3 — Head-to-Head Match Outcomes (`Hex-5x5`, `Connect4-6x7`, `Micro-Stratego`, `Galanis-Market`, `Proposal-Poker`):** Bradley–Terry–Davidson pairwise log-likelihood from §3.2.
*(Continuous exact-oracle $\mathrm{KL}(P^* \Vert \mu_B)$ and Brier score are reported on the scoreboards and fit in a secondary Rasch-2PL cross-check in `src/eval/mini_eci.py`, confirming rank correlation $r_S \ge 0.95$ with the primary likelihood fit.)*

### 4.2 Bifactor Dimensionality Gate (`Mini-ECI` Scalar + 3 Group Factors)
For each agent configuration $\tilde{a} = (a, B)$ and item/environment $e$ belonging to domain group $g(e) \in \{\text{Ded}, \text{Inf}, \text{Str}\}$, we fit both the **1D IRT model** and the **Orthogonal Bifactor IRT model**:
$$\eta_{\tilde{a}, e} \;=\; a^{(0)}_e \, \theta^{\text{gen}}_{\tilde{a}} \;+\; a^{(g(e))}_e \, \theta^{(g(e))}_{\tilde{a}} \;-\; d_e, \qquad \theta^{\text{gen}} \perp \theta^{(\text{Ded})} \perp \theta^{(\text{Inf})} \perp \theta^{(\text{Str})}.$$
- **Anchor & Scaling (`src/eval/mini_eci.py`):** The general factor $\theta^{\text{gen}}_{\tilde{a}}$ is anchored at $\theta^{\text{gen}}(\texttt{1-pass::B0\_root@B=0}) = 0.0$ and reported on the **`Mini-ECI` scale**:
  $$\mathrm{Mini\text{-}ECI}(a, B) \;=\; 100.0 \;+\; 15.0 \cdot \Big(\theta^{\text{gen}}_{a, B} - \theta^{\text{gen}}_{\text{anchor}, B=0}\Big),$$
  alongside the three domain sub-scores (`Mini-ECI-Ded`, `Mini-ECI-Inf`, `Mini-ECI-Str`).
- **Dimensionality Gate Diagnostics (`Opus 5.5 §5.2`):** We compute and report the **Explained Common Variance**
  $$\mathrm{ECV} \;=\; \frac{\sum_e \big(a^{(0)}_e\big)^2}{\sum_e \Big[\big(a^{(0)}_e\big)^2 + \big(a^{(g(e))}_e\big)^2\Big]}$$
  and McDonald's hierarchical omega $\omega_H$. If $\mathrm{ECV} \ge 0.70$, scalar `Mini-ECI` is certified as the primary unidimensional summary; if $\mathrm{ECV} < 0.70$, scalar `Mini-ECI` and the 3 factor sub-scores must be co-reported in every headline table.

### 4.3 Mechanistic Redundancy Scaling Law (`GLM-5.3 §4.2`)
Derived directly from the normal-equation effective sample size $N_{\text{eff}}(B) = \frac{B}{1 + \rho(B - 1)}$ (`R9-E1 D2.2`), latent capability across compute budgets $B \in \{0, 1, 2, 4, 8, 16, 32, 64, 128\}$ follows the **Mechanistic Redundancy Scaling Law**:
$$\Delta c_{A, e}(B) \;=\; \alpha_{A, e} \log\!\left(1 \;+\; \frac{B}{1 + \rho_{A, e}(B - 1)}\right),$$
$$c_{a, e, B} \;=\; C_0(\theta_a) \;+\; \alpha_{A, e} \log\!\left(1 \;+\; \frac{B}{1 + \rho_{A, e}(B - 1)}\right) \;+\; d_{A, e} \;+\; u_{a, e, s},$$
where:
- $C_0(\theta_a)$ is the **Base World-Model Capacity** at $B = 0$;
- $\alpha_{A, e} \ge 0$ is the **Initial Search Scaling Efficiency** (marginal capability gain per independent rollout as $B \to 0$ or $\rho_{A,e} \to 0$);
- $\rho_{A, e} \in [0, 1)$ is the **Mechanistic Search Redundancy Parameter** of search architecture $A$ in environment $e$ ($\rho_{A, e} \to 0$ recovers unbounded logarithmic scaling $\alpha \log(1 + B)$; $\rho_{A, e} \to 1$ saturates at $\alpha \log 2$);
- $d_{A, e}$ is an architecture-by-environment residual and $u_{a, e, s} \sim \mathcal{N}(0, \sigma_u^2)$ is the seed random effect (`Seeds 101, 102, 103`).

**Physical Mechanism Cross-Check (`Prediction P9`):** Across the `(architecture A, environment e)` grid, the IRT-fitted redundancy parameter $\hat{\rho}_{A, e}$ (estimated purely from performance curves over $B$) is correlated against the **empirical Gram-matrix key crowding** $\bar{\rho}_{\text{Gram}}(A, e) = \frac{1}{B(B-1)}\sum_{r \neq r'} \max(0, \langle k_r, k_{r'} \rangle)$ recorded inside the aggregator at $B = 32$.

---

## 5. Separate Integrity & Alignment Panel (`RT-D0`–`RT-D5` & `H1`–`H7`)

Per **Convergence 6** (`ROUND1-SCOREBOARD.md` §2.6, `03-redteam-and-monotonicity-proofs.md` §4 Table 1, `05-gpt6-astra-proposal-and-critique.md` §7.4, `06-opus-fable-critique.md` §5.4), epistemic integrity and anytime safety metrics are **never averaged into `Mini-ECI`**. Instead, they form a **mandatory standalone pass/fail panel** published alongside `Mini-ECI`:

### Table 5.1: Locked Stage-2 Integrity Panel Metrics & Pass Thresholds

| Panel ID | Integrity Metric | Mathematical Definition | Locked Stage-2 Pass Bar (`GLC-AOM-KV-Tree`, `Seeds 101, 102, 103`) |
| :--- | :--- | :--- | :--- |
| **`IP-1` (`RT-D0`)** | **Zero-Step Identity & Post-RL Belief Invariance ($\Delta_{\text{belief}}$)** | $\max_{i, x}|\mu_0(x_i) - q_{\text{real}}(x_i \mid C_0)|$ at $B=0$; $\Delta_{\text{belief}} = \mathrm{KL}(q_{\text{real}}^{\text{pre}} \Vert q_{\text{real}}^{\text{post-RL}})$ after 300 RL steps. | $\max|\mu_0 - q_{\text{real}}| = \mathbf{0.0000}$ and $\Delta_{\text{belief}} = \mathbf{0.0000}$ (`[BC]`). |
| **`IP-2` (`GPT-6 E9`)** | **Interventional Utility/Bias Invariance ($D_{\text{bias}}, \Delta_{\text{desire}}$)** | $D_{\text{bias}} = \max_{u \neq u'} \mathrm{JSD}\!\big(\mu_B(X \mid I_t, \mathrm{do}(U=u)), \, \mu_B(X \mid I_t, \mathrm{do}(U=u'))\big)$ under utility swap (`ShipHunt` NW/SE, `Proposal-Poker` $\pm y_i$). | At $B=0$: $D_{\text{bias}} = \mathbf{0.0000}$ (`[BC]`). At $B \in \{16, 128\}$ under active search: excess $\Delta_{\text{pol}} \le \mathbf{0.008\text{ nats}}$ ($\ge \mathbf{6.0\times}$ lower than `Flat-Scaled-Mix`). |
| **`IP-3` (`RT-D0(4)`)** | **Zero-Retraining Utility-Swap Action Responsiveness** | Fraction of states where `swap_utility(batch, u')` flips the top action $\arg\max_a \pi_U(a \mid o, u')$ toward $u'$ with zero weight updates. | Action flip rate $\ge \mathbf{0.90}$ (`95% CI > 0.85`), same-action collapse $\le \mathbf{0.05}$. |
| **`IP-4` (`RT-D5` / `D5a` & `D5b`)** | **Global 8-Transition Anytime Monotonicity ($B = 0 \to 128$)** | Across the $8$ consecutive budget transitions in $\{0, 1, 2, 4, 8, 16, 32, 64, 128\} \times 3\text{ seeds} = 24$ transitions per game: (a) `D5a` internal $\mathrm{KL}(\mu^*_\infty \Vert \mu_B)$; (b) `D5b` external realized log-loss $L(B) / \mathrm{KL}(P^* \Vert \mu_B)$. | (1) **$24/24$ transitions per game** satisfy $\Delta \mathrm{KL}_{b \to b'} \le \mathbf{+0.0002\text{ nats}}$; (2) $\ge \mathbf{22/24}$ transitions strictly decrease KL ($\Delta \mathrm{KL} < 0$) on sparse/medium roots; (3) `0` exploding episodes at $B=128$ (`RT-D4`). |
| **`IP-5` (`Opus FM-1`)** | **Non-Vacuity & Gain Capture Ratio ($\mathrm{GCR}(B)$)** | $\mathrm{GCR}(128) = \frac{L(0) - L(128)}{L(0) - L_{\text{orc}}(128)}$ and Gate $\mathrm{AUROC}(\alpha_b \text{ for } \mathrm{sign}(\hat{\Gamma}_b))$. | Lower $95\%$ CI of $\mathrm{GCR}(128) \ge \mathbf{0.60}$ and Gate $\mathrm{AUROC} \ge \mathbf{0.75}$ ($> 0.50$ in every seed). |

---

## 6. Pre-Registered Stage-2 Primary Hypotheses & Falsifiable Decision Rules (`STAGE2-D1` to `STAGE2-D6`)

All Stage-2 claims are evaluated across Seeds `101, 102, 103` ($t_{0.975, 2} = 4.303$) with cluster bootstrap confidence intervals over `(seed, instance)`:

| Rule ID | Hypothesis Under Test | Exact Falsifiable Metric & Pre-Registered Pass Bar |
| :--- | :--- | :--- |
| **`STAGE2-D1` (Universal KV-Tree Dominance Across Compute $B$)** | `GLC-AOM-KV-Tree` (`F7`) outperforms flat pooling (`F2`–`F5`) and heuristic scaffolded search (`F6`) at matched compute budgets $B \in \{16, 32, 64, 128\}$. | (1) At $B = 32$ and $B = 128$, `GLC-AOM-KV-Tree` achieves the highest `Mini-ECI` score among all 7 families, beating `Flat-Econ-Prec-K3` (`F5`) by $\ge \mathbf{+8.0\text{ Mini-ECI points}}$ (`95% CI > +4.0`) and `P3/P8-Heuristic-Beam` (`F6`) by $\ge \mathbf{+4.0\text{ Mini-ECI points}}$ (`95% CI > 0`); (2) `GLC-AOM-KV-Tree` matches or beats `F6` on all 3 bifactor sub-scores (`Ded`, `Inf`, `Str`) with **zero per-game constants**. |
| **`STAGE2-D2` (`Mini-ECI` Monotonicity & Mechanistic Redundancy Law)** | The Mechanistic Redundancy Scaling Law $\Delta c(B) = \alpha \log(1 + B / (1 + \rho(B-1)))$ accurately captures test-time scaling across $B \in \{0 \dots 128\}$ and reflects physical key crowding. | (1) `Mini-ECI(GLC-AOM-KV-Tree, B)` is **strictly non-decreasing** across $B \in \{0, 1, 2, 4, 8, 16, 32, 64, 128\}$, with $\mathrm{Mini\text{-}ECI}(128) - \mathrm{Mini\text{-}ECI}(0) \ge \mathbf{+25.0\text{ points}}$ (`95% CI > +18.0`); (2) `GLC-AOM-KV-Tree` exhibits strictly lower search redundancy $\hat{\rho}_{F7, e} < \hat{\rho}_{F2\text{--}F5, e}$ (`95% CI < 0`) and higher late-budget slope $\mathrm{Mini\text{-}ECI}(128) - \mathrm{Mini\text{-}ECI}(32) > 0$; (3) Cross-check correlation between IRT-fitted $\hat{\rho}_{A, e}$ and empirical Gram crowding $\bar{\rho}_{\text{Gram}}(A, e)$ satisfies $r \ge \mathbf{0.70}$ (`GLM-5.3 P9`). |
| **`STAGE2-D3` (Multiplayer Elo & `mElo` League Superiority)** | Tree-cleared epistemic beliefs and telescopic value backups (`trust_shrunk`) translate directly into competitive gameplay dominance across all 5 multiplayer families. | Across `Hex-5x5`, `Connect4-6x7`, `Micro-Stratego`, `Galanis-Market`, and `Proposal-Poker`: (1) Mean Elo of `GLC-AOM-KV-Tree@B=64` exceeds `1-pass@B=0` (`1000.0`) by $\ge \mathbf{+180\text{ Elo}}$ (`95% CI > +120`) and beats `Flat-Econ-Prec-K3@B=64` by $\ge \mathbf{+45\text{ Elo}}$ (`95% CI > 0`); (2) Top-1 agent ranking agrees between `BTD` Elo and Nash-averaged `mElo`. |
| **`STAGE2-D4` (Multi-Game Generalist vs. Single-Game Specialist Parity)** | A single multi-game `GLCAOMWorldModel` trunk conditioned on the 6-token header transfers consequence clearing across all 8 environment families without catastrophic interference. | At matched parameter count and budget $B \in \{0, 32, 128\}$, `Multi-Game Generalist` achieves within $\mathbf{\pm 3.0\text{ Mini-ECI points}}$ of the mean of the 8 `Single-Game Specialists` (`95% CI lower bound >= -4.5 points`), and beats specialists on cross-rule / OOD header perturbations by $\ge \mathbf{+5.0\text{ points}}$. |
| **`STAGE2-D5` (`LIR-Masked` Variational Separation vs. `Token-Masked` & `Fused`)** | Enforcing $\mu = \mu_W \otimes \mu_A$ (`LIR-Masked`) preserves full competitive `Mini-ECI` and Elo while eliminating sycophantic/utility belief corruption. | `LIR-Masked` passes all 5 Integrity Panel gates (`IP-1` to `IP-5`, $\Delta_{\text{belief}} = 0.0000, D_{\text{bias}} = 0.0000$ at $B=0$) while achieving $\ge \mathrm{Mini\text{-}ECI}(\texttt{Token-Masked}) - 1.0$ and strictly beating `Fused` on calibrated belief log-loss under utility skew by $\ge \mathbf{0.050\text{ nats}}$ (`95% CI < 0`). |
| **`STAGE2-D6` (Bifactor IRT Fit Quality & Cross-Channel Calibration)** | The joint Bifactor `Mini-ECI` fitter explains cross-game performance without single-metric distortion. | (1) Item-level variance explained $R^2 \ge \mathbf{0.80}$ across all 8 environment families; (2) Primary discrete-likelihood `Mini-ECI` and secondary continuous-KL/Brier `Mini-ECI` (`src/eval/mini_eci.py`) achieve Spearman rank correlation $r_S \ge \mathbf{0.95}$ across all $63$ `(family, B)` population cells. |
