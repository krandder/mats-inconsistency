# Pre-Registration Protocol: Round 14 (`R14-UNSUP-DISCOVERY`) — Unsupervised Task-Driven Search Discovery & Mechanistic Algorithmic Alignment with `GLC-Search-Transformer v2`

- **Document ID:** `PREREG-R14-UNSUP-DISCOVERY`
- **Status:** **LOCKED PRIOR TO ROUND-14 EXECUTION**
- **Date:** 2026-10-18
- **Repository:** `/home/agent/workspaces/inconsistency-search` (`futarchy-fi/inconsistency-search`)
- **Prerequisite Imitation & Expressivity Benchmark:** [`benchmarks/PREREG-R13-ALGO-IMITATION.md`](PREREG-R13-ALGO-IMITATION.md)
- **Parent Neural Baseline:** [`results/RESULTS-R12-GPU-V2.md`](../results/RESULTS-R12-GPU-V2.md) (`r12_gpu_v2_results.json`)
- **Governing Theory & Multi-Model Society Audits (`society/round4/`):**
  - [`society/round4/01-expressivity-theorems-and-mcts-gap.md`](../society/round4/01-expressivity-theorems-and-mcts-gap.md)
  - [`society/round4/02-glm53-round4-expressivity.md`](../society/round4/02-glm53-round4-expressivity.md)
  - [`society/round4/03-gpt6-astra-round4-expressivity.md`](../society/round4/03-gpt6-astra-round4-expressivity.md)
  - [`society/round4/04-opus-round4-expressivity.md`](../society/round4/04-opus-round4-expressivity.md)
  - [`society/round4/ROUND4-SCOREBOARD.md`](../society/round4/ROUND4-SCOREBOARD.md)

---

## 0. Scientific Purpose & Bitter-Lesson Mandate

Once `R13-ALGO-IMITATION` establishes that `GLC-Search-Transformer v2` (`C3_GLCSearch_v2_Universal`) **can represent (`CW`) and sample-efficiently learn (`SGD`)** all 6 canonical search algorithm families (`MCTS-PUCT`, `Minimax/Alpha-Beta & PNS`, `CDCL-Global-Nogood`, `MCGS-Disjunctive-BP`, `CFR+ Infoset Regret Matching`, and `Gumbel Sequential Halving`), Round 14 (`R14-UNSUP-DISCOVERY`) executes the primary Bitter-Lesson goal stated by Kelvin:

> *"I think we should have of course the general algorithm learn the best algorithm for each case without supervision on specific algorithmic implementations."*

### Core Hypotheses Tested in `R14-UNSUP-DISCOVERY`:
1. **Task-Only Search Discovery:** Trained from scratch (random initialization of `GLCSearchTransformerV2`, zero teacher labels, zero algorithm-ID tokens) purely to minimize end-of-search belief proper score $\mathrm{KL}(P^* \Vert \mu_B)$, maximize verified witness discovery, and maximize interactive game payoff via **Leave-Subtree-Out (`LSO`) counterfactual credit** + **GRPO**, the network discovers effective tree-search and belief-clearing strategies across all 12 environments.
2. **Elimination of the `R12` $B=128 \to 256$ Multi-Witness KL Regression (`0.1294 -> 0.1387`):** Under `LSO` counterfactual credit against multi-witness posteriors $P^*$ paired with `V2.2′` subtree/DAG normalization, a subtree that discovers a **second, distinct valid witness** receives positive credit $I_{\text{LSO}}(s) > 0$, whereas a redundant duplicate rollout inside an already-resolved witness subtree receives $I_{\text{LSO}}(s) \approx 0$ (`Opus 5.5` §3.2 proof). This directly eliminates single-witness over-reinforcement at $B=256$ on `ShipHunt` and `Micro-Stratego`.
3. **Mechanistic Algorithmic Alignment (`MAAP` & Causal Interchange Intervention `IIA`):** Post-hoc behavioral (`P1`), representational (`P2`), causal interchange (`P3`), mask-attribution (`P4`), and beneficial-deviation (`P5`) probes test whether the unsupervised learner spontaneously rediscovers MCTS/MCGS on `Hex/Connect-4`, CDCL/PNS on `Sudoku/Equiv-SAT`, and CFR+/Disjunctive-BP on `Proposal-Poker/ShipHunt/Stratego`—or discovers a superior non-classical hybrid!

---

## 1. Pure Task-Driven Objective & Leave-Subtree-Out (`LSO`) Credit Assignment

Using the frozen `R12` `Unified-Generalist-FiLM-IsoUpdate` world model $f_\phi$, `GLCSearchTransformerV2` ($\theta$) is trained strictly from task-level outcome signals:
1. **Differentiable Belief & Strategy Readout Loss:**
   $$\mathcal{L}_{\text{read}}(\theta) \;=\; \mathbb{E}_{b \sim \mathrm{LogUnif}(1, B_{\text{train}})} \Big[ \mathrm{KL}\!\big(P^*(V \mid C_0) \;\Vert\; \mu_b^\theta(V \mid \mathcal{T}_b)\big) \;+\; \lambda_{\text{mono}} \sum_{b'=1}^{B_{\text{train}}} \mathrm{ReLU}\!\big(\mathrm{KL}(P^* \Vert \mu_{b'}^\theta) - \mathrm{KL}(P^* \Vert \mu_{b'-1}^\theta)\big) \Big].$$
2. **Leave-Subtree-Out (`LSO`) + Pairwise-LSO Counterfactual Credit & GRPO Policy Optimization:**
   For each rollout $g \in \{1, \dots, G\}$ ($G = 8$) and expanded node $s_t$, let $q^{-s_t}$ be the `NeuralTreeAggregator` readout recomputed with $\mathrm{Desc}(s_t)$ masked out in a single batched forward pass, and let $S(q, y) = \mathrm{KL}(P^* \Vert q) - \kappa_{\text{wit}} \mathbf{1}[\text{verified witness}] - \kappa_{\text{pay}} u_i$:
   $$I_{g, t} \;=\; S(q^{-s_t}, y) \;-\; S(q^{\text{full}}, y) \qquad (\text{plus } 10\% \text{ pairwise complementarity audit } I_{s,s'} = S(q^{-\{s,s'\}}) - S(q^{-s}) - S(q^{-s'}) + S(q^{\text{full}})).$$
   The per-step advantage combines group-normalized episode outcome and normalized LSO credit:
   $$\hat{A}_{g, t} \;=\; \frac{S_{\bar{g}} - S_g}{\mathrm{std}_g(S) + \epsilon} \;+\; \eta_{\text{lso}} \frac{I_{g, t}}{\mathrm{std}(I) + \epsilon}, \qquad \mathcal{L}_{\text{sel}}(\theta) \;=\; -\mathbb{E}\Big[\sum_{t=1}^B \min\big(r_t \hat{A}_{g,t}, \mathrm{clip}(r_t, 1\pm \epsilon_{\text{clip}})\hat{A}_{g,t}\big)\Big] + \beta_{\text{KL}} \mathrm{KL}(p_\theta \Vert p_{\text{ref}}).$$

---

## 2. Pre-Registered Falsifiable Decision Gates (`R14-D0` through `R14-D6`)

### Gate `R14-D0`: Zero-Oracle-Leakage & Zero-Algorithmic-Supervision Firewall (Hard Prerequisite)
1. **Static AST & Provenance Audit (`S1`–`S6` + `S7_NO_TEACHER_ALGO`):** `0` static AST violations; `3/3` synthetic canaries caught; zero import paths or file reads from `R13` teacher modules, `R13` checkpoints (outside the explicitly isolated warm-start arm in `R14-D6`), or teacher decision-record files.
2. **Runtime `PoisonTensor` & Independent Verifier Audit:** `PoisonTensor` `global_max_diff == 0.000000000` and `r4_kv_equiv_max_diff == 0.000000000` across all 12 environments $\times$ 3 seeds (`1401, 1402, 1403`); 100% agreement between primary witness checker and an independent second reference verifier (`Reward-Hacking Audit`).
3. **Exact `LIR` & Infoset Hidden-Permutation Firewall:** $\|\nabla_{Z^\Pi} \ell_V\|_\infty = \mathbf{0.000000}$, post-RL belief drift $= \mathbf{0.000000}$, $D_{\text{bias}}(B=0) = \mathbf{0.000000}$, and infoset hidden-state permutation invariance $\max |\Delta \hat{\bar{\sigma}}| \le 10^{-6}$ (with opponent private-card probe `AUC <= 0.55`).

### Gate `R14-D1`: Overall `Mini-ECI` Lift Over `R12` Unified Generalist (`106.53`)
Evaluated across all 12 environments and $B \in \{0, 1, 2, 4, 8, 16, 32, 64, 128, 256\}$ on blind `split="test"` across 3 seeds:
- **Pass Criteria:** Unsupervised `GLCSearch-v2-Unsup` exceeds `R12`'s `Unified-Generalist-FiLM-IsoUpdate` (`106.53` `Mini-ECI`) by **$\ge +1.00$ `Mini-ECI` point** (`Mini-ECI >= 107.53`, paired 95% CI lower bound $> 106.53$), with stretch target matching or beating the `11.83x`-larger `Single-Game-Specialist` ensemble (`108.61`).

### Gate `R14-D2`: Elimination of the `R12` $B=128 \to 256$ Multi-Witness KL Regression
- **Pass Criteria:**
  1. **12-Environment Mean KL Monotonicity ($B=128 \to 256$):** `mean_kl_12env(B=256) <= mean_kl_12env(B=128) - 0.0030` nats **and** `mean_kl_12env(B=256) <= 0.1294` nats (eliminating `R12`'s `0.1294 -> 0.1387` uptick, paired 95% CI excluding zero).
  2. **Multi-Witness Partial-Observability Improvement (`ShipHunt` & `Stratego`):** On `ShipHunt-6x6`, `ShipHunt-8x8`, `Micro-Stratego-4x4`, and `Micro-Stratego-6x6`, `KL(B=256) <= 0.90 * KL(B=128)` with strictly higher witness-coverage entropy across distinct valid worlds than `R12`, while deductive environments retain `R12`'s ultra-deep gains (`Sudoku-6x6` $\le 0.0380$, `Sudoku-9x9-Extreme` $\le 0.0416$ at $B=256$).

### Gate `R14-D3`: Interactive Play Strength (`Elo`) & Verified Witness Discovery
- **Pass Criteria:**
  1. In 100% interactive turn-by-turn duplicate round-robin tournaments at $B=256$, `GLCSearch-v2-Unsup` beats `R12-V2` (`1205.1` Elo) by **$\ge +75.0$ Elo** (`ALL Elo >= 1280.1`, 95% CI $> 0$).
  2. 12-environment verified witness discovery rate at $B=256$ reaches **$\ge 75.0\%$** (vs. `73.61%` in `R12-V2`).

### Gate `R14-D4`: Imperfect-Information Equilibrium Quality (`Proposal-Poker` & `Micro-Stratego`)
- **Pass Criteria:**
  1. On `Proposal-Poker`, extracted average strategy $\hat{\bar{\sigma}}$ achieves best-response exploitability **$\le 1.20\times$** that of the exact `CFR+` teacher at matched node budget (and reduces exploitability by $\ge 35.0\%$ relative to `R12-V2`).
  2. On `Micro-Stratego-6x6`, `GLCSearch-v2-Unsup` achieves head-to-head win rate **$\ge 55.0\%$** against `R12-V2` while passing the hard infoset firewall bars in `R14-D0`.

### Gate `R14-D5`: Five-Probe Mechanistic Algorithmic Alignment (`MAAP` `P1`–`P5` & Causal Interchange `IIA`)
Evaluated post-hoc on frozen `GLCSearch-v2-Unsup` checkpoints across held-out trees:
- **`P1` (Behavioral Alignment Index $\mathrm{AI}_A^e$), `P2` (Hewitt-Liang Selectivity $R^2 - R^2_{\text{ctrl}}$ of Linear Probes for $Q^{\text{MCTS}}, V^{\text{MM}}, \log\mathrm{pn}/\mathrm{dn}, R_+$), `P3` (Geiger Causal Interchange Intervention Accuracy `IIA`), `P4` (Mask-Channel Attribution), and `P5` (Counterfactual Beneficial-Deviation Analysis).**
- **Pass Criteria:**
  1. In **$\ge 3$ of the 4 domain families** (`Hex/Connect-4` $\leftrightarrow$ `MCTS/MCGS`, `Sudoku/Equiv-SAT` $\leftrightarrow$ `CDCL/PNS`, `Proposal-Poker` $\leftrightarrow$ `CFR+`, `ShipHunt/Stratego` $\leftrightarrow$ `MCGS-Disjunctive-BP`), the pre-registered domain teacher achieves the highest Behavioral Alignment Index $\mathrm{AI}_A^e$ by a margin of $\ge +0.10$, representational probe selectivity $\ge 0.30$, and Causal Interchange Intervention Accuracy **$\mathrm{IIA} \ge 0.70$**.
  2. **Pre-Registered Decision Rule for Non-Classical Discovery:** If `GLCSearch-v2-Unsup` passes `R14-D1`–`R14-D4` while deviating from the classical teacher in `R14-D5`, `P5` (Beneficial-Deviation Analysis) tests whether one-step counterfactual substitution of the teacher's move *worsens* taskKL/payoff—formally distinguishing *"discovered a superior task-adapted search algorithm"* from *"failed to learn search"*.

### Gate `R14-D6`: Comparison Against Fixed-Algorithm Portfolio Oracle & `R13 -> R14` Warm-Start Curriculum
1. **Beating Fixed Hand-Coded Algorithms (`Portfolio-Oracle` Benchmark):** Evaluated at matched wall-clock / node budget against all 6 fixed teacher algorithms (`Algo 1`–`Algo 6`), `GLCSearch-v2-Unsup` beats **every individual fixed algorithm** on overall 12-environment `Mini-ECI` and on $\ge 8/12$ environments, and comes within $[-0.5, +\infty)$ `Mini-ECI` of the hindsight **per-environment Portfolio Oracle** ($\max_{A \in \{1..6\}} \text{Score}(A, e)$).
2. **Warm-Start (`R13 -> R14`) vs. Cold-Start (`R14` from Scratch) Sample-Efficiency Dividend:** Comparing `Arm-A` (cold-start `R14`) against `Arm-B` (initialized from the `R13-D6` multi-algorithm checkpoint and fine-tuned purely with the `R14` task objective), both arms pass `R14-D1` and `R14-D2`, and `Arm-B` reaches the `R14-D1` target `Mini-ECI` threshold in **$\le 50\%$ of the training steps** required by `Arm-A`.
