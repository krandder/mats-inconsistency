# Pre-Registration Protocol: Round 18 (`R18-OPENSPIEL-STRATEGO`) — Real `OpenSpiel` (`pyspiel`) & Full Multi-Turn `Stratego` External Benchmark & Training-Paradigm Shootout on `NVIDIA L4` GPU

- **Document ID:** `PREREG-R18-OPENSPIEL-STRATEGO`
- **Status:** **LOCKED PRIOR TO ROUND-18 GPU EXECUTION**
- **Date:** 2026-10-19
- **Author:** Society of Thinkers — Member 26 (Round 18 / Pillar 1 Lead)
- **Repository:** `/home/agent/workspaces/inconsistency-search` (`futarchy-fi/inconsistency-search`)
- **Target Hardware:** `NVIDIA L4` GPU (`24 GB` VRAM) via `infra/gpu-run.sh`
- **Environments & Engines:**
  - External C++ `OpenSpiel` (`pyspiel`): `kuhn_poker`, `leduc_poker`, `phantom_ttt`, `dark_hex(board_size=4)`, `hex(board_size=5)`, `connect_four`
  - Full Multi-Turn `StrategoEngine` ([`src/envs/stratego_full.py`](../src/envs/stratego_full.py)): `Stratego-4x4` (`Barrage-4x4`), `Stratego-6x6` (`Barrage-6x6`), and `Stratego-8x8` (`Barrage-8x8`)
- **Benchmark Script:** [`benchmarks/run_r18_openspiel_stratego.py`](run_r18_openspiel_stratego.py)
- **Pre-Registered Seeds:** `101, 102, 103` ($N=3$ independent training & evaluation seeds; two-sided 95% confidence intervals via paired Student-$t$ critical value $t_{0.975, 2} = 4.303$)

---

## 0. Mandate & Scientific Motivation

Kelvin's Mandate for Round 18:
> *"We need hard, cold results on either external benchmarks, or things that \*could\* be external benchmarks. I want numbers. On benchmarks I understand. I want to see open problems, or benchmarks that are unsaturated (or unsaturated on small models -> public benchmarks that require smaller models would be great). We need our algorithm to beat all other on stratego and other games, etc, we need generalization across games. I need to compare our training with other kinds of training and see real progress."*

Why existing self-play RL and search paradigms struggle on **unsaturated imperfect-information tactical games** (`Stratego`, `Dark Hex`, `Phantom Tic-Tac-Toe`, `Leduc Poker`) at small/medium model scale (`~3.2M` parameters):
1. **Model-Free Policy Gradient (`PPO_GAE`, `GRPO_RLOO`):** Lacks test-time tree search and explicit combinatorial constraint propagation when an opponent piece moves or reveals its rank.
2. **Perfect-Information Tree Search (`AlphaZero_PUCT`, `Gumbel_MuZero`):** Splits nodes by state history rather than information set $\mathcal{I}$ and lacks counterparty reach weighting $\pi_{-i}$ and regret matching, causing severe **strategy over-determinization (exploitability collapse)** in hidden-information games (`Kuhn/Leduc Poker`, `Phantom TTT`, `Stratego`).
3. **Imperfect-Information Game-Theoretic Baselines (`R_NaD_DeepNash`, `Deep_CFR`):** Regularized Nash Dynamics (`DeepNash`, Perolat et al. *Science* 2022) and `Deep CFR` (Brown et al. *ICML* 2019) converge toward low-exploitability equilibria in tabular/single-game regimes, but **lack multi-step deductive constraint propagation (`CDCL` nogoods + `Multi-Witness Disjunctive BP`)** over the opponent's hidden initial piece setup when a piece moves 1 square (`UNKNOWN_MOVED`: cannot be `Flag` or `Bomb`), moves $>1$ square (`Scout` revealed), or survives combat. Furthermore, their flat policy heads struggle to generalize **zero-shot across games (`LOGO`)** or **out-of-distribution across board sizes (`4x4 + 6x6 -> 8x8`)**.
4. **`GLC_v2` (`GLCAOMWorldModel` + `GLCSearch-v2`, Ours):** Unifies:
   - **`CDCL` Hidden-Layout Nogood Pruning:** Immediately eliminates inconsistent opponent setup assignments when pieces move (`not in {Flag, Bomb}`), sprint (`== Scout`), or engage in combat.
   - **`Multi-Witness Disjunctive BP` (`Z^W_aux`):** Tracks joint multi-witness posterior beliefs over the opponent's surviving hidden piece inventory (`Flag`, `Bomb`, `Spy`, `Scout`, `Miner`, `Sergeants..Marshal`) under strict LIR separation $\chi = (0, 1)$.
   - **`CFR+` Infoset Regret Matching (`M_infoset`):** Pools counterfactual values across infoset-equivalent tree nodes weighted by opponent reach $\pi_{-i}$ to prevent deterministic exploitability collapse.
   - **`Gumbel Sequential Halving` (`rank_sib`):** Guarantees monotone policy improvement at the root under small/medium simulation budgets $B \in \{1, 4, 16, 64\}$.

---

## 1. Matched Architecture (`~3.2M` Parameters) & 8-Arm Training-Paradigm Shootout

To ensure an apple-to-apple training shootout with zero confounders from parameter count or compute budget, all 8 paradigms use a **matched `~3.2M`-parameter neural backbone** (`d_model = 256, n_layers = 4, n_heads = 4, d_ff = 1024` + game/board spatial & infoset encoder + paradigm-specific readout/search heads, within $\pm 3\%$ parameter count of `3.20M` parameters) and are trained from scratch on `NVIDIA L4` GPU at matched training episodes and FLOPs per seed (`101, 102, 103`):

| Arm ID | Paradigm Name | Citation | Core Training & Search Mechanism |
| :---: | :--- | :--- | :--- |
| **1** | **`PPO_GAE`** | Schulman et al. (2017) | Clipped surrogate objective ($\epsilon=0.20$), Generalized Advantage Estimation ($\lambda=0.95, \gamma=0.99$), value critic head $V_\phi(I)$, entropy bonus $\beta_{\text{ent}}=0.02$. |
| **2** | **`GRPO_RLOO`** | DeepSeekMath / Ahmadian et al. (2024) | Group-Relative Policy Optimization over $G=8$ self-play trajectories per starting configuration without a value critic: $\hat{A}_g = (R_g - \bar{R}_{-g}) / (\mathrm{std}(R) + \epsilon)$ + KL anchor penalty. |
| **3** | **`AlphaZero_PUCT`** | Silver et al. (*Science* 2018) | MCTS with PUCT selection $Q(s,a) + c_{\text{puct}} P(s,a)\frac{\sqrt{N(s)}}{1+N(s,a)}$, leaf value backup to ancestors, and cross-entropy distillation of root visit distribution $\pi_{\text{MCTS}} \propto N(s_0, a)^{1/\tau}$ + MSE value loss. |
| **4** | **`Gumbel_MuZero`** | Danihelka et al. (*ICLR* 2022) | Root Gumbel-Top-$k$ sampling without replacement + Sequential Halving over budget $B$ + completed Q-value policy improvement target $\pi'(a|s_0) = \mathrm{softmax}(\log \pi_\theta(a|s_0) + \sigma(\hat{q}(a)))$. |
| **5** | **`R_NaD_DeepNash`** | Perolat et al. (*Science* 2022) | DeepMind's *Stratego* algorithm: entropy-regularized reward transformation $r_\eta(I, a) = r(I, a) - \eta \log\frac{\pi_\theta(a|I)}{\pi_{\text{reg}}^{(m)}(a|I)}$ ($\eta=0.20$), replicator-dynamics policy-gradient update, V-trace value estimation, and stage-wise anchor annealing $\pi_{\text{reg}}^{(m+1)} \leftarrow \pi_\theta^{(m)}$. |
| **6** | **`Deep_CFR`** | Brown et al. (*ICML* 2019) | External-sampling Counterfactual Regret Minimization with neural instantaneous regret network $R_\theta(I, a)$ trained on reservoir-sampled counterfactual regrets, paired with average-strategy buffer network $\bar{\sigma}_\phi(I, a)$ via ReLU regret matching $\sigma^{(t)}(a|I) \propto \mathrm{ReLU}(R_\theta(I,a))$. |
| **7** | **`GLC_v2_Specialist`** | **Ours (Single-Game)** | `GLCAOMWorldModel` + `GLCSearch-v2` trained per game with (i) any-order hidden-state belief loss + `CDCL` nogood consistency on opponent setup slots (`chi=(0,1)`), (ii) `LSO` counterfactual subtree credit, (iii) `CFR+` infoset regret matching over $\mathcal{M}_{\text{infoset}}$, and (iv) root `Gumbel Sequential Halving`. |
| **8** | **`GLC_v2_Generalist`** | **Ours (Unified Cross-Game)** | Single unified `~3.2M`-parameter `GLCAOMWorldModel` + `GLCSearch-v2` trained jointly across all games using the 6-token hierarchical game header `[GAME_FAMILY, GAME_ID, GRID_H, GRID_W, INVENTORY_SPEC, UTILITY_SPEC]` and coordinate-free spatial/infoset relational tokens. |

---

## 2. Pre-Registered Falsifiable Gates (`R18-D0` through `R18-D6`)

All statistical comparisons across `Seeds 101, 102, 103` report `mean ± std` and 95% confidence intervals $\bar{\Delta} \pm t_{0.975, 2} \cdot \mathrm{SE}(\Delta)$ where $t_{0.975, 2} = 4.303$.

### Gate `R18-D0`: Zero Oracle Leakage & Genuine External `OpenSpiel` / Multi-Turn `Stratego` Verification (Hard Prerequisite)
1. **Static AST Zero-Leakage Audit (`S1`–`S6`):** [`src/eval/ast_leak_auditor.py`](../src/eval/ast_leak_auditor.py) reports `0` violations across `src/models/glc_aom.py`, `src/search/neural_tree_controller.py`, `src/envs/stratego_full.py`, and `benchmarks/run_r18_openspiel_stratego.py`, with `3/3` synthetic leak canaries caught.
2. **Runtime `PoisonTensor` Invariance (`0.000000000`):** `3/3` `PoisonTensor` traps fire `OracleLeakError` on unauthorized access to `exact_marginals`, `target_tokens`, and `query_targets`. Evaluating test-time search policies on stripped vs. zeroed vs. permuted oracle batches yields exact `global_max_oracle_diff == 0.000000000`.
3. **Runtime Hidden-State Infoset Invariance (`0.000000`):** Across all imperfect-information `OpenSpiel` games (`kuhn_poker`, `leduc_poker`, `phantom_ttt`, `dark_hex`) and `StrategoEngine` ($4\times 4, 6\times 6, 8\times 8$), permuting the opponent's unrevealed private cards or unrevealed hidden piece identities consistently with the public movement/combat history produces **exact `0.000000` max policy drift** ($\max_{a} |\pi(a \mid s^{(1)}) - \pi(a \mid s^{(2)})| = 0.000000$).
4. **Genuine Engine Verification:**
   - Real C++ `pyspiel.load_game` verification for `kuhn_poker`, `leduc_poker`, `phantom_ttt`, `dark_hex(board_size=4)`, `hex(board_size=5)`, and `connect_four`.
   - Full multi-turn `StrategoEngine` (`src/envs/stratego_full.py`) unit verification covering:
     * Private initial setup phase on $4\times 4$ (6 pieces/side, 1 central lake pair), $6\times 6$ (10 pieces/side, 2 central lakes), and $8\times 8$ (16 pieces/side, 2 central $2\times 1$ lakes), up to `80` plies per game.
     * Exact combat rules: `Flag (0)` capture wins immediately; `Spy (1)` defeats `Marshal (10)` when `Spy` attacks first (`attacker=1, defender=10 -> Spy wins`), but loses when `Marshal` attacks `Spy` (`attacker=10, defender=1 -> Marshal wins`); `Scout (2)` moves/attacks across multiple open squares in a straight line without jumping over pieces or lakes, and moving $>1$ square sets public `UNKNOWN_SCOUT` flag; `Miner (3)` defuses `Bomb (11 / B)` (`attacker=3, defender=11 -> Miner wins`), whereas any other mobile piece attacking a `Bomb` is destroyed (`defender=11 wins`); equal ranks destroy both pieces; higher rank (`4..10`) defeats lower rank.
     * Strict fog-of-war public observation state: opponent pieces are observed only as `UNKNOWN_UNMOVED` (could be any remaining piece including `Flag` or `Bomb`), `UNKNOWN_MOVED` (moved 1 step $\implies \notin \{\text{Flag}, \text{Bomb}\}$), `KNOWN_SCOUT` (moved $>1$ step in a single turn), or `KNOWN_RANK(r)` (survived combat or revealed in battle).

### Gate `R18-D1`: External `OpenSpiel` Exact `NashConv` / Best-Response Exploitability
Evaluated on `kuhn_poker`, `leduc_poker`, `phantom_ttt`, and `dark_hex(board_size=4)` across training checkpoints (`25%, 50%, 75%, 100%` of training budget) and test-time search budgets $B \in \{1, 4, 16, 64\}$ using exact `open_spiel.python.algorithms.exploitability.nash_conv` (on `kuhn_poker` and `leduc_poker`) and exact game-tree best-response traversal over sampled public belief trees (on `phantom_ttt` and `dark_hex(board_size=4)`):
- **Pass Criteria:**
  1. At $B = 64$, `GLC_v2_Specialist` achieves the **#1 lowest `NashConv`** across all 4 imperfect-information `OpenSpiel` games, and `GLC_v2_Generalist` ranks top-2 (ahead of all 6 external baselines).
  2. `GLC_v2_Specialist` reduces mean 4-game `NashConv` at $B=64$ by **$\ge 40.0\%$** relative to `AlphaZero_PUCT` and `Gumbel_MuZero` (which lack infoset pooling and over-determinize), by **$\ge 50.0\%$** relative to `PPO_GAE` and `GRPO_RLOO`, and by **$\ge 15.0\%$** relative to `R_NaD_DeepNash` and `Deep_CFR`, with paired 95% CI excluding zero ($t > 4.303$).

### Gate `R18-D2`: Full Multi-Turn `Stratego` ($4\times 4, 6\times 6, 8\times 8$) Round-Robin Tournament (`Win-Rate Matrix`, `Elo` & `mElo`)
Full 8-agent round-robin tournament across `Stratego-4x4`, `Stratego-6x6`, and `Stratego-8x8` (up to 80 plies per game, alternating Red/Blue seats with paired initial setups) at test-time budgets $B \in \{1, 16, 64\}$, scored via [`BayesianEloLeague`](../src/eval/elo_league.py) (`Elo`) plus Balduzzi et al. (2018) multidimensional `mElo` ($k=1$ Hodge/Schur cyclic rock-paper-scissors decomposition $\hat{P}_{ij} = \sigma(\text{elo}_i - \text{elo}_j + \mathbf{u}_i^\top \mathbf{\Omega} \mathbf{u}_j)$):
- **Pass Criteria:**
  1. At $B=64$, `GLC_v2_Specialist` and `GLC_v2_Generalist` rank **#1 and #2** in pooled `Stratego` `BayesianEloLeague` rating and `mElo` across all three board sizes ($4\times 4, 6\times 6, 8\times 8$).
  2. Head-to-head win rate (wins $+ 0.5 \times$ draws) of `GLC_v2_Specialist` at $B=64$ exceeds **$\ge 56.0\%$ against `R_NaD_DeepNash`**, **$\ge 58.0\%$ against `Deep_CFR`**, and **$\ge 62.0\%$ against `Gumbel_MuZero`, `AlphaZero_PUCT`, `GRPO_RLOO`, and `PPO_GAE`** across `Stratego-4x4`, `Stratego-6x6`, and `Stratego-8x8` (paired 95% CI $> 50.0\%$).

### Gate `R18-D3`: Cross-Game Generalization — Leave-One-Game-Out (`LOGO`) Zero-Shot & Few-Shot Transfer
Train unified `~3.2M`-parameter generalist models (`GLC_v2_Generalist`, `PPO_Generalist`, `GRPO_Generalist`, `Gumbel_MuZero_Generalist`, and `R_NaD_Generalist`) on 5 games while holding out:
- **Holdout A:** `Stratego-6x6` (trained on `Kuhn Poker`, `Leduc Poker`, `Phantom TTT`, `Dark Hex 4x4`, and `Hex 5x5`)
- **Holdout B:** `Dark Hex 4x4` (trained on `Kuhn Poker`, `Leduc Poker`, `Phantom TTT`, `Stratego-6x6`, and `Connect-4`)
Evaluate **Zero-Shot (`0` episodes)** and **Few-Shot Adaptation (`25, 100, 400` self-play games)** on the held-out game against the fully trained single-game `R_NaD_DeepNash` specialist:
- **Pass Criteria:**
  1. `GLC_v2_Generalist` achieves a **Zero-Shot Win Rate $\ge 45.0\%$** against the fully trained single-game `R_NaD_DeepNash` specialist on both held-out games, outperforming `PPO_Generalist`, `GRPO_Generalist`, `Gumbel_MuZero_Generalist`, and `R_NaD_Generalist` by **$\ge +12.0\text{ pp}$** ($t > 4.303$).
  2. On the few-shot adaptation curve (`0, 25, 100, 400` games), `GLC_v2_Generalist` crosses **$50.0\%$ win rate against single-game `R_NaD_DeepNash` within $\le 100$ self-play games** and reaches **$\ge 56.0\%$ at `400` games**.

### Gate `R18-D4`: Out-of-Distribution (`OOD`) Board-Size Generalization on `Stratego` (`4x4 + 6x6 -> Zero-Shot 8x8`)
Train spatial-relational generalist policies on `Stratego-4x4` and `Stratego-6x6` only, and evaluate **zero-shot on `Stratego-8x8`** (64 squares, 16 pieces per side, full 12-rank Stratego combat roster) at $B = 64$:
- **Pass Criteria:**
  1. `GLC_v2_Generalist` (`4x4+6x6 -> 8x8` zero-shot) achieves a head-to-head win rate of **$\ge 55.0\%$ against `R_NaD_DeepNash`** (`4x4+6x6 -> 8x8` zero-shot) and **$\ge 60.0\%$ against `Gumbel_MuZero`** (`4x4+6x6 -> 8x8` zero-shot), with paired 95% CI excluding $50.0\%$.
  2. Against the in-distribution `R_NaD_DeepNash` trained directly on `Stratego-8x8`, zero-shot `GLC_v2_Generalist` (`4x4+6x6 -> 8x8`) retains **$\ge 48.0\%$ win rate** at $B=64$.

### Gate `R18-D5`: Opponent Hidden Setup Deduction Accuracy & Posterior KL Across Plies (`Ply 10, 20, 30`)
At plies $t \in \{10, 20, 30\}$ of `Stratego-4x4`, `Stratego-6x6`, and `Stratego-8x8` games, evaluate each agent's predicted marginal distribution $\hat{P}_t(Z_{\text{opp}} \mid h_{\le t})$ over the opponent's unrevealed surviving pieces (`Flag`, `Bomb`, `Spy`, `Scout`, `Miner`, `Marshal`, and intermediate officers) against the exact Bayesian constraint-and-likelihood posterior $P^*(Z_{\text{opp}} \mid h_{\le t})$ and true hidden identity:
- **Pass Criteria:**
  1. By ply $30$, `GLC_v2_Specialist` and `GLC_v2_Generalist` achieve **Key-Piece (`Flag`, `Bomb`, `Spy`, `Miner`, `Marshal`) Top-1 Localization Accuracy $\ge 68.0\%$** across `Stratego-4x4/6x6/8x8`, beating `R_NaD_DeepNash`, `Deep_CFR`, `Gumbel_MuZero`, `AlphaZero_PUCT`, `GRPO_RLOO`, and `PPO_GAE` by **$\ge +15.0\text{ pp}$**.
  2. Posterior $\mathrm{KL}(P^* \Vert \hat{P}_t)$ for `GLC_v2` decreases monotonically from ply $10 \to 20 \to 30$ and is **$\ge 35.0\%$ lower** at ply $30$ than every external baseline ($t > 4.303$).

### Gate `R18-D6`: Test-Time Compute Scaling ($B \in \{1, 4, 16, 64\}$) Across Perfect-Info (`Hex 5x5`, `Connect-4`) & Imperfect-Info Games
Evaluate test-time compute scaling across $B \in \{1, 4, 16, 64\}$ on both external `OpenSpiel` perfect-information tactical games (`hex(board_size=5)`, `connect_four`) and imperfect-information games (`Stratego-4x4/6x6/8x8`, `leduc_poker`, `phantom_ttt`, `dark_hex(board_size=4)`):
- **Pass Criteria:**
  1. On perfect-information `Hex 5x5` and `Connect-4`, `GLC_v2` matches or beats `AlphaZero_PUCT` and `Gumbel_MuZero` at $B=64$ (win rate $\ge 52.0\%$ vs. `AlphaZero_PUCT`, $\ge 50.0\%$ vs. `Gumbel_MuZero`, and $\ge 70.0\%$ vs. critic-free `GRPO_RLOO` / `PPO_GAE`).
  2. Across all environments (both perfect- and imperfect-information), `GLC_v2_Specialist` and `GLC_v2_Generalist` maintain **strictly monotone positive Elo scaling** across $B \in \{1, 4, 16, 64\}$ with total Elo gain **$\Delta \mathrm{Elo}(B=1 \to 64) \ge +140.0$** and log-linear fit **$R^2 \ge 0.92$**.
