# Society of Thinkers — Charter, Mathematical Brief & Round-1 Rubric

**Repository:** `/home/agent/workspaces/inconsistency-search` (`futarchy-fi/inconsistency-search`)  
**Goal:** Design, red-team, implement, benchmark, and iterate on a **Bitter-Lesson-Pilled, Multi-Game Any-Order World Model + Learned KV-Cache Search Machine (`GLC-AOM`)**, evaluated across a rich suite of games/puzzles/market simulations via **Elo leagues** and a **Mini-Model Epoch Capability Index (`Mini-ECI`)** across test-time compute steps $B = 0 \to 128+$.

---

## 1. Core Architectural Desiderata (Zero Hand-Picked Search Heuristics)

1. **UniversalKV-Cache Tree Machine (`SELECT_NODE(j)` + `STEP(v, z)`):**
   - At step $b \in \{1, \dots, B\}$, the search state is a tree-paged bank of KV-cache checkpoints $\mathcal{C}_b = \{C_0, C_1, \dots, C_b\}$, where each node $C_j$ is created by appending 1 variable/action assignment $(v_j, z_j)$ to its parent $C_{\text{parent}(j)}$ and emits a summary token $h_j \in \mathbb{R}^{D_{\text{amp}}}$.
   - **No hardcoded rollout depth ($4$ plies), no hardcoded stall threshold ($H \ge 0.25$), and no hardcoded beam wave count ($L=4$):**
     - Continuing the current tip ($j_{b+1} = b$) extends the current chain by 1 step.
     - Selecting an earlier node ($0 \le j_{b+1} < b$) terminates/pauses the current chain and forks from $C_{j_{b+1}}$ (or restarts from root $C_0$).
2. **Active, Self-Similar / Fractal Prospectus Plans ($p_r \in \mathbb{S}^{D_{\text{amp}}-1}$):**
   - Instead of conditioning only on the first step, a **Plan Head** $q_{\text{plan}}(p \mid C_j, K_{\text{done}})$ samples an active target direction $p_r \in \mathbb{S}^{D_{\text{amp}}-1}$ at any node $C_j$, and the step policy $\pi_{\text{step}}(v, z \mid C_t, p_r)$ is steered by $p_r$ along the chain.
   - Because the *same* $q_{\text{plan}}$ is applied at the root $C_0$ (coarse unresolved subspace $\to$ high-level plan), intermediate nodes $C_j$ (medium-level sub-plan), and near-leaf nodes $C_{j'}$ (low-level tactical micro-plan), plans are **self-similar across scales** with no hardcoded number of hierarchy levels.
   - Trained via **Hindsight Plan Conditioning (HER)** ($p = k^{\text{post}}_{j \to j'}$) + marginal consensus loss reduction $\Delta \mathcal{L}_{\text{consensus}}$.
3. **Bidirectional Latent Lemma Space ($D_{\text{amp}} \gg |V|$) & Direct Query Rollouts ($Q(A, B)$):**
   - $|V|$ is the compact, human-legible variable interface (cells, pieces, actions, market prices, utility mandates).
   - The extra $D_{\text{amp}} - |V|$ dimensions ($Z_{\text{aux}}$, LIR mask $\chi = (0, 1)$) store learned "alien" lemmas and couplings, cleared by weight-tied `looped_attn_T3` into $h_{\text{cons}} \in \mathbb{R}^{D_{\text{amp}}}$.
   - **Bitter-lesson bidirectional training:** Train the Any-Order World Model on its standard random-permutation autoregressive loss conditioned on $h_{\text{cons}}$:
     $$\mathcal{L}_{\text{any-order}}(h_{\text{cons}}; x^*) \;=\; -\mathbb{E}_{\pi}\!\left[\, \sum_{t=1}^{|V \setminus O|} \log q_\theta\!\Big(x^*_{\pi(t)} \;\Big|\; x_O, \, x^*_{\pi(<t)}, \, h_{\text{cons}}\Big) \,\right],$$
     plus optional relational query tokens $Q(A, B)$ so rollouts can directly target questions like `P(A == B)`.
4. **Present-Time Episodic Memory Variables ($M_t$) for Sequential Imperfect-Information Games:**
   - At turn $t$, the cleared consensus state $h_{\text{cons}}^{(t-1)}$ from turn $t-1$ enters as explicit **Memory Tokens $M_t$** at the root $C_0$.
   - Present-time search resolves inconsistencies between current hidden-state hypotheses $Z_t$ and memory tokens $M_t$ without re-simulating the past history from $t=0$.
5. **Anytime Monotonicity Guarantee ($\mathbb{E}[\mathrm{KL}(P^* \Vert \mu_{b+1})] \le \mathbb{E}[\mathrm{KL}(P^* \Vert \mu_b)]$ for $b = 0 \dots 128+$):**
   - Root anchor token $C_0$ always present + nested gated residual updates + normal-equation deduplication + zero-extra-pass self-consistency / $\theta_{\text{trust}}$ precision gating.

---

## 2. The Multi-Game Benchmark Suite, Agent Populations, Elo Leagues & `Mini-ECI`

We evaluate a **population of agents and model architectures** across **8+ environments** under a unified token interface:
1. **`Sudoku-6x6` & `Sudoku-9x9-Extreme`:** NP-complete constraint deduction & multi-stall search ($B = 0 \to 128$ steps).
2. **`Coupled-ShipHunt-6x6` & `8x8`:** Non-local fleet-coupled sensor inference & preference-skew policy independence.
3. **`Latent-Equivalence-Parity` (`Equiv-SAT`):** Synthetic and embedded combinatorial equivalence puzzles testing direct `P(A == B)` rollout queries and bidirectional $D_{\text{amp}} \gg |V|$ lemmas.
4. **`Minesweeper-6x6` / `Mastermind`:** Epistemic uncertainty vs. forced deduction under risk.
5. **`Hex-5x5` / `Connect-4`:** Perfect-information tactical lookahead & Elo tournament ladder.
6. **`Micro-Stratego-4x4` & `6x6`:** Hidden-information sequential game with Memory Tokens $M_t$ & Elo tournament ladder.
7. **`Galanis-Market` (from `futarchy-fi/simulations`):** 4 information-structure prediction-market equilibria & belief calibration.
8. **`Proposal-Poker` (from `futarchy-fi/simulations`):** Futarchy governance staking with private signal $s_i$ and private bias $y_i$, testing `LIR-Masked` vs. `Token-Masked` vs. `Fused` RL and zero-retraining utility swaps.

### Shared Evaluation Metrics:
- **Test-Time Compute Scaling Curves ($B \in \{0, 1, 2, 4, 8, 16, 32, 64, 128\}$):** Belief KL / Brier / Witness rate / Win rate vs. compute budget $B$.
- **Multiplayer Elo Ratings:** Round-robin & Swiss tournaments across model families (`1-pass`, `Flat-MC P3`, `PUCT`, `Hardcoded-Beam P8`, `Learned-KV-Tree`, `Fractal-Prospectus-Tree`, `Multi-Game Generalist` vs. `Single-Game Specialist`) and across compute budgets $B \in \{0, 4, 16, 64, 128\}$.
- **Mini-Model Epoch Capability Index (`Mini-ECI`):** A 1D IRT (Item Response Theory) latent capability score fitted jointly across all 8 benchmark tasks and Elo ladders, mapping how architecture + test-time compute $B$ shift general capability.
