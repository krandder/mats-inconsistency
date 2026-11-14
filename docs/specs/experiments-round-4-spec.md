# Round-4 Experimental Specification (`experiments-round-4-spec.md`)

**Author:** `gemini` (Argon) for Kelvin Azevedo Santos & `opus-mats`  
**Date:** 2026-10-16 18:22Z  
**Budget Cap:** $\le \$200$ total across Round 4 (estimated total below: **$\$38\text{--}\$52$** ceiling across all 3 experiments)  
**Target Turnaround:** $\le 4.5\text{ hours}$ wall time (all 3 experiments run in parallel across `eurisko` and `calibrated-ai`)  
**Hard Constraints:** Zero ground-truth posteriors or rule verifiers used during training or hyperparameter fitting; 3 seeds (`101, 102, 103`) per arm; paired 95% $t$-intervals ($\text{df}=2, t=4.303$); pre-registered `PREREG-R4.md` + SHA256 before launch.

---

## 0. Overview & Ranking of Round-4 Experiments

Rounds 1–3 established that any-order search cuts posterior KL by $13\text{--}46\%$ once the 1-step law is learned and that recall-floored null search cuts deep-contradiction Sudoku loss by $>99\%$, **but six core claims of the direction note (*Inconsistency Is All You Need*) remain untested or could not be separated on the $5\times 5$, $6\times 6$, and $12\times 10$ deterministic toys**:
1. **Policy-independent consensus via deduplication** (counting each distinct queried conditional once so skewed search policies converge to the same belief).
2. **Self-trustworthiness ($\theta_{\text{trust}}$) $\neq$ output entropy (`stiff`)** (in deterministic `E-CA`, $H(P^*) \approx 0$ so total surprise was dominated by output entropy, tying $\beta_{\text{int}}$ and $-\!H(q_\theta)$ at $0.700$ vs. $0.697$).
3. **Factored 4-block parameter split ($\theta_{\text{real}}, \theta_{\text{amp}}, \theta_{\text{trust}}, \theta_{\text{agg+pol}}$) + MCTS/Ataraxos inclusion** over a **learned world model** in a partially observed game where belief search and multi-step lookahead both matter.
4. **Consensus Landmark branching & subspace refuted mass vs. raw count** on a domain where greedy confidence-first chains genuinely stall (`hyb` never branched on $12\times 10$ `E-CA`, and full-board completion mass $C_M \sim 10^{-4}$ pinned $\theta=1000$ on $6\times 6$ Sudoku).
5. **Honesty under rising-reward RL ($\chi_{\text{belief}}=0, \chi_{\text{action}}=1$) + zero-retraining Faith/Utility swap** at model scale where reward genuinely rises and the forecaster is non-trivial (`lir-agent` Round 1 had flat reward and a collapsed 0.6B forecaster, and `faith_swap` was never run).

We rank three complementary experiments to run **in parallel**:

| Rank | ID & Title | Primary Claims Tested | Executor & Codebase | Est. Cost & Wall Time |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **`R4-E1` (`E-MS`)**: 4×4 Diagonal-Touch Micro-Stratego (+ 6×6 8-Piece Barrage Rung 2) with a Learned Factored World Model | Full 4-block split ($\theta_{\text{real}}, \theta_{\text{amp}}, \theta_{\text{trust}}, \theta_{\text{agg+pol}}$); Policy-independence under skewed Left/Right flank policies (`D2`); Trust $\neq$ Entropy (`D3`); Any-order belief vs. Ataraxos row-major (`D1`); Utility swap (`D4`); Tournament vs. Ataraxos 1-step & Tabular MCTS (`D5`). | **`eurisko`** (or `opus-mats` subagent) using `/home/agent/workspaces/mats/work/micro-stratego/env.py` + `eca/gpu/gpu-run-zones.sh` | **$\$8\text{--}\$12$**<br>(3–4 L4 VMs, $\sim 3.0\text{ h}$) |
| **2** | **`R4-E2` (`E-SUD-9x9`)**: Real $9\times 9$ Sudoku (81 Cells) with Landmark Tree Branching & Subspace Refuted Mass | Moves beyond $6\times 6$ toy; tests Consensus Landmark branching at stall points ($H \ge 0.40$ on all empty cells, fixing `R3-D2b`); **Subspace Refuted Mass $C_M^{\text{sub}}$ vs. Count** (fixing `R3-D1c` without $\theta=1000$); $\theta_{\text{real}}/\theta_{\text{amp}}$ distillation split. | **`calibrated-ai`** extending `/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/E-SUD/` | **$\$12\text{--}\$15$**<br>(3 L4 VMs, $\sim 3.5\text{ h}$) |
| **3** | **`R4-E3` (`E-LIR-RL`)**: Honest Beliefs, Self-Trust & Faith-Swap Under Rising-Reward RL (`Qwen3-1.7B` / `8B` on ShipHunt $7\times 7$ & Manifold/MiniBench) | Closes `lir-agent` Round-1 gaps: informative forecaster gate ($r > 0.25$), rising task reward ($\ge +15\%$), $0\%$ false-note belief drift under LIR parameter mask vs. fused desire distortion $\frac{\lambda}{2}\|\nabla U\|^2$, and live **Faith/Utility Swap** (`faith_swap`). | **`eurisko`** (`lir-agent/harness`) + **`calibrated-ai`** (`minibench` / Tinker `Qwen3-8B` LoRA) | **$\$18\text{--}\$25$**<br>(3 L4 VMs + Tinker, $\sim 4.0\text{ h}$) |

---

## 1. Rank 1 — `R4-E1` (`E-MS`): Micro-Stratego with a Learned Factored World Model

### 1.1 Question It Answers
Can a single factored any-order architecture ($\theta_{\text{real}}, \theta_{\text{amp}}, \theta_{\text{trust}}, \theta_{\text{agg+pol}}$)—trained only on realized game transitions with **zero simulator access during search**—(1) beat Ataraxos's fixed row-major belief unmasking (`D1`), (2) converge to a **policy-independent consensus** when rollout policies are heavily skewed toward the Left vs. Right Flank (`D2`), (3) separate **world-model trustworthiness ($\theta_{\text{trust}}$) from predictive entropy (`stiff`)** when genuine 50/50 hidden-piece risk coexists with out-of-distribution transition errors (`D3`), (4) preserve $0.000$ belief drift on $\theta_{\text{real}}$ during search distillation and adapt instantly to a test-time **Utility Swap** (`D4`), and (5) defeat 0-step policy, Ataraxos 1-step MMD search, and Tabular PUCT MCTS in head-to-head play (`D5`)?

### 1.2 Environment, Data & Code-Level Spec
- **Ready Prototype:** `/home/agent/workspaces/mats/work/micro-stratego/env.py` (already implemented and verified on 500 games).
- **Rung 1 Board ($4\times 4$ Diagonal-Touch, 6 pieces/side):**
  - Red starts on `[(1,0), (1,1), (0,0), (0,1), (0,2), (0,3)]` (`xxxx / xx..`); Blue starts on `[(2,2), (2,3), (3,0), (3,1), (3,2), (3,3)]` (`..oo / oooo`).
  - Pieces `{Flag(1), Bomb(2), Scout(3), Miner(4), General(5), Marshal(6)}` ($6! = 720$ setups/side).
  - **Disjoint Setup Split (0% Setup Overlap):** Partition the $720$ initial setups into **`540` Train / `90` Val / `90` Test setups** (following `E-CA` Round-3's disjoint-row-0 discipline so $\theta_{\text{real}}$ cannot memorize full initial setups!).
  - **Exact Bayesian Belief Oracle (`env.py:exact_slot_marginals`):** Computes the exact $6\times 6$ marginal posterior $P^*(Z_s = p \mid h_t)$ over all $720$ permutations consistent with public movement (`has_moved` $\implies \notin \{F, B\}$; `moved_long` $\implies = S$) and battle reveals. At turn 8, a mean of `38.1 / 720` setups remain consistent.
- **Rung 2 Board ($6\times 6$ 8-Piece Barrage Extension in `env_6x6.py`):**
  - Adds 2 center lake squares `(2,2), (3,3)` and 8 pieces/side (`Flag, 2 Bombs, 2 Scouts, Miner, General, Marshal`, $8!/(2!2!) = 10{,}080$ setups/side, mean `11.4` legal moves/turn) to verify that key-space deduplication (`D2`) and search win rates (`D5`) scale to higher branching factors.
- **Dataset Generation (`generate_data.py`):**
  - Generate $60{,}000$ self-play / mixed-temperature ($\tau \in \{0.5, 1.0, 2.0\}$ heuristic + MCTS) games on the `540` train setups, $5{,}000$ calibration games on `90` val setups, and $3{,}000$ evaluation games on `90` test setups.

### 1.3 Model Architecture & LIR Parameter Split (`model_ms.py`)
Adapt `/home/agent/workspaces/eurisko-study/li-net/eca/ecanet.py` and `/home/agent/workspaces/mats/work/linear-agg/b_agg.py`:
- **Backbone ($\theta_{\text{real}}$, $L=4, d=128, 4\text{ heads}, \sim 0.85\text{M}$ params, $\chi_{\text{real}}=1, \chi_{\text{search}}=0$):**
  - Token sequence for a single hypothetical world rollout of length $n \le 10$:
    - Tokens `0..15`: $4\times 4$ board state + public movement flags (`has_moved, moved_long, revealed`) + acting player's own 6 piece types.
    - Tokens `16..21`: 6 opponent slot variables $Z_0 \dots Z_5 \in \{\text{MASK}, 1\dots 6\}$ (unmasked in any order $\sigma$).
    - Tokens `22..29`: Up to $d=4$ rollout ply pairs `(A_τ, O_{τ+1})` (move action token + battle/board transition token).
  - Trained **strictly on real game trajectories** via any-order masked cross-entropy on $Z_s$ and 1-step transition cross-entropy on $O_{\tau+1}$.
- **Consequence Extension ($\theta_{\text{amp}}$, 2-layer MLP + LoRA rank-8 on top of `stop_grad(h_real)`, $\chi_{\text{real}}=0, \chi_{\text{search}}=1$):**
  - Emits 0-step baseline logits $\ell^{(0)}(S_j)$ (over opponent slot marginals $6\times 6$ and legal action values $Q(a)$), plus at the end of rollout $r$:
    - **Source-Attention Key $k_r \in \mathbb{S}^{15}$** (using `E-CA` Round-2/3's verified `k_v2` formula: normalized projection of last-layer attention over materialized opponent slots and board squares $(Z_r, A_r) \setminus E_0$, excluding target state $h_i$), and
    - **Log-Odds Claim Shift $v_r = \Delta \ell_r = \ell_r - \ell^{(0)}(S_j)$**.
- **Self-Trustworthiness Head ($\theta_{\text{trust}}$, 3-layer MLP on `stop_grad([h_real; h_search])`, $\chi_{\text{real}}=1, \chi_{\text{search}}=0$):**
  - **Crucial Code Fix vs. `E-CA` `Mechanism A`:** In `E-CA`, `Mechanism A` predicted raw surprise $-\log q_\theta(x_i \mid x_S)$, whose expectation is $H(P^*) + \mathrm{KL}(P^* \| q_\theta) \approx H(q_\theta) + \mathrm{KL}(P^* \| q_\theta)$—dominated by the model's own entropy $H(q_\theta)$!
  - Here, $\theta_{\text{trust}}$ takes `(state features, rollout depth d, step-wise transition entropies, any-order 2-path JSD, and |Δℓ_r|)` and is trained on the calibration split to predict the **excess surprise / calibration residual**:
    $$e_{\text{excess}}(r) \;=\; \max\!\Big(0, \;\; -\log q_{\text{rollout}}(x_{\text{real}}) - H(q_{\text{rollout}})\Big) + \text{BrierGap}(q_{\text{rollout}}, x_{\text{real}}), \qquad \beta_r = \big(\hat{e}_{\text{excess}}(r) + 0.05\big)^{-1}.$$
    Because $H(q_{\text{rollout}})$ is subtracted out of the target, a genuine 50/50 unmoved-piece battle where $q_{\text{rollout}} = [0.5, 0.5]$ has $\mathbb{E}[-\log q - H(q)] = \ln 2 - \ln 2 = 0$ (**high trust $\beta_r$**), whereas a confident hallucination ($q = 0.99$ on a losing move) has $-\log(0.01) - 0.056 = 4.55$ (**near-zero trust $\beta_r$**)!
- **Consensus Calculation & Deduplicating Aggregator ($\theta_{\text{agg}}$):**
  Compare four pooling rules over $K$ rollouts anchored at $\ell^{(0)}$ (with house weight $\beta_0$):
  1. **`mix`:** Plain weighted average of rollout probabilities.
  2. **`ataraxos_mmd` (`L=1`):** 1-step tabular MMD log-odds update $\ell^* = \ell^{(0)} + \eta \sum_{r=1}^K \beta_r \Delta \ell_r$.
  3. **`kc_dedup` (Key-Cluster Support Deduplication — exact implementation of the §2.2 principle):**
     Count each distinct queried subspace once by dividing rollout $r$'s weight by its key-cluster crowding $c_r = \sum_{b=1}^K \max(0, \langle k_r, k_b \rangle)^\gamma$ (with $\gamma=2$ sharp cosine threshold):
     $$\ell^*_{\text{kc}} \;=\; \ell^{(0)} + \frac{1}{\beta_0}\sum_{r=1}^K \frac{\beta_r}{c_r} \, \Delta \ell_r.$$
     *(Note: if Rollout 1 claims $\Delta \ell_1$ on Left Flank and is duplicated 9 times with identical key $k_1$, $c_r = 9$ for each copy, so $\sum_{r=1}^9 \frac{1}{9}\Delta \ell_1 = \Delta \ell_1$—counting that information **exactly once**, even if the 9 copies have different internal sample noise!)*
  4. **`attn_L3` (`L=3` Residual-Attention Aggregator on `(k_r, Δℓ_r, β_r, c_r)`):** 3-layer softmax/LayerNorm residual attention block trained on batches with randomized flank skew $p_{\text{Left}} \sim \mathrm{Beta}(0.5, 0.5)$ to match the unbiased full-width target.

### 1.4 Pre-Registered Decision Rules (`R4-E1`)
- **`E1-D0` (World-Model Law Gate):** On held-out test setups, $\theta_{\text{real}}$ 1-step battle outcome accuracy $\ge 97\%$ and 5-known-slot piece identity KL $\le 0.03$ nats; 0-step mid-game ($t \in [4, 12]$) belief KL vs. exact 720-permutation Bayes oracle $P^*(Z \mid h_t)$ is $\ge 0.12$ nats.
- **`E1-D1` (Any-Order Belief Unmasking vs. Ataraxos Row-Major):** At $K=8$ belief rollouts over $Z_0 \dots Z_5$, `stiff_voi` unmasking cuts marginal KL to $P^*(Z \mid h_t)$ by **$\ge 20\%$** vs. Ataraxos fixed `row_major` order and cuts invalid-permutation sample rate by **$\ge 30\%$** (3-seed 95% CI $>0$).
- **`E1-D2` (Policy-Independence Under Skewed Flank Policies + Conflicting Duplicates):** On 500 two-flank test boards, run $K=24$ rollouts under $\pi_{\text{Left}}$ (80% Left Flank / 20% Right Flank) vs. $\pi_{\text{Right}}$ (20% Left / 80% Right). Both `kc_dedup` and `attn_L3` reduce the Policy-Sensitivity Divergence $\mathrm{JSD}(\mu^*(\pi_{\text{Left}}) \parallel \mu^*(\pi_{\text{Right}}))$ by **$\ge 50\%$** compared to `mix` and `ataraxos_mmd`, and cut value MSE against the unbiased $K=128$ reference by **$\ge 25\%$**.
- **`E1-D3` (Self-Trustworthiness $\theta_{\text{trust}}$ $\neq$ Output Entropy `stiff`):** Evaluating rollouts across mixed checkpoints (30% trained + 100% trained $\theta_{\text{real}}$) and depths $d \in \{1..6\}$, $\theta_{\text{trust}}$ beats output entropy $-\!H(q_\theta)$ at predicting low rollout KL $-\mathrm{KL}(P^* \parallel q_{\text{rollout}})$ by **$\ge 0.10$ Spearman** overall and **$\ge 0.15$ Spearman within matched-entropy strata** (breaking the `E-CA` `R3-D3` tie!), and cuts delusional move selection (rollout claims $>0.80$ win when true value $<0.30$) by **$\ge 40\%$**.
- **`E1-D4` (LIR Parameter Mask $\chi$ & Zero-Retraining Utility Swap):**
  1. Distilling search consensus with $\chi_{\text{real}}=0$ leaves $\theta_{\text{real}}$'s 1-step transition and belief KL **bit-identical ($0.000$ drift)** while improving $\theta_{\text{amp}}$'s 0-step value MSE by $\ge 20\%$; unmasked fused distillation ($\chi_{\text{real}}=1$) under optimistic reward degrades belief KL by $>20\%$.
  2. Swapping the explicit utility factor at test time from $U_{\text{flag}}$ (win game) to $U_{\text{miner}}$ (+1 capture enemy Miner, -1 lose own Miner) with zero retraining increases Miner-capture rate by **$\ge 2.0\times$**.
- **`E1-D5` (Tournament Win Rate on Learned World Model):** Across 400 games/pair/seed at $K=16$ rollouts/move (using only $\theta_{\text{real}}$ for rollouts), `Unified-LIR` (`stiff_voi` + Landmark Memory + `kc_dedup`/`attn_L3` + Key-Space PUCT + $\theta_{\text{trust}}$) achieves **$\ge 55\%$ effective win rate** against `P0` (0-step), `Ataraxos-1Step` (`row_major` + flat root rollouts + `ataraxos_mmd`), and `Tabular-MCTS` (standard PUCT without key sharing or $\theta_{\text{trust}}$).

---

## 2. Rank 2 — `R4-E2` (`E-SUD-9x9`): 9×9 Sudoku with Landmark Branching & Subspace Refuted Mass

### 2.1 Question It Answers
When we scale from toy $6\times 6$ Sudoku (36 cells, where 93.7% of corruptions were depth-1 and 1 rollout solved 78% of boards) to **real $9\times 9$ Sudoku (81 cells, digits 1..9)** on hard puzzles that require multi-step lookahead and branching:
1. Does **Consensus Landmark Branching** (pinning stalled partial boards $S_{\text{stall}}$ where every remaining empty cell has $H \ge 0.25$ and branching on the highest-VoI cell's candidate digits) beat flat confidence-first rollouts (`P3-Conf`, Kim et al. `arXiv:2502.06768`) and random rollouts?
2. Does **Subspace Refuted Mass $C_M^{\text{sub}}$** (accumulating the model's believed probability mass $P_\theta(z^* = d \mid S_{\text{stall}}) \in [0.05, 0.60]$ of each refuted branch at the stall Landmark, rather than the tiny $10^{-15}$ product over all 55 empty cells) finally beat raw chain count $M$ in log-loss (`D1c`) without $\theta$ pinning at the grid ceiling?

### 2.2 Data, Model & Code-Level Changes (Relative to `calibrated-ai`'s `E-SUD`)
- **Codebase:** Extend `/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/E-SUD/` (reusing `calibrated-ai`'s $9\times 9$ `fu-b2-ardm-sudoku` board generator and exact bitmask/DLX solver).
- **Dataset (`s9_hard_train.npz`, `s9_hard_val.npz`, `s9_hard_test.npz`):**
  - $300{,}000$ training boards, $2{,}000$ validation boards, $2{,}000$ test boards ($50\%$ solvable / $50\%$ unsolvable, 20–28 givens).
  - **Hard Test Split Filter:** Require every test board (both solvable and unsolvable) to survive **at least 3 rounds of naked + hidden singles (`prop_depth >= 3`)** AND require that after exhausting all deterministic naked/hidden singles, at least 15 empty cells remain where every cell has $\ge 2$ legal candidates (so greedy single-cell certainty **must** stall!).
- **Model (`esud9_net.py`):**
  - 6-layer pre-LN Transformer ($d=192, 6\text{ heads}, \text{ff}=768, \sim 2.7\text{M}$ params) over 82 tokens (81 cell tokens $\in \{\text{MASK}, 1..9\}$ + 1 `Solvable` token $\in \{\text{MASK}, 0, 1\}$), trained for $150{,}000$ steps with LIR-separated $\theta_{\text{real}}$ (trunk + 1-step cell/Solvable heads) and $\theta_{\text{amp}}$ (`stop_grad` value/amortized marginal adapter).
- **Two Crucial Code Additions for `R4-E2`:**
  1. **Consensus Landmark Tree Search (`P6-Landmark`):**
     - Starting from root $S_0$, unmask cells greedily by lowest entropy $H(q_\theta(z \mid S_t))$ while $\min_{z \notin S_t} H(q_\theta(z \mid S_t)) < 0.25$ nats.
     - The moment $\min_{z} H(q_\theta(z \mid S_t)) \ge 0.25$ (the **stall point**), **pin $S_{\text{stall}} = S_t$ as a Landmark Scenario in Consensus Memory** (caching its KV state!).
     - Pick the empty cell $z^* \notin S_{\text{stall}}$ with lowest entropy $>0.25$ (typically 2 or 3 candidate digits $d_1, d_2, \dots$ with $q_\theta(z^* = d_j \mid S_{\text{stall}}) > 0.05$), and **branch without replacement** into child Landmarks $S_{\text{stall}} \cup \{z^* = d_j\}$, running rollout completions from each child Landmark!
  2. **Subspace Refuted Mass (`C_M^sub` — Fixing the $10^{-15}$ Full-Trajectory Product Bug of `R2/R3-D1c`):**
     - Why did $\theta$ pin at $1000$ in `E-SUD` Rounds 2–3? Because multiplying $q_\theta(z_t \mid S_{t-1})$ across 22 steps gave $C_1 \approx 10^{-4}$ (and on $9\times 9$ across 55 steps gives $C_1 \approx 10^{-12}$!), requiring $\theta = 10^4\text{--}10^{12}$ in $(1 - C_M)^\theta \approx \exp(-\theta C_M)$!
     - At a Landmark stall point $S_{\text{stall}}$ where cell $z^*$ partitions the world into disjoint candidate values $d \in \{1..9\}$ with believed probabilities $p_d = q_\theta(z^* = d \mid S_{\text{stall}})$ ($\sum_{d=1}^9 p_d = 1$), if all rollouts inside branch $\{z^* = d\}$ are refuted by the model's self-verifier ($\text{check}_r < 0.5$), the **refuted subspace mass** is simply:
       $$C_M^{\text{sub}} \;=\; \sum_{d \in \text{refuted}(z^*)} q_\theta(z^* = d \mid S_{\text{stall}}) \;\in\; [0, 1].$$
       Because $C_M^{\text{sub}}$ is a true partition probability on $[0, 1]$ (e.g., refuting two candidate digits with $p_{d_1}=0.55$ and $p_{d_2}=0.35$ immediately gives $C_2^{\text{sub}} = 0.90$, leaving only $1 - 0.90 = 0.10$ surviving mass with **$\theta = 1$**!), the recall-floored update
       $$\hat{P}(\text{Solvable}=1 \mid \text{null}) \;=\; \sigma\!\Big(\mathrm{logit}(p_0) + \log\big[(1 - \eta_{\max}) + \eta_{\max}(1 - C_M^{\text{sub}})^\theta\big]\Big)$$
       operates at $\theta \in [0.5, 5.0]$ without ever hitting a numerical ceiling!

### 2.3 Pre-Registered Decision Rules (`R4-E2`)
- **`E2-D0` ($9\times 9$ Law & Hardness Gate):** 1-step fully-constrained cell KL $\le 0.01$ nats and false-witness rate $\le 2\%$; 1-pass `P0` unsolvable log-loss $\ge 1.20$ nats on `s9_hard_test`; and `P6-Landmark` triggers branching on **$\ge 80\%$ of test boards** (fixing `R3-D2b`'s $0.0\%$ branch share).
- **`E2-D1` (Landmark Tree Search vs. Flat Rollouts on Solvable $9\times 9$ Boards):** At equal total forward-pass budget ($B = 16$ full-board equivalents), `P6-Landmark` beats flat `P3-Conf` (Kim et al. baseline) and `P1-Rand` by **$\ge 20\%$** in cell marginal KL on ambiguous solvable boards and increases valid witness discovery rate on hard solvable boards by **$\ge 15$ percentage points** (3-seed CI $>0$).
- **`E2-D2` (Subspace Refuted Mass $C_M^{\text{sub}}$ vs. Chain Count $M$ in Log-Loss):** At $M = 2$ and $M = 3$ branched rollouts on `s9_hard_test` (where different branches have heterogeneous prior masses $p_{d_1} \neq p_{d_2}$), `null_subspace` using $C_M^{\text{sub}}$ achieves (a) $\mathrm{AUC}(-C_M^{\text{sub}}) \ge 0.85$ and (b) **beats `count_floor` in `Solvable` log-loss by $\ge 15\%$** with fitted $\theta < 10$ across all 3 seeds.

---

## 3. Rank 3 — `R4-E3` (`E-LIR-RL`): Honest Beliefs, Self-Trust & Faith-Swap Under Rising-Reward RL

### 3.1 Question It Answers
In `lir-agent` Round 1 (`Qwen3-0.6B`), LIR parameter masks eliminated false-note sycophancy ($59.3\% \to 0.0\%$), **but task reward did not rise during RL, the 0.6B forecaster collapsed to a constant ($0.6501$), and the Faith-Swap test was never run**. When we use a warm-started, genuinely informative world model/forecaster ($r > 0.25$ with Bayes) on $7\times 7$ ShipHunt and real **Manifold Markets / MiniBench** questions at `1.7B–8B` scale:
1. Does LIR parameter masking ($\chi_{\text{real}} = \chi_{\text{trust}} = 0, \; \chi_{\text{amp}} = \chi_{\text{action}} = 1$) allow **task reward to rise significantly ($\ge +15\%$)** while keeping belief/forecast drift at **$0.000$** (whereas Fused RL exhibits the predicted desire-distortion bias $\langle \mu - P^*, \nabla U \rangle > 0$)?
2. Does swapping the explicit **Faith / Utility module $U$** at test time (`faith_swap`) immediately reverse or redirect the agent's actions while leaving its world beliefs completely unchanged?

### 3.2 Setup, Models & Code-Level Spec
- **Codebases:**
  - **Part A (`R4-E3a`, $7\times 7$ Multi-Ship ShipHunt POMDP with Sycophantic Commander Notes):** Extend `/home/agent/workspaces/eurisko-study/lir-agent/harness/` (`Qwen3-1.7B` with 4 LoRA role adapters: `real_belief`, `amp_rollout`, `trust_auditor`, `action_policy`, 1 job per L4 GPU per Lesson 2 of `wiki-lir-agent.md`).
  - **Part B (`R4-E3b`, Manifold / MiniBench Forecasting + Betting Allocation with Optimistic Rater Pressure):** Extend `/home/agent/workspaces/calibrated-ai/minibench/` using `Qwen3-8B` LoRA via Tinker (or `Qwen3-1.7B` on L4) on 2,500 resolved Manifold markets with pre-close probability histories and news context.
- **Four Code Fixes Addressing Every `lir-agent` Round-1 Issue (`wiki-lir-agent.md` §"Open decisions"):**
  1. **Pre-RL Forecaster Gate:** Supervised warm-start of `real_belief` on realized outcomes until held-out correlation with exact Bayes ($7\times 7$ ShipHunt) or Brier Skill Score (Manifold) is **$\ge 0.30$** (aborting if $< 0.20$).
  2. **Fix `E2` Alpha Dose Bug (`wiki-lir-agent.md` line 43):** In Round 1, scaling the reward loss by $\alpha$ before per-adapter AdamW normalized away $\alpha$! Fix `train.py` to combine `loss = loss_real + alpha * loss_reward` inside a single optimizer step on the unmasked parameters (for `Fused` and `Token-Masked`), while `LIR-Masked` structurally zeros gradients into `real_belief` and `trust_auditor`.
  3. **Rising-Reward Environment:** Start `action_policy` from a uniform/untrained LoRA adapter so RL has a large, clean reward headroom ($+25\text{--}40\%$ expected return gain over 400 GRPO steps) alongside a sycophantic/optimistic bonus $+\lambda_{\text{syc}} \mathbb{I}[\text{agree with biased hint}]$.
  4. **Implement & Run `faith_swap` + `Theory T2` Projected Error Audit:**
     - Evaluate every checkpoint under two utility specifications passed to the action/search module: $U_+$ (*Seek High-Payoff Target / Bet YES on High-EV*) and $U_-$ (*Avoid Target / Hedge Downside*).
     - Compute the empirical **Desire-Projection Bias** $\Delta_{\text{desire}} = \mathbb{E}\big[\langle \mu_{\text{belief}}(X) - P^*(X), \, \nabla_X U(X) \rangle\big]$.

### 3.3 Pre-Registered Decision Rules (`R4-E3`)
- **`E3-D0` (Informative Forecaster & Rising-Reward Gate):** Pre-RL `real_belief` achieves correlation $\ge 0.30$ with exact Bayes (`E3a`) and Brier skill $> 0.05$ over base rate (`E3b`); during RL, task reward rises by **$\ge 15\%$** in both `Fused` and `LIR-Masked` arms.
- **`E3-D1` (Zero Belief Drift & Zero Desire-Projection Bias Under Rising Reward):**
  - `LIR-Masked` maintains **$0.000$ belief drift** and false-note sycophancy $\le 2\%$ (vs. $\ge 35\%$ on `Fused` and $\ge 8\%$ leak on `Token-Masked`), while achieving **$\ge 95\%$ of `Fused`'s true environment reward**.
  - On `Fused`, the Desire-Projection Bias $\Delta_{\text{desire}}$ is strictly positive ($95\%\text{ CI} > 0$, confirming Richardson's $\frac{\lambda}{2}\|\nabla U\|^2$ distortion on real LLMs), whereas on `LIR-Masked` $\Delta_{\text{desire}} = 0.000$.
- **`E3-D2` (Zero-Retraining `faith_swap`):** Swapping $U_+ \to U_-$ at test time flips the agent's action selection in the utility-preferred direction on **$\ge 80\%$ of test episodes** while changing `real_belief` probabilities by **$0.000$**.

---

## 4. Execution Plan & Handoff Table

| Executor | Assigned Work | Where to Write Pre-Reg & Results | Budget Cap |
| :--- | :--- | :--- | :---: |
| **`eurisko`** (or `opus-mats` GPU subagent) | **`R4-E1` (`E-MS` Micro-Stratego)** — build `model_ms.py`, `train_ms.py`, `eval_ms.py` on top of `/home/agent/workspaces/mats/work/micro-stratego/env.py`; run 3 seeds on L4 via `eca/gpu/gpu-run-zones.sh`. | `/home/agent/workspaces/mats/work/micro-stratego/PREREG-MS.md` & `RESULTS-MS.md` | **$\$15$** |
| **`calibrated-ai`** | **`R4-E2` (`E-SUD-9x9`)** — build `s9_hard` dataset, `P6-Landmark` tree branching, and `C_M^sub` subspace refuted mass in `/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/E-SUD/`; run 3 seeds on L4. | `/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/E-SUD/PREREG-R4.md` & `RESULTS-R4.md` | **$\$20$** |
| **`eurisko` + `calibrated-ai`** | **`R4-E3` (`E-LIR-RL`)** — run `R4-E3a` (`Qwen3-1.7B` ShipHunt $7\times 7$ with rising reward + `faith_swap` + $\Delta_{\text{desire}}$) in `lir-agent/` and `R4-E3b` (Manifold forecasting + allocation) in `minibench/`. | `/home/agent/workspaces/eurisko-study/lir-agent/RESULTS-R2.md` | **$\$30$** |
