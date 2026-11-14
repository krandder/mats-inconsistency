# Pre-Registered Experimental Specification: 4×4 Diagonal-Touch Micro-Stratego (`E-MS`)

**Author:** Gemini (Argon) for Kelvin Azevedo Santos (`opus-mats` / MATS Autumn 2026)
**Date:** 2026-10-16
**Code & Oracle Prototype:** `/home/agent/workspaces/mats/work/micro-stratego/env.py`

---

## 1. Purpose & What This Suite Tests

In `research-direction-synthesis-2026-10-16.md` (§B, line 40), six core claims of the *Inconsistency Is All You Need* research direction are marked **Untested**:
1. **Any-order belief unmasking vs. fixed row-major order (Ataraxos)** on coupled hidden-information games.
2. **Policy-independent consensus via multi-layer attention deduplication** when the search policy is deliberately skewed toward one flank/branch.
3. **Self-trustworthiness ($\theta_{\text{trust}}$) $\neq$ output entropy (`stiff`)** when rollouts face both genuine 50/50 hidden-information risk (high entropy, high trust) and out-of-distribution world-model errors (low entropy, low trust).
4. **LIR parameter separation ($\chi$)** between the reality-grounded world model ($\theta_{\text{real}}$) and the consensus-distilled consequence extension ($\theta_{\text{amp}}$), plus zero-retraining **modular utility swap ($U$)**.
5. **MCTS & Ataraxos inclusion + cross-branch key sharing** via Consensus Landmark Memory and Key-Space PUCT over a **learned world model**.

`E-MS` tests all five in a single fast, exact-oracle environment: **4×4 Diagonal-Touch Micro-Stratego**, costing **$<\$3$ total GPU spend** ($\sim 30\text{--}45$ minutes on 1–2 L4 GPUs, 3 seeds).

---

## 2. Environment & Exact Bayesian Oracle (`/home/agent/workspaces/mats/work/micro-stratego/env.py`)

### 2.1 Board Geometry (`xxxx / xx.. / ..oo / oooo`)
- **Grid:** $4 \times 4$ (`rows 0..3, cols 0..3`).
- **Starting Deployment Slots (6 pieces per player, touching only diagonally at `(1,1)-(2,2)`):**
  - **Red (Player 0):** Front slots `0:(1,0), 1:(1,1)` (`xx..`); Back slots `2:(0,0), 3:(0,1), 4:(0,2), 5:(0,3)` (`xxxx`).
  - **Blue (Player 1):** Front slots `0:(2,2), 1:(2,3)` (`..oo`); Back slots `2:(3,0), 3:(3,1), 4:(3,2), 5:(3,3)` (`oooo`).
- **Natural Two-Flank Structure:**
  - **Left Flank (`cols 0..1`):** Open corridor `(2,0), (2,1)` directly in front of Red's front line toward Blue's left back rank `(3,0), (3,1)`.
  - **Right Flank (`cols 2..3`):** Open corridor `(1,2), (1,3)` directly in front of Blue's front line toward Red's right back rank `(0,2), (0,3)`.
- **Piece Set (6 distinct pieces per side $\implies 6! = 720$ initial setups):**
  1. `Flag (F, rank 0)`: Immobile; capture wins immediately (`+1`).
  2. `Bomb (B)`: Immobile; defeats any attacker *except* `Miner (M)` (attacker is removed, Bomb stays & is revealed).
  3. `Scout (S, rank 2)`: Moves 1–3 orthogonal squares through empty squares (moving $>1$ square publicly reveals it is a `Scout`).
  4. `Miner (M, rank 3)`: Moves 1 orthogonal square; defuses/captures `Bomb (B)`.
  5. `General (G, rank 4)`: Moves 1 orthogonal square; beats `Miner` and `Scout`.
  6. `Marshal (X, rank 5)`: Moves 1 orthogonal square; beats `General`, `Miner`, `Scout`, `Flag`; loses only to `Bomb`.
  *(Equal rank in battle removes both pieces. Horizon $T_{\max} = 24$ plies.)*

### 2.2 Empirical Sanity Check (`env.py`, 500 random games)
- **Win balance:** Red `35.0%`, Blue `33.6%`, Draw `31.4%`; mean game length `16.1` plies.
- **Branching factor:** Mean `4.96` legal moves/turn (max `14`).
- **Exact Bayesian Posterior Oracle $P^*(Z_{\text{opp}} \mid h_t)$:** By turn 8, public movement and battle constraints narrow the opponent's consistent setups from `720` down to a mean of **`38.1` consistent permutations**, inducing strong non-local coupling across the 6 slots (e.g., if slots `0, 1` have moved, slots `2..5` have elevated `Flag`/`Bomb` probability).

---

## 3. Architecture: The 4 LIR-Separated Parameter Blocks

All arms share one compact Any-Order Transformer backbone ($d_{\text{model}} = 128$, $4$ layers, $4$ heads, $\sim 850\text{k}$ params) operating over thesingle-world variable tape:
- **Public & Private Board State Tokens ($O_t$):** Current positions, alive flags, public movement history (`has_moved`, `moved_long`, `revealed`), and acting player's own 6 piece identities.
- **Hidden Opponent Slot Variables ($Z_0, \dots, Z_5 \in \{1\dots 6\}$):** Materialized in any permutation $\sigma$ chosen by the search policy.
- **Future Rollout Steps ($A_\tau, O_{\tau+1}$ for $\tau = t \dots t+d-1$):** Candidate move $A_\tau$ and predicted battle/board transition $O_{\tau+1}$ (winner of battle, revealed piece, terminal win/loss).

The parameters are partitioned into four blocks under strict LIR control masks $\chi$:

1. **$\theta_{\text{real}}$ (Base Reality-Grounded World Model; $\chi_{\text{real}}=1, \chi_{\text{search}}=0$):**
   - Trunk + any-order conditional heads predicting (a) hidden opponent piece $q_{\text{real}}(Z_s \mid O_t, Z_{<\sigma})$ and (b) 1-step battle/board transition $q_{\text{real}}(O_{\tau+1} \mid O_\tau, Z, A_\tau)$.
   - Trained **exclusively** via standard cross-entropy log-loss on realized self-play game states and transitions (zero train/test setup overlap: 540 train setups / 90 val / 90 test setups).
2. **$\theta_{\text{amp}}$ (Consequence Extension; $\chi_{\text{real}}=0, \chi_{\text{search}}=1$):**
   - 2-layer MLP/LoRA adapter reading **`stop_grad` features** from $\theta_{\text{real}}$.
   - Outputs:
     - 0-step baseline logits $\ell^{(0)}(S_j)$ over opponent beliefs ($6\times 6$) and root action values $Q(a)$, and
     - Per-rollout summary token $(k_r \in \mathbb{R}^{16}, \; v_r = \Delta \ell_r)$ at the end of an $n$-step rollout.
   - Trained on the resolved $K$-rollout consensus $\mu^*_K$ and realized outcome log-loss.
3. **$\theta_{\text{trust}}$ (Self-Trustworthiness Auditor; $\chi_{\text{real}}=1, \chi_{\text{search}}=0$):**
   - 2-layer MLP reading `stop_grad` features of `(State S_j + Rollout Trace Summary h_r: rollout depth, mean transition entropy, order disagreement, and |Δℓ_r|)`.
   - Trained **exclusively on real-world outcomes** to predict the realized KL/Brier error $e_r$ of rollout $r$'s claim against ground truth, emitting $\beta_r = \exp(-\hat{e}_r)$.
4. **$\theta_{\text{agg}} + \pi_{\text{pol}}$ (Consensus Residual-Attention Aggregator & Key-Space PUCT Policy):**
   - **Aggregator ($\theta_{\text{agg}}$):** $L=3$ layer residual attention over the $k$ rollout summaries $\{(k_r, v_r, \beta_r)\}$ with LayerNorm/spectral clipping (preventing the $L=6$ polynomial blowup seen in `linear-agg`):
     $$\text{res}^{(l)}_r = v_r - \mathrm{Attn}_l(q=k_r, K=\{k_{r'}\}, V=\{h^{(l-1)}_{r'}\}), \qquad \ell^{(l)} = \ell^{(l-1)} + \eta_l \sum_{r=1}^k \beta_r W_{\text{out}} \, \text{res}^{(l)}_r.$$
   - **Search Policy ($\pi_{\text{pol}}$):** Scores candidate Landmarks $S_j$ and next variable/action $v$ via:
     $$\mathrm{Score}_{\text{pol}}(v) = I_{\text{goal}}(v) \cdot \beta_\theta(v) \cdot \Big(1 + \sum_{r'<r} \beta_{r'} \max(0, \langle k_v, k_{r'} \rangle)\Big)^{-1/2}.$$

---

## 4. Pre-Registered Hypotheses & Decision Bars (3 Seeds: `101, 102, 103`)

### Gate `D0` (World-Model Law Learning on Held-Out Setups)
- **Check:** On the 90 held-out test setups, $\theta_{\text{real}}$ achieves 1-step battle outcome accuracy $\ge 98\%$ and fully conditioned last-slot piece identity KL $\le 0.02$ nats, while 0-step unrolled belief KL on mid-game boards ($t \in [4, 12]$) has a non-trivial gap ($\ge 0.15$ nats vs. exact Bayesian oracle $P^*(Z \mid h_t)$).

### Claim `D1` (Any-Order Belief Unmasking vs. Ataraxos Row-Major Order)
- **Setup:** At mid-game states ($t \in [4, 12]$), materialize $K=8$ belief rollouts over the 6 opponent slots $Z_0 \dots Z_5$ using:
  1. `row_major` (Ataraxos fixed order: back-row slots `5,4,3,2` then front-row `1,0`),
  2. `random` permutation order,
  3. `stiff_voi` (our policy: unmask slots with highest trustworthy stiffness $\beta_\theta(Z_s)$ first).
- **Pass Bar (`D1`):** `stiff_voi` reduces posterior marginal KL to the exact 720-permutation Bayesian oracle $P^*(Z \mid h_t)$ by **$\ge 20\%$** relative to `row_major` (with 3-seed 95% CI strictly above 0) and cuts invalid-permutation sample rate by $\ge 30\%$.

### Claim `D2` (Policy-Independence via Attention Deduplication under Skewed Flank Policies)
- **Setup:** On 500 mid-game test positions with active pieces on both the Left Flank (`cols 0..1`) and Right Flank (`cols 2..3`), run $K=24$ rollouts under two deliberately skewed policies:
  - $\pi_{\text{Left}}$: samples Left-Flank moves/slots with probability $0.80$ and Right-Flank with $0.20$.
  - $\pi_{\text{Right}}$: samples Left-Flank with $0.20$ and Right-Flank with $0.80$.
- **Pass Bar (`D2`):** Measure the **Policy-Sensitivity Divergence** $\Delta_{\text{pol}} = \mathrm{JSD}\big(\mu^*_K(\pi_{\text{Left}}) \;\big\|\; \mu^*_K(\pi_{\text{Right}})\big)$ on the root action-value distribution:
  - Our $L=3$ Deduplicating Aggregator reduces $\Delta_{\text{pol}}$ by **$\ge 50\%$** compared to unweighted pooling (`mix`) and Ataraxos 1-step MMD (`L=1`), and achieves lower MSE/KL to the unbiased full-width oracle value than both `mix` and 1-step Gram crowding (`dd`).

### Claim `D3` (Self-Trustworthiness $\theta_{\text{trust}}$ vs. Output Entropy `stiff` on Imperfect World Models)
- **Setup:** Evaluate rollouts using an early/imperfect world-model checkpoint ($\theta_{\text{real}}$ at $20\%$ training steps, or with held-out tactical interactions where $\theta_{\text{real}}$ exhibits confident errors alongside genuine 50/50 unmoved-piece battles).
- **Pass Bar (`D3`):**
  1. $\theta_{\text{trust}}$ achieves significantly higher Spearman/AUC at predicting actual rollout log-error than output entropy $1 - H(q_\theta)$ (beating `stiff` by **$\ge 0.10$ Spearman/AUC**—breaking the E-CA tie where dynamics were deterministic!).
  2. Weighting rollouts by $\beta_r = \theta_{\text{trust}}$ reduces "delusional value hacking" (picking a move whose rollout predicted $>0.80$ win probability when true value is $<0.30$) by **$\ge 40\%$** relative to weighting by `stiff` or uniform weights.

### Claim `D4` (LIR Parameter Separation $\chi$ & Zero-Retraining Modular Utility Swap)
- **Setup:** Compare:
  - **LIR-Protected ($\chi_{\text{real}}=0$ during search/reward updates):** $\theta_{\text{real}}$ frozen to search/utility gradients; only $\theta_{\text{amp}}$ and $\pi_{\text{pol}}$ update.
  - **Fused ($\chi_{\text{real}}=1$):** Search consensus and utility reward gradients flow into $\theta_{\text{real}}$.
  - Also test **Utility Swap** at test time: swap $U_{\text{win}}$ (capture Flag) to $U_{\text{miner}}$ (+1 for capturing enemy Miner, -1 for losing own Miner) with zero retraining of $\theta_{\text{real}}$.
- **Pass Bar (`D4`):**
  1. Under biased/optimistic reward pressure, **Fused** degrades $\theta_{\text{real}}$'s belief/transition calibration (ECE/KL increases by $>25\%$), whereas **LIR-Protected** has **exact $0.000$ belief drift** while matching or beating Fused on task reward.
  2. Under test-time Utility Swap to $U_{\text{miner}}$, the LIR-Protected search immediately increases Miner-capture rate by **$\ge 2.5\times$** over $U_{\text{win}}$ without any retraining.

### Claim `D5` (Full Unified Search vs. 0-Step Policy, Ataraxos 1-Step MMD, and Tabular MCTS on a Learned World Model)
- **Setup:** Round-robin tournament (400 games per pair per seed) at $K=16$ rollouts/move, all searching over the **learned world model $\theta_{\text{real}}$** (no ground-truth simulator access during search):
  1. `P0` (0-step fast policy $\theta_{\text{amp}}$),
  2. `Ataraxos-1Step` (row-major belief sampling + flat root rollouts + 1-step MMD update),
  3. `Tabular-MCTS` (standard PUCT tree over learned world model, no key deduplication or $\theta_{\text{trust}}$),
  4. `Unified-LIR` (Any-Order `stiff_voi` beliefs + Consensus Landmark Memory + $L=3$ Key-Deduplicating Aggregator + Key-Space PUCT + $\theta_{\text{trust}}$).
- **Pass Bar (`D5`):** `Unified-LIR` achieves **$>55\%$ effective win rate** (wins $+ 0.5\times$ draws) against `P0`, `Ataraxos-1Step`, and `Tabular-MCTS` across the 3 seeds.
