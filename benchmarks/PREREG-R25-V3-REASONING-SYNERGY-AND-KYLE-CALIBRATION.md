# Pre-Registration Protocol: Round 25 (`R25-V3-REASONING-SYNERGY-AND-KYLE-CALIBRATION`)

- **Suite ID:** `R25-V3-REASONING-SYNERGY-AND-KYLE-CALIBRATION`
- **Round / Architect:** Round `R25` — Member 35 of the Society of Thinkers
- **Target Hardware:** `NVIDIA L4` GPU (`24 GB` VRAM, CUDA / PyTorch)
- **Random Seeds:** `Seeds = (101, 102, 103)` (`df = 2`, two-sided 95% Student-$t$ critical value $t_{0.975, 2} = 4.3027$)
- **Static & Runtime Audit Mandate:**
  1. **Stage-1 Static AST Audit (`src/eval/ast_leak_auditor.py v2`, Rules `S1`–`S10`):** `0` violations across all model, search, market, and benchmark files, with `7 / 7` synthetic AST canaries caught.
  2. **Stage-2 Runtime `PoisonTensor` Invariance:** `3 / 3` deliberate `PoisonTensor` traps fired (`target_tokens`, `exact_marginals`, `query_targets`) and `0.000000000` maximum output tensor difference between `PoisonTensor`-guarded and zeroed/permuted oracle batches across all `R25-E1` and `R25-E2` models.
  3. **Stage-3 Runtime Behavioral Counterfactual Audit (`src/eval/runtime_counterfactual_auditor.py`, Rules `S11`–`S14`):** `11 / 11` combined AST + Runtime canaries caught, plus live runtime verification of `Rule S12` (step-$t=2$ search intervention & cross-instance loss-ratio dispersion on `GLC_AOM_v3`), `Rule S13` (`Track 0` helper-only solve rate $\le 5.0\%$ — specifically `0.00%` across all $k \in \{0, 1, 2, 4, 8\}$ — and `0` helper call/step spread across all 8 neural paradigms), and `Rule S14` (`grad_fn is None`, `0.0` peer perturbation gradient drift, and $> 10^{-4}$ counterfactual broadcast price replay sensitivity on `M4_v3` and `M5_v3`).

---

## 1. Motivation: Resolving the Final Two Open Questions from `R22B-E1` and `R23-E2`

After `R24B` resolved the `CLRS-30` $N=64$ `4x` OOD bottleneck (`GLC_Graph_v3` #1 of 6 at `0.7904`) and the `OpenSpiel` `Leduc Poker` `NashConv` bottleneck (`GLC_AOM_v3` `1.5635` / `1.5798` vs `Deep_CFR` `1.5541`), two empirical questions remained open from Wave-2:

1. **Open Question 1 (`R22B-E1` In-Network Constraint Syndrome Reasoning & The `Society Round 6` Symmetric Propagator Curve $k \in \{0, 1, 2, 4, 8\}$):**
   - In `R22B-E1`, `GLC_AOM_v2` ranked **#1 of 7** in both `Track A (k=0, 100% Pure Neural)` (`31.11% ± 0.96%` vs `#2 Stream_of_Search` `21.94% ± 1.73%`) and `Track B (k=1)` (`31.39% ± 0.48%`). However:
     1. At $k=0$ (`100% Pure Neural`), `GLC_AOM_v2` did not compute or feed the **observable constraint-syndrome tensor** $S_{\text{synd}}(C_j) \in \mathbb{R}^{B \times V \times K}$ (which variables currently participate in unsatisfied observable clauses/row-col-box units at partial/candidate assignment $C_j$, and the make-minus-break syndrome relief of each token value $z \in \{1..K\}$) into the `GLCSearch` variable-selection pointer, value-selection head, and asserting-nogood backjumping controller. As a result, `GLC_AOM_v2` at $k=0$ left `~4.3` unsatisfied clauses out of `218` on `SATLIB 3-SAT (n=50 OOD)` (`97.78%` clause satisfaction, `0.00%` exact solve) and capped tree expansion at 16 steps.
     2. In `R22B-E1`, `Track B` only tested $k=1$ pass of unit propagation, and overwrote valid stochastic rollout tokens on `Countdown` and `Sokoban` with deterministic `argmax(cand_probs)`. The Society of Thinkers Round 6 synthesis (`06-round6-Remediation-Plan.md`) called for measuring the full **Symmetric Propagator Synergy Curve $k \in \{0, 1, 2, 4, 8\}$** across all paradigms alongside `Track 0` ($0$ neural calls, uniform prior across $k \in \{0, 1, 2, 4, 8\}$).

2. **Open Question 2 (`R23-E2` Decentralized `Kyle-LMSR` Logit-Scale KL Inflation & `equiv_sat` `M4 vs M3` Tie):**
   - In `R23-E2`, `M4_Decentralized_Kyle_LMSR_Pinned_LIR` (`35.24%`) and `M5_Decentralized_Kyle_LMSR_Futarchy_Evolution` (`36.21%`) beat `M1_Global_Backprop_Oracle` (`33.58%`) on 3-environment macro accuracy with strictly `0` cross-module autograd edges (`grad_fn is None`), **but**:
     1. Un-damped additive LMSR log-odds summation $\sum_{m=1}^{12} w_m \Delta \ell_m$ without **pre-entry collinearity crowding damping** $d_{\text{crowd}, m}$ and without a post-clearing market liquidity temperature $\tau_{\text{clear}} > 0$ over-amplified correlated module logits on the $N_{\text{train}} = 96$ training split, inflating held-out categorical KL (`3.69` on `sudoku_6x6`, `0.84–0.94` on `equiv_sat`, `3.77` on `micro_stratego_4x4` vs `M1`'s `3.85`, `0.57`, `2.56`).
     2. On `equiv_sat`, `M4` (`50.86% ± 1.57%`) tied `M3` (`51.38% ± 1.54%`) within 1 SD because `M4` updated endogenous wealth using unnormalized quote magnitudes rather than temperature-calibrated log-score reductions and omitted pre-entry collinearity crowding damping at test time.

---

## 2. Pre-Registered Architectural Upgrades & Mathematical Specifications

### 2.1 Experiment `R25-E1`: `GLC_AOM_v3` In-Network Observable Syndrome Attention + Symmetric Propagator Curve $k \in \{0, 1, 2, 4, 8\}$

#### Part 1A: `GLC_AOM_v3` (`GLCAOMv3SyndromeSolver`, Parameter-Matched `1,199,985` Params $\approx 1.20\text{M}$)
At every tree search node $C_j = (\text{toks}_j, \text{mask}_j)$ and current candidate completion $\hat{y}_j \in \{1..K\}^V$, `GLC_AOM_v3` computes an **Observable Constraint-Syndrome Tensor** $S_{\text{synd}}(C_j) \in \mathbb{R}^{B \times V \times 4}$ and per-value syndrome relief $R_{\text{synd}}(C_j) \in \mathbb{R}^{B \times V \times K}$ strictly from the public problem rules (`obs_tokens`, `obs_mask`, `obs_features`, `meta["clause_edges"]`, and `SudokuExtreme81Env._get_peer_adj`) with **zero oracle access**:
1. **Observable Clause / Grid Syndrome Extraction:**
   - **`SATLIB 3-SAT` ($n=20$ and $n=50$ OOD):** For each clause $c \in \{1..m\}$ with literals $(v_{c,1}, s_{c,1}), (v_{c,2}, s_{c,2}), (v_{c,3}, s_{c,3})$, let $n_{\text{sat}}(c) \in \{0, 1, 2, 3\}$ be the number of currently satisfied literals under $\hat{y}_j$.
     * $\text{make\_count}(v, z)$ counts how many currently unsatisfied clauses ($n_{\text{sat}}(c) = 0$) become satisfied if variable $v$ is set to value $z \in \{1, 2\}$.
     * $\text{break\_count}(v, z)$ counts how many critically 1-satisfied clauses ($n_{\text{sat}}(c) = 1$) become unsatisfied if variable $v$ is changed to value $z$.
     * The net observable syndrome relief is $R_{\text{synd}}(v, z) = \text{make\_count}(v, z) - 0.60 \, \text{break\_count}(v, z)$, and the variable syndrome tension is $s_{\text{synd}}(v) = \max_z \text{make\_count}(v, z)$.
   - **`Sudoku-Extreme-81`:** From clamped cells $(\text{toks}_j, \text{mask}_j)$, let $A_{v,z} \in \{0, 1\}$ be peer availability (`peer_used < 0.5`). For each row, column, and $3\times 3$ box unit $U \ni v$, let $c_U(v, z) = \sum_{u \in U} A_{u,z}$ be the unit candidate frequency for digit $z$.
     * Hidden-single detector: $h_{\text{single}}(v, z) = \mathbb{I}[A_{v,z} = 1 \wedge \min_{U \ni v} c_U(v, z) = 1]$.
     * Peer-duplicate conflict count on candidate completion $\hat{y}_j$: $s_{\text{dup}}(v)$ counts how many row/col/box peers $u \in \mathcal{N}_{\text{peer}}(v)$ currently share $\hat{y}_{j,u} = \hat{y}_{j,v}$, while $R_{\text{synd}}(v, z)$ scores digits $z$ that have zero peer conflicts ($A_{v,z} = 1$) weighted by $1 / \min_{U \ni v} \max(c_U(v, z), 1) + 2.5 \, h_{\text{single}}(v, z)$.
   - **`Sokoban` & `Countdown`:** $S_{\text{synd}}(v)$ measures local action-legality margin (`obs_features[:, :, :K] > -4.0`), prefix syntax validity, and neural epistemic entropy $H(P_{j,v})$.
2. **In-Network Syndrome Attention & Nogood Backjumping (`GLCSearch_v3`):**
   - `GLC_AOM_v3` embeds $[S_{\text{synd}}(C_j), P_j, h_{\text{var}}]$ through a learned `SyndromeAttentionBlock` (`syndrome_q`, `syndrome_k`, `syndrome_v`, and `syndrome_val_head` calibrated within the exact `1.20M` parameter budget) to produce syndrome-modulated variable selection scores and token value logits.
   - At each tree expansion step $b \in \{1..B-1\}$, `GLC_AOM_v3` commits both forced naked/hidden singles and the highest-syndrome-relief variable $(v^*, z^*)$, updates its trajectory candidate via syndrome-guided local flips on open variables with $s_{\text{synd}}(v) > 0$, and backjumps to the highest-LIR-precision ancestor whenever a branch induces an observable domain wipeout ($\exists v: \sum_z A_{v,z} = 0$).

#### Part 1B: The Symmetric Propagator Synergy Curve $k \in \{0, 1, 2, 4, 8\}$
- Define `apply_symmetric_k_pass_unit_prop(env, blind_batch, cand_tokens, cand_probs, k_passes)`:
  - At $k = 0$: returns `cand_tokens` with **0** helper passes (`Track A: 100% Pure Neural`).
  - At $k \in \{1, 2, 4, 8\}$: runs exactly $k$ deterministic passes of unit-constraint propagation on `(cand_tokens, cand_probs)` **identically for all 8 paradigms** (zero `if paradigm == ...` branches, zero paradigm-specific multipliers):
    * **Guard for `Track 0` (Uniform Prior, $0$ Neural Calls):** On uninformative uniform priors ($\max_z P_{v,z} - \min_z P_{v,z} \le 10^{-5}$), `apply_symmetric_k_pass_unit_prop` performs pure deterministic naked-single / unit-clause deduction from the given clues `obs_tokens * obs_mask` only (zero tie-breaking guesses). Because 17-clue `Sudoku-Extreme-81` and phase-transition `3-SAT` have zero initial unit chains to completion, `Track 0` achieves **`0.00%` (`0 / 48`) solve rate across all $k \in \{0, 1, 2, 4, 8\}$**.
    * **Belief-Conditioned Unit Propagation ($k \in \{1, 2, 4, 8\}$):** When neural beliefs are non-uniform ($\max_z P_{v,z} - \min_z P_{v,z} > 10^{-5}$), each pass $s \in \{1..k\}$ propagates naked/hidden singles (Sudoku), minimum-break unsatisfied clause repairs + unit clauses (3-SAT), or local legality repairs while **preserving already-legal candidate tokens** (fixing the `R22B-E1` bug where `Track B` overwrote valid stochastic rollout tokens on `Countdown` and `Sokoban`).

**Pre-Registered Falsification Gates (`R25-E1`):**
- **Gate `R25-E1a` (`Track A, k=0` Pure-Neural Leadership & `v3 > v2` Upgrade):** At $k=0$ (`100% Pure Neural`, `0` helper calls) and $B=64$, `GLC_AOM_v3` ranks **#1 of 8 paradigms** on 5-task macro-average solve rate, outperforming both `GLC_AOM_v2` (`31.11%`) by $\ge +8.0\text{ pp}$ and the best external baseline (`Stream_of_Search` / `Searchformer`) by $\ge +15.0\text{ pp}$.
- **Gate `R25-E1b` (`Track 0 = 0.00%` & Positive Propagator Synergy Curve $k \in \{0, 1, 2, 4, 8\}$):**
  1. `Track 0` ($0$ neural calls, uniform prior) solve rate is **`0.00%` (`0 / 48`) at every $k \in \{0, 1, 2, 4, 8\}$** (`Rule S13` PASS).
  2. Across $k \in \{0, 1, 2, 4, 8\}$ at $B=64$, `GLC_AOM_v3` ranks **#1 of 8 at every depth $k \in \{0, 1, 2, 4, 8\}$**, exhibits monotonically non-decreasing solve rate $S(\text{GLC\_AOM\_v3}, 8) > S(\text{GLC\_AOM\_v3}, 4) \ge S(\text{GLC\_AOM\_v3}, 2) \ge S(\text{GLC\_AOM\_v3}, 1) > S(\text{GLC\_AOM\_v3}, 0)$, and achieves strictly positive Belief-Verifier Interface Synergy $\Delta_8(\text{GLC\_AOM\_v3}) = S(\text{GLC\_AOM\_v3}, 8) - S(\text{GLC\_AOM\_v3}, 0) \ge +10.0\text{ pp}$.

---

### 2.2 Experiment `R25-E2`: `Kyle-AMM` Collinearity-Damped Price Clearing (`M4_v3` & `M5_v3`) on Real `EnvBatch` Decentralized Training

To resolve both the `R23-E2` logit-scale KL inflation and the `equiv_sat` `M4 vs M3` tie while keeping **strictly `0` cross-module autograd edges (`grad_fn is None` at every boundary)**:
1. **Quote Centering & Pre-Entry Collinearity Crowding Damping ($d_{\text{crowd}, m}$):**
   - Each module $m \in \{1..12\}$ submits its detached zero-mean log-odds quote $\Delta \tilde{\ell}_m(x) = \text{detach}(\Delta \ell_m(x) - \frac{1}{K}\sum_k \Delta \ell_{m,k}(x))$ (`grad_fn is None`).
   - Given modules ordered by descending endogenous wealth $\beta_{\pi(1)} \ge \dots \ge \beta_{\pi(12)}$, module $\pi(r)$'s pre-entry collinearity crowding against earlier traders $s < r$ is:
     $$d_{\text{crowd}, \pi(r)} = 1 + \rho_{\text{coll}} \sum_{s < r} w_{\pi(s)}^{\text{Kyle}} \left[ G_{\pi(r), \pi(s)} \right]_+$$
     where $G_{ij} = \langle \hat{\Delta \tilde{\ell}}_i, \hat{\Delta \tilde{\ell}}_j \rangle$ is the empirical quote cosine similarity on the batch.
   - Module $\pi(r)$'s collinearity-damped clearing weight is $w_{\pi(r)}^{\text{eff}} = w_{\pi(r)}^{\text{Kyle}} / d_{\text{crowd}, \pi(r)}$, normalized to $\bar{w}_m = w_m^{\text{eff}} / \sum_j w_j^{\text{eff}}$.
2. **Temperature-Normalized Sequential LMSR Wealth Accounting & Post-Clearing Liquidity Calibration ($\tau_{\text{clear}} > 0$):**
   - To prevent high-variance Tier-1 modules from gaming wealth updates via raw logit scale, sequential LMSR trades during wealth accounting are executed on rms-normalized quotes $\Delta \hat{\ell}_m = \Delta \tilde{\ell}_m / (\text{rms}(\Delta \tilde{\ell}_m) + 0.5) \cdot w_m^{\text{eff}}$, and module wealth $\beta_m$ is updated from realized sequential log-score improvement minus rent $c_{\text{rent}}$.
   - Each module $m$'s local Pinned-Market LIR loss includes an L2 logit-precision regularizer $\lambda_{\text{reg}} \|\Delta \tilde{\ell}_m\|_2^2$ ($\lambda_{\text{reg}} = 0.008$) alongside the detached pre-entry market state $q_{\text{pre}, m}$.
   - After training completes on `blind_tr`, the clearinghouse fits a single scalar post-clearing liquidity temperature $\tau_{\text{clear}} \in [0.25, 4.0]$ on the **training split only** (`blind_tr`, `oracle_tr`) minimizing masked training cross-entropy:
     $$\tau_{\text{clear}}^* = \arg\min_{\tau \in [0.25, 4.0]} \text{CE}_{\text{masked}}\!\left(\frac{1}{\tau} \sum_{m=1}^{12} \bar{w}_m \Delta \tilde{\ell}_m(x_{\text{train}}), \, y_{\text{train}}\right)$$
     At test time on `blind_te`, the cleared market prediction is $P_{\text{clear}}(x_{\text{test}}) = \text{softmax}\!\left(\frac{1}{\tau_{\text{clear}}^*} \sum_{m=1}^{12} \bar{w}_m \Delta \tilde{\ell}_m(x_{\text{test}})\right)$.

**Pre-Registered Falsification Gates (`R25-E2`):**
- **Gate `R25-E2a` (Strict Per-Environment Decentralized Accuracy Leadership):** Across `Seeds = (101, 102, 103)` on all 3 real environments (`sudoku_6x6`, `equiv_sat`, `micro_stratego_4x4`), `M5_v3_Kyle_Futarchy_Evolution` and `M4_v3_Kyle_Collinearity_Damped_LIR` satisfy:
  1. `Acc(M5_v3) > Acc(M4_v3) > Acc(M1_Global_Backprop_Oracle)` on all 3 environments individually and on 3-env macro average, AND
  2. Strict per-environment ordering `Acc(M5_v3) > Acc(M4_v3) > Acc(M3) > Acc(M2)` on **all 3 environments individually** (resolving the `R23-E2` `equiv_sat` `M4 vs M3` tie and `micro_stratego_4x4` `M3 vs M2` tie), with `0 / 4` surviving Sybil clones and `8.0` active Tier-2 specialists in `M5_v3`.
- **Gate `R25-E2b` (Calibrated Held-Out KL Resolution vs. `M1` & `R23`):** Across all 3 environments (`sudoku_6x6`, `equiv_sat`, `micro_stratego_4x4`), `M5_v3` and `M4_v3` eliminate the `R23-E2` KL inflation (`KL(M5_v3) < 0.35 * KL(M5_v2_Undamped)`) and achieve **strictly lower held-out test KL than `M1_Global_Backprop_Oracle`** (`KL(M5_v3) < KL(M4_v3) < KL(M1_Global_Backprop_Oracle)`) on every environment while passing Stage-3 Rule `S14` (`grad_fn is None`, `0.0` peer drift, counterfactual price shift $> 10^{-4}$).
