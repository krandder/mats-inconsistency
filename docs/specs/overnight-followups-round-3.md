# Round-3 Follow-Up Experiments: Resolving the Setup Bugs to Test the Deep Claims (`2026-10-15`)

**Author:** `gemini` (Argon)  
**Target Coordinator:** `opus-mats` (for dispatch to `eurisko` on `E-CA-R3` and `calibrated-ai` on `E-SUD-R3`)  
**Estimated Total GPU Spend:** `~$6–$10` (well within remaining budget)

---

## 0. Diagnosis: Two Setup Bugs Masked the Deep Tests in Rounds 1 & 2

Direct inspection of `ecadata.py`, `boards.py`, `sud.py`, and the saved Round-2 evaluation arrays revealed that **both** domains suffered from a dataset-generation shortcut that allowed the 4-layer network's single forward pass (`P0`) to bypass multi-step reasoning on $>80\text{--}93\%$ of test instances:

1. **`E-CA` Bug — $2^8 = 256$ Attractor Memorization (`82%` Train/Test Grid Overlap):**
   - At $W = 8$, there are only $2^8 = 256$ possible row-0 bitstrings, each appearing $\sim 195$ times in the $50{,}000$ training grids ($81.5\text{--}82.1\%$ of test grids appeared verbatim in training).
   - Thus the 4-layer network solved `F` (Forward) in one pass (`KL 0.006` on Rule 90, `0.025` on Rule 110) by **memorizing the 256 noiseless attractors** rather than composing 5–7 steps of the local CA rule—leaving zero forward horizon for stiff-first search to help on.
   - Conversely, shrinking to a **1-layer** net in Round 2 (`R2a`) broke 1-step local rule learning (`S1` KL `0.037` on Rule 110, `0.077` on Rule 90) because a single 4-head attention layer over permuted tokens cannot cleanly route 3 parent values into one nonlinear truth table.
2. **`E-SUD` Bug — `93.7%` Depth-1 Contradictions, `77.9%` Zero-Dead-End Solvable Boards, and Tempered `logq`:**
   - `boards.py` only rejected corruptions with a Depth-0 direct peer clash (`peer_conflict`). Running a standard naked/hidden-single propagation check on `s6_test.npz` (`1,143` unsolvable boards) shows:
     - **Depth 1 (`1,071 / 1,143 = 93.7%`):** An empty cell already has 0 legal digits (or a unit has 0 places for a digit) directly from the givens. One-pass `P0` predicts `mean P0 = 0.0004` (`0.0%` false-positive rate).
     - **Depth 2 (`59 / 1,143 = 5.2%`):** Contradiction in 2 steps (`mean P0 = 0.162`).
     - **Depth $\ge 3$ (`13 / 1,143 = 1.1%`, including `4` Depth-999 branching-only boards):** One-pass `P0` is fooled (`mean P0 = 0.60` overall, `0.85` on Depth-999 where `100%` of boards have `P0 >= 0.5`, `P0` log-loss `1.51` nats), whereas search chains have `0%` false witnesses and refute `mean C4 = 0.84` of believed mass!
   - On the `857` solvable test boards, `77.9%` have **zero dead ends** under greedy MRV, so witness recall saturated at `99.9%`. Because `logit(P0)` already separated `98.9%` of boards with `AUC = 0.9997`, the chain-count control `count_floor` rode on top of `logit(P0)` for free.
   - Finally, in `sud.py` lines 176 and 202, setting `--tau 2.0` accumulated `logq` from the **tempered proposal** `softmax(lg / 2.0)` instead of the model's **untempered believed probability** `softmax(lg)`.

The two experiments below fix these setup bugs and directly test whether the deeper theoretical claims hold once memorization and Depth-1 shortcuts are removed.

---

## Rank 1: `E-SUD-R3` — Deep-Contradiction (`Depth >= 3`) & Non-Saturated-Recall (`20–60%`) Sudoku Evaluation on Saved 150k Checkpoints (`~$1.00`, ~20 min wall-time, 0 Retraining)

### Question Answered
1. On unsolvable boards where the contradiction requires **$\ge 3$ propagation steps or branching** (so the 4-layer network cannot spot a 1-step dead cell in one pass and `P0` fails with log-loss $> 1.0$ nat), does sequential search + recall-floored null-search still cut unsolvable-board log-loss by $\ge 60\%$ without hurting solvable boards?
2. Once `P0` is no longer `0.0004` on unsolvable boards, per-chain witness recall on solvable boards is in the non-saturated **`20–60%` regime**, and $C_M^{\text{bel}} = \sum_{z \in \text{refuted}} P_{\theta, \tau=1}(z \mid \text{board}, H_1)$ accumulates the model's **untempered believed posterior mass** (rather than the tempered proposal `softmax(lg / tau)`), does refuted believed mass $C_M^{\text{bel}}$ beat the chain-count control $M$?

### Exact Changes
1. **Zero retraining:** Reuse the 3 saved 150k-step checkpoints in `/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/E-SUD/r2models/model_s6_seed{0,1,2}.pt`.
2. **Generate a clean Hard Evaluation Split (`s6_hard_val.npz` with $1{,}000$ boards, `s6_hard_test.npz` with $1{,}000$ boards, $50\%$ solvable / $50\%$ unsolvable, matched clue counts `kgiv in [10, 15]`):**
   - **Unsolvable filter (`prop_depth(bb) >= 3`):** After corrupting 1–2 givens on a base grid and checking `not solve_all(bb, 6, 0)`, run naked-single + hidden-single constraint propagation (`prop_depth` below) and **accept only if `prop_depth(bb) >= 3`** (zero 1-step or 2-step contradictions).
   - **Solvable filter (`dead_end_rate(b) >= 0.25` / `20–60%` per-chain recall regime):** Accept solvable boards (`2 <= len(sols) <= 400`) where a 10-trial random MRV rollout hits a dead end at least $25\%$ of the time (or where `prop_depth(b) == 999`, i.e. naked/hidden singles stall before completion so branching is required and per-chain witness recall under exploratory sampling falls in `20–60%`).
   - Reference generator helper (verified at ~10 hard boards/sec on 1 CPU core, so 2,000 boards across 8 workers takes ~25 seconds):
     ```python
     PEERS = [set(j for j in range(36) if j != i and (j//6 == i//6 or j%6 == i%6 or box[j] == box[i])) for i in range(36)]
     UNITS = ([[r*6+c for c in range(6)] for r in range(6)] +
              [[r*6+c for r in range(6)] for c in range(6)] +
              [[i for i in range(36) if box[i] == b] for b in range(6)])

     def prop_depth(board):
         g = list(board)
         for step in range(1, 25):
             moves = []
             for i in range(36):
                 if g[i] != 0: continue
                 used = {g[j] for j in PEERS[i] if g[j] != 0}
                 if len(used) == 6: return step
                 if len(used) == 5: moves.append((i, (set(range(1, 7)) - used).pop()))
             for unit in UNITS:
                 for d in range(1, 7):
                     if any(g[i] == d for i in unit): continue
                     places = [i for i in unit if g[i] == 0 and all(g[j] != d for j in PEERS[i])]
                     if len(places) == 0: return step
                     if len(places) == 1: moves.append((places[0], d))
             if not moves: return 999
             for i, d in moves:
                 if g[i] != 0 and g[i] != d: return step
                 g[i] = d
             for i in range(36):
                 if g[i]:
                     for j in PEERS[i]:
                         if j > i and g[j] == g[i]: return step
         return 999
     ```
3. **Fix `sud.py` `chains()` to record Untempered Believed Log-Mass (`logp_bel`) Alongside Proposal Log-Mass (`logq_prop`):**
   - In `chains()`, when sampling at temperature `tau` (or under `rand` / `conf` / `mixed` where half of the $K=16$ chains are exploratory `rand` at $\tau=1.5\text{--}2.0$ and half are `conf` at $\tau=1.0\text{--}1.5$), compute:
     - `p_prop = F.softmax(lg / tau, -1)` (used to sample digit `d`), and
     - `p_bel  = F.softmax(lg, -1)` (the model's untempered conditional belief $q_{\theta, \tau=1}(z_t = d \mid S_{t-1}, H_1)$),
     and accumulate `logp_bel += log(p_bel[ar, z, d])`.
   - Define $C_M^{\text{bel}} = \min(1 - 10^{-9}, \sum_{z \in \text{distinct refuted}} \exp(\text{logp\_bel}(z)))$ as the fraction of the model's *believed* solution distribution refuted by the $M$ chains.

### Pre-Registered Decision Rules (`R3-D1a`, `R3-D1b`, `R3-D1c`)
- **`R3-D1a` (One-Pass Failure Gate on Deep Contradictions):** On `s6_hard_test.npz`, 3-seed mean one-pass `P0` log-loss on unsolvable boards is $\ge 0.50$ nats (confirming that removing Depth-1/2 contradictions eliminates the 1-pass shortcut).
- **`R3-D1b` (Null-Search Beats `P0` and `Witness-Only` on Deep Contradictions):** At $M = 8$ ($\tau = 1.0$), recall-floored null search (`null_floor` using $C_M^{\text{bel}}$, fitted on `s6_hard_val.npz`) cuts unsolvable-board log-loss by $\ge 50\%$ relative to both `P0` and `witness-only` (3-seed paired $t$-CI excluding 0) while also cutting solvable-board log-loss by $\ge 20\%$ relative to `P0` across `P1`, `P2`, and `P3`.
- **`R3-D1c` (Untempered Believed Mass $C_M^{\text{bel}}$ Beats Chain Count $M$):** On `s6_hard_test.npz` at $M \in \{1, 2, 4\}$ in the non-saturated recall regime ($\tau \in \{1.5, 2.0\}$ on `P1` and `P3`, plus a `mixed` bag where chains $1 \dots M$ alternate `P1` and `P3`):
  1. On unresolved (no-witness) boards at $M = 1$ and $M = 2$, $-C_M^{\text{bel}}$ achieves $\mathrm{AUC} \ge 0.80$ for separating solvable from unsolvable boards (vs. $0.50$ for chain count $M$ alone), and
  2. `null_floor` (using $C_M^{\text{bel}}$) achieves $\ge 10\%$ lower total `Solvable` log-loss than `count_floor` (using $M$) at $M = 2$ or $M = 4$ with 3-seed paired $t$-CI excluding 0.

---

## Rank 2: `E-CA-R3` — Zero-Overlap CA (`W = 14, T = 10` [or `W in {12, 14, 16}`], Disjoint Row-0 Splits) with 4-Layer Net, Anchored `sw_mix` Pool, and Hybrid `Propagate-Stiff-then-Branch` Search (`~$5–$8`, ~45–60 min wall-time)

### Question Answered
1. When row-0 memorization is mathematically impossible ($W = 14, T = 10$, $N = 140$ permuted cells—or $W \in \{12, 14, 16\}, T = 10$ per `eurisko`'s fast exact-inference prototype—with initial row-0 states partitioned **strictly disjointly** across train/cal/val/test so train/test row-0 overlap is **$0.0\%$**), does the **4-layer transformer** (or $L \in \{2, 3, 4\}$ if gated by `R3-D0`) learn the 1-step CA rule (`S1` KL $\le 0.015$) while leaving a genuine multi-step amortization gap (`F, B1, B2` KL $\ge 0.08$)?
2. On this bug-free regime (using Round 2's anchored `sw_mix` pool and $h_i$-free source-attention keys $k_{\text{v2}}$), how do **pure stiff-first (`P2-Stiff`)**, **variance-gated VoI / cutset branching (`P2-VoI-Unc`)**, **random order (`P4-Rand`)**, and a **hybrid `Propagate-Stiff-then-Branch` policy (`P6-Hybrid`)** compare on Forward (`F`) vs. Bridge (`B1, B2`) queries?
3. Once `P0` actually makes epistemic errors on multi-step queries, does a separate stiffness head (`Mechanism A` surprise head or a **1-step bootstrap/TD consistency head** $\hat{c}(i \mid S)$ predicting $|\mathrm{logit}\,q_\theta(X_i \mid S) - \mathbb{E}_{z}[\mathrm{logit}\,q_\theta(X_i \mid S \cup \{z\})]|$) beat plain predictive entropy $\ln 2 - H(q_\theta)$ at predicting where `P0` has low KL error?

### Exact Changes
1. **Grid & Disjoint Row-0 Split (`W = 14, T = 10` [or `W in {12, 14, 16}`], $\varepsilon = 0.015$, Rules 110 and 90, 4-Layer Transformer):**
   - Partition the $2^W$ possible row-0 bitstrings ($16{,}384$ at $W = 14$; $65{,}536$ at $W = 16$; $4{,}096$ at $W = 12$) via a fixed seeded hash/permutation into **strictly disjoint sets** across `train`, `cal`, `val`, and `test`.
   - **Result:** Train/test row-0 overlap is **$0.0\%$** by construction (eliminating the $82\%$ attractor-memorization shortcut of $W = 8$).
   - **Query families for $T = 10, W = 14$:**
     - `F` (Forward): Row 0 revealed ($W$ cells); query in rows 6–9 (steps 6–9, beyond the 4-layer one-pass horizon).
     - `B1` (Two-Row Bridge): Row 0 and row $T-1$ revealed ($2W$ cells); query in middle rows 4–5.
     - `B2` (Boundary-Anchor Bridge): Half of row 0 (cols $0 \dots W/2 - 1$), row $T-1$, and col 0–1 anchors revealed; query in unrevealed interior rows 4–5.
     - `S1` (One-Step Check): Target cell's 3 true past parents revealed.
2. **Fast Exact Posterior Marginals via Bit-Factored $O(W 2^W)$ Forward-Backward (`eurisko`'s verified prototype):**
   - Replace the dense $2^W \times 2^W$ matrix multiply in `ecadata.py` with the exact factored deterministic-scatter + hypercube bit-flip channel (verified exact to machine precision $10^{-15}$):
     ```python
     def next_det_state(rule: int, W: int) -> np.ndarray:
         B = bits(W)
         tab = rule_table(rule)
         f = tab[4 * np.roll(B, 1, axis=1) + 2 * B + np.roll(B, -1, axis=1)]
         return (f.astype(np.int32) << np.arange(W, dtype=np.int32)).sum(1)

     def noise_channel_fwht(p: np.ndarray, W: int, eps: float) -> np.ndarray:
         shape = p.shape
         x = p.reshape((-1,) + (2,) * W)
         a = 1.0 - 2.0 * eps
         for ax in range(1, W + 1):
             x = a * x + eps * x.sum(axis=ax, keepdims=True)
         return x.reshape(shape)
     ```
     In forward propagation: scatter `cur` along `fs = next_det_state(rule, W)` (`nxt = np.zeros_like(cur); np.add.at(nxt.T, fs, cur.T)`), then apply `noise_channel_fwht(nxt, W, eps) * e[:, t]`.
     In backward propagation: apply `tmp = noise_channel_fwht(cur * e[:, t+1], W, eps)`, then gather `cur = tmp[:, fs]`.
3. **4-Layer Primary Model + `R3-D0` Gate (`T1` training arm, 3 seeds per rule):**
   - Train the 4-layer transformer (`L = 4, d = 128, H = 4`) on $W = 14, T = 10$ (with $100\text{k}\text{--}150\text{k}$ training grids from the disjoint `train` row-0 pool so the 1-step rule over $N = 140$ permuted tokens converges cleanly to `S1` KL $\le 0.015$). If `L = 4` still has `F` KL $< 0.05$ on validation, step down to `L = 3` or `L = 2` via the `R3-D0` gate before running test chains.
4. **Add `P6-Hybrid` (`Propagate-Stiff-then-Branch`) and Epistemic Stiffness Comparison (`Mechanism A` vs. Entropy vs. TD/Self-Consistency):**
   - **`P6-Hybrid` (`hyb`):** At each step $t = 1 \dots M$ ($M \in \{1, 2, 4, 8, 16\}$):
     - If any unrevealed cell $z$ has one-pass predictive entropy $H(q_\theta(z \mid S_{t-1})) < 0.20$ nats ($p_{\max} > 0.95$, i.e. a newly unlocked stiff frontier deduction beyond the 1-pass depth horizon), pick the **stiffest** such cell ($\arg\min_z H(q_\theta(z \mid S_{t-1}))$) and clamp its value;
     - Otherwise (when all remaining unrevealed cells have $H \ge 0.20$, as on multimodal bridges once boundary rows are propagated), branch across the $K = 16$ chains on the **highest-variance / VoI-Unc** cell (`voiu`).
   - **All search arms (`stiff`, `conf`, `voi`, `voiu`, `hyb`, `rand`) are evaluated under both `sw_mix` (Round-2 anchored stiffness pool, primary) and `mix` (plain mixture), plus `dd` using Round-2's $h_i$-free key head $k_{\text{v2}}$.**
   - **Epistemic vs. Aleatoric Stiffness Diagnostic (`R3-D3`):** On the 2,000 held-out `(i, S)` evaluation pairs, compare three candidate stiffness scores at predicting **true conditional information gain** $\ln 2 - H^*(X_i \mid x_S)$ AND **low epistemic error** $-\mathrm{KL}(P^*(X_i \mid x_S) \,\|\, q_\theta(X_i \mid x_S))$:
     1. `Mechanism A` learned surprise head $\beta_{\text{int}}(i \mid S)$,
     2. Plain one-pass predictive confidence $\ln 2 - H(q_\theta(X_i \mid x_S))$, and
     3. **TD / Self-Consistency Epistemic Stiffness** $\beta_{\text{TD}}(i \mid S) = -\mathbb{E}_{z \sim q_\theta(z^* \mid S)}[\mathrm{KL}(q_\theta(X_i \mid S \cup \{z\}) \,\|\, q_\theta(X_i \mid S))]$ (or a calibration MLP predicting this 1-step rollout drift).

### Pre-Registered Decision Rules (`R3-D0`, `R3-D2a`, `R3-D2b`, `R3-D3`)
- **`R3-D0` (Zero-Overlap Regime Gate):** On $W = 14, T = 10$ with disjoint row-0 splits (`0.0%` train/test row-0 overlap), 3-seed mean validation `P0` has `S1` KL $\le 0.015$ and `P0` KL $\ge 0.08$ on at least 2 of `{F, B1, B2}` on Rule 110 (and $\ge 0.05$ on Rule 90).
- **`R3-D2a` (Forward Multi-Step Propagation Without Memorization):** On `F` (rows 6–9 given row 0) at $M = 16$ under `sw_mix`, `P2-Stiff`, `P3-Conf`, and `P6-Hybrid` each cut test KL by $\ge 25\%$ relative to `P0` AND beat `P4-Rand` by $\ge 15\%$ relative KL (3-seed paired $t$-CI of $\mathrm{KL}(\text{rand}) - \mathrm{KL}(\text{arm})$ excluding 0).
- **`R3-D2b` (Hybrid `Propagate-Stiff-then-Branch` Across Both Forward and Bridges):** Under `sw_mix` at $M = 16$, `P6-Hybrid` cuts test KL by $\ge 25\%$ relative to `P0` on **all three** long-range families (`F`, `B1`, `B2`) on Rule 110, beating pure `P2-Stiff` on Bridges (`B1, B2`) and beating pure `P4-Rand` on Forward (`F`) with paired 95% CIs excluding 0.
- **`R3-D3` (Epistemic vs. Aleatoric Stiffness Check):** Report Spearman rank correlations of $\beta_{\text{int}}$, $\ln 2 - H(q_\theta)$, and $\beta_{\text{TD}}$ against both $\ln 2 - H^*$ and $-\mathrm{KL}(P^* \,\|\, q_\theta)$ (overall and stratified by predictive entropy $H(q_\theta)$); **PASS** if either $\beta_{\text{int}}$ or $\beta_{\text{TD}}$ beats plain $\ln 2 - H(q_\theta)$ by $\ge 0.05$ Spearman on $-\mathrm{KL}(P^* \,\|\, q_\theta)$ within matched-entropy strata.
