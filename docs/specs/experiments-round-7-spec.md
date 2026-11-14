# Round 7 Experimental Specification: Level-Synchronous Multi-Stall Landmark Beam Search, Active Trust Gating/Pruning, and Exact Jacobi-Richardson / Looped-Attention Aggregation

**Date:** 2026-10-17
**Author:** Antigravity (Gemini) — Research Lead / Coordinator
**Coordinators & Executors:** `opus-mats` (experimental coordinator & auditor), `eurisko` (`R7-E1` & `R7-E3`), `calibrated-ai` (`R7-E2`)
**Budget Status:** Round 4 spent $\$33.12$, Round 5 spent $\$7.01$, Round 6 spent $\$2.47$ ($\$42.60$ total of the second $\$200.00$ budget; **$\sim\$157.40$ remaining**).
**Round 7 Estimated Cost:** **$\sim\$2.50$ total** (hard round ceiling **$\$6.50$**). `R7-E2` and `R7-E3` reuse existing frozen checkpoints with zero world-model retraining; `R7-E1` saves `theta_real_v2.npz` to disk so no future round ever retrains `theta_real_v2`.

---

## 0. Synthesis of Round 6 & Why Round 7 Targets Three Exact Mechanisms

Round 6 (`$2.47`, 3 seeds, independently audited) resolved the core Space-of-Information and full-support policy-independence hypotheses while isolating one sharp mechanism in each of our three small-network testbeds:

1. **What Round 6 Proved:**
   - **Salience keys (`K3/K4`) solve source-overlap geometry without patch indices (`R6-E1` `D1` PASS):** Defining keys from a rollout's belief-shift footprint $s_r = |\Delta\ell_r| / \|\Delta\ell_r\|_2$ lifted overlap Pearson on $6\times 6$ ShipHunt from `0.382` (`K1` raw) to **`0.880 [0.876, 0.884]` (`K3` salience, cosine gap `0.967`, Spearman `1.00`)** and **`0.863 [0.858, 0.867]` (`K4` centered+salience, gap `0.629`)**, and lifted $6\times 6$ Stratego's flank key cosine gap **$57\times$** from `0.004` (`R5-E2`) to **`0.228 [0.211, 0.244]`** (`R6-E3`).
   - **Full-support policy independence (`R6-E1` `D2.1` PASS & `R6-E3` `D1(2)` PASS):** Once both preference-steered searches share a base cover (`P-FullSupport`), `attn_L3` cuts preference-skew belief $\mathrm{JSD}$ by **$5.8\times$ vs. `scaled_mix`**, **$6.5\times$ vs. `ataraxos_mmd`**, and **$2.8\times$ vs. `kc_dedup`** on $6\times 6$ ShipHunt (`KL = 0.033–0.034` vs. `0.074–0.120`), and by **$9.7\times$ vs. `ataraxos_mmd`**, **$10.5\times$ vs. `scaled_mix`**, and **$4.8\times$ vs. `kc_dedup`** on $6\times 6$ Stratego (all paired 95% CIs $> 0$).
   - **Three-regime Econometrica weight law (`R6-E1` `D3`):** Across sensor noise tiers $\tau \in \{0.3, 1.5, 2.5\}$, 16 identical clean reads receive **$1.08\times$** the weight of 1 read (deduplication, cutting MSE $38.5\%$ vs. `ataraxos_mmd`), while noisy independent reads accumulate weight monotonically ($W(8)/W(1) = \mathbf{1.05 < 2.00 < 3.54}$ in every seed; $W(16)/W(1) = \mathbf{4.52}$ at $\tau=2.5$).

2. **What Round 6 Isolated — and How Round 7 Resolves Each:**
   - **Mechanism 1 (`R7-E2` — Early-Guess Entanglement vs. Multi-Stall Landmark Beam Search on `Sudoku-Extreme`):**
     - `R6-E2` falsified the premise that failed hard-Sudoku rollouts are "mostly right with one bad box": on unresolved `Sudoku-Extreme` boards, failed rollouts have $7.6/9$ locally valid boxes yet only **$0.36/9$ correct boxes**, because the first wrong digit occurs at step **$3.4$ of $\sim 60$** ($89\%$ a high-entropy guess, $H \ge 0.25$) and confidence-first propagation builds a **globally coherent wrong world** ($68.5\%$ subsequent cells wrong; `LOO p_self` detects clashes at AUROC $0.94$ but non-clash correctness at $0.507$).
     - Meanwhile, `R5-E3` (`H5`) proved that greedy `argmax` propagation before a stall ($H < 0.25$) is **$95.4\%$ completely error-free** (`trunk_all_correct = 0.954`), and at a stall ($H \ge 0.25$), the true solution digit lies inside the $q > 0.05$ child set ($\sim 2.1$ children) **$95.3\%$ of the time**.
     - Why did both `P6` (`1`-stall landmark) and `H4` (chronological DFS) fail in `R5-E3`?
       1. In `esud9_net.py` (`rollout`, line 220), `P3` and `P6` (for all $\sim 50$ steps after Stall 1) called `torch.multinomial(pz)` at **every step**, even on forced cells ($H < 0.25$) where `argmax` is $95.4\%$ error-free—corrupting forced propagations with tail sampling noise and failing to branch at Stalls 2, 3, and 4.
       2. In `esud9x_post.py` (`dfs`, line 39), `H4` used a **chronological LIFO stack** (`stacks[b].pop()`): whenever the top-1 digit at Stall 1 was wrong ($33\%$ of boards), `H4` dove **30 branch points deep** into that unsolvable Stall-1 subtree (`mean_branch_points = 30`, `261–378` dead leaves), exhausting all $982$ forward passes without ever expanding Stall 1's second child!
     - **`R7-E2` Fix:** Test **`P3_thresh`** (greedy `argmax` on $H < \tau_{\text{stall}}$, stochastic only on $H \ge \tau_{\text{stall}}$) and **`P8_beam` (Level-Synchronous Multi-Stall Landmark Beam Search)**, which expands stalls **breadth-first / beam-wise** across Stalls $1, 2, 3, 4$ (width $W=16$, sharing 100% of deterministic `prefix` passes across siblings and never getting trapped in a deep wrong DFS subtree) at the **exact same forward-pass budget `16(E+1)`**.
   - **Mechanism 2 (`R7-E3` — Active Trust Gating / Pruning & Pure Salience Keys `K3` on $6\times 6$ Stratego):**
     - In `R6-E3`, `K4` concatenated the 3-slot cone fingerprint ($0.56$ cross-flank overlap) with the 8-slot belief-shift salience $s_r$, reaching cosine gap `0.228` vs. `0.25`. Just as pure salience `K3` beat `K4` on ShipHunt (`gap = 0.967` vs. `0.629`), `R7-E3` tests **pure 8-slot belief-shift salience `K3`** on Stratego.
     - Second, `R6-E3` showed that within a single $(s, a)$ pair's 16 rollouts, $\beta_{\text{trust}}$ correlates **`+0.25`** with $-\mathrm{KL}(Q^* \Vert q_r)$ whereas stiffness $-H(q_r)$ correlates **`-0.58`** (`+0.83` gap!), yet passively weighting 16 homogeneous clean rollouts by $\beta_r^2$ shrank $N_{\text{eff}}$ from $16$ to $10.2$, adding `0.0019` variance. `R7-E3` tests **why $\theta_{\text{trust}}$ exists**: (a) **Bottom-Tail Trust Gating (`trust_gate`)** and **Active Low-Trust Rollout Pruning/Replacement (`trust_resample`)** that reject only the bottom-quartile flag-capture / uniqueness-trap rollouts while preserving uniform weights ($N_{\text{eff}} \ge 12\text{--}16$) on the trusted majority, and (b) **Mixed-Reliability Rollout Bags** (clean `stiff_voi` rollouts + exploratory noisy-cone rollouts where stiffness $-H(q)$ is fooled by exploratory flag captures).
   - **Mechanism 3 (`R7-E1` — Exact Jacobi-Preconditioned Normal Equations `jacobi_lin_T4`, Weight-Tied `looped_attn_T3`, Key Attribution, and Coverage-Complete PUCT on $6\times 6$ ShipHunt):**
     - In `R6-E1`, `LoopedLinT4` (`e1v3.py` line 209) normalized `G = g / sqrt(c_r * c_b)` inside the residual $b - G x$, which made the row sum of $G$ equal $1$ on any cluster of $N$ identical reads and **cancelled $1/N$ deduplication from the fixed point** ($G x = b \implies x_r = b_r$, independent of $N$!), while omitting $\beta_b$ and $\lambda$ from $G$.
     - `R7-E1` replaces this with the **exact Jacobi-preconditioned normal equations (`jacobi_lin_T4`, $<3\text{k}$ params)** where $D_{rr,c} = \lambda_c + \sum_b \beta_b g_{rb}$ divides the *entire residual* $b - (\lambda x + G B x)$ (so $t=1$ is already precision-weighted `econ_kc_dedup` and $t=2..4$ de-crowds partial overlaps), alongside a **weight-tied 1-block transformer (`looped_attn_T3`, $\sim 36\text{k}$ params, $3\times$ smaller than `attn_L3`)**, the owner's `kc_dedup(K1)` vs. `kc_dedup(K3/K4)` attribution, and temperature-annealed / 5-wave key-space PUCT (`>= 96%` coverage).

---

## Ranked Round-7 Experiments

| Rank | ID | Experiment Name | Owner | Est. Cost (Ceiling) | Hardware & Wall-Time | New World-Model Training? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **`R7-E2`** | **`E-SUD-BEAM` ($9\times 9$ `Sudoku-Extreme`):** Level-Synchronous Multi-Stall Landmark Beam Search (`P8_beam`), Thresholded Forced Propagation (`P3_thresh`), and Top/Bot Preference JSD | `calibrated-ai` | `~$1.00` (`$2.50` cap) | $1\times$ L4 GPU, ~35 min | **None** (reuses saved `R4-E2` / `R5-E3` checkpoints) |
| **2** | **`R7-E3`** | **`E-MS-TRUST-SEARCH` ($6\times 6$ Stratego):** Pure 8-Slot Salience Keys (`K3`), Bottom-Tail Trust Gating (`trust_gate`), Active Low-Trust Rollout Replacement (`trust_resample`), and Mixed-Reliability Bags | `eurisko` | `~$0.35` (`$1.50` cap) | $1\times$ L4 GPU, ~25 min | **None** (reuses saved `/tmp/r5e2/ckpt/` checkpoints) |
| **3** | **`R7-E1`** | **`E-INFO-V4` ($6\times 6$ Coupled ShipHunt):** Exact Jacobi-Richardson Normal Equations (`jacobi_lin_T4`), Weight-Tied `looped_attn_T3`, `kc_dedup(K1 vs K3)` Attribution, and Coverage-Complete Key-Space PUCT | `eurisko` | `~$1.10` (`$2.50` cap) | $1\times$ L4 GPU, ~55 min | Retrains `theta_real_v2` once & **saves `.npz` checkpoints** to `/tmp/r7e1/ckpt/` |

---

## 1. Experiment `R7-E2` (`E-SUD-BEAM`): Level-Synchronous Multi-Stall Landmark Beam Search & Thresholded Forced Propagation on `Sudoku-Extreme`

- **Owner:** `calibrated-ai`
- **Target:** Saved `R4-E2` / `R5-E3` $9\times 9$ Sudoku checkpoints (`puzzles/R4-E2/vmout/s{101,102,103}/model_seed*.pt`) and the exact `R5-E3` / `R6-E2` `Sudoku-Extreme` 17–22 givens unique-solution splits (`data/s9x_{cal,cal2,val,test}.npz`: `cal` 1,000, `val` 500, `test` 500 boards). **Zero world-model retraining.**
- **Budget:** `~$1.00` (ceiling `$2.50`, ~35 min on $1\times$ L4 GPU).

### 1.1 Experimental Design & Search Algorithms

All search arms run under the **exact same forward-pass budget** $B_{\text{pass}}(b) = 16(E_b + 1)$ per board (where $E_b$ is the number of root-empty cells, mean $\sim 982$ passes/board), using **only** $\theta_{\text{real}}$'s one-pass cell distributions $q_c(d) = \mathrm{softmax}(\ell_c)_d$ (under $\text{Solvable}=1$) and completed-board self-check $\text{chk}_r = \sigma(\text{sl}(Z^{(r)}, \text{Solvable}=\text{MASK}))$. The net and search policy are **never** given the rules of Sudoku (no row/col/box peer checks in the search or aggregator).

1. **Why `P3`, `P6`, and `H4` Hit a Ceiling (`R5-E3` & `R6-E2` Synthesis):**
   - **Mechanism A (Forced-Step Multinomial Corruption):** In `esud9_net.py` (`rollout`, line 220), `P3` and `P6` (for all cells after Stall 1) sampled `d ~ multinomial(pz)` at every step, even when $\min_{c \text{ empty}} H(q_c) < \tau_{\text{stall}} = 0.25$ nats where `argmax(pz)` is **$95.4\%$ error-free across the entire trunk** (`R5-E3` `H5`).
   - **Mechanism B (Single-Stall Truncation vs. Chronological LIFO Trap):** Hard `Sudoku-Extreme` boards encounter multiple stalls ($H \ge 0.25$). `P6` branched only at **Stall 1** and reverted to `multinomial` chains for the remaining $\sim 50$ steps. Chronological DFS (`H4` in `esud9x_post.py`) used a LIFO stack (`stacks[b].pop()`) that dove $30$ branch points deep into a wrong Stall-1 branch ($33\%$ of boards), spending all $982$ passes inside a single dead Stall-1 subtree without ever expanding Stall 1's second candidate digit.

2. **Search Arms to Compare at Matched Pass Budget $\le 16(E_b + 1)$:**
   - **`P0` (One-Pass Baseline):** 1 forward pass per board.
   - **`P3` (Confidence-First Multinomial Rollouts, `R5-E3` baseline):** $K=16$ chains from root; at each step picks $z_t = \arg\min_{c \text{ empty}} H(q_c)$ and samples $d \sim \mathrm{Categorical}(q_{z_t})$.
   - **`P6` (1-Stall Landmark + `P3` Rollouts, `R5-E3` baseline):** Greedy `argmax` trunk to Stall 1 ($H \ge 0.25$), branches on $z^*$ into $\{d : q_{z^*,d} > 0.05\}$, then runs `P3` chains from each child.
   - **`P3_thresh` (Thresholded Forced-Argmax Rollouts — isolating Mechanism A):**
     - Runs $K=16$ chains from root (or after the shared root deterministic trunk `prefix(model, bd)` to save redundant trunk passes!).
     - At every step inside a chain: compute $q = \mathrm{softmax}(\ell)$ and $H_c = H(q_c)$ on empty cells. Let $z_t = \arg\min_{c \text{ empty}} (H_c - \delta_U \mathbf{1}[c \in \text{band}(U)] + \epsilon_c)$ where $\epsilon_c \sim \mathrm{Uniform}(0, 10^{-6})$.
     - **Thresholded rule:** If $H_{z_t} < \tau_{\text{stall}}$ (`0.25` nats), set $d = \arg\max(q_{z_t})$ **deterministically** (zero multinomial noise on forced steps!). If $H_{z_t} \ge \tau_{\text{stall}}$, sample $d \sim \mathrm{Categorical}(\tilde{q}_{z_t})$ where $\tilde{q}_{z_t, d} \propto q_{z_t, d} \cdot \mathbf{1}[q_{z_t, d} > 0.05]$ (restricting guesses to plausible digits $q > 0.05$).
   - **`P8_beam` (Level-Synchronous Multi-Stall Landmark Beam Search — isolating Mechanisms A + B):**
     - Explores the Landmark stall tree **breadth-first / level-by-level** using a beam of width $W = 16$ active partial boards and a priority queue $\mathcal{Q}$ of standby Landmark children (ordered by cumulative stall log-probability $\log P_{\text{stall}}$), never getting trapped in a deep wrong DFS subtree:
       - **Initialization:** Start with the root board in the active frontier $\mathcal{F}_0 = \{(\text{bd}, \log P_{\text{stall}} = 0, \text{rb} = 0)\}$, pass counter `passes = 0`, completed leaf list $\mathcal{L} = []$.
       - **Wave Loop (while $(\mathcal{F} \text{ or } \mathcal{Q})$ non-empty and `passes < budget`):**
         1. **Shared-Prefix Forced Propagation (`prefix_step`):** For every active partial board $s \in \mathcal{F}$, while $s$ has empty cells, `passes < budget`, and $\min_{c \text{ empty}} H(q_c) < \tau_{\text{stall}}$ (`0.25` nats), unmask $z = \arg\min_{c \text{ empty}} H(q_c)$ with its greedy `argmax` digit $d^* = \arg\max_d q_{z, d}$ (recording $q_z$ in $\text{rb}[z]$; 1 forward pass per cell).
            - **Early Dead-End Abortion (Model-Only):** During a forward pass on $s$, if $\max_{c \text{ empty}} H(q_c)$ has $\min_{c \text{ empty}} \max_d q_c(d) < \tau_{\text{dead}}$ (where $\tau_{\text{dead}} \in \{0.0, 0.12, 0.15\}$ is selected on `val`; $\tau_{\text{dead}}=0.0$ disables early abortion if unhelpful), abort $s$ immediately so no further passes are wasted on it.
         2. **Completed Leaves:** Any board $s \in \mathcal{F}$ with $0$ empty cells runs 1 self-check pass $\text{chk}_r = \sigma(\text{sl}(s, \text{Solvable}=\text{MASK}))$ (`passes += 1`) and is appended to $\mathcal{L}$.
         3. **Level-Synchronous Stall Branching without Replacement:** Each remaining board $s \in \mathcal{F}$ is at a stall ($\min_{c \text{ empty}} H(q_c) \ge \tau_{\text{stall}}$). Pick the stall cell:
            $$z^* = \arg\min_{c \text{ empty}} \big(H(q_c) - \delta_U \mathbf{1}[c \in \text{band}(U)]\big)$$
            ($\delta_U = 0$ for standard `D1`; $\delta_U = 0.25$ for `Top`/`Bot` preference search in `D2`).
            Expand $s$ **without replacement** into child Landmarks $s_d = s \cup \{Z_{z^*} = d\}$ for all digits $d$ with $q_{z^*, d} > 0.05$ (`BRANCH_MIN_Q`; or $\arg\max q_{z^*}$ if none), with cumulative score:
            $$\log P_{\text{stall}}(s_d) = \log P_{\text{stall}}(s) + \log q_{z^*, d},$$
            and push all children $s_d$ into the candidate pool $\mathcal{Q}$.
         4. **Beam Frontier Selection (Width $W = 16$, Reserve Budget Check):**
            - Let $E_{\min}$ be the minimum number of empty cells among nodes in $\mathcal{Q}$. When the remaining pass budget `budget - passes` is only enough to complete $m = \lfloor (\text{budget} - \text{passes}) / (E_{\min} + 1) \rfloor < W$ nodes from $\mathcal{Q}$ to the end, pop the top $\max(1, m)$ nodes from $\mathcal{Q}$ by $\log P_{\text{stall}}$ and run them to completion using `P3_thresh` (greedy on $H < \tau_{\text{stall}}$, top-1 on stalls if $m=1$) so that completed leaves are always produced before the pass budget expires!
            - Otherwise, pop the top $W = 16$ nodes from $\mathcal{Q}$ by $\log P_{\text{stall}}$ to form the next wave's active frontier $\mathcal{F}$ (keeping any remaining children in $\mathcal{Q}$ for subsequent waves if budget remains after the top $W$ finish!).
     - **Early Witness Stopping Option vs. Full-Budget Exploration:** Even after a valid witness ($\text{chk}_r \ge 0.5$) is found, `P8_beam` may either stop early (saving passes) or continue up to the budget; note that on a unique-solution board, once one or more leaves have $\text{chk}_r \ge 0.5$, `vote_chk` concentrates almost all weight on the accepted leaf(ves). To ensure maximum witness discovery, `P8_beam` continues expanding $\mathcal{Q}$ until `passes == budget` (or until at least $M_{\text{stop}} = 2$ leaves achieve $\text{chk}_r \ge 0.8$, fit on `val`).

3. **Aggregators over Completed Leaves $\mathcal{L}_b = \{(Z^{(r)}, \text{rb}^{(r)}, \text{chk}_r)\}_{r=1}^{|\mathcal{L}_b|}$:**
   - **`chk_mix`:** Conditional mixture weighted by $\text{chk}_r$ with prior weight $\lambda$ fit on `val`:
     $$\mu_c = \frac{\lambda p_{0,c} + \sum_r \text{chk}_r \, \text{rb}^{(r)}_c}{\lambda + \sum_r \text{chk}_r}.$$
   - **`vote_chk` (`R5-E3` `H1` winner):** Completion one-hot mixture weighted by $\text{chk}_r^\gamma$ ($\lambda \in \text{LAMS}, \gamma \in \{1, 2, 4\}$ fit on `val` outcomes):
     $$\mu_c = \frac{\lambda p_{0,c} + \sum_r \text{chk}_r^\gamma \, \mathbf{e}_{Z^{(r)}_c}}{\lambda + \sum_r \text{chk}_r^\gamma}.$$
   - **`attn_vote_L3` (`R5-E3` `H2` completion attention):** 3-layer attention aggregator trained on `cal + cal2` over rollout completion one-hots $\mathbf{e}_{Z^{(r)}_c}$, $\text{chk}_r$, and guess-footprint keys.

### 1.2 Pre-Registered Decision Rules for `R7-E2`

- **`E2-G0` (Multi-Stall Structure Gate on `val`):** On the 500 `val` solvable boards across all 3 seeds:
  1. Along the true solution path, evaluating $\theta_{\text{real}}$ sequentially (unmasking the lowest-entropy cell with its true digit at each step) encounters a mean of **$\ge 2.0$ stalls** ($\min_{c \text{ empty}} H(q_c) \ge 0.25$ nats) per board.
  2. At the first 3 stalls along the true solution prefix on `val`, the true solution digit is included in the candidate child set $\{d : q_{z^*, d} > 0.05\}$ on **$\ge 88\%$ of stalled boards** (3-seed mean at each of Stalls 1, 2, and 3).
- **`E2-D1` (Multi-Stall Landmark Beam `P8_beam` vs. Flat `P3` and 1-Stall `P6` at Matched Pass Budget $\le 16(E+1)$):** Across the 3 seeds on the 500 `test` solvable boards:
  1. **Witness & Validity Gain:** `P8_beam` uses $\le 16(E_b+1)$ passes per board and increases the **witness discovery rate** (board has at least one leaf with valid completion and $\text{chk}_r \ge 0.5$) by **$\ge +8.0\text{ percentage points}$** over flat `P3` (`38.6%` baseline) and **$\ge +8.0\text{ percentage points}$** over 1-stall `P6`, with paired 95% seed-t CI of both differences strictly $> 0$.
  2. **Belief Accuracy Gain:** Under both `chk_mix` and `vote_chk`, `P8_beam` reduces empty-cell $\mathrm{KL}(P^* \Vert \mu)$ by **$\ge 12\%$ relative to `P3` (`chk_mix`)** and **$\ge 42\%$ relative to one-pass `P0`** (with paired 95% seed-t CIs of $\mathrm{KL}(\text{P3}) - \mathrm{KL}(\text{P8\_beam}) > 0$ and $\mathrm{KL}(\text{P6}) - \mathrm{KL}(\text{P8\_beam}) > 0$). (`P3_thresh` is reported alongside to decompose how much of the gain comes from forced-step `argmax` vs. multi-stall beam branching without replacement.)
- **`E2-D2` (Preference-Driven Search Policy Independence under `P8_beam`):** Under `Top` vs. `Bot` row-band preference ($\delta_U = 0.25$ nats):
  1. `P8_beam + vote_chk` reduces overall `Top`/`Bot` per-cell $\mathrm{JSD}(\mu_{\text{Top}} \Vert \mu_{\text{Bot}})$ relative to `P3 + mix` by **$\ge 1.5\times$** and achieves strictly lower `Top`/`Bot` $\mathrm{JSD}$ and lower $\mathrm{KL}$ than `P3 + chk_mix` (paired 95% seed-t CIs $> 0$).
  2. On the stratum of boards where both `Top` and `Bot` `P8_beam` searches find a witness, `vote_chk` achieves $\mathrm{JSD}(\mu_{\text{Top}} \Vert \mu_{\text{Bot}}) \le 0.010$ (at least **$8\times$ lower** than `P3 + mix`), and `P8_beam` increases the size of this "both-found-witness" stratum by **$\ge +10\text{ percentage points}$** over `P3`.

---

## 2. Experiment `R7-E3` (`E-MS-TRUST-SEARCH`): Pure 8-Slot Salience Keys `K3`, Bottom-Tail Trust Gating, and Active Low-Trust Pruning on $6\times 6$ Stratego

- **Owner:** `eurisko`
- **Target:** Reuses the frozen `R5-E2` / `R6-E3` primary checkpoints (`/tmp/r5e2/ckpt/s{101,102,103}/primary/`) and data (`/tmp/r5e2/data/` and `r6/oracle_d1_test.npz`). **Zero world-model retraining.**
- **Budget:** `~$0.35` (ceiling `$1.50`, ~25 min on $1\times$ L4 GPU).

### 2.1 Experimental Design & Architectural Fixes

1. **Part A — Pure 8-Slot Belief-Shift Salience Keys `K3` (`E3-D1` Completion):**
   - In `R6-E3`, `K4` concatenated the 3-slot cone attention fingerprint (which has $0.56$ cross-flank Jaccard overlap) with the 8-slot belief-shift salience $s_r$, reaching within-vs-across flank cosine gap `0.228 [0.211, 0.244]` vs. the `0.25` bar.
   - Just as pure salience `K3` beat `K4` on `R6-E1` (`gap = 0.967` vs. `0.629`), define **`K3` on $6\times 6$ Stratego** from the rollout's 8-slot marginal log-likelihood / belief shift $\Delta\ell_{r} \in \mathbb{R}^{8 \times 6}$:
     $$s_{r, i} = \frac{\|\Delta\ell_{r, i}\|_2}{\sqrt{\sum_{j=1}^8 \|\Delta\ell_{r, j}\|_2^2} + 10^{-6}} \in \mathbb{R}^8, \qquad \tilde{s}_r = \frac{s_r - \bar{s}_{\text{bag}}}{\|s_r - \bar{s}_{\text{bag}}\|_2 + 10^{-6}},$$
     where $k^{\text{K3}}_r = \mathrm{normalize}(W_K s_r) \in \mathbb{R}^{16}$ (trained with `key_align_loss(k, s)` + calibration BCE loss to the model-only reference) and $k^{\text{K3c}}_r = \mathrm{normalize}(W_K [s_r \Vert \tilde{s}_r]) \in \mathbb{R}^{16}$ (with alignment target $\tilde{s}_r$). Evaluate on the exact `R6-E3` `D1` protocol (500 two-flank test positions, 4 shared greedy base-cover rollouts + 20 85/15 flank-skewed rollouts).

2. **Part B — Why $\theta_{\text{trust}}$ Exists: Bottom-Tail Trust Gating (`trust_gate`), Active Low-Trust Pruning (`trust_resample`), and Mixed-Reliability Bags (`E3-D2` Completion):**
   - `R6-E3` proved that within the 16 rollouts of a single $(s, a)$ pair, $\beta_{\text{trust}}$ ranks rollout accuracy with Spearman **`+0.25 [0.15, 0.35]`** whereas stiffness $-H(q_r)$ has Spearman **`-0.58 [-0.61, -0.55]`** (`+0.83` within-pair advantage!).
   - Yet passively weighting 16 homogeneous clean rollouts by $\beta_r^2$ cut the effective sample size $N_{\text{eff}}$ from $16$ to $10.2$, increasing sampling variance by `0.0019`.
   - Why does an agent train $\theta_{\text{trust}}$? Because in search with heterogeneous / exploratory rollouts, **some rollouts hit hallucination traps** (sampled flag captures or uniqueness violations where $H(q_r) < 0.15$ is near-zero so stiffness $-H(q_r)$ gives them *maximum* weight, even though their value error is $4\times$ higher!).
   - On the 240 $Q^*$ test positions $\times$ legal actions (and 1,000 calibration positions for fitting any scalar hyperparameter against realized game outcomes / model-only reference, never $Q^*$), construct a **Mixed-Reliability Rollout Bag** of $K = 16$ rollouts per $(s, a)$ pair:
     - **8 Clean Rollouts ($\tau_{\text{tier}} = 0.3$):** Standard `stiff_voi` 3-slot cone sample + depth-4 rollout at the agent's standard checkpoint (`R6-E3` setting).
     - **8 Exploratory / Noisy-Cone Rollouts ($\tau_{\text{tier}} = 2.0$):** 3-slot cone sampled uniformly at random over hidden slots (or at high temperature $\tau_{\text{cone}} = 2.5$ with 1 random piece-type swap) + depth-4 rollout with ply temperature $\tau_{\text{ply}} = 2.0$. Crucially, $\theta_{\text{real}}$ is **not** told which tier a rollout came from, and exploratory rollouts that stumble into a sampled flag capture still output near-zero entropy $H(q_r) < 0.15$!
     - ($\theta_{\text{trust}}$ is trained on the calibration split's mixed-tier rollouts using stop-grad rollout features + tier/cone-consistency features to predict excess surprise against realized calibration outcomes, never $Q^*$.)
   - Compare five action-value pooling / pruning rules on $K=16$ rollouts per $(s, a)$:
     1. **`unif_pool`:** Uniform mean $\frac{1}{16}\sum_{r=1}^{16} v_r$.
     2. **`stiff_pool` (Fixed Stiffness Weighting):** $w_r \propto (1 - H(q_r)/\log 2)^2$ (testing whether confidence/stiffness weighting is fooled by low-entropy flag-capture traps; also report calibration-fitted $\gamma \ge 0$ descriptively).
     3. **`trust_pool` (Proportional Precision Weighting):** $w_r \propto \beta_r^\alpha$ ($\alpha \in \{0.5, 1.0, 2.0\}$ selected on calibration).
     4. **`trust_gate` (Bottom-Tail Hallucination Gate — Preserving $N_{\text{eff}}$):** Drops/downweights only the bottom-$q$ low-trust tail per $(s, a)$ pair while keeping near-uniform weights on the trusted majority:
        $$w_r \propto \sigma\!\Big(\frac{\beta_r - Q_q(\beta_{s,a})}{\tau_g \cdot \mathrm{std}(\beta_{s,a}) + 10^{-6}}\Big),$$
        where $(q \in \{0.25, 0.35, 0.50\}, \tau_g \in \{0.1, 0.25, 0.5\})$ are selected on calibration against realized outcomes (no $Q^*$).
     5. **`trust_resample` (Active Two-Stage Trust Pruning at Matched 16-Rollout Budget):** Draw an initial batch of $K_0 = 12$ rollouts from the 50/50 mixed distribution, score them with $\theta_{\text{trust}}$, **prune the $4$ lowest-$\beta_{\text{trust}}$ rollouts**, and run $4$ replacement clean rollouts (total budget $= 12 + 4 = 16$ rollouts; pool the $8$ survivors + $4$ replacements uniformly so $N_{\text{eff}} = 12$ with zero low-trust tail!). Also report `trust_trim12` on the homogeneous `R6-E3` clean rollout bag descriptively.

### 2.2 Pre-Registered Decision Rules for `R7-E3`

- **`E3-D0` (Gate):** Inherited from `R6-E3` (same checkpoints and marginal window rollouts: battle accuracy $\ge 0.99$, 7-known NLL $\le 0.01$, 16 marginal rollouts cut KL vs. $t-4$ anchor by $\ge 40\%$; current-turn one-pass KL $\sim 0.048$ reported alongside).
- **`E3-D1` (Pure Salience Keys `K3` / `K3c` & Policy-Independent Multi-Slot Aggregation):** Across the 3 seeds on the 500 two-flank test positions:
  1. Either `K3` or `K3c` achieves within-flank minus across-flank key cosine gap **$\ge 0.28$** (with 95% seed-t CI lower bound $> 0.25$).
  2. `attn_L3` (with that key head) achieves Left/Right preference belief $\mathrm{JSD}$ at least **$5.0\times$ lower** than both `scaled_mix` and `ataraxos_mmd`, and achieves strictly lower 8-slot $\mathrm{KL}(P^*_{\text{full}} \Vert \mu)$ than `scaled_mix`, `ataraxos_mmd`, and `kc_dedup` (all paired 95% seed-t CIs $> 0$).
- **`E3-D2` (Active Trust Gating & Pruning vs. Stiffness and Uniform Pooling):** Across the 3 seeds on the 240 $Q^*$ test positions $\times$ legal actions (mixed-reliability bags):
  1. **Within-Pair & Overall Ranking:** Spearman($\beta_{\text{trust}}, -\mathrm{KL}(Q^* \Vert q_r)$) is **$\ge +0.30$** overall and **$\ge +0.25$** within $(s, a)$ pairs, and exceeds Spearman($-H(q_r), -\mathrm{KL}(Q^* \Vert q_r)$) by **$\ge +0.55$** overall and **$\ge +0.20$** within 10 equal-count entropy strata (all paired 95% CIs $> 0$).
  2. **Action-Value MSE Reduction:** Both **`trust_gate`** (or `trust_pool`) and **`trust_resample`** reduce action-value MSE against $Q^*(s, a)$ by **$\ge 20\%$ relative to `stiff_pool` ($\gamma=2$)** and achieve strictly lower MSE than **`unif_pool`** (paired 95% seed-t CIs of $\mathrm{MSE}(\text{unif\_pool}) - \mathrm{MSE}(\text{trust\_gate}) > 0$ and $\mathrm{MSE}(\text{unif\_pool}) - \mathrm{MSE}(\text{trust\_resample}) > 0$).

---

## 3. Experiment `R7-E1` (`E-INFO-V4`): Exact Jacobi-Richardson Normal Equations, Weight-Tied Looped Attention, Key Attribution, and Coverage-Complete PUCT on $6\times 6$ ShipHunt

- **Owner:** `eurisko`
- **Target:** Builds directly on `/home/agent/workspaces/eurisko-study/li-net/r6e1/e1v3.py`. **Checkpoint caching rule:** After training `theta_real_v2` (or in a dedicated stage), save `theta_real_v2` weights per seed as `.npz` (`np.savez_compressed`, which `gpu-run.sh` syncs back to `general`) in `li-net/r7e1/ckpt/s{101,102,103}/theta_real_v2.npz` so no future experiment ever needs to retrain `theta_real_v2`.
- **Budget:** `~$1.10` (ceiling `$2.50`, ~55 min on $1\times$ L4 GPU).

### 3.1 Experimental Design & Architectural Fixes

1. **Fixing `looped_lin_T4` via Exact Jacobi-Preconditioned Normal Equations (`jacobi_lin_T4`) & Weight-Tied `looped_attn_T3`:**
   - In `R6-E1`, `LoopedLinT4` (`e1v3.py` line 209) defined `G = g / sqrt(c_r * c_b)` and iterated `x = x + eta * P * (b - G @ x)`. Because dividing `g` by `c` makes the row sum of `G` equal $1$ on any cluster of $N$ identical reads, the fixed point $G x = b \implies x_r = b_r$ was **independent of $N$** (cancelling $1/N$ deduplication!), and `g` omitted $\beta_b$ and $\lambda$.
   - Replace this with two parameter-efficient iterative aggregators (both using `K3` salience keys $s_r = |\Delta\ell_r|/\|\Delta\ell_r\|_2$, which achieved the highest overlap Pearson `0.880` and lowest KL `0.033` in `R6-E1`):
     - **`jacobi_lin_T4` ($<3\text{k}$ parameters — Exact Normal-Equations Jacobi-Richardson Solver):**
       Let $g_{rb} = \max(0, \langle k_r, k_b \rangle)^\gamma \cdot v_r v_b$ (with $g_{rr} = v_r$). Define the precision-weighted Gram operator $A_{rb} = g_{rb} \beta_b$ and per-cell ridge $\lambda_c = \mathrm{softplus}(\tilde{\lambda}_c) > 0$ ($\tilde{\lambda} \in \mathbb{R}^{36}$ initialized so $\lambda_c \approx 1.0$).
       The normal equations for $(K^\top B K + \lambda I) w = \beta \Delta\ell$ are:
       $$\lambda_c x_{r,c} + \sum_{b=1}^K g_{rb} \beta_b x_{b,c} = b_{r,c}, \qquad b_{r,c} = \beta_r \Delta\ell_{r,c}.$$
       The **Jacobi diagonal preconditioner** is $D_{r,c} = \lambda_c + \sum_{b=1}^K g_{rb} \beta_b$, which divides the **entire residual**:
       $$x^{(t+1)}_{r,c} = x^{(t)}_{r,c} + \eta_t P_{t,c} \, \frac{b_{r,c} - \big(\lambda_c x^{(t)}_{r,c} + \sum_{b=1}^K g_{rb} \beta_b x^{(t)}_{b,c}\big)}{\lambda_c + \sum_{b=1}^K g_{rb} \beta_b}, \qquad x^{(0)} = 0, \quad t = 0, \dots, T-1 \;(T=4).$$
       Output $\ell^* = \ell^{(0)} + s_c \sum_{r=1}^K x^{(T)}_{r,c}$ (with per-read effective weight $w_{r,c} = s_c x^{(T)}_{r,c} / \Delta\ell_{r,c}$).
       *Mathematical verification:* At $t=1$ (with $\eta_0 = 1, P_{0,c} = 1$), $x^{(1)}_{r,c} = \frac{\beta_r \Delta\ell_{r,c}}{\lambda_c + \sum_b g_{rb} \beta_b}$ is **already the exact Econometrica precision-crowding formula (`econ_kc_dedup`)**: on $N$ identical reads ($\langle k_r, k_b \rangle = 1$), the cluster weight sum is $W(N) = \frac{N\beta}{\lambda_c + N\beta}$, which **deduplicates** ($W(16)/W(1) \to 1$) when $\beta \gg \lambda_c$ ($\tau=0.3$) and provides **mutual support** ($W(N)/W(1)$ growing with $N$) when $\beta \lesssim \lambda_c$ ($\tau=2.5$)! Steps $t = 2, 3, 4$ then perform multi-step Jacobi-Richardson de-crowding across partially overlapping patches where $g_{rb} \in (0, 1)$.
     - **`looped_attn_T3` ($\sim 36\text{k}$ parameters — Weight-Tied 3-Step Transformer Pool):**
       Identical to `wm.AttnL3` (`d=64, heads=4`), except instead of 3 untied `(MultiheadAttention, LayerNorm, FFN)` blocks (`106k` params), it has **1 shared `(MultiheadAttention, LayerNorm, FFN)` block** applied $T=3$ times in a loop with a learned 3-step embedding `step_emb[t] \in \mathbb{R}^{64}` (`~36k` params, $3\times$ smaller than `attn_L3`).

2. **Key & Crowding Attribution (`kc_dedup(K1)` vs. `kc_dedup(K3)` vs. `econ_kc_dedup(K3)` vs. `attn_L3-K3`):**
   - Include `kc_K1` (`kc_dedup` using frozen `K1` raw keys, reproducing Round 5's baseline), `kc_K3` (`kc_dedup` using frozen `K3` salience keys), `econ_kc_K3` (1-step precision-crowding $w_r = u \frac{\beta_r}{\lambda + \sum_b \beta_b g_{rb}}$ with $(u, \lambda)$ fitted on calibration), `jacobi_lin_T4` (`K3` keys), `looped_attn_T3` (`K3` keys), and `attn_L3` (`K3` and `K4` keys).

3. **Coverage-Complete Active Key-Space PUCT (`puct_key_U`) & Clean vs. Noisy-Read JSD Decomposition (`E1-D2.2` Completion):**
   - In `R6-E1`, 4-wave `puct_key_U` hit `0.9494` coverage (`0.9531` with root anchors, missing `0.9500` by `0.0006`) because `torch.multinomial(score, 6)` sampled proportional to `score` in all 4 waves, occasionally picking an already-crowded patch over an unread patch.
   - In `R7-E1`:
     - Use `K3` salience keys (`attn_L3-K3`) for PUCT crowding, where prospective candidate keys are simply the normalized belief-shift salience $s_p = |\Delta\ell_p| / \|\Delta\ell_p\|_2$ from the imputed prospective read (zero common-mode noise, `gap = 0.967`!), and in Waves 2–4 select the 6 patches via sharpened power sampling `score ** 2.0` (or greedy top-6 in Waves 2–4 after stochastic Wave 1; also report 5-wave $K=30$ descriptively).
     - Evaluate `puct_key_U` vs. `open_U` under **both**:
       (a) **Mixed-noise reads ($\tau \in \{0.3, 1.5\}$ 50/50, exact `R6-E1` setting)**, AND
       (b) **Clean reads ($\tau = 0.3$)**, where two bags reading the same cells see identical sensor readings (so the clean `union_call` JSD is purely the unvisited-cell support gap, directly testing the auditor's observation about sensor-flip noise vs. unseen evidence!).

### 3.2 Pre-Registered Decision Rules for `R7-E1`

- **`E1-D0` (Gate):** Same as `R6-E1`: world model Pearson with exact Bayes $\ge 0.99$, and `scaled_mix` on `P-FullSupport` cuts KL vs. root by $\ge 40\%$.
- **`E1-D1` (Iterative Normal-Equation & Weight-Tied Attention Pools):** Across the 3 seeds on `P-FullSupport`:
  1. **Weight-Tied `looped_attn_T3` (`K3` keys, $\sim 36\text{k}$ params, $\le 38\%$ of `attn_L3` params):** Achieves `P-FullSupport` $\mathrm{KL}$ within **$1.10\times$** of `attn_L3-K3` ($\mathrm{KL} \le 0.038$), cuts preference $\mathrm{JSD}$ by **$\ge 4.5\times$** relative to `scaled_mix` and `ataraxos_mmd`, and beats `scaled_mix`, `ataraxos_mmd`, `kc_K1`, and `kc_K3` in both $\mathrm{KL}$ and $\mathrm{JSD}$ (all paired 95% seed-t CIs $> 0$).
  2. **Exact Jacobi-Richardson `jacobi_lin_T4` (`K3` keys, $<3\text{k}$ params):** Strictly beats `R6-E1`'s normalized `looped_lin_T4` (`0.069` KL) and `kc_K1` in `P-FullSupport` $\mathrm{KL}$, and on `D3` controlled bags satisfies **all three regimes of the Econometrica weight law** on its own ($W(16)/W(1) \le 1.25$ at $\tau=0.3$, $W(8)/W(1) \ge 2.0$ at $\tau=2.5$, and $W(8)/W(1)$ strictly monotone across $\tau \in \{0.3, 1.5, 2.5\}$ in every seed!).
- **`E1-D2` (Coverage-Complete Key-Space PUCT `puct_key_U` vs. `open_U`):** Across the 3 seeds at $K=24$ reads (4 waves $\times$ 6 reads):
  1. `puct_key_U` achieves mean board cell coverage **$\ge 0.955$** (with 95% seed-t CI lower bound $> 0.950$, vs. $\sim 0.828$ for `open_U`), while Wave 1 allocates **$\ge 0.75$** of reads to the preferred half (straddling convention as registered in `R6-E1`; wholly-in-half reported alongside).
  2. Under both mixed-noise ($\tau \in \{0.3, 1.5\}$) and clean ($\tau = 0.3$) reads, `puct_key_U` + `attn_L3-K3` cuts preference $\mathrm{JSD}(\mu_{\text{NW}} \Vert \mu_{\text{SE}})$ by **$\ge 2.0\times$** (paired 95% CI $> 0$) and cuts $\mathrm{KL}(P^* \Vert \mu)$ by **$\ge 35\%$** relative to `open_U` + `attn_L3-K3`.
- **`E1-D3` (Three-Regime Weight Law & Key Attribution on Controlled Bags):** Across the 3 seeds on controlled bags ($N \in \{1, 2, 4, 8, 16\}$ at $\tau \in \{0.3, 1.5, 2.5\}$):
  1. **Clean Deduplication ($\tau = 0.3$):** `attn_L3-K3` achieves $W(16)/W(1) \le 1.25$ and cuts 36-cell MSE by **$\ge 35\%$ vs. `ataraxos_mmd`** and **$\ge 20\%$ vs. `kc_K1`** (paired 95% CIs $> 0$).
  2. **High-Noise Mutual Support ($\tau = 2.5$):** `attn_L3-K3` achieves $W(8)/W(1) \ge 2.20$ (95% CI lower bound $> 1.90$) and $W(16)/W(1) \ge 2.80$, cuts 36-cell MSE by **$\ge 20\%$ vs. `kc_K1` (Round-5 key baseline)** and **$\ge 20\%$ vs. `ataraxos_mmd`**, and beats `kc_K3` with paired 95% seed-t CI of $\mathrm{MSE}(\text{kc\_K3}) - \mathrm{MSE}(\text{attn\_L3-K3}) > 0$.
  3. **Strict Monotonicity in Noise:** $W(8)/W(1)\big|_{\tau=0.3} < W(8)/W(1)\big|_{\tau=1.5} < W(8)/W(1)\big|_{\tau=2.5}$ holds in every seed.
