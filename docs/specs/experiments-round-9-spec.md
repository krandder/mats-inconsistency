# Round 9 Experiments Specification: Genuine Strong-Root Safety, 1-Step Precision-Aware Crowding (`econ_prec_K3`), and Hybrid Full-Budget Landmark Beam (`P8_hybrid`) with Self-Consistent Check Gating

**Date:** 2026-10-17  
**Author:** Gemini (`argon` / MATS Research Mode)  
**Coordinator:** `opus-mats`  
**Executors:** `eurisko` (`R9-E1`), `calibrated-ai` (`R9-E2`) — *Stratego lane (`E-MS`) is closed after `R8-E3`*  
**Budget:** Estimated **$\sim\$0.80$** total GPU spend (hard ceiling **$\$1.35$** across both experiments; $\sim\$47.50$ of second $\$200$ spent across Rounds 4–8, $\sim\$152.50$ remaining)

---

## 0. Executive Summary & Why Round 9 Has Only Two Focused Experiments

Round 8 ($\sim\$1.49$ across `R8-E1`, `R8-E2`, `R8-E3`, zero world-model retraining) settled two of Kelvin's primary questions and closed the Stratego lane:
1. **Strict Anytime Monotonicity (`R8-E1` `D1.1` PASS & `R8-E2` `D2.1` PASS):**
   - On $6\times 6$ ShipHunt (`R8-E1` `D1.1` PASS), anchor-aware `looped_attn_T3` (`39,428` params) decreased exact-Bayes KL **at every single step $K \in \{0, 1, 2, 4, 8, 16, 24, 32\}$ in every seed** (`0.2797 -> 0.2305 -> 0.1909 -> 0.1504 -> 0.1026 -> 0.0559 -> 0.0308 -> 0.0303`, cutting KL by **$89.2\%$** and JSD to **`0.00575`**), and improved over the root at every $K$ across all four nested root strengths $M_0 \in \{2, 6, 12, 18\}$, whereas anchor-free `ataraxos_mmd` and `scaled_mix` turned back up at large $K$ and `kc_K1` degraded at $K=1$ (`0.2797 -> 0.3523`).
   - On `Sudoku-Extreme` (`R8-E2` `D2.1` PASS), check-gated pooling (`chk_mix` and `vote_chk`) strictly decreased test KL across **all $36/36$ step transitions** $m \in \{0, 1, 2, 4, 8, 16, 32\}$ (`1.178 -> 1.036 -> 0.979 -> 0.920 -> 0.809 -> 0.734 -> 0.644` nats; witnesses `14% -> 43% -> 51%`), whereas unweighted `mix` jumped to `5.98` nats at $m=1$.
2. **Precision-Aware Jacobi Ridge & $2\times 2$ PUCT Attribution (`R8-E1` `D2.1/D2.2`):**
   - Equipping the $812$-param Jacobi solver with a 2-parameter precision-aware ridge $\lambda(\beta_r) = \lambda_c \cdot 2\sigma(a - b\beta_r)$ (`jacobi_lin_T4_prec`, `814` params) learned tier-specific ridges ($\lambda \approx 0.35, 1.37, 2.20$ at $\tau \in \{0.3, 1.5, 2.5\}$) and **passed all four three-regime weight-law clauses**: clean deduplication **`W(16)/W(1) = 1.164 [1.147, 1.182]`** (down from `1.455`, bar $\le 1.25$), strict tier monotonicity **`W(8)/W(1) = 1.150 < 1.765 < 2.501`**, and noisy mutual support **`W(16)/W(1) = 2.816`** at $\tau=2.5$.
   - The $2\times 2$ PUCT factorial cleanly decomposed coverage: `K1+R6 = 92.00%`, `K1+R7 = 95.80%` (`+3.8 pp`), `K3+R6 = 96.51%` (`+4.5 pp`), **`K3+R7 = 98.76% [98.59%, 98.93%]`**.
3. **Stratego Lane Closed (`R8-E3`):**
   - Starting from Stratego's strong current-turn one-pass anchor $\ell_t$ (`KL = 0.0488`), 4-ply window rollouts (`KL = 0.080`) carry almost no additional information: naive `raw_count` (`+0.591` nats) and fixed-step `ataraxos_mmd` (`+4.73` nats) wreck $\ell_t$ by double-counting, while `scaled_mix` ($\alpha=0.039$) and early-stopped `looped_attn_T3-K3c` stay strictly non-degrading across all $K$ (`-0.0025` nats at $K=16$). No further Stratego runs are needed.

### What Remains Open for Round 9 (Two Focused Experiments, $\sim\$0.80$ Total)

1. **`R9-E1` (`E-INFO-STRONG-ROOT`, `eurisko`, est. `~$0.35`, cap `$0.60`, 0 world-model training):**
   - **Why `R8-E1` missed 3 single clauses:**
     1. In `E1-D1.2`, $M_0 = 18$ anchor cells left the root at $\mathrm{KL}(0) = 0.1444$ (not $\le 0.050$) because each cell reading has a $15\%$ sonar flip rate; at $\mathrm{KL}(0) = 0.1444$ even `ataraxos_mmd` still improved the root at $K=16$ (`-0.0665` before turning back up to `-0.0457` at $K=24$). To test a **genuinely strong root** ($\mathrm{KL}(0) \le 0.045$), the root must see **$M_0 = 32$** (or $M_0 = 34$) anchor cells—where `mix` and standard-root `ataraxos_mmd` **genuinely degrade** $\mathrm{KL}(16) > \mathrm{KL}(0) + 0.015$, while anchor-aware `looped_attn_T3` **never degrades $\mathrm{KL}(0)$ at any $K \in \{1, 2, 4, 8, 16, 24\}$ in any seed**.
     2. In `E1-D2.2`, 1-step `econ_kc_K3` with a single tier-independent scalar $\lambda$ gave $W(16)/W(1) = 1.417$ at $\tau=0.3$ because a single $\lambda$ cannot simultaneously be small at $\tau=0.3$ and large at $\tau=2.5$. Equipping 1-step `econ_kc_K3` with the **exact same 2-parameter precision-scaled ridge** $\lambda(\beta_r) = 2\sigma(a - b\beta_r)$ as `jacobi_lin_T4_prec` (**`econ_prec_K3`**, `3` scalar params: $u, a, b$) gives closed-form $\frac{W(16)}{W(1)} = \frac{16(\lambda(\beta)+\beta)}{\lambda(\beta)+16\beta} \le 1.25$ at $\tau=0.3$ and $\ge 2.20$ at $\tau=2.5$, while `jacobi_lin_T4_prec` (`814` params) and `looped_attn_T3` (`39,428` params) beat 1-step `econ_prec_K3` on partially overlapping `P-FullSupport` bags (`KL <= 0.042` and `<= 0.032`).
2. **`R9-E2` (`E-SUD-HYBRID-SAFE`, `calibrated-ai`, est. `~$0.45`, cap `$0.75`, 0 world-model training):**
   - **Why `R8-E2` missed `D1.1/D1.2` and seed-103 `D2.2`:**
     1. In `R8-E2`, `P8_full` spent $99.5\%$ of the $16(E+1)$ pass budget (`D1.0` PASS), but replaced `R7-E2`'s Phase-B `P3_thresh` completion (`lead=(r==0)` greedy chain + forced-cell `argmax` for $H < 0.25$ + restricted stall sampling $q > 0.05$) with unrestricted multinomial sampling $d \sim q$ on every cell of every chain—causing a cumulative $1 - 0.985^{28} \approx 35\%$ clerical slip rate on forced cells after Wave 4! And running 3 seed processes concurrently on 1 L4 introduced a `2.9 pp` standalone-vs-prefix discrepancy (`42.8%` vs `45.7%`). `R9-E2` runs **one seed process at a time on the GPU** and evaluates the clean **$2\times 2$ Phase-B completion factorial** (`P8_beam`, `P8_full_stoch`, `P8_full_lead`, and **`P8_hybrid` = full-budget D'Hondt + greedy lead chain $r=0$ + forced-step `argmax` $H < 0.25$ + $q > 0.05$ stall sampling for $r \ge 1$**).
     2. In `E2-D2.2` (unresolved boards with zero witnesses at $m=16$), seeds 101 and 102 had exact `0.000` KL change vs. `P0`, while seed 103 degraded by `+0.065` due to its `7%` false accepts at $\text{chk} \ge 0.5$. Because the leaf forward pass `fwd(model, Z^{(r)})` *already* emits both `chk_r` and the 81-cell probability distribution $q_{\text{leaf}}(c, d)$ in the **same single pass** (zero extra passes, zero oracle/`valid9` access), weighting $\text{chk}_r$ by the leaf's own **internal cell-consistency floor** $\widetilde{\text{chk}}_r = \text{chk}_r \cdot (\min_{c \in E_b} q_{\text{leaf}}(c, Z^{(r)}_c))^{\eta_{\text{sc}}}$ with a strict acceptance floor $\tau_{\text{chk}}$ rejects seed 103's clashing false accepts and achieves **`KL(m=16) - KL(P0) <= +0.005` on unresolved boards across all 3 seeds** while preserving $36/36$ strict anytime monotonicity.

---

## Global Protocol (Mandatory for Both Executors)

1. **Pre-registration before GPU spend:** Commit `PREREG.md` (with SHA256 hashes of scripts and frozen cached checkpoints) before launching the main GPU job.
2. **Seeds & paired CIs:** 3 seeds (`101, 102, 103`), reporting 3-seed mean and paired `seed-t` 95% CIs ($\text{df}=2, t_{0.975,2}=4.303$).
3. **Zero world-model training & zero oracle leakage:** Load frozen world-model checkpoints (`R7-E1` for `R9-E1`, `R4-E2` for `R9-E2`). Never use exact Bayes $P^*$ or ground-truth Sudoku solutions / `valid9` in training, hyperparameter selection, or search control.
4. **Immediate VM teardown & cost log:** Tear down every GPU VM immediately on completion and log exact wall-clock and USD cost in `GPU-LOG.md`.

---

## 1. Experiment `R9-E1`: Genuine Strong-Root Anytime Safety & 1-Step Precision-Aware `econ_prec_K3` (`E-INFO-STRONG-ROOT`)

- **Executor:** `eurisko`
- **Target Path:** `/home/agent/workspaces/eurisko-study/li-net/r9e1/`
- **Estimated Cost:** $\sim\$0.35$ (hard cap **$\$0.60$**, 1 L4 GPU, reusing `li-net/r7e1/ckpt/s{101,102,103}/theta_real_v2.npz` and `li-net/r8e1/e1v5.py`).

### 1.1 Exact Architectural Additions over `R8-E1` (`e1v5.py` -> `e1v6.py`)

Keep `R8-E1`'s `e1v5.py` setup, checkpoints, and `E1-D0` gate unchanged, with two surgical extensions:

1. **Extended Nested Root Ladder $M_0 \in \{2, 6, 12, 18, 30, 32, 34\}$ for Strong-Root Safety (`D1`):**
   - In `R8-E1`, $M_0 \in \{2, 6, 12, 18\}$ gave root KLs `0.2797, 0.2452, 0.1930, 0.1444`, and at $M_0 = 36$ (all 36 cells revealed at the root) $\mathrm{KL}(0) = 0.0022$.
   - Extend the nested per-board anchor cell ladder to include **$M_0 \in \{30, 32, 34\}$** (where $M_0$ cells of the board's sonar grid + the commander's note are passed to `theta_real` to form the root belief $\ell^{(0)}(M_0)$ with anchor precision $\beta_0 = M_0 \beta_{\text{clean}} / 6$).
   - **Pre-scoring strong-root selection (on `cal` boards only):** Before scoring test boards, evaluate the calibration-board root error $\mathrm{KL}_{\text{cal}}(\ell^{(0)}(M_0) \Vert \text{ref}(36))$ for $M_0 \in \{30, 32, 34\}$ and select $M_0^{\text{strong}} \in \{30, 32, 34\}$ as the smallest $M_0$ whose calibration root $\mathrm{KL}_{\text{cal}} \le 0.035$ (if none is $\le 0.035$, pick $M_0 = 34$). This guarantees a genuinely strong test root $\mathrm{KL}(0) \le 0.050$ without touching test boards or exact Bayes.
   - **What a rollout read $r \ge 1$ is on the strong-root test (`D1.2`):**
     Wait—in a real search from a strong root $\ell^{(0)}(M_0^{\text{strong}})$, individual bounded-context rollouts do **not** re-feed all $M_0^{\text{strong}} = 32$ root cells into every single local patch call (that was why `R8-E1`'s reads at $M_0 = 18$ were actually 24-cell calls)! Instead, each local rollout $r \in \{1, \dots, 24\}$ is a **standard bounded-context local patch read** (the 2 base anchor cells + 6 patch cells $A_r$, with sensor noise tier $\tau_r \in \{0.3, 1.5, 2.5\}$) emitting its local posterior $\ell_r^{\text{local}}$ and its shift relative to the root on its inspected patch cells, $\Delta \ell_r = (\ell_r^{\text{local}} - \ell^{(0)}(M_0^{\text{strong}})) \odot \mathbf{1}_{A_r}$ (or full-board $\ell_r^{\text{local}} - \ell_{\text{prior}}$ with the anchor token $r=0$ carrying $\ell^{(0)}(M_0^{\text{strong}}) - \ell_{\text{prior}}$ at precision $\beta_0 = M_0^{\text{strong}} \beta_{\text{clean}} / 6$)!
     - **Report both read modes on $M_0^{\text{strong}}$:**
       - **Mode A (Cumulative-context reads, as in `R8-E1`):** Each read $r$ calls `theta_real` on the $M_0^{\text{strong}}$ root cells + patch $A_r$ (where any cell in $A_r$ gets the read's tier-$\tau_r$ reading) with three-tier noise $\tau_r \in \{0.3, 1.5, 2.5\}$ (`1/3` each), and `ataraxos_mmd` / `scaled_mix` use their standard ($M_0 = 2$) fitted step size.
       - **Mode B (Bounded-context local patch reads + strong root anchor):** Each read $r$ is a bounded 6-cell patch read (`M_0 = 2` base + 6-cell patch $A_r$, 90/10 skewed, $\tau_r \in \{0.3, 1.5\}$), while the root anchor token $r=0$ holds the strong root belief $\ell^{(0)}(M_0^{\text{strong}})$ with $\beta_0 = M_0^{\text{strong}} \beta_{\text{clean}} / 6$. Here naive `mix` and standard-step `ataraxos_mmd` (which add local patch shifts with fixed weight without knowing the root already has $M_0^{\text{strong}} \ge 30$ cells of evidence) strongly degrade $\ell^{(0)}(M_0^{\text{strong}})$, whereas anchor-aware `looped_attn_T3` and `attn_L3-K3` (which attend to the anchor token's footprint $s_0$ and precision $\beta_0$) downweight redundant patch reads on already-covered cells and only incorporate information on the remaining $36 - M_0^{\text{strong}}$ cells!
     - Train `looped_attn_T3` and `attn_L3-K3` on calibration boards across $M_0 \in \{2, 6, 12, 18, 30, 32, 34\}$ and $K \in \{0, 1, 2, 4, 8, 16, 24, 32\}$ against the model-only reference $\text{ref}(M_0)$ (using the `R8-E1` amendment's $\max(M_0)$ reference $\text{ref}(34)$ or $\max(\text{ref}(M_0), \ell^{(0)}(M_0))$ so the target is never weaker than the root).

2. **1-Step Precision-Aware Crowding (`econ_prec_K3`, `3` scalar parameters) vs. `jacobi_lin_T4_prec` (`814` params) (`D2`):**
   - Define **`econ_prec_K3`** with 3 learnable scalar parameters $(u > 0, a \in \mathbb{R}, b \ge 0)$:
     $$w_r \;=\; \frac{u \, \beta_r}{\lambda(\beta_r) \;+\; \sum_{b'=1}^K \beta_{b'} \max(0, \langle k_r, k_{b'} \rangle)}, \qquad \lambda(\beta_r) \;=\; \lambda_0 \cdot 2\sigma(a - b\beta_r) \quad (\lambda_0 = 1.0),$$
     where $k_r$ are the frozen `K3` salience keys and $\beta_r$ are the frozen `theta_trust` precisions.
   - Fit $(u, a, b)$ on the mixed-tier D3 calibration probes (by MSE to the calibration reference $\text{ref}$, just as in `R8-E1`).
   - Also train `jacobi_lin_T4_prec` (`814` params) with a combined loss on the calibration `P-FullSupport` bags + $0.25 \times$ the D3 controlled-bag MSE loss (all against the model-only calibration reference $\text{ref}$) so its `P-FullSupport` KL and D3 weight law are jointly calibrated.

### 1.2 Pre-Registered Decision Rules (`R9-E1`)

- **`E1-D0` (Checkpoint & $K=0$ Identity Gate — PASS required):**
  1. All 3 seeds load `li-net/r7e1/ckpt/s<seed>/theta_real_v2.npz` matching its SHA256 hash; clause-1 Pearson with exact Bayes $\ge 0.999$.
  2. At $K=0$, `looped_attn_T3` and `attn_L3-K3` return $\ell^{(0)}(M_0)$ identically ($\max |\Delta \mathrm{KL}| < 10^{-7}$) at both $M_0 = 2$ and $M_0^{\text{strong}}$.

- **`E1-D1` (Sparse-Root & Genuine Strong-Root Anytime Monotonicity — PASS if both `D1.1` and `D1.2` hold):**
  - **`E1-D1.1` (Sparse-Root Anytime Curve, $M_0 = 2$, $K \in \{0, 1, 2, 4, 8, 16, 24, 32\}$):**
    1. Across $K \in \{0, 1, 2, 4, 8, 16, 24, 32\}$ on the interleaved `P-FullSupport` stream, `looped_attn_T3` satisfies $\mathrm{KL}(K_{m+1}) \le \mathrm{KL}(K_m) + 0.0005$ at every step in every seed, reaches $\mathrm{KL}(32) \le 0.033$, and achieves $\mathrm{JSD}(24) \le 0.0075$.
  - **`E1-D1.2` (Genuine Strong-Root Safety at $M_0^{\text{strong}} \in \{30, 32, 34\}$, $K \in \{0, 1, 2, 4, 8, 16, 24\}$):**
    1. **Strong-root regime verified:** At $M_0^{\text{strong}}$, the $K=0$ root belief has 3-seed mean $\mathrm{KL}(0) \le 0.050$ nats.
    2. **Naive pooling degrades the strong root:** Under Bounded-Context Local Patch Reads (Mode B, or Mode A if Mode A also degrades), both `mix` and standard-root `ataraxos_mmd` degrade the strong root at $K=16$ by $\mathrm{KL}(16) - \mathrm{KL}(0) \ge +0.015$ nats (with 95% CI lower bound $> 0$).
    3. **Anchor-aware `looped_attn_T3` is strictly safe at every $K$:** Across all $K \in \{1, 2, 4, 8, 16, 24\}$ in both Mode A and Mode B, `looped_attn_T3` never degrades the strong root in any seed ($\max_{K \ge 1}[\mathrm{KL}(K) - \mathrm{KL}(0)] \le +0.0005$ in every seed), strictly improves $\mathrm{KL}(16) < \mathrm{KL}(0)$ (95% CI of $\mathrm{KL}(0) - \mathrm{KL}(16)$ above $0$), and beats `ataraxos_mmd` and `scaled_mix` at $K=16$ and $K=24$ with 95% CIs $> 0$.

- **`E1-D2` (1-Step `econ_prec_K3` & 4-Step `jacobi_lin_T4_prec` Three-Regime Weight Law — PASS if both `D2.1` and `D2.2` hold):**
  - **`E1-D2.1` (`econ_prec_K3` 3-Scalar Three-Regime Law on D3 Controlled Bags):**
    1. `econ_prec_K3` (`3` scalar params $u, a, b$) achieves clean deduplication $W(16)/W(1) \le 1.25$ at $\tau=0.3$ (with 95% CI upper bound $\le 1.30$, vs. `R8-E1`'s `1.417` for tier-independent `econ_kc_K3`),
    2. Strict tier monotonicity $W_{\tau=0.3}(8)/W_{\tau=0.3}(1) < W_{\tau=1.5}(8)/W_{\tau=1.5}(1) < W_{\tau=2.5}(8)/W_{\tau=2.5}(1)$ in every seed,
    3. Mutual support $W(16)/W(1) \ge 2.20$ at $\tau=2.5$, and
    4. Lower D3 all-tier MSE than both `kc_K3` and 2-scalar `econ_kc_K3` (paired 95% CIs $> 0$).
  - **`E1-D2.2` (`jacobi_lin_T4_prec` vs. 1-Step Pools on Partial-Overlap `P-FullSupport`):**
    1. `jacobi_lin_T4_prec` (`814` params) satisfies all three weight-law clauses ($W(16)/W(1) \le 1.25$ at $\tau=0.3$, strict $N=8$ tier monotonicity in every seed, $W(16)/W(1) \ge 2.20$ at $\tau=2.5$),
    2. Achieves `P-FullSupport` ($K=24$) $\mathrm{KL} \le 0.0420$, and
    3. Beats `looped_lin_T4`, `kc_K1`, `kc_K3`, and 1-step `econ_prec_K3` on `P-FullSupport` $\mathrm{KL}$ (all 4 paired 95% CIs $> 0$), demonstrating the exact gap between 1-step diagonal crowding (`econ_prec_K3`) and multi-step normal-equation inversion (`jacobi_lin_T4_prec` / `looped_attn_T3`) under partial overlap $0 < \rho < 1$.

---

## 2. Experiment `R9-E2`: Hybrid Greedy-Lead + Full-Budget Beam (`P8_hybrid`) & Self-Consistent Unresolved Safety on `Sudoku-Extreme` (`E-SUD-HYBRID-SAFE`)

- **Executor:** `calibrated-ai`
- **Target Path:** `/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/R9-E2/`
- **Estimated Cost:** $\sim\$0.45$ (hard cap **$\$0.75$**, 1 L4 GPU, running **1 seed process at a time** on the GPU and reusing frozen `R4-E2` checkpoints and `R7-E2`/`R8-E2` baselines).

### 2.1 Exact Architectural Additions over `R8-E2` (`esud9c.py` -> `esud9d.py`)

1. **Single-Process GPU Execution & Shared Wave-4 Landmarks:**
   - Run one seed at a time on the L4 GPU (eliminating the 3-process GPU contention from `R8-E2`).
   - Reuse frozen `P0`, `P3`, `P6`, and `R7-E2`'s `P8_beam` test outputs (with the `P0` bitwise integrity check), and reuse `R8-E2`'s Phase-A Wave-4 Landmarks (`L = 4`, `tau_dead = 0`) so all Phase-B completion arms start from the **exact same surviving Wave-4 Landmarks** on every board and spend the exact same remaining budget $B_{\text{rem}} = 16(E_b + 1) - \text{passes}_{\text{Phase A}}$.

2. **The $2\times 2$ Phase-B Completion Factorial (`D1`):**
   - Compare four Phase-B completion strategies at budget $16(E_b + 1)$:
     1. **`P8_beam` (`R7-E2` reference):** Starved allocation (`70%` budget) + `P3_thresh` completion.
     2. **`P8_full_stoch` (`R8-E2` baseline):** Full-budget D'Hondt allocation (`99.5%` spend) + unrestricted stochastic `conf` chains ($d \sim \text{multinomial}(q_{z^*})$ at every empty cell).
     3. **`P8_full_lead` (Full budget + Greedy lead chain $r=0$, unrestricted `conf` for $r \ge 1$):** Pass 1 of Phase B runs **1 deterministic greedy chain (`argmax` at every step)** on each surviving Landmark in rank order (as long as $E(s_j)+1$ fits in $B_{\text{rem}}$); Passes $2+$ run stochastic `conf` chains via D'Hondt until $B_{\text{rem}}$ is exhausted.
     4. **`P8_hybrid` (Full budget + Greedy lead chain $r=0$ + Forced-step `argmax` $H < 0.25$ & $q > 0.05$ stall sampling for $r \ge 1$):** Pass 1 of Phase B runs **1 deterministic greedy chain (`r=0`, `argmax` at every step)** on each surviving Landmark in rank order; Passes $2+$ allocate remaining chains via D'Hondt (`norm = logP / max(1, depth)`, $\tau_{\text{alloc}} = 2.0$ fixed across seeds so no tuning noise is introduced) and execute each $r \ge 1$ chain using **`P3_thresh` stall sampling**:
        - At any step where $\min_{c \in E} H(q_c) < 0.25$ (a forced/near-deterministic single), commit $\arg\max_d q_{z^*, d}$ deterministically (preventing cumulative clerical slips across the $\sim 28$ forced steps remaining after Wave 4!).
        - At any stall step where $\min_{c \in E} H(q_c) \ge 0.25$, sample $d \sim \text{normalize}(q_{z^*, d}^{1/\tau_{\text{stall}}} \cdot \mathbf{1}[q_{z^*, d} > 0.05])$ on the Gumbel-selected cell $z^*$, with $\tau_{\text{stall}} = 1.0$ (standard `P3_thresh`).
     - Also evaluate `P8_hybrid` along the full anytime budget curve $m \in \{0, 1, 2, 4, 8, 16, 32\}$ (using single-run prefix extraction from a single $m=32$ run where the first $m(E_b+1)$ passes match the $m$-budget schedule, or independent runs verified to match).

3. **Zero-Extra-Pass Self-Consistent Leaf Check ($\widetilde{\text{chk}}_r$) for Unresolved-Board Safety (`D2`):**
   - In `R8-E2`, the final forward pass `fwd(model, Z^{(r)})` on a completed leaf board $Z^{(r)}$ *already* returns both the global `Solvable` probability $\text{chk}_r \in (0, 1)$ **and** the model's own per-cell digit distributions $q_{\text{leaf}}(c, d)$ across all 81 cells in that **same single pass** (0 extra neural net passes, 0 `valid9` or oracle calls).
   - On a valid completion, the model's own leaf pass agrees with the placed digits across all empty cells ($\min_{c \in E_b} q_{\text{leaf}}(c, Z^{(r)}_c)$ is high), whereas on seed 103's false accepts (where a row/col/box clash occurred during completion), the leaf pass's own cell logits strongly disagree with the clashing digit ($\min_{c \in E_b} q_{\text{leaf}}(c, Z^{(r)}_c)$ is small, which achieved AUROC `0.94` for board-level clash detection in `R6-E2`).
   - Define the **Self-Consistent Leaf Check**:
     $$\widetilde{\text{chk}}_r \;=\; \text{chk}_r \;\times\; \Big(\min_{c \in E_b} q_{\text{leaf}}\big(c, Z^{(r)}_c\big)\Big)^{\eta_{\text{sc}}},$$
     and gate leaf weights with an acceptance floor $\tau_{\text{chk}}$:
     $$w_r \;=\; \widetilde{\text{chk}}_r^{\,\gamma} \cdot \mathbf{1}\big[\widetilde{\text{chk}}_r \ge \tau_{\text{chk}}\big].$$
   - When $\sum_r w_r = 0$ (no leaf passes the self-consistent check $\widetilde{\text{chk}}_r \ge \tau_{\text{chk}}$), both `chk_mix_sc` and `vote_chk_sc` fall back cleanly to the one-pass anchor `P0` (yielding $\mathrm{KL} - \mathrm{KL}(\text{P0}) = 0.0000$ on that board!).
   - **Calibration of $(\eta_{\text{sc}}, \tau_{\text{chk}}, \gamma, \lambda_{\text{mix}})$:** Fit $(\eta_{\text{sc}} \in \{0, 0.5, 1.0, 2.0\}, \tau_{\text{chk}} \in \{0.5, 0.75, 0.85, 0.90\}, \gamma \in \{2, 4, 8, 16\})$ on `val` boards to maximize `val` empty-cell log-likelihood (or without any labels: choose the smallest $(\eta_{\text{sc}}, \tau_{\text{chk}})$ on `cal` boards where the model's own leaf self-consistency $\min_{c \in E_b} q_{\text{leaf}}(c, Z_c) \ge 0.50$ holds on $\ge 98\%$ of accepted leaves!). Note: to comply strictly with the no-oracle-in-selection rule, select $(\eta_{\text{sc}}, \tau_{\text{chk}})$ using the **label-free self-consistency criterion on `val`** (require every accepted leaf to have $\text{chk}_r \ge 0.80$ and $\min_{c \in E_b} q_{\text{leaf}}(c, Z^{(r)}_c) \ge 0.50$, i.e. $\mathbf{1}[\text{chk}_r \ge 0.80 \wedge \min_{c \in E_b} q_{\text{leaf}}(c, Z^{(r)}_c) \ge 0.50]$), and report both the fixed label-free rule `(chk >= 0.80) & (min_c q_leaf >= 0.50)` and the `cal`-tuned $(\eta_{\text{sc}}, \tau_{\text{chk}}, \gamma)$.

### 2.2 Pre-Registered Decision Rules (`R9-E2`)

- **`E2-D1` (Full-Budget Hybrid Beam Completion `P8_hybrid` at $B = 16(E_b+1)$ — PASS if `D1.0`, `D1.1`, and `D1.2` hold):**
  - **`E2-D1.0` (Budget & Mechanism Isolation):**
    1. `P8_hybrid` never exceeds $16(E_b+1)$ passes on any board and spends $\ge 92\%$ of $16(E_b+1)$ on average in every seed.
    2. Report the full $2\times 2$ witness & KL table comparing `P8_beam` (`R7`), `P8_full_stoch` (`R8`), `P8_full_lead`, and `P8_hybrid`.
  - **`E2-D1.1` (Witness Rate Gain):**
    1. On the 500 test boards per seed, `P8_hybrid` achieves a higher true-witness discovery rate than `P3` (by $\ge +4.5\text{ pp}$, paired 95% CI $> 0$), `P6` (by $\ge +5.5\text{ pp}$, paired 95% CI $> 0$), and `P8_full_stoch` (paired 95% CI $> 0$), and is at least as high as `R7-E2`'s starved `P8_beam` ($\text{witness}(\text{P8\_hybrid}) \ge \text{witness}(\text{P8\_beam})$ in 3-seed mean).
  - **`E2-D1.2` (Empty-Cell KL Gain):**
    1. Under self-consistent check-gated pooling (`chk_mix_sc` and `vote_chk_sc`), `P8_hybrid` reduces empty-cell KL vs. `P0` by $\ge 40\%$ and beats `P3`, `P6`, and `P8_full_stoch` with paired 95% CIs $> 0$.

- **`E2-D2` (Strict Anytime Monotonicity & All-Seed Unresolved-Board Safety — PASS if both `D2.1` and `D2.2` hold):**
  - **`E2-D2.1` (Strict Anytime Monotonicity across $m \in \{0, 1, 2, 4, 8, 16, 32\}$):**
    1. Under `chk_mix_sc` and `vote_chk_sc`, test empty-cell KL is **strictly decreasing** at every budget step $m \in \{0, 1, 2, 4, 8, 16, 32\}$ in every seed ($36/36$ step transitions).
  - **`E2-D2.2` (All-Seed Unresolved-Board Safety):**
    1. On test boards where `P8_hybrid` at $m=16$ finds zero true witnesses, self-consistent check gating (`chk_mix_sc` and `vote_chk_sc` with the label-free self-consistency gate $\text{chk}_r \ge 0.80 \wedge \min_{c \in E_b} q_{\text{leaf}}(c, Z^{(r)}_c) \ge 0.50$) reduces seed 103's false-accept rate to $\le 1.0\%$ (from `7.0%` in `R8-E2`) and satisfies
       $$\mathrm{KL}(m=16) - \mathrm{KL}(\text{P0}) \;\le\; +0.005 \text{ nats in every seed (and 95\% CI upper bound } \le +0.005\text{)},$$
       while unweighted `mix` degrades unresolved boards by $> +0.10$ nats.

---

## 3. Summary of Round 9 Resource Allocation

| Experiment | Executor | World-Model Training | GPU Hardware | Est. Cost | Hard Ceiling | Primary Question Closed |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`R9-E1` (`E-INFO-STRONG-ROOT`)** | `eurisko` | **0 s** (loads `R7-E1` `theta_real_v2.npz`) | 1x L4 | `~$0.35` | **`$0.60`** | Genuine strong-root ($M_0 \in \{30,32,34\}, \mathrm{KL}(0) \le 0.050$) anytime safety + 3-scalar `econ_prec_K3` vs. `jacobi_lin_T4_prec` |
| **`R9-E2` (`E-SUD-HYBRID-SAFE`)** | `calibrated-ai` | **0 s** (loads `R4-E2` checkpoints) | 1x L4 (1 seed/proc) | `~$0.45` | **`$0.75`** | Full-budget `P8_hybrid` (greedy lead + forced-cell `argmax` + stall sampling) + zero-extra-pass self-consistent leaf check ($\min_c q_{\text{leaf}}$) for all-seed unresolved safety |
| **Total Round 9** | | **0 s** | | **`~$0.80`** | **`$1.35`** | |
