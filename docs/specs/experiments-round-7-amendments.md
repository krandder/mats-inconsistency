# Round 7 Pre-Registration Amendments (`R7-E2` Readings 4–7 & `R7-E1` Reading 9)

**Date:** 2026-10-17T09:24Z
**Author:** Antigravity (Gemini)
**Target:**
- `R7-E2` (`E-SUD-BEAM`, `calibrated-ai`): Readings 4–7 (all other readings 1–3 and 8–14 approved as stated).
- `R7-E1` (`E-INFO-V4`, `eurisko`): Reading 9 (all other readings 1–8 and 10 approved as stated).

---

## Part 1: `R7-E2` — Why Readings 4 & 5 Need a Crucial Fix Before Launch

Tracing Readings 4 & 5 step-by-step reveals two mechanical traps that would cause `P8_beam` to cut off its own rollouts with zero leaves:

1. **Cross-Wave Queue Depth Trap (Reading 4):**
   Because $\log q_{z^*, d} < 0$ at every stall, cumulative $\log P_{\text{stall}} = \sum_{j=1}^\ell \log q_j$ decreases monotonically with stall depth $\ell$. If a single priority queue $\mathcal{Q}$ is shared across waves and ordered by raw $\log P_{\text{stall}}$, then as soon as wave $\ell$ pops 16 nodes and advances them to stall $\ell+1$, their children (summing $\ell+1$ negative log-probabilities) have a **more negative** $\log P_{\text{stall}}$ than the rejected nodes left behind in $\mathcal{Q}$ at depth $\ell$. In the next wave, the queue abandons the deeper beam to pop the rejected shallower nodes, expanding breadth-first across all depths without completing any leaf until the reserve step triggers.
2. **Lockstep Truncation Trap (`E_min` vs. `E_max` in Reading 5 & Hard Cap in Reading 2):**
   In Reading 5, the reserve step triggers when $m = \lfloor (\text{budget} - \text{passes}) / (E_{\min} + 1) \rfloor < 16$ (i.e., $m = 15$, when $\text{budget} - \text{passes} \approx 15(E_{\min} + 1)$). Because different nodes in the popped set have different numbers of empty cells $E_{s_i} \ge E_{\min}$, running $m = 15$ nodes in lockstep costs $\sum_{i=1}^{15} (E_{s_i} + 1) > 15(E_{\min} + 1) \ge \text{budget} - \text{passes}$. Under Reading 2 (*"budget $16(E_b + 1)$ is a hard cap and an arm row that would exceed it is cut with no leaf"*), the lockstep rollout runs out of budget $E_{\max} - E_{\min}$ steps before completion and **cuts off almost all 15 nodes with zero leaves**.

### Amended Readings 4–7 for `R7-E2` (`P8_beam` & `P3_thresh`)

Replace Readings 4–7 with the following exact multi-level generalization of `P6` (`esud9_net.py`'s `prefix` + `alloc`), which is 100% immune to both traps and guarantees **zero truncated leaves**:

1. **Fixed $L$-Level Synchronous Landmark Expansion (`L_waves` $\in \{2, 3, 4\}$ chosen per seed on `val` by `vote_chk` outcome CE; note `P6` is $L_{\text{waves}} = 1$):**
   - Start at wave $\ell = 1$ with frontier $\mathcal{F}_0 = \{(\text{bd}, \log P_{\text{stall}} = 0)\}$ (1 node per board) and `passes_used = 0`.
   - For each wave $\ell = 1, \dots, L_{\text{waves}}$:
     - **Shared-Prefix Forced Propagation (`prefix`):** Every active node $s \in \mathcal{F}_{\ell-1}$ unmasks its lowest-entropy empty cell with `argmax(q_z)` while $\min_{c \text{ empty}} H(q_c) < \tau_{\text{stall}}$ (`0.25` nats), incrementing `passes_used` by 1 per forward pass.
       - *Dead-end check (Reading 6):* During each pass, if $\min_{c \text{ empty}} \max_d q_c(d) < \tau_{\text{dead}}$ ($\tau_{\text{dead}} \in \{0.0, 0.12, 0.15\}$ chosen on `val`), mark node $s$ as dead and drop it from the frontier (unless it is the last surviving node for that board).
     - **Level-Synchronous Stall Branching:** Each surviving node $s$ that stalls ($\min_{c \text{ empty}} H(q_c) \ge 0.25$) picks $z^* = \arg\min_{c \text{ empty}} (H(q_c) - \delta_U \mathbf{1}[c \in \text{band}(U)])$ (`1` pass already counted in `prefix`) and expands without replacement into child Landmarks $s_d = s \cup \{Z_{z^*} = d\}$ for all $d$ with $q_{z^*, d} > 0.05$ (`BRANCH_MIN_Q`; or $\arg\max q_{z^*}$ if none), with $\log P_{\text{stall}}(s_d) = \log P_{\text{stall}}(s) + \log q_{z^*, d}$ and `rb[z*] = one_hot(d)`. (Any node $s$ that already reached $0$ empty cells is carried forward unchanged with its $\log P_{\text{stall}}(s)$.)
     - **Strict Level-Synchronous Beam Pruning (No Cross-Wave Queue):** Form $\mathcal{F}_\ell$ by keeping only the **top $W = 16$ children of wave $\ell$** by $\log P_{\text{stall}}$ (discarding any overflow children beyond the top $16$ so shallower rejected nodes never carry over to preempt deeper nodes).

2. **Budget-Safe Completion Allocation (`E_max` Denominator — Zero Cutoff Guarantee):**
   - After wave $L_{\text{waves}}$, let $\mathcal{F}_{L_{\text{waves}}} = \{s_1, \dots, s_M\}$ ($1 \le M \le 16$) be the surviving Landmark states for board $b$, and let:
     $$E_{\max, b} = \max_{1 \le j \le M} (\text{empty cells in } s_j).$$
   - Compute the number of completion chains $R_b$ using **$E_{\max, b} + 1$** (not $E_{\min, b} + 1$!):
     $$R_b = \min\!\Big(64, \; \max\!\Big(1, \; \left\lfloor \frac{16(E_b + 1) - \text{passes\_used}_b}{E_{\max, b} + 1} \right\rfloor\Big)\Big).$$
     Because every Landmark $s_j \in \mathcal{F}_{L_{\text{waves}}}$ has $E(s_j) + 1 \le E_{\max, b} + 1$, running $R_b$ completion chains from $\mathcal{F}_{L_{\text{waves}}}$ (plus 1 check pass per completed chain) is **mathematically guaranteed to use $\le 16(E_b + 1)$ total passes**—so **no chain is ever cut off mid-rollout**.
   - Allocate the $R_b$ chains across the top $\min(M, R_b)$ Landmarks in $\mathcal{F}_{L_{\text{waves}}}$ using `alloc(P_stall, R_b)` (exactly as in `P6`, `esud9_net.py` line 253: 1 chain per Landmark without replacement, remainder proportional to $P_{\text{stall}}(s_j) = \exp(\log P_{\text{stall}}(s_j))$).

3. **Completion Rollouts (`P3_thresh` with Greedy Lead Chain):**
   - Each allocated chain from a Landmark $s_j$ (and each of the 16 chains from the root in the flat `P3_thresh` arm) runs to completion as follows:
     - At every step, pick $z_t = \arg\min_{c \text{ empty}} (H(q_c) - \delta_U \mathbf{1}[c \in \text{band}(U)] + \epsilon_c)$.
     - If $H(q_{z_t}) < 0.25$ **or** if this chain is the first chain ($r = 0$) allocated to its starting state, pick $d = \arg\max_d q_{z_t, d}$ deterministically.
     - Otherwise ($H(q_{z_t}) \ge 0.25$ and $r \ge 1$), sample $d \sim \mathrm{Categorical}(\tilde{q}_{z_t})$ where $\tilde{q}_{z_t, d} \propto q_{z_t, d} \cdot \mathbf{1}[q_{z_t, d} > 0.05]$.
   - Every completed chain runs its 1 own-check pass $\text{chk}_r = \sigma(\text{sl}(Z^{(r)}, \text{Solvable}=\text{MASK}))$.

---

## Part 2: `R7-E1` — Reading 9 Fix (Do Not Square `pref` in Waves 2–4)

In `R7-E1` Reading 9, `score` in Wave 1 ($w = 0$) is:
$$\text{score}_0(p) = \underbrace{(1 + \lambda_U \mathbf{1}[p \in \text{half}(U)])}_{\text{pref}(p) \in \{1, 4\}} \cdot \mathrm{Var}_\mu(p).$$
If Waves 2–4 ($w \in \{1, 2, 3\}$) literally square the entire `score` (`score ** 2`), that squares `pref(p)` from $4\times$ to **$16\times$** in Waves 2–4! A $16\times$ multiplier for the preferred half in Waves 2–4 overpowers the crowding penalty $(1 + \text{crowd})^{1.5}$ on lightly visited preferred-half patches (`16 / 2.5^3 = 1.02 >= 1`), keeping $\sim 75\%$ of reads in the preferred half even in Wave 4 and **reducing** opposite-half coverage below $95\%$.

**Amended Reading 9 for `R7-E1`:**
- In Wave 1 ($w = 0$): `score = pref * var`, sampled proportional to `score` without replacement (identical to `R6-E1`, guaranteeing `wave1_pref_share` $\approx 0.82 \ge 0.75$).
- In Waves $w \ge 1$ ($w \in \{1, 2, 3\}$): sharpen **only the novelty-weighted variance term** (and decay the initial preference nudge as exploration proceeds, $\text{pref}_w = 1 + (\lambda_U / (1 + w)) \mathbf{1}[p \in \text{half}(U)]$, or keep $\text{pref}$ unsquared):
  $$\text{score}_w(p) = \Big(1 + \frac{\lambda_U}{1 + w}\mathbf{1}[p \in \text{half}(U)]\Big) \cdot \left(\frac{\mathrm{Var}_\mu(p)}{(1 + \text{crowd}(p))^{\alpha_{\text{nov}}}}\right)^2,$$
  sampled in proportion to $\text{score}_w(p)$ without replacement (and where prospective candidate keys for `K3` use the imputed prospective read's belief-shift salience $s_p = |\Delta\ell_p| / \|\Delta\ell_p\|_2$, or `KeyNet("K3")` applied to $s_p$). All other readings 1–8 and 10 of `R7-E1` are approved as stated.

