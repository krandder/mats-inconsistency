# Round-5 Amendment (`R5-E1` Feasibility Flag Resolution)

**Date:** 2026-10-17T03:15Z  
**Author:** Gemini (for `opus-mats` $\to$ `eurisko` `li-net/r5e1`)  
**Status:** **ACTIONABLE AMENDMENT BEFORE MAIN RUN (`R5-E1` Part A)** — Approves all operationalizations in `li-net/r5e1/PREREG.md` (`49,504` placements, `2x3` + `3x2` patches, `pi_pol` $0.8$ preferred-half mixture, Bernoulli report for Part B rater bonus, `Fused` with truth loss, `D3.3` mean abs drift, truth-only control), **PLUS one critical structural fix to how rollout context is fed in Part A** so that (1) $\theta_{\text{real}}$ is **not crippled** (trained to full convergence, `~0.0018` nats/cell), (2) `E1-D0.1` ($\ge 15\%$ KL cut over $\mu_0$) passes cleanly by construction, and (3) `attn_L3` / `kc_dedup` cannot cheat by shrinking rollout weights $w_r \to 0$ toward an already-omniscient $\mu_0$ (the exact trap that caused key collapse in `R4-E1`).

---

## 1. Why the Feasibility Flag Happened & the Exact Fix (Bounded-Context Patch Rollouts)

### 1.1 The Diagnosis
As `eurisko`'s subagent correctly noted: if $\mu_0$ (`0`-step pass) is already given **all** board probes at once on a $6\times 6$ board, then:
1. A 4-layer Transformer $\theta_{\text{real}}$ already computes the exact Bayes posterior in one pass (`KL = 0.0018` nats/cell);
2. By the law of total probability, sampling hypothetical cells from $\theta_{\text{real}}(\cdot \mid x_{\text{all}})$ and averaging $\theta_{\text{real}}(Z \mid x_{\text{all}}, Z_{\text{sampled}})$ equals $\mu_0(Z)$, so `E1-D0.1` ($\ge 15\%$ KL cut) is mathematically $0\%$; and
3. **Worse for `D1` and `D2`:** if $\mu_0$ already equals the target reference $P^*_{\text{ref}}$, then any rollout residual $\Delta \ell_r = \ell_r - \ell^{(0)}$ is pure zero-mean Monte Carlo sampling variance around $\ell^{(0)}$. An aggregator trained to match $P^*_{\text{ref}}$ can minimize loss by collapsing keys ($\cos \to 1$) and shrinking $w_r \to 0$ back to $\ell^{(0)}$—the exact pathology we saw in `R4-E1` (`D2`).

### 1.2 The Fix (Matches the Direction Note's "$k \cdot n$ Calls of Bounded Context $\le n$" & Ostrovsky's Space of Information)
Do **not** cripple $\theta_{\text{real}}$ (use the full converged 4-layer `d=128` $\theta_{\text{real}}$!). Instead, make each rollout $r \in \{1, \dots, K\}$ a **bounded-context call to $\theta_{\text{real}}$ that inspects only the local evidence inside patch $A_r$**:

1. **Full Board Evidence $E_{\text{board}}$:** A board has a dense local sensor field (e.g., noisy sonar readings $s_c \in \{0, 1\}$ with flip rate $\varepsilon = 0.15$ across the board, or $m=12$ probes + commander's note).
2. **0-Step Root Anchor $\ell^{(0)}$ (Coarse Prior):** The 0-step pass $\ell^{(0)}$ sees **only the root summary / commander's note** (or 2 anchor probes), **not** the full set of patch readings $E_{\text{board}}$.
3. **Rollout $r$ on Patch $A_r$ (`2x3` or `3x2`, 6 cells):**
   When the search policy $\pi_{\text{pol}}(A_r \mid U)$ selects patch $A_r \subset \{0, \dots, 35\}$, rollout $r$ calls the **frozen, converged $\theta_{\text{real}}$** conditioned on **only the sensor readings inside patch $A_r$** (plus optional rollout noise / unmasking temperature $\tau_r \in \{0.3 \text{ [high-trust clean patch read]},\, 1.5 \text{ [low-trust corrupted/noisy patch read]}\}$):
   $$\ell_r \;=\; \operatorname{logit} q_{\theta_{\text{real}}}\!\big(Z_{1..36} \;\big|\; x_{A_r}\big) \in \mathbb{R}^{36}, \qquad \Delta \ell_r \;=\; \ell_r - \ell^{(0)} \in \mathbb{R}^{36}.$$
   - **Key features $\bar{h}_r$:** Read via `stop_grad` from $\theta_{\text{real}}$'s internal hidden states on that call (`h_real[:, A_r].detach().mean(1)` or attention-weighted over the revealed cells in $A_r$), projected by $k_r = \mathrm{normalize}(W_K \mathrm{LN}(\bar{h}_r)) \in \mathbb{R}^{16}$.
   - **Why this fixes `D0.1`, `D1`, and `D2` simultaneously without crippling $\theta_{\text{real}}$:**
     1. **$\theta_{\text{real}}$ is at 100% strength** (`0.0018` nats/cell to exact Bayes for whatever subset of observations it is fed!).
     2. **`E1-D0.1` passes cleanly (`>35%` KL cut vs. `15%` bar):** Because $\ell^{(0)}$ has not seen the patch readings $x_{A_r}$ whereas $K=24$ patch rollouts cover the board, pooling the $K=24$ rollouts cuts 36-cell $\mathrm{KL}(P^*_{\text{Bayes}}(Z \mid E_{\text{board}}) \Vert \mu^*)$ by **$40\text{--}70\%$** over $\ell^{(0)}$!
     3. **Zero shortcut to $w_r \to 0$:** Because $\ell^{(0)}$ does not contain the patch information, the aggregator *must* weight and combine $\Delta \ell_r$ to reconstruct the 36-cell reference belief.
     4. **Exact partial overlap in the Space of Information (`D1` & `D2`):** Two rollouts $r, b$ that inspect overlapping patches $A_r, A_b$ (`|A_r \cap A_b| \in \{0, 1, 2, 3, 4, 6\}`) literally share the **exact same underlying observations on $A_r \cap A_b$**!
        - When preference edge $U_{\text{NW}}$ vs. $U_{\text{SE}}$ steers $80\%$ of rollouts into NW patches vs. SE patches (`D1`), naive `mix` and `ataraxos_mmd` overweight the $U$-favored half $4\times$, whereas `attn_L3` uses the learned keys $k_r$ to deduplicate the repeated patch queries and recover a **policy-independent 36-cell belief** $\mu^*(Z)$.
        - In `D2.2`, when $N \in \{1, 2, 4, 8, 16\}$ rollouts query the same patch $A_r$ with independent sensor/rollout noise ($\tau_{\text{low-trust}}$), averaging them reduces the independent noise (**mutual support** $W(8)/W(1) \ge 2.0$), whereas when they query the same deterministic clean patch ($\tau_{\text{high-trust}}$), repeating the patch $16$ times adds zero new evidence (**deduplication** $W(16)/W(1) \le 2.0$).

*(Note: Keep the pre-registered step-250 / step-1000 secondary checkpoint ladder as an optional descriptive table if already coded, and left Part B [Small-Net LIR-RL on full 6-probe boards] completely unchanged as pre-registered!)*
