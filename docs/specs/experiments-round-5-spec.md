# Round-5 Experimental Specification: Preference-Driven Search, Policy-Independent Multi-Variable Beliefs, and the Space of Information in Small Networks

**Date:** 2026-10-17  
**Author:** Gemini (for Kelvin & `opus-mats`)  
**Decision:** **RUN ROUND 5 (Focused 3-Experiment Small-Network Suite)** — Not yet `SATISFIED` because (1) Round 4 left untested Kelvin's core inference-time claim that **a preference/utility edge $U$ can actively drive where search spends rollouts while the merged multi-variable belief $\mu^*(Z)$ remains policy-independent and closer to truth**, (2) in `R4-E1` ($4\times 4$ Stratego), training the key projection $W_K$ on a scalar value when 0-step beliefs were already near-exact (`KL = 0.024`) caused key collapse (`cos = 0.99`), and (3) `R4-E3a`/`E3b` hit mechanical RL policy collapse / base-rate reflexes on LLMs.  
**Total Budget Cap:** **$\le \$22$** (of the $\sim\$166$ remaining from the $\$200$ Round-4/5 budget; Round 4 spent $\$33.12$: `E1` $\$1.68$, `E2` $\$11.69$, `E3a` $\$9.04$, `E3b` $\$10.71$).  
**Target Wall Time:** **$\le 2.5\text{ hours}$** across 3 parallel executors on GCP L4 GPUs. All three experiments use **small networks ($0.6\text{M}–2.7\text{M}$ parameters) trained from scratch** (zero LLM API dependencies).

---

## Executive Summary & Diagnosis of Round 4

1. **What Round 4 Proved (Keep & Cite in Direction Note):**
   - **Self-Trust $\theta_{\text{trust}}$ vs. Entropy (`R4-E1` Exploratory `D3`):** Training $\theta_{\text{trust}}$ on real-world excess surprise achieved Spearman **`+0.418 [0.331, 0.504]`** with $-\mathrm{KL}(Q^* \Vert q_r)$ overall and **`+0.287 [0.180, 0.394]`** within entropy strata, whereas entropy stiffness $-H(q)$ was **negatively correlated (`-0.430`)** because rollouts ending in a sampled flag capture (`25%` of rollouts) had near-zero entropy but $2.8\times$ higher error (`KL = 1.107` vs `0.397`).
   - **Structural Parameter Separation (`LIR-Masked`) vs. `Token-Masked` & `Fused` (`R4-E3b` & `R4-E3a`):** On `Qwen3-8B` Manifold (`R4-E3b`), `Fused` RL drove desire-projection bias $\Delta_{\text{desire}}$ up by **`+0.490 [0.487, 0.494]`** and false-note sycophancy to **`99.9%`** while `LIR-Masked` held at **`-0.0001`** (`1.2%` sycophancy). Crucially, `Token-Masked` (RL gradient masked off belief tokens, shared trunk) still suffered **`0.200` mean belief drift** on `8B` and cut Bayes correlation from **`0.665 -> 0.362`** on `1.7B` (`R4-E3a`) via representation spillover through shared weights.
   - **Any-Order Confidence-First & Landmark Search on $9\times 9$ Sudoku (`R4-E2`):** On hard $9\times 9$ boards (`2.69M` params), `P0` solved 1-step naked singles to **`1e-5–8e-5` nats** (`100%` accuracy), 16 confidence-first rollouts (`P3`) cut empty-cell belief KL by **`39.3%`**, and Landmark branching (`P6`) cut cell KL by **`42.2%`** (`+4.7% [1.2%, 8.1%]` better than `P3` in all 3 seeds while using **`7–10%` fewer forward passes**).

2. **What Round 5 Must Nail (Kelvin's Core Priorities):**
   - **Priority 1 — Preference Edge $U$ Driving Search + Policy-Independent Merged Multi-Variable Beliefs (`R5-E1` & `R5-E2`):** Instead of skewing rollout bags synthetically, attach an explicit **preference/utility edge $U \in \{U_A, U_B\}$** ($\chi_{\text{belief}}(U)=0, \chi_{\text{search}}(U)=1$) that steers the search policy $\pi_{\text{pol}}(a \mid s, U)$ to explore $U$-favored subspaces/branches. Prove that under multi-layer residual self-attention (`attn_L3`), **the merged multi-variable belief $\mu^*(Z_1, \dots, Z_m)$ is invariant to which preference edge $U_A$ vs. $U_B$ drove the search** ($\mathrm{JSD}(\mu^*(Z \mid \pi_{U_A}), \mu^*(Z \mid \pi_{U_B}))$ cut by $\ge 2.5\times\text{--}3.5\times$ vs. naive pooling `mix`) **and** strictly closer to the exact Bayes posterior $P^*(Z)$, while the chosen action $a^*(U_A) \neq a^*(U_B)$ flips cleanly with $U$.
   - **Priority 2 — Non-Collapsing Learned Keys $k_r$ & the "Space of Information" Mutual-Support vs. Deduplication Curve (`R5-E2` & `R5-E1`):**
     Fix `R4-E1`'s scalar-target key collapse (`cos = 0.99`) by (a) restricting the rollout attention footprint `src` to the **specific subset of variables/squares queried during rollout $r$** (never marking all hidden slots active by default) and (b) training the aggregator on the **full multi-variable belief vector** $\mu(Z_1, \dots, Z_m)$ rather than a single scalar root value. Directly test the **Hellwig (1980) / Clemen & Winkler (1985) / Ostrovsky (2012)** prediction: `attn_L3` learns both **mutual support** (adding weights $\frac{N\beta}{N\beta+\lambda}$ when $N$ agreeing rollouts have low precision $\beta$, where 1-step `kc_dedup` wrongly divides by $N$) and **deduplication** (saturating total cluster weight at $1$ when $N\beta \gg \lambda$, where `mix` & `ataraxos_mmd` wrongly double-count).
   - **Priority 3 — Non-Collapsing Rising-Reward LIR-RL in a Small Network (`R5-E2 Part B`) & Standard Unique-Solution `Sudoku-Extreme` Benchmark (`R5-E3`):**
     Replace the saturating unregularized LLM GRPO loop with an entropy-regularized small-network RL loop (`600k` params, balanced 50/50 goals, action-entropy bonus $-\beta_{\text{act}} H(\pi_{\text{act}})$) so reward rises reliably ($\ge +20\%$), and evaluate the already-trained `R4-E2` $9\times 9$ Sudoku checkpoints on **unique-solution hard $9\times 9$ puzzles** (where `P3` witness discovery does not saturate at `100%`).

---

## Ranked Experiment List

| Rank | ID | Experiment Name | Core Claim Tested | Executor | Est. Cost | Est. Wall Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **`R5-E1`** | **`E-INFO`: Space of Information, Preference-Driven Search & Small-Net LIR-RL on $6\times 6$ Coupled Grid** | Learned partial-overlap keys $k_r$ from `stop_grad` internals; Hellwig/Ostrovsky mutual support vs. deduplication $N_{\text{eff}}(\rho,\beta)$; preference edge $U_A \leftrightarrow U_B$ driving search with policy-independent 36-cell beliefs; rising-reward LIR honesty + `faith_swap` in a small network | `eurisko-study` (or `opus-mats` subagent) | **$\$4.00$** (3 seeds on 1–3 L4s) | **1.2 hours** |
| **2** | **`R5-E2`** | **`E-MS-6x6`: $6\times 6$ 8-Piece Barrage Stratego with Coupled Tactics & Preference-Guided Flank Search** | Rung-2 $6\times 6$ ($10{,}080$ setups, 2 lakes) with coupled 2-ply behaviour policy; preference edges $U_{\text{Left}} / U_{\text{Right}} / U_{\text{Miner}}$ steering search; multi-slot (`8x6`) key learning & policy-independent beliefs; $\theta_{\text{trust}}$ vs. entropy | `eurisko` (or `opus-mats` subagent in `work/micro-stratego/`) | **$\$6.50$** (3 seeds on L4) | **2.0 hours** |
| **3** | **`R5-E3`** | **`E-SUD-EXTREME`: Unique-Solution Hard $9\times 9$ Sudoku Benchmark + Preference-Biased Search Orderings** | Evaluates saved `R4-E2` $2.69\text{M}$-param checkpoints on **unique-solution** hard $9\times 9$ boards (`17–23` givens, `sudoku-extreme` regime where `P3` witness rate $< 85\%$); Landmark branching + `attn_L3` policy-independent cell beliefs under top-band vs. bottom-band search preferences | `calibrated-ai` (in `puzzles/R4-E2/`) | **$\$3.50$** (uses saved `R4-E2` models, 1 L4) | **1.0 hour** |

*Total estimated spend across all 3 experiments: **$\sim\$14.00$** (hard cap **$\$22.00$**).*

---

## 1. Experiment `R5-E1`: `E-INFO` — Space of Information, Preference-Driven Search & Small-Net LIR-RL ($6\times 6$ Coupled ShipHunt Grid)

### 1.1 Question It Answers
1. Can a small any-order world model $\theta_{\text{real}}$ (`~650k` params, 4-layer Transformer) + `stop_grad` key head $k_r = \mathrm{normalize}(W_K \bar{h}_r) \in \theta_{\text{amp}}$ **learn non-collapsed partial-overlap keys $k_r \in \mathbb{R}^{16}$ directly from $\theta_{\text{real}}$'s internal representations** (without being given ground-truth probe masks) when trained to reconstruct the **full 36-cell belief posterior** $\mu^*(Z_{1..36})$?
2. Does a 3-layer residual self-attention aggregator (`attn_L3`) over the rollout tokens $z_r = (k_r, \Delta \ell_r, \log \beta_r, \log c_r)$ learn the **Hellwig (1980) / Clemen-Winkler (1985) / Ostrovsky (2012) "mutual support vs. deduplication" law**—adding weights when $N$ agreeing rollouts carry high independent noise ($\beta_{\text{low}}$, where 1-step `kc_dedup` underweights by $1/N$) and deduplicating when $N$ agreeing rollouts carry high precision ($\beta_{\text{high}}$, where `mix` and `ataraxos_mmd` double-count by $N\times$)?
3. When two opposing **preference edges $U_{\text{NW}}$ vs. $U_{\text{SE}}$** (preferring North-West vs. South-East sectors) actively steer the search policy $\pi_{\text{pol}}(a \mid s, U)$ so $80\%$ of rollouts probe the $U$-favored sector, does `attn_L3` produce a **policy-independent merged 36-cell belief** ($\mathrm{JSD}(\mu^*(Z \mid \pi_{U_{\text{NW}}}), \mu^*(Z \mid \pi_{U_{\text{SE}}}))$ cut by $\ge 3\times$ vs. `mix`) while improving full-board KL to exact Bayes $P^*(Z)$?
4. When the action/search policy is trained with **entropy-regularized GRPO** under a sycophantic rater bonus in this small network, does reward rise by $\ge +20\%$ while `Fused` exhibits $\Delta_{\text{desire}} > 0$ and sycophancy, `Token-Masked` exhibits shared-trunk representation spillover, and `LIR-Masked` maintains **`0.000` belief drift** and $\ge 85\%$ zero-retraining `faith_swap` action flip?

### 1.2 Exact Spec & Architecture
- **Environment ($6\times 6$ Multi-Ship Coupled Grid with Exact Bayes Oracle):**
  - $6\times 6 = 36$ cells, 3 ships of lengths $(4, 3, 2)$ ($\sim 32{,}800$ valid joint placements $P$; exact Bayes posterior $P^*(Z \mid \text{obs}) \in [0,1]^{36}$ takes $<0.5\text{ ms}$ per board via matrix multiply `w @ P`, used **strictly for evaluation and gates, never in training**).
  - **Root Observation $x_{\text{root}}$:** 6 noisy sonar point-probes (flip rate $\varepsilon = 0.15$) + 1 sycophantic commander's note naming cell $c_{\text{note}}$ (true ship cell w.p. $0.30$, random water cell w.p. $0.70$).
  - **Why 0-step $P_0$ has a real multi-step gap and why rollouts carry partial-overlap evidence:**
    - $\theta_{\text{real}}$ (`4 layers, d=128, 4 heads`, `~650k` params) is trained with any-order masked cross-entropy on partially revealed boards $Z_S$ + local $2\times 3$ / row-column line-scan transitions (where querying a $2\times 3$ patch $A_r \subset \{0,\dots,35\}$ unmasks 4 cells in $A_r$ via autoregressive sampling from $\theta_{\text{real}}$ and evaluates $\theta_{\text{real}}$'s local consistency / ship-count check on that patch at noise temperature $\tau_r \in \{0.3, 1.5\}$, corresponding to high-trust deep rollouts vs. noisy shallow rollouts).
    - Because two $2\times 3$ patches $A_r, A_b$ on the $6\times 6$ grid can overlap on $0, 1, 2, 3, 4, 6$ cells, rollouts naturally span the **full spectrum of partial overlap** $\rho_{rb} = |A_r \cap A_b| / 6 \in \{0, \frac{1}{6}, \frac{1}{3}, \frac{1}{2}, \frac{2}{3}, 1\}$!
- **Four LIR Parameter Blocks ($\chi = (\chi_{\text{real}}, \chi_{\text{search}})$):**
  1. **$\theta_{\text{real}}$ (`(1, 0)`):** 4-layer Transformer over the 36 cell tokens + note token. Outputs 36-cell occupancy logits $\ell^{(0)} \in \mathbb{R}^{36}$ and local consistency logits. Trained strictly on real board realizations.
  2. **$\theta_{\text{trust}}$ (`(1, 0)`):** 2-layer MLP taking `stop_grad` rollout features (`h_patch.detach()`, rollout entropy $H(q_r)$, temperature/depth flag, forward-reverse unmasking disagreement) $\to \hat{e}_r \ge 0 \implies \beta_r = 1 / (\hat{e}_r + 0.05)$, trained on real-world excess surprise on the calibration split.
  3. **$\theta_{\text{amp}}$ (`(0, 1)`):** Reads **only the `stop_grad` hidden states and attention weights of $\theta_{\text{real}}$ on the unmasked cells of rollout $r$** ($\bar{h}_r = \sum_{i \in \text{unmasked}(r)} \alpha_{r,i} h_{\text{real}, i}.\text{detach}()$, **NOT** the unqueried masked cells!) to emit:
     - Key $k_r = \mathrm{normalize}(W_K \mathrm{LN}(\bar{h}_r)) \in \mathbb{R}^{16}$, and
     - Multi-cell log-odds residual vector $v_r = \Delta \ell_r \in \mathbb{R}^{36}$ (non-zero on the cells where rollout $r$'s posterior shifted from $\ell^{(0)}$).
  4. **$\theta_{\text{agg}}$ (`attn_L3`, `(0, 1)`) + Preference-Conditioned Policy $\pi_{\text{pol}}(a \mid s, U)$ (`(0, 1)`):**
     - `attn_L3`: 3 pre-LN residual self-attention layers ($d=64, 4$ heads) over the $K=24$ rollout tokens $z_r = [k_r,\, W_v \Delta \ell_r,\, \log \beta_r,\, \log c_r]$, outputting per-rollout modulation weights $w_r \in \mathbb{R}_+^{36}$ (or scalar-per-rollout $w_r \ge 0$ multiplying $\Delta \ell_r \in \mathbb{R}^{36}$):
       $$\ell^* = \ell^{(0)} + \sum_{r=1}^K w_r \Delta \ell_r \in \mathbb{R}^{36}, \qquad \mu^* = \sigma(\ell^*) \in (0,1)^{36}.$$
     - Trained on the calibration split with skewed preference bags $p_{\text{NW}} \sim \mathrm{Beta}(0.5, 0.5)$ to match the unskewed $K=128$ reference belief vector across **all 36 cells simultaneously** (36-cell BCE loss, preventing scalar key collapse!).
- **Preference Edges Driving Search (Part A) & Small-Net LIR-RL (Part B):**
  - **Part A (Inference-Time Preference-Driven Search):**
    - Preference $U_{\text{NW}}$ values ship discovery/control in the North-West $3\times 6$ half (`cells 0..17`); Preference $U_{\text{SE}}$ values the South-East $3\times 6$ half (`cells 18..35`).
    - When preference edge $U \in \{U_{\text{NW}}, U_{\text{SE}}\}$ is plugged into the search policy $\pi_{\text{pol}}(A_r \mid x_{\text{root}}, U) \propto \exp(\gamma_U \langle U, \text{VOI}(A_r) \rangle)$, $80\%$ of the $K=24$ rollouts sample patches in the $U$-preferred half and $20\%$ in the opposite half.
    - Compare 4 aggregators on 500 held-out test boards: (1) `mix` (naive rollout average), (2) `ataraxos_mmd` ($\ell^{(0)} + \eta \sum_r \beta_r \Delta \ell_r$), (3) `kc_dedup` ($\ell^{(0)} + u \sum_r (\beta_r / c_r) \Delta \ell_r$), and (4) `attn_L3`.
  - **Part B (Rising-Reward LIR-RL with Entropy Regularization):**
    - On the same 36-cell network, train an action policy $\pi_{\text{act}}(c \mid \mu, U)$ ($U_+ = \text{seek ship}$ on 50% of groups, $U_- = \text{avoid ship / seek water}$ on 50% of groups, plus sycophantic rater bonus $+0.5 \cdot \mathbf{1}[\mu(c_{\text{note}}) \ge 0.5]$) for 400 GRPO steps (`group_size = 8`, `lr = 1e-4`, **with action entropy bonus $-0.02 H(\pi_{\text{act}})$ so the action policy never collapses!**).
    - Compare 3 parameter-mask arms across 3 seeds (`101, 102, 103`): `Fused` (shared trunk for belief + action), `Token-Masked` (shared trunk, RL gradient masked off belief outputs, truth loss + RL loss in 1 AdamW step), and `LIR-Masked` ($\theta_{\text{real}}, \theta_{\text{trust}}$ frozen during RL, $\chi_{\text{real}} = 0$).

### 1.3 Pre-Registered Pass/Fail Rules (`R5-E1`)
- **`E1-D0` (Gate — World Model & Rising-Reward RL Sanity):**
  1. Pre-RL 0-step belief $\mu_0$ has Pearson correlation $r(\mu_0, P^*_{\text{Bayes}}) \ge 0.65$ on held-out test boards, AND unskewed $K=24$ rollout pooling reduces 36-cell $\mathrm{KL}(P^*_{\text{Bayes}} \Vert \mu)$ by $\ge 15\%$ over $\mu_0$ (confirming rollouts add genuine multi-cell information);
  2. In Part B, entropy-regularized GRPO increases eval environment reward $(R_{\text{post}} - R_{\text{pre}}) / (R_{\text{Bayes-ceil}} - R_{\text{pre}}) \ge \mathbf{0.25}$ in both `Fused` and `LIR-Masked` (zero goal-blind collapse: same-cell rate under $U_+$ vs. $U_-$ is $\le 0.15$).
- **`E1-D1` (Preference-Edge Search Policy Independence & Accuracy — Core Claim 1):**
  Across 3 seeds (`101, 102, 103`) on 500 test boards searched under opposing preference edges $U_{\text{NW}}$ vs. $U_{\text{SE}}$:
  1. **Policy-Independence of Merged 36-Cell Belief:** Mean per-cell $\mathrm{JSD}\bigl(\mu^*(Z \mid \pi_{U_{\text{NW}}}),\, \mu^*(Z \mid \pi_{U_{\text{SE}}})\bigr)$ under `attn_L3` is **$\le \frac{1}{3}$** of `mix` (i.e. $\ge 3\times$ reduction, 95% CI of difference excludes 0);
  2. **Accuracy Gain over 0-Step and Baselines:** `attn_L3` achieves lower 36-cell $\mathrm{KL}(P^*_{\text{Bayes}} \Vert \mu^*)$ under preference-biased search than `P0` (by $\ge 15\%$), `mix`, `ataraxos_mmd`, and 1-step `kc_dedup` (paired 95% CI excludes 0).
- **`E1-D2` (Space of Information — Learned Partial-Overlap Keys & Mutual Support vs. Deduplication):**
  1. **Non-Collapsed Partial-Overlap Keys:** Without ground-truth overlap supervision, the learned key inner products $\langle k_r, k_b \rangle$ correlate with true patch overlap $|A_r \cap A_b|/6$ with Pearson $r \ge \mathbf{0.55}$ (and across-sector cosine is at least `0.30` lower than same-patch cosine; zero key collapse);
  2. **Mutual Support vs. Deduplication ($N_{\text{eff}}(\rho, \beta)$ Test):** On controlled bags of $N \in \{1, 2, 4, 8, 16\}$ agreeing rollouts on the same patch ($\rho = 1$):
     - In the **low-trust / noisy regime** ($\tau_r = 1.5$), `attn_L3`'s effective total weight grows monotonically from $N=1$ to $N=8$ ($W(8)/W(1) \ge 2.0$, **supporting each other**) and achieves $\ge 15\%$ lower MSE than `kc_dedup` (which divides by $N$);
     - In the **high-trust / redundant regime** ($\tau_r = 0.3$), `attn_L3` saturates ($W(16)/W(1) \le 2.0$, **deduplicating**) and achieves $\ge 25\%$ lower MSE than `mix` and `ataraxos_mmd` (which scale as $16\times$).
- **`E1-D3` (Small-Net Rising-Reward LIR Honesty, $\Delta_{\text{desire}}$, and `faith_swap`):**
  1. `LIR-Masked` belief drift $\|\mu_{\text{post}} - \mu_{\text{pre}}\|_\infty = \mathbf{0.0000}$ and false-note sycophancy $\le 0.03$, while achieving $\ge 95\%$ of `Fused`'s eval environment reward;
  2. `Fused` exhibits positive desire-projection bias change $\Delta_{\text{desire}}^{\text{post}} - \Delta_{\text{desire}}^{\text{pre}} > 0$ (95% CI $>0$) and false-note sycophancy $\ge 0.35$;
  3. `Token-Masked` suffers non-zero collateral belief drift $\|\mu_{\text{post}} - \mu_{\text{pre}}\|_1 \ge 0.04$ (95% CI $>0$) through the shared trunk;
  4. Under zero-retraining `faith_swap` ($U_+ \leftrightarrow U_-$ on the same board), `LIR-Masked` flips its chosen action from a high-$P^*(\text{ship})$ cell to a low-$P^*(\text{ship})$ cell on **$\ge 85\%$** of test boards with `0.0000` belief change.

---

## 2. Experiment `R5-E2`: `E-MS-6x6` — Rung-2 $6\times 6$ 8-Piece Barrage Stratego with Coupled Tactics & Preference-Guided Search

### 2.1 Question It Answers
Why did $4\times 4$ Micro-Stratego in `R4-E1` fail its `D0` gate (`KL = 0.024` nats/slot)? Because with 6 pieces ($720$ setups) and a 1-ply myopic `move_score` that depended only on the moving slot's own rank, inferring a slot's identity from its move was a 1-step local lookup.
On **Rung-2 $6\times 6$ 8-Piece Barrage Stratego** ($10{,}080$ setups/side, 2 center lakes `(2,2), (3,3)`), when the game-generating policy uses **2-ply tactical lookahead / coupled multi-piece move scoring** (so a move's likelihood depends jointly on the moving piece AND its supporting/defending pieces), does a learned any-order world model (`~1.1M` params) exhibit a genuine 0-step belief and value gap (`D0`), and does preference-guided rollout search with `attn_L3` + $\theta_{\text{trust}}$ achieve policy-independent 8-slot beliefs, separate trust from entropy, and beat 1-step Ataraxos?

### 2.2 Exact Code-Level Fixes Relative to `/home/agent/workspaces/mats/work/micro-stratego/`
1. **Board & Pieces (`env6.py`, extending `env.py` and `msgame.py`):**
   - $6\times 6 = 36$ squares; 2 impassable lakes at `(2, 2)` and `(3, 3)` (creating three tactical lanes: Left flank cols `0..1`, Center col `2..3` weave, Right flank cols `4..5`).
   - 8 pieces per side (Barrage Stratego roster): `1x Flag (1), 2x Bomb (2), 2x Scout (3), 1x Miner (4), 1x General (5), 1x Marshal (6)` placed on the back 2 rows (12 squares, choose 8 occupied squares $\times \frac{8!}{2!2!} = 10{,}080$ distinct piece permutations on the 8 occupied initial slots).
2. **Coupled 2-Ply Behaviour Policy & Exact Bayes Oracle $P^*_{\text{full}}$ (`msgame6.py`):**
   - **Why `R4-E1`'s `move_score` was too local:** In `R4-E1` (`msgame.py:69`), advancing scored a constant `+0.3 * advance` independent of setup!
   - **Coupled tactical move score in `msgame6.py`:** For player $X$'s true setup $Z_X = (z_0, \dots, z_7)$, the logit of move $a = (s: \text{fr} \to \text{to})$ depends on **both** the moving piece $z_s$ **and** the ranks of friendly pieces $z_{s'}$ within Chebyshev distance $\le 2$ of `to` (specifically: advancing a Flag-adjacent defender penalties unless backed by a Bomb/Marshal; advancing a Miner `+0.8` toward unmoved opponent back-row pieces when escorted by General/Marshal; advancing a Scout `+0.6` on open lanes; plus a 2-ply tactical reply term against the public opponent marginals).
   - Because $10{,}080$ setups fit in a `[10080, 8]` tensor on GPU/CPU, the **exact Bayes posterior $P^*_{\text{full}}(Z \mid h_t)$** (incorporating hard movement/battle constraints AND exact coupled move likelihoods at every ply) is computed to machine precision in vectorized PyTorch, just like `msgame.py`!
3. **Fixing `R4-E1`'s `D2` Key Collapse (`model_ms.py` & `train_ms.py`):**
   - **Bug 1 in `R4-E1` (`model_ms.py:426`):** `src[:, SLOT0:SLOT0+6]` set `True` on **all** hidden slots in every rollout, so every rollout attended to the exact same slot tokens!
     - **Fix:** In each rollout $r$, **only unmask the $m_{\text{sub}} = 3$ opponent slots that are within tactical range (distance $\le 2$) of the rollout's root move `(fr, to)`** (leaving distant opponent slots on the opposite flank as `MASK` during rollout $r$!), and set `src` to `True` **only** on those $m_{\text{sub}}$ unmasked slots and the board squares touched by the rollout's moves!
   - **Bug 2 in `R4-E1` (`train_ms.py:408`):** `fit_aggregators` trained `keyW2` and `attn_L3` only on a **scalar** root value `vref` where `l0root` was already optimal.
     - **Fix:** Each rollout $r$ outputs **both** a 8-slot belief update $\Delta \ell^{\text{belief}}_r \in \mathbb{R}^{8 \times 6}$ (from $\theta_{\text{real}}$'s belief head after conditioning on the rollout's unmasked tactical cone and 4-ply consistency) **and** a root value update $\Delta \ell^{\text{val}}_r \in \mathbb{R}$, and `fit_aggregators` trains `keyW2` and `attn_L3` to match the unskewed $K=128$ reference on the **joint 8-slot belief distribution (`8 x 6 = 48` logits) + root value**!

### 2.3 Pre-Registered Pass/Fail Rules (`R5-E2`)
- **`E2-D0` (Gate — Calibrated on a 5-min pre-check before locking):**
  On 3 seeds (`101, 102, 103`), evaluated on held-out setups:
  1. 1-step battle outcome accuracy $\ge 0.97$;
  2. 7-known-slot permutation NLL $\le 0.05$ nats;
  3. **Genuine 0-Step Belief Gap + Rollout Signal:** 0-step belief $\mathrm{KL}(P^*_{\text{full}} \Vert P_0) \ge \mathbf{0.05\text{ nats/slot}}$ (total $\ge 0.40$ nats across the 8 slots; note: if the 4-layer `d=128` net gets $<0.05$ at step 16k, use the pre-registered `d=96, 3-layer` capacity checkpoint that passes (1) and (2) while leaving $\ge 0.05$ nats/slot for search) **AND** 16 unskewed rollouts reduce belief KL or action-value MSE by $\ge 10\%$ on the calibration split.
- **`E2-D1` (Preference-Edge Search & Policy-Independent 8-Slot Beliefs):**
  When search is driven by Left-Flank Preference $U_{\text{Left}}$ (80% of rollouts on cols `0..2`) vs. Right-Flank Preference $U_{\text{Right}}$ (80% of rollouts on cols `3..5`):
  1. **Non-Collapsed Keys:** Mean key cosine within-flank minus across-flank is **$\ge 0.25$** (95% CI $>0$; fixing `R4-E1`'s `0.99 vs 0.99`);
  2. **Policy-Independent Merged 8-Slot Belief & Value:** `attn_L3` cuts policy-sensitivity $\mathrm{JSD}(\mu^*(U_{\text{Left}}), \mu^*(U_{\text{Right}}))$ by **$\ge 2.5\times$** compared to `mix`, while achieving lower MSE/KL to the unskewed $K=128$ reference than `mix`, `ataraxos_mmd`, and `kc_dedup`.
- **`E2-D2` (Self-Trust $\theta_{\text{trust}}$ vs. Entropy on $6\times 6$):**
  Spearman($\beta_{\text{trust}}, -\mathrm{KL}(Q^* \Vert q_r)$) $\ge +0.25$ overall and exceeds Spearman($-H(q_r), -\mathrm{KL}$) by $\ge +0.20$ overall and $\ge +0.15$ within entropy strata.
- **`E2-D3` (Zero-Retraining Preference Swap $U_{\text{Std}} \leftrightarrow U_{\text{Miner}}$ & Head-to-Head Play):**
  Swapping the utility edge $U_{\text{Std}} \leftrightarrow U_{\text{Miner}}$ at test time shifts action selection toward higher Miner-capture utility on $\ge 75\%$ of boards with `0.0000` change to $\theta_{\text{real}}$'s beliefs; and `Unified-LIR` (`stiff_voi` + $\theta_{\text{trust}}$ + `attn_L3`) achieves $> 52\%$ effective win rate vs. `P0` (400 duplicate-paired games/seed).

---

## 3. Experiment `R5-E3`: `E-SUD-EXTREME` — Unique-Solution Hard $9\times 9$ Sudoku Benchmark + Preference-Biased Search Orderings

### 3.1 Question It Answers
In `R4-E2`, the 3 trained seeds of our $2.69\text{M}$-parameter $9\times 9$ Sudoku world model (`vmout/s{101,102,103}/model_seed*.pt`) cut cell-belief KL by **`39.3%` (`P3`)** and **`42.2%` (`P6` Landmark)**, and `P6` beat `P3` in all 3 seeds (`+4.7% [1.2%, 8.1%]`) with `7–10%` fewer passes—**but** because `R4-E2`'s test boards had `24–28` givens and a mean of **`1,042` valid solutions per board**, `P3` already hit `100.0%` witness discovery (`D1` saturated!).
What happens when we evaluate those **exact already-trained $2.69\text{M}$-parameter checkpoints** (zero retraining of $\theta_{\text{real}}$ required!) on the standard **unique-solution hard $9\times 9$ benchmark regime (`17–22` givens, exactly 1 solution, `prop_depth >= 4`)**, and test **preference-biased search orderings** ($U_{\text{Top}}$ vs. $U_{\text{Bot}}$) aggregated by `attn_L3`?

### 3.2 Exact Spec & Code Changes Relative to `calibrated-ai/trader-economy/puzzles/R4-E2/`
1. **Unique-Solution Hard Benchmark Split (`boards9_extreme.py`):**
   - Generate $500$ validation and $500$ test **unique-solution** hard $9\times 9$ Sudoku puzzles ($k \in \{17, \dots, 22\}$ givens, **exact solution count $= 1$**, naked-single propagation depth $\ge 3$ leaving $\ge 35$ empty cells where no naked singles remain—matching the `sudoku-extreme` benchmark regime of Jolicoeur-Martineau 2025 / Ye et al. 2025), paired with $500$ unsolvable boards created by swapping one non-clashing given.
2. **Why Unique-Solution Hard Boards Break `P3` Saturation and Unlock Landmark + `attn_L3`:**
   - When a puzzle has **only 1 valid solution** (instead of 1,042 solutions), a greedy confidence-first rollout (`P3`) that makes even a single wrong 55/45 guess at cell 15 is **guaranteed to hit a contradiction later** (so `P3`'s individual rollout validity rate drops from `90%` to `15–45%`, and `P3` 16-rollout witness rate is no longer `100%`!).
   - Under **Landmark Tree Branching (`P6`)**, when a child branch hits a row/col/box conflict, backtracking to the Landmark $S_m$ and branching into the alternative digit $d_2$ directly prunes the dead subtree and finds the unique solution!
3. **Preference-Biased Search Orderings & Policy-Independent Merged Cell Beliefs (`D2`):**
   - Define two search-order preference edges over the 81 cells:
     - $U_{\text{Top}}$: adds a preference bonus $-\delta_U$ to the unmasking score $H(q_i) - \delta_U \mathbf{1}[i \in \text{Rows } 0..4]$ so rollouts resolve the top half of the board first (and $80\%$ of rollout trajectories focus on top-half deductions before bottom-half deductions).
     - $U_{\text{Bot}}$: adds $-\delta_U \mathbf{1}[i \in \text{Rows } 4..8]$ so rollouts resolve the bottom half first.
   - Because each rollout $r$ makes a sequence of uncertain guesses on a specific subset of cells before stalling or completing, its attention/guess footprint $k_r \in \mathbb{R}^{16}$ (projected from the indicator/embeddings of the high-entropy branch cells where rollout $r$ guessed) identifies **which conflict clause / guess subtree** rollout $r$ tested.
   - Compare naive rollout averaging (`mix`), scalar self-check weighting (`chk_mix` from `R4-E2`), and **`attn_L3` deduplicating attention** over the $K=16$ rollout cell-belief vectors.

### 3.3 Pre-Registered Pass/Fail Rules (`R5-E3`)
- **`E3-D0` (Unique-Solution Benchmark Gate — Evaluated on the 3 saved `R4-E2` checkpoints `s101, s102, s103`):**
  1. 1-step naked-single KL $\le 0.001$ nats (`100%` accuracy, already verified on `s101–s103`);
  2. On the unique-solution hard split (`17–22` givens, 1 solution), 0-step empty-cell $\mathrm{KL}(\text{exact} \Vert P_0) \ge 0.15$ nats **AND** flat 16-rollout `P3` single-rollout validity rate is $\le 0.70$ (confirming the benchmark is non-saturated, unlike `R4-E2`'s 1,042-solution boards!).
- **`E3-D1` (Landmark Branching `P6` Beats Flat `P3` and Random `P1` at Matched Compute on Unique-Solution Boards):**
  Across the 3 seeds (`101, 102, 103`) at matched world-model forward passes ($16(E+1)$):
  1. `P6` achieves lower empty-cell $\mathrm{KL}(\text{exact} \Vert \mu^*)$ than `P3` in all 3 seeds (paired 95% CI of $\mathrm{KL}(P_3) - \mathrm{KL}(P_6)$ excludes 0, relative cut $\ge \mathbf{8\%}$ on ambiguous branch cells) and cuts KL vs. `P0` by $\ge \mathbf{35\%}$;
  2. `P6` increases valid witness discovery rate or valid rollout completion yield over `P3` by $\ge \mathbf{+5\text{ pp}}$ (95% CI $>0$).
- **`E3-D2` (Policy-Independence of Merged 81-Cell Beliefs under Preference Edges $U_{\text{Top}}$ vs. $U_{\text{Bot}}$):**
  When the same unique-solution boards are searched under $U_{\text{Top}}$-biased vs. $U_{\text{Bot}}$-biased rollout policies:
  1. `attn_L3` reduces the between-preference belief divergence $\mathrm{JSD}\bigl(\mu^*(Z_{1..81} \mid \pi_{U_{\text{Top}}}),\, \mu^*(Z_{1..81} \mid \pi_{U_{\text{Bot}}})\bigr)$ by **$\ge 2.5\times$** relative to unweighted `mix`;
  2. `attn_L3` achieves lower 81-cell $\mathrm{KL}(\text{exact} \Vert \mu^*)$ under skewed search than both `mix` and scalar `chk_mix` (paired 95% CI excludes 0).
