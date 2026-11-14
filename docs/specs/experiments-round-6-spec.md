# Round-6 Experimental Specification: Resolving the Three Cross-Round Bottlenecks (Evidence-Floor Policy Independence, Key-Centering & Salience Variants, and Local Subproblem Rollouts vs. Global `chk_mix`)

**Date:** 2026-10-17T06:15Z  
**Author:** Gemini (for Kelvin & `opus-mats`)  
**Decision:** **RUN ROUND 6 (3 Focused Small-Network Experiments)** — Not yet `SATISFIED` per Kelvin's instruction (*"keep making rounds, even 5-10 rounds more if needed! Keep going until you understand every aspect, including variants if needed"*), and because Round 5 (`$7.01` total: `R5-E1` `$2.62`, `R5-E2` `$2.73`, `R5-E3` `$1.66`; `~$159` remaining of the `$200` budget) isolated three specific, solvable mechanisms across the entire project that Round 6 is designed to test directly.  
**Total Budget Cap:** **$\le \$12.00$** (estimated spend $\sim\$6.50\text{--}\$8.50$; two of the three experiments reuse already-trained world-model checkpoints on disk).  
**Target Wall Time:** **$\le 1.5\text{ hours}$** across 3 parallel executors on GCP L4 GPUs.

---

## Executive Synthesis of the Cross-Round Pattern (Rounds 1–5)

Across Rounds 4 and 5, **every pillar of the alignment and self-trust story is now solidly proven**, while **three sharp questions on search, keys, and aggregation** have emerged with exact diagnoses:

1. **What Round 5 Newly Proved (Major Wins to Lock into the Direction Note):**
   - **Rising-Reward LIR Parameter Masking (`R5-E1` Part B, `6/10` rules PASS):** Adding the action-entropy bonus (`0.02`) eliminated `R4-E3a`'s policy collapse: normalized reward gain hit **`0.876` (Fused), `0.946` (Token-Masked), and `0.902` (LIR-Masked)** of the Bayes ceiling (`0.798` vs `0.830` ceiling). Under **`LIR-Masked`**, belief drift was **`0.0000`** (`L-inf = 0.0000` across all 18,000 cells in every seed), correlation with exact Bayes held at **`0.9994 -> 0.9994`**, unwarranted false-note sycophancy was **`0.000`** (its raw `0.053` rate matched exact Bayes's own `0.055` rate on boards where nearby probes hit), and zero-retraining `faith_swap` flipped actions on **`100.0%` of boards in all 3 seeds**. Meanwhile, **`Fused`** jumped to **`+0.471` $\Delta_{\text{desire}}$** (`83.1%` sycophancy, `77.5%` unwarranted, Bayes $r$ dropping `0.9994 -> 0.722`), and **`Token-Masked`** drifted **`14x` more than the truth-only control (`0.0527` vs `0.0038`) specifically in the goal-utility direction (`+0.100` utility bias)**.
   - **Space of Information & Multi-Layer Attention (`R5-E1` Part A v2):** Under preference-biased search ($U_{\text{NW}}$ vs. $U_{\text{SE}}$), `attn_L3` cut 36-cell KL to exact Bayes by **`74.0% [73.7%, 74.4%]`** (`0.280 -> 0.073`), beating `mix` (`0.207`), `ataraxos_mmd` (`0.117`), and `kc_dedup` (`0.113`) in every seed (`E1-D1.2` PASS). It also deduplicated 16 identical clean reads (`W(16)/W(1) = 1.073`, cutting MSE **`53.1%` vs `mix`** and **`39.4%` vs `mmd`**), increased weight monotonically on noisy reads (`1.91x` at $N=8$, `2.12x` at $N=16$, cutting MSE **`17.3%` vs `kc_dedup`**), and on cells read by both searches its disagreement did **not** grow with read-count imbalance (`0.0172 -> 0.0158 -> 0.0150`, whereas `mmd` and `kc_dedup` grew **$3.5\times$** to `0.0466` and `0.0389`!).
   - **Why Policy Independence Requires Separating the *Support Gap* from *Frequency Deduplication* (`R5-E3` `H3` & `R5-E1` `§4.1`):**
     Both `R5-E3` and `R5-E1` uncovered the same theorem-level distinction:
     - When two preference-driven searches cover the **same** evidence with different visit frequencies (`both bags found witness` in `R5-E3`, or `shared cells with |diff| >= 3` in `R5-E1`), deduplication / check-weighting cuts JSD by **$3.1\times$ (`R5-E1` vs `mmd`)** to **$7\times\text{--}140\times$ (`R5-E3` `H3`: `0.086 -> 0.012 / 0.0006`)**.
     - When finite-budget open-loop search leaves some cells/witnesses **visited by only one search (`0` visits in the other bag)**, even an exact Bayesian oracle (`union call` in `R5-E1`, or `chk_mix` in `R5-E3`) *must* disagree on the unvisited evidence! Furthermore, using unweighted `mix` as the JSD denominator in `R5-E1` was a trap because `mix` dilutes every cell with $20/24$ uninformative prior reads (`KL = 0.207`), achieving tiny JSD (`0.0033`, worse than even the exact Bayesian union call `0.0136`!) by ignoring the evidence.

2. **The Three Cross-Round Questions Round 6 Answers Directly:**
   - **Question 1 (`R6-E1` — Key-Construction & Attention Variants + Full-Support vs. Active-PUCT Policy Independence in `E-INFO`):**
     1. *Why did raw keys in `R5-E1` have `0.71` cosine on disjoint patches and only `0.375` overlap Pearson?* Because all patch calls share the board/note hidden-state common mode, and input hidden states don't weight cells by how much the read actually shifted their log-odds. We test **4 key-construction variants (`K1 raw_h`, `K2 bag_centered_h`, `K3 delta_l_salience` $|\Delta \ell_r|$, `K4 centered_h + salience`)** and **2 multi-layer aggregator variants (`attn_L3` softmax attention vs. `looped_lin_T4` 4-step weight-tied linear attention from `arXiv:2402.14180`)**.
     2. *How does policy independence behave when (a) both preference-driven searches have full cell support (`6` base cover reads + `18` 90/10 preference-skewed reads, so $\mathrm{JSD}_{\text{union-clean}} = 0$), and (b) when **active multi-wave Key-Space PUCT** closes the support gap automatically without a pre-given base cover?*
   - **Question 2 (`R6-E2` — Why Did `attn_L3` Tie 1-Parameter `chk_mix` on Hard Sudoku (`R5-E3`), and When Does Multi-Rollout Attention Beat `chk_mix` on Self-Generated Rollouts?):**
     In `R5-E3`, every rollout unmasked **all 60 empty cells (`full-board rollout`)** and was scored by a single **board-wide check `chk_r`**. On a unique-solution board, a full-board rollout either finds the entire 81-cell answer (`chk_r = 1`, where 1 rollout already suffices!) or fails somewhere (`chk_r = 0`). On the **`61.4%` of `Sudoku-Extreme` boards where zero full-board rollouts succeed**, a rollout that solves **6 out of 9 boxes ($3\times 3$ subgrids) validly** before making one bad 50/50 guess in Box 9 gets global `chk_r = 0` and is discarded!
     In `R6-E2`, using the **saved `R4-E2` checkpoints**, each rollout is scored by **per-box local consistency checks $\text{chk}_{r, b} \in (0,1)$ ($b \in \{0,\dots,8\}$)** and/or runs as a **bounded local box-neighborhood rollout ($n = 18$ cells)**, allowing `attn_L3` (and per-box check pooling) to **stitch locally valid subgrid deductions across rollouts** even on the hard boards where global `chk_mix` fails!
   - **Question 3 (`R6-E3` — Why Did Self-Generated Rollouts Hurt on $6\times 6$ Stratego (`R5-E2`), and How Do Marginal Window Rollouts + Trust Weighting Fix It on the Saved `/tmp/r5e2/ckpt/` Checkpoints?):**
     `R5-E2`'s post-hoc diagnostics ([RESULTS-R5-E2.md](file:///home/agent/workspaces/mats/work/micro-stratego/RESULTS-R5-E2.md#L198-L214)) already proved the exact root cause:
     (i) Conditioning slot $s_1$ on *sampled* neighbor slots $z_{s_2}, z_{s_3}$ put hard zeros ($p < 0.001$) on $21\%$ of true piece types (due to 1-Marshal / 1-General uniqueness), which log-pooling turned into hard vetoes (`KL = 0.539`);
     (ii) In post-hoc diagnostic #2 (`expl_routeb.py`), when each slot was scored **marginally** (no neighbor sampling!) from the $t-4$ belief (`KL = 0.178` nats) using **$\theta_{\text{real}}$'s own learned policy head over the 4-ply window**, the learned forward model **cut belief KL by `57.3%` (`0.178 -> 0.076` nats)**!
     In `R6-E3`, using the **already-saved `/tmp/r5e2/ckpt/` checkpoints** ($<\$1.50$), we replace the neighbor-conditioned log-veto rollout with **marginal cone-slot window rollouts** from the $t-4$ / public-constraint anchor (`KL_0 = 0.178` nats), fix key collapse via bag-centering + salience keys (`K4`), and test whether weighting rollouts by $\theta_{\text{trust}}$ (`+0.23` Spearman vs `-0.36` for entropy) beats unweighted and entropy-weighted value/belief pooling.

---

## Ranked Experiment List

| Rank | ID | Experiment Name | Core Claim & Variants Tested | Executor | Est. Cost | Est. Wall Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **`R6-E1`** | **`E-INFO-V3`: Key & Attention Variants, Evidence-Floor Policy Independence, and Active Key-Space PUCT** | 4 key variants (`raw_h`, `centered_h`, `salience`, `centered+salience`); 2 attention variants (`attn_L3` vs. `looped_lin_T4`); full-support 90/10 preference skew ($\mathrm{JSD}_{\text{union}}=0$) + active 4-wave Key-Space PUCT; 3-noise-level Hellwig curve ($\tau \in \{0.3, 1.5, 2.5\}$) | `eurisko-study` (in `li-net/r5e1/`) | **$\$2.50$** (1 L4, reuses or fast-trains `wm.py`) | **45 min** |
| **2** | **`R6-E2`** | **`E-SUD-LOCAL`: Local Subgrid Self-Checks & Box-Neighborhood Rollout Stitching vs. Global `chk_mix` on `Sudoku-Extreme`** | Reuses saved `R4-E2` $2.69\text{M}$-param nets on `Sudoku-Extreme`; compares global 1-parameter `chk_mix` vs. per-box local self-checks $\text{chk}_{r,b}$ + `attn_L3` box-stitching on unresolved hard boards + two-wave stitched rollout amplification | `calibrated-ai` (in `puzzles/R6-E2/`) | **$\$2.50$** (reuses `R4-E2` checkpoints, 1 L4) | **45 min** |
| **3** | **`R6-E3`** | **`E-MS-MARGINAL`: $6\times 6$ Stratego Marginal Window Rollouts, Centered+Salience Keys & $\theta_{\text{trust}}$-Weighted Pooling** | Reuses saved `/tmp/r5e2/ckpt/` $6\times 6$ Stratego checkpoints; fixes the $21\%$ uniqueness-veto bug via marginal cone-slot window rollouts from the $t-4$ anchor (`KL_0 = 0.178`); tests `K4` centered+salience keys and $\theta_{\text{trust}}$-weighted vs. entropy-weighted action selection | `eurisko` (in `work/micro-stratego/r6/`) | **$\$1.80$** (reuses `/tmp/r5e2/ckpt/`, 1 L4 / CPU) | **40 min** |

*Total estimated spend: **$\sim\$6.80$** (hard group cap **$\$12.00$**).*

---

## 1. Experiment `R6-E1`: `E-INFO-V3` — Key & Attention Variants, Evidence-Floor Policy Independence & Active Key-Space PUCT

### 1.1 Exact Spec & Variants Relative to `li-net/r5e1/partA2.py`
Reuse the exact $6\times 6$ dense-sonar-field setup and $\theta_{\text{real}}$ v2 architecture from `R5-E1` (`partA2.py`; reload saved `theta_real` v2 checkpoints if cached, or re-run the 40k-step `wm.py` training in 40 min on 1 L4 for seeds `101, 102, 103`).

1. **Four Key-Construction Variants (Tested on the Same Rollout Bags):**
   - **`K1_raw` (`R5-E1` baseline):** $k_r = \mathrm{normalize}(W_K \mathrm{LN}(\bar{h}_r))$ where $\bar{h}_r \in \mathbb{R}^{128}$ is the attention-pooled `stop_grad` hidden state over the 6 read cells of patch $A_r$.
   - **`K2_centered` (Bag-Centered Hidden State Key):** Subtract the bag mean $\mu_{\bar{h}} = \frac{1}{K}\sum_{b=1}^K \bar{h}_b$ before projecting: $\tilde{h}_r = \bar{h}_r - \mu_{\bar{h}}$, $k_r = \mathrm{normalize}(W_K \mathrm{LN}(\tilde{h}_r))$.
   - **`K3_salience` (Belief-Shift Salience Footprint Key):** Let $s_r = \frac{|\Delta \ell_r|}{\| \Delta \ell_r \|_2 + 10^{-6}} \in \mathbb{R}_+^{36}$ be the normalized magnitude of rollout $r$'s log-odds shift across the 36 cells (computed purely from $\theta_{\text{real}}$'s output logits $\ell_r - \ell^{(0)}$, zero ground-truth patch coordinates given!). Project $k_r = \mathrm{normalize}(W_K s_r)$ (or directly use the centered/normalized 36-D salience vector $s_r$).
   - **`K4_combined` (Centered Hidden State + Salience Key):** Concatenate $[\mathrm{LN}(\bar{h}_r - \mu_{\bar{h}}) \,\|\, s_r] \in \mathbb{R}^{164} \to W_K \to \mathrm{normalize} \in \mathbb{R}^{16}$, trained with an auxiliary self-supervised **salience-inner-product alignment loss** $\mathcal{L}_{\text{key}} = \big(\langle k_r, k_b \rangle - \langle s_r, s_b \rangle\big)^2$ (which uses only $\theta_{\text{real}}$'s own output shift vectors $s_r$, never the true patch indices $A_r$!) alongside the 36-cell BCE aggregation loss.

2. **Two Multi-Layer Attention Aggregator Variants:**
   - **`attn_L3` (3-Layer Pre-LN Softmax Residual Self-Attention, `R5-E1` architecture):** Over tokens $z_r = [k_r, W_v \Delta \ell_r, \log \beta_r, \log c_r]$.
   - **`looped_lin_T4` (Weight-Tied 4-Step Linear Transformer / Preconditioned Richardson Iteration, `arXiv:2402.14180`):**
     Maintains per-rollout weight state $w^{(t)} \in \mathbb{R}^{K \times 36}$ for $t = 0, \dots, 3$ using a single weight-tied linear-attention block (`< 8k` parameters!):
     $$G_{rb} = \frac{\max(0, \langle k_r, k_b \rangle)^\gamma}{\sqrt{c_r c_b}}, \qquad r^{(t)}_r = \beta_r \Delta \ell_r - \sum_{b=1}^K G_{rb} \, \Phi_t(w^{(t)}_b, \Delta \ell_b, \beta_b), \qquad w^{(t+1)}_r = w^{(t)}_r + \eta_t \Psi_t(r^{(t)}_r).$$

3. **Two Preference-Driven Search Protocols (Fixing `R5-E1`'s `mix` Dilution & Support-Gap Confound):**
   - **Protocol P-FullSupport (Isolates Pure Repeat-Count Deduplication at Identical Evidence Floor):**
     Each search ($U_{\text{NW}}$ vs. $U_{\text{SE}}$, $K=24$ reads) includes the **6 disjoint $2\times 3$ tiles that partition the $6\times 6$ board once** (reads $1..6$, so **both** bags cover $100\%$ of the 36 cells and the clean union call has $\mathrm{JSD}_{\text{union-clean}} = 0.0000$!), plus **18 preference-skewed reads** ($90\%$ in the $U$-preferred half, $10\%$ in the opposite half; clean $\tau=0.3$).
     - Compare `ataraxos_mmd`, `kc_dedup`, `looped_lin_T4`, and `attn_L3` (plus an **amplitude-matched unweighted baseline `scaled_mix`** $\ell_0 + \bar{\alpha} \sum_{r=1}^K \Delta \ell_r$ where $\bar{\alpha}$ is chosen on the calibration split to minimize unskewed KL, so it cannot hide behind `mix`'s $1/24$ prior dilution!).
   - **Protocol P-ActivePUCT (Tests Active Multi-Wave Key-Space PUCT Closing the Support Gap without Pre-Given Base Cover):**
     No pre-given base cover! Starting from $0$ reads, compare:
     - **Open-Loop Preference Search (`open_U`):** All $K=24$ reads drawn open-loop with $85\%$ in the $U$-preferred half (`R5-E1` protocol).
     - **4-Wave Key-Space PUCT (`puct_key_U`, Direction Note §3.2):** Runs 4 waves of 6 reads. In wave $w$, candidate patch $p$ is scored by its preference-weighted VOI divided by the **learned key crowding** against already-executed reads $r' < r$:
       $$\mathrm{Score}_U(p) = \frac{\big(1 + \lambda_U \mathbf{1}[p \in \text{half}(U)]\big) \cdot \mathrm{Var}_{\mu^{(w-1)}}(p)}{\big(1 + \sum_{r' < r} \max(0, \langle k_p, k_{r'} \rangle)^2\big)^{\alpha_{\text{nov}}}}, \qquad (\lambda_U = 3.0, \, \alpha_{\text{nov}} = 1.5).$$
       In Wave 1, crowding is 0 so search focuses $85\%$ on $U$'s preferred half; by Waves 2–4, key crowding saturates the already-read preferred patches and automatically redirects later waves to uncovered patches!

4. **Three-Noise-Level Hellwig / Clemen-Winkler Curve ($\tau \in \{0.3, 1.5, 2.5\}$):**
   - On the controlled bags ($N \in \{1, 2, 4, 8, 16\}$ target-patch reads + 8 background reads), add a **high-noise tier $\tau = 2.5$** (each patch reading re-flipped w.p. $0.28$ fresh per read, effective flip $0.346$, $\theta_{\text{trust}}$ sees the $\tau$ tier flag).

### 1.2 Pre-Registered Pass/Fail Rules (`R6-E1`)
- **`E1-D0` (Gate):** World model $\theta_{\text{real}}$ has Pearson $r \ge 0.99$ with exact Bayes on the 6-reading format, AND on `P-FullSupport` bags `scaled_mix` cuts 36-cell KL over $P_0$ by $\ge 40\%$.
- **`E1-D1` (Key-Construction Variants — Graded Partial Overlap Without Supervision):**
  1. `K4_combined` (and/or `K3_salience`) achieves Pearson correlation $r(\langle k_r, k_b \rangle, |A_r \cap A_b|/6) \ge \mathbf{0.60}$ (95% CI $> 0.50$, up from `0.375` in `R5-E1`) **AND** same-patch minus across-sector cosine gap $\ge \mathbf{0.50}$ (up from `0.218`);
  2. Mean key cosine increases monotonically across overlap levels $\{0, 1, 2, 3, 4, 6\}$ shared cells (Spearman rank correlation across the 6 levels $\ge 0.94$).
- **`E1-D2` (Policy Independence under Full-Support Skew & Active Key-Space PUCT):**
  1. **Full-Support Repeat-Count Deduplication (`P-FullSupport`):** Under 90/10 preference skew on top of full support, `attn_L3` (with `K4_combined` keys) cuts between-preference $\mathrm{JSD}\bigl(\mu^*(U_{\text{NW}}), \mu^*(U_{\text{SE}})\bigr)$ by **$\ge 2.5\times$** compared to **each** of `scaled_mix`, `ataraxos_mmd`, and `kc_dedup` (all 3 paired CIs of JSD difference $>0$), while achieving the lowest 36-cell KL to exact Bayes (paired CIs $>0$ vs. all 3 baselines). *(Report `looped_lin_T4` alongside as a secondary architectural comparison: passes if within $15\%$ of `attn_L3`'s KL while using $<1/5$ the parameters.)*
  2. **Active Key-Space PUCT Closes the Support Gap (`P-ActivePUCT`):** Without any pre-given base cover, 4-wave `puct_key_U` raises mean 36-cell board coverage from $\le 87\%$ (`open_U`) to **$\ge 95\%$**, reduces full-board $\mathrm{JSD}\bigl(\mu^*(U_{\text{NW}}), \mu^*(U_{\text{SE}})\bigr)$ under `attn_L3` by **$\ge 1.8\times$** relative to `open_U` (95% CI of difference $>0$), and reduces full-board KL to exact Bayes by **$\ge 20\%$** relative to `open_U` while maintaining $\ge 75\%$ NW share in Wave 1 under $U_{\text{NW}}$.
- **`E1-D3` (Three-Regime Hellwig / Clemen-Winkler Weight Law):**
  On controlled bags ($N \in \{1, 2, 4, 8, 16\}$):
  1. At clean $\tau = 0.3$ (deduplication): $W(16)/W(1) \le \mathbf{1.25}$ and `attn_L3` cuts MSE by $\ge 35\%$ vs. `ataraxos_mmd`;
  2. At high-noise $\tau = 2.5$ (mutual support): $W(8)/W(1) \ge \mathbf{2.20}$ (95% CI lower bound $> 1.90$) and $W(16)/W(1) \ge \mathbf{2.80}$, with `attn_L3` cutting MSE by $\ge 15\%$ vs. `kc_dedup`;
  3. Monotonic noise scaling: $W_{\tau=2.5}(8)/W_{\tau=2.5}(1) > W_{\tau=1.5}(8)/W_{\tau=1.5}(1) > W_{\tau=0.3}(8)/W_{\tau=0.3}(1)$ in every seed.

---

## 2. Experiment `R6-E2`: `E-SUD-LOCAL` — Per-Box Local Self-Checks & Subproblem Rollout Stitching vs. Global `chk_mix` on `Sudoku-Extreme`

### 2.1 Question It Answers
In `R5-E3`, 16 full-board confidence-first rollouts cut empty-cell KL on `Sudoku-Extreme` by `34.1%` (`1.18 -> 0.78` nats), but 3-layer `attn_L3` tied 1-parameter global `chk_mix` (`0.84` vs `0.82` nats). Why?
Because when every rollout is a full 81-cell completion and `chk_r` is a single global board scalar, either a rollout solves the whole board (`chk_r = 1`, where 1 rollout has all 81 answers) or it hits a contradiction in at least one box (`chk_r = 0`). On the **`61.4%` of `Sudoku-Extreme` boards where zero rollouts achieve global completion**, a rollout that solves **6 or 7 out of 9 boxes ($3\times 3$ subgrids) validly** before making a wrong guess in Box 9 gets global `chk_r = 0`—wasting all its valid local deductions!
Can **per-box local consistency scoring ($\text{chk}_{r, b}$ for each $3\times 3$ box $b \in \{0,\dots,8\}$) + box-level attention stitching (`box_attn_L3`)** salvage those locally valid subgrid deductions, beat global 1-parameter `chk_mix` on hard unresolved boards, and boost 2-wave witness discovery?

### 2.2 Exact Spec Relative to `calibrated-ai/trader-economy/puzzles/R5-E3/`
Reuse the **exact 3 saved `R4-E2` $2.69\text{M}$-param checkpoints (`s101, s102, s103`)** and the `Sudoku-Extreme` splits (`s9x_{cal,val,test}.npz`, `17–22` givens, unique solution) from `R5-E3`—**zero world-model retraining!**

1. **Per-Box Local Self-Check & Subgrid Tokens (Purely from $\theta_{\text{real}}$'s Own Heads, No Rule Checker in Inference!):**
   - For each rollout $r \in \{1, \dots, K\}$ ($K=16$), in addition to thecompleted board $Z^{(r)}_{1..81}$ and its step-time conditional probabilities $q_r \in \Delta^9{}^{81}$, compute **$\theta_{\text{real}}$'s local box-consistency score $\text{chk}_{r, b} \in (0, 1)$ for each of the 9 boxes $b \in \{0, \dots, 8\}$**:
     - Specifically, on the completed board $Z^{(r)}$, pass $Z^{(r)}$ through $\theta_{\text{real}}$ (which costs **1 forward pass**, already done to read `chk_r`!) and define for each cell $c \in \{0, \dots, 80\}$ the **leave-one-out / one-pass self-consistency probability** $p^{\text{self}}_{r, c} = q_{\theta_{\text{real}}}\big(Z_c = Z^{(r)}_c \mid Z^{(r)}\big)$ (or the geometric mean of $\theta_{\text{real}}$'s cell probabilities across the 9 cells of box $b$ + its row/col peers: $\text{chk}_{r, b} = \big(\prod_{c \in \text{Box}(b)} p^{\text{self}}_{r, c}\big)^{1/9}$).
     - Notice why $\text{chk}_{r, b}$ is completely legal (100% world-model internal, zero external rule checker!): if rollout $r$ made a contradiction in Box 9 while Boxes 1–6 are locally valid, $\theta_{\text{real}}$'s own cell heads in Box 9 see the clashing digits and assign $p^{\text{self}}_{r, c} \approx 0$ in Box 9, while assigning $p^{\text{self}}_{r, c} \approx 1$ in Boxes 1–6!
2. **Two Regimes Tested (`Part A: Full-Board Rollouts with Box-Level Stitching` & `Part B: Bounded 2-Box Local Rollouts under Preference Skew`):**
   - **Part A (Full-Board $K=16$ Rollouts on `Sudoku-Extreme`):**
     Compare 4 aggregators on the exact same $K=16$ confidence-first rollouts per board:
     1. `mix`: unweighted average.
     2. `global_chk_mix` (`R5-E3` winner): 1 scalar weight per rollout ($\text{chk}_r$, with $\lambda$ fitted on `val`).
     3. `local_box_chk_mix`: per-cell/per-box local self-consistency weight $w_{r, c} = (p^{\text{self}}_{r, c})^{\gamma_c} (\text{chk}_{r, \text{box}(c)})^{\gamma_b}$ (with $(\lambda, \gamma_c, \gamma_b)$ fitted on `val` outcomes).
     4. `box_attn_L3`: 3-layer residual self-attention over $(r, b)$ box-rollout tokens $z_{r, b} = [k_{r,b},\, \text{summary}(q_{r, \text{box}(b)}),\, \log \text{chk}_{r,b},\, \log \text{chk}_r,\, \log c_{r,b}]$, outputting per-rollout per-box weights $w_{r, b} \ge 0$ trained on `cal` (`s9x_cal.npz` + `s9x_cal2.npz`, $5{,}000$ boards).
     - **Two-Wave Stitched Search (`P7_stitched` vs. `P3` at Matched Total Budget $K=16$):**
       Run Wave 1 with $K_1 = 8$ rollouts, aggregate via `local_box_chk_mix` / `box_attn_L3` to form the stitched belief $\mu^{(1)}$, clamp any cell where $\max_d \mu^{(1)}_c(d) \ge \tau_{\text{clamp}}$ (fitted on `val`, e.g. $0.95$) AND local consistency holds, and run Wave 2 with the remaining $K_2 = 8$ rollouts from the stitched Landmark!
   - **Part B (Bounded Local Subproblem Rollouts, $n = 18$ Unmasked Cells per Rollout, Driven by Preference Edges $U_{\text{Top}}$ vs. $U_{\text{Bot}}$):**
     - Instead of unmasking all 60 empty cells in every rollout, each rollout $r \in \{1, \dots, 24\}$ selects a target $3\times 3$ box $b_r$ (drawn with $80\%$ weight on Top-band boxes $\{0,1,2\}$ under $U_{\text{Top}}$ vs. Bottom-band boxes $\{6,7,8\}$ under $U_{\text{Bot}}$, plus 1 base rollout per box for full support) and unmasks **only the empty cells in box $b_r$ and its highest-confidence peers (up to $n = 18$ cells total)**, leaving the rest of the board masked!
     - Here, no single rollout ever completes the whole board, so `global_chk_mix` (1 scalar weight per rollout) dilutes every box by the $\sim 20$ rollouts that never unmasked that box, whereas `local_box_chk_mix` and `box_attn_L3` route each rollout's update specifically to the cells/boxes it unmasked and deduplicate the $80\%$ oversampled band!

### 2.3 Pre-Registered Pass/Fail Rules (`R6-E2`)
- **`E2-D0` (Gate — Local Self-Checks Separate Valid Boxes from Contradictory Boxes Within Failed Rollouts):**
  On the `Sudoku-Extreme` validation boards where `P3` finds **zero** global witnesses (`~61%` of boards):
  1. Within failed rollouts, cells where the rollout's digit matches the true solution have significantly higher $\theta_{\text{real}}$ local self-consistency $p^{\text{self}}_{r, c}$ than cells where the rollout guessed wrong (AUROC of $p^{\text{self}}_{r, c}$ for predicting per-cell correctness $\ge \mathbf{0.75}$ in all 3 seeds!).
- **`E2-D1` (Local Box Consistency & Attention Stitching Beat Global 1-Parameter `chk_mix` on Full-Board Rollouts):**
  Across the 3 seeds (`101, 102, 103`) on the 500 `Sudoku-Extreme` test boards:
  1. On the unresolved boards (where $K=16$ rollouts find no global witness) AND across all 500 test boards, `local_box_chk_mix` and `box_attn_L3` both achieve strictly lower empty-cell $\mathrm{KL}(\text{exact} \Vert \mu^*)$ than 1-parameter `global_chk_mix` (paired 95% CI excludes 0, relative KL cut on unresolved boards $\ge \mathbf{8\%}$);
  2. At matched $K=16$ total rollout budget, two-wave stitched search `P7_stitched` achieves strictly lower empty-cell KL than flat `P3` (`global_chk_mix`) **and** increases the valid rollout rate or board witness rate (paired 95% CI $>0$).
- **`E2-D2` (Bounded $n=18$ Local Box Rollouts under Preference Skew $U_{\text{Top}}$ vs. $U_{\text{Bot}}$):**
  On bounded local subproblem rollouts ($n=18$ cells unmasked per rollout):
  1. `box_attn_L3` cuts empty-cell $\mathrm{KL}(\text{exact} \Vert \mu^*)$ by **$\ge 25\%$** compared to 1-parameter `global_chk_mix` (paired 95% CI $>0$) and beats `ataraxos_mmd` and `kc_dedup`;
  2. `box_attn_L3` cuts between-preference $\mathrm{JSD}\bigl(\mu^*(U_{\text{Top}}), \mu^*(U_{\text{Bot}})\bigr)$ by **$\ge 2.0\times$** compared to `ataraxos_mmd` and `global_chk_mix` (paired 95% CI $>0$).

---

## 3. Experiment `R6-E3`: `E-MS-MARGINAL` — $6\times 6$ Stratego Marginal Window Rollouts, Centered+Salience Keys & $\theta_{\text{trust}}$-Weighted Decisions

### 3.1 Question It Answers
In `R5-E2` ($6\times 6$ Stratego), post-hoc diagnostics ([RESULTS-R5-E2.md](file:///home/agent/workspaces/mats/work/micro-stratego/RESULTS-R5-E2.md#L198-L218)) revealed two exact facts:
1. Rollouts ruined beliefs (`0.047 -> 0.539`) because `window_messages` conditioned each cone slot on **sampled neighbor slots** (which assigned $p < 0.001$ to the true rank on $21.4\%$ of claims due to 1-Marshal / 1-General uniqueness) and then log-pooled those zeros; whereas when `expl_routeb.py` scored each slot **marginally** (without neighbor sampling!) over the 4-ply window from the $t-4$ belief (`KL = 0.178` nats) using **$\theta_{\text{real}}$'s own learned policy head**, the 4-ply rollout cut belief KL by **`57.3%` (`0.178 -> 0.076` nats)**!
2. Furthermore, on action-value rollouts (`169,992` rollouts in `D2`), $\theta_{\text{trust}}$ achieved **`+0.228` Spearman with truth vs. `-0.362` for entropy (`+0.590` gap!)** because the lowest-entropy decile of rollouts (terminal flag captures) had **`1.01` nats error vs. `0.22` for the rest**—yet `D2`'s pre-registered absolute bar was `0.25` (missed by `0.022`), and we haven't yet tested whether **weighting action-value rollouts by $\theta_{\text{trust}}$ beats unweighted (`mix`) and entropy-weighted (`stiff`) pooling on action-value MSE and delusional-move avoidance on $6\times 6$**!

### 3.2 Exact Spec Relative to `/home/agent/workspaces/mats/work/micro-stratego/r5/` (Reusing `/tmp/r5e2/ckpt/`)
Reuse the **already-trained checkpoints in `/tmp/r5e2/ckpt/s{101,102,103}/primary/`** and data in `/tmp/r5e2/data/`—**zero retraining of $\theta_{\text{real}}$ needed!**

1. **Marginal 4-Ply Window Cone Rollouts from the $t-4$ Anchor (`window_messages_marginal`):**
   - Root anchor $P_{\text{anchor}}$ is $\theta_{\text{real}}$'s 0-step belief at $t_0 = \max(t - 4, t \bmod 2)$ (`KL = 0.178` nats on test, representing an agent whose belief cache was last updated 4 plies ago and must now integrate the last 4 plies of opponent moves by querying $\theta_{\text{real}}$'s learned forward policy head on 3-slot tactical cones!).
   - In each rollout $r \in \{1, \dots, K\}$, select the 3 hidden slots strictly **within tactical range or nearest to the root move `(fr, to)`** (`cone_r`), and for each slot $s \in \text{cone}_r$, compute its **marginal** 4-ply policy log-likelihood ratio $\Delta \ell_r[s, k] = \mathrm{ll}(Z_s = k, Z_{-s} = \text{MASK})$ (no neighbor sampling $\implies$ zero uniqueness vetoes!).
2. **Fixing Key Collapse via `K4_combined` (Bag-Centered Fingerprint + 8-Slot Salience Vector):**
   - Replace `R5-E2`'s uncentered `ebar` with the bag-centered attention fingerprint $\tilde{e}_r = \bar{e}_r - \frac{1}{K}\sum_b \bar{e}_b$ concatenated with the 8-slot coverage/salience indicator $s_r = \|\Delta \ell_r[s]\|_2 \in \mathbb{R}^8$, trained with the auxiliary salience-alignment loss $\mathcal{L}_{\text{key}} = (\langle k_r, k_b \rangle - \langle \hat{s}_r, \hat{s}_b \rangle)^2$ + calibration KL loss under `P-FullSupport` flank skew ($p_L \sim \mathrm{Beta}(0.5, 0.5)$).
3. **Direct Action-Value Test of $\theta_{\text{trust}}$ vs. Entropy Stiffness (`D2`):**
   - On the 240 $Q^*$ test positions ($K=16$ rollouts per legal action), compare three rollout weighting schemes for estimating $Q^*(s, a)$:
     1. `stiff_pool`: weights rollouts by $\exp(-\gamma H(q_r))$ (high confidence = high weight);
     2. `unif_pool`: uniform rollout average;
     3. `trust_pool`: weights rollouts by $\theta_{\text{trust}}$'s precision $\beta_r$ (which downweights confident hallucinated flag-capture terminals!).

### 3.3 Pre-Registered Pass/Fail Rules (`R6-E3`)
- **`E3-D0` (Gate — Marginal 4-Ply Window Rollouts Beat the $t-4$ Anchor):**
  On the 3 seeds (`101, 102, 103`), starting from the $t-4$ anchor belief (`KL_anchor >= 0.12` nats/slot):
  1. 1-step battle accuracy $\ge 0.99$ and 7-known-slot NLL $\le 0.01$;
  2. Pooling $K=16$ unskewed marginal cone rollouts cuts per-hidden-slot $\mathrm{KL}(P^*_{\text{full}} \Vert \mu)$ over the anchor by **$\ge 40\%$** on the calibration split (confirming marginal window rollouts strongly improve beliefs, fixing `R5-E2`'s `-1055%` neighbor-veto bug!).
- **`E3-D1` (Non-Collapsed Keys & Policy-Independent 8-Slot Beliefs under Flank Preference Skew):**
  Under Left-flank ($U_{\text{Left}}$) vs. Right-flank ($U_{\text{Right}}$) preference skew (4 base cover rollouts + 20 85/15 flank-skewed rollouts):
  1. **Non-Collapsed Keys:** Within-flank minus across-flank key cosine is **$\ge 0.25$** (95% CI $>0$, fixing `R5-E2`'s `0.004`);
  2. **Policy-Independence & Accuracy:** `attn_L3` cuts 8-slot belief $\mathrm{JSD}\bigl(\mu^*(U_{\text{Left}}), \mu^*(U_{\text{Right}})\bigr)$ by **$\ge 2.5\times$** compared to `ataraxos_mmd` and `scaled_mix` (paired 95% CIs $>0$), and achieves lower 8-slot belief KL to $P^*_{\text{full}}$ than `scaled_mix`, `ataraxos_mmd`, and `kc_dedup` (all paired 95% CIs $>0$).
- **`E3-D2` (Self-Trust $\theta_{\text{trust}}$ Beats Entropy Stiffness in Both Ranking and Action-Value Pooling):**
  1. Spearman($\beta_{\text{trust}}, -\mathrm{KL}(Q^* \Vert q_r)$) $\ge \mathbf{0.20}$ overall (95% CI $>0$), exceeds Spearman($-H(q_r), -\mathrm{KL}$) by $\ge \mathbf{0.45}$ overall (95% CI $>0$) and by $\ge \mathbf{0.15}$ within 10 equal-count entropy strata;
  2. **Decision Impact:** Weighting per-action rollouts by $\beta_{\text{trust}}$ (`trust_pool`) achieves strictly lower action-value $\mathrm{MSE}(Q^*, \hat{Q})$ and strictly lower delusional-move rate on the 240 $Q^*$ test positions than entropy-stiffness weighting (`stiff_pool`) (paired 95% CI of MSE reduction $>0$, relative MSE reduction $\ge \mathbf{20\%}$).
