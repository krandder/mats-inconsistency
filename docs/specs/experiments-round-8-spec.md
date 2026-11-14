# Round 8 Experiments Specification: Strong-Root Anytime Monotonicity, Full-Budget Beam Completion & Epistemic vs. Corner-World Trust

- **Status:** Ranked Round-8 specification (`R8-E1`, `R8-E2`, `R8-E3`), following Round-7 completion and audits (`R7-E1`, `R7-E2`, `R7-E3`).
- **Budget:** Round 1–7 spend is **$\$46.00$ of the second $\$200.00$ budget** ($\sim\$154.00$ remaining). Round 8 requires **zero world-model retraining across all three lanes** (all three load frozen, hash-verified checkpoints from disk) and is budgeted at **$\sim\$1.35$ expected ($\le \$2.20$ hard cap)**.
- **Core Scientific Goal of Round 8:**
  1. **Anytime Monotonicity & Strong-Root Safety (Kelvin's Question 1):** Prove across all three domains (`ShipHunt`, `Sudoku-Extreme`, `Stratego`) that our search and aggregation procedure **never degrades beliefs in expectation as rollout budget $K$ increases ($K = 0 \to 1 \to 2 \to 4 \to 8 \to 16 \to 32$)**—even when starting from an **already-strong one-pass root belief** where naive rollout pooling (`mix`, `ataraxos_mmd`, or raw replacement) makes beliefs significantly worse due to variance and double-counting.
  2. **Close the Exact Mechanistic Residuals Diagnosed in Round 7:**
     - **`R8-E1` (`E-INFO-ANYTIME`, `eurisko`, $\sim\$0.45$):** Test strong-root (`18`-anchor) and sparse-root (`2`-anchor) anytime monotonicity for `looped_attn_T3`; fix `jacobi_lin_T4`'s clean deduplication ratio ($1.449 \to \le 1.20$) via a 2-parameter precision-aware ridge $\lambda_{r,c}(\beta_r)$; fix `econ_kc_K3` by fitting $(u, \lambda)$ on mixed-tier D3 probes; and run the $2\times 2$ PUCT attribution (`K1` vs. `K3` $\times$ R6 vs. R7 wave-2+ score).
     - **`R8-E2` (`E-SUD-ANYTIME-BEAM`, `calibrated-ai`, $\sim\$0.55$):** Fix the three Phase-B completion bottlenecks discovered in `R7-E2`'s audit (where Phase A kept the true solution path in the top-16 beam on **$83.3\%$** of boards using only $15\text{--}33\%$ of the pass budget, but Phase B left **$22\text{--}45\%$ of the budget unspent**, concentrated chains by $\exp(\text{logP})$ on the stiffest wrong cascade, and set `lead=(r==0)` per Landmark—forcing Landmarks #2–#15 to run 1 deterministic greedy `argmax` chain that never sampled at Stalls 5–7), plus evaluate the anytime monotonicity curve across budgets $B \in \{0, 1, 2, 4, 8, 16\}\times(E_b+1)$.
     - **`R8-E3` (`E-MS-SAFE-ANCHOR`, `eurisko`, $\sim\$0.35$):** Anchor $6\times 6$ Stratego search at the **strong current-turn belief $\ell^{(0)}_t$** (`KL = 0.047` nats) instead of only the stale $t-4$ belief (`0.185`), proving `attn_L3-K3c` and `looped_attn_T3-K3c` strictly improve on $\ell^{(0)}_t$ across $K \in \{0, 1, 2, 4, 8, 16\}$ without ever degrading it; and separate **valid conditional corner worlds** (weighted by window consistency $\exp(\mathrm{LL}_{\text{window}}(Z^{(r)}))$) from **epistemic rollout corruption** (filtered by $\theta_{\text{trust}}$).

---

## Global Pre-Registration & Execution Protocol (Applies to All Three Lanes)

1. **Zero Oracle Leakage:** Exact Bayes posteriors (`R8-E1`, `R8-E3`), $Q^*$ (`R8-E3`), and ground-truth Sudoku solutions / `valid9` (`R8-E2`) are **strictly evaluation-only**. No oracle quantity may enter model inputs, search policies, stopping rules, or hyperparameter selection.
2. **Frozen World Models (Zero Retraining):**
   - `R8-E1` loads `/home/agent/workspaces/eurisko-study/li-net/r7e1/ckpt/s{101,102,103}/theta_real_v2.npz` (verifying `sha256` against `theta_real_v2.json`).
   - `R8-E2` loads `R4-E2` checkpoints `s{101,102,103}/model_seed*.pt` and `R5-E3`/`R7-E2` `Sudoku-Extreme` splits (`500` solvable val, `500` solvable test) unchanged.
   - `R8-E3` loads `R5-E2` checkpoints `/tmp/r5e2/ckpt/s{101,102,103}/primary/` and data splits unchanged.
3. **Immediate VM Teardown:** Every GPU script must wrap execution in `try / finally` (or shell `trap '... instance delete ...' EXIT`) so the VM is deleted immediately upon job completion (preventing idle VM spend like `R7-E2`).

---

## Priority 1 (`R8-E1`): `E-INFO-ANYTIME` — Strong-Root Anytime Monotonicity, Precision-Aware Jacobi Normal Equations & PUCT Attribution on $6\times 6$ ShipHunt

- **Executor:** `eurisko` (loading cached `theta_real_v2.npz` for seeds `101, 102, 103`; single L4 or CPU/GPU job, est. **$\$0.45$**, hard cap **$\$0.80$**).

### 1.1 Root-Cause Diagnosis & Design

1. **Kelvin's Question (1) — Strong-Root vs. Sparse-Root Anytime Monotonicity ($K \in \{0, 1, 2, 4, 8, 16, 24, 32\}$):**
   - When the 0-step root belief $\ell^{(0)}$ is already highly informative—specifically **`P-StrongRoot`**, where the root has already observed **$M_0 = 18$ clean anchor reads** (3 full 6-tile partitions or 18 distinct $2\times 3$ reads at $\tau=0.3$, so $\mathrm{KL}(P^* \Vert p^{(0)}) \approx 0.035\text{--}0.050$ nats)—adding $K$ additional skewed/noisy reads ($90/10$ NW/SE skew with $\tau_r$ drawn uniformly from $\{0.3, 1.5, 2.5\}$) causes naive aggregators (`ataraxos_mmd`, uncalibrated `mix`, and fixed-step `scaled_mix` without root-coverage accounting) to **degrade** the belief ($\mathrm{KL}(K) > \mathrm{KL}(0)$) by double-counting already-known tiles and injecting sensor noise.
   - To make `looped_attn_T3` and `attn_L3-K3` **anytime-safe across arbitrary root strengths $M_0$ and rollout counts $K$**:
     - Include an explicit **Anchor Token ($r = 0$)** in the rollout bag representing the information already present in $\ell^{(0)}$: its salience footprint is $s_0 = |\ell^{(0)} - \ell_{\text{prior}}| / (\|\ell^{(0)} - \ell_{\text{prior}}\|_2 + \varepsilon)$ mapped through the shared key head $k_0 = \mathrm{normalize}(W_K s_0)$, its shift is $\Delta \ell_0 = \mathbf{0}$ (since $\ell^{(0)}$ is the starting point of the residual stream), and its precision is the anchor mass $\beta_0 = \sum_{a=1}^{M_0} \beta_{\text{anchor}, a}$.
     - Because token $r=0$ sits in the attention key/value set $\{k_0, k_1, \dots, k_K\}$, any subsequent rollout $r \ge 1$ that re-reads subspace already covered by the strong root $\ell^{(0)}$ has high overlap $\langle k_r, k_0 \rangle$ with the anchor token and is **automatically deduplicated against $\ell^{(0)}$**, while at $K=0$ the residual sum over $r \ge 1$ is empty so $\ell^{(T)}(K=0) \equiv \ell^{(0)}$ exactly!
     - Train `looped_attn_T3` (`39,428` params) and `attn_L3-K3` across mixed root strengths $M_0 \in \{2, 6, 12, 18\}$ and rollout counts $K \in \{0, 2, 4, 8, 16, 24\}$ on `cal` boards using only realized board labels $Y$.

2. **Precision-Aware Jacobi-Richardson (`jacobi_lin_T4_prec`, $814$ params):**
   - In `R7-E1` `D1.2`, $812$-param `jacobi_lin_T4` cut KL by $40\%$ vs. `R6`'s `looped_lin_T4` (`0.0408 vs 0.0680`) and satisfied strict tier monotonicity ($W(8)/W(1) = 1.404 < 1.823 < 2.201$), missing only $W(16)/W(1) \le 1.25$ at $\tau=0.3$ (`1.449`) because a tier-independent ridge $\lambda_c \approx 1.58$ yields $\frac{16(\lambda + \beta)}{\lambda + 16\beta} = 1.43$ at $\beta = 3.40$.
   - In the exact normal equations, clean reads ($\tau=0.3$, high $\beta_r$) have near-zero observation noise variance whereas noisy reads ($\tau \in \{1.5, 2.5\}$, low $\beta_r$) require ridge shrinkage. Replace the static $\lambda_c$ with a **precision-scaled ridge** adding just **2 scalar parameters** $(a_\lambda, b_\lambda)$:
     $$\lambda_{r,c} \;=\; \mathrm{softplus}(\tilde{\lambda}_c) \cdot \sigma(a_\lambda - b_\lambda \beta_r), \qquad c_{r,c} \;=\; \lambda_{r,c} \;+\; \sum_{b=1}^K v_b \beta_b G_{rb}, \qquad d_{r,c} \;=\; \frac{v_r \beta_r}{c_{r,c}}.$$
     When $\beta_r$ is large ($\tau=0.3$), $\lambda_{r,c} \to 0$ so $d_{r,c} \to 1/N$ and $W(16)/W(1) \to 1.0$; when $\beta_r$ is small ($\tau=2.5$), $\lambda_{r,c}$ stays positive so $W(N)/W(1)$ grows strongly with $N$.

3. **Mixed-Tier Fitting for `econ_kc_K3` & $2\times 2$ PUCT Attribution:**
   - Fit the 2 scalars $(u, \lambda)$ of `econ_kc_K3` ($w_r = \frac{u \beta_r}{\lambda + \sum_b \beta_b G_{rb}}$) on the **mixed-tier D3 calibration bags** ($N \in \{1, 2, 4, 8, 16\} \times \tau \in \{0.3, 1.5, 2.5\}$) rather than single-tier `P-FullSupport`, and evaluate on both `P-FullSupport` and `D3`.
   - Run the $2\times 2$ PUCT factorial (`K1` vs. `K3` keys $\times$ `R6` score vs. `R7` Reading-9 score, all at 4 waves = 24 reads) to decompose the `0.9494 -> 0.9863` coverage gain.

### 1.2 Pre-Registered Decision Rules (`R8-E1`)

- **`E1-D0` (Checkpoint Reload & Anchor Identity Gate):**
  Loaded `theta_real_v2.npz` matches its saved `sha256` in all 3 seeds, world-model Pearson with exact Bayes is $\ge 0.999$, and at $K=0$ rollouts, `looped_attn_T3` returns $\ell^{(0)}$ with $\mathrm{KL}(K=0) \equiv \mathrm{KL}(\ell^{(0)})$ to machine precision ($< 10^{-7}$).
- **`E1-D1` (Anytime Monotonicity & Strong-Root Safety — Kelvin's Question 1):**
  1. **Sparse-Root Anytime Curve (`M_0 = 2` anchors, $K \in \{0, 1, 2, 4, 8, 16, 24, 32\}$ under `P-FullSupport` / progressive prefix ordering):** For `looped_attn_T3`, mean test KL to exact Bayes is **monotonically non-increasing across every consecutive step** $K_m \to K_{m+1}$ ($\mathrm{KL}(K_{m+1}) \le \mathrm{KL}(K_m) + 0.0005$ in all 3 seeds, with $\mathrm{KL}(32) \le 0.033$ nats and $\mathrm{JSD}(24) \le 0.0075$).
  2. **Strong-Root Safety (`M_0 = 18` clean anchors, $\mathrm{KL}(0) \le 0.050$, plus $K \in \{1, 2, 4, 8, 16, 24\}$ skewed noisy reads with $\tau_r \in \{0.3, 1.5, 2.5\}$):**
     - Naive `ataraxos_mmd` and `mix` **degrade** the strong root belief at $K=16$ ($\mathrm{KL}_{\text{mmd}}(16) - \mathrm{KL}(0) \ge +0.015$ nats, 95% CI $> 0$).
     - Anchor-aware `looped_attn_T3` **never degrades the strong root at any $K \in \{1, 2, 4, 8, 16, 24\}$** ($\max_{K \ge 1} [\mathrm{KL}(K) - \mathrm{KL}(0)] \le +0.0005$ nats across 3 seeds) AND achieves strictly lower KL at $K=16$ than at $K=0$ ($\mathrm{KL}(0) - \mathrm{KL}(16) > 0$ with 95% CI $> 0$, and paired $\mathrm{KL}_{\text{mmd}}(16) - \mathrm{KL}_{\text{looped\_attn\_T3}}(16) > 0$ with 95% CI $> 0$).
- **`E1-D2` (Precision-Aware `jacobi_lin_T4_prec` & Mixed-Tier `econ_kc_K3`):**
  1. **`jacobi_lin_T4_prec` ($814$ params):** On `P-FullSupport`, $\mathrm{KL} \le 0.040$ (beating `looped_lin_T4` and `kc_K1` in all 3 seeds, 95% CI $> 0$), AND on the D3 3-regime probe:
     - Clean deduplication at $\tau=0.3$: **$W(16)/W(1) \le 1.25$** (95% CI upper bound $\le 1.30$),
     - Strict noise monotonicity at $N=8$: **$W(8)/W(1)_{\tau=0.3} < W(8)/W(1)_{\tau=1.5} < W(8)/W(1)_{\tau=2.5}$** in every seed, with $W(16)/W(1)_{\tau=2.5} \ge 2.0$.
  2. **Mixed-tier `econ_kc_K3` & PUCT $2\times 2$ table:** Fitting $(u, \lambda)$ on mixed-tier D3 calibration probes gives $W(16)/W(1)_{\tau=0.3} \le 1.25$ and $W(16)/W(1)_{\tau=2.5} \ge 2.2$ while cutting `D3` MSE below `kc_K3` (95% CI $> 0$); and the $2\times 2$ PUCT table is reported across all 4 cells.

---

## Priority 2 (`R8-E2`): `E-SUD-ANYTIME-BEAM` — Full-Budget Stochastic Beam Completion & Anytime Monotonicity on `Sudoku-Extreme`

- **Executor:** `calibrated-ai` (reusing `R4-E2` checkpoints `s{101,102,103}` and `R7-E2` `Sudoku-Extreme` splits; single L4 GPU, ~25 min runtime, est. **$\$0.55$**, hard cap **$\$0.90$**).

### 2.1 Root-Cause Diagnosis from `R7-E2`

In `R7-E2`, Phase A of `P8_beam` ($L \in \{3, 4\}$ level-synchronous stall-branching waves of width $W=16$) succeeded: it kept the true solution path inside the top-16 beam across all $L$ waves on **$83.3\%$ (`91% / 71% / 88%`) of test boards** while spending only **$15\text{--}33\%$ (`145–329` passes)** of the `982`-pass budget, and made `P8_beam` the first search method on `Sudoku-Extreme` to beat `P3` and `P6` on every metric in every seed (`44.5%` vs. `40.2%` and `38.9%` witnesses; `-40.8%` KL vs. `P0`).

However, inspection of `code/esud9b.py` (lines 234–288) and `AUDIT-R7-E2.md` revealed **three mechanical bugs in Phase B (`alloc_chains` + `complete`)** that caused `P8_beam` to lose the true path on nearly half of the $83.3\%$ of boards where Phase A preserved it:
1. **Per-Landmark `lead = (r == 0)` Forced Deterministic Greedy `argmax` on Landmarks #2–#15 (`esud9b.py` line 253 & 276):**
   In `alloc_chains`, `lead = (r == 0)` was set *for the first chain of every Landmark*, and in `complete()`, `lead=True` forced deterministic `d = int(q.argmax())` even at high-entropy stalls ($H \ge 0.25$). Because Landmarks #2–#15 received only `k = 1` chain, **their single completion chain was a 100% deterministic greedy `argmax` rollout that never sampled at Stalls 5–7!** Furthermore, even for `r > 0`, `complete()` used `P3_thresh` (deterministic `argmax` whenever $H < 0.25$), which `R7-E2` showed is **$-3.2\text{ pp}$ worse** than standard `P3` `'conf'` stochastic rollouts (`37.0% vs 40.2%`).
2. **Conservative `Emax` Division Left $22\text{--}45\%$ of the Pass Budget Unspent (`541–764` of `982` passes):**
   Dividing remaining passes by $\max_{j} E(s_j) + 1$ capped total completion chains at $R = 12.8\text{--}19.3$ (fewer than $M=16$ on $36\text{--}48\%$ of boards in seeds 101–102, dropping surviving beam Landmarks entirely!) while leaving `220–440` passes unused per board.
3. **Unnormalized $\exp(\text{logP})$ Allocation Penalized Deeper Landmarks (Stiffness Trap):**
   Allocating extra chains by $\exp(\text{logP})$ concentrated chains on Landmark #0/#1 (often a wrong branch that triggered a low-entropy cascade with high cumulative $\log P$) and starved Landmarks #2–#15.

### 2.2 Experimental Design (`P8_full` & Anytime Monotonicity)

1. **`P8_full` (Phase-A Multi-Stall Beam + Full-Budget Balanced Stochastic `'conf'` Completion):**
   - **Phase A (Landmark Waves):** Run $L_{\text{waves}} \in \{3, 4\}$ level-synchronous stall-branching waves ($W = 16$, branching on $\{d : q_{z^*, d} > 0.05\}$ at $z^* = \arg\min_c (H(q_c) - \delta \cdot \text{band}_c)$ when $\min_c H(q_c) \ge 0.25$), ranking wave children by **depth-normalized log-probability** $\overline{\log P}_n = \text{logP}_n / \max(1, \text{depth}_n)$ (or cumulative $\text{logP}_n$ when comparing same-wave nodes; compare both on `val`).
   - **Phase B (Full-Budget Balanced Stochastic `'conf'` Completion):**
     Given the $M \le 16$ surviving Landmarks $(s_1, \dots, s_M)$ ordered by $\overline{\log P}_j$ and remaining pass budget $B_{\text{rem}} = 16(E_b + 1) - \text{wave\_passes}_b$:
     - Allocate completion chains across $j \in \{1, \dots, M\}$ using **tempered round-robin / balanced allocation**: in pass 1 over $j = 1 \dots M$, give 1 chain to each Landmark $s_j$ while $\sum \text{cost} + (E(s_j)+1) \le B_{\text{rem}}$; in subsequent passes, allocate chains proportional to tempered weights $w_j \propto \exp(\overline{\log P}_j / \tau_{\text{alloc}})$ ($\tau_{\text{alloc}} \in \{1.0, 2.0, \infty\}$ selected on `val`, where $\tau_{\text{alloc}} = \infty$ is pure round-robin across all $M$ Landmarks) until no Landmark $s_j$ has $E(s_j) + 1 \le B_{\text{rem}} - \text{scheduled\_cost}$.
     - **Crucial fix:** Every completion chain runs **standard `P3` `'conf'` stochastic completion** (`d ~ multinomial(q_{z*})` at the lowest-entropy cell $z^* = \arg\min_c (H(q_c) - \delta \cdot \text{band}_c)$, **with `lead = False` for all chains**—no greedy lock-in and no `P3_thresh` truncation!).
     - Because each Wave-$L$ Landmark has only $E(s_j) \approx 25\text{--}32$ empty cells remaining (vs. $E_b \approx 60$ at the root), the remaining $\sim 750$ passes fund **$R_b \approx 24\text{--}30$ full stochastic `'conf'` completions** per board ($\sim 1.6\text{--}2.0$ independent stochastic chains for *every* surviving Landmark in the 16-beam, spending **$96\text{--}100\%$** of the $16(E_b+1)$ budget!).
   - Also include a **2-Stage Beam + Stochastic Completion arm (`P8_2stage`)** on `val`: after Wave $L_1 = 3$, run 1 stochastic step + forced propagation per child and score each candidate Landmark by the model's own partial-board check head $\text{chk}_{\text{model}}(s_j)$ (or min cell max-prob) before allocating the full-budget `'conf'` completion chains. Select between `P8_full` and `P8_2stage` on `val` by `vote_chk` solution-free proxy or `cal` CE before touching `test`.

2. **Anytime Monotonicity Curve ($B / (E_b + 1) \in \{0, 1, 2, 4, 8, 16, 32\}$ — Kelvin's Question 1):**
   - Evaluate empty-cell KL to the true solution and board witness rate across compute budgets $B_m = m \cdot (E_b + 1)$ for $m \in \{0, 1, 2, 4, 8, 16, 32\}$ (where $m=0$ is 1-pass `P0`, and for $m \ge 1$ we take the first $K(B_m)$ leaves completed within $B_m$ passes).
   - Compare unweighted `mix` (which degrades worse than `P0` at small $m$) against prior-anchored check-gated aggregation $\hat{p}_{\text{safe}}(B_m)$ (`chk_mix` and `vote_chk` with $\lambda > 0$ fit on `cal`).

### 2.3 Pre-Registered Decision Rules (`R8-E2`)

- **`E2-D1` (Full-Budget Beam Completion Beats `P3`, `P6`, and `R7`'s Starved `P8_beam` at Equal $16(E_b+1)$ Budget):**
  1. **Budget utilization & zero overrun:** `P8_full` never exceeds $16(E_b+1)$ passes on any board (`max_passes_over_budget <= 0`) and spends a mean of **$\ge 92\%$** of the $16(E_b+1)$ budget across test boards (up from $55\text{--}78\%$ in `R7-E2`).
  2. **Witness discovery (`E2-D1.1`):** `P8_full` increases test witness rate by **$\ge +6.0\text{ pp}$ over flat `P3`** (`40.2%`) and **$\ge +7.0\text{ pp}$ over `P6`** (`38.9%`), AND beats `R7-E2`'s starved `P8_beam` (`44.5%`) by **$\ge +2.0\text{ pp}$**, with all three paired 3-seed 95% CIs strictly $> 0$.
  3. **Empty-cell belief KL (`E2-D1.2`):** Under both `chk_mix` and `vote_chk`, `P8_full` cuts empty-cell KL by **$\ge 10\%$ vs. `P3` `chk_mix`** (`0.766`) and **$\ge 42\%$ vs. 1-pass `P0`** (`1.178`), beating `P3`, `P6`, and `R7`'s `P8_beam` with all paired 95% CIs $> 0$.
- **`E2-D2` (Anytime Monotonicity Guarantee — Kelvin's Question 1):**
  Across budgets $m \in \{0, 1, 2, 4, 8, 16, 32\}$ multiples of $(E_b+1)$:
  1. **Strict Monotonicity:** For `P8_full` with `vote_chk` and `chk_mix`, mean test KL is **strictly decreasing at every budget step** $\mathrm{KL}(m_{k+1}) < \mathrm{KL}(m_k)$ for all $m_k \in \{0, 1, 2, 4, 8, 16, 32\}$ in all 3 seeds (never worse than 1-pass `P0` at any budget $m \ge 1$).
  2. **Unresolved-Board Safety:** Even on the stratum of boards that have **zero witnesses at $m=16$**, `P8_full` (`chk_mix` / `vote_chk`) has lower or equal KL than 1-pass `P0` ($\mathrm{KL}_{\text{unresolved}}(16) \le \mathrm{KL}_{\text{P0, unresolved}}$, paired 95% CI upper bound $\le +0.005$ nats), whereas unweighted `mix` degrades unresolved boards by $> +0.10$ nats.

---

## Priority 3 (`R8-E3`): `E-MS-SAFE-ANCHOR` — Current-Turn Anchor Safety & Epistemic Corruption vs. Conditional Corner Worlds on $6\times 6$ Stratego

- **Executor:** `eurisko` (reusing `R5-E2` checkpoints `/tmp/r5e2/ckpt/s{101,102,103}/primary/` and data splits; single L4 GPU, ~15 min runtime, est. **$\$0.35$**, hard cap **$\$0.50$**).

### 3.1 Root-Cause Diagnosis from `R7-E3` & Kelvin's Questions (1) & (6)

1. **Why `R5-E2`–`R7-E3` left a gap between the $t-4$ anchor (`0.185` nats) and current-turn one-pass belief $\ell^{(0)}_t$ (`0.047` nats):**
   In `R6-E3` and `R7-E3`, the aggregator took the stale $t-4$ belief $\ell^{(0)}_{t-4}$ (`0.185` nats) as its starting point and used 16 4-ply rollouts to reach `0.096` nats (`-48.2%`), while never being given $\ell^{(0)}_t$ (`0.047` nats). When `R5-E2` replaced $\ell^{(0)}_t$ with raw rollout survival counts, sampling noise on 8 pieces degraded KL from `0.047` to `0.539` nats.
   - **Fix (`Current-Turn Safe Anchor`):** Set the residual starting point to the **strong current-turn one-pass belief $\ell^{(0)}_t$** (`KL = 0.0472` nats), include an anchor token $r=0$ representing $\ell^{(0)}_t$ (or pass $\ell^{(0)}_t$ into the per-slot residual gate of `attn_L3-K3c` / `looped_attn_T3-K3c` initialized so $K=0 \implies \hat{\ell} = \ell^{(0)}_t$), and feed $K \in \{0, 1, 2, 4, 8, 16\}$ 4-ply consistency rollouts (each weighted by its trajectory log-likelihood $\mathrm{LL}_{\text{window}}(Z^{(r)})$ on the observed moves in $[t-4, t]$).
2. **Why `trust_gate` / `trust_pool` beat `stiff_pool` by $63.9\%$ in `R7-E3` (`0.0139 vs 0.0388`) but tied `unif_pool` (`0.0137`):**
   - `R7-E3`'s bias-variance decomposition proved two facts:
     - First, in clean posterior determinizations $Z^{(r)} \sim q_{\text{real}}(\cdot \mid h_{t-4})$, a rollout that hits an early Flag capture is a **legitimate conditional corner world**, not an epistemic model error—downweighting it shifts the mean away from $Q^*$ (`Bias^2` rose `0.0093 -> 0.0102`), whereas weighting worlds by their **Bayes trajectory consistency** $w_r^{\text{cons}} \propto \exp(\tau_{\text{cons}}^{-1} \mathrm{LL}_{\text{window}}(Z^{(r)}))$ weights each world by how well $Z^{(r)}$ explains the observed 4-ply history!
     - Second, `R7-E3`'s `exploratory` tier only swapped 1 piece type and still evaluated the exact depth-4 `primary` value head on legal moves, so its true rollout MSE (`0.0476`) was virtually identical to the clean tier (`0.0453`). When a bag instead mixes clean depth-4 rollouts with genuinely **epistemically noisy / corrupted rollouts** (e.g., 1-ply myopic / random-playout rollouts or off-manifold corrupted determinizations where `ply_lp`, `cone_lp`, and value-consistency features flag that the rollout itself was noisy, yielding single-rollout MSE $\approx 0.12\text{--}0.18$ while occasionally producing stiff $H < 0.15$ claims), $\theta_{\text{trust}}$ separates epistemic noise from valid high-confidence rollouts and beats both `unif_pool` and `stiff_pool`.

### 3.2 Pre-Registered Decision Rules (`R8-E3`)

- **`E3-D1` (Current-Turn Anchor Safety & Anytime Monotonicity — Kelvin's Question 1):**
  Anchored at the **current-turn one-pass belief $\ell^{(0)}_t$** ($\mathrm{KL}(0) \approx 0.0472$ nats) across rollout counts $K \in \{0, 1, 2, 4, 8, 16\}$ under Left/Right preference skew:
  1. **Naive baselines degrade the strong anchor:** Both `raw_count` replacement and fixed-step `ataraxos_mmd` ($\gamma = 0.25$) degrade KL relative to $\ell^{(0)}_t$ ($\mathrm{KL}(16) > \mathrm{KL}(0) + 0.010$ nats, 95% CI $> 0$).
  2. **`attn_L3-K3c` (and `looped_attn_T3-K3c`) never degrade $\ell^{(0)}_t$ and strictly improve it:** Across $K \in \{1, 2, 4, 8, 16\}$, $\max_{K \ge 1}[\mathrm{KL}(K) - \mathrm{KL}(0)] \le +0.0005$ nats in every seed, $\mathrm{KL}(16) < \mathrm{KL}(0)$ with paired 3-seed 95% CI strictly $< 0$, AND at $K=16$ `attn_L3-K3c` achieves Left/Right belief $\mathrm{JSD}$ **$\ge 8\times$ lower** than `scaled_mix` and `ataraxos_mmd` while beating `scaled_mix`, `ataraxos_mmd`, and `kc_dedup(K3c)` in exact-posterior KL (all paired 95% CIs $> 0$).
- **`E3-D2` (Conditional Corner Worlds vs. Epistemic Rollout Corruption — Kelvin's Question 6):**
  1. **Clean Determinization Bags (Window Consistency vs. Stiffness):** On homogeneous 16-rollout bags $Z^{(r)} \sim q_{\text{real}}(\cdot \mid h_{t-4})$, Bayes window-consistency pooling (`cons_pool`: $w_r \propto \exp(\tau_{\text{cons}}^{-1} \mathrm{LL}_{\text{window}}(Z^{(r)}))$, $\tau_{\text{cons}}$ fit on `cal`) achieves strictly lower root action-value MSE and lower $\mathrm{Bias}^2$ than `unif_pool`, `stiff_pool`, and `trust_pool` (paired 95% CI on $\mathrm{MSE}_{\text{unif}} - \mathrm{MSE}_{\text{cons\_pool}} > 0$ and $\mathrm{MSE}_{\text{stiff}} - \mathrm{MSE}_{\text{cons\_pool}} > 0$, cutting `stiff_pool` MSE by $\ge 60\%$).
  2. **Mixed Epistemic-Noise Bags (`10` clean depth-4 rollouts + `6` epistemically noisy 1-ply/high-temperature $\tau_{\text{ply}}=3.5$ rollouts with random piece swaps, tier label hidden from all aggregators):**
     - $\theta_{\text{trust}}$ (trained on `cal` excess surprise using stop-grad trust + `cone_lp` + `ply_lp` features, no tier label) achieves Spearman $\ge +0.40$ with $-\mathrm{KL}(Q^* \Vert q_r)$ overall and within $(s,a)$ pairs, beating stiffness $-H(q)$ by $\ge +0.60$ overall and $\ge +0.20$ within entropy strata.
     - Both `trust_gate` (dropping bottom-$M$ by $\beta_r$) and combined `cons_trust_pool` ($w_r \propto w_r^{\text{cons}} \cdot \beta_r^\alpha$, tuned on `cal`) beat **`unif_pool` by $\ge 20\%$ MSE** and **`stiff_pool` by $\ge 60\%$ MSE** on test (all paired 3-seed 95% CIs $> 0$), while on clean-only bags `cons_trust_pool` never underperforms `unif_pool` ($\mathrm{MSE}_{\text{cons\_trust}} \le \mathrm{MSE}_{\text{unif}}$).

---

## Summary Table of Round-8 Experiments

| Priority | ID | Experiment Name | Executor | World-Model Training | Est. Cost (Hard Cap) | Core Question Answered |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `R8-E1` | `E-INFO-ANYTIME` ($6\times 6$ ShipHunt) | `eurisko` | **None** (loads `theta_real_v2.npz`) | $\$0.45$ ($\$0.80$) | Strong-root (`18`-anchor) & sparse-root anytime monotonicity ($K=0\dots 32$); precision-aware `jacobi_lin_T4_prec` ($W(16)/W(1) \le 1.25$); mixed-tier `econ_kc_K3`; $2\times 2$ PUCT attribution. |
| **2** | `R8-E2` | `E-SUD-ANYTIME-BEAM` (`Sudoku-Extreme`) | `calibrated-ai` | **None** (loads `R4-E2` nets) | $\$0.55$ ($\$0.90$) | Fixes `R7-E2` Phase-B completion starvation (`55–78% -> 95%+` budget) & `lead=True` greedy lock-in via balanced `'conf'` rollouts; anytime monotonicity curve ($B = 0\dots 32 \times (E+1)$). |
| **3** | `R8-E3` | `E-MS-SAFE-ANCHOR` ($6\times 6$ Stratego) | `eurisko` | **None** (loads `R5-E2` nets) | $\$0.35$ ($\$0.50$) | Current-turn $\ell^{(0)}_t$ (`0.047` nats) anchor safety ($K=0\dots 16$ never degrades $\ell^{(0)}_t$); separates valid conditional corner worlds (`cons_pool`) from epistemic noise (`trust_gate` / `cons_trust_pool`). |
