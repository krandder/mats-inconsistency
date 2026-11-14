# Literature Monograph 10: *InfiLoop — Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals* (`arXiv:2610.11570v1`, Li et al., Oct 2026) and Its Relationship to Label-Free Inconsistency Search & LIR

- **Paper:** Pengxiang Li, Dilxat Muhtar, Di He, Guinan Su, Lu Yin, and Shiwei Liu (Oct 2026), *"Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals (`InfiLoop`)"*, [`arXiv:2610.11570v1`](https://arxiv.org/abs/2610.11570).
- **Official Public Repository:** [`https://github.com/pixeli99/InfiLoop`](https://github.com/pixeli99/InfiLoop) (`sudoku/model.py`, 166 lines; `sudoku/run.py`, 219 lines).
- **Preceding Lineage Cited by the Paper:**
  - Tiny Recursive Model (`TRM`, Jolicoeur-Martineau 2025, `arXiv:2510.04871`) & Hierarchical Reasoning Model (`HRM`, Wang et al. 2025, `arXiv:2506.21734`)
  - Fractional / Fixed-Point Recursive Models (`FPRM`, 2026, `arXiv:2606.18206`) & Probabilistic TRM (`arXiv:2605.19943`)
  - Attention Residuals (`AttnRes`, Kimi Team 2026, `arXiv:2603.15031`)
  - Looped Transformers (`Giannou et al. 2023`; `Saunshi et al. 2025`)

---

## 1. What `InfiLoop` (`arXiv:2610.11570v1`) Diagnoses and Builds

### 1.1 The "Latent Overthinking" Diagnosis in Carry-Last Recursive Transformers (`TRM-Att`)

Standard recursive reasoning models (`TRM-Att`, `arXiv:2510.04871`) refine an answer state $y$ and a latent reasoning state $z$ across $S$ detached deep-supervision segments, each containing $H$ outer cycles and $L$ inner reasoning steps of a shared $N _ {\text{blocks}}$-layer Transformer block $f_\theta$:

$$z^{(s, h, l)} = f_\theta\big(z^{(s, h, l-1)}, y^{(s, h-1)} + x\big), \quad l = 1, \dots, L, \qquad y^{(s, h)} = f_\theta\big(y^{(s, h-1)}, z^{(s, h, L)}\big), \quad h = 1, \dots, H.$$

Crucially, `TRM-Att` uses a **carry-last state transition** along both loops: the new state $h_t = z_t$ completely overwrites $h _ {t-1}$. Li et al. (2026) demonstrate that this carry-last transition causes **latent overthinking** when test-time iterations are scaled beyond the training horizon:
- In a controlled `72`-layer training setup on `Sudoku-Extreme` ($S _ {\text{train}}=3, H=2, L=4, N _ {\text{blocks}}=2$), `TRM-Att` improves from `40.4%` at `72` layers to a peak of `62.2%` at `468` layers ($S=20$), and then **degrades to `58.1%` at `1,872` layers** ($S=78$).
- Tracking individual cells and puzzles between outer updates `36` (`432` layers) and `72` (`864` layers), `TRM-Att` **overwrites `20.9%` of already-correct cell predictions** (`18.1%` net cell regression vs. `2.8%` newly fixed cells) and **loses `19.7%` of puzzles that were already completely solved at update `36`** (`Figure 3a–b`).
- Even in the full `504`-layer training regime ($S _ {\text{train}}=12, H=3, L=6, N _ {\text{blocks}}=2$, `6.83M` parameters), `TRM-Att` reaches `74.7%` at `504` layers, peaks at `80.1%` at `2,520` layers ($S=60$), and **degrades to `78.5%` at `4,032` layers** ($S=96$).

### 1.2 Why Feed-Forward Attention Residuals (`AttnRes`) Collapse in Looped Extrapolation

Kimi Team (2026, `arXiv:2603.15031`) introduced **Attention Residuals (`AttnRes`)** for fixed-depth feed-forward Transformers, replacing additive residuals with softmax attention over all preceding layer outputs $\lbrace z_0, \dots, z_t\rbrace $ using a learned layer-specific pseudo-query $q_t \in \mathbb{R}^d$:

$$h_t = \sum _ {i=0}^t \frac{\exp(s _ {t,i})}{\sum _ {j=0}^t \exp(s _ {t,j})} z_i, \qquad s _ {t,i} = q_t^\top \mathrm{RMSNorm}(z_i).$$

Li et al. show that naively transplanting `Full AttnRes` or `Block AttnRes` into a weight-tied looped Transformer fails (`40.4%` for `Full AttnRes` and `57.0%` for `Block AttnRes` vs. `75.2%` for `InfiLoop` in the 72-layer test):
1. **Unbounded Denominator Dilution ($O(1/t)$ Step-Size Decay):** Without temporal decay ($\beta = 1$), the denominator $D_t = \sum _ {j=0}^t e_j$ grows linearly with loop count $t$, forcing the effective update gain $\gamma_t = e_t / D_t = O(1/t) \to 0$. Later test-time iterations lose the ability to revise the state (`Full AttnRes` freezes at `40.4%` from 72 to 1,872 layers).
2. **Growing $O(t d)$ History Cache or Periodic Sawtooth Resets:** Caching $\lbrace z_0, \dots, z_t\rbrace $ consumes `21.85 MiB` per puzzle at $S=60$, while resetting every $B=6$ steps (`Block AttnRes`) periodically discards accumulated progress.

### 1.3 `InfiLoop`'s Loop-Native Streaming Attention Residuals (`Connection` in `sudoku/model.py`)

`InfiLoop` replaces carry-last transitions on both the inner loop ($L$) and outer loop ($H$) with a **constant-memory $O(d)$ streaming attention-residual recurrence** (`Connection` class in `sudoku/model.py`, lines 102–121):
1. **Shared Content-Scoring Pseudo-Query ($q \in \mathbb{R}^d$):** Because all recurrent steps share the same representation space, a single loop-invariant parameter vector $q \in \mathbb{R}^d$ (`self.weight`, initialized to `0`, `512` floats) scores each candidate state $z_t \in \mathbb{R}^d$ per token:
   $$s_t = \mathrm{clip}\big(q^\top \mathrm{RMSNorm}(z_t), -c, c\big), \quad c = 8, \qquad e_t = \exp(s_t) \in [e^{-8}, e^8].$$
2. **Discounted Streaming Numerator & Denominator ($\beta \in (0, 1)$):** A single learned scalar logit (`self.beta_logit`, initialized at $\beta_0 = 0.55$, converging to $\beta _ {\text{outer}} \approx 0.440$ and $\beta _ {\text{inner}} \approx 0.445$) maintains running numerator $N_t \in \mathbb{R}^d$ and scalar denominator $D_t \in \mathbb{R}_+$ per token:
   $$N_t = \beta N _ {t-1} + e_t z_t, \qquad D_t = \beta D _ {t-1} + e_t, \qquad h_t = \frac{N_t}{D_t} = (1 - \gamma_t) h _ {t-1} + \gamma_t z_t, \qquad \gamma_t = \frac{e_t}{\beta D _ {t-1} + e_t}.$$
   - **Inner Loop (`self.inner`):** Reset at the start of every outer cycle ($N_0^{\text{in}} = e_0^{\text{in}} z_0, D_0^{\text{in}} = e_0^{\text{in}}$) and updated across $l = 1 \dots L$.
   - **Outer Loop (`self.outer`):** Initialized once at $t=0$ ($N_0^{\text{out}} = e_0^{\text{out}} y_0, D_0^{\text{out}} = e_0^{\text{out}}$) and **carried persistently across all outer cycles $h = 1 \dots H$ and across all test-time segments $s = 1 \dots S _ {\text{eval}}$** (`state = (high, low, numerator, denominator)`).
3. **Additional Architectural Details in `sudoku/model.py` Worth Noting:**
   - **Grid Depthwise 2D Convolution Inside `Block` (`lines 91–97`):** Before adding the clue/state injection and applying the 2 RoPE+SwiGLU Transformer layers, `Block.forward` reshapes the 81 board tokens (`state[:, 16:]`) into a $9\times 9$ grid and applies a $3\times 3$ depthwise 2D convolution (`nn.Conv2d(512, 512, 3, padding=1, groups=512, bias=False)`), injecting explicit 2D spatial locality alongside global self-attention.
   - **`Stablemax` Supervised Cross-Entropy Loss (`lines 162–166`):** Instead of standard `softmax`, training uses `stablemax` ($g(x) = x + 1$ for $x \ge 0$, $1/(1-x)$ for $x < 0$) against the ground-truth solved board labels $y^\star$, plus sign-gradient updates (`prefix.grad.sign()`) on the first of 16 prefix tokens.
4. **Headline Empirical Results (`Table 1`):**
   - Adding `InfiLoop` adds just **`+1,026` parameters** (`513` inner + `513` outer) to the `6.83M`-parameter `TRM-Att` backbone (`582.8 KiB` state per puzzle) and lifts:
     * **`Sudoku-Extreme`:** `74.7%` (`TRM-Att`) / `87.4%` (`TRM-MLP`) / `94.2%` (`FPRM`) $\to$ **`95.2%`** at `504` training layers ($S=12$) and **`97.9%`** at `15,120` test-time layers ($S=360$), remaining stable at `97.8%` through `20,160` layers ($S=480$).
     * **`Maze-Hard` (`30x30`):** `85.3%` (`TRM-Att`) $\to$ **`87.4%`** (at `2,688` layers, continuing to `24,960` layers).
     * **`ARC-AGI-1` / `ARC-AGI-2` (`pass@2`):** `44.6%` / `7.8%` (`TRM-Att`) $\to$ **`49.1%` / `13.6%`** (`InfiLoop`).

---

## 2. Exact Mathematical & Architectural Relationship to Our Research (`GLC-AOM` & `LIR`)

### A. Three Striking Convergences That Independently Validate Our Core Thesis

| Dimension | `InfiLoop` (`arXiv:2610.11570v1`, Li et al. Oct 2026) | Our Program (`GLC-AOM`, `looped_attn_T3` / `jacobi_lin_T4_prec`, `LIR`) | Why the Convergence Matters |
| :--- | :--- | :--- | :--- |
| **1. "Latent Overthinking" vs. Strong-Root Degradation & Anytime Monotonicity** | Proves that along the *recurrent loop axis*, carry-last `TRM-Att` overwrites `20.9%` of already-correct cells and loses `19.7%` of already-solved puzzles between updates `36` and `72`, causing accuracy to drop after peaking (`§3.1`, `Fig. 3`). | In our audited **Rounds 8–9** (`R8-E1`/`R9-E1` strong-root `ShipHunt` `D1.1/D1.2` PASS; `R8-E2`/`R9-E2` `Sudoku-Extreme` `36/36` anytime monotonicity `D2.1/D2.2` PASS), we proved the exact same phenomenon across *rollout budgets*: without an anchor-preserving, self-trust-weighted aggregator, adding more rollouts overwrites strong roots and degrades solved states! | Both lines of work identify the **same fundamental pathology of test-time compute scaling**: un-anchored state replacement or naive averaging destroys already-resolved beliefs when extra compute steps are run. |
| **2. Emergent Content Score $s_t = \ln e_t$ vs. Explicit Self-Consistency Gate $m _ {\text{ql}} \cdot \beta _ {\text{trust}}$** | In `§4.5` (`Fig. 4a–b`), `InfiLoop` discovers an *"emergent self-verification behavior"*: the learned content score $s_t = q^\top \mathrm{RMSNorm}(z_t)$ tracks the number of board constraint errors (`+7.25` at `0` errors $\to$ `+4.19` at `10` errors $\to$ `+2.75` at `20` errors $\to$ `-0.58` at `33` errors). When a puzzle is solved ($0$ errors), $e_t = \exp(s_t)$ spikes ($\gamma_t = 0.91$), **inflating $D_t$ to lock in the solution** so subsequent noisy steps have tiny weight (`0.3%` cell turnover)! | In our audited **`R9-E2` `D2.2` PASS**, the internal leaf check-consistency read $m _ {\text{ql}} = \min _ {c \in E_b} q _ {\text{leaf}}(c, Z_c)$ separates valid from invalid Sudoku leaves with **`AUROC = 0.99998`** (median `0.12` vs. `0.004`), gating precision $\beta_r = \beta _ {\text{trust}} \cdot m _ {\text{ql}}$ so valid witnesses lock in (`0.0%` false accepts) and unresolved KL drift is `0.0000`! | `InfiLoop`'s empirical discovery in `Fig. 4a–b` is a direct, independent validation of our **consistency-weighted precision pooling law**: an iterative solver scales monotonically in test-time compute if and only if each step's aggregation weight $e_t$ is a monotone function of its **internal constraint consistency**! |
| **3. Bounded Effective Step Gain vs. $O(1/t)$ Running-Mean Stagnation** | `Theorem 1(ii)` proves that discounting past accumulator mass by $\beta \in (0, 1)$ keeps the effective step size $\gamma_t = \frac{e_t}{\beta D _ {t-1} + e_t}$ bounded away from $0$ and $1$ as $t \to \infty$, whereas unweighted $\beta=1$ (`Full AttnRes`) decays as $O(1/t)$ and stalls at `40.4%`. | In `R9-E1` (`jacobi_lin_T4_prec`), our precision-aware ridge $\lambda(\beta_r) = \mathrm{softplus}(a - b\beta_r)$ and three-regime effective weight law $W _ {\text{eff}}(N, \rho, \beta) = \frac{N\beta}{\lambda + (1+(N-1)\rho)\beta}$ prevent both $1/N$ dilution of high-precision witnesses and unbounded blowup of redundant reads. | Both show that raw softmax/mean accumulation over growing horizons fails without explicit precision-vs-prior normalization. |

---

### B. Four Structural Limitations of `InfiLoop` Where Our Framework Goes Further

Following our non-sycophantic technical standard, we identify four sharp mathematical limitations in `InfiLoop` that directly motivate our architecture:

1. **Critique 1 — `InfiLoop`'s Self-Verification Score $s_t$ Only Emerges Because It Is Trained with Ground-Truth Solved Boards $y^\star$:**
   - Why does `InfiLoop`'s single linear vector $q \in \mathbb{R}^d$ (`s_t = q^\top \mathrm{RMSNorm}(z_t)`) assign `+7.25` to zero-error Sudoku boards and `-0.58` to 33-error boards?
   - **Because `InfiLoop` is trained end-to-end with supervised `stablemax` Cross-Entropy against ground-truth solved boards $y^\star$ (`sudoku/model.py` lines 162–166) across `78,125` steps!** During supervised training, gradient descent on $\mathcal{L} _ {\text{CE}}(\text{head}(h_t), y^\star)$ pushes $q$ to up-weight whichever intermediate state $z_i$ lies closest to the oracle label $y^\star$.
   - If ground-truth solution labels $y^\star$ are unavailable during training, `InfiLoop`'s pseudo-query $q$ receives **zero gradient** and remains at its initialization $q = \mathbf{0}$ ($e_t \equiv 1$), collapsing to an un-gated exponential moving average!
   - Furthermore, on **multi-solution problems** (such as `N-Queens` where $10\times 10$ boards have up to $724$ solutions and $16\times 16$ boards have $14.8\text{ million}$, or under-constrained Sudoku/SAT), supervising a single deterministic `InfiLoop` trajectory against one solution label $y^\star$ drives $z_t$ toward the invalid average of solutions (`arXiv:2610.18649` Appendix C).
   - **Our Resolution (Label-Free Inconsistency Minimisation):** In our framework, the step/leaf weight $e_t$ does not need oracle labels $y^\star$ to discover consistency: it is driven directly by **world-model conditional inconsistency energy** $\mathcal{E}(z_t; \theta _ {\text{real}})$ and **excess-surprise self-trust** $\theta _ {\text{trust}}$ (`LIR` mask $(1, 0)$). A board with 0 constraint violations has $\mathcal{E}(z_t; \theta _ {\text{real}}) = 0$ by definition of the world model's rules, without ever seeing a pre-solved board $y^\star$!

2. **Critique 2 — Oblivious Scalar Age Decay ($\beta^{t-i}$) vs. Key-Space Covariance Deduplication ($(K^\top B K + \lambda I)^{-1}$):**
   - Look at `InfiLoop`'s effective weight on past step $i$ at step $t$:
     $$\alpha _ {t,i} = \frac{\beta^{t-i} e_i}{\sum _ {j=0}^t \beta^{t-j} e_j}, \qquad \beta _ {\text{outer}} \approx 0.440, \quad \beta _ {\text{inner}} \approx 0.445.$$
   - Because $\beta \approx 0.44$, `InfiLoop`'s memory decays **exponentially with wall-clock step lag $t - i$**: a state from 5 steps ago is discounted by $\beta^5 \approx 0.0165$ (`98.3%` forgotten) unless its scalar score $s_i$ was $\ge 4$ nats higher than current states!
   - More fundamentally, `InfiLoop` uses a **1D scalar weight $e_i \in \mathbb{R}_+$ (with a single fixed query $q \in \mathbb{R}^d$) and no key-key covariance matrix**:
     * Suppose step $t-6$ deduced an independent constraint region $A$ (key $k _ {t-6}$), while steps $t-3, t-2, t-1, t$ repeatedly re-evaluate an overlapping region $B$ ($k _ {t-3} \approx k _ {t-2} \approx k _ {t-1} \approx k_t$, pairwise overlap $\rho \to 1$).
     * Scalar exponential decay $\beta^{t-i}$ **forgets region $A$** while **over-counting the 4 collinear updates on region $B$**!
     * By contrast, preconditioned attention clearing (`looped_attn_T3` / `jacobi_lin_T4_prec`, solving $(K^\top B K + \lambda I) w = K^\top B v$) discounts updates by **informational redundancy $\langle k_i, k_j \rangle$**, not wall-clock age alone (`R9-E1` `D2.2` PASS).

3. **Critique 3 — Single-Trajectory Token-Wise Convex Hull vs. Disjunctive Branching & Nogood Exclusion:**
   - `InfiLoop` Theorem 1(i) proves that at every token position $v \in \lbrace 1..81\rbrace $, $h _ {t,v} = \sum _ {i=0}^t \alpha _ {t,i,v} z _ {i,v}$ lies strictly inside the convex hull $\mathrm{conv}(z _ {0,v}, \dots, z _ {t,v})$ of a **single continuous trajectory**.
   - On hard combinatorial problems requiring **disjunctive case-splitting** (e.g., when unit propagation stalls on `G4SATBench` hard 3-SAT or multi-branch strategic games, and the solver must test Hypothesis $A$: $x_v = 1$ vs. Hypothesis $B$: $x_v = 0$), a single continuous trajectory that averages $z _ {i,v}$ across incompatible hypotheses creates a fractional mid-point in latent space rather than maintaining separate conditioned witness branches $\lbrace S_1, \dots, S_M\rbrace $ with conflict-clause (`nogood`) exclusion.

4. **Critique 4 — Task-Specific Fixed Weights with Zero Rule-Belief $\mu(R _ {\text{rule}})$ or Utility Independence (`LIR`):**
   - `InfiLoop` trains separate model checkpoints for `Sudoku-Extreme` (`6.83M` params with a hardcoded `9x9` $3\times 3$ depthwise `Conv2d` in `sudoku/model.py` line 87!), `Maze-Hard`, and `ARC-AGI`. It does not maintain an explicit posterior $\mu(X _ {\text{ans}}, Z _ {\text{hid}}, R _ {\text{rule}})$ over unknown in-context rules, does not perform active Value-of-Information (`VoI`) probing to disambiguate hidden rules, and has no LIR mask separating epistemic beliefs from utility $U$.

---

## 3. Concrete Actionable Integration into Our Reoriented Roadmap (`R27`)

`InfiLoop` strengthens our reoriented `TRM`-family research program in two immediate ways:

1. **Direct Public Baseline in the `TRM` Comparison Class (`Section 6` of `research-direction-note-v7`):**
   - Alongside `TRMP` (`arXiv:2610.18649`, `16k–65k` MPNN/Factor-Graph) and `TRM` (`arXiv:2510.04871`, `6.83M` Transformer), `InfiLoop` (`arXiv:2610.11570`, `pixeli99/InfiLoop`, `6.83M` Transformer with `+1,026` streaming `(N_t, D_t)` parameters achieving `97.9%` on `Sudoku-Extreme`) is now the strongest public looped-Transformer baseline on `sapientinc/sudoku-extreme`.
2. **A Clean Architectural Bridge for `R27-CANDIDATE-A`: *Label-Free Inconsistency-Gated Streaming Residuals*:**
   - Look at `InfiLoop`'s `Connection` module (`sudoku/model.py` lines 102–121):
     $$N_t = \beta N _ {t-1} + e_t z_t, \qquad D_t = \beta D _ {t-1} + e_t, \qquad h_t = \frac{N_t}{D_t}.$$
   - In supervised `InfiLoop`, $e_t = \exp(q^\top \mathrm{RMSNorm}(z_t))$ only learns to proxy board consistency because ground-truth solved boards $y^\star$ are provided in the loss.
   - In **`Inconsistency-InfiLoop` / `Inconsistency-TRMP` (`R27-CANDIDATE-A`)**, we can drive the streaming residual weight $e_t$ **directly from the world model's local conditional consistency check** (`LIR` mask $(1, 0)$, with `stop_grad` so halting/weighting never distorts $z_t$):
     $$s_t(v) = -\tau \, \mathcal{E} _ {\text{local}, v}\big(\mathrm{sg}(z_t); \theta _ {\text{world}}\big) + q _ {\text{trust}}^\top \mathrm{RMSNorm}\big(\mathrm{sg}(z_t)\big), \qquad e_t(v) = \exp\big(\mathrm{clip}(s_t(v), -8, 8)\big).$$
   - This gives the exact `InfiLoop` solution-locking property ($e_t$ surges when constraint violations hit $0$, locking $N_t / D_t$ against latent overthinking across thousands of test-time iterations) **with zero ground-truth solution labels $y^\star$ during training**, while pairing it across $k$ parallel/tree-forked chains with `(K^\top B K + \lambda I)^{-1}` covariance deduplication so multiple chains do not collapse onto a single convex hull!
