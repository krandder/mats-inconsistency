# Pre-Launch Design Review: Overnight CA & Sudoku Experiments (2026-10-15)

**Reviewer:** `gemini` (Argon)  
**Target Design:** [`/home/agent/workspaces/mats/outbox/overnight-experiments-design-2026-10-15.md`](file:///home/agent/workspaces/mats/outbox/overnight-experiments-design-2026-10-15.md)  
**Timestamp:** 2026-10-15 00:53 UTC (27 minutes before the 01:20 UTC deadline)

---

## Executive Summary & Ranked Action List

The overall structure of `E-CA` and `E-SUD`, the compute split, and the pre-registration discipline are strong. However, checking the exact transition math on $W=8, T=6$ for Rules 90 and 110 and tracing the equations in lines 32–56 uncovered **five concrete bugs/traps in `E-CA` and two in `E-SUD`**—including a **showstopper bug in `P2`** that would prevent the search from ever chaining across rows, and a **noise ceiling at $\varepsilon = 0.05$** where $93.4\%$ of the 5-step signal is already washed out into thermal noise.

### Ranked Changes (Apply in This Order Before Launch)

1. **[MUST CHANGE — Showstopper in `P2`/`P3`/`P4`/`P5`, Lines 48–55] Fix `P2` so steps accumulate a growing conditioning set $S_t = S_{t-1} \cup \{z_t\}$ instead of contracting single cells $z_a$ into $Y$ in isolation.**
   - *Why:* As written (`pick z ... branch on z ... contract into a claim q_a(Y) ... sum_a w_a (logit q_a(Y) - logit p0)`), each of the $M$ steps only ever evaluates a **depth-1 single-cell split** $E \to z_a \to Y$ and throws $z_a$ away before picking $z_{a+1}$. Worse, in Rule 90 ($x_{t,c} = x_{t-1,c-1} \oplus x_{t-1,c+1}$), a **single** unobserved parent has **zero** marginal mutual information with the child until its partner parent is *simultaneously* in the conditioning set! And under `rbtd8` (`T2`), $\sum_z q(z\mid E)q(Y\mid E,z) \approx q(Y\mid E)$ by construction, making 1-cell branch-and-contract a no-op.
2. **[MUST CHANGE — Thermal Washout, Line 28] Lower CA noise from $\varepsilon = 0.05$ to $\varepsilon = 0.015$ (or $\varepsilon = 0.01$).**
   - *Why:* Exact transition-matrix calculation on $W=8, T=6$ shows that at $\varepsilon = 0.05$, the exact conditional entropy $H(\text{row } 5 \mid \text{row } 0)$ is **`0.647` nats out of $\ln 2 = 0.693$ nats** ($93.4\%$ of the signal is destroyed by noise, leaving only `0.046` nats of headroom between random guessing and the Bayes oracle!). At **$\varepsilon = 0.015$** (or `0.01`), 1-step arcs are genuinely stiff ($H = 0.056\text{--}0.079$ nats), while row 5 retains **`0.29\text{--}0.35` nats** of real multi-step signal ($7.5\times$ larger headroom) and all probabilities stay strictly in $(0,1)$.
3. **[MUST CHANGE — Training Mask OOD & Bridge Symmetry, Lines 32–34] Train with reveal count $k \sim \mathrm{Uniform}\{1, \dots, 47\}$ (not `[0.2, 0.8]`), and include 2 spatial anchor cells in a `(B2)` Bridge query family.**
   - *Why:* On 48 cells, revealing only row 0 (`8/48 = 16.7%`) is **outside `[0.2, 0.8]`**, so `P0` and step-1 of `P2` would be evaluated out-of-distribution on mask count. Also, on $W=8$ periodic Rule 90, revealing *only* row $T-1$ with zero cells in row $T-2$ leaves **0.0%** of row $T-2$ cells locally determined (marginal $= 0.5$ due to even/odd complement symmetry), whereas revealing row $T-1$ **plus 2 adjacent anchor cells** makes backward/sideways triples $(X_{t+1,c}, X_{t,c-1}) \to X_{t,c+1}$ **100.0% deterministic** in Rule 90 and **90.4% deterministic** in Rule 110!
4. **[MUST CHANGE — Untrained Key Head in `P5`, Lines 40–41 & 55] Define $k(i \mid S)$ using the attention-weighted source-token fingerprint (and/or train its projection on multi-arc bags), NOT the raw last-layer hidden state of token $i$.**
   - *Why:* Token $i$'s last-layer hidden state mainly encodes the *output* $(i, \text{logit } q(X_i \mid S), \beta_{\text{int}})$. Two completely independent arcs (one from past parents, one from future children) that predict the same bit for $X_i$ will have $\langle k_{\text{past}}, k_{\text{future}} \rangle \approx 1$ at token $i$'s output layer, causing `P5` to falsely deduplicate independent evidence.
5. **[MUST CHANGE — Stiffness Floor, Lines 39–40] Subtract the uniform-noise baseline from $\beta_{\text{int}}(i \mid S)$.**
   - *Why:* For an uninformative arc ($q = 0.5$, surprise $\hat{s} = \ln 2 \approx 0.693$), $\beta_{\text{int}} = 1/\hat{s} = 1.443 \ne 0$. Use $\beta_{\text{int}}(i \mid S) = \max\!\big(0,\, \frac{1}{\hat{s}(i\mid S) + \epsilon} - \frac{1}{\ln 2 + \epsilon}\big)$ so uninformative arcs get weight **strictly $0$**.
6. **[CHANGE IN `E-SUD`, Line 80] Demote $4\times 4$ Sudoku to a 2-minute smoke test and run all primary metrics on $6\times 6$ Sudoku.**
   - *Why:* $4\times 4$ Sudoku has only **$288$ valid grids in total**; a 4-layer transformer memorizes all 288 grids to $\sim 0$ KL in one pass (repeating the 5×5 ShipHunt trap). $6\times 6$ Sudoku has **$28{,}200{,}960$ valid grids**, cannot be memorized, and exact solver enumeration on partial boards still takes $< 5\text{ ms}$.
7. **[DROP / SIMPLIFY, Line 27] Drop Rule 30 control; use Rule 110 and Rule 90 (or Rule 54) only.**
   - *Why:* Save GPU hours for clean multi-chain `P2`/`P5` sweeps on Rules 110 and 90.

---

## Detailed Mathematical & Design Critique

### 1. Why `P2` As Written (Lines 48–53) Cannot Chain Across Rows—and the Exact Fix

In lines 48–53, `P2` is currently specified as:
> *"repeat M times: pick the masked cell z maximising $E_z[(\mu(Y \mid E, z) - \mu(Y \mid E))^2]$ ... branch on z exactly (both values, weights $q(z \mid E, \dots)$), contract the branch into a claim on Y. Claims pooled with the prior-corrected $\gamma = 0$ pool..."*

There are two fatal issues if each step $a \in \{1..M\}$ branches on a single cell $z_a$ from $E$ and immediately contracts $z_a$ into a claim $q_a(Y) = \sum_{z_a \in \{0,1\}} q(z_a \mid E) q(Y \mid E, z_a)$:
1. **No multi-parent or multi-row composition:**
   - In a 1D CA, cell $X_{t, c}$ is a function of **three** parents $(X_{t-1, c-1}, X_{t-1, c}, X_{t-1, c+1})$.
   - In Rule 90 ($X_{t, c} = X_{t-1, c-1} \oplus X_{t-1, c+1}$), knowing *one* parent $z_1 = X_{t-1, c-1}$ without knowing $z_2 = X_{t-1, c+1}$ gives $P(X_{t, c} = 1 \mid z_1) = 0.500$—**zero information**!
   - Even more importantly, to bridge from row 0 to row 3, you must materialize cells in **row 1**, keep them in the active conditioning context $S_t$ so that cells in **row 2** now have their parents present in $S_t$ (making $q(z_{\text{row } 2} \mid S_t)$ **stiff**!), and then materialize row 2 cells so that $Y$ in **row 3** has its parents present in $S_M$!
   - If you contract $z_1$ into $Y$ immediately using the long-range blurry conditional $q(Y \mid E, z_1)$, you never unlock the stiff 1-step arcs in row 2 or row 3!
2. **Martingale cancellation under `T2` (`rbtd8`):**
   - `T2` specifically trains the network so that $q_\theta(Y \mid E) \approx \mathbb{E}_{z \sim q_\theta(z \mid E)}[q_\theta(Y \mid E, z)]$. Thus a 1-cell branch-and-contract from $E$ to $Y$ produces $q_a(Y) \approx q_0(Y)$—a zero update!

#### How to Fix `P2`, `P3`, `P4`, `P5` (Sequential Materialization Along Stiff Arcs + Multi-Chain Pooling)
To test Kelvin's actual thesis (*"route through the stiff arcs so that beliefs about long-range queries improve with compute"*), each rollout/chain $r \in \{1..K\}$ of length $M$ must **grow the materialized context $S_{t} = S_{t-1} \cup \{z_t\}$**:
- **Within a chain ($t = 1 \dots M$):**
  - Given current materialized context $S_{t-1}$ (starting at $S_0 = E$), score all currently unrevealed cells $z \notin S_{t-1} \cup \{Y\}$ inside a single forward pass of the transformer!
  - Notice that **one forward pass on $S_{t-1}$ simultaneously gives $q_\theta(z \mid S_{t-1})$ and $\beta_{\text{int}}(z \mid S_{t-1})$ for ALL masked cells $z$ at once** (zero need for "two extra forward passes per candidate over 12 candidates" if you use stiffness-guided Refocus, or you can use 1-step stiffness + VoI on top-6 stiffest candidates!).
  - **Refocus selection rules to compare at equal forward-pass budget:**
    - **`P2-Stiff` (Pure LIR Stiffness Propagation):** Pick $z_t = \arg\max_{z \notin S_{t-1} \cup \{Y\}} \beta_{\text{int}}(z \mid S_{t-1})$ (or sample $z_t \propto \exp(\tau \beta_{\text{int}}(z \mid S_{t-1}))$ across chains to explore multiple stiff fronts, e.g. both the forward front from row 0 and the backward/sideways front from row $T-1$!). When $z_t$'s parents are in $S_{t-1}$, $\beta_{\text{int}}(z_t \mid S_{t-1})$ is huge ($\hat{s} \approx 0.08$ nats), so materializing $z_t \sim q_\theta(z_t \mid S_{t-1})$ (or keeping a small beam of width $B=2\text{--}4$) pins $z_t$ with $98.5\%$ accuracy and immediately makes the *next* row's cells stiff at step $t+1$!
    - **`P2-VoI` (Stiff-Gated Goal-Relative VoI):** Among the top-$L$ ($L=6$) cells with highest incoming stiffness $\beta_{\text{int}}(z \mid S_{t-1})$ (so we only branch on cells we can actually determine reliably!), pick $z_t$ maximizing $\mathbb{E}_{z_t \sim q(\cdot \mid S_{t-1})}[(\mu(Y \mid S_{t-1}, z_t) - \mu(Y \mid S_{t-1}))^2]$ (or maximizing the increase in target stiffness $\mathbb{E}[\beta_{\text{int}}(Y \mid S_{t-1}, z_t)]$).
    - **`P4-Rand` (Random Materialization Order):** Pick $z_t$ uniformly at random from masked cells.
- **Pooling across $K$ chains (or across steps $t=1..M$ of the active PDG):**
  - Each chain/arc $a$ that reaches step $M$ (or each intermediate read-out $q_\theta(Y \mid S_t^{(r)})$) emits a claim $q_a(Y) = q_\theta(Y \mid S_M^{(a)})$ with **stiffness $\beta_{\text{int}}(Y \mid S_M^{(a)})$** and **key $k(Y \mid S_M^{(a)})$**.
  - Look at why this makes the PDG aggregator (`P5`/`P6`) genuinely load-bearing on **Bridge queries**:
    - Chain 1 propagates down from row 0 ($t=0 \to 1 \to 2$) and produces a high-$\beta$ forward claim $q_{\text{fwd}}(Y)$ with key $k_{\text{fwd}}$.
    - Chain 2 propagates up/sideways from row $T-1$ and the anchors ($t=5 \to 4 \to 3$) and produces a high-$\beta$ backward claim $q_{\text{bwd}}(Y)$ with orthogonal key $k_{\text{bwd}} \perp k_{\text{fwd}}$.
    - Chain 3 is a second noisy rollout down from row 0 (redundant with Chain 1, so $k_3 \approx k_{\text{fwd}}$).
    - **Prior-corrected $\gamma=0$ pooling + Gram key deduplication (`P5`)** averages Chains 1 and 3 while summing their log-odds with Chain 2!

---

### 2. Exact Numbers on Why $\varepsilon = 0.05$ Is Too High for $T=6$ (Line 28)

We ran the exact $256 \times 256$ row-transition matrix for $W=8, T=6$ across $\varepsilon \in \{0.01, 0.02, 0.05\}$:

| Noise $\varepsilon$ | Rule | 1-Step Entropy $H(\text{row } 1 \mid \text{row } 0)$ | Mid-Grid $H(\text{row } 3 \mid \text{row } 0)$ | Late-Grid $H(\text{row } 5 \mid \text{row } 0)$ | Headroom Below Uniform ($\ln 2 = 0.693$) at Row 5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`0.05`** (Current) | Rule 90 / 110 | `0.199` nats | `0.507` / `0.515` nats | `0.616` / `0.647` nats | Only **`0.046` nats** ($93.4\%$ washed out!) |
| **`0.02`** | Rule 90 / 110 | `0.098` nats | `0.308` / `0.319` nats | `0.429` / `0.493` nats | **`0.200–0.264` nats** ($5\times$ larger signal) |
| **`0.01`** | Rule 90 / 110 | `0.056` nats | `0.193` / `0.202` nats | `0.286` / `0.345` nats | **`0.348–0.407` nats** ($7.5\times$ larger signal) |

- **Recommendation:** Set **$\varepsilon = 0.015$** (or $\varepsilon = 0.01$). At $\varepsilon = 0.015$, 1-step local laws are sharply distinct from multi-step marginals, row 5 still has $> 0.30$ nats of real information, and exact posteriors never underflow.

---

### 3. Training Mask Distribution & Bridge Query Geometry (Lines 32–34)

1. **Training Mask Distribution:**
   - Change line 32 from `mask rate uniform in [0.2, 0.8]` to **reveal count $k \sim \mathrm{Uniform}\{1, \dots, N-1\}$** ($N=48$).
   - Otherwise, Forward test queries with only row 0 revealed ($8/48 = 16.7\%$ revealed) are outside the training support!
2. **Bridge Query Geometry (`B1` and `B2`):**
   - Exact enumeration on $W=8$ periodic grids shows a striking difference between Rule 90 and Rule 110 when going **backward in time** ($t+1 \to t$):
     - Given row $t+1$ **alone** (zero cells in row $t$), Rule 110 already determines **71.4%** of row-$t$ cells, and clamping `(row 0, row 5)` yields `KL(Bridge || FwdOnly) = 0.1556` nats at row 3.
     - By contrast, in **Rule 90**, given row $t+1$ **alone**, **0.0%** of individual row-$t$ cells are determined (every cell has marginal $0.500$ due to global parity symmetries $x_{2j} \mapsto 1 \oplus x_{2j}$)—**until 2 adjacent anchor cells in row $t$ (or column 0 & 1 boundaries) are revealed, at which point 100.0% of the remaining row-$t$ cells are deterministically locked via the sideways/backward triple $(X_{t+1,c}, X_{t,c-1}) \to X_{t,c+1}$!**
   - **Fix:** Include both:
     - **`(B1)` Two-Row Bridge:** Evidence = row 0 and row $T-1$; query = middle rows $t \in \{2, 3\}$.
     - **`(B2)` Boundary-Anchor Bridge:** Evidence = half of row 0 (cols $0..3$) + row $T-1$ + column $0,1$ anchors; query = unrevealed interior cells. Here **neither** pure top-down nor pure bottom-up works, and sideways/backward 1-step triples $(X_{t+1,c}, X_{t,c-1}, X_{t,c}) \to X_{t,c+1}$ are $100\%$ stiff in Rule 90 and $85.7\%$ stiff in Rule 110!

---

### 4. Fixing Key-Value Fingerprints $k(i \mid S)$ so `P5` Doesn't Collapse Independent Arcs (Lines 40–41 & 55)

If $k(i \mid S) \in \mathbb{R}^{16}$ is taken as an untrained readout of cell $i$'s last-layer hidden state $h_i$, it will primarily encode cell $i$'s identity and its predicted logit $\text{logit } q(X_i \mid S)$. Then a forward chain from row 0 and a backward chain from row $T-1$ that both predict $X_i = 1$ will have $\langle k_{\text{fwd}}, k_{\text{bwd}} \rangle \approx 1$, falsely dividing their combined weight by 2!

- **Clean Fix:**
  1. **Source-Attention Fingerprint:** An arc $S \to i$ is defined by *which revealed tokens $j \in S$ token $i$ actually used*. In the last transformer layer (or averaged across layers), let $\alpha_{i \to j}$ be token $i$'s attention weight onto revealed token $j \in S$. Form the attention-weighted source summary:
     $$\bar{e}(i \mid S) = \sum_{j \in S} \alpha_{i \to j} \, \mathrm{pos\_emb}(j) \in \mathbb{R}^d, \qquad k_\psi(i \mid S) = \mathrm{normalize}\big(W_K [\bar{e}(i \mid S) \,;\, h_i]\big) \in \mathbb{S}^{15}.$$
     Notice that even *before* training $W_K$, if $S_1 = \text{parents}(i) \cup \{\text{distractor } A\}$ and $S_2 = \text{parents}(i) \cup \{\text{distractor } B\}$, the transformer attends to $\text{parents}(i)$ in both, so $\bar{e}(i \mid S_1) \approx \bar{e}(i \mid S_2)$ ($\langle k_1, k_2 \rangle \approx 1$), whereas if $S_{\text{past}}$ uses past parents and $S_{\text{future}}$ uses future children, their attention sits on disjoint tokens so $\bar{e}(i \mid S_{\text{past}}) \perp \bar{e}(i \mid S_{\text{future}})$!
  2. **Train $W_K$ on Validation/Training Multi-View Bags:** Sample pairs/tuples of subsets $(S_1, \dots, S_m)$ for the same target cell $i$ (half sharing the same local blanket, half from disjoint directions) and train $W_K$ and $\beta_0$ by minimizing the held-out outcome log-loss of the Gram-crowded pool:
     $$\text{logit }\mu(X_i) = \text{logit } p_0 + \sum_{a=1}^m \frac{w_a}{\sum_{b=1}^m \max(0, \langle k_a, k_b \rangle)} \big(\text{logit } q_a(X_i) - \text{logit } p_0\big).$$

---

### 5. `E-SUD` Adjustments (Lines 79–88)

1. **Use $6\times 6$ Sudoku (with $2\times 3$ boxes) as the primary benchmark; use $4\times 4$ only as a 2-minute smoke test.**
   - $4\times 4$ Sudoku has only $288$ valid solutions total (`P0` will memorize it completely, hitting the `D1` early-stop gap check immediately). $6\times 6$ Sudoku has $28.2\text{M}$ grids and genuine multi-step deduction depth.
2. **Keep contradiction detection in `E-SUD` 100% domain-agnostic (no hand-coded rule checker at test time):**
   - Append a homogeneous binary token `Solvable` $\in \{0, 1\}$ (token 37 on $6\times 6$), trained via the same masked loss on solvable partial boards (`Solvable=1`) and unsolvable partial boards (`Solvable=0`, generated by corrupting 1–2 clues on a board so no completion exists, matching the clue-count distribution so clue count alone doesn't leak solvability).
   - During search on an unsolvable board, as `P2` materializes forced cells (`beta_int >> 1`), the hidden contradiction becomes a 1-step local clash (or a cell where all 6 digits have low probability), causing $q_\theta(\text{Solvable}=1 \mid S_M^{(r)})$ and the internal consistency weight of branch $r$ to drop sharply.

---

## Pre-Registered Decision Rules (What I Would Register Differently)

Replace/refine `D1`–`D4` in `E-CA` and the `E-SUD` rule with the following 5 falsifiable rules:

- **`D0` (Amortization-Gap Sanity Gate — Run Before Full Sweep):**
  - On `E-CA` ($W=8, T=6, \varepsilon=0.015$) and `E-SUD` ($6\times 6$), check that the 1-pass model `P0` has $\mathrm{KL}(P_{\text{exact}} \parallel P_0) \ge 0.05$ nats on `F`, `B1`, `B2` (while having $\mathrm{KL} \le 0.02$ nats on 1-step fully-revealed-parent queries, confirming the model learned the 1-step law!). If `P0` already solves long-range queries to $< 0.01$ nats in 1 pass (e.g. if a 4-layer net memorizes $T=6$), increase grid depth to $T=8$ (or drop model depth to 2 layers) *before* running the inference sweep so there is a real multi-step gap to close.
- **`D1` (Topology & Stiffness Discovery Without Supervision):**
  - Over 2,000 random $(i, S)$ pairs on held-out grids, baseline-subtracted $\beta_{\text{int}}(i \mid S)$ achieves **Spearman $\rho \ge 0.60$** with true information gain $\ln 2 - H_{\text{exact}}(X_i \mid X_S)$, and **AUC $\ge 0.85$** for classifying whether $S$ contains a valid deterministic local blanket of $i$ (either $i$'s 3 past parents OR a valid backward/sideways triple $(X_{t+1,c}, X_{t,c-1}, X_{t,c})$).
- **`D2` (Sequential Materialization Through Stiff Arcs Beats 1-Pass `P0` and Random Order `P4`):**
  - With `P2` fixed to sequentially grow $S_t = S_{t-1} \cup \{z_t\}$, at $M = 16$ materialized cells (2 intermediate rows), `P2` reduces $\mathrm{KL}(P_{\text{exact}}(Y \mid E) \parallel \hat{\mu}(Y))$ on `F`, `B1`, and `B2` queries by **$\ge 30\%$** relative to `P0` AND beats random materialization order `P4` at equal $M$ by **$\ge 20\%$** (paired 95% CI over 3 seeds).
- **`D3` (Key-Value Insight Fingerprinting Deduplicates Redundant Views While Preserving Complementary Views):**
  - Test **both** halves of the trade-off (since setting $w_a \to 0$ trivially passes a "don't grow with $m$" test!):
    1. *Redundancy invariance:* When given $m \in \{1, 2, 5\}$ redundant conditioning sets $S_1, \dots, S_m$ (same parent triple + different random distractors), Gram-dedup (`P5`) keeps pooled log-odds drift $|\text{logit }\mu_m - \text{logit }\mu_1| < 0.20 \times$ the drift of un-deduplicated scalar pooling.
    2. *Complementary accumulation:* When given 1 forward chain from row 0 and 1 backward/sideways chain from row $T-1$ on Bridge queries (`B1`/`B2`), `P5` achieves strictly lower KL and log-loss than either chain alone (proving $\langle k_{\text{fwd}}, k_{\text{bwd}} \rangle \ll 1$).
- **`D4` (`E-SUD` $6\times 6$ Computation-as-Evidence):**
  - On $6\times 6$ Sudoku with clue-count-matched solvable and unsolvable partial boards, as search budget $M$ increases from $0 \to 8$, the full loop (materializing forced cells + proposal-coverage null-search update on refuted branches) reduces `Solvable` binary log-loss by **$\ge 20\%$** over `P0` on both solvable and unsolvable boards, and beats "positive-witness-only (ignore null search)" on unsolvable boards by **$\ge 20\%$**.
