# Guardian Review: Learned Securities Design (`learned-securities-design-2026-10-21.md`)

**Context:** Review requested by `opus-mats` in `#3875` against Kelvin's verbatim target:
- *"the problem is the non-separable information... The good example is the XOR example, when we actually only care about a single bit which is the parity. So we need to find a way to 'learn' the securities we need. This is what I'd most love to see."*
- *"I dream of S representing a market state, a combinatorial AMM kind-of-thing... a clear loss function, based on a market scoring rule applied to the changes to S, depending on the settlement of S... learned markets sounds like a weird thing but this is what I am looking for."*
- Reference implementation: Kelvin's `/home/agent/workspaces/bayes-engine/inference/factored_market.py`.

---

## 1. Where the Current Design (`Arms A–C`) Waters Down Kelvin's Idea (and Two Mathematical Traps to Avoid)

### Trap / Watering-Down 1: Fixed Trader-to-Latent Assignment ("Traders trade on 'their' $z_j$") vs. Learned Routing
- **What the doc says (Arm B):** *"m = k latent bits with a learned chain of partial parities (width 2), y linked at the end. Traders trade on 'their' $z_j$"*.
- **Why this waters down the idea (and breaks on unaligned `Z9`):**
  1. In `Z9` (`readers_per_bit = 3`, `n_distractor_agents = 2`, $N = 3k + 2$), there are $N > k$ traders arriving in a random permutation, multiple traders read the *same* parity bit, and some read *distractor* bits. Under `alignment="unaligned"`, trader $i$'s own-model slot vector is permuted by a private $T_i$, and even with aligned own-models or keys, trader position $i$ does not index bit $j \in \{1..k\}$.
  2. If each trader is hard-wired to a latent slot $z_j$, the market is not acting as a shared blackboard where redundant readers of the same bit see that $z_j$'s price has *already* moved and abstain (`E2`/`E3-sybil` deduplication), nor are traders selecting which security matches their evidence.
- **Fix for Arm B (even in the first experiment):**
  Give the market $m$ latent variables $z_1, \dots, z_m$. Each trader $i$ reads the current market prices $S_{i-1}$ (`q = unbounded`, per `#3874`) and its own evidence $x_i$, and **selects which security scope $C_{\ell_i}$ to trade on** (either via a learned router $\ell_i \in \{1..L\}$ included in the $b$-bit budget as $\lceil \log_2 L \rceil$ routing bits + $(b - \lceil \log_2 L \rceil)$ value bits, or a Gumbel-Softmax / straight-through scope selector) plus its $b$-bit log-potential trade $\Delta \psi_i(C_{\ell_i})$.
  *(Note on `Z9` alignment: under `alignment="unaligned"` with `visibility="none"`, a single binary reading's `own` one-hot is randomly permuted per trader, so no two traders share a coordinate system for which bit is which unless `visibility in {"noisy", "full"}` or `alignment="aligned"` or each trader sees multiple readings. Make sure the builder checks this so Arm B isn't blocked by the `floor_none` permutation symmetry!)*

### Trap / Watering-Down 2: The Zero-Marginal-Payout Trap on XOR When Scoring Only $\log p_{\text{after}}(y) - \log p_{\text{before}}(y)$
- **What the doc says (§Loss):** *"Each trader's loss is its LMSR profit at settlement, $\log p_{\text{after}}(y) - \log p_{\text{before}}(y)$, with $y$'s marginal computed by sum-product over $z$."*
- **Why this is a critical trap on $k$-bit parity (`Z9`):**
  Suppose the market has the exact parity chain $Y = z_1 \oplus \dots \oplus z_k$ with uniform prior $P_0(z_j = 1) = 0.5$.
  When Trader 1 arrives and trades on $z_1$, moving $P_1(z_1 = 1)$ from $0.5$ to $0.9$, what is the marginal $P_1(Y = 1)$ while $z_2, \dots, z_k$ are still at $0.5$?
  $$P_1(Y = 1) = \frac{1 - \prod_{j=1}^k (1 - 2 P(z_j = 1))}{2} = \mathbf{0.5}.$$
  For **every trader before the final bit of the parity is filled**, the marginal of $Y$ does not move at all:
  $$\log p_i(y^*) - \log p_{i-1}(y^*) = \log 0.5 - \log 0.5 = \mathbf{0}!$$
  - If you only optimize the **summed** telescoping loss $\sum_{i=1}^N [\log p_i(y^*) - \log p_{i-1}(y^*)] = \log p_N(y^*) - \log p_0(y^*)$ by backpropagating from the end of the episode ($t=N$) backward through the junction-tree sum-product into Trader 1's parameters, then Trader 1 is being trained by **global backpropagation through future traders**, NOT by its own LMSR profit!
  - Conversely, if you try to train Trader $i$ on its own immediate marginal change $\log p_i(y^*) - \log p_{i-1}(y^*)$ (`detach`ing across traders), Traders $1 \dots k-1$ get **identically zero gradient**!
- **How Kelvin's vision ("a market scoring rule applied to the changes to $S$, depending on the settlement of $S$") solves this locally:**
  In `factored_market.py` (`market.condition("Y", y_star)`, line 690), when $Y = y^*$ is revealed at settlement, the final market state $S_N(y, z)$ is Bayesian-conditioned on $Y = y^*$, inducing the **settled posterior over the latent securities** $q^*(z) := \text{stopgrad}[S_N(z \mid Y = y^*)]$.
  Under Hanson's joint combinatorial LMSR on $S(y, z)$, Trader $i$'s trade changed the full market state from $S_{i-1}(y, z)$ to $S_i(y, z)$. Evaluated against the settled state $(y^*, z \sim q^*(z))$, Trader $i$'s **Settled-State Combinatorial LMSR Profit** is:
  $$r_i^{\text{settled}} = \mathbb{E}_{z \sim \text{stopgrad}[S_N(z \mid y^*)]}\!\left[\log \frac{S_i(y^*, z)}{S_{i-1}(y^*, z)}\right] = \big(\log p_i(y^*) - \log p_{i-1}(y^*)\big) + \mathbb{E}_{z \sim S_N(z \mid y^*)}\!\left[\log \frac{S_i(z \mid y^*)}{S_{i-1}(z \mid y^*)}\right].$$
  1. **Exact Telescoping:** Summing $r_i^{\text{settled}}$ over $i = 1 \dots N$ still telescopes exactly:
     $$\sum_{i=1}^N r_i^{\text{settled}} = \big(\log p_N(y^*) - \log p_0(y^*)\big) + \mathrm{KL}\!\big(S_N(z \mid y^*) \,\|\, S_0(z \mid y^*)\big).$$
  2. **Strictly Positive Local Credit on XOR:** Even though $\log p_1(y^*) - \log p_0(y^*) = 0$ when Trader 1 trades on $z_1$, once later traders move $z_2 \dots z_k$ and $Y = y^*$ settles the parity chain, $S_N(z_1 \mid y^*)$ concentrates on $y^* \oplus z_2 \oplus \dots \oplus z_k$, so Trader 1's local profit $r_1^{\text{settled}}$ is **immediately non-zero and positive** iff Trader 1 moved $z_1$ toward the truth!
  3. **Recommendation:** Include both training modes in the experiment: (i) end-to-end backprop of $\log p_N(y^*)$ through the junction tree, and (ii) local per-trader training on $r_i^{\text{settled}}$ (detached across traders, testing whether the market scoring rule itself coordinates the traders!).

### Watering-Down 3: Pre-Fixed Chain Structure vs. `factored_market.py`'s Dynamic Re-Triangulation under Treewidth Budget $w$
- In `factored_market.py` (`_restructure(new_scope)`, lines 606–613 & 717–758), the junction tree is **not** fixed in advance: the market starts sparse/disconnected, and when a trader submits a conditional trade on `scope = (variable_id, *context)` not covered by any current cluster, `_restructure(scope)` dynamically re-triangulates the junction tree with `scope` as a forced clique, rejecting the trade only if it would exceed `max_width = w`.
- Pre-wiring a fixed width-2 chain (`Arm B`) reduces "learning the securities" to learning $2 \times 2 \times 2$ CPT link weights on a hand-designed graph.

---

## 2. What the Ambitious Version Is (Answers to the Two Open Design Questions)

1. **Question 1: Do traders choose which listed scope to trade on (learned routing), or is each trader assigned one?**
   - **Answer: Traders MUST choose which scope to trade on (learned routing).**
   - Assigning trader $i$ to scope $i$ makes the security identity depend on arrival rank $i$, which fails whenever arrival order is random, when multiple traders observe the same underlying factor (`Z9` has 3 readers per bit; `E2`/`E3-sybil` have overlapping/duplicated pieces), or when distractions exist.
   - With learned routing, the market exposes a book of $L$ securities/cliques $\{C_1, \dots, C_L\}$ with learned key embeddings $\theta_{\text{book}}$ and live marginal tables $S_{i-1}(C_\ell)$. Trader $i$ reads $S_{i-1}$ (`q = unbounded`), matches its evidence against the book + current prices, and emits `(scope_idx, quantized_log_potential)` within its $b$-bit uplink budget.
   - **Bonus test on `E3-sybil` / `E2`:** When a Sybil clone arrives after an identical trader has already traded on $C_\ell$, reading the updated price $S_{i-1}(C_\ell)$ lets the clone see zero discrepancy between its posterior and the market price, so it trades zero!

2. **Question 2: Grow the structure during an episode, or learn a fixed structure across episodes first?**
   - **Answer: Stage it as a 2-step ladder (`Arm B_fixed` $\to$ `Arm D_dynamic`), because `factored_market.py` already makes within-episode growth cheap at width $w=2$:**
     - **Step 1 (`Arms A, B, C` — what your builder is coding now):** Keep the builder moving! Have `Arm B` learn the junction-tree potentials $\theta$ + **learned trader-to-clique routing** over a width-$w$ latent template across episodes.
     - **Step 2 (`Arm D` — Dynamic Within-Episode Security Creation / Combinatorial Call-Off Bets):**
       Start the episode with disconnected latent variables $z_1, \dots, z_m$ and $Y$ (`treewidth = 0`). Allow a trader (or an arbitrage/combiner trader) to submit a **conditional trade** $\Delta \psi(Y \mid z_a, z_b)$ or $\Delta \psi(z_c \mid z_a, z_b)$ that invokes `FactoredMarket._restructure(scope)` (or a differentiable soft mask over candidate width-2 cliques subject to treewidth $\le w$), dynamically binding the partial parities that were actually traded in that episode!
