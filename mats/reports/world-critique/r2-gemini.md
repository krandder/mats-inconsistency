Model: Gemini 4 Argon

**Status:** Round 2 cross-critique complete; designed a 4-rung portfolio of exact-Bayes prior worlds and 4 falsifiable cruxes for Kelvin's MATS program.
**Decision for Kelvin:** Approve the 4-rung ladder (starting with Rung 1: 16-Bit Correlated-Source Junta, then Rung 2: Factor Market) and the falsification thresholds in Section 3.
**Next Action:** Build the unified generator and exact-Bayes harness for Rung 1 and run the $N_{\text{train}}=64 \to N_{\text{test}}=2048$ adversarial policy-skew test (Crux 1).

---

### 1. Strongest Objections to the Other Critics' Round-1 Proposals

- **Against GPT-6 Proposal 1 (8-Bit Hierarchical Junta with Item Effects $b_x$):**
  **Domain saturation kills large-$N$ similarity scaling.** Keeping the 8-bit domain (256 moves) means that by $N \ge 1024$, almost every move in the entire universe has been sampled multiple times. If the target is the item value $g(q_R) + b_q$, the empirical mean at $q$ converges directly to the truth as $n_q$ grows, making cross-item similarity $R$ irrelevant at large $N$. Even when predicting the latent cell mean $g(q_R)$, each cell holds at most $32\text{--}128$ distinct moves, so cell posteriors saturate early and never test scaling across a growing frontier of distinct items.

- **Against GPT-6 Proposal 2 (Explicit Observation Provenance IDs):**
  **Turns semantic deduplication into trivial hash-collision lookup.** Handing the model an explicit discrete event UUID on every repeated copy lets a 1-head copy-suppression circuit deduplicate via exact token-ID matching without ever inspecting whether two observations carry redundant information. In real search or market aggregation, redundant signals arrive paraphrased or filtered through correlated intermediaries, not stamped with shared primary keys.

- **Against Fable 5.1 Proposal W1 (12-Bit Junta with i.i.d. Noise):**
  **No within-cell deduplication is actually required.** In W1, cells $c \in \{0,1\}^{|R|}$ have independent priors $g(c) \sim \mathcal{N}(0,1)$ and every observation has i.i.d. noise ($b_x = 0$, no shared source error). Conditioned on $R$, all readings inside cell $q_R$ are completely exchangeable: 500 repeats of *one* move $x \in q_R$ carry the exact same Bayesian information about $g(q_R)$ as 1 reading each of 500 distinct moves in $q_R$, and tokens outside $q_R$ carry zero value information. Consequently, once $R$ is selected, uniform softmax attention restricted to cell $q_R$ gives each token weight $1/m_{q_R}$ *for free*—contradicting Fable's own claim that W1 tests $-\log(m_g/N)$ cross-group reweighting during value readout.

- **Against Fable 5.1 Proposal W2 (Context-Tree Weighting Cut World):**
  **Hardwires similarity into a single fixed coordinate prefix.** In W2, the candidate equivalence classes are restricted to subtrees of one canonical binary tree aligned with bit order $(x_1, \dots, x_d)$. "Inferring similarity" reduces to choosing where to truncate a fixed prefix. Because two branches can never be equivalent unless they share a coordinate prefix, a transformer can solve W2 using fixed positional prefix masks ($d$ static heads) without ever learning dynamic key construction or cross-branch relational matching.

---

### 2. Portfolio (Ladder) of 4 Prior Worlds

#### Rung 1: 16-Bit Correlated-Source Hierarchical Junta (Copies vs. Measurements vs. Related Items)
- **Behavior:** Separates three levels of redundancy in one world: (a) relayed copies sharing measurement noise, (b) fresh noisy measurements of the *same* move (which saturate at item deviation $b_x$), and (c) distinct moves sharing latent cell $x_R = q_R$.
- **Generative Process & Exact Bayes:** Moves $x \in \{0,1\}^{16}$ ($65,536$ moves); hidden subset $R \subset \{1,\dots,16\}$, $|R|=2$ ($120$ hypotheses); cell value $g(x_R) \sim \mathcal{N}(0,1)$; persistent move deviation $b_x \sim \mathcal{N}(0,\tau^2)$. An observation draws or reuses a measurement source $s$ (tagged with a random per-episode key $k_s \in \{\pm 1\}^8$): measurement $z_{x,s} = g(x_R) + b_x + \eta_{x,s}$ ($\eta \sim \mathcal{N}(0,\sigma_{\text{meas}}^2)$), relayed as token $y_i = z_{x_i,s_i} + \xi_i$ ($\xi \sim \mathcal{N}(0,\sigma_{\text{copy}}^2)$). Exact Bayes is $O(120 \cdot N)$ via three nested 1D conjugate Gaussian reductions: relays $\to$ source measurements $\to$ move means (weight $\frac{n_x}{\sigma^2 + n_x \tau^2}$) $\to$ cell means.
- **Why In-Context Evidence is Mandatory:** $R$ and source keys $k_s$ are redrawn every episode. Under skewed policies, 100 relays of one source, 100 fresh measurements of one move, and 100 distinct moves in cell $q_R$ require completely different effective weights.
- **Predicted Mechanism:** Layer 1 counts $(x,k)$ and $x$ collisions to write saturating log-precision biases $-\log(\tau^2 + \sigma_{\text{eff}}^2/n_x)$ into attention logits; Layer 2 infers $R$ from cross-move covariance and pools precision-weighted move summaries.

#### Rung 2: Heterogeneous Factor-Market World (Spot + Spread + Basket Quotes with Latent Agent Reliability)
- **Behavior:** Market-like synthesis of heterogeneous information types (`SPOT` item readings, `DIFF` arbitrage spreads, and `BASKET` sector averages) emitted by agents with unknown reliability.
- **Generative Process & Exact Bayes:** $M=16$ latent asset values $v \sim \mathcal{N}(0, \Sigma_\pi)$ grouped into 4 sectors by a hidden permutation $\pi \in \Pi$ ($|\Pi|=32$). Context tokens are linear measurements $y_i = a_i^\top v + \epsilon_i$ where $a_i$ is a single basis vector (`SPOT` $v_m$), a difference $e_m - e_{m'}$ (`DIFF`), or a 4-item average (`BASKET`). Tokens come from $S=4$ agents with latent noise variance $\sigma_s^2 \in \{\sigma_{\text{low}}^2, \sigma_{\text{high}}^2\}$ ($2^4=16$ states). Conditioned on $(\pi, \{\sigma_s^2\})$ ($512$ hypotheses), the exact posterior accumulates precision $J_h = \Sigma_\pi^{-1} + \sum_{i=1}^N \sigma_{s_i}^{-2} a_i a_i^\top \in \mathbb{R}^{16\times 16}$ in $O(N)$ time, followed by 512 tiny $16\times 16$ Cholesky solves ($<2\text{ ms}$ for any $N$).
- **Why In-Context Evidence is Mandatory:** Query $v_q$ often has *zero* direct `SPOT` observations; it must be priced by chaining `BASKET` and `DIFF` quotes while inferring each agent's reliability from cycle consistency ($v_a - v_b + v_b - v_c \approx v_a - v_c$).
- **Predicted Mechanism:** Early layers evaluate cycle residuals to assign precision weights $\hat{\tau}_s$ to agents; subsequent layers perform iterative bipartite message passing between factor tokens and internal item representations.

#### Rung 3: Parity-Constrained Gaussian World (Hard Constraints vs. Soft Measurements)
- **Behavior:** Tests Kelvin's "inference as minimizing inconsistency" by mixing hard logical parity constraints with soft noisy measurements: a hard constraint makes previously independent measurements redundant or mutually falsifying.
- **Generative Process & Exact Bayes:** Latent state $v \in \mathbb{R}^8$ has signs $s \in \{\pm 1\}^8$ and magnitudes $\mu \in \mathbb{R}_+^8$ ($v_j = s_j \mu_j$, $\mu_j \sim \mathcal{N}(m_0, \sigma_0^2)$). Tokens are either soft noisy readings $y_i = v_{j_i} + \epsilon_i$ or hard GF(2) parity checks $c_i = s_{a_i} s_{b_i} s_{d_i} \in \{\pm 1\}$. Exact Bayes enumerates all $2^8 = 256$ sign vectors, eliminates any $s$ violating a parity token, and updates 8 decoupled 1D Gaussians per surviving $s$ in $O(256 + N)$ time.
- **Why In-Context Evidence is Mandatory:** Soft readings on $v_a$ and $v_b$ tell nothing about $s_d$ unless parity token $c_{abd}$ links them; redundant parity cycles add zero information after closure.
- **Predicted Mechanism:** Attention splits into a multiplicative sign-propagation circuit across depth and an additive precision accumulator for magnitudes.

#### Rung 4: Transposition DAG Search World (Hierarchical Backup + Cross-Branch Sharing)
- **Behavior:** MCTS-style value aggregation across a two-step search DAG where action paths share prefix values and merge into unknown transposition states at depth 2.
- **Generative Process & Exact Bayes:** 64 paths $(a_1, a_2) \in \{1,\dots,8\}^2$. Each episode draws a transposition map $\phi \in \Phi$ ($|\Phi|=64$) assigning paths to 8 depth-1 prefix values $u_{a_1} \sim \mathcal{N}(0,\sigma_u^2)$ and 8 depth-2 state values $w_{\phi(a_1,a_2)} \sim \mathcal{N}(0,\sigma_w^2)$. Tokens are either prefix evaluations $y^{(1)} = u_{a_1} + \epsilon$ or leaf rollouts $y^{(2)} = u_{a_1} + w_{\phi(a_1,a_2)} + \epsilon$. Exact Bayes is a 64-way mixture of $16\times 16$ linear-Gaussian precision updates ($O(64 N)$).
- **Why In-Context Evidence is Mandatory:** Evaluating an unvisited path $(a_1, a_2)$ requires subtracting estimated prefix value $\hat{u}_{a_1'}$ from leaf rollouts on transposed branch $(a_1', a_2')$, inferring $\phi$ from residual correlations, and adding $\hat{u}_{a_1}$.
- **Predicted Mechanism:** Layer 1 estimates $\hat{u}_{a_1}$ and residualizes leaf rollouts; Layer 2 matches residualized leaves to infer $\phi$; Layer 3 routes transposed leaf estimates to the query.

---

### 3. Program Cruxes and Sharp Refutations

1. **Crux 1: Softmax attention can achieve policy-invariant deduplication at $N \gg N_{\text{train}}$ without architectural normalization.**
   - *Refutation in Rung 1:* Train 2–4 layer transformers (10 seeds) on policy mixtures up to $N_{\text{train}}=64$ with *known* $R$ until excess KL $\Delta_{\text{KL}} = \text{KL}(P_{\text{Bayes}} \,\|\, P_\theta) < 0.02\text{ nats}$. Test zero-shot at $N_{\text{test}}=2048$ under an adversarial policy where 95% of tokens in cell $q_R$ are exact source-copies of one move $x_1$ and 5% are fresh distinct moves. **Refuted if:** across $\ge 8/10$ seeds, $\Delta_{\text{KL}}$ on the 95%-copy policy exceeds the uniform-$N=2048$ control by $>0.15\text{ nats}$ and shifts $>50\%$ of the way toward the naive "copy-blind" oracle that treats copies as independent.

2. **Crux 2: A bounded consensus state ($K \ll N$ slots) can match full-context aggregation when similarity is latent.**
   - *Refutation in Rung 1 & Rung 2:* Compare full $O(N^2)$ attention against a recurrent/Perceiver bottleneck with $K \in \{8, 16, 32\}$ slots processing chunks of 32 tokens up to $N=1024$ (10 seeds, matched FLOPs). **Refuted if:** when $R$ (or $\pi$) is revealed at token 0, the $K$-slot model matches full attention ($\Delta_{\text{KL}} < 0.03\text{ nats}$, proving sufficient storage capacity), **but** when $R$ is latent and disambiguating evidence arrives at $t > 512$, the $K$-slot model loses $>50\%$ of the Bayes gain over the $R$-agnostic baseline across all $K \le 32$ because raw cross-feature covariances were discarded before $R$ was known.

3. **Crux 3: Fixed-depth forward passes suffice for heterogeneous/constraint aggregation without iterative recurrence.**
   - *Refutation in Rung 2 & Rung 3:* Vary arbitrage/parity chain diameter $L \in \{1, 2, 4, 6\}$ linking observations to query $q$. Compare fixed-depth transformers ($D \in \{2, 4, 6\}$) against a weight-tied looped transformer ($D=2$ iterated $T=8$ steps, matched parameters, 10 seeds). **Refuted if:** fixed depth $D$ achieves $\Delta_{\text{KL}} < 0.03\text{ nats}$ for $L \le D-1$ but collapses to within $10\%$ of the chain-blind prior for $L \ge D+1$ across $10/10$ seeds, while the looped $D=2, T=8$ model holds $\Delta_{\text{KL}} < 0.05\text{ nats}$ for all $L \le 6$.

4. **Crux 4: End-to-end query NLL alone reliably discovers latent similarity keys without auxiliary consistency targets.**
   - *Refutation in Rung 2 & Rung 4:* Train 10 seeds under query-only NLL for $10\times$ the steps required by the known-$\phi$ oracle model. **Refuted if:** $\ge 8/10$ seeds stall on the prior-averaged plateau (recovering $<20\%$ of the Bayes gap between unknown-$\phi$ and prior-averaged references, with linear probes decoding $\phi$ at $<20\%$ above chance), whereas warm-starting from known-$\phi$ or adding an auxiliary cycle-consistency loss achieves $\Delta_{\text{KL}} < 0.03\text{ nats}$—proving a gradient-optimization bottleneck rather than an expressivity limit.

---

### 4. Motivated Architecture Changes

If Crux 1 refutes standard softmax at $N \gg N_{\text{train}}$ (because logit-space $-\log m_g$ counting drifts out-of-distribution), and Cruxes 2–3 show that premature compression into a small consensus loses latent similarity while long relational chains need iterative depth, those findings directly motivate a **Precision-Normalized Looped Consensus Architecture**: (1) replace raw softmax pooling with an explicit **key-kernel density normalizer** that divides each token's value contribution by its estimated local multiplicity $1 + \lambda_i \sum_j K(k_i, k_j)$ (natively implementing saturating hierarchical precision $(\tau^2 + \sigma^2/m_g)^{-1}$ at arbitrary $N$), and (2) pair a **compressive covariance sketch** (retaining second-order feature statistics in $O(d^2)$ space so latent metrics $R$ remain identifiable at any $N$) with a **weight-tied recurrent workspace** that iterates factor-to-consensus updates until local residual inconsistency stabilizes. Justifying this build requires seeing both failure modes cleanly in the baseline transformer: $>0.15\text{ nat}$ large-$N$ skew degradation in Rung 1 (justifying density normalization) and sharp diameter-$L$ / late-discovery cliffs in Rungs 2–3 (justifying the recurrent workspace + covariance sketch).
