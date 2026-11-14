Model: Gemini 3.8 Flash

**Status:** Critique complete. Candidates B and C are non-starters; Candidate A is salvageable only if expanded; Candidate D is intractable under Bellman propagation.
**Decision for Kelvin:** Approve shifting from 8-bit Candidate A to a 16-bit Subspace Junta or a Commutative Transposition Graph to eliminate trivial duplicate shortcuts.
**Next Action:** Implement data-generator and exact-Bayes benchmark for the 16-bit Expanded Junta.

---

### 1. Literature Reality Check: What Is Actually New?
Kelvin is right to suspect overlap with contextual bandits and Bayesian optimization (BO), but our core research question remains distinct if framed precisely:
- **What is already done:** PFNs4BO (Mueller et al. 2023) and DPT (Lee et al. 2023) train transformers on bandit/GP rollouts, but focus on downstream policy regret. In-context linear regression (Garg et al. 2022) examines passive learning, but with a fixed metric and near-uniform covariates. KR-UCT (Yee et al. 2016) shares evidence across moves, but assumes a *fixed, hand-crafted* kernel.
- **What is genuinely NEW:** 
  1. *Policy invariance of aggregation:* Standard PFNs rely on prior-level shrinkage and degrade when the rollout policy is heavily skewed or redundant. Learning an estimator whose aggregation weights adjust in-context to arbitrary, non-stationary sampling frequencies without overcounting is unsolved.
  2. *In-context metric discovery:* Inferring *which* actions are equivalent purely from label consistency $y$, rather than having the similarity kernel hard-coded.
  3. *Length generalization ($N \gg N_{\text{train}}$):* Demonstrating that attention can implement an exact sufficient-statistic accumulator that does not dilute or saturate when $N$ scales $10\times$ beyond training.

---

### 2. Autopsy of Candidate Worlds

- **Candidate A (Junta World):**
  - *Strongest Objection:* **Pigeonhole shortcut.** 256 moves (8 bits) is far too small. When testing scaling ($N = 500\text{--}5000$), identical 8-bit moves repeat dozens of times. The transformer can ignore $R$ entirely, solve the task via exact string lookup (solving (i)), and completely bypass in-context similarity inference (iii). Furthermore, coordinate masking is feature selection, not metric similarity.
- **Candidate B (GP with ARD):**
  - *Strongest Objection:* **$O(N^3)$ Bayes bottleneck and low novelty.** Exact Bayes over a hyperparameter grid requires inverting an $N \times N$ Gram matrix per grid point. At $N=1000$, generating 100k episodes will crush the compute budget. Additionally, PFNs4BO already tackled GP priors; continuous RKHS smoothing conflates interpolation with discrete deduplication.
- **Candidate C (Multi-Index $g(Ux)$):**
  - *Strongest Objection:* **Fatal violation of method constraints.** Approximate Bayes (MCMC/VI) introduces baseline errors. We cannot mechanistically attribute transformer errors if ground truth itself has approximation slop. Drop immediately.
- **Candidate D (Tree/MCTS-Shaped):**
  - *Strongest Objection:* **Bellman `max` destroys conjugacy.** Propagating values via `max` creates non-Gaussian, non-conjugate posteriors. Exact Bayes becomes intractable. If linearized into expectations without transpositions, it is simply a linear Markov tree with no cross-branch similarity.

---

### 3. Two Better / Modified Worlds

#### Proposal 1: 16-Bit Subspace Junta (Anti-Pigeonhole Junta)
- **Generative Process:** Moves are $x \in \{0,1\}^{16}$ ($65,536$ moves). Latent relevant subset $R \subset \{1,\dots,16\}$ with $|R|=2$ ($\binom{16}{2} = 120$ hypotheses). Cell values $g: \{0,1\}^2 \to \mathbb{R}$, $g(c) \sim \mathcal{N}(0, 1)$. Observations $y_i = g(x_{i,R}) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$. Policies vary from uniform to concentrated (e.g., repeatedly sampling moves with $x_{i,R}=00$ but differing across the other 14 bits). Query $x_*$ is strictly unobserved.
- **Exact Bayes:** 120-way discrete mixture over 4 independent 1D conjugate Gaussian posteriors. $O(N)$ operations, $<1\text{ ms}$ per episode even at $N=5,000$.
- **Reference Baselines:**
  - *(i) Duplicate pooling:* Present identical 16-bit moves with known $R$.
  - *(ii) Known similarity:* Supply $R$ explicitly as a prompt token; moves differ across the remaining 14 bits.
  - *(iii) In-context discovery:* $R$ is hidden; model must deduce $R$ from residual label correlations.
- **Expected Mechanism:** Layer 1 computes bit-agreement masked by label covariance; Layer 2 executes grouped softmax pooling over matching cells.
- **Main Risk:** Network learns discrete bitwise Boolean logic rather than general metric clustering.

#### Proposal 2: Transposition Search Graph (Commutative Path World)
- **Generative Process:** Explicitly models MCTS transpositions. Two-step paths $(a_1, a_2)$ with $a_1, a_2 \in \{1,\dots,16\}$ (256 paths). Moves commute to latent states $s = \phi(a_1, a_2) \in \{1,\dots,S\}$ under an unknown discrete symmetry $\phi \in \Phi$ ($|\Phi|=32$ permutation/reflection rules). State values $V(s) \sim \mathcal{N}(0, 1)$; $y = V(s) + \epsilon$. Exogenous rollout policy samples asymmetric branches (e.g., $(a_1, a_2)$ explored 50 times, $(a_2, a_1)$ 0 times). Target: evaluate unvisited transpositions.
- **Exact Bayes:** 32-way mixture over conjugate Gaussian updates. $O(|\Phi| \cdot N)$ compute.
- **Reference Baselines:**
  - *(i):* Exact duplicate path repeats $(a_1, a_2)$.
  - *(ii):* Symmetry rule $\phi$ provided explicitly; tests pooling across transposed pairs.
  - *(iii):* $\phi$ latent; model must infer commutativity from rollout rewards.
- **Expected Mechanism:** Bilinear token interactions in Layer 1 recognize path transpositions; Layer 2 pools across equivalent trajectories.
- **Main Risk:** If path encoding is order-agnostic by default, the network gets transposition invariance "for free" without learning.

---

### 4. Ranking & First Experiment

**Ranking:**
1. **16-Bit Subspace Junta (Proposal 1):** Cleanest math, cheapest exact Bayes, perfectly isolates in-context grouping from duplicate memorization.
2. **Transposition Search Graph (Proposal 2):** Best structural match for MCTS, slightly higher engineering overhead.
3. **Candidate A (Junta 8-bit):** Too small; easily cheated.
4. **Candidate B (GP-ARD):** Too slow at large $N$; not novel.
5. **Candidates C & D:** Reject (no exact Bayes).

**First Experiment to Run:**
Train a 2-layer, 4-head set transformer (params $\sim 5\times 10^4$) on the **16-Bit Subspace Junta** at $N=32$ with uniform policy. Evaluate zero-shot at $N \in \{32, 128, 512, 2048\}$ under two test policies: (a) uniform, and (b) heavily skewed (80% of samples concentrated on one latent cell). Compare MSE and attention entropy against Exact Bayes and the (i)/(ii)/(iii) references to verify whether attention weights shrink proportionally to cluster frequency without overcounting.
