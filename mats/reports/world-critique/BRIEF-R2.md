# Round 2: cross-critique and a portfolio of prior worlds

You are one of three critics from different model families (GPT, Gemini, Claude). Round 1 answers from all three are
below. Read BRIEF.md context (also below) and Kelvin's broadened goal, then do round 2. Be sharp; disagree where warranted.

## Kelvin's broadened goal (verbatim, 2026-10-28 ~01:35Z)
"I'd like to find the best possible priors that we can use that show transformers learning interesting dedup and
aggregation behaviors (market-like behaviors, let us say, in the sense that they aggregate different kinds of info
into a single estimate whenever needed in complex ways). Then let us try to learn mechanistically from the transformers
to figure out how they learn that and how they do it, and then go to find what kinds of new architectures in line with
my MATS program could scale to more observations or "more reasoning" according to my program; (or address other cruxes
about whether this program is good; to disprove any high-level hypothesis we need very sharp observations that I and
gemini are very likely to accept and really change my world-model)"

Program background (one paragraph): Kelvin's MATS program is information aggregation with rich messages and a bounded
consensus; search/inference targets minimised inconsistency instead of RL reward; transformers are seen as already
doing aggregation and in-context search; he wants to learn from them and extend them (more observations, more
reasoning) rather than invent unrelated architectures.

## Your tasks (plain text, <= 1100 words)
1. Attack the other two critics' proposals (not your own): the single strongest objection to each proposed world.
2. Propose a PORTFOLIO (ladder) of 3-5 prior worlds that together serve the goal: each shows a distinct, interesting
   aggregation/dedup behaviour (e.g. copies vs fresh measurements vs related items; inferred similarity; heterogeneous
   information types combined into one estimate, market-like; hierarchical/tree evidence; constraints vs measurements),
   with exact Bayes, a reason a transformer must use in-context evidence (not prior-level shortcuts), and the
   mechanism you predict. Order them from first to build to last. You may reuse round-1 proposals.
3. CRUXES: list 3-5 high-level hypotheses of Kelvin's program (e.g. "attention-style aggregation can be policy-invariant
   at N >> N_train", "in-context dedup needs depth that grows with redundancy structure", "a bounded consensus state can
   match full-context aggregation") and for each: the sharp observation in these worlds that would REFUTE it, stated so
   that a skeptical reader would accept the refutation (controls, references, seeds, what number crosses what line).
4. One paragraph: which architecture changes the mechanistic findings could motivate, and what result would justify
   building them.


===== BRIEF.md =====
# Choosing a prior world for learning in-context similarity (critique round brief)

You are one of several independent critics (different model families) helping choose the synthetic "prior world"
for the next experiments in Kelvin's MATS research. Be critical and concrete. Do not flatter the candidates.

## The research goal (Kelvin's words)
- "I'd like to understand potential *scalable* in-context aggregation mechanisms, that could potentially scale to N
  observations, with no clear limit on N, and be more likely to 'converge to truth' as long as the observations
  'cover' as much as possible, without depending much, if possible, on the 'policy' that determines what sources are
  observed more frequently etc."
- "I suspect that a large transformer would be able to learn how to do this in a way with a PFN environment."
- The motivating idea (10-15): "the default UCT has no 'similarity measure'; you can test one 'move', then after
  expensive multiple rollouts decide this 'line' is not promising, but then you can sample another move that may turn
  out to be semantically similar in some relevant sense, but you can't easily learn that this other move is really
  just very similar to the other that wasn't good. In some ways transformer LLM + CoT solved this..." and "the
  learning of redundancy ... depends on the search-generation policy (more copies of 'similar' edges -> have to learn
  smaller weights)". He wants keys/fingerprints learned so redundancy is handled without depending on the policy.
- He cares about in-context aggregation and deduplication (using within-episode evidence), NOT prior-level
  discounting (a fixed shrinkage learned from the training distribution).

## Method constraints
- PFN-style training: small transformers (1e3 to 1e6 params) trained from scratch on fresh episodes sampled from the
  prior world; set-style tokens; one forward pass; output a predictive distribution.
- Exact (or numerically exact) Bayes posterior must be computable per episode, cheaply, so deviations from Bayes can
  be measured and later explained mechanistically (interpretability is a goal: we want to find HOW the network does it).
- The world must allow: varying N (train small, test much larger), varying observation policy (uniform, concentrated,
  adaptive), references that separate (i) pooling repeats of the same item, (ii) sharing evidence across similar items
  when similarity is known, (iii) inferring the similarity itself in context.
- Compute: a few hundred dollars at most; CPU/L4 scale.

## What we already know (from our runs)
- Set-style PFN on a linear-Gaussian source world: attention does all the dedup there is; with explicit group keys,
  depth 2+ dedups 80%; without keys, about 45% via matching residual values; 1-layer models are prior-level only.
- Tiny models (3 sources, unknown reliability): 1 layer = prior-level ("query-only"); 2 layers ~ Bayes; ~20% of seeds
  get stuck at the prior-level solution.

## Candidate worlds (attack them; propose better ones)
A. Junta world: 256 moves = 8 surface bits; hidden relevant subset R (|R| in {1,2,3}) per episode; value = g(bits in R),
   g a random N(0,1) table; observations y = value + noise at moves chosen by a varying policy; target = value of a
   query move (often unobserved), or the best move's value. Exact Bayes = 92-way mixture over R with conjugate per-cell
   posteriors (count + sum). "Similar" = agree on R.
B. GP with per-episode relevant dimensions/lengthscales (ARD) over continuous keys; exact GP posteriors mixed over a
   discretised hyperparameter grid.
C. Low-dimensional hidden projection f(x) = g(Ux) (multi-index model); approximate Bayes only.
D. Tree/MCTS-shaped world: moves lead to sub-moves; values propagate (expectation or max); similarity across branches.


## Related literature you must account for (Kelvin asked: "Isn't this similar to the contextual bandit literature?")
Yes: candidate A is a structured (sparse/junta) contextual- or multi-armed bandit / Bayesian-optimisation reward model,
with the action policy made exogenous. Known work (verify, extend): sparse linear bandits; GP-UCB / kernelised bandits;
Thompson sampling; KR-UCT (Yee, Lisy, Bowling 2016, MCTS sharing evidence across similar actions by kernel
regression); PFNs4BO (Mueller et al. 2023, PFN trained on GP priors for Bayesian optimisation); Decision-Pretrained
Transformer (Lee et al. 2023, transformers pretrained on bandit tasks behave like posterior sampling); Algorithm
Distillation (Laskin et al. 2022); in-context linear regression (Garg et al. 2022). Say what, given this literature,
would still be NEW in our study (we think: the aggregation mechanism itself, its invariance to an exogenous policy,
scaling to N far beyond training, and in-context dedup vs prior-level discounting) or whether it is already done.

## What to deliver (plain text, ≤ 900 words)
1. For each candidate: the strongest objection (does it actually test the goal? can a model pass by a shortcut?
   is Bayes really policy-independent here? does "similarity" mean what Kelvin means? does it scale in N?).
2. Up to two better or modified worlds you would propose instead, each with: generative process, exact-Bayes method,
   the references that separate (i)/(ii)/(iii), the expected mechanism a transformer would need, and the main risk.
3. Your ranking and the one experiment you would run first.

===== r1-codex.md =====
I’m GPT-6.

I would start with a modified junta world that explicitly distinguishes repeated measurements from repeated evidence.

1. Strongest objections

A. Junta: This tests learning an episode-specific equivalence relation, but its independent observation noise makes repeated sampling genuinely informative. Bayes should increase confidence with additional measurements; suppressing their influence indiscriminately would be wrong. It therefore needs a separate copying or shared-noise mechanism to test deduplication. Eight visible bits also permit enumeration of 92 hypotheses: success could mean learning a small hypothesis catalogue rather than a scalable similarity algorithm. Increasing N over 256 moves tests repeated-data handling, not growth in the number of distinct items. For posterior weights over R, count and sum alone are insufficient: retain sums of squares or equivalent residual information. Query-value prediction is the clean initial target; maximum-value prediction introduces a separate nonlinear inference problem.

B. GP-ARD: Closest to conventional similarity learning, but expensive as an oracle: H grid points cost roughly O(HN³) with dense inference. A grid is exact only for a prior actually supported on that grid. Interpolation with an average kernel could perform adequately without learning episode-specific relevance. Smooth correlation also differs from evidence duplication: nearby observations generally contain additional information. This world risks spending the budget on matrix algebra and oracle computation before resolving the mechanism question.

C. Hidden projection: Approximate Bayes violates the main methodological constraint and confounds network error with oracle error. U is generally nonidentifiable up to transformations absorbed by g; probes for “recovering U” could misdiagnose correct prediction. Restricting U to a finite catalogue restores tractability but largely recreates B or a finite feature model.

D. Tree/MCTS: Too many simultaneous problems: latent similarity, sampling, backup operators, exploration and depth. Independent Gaussian leaves permit tractable maximum distributions, but unknown cross-branch sharing destroys that simple factorization. Fixed trees invite topology shortcuts. Use this as a transfer environment after identifying an aggregation mechanism.

Across all four, policy independence needs qualification. For complete histories generated by policies depending only on recorded past observations and independent randomness, policy factors cancel from posterior inference. Policies using hidden information, selective reporting or omitted observations need explicit modelling. Different policies still produce different evidence and therefore different posteriors. Coverage must identify the relevant hypotheses; visiting many redundant points is insufficient.

2. Two modified worlds

First: hierarchical junta with persistent item effects.

Draw R and Gaussian cell values g as in A. For each item x, draw a persistent deviation b_x ~ N(0, τ²). Measurements are y_xt = g(x_R) + b_x + ε_xt, with independent Gaussian ε. Predict g(q_R) or the explicitly specified item value g(q_R)+b_q.

Exact Bayes: enumerate 92 subsets. Within each, reduce measurements to per-item counts, sums and squared sums. Item means have variance τ²+σ²/n_x; integrate each Gaussian cell analytically and mix using marginal likelihoods. Cost is linear in observations times 92, with small cell summaries.

References: independent item posteriors for (i); true-R hierarchical Bayes for (ii); mixture-over-R Bayes for (iii). Include a fixed prior-averaged similarity baseline.

Expected mechanism: pool item measurements, compute saturating item precision, infer relevance from cross-item evidence, then aggregate within inferred cells.

Main risk: the network may solve only the finite catalogue. Nevertheless, this makes the intended distinction measurable: endlessly resampling one item cannot eliminate its persistent deviation.

Second: the same world with explicit observation provenance.

Give every fresh measurement an event ID. The presentation policy can repeat an existing event arbitrarily, preserving its ID and value, or obtain a fresh measurement with a new ID. Copies add no evidence; fresh measurements do.

Exact Bayes first collapses identical event IDs, then uses the hierarchical calculation above. References are event-aware independent-item Bayes, event-aware true-R Bayes, and event-aware full-mixture Bayes. Add an oracle that incorrectly treats copies as independent.

Expected mechanism: event deduplication, item pooling, then inferred-cell pooling. Randomized IDs prevent memorizing particular identifiers.

Main risk: provenance makes deduplication easy and artificial. That is useful initially: failure cannot be blamed on inherently ambiguous duplicate detection.

3. Literature, ranking and first experiment

Sparse linear bandits, GP-UCB and Thompson sampling already study structured reward inference and adaptive acquisition. KR-UCT directly precedes similarity-based sharing in tree search. PFNs4BO precedes amortized Bayesian prediction for optimisation; Decision-Pretrained Transformers and Algorithm Distillation precede learned in-context decision procedures; Garg et al. precedes controlled synthetic function-learning studies. Also compare Deep Sets and Set Transformers for aggregation architecture.

The plausible contribution is a causally identified aggregation mechanism that distinguishes copies, measurements and related items, remains close to Bayes under policy shifts, and extrapolates in N. Neither the reward world nor transformer-based Bayesian learning alone is new. Literature verification here was partial: Crossref confirmed Garg et al.; other queries were mismatched or rate-limited, so priority remains unverified.

Ranking: hierarchical/provenance junta, original A, B, D, C.

First experiment: train small two- and three-layer models at N=16–64; test through N=1024 with paired interventions adding copies, fresh same-item measurements, or new related items. Measure predictive KL to Bayes, calibration and context-attention ablations across seeds and sampling policies. Copies should leave predictions unchanged; fresh evidence should change them appropriately. Report runtime too: ordinary dense attention remains quadratic, so length extrapolation alone does not establish unlimited scalability.

===== r1-gemini.md =====
Model: Gemini 3.8 Flash (round 1; round 2 uses Gemini 4 Argon)

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

===== r1-fable.md =====
Model: Fable 5.1 (claude-fable-5-1). (Saved by the coordinator from the board-fable agent's final message, 2026-10-28.)

1. Strongest objections

A (junta). The similarity structure is a 92-item menu fixed in the prior, so "inferring similarity in context" reduces to in-context model selection among 92 templates; the sufficient statistics (mean reading at bit=0 vs bit=1, pairwise interaction contrasts, per-cell counts) are one-layer linear statistics, and Bai et al. 2023 ("Transformers as Statisticians") already show transformers can implement this kind of in-context model selection. It is case (iii) formally, but its easiest version. Policy leak: under any value-adaptive policy, per-cell counts correlate with value; a net trained on one policy learns count->value and the invariance claim is untestable. Bayes is policy-invariant here only because the policy is ignorable (depends on observed data, not on R or g); keep it so, train on a policy mixture, test on a held-out family. Scaling: with 256 moves, N > 256 is only repeats, so the large-N regime tests (i) alone; (ii)/(iii) saturate once each cell has a few readings. With a random g table an unobserved cell carries no information, so "query unobserved" is just "which cell is the query in"; the best-move target is a different problem (argmax under uncertainty) and should not share a model.

B (GP-ARD). This training setup is already done (PFNs4BO: GP priors with hyperparameter priors); novelty would lie only in the analysis. Exact Bayes holds only if hyperparameters are drawn from the grid you mix over, and O(N^3) per grid point caps N near 2000. The useful contrast is that attention is Nadaraya-Watson (policy-dependent smoother) whereas the GP posterior is K^-1-weighted (policy-invariant), but that is a known-similarity (ii) contrast; lengthscale selection is a weak (iii). Interpretability is hardest here: continuous keys, no discrete structure to read off.

C. No exact Bayes breaks a stated constraint. Drop, or discretise U to a finite set, which turns it into A with continuous inputs.

D. Max-propagation destroys conjugacy; exact Bayes on maxima of correlated Gaussians is intractable beyond toy trees. Expectation-propagation is linear, which makes it a linear-Gaussian world with a tree covariance; then "similarity across branches" must be defined, and the MCTS-meaningful notion (transpositions) is case (i) with hidden identity, not a new case. The MCTS policy is intrinsically endogenous; making it exogenous removes the motivation.

Cross-cutting: policy invariance at large N requires weighting each reading by 1/(multiplicity of its group), a counting operation. Softmax attention averages within a group for free, but across groups a query gives group g weight m_g/N (policy-dependent) unless an earlier layer computes m_g/N and feeds -log(m_g/N) into the logits. Transformers count badly beyond training length (Yehudai et al. 2024; Barbero et al. 2024), so expect policy-dependence to reappear at N >> N_train when the policy is concentrated (rare groups with tiny m_g/N). This is a concrete mechanistic prediction to test in any world.

2. Proposed worlds

W1 (modified A; run first). 12 bits, |R| <= 2 (78 hypotheses) or <= 3 (298); N_train <= 64, N_test up to 4096; policies: uniform, Zipf-concentrated, value-adaptive (UCB-like on running means), all ignorable so the Bayes reference is unchanged. References: (i) per-move pooling Bayes (each move its own cell); (ii) R-oracle Bayes; (iii) full mixture; plus a shortcut reference, regression of value on counts under the training policy. Mechanism needed: layer 1 computes per-bit and per-pair contrasts and per-cell counts; layer 2 computes mixture weights and 1/m weighting. Main risk: the count shortcut; the held-out-policy test is the guard.

W2 (hierarchical cut world; tree-shaped, exact via context-tree weighting). Moves are leaves of a known binary tree of depth d (ids are d-bit paths). Per episode each internal node is independently a "cut" (prob 1/2); the cut nodes partition the leaves into blocks, each block gets its own mean ~ N(0,1), readings = block mean + noise. Exact Bayes by the CTW recursion (Willems, Shtarkov, Tjalkens 1995) with Gaussian-conjugate node likelihoods: O(N d) per episode for any N, mixing exactly over 2^(2^d - 1) partitions. Similarity = sharing a block, inferable in context from within-subtree variance; (ii) oracle receives the cut, (i) per-leaf pooling, (iii) full CTW. Transformer mechanism: d prefix-match heads (one per level) producing pooled mean, count and within-subtree variance; an MLP computing merge-vs-split evidence; a second layer combining. Interpretable as one head per level. Risks: evidence for deep cuts is weak at small N, the prior is unnatural, and the mechanism is fairly obvious, so the interesting question is only whether the net does it policy-invariantly at large N.

3. Ranking and first experiment

W1 > W2 > B > D > C. First experiment: W1 trained on a policy mixture at N <= 64, tested at N = 64/512/4096 under held-out concentrated and adaptive policies; report the Bayes gap decomposed against the three references and the count-regression shortcut, and check whether attention logits track -log(m_g/N).

What would be new: DPT (Lee et al. 2023) shows behavioural posterior sampling that tolerates the collection policy; PFNs4BO trains on GP priors; Bai et al. 2023 prove in-context model selection is implementable; KR-UCT shares evidence by kernel with known similarity. None measures the gap to exact Bayes under policy shift at N far beyond training, splits in-context dedup from prior-level discounting with intervention references, or identifies the mechanism. That decomposition is the contribution; "a transformer approximates Bayes on bandit data" alone is not.
