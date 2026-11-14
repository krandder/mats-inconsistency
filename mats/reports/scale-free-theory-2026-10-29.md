# Scale-free, untangled information combination: definitions, theorems, tests (29 October 2026)

For Kelvin (theory lane; brief from main). Revised after adversarial reviews by GPT-6 and Gemini (section 10).

Status tags: **PROVEN** (proof here, or a cited theorem), **PROVEN UNDER X** (hypotheses named), **HEURISTIC**
(an argument, no proof), **CONJECTURE**, and **MEASURED** (a number from our own runs; it supports a claim, it does
not prove it). Scripts, outputs, checked citations and reviews: `work/scale-free-theory/`.

## Plain summary

1. **The first attempt mixed three costs.** Counting bits across the cuts of one fixed hierarchy calls a task tangled
   when it is only *routed*: binding reports to their sources (World FA) needs about N bits at some cut of every
   fixed hierarchy, but with a short label per item ("which source") every message is a couple of bits. It also counts
   precision, which you said is not the point.
2. **The definition that fits your intuition.** Information combines along a hierarchy that may depend on the input.
   Keep three things separate: *routing* (short labels say which items combine; cheap when the rule is "same label",
   "before/after" or "contiguous chunk"; labels may be updated between rounds), *content* (the message at each scale,
   allowed to grow), and the *combination rule*. For Bayesian aggregation the rule is the same at every scale: add
   log-likelihoods, and sum out the hidden variables that are complete at that node. The message at a node is the
   likelihood of the hidden variables still shared with the outside ("open"). What grows with scale is the number and
   richness of open hidden variables, not the number of items: larger structures may need more dimensions, while how
   information combines stays fixed. This is junction-tree message passing read for your question; the coordinates
   matter: World FA's correlated errors look global (the whole history must cross a trader cut), but in time
   coordinates each side sends only O(T^2) numbers that add up (Gemini's correction to our draft).
3. **Rounds matter more than the shape of the hierarchy.** If sources write once without seeing anything, each
   message must describe a whole function of the open hidden variables. If the root first broadcasts a consensus and
   sources then write relative to it (your "messages written after reading the consensus"), a few local numbers
   suffice and the error falls as N grows. On the tiny-PFN world: one round needs 48-64 numbers per source for
   KL <= 10^-3 at any N from 3 to 4096; two rounds with 8 numbers reach KL of about 1e-7 at N = 1024 (below N of
   about 64 the consensus is too vague for local messages). With 3 open hidden variables, a one-round linear code
   needs more than 1,600 numbers per source, against a few hundred for two rounds. The mechanism is the one-step estimator of statistics, and transformers are known to run such
   multi-step estimation in context; what is ours is the link to your design and to architectures, and these numbers.
4. **What provably needs tangled or dense structure.** Binding: tangled for every fixed hierarchy, untangled with
   labels; one attention head can route by ID, so World FA's failure looks like a learning failure (argued, not
   proved). Multi-hop lookups need rounds (depth). Parity is untangled under every definition yet hard for
   statistical-query learning, which covers gradient training at coarse precision (World 3's XOR chains). Genuinely
   global couplings (expander-like) are tangled under every hierarchy, though algebra (Gaussian, GF(2)) can keep
   expectations easy. Grouping items by correlation of their histories needs keys of about min(T, 1/rho^2) bits per
   item. Sharp attention (putting almost all weight on one of N items) needs logits of range log N, exactly the regime
   where Alman and Song show no generic subquadratic algorithm exists (under SETH); label routing needs only sorting or
   hashing.
5. **Treewidth is the wrong measure.** A complete interaction graph can be maximally untangled (2 bits per cut), and
   Bayes with one shared answer makes every pair of items dependent (treewidth N - 1) though a single hidden variable
   carries everything. The right measure is the rank of what crosses a cut, after cheap algebra (Gaussian parts) is
   taken out; in a ReLU net the "non-Gaussian part" is coupled through the Gaussian pre-activations, so it must be
   counted as the entropy of the gate pattern.
6. **Estimation (M1/M3).** Exact tails are cheap when the input law is untangled along the *same* hierarchy as the
   computation (cost cubic in the two bond dimensions); for two random trees on the same leaves the combined treewidth
   grows with N (measured). Expectations need only low rank; tails need low dimension (a rank-2 linear function has
   #P-hard tails). Under activation noise with bounded activations, message dimension must grow like log N (proved
   for a Gaussian model): more dimensions at larger scales is a necessity, not just a permission.
7. **Transformers.** Softmax attention with a sink is exactly the merge of Gaussian messages (logit = log precision;
   the prior is the sink; its weight is prior precision times posterior variance). For one query, attention is an
   associative merge along any hierarchy, so a dense layer is N such merges; "dense attention pays for routing"
   becomes "few, label-determined routing classes per layer, with simple content". Measurements on our nets (the
   addendum has the pre-registered substitution tests): in 2-layer tiny-PFN nets both layers are very sensitive to
   label alignment (scrambling costs 0.4-4.4 nats) and to episode-specific attention (0.2-1 nats), so attention does
   more than route; a "label table plus one weight per key" surrogate meets the registered criterion for the combining
   layer but not for block 1; the query token's self-attention correlates with posterior variance in most nets (the
   prior-as-sink reading is a conjecture); what block 2 reads per context token compresses to 3-4 dimensions. In 52
   nets, including the 12 phase-B nets, each net's departure from Bayes has a cross-source interaction part (about
   16-32% of its variance); the raw interaction mostly tracks the size of the error, as a reviewer predicted, so it is
   not by itself evidence of tangling.
8. **Major advance and shortest path.** Either show that trained transformers do in-context Bayes as this kind of
   computation, with each departure from Bayes predicted as the error of the architecture's merge rule at some scale
   (the big-miss departure already fits), or build an untangled architecture at transformer loss with exact tails.
   The addendum (M3-M5, pre-registered, run after the coordinator's go) tests the transformer side on our nets. Next:
   the tree experiment specified in 6.5, and a causal test of the cross-source part of the error. No decision needed.

## 1. Setting

Items x_1..x_N (tokens, sources, reports). The target is a function f(x) or a posterior p(y | x). A **hierarchy** is a
rooted binary tree whose leaves are the items; node v covers the item set S_v, and its **message** m_v depends only
on x_{S_v}, computed from its children's messages by a **merge** g_v. A **round** is one upward pass; between rounds
the root may **broadcast** something down. The input law P is the prior over episodes; its hidden variables (the
answer, each source's reliability, shared errors, group memberships) are the **latents** u.

## 2. Six candidate definitions

**D1. Bits across the cuts of a fixed hierarchy.** Two contents x_S, x'_S of a cut S are equivalent if
f(x_S, z) = f(x'_S, z) for every completion z; K_f(S) is the number of classes (the Myhill-Nerode relation for tree
automata, Thatcher and Wright 1968). Untangled: log2 K_f(S) grows slower than |S| (sub-volume), strongly untangled if
polylog.

- T1, minimal tree computation (**PROVEN**). Every exact tree computation along T has at least K_f(S_v) distinct
  messages at v, and sending the class achieves it. Proof: if two inequivalent contents shared a message, the output
  (a function of m_v and of z) would agree on the completion z that separates them. Conversely the class is a
  function of the children's classes: if x_a ~ x'_a and x_b ~ x'_b, then f(x_a, x_b, z) = f(x'_a, x_b, z) =
  f(x'_a, x'_b, z) for every z. K_f(S) is the number of distinct rows of the communication matrix, and
  ceil(log2 K_f(S)) the deterministic one-way communication complexity from S to the rest.
- With tolerance (**PROVEN**): at each node choose an *internal* epsilon-net (representatives are actual contents of
  that node) in sup norm over completions. The same substitution gives e_v <= e_a + e_b + epsilon, so the root error
  is at most epsilon times the number of quantised nodes (<= 2N - 1); internal covering numbers at
  epsilon = delta / (2N - 1) replace K. With an external net (arbitrary functions) the radius doubles per node.

**D2. Bits across the cuts of the input's own hierarchy (routing).** Each item gets a label lambda(x_i) of b bits; a
fixed **rule** builds the hierarchy from the labels; content is counted as Nerode classes given that both sides know
the labels. Labels may be recomputed between rounds from what was broadcast (as later attention layers do). The rule
must be restricted (group-by-label, sort-by-label, contiguous chunks), and the labels charged: otherwise labels can
carry the whole item and the "router" does the whole computation (**PROVEN**, trivially).

**D3. Bond dimension and min-cut.** The rank of the matricisation M_S[x_S, x_{S^c}] = F(x) across an item cut, for F
a function or a probability table. For a probability table, the nonnegative rank is the smallest number of latent
*states* making the two sides conditionally independent (**PROVEN**, standard). For deep arithmetic-circuit networks
the min-cut of the network's tensor-network graph bounds this rank (Levine et al., arXiv:1704.01552); hierarchical
tensor formats are built on these ranks, with quasi-optimal truncation (Grasedyck 2010). A different rank matters for
messages: the parameter-observation rank of a likelihood family, the rank of [log p(x_S | theta)] over (x_S, theta).
For a d-parameter exponential family it is at most d + 2, and d after dropping the terms that depend on x alone or on
theta alone (they do not change the posterior); as a likelihood rather than a log-likelihood it is typically much
larger, infinite for continuous non-degenerate families (**PROVEN**). This second rank bounds a message's dimension
when theta is the only open latent (section 5.1).

**D4. Treewidth of the dependency structure, and its non-Gaussian part.** Treewidth of (a) the computation's DAG,
(b) the graphical model of the items or activations, (c) the discrete (non-Gaussian) part of a conditional-Gaussian
model. D4(a) implies D1: a Boolean circuit of treewidth w compiles, in time linear in its size and singly exponential
in w, into a structured d-DNNF (Amarilli, Capelli, Monet, Senellart, arXiv:1811.02944), which supports exact model
counting and expectations under product laws. A structured circuit along a tree over the inputs is related to D1's
deterministic messages but not the same object: a node with w gates can carry up to 2^w residual classes. Lower
bounds on DNNF size go through multi-partition communication complexity (Bova, Capelli, Mengel, Slivovsky, IJCAI
2016). **PROVEN** (cited).

**D5. Mutual information across scales; renormalisation.** The block law I(X_B; X_{B^c}) of the data as the block
grows (volume, area, log); renormalisation as the merge map acting on the laws of messages, and its fixed points.

**D6. Hierarchical low rank (H-matrices, fast multipole).** For interactions y_i = sum_j K(p_i, p_j) v_j over a
hierarchy (possibly built from the inputs), every well-separated block has epsilon-rank at most k(s) at scale s.
Scale-free: k independent of s, with the same translation rules at every scale (Greengard and Rokhlin 1987).

## 3. How the definitions relate: separations

Each line is a task on which two definitions disagree.

| Separation | Evidence | Status |
|---|---|---|
| D1 vs D2: keyed aggregation (binding) | Omega(N) bits at a node of every fixed hierarchy; O(1) bits plus log N label bits routed (T5) | PROVEN |
| D1 vs D3: f = sum_i a_i x_i, x in {0,1}^n, weights with distinct subset sums (e.g. a_i = 2^i) | rank 2 across every cut; 2^{\|S\|} classes | PROVEN |
| D3 vs estimation: the same f, integer weights | E f is trivial; the tail P(f > t) is the expectation of the indicator 1[f > t], whose rank across a balanced cut can reach 2^{n/2}; computing 2^n P(f > t) is counting knapsack solutions, #P-complete (classical; stated in Dyer 2003). Low rank does not survive a nonlinear readout | PROVEN |
| D4 vs D1/D3: f = x^T U x mod 2, U the complete graph | the pairwise interaction graph has treewidth n - 1; f depends only on \|x\| mod 4, so 2 bits suffice at every cut (rank-width 1) | PROVEN |
| D4 vs SFU: Bayes with one shared target theta, Gaussian case x_i = theta + e_i | with theta integrated out, every pair of items stays dependent given all the others (the precision matrix of x has no zero off-diagonal entry), so the item graph is complete (treewidth N - 1); kept explicit, theta is a single open latent at every cut | PROVEN (Gaussian case; generic otherwise) |
| D1/D3 vs tractability: x^T U x mod 2 on random graphs | linear rank-width (Lee, Lee, Oum, arXiv:1001.0461), yet sum_x (-1)^{x^T U x} takes polynomial time (Dickson) | PROVEN |
| untangled vs learnable: parity | 1 bit across every cut of every hierarchy; hard for statistical-query learners, which include gradient methods at coarse precision; World 3's XOR chains were never learned | PROVEN; MEASURED (World 3) |
| D5 vs D1: inner product mod 2 across a cut, independent uniform inputs | I(X_S; X_{S^c}) = 0, yet Omega(N) bits across the cut even on average (Chor and Goldreich 1988; for disjointness the hard distribution is not a product one) | PROVEN |

So: D1 counts precision and misses routing; D3 counts dimensions but misses tails; D4 at the item level measures a
graph, not a function, and calls the simplest aggregation maximally tangled; D5 describes the data, not the
computation.

## 4. Verdict: the open-latent profile, with routing and rounds

Kelvin's intuition has three parts: complexity may grow with scale; it is fine for larger structures to have more
dimensions; what stays simple is how information combines. The definition that matches all three, and that the
theorems below support, is:

**Definition (scale-free untangled, SFU).** A task family admits an SFU solution with routing cost b, open-latent
profile o(.) and R rounds if
1. (routing) each item carries a label of at most b bits, and a restricted rule (group-by, sort, contiguous chunks)
   builds the hierarchy from the labels; labels may be recomputed between rounds from what was broadcast;
2. (content) the message at a node is the likelihood of the latents still open there (attached inside and outside
   it), with its size counted in latent states or in the epsilon-rank of the message family at a stated precision:
   a global function of the open latents in one round, a local expansion around a broadcast estimate with more
   rounds;
3. (combination) every merge, at every scale, is the same two operations: add log-likelihoods, and sum out the
   latents that close at that node.

Untangled means b = O(log N) and, at every merge, the open latents of the two children taken together (which include
those that close there) stay small in states or epsilon-rank: bounded, or growing with the number of groups rather
than of items. Bounding only the parent's open latents is not enough: summing out a closing block whose internal
structure is an expander is #P-hard even when nothing stays open.

This is junction-tree message passing with the items as leaves. Open latents are the separators (Lauritzen and
Spiegelhalter 1988); the size of what is merged along an elimination order is the induced width (Dechter 1999);
routing by labels is a form of context-specific independence (Boutilier, Friedman, Goldszmidt, Koller 1996). What is
added here is the reading of Kelvin's three phrases in these terms, the role of rounds, and the measurements.

The intrinsic quantity is the task's minimal message (the Nerode class of T1, or its internal epsilon-cover at a
stated precision); the open latents explain it. Counting latent *variables* is not invariant (two can be merged into
one), so sizes are counted in states or epsilon-rank. For a single cut, the smallest number of latent states making
the two sides conditionally independent is the nonnegative rank of the joint law across that cut (**PROVEN**); this
bounds the separator at that cut, not the profile of one latent model consistent across all cuts.

How the three parts of the intuition map:
- "zooming out, things get more complicated": at larger scales the open latents are those of larger structures
  (groups, groups of groups); they may be richer, so the message dimension may grow with scale;
- "it is fine for larger structures to have more dimensions": the message size is set by the open latents at that
  scale, not by how many items lie below; under noise it must in fact grow slowly (5.7);
- "how information combines stays simple": every merge is addition of log-likelihoods plus summing out what closes,
  at every scale; what changes with scale is which latents are open.

This note is about cuts across items. Cuts across depth (how many directions of layer l still matter for the output)
are the Lyapunov lane's subject (pool-yw7kwn.1-.8); in a computation with R rounds, R messages cross each item cut, so
the two pictures meet in the (items x depth) grid of a transformer.

In this language D1 is SFU with a fixed hierarchy, one round, counting bits; D3 is the message dimension; D4 at the
item level is what remains after the latents are summed out (a shared latent turns its items into a clique), which is
why it overstates tangledness; D5 bounds how much the input law itself must carry across a cut; D6 is the linear
special case where the "latents" are multipole moments. Status: the definition is a choice, not a claim; T2-T4 below
are what make it the natural one.

## 5. Theorems and sharp claims

### 5.1 Messages are likelihoods of the open latents (T2)

Take a joint model p(x, u) = prod_c psi_c(u_c) prod_i p(x_i | u_{L(i)}), with items attached to latent sets L(i).
Assign each factor psi_c to the lowest hierarchy node whose subtree contains every item attached to the latents in
its scope (prior factors of latents open everywhere end at the root). At a node S a latent is **closed** if every item
and every factor that involves it is assigned inside S, **open** otherwise; the target is never closed (it stays open
up to the root, which normalises over it). The message m_S(u_open) is the sum over the
latents closed inside S of the product of the item likelihoods and assigned factors inside S; the merge multiplies the
children's messages and the factors assigned to the node, and sums out the latents that close there (variable
elimination along the hierarchy; junction-tree separators). **PROVEN** (standard). The posterior of the target is the
normalised root message.

Minimality. If the outside data can produce likelihoods that separate functions of the open latents, two contents
with non-proportional messages give different joint posteriors of the open latents for some completion, so no smaller
message is exact for that joint posterior (**PROVEN UNDER** this richness condition). For a single target y the
minimal message can be smaller: a direction of m_S that no completion couples to y (an open nuisance latent the
outside data never links to y) can be dropped.

Consequences:
- Tiny-PFN world (`work/tiny-pfn/world.py`): each source's reliability v_s closes inside the source; the hidden answer
  y* is attached to every source's query reading, so it is the only open latent at a cut between sources: messages
  are functions of y*. At a cut inside a source v_s is open, and y* is open only if the cut contains that source's
  query reading. **PROVEN** (from the model).
- Shared-error groups with visible membership: each group's shared errors are open at every cut that splits the
  group, so message size grows with the number of **straddling groups**, not of items; under a hierarchy aligned with
  the groups (routing by the membership label) only y* is open. With membership hidden and T past questions, each
  source's likelihood involves every group's shared errors on every past question, so about G T + 1 latents are open
  at a cut between sources, until a pilot round lets sources be re-routed into group-aligned subtrees. **PROVEN**
  (from the model).
- Correlated errors with an inverse-Wishart covariance (World FA variant (b); N traders, T past questions; E is the
  T x N error history). The full posterior of Sigma depends on the scatter E^T E, whose cross block E_S^T E_{S^c}
  would need the whole history E_S (T |S| numbers) to cross a trader cut. The predictive does not need Sigma. With
  prior scale Psi = diag(Psi_S, Psi_{S^c}), Sylvester's identity gives
  |Psi + E^T E| = |Psi| |I_T + E_S Psi_S^{-1} E_S^T + E_{S^c} Psi_{S^c}^{-1} E_{S^c}^T|, so the marginal likelihood,
  and the query predictive (append the query row o - y* 1 as one more time step), depend on each side only through
  the (T + 2) x (T + 2) matrix sum_{j in S} psi_j^{-1} w_j w_j^T, with w_j = (trader j's T past errors, its query
  reading, 1) (diagonal Psi_S shown). These add across trader cuts: O(T^2) numbers,
  independent of |S|. If Psi^{-1} is block-diagonal plus rank k (a common source), each side also sends the k summed
  vectors sum_j w_j u_j^T, so O(T^2 + k T) numbers, still additive; a cut inside a dense block of the prior scale is
  not additive. **PROVEN UNDER** the matrix-normal / inverse-Wishart model with such a prior scale (the
  dual-coordinate argument is Gemini's review correction of our first draft, which claimed T |S|). So this apparently
  global coupling is untangled in time coordinates: what is open across a trader cut is the T-dimensional time
  structure, not the N(N+1)/2 entries of Sigma. A volume law in |S| returns only if Sigma itself is the target, or
  the prior scale couples the two sides densely.

### 5.2 Fixed-size additive messages: exactly the exponential families; smooth families pay polylog

- **PROVEN UNDER** regularity (densities with support independent of theta and a continuously differentiable
  sufficient statistic; Koopman 1936, Pitman, Darmois): for iid sources, a sufficient statistic of fixed dimension for
  every N exists only for exponential families, where it is additive. Uniform(0, theta) shows the regularity is
  needed: its statistic max_i x_i merges by max, a monoid but not a sum.
- Distributed-statistics form: for combining local maximum-likelihood estimates, the asymptotic (second-order)
  information loss of the best one-shot combination is governed by how far the family is from a full exponential
  family, i.e. Efron's statistical curvature (Liu and Ihler, arXiv:1410.2653, under their regularity conditions).
  **PROVEN** (cited).
- Smooth families (**PROVEN UNDER** a prior supported on a compact box B, and l(theta; x) minus a constant c(x)
  analytic in theta on a common complex neighbourhood of B and bounded there by M uniformly in x): Chebyshev
  interpolation reaches sup error epsilon with O(log(M/epsilon)^d) terms. Per-source sup error epsilon gives
  log-posterior error at most 2 N epsilon, so KL <= 2 N epsilon, and O(log(M N/delta)^d) additive numbers per source
  suffice for KL <= delta. The tiny-PFN world meets this only approximately (Gaussian prior truncated to [-5.5, 5.5];
  unbounded readings make M grow slowly with the reading).
- **MEASURED** (N1e: the tiny-PFN world scaled to N sources; each source sends the r Chebyshev-interpolation
  coefficients of its log-likelihood of y* on [-5.5, 5.5]; the root sums them; KL(Bayes || approx) with the
  approximation normalised over the whole range, 95th percentile over 200 episodes at N = 3 and 64, 60 at N = 1024,
  25 at N = 4096; N1c gave the same numbers within a factor of about 1.5 with local normalisation):

  | N | r = 24 | r = 32 | r = 48 | r = 64 | r = 96 |
  |---|---|---|---|---|---|
  | 3 | 8.7e-3 | 2.7e-3 | 1.6e-4 | 8.0e-6 | 3.8e-8 |
  | 64 | 2.9e-2 | 1.2e-2 | 1.1e-3 | 7.0e-5 | 4.2e-7 |
  | 1024 | 7.3e-2 | 2.5e-2 | 1.0e-3 | 9.8e-5 | 8.7e-7 |
  | 4096 | 1.6e-1 | 4.7e-2 | 9.8e-4 | 1.0e-4 | 9.8e-7 |

  KL <= 1e-3 is crossed between r = 48 and r = 64 at every N from 64 to 4096 (r = 48 at N = 3).

  At fixed r = 64 the KL grows about tenfold from N = 3 to N = 64 and then barely moves up to N = 4096, far gentler
  than the worst-case 2 N epsilon; each extra 32 coefficients cut it about 100-fold, so the needed r grows like
  log N at most, as T3a says. The constant is large (tens of numbers) because the per-source log-likelihood
  switches between its three variance components over short distances.

### 5.3 Rounds: broadcasting a consensus turns global functions into local statistics (T4)

One round (each source writes a message without seeing anything else; merge = sum): the message must fix the
log-likelihood on the whole region where the final posterior could lie, i.e. a global function of the open latents.
For location-type families its size is roughly the number of resolution cells of that region, (prior width / feature
width)^D times a polylog in N/delta, so it grows exponentially in the number D of open latents (**HEURISTIC**: it is
the eigenvalue count of a smooth translation-invariant kernel on a box, as in Landau-Slepian time-frequency limiting;
a reviewer expects standard n-width lower bounds to make it a theorem, not checked here; **MEASURED** below). In
words: one round must ship the likelihood everywhere it might matter; two rounds ship it only where it does.

Two rounds: round 1 sends a few additive numbers per source for a pilot; the root broadcasts it; round 2 sends the
C(D + p, p) - 1 non-constant coefficients of a degree-p expansion of each source's log-likelihood on a window of
radius about sqrt(log N / N) around the pilot; the root sums, truncates to the window and normalises. Inside the
window the per-source remainder is O(h^{p+1}), so the log-posterior error is about N (log N / N)^{(p+1)/2}, and the
KL between the posterior restricted to the window and its approximation, about half the variance of that error, is
about N^{1-p} up to logs. Hard truncation makes the full forward KL infinite when Bayes has mass outside the window;
a positive tail extension (for example the coarse round-1 message outside the window) adds a tail term bounded by the
outside mass, not analysed here. **PROVEN UNDER** (for the window-conditional KL): p + 1
bounded derivatives near the truth; posterior concentration, with mass outside the window o(1) (a global
identifiability / Bernstein-von Mises condition); a pilot inside the window with high probability (moderate-deviation
concentration at rate sqrt(log N / N)); and a unimodal posterior (label symmetry, as with exchangeable hidden groups,
must be broken first). The bound holds with high probability over the data. The pilot condition holds for the models
simulated here (the tiny-PFN GLS pilot; the N2 contamination pilot sum x / ((1 - eta) N)); for general mixtures,
moment pilots can converge more slowly than root-N (Wu and Yang, arXiv:1807.07237).
This is the classical one-step estimator (van der Vaart 1998, sec. 5.7); related distributed methods add one round
of communication (Huang and Huo, arXiv:1511.01443; Jordan, Lee and Yang, arXiv:1605.07689), with lower bounds for
protocols with and without broadcast (Duchi, Jordan, Wainwright, Zhang, arXiv:1405.0782). It is also Kelvin's design
for the aggregation track: agents write their messages after reading the consensus.

**MEASURED**:
- Tiny-PFN world, one open latent (N1e; one round normalised over the whole range; two rounds = a 3-number GLS pilot
  from the labelled residuals, then p + 1 Chebyshev coefficients of each source's log-likelihood on a window of 10
  pilot standard deviations, normalised over the window, so this column is a window-conditional KL; KL at the 95th
  percentile; "fails" = the exact posterior had mass above 1e-12 of its peak outside the window in more than 5% of
  episodes):

  | N | one round, 16 numbers | one round, 64 numbers | two rounds, 3 + 5 = 8 numbers | two rounds, 3 + 7 = 10 numbers |
  |---|---|---|---|---|
  | 64 | 0.36 | 7.0e-5 | fails (window missed the posterior in 10% of episodes; N1c's looser test: 1.8e-3) | fails (same; N1c: 7.6e-5) |
  | 1024 | 3.3 | 9.8e-5 | 8.6e-8 | 5.4e-11 |
  | 4096 | 14 | 1.0e-4 | 2.5e-10 | 1.7e-12 |

  At N >= 1024 the two-round error falls steeply with N at fixed size (about N^{-4} from 1024 to 4096 at p = 4,
  at least as fast as N^{1-p} predicts), while the one-round error stays put or creeps up; at a similar budget, one
  round is useless (16 numbers: KL 3.3 at N = 1024, against 8.6e-8 for 8 numbers over two rounds). Below N of about
  64 the window is not local: two rounds fail at N = 3 in 89% of episodes and at N = 64 in 10% (N1c, with a looser
  window test, found 2.5% at N = 64). Rounds pay off once the posterior is narrow compared with the features of each
  source's likelihood.
- Small codes of the kind a network might learn (N1d; mean KL over 300 episodes, 100 at N = 1024):

  | N | one round, 2 numbers (GLS with E[1/v given residuals]) | two rounds, 2 + 2 numbers (pilot, then sums of l' and l'' at it; Laplace) | three rounds, 2 + 2 + 2 numbers |
  |---|---|---|---|
  | 8 | 0.13 | 1.2e-2 | 7.0e-3 |
  | 64 | 0.23 | 1.9e-4 | 8.6e-5 |
  | 256 | 0.20 | 1.1e-5 | 4.5e-6 |
  | 1024 | 0.16 | 4.9e-7 | 2.7e-7 |

  The 2-number one-round code has a floor that does not fall with N (it cannot use how far each reading sits from the
  consensus, which also signals reliability); 4 numbers over two rounds fall roughly like N^-2 here. Budgets differ
  (2 vs 4 numbers), but the equal-budget comparison in the table above points the same way. At N = 3 neither works
  (Laplace is poor for a wide posterior). A re-implementation of exact Bayes in the same script gives KL below 1e-15.
- D open latents (N2; sources of unknown reliability reporting on D shared latents: reliable with probability 0.7,
  N(theta, 0.25 I), otherwise N(0, 4 I)). Column 2 is the empirical linear epsilon-rank of 3,000 sampled
  log-likelihood functions on the box (RMS error), a property of the sample, not a lower bound for all encoders;
  columns 3-4 are tensor Chebyshev codes scored on a window around the exact posterior (approximation mass elsewhere
  ignored):

  | D | empirical one-round linear rank, RMS 1e-2 / 1e-3 | one round, tensor code, KL <= 1e-3 | two rounds, tensor code, KL <= 1e-3 |
  |---|---|---|---|
  | 1 | 12 / 19 | 24 (N = 512) | 7 (N = 512) |
  | 2 | 128 / 334 | 1,024 (N = 512) | 28 (N = 512) |
  | 3 | 746 / 1,646 (likely low: 3,000 samples) | 32,768 (N = 256) | 347 (N = 256) |

  The empirical one-round rank grows about tenfold per extra open latent. The two-round tensor codes are much smaller,
  though a tensor basis itself grows exponentially in D ((p + 1)^D); a total-degree basis would need
  C(D + 4, 4) + D + 1 = 7, 18, 39 numbers at degree 4 (count only, not run). Two rounds fail at N = 64 for D = 3 with
  degree <= 6 (window not yet local).

Reading: rounds, not hierarchy shape, are the main lever on message size once routing is solved. A transformer gets
rounds for free (each layer reads pooled context, then writes), which is one way depth can substitute for width.

### 5.4 Routing: what it costs, and when density is needed

**T5, routing separation (PROVEN).** Items (k_i, a_i) with keys k_i and bits a_i; f = 1 iff some key carries two
items with a = 1. (i) Every fixed hierarchy has a node with N/3 <= |S_v| <= 2N/3; placing the keys so that about N/3
of them straddle that cut (other values 0) turns f into the complement of set disjointness on N/3 bits. A tree
computation induces a one-way protocol across that cut, so the message at v needs Omega(N) bits even with public
randomness and error 1/3 (Razborov 1992; Kalyanasundaram and Schnitger 1992). (ii) With labels = keys (log N bits)
and the group-by rule, every content message is at most 2 bits.
The Bayes analogue (World FA: flat tokens in random order, per-trader statistics combined nonlinearly, so a balanced
positional node must pass every straddling trader's partial statistic) is **HEURISTIC**: it needs its own
indistinguishability argument, not given here. The reading of World FA's failure as a learning failure (one head can
route by trader ID, T6, and the binding-given arms learned) is likewise **HEURISTIC**: no full construction of the
trained architecture is given.

**T6, routing cost in attention = sign-rank (PROVEN).** A hard-threshold head (i attends to j iff q_i . k_j >= t)
realises the routing relation R with key dimension d only if the sign pattern of R has sign-rank at most d + 1
(absorb t into one extra coordinate), and sign-rank d suffices (t = 0); this is the definition of sign-rank.
Equality on N labels has sign-rank at most 3 (1/2 - (i - j)^2 = (2i, 1/2 - i^2, 1) . (j, 1, -j^2)), order has 2;
with noise or bounded logits, random +-1 codes of length O(log N) give margin 1/2 (Hoeffding and a union bound).
Hadamard-type relations ("attend iff <i, j> mod 2 = 0") need sign-rank at least sqrt(N) (Forster 2002). Natural
routing (same source, same question, before/after, nearby) is cheap; parity-like routing is not.

**Sharp selection needs the logit range of Alman and Song's hard regime (PROVEN, arithmetic plus arXiv:2302.13214).**
For soft attention to put 1 - delta of its mass on one item out of N, the competitors' weights must sum to at most
delta / (1 - delta), so the logit range max_j L_j - min_j L_j must be at least log((N - 1)(1 - delta) / delta). With
A = exp(Q K^T / d) and entries of Q, K in [-B, B], every logit lies in [-B^2, B^2], so B >= sqrt(log((N-1)(1-delta)/delta) / 2),
of order sqrt(log N). Alman and Song prove (worst case, d = O(log n), additive error 1/poly(n)) that attention can be
approximated in n^{1+o(1)} time when B = o(sqrt(log n)) and not in n^{2 - Omega(1)} time when B = Theta(sqrt(log n)),
assuming SETH. So attention that selects sharply sits in the regime where no generic fast algorithm exists; this does
not make every sharp routing task hard. Label routing (equality) is computed in O(N log N) by sorting or O(N) by
hashing; quadratic cost is unavoidable for generic sharp similarity search (also Rubinstein, arXiv:1803.00904).

**Routing by keys is a join (correspondence; cited theory).** Grouping items whose labels are equal and aggregating
within groups is an equi-join followed by an aggregate. Database theory treats joins, constraint satisfaction and
graphical-model inference as one sum-product problem whose exact cost is governed by width parameters of the query
(FAQ: Abo Khamis, Ngo, Rudra, arXiv:1504.04044): the routing varies with the data, the query structure does not.

**Per query, attention is already scale-free (PROVEN).** For one query, the triples (max logit m, normaliser l,
weighted value sum o) over any set of keys merge associatively: m = max(m_a, m_b), l = l_a e^{m_a - m} + l_b e^{m_b - m},
o likewise (the online-softmax rule used by FlashAttention). So one query's attention is a monoid reduction along any
hierarchy with d + 2 numbers per message. Dense attention runs N such reductions, one per query, per layer; the number
of genuinely different reductions per layer is the number of **routing classes** (distinct attention rows up to a
tolerance). This is the measurable form of "dense attention pays for routing": cost N x N, information about C x N,
with C the number of routing classes.

**T7, correlation-based routing (PROVEN for the promise problem on sign strings; HEURISTIC for iid histories).** Let
items carry T-step sign histories and define correlation as their normalised inner product. Deciding correlation
>= rho versus <= -rho needs Omega(min(T, 1/rho^2)) bits of communication per pair, with public randomness: for
rho >= 2/sqrt(T), gap-Hamming (Chakrabarti and Regev, arXiv:1009.3460) on n = floor(4/rho^2) coordinates, each
repeated about T rho^2 / 4 times (rounding changes constants only); for smaller rho, the unrepeated length-T
gap-Hamming instance already has the larger gap. Upper bound: with shared random coordinates, sending
O(log(N/delta) / rho^2) of them per item (capped at T) lets every pair among N items be decided correctly with
probability 1 - delta (union bound). So per-item keys need about min(T, 1/rho^2) bits, up to the log N factor: cheap
above the statistical detection threshold, linear in T at it.

### 5.5 Multi-hop needs rounds; parity is untangled but hard to learn

- k-step pointer chasing (two tables f_A, f_B held on the two sides of a cut): protocols with k - 1 rounds need
  Omega(n - k log n) bits deterministically and Omega(n/k^2 - k log n) randomised, while k rounds need O(k log n)
  (Nisan and Wigderson 1993; constants as reported by later papers, the primary text was not checked). A one-pass tree
  computation induces a one-way protocol across each of its cuts, so at any cut that puts the two tables on opposite
  sides, lookups of 3 or more hops need Omega(n) bits (**PROVEN** for such cuts). It is not a statement about every
  hierarchy: an unbalanced pass whose leaf order follows the hop order can carry the current pointer upward and do two
  hops with log n bits. Transformers: constant attention layers simulate constant rounds of massively parallel
  computation, and log depth suffices for k-hop tasks by pointer doubling (Sanford, Hsu, Telgarsky,
  arXiv:2402.09268), which moves Omega(n log n) bits per layer; one layer needs exponentially more size than two for
  induction heads (arXiv:2408.14332). **PROVEN** (cited).
- Parity is maximally untangled under every definition here (1 bit across every cut; rank 1 in +-1 form), yet a family
  of unknown parities is hard for statistical-query learners. Gradient methods fall in that class when the gradient
  precision is coarse relative to the batch size; with fine enough precision SGD can simulate any PAC learner (Abbe,
  Kamath, Malach, Sandon, Srebro, arXiv:2108.04190), so the barrier is a regime, not an impossibility; gradients carry
  exponentially little signal about which parity is the target (Shalev-Shwartz, Shamir, Shammah, arXiv:1703.07950).
  **PROVEN** (cited). World 3 (XOR chains) is consistent with this: the equality version was learned, the XOR version
  never was. **MEASURED** (outbox/world3-chains-pilot). Lesson: untangledness is about what a solution must carry,
  not whether training finds it.

### 5.6 Pairwise structure, density, and the non-Gaussian part

- For f(x) = x^T U x mod 2 with interaction graph A, the classes across a cut are determined by (q_S(x_S), B^T x_S)
  with B = A[S, S^c], so rank_GF2(B) <= log2 K_f(S) <= rank_GF2(B) + 1 (each fibre of x_S -> B^T x_S contributes one
  or two values of q_S). The best hierarchy's worst message is therefore the rank-width of A, up to 1 bit.
  **PROVEN**. Complete graph: rank-width 1, treewidth n - 1; G(n, 1/2): rank-width about n/3; sparse random
  G(n, c/n), c > 1: linear (Lee, Lee, Oum). **MEASURED** brute-force check (N3b, n = 12, balanced cut, all 2^12
  inputs): complete graph 4 classes = 2^(1+1), cut-rank 1, treewidth 11; G(12, 1/2) 64 classes, cut-rank 6; a random
  3-regular graph 48 classes, cut-rank 5 (between 2^5 and 2^6, as the bound allows).
  (The real rank of the +-1 matrix (-1)^f across S is 2^{rank_GF2(B)}: D3 sees the same structure, exponentially.)
  So density is not tangledness. The measure is the rank of what crosses the cut.
- Algebra beats structure for expectations: Gaussian (any density, cubic time), GF(2) quadratic forms under the
  uniform law (Dickson), planar zero-field Ising (Kasteleyn, Fisher). **PROVEN**. Tails and non-uniform laws bring
  hardness back (knapsack above).
- Non-Gaussian part. In conditional-Gaussian models (discrete parents, Gaussian children) exact propagation works on
  a strong junction tree whose discrete cliques include every discrete variable coupled through continuous paths
  (Lauritzen 1992; Lauritzen and Jensen 2001), and inference is NP-hard even on polytrees, approximately too (Lerner
  and Parr 2001, arXiv:1301.2288). **PROVEN** (cited). In a ReLU net the gates are discrete children of Gaussian-like
  pre-activations, so every gate is coupled to every other through the dense pre-activations: "treewidth of the
  non-Gaussian part" is only meaningful after that coupling, i.e. as the entropy of the gate pattern. **HEURISTIC**.
  This matches the latent-hub design: a latent xi is a separator only if the next layer's gates depend on the
  Gaussian residual weakly enough to close (the boundary-family kill tests failed at depth).

### 5.7 Estimation (M1/M3): two bond dimensions, alignment, expectations vs tails, noise

- **T11a, exact output law (PROVEN, as an algorithm).** If f has K_v classes along T and the input law is a latent
  tree along the same T with R_v latent states at node v (given u_v, the two child subtrees are independent), then
  Q_v(c | u_v) = sum over (c_a, c_b) merging to c of [sum_{u_a} P(u_a | u_v) Q_a(c_a | u_a)] [sum_{u_b} P(u_b | u_v) Q_b(c_b | u_b)],
  so the exact law of f(X) (every tail) costs O(sum_v R_v (R_a K_a + R_b K_b + K_a K_b)), i.e. O(R K (R + K)) per
  node for uniform sizes. This is a sufficient route, not a necessary one: algebraic structure (5.6) can bypass it,
  and a symmetric f can be evaluated along the law's own tree. What it shows is that alignment of the computation's
  hierarchy with the input law's makes exact tails cheap.
- **Misalignment.** If the law is a tree along T1 and the computation along T2, contraction of the union costs
  poly(size) x D^{O(w)}, with D the bond dimension and w the treewidth of T1 union T2 (Markov and Shi,
  arXiv:quant-ph/0511069; upper bound). **MEASURED** (N3a; random binary trees by random pairwise merging; treewidth
  of the union graph bracketed by a contraction lower bound (MMD+) and a min-fill upper bound; 5 pairs per N):

  | N leaves | T2 = copy of T1 | 4 random prune-and-regraft moves | 32 moves | T2 independent of T1 |
  |---|---|---|---|---|
  | 16 | 2 | 3.4-4.0 | 4.0-6.0 | 4.2-6.6 |
  | 64 | 2 | 3.8-4.2 | 5.4-14.2 | 6.8-23.6 |
  | 256 | 2 | 4.0-4.8 | 5.8-17.8 | 12.6-95.6 |

  Aligned hierarchies stay at treewidth 2; misalignment raises it steadily, and for independent trees the upper
  bound grows about linearly (0.37 N) while the lower bound grows slowly, so linear growth is not established
  (**CONJECTURE**: linear for independent random trees; a reviewer suggested an expander-minor argument, not checked).
- **Expectations need rank; tails need dimension (PROVEN).** A rank-chi tree network gives E_p f by contraction in
  O(N chi^3 R^3). Tails are expectations of indicators, whose rank can be exponentially larger: for integer weights,
  2^n P(sum_i a_i x_i > t) under the uniform law counts knapsack solutions, #P-complete in the worst case. With
  activation noise sigma and range R per coordinate, a d-dimensional message has about (R/sigma)^d distinguishable
  states, and the law of a sum is a convolution on that grid (pseudo-polynomial; exact up to the discretisation set by
  sigma).
- **Noise forces dimension to grow with scale (PROVEN UNDER** theta ~ N(0, 1); sources x_i | theta ~ N(theta, 1/I)
  independently, so the posterior is N(mu_N, v_N) with v_N = 1/(1 + N I); the root sees one m-dimensional message
  M = s(x) + noise with |s_k| <= R and independent N(0, sigma^2) noise per coordinate; the root's estimate of theta
  has mean-squared error at most c v_N**).** Rate-distortion gives I(M; theta) >= (1/2) log2((1 + N I) / c), and the
  Gaussian channel gives I(M; theta) <= I(M; s) <= (m/2) log2(1 + R^2/sigma^2), so
  m >= log2((1 + N I) / c) / log2(1 + R^2/sigma^2). A Gaussian output within forward KL delta of the posterior on an
  input has squared error at most (e^{2 delta} - 1) v_N about mu_N there, so c = e^{2 delta} if that holds for almost
  every input and noise draw. Bounded (normalised) noisy representations must widen like log N; unnormalised sums
  need not. So "larger structures need more dimensions" is a necessity, not a permission.
- **Additive messages give certified tails (PROVEN UNDER** finite moment generating function of each message near zero,
  e.g. bounded messages**).** Given the latents, a sum of independent messages has a computable cumulant generating
  function, so Chernoff bounds certify upper tails of any linear readout of the sum; saddlepoint approximations have
  relative error O(1/N) under the usual regularity (Daniels 1954). This applies to the additive log-odds message
  before any nonlinear readout. It is the M3 route for an SFU architecture.

### 5.8 The renormalisation view, made precise

- Latent-tree data (a complete b-ary tree with q states per latent, leaves in their natural left-to-right order) have
  block mutual information at most (2 log_b L + O(1)) log2 q for a contiguous block of L leaves: the ancestors along
  the block's two boundary paths separate it from the rest. **PROVEN**. Log law, like 1D critical states and MERA,
  against the volume law of generic data and the constant of Markov chains.
- Correlations: in the same tree, two leaves at distance l have tree distance at least 2 log_b l; for a stationary
  binary symmetric channel with second eigenvalue lambda = 1 - 2 epsilon, their correlation is lambda^{tree distance},
  hence at most l^{-2 log_b(1/lambda)}: power-law decay at most. **PROVEN** (bound). Lin and Tegmark
  (arXiv:1606.06737) make the grammar version.
- Whether information about the root survives coarse-graining is a sharp threshold: for a binary symmetric channel on
  a tree, reconstruction is possible if br(T)(1 - 2 epsilon)^2 > 1 and impossible if it is < 1 (Evans, Kenyon, Peres,
  Schulman 2000, the Kesten-Stigum bound); the equality case is not settled by this criterion. **PROVEN** (cited).
  Relevant vs irrelevant directions in the RG sense have this exact meaning on trees.
- For additive merges with product input laws, the rescaled law of the message converges to a Gaussian (finite
  variance) or a stable law (in a domain of attraction); max-merges converge to an extreme-value law when the per-item
  law is in its domain of attraction. **PROVEN** (classical). Under SFU with additive merges the large-scale messages
  are therefore near-Gaussian, the rationale for the latent-hub family keeping its non-Gaussian structure in a few
  latents (**HEURISTIC**).

### 5.9 Which tasks are untangled, by definition

| Task | D1 fixed cuts | D2 routed | D3 rank | D4 item treewidth | D5 data MI | SFU |
|---|---|---|---|---|---|---|
| Conditionally independent sources, exponential family | exact: infinite (continuous) or d log2(s + 1) bits (d count statistics); to posterior precision about (d/2) log s | same | parameter-observation rank <= d + 2 (log space) | N - 1 | about (d/2) log s | 1 open latent, d numbers, 1 round |
| Same, smooth non-exponential (tiny-PFN) | polylog at tolerance | same | r(delta/N) | N - 1 | log | 1 latent; 48-64 numbers in 1 round, about 4-8 in 2 rounds (N >= 64) |
| G shared-error groups, labels visible | about G r at a positional cut | r (group-aligned) | as D1/D2 | >= group size | log | log G label bits; open = straddling groups |
| G groups, membership hidden, T past questions | functions of about G T latents | no labels (re-route after a pilot) | grows fast | N - 1 | log | 1 round: exponential in the open latents; with a pilot and re-routing: local statistics, if label symmetry is broken |
| Keyed aggregation / binding (World FA) | Omega(N) | O(1) + log N label bits | full rank | high | 0 | untangled with routing |
| k-hop lookup (pointer chasing) | Omega(n) at a cut separating the tables | k sequential rounds: O(k log n) bits; pointer doubling: log k rounds, Omega(n log n) bits per round | high | high | 0 | untangled with k rounds |
| Parity | 1 bit | 1 bit | 1 (+-1 form) | n/a | 0 | untangled; hard for SQ learners |
| GF(2) quadratic on an expander | Theta(N) | Theta(N) | 2^{Theta(N)} (log-rank Theta(N)) | Theta(N) | 0 | tangled; uniform expectation still polynomial |
| Correlated errors, inverse-Wishart (World FA (b)) | full posterior of Sigma: T \|S\| numbers | same | - | dense | - | predictive: O(T^2) additive numbers per cut (time coordinates) |
| One-query retrieval (attention) | volume without broadcast | - | - | - | - | d + 2 numbers after broadcasting the query |

## 6. Transformers: routing plus combination, and how to test it

### 6.1 Attention is the merge of Gaussian messages, with the prior as a sink (PROVEN, identity)

Let tokens j carry Gaussian messages about y with precision tau_j and mean mu_j. A head with logits
L_j = log tau_j + c, values mu_j e_mu, plus a sink token with logit log tau_0 + c and value mu_0 e_mu + e_0, outputs
the posterior mean in the e_mu channel and a_0 = tau_0 / sum_{j >= 0} tau_j = tau_0 x (posterior variance) in the e_0
channel (the sum includes the sink). Without the sink (or another channel carrying the normaliser) the merged message
loses its total precision and is not closed under further merges. With a Gaussian prior N(mu_0, 1/tau_0), the sink is
the prior. In a multi-level merge the sink must enter once (at the root), or be subtracted at intermediate nodes;
otherwise merging two sunk subtrees counts tau_0 twice. Oracle GLS in the tiny-PFN world (reference b) is exactly one
such head. Exponential-family messages merge the same way in natural parameters, with an MLP undoing the
normalisation (construction; exact up to MLP approximation, which degrades as the sink weight a_0 ~ 1/N shrinks).
Attention sinks as such were named by Xiao et al. (arXiv:2309.17453); the identity with the Bayesian prior is the
point here.

### 6.2 The decomposition

For layer l and head h, write the logits as L_ij = R(lambda_i, lambda_j) + w_j + e_ij: a **routing** term from
discrete labels (same source, same question, token type), a **weighting** term per key (a log-precision-like scalar
from the key's content), and a residual. Values are the **content** (natural-parameter-like messages). Layers are
**rounds**. "Dense attention mostly pays for input-dependent routing" becomes four testable claims:
(a) rows cluster into few routing classes C, explained by labels; (b) given routing and weighting, the content is
low-dimensional and merges additively; (c) a static (input-independent) routing of the same sparsity fails while
label routing does not; (d) the union of supports over inputs is dense while each input's pattern is sparse.

Two consequences.
(i) Expressivity (**CONJECTURE**, with a construction sketch for one-level routing): a transformer whose labels are
linearly readable can implement a one-level R-round SFU computation with message size k in O(R) layers and width
O(k + label dimension): per round, one head gathers each group's natural parameters (normaliser kept by a sink), and
an MLP turns sums into the next round's local statistics. The sketch does not account for the cost of approximating
the local maps by MLPs or for attention precision, which is why it is not marked PROVEN. A multi-level
hierarchy of depth H needs about H layers per round, and undoing a normaliser of order 1/N needs MLP precision growing
with N. So density is not needed to *represent* SFU computations of this kind; it is needed when routing is by
similarity rather than labels (T6a, T7). Multi-step in-context estimation by transformers is established in other
settings (von Oswald et al., arXiv:2212.07677; Ahn et al., arXiv:2306.00297; Bai et al., arXiv:2306.04637).
(ii) Learning (**HEURISTIC**): at uniform attention (World FA's measured own/other ratio 1.00) the gradient on the
query-key form along the binding direction is a covariance between "same ID" and how useful the gathered content is,
which is near zero until the content path is useful, and the content path is useless until binding exists: a saddle
neither part can leave alone. Consistent with World FA's 0 of 15 flat runs and with success once binding was given;
plateaus of this routing-content kind are analysed for induction heads and in-context classification (Bietti et al.,
arXiv:2306.00802; Reddy, arXiv:2312.03002).

A caution about our worlds. In the tiny-PFN world the grouping (3 sources x 5 questions) is the same in every episode;
the net must find it by matching random per-episode ID embeddings (no positional encoding, `work/tiny-pfn/pfn.py`),
but an architecture designer can hard-wire it, and with a fixed token order a "static" pattern already encodes the
grouping. So the PFN prior tests content and combination; routing is tested by World FA (flat tokens, random order,
binding by trader ID), or by permuting token order per episode.

### 6.3 Measurements on our trained nets (all CPU; forward passes plus small fits)

Each measurement names what it tests, the Bayes reference, a prediction, and what would refute it. "Cut" means a set
of tokens; "replace" means redraw those tokens' errors from the prior while keeping the shown labels, the IDs and y*
(for a query reading, o' = y* + e'), so every modified episode is a valid draw.

| # | Measurement | Bayes reference | Prediction | Refuted if |
|---|---|---|---|---|
| M1 | **Interaction profile.** For a cut C and its complement, Delta = L(a,b) - L(a',b) - L(a,b') + L(a',b'), where L = log p(y1 \| x) - log p(y2 \| x) at points around y* and a, a' (b, b') are two draws of C's (the rest's) errors. Share = Var(Delta) / (Var of the two main effects). Report it for the net's log-odds and for its *deviation* from Bayes (net minus Bayes), which removes the trivial effect of the error's size. Cuts: one source vs the rest; two of one source's labelled questions vs the rest. | 0 exactly across sources (log-odds add over sources given y*, **PROVEN**); 0.039 within a source. Reference values, not floors: Bayes projected to a Gaussian 0.054; oracle GLS 0 | nets that cannot apply a per-source nonlinearity before mixing show less within-source interaction than Bayes (observed in 6.4, not pre-registered; to be re-tested on new nets); the cross-source deviation share separates architectures (**CONJECTURE**; did not hold in 6.4: 0.28-0.48 in every net) | within-source shares of one-layer nets at or above Bayes's |
| M2 | **Routing classes and label fit.** Per layer and head, cluster attention rows (greedy, total variation <= 0.05, 0.1, 0.2); C = clusters among the 15 context rows. On row-centred log-attention, remove per-key column means (weighting w_j) and regress the rest on pair-class one-hots (same source?, same question?, query-reading endpoints, self); report the column share and the label R^2, and the TV between softmax rows and their fitted version. Last layer: query-row mass on each source's query reading vs that source's log E[1/v \| residuals]. | GLS: one query class weighting readings by log precision; per-source gathering of residuals | **CONJECTURE**: C well below 15 in layer 1; labels explain most of the pair-specific part; query-row mass tracks log precision (phase B E9-E10 found reliability shaping layer-2 weights) | C near 15 with labels explaining little: routing is not label-like |
| M3 | **Routing substitutions (causal).** Replace a layer's logits by (i) R + w (fitted), (ii) R alone (w averaged within class), (iii) the net's logits from another episode, (iv) the episode-averaged pattern; KL to the net and to Bayes. Permute token order per episode first, or (iii)-(iv) do not ablate routing in the tiny-PFN layout. | - | (i) close to the net; (ii) drifts toward equal-precision weighting (reference a); (iii), (iv) worse than (i); in World FA (random order) (iv) fails outright | (ii) or (iv) as good as (i) after permutation: the input-dependent part of attention is not needed |
| M4 | **Internal message size across a cut.** Insert a k-dimensional linear bottleneck inside the net on the cross-source path (the residual stream of source s's tokens before the combining layer, or the values that layer reads from them), fine-tune only the bottleneck maps, and find the smallest k with KL(net \|\| bottlenecked net) <= 0.01. An external encoder from raw inputs would not do: for a net near Bayes it reaches k = 2 whatever the net does inside. | Bayes needs 2 numbers per source with a general merge, tens with an additive merge (N1e), 4-8 over two rounds (N1d) | **CONJECTURE**: nets near Bayes pass 2-4 dimensions per source | k >> 4: the net passes more than the task needs between sources (tangled beyond the task) |
| M5 | **Sink and posterior variance.** In the combining layer, regress the query token's attention to itself (or to the token whose value is most source-independent) on the net's predicted variance and on Bayes's. | a_0 = tau_0 x posterior variance (identity, 6.1) | **CONJECTURE**: R^2 >= 0.5 in some nets; if not, find the precision-sum channel with a linear probe | neither the sink weight nor any linear feature carries the precision sum |
| M6 | **Effective rounds.** Sensitivity of source s's representation at its query token after layer 1 to other sources' errors, relative to its own. | 0 for a one-round computation | descriptive at N = 3 (rounds help only from N of about 64, N1c); a test when N is scaled | - |
| M7 | **World FA.** M1 across trader cuts in the binding-given arms; M2 in the flat arms; M4 across a trader cut. | interaction non-zero exactly for cuts that split correlated traders; predictive message O(T^2) additive numbers per cut (5.1) | flat arms: C = 1, no routing learned (consistent with the measured own/other ratio 1.00); binding-given arms show Bayes's interaction pattern across trader cuts | binding-given arms reach 0.95 of the history value with an interaction pattern unlike Bayes's: they get there by a different combination |

Two cautions for M1. Any functional that vanishes at Bayes shrinks as a net approaches Bayes, so raw shares track
the size of the error; the deviation share corrects for that. And a two-round computation that is not exact couples
sources through the shared pilot, so it has non-zero interaction too: M1 measures distance from exact additivity, not
"SFU-ness" as such.

Costs: M1, M3, M5, M6 need only forward passes (minutes on CPU); M2 is a least-squares fit on stored attention maps;
M4 fine-tunes small bottleneck maps (minutes each).

### 6.4 First measurements: interaction profiles (52 nets) and routing classes (15 nets)

**M1 on the clean tiny-PFN cohort** (N5: `work/tiny-pfn/runs`, 8 configurations x 5 seeds, 4,000 episodes x 4 variants
per cut; log-odds of y* + {-0.6, -0.3, 0.3, 0.6} against y*; `n5_interaction.py`, `n5_interaction_v2.json`). The
share r is Var(Delta) over the summed variances of the two main-effect differences; for a pure-interaction term it
corresponds to an interaction fraction of the variance phi = r / (2 - r) (GPT-6's re-check), given in brackets.

| Nets | cross-source share | deviation share (error), cross-source | within-source share | KL to Bayes |
|---|---|---|---|---|
| exact Bayes | 0 (3.6e-31) | - | 0.039 [0.020] | 0 |
| Bayes projected to a Gaussian (reference) | 0.054 | - | 0.054 | 0.010 |
| 2 layers, mixture head, learned the task (12) | 0.012-0.041 | 0.34-0.47 [0.20-0.31] | 0.037-0.045 | 0.007-0.032 |
| 2 layers, Gaussian head (15) | 0.054-0.087 | 0.36-0.47 [0.22-0.31] | 0.050-0.062 | 0.016-0.050 |
| 2 layers, mixture head, stuck far from Bayes (3) | 0.109-0.114 | 0.33-0.35 [0.20-0.21] | 0.018 [0.009] | 0.144-0.145 |
| 1 layer, Gaussian head (10) | 0.186-0.206 | 0.37-0.48 [0.23-0.32] | 0.020-0.024 [0.010-0.012] | 0.174-0.197 |

**M1 on the 12 noisy phase-B nets** (N6, pre-registered in `PREREG-N6.md` before running; noise-marginalised
predictive over 8 draws; `n6_interaction_noisy.py`, `n6_interaction_noisy_v2.json`): group A cross-source share
0.012-0.014, group B 0.015-0.021 (all 6 A nets below the B median); cross-source below within-source in 12 of 12;
correlation with KL 0.95. The three pre-registered predictions hold as stated. But the deviation shares do not
differ: 0.31-0.33 (A) and 0.28-0.34 (B).

**MEASURED, and what it means.**
- The raw cross-source share tracks KL (0.96 over the 40 clean nets) because the error itself has a cross-source
  interaction share of 0.28-0.48 (an interaction fraction of about 16-32% of its variance) in every net, good or bad,
  clean or noisy: the raw share mostly measures the size of the error. This confirms the reviewers' smoothness
  objection; the A/B difference in raw share goes with B being further from Bayes.
- What is informative, and consistent with the proposed mechanism pending causal tests: (1) every net's departure
  from Bayes has a cross-source part of roughly 16-32% of its variance at this cut, so a per-source account of the
  error leaves that part out (cross-source coupling can enter through the normalisation of attention over sources even
  when the mistake starts within a source); (2) one-layer nets and stuck seeds have *less* within-source interaction
  than Bayes (0.018-0.024 vs 0.039), the signature expected when the per-source nonlinearity (reliability from the
  sum of squared errors) is not applied before sources are mixed, as in a single pool-then-transform round;
  (3) post hoc, not pre-registered: B's error is more interactive *within* a source than A's (deviation share
  0.22-0.24 vs 0.19-0.21, no overlap), in line with phase B's finding that B's encoding depends on how a source's
  errors are spread.

**M2 on 15 clean nets** (N7: `n7_routing_classes.py`, `.json`; w32_h2_L2 and w32_h2_L2_mix4 x 5 seeds, w32_h2_L1 x 5;
1,500 episodes; 16 tokens, fixed layout). The last column correlates the centred log attention mass the query puts on
each source's query reading with that source's centred log E[1/v | residuals].

| Heads | routing classes C among 15 rows (TV <= 0.1) | per-key weighting share of row-centred log-attention | label R^2 of the pair-specific part | query log mass vs source log precision, r |
|---|---|---|---|---|
| 2-layer nets, layer 1 (18 heads, without the stuck seed) | 2.7-8.1 | 0.02-0.31 | 0.75-0.89 (median 0.77) | - |
| stuck seed, layer 1 (2 heads) | 2.4-5.9 | 0.01-0.35 | 0.27-0.91 | - |
| 2-layer nets, layer 2 (18 heads, without the stuck seed) | 3.1-5.6 | 0.52-0.96 | 0.00-0.73 (median 0.11) | 0.41-0.95 (median 0.87) |
| stuck seed, layer 2 (2 heads) | 2.0-4.4 | 0.58-0.81 | 0.07-0.17 | -0.04 to 0.32 |
| 1-layer nets (10 heads) | 3.7-5.2 | 0.67-0.79 | 0.37-0.63 (median 0.50) | 0.13-0.21 |

**MEASURED** (attention maps only; the substitution tests M3 in the addendum revise this reading: both layers are
sensitive to labels and to episode-specific attention, and a per-key surrogate captures layer 2 but not block 1). In the 2-layer nets that learned the task, layer 1 is
mostly label routing (few classes; labels explain about 77% of the pair-specific logits) and layer 2 is mostly content
weighting (one scalar per key carries most of the
variation, and the query's log weights on the sources' readings track their log precision, median r = 0.87): routing,
then a precision-weighted merge, consistent with the decomposition and the sink identity. These one-layer nets show
only a weak correlation (r about 0.18), as expected if reliability must be gathered within a source before the query
can use it. Caveat: with the fixed token layout, label routing coincides with positional routing; permuting token
order is the next check (M3), and causal substitutions are needed before reading these as mechanisms.

### 6.5 Specification for the hierarchy-vs-dense experiment (for the experimental agent)

The comparison is only informative if it separates routing from combination, and rounds from hierarchy shape.

**Design: 2 x 2, plus rounds.**
- Routing: *oracle* (tokens arrive grouped by source, or a mask from the true grouping) vs *learned* (flat tokens with
  per-episode ID embeddings, as in the phase-B and World FA formats, with token order permuted per episode).
- Combination: *dense* (the phase-B transformer, matched parameters and FLOPs) vs *SFU tree*: per-source encoder (a
  set network over that source's tokens) to k numbers; merge = sum; readout = small MLP to the mixture-density head.
  For W2 use a two-level tree (tokens -> sources -> shared-error groups -> root), or the source and group boundaries
  collapse.
- Rounds, tree only: one pass, or two passes (pass 1 sends a few numbers per source; a learned readout broadcasts a
  pilot; pass 2's encoder also sees the pilot and sends k numbers). Match the total message size and parameter count
  between one-pass and two-pass trees, or rounds are confounded with capacity.
- Learned routing for the tree: cluster tokens by pairwise dot products of their ID embeddings (the same
  rotation-invariant information dense attention uses). Not hashing (tokens of a source share an identical ID vector,
  so hashing is oracle routing in disguise) and not learned prototypes (they break the rotation symmetry of random
  per-episode IDs).

**Worlds.**
- W1: tiny-PFN prior with N in {3, 8, 24, 64, 256} sources (same generator; exact Bayes from the per-source
  log-likelihood sum on a grid, `work/scale-free-theory/n1c_rounds.py: ell_grid`).
- W2: W1 plus G shared-error groups (one shared error per group and question), membership visible as a label or
  hidden; G in {1, 2, 4, 8}. With hidden membership, break the label symmetry (for example order groups by their
  shared-error variance) or the posterior has G! modes and local two-round messages fail.
- W3: World FA, binding-given vs flat.

**Read-outs.** KL to Bayes against k; smallest k reaching KL <= 0.01 and 0.001 (95th percentile over episodes) per N;
M1 (with deviation shares), M2 and an internal M4 on the dense nets; FLOPs.

**Predictions (to pre-register).**
- P1 (W1, oracle routing): a two-pass tree at small k (about 4-8) reaches the KL that a one-pass tree reaches only with
  larger k, once N >= 64; at N = 3 the two are similar. Calibration (N1d, N1e): a 2-number one-round code stays at KL
  about 0.2 at every N, while 4 numbers over two rounds reach 2e-4 at N = 64 and 5e-7 at N = 1024; an 8-number
  one-round Chebyshev code is far worse than an 8-number two-round code (N1e). Learned nonlinear encoders may beat our
  linear codes, so the numbers are calibration, not bounds. Refuted if the two-pass tree is no better than the
  one-pass tree at matched size for N >= 256.
- P2 (W2, visible membership): a group-aligned tree needs the same k as W1; a tree whose groups split the shared-error
  groups needs k growing with the number of straddling groups.
- P3 (W2, hidden membership): one-pass k grows quickly with the number of open latents (N2: empirical linear one-round
  codes need about 19, 330 and 1,650 numbers for 1, 2 and 3 open latents at RMS error 1e-3); two passes with
  re-routing after the pilot stay far smaller.
- P4 (routing): with learned routing on flat tokens, both architectures fail to learn binding without help (World FA:
  0 of 15 flat runs), and both succeed with oracle routing. Refuted for "density pays for routing" if dense with
  learned routing matches dense with oracle routing, or if the tree with oracle routing falls far short of dense with
  oracle routing at matched compute.
- P5 (M3 payoff): for a one-pass SFU tree, given y* and the latents, the summed message is a sum of independent
  per-source terms; for the additive log-odds message (before any nonlinear readout), Chernoff bounds certify tails and
  a grid convolution is exact when the message has 1-3 dimensions. For a two-pass tree, condition on a discretised
  pilot first (the pilot couples the sources). Compare against sampling at equal compute (the M3 bar).

## 7. What would count as a major advance, and the shortest path

What is known and what is ours. Known (cited): Nerode classes and one-way communication; junction-tree separators,
induced width and context-specific independence (the open-latent profile is their reading for this problem);
Koopman-Pitman-Darmois; one-step estimators and their distributed versions; transformers as multi-step in-context
estimators; the disjointness, gap-Hamming and pointer-chasing bounds; sign-rank; the Alman-Song dichotomy; rank-width;
conditional-Gaussian hardness; fast multipole methods; attention sinks. Ours, as far as the checks in `CITATIONS.md`
go: the comparison of the six definitions with separating examples; the definition by open latents, label routing and
rounds as the reading of Kelvin's three phrases; attention's routing cost as sign-rank and sharp selection as the
Alman-Song regime; routing classes as the measure of what dense attention pays for; the sink as the Bayesian prior in
the Gaussian merge identity; the two-bond-dimension view of exact estimation and the misalignment numbers; the
capacity bound for bounded noisy messages; the measurements in our worlds (N1-N7), including the deviation-normalised
interaction profile.

Three candidates for the advance itself, each tied to a program target.

1. **M2: in-context Bayes in transformers as scale-free computation.** Show, across the phase-B nets and scaled
   worlds, that trained transformers route by labels, merge natural-parameter-like messages through attention (the
   prior as sink), use layers as rounds, and carry messages of the size the open-latent profile requires; and that
   each departure from Bayes is the error of the architecture's merge family at some scale. The measurements are
   consistent with the first half for the combining layer (addendum: a label-plus-per-key surrogate meets the
   criterion; the query's self-attention correlates with posterior variance), while block 1's attention is not
   captured by that surrogate. The big-miss
   departure fits the second half: at the token-to-source scale, Bayes adds the unbounded statistic e^2, while the
   first layer averages normalised (bounded) per-token features, so the Bayes merge lies outside the family (the E20
   toy's "capped per-error evidence"); depth adds a per-token transformation before pooling, which enlarges the
   family (E17, E19). **CONJECTURE** (general form): the departure equals the projection of the Bayes merge onto the
   merge family, scale by scale, and is predictable from the architecture before training. The interaction profile
   adds a constraint any such account must meet: every net's error has a cross-source interaction part (about 16-32%
   of its variance).
2. **M3: an untangled architecture at transformer loss with exact tails.** An SFU tree (label routing, additive
   natural-parameter messages, a broadcast round) that matches the dense transformer on W1-W2 and whose output law is
   computed exactly given the latents, beating sampling on certified tail bounds. Our literature check found no
   strictly low-treewidth model at transformer loss on a natural task; this would be the synthetic first step, and it
   would have to survive hidden membership (W2) to matter.
3. **A necessity map.** Which in-context tasks need dense routing (sharp similarity search; correlation discovery
   near the detection threshold, T7), which need only label routing (binding, T5), and which need rounds (multi-hop;
   hidden membership, T4), with matching lower bounds. Most pieces exist (5.3-5.5); hidden membership has only upper
   bounds and numerics here.

A fundamental blocker would look like this: if two-pass SFU trees cannot match the dense transformer on hidden
membership at any moderate k, dense attention is earning its cost through structure discovery, which T7 says costs
about min(T, 1/rho^2) bits per item plus comparisons. That would itself be a sharp result to report.

**Shortest path.**
1. Done today: definitions, theorems, numerics N1-N3, and first measurements N5-N7 on 52 trained nets.
2. Next on the existing nets (CPU, hours): M3 routing substitutions with permuted token order; the internal
   bottleneck M4; M5 (sink and posterior variance); an intervention that removes the cross-source part of the error
   and checks whether KL falls.
3. W1 with oracle routing: tree one-pass vs two-pass vs dense, N up to 256 (CPU or one L4, a day).
4. Exact output law of the tree (Chernoff and low-dimensional convolution on the log-odds message) against sampling.
5. W2 hidden membership and W3 routing; then a natural multi-source dataset.

## 8. Back to the original question: can dense nets be untangled, and is their tangledness needed?

- Needed to represent? For the aggregation tasks we study, no: SFU solutions exist (sections 5.1-5.3), and for
  one-level routing a transformer should be able to implement them in O(rounds) layers (A2, a conjecture with a
  construction sketch). Tangledness is necessary for
  similarity-based routing (T6a, T7), hidden structure near the detection threshold (T4a, T7), multi-hop lookups
  without depth (T8), and genuinely global couplings (T10).
- Used by trained nets? Partly measurable without opening the net (M1) and with attention maps (M2). In the tiny-PFN
  nets, both layers are sensitive to label alignment and to episode-specific attention (substitutions, addendum); a
  label-plus-per-key surrogate captures the combining layer but not block 1; what block 2 reads per context token
  compresses to 3-4 dimensions; every net's error is partly cross-source; nets that cannot apply the per-source
  nonlinearity before mixing carry the expected signature. Whether the nets' internal messages are as small as the task allows (M4) is not yet measured.
- Learnable? Separate question. World FA (binding) and World 3 (parity) are cheap to represent and were not learned.

## 9. Claim register

Statuses after the GPT-6 and Gemini reviews (findings and our responses: section 10).

| ID | Claim | Status | Section |
|---|---|---|---|
| T1 | Minimal exact tree message at v = number of Nerode classes; ceil(log2) of it = one-way CC; internal epsilon-nets add at most epsilon per node | PROVEN | 2 |
| L1 | Unrestricted routing rules absorb the whole computation | PROVEN | 2 |
| T2 | Messages are likelihoods of the open latents; merges add log-likelihoods and sum out closing latents | PROVEN (standard) | 5.1 |
| T2m | Minimality for the joint posterior of the open latents | PROVEN UNDER richness; smaller for a single target | 5.1 |
| T2a | Tiny-PFN: only y* is open between sources | PROVEN | 5.1 |
| T2b | Inverse-Wishart errors: predictive message across a trader cut = O(T^2) additive numbers (Sylvester); full posterior of Sigma needs T \|S\| | PROVEN UNDER the matrix-normal / IW model, prior scale block-diagonal or diagonal plus low rank | 5.1 |
| T3 | Fixed-dimension sufficient statistics for all N only in exponential families | PROVEN UNDER regularity (Koopman-Pitman-Darmois) | 5.2 |
| T3a | Analytic families: O(log(MN/delta)^d) additive numbers per source for KL <= delta | PROVEN UNDER compact prior and a uniform complex-analytic bound | 5.2 |
| T4 | Two rounds (pilot, then local degree-p statistics) give a window-conditional KL of about N^{1-p} up to logs | PROVEN UNDER smoothness, posterior concentration, a pilot inside the window w.h.p., unimodality (one-step estimators; cited) | 5.3 |
| T4a | One-round size grows exponentially in the number of open latents for location-type families | HEURISTIC; MEASURED (N2) | 5.3 |
| T5 | Keyed aggregation: Omega(N) bits at a node of every fixed hierarchy; O(1) + log N label bits routed | PROVEN | 5.4 |
| T5b | Bayes analogue (World FA) and "learning, not representation" | HEURISTIC | 5.4 |
| T6 | Attention routing cost = sign-rank of the routing relation; equality 3, order 2, Hadamard >= sqrt(N) | PROVEN | 5.4 |
| T6a | Sharp selection needs logit range ~ log N, Alman-Song's worst-case hard regime; label routing is O(N log N) | PROVEN (arithmetic + cited theorem) | 5.4 |
| T6b | Per query, softmax attention is an associative merge with d + 2 numbers | PROVEN | 5.4 |
| T7 | Correlation-based grouping needs Omega(min(T, 1/rho^2)) bits per pair; keys of O(min(T, log(N/delta)/rho^2)) bits suffice | PROVEN (promise problem, sign strings); HEURISTIC (iid histories) | 5.4 |
| T8 | 3+ hop lookups need Omega(n) bits at any cut separating the tables in one pass; k rounds suffice | PROVEN (for such cuts; cited constants unverified in primary text) | 5.5 |
| T9 | Parity: untangled under every definition, SQ-hard; gradient methods only in a coarse-precision regime | PROVEN (cited); MEASURED (World 3) | 5.5 |
| T10 | GF(2) pairwise functions: worst message of the best hierarchy = rank-width (+1 bit) | PROVEN | 5.6 |
| T10a | ReLU gates couple through Gaussian pre-activations, so the non-Gaussian part is the gate-pattern entropy | HEURISTIC | 5.6 |
| T11a | Exact output law in O(R K (R + K)) per node along a shared hierarchy (sufficient, not necessary) | PROVEN | 5.7 |
| T11b | Misaligned random trees: union treewidth grows with N | MEASURED (N3a); linear growth CONJECTURE | 5.7 |
| T11c | Rank-2 functions can have #P-complete tails (integer weights) | PROVEN | 5.7 |
| T11d | Bounded noisy messages: m >= log2((1 + N I)/c) / log2(1 + R^2/sigma^2) for mean-squared error c v_N | PROVEN UNDER the Gaussian setting stated | 5.7 |
| T12 | Latent-tree data: block MI O(log L); correlations at most power-law; KS threshold (strict inequalities) | PROVEN | 5.8 |
| A1 | Softmax attention with a sink = merge of Gaussian messages; sink weight = tau_0 x posterior variance | PROVEN (identity) | 6.1 |
| A2 | Transformers implement one-level R-round SFU computations in O(R) layers | CONJECTURE (construction sketch; approximation and precision costs not accounted) | 6.2 |
| A3 | Flat-token binding stalls at a routing-content saddle | HEURISTIC | 6.2 |
| C1 | Every trained net's error has a cross-source interaction part (16-32% of its variance); raw interaction tracks error size; one-layer and stuck nets lack within-source interaction | MEASURED (52 nets, N5-N6) | 6.4 |
| C1b | 2-layer nets: both layers sensitive to labels and to episode-specific attention; a label + per-key surrogate meets the criterion for layer 2, not layer 1 | MEASURED (N7 correlational; M3 substitutions, addendum) | 6.4, addendum |
| C2 | Each departure from Bayes is the error of the merge family at some scale | CONJECTURE | 7 |
| A4 | The query token's self-attention in the combining layer correlates with posterior variance (log-log slope 0.26-0.62) | MEASURED (7 of 9 clean, 8 of 12 noisy nets; pre-registered); the prior-as-sink mechanism: CONJECTURE | addendum |
| A5 | Both layers are sensitive to label alignment and to episode-specific attention; a "labels plus one weight per key" surrogate meets the criterion for layer 2, not layer 1 | MEASURED (M3; pre-registered; P-M3b and P-M3c failed as stated) | addendum |
| A6 | A shared rank of 3-4 per context token suffices for what block 2 reads (KL <= 0.01 to the net, query state untouched) | MEASURED (7 of 9 nets; pre-registered; an achieved compression, not an intrinsic dimension) | addendum |

## 10. Reviews and what changed

GPT-6 (gpt-6-astra, high effort; 28 findings) and Gemini (gemini-4-argon-high; about 20 findings) reviewed the first
draft adversarially; both reports are in `work/scale-free-theory/review-*-2026-10-29.md`, and the first draft is kept
as `draft-v1-before-review.md`. What changed:
- Inverse-Wishart errors (T2b): the query predictive needs O(T^2) additive numbers per trader cut by Sylvester's
  identity, not the whole history; the draft's volume-law claim holds only for the full posterior of Sigma (Gemini).
- Rounds (T4): the KL rate is the square of the log-posterior error, about N^{1-p}; concentration, truncation, pilot
  and unimodality hypotheses added; the claim that Gaussian location mixtures have root-N moment pilots was dropped
  (both reviewers).
- Interaction profile: the deviation share was added and shows that the raw share mostly tracks the size of the
  error, as Gemini predicted; the Gaussian projection is a reference, not a floor (both). The 0.96 and 0.98
  correlations of the draft are kept as numbers but no longer read as evidence of tangling.
- M4 redesigned as an internal bottleneck: an external encoder would reach k = 2 for any net near Bayes (Gemini).
- Sharp selection: logit range, not gap; Alman and Song's lower bound is worst case (both).
- Pointer chasing: restricted to cuts that separate the tables (GPT-6). Parity: SQ hardness covers gradient training
  only at coarse precision (GPT-6).
- Statuses lowered: the Bayes version of T5 and the World FA "learning failure" reading (HEURISTIC), A2 (PROVEN only
  for one-level routing; then lowered to CONJECTURE after the re-check), T11a (a sufficient algorithm, not a necessity), T11d (exact Gaussian model; the forward-KL
  factor is e^{2 delta}).
- Credit added: junction-tree separators, induced width, context-specific independence; transformers as in-context
  estimators; attention sinks; tree automata; routing-content learning plateaus.
- Smaller fixes: D3 ranks (item cut vs parameter-observation), internal epsilon-nets, knapsack tails as an indicator
  of rank up to 2^{n/2}, the latent-tree separator (2 log_b L), strict inequalities in the reconstruction threshold,
  assumptions for cumulant bounds and limit laws, and the 2 x 2 specification (router by pairwise ID products, a
  two-level tree for W2, matched capacity across rounds, conditioning on the pilot for exact tails).
- Not adopted: upgrading T4a and T11b to PROVEN via the reviewers' suggested arguments, which were not checked here.
- A GPT-6 re-check of the revision (`review2-gpt6-2026-10-29.md`, 7 findings) led to: the low-rank prior count
  O(T^2 + kT) for the inverse-Wishart case; labelling the two-round KL as window-conditional (hard truncation makes the
  full forward KL infinite); corrected M2 table values; interaction shares converted to variance fractions
  (phi = r/(2 - r), about 16-32%), which removed an unsupported "at most two thirds" ceiling; keeping the target open
  to the root and defining closure by factor scopes in T2; and A2 lowered to CONJECTURE. It confirmed the corrected
  capacity bound and that the N5/N6 numbers match the outputs.

## Files

All under `/home/agent/workspaces/mats/work/scale-free-theory/` unless stated.
- `QUESTIONS.md` (definitions and checks); `CITATIONS.md` (references checked, arXiv abstracts in `cites/`).
- N1e one round vs two rounds, globally normalised (supersedes N1c): `n1e_global_kl.py`, `.json`, `.log`;
  N1c (local normalisation): `n1c_rounds.py`, `.json`, `.log`.
- N1d small codes: `n1d_small_messages.py`, `.json`, `.log`.
- N2 open latents: `n2_open_latents.py`, `.json`, `.log`.
- N3 tanglegrams and GF(2) classes: `n3_tanglegram_gf2.py`, `.json`, `.log`.
- N5 interaction profile, clean nets: `n5_interaction.py`, `n5_interaction_v2.json` (with deviation shares),
  `n5_interaction_v1.json`.
- N6 interaction profile, noisy phase-B nets: `PREREG-N6.md`, `prereg_hashes.txt`, `n6_interaction_noisy.py`,
  `n6_interaction_noisy_v2.json`, `n6_interaction_noisy_v1.json`, logs.
- N7 routing classes: `n7_routing_classes.py`, `.json`, `.log`.
- Reviews: `review-gpt6-prompt.md`, `review-gpt6-2026-10-29.md`, `review-gemini-prompt-head.md`,
  `review-gemini-2026-10-29.md`, re-check `review2-gpt6-prompt.md`, `review2-gpt6-2026-10-29.md`; first draft:
  `draft-v1-before-review.md`.
- Superseded numerics: `n1_pfn_rank.py` (uniform-weight SVD basis), `n1b_pfn_weighted.py` (per-source weighted fits,
  unstable for small N).
- Addendum: `PREREG-SF.md` and `prereg_hashes.txt`; nets copied to `nets/` (`nets/MANIFEST.sha256`); `sf_net.py`
  (re-implemented forward); `m3_subst.py`, `m3_subst.json`; `m4_bottleneck.py` and its output-file copies,
  `m4_bottleneck.json`, `m4_bottleneck_mix4.json`, `m4_bottleneck_mix4_s2.json`, `m4_bottleneck_mix4_s4.json`;
  `m5_sink.py`, `m5_sink.json`; scoring `sf_analyze.py`, `sf_scores.json`; check `review3-gpt6-prompt.md`,
  `review3-gpt6-2026-10-29.md`.

## Addendum (29 October, 20:28-21:45Z): M3, M4 and M5 on the existing nets

Run after the coordinator approved them. Pre-registered in `work/scale-free-theory/PREREG-SF.md` (hashed at 20:28Z,
before any outcome; plumbing checks disclosed there). Nets copied to `work/scale-free-theory/nets/` with a hash
manifest. Phase B's registered interventions were not repeated: E8.2's layer-2 query-row patches on the noisy nets,
E22's uniform within-source attention, and E24/E25's matched-pair transplants. A GPT-6 check of code, numbers and
wording (`review3-gpt6-2026-10-29.md`) found no implementation bug, confirmed every count and range, and asked for the
narrower interpretations used below. Learned set: the 9 clean 2-layer nets that learned the task (w32_h2_L2 s0-s4,
w32_h2_L2_mix4 s0, s1, s2, s4); the stuck seed mix4_s3 is reported separately. Numbers are **MEASURED**; readings of
mechanism are marked separately.

**Scorecard.**

| Prediction (PREREG-SF) | Result | Verdict |
|---|---|---|
| P-M3a: K_fit < K_aligned < K_misaligned in both layers, >= 7 of 9 | 7 of 9 | holds |
| P-M3b: routing cost (misaligned minus aligned) larger in layer 1 than in layer 2, >= 7 of 9; weighting cost (aligned minus fit) larger in layer 2, >= 7 of 9 | 4 of 9; 6 of 9 | fails |
| P-M3c: label-plus-own-weight fit loses <= 1/4 of what an aligned transplant loses, both layers, >= 7 of 9 | 4 of 9 (layer 2: 9 of 9; layer 1: 4 of 9) | fails |
| P-M4a: smallest k with KL(net \|\| bottleneck) <= 0.01 is <= 4, >= 7 of 9 | 7 of 9 (k = 3 in three nets, 4 in four, 6 in two) | holds |
| P-M4b: removing context-token variation (k = 0) costs >= 0.05, all 9 | 9 of 9 (0.32-0.82 nats) | holds |
| P-M5a: best head's r(log self-attention, log Bayes variance) >= 0.5, >= 5 of 9 clean nets | 7 of 9 | holds |
| P-M5b: the same, >= 6 of 12 noisy phase-B nets | 8 of 12 | holds |

**M3: sensitivity of each layer to its attention.** KL(net || net with one layer's attention substituted), nats,
ranges over the 9 learned nets; the nets' own KL to Bayes is 0.006-0.022.

| Layer | fit: label table + the episode's own per-key effects | another episode's attention, labels aligned | the same, labels scrambled |
|---|---|---|---|
| 1 | 0.04-0.34 | 0.20-1.05 | 0.61-4.37 |
| 2 | 0.02-0.04 | 0.21-0.41 | 0.44-4.45 |

- Both layers are very sensitive to label alignment: scrambling which token is which costs 0.4-4.4 nats, 20 to several
  hundred times the nets' distance from Bayes. The registered routing contrast (scrambled minus aligned) is positive
  in both layers for 8 of 9 nets (negative in layer 1 of s0).
- Both layers are also sensitive to episode-specific attention: keeping the labels but taking another episode's
  attention costs 0.2-1.05 nats. Part of this may be dependence on the random per-episode IDs rather than on the
  content, so this measures episode-specific attention, not content weighting alone. Even so, these nets' attention
  does more than route by labels.
- In layer 2, the fitted surrogate (label table plus per-key effects) meets the registered relative criterion in 9 of
  9 nets (0.02-0.04 nats, a tenth to a fifth of an aligned transplant). In absolute terms it moves the nets away from
  Bayes (KL(Bayes || fit) 0.026-0.063 against 0.006-0.022). That the per-key effects are precisions, or that layer 2 is
  a Gaussian per-source merge, is not shown by this test (**CONJECTURE**, supported by N7's correlation of the query's
  weights with log precision).
- In layer 1 the surrogate fails the criterion in 5 of 9 nets (0.04-0.34 nats): this particular per-key surrogate does
  not capture block 1's attention, whose episode-specific part seems pair-specific. That fits phase B's E22-E25, which
  locate the big-miss departure in block 1's within-source pooling, but does not identify block 1's computation.
- The layer-role prediction from N7 failed (P-M3b): N7's variance reading ("layer 1 routes, layer 2 weights") is not
  supported causally. In layer 2 the per-key share included label-determined token-type effects (query readings vs
  labelled tokens).
- The stuck seed is sensitive to labels but barely to episode-specific attention (aligned transplants 0.05 and 0.06
  nats in layers 1 and 2; scrambled 1.4 and 0.37), consistent with a net that never learned reliability.

**M4: compression of what block 2 reads from the context tokens.** After block 1, the residual streams of the 15
context tokens (32 dimensions each) were passed through a shared rank-k linear map trained by distillation, with the
query token's state left untouched; that state already attended to the context in block 1, so k = 0 removes the
context tokens' variation, not all context information. Removing it costs 0.32-0.82 nats. A shared rank of 3-4 per
context token sufficed (KL to the net <= 0.01) in 7 of 9 nets (6 in the other two), given the untouched query state; at
k = 4 the bottlenecked nets are about as close to Bayes as the nets (0.009-0.034 against 0.006-0.022 nats). This is an
achieved compression, an upper bound for linear codes shared by both token types, not an intrinsic dimension, not a
per-source message size (a source has five tokens; its Bayes sufficient statistic is two numbers), and not evidence
about the whole cut, since the query's own block-1 channel was not bottlenecked. Distillation may stop at a local
optimum (in one net the k = 2 fit stayed at 0.20 while k = 3 reached 0.009). Outputs: `m4_bottleneck.json` plus the
shards `m4_bottleneck_mix4.json`, `m4_bottleneck_mix4_s2.json`, `m4_bottleneck_mix4_s4.json`.

**M5: self-attention and posterior variance.** In the combining layer, the log of the query token's attention to itself
correlates with the log of Bayes's posterior variance: best-head r 0.56-0.85 in 7 of 9 learned clean nets (s0: 0.27,
s3: 0.43) and 0.56-0.81 in 8 of 12 noisy phase-B nets. Pre-registered exploratory measures: the same correlation with
the nets' own predictive variance is about as high; slopes in log-log are 0.26-0.62, not the 1 an exact sink would give;
one-layer nets reach r 0.16-0.23; the stuck seed -0.02. Post hoc, not pre-registered: all six group-A noisy nets have
r 0.70-0.81, group B 0.34-0.72 (two of six above 0.5). That the query token acts as the prior sink of the
Gaussian-merge identity is a **CONJECTURE** these correlations are consistent with, not something they establish: they
do not show prior-valued values, precision logits or causal use, and Bayes's predictive here is a mixture, for which
the single-Gaussian identity is only approximate.

**What the addendum changes in the report.**
- The routing-plus-weighting decomposition (6.2) gives a surrogate that meets the registered criterion for the
  combining layer but not for block 1; whether layer 2 is a per-source precision-weighted merge with a sink remains a
  conjecture with correlational support (N7, M5).
- "Dense attention mostly pays for routing" is refined: these nets' attention is highly sensitive both to label
  alignment and to episode-specific attention, in both layers.
- What block 2 reads from each context token can be compressed linearly to 3-4 dimensions with little loss (given the
  untouched query state); M4 in 6.3 has a first, partial answer for this cut.
