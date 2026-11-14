You are an independent, adversarial reviewer (theoretical computer science, information theory, statistics, tensor
networks). Below is a theory note written for a research lead. Every statement carries a status tag: PROVEN,
PROVEN UNDER X, HEURISTIC, CONJECTURE, MEASURED. Find errors; do not praise; do not rewrite the note.

Answer in this order, concisely, as numbered findings with severity (high / medium / low), the exact text at issue,
and the fix:
1. Mathematical errors or missing hypotheses in any PROVEN / PROVEN UNDER claim (check the proofs given: T1 Nerode
   argument and the (2N - 1) epsilon tolerance bound; T2 open latents; T3a KL <= 2 N epsilon; T4 two-round error rate;
   T5 disjointness reduction at a balanced node; T6 sign-rank, the equality construction 1/2 - (i - j)^2, the
   Alman-Song logit-gap argument; T6b online softmax; T7 gap-Hamming padding; T10 GF(2) cut classes and rank-width;
   T11a cost; T11d capacity bound; T12 latent-tree mutual-information bound; A1 attention-sink identity).
2. Statuses that are too strong (e.g. something called PROVEN that is only a heuristic) or too weak.
3. Known results presented as new, and important prior work the note misses (give exact references only if you are
   sure they exist).
4. The central proposal (section 4: "open-latent profile + routing + rounds" as the definition of scale-free,
   untangled information combination): what is the strongest objection or counterexample? Is anything in the
   separations table (section 3) wrong?
5. The transformer measurements (section 6.3) and the experiment spec (6.4): can each one actually discriminate the
   hypothesis it targets? Name any confound.

THE NOTE:

# Scale-free, untangled information combination: definitions, theorems, tests (29 October 2026)

For Kelvin (theory lane; brief from main). DRAFT, being reviewed.

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
   "before/after" or "contiguous chunk"), *content* (the message at each scale, allowed to grow), and the
   *combination rule*. For Bayesian aggregation the rule is the same at every scale: add log-likelihoods, and sum out
   the hidden variables that are complete at that node. The message at a node is the likelihood of the hidden
   variables still shared with the outside ("open"). What grows with scale is the number and richness of open hidden
   variables, not the number of items: larger structures may need more dimensions, while how information combines
   stays fixed.
3. **Rounds matter more than the shape of the hierarchy.** If sources write once without seeing anything, each
   message must describe a whole function of the open hidden variables. If the root first broadcasts a consensus and
   sources then write relative to it (your "messages written after reading the consensus"), a few local numbers
   suffice and the error falls as N grows. On the tiny-PFN world: one round needs 48-64 numbers per source for
   KL <= 10^-3 at any N from 3 to 4096; two rounds need 8 numbers from N = 64 on, with KL 2e-3 at N = 64 and 4e-10 at
   N = 4096. With 3 open hidden variables, one round needs thousands of numbers. The mechanism is known in
   distributed statistics (one-step estimators); what is ours is the link to architectures and these measurements.
4. **What provably needs tangled or dense structure.** Binding: tangled for every fixed hierarchy, untangled with
   labels; one attention head can do it, so World FA's failure was a learning failure. Multi-hop lookups need rounds
   (depth). Parity is untangled under every definition yet not learnable by gradient methods (World 3's XOR chains).
   Genuinely global couplings (expander-like) are tangled under every hierarchy, though algebra (Gaussian, GF(2)) can
   keep expectations easy. Grouping items by correlation of their histories needs keys of about min(T, 1/rho^2) bits
   per item. Dense attention's quadratic cost is necessary (under SETH) only for sharp selection among similar keys,
   not for label routing, which sorting or hashing does in N log N.
5. **Treewidth is the wrong measure.** A complete interaction graph can be maximally untangled (2 bits per cut), and
   Bayes with one shared answer makes every pair of items dependent (treewidth N - 1) though a single hidden variable
   carries everything. The right measure is the rank of what crosses a cut, after the cheap Gaussian algebra is taken
   out; in a ReLU net the "non-Gaussian part" is coupled through the Gaussian pre-activations, so it must be counted
   as the entropy of the gate pattern.
6. **Estimation (M1/M3).** Exact estimation needs the input law untangled along the *same* hierarchy as the
   computation; the cost is the product of the two. Expectations need only low rank; tails need low dimension (a
   rank-2 linear function has #P-hard tails). Under activation noise with normalised activations, message dimension
   must grow like log N (proved): more dimensions at larger scales is a necessity, not just a permission.
7. **Transformers.** Softmax attention with a sink is exactly the merge of Gaussian messages (logit = log precision;
   the prior is the sink; its weight is prior precision times posterior variance). For one query, attention is an
   associative merge along any hierarchy, so a dense layer is N such merges; "dense attention pays for routing" becomes
   "few, label-determined routing classes per layer, with simple content". Section 6 gives seven CPU measurements on
   our nets and a 2 x 2 design (oracle vs learned routing x dense vs tree, plus rounds) for the experimental agent.
   First measurement, on 40 trained tiny-PFN nets: how non-additively a net combines its sources (exact Bayes: not at
   all) tracks its distance from Bayes (correlation 0.96; 0.98 among the 12 best). One-layer nets mix sources before
   judging each source's reliability, as one pool-then-transform round must, and sit 0.17-0.20 nats from Bayes.
8. **Major advance and shortest path.** Either show that trained transformers do in-context Bayes as this kind of
   computation, with each departure from Bayes predicted as the error of the architecture's merge rule at some scale
   (the big-miss departure already fits), or build an untangled architecture at transformer loss with exact tails.
   Next: the same interaction profile on the noisy phase-B nets (A/B groups) with an intervention, the routing-class
   measurement, then the tree experiment. No decision needed.

## 1. Setting

Items x_1..x_N (tokens, sources, reports). The target is a function f(x) or a posterior p(y | x). A **hierarchy** is a
rooted binary tree whose leaves are the items; node v covers the item set S_v, and its **message** m_v depends only
on x_{S_v}, computed from its children's messages by a **merge** g_v. A **round** is one upward pass; between rounds
the root may **broadcast** something down. The input law P is the prior over episodes; its hidden variables (the
answer, each source's reliability, shared errors, group memberships) are the **latents** u.

## 2. Six candidate definitions

**D1. Bits across the cuts of a fixed hierarchy.** Two contents x_S, x'_S of a cut S are equivalent if
f(x_S, z) = f(x'_S, z) for every completion z. K_f(S) is the number of classes (Myhill-Nerode for trees).
Untangled: log2 K_f(S) grows slower than |S| (sub-volume), strongly untangled if polylog.

- T1, minimal tree computation (**PROVEN**). Every exact tree computation along T has at least K_f(S_v) distinct
  messages at v, and sending the class achieves it. Proof: if two inequivalent contents shared a message, the output
  (a function of m_v and of z) would agree on the completion z that separates them. Conversely the class is a
  function of the children's classes: if x_a ~ x'_a and x_b ~ x'_b, then f(x_a, x_b, z) = f(x'_a, x_b, z) =
  f(x'_a, x'_b, z) for every z. log2 K_f(S) is the deterministic one-way communication complexity from S to the rest
  (number of distinct rows of the communication matrix).
- With tolerance (**PROVEN**): quantising each node's class to an epsilon-net in sup norm gives
  e_v <= e_a + e_b + epsilon (same substitution argument), so the root error is at most (2N - 1) epsilon; covering
  numbers at epsilon = delta / 2N replace K.

**D2. Bits across the cuts of the input's own hierarchy (routing).** Each item gets a label lambda(x_i) of b bits; a
fixed **rule** builds the hierarchy from the labels; content is counted as Nerode classes given that both sides know
the labels. The rule must be restricted (group-by-label, sort-by-label, contiguous chunks), and the labels charged:
otherwise labels can carry the whole item and the "router" does the whole computation (**PROVEN**, trivially).

**D3. Bond dimension and min-cut.** The rank of the matricisation M_S[x_S, x_{S^c}] = F(x) across each cut, for F a
function, a log-likelihood, or a probability table. For a probability table, the nonnegative rank is the smallest
number of latent states making the two sides conditionally independent (**PROVEN**, standard). For deep
arithmetic-circuit networks the min-cut of the network's tensor-network graph bounds the rank across a cut
(Levine et al. 2018, arXiv:1704.01552). The value depends on the representation: an exponential-family
log-likelihood has rank d + 1 in log space and infinite rank as a likelihood (**PROVEN**).

**D4. Treewidth of the dependency structure, and its non-Gaussian part.** Treewidth of (a) the computation's DAG,
(b) the graphical model of the items or activations, (c) the discrete (non-Gaussian) part of a conditional-Gaussian
model. D4(a) implies D1: a Boolean circuit of treewidth w compiles, in time linear in its size and singly exponential
in w, into a structured d-DNNF, i.e. a computation along a tree over the inputs with bounded classes per node, which
supports exact model counting and expectations under product laws (Amarilli, Capelli, Monet, Senellart,
arXiv:1811.02944). The converse fails (section 3). Lower bounds for such circuits go through communication
complexity across the partitions the tree induces (Bova, Capelli, Mengel, Slivovsky, IJCAI 2016): D1 is the
knowledge-compilation notion of structured decomposability. **PROVEN** (cited).

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
| D1 vs D3: f = sum_i a_i x_i, generic real a, x in {0,1}^n | rank 2 across every cut; 2^{|S|} classes | PROVEN |
| D3 vs estimation: same f | E f is trivial; P(f > t) under the uniform law is counting knapsack solutions, #P-complete (Dyer 2003) | PROVEN |
| D4 vs D1/D3: f = x^T U x mod 2 on the complete graph | treewidth n - 1; f depends on |x| mod 4, so 2 bits suffice at every cut (rank-width 1) | PROVEN |
| D4 vs everything: Bayes with one shared target theta | items are pairwise dependent (a clique, treewidth N - 1); one latent is open at every cut (section 5.1) | PROVEN |
| D1/D3 vs tractability: x^T U x mod 2 on random graphs | linear rank-width (Lee, Lee, Oum, arXiv:1001.0461), yet sum_x (-1)^{x^T U x} takes polynomial time (Dickson) | PROVEN |
| untangled vs learnable: parity | 1 bit across every cut of every hierarchy; statistical-query hard; World 3's XOR chains were never learned | PROVEN; MEASURED (World 3) |
| D5 vs D1: inner product mod 2 across a cut, independent uniform inputs | I(X_S; X_{S^c}) = 0, yet Omega(N) bits across the cut even on average (Chor and Goldreich 1988; for disjointness the hard distribution is not a product one) | PROVEN |

So: D1 counts precision and misses routing; D3 counts dimensions but misses tails; D4 measures a graph, not a
function, and calls the simplest aggregation maximally tangled; D5 describes the data, not the computation.

## 4. Verdict: the open-latent profile, with routing and rounds

Kelvin's intuition has three parts: complexity may grow with scale; it is fine for larger structures to have more
dimensions; what stays simple is how information combines. The definition that matches all three, and that the
theorems below support, is:

**Definition (scale-free untangled, SFU).** A task family admits an SFU solution with routing cost b, open-latent
profile o(s) and R rounds if
1. (routing) each item carries a label of at most b bits, and a restricted rule (group-by, sort, contiguous chunks)
   builds the hierarchy from the labels;
2. (content) the message at a node covering s items is the likelihood of the latents that are still open at that
   node, i.e. attached both inside and outside it, o(s) of them; its size is the epsilon-rank of that function family
   (one round) or a local expansion of it around a broadcast estimate (two or more rounds);
3. (combination) every merge, at every scale, is the same two operations: add log-likelihoods, and sum out
   (contract) the latents that close at that node.
Untangled means b = O(log N) and o(s) small (bounded, or growing with the number of groups, not of items).

The intrinsic quantity is the minimal message (the Nerode class of T1, or its epsilon-cover); the open latents are the
generative model's explanation of it, and two latent models of the same law give the same minimal message. The
smallest number of latent states that makes the two sides of a cut independent is the nonnegative rank of the joint
law across the cut, so the profile is a property of the law, not of a chosen parametrisation (**PROVEN**).

How the three parts of the intuition map:
- "zooming out, things get more complicated": at larger scales the open latents are those of larger structures
  (groups, groups of groups); they may be richer, so the message dimension may grow with scale;
- "it is fine for larger structures to have more dimensions": the message size is set by the open latents at that
  scale, not by how many items lie below; under noise it must in fact grow slowly (5.7);
- "how information combines stays simple": every merge is addition of log-likelihoods plus summing out what closes,
  at every scale; what changes with scale is only which latents are open.

In this language D1 is SFU with a fixed hierarchy, one round, counting bits; D3 is the message dimension; D4 at the
item level is what remains after the latents are summed out (a shared latent turns its items into a clique), which is
why it overstates tangledness; D5 bounds how much the input law itself must carry across a cut; D6 is the linear
special case where the "latents" are multipole moments. Status: the definition is a choice (not a claim); the
theorems that make it the natural one are T2-T4 below.

## 5. Theorems and sharp claims

### 5.1 Messages are likelihoods of the open latents (T2)

Take a joint model p(x, u) = prod_c psi_c(u_c) prod_i p(x_i | u_{L(i)}), items attached to latent sets L(i). At a
node S, a latent is **closed** if all its items and factors lie inside S, **open** if it is attached inside and
outside. The message m_S(u_open) = sum over closed latents of the product of the factors and item likelihoods inside S,
and the merge multiplies children's messages and sums out the latents that close there (variable elimination along
the hierarchy). **PROVEN** (standard). The posterior of the target is the normalised root message.
Minimality: if the outside data can produce likelihoods that separate functions of the open latents, two contents with
non-proportional messages are Nerode-inequivalent, so no smaller message is exact. **PROVEN UNDER** that richness
condition.

Consequences:
- Tiny-PFN world (`work/tiny-pfn/world.py`): each source's reliability v_s attaches only to that source, so it closes
  inside the source; the hidden answer y* attaches to every source's reading, so it is the only open latent at any
  cut between sources. Messages are functions of y*. Within a source, v_s is open too. **PROVEN** (from the model).
- Shared-error groups: each group's shared errors are open at every cut that splits the group, so message size
  grows with the number of **straddling groups**, not of items. Under a hierarchy aligned with the groups (possible
  when membership labels are visible: routing), only y* is open. **PROVEN**.
- Unstructured correlated errors (World FA's inverse-Wishart variant (b), N traders, T past questions): the
  covariance Sigma attaches to every trader, and each factor N(e_t; 0, Sigma) couples all traders at time t. The
  posterior of Sigma depends on the full scatter matrix, whose cross block E_S^T E_{S^c} must be formed across a
  trader cut S. Two error histories E_S != E'_S give different cross blocks for some completion E_{S^c}, so the exact
  message is the whole history E_S: T |S| numbers, a volume law in both T and |S|. With k common factors of known
  loadings, the open latents are the k factor values per question, and the message is a Gaussian likelihood of them:
  O(k T) numbers, independent of |S|. **PROVEN UNDER** the Gaussian model (exact posterior of Sigma; for the
  predictive of the query, generically).

### 5.2 Fixed-size additive messages: exactly the exponential families; smooth families pay polylog

- **PROVEN UNDER** regularity (support independent of theta; Koopman 1936, Pitman, Darmois): for iid sources, an
  additive sufficient statistic sum_i phi(x_i) of fixed dimension for every N exists only for exponential families.
  Uniform(0, theta) shows the regularity is needed: its statistic max_i x_i merges by max, a monoid but not a sum.
- Distributed-statistics form: one-shot combination of local estimates loses information exactly to the extent the
  family deviates from a full exponential family (Efron's statistical curvature; Liu and Ihler 2014,
  arXiv:1410.2653). **PROVEN** (cited).
- Smooth families (**PROVEN UNDER** analyticity of l(theta; x) in a complex neighbourhood of the prior range,
  uniformly in x): Chebyshev interpolation gives epsilon-rank r(epsilon) = O(log(1/epsilon)^d). Per-source sup error
  epsilon gives log-posterior error at most 2 N epsilon (so KL <= 2 N epsilon); hence r = O(log(N/delta)^d) additive
  numbers per source suffice for KL <= delta.
- **MEASURED** (N1c: the tiny-PFN world scaled to N sources; each source sends the r Chebyshev-interpolation
  coefficients of its log-likelihood of y* on the prior range [-5.5, 5.5]; the root sums them; KL(Bayes || approx)
  at the 95th percentile over 200 episodes, 80 at N = 1024, 30 at N = 4096):

  | N | r = 24 | r = 32 | r = 48 | r = 64 | r = 96 | smallest r for KL <= 1e-3 |
  |---|---|---|---|---|---|---|
  | 3 | 9.4e-3 | 3.3e-3 | 1.6e-4 | 7.6e-6 | 2.7e-8 | 48 |
  | 64 | 2.4e-2 | 1.5e-2 | 7.0e-4 | 9.2e-5 | 5.6e-7 | 48 |
  | 1024 | 7.9e-2 | 4.1e-2 | 1.0e-3 | 8.4e-5 | 7.3e-7 | 64 |
  | 4096 | 2.2e-1 | 5.4e-2 | 1.2e-3 | 9.7e-5 | 8.6e-7 | 64 |

  At fixed r = 64 the KL grows 13-fold while N grows 1,365-fold (about N^0.36, gentler than the worst-case 2 N
  epsilon), and each extra 32 coefficients cut it about 100-fold: r grows like log N, as T3a says. The constant is
  large (tens of numbers) because the per-source log-likelihood switches between its three variance components over
  short distances.

### 5.3 Rounds: broadcasting a consensus turns global functions into local statistics (T4)

One round (each source writes a message without seeing anything else; merge = sum): the message must fix the
log-likelihood on the whole region where the final posterior could lie, i.e. a global function of the open latents.
For location-type families its size is roughly the number of resolution cells of that region, (prior width / feature
width)^D times a polylog in N/delta, so it grows exponentially in the number D of open latents (**HEURISTIC**: it is
the eigenvalue count of a smooth translation-invariant kernel on a box, as in Landau-Slepian time-frequency limiting;
**MEASURED** below). In words: one round must ship the likelihood everywhere it might matter; two rounds ship it only
where it does.

Two rounds (round 1: additive moments give a root-N-consistent pilot; the root broadcasts it; round 2: each source
sends the degree-p expansion of its log-likelihood on a window of width about sqrt(log N / N) around the pilot):
C(D + p, p) numbers, with log-posterior error about N (log N / N)^{(p+1)/2}, which vanishes for p >= 2.
**PROVEN UNDER** local asymptotic normality, p + 1 bounded derivatives near the truth, and a root-N pilot from
finitely many moments (true for contamination and Gaussian location mixtures; Wu and Yang, arXiv:1807.07237).
This is the classical one-step estimator (van der Vaart 1998, sec. 5.7), known in distributed statistics as a
one-step or surrogate-likelihood round (Huang and Huo, arXiv:1511.01443; Jordan, Lee and Yang, arXiv:1605.07689);
lower bounds for protocols with and without broadcast: Duchi, Jordan, Wainwright, Zhang, arXiv:1405.0782.
It is also Kelvin's design for the aggregation track: agents write their messages after reading the consensus.

**MEASURED** (N1c tiny-PFN, D = 1 open latent; N2, D = 1-3 open latents, contamination model):
- Tiny-PFN world, one open latent (N1c; two rounds = a 3-number GLS pilot from the labelled residuals, then p + 1
  Chebyshev coefficients of each source's log-likelihood on a window of 10 pilot standard deviations):

  | N | one round, KL at r = 64 numbers | two rounds, KL at 3 + 5 = 8 numbers | two rounds, KL at 3 + 7 = 10 numbers |
  |---|---|---|---|
  | 64 | 9.2e-5 | 1.8e-3 | 7.6e-5 |
  | 256 | 7.8e-5 | 1.5e-5 | 5.6e-8 |
  | 1024 | 8.4e-5 | 6.7e-8 | 2.7e-11 |
  | 4096 | 9.7e-5 | 3.5e-10 | 8.7e-12 |

  At fixed size the two-round error falls steeply with N (theory: about N^{(1-p)/2} up to logs), while the one-round
  error creeps up. Below N = 64 the window is not local and two rounds fail at these sizes (at N = 3, 86% of episodes
  have posterior mass outside the window): rounds pay off once the posterior is narrow compared with the features of
  each source's likelihood.
- Small codes of the kind a network might learn (N1d, same world; mean KL over 300 episodes, 100 at N = 1024):

  | N | one round, 2 numbers (GLS with E[1/v given residuals]) | two rounds, 2 + 3 numbers (pilot, then l' and l'' at it; Laplace) | three rounds, 7 numbers |
  |---|---|---|---|
  | 8 | 0.13 | 1.2e-2 | 7.0e-3 |
  | 64 | 0.23 | 1.9e-4 | 8.6e-5 |
  | 256 | 0.20 | 1.1e-5 | 4.5e-6 |
  | 1024 | 0.16 | 4.9e-7 | 2.7e-7 |

  The small one-round code has a floor that does not fall with N (it cannot use how far each reading sits from the
  consensus, which also signals reliability); the same budget spent over two rounds falls roughly like N^-2 here. At
  N = 3 neither works (Laplace is poor for a wide posterior). A re-implementation of exact Bayes in the same script
  gives KL below 1e-15 (sanity check).
- D open latents (N2, sources of unknown reliability reporting on D shared latents: reliable with probability 0.7,
  N(theta, 0.25 I), otherwise N(0, 4 I)):

  | D | optimal one-round linear code, RMS error 1e-2 / 1e-3 | one round, tensor Chebyshev, KL <= 1e-3 | two rounds, tensor Chebyshev, KL <= 1e-3 |
  |---|---|---|---|
  | 1 | 12 / 19 | 24 (N = 512) | 7 (N = 512) |
  | 2 | 128 / 334 | 1,024 (N = 512) | 28 (N = 512) |
  | 3 | 746 / 1,646 (likely low: 3,000 sampled functions) | 32,768 (N = 256) | 347 (N = 256) |

  The optimal one-round code grows about tenfold per extra open latent; the two-round size grows polynomially (with a
  total-degree instead of a tensor basis the count at degree 4 would be C(D + 4, 4) + D + 1 = 7, 18, 39 numbers;
  not run).
  Two rounds fail at N = 64 for D = 3 with degree <= 6 (window not yet local).

Reading: rounds, not hierarchy shape, are the main lever on message size once routing is solved. A transformer gets
rounds for free (each layer reads pooled context, then writes), which is one way depth can substitute for width.

### 5.4 Routing: what it costs, and when density is needed

**T5, routing separation (PROVEN).** Items (k_i, a_i) with keys k_i and bits a_i; f = 1 iff some key carries two
items with a = 1. (i) Every fixed hierarchy has a node with N/3 <= |S_v| <= 2N/3; placing the keys so that about
N/3 of them straddle that cut (other values 0) turns f into the complement of set disjointness on N/3 bits, so the
message at v needs Omega(N) bits even with randomness and error 1/3 (Razborov 1992; Kalyanasundaram and Schnitger
1992; a tree is a one-way protocol across the cut). (ii) With labels = keys (log N bits) and the group-by rule, every
content message is at most 2 bits. The Bayes version (World FA, flat tokens in random order, per-trader statistics
combined nonlinearly) is the same: at a balanced positional node every straddling trader's partial statistic must
cross (**PROVEN**, by T1, for exact Bayes). So World FA's failure was a routing failure, and since one attention head
can do this routing (below), a learning failure, not a representational one.

**T6, routing cost in attention = sign-rank (PROVEN).** A hard-threshold head (i attends to j iff q_i . k_j >= t)
realises the routing relation R with key dimension d only if the sign pattern of R has sign-rank at most d + 1
(absorb t into one extra coordinate), and sign-rank d suffices (t = 0); this is the definition of sign-rank.
Equality on N labels has sign-rank at most 3 (1/2 - (i - j)^2 = (2i, 1/2 - i^2, 1) . (j, 1, -j^2)), order has 2; with
noise or bounded logits, random +-1 codes of length O(log N) give margin 1/2 (Hoeffding and a union bound).
Hadamard-type relations ("attend iff <i, j> mod 2 = 0") need sign-rank at least sqrt(N) (Forster 2002). Natural
routing (same source, same question, before/after, nearby) is cheap; parity-like routing is not.

**Sharp routing is exactly the hard regime of attention (PROVEN, from Alman and Song, arXiv:2302.13214).** For soft
attention to put 1 - delta of its mass on one item out of N, the logit gap must be at least about log(N / delta). With
A = exp(Q K^T / d) and entries in [-B, B], every logit lies in [-B^2, B^2], so the gap is at most 2 B^2 and B must be
at least sqrt(log(N / delta) / 2), of order sqrt(log N): the
regime where approximating attention in n^{2 - Omega(1)} time is impossible under SETH, whereas B = o(sqrt(log n))
admits n^{1+o(1)}. But a specific routing relation (equality of labels) is computed in O(N log N) by sorting or O(N)
by hashing, so quadratic cost is necessary for generic sharp similarity search (also Rubinstein, arXiv:1803.00904),
not for label routing.

**Routing by keys is a join (PROVEN as a correspondence; cited theory).** Grouping items whose labels are equal and
aggregating within groups is an equi-join followed by an aggregate. Database theory treats joins, constraint
satisfaction and graphical-model inference as one sum-product problem whose exact cost is governed by width
parameters of the query, not of the data's routing (FAQ: Abo Khamis, Ngo, Rudra, arXiv:1504.04044). This is the
input-dependent-routing counterpart of treewidth: the routing varies with the data, the query structure does not.

**Per query, attention is already scale-free (PROVEN).** For one query, the triples (max logit m, normaliser l,
weighted value sum o) over any set of keys merge associatively: m = max(m_a, m_b), l = l_a e^{m_a - m} + l_b e^{m_b - m},
o likewise (the online-softmax rule used by FlashAttention). So one query's attention is a monoid reduction along any
hierarchy with d + 2 numbers per message. Dense attention runs N such reductions, one per query, per layer; the
number of genuinely different reductions per layer is the number of **routing classes** (distinct attention rows up
to a tolerance). This is the measurable form of "dense attention pays for routing": cost N x N, information about
C x N, with C the number of routing classes.

**T7, correlation-based routing (PROVEN for the promise problem; HEURISTIC for iid histories).** If items must be
grouped by whether their T-step error histories correlate (who copies whom), deciding correlation >= rho versus
<= -rho needs Omega(min(T, 1/rho^2)) bits per pair: gap-Hamming (Chakrabarti and Regev, arXiv:1009.3460) on
n = 4/rho^2 coordinates, each repeated T rho^2 / 4 times (valid for rho >= 2/sqrt(T)). Sending O(log(1/delta)/rho^2)
random coordinates achieves it. So per-item keys must have about min(T, 1/rho^2) bits: cheap above the statistical
detection threshold, linear in T at it.

### 5.5 Multi-hop needs rounds; parity is untangled but unlearnable

- k-step pointer chasing: (k - 1)-round protocols need Omega(n - k log n) bits deterministically and
  Omega(n/k^2 - k log n) randomised; k rounds need O(k log n) (Nisan and Wigderson 1993). A single upward pass of a
  hierarchy is one round, so multi-hop lookups are volume-law for any one-pass hierarchy and cheap with depth.
  **PROVEN** (cited). Constant attention layers simulate constant rounds of massively parallel computation, and
  log depth suffices for k-hop tasks (Sanford, Hsu, Telgarsky, arXiv:2402.09268); one layer needs exponentially more
  size than two for induction heads (arXiv:2408.14332). **PROVEN** (cited).
- Parity is maximally untangled under every definition here (1 bit across every cut; rank 1 in +-1 form), yet hard for
  statistical-query learners, which include gradient methods at bounded precision (Abbe et al., arXiv:2108.04190;
  Shalev-Shwartz, Shamir, Shammah, arXiv:1703.07950). **PROVEN** (cited). World 3 (XOR chains) is our instance:
  the equality version was learnable, the XOR version never was. **MEASURED** (outbox/world3-chains-pilot).
  Lesson: untangledness is about what a solution must carry, not whether training finds it.

### 5.6 Pairwise structure, density, and the non-Gaussian part

- For f(x) = x^T U x mod 2 with interaction graph A, the classes across a cut are determined by (q_S(x_S), B^T x_S)
  with B = A[S, S^c], so rank_GF2(B) <= log2 K_f(S) <= rank_GF2(B) + 1 (each fibre of x_S -> B^T x_S contributes one
  or two values of q_S). The best hierarchy's worst message is therefore the rank-width of A, up to 1 bit.
  **PROVEN**. Complete graph: rank-width 1, treewidth n - 1; G(n, 1/2): rank-width about n/3; sparse random
  G(n, c/n), c > 1: linear (Lee, Lee, Oum). **MEASURED** brute-force check (N3b, n = 12, balanced cut, all 2^12
  inputs): complete graph 4 classes = 2^(1+1), cut-rank 1, treewidth 11; G(12, 1/2) 64 classes, cut-rank 6; a random
  3-regular graph 48 classes, cut-rank 5 (between 2^5 and 2^6, as the bound allows).
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

- **T11a, exact output law (PROVEN).** If f has K_v classes along T and the input law is a latent tree along the same
  T with R_v latent states at node v (given u_v, the two child subtrees are independent), then
  Q_v(c | u_v) = sum over (c_a, c_b) merging to c of [sum_{u_a} P(u_a | u_v) Q_a(c_a | u_a)] [sum_{u_b} P(u_b | u_v) Q_b(c_b | u_b)],
  so the exact law of f(X) (every tail) costs O(sum_v R_v (R_a K_a + R_b K_b + K_a K_b)). Exact estimation needs the
  computation and the input law untangled along the same hierarchy; the cost scales with the product of the two bond
  dimensions.
- **Misalignment.** If the law is a tree along T1 and the computation along T2, contraction costs exp(O(treewidth of
  T1 union T2)) (Markov and Shi, arXiv:quant-ph/0511069; upper bound). **MEASURED** (N3a; random binary trees by random
  pairwise merging; treewidth of the union graph bracketed by a contraction lower bound (MMD+) and a min-fill upper
  bound; 5 pairs per N):

  | N leaves | T2 = copy of T1 | 4 random prune-and-regraft moves | 32 moves | T2 independent of T1 |
  |---|---|---|---|---|
  | 16 | 2 | 3.4-4.0 | 4.0-6.0 | 4.2-6.6 |
  | 64 | 2 | 3.8-4.2 | 5.4-14.2 | 6.8-23.6 |
  | 256 | 2 | 4.0-4.8 | 5.8-17.8 | 12.6-95.6 |

  Aligned hierarchies stay at treewidth 2; misalignment raises it steadily, and for independent trees the upper
  bound grows about linearly (0.37 N) while the lower bound grows slowly, so linear growth is not established
  (**CONJECTURE**: linear for independent random trees, as for expanders).
- **Expectations need rank; tails need dimension (PROVEN).** A rank-chi tree network gives E_p f by contraction in
  O(N chi^3 R^3), but the rank-2 function sum_i a_i x_i has #P-complete tails. With activation noise sigma and range R
  per coordinate, a d-dimensional message has about (R/sigma)^d distinguishable states, and the law of a sum is a
  convolution on that grid (pseudo-polynomial; exact up to the discretisation set by sigma).
- **Noise forces dimension to grow with scale (PROVEN UNDER** a Gaussian prior on theta, N sources with Fisher
  information I each, and a root that sees one m-dimensional message with signal range R and noise sigma per
  coordinate**).** If the root outputs a Gaussian within KL delta of Bayes's N(mu_N, v_N) on every input, its
  mean-squared error is at most about (1 + 2 delta) v_N, with v_N = 1/(1 + N I), so
  I(M; theta) >= (1/2) log2((1 + N I) / (1 + 2 delta)) by rate-distortion, while I(M; theta) <= (m/2) log2(1 + R^2/sigma^2)
  by capacity. Hence m >= log2((1 + N I)/(1 + 2 delta)) / log2(1 + R^2/sigma^2). Normalised (bounded) mean-pooled
  representations must widen like log N; unnormalised sums need not. This is "larger structures need more
  dimensions" as a necessity, not a permission.
- **Additive messages give certified tails (PROVEN, classical).** Given the latents, a sum of independent messages
  has a computable cumulant generating function, so Chernoff bounds certify upper tails and saddlepoint
  approximations have relative error O(1/N) (Daniels 1954). This is the M3 route for an SFU architecture.

### 5.8 The renormalisation view, made precise

- Latent-tree data (b-ary, q states, leaves in a line) have block mutual information at most
  (2(b - 1) log_b L + O(1)) log2 q for a contiguous block of L leaves: the roots of the maximal subtrees covering the
  block separate it from the rest. **PROVEN**. Log law, like 1D critical states and MERA, against the volume law of
  generic data and the constant of Markov chains.
- Correlations: two leaves at distance l have tree distance at least 2 log_b l, so for a symmetric channel with
  second eigenvalue lambda their correlation is at most l^{-2 log_b(1/lambda)}: power law at most, exactly on average.
  **PROVEN** (bound). Lin and Tegmark (arXiv:1606.06737) make the grammar version.
- Whether information about the root survives coarse-graining is a sharp threshold: for a binary symmetric channel,
  reconstruction is possible iff br(T)(1 - 2 epsilon)^2 > 1 (Evans, Kenyon, Peres, Schulman 2000, the Kesten-Stigum
  bound). **PROVEN** (cited). Relevant vs irrelevant directions in the RG sense have this exact meaning on trees.
- For additive merges with product input laws, the rescaled law of the message converges to a Gaussian or stable
  law (the CLT is the fixed point of the merge map); max-merges converge to extreme-value laws. **PROVEN**
  (classical). Under SFU with additive merges the large-scale messages are therefore near-Gaussian, which is the
  rationale for the latent-hub family keeping its non-Gaussian structure in a few latents (**HEURISTIC**).

### 5.9 Which tasks are untangled, by definition

| Task | D1 fixed cuts | D2 routed | D3 rank | D4 item treewidth | D5 data MI | SFU |
|---|---|---|---|---|---|---|
| Conditionally independent sources, exponential family | (d/2) log s bits | same | d + 1 (log space) | N - 1 | (d/2) log s | 1 open latent, d numbers, 1 round |
| Same, smooth non-exponential (tiny-PFN) | polylog | same | r(delta/N) | N - 1 | log | 1 latent; 48-64 numbers in 1 round, 5-8 in 2 (N >= 64) |
| G shared-error groups, labels visible | ~ G r at a positional cut | r (group-aligned) | as D1/D2 | >= group size | log | log G label bits; open = straddling groups |
| G groups, membership hidden | function of G latents | no labels | grows fast with G | N - 1 | log | 1 round: exponential in G; broadcast: O(G^2) |
| Keyed aggregation / binding (World FA) | Omega(N) | O(1) + log N label bits | full rank | high | 0 | untangled with routing |
| k-hop lookup (pointer chasing) | Omega(n) per pass | O(k log n) in k rounds | high | high | 0 | untangled with k rounds (log k layers) |
| Parity | 1 bit | 1 bit | 1 | n/a | 0 | untangled; not learnable by SQ |
| GF(2) quadratic on an expander | Theta(N) | Theta(N) | Theta(N) | Theta(N) | 0 | tangled; uniform expectation still polynomial |
| Unstructured correlated errors (inverse Wishart) | T \|S\| numbers (whole history) | same | same | dense | - | tangled unless k known-loading factors (then O(k T)) |
| One-query retrieval (attention) | volume without broadcast | - | - | - | - | d + 2 numbers after broadcasting the query |

## 6. Transformers: routing plus combination, and how to test it

### 6.1 Attention is the merge of Gaussian messages, with the prior as a sink (PROVEN, identity)

Let tokens j carry Gaussian messages about y with precision tau_j and mean mu_j. A head with logits
L_j = log tau_j + c, values mu_j e_mu, plus a sink token with logit log tau_0 + c and value mu_0 e_mu + e_0, outputs
the posterior mean in the e_mu channel and a_0 = tau_0 / sum_{j >= 0} tau_j = tau_0 x (posterior variance) in the e_0
channel (the sum includes the sink).
Without the sink (or another channel carrying the normaliser) the merged message loses its total precision and is not
closed under further merges. With a Gaussian prior N(mu_0, 1/tau_0), the sink is the prior. Oracle GLS in the
tiny-PFN world (reference b) is exactly one such head. Exponential-family messages merge the same way in natural
parameters, with an MLP undoing the normalisation (construction; **PROVEN** up to MLP approximation error).

### 6.2 The decomposition

For layer l and head h, write the logits as L_ij = R(lambda_i, lambda_j) + w_j + e_ij: a **routing** term from
discrete labels (same source, same question, token type), a **weighting** term per key (a log-precision-like scalar
from the key's content), and a residual. Values are the **content** (natural-parameter-like messages). Layers are
**rounds**. "Dense attention mostly pays for input-dependent routing" becomes four testable claims:
(a) rows cluster into few routing classes C, explained by labels; (b) given routing and weighting, the content is
low-dimensional and merges additively; (c) a static (input-independent) routing of the same sparsity fails while
label routing does not; (d) the union of supports over inputs is dense while each input's pattern is sparse.

Two consequences. (i) Expressivity: a transformer whose labels are linearly readable can implement an R-round SFU
computation with message size k in O(R) layers and width O(k + label dimension): per round, one head per routing
pattern gathers the group's natural parameters (normaliser kept by a sink), and an MLP turns sums into the next
round's local statistics (**PROVEN** as a construction sketch; MLP steps are approximations). So density is never
needed to *represent* SFU computations; it is needed when routing is by similarity rather than labels (T7, SETH).
(ii) Learning: at uniform attention (World FA's measured own/other ratio 1.00) the gradient on the query-key form
along the binding direction is a covariance between "same ID" and how useful the gathered content is, which is near
zero until the content path is useful, and the content path is useless until binding exists: a saddle that neither
part can leave alone (**HEURISTIC**; consistent with World FA's 0 of 15 flat runs and with success once binding was
given).

A caution about our worlds. In the tiny-PFN world the grouping (3 sources x 5 questions) is the same in every episode;
the net must find it by matching random per-episode ID embeddings (no positional encoding, `work/tiny-pfn/pfn.py`),
but an architecture designer can hard-wire it. So the PFN prior tests content and combination; routing is tested by
World FA (flat tokens, random order, binding by trader ID).

### 6.3 Measurements on our trained nets (all CPU; forward passes plus small fits)

Each measurement names what it tests, the Bayes reference, a prediction, and what would refute it. "Cut" means a set
of tokens; "replace" means redraw those tokens' errors from the prior while keeping the shown labels, the IDs and y*
(for a query reading, o' = y* + e'), so every modified episode is a valid draw.

| # | Measurement | Bayes reference | Prediction | Refuted if |
|---|---|---|---|---|
| M1 | **Interaction profile.** For a cut C and its complement, Delta = L(a,b) - L(a',b) - L(a,b') + L(a',b'), where L = log p(y1 \| x) - log p(y2 \| x) at points around y* and a, a' (b, b') are two draws of C's (the rest's) errors. Interaction share = Var(Delta) / (variance of the two main effects). Cuts: one source vs the rest (source-aligned); two of one source's labelled questions vs the rest (source-splitting). | exactly 0 for source-aligned cuts (log-odds add over sources given y*, **PROVEN**); 0.039 within a source; for Gaussian heads use Bayes projected to a Gaussian (0.054 both) | source-aligned share near the output family's floor and below the within-source share; lower in group A than in group B (**CONJECTURE**; first run in 6.4) | source-aligned share >= within-source share well above the floor: the net combines sources non-additively (tangled at the source scale) |
| M2 | **Routing classes and label fit.** Per layer and head, cluster attention rows (greedy, total variation <= 0.05 and 0.1); C = number of clusters per episode. Fit L_ij = R(pair class) + w_j + e_ij, pair class = (same source?, same question?, token types), w_j free per key and episode; R^2 of the full fit and of R alone on held-out episodes. | GLS needs one class for the query (all query readings, weight = log precision) and per-source classes to gather residuals | **CONJECTURE**: C <= 7 of 16 tokens in layer 1; R explains most logit variance in layer 1; w_j tracks the source's log precision in the combining layer (phase B found reliability shaping layer-2 weights, E9-E10) | C close to the number of tokens, or label features explain little: routing is not label-like |
| M3 | **Routing substitutions (causal).** Replace a layer's logits by (i) R + w, (ii) R alone (w averaged within its class), (iii) the same net's logits from another episode with the same layout, (iv) the episode-averaged pattern. KL to the net and to Bayes. | - | (i) close to the net; (ii) drifts toward equal-precision weighting (reference a); (iii), (iv) worse than (i); in World FA (random token order) (iv) fails outright | (ii) or (iv) as good as (i): the input-dependent part of attention is not needed |
| M4 | **Message size across a cut (bottleneck probes).** Distil the net's predictive into an encoder of source s's tokens to k numbers plus a decoder that also sees the other tokens; decoder either general (any network) or additive (sum of per-source k-dim codes, then a readout). Smallest k with KL(net \|\| probe) <= 0.01. | general decoder: k = 2 (the reading and the sum of squared residuals); additive decoder: tens of numbers with a fixed linear basis (N1c), a few with learned codes (N1d) | **CONJECTURE**: the net's general k is 2-3, its additive k larger; nets nearer Bayes have smaller additive k | general k >> 2: the net passes more than the sufficient statistic between sources (tangled beyond the task) |
| M5 | **Sink and posterior variance.** In the combining layer, regress the query token's attention to itself (or to the token whose value is most source-independent) on the net's predicted variance and on Bayes's. | a_0 = tau_0 x posterior variance (identity, 6.1) | **CONJECTURE**: R^2 >= 0.5 in some nets; if not, find the precision-sum channel with a linear probe | neither the sink weight nor any linear feature carries the precision sum: the net does not merge in natural parameters |
| M6 | **Effective rounds.** Sensitivity of source s's representation at its query token after layer 1 to other sources' errors, relative to its own. | 0 for a one-round computation | descriptive at N = 3 (rounds help only from N of about 64, N1c); becomes a test when N is scaled | - |
| M7 | **World FA.** M1 across trader cuts in the binding-given arms; M2 in the flat arms; M4 across a trader cut. | interaction non-zero exactly for cuts that split correlated traders, zero between independent blocks; exact message across a trader cut = that side's whole error history (5.1) | flat arms: C = 1, no routing learned (consistent with the measured own/other ratio 1.00); binding-given arms (which need no routing: the column is in the token) show Bayes's interaction pattern across trader cuts | binding-given arms reach 0.95 of the history value with an interaction pattern unlike Bayes's: they get there by a different combination |

Costs: M1, M3, M5, M6 need only forward passes (10^4 samples per cut is minutes on CPU for the phase-B nets); M2 is a
least-squares fit on stored attention maps; M4 trains small probes (minutes each).

### 6.4 First measurement: the interaction profile on 40 trained tiny-PFN nets (N5)

M1 run on the clean tiny-PFN cohort (`work/tiny-pfn/runs`, 8 configurations x 5 seeds, 4,000 episodes x 4 variants
per cut, log-odds of y* + {-0.6, -0.3, 0.3, 0.6} against y*; `work/scale-free-theory/n5_interaction.py`). Gaussian-head
nets cannot represent Bayes's non-Gaussian posterior, and projecting Bayes onto a Gaussian already creates
interaction, so their reference is Bayes's moment-matched Gaussian.

| Nets | cross-source interaction share | within-source share | KL to Bayes |
|---|---|---|---|
| exact Bayes | 0 (3.6e-31) | 0.039 | 0 |
| Bayes projected to a Gaussian (floor for Gaussian heads) | 0.054 | 0.054 | 0.010 |
| 2 layers, mixture head, learned the task (12 nets) | 0.012-0.041 | 0.037-0.045 | 0.007-0.032 |
| 2 layers, Gaussian head (15 nets) | 0.054-0.087 | 0.050-0.062 | 0.016-0.050 |
| 2 layers, mixture head, seeds stuck far from Bayes (3 nets) | 0.109-0.114 | 0.018 | 0.144-0.145 |
| 1 layer, Gaussian head (10 nets) | 0.186-0.206 | 0.020-0.024 | 0.174-0.197 |

**MEASURED.** Across all 40 nets the cross-source share correlates with KL to Bayes at 0.96; within the 12 successful
mixture-head nets at 0.98; within the 15 Gaussian-head 2-layer nets at 0.74. The one-layer nets and the stuck seeds
show an inverted profile: too much interaction across sources and too little within a source. That is what a single
pool-then-transform round must do (the per-source nonlinearity, reliability from the sum of squared errors, is applied
after sources are mixed), and it is the scale-1 version of conjecture C2 below.

Limits: one world; clean nets, not the noisy phase-B nets; correlation across nets, not a causal test. Interaction is
not a generic proxy for distance from Bayes: equal weighting (reference a) is exactly additive across sources yet
0.36 nats from Bayes. Next: the same profile on the 12 noisy phase-B nets against their A/B grouping, and an
intervention that removes the cross-source interaction (M3-style) to see whether KL falls.

### 6.5 Specification for the hierarchy-vs-dense experiment (for the experimental agent)

The comparison is only informative if it separates routing from combination, and rounds from hierarchy shape.

**Design: 2 x 2, plus rounds.**
- Routing: *oracle* (tokens arrive grouped by source, or a mask from the true grouping) vs *learned* (flat tokens with
  per-episode ID embeddings, as in the phase-B and World FA formats).
- Combination: *dense* (the phase-B transformer, matched parameters and FLOPs) vs *SFU tree* (per-group encoder, a
  set network over that group's tokens, to k numbers; merge = sum; readout = small MLP to the mixture-density head).
- Rounds, tree only: one pass, or two passes (pass 1 sends 3 numbers per group, the root broadcasts a pilot mean and
  scale; pass 2's encoder also sees the pilot and sends k numbers).
- Learned routing for the tree: a router that assigns tokens to groups from their ID embeddings (attention to learned
  prototypes, or hashing); the dense net routes by itself.

**Worlds.**
- W1: tiny-PFN prior with N in {3, 8, 24, 64, 256} sources (same generator; exact Bayes from the per-source
  log-likelihood sum on a grid, `work/scale-free-theory/n1c_rounds.py: ell_grid`).
- W2: W1 plus G shared-error groups (one shared error per group and question), membership visible as a label or
  hidden; G in {1, 2, 4, 8}.
- W3: World FA, binding-given vs flat.

**Read-outs.** KL to Bayes against k; smallest k reaching KL <= 0.01 and 0.001 (95th percentile over episodes) per N;
M1-M4 on the dense nets; FLOPs.

**Predictions (to pre-register).**
- P1 (W1, oracle routing): two-pass tree at small k (about 5-7) reaches the Bayes KL that the one-pass tree reaches
  only with larger k, once N >= 64; at N = 3 the two are similar. Calibration for small learned codes from N1d:
  a 2-number one-round code stays at KL about 0.2 at every N, while 5 numbers over two rounds reach 2e-4 at N = 64
  and 5e-7 at N = 1024. Refuted if the two-pass tree is no better than one-pass at N >= 256.
- P2 (W2, visible membership): a group-aligned tree needs the same k as W1; a tree whose groups split the shared-error
  groups needs k growing with the number of straddling groups.
- P3 (W2, hidden membership): one-pass k grows quickly with G (N2: an optimal fixed linear one-round code needs about
  19, 330 and 1,650 numbers for 1, 2 and 3 open latents at RMS error 1e-3); two passes stay polynomial.
- P4 (routing): with learned routing on flat tokens, both architectures fail to learn binding without help (World FA:
  0 of 15 flat runs), and both succeed with oracle routing. Refuted for "density pays for routing" if dense with
  learned routing matches dense with oracle routing, or if the tree with oracle routing falls far short of dense with
  oracle routing at matched compute.
- P5 (M3 payoff): for the SFU tree, given y* and the latents, the summed message is a sum of independent per-group
  terms, so the law of the output (all tails) is a convolution that can be computed exactly on a grid or bounded by
  Chernoff; compare against sampling at equal compute (the M3 bar).

## 7. What would count as a major advance, and the shortest path

Three candidates, each tied to a program target.

1. **M2: in-context Bayes in transformers as scale-free computation.** Show, across the phase-B nets and scaled
   worlds, that trained transformers route by labels, merge natural-parameter-like messages through attention (the
   prior as sink), use layers as rounds, and carry messages of the size the open-latent profile requires; and that
   each departure from Bayes is the error of the architecture's merge family at some scale. The big-miss departure
   already fits this form: at the token-to-source scale, Bayes adds the unbounded statistic e^2, while the first
   layer averages normalised (bounded) per-token features, so the Bayes merge lies outside the family (the E20 toy's
   "capped per-error evidence"); depth adds a per-token transformation before pooling, which enlarges the family
   (E17, E19). **CONJECTURE** (general form): the departure equals the projection of the Bayes merge onto the merge
   family, scale by scale, and is predictable from the architecture before training.
2. **M3: an untangled architecture at transformer loss with exact tails.** An SFU tree (label routing, additive
   natural-parameter messages, a broadcast round) that matches the dense transformer on W1-W2 and whose output law is
   computed exactly by convolution given the latents, beating sampling on certified tail bounds. Our literature check
   found no strictly low-treewidth model at transformer loss on a natural task; this would be the synthetic first
   step, and it would have to survive hidden membership (W2) to matter.
3. **A necessity map.** Which in-context tasks need dense routing (sharp similarity search; correlation discovery
   near the detection threshold, T7), which need only label routing (binding, T5), and which need rounds (multi-hop;
   hidden membership, T4), with matching lower bounds. Most pieces exist (5.3-5.5); hidden membership has only upper
   bounds and numerics here.

A fundamental blocker would look like this: if two-pass SFU trees cannot match the dense transformer on hidden
membership at any moderate k, dense attention is earning its cost through structure discovery, which T7 says costs
about min(T, 1/rho^2) bits per item plus comparisons. That would itself be a sharp result to report.

**Shortest path.**
1. Done today: definitions, theorems, N1c/N1d/N2/N3 numbers (this note).
2. M1 is done on the 40 clean tiny-PFN nets (6.4). Next, M1 on the noisy phase-B nets against their A/B groups, an
   intervention that removes cross-source interaction, and M2/M5 (CPU, hours). This decides whether candidate 1 is
   real.
3. W1 with oracle routing: tree one-pass vs two-pass vs dense, N up to 256 (CPU or one L4, a day).
4. Exact output law of the tree by convolution against sampling (CPU).
5. W2 hidden membership and W3 routing; then a natural multi-source dataset.

## 8. Claim register

| ID | Claim | Status | Section |
|---|---|---|---|
| T1 | Minimal exact tree message at v = number of Nerode classes; equals one-way CC; tolerance adds at most epsilon per node | PROVEN | 2 |
| L1 | Unrestricted routing rules absorb the whole computation | PROVEN | 2 |
| T2 | Messages are likelihoods of the open latents; merges add log-likelihoods and sum out closing latents; minimal under richness | PROVEN; PROVEN UNDER richness | 5.1 |
| T2a | Tiny-PFN: only y* is open between sources | PROVEN | 5.1 |
| T2b | Inverse-Wishart errors: exact message across a trader cut = that side's whole error history; k known-loading factors: k T | PROVEN UNDER the Gaussian model | 5.1 |
| T3 | Fixed-dimension sufficient statistics for all N only in exponential families | PROVEN UNDER regularity (Koopman-Pitman-Darmois) | 5.2 |
| T3a | Analytic families: r = O(log(N/delta)^d) additive numbers per source for KL <= delta | PROVEN UNDER analyticity | 5.2 |
| T4 | Two rounds (broadcast pilot, local degree-p statistics) reach vanishing error with C(D+p,p) numbers | PROVEN UNDER LAN, smoothness, moment-identifiable pilot (one-step estimator; cited) | 5.3 |
| T4a | One-round size grows exponentially in the number of open latents for location-type families | HEURISTIC; MEASURED (N2) | 5.3 |
| T5 | Keyed aggregation: Omega(N) bits at a node of every fixed hierarchy; O(1) + log N label bits routed | PROVEN | 5.4 |
| T6 | Attention routing cost = sign-rank of the routing relation; equality 3, order 2, Hadamard >= sqrt(N) | PROVEN | 5.4 |
| T6a | Sharp selection needs logit range of order log N: the SETH-hard regime of attention; label routing is O(N log N) | PROVEN (with Alman-Song) | 5.4 |
| T6b | Per query, softmax attention is an associative merge with d + 2 numbers | PROVEN | 5.4 |
| T7 | Correlation-based grouping needs Omega(min(T, 1/rho^2)) bits per item | PROVEN (promise problem); HEURISTIC (iid histories) | 5.4 |
| T8 | Multi-hop needs rounds (pointer chasing); transformers: log depth for k-hop | PROVEN (cited) | 5.5 |
| T9 | Parity: untangled under every definition, SQ-hard to learn | PROVEN (cited); MEASURED (World 3) | 5.5 |
| T10 | GF(2) pairwise functions: worst message of the best hierarchy = rank-width (+1); density is not tangledness | PROVEN | 5.6 |
| T10a | ReLU gates couple through Gaussian pre-activations, so the non-Gaussian part is the gate-pattern entropy | HEURISTIC | 5.6 |
| T11a | Exact output law cost = product of computation and input-law bond dimensions along a shared hierarchy | PROVEN | 5.7 |
| T11b | Misaligned random trees: union treewidth grows with N | MEASURED (N3a); upper-bound theory cited | 5.7 |
| T11c | Rank-2 functions can have #P-complete tails | PROVEN | 5.7 |
| T11d | Noisy bounded messages: dimension >= log2((1 + N I)/(1 + 2 delta)) / log2(1 + R^2/sigma^2) | PROVEN UNDER the Gaussian setting stated | 5.7 |
| T12 | Latent-tree data: block MI is O(log L); correlations decay at least as a power law; KS threshold | PROVEN | 5.8 |
| A1 | Softmax attention with a sink = merge of Gaussian messages; sink weight = tau_0 x posterior variance | PROVEN (identity) | 6.1 |
| A2 | Transformers implement R-round SFU computations in O(R) layers | PROVEN (construction sketch) | 6.2 |
| A3 | Flat-token binding stalls at a saddle under uniform attention | HEURISTIC | 6.2 |
| C1 | Nets closer to Bayes combine sources more additively; one-layer nets are tangled across sources | MEASURED (40 clean tiny-PFN nets, N5); untested on noisy phase-B nets | 6.4 |
| C2 | Each departure from Bayes is the error of the merge family at some scale | CONJECTURE | 7 |

## Files

All under `/home/agent/workspaces/mats/work/scale-free-theory/` unless stated.
- `QUESTIONS.md` (definitions and checks), `CITATIONS.md` (29 references checked; arXiv abstracts in `cites/`).
- N1c one round vs two rounds, tiny-PFN world: `n1c_rounds.py`, `n1c_rounds.json`, `.log`.
- N1d small codes: `n1d_small_messages.py`, `.json`.
- N2 open latents: `n2_open_latents.py`, `.json`, `.log`.
- N3 tanglegrams and GF(2) classes: `n3_tanglegram_gf2.py`, `.json`, `.log`.
- N5 interaction profile on trained nets: `n5_interaction.py`, `n5_interaction.json`, `.log`.
- Reviews: `review-gpt6-*.md`, `review-gemini-*.md`.
- Earlier, superseded numerics: `n1_pfn_rank.py` (uniform-weight SVD basis; slow decay), `n1b_pfn_weighted.py`
  (per-source weighted fits; unstable for small N).
