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
   each departure from Bayes is the error of the architecture's merge family at some scale. The first measurements
   fit the first half (6.4: layer 1 routes by labels, layer 2 weights sources by log precision). The big-miss
   departure fits the second half: at the token-to-source scale, Bayes adds the unbounded statistic e^2, while the
   first layer averages normalised (bounded) per-token features, so the Bayes merge lies outside the family (the E20
   toy's "capped per-error evidence"); depth adds a per-token transformation before pooling, which enlarges the
   family (E17, E19). **CONJECTURE** (general form): the departure equals the projection of the Bayes merge onto the
   merge family, scale by scale, and is predictable from the architecture before training. The interaction profile
   adds a constraint any such account must meet: about a third to a half of every net's error is cross-source
   interaction.
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
  one-level routing a transformer can implement them in O(rounds) layers (A2). Tangledness is necessary for
  similarity-based routing (T6a, T7), hidden structure near the detection threshold (T4a, T7), multi-hop lookups
  without depth (T8), and genuinely global couplings (T10).
- Used by trained nets? Partly measurable without opening the net (M1) and with attention maps (M2). In the tiny-PFN
  nets, the computation decomposes as the theory says (label routing, then precision weighting), and every net's
  error is partly cross-source; nets that cannot apply the per-source nonlinearity before mixing carry the expected
  signature. Whether the nets' internal messages are as small as the task allows (M4) is not yet measured.
- Learnable? Separate question. World FA (binding) and World 3 (parity) are cheap to represent and were not learned.
