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
   about 64 the consensus is too vague for local messages). With 3 open hidden variables, one round needs thousands
   of numbers. The mechanism is the one-step estimator of statistics, and transformers are known to run such
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
   becomes "few, label-determined routing classes per layer, with simple content". First measurements on our nets:
   in 2-layer tiny-PFN nets, layer 1 routes by labels (labels explain about 77% of the pair-specific attention) and
   layer 2 weights each source by its log precision (median r = 0.87); one-layer nets cannot (r about 0.18). Every
   net's departure from Bayes, in 52 nets including the 12 phase-B nets, is about a third to a half cross-source
   interaction; the raw interaction mostly tracks the size of the error, which the reviewers predicted, so it is not
   by itself evidence of tangling.
8. **Major advance and shortest path.** Either show that trained transformers do in-context Bayes as this kind of
   computation, with each departure from Bayes predicted as the error of the architecture's merge rule at some scale
   (the big-miss departure already fits), or build an untangled architecture at transformer loss with exact tails.
   Next: routing substitutions with permuted tokens, an internal bottleneck, and the sink test on the existing nets
   (hours, CPU), then the tree experiment specified in 6.5. No decision needed.
