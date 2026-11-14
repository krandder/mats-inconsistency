You are gpt-6-astra acting as an independent co-prover (not a reviewer) on a mathematics question. Read
coprover/STATEMENT_r1.md in this directory (read-only sandbox; do not modify files; do not read other files or
directories: work only from that statement and your own knowledge). It states a setting (critical random bias-free
He-ReLU networks cut at layer c into independent random front and back weights), what is already proven about
one-way LINEAR summaries of the front (the eps-rank of the cut operator grows like eps^{-2(n-1)/3} at fixed
architecture, through gate kinks and, for the genuine ensemble, through a rare event on which the cut state is a ray),
and three targets: (1) nonlinear readouts of a front summary under regularity classes (continuous, Lipschitz, bit
messages, ARC-style moment closures, others); (2) bounded back-and-forth protocols; (3) quantifiers and what stays
outside.

Work independently: I want your own proofs and constructions, not a review of mine (you have not seen mine).

Deliver, in this order, in plain text with numbered sections:
1. For each regularity class in Target 1 (and any other class you think is natural), your verdict: does the precision
   scaling (1/eps)^{Theta(n)} survive? Where it survives, a full proof of the lower bound (constants and hypotheses
   explicit). Where it breaks, the explicit construction and the exact step of the linear argument that fails.
2. Target 2: a lower bound on total communication for bounded back-and-forth protocols under each class, or a
   construction that breaks the scaling. Address Kelvin's frame-naming intuition and say whether the forward-
   measurability result (K7) helps, and how.
3. Any exact structure you find (identities, reductions between classes, tight characterisations), as lemmas with
   proofs.
4. Target 3: quantifiers per result (ensemble vs typical fixed net, per protocol) and the depth window; what remains
   outside and why it needs computational lower bounds.
5. A conjecture ledger: for each step you cannot prove, the weakest precise conjecture that would close it, the
   conditional proof, whether it is a known open problem (references you are confident of; say "unsure" otherwise),
   evidence for/against, and the cheapest numerical experiment that could refute it. Mark the single bottleneck.
6. Literature you are confident is relevant (nonlinear/manifold/Bernstein/stable widths, Carl inequalities, metric
   entropy, communication complexity with real-valued or bounded-precision messages, approximate rank, information-
   based complexity), each with one line on what it gives here. Do not invent references; mark uncertain ones.

Be rigorous and concise. Prefer a correct weaker theorem to an incorrect strong one. Write everything to stdout.
