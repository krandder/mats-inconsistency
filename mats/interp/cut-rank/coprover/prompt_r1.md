You are gpt-6-astra acting as an independent co-prover (not a reviewer) on a mathematics question. Read
coprover/STATEMENT_r1.md in this directory (read-only sandbox; do not modify files). It states a setting (critical
random bias-free He-ReLU networks, a cut at layer c into independent random front and back weights), a protocol class
(one-way linear summaries: the estimate of the mean output is <phi(front), psi(back)>), the eps-rank r(eps) of the
"cut operator" under the weight ensemble, and a target statement (T1): in a depth window, r(eps) >= exp(c0 k*), where
k* is the per-path count of back-Jacobian singular values within a factor delta of the top. It also states (T2) a
predicted collapse past synchronisation, (T3) a fixed-network quantifier, and a crux lemma.

Work independently. I want your own proof attempt and strategy, not a review of mine (you have not seen mine).

Deliver, in this order, in plain text with numbered sections:
1. Your reading of (T1): is it plausibly true as stated? If you think it is false in some regime (for example at fixed
   eps as n grows with L' fixed, or because of a symmetry of the back ensemble), give the argument and the corrected
   statement you would try to prove instead. Be concrete about what quantity should replace k*, if any.
2. A proof strategy for the strongest version you believe is provable, in the simplest setting where an exponential
   lower bound on r(eps) is real (you may simplify: one or few back layers, infinite width at fixed depth, fronts with a
   simple law, etc., but say exactly what you assume). Give the full chain of lemmas, with proofs for the steps you can
   prove, and mark each step PROVEN / HEURISTIC / CONJECTURE.
3. Useful exact structure you see in the problem (symmetries, kernel representations, spectral identities), stated as
   lemmas with proofs.
4. The crux lemma (the back's value function restricted to the k* relevant directions has >= exp(c k*) independent
   components with respect to the fronts' distribution): a precise formalisation that is true and implies a lower bound
   on r(eps); or an argument that it is false or unprovable as stated.
5. (T2) past synchronisation: does r(eps) collapse? Argue either way.
6. (T3): what the ensemble bound gives for a typical fixed network, per protocol; what more is needed for "almost
   always".
7. A conjecture ledger: for each step you cannot prove, the weakest precise conjecture that would close it, whether it
   is a known open problem (with references you are confident of; say "unsure" otherwise), and the cheapest numerical
   experiment that could refute it. Mark the single bottleneck step.
8. Literature you are confident is relevant (approximate rank, Kolmogorov/linear widths, dot-product kernels on spheres,
   random-feature and kernel spectra, arc-cosine kernels, information-based complexity of integration, low-degree or
   overlap-gap methods), each with one line on what it gives here. Do not invent references; mark uncertain ones.

Be rigorous and concise. Prefer a correct weaker theorem to an incorrect strong one. Write everything to stdout.
