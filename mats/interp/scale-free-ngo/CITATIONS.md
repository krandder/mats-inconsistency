# Citations for outbox/scale-free-agency-ngo-2026-10-29.md (checked 2026-10-29)

Read in full (text in sources/):
- R. Ngo, "Towards a scale-free theory of intelligent agency", LessWrong 5tYTKX4pNpiG4vzYg, 2025-03-21, and its 55 comments (LessWrong GraphQL API).
- Ngo, later LessWrong posts: rSaDFQ6JSzhC9w7Hh (well-foundedness, 2025-04-07), MEkafPJfiSFbwCjET (On Goal-Models, 2026-02-02), AnDe5FM5wss2sPuKd (Distributed vs centralized agents, 2026-02-09), NKFtDgSH7iZo8skTK (Towards a Formal Scientific Epistemology, 2026-06-09), M39Z2CvyfaxZdaxR4 (Agents as Webs of Beliefs, 2026-06-27), pYFBD2SnqiWkuNns5 (Explaining Knightianism on one foot, 2026-09-02), DwHvtm8vLipAQPest (Social Theory and Agent Foundations, 2026-10-08).
- S. H. Lee, R. Kondor, R. Ngo, "Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks", arXiv:2509.06701 (v1 2025-09-08, v2 2026-05-12); PDF header: ICML 2026, PMLR 306. Definition 2: log pool, non-negative weights summing to one. Sec. 2: "Goals are thus formalized as distributions over desirable observations".
- M. Ostrovsky, "Information aggregation in dynamic markets with strategic traders", Econometrica 80(6):2595-2647, 2012. Example 1 (X = 1 on {A, D}, -1 on {B, C}; partitions {AB, CD}, {AC, BD}): "they will never learn the true value of the security". Definition 2 (non-separable).
- F. Dietrich, C. List, "Probabilistic opinion pooling", Oxford Handbook of Probability and Philosophy (Hajek, Hitchcock eds), OUP 2016; preprint MPRA 54806. Theorem 3: "The multiplicative pooling functions are the only individualwise Bayesian pooling functions (with domain P0^n)." Sec. 8: multiplicative pooling for private information with fixed background information.

Checked by abstract or publisher record:
- Y. Chen, S. Dimitrov, R. Sami, D. Reeves, D. Pennock, R. Hanson, L. Fortnow, R. Gonen, "Gaming prediction markets: equilibrium strategies with a market maker", Algorithmica 58(4):930-969 (online 2009, issue 2010), doi 10.1007/s00453-009-9323-2 (abstract via Harvard DASH). Finite-stage LMSR: conditionally independent signals -> truthful betting is a PBE and the unique weak PBE; unconditionally independent -> truthful betting is not a weak PBE. Ostrovsky (2012, sec. 1.2) describes the model as a security with two values.
- A. Beygelzimer, J. Langford, D. Pennock, "Learning performance of prediction markets with Kelly bettors", AAMAS 2012, arXiv:1201.6655: price = wealth-weighted average belief; market reacts as if updating by Bayes' law.
- A. Storkey, "Machine learning markets", AISTATS 2011 (PMLR 15:716-724), arXiv:1106.4509: markets implement product- and mixture-of-experts and "message passing methods".
- K. B. Laskey, W. Sun, R. Hanson, C. Twardy, S. Matsumoto, B. Goldfedder, "Graphical model market maker for combinatorial prediction markets", JAIR 63:421-460, 2018 (junction trees).
- R. Hanson, "Logarithmic market scoring rules for modular combinatorial information aggregation", Journal of Prediction Markets 1(1):3-15, 2007 (record only; the LMSR identity in the note is our algebra, re-derived by GPT-6).
- M. Wainwright, M. Jordan, "Graphical models, exponential families, and variational inference", FnTML 1(1-2):1-305, 2008: local consistency is necessary and sufficient for global consistency on a junction tree; strict outer bound with cycles (sec. 2.5, 4.1).

Not cited in the final note: Lee 1993 / Bikhchandani-Hirshleifer-Welch on cascades (dropped, GPT-6 finding 14); Wainwright and Jordan 2008 (the local/global consistency conjecture was cut for length).
