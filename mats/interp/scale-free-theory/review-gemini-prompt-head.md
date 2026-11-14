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

