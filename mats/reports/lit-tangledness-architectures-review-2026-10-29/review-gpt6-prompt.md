You are an adversarial fact-checker for a short literature report. Read
outbox/lit-tangledness-architectures-2026-10-29.md (relative to the working directory).

Check, claim by claim:
1. Every statement about a cited paper: numbers, what the paper shows, and whether the label is right
   (P = proven theorem, M = measured/empirical, C = conjecture or informal argument). Flag anything wrong,
   overstated, misattributed, or mislabelled. If you cannot verify a claim from your own knowledge, say
   "cannot verify" rather than guessing.
2. The reasoning claims in the answer summary, in particular:
   - "dense linear-Gaussian dependence is cheap / tractable at any treewidth";
   - "compositional sparsity is bounded fan-in, which every Boolean circuit has, so it bounds neither treewidth nor
     estimation cost";
   - "random graphs with over 1.073 edges per node have treewidth linear in size";
   - "the exponential is in input dimension, not depth" (Hanin & Rolnick);
   - "low treewidth is provably the only structural route to tractable inference";
   - "on tree-generated data, dense nets that learn features find the tree's exact inference from polynomially many
     samples";
   - the proposed formal definition of "scale-free" (hierarchy T, message dimension k(s), epsilon-rank across cuts).
3. The proposal's claims about the tiny-PFN world: check them against work/tiny-pfn/world.py (sources independent
   given the answer; each source enters exact Bayes through its query reading and the sum of squares of its labelled
   errors; Bayes multiplies per-source likelihoods of y*). Also check whether the proposed deductive estimation
   (condition on y* and the variance levels, then the root's law is a convolution) is sound, and whether the
   "for/against" criteria can actually discriminate.

Output: a numbered list of findings, each with severity (high / medium / low), the exact text at issue, and the
correction you propose. Do not rewrite the report and do not edit any file. Be concise.
