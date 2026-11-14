You are gpt-6-astra, co-prover on the unrestricted-estimators lane (round 3). Read-only sandbox; do not modify files.
Context: coprover/astra_r1_answer.md, coprover/astra_r2_answer.md (your answers), FORMAL.md (v2.1, merged theory),
lib/bnb.py (the certified branch-and-bound), coprover/PROVENANCE.md.

Two tasks, numbered, labelling every claim PROVEN / PROVEN UNDER X / HEURISTIC / CONJECTURE:

1. Red-team the bottleneck conjecture C0 (M-whole). Try hard to REFUTE it: propose the most promising estimator
   families that might beat optimally implemented radius-exact MC by a factor growing with n at L = alpha n on random
   critical He-ReLU nets (sigma = 0), using full access to the weights. Consider at least: exploiting the collapse of
   all inputs onto a common direction at depth (angle about 3 pi / l), the shared multiplicative output scale (a
   product of per-layer factors), the exact final-row identity and its depth recursion, low-rank or low-degree
   structure in the first layers, multilevel / control-variate schemes whose control has a computable integral, and
   designs in the input directions. For each, estimate its charged cost and its error scaling with n and L (with the
   reasoning), and say whether it plausibly gives a growing gain, a bounded gain, or none, and why. If any family
   plausibly gives a growing gain, state precisely what would have to be true for it to work and the cheapest
   numerical test.

2. A lower bound for partition certificates. For estimators that certify mu by a partition of the sphere into
   K cells with bounded aspect ratio (axis-aligned gnomonic boxes, as in lib/bnb.py, or convex cells of bounded
   eccentricity), on each of which they produce affine lower and upper bounds of the net, prove the best lower bound
   you can on the certificate width (sum over cells of the integrated bound gap) in terms of K, the dimension n - 1,
   and the expected kink geometry of random critical nets (expected (n-2)-dimensional area of the kink set on the
   sphere and the typical gradient jump across it, as functions of n and L). State the hypotheses exactly. Then say
   what it implies for the number of cells needed for a certificate of width eps sqrt(V) at L = alpha n.

Be concise. Prefer a correct weaker statement to an incorrect strong one. Write everything to stdout.
