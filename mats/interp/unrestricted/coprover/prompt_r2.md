You are gpt-6-astra, co-prover on the unrestricted-estimators lane (round 2: lemma exchange). Read-only sandbox; do
not modify files. Files in this directory:
- coprover/prompt_r1.md: the round-1 questions; coprover/astra_r1_answer.md: your round-1 answer.
- coprover/FORMAL_v1_as_written.md: Claude's own statements, written before your answer arrived (Claude saw only
  your last progress lines, disclosed in coprover/PROVENANCE.md).
- lib/bnb.py: Claude's certified branch-and-bound over input cones (gnomonic boxes on the 2n cube faces; 1-D t-quadrature
  for cell moments; forward symbolic linear bounds; homogeneous extension from the slice x_k = s to the cone).
- PREREG.md: the registered experiments (results not yet in; dev results: the B&B beats scrambled-Sobol RQMC only at
  width 3 and large budgets, loses 100-4000x at widths 4-8, and its certificate is vacuous at depth 32).

Please do, in numbered sections, concisely, labelling every claim PROVEN / PROVEN UNDER X / HEURISTIC / CONJECTURE:
1. Check Claude's statements in FORMAL_v1_as_written.md, each with a verdict (CORRECT / CORRECT AFTER FIX / WRONG) and
   the fix: Prop. 1 (digits of weights as a seed: Riemann-Lebesgue, weak convergence, uniform integrability), the
   Chebyshev remark (consequence 2), Prop. 2 and its consequences (including the GP-limit rate N^{-1-3/(n-1)} and the
   IBC remark), 3.1-3.4, 4.1 (the identity, the bound, and the heuristic: the fixed point c* = kappa(rho c*) of the
   mean-field cosine map with correlated weights, 1 - c* ~ (3 pi/(2 sqrt 2))^{2/3} delta^{2/3}, reached after about
   delta^{-1/3} layers, and its transfer to 1 - S(1 - delta)/S(1) ~ C delta^{2/3}, w_k ~ k^{-5/3} up to k ~ L^3),
   4.2-4.3, 5 (one-layer kappa'' = 1/(pi sqrt(1 - q^2)); concavity of (xi'')^{-1/2}; the external-field caveat; what the
   spherical spin-glass theorems actually require), 6.1 (conditional exact cost), 6.2 (partition rate), 7 (C0-C4).
2. Check lib/bnb.py for correctness of the certificate: the homogeneous extension of slice bounds to cones, the
   relaxations, the cell moments, and whether "exact on certified cells" is right. Name any bug.
3. Strengthen where you can:
   (a) the noise-sensitivity exponent: prove or refute, at least in the infinite-width mean-field model, that
       1 - Corr(mu(W), mu(W_rho)) ~ C delta^{2/3} for L >> delta^{-1/3} (or give the correct law), and what it implies
       for the Hermite tail of mu and for degree-D estimators;
   (b) combine your bound (2.5) with kernel values: is sum_{even d >= 4} b_d = (1 + k(-1))/2 - k(0) - b_2 exact, and
       is b_2 = k''(0)/2 measurable from the degree-2 spherical-harmonic content? Give the cheapest unbiased
       estimators of k(-1), k(0), b_1, b_2 from net samples, so that M-design can be tested at n up to 256;
   (c) can the final-row identity (8.4) be iterated to earlier layers, or combined with (8.2), into a stronger
       necessary condition (e.g. depth-recursive), and what would it say about estimators that are cheap per layer?
   (d) the strongest precise statement you can give of "what an unrestricted deterministic estimator would need" to
       beat optimally implemented radius-exact MC by a growing factor at L = alpha n.
3e. Two analogies Kelvin asked about (he wants an analogy that could become a proof of similarity). For each, say
   precisely how close the objects are, what is known (with citations marked unverified if unsure), what a proof of
   similarity would have to show, and whether it is plausible:
   (i) derandomising threshold circuits: a random deep ReLU net as a random depth-L threshold-like circuit, mean
       estimation as approximate counting (CAPP); unconditional PRGs / deterministic approximate counting are known for
       halfspaces, PTFs, intersections of few halfspaces (depth 1), while depth >= 2 TC0 is open and tied to circuit
       lower bounds (IKW, Williams). How close are ReLU nets (real weights, Gaussian input, ARC's SD-normalised error)
       to that class? Is an average-case version for random weights studied? Does the depth-1 positive theory
       transfer to depth 1-2 ReLU nets with Gaussian inputs, and where exactly does it stop?
   (ii) convex-body volume: randomised poly-time approximation (Dyer-Frieze-Kannan) versus the deterministic
       oracle lower bounds (Elekes; Barany-Furedi). Do the gate-region masses of a ReLU net admit a white-box analogue
       (each region is an explicit polyhedral cone; the mean is a sum of their Gaussian first moments)? Is there a
       white-box deterministic lower bound for Gaussian volumes of explicit polytopes (H-polytopes), or is it open?
       How does a black-box "no query algorithm beats MC by more than about 4x" result relate?
4. A merged theorem list (both of us), each PROVEN / PROVEN UNDER X / HEURISTIC, with who proposed it.
5. A merged conjecture ledger (statement, status, evidence, cheapest refutation) and the single bottleneck.
Prefer a correct weaker statement to an incorrect strong one. Write everything to stdout.
