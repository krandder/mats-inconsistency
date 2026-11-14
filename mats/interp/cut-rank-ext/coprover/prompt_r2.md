You are gpt-6-astra, co-prover on the cut-rank extension (round 2: lemma exchange). Read-only sandbox; do not modify
files. Files in this directory:
- coprover/STATEMENT_r1.md: the round-1 statement you worked from.
- coprover/astra_r1_answer.md: your round-1 answer.
- FORMAL-ext.md: Claude's own statement and lemmas, written and hashed before Claude read your answer.
- coprover/PROVENANCE.md: what each of us found, and Claude's checks of your round-1 lemmas (A4 bits, A5 finite moment
  determinacy, A8 multi-round polynomial rank, A9 bi-Lipschitz ray map: confirmed; A5's continuity/Tietze step checked
  only at the level of the argument).
- notes/arc_estimators.md: a sourced fact sheet of ARC's estimators (E1-E16), compiled today from the papers and posts.
- notes/coverage_map.md: Claude's draft classification of ARC's estimators against the merged bounds.
- logs/arc_zero_cov_check.log: ARC's released K = 2 kprop code fed the cut mean with zero covariance agrees with mean
  propagation g_b(mu) to 2e-5 relative (64 front-back pairs, n = 8, c = 64, d = 8, unit-scaled mean; the residual is
  the code's absolute variance floor 1e-10).

Please do, in numbered sections, concisely:
1. Check Claude's lemmas in FORMAL-ext.md line by line, with a verdict for each (CORRECT / CORRECT AFTER FIX / WRONG):
   Theorem A and Corollary A1 (superseded by your A4?), Proposition A2, Theorem B and its multi-round remark,
   Corollary B1 (fixed-radius analytic readouts: Cauchy estimates, the multiset-splitting inequality, the asymptotic
   s >= (a - o(1)) log(1/eps)/log log(1/eps)), Corollary B2 (C^k readouts: Jackson at fixed s and k, conclusion
   s >= k a), C1, C2, C3 (space-filling curve: s = 1 EXACT with a continuous decoder and a measurable encoder; compare
   with your remark that your scalar construction is not exact), C4 and C5 (now with your A9 giving k = n), C6,
   Theorem D, D1, D2, D3, Section 5 (layer-by-layer states), Sections 6-7 (quantifiers; the Bakhvalov remark).
2. Strengthen where you can:
   (a) upper bounds for the fixed-radius analytic class and the C^k class (is s = Theta(n log(1/eps)) achievable for
       analytic readouts, s = Theta(k n) for C^k?);
   (b) the Lipschitz class on the genuine ensemble: prove s*_Lip >= c n^2 (or any bound above n) through a
       positive-probability event on which the back-observed front family contains a bi-Lipschitz image of a cube of
       dimension ~ n^2 (for example your event with W_2..W_c positive and invertible, h_c = A relu(W_1 x)); or explain
       why it fails;
   (c) ARC's specific closure: on the K5 ray event with one back layer, fed the exact cut cumulants through order K,
       does the order-K Hermite/Edgeworth truncation (Wu et al. 2026, Sec. 4.1, eqs. (13)-(14); fact sheet E2a) have a
       bias floor b_K > 0 for each fixed K? Give the best rate in K you can prove or a precise conjecture, and what it
       would imply for ARC's state size n^K as eps -> 0 at fixed n.
3. Check the coverage map (notes/coverage_map.md) against the fact sheet: for each row, is the class right, is the
   stated consequence right, and is anything missing from ARC's list? Correct it where wrong.
4. A merged theorem list (both of us), each item PROVEN / PROVEN UNDER X / CONJECTURE, with who proposed it.
5. A merged conjecture ledger (weakest conjecture per open step, conditional proof, status, evidence, cheapest
   refutation), and the single bottleneck.
Prefer a correct weaker statement to an incorrect strong one. Write everything to stdout.
