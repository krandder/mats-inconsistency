# Co-prover provenance log (cut-rank-ext lane)

"Claude" = opus-mats-cutrank-ext (Claude Opus 5.5); "astra" = gpt-6-astra via codex-any (read-only sandbox, xhigh
effort, nice 19).

## Round 1 (10:48-11:10Z): independent attempts
- Astra got only the lemma-free `coprover/STATEMENT_r1.md` (sha dfa5d3a8...) and was told not to read other files. The
  statement poses the targets neutrally ("does the scaling survive ... or find where it breaks") and names candidate
  tools (DHM widths, Bernstein widths, CDPW, entropy, communication complexity). It contains no new lemma.
- Claude wrote `FORMAL-ext.md` (sha f8e1db9b..., hashed ~11:02Z) after launching astra and before reading its answer.
  Disclosure: at ~11:03Z, after hashing, Claude saw the last line of astra's in-progress log ("finite moments determine
  every fixed-architecture cut law ... Hilbert basis theorem"); nothing in FORMAL-ext.md depends on it.
- Found independently by both (same conclusion, different proofs where noted):
  - weights as an exact summary with a locally Lipschitz decoder (Claude C1, astra 1.a);
  - on ray fronts M = g_b(E h): the mean suffices exactly (Claude C2, astra 1.a);
  - any-round B-bit protocols have separation rank <= 2^B (Claude Theorem A, astra 2.b);
  - degree-D polynomial readouts keep the exponent divided by D (Claude Theorem B, astra 1.e);
  - two rounds, back weights back and M forward, exact (Claude C1/D3, astra 2.a);
  - Lipschitz decoders are bounded below by an intrinsic dimension via covering (Claude C4/C5 with k = n - 1 on the
    ray event, astra 1.b/3.b with a k-cube hypothesis);
  - the bottleneck is computational, not informational (Claude Sec. 7, astra 5.d).
- New from astra (round 1):
  - A1: one bounded scalar message with Lipschitz encoder and decoder (constants depending on eps) reaches any positive
    error (simple-function approximation plus Lipschitz extension of a finite labelling);
  - A2: clipping construction: bounded encoder, decoder Lipschitz constant sqrt(c) R^L with R ~ sqrt(log 1/eps);
  - A3: a fixed Lipschitz bound with a fixed radius forces an error floor regardless of s (unbounded target variation);
    coordinatewise-bounded messages with a fixed Lipschitz constant need only polylog(1/eps) coordinates;
  - A4: stronger bit bound Err >= C_bit 2^{-2B/n}, hence B >= n log2(1/eps) + O(1) for the genuine ensemble (K5 ray
    event times a positive-probability back event on which g_b is linear with sigma_min >= 2^{-d}, then a quantisation
    lemma); upper bounds p log2(1/eps) one-way, (q + n) log2(1/eps) two-way;
  - A5: finite moment determinacy: for fixed (n, c) some (non-effective) moment order Q determines every cut law
    (Gaussian integration by parts on cone faces, Hilbert basis theorem, Gaussian tails); an exact continuous decoder
    of the moments through Q exists (compactness and Tietze); exact layerwise moment propagation;
  - A6: c = 1: moments through order 2 suffice; first moments alone have a positive error floor for n >= 2;
  - A7: e_K^2 = E||F - E[F | Z_K]||^2 is the information limit of order-K moments, attained by continuous readouts;
    moment-restricted interaction cannot beat it (generalises Claude's feature-span Theorem D to nonlinear readouts);
  - A8: multi-round protocols whose messages are degree-<= D polynomials of earlier messages (no branching): separation
    rank <= H_C^S, S = 1 + D + ... + D^t, so C >= C_*^{1/(DS)} eps^{-a/(DS)} - D;
  - A9: bi-Lipschitz ray feature map 2^{-d}||h - h'||^2 <= E_b||g_b(h) - g_b(h')||^2 <= ||h - h'||^2 (cleaner than
    Claude's a_1 argument; gives k = n on the ray-times-scale cube);
  - A10: Borsuk-Ulam widths are uniform-error statements and do not reproduce the covariance tail of K1.
- Not in astra round 1 (Claude only): fixed-radius analytic readouts (B1) and C^k readouts (B2); the exact s = 1
  Peano-curve protocol with a continuous decoder (C3); the atlas-size bound 2^B C(s+p,p) >= r (D2); the streaming
  corollaries (Sec. 5); the Bakhvalov remark on black-box cubature (Sec. 7).
- Claude's checks of astra round 1 (11:12-11:20Z, by hand):
  - A4: re-derived. For each fixed back and seed the output takes <= 2^B values (also for interactive protocols: the
    transcript has <= 2^B leaves); on the event the target is A_b z with z = alpha v of density >= rho on [a, 2a]^n and
    ||A_b(z - z')||_n >= 2^{-d}|z - z'|/sqrt(n); the quantisation lemma (isodiametric volume bound, half the cube
    uncovered at radius t) gives the constant as stated. CONFIRMED.
  - A5: re-derived the face recursion (x_i gamma_F = -(P_F e_i).grad gamma_F on span F; boundary density
    gamma_F = (2 pi)^{-1/2} gamma_G on a facet through 0) and the Hilbert-basis step (equality on a finite generating set
    of the ideal forces equality of all generators). CONFIRMED. The compactness/Tietze continuity argument (3.d):
    checked at the level of the argument, not line by line.
  - A8: rank of a product of D rank-K functions <= K^D; one-party coefficients keep rank. CONFIRMED.
  - A9: p - q = a - b, (a-b)^2/2 <= p^2 + q^2 <= (a-b)^2, E p^2 = E q^2 by sign symmetry. CONFIRMED.
  - A3: the decoder's range over the encoder ball has diameter <= 2 A Lambda while target differences are unbounded.
    CONFIRMED. Consequence for Claude's C5: the Lipschitz lower bound is meaningful only with constants allowed to grow;
    C5 already assumed log(rho gamma) = o(log 1/eps).

## Round 2 (11:16-11:33Z): lemma exchange
- Astra got FORMAL-ext.md (as hashed at 11:02Z; Addendum 1 was appended at 11:22Z while round 2 ran, Addendum 2 at
  11:32Z, so round 2 may have seen Addendum 1 and not Addendum 2), its round-1 answer, PROVENANCE.md, the ARC fact sheet,
  the draft coverage map and the zero-covariance log. Answer: coprover/astra_r2_answer.md.
- Astra's verdicts on Claude's lemmas: CORRECT: Theorem A, Corollary A1 (superseded by A4), Theorem B and its
  multi-round remark (with the threshold counterexample for adaptive backward bits), B2 (as a necessary condition), C2,
  C4, Theorem D, D2. CORRECT AFTER FIX: A2 (two-way count p_b + n, not p_b + 1, for the vector answer), B1 (the displayed
  multiset tail used C(p+s, s) where C(p+s, s-1) is right, and the claimed asymptotic l/log l is WRONG: with
  s = sigma l/log l and p = Theta(l) the binomial is exp(o(l)); the corrected bound is stronger, s = Omega_{a,q}(l)),
  C1 (n reals for the vector answer), C3 (needs a compact source or an unbounded scalar domain; Holder exponent needs
  a suitable Peano curve), C5 (restrict to alpha in [u, 2u]; dimension n with scale), C6, D1 (only within Theorem D's
  class), D3, Sections 5-7 (hypotheses carried over; Bakhvalov needs truncation for Gaussian inputs; stratified
  cubature is itself sampling).
- New from astra (round 2):
  - R2.1: fixed (eps-independent) analytic amplitude or C^k derivative budgets force an error floor at every fixed s
    (Taylor polynomial of rank C(s+k-1, k-1) plus a bounded remainder cannot follow unbounded targets); so the
    meaningful classes have budgets growing subpolynomially (or, for the analytic upper bound, polynomially);
  - R2.2: B1 strengthened to s >= (c(a, q) - o(1)) log(1/eps) via homogeneous Taylor components and one-variable
    Cauchy estimates (dimension-free tail A q^{p+1}/(1-q));
  - R2.3: C^k upper bound s <= (k+1) p_f with polylog derivative budgets (k+1 interleaved Cantor-coordinate streams,
    Whitney extension of a jet with zero positive-order terms). Same mechanism as Claude's Addendum 1 (snowflake via
    Assouad + Whitney); whether found independently is unknown (Addendum 1 was on disk during the run, and astra's
    lemma table has no C7 row);
  - R2.4: Lipschitz lower bound n(n+1)/2 on the genuine ensemble: on the event W_2..W_c near positive invertible
    matrices, h_c = A relu(Z), Z ~ N(0, Sigma), and the ridge means F_Sigma(w) = E relu(w^T A relu(Z)) recover Sigma
    locally bi-Lipschitzly (explicit values at w = A^{-T} e_i and A^{-T}(e_i - e_j)); hence also one-way bits
    >= (n(n+1)/2) log2(1/eps) - O(1);
  - R2.5: ARC's basic closure (Wu et al. eqs. (13)-(14)), fed exact cut cumulants, one back layer, has a positive bias
    floor on the K5 ray event for EVERY fixed K: along A_t = X_+ + t(-X)_+ the closure error is real-analytic in t on
    all of R (variance V_t > 0), and at t = -1 the formal continuation is Gaussian, where the closure is off by
    q/sqrt(2 pi); so the error cannot vanish on any interval of physical t > 0. Quantitatively
    b_K >= c exp(-C K log(K+2)) (two-constants theorem), so the oracle estimator needs K log K >~ log(1/eps), i.e. a
    symmetric state of size Omega((l/log l)^n) at fixed n;
  - R2.6: conjecture I1 as stated (generic rank p - n(n-1)/2 - n(c-1)) is false almost everywhere: dead fronts and the
    positive-layer event (only the product A matters) have positive probability and extra symmetries;
  - R2.7: coverage-map corrections (mean propagation conflated with CNX22's arithmetic-circuit version and with Wu's
    K = 1, which tracks a variance trace; split the covariance-propagation variants; the zero-covariance log tests a
    different input from the true-covariance closure; sketching gives m reals, not O(m) bits without precision
    assumptions; missing rows E1.1, E1.5, E2d-g, E7c, E9c-d, E10, E12c, E14).
- Claude's checks of astra round 2 (11:38-11:46Z, by hand):
  - R2.2: the homogeneous-component Cauchy bound |P_m(z)| <= A q^m is the one-variable Cauchy estimate applied to
    t -> R(t z/q); with s = sigma l / log l the binomial is exp(sigma l log log l / log l) = exp(o(l)): my B1 asymptotic
    was wrong, astra's correction stands. CONFIRMED.
  - R2.4: re-derived both identities (E relu(Z_i^+ - Z_j^+) = (1/2)(E[Z_i^+ - Z_j^+] + E|Z_i^+ - Z_j^+|) and, by sign
    symmetry and |u_+ - v_+| + |(-u)_+ - (-v)_+| = |u - v|, E|Z_i^+ - Z_j^+| = E|Z_i - Z_j|/2). CONFIRMED.
  - R2.5: checked E A_t^r = c_r (1 + t^r) (disjoint supports), V_t > 0 on R (minimum 0.266 at t = 1/(pi - 1)),
    realisability of A_t for t > 0 on the K5 event with c >= 3 (W_1 rows +-e_1, positive bottleneck row (1, t, 0, ...)),
    and exactness of the K >= 2 Gaussian closure at the Gaussian point t = -1. CONFIRMED for the oracle one-step
    statement; the quantitative rate was checked at the level of the argument.
  - R2.1, R2.3: CONFIRMED at the level of the argument (Taylor floor; Whitney with zero jets is the same mechanism as
    Addendum 1).
- Not checked by astra: Claude's Addendum 2 (multilinear decoding of quantised bits as an analytic upper bound
  (p + o(1)) log2(1/eps) with polynomial amplitude). Left to the final review.

## Final review (GPT-6 = gpt-6-astra, fresh session, high effort, 12:00-12:09Z; review/review_gpt6_final.md)
- Verdict: major revision, 15 findings. Passed: the any-round n log2(1/eps) bit bound, the covariance chart and the
  one-way bit bound, the cone-face moment recursion and Hilbert-basis determinacy with continuous decoding, the
  fixed-budget floors, polynomial rank lifting, realisability of alpha_t on the K5 event and analyticity of the closure in
  t, the streaming cut reduction, the cubature remark (with truncation). It recomputed all four NC1 configurations, the
  NC2 slope (-2.115) and the NC3 counts from the raw files and verified both PREREG hashes.
- Corrections to Claude's lemmas from the review: Addendum 1 gives C^{k-1,1}, not C^k (for C^k use astra's Cantor code
  with inverse exponent > k); Addendum 2 is correct on the radius-2 polydisc (ratio 1/2) only, with both clippings, and
  the sharper amplitude sqrt(5)^B; C3 is exact on all weights only with a continuous surjection R -> R^p; D1/D2 need
  non-adaptive naming and output coefficients.
- Data findings: NC2's norm code used an uncounted front scale (fixed with a global code, same numbers); ARC3 lacked the
  fourth-cumulant trace (rerun: still unstable); "exact moments" in the measurements were sample estimates (convergence
  check added: ARC2/K2X errors move by <= 0.002 Vbar from 8,192 to 65,536 directions).
- All 15 findings applied (Amendment 2 for the reruns; report sections 2-8 and 11 revised; notes/coverage_map.md
  revised). A focused re-check was launched at 12:26:59Z; one wording patch (report section 6) was applied at ~12:28Z
  while it ran.
- Post hoc, after round 2: `scratch/arc_bias_alpha_t.py` shows ARC's own code fed exact cumulants of alpha_t v (n = 4,
  one back layer) has relative bias 1-4% at K = 4 and more at lower K, consistent with astra R2.5.
