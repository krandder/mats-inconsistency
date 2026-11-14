# Co-prover provenance log (cut-rank lane)

Who proposed which step, and what survived the cross-checks. "Claude" = opus-mats-cutrank (Claude Opus 5.5);
"astra" = gpt-6-astra via codex-any (read-only sandbox, xhigh effort).

## Round 1 (23:16-23:32Z)
- Claude: FORMAL.md (fixed one-way statement, Lemma B harmonic structure, Lemma C cap toy, Heuristic H that the rank
  is n^D(eps) rather than exp(k*) at fixed eps). Astra got only the lemma-free copy coprover/STATEMENT_r1.md; the prompt
  did hint at two directions to examine ("fixed eps as n grows with L' fixed", "a symmetry of the back ensemble").
- Astra (coprover/astra_r1_answer.md):
  - A1 (3a) rotation-invariant harmonic decomposition: same as Claude's Lemma B, found independently (with the hint).
  - A2 (3b) averaged gradient identity E_b ||Dg_b(h) v||^2 = ||v||^2, hence sum_l l(l+n-2) a_l = n - 1 and
    sum_{l>m} a_l <= (n-1)/((m+1)(m+n-1)): NEW (astra).
  - A3 (3c, section 1) truncation protocol: r(eps) <= S(n, m) <= C(n+m, m), m = ceil(1/(eps^2 Vbar)) at every finite
    width and depth; refutes (T1) at fixed depth and eps as n grows, and along a diagonal sequence: NEW (astra).
    This is the rigorous form of Claude's Heuristic H (upper-bound side).
  - A4 (section 2) exponential lower bound, simplified setting: ray fronts sqrt(k) R U with U uniform on the positive
    orthant, one Gaussian ReLU back layer: r_+(eps) >= exp((log 2 - o(1)) k) for eps^2 <= C k^{-3/2}. Ingredients:
    closed-form Funk-Hecke eigenvalues of (u.v)_+, Stirling on the shell k <= l <= 2k, orthant-sign symmetrisation
    r_full <= 2^k r_+: NEW (astra).
  - A5 (section 4) block lemma Err >= tr H_B - s ||H_B||op: same idea as Claude's restriction argument (unpublished
    draft), astra's statement adopted.
  - A6 scale-only fronts give rank one whatever the back's Jacobian (the crux does not follow from k* alone): NEW.
  - A7 (3d) one-layer parity obstruction (odd degrees >= 3 invisible): NEW.
  - A8 (section 5, (C)) rank-one collapse for long backs in the iterated infinite-width limit:
    Err_1/Vbar -> (q_{c+d} - q_{d+1})/(1 - q_{c+d}) ~ 2(c-1)/d: NEW.
  - A9 (section 6) quantifiers: ensemble bound gives a positive fraction (Paley-Zygmund), "almost always" needs a
    concentration or small-ball statement; "on most nets every protocol fails" is impossible: NEW (agrees with
    Claude's plan in FORMAL.md T3).
  - A10 conjecture 7b (bottleneck: retained, spread energy in a harmonic block after front averaging).
- Claude's checks of astra's round-1 lemmas (logs/check_astra_r1.log, scratch/check_astra_r1.py, 23:40Z):
  - A2 identity: Monte Carlo over 4000 backs at (n, d) = (8,4), (16,8), (32,16): 0.968 +- 0.036, 1.031 +- 0.044,
    0.967 +- 0.046 (expected 1). Proof re-derived by hand (sign symmetry per layer, induction): CONFIRMED.
  - A4 closed-form eta_l: quadrature agreement to 2-5e-8 for k = 3, 5, 8, l <= 12, odd l >= 3 zero: CONFIRMED.
    Orthant symmetrisation argument re-derived: CONFIRMED.
  - A3 truncation bound: re-derived (Jensen + rotation invariance + He isometry E|h_c|^2 = n): CONFIRMED.
  - A8 formula: kernel recursion gives Err_1/Vbar = 0.197, 0.027, 0.029, 0.0015 at (c, d) = (8, 64), (8, 512),
    (16, 1024), (4, 4000) against 2(c-1)/d = 0.219, 0.027, 0.029, 0.0015: CONFIRMED (asymptotic).

## Round 2 (23:37-23:50Z): lemma exchange
- Astra checked Claude's Lemma C (coprover/astra_r2_answer.md section 1): the spectrum (mu_d b_d^2, multiplicity N(k,d))
  is CORRECT; the displayed rank lower bound in FORMAL.md is WRONG as written (it counts the null odd degrees and the
  whole boundary degree; corrected exact formula r = N_{<D} + ceil((T_{>=D} - delta)/e_D)); my D ~ sqrt(2 ln(1/eps))/s is
  only a leading-log approximation (correct regimes: D ~ (eps^2 s^2)^{-1/3} for s << eps^2; heat-damped otherwise); two
  toy entries corrected by quadrature error (713,593 and 46,983,832 at s = 0.03, k = 8, 12). Fixed-s, fixed-eps vMF
  fronts give NO exponential in k as k grows (they approach uniform). ACCEPTED.
- New from astra, round 2:
  - B1 synchronised-front lower bound r >= [beta T_B - eps^2 Vbar]_+ / (beta rho 2^n lambda*_B) with the
    squared-radial-mean-weighted direction law, density bound rho on the orthant, face-by-face for real ReLU fronts,
    and an energy-weighted L2 certificate for approximate synchronisation: NEW (astra).
  - B2 finite-width depth recursion kappa_{n,d} = kappa_{n,d-1}/2 + R_{n,d} (R PSD, rotation invariant), hence
    a_l^{(d)} >= 2^{-(d-1)} a_l^{(1)} for every l at every finite width: NEW (astra). Claude re-derived it (relu(z) =
    (z+|z|)/2, the cross terms vanish under w -> -w, R_d = E[X(h)X(h')]/4 with X = |w.g_{d-1}|): CONFIRMED.
  - B3 gate-jump identity E ||[J_g]||_F^2 = ||w_i||^2 (kinks persist with depth): NEW (astra). Claude re-derived
    (jump = Dg_rest(v) e_i w_i^T and the averaged gradient identity for the rest): CONFIRMED.
  - B4 combined theorem for synchronised fronts and multilayer finite-width backs: r >= exp[(ln 2) k - (d-1) ln 2 -
    ln rho - O(log k)] at eps^2 Vbar <= c' beta 2^{-(d-1)} k^{-3/2}; for exact synchronised ReLU fronts (faces, |S| ~ n/2):
    log r >= (ln 2 / 2) n - (d-1) ln 2 - O(sqrt n): NEW (astra).
  - Astra rejected Claude's attenuation objection to its conjecture 7b (total vs per-direction radius): ACCEPTED.
  - Astra: n^D(eps) is proven only as an upper bound; uniform orthant rays are a modified ensemble; front
    synchronisation alone gives no collapse theorem. ACCEPTED.
- Claude, new (after round 2): Theorem E (exact, one back layer, ray fronts uniform on S^{n-1}, fixed n):
  r(eps) = Theta_n((eps^2 vR)^{-(n-1)/3}); exact computation (scratch/ray_eps_scaling.py, logs/ray_eps_scaling.log)
  gives local exponents 1.33, 1.99, 2.66, 3.33, 4.66, 5.99, 7.31, 9.96 at n = 3, 4, 5, 6, 8, 10, 12, 16 against
  2(n-1)/3. And Theorem E' (multilayer finite width, from B2 + astra's gradient bound + block lemma):
  r(eps) >= c_{n,d} eps^{-2(n-3)/3}, faster than sampling's 1/eps^2 for n >= 7. Sent to astra for checking (round 3).

## Round 3 (23:58-00:12Z): astra checks Theorems E and E', ledger
- Theorem E (Claude): PROVEN per astra after correcting the harmonic count (count through even degree D is
  n + C(n+D-1, n-1) ~ D^{n-1}/(n-1)!, not twice that); sharper r ~ [1/(n-1)!][C_n/(3 eps^2 vR)]^{(n-1)/3}. ACCEPTED.
- Theorem E' (Claude): astra found a dropped factor q = 2^{-(d-1)} in step (iii) (my displayed bound was not what the
  substitution gives). It then proved a STRONGER statement: K_d - q K_1 is PSD, so G_d >= q G_1 and
  Err_{d,opt}(s) >= q Err_{1,opt}(s) for every s and ANY front laws; for uniform rays r_d(eps) >= c_n [q beta/(eps^2
  Vbar)]^{(n-1)/3}. Plus a matching upper bound T_d(D) = O_{n,d}(D^{-3}) for fixed finite architectures (polyhedral
  cones, crossing probability of a rotated geodesic segment, second differences). So r_d(eps) = Theta_{n,d}(eps^{-2(n-1)/3})
  for ray fronts at every fixed finite depth; n >= 5 suffices to outgrow 1/eps^2. NEW (astra), built on Claude's E and
  astra's B2. Claude re-derived the PSD ordering and Weyl step: CONFIRMED. The polyhedral upper bound: checked at the
  level of the argument (not line by line).
- Astra: the gradient-identity upper bound has precision exponent n - 1 (m ~ sqrt(beta/delta)), not 2(n-1) as Claude
  wrote. ACCEPTED.
- Astra, new: if the radially weighted front laws have densities in L^2 of a common angular measure with
  E_f ||q_f||^2 < infinity, the cut operator is a product of two Hilbert-Schmidt operators, so its singular values are
  summable and r(eps) = o(eps^{-2}): smooth front ensembles let linear summaries beat sampling's count. Gaussian ReLU
  fronts violate the condition (one-active-unit cones have positive mass and produce ray atoms). NEW (astra); Claude
  checked the trace-class step: CONFIRMED.
- Astra on the data: the depth series 117/89/60 is at F = 512 (86/67/48 at F' = 256); full-F in-sample secant exponents
  between eps = 0.1 and 0.03 are 0.46-0.94; the control tails are negative, so "3x the control tail" must be read as a
  magnitude. ACCEPTED (report corrected).
- Ledger 4a-4f (astra): 4a front averaging keeps dispersed energy = single bottleneck; 4b approximate synchronisation;
  4c depth-uniform constants; 4d fixed back with rich readouts; 4e local slices; 4f small-ball for almost every net.
  Merged with Claude's ledger in the report.
- Claude, new measurement after round 3 (logs/kink_depth_n16.log): at n = 16 the finite-width kink coefficient grows
  with depth (relative deficit/theta at theta = 0.1: 0.23, 0.38, 0.75, 1.17, 2.3, 4.8 for d = 1, 2, 4, 8, 16, 32;
  infinite-width prediction 0.21 d). Evidence for conjecture 4c (no 2^{-d} decay of kinks).

## Final review (fresh GPT-6 session, 00:20-00:28Z; review/review_gpt6_final.md)
- 19 findings (2 critical, 14 major, 3 minor); verdict "substantial revision". All applied in the report (section 12).
- New theorem from the reviewer (finding 5): rare-event reduction for GENUINE Gaussian fronts (c >= 3): a layer with one
  entrywise-positive row and all other rows entrywise negative synchronises every input onto e_1; a positive first
  column of the next layer makes the cut a ray uniform on the positive orthant, independent of its norm; restricting the
  error to this event gives r(eps) = Omega_{n,c,d}(eps^{-2(n-1)/3}), matching the cone upper bound. Claude re-derived each
  step (rows acting on nonnegative h_{c-2}; conditional half-normal column has orthant-uniform direction independent of
  its norm; restriction inequality Err_s(T) >= p Err_s(T|E)): CONFIRMED.
- Critical finding 1 (energy undersampling in deep configurations) confirmed by Claude's diagnostics (sample E a^2
  8e-12 to 2e-3 in the five synchronised runs; 1-16 energy-bearing nets) and answered with a post-hoc scale-normalised
  analysis (scratch/scale_norm.py, logs/scale_norm.log): no collapse after normalisation.
