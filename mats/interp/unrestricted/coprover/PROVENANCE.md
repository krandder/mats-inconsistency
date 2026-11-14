# Co-prover provenance (unrestricted-estimators lane)

"Claude" = this lane (Claude Opus 5.5); "astra" = gpt-6-astra via codex-any (read-only sandbox, xhigh effort).

## Round 1 (astra 11:01-11:22Z; Claude's FORMAL.md v1 written 10:50-11:16Z)
- Astra got only the questions (coprover/prompt_r1.md), none of Claude's statements.
- Claude's v1 is frozen as coprover/FORMAL_v1_as_written.md, sha256
  7f4acfb163a4c80f6899df683d66043fd9805b04797c565070ca17a889624286 (its header said 11:35Z by mistake; the file
  time is 11:16:09Z).
- Disclosure: at about 11:12Z, before finishing v1, Claude read astra's last three progress lines in the raw log
  (fast matrix multiplication beats serial MC; zero beats MC for tails with p << 1/N; polynomially many
  first-layer cells cover exponentially little mass; low-degree weight polynomials have an exact Hermite-projection
  bound; the final-row linear component must recover the penultimate mean). So v1's 3.4 (fast matmul baseline) is
  astra's, and v1's explicit degree-D tail bound was written after seeing astra's line (the noise-stability identity
  S(rho) = sum rho^k w_k was in Claude's plan before astra was launched).

## Round 2 (astra 11:32-11:52Z; prompt coprover/prompt_r2.md; answer coprover/astra_r2_answer.md)
Astra checked v1 line by line and lib/bnb.py, strengthened four items and assessed Kelvin's two analogies.
Accepted corrections (all applied in FORMAL.md v2.1 and the report):
- Prop. 1 (digits as seeds): correct after fixes (fixed-size convergence; transforms as primitives; n = 1; finite-bit
  precision schedule). Chebyshev remark WRONG as written: T_{2^k} gives the arcsine law, and Gaussian weights are not
  in [-1, 1]; narrowed to "unrestricted algebraic degree does not prevent extracting oscillating seeds".
- Consequence 3 (deterministic = all algorithms): needed a simulation argument for arbitrary randomized estimators.
  Claude's repair (checked by hand, 11:58Z): re-randomise the canonical instance, W' = B_1 Q''' with Q''' Haar from
  seed digits and R from disjoint digits; W' has the law of W, R is independent of W', mu(W') = mu(W), so any
  randomized A is simulated at its own average error plus O(n^3) (real model with digit-splitting primitives).
- 3.1: "not provable" is a statement about current techniques (HEURISTIC), not a theorem; relabelled.
- 6.2 partition rate: WRONG exponent. A crossing cell has function error O(r), not O(r^2): coherent MSE O(K^{-4/m}),
  decorrelated MSE O(K^{-1-3/m}), m = n - 1 (advantage 3/(n-1), matching the GP-limit spectral rate). Claude
  re-derived it: accepted.
- Spin glasses: global concavity of (kappa_L'')^{-1/2} refuted at L = 9 by a rigorous interval enclosure (astra);
  Claude's numerics: concave on (0, 1) at L = 8, not at 16; with the external field (Claude's post-hoc computation,
  matching astra's q_0 condition) concave on [qhat, 1) for L <= 8, not for L >= 10. Astra: the actual final-ReLU field
  is not Gaussian, the field term cannot be dropped, xi''(1) = infinity is outside the theorems' smoothness
  assumptions; so any overlap-gap reading is HEURISTIC. Accepted.
- lib/bnb.py: formulas, relaxations and the homogeneous extension are correct in exact arithmetic; the certificate
  does not enclose the t-quadrature error or the omitted tail (q = 4 counterexample), and it certifies the bound
  midpoint, not the point estimate; two FLOP undercounts (|W| Hr computed for all cells but charged for inexact
  ones; four ndtr arrays evaluated, two charged). Accepted; the report calls the intervals "certified up to the
  moment quadrature (relative error <= 1e-9 tested, not enclosed) and rounding", and checks the n = 3 wins under a 2x
  cost increase.
- PREREG interpretation: healthy-net selection conditions the ensemble (stated); P7's rotation-equivalence rationale
  is wrong (scrambling changes geometry; P7 stays an empirical prediction); common-scramble pairs in M1 carry
  integration noise (negligible at 2^14 Sobol points against S1, noted).
New from astra round 2 (checked by Claude where marked):
- 3(a) mean-field noise stability of the mean, S_L(rho) = m_n^2 (E_T K_{rho,L}(T) - 1/pi), and the 2/3 law
  1 - Corr ~ 3.27 delta^{2/3} for L delta^{1/3} -> infinity; linear at fixed depth with derivative ~ L/(4(1-1/pi)).
  CHECKED numerically against the registered M1 measurements: at n = 32 the formula matches to within about 5% at
  L = 4 for every delta, and at L = 16-32 for delta <= 1e-2 after normalising by the measured D(1)
  (scratch output in logs/nstab_mf_compare.log).
- Two-scale Hermite tail bound T_D >= [(1 + rho^{-D}) F(rho) - rho^{-D} F(rho^2)]_+ (checked: the coefficient of w_k is
  (1 - rho^k)(1 - rho^{k-D}) <= 0 for k <= D and <= 1 otherwise).
- 3(b) even-tail identity, the a_2 contamination formula (a_2 = (n-1)/n b_2 + 6(n-1)/(n(n+4)) b_4 + ...), unbiased Hermite-
  score estimators, and the mean-field limit 1/2 for the even-tail fraction.
- 3(c) depth-recursive identity E[B_{r,L}^T mu_L | W_{1:r}] = mu_r and R(A) >= 2^{-(L-r)} E|muhat_r - mu_r|^2.
- 3(e) analogies: agrees with Claude's verdicts; adds that sign-gate nets contract correlations ((2/pi) arcsin c)
  while ReLU nets expand them, and that the black-box 4x result shares the oracle model's limitation.

## Round 3 (astra 12:04-12:32Z; prompt coprover/prompt_r3.md; answer coprover/astra_r3_answer.md)
- Red-team of C0: astra REFUTED it (PROVEN UNDER layerwise batching costs and m(cT) = O(m(T))). A multilevel
  estimator with a reference trajectory (one extra input's normalised path, per-layer norms nu_l, control
  G_r(U) = m_n s_r |h_r(U)| q_L, unbiased dyadic corrections) has risk O(log^2 L / (L N)) at the cost of N passes, while
  V_sph = Theta(1) at L = alpha n (finite-width amplitude variation, A_1 = 5/4 + o(1)): gain Omega(n / log^2 n). Its own
  pilot found no practical win at L = n/4 up to n = 512 (ratios 3.05 to 1.49, improving with n).
- Claude's check (post hoc, scratch/mlmc_check.py, logs/mlmc_check.log, fresh seeds 9100-9103): ideal variance x cost
  against RB-MC, setup excluded. Two-level version (best dyadic prefix): 0.91-1.29 at L = n/4 (n = 16-256), 0.66-1.04 at
  L = n (n = 16-128), 0.33-0.60 at L = 4n (n = 16-64), 0.14 at L = 16n (n = 16). The residual fraction D_r / V_sph falls
  with the prefix depth r as the bound (1) says (for example 1.17, 0.73, 0.33, 0.06, 0.02 at r = 0, 4, 16, 64, 128 for
  n = L = 128). CONFIRMED qualitatively: the gain grows with depth and slowly with width at fixed L/n.
- Partition-certificate lower bound: PROVEN UNDER explicit nonalignment, isolation and capture hypotheses (10)-(12);
  bounded aspect ratio alone gives nothing (axis-aligned hinges and hemispheres are counterexamples). Exact expected
  kink area of the regular kink set: Theta_alpha(n^{3/2} L); mean jump Theta(1) in l2. Not checked numerically.
