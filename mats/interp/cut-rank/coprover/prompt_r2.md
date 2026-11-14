You are gpt-6-astra, co-prover on the cut-rank problem (round 2: lemma exchange). Read-only sandbox; do not modify
files. Files in this directory:
- coprover/astra_r1_answer.md: your round-1 answer.
- FORMAL.md: my (Claude's) own statement and lemmas, written before I saw your answer (Lemma B = your 3a; Lemma C =
  a cap toy; Heuristic H = the n^D(eps) picture that your section 1 now proves as an upper bound).
- coprover/PROVENANCE.md: who proposed what, and my checks of your lemmas (all of A2, A3, A4, A8 confirmed, including
  Monte Carlo of the averaged gradient identity and quadrature of your eta_l).
- logs/toy_cap.log and scratch/toy_cap.py: my Lemma C toy computed exactly (one ReLU layer back, von Mises-Fisher
  fronts of per-direction angular spread s with uniform centres on S^{k-1}; eigenvalue mu_d lambda_d^2 with
  multiplicity N(k,d), lambda_d = I_{d+nu}(K)/I_nu(K), K = 1/s^2). Example (eps = 0.1): s = 0.03 gives r = 22, 217,
  1622, 46479, 713598, 4.7e7 for k = 2, 3, 4, 6, 8, 12; s = 0.1 gives 11, 39, 112, 462, 1527, 7114.
- runs/analysis_cut_n8_c4_L4.json: first measured cut matrix (n = 8, c = 4, L' = 4; 512 fronts x 512 columns; 8 input
  splits per front, pair-averaged Gram so Monte Carlo noise does not bias the spectrum). eps-ranks (in-sample /
  cross-fitted) at eps = 0.3, 0.1, 0.03, 0.01: 6/6, 27/28, 84/91, 150/216. Rank on the first F' fronts at eps = 0.1:
  18, 22, 25, 27 for F' = 64, 128, 256, 512; at eps = 0.01: 52, 86, 119, 150 (still growing). E M^2 / Vbar = 1.4.

Please do, in numbered sections:
1. Check my Lemma C and the toy numbers (formulae, normalisation, the claim that isotropic centres make the operator
   diagonal). If right, combine it with your section 2: what is the exponential rate in k for fronts that are caps of
   per-direction angular spread s rather than rays, and at what eps? (I expect exp(k ln(1 + D/k)) with D ~ sqrt(2 ln
   (1/eps))/s, but check.)
2. The synchronisation asymmetry. Your (C) shows collapse for a long back (d -> infinity after n -> infinity), while
   your section 2 theorem is for ray fronts, which is what a front looks like after it synchronises (at finite width,
   c >> n ln(1/tol), the cut law is approximately h = |x| G_f(x/|x|) u_f: one direction per front, a radial factor).
   State and prove, as far as you can, a lower bound for synchronised fronts with a general multi-layer finite-width
   back: r(eps) >= [tail energy of kappa_{n,d} above degree l0 (after the radial factor) - eps^2 Vbar] / [rho x
   max_{l >= l0} lambda_l], where rho bounds the density of the law of u_f relative to the uniform law on the positive
   orthant, combined with your orthant symmetrisation. Make every hypothesis explicit (radial factor independent of
   the direction or not; how the 2^n orthant loss and the density bound enter).
3. A lower bound on the high-degree energy of kappa_{n,d} at finite width, to pair with your upper bound
   sum_{l>m} a_l <= (n-1)/((m+1)(m+n-1)). For one layer it is explicit. For d layers: is there a curvature or
   gate-flip identity (for example via E over backs of the jump of the Jacobian across a gate hyperplane) that gives
   a_l >= c(n, d) l^{-p} or sum_{l > m} a_l >= c / m^q? Or a counterexample showing that deep critical backs can be
   smooth at the relevant scale? State what you can prove and the weakest conjecture for the rest.
4. Your conjecture 7b: front averaging attenuates degree l by roughly exp(-l^2 r_f^2 / 2) for a front law of angular
   radius r_f ~ 1/c (kernel limit), so a block at degrees ~ d^2 with c = d seems to lose its energy. Do you agree? Is the
   right bottleneck conjecture the front-synchronised regime of item 2 instead? Restate the single bottleneck.
5. Which of your round-1 claims do you now consider wrong or in need of qualification after seeing FORMAL.md?
Be rigorous and concise; mark PROVEN / HEURISTIC / CONJECTURE. Write to stdout.
