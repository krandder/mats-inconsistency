You are a co-prover and red team (probability, random dynamical systems, random matrices), read-only. This is round 2.
You may now read /home/agent/workspaces/mats/work/stationary/ (the other prover's work). Label every claim PROVEN,
PROVEN UNDER <hypothesis>, HEURISTIC or CONJECTURE; flag uncertain citations "unverified". Plain markdown, ASCII math.
Do not edit files. Be concrete and short where you can; compute constants when possible.

Files:
- FORMAL.md: Claude's theory v1, frozen 13:15Z before your round-1 answer (coprover/astra_r1_answer.md is your answer).
- PREREG.md (Part A predictions, hashed 13:22Z) and results/partA_tables.md, results/partA_scores.json (evaluation
  results, seeds 1000-1003, n = 8..256, Model S with sigma = 1 unless stated; Model R sub-study at the end).
- scratch/partA.py, scratch/radius_law.py, scratch/partR.py, lib/st.py (the code that produced them).
- runs/cloudhist_*.json may appear while you work: a post-hoc test of a radius-history model (see below).

Tasks.
1. Audit. Compare FORMAL v1 with your round-1 answer and with the data. List every statement of v1 that is wrong,
   overstated or mislabelled, with the fix. (Known to me: v1 item 4's lambda_1 "along noisy trajectories" ignores your
   P7; v1's uniqueness is weaker than your P1; v1's Lemma 2 proof is a sketch where your P2 is a proof; v1's birth
   definition is resampling, yours deletion.) Check the PREREG scoring in results/partA_scores.json against PREREG.md:
   is anything scored wrongly or generously?
2. The cloud. Measured: covariance eigenvalues of h_0 over the noise at fixed weights have log-log slope near -1 over
   k = 2..n/8, but are 1.2x (n = 8) to 5-8x (n >= 64) the coordinator's sigma^2/(1 - exp(-2(k+1/4)/n)), with the ratio
   larger on layers with larger radius X; the radial (k = 1) eigenvalue is 14-160x. Claude's hypothesis (post hoc,
   being tested in runs/cloudhist_*.json): relative angular deviations decay at the gaps gamma_1 - gamma_k and receive
   relative noise sigma^2 / R_{-j}^2 when injected, so the absolute angular variance along k is
   R_0^2 sum_j sigma^2 exp(-2 gap_k j) / R_{-j}^2: it depends on the radius history, and with a constant radius it is
   sigma^2/(2 gap_k) = sigma^2 n/(2(k-1)), not sigma^2/(2|gamma_k|). Is this right? Derive what you can: the stationary
   angular covariance and the quenched radial variance (measured within-net CV of |h|: about 0.35 at n = 32-64, 0.25 at
   128, 0.2 at 256), as functions of n.
3. Interactions. The summed total-effect shares (Sobol total indices by noise age, for h_0) are 1.9-2.2 at n = 16,
   2.5-2.6 at 64, 3.0-3.1 at 128, 3.6-4.3 at 256 (an additive state would give 1), and ages >= n carry 4-52% of the
   sum. Explain this (e.g. the direction-dependent gain cascade multiplying older contributions) and predict its
   growth with n. What does it imply for estimators that share the deep past, and for a "live-bit budget"?
4. Model R (relative noise, fixed input): births match the Gaussian-unit formula; the angular cloud is not
   proportional to sigma^2 (ratio 13-133 between sigma = 1e-2 and 1e-3) and its spectrum is steep at small sigma; the
   spread of log|h| over the noise at fixed input grows with depth even at sigma = 1e-3 (0.01-0.04 at L = 4n, 0.08-0.33
   at L = 16n). Explain.
5. Part B design (estimators of E[a], a = relu(W_1 h_0), and P(|a| > t) at fixed weights under Model S's stationary
   measure; cost in matrix-vector products). Dev facts at n = 16: burn-in MLMC has variance x cost 156 trVar against
   65 for plain MC with burn-in 64 (bias^2 1.6e-7), i.e. it only wins asymptotically; RQMC on the noise of all layers
   gives 3-6x at N = 64-1024; a Gaussian surrogate control variate (K2-Stein closure for mean and gate probabilities p_l,
   surrogate delta_l = diag(p_l) W_l delta_{l-1} + xi_l driven by the same noise, exact E[relu(W_1(m + delta))]) leaves
   29% of the variance at twice the per-sample cost; the K2 closure alone has nMSE 2e-3. Propose the estimator(s) most
   likely to have a constant that improves with width, or a better exponent, given the structure in tasks 2-3, with a
   short argument and the cheapest test. Consider in particular methods that treat the radius separately (its quenched
   law is near-Gaussian with CV ~ 0.2-0.35, while it multiplies everything), radial importance sampling for the tail,
   and final-noise Rao-Blackwellisation.
End with a ledger (claim, label, cheapest check).
