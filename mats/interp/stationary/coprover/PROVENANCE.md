# Provenance: Claude (C) and gpt-6-astra (A), stationary-regime lane

## Round 1 (A: 13:03:18Z-13:28:26Z, xhigh, read-only, told not to read work/stationary/)
Prompt `prompt_r1.md` (sha dc2b81a8), raw `astra_r1.raw`, answer `astra_r1_answer.md`.
My v1 (`FORMAL_v1_as_written.md`, frozen 13:15Z, sha b88f4579) was written before A's answer. Disclosure: at about
13:13Z I saw the last lines of A's log (a table "tail_C ... C/n ~ 0.40" for n = 8..256). My scalar radius chain and the
0.4n/Z bulk were derived and checked on dev data (dev_chain.py output at about 13:10Z) before that.

Found independently by both [C+A]:
- scale invariance of Model S (sigma is not a shape parameter; sigma/tau ~ n^-1/2 set by width);
- the exact annealed scalar chain for the radius (C: n Q' = (sqrt(g n Q) + eta)^2 + C; A: Q' = A Q + B), the Kesten
  tail index 1 (C by Goldie with bounded perturbation; A with the explicit constant), and the inverse-gamma / Dufresne
  bulk Q/n -> 2/(5Z) (C heuristic; A proof);
- the input-forgetting time T_mem ~ (4n/5) ln(1/(sigma sqrt n)) + O(n);
- births of order sqrt(n) per layer in Model S and n sigma in Model R;
- pairwise contraction of the noisy RDS (C: Lemma 1 identity and a compactness sketch; A: uniform constant rho_{n,p}).

New from A (adopted in v2): exact global resets and uniqueness among all entrance laws; the explicit uniform
contraction constant with a sign-symmetrisation proof; quenched exponential tails versus the annealed power tail; the
tail constant C_n = 1/E[A log A]; infinite-time exponents are -infinity at fixed n, and the rank-conditioned exponent
formula with its n^-2 corrections; the scope of the Lyapunov covariance formula (fixed coefficients only) and
E[J^T J | past] = I; deletion-based flip formula and the band-count reading of sqrt(2/pi); Model R's exact scale/shape
decomposition; the nested-sharing variance tradeoff; the hinge-residual constants; final-noise Rao-Blackwellisation (I had it in my session plan before A's answer, but
it entered partB.py only after I read A's P24, and no file documents the earlier idea, so it is counted as A's).
New from C (not in A's round 1): the radial-prefactor point (the top singular value of the Jacobian product carries
the Frobenius mass of all n directions), the resampling flip formula arccos(r)/pi, and all of the Part A measurements.

## Round 2 (A: 13:36:00Z-, xhigh, read-only, may read work/stationary/)
Prompt `prompt_r2.md`: audit of v1 and of the Part A scoring, theory of the cloud excess and of the interactions,
Model R, Part B design. PENDING.
