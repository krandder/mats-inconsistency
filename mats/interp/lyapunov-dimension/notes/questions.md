# lyapunov-dimension: definitions, assumptions, open questions (29 Oct 2026, 18:50Z)

## Objects
- Net (ARC, as work/arc-noisy): z_l = W_l h_{l-1}, a_l = relu(z_l), h_l = a_l + s_l eps_l, W_l iid N(0, 2/n), x ~ N(0, I_n).
- Cut at layer c, remaining depth L' = L - c. Downstream mean map F(z) = E[a_L | z_c = z] (downstream noise integrated).
- Path Jacobian J_x = d a_L / d z_c = D_L W_L ... D_{c+1} W_{c+1} D_c (L' weight matrices).
- M = E_x[J_x^T J_x] (input- and noise-averaged); Sigma = Cov(z_c); S = Sigma^{1/2} M Sigma^{1/2}.

## Error notions and the k* they define (U = the jointly exact block, dim k)
- E1 operator, one linear map: kappa_op(U) = ||J P_{U-perp}|| / ||J||; k*_op(delta) = #{j : s_j(J) > delta s_1(J)} (Eckart-Young).
- E2 Frobenius, one map, isotropic error: k*_F(delta) = min{k : sum_{j>k} s_j^2 <= delta^2 sum_j s_j^2}.
- E3 input-averaged, one fixed U for all paths: the same with the eigenvalues of M.
- E4 spread-weighted (whitened block functionals): the same with the eigenvalues of S.
- E5 the estimand: mean error e = E_Q F - E_P F of a family Q exact on the block; nMSE = |e|^2 / tr Var(a_L).

## Assumptions, each with its check
- A1 Gaussian products at finite L'/n follow Newman's exponents for the top directions. Check: numerics at
  n = 128-512, L'/n = 1/8-1/2 against Newman and against free probability (Fuss-Catalan).
- A2 Along one noiseless ReLU path the angular part of J is exactly a product of independent Gaussian matrices of size
  (m_l - 1) x (m_{l-1} - 1), m_l ~ Bin(n, 1/2). Check: proof, plus singular-value numerics on ARC nets with and
  without sigma = 0.001 against the Gaussian product.
- A3 Input averaging can only increase k* (Ky Fan); its size is set by the mask overlap of input pairs. Check: proof
  (Ky Fan), two-replica recursion vs numerics for tr M, tr M^2 on ARC nets.
- A4 The linear-response (pathwise) model governs how the PB_k mean error falls with k. Check: against the nongauss
  PB ladder and the state spectrum tail (both measured by that lane).
- A5 Gate flips add O(eps^2) to pathwise errors of relative size eps. Check: finite-difference numerics.
- A6 Exact identity for distributional errors: e = E_Q[F - E_P(F | block)]. Check: proof.

## Open questions
- Q1 Is k* ~ 64 at 256 x 64 (cut 32) set by the state spectrum, the sensitivity spectrum, or their product S?
- Q2 Scaling of k* with n at fixed L/n: O(1) (coordinator's heuristic) or about linear (nongauss prediction)?
- Q3 Single path vs input-averaged: a constant factor or a different scaling?
- Q4 Cost of an exact non-Gaussian k*-dim block, and what structure inside the block could beat sampling.
