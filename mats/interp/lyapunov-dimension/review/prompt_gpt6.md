You are an adversarial referee (applied probability, random matrix theory, deep-learning theory). Review a theory
report and its numerics. Read-only: do not edit any file. You may run short read-only Python checks (CPU only, at most
2 threads, under 2 minutes each, with `nice -n 19`), using /home/agent/.venvs/arc-noisy/bin/python.

Report: /home/agent/workspaces/mats/outbox/lyapunov-dimension-2026-10-29.md
Code and logs: /home/agent/workspaces/mats/work/lyapunov-dimension/ (scratch/*.py, lib/fc.py, logs/*.log, runs/*.json).
Context it uses (read-only): /home/agent/workspaces/mats/outbox/nongauss-family-2026-10-29.md,
/home/agent/workspaces/mats/work/hardness/draft.md (sections 6.1-6.2).

Setting: ARC's random MLPs, x ~ N(0, I_n), z_l = W_l h_{l-1}, a_l = relu(z_l), h_l = a_l + noise, W_l iid N(0, 2/n),
no bias. The question: how many state directions must a description carry jointly so that errors confined to the other
directions stay small at the output?

Check, and report every problem you find:
1. Each theorem and proof: Theorem A (exact identity; chi-square bound and its attainment), Theorem B (optimal block
   = Sigma^(-1/2) V_k; the claim that PBW's orth(Sigma^(1/2) V_k) is suboptimal), Theorem C (exact reduction of one
   noiseless ReLU path's Jacobian to independent rectangular Gaussian matrices of size (m_l - 1) x (m_{l-1} - 1); the
   conditional-independence argument; the decomposition J = e_N r^T + Jhat and the interlacing
   s_k(Jhat) <= s_k(J) <= s_{k-1}(Jhat)), Corollary C1 (exponents), Corollary C2 (counts from Fuss-Catalan at half
   width), Theorem D (Ky Fan), the edge law G(delta) (large-N asymptotics of the Fuss-Catalan parametrisation), and the
   two-replica Wick recursion (are the three one-layer expectation formulas right?). Give counterexamples where a claim
   is false, and missing hypotheses where it is incomplete.
2. The status labels (PROVEN, PROVEN UNDER, HEURISTIC, MEASURED, CONJECTURE): is anything labelled stronger than its
   support? In particular the free-probability limit used at L' proportional to n, and the noise case of Theorem C.
3. Numbers: spot-check at least 10 numbers in the report against the logs and json files (say which ones you checked).
4. The empirical law nMSE = C x tail with beta = 1 (section 6, scratch/a4_fit.py): point selection, floors, the mix of
   the nongauss lane's polar-chart families (weighted PCs of the direction law) with tails computed in the z-chart with
   the mean direction projected out, and whether R^2 = 0.96 on 10 net-mean points supports the use made of it.
5. Section 7 (input-dependent blocks): the cross-fitting, the K^(-1/d) extrapolation to 1e7-1e10 classes, and whether
   the conclusion follows.
6. Sections 9-10: the width claim with two widths for C, and the "best we can hope for" argument. Is any conclusion
   stronger than the evidence? Is anything important missing (e.g. a route the argument does not cover)?

Output: a severity-ordered list (critical / major / minor). For each: location, what is wrong, why, and the fix. Then
one paragraph on whether the summary for Kelvin is accurate. Be concrete and terse.
