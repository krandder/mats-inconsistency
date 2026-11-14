You are an adversarial referee (probability, random matrices, mean-field theory of deep networks) checking a draft
before its claims reach a research lead. Work read-only. Scratch files only under /tmp/hardness_review_r2/. Any
computation: at most ONE process at a time, under `nice -n 19`, OMP_NUM_THREADS=1, at most about 3 CPU-minutes in
total; Python at /home/agent/.venvs/arc-noisy/bin/python (numpy, torch available).

File to check: /home/agent/workspaces/mats/work/hardness/draft.md, SECTION 6 ONLY (Section 1 for
definitions): Lemmas 6.1-6.4, Theorem 6.5 and its table, Lemma 6.6, Propositions 6.7, 6.8, Corollary 6.9,
Lemma 6.10 and its example, Proposition 6.11, Theorem 6.12, Proposition 6.13. Numerical checks the author ran:
numerics/lemma_checks.py with logs in numerics/logs/check_J.txt, check_backward.txt, check_survival_hermite.txt.

Your job: try to BREAK each numbered statement. For each one give a verdict: CORRECT, CORRECT WITH A FIXABLE GAP (say
exactly what to change), or WRONG (counterexample or failing step). Check in particular:
1. Lemma 6.1: exact iid structure of the norm ratios (including the h = 0 edge case), Var rho = 5/n.
2. Lemma 6.2: the conditional formula, the closed form of J(c) (degree-2 arc-cosine kernel normalization), monotonicity.
3. Lemma 6.3: the one-sided inequality for nonnegative summands as used (is it P(S <= ES - s) <= exp(-s^2/(2 sum E X^2))?),
   the minimum 0.01267 over c in [0,1], the Chernoff constant 0.0646, and the conclusion c_l >= c_*.
4. Lemma 6.4: the stopped-product argument (is E[Y_{l-1} Z_l] >= (1 + delta_*) E[Y_{l-1}] valid; is the subtraction
   step valid; is E rho^4 = 1 + 30/n + O(n^-2) right; is the error term right and does it vanish for L = O(n)?).
5. Theorem 6.5: the conditional expectation E[S | A] (does the norm law for the last k layers really give it, given that
   S averages over x?), the ANOVA orthogonality argument, the pointwise use of Lemma 6.4 under the expectation over
   (x, x') with weights ||x||^2 ||x'||^2/n^2, the Monte Carlo upper bound, and whether the table's numbers and the
   "N_* = 1-3 samples" reading are justified by the code (numerics/lemma_checks.py check_backward).
6. Lemma 6.6 (telescoping identity): conditions (estimators that are not measures, e.g. cumulant propagation).
7. Proposition 6.7: exactness, and the claim about noise.
8. Proposition 6.8: the independence argument (w . delta independent of {w . h : h in supp Q} when delta is orthogonal
   to the support), first-order validity, and whether the numerical check tests what is claimed.
9. Corollary 6.9: the asymptotics 1 - c_l ~ 9 pi^2 /(2 l^2), theta_l ~ 3 pi / l, survival ~ (ell/L)^3, and the
   accumulated error e^2 L / 4. Check the expansion 1 - kappa(c) = (1-c) - (2 sqrt 2/(3 pi))(1-c)^{3/2} + ...
10. Lemma 6.10: the martingale/orthogonality proof, and whether the Hermite-truncation example satisfies its hypotheses
   (constant conditional second moments; orthogonality); the r_D values.
11. Proposition 6.11: the second-moment computation (Var of w^T M w for Gaussian w and symmetric M).
12. Theorem 6.12 and Proposition 6.13.
13. Anything overclaimed or mislabelled (PROVEN vs conditional vs heuristic); anything that contradicts known results
   (Hanin 2018, Hanin-Nica 2020, Poole et al. 2016, Schoenholz et al. 2017).
Write your report to /tmp/hardness_review_r2/review.md AND print it to stdout. Start with a one-paragraph verdict, then
one section per statement, then required fixes ordered by severity.
