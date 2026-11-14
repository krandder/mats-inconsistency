You are an adversarial referee (theoretical computer science / probability) checking a draft before its claims reach a
research lead. Work read-only. Scratch files only under /tmp/hardness_review_r1/. Any computation: at most ONE process
at a time, under `nice -n 19`, OMP_NUM_THREADS=1, at most about 2 CPU-minutes in total; Python at
/home/agent/.venvs/arc-noisy/bin/python.

File to check: /home/agent/workspaces/mats/work/hardness/draft.md, SECTIONS 1-5 ONLY (Section 1 definitions;
Theorem 2.1, Corollary 2.2, Theorem 2.3 and remarks; Propositions 3.1-3.5 and Corollary 3.3; Theorem 4.1 (a)(b)(c),
Corollaries 4.2 and 4.3, Proposition 4.4; Theorem 5.1 (a)(b)(c), the SH(F) instantiation, Proposition 5.2).
Local copies of the cited papers' text: papers/ikw/ikw02.txt (Impagliazzo-Kabanets-Wigderson; Section 5.1, Theorem 32,
Remark 33, Corollary 34), papers/chen-tell/ct.txt (Chen-Tell STOC 2021; Section 6, Theorem 6.4 and its proof,
Assumption 6.1, Lemma 6.3, Definitions 3.1-3.3), papers/wu/wu.txt (ARC's paper). ARC's MSP is quoted in Section 1.5 of
the draft from their blog post. A numerical check of Theorem 4.1 is in numerics/capp_check.py and
numerics/logs/check_capp.txt.

Your job: try to BREAK each numbered statement. For each one give a verdict: CORRECT, CORRECT WITH A FIXABLE GAP
(say exactly what to change), or WRONG (give a counterexample or the failing step). Check in particular:
1. Theorem 2.1: is the posterior argument right for adaptive randomized algorithms? Are the constants in Corollary 2.2
   (factor 4(1 - 1/(2N)) for uniform input; <= 5.9/(1-3.3w')^2 for Gaussian with intervals in [-1, 1]) right? Is the
   width/depth embedding claim plausible? Is Theorem 2.3 right?
2. Proposition 3.2: is the Levy-distance argument right (directions of inequalities, the role of eta)?
   Corollary 3.3: is the PRF argument right? Proposition 3.4 algebra. Proposition 3.5 rotation argument.
3. Theorem 4.1: check each gate construction and the invariants, the bias-free version (independence of r = |x_0| from
   the sign bits; the error bound m/(2K)), and the noisy version (restore layer thresholds, the 4 tau accounting, the
   tail bound 2U exp(-1/(512 sigma^2))). Is "depth d + 1 / d + 3" right? Edge/unit counts?
4. Corollary 4.2: does it correctly instantiate IKW Theorem 32 (definitions of "nontrivially approximated": the time,
   advice and i.o. quantifiers, additive error 1/6)? Corollary 4.3: does the adaptation of Chen-Tell's Theorem 6.4
   proof (case k = 1) really go through with a deterministic mean estimator for ReLU DAG networks running in time
   O(E^{2 - eps})? Check sizes (circuit size n polylog n), the random-point sampling, soundness probability, whether
   #NSETH (not NSETH) suffices, and whether "DAG network" vs MLP matters. Is the claim "Monte Carlo does it in
   O(E log(1/delta))" a fair comparison?
5. Theorem 5.1: posterior variance in (a), (b); the bound R(Q) >= 1/(Q+2) - 2^{-J}; Monte Carlo variance
   J - 1 + 2^{-J}; the advantage bounds 4 J^2 and J(J-1)/4; part (c) inequality chain (Cauchy-Schwarz, Horvitz-Thompson
   variance). Proposition 5.2 logic.
6. Anything overclaimed, mislabelled (PROVEN vs conditional), or misattributed to the cited papers.
Write your report to /tmp/hardness_review_r1/review.md AND print it to stdout. Start with a one-paragraph verdict, then
one section per statement, then a list of required fixes ordered by severity.
