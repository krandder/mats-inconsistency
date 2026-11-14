You are an adversarial referee (random matrix theory, deep-learning theory, Monte Carlo methods). Review a research
report and its code. Read-only: do not edit any file. You may run short read-only Python checks (CPU only, at most
2 threads, under 3 minutes each, prefixed with `nice -n 19`), using /home/agent/.venvs/arc-noisy/bin/python and the
code in /home/agent/workspaces/mats/work/narrow-deep/lib (add it to sys.path).

Report: /home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md
Code, logs, results: /home/agent/workspaces/mats/work/narrow-deep/ (lib/nd.py, lib/k2.py, lib/tnets.py,
lib/gauss_relu.py; scratch/*.py; logs/*.log; runs/*.json and runs/p3/*.json; results/*.md; PREREG.md with
prereg_hashes.txt; notes/refs.md for the literature).
Context it builds on (read-only): /home/agent/workspaces/mats/outbox/lyapunov-dimension-2026-10-29.md (Theorem C,
Corollary C1), /home/agent/workspaces/mats/outbox/cm-interp-program-2026-10-28.md ("The bar").

Check, and report every problem you find, most serious first:
1. Theory. (a) The claim that two inputs synchronise at the rate gap_1 = (1/2) E[psi(m/2) - psi((m-1)/2)],
   m ~ Bin(n, 1/2), derived from Theorem C / Corollary C1 for an infinitesimal separation: is the derivation right,
   which conditioning on m is needed, and is "matches the measured 0.0867 per layer" a fair test (median angle of
   finite separations, layers 32-128)? (b) rank(J_x) <= min_l m_l; the per-path death probability
   1 - (1 - 2^-n)^L; collective death after synchronisation. (c) The heuristic that K2's error on 1/sqrt(L) residual
   nets does not grow with L because closure errors of different layers enter along independent directions V_l.
   (d) The statement that no gate-free subspace bound exists: is the evidence (unmasked products, random masks, PR of
   random-mask top directions, the operator-norm gain bound) adequate, and is the "weights plus one trajectory's
   gates" positive statement supported?
2. Code. Look for bugs that would change a headline number: the Jacobian chains and candidate subspaces
   (scratch/p2_jac.py), the collapse census (scratch/p2_collapse.py), the K2 closures (lib/k2.py: plain, residual
   with the Stein cross-covariance, RNN), the FLOP accounting in MC-path units (lib/nd.py flops_path, k2.py counts,
   scratch/p3_att.py, p3_samplers.py, p3_resdepth.py, p4_est.py), the regression control variate, RQMC, the radial
   Rao-Blackwellised tail estimator and the Imhof formula (scratch/p3_tails.py), the truth runs (p3_truth.py).
   Is the comparison "K2 vs the best simple sampler at equal FLOPs" fair (e.g. cost conventions favouring K2, the
   Stein K2 counted at n^3 per layer while samplers pay n^2 per path per layer, truth error subtracted correctly)?
3. Numbers: spot-check at least 12 numbers in the report against logs/ and runs/ (say which ones).
4. Claims vs evidence: is anything stated more strongly than its support (status labels PROVEN / MEASURED /
   HEURISTIC; number of nets; selection of alive nets at n = 8; the prereg scorecard and amendments)? Are there
   alternative explanations for the residual K2 result (e.g. the nets being nearly linear, noise, small n)?
5. Literature section: any citation that looks wrong or overstated relative to notes/refs.md.
Write your findings as a numbered list with file:line references where possible, then a short verdict.

Note: the report's L = 1024 depth-scaling row and the placeholders [RD1024], [RD-RATIO], [RD-DEPTH], [RD-SHORT],
[RD-P10] are still being computed (runs/resdepth_n16_L1024.json and runs/resdepth_n8_L1024.json may appear while
you review; logs/resdepth.log). Review everything else; if those files exist when you get there, check them too.
[REVIEW] and [END] are filled after your review.
