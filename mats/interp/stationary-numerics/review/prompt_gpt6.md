You are a critical reviewer (probability, random dynamical systems, deep-learning theory at finite width, Monte Carlo),
read-only, fresh session. Do not edit any file.

Review the report /home/agent/workspaces/mats/outbox/stationary-numerics-2026-10-31.md and its evidence in
/home/agent/workspaces/mats/work/stationary-numerics/ : PREREG.md with prereg_hashes.txt (hash times), notes/proofs.md,
notes/literature.md, notes/questions.md, lib/*.py, tests/*.py with logs/test_*.log, scratch/*.py (batch, scoring and
figure scripts), runs/*.json, results/*.json and *.md, figs/*.png. Section 4 (exact densities at n = 2-4) and the
scorecard table are still pending; skip them. The sibling lane's data used for comparison are in
/home/agent/workspaces/mats/work/stationary/ (runs/, results/partI_scores.json, FORMAL.md).

Check, in order of importance:
1. Every claim in "Summary for Kelvin" and sections 1-3, 5-6: is it supported by the cited files at the stated label
   (PROVEN / NUMERICAL / MEASURED / HEURISTIC)? Recompute key numbers from runs/*.json and results/*.json where cheap
   (e.g. the medians in section 3.1, the shares in 3.2, the G7 ratios, the radius table).
2. The proofs in notes/proofs.md: the radius reduction Q = (S/n) r with r' = 1 + A r (independence claims), the replica
   Gram chain (exactness, which quenched quantities it gives and which it does not), the eps-curve lower bound, the
   frozen-gate affine claim, the Kramers-Moyal expansion (is calling the closed forms PROVEN as a formal series and
   NUMERICAL as asymptotics fair?), the left-tail quasi-potential (HEURISTIC).
3. Pre-registration: were predictions hashed before the data they score (compare prereg_hashes.txt times with run files'
   mtimes and logs)? Is any prediction scored generously or any deviation unreported? Note the header-time rehashes
   and the bug fix in response_net.py.
4. Methods: the unbiasedness of the single-layer-refresh estimator of the mean dimension; the causal "free_k" variants
   (are they well-defined functions of the window noise?); the CLV computation with rank loss; whether the response
   collapse in section 5.1 is real; whether the "5-8x" explanation in 2.5 is a fit in disguise.
5. Statistical resolution: are differences claimed that the replicates cannot resolve (e.g. the n-dependence in 3.2,
   the annealed Gram-chain D_h)?
6. The plain-language summary: anything misleading or overstated for a reader who will not open the files?
Give a verdict (accept / minor revision / major revision), then numbered findings, each with the file and line or the
number at issue and the fix. Be concrete and terse; do not pad.
