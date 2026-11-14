You are a critical reviewer (probability, random dynamical systems, Monte Carlo methods), read-only, fresh session.
Review the report /home/agent/workspaces/mats/outbox/stationary-regime-2026-10-31.md and its evidence in
/home/agent/workspaces/mats/work/stationary/ (FORMAL.md theory v2, PREREG.md with hashes in prereg_hashes.txt,
results/*.json and *.md, runs/*.json, scratch/*.py, lib/st.py, coprover/ for the co-prover rounds). Do not edit files.

Check, in order of importance:
1. Every claim in the "Summary for Kelvin" and the scorecards: is it supported by the cited runs, at the stated label
   (PROVEN / MEASURED / HEURISTIC / CONJECTURE)? Recompute key numbers from runs/*.json where cheap.
2. The preregistration: were predictions hashed before the data they score (compare prereg_hashes.txt times with the
   runs' timestamps and logs/queue_*.log)? Is any prediction scored generously, or any deviation unreported?
3. The cost accounting of each estimator (units: n x n matrix-vector products), the fairness of the comparisons with
   MC with burn-in, and whether "gain", "exponent" and "constant" are computed as stated (scratch/analyze_B.py).
4. Statistical resolution: are differences claimed where the replicates or truth resolution cannot support them?
5. The theory: errors, overstated labels, missing hypotheses (FORMAL.md, report sections 2-3).
6. The plain-language summary: anything misleading for a reader who will not open the files?
Give a verdict (accept / minor revision / major revision), then numbered findings, each with the file and line or
the number at issue and the fix. Be concrete; do not pad.
