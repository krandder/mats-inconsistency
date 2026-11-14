You are an adversarial reviewer (a different model family from the author). Second pass: check the CLAIMS of the report
/home/agent/workspaces/mats/outbox/arc-noisy-deep-2026-10-29.md against the code and data in
/home/agent/workspaces/mats/work/arc-noisy/ (read-only: do not modify anything there; scratch only under
/tmp/arcn_review/). Python: /home/agent/.venvs/arc-noisy/bin/python with OMP_NUM_THREADS=1, at most 2 processes, checks under
~15 CPU-minutes in total (the machine is shared). Your first-pass review is /tmp/arcn_review/review_code.md; check that its
11 findings were actually fixed (gpu_samplers.py v2, hybrid.py v2, combine.py v2, analyze.py, LEDGER.md), and say which are
not.

Regenerate the tables yourself where feasible: `python analyze.py`, `python hybrid.py <seeds> <sigma> <L>`,
`python combine.py <sigma> <L>` (run from the work directory; they only read results/, truth/, samp/, samp_v2/).
Then rank problems by severity, with file:line and evidence:
1. Is every "win" in the report real by the PREREG definition (lower nMSE than the best entry of the same type at every
   budget of a stated contiguous range, bootstrap 95% CI of the per-net ratio above 1 for both median and mean, on the 24
   evaluation nets), and is each win's smallest noise level stated? Is any win produced by a cost error, a lookup error,
   a dev/eval leak, or comparing the hybrid with a weaker baseline than the best sampler?
2. Is every "no win" / blocker claim supported at the stated strength? In particular the claims in sections 4-5: the
   meet-in-the-middle and copula seed oracles (scratch/seeded_oracle.py, scratch/copula_seed.py, logs/seeded_*.log,
   logs/copula_seed_24.log), the Edgeworth oracle (scratch/edgeworth_oracle.py, logs/edge_oracle_*.log), the H1 oracle and
   closure lines, the multilevel argument (is "worse than one Kerdock level for every split we computed" justified by
   arithmetic shown?), the error-survival measurement.
3. Are the competitor numbers (kprop variants, Kerdock-MUB, Sobol, MC) correctly computed and correctly described,
   including FLOP conventions and the statement that ARC's counter favours kprop by about 1.3x?
4. Any number in the summary that does not match the tables or logs; any claim stated more strongly than the evidence.
Write findings to /tmp/arcn_review/review_claims.md (first line: the model you run on), ranked and concise.
