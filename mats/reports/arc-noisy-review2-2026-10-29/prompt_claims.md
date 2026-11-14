You are an adversarial reviewer (different model family from the author) of the CLAIMS in a research report. Read-only
everywhere except /tmp/arcn_review2/ (write scratch and your review there). Python: /home/agent/.venvs/arc-noisy/bin/python
(OMP_NUM_THREADS=1, at most 2 processes, each check under ~10 CPU-minutes; shared machine; no GPU).

The report section to review is "Part 2: rare output events at depth" at the end of
/home/agent/workspaces/mats/outbox/arc-noisy-deep-2026-10-29.md (Part 1 above it is already reviewed; ignore it
except for context). Its pre-registration is /home/agent/workspaces/mats/work/arc-noisy/tails/PREREG-TAILS.md
(with amendments 1-2; hashes in ../prereg_hashes.txt). Code: /home/agent/workspaces/mats/work/arc-noisy/tails/
(analyze_tails.py produces every table: its full output is in tails/results/tables_s0.001.txt, tables_s0.003.txt, tables_s0.01.txt (and report_tables_s*.md, rendered from the summary_s*.json files); killtests.py produces the
kill-test numbers in tails/results/killtests.txt). Raw data: tails/runs/ (thresholds, truths, kprop outputs, Monte Carlo
moments and counts, Kerdock counts). A code review was done earlier (/tmp/arcn_review2/review_code.md); the author's
fixes are listed in the report's review section.

Check every quantitative claim and every verdict in Part 2 against the raw outputs and the pre-registration:
1. Does each number in the summary and tables match tables_part2.txt / killtests.txt? Are CIs, medians and fractions
   computed as stated (per-net medians, bootstrap over nets)?
2. Are the verdicts (kill tests, wins, "better than zero", smallest p per method, sigma trend) what the pre-registered
   rules give? Flag any claim that goes beyond the rules or the data, any selective reporting, and any post-hoc result not
   labelled as such (the saddlepoint readout of ARC's description, U2, failed its kill test and must be labelled post hoc).
3. Truth quality: provisional events, cross-check failures; does any claim depend on provisional events?
4. Fairness: is ARC's tail method run as ARC ran it (factorized K = 3, heaviside readout, run_lpe settings) and charged
   with ARC's own counter? Is Monte Carlo charged and amortised fairly? Is splitting charged fairly (pilot, runs)? Are
   amortised and per-event costs compared like with like?
5. Wording: anything overstated, ambiguous or missing a caveat (deductive vs sampling, biased vs unbiased, depth and sigma
   ranges, number of nets).

Write /tmp/arcn_review2/review_claims.md: numbered findings ranked by severity (critical / major / minor) with evidence and
the exact fix (replacement wording where useful); then the claims you checked and found correct.
