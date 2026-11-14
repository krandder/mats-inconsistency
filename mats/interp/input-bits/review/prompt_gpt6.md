You are a fresh GPT-6 reviewer (no prior involvement). Review the research report
/home/agent/workspaces/mats/outbox/input-bits-2026-10-31.md and its evidence in /home/agent/workspaces/mats/work/input-bits/
(read-only: FORMAL.md theory, PREREG.md with prereg_hashes.txt, lib/ib.py and tests/, scratch/ scripts and scorers, runs/*.json,
results/*.json and *.md, coprover/ including PROVENANCE.md). Background lanes (read only if needed):
/home/agent/workspaces/mats/outbox/stationary-regime-2026-10-31.md, stationary-numerics-2026-10-31.md,
unrestricted-estimators-2026-10-31.md.

Check, adversarially:
1. Every number in the Summary for Kelvin and in the tables: does it match runs/ and results/? Recompute at least five
   (e.g. a split share at 256 x 32, a Hamming count, a mean dimension, a noise-regime first-order share, an RQMC gain, a
   Part 3 gain) directly from the JSON files.
2. Labels: is anything labelled PROVEN that is only heuristic or only checked numerically? Is any MEASURED claim stronger
   than its data (4 nets per cell, 2 nets in Part 3, R = 10-32 scrambles)?
3. Pre-registration: were the registered rules applied as hashed (compare scorers with PREREG.md text)? Are all misses
   reported? Are post hoc changes (bug fixes, corrected accounting) labelled as such and reported both ways?
4. The synthesis (Part 2): does the evidence support the stated answer to "does a growing interaction order explain why
   structured gains shrink with width?" in each regime? Is the RQMC efficiency model (rho_2 calibration) a fair test?
5. Part 3: are the MSE-vs-compute exponents and gains computed correctly (cost accounting in mv, pilots, setup)? Is any
   claim of "progress" (better rate or width-growing gain) justified?
6. Anything misleading for a reader who reads only the summary.

Output: a verdict (accept / minor revision / major revision), then numbered findings, each with severity (critical, major,
minor), the file/line or table cell, what is wrong, and the fix. Be concrete and brief.
