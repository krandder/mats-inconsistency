You reviewed /home/agent/workspaces/mats/outbox/kstar-frames-2026-10-30.md earlier (your review:
/home/agent/workspaces/mats/work/kstar-frames/review/review_gpt6.md, verdict "major revisions"). The report and code were
revised. Read-only: do not edit any file; short read-only Python checks are allowed (nice -n 19, at most 2 threads,
under 3 minutes each, /home/agent/.venvs/arc-noisy/bin/python, lib in /home/agent/workspaces/mats/work/kstar-frames/lib).

Focused re-check. For each of your 19 findings, say whether it is fixed, partly fixed, or not fixed, with one line of
evidence. Then check the new material:
1. lib/cover.py v2 (float64 GEMMs; Cholesky with an eigh pseudo-inverse fallback when a pivot^2 < 1e-6; ridge 1e-15) and
   scratch/r2c_cover.py: is the projector convention now stable and correct (tests/test_cover.py includes your
   ill-conditioned case)? Spot-check two v2 statistics in runs/Bcover2_*.json against the report's section 5 table.
2. notes/props.md and report section 2: are the restated Propositions 1 and 2 (rank conditions; spectral-gap event G;
   fixed or past-measurable k; "chance plus one direction") now correct? Is scratch/t2_prop2_cond.py (fixed pasts,
   resampled futures) a valid conditional check?
3. scratch/r9_score.py: net-mean aggregation for P2-P7 and P11c, the P13 scoring on Bcover2, the binning/per-net
   sensitivity, and the COHERENT direct alternative. Do the report's scorecard lines match logs/scorecard.log?
4. PREREG.md Amendment 2 and prereg_hashes.txt: declared correctly?
5. Any remaining claim in the report (summary first) stronger than its evidence, or any number that does not
   reproduce. Be brief: numbered findings, then a one-line verdict (accept / minor revisions / major revisions).
