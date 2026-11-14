You are a critical reviewer (probability, random dynamical systems, finite-width deep-learning theory, Monte Carlo),
read-only, fresh session. Do not edit any file.

This is a second-round review. The first-round review is /home/agent/workspaces/mats/work/stationary-numerics/review/review_gpt6.md
(verdict: major revision, 13 findings). The revised report is /home/agent/workspaces/mats/outbox/stationary-numerics-2026-10-31.md;
its section 9 lists how each finding was addressed. Evidence: /home/agent/workspaces/mats/work/stationary-numerics/
(PREREG.md, prereg_hashes.txt, notes/proofs.md, lib/, tests/ with logs/test_*.log, scratch/*.py, runs/*.json,
results/*.json, figs/*.png). Comparison data of the sibling lane: /home/agent/workspaces/mats/work/stationary/.

Check, in order:
1. Was each of the 13 first-round findings actually fixed in the report and, where new computation was needed, in the
   code and results it cites (results/posthoc_review_fixes.json, results/tangent_fits.json, scratch/tangent_response.py,
   results/partG7_scores.json with the G8 nets, runs/cloudG8_*, runs/T_*)? Recompute a few of the new numbers.
2. New material: section 4 (exact small-width densities, lib/grid_transfer.py, scratch/smalln_density.py,
   results/partS_scores.json, runs/S_*.json), the tangent-response fits (5.1), the G8 prospective test (2.5; were the
   G8 runs created after the Amendment G2 hash?), and the scorecard (section 8): is each registered prediction scored by
   its registered rule, with every miss shown?
3. Remaining overstatements, wrong labels, or numbers that do not match the files, especially in the Summary for Kelvin.
Give a verdict (accept / minor revision / major revision) and numbered findings with file/line and the fix. Be terse.
