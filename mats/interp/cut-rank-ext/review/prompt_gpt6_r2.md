You are GPT-6, a fresh reviewer doing a focused re-check (read-only; do not modify files). An earlier review of the
report /home/agent/workspaces/mats/outbox/cut-rank-ext-2026-10-31.md is in review/review_gpt6_final.md (15
findings, verdict major revision). The report has been revised. Post-review reruns: PREREG.md Amendment 2,
scratch/post_review_fixes.py, runs/*_nc2g.npz, runs/*_arc3k4.npz, runs/*_conv.npz, logs/conv_check.log,
logs/arc_bias_alpha_t.log (scratch/arc_bias_alpha_t.py), notes/coverage_map.md, scratch/nl_analyze.py (now also scores
the reruns).
1. For each of the 15 findings: ADDRESSED / PARTLY / NOT ADDRESSED, with one line of evidence (report location).
2. Any new error introduced by the revision (statements, numbers, labels). Recompute a few of the new numbers: the
   NC2 global-code slope and 48-bit threshold, the ARC3 rerun medians and blow-up fractions, the convergence-check errors,
   and the alpha_t bias ranges in report section 5.
3. Verdict (accept / minor revision / major revision) and the remaining required changes, if any. Be terse.
