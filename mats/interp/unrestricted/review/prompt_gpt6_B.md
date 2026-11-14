You are an adversarial referee (GPT-6), in a fresh session, doing the FINAL review of a research report before it
goes to its reader. Work read-only; do not modify files. Be concrete and terse; number your findings.

Files (read-only):
- The report: /home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md (complete).
- An earlier review of its theory sections and the changes made: review/review_gpt6_A.md, FORMAL.md sections 9-10.
- Theory: FORMAL.md, coprover/astra_r1_answer.md, coprover/astra_r2_answer.md, coprover/astra_r3_answer.md,
  coprover/PROVENANCE.md, notes/analogies.md, notes/literature.md.
- Pre-registration: PREREG.md (with Amendment 1 and a post-hoc addition), prereg_hashes.txt.
- Code: lib/nets.py, lib/bnb.py, lib/mlmc.py, tests/test_bnb.py, tests/test_mlmc.py, scratch/*.py.
- Results: runs/*.json, results/eval_tables.md, results/eval_scores.json, results/theory_scores.json, logs/*.log.

Tasks:
1. Experiments (section 2 and the scorecard in section 6): check the B&B results against runs/*.json and
   results/eval_*; reproduce at least five numbers; check that each registered prediction P1-P13 is scored by its
   registered rule and that every deviation and post-hoc analysis is declared. Check the equal-cost comparison and
   the FLOP accounting, including the factor-2 sensitivity check.
2. The multilevel sampler (section 3.7, lib/mlmc.py, scratch/mlmc_check.py): is the construction unbiased, is the
   cost accounting right, do the measured ratios in the table follow from logs/mlmc_check.log and runs/mlmc_*.json,
   and are the proven / measured / heuristic labels deserved? Is the refutation of C0 correctly stated (which cost
   model, which baseline)?
3. The summary: does every claim follow from the body, with the right label and no overstatement? In particular
   "which prong wins", the 10M+ parameter statements, and the numbers quoted.
4. Check that the fixes for review A were applied correctly and did not introduce new errors.
5. Anything else that is wrong, unsupported, or unclear to a reader who has not followed the lane.
Give a verdict (accept / minor revision / major revision) and a prioritised list of required changes.
