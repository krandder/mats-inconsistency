You are an adversarial referee (GPT-6), in a fresh session, reviewing a research report before it goes to its
reader. Work read-only; do not modify files. Be concrete and terse; number your findings.

Files (all read-only):
- The report: /home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md
- Theory: FORMAL.md (v2), coprover/FORMAL_v1_as_written.md, coprover/astra_r1_answer.md, coprover/astra_r2_answer.md,
  coprover/PROVENANCE.md, notes/analogies.md, notes/literature.md
- Pre-registration: PREREG.md, prereg_hashes.txt (check the hashes and times against the files)
- Code: lib/nets.py, lib/bnb.py, tests/test_bnb.py, scratch/*.py
- Results: runs/*.json, results/eval_tables.md, results/eval_scores.json, logs/*.log

Tasks:
1. For every numbered claim in the report's summary and sections, check that its label (PROVEN, PROVEN UNDER X,
   COMPUTED, MEASURED, HEURISTIC, CONJECTURE) is deserved. Check each proof or proof sketch; give a counterexample or
   the missing step where one fails.
2. Check the certified branch-and-bound (lib/bnb.py): the cone parametrisation, the 1-D moment quadrature, the
   symbolic bounds and relaxations, the homogeneous extension to cones, exactness on certified cells, the FLOP
   accounting (is every operation of order n or more charged?), and whether the comparison with RQMC/RB-MC/MC at
   equal cost is fair. Reproduce at least three numbers in the report from runs/*.json.
3. Check the scoring of every pre-registered prediction (PREREG P1-P13) against the registered rule, and list every
   deviation, post-hoc analysis or unregistered change that the report does not declare.
4. Check the literature claims: which are verified, which are from memory, and whether any is wrong or overstated.
5. Check the analogies section (threshold circuits, convex-body volume, spin glasses): is each correspondence stated
   correctly, and is each "plausible / not plausible" verdict justified?
6. Check that the report's conclusions (which prong wins; what scales to 10M+ parameters; the bottleneck conjecture)
   follow from the evidence and are not overstated.
Give a verdict (accept / minor revision / major revision) and a prioritised list of required changes.
