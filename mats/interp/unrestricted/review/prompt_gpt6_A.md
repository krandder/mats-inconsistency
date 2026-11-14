You are an adversarial referee (GPT-6), in a fresh session, reviewing the THEORY part of a research report before it
goes to its reader. Work read-only; do not modify files. Be concrete and terse; number your findings.

Files (read-only):
- The report draft: /home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md. Sections 2
  (experiments), 6 (scorecard) and 7 (process) and the summary are PENDING; review sections 1, 3, 4 and 5 only.
- FORMAL.md (theory v2.1), coprover/FORMAL_v1_as_written.md, coprover/astra_r1_answer.md, coprover/astra_r2_answer.md,
  coprover/PROVENANCE.md, notes/analogies.md, notes/literature.md, PREREG.md, prereg_hashes.txt.
- Code for the theory computations: scratch/spin_glass.py, scratch/spin_glass_field.py, scratch/noise_stability.py,
  scratch/nstab_mf_compare.py and its log logs/nstab_mf_compare.log, runs/nstab_*.json, runs/spin_glass*.json.

Tasks:
1. For every claim in sections 3-5, check that its label (PROVEN, PROVEN UNDER X, COMPUTED, MEASURED, HEURISTIC,
   CONJECTURE) is deserved. Check each proof or proof sketch; give a counterexample or the missing step where one
   fails. Pay special attention to: the deterministic simulation of an arbitrary randomized estimator (3.1, fourth
   bullet; FORMAL 1, consequence 3), the threshold-circuit seed construction, the barrier wording (3.3), the
   restricted-class numbers in 3.5 (reproduce them from runs/nstab_*.json and the log), and the spin-glass statements
   (4.3).
2. Check the provenance tags [C], [A], [C+A] against coprover/PROVENANCE.md and the two astra answers.
3. Check the literature statements: which are verified, which from memory, and whether any is wrong or overstated.
4. Check the analogies (section 4): is each correspondence stated correctly, and is each verdict justified?
5. Check the conjecture ledger (section 5): is the bottleneck correctly identified and stated precisely?
Give a verdict (accept / minor revision / major revision) and a prioritised list of required changes.
