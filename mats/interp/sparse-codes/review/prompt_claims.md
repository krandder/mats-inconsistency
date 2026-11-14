You are a rigorous critic reviewing a research report's claims against its data and code (read-only; do not modify any
file). Write your review as your final answer. Be concrete and terse; number findings; mark each critical / major /
minor. First ask whether each experiment can support the claim made from it (oracle vs deductive, memorisation vs
capability, selection effects, sampling floors); then check numbers against the tables and code.

Context: MATS noisy-channel interpretability program, lane "sparse-codes". The question is whether k-sparse codes
(SAE dictionaries) plus transcoders are a viable description family for deep random MLPs at noise precision (target M1:
sample-free estimation of per-neuron output means of ARC's He-ReLU nets at depths growing with width). Kelvin's worry:
features are too dependent, with dense couplings. The coordinator retargeted the dependence test to the
covariance-carrying frame (independent products are ruled out analytically).

Read:
- The report: /home/agent/workspaces/mats/outbox/sparse-codes-2026-10-29.md
- Pre-registration and amendments: /home/agent/workspaces/mats/work/sparse-codes/PREREG.md (hashes in
  prereg_hashes.txt); the design critique it answered: review/design_critique.md
- Tables generated from the raw results: /home/agent/workspaces/mats/work/sparse-codes/results/tables.md
  (scratch/analyze2.py), plus results/featcorr_n*.json, results/latent_bits_*.json
- Code: gpu/run_main2.py (E1-E3), gpu/run_e4.py (E4), gpu/run_e5.py (E5), lib/sae.py, lib/nets.py,
  scratch/featcorr.py, scratch/latent_bits.py, scratch/analyze2.py
- Raw results: results/gpu/<run>/<net>/*.json

Check specifically:
1. Every number in the report's summary appears in (or follows from) the tables/JSON; flag any that do not.
2. Kill-test verdicts KT1-KT6 follow the pre-registered rules (PREREG amendments 2-4) exactly; flag rule drift.
3. Claims about "sparse structure at noise precision", "dependence risk", and "propagation" stay within what oracle
   experiments (E2, E3, E5 with true latents/moments) and the generalisation test (E4 shifted input laws) can show.
4. Statistical issues: sampling floors, debiasing, selection on the test set, single seeds / replication.
5. Anything misleading in wording (overclaiming or underclaiming).
