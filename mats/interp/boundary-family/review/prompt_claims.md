You are a rigorous claims reviewer (read-only; do not modify any file). Review the report
/home/agent/workspaces/mats/outbox/boundary-family-2026-10-29.md against its raw results, pre-registration and
code, and write your review as your final answer: numbered findings marked critical / major / minor, each with the
exact location and the correction. Be terse.

Check in particular:
1. Every number in the Summary and section 2 against the raw logs and JSON files (paths below). Recompute ratios
   (e.g. "2.2-4.3x", "0.85-8.2x", "3.7x / 7.6x / 240x", "26-68%").
2. The two pre-registered kill tests against PREREG.md (and its amendment 1) and the hashes in prereg_hashes*.txt:
   were the rules applied as written, were the nets and scripts the registered ones, is the verdict wording right?
   Note: scratch/diag5.py gained a SPLIT family branch after the K1' runs started (it does not touch the registered
   families) — judge whether that matters.
3. Overclaiming or underclaiming: e.g. "gates do not compress" (thresholds used, single-linkage caveat), "factored
   models fail" (only conditional independence given <= 8 PCA dims tested), "cells represent the state well on one net",
   the tails ceiling (8 outputs per level, one net), the width-64 Monte Carlo-at-equal-cost estimate, the scale
   mechanism, the cost estimates. Are the oracle comparisons fair (fit sizes, paired vs independent references,
   sampling floors)?
4. Whether the conclusions about M1 follow from dev-net evidence (3 width-256 nets) and are worded humbly enough.
5. Code correctness of the measurements that carry the conclusions: scratch/diag5.py (representation vs push),
   scratch/diag7.py (gate statistics), scratch/diag8.py (factored models), scratch/diag9.py (tails), lib/lcg.py,
   lib/gmprop.py (push, readout).

Raw results (all under /home/agent/workspaces/mats/work/boundary-family/):
- logs/diag5_100_m24.log, logs/k1_101_m24.log, logs/k1_102_m24.log, logs/split3_101_m24.log (representation vs push)
- runs/diag7/*.json and logs/diag7_*.log (gate statistics, widths 64/128/256)
- logs/kf_101.log, logs/kf_102.log, runs/diag8/*.json (factored models)
- logs/diag9_102.log, runs/diag9/diag9_102.json (tails ceiling)
- runs/diag1/*.json, logs/diag1*.log, logs/diag2_500.log, logs/diag3_500.log, logs/diag4_100.log (one-step oracles)
- logs/lcg256_*.log, runs/lcg256/*.json (deductive LCG at width 256, may still be running)
- DESIGN.md, PREREG.md, prereg_hashes.txt, prereg_hashes_amend1.txt, review/design_critique.md
Context: prior lane report /home/agent/workspaces/mats/outbox/arc-noisy-deep-2026-10-29.md (ARC K3f 2.2e-4
median at depth 32; Kerdock sampler 6.7e-6 at 2.72e11 FLOPs); gate-entropy census
/home/agent/workspaces/mats/outbox/gate-entropy-2026-10-29.md.
Python with torch: /home/agent/.venvs/arc-noisy/bin/python (you may run small read-only checks; do not start long jobs;
the host is heavily loaded).
