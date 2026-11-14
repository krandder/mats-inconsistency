You are a rigorous reviewer checking the claims of a research report against its code and raw outputs (read-only; do not
modify any file). Write your review as your final answer (it will be saved to a file). Number your findings and mark
each critical / major / minor. For each claim in the report's Summary, say whether the raw outputs support it as
worded; recompute the key numbers yourself from the JSON/NPZ outputs where you can (you may run read-only Python with
/home/agent/.venvs/arc-noisy/bin/python; write only under /tmp). Be concrete and terse.

Context: MATS noisy-channel interpretability program; ARC's critical He-ReLU MLPs with small activation noise
(sigma = 0.001). Kelvin asked for "a family that is fully non-gaussian, even if its bad at the input, and hope that it
at least scales on depth", and later "How can you tell it's not just that we haven't found the right structure that has
low-error per layer, or that can even self-correct?". This lane fitted oracle families to true Monte Carlo paths at a
cut, ran the exact remaining layers, and measured the output error as depth grows; plus composition (re-fitting at
every layer), a fixed-L/n width series, Lyapunov / sensitivity spectra, a learned compressor, and descriptive
statistics of the direction law.

Files (all under /home/agent/workspaces/mats/work/nongauss-family/ unless absolute):
- Report under review: /home/agent/workspaces/mats/outbox/nongauss-family-2026-10-29.md
- Pre-registration (hashed before the runs; four amendments; deviations list): PREREG.md, prereg_hashes.txt;
  design: DESIGN.md; design critique: review/design_critique.md
- Code: lib/ng.py (families, propagation, scores), lib/ng2.py (Amendment families), scratch/run.py, scratch/run2.py,
  scratch/amp.py (sensitivity spectra), scratch/lyap.py (Lyapunov), scratch/comp.py (compressor), scratch/e0.py,
  scratch/analyze.py, scratch/analyze2.py, scratch/e0_table.py
- Raw outputs: runs/gpu/ngfam1/ (r1 single cuts 256x64; r2 composition; r3 128x32 and 512x128; r4 linearised
  composition), runs/gpu/ngfam2c/a1 (Amendment 1), runs/gpu/ngfam3/a3 (Amendment 3), runs/gpu/ngfam4/a2 (T5a, T5b,
  T6), runs/gpu/ngfam5t2/a4 (Amendment 4), runs/cpu_a2 (T5b at 512 on CPU), runs/e0 (descriptive);
  generated tables: results/tables.md, results/tables2.md, results/e0_table.md
- Parallel oracle study by another lane (Gaussian cells at width 256 depth 32, nets 101-110):
  /home/agent/workspaces/mats/work/boundary-family/logs/diag10_1*.log

Check in particular:
1. Are the pre-registered rules applied exactly as written (thresholds, interval rule, inconclusive region, replicate
   handling)? Any verdict that is wrong? Are deviations disclosed?
2. Are floors handled correctly, and could any reported difference be noise (replicates, four nets, 512 runs with
   smaller fit sets, the noisy B = 2C(2^21) - C(2^20) extrapolation)?
3. Is "fully non-Gaussian" true for every family called so, and is the report clear that PB64 / PB128 and the
   compressor are oracle constructs (structured sampling / trained on samples), not mechanistic families?
4. Does the report over-generalize (oracle fits vs deductive families; one noise level; L/n = 1/4 only; four nets;
   compressor on one net)?
5. Are the interpretations (growth after the cut, the PB ladder, polar chart, attractor statistics, width series,
   Lyapunov and sensitivity counts, compressor, composition) supported by the numbers?
6. Anything missing that a reader needs to judge the result.
