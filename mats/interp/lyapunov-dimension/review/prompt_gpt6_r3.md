Focused adversarial check, terse. Read-only; short Python checks allowed (nice -n 19, 2 threads, < 2 min; /home/agent/.venvs/arc-noisy/bin/python).

Report: /home/agent/workspaces/mats/outbox/lyapunov-dimension-2026-10-29.md, section 13 (addendum) and the summary item 3.
Pre-registration: /home/agent/workspaces/mats/work/lyapunov-dimension/pbo/PREREG-PBO.md, PREREG-PBO-predictions.txt, prereg_hashes.txt (written 20:24Z, before the GPU run at 20:39Z; GPU-LOG.md has the times).
Code: pbo/lib/ng3.py (PBO family; ng.py/ng2.py are copies of work/nongauss-family/lib), pbo/scratch/run3.py, pbo/scratch/analyze_pbo.py, pbo/scratch/chart_tails.py.
Results: pbo/runs/gpu/lyapdim1/ra/*.json (single cuts), rb/*.json (composition), pbo/logs/analyze_pbo.log, pbo/logs/chart_tails.log, pbo/logs/gpu_lyapdim1_stream.log.

Check: (1) is PBO correctly implemented as the optimal block of Theorem B (U = orth(Pi M R V_k) equals span(R^+ V_k))? (2) Were the pre-registered rules applied as written (T-A to T-E), including floors and the composition rule? (3) Do the numbers in section 13 and summary item 3 match the json/logs (spot-check at least 8)? (4) Any claim stronger than its support, especially about composition and about the chart-tail agreement. Severity-ordered list, then one line: is the addendum accurate?
