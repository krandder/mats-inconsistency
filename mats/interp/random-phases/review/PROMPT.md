You are reviewing a short research study before its claims are reported. Be a rigorous, skeptical reviewer. Work
read-only on the study directory (do not edit anything there; put any scratch files under /tmp/rp_review/). The host
is overloaded and the study's own 3 job workers are still running: run at most ONE process at a time, each under
`nice -n 19` with OMP_NUM_THREADS=1, and keep any computation you do under about 3 CPU-minutes in total (reading code,
regenerating tables with analyze.py into /tmp/rp_review/, and small numerical spot checks are fine; do not rerun the
experiments). Use the Python at /home/agent/.venvs/arc-noisy/bin/python.
State of the runs when this review started: all width-256 forgetting runs and the width-256 backward runs at
sigma = 0 and 0.01 (plus some at 0.1) are done; the rest of the width-256 backward runs (sigma = 0.1, 0.001) and all
width-64 runs are still in the queue (jobs_main.txt; finished lines in jobs_main.txt.done). The draft report has the
width-256 numbers; sections 5-6 are not yet written. Review what exists.

Study directory: /home/agent/workspaces/mats/work/random-phases
- PREREG.md (predictions written before the main runs; hashes and times in prereg_hashes.txt; one amendment)
- mf.py (mean-field theory), mf_predictions.json, pnets.py (nets), branch.py (forgetting by nested sampling,
  Lyapunov exponents, gate/collapse statistics), backward.py (truth, stand-ins pushed through the last k layers,
  input-law swaps, ARC's kprop K = 2 started from a stand-in), analyze.py -> results/tables.md
- results/branch/*.json, results/backward/*.json (raw results)
- Draft report: /home/agent/workspaces/mats/outbox/random-net-phases-2026-10-29.md
- Context (read only if needed): work/arc-noisy (ARC's He-ReLU family with noise; ARC's kprop code under
  work/arc-noisy/ext/mlp_cumulant_propagation), work/cm-interp-R (earlier LN nets).

The question the study answers (from the research lead): in deep random nets, is the input forgotten in the chaotic,
ordered and critical phases (with small activation noise sigma = 0.001-0.1 as a fraction of the activation scale), can
an input-independent "backward" description of the last k layers give E[output], and what does that imply for a
sample-free estimator that matches sampling on deep random MLPs at depths growing with width?

Please check, in this order, and rank findings by severity (critical / high / medium / low):
1. Correctness of the estimators in code: the ANOVA share R(m -> L) and its debiasing and jackknife; the whitened
   between/within statistic S and the Gaussian-proxy TV; the Lyapunov QR; the stand-in definitions (mean-field MF,
   moment-matched MM, scale-matched MFr) and their pushes; the nMSE debiasing; the use of ARC's linear_kprop /
   nonlin_kprop with bias, tanh, noise on K2 and a non-input starting cumulant; the FLOP accounting for the equal-cost
   comparison with sampling; the mean-field recursions in mf.py (noise enters the diagonal only; branch copies).
2. Whether each claim in the draft report follows from the raw results (regenerate the tables with analyze.py; spot-
   check a few numbers against the JSONs), whether every number states regime, width, depth and sigma, and whether
   any claim is stronger than the evidence (seeds, widths, resolution limits, multiple comparisons).
3. Whether the prediction-vs-result scoring is honest with respect to PREREG.md (including the declared smoke tests
   and the amendment), and whether anything was changed after results were seen without being declared.
4. Whether the stated implications for the target (a sample-free estimator matching sampling at depths growing
   with width) are justified, overstated, or missing an obvious counter-argument.
Write your review to /tmp/rp_review/review.md (start with "Model: <your model name>"), with each finding citing file
and line or table cell, and what would fix it. Keep it under 1,500 words.
