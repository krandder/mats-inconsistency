You are an adversarial reviewer (different model family from the author) of the CLAIMS in "Part 3: a better
sample-free readout of ARC's own descriptions" at the end of
/home/agent/workspaces/mats/outbox/arc-noisy-deep-2026-10-29.md (Parts 1-2 above it are reviewed; use them only
for context). Read-only everywhere except /tmp/arcn_review2/ (write scratch and your review there). Python:
/home/agent/.venvs/arc-noisy/bin/python (OMP_NUM_THREADS=1, at most 2 processes, each check under ~10 CPU-minutes; the
machine is heavily loaded; no GPU).

Pre-registration: /home/agent/workspaces/mats/work/arc-noisy/tails/PREREG-TAILS-p3.md (hashes in
../prereg_hashes.txt). Code: tails/p3_explore.py (exploration on development nets 100-101), tails/p3_score.py (kill test
and confirmatory test), tails/deduce.py (readouts: radial = chi-mixture saddlepoint, saddle, nig_*), tails/kprop_tail.py
and tails/run_kprop_tail.py (ARC's kprop with heaviside readout and the description's moments), tails/truth.py and
tails/thresholds.py (truth; the part 3 CPU runs used smaller sizes, see the prereg). Outputs: tails/results/p3_killtest.txt,
tails/results/p3_confirm.txt; raw: tails/runs/fresh3/ (fresh-net thresholds and truths), tails/runs/kp_fresh3/ and
tails/runs/kp/ (kprop outputs), tails/runs/kp_x1/ (exploration).

Check: (1) every number and verdict in the Part 3 summary and sections against the raw outputs and the prereg (kill bar,
primary confirmatory test, secondary results, costs, better-than-zero); (2) whether the comparison is fair to ARC (same
description, ARC's run_lpe settings, ARC's FLOP counter, the heaviside readout as ARC does it; and whether comparing with
the matched K3fa description rather than ARC's cheaper K3f is stated honestly, with costs); (3) the truth quality of the
fresh nets under the reduced CPU sizes (provisional events; does any claim depend on them?); (4) anything overstated or
missing a caveat, including the "sample-free" and "deductive" labels and what was seen before the prereg. Write
/tmp/arcn_review2/review_claims3.md: numbered findings by severity (critical / major / minor) with evidence and the exact
fix; then what you checked and found correct. Be concise.
