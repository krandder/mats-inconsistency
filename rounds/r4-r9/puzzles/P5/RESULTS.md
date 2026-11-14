# P5: private memory slots priced online (maze path-parity, answered only at T)

Status (2026-09-29 15:00Z): **stopped at gate G0, no P5 verdict.** Tuned A1 and A2 stay near chance (50%) on 9x9
MP-sparse validation, with and without the curriculum at 12k iterations. The pre-registered rule is to stop and report
the task as too hard at this scale. A1 also misses the dense validity bar. No stage-1 run of any market arm took place.
The derivation below was written before any GPU run (12:45Z).

## Derivation (written before the runs)

Setup. Traders i = 1..N share the belief logit b at T and each owns a bank c_i (K channels per cell).
One step t: every trader reads (x, tanh(b/4), tanh(c_i)) and emits db_i, writes w_{i->j} into every bank j and a gate g_i.
Update: c_i(t+1) = (1 - g_i(t)) c_i(t) + sum_j w_{j->i}(t); b(t+1) = b(t) + Sigma(t), Sigma = sum_i s_i db_i (masked to T).
The per-step score is the LMSR log-score change dscore(t) = y Sigma - (softplus(b + Sigma) - softplus(b)); summed over t it
telescopes to the final log score minus the prior's. Belief pay pay_i = s_i db_i (y - pbar) with
pbar = (softplus(b + Sigma) - softplus(b)) / Sigma, so sum_i pay_i = dscore exactly (Aumann-Shapley on a linear path).

1. Own-profit gradient = BPTT of the owner's share. Hold prices fixed (stop-gradient) and set phi = 1 and pointwise prices.
   Define Lambda_i(t) = d/dc_i(t) [ pay_i(t) + sum_j <w_{i->j}(t), sg Lambda_j(t+1)> + <(1 - g_i(t)) c_i(t), sg Lambda_i(t+1)> ].
   Differentiating c_j(t+1) = (1 - g_j) c_j + sum_k w_{k->j} with respect to c_i(t) gives exactly these terms: trader i's
   step-t outputs depend on c_i(t) only through its own module (the owner reads only its bank), and c_i(t) reaches c_i(t+1)
   directly through (1 - g_i) c_i (including the gate's own dependence on c_i, which is why the gate term sits inside the
   bracket rather than as a constant (1 - g) multiplier). This is the reverse-mode recursion for the adjoint of c_i with
   respect to sum_t sum_k pay_k(t) restricted to the paths that start at trader i's own modules, i.e. Lambda is the bank
   adjoint of the telescoped score, with b detached (b enters each step only as an input, so detaching it removes the b
   path from the adjoint exactly as in A2t). The trader's own-profit objective at step t,
   pay_i(t) + sum_j <w_{i->j}(t), sg Lambda_j(t+1)> (royalties received for its writes) + <(1-g_i)c_i, sg Lambda_i(t+1)>
   (the value it keeps in its own bank) minus royalties it pays (constant in its own parameters given prices), has parameter
   gradient equal to the step-t term of the BPTT gradient of the global telescoped score with respect to trader i's
   parameters. Summing over t gives the full BPTT gradient restricted to trader i's parameters.
2. Royalties telescope. Owner i pays sum_j <w_{j->i}(t), Lambda_i(t+1)> at every step. Since
   sum_j w_{j->i}(t) = c_i(t+1) - (1 - g_i(t)) c_i(t), the payments summed over writers and time are
   sum_t <c_i(t+1) - (1-g_i(t)) c_i(t), Lambda_i(t+1)>, which regroups to sum_t <c_i(t), pi_i(t)> with
   pi_i(t) = Lambda_i(t) - (1-g_i(t)) Lambda_i(t+1) (the pointwise, own-pay part of the price) plus the boundary terms at
   t = 0 (c = 0) and t = D (Lambda(D) = 0). So money moved for memory equals the value the owner's reads extract.
3. Hence offline P5 (A5) differs from A2t only by (i) IG secant Jacobians (m = 4) instead of pointwise ones, (ii) the own-AS
   split of the score change at T among the N = 4 parallel traders (each trader follows its own pay, not the total), and
   (iii) the wealth caps s_i < 1 when they bind. b is detached in both. Test (tests.py test_adjoint): with N = 1, pointwise
   prices and no caps, A5 equals A2t autograd to 1e-5 in float64; A6 given the true global adjoint in place of its synthetic
   gradient head also equals A2t.
4. Online arms. A7 replaces the swept Lambda_i(t+1) by the owner's posted Lambda_hat_i(t+1) (a 1x1 head, TD(0) towards the
   one-step expression evaluated at the posted t+2 price), and settles royalties at write time, so the training graph is
   one step deep (O(1) memory in D). A6 uses the same head as a DNI synthetic gradient for the global score (no owners, no
   ledger). A4 keeps only pay_i(t) in the bracket (no recursion): a one-hop truncated gradient, predicted to be about A3
   (belief-only) beyond paths of 2 cells.

Predicted before the runs: A5 ~ A2t <= A2; A4 ~ A3.

## Implementation choices and deviations from the spec (recorded before the results)

- Ledger caps use outcome-free wealth. The first smoke of A7 reached train accuracy 1.0 within 20 iterations: realised
  wealth depends on y, so caps computed from it leak the label into b during training (a trader that bet wrong goes broke
  and is capped), while test runs with s = 1. The caps now use the guaranteed wealth G = W0 + min over y of the cumulative
  belief pay (+ royalty cash at posted, y-free prices for A7/A7p, whose royalties are also scaled to G). Royalties at
  realised-y prices (A4 one-hop IG, A5/A8 swept adjoints) are scaled to realised wealth and never feed the caps, so their
  payers can end below zero (reported as neg_wealth). tests.py test_caps_label_free checks acc(y) + acc(1 - y) = 1 for
  A3, A4, A5, A7, A7p from the same weights.
- DNI/A7 price heads start at zero (zero-initialised output layer, as in the DNI paper), their input is detached from the
  body, and body and head gradients are clipped separately.
- Main arms run 3 seeds, not 5. The benchmark (30 training iterations per arm, 200 evaluation iterations per group)
  projected about 20 GPU-h of evaluation at full sets (so the long-path and stability sets are halved, as the spec
  allows) and 16-20 VM-hours in total at 5 seeds, against 17.5 VM-hours ($7 at the $0.40/h estimate). 3 seeds is the
  spec's minimum. Stage-2 arms keep 3 seeds.
- Order of tuning. A1 tuning at 1500 iterations stayed at chance for all 4 configs (val 0.50-0.51), so the gate is
  likely to need the curriculum. The driver therefore tunes the gate arms (A1, A1w, A2) first, runs G0, and only then
  tunes A2t, A5, A6 and A7 (with the curriculum when G0 switched to it, since every arm then trains with it).

## Results

Checked at 2026-09-29 14:58Z from `results/vm` (161 files synced from trader-cat-p5-1, each verified with sha256).
Data: `results/summary.json` (all numbers), `results/chart.json` (chart: val9 accuracy by S-T distance),
`results/gate_diag.json`. Rebuild with `python3 gate_diag.py > results/gate_diag.json && python3 summarize_gate.py results/gate_diag.json`.

### Phase 0 (CPU and VM)
All unit tests pass (`results/tests.json`, `results/vm/tests_vm.json`):

| Test | Result |
|---|---|
| BFS against brute force | 0 failures in 200 mazes |
| Neighbourhood logistic, held out, d > 4 | 0.503 (bar 0.52 or less) |
| Pay completeness | error 1e-14 |
| Ledger conservation | 2e-8 |
| IG completeness | 0.15 at m=4, 0.007 at m=64 |
| Grouped conv against the loop version | exact |
| A5 (N=1, pointwise) against A2t | 2e-16 |
| A6 with the true adjoint against A2t | 2e-16 |
| Label-free caps | acc(y) + acc(1-y) = 1 |
| 7x7 smoke run | every arm |

9x9 S-T distances: median 18, p90 39, L_train = max 70.

### Tuning
Four configs per arm, seed 0, 1500 iterations; the score is the mean of val9 and val13. Every config is at chance:

| Arm | Scores (4 configs) | Picked |
|---|---|---|
| A1 | 0.497, 0.497, 0.501, 0.509 | lr 1e-3, alpha 0.5 |
| A1w | 0.494, 0.508, 0.499, 0.497 | lr 3e-4, alpha 0.5 |
| A2 | 0.526, 0.521, 0.501, 0.511 | lr 3e-4, alpha 0.01 |

### Gate G0 (bar: 90% or more on val9; seed 0 by design; 1000 mazes; Wilson 95% interval over mazes)

| Run | val9 | 95% CI | d=1 | d=2 | d=3 | d 4-5 | d 6-10 | d 11-20 | d 21-40 | d 41+ |
|---|---|---|---|---|---|---|---|---|---|---|
| A1, 4000 it | 0.528 | 0.497-0.559 | 1.00 | 0.57 | 0.58 | 0.59 | 0.41 | 0.54 | 0.52 | 0.54 |
| A2, 4000 it | 0.484 | 0.453-0.515 | 0.52 | 0.54 | 0.48 | 0.54 | 0.45 | 0.45 | 0.50 | 0.49 |
| A1, spec curriculum, 12k it | 0.517 | 0.486-0.548 | 0.95 | 0.57 | 0.61 | 0.62 | 0.43 | 0.53 | 0.48 | 0.53 |
| A2, spec curriculum, 12k it | 0.533 | 0.502-0.564 | 0.90 | 0.46 | 0.55 | 0.59 | 0.51 | 0.53 | 0.50 | 0.58 |

Result: G0 fails. The baselines learn only d=1 (the answer is T's own token) and stay at chance for every path of two or
more cells. Final training accuracy (running mean) is 0.50-0.52. The spec's curriculum (distance uniform over 1..L_train)
adds no short paths, because the natural distribution is already weighted towards short paths (median 18).

### Dense validity bar (A1, 9x9 MP-dense, iteration 128; bar: 95% or more exact-maze)
- A1, 4000 it: per-cell 0.620, exact-maze 0.000. **Misses the bar.**
- Diagnostic, 12k it: per-cell 0.847, exact-maze 0.047.

Even with a label at every cell, the recurrent conv net does not learn to carry prefix parity along a 2D path at this
scale. By the spec's own rule, this is a validity kill ("design broken, not the idea").

### Diagnostics outside the spec (seed 0; not part of any verdict)
An incremental curriculum (the maximum S-T distance grows from 2 to L_train over the first 75% of training, then the
natural distribution; flag `--curr_inc`) at 12k iterations:

| Run | val9 | 95% CI | val13 |
|---|---|---|---|
| A1 | 0.555 | 0.524-0.586 | 0.534 |
| A1w | 0.603 | 0.572-0.633 | 0.580 |

A1w reaches 0.96-1.00 for d of 3 or less, 0.92 at d 4-5 and 0.75 at d 6-10, then falls to chance beyond d 10. So the
parity chain can be learned over short paths when the lengths grow slowly, but not over the 20-70-cell paths that most
9x9 pairs need within 12k iterations.

### Interpretation
- This is a task finding, not a P5 result. Parity over a 2D path, supervised only at the end (and densely, at this
  budget), is too hard for the baselines at 9x9 with D = 64 and 4k-12k iterations.
- This matches the known hardness of end-supervised parity (Shalev-Shwartz et al. 2017).
- No statement about A5, A6 or A7 is made or implied.
- A redesign would need one of these:
  - shorter paths (for example 5x5, or S-T distance capped at about 8, with extrapolation measured from there);
  - an incremental length curriculum written into the spec;
  - a much larger budget.
  Those choices belong to whoever owns the design, not to this run.

### Cost and resources
- One VM, trader-cat-p5-1 (us-central1-c, g2-standard-4 + L4 spot), from 12:37 to 14:57Z: 2.33 h, est. $0.93 at the
  $0.40/h ceiling. The row is in `../../GPU-LEDGER.md`.
- Before deletion: sha256 manifest built on the VM, 161 files verified locally.
- After deletion: `gcloud compute instances list` and `gcloud compute disks list` with `--filter="name~^trader-cat-p5"`
  both returned 0 items (checked at 14:58Z).

### Deviations (in addition to the list above)
- Output directory is `puzzles/P5/` (as the task assigned), not `p5-private-slots/`. The runner is `driver.py`
  with `gpu.sh` (launcher and watchdog), not run_grid.sh and watchdog.sh.
- The 3-seed stage, stage 2, memory against depth, FLOPs and stability passes did not run: G0 failed and the spec stops
  there.
- The gate diagnostics (incremental curriculum, and dense at 12k) were added after G0 failed, used idle VM time
  (about 15 min), and are labelled as outside the spec.
