# PREREG M1 item (a), worker 1 — wealth vs norm growth and movement (read-only on wealth-net)

Written 2026-09-27 23:20Z before any run.

**Data (read-only):** `wealth-net/results/train-s{1,2,3}/snapshots/step-*.json` (per-unit cumulative
wealth every 100 steps), `wealth-net/results/weights/train-s*-step{8000..30000}.pt` (every 2k) and
`train-s*.pt` (final, 31.1k), `frozen-s*.npz` (control). Adam, lr eta=1e-3, B=128, Q=64.

**Identity under test (T1 lemma 1, settled premise):** for a unit whose contribution is
1-homogeneous in its parameters theta (neuron: fc1 row+bias, or fc2 column; head: V rows+bias or
O columns; readout: rw[q], rb[q]), income per step = -Q * theta . grad L_mean (books' gradient, LN
denominators frozen). Under plain SGD, Delta||theta||^2/(2 eta) = -theta.grad L + eta||grad||^2/2,
so wealth ~ Q * Delta||theta||^2/(2 eta). Under Adam the step is preconditioned, so the identity
breaks; item (a) measures by how much.

**Units:** neurons (512), heads (8), readouts (64). Per seed.

**Predictors (per unit, per interval [t_k, t_{k+1}] between saved weights, and cumulative 8k->final):**
- X_norm_out = Q * Delta||theta_out||^2 / (2 eta) (theta_out = fc2 column / O columns / rw,rb)
- X_norm_in  = Q * Delta||theta_in||^2 / (2 eta) (fc1 row+bias / V rows+bias)
- X_move = Q * sum_k theta_k . (theta_{k+1} - theta_k) / eta on out-weights (discrete movement,
  weights moving away from zero; Sanh et al. https://arxiv.org/abs/2005.07683 accumulate -grad.W;
  with checkpoints every 2k steps only the displacement form is available)
- X_grad = sampled gradient-form movement score: at each checkpoint, mean over 8 train batches
  (B=128) of -Q theta.grad L (live LN, i.e. the optimiser's gradient), trapezoid-integrated over the
  interval. Also computed with the books' frozen-LN gradient (X_grad_books) to isolate the LN
  convention from Adam.
**Target:** Y = wealth(t_{k+1}) - wealth(t_k) from snapshots (total over families).

**Metrics:** OLS R^2 (with intercept), fitted slope vs the SGD expectation 1, Spearman rho,
relative residual ||Y - fit|| / ||Y||, and sign agreement. Pooled over intervals and per interval;
per seed, reported mean ± sd over 3 seeds.
Control: frozen runs (no updates, so Delta theta = 0): wealth per step there is income that no
norm growth can explain; reported as a ratio to the train-run income per step at the final ckpt.
Instantaneous check: at each checkpoint, books income (account()) vs -Q theta.grad L_books on the
same batch; should agree to float error (validates the identity; any gap is a bug).

**Seeds:** 1, 2, 3 (the wealth-net seeds). No new training.

**Stop / decision rule:** none beyond reporting. Reading: if pooled R^2(Y ~ X_norm_out) >= 0.9 in
3/3 seeds, Adam-era wealth is essentially norm growth; if R^2(Y ~ X_grad) >> R^2(Y ~ X_norm_out),
the gap is Adam preconditioning, not the books. Worker 2's SGD reruns (item b) test the SGD case.

**Compute:** CPU, 1 thread, one seed at a time (load ~26 on 16 CPUs at 23:14Z). No GPU.
