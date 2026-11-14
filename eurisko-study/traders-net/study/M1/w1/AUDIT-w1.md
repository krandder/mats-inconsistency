# M1 (a): is wealth-net wealth just norm growth or movement? (worker 1)

Read-only on wealth-net (Adam 1e-3, 3 seeds, 31.1k steps). Pre-registration: `w1/PREREG-w1.md`.
Code: `w1/norm_growth.py` (per seed), `w1/analyze_a.py` (fits). Checked at 2026-09-27 23:35Z.
All numbers are mean ± sample sd over seeds 1-3; sources are under `M1/w1/results/`.
Rerun: `cd M1/w1 && for s in 1 2 3; do PYTHONPATH=<wealth-net>/.pylib python3 norm_growth.py --seed $s; done && python3 analyze_a.py`.

## 1. Identity (exact): wealth-net books = movement score under a modified gradient

At every saved checkpoint (8k..final and the reconstructed init), on 2 train batches per checkpoint,
the books' per-unit income `account()` equals `-Q * theta . grad L` computed with the books' own
forward (LayerNorm denominators detached), on the input side and the output side of every unit
(`results/inst_summary.csv`, rows `books_ln`):
relative error 2.6e-7 (neurons), 4.8e-7 (heads), 2.9e-7 (readouts), max 1.3e-6.
So the books are, exactly, the per-step gradient-form movement score (Sanh et al.
https://arxiv.org/abs/2005.07683: S = -sum_t W . dL/dW) times Q, **but under the frozen-LN
gradient, not the optimiser's gradient**. With the optimiser's gradient (live LN, rows `live_ln`):

| unit | rel. error vs books | corr with books |
|---|---|---|
| readouts (64) | 2.9e-7 (identical: no LN between rw and z) | 1.0000 |
| MLP neurons (512) | 0.179 ± 0.013 (max 0.47) | 0.982 ± 0.003 |
| heads (8) | 0.41 ± 0.14 (max 1.59) | 0.925 ± 0.045 |

Reading: for readouts wealth IS movement score; for neurons and heads it is movement score
plus an LN-convention term of 18% / 41% of the per-batch income. (Worker 4, item (d), splits this.)

## 2. Regression on norm growth and displacement-form movement (Adam era)

Y = wealth change of each unit; predictors from the saved weights (every 2k steps; init reconstructed
from the seed, unverified against a saved step-0 file, which does not exist). X_norm_out =
Q Delta||theta_out||^2 / (2 eta); X_move_out = Q sum_k theta_k . Delta theta_k / eta (2k-step
checkpoints). Under SGD the expected slope of Y on X_norm is 1. `results/fits_summary.csv`:

| window | unit | R^2 (Y ~ X_norm_out) | slope (SGD: 1) | Spearman | R^2 (Y ~ X_move_out) | sign agreement |
|---|---|---|---|---|---|---|
| init -> final | neurons | 0.19 ± 0.15 | 4e-4 ± 2e-4 | 0.56 ± 0.09 | 0.14 ± 0.13 | 0.62 |
| 8k -> final | neurons | 0.12 ± 0.11 | 3e-4 ± 2e-4 | 0.47 ± 0.10 | 0.11 ± 0.10 | 0.52 |
| 2k intervals, pooled | neurons | 0.13 ± 0.10 | 3e-4 ± 2e-4 | 0.49 ± 0.09 | 0.12 ± 0.10 | 0.51 |
| init -> final | heads | 0.56 ± 0.43 | 1e-3 ± 6e-4 | 0.68 ± 0.20 | 0.48 ± 0.45 | 0.08 |
| init -> final | readouts | 0.11 ± 0.02 | 5e-5 | 0.31 ± 0.11 | 0.09 ± 0.01 | 0.89 |
| init -> 8k | readouts | 0.61 ± 0.03 | 2.5e-4 | 0.80 ± 0.01 | 0.26 ± 0.06 | 0.97 |

- Under Adam, wealth is **not** norm growth: R^2 <= 0.2 for neurons in every window, and the fitted
  slope is ~3000x below the SGD value (Adam's per-coordinate step is ~eta, not eta*g, so norms move
  far more than the books pay). The ranks agree moderately (Spearman ~0.5), mostly through size:
  final ||theta_out|| alone gives Spearman 0.56 ± 0.09 with neuron wealth (row X_normT_out).
- Signs do not agree beyond chance for neurons after 8k (0.51-0.52): most neuron norms grow while
  about 40% of neurons have negative wealth. Heads: all 8 grow in norm, 7-8 of 8 have negative wealth
  (sign agreement 0.04-0.08).
- Readouts early in training (init -> 8k) are the one place norm growth tracks wealth (R^2 0.61).
- Endpoint-sampled gradient movement (the books' own quantity measured at the two ends of each 2k
  interval, trapezoid rule) explains nothing: R^2 0.002 ± 0.002 for neurons (row X_grad_books_out).
  Per-step income is dominated by oscillation around the optimum; wealth is its small time-average.

## 3. Frozen control: income with zero movement

With the final weights frozen (wealth-net `frozen-s*`, 500 steps on training batches), mean |income per
step| per neuron is 43.6 ± 14.4 times the trained run's over its last 1.1k steps, and it is unrelated
to it (Spearman -0.07 ± 0.08) (`results/frozen_control.csv`). A frozen snapshot is not a quiet
null: training keeps books near zero-mean (Adam cancels any persistent -theta.g), freezing lets the
snapshot's residual gradient accrue linearly. Do not use frozen-run wealth as a "no-learning" baseline
for magnitudes; use it only for ranks, or use the random-init control.

## 4. What this means for the currency (input to CURRENCY.lock)

- The "wealth = norm growth" premise is an SGD statement (T1 lemma 1); worker 2's SGD/SGD+wd
  reruns (`M1/w2/`) test it directly. Under Adam, wealth-net's wealth is exactly the per-step
  movement score (frozen-LN gradient), not norm growth, and only about 12-19% of its neuron variance is
  norm growth.
- Wealth-net books and the movement score differ only by the LN convention (18% neurons, 41% heads
  per batch, corr 0.98 / 0.93). Any claim that "wealth adds something beyond movement pruning" must
  therefore come from the settlement (prequential / LMSR / baseline, item c), not from v1 books.
- Not done here (duplicate of worker 2's adam arm): the cumulative live-LN movement score per step.
  `w1/replay.py` is written but was not run.

Unverified: the reconstructed init (seeded Net() on CPU; the VM ran torch on CUDA, but init is on
CPU before .to(device), so it should match); affects only the "init ->" rows.
