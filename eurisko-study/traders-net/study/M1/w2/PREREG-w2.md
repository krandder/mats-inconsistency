# M1 item (b) pre-registration (worker 2), written 2026-09-27 before any run

Question: does the wealth-net book income equal norm growth under plain SGD (T1 lemma 1:
income = -theta.grad L), what is the Adam residual, and does weight decay add a rent term?

Setup: wealth-net unchanged (Q=64, W=32, d=64, L=2, H=4, B=128, T=24000, first 80% train,
wealth-net streams incl. pi, books = wealth-net `books.account` every step, in-sample random
minibatches). Code copied to `code/` (streams/model/books verbatim), driver `code/train_opt.py`.
Only the optimiser changes. Steps: 10000 per run (fixed below after a <=300-step timing/lr pilot;
the pilot's lr choice is logged in STATUS.md before the main runs).

Arms (3 seeds each: 1, 2, 3):
- `adam`: Adam lr 1e-3 (wealth-net's optimiser; control).
- `sgd`: plain SGD (no momentum), lr eta chosen by the pilot from {0.3, 1, 3} as the largest that
  does not diverge in 300 steps.
- `sgdwd`: same eta, coupled weight decay lambda chosen so eta*lambda*steps ~ 1 (torch SGD
  `weight_decay`).

Units: readouts (64; theta = rw_q, rb_q), MLP neurons (512; theta_out = fc2 column, theta_in =
fc1 row + bias), heads (8; theta = o-weight column block). Per unit, per step, accumulate
- W: books income (wealth-net book, LN denominators frozen, live attention),
- T: tangent/movement = -Q theta.g (true loss gradient, pre-update theta) [= signed movement score x Q],
- N: norm growth Q(|theta_t|^2 - |theta_0|^2)/(2 eta) (eta = lr of the arm),
- rent Q lambda sum|theta|^2 and the second-order term (Q eta/2) sum|g + lambda theta|^2.
Exact identity under (W)SGD: T = N + rent - second_order. Checked per unit (relative max error).

Primary metric: R^2 (OLS with intercept) and Spearman of W on N per granularity at the final
step; for sgdwd also W on N + rent. Secondary: R^2(W, T) (the LN/attention leak), R^2(T, N) (the
Adam residual), eval BCE per family (learning check).

Predictions: sgd R^2(T,N) >= 0.99 and identity error < 1e-4; readouts R^2(W,N) >= 0.99 under sgd
(no LN between rw and z); neurons/heads lower (LN leak). adam R^2(W,N) clearly below sgd
(>= 0.1 lower on neurons). sgdwd: R^2(W,N) below R^2(W,N+rent).

Stop rule: fixed budget; if an sgd seed diverges (NaN) the arm is rerun once at eta/3 for all
seeds and reported as such. If CPU timing makes 9x10k steps exceed 4 h wall, cut steps to fit
(logged in STATUS.md) or submit one G1 job (<= $1.5).
Report 3 seeds mean +- sd (sample sd).

## Amendment 1 (2026-09-28 00:25Z, after the main runs; EXPLORATORY, post hoc)

The pilot rule picked eta=3; at that eta the second-order term (Q eta/2) sum|g|^2 is 0.6x (readouts)
to 2.9x (neurons) of |T|, so R^2(T,N) failed the prediction (gpu/analysis/summary.csv). The pilot
(300 steps, gpu/pilot) shows the term scaling with eta. Exploratory arm, labelled as such in the
audit: `sgd` and `sgdwd` at eta=0.3 (wd = 1/(0.3*10000) = 3.33e-4), seeds 1-3, 10k steps, same
code (`code/run_small.sh`), results in `gpu-eta03/`. Prediction: second_share for neurons < 0.3
and R^2(T,N) >= 0.9 for neurons at eta=0.3. No other change.
