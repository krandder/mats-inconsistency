# wealth-net: wealth accounting inside a small neural network

An experiment for Kelvin and Jessica Taylor (2026-09-27). We keep books on an ordinary
training run and watch; nothing is taxed and no routing changes.

## The idea

A softmax (here: sigmoid, the two-outcome softmax) turns a logit into a price, exactly like
an LMSR market maker turns the net position into a price. The logit is a sum of
contributions from the units that feed it. Cross-entropy's gradient at the logit is
p - y, which is the market settlement with the sign flipped: a position of size z pays
z * (y - p) when the outcome y is revealed.

So read every unit as a trader and the computation graph as a supply chain.

* **Output layer.** Question q's readout contributes z_q to its own logit and is settled
  z_q * (y_q - p_q) per example. Summed over questions this is the total settlement R.
* **Inside.** A consumer pays each of its suppliers that supplier's first-order share of the
  consumer's own revenue: activation * d(consumer revenue)/d(activation). Because the chain
  rule composes these shares, one backward pass of R (with y - p held fixed, and separate
  from the loss backward pass) gives every unit's revenue a * dR/da at once.
* **Wealth** of a unit = its cumulative revenue over training.
* **Partial ownership** is automatic: a unit feeding many consumers is paid by all of them.
* **Big traders**: any set of units in the same layer (a head = its value channels, a layer
  = its heads or neurons plus its output bias) has wealth equal to the sum of its members.
  Across layers a set's net wealth is what it receives from outside minus what it pays to
  outside; payments between members cancel.
* **Conservation.** Any complete cut of the graph (a residual-stream position in a
  transformer) should receive R in total, up to first-order error. Constant inputs
  (biases, LayerNorm shifts, question-id embeddings) are suppliers too and are booked.
  Where a consumer is not homogeneous of degree one in its inputs (the attention softmax),
  its payments to suppliers do not add up to its revenue; that gap is the reported residual.
* **Reading of the number.** Revenue = -a * dL/da at the logit-linearised loss: to first
  order, it is how much the loss would rise if the unit were switched off. A unit with
  persistently negative wealth ("parasite") would, to first order, improve the loss if
  removed.
* **Compute cost** (optional, default 0): a flat charge per activation per example.

Two conventions the books need:

1. LayerNorm denominators are treated as constants in the books pass (LayerNorm is
   scale-invariant, so activation * gradient through it is zero and every inner cut
   would look bankrupt). The values are unchanged; only the books' derivatives differ.
2. The frozen-attention variant (attention patterns treated as constants) is used only as
   an exact check of the implementation: with it, conservation holds to float precision.

## Files

* `streams.py`: question families from logical-soup (periodic, LFSR-16, CA rule 30,
  CA rule 110, primality of n, pi-digit parity, A AND B, A OR B), deterministic per seed.
* `model.py`: 2-layer pre-LN transformer, d=64, 4 heads, ReLU MLP 4x, one token per
  question (window of the last 32 outcomes + question-id embedding), one logit per
  question; plus a per-question logistic baseline.
* `books.py`: the accounting (per unit, split by the question family that paid).
* `test_books.py`: conservation test on a tiny float64 model (exact with frozen attention).
* `train.py`: training + books, snapshot every 100 steps to `<out>/<tag>/snapshots/`,
  rolling `<out>/<tag>/latest.json`; modes `train`, `frozen` (trained weights, no updates),
  `random` (fresh init, no updates).
* `analyze.py`: builds `results/index.html` and `results/summary.json` from snapshots.
* `run_gpu.sh`: what runs on the GPU VM (3 seeds in parallel, then controls).
* `sync_loop.sh`: pulls snapshots from the VM every 2 minutes and republishes the page.

## Run

    PYTHONPATH=.pylib python3 test_books.py
    PYTHONPATH=.pylib python3 train.py --steps 60000 --max-min 50 --seed 1 --out results
    PYTHONPATH=.pylib python3 analyze.py results
