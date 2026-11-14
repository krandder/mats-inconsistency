# M3-w2 pre-registration: which currency survives a regime shift, for committed traders only

Written 2026-09-28 before any run. Scope follows CURRENCY.lock (M1, 01:40Z): no currency for generic
neurons; only committed traders that post a price on a question, settled by exact LMSR log score at the
trade price, prequentially. This asks the Q6 lock-in question for that currency.

**Stream.** `S0/substrate/streams_ext.make_streams(Q=64, T=40000, seed=s, heldout=8, regime_every=10000,
regime_frac=0.25)`; boundaries at 10k/20k/30k, 16 of 64 questions switch at each. Held-out questions are
scored only while active.

**Traders (35 per question, all committed, bounded).** prior = running base rate (n1+1)/(n+2) (the lock's
prior); lag-k copy traders k=1..32 posting 0.95/0.05; an LFSR16 parity trader (y_{t-16}^y_{t-14}^y_{t-13}^y_{t-11},
0.95/0.05); an online logistic learner on the 32-bit window (SGD lr 0.1, clipped to [0.02, 0.98]).
Trader predictions do not depend on the arm.

**Pool.** Per question, price p = sum_i w_i p_i with w = softmax(v). Settlement of trader i at step t:
s_i = log p_i(y) - log p(y) (Kelly bettor's log-wealth change = LMSR log-score gain against the price).

**Arms (currency -> v).**
- kelly: v += s (cumulative settlement; = Bayes mixture)
- ema100, ema1k, ema10k: v = lambda v + s, half-lives 100/1000/10000 steps (depreciation of log wealth; the
  plan's "depreciation delta" is this family)
- fs1e-3, fs1e-2: kelly then fixed share w <- (1-a) w + a/N (Herbster-Warmuth)
- win500: v = settlement summed over the previous complete 500-step window only (disjoint-window income)
- controls: uniform (v = 0); shuffled (kelly settled against a random label); grad0.1, grad1: v trained
  by online SGD on the pool's log loss (dv_i = lr * w_i (r_i - 1), r_i = p_i(y)/p(y)), the analog of
  gating by recent gradient credit.

**Metrics (per seed, then mean +- sd over seeds 1, 2, 3).**
1. Primary: post-shift loss = mean log loss (nats) on switched questions in the 2000 steps after each
   boundary, restricted to switches whose new family is learnable by some trader (periodic, lfsr16).
2. Lag: steps after the boundary until the trailing-100-step mean log loss of that question < 0.1 nats;
   censored at 10000 (report the censored count); median over learnable switches.
3. Stationary cost: mean log loss on never-switched periodic/lfsr16 questions, steps 5000-40000.
4. Overall prequential log loss, all active questions, all steps.

**Predictions.** kelly locks in (lag of thousands of steps, growing with time spent in the old regime);
fs and ema100/ema1k recover within about 100 steps; fixed share costs < 0.01 nats on stationary questions;
gradient gates are slower than kelly because a trader's gradient credit is scaled by its current weight.

**Stop / decision rule.** If no currency beats kelly on post-shift loss by >= 0.05 nats in 3/3 seeds,
report "cumulative settlement is enough" and stop. Otherwise name the best currency whose stationary cost
is within 0.01 nats of kelly. One run per seed, no retuning after seeing results; any extra arm added later
is labelled post hoc.
