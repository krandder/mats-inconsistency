# M5 worker 1 PREREG: calibration-bin shadow traders (item 1 of M5's class C)

Written 2026-09-28 ~02:05Z before any run.

## Trader class C_cal (fixed, no fitting)
- 10 equal-width price bins on p in [0,1], 2 signs, 3 bump sizes delta in {0.1, 0.3, 1.0} logit
  = 60 traders. Belief q_T(p) = sigmoid(logit p + s * delta * k_b(p)), k_b a hat kernel
  (1 at the bin centre, 0 at the neighbouring centres; price-continuous, as T1 lemma 5 uses).
  Each trader trades every question at every position (it abstains where k_b(p)=0).
- C_cal_fam (secondary): the same 60 traders conditioned on the question's family (a trader
  sees the question id's family; 8 families x 60 = 480 traders).
- cal-online (one trader, no selection): per bin (10) a prequential Kelly correction
  c_b = logit(freq_b) - mean logit(p) over resolved trades so far in that bin (add-1/2 counts),
  clipped to |c_b| <= 2, belief q = sigmoid(logit p + k-weighted c); it only uses labels
  resolved before the trade.

## Markets (price sources), 3 seeds each (1,2,3)
- A. Prequential net, P4 arm of S0 worker 1 (wealth-net architecture, test-then-train,
  wealth-net streams incl. pi, Q=64, 64000 positions): trade-time logits `z` in
  /tmp/traders-net/S0/w1/P4-s{1,2,3}.npz.
- B. Prequential logistic in the same runs (`zb`).
- C. P16 net (same file family) as a batch-size check.
- D. Wealth-net checkpoints (multi-epoch, in-sample trained on positions < split), frozen, at
  steps 8000, 16000, 30000, priced on the held-out tail [split, n_pos) (4794 positions,
  time-ordered). Read-only load of ../wealth-net/results/weights/train-s{1,2,3}-step*.pt.
- E. Soup: no calibration shadow exists; soup.dump is a state snapshot, not a price history
  (no replay; new soup runs are out of scope). Report sh_c50 (constant-0.5 one-share trader,
  the soup's only price-reading bin trader) slope from runs/phase3/*/metrics.csv as the
  nearest soup analogue, clearly labelled.

## Metrics
- E_T(t) = sum over positions s<=t and questions of log q_T(y)/p(y) (nats; shadow at the
  market price, T1 5.2: no fixed point is needed because shadows do not move the price).
- E_C(t) = max_T E_T(t). Slopes via S0 panel.ec_slopes: loglog exponent b and last-half linear
  slope (nats/position); E_C final.
- Invasion fitness phi_T at wealth eps=1e-3: per-position mean of log q_T(y)/p'(y) with
  p' = (1-eps) p + eps q_T (the invader's own price impact), last half of the run; report the
  best trader's phi and cal-online's phi.
- One-share linear P&L (soup units, sign(q-p)*(y-p)) as secondary.
- Selection null: the same traders against y_null ~ Bernoulli(p) (market calibrated by
  construction), 3 null draws per source/seed. "Exploitable" = E_C final above the null's
  max over draws in 3/3 seeds AND cal-online phi > 0 in 3/3 seeds.
- Per-family attribution of the best trader's final E, and ECE per source.

## Expectations (recorded before running)
- Prequential net and logistic: E_C slope near 0 on the whole stream except early-training
  miscalibration; loglog b < 1. Logistic sits above ln2 on pi (overconfident on noise), so
  C_cal should exploit it linearly on pi/ca30 (b ~ 1).
- Wealth-net frozen checkpoints: memorised pi, overconfident held-out -> C_cal exploits
  linearly (b ~ 1), growing with checkpoint step.

## Stop rule
Harness: no scientific stop rule (M5 is a harness). Operational: each job single-threaded,
<= 2 CPU jobs at once, kill any job > 30 min; done when A-D x 3 seeds are written.
