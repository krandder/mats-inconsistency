# M9(c) worker 3 pre-registration: K/Q sweep in M6's trader layer

Written 2026-09-28 ~09:00Z before any run. Code: `run_kq.py`, `summarize.py` (this dir).

## Question
Q7: does the soup's K/Q law (traders per question K/Q buys "pricing" of a family as 1-(1-pi_f)^{K/Q},
Brier collapsing with density) reappear when the traders are heads of a neural output market layer
sharing one trunk? Two readings: at **fixed trader size** (more traders = more parameters and FLOPs)
and at **fixed total layer parameters** (more traders = thinner traders).

## Setup (identical to M6 w1 regime P; harness imported unchanged from `../../M6/w1/m6layer.py`)
- Stream: S0 `streams_ext.make_streams(Q=64, T=24032, seed=s, heldout=0, regime_every=0)`, W=32;
  8 families x 8 questions (periodic, lfsr16, ca30, ca110, prime, prng, and, or).
- Trunk S0 `model.Net` body d=64, H=4, L=2; market l0 = per-question logistic, detached in the layer.
- Trader layer: K trader heads per question (so K/Q = K traders per question; Q=64 questions),
  head width hw. Arms z = l0 + sum_i d_i (**B**, M6's anchored layer, primary) and z = sum_i d_i (**A**, plain sum).
- Prequential single pass (predict, then train), B=4, 6000 steps, Adam 1e-3 (net), 1e-2 (market);
  scored on positions [12000, 24000). torch threads 1.

## Arms (K in {1, 4, 16, 64})
- **FS** fixed trader size: hw = 8 for every K (layer params grow ~linearly in K).
- **FT** fixed total layer parameters: K*hw = 128 (hw = 128, 32, 8, 2). The per-(question,head) a,b
  scalars (2*Q*K) are not held fixed; exact parameter counts are reported per config.
- K=16 is the same config in FS and FT (M6's default); run once and shared.
- 7 configs x 2 arms (A, B) x seeds {1,2,3} = 42 CPU runs.

## Metrics (per run; mean +- sample sd over 3 seeds)
- Primary: **learnable BCE** (macro over periodic, lfsr16, ca110, prime, and, or), as M6.
- **Priced fraction** per family: share of the family's questions whose scored-window Brier < 0.1
  (the soup's "priced" notion; the plan's Brier 0.1 threshold).
- **Lottery fit**: per family, fit priced(K) = 1-(1-pi_f)^K (K = traders/question) by least squares
  over the 4 K points x 3 seeds; report pi_f and R^2 vs a constant-mean model (R^2 <= 0 = law absent).
- **FLOPs to Brier 0.1**: training FLOPs (fwd+bwd measured by torch FlopCounterMode on one batch,
  x steps) until the 500-position rolling Brier of the learnable families first drops below 0.1
  (inf if never).
- Secondary: humility (max(BCE-ln2) on prng, ca30), LFSR16 learnt (BCE < 0.35), AND/OR violations,
  effective traders per question (participation ratio of |d_i| over heads).

## Stop rule (plan §3 M9: "two sweep points with no trend: report the law as absent")
- A trend in a series (arm x axis) = learnable BCE differs between K=1 and K=64 by more than
  2 x pooled seed sd AND the 4-point means are monotone. If a series has no trend, the law is reported
  absent for that series. The soup-like law is "present in the net" only if priced fraction rises with K
  and the lottery fit has R^2 > 0.5 on at least 2 families under FT (fixed total params; FS confounds K
  with capacity).
- Wall cap 2 h CPU; at most 3 concurrent jobs, none launched if 1-min load > 40.

## Expected (prior)
The net's heads share a trunk and all see the full residual, so K should act like width, not like
independent search draws: small or no K effect under FT; mild gains under FS from capacity only.
