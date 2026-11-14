# M6 worker 1 pre-registration: base arms (a), (b) and the kill controls

Written 2026-09-28 ~06:35Z before any run. Code: `m6layer.py` (layer + arms), `run_w1.py` (runs), `summarize.py`.

## Setup (identical for every arm)
- Stream: `S0/substrate/streams_ext.make_streams(Q=64, T=24032, seed=s, heldout=0, regime_every=0)`,
  W=32, n_pos=24000, families periodic, lfsr16, ca30, ca110, prime, prng, and, or (8 per family).
- Trunk: S0 `model.Net` body (d=64, H=4, L=2, pre-LN), final LN output y [B,Q,64].
- Market price l0: per-question logistic (S0 `model.Logistic`), trained on the same batches by its own
  Adam(1e-2) and **detached** in the trader layer (traders never move the price's weights).
- Trader layer: 16 trader heads per question. Head i: g_i = relu(A_i y + c_i) (width 8, A_i shared over
  questions), d_{q,i} = r_{q,i}.g_i + a_{q,i}*l0 + b_{q,i} (per-question readout; reads the residual and l0).
  Readout init std 0.01 so every arm starts at its anchor.
- Net optimiser Adam(1e-3) (AdamW for WD). torch threads 1. Seeds s in {1,2,3} (stream seed = init seed).

## Arms
- **A** (a) plain logit sum: z = sum_i d_i (l0 only as an input feature).
- **B** (b) additive corrections on the market: z = l0 + sum_i d_i.
- Kill controls (applied to the plain net A, the thing a humble layer must beat):
  - **ES** early stopping: A, epoch chosen by validation BCE (all families). Multi-epoch regime only.
  - **WD** weight decay: A trained with AdamW(1e-3, weight_decay=0.1) on all net parameters.
  - **BOOST** boosting with shrinkage, market as offset: z = l0 + nu*sum_i d_i, nu = 0.1 inside the loss
    (Friedman shrinkage as a small step on the correction; nu=1 would be B).
- Reference (free): **L0** the market price alone.

## Regimes
- **P** prequential single pass (test, then train), B=4, 6000 steps; scored on positions [12000, 24000).
- **ME** multi-epoch: train positions [0,16800), validation [16800,19200), test [19200,24000);
  `assert_disjoint` on train/val and train/test; 20 epochs, batch 128; test scored at the final epoch
  (A, B, WD, BOOST) and at the val-chosen epoch (ES; also B-ES reported, not an arm).

## Metrics (per run, then mean ± sample sd over 3 seeds)
- Primary 1, **learnable BCE**: macro mean of family BCE over periodic, lfsr16, ca110, prime, and, or.
- Primary 2, **humility**: max over {prng, ca30} of (BCE − ln 2) (0 = holds the coin at ln 2).
- Secondary: per-family BCE/Brier/ECE (`panel.scores`), AND/OR Fréchet violation mean (`panel.violations`,
  tol 0.01, constraints parsed from `s.desc`, prices at trade time), LFSR learnt count (lfsr16 BCE < 0.35).
- Coherence on the D families is a stress item (not in this pass).

## Stop rule (literal STUDY-PLAN M6 rule applied to the base arms) and the flag
- Kill for the anchored arm: B learnable BCE ≥ A learnable BCE × 1.01 in 3/3 seeds in both regimes.
- `base_arms_survive` = B is not killed by that rule. (Pools (c)-(g) are w2-w4's; the pool rule is theirs.)
- Humility comparison reported, not gating: B's ME humility vs the best kill control (min over ES, WD, BOOST);
  "B beats the kill controls" only if lower in 3/3 paired seeds.
- Expected (prior): B ≈ A on learnable; B better humility than A in ME; ES/WD close the gap.
- Wall cap: 3 h of this worker's CPU; at most 6 concurrent jobs, fewer if load > 40.
