# M6 worker 3 pre-registration: (e) log- vs exp-utility pool, (f) bounded bets (base pass)

Written 2026-09-28 before any run. Scope: base arms only (prequential single pass, static stream);
no stress tests (multi-epoch, noise traders, D families) in this pass (Gate B).

## Substrate
S0 `streams_ext.make_streams(Q=64, T=24000, seed=s, heldout=8, release=12000, regime_every=0)`,
`to_array_stream`, W=32, prequential test-then-train, B=4 positions per step (one pass, each label
scored once before it is trained on; asserted in the loop). Trunk: S0 `model.Net` d=32, H=4, L=1,
Adam lr 1e-3, `torch.set_num_threads(1)`. Market price l0 = S0 `model.Logistic` (Adam 1e-2) trained
on the same revealed labels, detached inside the layer (traders read it, cannot move it).
Output layer: N=16 trader heads per question read [LN(residual), l0] and emit d_i (belief as a
log-odds correction to l0) and, for pool arms, confidence k_i = softplus(c_i) (input-dependent).

## Arms
Owned by w3:
- `elog` log-utility (Kelly) pool: clearing price of log-utility traders is the wealth-weighted
  linear pool in probability space: p = (s(l0) + sum k_i s(l0+d_i)) / (1 + sum k_i).
- `eexp` exponential (CARA) pool: CARA demand is tau (logit q - logit p), so clearing is the
  weighted mean of log-odds: l = l0 + sum k_i d_i / (1 + sum k_i).
- `fb01`, `fb05` bounded bets: l = l0 + sum_i b tanh(d_i / b), b = 0.1 and 0.5 nats per trader.
In-harness references (owned by w1, rerun here only to have a paired baseline):
- `a` plain logit sum l = sum d_i; `b` l = l0 + sum d_i; `kwd` arm a with AdamW wd 0.1;
  `kboost` boosting with shrinkage, market as offset: l = l0 + 0.1 sum d_i.
Early stopping is not defined in a single prequential pass (w1's claim covers it in multi-epoch).

## Metrics (positions in the second half, 12000..end, i.e. after the held-out release)
- primary: learnable-family BCE = mean prequential BCE over entries of families periodic, lfsr16,
  ca110, prime, and, or, periodic_ho.
- noise protection: max over {prng, ca30} of (BCE - ln 2).
- per-family BCE; l0 (market) BCE on the same entries; AND/OR violations (panel.violations on
  and/or questions vs their operand prices, all live steps; resolution is immediate in this stream,
  so "before resolution" = at trade time).
- committed-trader settlement per CURRENCY.lock: each head i settled leave-one-out at trade time,
  s_i = log p(y) - log p_{-i}(y) (prequential); report wealth Gini and the fraction of heads with
  negative wealth (diagnostic, not an arm criterion).

## Seeds
Stream and init seeds 0, 1, 2 (paired across arms). Mean +- sample sd over the 3 seeds.

## Stop rule (from STUDY-PLAN section 3, applied to my arms)
An arm in {elog, eexp, fb01, fb05} is DROPPED if its learnable-family BCE >= 1.01 x arm (a)'s in
3/3 seeds and its noise-protection metric is not better than the best kill control (kwd, kboost)
by more than 1 sd. gate_flags.base_arms_survive = true iff at least one of my arms is not dropped.
Secondary (report only): an arm "beats baseline" if it is below min(a, b, kwd, kboost) on
learnable BCE in 3/3 seeds.

## Cost
CPU only, 8 arms x 3 seeds = 24 runs, at most 3 in parallel (box load ~40 at start).
