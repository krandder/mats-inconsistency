# M7 w2 PREREG: DeepSeek bias + per-family wealth prior (stage 1)

Written 2026-09-28 ~06:40Z, before any run. Worker 2 of 5 on M7 (claim in `../STATUS.md`).

## Substrate and model
- Stream: `S0/substrate/streams_ext.make_streams(Q=64, T=30000, seed=s, heldout=8, release=15000,
  regime_every=10000)` -> `to_array_stream`; 72 questions, W=32. Prequential single pass, B=4 positions/step
  (7492 steps), delay 0: test at trade price, then settle, then train on the same revealed labels.
- Net: S0 `model.Net` layout (d=64, 4 heads, L=2, pre-LN), MLP replaced by an MoE in both layers:
  E experts (E in {8, 64}), expert = ReLU MLP d->64->d, top-2, router = Linear(ln2(x)) -> E logits,
  gates = softmax over the 2 selected logits (renormalised, T1 7.6 broker convention). Adam lr 1e-3.
  Active FLOPs are identical across arms at a given E (the wealth arms add settlement forwards; reported).

## Arms (selection score = router logit + bias terms; bias terms never enter gate values)
- `none`: no balancing.
- `ds3`: DeepSeek aux-loss-free bias (arXiv 2408.15664), b_e += u*sign(mean load - load_e) per step, u=1e-3.
- `ds2`: same, u=1e-2 (L1 flag F14: compare DeepSeek at more than one step size).
- `dsfw`: ds3 + per-family wealth prior, discount gamma=0.999 per step.
- `dsfw_fast`: ds3 + per-family wealth prior, gamma=0.99.
- `dsrw`: control: ds3 + the same prior with questions assigned to 9 random groups of the family sizes
  (fixed per seed) instead of families: does the gain need family structure?
Wealth (CURRENCY.lock committed-trader rule): each routed (layer, slot) is a committed trade on its token's
question. Payoff = log p(y) - log p_{-slot}(y), p_{-slot} = the trade-time price with that slot's gated output
removed from the residual at that layer for all tokens at once (one no-grad forward per layer x slot, at trade
time, before the label trains anything). Books per (layer, family of the question at creation = fam0, expert):
discounted payoff sum S and routed count C. Prior bonus_{l,f,e} = (N_{l,f}/E) * S/(C+1), N = discounted
family slot count (per-token mean edge scaled to the evidence an expert has at uniform load; the Hedge/Bayes
eta=1 on log-score nats). Only gamma is a coefficient.

## Metrics (3 seeds each, mean +- sd, sample sd)
Primary: prequential BCE (nats) over live main-question tokens, all positions and last 10k positions.
Secondary: BCE in the 1000 positions after each regime boundary on switched questions; held-out family BCE
after release; max load (share of routed slots of the busiest expert, last 1000 steps, max over layers);
dead experts (share of experts with load < 10% of uniform in the last 1000 steps); expert x family mutual
information (bits, last 2000 steps, mean over layers); active FLOPs per token incl. settlement.

## Decision rules
- "Beats DeepSeek" (feeds gate_flags.wealth_beats_deepseek): a wealth arm's all-position BCE is lower than the
  better of ds3/ds2 in 3/3 paired seeds AND by more than the larger of the two arms' seed sd, at both E, with
  max load <= 50%. At one E only: reported as "partial", flag stays false.
- Stop rule (STUDY-PLAN M7): max load > 50% in 2/3 seeds, or BCE > 2% worse than DeepSeek at every E: stop the arm.
- Prediction (expected, STUDY-PLAN Q5): tie with DeepSeek; dsrw ties dsfw if family structure adds nothing.
- Seeds 1,2,3. CPU only, <= 4 concurrent jobs, torch 1 thread.

## Amendment 2026-09-28T06:42Z, after a CPU smoke test only (T<=2432, no result read beyond scale/timing)
- Horizon harmonised with siblings w1/w3/w5 (their queued GPU jobs): T=24000, B=8 (2996 steps), release 12000,
  regime boundaries at 10000 and 20000. Same-code comparators unchanged.
- Smoke showed the N/E-scaled prior reaches |bonus| ~100 nats (sd over experts ~57) within 800 steps, i.e. it
  overrides the router. Added arm `dsfw_mean`: bonus = S/(C+1) (per-token mean edge, no evidence scaling), the
  small-prior bracket. dsfw and dsfw_fast stay as preregistered. 7 arms x E{8,64} x seeds{1,2,3} = 42 runs, on
  one L4 via G1 (CPU load ~57 on 16 cores at 06:40Z, so CPU is not an option).
- "Beats DeepSeek" applies to dsfw, dsfw_fast, dsfw_mean; dsrw is a control and cannot count as a win.
