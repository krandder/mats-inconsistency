# M7 worker 1 pre-registration (written 2026-09-28 ~06:40Z, before any Stage-1 run)

Scope: Stage 0 (router-gradient check, gate convention) and the Stage-1 *control* arms on the
shared harness `w1/moe_harness.py`. Wealth arms belong to siblings w2-w5 (see ../STATUS.md).

## Stage 0
`check_stage0.py`: autograd -dL/dz against g_i(r_i - R) in the real harness layer, float64, E in
{8, 64}, seeds 1-3, both layers. Pass: max relative error < 1e-10 for the harness convention
(renormalised softmax over the top-k set), with a nonzero DeepSeek selection bias, for expert
choice in its own (ii)-form, and for Switch after subtracting the aux-loss term.
(Run and passed at 06:38Z, see results/stage0.json; this PREREG was written before Stage 1.)

## Stage 1 setup (fixed)
S0 streams `make_streams(Q=64, T=24000, seed=s, heldout=8, regime_every=10000)` (72 questions
incl. 8 held-out periodic released at t=12000; regime shifts at t=10000 and 20000). Prequential
single pass (S0 `run_prequential`), B=8 positions/step (576 tokens), 3000 steps, Adam lr 1e-3.
Model: S0 Net shape (d=64, H=4, L=2) with each MLP replaced by an MoE of E experts
(ReLU, h_e=64), top-2, no capacity limit or token dropping (except expert choice's intrinsic
capacity C = N*k/E). Seeds 1, 2, 3 (seed sets stream and init).

## Arms (controls)
- `dense`: one MLP of width 2*h_e = 128 (same active FLOPs as top-2), E-independent.
- `none`: top-2, renormalised softmax, no balancing.
- `switch`: + Switch aux loss alpha*E*sum f_i P_i, alpha = 1e-2 (arXiv 2101.03961 default).
- `deepseek-u1e-3`, `deepseek-u1e-2`: loss-free bias b_i += u*sign(mean load - load_i) on the
  selection only (arXiv 2408.15664); two step sizes per NOVELTY-FLAGS F14.
- `expertchoice`: arXiv 2202.09368, C = N*k/E.
E in {8, 64}.

## Metrics (all from the trade-time/test forward pass)
- Primary: prequential BCE over the second half of positions (`bce_2nd_half`), mean +- sd over
  3 seeds. Loss per active FLOP = that BCE at fixed active FLOPs (identical by design across
  token-choice arms; the router adds 2*d*E per layer and is reported).
- `max_load`: max over experts of the fraction of tokens routed to it, aggregated over the last
  20% of steps, mean over layers (and worst layer).
- `dead_frac`: fraction of experts with < 0.1 x fair share (k/E) over the last 20% of steps.
- `mi_expert_family_bits`: I(expert; question family fam0) over routed assignments, last 20%.
- `bce_post_shift_2k`: BCE in the 2000 positions after each regime boundary (mean of two).
- leak: n/a for controls (router state uses no labels); wealth arms must report theirs.

## Stop rules (from STUDY-PLAN section 3, applied by the M7 merger)
- Any arm with max_load (worst layer) > 0.5 in >= 2/3 seeds is flagged "collapsed".
- Reference for "beats DeepSeek": the better DeepSeek u at that E (lower mean bce_2nd_half).
  A wealth arm beats DeepSeek at E if its bce_2nd_half is lower in 3/3 paired seeds and it is
  not collapsed. A wealth arm > 2% worse than DeepSeek at every E: stop that arm.
- If no wealth arm beats DeepSeek: skip Stage 2 ("DeepSeek already built the price").
Expected (my prior): none collapses at E=8 top-2 on this small task; none's max load high at
E=64; DeepSeek ~ Switch within 1 sd on BCE; expert choice best MI.
