# M7 w5 PREREG: bankruptcy and birth (stage 1)

Written 2026-09-28 before any run. Worker 5 of 5; item 9 of the M7 section (see ../STATUS.md).

## Question
Q5: can a wealth rule (bankruptcy + birth) take over the anti-collapse role of load balancing
without a tuned balancing coefficient? Prior art: Hayek machine (Baum), Kwee et al.
https://arxiv.org/abs/cs/0105025 (birth above a wealth threshold, stability "an art"), Economy of Minds
https://arxiv.org/abs/2606.02859 (rent, bankruptcy) (L1 NOVELTY-FLAGS F17: expect instability).

## Currency (imports M1/CURRENCY.lock: committed-trader settlement only)
The MoE layer is the last block before the readout, so each routed expert's contribution to the
question logit is an explicit committed position. Expert i's payoff on a routed token, booked at
the trade (test-time, before training on that label) price:
  pay_i = log p_full(y) - log p_{-i}(y), p_{-i} = price with expert i removed and the remaining
  top-k gates renormalised (exact log-score settlement of its committed correction; LOO).
Unrouted experts earn 0 (sleeping experts, T1 7.7).
Wealth: W_i <- W_i + sum_routed pay_i - rho, with rent rho per step = mean over experts of that
step's total payoff if positive (relative wealth: the market pays out what it earned, so the
only free constant is the endowment W0). Endowment W0 = 1 nat. No aux loss, no DeepSeek bias.

## Arms (E = 8 and 64, top-2 softmax renormalised over S, 3 seeds each)
- `none`: no balancing (same-code control).
- `deepseek`: DeepSeek loss-free bias, u = 1e-3 (same-code comparator; the item itself is w1's).
- `bb`: bankruptcy + birth. W_i < 0 -> bankrupt: expert i is replaced by a child of the richest
  expert (weights copied + N(0, 0.01 * std) noise, router row copied then its bias lowered
  by log 2 for the parent/child pair, i.e. the parent's routing mass is split), child W = parent
  W/2, parent keeps W/2 (split, Kwee-style copyright 0 share). At most one bankruptcy per 50 steps
  (a cool-down; reported).
- `bb-fresh` (sensitivity): bankrupt expert reinitialised fresh, router row = mean router row, W = W0.

## Metrics (from the harness; same as siblings where w1's harness exists)
Primary: prequential BCE over all scored positions (loss per active FLOP is proportional: same
top-2 FLOPs in every arm). Secondary: dead experts (fraction with load < 1/(10E) over the last
2k steps), max load (fraction of routed slots to the busiest expert, last 2k), expert x family
mutual information, number of bankruptcies, wealth Gini.

## Stop rule (from STUDY-PLAN sec 3)
Max load > 50% in 2/3 seeds, or BCE > 2% worse than DeepSeek at every E: the arm stops (no
extra variants). "Beats DeepSeek" = lower mean BCE than deepseek by more than 1 sd of the
difference across the 3 paired seeds, at some E, without max load > 50%. This sets
gate_flags.wealth_beats_deepseek for my arm.

## Seeds / scale
Seeds 1,2,3 (stream and init). Substrate S0 streams_ext.make_streams(Q=64, regime shift on),
single prequential pass. Scale set to fit the CPU budget (recorded in RESULTS with the exact T).
