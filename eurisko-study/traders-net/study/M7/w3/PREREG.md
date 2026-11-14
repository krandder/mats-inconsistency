# M7 w3 PREREG: Exp3 shadow credit (stage 1)

Written 2026-09-28 ~06:45Z, before any run. Worker 3 of 5 on M7 (claim: `../STATUS.md`, 06:30:16Z).

## Substrate, model, loop (from w1's shared harness, imported; its sha256 is recorded per run)
`../w1/moe_harness.py` `run()` with its defaults: `streams_ext.make_streams(Q=64, T=24000, seed=s,
regime_every=10000)` (72 questions incl. 8 held-out periodic released at T/2; regime shifts at 10k, 20k),
S0 prequential single pass (test at trade price, then train on the same revealed labels), B=8 positions/step,
Adam 1e-3, MoE in both layers, expert d->64->d, top-2, gates = softmax renormalised over the selected set
(w1 Stage-0 convention). Seeds 1, 2, 3. E in {8, 64}.

## Arms
- `none` (same-code control), `deepseek` (u=1e-3, same-code control; w1 owns the control item).
- `exp3` (the arm): after each train step, for each token sample ONE unrouted expert e uniformly
  (q_e = 1/(E-k)); read its revenue r_e = <E_e(x), -dL/dh> (the expert at zero gate; the harness computes it,
  the policy reads rev_all only at e and at the routed set). Credit on logit e:
  c_e = ghat_e (r_e - R) / q_e, with ghat_e = exp(z_e)/(exp(z_e) + sum_S exp(z_i)) (the gate e would get if
  added to S) and R = sum_S g_i r_i. This is the importance-weighted estimate of the dense-softmax router
  gradient for the unrouted experts (T1 Lemma 7: the top-k router gradient is relative revenue under censored
  feedback; shadow credit removes the censoring). No tuned coefficient: q is uniform, scale is the loss's own.
  Applied as an extra router-weight gradient -sum_n c[n,e] x_n on the NEXT step (one-step lag, same Adam),
  because the harness hook runs after the optimiser step.
- `fullinfo` (diagnostic upper bound, not a candidate): the same credit for every unrouted expert with q=1
  (full-information dense router gradient on the unrouted part). If fullinfo does not prevent collapse,
  exp3 cannot.
Cost: exp3 evaluates 3 experts per token in training (+50% expert FLOPs in training; eval unchanged).
Wealth ledger per expert (reported, not used for routing): cumulative realised routed revenue sum_n g r.

## Metrics (3 seeds, mean +- sd, sample sd), from the harness summary
Primary: prequential BCE (nats) over the 2nd half (`bce_2nd_half`) and all positions (`bce_all`).
Secondary: max load (share of tokens of the busiest expert, last 20% of steps, worst layer), dead fraction
(< 10% of fair share), expert x family MI (bits, normalised), post-shift BCE (2000 positions after each
boundary), active FLOPs per token (eval) and training expert FLOPs; wealth Gini per layer.

## Stop rule and decision (STUDY-PLAN M7 row)
- Stop the arm: max load > 50% in 2/3 seeds, or bce_2nd_half > 2% worse than deepseek at every E.
- "exp3 beats DeepSeek" (my input to gate_flags.wealth_beats_deepseek): bce_2nd_half lower than deepseek in
  3/3 paired seeds AND mean gap > the larger seed sd, at both E, with max load <= 50%. At one E only =
  "partial" (flag false). Otherwise false: "DeepSeek already built the price".
- Compute: CPU, 1 thread per run, few concurrent runs and only while the box load allows (16 CPU, already
  heavily loaded); otherwise one G1 job on an L4 VM's 8 vCPUs (<= $2).
