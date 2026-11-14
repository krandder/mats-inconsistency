# M5 worker 2 PREREG: coherence arbitrageurs (AND / OR / exclusive / D-prove) as class-C shadow traders

Written 2026-09-28 ~02:00Z, before any run. Scope: STUDY-PLAN M5 item 2; theory from T1 Lemma 5 and 4.7
(`study/T1/THEORY.md`): E_T(t) = sum_s log q_T(y_s)/p_s(y_s), E_C = max_{T in C} E_T, invasion fitness phi_T = slope
of E_T for a shadow trader (wealth -> 0); members are bounded by ln(1/eps); D-prove is booked at proof (reveal) time.

## Markets (price sources)
1. **LI-net**: S0 `make_li_stream(seed, T=20000)` (seeds 0,1,2), prequential Net d=32 H=4 L=1, Adam 1e-3, B=4 (same
   recipe as `S0/w3/baseline.py`); price = sigmoid(logit) at trade time for all Q questions.
2. **LI-logistic**: the per-question Logistic trained in the same run (Adam 1e-2).
3. **wealth-net**: frozen checkpoints `../wealth-net/results/train-s{1,2,3}.pt` (final, ~31k steps) and
   `weights/train-s{1,2,3}-step8000.pt`, evaluated on held-out positions [split, n_pos) of their own stream
   (`make_questions(64, 24000, seed)`). Only AND/OR constraints exist there.

## Class C_arb (my traders), one trader per constraint instance
Kinds: `and`, `or` (c = a op b), `ex3` (exclusive-exhaustive triple, sum = 1), `taut_or`, `impl` (=1), `contra` (=0),
and extra `twin` (D-delay twin must quote its base's price; labelled extra, not in the plan's list).
- **K (Kelly projection trader)**: belief q = Euclidean projection of the tuple's prices onto the coherent polytope
  (Dykstra, clipped to [1e-3, 1-1e-3]); trades every question in the tuple; each trade booked at the trade-time price,
  settled at the question's reveal time (D-prove = proof time). E_T(t) summed over settled trades up to t.
- **B (bounded bundle arbitrageur, LI-style)**: at most 1 unit of the cheapest dominating bundle per constraint per step
  when the violation exceeds 0.01; P&L realised at settlement. Riskless by construction (check: min per-trade P&L >= -1e-9).

## Metrics
- E_C(t) per kind and over all kinds (max over traders, and sum), at settlement time; slopes via `panel.ec_slopes`
  (log-log exponent b, last-half linear slope).
- Invasion fitness phi = linear slope of shadow E_T over the run (and first/second half). Member test: 2-member Kelly pool
  per question (market weight 1-eps, trader eps=1e-3), fixed-point clearing (price-reading trader, T1 5.2), final wealth
  share and max member E (must be <= ln(1/eps) = 6.9078).
- Coherence before resolution: S0 `violations(..., t)` `pre:` means, recomputed from the saved prices.

## Seeds, reporting, stop rule
3 seeds per market, mean +- sd (sample sd). Harness: no stop rule (plan). Wall cap 6 h. CPU only, <= 3 jobs at once.

## Predictions (written before running)
- LI-net: AND E_C sublinear-ish (small violations), OR and ex3 linear (ex3 live violation ~0.13 per S0);
  D-prove: large E early (pre-resolution violation ~0.5), then flat after proof reveals (b < 1).
- LI-logistic: D-prove pre-resolution prices 0.5 -> E grows until each question's first reveal; ex3 linear.
- wealth-net: AND/OR E_C linear with small slope (a frozen net is a fixed incoherent map).
