# M8 worker 2 pre-registration: the Gate-B reduced arm (|a.g|-gated MLP neurons with rebirth)

Written 2026-09-28 ~09:45Z before any run. Claim: `../STATUS.md` (worker1 took the sign test).
Why reduced: Gate B did not pass (`../../M3/CURRENCY.md` "Gate B input": no currency beats recent
|a.g| by 0.05 in 3/3 seeds; `../../M2/BAKEOFF.md`: no coalition win), and the RETARGET cut M8
to its minimum. The plan's reduced arm is "only the |a.g|-gated arm", run here against the plan's
M8 controls. No eta/alpha/c grid, no wealth currency (CURRENCY.lock: no bookkeeping currency for
generic neurons), no GPU.

## Setup (identical for every arm)
- Stream: `S0 streams_ext.make_streams(Q=64, T=40000, seed, heldout=8, regime_every=10000,
  regime_frac=0.25)`: shifts at t = 10k, 20k, 30k (14 of 56 main questions switch family each).
- Net: `S0 model.Net(Q=64, W=32, d=64, H=4, L=2, mlp=4)` -> 256 MLP neurons per layer, 512 total.
  Adam lr 1e-3, prequential single pass (`run_prequential` semantics, test-then-train), B=4.
- Neuron gate: a 0/1 mask per MLP neuron multiplies the ReLU output. Budget k = 128 of 256
  active per layer (50%) in every gated arm (matched active units).
- Every R = 100 train steps: (1) rebirth: among mature units (age >= 500 steps) reinit the
  2 per layer with the lowest score (fresh fc1 row and bias, fc2 column at 0.1x default init,
  Adam state zeroed, score reset to the layer median); (2) gate: active = all young units
  (age < 200, a guaranteed slot, the exploration floor) + top-(k - #young) mature units by score.
- Scores (EMA over train steps, half-life 500 steps; gradient scores use the mask gradient, so
  inactive units get first-order shadow credit a.g for switching on):
  - `absag` (THE ARM): sum over tokens of |a_i * dL/d(act_i)|, per-token absolute value.
  - `taylor` (Molchanov 2017, https://arxiv.org/abs/1611.06440): |sum over tokens of a_i * g_i|.
  - `movement` (Sanh 2020, https://arxiv.org/abs/2005.07683, first order): -sum a_i g_i (signed).
  - `magnitude`: ||fc1 row_i|| * ||fc2 col_i|| (current weights, no EMA).
  - `cbp` (continual backprop utility, Dohare 2024, https://arxiv.org/abs/2306.13812):
    mean |a_i| * sum_j |fc2_ji|.
  - `random` (random reinit control): static random active set per layer, rebirth picks 2
    random mature units per layer per event (same rate), no score.
- References (no rebirth): `static128` (fixed first 128 neurons per layer, no mechanism),
  `dense256` (all 256 active; 2x the active units, an upper reference, not matched).

## Metrics
- Primary 1, "BCE at matched active units": prequential BCE (nats) over all live (position,
  question) pairs, whole run.
- Primary 2, "loss in the 2k steps after a shift": prequential BCE on the switched questions in
  the 2000 stream steps after each shift (t in [b, b+2000), b = 10k/20k/30k), pooled.
- Secondary: BCE on non-switched questions in the same windows; BCE on the last 10k steps;
  turnover (births that were ever active); dead fraction (active units with zero activation over
  the last 100 steps).
## Seeds and reporting
Seeds 1, 2, 3 (stream and init). Mean +- sd (ddof 1) over seeds, per-seed values in CSV.

## Stop rule (plan table, M8 row)
"No gain over random reinit in 3 seeds: stop." Operationally: `absag` gains over `random` iff
its primary-2 BCE is lower than `random`'s in 3/3 seeds AND its primary-1 BCE is not higher by
more than 0.005 nats in any seed. If not: M8 stops; wealth/|a.g| gating adds nothing over random
reinit here. Separately reported (not the stop rule): absag vs cbp, taylor, magnitude, movement
and static128 under the same 3/3 criterion.

## Prediction (written before running)
absag ~ taylor ~ cbp, all within 0.01 nats of static128 on primary 1; gating with shadow credit
beats random on primary 1 (random keeps paying for fresh active units); primary 2 gain over
random small or absent.
