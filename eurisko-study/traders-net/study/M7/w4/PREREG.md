# PREREG, M7 worker 4: global wealth prior on the router (stage 1)

Written 2026-09-28 06:45Z, before any main run. Two calibration runs (arm `none`, 2000 steps, E=8/64,
seed 1, `/tmp/traders-net/M7w4/calib-*`) were used only to set the beta grid; they are not results.
Calibration read: per-expert settlement rate sd across experts 0.0048 (E=8) / 0.0018 (E=64)
nats/token, so beta 100 moves selection scores by ~0.2-0.5 and beta 1000 by ~2-5.

**Question (Q5, item "global wealth prior, expected leak").** If each expert's router selection
score gets a bias equal to its recent market earnings summed over all families, does that
replace load balancing, and does it leak (route experts to families where they lose money)?
The per-family version is the comparator that isolates the leak.

**Code.** `job/moe_w4.py` (harness, own copy; S0 substrate copied into `job/substrate/`),
`analyze_w4.py`. Model: 2-block transformer over the S0 stream tokens (Q=64 + 8 held-out, W=32),
d=32, H=4; block 0 dense MLP (shared core), block 1 MoE, E ∈ {8, 64} experts of hidden 32,
top-2, gates = softmax over the two selected router logits; any bias affects selection only.
Stream: `streams_ext.make_streams(Q=64, T=40032, seed, regime_every=10000, regime_frac=0.25)`,
held-out periodic family released at T/2. Prequential single pass (test, settle, then train),
B=8 positions per step, 5000 steps, Adam 1e-3, 1 thread per run.

**Currency.** CURRENCY.lock committed-trader rule, exact leave-one-out log score at the output:
s = log p(y) − log p_without_expert(y) for each routed (token, expert), computed at test time
before training on the label. Wealth rate R = EMA (half-life 1000 steps) of settlement per live
token. Router = broker (paid nothing).

**Arms** (3 seeds each, seeds 1-3; stream seed = model seed), at E=8 and E=64:
1. `none` — no balancing.
2. `deepseek` — loss-free bias, u = 1e-3 (arXiv 2408.15664).
3. `global` beta ∈ {100, 1000} — selection bias beta·(R_e − mean R), R over all families.
4. `perfam` beta ∈ {100, 1000} — selection bias beta·(R_{e,f} − mean_e R_{e,f}) for the token's family.
Total 36 runs.

**Metrics** (definitions in `analyze_w4.py`): primary = prequential BCE on the last 25% of
positions (active FLOPs per token identical across arms at fixed E, so this is loss per active
FLOP); secondary = max load (fraction of tokens routed to the busiest expert), dead-expert
fraction (<0.1% of slots), expert×family MI (bits), leak_frac / leak_nats (slots and nats lost
in (family, expert) pairs with negative mean settlement), BCE in the 2000 positions after each
regime boundary, wealth Gini.

**Predictions.** global leaks more than perfam (higher leak_frac) and does not prevent collapse
(rich-get-richer: max load ≥ none); perfam ties or loses to DeepSeek on BCE.

**Decision rules (STUDY-PLAN M7).** An arm "beats DeepSeek" only if its mean bce_last is lower
than DeepSeek's by more than the larger of the two seed sds at that E, and it is not worse at the
other E by more than that margin. Stop rule per plan: max load >50% in 2/3 seeds, or BCE >2% worse
than DeepSeek at every E → the arm is stopped (reported as failed). If no wealth arm beats
DeepSeek → gate_flags.wealth_beats_deepseek = false for my item. No reruns except crashes;
fixed budget of 36 runs, each capped at 60 min.

**Compute.** One G1 job (CPU processes on the L4 VM, 8 at a time), ≤ 1.5 h, ≈ $1.5 ceiling; fall
back to local CPU at ≤ 4 jobs if G1 refuses.
