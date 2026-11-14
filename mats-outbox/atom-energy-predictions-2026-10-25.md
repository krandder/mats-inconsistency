# Predictions registered before the bounded-weights / model+data results (opus-mats, 2026-10-25 ~23:00Z)
Part A (bounded weights):
- Catastrophic worlds (KL>1) drop to ~0 for learned and self-estimated arms; worst cell <= ~1.5x uniform-weight atoms.
- Learned C overall ~0.6-0.7x (slightly worse on average than unclamped 0.58, but no blow-ups); self-estimated ~0.75x.
- Equal-precision cells: estimated arms ~1.0x (shrinkage removes the earlier losses).
Part B (model + data):
- Data alone ~ (2^n - 1)/(2N) nats: poor at N=10-30. Model-only beats it there by a wide margin.
- Crossover where data's error falls below the model's error (~0.005-0.03): N ~ 100-1000, earlier for tempered (biased) models.
- Model+data <= min(model, data) at almost every N; converges to data at large N.
- Most likely failure: choosing the trust scale by held-out likelihood is unreliable at N=10-30, so model+data may lose to model-only there.
- Data-calibrated weights help most in tempered (biased) worlds; oracle-with-data best everywhere.

## Scored against results (2026-10-26 ~01:10Z; outbox/atom-energy-bounded-and-data-2026-10-25.md)
- Worst cell <= 1.5x: right (learned C 1.36x, from 74.9x unbounded).
- Catastrophic worlds drop to ~0: wrong under heavy tails (learned C and IRLS ~10 worlds with KL>1 vs 3 for uniform).
- Learned C ~0.6-0.7x: right (0.68x).
- Model+data beats data at small N: right, by more than expected (0.051x of empirical KL at N=10; never worse in 60 cells up to N=100).
- Crossover at N ~100-1000 and convergence to data: partly wrong. Still better than data on average at N=1000 (0.80x), but worse than data in 4/60 cells (too-flat worlds), with a fixed trust t.
- Main predicted failure (choosing trust by held-out likelihood at N=10-30): right, and worse than expected (hundreds of catastrophic worlds).
