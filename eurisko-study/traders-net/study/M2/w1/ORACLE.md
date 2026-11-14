# M2 worker 1: the exact ablation oracle

What this is: the ground truth that every M2 predictor is scored against. It is an exact held-out ablation (zero and mean) of every unit of the wealth-net model: 512 MLP neurons, 8 attention heads and 64 readouts. It covers 3 seeds and 250 coalitions per seed. Plan: `PREREG.md` (with amendment 1). Code: `oracle.py`, `summarize.py`.
Held-out segment: the fresh positions predicting bits[T..T+4095] (E=4096), split into half A (even positions) and half B (odd positions). Δloss = BCE(ablated) − BCE(intact); a positive value means the unit helps.
Unit ids are `N{l}.{i}` (flat l*256+i), `H{l}.{h}` (flat l*4+h) and `R{q}`. w2 and w3 use the same ids.

All paths below are relative to `study/M2/w1/`. Numbers are mean ± sd over 3 seeds at the final checkpoint (s1 step 31100, s2 step 31200, s3 step 31100) unless marked otherwise.

## Checks (PREREG §checks) — all pass
| check | s1 | s2 | s3 | file |
|---|---|---|---|---|
| intact BCE vs wealth-net ablation.csv frac=0 (tol 1e-4) | 1.063302 = 1.063302 | 1.055149 = 1.055149 | 1.035810 = 1.03581 | `results/oracle_checks_s*_step*.json` |
| fast path vs generic forward, singles (tol 1e-5) | 6.7e-9 | 3.3e-9 | 4.6e-9 | same |
| stage-skipping coalition vs generic forward (4 coalitions incl. layer-0 units) | 0.0 | 0.0 | 0.0 | same (exactly 0: identical float ops) |

## Reliability: is a single-unit oracle meaningful? (stop rule: split-half neuron Spearman < 0.5 in 2 of 3 seeds)
Split-half Spearman compares half A with half B (`results/summary.json`, per seed in `results/summary_reliability.csv`):
- Neurons, zero: **0.916 ± 0.015** (L0 0.867 ± 0.027, L1 0.952 ± 0.014). Neurons, mean: 0.879 ± 0.041. **The stop rule is not triggered.**
- Heads (n=8): zero 0.73 ± 0.30 (s1 is only 0.38); mean 0.75 ± 0.11. With only 8 units, per-seed head rankings are noisy.
- Readouts (n=64): zero 0.62 ± 0.09; mean 0.65 ± 0.05. This is the weakest oracle; readout Δs are small (max about 0.011).
- Zero vs mean ablation Spearman: neurons 0.83 ± 0.06 (L0 0.72 ± 0.06, L1 0.90 ± 0.06), heads 0.93 ± 0.06, readouts 0.81 ± 0.04. The choice of ablation matters most for layer-0 neurons, so w3 should report both.

## Shape of the oracle (neurons, final step)
- Helpful neurons: 453, 428 and 413 of 512 have Δ_zero > 0. Significantly helpful (Δ > 2·se, se from the half difference): 288 ± 8 zero, 288 ± 27 mean. Significantly harmful: 15 ± 8 zero (`results/summary_concentration.csv`, `summary_reliability.csv`).
- Heavy tail: the top 1% (5 neurons) carry 0.45 ± 0.12 of the total positive Δ_zero, and the top 10% carry 0.64 ± 0.08. One neuron per seed has Δ_zero between 0.58 and 1.98 nats; the median neuron has 0.004 (`summary_reliability.csv` max/median columns).
- vs wealth-net's own 512-position decay oracle (final step, neurons): Spearman **0.880 ± 0.026** (0.850 / 0.895 / 0.894) (`results/summary_vs_wealthnet.csv`). On this larger, disjoint segment the oracle agrees with wealth-net's but is less noisy.

## Coalitions: are units additive? (half A, `results/summary_coalitions.csv`, raw rows in `results/oracle_coalitions.csv`)
gap = Δ(coalition) − Σ Δ(singles).
| group (n per seed) | Spearman Δ vs ΣΔsingles, zero | mean gap zero | abs gap / abs Δ, zero |
|---|---|---|---|
| random neuron sets (100, size 2-64) | 0.975 ± 0.002 | −0.012 ± 0.017 | 0.23 ± 0.04 |
| random single-layer sets (30) | 0.946 ± 0.031 | −0.011 ± 0.007 | 0.23 ± 0.08 |
| random head sets (40) | 0.970 ± 0.017 | −0.007 ± 0.006 | 0.11 ± 0.01 |
| mixed neurons+heads (30) | 0.968 ± 0.008 | −0.009 ± 0.007 | 0.11 ± 0.05 |
| pairs among top-6 neurons (15) | 0.84 ± 0.16 | −0.07 ± 0.07 | 0.14 ± 0.06 |
| top-8 per family (11) | 0.89 ± 0.03 | −0.25 ± 0.23 | 0.33 ± 0.20 |
| top-k neurons, k=2..32 (5) | — | −0.72 ± 0.47 (k=4..32 pooled) | 0.37 ± 0.20 |
| all neurons of a layer (2) | — | −1.97 ± 0.47 | 2.6 ± 0.9 |
Readout sets are exactly additive (readouts act on separate questions), which confirms the harness.

Reading: for typical (random) coalitions, the sum of single-unit ablations ranks coalitions almost perfectly (ρ ≈ 0.95-0.99). Additivity breaks only among the high-Δ units, and it is **subadditive**: the important neurons are partly redundant. For example, the top-32 neurons' ΣΔsingles is 2.9 / 2.4 / 4.6 nats, while the joint Δ is 1.66 / 1.99 / 2.51. Zero-ablating a whole layer costs far less than the sum of its units. There is no evidence of synergistic (superadditive) pairs among the top neurons: in the zero mode, 0.29 ± 0.17 of top-6 pairs are superadditive. So any **coalition-based credit (Shapley-like) can matter only for ranking the top few units**; on the bulk, single-unit Δ is already the answer. I do not set a gate flag coalition_win from these data; the verdict belongs to w3.
The "poorest-wealth" coalitions (the 5 and 26 poorest traders in the wealth-net snapshot) are mildly superadditive (+0.06 ± 0.12 zero, n=2 per seed). Too few to conclude.

## Checkpoints 8k / 20k / 30k (all 3 seeds; `results/summary_reliability.csv`, `summary_coalitions.csv`, `summary_step_stability.csv`)
All 12 (seed, step) runs pass both checks: intact BCE matches ablation.csv to about 1e-6, and fast vs generic is ≤ 8e-9 (`results/oracle_checks_s*_step*.json`). Held-out intact BCE rises with training, from 0.71-0.76 at 8k to 0.92-0.93 at 20k, 1.02-1.06 at 30k and 1.04-1.06 at final. The late checkpoints are memorising, which matches w2's note.
| step | split-half neurons, zero | heads, zero | readouts, zero | zero vs mean, neurons | top-1% share, zero | Spearman of Δ_zero vs final |
|---|---|---|---|---|---|---|
| 8000 | 0.945 ± 0.025 | 0.62 ± 0.19 | 0.79 ± 0.07 | 0.82 ± 0.08 | 0.42 ± 0.14 | 0.71 ± 0.10 |
| 20000 | 0.925 ± 0.025 | 0.83 ± 0.14 | 0.69 ± 0.07 | 0.84 ± 0.01 | 0.46 ± 0.12 | 0.90 ± 0.01 |
| 30000 | 0.922 ± 0.012 | 0.74 ± 0.35 | 0.63 ± 0.10 | 0.80 ± 0.11 | 0.45 ± 0.13 | 0.92 ± 0.02 |
| final | 0.916 ± 0.015 | 0.73 ± 0.30 | 0.62 ± 0.09 | 0.83 ± 0.06 | 0.45 ± 0.12 | 1 |
Coalitions at 20k (the same design as at final) repeat the final picture. Random neuron sets: Spearman(Δ, ΣΔsingles) 0.971 ± 0.012 zero and 0.989 ± 0.002 mean. Heads: 0.980 ± 0.007. Pairs among the top 6: 0.88 ± 0.05. Top-k neurons: gap −0.56 ± 0.44 (subadditive). Whole layers: gap −1.66 ± 0.41.
The neuron ranking changes a lot from 8k to 20k (ρ 0.71 vs final) and is stable after 20k. Predictors must therefore be scored against the oracle at the same checkpoint, never against the final oracle.

## Unverified / caveats
- Heads: n=8, and one seed's zero split-half is 0.38. Treat head verdicts as low-power.
- The coalition fast-vs-generic check is exactly 0.0 on 4 coalitions, not on all 250.
- Mean-ablation means come from 1024 training positions (not the full train set).
- Final step differs by seed (31100 / 31200 / 31100), matching wealth-net's saved final checkpoints.
