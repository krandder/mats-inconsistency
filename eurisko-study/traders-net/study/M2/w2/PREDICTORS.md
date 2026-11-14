# M2 item 2 (worker 2): predictor table

Status 2026-09-28 ~04:25Z. Plan and amendments: `PREREG.md`. Code: `predictors.py`, `merge.py`.
Scoring against the w1 oracle (Spearman, additivity, frontier, stop rule) is **worker 3's job**.
This page describes the table and gives one descriptive cross-check against a held-out proxy.

## What was produced (all paths under `study/M2/w2/`)
- `results/predictors.csv`: long format with columns seed, step, final, gran, flat, unit, predictor, value. It has 171,144 rows and 90 (predictor × gran) pairs. Unit ids and flat order match w1: `N{l}.{i}` flat l*256+i, `H{l}.{h}` flat l*4+h, `R{q}`.
- `results/pred-s{S}-{parts}-step{K}.csv`: per-seed, per-step source files that `merge.py` globs. w3's metrics.py reads these directly.
- `results/coverage.csv`: 72 pairs have all 12 seed-steps (3 seeds × 8000/20000/30000/final). 18 pairs (`*_val`) have final only (amendment 2).
- `results/checks-s{S}-*.json`: C1, C2 and C3 checks plus base BCE. `results/headtable-s{S}-step{K}*.npy`: v(S) for all 256 head coalitions.
- `results/proxy_spearman_ablval_final.csv`: Spearman of each predictor against `abl_val` at final, mean ± sd over 3 seeds (amendment 4).
- `results/preview_spearman*.csv`: written by `merge.py` only once w1's `oracle_units_*.csv` are final (they were still `.part` at 04:20Z).

## Predictors (raw sign; higher = predicted more important unless noted)
| name | definition | data |
|---|---|---|
| v1_wealth, v1_recent2k | wealth-net books v1: snapshot `wealth` at step t, and its change over the last 2k steps | training logs |
| v2_{tangent,lmsr,meanbase}_{int,inst} | M1 BooksV2 (CURRENCY.lock sha 15100104…, cp rule freeze_attn, one family group). `_inst` = the book at t; `_int` = sum over checkpoints 8000, 10000, …, ≤ t. Steps 0-8k are missing because wealth-net saved no earlier weights | training sample |
| absag_acc / absag_int | Σ\|window income\| from snapshots / Σ per-example \|c·s\| from BooksV2 | logs / training sample |
| movement, movement_out | Sanh et al. 2020 movement score 0→t (in+out / out only), from M1/w1 `units-s*.csv` | M1/w1 replay |
| grad_live_cum, v1_replay_books | cumulative live grad×input and replayed v1 books, from the same M1/w1 file | M1/w1 replay |
| wout_norm, winout_norm | output-weight norm; in·out norm | weights |
| gradnorm_cum, gradnorm_inst | per-unit loss-grad norm (snapshot running mean / at t) | logs / training sample |
| taylor, taylor_abs, taylor_sq | Molchanov 2019 first order: mean(−a·dL/da), mean\|a·dL/da\|, (mean a·dL/da)² | training sample |
| cond | conductance for F = Σ log p(y), input path from window baseline 0.5, 32-step midpoint | training sample |
| abl_train | exact zero-ablation ΔBCE (= IG along the unit's own gate 0→1) | training sample |
| shap_head_exact, shap_head_s64 | head Shapley from all 2⁸ coalitions / 64 sampled permutations | training sample |
| shap_readout | readout Shapley; the loss is separable across readouts, so this equals abl_train exactly | training sample |
| *_val | taylor*, cond, abl, shap on held-out positions T+8192..T+8703, final step only. Answers "estimator or data?"; not deployable at train time | held-out |

Samples: training = 512 positions, randint(0, split) with generator seed+77; val = 512 positions T+8192..T+8703. That val segment is disjoint from w1's oracle segment T..T+4095 and from w3's frontier segment T+4096..T+8191 (amendment 3). Each position has 64 questions.

## Checks
- **C2, Shapley efficiency**: Σ_h shap_head_exact = v(all) − v(∅), with worst |error| 5.6e-17 over 3 seeds × 4 steps plus val (`results/checks-s*-abl_shap.json`). Passes.
- **C1, readout lmsr book vs log-score gain**: the relative difference is −0.010 at 8000 and grows to −0.026 at final, consistent across seeds (`results/checks-s*-snap_norm_taylor_move_v2_cond.json`). Not exact, because the readout bias rb is booked in the bias account. The prereg's 1e-3 target was dropped in amendment 2.
- **C3, conductance completeness**: the readout conductance sum matches ΔF within about 0.1-3.5% (32-step Riemann); for example s1 final is 206.96 vs 207.15 (same JSON). Per-layer MLP and head sums are not complete, because of the residual bypass. This is expected and reported only.

## Memorisation caveat (affects every oracle on held-out data)
At final, BCE on the training sample is 0.105/0.137/0.138 but on val it is 1.074/1.124/1.091 for s1/s2/s3 (`results/checks-s*-abl_shap.json`, `base_bce`). Held-out loss is worse than chance (ln 2 = 0.69). Zero-ablating a unit lowers held-out loss for 16-23% of neurons, 34-38% of readouts and 0-50% of heads, and the readout abl_val sums are negative in all 3 seeds (−0.38/−0.43/−0.40) (from `results/pred-s*-abl_shap-step31*.csv`). Every training-sample predictor therefore measures in-sample importance. On held-out data, "important" partly means "important to overfitting".

## Descriptive cross-check vs abl_val (final, 3 seeds; `results/proxy_spearman_ablval_final.csv`)
Mean ± sd of Spearman against the exact held-out zero-ablation ΔBCE. This is not w1's oracle segment, and it is not a gate.
- **Neurons (512)**: taylor_val .53±.06; abl_train .39±.08; absag_int .30±.09. Best books: v2_lmsr_inst .18±.06 and v2_lmsr_int .13±.03. taylor_abs .11±.07; v1_wealth .05±.03; movement −.08±.04; wout_norm −.02±.04.
- **Heads (8)**: shap_head_exact_val .86±.07; abl_train .69±.23; shap_head_exact .64±.24; v2_lmsr_int .63±.21; taylor_abs .56±.04; movement .44±.11; v1_wealth .45±.30.
- **Readouts (64)**: movement .80±.04; abl_train = shap_readout .73±.02; wout_norm .71±.06; v2_lmsr_int .29±.04; v1_wealth −.20±.07. The magnitude predictors are negative: taylor_abs −.74±.02, absag_acc −.90±.02, gradnorm_cum −.74±.01. Readouts with big in-sample |a·g| are the overconfident ones, and ablating them helps held-out loss.
- **Stop-rule preview** (w3 makes the real call on w1's oracle). On this proxy the best book (v2_lmsr) ties Taylor within seed sd on neurons (.18±.06 vs taylor_abs .11±.07) and heads (.63±.21 vs .56±.04). It loses clearly to movement on readouts (.29 vs .80). The books beat neither Taylor nor movement at any granularity by more than seed sd. The in-sample exact ablation, abl_train, beats every book on neurons (.39 vs .18).

## Unverified / limits
- movement and grad_live_cum come from the M1/w1 replay, whose init is reconstructed from the seed. That is unverified there.
- v2 `_int` is a checkpoint-sampled, in-sample book that misses steps 0-8k. It is not the running book.
- `_val` exists at final only. The preview vs w1's oracle is pending on w1's files.

## Prior art
Movement pruning, Sanh et al. 2020: https://arxiv.org/abs/2005.07683 · Taylor importance, Molchanov et al. 2019: https://arxiv.org/abs/1906.10771 · Integrated Gradients, Sundararajan et al. 2017: https://arxiv.org/abs/1703.01365 · Conductance, Dhamdhere et al. 2018: https://arxiv.org/abs/1805.12233 · Neuron Shapley, Ghorbani & Zou 2020: https://arxiv.org/abs/2002.09815
