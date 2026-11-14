# M2 worker 2 prereg: predictors (`predictors.py`)

Written 2026-09-28 ~02:05Z, before any run. Item 2 of M2 (claim in ../STATUS.md).

**What.** For every unit the oracle (M2/w1) ablates, a table of predictor scores computed WITHOUT the
oracle's held-out segment. Worker 3 scores them (Spearman vs Δloss, additivity, frontier); this item
only produces scores plus two sanity checks.

**Models.** wealth-net `train-s{1,2,3}` at steps 8000, 20000, 30000, final (31100/31200/31100), same
as w1. Unit ids and flat order as w1 (`N{l}.{i}` flat l*256+i, `H{l}.{h}` flat l*4+h, `R{q}`).

**Data the predictors may see.** (i) wealth-net training logs (snapshots every 100 steps),
(ii) weights, (iii) a training sample: 2048 positions drawn from the training range
randint(0, 0.8(T-W)) with generator seed+77, (iv) a "val" sample: positions T+4096 .. T+6143
(fresh, disjoint from the oracle's T .. T+4095 and from training). Val variants are marked `_val`;
they answer "estimator or data?" and are not deployable-at-train-time predictors.

**Predictors (arms)** — higher raw value = predicted more important unless noted; magnitude
variants are separate rows.
1. `v1_wealth` cumulative books v1 (snapshot `wealth` at step); `v1_recent2k` = wealth(t) - wealth(t-2000).
2. `v2_{tangent,lmsr,meanbase}_int` M1 BooksV2 (CURRENCY.lock sha-checked, cp rule freeze_attn,
   single family group) on the training sample at every saved checkpoint 8000,10000,...<=t, summed
   (checkpoint-sampled in-sample book; wealth-net has no step-0..8k weights, so the 0-8k part is
   missing; at t=8000 it is the instantaneous book). `v2_*_inst` = the book at t only.
3. `absag_acc` = sum over 100-step windows <= t of |window income| (snapshots); `absag_int` =
   BooksV2 `absag` (per-example |c.s|) summed over checkpoints <= t.
4. `movement` = Sanh et al. 2020 displacement-form movement score 0 -> t (in+out weights), summed
   from M1/w1/results/units-s*.csv 2k windows (init reconstructed from seed, unverified there).
   `grad_live_cum` from the same file (live grad x input, cumulative).
5. `wout_norm` ||fc2[:,i]|| / ||o.weight[:,head]||_F / ||rw[q]||; `winout_norm` in+out.
6. `gradnorm_cum` snapshot per-unit loss-grad norm (running mean) — neurons/readouts per unit,
   heads as available; `gradnorm_inst` at t on the training sample (small = unimportant).
7. `taylor` = mean over the training sample of -a.dL/da (live LN, live attention; Molchanov 2019
   first order), `taylor_abs` = mean |a.dL/da| per example, `taylor_sq` = (mean a.dL/da)^2.
8. `cond` = conductance (Dhamdhere 2018) of each unit for F = log p(y), input path from window
   baseline 0.5 (±1 encoding -> 0) to the input, 32-step midpoint Riemann; question-id embedding
   kept. `ig_self` = IG along the unit's own gate 0 -> 1 = exact zero ablation Δloss on the TRAINING
   sample (`abl_train`), reported as the in-sample ablation predictor.
9. `shap_head_exact` (all 2^8 head coalitions, training sample, v(S) = -BCE with heads outside S
   zeroed) and `shap_head_s64` (64 random permutations from the same table, rng seed*1000+step).
   Readouts: v(S) is separable in the readouts (z_q only enters question q's loss), so Shapley =
   single zero ablation exactly; `shap_readout` = `abl_train` for readouts, no sampling.
10. `_val` variants of 7-9 on the val sample.

**Seeds.** 3 (wealth-net seeds 1-3); mean ± sd across seeds is worker 3's job.

**Checks.** (C1) BooksV2 identity: sum of readout `lmsr` = log-score gain on the sample within
1e-3 relative, from its `acct`; (C2) exact-Shapley efficiency: Σ_h shap_head_exact = v(all) - v(∅)
to 1e-6; (C3) conductance completeness per layer is reported (not exact: residual stream bypass).

**Stop rule.** None beyond finishing the table (a measurement). If a predictor costs > 2 CPU-hours
in total it is dropped and listed as not computed.

**Compute.** CPU, <= 3 jobs x 2 threads (fleet cap 12), torch threads set explicitly; outputs
M2/w2/results/predictors.csv (long: seed,step,final,gran,flat,unit,predictor,value) + checks JSON.

**Amendment 1 (02:25Z, before the first full run).** Timing at n=128 under load ~49 (1 thread per job):
per checkpoint and sample, abl 100 s, head table 45 s, cond 20 s, v2 8 s. The training and val
samples are cut from 2048 to **512 positions** each (x 64 questions = 32768 settled predictions per
sample). Jobs run 1 thread each, launched by a scheduler that keeps my jobs <= 3 and the
traders-net total < 12. C1 now reports readout lmsr + bias_lmsr vs the gain (the readout book
excludes the rb bias account).

**Amendment 2 (02:40Z, first full attempt killed after 20 min, no results kept).** Under load ~90
one checkpoint x sample of abl+shap took ~17 min. The `_val` sample is computed at the **final
checkpoint only** (`--val-steps final`); every step is written to its own file as soon as it is done
(`results/pred-sS-<parts>-stepK.csv`, atomic rename; reruns skip finished steps). C1 is reported as the
relative difference of the readout lmsr book and the log-score gain (not expected to be exact: the
readout bias rb is booked in the bias account).

**Amendment 3 (02:50Z, before any `_val` number existed).** M2/w3's pruning frontier evaluates on
positions T+4096 .. T+8191, which overlapped my val sample. The val sample moves to **T+8192 ..
T+8703** (512 positions), disjoint from the oracle segment, the w3 frontier segment and training.
Finished per-step files (training sample only) are unaffected (prefix-stable streams, same train draw).

**Amendment 4 (03:31Z, before any `abl_val` number existed).** w1's oracle files land late
(final ~04:30Z, 20k ~07:00Z). As a descriptive cross-check only (not a gate, w3 owns scoring), merge.py
also reports Spearman of every final-step predictor vs `abl_val` (exact zero-ablation ΔBCE on
T+8192..T+8703, the oracle's quantity on a different segment), mean ± sd over 3 seeds, in
results/proxy_spearman_ablval_final.csv.
