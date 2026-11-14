# P1 pre-registration: multi-horizon anytime-value heads for halting a frozen recurrent net

Written 2026-09-29, before any evaluation trajectory was produced. The full spec is the P1 JSON task
(workflow wf_f335b9e5-391). This file fixes the primaries, metrics, gates and the deviations from the spec.

## Primaries (fixed in advance)
- T2m (E-P1-1): R7 (multi-horizon value heads, H in {1,2,4,8,16,32,64,cap}, continue while max_H V_H/H_eff > c).
  Pools L=64 and L=128 (both must pass); L=256 is reported but not part of S2.
- Mazes (E-P1-2): R9 (KL-settled heads, test-time refit on the unlabelled pool, label-free). Only if the
  headroom gate passes.
- Diagnostic (E-P1-3): calibration of R9 after refit at 59 and 99.

## Metrics
- T2m score S = sum_i (log q(y_i) - log 0.5) at the stop; a* = 0.98 x best fixed-T mean score of the pool.
  Cost = steps (primary); FLOP-adjusted cost = steps x (1 + r_head) with r_head = head MACs / backbone MACs.
- Mazes: exact match (open-pixel mask); a* = 0.98 x best fixed-T exact match; cost = pixel-steps
  npix x (t + r_head x floor(t/4)) (heads decide every 4 steps; simple rules decide every step).
- Cost at a* = the cheapest knob of the rule with mean accuracy >= a* (no interpolation).
- Oracle at a*: cheapest subset of solvable examples, each at its earliest exact step (others cost 0).
- Bootstrap: 2000 resamples; each draws a (backbone seed, head seed) pair and multinomial example weights;
  a* and the envelope are recomputed in every resample. Holm correction over (R8, R9s, R7) x pools.

## Criteria (unchanged from the spec)
S2, S1, S3 and F1-F4 as in the spec. Verdicts: Gain (mechanism) if S2 and S3 hold; Gain (mazes) if S1 also
holds; Informative if only S3 holds; E-P1-2 = "No gain (no headroom)" if the headroom gate fails.

## Headroom gate (mazes, before any maze head arm is scored)
On the dev splits at 33 (200 mazes) and 59 (100 mazes): headroom = 1 - oracle/envelope in pixel-steps at a*,
envelope over R0, R2 (lag {1,2} x patience {1,8}) and R3 (sum, min, entropy). Within one size R0s and R0L are
fixed-T rules, so they cannot lower the single-size envelope below R0; they enter only on the mixed pool.
Gate fails if headroom < 30% at both sizes. The result is appended below as an addendum, with a timestamp.

## Validity gates
Maze reproduction (13 and 59 on a split disjoint from dev/pool/eval, indices 1500+), T2m backbone gate
(resolvable exact >= 0.95 at L=64, >= 0.90 at L=128, blocked mean|p-0.5| <= 0.1; one retry at w=128),
head capacity (predicted vs realized totals, R7/R7abl at H=4 and 16), unit tests (a)-(e).

## Deviations and simplifications (stated in advance)
1. Location: puzzles/P1/ instead of voc-halting/ (task-level instruction).
2. Test-time fitting streams over the pool (32 shared decision times per batch, 2 passes per batch) instead of
   storing all pool features (about 52 GB at 59).
3. Head seeds share the training trajectories and decision times; only init and order differ.
4. R5 (PonderNet) is T2m-only. R13's cost grid is 200 log-spaced c over the training range, not dev quantiles.
5. Maze heads decide every 4 steps; the 9x9 head-training cap is 160 steps (decisions at even t <= 96).
6. Knob grids: T2m from a 256-example training-distribution dev pool (L=64 and 128) per backbone seed and
   head seed; mazes from the dev split at the same size, after test-time fitting.
7. The paper value for 33 is not available; the reproduction gate uses 13 (99.88%) and 59 (97.30%, peak at
   iteration 984 > T_cap 708, so 59 is run to T=1000).
8. The CPU was at load 110-120 on 16 cores during phase 0, so the T1 smoke test and the pipeline smoke tests run
   on the VM; unit tests (a)-(e) run on CPU.

## Addendum A (2026-09-29 13:10Z, before any gate, head or pool result was inspected)
Analysis details fixed before looking at results (analyze.py):
1. GPU sharing: every step on the VM caps its memory with torch.cuda.set_per_process_memory_fraction
   (P1_GPU_FRAC). This changes cuDNN algorithm choice (speed, tiny numerical differences), not the method.
2. S2 "at least 20% cheaper" is read strictly: the lower 95% bound of 1 - cost(R7)/cost(best competitor)
   must exceed 0.20. The lenient reading (point >= 0.20 and lower bound > 0) is also reported.
   "Best competitor" = the cheapest at a* among R0, R2 x4, R3 x3, R4, R4c, R4e, R5 (both), R11, R12, R13,
   recomputed per resample. R8 within 10% of R7 = upper bound of cost(R8)/cost(R7) - 1 below 0.10.
3. S3 and F3 use all 7 KL horizons (2..cap) of R9, at decision times t < T_cap, pooled over head seeds.
4. T2m oracle: per-example stop at argmax_t S_i(t) - lambda t (lambda on a log grid); the oracle cost is the
   smallest mean t whose mean score reaches a* (the maze oracle stays as written above: exact match).
5. Calibration CIs use 200 bootstrap resamples over examples (costs and savings use 2000).
6. Matched compute: budgets at 0.5x, 1x and 2x the oracle's mean cost; best accuracy of a rule whose mean cost
   is within the budget; AUC = mean accuracy over 25 log-spaced budgets in [0.5x, 2x].
7. Slices: 2-cycle = KL lag-1 at T_cap > 1e-3 and KL lag-2 < 0.1 x KL lag-1 (totals over positions);
   overthinking = exact at some t < T_cap but not at T_cap; never-solved = no exact t <= T_cap.
8. Savings with an unreached a* are clipped to [-10, 1] (a rule that never reaches a* counts as -10).

## Addendum B (2026-09-29 13:40Z): deviation 9, backbone warm-up curriculum (before any gate result)
Observation: with the spec recipe (L in {64,128}, d ~ U[L/8, L-1], 10k iterations) both backbone seeds sat at
chance (final-step loss 0.692-0.705 nats/bit, = ln 2) for their first ~4.4k iterations. Diagnostics on the VM
(w=64, bs 64, same model and loss; logs in results/vm/.../diag): plain L=16 stays at chance for 3k iterations;
a d-ramp (d from 1-2 upward) at L=32 stays at chance for 1.3k; a d-ramp at L=16 reaches 100% resolvable
exact by ~700 iterations. The two-bit parity has no first-order gradient signal until both bits meet, so the
recipe needs short distances first.
Change: both backbone seeds were stopped and restarted with a 3k-iteration warm-up (data.t2m_curr_batch:
L=16 with d_max ramping 2->15, then L=32, L=64, then L in {64,128}, d ~ U[1, d_max]) followed by 7k
iterations of the spec distribution (10k total, same optimizer and schedule). The w=128 retry, if needed,
uses the same warm-up. The gate and every evaluation pool are unchanged (spec distribution, d >= L/8).
The frozen backbone is not the object under test; this changes only how it is trained.

Note (2026-09-29 13:40Z): the first warm-up restart (13:33Z) resumed from stale no-warm-up partial checkpoints
that the resume push had restored, so it skipped the warm-up. It was stopped, the partials were moved aside
(and trashed locally), and both seeds were restarted clean at 13:39Z (log starts at it 0, L=16). No gate or
evaluation had run on either backbone.

## Addendum C (2026-09-29 13:47Z): maze headroom gate result (recorded before any VOC head was trained or evaluated)
The gate FAILED. On dev (a* = 0.98, best fixed accuracy 1.0 at both sizes), headroom = 1 - oracle/envelope in
steps = 0.059 at n=33 (envelope R2:2:1 104.6 steps vs oracle 98.5; N=200) and 0.053 at n=59 (R2:2:1 280.3 vs
oracle 265.4; N=100); threshold 0.30. As pre-registered, E-P1-2 is "No gain (no headroom)", the maze pipeline
runs only sizes 59 and 99 (for Q3, the label-free settlement diagnostic), and the mixed pool is not built.
Source: results/vm/trader-cat-p1-1/maze/headroom_gate.json.

## Addendum D (2026-09-29 13:56Z): deviation 9b, backbone recipe v2 (before any gate result)
With the Addendum B warm-up both seeds learned at L=16 and L=32 (final-step loss 0.27-0.40) but became
unstable once L reached 64 (OneCycle lr still ~0.9e-3): the loss oscillated between 0.35 and 0.8 and both seeds
sat at chance (0.69-0.72) from it ~3000 on the spec distribution, where a chance-level net gets no gradient.
They were stopped at it ~3400 (logs: results/vm/.../diag/bb*.log.curr1). Recipe v2 (train.py --recipe v2,
data.t2m_curr2_batch, v2_lr_factor), 8000 iterations: L = 16 (d_max ramp 2->15 over 700 it) until it 1000,
32 until 1600, 48 until 2200, 64 until 3000, 96 until 3600, then L in {64,128}; d ~ U[1, L-1] until it 6000,
then the spec distribution (d ~ U[L/8, L-1]) to the end. lr 1e-3 (100-it warm-up) while L <= 32, then 3e-4 with a
cosine decay to 9e-6. Same model, loss, batch, gate, w=128 retry rule and evaluation pools. 8000 instead of
10000 iterations to fit the VM's 6-h limit. Step names bb{s}v2 (t2m_run_vm.sh, next to run_vm.sh's maze chain).
Update (14:01Z, before any gate result): in the first v2 run seed 0 did not learn at L=16 by it 1000 and moved on
at chance (seed 1 learned and stayed stable through L=64 at lr 3e-4; logs diag/bb*.log.v2noholds). Both seeds were
restarted with stage holds: at the end of each curriculum stage (it 1000/1600/2200/3000/3600) the stage is
repeated while the EMA (0.98) of the final-step loss is above 0.5 (chance = 0.69), for at most 3000 extra
iterations in total; the lr and data schedule follow the recipe position, so a held run only takes longer.

## Addendum E (2026-09-29 14:43Z): spot preemption and resume (procedural, before any gate or pool result)
trader-cat-p1-1 was preempted at 14:32:14Z (GCE system event; termination action DELETE, disk removed with it).
The last verified sync was 14:25:57Z. Lost: backbone iterations after the 3999 partial (bb0 at recipe step ~5500,
bb1 ~5800) and the first 22 min of the size-59 test-time fit; no gate, pool or head result was produced after the sync.
Resume on trader-cat-p1-2 (resume_run_vm.sh): /root/out restored from the p1-1 sync; maze data for 59 re-downloaded
and 99 regenerated with the same seeds; bb{0,1}v2 resume from their it-3999 partials (model, optimizer, generator,
hold delay and EMA restored), so the recipe and data stream are unchanged; msize59 and msize99 start over.
Backbone partials are now written every 500 iterations (atomic rename) instead of every 2000.

## Addendum F (2026-09-29 15:15Z): head-target normalization (before any T2m head, capacity or pool result)
A CPU smoke run of `eval.py t2m` (tiny pools, a partial backbone) showed head losses of ~3800 (R4, R7) and 1.9e9 (R8):
each arm's per-horizon target scale was taken from ONE random decision time of the first batch, which can be
near-degenerate (e.g. all targets ~0 at a late time), and it also mis-weights horizons in R7's shared trunk.
Fix (train_heads.py fit_batch): the scale is now the SD of the targets pooled over ALL decision times of the first
batch (unit test: losses ~1-14, R7 scales increase with H). Same arms, targets, optimizer, batches and pools.
Pushed to trader-cat-p1-2 at 15:13:52Z (sha256 a4ee6e9d0e7f...) while both backbones were still training, so every
T2m head is trained with it. Maze heads (step mheads, trained on p1-1 before this fix) keep the old single-time
scale and are not retrained: the maze capacity result (R7 Spearman 0.31-0.50 at H >= 2) and the Q3 KL-settled
predictions may be affected by it and are reported with that caveat. Mazes had already failed the headroom gate.

## Addendum G (2026-09-29 15:50Z): second spot preemption; partial checkpoints (procedural, before any head or pool result)
trader-cat-p1-2 was preempted at 15:32:27Z (disk auto-deleted). The 15:29:40Z sync holds both final backbones and
their gates (both pass: resolvable exact 1.00 at L64 and L128, blocked mean |p-0.5| 0.003-0.016), so no backbone work
was lost. Lost: T2m head training (batch ~60 of 125, both seeds, restarted from scratch with the same seeds) and
44 min of the size-59 test-time fit. Resume on trader-cat-p1-3 (resume3_run_vm.sh). To bound further losses, eval.py
now writes partial checkpoints: T2m head training every 10 batches and the maze TT fit after every batch (heads,
optimizers, target nets, head-set generators, loop generator and torch RNG), and maze pools in chunks of 4 batches.
A local test shows a resumed T2m head run is bit-identical to an uninterrupted one. Same data, seeds and settings.
Update (16:20Z): the maze resume test (size 13, 5 pool and 5 dev mazes, batch 1, simulated crashes in the TT fit and
in the second dev chunk) was too slow on General's loaded CPU (18 min per TT batch), so it ran on trader-cat-p1-3's GPU
with deterministic cuDNN, next to the T2m heads. Resumed vs uninterrupted: TT heads max diff 0.0, obs and eval
arrays max diff 0.0. msize59 started only after it passed (GO_MAZE at 16:19Z).

## Addendum H (2026-09-29 16:40Z): third spot preemption; T2m results recorded (procedural; no criterion changes)
trader-cat-p1-3 was preempted at 16:31:54Z (disk auto-deleted). The T2m part had finished (all heads, capacity
reports and L64/L128/L256 pools for both backbone seeds, synced and sha256-verified at 16:29:13Z), so T2m is not
rerun. The size-59 test-time fit had checkpointed batch 6 of 16 (maze/heads_tt_n59.pt.partial, synced 16:29:01Z).
Resume on trader-cat-p1-4 with resume4_run_vm.sh: maze chain only, maze data for 59/99 re-created with the same
seeds, msize59 resumes the TT fit from the partial, then msize99.
Recorded before any maze pool result: the T2m capacity gate FAILS as specified (R7 Spearman 0.46-0.63 at H in {4,16},
mean ratios 0.79-1.50), so under the validity-gate rule T2m cannot count as a falsification. The primary T2m pools
were analysed (2000 resamples) before this addendum. Any split of the capacity statistics by instance type in
RESULTS.md is a post-hoc diagnostic; it changes no gate, criterion or verdict rule.
Update (16:57Z): trader-cat-p1-4 (us-east1-b) was preempted at 16:50:04Z after 10 min, before any sync (TT batches
7-9 lost); L4 spot was out of stock in 15 other zones. Resumed from the same batch-6 checkpoint on trader-cat-p1-5
(us-east1-b). A local rsync pull of the maze checkpoints every 150 s now backs up the 15-min verified sync.
Update (18:45Z): size 59 finished on trader-cat-p1-5 (TT fit, dev and eval pools, sha256-checked against the VM).
trader-cat-p1-5 was then preempted at 18:31:42Z, 3 min after dev_n99 chunk 0; the TT fit at 99 (heads_tt_n99.pt)
and that chunk were already pulled. Resumed on trader-cat-p1-6 (us-east1-d) with the same resume4_run_vm.sh: maze
data for 99 re-generated from the same per-chunk seeds, msize59 skipped by its marker, msize99 continues at dev chunk 1.
A preliminary size-59 analysis (analyze.py, 2000 resamples) was run on the pulled files while 99 ran; nothing on the
VM depends on it and no code, knob grid or criterion was changed after it. analyze.py gained a --procs option (a fork
pool over bootstrap chunks and calibration arms); a check on a stand-in pool gave byte-identical results with 1 and 5
processes.
Update (19:12Z): trader-cat-p1-6 was preempted at 19:02:41Z after dev_n99 and eval_n99 chunk 0 (both pulled and
loadable). Resumed on trader-cat-p1-7 (us-central1-b) at 19:11Z with the same script; the remaining 4 eval chunks at 99
run there. Spend before p1-7: $4.06.

## Addendum I (2026-09-29 21:42Z): float16 overflow at maze size 99 (found after the final analysis; post hoc)
All pools finished on trader-cat-p1-7 (teardown 20:18Z; no trader-cat-p1 instance or disk left). The final analysis
(analyze.py, 2000 resamples, 21:30Z) showed NaN harmful shares and 0 or NA bins at H=64 and cap for size 99 only.
Cause: eval.py stores the realized KL sums K (line 125) and every head output as float16, whose maximum is 65504.
Summed over the open pixels of a 99x99 maze, K at the longest horizons exceeds it and is stored as +inf
(6.6% of realized K at cap; 3-6% of R9's cap predictions). Size 59 is not affected (maximum 57056), nor is T2m.
Treatment, fixed before looking at its result:
- At 99, calibration and the harmful share are read at H <= 32 only; H=64 and cap are reported as not computable.
- Stored +inf can only make a VOC rule continue, so the halting costs at 99 are an upper bound. sens_f16.py reruns
  the size-99 analysis with +inf replaced by 65504 (the other bound; 200 resamples) and writes
  results/sensitivity_f16_n99.json. Criteria are not changed; the verdict counts only if both bounds agree.
Result: they agree. R9 costs 1001 (main) and 990 (bound) at a*; R9 - R7TT at matched compute is -0.09 / -0.12 / +0.10
(main) and -0.09 / -0.24 / +0.11 (bound) at 0.5x / 1x / 2x the oracle budget; the harmful share with the bound is
at most 15.9% at every H (H=64 2.5%, cap 1.0%). S3 fails and F3 fires at 99 either way. With both sides clipped
at 65504, the top-bin ratios at H=64 and cap (about 1.0) are artifacts and are not used.
