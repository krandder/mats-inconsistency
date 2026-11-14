# P12 results: in-loop flow BP for a DT maze solver on weighted mazes, under size extrapolation

Run on 2026-09-29 on spot L4 VMs trader-cat-p12-1 to -5 (13:55Z to 21:10Z); CostNet+BB and A0-flow were evaluated on CPU on General afterwards. The GPU-LEDGER section "P12 flow-BP mazes" has the rows and the delete check.

## Verdict

**No gain. The pre-registered falsification holds.**

- **Primary cell** (n in {33, 49, 65} pooled, shortest-path length L > 21, the 9x9 pool's p99). DTxARB finds the exact shortest path in 2.5% of mazes, t 95% CI [-0.5, 5.5]. The per-seed best baseline (selected on val) averages 3.1%; it is the post-hoc decoder DT+dec in seeds 0 and 3 and the post-hoc flow BP DT+ARB(test) in seeds 1 and 2.
  - Paired difference: -0.6 points, t 95% CI [-3.5, 2.3] (n = 4 seeds). Hierarchical bootstrap: [-2.6, 1.1].
  - The success bar was +5 points. The MDE from val is 16.6 points (sd 11.9, n = 4), so this study could not have detected a gain of that size anyway; that is secondary, because the second falsifier settles it.
- **Valid paths at n = 49:** DTxARB 1.5% (the bar was 99% for success; below 95% falsifies). At n = 49 the network no longer outputs a connected S-to-G path in 98.5% of mazes.
- **In distribution, DTxARB does help.** At n = 9 it is exact in 65.8% of mazes against 56.8% for DT and 58.6% for DT+fb, and it outputs a valid path in 94% against 66% and 67%. The prices, not the proposer, carry this: decoding DTxARB's own proposer u gives 0% exact in the primary cell, and the prices b = u + sg(kappa tanh(delta / kappa)) give 2.5%.
- **But a post-hoc decoder on plain DT does more, and needs no training change.** DT+dec (min-cost path on -log sigma(u) + c) and DT+ARB(test) (flow BP on DT's logits at test time) reach 73.4% and 73.2% at n = 9 and 58.5% at n = 15, above DTxARB (65.8%, 40.2%), and they stay at or above it at every size (primary 3.1% and 2.9% against 2.5%).
- **The reference arm that does extrapolate is CostNet+BB** (not one of the pre-registered comparison baselines; 1 seed, evaluated from its step-9000 checkpoint). It is a small local net (4.6k parameters) that predicts each cell's cost, followed by an exact shortest-path solver. It is exact in 27.0% of the primary cell against 2.5% for DTxARB, and in 40.5% / 18.0% / 9.0% of mazes at n = 33 / 49 / 65 against 4.6% / 0.5% / 0.2%. Its paths are always valid, and it is also best in distribution (78.0% at n = 9).

## Primary table (mean over seeds, t 95% CI)

| arm | seeds | primary exact % [95% CI] | primary valid % |
|---|---|---|---|
| DT | 4 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| DT+fb | 4 | 0.2 [-0.3, 0.8] | 0.4 [-0.6, 1.3] |
| DTxARB | 4 | 2.5 [-0.5, 5.5] | 9.9 [-2.1, 22.0] |
| DT+dec | 4 | 3.1 [2.5, 3.7] | 100.0 [100.0, 100.0] |
| DT+ARB(test) | 4 | 2.9 [2.1, 3.8] | 82.7 [79.5, 86.0] |
| ARB-kappa | 4 | 0.0 [0.0, 0.0] | 0.2 [0.2, 0.2] |
| CostNet+BB | 1 | 27.0 (n=1) | 100.0 (n=1) |

CostNet+BB is one seed, evaluated from its step-9000 checkpoint (of 12,000), so it has no interval.

## Exact / valid path rate by maze size (%, mean over seeds)

The selected T = k n per seed; k chosen on val (n = 25, 200 mazes). Shift sets: extra open walls with probability p, n = 33.

| arm | n=9 | n=15 | n=25 | n=33 | n=49 | n=65 | shift_p0.1 n=33 | shift_p0.5 n=33 |
|---|---|---|---|---|---|---|---|---|
| DT | 56.8 / 66 | 29.7 / 36 | 0.2 / 0 | 0.0 / 0 | 0.0 / 0 | 0.0 / 0 | 0.0 / 0 | 0.0 / 0 |
| DT+fb | 58.6 / 67 | 36.9 / 47 | 4.9 / 7 | 0.5 / 1 | 0.0 / 0 | 0.0 / 0 | 0.9 / 1 | 0.0 / 0 |
| DTxARB | 65.8 / 94 | 40.2 / 75 | 11.4 / 35 | 4.6 / 19 | 0.5 / 2 | 0.2 / 1 | 17.5 / 28 | 1.2 / 12 |
| DT+dec | 73.4 / 100 | 58.5 / 100 | 16.7 / 100 | 5.5 / 100 | 1.0 / 100 | 0.2 / 100 | 32.0 / 100 | 0.9 / 100 |
| DT+ARB(test) | 73.2 / 100 | 58.5 / 100 | 17.0 / 98 | 5.4 / 95 | 0.8 / 80 | 0.2 / 60 | 30.2 / 95 | 0.6 / 85 |
| ARB-kappa | 33.0 / 59 | 11.0 / 35 | 2.3 / 10 | 0.0 / 0 | 0.0 / 0 | 0.0 / 0 | 16.0 / 32 | 0.0 / 0 |
| CostNet+BB | 78.0 / 100 | 68.3 / 100 | 49.0 / 100 | 40.5 / 100 | 18.0 / 100 | 9.0 / 100 | 70.0 / 100 | 18.0 / 100 |

Non-learned references: Dijkstra is the label (100%). ARB-kappa (flow BP with the same cost kappa on every open edge, so it ignores weights) is in the table. A0-flow (flow BP on the true costs, run on CPU on n = 9 and 15 only) is exact in 99.2% and 99.0% of mazes: the BP layer is not the limit when its costs are right. A0-deg was not run.

## Readouts (no bar)

- **Error detection.** The summed semantic loss of the output flags wrong mazes in the primary cell with AUROC 0.918 for DTxARB (3 seeds with both classes) against 0.587 for DT+fb (2 seeds; DT has no exact maze there, so no AUROC). So the arbitrageur's prices know when the path is wrong, even where they cannot fix it.
- **Degree-valid but wrong (coherence pay is blind to it).** Of DTxARB's non-exact outputs at n = 33, 15% satisfy every cell's flow constraint but are still not the shortest path (a valid path with a detour, or a path plus a stray cycle). For DT and DT+fb it is 0% and 0.3%: they rarely produce anything coherent at that size.
- **Replication of the MATS post-hoc finding.** Post-hoc projection of DT's logits onto a path (DT+dec) adds +16.4 points at n = 9, +28.8 at n = 15, +16.8 at n = 25, +5.4 at n = 33 and under 1 point at n = 49 and 65. The effect is large where DT is partly right and vanishes where DT's logits are no longer informative.
- **Halting (n = 33).** A stability rule (the decode stops changing) halts DTxARB in 48-94% of mazes (by seed), at a median t of 60-305, but exact-at-halt is only 0-9%.
- **Overthinking.** Rare: DTxARB loses a path it once had in 0-9.5% of mazes at n = 33 (by seed); DT+fb in 0-1%; DT never had one to lose.
- **Exactness by path length (primary cell).** Every arm is at 0% for L > 100; for L below 40, DTxARB reaches 8.4%, DT+dec 6.6% and DT+ARB(test) 6.9%; for L 40-60 the order flips (1.5% against 3.8% and 3.5%).
- **Compute.** DTxARB has 241k parameters (DT 240k, DT+fb 296k). A DTxARB training run took 1.4-3.0 h of wall time on the shared 4-vCPU VM against 0.6-2.1 h for DT and 0.9-2.1 h for DT+fb.
- **Ablations: not completed.** DTxARB-noFB s0, DTxARB-bp s0 and DT+pen s0 were training when p12-3 reached its age limit, and both replacement VMs were preempted before their jobs started. Partial training logs only: DTxARB-bp s0 reached step 5500 of 12,000 with a median gradient norm of 2.6e4 before clipping (max 9e12), against 0.19 for DTxARB-noFB, so backprop through 30 BP iterations is badly conditioned; its training loss at step 5500 was 0.42 against 0.30 for DTxARB s0. DTxARB-noFB s0 reached step 3500 and DT+pen s0 step 500. None was evaluated.

## What the numbers say about the question

Flow BP inside the loop, as a stop-gradient arbitrageur that corrects the proposer's prices, does what a market-maker should do in distribution: it enforces the flow constraints (valid paths 94% vs 66%) and it knows when it is wrong (AUROC 0.92). It does not make the recurrent proposer extrapolate. At n >= 33 the proposer's logits carry almost no path information, and no amount of arbitrage between the cells recovers it; the post-hoc decoders are at the same floor. The bottleneck is the proposer's extrapolation (DT itself fails the 9x9 gate: 60% at w = 64, bar 90%), not the consistency layer.

CostNet+BB shows the split that works here. A local signal (each cell's cost, which means the same thing at any maze size) is priced by a small net, and a global mechanism that is exact at any size (the shortest-path solver) does the aggregation. With true costs, the same flow BP is exact in 99% of mazes (A0-flow). So the market layer is not the limit. What limits the result is whether the signals it aggregates are size-invariant. In trader terms: pay the traders for local, transferable signals, and let an exact clearing rule combine them. A recurrent trader that must carry the whole global state inside itself does not transfer.

## Deviations from the pre-registration

- **Gate failed.** DT s0 reached 57.8% (w = 32) and 59.8% (w = 64) exact on the 9x9 gate set, bar 90%. The grid ran at w = 64 (GATE_MAX_W = 64; w = 128 was skipped for time and budget). So every learned arm is weaker than planned, and the extrapolation test starts from a model that does not solve 9x9.
- **BP factor.** Each open edge is one 3-state variable (off, u to v, v to u), so a cell factor has 51 configurations (19 ordinary, 16 at S, 16 at G), instead of 8 binary arcs (70 / 56 configurations). The binary form allows 2-cycles, which learned positive log-odds exploit.
- **Unary and prices.** BP's unary is log sigma(u) - c (logprob, c = 0.3, tau = 0.3, damping 0.5), chosen by the CPU-phase proxy. Two price fixes were needed before the grid: b = u + sg(delta) instead of theta + sg(delta) (with theta <= -c the target was out of reach and the gradient was exactly 0), then the position limit kappa tanh(delta / kappa), kappa = 5, instead of a clamp at +-30 (the proposer had to post |u| > 30 to overrule a wrong arbitrageur and training diverged).
- **Training details.** DT+fb and DTxARB feed back 8 extra channels; the LR has a 200-step warmup and cosine decay to 10%; augmentation uses one dihedral map per group; a NaN guard skips non-finite updates.
- **Divergence rollback, added mid-grid.** DT+fb s1 (from step 3300) and DTxARB s2 (at step 2800) froze at loss 1e17-1e19 with every update skipped. Both were resumed from their last good checkpoint, and train.py now rolls back to the last good snapshot when a 100-step window is at least half skipped (at most 5 rollbacks). DT+fb s1 was restarted by hand from its step-3000 checkpoint. With the rollback in place, DTxARB s2 (resumed from step 2000) rolled back once, from step 3100 to 2800, and DT+fb s3 once, from 3700 to 3500. No other run needed one.
- **Validity** is defined by connectivity: the decoded edge set must contain a connected S-to-G path with degree 1 at S and G and 2 on the path; stray cycles make it invalid.
- **Post-hoc grids.** DT+dec: c in {0.01, 0.1, 0.3, 1, 3} x k. DT+ARB(test): 10 (tau, unary, c) settings x k. Both tuned per seed on val. ARB-kappa: 9 (tau, kappa) settings; its val grid was flat (1% at best), so the first setting was kept.
- **Reduced grid and seeds.** The projection exceeded 4 h, so the reduced grid applied; the VMs were CPU-bound (4 vCPUs, the L4 mostly idle), so throughput was lower still. Seeds: DT, DT+fb and DTxARB 4 each (seeds 0-3); DT+dec and DT+ARB(test) on those 4 DT checkpoints; ARB-kappa is deterministic; CostNet+BB 1 seed (seed 0), evaluated on CPU from its step-9000 checkpoint because no VM survived long enough to finish it. Seed 4 of the main arms was deferred once the falsification was locked (DTxARB valid at n = 49 was 1.7% with 3 seeds, so seeds 3-4 could not bring it to 95%). DT-TD and the U1 anchor were not run.
- **Interruptions.** trader-cat-p12-2 was spot-preempted at 16:02Z; resuming on p12-3 exposed a bug (CUDA RNG state saved on the GPU), fixed by saving it on the CPU. CostNet+BB s0 restarted twice from its checkpoint. p12-4 (20:55Z) and p12-5 (21:04Z), started to finish CostNet+BB s0 and the ablations, were both spot-preempted within 7 minutes, before their jobs ran.

## Cost

Estimated $4.88 at the on-demand rate ceiling ($0.72/h): p12-1 $0.07, p12-2 $1.30, p12-3 $3.35, p12-4 $0.09, p12-5 $0.07 (spot billing is lower; cap $8). p12-2, p12-4 and p12-5 were spot-preempted, the last two within 7 minutes of creation, so GPU work stopped there. All trader-cat-p12 instances and disks are deleted. Checked at 21:35Z: `gcloud compute instances list --filter=name~trader-cat-p12` and `gcloud compute disks list --filter=name~trader-cat-p12` (project GCP_PROJECT) both listed 0 items. The watchdog is stopped.

## Files

- `RESULTS.md`, `results/summary.json`, `charts/p12_charts.json`
- Code: `maze.py`, `make_data.py`, `flow.py`, `nets.py`, `train.py`, `evaluate.py`, `analyze.py`, `cpu_phase.py`, `tests.py`, `run_vm.sh`, `run_one.sh`, `gpu.sh`
- Per-run logs and evals: `results/vm/trader-cat-p12-*/` (logs, results/eval/*.json)
- CPU evals run on General: `results/eval/CostNet+BB_s0_w32.json` (from the step-9000 checkpoint), `results/eval/a0flow.json`
- Tests: `results/tests.json` (all pass: semantic loss equals brute force, BP max-sum MAP equals brute force on 24 loopy cases, A0-flow exact on 20/20 true-cost mazes, Bellman-Ford equals Dijkstra)
