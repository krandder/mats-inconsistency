# P15 - Local EM on a tied discrete cellular automaton at depth 8-32

**Verdict: No gain.** Likelihood-message settlement (Bl, local EM, no cross-level gradient) matches plain
backprop-through-depth (A) on the prefix task at depth 16. It does not keep up at depth 32. It loses clearly
on mazes. It never recovers the exact rule, so it does not extrapolate better. Two pre-registered gates fail
(G3 on mazes; G5 on our retrained DT-recall, as scored), so the maze task and every comparison against P-* are formally void.

Numbers below come from `results/analysis.json` (re-run with `python3 analyze.py report`, checked at
2026-09-30 05:48Z). Intervals are 95% t-intervals over seeds unless marked CP (Clopper-Pearson).

## Answers

| Question | Prefix (mod 2) | Maze | Pre-registered call |
|---|---|---|---|
| Q1 / S1 keeps up: Bl in-dist exact >= 0.99 | D16: 10/10 seeds (pass). D32: 1/5 (fail; A 4/5) | D32: 0/5. D16: 0/5 | **S1 false on both tasks** |
| Q1 / S1 gap does not grow: Delta_D = logloss(Bl) - logloss(best A-arm), nats/cell | D8 0.103, D16 0.394, D32 0.406 (needs <= 0.123). FLOPs ratio to 0.05 nats: not defined, Bl never reaches 0.05 | D16 0.241, D32 0.291. Growth not evaluable (maze D8 dropped) | growth false (prefix); wall-clock deltas 0.102 / 0.393 / 0.270 |
| Q2 / S2 extrapolates better (primary cell, frozen c x size rule) | 1024 bits: Bl 1.000 vs A 1.000, diff 0.000, solves 10/10 vs 10/10, Fisher p = 1.0 | 59x59: Bl 0.000 vs A-ds 0.128 (best non-P arm; P-pub 1.000 is void under G5), Fisher p = 1.0 | **S2 false on both tasks** |
| Q2 rule recovery (automaton equivalent to the true rule) | Bl 0/10, A 5/10, A-1step 3/10, O 5/5 | Bl 0/5, O 4/5 | Bl finds a working but non-equivalent machine |

Falsifiers: **F1 fires** (prefix: Delta rises monotonically to 0.41 > 0.05; maze: Bl and Bl-ac both < 0.9 in
5/5 seeds). **F2 fires** (the best baseline is >= Bl - 0.02 at every test size on both tasks). **F3 fires**
(prefix: A and A-1step score at least Bl - 0.05 at 1024 bits; maze: A, A-ds, A-soft-prog, Bl-local and
Bl-shuffle all score at least Bl - 0.05 at n=59, since Bl scores 0.0).

## Gates

| Gate | Result | Consequence |
|---|---|---|
| G0 data (lookup/majority baselines, maze round-trip, KS vs easy-to-hard n=9) | pass | - |
| G0_C frozen random trader | prefix 0.001 exact (pass); maze 0.000 exact (pass) | - |
| G1 message = exact gradient/posterior (float64) | pass, max err 5e-15 | - |
| G2 exact posterior on a chain | pass, max err 1e-16 | G2b (reported only): error <= 1.4e-3 by level |
| G3 true rule is a fixed point of settlement | **prefix pass; maze FAIL** | per pre-registration the run is void until fixed; not fixed in budget, so maze results are void |
| G4 analytic FLOPs vs counted | pass (Bl prefix D16: 5.8% off) | - |
| G5 external bar (DT-recall) | port and published nets pass; **retrain FAIL as scored** (centred 1/3 reach 0.92; the uncentred ablation gets 2/3) | S2/F2 against P-* void; the verdict holds against the non-P arms |
| G6 stacked seeds = single seed | pass (diff 1.6e-14) | - |

G3 maze detail (`results/g3_margin_scan.json`, `results/vm_out/gates/G3*.json`): with the true dead-end-filling
rule planted, one flipped "dead end kept" pattern lowers log p(y) after a Bl step, so the true rule is not a
fixed point of Bl's settlement on mazes. That is a blind spot of the method on this task, not a bug we found.

G5 numbers (published: 97.12% at 512 bits, 97.30% at 59x59, both at oracle T):
- Published checkpoints through our eval: 512 bits 1.000 (T=225, CP lower 0.998); 59x59 mazes 1.000
  (T=700, CP lower 0.988). Port logits match to 4e-7.
- Our centred retrains (Bansal's prefix recipe, 3 seeds): peaks 0.406 / 0.944 / 0.342 at 512 bits; only 1/3
  reaches 0.92, so G5 fails as scored. Uncentred ablation (same recipe, no centring): peaks 0.000 / 0.996 / 0.957,
  so 2/3 reach 0.92. Centring made no clear difference (1/3 vs 2/3, 3 seeds each); the seed-0 uncentred
  failure that prompted it looks like seed variance, and the recipe passes in 3 of 6 retrains overall. Had the uncentred set been scored, G5 would pass.
  The verdict does not depend on it: S2 and F2 give the same answer against the non-P arms.
- Other possible cause of the unstable retrains: our schedule is x0.1 at epochs 60 and 100 (Bansal's repo); the
  spec text says "x0.01".
- Published maze net on our maze sets (one net, c = 1 frozen on val): 1.000 exact at n = 19 / 37 / 59.
- `results/vm_out/gates/G5_pub.json` says pass=false because its prefix part used raw 0/1 bits; the
  centred re-eval in `G5_pub_prefix.json` supersedes it (1.000 at 512 bits).

## Main table (in-distribution greedy exact, mean [95% CI]; test at the largest size under c x size)

| Cell | Bl | best backprop arm | O (oracle targets) | others |
|---|---|---|---|---|
| prefix D8 | 1.000, 5/5; test 8192 bits 1.000 | A 1.000 | - | A-ds 0.998, A-prog 0.972 |
| prefix D16 (primary, 10 seeds) | 1.000, 10/10; ll 0.395; test 1024: 1.000 | A 1.000, 10/10; ll 0.001; test 1.000 | 1.000 | A-1step 0.992 (8/10); A-soft-prog 1.000 (test 0.90); A-prog 0.854; A-ds 0.766; Bl-ac 0.890; Bl-local 0.677; Bl-shuffle 0.002; C 0.001; P-eq 0.624 (test 0.005); P-pub (4 nets) test 1.0 / 0.81 / 0.48 at 64/256/1024 |
| prefix D32 | 0.734 [0.50,0.97], 1/5; ll 0.533; test 2048: 0.769 | A 0.798, 4/5; test 0.800 | - | A-ds 0.775, A-prog 0.745 |
| prefix D16 mod 3 | 0.936, 2/5; test 1024: 0.800 | A-prog 0.899; test 1.000 | 1.000 | Bl-ac 0.727, A-ds 0.589 |
| maze D16 | 0.580 [0.25,0.91], 0/5; test 59: 0.000 | A-prog 0.913; test 59: 0.300 | - | A 0.776 (test 0.280); A-ds 0.968 (test 59: 0.044; best arm is picked by log loss, where A-prog wins) |
| maze D32 (primary, 5 seeds) | 0.220 [0.04,0.40], 0/5; test 59: 0.000 | A-ds 0.830; test 59: 0.128 | 1.000; test 59: 0.996 | A 0.107; A-soft-prog 0.152 (one seed 0.76; test 59: 0.192, the best 59x59 score of any arm trained here without oracle targets); A-prog 0.227 (seeds 0 / 0 / 0.25 / 0.05 / 0.84; test 59: 0.000); A-1step 0.014 (test 59: 0.000); P-eq 0.928 (test 0.000); Bl-ac 0.097; Bl-local 0.039; Bl-shuffle 0.000; P-pub 1.000 at 19 / 37 / 59 (void: G5); C 0.000 |

Bl's sampled log loss stays high even where greedy exact accuracy is 1.0 (0.40 nats/cell at prefix D16):
its latent posteriors stay spread out. Halting variants change nothing material: at prefix D16, Bl gets 1.000
under fixed-point halt, c=4, c=16 and oracle T alike.

Latent diagnostics, Bl prefix D16 (per level k): nmi(z_k; g_k) falls from 0.53 at k=1 to 0.08-0.29 after,
about 12 states in use per level, KL(q||p) 0.005-0.037, entropy of q 0.70 falling to 0.31 nats, mean |lambda|
rising from 0.15 at k=1 to 0.89 at k=15 (0.47 at the top level). Bl-shuffle
(mismatched messages) sits at chance, so the messages carry the signal; Bl-local (local targets only)
reaches 0.68, so the cross-level message adds about 0.3 exact at D16.

## Compute

- Equal analytic training FLOPs per cell: B_D = Bl's FLOPs for 6000 steps at batch 32 (prefix D8/16/32:
  3.6e11 / 1.49e12 / 6.08e12; maze D16/32: 9.55e12 / 4.35e13). Backprop arms get 12k-47k steps for that.
- Tuning FLOPs: maze 2.42e15, prefix 8.29e13. Process-hours: tuning 7.22 h, pilot 2.17 h.
- Seconds per step on an L4 (prefix D8/16/32): Bl 0.021 / 0.036 / 0.067; A 0.019 / 0.033 / 0.062.
- Peak memory per step (prefix, B=32): Bl 23 / 46 / 690 / 11012 MiB at D = 16 / 32 / 128 / 1024; A 3.4 / 12.9 /
  196 / 12757; A-1step 0.5 / 1.1 / 4.4 / 38; checkpointed BPTT 1.5 / 3.7 / 23 / 475. Bl saves no memory
  against BPTT at these depths; checkpointed BPTT is 20x cheaper.

## Secondaries

- 1000x prefix point (D=8, 8192 bits): Bl 1.000, A / A-ds / A-prog 1.000.
- mod-3 prefix D16: see table; Bl is below A-prog at test size.
- market-resnet phase 3 (read-only copy, `results/market_resnet_p3_dtr.json`): DT-recall at equal FLOPs solves
  about 1.0 at 1x / 10x / 100x train length on prefix parity; not rerun here.
- Maze n=72: skipped (budget).

## Simplifications (from the spec)

- Maze D8 dropped, so maze growth is "not evaluable". Maze primary cell at 5 seeds, not 10.
- Sampled-mode extrapolation not measured; greedy only.
- Maze test sets come from our generator (KS-matched to easy-to-hard at n=9, p = 0.71 / 0.24), 200 / 50 / 50
  mazes at n = 19 / 37 / 59. Test T capped at 4 x size.
- P-pub prefix = published net plus 3 centred retrains (4 nets, so the CP rule applies).

## Caveats

- Spot churn: 11 of 14 VMs were preempted (VM-1/2/3/5/6/7/9/10/11/12/13, from GCP operations). Jobs resumed
  from checkpoints (every ~10 min) on new VMs. VM-12 and VM-13 were preempted during startup in us-central1;
  the watchdog then sat in its 25-min startup wait until I patched it to detect a startup preemption, and the
  last lane moved to us-east1 (VM-14). VM-11 lost the unsynced A-1step and C maze D32 progress and the P-pub
  maze eval, which were rerun.
- CUDA OOM: A-prog and A-1step maze D32 ran out of memory at eval while the P-pub maze eval (15.7 GiB) shared
  the L4; both were re-evaluated from their final checkpoints. Earlier, some jobs were rerun after OOM with two
  jobs sharing one L4, and two maze jobs were moved to a second VM.
- The G5 retrain first ran with uncentred inputs, and I called that a bug after seed 0 scored 0.0; the centred
  rerun is the one scored. Seeds 1-2 of the uncentred run later passed (see G5), so the "bug" call was wrong.
- A watchdog bug (it could not see a finished VM) left one VM idle for 20 min (~$0.17); fixed during the run.
  The final VM idled 6 min between plan end and deletion (5-min watchdog cycle).
- Maze results are void under G3 (pre-registered), and every comparison against P-* is void under G5. The
  prefix results alone carry the verdict.

## Cost

**$9.26 at the $0.52/h published spot price** (17.81 VM-h from GCP operations, insert to preempt/delete;
`python3 gpu/actual_spend.py`). $15.31 if every hour were billed at the $0.86/h ceiling. The watchdog ledger
(`gpu/LEDGER.md`, 19.36 VM-h, $10.07 at spot) overcounts gone VMs until it noticed them; the budget file keeps
the $10.07. Cap $20, watchdog at $17. All 14 VMs and their disks are deleted (`./gpu.sh check` empty, gcloud
instances and disks lists empty, checked 2026-09-30 05:47Z). No paid LLM APIs.

## Files

- Code: `p15.py` (arms, settlement, eval), `p15_dtrecall.py`, `g5.py`, `gates.py`, `g0.py`, `mem.py`,
  `analyze.py`, `runq.py`, `run_vm.sh`, `gpu.sh`, queue files `queue_*.json`, frozen rule `frozen.json`, task spec `spec.txt`.
- Results: `results/analysis.json`, `results/charts.json`, `results/vm_out/results/*.json` (per job),
  `results/vm_out/gates/*.json`, `gates/*.json` (CPU gates), `results/g3_margin_scan.json`.
- Final checkpoints: `results/vm_out/ckpt/full_*.pt` (tuning, timing, smoke and pilot checkpoints were trashed).
  The six DT-recall retrain weights (net only, 72 MB) are archived at
  `/home/agent/data/archive/p15-ppub-retrains-20260930/`.
- GPU log: `gpu/LEDGER.md`; spend check `gpu/actual_spend.py`.
