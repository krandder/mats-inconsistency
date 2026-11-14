
## market-resnet (trader-mresnet-*, cap $20)
Rate estimate: g2-standard-4 + 1x L4 SPOT us-central1 + 100 GB disk ~= $0.32/h (conservative). Watchdog: market-resnet/watchdog.sh (deletes all trader-mresnet-* VMs when estimated cumulative spend >= $18). The cap covers all runs together: the watchdog sums hours of every trader-mresnet-* VM, deleted ones included; worst case with -3..-6 all running to their 12 h max-run is 0.64 + 4 x 12 x 0.32 = $16.0. Phases 1-2 final (2026-09-29 18:46, all nine VMs deleted, gcloud instances and disks lists empty): 13.3 VM-hours, est. $4.26 of $20. Phase 3 (-10 to -15, from 19:07): spot VMs with termination action STOP, so a preempted VM keeps its disk; market-resnet/p3ctl.sh restarts it (its startup-script resumes from the <=10-min checkpoints) and deletes each VM only after a sha256-verified sync; watchdog3.sh rsyncs ~/p3 every 5 min and keeps the $18 cutoff. A stopped VM costs only its disk (about $0.01/h).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 10:45 | trader-mresnet-1 | us-central1-a | g2-standard-4 + L4 spot, max-run 12h | 2026-09-29 12:31 (spot preempted, auto-deleted with disk) | 1.75 | 0.56 |
| 2026-09-29 12:12 | trader-mresnet-2 | us-central1-a | g2-standard-4 + L4 spot, max-run 12h (phase 2) | 2026-09-29 12:31 (spot preempted, auto-deleted with disk) | 0.26 | 0.08 |
| 2026-09-29 12:42 | trader-mresnet-3 | us-central1-b | g2-standard-4 + L4 spot, max-run 12h (phase 1 resume) | 2026-09-29 14:59 (sha256-verified sync of 79 files, deleted with disk, confirmed by gcloud list) | 2.28 | 0.73 |
| 2026-09-29 12:42 | trader-mresnet-4 | us-central1-b | g2-standard-4 + L4 spot, max-run 12h (phase 2 grid, memdepth, bptt deep-supervision arm) | 2026-09-29 16:32 (spot preempted, auto-deleted with disk; the 16:26 sync holds every result except the 3 bptt_cov_ds runs, lost in training and rerun on -8) | 3.83 | 1.23 |
| 2026-09-29 16:38 | trader-mresnet-7 | us-central1-c | g2-standard-4 + L4 spot, max-run 3h (rerun of the 3 bptt_cov_ds runs lost on -4) | 2026-09-29 16:39 (spot preempted 2 min after boot, before any work, auto-deleted with disk) | 0.03 | 0.01 |
| 2026-09-29 16:45 | trader-mresnet-8 | europe-west1-b | g2-standard-4 + L4 spot, max-run 3h (rerun of the 3 bptt_cov_ds runs lost on -4, bptt_cov_s0 L32000 eval, then the bptt_cov_dscov control, 3 seeds; us-central1/us-east/us-west L4 spot were exhausted) | 2026-09-29 18:02 (spot preempted, auto-deleted with disk; the 17:55 sync holds bptt_cov_ds s0-2, bptt_cov_s0 and bptt_cov_dscov s0-1 to L320; dscov s2 and the L3200/L32000 evals rerun on -9) | 1.28 | 0.41 |
| 2026-09-29 18:07 | trader-mresnet-9 | us-central1-b | g2-standard-4 + L4 spot, max-run 3h (bptt_cov_dscov: s2 retrain, s0-1 L3200/L32000 evals from checkpoints) | 2026-09-29 18:46 (sha256-verified sync of 14 files, deleted with disk, confirmed by gcloud list) | 0.65 | 0.21 |
| 2026-09-29 13:44 | trader-mresnet-5 | us-central1-c | g2-standard-4 + L4 spot, max-run 12h (phase 1 grid2 seeds 1-2 + Ctt_eps train-cost sweep 0.01/0.2) | 2026-09-29 15:48 (sha256-verified sync of 44 files, deleted with disk, confirmed by gcloud list) | 2.07 | 0.66 |
| 2026-09-29 13:47 | trader-mresnet-6 | us-central1-a | g2-standard-4 + L4 spot, max-run 12h (phase 2b bounties + priced-halting cost sweep 1e-3/3e-3/1e-2) | 2026-09-29 14:56 (sha256-verified sync of 42 files, deleted with disk, confirmed by gcloud list) | 1.15 | 0.37 |
| 2026-09-29 19:07 | trader-mresnet-10 | us-central1-b, then us-east4-a, then us-central1-a | g2-standard-4 + L4 spot, max-run 8h, stop-on-preempt + auto-resume (phase 3: economy-vs-backprop ablation + Bansal DT-recall baseline, seed 0; smoke + FLOP count; preempted 20:22 and restarted 20:26; preempted again ~21:30, three failed restarts, so the stopped VM was deleted with its disk at 21:40 (instances and disks lists empty) and recreated at 21:41 in us-east4-a with termination DELETE, max-run 4 h, resumed from the synced state; now under p4ctl.sh) | 2026-09-29 23:39 (P3_DONE, sha256-verified sync of 73 files, deleted with disk, confirmed by gcloud instances and disks lists; 1 relaunch by p3ctl.sh / p4ctl.sh after spot preemptions) | 4.20 | 1.34 |
| 2026-09-29 19:29 | trader-mresnet-11 | us-central1-a, then us-east1-c | g2-standard-4 + L4 spot, max-run 8h, stop-on-preempt + auto-resume (phase 3 ablation + DT-recall, seed 1; stopped by a preemption before any work, restarted 19:44; preempted again ~20:02, no capacity to restart, so the stopped VM was deleted with its disk at 20:07 and recreated at 20:10 in us-east1-c with termination DELETE, max-run 4 h, resumed from the synced checkpoints; now under p4ctl.sh) | 2026-09-29 23:24 (P3_DONE, sha256-verified sync of 71 files, deleted with disk, confirmed by gcloud instances and disks lists; 1 relaunch by p3ctl.sh / p4ctl.sh after spot preemptions) | 3.61 | 1.16 |
| 2026-09-29 19:32 | trader-mresnet-12 | us-west1-b | g2-standard-4 + L4 spot, max-run 8h, stop-on-preempt + auto-resume (phase 3 ablation + DT-recall, seed 2; five jobs moved to trader-mresnet-21 at 21:41; preempted ~22:03, four+ failed restarts in us-west1-b, so the stopped VM was deleted with its disk at 22:18 (instances and disks lists empty) and recreated at ~22:20 by launch4.sh with termination DELETE, max-run 4 h, resumed from the synced state; now under p4ctl.sh) | 2026-09-29 23:46 (P3_DONE, sha256-verified sync of 71 files, deleted with disk, confirmed by gcloud instances and disks lists; 1 relaunch by p3ctl.sh / p4ctl.sh after spot preemptions) | 4.14 | 1.33 |
| 2026-09-29 19:32 | trader-mresnet-13 | us-west1-b | g2-standard-4 + L4 spot, max-run 8h, stop-on-preempt + auto-resume (phase 3 belief regularizers: liquidity t / sqrt t, floor, L2, cap, trade cap; seed 0; preempted ~22:03, four+ failed restarts in us-west1-b; its remaining 4 deep-eval jobs and memdepth4 moved to trader-mresnet-15) | 2026-09-29 22:18 (stopped VM deleted with its disk, instances and disks lists empty; results from the last 5-min sync before the preemption) | 2.73 | 0.87 |
| 2026-09-29 19:34 | trader-mresnet-14 | us-east1-d, then us-central1-b | g2-standard-4 + L4 spot, max-run 8h, stop-on-preempt + auto-resume (phase 3 belief regularizers, seed 1; preempted ~21:00, two failed restarts, so the stopped VM was deleted with its disk at 21:09 and recreated at 21:10 in us-central1-b with termination DELETE, max-run 4 h, resumed from the synced state; under p4ctl.sh) | 2026-09-29 22:15 (P3_DONE, sha256-verified sync of 43 files, deleted with disk, confirmed by gcloud instances and disks lists) | 2.56 | 0.82 |
| 2026-09-29 19:34 | trader-mresnet-15 | us-west4-a | g2-standard-4 + L4 spot, max-run 8h, stop-on-preempt + auto-resume (phase 3 belief regularizers, seed 2; at 22:17 also took trader-mresnet-13's 4 deep-eval jobs and memdepth4) | 2026-09-29 22:34 (P3_DONE, sha256-verified sync of 66 files, deleted with disk, confirmed by gcloud instances and disks lists) | 2.97 | 0.95 |
| 2026-09-29 20:02 | trader-mresnet-16 | us-central1-b, then us-central1-a | g2-standard-4 + L4 spot, max-run 4h, termination DELETE; p4ctl.sh recreates it from the synced state after a preemption (phase 3b: the audit's controls: BPTT with VOC-supervised v, replica of the economy's optimizer, progressive-loss alpha 0 / 0.5, online economy, halting at L64 vs entropy, memory at fixed L=32; seed 0; spot-preempted ~21:29 (deleted by GCE with its disk), recreated by p4ctl.sh at 21:31 in us-central1-a) | 2026-09-29 23:49 (P3_DONE, sha256-verified sync of 64 files, deleted with disk, confirmed by gcloud instances and disks lists; 3 relaunches by p3ctl.sh / p4ctl.sh after spot preemptions) | 3.39 | 1.08 |
| 2026-09-29 20:02 | trader-mresnet-17 | us-central1-b, then us-central1-c, then us-west1-b | g2-standard-4 + L4 spot, max-run 4h, termination DELETE; p4ctl.sh recreates it from the synced state after a preemption (phase 3b: the audit's controls: BPTT with VOC-supervised v, replica of the economy's optimizer, progressive-loss alpha 0 / 0.5, online economy, halting at L64 vs entropy; seed 1; spot-preempted ~21:30, recreated by p4ctl.sh at 21:40 in us-central1-c and preempted again within minutes; p4ctl.sh relaunches it) | 2026-09-29 23:07 (P3_DONE, sha256-verified sync of 64 files, deleted with disk, confirmed by gcloud instances and disks lists; 2 relaunches by p3ctl.sh / p4ctl.sh after spot preemptions) | 2.71 | 0.87 |
| 2026-09-29 20:02 | trader-mresnet-18 | us-central1-b, then us-central1-c, then us-east1-c | g2-standard-4 + L4 spot, max-run 4h, termination DELETE; p4ctl.sh recreates it from the synced state after a preemption (phase 3b: the audit's controls: BPTT with VOC-supervised v, replica of the economy's optimizer, progressive-loss alpha 0 / 0.5, online economy, halting at L64 vs entropy; seed 2; spot-preempted ~21:30, recreated by p4ctl.sh at 21:35 in us-central1-c, set-up ssh timed out and the VM was preempted; p4ctl.sh relaunches it) | 2026-09-29 23:04 (P3_DONE, sha256-verified sync of 63 files, deleted with disk, confirmed by gcloud instances and disks lists; 2 relaunches by p3ctl.sh / p4ctl.sh after spot preemptions) | 2.70 | 0.86 |
| 2026-09-29 21:41 | trader-mresnet-19 | us-east4-a, then europe-west4-b | g2-standard-4 + L4 spot, max-run 4h, termination DELETE; p4ctl.sh recreates it from the synced state after a preemption (phase 3: re-evaluation only, on 200 fresh L3200 strings, of jobs moved off trader-mresnet-10, seed 0: tb1_rand, tb1, progcov, DT-recall eqtime / eqflop) | 2026-09-29 23:56 (P3_DONE, sha256-verified sync of 27 files, deleted with disk, confirmed by gcloud instances and disks lists; 2 relaunches by p3ctl.sh / p4ctl.sh after spot preemptions) | 1.07 | 0.34 |
| 2026-09-29 21:41 | trader-mresnet-20 | us-east4-a | g2-standard-4 + L4 spot, max-run 4h, termination DELETE; p4ctl.sh recreates it from the synced state after a preemption (phase 3: re-evaluation only, on 200 fresh L3200 strings, of jobs moved off trader-mresnet-11, seed 1: tb1_rand, tb1, progcov, DT-recall eqtime / eqflop) | 2026-09-29 23:28 (P3_DONE, sha256-verified sync of 27 files, deleted with disk, confirmed by gcloud instances and disks lists; 1 relaunch by p3ctl.sh / p4ctl.sh after spot preemptions) | 1.42 | 0.45 |
| 2026-09-29 21:41 | trader-mresnet-21 | us-east4-a, then europe-west4-b | g2-standard-4 + L4 spot, max-run 4h, termination DELETE; p4ctl.sh recreates it from the synced state after a preemption (phase 3: re-evaluation only, on 200 fresh L3200 strings, of jobs moved off trader-mresnet-12, seed 2: tb1_rand, tb1, progcov, DT-recall eqtime / eqflop) | 2026-09-29 23:43 (P3_DONE, sha256-verified sync of 27 files, deleted with disk, confirmed by gcloud instances and disks lists; 1 relaunch by p3ctl.sh / p4ctl.sh after spot preemptions) | 1.71 | 0.55 |
| 2026-09-29 21:41 | trader-mresnet-22 | us-east4-a, then us-west4-a | g2-standard-4 + L4 spot, max-run 4h, termination DELETE; p4ctl.sh recreates it from the synced state after a preemption (phase 3: re-evaluation only, on 200 fresh L3200 strings, of jobs moved off trader-mresnet-16, seed 0: progcov alpha 0.5 / 0, online economy, full BPTT with the cap) | 2026-09-29 23:22 (P3_DONE, sha256-verified sync of 26 files, deleted with disk, confirmed by gcloud instances and disks lists; 1 relaunch by p3ctl.sh / p4ctl.sh after spot preemptions) | 1.45 | 0.47 |
| 2026-09-29 21:41 | trader-mresnet-23 | us-east4-a, then us-east1-d | g2-standard-4 + L4 spot, max-run 4h, termination DELETE; p4ctl.sh recreates it from the synced state after a preemption (phase 3: re-evaluation only, on 200 fresh L3200 strings, of jobs moved off trader-mresnet-17, seed 1: progcov alpha 0.5 / 0, online economy, full BPTT with the cap) | 2026-09-29 23:48 (P3_DONE, sha256-verified sync of 23 files, deleted with disk, confirmed by gcloud instances and disks lists; 1 relaunch by p3ctl.sh / p4ctl.sh after spot preemptions) | 1.46 | 0.47 |
| 2026-09-29 21:41 | trader-mresnet-24 | us-east4-a | g2-standard-4 + L4 spot, max-run 4h, termination DELETE; p4ctl.sh recreates it from the synced state after a preemption (phase 3: re-evaluation only, on 200 fresh L3200 strings, of jobs moved off trader-mresnet-18, seed 2: progcov alpha 0.5 / 0, online economy, full BPTT with the cap) | 2026-09-29 23:32 (P3_DONE, sha256-verified sync of 23 files, deleted with disk, confirmed by gcloud instances and disks lists; 2 relaunches by p3ctl.sh / p4ctl.sh after spot preemptions) | 1.35 | 0.43 |
| 2026-09-29 21:41 | trader-mresnet-25 | us-east4-a, then us-central1-a, then us-east1-d | g2-standard-4 + L4 spot, max-run 4h, termination DELETE; p4ctl.sh recreates it from the synced state after a preemption (phase 3: re-evaluation only, on 200 fresh L3200 strings, of jobs moved off trader-mresnet-14, seed 1: the L2, cap and trade-cap regularizer arms) | 2026-09-30 00:47 (P3_DONE, sha256-verified sync of 19 files, deleted with disk, confirmed by gcloud instances and disks lists; 4 relaunches by p3ctl.sh / p4ctl.sh after spot preemptions) | 1.46 | 0.47 |

## P6 trader entry (trader-cat-p6-*, cap $8)
Rate ceiling: g2-standard-4 + 1x L4 SPOT + boot disk <= $0.71/h (on-demand list price; spot is lower). Watchdog: puzzles/P6/gpu.sh watchdog (5-min sync + sha256; deletes at est. spend >= $7, age, or idle 20 min).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 12:15 | trader-cat-p6-1 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 14:02 (spot preempted) | 1.78 | 1.26 |
| 2026-09-29 13:34 | trader-cat-p6-2 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 14:21 | 0.78 | 0.55 |
| 2026-09-29 14:12 | trader-cat-p6-3 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 14:21 | 0.15 | 0.11 |

## P10 masked cells (trader-cat-p10-*, cap $6)
Rate ceiling: g2-standard-4 + 1x L4 SPOT + boot disk <= $0.71/h (on-demand list price; spot is lower). Watchdog: puzzles/P10/gpu.sh watchdog (5-min sync + sha256; deletes at est. spend >= $5, age, or idle 20 min).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 12:35 | trader-cat-p10-1 | us-central1-c | g2-standard-4 + L4 spot, max-run 4h | 2026-09-29 14:21 | 1.77 | 1.26 |

## P5 private slots (trader-cat-p5-*, cap $8)
Rate ceiling: g2-standard-4 + 1x L4 SPOT + boot disk <= $0.40/h (on-demand list price; spot is lower). Watchdog: puzzles/P5/gpu.sh watchdog (10-min sync + sha256; deletes at est. spend >= $7, age, or idle 20 min).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 12:37 | trader-cat-p5-1 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 14:57 | 2.33 | 0.93 |

## P1 voc-halting (trader-cat-p1-*, cap $8)
Rate ceiling: g2-standard-4 + 1x L4 SPOT + boot disk <= $0.71/h (on-demand list price; spot is lower). Watchdog: puzzles/P1/gpu.sh watchdog (every 5 min; 15-min sync + sha256; deletes at est. spend >= $7, age, or idle 20 min).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 12:39 | trader-cat-p1-1 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 14:32 (spot preemption, auto-deleted with disk) | 1.88 | 1.33 |
| 2026-09-29 14:44 | trader-cat-p1-2 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 15:32 (spot preemption, auto-deleted with disk) | 0.81 | 0.57 |
| 2026-09-29 15:47 | trader-cat-p1-3 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 16:31 (spot preemption, auto-deleted with disk) | 0.73 | 0.52 |
| 2026-09-29 16:40 | trader-cat-p1-4 | us-east1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 16:50 (spot preemption, auto-deleted with disk) | 0.16 | 0.11 |
| 2026-09-29 16:52 | trader-cat-p1-5 | us-east1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 18:31 (spot preemption, auto-deleted with disk) | 1.66 | 1.18 |
| 2026-09-29 18:33 | trader-cat-p1-6 | us-east1-d | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 19:02 (spot preemption, auto-deleted with disk) | 0.48 | 0.34 |
| 2026-09-29 19:06 | trader-cat-p1-7 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 20:18 | 1.19 | 0.85 |
P1 total: 6.91 h, est. $4.90 at the on-demand ceiling (7 VMs; p1-1 to p1-6 were spot-preempted and resumed from synced or pulled checkpoints; p1-7 finished and was deleted by `gpu.sh down` after a sha256-verified final sync). Verified 20:19Z: no trader-cat-p1 instance or disk left (gcloud instances list / disks list, name filter ^trader-cat-p1-).

## P4 latent worth (trader-cat-p4-*, cap $8)
Rate ceiling: g2-standard-4 + 1x L4 SPOT + boot disk <= $0.71/h (on-demand list price; spot is lower). Watchdog: puzzles/P4/gpu.sh watchdog (5-min sync + sha256; deletes at est. spend >= $6, age, or idle 20 min).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 12:52 | trader-cat-p4-1 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 14:02 | 1.17 | 0.83 |
| 2026-09-29 14:21 | trader-cat-p4-2 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 14:49 | 0.47 | 0.33 |
P4 total: 1.64 h, est. $1.16 at the on-demand ceiling (p4-1 was preempted at 14:02Z after 100 synced files; p4-2 finished the 9 lost seed-4 runs). Verified 14:50Z: no trader-cat-p4 instance or disk left (gcloud instances/disks list).

## P9 futarchy on z (trader-cat-p9-*, cap $8)
Rate ceiling: g2-standard-4 + 1x L4 SPOT + boot disk <= $0.86/h (on-demand list price; spot is lower). Watchdog: puzzles/P9/gpu.sh watchdog (5-min sync + sha256; deletes at est. spend >= $7, age 3.95h, or idle 30 min).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 12:56 | trader-cat-p9-1 | us-central1-a | g2-standard-4 + L4 spot, max-run 4h | 2026-09-29 14:25 | 1.48 | 1.28 |
| 2026-09-29 14:30 | trader-cat-p9-2 | us-east1-c | g2-standard-4 + L4 spot, max-run 4h | 2026-09-29 15:34 (sha256-verified sync of 435 files, deleted with disk, no trader-cat-p9 VM or disk left per gcloud list) | 1.07 | 0.92 |
P9 total: 2.55 h, est. $2.20 at the on-demand ceiling (p9-1 was deleted mid-queue at 14:25Z by a bug in the P9 watchdog's busy check, after a verified sync; fixed, and the work resumed on p9-2). Watchdog stopped 15:36Z. Verified 15:35Z: no trader-cat-p9 instance or disk left (gcloud instances/disks list).

## P7 hindsight-posterior settlement (trader-cat-p7-*, cap $8)
CPU only, no GPU. Rate ceiling: c2d-highcpu-32 SPOT + boot disk <= $1.20/h (on-demand list price; spot is lower). Watchdog: puzzles/P7/vm.sh watchdog (5-min sync + sha256; deletes at est. spend >= $7, age, or idle 20 min).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 12:57 | trader-cat-p7-1 | us-central1-a | c2d-highcpu-32 spot (CPU only), max-run 6h | 2026-09-29 13:31 | 0.57 | 0.68 |
| 2026-09-29 13:33 | trader-cat-p7-2 | us-central1-a | c2d-highcpu-16 spot (CPU only), max-run 6h | 2026-09-29 13:50 | 0.27 | 0.33 |
| 2026-09-29 14:04 | trader-cat-p7-3 | us-central1-b | c2d-highcpu-8 spot (CPU only), max-run 6h | 2026-09-29 14:40 | 0.59 | 0.71 |

## P11 wealth cap (trader-cat-p11-*, cap $8)
Rate ceiling: g2-standard-4 + 1x L4 SPOT + boot disk <= $0.71/h (on-demand list price; spot is lower). Watchdog: puzzles/P11/gpu.sh watchdog (5-min sync + sha256; deletes at est. spend >= $8, age, or idle 20 min). From 29 Sep 14:10Z the cap check counts VM-hours at $0.45/h (the spec's spot estimate is $0.32/h), because the measured core eval cost (~38 min per model at 4 per L4) makes the full plan ~14 VM-h, over $8 only at the on-demand ceiling; the est $ column stays at the $0.71/h ceiling (an upper bound).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 13:02 | trader-cat-p11-1 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 13:33 | 0.53 | 0.37 |

## P0 puzzle baselines (trader-cat-p0-*, cap $8)
Rate ceiling: g2-standard-4 + 1x L4 SPOT + 100 GB boot disk <= $0.72/h (ceiling; spot is lower). Watchdog: puzzles/P0/p0_gpu.sh watchdog (every 5 min; sync on each finished job and every 15 min + sha256 -c; deletes every P0 VM at est. spend >= $7.5; age 5.2 h; idle 20 min; <= 2 preemption recreates per lane).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 13:11 | trader-cat-p0-b | us-central1-b | g2-standard-4 + L4 spot, max-run 5h | 2026-09-29 17:23 | 4.20 | 3.03 |
| 2026-09-29 13:12 | trader-cat-p0-a | us-east4-a | g2-standard-4 + L4 spot, max-run 5h | 2026-09-29 16:42 | 3.49 | 2.51 |
| 2026-09-29 13:30 | trader-cat-p0-a-r1 | us-central1-a | g2-standard-4 + L4 spot, max-run 5h | 2026-09-29 13:32 | 0.03 | 0.02 | (duplicate from a watchdog bug; deleted before use)
| 2026-09-29 16:45 | trader-cat-p0-a-r2 | us-central1-c | g2-standard-4 + L4 spot, max-run 2h | 2026-09-29 17:38 | 0.88 | 0.63 |
| | **P0 total** | | 4 VMs, none left (instances/disks list empty 17:38) | | 8.60 | **6.19** | (row a is billed to 16:42; its real preemption was 16:31, so about 0.17 h / $0.12 over)

## P8 intermediates by bits saved (trader-cat-p8-*, cap $8)
Rate ceiling: g2-standard-8 + 1x L4 SPOT + boot disk <= $0.90/h (on-demand list price; spot is lower). Watchdog: puzzles/P8/gpu.sh watchdog (5-min sync + sha256; deletes at est. spend >= $8, age, or idle 20 min).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 13:14 | trader-cat-p8-1 | us-central1-a | g2-standard-8 + L4 spot, max-run 6h | 2026-09-29 15:28 | 2.23 | 2.01 |

## P2 joint-structure markets (trader-cat-p2-*, cap $8)
Rate ceiling: g2-standard-4 + 1x L4 SPOT + 50 GB boot disk <= $0.75/h (on-demand list price; spot is lower). Watchdog: puzzles/P2/gpu.sh watchdog (5-min checks, 15-min sync + sha256; deletes at est. spend >= $6, age 4.9 h, or idle 20 min).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 13:30 | trader-cat-p2-1 | us-central1-a | g2-standard-4 + L4 spot, max-run 5h | 2026-09-29 13:49 | 0.32 | 0.24 |
| 2026-09-29 13:36 | trader-cat-p11-s0-1336 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 16:32 | 2.93 | 2.08 |
| 2026-09-29 13:37 | trader-cat-p11-s1-1336 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 14:47 | 1.17 | 0.83 |
| 2026-09-29 14:04 | trader-cat-p2-2 | us-central1-b | g2-standard-4 + L4 spot, max-run 5h | 2026-09-29 17:17 | 3.23 | 2.42 |
| 2026-09-29 14:50 | trader-cat-p2-3 | us-central1-a | g2-standard-4 + L4 spot, max-run 5h | 2026-09-29 15:03 | 0.22 | 0.17 |
| 2026-09-29 15:15 | trader-cat-p2-4 | us-central1-a | g2-standard-4 + L4 spot, max-run 5h | 2026-09-29 15:36 | 0.36 | 0.27 |
| 2026-09-29 15:43 | trader-cat-p2-5 | us-west1-b | g2-standard-4 + L4 spot, max-run 5h | 2026-09-29 17:33 | 1.83 | 1.38 |

## P3 latent-market factor selection (trader-cat-p3-*, cap $8)
CPU only, no GPU. Rate ceiling: c2d-highcpu-32 SPOT + boot disk <= $1.20/h (on-demand list price; spot is lower). Watchdog: puzzles/P3/vm.sh watchdog (5-min sync + sha256; deletes at est. spend >= $6, age 4.9 h, or idle 20 min).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 13:37 | trader-cat-p3-1 | us-central1-c | c2d-highcpu-32 spot (CPU only), max-run 5h | 2026-09-29 14:10 (spot preemption, auto-deleted) | 0.55 | 0.66 |
| 2026-09-29 14:18 | trader-cat-p3-2 | us-central1-f | c2d-highcpu-32 spot (CPU only), max-run 5h | 2026-09-29 14:42 | 0.40 | 0.48 |

## P12 flow-BP mazes (trader-cat-p12-*, cap $8)
Rate ceiling: g2-standard-4 + 1x L4 SPOT + 50 GB boot disk <= $0.72/h (on-demand list price; spot is lower). Watchdog: puzzles/P12/gpu.sh watchdog (5-min checks, 15-min sync + sha256; deletes at est. spend >= $6, age 4.9 h, or idle 20 min).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 13:55 | trader-cat-p12-1 | us-central1-a | g2-standard-4 + L4 spot, max-run 5h | 2026-09-29 14:01 | 0.10 | 0.07 |
| 2026-09-29 14:13 | trader-cat-p12-2 | us-central1-a | g2-standard-4 + L4 spot, max-run 5h | 2026-09-29 16:02 (spot preemption, auto-deleted) | 1.80 | 1.30 |
| 2026-09-29 14:49 | trader-cat-p11-s1-1448 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 15:01 | 0.21 | 0.15 |
| 2026-09-29 15:04 | trader-cat-p11-s1-1502 | us-west1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 17:01 | 1.95 | 1.39 |
| 2026-09-29 15:56 | trader-cat-p11-s2-1556 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 18:01 | 2.08 | 1.48 |
| 2026-09-29 16:13 | trader-cat-p12-3 | us-central1-c | g2-standard-4 + L4 spot, max-run 5h | 2026-09-29 20:53 | 4.66 | 3.35 |
| 2026-09-29 16:34 | trader-cat-p11-s0-1633 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 19:46 | 3.20 | 2.27 |
| 2026-09-29 17:02 | trader-cat-p11-s1-1702 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 17:31 | 0.48 | 0.34 |
| 2026-09-29 17:34 | trader-cat-p11-s1-1732 | us-west1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 19:23 | 1.82 | 1.29 |
| 2026-09-29 18:06 | trader-cat-p11-s2-1806 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 18:52 | 0.76 | 0.54 |
| 2026-09-29 20:55 | trader-cat-p12-4 | us-central1-a | g2-standard-4 + L4 spot, max-run 5h | 2026-09-29 21:02 (spot preemption, auto-deleted) | 0.12 | 0.09 |
| 2026-09-29 21:04 | trader-cat-p12-5 | us-central1-c | g2-standard-4 + L4 spot, max-run 5h | 2026-09-29 21:10 (spot preemption, auto-deleted) | 0.10 | 0.07 |

P11 total (closed 2026-09-29 19:46Z): 10 VMs, 15.13 VM-h (7 spot preemptions). Watch estimate $6.80 at $0.45/h (the rate the $8 cap was checked at from 14:10Z); on-demand list ceiling $10.74 at $0.71/h, above $8. Billed spot price not readable (Cloud Billing API disabled in GCP_PROJECT). Teardown verified 19:51Z: instances list and disks list filtered on ^trader-cat-p11 both empty.

P12 total (closed 2026-09-29 21:35Z): 5 VMs, 6.78 VM-h (3 spot preemptions), estimated $4.88 at the $0.72/h on-demand ceiling (spot is billed lower; cap $8). Teardown verified 21:35Z: instances list and disks list filtered on name~trader-cat-p12 both empty; watchdog stopped.

## P14 value-priced event publishing (trader-b3-p14-*, cap $15)
CPU only, no GPU. Rate ceiling: c2d-highcpu-32 SPOT + boot disk <= $1.20/h (on-demand list price; spot is lower). Watchdog: puzzles/P14/vm.sh watchdog (5-min sync + sha256; deletes at est. spend >= $15, age, or idle 20 min).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 20:13 | trader-b3-p14-1 | us-central1-a | c2d-highcpu-32 spot (CPU only), max-run 6h | 2026-09-29 20:52 | 0.65 | 0.78 |
| 2026-09-29 20:50 | trader-b3-p14-2 | us-central1-c | c2d-highcpu-32 spot (CPU only), max-run 6h | 2026-09-29 23:47 | 2.94 | 3.53 |
| 2026-09-29 20:56 | trader-b3-p14-1-r1 | us-central1-c | c2d-highcpu-32 spot (CPU only), max-run 6h | 2026-09-29 23:45 | 2.81 | 3.38 |

## P16 amortized valuation (trader-b3-p16-*, cap $15)
Rate ceiling: g2-standard-4 + 1x L4 SPOT + 50 GB boot disk <= $0.71/h (on-demand list price; spot is lower). Watchdog: puzzles/P16/gpu.sh watchdog (5-min checks, 15-min sync + sha256; sums all p16 VM-hours; deletes at est. spend >= $13, age 5.9 h, or idle 20 min; at most 2 VMs).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 20:27 | trader-b3-p16-1 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 21:34 | 1.12 | 0.80 |
| 2026-09-29 21:52 | trader-b3-p16-2 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 22:09 | 0.28 | 0.20 |
| 2026-09-29 22:11 | trader-b3-p16-3 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 22:18 | 0.12 | 0.08 |
| 2026-09-29 22:19 | trader-b3-p16-4 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 02:11 | 3.87 | 2.74 |
| 2026-09-29 23:25 | trader-b3-p16-5 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 23:30 | 0.07 | 0.05 |
| 2026-09-29 23:31 | trader-b3-p16-6 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 01:32 | 2.01 | 1.42 |
| 2026-09-30 01:32 | trader-b3-p16-7 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 02:32 | 1.00 | 0.71 |
| 2026-09-30 02:13 | trader-b3-p16-8 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 02:21 | 0.14 | 0.10 |
| 2026-09-30 02:23 | trader-b3-p16-9 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 02:33 | 0.17 | 0.12 |
| 2026-09-30 02:34 | trader-b3-p16-10 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 04:03 | 1.50 | 1.06 |
| 2026-09-30 02:39 | trader-b3-p16-11 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 03:33 | 0.90 | 0.64 |
| 2026-09-30 03:33 | trader-b3-p16-12 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 05:32 | 1.97 | 1.40 |
| 2026-09-30 05:39 | trader-b3-p16-13 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 06:21 | 0.70 | 0.49 |

## P15 tied-CA local EM (trader-b3-p15-*, cap $20)
Rate ceiling: g2-standard-8 + 1x L4 SPOT + 50 GB boot disk <= $0.86/h (on-demand list price; spot is lower). Watchdog: puzzles/P15/gpu.sh watchdog (5-min checks, 15-min sync + sha256; sums all p15 VM-hours; deletes at est. spend >= $17, age 5.9 h, or idle 20 min; at most 2 VMs).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 20:42 | trader-b3-p15-1 | us-central1-c | g2-standard-8 + L4 spot, max-run 6h | 2026-09-29 20:55 | 0.21 | 0.18 |
| 2026-09-29 20:57 | trader-b3-p15-2 | us-east1-c | g2-standard-8 + L4 spot, max-run 6h | 2026-09-30 00:31 | 3.57 | 3.07 |
| 2026-09-29 21:55 | trader-b3-p15-3 | us-central1-a | g2-standard-8 + L4 spot, max-run 6h | 2026-09-29 22:35 | 0.67 | 0.57 |
| 2026-09-29 21:56 | trader-b3-p15-4 | us-east1-b | g2-standard-8 + L4 spot, max-run 6h | 2026-09-30 03:06 | 5.18 | 4.45 |
| 2026-09-29 22:36 | trader-b3-p15-5 | us-central1-c | g2-standard-8 + L4 spot, max-run 6h | 2026-09-29 23:02 | 0.43 | 0.37 |
| 2026-09-29 23:09 | trader-b3-p15-6 | us-central1-b | g2-standard-8 + L4 spot, max-run 6h | 2026-09-30 00:47 | 1.63 | 1.40 |
| 2026-09-30 00:34 | trader-b3-p15-7 | europe-west1-b | g2-standard-8 + L4 spot, max-run 6h | 2026-09-30 01:35 | 1.03 | 0.89 |
| 2026-09-30 00:48 | trader-b3-p15-8 | us-west1-a | g2-standard-8 + L4 spot, max-run 6h | 2026-09-30 03:25 | 2.61 | 2.25 |
| 2026-09-30 01:36 | trader-b3-p15-9 | us-central1-b | g2-standard-8 + L4 spot, max-run 6h | 2026-09-30 02:02 | 0.43 | 0.37 |
| 2026-09-30 02:07 | trader-b3-p15-10 | us-central1-a | g2-standard-8 + L4 spot, max-run 6h | 2026-09-30 02:37 | 0.49 | 0.42 |
| 2026-09-30 02:38 | trader-b3-p15-11 | us-central1-b | g2-standard-8 + L4 spot, max-run 6h | 2026-09-30 03:35 | 0.96 | 0.82 |
| 2026-09-30 03:36 | trader-b3-p15-12 | us-central1-c | g2-standard-8 + L4 spot, max-run 6h | 2026-09-30 03:46 | 0.16 | 0.14 |
| 2026-09-30 03:46 | trader-b3-p15-13 | us-central1-c | g2-standard-8 + L4 spot, max-run 6h | 2026-09-30 03:56 | 0.17 | 0.14 |
| 2026-09-30 03:57 | trader-b3-p15-14 | us-east1-b | g2-standard-8 + L4 spot, max-run 6h | 2026-09-30 05:46 | 1.82 | 1.57 |

P15 final (checked 05:47Z: `./gpu.sh check` and gcloud instances/disks list both empty): 14 VMs, ledger 19.36 VM-h ($16.64 at the $0.86/h ceiling, $10.07 at the $0.52/h spot price). GCP operations (insert to preempt/delete; re-run `python3 puzzles/P15/gpu/actual_spend.py`) show 17.81 VM-h = $9.26 at spot, $15.31 at the ceiling; the ledger overcounts because it runs each gone VM until the watchdog noticed it.

## P13 market ResNet mazes (trader-b3-p13-*, cap $40)
Rate estimate $0.35/h (g2-standard-4 + 1x L4 spot + 100 GB disk (image minimum); spend = (last seen - created + 5 min) x rate). Watchdog: puzzles/P13/gpu/watchdog.sh (5-min loop; deletes all at est. spend >= $40, any VM at 5.8 h or 20 min idle). Controller: puzzles/P13/gpu/ctl.sh (3-min sha256-verified sync, resume on -rN after preemption). Est. total $37.52.
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 20:49 | trader-b3-p13-1 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 22:03 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.26 | 0.44 |
| 2026-09-29 21:55 | trader-b3-p13-5 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 22:32 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.68 | 0.24 |
| 2026-09-29 21:55 | trader-b3-p13-2 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 22:36 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.68 | 0.24 |
| 2026-09-29 21:55 | trader-b3-p13-6 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 22:32 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.68 | 0.24 |
| 2026-09-29 21:55 | trader-b3-p13-3 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 23:03 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.19 | 0.42 |
| 2026-09-29 21:55 | trader-b3-p13-4 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 23:03 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.19 | 0.42 |
| 2026-09-29 21:55 | trader-b3-p13-8 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 22:32 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.68 | 0.24 |
| 2026-09-29 21:55 | trader-b3-p13-7 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 22:32 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.68 | 0.24 |
| 2026-09-29 22:17 | trader-b3-p13-10 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:33 (runner done (queue ended, no pending jobs); sha256-verified sync of 24 files) | 4.29 | 1.50 |
| 2026-09-29 22:17 | trader-b3-p13-11 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:38 (runner done (queue ended, no pending jobs); sha256-verified sync of 24 files) | 4.37 | 1.53 |
| 2026-09-29 22:17 | trader-b3-p13-9 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:32 (runner done (queue ended, no pending jobs); sha256-verified sync of 24 files) | 4.29 | 1.50 |
| 2026-09-29 22:37 | trader-b3-p13-5-r1 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 23:03 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.48 | 0.17 |
| 2026-09-29 22:37 | trader-b3-p13-6-r1 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 23:04 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.48 | 0.17 |
| 2026-09-29 22:37 | trader-b3-p13-8-r1 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 23:04 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.48 | 0.17 |
| 2026-09-29 22:37 | trader-b3-p13-7-r1 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 23:04 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.48 | 0.17 |
| 2026-09-29 23:08 | trader-b3-p13-4-r1 | us-east1-d | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 23:32 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.48 | 0.17 |
| 2026-09-29 23:09 | trader-b3-p13-2-r3 | us-east1-d | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 23:17 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.12 | 0.04 |
| 2026-09-29 23:10 | trader-b3-p13-8-r2 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 23:33 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.45 | 0.16 |
| 2026-09-29 23:10 | trader-b3-p13-6-r2 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 00:04 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.96 | 0.33 |
| 2026-09-29 23:10 | trader-b3-p13-7-r2 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-29 23:58 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.70 | 0.25 |
| 2026-09-29 23:21 | trader-b3-p13-5-r3 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 01:37 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 2.22 | 0.78 |
| 2026-09-29 23:22 | trader-b3-p13-3-r2 | us-east1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 00:31 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.17 | 0.41 |
| 2026-09-29 23:23 | trader-b3-p13-2-r4 | us-east1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 00:34 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.17 | 0.41 |
| 2026-09-29 23:29 | trader-b3-p13-12 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 00:05 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.56 | 0.19 |
| 2026-09-29 23:29 | trader-b3-p13-13 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 01:06 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.57 | 0.55 |
| 2026-09-29 23:41 | trader-b3-p13-4-r2 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 01:20 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.62 | 0.57 |
| 2026-09-29 23:42 | trader-b3-p13-8-r3 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:03 (runner done (queue ended, no pending jobs); sha256-verified sync of 38 files) | 2.36 | 0.83 |
| 2026-09-30 00:11 | trader-b3-p13-6-r3 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 01:37 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.38 | 0.48 |
| 2026-09-30 00:11 | trader-b3-p13-12-r1 | us-east1-d | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 01:33 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.38 | 0.48 |
| 2026-09-30 00:14 | trader-b3-p13-7-r4 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:13 (runner done (queue ended, no pending jobs); sha256-verified sync of 38 files) | 2.00 | 0.70 |
| 2026-09-30 00:35 | trader-b3-p13-3-r3 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 01:37 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.98 | 0.34 |
| 2026-09-30 00:37 | trader-b3-p13-14 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 00:50 (runner done (queue ended, no pending jobs); sha256-verified sync of 27 files) | 0.26 | 0.09 |
| 2026-09-30 00:41 | trader-b3-p13-2-r5 | us-west1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 01:04 (runner done (queue ended, no pending jobs); sha256-verified sync of 21 files) | 0.45 | 0.16 |
| 2026-09-30 01:13 | trader-b3-p13-13-r1 | us-east1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 01:33 (runner done (queue ended, no pending jobs); sha256-verified sync of 23 files) | 0.35 | 0.12 |
| 2026-09-30 01:26 | trader-b3-p13-4-r3 | us-east1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:12 (runner done (queue ended, no pending jobs); sha256-verified sync of 40 files) | 0.81 | 0.28 |
| 2026-09-30 01:27 | trader-b3-p13-15 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:34 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.13 | 0.39 |
| 2026-09-30 01:38 | trader-b3-p13-12-r2 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:03 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.44 | 0.15 |
| 2026-09-30 01:42 | trader-b3-p13-3-r4 | us-east1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:18 (runner done (queue ended, no pending jobs); sha256-verified sync of 40 files) | 0.63 | 0.22 |
| 2026-09-30 01:42 | trader-b3-p13-5-r4 | us-east1-d | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:01 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.37 | 0.13 |
| 2026-09-30 01:43 | trader-b3-p13-6-r4 | us-east1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:23 (runner done (queue ended, no pending jobs); sha256-verified sync of 40 files) | 0.70 | 0.24 |
| 2026-09-30 02:09 | trader-b3-p13-5-r5 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:32 (runner done (queue ended, no pending jobs); sha256-verified sync of 40 files) | 0.43 | 0.15 |
| 2026-09-30 02:19 | trader-b3-p13-12-r4 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:39 (runner done (queue ended, no pending jobs); sha256-verified sync of 24 files) | 2.36 | 0.83 |
| 2026-09-30 02:34 | trader-b3-p13-20 | europe-west4-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:29 (runner done (queue ended, no pending jobs); sha256-verified sync of 24 files) | 0.08 | 0.03 |
| 2026-09-30 02:35 | trader-b3-p13-16 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 07:47 (runner done (queue ended, no pending jobs); sha256-verified sync of 37 files) | 5.28 | 1.85 |
| 2026-09-30 02:35 | trader-b3-p13-22 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:31 (runner done (queue ended, no pending jobs); sha256-verified sync of 24 files) | 3.99 | 1.40 |
| 2026-09-30 02:35 | trader-b3-p13-15-r1 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:52 (runner done (queue ended, no pending jobs); sha256-verified sync of 21 files) | 0.34 | 0.12 |
| 2026-09-30 02:35 | trader-b3-p13-17 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:24 (runner done (queue ended, no pending jobs); sha256-verified sync of 24 files) | 3.82 | 1.34 |
| 2026-09-30 02:35 | trader-b3-p13-18 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:25 (runner done (queue ended, no pending jobs); sha256-verified sync of 24 files) | 3.91 | 1.37 |
| 2026-09-30 02:35 | trader-b3-p13-19 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:27 (runner done (queue ended, no pending jobs); sha256-verified sync of 24 files) | 3.91 | 1.37 |
| 2026-09-30 02:35 | trader-b3-p13-21 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:30 (runner done (queue ended, no pending jobs); sha256-verified sync of 24 files) | 3.99 | 1.40 |
| 2026-09-30 02:35 | trader-b3-p13-23 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:33 (runner done (queue ended, no pending jobs); sha256-verified sync of 24 files) | 3.99 | 1.40 |
| 2026-09-30 02:35 | trader-b3-p13-24 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:34 (runner done (queue ended, no pending jobs); sha256-verified sync of 24 files) | 3.99 | 1.40 |
| 2026-09-30 02:35 | trader-b3-p13-20 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:29 (runner done (queue ended, no pending jobs); sha256-verified sync of 24 files) | 3.91 | 1.37 |
| 2026-09-30 06:11 | trader-b3-p13-25 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:34 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.39 | 0.14 |
| 2026-09-30 06:39 | trader-b3-p13-30 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 07:04 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.44 | 0.15 |
| 2026-09-30 06:39 | trader-b3-p13-29 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 07:04 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.43 | 0.15 |
| 2026-09-30 06:39 | trader-b3-p13-28 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 07:04 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.43 | 0.15 |
| 2026-09-30 06:39 | trader-b3-p13-27 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 07:04 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.42 | 0.15 |
| 2026-09-30 06:40 | trader-b3-p13-33 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 07:05 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.42 | 0.15 |
| 2026-09-30 06:40 | trader-b3-p13-26 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 07:03 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.42 | 0.15 |
| 2026-09-30 06:40 | trader-b3-p13-32 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 07:04 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.42 | 0.15 |
| 2026-09-30 06:40 | trader-b3-p13-25-r1 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 07:03 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.42 | 0.15 |
| 2026-09-30 06:40 | trader-b3-p13-31 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 07:04 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.41 | 0.14 |
| 2026-09-30 07:09 | trader-b3-p13-25-r2 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:05 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.88 | 0.31 |
| 2026-09-30 07:09 | trader-b3-p13-28-r1 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:05 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.88 | 0.31 |
| 2026-09-30 07:09 | trader-b3-p13-30-r1 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:05 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.88 | 0.31 |
| 2026-09-30 07:09 | trader-b3-p13-27-r1 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:05 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.88 | 0.31 |
| 2026-09-30 07:09 | trader-b3-p13-26-r1 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:05 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.88 | 0.31 |
| 2026-09-30 07:09 | trader-b3-p13-32-r1 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:05 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.88 | 0.31 |
| 2026-09-30 07:09 | trader-b3-p13-33-r1 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:02 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.88 | 0.31 |
| 2026-09-30 07:09 | trader-b3-p13-29-r1 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:05 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.88 | 0.31 |
| 2026-09-30 07:09 | trader-b3-p13-31-r1 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:05 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.88 | 0.31 |
| 2026-09-30 08:06 | trader-b3-p13-33-r2 | us-east1-d | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:55 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 0.86 | 0.30 |
| 2026-09-30 08:10 | trader-b3-p13-26-r2 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:32 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.36 | 0.13 |
| 2026-09-30 08:10 | trader-b3-p13-27-r2 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:32 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.36 | 0.13 |
| 2026-09-30 08:11 | trader-b3-p13-28-r2 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:52 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 0.70 | 0.24 |
| 2026-09-30 08:11 | trader-b3-p13-30-r2 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:12 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.10 | 0.04 |
| 2026-09-30 08:11 | trader-b3-p13-31-r2 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 09:04 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 0.95 | 0.33 |
| 2026-09-30 08:11 | trader-b3-p13-25-r3 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 10:29 (runner done (queue ended, no pending jobs); sha256-verified sync of 14 files) | 2.32 | 0.81 |
| 2026-09-30 08:11 | trader-b3-p13-29-r2 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:17 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.10 | 0.03 |
| 2026-09-30 08:12 | trader-b3-p13-32-r2 | europe-west4-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 09:05 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 0.08 | 0.03 |
| 2026-09-30 08:12 | trader-b3-p13-32-r2 | europe-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 09:05 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 0.93 | 0.33 |
| 2026-09-30 08:22 | trader-b3-p13-29-r3 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 09:02 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 0.69 | 0.24 |
| 2026-09-30 08:22 | trader-b3-p13-30-r3 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 09:03 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 0.69 | 0.24 |
| 2026-09-30 08:38 | trader-b3-p13-26-r3 | europe-west4-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 08:59 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 0.42 | 0.15 |
| 2026-09-30 08:39 | trader-b3-p13-27-r3 | us-east4-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 09:01 (runner done (queue ended, no pending jobs); sha256-verified sync of 16 files) | 0.40 | 0.14 |

## DT baseline, official deep-thinking reproduction (trader-dt-*, cap $25)
Rate estimate: g2-standard-4 + 1x L4 SPOT + 100 GB pd-balanced ~= $0.40/h (conservative spot). Watchdog: dt-baseline/dt_gpu.sh watchdog (every 5 min; sha256-verified sync every 15 min and on each finished step; deletes every trader-dt VM at est. spend >= $22; lanes roll over to a new VM before the 6h max-run; preempted lanes resume from per-epoch checkpoints on a new VM). Log: dt-baseline/gpu/LEDGER.md.
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 21:33 | trader-dt-p0 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 21:39 (vanished: spot preemption or max-run, auto-deleted with disk) | 0.09 | 0.04 |
| 2026-09-29 21:41 | trader-dt-p0 | us-east1-c | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 00:37 (vanished: spot preemption or max-run, auto-deleted with disk) | 2.93 | 1.17 |
| 2026-09-29 21:41 | trader-dt-m1 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 22:04 (vanished: spot preemption or max-run, auto-deleted with disk) | 0.39 | 0.16 |
| 2026-09-29 21:41 | trader-dt-m0 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 22:04 (vanished: spot preemption or max-run, auto-deleted with disk) | 0.38 | 0.15 |
| 2026-09-29 21:42 | trader-dt-m2 | us-east4-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 22:04 (vanished: spot preemption or max-run, auto-deleted with disk) | 0.38 | 0.15 |
| 2026-09-29 21:42 | trader-dt-p2 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 01:18 (plan p2 finished; deleted with disk, confirmed by gcloud list) | 3.61 | 1.44 |
| 2026-09-29 21:43 | trader-dt-p1 | us-east4-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 22:02 (vanished: spot preemption or max-run, auto-deleted with disk) | 0.30 | 0.12 |
| 2026-09-29 22:12 | trader-dt-p1 | us-east1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 03:53 (rollover before max-run; deleted with disk, confirmed by gcloud list) | 5.69 | 2.28 |
| 2026-09-29 22:12 | trader-dt-m2 | us-east1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 03:53 (rollover before max-run; deleted with disk, confirmed by gcloud list) | 5.69 | 2.27 |
| 2026-09-29 22:28 | trader-dt-m1-r1 | us-east1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 04:08 (rollover before max-run; deleted with disk, confirmed by gcloud list) | 5.67 | 2.27 |
| 2026-09-29 22:28 | trader-dt-m0-r1 | us-east1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 23:51 (vanished: spot preemption or max-run, auto-deleted with disk) | 1.38 | 0.55 |
| 2026-09-29 23:51 | trader-dt-m0-r2 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 23:58 (plan m0 finished; deleted with disk, confirmed by gcloud list) | 0.11 | 0.05 |
| 2026-09-30 00:07 | trader-dt-m0-r3 | us-east1-d | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 01:08 (vanished: spot preemption or max-run, auto-deleted with disk) | 1.01 | 0.41 |
| 2026-09-30 00:37 | trader-dt-p0-r1 | us-east1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 03:49 (plan p0 finished; deleted with disk, confirmed by gcloud list) | 3.19 | 1.28 |
| 2026-09-30 01:09 | trader-dt-m0-r4 | us-east1-c | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 06:51 (rollover before max-run; deleted with disk, confirmed by gcloud list) | 5.70 | 2.28 |
| 2026-09-30 03:53 | trader-dt-m2-r1 | us-east1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 07:38 (plan m2 finished; deleted with disk, confirmed by gcloud list) | 3.74 | 1.50 |
| 2026-09-30 03:53 | trader-dt-p1-r1 | us-east1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 05:53 (plan p1 finished; deleted with disk, confirmed by gcloud list) | 1.99 | 0.80 |
| 2026-09-30 04:08 | trader-dt-m1-r2 | us-east1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 07:42 (plan m1 finished; deleted with disk, confirmed by gcloud list) | 3.56 | 1.43 |
| 2026-09-30 06:52 | trader-dt-m0-r5 | us-east1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 08:39 (plan m0 finished; deleted with disk, confirmed by gcloud list) | 1.79 | 0.72 |

## P17 breadth-depth market (trader-bxd-p17-*, cap $30)
Rate ceiling: g2-standard-4 + 1x L4 SPOT + 50 GB boot disk <= $0.71/h (on-demand list price; spot is lower). Watchdog: puzzles/P17-breadth-depth/gpu.sh watchdog (5-min checks, 15-min sync + sha256; sums all p17 VM-hours; deletes at est. spend >= $27, age 5.9 h, or idle 20 min; at most 4 VMs; a failed gate deletes all).
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 22:25 | trader-bxd-p17-1 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 23:18 | 0.89 | 0.63 |
| 2026-09-29 23:48 | trader-bxd-p17-2 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h | 2026-09-29 23:59 | 0.18 | 0.13 |
| 2026-09-29 23:59 | trader-bxd-p17-3 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 00:36 | 0.62 | 0.44 |
| 2026-09-30 00:38 | trader-bxd-p17-4 | us-east1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 03:31 | 2.89 | 2.05 |
| 2026-09-30 00:59 | trader-bxd-p17-5 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 01:14 | 0.24 | 0.17 |
| 2026-09-30 01:02 | trader-bxd-p17-6 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 01:14 | 0.19 | 0.14 |
| 2026-09-30 01:05 | trader-bxd-p17-7 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 01:14 | 0.14 | 0.10 |
| 2026-09-30 01:16 | trader-bxd-p17-8 | us-west4-c | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 02:31 | 1.25 | 0.89 |
| 2026-09-30 01:23 | trader-bxd-p17-9 | us-east1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 04:29 | 3.10 | 2.20 |
| 2026-09-30 01:28 | trader-bxd-p17-10 | us-east1-d | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 02:02 | 0.57 | 0.40 |
| 2026-09-30 02:03 | trader-bxd-p17-11 | us-east1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 05:04 | 3.02 | 2.14 |
| 2026-09-30 02:32 | trader-bxd-p17-12 | us-east1-b | g2-standard-4 + L4 spot, max-run 6h | 2026-09-30 04:36 | 2.06 | 1.46 |

## Batch-2 follow-ups: CostNet+BB seeds, P10 init check, P8 N=4096 (trader-fu-*, cap $20)
CPU only (c2d-highcpu spot, termination DELETE, max-run 6h). Rate ceiling = on-demand list price incl. boot disk (c2d-highcpu-56 $2.10/h, -32 $1.20/h, -16 $0.60/h). Watchdog: puzzles/followups-batch2/vm.sh all watchdog (5-min sync + sha256, auto-resume after preemption, deletes at est. total >= $19, item cap, age 5.9 h or idle 20 min). Logs: followups-batch2/item*/vm/LEDGER.md.
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-29 22:41 | trader-fu-p8-1 | us-central1-a | c2d-highcpu-32 spot (CPU only), max-run 6h | 2026-09-29 23:10 | 0.47 | 0.57 |
| 2026-09-29 22:57 | trader-fu-p10-1 | us-central1-a | c2d-highcpu-32 spot (CPU only), max-run 6h | 2026-09-30 01:27 | 2.51 | 3.02 |
| 2026-09-29 23:13 | trader-fu-p8-1-r1 | us-central1-c | c2d-highcpu-32 spot (CPU only), max-run 6h | 2026-09-30 01:50 | 2.62 | 3.14 |
| 2026-09-29 23:15 | trader-fu-c1-1 | us-central1-c | c2d-highcpu-56 spot (CPU only), max-run 6h | 2026-09-29 23:34 | 0.33 | 0.68 (corrected 23:50Z: booked at the -32 rate $1.20/h; -56 ceiling is $2.10/h) |

## P19 fast-reader pricing (trader-p19-*, cap $5)
CPU only (c2d-highcpu; trader-p19-1 spot, preempted; trader-p19-2 on-demand at the same ceiling; termination DELETE). Rate ceiling = on-demand list price incl. boot disk (c2d-highcpu-32 $1.20/h). Watchdog: puzzles/P19-fast-reader/vm.sh watchdog (5-min sync + sha256, deletes at est. spend >= $4.50, age 3.9 h or idle 20 min). Log: P19-fast-reader/vm/LEDGER.md.
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-30 01:20 | trader-p19-1 | us-central1-c | c2d-highcpu-32 spot (CPU only), max-run 4h | 2026-09-30 02:03 | 0.71 | 0.85 |
| 2026-09-30 02:08 | trader-p19-2 | us-east1-b | c2d-highcpu-32 on-demand (CPU only), max-run 3h | 2026-09-30 03:49 | 1.67 | 2.00 |

## P18 detached mazes (trader-dds-*, cap $40)
Rate ceiling $0.71/h (g2-standard-4 + 1x L4 at on-demand list price + 100 GB disk; spot is lower; spend = (last seen - created + 5 min) x rate). Watchdog: puzzles/P18-detached-mazes/gpu/watchdog.sh (5-min loop; deletes all at est. spend >= $40, any VM at 5.8 h or 20 min idle). Controller: puzzles/P18-detached-mazes/gpu/ctl.sh (3-min sha256-verified sync, resume on -rN after preemption). Est. total $36.40.
| created (UTC) | VM | zone | shape | deleted (UTC) | hours | est $ |
|---|---|---|---|---|---|---|
| 2026-09-30 01:25 | trader-dds-1 | us-east1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 01:53 (runner done (queue ended, no pending jobs); sha256-verified sync of 25 files) | 0.48 | 0.34 |
| 2026-09-30 01:56 | trader-dds-11 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 03:55 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 1.99 | 1.41 |
| 2026-09-30 01:56 | trader-dds-12 | us-east4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 03:03 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.14 | 0.81 |
| 2026-09-30 01:57 | trader-dds-10 | us-east4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:58 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.05 | 0.74 |
| 2026-09-30 01:57 | trader-dds-9 | us-east4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:00 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.13 | 0.09 |
| 2026-09-30 01:57 | trader-dds-5 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:13 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.28 | 0.20 |
| 2026-09-30 01:57 | trader-dds-8 | us-east4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 03:03 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.12 | 0.80 |
| 2026-09-30 01:57 | trader-dds-6 | us-east4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:08 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.19 | 0.13 |
| 2026-09-30 01:57 | trader-dds-7 | us-east4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:00 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.12 | 0.08 |
| 2026-09-30 01:58 | trader-dds-2 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 01:59 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.10 | 0.07 |
| 2026-09-30 01:58 | trader-dds-4 | us-east4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:32 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.60 | 0.43 |
| 2026-09-30 01:58 | trader-dds-13 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:00 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.10 | 0.07 |
| 2026-09-30 01:58 | trader-dds-3 | us-east4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 03:02 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.11 | 0.79 |
| 2026-09-30 02:06 | trader-dds-9-r1 | us-west1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:10 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 2.15 | 1.53 |
| 2026-09-30 02:06 | trader-dds-7-r1 | us-west1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:09 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 2.06 | 1.46 |
| 2026-09-30 02:09 | trader-dds-2-r1 | us-east1-d | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:08 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.08 | 0.06 |
| 2026-09-30 02:09 | trader-dds-2-r1 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:08 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 2.02 | 1.43 |
| 2026-09-30 02:11 | trader-dds-13-r1 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:19 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.13 | 0.09 |
| 2026-09-30 02:14 | trader-dds-6-r1 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:18 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.09 | 0.06 |
| 2026-09-30 02:19 | trader-dds-5-r1 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 02:23 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.09 | 0.06 |
| 2026-09-30 02:24 | trader-dds-13-r2 | us-east4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 03:03 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.68 | 0.48 |
| 2026-09-30 02:24 | trader-dds-6-r2 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 03:10 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.08 | 0.06 |
| 2026-09-30 02:24 | trader-dds-6-r2 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 03:10 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.76 | 0.54 |
| 2026-09-30 02:27 | trader-dds-5-r2 | us-west1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:27 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 2.05 | 1.46 |
| 2026-09-30 02:38 | trader-dds-4-r1 | us-east1-d | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 03:02 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.44 | 0.31 |
| 2026-09-30 03:04 | trader-dds-10-r1 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 03:07 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.13 | 0.10 |
| 2026-09-30 03:07 | trader-dds-4-r2 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:26 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 1.39 | 0.99 |
| 2026-09-30 03:07 | trader-dds-3-r1 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:08 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 1.05 | 0.75 |
| 2026-09-30 03:08 | trader-dds-13-r3 | us-west4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:36 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 1.54 | 1.09 |
| 2026-09-30 03:08 | trader-dds-12-r1 | us-west4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:11 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 1.12 | 0.79 |
| 2026-09-30 03:09 | trader-dds-8-r1 | us-west4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:10 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 1.02 | 0.73 |
| 2026-09-30 03:11 | trader-dds-10-r2 | us-east4-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:20 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 1.15 | 0.82 |
| 2026-09-30 03:15 | trader-dds-6-r3 | us-east4-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:27 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 1.26 | 0.89 |
| 2026-09-30 04:01 | trader-dds-14 | europe-west4-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:32 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.49 | 0.35 |
| 2026-09-30 04:15 | trader-dds-2-r2 | us-west4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:25 (runner done (queue ended, no pending jobs); sha256-verified sync of 16 files) | 0.08 | 0.06 |
| 2026-09-30 04:16 | trader-dds-2-r2 | europe-west4-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:25 (runner done (queue ended, no pending jobs); sha256-verified sync of 16 files) | 0.16 | 0.12 |
| 2026-09-30 04:16 | trader-dds-18 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:20 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 2.13 | 1.51 |
| 2026-09-30 04:16 | trader-dds-17 | europe-west4-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:32 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.23 | 0.17 |
| 2026-09-30 04:16 | trader-dds-16 | europe-west4-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:32 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.23 | 0.16 |
| 2026-09-30 04:17 | trader-dds-15 | europe-west4-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:32 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.23 | 0.16 |
| 2026-09-30 04:17 | trader-dds-19 | us-east1-d | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:33 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.31 | 0.22 |
| 2026-09-30 04:30 | trader-dds-20 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:37 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.09 | 0.06 |
| 2026-09-30 04:33 | trader-dds-23 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:39 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 2.10 | 1.49 |
| 2026-09-30 04:34 | trader-dds-22 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:38 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 2.09 | 1.49 |
| 2026-09-30 04:34 | trader-dds-21 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:41 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.11 | 0.08 |
| 2026-09-30 04:39 | trader-dds-19-r1 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:41 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.11 | 0.08 |
| 2026-09-30 04:39 | trader-dds-16-r1 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:33 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 1.92 | 1.36 |
| 2026-09-30 04:39 | trader-dds-17-r1 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:41 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.11 | 0.08 |
| 2026-09-30 04:39 | trader-dds-15-r1 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:32 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 1.91 | 1.36 |
| 2026-09-30 04:40 | trader-dds-14-r1 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 05:00 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.35 | 0.25 |
| 2026-09-30 04:41 | trader-dds-20-r1 | us-east4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 04:46 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.16 | 0.11 |
| 2026-09-30 04:48 | trader-dds-21-r1 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:33 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.77 | 1.26 |
| 2026-09-30 04:48 | trader-dds-19-r2 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:33 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.77 | 1.26 |
| 2026-09-30 04:51 | trader-dds-17-r2 | us-east4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 05:28 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.08 | 0.06 |
| 2026-09-30 04:51 | trader-dds-17-r2 | europe-west4-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 05:28 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.59 | 0.42 |
| 2026-09-30 04:52 | trader-dds-20-r2 | us-central1-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 05:35 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.66 | 0.47 |
| 2026-09-30 05:07 | trader-dds-14-r2 | us-west4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 05:13 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.19 | 0.13 |
| 2026-09-30 05:25 | trader-dds-14-r3 | us-central1-c | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:37 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 1.23 | 0.87 |
| 2026-09-30 05:34 | trader-dds-17-r3 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 07:00 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 1.51 | 1.07 |
| 2026-09-30 05:43 | trader-dds-20-r3 | us-central1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:33 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.84 | 0.60 |
| 2026-09-30 06:38 | trader-dds-20-r4 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 07:22 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 0.80 | 0.57 |
| 2026-09-30 06:38 | trader-dds-19-r3 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 06:48 (runner done (queue ended, no pending jobs); sha256-verified sync of 16 files) | 0.20 | 0.14 |
| 2026-09-30 06:38 | trader-dds-21-r2 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 07:01 (runner done (queue ended, no pending jobs); sha256-verified sync of 15 files) | 0.45 | 0.32 |
| 2026-09-30 06:42 | trader-dds-14-r4 | us-west1-b | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 07:04 (gone (spot preemption or max-run; GCE deleted VM and disk)) | 0.37 | 0.26 |
| 2026-09-30 07:09 | trader-dds-14-r5 | us-west4-a | g2-standard-4 + L4 spot, max-run 6h, termination DELETE, 100 GB | 2026-09-30 07:21 (runner done (queue ended, no pending jobs); sha256-verified sync of 16 files) | 0.27 | 0.19 |
