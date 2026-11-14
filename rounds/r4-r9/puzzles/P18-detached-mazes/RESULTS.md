# P18 results: detached per-step deep supervision in random order, on mazes

## Status

**A1 KILLED: clearly worse** (the pre-registered rule, `analyze18.py`, run at 2026-09-30T07:29Z on all 22 runs).
Verification: an independent verifier recomputed the verdict from the raw eval files and agrees (with corrections, now
applied; section 11).

**Question.** On mazes, does training each recurrent step on its own loss from a detached state, with the block
drawn in random order (A1), extrapolate from 9x9 to 59x59 better than DT-Recall trained with the progressive loss,
at equal compute? The comparison uses 1.7 L4-hours of training per run, width 128, and exact maze accuracy at
T = 12n.

**Answer.** No. A1 extrapolates worse than the best DT-Recall progressive arm (alpha 1). On 59x59 it solves 1.7% of
mazes, against 8.1% for alpha 1; the difference is -6.4 points, and the one-sided 95% bounds are [-11.1, -1.7]. On
large mazes A1's four untied blocks lock into a period-4 cycle (69% of 59x59 mazes), and DT-Recall does not (2.5%).
The prefix-sum finding (random order >> round robin) shows up weakly at 19x19 and is gone by 59x59.

This matches the pre-registered prior: kill expected, at about 75%.

## 1. Primary result (PREREG section 7)

A4\* is chosen automatically from dev m59 over seeds 0-2 (`gpu/main_plan.json`):

| alpha | m59 exact@12n | m13 exact@12n |
|---|---|---|
| 0 | 0.020 | 0.285 |
| 0.5 | 0.017 | 0.240 |
| **1 (A4\*)** | **0.110** | 0.995 |

| quantity | value |
|---|---|
| A1 e59 exact@12n, 5 seeds | 0.017 (per seed 0.005, 0.050, 0.005, 0.005, 0.020) |
| A4\* e59 exact@12n, 5 seeds | 0.081 (per seed 0.135, 0.100, 0.025, 0.035, 0.110) |
| d59 = A1 - A4\* | -0.064 |
| LB59 / UB59 (Welch one-sided 95%, df 5.3) | -0.111 / -0.017 |
| d33 (e33) | -0.149 |
| rule | survives iff LB59 > 0 and d33 >= -0.05; killed "clearly worse" if UB59 < 0 |
| **verdict** | **A1 KILLED: clearly worse** |

Both arms trained on every seed (v9 exact@12n >= 0.9 in 5/5 and 5/5), so the budget clause does not apply.

## 2. Exact accuracy at T = 12n, by maze size

Seed mean (SD). v9, m13 and m59 are dev sets; e19, e33 and e59 are the test sets (200, 300 and 200 mazes).

| arm | seeds | trained | v9 | m13 | m59 | e19 | e33 | e59 |
|---|---|---|---|---|---|---|---|---|
| **A1** detached, random order (K=4) | 5 | 5/5 | 0.998 (0.00) | 0.986 (0.01) | 0.016 (0.02) | 0.841 (0.07) | 0.301 (0.18) | **0.017 (0.02)** |
| A2 detached, round robin (K=4) | 3 | 3/3 | 0.997 (0.00) | 0.947 (0.02) | 0.037 (0.06) | 0.595 (0.08) | 0.137 (0.10) | 0.030 (0.05) |
| A3 full BPTT, random order (K=4) | 3 | 0/3 | 0.530 (0.13) | 0.318 (0.18) | 0.007 (0.01) | 0.190 (0.21) | 0.078 (0.13) | 0.007 (0.01) |
| A4 DT-Recall progressive, alpha 0 | 3 | 1/3 | 0.369 (0.49) | 0.285 (0.39) | 0.020 (0.03) | 0.198 (0.29) | 0.063 (0.09) | 0.008 (0.01) |
| A4 DT-Recall progressive, alpha 0.5 | 3 | 0/3 | 0.412 (0.37) | 0.240 (0.22) | 0.017 (0.03) | 0.070 (0.08) | 0.027 (0.05) | 0.010 (0.02) |
| **A4\*** DT-Recall progressive, alpha 1 | 5 | 5/5 | 0.997 (0.01) | 0.997 (0.01) | 0.106 (0.04) | 0.885 (0.08) | 0.450 (0.16) | **0.081 (0.05)** |
| A5 detached, K=1 (exploratory) | not run: spend rule (see section 7) | | | | | | | |

Per seed. "e59 best over t" is an oracle stopping time, labelled as such. Cycle is the share of mazes whose
prediction is in a limit cycle of period 2-8 at T = 12n.

| run | v9 | e19 | e33 | e59 | e59 best over t (t) | e59 cycle | steps | train h |
|---|---|---|---|---|---|---|---|---|
| a1_s0 | 0.991 | 0.830 | 0.187 | 0.005 | 0.005 (50) | 0.76 | 22352 | 1.69 |
| a1_s1 | 1.000 | 0.965 | 0.613 | 0.050 | 0.050 (170) | 0.65 | 22352 | 1.74 |
| a1_s2 | 0.999 | 0.800 | 0.230 | 0.005 | 0.005 (130) | 0.73 | 22352 | 1.70 |
| a1_s3 | 1.000 | 0.775 | 0.223 | 0.005 | 0.005 (50) | 0.66 | 22352 | 1.71 |
| a1_s4 | 1.000 | 0.835 | 0.253 | 0.020 | 0.020 (60) | 0.65 | 22352 | 1.73 |
| a2_s0 | 0.992 | 0.670 | 0.257 | 0.090 | 0.095 (160) | 0.42 | 22376 | 1.70 |
| a2_s1 | 1.000 | 0.610 | 0.087 | 0.000 | 0.000 (10) | 0.49 | 22376 | 1.72 |
| a2_s2 | 1.000 | 0.505 | 0.067 | 0.000 | 0.000 (10) | 0.73 | 22376 | 1.69 |
| a3_s0 | 0.418 | 0.015 | 0.000 | 0.000 | 0.000 (10) | 0.34 | 20930 | 1.68 |
| a3_s1 | 0.669 | 0.430 | 0.233 | 0.020 | 0.020 (700) | 0.74 | 20930 | 1.68 |
| a3_s2 | 0.504 | 0.125 | 0.000 | 0.000 | 0.000 (10) | 0.30 | 20930 | 1.69 |
| p0_s0 | 0.923 | 0.535 | 0.163 | 0.020 | 0.035 (510) | 0.01 | 17672 | 1.70 |
| p0_s1 | 0.185 | 0.060 | 0.027 | 0.005 | 0.005 (40) | 0.00 | 17672 | 1.72 |
| p0_s2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 (10) | 0.00 | 17672 | 1.71 |
| p05_s0 | 0.811 | 0.050 | 0.000 | 0.000 | 0.000 (10) | 0.00 | 17485 | 1.68 |
| p05_s1 | 0.089 | 0.000 | 0.000 | 0.000 | 0.000 (10) | 0.00 | 17485 | 1.67 |
| p05_s2 | 0.337 | 0.160 | 0.080 | 0.030 | 0.050 (570) | 0.12 | 17485 | 1.72 |
| p1_s0 | 1.000 | 0.955 | 0.670 | 0.135 | 0.135 (490) | 0.01 | 60895 | 1.78 |
| p1_s1 | 0.987 | 0.795 | 0.317 | 0.100 | 0.100 (260) | 0.06 | 60895 | 1.76 |
| p1_s2 | 1.000 | 0.820 | 0.317 | 0.025 | 0.025 (70) | 0.06 | 60895 | 1.77 |
| p1_s3 | 1.000 | 0.870 | 0.370 | 0.035 | 0.035 (70) | 0.01 | 60895 | 1.76 |
| p1_s4 | 1.000 | 0.985 | 0.577 | 0.110 | 0.115 (190) | 0.00 | 60895 | 1.77 |

- **No run went non-finite.** The divergence gate never fired.
- **p0_s2 learned, then collapsed.** Its training log (`logs/p0_s2.log`, val9_acc30 on 2,000 held-out 9x9 mazes)
  peaked at 0.998 at 14.4k steps. The loss spiked at 15.0k, and val9 was 0.066 at 15.2k and 0 at 16.0k (loss
  0.59).
- **p0 and p05 are unstable at this budget.** They used the official maze recipe, which has no gradient clipping
  (`clip=None` in dt-baseline/code/timing.py).
- **Oracle stopping does not rescue A1.** Its best-over-t equals its value at T = 12n on every seed. This holds on
  the recorded grid only: multiples of 10, plus 12n. For a period-4 cycle, the phases t mod 4 = 1 and 3 were never
  scored.

## 3. Mechanism (secondary contrasts; none decides the verdict)

Welch two-sided 95% intervals (`results/analysis.json`, "secondary").

| contrast | e19 | e33 | e59 |
|---|---|---|---|
| A1 - A2 (order) | +0.246 [0.082, 0.410] | +0.165 [-0.078, 0.408] | -0.013 [-0.130, 0.104] |
| A1 - A3 (detachment) | +0.651 [0.159, 1.143] | +0.224 [-0.055, 0.502] | +0.010 [-0.017, 0.037] |
| A3 - A4\* | -0.695 [-1.178, -0.212] | -0.372 [-0.645, -0.100] | -0.074 [-0.134, -0.015] |
| A2 - A4\* | -0.290 [-0.454, -0.126] | -0.313 [-0.546, -0.080] | -0.051 [-0.153, 0.051] |
| A1 - alpha 0 | +0.643 [-0.051, 1.336] | +0.238 [0.009, 0.467] | +0.009 [-0.017, 0.035] |
| A1 - alpha 0.5 | +0.771 [0.610, 0.932] | +0.275 [0.059, 0.490] | +0.007 [-0.028, 0.042] |
| A1 - alpha 1 | -0.044 [-0.158, 0.070] | -0.149 [-0.396, 0.099] | -0.064 [-0.123, -0.005] |

The pre-registered readings were:
- A1 about A3, both above A2: order carries the effect;
- A1 above A3: detachment adds something;
- A1 about A2: order does not matter.

What happened fits none of them cleanly.

- **Order helps a little, and only at moderate size.** Random order beats round robin by +0.25 on 19x19
  (interval above 0). The e33 interval includes 0, and on 59x59 both are near 0. The prefix-sum effect replicates
  in direction at 2x extrapolation and is gone by 6.5x.
- **Detachment helps trainability, not extrapolation.** Full BPTT through 30 steps of four untied blocks, with no
  gradient clipping, did not train in 1.7 h: 0/3 seeds, v9 0.53. A1 trained on 5/5. The large A1 - A3 gap on e19 is
  therefore an optimisation result. On 59x59 both are about 0.
- **The K=4 models fall into the test-time round-robin period on large mazes.** Periods of the prediction on e59 at
  T = 12n, as a share of mazes (from `ev_*.npz`, `e59/cyc`; 0 = no period up to 8, still changing):

  | arm | fixed point | period 4 | other period 2-8 | no period up to 8 |
  |---|---|---|---|---|
  | A1 | 0.055 | 0.685 | 0.004 | 0.256 |
  | A2 | 0.372 | 0.545 | 0.003 | 0.080 |
  | A3 | 0.007 | 0.457 | 0.005 | 0.532 |
  | A4\* (alpha 1) | 0.622 | 0.001 | 0.024 | 0.353 |

  The cycle share (any period 2-8) grows with size for A1: 0.009 on m13, 0.088 on e19, 0.477 on e33 and 0.689 on
  e59.
  - On small mazes the four untied blocks agree on one answer: m13 cycles only 0.9% of the time.
  - On 59x59 they mostly do not. Only 5.5% of mazes reach a fixed point, and 69% alternate with the round-robin
    period.
  - The tied DT-Recall block settles to a fixed point on 62% of 59x59 mazes, and cycles on 2.5%.
- **The winner is also trained from detached states.** Alpha 1 is Bansal et al.'s progressive term alone: n steps
  without gradient, then k steps with gradient, for random n and k. So "detached versus not" is not what separates
  A1 from A4\*. What does separate them is:
  - one tied block against four untied ones;
  - random-length gradient windows against one-step windows.

  This experiment does not isolate which of the two matters; A5 (K = 1, one-step windows) would have been the
  pointer, and it did not run.

## 4. References outside equal compute (reported, not tested)

All are on the same e19, e33 and e59 mazes, at exact@12n.

| reference | training compute | e19 | e33 | e59 |
|---|---|---|---|---|
| M-ref: released official DT-Recall checkpoint, evaluated here | paper recipe | 1.000 | 1.000 | 0.995 |
| Official 50-epoch reproduction (dt-baseline `trader-dt-m*`) | ~7.1 L4-h per seed in strict FP32 (m0 7.08, m1 7.13, m2 7.25; 40,000 steps) | 0.945 (m0 0.845, m1 1.0, m2 0.99) | 0.965 (m1 1.0, m2 0.93) | 0.915 (m1 0.99, m2 0.84) |
| P13 DTR-eq: DT-Recall alpha 0.01, same recipe, 30456 steps | ~3.1 L4-h | 0.701 | 0.546 | 0.308 (0.075, 0.925, 0.23, 0; s3 did not train) |
| CostNet+BB (followups-batch2 item 1, 20 seeds; different model class) | n/a | - | 0.580 [0.548, 0.611] | 0.234 [0.214, 0.254] |
| **A4\*** here (alpha 1) | 1.7 L4-h | 0.885 | 0.450 | 0.081 |
| **A1** here | 1.7 L4-h | 0.841 | 0.301 | 0.017 |

- **M-ref** passed the pipeline gate: m59 1.0 against the 0.95 gate. Its dev scores are v9, m13 and m59 = 1.0.
- **dt-baseline m0.** Its e33 and e59 evals had not landed when this was checked at 07:30Z. PREREG allows until
  16:30Z; the dt-baseline lane may add them.
- **Maze identity for dt-baseline.** The code sha256 of its P13 e-sets equals P18's frozen manifest, and the packed
  solution sha256 equals P13's.
- **CostNet.** It is scored from its per-maze arrays on exactly these slices. The maze-order check passed: the
  e33/e59 lattice codes rebuilt from the raw official tarballs match the frozen sha256
  (`results/costnet_order_check.json`).
- **Budget.** Every equal-compute arm is well below the official recipe.
  - The official recipe trains 40,000 steps (50 epochs) in strict FP32. dt-baseline's smoke timing
    (`dt-baseline/results/smoke_p0/timing.json`) puts that at 0.568 s per batch, against 0.376 s with TF32. P18
    and P13 trained with TF32.
  - In TF32-equivalent hours the official recipe is about 2.5-2.8x P18's B = 1.7 h: 4.2 h of pure training by the
    smoke timing, or 4.7 h scaling the realised 7.1 h.
  - In steps it is 1.8x A1's 22,352 and 2.3x alpha 0's 17,672.
  - At this budget even the winner (alpha 1) solves 8% of 59x59 mazes. The official recipe solves 92% and P13's
    30k-step run 31%.

## 5. Equal compute audit

Equal compute is equal L4 wall time, fixed at B = 1.7 h of training per run:
- steps(arm) = floor(B x 3600 / s(arm)), from clean solo timings in Phase 1;
- B came from `plan.py main`: V 56.34 VM-h, P1 0.483 h, O 0.421 h, N 22;
- realised training time was 1.67-1.78 h per run.

FLOP counters, per training step, then the run total in block forward-equivalents. A backward pass counts as two
forwards, and the head costs 5.3% of one block application at width 128.

| arm | steps | blocks with grad | blocks without grad | head with grad | total (M), blocks only | total (M), with head | vs A1 |
|---|---|---|---|---|---|---|---|
| A1 | 22352 | 30 | 0 | 30 | 2.01 | 2.12 | 1.00 |
| A2 | 22376 | 30 | 0 | 30 | 2.01 | 2.12 | 1.00 |
| A3 | 20930 | 30 | 0 | 30 | 1.88 | 1.98 | 0.94 |
| alpha 0 | 17672 | 38.19 | 14.55 | 2 | 2.28 | 2.29 | 1.08 |
| alpha 0.5 | 17485 | 38.19 | 14.57 | 2 | 2.26 | 2.26 | 1.07 |
| alpha 1 | 60895 | 8.25 | 14.49 | 1 | 2.39 | 2.40 | 1.13 |

- **A4\* got more than A1 on every compute measure except intended wall time.**
  - About 13% more counted FLOPs, because no-gradient forwards are cheap on the L4.
  - 2.7x the optimizer steps and data passes: 60,895 against 22,352.
  - About 3% more realised wall time: 1.76-1.78 h against A1's 1.69-1.74 h. Validation and checkpoint overhead
    scales with step count, and it sits outside the per-step timing.

  At equal FLOPs, A1 would have had about 25k steps. That is unlikely to lift e59 from 0.017 to above 0.081,
  since A1 is in a period-4 cycle on 69% of those mazes, but it was not tested.
- **The alpha 0 arm wastes about 30% of its compute.** It computes the progressive branch even at weight 0: the
  code is inherited bit-identical from P13's `dtr_maze.py` (unit test (a)), so the 8.25 grad + 14.5 no-grad block
  applications per step do nothing. A clean alpha 0 would have had about 44% more steps at equal counted FLOPs.
  At equal wall time it would have had about 18-26% more: its 0.346 s per step, against 0.292 s for A3 and
  0.274 s for A1.
  - That handicap could only have hidden a stronger A4\* (P13's alpha 0.01 reaches e59 0.31 with 3.1 h).
  - A stronger A4\* would make A1's kill clearer, not weaker.

## 6. Scope and caveats

- **Short budget.** B = 1.7 L4-h is about 36-40% of the official 50-epoch recipe in TF32-equivalent hours
  (section 4). The kill holds at this budget.
  Whether A1 gains more than DT-Recall from longer training is untested. A1's 59x59 failure mode is a period-4
  cycle, and nothing in its objective pushes the four blocks to agree.
- **Test order.** K=4 models are evaluated in round robin, as pre-registered. Other test-time schedules were not
  evaluated: random order, one block repeated, or averaging the four blocks.
- **No gradient clipping,** as in the official maze recipe.
  - market-resnet's prefix-sum full-BPTT arm (449/600) clipped globally at norm 1.0 (`econ2.py`, lines 612 and
    683). Only its 529/600 twin clipped per trader.
  - That full-BPTT arm trained, while A3 here did not. The difference between tasks may be partly clipping.
- **Seeds.** 5 per primary arm; 3 for A2, A3 and alpha 0 / 0.5. The secondary intervals are wide.

## 7. Other checks from the assignment

- **market-resnet `tb1_rand`.** Its §17.2 was rechecked for this write-up: prefix sums, 4 untied blocks, 100x
  length extrapolation, 600 fresh strings.

  | arm | solved | 95% interval |
  |---|---|---|
  | detached, random order (`tb1_rand`) | 482/600 | [0.77, 0.83] |
  | detached, round robin (`tb1`) | 71/600 | |
  | full BPTT, random order | 449/600 | |
  | full BPTT, round robin | 56/600 | |
  | DT-Recall at equal FLOPs | 600/600 | |

  So random order, not detachment, carried the effect there. On mazes that order effect is small and does not
  reach 59x59. DT-Recall wins in both settings.
- **A5 (K=1, exploratory) was not run.** The pre-registered spend rule was: after all primary evals, run it only if
  the estimated spend + 2 x (B + O) x $0.71 x 1.3 + $2 <= $40. The estimate was $35.73 + $5.92 = $41.65, so it did
  not run (`gpu/main_plan.json`, "a5").
- **Unit tests a-f passed** before any GPU use (`results/unit_tests/`). The M-ref gate passed. The data sha256
  gates passed on every VM start.

## 8. Cost and cleanup

- **Spend.** $36.40 estimated at the $0.71/h ceiling, of the $40 cap. 61 VMs were created (spot L4,
  g2-standard-4).
  - The watchdog's figure was $35.73. It never saw 7 VMs that spot reclaimed within 1-6 minutes of creation,
    because it only looks every 5 minutes.
  - Those 7 were added to `gpu/spend-state.tsv` from `gpu/vms.tsv` and `gpu/deleted.tsv` at write-up (+$0.67),
    and the ledger was rerendered.
  - The ledger's 65 rows also include 4 create attempts that lived for seconds in a stocked-out zone. Each is
    charged the 5-minute minimum, so the figure is conservative.
  - A5's spend rule gives the same answer with the corrected figure: $36.40 + $5.92 > $40.
  - 23 slots: Phase 1 plus 22 runs.
  - `gpu/ctl.log` records 38 spot-preemption (GONE) events, each followed by a relaunch (38 `-rN` VMs in `gpu/vms.tsv`; 36 of them started jobs before their own
    preemption or completion). Each relaunch resumed from the last
    sha256-verified checkpoint sync. Kill-and-resume is bit-identical (unit test (d)).
- **Cleanup.** All P18 VMs and disks were deleted: `gcloud compute instances list` and `disks list` returned no
  trader-dds-* at 2026-09-30T07:23:43Z. ctl stopped at 07:25:47Z; the watchdog's last pass (07:26:04Z) saw 0 VMs.
  The same two gcloud lists, filtered to `name~^trader-dds` in project GCP_PROJECT, were empty again at 07:34:17Z.
- **Ledger:** `trader-economy/GPU-LEDGER.md`, section "P18 detached mazes".

## 9. Deviations from PREREG

1. **`gpu/common.sh` edited twice (infra only; this is why its sha256 no longer matches `PREREG.sha256`).** The
   zone preference list `ZONES_FIRST` changed after spot preemption storms (`gpu/DEVIATIONS.log`). Arms, budgets
   and analysis were unchanged.
   - 02:27Z: us-central1 L4s were preempted within minutes.
   - 04:53Z: us-east4-a lost 10/10 launches, us-central1-a 5/5, us-central1-c 8/14 and europe-west4-b 4/5.
2. **`refs18.py` was written after the freeze.** It implements the section 3 reference rules (the CostNet order
   check and the collection). The P13 DTR-eq and dt-baseline numbers were read by hand into
   `results/references_extra.json`, with sources and sha checks in its notes. References are not tested.
3. **P1 was measured while a Phase 1 VM was still alive** (0.483 VM-h), so it slightly understates Phase 1. B
   floors to 1.70 for any P1 up to about 1.2 VM-h, so this does not change B.
4. **dt-baseline was reported early.** It is given at 07:30Z with m0 incomplete (e19 only), rather than waiting
   for 16:30Z.
5. **Spend accounting.** Two corrections were made at write-up (section 8):
   - the ledger now includes 7 VMs, of 1-6 minutes each, that the 5-minute watchdog missed (+$0.67);
   - in the ledger, 4 slot names appear twice, once for a seconds-long create attempt in a stocked-out zone and
     once for the real VM (e.g. `trader-dds-2-r1`).

   While running, the $40 spend gate saw at most $0.67 less than the true estimate, and it never came close to
   firing.

## 10. Files

- `PREREG.md`, `PREREG.sha256` (frozen 01:23:48Z), `power.py`, `results/power.json`
- Code:
  - `train18.py`, `eval18.py`, `dtmod.py`, `dtnet.py`, `check.py`;
  - `analyze18.py`, `refs18.py`, `tests_cpu.py`;
  - `gpu/*`: controller, watchdog, plan, ledger, `DEVIATIONS.log`, `main_plan.json`.
- `results/analysis.json`: every number in sections 1-3.
- `results/references.json`, `results/references_extra.json`, `results/costnet_order_check.json`.
- `results/vm/<slot>/out/`: per-run training logs, `ev_<run>.json`, `ev_<run>.npz`.
- Charts:
  - `charts/exact12n_by_size.json`: per-arm seed values on e19/e33/e59, with the references;
  - `charts/e59_exact_vs_t.json`: e59 exact against test step, per run.

## 11. Independent verification (2026-09-30, about 07:45Z)

A separate read-only agent **agrees** with "A1 KILLED: clearly worse".

**What it recomputed from the raw files:**
- **Primary.** From the 22 `ev_<run>.json` files with `.done` markers: d59 -0.064, LB59 -0.1105, UB59 -0.0175,
  df 5.27, d33 -0.149. A one-sided Mann-Whitney test also gives p = 0.016, so the kill does not rest on Welch's
  normality assumption.
- **A4\* selection.** It re-applied the selection rule from the dev m59 evals and got alpha 1 again.
- **Tables.** Every table value and trained count in sections 1-3 matches the ev json and npz files.
- **PREREG hashes.** Only `gpu/common.sh` fails. Removing its one added `ZONES_FIRST` line restores the frozen hash
  exactly, and `analyze18.py` is unchanged.
- **Ledger.** Its rows sum to $36.39, against $36.40 from `spend-state.tsv` (a rounding difference). The +$0.67
  for the 7 short VMs checks out.
- **References.** The dt-baseline, P13 DTR-eq and CostNet figures match their sources.
- **Cleanup.** gcloud lists were empty at 07:42Z.

**Corrections it asked for, all applied above:**
- 38 relaunches, not 35.
- A2's "other period" share is 0.003, not 0.004.
- The official recipe was FP32, so the budget gap is about 2.5-2.8x in TF32-equivalent hours, not 4x.
- The clean alpha-0 step gain is 18-26% at equal wall time, not 40%.
- market-resnet's full-BPTT arm clipped globally, not per trader.
- "never agree" overclaimed.
- Added caveats: A4\* had 2.7x the optimizer steps and about 3% more realised wall time, and best-over-t is limited
  to its recording grid.
- p0_s2's collapse timing was corrected.

**Not checked by the verifier:**
- the content of the 02:27Z `common.sh` edit (no copy of that version was kept; `gpu/DEVIATIONS.log` records it);
- a rerun of the CostNet raw-tarball order check;
- checkpoint integrity across relaunches (covered by unit test (d), not re-run).

