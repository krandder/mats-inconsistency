# P18 pre-registration: detached per-step deep supervision in random order, on mazes

Frozen before any GPU spend (the timing phase included). The sha256 of this file and of the code it names is in
`PREREG.sha256`. Any later change goes in a dated "Deviations" section at the end of RESULTS.md, never in this file.

## 1. Question and prior

**Question.** On mazes, does training each recurrent step on its own loss from a detached state, with the block used
at each step drawn in random order (A1), extrapolate from 9x9 to 59x59 better than DT-Recall trained with the
progressive loss, at equal compute?

**Where it comes from.** THEORY-3.md §5 item 1 (the "§6(a)" of the assignment). market-resnet/RESULTS.md §17.2
(prefix sums, 4 untied blocks, 100x length extrapolation, 600 fresh strings):
- detached one-step training in random order (`tb1_rand`): 482/600 [0.77, 0.83];
- the same in round-robin order (`tb1`): 71/600;
- full BPTT in random order: 449/600; in round-robin order: 56/600.

So training order, not detachment, carried the effect there ("Detachment alone fails"). On the same strings,
DT-Recall with its published recipe scored 589/600, and 600/600 at equal FLOPs.

**Prior.** A1 is about equal to A3 (full BPTT, random order) and better than A2 (round robin). A1 does not beat
DT-Recall with progressive loss on 59x59 mazes. The kill (section 7) is the expected outcome, at maybe 75%.

## 2. Data (frozen copies, sha256 in `frozen/SHA256SUMS.inputs`; every VM checks them before any job)

| role | set | source | mazes | side (pixels) | T = 12n |
|---|---|---|---|---|---|
| train | `maze_train9` | official easy-to-hard `maze_data_train_9`, 40k split (random_split 40000/10000, seed 42; P0 frozen) | 40,000 | 24 | (T = 30 in training) |
| dev | `v9` | the first 1,000 of the 10k held-out split of `maze_data_train_9` (never trained on; the training log uses the first 2,000) | 1,000 | 24 | 108 |
| dev | `m13` | `test_13[0:200]` (P0) | 200 | 32 | 156 |
| dev (alpha selection) | `m59` | `test_59[0:100]` (P0) | 100 | 124 | 708 |
| test | `e19` | `test_19[200:400]` (P13 frozen) | 200 | 44 | 228 |
| test | `e33` | `test_33[400:500] + [600:800]` (P13) | 300 | 72 | 396 |
| test | `e59` | `test_59[500:700]` (P13) | 200 | 124 | 708 |

The test slices are disjoint from the dev slices. Both freezes predate P18: P0 is from 2026-09-29 12:47Z and P13
from 20:12Z.

## 3. Model, recipe and arms

**Shared recipe.** This is P13's `dtr_maze.py`, which is P0's reproduction of the official deep-thinking maze
recipe:
- DT-Recall 2D at width 128, T = 30 training iterations, batch 50.
- Adam at lr 1e-3 with weight decay 2e-4. Recurrent parameters use lr/T (the official `lr_throttle`).
- Exponential warm-up in 10 stages over the first 20% of the step budget. No gradient clipping.
- Cross entropy over open pixels. TF32 in training.

The shared recipe gets no per-arm tuning. `train18.py --mode prog` is bit-identical to P13's `dtr_maze.py` (unit
test (a)).

**K-block model (A1-A3).** `dtmod.DTRecallK`:
- It has K = 4 untied recurrent blocks, each exactly DT-Recall's recurrent block (recall conv w+3 -> w, then 2 basic
  blocks).
- The projection and the head are shared.
- One step costs exactly one DT-Recall step.
- At test the blocks run in round-robin order: step t uses block (t-1) mod 4.

| arm | model | loss per training iteration (T = 30) | order of blocks in training |
|---|---|---|---|
| **A1** | K = 4 | mean over t of CE(head(h_t)); h_t = block_i(stopgrad(h_(t-1)), x) for t >= 2 (each step trained on its own loss from a detached state) | uniform random per step (one draw per batch; market-resnet `tb1_rand`) |
| A2 | K = 4 | as A1 | round robin (t-1) mod 4 (market-resnet `tb1`) |
| A3 | K = 4 | as A1 but no stopgrad: full BPTT through all 30 steps | random, as A1 |
| A4(alpha) | DT-Recall (1 tied block) | Bansal et al. progressive loss (1-alpha) CE(T = 30) + alpha CE(k steps after n detached), n ~ U[0, T-1], k ~ U[1, T-n]; alpha in {0, 0.5, 1} | n/a |
| A5 (exploratory) | K = 1 | as A1 | n/a |

- **A4\*** is the A4 alpha with the best mean dev `m59` exact@12n over seeds 0-2. Ties go to the higher mean `m13`
  exact@12n, then to alpha 0.5.
- **References outside equal compute.** These are reported, not tested.
  - P13's DT-Recall "DTR-eq": alpha 0.01, the same recipe, 3 L4-hours, 4 seeds, evaluated by P13 on the same
    e-sets.
  - The official deep-thinking reproduction (dt-baseline lanes `trader-dt-m*`): the full 50-epoch recipe, 3 seeds,
    if its evaluations on the P13 e-sets exist by 16:30Z.
  - The released official checkpoint (M-ref), evaluated here.
  - CostNet+BB: followups-batch2 item 1, 20 seeds. It is scored on exactly the e33 and e59 mazes, from its per-maze
    arrays, only if the maze order is verified against the raw easy-to-hard files. Otherwise its own full-test-set
    numbers are quoted and labelled as such.

## 4. Equal compute, seeds, launch order

**Compute.** Every training run gets the same L4 wall time, B hours of training steps:
- steps(arm) = floor(B x 3600 / s(arm)), where s(arm) is that arm's seconds per step.
- s(arm) comes from a clean solo timing on one L4 in Phase 1: 300 steps after 20 warm-up steps, one job on the GPU.
  One training job runs per VM.
- The counters of block and head applications, with and without gradient, are saved for a FLOP audit (reported,
  not used for budgets).

**B.** B = floor to 0.05 h of (V (1 - 0.15) - P1 - N x O) / N, clipped to [1.25, 2.0]:
- V = $40 / $0.71 = 56.3 VM-hours;
- P1 = VM-hours used by Phase 1;
- N = 22 runs (the 20 primary runs plus the 2 extra seeds of A4\*);
- O = the M-ref evaluation time on all six sets, plus 0.2 h for boot, push, sync and delete.

`gpu/plan.py main` computes B and writes it to `gpu/main_plan.json`, with no human choice.

**Seeds** (section 5):
- A1: 5 (seeds 0-4).
- A4: 3 per alpha (seeds 0-2); A4\* then gets seeds 3 and 4, so it has 5.
- A2 and A3: 3 each (seeds 0-2).
- A5: 2, only if spend allows. After all primary evals are done, it runs if the estimated spend plus
  2 x (B + O) x $0.71 x 1.3 + $2 is at most $40.

**Launch order** (at most 12 VMs at once):
1. The 9 A4 runs.
2. A1.
3. A2.
4. A3.
5. The A4\* extra seeds, as soon as the 9 A4 evals exist (`gpu/auto.sh` runs `plan.py extra`).
6. A5, if the spend rule allows.

**Evaluation.** Every run is evaluated on its own VM right after training (`eval18.py`, TF32 off) on
v9, m13, m59, e19, e33 and e59, to T = 12n, recording every 10 steps.

## 5. Power (`power.py`, `results/power.json`)

The test is Welch one-sided at 0.05 on seed-level e59 exact@12n.

With normally distributed seed scores, SD 0.25 and a true gap of 0.5, power is:

| seeds (A1 / A4\*) | power |
|---|---|
| 4 / 3 | 0.69 |
| 4 / 4 | 0.79 |
| 5 / 5 | 0.89 |

The dangerous case is a mixture. P13's DT-Recall left the ln 2 loss plateau in only 2 of 4 seeds by about 1.4
L4-hours. If A1 always trains, DT-Recall trains with p = 0.5, and both score 0.9 when trained, power is only 0.46 at
5/5. If DT-Recall trains with p = 0.25, power is 0.86. The false-positive rate is at most 0.05 in every model
checked.

5 / 5 seeds for the primary contrast is the most the $40 cap allows, together with the alpha sweep and the A2 and
A3 mechanism arms. THEORY-3 also asked for 5 seeds.

## 6. Metrics

- **Exact maze accuracy.** A maze counts as solved when every open pixel's prediction (logit_1 > logit_0) equals
  the solution.
- **Primary: exact at T = 12n** (the P0/P13 convention). It is reported per size (e19, e33, e59) and on dev.
- **Secondary:**
  - exact at T = 30;
  - best over t of the mean exact (an oracle stopping time, labelled as such);
  - log loss per open pixel;
  - cycle rate;
  - trained rate (a seed "trained" if v9 exact@12n >= 0.9).
- **Unit.** A seed's score is its accuracy over the maze set; arms are compared over seed scores.

## 7. Analysis and decision rule (`analyze18.py`)

**Primary.** d59 = mean over A1 seeds - mean over A4\* seeds, on e59 exact@12n. LB59 and UB59 are its Welch
one-sided 95% bounds (lower and upper).

- **A1 survives** if LB59 > 0 and d33 >= -0.05 (the same difference on e33).
- **Otherwise A1 is killed** (the assignment's kill rule: arm 1 does not beat DT-Recall progressive at the largest
  sizes). A kill is reported as:
  - "clearly worse" if UB59 < 0;
  - "no gain shown" otherwise.
- **Too few seeds.** If A1 or A4\* has fewer than 3 finished seeds, the verdict is "inconclusive (runs missing)".
- **Diverged runs** are evaluated as saved and counted.

**Budget clause.** If A1 survives but A4\*'s trained rate is below A1's, RESULTS must say so. The claim is then
limited to "at B L4-hours", and A1 is set beside the references outside equal compute (P13 DTR-eq, the official
50-epoch reproduction and M-ref).

**Secondary contrasts.** These are Welch 95% two-sided intervals on e59, then e33; they are reported, and none
decides the verdict:
- A1 - A2 (order);
- A1 - A3 (detachment);
- A3 - A4\*;
- A2 - A4\*;
- every A4 alpha against A1.

A5 is descriptive only.

**Reading of the mechanism**, stated before the data:
- A1 about A3 and both > A2: order carries it, as on prefix sums.
- A1 > A3: detachment adds something on mazes.
- A1 about A2: order does not matter on mazes.

## 8. Gates (automatic)

| gate | where | action |
|---|---|---|
| data sha256 | `gpu/run_p18.sh` on every VM start | mismatch: no job runs, `GATE_DATA_FAIL`, `ctl.sh` touches STOP (all P18 VMs deleted) |
| M-ref pipeline | `plan.py main` | released checkpoint m59 exact@12n < 0.95: STOP, no main runs |
| spend | `gpu/watchdog.sh`, 5-min loop | estimated spend (every trader-dds VM, (last seen - created + 5 min) x $0.71) >= $40: STOP, delete every trader-dds VM |
| deadline | `ctl.sh` | 17:45Z: final sync, delete all; no launch within 30 min of it; `watchdog.sh` and a final gcloud sweep make sure all are deleted before 18:30Z |
| age / idle / stale | `watchdog.sh`, `ctl.sh` | VM at 5.7-5.8 h, idle 10-20 min, or heartbeat stale 20 min: sync, delete; relaunch from checkpoint if work remains |
| divergence | `train18.py` | non-finite loss: stop, save, mark diverged |
| preemption | `ctl.sh` | spot VM gone: relaunch `SLOT-rN` from the last sha256-verified sync (checkpoints every 5 min, sync every 3 min) |

## 9. Unit tests before any GPU (`tests_cpu.py`, `results/unit_tests/`)

- (a) Prog mode is bit-identical to P13's `dtr_maze.py`.
- (b) DTRecallK with K = 1 initialises like DT-Recall.
- (c) The detach 1 and detach 0 gradients match independent references, and differ from each other. The rr
  picks are 0, 1, 2, 3, ...
- (d) Kill-and-resume is bit-identical.
- (e) `eval18.py`:
  - M-ref scores 1.0 on 10 m13 mazes;
  - K = 4 evaluation uses round robin;
  - eval resume is identical.
- (f) The divergence gate works.
