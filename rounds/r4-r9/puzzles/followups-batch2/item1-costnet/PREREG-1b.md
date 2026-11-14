# Item 1b pre-registration: CostNet+BB vs official DT-Recall on the Bansal et al. maze benchmark

Written 2026-09-29 ~23:23Z, before any 1b seed run. No 1b test number (e2h_maze_13/33/59/201) has been computed.
The pilots below touched only maze_data_train_9 (train rows 0–47999, val rows 48000–49999). The sha256 is frozen
in `PREREG-1b.sha256`.

## Why 1b exists
P12's DT baseline failed its own validity gate, so 1a cannot show that CostNet+BB beats a working DT. 1b puts
CostNet+BB on the official easy-to-hard maze data and scores it with the official exact rule. The DT reference is
the released DT-Recall maze checkpoint (`dt-baseline/released/training-rusty-Tayla`), evaluated by the
dt-baseline agent's `dt_eval.py` on the same mazes. The set sizes and first-N counts match `dt_eval.py` SETS.

## Arm (code/bansal.py; the port is described in its docstring)
- **Input.** Blocks = pixels[3:-3:2, 3:-3:2] (n×n). This is a fixed tokenisation that DT does not get. S is the
  green block and G is the red block.
- **Graph.** The graph is every 4-neighbour block pair; walls are not removed. The net must price the walls.
- **Network.** CostNet is conv3x3(3→32) ReLU conv3x3(32→1), then softplus + 0.01. Edge cost is the mean of the
  two cells.
- **Solver.** Training uses exact Bellman-Ford with blackbox differentiation (λ = 20) and a Hamming loss over
  edges. Test uses exact scipy Dijkstra; `selftest` checks it against Bellman-Ford.
- **Optimiser.** Adam 1e-3 with 200-step warm-up then cosine to 10%, batch 64, clip 1.0, 12000 steps, and the 8
  dihedral maps.
- **Two changes from P12**, both chosen on train/val:
  1. **BBPos.** The perturbed costs are clamped at 1e-3 instead of 0. With 0, a target edge whose cost goes
     below λ|dL/dy| costs exactly 0, and the solver's walk-back cycles. On 64 val mazes with λ = 1280 and oracle
     targets, 2 of 64 walks ended with clamp 0 and 64 of 64 with clamp ≥ 1e-6.
  2. **--norm 1.** Each maze's costs are divided by their mean.
- **Seeds** 200–219 (n = 20; power in `power.py` / `power.txt`, written before this file). The command is
  `code/run1b.sh`.

## Pilot history (train/val only, seeds 900–930; all logs in /tmp/fu-b2/smoke, summarised here)

| recipe | result |
|---|---|
| P12 verbatim, λ 20 | ~15% train exact after 12k steps; costs inflate to 60–390 |
| λ 1280 | 18% |
| norm (clamp-0 bug still in) | 7% |
| Free per-maze costs, no network, clamp 0 | no learning at λ 20 or 1280. This found the clamp bug. |
| Free costs with BBPos, λ 1280 | 100% in 150 steps |
| Free costs with BBPos, λ 20 | no gradient, because the perturbation is too small |
| BBPos + network, no norm, λ 20 / 320 / 1280 | all collapse to the 0.01 floor with zero gradient by step ~300 |
| BBPos + norm, λ 20 (seed 930) | val exact 0.999 at step 500 |
| BBPos + norm, λ 80 and λ 320 | train exact 0.99 by step 300; stopped |

**Chosen:** BBPos + norm with P12's λ = 20. No other value was tuned.

## Gates (automatic; code/gate.py 1b and run1b.sh; a failed gate stops all test evaluation)
- **G1.** The data sha256 check and `bansal.py selftest` pass. Selftest covers Dijkstra = Bellman-Ford at
  n = 9/13/33, the pixel and edge round trips, and oracle exact = 1.0 on val.
- **G2.** The median over the 20 seeds of val exact is ≥ 0.90. Failed seeds count as 0, and at least 90% must
  train.
- **DT validity gate.** The released DT-Recall checkpoint must reach peak exact ≥ 95% on e2h_maze_13, as reported
  by dt-baseline. The equivalence check already showed 100% on the first 500 mazes.
  - If dt-baseline has no e2h_maze_13/59 result for the released checkpoint when I write RESULTS, the
    head-to-head is reported as "DT side pending". CostNet's absolute numbers are then reported alone. No DT
    number is invented or taken from the paper as a substitute; paper numbers may be quoted only as context,
    labelled.

## Primary
- **Metric.** e2h_maze_59 exact on the first 1000 mazes, with the official rule: every pixel of the open-masked
  prediction equals the solution.
- **CostNet.** The mean over the 20 seeds and its 95% t-CI.
- **DT, two numbers.**
  - (a) Peak over iterations, as dt_eval reports it. This is an oracle stopping choice made on test, so it is
    generous to DT.
  - (b) Accuracy at the final evaluated iteration.
  - The primary comparison uses (a). DT is a single released model; its uncertainty is not modelled.

## Bars (fixed now)
- **Non-inferior.** The CostNet CI lower bound is > DT(a) − 5 points.
- **Superior.** The CostNet CI lower bound is > DT(a) + 5 points.
- **Worse.** The CostNet CI upper bound is < DT(a) − 5 points.
- Anything else is **inconclusive**.

## Secondaries (no bars)
- e2h_maze_13, e2h_maze_33 and e2h_maze_201 (first 100) exact, with the same statistics.
- Valid rate: the path uses no wall block.
- The median wall/open cost ratio.
- The DT from-scratch lanes (dt-baseline m0–m2), if their maze evals exist.

## Fixed now
- No retuning after test numbers, no seed exclusion, and no checkpoint choice. The final step-12000 model is used.
- **What this comparison is.** DT learns the whole algorithm from pixels. CostNet+BB is handed three things:
  - the block tokenisation;
  - the grid graph;
  - an exact shortest-path solver.

  It only has to learn "black block = expensive". A CostNet win therefore measures the value of that built-in
  structure, not a better learner. RESULTS must say this next to the numbers.

## Compute
- `trader-fu-c1-1` (the 1a VM, c2d-highcpu-56). The 1b queue runs after or beside 1a.
- The item-1 cap is $9.
