# P11: market-shaped wealth budget vs shape-free budgets for long-depth maze extrapolation

Run 29 Sep 2026, 13:02 to 19:46Z, on spot L4 VMs in GCP_PROJECT. Width 32, T_train = 30, 6000 steps, core seeds 1 to 5, R-final seeds 1 to 3. Every number below is exact-maze accuracy in %. Intervals are 95% t-intervals over seeds unless marked as a bootstrap.

## Verdict

**FALSIFIED via F2.** The spec names this mixed outcome **"LMSR beats TV, but budgets lose to known fixes."**

- **Recorded prediction: FALSIFIED via F2, about 60%.** That is what happened.
- **The prediction erred in two ways:**
  - It expected the LMSR shape to give "only a small accuracy gain". The LMSR shape does beat TV clearly, but only because it costs less.
  - It expected M-cap to fail by the velocity law. The pilot did not show that (see Secondary).
- **One gate failed.** The sensitivity gate needs a recall-family model to lose at least 10 points from 10x to 1000x; none does, so there is no overthinking in the recall family at this scale.
  - The spec's text for this gate says stability claims are then moot, but the primary endpoint is still reported.
  - A stricter reading, "any failed gate is INCONCLUSIVE", would label the run INCONCLUSIVE. That would not change the answer to question 2: the bars' margins are 58 vs 21 on E2 and 44 vs 1 on E3.

## Answers

**1. Does the LMSR shape of the budget beat the shape-free TV budget? Yes, but only in the sense that it hurts less.**

| comparison | result | criterion |
|---|---|---|
| M-budget minus M-TV, mean of E1-E3 | +7.9 [+5.0, +10.9], 5 of 5 seeds | S-A, part 1: holds |
| LMSR tracker minus TV tracker on R-prog rollouts | +8.2 [+1.5, +15.0], hierarchical bootstrap | S-A, part 2: holds |
| Late correction, M-budget minus M-TV (3.5% vs 0.4%) | +3.1 [+0.7, +5.5], 5 of 5 | S-C: holds |

Neither budget is a gain, though:

- M-budget scores the same as its uncapped twin M-B: -1.6 [-9.0, +5.9], 3 of 5 seeds.
- M-TV costs about 9 points relative to M-B.
- On R-prog rollouts, both test-time trackers score below the plain inst readout:
  - inst: 53.9
  - LMSR tracker: 39.2
  - TV tracker: 31.0
- DT-L and R-final show the same order: inst ≥ LMSR ≥ TV.

The market shape is therefore the less harmful of the two spending rules. It is not a better solver.

**2. Does the closed-loop budget solver extrapolate better than R-prog and DT-L? No.**

| arm | mean of E1-E3 | E3 (59x59 at 100x) |
|---|---|---|
| R-prog, all 5 seeds | 53.9 | 44.4 |
| R-prog, the 3 seeds that trained | 89.8 | |
| DT-L | 44.4 | 8.4 |
| M-B | 32.6 | 2.5 |
| M-budget | 31.0 | 0.9 |
| M-TV | 23.1 | 0.0 |

- F2 holds on all three cells.
- DT-L beats M-budget on the mean in 5 of 5 seeds: -13.4 [-28.9, +2.1].
- On seeds where both trained, R-prog beats M-budget by 58.0 points [45.5, 70.5], 3 of 3 seeds.

## Why the budget arms fail (mechanism)

**The M arms stop computing within a few multiples of T_train.** Their accuracy at 3x equals their accuracy at 1000x (see accuracy vs depth). R-prog keeps propagating the path: on 59x59 it goes 0, 2.5, 22.5, 42.2, 44.4 from 1x to 100x. The M arms stay flat at 0-3%.

**Budgets are always spent.**

- In both budget arms, 98-99% of on-path pixels exhaust their budget; the mean remaining budget is under 0.2%.
- On 29x29, 38% (M-budget) and 36% (M-TV) of on-path pixels run out while still wrong. On 59x59 it is 53% and 47%.
- A budget that is always fully spent acts as a hard stop.
- Late correction, meaning a confidently wrong pixel after T_train that ends up right, is therefore rare:

| arm | late correction |
|---|---|
| M-TV | 0.4% |
| M-budget | 3.5% |
| M-B | 4.8% |
| DT-L | 46% |
| R-prog | 76% |

**The uncapped twin fails too, so the budget is not the main cause.** M-B reaches only 2.5% on E3, with max |b| = 6e5 at 1000x. The main cause is the closed-loop accumulating-belief design itself: deep supervision with a stop-gradient, fed back as tanh(b/4). The budget adds a hard stop on top. The LMSR shape leaves accuracy unchanged, while TV lowers it.

**Adaptive spending fades on larger mazes.** The measure is the Spearman correlation between the step at which a pixel has spent 50% of its budget and its BFS distance to the nearer endpoint.

| size | M-budget | M-TV |
|---|---|---|
| 17 | 0.45 | 0.39 |
| 29 | 0.23 | 0.23 |
| 59 | 0.08 | 0.08 |

Spending tracks the path on small mazes and loses that structure on large ones.

**Readout choice does not rescue the bars' competitors.**

- R-prog's best validated readout is inst.
- DT-L's is freeze (tau 0.3, k 3), at +3.0 [+1.2, +4.7] over inst.
- R-final's is "stop at 110 iterations". A fixed stop does not scale to larger mazes: on E3 it gives 4.7 against 38.0 for inst.
- No arm's best validated readout is a budget tracker (R-prog inst, DT-L freeze, R-final stop), so the "inference-time effect only" outcome does not arise.

## Gates

| gate | result |
|---|---|
| Training (pilot, seed 0) | R-prog 100% on size 9 at T_train: pass |
| Floor | pass (val17 peak 100, val29 peak 97.7) |
| Endpoint floor | pass on all three cells |
| Sensitivity | FAIL. R-final inst and M-B lose 0 points from 10x to 1000x on sizes 9 and 17. |
| Headroom | R-prog best readout at 59 / 58 / 44 < 95, so S3 ran |

In the core, 2 of 5 R-prog seeds never learned the task:

- seed 3 stuck at loss ln 2;
- seed 1 at loss 0.54, train exact 0.0004.

Bear et al. 2024 report the same unreliability for width-32 recall. These failures lower a bar, so they bias against F2, and F2 holds on every cell anyway. All seeds are kept in the pre-registered numbers; a trained-seed table is given below as post hoc.

## Secondary arms (S3)

**R-final (a = 0), 3 seeds.**

- Seed 2 trained, seed 1 partly (68.8% on size 9 at 1x), seed 3 never.
- On its inst readout, seed 2 scores 100 / 100 / 100 on E1 / E2 / E3: perfect on 59x59 at 100x, without progressive loss. Seed 1 scores 78 / 56 / 14.
- Its validated "stop at 110" readout cuts that to E1 100, E2 75, E3 12.5.
- R-final's mean inst accuracy does not drop from 10x to 1000x on size 9 or 17, which agrees with the failed sensitivity gate. In this run the progressive loss was not needed against overthinking. What limits width-32 recall here is whether a seed trains at all: 3 of 8 recall seeds never learned, and one more learned only partly.

**M-cap and M-clip were dropped for budget**, as the spec's "drop from the bottom" allows.

- The one M-cap seed that finished training (seed 1, eps 0.3) diverged: the loss stuck at about 1.35, then became NaN at step 1200. It is not analysed. Its train.json is kept under results/vm/secondary/.

**Velocity law (pilot M-cap models on size 9 to 1000x): not confirmed.**

- eps 0.1 never learned the task.
- eps 0.3 and 1.0 hold 100% out to 1000x, with zero flips after 3x.
- The beliefs never need to flip, so the ln2/eps bound never binds at size 9.

## Scope

- Wealth here is only spent, never earned: labels are unseen at inference.
- P11 therefore does not bear on THEORY.md's programme falsifier, or on the 09-29 hidden-variable questions.
- The claims are "versus our width-32 reimplementation", not versus Bansal et al.'s width-128 numbers.

## Pre-registered analysis (every seed; generated by `tables.py` from `results/summary.json`)

### Endpoints (exact-maze accuracy, %, mean ± 95% t-interval over seeds)

| arm | scored readout | E1 17@1000x | E2 29@100x | E3 59@100x | mean | size 9 @1x |
|---|---|---|---|---|---|---|
| R-prog | inst | 59.2 ± 67.1 | 58.0 ± 65.7 | 44.4 ± 52.2 | 53.9 ± 61.2 | 60.0 ± 68.0 |
| R-prog | lmsr48 (eval-only) | 56.9 ± 64.6 | 36.9 ± 58.1 | 23.8 ± 41.2 | 39.2 ± 51.1 | |
| R-prog | tv64 (eval-only) | 52.2 ± 59.8 | 29.5 ± 49.8 | 11.2 ± 20.8 | 31.0 ± 40.5 | |
| DT-L | frz0.3_3 | 79.1 ± 11.5 | 45.6 ± 24.2 | 8.4 ± 12.7 | 44.4 ± 15.7 | 99.7 ± 0.9 |
| DT-L | inst (eval-only) | 74.1 ± 16.1 | 41.7 ± 25.9 | 8.4 ± 11.5 | 41.4 ± 17.6 | |
| DT-L | lmsr48 (eval-only) | 73.4 ± 17.1 | 40.8 ± 27.7 | 7.8 ± 11.1 | 40.7 ± 18.4 | |
| DT-L | tv64 (eval-only) | 72.2 ± 17.0 | 37.7 ± 26.4 | 7.2 ± 9.7 | 39.0 ± 17.5 | |
| M-B | inst | 71.2 ± 12.0 | 23.9 ± 8.5 | 2.5 ± 2.2 | 32.6 ± 6.9 | 99.7 ± 0.9 |
| M-budget | inst | 70.6 ± 2.0 | 21.4 ± 4.3 | 0.9 ± 1.7 | 31.0 ± 1.4 | 100.0 ± 0.0 |
| M-TV | inst | 56.2 ± 8.0 | 13.0 ± 5.3 | 0.0 ± 0.0 | 23.1 ± 3.5 | 100.0 ± 0.0 |
| R-final | stop110 | 62.0 ± 134.5 | 40.1 ± 93.8 | 4.7 ± 16.9 | 35.6 ± 79.9 | 54.2 ± 120.6 |
| R-final | inst (eval-only) | 59.4 ± 130.6 | 51.8 ± 124.5 | 38.0 ± 134.5 | 49.7 ± 124.2 | |
| R-final | lmsr48 (eval-only) | 60.7 ± 132.4 | 42.4 ± 127.0 | 33.3 ± 140.1 | 45.5 ± 124.5 | |
| R-final | tv64 (eval-only) | 52.3 ± 120.1 | 31.5 ± 106.8 | 10.9 ± 43.7 | 31.6 ± 86.6 | |

### Accuracy vs depth (%, scored readout; ratio = iterations / 30)

test9:

| arm | 1x | 3x | 10x | 30x | 100x | 300x | 1000x |
|---|---|---|---|---|---|---|---|
| R-prog (inst) | 60.0 | 60.0 | 60.0 | 60.0 | 60.0 | 60.0 | 60.0 |
| DT-L (frz0.3_3) | 99.7 | 98.8 | 98.8 | 98.8 | 98.8 | 98.8 | 98.8 |
| M-B (inst) | 99.7 | 99.7 | 99.4 | 99.4 | 99.4 | 99.4 | 99.4 |
| M-budget (inst) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.7 |
| M-TV (inst) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| R-final (stop110) | 54.2 | 62.0 | 62.5 | 62.5 | 62.5 | 62.5 | 62.5 |

test17:

| arm | 1x | 3x | 10x | 30x | 100x | 300x | 1000x |
|---|---|---|---|---|---|---|---|
| R-prog (inst) | 22.8 | 59.1 | 59.4 | 59.4 | 59.4 | 59.2 | 59.2 |
| DT-L (frz0.3_3) | 24.5 | 76.7 | 78.9 | 79.1 | 79.1 | 79.1 | 79.1 |
| M-B (inst) | 64.5 | 69.8 | 70.6 | 71.1 | 71.2 | 71.2 | 71.2 |
| M-budget (inst) | 69.7 | 70.9 | 70.5 | 70.6 | 70.6 | 70.6 | 70.6 |
| M-TV (inst) | 56.2 | 56.6 | 56.2 | 56.2 | 56.2 | 56.2 | 56.2 |
| R-final (stop110) | 19.5 | 58.1 | 62.0 | 62.0 | 62.0 | 62.0 | 62.0 |

test29:

| arm | 1x | 3x | 10x | 30x | 100x |
|---|---|---|---|---|---|
| R-prog (inst) | 5.3 | 31.4 | 57.3 | 57.5 | 58.0 |
| DT-L (frz0.3_3) | 1.9 | 26.4 | 45.2 | 45.6 | 45.6 |
| M-B (inst) | 18.9 | 22.3 | 23.1 | 23.9 | 23.9 |
| M-budget (inst) | 20.0 | 21.6 | 21.4 | 21.4 | 21.4 |
| M-TV (inst) | 12.8 | 13.0 | 13.0 | 13.0 | 13.0 |
| R-final (stop110) | 1.6 | 27.3 | 40.1 | 40.1 | 40.1 |

test59:

| arm | 1x | 3x | 10x | 30x | 100x |
|---|---|---|---|---|---|
| R-prog (inst) | 0.0 | 2.5 | 22.5 | 42.2 | 44.4 |
| DT-L (frz0.3_3) | 0.0 | 0.6 | 4.1 | 8.4 | 8.4 |
| M-B (inst) | 1.9 | 2.5 | 2.8 | 2.8 | 2.5 |
| M-budget (inst) | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| M-TV (inst) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| R-final (stop110) | 0.0 | 2.6 | 4.7 | 4.7 | 4.7 |

### Paired by seed (M-budget minus other, points; t-interval; sign count)

| comparison | E1 | E2 | E3 | mean |
|---|---|---|---|---|
| M-budget - M-TV | +14.4 [+7.3, +21.5], 5+/0- of 5 | +8.4 [+0.5, +16.4], 4+/1- of 5 | +0.9 [-0.8, +2.7], 2+/0- of 5 | +7.9 [+5.0, +10.9], 5+/0- of 5 |
| M-budget - R-prog | +11.4 [-54.1, +76.9], 2+/3- of 5 | -36.6 [-99.3, +26.2], 2+/3- of 5 | -43.4 [-96.5, +9.6], 1+/3- of 5 | -22.9 [-82.7, +37.0], 2+/3- of 5 |
| M-budget - DT-L | -8.4 [-20.6, +3.7], 0+/5- of 5 | -24.2 [-45.7, -2.8], 0+/5- of 5 | -7.5 [-21.4, +6.4], 2+/3- of 5 | -13.4 [-28.9, +2.1], 0+/5- of 5 |
| M-budget - M-B | -0.6 [-12.5, +11.2], 3+/2- of 5 | -2.5 [-13.8, +8.8], 2+/3- of 5 | -1.6 [-2.9, -0.2], 0+/4- of 5 | -1.6 [-9.0, +5.9], 3+/2- of 5 |
| in_dist: M-budget - R-prog | | | | +40.0 [-28.0, +108.0], 2+/0- of 5 |
| late_correction: M-budget - M-TV | | | | +3.1 [+0.7, +5.5], 5+/0- of 5 |

### Readouts within a model (hierarchical bootstrap over seeds and mazes, mean over E1-E3, points)

- R-prog: lmsr - tv tracker: +8.2 [+1.5, +15.0]
- R-prog: best - inst: +0.0 [+0.0, +0.0]
- DT-L: lmsr - tv tracker: +1.7 [+0.7, +2.7]
- DT-L: best - inst: +3.0 [+1.2, +4.7]
- R-final: lmsr - tv tracker: +13.9 [+0.0, +29.7]
- R-final: best - inst: -14.1 [-37.1, +0.0]

### Mechanism

| arm | late correction rate | persistence 10x->1000x test17 | path-indep. spread | rel. state dist |
|---|---|---|---|---|
| R-prog | 76.2 ± 53.5 | 59.8 ± 67.8 | 1.9 ± 3.5 | 10.79 ± 23.40 |
| DT-L | 45.6 ± 15.4 | 100.0 ± 0.0 | 9.4 ± 5.5 | 13.23 ± 9.93 |
| M-B | 4.8 ± 4.2 | 100.0 ± 0.0 | 11.9 ± 5.1 | 48.42 ± 18.24 |
| M-budget | 3.5 ± 2.5 | 100.0 ± 0.0 | 10.6 ± 7.6 | 56.93 ± 9.27 |
| M-TV | 0.4 ± 0.3 | 100.0 ± 0.0 | 0.0 ± 0.0 | 74.98 ± 10.86 |
| R-final | 63.3 ± 122.5 | 66.7 ± 143.4 | 0.0 ± 0.0 | 63.60 ± 137.24 |

- budget R-prog:lmsr48 test17: exhausted on path 25.4 ± 50.3, exhausted while wrong 16.5 ± 45.6, mean remaining 37.3 ± 41.8
- budget R-prog:lmsr48 test29: exhausted on path 18.2 ± 30.1, exhausted while wrong 12.1 ± 29.0, mean remaining 39.7 ± 26.1
- budget R-prog:lmsr48 test59: exhausted on path 32.9 ± 39.4, exhausted while wrong 20.1 ± 28.8, mean remaining 31.7 ± 29.8
- budget R-prog:tv64 test17: exhausted on path 27.4 ± 51.2, exhausted while wrong 17.1 ± 46.3, mean remaining 36.2 ± 42.9
- budget R-prog:tv64 test29: exhausted on path 23.2 ± 36.1, exhausted while wrong 15.4 ± 34.0, mean remaining 35.5 ± 29.3
- budget R-prog:tv64 test59: exhausted on path 37.0 ± 44.8, exhausted while wrong 23.5 ± 33.3, mean remaining 27.0 ± 29.5
- budget DT-L:lmsr48 test17: exhausted on path 23.7 ± 46.0, exhausted while wrong 6.1 ± 11.0, mean remaining 46.6 ± 26.9
- budget DT-L:lmsr48 test29: exhausted on path 11.1 ± 22.1, exhausted while wrong 2.7 ± 2.6, mean remaining 47.0 ± 16.8
- budget DT-L:lmsr48 test59: exhausted on path 11.0 ± 22.1, exhausted while wrong 2.7 ± 3.2, mean remaining 42.3 ± 14.2
- budget DT-L:tv64 test17: exhausted on path 26.2 ± 47.4, exhausted while wrong 7.5 ± 12.8, mean remaining 44.4 ± 27.5
- budget DT-L:tv64 test29: exhausted on path 16.5 ± 28.1, exhausted while wrong 5.3 ± 4.9, mean remaining 40.1 ± 17.6
- budget DT-L:tv64 test59: exhausted on path 16.9 ± 27.9, exhausted while wrong 5.7 ± 6.0, mean remaining 32.1 ± 12.6
- budget M-budget:inst test17: exhausted on path 99.0 ± 0.3, exhausted while wrong 12.9 ± 2.6, mean remaining 0.0 ± 0.0
- budget M-budget:inst test29: exhausted on path 98.3 ± 0.5, exhausted while wrong 38.0 ± 8.7, mean remaining 0.1 ± 0.1
- budget M-budget:inst test59: exhausted on path 98.6 ± 0.5, exhausted while wrong 52.6 ± 15.8, mean remaining 0.1 ± 0.1
- budget M-TV:inst test17: exhausted on path 98.3 ± 1.5, exhausted while wrong 13.7 ± 4.1, mean remaining 0.1 ± 0.1
- budget M-TV:inst test29: exhausted on path 97.6 ± 2.3, exhausted while wrong 35.7 ± 12.4, mean remaining 0.1 ± 0.1
- budget M-TV:inst test59: exhausted on path 97.6 ± 2.1, exhausted while wrong 46.7 ± 19.1, mean remaining 0.1 ± 0.1
- budget R-final:lmsr48 test17: exhausted on path 66.1 ± 139.8, exhausted while wrong 0.7 ± 2.4, mean remaining 16.6 ± 70.4
- budget R-final:lmsr48 test29: exhausted on path 62.0 ± 134.5, exhausted while wrong 2.3 ± 9.2, mean remaining 16.9 ± 66.9
- budget R-final:lmsr48 test59: exhausted on path 63.2 ± 136.5, exhausted while wrong 5.4 ± 22.3, mean remaining 16.0 ± 65.0
- budget R-final:tv64 test17: exhausted on path 66.5 ± 139.9, exhausted while wrong 0.7 ± 2.4, mean remaining 15.3 ± 65.0
- budget R-final:tv64 test29: exhausted on path 63.2 ± 136.4, exhausted while wrong 2.2 ± 8.6, mean remaining 14.9 ± 60.0
- budget R-final:tv64 test59: exhausted on path 64.2 ± 138.1, exhausted while wrong 4.5 ± 18.5, mean remaining 13.7 ± 56.4

### Gates

- training: {"value": 1.0, "threshold": 0.9, "pass": true}
- floor: {"val17_peak": 1.0, "val29_peak": 0.9765625, "thresholds": [0.5, 0.25], "pass": true}
- sensitivity: {"rows": {"R-final test9": {"10x": 0.6197916666666666, "1000x": 0.6197916666666666, "drop": 0.0}, "R-final test17": {"10x": 0.59375, "1000x": 0.59375, "drop": 0.0}, "M-B test9": {"10x": 0.99375, "1000x": 0.99375, "drop": 0.0}, "M-B test17": {"10x": 0.70625, "1000x": 0.7125, "drop": -0.006249999999999978}}, "pass": false}
- headroom: {"readout": "inst", "val_score": 0.596875, "n_seeds": 5, "endpoints": {"E1": 0.5921875, "E2": 0.5796875, "E3": 0.44375}, "skip_s3": false}
- endpoint floor (best arm < 20% = inconclusive): {"E1": {"best_arm": "DT-L", "best": 0.790625, "inconclusive": false}, "E2": {"best_arm": "R-prog", "best": 0.5796875, "inconclusive": false}, "E3": {"best_arm": "R-prog", "best": 0.44375, "inconclusive": false}}

### Verdict

{"S-A": true, "S-B": false, "S-C": true, "F1": false, "F2_cells": {"E1": {"M-budget": 0.70625, "best_bar": 0.790625, "bar_wins": true}, "E2": {"M-budget": 0.2140625, "best_bar": 0.5796875, "bar_wins": true}, "E3": {"M-budget": 0.009375, "best_bar": 0.44375, "bar_wins": true}}, "F2": true, "gates_ok": true, "above_floor": true, "verdict": "FALSIFIED (F2)", "mixed": ["LMSR beats TV, but budgets lose to known fixes"]}

## Post hoc: every seed, and trained seeds only (generated by `posthoc.py`; not pre-registered)

### Every seed (exact-maze accuracy, %; scored readout)

| arm | seed | test9 @1x inst | final train loss | E1 17@1000x | E2 29@100x | E3 59@100x | mean | trained |
|---|---|---|---|---|---|---|---|---|
| R-prog | 1 | 0.0 | 0.539 | 0.0 | 0.0 | 0.0 | 0.0 | NO |
| R-prog | 2 | 100.0 | 0.00208 | 99.2 | 96.9 | 92.2 | 96.1 | yes |
| R-prog | 3 | 0.0 | 0.693 | 0.0 | 0.0 | 0.0 | 0.0 | NO |
| R-prog | 4 | 100.0 | 0.00229 | 100.0 | 96.9 | 62.5 | 86.5 | yes |
| R-prog | 5 | 100.0 | 0.00171 | 96.9 | 96.1 | 67.2 | 86.7 | yes |
| DT-L | 1 | 100.0 | 0.00437 | 71.9 | 18.8 | 0.0 | 30.2 | yes |
| DT-L | 2 | 100.0 | 0.00437 | 85.2 | 53.9 | 9.4 | 49.5 | yes |
| DT-L | 3 | 100.0 | 0.00276 | 92.2 | 71.9 | 25.0 | 63.0 | yes |
| DT-L | 4 | 98.4 | 0.00318 | 71.1 | 39.1 | 0.0 | 36.7 | yes |
| DT-L | 5 | 100.0 | 0.00387 | 75.0 | 44.5 | 7.8 | 42.4 | yes |
| M-B | 1 | 98.4 | 0.0253 | 83.6 | 32.8 | 4.7 | 40.4 | yes |
| M-B | 2 | 100.0 | 0.026 | 78.9 | 28.9 | 0.0 | 35.9 | yes |
| M-B | 3 | 100.0 | 0.0281 | 63.3 | 18.8 | 3.1 | 28.4 | yes |
| M-B | 4 | 100.0 | 0.0298 | 61.7 | 16.4 | 3.1 | 27.1 | yes |
| M-B | 5 | 100.0 | 0.0256 | 68.8 | 22.7 | 1.6 | 31.0 | yes |
| M-budget | 1 | 100.0 | 0.0219 | 69.5 | 15.6 | 3.1 | 29.4 | yes |
| M-budget | 2 | 100.0 | 0.0209 | 71.9 | 25.0 | 0.0 | 32.3 | yes |
| M-budget | 3 | 100.0 | 0.0233 | 68.8 | 21.9 | 0.0 | 30.2 | yes |
| M-budget | 4 | 100.0 | 0.0224 | 70.3 | 22.7 | 1.6 | 31.5 | yes |
| M-budget | 5 | 100.0 | 0.0209 | 72.7 | 21.9 | 0.0 | 31.5 | yes |
| M-TV | 1 | 100.0 | 0.0206 | 50.0 | 17.2 | 0.0 | 22.4 | yes |
| M-TV | 2 | 100.0 | 0.0203 | 63.3 | 14.1 | 0.0 | 25.8 | yes |
| M-TV | 3 | 100.0 | 0.0243 | 50.0 | 10.2 | 0.0 | 20.1 | yes |
| M-TV | 4 | 100.0 | 0.0209 | 62.5 | 16.4 | 0.0 | 26.3 | yes |
| M-TV | 5 | 100.0 | 0.0211 | 55.5 | 7.0 | 0.0 | 20.8 | yes |
| R-final | 1 | 68.8 | 0.0216 | 85.9 | 45.3 | 1.6 | 44.3 | NO |
| R-final | 2 | 93.8 | 4.46e-05 | 100.0 | 75.0 | 12.5 | 62.5 | yes |
| R-final | 3 | 0.0 | 0.796 | 0.0 | 0.0 | 0.0 | 0.0 | NO |

### Trained seeds only (post hoc; mean ± 95% t-interval)

| arm | trained / seeds | E1 | E2 | E3 | mean |
|---|---|---|---|---|---|
| R-prog | 3/5 | 98.7 ± 4.0 | 96.6 ± 1.1 | 74.0 ± 39.6 | 89.8 ± 13.6 |
| DT-L | 5/5 | 79.1 ± 11.5 | 45.6 ± 24.2 | 8.4 ± 12.7 | 44.4 ± 15.7 |
| M-B | 5/5 | 71.2 ± 12.0 | 23.9 ± 8.5 | 2.5 ± 2.2 | 32.6 ± 6.9 |
| M-budget | 5/5 | 70.6 ± 2.0 | 21.4 ± 4.3 | 0.9 ± 1.7 | 31.0 ± 1.4 |
| M-TV | 5/5 | 56.2 ± 8.0 | 13.0 ± 5.3 | 0.0 ± 0.0 | 23.1 ± 3.5 |
| R-final | 1/3 | 100.0 | 75.0 | 12.5 | 62.5 |

### Paired over seeds where both arms trained (M-budget minus other, points)

- M-budget - M-TV: mean +7.9 [+5.0, +10.9], 5+/0- of 5 (seeds [1, 2, 3, 4, 5]); E1 +14.4, E2 +8.4, E3 +0.9
- M-budget - M-B: mean -1.6 [-9.0, +5.9], 3+/2- of 5 (seeds [1, 2, 3, 4, 5]); E1 -0.6, E2 -2.5, E3 -1.6
- M-budget - R-prog: mean -58.0 [-70.5, -45.5], 0+/3- of 3 (seeds [2, 4, 5]); E1 -27.1, E2 -73.4, E3 -73.4
- M-budget - DT-L: mean -13.4 [-28.9, +2.1], 0+/5- of 5 (seeds [1, 2, 3, 4, 5]); E1 -8.4, E2 -24.2, E3 -7.5

## S1 pilot (seed 0, validation only; selection = mean val17 exact at 10x and 100x)

| arm | hp | chosen | val17 score (10x,100x) | readout | val17 inst 1x / 10x / 100x | val29 peak inst | val9 1x | sat. pixel-steps | train s |
|---|---|---|---|---|---|---|---|---|---|
| R-prog | 0.01 |  | 100.0 | inst | 29 / 100 / 100 | 99.2 | 100 |  | 1883 |
| R-prog | 0.1 | yes | 100.0 | inst | 50 / 100 / 100 | 97.7 | 100 |  | 1883 |
| R-prog | 0.5 |  | 96.9 | inst | 59 / 97 / 97 | 82.8 | 100 |  | 1882 |
| R-prog | 1.0 |  | 98.0 | ces | 62 / 97 / 98 | 76.6 | 100 |  | 573 |
| DT-L | 0.01 |  | 53.1 | frz1.0_3 | 10 / 18 / 15 | 2.3 | 98 |  | 1778 |
| DT-L | 0.1 | yes | 75.0 | frz0.3_3 | 14 / 68 / 67 | 21.9 | 100 |  | 1760 |
| M-budget | 3 |  | 7.0 | inst | 7 / 7 / 7 | 0.0 | 83 | 10.0% | 1949 |
| M-budget | 6 |  | 30.5 | inst | 30 / 30 / 30 | 0.8 | 100 | 0.5% | 1955 |
| M-budget | 12 |  | 60.9 | inst | 61 / 61 / 61 | 12.5 | 100 | 0.0% | 1955 |
| M-budget | 24 | yes | 83.6 | inst | 78 / 84 / 84 | 23.4 | 100 | 0.0% | 1954 |
| M-TV | 4 |  | 0.0 | inst | 0 / 0 / 0 | 0.0 | 48 | 46.2% | 1817 |
| M-TV | 8 |  | 7.0 | inst | 7 / 7 / 7 | 0.0 | 94 | 0.2% | 1822 |
| M-TV | 16 |  | 39.1 | inst | 39 / 39 / 39 | 1.6 | 100 | 0.0% | 1820 |
| M-TV | 32 | yes | 62.5 | inst | 64 / 62 / 62 | 11.7 | 100 | 0.0% | 1823 |
| M-cap | 0.1 |  | 0.0 | inst | 0 / 0 / 0 | 0.0 | 0 | 2.3% | 1339 |
| M-cap | 0.3 | yes | 73.4 | inst | 66 / 73 / 73 | 22.7 | 100 | 92.8% | 1409 |
| M-cap | 1.0 |  | 73.4 | inst | 70 / 73 / 73 | 21.9 | 100 | 71.3% | 1402 |
| M-clip | 0.1 |  | 0.0 | inst | 0 / 0 / 0 | 0.0 | 0 | 1.8% | 1823 |
| M-clip | 0.3 |  | 60.2 | inst | 24 / 58 / 62 | 15.6 | 100 | 69.2% | 1674 |
| M-clip | 1.0 | yes | 69.9 | inst | 60 / 70 / 70 | 22.7 | 100 | 73.9% | 1632 |
| M-clip | 3.0 |  | 58.2 | inst | 53 / 58 / 59 | 19.5 | 100 | 0.1% | 1508 |
| M-B | None | yes | 58.2 | inst | 55 / 59 / 58 | 17.2 | 100 | 13.1% | 1778 |
| R-final | None | yes | 100.0 | inst | 34 / 100 / 100 | 100.0 | 97 |  | 1369 |

## Prior work and honest novelty

Known results that P11 does not retest:

- Bansal, Schwarzschild et al. 2022 (arXiv 2202.05826) introduced recall and the progressive loss. R-prog is our width-32 reimplementation of that recipe. They report 97% on 59x59 and 74% on 201x201 at width 128.
- Bear, Pruegel-Bennett and Hare 2024 (arXiv 2410.23451), DT-L: a spectral-norm contractive recurrence. Our DT-L ports P6's 1-D `DTL` class to Conv2d.
- Anil et al. 2022 (arXiv 2211.09961), path independence.
- Schwarzschild et al. 2021 (arXiv 2106.04537, 2108.06011).
- Knutson et al. 2024 (2410.03020): maze DT nets settle into limit cycles, not fixed points.
- NeuralThink (2402.15393).
- Learning to Stop Overthinking at Test Time (2502.10954).
- Other related ideas:
  - the PPO ratio clip (1707.06347);
  - bounded-loss market makers (Chen and Pennock 2007, 1206.5252);
  - damped fixed-point iteration and damped loopy BP;
  - ACT and PonderNet halting;
  - deeply-supervised nets (Lee et al. 2015, 1409.5185). The M arms' per-step MSR pay telescopes, so it is deep supervision with a stop-gradient.
  - Haber and Ruthotto 2017 (1705.03341);
  - Miller and Hardt 2019 (1805.10369).

What is new here is narrow. It is one output-logit clip function: the LMSR / D-infinity worst-case-loss bound, used as a per-pixel spending budget that the network can observe. P11 tests it against:

- a logit total-variation budget of the same form (its shape-free twin);
- a logit clip;
- freeze, early-stop, running-mean and EMA readouts;
- DT-L.

If it does not beat its TV twin, the market framing adds nothing.

## Deviations from the spec (all applied to every arm)

- **Location.** The code lives in `puzzles/P11/`, as the task asked, not in `p11-wealthcap/` as the spec said.
- **Compute.**
  - Hardware: spot L4 VMs (g2-standard-4) in project GCP_PROJECT, running 4 runs at a time per GPU.
  - Scheduling: a sharded task queue (`queue_p11.py`), with 2 VMs for the pilot and 3 for the core.
  - Preemption handling: spot preemptions were frequent. `gpu.sh keep` re-creates a lost shard VM, and `gpu.sh relay` copies results between VMs every 2 min. Training writes a full-state checkpoint every 400 steps and resumes from it with the same batches and draws.
  - Records: every VM is logged in `gpu/LEDGER.md` and `../../GPU-LEDGER.md`.
  - Spend: the watchdog counts spend at an assumed spot rate of $0.45/h, against a list-price ceiling of $0.71/h. The ledger reports both.
- **Vendored sources.** P7/maze.py and P6/p6_entry.py had changed since the spec pinned their hashes. `VENDORED.json` records the hashes of the versions actually copied.
- **Training data.** Training mazes come from a fixed pool of 400k seeded size-9 mazes (seeds [0, 4e5)), not a fresh maze every step. Each run draws its own permutation, so no maze repeats within a run: 6000 x 64 = 384k draws.
- **Seeds.** Core seeds are 1-5 and fresh. The pilot seed 0 is not reused.
- **DT-L pilot.** Only 2 values of a are tried: the R-prog pilot winner and its better neighbour, as the queue's `dtl_grid` step chose them. These were a in {0.1, 0.01}.
- **Floor gate.** The gate reads R-prog's peak inst accuracy over the validation checkpoints up to 100x.
- **Readout choice.** Each readout is chosen on the validation mean across the core seeds, then scored on test.
- **Late correction.** The late-correction rate is pooled over test17, test29 and test59.
- **Path independence.** It is measured on 32 test17 mazes, with 4 random initial states.
- **Freeze readout.** It freezes per maze, not per pixel.
- **No width-128 anchor.** Every claim is "versus our width-32 reimplementation", not versus the published numbers.

## Further deviations during the run

- **Spot preemptions.** 7 VMs were preempted; `gpu.sh keep` re-created each one. Training resumes from its 400-step checkpoint, but an evaluation restarts from scratch, so every preemption lost the evaluation work in progress.
- **Relay bug.** The relay's rsync once overwrote newer shard-0 checkpoints with older copies, so shard 0's runs resumed from step 400. It was fixed with `rsync --update`. Resumed runs list the resume steps in `train.json` (`resumed_at`).
- **Cap check.** `gpu.sh up` checked the cap at the $0.71/h ceiling while the watchdog used the $0.45/h watch rate. `up` was changed to use the watch rate as well, as the ledger header records.
- **Secondary arms.** R-final ran with 3 seeds. M-cap and M-clip were dropped for budget, and their partial training (killed when shards 1 and 2 were retired) is not analysed.
- **Pilot grid edges.** The pilot picked M-budget W = 24 and M-TV V = 32, both at the top edge of their grids. Larger budgets might score higher on validation. Given that 98-99% of the budget is spent, a larger budget would most likely just move the hard stop later. That is untested.
- **Post-hoc table.** The trained-seed table (`posthoc.py`) is not pre-registered. Its rule is: a seed counts as trained if its test9 accuracy at 1x (inst) is at least 90%.
- **Endpoint noise.** R-prog's test9 inst curve is flat at 60% because 2 of its 5 seeds score 0. The pre-registered R-prog intervals are therefore very wide.

## Cost and teardown

- **VM time:** 15.13 VM-hours of spot g2-standard-4 + 1x L4 across 10 VMs, from 13:02 to 19:46Z.
- **Cost:**
  - Watch estimate (the rate the cap was checked at from 14:10Z): **$6.80** at $0.45/h.
  - On-demand list ceiling: **$10.74** at $0.71/h. **That is above the $8 cap.** The billed spot price could not be read, because the Cloud Billing API is disabled in GCP_PROJECT.
- **Per run (L4 shared 4 ways):**
  - Training: 1340-2020 s.
  - Evaluation: 1300-2470 s, all readouts from one rollout.
  - Parameters: 43.6k (DT-L), 53.6k (R) and 54.2k (M).
- **Records:** every VM is in `gpu/LEDGER.md` and `../../GPU-LEDGER.md`.
- **Teardown check at 19:46Z:** `gcloud --project=GCP_PROJECT compute instances list --filter="name~^trader-cat-p11"` and `gcloud --project=GCP_PROJECT compute disks list --filter="name~^trader-cat-p11"` both returned nothing.

## Files

- **Code:** `gen.py` (mazes), `caps.py` (LMSR, TV and clip caps), `model.py`, `train.py`, `eval.py`, `queue_p11.py` (sharded task queue), `select_p11.py` (pilot selection and gates), `s0.py` (S0 tests), `startup.sh` (VM boot), `vendored.py` and `vendored_upstream.py`, `gpu.sh` (VMs, relay, watchdog and keep), `analyze.py`, `tables.py` and `posthoc.py`.
- **Results:**
  - `results/summary.json`: every pre-registered quantity, gate and the verdict.
  - `results/chart.json`: accuracy vs depth per size, arm and readout, plus the velocity law.
  - `results/posthoc.json`
  - `results/vm/`: per-run `train.json` and `eval_core.json`, pilot, params and gates.
- **Rebuild the analysis:** run `~/.venvs/trader-cat/bin/python analyze.py && ~/.venvs/trader-cat/bin/python tables.py && ~/.venvs/trader-cat/bin/python posthoc.py`.
