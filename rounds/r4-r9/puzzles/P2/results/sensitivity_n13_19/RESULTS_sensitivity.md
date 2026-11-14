# P2: tied maze paths, quadtree latent-parent market vs commitment baselines

Status (computed from the pre-registered rules): **Problem**. Primary set: tie mazes, dead-end start, p in {0.05, 0.1}, n in [13, 19] (floor rule), T = 300, 600 mazes; seeds {'U': [0, 1, 2], 'U-canon': [0, 1, 2, 3, 4], 'U-min': [0, 1, 2, 3, 4], 'U-any': [0, 1, 2], 'M-tree': [0, 1, 2, 3, 4], 'M-tile': [0, 1, 2]}. Intervals: 95% t over paired seeds; [b ...] = nested bootstrap.

**SENSITIVITY RUN: primary sizes forced to [13, 19]; the pre-registered floor rule gives [13]. Not a pre-registered result.**

| arm | tie VSP % (pre-named decoder) | no-tie VSP % |
|---|---|---|
| U (cl) | 20.6 [18.5, 22.7] | 62.2 [61.4, 63.1] |
| U-canon (cl) | 31.6 [26.1, 37.1] | 63.1 [56.7, 69.4] |
| U-min (cl) | 43.3 [38.0, 48.5] | 66.7 [61.0, 72.4] |
| U-any (cl) | 41.7 [23.8, 59.5] | 67.3 [47.9, 86.6] |
| M-tree (tv) | 28.4 [9.8, 47.0] | 52.6 [31.7, 73.4] |
| M-tile (tv) | 33.6 [4.7, 62.4] | 70.4 [34.6, 106.2] |

| pre-registered test | estimate (points) | pass |
|---|---|---|
| (a) M-tree-tv - U-canon-cl >= 8, lower > 0 | -3.2 [-24.9, 18.4] [b -14.4, 11.9] | False |
| (b) DiD vs U-canon >= 5, lower > 0 | -4.0 [-9.7, 1.7]; OLS tie coef 6.4 [-4.6, 17.3] | False |
| (a) M-tree-tv - U-min-cl >= 8, lower > 0 | -14.9 [-32.0, 2.2] [b -24.6, -3.4] | False |
| (b) DiD vs U-min >= 5, lower > 0 | -13.0 [-18.6, -7.4]; OLS tie coef -1.6 [-10.7, 7.5] | False |
| (c) p=0: U-canon - M-tree upper <= 2 | 26.5 [8.2, 44.9] | False |
| LATENT I/H >= 0.8 | 0.284 [0.203, 0.366] (pooled 0.232 [0.158, 0.307]) | False |
| falsifier 1_upper_M_minus_maxbase_lt_3 | -14.9 [-32.0, 2.2] | falsified=True |
| falsifier 2_DiD_le_0 |  | falsified=True |
| falsifier 3_Mtree_le_Mtile_plus_3_gt32 | -3.2 [-40.2, 33.7] | falsified=True |
| falsifier 4_wins_but_code_lt_30pct |  | falsified=False |
| secondary M-tree - M-tile, >32 px | -3.2 [-40.2, 33.7] p_holm=1 | reject=False |
| secondary U-any - U-min, primary tie | -0.8 [-32.6, 31.0] p_holm=1 | reject=False |
| secondary slope U-canon - slope U-min | 0.91 [-0.12, 1.94] p_holm=0.282 | reject=False |
| secondary slope U-canon - slope M-tree | -0.23 [-3.13, 2.67] p_holm=1 | reject=False |
| partial: M-tree - U >= 10 | 12.6 [-34.6, 59.8] | False |

Validity: unit_tests=True; unit_tests_general=None; trap=False; toy=False; gate=True; floor_sizes=True; no_collapse=True; gpu_util_pilot=True

Simplifications against the spec: the cl decoder is binary (<= 5 local sweeps, no 0.5 damping); beliefs and codes are soft-clamped at +-30 logits; the toy uses a generic 1D trunk; eval recomputes BFS distances and asserts the stored DAG; M-tree has 1.7% more parameters than U (108.5k vs 106.7k); code-carries labels a sample's branch by the contested pixels of the best shortest path inside the sample; joint log loss is averaged over 4 uniformly drawn shortest paths per maze; ECE targets the uniform-path marginal; M health comes from the same 256 samples (leaf and root nodes); total correlation is jll_fact - jll on the same labels (a sample estimate is kept as tc_sample); FLOPs are estimated from parameter counts. The dev split holds no-tie mazes only, as specified.

## Run notes (written after the run)

- **Why the status is Problem.** The spec says a failed validity condition is "reported as a problem, not as evidence". Three conditions failed:
  - Toy: M-tile scored 6.2% at n=64 against a limit of 5% (seeds 12.0, 5.7 and 0.8%) and 1.9% at n=128. M-tree passed with 50.7% and 50.8%.
  - Floor rule: it passed only at n=13.
  - Trap test: it was cut short by preemption, so it counts as not passed.
- **Floor rule.** It was coded before any data came in, on the dev cells that match the primary set (dead-end start, p in {0.05, 0.1}).
  - U-canon no-tie VSP was 80.5, 43.9, 24.3 and 14.0% at n = 13, 19, 25 and 33, so only n=13 enters.
  - The other reading, all dev cells at a given n, gives 89, 59, 35 and 17%, which admits n = 13 and 19.
  - The sensitivity run on n in {13, 19} (`results/sensitivity_n13_19/`, not pre-registered) gives the same direction.
- **Trap test.** A spot preemption cut it at 26 of 30 cells.
  - Maximum VSP over the finished cells: 1.0%, which equals the limit, so those cells pass.
  - Missing cells: n=33, p in {0.05, 0.1}, both starts.
- **Spot preemptions.** Four VMs were preempted: trader-cat-p2-3 at 15:03, p2-4 at 15:35, p2-2 at 17:17 and p2-5 at 17:33 UTC.
  - Training resumed from partial checkpoints synced every 15 minutes with sha256 checks.
  - The resumed M-tree s0 run repeated the step-500 loss to 5 digits.
  - Seeds fix the initialisation, the data and the labels, so arms trained on different VMs are still paired.
  - Split: U-canon, U, U-any, M-tile and the trap ran on p2-2; M-tree, U-min and the final unit tests ran on p2-3, p2-4 and p2-5.
- **Eval coverage.** Evals ran in three parts: core, deep and extra.
  - Core covers the primary cells, the floor cells and p=0. It finished for all 24 runs.
  - Deep finished only for M-tree s0, U-min s0-1 and U-canon s0-3.
  - Extra covers the remaining dev and test cells and never ran.
  - No fifth VM was launched after the fourth preemption. The missing deep, extra and trap cells cannot change the pre-registered status, because the toy and the floor rule already fail. Depth results are therefore descriptive, with 1 to 4 seeds.
- **Fix during the run.** U-any training first crashed: the fast in-place DAG pass breaks autograd. The fix sends gradient calls to the reference DP, and a unit test now checks this. U-any then trained from scratch.
- **Unit tests.** All 21 passed, both on p2-2 and in the final run on p2-5.
- **Settings.** fp32 throughout. bf16 was 1.7x faster but moved beliefs by up to 0.09, so it was not used. Each model is about 420 KB.
- **GPU use.** Five spot L4 VMs ran 5.96 VM-hours, which is $4.48 at the $0.75/h ceiling (actual spot price is lower). All VMs and disks are deleted, confirmed with instances list and disks list.
  - The P2 section of GPU-LEDGER.md also holds two P11 rows, which another experiment added.

Details, strata, depth, calibration, M health and compute: `results/p2_results.json`. Charts: `results/charts.json`.

```json
[{"id":"p2_primary","type":"bar","title":"Tie mazes, primary set: valid shortest path at T=300 (pre-named decoders)","x_label":"arm","y_label":"VSP (%)","x":["U","U-canon","U-min","U-any","M-tree","M-tile"],"series":[{"name":"tie","y":[20.611111,31.6,43.266667,41.666667,28.366667,33.555556],"lo":[18.527241,26.094356,38.003612,23.849243,9.770817,4.733095],"hi":[22.694981,37.105644,48.529721,59.48409,46.962517,62.378016]},{"name":"matched cells, no tie","y":[62.222222,63.066667,66.7,67.277778,52.566667,70.388889]}]},{"id":"p2_vs_T","type":"line","title":"Tie VSP against iterations (primary set)","x_label":"T","y_label":"VSP (%)","x":[30,60,100,200,300],"series":[{"name":"U","y":[18.222222,20.388889,22.444444,21.611111,20.611111]},{"name":"U-canon","y":[27.6,30.666667,31.2,30.9,31.6]},{"name":"U-min","y":[35.933333,41.033333,42.3,42.466667,43.266667]},{"name":"U-any","y":[35.833333,40.111111,40.222222,41.5,41.666667]},{"name":"M-tree","y":[23.266667,26.833333,27.2,28.1,28.366667]},{"name":"M-tile","y":[26.611111,31.388889,32.555556,32.722222,33.555556]}]},{"id":"p2_strata","type":"bar","title":"Tie VSP by size of the contested region","x_label":"contested pixels","y_label":"VSP (%)","x":["contested <=8","contested 9-32","contested >32"],"series":[{"name":"U","y":[20.33543,22.996516,16.450216]},{"name":"U-canon","y":[35.849057,35.191638,20.519481]},{"name":"U-min","y":[46.666667,47.944251,31.038961]},{"name":"U-any","y":[47.379455,44.367015,30.735931]},{"name":"M-tree","y":[21.63522,31.498258,29.480519]},{"name":"M-tile","y":[27.463312,34.610918,37.878788]}]},{"id":"p2_per_n","type":"line","title":"Tie VSP against maze size (dead-end start, p in {0.05, 0.1})","x_label":"n","y_label":"VSP (%)","x":[13,19,25,33],"series":[{"name":"U","y":[33.444444,7.777778,2.555556,1.0]},{"name":"U-canon","y":[50.0,13.2,4.333333,2.266667]},{"name":"U-min","y":[64.4,22.133333,6.066667,3.266667]},{"name":"U-any","y":[62.111111,21.222222,7.333333,3.444444]},{"name":"M-tree","y":[46.066667,10.666667,6.066667,5.133333]},{"name":"M-tile","y":[43.333333,23.777778,11.444444,6.222222]}]},{"id":"p2_depth","type":"bar","title":"Depth: tie VSP at more iterations (decisive arms)","x_label":"cell","y_label":"VSP (%)","x":["n33_T300_first50","n33_T1000","n33_T3000_first50","n59_T300","n59_T1000"],"series":[{"name":"M-tree","y":[0.0,0.0,0.0,0.0,0.0]},{"name":"U-canon","y":[1.5,1.75,1.75,0.125,0.25]},{"name":"U-min","y":[5.0,5.0,4.5,0.25,0.5]}]},{"id":"p2_latent","type":"bar","title":"Share of the route entropy the M-tree code carries (I/H) per seed","x_label":"seed","y_label":"I/H","x":["s0","s1","s2","s3","s4"],"series":[{"name":"M-tree","y":[0.21543,0.38194,0.28841,0.30301,0.23292]}]},{"id":"p2_toy","type":"bar","title":"Stage-0 toy: exact match while U has not resolved","x_label":"n","y_label":"exact match (%)","x":["32","64","128"],"series":[{"name":"M-tree","y":[50.6,50.678334,50.781945]},{"name":"M-global","y":[47.908334,50.068888,49.163334]},{"name":"M-tile","y":[26.616667,6.172778,1.933889]}]}]
```
