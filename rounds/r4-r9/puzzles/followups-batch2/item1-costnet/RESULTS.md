# Item 1 results: CostNet+BB (learned local costs + exact blackbox solver)

Written 2026-09-29 ~23:41Z, corrected at 23:53Z after the independent verifier, from the synced raw outputs in `vmout/` by `analyze1.py`, which writes `results1.json`.
Updated 2026-09-30 ~02:50Z: the dt-baseline released-maze evals landed (e2h_maze_59 at 02:44Z; sha256 matches the lane's
`SHA256SUMS.vm`), `analyze1.py` was rerun unchanged, and the pre-registered 1b verdict is now computed: **worse**.
A second independent verifier (~03:00Z) recomputed the CostNet CI from the 20 per-seed test files and DT's peak, peak
iteration and final accuracy from the raw `acc` arrays, checked every sha256 and the shared maze slice, and found no mismatch.
Pre-registrations were frozen before any run: `PREREG.md` (1a, 23:14Z) and `PREREG-1b.md` (1b, 23:24Z).

## Status

**1a replicates.** P12's single-seed 27% was typical, not lucky. The only baseline it beats, though, is no
learning and P12's DT, and that DT failed its own validity gate.

**1b: worse (pre-registered primary).** On e2h_maze_59 (first 1,000 mazes), CostNet+BB solves 25.05% [23.56, 26.53]
exactly (20 seeds). The released DT-Recall checkpoint, evaluated in-house by the dt-baseline lane `trader-dt-p1`,
solves **100%** (peak at iteration 726 of T = 1000, and still 100% at the final iteration). The CI upper bound 26.53 is
far below DT − 5 = 95, so the verdict is "worse", by about 75 points. The DT validity gate passes (e2h_maze_13: 100%).
DT is also ahead on 33×33 (100% against 58.7%) and marginally on 13×13 (100% against 99.84%); secondaries have no bars.

**Paper context (not the pre-registered number).** Bansal et al. 2022, Table 4, report DT-Recall (α = 0.01) at
97.30 ± 0.68% peak on 59×59 mazes; their text (not the table) says 74% on 201×201. The in-house reproduction is at or
above the paper on every set evaluated. The 201×201 DT eval had not landed at 02:50Z; it is a secondary set and does
not change the verdict.

**The market framing adds nothing measurable.** This is Vlastelica et al. 2020's method (next section).

## Prior work (verified)

**Vlastelica, Paulus, Musil, Martius and Rolínek, "Differentiation of Blackbox Combinatorial Solvers", ICLR 2020,
arXiv 1912.02175.**
- It has the same backward pass: perturbed costs w' = w + λ·dL/dy and gradient −(y − y_λ)/λ.
- It has the same task shape: Warcraft shortest path, with a CNN predicting vertex costs, Dijkstra as the solver,
  and a Hamming loss.
- Each map size there was trained and tested separately, so size extrapolation was not their question.

**What the "market" reading adds.** The reading is that local traders price cells and an exact clearing rule
aggregates them.
- No component of the arm differs from Vlastelica et al.: no trader, wealth, payment or price-discovery mechanism
  was built or tested.
- The reading is a relabelling. The measurable content is "local, size-invariant cost features plus an exact
  global solver extrapolate in size better than one recurrent net carrying global state." That claim is itself
  refuted against a working DT: the in-house released DT-Recall solves 100% of 59×59 mazes where CostNet+BB solves 25%.

**The DT side of the comparison is Bansal et al. 2022.** Bansal, Schwarzschild, Borgnia, Emam, Huang, Goldblum and
Goldstein, "End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking",
NeurIPS 2022, arXiv 2202.05826.

## 1a: 32 seeds of P12's CostNet+BB on P12's weighted mazes

The arm is the P12 recipe verbatim: w = 32, 12000 steps, λ = 20, seeds 100–131, on a spot c2d-highcpu-56.

### Gates (automatic, `vmout/eval1a/GATE_1a.json`)
- **G1 passed.** The oracle's exact rate is 1.0 on every set.
- **G2 passed.** The median gate_W9 exact is 0.79 (min 0.766) against a 0.60 bar, and 32 of 32 seeds trained.

### Primary: pooled exact on test_W33 ∪ W49 ∪ W65 with L > 21 (400 mazes, 200 + 100 + 100)

| | mean over 32 seeds | 95% t-CI | SD | median | min–max |
|---|---|---|---|---|---|
| final (step 12000), primary | **28.66%** | 28.19–29.14 | 1.33 | 28.62 | 25.75–31.00 |
| step 9000 (P12's checkpoint) | 28.05% | 27.54–28.56 | 1.42 | 28.00 | 25.50–32.25 |
| control_uniform (all costs 1) | 1.25% | | | | |
| control_oracle (true costs) | 100% | | | | |
| P12 seed 0, step 9000 (recomputed from P12's npz) | 27.0% | | | | |

### Bars
- **Bar 1 holds.** The CI lower bound, 28.19, is above uniform + 5 = 6.25.
- **Bar 2 holds against an invalid DT baseline.** The lower bound, 28.19, is above 8.06 (P12's best DT arm + 5).
  P12's DT failed its own 9×9 validity gate (60% against a 90% bar), so this bar is labelled invalid.
- **Bar 3: 27% was typical.** 26 of 32 new seeds (81.25%) reach at least 27.0% at step 9000, which puts P12
  seed 0 at the 18.75th percentile. That counts seeds strictly below 27.0; counting s119, which ties at exactly
  27.0, gives the 21.9th.
- **Verdict: replicates.**

**Caveats.**
- The seed CI covers training randomness only. All seeds are scored on the same 400 test mazes; the
  binomial SE of one 400-maze sample at 28.7% is about 2.3 points, and that is not in the CI.
- The planned per-seed SD was 10–15 points and the observed SD is 1.3, so n = 32 was far more than needed.

### Secondaries: exact by set, final checkpoint (mean over 32 seeds; uniform control in brackets)

| set | exact | uniform |
|---|---|---|
| test_W9 | 78.9 | 41.0 |
| test_W15 | 66.5 | 23.7 |
| test_W25 | 49.1 | 8.0 |
| val_W25 | 46.2 | 5.5 |
| test_W33 | 41.2 | 1.5 |
| test_W49 | 20.7 | 1.0 |
| test_W65 | 11.7 | 1.0 |
| shift p0.1 (W33) | 69.1 | 33.0 |
| shift p0.5 (W33) | 17.1 | 0.0 |

- The valid-path rate is 100% on every set: every cell is passable in P12's weighted mazes.
- Exact still falls steeply with size, from 79% at 9×9 to 12% at 65×65. Local costs do not make it size-invariant.

## 1b: CostNet+BB on the official Bansal et al. easy-to-hard mazes

The setup:
- **Data and training.** Train on maze_data_train_9 (48k train and 2k val rows).
- **Scoring.** The official exact rule: the open-masked prediction must equal the solution on every pixel.
- **Recipe.** It is P12's, with two changes that were chosen on train/val before any test run (see PREREG-1b's
  pilot table):
  - **BBPos.** The perturbed costs are clamped at 1e-3 instead of 0.
  - **Per-maze cost normalisation (`--norm 1`).** Each maze's costs are divided by their mean.
- **Seeds.** 200–219, n = 20.

**CostNet+BB was handed things DT is not.** It gets a fixed block tokenisation (pixels[3:-3:2, 3:-3:2]), the grid
graph and an exact shortest-path solver, and it only has to learn that black blocks are expensive. DT learns
everything from pixels, so a CostNet win would measure the value of that built-in structure, not a better learner.

### Gates (automatic, `vmout/eval1b/GATE_1b.json`)
- **G1 passed.** The data sha256 check passed, and selftest confirmed Dijkstra = Bellman-Ford at n = 9/13/33, the
  pixel and edge round trips, and an oracle exact rate of 1.0 on val.
- **G2 passed.** The median val exact is 1.0 against a 0.90 bar, and 20 of 20 seeds trained.
- **Backward solver.** 0 of 15,360,000 blackbox backward solves failed (768,000 per seed × 20). P12's clamp-0 code failed on some; see Defects.
- **DT validity gate passed.** The released DT-Recall scores peak 100% on e2h_maze_13 (all 10,000 mazes; bar ≥ 95%).
  Checkpoint: `dt_net_recall_2d`, width 128, α = 0.01, trained on 9×9 (`model_meta.json` in the dt-baseline lane).

### Results (exact rate; the valid rate is identical on every set)

| set (mazes scored) | CostNet+BB mean, 20 seeds | 95% t-CI | SD | min–max | DT-Recall released (dt-baseline): peak (iter) / final, T |
|---|---|---|---|---|---|
| e2h_maze_13 (all 10,000) | 99.84% | 99.77–99.91 | 0.15 | 99.51–99.99 | 100% (68) / 100%, T 200 |
| e2h_maze_33 (first 1,000) | 58.72% | 55.48–61.96 | 6.93 | 47.9–71.3 | 100% (278) / 100%, T 500 |
| **e2h_maze_59 (first 1,000), primary** | **25.05%** | 23.56–26.53 | 3.17 | 19.8–30.9 | **100% (726) / 100%, T 1000** |
| e2h_maze_201 (first 100) | 2.00% | 2.00–2.00 | 0 | 2–2 | not landed at 02:50Z |

- **Primary verdict: worse.** The pre-registered bars against DT's e2h_maze_59 peak_acc (D = 100%) are:
  - superior if the CI lower bound is above D + 5;
  - non-inferior if the lower bound is above D − 5;
  - worse if the CI upper bound is below D − 5;
  - inconclusive otherwise.

  The upper bound 26.53 is below 95, so the verdict is "worse" (`results1.json` `item1b.primary_verdict`). DT's (a)
  peak and (b) final number agree (both 100%), so the peak's oracle stopping does not matter here. DT does need many
  iterations at this size: 62.1% at iteration 300, 92.0% at 500, first ≥ 95% at 528 (the dt-baseline lane chose T = 1000).
  It is not monotone after the peak (dips to 99.9% as late as iteration 988) but ends at 100%.
- **The median wall/open cost ratio is 241–288** across the sets (per-seed range 201–358).
- **Valid equals exact.** These are perfect mazes: a start-to-goal path that avoids every wall is the solution.
  Every failure is therefore a path that cuts through a wall.

### Why 1b fails at size (exploratory, NOT pre-registered; `why1b_exploratory.json`, seeds 200 and 201)

**Walls are priced at a finite multiple of open cells.** The network learns walls at about 250–295× the cost of
open cells, from 9×9 mazes where no detour is that long. Once the true path is longer than that ratio, crossing
one wall is cheaper than the detour, and the exact solver takes it.

Exact rate by solution path length in blocks, seed 200 on e2h_maze_59 (wall ratio 252):

| path length | < 100 | 100–200 | 200–300 | 300–500 | 500+ |
|---|---|---|---|---|---|
| exact | 100% | 98% | 59% | 16% | 0% |

- Seed 201 (ratio 295) follows the same curve, shifted slightly right.
- **So "an exact solver extrapolates by construction" is false once walls are in the priced graph.** The solver is
  only as size-invariant as the cost scale the network learned.
- Removing walls from the graph would make extrapolation trivial, and would hand the solver the answer structure.

## Defects and deviations (all disclosed; none chosen by test numbers)

1. **Clamp-0 bug in P12's blackbox backward (`nets.BBShortestPath.backward`).**
   - What happens: the code does `(ecost + λ·g).clamp_min(0)`. An edge that ends up at cost 0 makes the
     solver's walk-back cycle, so y_λ, and with it that maze's gradient, is garbage.
   - Measured on P12's seed-0 training batches: 7 of 640 backward solves failed at step 9000, 3–4 of 640 at step
     3000 (depending on which batches are drawn) and 0 at init. Saved in `clamp0_diag.txt`; script
     `code/diag_clamp0.py`.
   - 1a keeps the bug, because 1a replicates P12 as produced. 1b fixes it with BBPos.
   - So P12's CostNet+BB number, and 1a's, were obtained with a slightly broken gradient.
2. **gate.py was changed after the freeze, in the write step only.**
   - Why: three 1a seeds (s108, s112, s118) crashed with FileNotFoundError while renaming the shared `.tmp` file,
     just after GATE_1a had already passed.
   - Fix: the tmp name is now per-process (line 71; `diff code/gate.py.frozen code/gate.py`). The frozen copy's
     sha256 f422e9d8… matches both PREREG sha256 files; the current file is 8d32e5a6….
   - `code/bansal.py` also fails the 1a manifest `PREREG.sha256` (65864ce9 then, 716faa35 now), because the 1b
     fixes were added after the 1a freeze. 1a never imports bansal.py. The current file matches `PREREG-1b.sha256`.
   - Those three seeds' test evals were then run. They were never missing from the analysis, and no seed was
     excluded (logged in `vm/LEDGER.md`).
3. **1b ran beside 1a** on the same VM (started 23:24:50Z, after PREREG-1b was frozen at 23:24:01Z).
4. **1b's two recipe changes were chosen on train/val pilots** (seeds 900–930, summarised in PREREG-1b). PREREG-1b
   points at /tmp/fu-b2/smoke for the logs; that dir was deleted at cleanup, and the logs and scripts were first
   archived in `prefreeze_pilots/`. λ stayed at P12's 20, and nothing was tuned after any test number.

## Cost

- **VM:** `trader-fu-c1-1`, c2d-highcpu-56 spot, 23:15:20–23:34:53Z (0.33 h).
- **Spend at the ceiling rate ($2.10/h):** $0.68.
- **Cleanup:** the VM and its disk were deleted after a verified sync (472 files, 0 sha256 mismatches).
  `gcloud compute instances list` and `disks list` show no trader-fu-c1 resources.

## Files

- **Pre-registrations:** `PREREG.md`, `PREREG.sha256`, `PREREG-1b.md`, `PREREG-1b.sha256`, `power.py`, `power.txt`.
- **Code:** `code/` — P12's code with three patches, `bansal.py`, `gate.py`, `gate.py.frozen`, `run1a.sh`,
  `run1b.sh`, and `bansal_data/SHA256SUMS`. The dataset copies were trashed as duplicates of P12's data and
  dt-baseline's official tarballs (sha256 identical; see `code/DATA-MOVED.txt`).
- **Raw outputs:** `vmout/eval1a/*.npz|json`, `vmout/eval1b/{val,test}/*.json|npz`, `vmout/runs1a`, `vmout/runs1b`,
  `vmout/logs`.
- **Analysis:** `analyze1.py`, `results1.json`, `why1b_exploratory.json`.
- **Logs:** `vm/LEDGER.md`, and `../../../GPU-LEDGER.md` (batch-2 follow-ups section).
