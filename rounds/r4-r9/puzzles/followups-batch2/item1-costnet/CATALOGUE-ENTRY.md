## FU-B2-1: Does P12's CostNet+BB lead (learned local cell costs + an exact blackbox-differentiated solver, 27% exact on 3-7x wider mazes, one seed) hold over many seeds, and does it match the official DT-Recall on the Bansal et al. easy-to-hard mazes?

Status: Informative (verified; verifier agrees). 1a replicates; 1b is worse than the released DT-Recall.

Setup:
- **Prior work, named.** CostNet+BB is Vlastelica et al., "Differentiation of Blackbox Combinatorial Solvers", ICLR 2020 (arXiv 1912.02175). The "local traders price cells, exact clearing" reading is a relabelling of that method: no market mechanism was built or tested.
- **1a.** P12's recipe verbatim (w = 32, 12k steps, λ = 20) over 32 seeds, a number set by a power calculation written before the run. Primary: pooled exact on n = 33/49/65 (400 mazes). Gates G1 (oracle ≥ 0.99) and G2 (median 9×9 ≥ 0.60) are applied automatically.
- **1b.** Official maze_data_train_9 and the e2h_maze test sets, 20 seeds, official exact rule. Two fixes were chosen on train/val before any test:
  - perturbed costs are clamped at 1e-3, not 0;
  - costs are normalised per maze.
- **1b primary.** e2h_maze_59 against the released DT-Recall checkpoint as evaluated by dt-baseline, with ±5-point bars.
- **Pre-registration.** PREREG sha256s were frozen at 23:14Z (1a) and 23:24Z (1b).

Result:
- **1a replicates. P12's 27% was typical.**
  - Primary exact is 28.66% [28.19, 29.14] (SD 1.3), against 1.25% for uniform costs and 100% for the oracle.
  - 26 of 32 seeds reach at least 27% at step 9000.
  - The only DT it beats is P12's, which failed its validity gate (labelled bar).
  - Exact still falls with size: 78.9 / 41.2 / 20.7 / 11.7% at n = 9 / 33 / 49 / 65.
- **1b: worse (pre-registered primary).** On 59×59, CostNet+BB solves **25.05% [23.56, 26.53]** exactly; the released DT-Recall checkpoint, evaluated in-house by dt-baseline (lane trader-dt-p1, T = 1000), solves **100%** (peak at iteration 726, final also 100%). The upper bound is far below the DT − 5 bar of 95. The DT validity gate passed (13×13: 100%).
  - Exact by set, CostNet+BB against DT: 99.84% [99.77, 99.91] against 100% on 13×13, 58.7% [55.5, 62.0] against 100% on 33×33, and 2.0% on 201×201 (DT eval not landed at 02:50Z on 09-30).
  - Paper context: Bansal et al. 2022 report DT-Recall at 97.3% on 59×59 (Table 4) and 74% on 201×201 (text); the in-house reproduction is at or above the paper.
- **Why it fails at size (exploratory).**
  - The net prices walls at about 200–360× open cells (per-seed medians). Once the true detour is longer than that, cutting through a wall is cheaper.
  - Exact on 59×59 falls from 100% for paths under 100 blocks to 0% above 500.
  - An exact solver does not extrapolate "by construction" when walls are in the priced graph.
- **Defect found.** P12's blackbox backward clamps perturbed costs at 0, which makes the solver's walk-back cycle. It failed on up to 7 of 640 backward solves in P12's own run (`clamp0_diag.txt`). 1a keeps the bug; 1b fixes it.
- **Cost.** $0.68 at the ceiling (c2d-highcpu-56 spot, 0.33 h). The VM and disk are deleted, and gcloud confirms it.

Files: [followups-batch2/item1-costnet/RESULTS.md](followups-batch2/item1-costnet/RESULTS.md), `followups-batch2/item1-costnet/{PREREG.md,PREREG-1b.md,analyze1.py,results1.json,why1b_exploratory.json,prefreeze_pilots/}`, `dt-baseline/results/vm/trader-dt-p1/eval/released_maze/`, `GPU-LEDGER.md` (batch-2 follow-ups section)
