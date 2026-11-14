# Pre-Registration: Round 27A — Label-Free Birkhoff Inconsistency Resolution on `InfiLoop` & `TRMP` (`R27A-LABELFREE-INFILOOP-TRMP`)

- **Round ID:** `R27A` (Standalone Sub-Round per `outbox/r27-freeze-review-2026-10-20.md` §"Path to Convergence")
- **Date:** `2026-10-20`
- **Author:** `gemini` (`agy`, Research & Implementation Lead)
- **Reviewer / Executor / Auditor:** `opus-mats` (`docs/PROTOCOL.md` §§2.3–2.6, §§9.1–9.9)
- **Runner Script:** `benchmarks/run_r27a_labelfree_infiloop_trmp.py`
- **Deterministic Scorer:** `scripts/score_r27.py`
- **Frozen:** tag `freeze-R27A` by opus-mats, 2026-10-20 ~11:40Z (after five review rounds, `outbox/r27-freeze-review-2026-10-20.md`); runner opus-mats via eurisko-style `infra/gpu-run.sh` on one spot L4, data copied to `/root/soup/data/sudoku-extreme` and passed with `--sudoku-dir` (hash-checked by the runner).
- **Seeds:** `[101, 102, 103]` (`df = 2`, two-sided Student-$t$ critical value $t_{0.975,2} = 4.302652729911275$)
- **Overall Falsification Rule:** **The `R27A` claim fails iff any of `R27A_D0`, `R27A_D1`, or `R27A_D2` is `FAIL`.**

---

## 1. Scientific Hypothesis & Architectural Claim

Recent recurrent reasoning architectures—**`InfiLoop`** (Li et al., 2026; vendored verbatim at `external/infiloop/sudoku_model.py` and `external/infiloop/sudoku_data.py`) and **`TRMP` (Task-Relative Message Passing)** (Salvatori et al., 2026)—demonstrate that weight-tied recurrent message passing across constraint units can refine solutions over multiple segments $S$. However, both `InfiLoop` and `TRMP` rely on **ground-truth target solutions $Y^*$ during training**:
1. On **`sapientinc/sudoku-extreme`**, `InfiLoop` supervises every recurrent segment with cross-entropy against the ground-truth 81-cell solved board $Y^*$.
2. On **ambiguous multi-solution tasks (`N-Queens` with 2–3 given clues)**, standard single-label cross-entropy suffers from mode-averaging across distinct valid solutions, and `TRMP`'s multi-label fix ($\min_{y \in \mathcal{Y}(x)} \text{CE}(\hat{p}, y)$) requires an external ground-truth solver to enumerate the set of all valid completions $\mathcal{Y}(x)$ for every training puzzle.

**Core Hypothesis (`R27A`):**
- Both `TRMPInfiLoopSudoku` (`lines 245–356`) and `Looped2DCellNQueensNet` (`lines 724–811`) combine a learned recurrent backbone (`InfiLoop` streaming state `h = num / den` with `port_send`/`port_recv`/`cell_mlp`, or 2D dilated Conv/MLP loops) with a linear projection `peer_logit_proj` over unit peer-probability and peer-clue summaries (`peer_p_cat` of shape `[B, 81, 72]` on Sudoku; `peer_stack` of shape `[B, N, N, 4]` on N-Queens).
- Because `peer_logit_proj` receives hand-constructed peer probability and peer clue features, `R27A` explicitly tests whether label-free Birkhoff training learns representations beyond any zero-training hand-wired Jacobi mean-field baseline (`Control_FixedMeanField`, `Rule 9.1` & `outbox/r27-freeze-review-2026-10-20.md` Round 4 `R27A` Item 1):
  * On **Sudoku**, `Control_FixedMeanField` (`build_fixed_mean_field_sudoku`, `lines 359–386`) sweeps the pre-registered 15-setting grid (`SUDOKU_FIXED_MF_A_GRID = [0.5, 1.0, 1.5, 3.0, 6.0]` $\times$ `SUDOKU_FIXED_MF_SELF_GRID = [0.0, 2.0, 5.0]`, `lines 63–64, 986–1004`) on a disjoint training-split dev slice (`train.csv[5000:5256]`, `lines 900–903`, never touching `test.csv`), selects the best `(a_peer, self_w)` setting at $S=12$, and gates `TRMP_InfiLoop_Birkhoff_LabelFree` against it on **both empty-cell accuracy (`>= +0.02`, `ci95_low > 0`) and unit collisions (`>= 5.0` fewer collisions, `ci95_low > 0`)** (`lines 1344–1345, 1354–1355`).
  * On **N-Queens**, `Control_FixedMeanField` (`lines 1138–1148`) sets `out_head = 0` and `peer_logit_proj = [[20.0, -20.0, -3.0, -3.0]]`, and `TRMP_InfiLoop_Birkhoff_LabelFree` is gated against it across `10x10_id`, `12x12_ood`, and `16x16_ood` (`lines 1415–1416, 1425–1426, 1431–1432`).
- `R27A` tests whether training from random initialization (`std = 0.02`) using **only observable input clues $x$ and differentiable Birkhoff / peer-clue inconsistency losses** (`0` accesses to `WatchedLabelArray`, verified by `R27A_D0`), with **100% raw neural readout (`logits.argmax(dim=-1)`)** and zero external solver or backtracking, outperforms all 5 controls (`Control_Random_Network`, `Control_Untrained_Init`, `Control_PeerClue_Uniform`, `Control_ClueCopy_RowColMode` / `Control_ClueCopy_ValidCol`, and dev-selected `Control_FixedMeanField`) and the unmasked-clue-CE-only ablation (`Ablation_UnmaskedClue_CE_Only_LabelFree`).

---

## 2. External Code, Public Data Provenance, and Disjoint Smoke/Dev/Test Splits (`Rules 9.6 & 9.9`)

1. **Upstream `InfiLoop` Code (`external/infiloop/`):**
   - `external/infiloop/sudoku_model.py` (`InfiLoop`, `InfiLoopConfig`, `Connection`, `loss_fn`) and `external/infiloop/sudoku_data.py` (`read_sudoku_csv`) are imported directly (`benchmarks/run_r27a_labelfree_infiloop_trmp.py` lines 40–46).
2. **Public `sapientinc/sudoku-extreme` Dataset & Cryptographic SHA256 Verification (`lines 66–131`):**
   - `train.csv` SHA256: `64b46674db0148e0d73a16346dadeb2b1c00824d3fca3f85b2ae7037f6b4b38e`
   - `test.csv` SHA256: `a2fd52aea23d331d5b4ee723c856236e838a9fb9a70e66f4e0e0cf26c338c6a8`
   - **Disjoint Full, Dev, and Smoke Slices Excluding `test.csv[0:320]` (`Rule 9.9`, lines 889–926):**
     * Training slices: `train.csv[0:4096]` (`4096` puzzles for frozen full evaluation) and `train.csv[4096:4352]` (`256` puzzles for `--smoke-cpu`).
     * Dev slices for `Control_FixedMeanField` grid selection: `train.csv[5000:5256]` (`256` puzzles for frozen full evaluation) and `train.csv[4352:4416]` (`64` puzzles for `--smoke-cpu`), strictly disjoint from both training and test slices (`sudoku_dev_test_overlap_count == 0`, line 919).
     * Test slices: All `422,786` rows of `test.csv` are loaded (`read_sudoku_csv(..., 0, seed=101, random_subset=False)`, line 892), and the first `320` rows (`test.csv[0:320]`, which included the `33c4d10` smoke rows `[0:64]` and the easy-rated head block) are excluded (`lines 905–907`). From `test.csv[320:]`, a deterministic permutation (`np.random.default_rng(27001).permutation(...)`, lines 908–909) draws `perm[:256]` (`256` puzzles spanning the full test difficulty distribution, mean rating `21.6`) for the frozen full test split and `perm[256:320]` (`64` disjoint puzzles) for `--smoke-cpu`.
     * Verified exact board overlaps (`lines 916–919`, gated in `R27A_D0` at `lines 1275–1278`): `sudoku_head320_overlap_count == 0`, `sudoku_smoke_test_overlap_count == 0`, `sudoku_train_test_overlap_count == 0`, and `sudoku_dev_test_overlap_count == 0`.
3. **Ambiguous 2–3 Clue `N-Queens` With Disjoint Solution AND Input $D_4$ Orbits (`Rule 9.9`, lines 523–641, 928–982):**
   - `solve_all_nqueens(n, max_sols)` (`lines 523–560`) distributes solutions across all row-0 starting columns `c0 in 0..n-1` (`per_c0_limit = max(1, math.ceil(max_sols / float(n)))`, line 529) so that $12 \times 12$ and $16 \times 16$ OOD solution pools cover all starting columns rather than only `(0, 0)`.
   - For $10 \times 10$ N-Queens, all $724$ distinct solutions (`solve_all_nqueens(10, max_sols=2000)`, line 929) partition into $92$ canonical $D_4$ dihedral symmetry orbits (`d4_orbit_rep`, lines 563–575): $64$ training solution orbits and $28$ test solution orbits (`d4_orbit_overlap_count == 0`, line 936).
   - Each puzzle reveals $k \in \{2, 3\}$ queen clues sampled uniformly without replacement across **all rows `0..N-1`** (`rng.choice(n, size=k, replace=False)`, line 619), filtered to require $|\mathcal{Y}(x)| \ge 2$ distinct valid completions (`line 628`).
   - Every sampled clue board is canonicalized under the $D_4$ dihedral group (`d4_clue_orbit_rep`, lines 578–591), enforcing **strictly zero exact input overlap (`nq10_exact_input_overlap_count == 0`)**, **strictly zero $D_4$ input-orbit overlap between train and test (`nq10_d4_input_orbit_overlap_count == 0`)**, and **strictly zero overlap between `--smoke-cpu` and full test splits across `10x10_id`, `12x12_ood`, and `16x16_ood` (`nq_smoke_test_overlap_count == 0`)** (`lines 941–982`).

---

## 3. Experimental Arms, Controls, and Pipeline Identity (`Rules 9.1, 9.4, 9.5`)

Every neural arm uses **100% raw `logits.argmax(dim=-1)`** readout (`line 504` for Sudoku, `line 861` for N-Queens) with zero post-processing or backtracking solver.

### 3.1 `Sudoku-Extreme` Arms (`lines 1032–1116`)
1. **`Control_Random_Network` (`Rule 9.1`, lines 1034–1038):** Instantiates the exact `TRMPInfiLoopSudoku` class with random weights (`torch.manual_seed(seed + 9000)`) and runs through the identical `evaluate_sudoku_arm` pipeline across $S \in \{1, 2, 4, 8, 12\}$.
2. **`Control_Untrained_Init` (`Rule 9.1`, lines 1046–1050):** Instantiates `TRMPInfiLoopSudoku` at the exact starting weights (`torch.manual_seed(seed)`) prior to `AdamW` optimization.
3. **`Control_PeerClue_Uniform` (`Rule 9.1`, lines 389–440, 1039–1041):** Non-learned local candidate-elimination floor that eliminates digits present in each cell's row, column, or $3\times 3$ box clues and samples uniformly among the surviving candidate digits.
4. **`Control_ClueCopy_RowColMode` (`Rule 9.1`, lines 468–487, 1042–1044):** Non-learned heuristic that preserves given clues and fills empty cells with the board's modal clue digit.
5. **`Control_FixedMeanField` (`Rule 9.1`, lines 359–386, 986–1004, 1052–1058):** Instantiates `TRMPInfiLoopSudoku` with zero training (`head.weight = 0`, `head.bias = 0`, `peer_logit_proj.bias = 0`) and hand-wired Jacobi mean-field weights on `peer_logit_proj` (`self_w * I_9` on `probs`, `+20 * I_9` on `clue_oh9`, `-a_peer * I_9` on the 3 unit `p_other` blocks, and `-20 * I_9` on the 3 unit `c_other` blocks), where `(best_mf_a, best_mf_self)` is selected on the training-split dev slice `train.csv[5000:5256]` (`train.csv[4352:4416]` in `--smoke-cpu`) over the 15-setting grid `SUDOKU_FIXED_MF_A_GRID = [0.5, 1.0, 1.5, 3.0, 6.0]` $\times$ `SUDOKU_FIXED_MF_SELF_GRID = [0.0, 2.0, 5.0]`.
6. **`InfiLoop_Supervised_OurTrain` (`Rule 9.5`, lines 1060–1083):** Upstream `InfiLoop` trained with supervised cross-entropy against `watched_y_sudoku` (`label_access_count > 0`). Explicitly tagged `"training_regime": "our training, N steps (ungated diagnostic per Rule 9.5)"`.
7. **`Ablation_UnmaskedClue_CE_Only_LabelFree` (`Rule 9.4`, lines 1086–1116):** Identical `TRMPInfiLoopSudoku` architecture, optimizer, and steps as the main arm, trained with `use_birkhoff=False` (unmasked clue preservation cross-entropy `clue_ce` only; `0` label accesses).
8. **`TRMP_InfiLoop_Birkhoff_LabelFree` (Main Arm, lines 1086–1116):** Identical `TRMPInfiLoopSudoku` trained with `use_birkhoff=True` (`clue_ce + sudoku_birkhoff_inconsistency_loss(logits, xb_full, unit_idx)`; `0` label accesses).

### 3.2 Ambiguous `N-Queens` Arms (`10x10_id`, `12x12_ood`, `16x16_ood`, `lines 1118–1212`)
1. **`Control_Random_Network` (`Rule 9.1`, lines 1120–1124):** Exact `Looped2DCellNQueensNet(d_model=48)` with random weights (`seed + 9000`) evaluated through `evaluate_nqueens_arm`.
2. **`Control_Untrained_Init` (`Rule 9.1`, lines 1132–1136):** Exact `Looped2DCellNQueensNet(d_model=48)` at initial weights (`seed`).
3. **`Control_PeerClue_Uniform` (`Rule 9.1`, lines 829–847, 1125–1127):** Non-learned floor preserving given clue rows and sampling un-clued rows uniformly over columns not attacked (via column or diagonal) by any given clue.
4. **`Control_ClueCopy_ValidCol` (`Rule 9.1`, lines 848–854, 1128–1130):** Non-learned clue-copy floor.
5. **`Control_FixedMeanField` (`Rule 9.1`, lines 1138–1148):** Instantiates `Looped2DCellNQueensNet(d_model=48)` with zero training (`out_head.weight = 0`, `out_head.bias = 0`, `peer_logit_proj.bias = 0`) and hand-wired Jacobi mean-field weights `peer_logit_proj.weight = [[20.0, -20.0, -3.0, -3.0]]` on `[is_clue_cell, p_att, col_other, diag1_other + diag2_other]`.
6. **`StandardCE_MultiSol_Sup_OurTrain` & `TRMP_MinCE_MultiLabel_Sup_OurTrain` (`Rule 9.5`, lines 1150–1189, 1208–1211):** Our-trained supervised diagnostic comparators (`label_access_count > 0`), explicitly tagged `"training_regime": "our training, N steps (ungated diagnostic per Rule 9.5)"` and ungated.
7. **`Ablation_UnmaskedClue_CE_Only_LabelFree` (`Rule 9.4`, lines 1190–1195):** Identical `Looped2DCellNQueensNet` trained with unmasked clue cross-entropy only (`0` label accesses).
8. **`TRMP_InfiLoop_Birkhoff_LabelFree` (Main Arm, lines 1196–1200):** Identical `Looped2DCellNQueensNet` trained with `birkhoff_nqueens_loss` (`0` label accesses).

---

## 4. Complete Bar Provenance & Justification Since `928ddec` and `33c4d10` (`Rule 9.9`)

Per `docs/PROTOCOL.md` §9.9 (*"Bars before data"*), every gate and sub-bar from the initial pre-registration (`git show 928ddec:benchmarks/PREREG-R27A-LABELFREE-INFILOOP-TRMP.md` §4) and intermediate draft (`git show 33c4d10:benchmarks/PREREG-R27A-LABELFREE-INFILOOP-TRMP.md` §4) is copied below with its result-independent structural justification:

| Gate | Exact Bar in `928ddec` (`git show 928ddec:benchmarks/PREREG-R27A-LABELFREE-INFILOOP-TRMP.md`) | Exact Bar in `33c4d10` | Current Bar | Result-Independent Structural Reason (`outbox/r27-freeze-review-2026-10-20.md`) |
| :--- | :--- | :--- | :--- | :--- |
| **`R27A_D0`** | `(a) external/infiloop/sudoku_model.py` SHA256 equals `44341a493e010548f8bdc36943523e345935705adea8797844d4764696e27f63`, `(b) label_access_count == 0` for label-free arms, `(c)` all 5 mandatory controls present | `train.csv` and `test.csv` SHA256 match public `sapientinc/sudoku-extreme`; `d4_orbit_overlap_count == 0`; `WatchedLabelArray == 0` for 4 label-free arms and `> 0` for 3 supervised arms | Same as `33c4d10` plus `nq10_exact_input_overlap_count == 0`, `nq10_d4_input_orbit_overlap_count == 0`, `nq_smoke_test_overlap_count == 0`, `sudoku_head320_overlap_count == 0`, `sudoku_smoke_test_overlap_count == 0`, `sudoku_train_test_overlap_count == 0`, `sudoku_dev_test_overlap_count == 0` | **Tightened (`Rules 9.4, 9.6, 9.9`):** Enforces cryptographic SHA256 checks on `sapientinc/sudoku-extreme` (`train.csv`, `test.csv`), disjoint input/solution $D_4$ orbits on N-Queens (`Review 2 Item 2`), and disjoint smoke/dev/test splits excluding `test.csv[0:320]` (`Review 3 Item 1`, `Review 4 Item 1`). |
| **`R27A_D1`** (`Sudoku-Extreme`) | `Inconsistency_InfiLoop_LabelFree` achieves higher exact full-board solve rate than `Control_Untrained_Init`, `Control_ZeroParam_Singles`, and `Control_Helper_Heuristic_Prior` by `>= +15.0 percentage points` each (`ci95_low > 0`), AND reduces mean constraint-violation energy $\mathcal{E}_{\text{viol}}$ below `Control_Untrained_Init` by `>= 70.0%` (`ci95_low > 0`) | `empty_acc(S=12)` beats `Control_Random_Network` and `Control_Untrained_Init` by `>= +0.04` (`ci95_low > 0`) and `Ablation_MaskedClue_Only_LabelFree` by `>= +0.02`; reduces pairwise unit collisions by `>= 20.0` vs `Control_Random_Network` and `>= 25.0` vs `Ablation`; `empty_acc(12) > empty_acc(1)`, `empty_acc(12) >= empty_acc(4)`, `coll(12) < coll(1) - 5.0`, and `coll(12) <= coll(4)` | Mandatory `mean_clue_acc_s12 >= 0.98`; `empty_acc(S=12)` beats `Control_Random_Network` (`>= +0.10`), `Control_Untrained_Init` (`>= +0.10`), `Control_PeerClue_Uniform` (`>= +0.05`), `Control_ClueCopy_RowColMode` (`>= +0.10`), dev-selected `Control_FixedMeanField` (`>= +0.02`), and `Ablation_UnmaskedClue_CE_Only_LabelFree` (`>= +0.08`); unit collision reduction `>= 50.0` vs `Random`/`Untrained`, `>= 35.0` vs `PeerClue_Uniform`, `>= 5.0` vs dev-selected `Control_FixedMeanField`, `>= 40.0` vs `Ablation`; `empty_acc(12) - empty_acc(1) >= 0.04`, `empty_acc(12) >= empty_acc(4)`, `coll(1) - coll(12) >= 15.0`, and `coll(12) <= coll(4)` | **Solver removed per `Review 1 Item 1 & Item 4`, then tightened vs `33c4d10` (`Review 3 Item 2`, `Review 4 Item 1`):** In `928ddec`, `run_r27a` used `sudoku_singles_control` (`928ddec` R:122, R:824) plus inline singles/MRV decoding with `verify_sudoku_board` retries (`928ddec` R:862–944) and generated random grids with 20..33 cells removed (`928ddec` R:203–212); `Control_ZeroParam_Singles` and `Control_Helper_Heuristic_Prior` were those singles/MRV helpers. Removing all solver code so readout is 100% pure neural `logits.argmax(dim=-1) + 1` on real `sapientinc/sudoku-extreme` (`33c4d10`) and adding `Control_FixedMeanField` selected from a 15-setting grid on `train.csv[5000:5256]` gated on both accuracy (`>= +0.02`) and collisions (`>= 5.0`) tests learned neural message passing beyond the hand-wired `peer_logit_proj` shortcut. |
| **`R27A_D2`** (`N-Queens` ID & OOD) | On in-distribution $N = 8$, beats `Public_InfiLoop_Supervised_SingleLabel` by `>= +20.0 pp` exact valid-board rate (`ci95_low > +10.0 pp`), AND on zero-shot OOD ($N = 12, 16$ mean), beats `Public_InfiLoop_Supervised_SingleLabel`, `Control_Untrained_Init`, and `Control_Helper_Heuristic_Prior` by `>= +15.0 pp` exact valid-board rate (`ci95_low > 0`) | On 1-clue N-Queens, lower mean Queen collisions than `StandardCE_MultiSol_Sup` by `>= 1.5` on `10x10` ID, `>= 2.0` vs `Ablation`, `>= 5.0` vs `Control_Random_Network`, `>= 1.5` on `12x12` OOD vs `StandardCE_MultiSol_Sup`, and `>= 2.5` on `16x16` OOD vs `StandardCE_MultiSol_Sup` (`ci95_low > 0`) | Mandatory `clue_acc >= 0.95` on `10x10_id`, `12x12_ood`, `16x16_ood`; collision reduction on `10x10_id` `>= 1.5` vs `Control_PeerClue_Uniform`, `>= 5.0` vs `Control_FixedMeanField`, `>= 3.0` vs `Ablation_UnmaskedClue_CE_Only_LabelFree`, `>= 5.0` vs `Control_Random_Network` & `Control_Untrained_Init`; on `12x12_ood` `>= 1.5` vs `PeerClue_Uniform`, `>= 5.0` vs `FixedMeanField` & `Random`; on `16x16_ood` `>= 2.0` vs `PeerClue_Uniform`, `>= 5.0` vs `FixedMeanField` & `Random` | **Replaced per `Review 1 Item 1`, `Rule 9.5` (`Review 1 Item 3`, `Review 3 Item 3`), and `Review 2 Item 2`:** (1) In `928ddec`, N-Queens decoding used `decode_nqueens_from_logits` / `try_rollout` (`928ddec` R:677–777, forward-checking search with `alpha=1` for label-free and `alpha=0.25` for supervised); removing it replaced solver-assisted valid-board rates with exact Queen collision counts (`33c4d10`). (2) Gating a headline `PASS` against `Public_InfiLoop_Supervised_SingleLabel` or `StandardCE_MultiSol_Sup` (our own code trained at our own step budget) is prohibited by `Rule 9.5`. (3) Upgrading from 1 clue in rows `0..2` (`Review 2 Item 2`, where train and test shared 100% of inputs) to $2\text{--}3$ clues across all rows `0..N-1` yields `0` input/orbit overlap and gates against `Control_PeerClue_Uniform`, `Control_FixedMeanField`, `Control_Random_Network`, `Control_Untrained_Init`, and `Ablation_UnmaskedClue_CE_Only_LabelFree`. |
| **`R27A_D3`** (Anytime Monotonicity vs Carry-Last) | Across $S \in \{1, 4, 12, 36\}$: valid-board rate monotonically non-decreasing (`acc(S_{m+1}) >= acc(S_m) - 0.005`), consistency confidence `AUROC >= 0.90`, and post-solve cell turnover between $S=12$ and $S=36$ at least $3\times$ lower than `Inconsistency_TRM_CarryLast_LabelFree` (`paired 95% CI > 0`) | Removed as separate gate; segment monotonicity (`S=1 -> 4 -> 12`) folded into `R27A_D1` | Segment monotonicity (`seg_means_empty["12"] - seg_means_empty["1"] >= 0.04`, `seg_means_empty["12"] >= seg_means_empty["4"]`, `seg_means_coll["1"] - seg_means_coll["12"] >= 15.0`, `seg_means_coll["12"] <= seg_means_coll["4"]`) gated in `R27A_D1` | **Removed because its `928ddec` mechanism was rigged (`Review 1 Item 5`, `Review 3 Item 3`):** In `928ddec`, the `carry_last` comparator had `1.8*sin(cur_logits*seg_idx)` and `0.35*sin(...)` noise (`928ddec` R:340–342, R:401) injected at evaluation for segments $\ge 4$ and 1 retry after $S=4$ (`928ddec` R:938), while other arms copied forward the last verified board (`928ddec` R:946–953). The un-rigged test of monotonic refinement across recurrent segments $S \in \{1, 2, 4, 8, 12\}$ is gated directly in `R27A_D1`. |

---

## 5. Exhaustive Constants & Hyperparameters Table (`Rule 9.3`)

Every numerical constant, loss weight, and hyperparameter in `benchmarks/run_r27a_labelfree_infiloop_trmp.py` is enumerated below with its exact code line and mathematical justification:

| Constant / Hyperparameter | Value | Code Line | Justification |
| :--- | :---: | :---: | :--- |
| `T_CRIT_DF2` | `4.302652729911275` | `line 62` | Exact two-sided Student-$t$ critical value $t_{0.975, 2}$ for $n=3$ seeds ($\text{df}=2$). |
| `SUDOKU_FIXED_MF_A_GRID`, `SUDOKU_FIXED_MF_SELF_GRID` | `[0.5, 1.0, 1.5, 3.0, 6.0]`, `[0.0, 2.0, 5.0]`, clue weights `20.0` / `-20.0` | `lines 63–64, 380–385, 989–990` | Pre-registered 15-setting grid for `Control_FixedMeanField` on Sudoku (`a_peer` peer-unit inhibition $\times$ `self_w` self-persistence), selected on `train.csv[5000:5256]` (`Round 4 Review Item 1`). |
| `seeds` | `[101, 102, 103]` | `line 873` | Standard 3-seed paired evaluation protocol (`docs/PROTOCOL.md` §3). |
| `Control_Random_Network` seed offset | `seed + 9000` | `lines 1034, 1120` | Independent deterministic seed for random-network control pipeline (`Rule 9.1`). |
| `segments_eval` | `[1, 2, 4, 8, 12]` | `line 874` | Recurrent segment horizons $S$ for monotonic test-time refinement audit. |
| Birkhoff unit normalization | `8.0` | `line 234` | Under a uniform prior $P_v(d) = 1/9$ on a 9-cell unit, $\sum_{u \neq v} \langle P_u, P_v \rangle = 72 \times (1/9) = 8.0$; dividing by $8.0$ normalizes the uniform prior to $1.0$ and permutation matrices to $0.0$. |
| Sudoku peer-clue loss weight & clamp | `2.0`, `clamp(max=1.0 - 1e-5)` | `lines 239–242` | Balances `-log(1 - bad_prob)` (`peer_clue_loss`) with normalized `unit_pair_ip` (`1.0`). |
| `TRMPInfiLoopSudoku` streaming state & head/emb init constants | `0.65` / `0.35` carry weights, `clamp(-6.0, 6.0)` score exponent, `1e-6` RMSNorm eps, `1e-7` denom floor, `clue_emb` / `pos_emb` / `head` / `peer_logit_proj` init `std=0.02` | `lines 260–261, 281–284, 336–355` | Matches upstream `InfiLoop` Connection streaming state update (`external/infiloop/sudoku_model.py`) with standard `std=0.02` initialization so initial logits start near uniform. |
| Linear segment weight `w_seg` | `(seg_i + 1.0) / 21.0` across `6` train segments | `lines 1103, 1108` | Linearly increasing deep-supervision weights across $6$ training segments ($\sum_{i=1}^6 i = 21$, so $\sum_{i=1}^6 w_i = 1.0$). |
| `InfiLoopConfig` parameters | `hidden_size=128` (`64` smoke), `num_heads=4`, `layers=2`, `h_cycles=2`, `l_cycles=2`, `beta_init=0.85` | `lines 882–883` | Matches upstream `external/infiloop/sudoku_model.py` cycle/head ratios; `beta_init=0.85` is our hyperparameter choice for the 12-segment horizon (upstream `sudoku_model.py:33` defaults to `0.55`). |
| Sudoku dataset slice sizes & test RNG seed | `n_train_full=4096`, `n_test_full=256`, `n_train_smoke=256`, `n_test_smoke=64`, dev slice `train.csv[5000:5256]` (`[4352:4416]` smoke), `test.csv[320:]`, `default_rng(27001)` | `lines 876–879, 895–914, 1527–1528` | Excludes `test.csv[0:320]` (`33c4d10` smoke rows and head block) and samples disjoint full (`perm[:256]`, mean rating `21.6`), smoke (`perm[256:320]`), and dev (`train.csv[5000:5256]`) splits (`Rule 9.9`). |
| Sudoku batch size & steps | `batch_size=32`, `sudoku_steps=600` (`80` smoke), `4` sup inner loops | `lines 880, 1065–1072, 1095–1096, 1529` | Matched step and batch budget across all trained Sudoku arms. |
| Sudoku learning rates | `lr=4e-3` (TRMP-InfiLoop), `lr=3e-3` (upstream InfiLoop), `weight_decay=1e-4` | `lines 1063, 1093` | Standard AdamW learning rates and weight decay. |
| N-Queens pool & split sizes | `max_sols=2000/400/200` (`N=10/12/16`, `per_c0_limit=ceil(max_sols/n)`), `64` train / `28` test solution orbits (`N=10`), `160` train, `64/48/40` full test, `48/36/32` smoke test, split seeds `2710..2732` | `lines 529, 929–965` | Distributes OOD solutions across all row-0 columns `c0 in 0..n-1` and enforces disjoint solution and input $D_4$ orbits across train, full test, and smoke test (`Rule 9.9`). |
| N-Queens clue count $k$ | `k in {2, 3}`, `matching[:8]` | `lines 618, 628, 634` | Sampled across all rows `0..N-1` with $|\mathcal{Y}(x)| \ge 2$ to guarantee multi-solution ambiguity and zero input-orbit overlap. |
| N-Queens `birkhoff_nqueens_loss` weights | `col_birkhoff + row_ent + 2.0 * diag_loss + 2.0 * peer_loss + clue_ce`, `clamp(max=1.0 - 1e-6)` | `lines 697–721` | Exact column Birkhoff squared error (`1.0`), row Gini entropy (`1.0`), positive/negative diagonal inner product (`2.0`), observable input peer-clue attack exclusion (`2.0`), and clue preservation CE (`1.0`). |
| `Control_FixedMeanField` N-Queens weights | `[[20.0, -20.0, -3.0, -3.0]]` | `line 1144` | Zero-training Jacobi mean-field baseline on `[is_clue_cell, p_att, col_other, diag1_other + diag2_other]`. |
| N-Queens model & training budget | `d_model=48`, `in_proj(7->48)`, `prob_proj(6->48)`, `peer_logit_proj(4->1)` (`std=0.02`), `kernel_size=3`, `dilation=1,2,4`, `n_segments=6` (train) / `8` (eval), `n_loops=2`, `lr=3e-3`, `weight_decay=1e-4`, `batch_size=32` (`16` MinCE), `nq_steps=400` (`60` smoke) | `lines 732–811, 860, 881, 1157–1201, 1530` | Identical size-invariant backbone and step budget across all N-Queens neural arms. |

---

## 6. Gate–Text Table (`Rule 9.2` — Exact Line-by-Line Match to `evaluate_r27a_gates`)

Every boolean conjunct in `evaluate_r27a_gates` (`benchmarks/run_r27a_labelfree_infiloop_trmp.py` lines 1259–1519) is tabulated below:

| Gate ID | Sub-Condition / Metric | Exact Code Expression | Code Line | Threshold / Direction |
| :--- | :--- | :--- | :---: | :--- |
| **`R27A_D0`** | Public `train.csv` SHA256 | `hashes.get("train.csv") == SUDOKU_EXTREME_TRAIN_SHA256` | `line 1269` | Exact match |
| **`R27A_D0`** | Public `test.csv` SHA256 | `hashes.get("test.csv") == SUDOKU_EXTREME_TEST_SHA256` | `line 1270` | Exact match |
| **`R27A_D0`** | N-Queens $D_4$ solution orbit overlap | `disj["d4_orbit_overlap_count"] == 0` | `line 1271` | `== 0` |
| **`R27A_D0`** | N-Queens $10\times 10$ exact input overlap | `disj["nq10_exact_input_overlap_count"] == 0` | `line 1272` | `== 0` |
| **`R27A_D0`** | N-Queens $10\times 10$ $D_4$ input-orbit overlap | `disj["nq10_d4_input_orbit_overlap_count"] == 0` | `line 1273` | `== 0` |
| **`R27A_D0`** | N-Queens smoke-vs-test input-orbit overlap | `disj["nq_smoke_test_overlap_count"] == 0` | `line 1274` | `== 0` |
| **`R27A_D0`** | Sudoku `test.csv[0:320]` overlap | `disj["sudoku_head320_overlap_count"] == 0` | `line 1275` | `== 0` |
| **`R27A_D0`** | Sudoku smoke-vs-test board overlap | `disj["sudoku_smoke_test_overlap_count"] == 0` | `line 1276` | `== 0` |
| **`R27A_D0`** | Sudoku train-vs-test board overlap | `disj["sudoku_train_test_overlap_count"] == 0` | `line 1277` | `== 0` |
| **`R27A_D0`** | Sudoku dev-vs-test board overlap | `disj["sudoku_dev_test_overlap_count"] == 0` | `line 1278` | `== 0` |
| **`R27A_D0`** | Label-free & supervised `WatchedLabelArray` counts | `lac["TRMP_InfiLoop_Birkhoff_LabelFree"] == 0 and lac["Ablation_UnmaskedClue_CE_Only_LabelFree"] == 0 and lac["NQ_TRMP_InfiLoop_Birkhoff_LabelFree"] == 0 and lac["NQ_Ablation_UnmaskedClue_CE_Only_LabelFree"] == 0 and lac["InfiLoop_Supervised_OurTrain"] > 0 and lac["NQ_StandardCE_MultiSol_Sup_OurTrain"] > 0 and lac["NQ_TRMP_MinCE_MultiLabel_Sup_OurTrain"] > 0` | `lines 1279–1285` | `== 0` for all 4 label-free arms, `> 0` for all 3 `_OurTrain` supervised arms |
| **`R27A_D1`** | Sudoku clue preservation at $S=12$ | `mean_clue_acc_s12 >= 0.98` | `line 1335` | `>= 0.98` |
| **`R27A_D1`** | Sudoku empty-cell acc vs `Control_Random_Network` | `ci_empty_vs_rand["mean_diff"] >= 0.10 and ci_empty_vs_rand["ci95_low"] > 0.0` | `lines 1336–1337` | `mean_diff >= +0.10`, `ci95_low > 0` |
| **`R27A_D1`** | Sudoku empty-cell acc vs `Control_Untrained_Init` | `ci_empty_vs_untr["mean_diff"] >= 0.10 and ci_empty_vs_untr["ci95_low"] > 0.0` | `lines 1338–1339` | `mean_diff >= +0.10`, `ci95_low > 0` |
| **`R27A_D1`** | Sudoku empty-cell acc vs `Control_PeerClue_Uniform` | `ci_empty_vs_peer["mean_diff"] >= 0.05 and ci_empty_vs_peer["ci95_low"] > 0.0` | `lines 1340–1341` | `mean_diff >= +0.05`, `ci95_low > 0` |
| **`R27A_D1`** | Sudoku empty-cell acc vs `Control_ClueCopy_RowColMode` | `ci_empty_vs_mode["mean_diff"] >= 0.10 and ci_empty_vs_mode["ci95_low"] > 0.0` | `lines 1342–1343` | `mean_diff >= +0.10`, `ci95_low > 0` |
| **`R27A_D1`** | Sudoku empty-cell acc vs dev-selected `Control_FixedMeanField` | `ci_empty_vs_mf["mean_diff"] >= 0.02 and ci_empty_vs_mf["ci95_low"] > 0.0` | `lines 1344–1345` | `mean_diff >= +0.02`, `ci95_low > 0` |
| **`R27A_D1`** | Sudoku empty-cell acc vs `Ablation_UnmaskedClue_CE_Only_LabelFree` | `ci_empty_vs_abl["mean_diff"] >= 0.08 and ci_empty_vs_abl["ci95_low"] > 0.0` | `lines 1346–1347` | `mean_diff >= +0.08`, `ci95_low > 0` |
| **`R27A_D1`** | Sudoku unit collision reduction vs `Control_Random_Network` | `ci_coll_vs_rand["mean_diff"] >= 50.0 and ci_coll_vs_rand["ci95_low"] > 0.0` | `lines 1348–1349` | `>= 50.0` fewer collisions, `ci95_low > 0` |
| **`R27A_D1`** | Sudoku unit collision reduction vs `Control_Untrained_Init` | `ci_coll_vs_untr["mean_diff"] >= 50.0 and ci_coll_vs_untr["ci95_low"] > 0.0` | `lines 1350–1351` | `>= 50.0` fewer collisions, `ci95_low > 0` |
| **`R27A_D1`** | Sudoku unit collision reduction vs `Control_PeerClue_Uniform` | `ci_coll_vs_peer["mean_diff"] >= 35.0 and ci_coll_vs_peer["ci95_low"] > 0.0` | `lines 1352–1353` | `>= 35.0` fewer collisions, `ci95_low > 0` |
| **`R27A_D1`** | Sudoku unit collision reduction vs dev-selected `Control_FixedMeanField` | `ci_coll_vs_mf["mean_diff"] >= 5.0 and ci_coll_vs_mf["ci95_low"] > 0.0` | `lines 1354–1355` | `>= 5.0` fewer collisions, `ci95_low > 0` |
| **`R27A_D1`** | Sudoku unit collision reduction vs `Ablation_UnmaskedClue_CE_Only_LabelFree` | `ci_coll_vs_abl["mean_diff"] >= 40.0 and ci_coll_vs_abl["ci95_low"] > 0.0` | `lines 1356–1357` | `>= 40.0` fewer collisions, `ci95_low > 0` |
| **`R27A_D1`** | Monotonic recurrent refinement across $S \in \{1..12\}$ | `monotonic_empty and monotonic_coll` | `lines 1331–1332, 1358–1359` | Both true |
| **`R27A_D2`** | N-Queens clue preservation across `10x10_id`, `12x12_ood`, `16x16_ood` | `clue_acc_10 >= 0.95 and clue_acc_12 >= 0.95 and clue_acc_16 >= 0.95` | `lines 1410–1412` | `>= 0.95` on all 3 board sizes |
| **`R27A_D2`** | N-Queens `10x10_id` collision reduction vs `Control_PeerClue_Uniform` | `ci_nq10_vs_peer["mean_diff"] >= 1.5 and ci_nq10_vs_peer["ci95_low"] > 0.0` | `lines 1413–1414` | `>= 1.5` fewer collisions, `ci95_low > 0` |
| **`R27A_D2`** | N-Queens `10x10_id` collision reduction vs `Control_FixedMeanField` | `ci_nq10_vs_mf["mean_diff"] >= 5.0 and ci_nq10_vs_mf["ci95_low"] > 0.0` | `lines 1415–1416` | `>= 5.0` fewer collisions, `ci95_low > 0` |
| **`R27A_D2`** | N-Queens `10x10_id` collision reduction vs `Ablation_UnmaskedClue_CE_Only_LabelFree` | `ci_nq10_vs_abl["mean_diff"] >= 3.0 and ci_nq10_vs_abl["ci95_low"] > 0.0` | `lines 1417–1418` | `>= 3.0` fewer collisions, `ci95_low > 0` |
| **`R27A_D2`** | N-Queens `10x10_id` collision reduction vs `Control_Random_Network` & `Control_Untrained_Init` | `ci_nq10_vs_rand["mean_diff"] >= 5.0 and ci_nq10_vs_rand["ci95_low"] > 0.0 and ci_nq10_vs_untr["mean_diff"] >= 5.0 and ci_nq10_vs_untr["ci95_low"] > 0.0` | `lines 1419–1422` | `>= 5.0` fewer collisions, `ci95_low > 0` |
| **`R27A_D2`** | N-Queens `12x12_ood` zero-shot OOD collision reduction vs `Control_PeerClue_Uniform`, `Control_FixedMeanField`, & `Control_Random_Network` | `ci_nq12_vs_peer["mean_diff"] >= 1.5 and ci_nq12_vs_peer["ci95_low"] > 0.0 and ci_nq12_vs_mf["mean_diff"] >= 5.0 and ci_nq12_vs_mf["ci95_low"] > 0.0 and ci_nq12_vs_rand["mean_diff"] >= 5.0 and ci_nq12_vs_rand["ci95_low"] > 0.0` | `lines 1423–1428` | `>= 1.5` vs PeerClue, `>= 5.0` vs FixedMF & Random, `ci95_low > 0` |
| **`R27A_D2`** | N-Queens `16x16_ood` zero-shot OOD collision reduction vs `Control_PeerClue_Uniform`, `Control_FixedMeanField`, & `Control_Random_Network` | `ci_nq16_vs_peer["mean_diff"] >= 2.0 and ci_nq16_vs_peer["ci95_low"] > 0.0 and ci_nq16_vs_mf["mean_diff"] >= 5.0 and ci_nq16_vs_mf["ci95_low"] > 0.0 and ci_nq16_vs_rand["mean_diff"] >= 5.0 and ci_nq16_vs_rand["ci95_low"] > 0.0` | `lines 1429–1434` | `>= 2.0` vs PeerClue, `>= 5.0` vs FixedMF & Random, `ci95_low > 0` |

---

## 7. Exact Execution & Scoring Commands (`Rule 9.8`)

```bash
# 1. Dev Smoke Verification (CPU, disjoint smoke split, outputs SMOKE_NO_VERDICT):
/home/agent/.venvs/trader-cat/bin/python3 benchmarks/run_r27a_labelfree_infiloop_trmp.py --smoke-cpu --out-dir /tmp/r27a_smoke

# 2. Frozen Full L4 GPU Execution (dispatched by opus-mats):
python3 benchmarks/run_r27a_labelfree_infiloop_trmp.py --out-dir /root/runs/r27

# 3. Deterministic Scoring (Standalone R27A; refuses smoke_cpu=true by default):
/home/agent/.venvs/trader-cat/bin/python3 scripts/score_r27.py --results-dir results/r27 --subrounds r27a --out results/r27/r27a_verdicts.json
```
