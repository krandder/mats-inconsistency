# R27A audit (board-opus, Opus 5.5, read-only, 2026-10-20 11:49–11:58Z)

Repo at e178dde; tag freeze-R27A at ba6551d. I wrote nothing in the repo. Scripts and outputs: `scratch/r27a/work/`.

**1. Provenance: OK.**
- `git archive freeze-R27A` passes `sha256sum -c` (6/6 files), and those files are unchanged up to e178dde.
- The hashes in the JSON match the sha file. The data hashes match the HuggingFace LFS oids.
- `/tmp/r27a-run` is at ba6551d with no tracked changes. Its output JSON and log are byte-identical to the committed ones.

**2. Numbers:** all 40 table cells match the 3-seed means recomputed from `per_seed`. RESULTS-R27A.md is hand-written. `score_r27.py`, run on a copy, gives the same verdicts.

**3. Bars:** I re-checked every gate with my own `regate.py`. The verdicts match.
- D0: 17/17 conditions pass.
- D1: 15/16 pass. The one failure is empty-cell accuracy against the hand-set control (−0.128, lower CI bound −0.134).
- D2: 14/14 pass.

**4. Leakage:**
- Data flow checks out: the label-free arms train on clues only, and labels are used only for scoring.
- The label-access counter is not proof by itself, because the raw labels stay reachable from the code.
- The hand-set control's two weights were chosen on labelled dev data, yet its label count is reported as 0.
- Several sub-bars pass by construction:
  - clue accuracy: clues are inputs to the model;
  - one queen per row: the readout takes the argmax of each row;
  - D2 against the random net, untrained net and ablation;
  - D2 against the N-Queens hand-set control, which was never tuned and is worse than the clue floor (9.75 vs 6.42 collisions).
- Only the D2 bar against the clue floor is informative.

**5. D1 substance (CPU re-run of seed 101).**

| | label-free | hand-set (selected) | hand-set (other grid point) |
|---|---|---|---|
| empty-cell accuracy / collisions | 0.385 / 11.8 | 0.508 / 33.0 | 0.479 / 16.4 |
| boards solved | 0/256 | 4 | 4 |
| units that are full permutations | 0.59 | 0.18 | 0.45 |
| empty cells contradicting a clue | 0.25% | 0 | 0 |

- The model does not copy clues. It writes locally consistent but wrong permutations.
- Accuracy on conflict-free cells is 0.42, only +0.11 above picking a random clue-legal digit (0.276).
- The Birkhoff loss is a soft version of the collision metric.
- The hand-set control was selected on accuracy, yet near the same collision level it still reaches 0.48 accuracy.
- So the trade-off is mostly an artefact of the metric and of how the control was selected.

**6. D2 substance.**
- What matches between label-free and min-CE: architecture, steps, learning rate, evaluation and training inputs.
- How min-CE is handicapped:
  - batch size 16 instead of 32;
  - targets capped at 8 completions, which truncates 33% of training inputs;
  - the label-free loss is the metric itself, made smooth;
  - neither arm was tuned, and this comparison is not gated.
- OOD test pools:
  - the (0,0) issue is fixed: every starting column appears, and no test board has a clue at (0,0);
  - the pools are still lexicographic-first, and near-neighbour solutions share 7.9/12 and 10.8/16 rows;
  - the "at least 2 solutions" rule is enforced only at N=10.
- Fully valid boards (CPU re-run of seed 101; the JSON does not record this):

| arm | 10×10 | 12×12 | 16×16 |
|---|---|---|---|
| label-free | 21.9% | 6.2% | 5.0% |
| min-CE | 18.8% | 2.1% | 0% |
| standard CE | 1.6% | 0% | 0% |

**7. Re-run:**
- The controls match the GPU run exactly.
- Label-free collisions: 0.94/1.29/1.90 (GPU 0.86/1.27/1.88).
- Min-CE collisions: 1.28/1.88/2.83 (GPU 1.28/1.63/3.70).

**8. For Kelvin's question:**
- Shown: with our training, label-free gets fewer N-Queens collisions than small supervised baselines, both in and out of distribution.
- Not shown:
  - a Sudoku win: it loses to a hand-set control with 4 weights;
  - a public-metric win: the valid-board rate is at most 22%, and the solve and valid-board bars were removed;
  - anything about properly trained supervised models: the supervised Sudoku arm failed to train (clue accuracy 0.11–0.22, cell accuracy 0.112, which is chance).

**Verdicts**
- D0: supported.
- D1 FAIL: supported. The loss is real, and the collision gain is mostly a metric artefact.
- D2 PASS: partly supported. The collision gap replicates, but the bars are weak, valid boards are rare, and the supervised comparison is handicapped.

**Summary:** The run used the frozen code and public data, and every number and verdict reproduces. On Sudoku, label-free training writes consistent but mostly wrong fillings and solves no board, so it loses to a hand-set control. On N-Queens it beats our small supervised baselines on collisions, but most boards are still invalid, so this does not show label-free training beating supervised training on public puzzles.
