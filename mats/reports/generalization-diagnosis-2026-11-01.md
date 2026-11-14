# Generalization diagnosis: why energy plus search does not learn a size-general algorithm

For Kelvin, 2026-11-01. opus-mats, generalization-diagnosis workflow, final report.

Built from `runs.md`, `theory.md`, `diagnosis.md` and the review `critique.md`, all in `/home/agent/workspaces/mats/work/generalization-diagnosis/`. Every review finding is applied or rejected in §2.5. Checked at 2026-11-01 22:20Z: I re-ran the review's main measurements from the stored run outputs and added three new checks: single-move local minima and recall against the supervised targets (§2.3), and a calibration of the Sudoku search test (§2.4). Scripts are listed in §6. Read-only everywhere; nothing was run inside any repo.

Labels: [K] read in the cited file, code or paper. [M] measured from stored run outputs, re-run here unless marked otherwise. [I] my inference. [C] conjecture, untested. Run IDs (E1, S2, ...) follow `runs.md`.

## TL;DR

1. In R27A N-Queens, the only energy run tested on bigger boards (trained at 10×10), the network learns a local rule that holds up with size: its per-row error relative to a zero-parameter control is as low at 16×16 as at 10×10. Valid boards still fall from 22% to 6% to 5% at 10, 12 and 16, because its outputs usually stop one or two collisions short and nothing after it can finish.
2. Most likely reason: what is learned is a fixed number of local updates, and they end in near-solutions that no single move fixes (78 of 82 one-collision outputs); the pipeline has no backtracking, repair or restart, learned or hand-written. The boards themselves are easy: textbook min-conflicts on the same energy solved every fresh 2–3-clue board we tried, up to 32×32.
3. Two things were never tested: Sudoku at a new size (our 9×9 nets cannot run at other sizes, and they fail at 9×9 already, 0 of 256 solved), and a learned search over our energy at a bigger size. The one learned search we did test at a bigger size, Searchformer, mostly fails before it searches, usually by miscopying the start cell of 12×12 mazes.
4. Cheapest test (existing code with small edits, CPU only, about an hour on 6 cores): retrain R27A with and without its four coordinate inputs on fresh boards, and measure it against the control out to 24×24. If the ratio stays flat, the drop comes from the missing finishing step, and the next test is a repair policy trained at 10×10 and run at 16–32 against plain min-conflicts.
5. Bach's guarantees assume test data come from the training distribution, and without an assumption linking the two he says none can be obtained (pp. 23, 103), so size transfer is outside the book. After Ch. 2, read §12.1 (which of many equally good fits training picks) and Ch. 13 (a learned score plus a decoding step).

## 1. The runs

Runs that test bigger instances:

| Run (date, owner) | What is learned, and what searches | Train → test | Result | What it can show |
|---|---|---|---|---|
| E1 R27A-D2 N-Queens (10-20; gemini design, opus-mats ran it, audited) | Label-free looped conv net that minimizes a hand-written collision energy. No search: 8 segments × 2 updates, then a parallel argmax | 10×10, 160 boards, 400 steps → 12, 16 | Valid boards 21.9 / 6.2 / 5.0% (seed 101); rows conflict-free 0.81 / 0.81 / 0.78 (3 seeds) | The only energy run with a bigger test |
| E6 R22B-E1 SAT (10-19, gemini) | GLC_AOM_v2 (about 1.2M parameters) with neural tree search, B ≤ 64 | n = 20 → 50; 24 optimizer steps in total | Solved 41.7 → 0%; clauses satisfied 98.95 → 97.78% | An untrained scaffold on home-made benchmarks |
| E7 P12 mazes (09-29, calibrated-ai) | (a) DT proposer plus flow BP; (b) CostNet: learned cell costs plus an exact shortest-path solver | 9×9 → 15–65 | (a) fails its own 9×9 gate; (b) exact 78 → 9%, all paths valid (1 seed) | Arm (b) only |
| E8 FU-B2-1 CostNet+BB (09-30, calibrated-ai) | Learned cell costs plus an exact solver | 9×9 → 13, 33, 59, 201 | Exact 99.8 / 58.7 / 25.1 / 2.0% (20 seeds) | Yes; the run measured its own failure mechanism |
| S2 Searchformer run010 (10-27, astra) | 6.25M-parameter transformer imitating A* traces, then the plan | Size 10, 65,536 updates → 12, 20 | Shortest plans 57 / 2 / 0 of 64 | The only learned search with a bigger test |
| R1 P13 Trader2D (09-30, calibrated-ai) | Per-pixel market ResNet; no search | 9×9 → 19–99 | 0.93–0.99 at 9; at most 0.04 at 33; 0 at 99 | Yes |
| R2 P18 detached blocks (09-30, calibrated-ai) | Four untied recurrent blocks | 9×9 → 19–59 | 0.017 at 59, against 0.081 for DT-Recall at equal compute | Yes |
| P1 DT-Recall reproduction (09-30, calibrated-ai) | Official recurrent net with recall and progressive loss | 9×9 → 59, 201; 32 → 512-bit prefix sums | 59×59 by seed 9.1 / 99.8 / 87.2% (best iteration); 201×201 0 / 35 / 0% on the first 20 mazes; 512-bit ≥ 98.97% | The positive control: a replicated success that depends on the seed |

Runs that fail at the training size, or have no result:

| Run (owner) | What is learned, and what searches | Data | Result | What it can show |
|---|---|---|---|---|
| E2 R27A-D1 Sudoku (gemini design) | Label-free looped message-passing net with a Birkhoff permutation loss; no search | 9×9 only | 0/256 solved; empty-cell accuracy 0.380, against 0.508 for a 4-weight hand-set control | The net cannot run at another size |
| E3 expaomppl (gemini) | Any-order nets; up to 32 lookahead branches weighted by the net's inconsistency; no backtracking | Sudoku 9×9; N-Queens 10 → 8 | Sudoku 0%; N-Queens valid 0.29–0.53 at 10, 0.68–0.86 at 8 | Only a smaller board |
| E4 exptrustamp (gemini) | E3 nets with a supervised base, plus a trust head; B = 16 | As E3 | Sudoku 0%; the search does not raise N-Queens valid boards (0.53 base, 0.48–0.52 with search) | The trust head games itself |
| S1 R4-E2 nets (calibrated-ai) | Supervised any-order transformer with a learned solvability check; rollouts and beam | Own 9×9 boards → Sudoku-Extreme 9×9 | Witness rate 100% → 32–47% | Harder boards at the same size |
| E5 R28A (gemini) | Preregistration only | - | Never run; its smoke numbers cannot be checked | Nothing |
| N1 general-search lane (kimi-research) | Imitation of exact DPLL and BFS decisions | 4×4 and 6×6 Sudoku, each size on its own | No learned search result | Nothing yet |

Every row: `runs.md` §1–2 and the files in §6 [K].

## 2. Diagnosis

### 2.1 What the runs can answer

- No run trained Sudoku at one size and tested another [K, `runs.md` §3]. The E2 and E3 Sudoku nets have 81 position embeddings and a 9-way digit head, so they cannot run anywhere else [K, code]. The general-search lane's token interface can run one network at 4×4 and 6×6, with cell features normalized by board size, but its learned 4×4 policy is below a uniform legal-move control (0.717 against 0.742 valid) [K].
- No run trained a search over our energy and tested it at a bigger size. In E1 nothing searches. In E3, E4 and E5 the search is a bounded, hand-written lookahead that does not backtrack to completion. E6's tree search sits on a net trained for 24 optimizer steps [K]. Searchformer (S2) is the only learned search with a bigger test.
- E1 is the only energy run with a bigger test; E3 and E4 tested N-Queens only on a smaller board (8×8) [K].
- The clean size tests, where the training size is solved, are the maze runs (E8, R1, R2, P1) and S2 [K].

### 2.2 Ranked mechanisms

Ranked by how directly the evidence bears on size transfer, not by how many runs mention a mechanism (review finding F3). The codes H1–H9 follow `diagnosis.md`.

| # | Mechanism | Status | Evidence |
|---|---|---|---|
| 1 | Stuck next to a solution: a fixed number of learned updates ends in states that no single change improves, and nothing finishes the board (H1) | Stuck states measured in E1; the causal link is [I]. A property of the pipeline, since nothing learned searches | Local quality against controls is flat from 10 to 16; 78 of 82 one-collision outputs cannot be improved by any single queen move; invalid boards are a median of 4 queens from valid; the readout is a parallel argmax after 8 × 2 updates, with no backtracking, repair or restart [M; K code] |
| 2 | Which of several equally good fits training lands on (H2b) | Measured in the maze lane | Three seeds of the official DT-Recall recipe fit 9×9 (99.84–99.96% training accuracy) and score 9.1 / 99.8 / 87.2% at 59×59; P13's DTR-eq seeds score 0.075 / 0.925 / 0.23 / 0 [K] |
| 3 | A learned penalty fitted to the trade-offs of the training size (H4) | Measured by the run, E8 | Walls priced at 201–358 times an open cell; on 59×59 (seed 200) exact 100% for paths under 100 blocks, 59% at 200–300, 0% beyond 500 [K] |
| 4 | Bigger prompts read wrongly, cause unknown (S2) | Symptom measured | At 12×12 the start cell is copied right in 27/28 mazes with start row ≤ 3, 5/25 with rows 4–9 and 0/11 with a coordinate ≥ 10, at any prompt length. The first event is wrong in 44/64. The 4,096-token cap does not bind: the longest correct trace has 2,351 tokens [M]. Candidates: the size field "1 2", never seen in training (H3); two-digit numbers in coordinate slots (H6) |
| 5 | Unstable long iteration (H8) | Measured in R2 | Period-4 cycles on 69% of 59×59 mazes, against 2.5% for DT-Recall [K] |
| 6 | Too little training (H2a) | Untested | E1: 400 steps on 160 boards, where TRMP trains on 13,442 boards [K]. S2: 65,536 of the paper's 400,000 updates, on a cosine schedule set for 400,000, so the learning rate was still about 0.94 of peak [K; I for 0.94]. E6: 24 steps. E2's supervised arm sits at chance [K] |
| 7 | Inputs whose meaning changes with size (H3) | Present, untested | E1: 4 of 7 input channels are coordinates `linspace(0, 1, n)` and their sine and cosine. Every pass starts from uniform probabilities, so in the first segment the row-maximum feature is 1/16 at N = 16, below the smallest value possible at N = 10 [K code]. No damage to local quality is visible up to N = 16 [M] |
| 8 | Training data missing a needed structure (H6) | Inferred | No long detours in 9×9 mazes (E8); no coordinate ≥ 10 in size-10 mazes (S2); hard chains rare in S1's training boards [K for the facts; I for the mechanism] |
| 9 | Hand-written parts doing the work (H5) | A confound, not a cause | E1's hand-written mean-field step in every segment; E3's forward checking; E5's exact legality features; E6's legal-move masks; E8's exact solver [K code] |
| 10 | An objective that rewards a shortcut (H7) | Only in E4 | E4's trust head, trained on its own inconsistency, games it (calibration error 0.10 → 0.45); trained on labels it does not [K]. The E1 and E2 losses are zero exactly on valid boards [K code] |
| 11 | Fixed depth by itself (H9) | Contradicted where tested | R1 and R2 fail at T = 12n iterations [K]; E2 has no size shift; untested in E1 |

### 2.3 N-Queens (E1) in numbers

Seed 101 unless marked, on 64, 48 and 40 test boards; every row is [M] except the per-row ratio. The 12×12 and 16×16 pools are the lexicographically first 34 and 13 solutions per starting column (runner `solve_all_nqueens`) [K], so per-size differences are provisional (F11).

| | N = 10 | N = 12 | N = 16 |
|---|---|---|---|
| Valid boards | 14/64 | 3/48 | 2/40 |
| Outputs with exactly one collision | 40 | 28 | 14 |
| ... of which no single queen move lowers the collision count | 39 | 26 | 13 |
| Invalid boards: median queens to move to reach a valid board (maximum) | 4 (6) | 4 (6) | 4 (6) |
| Per-row collisions, network ÷ zero-parameter control (3 seeds) [K, computed] | 0.148 | 0.133 | 0.145 |
| Exact repair distance, network ÷ clue-elimination start, 95% CI | 0.64 [0.53, 0.74] | 0.67 [0.58, 0.75] | 0.52 [0.46, 0.58] |
| Min-conflicts from a random start: runs solved | 320/320 | 240/240 | 200/200 |

What the table says:
- Relative to controls, the learned local rule holds with size on both low-noise measures [M]. The draft's "some of what was learned is lost at larger N" came from noisy move counts (F2): my repair medians from the network output, 76 / 128 / 107 moves, differ from both earlier runs (59.5 / 118 / 121.5 and 73 / 146 / 138).
- Valid boards fall because a nearly size-stable per-row error meets more rows, and the outputs end in states that no single move improves [M for the facts, I for the cause]. If collisions were independent, 39%, 28% and 15% of boards would have none; the observed 22%, 6% and 5% are lower because the outputs pile up at exactly one collision [M].
- Whether more updates would move the network out of those states is untested; the draft's 64-segment readout was never run.
- The learned parts add something beyond the hand-written ones: per-row collisions are 0.095 for the network and 0.975 for the fixed mean-field control at N = 10 (3 seeds) [K, computed]. But every segment also adds a hand-written mean-field step with four learned weights (`peer_logit_proj`, F10), and how much of the size stability comes from that path is unknown.
- The valid N = 10 boards show no recall beyond chance [M]. Of the label-free arm's 14, 4 are training-orbit solutions, against a base rate of 0.47. Of the supervised arm's 12, 9 are among its own training targets, against a base rate of 0.68; its targets included 127 test-orbit solutions, so training orbits are the wrong reference for that arm (F13). With 12–14 boards and base rates this high, though, N = 10 has little power to separate recall from computation; the bigger sizes are the real test.
- Hardness does not explain it either. N-Queens completion is NP-complete in the worst case [K, Gent et al. 2017, via `theory.md`], but on this family min-conflicts solved 450 of 450 runs on fresh boards at N = 10 to 32, in a median of 153–242 moves [M] (F9).
- What E1 cannot say: whether a learned finishing step would transfer, since nothing learned searches (F1). Experiment 2 asks that.

### 2.4 Sudoku at 9×9 (E2, E3, S1)

- E2 solves 0 of 256 boards. Its empty-cell accuracy is 0.380, against 0.508 for a 4-weight hand-set mean-field control, and 59% of its units are full permutations [K]. More segments lower collisions from 21.9 to 12.5 while accuracy moves from 0.367 to 0.380 [K]: the dynamics settle into consistent-looking wrong fillings [I].
- The energy is right: the loss is zero exactly when every unit is a permutation and no digit clashes with a clue in a peer cell [K, code]. The failure is in reaching the minimum [I].
- The training recipe is suspect: the supervised arm, trained the same way for 600 steps, sits at chance (0.112) [K].
- The network's digits do not help a complete search, and the test is sensitive enough to show it. A depth-first search that branches on the cell with the fewest legal digits needs, as a median per-board ratio against natural digit order, 1.05 with the network's digit tried first and 1.01 with the hand-set control's. Noisy copies of the true digits at accuracy 0.5, 0.7 and 0.9 give 0.64, 0.40 and 0.04 (64 boards) [M]. So at the cells where the search must choose, both networks' first choice is no better than trying digits in order [I]. This rejects review finding F7.
- E3's label-free arm gets 0.376 accuracy, against 0.368 for a zero-parameter control [K].
- S1 is a difficulty shift at the same size. On its own boards it is exact; on Sudoku-Extreme its failed rollouts are internally coherent and wrong, with 7.5–7.7 of 9 boxes clash-free and 0.35–0.37 of 9 correct [K].

### 2.5 Review findings and what I did with each

| Finding (severity) | Verdict | Re-checked here |
|---|---|---|
| F1 (High): repair success says nothing about the network, and the draft's repair test cannot fail | Applied. Rank 1 is stated as a property of the pipeline; the repair arm is dropped as a test; Experiment 2 tests a learned search instead | Min-conflicts from a random start 320/320, 240/240, 200/200. The 5 failures from the network output at N = 12 are all board 8, whose clue the network overwrote; with clues reset, 240/240 |
| F2 (High): the move-count ratios are noise; low-noise measures are flat | Applied | Distance and per-row ratios reproduced (§2.3); my move medians differ from both earlier runs |
| F3 (High): the ranking counted same-size and untrained runs, and the maze budget comparison is confounded | Applied: two run tables; H2 split into H2a (untested) and H2b (measured) | P18: at 1.7 L4-h the α = 0 and α = 0.5 arms did not fit 9×9; the 8% arm ran 60,895 steps, the 31% arm 30,456 steps with another loss |
| F4 (High): S2 is misfiled, is mostly not about two-digit coordinates, and the draft's padded-maze variant changes three things at once | Applied | Row split 27/28 against 5/25, and the same pattern at the three earlier checkpoints; size-10 prompts begin `size 1 0`; 54 of 64 size-10 outputs contain a two-digit cost; the schedule was set for 400,000 updates |
| F5 (High): no test of E1's coordinate inputs; uniform 1/n first-segment features; uninformative nulls; the longer-training test reuses 160 boards | Applied, with two corrections. Of the first-segment features, only the row maximum (1/n) is outside the N = 10 range; the column and diagonal sums can occur at N = 10 in later segments. The review's fix, feeding n·p, would make a one-hot row read n, so Experiment 1 zeroes the first segment's probability features instead | Runner code at `c15146ca`, l. 759–809 |
| F6 (Medium): "every Sudoku net is 9×9-only" is false | Applied | The GS cell features are normalized by size; learned 0.717 against uniform 0.742 |
| F7 (Medium): the Sudoku search-node test is insensitive | Rejected. I calibrated it as the review suggested, and the calibration shows the test is sensitive (§2.4) | Node ratios for predictors of known accuracy |
| F8 (Medium): several labels contradict their definitions | Applied: H7 dropped for E1 and E2, H9 not counted for E2, H4 not counted for S1 | The E2 loss is zero only on all-permutation, clue-consistent fillings (code) |
| F9 (Medium): NP-completeness does not bear on these boards | Applied | 450 of 450 runs on fresh boards up to N = 32 |
| F10 (Medium): E1's forward pass contains a hand-written propagation step | Applied, under H5 | `peer_logit_proj = nn.Linear(4, 1)` over clue cell, attacked-by-clue, column mass and diagonal mass |
| F11 (Medium): the N = 12 and 16 pools are lexicographic and correlated | Applied: per-size numbers marked provisional; Experiment 1 uses fresh pools | `solve_all_nqueens(12, max_sols=400)` and `(16, max_sols=200)` |
| F12 (Medium): the draft's scripts were not saved | Partly applied: my scripts are kept in scratch (§6), since this task lets me write only this report | - |
| F13 (Low): the recall check could hardly fail on the label-free arm; the supervised arm is the informative one | Applied, with a correction: the supervised arm must be checked against its own training targets (§2.3) | 474 of the 724 solutions were supervised targets, 127 of them from test orbits |
| F14 (Low): counting slips; m = 213 clauses at n = 50 | Applied | Generator code, `round(4.26 × 50)` |

## 3. The learning theory in plain words

### 3.1 What the book proves, and for what

A learner gets examples drawn independently from one fixed distribution and picks a function from a class by minimizing its average training loss, possibly plus a penalty. The theory bounds how much worse that function does on new draws from the same distribution, and splits the gap into three parts: approximation error (the best function in the class is imperfect), estimation error (finite data, controlled by quantities such as Rademacher complexity) and optimization error [K, Bach §2.1–2.4, §4.2–4.5 and Ch. 5]. The same-distribution assumption is stated at the start: training pairs are "independent and identically distributed (i.i.d.) with the same distribution as the testing distribution" (p. 23). On mismatched distributions the book says: "If no assumption is made on the proximity of these two distributions, no guarantee can be obtained" (p. 103) [K, checked in the author's PDF].

### 3.2 Why a bigger board is outside it

A 16×16 board has more variables than a 10×10 board, so it lies in a different input space and has probability zero under the training distribution [I]. The tools the book mentions for mismatched distributions, such as importance reweighting, need the two distributions to overlap, and here they do not [I]. So the book's bounds say nothing about our question in either direction.

### 3.3 What decides behavior at a new size

- Many functions fit the training size equally well, and they can disagree at other sizes. Yehudai et al. (ICML 2021, Thm 5.1) construct, whenever the test graphs contain a local pattern absent from training, networks that are perfect on the training distribution and arbitrarily wrong at test [K, via `theory.md`]. The theorem shows such fits exist; it does not show that training finds them.
- Which fit training lands on is set by the architecture, the optimizer and the data mix (the implicit bias). Once the training size is covered, more data of that size cannot choose between fits that differ only at other sizes [I]. Bach §12.1 poses the question for ordinary fitting: "The main question is: which one?" (p. 344) [K, checked]. Abbe et al. (ICML 2023) make it the central question for inputs never seen in training, and prove for Boolean targets and some simple architectures that gradient descent picks the lowest-degree interpolant [K, via `theory.md`]. Our maze lane shows the effect: three DT-Recall seeds fit 9×9 equally well and score 9%, 100% and 87% at 59×59 [K].
- The one clean positive theorem the theory step found is Huang et al. (ICLR 2025, Thm 7). An idealized learner that fits every input length up to n/2 exactly, and picks the simplest transformer by a stated complexity measure, generalizes to all lengths if and only if the target can be written as a "Limit Transformer" that uses position only in restricted ways (periodic and local) [K, via `theory.md`]. It needs training at every size up to half the test size.
- The published cases that do transfer share four features [I, my reading of the cases in `theory.md`]:
  - the same local computation at every size, with no input that depends on n (TRMP: "no part of the model depends on N" [K, checked]);
  - a learned step that is simple, close to linear once the architecture supplies the nonlinearities (Xu et al., ICLR 2021) [K, via `theory.md`];
  - compute that grows with the instance at test time and stays stable when run longer (DT-Recall: 97.30 ± 0.68% at 59×59 with the progressive loss, 82.72 ± 15.14% without it [K, checked]);
  - training data that make size-specific shortcuts useless: several sizes, and a balanced spread of difficulty (Saparov et al., ICLR 2025: a model trained on lookaheads up to 12 "was able to generalize to lookaheads 13 and 14 but not on any larger lookaheads" [K, checked]).
- Even TRMP's single pass falls to 0.63 ± 0.35 board accuracy at 16×16 after training at 10×10; it is near-perfect up to 20×20 only when a verifier picks among sampled passes [K, checked].

### 3.4 What this predicts for our runs [I]

- R27A breaks the first and third features by construction: 4 of its 7 input channels are normalized coordinates, every pass starts from uniform probabilities 1/n, and it runs the same 8 × 2 updates at every N [K, code]. The theory predicts damage from these inputs at some size. Our data show none in local quality up to N = 16, so the prediction is open at larger N (Experiment 1).
- Training at one size gives the learner no reason to prefer the size-general fit over a size-specific one; Huang's argument needs every size up to n/2. Expect a large seed-to-seed spread at bigger sizes, as TRMP reports and our mazes show. E1's valid-board numbers come from one seed.
- The label-free energy is exact (zero only on valid boards), so in Du et al.'s terms (IREM, ICML 2022) the representation half is settled by hand, and what remains is the minimization [I, after `theory.md` §2.7]. That matches §2.3: the minimizer stops in states that single moves cannot improve.

### 3.5 What to read next in Bach

At 5% you are about at the end of Chapter 1 [I]. For this question:

| Order | Where | Why |
|---|---|---|
| 1 | Ch. 2, Introduction to Supervised Learning (pp. 21–43), all of it | Defines generalization: the i.i.d. assumption (p. 23), the no-free-lunch theorems (§2.5, p. 38) and what is out of scope (§2.7, p. 40). Here you can see which assumption size transfer breaks |
| 2 | Ch. 4, Empirical Risk Minimization, §4.1–4.5 (pp. 71–102), with the list of extensions on pp. 102–103 | What the bounds control (risk under the training distribution); §4.1 on surrogate losses and when they are calibrated for the true loss; the "no guarantee" remark on mismatched distributions |
| 3 | Ch. 12, Overparameterized Models, §12.1 (pp. 344–355) | Implicit bias of gradient descent: with many minimizers, which one it picks. The book's nearest point to our question |
| 4 | Ch. 13, Structured Prediction, §13.2–13.3 (pp. 387–393) | A learned score function plus a decoding step, and when a smooth surrogate loss is consistent with the true loss. Our system is a learned score (the energy) plus a decoder (the search); the book analyzes this pairing for a fixed output space |
| 5 | Ch. 7 §7.1–7.3 and Ch. 9 §9.1–9.5, then §12.4 (p. 375) | Background for the kernel (NTK) regime in which Xu et al. 2021 prove how ReLU networks extrapolate: linearly along rays, outside the data |
| Low | Ch. 3, 5, 6, 8, 10, 11, 14, 15 | Little bearing on size transfer |

Page numbers are those of the author's PDF dated 26 May 2025 (www.di.ens.fr/~fbach/ltfp_book.pdf). If your copy is the print edition, go by chapter and section numbers; I could not check its pagination.

### 3.6 What the book does not cover

1. Test data from a different distribution, beyond the remark that no guarantee is possible without assumptions (p. 103) [K].
2. Inputs whose size varies, with one set of weights for every size. Invariance appears only as translation-invariant kernels (§7.3), and CNNs, GNNs and transformers get a short pointer in §9.7 [K, via `theory.md`].
3. Test-time computation that grows with the instance: iteration, search, halting. A predictor in the book is a fixed function; Ch. 13's decoding step is the nearest thing, for a fixed output space [I].
4. How hard the task family is to compute. The book's lower bounds are statistical, or for first-order convex optimization (§15.2) [K].
5. Label-free training: unsupervised learning is listed as out of scope (§2.7) [K, checked].

The partial theory that exists is in papers: Abbe et al. 2023 (what gets learned on unseen inputs), Xu et al. 2021 (how ReLU networks extrapolate), Yehudai et al. 2021 (bad fits exist), Huang et al. 2024 (identification under a simplicity bias), and Levin & Díaz 2024 (networks defined at every dimension often do not extend correctly, plus a compatibility condition under which they do) [K, via `theory.md`]. None of the papers the theory step read covers a learned energy, a search and size transfer together [I].

## 4. The cheapest discriminating experiments

All three are proposed, not run. All run on CPU at `nice -n 19 ionice -c3`, from scratch copies of the code, never inside the `inconsistency-search` tree or another agent's directory. Marginal cost $0.

| | Tests | Code | Compute | Needs |
|---|---|---|---|---|
| 1 | Whether E1's local rule stays size-stable past 16, and the effect of coordinates (H3), training budget (H2a) and several training sizes | Existing, copied, with small edits | About 3–4 CPU-hours, about 1 hour wall on 6 cores | Nothing |
| 2 | Whether a learned search step transfers | New | About 1–2 CPU-hours [I] | About half a day of agent work [I] |
| 3 | What breaks Searchformer at 12×12: the size field (H3) or two-digit coordinates (H6) | Existing checkpoint | About 1–2 CPU-hours [I] | astra's OK to load the checkpoint |

### Experiment 1. R27A retrained, with and without size-indexed inputs, on fresh boards

- Arms, 3 seeds each:
  - A0, the frozen recipe (160 fixed boards, 400 steps), which reproduces E1;
  - A1, the same net trained for 4,000 steps on fresh boards every batch (free, since the loss needs no labels), with N = 10 boards built from training-orbit solutions only;
  - A2, as A1 without the four coordinate channels, and with the first segment's probability-derived features set to zero instead of computed from a uniform start, so the first segment sees only clue features, which mean the same at every size;
  - A3, as A2 with the training size drawn from {8, 9, 10, 11} for each batch.
- A2 keeps the softmax over n columns, so a diffuse row still reads about 1/n in later segments. A per-cell queen state, as in TRMP's factor graph, would remove that, but it is a larger change [I].
- Test: 100 fresh boards per N at 10, 12, 16, 20 and 24, with 2–3 clues taken from random solutions; the N = 10 boards come from the test orbits.
- Measures, with 95% bootstrap intervals over boards: per-row collisions ÷ the zero-parameter control, and exact repair distance ÷ the clue-elimination start where branch and bound finishes. Valid boards are secondary. A 64-segment readout is also secondary and is read only if it wins, since without progressive-loss training a null says nothing (F5).
- Decision rules, fixed before running. "Flat" means the N = 24 ratio lies inside the N = 10 ratio's 95% interval; "better" and "worse" mean non-overlapping intervals.
  - A0's ratio flat out to N = 24: the learned local rule transfers, and the valid-board drop comes from the missing finishing step (supports rank 1).
  - A0 rising at 20–24 while A2 stays flat: the coordinate and 1/n inputs break transfer beyond 16 (H3).
  - A1 better than A0 at N ≥ 16: training budget mattered (H2a). A1 worse at N ≥ 16 while better at 10: longer training drifts toward a size-specific fit (H2b).
  - A3 against A2: the effect of training at several sizes, free of the coordinate confound.
- Cost: 37,200 training steps at 0.26–0.35 s each, about 3–4 CPU-hours (step time measured by the draft at N = 10 on one thread). On 6 cores, about an hour of wall time; evaluation takes minutes.

### Experiment 2. A learned repair step, trained at 10×10 and run at 16–32

- Search: min-conflicts on the hand-written collision energy, except that the new column for the chosen row comes from a learned scorer instead of the least-conflict rule.
- Scorer: a small convolutional net over the current board (queens, clue rows, attack counts), with no coordinate inputs. With hand-made size-free features, transfer would hold by construction (H5), so the scorer should see the board itself [I].
- Training at N = 10 on fresh boards: the target is the column that the chosen row has in the nearest valid completion, found exactly by branch and bound, which is cheap at N = 10.
- Test: 100 fresh boards × 20 repeats at N = 10, 16, 24 and 32. Arms: the learned scorer, least-conflict, and a random legal column, all with the same zero-collision stop and a 20,000-move cap.
- Measure: median moves to a valid board, learned ÷ least-conflict, with bootstrap intervals.
- Reading: a saving at 10 that holds at 32 means a learned search step transferred in this toy. A saving at 10 that vanishes by 24–32 means it did not, and Experiment 1's arms point to which input to suspect. No saving at 10 means the teacher or the scorer is too weak, and nothing is learned about transfer.
- Cost: new code; compute about 1–2 CPU-hours [I].

### Experiment 3. The Searchformer checkpoint on size-10 mazes, in a 2×2

- Load run010's final checkpoint (`/var/tmp/astra/teacher-graphs-20261024/fresh-maze-throughput-010-terminal-copy-001/control/results/worker/A/checkpoint-1048576.pt`) with `torch.load(..., weights_only=True)` in a scratch copy. Ask astra first: run010's auditors chose not to deserialize checkpoints.
- On the 64 size-10 benchmark mazes, cross the size field {10, 12} with 44 extra walls at x or y ∈ {10, 11} {absent, present}. With the size field at 10 the extra walls lie off the grid; with it at 12 they pad the maze to 12×12. The correct first event is the same in all four cells, and the correct full trace is the original one in three of them. Build the token IDs directly, since the codec rejects off-grid walls.
- Score the first event in all four cells, separately for start rows ≤ 3 and ≥ 4 (at size 10 all 64 starts are copied right). Score full traces in the three cells whose correct trace is unchanged.
- Reading: if the size field alone breaks the row ≥ 4 copies (below 50% right), the size field is the trigger (H3). If the off-grid two-digit walls alone break them, it is two-digit numbers in coordinate slots (H6). If only both together do, the two interact. If neither does, the failure needs real 12×12 content, and both candidates are out.
- Cost: first events take seconds; full decodes in three cells about 45–120 CPU-minutes [I, scaled from the draft's estimate of 15–40 minutes per cell].

## 5. Open questions

1. Are E1's one-collision outputs attractors of the learned update, or only of the argmax readout? If the soft state also sits at a local minimum, more updates will not help, and the learned minimizer has inherited the energy's local minima [C]. Tracking the soft energy over 8–64 segments in Experiment 1 would answer it.
2. With a learned energy (the world-model version of your idea), what label-free signal would put its minima on exact solutions rather than near-solutions? IRED, which shapes its energy with labels, still reports low energy on partly right answers [K, via `theory.md`]. Our zero-violation check is exact and free; whether it can stand in for those labels is open [C].
3. Why does the size-10 Searchformer miscopy start rows 4–9 of 12×12 mazes (5 of 25 right) but not rows 0–3 (27 of 28), at any prompt length? Experiment 3 separates the size field from two-digit coordinates; if neither triggers it, what else in a 12×12 prompt does?
4. You mentioned Sudoku sizes. We found no run that trained Sudoku at one size and tested another (search scope in `runs.md` §6). If you remember one, where is it? If not, do you want that test, or is N-Queens enough for the size question? The cheapest Sudoku route is the general-search token interface (4×4 → 6×6), once a learner beats its uniform control at 4×4.
5. The cheap toy you asked for, a model that learns search and generalizes: DT-Recall on mazes is our only replicated learned procedure that extrapolates (9×9 → 59×59), at about 7 L4-hours per seed, and one of its three seeds failed (9.1%). Would a CPU-scale learned repair policy (Experiment 2) serve as that toy for energy plus search, or should the toy also learn the energy?
6. Which parts can transfer by construction? The pairwise attack energy of a partial placement is unchanged when the placement is embedded in a larger empty board [I, from `theory.md`], and any non-attacking placement can be included in a solution of some larger board (Gent et al. 2017 [K, via `theory.md`]); the map from clues to a completion has no such property. Does that make "learn the energy and the local propagation, search the completion" the natural split, and could Huang et al.'s argument (fit every size up to n0, prefer the simplest local update) say how large n0 must be for N-Queens? [C]

## 6. Sources

Workflow inputs: `runs.md`, `theory.md`, `diagnosis.md`, `critique.md` in `/home/agent/workspaces/mats/work/generalization-diagnosis/`.

Run files, read-only. `IS` = `/home/agent/workspaces/inconsistency-search`, read at `origin/main` `c15146ca` unless noted.
- R27A: `IS/results/RESULTS-R27A.md`; `IS/results/r27/r27a_results.json`; `IS/docs/audits/R27A-audit.md`; `IS/benchmarks/run_r27a_labelfree_infiloop_trmp.py` (l. 216–242 Sudoku loss; 259–280 Sudoku net; 523–560 test pools; 600–640 boards and targets; 697–721 N-Queens loss; 750–809 N-Queens net; 860 readout; 929–959 splits; 1103 and 1168–1199 training segments). The audit's seed-101 outputs: `/home/agent/workspaces/fable-strategy/scratch/r27a/work/{nq_splits,nq_preds101,sudoku_preds101}.npz` and `nq_rerun101.json`.
- E3, E4: `IS/results/explore/expaomppl/README.md`; `IS/results/explore/exptrustamp/README.md`. E5: `IS/benchmarks/PREREG-R28A-EQUIVARIANT-CELLDIGIT-LIR-SUDOKU-NQUEENS.md` (untracked).
- E6: `IS/results/RESULTS-R22B-GENUINE-PUBLIC-REASONING-AND-CLRS.md`; `IS/benchmarks/run_r22b_genuine_public_reasoning_clrs.py` l. 2512–2534; `IS/src/envs/public_reasoning_benchmarks.py` l. 648.
- Mazes: `IS/rounds/r4-r9/puzzles/P12/RESULTS.md`; `.../P13/RESULTS.md`; `.../P18-detached-mazes/RESULTS.md` (l. 45–56, 150–157, 190–195); `.../followups-batch2/item1-costnet/RESULTS.md`; `/home/agent/workspaces/calibrated-ai/trader-economy/dt-baseline/RESULTS.md` l. 12–24.
- S1: `IS/rounds/r4-r9/puzzles/R5-E3/RESULTS-R5-E3.md` (working tree); `IS/docs/papers/experiments-rounds-4-9-paper-v3.md`.
- S2: `AT/training_search_scale010_execution/FINAL_REPORT.md` and `CLOSING_SUMMARY.md`, where `AT` = `/home/agent/workspaces/inconsistency-search-astra-teacher-graphs-20261024/experiments/teacher_graphs/designs`; `/var/tmp/astra/teacher-graphs-20261024/fresh-maze-throughput-010-terminal-copy-001/control/results/worker/A/benchmark.jsonl` and `../ledger.sqlite` (opened read-only, immutable); codec `/var/tmp/astra/teacher-graphs-20261024/fresh-maze-throughput-010-capsule-002/experiments/teacher_graphs/designs/training_search_fresh_maze_codec/codec.py` l. 12–13, 57–63, 200–217; schedule `AT/training_search_scale011_design/primary-sources/run010-limits-excerpt.txt` (original l. 41–42).
- General search: `/home/agent/workspaces/inconsistency-search-general-search-20261024/benchmarks/run_general_search_s1.py` l. 563–640 and 851; `.../results/explore/general-search-gap-v2-five-seed-20261024/README.md` l. 15.

The book:
- Bach, *Learning Theory from First Principles*, MIT Press 2024, read in the author's PDF dated 26 May 2025 (www.di.ens.fr/~fbach/ltfp_book.pdf; local copy `/tmp/gd-theory/bach/ltfp_book.pdf`). Checked here: p. 23 (i.i.d.), p. 103 ("no guarantee can be obtained"), p. 344 ("The main question is: which one?"), §2.7, and the chapter and section pages from the table of contents.
- Which book: on 2026-10-29 at 15:40:40Z you wrote "No, I've gotten Bach already. I will study it on my schedule." (opus-mats session `52cc519b`, jsonl line 37580), replying to a list whose first item was Bach's *Learning Theory from First Principles* (line 37486). `outbox/kelvin-study-list.md` lists it under "Already in hand", and your later messages name no other book (searched through 2026-11-01). Confirmed, though you did not name the book on 2026-11-01.

Papers. The theory step looked up each arXiv record through the arXiv API and checked each claim against the full text (`theory.md`, Sources table, with versions). I re-checked the TRMP, Bansal and Saparov figures and quotes in the local copies (`/tmp/gd-theory/txt`).
- Abbe, Bengio, Lotfi, Rizk, "Generalization on the Unseen, Logic Reasoning and Degree Curriculum", ICML 2023, arXiv 2301.13105.
- Bansal et al., "End-to-end Algorithm Synthesis with Recurrent Networks", NeurIPS 2022, arXiv 2202.05826 (the 59×59 table: 97.30 ± 0.68 with the progressive loss, 82.72 ± 15.14 without).
- Du, Li, Tenenbaum, Mordatch, "Learning Iterative Reasoning through Energy Minimization" (IREM), ICML 2022, arXiv 2206.15448; Du, Mao, Tenenbaum, "Learning Iterative Reasoning through Energy Diffusion" (IRED), ICML 2024, arXiv 2406.11179.
- Gent, Jefferson, Nightingale, "Complexity of n-Queens Completion", JAIR 59 (2017).
- Huang et al., "A Formal Framework for Understanding Length Generalization in Transformers", ICLR 2025, arXiv 2410.02140.
- Levin, Díaz, "Any-dimensional equivariant neural networks", AISTATS 2024, arXiv 2306.06327.
- Salvatori, Forasassi, Toumi, de Felice, "Tiny Recursive Message Passing" (TRMP), 2026 preprint, arXiv 2610.18649 (§5.2: 13,442 training boards at 10×10; the size-transfer table of Fig. 3; verifier-picked sampling near-perfect up to 20×20).
- Saparov et al., "Transformers Struggle to Learn to Search", ICLR 2025, arXiv 2412.04703.
- Xu et al., "How Neural Networks Extrapolate", ICLR 2021, arXiv 2009.11848.
- Yehudai et al., "From Local Structures to Size Generalization in GNNs", ICML 2021, arXiv 2010.08853.

Re-check scripts (scratch, not durable): `/tmp/gdr-verify/scripts/`. `npyread.py` is a pure-Python `.npz` reader. `verify_e1.py` and `e1lib.py` cover collisions, the overwritten clue, recall, min-conflicts and exact repair distance (the distance code is the review's Appendix A.1). Also `verify_localmin.py`, `verify_fresh.py`, `verify_recall_targets.py`, `verify_e2.py` and `verify_s2.py`. All ran with `python3 -I` at `nice -n 19 ionice -c3`.
