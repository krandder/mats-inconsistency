# A cheap toy for learned search that generalises

Written 2026-11-01, about 23:45Z, by a subagent of opus-mats (toy-search workflow, write-up step), for Kelvin. It condenses five working files in `work/toy-search/`: `ours.md` (our Searchformer work), `survey.md` (candidate toys), `plan.md` (the pick and its plan), `pilot_result.md` (a one-hour CPU pilot) and `review.md` (a review of the pilot). I re-read the code the review cites and repeated six of its checks in four short CPU runs (results in section 4, method in section 6). No GPU, $0. Paths are relative to `/home/agent/workspaces/mats/` unless absolute.

Tags: [K] known, with the source that was read; [M] measured, by the pilot, the review or me, as marked; [I] inference; [C] conjecture, untested.

## TL;DR
- A smaller Searchformer can't give you this toy: the paper trains and tests each model at one maze size, so it has no result on bigger mazes to replicate. Our best run solved 57 of 64 new 10×10 mazes, but 2 of 64 at 12×12 and none at 20×20.
- Recommended toy: DNAR, a small graph network that learns BFS, DFS and Dijkstra by copying every step of the algorithm on 16-node graphs; its paper reports it exact on 1,600-node graphs. The code is public and it trains on CPU.
- Our CPU pilot (BFS, one seed, 69 minutes, $0) was exact on every test graph at 160 and 1,600 nodes, and on 200-node paths that need 199 steps, after training on graphs that need at most 8. Trained on 3-node graphs instead, it is exact at that size and wrong at 160 nodes, so the test can fail.
- The limit: DNAR's design already contains the "pick the lowest" step these searches rely on, so the pilot shows that copying steps carries over when the design supplies that step. The next runs test whether a network can find the search itself, by removing that help: no step labels, our consistency penalty in their place, softened states and attention.
- The full replication runs on CPU for $0: at least two days at low priority on a quiet machine, several times that at tonight's load, plus one to two days of agent work. No decision is needed now; only one later step needs GPU (DT-Recall mazes trained on a consistency energy, about $11-16), and it would come to you first.

## 1. What we tried before, and the gap

Today (11-01) you asked for "a 'toy-example-of-learning-search-and-generalizing' that we could replicate without huge cost", and said our energy-minimiser runs don't generalise "beyond the specific problems (e.g. sudoku sizes, n-queens) to others slightly bigger". My reading [I]: a small case where ordinary training on search traces makes a network run the search itself, on new instances and on slightly bigger ones, cheap enough to study and extend.

### 1.1 What we ran
From ours.md §2, which draws on astra's reports and the hub log [K]:
- gemini's "Searchformer" row (10-19, rounds R19-R25) was a 4-layer stand-in model in our own harness. It never saw A* traces or the paper's tasks.
- astra's lane (10-25 15:30Z to 10-27 05:17Z) used the public Searchformer code and its small model: 6.3M trainable parameters, A* traces on 10×10 mazes.
  - The released checkpoints reproduce the paper's example trace.
  - Runs 3-6 reused one fixed set of 256 mazes. The longest (run 6, about 720 passes per maze) fit all 256 exactly and got none of 64 held-out mazes right; on 10-26 you called this memorisation ("We have proven capacity to memorize mazes").
  - Run010 trained on 1,048,576 fresh mazes in one pass: 65,536 of the paper's 400,000 updates, about 6 A100-hours, $4.94 (a quote, not an invoice).
- The lane cost about $26 of VM time and about 38 hours of astra's session [I: ours.md, from ledger estimates].

Run010, final checkpoint, one output per maze on 64 unseen mazes per size. Only 10×10 was trained [K: re-counted today from `benchmark.jsonl`]:

| maze size | valid shortest plan | exact A* trace | ran into the 4,096-token cap |
|---|---|---|---|
| 10×10 | 57/64 | 54/64 | 4 |
| 12×12 | 2/64 | 0/64 | 45 |
| 20×20 | 0/64 | 0/64 | 51 |

### 1.2 The gap
1. The paper never tests bigger instances. It trains one model per maze size and tests at that size, and its tokeniser has no tokens for coordinates past the training size, so a 10×10 model cannot read an 11×11 maze [K: survey, `searchformer/maze.py` l. 572-575 and `config/sweep/maze_sweep_rep_nondet_small.json`]. Run010 could read bigger mazes only because astra wrote a decimal number codec. A faithful small replication would show, at best, generalisation to new mazes of the training size [I].
2. We partly replicated that. Run010 produced the exact A* trace on 54/64 new 10×10 mazes (84%, 95% CI 74-91%); the paper reports more than 90% of test mazes with no wrong token [K: ours.md §3]. Run010 stopped at about 0.93× the peak learning rate, so it is an unannealed snapshot [I: ours.md].
3. It did not carry over to bigger mazes, and trace length does not explain the 12×12 result. Of the 64 correct 12×12 traces, 54 are no longer than the longest 10×10 benchmark trace, yet the model ran into the token cap on 38 of those 54 and solved 2 [K: ours.md, from `benchmark.jsonl`]. It learned a size-10 procedure that runs away on a bigger grid, the failure you describe for our Sudoku and N-Queens runs [I]. Possible causes, none tested [C]: longer inputs and unseen grid positions; two-digit coordinates that never occur at size 10; size-specific shortcuts; the unannealed schedule.
4. We never tested the paper's own two claims: traces against plan-only training at matched data (built, then shelved), and bootstrapping to traces shorter than A*'s [K: ours.md].
5. It is not a toy: 6.3M parameters, traces of hundreds to thousands of tokens, and the full 400k-update recipe projects to 32.55 hours of training and up to $37.70 [K: astra's run011 design, via ours.md]. That is too slow for CPU and too large to read mechanistically [I].

Status [K: `br show pool-q144io` and `ls`, 2026-11-01 23:23Z]: your choice for the lane is still open ("close it | run the 400k-step replication | keep paused"; default keep paused on 2026-11-05). Run010's 16 checkpoints and per-maze outputs are on local disk, a free CPU testbed for the causes in item 3 [I].

## 2. Candidates

The survey searched papers and code (arXiv and GitHub, 2026-11-01) for settings where a network learns a search or propagation procedure and is tested on bigger instances, and the plan ranked the cheap ones [K: survey Tables 1-2, plan §1-2; published results are the papers' own, as read by the survey].

| Candidate | Published train-to-test result | Cost here | Link to our objective | Verdict |
|---|---|---|---|---|
| DNAR (Rodionov & Prokhorenkova 2024): a graph network with discrete states and hard attention that copies every step of BFS, DFS or Dijkstra | 16 to 1,600 nodes, 100% graph-exact on all three. Without the step labels, BFS is exact on 34% of graphs at 16 nodes and 0% at 64 | CPU, $0. One BFS training run: 15-70 min, depending on load [I, from the pilot's step times] | None built in; a consistency penalty can replace the step labels [C] | Chosen; pilot done |
| DT-Recall (Bansal et al. 2022): one convolutional block, run for more iterations on bigger inputs | Mazes 9×9 to 59×59: 97%. Prefix sums 32 to 512 bits: 97%. Our September reproduction (about $21): prefix sums 3/3 seeds; mazes 1 of 3 seeds at the paper's level (9.1%, 99.8%, 87.2% at 59×59) | Prefix sums 0.7-2.7 CPU-hours per run [M: survey]; mazes about 7 L4-hours per seed [K] | Close: retrain it on a label-free energy | Second pick, as the bridge. Its maze rule is dead-end filling, an elimination rule |
| Factor-graph N-Queens: TRMP's recipe (2026) against our loss and RUN-CSP's | TRMP: trained at 10×10, 63% ± 35 of 16×16 boards fully solved in one pass. Our R27A: 0-5% at 16×16 | 5-15 CPU-hours per arm; no public TRMP code | Closest (your failing task) | Later: our gap already starts at the training size, so the recipe comes first |
| Label-free constraint nets: RUN-CSP, QuerySAT, ANYCSP | Solution quality holds on 10× bigger instances, but exact solves fall (RUN-CSP 3-colouring: 2.6% at 2× the training size). QuerySAT is the exception: 3-SAT trained at 5-100 variables, 93.3% fully solved at 5-405 | RUN-CSP 1-5 CPU-hours [I]; QuerySAT and ANYCSP need GPU hours to days | Close | Later |
| Learned energies minimised at test time: IREM, IRED (Du et al. 2022, 2024) | IREM: 2-10 to 15 nodes, reported as MSE only. IRED: Sudoku stays 9×9 | 1-10 L4-hours [I] | Close (an explicit energy) | Not exact size tests |
| Neural cellular automata on mazes (2026) | 9×9 to 201×201 | No code; 1-2 days to rebuild [I] | Some | Not now |
| Min-aggregation GNN (Nerem et al. 2025) | Proven to be Bellman-Ford, so any size | CPU minutes | Some | No: the aggregation already is the algorithm |
| Smaller Searchformer, Stream of Search, maze transformers, Saparov et al., looped transformers | No size test (Searchformer, Stream of Search), zero on bigger mazes (MLM-U), or fixed depth (Saparov et al.) | n/a | Little | Out |

What the survey's evidence says [I, from survey "What the evidence says"]:
- Size transfer appears where the update is local and weight-tied and runs for more steps on bigger inputs (convolution, message passing, cellular automata). Sequence transformers with position or coordinate tokens fail on bigger instances or cannot take them as input.
- Exact answers at bigger sizes come with step supervision (DNAR), an algorithmic prior built into the architecture (min-aggregation), or a check at test time (sampling with a verifier, best of 64). Label-free energy training alone gives good approximate answers, and its exact-solve rate falls as instances get harder. DNAR, as section 3.1 shows, has both of the first two.
- No paper in the survey shows exact Sudoku transfer to bigger boards: SE-RRM goes from 9×9 to 16×16 with 0% of boards solved (52% of cells) [K: survey].

## 3. The chosen toy and plan

### 3.1 DNAR, and what it can and cannot show
DNAR [K: arXiv 2402.11628; code `yandex-research/dnar` at `12f3f0bd`, Apache-2.0] is a graph network whose node and edge states are a few discrete bits; the BFS model has 250,635 parameters [M: counted today]. It is trained on 10,000 random 16-node graphs to predict the algorithm's next state from the current one, at every step of the trace; these step labels are what the paper calls hints. At test it starts from the initial state and runs alone, and hard attention makes each node read one message per step.

Why it is the pick [I: plan §0, §3.1]: it uses Searchformer's recipe (supervise every step of the search trace) in a model small enough for CPU. Its paper also produces the failure we keep seeing, one change at a time in the same code: continuous states (BFS 0% graph-exact at 800 nodes), soft attention (wrong from 800 nodes on adversarial graphs), no step labels (34% at 16 nodes, 0% at 64), training on 3-node graphs (41% node-level at 160) [K: paper Tables 3, 4, 5 and 7, via plan §3.1]. The paper's §6 gives the learning-theory claim behind this: with few enough discrete states, and validation data that covers every local case, perfect validation implies correctness at any size; its unlabelled model instead uses its states as distance counters and fails past the depths it saw [K: plan §3.1]. Our size failures may be of that kind [C].

What it cannot show (review finding 1; I re-read the code it cites) [K]:
- The "pick the lowest" step is part of the architecture. A fixed operator flags, for each node, its lowest-position neighbour in each state; for BFS that is the parent rule, for DFS the next child (`processors.py` l. 22-42 and 127-142; BFS edges carry the sender's position, `generate_data.py` l. 46). A second copy flags the lowest-valued node in each state per graph, which for Dijkstra is the priority-queue pop (l. 123-125). A fixed comparison bit computes d(u) + w(u, v) < d(v), Dijkstra's relaxation test (l. 112-121).
- Positions and weights reach the network only through these flags and comparisons, so at test time each node's update is a lookup on a few discrete codes. The trained network is a finite table of local cases. If it handles every case correctly, it is exact on graphs of any size [I: the paper's §5 argument].
- So the result with step labels tests whether 16-node data covers the local cases. It does not test whether training finds a search procedure, which is what our Sudoku and N-Queens models must do [I]. The plan dropped the min-aggregation GNN because "the aggregation already is Bellman-Ford"; the labelled DNAR result is partly evidence of the same kind [I].
- My refinement [I]: training still has to pick the right table. The 3-node model picks a wrong one, exact at its own size and wrong at 160 nodes, which makes DNAR a clean toy of when imitation lands on a size-independent rule and when it doesn't.

DNAR stays the pick because the help can be removed one piece at a time in the same code, and those arms are where it can fail informatively [I: review, "Changes"].

### 3.2 Plan
From plan.md §3.2-3.7 [K], except where marked:
- Tasks: BFS, DFS, Dijkstra. Output: one parent pointer per node. Score: graph-exact (every pointer right) and node-level.
- Training: 10,000 connected Erdős–Rényi graphs at n = 16; AdamW at the repo's learning rate 1e-4, batch 32, 1,001 steps, sampling temperature annealed from 3.0 to 0.01. Controls trained at n = 5 and n = 3.
- Tests: Erdős–Rényi graphs at 16, 80, 160, 800 and 1,600 nodes; Watts–Strogatz and Delaunay graphs; grid mazes of 10×10 and 30×30 cells, perfect and with 5% extra openings; the adversarial K_{2,n−2}.
- Controls: no test graph seen in training; other graph families; a run at twice the step count, which for BFS follows from the fixed point and matters for DFS, whose state is not monotone [I]; the exhaustive rule check; one-change ablations; truncation at half the steps.
- Criterion, fixed before any run. R1: per task, in ≥ 4 of 5 seeds (DFS: 3 of 3), graph-exact on ≥ 99% of the 320 graphs at 160 nodes and on the largest sizes (BFS ≥ 317/320 at 1,600). R2: every BFS and Dijkstra model that meets R1 passes the rule check. R3: the 3-node BFS model scores ≤ 80% node-level at 160 nodes; otherwise the size test does not discriminate.

Changes from the review, all adopted [I]:
1. Relabel every registered test set so that node index order is independent of position. On upstream's graphs the lowest index is also the lowest position, so a model with its tie-break removed still passes (finding 3).
2. Register depth families for BFS and DFS: paths, grids and the plan's mazes. Erdős–Rényi depth grows like log n, so those tests grow width only (finding 2).
3. Run the exhaustive rule check first; for BFS it settles transfer on every graph for little compute. My addition: it must also confirm that the attention ranking is strict wherever BFS needs it, because ties fall back to index order (finding 3).
4. Before the seeds with step labels, run the arms that can fail: BFS with no step labels (paper: 34% at 16 nodes, 0% at 64); a consistency penalty in place of the labels (below); continuous states; soft attention. Two or three labelled seeds per task are then enough, so R1's "4 of 5 seeds" becomes "every seed". Proposed readout for each arm, to be fixed in the prereg: graph-exact at the training size and at 160 and 1,600 nodes, on the relabelled and depth sets. An arm reproduces our failure if it is ≥ 95% graph-exact at 16 nodes and ≤ 20% at 160.
5. Report each control at its training size and across seeds (finding 4), and hash or commit the prereg before the first run (finding 9).

Bridges to our objective, after the replication [C]:
- Consistency penalty in place of step labels (plan 3.10(b)). Keep the output loss, drop the step-label loss, and penalise predicted BFS layers ℓ unless ℓ(start) = 0, no edge spans more than one layer, and every other node has a neighbour one layer closer. Its only solution is the BFS distance [I: plan]. The question is whether it moves the unlabelled model off the distance-counting states of the paper's §6 and toward the rule that transfers.
- A* on mazes with no retraining (plan 3.10(a)). With a consistent heuristic, A* expands nodes in Dijkstra's order on the reduced costs w(u, v) + h(v) − h(u) [K: standard result], so the 16-node Dijkstra models can run on mazes from 10×10 to 30×30 and be compared with run010's 57, 2 and 0 of 64. The priority-queue pop is built in (3.1), so a pass here would show the architecture's prior at work more than learned search [I].

### 3.3 Second pick: DT-Recall on a label-free consistency energy
From plan.md §4 [K], except where marked:
- Take DT-Recall, which extrapolates under supervision and which we already reproduced, and replace its cross-entropy with an energy whose only zero is the answer.
- Stage A, CPU: prefix parity, with E = (p_0 − x_0)² + Σ_{i≥1} (p_i − [p_{i−1}(1 − x_i) + (1 − p_{i−1}) x_i])² on the network's output probabilities. Train at 32 bits, test at 48, 128 and 512; supervised and label-free arms, 3 seeds each.
- Stage B, GPU, only if stage A passes: perfect mazes, with an energy on path degrees (start and goal have one path neighbour, every other path cell two).
- Limit [K: plan §1]: the released maze net solves 7% of 33×33 mazes once 1% extra openings add loops, and its first stable iteration tracks the number of dead-end-filling rounds (correlation 0.93). Stage B therefore asks whether label-free training finds an elimination rule, which is propagation rather than search [I].
- Its CPU pilot (≤ 60 min) is designed and has not run.

## 4. Pilot result

### 4.1 What ran
From pilot_result.md and PREREG.md [K], except where marked:
- BFS only, seed 40. The upstream DNAR code is unchanged, with logged deviations: a three-function stand-in for PyG, traces padded to equal length, a sparse test path checked bit for bit against upstream's dense one, and an exact early stop once the state stops changing.
- Three models: M16 (trained at n = 16), and M5 and M3 (n = 5 and n = 3, edge probability 0.5 as in the paper's Table 7).
- 69 minutes on CPU at nice 19, one process for most of it, $0.
- The go criterion was registered before training. As the machine load rose to 25-36, the test sets were cut three times, each cut logged before the runs it governs.
- M16 stopped at step 243 of 1,001, at its deadline. Its sampling temperature there was 0.75; the schedule ends at 0.01 [M: computed from `utils.py` l. 16-20]. The 250-step stop was set after an aborted run of the full schedule validated 320/320 at n = 16 at step 251 (`logs/aborted_train_n16_official_schedule.log`, 21:28:48Z; PREREG scale-down 2, 21:37Z). That is early stopping on validation data at the training size, which is fine, but the pilot write-up did not say so (review finding 5).

### 4.2 Results
Registered sets [M: pilot; the numbers match `results/*.json`]:

| model | test set | graphs | graph-exact | node-level |
|---|---|---|---|---|
| M16 | ER, n = 16 | 320 | 320 | 1.000 |
| M16 | ER, n = 160 | 64 | 64 | 1.000 |
| M16 | ER, n = 1,600 | 16 | 16 | 1.000 |
| M16 | K_{2,n−2}, n = 160 and 1,600 | 16 and 8 | 16 and 8 | 1.000 |
| M5 | ER, n = 160 | 32 | 32 | 1.000 |
| M3 | ER, n = 160 | 32 | 0 | 0.442 |

- All three go clauses hold. G1: M16 ≥ 99% graph-exact at 160 and 1,600 nodes. G2: M5 ≥ 99.5% node-level at 160. G3: M3 ≤ 80% node-level at 160; the paper's figure is 41%.
- Bigger draws, run after the fact, agree: M16 320/320 at 160 and 64/64 at 1,600; M5 128/128; M3 0.436 node-level on 128.
- Sample size: no failures in 16 graphs bounds the failure rate at 1,600 nodes only below 17% (95% level); the 64 later graphs bring it to 4.6% (review finding 8; I re-did the arithmetic).

Checks the review added [M: review; "re-run" means I repeated it today with the pilot's code and checkpoints]:

| check | result | re-run |
|---|---|---|
| BFS rounds needed by the training graphs (n = 16) | 2-8. Graphs needing 2, 3, ..., 8 rounds: 3,972, 4,307, 1,294, 324, 78, 23, 2 | yes, identical |
| BFS rounds needed by the test graphs | 3-7 at 160 nodes, 4-6 at 1,600 | re-counted from the eval JSON |
| Depth probe: 4 paths of 200 nodes (199 rounds) and 4 grids of 30×30 (58 rounds), start at an end or corner | M16 and M5 exact at every step on all of them, and their state stops changing when BFS stops | yes, identical |
| Relabelled ER, n = 160, 32 graphs; relabelling changes the correct parent of 1,918 of 5,120 nodes | M16 is 32/32 when the parent is the lowest position and 0/32 when it is the lowest index. M16 with its tie-break flag removed: 32/32 on the registered graphs, 0/32 relabelled | yes. Graph-level identical; the ablated model's node-level is 0.663 in my runs (batch 4, 8 and 32) against the review's 0.647 |
| M3 by test size | graph-exact 320, 308, 301, 213 and 31 of 320 at n = 3, 4, 5, 8 and 16 | no |
| M3 errors, 32 graphs at 160 | visited bits all right; 2,850 of 5,088 non-start pointers wrong. Only 361 are the paper's case (previous layer, not the lowest); 1,663 point within the same layer, 743 to the next layer, 83 to the node itself | yes, identical. Also 8 of the 32 start nodes point away from themselves; 2,238 + 24 right pointers of 5,120 give the pilot's 0.442 |
| Seeds 41 and 42 | G2 and G3 hold for all three seeds (M3 at 160: 0.26-0.44 node-level); M16 at seed 41 exact on every set tried, including relabelled ER | no |

### 4.3 What the pilot shows [I]
- With step labels, DNAR's learned BFS transfers exactly in width (10× and 100× more nodes; two seeds for M16, three for M5) and in depth (199 rounds after training on at most 8; seed 40 only).
- The size test can fail, and the 3-node control fails the way our models do: exact at its training size and wrong when bigger, from 96% graph-exact at 4 nodes to 10% at 16 and 0% at 160. The paper's explanation covers 13% of its errors, so the missing local case is unknown [C: the review suggests it involves how many neighbours are active at once].
- The positive result comes mostly from the architecture (3.1). The pilot does not show that ordinary training discovers search.
- Three claims in the pilot write-up are corrected. "The model halts by itself": the state stops changing because BFS's update is monotone, so this is no evidence of a learned halt. "Tie-breaking at scale works": true, but the registered tests could not show it; the relabelled graphs do. The pilot's guess that M3 errs at nodes with two or more visited parents is refuted.

### 4.4 The review's findings: applied or rejected

| # | Finding | Decision | My check |
|---|---|---|---|
| 1 | Labelled DNAR is a finite local table with the next-node choice built in; transfer follows from covering the local cases | Applied, refined (3.1) | Re-read `models.py` l. 17-19, `processors.py` l. 22-42, 112-142, 205-206 and 319-320, `generate_data.py` l. 24-25 and 46 |
| 2 | The size tests grew width, never depth; the 2T column is filled in from the fixed point; a depth probe passes | Applied (3.2, change 2) | Re-ran the depth count and the probe; read `dnar_pilot.py` l. 295-302 |
| 3 | The registered tie-break tests could not fail; relabelled ER can, and M16 passes it | Applied (3.2, changes 1 and 3) | Re-ran on 32 graphs; read `generate_data.py` l. 24-25 and `utils.py` l. 31-33 |
| 4 | The controls hold on three seeds; M3 is exact at its own size; the paper's explanation covers 13% of M3's errors | Applied (3.2 change 5; 4.3) | Re-ran the error breakdown; seeds not re-run |
| 5 | M16 is an early checkpoint at temperature 0.75; the stop followed a validation at n = 16; seed 41 agrees; one seed sets both data and initialisation | Applied (4.1) | Computed the temperature; read the aborted log and PREREG; seed 41 not re-run |
| 6 | No leakage between sizes | Accepted | Not re-run; overlap is impossible at 160 and 1,600 nodes |
| 7 | The evaluation is correct; present the 2T column as derived | Applied (dropped from 4.2) | Read `processors.py` l. 199-206 and 283-320: step labels reach the network only through the loss and teacher forcing, both off at test |
| 8 | 16 graphs bound the failure rate only below 17% | Applied (4.2) | Re-did the arithmetic |
| 9 | PREREG.md was edited after the results; it still checks out by file times | Applied (3.2, change 5) | `stat`: PREREG 22:26:28Z, last result 22:25:21Z, plan 21:03Z, `run_lane_e.sh` 21:52Z |
| 10 | `summarize_pilot.py` takes the first matching set; a `float(loss)` warning fills the logs | Applied, for the next code change | Read l. 35-37 and the training logs |

I rejected none. I refined two. In finding 1, training still has to select the table, and M3 shows it can select a wrong one. In finding 2, a longer computation is one axis that grows with board size; whether it is where our Sudoku and N-Queens models fail is untested.

## 5. Next step and cost

Next, in this order [I: the review's "Changes", adopted]:
1. The BFS rule checker, relabelled test sets and depth families: about half a day of agent work and minutes of CPU.
2. The BFS arms that can fail: no step labels, the consistency penalty, continuous states, soft attention. Each needs a patch, since the repo has no flags for them (plan 3.11). One run is 1,001 training steps, about 15 minutes at load 7 and 70 at load 31 [I, from the pilot's measured 0.72-0.96 s per step at load 7 and 4.2 s at load 31], plus up to 50 minutes for 320 test graphs at 1,600 nodes (about 9 s each at load 27).
3. Two or three labelled seeds per task on the full schedule, then DFS and Dijkstra.
4. DT-Recall stage A (prefix parity, CPU): 6 runs, about 16 CPU-hours [I: plan §4.8].

Cost of the full replication (items 1-3 for all three tasks, with controls and family tests) [I]:
- Plan §3.8 estimates 30-45 hours of wall time at nice 19, about two days, $0, from a timing proxy.
- The pilot showed the proxy is optimistic. The real BFS training step took 2-4× the proxy's time even at load 7, mainly because traces are padded to n steps, and at load 25-31 a step took 4-7× longer than at load 7. DFS and Dijkstra were never timed with the real code.
- So two days is the floor on a quiet machine, and it could be several times that if General stays as busy as tonight. Halving the 800- and 1,600-node test counts for DFS and Dijkstra, which plan §3.8 allows, saves most of the time; the rule check carries the general claim for BFS and Dijkstra.
- Agent work: one to two days, for the patches, the rule checker, the graph families and the penalty arm.

GPU, each needing your approval; none is requested now:
- The whole DNAR set on one L4: a few hours, about $2-4 [I: plan §3.8]. Worth asking only if General stays saturated.
- DT-Recall stage B (label-free mazes): 15-23 L4-hours, about $11-16, and only after stage A passes [I: plan §4.8].
- The Searchformer 400k-step run (up to $37.70) is still your open choice in pool-q144io, default keep paused on 2026-11-05. The toy work does not depend on it.

Decision needed now: none. Items 1-4 are CPU, $0 and reversible.

## 6. Sources
- Working files, all 2026-11-01: `work/toy-search/ours.md`, `survey.md`, `plan.md`, `pilot_result.md`, `review.md`.
- Pilot: `work/toy-search/pilot/`: `PREREG.md`, `dnar_pilot.py`, `summarize_pilot.py`, `run_lane_{a..f}.sh`, `results/*.json`, `logs/`, `models/bfs_n{16,5,3}_s40_last.pt`.
- DNAR: Rodionov & Prokhorenkova, "Discrete Neural Algorithmic Reasoning", arXiv 2402.11628; code `yandex-research/dnar` at `12f3f0bd` (Apache-2.0), copied unchanged in `work/toy-search/pilot/upstream/`. Read for this write-up: all of `processors.py`, `utils.py`, `train.py`, `configs/bfs.yaml` and `configs/base_config.py`; `models.py` l. 1-40; `generate_data.py` l. 1-460; `dnar_pilot.py` l. 1-190, 220-365, 448-470 and 510-600; `summarize_pilot.py`.
- Searchformer: Lehnert et al., arXiv 2402.14083; `facebookresearch/searchformer` (tokeniser `searchformer/maze.py` l. 572-575 and the per-size sweep config), as read by the survey.
- Our Searchformer runs: astra's reports under `/home/agent/workspaces/inconsistency-search-astra-teacher-graphs-20261024/experiments/teacher_graphs/designs/training_search_*` (as cited in ours.md); run010 per-maze results in `/var/tmp/astra/teacher-graphs-20261024/fresh-maze-throughput-010-control/results/worker/A/benchmark.jsonl`, which I re-counted.
- DT-Recall: Bansal et al., arXiv 2202.05826. In-house reproduction: `/home/agent/workspaces/calibrated-ai/trader-economy/dt-baseline/RESULTS.md` (read today). Maze-rule analysis: `rounds/r4-r9/puzzles/P0/RESULTS.md` in `futarchy-fi/inconsistency-search`, via plan §1.
- Other papers in section 2: survey.md Tables 1-2, with arXiv IDs resolved through the arXiv API on 2026-11-01. The survey's paper copies are in `/tmp/toy-search-scratch/dl/`, which is not durable.
- Status checks, 2026-11-01 23:23Z: `br show pool-q144io`; `fleet-now grep searchformer` (no Searchformer work since 10-27); `ls` of the run010 directory (16 checkpoints, `benchmark.jsonl`).
- My re-runs, 2026-11-01 23:12-23:36Z, under `nice -n 19 ionice -c3` with one torch thread, using `/home/agent/.venvs/arc-noisy/bin/python -I -B` with the script on stdin, importing the pilot's module and checkpoints, and writing nothing:
  1. Training depth: upstream's sampler, `np.random.seed(40)`, 10,000 graphs at n = 16.
  2. Relabelled ER and tie-break ablation: the first 32 graphs of the seed-100 draw at n = 160, relabelled with `np.random.default_rng(7)` as in the review's appendix. The ablation sets the `best = 1` rows of `select_best_by_reciever.emb` equal to the `best = 0` rows. Scored at batch sizes 4, 8 and 32.
  3. M3 error breakdown on the same 32 registered graphs.
  4. Depth probe: `np.random.seed(5)`; 4 randomly labelled paths of 200 nodes and 4 grids of 30×30, start at an end or corner, scored with the pilot's `run_eval`.
