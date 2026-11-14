# Learned search with the help removed: DNAR's BFS, one scaffold at a time

Written 2026-11-02, finished about 06:15Z, by the toy-search follow-up lane (a subagent of opus-mats), for Kelvin. It continues `outbox/toy-search-generalization-2026-11-01.md` (the pilot: DNAR learns BFS by copying each step on 16-node graphs and is exact at 1,600 nodes, but its design supplies the "pick the lowest" step). Pre-registered before any registered run (`followup/PREREG.md`, sha256 `8a0b08fc…`, 2026-11-02T00:47:38Z; two dated amendments). Reviewed twice by gpt-6-astra: the first round found an evaluator bug, whose fix changed one cell's verdict (every number below is after the fix); the second signed off on the revised registered result once its remaining corrections were made, and they are (section 9). Two spot L4 runs, 4.6 hours, $4.02 at the list-price ceiling; both VMs are deleted. Paths are relative to `/home/agent/workspaces/mats/work/toy-search/` unless absolute.

Tags: [K] known, source read; [M] measured here; [I] inference; [C] conjecture, untested.

## TL;DR

- **No label-free cell met the registered exact-transfer bar, but one met the registered "found the search, not the tie-break" bar.** A plain recurrent graph network (D4) trained on our consistency energy alone, with no answer, hint or label, builds a valid BFS tree on every test graph in 2 of 3 seeds: 16, 160 and 1,600 nodes, the 1,598-candidate case, 1,000-node paths (999 rounds, against at most 9 in training) and 30×30 grids, with no drift after settling. Its controls fail as they should. What it does not learn is the exact tie-break.
- **Step 1, selection removed (step labels kept):** without DNAR's built-in comparator, with plain positions, the network learns the parent choice from the labels and is exact on every test graph in 3 of 3 seeds, as long as it keeps the hard argmax. With softmax attention, depth still transfers but a node with many candidates cannot set its pointer; the plain recurrent network keeps a valid search at 1,600 nodes in 2 of 3 seeds but loses the fine tie-break and long depth.
- **Step 2, final answers only:** the DNAR-style models fail even at 16 nodes under this recipe (0-118 of 320). The plain network learns BFS (318-319/320) and keeps a valid search at 160 nodes, but is at chance on every path in every seed; seed 40 is right only to depth 17-18.
- **The energy's outcome depends on the run.** At the learning rate the evaluator bug had selected (3e-4), 2 of 3 seeds learned a clock schedule instead: pointers set at the end of the 15-step training rollout and lost a few steps later, nearly right at 16 nodes and wrong at 160 (4 and 2 of 64). That resembles our failure pattern, though it misses the registered failure-mode definition (section 4). Read at a random horizon (exploratory), the clock mostly disappears.
- **Answer:** in this toy a network can find the BFS search itself from a label-free consistency energy and carry it from 16 to 1,600 nodes and from 9 to 999 rounds; an exact choice among many or close candidates came only with a hard argmax and step labels. Three seeds per cell and finite test sets: a stress-tested existence result, not a reliability estimate. No decision needed; the DT-Recall maze plan (section 8) would need about $20 of GPU and your OK.

## 1. What was run

### 1.1 The question
You asked for a cheap toy in which a network learns a search and generalises to bigger instances. The pilot gave one, with a caveat: DNAR's architecture contains the selection BFS needs (a fixed operator flags each node's lowest-position neighbour in each state), its hard attention reads exactly one message per step, and its states are a few discrete bits. Here the help is removed one piece at a time, and every model is tested at the training size (16 nodes), at 10× and 100× the nodes, and on long paths:
1. remove the built-in selection: first the comparison operators, then hard attention, then the discrete states;
2. remove the step labels: train on the final answer only;
3. replace the step labels with a consistency energy: label-free training on how far each node's state is from its neighbours' states under BFS's local rule.

For BFS the search is the layer-by-layer expansion: a node is reached one round after its first reached neighbour and points to a neighbour reached the round before. "Selection" is only the choice among several such neighbours (the lowest position). So every model gets two scores: **exact** (every pointer equals the BFS parent, with the position tie-break) and **valid** (every pointer goes to a neighbour one layer closer: a correct BFS tree under any tie-break). Valid measures the search; exact adds the selection.

### 1.2 Arms
Architectures [K: `gpu_src/arms.py`; PREREG §3]:

| | built-in comparisons | attention | state carried between steps |
|---|---|---|---|
| D0 | upstream DNAR: SelectBest flags, virtual-node flag, two comparison bits | hard (argmax at test) | discrete bits: 1 per node, 2 per edge |
| D1 | none; positions enter as plain numbers through linear maps | hard | discrete bits |
| D2 | none | softmax, at training and test | discrete bits |
| D3 | D2's trained weights | softmax | relaxed bits: probabilities fed back, never snapped |
| D4 | none | softmax | 128-d node and edge features (pre-LayerNorm), read by linear heads |

D0→D1→D2→D3 are one change each. D2→D4 is not: D4 also changes the normalisation, carries features instead of states, re-injects its inputs every step, and with step labels trains through its own rollout where D0-D2 are teacher-forced [K: code].

Supervision:
- **H**, step labels (upstream's hints at every step).
- **O**, the final pointers only (cross-entropy at the last of 15 steps, back-propagated through the rollout).
- **E**, the consistency energy only, no answer labels. With ℓ(v) the reach time read from the rollout's visited bits (the number of steps a node is not yet visited) and p_v the final pointer distribution: Bellman's equation (ℓ(v) − 1 − min over neighbours ℓ(u))², a parent term −log Σ_u p_v(u) c_v(u) where c_v(u) = 1 exactly when u is v's lowest-position neighbour reached one step earlier, and ℓ(start) = 0 with the start pointing to itself. Its only zero in the answer (ℓ, p) is the BFS answer (`arms.py selftest`). It does not pin down the procedure: ℓ sees the visited bits only through their sum, so a non-BFS trace, even a constant fractional one, can give the right ℓ. It is a hand-written specification of BFS (the min, the position order and the parent rule are task knowledge), computed from the graph, the start and the positions, never from the answer.
- **OE**, O + E (D0 and D4 only).

Without labels, D0-D2 train through Gumbel-relaxed states (the paper's no-hint recipe) and snap their states at test [K: DNAR paper §3.2]. Every model reads its pointers by an argmax over a node's edges.

### 1.3 Tests, criteria, controls
- Training: 10,000 random graphs with 16 nodes (upstream's sampler), node order shuffled so that it no longer equals position order; 2,000 steps; learning rate chosen from {1e-4, 3e-4, 1e-3} on seed 40 using 16-node validation graphs only (for E without labels); seeds 41 and 42 at the chosen rate [K: PREREG §4-5].
- Test sets, all with node order independent of position: 320 graphs at 16 nodes; 64 at 160; 32 at 1,600; K_{2,n−2} at 160 and 1,600 (the start's partner must pick the lowest of 158 or 1,598 candidates); paths of 200 and 1,000 nodes started at an end (199 and 999 rounds, against at most 8 in seed 40's training graphs and 9 in seeds 41 and 42's, where 3 and 2 of 10,000 graphs reach depth 9 [M]); 30×30 grids from a corner (58 rounds, with ties).
- A model passes iff it is exact on ≥ 317/320 at 16 nodes and on every graph of every other set, settled. A cell **transfers** iff ≥ 2 of 3 seeds pass; the same with valid in place of exact means the cell "found the search but not the tie-break"; for a label-free cell without built-in selection, either needs its M3 control to fail at 160 and its SH control to be below 99% exact there. A cell **reproduces our failure** iff ≥ 2 of 3 seeds are ≥ 317/320 exact at 16 nodes and ≤ 20% exact at 160 [K: PREREG §7].
- Stopping: snapped-state models stop at a fixed point of their state (exact); soft models when their pointers have not changed for 8 steps, confirmed by the step cap (200 at ≥ 800 nodes on random graphs, n + 16 elsewhere), then 16 more steps check for drift. A graph not settled by the cap is scored on its answer at the cap and fails [K: PREREG §6; `arms.py run_batch` after the review fix].
- These are finite stress tests, not proofs of correctness at any size. For one model, no failure in 32 graphs bounds its failure rate on that sampler below 8.9% (one-sided 95%), in 64 below 4.6%, in 8 below 31% and in 2 below 78%. The paths and grids are one topology per size with random labels and positions. All seeds see the same test graphs, so counts are not pooled across seeds, and three seeds say little about how often training succeeds [I: exact binomial bounds].
- Controls, seed 40 at the cell's rate: **M3**, trained on 3-node graphs (it must fail at 160 nodes for the size test to discriminate); **SH**, the shuffled-label model, whose supervision breaks ties by a random order of nodes it never sees (if it is still exact at 160 nodes, the tie-break comes from the architecture); untrained models.

### 1.4 Order, development, amendments, review fix
- **Order changed.** Steps 2 and 3 were run on every architecture of step 1, not only on the last one: the paper reports the soft network failing at 160-800 nodes even with labels (its Table 4), and a step stacked on a broken architecture can only fail [K: PREREG §2].
- **Development before registration** (seed 7, 400 steps, 16-node validation only; `runs/dev*/`): the energy changed twice. A one-sided Bellman rule had a stationary point (every node reached at step 1 and pointing to itself), so it became Bellman's equality; squared pointer terms saturated on the self-loop, so they became log terms [K: PREREG, "Already run"].
- **Amendments 1 and 2** (02:40-02:44Z, after the seed-40 test results, forced by the VM's 5-hour limit): a model below 317/320 at 16 nodes on both scores is not evaluated beyond 160 nodes, and the controls of a cell whose seed-40 model fails at 16 nodes are not trained. Neither changes a criterion [K: PREREG, "Amendments"].
- **Review fix** (04:43Z): soft models ran 24 steps past the cap, so an unsettled graph could settle, or have its answer read, after it. Fixed; every soft-model evaluation was re-run (`runs/eval_v2/`; the originals stay in `runs/eval/`). Under the fix the registered learning-rate rule picks 1e-3 for D4-E, not 3e-4 (16-node validation graphs with zero energy at the read-out: 314 against 307; the bug had them at 314 against 317). So D4-E seed 40 at 1e-3 (trained in phase 1) was evaluated, and its seeds 41-42 and controls were trained, after the review, with the same code and recipe [M: `results/selection_v2.json`, `prereg_hashes.txt`]. The 3e-4 runs are kept as "D4-E@3e-4", matched to the exploratory arm.

## 2. Step 1: the selection removed, step labels kept

Graphs exact and settled, for seeds 40/41/42; graphs valid and settled in brackets where they differ. The registered in-distribution clause reads the exact flag alone, which can differ from the 16-node column here when a graph is exact but unsettled [M: `results/tables.md`]:

| arm | 16 nodes (of 320) | 160 (64) | 1,600 (32) | K2 160 (8) | K2 1,600 (8) | path 200 (4) | path 1,000 (2) | grid 30 (4) | transfers |
|---|---|---|---|---|---|---|---|---|---|
| D0-H | 320/320/320 | 64/64/64 | 32/32/32 | 8/8/8 | 8/8/8 | 4/4/4 | 2/2/2 | 4/4/4 | 3/3 |
| D1-H | 320/320/320 | 64/64/64 | 32/32/32 | 8/8/8 | 8/8/8 | 4/4/4 | 2/2/2 | 4/4/4 | 3/3 |
| D2-H | 320/320/320 | 64/64/64 | 23/25/25 | 0/1/1 | 0/0/0 | 4/4/4 | 2/2/2 | 4/4/4 | 0/3 |
| D3-H | 320/320/319 (320/320/320) | 63/62/63 (64/63/64) | 17/19/18 (20/19/21) | 0/1/0 | 0/0/0 | 4/4/4 | 2/2/2 | 4/4/4 | 0/3 |
| D4-H | 320/318/313 (320/320/316) | 59/49/0 (64/64/13) | 10/0/0 (32/32/0) | 8/7/8 | 1/0/8 | 4/0/0 | 1/0/0 | 1/0/0 (3/4/0) | 0/3 |

Where it survives and where it breaks [M, I]:
- **Removing the built-in comparator costs nothing here.** D1 has no SelectBest, no virtual-node flag and no comparison bits; positions enter as plain numbers. With step labels and hard attention it is exact on every graph in all three seeds, including the K2 partner choosing among 1,598 candidates and the 999-round paths. Its shuffled-label control keeps the search (64/64 valid at 160 nodes) and loses the tie-break (0/64 exact), so the comparison was learned from the labels.
- **With softmax attention, a node with many candidates cannot set its pointer.** D2, with everything else equal, transfers depth exactly (999 rounds), and its learned comparison is exact even between candidates closer than 10⁻⁴ in position (seed 40, 16 of 16 such nodes at 1,600 nodes). It fails on the K2 partner at 160 nodes in 7 or 8 of 8 graphs per seed: with 158 candidates its attention on the right one is about 0.32 and its pointer never leaves the self-loop. At 1,600 nodes 7-9 of 32 graphs fail the same way (seed 40: 11 invalid pointers, all at nodes with 4 or more candidates) [M: `results/profile_D2_H_n16_s40_lr1e-4_none.json`, `results/k2_partner_check.txt`]. Feeding back relaxed states (D3) changes little.
- **The plain recurrent network (D4) keeps the search in two seeds but not the selection, and depth only partly.** Seeds 40 and 41 build a valid BFS tree on every 160- and 1,600-node random graph. Their learned comparison is coarse: seed 40 is exact on every node whose two lowest candidates differ in position by at least 0.01, on 99.2% at 0.001-0.01 and on 71-79% below 0.001, and 31 of the 32 test graphs at 1,600 nodes contain such a node. In depth, seed 40 is right through 199 rounds, then wrong in the middle of the 1,000-node paths (right to depth 137, 94% of nodes) and on grids from depth 14 [M: `results/profile_D4_H_n16_s40_lr3e-4_none.json`]. Seed 41 never settles on the paths; seed 42 settles on 14 of 64 graphs at 160 nodes and none at 1,600.
- Reading [I]: the "pick the lowest" step was learnable from labels. Under this recipe, an argmax over a learned score was what applied it exactly on every tested size, and the low-dimensional-state models (D2, D3) were exact on every path while D4 was not. These are observed failures, not proofs that the argmax or the state is necessary; D4 differs from D2 in more than its state (section 1.2).

## 3. Step 2: the step labels removed (final answers only)

Same layout; "-" marks sets not run under amendment 1 (the model already fails at 16 nodes) [M]:

| arm | 16 nodes | 160 | 1,600 | K2 160 | K2 1,600 | path 200 | path 1,000 | grid 30 | transfers |
|---|---|---|---|---|---|---|---|---|---|
| D0-O | 0/0/0 | 0/0/0 | 0/-/- | 0/-/- | 0/-/- | 0/-/- | 0/-/- | 0/-/- | 0/3 |
| D1-O | 0/0/0 | 0/0/0 | 0/-/- | 0/-/- | 0/-/- | 0/-/- | 0/-/- | 0/-/- | 0/3 |
| D2-O | 114/0/0 (134/0/0) | 0/0/0 | 0/-/- | 0/-/- | 0/-/- | 0/-/- | 0/-/- | 0/-/- | 0/3 |
| D3-O | 118/0/0 (143/0/0) | 0/0/0 | - | - | - | - | - | - | 0/3 |
| D4-O | 319/319/318 (320/320/320) | 45/35/41 (64/64/64) | 1/0/0 (30/32/17) | 8/8/8 | 0/0/0 (8/8/8) | 0/0/0 | 0/0/0 | 0/0/0 (2/0/0) | 0/3 |

- **The discrete-state models did not learn BFS from final answers, even at 16 nodes**, at any of the three learning rates and 2,000 steps (best: D2-O seed 40, 114/320, and the same weights run as D3, 118/320; every other run 0). They mostly do not reach a fixed point either. The DNAR paper's best no-hint model, after a search over training size, state count and temperature schedule, reached 34% graph-exact at 16 nodes [K: paper Table 3]; here D2-O's seed 40 reached 36% and every other discrete run 0%.
- **The plain recurrent network did** (318-319/320 in all seeds), and its search transfers in width: a valid tree on every 160-node graph in every seed, on 17-32 of 32 at 1,600 nodes, and for every K2 partner [M]. Its tie-break breaks as D4-H's does.
- **It breaks in depth, in every seed:** node accuracy on the paths is 0.43-0.55, chance on a path. Seed 40 is right only to depth 17 on the 200-node paths and to depth 18 on the 1,000-node paths [M: `results/profile_D4_O_n16_s40_lr3e-4_none.json`]. On the first test graphs its pointers are right within a few steps of a node being reached and stay right through step 40 on random graphs (its hidden state is still changing slightly at step 40, so this is output stability, not a fixed point); what fades with distance is the propagation [M: `results/trace_D4_O_n16_s40_lr3e-4_none.json`].

## 4. Step 3: the consistency energy in place of the step labels

| arm | 16 nodes | 160 | 1,600 | K2 160 | K2 1,600 | path 200 | path 1,000 | grid 30 | transfers |
|---|---|---|---|---|---|---|---|---|---|
| D0-E, D1-E, D2-E | 0/0/0 | 0/0/0 | 0/-/- | 0/-/- | 0/-/- | 0/-/- | 0/-/- | 0/-/- | 0/3 |
| D3-E | 0/0/0 | 0/0/0 | - | - | - | - | - | - | 0/3 |
| **D4-E** (1e-3) | 319/309/320 (320/320/320) | 38/3/61 (64/64/64) | 0/0/13 (32/32/32) | 8/8/8 | 8/8/8 | 3/4/4 | 0/2/2 | 0/0/0 (0/4/4) | exact 0/3, **valid 2/3** |
| D4-E@3e-4 | 279/315/310 (279/320/317) | 4/16/2 (4/64/4) | 0/0/0 (0/32/0) | 8/8/0 | 3/8/0 | 0/4/0 | 0/2/0 | 0/0/0 (0/4/0) | exact 0/3, valid 1/3 |
| D0-OE | 0/0/0 | 0/0/0 | 0/-/- | 0/-/- | 0/-/- | 0/-/- | 0/-/- | 0/-/- | 0/3 |
| D4-OE | 317/317/288 (320/320/295) | 14/27/2 (63/59/6) | 0/0/- (26/1/-) | 8/8/- | 8/8/- | 0/0/- | 0/0/- | 0/0/- | 0/3 |

**The registered energy cell, D4-E at 1e-3** [M]:
- **Seeds 41 and 42 find the search and carry it.** Each builds a valid BFS tree on every one of the 442 test graphs: all 320 at 16 nodes, 64 at 160, 32 at 1,600, every K2 partner, both path sizes (all nodes right through 999 rounds) and the grids. No answer changes in the 16 steps after its settlement is confirmed, on any test graph. Read at fixed steps 15, 20 and 30, on the 16- and 160-node graphs and the 160-node K2, they give the same scores, so they are not timed to the training horizon there. At 16 nodes both follow BFS's visited layers step by step [M: `runs/eval_v2/`, `results/trace_D4_E_*_lr1e-3_none.json`, `results/analog_check_v2.txt`].
- **Seed 40 carries the search in width but not all the way in depth:** valid on every random graph and K2 partner, and exact on K2, but right only to depth 142-144 on the paths (796/800 and 1,955/2,000 nodes) and to depth 5 on the grids (96% of nodes) [M: `results/profile_D4_E_n16_s40_lr1e-3_none.json`].
- **The tie-break is where it falls short.** Seed 42 is exact on 61/64 graphs at 160 nodes and 13/32 at 1,600. On the 1,600-node graphs it is right on every node whose two lowest candidates differ by at least 0.01, on 99.8% at 0.001-0.01, 84% at 10⁻⁴-0.001 and 41% below 10⁻⁴, about as well as the step-label D4. Its errors are not only between close candidates, though: on the grids it is exact on 411 of 625 nodes with gaps of 0.01-0.1 and 2,481 of 2,677 with larger gaps, and seed 41 misses 1,111 of 21,856 nodes with gaps of at least 0.1 at 1,600 nodes (3/64 graphs exact at 160) [M: `results/profile_D4_E_n16_s4{1,2}_lr1e-3_none.json`]. The learned tie-break is imperfect, not only coarse.
- **Against the registered definitions:** exact transfer 0/3; valid transfer 2/3, with its M3 control fitting 3-node graphs (320/320) and failing at 160 (node-level 0.087), and its SH control at 0/64 exact at 160. That is the registered clause for finding the search but not the tie-break [M: `results/summary.json`, `learned_search_cells`].

**The 3e-4 runs, which the bug had selected** [M]:
- Seed 41 is a valid search on every set too, but drifts on K2 after settling. Seeds 40 and 42 learned the wavefront and a clock: seed 40's visited bits follow BFS's layers exactly at every step, but on the first four 16-node test graphs only 0-11% of its reached nodes have the right pointer at steps 2-8, 94% at step 10, 100% at step 15 (the end of the training rollout), then 92%, 83% and 64% at steps 20, 30 and 40. Read at exactly step 15, an oracle stopping time, it is right far beyond the training size (64/64 valid at 160 nodes, 31/32 at 1,600); five steps later it is wrong on all 64 graphs at 160. Under the oracle-free stopping rule seeds 40 and 42 are 4 and 2 of 64 at 160 nodes [M: `results/trace_D4_E_n16_s40_lr3e-4_none.json`].
- That is our failure pattern, right at the training size and wrong at 10×, but it does not meet the registered failure-mode definition: after the fix seeds 40 and 42 are exact on 307 and 310 of 320 at 16 nodes, below the in-distribution bar [M].

**Answers plus energy (D4-OE)** behaves like answers alone: a valid search at 160 nodes in two seeds (63/64, 59/64), little at 1,600, no depth in any seed [M].

**The discrete-state models** did not learn the pointers from the energy (0/320 at 16 nodes, every seed and rate). In development round 3 the Bellman term of D0 and D1 reached zero while their pointer term stayed near 2; D2's did not reach zero [M: `runs/dev3/*.log`].

Reading [I]: the energy can teach this soft network the size-general search; whether a run lands on it or on a horizon-timed schedule depended here on the learning rate and the seed. Seed sensitivity of size transfer is not new to us: the three September DT-Recall maze seeds scored 9.1%, 99.8% and 87.2% at 59×59 (peak scores over test-time iteration counts, a different readout from ours) [K: `dt-baseline/RESULTS.md`]. A loss read once at a fixed horizon allows the clock solution; nothing in the energy rules it out.

## 5. Controls and checks

Seed 40, at each cell's rate; cells that fail at 16 nodes have no controls (amendment 2) [M: `results/tables.md`]:

| cell | M3: exact at its own size (3 nodes, of 320) | M3: node-level at 160 | SH: exact at 160 | SH: valid at 160 |
|---|---|---|---|---|
| D0-H | 320 | 0.102 | 0/64 | 64/64 |
| D1-H | 320 | 0.520 | 0/64 | 64/64 |
| D2-H | 320 | 0.263 | 0/64 | 64/64 |
| D3-H | 320 | 0.464 | 0/64 | 64/64 |
| D4-H | 320 | 0.232 | 0/64 | 1/64 |
| D4-O | 320 | 0.650 | 0/64 | 64/64 |
| D4-E (1e-3) | 320 | 0.087 | 0/64 | 0/64 |
| D4-OE | 320 | 0.143 | 0/64 | 5/64 |
| D4-E@3e-4 | 162 | 0.178 | 0/64 | 13/64 |

- **The size test can fail:** every M3 control fits its 3-node graphs (320/320) and is ≤ 0.65 node-level at 160 nodes, except D4-E@3e-4's, which does not fit its own size (162/320) and so is inconclusive as a control.
- **The tie-break came from the supervision:** every shuffled-label model is 0/64 exact at 160 nodes. That includes D0: trained with shuffled tie-breaks it ignores its built-in flag. Where the main cell transfers in width with step labels or final answers, the shuffled model keeps the search (64/64 valid); for D4-H and the energy cells the shuffled tie-break also destroys the search (0-13/64 valid).
- **Floors:** untrained models are exact on no graph (node-level 0.005-0.41) [M].
- **Energy check:** in every evaluation, no graph has zero energy without being exact [M: `results/summary.json`, `checks`].
- **Link to the pilot:** the pilot's M16 checkpoint is exact on every new set, including relabelled 1,600-node graphs, the 1,000-node paths and the grids [M].

## 6. Exploratory, not registered: the energy read at a random horizon

Designed after the registered results, at the learning rate the bug had selected (3e-4), to test whether removing the clock removes the 3e-4 runs' failure: each batch's rollout length is drawn uniformly from its BFS depth + 2 to 32 steps. The lower bound uses the true BFS depth of the batch, a label-derived quantity in the training schedule (the network never sees it), and longer rollouts also change compute and back-propagation depth. Same data, seeds, test sets and stopping rule as D4-E@3e-4 and D4-O [K: `gpu_src/queue_staging/21_explore_now.sh`, `arms.py --t_hi`]:

| arm | 16 nodes | 160 | 1,600 | K2 160 | K2 1,600 | path 200 | path 1,000 | grid 30 |
|---|---|---|---|---|---|---|---|---|
| D4-E@3e-4, random horizon | 318/320/319 (319/320/320) | 50/59/61 (64/64/64) | 1/8/5 (32/32/22) | 0/8/0 | 0/8/0 | 4/4/0 | 2/2/0 | 0/2/0 (4/4/0) |
| D4-O, random horizon | 320/319/319 (320/320/320) | 54/42/31 (64/64/64) | 3/0/0 (32/32/31) | 8/0/6 (8/8/8) | 3/0/0 (3/8/8) | 0/0/0 | 0/0/0 | 0/0/0 |

- **Energy:** every seed is in distribution (318-320/320 exact; at the fixed horizon 307-315). Seeds 40 and 41 never drift and are right on every path node (fixed horizon: seed 41 only). Seed 42 still drifts (24 of 32 graphs at 1,600 nodes, every path) and fails in depth. The K2 partner fails in seeds 40 and 42. Valid on every set: 1 of 3 seeds, as at the fixed horizon [M: `runs/explore/eval_v2/`].
- **Final answers:** the same width as at the fixed horizon and still nothing in depth in any seed [M].
- Reading [I]: it fits the clock explanation (more seeds keep their answer and reach depth) without isolating it, and it does not make training reliable. Three seeds and a post-hoc design: a lead, not a result.

## 7. What this answers, and what it does not

**Could the positive results come from memorisation, the architecture or the readout?** [I]
- Lookup of whole training graphs is excluded: the test graphs have 10-100× the nodes, and the paths need 999 rounds after training on at most 9.
- D1-H's only remaining selection help is the argmax in its attention: no comparator, plain positional numbers. Its shuffled-label control learns the search but not the tie-break. So the comparison and the propagation came from the step labels; the architecture supplied an exact way to apply a learned comparison.
- For the energy cell's search (D4-E, seeds 41 and 42), the controls show that training and its objective matter: the untrained network scores 0, the same cell at 3e-4 learned something else in two seeds, and with a shuffled tie-break in the energy the network does not even learn the search (0/64 valid). They do not exclude an inductive bias of the architecture that training then selects. The energy encodes BFS's definition (section 1.2) and the pointer readout is an argmax. What training produced in two seeds is valid BFS trees that hold 100× wider and 100× deeper, while often missing the energy's exact-parent requirement: at 1,600 nodes and on the grids those answers are not zeros of the energy (seed 41: 0 of 32 graphs at zero energy at 1,600 nodes; both seeds 0 of 4 grids).
- The stopping rule uses no oracle step count, and it matters: the 3e-4 clock seeds look right at the training horizon and nowhere else.

**The answer to "can a network find the search itself when the help is removed", for BFS** [I]:
- With step labels: it learns the selection rule itself (D1). Under this recipe, applying that rule exactly on every tested size came with the argmax, and long iteration without error with the low-dimensional state.
- Without step labels: the DNAR-style models did not train under this recipe. The plain recurrent network trains, and from final answers it keeps a width-general search but loses depth. From the consistency energy alone it found the size-general search in 2 of 3 seeds at the registered learning rate (and 1 of 3 at the other), without the tie-break.
- So yes, in this toy, from a label-free consistency energy; not reliably (seed- and rate-dependent), and not the tie-break.

**What it says about the energy program** [I, C]:
- Our kind of objective (local consistency of each node's state with its neighbours') was enough to teach a propagation that transfers in size and depth, in a 199k-parameter network trained for 2,000 steps (under 10 minutes on one shared L4). That is the cheap toy you asked for.
- Its failure mode is also visible: read once at a fixed horizon, the energy also admits a schedule timed to that horizon, which is right at the training size and wrong beyond. It shares one ingredient with the diagnosis in `outbox/generalization-diagnosis-2026-11-01.md`, a fixed number of learned updates with nothing that checks the answer is finished; it differs in that the N-Queens network stopped short of solutions, while these seeds reach the answer and then lose it. Reading the energy at random horizons, or requiring the state to stay a zero once reached, are the obvious guards [C].

**Not covered** [I]:
- A global frontier choice: Dijkstra's priority-queue pop or A*, where Searchformer's selection lives. BFS's selection is local, among a node's own neighbours.
- A learned energy: ours is hand-written and exact.
- Other recipes for the discrete models without labels (longer training, curricula, straight-through estimators); the paper needed a hyperparameter search and still did not reach a perfect model.
- An exact choice among many or close candidates without an argmax: D2 and D3 compare exactly but cannot write the pointer when many candidates share the attention, and D4 compares coarsely.

## 8. Plan, not run: DT-Recall mazes on a consistency energy

The earlier report's second pick [K: `outbox/toy-search-generalization-2026-11-01.md` §3.3; `plan.md` §4]: take DT-Recall, which extrapolates under supervision and which we reproduced in September (9×9-trained seeds solve 9.1%, 99.8% and 87.2% of 59×59 mazes; the released checkpoint 100%) [K: `/home/agent/workspaces/calibrated-ai/trader-economy/dt-baseline/RESULTS.md`], and replace its per-pixel cross-entropy with a label-free energy.

**Energy.** On P13's lattice (`to_lattice`: an n×n grid of cells coded wall, open, start, goal; each cell is a 2×2 pixel block) with y_c the network's path probability per cell [K: `inconsistency-search/rounds/r4-r9/puzzles/P13/check.py`]:
- degree: Σ_c y_c (Σ_{c' ∈ N4(c)} y_{c'} − d_c)², with d = 1 at start and goal and 2 elsewhere;
- ends: (1 − y_start)² + (1 − y_goal)²; walls: Σ_{wall} y_c²; pixel blocks: the variance of y over each 2×2 block.

At zero the path cells form the start-goal path plus disjoint cycles; a perfect maze is a tree, so the zero is the solution [I: plan §4.2]. It is the soft form of P13's exact checker `check` without its length clause, so on loopy mazes any chordless start-goal path (no two non-consecutive path cells adjacent) has zero energy, and the energy does not ask for the shortest one. The BFS energy used here (Bellman reach times plus parent pointers) would ask for it, at the cost of extra output channels.

**What this follow-up changes in the plan** [I]:
- Read the energy at random iteration counts. DT-Recall's progressive loss already trains from random partial rollouts; keep it with the energy, since a fixed-horizon energy allowed a clock schedule here.
- Check the energy's stationary points before any training run (stage 1); two of the three BFS energies tried in development had a plateau that training did not leave in 400 steps.
- Run at least 3 seeds and both neighbouring learning rates, and report the spread: here the energy's outcome depended on both.
- Fix an oracle-free stopping rule in advance, and test that the evaluator respects its own cap (the bug found in review).

**Stages.**
1. CPU, about 1 hour: audit the energy on the existing checkpoints (released, and the three September seeds) on P13's frozen e19, e33 and e59 sets: zero energy on every exact output, positive on every wrong one. Then minimise the energy directly over y (no network, Adam from uniform and from random starts) on 100 mazes per size, as a diagnostic of the objective: stationary points that trap plain descent on y are candidates to check for in training, though a network trained across many mazes need not meet them.
2. CPU: plan §4's prefix-parity stage (6 runs, about 16 CPU-hours).
3. GPU, only if stage 1 passes: three label-free seeds with the September recipe (`dt-baseline/code/train_resume.py`, width 128, 50 epochs, progressive loss with the energy in place of cross-entropy): about 7.1 L4-hours per seed in FP32, 21-23 L4-hours with evaluation, about $18-20 at the $0.87/h list ceiling (September's spot rate was about half that). Above this lane's $10 cap, so it needs your OK.

**Criterion** (plan §4.6, unchanged): the label-free arm passes iff ≥ 2 of 3 seeds solve ≥ 80% of 59×59 mazes exactly; it reproduces our failure iff every seed is ≥ 95% on 9×9 validation and < 20% on 33×33. **Controls:** the energy with the goal moved to another dead end (the shuffled-label analogue; it must fail), training on 5×5 mazes (the M3 analogue), and the existing supervised seeds.

## 9. Cost, deviations, review, files

**Cost** [K: `followup/GPU-LOG.md`]: two spot L4 runs (g2-standard-8): 00:18-04:17Z in us-west1-b (3.96 h, $3.45) and 04:55-05:36Z in us-east4-c (0.66 h, $0.57), $4.02 in all at the $0.87/h list ceiling (spot is lower; the Billing API is off, so no invoice figure). Lane cap $10. Both VMs are deleted (`gcloud compute instances list`, 05:37Z: none). Diagnostics ran on General's CPU at `nice -n 19 ionice -c3`, at most two processes of two threads.

**Deviations from upstream DNAR and from the pilot** [K: PREREG §10]: training in float32 on GPU (evaluation in float64); top-1 selection by an exact segment maximum (identical to PyG's normalised sort on 300 random cases with ties and on the pilot's own records); training graphs relabelled; 2,000 steps instead of 1,001, with the learning rate chosen from three; D1-D4 drop the ScalarUpdater (BFS's positions are constant inputs). Plus: the energy's development rounds, amendments 1 and 2, the review fix with the D4-E cell completed at 1e-3 afterwards, and the exploratory arm.

**Review** (gpt-6-astra via codex-any, `model: gpt-6-astra` in `review/review_r1.err`; read-only; `review/review_r1.md`). It did not sign off on the first draft. Every finding was applied:

| # | Finding | What changed |
|---|---|---|
| 1 | Serious: soft models ran 24 steps past the cap; D4-E seed 40 at 3e-4 falls from 317 to 307/320 under the cap | Fixed in `run_batch`; all soft-model evaluations re-run; the fix moved D4-E's registered rate to 1e-3, so that cell was completed and scored at 1e-3; I reproduced 307/320 |
| 2 | Serious: the headline claimed success the registered criterion did not give | Headline now leads with the registered verdicts (after the fix, D4-E meets the valid clause) |
| 3 | Serious: seed 41 at 3e-4 called stationary while drifting on K2 | Stationarity claims dropped for it; drift-free reading added (`results/tables.md`); D4-E at 1e-3 seeds 41-42 checked for drift (none) and at fixed steps 15, 20, 30 |
| 4 | Moderate: the depth diagnostic used a different stopping rule (D4-O: depth 91 against the evaluator's 17) | Diagnostics now read the evaluator's own saved predictions (`profile_from_pred.py`); depth 17-18 reported |
| 5 | Moderate: a unique answer is not a unique procedure | Section 1.2 says so; the energy is described as a hand-written specification |
| 6 | Moderate: necessity and untrainability overclaimed | "Under this recipe" and "observed" throughout; D2→D4 is not one change (section 1.2) |
| 7 | Moderate: the random horizon uses the true depth and changes compute | Disclosed in section 6; causal language weakened |
| 8 | Moderate: D4-E@3e-4's M3 does not fit its own size; SH does not always keep the search | Own-size column added; that control called inconclusive; SH claim narrowed; "memorisation" narrowed to whole-graph lookup |
| 9 | Moderate: sampling uncertainty | Binomial bounds and the shared-test-set caveat in section 1.3; "size-proof" dropped |
| 10 | Moderate: numbers (training depth 8-9, D3-H valid, untrained floor, D2-E Bellman) | Corrected; I re-counted the training depths |
| 11 | Minor: scorer provenance; failure mode should read exact; valid pathway missing | `analyze.py` reads exact for the failure mode, implements both pathways, and is hashed in `prereg_hashes.txt` |
| 12 | Minor: maze plan overstatements | "Chordless" path; direct minimisation framed as a diagnostic |

Second round (`review/review_r2.md`, `model: gpt-6-astra` in `review/review_r2.err`), on the revision: "I would sign off on the revised registered result, but not the report verbatim until the issues above are corrected." It recomputed the 1e-3 selection from seed-40 validation alone (314 against 307 and 0), verified that seeds 41 and 42 pass all 442 test graphs on validity and settlement with no recorded drift, checked the controls, and called the completed D4-E runs "a disclosed repair and completion, not an independent replication". Its seven remaining points were applied: the controls show that training and the objective matter, not that the architecture contributes nothing, and the two seeds' trees are not zeros of the energy at 1,600 nodes or on the grids (section 7); output stability is no longer called a fixed point, and "no drift" is scoped to the 16-step window and the sets checked at fixed steps (sections 3-4); tie-break errors also occur between distant candidates (section 4); D4-H seed 42 settles on 14/64, not 15; table metrics are labelled; the paper comparison is qualified; measured zeros are kept apart from unrun sets; the DT-Recall numbers are labelled as peak-over-iteration scores and the comparison with the N-Queens diagnosis is narrowed; the supplementary drift-free reading now also applies its filter to the 16-node gate (verdicts unchanged). One point is noted rather than changed: `run_batch`'s `all_visited` field reads the visited bits at confirmation rather than at the backdated settle step; no verdict uses it.

**Files** (all under `followup/`):
- `PREREG.md` (with amendments), `prereg_hashes.txt` (hashes and times of every registered, amended or fixed file).
- Code: `gpu_src/arms.py` (models, data, energy, training, evaluation, self-tests), `gpu_src/driver.py` and `gpu_src/driver_v2.py` (job matrices), `gpu_src/upstream/` (DNAR at `12f3f0bd`, unchanged), `gpu_src/queue*/` (VM job files), `scratch/` (VM scripts).
- Scoring and diagnostics: `analyze.py` → `results/tables.md`, `results/summary.json`; `results/selection_v2.json`; `profile_from_pred.py` → `results/profile_*.json`; `timing_trace.py` → `results/trace_*.json`; `analog_check.py` → `results/analog_check*.txt`; `results/k2_partner_check.txt`. The earlier `depth_profile.py` outputs (`results/depth_*`) are superseded (finding 4).
- Raw results: `runs/train/*/` (every model and its training log), `runs/eval_v2/` (corrected soft-model evaluations, with saved predictions), `runs/eval/` (hard-model evaluations; the pre-fix soft ones kept for comparison), `runs/selection.json`, `runs/explore/`, `runs/dev*/`, VM logs `runs/logs*/`.
