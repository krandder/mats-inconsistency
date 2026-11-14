# Sudoku in Kelvin's architecture: a trained world model, routes, trust, and a target the search is corrected against

Model: Claude Fable 5.2 (`claude-fable-5-2[1m]`). Written 2026-11-10, 01:30-04:50Z. One researcher, CPU only.
Pre-registration: `PREREG.md` (02:00-02:06Z) and nine dated addenda in `PREREG-ADDENDA.md`.
All paths are under `/home/agent/workspaces/mats/work/deep-study/wf7/sudoku/`. Every number below names its output file.

Labels: PROVEN (argument given here), CHECKED (a script and its output are named), SOURCE (I read it; sentence quoted),
CONJECTURE, NOT RUN. "Post hoc" marks anything that was not in `PREREG.md`.

---

## 0. In plain words

- I built it the way you described. M is a small masked transformer trained only on solved grids: no puzzle, no
  consistency loss. The graph is Bengio's: partial fillings, one cell per edge. An order policy is learned from real
  grids by asking which route surprises M least. T is that route with M's atoms. Several search policies walk the
  same graph and are corrected against the same T.
- M does encode which placements are valid. Whenever it says "impossible" it is right (precision 1.000 on every
  scored state), and with one cell of context nothing fools it: a gap of 6.6 nats separates valid grids from every
  wrong board the search proposed (5.9 nats from the invalid boards T itself draws).
- At 4x4 everything can be enumerated. There is a route through the trained network along which its law is almost
  exactly reality: 0.005 nats from uniform over the 288 grids. M's own confidence gives 0.12, a uniform mixture over
  orders 0.10. Dynamic programming on the trust MDP finds that route. No invalid board among all 4^16 passes T's test.
- Two things did not work as specified.
  - Trust fitted on the grids M was trained on rewards memorisation. At 4x4 the exact trust optimum for the training
    grids reproduces the training set and gives held-out grids nine times too little. Trust has to be fitted on real
    data M has not seen.
  - At 6x6, "T > 0" read from the long route alone let 33 of 33,524 wrong candidates through. That was enough for a
    wrong accepted answer on 2% of hard puzzles with the good search policy and on 10% with the deliberately bad
    one: the answer depended on the search. A route has to put its probability somewhere even after the board has
    become contradictory.
- What fixed the second: read the zero from M's smallest contexts (is each placement allowed given any one other
  placement), and keep the route as the soft law. Then 0 of 16,566 wrong candidates passed, and four search policies
  gave identical answers on every hard puzzle two of them solved, at up to 4.5 times different compute. The search
  that has never seen a puzzle solved 299 of 300 hard held-out puzzles, all correct. One pass of the same M with no
  check gets 75%; one pass of a network trained on 12,000 puzzle-solution pairs gets 68-71%, and it is a worse
  search policy than the world model because it is sure when it is wrong.
- Learned trust against M's own confidence, as the definition of T at 6x6: 0.19 against 0.25 nats on held-out grids;
  the interval touches zero, so by the pre-registered rule that is a null. In distribution it is 0.21 against 0.31 and
  clear. A plain row-by-row order beat both (0.08), and a learned mixture of six routes was best (0.06). My trust
  learner is far from the optimum. The thing that worked best is the aggregate over routes that you asked for.
- I do not know: whether a better learner finds a near-exact route at 6x6 as DP does at 4x4; what replaces "smallest
  contexts" when the rules are not pairwise; anything at 9x9 (the GPU queue estimate was 11 days, so nothing ran there).

---

## 1. What was built

### 1.1 The design in Bengio's objects

- **State graph.** A state is a partial filling of the board. An edge fills one empty cell with one digit. A
  trajectory from the empty board to a complete board is an order together with the values. Terminal states are the
  complete boards.
- **M, the world model.** For every state s and every empty cell i, M gives a distribution q(v | s, i) over the digit
  placed there. M is trained on real boards only, with the any-order (masked) loss. Nothing makes it consistent: two
  routes to the same complete board can get different probabilities, and they do (section 2.1, 2.3).
- **Order policy and forward policy.** An order policy pi(i | s) picks the next cell from the revealed cells only.
  Together with M's atoms it is a forward policy P_F(s -> s + (i, v)) = pi(i | s) q(v | s, i). If pi is deterministic,
  every complete board x has exactly one route, so the terminating law T_pi(x) is the product of M's atoms along that
  route: normalised, and one route of network calls to evaluate.
- **Chain rule (PROVEN, standard; checked exactly at 4x4).** For any such pi, the expected total surprise of M on a
  real board is the entropy of reality plus KL(reality || T_pi). The entropy is the same for every route. So real
  boards rank routes by their excess alone.
- **Trust MDP.** State = partial filling of a real board. Action = which cell reality reveals next. Cost = M's
  surprise at the revealed value. Its optimal policy pi_trust is the deterministic order policy whose T is closest
  to reality in KL. No searcher is involved.
- **T.** T(x) := T_{pi_trust}(x), with the route starting at the empty board. For a puzzle with clues c the target is
  T conditioned on the clues: T(x) 1[x agrees with c] / Z. The clues condition T; they never clamp M inside T.
- **Search.** A search policy is any other forward policy on the same graph, started at the clue state. Its
  candidates are complete boards that agree with the clues. A candidate is accepted if it passes T's test; for
  puzzles with several solutions the candidates get importance weights T(x) / proposal(x).
- **The goal** is the terminal condition "the board is complete and T does not rule it out". There is no utility here.

### 1.2 Networks, data, arms

| item | what | file |
|---|---|---|
| world | uniform law over valid grids: 288 at 4x4, 28,200,960 at 6x6 (entropy 5.663 and 17.155 nats) | `oracle.py`, `out/gen_base.out` |
| splits | by essentially-different class (orbit under cell symmetries and relabelling): 4x4 has 2 orbits (192 train, 96 held out); 6x6 has 49 (34 train, 6 validation, 9 test) | `out/gen_base.out` |
| puzzles | unique solution by the exact counter; minimal ones have 8-13 clues; "hard" = naked and hidden singles do not finish it (300 of 5,719 test puzzles) | `out/gen_puzzles_eval.out` |
| M_A | grids only. 6x6: `mA6`, 201,030 parameters, 3,000 steps. 4x4: `mA4`, 85,444 parameters, trained on the 192-grid orbit | `out/train_mA6.out`, `out/train_mA4.out` |
| M_B | puzzle + real solution, clue cells flagged: `mB6` (201,030 parameters, 12,000 minimal training puzzles), `mB6t` (25,958) | `out/train_mB6.out`, `out/train_mB6t.out` |
| D6 | proposal distilled from the search: trained on 2,498 training puzzles with 12-16 clues and the answers the search accepted. It never sees a real solution | `out/e_distill.out`, `out/train_D6.out` |
| trust nets | 2 layers, about 100k parameters; inputs are the state and M's atoms at it | `out/e_trust6_fit.out`, `out/e_trustB_mB6.out` |

Relational prior of every network, disclosed: no positional embedding; three of four attention heads are restricted
to the cell's row, column and box, one is global. The network is told which cells share a group. It is never told
that sharing means "different digits".

Fraction of test inputs seen in training: 0. No test grid and no solution of a test puzzle shares an orbit with any
training grid, so none is a relabelled, row-permuted or column-permuted training grid (`out/e_posthoc2.out`, item 3).

Firewall: `oracle.py` (enumeration, exact counting, validity, rating) is imported by generators and scoring code
only. `check_firewall.py` confirms that `core.py`, `data.py`, `train_m.py`, `e_distill.py`, `e_trustB.py` do not import
it, and that `e_search6.py` imports it only in its scoring branch for multi-solution puzzles.

The trust learner (`core.fit_trust`), in words: take a real board; walk the current policy's route to it; at one random
step reveal a different cell and then continue with the policy. Both routes end at the same real board, so the
difference of M's total surprise is a sample of the difference in excess (reality's entropy cancels exactly). A small
network regresses that difference; the next policy reveals the cell with the lowest prediction. Four rounds.

Search arms (cap 256 candidates per puzzle; candidate 0 is the proposal's greedy route unless stated):
- S1: M_A in its own-confidence order (largest top-1 minus top-2), values drawn from M_A's atoms.
- S2: deliberately bad, with a floor: M_A, least-confident cell first, values from 0.95 x atom + 0.05 x uniform; no
  greedy candidate.
- S3: D6 in its own-confidence order.
- S4: M_B in its own-confidence order.
- Baselines with no check: K0 = one pass of M_A; K1 = one pass of M_B (the Kim et al. recipe as the task states it: one
  cell per step, most confident cell, most probable value); K2 = best of k passes of M_B by its own route
  log-probability; KT = one pass of M_B in the order of its trust policy.

Compute is counted as FLOPs = 2 x parameters x 36 tokens per forward pass, for every network an arm calls.

---

## 2. Results

### 2.1 4x4, exact (`out/e_4x4_mA4.out`, `out/e_4x4_mA4_nodp.out`, `out/e_4x4_mA4_order.out`)

M_A here was trained on one of the two essentially different 4x4 grids (192 grids) and never saw the other (96).
"Excess" is the mean over grids of -log T(x) minus ln 288. On all 288 grids it is exactly KL(reality || T).

| order policy defining T | all 288 | held-out 96 | training 192 | mass of T on valid grids |
|---|---|---|---|---|
| exact optimum of the trust MDP, true law (DP; reference) | **0.0046** | 0.0056 | 0.0041 | 0.9964 |
| best of 256 random fixed orders (picked with the true law) | 0.0472 | 0.1306 | 0.0055 | 0.9907 |
| learned mixture of six routes (weights by EM on training grids; post hoc) | 0.0663 | 0.2960 | -0.0486 | 0.9927 |
| pi_trust, learned by paired routes on training grids | 0.0935 | 0.3453 | -0.0323 | 0.9922 |
| uniform mixture over 256 random orders | 0.0956 | 0.2796 | 0.0036 | 0.9167 |
| raster order | 0.1018 | 0.3341 | -0.0144 | 0.9926 |
| own confidence: lowest entropy | 0.1220 | 0.2513 | 0.0574 | 0.9940 |
| own confidence: top-1 minus top-2 | 0.1636 | 0.1545 | 0.1682 | 0.9543 |
| one random fixed order (mean over 256) | 0.1666 | 0.3461 | 0.0768 | 0.9167 |
| own confidence: top probability | 0.2026 | 0.0067 | 0.3006 | 0.9537 |
| least confident first | 0.4005 | 0.5136 | 0.3439 | 0.7157 |
| tabular trust policy: DP with the training grids as reality | 0.5038 | **2.2174** | **-0.3530** | 0.9954 |

What the table says:

1. **A reliable route exists in the trained network, and the trust MDP finds it** (CHECKED, exact). With the true
   law, DP over all reachable states gives a T that is 0.0046 nats from reality. That is 26 times closer than M's
   lowest-entropy order, 21 times closer than the uniform mixture over orders and 10 times closer than the best of 256
   fixed orders. The coordinator's result on hand-made tables holds on a trained network.
   - M is far from consistent: over 256 random orders, log q_sigma(x) has a standard deviation of 0.38 nats per grid.
   - Checks on the DP: M's atoms are equivariant under the 64 cell symmetries to 7e-06, which lets the DP run on
     about 340,000 states instead of 15,802,049. V*(empty) - ln 288 = 0.004597 and the route surprise from walking the
     policy is 0.004597: the chain rule, to six digits.
2. **The order decides how much of T lands on invalid boards** (CHECKED, exact). Least-confident-first puts 28% of
   T on invalid boards, a random fixed order 8%, the lowest-entropy order 0.6%, the DP route 0.36%.
3. **Trust fitted on the grids M trained on rewards memorisation** (CHECKED, exact). Run the same DP with the 192
   training grids as "reality": on those grids its T is 0.35 nats *better* than uniform over 288, which is nearly
   the training set's own law (ln(288/192) = 0.405). On the held-out orbit it is 2.2 nats worse: those grids get
   one ninth of their share. The same network holds both routes: one that applies the rules, one that replays the
   training orbit. Which one trust picks depends on the data trust is fitted on. The learned trust policy sits in
   between (-0.03 on training grids, 0.35 held out), and M's own top-probability order is nearly exact on the held-out
   orbit (0.0067) and poor on the training one.
   - Reading (CONJECTURE, stated as a design rule): fit trust on real data that M has not been trained on. At 4x4
     there is no such data to test this with, because the training orbit is all M ever saw. At 6x6 M sees under 5% of
     the grids of its training orbits, and there the learned trust policy does not favour the training orbits
     (section 2.3).

**T on complete boards, exhaustively** (`[4b]` in the output; every board with T >= 1e-9 enumerated, pruned mass
below 2e-05):

| T defined by | boards with T >= 1e-9 | smallest T on a valid grid | largest T on an invalid board | invalid accepted at tau | valid rejected |
|---|---|---|---|---|---|
| pi_trust (learned) | 25,725 (288 valid) | 1.50e-03 | 1.11e-04 | 0 of 4^16 - 288 | 0 of 288 |
| own confidence, lowest entropy | 17,125 | 1.05e-03 | 7.64e-05 | 0 | 0 |
| DP optimum (reference) | 15,500 | 2.99e-03 | 2.05e-05 | 0 | 0 |

tau is the smallest log T over the training grids minus 2.0 nats. At 4x4 the test is exact: no invalid board among
4,294,967,008 passes, no valid grid fails, including the 96 the network never saw. The margin between the least
likely valid grid and the most likely invalid board is a factor 13.5 for the learned route and 146 for the DP route.

Search at 4x4 is too easy to say much (one pass of M with no check already solves all 576 minimal puzzles). It does
show the pattern: S1, S2 and the trust-order search all return the same accepted answers, all correct, at 1.26e8,
1.70e8 and 1.63e8 FLOPs on the held-out orbit; S2 needs 1.55 candidates on average (2.87 on the training orbit).

**What the optimal route looks like** (`out/e_4x4_mA4_order.out` [2c]; all 288 grids, every step scored exactly).

| | DP optimum | pi_trust (learned) | lowest entropy | top probability | random order |
|---|---|---|---|---|---|
| free choices per route (of 16 steps) | 5.5 | 6.7 | 6.7 | 5.8 | 5.6 |
| mean step index of the free choices (0-15) | 3.2 | 4.1 | 4.4 | 4.6 | 2.7 |
| a locally forced cell exists and is taken | 76% | 90% | 92% | 90% | 66% |
| exact excess paid at free choices | 0.0024 | 0.0912 | 0.1194 | 0.1565 | 0.1061 |
| at cells forced but not locally | 0.0003 | 0.0002 | 0.0003 | 0.0447 | 0.0575 |
| at locally forced cells | 0.0019 | 0.0022 | 0.0023 | 0.0014 | 0.0016 |
| total (= KL to reality) | 0.0046 | 0.0935 | 0.1220 | 0.2026 | 0.1652 |

So the answer to "forced cells first, or calibrated free choices?" at 4x4 is: calibrated free choices first. The
optimal route pays reality's entropy early, in few choices, at contexts where M's free-choice atoms are almost
exactly right (0.0024 nats in total), and then fills forced cells. It is less greedy for forced cells than M's own
entropy order (76% against 92%). Nearly all of the other routes' excess is paid at free choices made in the wrong
contexts. Its first two cells share no row, column or box. This matches the stiffness table of section 2.2 for
6x6: M is reliable in small contexts and on nearly full boards, and the best route goes from one to the other with
as few guesses in between as possible.

### 2.2 6x6, E1: where M's atoms are reliable (`out/e_trust6_analyse.out` [d], `out/e_stiff.out`)

1,200 states on routes of held-out test grids, 22,568 atoms, scored against the exact conditionals (counter):

| context type of the cell | share | M's surprise | exact entropy | exact excess | M's mass on rule-breaking values | on values with no completion |
|---|---|---|---|---|---|---|
| (a) locally forced: one value not used by a peer | 9.2% | 0.0001 | 0 | 0.0001 | 6.4e-05 | 0 |
| (b) forced, but not locally | 9.0% | 0.0257 | 0 | 0.0257 | 1.6e-04 | 2.2e-02 |
| (c) free: several values have completions | 81.8% | 1.3070 | 1.2888 | 0.0160 | 1.8e-04 | 2.4e-03 |

- **Pre-registered reading "M encodes the rule": SHARP.** Mean mass on directly rule-breaking values is 1.7e-04 per
  atom (99th percentile 1.2e-03, maximum 3.3e-02).
- Free choices are calibrated up to about 0.6: where M says 0.23, 0.37, 0.56 the exact probability is 0.226, 0.368,
  0.528. Above that it is overconfident: 0.77 against 0.68 (n = 415), 0.93 against 0.76 (n = 17).
- The excess sits in the middle of the board: 0.002 nats per atom with 0-5 cells filled, 0.027 with 12-23, 0.001 with
  30-35. So does the mass on dead-end values: 8.7e-03 per atom with 12-17 cells filled.

**Stiffness by context size (post hoc, `e_stiff.py`).** Contexts are d cells of a real validation grid, 1,000 grids
per row. "Rules out" means q < 1e-3.

| cells revealed | share of placements M rules out | real values M rules out | worst log q of a real value | ruled out and truly impossible | truly impossible and ruled out: rule-breaking | : no completion |
|---|---|---|---|---|---|---|
| 1 | 5.7% | 0 | -2.07 | 100% | 100% | (none exist) |
| 2 | 11.2% | 0 | -2.73 | 100% | 100% | (none exist) |
| 3 | 16.4% | 0 | -3.12 | 100% | 99.98% | 10% |
| 6 | 31.4% | 0 | -4.13 | 100% | 99.8% | 29% |
| 9 | 46.5% | 0 | -4.84 | 100% | 99.3% | 43% |
| 12 | 61.5% | 4.2e-05 | -7.40 | 100% | 98.9% | 57% |
| 15 | 73.1% | 0 | -6.81 | 100% | 99.1% | 77% |
| 18 | 79.3% | 0 | -3.83 | 100% | 99.8% | 91% |
| 24 | 82.9% | 0 | -1.19 | 100% | 100% | 99% |
| 30-35 | 83.3% | 0 | -0.00 | 100% | 100% | 100% |

The exact columns are scored on 150 states per row. Reading (CHECKED for this network):
- When M rules a placement out, it is right. In about 30,000 scored placements per row I found none that was ruled
  out and possible.
- M knows the direct rule at every depth (98.9% to 100%).
- M does not know dead ends in the middle of the board. With 3 to 12 cells revealed it rules out only 10% to 57% of
  the placements that are locally fine but have no completion. Past 24 cells it knows them all.
- So M is stiff at the two ends, tiny contexts (the bare rule) and nearly full boards (forced cells), and soft in
  between. The two columns "real values M rules out" and "worst log q of a real value" need real data only; the
  others need the oracle.

### 2.3 6x6, E2: trust against own confidence as the definition of T (`out/e_trust6_fit.out`, `out/e_trust6_analyse.out` [a]-[c], [e])

Excess = mean of -log T(x) minus ln 28,200,960, with 95% bootstrap intervals. Test grids come from the 9 held-out
orbits (1,000 grids); "in distribution" is 500 fresh grids from the training orbits (M's training drew 768,000 of the
18,766,080 grids of those orbits, so about 4% of these 500 were probably seen by M).

| order policy defining T | held-out orbits | minus pi_trust, paired | in distribution | minus pi_trust, paired |
|---|---|---|---|---|
| learned mixture of six routes (weights on 256 validation grids; post hoc) | **0.062** [0.037, 0.085] | -0.132 [-0.158, -0.107] | | |
| raster order | 0.081 [0.045, 0.114] | -0.113 [-0.163, -0.062] | | |
| pi_trust (paired routes; best of 4 rounds on validation grids) | 0.194 [0.155, 0.232] | 0 | 0.209 [0.156, 0.264] | 0 |
| own confidence: top probability | 0.246 [0.201, 0.293] | +0.052 [-0.004, +0.111] | 0.390 | +0.181 [+0.100, +0.261] |
| own confidence: lowest entropy (= conf*, chosen on validation) | 0.248 [0.202, 0.293] | +0.054 [-0.001, +0.114] | 0.311 | +0.103 [+0.022, +0.185] |
| own confidence: top-1 minus top-2 | 0.291 [0.243, 0.343] | +0.097 [+0.034, +0.163] | 0.420 | +0.212 [+0.129, +0.294] |
| uniform mixture over 8 random orders (200 grids; estimate) | 0.400 | | | |
| one random order | 0.595 [0.549, 0.645] | +0.401 [+0.342, +0.462] | | |
| least confident first | 1.385 [1.349, 1.424] | +1.191 [+1.138, +1.245] | | |

- **Pre-registered headline: NULL.** excess(pi_trust) = 0.194, excess(conf*) = 0.248, difference +0.054 with a
  paired interval of [-0.001, +0.114] that includes zero. (A second bootstrap seed gave [+0.0002, +0.106]. It is at
  the edge; I report the pre-registered computation.)
- In distribution the same comparison is clear: 0.209 against 0.311, difference +0.103 [+0.022, +0.185] (post hoc
  column, addendum A4). Unlike 4x4, the learned policy does not do better on the training orbits than on held-out
  ones (0.209 against 0.194), so there is no sign here that it leans on memorised orbits.
- **The learner is far from the optimum of its own problem.** A fixed row-by-row order, which is one of the policies
  it could have represented, has less than half its excess (0.081). The four rounds were also unstable: validation
  excess 0.497, 0.280, 0.358, 0.175 (`out/e_trust6_fit.out`). So this section measures my learner as much as the
  idea. The 4x4 DP shows what the idea can do when the optimisation is exact.
- **Aggregating routes helps beyond any single route** (post hoc, addendum A6). The mixture puts weight 0.46 on
  pi_trust, 0.23 and 0.14 on two own-confidence rules and 0.16 on raster, and reaches 0.062. It is normalised and
  its density costs six routes. This is the smallest instance of "score the aggregate on real data and learn how to
  aggregate": the weights were fitted by likelihood on 256 real validation grids.
- M's inconsistency at 6x6: over 8 random orders, log q_sigma(x) has a standard deviation of 0.68 nats per grid and
  a range of 2.1 nats.

**What order did the trust policy learn?** (100 held-out grids, every step scored exactly.)

| | pi_trust | lowest entropy | random order |
|---|---|---|---|
| steps where the chosen cell is locally forced / forced not locally / free | 36% / 16% / 48% | 39% / 11% / 50% | 41% / 21% / 38% |
| a locally forced cell exists and is taken | 79% | 87% | 59% |
| exact excess along the route | 0.214 | 0.290 | 0.751 |
| exact excess per free choice | 0.0116 | 0.0154 | 0.0338 |
| exact excess per forced-not-locally step | 0.0021 | 0.0034 | 0.0388 |
| sum over steps of the smallest excess available at that step | 0.016 | 0.017 | 0.006 |

In words: forced cells first, as M's own entropy would do, but not slavishly. It takes more of the cells that are
forced without being locally forced, and M is right about those when the trust policy picks them (0.002 nats per
step). Its free choices are better calibrated than those of the entropy order. The last row says a route with
almost no excess is available step by step (0.016), which the learner does not find. The exact sum of per-step
excess along the route (0.214) agrees with the surprise-based estimate (0.194 on 1,000 grids): the chain rule again.

### 2.4 6x6, E3: T on complete boards. The pre-registered test fails; what fixes it

**Pre-registered test** (`out/e_trust6_analyse.out` [f], `out/e_report6.out`): accept x iff log T(x) >= tau, with
tau = the minimum of log T over 2,000 real validation grids minus 2.0 = -24.28. (The validation mean is -17.41. One
validation grid in 2,000 sits at -22.28, five nats below the rest.)

| class of boards | T by pi_trust | T by lowest entropy (comparison) |
|---|---|---|
| held-out valid grids rejected | 0 of 1,000 | 0 of 1,000 |
| near-miss, one cell changed: accepted | 1 of 1,000 | 0 of 500 |
| near-miss, two cells of a row swapped: accepted | 1 of 1,000 | 0 of 500 |
| mass of T on valid grids (draws from T) | 0.9665 [0.958, 0.974] (2,000 draws) | 0.9870 [0.978, 0.992] (1,000 draws) |
| invalid draws from T accepted | 51 of 67 | 6 of 13 |
| wrong candidates proposed by the search arms: accepted | 33 of 33,524 (0.10%; 95% upper bound 0.14%) | |

- **Pre-registered bin: FAIL.** T by pi_trust puts 3.3% of its mass on invalid boards, and three quarters of the
  invalid boards it draws pass the threshold. Their log T reaches -16.2, higher than a typical valid grid. No
  threshold on the route density can remove them: a board the route failed to see through looks as likely to T as a
  valid grid. Near-miss boards made by hand are rejected 99.9% of the time, so that test alone would have looked fine.
- The trust route leaks more than the entropy route (3.3% against 1.3%) while fitting valid grids better (0.19
  against 0.25 nats). The trust MDP optimises the second, not the first.
- On the candidates the search itself proposes for hard puzzles, 16 of 2,254 wrong boards were accepted in S1
  (`out/search_S1_HARD.out`), and 6 of 300 puzzles ended with a wrong accepted answer.

**Why** (`out/e_posthoc.out` [1], [2]; post hoc).
- PROVEN (it is G1.1 of the earlier general-reasoner note): take an invalid complete board and the first step on its
  route where the revealed prefix stops having a valid completion. The context before that step is a partial filling
  of a valid grid, and the true probability of the revealed value is 0. T(x) is at most M's probability of that one
  value. So T is zero on every invalid board if M is exact there.
- CHECKED: M is not exact there, and cannot be expected to be. Either that value repeats a revealed peer, which M
  sees (98.9% to 100% in section 2.2), or the prefix simply has no completion, which in the middle of the board M
  sees 10% to 57% of the time. After that step every context lies outside anything M was trained on, and a softmax
  still has to sum to one.
- The 7 distinct wrong boards accepted in S1 each contain exactly 2 rule violations (the fewest a wrong candidate
  can have here). The smallest atom on their route has median probability 0.005, where a typical wrong candidate
  has 3e-05. Nothing on the route ever looked impossible to M.
- A tighter tau trades one error for the other and never reaches zero on both: the lowest valid validation grid
  (-22.28) is below the highest wrong candidate (-21.01), and below T's own invalid draws (-16.2).

**What fixes it: read the zero from M's smallest contexts** (`out/e_posthoc.out` [3]; post hoc, addendum A8).
Pair check: the smallest log q(x_j | only x_i revealed) over all ordered pairs of distinct cells. It costs 36 calls
of M per board, no trust net, and 9.9 calls on average for a wrong candidate with early stopping.
- PROVEN for an M that is exact on real contexts (this is the scope test G1.3 of the earlier note, for rules between
  two cells): a one-cell context is a partial filling of a valid grid whatever the board is. On an invalid board two
  peers share a digit, and the true probability of the second given the first alone is 0. On a valid board every
  pair atom is a true conditional of the uniform law, bounded below by one constant.
- CHECKED for the trained M_A: every valid grid scores -2.067 (1,000 validation and 500 test grids; the same value
  for all, as the argument predicts). Wrong candidates score at most -8.71 (235 distinct boards), near-miss boards
  at most -8.95 (1,000), and the 7 boards the route test accepted at most -8.80. The gap there is 6.6 nats. With
  theta = -4.07 (validation minimum minus 2.0): 0 valid grids rejected, 0 invalid boards accepted. The
  67 invalid boards drawn from T itself, of which the route test accepted 51, score at most -7.99: all rejected
  (`out/e_posthoc2.out`). So do all 20 distinct wrong boards that the route test accepted in any search arm (at most
  -8.49). The smallest gap over every set tried is 5.9 nats.
- One-cell contexts are the stiffest atoms M has by the reality-only columns of the table in 2.2: M never rules out a
  real value there and its worst surprise on a real value is the smallest of any context size.

**T2** := pair check, then the pre-registered route threshold. In the terms of the design: infinite trust on the
atoms that reality never contradicts, and the trusted route as the soft law over what they leave. T2 is defined
by M and a small specification (theta and tau from real grids), and no search policy enters it.

### 2.5 6x6, E4: search corrected against T (`out/search_*.out`, `out/base_HARD.out`, `out/e_report6.out`)

HARD = the 300 held-out puzzles that naked and hidden singles do not finish (8-12 clues). ALL = 500 held-out minimal
puzzles of every difficulty (19 of them hard). Cap: 256 candidates.

**With the pre-registered test (route threshold only).**

| search policy | puzzles | accepted within the cap | accepted answers that are the solution | candidate 0 correct (one pass, no check) | candidates to accept | FLOPs to accept, mean / median |
|---|---|---|---|---|---|---|
| S1: M_A, own confidence | 300 | 99.7% | 98.0% (6 wrong) | 75.3% | 4.3 | 3.8e9 / 1.2e9 |
| S2: M_A, least confident first, 5% floor (bad on purpose) | 60 | 98.3% | 89.8% (6 wrong) | 1.7% | 43.5 | 3.1e10 / 1.9e10 |
| S3: D6, distilled from the search | 300 | 83.3% | 98.8% (3 wrong) | 65.3% | 19.7 | 1.7e10 / 1.2e9 |
| S4: M_B, trained on puzzle-solution pairs | 100 | 83.0% | 96.4% (3 wrong) | 65.0% | 17.5 | 1.4e10 / 1.2e9 |

- **E4(a), pre-registered: FAIL.** 18 of 691 accepted answers are not the solution, and two policies agree on only
  530 of the 565 puzzle pairs where both accepted something. The wrong answers depend on the policy: the bad policy
  proposes about ten times more wrong boards per puzzle and ends on a wrong one for 10% of puzzles, the good one
  for 2%. Over all arms the route test accepted 33 of 33,524 wrong candidates (0.10%) and rejected none of
  the 1,936 candidates that were the solution.
- On ALL (500 minimal puzzles of every difficulty) S1 accepted an answer for every puzzle and 499 are the solution;
  one pass of M_A alone gets 99.0%.

**With T2 (pair check, then the route threshold; post hoc).**

| search policy | puzzles | accepted within the cap | accepted answers that are the solution | candidates to accept | FLOPs to accept, mean / median | with the pair check alone |
|---|---|---|---|---|---|---|
| S1 | 300 | 99.7% | 100% | 5.4 | 4.1e9 / 1.7e9 | 3.3e9 / 9.0e8 |
| S2 (bad on purpose) | 60 | 96.7% | 100% | 44.6 | 2.1e10 / 1.2e10 | 2.0e10 / 1.1e10 |
| S3 | 60 | 78.3% | 100% | 17.8 | 1.0e10 / 1.7e9 | 9.5e9 / 9.0e8 |
| S4 | 60 | 85.0% | 100% | 18.6 | 1.1e10 / 1.7e9 | 9.9e9 / 9.0e8 |

- **Same answers: PASS under T2.** All 292 pairs of accepted answers are identical and all 455 accepted answers are
  the solution. Of 16,566 wrong candidates checked under T2, none was accepted (95% upper bound 2.3e-04),
  and none of the 1,118 candidates that were the solution was rejected. On ALL, S1 with T2 returned the
  solution for 500 of 500 puzzles.
- **Different compute.** On the puzzles that two policies both solved under T2, the deliberately bad policy needed
  4.5 times the FLOPs of S1 on average (2.1e10 against 4.6e9; 7 times at the median), S3 2.4 times and S4 2.6
  times. The answers were the same.
- **The learned solvers are worse search policies than the world model.** On hard puzzles S3 and S4 find an
  acceptable board for 78-85% of puzzles within 256 candidates; M_A's own conditionals do it for 99.7%, and even the
  deliberately bad order over M_A's atoms does it for 97%. D6 and M_B were trained on targets that are always a
  single value. When they are wrong they are sure, so their draws keep repeating the same board. M_A's atoms keep
  honest spread, so its draws explore. (The "distilled" policy here was distilled on easy puzzles, 12-16 clues,
  where candidate 0 was accepted for 99.8% of training puzzles and all 2,498 accepted labels were the true solution,
  `out/e_distill.out`. It learned those and little about the hard ones.)

**E4(c): against one-pass decoding at equal compute, on HARD** (`fig/fig3_accuracy_vs_flops.png`).

| arm | FLOPs per puzzle | correct on HARD |
|---|---|---|
| K1: one pass of `mB6t` (25,958 parameters), no check | 4.9e7 | 52.3% |
| K1: one pass of M_B, own confidence, no check (Kim et al. recipe) | 3.8e8 | 68.3% (70.7% with the entropy rule) |
| K0: one pass of M_A, own confidence, no check | 3.8e8 | 75.3% |
| KT: one pass of M_B in its trust order, no check | 5.6e8 | 74.3% |
| K2: best of k passes of M_B by its own score, no check, k = 2 ... 32 | 7.5e8 ... 1.2e10 | 68.3% at every k |
| S1 with the route test, budget 3e9 / 1e10 / 3e10 / 1e11 | | 79.3% / 92.3% / 96.0% / 97.7% |
| S1 with T2 (post hoc), same budgets | | 80.0% / 93.7% / 97.3% / 99.3% |

- Anytime rule: until something is accepted the answer is candidate 0, unchecked; an accepted answer is final. A
  wrong accepted answer counts as wrong.
- **Pre-registered headline, S1 against the no-check baselines: MAJOR.** At 6.0e9 FLOPs (the budget of K2 with
  k = 16) S1 is at 86.7% and the best baseline of equal or lower FLOPs is 75.3%, which is S1's own first candidate.
  With T2 the same comparison gives 90.3%. S1 ends at 97.7% (route test) and 99.7% (T2).
- For S4 (M_B as proposal, checked) the bin is MINOR: it ends at 80.0% (85.0% with T2 on 60 puzzles), above every
  unchecked point, but needs about 100 to 300 times the FLOPs of the point it passes.
- The larger one-pass model `mB6L` was not trained, so the one-pass size sweep has two sizes only.
- M_A, which has never seen a puzzle, beats M_B in one pass on the hard set (75.3% against 68.3%), while on all
  minimal puzzles M_B is at 99.2% on validation (`out/e_trustB_mB6.out`). Hard puzzles are 5% of M_B's training data.

**E4(d): puzzles with several solutions** (`out/e_multi.out`; 40 puzzles with 2-11 solutions, 64 draws per policy,
weights T(x)/proposal(x), route test; reduced from 200 puzzles and 256 draws, addenda A1 and A9).

| policy | draws accepted | solutions found | distance (TV) to the exact target | effective sample size of 64 |
|---|---|---|---|---|
| S1 | 91% | 95% | 0.15 | 39.7 |
| S2 (bad on purpose) | 7% | 52% | 0.50 | 3.6 |
| S4 (M_B) | 92% | 51% | 0.51 | 17.4 |

- **Pre-registered rule (median distance between two policies' estimates at most 0.10): FAIL** at this budget. The
  medians are 0.50, 0.53 and 0.62. Importance weights are consistent in the limit (a guarantee, not a finding); at
  64 draws S2 has under four effective draws, and M_B, trained on unique-solution puzzles, keeps drawing the same
  solutions and misses half of them. I make no claim that the estimates agree.
- The exact target here is T conditioned on the true solution set. It is not the uniform law over solutions: the
  distance between the two is 0.19 on average (maximum 0.42). T is defined by M and it is not the truth; its excess
  of 0.19 nats on valid grids shows up as uneven weights across solutions. The mixture of section 2.3 would be more
  even; I did not test it here.

### 2.6 E5: own confidence against reality-trained trust where M is confidently wrong

**Found contexts, M_B (the puzzle-trained network)** (`out/base_HARD.out`, `out/e_posthoc2.out`).
- One pass of M_B in its own-confidence order fails on 95 of 300 hard puzzles. At its first wrong step M_B gives the
  wrong value probability 0.988 on average (median 0.994) and the true value 6.2 nats of surprise. That state is
  broadly poisoned: of its 17 empty cells on average, M_B's most probable value is wrong at 42%.
- At the same states the trust policy picks the same cell half the time (50.5%), and M_B's top value at the cell it
  picks is right 45% of the time, with M_B still 0.986 confident there. So an order policy helps a little and cannot
  rescue a state where M is confidently wrong almost everywhere.
- One-pass accuracy on HARD: trust order (KT) 74.3%; own confidence 70.7% (lowest entropy, the rule chosen on
  validation), 69.7% (top probability), 68.3% (top-1 minus top-2). Against the pre-registered
  comparison rule (lowest entropy) the gain is 3.7 points: 34 puzzles only KT solves, 23 only K1, exact sign test
  p = 0.18. **Pre-registered bin: NULL.** Against top-1 minus top-2 it is 6.0 points, 42 against 24, p = 0.036.
- On validation puzzles of all difficulties the trust order was not better (99.0% against 99.2%; route surprise
  0.053 against 0.044 nats), and none of the learner's four rounds beat M_B's own order there
  (`out/e_trustB_mB6.out`). At the start only 0.5% of deviations from M_B's own order lowered the surprise: on typical
  puzzles M_B's own order is already close to the best order.
- More passes do not help the unchecked baseline: best-of-k by M_B's own route probability stays at 68.3% for every k
  up to 32, because the greedy board always scores highest, and the solution is among 32 draws for only 71.7% of the
  puzzles. When M_B is wrong, its draws repeat the mistake.
- Variant B's own target (E4(e)): the rule "T_B(x | clues) > 1/2" accepts an answer on 297 of 300 hard puzzles, and
  74 of those answers are wrong. M_B's own route density cannot serve as the target on hard puzzles.

**Constructed contexts, M_A (the grids-only network)** (`out/e_trust6_analyse.out` [g]). I could not find them. In
10,000 partial fillings of real test grids, M_A has 71,778 atoms with top probability at least 0.9. Their mean exact
probability is 0.9990 (M says 0.9977). Exactly one has exact probability below 0.1. No order rule picks that cell.
So the comparison this section asked for has nothing to bite on for M_A: on real contexts the grids-only model is
almost never confidently wrong.

Why the two networks differ (CONJECTURE). M_A is trained on contexts whose true answer is often open, so it has to
learn calibrated uncertainty, and it does (section 2.2). M_B is trained only on contexts whose true answer is a
single value. On the rare puzzles where finding that value needs more than one step of reasoning, it has no
calibrated way to say so. In Kelvin's terms: M_B is asked to output the deduction, M_A only the step.

### 2.7 E6: out of distribution

**Fewer clues than any training puzzle** (`out/e_report6.out`, `out/base_ALL.out`, `out/base_T12.out`). D6 saw
only puzzles with 12-16 clues. T12 = 500 held-out puzzles with 12-16 clues. FEW = the 491 puzzles of ALL with 8-11
clues. HARD has 8-12 clues and is hard as well.

| | T12 (12-16 clues) | FEW (8-11 clues) | HARD |
|---|---|---|---|
| D6, one pass, no check | 100% | 97.4% | 65.3% |
| M_B, one pass, no check (trained on minimal puzzles, so FEW and HARD are in its distribution) | 100% | 98.2% | 68.3% |
| M_A, one pass, no check (never saw a puzzle) | 100% | 99.0% | 75.3% |
| S3: D6 as proposal, route test | 500 of 500 accepted, 0 wrong, 1.10e9 FLOPs | 98 of 98 accepted, 0 wrong, 1.16e9 FLOPs (first 100 puzzles of ALL) | 250 of 300 accepted, 3 wrong |
| S1: M_A as proposal, route test | not run | 491 of 491 accepted, 1 wrong | 299 of 300 accepted, 6 wrong |
| S1 with T2 (post hoc) | | 500 of 500 on ALL, 0 wrong | 299 of 300 accepted, 0 wrong |

- Pre-registered pass rule (accepted answers at least 99.9% correct on FEW): S3 passes on a small sample, 98 of 98.
  S1 with the route test misses it by one puzzle (490 of 491); with T2 no answer on ALL is wrong.
- Fewer clues by themselves cost the checked search almost nothing: the compute ratio FEW / T12 for S3 is 1.05,
  because candidate 0 is accepted in both. Difficulty is what costs: on HARD, D6's draws find no acceptable board
  for one puzzle in six within the cap.
- The target does not move with the clue count. T is defined on complete grids by a network that never saw a puzzle.

**A grid size never trained on: NOT RUN.** I wrote the scripts (`train_gen.py`: one network with 9 symbol slots
trained on 4x4 and 6x6 grids; `e_size9.py`: its route density, the rule leak and a checked search on 9x9) and did not
run them: the three CPU slots were full until the end. Nothing is claimed about 9x9 or about size transfer.

---

## 3. What I take from this for the design

Each item says what it rests on.

1. **The world model does carry the rules, and it carries them best in small contexts.** (CHECKED, sections 2.2, 2.4.)
   "Which empty->filled placements are valid" is exactly what M knows reliably: given any one other placement, it is
   never wrong on real boards, and at least 5.9 nats separate valid from invalid on every set tried. What it does not
   know reliably is whether a half-filled board still has a completion. That is the search's job, not the world
   model's.
2. **"T := pi_trust with M's atoms" is a good soft law and a poor rule check.** (CHECKED at 6x6; exact pass at 4x4.)
   A long route must condition on everything revealed so far, and for an invalid board those contexts leave the
   world M was trained on. The hard zero should come from atoms whose contexts always occur in reality.
   - Proposed form (CONJECTURE as a general design; CHECKED here): T(x) proportional to T_route(x) times the indicator
     that no stiff atom rules x out. For Sudoku the stiff atoms are the one-cell contexts.
   - This is the "do not average stiff with non-stiff" rule in a concrete case: the stiff atoms veto, the rest weigh.
   - The stiff set can be learned from reality alone, at least here: it is the set of context sizes at which M never
     rules out a real value (table in 2.2, columns 3 and 4). I chose "one cell" after seeing the failure; a learner
     that picks veto contexts by that criterion is untested (CONJECTURE).
   - Limit: one-cell contexts suffice because Sudoku's rules are between two cells. A rule over r cells needs
     contexts of r - 1 cells (G1.3 of the earlier note), and a global rule has no bounded context. I do not know what
     plays this role there.
3. **A near-exact route exists inside the trained network.** (CHECKED, exact, 4x4.) The trust MDP's optimum is 0.005
   nats from reality where M's own confidence gives 0.12. My learner for it is weak: at 6x6 a plain raster order
   beats it. Whether the 6x6 optimum is as good as the 4x4 one is unknown; DP is out of reach there.
4. **Trust needs real data that M has not seen.** (CHECKED, exact, 4x4; not contradicted at 6x6.) Otherwise the
   routes that look most reliable are the ones that replay the training set.
5. **Aggregating over routes, with the aggregate scored on real data, beat every single route** (post hoc; six
   routes, weights by likelihood on 256 real grids: 0.062 nats at 6x6). This is the direction of Kelvin's 10 Nov
   message. What I did is its smallest version: fixed routes, scalar weights. A weight that depends on the board and
   on the sampled trajectories is the obvious next object, and I have not built it.
6. **With T2 the search policy did not move the target.** (CHECKED: S1 on 300 hard puzzles, S2, S3 and S4 on 60.)
   Policies differed by a factor of up to 4.5 in mean compute (7 at the median) and returned the same answers. Under
   the pre-registered route test this was not true: 2% to 10% of hard puzzles got a wrong accepted answer, and which
   one depended on the proposal.
7. **A network trained to output solutions is the wrong thing to trust on hard instances.** (CHECKED.) It is
   confidently wrong on about 30% of hard puzzles, its own density accepts 74 wrong answers of 297, and more
   unchecked passes do not help. The grids-only world model, which never saw a puzzle, is calibrated, and with a
   checked search it solved 299 of 300.

---

## 4. Every pre-registered outcome

| id | outcome as pre-registered | result | where |
|---|---|---|---|
| E1 | "M encodes the rule": sharp / partial / no | **sharp** (1.7e-04 mass per atom on rule-breaking values) | 2.2 |
| E2 | trust against conf* on held-out grids: major / minor / null / negative | **null** (0.194 against 0.248; paired +0.054 [-0.001, +0.114]) | 2.3 |
| E2, 4x4 | exact table incl. the DP optimum | reported; learned trust is better than own confidence on all 288 (0.094 against 0.122) and on training grids, worse on the held-out orbit (0.345 against 0.251) | 2.1 |
| E3, 4x4 | exact mass on valid; largest T on an invalid board | 0.992 (learned route); 1.1e-04; 0 invalid boards accepted, 0 valid rejected | 2.1 |
| E3, 6x6 | pass / partial / fail at the pre-set tau | **fail** (51 of 67 invalid draws from T accepted; 33 of 33,524 wrong search candidates accepted) | 2.4 |
| E4(a) | same answers; accepted answers >= 99.9% correct | **fail** with the pre-registered route test (HARD: 18 of 691 accepted answers wrong; 530 of 565 pairs agree). Post hoc with T2: 455 of 455 correct, 292 of 292 pairs agree | 2.5 |
| E4(b) | compute per policy | mean FLOPs to an accepted answer on HARD: S1 3.8e9, S2 3.1e10, S3 1.7e10, S4 1.4e10 (route test); S1 accepts for 99.7% of puzzles, S2 98%, S3 and S4 83% | 2.5 |
| E4(c) | S1 against K1/K2 at equal FLOPs: major / minor / null / negative | **major** for S1 (86.7% against 75.3% at 6.0e9 FLOPs; final 97.7%, or 99.7% with T2). Minor for S4. On ALL: null (one pass of M_A is already at 99.0%) | 2.5 |
| E4(d) | several solutions: median distance between policies <= 0.10 | **fail** at 64 draws (medians 0.50-0.62); not converged, no claim | 2.5 |
| E4(e) | variant B's own target | accepts on 297 of 300 hard puzzles; 74 accepted answers are wrong | 2.6 |
| E5 | KT against K1 on HARD: major / minor / null / negative | **null** (74.3% against 70.7% for the pre-registered rule; exact sign test p = 0.18) | 2.6 |
| E5 | same, against the other two own-confidence rules | +4.7 points (p = 0.10) and +6.0 points (p = 0.036); not the pre-registered comparison | 2.6 |
| E5 | found contexts | reported | 2.6 |
| E5 | constructed contexts for M_A | 1 confidently wrong atom in 71,778; no comparison possible | 2.6 |
| E6 | fewer clues: accepted answers >= 99.9% correct | S3: 98 of 98 on FEW (pass, small sample). S1: 490 of 491 with the route test (fail by one puzzle); 500 of 500 on ALL with T2 | 2.7 |
| E6 | unseen size | NOT RUN | 2.7 |

Guarantees, not findings (PREREG section 6): T is normalised for a deterministic order policy; the chain rule;
first-accept never revokes; if the test accepts only the solution, every full-support search returns the same answer.

Deviations from PREREG, all dated in `PREREG-ADDENDA.md`: smaller samples throughout (A1, A2, A5, A7, A9: the host ran
at load 60-80 and gave each process about 40% of a core); the trust learner used 1,024 boards per round, not 1,536;
E2 used 1,000 test grids, not 2,000, and one own-confidence number on 2,000 grids had been seen before that cut (A7);
S5, `mB6L`, `mA4full`, S1 on T12 and the unseen-size experiment were not run; S2 ran on 60 hard puzzles, S4 on 100;
E4(d) ran on 40 puzzles with 64 draws.
Added after the pre-registration and labelled post hoc wherever they appear: the in-distribution column of E2 (A4),
the learned mixture (A6), T2 and the pair check (A8), the stiffness table and the follow-up checks (A9).

---

## 5. What I do not know

- Whether the trust MDP's optimum at 6x6 is as close to reality as at 4x4. My learner's result is an upper bound
  on the excess (0.19), raster gives 0.08, the mixture 0.06; the per-step minimum suggests about 0.02 is available.
- Whether "fit trust on data M has not seen" is enough at scale, or whether subtler memorisation survives it.
- What replaces one-cell contexts for rules that are not pairwise, and whether a learner can find the veto contexts
  from the reality-only columns alone.
- How any of this behaves at 9x9, where hard puzzles need long chains of deduction and the proposals may find the
  solution far less often within a cap.
- Everything here rests on one trained network per role and one seed.

---

## 6. Claims and their status

| # | claim | status | evidence |
|---|---|---|---|
| 1 | For a deterministic order policy, expected route surprise on real boards = entropy of reality + KL(reality \|\| T). | PROVEN (telescoping; standard); CHECKED exactly at 4x4 | `out/e_4x4_mA4.out`: 0.004597 both ways |
| 2 | At 4x4 the trust MDP's exact optimum on the trained network gives KL 0.0046 nats; own-confidence orders 0.12-0.20; uniform mixture over 256 orders 0.096; best of 256 fixed orders 0.047. | CHECKED (exact) | `out/e_4x4_mA4.out`, `out/e_4x4_mA4_nodp.out` |
| 3 | The optimal 4x4 route makes its free choices early at calibrated contexts (0.0024 nats of excess there) and then fills forced cells. | CHECKED (exact) | `out/e_4x4_mA4_order.out` |
| 4 | At 4x4 the test "log T >= tau" accepts none of the 4^16 - 288 invalid boards and rejects none of the 288 valid grids. | CHECKED (exhaustive; pruned mass < 2e-05, every pruned board has T < 1e-9) | `out/e_4x4_mA4.out` [4b] |
| 5 | Trust solved exactly on M's own training grids reproduces the training set: -0.353 nats on training grids, +2.217 on the held-out orbit. | CHECKED (exact) | `out/e_4x4_mA4.out` |
| 6 | "Fit trust on real data M has not seen" removes that failure. | CONJECTURE (5 supports it; not tested directly) | |
| 7 | M_A at 6x6 encodes the direct rule: 1.7e-04 mass per atom on rule-breaking values; when it rules a placement out it is right on every scored state. | CHECKED | `out/e_trust6_analyse.out` [d], `out/e_stiff.out` |
| 8 | M_A does not know dead ends in the middle of the board: it rules out 10-57% of them with 3-12 cells revealed, 99-100% past 24. | CHECKED (150 states per row) | `out/e_stiff.out` |
| 9 | Learned trust against own confidence as T at 6x6, held-out orbits: 0.194 against 0.248, paired +0.054 [-0.001, +0.114]. Pre-registered bin: null. | CHECKED | `out/e_trust6_analyse.out` [c] |
| 10 | In distribution the same comparison is +0.103 [+0.022, +0.185]. | CHECKED (post hoc column) | same |
| 11 | A raster order (0.081) and a learned mixture of six routes (0.062) are closer to reality than the learned trust route. | CHECKED (mixture post hoc) | same |
| 12 | T by the trust route puts 3.3% of its mass on invalid 6x6 boards, and its threshold accepts 51 of 67 invalid draws. Pre-registered bin: fail. | CHECKED | `out/e_trust6_analyse.out` [f] |
| 13 | T(x) of an invalid board is at most M's probability of one impossible value at a context that is a partial filling of a valid grid. | PROVEN (G1.1 of the earlier note) | section 2.4 |
| 14 | Pair check: for an M exact on real contexts, valid boards score a constant and invalid boards score a leak of a one-cell-context atom. | PROVEN (G1.3 of the earlier note, pairwise case) | section 2.4 |
| 15 | For the trained M_A the pair check separates by at least 5.9 nats: no valid grid rejected (1,800 scored), no invalid board accepted (search candidates, near-miss boards, T's own invalid draws). | CHECKED (post hoc) | `out/e_posthoc.out`, `out/e_posthoc2.out`, `out/e_report6.out` |
| 16 | With the route test alone, accepted answers depend on the search policy (18 of 691 wrong on HARD; 2% of puzzles for S1, 10% for S2). | CHECKED | `out/e_report6.out` |
| 17 | With T2, four search policies return identical answers (292 of 292 pairs), all correct (455 of 455), at different compute. | CHECKED (post hoc; S2-S4 on 60 puzzles) | `out/e_report6.out` |
| 18 | The search over M_A's atoms, with no puzzle ever seen in training, solves 299 of 300 hard held-out 6x6 puzzles under T2; one pass of M_A gets 75.3%, one pass of M_B 68.3-70.7%, best-of-32 of M_B 68.3%. | CHECKED | `out/search_S1_HARD_T2.out`, `out/base_HARD.out` |
| 19 | Pre-registered headline E4(c): major (S1 86.7% against 75.3% at 6.0e9 FLOPs). | CHECKED | `out/e_report6.out` |
| 20 | M_B is confidently wrong on hard puzzles (0.988 on the wrong value at its first error); its own density accepts 74 wrong answers of 297. | CHECKED | `out/base_HARD.out` |
| 21 | M_A is almost never confidently wrong on real contexts (1 atom in 71,778). | CHECKED | `out/e_trust6_analyse.out` [g] |
| 22 | The right form of T is "trusted route times vetoes from the stiffest atoms", and the stiff set can be learned from reality-only statistics. | CONJECTURE | section 3, item 2 |
| 23 | M_B is overconfident because its training targets are always a single value. | CONJECTURE | section 2.6 |
| 24 | Anything at 9x9 or at an unseen size. | NOT RUN | |

---

## 7. Commands, files, compute

```
PY=/home/agent/.venvs/aggext/bin/python      # torch 2.14.1 CPU, numpy 2.5.4, matplotlib 3.11.2
R="nice -n 19 ionice -c3 $PY -u"             # every job: one thread, at most three at a time
$R gen_base.py                > out/gen_base.out            # grids, orbits, splits, counter self-test, puzzle pilot
$R gen_puzzles.py train       > out/gen_puzzles_train.out   # and: evalsets
$PY gen_syms.py
$R train_m.py mA6  A 6 64 4 3000 256 11 k6:train                 > out/train_mA6.out
$R train_m.py mA4  A 4 48 3 1500 256 21 k4:orbit1                > out/train_mA4.out
$R train_m.py mB6  B 6 64 4 3000 256 12 data/k6_train_min.npz    > out/train_mB6.out
$R train_m.py mB6t B 6 32 2 3000 256 12 data/k6_train_min.npz    > out/train_mB6t.out
$R e_4x4.py mA4 skipdp > out/e_4x4_mA4_nodp.out ; $R e_4x4.py mA4 > out/e_4x4_mA4.out ; $R e_4x4.py mA4 order > out/e_4x4_mA4_order.out
$R e_trust6.py mA6 fit > out/e_trust6_fit.out ; $R e_trust6.py mA6 analyse > out/e_trust6_analyse.out
$R e_distill.py > out/e_distill.out
$R train_m.py D6 B 6 64 4 2000 256 13 data/k6_train_12p.npz out/distill_labels.npz > out/train_D6.out   # 1,500 steps via out/override_D6.json
$R e_trustB.py mB6 > out/e_trustB_mB6.out
$R e_search6.py arm S1 HARD > out/search_S1_HARD.out        # likewise S2 HARD 60, S3 HARD, S4 HARD 100, S1 ALL, S3 ALL 100, S3 T12
$R e_search6.py arm S1 HARD 300 T2 > out/search_S1_HARD_T2.out   # post hoc check; likewise S2/S3/S4 HARD 60 T2, S1 ALL 500 T2
$R e_search6.py base HARD > out/base_HARD.out               # likewise ALL, T12
$R e_search6.py multi 40 64 > out/e_multi.out
$R e_posthoc.py > out/e_posthoc.out ; $R e_stiff.py > out/e_stiff.out ; $R e_posthoc2.py > out/e_posthoc2.out
$R e_report6.py > out/e_report6.out ; $PY e_fig1.py ; $PY check_firewall.py
```

- Code: `oracle.py` (world and exact truth, firewalled), `core.py` (network, routes, trust learner, searchers, pair
  check), `data.py`, `eval_lib.py` (scoring), the `e_*.py` experiment scripts, `queue/chain*.sh` and `run_step.sh` (job
  queues; `out/CHAIN_LOG` has the completion times; the one entry in `out/CHAIN_FAILURES` is the multi-solution run
  that I stopped on purpose, addendum A9).
- Figures: `fig/fig1_orders.png` (every order policy as a definition of T, 4x4 exact and 6x6), `fig/fig2_T_separation.png`
  (log T of valid grids and of the wrong boards the search proposed, with tau), `fig/fig3_accuracy_vs_flops.png`
  (hard puzzles: accuracy against FLOPs, search arms and one-pass baselines).
- Compute: CPU only, about 3 hours 10 minutes of wall clock on a shared host at load 60-80, three one-thread processes. No GPU job was submitted. Slurm
  dry runs (`sbatch --test-only --gres=gpu:1`) at 01:39Z and 02:31Z both returned a start time of 2026-11-21 (dry-run
  ids 57288, 57289, 57316), far over the 30-minute limit. No paid API, no message to anyone, nothing pushed.

## 8. What I read

- `wf7/COMMON.md`, all of it. "T should be symmetric over orders in its form, so that what is stiff can be LEARNED
  rather than hand-chosen."
- `kelvin-answers/2026-11-09-answers.md`, all of it. "The trust I meant here is part of the world-model, and trained
  only on reality."
- `theory/trust-mdp/check_trust_mdp.py` and `.out`. "pi_trust recovers reality exactly (KL 0) ... ordering by M's own
  confidence 3.0-3.8" (as summarised in COMMON.md; the script's own lines agree).
- `experiments/portfolio-core-demonstrations.md` sections 1.1 and 1.2. "Firewall against hidden solvers."
- `theory/G1-general-reasoner/note.md` sections 0 to 3.3, and the scripts `s2b_matched_verifier.py`,
  `va3_learned6x6.py` with their outputs. G1.1 and G1.3 are used in section 2.4 and credited there. "the test reads
  contexts of at most r - 1 variables."
- I did not re-read Kim et al. (arXiv 2502.06768) or any GFlowNet paper for this note. "The Kim et al. baseline" is
  taken as the task states it: one pass, most confident cell first, no check.

