# The full loop on trained world models: rollout policy, amplification, distillation, and an audit the searcher steers

Model: Claude Fable 5.2 (`claude-fable-5-2[1m]`). 2026-11-10, 16:30-18:45Z. Key `loop`, wf13. CPU only. Words follow
`wf9/FRAME.md` section 5: demonstration (the terminal state), score, target T (the score normalised), rollout policy
(the GFlowNet), amplification (the search on top of rollouts), distillation.

## 0. In plain words

Context. On 10 Nov you said the GFlowNet is the rollout policy, like AlphaZero's policy network; the search is the
amplification on top of many rollouts; the rollout policy is distilled from the amplification and improves over time;
network errors should cost only efficiency. Nobody had run that loop. I ran it in the two worlds with trained world
models: 6x6 Sudoku and the parity world (20 hidden assumptions, 30 noisy checks). The amplification has two forms:
weighted counting of K finished rollouts, and a tree in which the rollout policy chooses what to open while the
specification keeps the books and returns an interval around its answer.

1. Your safety claim held. For every rollout policy I tried (untrained, trained, trained toward a wrong goal, one that
   refuses part of the space) the exact answer of the score was inside the interval: no failure in about 11 million
   checked intervals. Bad policies cost openings or width, never the answer. (Three false alarms were a rounding error
   in my own bookkeeping; fixed and re-checked.)
2. The protection comes from the books, not from the rollouts. Weighted counting alone, fed by the policy that refuses
   part of the space, is off by 0.3 to 0.4 at any number of rollouts and nothing in its output shows it.
3. The loop did not compound. The first round of distillation gives almost all the gain (openings to close a hard
   Sudoku: 785 and 1,796 become 73 and 95; error of weighted counting in the parity world: 0.44 becomes 0.04 to 0.07).
   Rounds 2 to 5 add nothing, and in Sudoku the cost climbs back to about 170. Training once on what the untrained
   search found, with the same puzzles and steps, did as well in one seed and worse in the other.
4. A hand-made order is still cheaper: fewest-values-left closes hard puzzles in 43 openings against 73 for the best
   learned round; in the parity world the narrowest interval comes from the specification searching alone.
5. Why, as far as I can tell: finding the score and proving that nothing else is there want different orders. Your
   target says which values, never which order (every order of a demonstration scores the same), and almost all the
   cost is in branches whose completions score nothing, where distillation has no target. Started at the hand rule's
   level, each round of distillation made the Sudoku tree dearer (46, 67, 86, 86, 86, 95 openings) while the rollouts
   got better. Your "efficiency" target needs a second signal: the cost the search measured. That last sentence is a
   conjecture; I did not build it.
6. What the learned policy does buy: it finds the score sooner, and its rollouts get better every round (Sudoku
   rollouts ending in the solution: 0, 50, 52, 54, 61, 61 percent). Its own one-pass belief stays worse than one pass of M.
7. At 40 assumptions the certain interval is still 0.2 to 0.9 wide after 16,384 openings; the interval from the
   specification's own draws is at most 0.04 wide there. Your probabilistic bound is the one that survives size.

I don't know: whether an order trained on measured cost would compound; a world where the budget keeps binding (9x9) or
with no hand order; a trust head on top (not built: answers are compared with the score's exact answer); decisions.

## 1. What was built

All paths are under `/home/agent/workspaces/mats/work/deep-study/wf13/loop/`. Labels: GUARANTEE (by construction; verified),
PROVEN, CHECKED (script and output named), FINDING, POST HOC (not in `PREREG.md`, hashed 17:19:18Z before any registered
run; `PREREG-ADDENDA.md`). CPU only: a GPU job would have started on 21 Nov (`srun --test-only` 57871, 57872; none run).

- **Score and terminal state** (FRAME section 5). A terminal state is a demonstration: the filling with its order
  (Sudoku), the assumptions and then the deductions (parity world). The simple world model scores it step by step.
  Sudoku: each step is checked against the rule table read from M's one-cell contexts (`mk_tables.py`: equal to the
  true rules in all 46,656 entries), goal = all cells filled. Parity world: wf7's score from M's small-context numbers,
  the observed checks as a factor (`out/tab_w*.npz`). In both worlds the score is the same for every order of a
  filling and every filling has equally many orders, so every answer about fillings is a ratio of filling totals. The
  order in which the search fills is the search's own route and is dropped at submission (FRAME row 15).
- **Books** (the specification; `pw.py`, `sk.py`). For a partial demonstration the factors already fixed bound what its
  completions score in total (FRAME row 13; not the flow). Sudoku: 0 if some empty cell has no value left, 1 if
  complete, else at most the product of the numbers of values left. Parity: a check with all parents assigned is
  fixed; one with one open parent is summed out exactly with that parent; one with two or three open parents is
  bounded by its largest and smallest number. A state with no such check is closed: its total is known.
- **Rollout policy** (search side; starts untrained; `spol.py`, `ppol.py`). A small network giving value probabilities
  for every open variable at a partial state, plus an order rule; no extra internal state. Sudoku: 70,518 parameters,
  told which cells share a row, column or box (M's prior), reads each cell's value or remaining values; its output is
  not masked, so untrained it proposes at random; order = its most confident cell. Parity: 41,057 parameters, message
  passing on the score's hand-written graph, reads the books' local numbers; order in the importance form = its most
  confident assumption; in the tree the branching variable is the books' one-step lookahead (hand-made, the same for
  every policy), because in a dev pilot even exact conditionals with the confident-first order gave an interval ten
  times wider (`out/pilot2_w311.out`).
- **Amplification, importance form** (`sk.run_is`, `ptree.run_is`). K rollouts; a finished one counts score / (product
  of the policy's value probabilities), with a floor of 0.02 on those probabilities.
- **Amplification, tree form** (`sk.grow`, `ptree.grow`). A rollout walks down by the policy and opens the first
  unopened state on its path; the policy picks the variable; the specification creates every child and books it. A
  share of 0.5 of the openings is the specification's (largest upper bound first). `books()` returns, per query, found
  score for and against and the bounded score of what was not opened: a certain interval. Parity world only: `audit()`
  draws completions below the open frontier with the specification's own numbers (4,096 draws, no network) and
  returns a probabilistic interval (empirical Bernstein, 95% per query) and an unbiased point.
- **Distillation** (`spol.collect`, `ploop.collect`). The value head is trained on the amplified next-step value
  distribution at the states the tree opened (the share of the accounted score with each value, below that state).
  Fixed budget per training instance: 512 openings (Sudoku), 256 openings + 4,096 draws (parity). Five rounds, fresh
  instances each round (200 puzzles; 128 check vectors), earlier rounds replayed.
- **Controls.** One-shot: same instances and gradient steps, every instance amplified by the untrained policy.
  Hand-made: fewest-values-left first (Sudoku); the books' lookahead and local numbers (parity). Bad policies: random;
  wrong goal (the round-5 network shown the opposite observations, or with its value preferences shifted by one
  digit); avoiding part of the space (never proposes 1 for five assumptions; never digit 0 in the first two rows).
- **Data, firewall, compute.** Sudoku test puzzles come from wf7's test orbits (0% of test grids seen; stored solutions
  never read in training). Parity: 60 test instances per world from wf7's test seed; training on fresh real check
  vectors; hidden values never read. Exact answers (`sexact.py`, `pexact.py`: enumeration) only score finished searches
  (`out/check_firewall.out`: HOLDS). One call = one forward pass of the rollout policy on one state: 5.0e6 FLOPs
  (Sudoku), 2.0e6 (parity); one pass of M: 1.45e7 and 8.2e7; the score tables cost 216 and 241 calls of M once. A
  hand-ordered opening is table arithmetic (about 1e4 FLOPs), two orders of magnitude below a network-steered one.

## 2. Guarantees, and their verification

- **Counting once; the order cancels** (PROVEN; CHECKED exactly, `check_order.py`, `out/check_order.out`). For a policy
  that picks the next variable with pi(i | s) and its value with q(v | s, i), E[score / prod q; filling = x] = score(x):
  for a fixed x, "pick by pi, reveal x's value" is a law over orders and sums to 1. So the weighted share tends to T for
  every policy whose value probabilities are positive where the score is; the order part needs no condition. Dividing
  by pi as well counts each (order, filling) once and gives the same answers (200 random worlds, every route
  enumerated: errors below 3e-15; counting visits without dividing: off by up to 0.72 in total variation).
- **The interval** (PROVEN). The opened tree splits the fillings into closed states (exact totals) and open states with
  L <= total <= U, whatever chose the openings and the variables. Hence (found for + L for) / (found for + L for +
  found against + U against + U undecided) <= T(A), symmetrically above, and opening a state can only shrink it. The
  audit holds with probability 0.95 for every tree: given the tree, its draws are independent with mean exactly
  (unfound score) / (frontier bound), and they ask the policy nothing.
- **Verified** (GUARANTEE; `out/s_s{1,2}.json`, `out/p_w10{1,2,3}_s1.json`; every arm in `out/report.md`). Sudoku: 5,220,864
  intervals per seed (16 arms, 260 puzzles, 8 budgets, every empty cell and value). Seed 2: 0 outside. Seed 1: 3 outside
  in the first scoring, all a rounding error in my `books` (a bound formed by subtracting numbers near 1e9 came out
  -1.4e-06 instead of 0; `sdebug.py`, `out/sdebug_*.out`: 0 with the sum done directly). The fix was in place for all of
  seed 2; of seed 1 the two affected arms and rounds 0-1 were re-run (0 outside), the others were not (machine load;
  addendum A1). All 7,749 closed trees returned exactly the exact solution set. Parity: 518,400 certain intervals and
  25,920 bounds on the total score, 0 outside; the audit missed 7 of 518,400.
- **Same answers across rounds** (H2). Sudoku: all closed trees return the same set. Parity (worlds 101 / 102 / 103
  throughout): the certain intervals of rounds 0 to 5 share a point in every (instance, assumption) pair; the six audit
  points differ on average by 0.022 / 0.009 / 0.039 at 256 openings (worst pair 0.71) and by under 0.0004 at 2,048.

## 3. (a) The loop, round by round

**Sudoku** (registered; `out/s_s{1,2}.json`; figure 1). C = openings until the interval has closed (the solution set
proven), mean over 100 held-out puzzles, cap 2,048. Hard = singles do not finish the puzzle. Round 3: 145 / 142, 47 / 48.

| rollout policy | hard, seed 1 | hard, seed 2 | mixed, seed 1 | mixed, seed 2 | rollouts ending in the solution (hard) |
|---|---|---|---|---|---|
| round 0 (untrained) | 785 | 1,796 | 704 | 1,649 | 0.00 / 0.00 |
| round 1 | 73 | 95 | 35 | 56 | 0.50 / 0.37 |
| round 2 | 145 | 93 | 47 | 41 | 0.52 / 0.50 |
| round 4 | 144 | 152 | 47 | 52 | 0.61 / 0.58 |
| round 5 | 166 | 171 | 54 | 56 | 0.61 / 0.60 |
| one-shot control | 150 | 246 | 55 | 176 | 0.57 / 0.22 |
| hand-made order (fewest values left) | 43 | 44 | 30 | 30 | 0.36 / 0.35 |

- The answers do not change: every closed tree is the exact set. The compute falls 11 to 19 times in round 1, then
  rises about 2 times by round 5. **H3: null** by the registered rule (final / one-shot = 1.10 and 0.98 in seed 1, 0.69
  and 0.32 in seed 2). The seeds differ in their start: the untrained network closed 58% of training puzzles inside
  the budget in seed 1 and 10% in seed 2, so the one-shot control had 13,007 training states against 2,256. The loop
  beats training once only when the untrained search finds little.
- The value head improves every round (last column; -log probability of the solution along the policy's own route
  1.68 -> 1.32 and 3.36 -> 1.42) and for rollouts it passes the hand-made policy. The tree's cost does not follow: the
  learned policy is 1.7 to 2.1 times dearer than the hand-made order at its best round and 3.9 times at round 5.
  Round-5 values with the hand-made order cost 43 / 44, the hand rule's own cost. No round solves a hard puzzle in one pass.
- **Where the openings go** (POST HOC; `sdiag.py`, `out/sdiag_s1.out`; hard puzzles, seed 1). Every policy spends the
  same 25.9 openings on states that still contain the solution. The rest is inside branches with no solution:

| | round 1 | round 2 | round 5 | one-shot | hand-made |
|---|---|---|---|---|---|
| openings in branches that contain no solution | 46.8 | 118.9 | 139.4 | 122.0 | 16.8 |
| such branches entered per puzzle; mean size | 8.3; 5.6 | 11.4; 10.4 | 12.2; 11.4 | 12.1; 10.1 | 1.7; 10.1 |

  The cost of the interval is the cost of showing that the other branches are empty. Distillation has no target
  there (below a state whose completions score nothing the amplified next-step distribution is 0/0), and the target
  cannot rank orders: every order of a demonstration has the same score.
- **Started at the hand rule's level, distillation makes the tree dearer every round** (POST HOC, two seeds;
  `srun.py S order=confmask`, `out/mask_summary.out`). With the order "most confident after the books have dropped the
  values that collide", the untrained policy closes hard puzzles in 46 / 45 openings (hand rule 43 / 44). Rounds 1 to 5:
  67, 86, 86, 86, 95 and 64, 76, 83, 87, 89, while its rollouts improve (35% -> 57%, 34% -> 52% end in the solution).
  Being sure of the right value does not remove the wrong values that have to be refuted.

**Parity world** (registered; `out/p_w10{1,2,3}_s1.json`; 60 held-out instances per world). Tree at the training budget
(256 openings, 4,096 draws); importance form at 64 rollouts (about 1,140 calls). Errors are mean absolute, against
the exact posterior of the score.

| rollout policy | audit width | share of the score found | error of the audit point | error, importance form | KL(T, policy) |
|---|---|---|---|---|---|
| round 0 (untrained) | 0.142 / 0.065 / 0.204 | 0.36 / 0.52 / 0.20 | 0.007 / 0.003 / 0.009 | 0.424 / 0.436 / 0.464 | 10.9 / 11.5 / 11.6 |
| round 1 | 0.140 / 0.055 / 0.195 | 0.60 / 0.74 / 0.49 | 0.008 / 0.002 / 0.014 | 0.071 / 0.036 / 0.035 | 1.54 / 1.52 / 1.16 |
| round 5 | 0.125 / 0.054 / 0.188 | 0.65 / 0.79 / 0.58 | 0.008 / 0.003 / 0.011 | 0.063 / 0.021 / 0.044 | 1.45 / 1.11 / 1.20 |
| one-shot control | 0.133 / 0.050 / 0.174 | 0.62 / 0.79 / 0.59 | 0.007 / 0.003 / 0.012 | 0.071 / 0.049 / 0.046 | 1.51 / 1.07 / 1.19 |
| hand-made, share 0.5 | 0.157 / 0.065 / 0.228 | 0.41 / 0.53 / 0.18 | 0.008 / 0.003 / 0.012 | 0.153 / 0.128 / 0.191 | - |
| specification alone (no network) | 0.093 / 0.042 / 0.159 | 0.64 / 0.75 / 0.31 | 0.005 / 0.002 / 0.012 | - | - |

- **H4: null** (audit width, final / round 0 = 0.88, 0.83, 0.92; expected from the dev pilot). The openings needed for an
  audit width of 0.1 are the same for rounds 0 to 5 and the one-shot control (256 / 128 / 256 in the median instance).
  The interval is narrowest when the specification searches alone; distillation changes how soon the score is found
  (second column), not the width. With the network's own order instead of the books' lookahead the audit width at
  1,024 openings is 0.22 / 0.34 / 0.29 against 0.003 / 0.001 / 0.009.
- **H5: null** by the registered rule. Against the untrained policy the importance form's error falls to 0.15, 0.05,
  0.10 of its start (threshold 0.25: met), but against the one-shot control it is 0.89, 0.42, 0.97 (threshold 0.5: met
  in one world). Round 1 has the gain; rounds 2 to 5 are inside the noise of 60 instances.

## 4. (b) Deliberately bad rollout policies

No answer left its interval (section 2). The price (FINDING):

| policy (built from the round-5 network) | Sudoku: C on hard puzzles, seeds 1 / 2 | parity: audit width at 256, at 2,048 openings |
|---|---|---|
| round 5 itself | 166 / 171 | 0.125, 0.000 / 0.054, 0.000 / 0.188, 0.001 |
| random | 411 / 411 | 0.96, 0.32 / 0.94, 0.15 / 0.90, 0.16 |
| trained toward a wrong goal | 164 / 173 | 0.185, 0.001 / 0.087, 0.000 / 0.247, 0.003 |
| avoids part of the space, no guard | closes 37% / 34% of puzzles | 0.98, 0.98 / 0.85, 0.84 / 0.92, 0.91 |
| avoids, floor 0.02 only | 166 / 168 | 0.93, 0.54 / 0.66, 0.17 / 0.82, 0.51 |
| avoids, specification's share only | 314 / 296 (closes 92% / 93% in the cap) | 0.176, 0.001 / 0.069, 0.000 / 0.221, 0.003 |

- Random costs 2.5 times more openings in Sudoku and 8 to 16 times more for an audit width of 0.1 in the parity world.
  The wrong goal costs nothing in the Sudoku tree (the shift changes values, not the order) and 1.3 to 1.6 times the
  width in the parity world; in the importance form it is ruinous (0.1% of Sudoku rollouts end in the solution).
- The avoiding policy with no guard leaves the interval open for ever, and the open interval says so. In the importance
  form the same policy answers with error 0.42 / 0.35 / 0.30 at 256 rollouts and no warning.
- Two registered sentences were wrong (addendum A2). "Never closes a Sudoku puzzle": it closes 34 to 37%, because the
  specification books every child of an opened state and a child that completes the board needs no opening. "The share
  closes every Sudoku puzzle inside the cap": 92 to 93%; largest-bound-first is breadth-first in a rule world, where
  the floor is the better guard. In the parity world the share makes the avoiding policy as good as the honest one.

## 5. (c) Does distillation make the rollout policy narrow?

- Not by the registered test (H7: no). KL(T, policy) is flat after round 1 (1.1 to 1.5); -log P(solution) falls every
  round. The share of T's own samples that the policy under-proposes 100-fold is 2 to 8% from round 1 on, no trend, but
  lower for the one-shot control in two worlds (round 5: 0.052 / 0.029 / 0.043; one-shot: 0.080 / 0.008 / 0.016), and
  in Sudoku seed 1 the share of hard puzzles whose solution gets under 1% rises 0.04 -> 0.09 (seed 2 falls). A weak sign.
- Guards. The floor changes little for an honest policy (importance error at 64 rollouts with / without: 0.063 / 0.070,
  0.021 / 0.034, 0.044 / 0.046). The specification's share is what the interval needs: without it (round-5 policy,
  share 0) the audit width at 1,024 openings is 0.117 / 0.048 / 0.211 instead of 0.003 / 0.001 / 0.009.

## 6. (d) Parity world: the amplified belief against the exact posterior, on one axis

Mean absolute error against the exact posterior of the score (round-5 policy; figure 2). One pass of M costs 41 calls.

| belief | compute per instance | error | interval width (certain; audit) |
|---|---|---|---|
| the score's prior numbers | 0 | 0.356 / 0.371 / 0.353 | - |
| the rollout policy's own one pass, untrained -> round 5 | 1 call | 0.42 -> 0.284 / 0.45 -> 0.272 / 0.45 -> 0.263 | - |
| one pass of M (assumption given all checks) | 41 calls' worth | 0.222 / 0.185 / 0.215 | - |
| importance form, 64 rollouts | about 1,140 calls | 0.063 / 0.021 / 0.044 | none |
| importance form, 256 rollouts | about 4,560 calls | 0.029 / 0.015 / 0.026 | none |
| tree, audit point, 64 openings | 64 calls + 4,096 draws | 0.041 / 0.023 / 0.058 | 1.00; 0.67 / 0.49 / 0.76 |
| tree, audit point, 256 openings | 256 calls + draws | 0.0075 / 0.0029 / 0.0105 | 0.60 / 0.44 / 0.72; 0.125 / 0.054 / 0.188 |
| tree, audit point, 1,024 openings | 1,024 calls + draws | 0.0003 / 0.0001 / 0.0007 | 0.066 / 0.024 / 0.112; 0.003 / 0.001 / 0.009 |
| specification alone, 256 openings | 0 calls, table arithmetic | 0.0045 / 0.0020 / 0.0116 | 0.50 / 0.37 / 0.73; 0.093 / 0.042 / 0.159 |

- **H8: major** (audit point at 256 openings / one pass of M = 0.034, 0.016, 0.049; threshold 0.1). The distilled
  policy's own one-pass belief is worse than one pass of M in all three worlds: distillation made it a good proposer
  step by step, not a one-pass predictor of the root query.
- At about 256 calls the tree with the specification's draws is 10 to 20 times closer than the importance form (16
  rollouts: 0.137 / 0.055 / 0.100), and the specification alone is as close with no network call (the score here is a
  small table; where the score needs the network that row is not free). Without the draws the found-score point is
  poor (0.137 / 0.071 / 0.162 at 256). The score's exact posterior is 0.060 / 0.006 / 0.010 from the true one (M's errors).

## 7. (e) Where the interval stays wide

World 201: 40 assumptions, 60 checks, wf7's trained M (registered, `pscale.py`, `out/pscale.json`, 10 instances, figure 3).
20, 30 or 40 assumptions are hidden; the others are shown and enter as evidence. Median width, specification alone:

| hidden | 64 openings | 256 | 1,024 | 4,096 | 16,384 |
|---|---|---|---|---|---|
| 20: certain / audit 4,096 draws / audit 65,536 draws | 0.002 / 0.000 / 0.000 | 0 | 0 | 0 | 0 |
| 30 | 1.00 / 0.068 / 0.006 | 0.87 / 0.006 / 0.001 | 0.000 | 0 | 0 |
| 40 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0.97 / 0.114 / 0.013 | 0.19 / 0.001 / 0.000 |
| 40, only 30 of the checks observed (POST HOC, 5 instances) | - | - | 1 / 1 / 0.66 | 1.00 / 0.94 / 0.16 | 0.94 / 0.21 / 0.039 |

- At 20 hidden: no exact answer outside any interval, no audit miss. At 30 and 40 there is no exact answer; intervals
  were nested in every run, and (POST HOC, `pscale2.py`, `out/pscale2_summary.out`) the audit intervals at 1,024 and
  4,096 openings always met the certain interval of the same tree at 16,384 and contained its audit point (200 pairs).
- The certain interval is the limit of the guarantee: it needs every state with a non-negligible bound opened. The
  openings for a width of 0.1 to 0.2 grow at least 20-fold from 20 assumptions with 30 checks (512 to 768) to 40 with
  60 (16,384); with half the checks it is still 0.94 wide there. The interval from the specification's own draws is
  useful one to two budget steps earlier (0.013 at 4,096; 0.039 at 16,384 with half the checks) and costs table
  arithmetic only. Before the first closed state is found neither says anything. A rollout policy moved from world
  103 is behind the specification alone (4,096 openings: audit 0.272 / 0.028); a random one found nothing in 16,384.

## 8. What this says about the claim, and what it does not

- "Alien or bad search policies change the cost, not the answer" holds on trained world models in the one form that
  can hold at finite compute: the specification keeps the books, the answer is an interval, the policy owns the point
  inside it and the speed at which it closes. The importance form by itself has only the limit statement (FRAME row i).
- "The loop makes the search cheaper over rounds": the first round does (11 to 19 times in Sudoku), later rounds do
  not, in either world. Two reasons I can name. (1) After one round the amplified targets at the fixed budget are
  exact (Sudoku) or as good as the specification's draws make them (parity): no amplification error is left for a
  better policy to shrink. Compounding needs a budget that keeps binding. (2) What remains is the proof that the rest
  is empty; its lever is the order, and the score-proportional target is silent about the order. AlphaZero's policy
  target (visit counts) carries the search's judgement of where effort pays; T's next-step distribution does not.
  CONJECTURE: an order head trained on the cost the search measured (the size of the branch that had to be opened),
  with exploration over orders, is the missing half: your "trained for efficiency", with a signal.
- So "search policy" covered three things: finding the score (the value head; distilled in one round), proving the
  rest empty (the specification's bound and share; nothing learned helped), and the order (a hand rule won twice).
- Limits. Two small worlds whose score is a cheap table and does not depend on the demonstration's order; with a soft M
  whose numbers change with the order the tree would have to count (order, filling) pairs. Two seeds and three worlds;
  6x6 Sudoku leaves little above the hand rule; the parity policy reads the hand-written graph. No trust head, no
  decisions (Q4), no GPU. The rounding bug shows the books need outward rounding before anyone relies on them.

## 9. Claims

| # | claim | status | evidence | depends on |
|---|---|---|---|---|
| 1 | Weighted counting needs only the value probabilities; the search's order cancels; any full-support policy converges to T | PROVEN; CHECKED exactly | section 2; `out/check_order.out` | score does not depend on the demonstration's order (true in both worlds) |
| 2 | The books' interval contains the exact answer for every policy; nested | PROVEN; verified: 0 outside in 10.4M + 0.5M intervals, apart from 3 rounding artifacts of the first Sudoku scoring (0 where re-run after the fix) | section 2; `out/s_s*.json`, `out/p_w*.json`, `out/sdebug_*.out` | Q3 (in proportion), settled |
| 3 | Bad policies (random, wrong goal, avoiding) cost openings or width only; unguarded avoidance leaves the interval open and visible | FINDING, CHECKED | section 4 | - |
| 4 | The importance form alone gives a wrong answer with no warning under an avoiding policy (0.30-0.42) | CHECKED | section 4 | - |
| 5 | One round of distillation cuts Sudoku closing cost 11-19 times and parity importance error 6-13 times | CHECKED, 2 seeds / 3 worlds | section 3 | - |
| 6 | Rounds 2-5 do not compound; Sudoku cost rises about 2 times; H3, H4, H5 null | CHECKED (registered) | section 3 | fixed budget that stops binding after round 1 |
| 7 | The loop beats the one-shot control only when the untrained search finds little | CHECKED on 2 seeds (one each way) | section 3 | - |
| 8 | Hand-made order beats the learned one (1.7-3.9 times in Sudoku; narrowest interval in parity) | CHECKED | sections 3, 6 | - |
| 9 | All extra cost is in branches with no solution; from the hand rule's level each round of distillation makes the tree dearer (46 -> 95) and the rollouts better | CHECKED, POST HOC (diagnosis one seed; order run two seeds) | `out/sdiag_s1.out`, `out/mask_summary.out` | - |
| 10 | An order trained on measured search cost would compound. Anything with a trust head, decisions, an order-dependent soft score, 9x9 | CONJECTURE, not run; UNKNOWN | section 8 | Q4, Q7 open |
| 11 | No narrowing by the registered test (weak tail signal); the guard is the specification's share in the parity world, the floor in Sudoku | CHECKED | sections 4, 5 | - |
| 12 | Tree + specification's draws: 0.003-0.011 error at 256 calls against 0.19-0.22 for one pass of M (H8 major); the policy's own one pass is worse than M's | CHECKED | section 6 | - |
| 13 | At 40 assumptions the certain interval stays 0.2-0.9 wide at 16,384 openings; the probabilistic one is at most 0.04 | CHECKED (registered + post hoc; no exact answer at this size, cross-check only) | section 7 | - |

## 10. Files (all under `wf13/loop/`)

`PREREG.md`, `PREREG.sha256`, `PREREG-ADDENDA.md`, `RESULT-work.json`. Tables: `mk_tables.py`, `mk_onepass.py`. Sudoku: `sk.py`, `spol.py`,
`srun.py`, `sscore.py`, `sexact.py` (scoring only), `sdebug.py`, `sdiag.py`. Parity: `pw.py`, `ptree.py`, `ppol.py`, `ploop.py`, `prun.py`,
`pscore.py`, `pexact.py` (scoring only), `pscale.py`, `pscale2.py`. Also `check_order.py`, `check_firewall.py`, `report.py`, `summ.py`, `figs.py`;
pilots `pilot1.py`, `pilot2.py`, `spilot.py` (dev worlds 7, 311). Outputs `out/` (`report.md`: every arm), `ckpt/`, `logs/`, `fig/`. Python: `/home/agent/.venvs/aggext/bin/python -B`.
