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
