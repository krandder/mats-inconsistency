# PREREG: Sudoku in Kelvin's architecture (wave 7)

Written 2026-11-10, about 02:00-02:06Z, by one Claude Fable 5.2 researcher. No registrar or skeptic agent exists in this
wave, so the timestamp is the file's mtime and the hash list in section 9. After the first comparison run, changes
go into `PREREG-ADDENDA.md` only, dated, and they never change a threshold.

**What had been run when this file was written.**
- `gen_base.py` (world generator only, no model): grid enumeration, orbits, the split, the counter self-test and
  the puzzle pilot. Output `out/gen_base.out`. I used its puzzle statistics to set the generator parameters below.
- `gen_puzzles.py` (world generator only) was running.
- Training of the 6x6 grids-only model had started (`out/train_mA6.out`, step 250 of 3000). No model had been
  evaluated on anything.

## 1. The question

Kelvin, 10 Nov: "The world-model should encode the rules of Sudoku (which empty->filled placement are valid), and once
M also encodes the goal (for all cells to be filled), then the search over all possible internal orders will search
over all internal possible fillings to find the best possible solution."

Build exactly that with a trained network, in Bengio's state graph, and measure:
1. where M's atoms are reliable and where they are guesses, on real held-out boards;
2. the trust MDP: an order policy trained on real boards only, and the target T it defines;
3. whether T gives (near) zero to invalid complete boards;
4. whether different search policies, each corrected against the same T, return the same answers at different cost;
5. how that compares with one-pass decoding by M's own confidence with no check, at equal compute;
6. what ordering by M's confidence and ordering by reality-trained trust do where M is confidently wrong;
7. out of distribution: fewer clues than any training puzzle; a grid size never trained on (only if time allows).

## 2. World, data, seeds (generator parameters are fixed here)

- **Boards.** K x K Sudoku: K = 4 (2x2 boxes), K = 6 (2x3 boxes). 9x9 only if the L40 is free (section 8).
- **Reality.** The uniform law over all valid complete grids: 288 grids at 4x4, 28,200,960 at 6x6. Entropy
  ln 288 = 5.663 and ln 28,200,960 = 17.155 nats.
- **Essentially different grids.** Two grids are in the same orbit if a cell symmetry (rows within a band, bands,
  columns within a stack, stacks, transposition at 4x4) and a digit relabelling map one to the other. 4x4 has 2
  orbits (96 and 192 grids); 6x6 has 49 (`out/gen_base.out`).
- **Splits are by orbit**, because the network has no positional embedding and so cannot tell symmetric grids apart.
  - 6x6: train 34 orbits (66.5% of grids), validation 6 orbits (12.1%), test 9 orbits (21.3%). Orbit shuffle seed 606.
  - 4x4: train = the 192-grid orbit, test = the 96-grid orbit. A second 4x4 model is trained on all 288 grids; it
    supports only exactness checks, never a held-out claim.
  - Fraction of test grids seen in training, up to every symmetry of the game: 0 by construction.
- **Puzzles** (6x6; `gen_puzzles.py`). A puzzle is made from a grid by deleting clues in random order while the
  solution stays unique (uniqueness by the exact counter).
  - `train_min`: 12,000 minimal puzzles, train orbits, seed 1001.
  - `train_12p`: 4,000 puzzles with 12-16 clues (target uniform), train orbits, seed 1002.
  - `val_min`: 1,000 minimal puzzles, validation orbits, seed 2001.
  - `test_min`: minimal puzzles, test orbits, seed 3001, generated until 300 are "hard" or 8,000 exist.
  - `test_12p`: 500 puzzles with 12-16 clues, test orbits, seed 3002.
  - `test_multi`: 200 puzzles with 2-12 solutions, test orbits, seed 3003.
  - Pilot: minimal 6x6 puzzles have 9-13 clues (mode 10).
- **"Hard"** is fixed in advance by the oracle's rating, never by a model: a puzzle is hard if repeated naked singles
  and hidden singles do not finish it (rating 2). In the pilot that is about 4% of minimal puzzles.
- **Test sets used below.** `ALL` = the first 500 puzzles of `test_min`. `HARD` = every rating-2 puzzle of
  `test_min` (at most 300). `FEW` = the puzzles of `ALL` with at most 11 clues. `MULTI` = `test_multi`.

## 3. The architecture (portfolio section 1.1), component by component

- **M (world model).** A masked any-order transformer over cells (`core.RelFormer`), trained with the any-order loss
  on real data only. No consistency loss, no negatives, no solver output.
  - Variant A, "grids only": `mA6` (d 64, 4 layers, 201,030 parameters; 3,000 steps, batch 256) on uniform grids of
    the train orbits. 4x4: `mA4` (train orbit only) and `mA4full` (all 288 grids), d 48, 3 layers.
  - Variant B, "puzzle + solution": `mB6` (same size) on `train_min` with the real solutions; clue cells carry a flag.
    Size sweep for the equal-compute rule: `mB6t` (d 32, 2 layers) and, if it finishes in time, `mB6L` (d 128,
    4 layers).
  - Relational prior, disclosed: no positional embedding; three of the four attention heads are restricted to the
    cell's row, column and box, and one is global. The network is told which cells share a group. It is never told
    what sharing implies.
  - M's inconsistency is measured, never removed: the spread over orders of log q_sigma(x) on held-out grids.
- **State graph (Bengio).** State = partial filling. Edge = fill one empty cell. Trajectory = an order together with
  the values. Terminal state = complete board. An order policy pi(cell | state) and M's atoms q(value | state, cell)
  together are a forward policy; with a deterministic pi each complete board has exactly one route.
- **Trust (part of the world model, reality only).** The trust MDP of the common brief: state = partial filling of a
  REAL board; action = the cell reality reveals next; cost = M's surprise at the revealed value.
  - 4x4: exact dynamic programming over every reachable state of the 288 grids (oracle reference: it uses the true
    law), and the learned policy below trained on the train orbit only.
  - 6x6: `core.fit_trust`, paired-route policy iteration. For a real board x, compare two routes to the same x
    (the current policy's route, and the same route with one deviating cell). The difference of M's total surprise
    is a sample of the difference in excess error (reality's entropy cancels exactly). A small network (d 64,
    2 layers; inputs: the state and M's atoms at it) regresses that difference; the next policy reveals the cell with
    the lowest predicted value. 4 iterations x 1,536 real boards x 6 deviations, starting from "lowest entropy of M
    first". Start states for variant A: 60% the empty board, 40% a uniformly random subset of the same real grid.
    For variant B: the puzzle's clue state, with its real solution.
  - pi_trust := the iterate with the lowest mean route surprise on 512 real validation boards. The starting policy is
    not a candidate. Trust never reads a search statistic.
- **T.** T_A(x) := the density of x under "pi_trust picks the cell, starting from the empty board; M_A's atom gives
  the value". It is normalised, and it costs 36 calls of M_A and 36 of the trust net. For a puzzle with clues c the
  target is T_A conditioned on the clues, T_A(x) 1[x agrees with c] / Z. Evidence conditions T; it does not clamp M.
  - Acceptance rule for "T > 0" with a network that never outputs an exact zero: accept x iff log T_A(x) >= tau,
    where tau := (the minimum of log T_A over 2,000 real validation grids) - 2.0 nats. tau is set from real grids
    only, with no negative example. The route is abandoned once its running sum is below tau.
  - Second variant: T_B(x | c) := the route density from the clue state under pi_trust^B and M_B's atoms. Its
    acceptance rule: accept x iff T_B(x | c) > 1/2 (at most one board can pass).
- **Amplifier (search).** Candidates are complete boards that agree with the clues. A candidate is produced by a
  search policy walking the same graph from the clue state. First-accept, never revoked, nested budgets. For
  puzzles with several solutions: self-normalised importance weights T_A(x)/proposal(x).
- **Distillation** goes only into a proposal: `D6`, a variant-B-shaped network trained on `train_12p` puzzles with
  the answers the search itself accepted (no real solution is used).
- **Utility:** none in this experiment. The "goal" is the terminal condition "complete board with T > 0".

## 4. Arms

Order rules called "M's own confidence": `maxprob` (highest top probability first; Kim et al. top-prob), `margin`
(largest top-1 minus top-2), `entropy` (lowest entropy first). The rule used for a network's own-confidence order
in sections E4-E6 is `margin`.

Search policies, all corrected against the same T_A (cap 256 candidates per puzzle; candidate 0 is the greedy
route where stated):
- **S1** M_A, own confidence (`margin`), values drawn from M_A's atoms; candidate 0 greedy.
- **S2** deliberately bad, with a floor: M_A, least-confident cell first, values from 0.95 x M_A's atom + 0.05 x
  uniform; no greedy candidate.
- **S3** distilled: `D6`, own confidence, values from its atoms; candidate 0 greedy.
- **S4** M_B, own confidence, values from its atoms; candidate 0 greedy.
- **S5** M_A in the order of pi_trust^A, values from M_A's atoms; candidate 0 greedy.

Baselines with no check:
- **K1** (Kim et al.): one pass of M_B, one cell per step, own confidence, most probable value. Also with `maxprob`
  and `entropy`, and for `mB6t` / `mB6L`.
- **K2** best of k passes of M_B scored by M_B's own route log-probability (the natural no-check use of more compute).
- **K0** one pass of M_A, own confidence.
- **KT** one pass of M_B in the order of pi_trust^B (same calls of M_B plus the trust net).

Oracle (scoring only): exact counter, validity check, rating. Enforced by `check_firewall.py`: `core.py`, `data.py`,
`train_m.py` do not import `oracle`; the experiment scripts call the oracle only in their scoring sections.

Compute is counted for every arm as FLOPs = 2 x parameters x 36 tokens per forward pass of each network (M,
proposal, trust net), summed. "Calls" are forward passes on one board.

## 5. Experiments, metrics, pre-stated outcomes

**E1. Per-step held-out surprise by context type** (6x6, M_A; exact conditionals from the counter, scoring only).
States: 3,000 states on routes of held-out test grids (random order, own-confidence order, trust order: 1,000
each). For every empty cell: M's surprise at the real value, the exact entropy, the exact excess KL(p || q), the mass
M puts on values that break a rule directly (a filled peer has the value), and on values that are locally allowed
but have no completion. Context types: (a) locally forced (one locally allowed value); (b) forced but not locally
(the exact conditional is a point mass, two or more values locally allowed); (c) free (exact entropy > 0).
- Pre-stated reading of "M encodes the rule": *sharp* if the mean mass on directly rule-breaking values is < 1e-3
  per atom; *partial* if < 1e-2; *no* otherwise.
- Descriptive, no threshold: excess by type and by number of filled cells; calibration of type (c).

**E2. Trust MDP and the routes it prefers.**
- Metric: excess route surprise on 2,000 held-out test grids, mean of -log T_pi(x) minus 17.155, for pi in:
  pi_trust, the three own-confidence rules, least-confident-first, raster, a random order (mean over orders), and the
  uniform mixture over 32 random orders (log-mean-exp; an estimate). Paired bootstrap 95% intervals.
- 4x4, exact: the same list plus the exact optimum of the trust MDP, on all 288 grids and on the 96 held-out ones.
- **Headline bins, trust against own confidence as the definition of T** (conf* = the own-confidence rule with the
  lowest validation surprise):
  - *major*: excess(pi_trust) <= 0.5 x excess(conf*) and the paired interval of the difference excludes 0;
  - *minor*: excess(pi_trust) < excess(conf*), interval excludes 0;
  - *null*: interval includes 0;
  - *negative*: pi_trust is worse and the interval excludes 0.
- Description of the learned order (no threshold): how often the chosen cell is of type (a), (b), (c) compared with
  what is available; the mean exact excess of the chosen cell against the mean and the minimum over available cells.

**E3. T on complete boards.**
- 4x4, exact: the mass of T on the 288 valid grids; the largest T on any invalid board (exhaustive walk of T's
  tree, pruned at 1e-9); the same for the own-confidence order.
- 6x6: the mass on valid grids from 4,000 draws of T (fraction valid, Wilson interval). log T on 2,000 held-out
  valid test grids; on near-miss invalid boards (one cell changed; two cells of a row swapped; 2,000 each); and on
  every wrong candidate that any search arm proposes in E4.
- **Bins for "T separates valid from invalid" at the pre-set tau**: *pass* if false rejects on held-out valid grids
  <= 1% and false accepts <= 1e-3 in each class of invalid boards; *partial* if <= 5% and <= 1e-2; *fail* otherwise.

**E4. Search corrected against T.** Sets `ALL` and `HARD`.
- (a) Same answers. For every puzzle and every pair of policies that both returned an accepted answer: are the
  answers identical? Is every accepted answer the real solution? *Pass*: 100% identical and >= 99.9% of accepted
  answers correct. That the answers agree *if* T accepts only the solution is a guarantee, not a finding; the finding
  is whether the learned T accepts only the solution.
- (b) Different compute. Per policy: fraction solved within the cap, and mean and median FLOPs to the accepted answer.
- (c) Against the Kim baseline at equal compute, on `HARD`. Anytime rule for the search: before anything is accepted
  the answer is candidate 0 (unchecked); once a candidate is accepted it is the answer for good. Accuracy as a
  function of FLOPs for S1-S5, with K0, K1 (three sizes), K2 (k = 1..64) and KT as points or curves on the same axes.
  **Headline bins**, S1 (no puzzle ever seen in training) against K1/K2:
  - *major*: at the FLOPs of some K1/K2 point, S1 is at least 10 accuracy points above every baseline point of equal
    or lower FLOPs, or S1 reaches >= 99% while every baseline point stays below 90%;
  - *minor*: S1's final accuracy exceeds the best baseline point, but only at more than 10 times that point's FLOPs;
  - *null*: the best baseline point is already >= 99% on `HARD` (no headroom); then only compute-to-solve is reported;
  - *negative*: S1's final accuracy is below the best baseline point.
  The same bins are reported for S4 (M_B as proposal, checked) against K2 (M_B, unchecked).
- (d) Several solutions (`MULTI`). 256 candidates per policy (S1, S2, S4). Estimate the target over solutions by
  self-normalised weights. Report the total-variation distance to the exact conditional of T_A on the true solution
  set (oracle list, scoring only), the distance to the uniform law over solutions, and the distance between
  policies. *Pass* if the median distance between any two policies' estimates is <= 0.10 at 256 candidates.
- (e) Variant B's own target. With acceptance "T_B(x | c) > 1/2": how often an accepted answer exists, and how often
  it is wrong. Descriptive.

**E5. Own confidence against reality-trained trust where M is confidently wrong.**
- Found contexts (M_B). On `ALL` and `HARD`, take every puzzle where K1 fails; the state of its first wrong step is
  a context where M_B's most confident atom is wrong. Report M_B's probability on the wrong value there, the cell
  pi_trust^B picks at the same state, whether M_B's most probable value at that cell is right, and M_B's surprise at
  the truth for both cells.
- One-pass accuracy KT against K1 (best own-confidence rule on validation) on `HARD` and `ALL`. **Bins**: *major* if
  KT is >= 5 points above on `HARD` with McNemar p < 0.01; *minor* if above with p < 0.05; *null*; *negative*.
- Constructed contexts (M_A, exact counter for scoring). Screen 20,000 states on real test grids for atoms that are
  confidently wrong: top probability >= 0.9 on a value whose exact probability is <= 0.1. At states that contain one:
  how often each order rule picks a confidently wrong cell, and the mean exact excess of the picked cell.

**E6. Out of distribution.**
- Fewer clues. `D6` has seen only puzzles with 12-16 clues. On `FEW` (9-11 clues) against `test_12p`: its one-pass
  accuracy without a check; and S3 (same network as proposal, corrected against T_A): fraction solved, correctness of
  accepted answers, FLOPs. S1 on the same sets. *Pass* for the architecture: accepted answers >= 99.9% correct on
  `FEW`. Descriptive: the compute ratio `FEW` / `test_12p`.
- Unseen size, only if time allows: one size-generic network (9 symbol slots, random slot assignment per board)
  trained on 4x4 and 6x6 grids, applied to 9x9 with no further training. Whether its route density separates valid
  9x9 grids from near-miss boards, and whether S1 solves 9x9 puzzles within a cap. Reported whatever it shows; if it
  is not run, the note says so.

## 6. What is true by construction (reported as guarantees, not findings)

- T is normalised and has one route per board (deterministic order policy).
- The chain rule: for every adapted order policy, expected route surprise on real boards = entropy of reality +
  KL(reality || T_pi). Checked exactly at 4x4.
- First-accept search never revokes, so accuracy-if-accepted is non-decreasing in budget for any acceptance test.
- If T accepts only the solution, every search policy with full support returns the same answer.
- Importance weights are consistent for any proposal with full support (the floor in S2 gives that).

## 7. Analysis rules

- Every reported number comes from a script in this directory with its output in `out/`.
- Held-out means test orbits and the seeds of section 2. Thresholds, tau, the choice of pi_trust and the choice of
  conf* use validation data only.
- Every outcome named in section 5 is reported, including nulls and negatives. Anything not run is listed as not run.
- Intervals: bootstrap over puzzles or grids (2,000 resamples), Wilson for proportions.
- One seed per network. I say so wherever a claim rests on one trained network.

## 8. Compute

- CPU only unless the queue opens: `sbatch --test-only --gres=gpu:1` on the MATS cluster at 01:39Z returned a start
  time of 2026-11-21 (dry-run ids 57288, 57289; nothing was submitted). That is over the 30-minute limit, so 9x9
  training is not planned. I re-check once before the end.
- Host: shared, load about 50 on 16 cores. At most 3 processes, `nice -n 19 ionice -c3`, one thread each.

## 9. Code at registration (sha256, first 16 hex digits)

```
4ea171ee995779bf  oracle.py        world generator and exact truth (firewalled)
d9199e03c24a7789  core.py          networks, order policies, routes, trust learner, searchers
a3358487b20e6d7a  data.py          streams of real data
4ef100e954ec73f8  gen_base.py
b9195b6b4d37275c  gen_puzzles.py
89b3861f7d072e58  train_m.py
```
The experiment scripts (`e_4x4.py`, `e_trust6.py`, `e_search6.py`, `e_ood.py`) are written after this file; their
hashes are appended to `PREREG-ADDENDA.md` before each is first run on test data.
