# Addenda to PREREG.md (SHA-256 82efe89c67c7..., hashed 2026-11-10T17:19:18Z; `PREREG.sha256`)

**A1 (17:27Z; a bug found by the registered guarantee check; fixed; affected arms re-run with the same seeds).**
The first scoring of Sudoku seed 1 reported 3 exact answers outside their certain interval (2 in the one-shot arm, 1 in
avoid/no-guard). Cause: `sk.books` computed the bound of the frontier states that exclude (cell, value) as
`U.sum() - UA - Uu`. With frontier bounds near 1e9 to 1e10 the subtraction returned about -1.4e-06 where the true value
is 0, and the lower bound came out a hair above 1 (exact answer 1.0000). Fix: the term is summed directly.
- `sdebug.py oneshot` and `sdebug.py avoid` re-grow those two arms on all 260 test puzzles and count both ways: old
  formula 2 and 1 violations, fixed formula 0 and 0 (`out/sdebug_oneshot.out`, `out/sdebug_avoid.out`).
- A full rescoring of seed 1 (`srescore.py 1`) was started and stopped after rounds 0 and 1 (identical numbers, 0
  violations; `out/s_s1_rescore_partial.json`) because the shared machine was at load 40 and the slot was needed. So
  `out/s_s1.json` is the FIRST scoring (old formula; its violation column shows the 3 artifacts); the other 13 arms of
  seed 1 had 0 violations under the old formula and were not re-run. This is a deviation from "rerun with the same
  seeds" in PREREG section 6, stated here.
- The loop does not call `books`, so the trained networks are unchanged. Seed 2 and every later run used the fixed
  formula for all arms: 0 violations. The parity books never subtract bounds.

**A2 (17:37Z; two sentences of H6 that the results contradict; reported as they came out).**
- "Avoid/no-guard never closes a Sudoku puzzle": it closes 34 to 37% of hard puzzles. The specification books every
  child of an opened state, and a child that completes the board is found without being opened, so the avoiding policy
  still closes a puzzle when its only avoided step comes last.
- "Avoid + share closes every Sudoku puzzle inside the cap": 92 to 93% of hard puzzles. The specification's share
  (largest upper bound first) is breadth-first in a rule world and reaches deep avoided states slowly.

**A3 (post hoc runs, added after the registered results were seen; marked POST HOC in NOTE.md).**
- `sdiag.py 1` (18:11Z): where the openings go (states that contain the solution, or branches with none).
- `srun.py 1 order=confmask tag=_mask arms=min n_mixed=60 n_multi=20` and the same with seed 2 (18:30Z): the order rule
  reads the books' candidates (most confident after colliding values are dropped). Scored on the loop rounds, the
  one-shot control and the hand rule only; same hard test puzzles; 60 mixed and 20 multi-solution puzzles.
- `pscale2.py 60 5` and `pscale2.py 30 5` (18:29Z): world 201, 40 hidden assumptions, the specification alone;
  cross-check of the audit against the certain interval at a larger budget; and the same with only the first 30 of
  the 60 checks observed.
- Not run although listed in an earlier draft of this addendum: the tight-budget variant (`B_train=128`).

**A4 (pilots and smoke runs before the hash; disclosed in PREREG section 4).** Dev worlds 7 and 311 (`pilot1.py`,
`pilot2.py`, `prun.py 7 0 ... tag=_smoke*`, `prun.py 311 0 ... tag=_smoke*`, `pscale.py ... 311`); Sudoku training
puzzles (`spilot.py`) and seed-0 smoke runs that scored the first 16 hard, 8 mixed and 6 multi-solution test puzzles
(excluded from the registered test sets by `offset=1`). Their outputs are kept in `out/` and `logs/` with those tags.
