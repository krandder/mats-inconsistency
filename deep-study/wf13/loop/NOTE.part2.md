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
