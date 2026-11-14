# Addenda to PREREG.md (dated; thresholds are never changed here)

## A1. 2026-11-10 02:10Z, before any experiment script was run on any model

Reductions for compute (the host runs at load 50 and gives about one core per process), and details the main file
left open. None of them touches a threshold or an outcome bin.

1. E2: the "uniform mixture over 32 random orders" is estimated on the first 500 of the 2,000 test grids.
2. E2 description of the learned order: 300 held-out grids (every step of every route is scored with the counter).
3. E4(d): 100 puzzles of `test_multi` and 128 draws per policy (not 200 and 256). The pass rule is read at 128 draws.
   Draws only: the greedy candidate is not a draw from the proposal, so it carries no importance weight.
   Candidates rejected by tau get weight 0.
4. E4: S2 (the deliberately bad policy) is run on the first 150 puzzles of ALL and the first 100 of HARD, because it
   is expected to use the full cap on most puzzles. "Same answers" is judged on the puzzles it shares with the others.
5. 4x4 has no validation orbit. The selection of the trust iterate and tau use the training-orbit grids (real
   training boards). `mA4full` is trained only if time allows.
6. Distillation: the labels for `D6` come from S1 with a cap of 32 candidates on `train_12p`; puzzles without an
   accepted answer are dropped. `D6` trains for 2,000 steps (M_A and M_B: 3,000).
7. E4(e): a board with T_B(x | c) > 1/2 must be the greedy route of M_B in the trust order (each factor on its route
   is then above 1/2, so each value is M_B's most probable one). So the rule is evaluated on that one board.
8. Search candidates in E3: a candidate that agrees with the clues and is not the solution is an invalid board
   (the solution is unique), so "wrong candidate accepted" is the false-accept count.
9. 4x4 exact DP: off the table of the tabular (training-grids) policy the route continues in M's lowest-entropy order.

Hashes of the experiment scripts at first run (sha256, first 16 hex digits):
```
d9199e03c24a7789  core.py
e3f65ecadfb61d6c  e_4x4.py
f4ee9966d92df3c6  e_trust6.py
e8d98be2059c706d  e_trustB.py
7af7f3264be1431e  e_search6.py
29de4031b78ce208  e_distill.py
8d812d495a095dda  eval_lib.py
4d84e28006493a20  check_firewall.py
```
`core.py` is unchanged since PREREG.md (same hash).

## A2. 2026-11-10 02:27Z, before any experiment script was run on a 6x6 model

The host load rose to about 70 (each of my processes gets under half a core). Further reductions, none touching a
threshold or a bin:

1. Trust learner (both variants): 4 iterations x 1,024 real boards x 6 deviations (PREREG said 1,536 boards).
2. `e_trust6.py` is split into `fit` (trust policy, selection, tau; needed by the search) and `analyse` (E1, E2, E3,
   E5 constructed), so that the search does not wait for the analysis. Same seeds, same test and validation grids.
3. E2: the mixture over random orders uses 16 orders on 300 grids; non-selected trust iterates are scored on 1,000 grids.
4. E2 order description: 200 held-out grids. E5 constructed: 10,000 states screened (PREREG said 20,000).
5. E3 for the own-confidence order (the comparison T): 1,000 near-miss boards per kind and 2,000 draws. For pi_trust the
   numbers of PREREG stand.
6. The first 4x4 run (`e_4x4.py mA4 skipdp`) started at 02:17Z before this addendum; it does not use any 6x6 model.

Hashes at 02:27Z:
```
3eaae5d383e15057  e_trust6.py
f81057f83575b3c6  e_trustB.py
2be360d6653a159b  e_search6.py
5f030869eafe8c24  e_distill.py
572c6d2235f7daa5  e_report6.py
6c50054cf5fcab9f  e_fig1.py
dd4200809841334f  train_gen.py
da0b0586849bea03  e_size9.py
```

## A3. 2026-11-10 02:31Z

1. 4x4 exact DP: computed on one mask per symmetry class of the network (the network is exactly equivariant under
   the 64 cell permutations that keep rows, columns and boxes; the DP value is invariant). Two checks are printed:
   the numerical equivariance of M's atoms, and V*(empty) against the route surprise from walking the policy.
   `e_4x4.py` changed for this (the run without DP that started at 02:17Z used the earlier version; its sections
   [1], [3], [4b], [5] are unchanged in the new version).
2. K2 (best of k passes, no check): k up to 32 on HARD and up to 16 on ALL and T12 (PREREG said 64).
3. S5 is run on HARD only.

## A4. 2026-11-10 02:35Z, after the first 4x4 table (`out/e_4x4_mA4_nodp.out`), before any 6x6 analysis

The 4x4 table shows the trust policy doing best on the training orbit and worse than M's own confidence on the
held-out orbit. The held-out orbit has probability zero under the training law, so "trust from reality" cannot see it.
To separate the two effects at 6x6 I add one column that PREREG did not have: the E2 table on 1,000 fresh grids drawn
from the TRAINING orbits (in distribution for M and for the trust learner). The pre-registered headline stays the
test-orbit number; the new column is reported next to it and labelled as added.

## A5. 2026-11-10 02:42Z (compute; before the distillation step ran)

The host is slower than planned (each process gets about 40% of a core). To finish:
1. Distillation labels: S1 on the first 2,500 puzzles of `train_12p`, cap 8 candidates. `D6` trains for 1,500 steps.
2. `mB6L` (the larger one-pass model) and the unseen-size experiment (9x9) are dropped unless a slot frees up; the note
   says so if they were not run.

## A6. 2026-11-10 02:43Z (an added arm, not pre-registered; before the 4x4 DP run and the 6x6 analysis)

The common brief says the general object should be a learned mixture over routes, and Kelvin's 10 Nov message asks to
"learn how to aggregate". I add the cheapest version: T_mix(x) = sum_k w_k T_k(x) over the order policies already
evaluated (own-confidence rules, least-confident-first, raster, pi_trust), with the weights fit by EM on real boards
(4x4: the training grids; 6x6: 512 validation grids) and scored on held-out grids. T_mix is normalised and its
density costs one route per component. It is reported next to the pre-registered rows and labelled as added.

## A7. 2026-11-10 03:12Z (compute)

The 6x6 analysis (`e_trust6.py analyse`) was on course for about an hour at the current load, ahead of the search
arms in the queue. I stopped it and cut its sample sizes. One number had been printed before the stop and I had seen
it: the test excess of the lowest-entropy order on 2,000 grids, 0.2568 (`out/e_trust6_analyse_killed.out`). Nothing
about pi_trust on test grids had been computed. New sizes:
1. E2 table: the first 1,000 of the 2,000 test grids; random-order mixture 8 orders x 200 grids; the non-selected
   trust iterates are not re-scored on test grids (their validation numbers are in `out/e_trust6_fit.out`).
2. Learned mixture: weights on 256 validation grids. In-distribution column: 500 fresh grids, four policies.
3. E1: 1,200 states. Order description: 100 grids. E3: 1,000 near-miss boards per kind and 2,000 draws for pi_trust
   (500 and 1,000 for the own-confidence T); false rejects on the 1,000 test grids.
4. The order of the queue changed: S1 on HARD runs before the analysis. S2 on ALL uses 100 puzzles.
5. `e_4x4.py` gained a section [2c] (description of the optimal route) and is re-run once with it
   (`out/e_4x4_mA4_order.out`); its first DP attempt crashed on a variable-name clash (`out/e_4x4_mA4_dpfailed.out`),
   was fixed, and the numbers of sections [1], [3], [4b] were identical across the runs.

## A8. 2026-11-10 03:19Z (a result, and an added arm that is NOT pre-registered)

Result first: S1 on HARD (`out/search_S1_HARD.out`) accepted 16 of 2,254 wrong candidates at the pre-set tau, and 6 of
300 puzzles ended with a wrong accepted answer. That fails E4(a) and puts E3 in the "partial" bin at best for this
class of boards. It is reported as it is.

Added arm "T2". My reading of the failure, to be tested by `e_posthoc.py`: a route density has to put probability
somewhere even at a state that is already contradictory, so a wrong board can stay likely along the route if the
first inconsistent step is one M cannot see locally. A check that reads M only at one-cell contexts (the smallest
log q(x_j | x_i alone) over all ordered pairs of cells; threshold theta = the minimum over real validation grids minus
2.0) cannot be fooled that way, because every one-cell context occurs in real boards. T2 := pair check first, then the
pre-registered route threshold. T2 runs are labelled as post hoc everywhere; the pre-registered runs stay the
primary record. S5 is dropped for time; S2 on ALL is dropped (S2 runs on HARD only).

## A9. 2026-11-10 03:59Z (compute, and two post hoc scripts)

1. S3 and S4 are slow on HARD because their proposals fail on many hard puzzles all the way to the cap of 256
   candidates. S4 on HARD was stopped at about 60 of 300 puzzles (`out/search_S4_HARD_killed300.out`: after 50 puzzles,
   82% accepted, 3 of 3,611 wrong candidates accepted) and restarted on the first 100 HARD puzzles. S3 on HARD runs on
   all 300. S2 runs on the first 60 HARD puzzles. S4 on ALL uses 200 puzzles, S3 on ALL 100. T2 re-runs: S1 on all 300
   HARD and on ALL, S2/S3/S4 on the first 60 HARD puzzles. S1 on T12 is dropped.
2. Post hoc scripts: `e_stiff.py` (what M rules out, by context size) and `e_posthoc2.py` (the pair check on T's own
   invalid draws; the sign test for KT against K1; a memorisation audit).
3. The line "states screened 20,000" in `out/e_trust6_analyse.out` [g] is a stale string; 10,000 states were screened
   (addendum A2).
4. (04:04Z) The T2 re-runs of S2, S3 and S4 are dropped for time. Instead `e_posthoc2.py` applies the pair check to
   every wrong board the route check accepted in any arm. T2 search runs exist for S1 only (HARD, and ALL if it
   finishes).
5. (04:19Z) E4(d) was on course for over half an hour (my estimate of its cost was off by a factor of three). It was
   stopped before any output and restarted on 40 puzzles with 64 draws per policy. The pass rule is read at 64 draws.

## Final state, 2026-11-10 04:36Z

Every pre-registered outcome is listed with its result in NOTE.md section 4, including the nulls and the failures.
Not run: S5, `mB6L`, `mA4full`, S1 on T12, the unseen-size experiment (9x9). Hashes of the code as last run:
```
4ea171ee995779bf  oracle.py
f49f317ac83bb499  core.py
a3358487b20e6d7a  data.py
9aed3b8c4b5a5119  eval_lib.py
9224846353989870  train_m.py
6187853c5b44e4cc  e_4x4.py
2517dcfa01718728  e_trust6.py
f81057f83575b3c6  e_trustB.py
464a29d57b045cb0  e_distill.py
58ce2d5fbed18bf6  e_search6.py
d795f1659d820150  e_posthoc.py
9dbd83d96e522ef0  e_posthoc2.py
a9e03610bcf26c91  e_stiff.py
2ff305714ce2a8a9  e_report6.py
4f06bdb38cae1d5d  e_fig1.py
4d84e28006493a20  check_firewall.py
```
`oracle.py` and `data.py` are unchanged since PREREG.md. `core.py` gained the pair check (A8); nothing else in it changed.
