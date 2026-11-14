
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
