# PREREG: the full loop (rollout policy, amplification, distillation) with an audit the searcher steers

Key `loop`, wf13. Claude Fable 5.2. Written 2026-11-10 ~17:20Z, before any registered run. Frame: `wf9/FRAME.md` section 5.
All code is in this directory; the registered commands are in section 5. CPU only (cluster: `srun --test-only` gave a start
date of 21 Nov for a one-GPU job; test ids 57871, 57872; no job submitted).

## 1. Objects (fixed before the runs)

- **Score** (one, on terminal states; no search policy inside). Sudoku 6x6: step checks from the rule table read from M's
  one-cell contexts (`out/sudoku_rules.npz`, wf7's mA6 and threshold), goal = all cells filled. Parity world: wf7's
  hand-built score from M's small-context numbers (`out/tab_w*.npz`), the observed checks as a factor.
- **Terminal state** = a demonstration (filling + order; assumptions then deductions). In both worlds the score does not
  depend on the order and every filling has equally many orders, so answers about fillings are ratios of filling totals.
  The order in which the SEARCH fills is its own route (search side) and is dropped at submission.
- **Books** (specification): for a partial demonstration, the factors already fixed give a lower and an upper bound on
  the total score of its completions (`pw.Inst.ev`, `sk.log_bound`). Exact ("closed") when nothing is left to bound.
- **Rollout policy** (search side, starts untrained): a small network giving, at a partial state, value probabilities
  for every open variable, plus an order rule. Sudoku: order = its own most confident cell. Parity: in the tree the
  branching variable is the books' one-step lookahead (`greedy1`, hand-made, same for every policy); in the importance
  form the network's own most-confident-first order. Reason: dev pilot, section 4.
- **Amplification, importance form**: K rollouts; weight = exact total of the reached closed state / product of the
  policy's value probabilities (floor eps = 0.02). Converges to T for any policy with full support (PROVEN, standard).
- **Amplification, tree form**: a rollout opens the first unopened state on its path (drawn without replacement by path
  probability); a share rho = 0.5 of openings is the specification's (largest upper bound first). The books give a
  CERTAIN interval for every query. Parity only: the specification's own draws below the open frontier give a
  PROBABILISTIC interval (empirical Bernstein, delta = 0.05 per query; 4,096 draws) and an unbiased point estimate.
- **Distillation**: train the value head on the amplified next-step distribution at the states the tree opened (found
  score plus, in the parity world, the audit draws' weighted completions). Budget per training instance fixed:
  512 openings (Sudoku), 256 openings + 4,096 draws (parity). R = 5 rounds, fresh instances each round, replay of all
  earlier rounds, 400 (Sudoku) or 500 (parity) gradient steps per round.
- **One-shot control**: the same instances and the same total number of gradient steps, but every instance is amplified
  by the untrained policy. **Hand-made comparison**: Sudoku fewest-candidates-first; parity `greedy1` + the books'
  local numbers, with rho = 1 (specification alone) and rho = 0.5.
- **Bad policies** (built from the final network): random; wrong goal (parity: it is shown the opposite observations;
  Sudoku: its value preferences shifted by one digit); avoids part of the space (parity: never proposes 1 for the first
  five assumptions; Sudoku: never proposes digit 0 in the first two rows), with no guard (rho = 0, eps = 0), with the
  floor only, with the specification's share only.

## 2. What is a guarantee and what is a finding

GUARANTEE (to verify; any violation is a bug or a refutation): G1 the exact answer of the score is inside the certain
interval at every budget for every policy; G2 certain intervals are nested as the budget grows; G3 a closed Sudoku tree
returns exactly the exact solution set; G4 the audit interval misses with frequency <= 0.05 per query.
FINDING: everything about cost, compounding, the bad policies' price, narrowing, and scale.

## 3. Hypotheses, metrics, thresholds, bins

Metrics. Sudoku: C = openings until the frontier is empty, per puzzle, censored at 2,048, mean over a test set.
Parity: W_cert(B), W_aud(B) = mean interval width over instances and assumptions at B openings; E_aud(B), E_IS(K) = mean
absolute error against the exact posterior of the score (enumeration over 2^20, scoring code only); F(B) = exact share
of the score in closed states. One network call = one forward pass of the rollout policy on one state.

- **H1 (guarantees)**: zero violations of G1-G3 over all arms, budgets, instances; audit miss rate <= 0.05.
- **H2 (same answers)**: Sudoku: every closed tree, any policy, returns the same solution set. Parity: for rounds 0..5 on
  the same instance and budget the certain intervals share a point (guarantee) and the audit points differ by less than
  the mean audit width (finding).
- **H3 (compounding, Sudoku; primary)**: major: C(final) <= 0.5 C(one-shot) on hard and on mixed puzzles, both seeds;
  minor: <= 0.8 on both, both seeds; else null. Reported with it: C per round, C(final) / C(hand).
- **H4 (compounding, parity tree)**: major: W_aud(final, 256) <= 0.5 W_aud(round 0, 256) and <= 0.8 W_aud(one-shot, 256)
  in >= 2 of 3 worlds; minor: <= 0.8 x round 0 in >= 2 worlds; else null. Dev expectation: null (section 4).
- **H5 (compounding, parity importance form; primary for this world)**: major: E_IS(final, 64) <= 0.25 E_IS(round 0, 64)
  and <= 0.5 E_IS(one-shot, 64) in >= 2 of 3 worlds; minor: <= 0.5 x round 0 and <= 0.8 x one-shot; else null.
- **H6 (bad policies cost width or compute only)**: H1 holds for them. Reported: width and compute ratios against the
  final policy. "No guard fails": avoid/no-guard never closes a Sudoku puzzle and has W_aud(2,048) >= 0.5 in the parity
  world. "The share guards": avoid + share closes every Sudoku puzzle inside the cap and has parity W_aud(2,048) <=
  2 x the final policy's W_aud(1,024).
- **H7 (narrowing)**: parity: KL(T || policy) and the share of T's samples the policy under-proposes 100-fold, by round,
  eps = 0; Sudoku: -log P_policy(solution). "Narrowing" if the final value > 1.5 x its minimum over rounds and the
  under-proposed share > 0.05. Reported: what the floor and the share change.
- **H8 (parity, belief against compute)**: major: E_aud(final, 256) <= 0.1 x one pass of M's error in all three worlds;
  minor: <= 0.5 x. Reported on one axis (network FLOPs): tree found-point, audit point, importance form, the intervals,
  one pass of M, the policy's own one-pass belief.
- **H9 (scale; exploratory, no bin)**: world 201 with 20, 30, 40 hidden assumptions (the others shown), 10 instances,
  budgets 64..16,384 openings, 4,096 and 65,536 draws. Reported: W_cert, W_aud, nestedness, violations where an exact
  answer exists (20 hidden). "Useful" := median W_aud(65,536 draws) <= 0.1 at <= 16,384 openings.

## 4. What the dev pilots already showed (worlds 7 and 311; Sudoku training puzzles)

- `pilot1.py`, `out/pilot2_w311.out` (3-6 instances): no violation; with the hand order and rho = 1 the certain width is
  0.035 at 1,024 openings, the audit width 0.001. A PERFECT value head (exact conditionals used as the policy, pilot
  only) with the most-confident-first order is ten times worse (0.44 at 1,024), and with the lookahead order it finds
  the score sooner (0.975 against 0.67 at 256) but does not narrow the interval. Hence the order rule in section 1 and
  the expectation "null" for H4. The network's own order is kept as one scored arm (`final_conforder`).
- Sudoku smoke runs (seed 0; they scored the first 16 hard, 8 mixed and 6 multi-solution TEST puzzles, which are
  therefore excluded: the registered test sets start at offsets 150, 1,000 and 100): an untrained network closes 8% of
  training puzzles in 512 openings and none in 128, so the fixed budget is 512.
- Parity: at 256 openings an untrained policy finds almost no closed state; the audit draws supply the first targets.

## 5. Registered runs (seeds), in this order, at most three processes

    srun.py 1 ; srun.py 2                       # Sudoku, seeds 1 and 2       -> out/s_s1.json, out/s_s2.json
    prun.py 101 1 ; prun.py 102 1 ; prun.py 103 1   # parity worlds, seed 1   -> out/p_w10x_s1.json
    pscale.py ckpt/p_w103_s1.pt 10              # H9, policy moved from world 103 -> out/pscale.json
    report.py                                   # tables of NOTE.md           -> out/report.md

Defaults in `srun.py` and `prun.py` (CFG) are the registered settings. Test data: Sudoku 100 hard (rating 2) + 100 mixed
unique-solution puzzles + 60 puzzles with several solutions, all from wf7's test orbits (no test grid shares an orbit
with a training grid); parity 60 test instances per world from wf7's test seed, check vectors seen in M's training
removed; training instances from fresh seeds (`pdata.fresh_x`), real check vectors only (the hidden values are never
read by the loop). Worlds 101-103 and 201 have not been searched by any code of mine before this file was hashed.

## 6. Rules

No exact inference or solver in any arm (`check_firewall.py`); exact answers only in `pexact.py`, `sexact.py`, used by
`pscore.py`, `sscore.py`, `pscale.py` after the searches. Anything not listed here is reported as post hoc. If a run
crashes it is fixed and rerun with the same seeds, and the fix is listed in `PREREG-ADDENDA.md`.
