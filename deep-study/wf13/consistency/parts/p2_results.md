## 2. Results (registered unless marked post hoc)

Accuracy is **excess** log loss in nats against the exact posterior (0 = exact) over all test questions (7,922 /
7,681), mean of K = 1 and K = 4 unless a K is named. Cells: world 101 / world 102, mean of two runs on disjoint real
subsets. `*` = the 95 % paired-bootstrap interval excludes 0 in both runs, same sign. Every arm, weight, n and K, for
excess and for real outcomes: `out/tables.md`. Figures: `fig/fig1_excess_by_n.png`, `fig/fig2_pair_mismatch_by_n.png`.

### (a) Where the numbers sit (CHECKED; `out/w10*/summary.json`, `parts/refs.json`)
- Always one half: 0.046 / 0.063. One pass of M: 0.034 / 0.036 (size S 0.039 / 0.049, size L 0.032 / 0.034). The
  score's exact answer: 0.016 / 0.0001; in world 101 the score is wrong on checks 12 and 25 (registered rule).
- The amplified predictor with 20,480 real vectors: 0.034 / 0.034 at K = 1, 0.029 / 0.029 at K = 4. The gain is on
  questions with 20-29 revealed checks: one pass 0.187 / 0.182, K = 4 0.140 / 0.129. With 64 real vectors: 0.045 /
  0.047, worse than one pass. The score's exact weights on the same trajectories: 0.72 / 0.78 and 0.20 / 0.24.

### (b) Accuracy by number of real check vectors, for each form and weight

**Table A. Excess against the exact posterior.**

@@A@@

**Table B. Gain = 1 - excess(arm) / excess(real outcomes only).**

@@B@@

Log loss on the held-out real outcomes moves the same way (the exact posterior scores 0.644 / 0.648): at n = 64 and
n = 20,480, real-outcomes-only 0.691 / 0.687 and 0.676 / 0.671; (i) at weight 10 0.694 / 0.691 and 0.682 / 0.675; (ii-s)
at weight 1 0.684 / 0.681 and 0.678 / 0.673; M's samples 0.685 / 0.681 and 0.676 / 0.670.

- **Your cross-order form (i) did not improve data efficiency.** With 64 and 256 real vectors: +5 % / -7 %* and
  -3 % / +4 % at weight 1; -3 % / -20 %* and +8 % / -13 % at weight 10. The same condition on whole orders (ii-o):
  +15 % / -2 % and +4 % / +6 % at weight 1, not significant in both runs. From 1,024 on, both change nothing at weight
  1 (+3 % to -7 %) and cost 6-30 % at weight 10. No registered bin fits exactly; I read both as NULL.
- **Anchored to the score (ii-s) it helps a little, where data are fewest:** +17 %* / +6 % and +15 %* / +6 %* at
  n = 64, 256 (weight 1); nothing at n >= 1,024; a loss of 21-56 % at weight 10. Registered bin: MINOR. "Largest
  when data are fewest": yes in direction in both worlds, by the registered margin only in world 101.
- **It adds nothing over the plain alternative.** Outcomes sampled from M give +15 %* / +12 %* and +18 %* / +7 %, and
  never hurt at large n. With NO real outcome that arm scores 0.035 / 0.034, as good as about 1,000 real vectors.
  (ii-s) alone reaches 0.048 / 0.055: as built (96 terminal vectors, 12 trajectories per batch) it passes the score's
  law on inefficiently. (i), (ii-o) alone collapse to the constant, as wf12's table predicted; (iii) alone diverges.
- **With few real data this head should not be trained freely at all:** at n <= 256 every arm (0.035-0.057) is at or
  above one pass of M. (iii), wf10's form: NULL again. A one-pass network trained on the same real vectors never
  leaves the constant's level: nothing here is memorised.

### (c) The variance over orders: M itself, the amplified predictor before and after, and compute

D = mean squared log difference between the two orders' joint of a pair (192 held-out real triples). V_ord = variance
over four orders of the summed log step predictions of one real check vector (24 vectors; the log probability is
about -18). **Table C**; in brackets, the same quantity between two independent trajectory draws of ONE order (the
noise floor).

@@C@@

- **M itself is almost consistent** (first row), because one pass of M says little about one check given others.
  **The amplified predictor's order variance is its sampling noise:** D is 8 to 30 times M's and equals its noise
  floor to within 0.03. For whole orders, V_ord minus floor is 0.36-0.61 from 1,024 real vectors on: M's own level.
- **It does not fall from K = 1 to K = 4.** D rises in five of six cells, with its floor: at K = 4 the predictor
  leans harder on its samples. For the score's exact weights it falls fast, as the guarantee says, from a level about
  100 times higher. Beyond K = 4: UNKNOWN.
- **The losses do remove it.** (i) at weight 10 cuts D 20 to 40 times, below M's own, and the floor to zero; (ii-o)
  cuts V_ord 2 to 10 times. (ii-s), (iii) and M's samples leave both where they were (`out/tables.md`).

### (d) The October failure, on the amplified predictor

On each held-out triple, the order whose pair joint is closer (KL) to the exact joint under real-outcomes-only is the
"better" one. **Table D**: the change of that order's distance in the arm (K = 4), minus the same change in a second
real-outcomes-only run with another initialisation. Positive = the better order got worse. The gap between the two
orders under real-outcomes-only is 0.05-0.12. `*` = interval excludes 0.

@@D@@

- **It reappears.** With (i) or (ii-o) at weight 10 the better order gets worse by 0.03-0.10 at n >= 1,024 in both
  worlds, and at n = 64 in world 102 (there (i) drags at weight 1 too), while the mismatch falls 20-fold. The
  registered rule (n = 64 and 1,024, both worlds) is one cell short: world 101 at n = 64.
- **The mechanism** (post hoc; `cmech.py`, `out/mechanism_w10*.json`; n = 20,480, K = 4). Real-outcomes-only moves
  0.050 / 0.044 in probability between two trajectory draws and scores 0.140 / 0.129 on 20-29 revealed checks. With
  (i) at weight 10 it moves 0.003 / 0.004 and scores 0.187 / 0.190, which is one pass of M (0.187 / 0.182): the loss
  switched the amplification off. (iii-sym), built to drag, does the same (0.004 / 0.006; 0.183 / 0.174) and costs
  6-14 % from n = 1,024 on; (iii) with its stop-gradient does not.
- **Anchoring to the score does not prevent it; a small weight does.** (ii-s) at weights 0.1 and 1 shows no drag up
  to n = 1,024 and a small one at 20,480 (weight 1). At weight 10 it drags like the unanchored forms, also in world
  102, where the score is right: the predictor still reads its samples (0.035) but follows the 128-trajectory
  estimate of the score (mean absolute error 0.20 / 0.26 nats per vector) and lands at 0.169 / 0.174.
- Where the score is wrong (world 101, checks 12, 25; K = 4, n = 20,480): real-outcomes-only 0.044; (ii-s) 0.043,
  0.047, 0.051 at weights 0.1, 1, 10; M's samples 0.046. Small, in the predicted direction.

### (e) Post hoc: a world where reality scores only one kind of question

One change: real outcomes are used (training and checkpoint choice) only on questions with at least 25 of the 30
checks revealed. **Table E**: excess on all test questions and by number of revealed checks, K = 4, one run per cell
(`ctrain_ph.py`, `out/w10*_ph25/`).

@@E@@

- With real outcomes on one family only, the predictor is wild elsewhere. **Either cross-order term repairs it**:
  0.032-0.040 at n = 20,480, against 0.029 with real outcomes on every question. On the questions real data never
  scored the result is about one pass of M (10-19 revealed: 0.040-0.047 against 0.037 / 0.035), not the amplified
  level: the terms carry consistency down, not the amplification.
- (iii), over compute, cannot do this (it compares a question with itself). M's samples can (0.031 / 0.030).

### (f) Post hoc: a pair term with the sampling noise removed (failed)
Product of the mismatches from two disjoint trajectory draws; its expectation is the squared expected mismatch
(`ctrain_ph.py` arm `idb`, world 101, `out/w101_phdb/`). Worse than real-outcomes-only at every n and weight (excess
+0.012 to +0.031); at weight 100 it collapsed to a constant. The product is not bounded below and is very noisy.
