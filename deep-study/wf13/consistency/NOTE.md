# Your consistency requirement on amplified predictions, tested as a data-efficiency device (wf13/consistency)

Model: Claude Fable 5.2 (`claude-fable-5-2[1m]`). 2026-11-10, 16:25-19:15Z. CPU only. Words follow `wf9/FRAME.md` section 5.

## 0. In plain words

The context first. On 10 Nov you said your consistency requirement is on the amplified predictions: amplified P(X)
times P(Y given X) should agree with P(Y) times P(X given Y) in the limit of compute, any variance over orders is a
signal, and the point is data efficiency. Only a consistency-over-compute loss had been tried. I tested your form on
the parity world's trained network: the learned aggregator (your trust head: it reads K sampled trajectories and
forecasts a check) extended to "any check given any subset of the others", trained with 64 to 20,480 real check
vectors, with and without a consistency term that sees no real outcome. No GPU was free, so compute is 1 and 4
trajectories per forecast (not 256) and two worlds.

1. **Your cross-order form did not buy data efficiency here.** As a loss on pairs or on whole orders it changed
   accuracy by between +15 % and -23 % with 64 or 256 real vectors, nothing that held in both worlds, and cost 6-30 %
   at a strong weight once data were plentiful.
2. **Anchored to the simple world model's score it helped a little, and only where data were fewest**, as you
   said: 6-17 % less error with 64 or 256 real vectors, nothing from 1,024. But training on outcomes sampled from M
   does about the same, with no real outcome at all it matches about 1,000 real vectors, and with that little
   real data every version is still worse than one pass of M.
3. **At this compute the variance over orders is sampling noise.** Each amplified number comes from its own sampled
   trajectories; two orders disagree about as much as two draws of one order. It did not fall from 1 to 4
   trajectories. M itself is nearly consistent, because it says little.
4. **The October failure reappears.** The loss removes the variance by making the forecast stop reading its
   trajectories: error on the informative questions goes back to one pass of M. Anchoring to the score does not
   prevent it; a small weight does.
5. **Where your idea does work (found afterwards, both worlds):** when real outcomes score only one kind of question
   (here: only questions with 25 or more checks revealed), the predictor is wild on all the others, and either
   cross-order term repairs that: error 0.071 / 0.283 becomes 0.032-0.040, against 0.029 with real outcomes
   everywhere. Consistency carries knowledge from large contexts down to small ones, never the other way.

I don't know: what happens at 64 or 256 trajectories, where the noise may fall below the real order effect; whether a
predictor with many free numbers per question differs; how to estimate the order variance without the sampling noise.

## 1. What was built

Paths are under `/home/agent/workspaces/mats/work/deep-study/wf13/consistency/`. Labels: GUARANTEE (by construction),
PROVEN, CHECKED (script and output named), CONJECTURE, NOT RUN. "Post hoc" = not in `PREREG.md`. Nothing here depends
on FRAME's Q4 or Q7; Q6 is taken as you answered it, (b).

- **World and M.** wf7's parity world (20 hidden assumptions, 30 noisy checks, each the parity of three assumptions
  flipped 5 % of the time) and wf7's trained size-M masked transformer (wf10's network), unchanged; worlds 101, 102.
- **The amplified predictor, any-order** (`cmodel.py`). A question is "check i given any subset S of the other
  checks". A searcher that sees only x_S proposes K fillings of the assumptions (wf10's helpful searcher). Along each,
  M gives its number for every check with all assumptions in context, and with the assumptions and x_S in context.
  A_K(i given x_S) is wf10's learned aggregator (a set transformer, 124k parameters) reading those K trajectories and
  one pass of M: the reality-trained mapping from the search's output to a forecast, your trust head. 22K + 1 network
  calls per context, the same for every arm.
- **Real data.** 20,480 real check vectors, assumptions dropped. n = 64, 256, 1,024, 4,096, 20,480; two disjoint
  subsets per n; 16, 4, 2, 1, 1 random contexts per vector, sizes uniform on 0-29. The last quarter of the n vectors
  picks the checkpoint. No arm sees a real outcome outside its n vectors.
- **Consistency terms see no real outcome and no real context**: their contexts are M's own samples (assumptions from
  M's empty-context numbers, then checks from M's numbers given them): 768 triples and 96 complete vectors.
  (i) pairs: A(X given C) A(Y given C, X) against A(Y given C) A(X given C, Y), four value pairs, squared log
  difference. (ii-o): variance over two sampled orders of the summed log step predictions for one complete vector.
  (ii-s): variance over sampled trajectories of that sum minus the log score of the terminal vector; the score of a
  check vector is M's own (sum over assumption fillings of prod M(assumption) prod M(check given assumptions)),
  estimated from 128 searcher trajectories with exact weights, no enumeration. (iii): wf10's over-compute form;
  (iii-sym): the same with the gradient flowing both ways, built to drag. Reference arm: outcomes sampled from M.
- **Test.** Fresh real vectors (2-5 % of the draws occurred in M's or the arms' data and were dropped, so none was
  seen): 512 contexts with every outside check as a target, 192 triples, 24 complete vectors in four orders. Exact
  posteriors by enumeration over 2^20 in `cexact.py` only, on another machine after the arms ran.
- **Compute.** Equal at test time. In training the consistency arms also read unlabelled trajectories: 0.34 million
  network calls for (i), (iii), M's samples; 0.78 million for (ii-o), (ii-s); labelled contexts: 0.09 to 2.2 million.

**Not as the brief asked.** (1) No L40: the queue estimate was 2 h 40 min, so the work ran as one CPU-only Slurm job
at a time (10 cores, about 200 network calls per second per core), hence **K = 1 and 4 only** and two worlds. "The
limit of more compute" is one step here. (2) I first registered the cheaper size-S networks; a label-only check showed
their searcher contradicts 5 of the revealed checks, leaving nothing to amplify, so I amended the registration to
size M before any arm ran on a test world (`PREREG.md` section 9). (3) An accident: the first CPU-only job (57885) ran
for about a minute on a GPU it had not requested (Slurm there does not hide GPUs from jobs that request none). I
cancelled it and set its outputs aside; every later task hides the GPUs and prints the check.

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

| arm, weight | no real outcome | n = 64 | n = 256 | n = 1,024 | n = 4,096 | n = 20,480 |
|---|---|---|---|---|---|---|
| real outcomes only | - | 0.045 / 0.047 | 0.043 / 0.040 | 0.033 / 0.035 | 0.032 / 0.034 | 0.031 / 0.032 |
| (i) pairs, weight 1 | 0.046 / 0.063 | 0.043 / 0.050 | 0.044 / 0.039 | 0.034 / 0.037 | 0.032 / 0.034 | 0.032 / 0.032 |
| (i) pairs, weight 10 | - / - | 0.046 / 0.056 | 0.039 / 0.045 | 0.037 / 0.044 | 0.037 / 0.041 | 0.036 / 0.041 |
| (ii-o) other order, 1 | 0.046 / 0.063 | 0.039 / 0.048 | 0.041 / 0.038 | 0.033 / 0.034 | 0.032 / 0.034 | 0.031 / 0.032 |
| (ii-o) other order, 10 | - / - | 0.046 / 0.057 | 0.037 / 0.044 | 0.036 / 0.041 | 0.035 / 0.040 | 0.035 / 0.040 |
| (ii-s) score, 0.1 | - / - | 0.043 / 0.045 | 0.040 / 0.039 | 0.033 / 0.035 | 0.031 / 0.033 | 0.031 / 0.031 |
| (ii-s) score, 1 | 0.048 / 0.055 | 0.038 / 0.044 | 0.036 / 0.038 | 0.034 / 0.037 | 0.035 / 0.037 | 0.033 / 0.036 |
| (ii-s) score, 10 | - / - | 0.040 / 0.046 | 0.045 / 0.047 | 0.040 / 0.050 | 0.039 / 0.043 | 0.041 / 0.049 |
| (iii) over compute, 1 | 0.071 / 3.777 | 0.044 / 0.042 | 0.046 / 0.039 | 0.033 / 0.036 | 0.031 / 0.033 | 0.031 / 0.031 |
| outcomes sampled from M, 1 | 0.035 / 0.034 | 0.039 / 0.041 | 0.035 / 0.037 | 0.033 / 0.033 | 0.031 / 0.032 | 0.031 / 0.031 |
| one-pass net trained on the n real vectors | - | 0.056 / 0.080 | 0.052 / 0.065 | 0.045 / 0.061 | 0.045 / 0.061 | 0.045 / 0.061 |

**Table B. Gain = 1 - excess(arm) / excess(real outcomes only).**

| arm, weight | n = 64 | n = 256 | n = 1,024 | n = 4,096 | n = 20,480 |
|---|---|---|---|---|---|
| (i) pairs, weight 1 | +5% / -7%* | -3% / +4% | -0% / -7% | -1%* / -1%* | -2%* / -2%* |
| (i) pairs, weight 10 | -3% / -20%* | +8% / -13% | -11%* / -27%* | -16%* / -22%* | -15%* / -30%* |
| (ii-o) other order, 1 | +15% / -2% | +4% / +6% | +2% / +3% | -2%* / -1%* | +0% / -0% |
| (ii-o) other order, 10 | -2% / -23%* | +13% / -9% | -6% / -18%* | -10%* / -18%* | -11%* / -26%* |
| (ii-s) score, 0.1 | +5%* / +2% | +6% / +3% | +1% / +1% | +3% / +2% | +1%* / +2%* |
| (ii-s) score, 1 | +17%* / +6% | +15%* / +6%* | -1% / -7% | -9% / -9% | -5% / -15%* |
| (ii-s) score, 10 | +12%* / +1% | -6% / -17% | -21%* / -42%* | -22%* / -28%* | -30%* / -56%* |
| (iii) over compute, 1 | +2% / +9%* | -8% / +3%* | +2% / -3% | +3% / +2% | +1% / +1% |
| outcomes sampled from M, 1 | +15%* / +12%* | +18%* / +7% | +1% / +7%* | +2% / +4% | +1% / +2% |

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

| predictor | pair mismatch D, K = 1 | D, K = 4 | V_ord, K = 1 | V_ord, K = 4 |
|---|---|---|---|---|
| one pass of M (M itself) | 0.005 / 0.007 (0.000 / 0.000) | 0.005 / 0.007 (0.000 / 0.000) | 0.55 / 0.54 (0.00 / 0.00) | 0.55 / 0.54 (0.00 / 0.00) |
| the score's exact weights on the K trajectories | 9.029 / 9.567 (8.623 / 9.321) | 2.450 / 2.266 (2.406 / 2.312) | 66.14 / 53.88 (55.57 / 42.77) | 16.57 / 16.78 (17.41 / 15.73) |
| real outcomes only, n = 64 | 0.148 / 0.057 (0.130 / 0.051) | 0.145 / 0.095 (0.146 / 0.085) | 2.86 / 0.92 (1.66 / 0.37) | 2.67 / 1.29 (1.24 / 0.53) |
| real outcomes only, n = 1,024 | 0.065 / 0.060 (0.047 / 0.065) | 0.079 / 0.088 (0.072 / 0.063) | 0.97 / 0.98 (0.41 / 0.45) | 0.93 / 1.07 (0.42 / 0.46) |
| real outcomes only, n = 20,480 | 0.043 / 0.063 (0.035 / 0.071) | 0.058 / 0.111 (0.061 / 0.084) | 0.87 / 0.75 (0.35 / 0.37) | 0.74 / 0.89 (0.37 / 0.53) |
| (i) pairs, weight 10, n = 64 | 0.005 / 0.003 (0.000 / 0.000) | 0.004 / 0.003 (0.000 / 0.000) | 0.34 / 0.22 (0.01 / 0.00) | 0.31 / 0.21 (0.01 / 0.00) |
| (i) pairs, weight 10, n = 20,480 | 0.002 / 0.003 (0.000 / 0.000) | 0.002 / 0.003 (0.000 / 0.000) | 0.12 / 0.20 (0.00 / 0.00) | 0.12 / 0.19 (0.00 / 0.00) |
| (ii-o) other order, 10, n = 20,480 | 0.006 / 0.009 (0.003 / 0.007) | 0.011 / 0.018 (0.009 / 0.014) | 0.17 / 0.29 (0.03 / 0.06) | 0.22 / 0.45 (0.05 / 0.10) |
| (ii-s) score, 1, n = 20,480 | 0.035 / 0.073 (0.024 / 0.089) | 0.045 / 0.150 (0.045 / 0.095) | 0.47 / 0.92 (0.26 / 0.41) | 0.60 / 1.07 (0.22 / 0.52) |

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

| arm, weight | n = 64 | n = 1,024 | n = 20,480 |
|---|---|---|---|
| (i) pairs, weight 1 | -0.024 / +0.032* | +0.009 / +0.038* | +0.018* / +0.026* |
| (i) pairs, weight 10 | +0.012 / +0.079* | +0.043* / +0.090* | +0.058* / +0.098* |
| (ii-o) other order, 1 | -0.041* / +0.017 | -0.005 / +0.008 | +0.009* / +0.013* |
| (ii-o) other order, 10 | -0.006 / +0.052* | +0.029* / +0.047* | +0.042* / +0.065* |
| (ii-s) score, 0.1 | -0.051* / -0.013 | -0.008 / -0.008* | +0.002 / +0.006 |
| (ii-s) score, 1 | -0.059* / -0.019 | +0.006 / +0.010 | +0.019* / +0.014* |
| (ii-s) score, 10 | -0.042* / +0.019 | +0.045* / +0.056* | +0.050* / +0.079* |
| (iii) over compute, 1 | -0.030 / -0.003 | -0.011 / +0.005 | +0.000 / +0.003 |
| (iii-sym), 10 | -0.015 / +0.049* | +0.035* / +0.056* | +0.045* / +0.078* |
| outcomes sampled from M, 1 | -0.034* / -0.030* | -0.000 / -0.008 | +0.003 / -0.003 |

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

| arm, weight | n = 1,024: all queries | 0-9 revealed | 10-19 | 20-29 | n = 20,480: all queries | 0-9 | 10-19 | 20-29 |
|---|---|---|---|---|---|---|---|---|
| real outcomes only (restricted) | 0.193 / 0.329 | 0.192 / 0.391 | 0.192 / 0.285 | 0.199 / 0.204 | 0.071 / 0.283 | 0.036 / 0.312 | 0.100 / 0.284 | 0.154 / 0.163 |
| + (i) pairs, 10 | 0.038 / 0.054 | 0.005 / 0.009 | 0.043 / 0.063 | 0.187 / 0.211 | 0.037 / 0.036 | 0.003 / 0.004 | 0.047 / 0.041 | 0.169 / 0.149 |
| + (ii-o), 10 | 0.051 / 0.048 | 0.021 / 0.009 | 0.057 / 0.056 | 0.181 / 0.185 | 0.032 / 0.040 | 0.004 / 0.007 | 0.040 / 0.047 | 0.147 / 0.148 |
| + (ii-s) score, 1 | 0.040 / 0.041 | 0.007 / 0.007 | 0.049 / 0.047 | 0.170 / 0.156 | 0.031 / 0.045 | 0.003 / 0.009 | 0.039 / 0.060 | 0.143 / 0.146 |
| + (iii) over compute, 1 | - / - | - / - | - / - | - / - | 0.228 / 0.504 | 0.305 / 0.725 | 0.131 / 0.314 | 0.145 / 0.146 |
| + outcomes sampled from M | 0.038 / 0.045 | 0.008 / 0.014 | 0.045 / 0.050 | 0.166 / 0.154 | 0.031 / 0.030 | 0.004 / 0.003 | 0.039 / 0.033 | 0.140 / 0.129 |
| real outcomes on every query (section 2a) | 0.031 / 0.031 | 0.003 / 0.003 | 0.038 / 0.032 | 0.145 / 0.139 | 0.029 / 0.029 | 0.002 / 0.002 | 0.035 / 0.031 | 0.140 / 0.129 |
| one pass of M | 0.034 / 0.036 | 0.001 / 0.001 | 0.037 / 0.035 | 0.187 / 0.182 | 0.034 / 0.036 | 0.001 / 0.001 | 0.037 / 0.035 | 0.187 / 0.182 |

- With real outcomes on one family only, the predictor is wild elsewhere. **Either cross-order term repairs it**:
  0.032-0.040 at n = 20,480, against 0.029 with real outcomes on every question. On the questions real data never
  scored the result is about one pass of M (10-19 revealed: 0.040-0.047 against 0.037 / 0.035), not the amplified
  level: the terms carry consistency down, not the amplification.
- (iii), over compute, cannot do this (it compares a question with itself). M's samples can (0.031 / 0.030).

### (f) Post hoc: a pair term with the sampling noise removed (failed)
Product of the mismatches from two disjoint trajectory draws; its expectation is the squared expected mismatch
(`ctrain_ph.py` arm `idb`, world 101, `out/w101_phdb/`). Worse than real-outcomes-only at every n and weight (excess
+0.012 to +0.031); at weight 100 it collapsed to a constant. The product is not bounded below and is very noisy.

## 3. Why, and what kind of world would make it help

- **What an order-against-order condition can carry (PROVEN, elementary).** For checks X, Y and context C, the two
  larger-context numbers P(Y given C, X) and P(X given C, Y) fix the joint of (X, Y) given C, hence both
  smaller-context numbers: P(X=1 given C) / P(X=0 given C) = [P(X=1 given C, y) / P(X=0 given C, y)] x [P(y given C,
  X=0) / P(y given C, X=1)]. The converse fails: the two smaller-context numbers leave the dependence free. So the
  condition carries knowledge **from larger contexts down to smaller ones**, and it adds to real data only where
  real outcomes score the larger-context questions and not the smaller-context ones.
- **In this world real data need no such transport.** One real check vector scores every question about itself, of
  every context size. The condition can then only tie together estimates the predictor would otherwise fit
  separately, and a small head with the same weights for every question fits nothing separately. Hence no gain in
  2b, and a large one in 2e, where real outcomes score one family only.
- **At small compute the measured mismatch is sampling noise (2c), and the cheapest way to remove sampling noise is
  to stop reading the samples (2d).** That is the October failure on the amplified predictor: the amplification is
  informative but noisy, one pass of M is stable but says little, and the loss drags the first toward the second.
- **Why the anchor helps and where it stops.** (ii-s) is (ii-o) plus the variance between terminal vectors, which is
  zero only if the amplified joint is proportional to the score. That part is real information (the score's own law,
  the same thing as outcomes sampled from M) and needs no real outcome, so it helps where real outcomes are few. The
  (ii-o) part is the same noise penalty, so a large weight drags too; and it can teach only what the score knows.
- **Worlds where your claim should hold** (item 1 CHECKED post hoc in two worlds; the rest CONJECTURE). (1) Reality
  scores only some kinds of question, for instance only the last step with everything else revealed, and forecasts
  are needed for the others. (2) The amplified predictor has many free numbers per question (your distilled
  one-pass head, params_amp), not a 124k-parameter head on M's numbers. (3) Compute is large enough that sampling
  noise is below the order effect. (4) The variables asked about are strongly dependent given typical contexts, so
  the trivially consistent answers (one half; one pass of an M that says little) are far from right.
- **How I would word the requirement.** As a condition on the EXPECTED amplified prediction: the mean over trajectory
  draws of amplified P(X) P(Y given X) equals that of P(Y) P(X given Y). A loss on single draws also penalises the
  sampling noise, and I do not have a working estimator without it (2f).

## 4. Registered predictions (`PREREG.md` sections 6 and 9; all CHECKED; sections 2e, 2f and the mechanism are post hoc)

| # | Registered item | Outcome | My prediction |
|---|---|---|---|
| H1 (i) | pairs term improves accuracy at n = 64, 256 | no: +5 % / -7 %*, -3 % / +4 % at the validation-chosen weight | NULL or MINOR: right |
| H1 (ii-o) | order-variance term, same | no gain in both worlds: +15 % / -2 %, +4 % / +6 % | NULL or MINOR: right |
| H1 (ii-s) | score-anchored term, same | MINOR: +17 %* / +6 %, +15 %* / +3 %; largest at fewest data in direction | MAJOR: wrong |
| H1 (iii) | over-compute control | NULL: +2 % / +9 %*, -8 % / +3 %* | NULL: right |
| H1 n = 0 | the term alone, no real outcome | (ii-s) 0.048 / 0.055 against 0.031 / 0.032 for 20,480 real vectors; (i), (ii-o) = the constant; (iii) diverges | (ii-s) within 0.02: right in world 101, wrong in 102 (0.023); the others: right |
| H2 | D and V_ord fall from K = 1 to K = 4 (n = 20,480) | no: D rises 0.043 / 0.063 to 0.058 / 0.111; V_ord 0.87 / 0.75 to 0.74 / 0.89 | falls: wrong; "mostly noise floor": right |
| H3 | (i) cuts D, (ii-o) cuts V_ord by 30 % or more at every n | yes: D to 2-4 % of its value; V_ord to 12-55 % | yes: right |
| H4 | M's D below the amplified predictor's at K = 1, above it at the top K | below at both K (0.005 / 0.007 against 0.04-0.19) | half wrong |
| H5 | drag at n = 64 and 1,024 in both worlds for (i) or (ii-o), weight 10 | 3 of 4 cells (not world 101 at n = 64); every cell at n >= 1,024 | "under half of the cells": wrong |
| H5 anchor, built | (ii-s) shows no drag in those cells; (iii-sym) 5 % worse than (iii) | holds at weights 0.1, 1, fails at 10; (iii-sym) 14-24 % worse from n = 1,024 | first wrong at weight 10; second right |
| wrong score | (ii-s), M's samples worse than data-only on checks 12, 25 at n >= 4,096 | by 0.000-0.007 at weights 1, 10 and for M's samples; not at 0.1 | right in direction, small |

## 5. Limits, what was not run, files

- **Compute is the main limit.** K = 1 and 4 only. At K = 4 the amplified predictor removes a quarter of one pass's
  error on large contexts and almost nothing elsewhere; wf10's K = 256 predictor is a different object. Whether the
  order variance falls below the sampling noise at K = 64 or 256 is UNKNOWN and needs the L40.
- Two worlds, one network each, two real subsets per n, one initialisation per arm. A second initialisation moves
  real-outcomes-only by up to 0.004 nats at n = 64, so gains under about 10 % at n <= 256 are inside run noise.
- Uniform context sizes make most questions easy; the mean is moved mostly by the 11-13 % with 20-29 revealed checks.
  The weights were chosen on a dev world where the amplifier did not work and kept as registered: scale guesses. The
  score reference of (ii-s) is a 128-trajectory estimate; a better one might change its high-weight result.
- Sections 2e, 2f and the mechanism numbers were designed after seeing world 101's registered results; 2e was then
  repeated on world 102 unchanged. Changed after the first registration, evaluation side only: `cexact.py`, `cU.py`
  (cached table), `creport.py` (bad-check rule, score-estimate numbers), `mktables.py`, `cdirect.py` (800 steps, on
  this host; on the cluster it blocked the queue and job 57889 was cancelled for that after its 114 runs finished).
- NOT RUN: worlds 103, 311; K >= 16; consistency contexts from real check vectors; another searcher; decisions.
- **Jobs** (cluster account k.santos, CPU-only, 10 cores): 57881 benchmark; 57885 (the GPU accident, cancelled);
  57886 dev world; 57889 world 101; 57904 world 102. L40 jobs 57879, 57880: cancelled in the queue, never ran.
- **Files.** `PREREG.md`, `PREREG.sha256` (17:26:32Z; amendment 17:30:10Z: its text says 17:35Z, and a cut pipe left
  only `PREREG.md` and `cgen.py` in its hash entry; the other arm files are unchanged from the first entry and the
  cluster task logs print their hashes). Arms, no exact inference (`check_firewall.py`, `.out`): `cgen.py`,
  `cmodel.py`, `ctrain.py`, `mkjobs.py`, `cdirect.py`; post hoc `ctrain_ph.py`, `cgen_c1b.py`. Evaluation: `cexact.py`,
  `cU.py`, `creport.py`, `mktables.py`, `mknote_tables.py`, `mkfigs.py`, `cmech.py`. `cluster/` (tasks as run).
  Outputs: `out/w101/`, `out/w102/` (every run's forecasts, logs, `summary.json`), `out/tables.md`, `out/w10*_ph25/`,
  `out/w101_phdb/`, `out/mechanism_w10*.json`, `out/pilot_w7/` (dev), `logs/`, `fig/`, `RESULT-work.json`.
  Trajectories: `data/w101/`, `data/w102/` (29 MB each; 4.2 million network calls per world).
