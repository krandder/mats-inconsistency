# PREREG: Kelvin's consistency requirement on the amplified predictions, as a data-efficiency device (wf13/consistency)

Model: Claude Fable 5.2 (`claude-fable-5-2[1m]`). Written 2026-11-10 before any run on a test world. Hash and time are in
`PREREG.sha256`. Frame: `wf9/FRAME.md` section 5 (the simple world model scores the terminal state; the GFlowNet is the
rollout policy; the search is the amplification on top of many rollouts; a trust head trained on real data only maps
the amplification's output to reality; the consistency requirement is on the amplified predictions).

## 1. His statement under test
"When we ask for 'amplified predictions' of P(X), then 'amplified predictions' of P(X|Y), that gives us some amplified
prediction of P(X,Y), and we expect this to get consistent to asking amplified predictions the other way aroudn P(Y)
first then P(X|Y), in the limit of more compute. ... any variance is a signal that this amplification procedure can be
improved; of course we still want this amplified any-order prediction to also be accurate in real-data, but the
consistency may help us deal with data efficiency. But actually estimating the variance over permutations is hard; in
the Bengio perpective this would be a variance over trajectories I think." "The consistency loss idea on the amplified
predictions is meant as a data-efficiency thing."

## 2. Deviation from the brief, decided before any test run: CPU scale
The L40 queue estimate for this account at 16:50Z was 19:30Z (2 h 40 min; jobs 57879 and 57880 cancelled unstarted), so
by the brief's rule the work falls back to CPU: one CPU-only Slurm job on the same cluster account (57885; 10 cores,
no GPU; the benchmark job 57881 measured 1,726 calls/s per core for wf7's size-S network and 379 for size M). At that
speed the planned GPU design (size-M networks, K up to 64, 140 million network calls per world) is out of reach.
Reduced design, fixed here:
- M = wf7's size-S masked transformers of worlds 101 and 102 (`M_w101_S.pt`, `M_w102_S.pt`; trained by wf7 on 50,000
  real samples that show the assumptions). World 103 is added only if time allows, with the same code. wf10's
  aggregator results were on the size-M networks; nothing here is compared with those numbers.
- Compute levels K in {1, 4, 16} (wf10: up to 256). Searcher: wf10's helpful searcher H, unchanged.
- Dev world 7 (size-S network) is used for the pipeline check and for choosing the two weights per form; it is never
  reported as a result.

## 3. World, amplified predictor, data
- Parity world of `wf7/probworld/world.py`: 20 hidden assumptions, 30 noisy 3-parity checks.
- **Any-order query**: the probability of check i given any subset S of the other checks (with values). Context sizes
  are uniform on 0..29, subsets uniformly random.
- **Amplified predictor** A_K(i | x_S) (`cmodel.py`): wf10's free-form learned aggregator with compact tokens, extended
  to any context and target. For a context, the searcher H (sees x_S only) proposes K fillings of the assumptions
  (20 network calls each); along each, M gives its number for every check with all assumptions in context and its
  number with the assumptions and x_S in context (2 calls). The aggregator reads the K trajectories and one pass of M.
  22K + 1 network calls per context, the same for every arm.
- **Real labelled data** (`lab`): 20,480 real check vectors (assumptions dropped, seed tag 31, disjoint from wf7 and
  wf10). Sweep n in {64, 256, 1,024, 4,096, 20,480}. Two disjoint subsets d = 0, 1 per n (for 20,480: the same data,
  another initialisation). Contexts per vector: 16, 4, 2, 1, 1. Inside a subset the last quarter of the vectors (at
  most 1,024) is the validation part; it picks the checkpoint of every arm by log loss on real outcomes. No arm uses a
  real outcome outside its n vectors, for training or for selection.
- **Unlabelled contexts for every consistency term: M's own samples only** (h from M's empty-context numbers, x from
  M(x | h); sets `c1`, `c2`). No real check vector enters a consistency term.
- **Test** (fresh real vectors, seed tags 34-36; vectors seen in M's training set or in `lab` are dropped): `t1` 512
  contexts with every outside check as a target; `t2` 256 triples (C, X, Y) with the five contexts C, C+X=0, C+X=1,
  C+Y=0, C+Y=1; `t3` 32 complete vectors, four random orders of revealing all 30 checks each.
- Exact references by enumeration over 2^20, in `cexact.py` only (evaluation): the true posterior for every test
  context, and the same under the score T (M's empty-context numbers times M(x_c | h)).

## 4. Arms (`ctrain.py`)
Loss = log loss on the real outcomes of the n vectors + weight x consistency term. Forms:
- **(i) order against order on pairs**: J1 = A(X|C) A(Y|C,X), J2 = A(Y|C) A(X|C,Y), all four value pairs; loss = mean
  over sampled triples of sum_{x,y} w (log J1 - log J2)^2, w = stop-gradient (J1 + J2)/2.
- **(ii-o) variance over trajectories, reference = another order**: for a sampled terminal vector, the summed log step
  predictions along two sampled orders of revealing the 30 checks; loss = their sample variance / 30.
- **(ii-s) variance over trajectories, reference = the simple world model's score**: variance over the batch's
  trajectories (terminal vector, order) of [summed log step predictions - log R-hat(x)] / 30; log R-hat(x) = the
  score of the terminal vector, summed over assumption fillings by 256 searcher trajectories with exact weights
  R(h, x)/psi(h), R(h, x) = prod M(h_i | nothing) x prod M(x_c | h). No enumeration, no hand-picked parents.
  (`cgen.py` computes it with 64; `cref.py` replaces it with 256 before any arm runs. Its error against the exact
  log T(x) is reported for both.)
- **(iii) over compute** (wf10's form, the control): the K forecast predicts the stop-gradient 4K forecast.
- **(iii-sym)**: the same without the stop-gradient, at one weight: the constructed drag test (section 6, H5).
- **synth** (reference, not a consistency term): log loss on outcomes sampled from M itself, weight 1.
- Weights: two per form, chosen on the dev world and written in section 8 before the test runs.
- n = 0 runs (the term alone, no real outcome, last checkpoint): each form (two initialisations for ii-s and synth).
- Runs: the list printed by `mkjobs.py` (114 lines per world), in that order so that the first block (subset d = 0,
  every arm, every n) finishes first. Subset d = 1: every arm for n <= 1,024; data-only, i@10, iio@10, iis@1 and synth
  for n >= 4,096. Data-only also runs with a second initialisation (d = 0, r = 1). If the job runs out of time the
  missing runs are named in the note. 700 steps, AdamW, one model conditioned on K. Same initialisation and same data
  batches across arms at equal (n, d, r). Batches: 128 labelled queries; 48 triples; 6 terminal vectors x 2 orders.
- References on the same test sets: one pass of M (sizes S; M and L where a checkpoint exists); the score with exact
  weights on the same K trajectories (`exu`); plain average (`meanu`); a one-pass any-order network trained only on
  the same n real vectors (`cdirect.py`, the memorising baseline).

## 5. Metrics (`creport.py`)
- **excess**: mean over t1 queries of KL(exact posterior || forecast), nats, averaged over K in {1, 4, 16}. Primary.
- **real**: log loss on the held-out real outcome of the same queries.
- **pair mismatch D(K)** on t2: mean of sum_{x,y} w (log J1 - log J2)^2; **noise floor**: the same between two
  independent trajectory draws of one order (K = 1, 4).
- **V_ord(K)** on t3: variance over four orders of the summed log step predictions for one terminal vector; noise
  floor as above; **var_score**: the (ii-s) quantity on test.
- **gain** G(arm, n) = 1 - excess(arm) / excess(data-only), means over the matched runs.
- Intervals: paired bootstrap over test contexts.

## 6. Hypotheses, bins, my predictions
- **H1 (his claim: data efficiency).** Per form, at its weight with the lower validation loss. MAJOR: G >= 0.20 at
  n = 64 and at n = 256 in both worlds, intervals excluding 0. MINOR: G >= 0.05 at one of them in both worlds.
  NULL: |G| < 0.05. NEGATIVE: G <= -0.05 in both worlds. "Largest when data are fewest": mean G at n <= 256 exceeds
  mean G at n >= 4,096 by 0.10 or more.
  Predictions: (ii-s) MAJOR and largest at small n (it distils the score; `synth` does the same). (i) and (ii-o): NULL
  or MINOR at n = 64 only (a regulariser with no anchor), NULL from 1,024. (iii): NULL. Alone (n = 0): (ii-s) within
  0.02 nats of the 20,480-vector data-only arm; (i), (ii-o), (iii) near the constant one half (excess above one pass).
- **H2 (variance falls with compute).** For the data-only arm at n = 20,480: D(K) and V_ord(K) fall from K = 1 to 16.
  Prediction: yes, and most of D at K <= 4 is the noise floor.
- **H3 (the loss lowers the variance on held-out real vectors).** (i) lowers D(16) and (ii-o) lowers V_ord(16) by
  30 % or more against data-only at every n, at the larger weight. Prediction: yes.
- **H4 (M itself).** One pass of M has D and V_ord reported as the baseline. Prediction: M's D is below the amplified
  predictor's at K = 1 (M has no sampling noise) and above it at K = 16.
- **H5 (October failure: the reliable direction dragged toward the unreliable one).** Per t2 triple at K = 4 and 16,
  the order whose joint is closer (KL) to the exact joint under the data-only run is "better". Drag = change of the
  better order's KL in the arm minus the same change in a second data-only run with another initialisation (placebo,
  removes regression to the mean). REAPPEARS: drag > 0 with interval excluding 0, while D falls, for (i) or (ii-o) at
  the larger weight, at n = 64 and n = 1,024, in both worlds. PREVENTED BY THE ANCHOR: for (ii-s) the same quantity is
  <= 0 or its interval contains 0 in those cells. Constructed version: (iii-sym) against (iii): excess at K = 16
  worse by 5 % or more = drag by construction.
  Predictions: (iii-sym) shows it; (i) shows it weakly (in under half of the cells); (ii-s) does not.
- **Where the score is wrong.** If the score T's own excess exceeds 0.05 nats on some checks of a world (computed by
  `creport.py` from `exact.npz` alone), those checks are listed and the excess is split. Prediction: there (ii-s)
  and synth are worse than data-only at n >= 4,096 at the larger weight.

## 7. Guarantees (true by construction, not findings)
The exact posterior satisfies (i), (ii-o) with zero mismatch. The constant forecast one half satisfies (i), (ii-o),
(iii) exactly. (ii-s) at zero loss means the amplified joint equals the estimated score R-hat on the sampled vectors.
Test-time compute is equal across arms. Consistency arms use extra unlabelled compute at training time (network calls
of sets c1 and c2, reported); the data-only arm has nothing to spend it on.

## 8. Weights chosen on the dev world (filled in before the hash)
Dev run: world 7, size-S network, scale `cpupilot` (job 57886, task t01; outputs `out/pilot_w7/`). What it showed:
(a) weights 1 and 10 on (i) and (ii-o) cut the held-out mismatch by 2 to 100 times at n = 64 and 1,024 with excess
within 0.005 of data-only either way; 100 added nothing. (b) (ii-s) at weights 10 and 100 made validation loss on real
outcomes worse than the untrained start (the checkpoint rule then returns the start); at weight 1 it was neutral. In
that dev world the score estimate is poor (the same exact weights used as a forecaster score 0.35 against 0.04 for one
pass), so the dev world says little about (ii-s) where the score is good. (c) Validation minima were at steps 100 to
600 of 1,000 for the unregularised arms, hence 700 steps. (d) Run time 100 to 235 s per run on one core.
Chosen: (i) 1 and 10; (ii-o) 1 and 10; (ii-s) 0.1, 1 and 10; (iii) 1 (wf10's weight); (iii-sym) 10; synth 1.
The dev world also showed that with uniform context sizes most queries are near one half (one pass of M 0.039 nats
from exact); the excess by context size (0-9, 10-19, 20-29 revealed checks) is therefore reported beside the mean.
One accident, disclosed: the first CPU-only job (57885) ran for about one minute with a GPU visible, because Slurm on
this node does not hide GPUs from jobs that request none. It was cancelled at 17:02:47Z, its outputs set aside
(`*_job57885_gpu_accident`), and every later task sets CUDA_VISIBLE_DEVICES to empty and prints the check.
Order of events: data generation for world 101 (task t110: trajectories only, no arm, no outcome read) started at
17:24Z, before this hash; no arm was trained or evaluated on a test world before it.

## 9. Amendment, 17:35Z, before any arm was trained or evaluated on a test world
What happened. Generation for world 101 with the size-S network had run for five minutes (job 57886, task t110) when
I ran a label-only diagnostic on its test contexts (`dev/diag_t1.py`: real log loss of fixed predictors, no trained
arm, no exact reference): with the size-S network the searcher's trajectories contradict 5.3 of the revealed checks on
average at 20-29 revealed checks, and the plain average over 16 trajectories scores 0.663 against 0.672 for one pass.
The same in the dev world (7.0 contradicted; the trained aggregator there equals the constant one half). With the
size-S network there is no amplified predictor to test. Job 57886 was cancelled; its partial data are set aside
(`data/w101_S_abandoned`) and not used.
Amended design (replaces section 2's network and compute levels; everything else stands):
- M = wf7's **size-M** networks of worlds 101 and 102 (`M_w101_M.pt`, `M_w102_M.pt`), the networks of wf10's
  aggregator. On CPU they cost 4.5 times more per call, so:
- **Compute levels K in {1, 4}** (scale `cpuM` in `cgen.py`: 4 trajectories per training context, 8 per test context so
  that K = 4 has a noise floor from two disjoint draws). "Falls with compute" (H2) is K = 1 against K = 4. H3 and H5
  are read at K = 4 (and K = 1).
- Sets: `lab` as before; `c1` 768 triples; `c2` 96 terminal vectors; score reference from 128 trajectories (not 256,
  no `cref.py` step); `t1` 512 contexts; `t2` 192 triples; `t3` 24 terminal vectors x 4 orders.
- Exact references are computed on the evaluation side on another machine (`cU.py`, `cexact.py`, `creport.py`).
- One-pass references: M sizes S, M, L.
- The weights of section 8 were chosen on a dev world where the amplifier did not work, so they are scale guesses.
  They stay as registered: (i) 1, 10; (ii-o) 1, 10; (ii-s) 0.1, 1, 10; (iii) 1; (iii-sym) 10; synth 1.
- Checkpoint rule, 700 steps, job list (`mkjobs.py`), metrics, hypotheses and bins: unchanged, with "K = 16" read as
  "K = 4" and the mean over K in {1, 4}.
