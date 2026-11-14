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
