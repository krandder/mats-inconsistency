# E-SUD results: any-order Sudoku, search chains, and "searched and found nothing" as evidence

Run 2026-10-15 00:54-04:15Z by calibrated-ai for opus-mats (authorised by Kelvin 00:3xZ). Design: base design + v2
(gemini's review adopted in full). Pre-registration: `PREREG.md` (frozen before any 6x6 model existed; sha256 in
`PREREG.sha256`). A post hoc addendum (`PREREG-ADDENDUM-1b.md`) was written after round-1 seed 0 and before any
round-1b number.

## Verdict

- **Pre-registered (round 1, 30k training steps): D4 FAILS.** Two of the three seeds never learned the local Sudoku
  law. Their own Solvable check then accepted 39-83% of invalid completions as witnesses (depending on seed and rule), so unsolvable boards got
  log loss of about 2.5 against 0.39 for one pass. The seed that did learn the law passes every part of D4 by a wide
  margin.
- **Post hoc (round 1b, 150k steps, otherwise identical): all 3 seeds learn the law.**
  - D4 passes for P1 (random order) and P3 (confidence-first): unsolvable-board log loss is cut by 98-100% against
    one pass and against witness-only, and solvable boards by 99.99%.
  - D4 fails for the declared primary, P2 (stiff-gated VoI), on part (a)-solvable only. In seed 1, one of 857 solvable
    boards found no witness in 8 chains, and the null update (theta = 79) put it at p = 1e-6, a 13.8-nat loss. The
    seed's cut is therefore 31% and the seed-t CI crosses 0.
- **D0 (round 1b): "small gap".** KL(exact || P0) is 0.043-0.048 nats, just under the 0.05 bar; one-step cells are
  at 0.000 (the bar is <= 0.02). Round 1 had a large gap but failed the one-step half in 2 of 3 seeds.
- **Cell marginals improve with search once the law is learned** (round 1b, M = 16 chains vs P0, 3 seeds):
  confidence-first -46% KL [41.7, 50.9], random order -45% [42.3, 46.8], stiff-gated VoI -41% [35.8, 45.7]. The v2
  D2 bar of 30% is reached at M = 16 by all three and at M = 8 by P3 only.
- **VoI did not beat the cheap rules.** At M = 16 its KL is 7% higher than random order's (CI [-19.5, +5.8]), and confidence-first's KL is 9.4%
  lower than VoI's [8.3, 10.4], while costing about 37 forward passes per step instead of 1.
- **Mass vs count.** The count-only control (logit p0 - kappa*M) does as well as the proposal-mass update in every
  setting where either works. Once the network has learned the law, any failed search is near-conclusive here, so
  this test cannot tell refuted MASS from refuted COUNT.

Status label for /experiments: **Informative** (pre-registered rule failed; the mechanism works once the network
learns the local law, post hoc).

## Setup (as run)

- **Boards (6x6, 2x3 boxes).** 10-20 givens taken from a random grid.
  - Solvable boards have 2-400 completions; on test the mean is 12.1.
  - Unsolvable boards come from the same generator with 1-2 givens changed: no direct peer clash, zero completions,
    clue count matched.
  - Mean clue count on test: 13.6 on solvable boards and 14.8 on unsolvable ones (see Deviations).
  - Splits: train 200,000 boards, validation 2,000, test 2,000 (42.8% solvable).
  - Exact marginals come from a bitmask DFS over all completions. They were cross-checked against an independent
    row-permutation enumerator on 300 test boards (max |diff| 3e-8) and, for 4x4, against the full 288-grid table
    (`results/check_exact.json`).
- **Model.** One pre-LN transformer: d 128, 4 layers, 4 heads, 37 tokens (36 cells + a homogeneous Solvable token),
  about 0.8M parameters. It has learned position ids and no knowledge of rows, columns, boxes or the rule.
  - Any-order masked loss on outcomes only: one drawn completion per solvable board, plus the solvable bit.
  - Reveal count k ~ Uniform{0..#empty}.
  - AdamW, lr 1e-3, cosine schedule, batch 512.
  - The checkpoint is chosen by validation outcome loss.
- **Arms** (frozen network; M chains per board, nested prefixes of 16).
  - Chains run under Solvable = 1 and grow S_t = S_{t-1} + {z_t} by sampling z_t from q(z_t | S_{t-1}) until the
    board is full. Three rules pick the next cell:
    - P1: random order.
    - P2: stiff-gated VoI (top-6 lowest-entropy cells, then expected squared change of the other marginals).
    - P3: confidence-first (lowest entropy).
  - Cell estimate: (lambda p0 + sum_r w_r rb_r) / (lambda + sum_r w_r), where rb_r is the conditional each cell was
    sampled from and w_r is the model's own q(Solvable = 1 | completed board). lambda is fitted on validation.
  - Solvable query:
    - A witness is a completion that the model's own check accepts (>= 0.5).
    - witness-only: found ? max(p0, best check) : p0.
    - null search: found ? same : sigmoid(logit p0 + theta log(1 - C_M)), where C_M is the summed proposal mass of
      the distinct refuted completions. This is eta(C) = 1 - (1 - C)^theta, with theta fitted on validation outcomes.
    - count control: sigmoid(logit p0 - kappa M).
  - No rule checker at test time. The scorer alone uses the solver, for exact marginals and diagnostics.
- **Seeds 0, 1, 2.** Data and test boards are shared, so comparisons are paired. CIs are seed-level t (df 2), with a
  hierarchical bootstrap (seeds, then boards; 10,000 draws) as a second interval.

## Round 1 (pre-registered, 30k steps)

| seed | best step | val cell CE | val Solvable BCE | KL(P0) all | KL(P0) one-step | own-check false-positive rate (P1/P2/P3) |
|---|---|---|---|---|---|---|
| 0 | 28k | 1.006 | 0.574 | 0.549 | 0.475 | 0.39 / 0.64 / 0.74 |
| 1 | 29k | 1.039 | 0.594 | 0.583 | 0.523 | 0.58 / 0.77 / 0.83 |
| 2 | 30k | 0.665 | 0.292 | 0.205 | 0.000 | 0.00 / 0.00 / 0.00 |

D4 at M = 8 (relative log-loss cut, per seed; mean; seed-t 95% CI; bootstrap CI):

| rule | (a) vs P0, solvable | (a) vs P0, unsolvable | (b) vs witness-only, unsolvable | overall |
|---|---|---|---|---|
| P2 (primary) | 0.963, 0.992, 1.000; 0.985 [0.937, 1.033] [0.951, 1.000] | -6.72, -8.11, 0.958; -4.63 [-16.8, 7.5] [-8.08, 0.955] | 0.013, 0.003, 0.958; 0.32 [-1.04, 1.69] [0.003, 0.955] | **FAIL** |
| P1 | 0.85 [0.34, 1.35] | -4.70 [-16.3, 6.9] | 0.21 [-0.67, 1.08] | FAIL |
| P3 | 0.995 [0.985, 1.006] | -4.64 [-16.9, 7.6] | 0.34 [-1.08, 1.76] | FAIL |

Solvable log loss at M = 8, mean of 3 seeds (P2 chains): unsolvable boards P0 0.391, witness-only 2.604, null 2.495,
count 2.482; solvable boards P0 0.606, witness-only 0.006, null 0.011. ECE: P0 0.027, null 0.278.

Cell KL (mean over 3 seeds; seed 2 alone in brackets): P0 0.445 [0.205]. At M = 16: P1 0.411 [0.128], P2 0.384
[0.066], P3 0.375 [0.054]. Seed 2 alone at M = 16: P3 -74%, P2 -68%, P1 -38% against P0.

Curves: `results/r1/curves_r1-30k.png` (+ .csv) and `results/r1/curves_r1-seed2-only.png` (+ .csv).

## Round 1b (post hoc, 150k steps; everything else identical)

| seed | best step | val cell CE | val Solvable BCE | KL(P0) all | KL(P0) one-step | own-check accuracy on chain completions |
|---|---|---|---|---|---|---|
| 0 | 121k | 0.507 | 0.017 | 0.0475 | 0.000 | 1.000 (all rules) |
| 1 | 133k | 0.503 | 0.026 | 0.0467 | 0.000 | 1.000 |
| 2 | 95k | 0.503 | 0.017 | 0.0433 | 0.000 | 1.000 |

Chains complete the board validly 90-93% of the time (P1) and 99% (P2, P3).

D4 at M = 8:

| rule | (a) vs P0, solvable | (a) vs P0, unsolvable | (b) vs witness-only, unsolvable | overall |
|---|---|---|---|---|
| P2 (primary) | 1.000, 0.309, 1.000; 0.770 [-0.221, 1.760] [-0.234, 1.000] FAIL | 0.973, 1.000, 1.000; 0.991 [0.953, 1.029] STRONG PASS | same as (a)-unsolvable; STRONG PASS | **FAIL** (one board, see Verdict) |
| P1 | 0.9999 [0.9999, 1.0] STRONG PASS | 0.981 [0.910, 1.051] STRONG PASS | 0.981 [0.910, 1.051] STRONG PASS | PASS |
| P3 | 0.9999 [0.9999, 1.0] STRONG PASS | 1.000 [1.000, 1.000] STRONG PASS | 1.000 STRONG PASS | PASS |

Guard (pooled log loss at M = 8, null <= witness-only): holds in every seed and rule (for example P2: 0.0006 / 0.0069
/ 0.0000 vs 0.022 / 0.025 / 0.023).

Solvable log loss at M = 8, mean of 3 seeds:
- P0: unsolvable 0.041, solvable 0.018, ECE 0.0075.
- witness-only: unsolvable 0.041, solvable 0.000.
- null search: unsolvable 0.0003-0.0008 (P2, P1) and 0.0000 (P3).
- count: 0.000.

Cell KL vs M (mean ± sd over 3 seeds):

| M | P1 random | P2 VoI | P3 confidence-first |
|---|---|---|---|
| 0 (P0) | 0.0458 ± 0.0022 | same | same |
| 1 | 0.0438 | 0.0430 | 0.0423 |
| 4 | 0.0389 | 0.0369 | 0.0355 |
| 8 | 0.0333 ± 0.0014 | 0.0322 ± 0.0014 | 0.0303 ± 0.0014 |
| 16 | 0.0254 ± 0.0012 | 0.0271 ± 0.0013 | 0.0246 ± 0.0011 |

Cuts vs P0 (seed-t CI):

| rule | M = 8 | M = 16 |
|---|---|---|
| P1 | 27.4% [25.6, 29.1] | 44.5% [42.3, 46.8] |
| P2 | 29.8% [27.0, 32.5] | 40.8% [35.8, 45.7] |
| P3 | 33.9% [32.2, 35.5] | 46.3% [41.7, 50.9] |

Comparisons between rules:
- P2 vs P1: +3.3% [-1.8, 8.5] at M = 8 and -6.9% [-19.5, 5.8] at M = 16.
- P3 vs P2: +5.8% [4.3, 7.3] at M = 8 and +9.4% [8.3, 10.4] at M = 16.

Weighting chains by the model's own check helps P1 at M = 16 (0.0254 vs 0.0337 unweighted) and barely changes
P2/P3.

Curves: `results/r1b/curves_r1b-150k.png` (+ .csv).

## What this says about the thesis

- Search through the network's own stiff (low-entropy) conditionals does improve beliefs on long-range queries that
  one pass gets wrong. KL is cut about 45% at 16 chains, and solvability becomes near-certain both ways. But this
  only holds after the network has learned the local law. With an under-trained network, search manufactures
  confident false witnesses and makes things much worse. The learned check is the load-bearing part.
- Picking the stiffest cell (confidence-first) was the best rule at every M and the cheapest. The VoI rule spent
  about 37x the compute and did no better than random order at M = 16.
- "Searched and found nothing" is strong evidence here, so strong that the fitted theta goes to 7-80. That is
  calibrated on the test set on average (ECE 0.0003), but it has a tail risk. One solvable board in 857 with no
  witness in 8 chains got p = 1e-6. A floor on the null update, or a theta fitted with a held-out tail penalty, would
  remove that; neither was pre-registered.
- The mass-vs-count question is not answered by this testbed. Both updates reach about 0 loss because failures are
  near-deterministic once the law is learned. A harder testbed (bigger boards, fewer chains, or a weaker proposal)
  is needed to separate them.

## Deviations from the pre-registration

1. Round 1b (150k steps) is post hoc, triggered by the round-1 D0 one-step failure. It is labelled post hoc
   everywhere; the pre-registered verdict is round 1's FAIL.
2. The theta search grid was widened from [0.01, 100] to [0.01, 1000] after the 4x4 smoke hit the upper bound,
   before any 6x6 eval. `jobs/run.sh` gained a ROOT variable so two runs could share a VM (mechanical).
3. Compute: spot L4s were stocked out or preempted (2 failed creates at 01:11Z, 1 preemption at about 02:09Z). esud-f
   was created on demand (STANDARD) after a spot stockout, through gpu-up.sh's fallback, so the three
   round-1 seeds and one round-1b seed shared a single GPU. This has no effect on the numbers, only on wall time.
4. About 42.8% of test boards are solvable (target 50%), because the generator rejects more solvable candidates
   (fewer than 2 or more than 400 completions).
5. Clue counts are matched per board (an unsolvable board keeps its source board's count), but not by distribution:
   more solvable candidates are rejected at low clue counts, so test solvable boards average 13.6 givens and
   unsolvable ones 14.8. Clue count is therefore a weak shortcut for P0's Solvable head. It does not affect the
   search arms' witnesses, which are checked completions. (Found by the audit.)
6. Hashes: `PREREG.sha256` was rebuilt at about 02:03Z when the addendum was added, so it cannot prove the
   01:01Z freeze on its own. `jobs/run.sh` no longer matches (the ROOT change in item 2). `code/analyze.py` was never
   hashed and was edited after the round-1 results (a JSON float fix, and cell-cut tables added). Its D4 logic is the
   PREREG statistic, and the audit recomputed every verdict independently.
7. The D0 "fully revealed parents" query is implemented for Sudoku as naked-single cells given the board (scorer
   only).

## Cost

GCP L4 (g2-standard-8), budget group `overnight-sud`, cap $50. Ceiling-rate spend (on-demand list price x age; the
spot VMs bill less) from the DELETE lines in the shared logical-soup GPU-LOG.md:
- esud-a: 3.33 h, $2.90 (spot);
- esud-f: 2.00 h, $1.74 (on demand);
- esud-d: about 5 min before preemption, about $0.10.

Total about $4.74. The ledger's group total (`spend_usd overnight-sud`, $26.90 at 04:15Z) also carries $22.26 of
reservations from 3 failed creates and the preempted VM, which were never billed. All E-SUD VMs and disks are
deleted: at 04:15Z `gcloud compute instances list --filter=name~esud` and `gcloud compute disks list
--filter=name~esud` both returned nothing.

## Audit

Independent audit: `AUDIT.md`. Verdict: supported with corrections. Every D4 cut, CI and cell-KL cut was
recomputed from the per-board files and matches. The corrections (false-positive range, clue-count distribution,
hash and provenance notes, the never-billed reservations, esud-f on demand) are applied above.

## Files

- Code: `code/` (`boards.py` generator, `sud.py` model/train/eval, `check_exact.py`, `analyze.py`, `plot.py`),
  `jobs/run.sh`, `vm.sh`, `ckpull.sh`.
- Pre-registration: `PREREG.md`, `PREREG-ADDENDUM-1b.md`, `PREREG.sha256`.
- Results: `results/r1/`, `results/r1b/`. Each holds `eval_s6_seed*.json`, `train_s6_seed*.json`,
  `perboard_s6_seed*.npz` (per-board KL and log loss, used for the paired CIs), logs, `ANALYSIS.json` and curves.
  The exact-marginal cross-check is in `results/check_exact.json`.
- Trained models: `vmout/` (round 1: `esud-a/model_s6_seed*.pt`; round 1b: `r1b-a/`, `esud-f/`). Data:
  `data/s{4,6}_{train,val,test}.npz`.
- Reproduce one seed: `python code/boards.py 6 200000 44 data/s6_train.npz 8` (and the val/test lines in
  `data/gen.log`), then `jobs/run.sh SEED` with `STEPS=30000` or `150000`, then `python code/analyze.py results/<round>`.
