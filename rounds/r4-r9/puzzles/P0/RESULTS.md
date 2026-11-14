# P0 puzzle ruler: results

Run on 2026-09-29, 13:11 to 17:38 UTC, plus CPU evals on General afterwards. Numbers are from
`results/p0_results.json`, which `analyze.py` rebuilds from the synced eval files. The chart is `results/chart_p0.json`.
Intervals are 95% t-intervals over seeds (mean ± half-width). Exact accuracy counts a sample only when the whole
output is right. Every eval ran in strict fp32.

## TL;DR

- **Maze bar (R2): FAIL.** Our width-64 DT-Recall retrain (M1) scores 0.0 / 0.0 / 0.0 on 33x33 at 300 iterations
  after 15k steps. The 30k-step run (M1-long) scores 0.086. The pretrained width-128 model (M-ref) scores 1.0. This is
  falsify branch (b), so **the maze bar becomes M-ref (w=128), and maze economy experiments should run at w=128.**
- **Prefix bar (R3): PASS.** P1 solves fL512 at 300 iterations on 8/8 seeds: 0.9998 ± 0.0006.
- **Credit horizon (F1): refuted on prefix sums, inconclusive on mazes.** k=1 credit (P4) also solves fL512 on 8/8
  seeds (0.9948 ± 0.0068; Fisher p = 1.0; median gap 0 pp). On mazes every arm is under-trained at 15k steps.
- **Market-resnet vs progressive loss (F2):** the spec's test (T=24 on fL256 and fL512) is at floor. MR-P, pilot A,
  pilot B and pilot Bds all score 0.0 there. Where the arms are off the floor (legacy sets), MR-P is about level with
  pilot B and above pilot A. The pilot's gain is therefore not clearly from per-step pay, but pilot B is a single seed.
- **Halting headroom (F3): falsifier 4 fires.** A label-free KL stop rule with one fixed threshold (3e-4) comes within
  1 pp of the stably-correct oracle. It does this at 1.013 to 1.018x the oracle's mean iterations, on 33x33 (M-ref)
  and on fL512 (all 8 P1 seeds). These tasks leave only about 1 to 2% of compute for a value-of-computation market.
- **Cost:** $6.19 of GPU at the $0.72/h ceiling (4 spot L4 VMs, 8.60 h) against the $8 cap. Every trader-cat-p0 VM
  and disk is deleted: `gcloud compute instances list` and `disks list` filtered on `^trader-cat-p0` were both empty
  at 17:38 UTC.

## R1: harness (checks from `tests.py`, the resume test and the pilot evals)

| check | result |
|---|---|
| DTNet reimplementation vs the published code (same weights) | max abs diff 0.0; parameter counts match |
| DT-L reimplementation budget | width 89, 157,261 params (< 160k) |
| kill-and-resume from `.partial` (maze and prefix) | weights and logs identical |
| M-ref on 59x59 at 3000 iterations | 1.0 |
| M-ref on shift cell n9, p=0, dead-end start off (target 0.72 ± 0.08) | 0.72 |
| `check.py` lattice checker vs exact match | 0 disagreements in 80 evals |
| market-resnet pilot reload through `mr_adapter.py` | max diff 0.002 vs the pilot's own numbers |
| bitwise repeat of an eval | identical |
| TF32 vs strict fp32 (M-ref, m59 x 20) | both 1.0; 115 s vs 181 s (reported only; evals stay fp32) |
| n33 easy-to-hard vs maze-dataset curves for M-ref | **max diff 0.19 at 75 to 100 iterations** |

R1 reads `ok: false` only because of the last row. The two 33x33 generators make mazes of different depth.
Normalised per dead-end-removal round, the first stable iteration matches (median 0.604 vs 0.599 per round, with
correlation 0.93 against the number of dead-end rounds). So the gap is in the data, not the harness. It matters only
for comparisons across generators. P0 compares arms on the same frozen sets.

## R2: maze bar (w=64, bansal progressive loss, alpha 0.01)

| arm | steps | seeds | 33x33 @300 | 33x33 @1000 | 59x59 @3000 | best val 9x9 |
|---|---|---|---|---|---|---|
| M1 | 15k | 3 | 0.0 / 0.0 / 0.0 | 0.0 / 0.01 / 0.0 | not run (cut) | 0.005 / 0.007 / 0.042 |
| M1-long | 30k | 1 | 0.086 | 0.088 | 0.06 | 0.996 |
| M-ref (pretrained, w=128) | published | 1 | 1.0 (0.538 at 100) | 1.0 | 1.0 | n/a |

- M1 stays at floor through its roughly 8k-step warm-up, and seed 1 collapses to loss ln 2. The 33x33 probe of
  M1-long reads 0 at 5k, 10k and 15k steps. It lifts only after 15k: 0.10, 0.08 and 0.16 at 20k, 25k and 30k steps.
  Validation 9x9 goes from 0.005 at 15k to 0.98 at 20k and 0.996 at 30k. M1-long solves 9x9 (1.0) and mostly solves
  13x13 (0.883), but does not extrapolate to 33x33 (0.086; 0.08 on the 100 hardest at 3000).
- Falsify rules. (a) does not apply: the probe was flat at 0 through 15k rather than rising, and M1-long does not
  pass. **(b) applies:** M1-long is below 0.50 on 33x33 while M-ref passes. (c) does not apply.
- Caveat on (b): M-ref is the published checkpoint with its own, longer schedule. So "only width differs" is
  approximate, and this run cannot separate width from training length. What the run does show is that w=64 does not
  reach the bar within a 30k-step budget, while w=128 already exists and passes.
- Overthinking: no arm loses more than 5 pp between its chosen iteration count and 3000 iterations.

## R3: prefix bar (fL = lengths far beyond training, strict fp32)

| arm | seeds | fL128 @300 | fL512 @300 | fL512 @3000 | successes (>= 0.90) |
|---|---|---|---|---|---|
| P1 (DT-Recall 1D, bansal, alpha 1) | 8 | 0.9998 ± 0.0006 | 0.9998 ± 0.0006 | 0.9998 ± 0.0006 | 8/8 |
| P-L (our DT-L reimplementation) | 2 | 0.0 | 0.0 | 0.0 | 0/2 |

**P-L is a Problem.** Our DT-L reimplementation does not learn the task at all. That is a defect in the
reimplementation, not evidence about DT-L, and P-L should not be used as a baseline until it is fixed.

## F1: credit horizon (matched ladder)

Prefix sums, fL512 at 300 iterations:

| arm | credit | seeds | mean ± 95% | median | successes |
|---|---|---|---|---|---|
| P1 | full progressive (bansal) | 8 | 0.9998 ± 0.0006 | 1.0 | 8/8 |
| P4 | k=1 chunk | 8 | 0.9948 ± 0.0068 | 1.0 | 8/8 |
| P4-cm | k=1, compute-matched (22.4k steps) | 3 | 0.966 ± 0.130 (1.0, 0.992, 0.906) | 0.992 | 3/3 |
| P5 | k=8 chunk | 2 (cut) | 1.0 | 1.0 | 2/2 |

- The prediction needed P4 to have fewer successes than P1 (Fisher p < 0.05) and a median at least 10 pp lower. It
  gets p = 1.0 and a median gap of 0 pp. The per-seed paired bootstrap of P1 − P4 is at most +2.2 pp (95% interval
  up to 3.6 pp). **This meets the refutation rule on prefix sums.** P4 and P4-cm are within 3 pp of P1 in median and
  within one success. P4-cm has no gap to close.
- Mazes, 33x33 at 300 iterations, M4 (k=1) minus M-prog (bansal, alpha 1) for each seed pair: +0.0, −31.6 and
  +0.2 pp. M-prog scores 0.0 / 0.316 / 0.006 and M4 scores 0.0 / 0.0 / 0.008. The prediction (≥10 pp lower on all 3
  pairs) fails on 2 pairs, and the refutation rule (within 3 pp on every pair) fails on seed 1. **Inconclusive.**
  Every maze arm is under-trained at 15k steps (see R2). At smaller sizes the sign is mixed: on 9x9 at 30 iterations
  M4 minus M-prog is +61.8, −26.1 and −8.9 pp.
- **Side finding: k=1 credit learns accurate but non-convergent dynamics.** On fL512 at 300 iterations, all 8 P1 seeds
  end at a fixed point on 500/500 strings. The 8 P4 seeds end at a fixed point on 0, 497, 500, 264, 84, 18, 98 and
  434 strings; the rest end in 2-cycles or wander, yet their answers stay correct. P4 seed 0 is still at 1.0 after
  30,000 iterations on 20 strings, with 0 fixed points, 6 2-cycles and 14 other end states. Any halting or pricing rule
  that relies on the state converging would never fire on such models. (M-ref on 33x33 also ends in 2-cycles:
  411/500.)

## F2: market-resnet (MR) backbone vs the progressive loss (P0-7)

MR-P uses the market-resnet backbone (w=128) trained with P1's progressive loss and random starts: T=24, 10k steps,
3 seeds. MR-A, MR-B and MR-Bds are market-resnet's own pilot checkpoints, one seed each. We re-evaluated them in this
harness on CPU.

| set @ iterations | MR-P (3 seeds) | pilot A | pilot B | pilot Bds | P1 |
|---|---|---|---|---|---|
| fL256 @24 (spec test) | 0.0 / 0.0 / 0.0 | 0.0 | 0.0 | 0.0 | — |
| fL512 @24 (spec test) | 0.0 / 0.0 / 0.0 | 0.0 | 0.0 | 0.0 | — |
| fL256 @100 | 0.019 (0.05, 0.008, 0.0) | 0.0 | 0.006 | 0.0 | — |
| fL512 @100 | 0.0 | 0.0 | 0.0 | 0.0 | — |
| fL512 @300 | 0.0007 ± 0.0029 | — | — | — | 0.9998 |
| legacy L64 @100 | 0.999 ± 0.001 | 0.852 | 1.0 | 0.962 | — |
| legacy L128 @100 | 0.244 (0.508, 0.164, 0.06) | — | 0.334 | — | — |
| legacy L128 @256 | 0.552 (0.628, 0.262, 0.766) | 0.054 | 0.41 | 0.07 | — |
| legacy mix 8–64 @24 | 0.74 | 0.556 | 0.75 | — | — |

- Falsifier 5 ("MR-P matches or beats MR-B within 1 pp on fL256 and fL512") holds as written: `analyze.py` reports
  `falsifier5_MRP_matches_or_beats_B = true`. But it holds only at floor, so on its own it says little.
- The informative rows are the legacy sets. There, MR-P is level with pilot B on L64 @100 (0.999 vs 1.0) and on
  mix @24 (0.74 vs 0.75). It is above pilot B on L128 @256 (0.552 vs 0.41) and below it on L128 @100 (0.244 vs 0.334,
  within MR-P's seed spread). It is above pilot A everywhere. **Reading: the progressive loss with random starts
  recovers about all of pilot B's gain over pilot A. The pilot gain is most likely a loss or schedule effect, not
  evidence for per-step pay.** Pilot B is one seed, and MR-P's seeds spread by up to 0.5.
- Backbone finding: on these extrapolation sets the market-resnet backbone is far behind DT-Recall. At fL512 @300 it
  scores 0.0007 against P1's 0.9998. Any market-resnet economy result on prefix sums should be read against that bar.
- P-mr (evaluating the final market-resnet grid checkpoints) was not run. No `GRID_DONE` exists under
  `market-resnet/results/vm`, and the spec forbids `.partial` files and taking files from that VM. Market-resnet
  already has its own evals of its P_s0..2 in its own harness.

## F3: halting headroom (P0-6)

Stably-correct oracle: stop at the first iteration after which the answer stays right. Rules: fixed T, KL between
successive predictions (eps), max confidence, a logistic stop probe, and a size-aware T.

| subject | oracle mean iterations | best rule gap at matched iterations | gap at 1.05x | iterations to within 1 pp | fixed KL eps 3e-4 |
|---|---|---|---|---|---|
| M-ref 33x33 | 102.1 | 7.8 pp | 0.6 pp | 1.013x | 0.99 at 1.013x |
| M-ref 33x33 hardest 100 | 101.6 | 8.0 pp | 0.0 pp | 1.008x | 1.0 at 1.014x |
| P1 fL512 (8 seeds, median) | ~101 | 45.4 pp | 0.0 pp | 1.011x (1.010 to 1.014) | 0.992 to 1.0 at 1.013 to 1.018x |

- The F3 prediction holds only in its letter: the best rule is ≥ 2 pp from the oracle at exactly matched iterations.
  That gap is a cliff, though, because accuracy jumps from near 0 to 1 within a few iterations of the needed depth.
  **Falsifier 4 fires on both tasks:** within 1 pp at ≤ 1.2x (actually ≤ 1.018x). One fixed KL threshold (3e-4)
  does this for all 10 subjects, so the result does not depend on the per-point Pareto choice of eps.
- Caveats: at eps ≤ 3e-5 the KL rule never fires on 12 to 36% of M-ref mazes, because of the 2-cycles above. M1 and
  M1-long are at floor, so the maze subject is M-ref.

## Distribution shift (P0-5; M-ref at 300 iterations, exact accuracy, 100 mazes per cell)

| size | p (extra openings) = 0, de 0 / de 1 | p = 0.01, de 0 / de 1 | p = 0.05, de 0 / de 1 |
|---|---|---|---|
| 9x9 | 0.72 / 1.0 | 0.63 / 0.84 | 0.33 / not generated |
| 19x19 | 0.74 / 1.0 | 0.31 / 0.47 | 0.02 / 0.04 |
| 33x33 | 0.77 / 0.99 | 0.07 / 0.07 | 0.0 / 0.0 |

"de" is dead-end start. M-ref is near perfect on its own distribution (de 1, p 0), which reproduces the published
0.72 on the de 0 cell. It collapses when mazes get loops (p > 0) and as size grows. Exact, `check.py` and Knutson
scoring agree in every cell. The cell 9x9 p 0.05 de 1 could not be generated (maze-dataset raises
NoValidEndpointException).

## Simplifications and cuts (all logged in `gpu/LEDGER.md`)

1. **Strict-fp32 eval was about 6x the spec's cost estimate.** Maze evals gain nothing from sharing the GPU. At
   14:00 both lanes were re-planned (`main-a2`, `main-b2` in `vm/plan.py`; M-prog moved from VM a to VM b). Three
   load-balancing moves followed: the M-ref evals went to VM b (`vm/side_b.sh`), P4-cm and P-L went to VM a
   (`vm/side_run.py`), and M-prog seed 2 went to a forced side unit (`vm/side_force.py`).
2. **All five cuts in the spec's order:** stretch (201x201) not run; P6 dropped; M5 dropped; M1-long stops at 30k; P-L
   at 2 seeds. Also cut: maze M4-cm (so compute-matched k=1 is answered on prefix only); P5 seeds 2 and 3 (P5 is
   descriptive, and P1 and P4 are at ceiling); the 59x59, shift and 30k-stability tail evals of M1 and M4 (at floor on
   33x33, so 0 by construction); and the 59x59 evals of M-prog (for GPU time; seeds 0 and 2 are at floor on 33x33,
   seed 1 scores 0.316). Seed floors kept: P1 and P4 at 8 seeds; M1, M-prog and M4
   at 3.
3. Evals use the best-validation checkpoint. Final checkpoints were kept on the VMs but not evaluated.
4. Stability test: 20 strings (P1 and P4, seeds 0 and 1) instead of 100; the M1 maze stability test was cut.
5. Halting probe: a logistic stop probe trained on the training-size validation set. The size-aware T tunes c on
   v33 / vL512.

## Incidents

- **VM a was preempted at 16:31:48** (last verified sync 16:30:13, so no finished result was lost). The watchdog's
  recreate failed on an L4 stockout in all 5 of its zones. A manual recreate over 13 zones with a 2 h max-run started
  `trader-cat-p0-a-r2` in us-central1-c at 16:45. M1 seed 2 and M1-long resumed from `.partial`. The M1-long evals
  ran first, then its 59x59 eval.
- VM b's lane restarted at 13:45 when unattended-upgrades re-executed systemd. Apt timers were then masked on both
  VMs.
- Resumed runs first crashed ("RNG state must be a torch.ByteTensor"). Fixed with `.cpu()` in `train.py` and
  re-verified.
- The first watchdog mistook a slow `gcloud list` for a vanished VM and made a duplicate (`trader-cat-p0-a-r1`,
  0.03 h, deleted unused).
- The ledger row for VM a is billed to 16:42 (the recreate), not to the 16:31 preemption, so it over-counts by about
  $0.12.

## Cost

| VM | hours | est $ (at $0.72/h) |
|---|---|---|
| trader-cat-p0-b | 4.20 | 3.03 |
| trader-cat-p0-a | 3.49 | 2.51 |
| trader-cat-p0-a-r1 (duplicate) | 0.03 | 0.02 |
| trader-cat-p0-a-r2 | 0.88 | 0.63 |
| **total** | **8.60** | **6.19** |

The CPU evals on General (pilot A, B and Bds on fL256 and fL512) were nice'd, 2 threads each. No paid LLM API was used.

## Files

- Code: `dtnet.py`, `dtl.py`, `train.py`, `mr_adapter.py`, `p0eval.py`, `evalrun.py`, `check.py`,
  `build_frozen.py`, `tests.py`, `analyze.py`, `vm/plan.py`, `vm/run_lane.py`, `vm/side_run.py`, `vm/side_b.sh`,
  `vm/side_force.py`, `p0_gpu.sh`.
- Frozen sets with sha256: `frozen/manifest.json`. Synced VM outputs (sha256-verified): `results/vm/trader-cat-p0-a`,
  `results/vm/trader-cat-p0-b`. General CPU evals: `results/vm/general/eval`.
- Results: `results/p0_results.json`, `results/chart_p0.json`. Ledgers: `gpu/LEDGER.md`, `../../GPU-LEDGER.md`.
