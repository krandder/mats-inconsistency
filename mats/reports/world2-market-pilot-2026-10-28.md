# W2 market world: build and first pilot (2026-10-28)

## Summary
- The world is built and its exact Bayes oracle passes all tests.
- On paper the world separates in-context inference from the shortcuts:
  - Each shortcut reference sits clearly above Bayes on its own split or diagnostic.
  - The C4 and C7 sets carry the pre-registered effect sizes.
- The pilot models do not learn the task, so the pilot cannot yet test the cruxes:
  - At N=128 the arms sit 0.41 to 0.52 nats above Bayes. That is worse than every shortcut reference except the prior; the bias-naive reference is about as bad.
  - Under the pre-registered rule this counts as a training failure, so C4 and C7 cannot be read as tests of the models.
- Cost: about $2.11 on GPU (one on-demand L4 for 2.43 h). The VM has been deleted.

## What was built (`work/worlds/`)
- `world2_market.py`: the generator (follows the spec in DESIGN-v1 W2).
  - Biases are continuous, and up to 8 sources per episode have an unknown noise level.
  - Item, source and event ids are random vectors drawn fresh for each episode.
  - Quote types are SPOT, DIFF and BASKET. DIFF is bias-free: an additive bias cancels in a spread.
  - A relay re-quotes an earlier measurement through the relayer's own bias, so it shares that measurement's noise. The token carries the root event id. The original quote is either in the context or hidden.
  - K and S grow with N.
- Exact Bayes: a mixture over at most 256 noise hypotheses, each a linear-Gaussian solve. It is checked against:
  - an independent brute-force implementation (covariance form);
  - Monte Carlo calibration (PIT);
  - a policy-invariance test. A value-adaptive policy that is ignorable stays calibrated; a policy that leaks hidden values is detected.
- References: prior, frequency-only, bias-naive, provenance-naive, noise-level oracle, Bayes.
- Diagnostic sets:
  - C4: frequency-matched pairs.
  - C7: two pairs that test whether shared evidence is discounted (a diamond against k-anchor graphs, k = 2 or 4) and three pairs that test provenance (relays against independent quotes).
- Other files:
  - `test_world2_market.py`: 23 tests.
  - `w2_summarize.py`: summary script.
  - `run_w2.sh` and `launch_w2.sh`: pilot run and cloud launcher.
  - `PREREG-W2.md`: written before training, with its sha256. The same section is appended to `PREREG.md`.

## Does the world tell inference apart from shortcuts? (references only, no model)
Gap above Bayes at N=128, policy `dir`:

| Reference | Gap (nats) |
|---|---|
| frequency-only | 0.10 (0.11 on the unknown-noise split) |
| bias-naive | 0.50 |
| provenance-naive | 0.03 (0.07 under the relay-heavy policy, rising to 0.13 at N=2048) |
| noise oracle | 0.00 |

Diagnostic sets:
- C4 at N=96: KL(Bayes‖frequency-only) has median 0.63 nats and minimum 0.33.
  - The pre-registered effect size was met on every pair: the median shift in Bayes between the two members of a pair is 4.8 nats.
- C7 at N=96 (N=512 in brackets), median KL(Bayes‖naive):

  | Set | Median KL (nats) |
  |---|---|
  | D/T | 0.06 (0.10) |
  | D4/T4 | 0.22 (0.40) |
  | HR4/HI4 | 0.21 (0.49) |

- The provenance-naive gap on the natural distribution is small. Provenance is therefore judged on the C7 sets.

## Pilot results
Setup:
- Arms: L2w128, L4w128 and L4w64, 2 seeds each.
- Training: 30k steps, batch 256, N between 8 and 128.

Training-distribution validation, gap to Bayes at the end of training:

| Arm | Gap (nats) |
|---|---|
| L4w128 | 0.20 to 0.21 |
| L2w128 | 0.25 |
| L4w64 | 0.23 to 0.26 |

Validation loss was still falling at the end, so the models are undertrained.

Gap to Bayes on `dir` (nats), by N:

| N | L2w128 | L4w128 | L4w64 |
|---|---|---|---|
| 64 | 0.36 | 0.29 | 0.36 |
| 128 | 0.49 | 0.41 | 0.52 |
| 512 | 0.78 | 0.77 | 0.91 |
| 2048 | 1.03 | 0.99 | 1.22 |

Pre-registered readings:

| Reading | Result |
|---|---|
| R3 (scaling) | Fails. KL grows 0.12 to 0.17 nats per doubling of N. |
| R1 (C4) | Fails for every arm and seed: 2 to 10% of episodes close half the gap, against 80% required. The mean-shift between pair members is 0.00, so the models make no source-specific inference at all. |
| R2 (C7) | Fails. Dedup fraction is at most 0.11 on every pair (0.02 to 0.08 on D4/T4 and HR4/HI4). |
| Training validity | Fails. No arm is within 0.10 nats of Bayes at N=128, so R1 to R3 are reported as a training failure, not as refutations. |

## Next options
- **Train longer or more efficiently.**
  - Data generation takes about 170 ms per step and is the bottleneck. Vectorising `pack` and prefetching data would speed it up.
  - Run more steps, or a curriculum over N.
- **Simplify identity matching.** For example, use learned id embeddings instead of random vectors, or narrow the range of K and S at the start.
- **Add the graded-overlap variant** the coordinator suggested. It is not implemented; it is noted in the prereg as planned.

Results: `work/worlds/runs_w2_1028a/` (`summary.md`, per-episode predictions, logs).

---

# W2b: making W2 learnable (2026-10-28, appended)

**TL;DR.** Models now learn W2 in the training range and pass C4 (in-context source inference) at N=96, which the
pilot failed completely. They still do not deduplicate shared evidence (C7 ~0, sometimes inverted) and the gap to
Bayes still grows with N. Formally, the pre-registered training-validity rule (within 0.10 nats of Bayes at N=128)
is still not met (best 0.24), so the C4 pass is reported as provisional. Spend $8.85 (two on-demand L4, 5.2 h and
5.0 h); both VMs deleted (`gcloud compute instances list` shows none). Total W2 + W2b: $10.96.

## What changed (PREREG-W2.md, W2b amendment, written before training)
- `w2fast.py`: generator, tokens and exact Bayes (per-item mixture moments and the query mixture) batched in torch on
  the GPU. Tested equal to the numpy path (tokens to 1e-5; Bayes per item to 1e-6) and distribution-matched to the
  pilot generator (`test_w2fast.py`, 8 tests). Query-only training step at full size: 28-36 ms with three arms
  sharing one L4 (pilot: 240-300 ms, of which 170-245 ms data). Dense arms: 110-200 ms/step, dominated by the fp64
  Bayes targets.
- Curriculum (N<=24, K<=4, S<=4, U<=1, no relays -> full distribution from 40% of steps), 150k steps, bf16.
- Dense arm (C5): one extra query token per item, loss += cross-entropy to the Bayes per-item posterior; same
  architecture and query function as the control (harness eval unchanged).
- Stepping stone (SPOT+DIFF, known noise, no relays): learned easily, so the full world was kept as the main target.
- Deviation: D6 was stopped at 120k of 129k steps by a deadline watchdog (VM timeout); its meta.json was written
  from its curve. D arms ran fewer steps than Q arms (142k / 120k vs 150k). Single seed per arm.

## Results (best.pt; seed 0)
Validation gap on the training distribution (pilot 0.20-0.26): Q4 0.124, Q6 0.102, D4 0.080, D6 0.071.
Stepping stone: SQ4 0.047, SD4 0.032.

Excess NLL over Bayes, policy `dir` (references: freq_only 0.10, bias_naive 0.50 at N=128):

| N | Q4 | Q6 | D4 | D6 | pilot best |
|---|---|---|---|---|---|
| 64 | 0.185 | 0.152 | 0.138 | 0.108 | 0.29 |
| 128 | 0.310 | 0.284 | 0.242 | 0.277 | 0.41 |
| 512 | 0.672 | 0.740 | 0.701 | 0.703 | 0.77 |
| 2048 | 1.009 | 1.155 | 1.175 | 1.285 | 0.99 |

- R3 (scaling): fails. KL grows 0.18-0.25 nats per doubling of N (steeper than the pilot, from a lower start).
  Even inside the training range the gap rises with N (0.11-0.19 at 64, 0.24-0.31 at 128).
- R1 (C4, N=96): closure >= 0.5 on D4 94%, D6 92%, Q6 84%, Q4 80% (threshold 80%; Q4 is 79.5%). The between-pair
  mean shift is 0.87-0.96 of Bayes's, with the right sign on 95-98% of pairs (pilot: 0.00, chance sign).
  At N=512, C4 does not hold up: 17-29% of episodes, shift 0.18-0.36.
- R2 (C7, N=96): dedup fraction D4/T4 -0.16 to -0.03; HR4/HI4 -0.05 to 0.19 (D6 best); N=512 all ~0. The models
  treat shared and independent evidence alike, and on diamonds are slightly more confident than with independent
  anchors. No topological or provenance dedup.
- R5 (C5, dense vs query-only at equal depth): val gap -0.044 (depth 4) and -0.031 (depth 6), below the 0.05
  threshold; C4 better (94 vs 80%, 92 vs 84%) and reached per step much earlier (C4 >= 80% at ~60k steps for
  D vs ~140-150k for Q). Per wall-clock hour, dense is not cheaper here (3-4x slower steps).

## Does W2 now separate in-context inference from shortcuts?
Yes for source inference: frequency-matched C4 pairs are solved by trained models (closure 0.8-0.94 at N=96),
which no frequency-only rule can do. No for provenance and diamond dedup: no model learned it, so W2 cannot yet
say whether dedup is learnable. The natural training distribution gives dedup a weak signal (prov_naive gap
0.02-0.07 nats), which is the likely reason.

## Next options (each needs your call)
1. Dedup signal: an ignorable training policy rich in relays and diamonds (prov gap >= 0.2), then re-run C7.
2. Length: train with N up to 512 on the fast path (cheap now for query-only arms).
3. Speed: fp32 or Woodbury-style Bayes targets for the dense arm.
Files: `work/worlds/w2fast.py`, `w2_train.py`, `test_w2fast.py`, `run_w2b.sh`, `launch_w2b.sh`; results
`work/worlds/runs_w2b_1028/` (summary.md, curves_a.md, curves_b.md, checkpoints).

---

# W2c: in-context dedup (2026-10-28, appended 13:16Z)

**TL;DR.** No dedup. On every C7 relay and diamond pair, at N=96 and 512, every arm has a dedup fraction between
-0.23 and 0.09 (threshold 0.5), including the arms trained on the relay-rich mix. The cue check therefore has
nothing to explain: removing provenance ids changes nothing in any arm. The small HR/HI signal (D6 arms, 0.08-0.2 at
N=96) disappears when the relay flag and origin-source fields are removed, so it is a relay-flag cue, not
provenance. The one clear gain is from training on N up to 512: the gap at N=512 roughly halves and C4 now holds at
N=512 (76-86% vs 17-29%). Spend $5.28 (two on-demand L4s, 3.0 h each). Both VMs deleted (`gcloud` lists none).
Total W2 + W2b + W2c: $16.24.

## Design (PREREG-W2.md, W2c amendment, written before training)
- Rich mix: 60% of episodes shared-rich, with relay rate 0.3-0.6, root choice concentrated (8.7 relays per top
  root vs 1.8), half of them with sparse anchors (most items reached by DIFF paths), and 70% of queries on the
  most-relayed item. Episodes with relays: 99%. prov_naive gap: 0.25 nats on rich episodes vs 0.03 on the old mix
  (mean; heavy-tailed, median 0.01). C7 graphs are constructed and never produced by the training generator.
- N_train log-uniform up to 512 (one range per batch). Warm starts from the W2b checkpoints because of the
  deadline. Arms: Q6-rich, D6-rich (dense targets on batches with N <= 128), and the control Q6-old (old mix, same
  N range and budget). Training stopped at 13:10Z after 99k / 99k / 112k steps, about 52-81% of the planned cosine
  schedule, so the learning rate was not annealed. Evaluated with model.pt. Single seed.

## Results (W2b arms, trained to N <= 128, shown for reference)

| | Q6-old | Q6-rich | D6-rich | W2b Q6 | W2b D6 |
|---|---|---|---|---|---|
| gap dir N=128 | 0.153 | 0.185 | 0.164 | 0.284 | 0.277 |
| gap dir N=512 | 0.350 | 0.418 | 0.457 | 0.740 | 0.703 |
| gap dir N=2048 | 0.657 | 0.672 | 0.909 | 1.155 | 1.285 |
| KL slope / doubling (128-2048) | 0.125 | 0.123 | 0.186 | 0.220 | 0.252 |
| C4 N=96 (closure >= 0.5) | 94% | 93% | 97% | 84% | 92% |
| C4 N=512 | 86% | 76% | 79% | 17% | 29% |
| C7 N=96 dedup D4/T4 | 0.02 | -0.12 | -0.06 | -0.16 | -0.03 |
| C7 N=96 dedup HR4/HI4 | 0.00 | 0.04 | 0.08 | 0.02 | 0.19 |
| C7 N=512 dedup HR4/HI4 | 0.00 | 0.02 | -0.01 | -0.03 | -0.01 |

Cue check on C7 relay pairs (dedup fraction: none / noprov / noflag / both; ties are tested on HI4 only):
- D6-rich at N=96: HR/HI 0.075 / 0.074 / -0.103 / -0.108. HR4/HI4 0.010 / 0.008 / -0.054 / -0.056.
- W2b D6 at N=96: HR/HI 0.216 / 0.217 / 0.054 / 0.064.
- Q arms: all |values| < 0.03.
- Value ties in HI4: model log-sd change at most 0.006 in every arm, so ties are not used as a copy cue.

## Reading
- R2 (C7): no dedup in any arm. The rich mix (R2 vs control) gives no improvement; Q6-old is at least as good as
  Q6-rich everywhere.
- R6 (cue): nothing to attribute. The faint relay effect uses the relay flag and origin fields, never event ids.
- Important caveat: the models are far from Bayes on the C7 graphs themselves, with KL 0.5-0.9 nats on the shared
  arms. That is worse than the naive double-counting reference (0.05-0.2). The C7 queries are reached only through
  2-hop DIFF paths, which the models do not propagate well. So C7 is currently limited by multi-hop propagation
  before dedup even comes into play. The A2 lesson (make copies common and costly) did not transfer, possibly for
  this reason.
- R3 (scaling): still fails (slope 0.12-0.19 per doubling), but long-N training halves the slope.

## Suggested next test (needs your call)
A one-hop dedup set the models can already solve: the query item gets one SPOT plus k relays of it, against k+1
independent SPOTs, with k in {1, 3, 7}. Train with that structure common, from scratch, to the full schedule.
Files: `work/worlds/runs_w2c_1028/` (summary.md, a/ and b/ with curves.md, cue.md, checkpoints); code
`w2fast.py` (rich mix), `w2_train.py`, `w2c_cue.py`, `run_w2c.sh`, `launch_w2c.sh`.
