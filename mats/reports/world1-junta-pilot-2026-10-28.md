# World 1 (hierarchical junta with copies): harness and pilot, 2026-10-28

Code, prereg and runs are in `work/worlds/`:
- `harness.py`: world-agnostic. The docstring documents the interface; `HARNESS_READY.md` tells the W2/W3 builders how to use it.
- `junta.py`, `test_junta.py` (23 tests, passing locally and on the VM), `diag_junta.py`.
- Prereg: `PREREG-W1.md`, hashed before training in `code_sha256.txt`.
- Results: `runs_w1pilot/`, with every table in `runs_w1pilot/tables.md` and per-episode predictions as npz files.

## Short answer

The world separates the abilities cleanly. On the same episodes, the references are ordered as designed:
- item-only Bayes is far behind full Bayes (0.4 nats at N=64 with τ²=0.25; 1.4 with τ²=0);
- copy-naive Bayes sits in between (0.18 / 0.29);
- R-known Bayes is slightly ahead.

The models are ordered by depth (1 < 2 < 4) on every metric.

The transformers do infer similarity in context: at N ≤ 64, depth 4 gets 55-60% of the item-to-Bayes gain. They dedup copies only partly. Their gap to Bayes grows steadily with N past the training range (N ≤ 64). So by the preregistered transversal-scaling rule, "scales" is refuted for this architecture.

## World as built

The spec is DESIGN-v1 W1 plus the coordinator's amendments.
- **Items and values.** 16-bit items, |R| = 2 (120 hypotheses). Cells g ~ N(0,1), items v = g + b with b ~ N(0, τ²), readings y ~ N(v, σ²), σ² ∈ {0.1, 0.5}.
- **Copies.** Exact copies carry the same event-id vector; the copy rate is 0 with probability 1/3, otherwise U(0, 0.6).
- **Policies.** Uniform, Zipf, pattern (a visible bit pattern on random coordinates, independent of R) and repeat are used in training; UCB on running item means is held out.
- **Randomised counts.** Every structural count is randomised per episode, and training N is log-uniform in [8, 64].
- **Relays dropped.** Hidden-ancestry relays were dropped, as you decided. One more reason: when a copy's provenance is hidden, the fresh-vs-copy odds include the policy's item probability, so Bayes would no longer be policy-free.
- **Exact Bayes** is a 240-component Gaussian mixture. It is tested against:
  - dense joint-Gaussian conditioning (agreement to 1e-9);
  - ABC on the generative spec;
  - reductions (no copies, copies removed, a single hypothesis);
  - a replay ignorability test for each policy: same policy randomness and same observed values under a different hidden world give identical items, copies and query;
  - z-score calibration under all five policies.
- **Cost.** About 4 ms per episode at N=64 and 25-85 ms at N=4096, for all references together.

## Pilot

24 arms: τ² ∈ {0, 0.25} × depth {1, 2, 4} × attention sink {0, 1} × 2 seeds. Width 64, 4 heads, a 4-component mixture head, 40k steps at batch 256. Gaps are model NLL minus Bayes NLL in nats per episode, averaged over seeds.

Training had not fully converged. The last-quarter improvement was 0.004-0.029 nats, so every model number is an upper bound on what these architectures can reach.

**Gap to Bayes, τ²=0.25, mean over the 4 training policies (held-out UCB in brackets)**

| predictor | N=16 | N=64 | N=256 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| depth 1 | 0.105 (0.112) | 0.311 (0.290) | 0.405 (0.520) | 0.476 (0.769) | 0.556 (0.946) |
| depth 2 | 0.077 (0.058) | 0.230 (0.177) | 0.294 (0.357) | 0.365 (0.594) | 0.447 (0.739) |
| depth 4 | 0.067 (0.037) | 0.191 (0.145) | 0.265 (0.302) | 0.369 (0.547) | 0.429 (0.575) |
| depth 4 + sink | 0.065 (0.042) | 0.193 (0.136) | 0.254 (0.296) | 0.373 (0.512) | 0.471 (0.651) |
| item-only Bayes | 0.174 (0.040) | 0.433 (0.211) | 0.564 (0.421) | 0.561 (0.474) | 0.549 (0.313) |
| copy-naive Bayes | 0.123 (0.109) | 0.183 (0.141) | 0.172 (0.057) | 0.149 (0.145) | 0.331 (0.067) |
| fixed shrinkage r0 | 0.233 | 0.506 | 0.615 | 0.604 | 0.665 |
| R-known Bayes | -0.138 | -0.058 | -0.014 | -0.002 | 0.000 |

**Gap to Bayes, τ²=0, training policies**

| predictor | N=16 | N=64 | N=256 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| depth 1 | 0.373 | 1.070 | 1.791 | 2.554 | 3.238 |
| depth 2 | 0.282 | 0.807 | 1.412 | 2.189 | 2.915 |
| depth 4 | 0.239 | 0.585 | 0.980 | 1.650 | 2.273 |
| item-only Bayes | 0.583 | 1.438 | 2.247 | 2.984 | 3.616 |
| copy-naive Bayes | 0.191 | 0.287 | 0.521 | 0.931 | 1.431 |

**Why the gap grows: bias and variance.** The growth is mostly precision that the model does not extrapolate.
- With τ²=0, the Bayes predictive sd shrinks roughly as 1/sqrt(cell count). At N=4096 the depth-4 model's predictive variance is e^5.4 ≈ 200 times Bayes's, and its mean error is 19 Bayes sds.
- With τ²=0.25 the item effects put a floor under the Bayes variance. There the variance log ratio stays at 0.4-0.55 and the mean error at 0.8-1.3 sd.
- So τ²=0.25 is the informative test of aggregation. τ²=0 mainly tests whether the model's precision grows with N, and it does not.

### Similarity inference

This is the fraction of the item-only-to-Bayes gain the model achieves, on training policies:

| config | N=16 | N=64 | N=256 | N=1024 | N=4096 |
|---|---|---|---|---|---|
| depth 1, τ²=0.25 | 0.37 | 0.27 | 0.27 | 0.12 | -0.02 |
| depth 2, τ²=0.25 | 0.55 | 0.47 | 0.47 | 0.33 | 0.18 |
| depth 4, τ²=0.25 | 0.61 | 0.56 | 0.52 | 0.33 | 0.22 |
| depth 4, τ²=0 | 0.59 | 0.59 | 0.56 | 0.44 | 0.36 |

The R-evidence pairs (D1) test this directly: same items, same multiset of values, values permuted among non-query readings, keeping pairs where Bayes moves by more than 0.5 sd. The model moves the right way. Regressing the model's shift on Bayes's gives a slope of:
- 0.22-0.27 for depth 1;
- 0.41-0.45 for depth 2;
- 0.50-0.61 for depth 4.

At N=256 the slope is higher, 0.68-0.79 for depth 4.

On held-out UCB with τ²=0.25, every model falls below item-only Bayes from about N=1024.

### Dedup

The copy-naive reference does all the similarity inference and only mishandles copies. The models trail it at N ≥ 32, so the "fraction of the count-to-Bayes gain" metric comes out negative; it is confounded and I do not use it.

The clean test is the D2 control from arXiv 2610.16801: add k exact copies of the query's single reading (Bayes unchanged) or k fresh readings with the same value (Bayes sharper). The table shows the change in log predictive variance; Bayes's change for copies is 0.

| config, τ²=0.25 | copies k=4 | fresh k=4 | copies k=64 | fresh k=64 |
|---|---|---|---|---|
| Bayes | 0 | -1.20 | 0 | -4.38 |
| depth 1 | -0.06 | -0.11 | -0.28 | -0.38 |
| depth 2 | -0.28 | -0.93 | -0.54 | -2.32 |
| depth 4 | -0.27 | -0.81 | -0.28 | -2.65 |
| depth 4 + sink | 0.00 | -0.23 | +0.29 | -0.37 |

- **Depth ≥ 2 without a sink** uses the event ids partly. Copies have about a third of the effect of fresh readings, and that effect stops growing with k. But it is not zero.
- **Depth 1** cannot tell copies from fresh readings (identical responses with τ²=0).
- **The sink** removes most of the reaction to copies, but it also suppresses the reaction to genuine fresh readings.

The whole-context ×k test (D4) behaves as Barbero predicts. Without a sink, the model is exactly invariant to ×8 copies (0.000), which is correct. For ×8 agreeing fresh readings it gets only 12-49% of the Bayes variance drop, and depth 1 gets about 0%.

Probes point the same way. In depth ≥ 2 arms, a token's log raw count (copies included) is decodable after layer 1 with R² 0.80-0.94, its deduplicated count with 0.60-0.83, and its cell count only with 0.10-0.33.

### C1 contamination

Setup: a clean balanced context of 48 items, plus M readings of one item at Hamming distance 8 from the query, in another cell.
- **Bayes barely moves.** Its contamination KL(p(D_M) || p(D_0)) is at most 0.05 nats (τ²=0) and 0.002 (τ²=0.25); for copies it is about 0.
- **The models move a lot.** At M=4048 their contamination is 0.27-2.0 nats, and copies contaminate as much as fresh readings.
- **Attention follows proportion.** Layer-1 query attention mass on the dominant item at M=512 is 0.70-0.89, against its share of 0.91.
- **Excess over length alone.** With τ²=0.25, the matched-length control stays flat (KL 0.25-0.42) while the dominated context rises: depth 4 reaches 0.68 at M=4048 against 0.30 for the control.
- **The rule is vacuous.** The preregistered C1 rule needs a control KL below 0.05, and no arm gets there, because the in-distribution KL at M=0 is already 0.34-1.3. A non-vacuous C1 needs a model much closer to Bayes first.

### Predictions scorecard

- **P1** similarity ≥50% (depth 4) and ≥25% (depth 2) at N=64: held (0.56-0.59, 0.44-0.47). Depth 1 under 25%: missed narrowly (0.25-0.29).
- **P2** dedup ≥70% of the count-to-Bayes gain: failed, and the metric is confounded (above).
- **P3** gap grows with N:
  - held for τ²=0;
  - for τ²=0.25 the ratio gap(1024)/gap(64) is about 1.9 for depth 4, just under the predicted 2;
  - worse than item-only at N=4096 on UCB: held.
- **P4** UCB at N ≤ 64 within 0.05 of in-distribution: held (UCB is easier there).
- **P5** the sink slows gap growth: failed (no better at depth 2-4, worse at depth 1).
- **P6** attention roughly proportional and contamination above Bayes: held for depth 2-4, and the rule was vacuous as predicted.
- **P7** D4: held.
- **P8** copies |Δlog var| < 0.1: failed without a sink. Fresh k=4 ≥50%: held without a sink.
- **P9** D1 slope ≥0.5 for depth 4: held. "Lower at N=256": failed (higher).
- **P10** count probe R² > 0.5: held.

## Cost and wall clock (for planning the full sweep)

One on-demand L4 VM (g2-standard-8) ran 02:00-06:01Z: 4.0 h × $0.87/h list = about $3.5. It was deleted by the run script, and `gcloud compute instances list` shows no w1junta VM.

With 8 arms sharing one GPU and 8 vCPUs, the per-arm step time was:
- 35 ms for depth 1;
- 64 ms for depth 2;
- 123 ms for depth 4.

Episode generation takes about 35 ms of CPU per batch of 256, so data generation becomes the bottleneck once the GPU is free. 40k steps took 23 / 43 / 82 min.

Evaluation per arm (45 cells, KL by quadrature on CPU) takes about 20 min. Diagnostics take about 6 min. Building the reference set takes about 4 min per world.

The forward pass costs 0.02 ms per episode at N=64 and 48 ms at N=4096 (depth 4, under contention).

Average cost is about $0.15 per arm for training plus evaluation. A 10-seed sweep of 6 configs per τ² would be about $18. The cheapest speed-up is moving episode generation to a second process.

## Notes

- `PREREG.md` is shared between lanes. My write at 01:59Z overwrote the W3 section, which that lane restored by appending it. The W1 prereg as hashed is `PREREG-W1.md` (identical bytes).
- What to change before a full sweep:
  - train longer (still improving at 40k steps);
  - report τ²=0.25 as the main world;
  - replace the count-to-Bayes ability with the D2/D4 controls;
  - make C1 non-vacuous by comparing against the control's KL rather than an absolute 0.05.

# W1b: does predictive precision scale with evidence? (2026-10-28, appended)

The prereg amendment is in `PREREG-W1.md` (W1b section plus a continuation note). Code is in `harness.py`, which
gains `--sum-head`, `--nmode batch` and `--init`, all off by default, plus `run_w1b*.sh`. Tables:
`runs_w1b/tables_100k.md` and `runs_w1b2/tables_200k.md`.

Setup:
- **World and arms.** τ²=0. Depth 4, width 64, 3 seeds per arm. One N per batch, at about 16k context tokens per step.
  - A: N_train ≤ 64.
  - B: N_train ≤ 512.
  - C: B plus one unnormalised sum head per layer, sigmoid-gated with sign·log1p output.
  - D: B plus a learned sink.
  - E: C at N_train ≤ 64.
- **Training length.** Every run did 100k steps, then 100k more from its final weights. That continuation is a
  deviation, recorded in the prereg before it ran.
- **Convergence.** No run meets the convergence rule even at 200k: the last-quarter improvement is 0.015-0.06 nats,
  and B, C and D are still improving. So every model number is an upper bound.
- **Stability.** The qualitative verdicts are the same at 100k and 200k, except the (a)/(b) threshold below.

**Gap to Bayes at 200k, training policies, mean over 3 seeds (sd ≤ 0.04 up to N=1024)**

| arm | N=64 | N=256 | N=512 | N=1024 | N=4096 | held-out UCB N=4096 |
|---|---|---|---|---|---|---|
| A (≤64) | 0.440 | 0.755 | 1.090 | 1.439 | 2.100 | 2.79 |
| B (≤512) | 0.440 | 0.450 | 0.494 | 0.606 | 1.013 | 1.45 |
| C (≤512 + sum head) | 0.418 | 0.430 | 0.476 | 0.641 | 2.204 | 2.92 |
| D (≤512 + sink) | 0.433 | 0.447 | 0.490 | 0.601 | 1.038 | 1.47 |
| E (≤64 + sum head) | 0.428 | 1.165 | 2.965 | 3.537 | 4.319 | 3.84 |
| item-only Bayes | 1.438 | 2.247 | 2.601 | 2.984 | 3.616 | 2.91 |

**Split of the KL at N=4096 (variance part / mean part / total), and log of model variance over Bayes variance**

| arm | split at N=4096 | log var ratio, N=64 → 512 → 4096 |
|---|---|---|
| A | 1.99 / 0.33 / 2.10 | 1.25 → 2.86 → 4.96 |
| B | 0.99 / 0.23 / 1.02 | 1.22 → 1.53 → 2.88 |
| C | 1.47 / 0.52 / 2.21 | 1.15 → 1.46 → 3.88 |
| D | 1.06 / 0.27 / 1.05 | 1.18 → 1.53 → 3.04 |
| E | 3.35 / 0.99 / 4.32 | 1.18 → 4.60 → 7.70 |

## What it says

**The failure is mostly precision.** For A, the variance part is 94% of the KL at N=4096. The model's predictive
variance stays about e^5 ≈ 140 times Bayes's.

**Training on larger N fixes about half, and the rest grows even inside the training range.**
- B halves the gap at every large N (1.01 against 2.10 at N=4096), and it degrades again past its range: gap(4096)
  is 2.0-2.2 times gap(512).
- Inside its own range, B's gap still rises (0.44 → 0.49) and its variance ratio grows from e^1.22 to e^1.53.
- So the cause is part length extrapolation and part representational.
- By the preregistered thresholds, B/A at N=512 was 0.50-0.52 at 100k, which reads as (b). At 200k it is 0.43-0.47,
  but B/A at N=256 is 0.58-0.62, so the (a) criterion fails too. Neither clean verdict holds, which fits the mixed
  reading.

**The sharp architecture result is not supported.**
- E extrapolates worse than A. Its gap at N=4096 is 4.3 against 2.1, and its variance part is 1.5-2 times A's.
- C matches B inside its range, then blows up past N=1024 (2.2 against 1.0 at N=4096).
- The sum head as built feeds features that leave their trained range as N grows. It is not a precision channel
  that extrapolates.
- The sink (D) is indistinguishable from B (variance part within 7%).

**Copies vs fresh readings of the query (D2): only B handles both.** The table gives the change in log variance,
copies / agreeing fresh; Bayes is 0 for copies and -0.23 / -0.71 / -1.69 / -3.67 for fresh at k = 4 / 16 / 64 / 256.
- B: copies +0.01 to +0.27, fresh -0.34 / -0.74 / -1.58 / -2.27. It dedups copies and tracks Bayes's fresh reaction
  up to k=64.
- A: copies -0.14, fresh -0.53 at k=4. It over-reacts to fresh readings and partly to copies.
- C, D and E barely react to either (|change| ≤ 0.15 for k ≤ 16), so their copy/fresh ratio of about 1 is
  uninformative.

**Whole context ×8 (D4).**
- A and B are exactly invariant to ×8 copies, which is correct.
- The sink and sum arms are not, and they move the wrong way: their variance grows (+0.14, +0.33, and +1.67 for E).
- For ×8 agreeing fresh readings, B gets 22% of the Bayes variance drop.

**C1** is vacuous for every arm. The best matched-control KL is 0.18 (C at M=64), never below 0.05.

## Scorecard
- Q1 held.
- Q2 held descriptively, but the formal (a) threshold failed at N=256.
- Q3 failed.
- Q4: held that E does not extrapolate, but its variance part grew instead of halving.
- Q5 held.
- Q6: C and E show a ratio of about 1, but only because they ignore the query's readings.
- Q7 held.

## Spend
6 on-demand L4 VMs at about 2.1 h each (12.55 VM-h × $0.87) ≈ $10.9 of the $15 cap. Four further launches failed on
a us-central1 L4 stockout or a zone without g2 machines, and created no instance. Every VM was deleted by its run
script, and `gcloud compute instances list` shows none.

## Next, if wanted
The architecture question needs a precision channel whose features stay in range as N grows. Two candidates:
- a learned count fed through log(N), normalised by a per-layer statistic;
- attention that outputs both the normalised mean and log Σ exp(scores).

Train it at N ≤ 64 and test it as arm E was. Longer training of B is the other open item, since it has still not
converged.
