# How big must the discrete part be? Gate entropies of noisy networks (29 Oct 2026)

- **Where everything is.** Work directory `work/gate-entropy/`: code, logs and `results/*.json`. Tables are in
  `results/report_tables.md` (from `report_tables.py`), `results/scaling.md` (from `scaling.py`) and
  `results/tables.md` (from `summarize.py`).
- **Pre-registration.** `PREREG.md` was hashed before the main runs. Two dated amendments are hashed in
  `prereg_hashes.txt`.
- **Brief.** Main, relaying Kelvin: replace Gaussians by families that put mass correctly on each side of the decision
  boundaries. This census measures how large the discrete part of such a family must be. It builds no family.
- **Review.** One GPT-6 review was done before the claims (`review/`).

## Summary

For each layer (each step for the RNNs) we measured three entropies, over inputs and noise:
- **H(G):** the gate pattern, i.e. which side of each kink every unit is on;
- **H(G_R):** the D gates whose flip changes the question's answer on at least 1% of held-out samples;
- **E3:** the coarsest merging of gate patterns we found in which each merged class shares one pattern and at most 1%
  (or 0.2%) of answers change. It is checked on held-out samples by running the real network with the shared pattern.

E3 is a size we achieved. It is an upper bound on what this kind of family needs, not a lower bound. Units are bits.

1. **AlgZoo 2nd-argmax RNNs: the discrete part is small, and merging cuts it by 1-3 bits per step.**
   - T16 (ARC's example M_{16,10} fine-tuned at sigma = 0.001, 16 gates per step):
     - H(G_t) = 4.1-6.0 at steps 2-10. Plug-in on 2^21 samples is within 0.02 of the reference; NSB and MADE agree.
     - All 16 gates are relevant from step 4 on.
     - E3 = 2.3-4.1 at a 1% answer-change rate and 2.4-5.8 at 0.2% (ARC's bar), so at most about 60 cells per step.
   - W32 and W64 (sigma = 0.01):
     - H(G_t) = 6.8-10.3 and 8.9-12.8 at steps 2-10;
     - E3(1%) = 2.4-7.2 at steps 2-8 (W32) and 2.5-7.4 at steps 2-6 (W64);
     - at the later steps the tree, trained on 8,192 samples, stopped with 1.0-2.5% of answers changed at 8.5-9.6 bits.
   - At equal steps, E3(1%) is 1-2 bits higher at width 64 than at width 32.
2. **Phase B PFN (sigma = 0.05): no merging found.** The question is the predictive mean, to 10% of its RMS deviation
   from Bayes (a tolerance of 0.0063).
   - Layer 1 has 243 relevant GELU gates out of 1,024. Almost all sit on the three tokens that carry the sources' query
     readings and on the query token.
   - Their pattern never repeats in 2^15 episodes. H(G_R) lies between 27 (approximate collision bound) and 61 (FVSBN).
   - E3(1%) was not reached in either layer:
     - layer 1: 90% of answers still change at 8.4 bits;
     - layer 2: 52 relevant query units, H(G_R) = 11.6-16.0; 18% of answers still change at 8.3 bits.
3. **Random MLPs: the merged entropy grows in proportion to width at every depth.**
   - Setup: depth 32. The question is the output mean, and a sample counts as changed if its output moves by more than
     10% of the output's spread.
   - Every gate is relevant at every layer (D = m).
   - ARC He-ReLU at sigma = 0.01, H(G) by width 32 / 64 / 128 / 256:
     - layer 1 (rigorous Gaussian bound, with FVSBN upper bounds): 22-28 / 45-56 / 87-110 / 173-221 bits, about 0.7 bit
       per gate;
     - layer 32: 3.7 / 8.8 / 16-18 / 28-40 bits, about 0.13 bit per gate.
   - Chaotic tanh does not fall with depth: 20-43 bits per layer at width 32 and up to 150-172 at width 128.
   - E3(1%) was reached only where H(G) is already below about 9 bits (ARC width 32 at layers >= 8; width 64 at layer
     32). At width 256 the tree's merges still change 92-100% of answers at 8.0-8.4 bits.
   - The tree's substitutions change 92-100% of the answers they touch at width 256 (and at shallow layers of every
     width). If every merging behaved that way, Fano's inequality would give E3(1%) >= H(G) - 3 bits. That is about
     170 bits at layer 1 and 25 at layer 32 for width 256. This bound is heuristic.
   - For the mean, where errors may cancel, deep layers merge further: bias nMSE <= 1e-4 at 4.4-9.0 bits at layers 8-32
     of width 64. Not at width 256 (bias 6e-3 to 0.17 at 8 bits).
4. **The capacity bound never binds.**
   - C_l = (d/2) log2(1 + 1/sigma^2) is 159-2,214 bits, and the covariance version is 17-1,332 bits. Both exceed m in
     every net.
   - What keeps entropies low is geometry: a step-t RNN pattern depends on t scalars plus noise, and the MLPs collapse
     with depth.
   - min(m, C_l) bounds only the input-driven part I(G; X), not the noise-driven part H(G | X), which:
     - reaches 3.5 bits at early RNN steps;
     - is 40-50% of H(G) at layer 32 of ARC at sigma = 0.01;
     - dominates chaotic tanh from layer 8 on.

**What this implies for a boundary-exact family at scale (the coordinator's question; targets M1 and M3).**
- **AlgZoo (M1's AlgZoo target).** A family whose discrete part lists gate cells, with a Gaussian piece per cell, is
  affordable.
  - M_{16,10} needs 2^4-2^6 cells per step at ARC's bar.
  - The trained noisy RNNs need 2^7-2^10 at width 64.
  - Going from width 32 to 64 adds 1-2 bits.
  - So the discrete part is not the obstacle. Whether one Gaussian per cell is accurate enough is not measured here.
- **Deep random MLPs (M1's 256 x 32 target).** Listing cells is not affordable.
  - The merged entropy cannot be far below H(G) at the per-sample tolerance (heuristic Fano), and H(G) grows linearly
    with width at every depth.
  - At width 256 that means at least about 170 bits at layer 1 and about 25 at layer 32. Even 2^25 cells x d^2 is
    about 2e12 FLOPs per layer, against ARC's whole budget of 2.7e11.
  - For the mean alone, cancellations let deep layers merge further at width 64, but not at width 256 within 8 bits.
  - A boundary-exact family for this target would therefore need one of:
    - a factored discrete part. The FVSBN autoregressive-logistic model, with m^2 parameters, comes within about 22% of
      the rigorous layer-1 lower bound;
    - a non-cell treatment of the layers where the entropy is linear in width.
  - This census does not show either is possible.
- **M3 (certified rare-event bounds on trained nets).**
  - The cell counts say AlgZoo-type RNNs are candidates.
  - The PFN at the precision of its Bayes deviation is not: at least about 27 bits of relevant pattern, and no merging
    found.
  - E3 resolves typical answers (1%), not rare events.

## 1. What was measured

[draft_methods.md, unchanged]

## 2-5. Results, checks and caveats

[draft_sections.md; corrected: W64 variance directions 1-4 (step 1) and 2-4 later; cells bound 18-61 for W64; tree
stops 1.3-100% elsewhere; FVSBN-to-Gaussian-bound gap 5-48 bits (22% at width 256)]

## Scaling table (ARC He-ReLU, sigma = 0.01, depth 32)

Each cell gives H(G) [lower-upper] / E3(1%) / E3 at bias nMSE <= 1e-4 / the noise-driven H(G|X) (plug-in). nr means
not reached. A * marks the approximate zero-collision bound.

| width | layer 1 | layer 4 | layer 8 | layer 16 | layer 24 | layer 32 |
|---|---|---|---|---|---|---|
| 32 | 26-28 / nr / nr / 0 | 15-16 / nr / nr / 0.7 | 6.8 / 6.3 / 1.2 / 0.7 | 5.3 / 3.7 / 1.1 / 1.7 | 2.5 / 1.1 / 0.7 / 1.0 | 3.7 / 0 / 1.0 / 2.1 |
| 64 | 45-56 / nr / nr / 0 | 21-28 / nr / nr / 1.2 | 15-17 / nr / 8.9 / 1.6 | 12.3 / nr / 9.0 / 2.9 | 9.5 / nr / 4.4 / 3.6 | 8.8 / 7.3 / 5.6 / 4.2 |
| 128 | 87-110 / nr / nr / 0 | 19 [35*]-63 / nr / nr / 2.2 | 27-37 / nr / nr / 2.8 | 17-22 / nr / nr / 4.1 | 17-23 / nr / nr / 5.5 | 16-18 / nr / nr / 5.9 |
| 256 | 173-221 / nr / nr / 0 | 17 [31*]-130 / nr / nr / 4.1 | 17 [31*]-82 / nr / nr / 5.6 | 17 [31*]-52 / nr / nr / 6.9 | 30-43 / nr / nr / 7.5 | 28-40 / nr / nr / 7.9 |

Pending at review time: chaotic tanh at widths 64 and 256; ordered tanh; ARC width 64 at sigma = 0.1; chaotic tanh
at sigma = 0.001 and 0.1; seed-1 RNNs.

## 6. Predictions (PREREG.md) scored (so far)

- **P1.**
  - Hits: C_l >= m everywhere; the RNN upper bounds are below 0.5 m at every step.
  - Pending: the covariance bound below m for ARC width 64 at sigma 0.1 at depth >= 16 (50%).
- **P2.**
  - Hits: T16 at 4.4-6.0 bits at steps 3-10 (the full run moved the smoke values by <= 0.05); W32 at 8.5-10.3 and W64
    at 9.7-12.8 at steps >= 4.
  - Miss: H(G|X) < 1 bit per step holds for T16 but not for W32 (2.6, 2.0 and 1.5 bits at steps 2-4) or W64 (3.5-1.2
    bits at steps 2-5).
- **P3.**
  - Hit: T16 D >= 13 at steps >= 3.
  - Misses: W32 is within 50-95% only at steps 2-5 (97-100% later), and W64 lies above 90% at steps >= 4. The wide nets
    are not sparser in this sense.
- **P4.**
  - Hits: T16 E3(1%) <= H(G_R) - 1.4 at every step >= 3, and <= 4.1. E3(0.2%) was not reached at some steps of W32 and
    W64.
  - Partly unmeasured: E3 <= 8 bits for W32/W64 holds where it was reached (<= 7.4); steps 9-10 (W32) and 7-10 (W64)
    were not reached.
- **P5.** Hit on all three: k(99% var) <= 7 / 9 / 4 for T16 / W32 / W64, and k(agree) is larger.
- **P6.**
  - Hits: the layer-1 upper bound is >= 0.6 m (0.86-0.87 m); the layer-32 upper bound is <= 0.6 of layer 1's (0.16-0.18
    at widths 64-256); D >= 0.9 m at width 64 and >= 0.5 m at width 256 (D = m); E3(1%) was not reached at layers <= 16
    at width 64.
  - Miss: H(G|X) at layer 32 < 0.5 bit at sigma 0.001 (measured 0.59).
  - Pending: H(G|X) at sigma 0.1.
- **P7.**
  - Pending: the ordered regime. The layer-1 bound already gives 8.5 bits at width 64, so "< 1 bit" will miss at layer
    1.
  - Hits so far: chaotic tanh's upper bound is >= 0.5 x 1.58 d at every layer (widths 32 and 128), and E3(1%) was not
    reached at layers <= 16.
- **P8.**
  - Hits: the layer-1 lower bound is >= 20 (approximate collision bound 27; plug-in 15); FVSBN gives 199, inside
    100-600; layer 2 at 11.6-16.0 is inside 5-30; D = 243 is inside 30-600; D = 52 is inside 20-64.
  - Miss: E3(1%) <= 8 bits for layer 2 (60%) was not reached.
- **Amendment 1.** Hits: E3(1%) on T16 was unchanged; E3(0.2%) was reached at every T16 step and is <= H(G) - 0.3 at 6
  of 9 steps.
