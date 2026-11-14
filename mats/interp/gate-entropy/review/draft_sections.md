## 2. AlgZoo RNNs (question: "is the output the 2nd argmax?")

- **T16:** ARC's example M_{16,10}, fine-tuned at sigma = 0.001 (`work/algzoo-noisy/runs/ft_16_10_ex_s0.001.npz`).
- **W32, W64:** trained from scratch at sigma = 0.01 (`runs/gpu/w{32,64}_s0_sig0.01.npz`, seed 0). Both pass Kelvin's
  criterion in the AlgZoo report.
- **Sample sizes:** 2^21 entropy samples, 8,192 flip/tree samples, 65,536 held-out samples per net.
- **Tables:** `results/report_tables.md` gives the per-step tables.

Per step:

| step | T16 H(G) | T16 E3 1% / 0.2% | W32 H(G) | W32 E3 1% / 0.2% | W64 H(G) | W64 E3 1% / 0.2% |
|---|---|---|---|---|---|---|
| 1 | 1.0 | 1.0 / 1.0 | 1.0 | 1.0 / 1.0 | 1.0 | 1.0 / 1.0 |
| 2 | 4.1 | 2.4 / 2.4 | 6.8 | 2.4 / 2.4 | 8.9 | 2.5 / 3.6 |
| 3 | 4.7 | 2.8 / 3.7 | 7.7 | 3.1 / 4.3 | 9.8 | 4.9 / 8.0 |
| 4 | 5.3 | 3.6 / 4.7 | 8.5 | 4.1 / 6.6 | 9.7 | 6.1 / nr |
| 6 | 6.0 | 4.0 / 5.8 | 9.2 | 5.8 / nr | 10.6 | 7.4 / nr |
| 8 | 5.3 | 3.5 / 4.9 | 9.4 | 7.2 / nr | 10.9-12.2 | nr / nr |
| 10 | 4.4 | 2.3 / 3.6 | 9.5 | nr / nr | 11.4-12.8 | nr / nr |

H(G) is the Miller-Madow estimate where the bracket is narrower than 1 bit, otherwise [lower, upper]. nr means not
reached.

What the per-step data show:

- **Where E3 was not reached.** At the late steps of W32 and W64 the tree used up its 8,192 training samples. It
  stopped at 1.0-2.5% of answers changed, at 8.5-9.6 bits. The true E3(1%) at those steps therefore lies between "not
  measured" and H(G) (9.5-12.8).
- **Every gate matters.** D(1%) = 16 of 16 from step 4 on (T16), 29-32 of 32 (W32) and 61-64 of 64 (W64). The wide nets
  are not sparser in this sense. Prediction P3 failed.
- **Merging works because most substitutions are harmless.** At W64 step 10 the final tree changes at least one gate on
  33% of samples, but only 7.7% of those answers change. The flip test is a strong intervention (it moves a unit by
  |z|), so almost every gate is "relevant". Merging only substitutes within a class, where the minority values are
  mostly near the threshold.
- **Noise-driven entropy H(G | X).**
  - At most 0.67 bit per step in T16.
  - 2.6-3.5 bits at step 2 of W32 and W64, falling to 0.6-0.9 by step 10. The early-step part comes from units whose
    pre-activation is mostly noise (dead after step 1). These are irrelevant: H(G_R) < H(G) at those steps.
- **Rotate and concentrate.**
  - The relevant gates' pre-activations put 99% of their variance in 1-7 (T16), 1-9 (W32) and 2-4 (W64) directions.
  - Reproducing 99% of their gate values takes 3-12, 4-20 and 4-29 directions.
  - That cell-count bound (log2 sum_{j<=k} C(D, j)) is 9-16 (T16), 13-32 (W32) and 18-60 (W64). That is far above the
    measured H(G), so the data use only a small fraction of the cells the geometry allows.
- **Capacity.** C_l = 159 (T16), 213 (W32) and 425 (W64) bits per step, against m = 16, 32 and 64. The covariance
  bound is 17-325 bits. Neither binds.

## 3. Phase B PFN (sigma = 0.05; the predictive mean to 10% of its deviation from Bayes)

- **Net:** `work/cm-interp-B/runs/L2_s0.05` (2 blocks, width 32, 16 tokens, 64 GELU units per token).
- **Sample sizes:** 2^15 entropy episodes, 1,024 flip/tree episodes, 8,192 held-out episodes.
- **Tolerance:** the net's single-draw RMS deviation from exact Bayes is 0.063 (predictive-mean SD 0.92), so the
  tolerance is 0.0063.
- **Layer 1.**
  - All 1,024 gates: H(G) between 15 (plug-in, all 2^15 patterns distinct) [27* collision bound] and 199 (FVSBN).
  - D(1%) = 243 and D(0.1%) = 330. They concentrate on the tokens that carry each source's reading of the query
    question (51 of 64 units on each of the 3 tokens) and the query token (42). The 12 labelled-question tokens have
    3-5 relevant units each.
  - H(G_R) lies between 15 [27*] and 61.
  - The relevant pre-activations need 30 directions for 99% of their variance and 40 for 99% of gate values.
- **Layer 2.** All 1,024 gates: H(G) between 15 [27*] and 188. Only the query token's units can reach the output (a
  code check confirmed that the other tokens' layer-2 flips change nothing). D = 52 of its 64. H(G_R) = 11.6-16.0, in
  5 / 7 directions.
- **Merging.** [After the rerun with the Amendment-1 tree:] E3(1%) was not reached in either layer.
  - Layer 1: 88% of predictive means still move by more than 0.0063 at 7.6 bits of leaves (first tree).
  - Layer 2: 19% at 5.5 bits (first tree).
  - At this precision the PFN's gate patterns behave as unique per episode.

## 4. Random MLPs (question: the output mean)

- **Nets.**
  - ARC's He-ReLU family: no bias, no LayerNorm, noise after every ReLU at sigma x RMS.
  - Phase R's tanh + LayerNorm nets with bias N(0, 0.25): ordered g = 0.25 and chaotic g = 3.
  - All have depth 32 and weight seed 0.
- **Per-sample tolerance.** 10% of the output's spread, sqrt(sum_j Var y_j). We also give the mean's squared bias in
  ARC's nMSE units.
- **Sample sizes.** Widths 32 and 64: 2^19 entropy samples, 4,096 tree samples, 32,768 held-out. Width >= 128 (tanh) or
  256: 2^17, 1,024 and 8,192.

[scaling table here]

What the scaling data show:
- **Every gate is relevant at every layer.** D(1%) = m everywhere. The median single flip moves the output by more than
  10% of its spread on 58-98% of inputs (ARC) and 77-100% (chaotic tanh).
- **H(G) grows linearly with width at every depth.**
  - Layer 1 (rigorous bound): at least 0.68-0.70 m for ARC (22, 45, 87, 173 bits at widths 32-256) and 0.70 x
    log2(3) m for chaotic tanh.
  - ARC deeper down: H(G) falls with depth (collapse). At sigma = 0.01 it is 3.7, 8.8 and 15.6-18.4 bits at layer 32
    for widths 32, 64 and 128, i.e. 0.12-0.14 bit per gate.
  - Chaotic tanh does not fall: 150-172 bits at every layer of width 128 (upper bound).
- **Noise-driven part.**
  - ARC at sigma = 0.01: it grows with depth to 4.2-5.3 of 8.8 bits (width 64) and 5.9-8.3 of 15.6-18.4 bits (width
    128) at layer 32.
  - Chaotic tanh at width 128: the noise alone randomises most of the pattern by layer 8. Per-input plug-in hits its
    8-bit cap; the per-unit sum is 28-188 bits.
  - The capacity bound does not touch this part.
- **Merging at the 1% per-sample tolerance.**
  - It works only where H(G) is already small: ARC width 32 at layers 8-32 (E3 = 6.3, 3.7, 1.1, 0.0) and width 64 at
    layer 32 (7.3).
  - Everywhere else the tree's substitutions change 35-100% of answers at about 10 bits (8 for chaotic tanh).
  - At shallow layers the tree's substitutions change 81-100% of the answers they touch. If every merging behaved like
    that (rho >= 0.8), Fano's inequality would give E3(1%) >= H(G) - h2(eps/rho) - (eps/rho) m. That is H(G) minus at
    most 1.0 bit at width 64 and 3.3 bits at width 256. This part is heuristic, because a cleverer merging could
    substitute only where it is harmless.
- **Merging for the mean (cancellations allowed).**
  - Bias nMSE <= 1e-4 is reached at 4.4-9.0 bits at layers 8-32 of ARC width 64, and <= 1e-5 at 7.4-7.9 bits at
    layers 24-32.
  - At layers 1-4 the bias stays above 1e-3 at 10 bits (width 64). At width 128 not even 1e-4 is reached at layers
    >= 16 within 10 bits.
  - For chaotic tanh a single class meets 1e-3 at shallow layers (the substitution errors average out), but not 1e-4.

## 5. Estimator checks

- **T16 (16 gates per step; reference = plug-in on 2^21 samples, 2-842 cells, bias < 0.02).** Biases against the
  reference on disjoint subsamples (mean of 3; `results/exact_check_T16.json`):
  - NSB: within 0.07 bit at every n from 2^10 to 2^16.
  - Plug-in: -0.2 at n = 2^10, < 0.04 at n >= 2^14.
  - Renyi-2: 0.8-1.8 below, a valid but loose lower bound.
  - FVSBN: +0.24-0.50 at n = 2^16; MADE: +0.01-0.10 at n = 2^16.
  - At n = 2^12 both are 2-6 bits too high, so upper bounds from small samples are loose.
- **Synthetic checks.**
  - Independent bits with p = 0.5 and m = 64: FVSBN/MADE are exact (64.0). The approximate zero-collision bound gives
    21-29.
  - Gaussian orthant patterns with m = 16: MADE 11.33-11.41 against the true 11.41 at n = 2^14-2^16.
- **Layer-1 Gaussian chain-rule bound.** Rigorous, with Monte Carlo SE 0.02-0.09 bit. It is 3-23 bits below the FVSBN
  upper bound at widths 32-128.

## 6. Predictions (PREREG.md) scored

[to fill]

## 7. Caveats

- **E3 is an achieved, verified size.**
  - It bounds from above what this kind of family (classes over gate values, one shared pattern per class) needs, at
    this sample size.
  - It is not a lower bound. "Not reached" means the tree ran out of samples (8,192, 4,096 or 1,024) or split budget
    first.
  - The Fano-type lower bound is heuristic, because it assumes every merging's substitutions change answers as often
    as ours do.
- **The flip is the linear-extension flip.** It moves a unit by |z|, so far-from-threshold flips are large and almost
  every gate counts as relevant. E3 does not have this problem, because substitutions happen only within a class.
- **Tolerances are our choices.**
  - The 10% per-sample tolerance for the MLP output mean is strict compared with what the mean needs; the bias columns
    show how much cancellation buys.
  - The PFN tolerance (10% of the Bayes deviation) is what the deviation question needs.
- **The entropy is the number of distinguishable cells, not the description length.** A factored or autoregressive
  discrete part (FVSBN-like, m^2 parameters) can carry a 150-bit gate distribution. Whether such a part can be pushed
  through the next layer without listing cells is exactly what a family would have to solve.
- **Coverage.** One weight seed per random-MLP configuration. Width 256 only at sigma = 0.01. Seed-1 RNN checks
  [done / not run]. The RNN width comparison 16 -> 32 mixes sigma (0.001 vs 0.01) and training (fine-tune vs from
  scratch).
- **No cost model.** Tractability is argued from cell counts, not from a full cost model of a family. Each cell
  carries at least a d-dimensional Gaussian (d^2 per layer to propagate), so 2^H cells cost at least about 2^H d^2.
