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


**Gates.** A gate records which side of a kink a unit is on.
- ReLU: g = 1[z > 0], where z is the pre-activation.
- tanh (the saturation region): the two kinks of hardtanh(z) = z - ReLU(z - 1) + ReLU(-z - 1), up = 1[z > 1] and
  dn = 1[z < -1]. That gives three states per neuron.
- GELU: g = 1[u > 0]. This is the middle of the switching region, where GELU(u) = u Phi(u) has Phi(u) = 1/2.

**Flip.** A flip swaps the linear piece a gate selects. The pre-activation is held fixed.
- ReLU: a = g z, so a flip gives a = (1 - g) z.
- tanh: the hardtanh piece change is added to tanh(z).
- GELU: u Phi(u) becomes u Phi(-u). Far from 0 this equals the ReLU flip; at u = 0 it is 0.

Everything downstream is recomputed by the actual network, with the same input and the same noise draws (common random
numbers). These flips are what a family that assigns the wrong piece to a unit would get wrong.

**Question and tolerance, per sample.**
- AlgZoo: the answer is "is the output the 2nd argmax?". It changes if it flips.
- Random MLPs: the answer is the output vector, a_L for ARC's nets and W_out s_L for the tanh nets. It changes if
  ||delta|| > 0.1 sqrt(sum_j Var y_j). We also report the bias of the mean in ARC's nMSE units.
- PFN: the answer is the predictive (mixture) mean of the query. It changes if it moves by more than 10% of the net's
  RMS deviation from exact Bayes (0.063, so a tolerance of 0.0063 against a predictive-mean SD of 0.92). Since Bayes is
  fixed per episode, this is also the question "the deviation from Bayes, to 10%".

**The three entropies (bits; per layer, or per step for the RNNs; over inputs and noise).**
- E1 = H(G_l): all m gates.
- E2 = H(G_R). Let f_i be the fraction of held-out samples whose answer changes when gate i alone is flipped. Then
  R = {i : f_i >= 1%} and D = |R|. D(0.1%) is also reported.
- E3 = the merged entropy, built as follows.
  - A greedy decision tree over gate values is grown on a flip set of 1,024-8,192 samples.
  - Each leaf is one merged class. It replaces the whole gate pattern of its members by the leaf's majority pattern.
  - The tree splits first to remove samples whose single-flip "care" gates disagree with the leaf pattern. When none
    are left, it splits on flip-rate-weighted mismatches (Amendment 1).
  - Every checkpoint is verified on 8,192-65,536 held-out samples by running the network with the substituted
    pattern.
  - E3(eps) is the held-out entropy of the leaves at the first checkpoint whose verified change rate is <= eps, with
    eps = 1% (and 0.2% for the RNNs, ARC's bar).
  - E3 is an achieved size. It is an upper bound on the discrete part this kind of family needs, not a lower bound.
  - "Not reached" means the tree ran out of samples or budget first. The trivial answer, no merging (E3 = H(G) at
    zero change), always exists.

**Estimators** (each checked on synthetic patterns and on T16 against a 2^21-sample reference):
- Lower bounds:
  - plug-in (biased low for every distribution);
  - Renyi-2 entropy from pair collisions (H >= H_2). With zero collisions this becomes log2(pairs/3), marked *;
    approximate, because pairs are not independent;
  - for layer 1 of the random MLPs, where z is exactly Gaussian, a rigorous chain-rule bound,
    sum_i H(g_i | z_{<i}), computed from the Cholesky factor.
- Upper bounds:
  - the sum of per-unit entropies;
  - the held-out cross-entropy + 2 SE of an autoregressive FVSBN, and of a MADE for m <= 64.
- NSB and Miller-Madow are quoted as point estimates only when the bracket is narrower than 1 bit.

**Subspace ("rotate and concentrate").**
- We ran PCA on the relevant gates' pre-activations over the data.
- k(99% var) is the number of components that hold 99% of the variance.
- k(agree 99%) is the number of components after which the reconstructed pre-activations reproduce 99% of the gate
  values.
- log2 sum_{j<=k} C(D, j) is the largest entropy D hyperplanes in general position can produce in k dimensions.

**Capacity.**
- C_l = (d/2) log2(1 + 1/sigma^2), with noise relative to the unit activation scale.
- C_cov = (1/2) log2 det(I + Cov(signal)/noise variance), computed on the noisy state feeding layer l.
- Both bound I(G_l; X) for layers >= 2. Neither bounds the noise-driven part H(G_l | X), which we measured on 128-512
  inputs x 256 noise draws.


## 2-5. Results, checks and caveats

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
  - The relevant gates' pre-activations put 99% of their variance in 1-7 (T16), 1-9 (W32) and 1-4 (W64) directions.
  - Reproducing 99% of their gate values takes 3-12, 4-20 and 4-29 directions.
  - That cell-count bound (log2 sum_{j<=k} C(D, j)) is 9-16 (T16), 13-32 (W32) and 18-61 (W64) at steps 2-10. That is far above the
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
- **Merging.** E3(1%) was not reached in either layer (Amendment-1 tree).
  - Layer 1: 90% of predictive means still move by more than 0.0063 at 8.4 bits of leaves.
  - Layer 2: 18% at 8.3 bits.
  - The population-mean bias at those points is 0.002 (layer 1) and 0.001 (layer 2). The average predictive mean
    survives merging; the per-episode deviation from Bayes does not.

## 4. Random MLPs (question: the output mean)

- **Nets.**
  - ARC's He-ReLU family: no bias, no LayerNorm, noise after every ReLU at sigma x RMS.
  - Phase R's tanh + LayerNorm nets with bias N(0, 0.25): ordered g = 0.25 and chaotic g = 3.
  - All have depth 32 and weight seed 0.
- **Per-sample tolerance.** 10% of the output's spread, sqrt(sum_j Var y_j). We also give the mean's squared bias in
  ARC's nMSE units.
- **Sample sizes.** Widths 32 and 64: 2^19 entropy samples, 4,096 tree samples, 32,768 held-out. Width >= 128 (tanh) or
  256: 2^17, 1,024 and 8,192.

(see the scaling table below)

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
  - Everywhere else the tree stops with 1.3-100% of answers changed (above 12% except at layer 24 of width 64) at 8-10 bits.
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
