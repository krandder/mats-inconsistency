# How big must the discrete part be? Gate entropies of noisy networks (29 Oct 2026)

- **Where everything is.** Work directory `work/gate-entropy/`: code, logs and `results/*.json`. Per-network tables are
  in `results/report_tables.md` and width-by-depth tables in `results/scaling.md`.
- **Pre-registration.** `PREREG.md` was hashed before the main runs. Three dated amendments, with hashes and times, are
  in `prereg_hashes.txt`.
- **Brief.** From main, relaying Kelvin: replace Gaussians by families that put mass correctly on each side of the
  decision boundaries. This census measures how large the discrete part of such a family must be. It builds no
  family.
- **Review.** One GPT-6 review was done before the claims (`review/review1.md`, twelve findings). The fixes are listed
  in section 7.

## Summary

For every layer (every step of the RNNs) we measured three entropies over inputs and noise:
- **H(G):** the gate pattern, i.e. which side of each kink every unit is on.
- **H(G_R):** the same, restricted to the D gates whose flip changes the question's answer on at least 1% of held-out
  samples.
- **E3:** the merged pattern. A decision tree over gate values groups patterns into classes, and each class shares one
  pattern. E3 is the coarsest tree for which at most 1% (or 0.2%) of answers change, measured by running the real
  network. It was re-checked on fresh samples.

How to read the numbers:
- E3 is a size we achieved. It is an upper estimate of what this kind of discrete part needs, not a lower bound.
- 2^E3 is an effective number of classes. The trees use more leaves than that, and we give both.
- Entropy brackets are estimated bounds, not guaranteed ones.
- Units are bits.

1. **AlgZoo 2nd-argmax RNNs: the discrete part is small, and merging saves 1.4-6.4 bits per step.**
   - T16 (ARC's example M_{16,10} fine-tuned at sigma = 0.001; 16 gates per step):
     - H(G_t) = 4.1-6.0 at steps 2-10. This is plug-in on 2^21 samples, within 0.02 of the reference.
     - All 16 gates are relevant from step 4 on.
     - E3 = 2.3-4.1 at 1% answer change (steps 2-10). On fresh samples the change is 0.14-0.85%, with 9-49 leaves.
     - E3 = 2.4-5.8 at 0.2%, ARC's bar. Fresh change is 0.08-0.20%, with 95% upper limits up to 0.23% and 9-193 leaves.
   - W32 and W64 (trained from scratch at sigma = 0.01):
     - H(G_t) = 6.8-9.6 (W32) and 8.9 to 11.4-12.8 (W64) at steps 2-10.
     - E3(1%) = 2.4-7.2 at steps 2-8 (W32) and 2.5-7.4 at steps 2-6 (W64). Fresh change is up to 1.0%, with 7-513
       leaves.
     - At later steps the tree ran out of its 8,192 training samples. It stopped with 1.0-2.5% of answers changed, at
       8.5-9.6 bits.
   - At steps 3-6, E3(1%) is 1.6-2.0 bits higher at width 64 than at width 32 (same sigma, same training).
2. **Phase B PFN (sigma = 0.05): no merging found.**
   - The question is the predictive mean, to 10% of its RMS deviation from Bayes (tolerance 0.0063).
   - Layer 1 has 243 relevant GELU gates out of 1,024, almost all on the three tokens that carry the sources' readings
     of the query, and on the query token.
   - Their joint pattern never repeats in 2^15 episodes. H(G_R) lies between 15 (plug-in; the approximate collision
     bound gives 27) and 61 (FVSBN).
   - The tree did not reach 1% in either layer:
     - layer 1: 90% of answers still change at 8.4 bits;
     - layer 2 (52 relevant units, all on the query token): 18% at 8.3 bits.
   - The population-average predictive mean moves by only 0.001-0.002. It is the per-episode deviation from Bayes that
     resists merging.
3. **Random depth-32 MLPs: H(G) grows in proportion to width at every depth, and no merging was found once width >=
   128.**
   - The question is the output mean. A sample counts as changed if its output moves by more than 10% of the output's
     spread.
   - All gates are relevant (D = m) at every layer of ARC's He-ReLU nets and of the chaotic tanh nets.
   - ARC at sigma = 0.01, H(G) by width 32 / 64 / 128 / 256:
     - layer 1: 26-28 / 45-56 / 87-110 / 173-221 (0.7-0.87 bit per gate). The lower ends are a Gaussian chain-rule
       inequality evaluated by Monte Carlo.
     - layer 32: 3.7 / 8.8 / 16-18 / 28-40 (0.11-0.15 bit per gate).
   - Chaotic tanh stays high at every depth: upper estimates are 73-86 bits at width 64 and 317-367 at width 256.
   - Ordered tanh at width 64 has 12 bits at layer 1, but no gate in layers 1-16 is relevant. E3 = 0 below layer 32.
   - ARC's E3(1%) was reached only where H(G) is already below about 9 bits: width 32 at layers 8-32, and width 64 at
     layer 32 (7.4 bits, fresh change 0.84%).
     - At width 256 the tree still changed 92-100% of answers at 8.0-8.4 bits.
     - 92-100% of the samples the merges touched changed their answer. If every possible merging behaved like that,
       Fano's inequality would put E3 within about 3 bits of H(G): about 170 bits at layer 1 and 25 at layer 32 for
       width 256.
     - This is a conditional illustration, not a bound.
4. **Capacity.**
   - Kelvin's C_l = (d/2) log2(1 + 1/sigma^2) is 159-2,214 bits, above m in every net.
   - The covariance version falls below m in three places, and stays above the measured entropy in all of them:
     - step 2 of W32 and W64 (21.6 and 23.5 bits, including the fresh-input allowance; measured H(G) 6.8 and 8.9);
     - layers >= 8 of ordered tanh (8-17 bits; measured H(G) <= 0.9);
     - layers 24-32 of ARC width 64 at sigma = 0.1 (57-63 bits; measured H(G) 16-22).
   - What keeps the RNN entropies low is geometry: a step-t pattern depends on t scalars plus noise.
   - C_l bounds only the input-driven part I(G; X), not the noise-driven part H(G | X). That part is:
     - 0.6-3.5 bits at step 2 of the RNNs;
     - 4.2-5.3 of 8.8 bits at layer 32 of ARC width 64;
     - unresolved for chaotic tanh, somewhere between 8 bits (capped) and a 28-188 bit per-unit sum.

**What this implies for a boundary-exact family at scale (the coordinator's question; M1 and M3).**
- **AlgZoo.** With the continuous state left exact, the RNNs' answer needs only a few bits of gate information per
  step:
  - 2.3-5.8 bits for M_{16,10}, up to ARC's bar;
  - up to 7.4 bits at width 64 at steps 2-6; at steps 7-10 it was not reached (the tree stopped at 9.4-9.6 bits with
    1.1-2.5% of answers changed, whereas keeping every pattern costs 11-13 bits);
  - about 2 bits more per doubling of width, measured once (32 to 64).

  So the discrete part is not what blocks M1's AlgZoo target. This census does not show that a family with one
  Gaussian per class would be accurate: E3 keeps the network's exact pre-activations and noise, and changes one layer
  at a time.
- **Deep random MLPs (M1's 256 x 32 target).** Measured:
  - H(G) grows in proportion to width at every depth;
  - D = m everywhere;
  - our tree found no merging at width >= 128 within 8-10 bits. This holds even for the mean alone: at width 256 the
    bias nMSE stays >= 6e-3 at 8 bits.

  If nothing merges much better, as the conditional Fano illustration suggests, a family that lists classes needs
  about 2^25 of them at layer 32 and about 2^170 at layer 1 for width 256. Even 2^25 classes x 256^2 is about 2e12
  FLOPs per layer, against ARC's whole budget of 2.7e11. A boundary-exact family for this target would then need a
  factored discrete part, not a list. An autoregressive-logistic model with m^2 parameters already describes the
  layer-1 gate distribution to within 24-28% of the lower estimate. Whether such a factored part can be propagated
  layer to layer is open, and this census does not settle it.
- **M3.**
  - From the discrete side, the AlgZoo RNNs look like candidates.
  - The PFN, at the precision of its Bayes deviation, does not: about 15-61 bits of relevant pattern, and no merging
    found.
  - None of our tolerances (1% or 0.2% of answers) says anything about rare-event accuracy.

## 1. What was measured

**Gates.**
- ReLU: g = 1[z > 0].
- tanh (the saturation region): the two kinks of hardtanh(z) = z - ReLU(z - 1) + ReLU(-z - 1), up = 1[z > 1] and
  dn = 1[z < -1], so three states per neuron.
- GELU: g = 1[u > 0], the middle of the switching region where GELU(u) = u Phi(u) has Phi = 1/2.

**Flip (the registered intervention).** The linear piece is swapped while the pre-activation is held:
- ReLU: a = g z becomes (1 - g) z.
- tanh: the hardtanh piece change is added to tanh(z). A single dn-flip on a unit with z > 1 asks for the impossible
  "both saturated" state; we kept it as defined.
- GELU: u Phi(u) becomes u Phi(-u). This equals the ReLU flip far from 0 and is 0 at u = 0.

Everything downstream is recomputed by the actual network with the same input and noise draws. A code check confirms
that the actual pattern reproduces the forward pass. D therefore measures sensitivity to this intervention, which can
move a unit by |z| far from its threshold. It is not a measure of boundary proximity.

**Answer and "changed", per sample.**
- AlgZoo: is the output the 2nd argmax? It changes if it flips.
- Random MLPs: the output vector (ARC: a_L; tanh: W_out s_L). It changes if ||delta|| > 0.1 sqrt(sum_j Var y_j).
- PFN: the predictive (mixture) mean. It changes if it moves by more than 0.0063, i.e. 10% of the net's RMS deviation
  from exact Bayes (0.063). The predictive-mean SD is 0.92.

These per-sample tolerances certify neither the error of a mean nor that of a rare event. For the MLPs we also report
the squared bias of the mean, in ARC's nMSE units.

**E2.** f_i is the fraction of held-out samples whose answer changes when gate i alone is flipped. Then D(tau) =
#{i : f_i >= tau} for tau = 1% and 0.1%, and H(G_R) is computed on those gates.

**E3.**
- A greedy decision tree over gate values is grown on a flip set of 1,024-8,192 samples. Each leaf replaces its
  members' whole gate pattern by the leaf's majority pattern.
- Phase 1 splits to remove samples whose single-flip "care" gates disagree with that pattern. Phase 2 (Amendment 1)
  splits on flip-rate-weighted mismatches once no hard split improves.
- Checkpoints are scored on a held-out set by running the network with the substituted patterns. E3(eps) is the leaf
  entropy at the first checkpoint whose change rate is <= eps.
- Because that set also chose the checkpoint, every reached E3 was re-evaluated with its tree frozen, on fresh samples
  (Amendment 3; 2^16 for the RNNs, 2^15 for the MLPs). We report:
  - the fresh Miller-Madow leaf entropy;
  - the fresh change rate, with a one-sided 95% Clopper-Pearson limit;
  - the number of leaves.
- "Not reached" means the tree ran out of samples or budget. It is not evidence that no merging exists.

**Entropy estimators.**
- Lower, in expectation:
  - plug-in;
  - Renyi-2 from pair collisions (H >= H_2). With zero collisions, log2(pairs/3), marked *; this is approximate,
    because pairs are not independent;
  - for layer 1 of the random MLPs, where z is exactly Gaussian, the exact inequality H(G) >= sum_i H(g_i | z_{<i}).
    It is evaluated by Monte Carlo with SE 0.01-0.09 bit.
- Upper, in expectation: the sum of per-unit entropies; the held-out cross-entropy + 2 SE of an FVSBN (masked
  logistic), and of a MADE for m <= 64.
- No simultaneous coverage, so the brackets are estimates.
- NSB and Miller-Madow are quoted only when the bracket is narrower than 1 bit. (The NSB alphabet for tanh, 2^(2d)
  rather than 3^d, never enters a quoted number.)

**Subspace.** PCA of the relevant gates' pre-activations on the data.
- k(var99): components for 99% of the variance.
- k(agree99): components after which the reconstructed pre-activations reproduce 99% of the gate values.

**Capacity.**
- C_l = (d/2) log2(1 + 1/sigma^2).
- The covariance bound (1/2) log2 det(I + Cov(signal)/noise variance), on the noisy state feeding the layer. For the
  RNNs we add log2(d+1) for the noiseless fresh input.
- These bound I(G_l; X), not H(G_l). For the LayerNorm tanh nets the exact constant would use the centred
  (d-1)-dimensional power d/(d-1). This changes no conclusion.

**Noise-driven part H(G | X).** 128-512 inputs x 256 noise draws each. We report the per-input plug-in (capped at 8
bits) and the per-unit sum.

**Networks and sample sizes.**

| net | entropy samples | tree samples | held-out |
|---|---|---|---|
| AlgZoo T16, W32, W64 (seed 0) | 2^21 | 8,192 | 65,536 |
| PFN `cm-interp-B/runs/L2_s0.05` | 2^15 | 1,024 | 8,192 |
| ARC He-ReLU, width <= 128 | 2^19 | 4,096 | 32,768 |
| tanh + LN, width <= 64 | 2^19 | 4,096 | 32,768 |
| ARC width 256; tanh width >= 128 | 2^17 | 1,024 | 8,192 |

- ARC's nets have no bias and no LayerNorm. Phase R's tanh + LayerNorm nets have bias N(0, 0.25).
- All random MLPs have depth 32 and weight seed 0.

## 2. AlgZoo RNNs

Each E3 cell gives the entropy, then in brackets the fresh change rate and the number of leaves. "nr" means not
reached; its bracket gives the last change rate and leaf entropy. H(G|X) is the per-input plug-in. The last column is
k(var99) / k(agree99).

**T16** (sigma = 0.001; held-out accuracy 0.958)

| step | H(G) | H(G|X) | D(1%) | E3(1%) | E3(0.2%) | k var / agree |
|---|---|---|---|---|---|---|
| 1 | 1.0 | 0.00 | 5 | 1.0 (0.00%, 2) | 1.0 (0.00%, 2) | 1 / 1 |
| 2 | 4.1 | 0.62 | 13 | 2.4 (0.14%, 9) | 2.4 (0.14%, 9) | 3 / 3 |
| 3 | 4.7 | 0.21 | 14 | 2.8 (0.55%, 13) | 3.7 (0.08%, 25) | 3 / 5 |
| 4 | 5.3 | 0.16 | 16 | 3.6 (0.77%, 25) | 4.7 (0.16%, 49) | 5 / 8 |
| 5 | 5.7 | 0.15 | 16 | 4.1 (0.78%, 33) | 4.9 (0.20%, 65) | 5 / 10 |
| 6 | 6.0 | 0.13 | 16 | 4.0 (0.85%, 49) | 5.8 (0.08%, 193) | 6 / 11 |
| 7 | 5.8 | 0.12 | 16 | 3.8 (0.76%, 49) | 5.7 (0.13%, 193) | 6 / 11 |
| 8 | 5.3 | 0.08 | 16 | 3.4 (0.72%, 49) | 4.9 (0.18%, 129) | 6 / 11 |
| 9 | 4.8 | 0.08 | 16 | 3.4 (0.68%, 49) | 4.7 (0.08%, 175) | 6 / 12 |
| 10 | 4.4 | 0.07 | 16 | 2.3 (0.77%, 25) | 3.7 (0.20%, 65) | 7 / 12 |

**W32** (sigma = 0.01; accuracy 0.969)

| step | H(G) | H(G|X) | D(1%) | E3(1%) | E3(0.2%) | k var / agree |
|---|---|---|---|---|---|---|
| 2 | 6.8 | 2.58 | 23 | 2.4 (0.07%, 7) | 2.4 (0.07%, 7) | 1 / 4 |
| 3 | 7.7 | 1.99 | 26 | 3.1 (0.87%, 17) | 4.3 (0.18%, 49) | 3 / 9 |
| 4 | 8.5 | 1.48 | 29 | 4.1 (0.87%, 49) | 6.6 (0.15%, 385) | 5 / 13 |
| 5 | 8.9 | 0.97 | 30 | 5.1 (0.91%, 97) | nr (0.3%, 8.2) | 5 / 16 |
| 6 | 9.2 | 0.87 | 31 | 5.8 (0.85%, 129) | nr (0.3%, 8.4) | 6 / 17 |
| 7 | 9.6 | 0.79 | 31 | 7.1 (1.00%, 385) | nr (0.7%, 8.6) | 7 / 19 |
| 8 | 9.4 | 0.66 | 32 | 7.2 (1.01%, 513) | nr (0.7%, 8.4) | 8 / 19 |
| 9 | 9.6 | 0.70 | 32 | nr (1.0%, 8.6) | nr | 8 / 20 |
| 10 | 9.5 | 0.62 | 32 | nr (1.3%, 8.5) | nr | 9 / 20 |

**W64** (sigma = 0.01; accuracy 0.980)

| step | H(G) | H(G|X) | D(1%) | E3(1%) | E3(0.2%) | k var / agree |
|---|---|---|---|---|---|---|
| 2 | 8.9 | 3.48 | 50 | 2.5 (0.56%, 9) | 3.6 (0.17%, 49) | 2 / 4 |
| 3 | 9.8 | 2.44 | 57 | 4.9 (0.87%, 65) | 8.0 (0.13%, 1,025) | 3 / 13 |
| 4 | 9.7 | 1.38 | 61 | 6.1 (0.57%, 193) | nr (0.23%, 8.5) | 3 / 16 |
| 5 | 10.4 | 1.17 | 62 | 7.0 (0.79%, 385) | nr (0.4%, 9.0) | 3 / 22 |
| 6 | 10.6 | 0.94 | 63 | 7.4 (0.97%, 513) | nr (0.7%, 9.2) | 3 / 24 |
| 7 | 10.9-11.9 | 0.93 | 63 | nr (1.1%, 9.4) | nr | 4 / 25 |
| 8 | 10.9-12.2 | 0.89 | 64 | nr (1.2%, 9.4) | nr | 4 / 27 |
| 9 | 11.1-12.3 | 0.83 | 63 | nr (1.8%, 9.5) | nr | 4 / 27 |
| 10 | 11.4-12.8 | 0.85 | 62 | nr (2.5%, 9.6) | nr | 4 / 29 |

At step 1 of W32 and W64, H(G) = 1.0 and E3 = 1.0 (the pattern is the sign of x_1).

**Seed check (seed 1, sigma = 0.01).** E3 here was selected on held-out data and not re-checked on fresh samples.
- W32 seed 1 (accuracy 0.958):
  - H(G_t) = 6.7-10.8 at steps 2-10 (seed 0: 6.8-9.6);
  - D(1%) = 20-32;
  - E3(1%) = 2.4-9.4 at steps 2-8; not reached at steps 9-10 (1.4-1.5% at 9.4-9.6 bits).
- W64 seed 1 (accuracy 0.978):
  - H(G_t) = 9.3 to 12.5-13.9 at steps 2-10 (seed 0: 8.9 to 11.4-12.8);
  - D(1%) = 46-63;
  - E3(1%) = 2.7-8.1 at steps 2-7; not reached at steps 8-10 (1.2-2.3% at 9.9-10.3 bits).
- The picture holds across seeds, with seed 1 running 0.5-1 bit higher at late steps.

**Merging works because most substitutions are harmless.**
- At W64 step 10 the last tree changes at least one gate on 33% of samples, but only 7.7% of those answers change.
- Almost every gate is "relevant" (f_i is high) because the flip moves a unit by |z|.
- Within a class, the substituted minority values are mostly harmless.

**Noise-driven entropy.**
- Steps 2-3 of W32 and W64 have 2.0-3.5 bits of noise-driven entropy. This comes from units whose pre-activation is
  mostly noise.
- Those units are mostly irrelevant: D(1%) is 23 of 32 and 50 of 64 at step 2.

**Rotate and concentrate.**
- The relevant pre-activations' variance concentrates in 1-9 directions.
- Reproducing 99% of their gate values needs 3-12 directions (T16), 4-20 (W32) and 4-29 (W64) at steps 2-10.
- With that many directions, the cell bound log2 sum_{j<=k} C(D, j) is 9-16, 13-32 and 18-61 bits. That is far above
  the measured H(G), so the data use only a small part of what the geometry allows.

## 3. Phase B PFN (sigma = 0.05)

- **Network.** `work/cm-interp-B/runs/L2_s0.05`: 2 blocks, width 32, 16 tokens, and 64 GELU units per token and
  block.
- **Layer 1.**
  - All 1,024 gates: H(G) lies between 15 (plug-in; all 2^15 patterns are distinct; the approximate collision bound is
    27*) and 199 (FVSBN).
  - D(1%) = 243 and D(0.1%) = 330.
    - On each of the three tokens that carry a source's reading of the query question, 51 of 64 units are relevant.
    - On the query token, 42 are.
    - Each of the 12 labelled-question tokens has 3-5.
  - H(G_R) lies between 15 [27*] and 61, in 30 directions for 99% of the variance and 40 for 99% of gate values.
- **Layer 2.**
  - All gates: H(G) between 15 [27*] and 188.
  - Only the query token's units can reach the output. A code check confirmed that the other tokens' layer-2 flips
    change nothing.
  - D = 52 of its 64. H(G_R) = 11.6-16.0, in 5 directions for 99% of the variance and 7 for 99% of gate values.
  - Entropy of all 64 query-token gates (the registered quantity): 13.5-16.8 bits (plug-in / MADE), NSB 14.6.
- **Merging.** E3(1%) was not reached in either layer.
  - Layer 1: 90% of answers still change at 8.4 bits (428 leaves).
  - Layer 2: 18% at 8.3 bits (393 leaves).
  - At those points the population-mean predictive mean moves by only 0.002 and 0.001. Merging preserves the average;
    it does not preserve the per-episode deviation from Bayes.

## 4. Random MLPs (depth 32; question: the output mean)

ARC He-ReLU, sigma = 0.01. Each cell gives H(G) / E3(1%) / E3 at bias nMSE <= 1e-4 / the noise-driven H(G|X) (plug-in).
A * marks the approximate zero-collision bound. D(1%) = m in every cell.

| width | layer 1 | layer 4 | layer 8 | layer 16 | layer 24 | layer 32 |
|---|---|---|---|---|---|---|
| 32 | 26-28 / nr / nr / 0 | 15-16 / nr / nr / 0.7 | 6.8 / 6.3 / 1.2 / 0.7 | 5.3 / 3.7 / 1.1 / 1.7 | 2.5 / 1.1 / 0.7 / 1.0 | 3.7 / 0.0 / 1.0 / 2.1 |
| 64 | 45-56 / nr / nr / 0 | 21-28 / nr / nr / 1.2 | 15-17 / nr / 8.9 / 1.6 | 12.3 / nr / 9.0 / 2.9 | 9.5 / nr / 4.4 / 3.6 | 8.8 / 7.4 / 5.7 / 4.2 |
| 128 | 87-110 / nr / nr / 0 | 19 [35*]-63 / nr / nr / 2.2 | 27-37 / nr / nr / 2.8 | 17-22 / nr / nr / 4.1 | 17-23 / nr / nr / 5.5 | 16-18 / nr / nr / 5.9 |
| 256 | 173-221 / nr / nr / 0 | 17 [31*]-130 / nr / nr / 4.1 | 17 [31*]-82 / nr / nr / 5.6 | 17 [31*]-52 / nr / nr / 6.9 | 30-43 / nr / nr / 7.5 | 28-40 / nr / nr / 7.9 |

Chaotic tanh (g = 3), sigma = 0.01. The upper estimate per layer is:
- width 32: 34-43;
- width 64: 73-86;
- width 128: 150-172;
- width 256: 317-367.

The lower ends are 37, 71, 143 and 285 at layer 1 (Gaussian inequality; 37 for width 32 comes from collisions), and the
approximate collision bound (31-35*) deeper down.

**What the MLP results show.**
- **Every gate is relevant to the intervention, everywhere.** The median single flip moves the output by more than 10%
  of its spread on 58-98% of inputs (ARC) and 67-100% (chaotic tanh). Ordered tanh is the exception:
  - D = 0 at layers 1-16 (width 64), where nothing reaches the output;
  - its H(G) is 11.8 / 8.9 / 1.0 / 0.9 / 0.0 / 0.8 / 0.8 at layers 1-32;
  - E3 = 0 below layer 32.
- **H(G) scales with width.**
  - At layer 1 it is about 0.7-0.87 m.
  - For ARC it then falls with depth, through the collapse, to 0.11-0.15 m at layer 32. Chaotic tanh does not fall.
- **More noise raises the deep-layer entropy and blocks merging.** ARC width 64 at sigma = 0.1 has H(G) = 16-22 bits at
  layers 8-32, against 9-17 at sigma = 0.01 and 8-17 at sigma = 0.001.
  - The noise-driven part is large. Its per-input plug-in hits the 8-bit cap, and its per-unit sum is 21-26 bits.
  - E3(1%) is not reached at any layer (48-99% of answers changed at 10 bits).
  - Layers 1-4 hardly depend on sigma.
- **Chaos amplifies even small noise.** In chaotic tanh at width 64, the gate-entropy upper estimates are the same at
  sigma = 0.001 and 0.01 (73-86 bits per layer). The noise-driven part H(G|X) is 1.3 bits at layer 8 and sigma =
  0.001, against 6.6-12.6 at sigma = 0.01. By layers 24-32, at both sigmas, the per-input plug-in sits at its 8-bit
  cap and the per-unit sums are 69-94 bits, so most of the pattern there may be noise-driven. It is not resolved.
- **The noise-driven part grows with depth.** At layer 32 of ARC it is 4.2-5.3 of 8.8 bits (width 64), 5.9-8.3 of
  16-18 (width 128), and 7.9 (plug-in, at its cap) to 18.7 (per-unit sum) of 28-40 (width 256).
- **Merging at 1% per sample works only where H(G) is already small.** Fresh re-checks:
  - ARC width 32, layers 8 / 16 / 24 / 32: E3 = 6.3 / 3.7 / 1.1 / 0.0 bits, with fresh change 0.78 / 0.29 / 0.12 /
    0.37%;
  - ARC width 64, layer 32: 7.4 bits, 0.84%, 385 leaves.
  - At sigma = 0.001, ARC width 64: 8.0 bits at layer 24 and 6.0 at layer 32. Fresh change is 1.00% for both (95%
    upper limit 1.10%), so both are at the edge.
  - Everywhere else the tree stops with 1.3-100% of answers changed. That is above 12%, except at layer 24 of width
    64, at 8-10 bits.
- **For the mean (errors may cancel), deep layers merge further** (fresh re-check):
  - ARC width 64: bias <= 1e-4 at 8.9, 9.0, 4.4 and 5.7 bits at layers 8-32. The fresh bias is 0.57-1.03e-4 (one point,
    layer 24, slightly above).
  - <= 1e-5 at 7.4 and 8.0 bits at layers 24-32.
  - Not at layers 1-4 (bias above 1e-3 at 10 bits), and not at width >= 128.
  - In chaotic tanh, a single class meets 1e-3, and at width 32 even 1e-4, at shallow layers. The substitution errors
    average out. This was not re-checked on fresh samples.
- **Fano illustration (conditional).**
  - Let rho be the fraction of samples touched by a merge whose answer changes. Then P(pattern replaced) <= eps / rho,
    and E3 >= H(G) - h2(eps/rho) - (eps/rho) m.
  - With rho as observed for our trees (0.92-1.00 at width 256; 0.81-0.93 at layers 1-2 of width 32), this gives
    about 170 bits at layer 1 and 25 at layer 32 for width 256.
  - With the more conservative rho = the smallest single-flip rate (0.07-0.73), the bound vanishes at deep layers.
  - Neither rho is guaranteed for every merging.

## 5. Estimator checks

T16 per-step patterns, with a reference from plug-in on 2^21 samples (2-842 cells; bias < 0.02). The biases below are
means of 3 disjoint subsamples (`results/exact_check_T16.json`):
- NSB: within 0.07 at every n from 2^10 to 2^16.
- Plug-in: -0.2 at n = 2^10, and less than 0.04 in size at n >= 2^14.
- Renyi-2: 0.8-1.8 below, as it must be.
- FVSBN: +0.24-0.50 at n = 2^16 (2-6 at n = 2^12).
- MADE: +0.01-0.10 at n = 2^16.

Synthetic checks:
- 64 independent fair bits: FVSBN and MADE give 64.0, and the zero-collision bound gives 21-29.
- Gaussian orthant patterns with m = 16: MADE gives 11.33-11.41 against the true 11.41 (n = 2^14-2^16).

## 6. Predictions (PREREG.md) scored

- **P1.**
  - C_l >= m everywhere: hit.
  - RNN upper bounds below 0.5 m at every step: hit.
  - Covariance bound below m for ARC width 64 at sigma 0.1 at depth >= 16: partly hit. It is below m at layers 24
    and 32 (57 and 63 bits), not at layer 16 (79).
- **P2.**
  - T16 at 4-7 bits at steps 3-10: hit (4.4-6.0; the full run moved the smoke values by at most 0.05).
  - W32 at 6-14 and W64 at 8-20 bits at steps >= 4: hit.
  - H(G|X) < 1 bit everywhere: miss for W32 (steps 2-4) and W64 (steps 2-5).
- **P3.**
  - T16 D >= 13 at steps >= 3: hit.
  - W32 within 50-95% of m: only at steps 2-5.
  - W64 within 30-90%: only at steps 2-3.
  - So mostly a miss: the wide nets are not sparser.
- **P4.**
  - T16: E3(1%) <= H(G_R) - 1 at every step >= 3 and E3 <= 5: hit (savings 1.4-2.1; E3 <= 4.1).
  - E3 <= 8 for W32 and W64: holds where reached (<= 7.4). Steps 9-10 (W32) and 7-10 (W64) are unmeasured within the
    budget.
  - E3(0.2%) not reached at some steps: hit (W32, W64).
- **P5.** k(var99) <= 8 / 16 / 24 for T16 / W32 / W64: hit (<= 7 / 9 / 4). "k(agree) above k(var)" fails where they are
  equal: step 1 everywhere and T16 step 2.
- **P6.**
  - Layer-1 upper bound >= 0.6 m: hit (0.86-0.87 m).
  - Layer-32 upper bound <= 0.6 x layer 1's: hit (0.14-0.18).
  - D >= 0.9 m at width 64: hit; D >= 0.5 m at width 256: hit.
  - E3(1%) not reached at layers <= 16 at width 64: hit.
  - H(G|X) at layer 32 below 0.5 bit at sigma 0.001: miss (0.59).
  - H(G|X) at layer 32 >= 1 bit at sigma 0.1: hit (8 capped to 21 bits).
- **P7.**
  - Ordered H(G) < 1 bit at every layer: miss at layers 1-2 (11.8, 8.9).
  - Ordered D <= 5: holds at layers 1-16 (D = 0); miss at layers 24 and 32 (37 and 128, where the output spread is
    tiny).
  - Chaotic upper bound >= 0.5 x 1.58 d: hit.
  - Chaotic E3(1%) not reached at layers <= 16: hit.
- **P8.**
  - Layer-1 lower bound >= 20: rests on the approximate collision bound (27*); the plug-in gives 15.
  - FVSBN 100-600: hit (199).
  - Query-token entropy 5-30: hit (13.5-16.8 over all 64 query gates).
  - D in 30-600 (243) and in 20-64 (52): hit.
  - E3(1%) <= 8 for layer 2: unmeasured (not reached at 8.3 bits).
- **Amendment 1.**
  - E3(1%) on T16 unchanged: hit.
  - E3(0.2%) reached at every step: hit.
  - E3(0.2%) <= H(G) - 0.3 at most steps: 6 of 10.

## 7. Review fixes (GPT-6, `review/review1.md`)

| finding | issue | fix |
|---|---|---|
| 1 | E3 checkpoints were chosen on the held-out set that measured them | all reached E3 re-evaluated on fresh samples with frozen trees (Amendment 3); fresh numbers and 95% limits reported |
| 2 | "At most N cells" confused entropy with cell count | leaf counts reported separately; the claim that this establishes an affordable boundary-exact family for AlgZoo removed |
| 3 | Scaling conclusions assumed their premise | the Fano numbers are now conditional illustrations; "E3 grows linearly" replaced by what was measured |
| 4 | Brackets presented as guaranteed bounds | relabelled as estimated |
| 5 | "Capacity never binds" was wrong | corrected; the fresh-input allowance is added |
| 6 | The mean-error caveat was missing | added; fresh bias re-checks reported |
| 7 | Chaotic noise dominance claimed without support | marked unresolved; ARC width-64 share corrected |
| 8 | D presented as boundary proximity | D called intervention sensitivity; the tanh flip caveat added |
| 9 | Layer-1 bound called rigorous | the exact inequality is now distinguished from its Monte Carlo estimate |
| 10, 11 | Numbers and scoring | corrected |
| 12 | Point-estimate and tanh-key slips in the tables | fixed in `scaling.py`; the soft-phase start condition documented |

## 8. Caveats

- **E3 is an achieved size for one family type**: classes over one layer's gate values, one shared pattern per class.
  - The rest of the network is exact. That is easier than a full family, where every layer is described.
  - E3 is not a lower bound. "Not reached" usually reflects the 1,024-8,192 tree samples.
- **One weight seed per random-MLP configuration.** Width 256 was run only at sigma = 0.01. The RNNs have a seed-1 check
  (section 2).
- **Tolerances are choices.** A per-sample 10% is strict for a mean. A 10% share of the Bayes deviation is what the
  PFN question needs.
- **No cost model of an actual family.** Tractability is argued from class counts times d^2.

## Progress log (UTC)

- 13:48-14:10: read the briefs and code of the four lanes; wrote the estimators, adapters and the merging tree;
  synthetic estimator checks; smoke runs (only T16's numbers were read).
- 14:13: PREREG.md hashed; main runs started (3 processes, nice).
- 14:20: Amendment 1, after T16: the tree's hard phase stopped early. A soft continuation was added and all E3 were
  recomputed.
- 14:24: the coordinator asked for width scaling. ARC and chaotic tanh at widths 32 and 128 were added, plus the
  layer-1 Gaussian bound.
- 14:55: Amendment 2: two width-256 sigma points were dropped for time.
- 15:06-15:14: GPT-6 review (twelve findings).
- 15:20: Amendment 3: fresh-sample re-check of every reached E3; four low-priority nets dropped.
- 15:20-15:27: fresh re-checks and the PFN query-gate entropy done.
- 15:45: Kelvin's rule relayed: nice 19 + ionice idle for all CPU jobs. The queue runner was switched; the last job
  (W64 seed 1) was already reniced.
- 15:51: last run (W64 seed 1) done; report final.
