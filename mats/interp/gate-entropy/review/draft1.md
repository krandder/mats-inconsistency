# How big must the discrete part be? Gate entropies of noisy networks (29 Oct 2026)

Work directory: `work/gate-entropy/`. It holds the code, the logs, `results/*.json`, `results/tables.md`
(`summarize.py`) and `results/scaling.md` (`scaling.py`). `PREREG.md` was hashed before the main runs, and one dated
amendment was hashed before its results (`prereg_hashes.txt`).

Brief from main, relaying Kelvin: move from Gaussians to families that put mass correctly on each side of the decision
boundaries. This study measures how large the discrete part of such a family must be. It does not build the family.
One GPT-6 review was done before the claims (`review/`).

## Summary

For every layer (every step for the RNNs) we measured three quantities over inputs and noise:
- H(G), the entropy of the gate pattern, i.e. which side of each kink every unit is on;
- H(G_R), the same for the D gates whose flip changes the question's answer on at least 1% of held-out samples;
- E3, the entropy of the coarsest merging of gate patterns we could find whose members can share one pattern while at
  most 1% (or 0.2%) of answers change. It is verified on held-out samples by running the real network with the
  shared pattern.

E3 is a size we achieved. It is not a lower bound, and better merging could beat it. All values are in bits. Every
number states the network, sigma and the estimator.

1. **AlgZoo 2nd-argmax RNNs: the discrete part is small.**
   - T16 is ARC's example M_{16,10} fine-tuned at sigma = 0.001, with 16 gates per step.
     - H(G_t) = 4.1-6.0 at steps 2-10. This is the plug-in estimate on 2^21 samples, exact to 0.02. NSB and MADE agree
       to within 0.1 on subsamples.
     - Every gate is relevant from step 4 on.
     - E3 = 2.3-4.1 at a 1% answer-change rate and 2.4-5.8 at 0.2% (ARC's bar), so at most about 60 cells per step.
   - Widths 32 and 64 (sigma = 0.01): H(G_t) = 6.8-10.3 and 8.9-12.8 at steps 2-10.
     - E3(1%) = 2.4-7.2 at steps 2-8 (width 32) and 2.5-7.4 at steps 2-6 (width 64).
     - At later steps the tree, built from 8,192 samples, stopped at 1.0-2.5% of answers changed, at 8.5-9.6 bits. With
       no merging, H(G) = 9.5-12.8 gives 0%.
   - Growth: at step 6, E3(1%) is 4.0, 5.8 and 7.4 at widths 16, 32 and 64. Width 16 has a different sigma and
     training, so only the 32-to-64 step is clean: about +1.6 bits for twice the width.
   - Rotate and concentrate: 99% of the relevant pre-activations' variance lies in 2-9 directions, but reproducing 99%
     of their gate values needs 12-29.
2. **Phase B PFN (sigma = 0.05): large, and no merging found.**
   - The question is the predictive mean to 10% of its deviation from exact Bayes (tolerance 0.0063).
   - Layer 1 has 243 relevant GELU gates of 1,024. They sit on the three tokens that carry the sources' readings of the
     query (51 each) and the query token (42). Their joint pattern never repeats in 2^15 episodes: H(G_R) lies between
     27 (collision bound) and 61 (FVSBN upper bound). They span at least 30 directions.
   - Layer 2: 52 relevant units, all on the query token, with H(G_R) = 11.6-16.0.
   - E3 at 1% was not reached in either layer [numbers after the rerun].
3. **Random MLPs (question: the output mean; per-sample tolerance 10% of the output's spread).**
   - Every gate is relevant (D = m) at every layer, width and sigma we ran. A single flip moves the output by more than
     10% of its spread on a median 58-98% of inputs.
   - The gate entropy grows linearly with width at every depth and falls with depth. ARC He-ReLU at sigma = 0.01:
     - layer 1: at least 0.68-0.70 m (a rigorous Gaussian chain-rule bound): 22, 45, 87 and 173 bits at widths 32, 64,
       128 and 256;
     - layer 32: 3.7, 8.8 and 15.6-18.4 bits at widths 32, 64 and 128 [256 pending], i.e. 0.12-0.14 bit per gate.
   - Chaotic tanh (g = 3) does not fall with depth: [numbers].
   - Merging:
     - At the 1% per-sample tolerance, the tree finds no merging at layers <= 16 for widths >= 64. Substituting any
       pattern changes 81-93% of the answers it touches (width 32, layers 1-2). Under that condition Fano's inequality
       puts E3 within about 1 bit of H(G).
     - Measured on the mean instead (cancellations allowed), merging works at deep layers: bias nMSE <= 1e-4 at 4.4-9.0
       bits at layers 8-32 of width 64. It fails at layers 1-4 (bias above 1e-3 at 10 bits).
4. **Capacity never binds.** C_l = (d/2) log2(1 + 1/sigma^2) is 159-2,214 bits, far above m in every net. The
   covariance version, (1/2) log2 det(I + Cov/noise), is 17-650 bits. What keeps the RNN entropies far below m is
   geometry: a step-t pattern is a function of t scalars plus noise. In the random MLPs it is the collapse with depth.
   H(G) <= min(m, C_l) is true but loose here. It also bounds only I(G; X), not the noise-driven part H(G | X).
5. **Noise-driven gate entropy H(G | X).**
   - RNNs: small, except early steps of the wide RNNs (2.6-4.9 bits at steps 2-3, from units that only see noise).
   - Random MLPs at sigma = 0.01: grows with depth, reaching about 40-50% of H(G) at layer 32 (4.2-5.3 of 8.8 bits at
     width 64; 5.9-8.3 of 15.6-18.4 at width 128). At sigma = 0.001 it stays under 0.6 bit.

**What this implies for a boundary-exact family at scale (the coordinator's question; targets M1 and M3).**
[Final wording after all runs.]
- A family whose discrete part enumerates gate cells, one Gaussian piece per cell, is affordable for the AlgZoo RNNs.
  It needs about 2^4-2^6 cells per step at ARC's bar for M_{16,10}, and at most about 2^10-2^13 at width 64, and it
  grows by about 1.6 bits per doubling of width.
- For random MLPs such a family is not affordable as an enumeration. At the output tolerance used here, the merged
  entropy is close to H(G) wherever we can measure it, and H(G) grows linearly with width at every depth: about 0.7
  bit per gate at layer 1 and about 0.13 at layer 32. At width 256 that means at least 173 bits at layer 1 and, by
  extrapolation, about 30 at layer 32.
- The PFN's question needs about 27-61 bits in layer 1.
- What survives:
  - the gate distributions are close to an autoregressive-logistic model (FVSBN within about 10-20% of the bounds),
    so a factored discrete part with polynomially many parameters could represent them even when 2^H cells cannot
    be listed;
  - for the output mean, cancellation lets deep layers be merged to 2^4-2^9 cells.
- Whether a factored family can propagate through the shallow layers without enumerating cells is the open question.
  This census cannot answer it.
