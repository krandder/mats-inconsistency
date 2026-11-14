# PREREG: gate entropies of noisy networks (work/gate-entropy, 2026-10-29)

Written 14:15Z, before any main run. Brief from main (Kelvin's direction: a family that represents decision boundaries
exactly; this measurement says how big its discrete part must be). Nothing here builds the family.

## What was seen before this file
- A smoke run on T16 (ARC's example M_{16,10} fine-tuned at sigma = 0.001) with N = 8,192 entropy samples, 256 flip
  samples, 2,048 held-out samples and at most 64 tree splits: per-step H(G_t) plug-in 1.0, 4.1, 4.7, 5.3, 5.7, 6.0,
  5.8, 5.3, 4.7, 4.4 bits (16 gates per step); D(1%) = 5 at step 1 and 13-16 at steps 2-10; noise-only entropy
  0-0.6 bits; components for 99% variance 1-6; the 64-split tree reached a 1% change rate only at steps 1-2.
- Smoke runs of the other nets were checked only for errors and run time (numbers not read). A code check confirmed
  that recomputing answers with the actual gate pattern reproduces the forward pass (0 discrepancies; PFN 5e-7), and
  that flipping layer-2 gates of non-query PFN tokens changes nothing (0 of 64 x 64).
- Synthetic estimator checks (iid bits, Gaussian orthant patterns): FVSBN/MADE held-out cross-entropy within 0.1 bit
  of the truth at N >= 2^14 train; NSB biased low by 1-2 bits when undersampled (iid 24 bits at N = 2^16: 19.9 vs 21.2).

## Definitions (fixed)
- Gate: ReLU g = 1[z > 0]; tanh: the two kinks of hardtanh, up = 1[z > 1], dn = 1[z < -1] (saturation region; 3 states
  per neuron, entropies counted on the joint pattern); GELU: g = 1[u > 0], the middle of the switching region.
- Flip: the gate's linear piece is swapped while the pre-activation is held (ReLU: a = g z; tanh: hardtanh piece change
  added to tanh(z); GELU: u Phi(u) -> u Phi(-u)); everything downstream is recomputed by the actual network with the
  same input and noise draws (common random numbers).
- Per-sample answer and "changed":
  - AlgZoo RNNs: correct = 1[argmax logits = 2nd argmax of x]; changed if it differs.
  - Random MLPs: the output vector (ARC: a_L; tanh: W_out s_L); changed if ||delta|| > 0.1 sqrt(sum_j Var y_j).
  - PFN: the predictive (mixture) mean of the query; changed if |delta| > 0.1 x RMS(net mean - Bayes mean) (flip set).
- E1 = H(G_l), all gates of layer l, over inputs and noise.
- E2: f_i = fraction of held-out samples whose answer changes when gate i alone is flipped. D(tau) = #{i: f_i >= tau},
  tau = 1% (primary) and 0.1%. E2 = H(G_R) for R = the D(tau) gates.
- E3: greedy decision tree over the gate values, built on the flip set (N_flip samples, single-flip care sets); each
  leaf replaces all gates of its members by the leaf's majority pattern. Each checkpoint is verified on held-out samples
  by actual substitution. E3(eps) = held-out entropy of the leaves at the first checkpoint whose verified change rate
  is <= eps. eps = 1% (primary) and 0.2% (RNN: ARC's bar needs a bias below about 0.2 points). For the MLPs also the
  entropy at which the mean's squared bias (ARC's nMSE units) falls below 1e-3, 1e-4, 1e-5. If not reached within the
  split budget, E3 is reported as not reached with the last entropy and change rate (not a lower bound).
- Estimators. Lower: plug-in (biased low in expectation), Renyi-2 from pair collisions (H >= H_2; with zero collisions
  the approximate bound log2(pairs/3) is reported separately). Upper: sum of per-unit entropies; FVSBN (masked
  logistic) and, for m <= 64 in E1, MADE held-out cross-entropy + 2 SE. NSB and Miller-Madow as point estimates only
  when the bracket [lower, upper] is narrower than 1 bit. Otherwise the bracket is the result.
- Subspace: PCA of the relevant gates' pre-activations on the flip set: components for 99% variance, and for 99% /
  99.9% mean gate agreement after reconstruction; log2 sum_{j<=k} C(D, j) as the cell bound for D hyperplanes in k dims.
- Capacity: C_l = (d/2) log2(1 + 1/sigma^2) (noise relative to unit activation scale); the covariance bound
  (1/2) log2 det(I + Cov(signal)/noise var) on the noisy state feeding layer l, which bounds I(G_l; X) for MLP layers
  >= 2 (RNN: plus log2(d+1) for the noiseless fresh input). It does not bound H(G_l | X) (noise-driven gate entropy),
  measured on 128-512 inputs x 256 noise draws.
- Exact check: T16 per-step patterns (16 gates) at N = 2^21 (plug-in bias < 0.02 bits) as the reference for the
  estimators run on subsamples 2^10-2^16.

## Networks (one weight seed each unless stated)
- (a) T16 (runs/ft_16_10_ex_s0.001.npz), W32 (runs/gpu/w32_s0_sig0.01.npz), W64 (runs/gpu/w64_s0_sig0.01.npz); seed 1 of
  W32/W64 if time allows. N_ent = 2^21, N_flip = 8,192, N_ho = 2^16, up to 4,096 splits.
- (b) PFN L2_s0.05 (phase B main net), layers 1-2 (16 tokens x 64 GELU units each); N_ent = 2^15, N_flip = 1,024,
  N_ho = 8,192, up to 512 splits.
- (c) Depth-32 nets, layers {1, 2, 4, 8, 16, 24, 32} at width 64 and {1, 4, 8, 16, 24, 32} at width 256:
  - ARC He-ReLU (no bias, no LN), width 64 at sigma 0.001 / 0.01 / 0.1; width 256 at 0.001 / 0.01 / 0.1;
  - tanh + LN with bias N(0, 0.25) (phase R's E[output] nets), ordered g = 0.25 and chaotic g = 3, width 64 at
    0.001 / 0.01 / 0.1; width 256 at 0.01 only.
  Width 64: N_ent = 2^19, N_flip = 4,096, N_ho = 2^15, up to 2,048 splits. Width 256: 2^17, 1,024, 8,192, 1,024.

## Predictions (written before the main runs)
- P1 Capacity: C_l >= m for every net here (arithmetic for sigma <= 0.1), so min(m, C_l) = m never binds; the measured
  H(G_l) upper bound is below 0.5 m at most RNN steps. The covariance bound falls below m for ARC width 64 at sigma
  0.1 at depth >= 16 (50%).
- P2 RNN E1: T16 4-7 bits per step at steps 3-10 (the full run moves the smoke values by < 0.3 bits); W32 6-14 bits,
  W64 8-20 bits at steps >= 4. H(G|X) < 1 bit per step in all three.
- P3 RNN D(1%): T16 >= 13 of 16 at steps >= 3; W32 between 50% and 95% of 32; W64 between 30% and 90% of 64.
- P4 RNN E3(1%) <= E2 - 1 bit at every step >= 3, and <= 5 bits (T16), <= 8 bits (W32, W64). E3(0.2%) not reached
  at some steps (40%).
- P5 RNN subspace: 99%-variance components <= 8 (T16), <= 16 (W32), <= 24 (W64); gate-agreement k above them.
- P6 ARC ReLU: layer-1 upper bound >= 0.6 m; at layer 32 the upper bound is <= 0.6 x its layer-1 value at sigma
  <= 0.01 (collapse); H(G|X) at layer 32 >= 1 bit at sigma 0.1 and < 0.5 bits at 0.001. D(1%) >= 0.9 m at every layer
  at width 64 and >= 0.5 m at width 256. E3(1%) not reached within the budget at layers <= 16 (width 64).
- P7 tanh: ordered, H(G_l) < 1 bit and D(1%) <= 5 at every layer; chaotic, upper bound >= 0.5 x 1.58 d at every layer
  and E3(1%) not reached at layers <= 16.
- P8 PFN: layer-1 entropy beyond plug-in reach (lower bound >= 20 bits), FVSBN upper bound 100-600 bits; layer 2
  (query token) 5-30 bits; D(1%) between 30 and 600 of 1,024 (layer 1) and 20-64 of 64 query units (layer 2);
  E3(1%) <= 8 bits for layer 2 (60%).

## Rules
CPU only, at most 3 processes, nice 10, OMP_NUM_THREADS=1. One GPT-6 review (codex-any exec) before claims. Amendments
are appended below with times.

## Amendment 1 (14:20Z, after the T16 run, before any result of the changed tree)
On T16 the tree stopped when the training care-violations ran out (48-67 leaves) and E3(0.2%) was not reached at
steps 5-10 (verified change rates 0.23-0.52% at 4.0-4.9 bits); E3(1%) was reached at every step. Change: after the
hard phase, the tree continues with a soft criterion, the drop in flip-rate-weighted mismatches
sum_x sum_j f_j [G_xj != rep_j] per bit (f_j = training flip rate), up to the same split budget; every checkpoint is
still verified on held-out samples by actual substitution, so the reported E3 remains an achievable (verified) rate.
All E3 numbers are recomputed with this tree (e3_rerun.py for nets that ran before the change); the first version is
kept in the JSON as E3_v1. Prediction for the change: E3(1%) moves by less than 0.3 bits on T16; E3(0.2%) is reached
at every T16 step and is <= H(G_t) - 0.3 bits at most steps (50%).

## Amendment 2 (written 14:55Z; scope changes for time and for the coordinator's question)
- Added at 14:24Z, before their results, but recorded here only at 14:55Z, after the first of them had finished: at the
  coordinator's request (how the merged entropy grows with width), ARC He-ReLU and chaotic tanh at widths
  32 and 128, sigma = 0.01, same protocol as width 64 (tanh width 128 uses the width-256 sizes because it has 256
  gates per layer). Added a rigorous layer-1 lower bound for the random MLPs (layer1_bound.py: z_1 is exactly
  Gaussian, so H(G_1) >= sum_i H(g_i | z_{<i}) from the Cholesky factor).
- Dropped for time, before any result of them: ARC width 256 at sigma = 0.001 and 0.1 (width 256 is kept at sigma =
  0.01). The seed-1 RNN checks
  run last and only if time allows.

## Amendment 3 (15:20Z, after the GPT-6 review, before these validations ran)
- E3 checkpoints were selected on the same held-out set that measured them (review finding 1). Each reached E3 is now
  re-evaluated with its tree frozen at the selected checkpoint on fresh samples (seed 3; 2^16 for the RNNs, 2^15 for the
  MLPs), reporting the change rate with a one-sided 95% Clopper-Pearson upper limit, the leaf entropy and the number of
  leaves (e3_fresh.py). Reported E3 values are the fresh ones.
- Dropped for time: chaotic tanh width 64 at sigma 0.1, ordered tanh width 64 at sigma 0.001 and 0.1, ordered tanh
  width 256.
