# Narrow-deep networks: pre-registration, Part 1 (written 2026-10-29 ~22:20Z, before any main run)

Ask (Kelvin, via main): what is the theory/literature for very narrow but very deep networks, can they be trained for
interesting tasks; "I would hope that the structure of the matrices, regardless of the gates, would give us a clear
bound on what dimensions can matter first. We should explore the narrow-but-deep networks; maybe n<=16, but L>=64."
Target this feeds: M1 (sample-free estimator matching sampling at depth growing with width) and M3 (certified tails),
outbox/cm-interp-program-2026-10-28.md "The bar".

## Seen before writing this (declared)
- Theory numbers only (scratch/theory_numbers.py, no network run): half-width Lyapunov exponents along one path
  (Theorem C / Corollary C1 of outbox/lyapunov-dimension-2026-10-29.md), m ~ Bin(n, 1/2):
  n = 8: radial lambda_r = -0.199/layer; angular mu_1..4 = -0.413, -0.643, -0.852, -1.021 (gaps to radial
  0.214, 0.444, 0.652, 0.821). n = 16: lambda_r = -0.088; mu_1..4 = -0.175, -0.281, -0.411, -0.565 (gaps 0.086,
  0.193, 0.323, 0.477). n = 256: gap_1 = 0.004.
  A random path dies (some layer with no active unit) by depth 64/128/256 with probability 0.22/0.39/0.63 (n = 8),
  0.001/0.002/0.004 (n = 16). P(some layer with <= 1 active unit) by L = 64: 0.90 (n = 8), 0.017 (n = 16); <= 2
  active units by L = 256: 1.00 (n = 8), 0.41 (n = 16). rank(J_x) <= min_l m_l(x) exactly.
  Mean-field (infinite width) angle between two inputs from pi/2: 0.38 / 0.13 / 0.035 rad at depth 16 / 64 / 256.
- Code pilots (timing and "does it run" only, no analysis numbers kept): RNN 2nd-argmax T = 64 width 16, 200 steps on
  CPU (loss 4.13, chance 4.16); residual MLP sorting n = 16, L = 64, 400 steps (nMSE 0.36); plain MLP 300 steps
  (nMSE 0.96). These pilots fix nothing below.
- Prior lanes read: lyapunov-dimension, nongauss-family, random-net-phases, hardness-mean-estimation, gate-entropy,
  algzoo-noisy (summaries).

## Objects
Random nets (lib/nd.py). x ~ N(0, I_n), n in {8, 16} (n = 4 and 32 only as scaling probes where stated).
- Plain He-ReLU (critical, ARC's form): z_l = W_l h_{l-1}, a_l = ReLU(z_l), h_l = a_l + s_l eps_l, W_l iid
  N(0, 2/n), no bias. Output a_L. Depth-64 and 128 nets are truncations of the depth-256 net (same W_1..W_L).
- Residual variant: u_l = W_l h_{l-1}, b_l = ReLU(u_l) + s_l eps_l, h_l = h_{l-1} + V_l b_l / sqrt(L), W_l iid
  N(0, 2/n), V_l iid N(0, 1/n), h_0 = x. Output h_L. (Branch variance 1/L; E|h_l|^2 grows by (1 + 1/L) per layer.)
  Each depth L has its own 1/sqrt(L); the first L matrices are shared across depths.
- Noise s_l = sigma r_l, r_l = RMS of the clean activation of layer l in the noisy net (seeded pilot, 2^14 paths,
  float64). sigma in {0, 1e-3} (main), 1e-2 secondary. Weights from numpy RandomState(2_000_003 + 7919 seed + 104729 n),
  seeds 0-7 for the plain family (all eight used for population statistics, the first four for per-net analyses),
  0-3 for the residual family. float64 throughout.
- Estimands: E[a_L] (vector; residual: E[h_L]); nMSE = |est - truth|^2 / tr Var(output), as ARC. Tail:
  p_t = P(|a_L| > t) with t the (1 - 1e-4) quantile of |a_L| from the truth run (plain; residual: |h_L|).
- Dead nets: a plain net whose ground-truth E|a_L|^2 is below 1e-12 of its value at layer 1 times the radial drift is
  "dead"; reported as a fraction, excluded from estimation comparisons (estimating 0 is trivial).

Trained nets (train/).
- RNN: ARC's 2nd-argmax architecture (ReLU, no bias, h_0 = 0, logits W_oh h_T), width 16, T = 64 (main) and T = 32
  (trend), sigma in {0, 0.01} (noise as algzoo-noisy: per-neuron, per-step, relative to the activation RMS), seeds 0-3
  at T = 64 and 0-1 at T = 32, ARC's recipe (AdamW, lr 0.04/sqrt(16), wd 0.001, batch 1024, decay to 0.01x at half),
  2^18 steps (2^28 sequences, a quarter of ARC's budget) on one on-demand L4 if available, else fewer steps on CPU.
- MLP (task of my choice): sorting x ~ N(0, I_16) (target sort(x); sorting networks are narrow-deep comparator
  circuits and the map is positively homogeneous like the nets). Bias-free residual ReLU MLP width 16 (as the residual
  variant, V init 1/sqrt(nL)) at L = 16, 64, 128, and a plain He-ReLU MLP width 16 (+ linear readout) at L = 16 and
  64, sigma in {0, 0.01}, Adam lr 2e-3, cosine decay, batch 1024, 2^15 steps (L4) or 2^14 (CPU).

## Part 2 measurements (random nets), and predictions
Per-path Jacobian J_x = d a_L / d z_c (plain; residual: d h_L / d h_c), cut c = L - L', L' in {4, 8, 16, 32, 64,
128, 192, 248} as allowed by L. N_surv(J; delta) = #{k : s_k >= delta s_1}, delta in {e^-1, e^-2, 0.01, 0.001};
k*_F(A; eta) = min{k : sum_{j>k} lambda_j(A) <= eta tr A}. 1024 paths per net and cut (noise on).
P1 (plain, Lyapunov regime). Median N_surv(0.01) per path <= 2 at L' >= 32 (n = 8) and at L' >= 64 (n = 16); the mean
   of log(s_2/s_1) per layer at L' >= 64 is within 25% of -gap_1 (n = 8: -0.214; n = 16: -0.086) once paths whose
   min m_l <= 1 (rank 1) are excluded. Fraction of paths with rank(J_x) = 1 (some layer with m_l <= 1) matches the
   binomial prediction within 0.05 at L' = 64 (n = 8: 0.90).
P2 (collapse, plain). Mean pairwise angle between two inputs' states decays to below 0.05 rad by depth 64 (n = 8) and
   by depth 128 (n = 16) at sigma = 0, i.e. faster than mean field (0.13 rad at 64); its log-slope at depth 32-128 is
   within 30% of -gap_1 (sigma = 0). With sigma = 1e-3 the angle floors at 1e-3-1e-2 rad. Per-layer gate
   disagreement between two inputs falls below 1% by depth 64 (n = 8) and 128 (n = 16) at sigma = 0 (wide nets: 6-10%
   at depth 32). Fraction of inputs dead by depth L, averaged over nets, within 0.1 of the per-path formula; per net it
   is bimodal (most nets near 0 or near 1 at L = 256, n = 8).
P3 (fixed block, plain). k*_F(M; 1e-2) for M = E_x J_x^T J_x is <= 3 at cuts c >= 64 (n = 8) and c >= 128 (n = 16),
   against >= n/2 at c <= 4; k*_F(S; 1e-2) for S = Sigma^(1/2) M Sigma^(1/2) <= 2 at the same cuts.
P4 (residual). No collapse: mean pairwise angle at depth L stays above 0.5 rad; per-layer gate disagreement stays
   above 10%; median N_surv(e^-1) >= n/2 at every L'; k*_F(M; 1e-2) >= n - 2.
P5 (Kelvin's gate-independent bound). Candidate weight-derived subspaces U_k (k = 1, 2, 4) at a cut, scored by the
   share of tr S (and of tr M) they capture, against the optimum (top-k of S):
   (a) unmasked product W_L ... W_{c+1}: top-k right singular vectors;
   (b) random masks: top-k eigenvectors of E_D J_D^T J_D, D_l iid Bernoulli(1/2) independent of the state;
   (c) worst case over masks: PR of the span of top right singular vectors over 1024 random mask sequences, and
       sup-gain ratio max_D ||J_D|| (over those, a lower bound) / median_x ||J_x||;
   (d) mask-averaged product with marginal gate probabilities, Dbar_L W_L ... Dbar_{c+1} W_{c+1} Dbar_c;
   (e) attractor path: J along one reference input's trajectory (gates of that path only).
   Predictions (plain, deep cuts as in P3): (a), (b) capture <= 2k/n + 0.1 of tr S (no better than chance);
   (c) PR >= n/2 and sup-gain ratio >= 10 at L' >= 32; (d) and (e) capture >= 0.9 of tr S at k = 1-2. Residual: no
   subspace with k < n captures >= 0.9 except the trivial k >= n - 1 cases. Verdict rule: "gate-independent bound
   exists" only if (a), (b) or (c) capture >= 0.9 at k <= 2 on all nets at the deep cuts.

## Part 3: deductive estimation (protocol)
Truth: plain MC with 2^24 paths per (net, sigma) (n = 8: also 2^26 at L = 64 for tails), float64, and for sigma = 0
the radial Rao-Blackwellised estimator E|x| * mean_u F(u) as a cross-check. Report its own standard error.
FLOPs: flopscope conventions as in work/arc-noisy/anet.py (matmul M N (2K - 1); elementwise 1; normal draw 16;
exp/log/erf 16; normal cdf 96, pdf 54); deductive methods counted by the same rules.
Estimators of E[output]:
- MC: plain sampling (input and noise draws). Simple samplers: antithetic (x, -x) with shared noise; RQMC (scrambled
  Sobol inputs, iid noise; 16 replicates). "Best simple sampler" = lowest measured MSE at equal FLOPs per config.
- K2: ARC's Gaussian closure (mean + covariance through ReLU in closed form, noise added to the diagonal), from the
  input (copied from work/arc-noisy, not edited there).
- ATT-oracle(l0): exact law at layer l0 (2^22 sampled states, an oracle) pushed through layers l0+1..L by the
  attractor-linear map J* (gates of one reference trajectory) plus a per-radius-bin Gaussian (K2) correction for the
  noise; l0 in {8, 16, 32, 48, 64, 96, 128}. Measures the depth l0(n, sigma, tol) after which the deep part needs no
  more than one weight-derived trajectory. Prediction P6: nMSE of ATT-oracle <= 1e-4 for l0 >= 48 (n = 8) and
  l0 >= 96 (n = 16) at sigma = 0 and 1e-3; l0 needed for 1e-4 grows roughly in proportion to n (ratio 16/8 within
  1.4-3).
- HYB(l0): MC for layers 1..l0 + the ATT deep part; at equal FLOPs its MSE beats plain MC by a factor that grows with
  L/l0 (prediction P7: >= 0.5 L/l0 at L = 256).
- Sample-free candidates for the shallow part (built after Part 2 and the ATT measurement, declared in an amendment
  before their main runs): at least K2 in polar form (radius integrated exactly per bin) and one gate-splitting
  mixture. Prediction P8: K2 from the input misses E[a_L] by nMSE >= 1e-2 at L >= 64 (plain, both n); no sample-free
  method of this lane reaches the best simple sampler at equal FLOPs at L = 64 for n = 16 (we expect a blocker in the
  shallow part, not in the depth).
- Tails: p_t by MC, by the polar deductive form (radius law at l0 times the attractor gain), and by the radial
  Rao-Blackwellised sampler; same FLOP accounting.
A "win" for a deductive estimator = lower MSE than the best simple sampler at equal FLOPs, on at least 3 of 4 nets.

## Part 4: trained nets
Trainability: an RNN config "trains" if its final accuracy (noise on) exceeds 5x chance (T = 64: 7.8%) on all seeds;
an MLP config "trains" if final nMSE <= 0.05 (noise on). Predictions: T = 32 trains to >= 50% accuracy; T = 64 trains
above 5x chance but below the T = 32 accuracy; residual sorting MLPs train at all depths (nMSE <= 0.05 at L = 64);
the plain MLP at L = 64 does not (nMSE >= 0.5), at L = 16 it does better than at L = 64.
On the trained nets, repeat: per-path N_surv over steps/layers (RNN: J = d h_T / d h_t), gate disagreement across inputs
per step/layer, k*_F(M) and k*_F(S), the gate-independent comparison of P5 (RNN: powers of W_hh unmasked, random
masks, marginal-gate product, one reference path), and K2 / MC / best simple sampler for E[logits] (RNN) and E[out]
(MLP) at equal FLOPs. Prediction P9: trained narrow nets do not collapse (they must carry the task's state), gate
disagreement stays >= 10% per step/layer, and K2 loses to MC (as on AlgZoo T16).

## Rules
- Every number in the report comes from a logged run in logs/ or runs/; per-net values reported, plus means.
- Deviations and new estimators go into dated amendments appended below, hashed before their runs.
- A GPT-6 review (codex-any exec) of the claims before the report is final.

## Amendment 1 (2026-10-29 ~22:25Z; before any Part 3 sampler or Part 4 result was looked at)
Seen since Part 1: Part 2 tables (collapse census, Jacobians), the smoke runs of the Part 3 scripts on one net
(n = 16, L = 64, seed 0, 2^18-path truth), the variance split of the random nets, cell counts for n = 8 / 16.
1. Header time: Part 1 was written ~21:45-21:57Z (hash 21:57:44Z), not ~22:20Z.
2. n = 8 plain nets: 39 of 64 die whole by L = 256 (sigma = 0), so per-net analyses use the first four seeds alive at
   L under sigma = 0 (L = 64: seeds 0, 2, 3, 4; L = 256: seeds 2, 4, 5, 7). This conditions on survival.
3. P5 scoring: after collapse the state is one-dimensional, so the tr S share is captured by any block not orthogonal
   to it; we add capM = tr(P_U M) / tr M (sensitivity only) and apply the P5 verdict rule to both scores.
4. Coordinator's request (~22:10Z): Part 3's headline is the non-collapsed nets (residual variant, trained nets);
   plain nets are reported as the easy case. For every net we report the shares of tr Var(output) explained by the
   input, its direction and its norm (one-way ANOVA, 4096 x 8).
5. Training deviations (GPU contention): RNNs use batch 4096, lr scaled by sqrt(4), warmup 4096, 65536 steps
   (2^28 sequences, as planned); the residual MLP at L = 128 was stopped to free the GPU; if the T = 64 run is cut
   by the 2.5 h VM timeout, its 32768-step checkpoint (learning rate already at its 0.01x floor) is the model.
6. Sample-free shallow part: K2 from the input, and a cell census (number of joint gate patterns of layers 1..l
   among 2^20 inputs) as the size of an exact cell representation; no gate-splitting mixture is built.
   HYB deep part: 8 log-radius bins, Stein-linearised K2 per bin (or the noiseless map of each bin mean, "lin").

## Amendment 2 (2026-10-29 ~22:30Z; compute trim, before any sampler comparison beyond N_eq = 256 on one net was seen)
The replicated sampler runs were ~10x slower than planned on the loaded host. Trim: R = 16 replicates (32 planned),
budgets N_eq in {256, 1024, 4096, 16384}, HYB cut l0 in {16, 32} (L >= 128) or {8, 16} (L = 64), samplers on two
seeds per configuration (residual first), the deep-part oracle (p3_att) on every truth. Tails (p3_tails.py):
q in {1e-3, 1e-4}, N_eq in {1024, 4096, 16384}, R = 16, estimators MC, RB (radius integrated exactly through
positive homogeneity; exact at sigma = 0, approximate with noise), HYB32-lin and HYB32-RB (plain only).

## Amendment 3 (2026-10-29 ~22:35Z; before any sampler result beyond N_eq = 1024 on one plain net was seen)
Add to the simple samplers a linear regression control variate on the input (E x = 0 is known):
MC-reg = y_bar - B_hat x_bar and RQMC-reg likewise, B_hat the least-squares coefficient of y on x from the same
samples (its ~n^2 per-sample cost is not charged; O(1/N) bias). The "best simple sampler" is the best of MC, ANT,
RQMC, MC-reg, RQMC-reg. Applies to p3_samplers (random nets, from the second net on) and p4_est (trained nets,
budgets N_eq >= 256, R = 16).

## Amendment 4 (2026-10-29 ~22:36Z; before any tail result was seen)
Deductive tail for the residual family: P(|h_L| > t) under the K2 output Gaussian N(mu, C) (exact ReLU moments),
computed by Imhof's (1961) formula for the generalised chi-square (checked against sampling of that Gaussian).
Scored against MC at the same FLOPs (relative MSE (1 - p) / (N p)).

## Amendment 5 (2026-10-29 ~22:45Z; after seeing K2's nMSE on the four residual n = 16, L = 64 nets and one sampler run)
Depth scaling of K2 on the residual family (p3_resdepth.py): L in {16, 64, 256, 1024}, n in {8, 16}, sigma = 1e-3,
seeds 0-3 (L = 1024: seeds 0-1; matrices drawn from a separate stream, see lib/nd.py), truth 2^20 paths (L = 1024)
or 2^21. Prediction P10 (written now, after seeing L = 64): K2's nMSE does not grow with L (ratio L = 1024 / L = 64
within 3x), because closure errors of different layers enter along independent random directions V_l.
Its cost grows linearly in L like MC's, so its advantage over MC at equal FLOPs stays roughly constant in L.
Also the share of tr Var(output) explained by the best linear / quadratic function of x (p3_linshare.py).

## Amendment 6 (2026-10-29 ~23:32Z; trained-net analysis plan, after the training finals were seen, before any analysis)
Training ran to the end (no checkpoint fallback). Seen: final accuracies / nMSE (RNN T=64: 29-44%; T=32: 33-41%;
sorting MLP residual L=64: nMSE 0.034-0.041, L=16: 0.074-0.087; plain L=64: 0.81, 1.14 and one diverged; plain
L=16: 0.32-0.41). Analysed models: the best seed per (architecture, size, sigma) by noise-on accuracy / nMSE
(scratch/p4_select.py). Structure (p4_analyze.py) on all of them; estimation (p4_est.py: truth 2^21 paths, K2
exact and Stein, MC, ANT, RQMC, MC-reg, RQMC-reg, HYB at l0 = depth/8, /4, /2; R = 16) on RNN T=64 (both
sigma), RNN T=32 sigma=0, MLP residual L=64 (both sigma), MLP plain L=16 sigma=0. The sampler queue for the
remaining plain random nets was stopped to free the CPU slot (plain samplers ran on n=16 L=256 seeds 0-1).

## Amendment 7 (2026-10-30 ~00:55Z; after the GPT-6 review, review/review_gpt6.md)
Corrections, recomputed where code changed:
1. p2_jac.py v2: M = E J^T J now includes every path with a nonzero Jacobian (noise can revive a path that once
   died; v1 excluded them), the random-mask PR ignores zero products, and the reference path is chosen among paths
   with a nonzero Jacobian at the cut. All plain-net Jacobian runs are redone (v1 outputs kept in runs/v1/).
2. gap_1 is recomputed with the same conditioning (m >= 2) for the radial and the angular exponent:
   0.2503 (n = 8), 0.0868 (n = 16) (scratch/theory_gap_v2.py).
3. Not amended before, recorded now: the main truth runs used 2^21-2^22 paths, not the 2^24 of Part 1; the
   noiseless (sigma = 0) arm of P6 was not run; several diagnostics use 2 nets, the plain sampler comparison 1.


# Part 2: follow-up on trained nets (coordinator, 2026-10-30 ~01:05Z), written ~01:20Z before any trained-net result
# of these analyses was seen (smoke tests ran only on the barely trained pilot models in train/pilot/)

Question: why does K2 fail on trained narrow nets, and is the failure concentrated (a few neurons or layers: a
scalable fix) or spread out (a blocker for Gaussian-plus-corrections families)? CPU only.

Nets: the four trained nets of section 4: residual sorters width 16, depth 64 (mlp_res_L64_s0, sigma = 0 and 0.01)
and RNNs width 16, T = 64 (rnn_T64_s0 sigma = 0, rnn_T64_s1 sigma = 0.01). Code: lib/k2mix.py, scratch/p5_*.py.

Step 1 (p5_locate.py, 2^20 sampled paths per net). Per (layer or step, unit): K2's predicted pre-activation mean,
variance and gate-on probability against sampled truth, (a) accumulated (K2 from the input) and (b) one-step (K2
from the exact moments of the previous state; its gate error is the non-Gaussianity of z). Concentration: share of
the summed |gate error| in the worst k pairs (k = 1, 2, 4, 8, ..., 128 of 1,024), per-layer and per-unit sums.
Where the output error arises: suffix nMSE E_out(l0) of K2 started from the exact Gaussian moment-match at l0.
P11 (sorters): spread. The worst 8 of 1,024 pairs carry < 20% of the summed accumulated |gate error|, no layer
carries > 20%, and E_out(l0) falls gradually (no single gap between grid points accounts for > 50% of the drop in
log E_out from l0 = 0 to the last grid point).
P12 (RNNs): concentrated in units, spread in time: the worst 4 units (summed over steps) carry > 50% of the summed
|gate error|, while the worst 8 (step, unit) pairs carry < 20%.

Step 2 (p5_split.py). Mixture closure: at each selected (layer, unit) pair every component splits into its on and
off pieces (exact truncated first two moments, gate fixed in each piece), never merged (up to 2^k components).
Selection: S1 = worst k pairs by accumulated gate error (primary), S2 = by one-step error, S0 = k random pairs
(control); k = 1, 2, 4, 8; exact ReLU moments (primary) and Stein. Scored on E[output] nMSE against an independent
2^21-path truth, on P(|output| > t) at q = 1e-3 and 1e-4 (Imhof per component), and against MC and RQMC at equal
FLOPs (R = 16).
P13 (the coordinator's criterion): refuted if no k <= 8 gives at least a 3x lower nMSE than K2. Prediction: refuted
on both sorters and both RNNs. Predicted k for a 10x gain: more than 8 on every net (none of k <= 8 reaches 10x).
Alternative we would not be surprised by: a 3x gain on an RNN at k = 8 if P12 holds strongly.
P14: at equal FLOPs every split closure with k <= 8 loses to RQMC by >= 30x in nMSE on every net.
P15: the split closure's tail estimate stays wrong by more than 2x (relative error^2 > 0.25) at q = 1e-4 for every
k <= 8 on every net.
Control: random splits (S0) give <= 1.5x over K2.

Step 3 (p5_rb.py, 2^22-path truth). Exact radius integration (positive homogeneity) for P(|output| > t), the
sorters' P(max_i |out_i - sort(x)_i| > t) and the RNNs' P(max_{j != y} logit_j - logit_y > t), q = 1e-3, 1e-4,
against MC and RQMC at equal FLOPs (1,024-16,384 paths, R = 16).
P16: on the two noiseless nets RB lowers the relative MSE by >= 3x against MC at 16,384 paths for every event at
q = 1e-4. On the noisy nets (sigma = 0.01, where RB is approximate) RB's large-N estimate is within 10% of the truth
at q = 1e-3, and its gain over MC is >= 3x.
Review: one GPT-6 review of the new report section before it is final.
