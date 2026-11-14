# Bits per layer, measured as well as we can: bits-vs-accuracy curves for ten layers of ARC 256 x 32 and eight of narrow-deep 8 x 256 (bit-accounting-arc, phase 1, 2 November 2026)

Status: FINAL (17:05Z). Lane bead pool-qxszj1.26 (target M1). Everything below is regenerated from the run files by
the scripts named with each table (results/ holds the generated tables). Labels: PROVEN, MEASURED, HEURISTIC. Numbers
marked (net 4000) come from one net. Reviewed by gpt-6-astra in three rounds (section 7).

## Summary for Kelvin

1. **The curves.** For each measured layer (ten of ARC's 32, eight of narrow-deep's 256), one encoder per start (bit
   price as an input) gives the bits needed for every output accuracy; the count at each accuracy is the better of two
   such encoders (the reference and look-ahead starts), so different accuracies can come from different encoders.
   ARC 256 x 32, a code of the layer that one ReLU layer can compute (local), bits for 10% / 1% / 0.1% output error
   (nMSE 0.1 / 0.01 / 0.001; median over the four nets):

   | layer | 0 | 2 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 31 | output |
   |---|---|---|---|---|---|---|---|---|---|---|---|
   | 10% | 630 | 358 | 230 | 111 | 74 | 39 | 27 | 21 | 19 | 19 | 19 |
   | 1% | 1139 | 849 | 654 | 409 | 289 | 205 | 148 | 103 | 89 | 95 | 89 |
   | 0.1% | 1625 | 1302 | 1062 | 767 | 583 | 469 | 372 | 293 | 266 | 307 | 266 |

   (output = the lower envelope of the base and look-ahead codes of the net.)
   Figures: figs/profile_arc.png (bits vs layer, every net), figs/curves_arc_s4000.png (every layer's curve),
   figs/slopes_arc_s4000.png, figs/spectra_arc_s4000.png. MEASURED.
2. **Bits fall steeply over the first ~16 layers, then sit on the output's curve.** Layer 0 needs 26-38 times the
   output's bits at 10% error and 10-14 times at 1%; layer 8 about 5 and 4 times; layer 16 about 2; from layer ~24 on
   1.0-1.2 times (ratios to the output's curve, the lower envelope of the base and look-ahead codes of the net). **What tracks these
   numbers is the layer's future-weighted spectrum** (how strongly the output depends on each direction of the layer,
   eig of the whitened E[J^T J]): the learned count is 0.69-1.01 of reverse water-filling over it, a closed form with no
   training, at 1% error on every measured layer and net (0.51-0.83 at 10%). The ratios are MEASURED; reading the
   spectrum as the cause is HEURISTIC.
3. **Cost of precision.** Each halving of the output's RMS error costs, at layer 0, 270-300 bits, more than the 256
   input dimensions (MEASURED). A candidate reason (HEURISTIC): at these accuracies the output error falls more slowly
   than the state error (exponent 0.94-0.98 for isotropic perturbations), which would put about 270 bits per halving on
   256 directions. Near the output the cost per halving grows with precision (layer 31: 15 bits near 30% error, ~150
   near 0.2%) as more directions are used; the codes' active latents are 0.9-1.6 times the relevance lane's linear
   feature count N(l, eps) (a different quantity, under a Gaussian convention; context only).
4. **Narrow-deep 8 x 256 collapses to one number.** From layer 8-32 on, every measured layer's curve is the output's:
   1.4 bits for 10%, 2.9 for 1%, 4.5 for 0.1% (six nets; 0.4-0.5 bit below the one-Gaussian-number bound), and its
   unrestricted codes at layers 0, 16, 64 and 255 match the output's curve (the decoder inverts up to 256 layers). A local code of the 8 inputs needs 9.5 / 23 / 37 bits. MEASURED.
5. **Your single variable-rate encoder approaches dedicated single-price encoders with more training, except at the
   cheap end of layer 0.** Layer 0: within 1% of dedicated models at 6.5% and 0.2% error. Layer 31: three dedicated
   models of 12000 steps each need 6-17% fewer bits than one variable-rate model of 12000 steps, and it comes within
   -1% to +5% of them at 48000 steps (local; unrestricted -6% to +7%). At the cheap end of layer 0 (40-73% error)
   four of seven checks fail, by up to 10x (4 bits for a dedicated model against 43 for the curve at 73% error), and
   the dedicated models disagree with each other there too. MEASURED.
6. **Is it the network or us?** We took a registered subset (net 4000; layers 0, 16, 31; local and unrestricted; 10%
   and 1% error) through doublings of steps and depth, other starts and your backward curriculum (A18-A22). By the
   registered rule (each last doubling < 3% with a matched partner, and no start lower by 3% or more):
   - converged: layer 0 local at 10% and 1% error (573 and 1065 bits; registered count 575 and 1084);
   - not converged, still falling: layer 16 local at 1% (170 -> 164 -> 157 -> 150 bits with 2x, 4x, 8x steps:
     -4%, -4%, -5% per doubling); layer 31 local at 1% (90 -> 78 with depth 6 and 4x steps -> 75 with 8x: -3% at the
     last doubling); the unrestricted codes at 1% at layers 16 and 31 (111 -> 85 -> 81 and 88 -> 73 -> 70 bits with
     depth 6 and 4x, then 8x steps: -5% and -4% at the last doubling);
   - unverified (a matched partner was lost with bitarc-7 or not run): at 10% error every remaining cell, though none
     moved by 3% or more at its last doubling (layer 16 local 27 -> 24, layer 31 local 15.4 -> 15.0, unrestricted
     17.8 -> 17.4, 16.4 -> 15.4, 15.5 -> 14.7 at layers 0, 16, 31);
   - not reached: layer-0 unrestricted at 1% error.
   So at 10% error the registered counts are 0-11% above the lowest found (layer 16 local the largest), and at 1%
   error 2% (layer 0) to 14-37% (layers 16 and 31) above it, with those lowest values still falling 3-6% per doubling:
   at layers 16 and 31 the 1% counts are upper bounds whose converged values we do not know. Your backward curriculum
   lowers no cell (section 4.2). MEASURED.
7. **Where to trust the counts** (ARC; "upper bound" = what we found, the class may need fewer bits; the convergence
   verdicts are for net 4000 at 10% and 1% error only, the other checks as named):

   | | 30-10% error | 3-1% error | 0.3-0.1% error |
   |---|---|---|---|
   | local, layers 0-8 | upper bound: the two informed starts differ by 10-45%; above 30% error at layer 0 dedicated models and the starts disagree by large factors; random starts find 25-35% cheaper codes at 20-30% error at layer 8. Layer 0 at 10%: converged | layer 0 at 1%: converged (A18); layers 2-8 not checked | layer 0: 2x steps, 2x decoder depth and restart 2 within 1% (C3); not in the subset |
   | local, layers 12-20 | layer 16 at 10%: 27 -> 24 bits with 8x steps, last doubling < 3% (depth partner missing) | layer 16 at 1%: not converged, -5% per doubling at 8x steps | layer 16: C3 within 1% (base); not in the subset |
   | local, layers 24-31; the output | layer 31 at 10%: 15.4 -> 15.0, last doublings <= 1% | layer 31 at 1%: not converged, 90 -> 75 bits, -3% at the last doubling; layer 24 and the output cell -9% and -17% with 4x steps | not in the subset; -18 to -24% with 4x steps or depth 6 (layers 24, 31, the output cell) |
   | unrestricted | at 10%: the last doubling moves layers 0, 16, 31 by 1-2% (depth and width partners missing at 8x steps); shortfall to the best-known envelope 5-21% toward the input | at 1%: not converged at layers 16 and 31 (-5% and -4% at 8x steps); layer 0 never reaches 1% | layers 0-24 stop short |
   | narrow-deep local, layer 0 | search-limited: a random start needs 2-3x fewer bits at 30% error | starts differ by 20-30% | starts differ by 13-21% |
   | narrow-deep, layers 32-255 and the output | every measured class, start and restart within 0.1 bit | same | same |

## 1. What is measured (PREREG Part P1, amendments A1-A22)

- The code at layer l: z = g(lambda) nu(h_l) + eps, eps ~ N(0, I) fixed; the encoder chooses nu and, through the gain
  g, each latent's signal-to-noise ratio. Your point: fixed noise with a chosen power is the same channel as fixed
  power with a chosen noise; only each latent's SNR matters. The decoder maps z to a state at layer l and the real rest
  of the net (fp32) computes the output; distortion = output nMSE E|a_hat - a|^2 / tr Var a (the mean error included)
  on 32768 held-out inputs (the same inputs for every code of a net).
- The rate: E[log2 q(z|h) - log2 p(z)] with a learned prior p (a unit lower-triangular map, then an 8-component mixture
  per latent), bits per sample on the held-out inputs: an upper bound on the information I(h_l; z) the code carries
  (gap = KL(code marginal || p)).
- The output's curve: every code of every layer is also a code of the output (its rate bounds I(a; z) from above and
  f(h_hat) reconstructs a), so every measured curve is an upper bound on the output's rate-distortion function R_a(D)
  and, PROVEN, every layer's true count in any class is at least R_a(D). Three output baselines are named apart: the C1
  baseline (the output cell's own base code, as registered), the table envelope (the lower envelope of the base and
  look-ahead runs of every layer and the output cell; the "x output" ratios use it, and since it includes the layer's
  own curve those ratios are >= 1 by construction) and the best-known envelope (all runs, including the longer and
  deeper ones). That the unrestricted count equals R_a at every layer, and is monotone across layers, needs more: the
  wanted reconstruction must be reachable through the remaining layers (the image of the rest of the net can be
  nonconvex), so an unrestricted shortfall can be structural as well as a decoder or training limit.
- Each run's distortion is normalised by one output variance per net (A20).
- One model per (net, layer, class) covers every accuracy: the bit price lambda is an input (every lambda-dependent
  parameter is piecewise linear in log10 lambda over [-6, 0]); each step draws 32 lambdas and normalises each one's
  Lagrangian by its running mean, so cheap and expensive codes are trained alike.
- Classes. Local: one ReLU layer of width n (256 or 8), then affine; for each fixed lambda the encoder is exactly in
  this class. Unrestricted: an MLP of width 4n (1024; 256 for narrow-deep), depth 3, on h_l and the exact output (any
  function of h_l is allowed and the output is one).
- Starts (your identity-anchored start). The reference start: the analytic linear reference code, the eigen-directions
  of the future-weighted Jacobian Gram G = E[J^T J] (J = da/dh_l, whitened) with reverse-water-filled SNR capped at 2^16
  (at a high budget a scaled identity on those directions), decoded by its linear MMSE inverse; the bit price rises over
  the first 30% of training and each new price starts from the previous one, so directions are shed least important
  first. The look-ahead start ('h1', A3): the hidden layer computes h_{l+1} and the code is the reference code of
  h_{l+1}. The backward curriculum ('bwd', A19): your proposal, the next measured layer's trained code composed with the
  layer map and distilled into the local class (section 4). Random starts with a high encoder learning rate (A15, A17)
  as a search check. Unrestricted codes start as the output's own linear analog code. Aligned with the sibling lane's
  phase-1 method (linear-reference start, water-filled capped power, per-lambda normalisation, the per-lambda spot
  check), except the SNR cap (2^16, not 256: 256 stopped narrow-deep codes at nMSE 0.004, A1), a wider lambda range and
  12000 base steps.
- Curves: R(eps) = bits needed for nMSE eps, the lower convex hull of a run's evaluated (nMSE, rate) points
  (achievable by time-sharing any two of its codes; A20). Slope = bits per halving of the output RMS error = 2 dR / dlog2(1/eps);
  for a Gaussian source at high resolution it is the number of directions being resolved. The local count of a net is
  the better of its starts at each accuracy.
- Training: 12000 steps of 1024 (ARC) or 2048 (narrow-deep) fresh inputs, Adam; base restarts 0-1 at ARC layers 0
  (unrestricted) and 31 and the output and narrow-deep everywhere (restarts 0-2), restart 0 elsewhere (A8). The
  capacity checks double the depth of the decoder and conditioning networks; for local codes the encoder is the class
  (one ReLU layer) and is not enlarged.

## 2. ARC 256 x 32

Local code (the better of the two starts), median over the four nets [min-max]; x output = local / the output's curve;
slope = bits per halving of the output RMS error (results/P1_tables_final.md, scratch/report_tables.py):

| layer | 10% error | 1% | 0.1% | x output, 10% | x output, 1% | slope near 2% / 0.2% | unrestricted 10% / 1% (net 4000) |
|---|---|---|---|---|---|---|---|
| 0 | 630 [575-692] | 1139 [1084-1206] | 1625 [1577-1687] | 34 [26-38] | 12.5 [9.8-13.5] | 302 / 274 | 18 / not reached |
| 2 | 358 [308-422] | 849 [780-907] | 1302 [1240-1355] | 18 [16-23] | 9.1 [7.4-10.1] | 277 / 267 | not run (A8) |
| 4 | 230 [192-276] | 654 [573-700] | 1062 [972-1106] | 11 [9.8-15] | 6.8 [5.7-7.9] | 251 / 249 | not run (A8) |
| 8 | 111 [82-132] | 409 [360-454] | 767 [716-806] | 5.1 [4.9-7.1] | 4.2 [3.6-5.2] | 195 / 213 | 17 / 185 |
| 12 | 74 [44-97] | 289 [244-333] | 583 [523-632] | 3.3 [2.9-5.0] | 2.8 [2.7-3.8] | 151 / 184 | not run (A8) |
| 16 | 39 [27-62] | 205 [170-271] | 469 [416-552] | 2.1 [1.8-2.4] | 2.1 [2.0-2.6] | 120 / 171 | 16 / 111 |
| 20 | 27 [20-42] | 148 [125-203] | 372 [343-457] | 1.5 [1.3-1.5] | 1.6 [1.5-1.8] | 94 / 154 | not run (A8) |
| 24 | 21 [17-31] | 103 [94-154] | 293 [273-383] | 1.1 [1.1-1.2] | 1.2 [1.1-1.2] | 64 / 130 | 16 / 84 |
| 28 | 19 [16-28] | 89 [84-123] | 266 [260-333] | 1.0 | 1.0 | 54 / 127 | not run (A8) |
| 31 | 19 [15-27] | 95 [90-131] | 307 [290-365] | 1.0 | 1.1 [1.0-1.1] | 60 / 155 | 15 / 88 |
| output (table envelope) | 19 [15-27] | 89 [84-123] | 266 [260-333] | 1 | 1 | 54 / 127 | Gaussian bound 25 / 171 / 439 |

The "x output" ratios divide by the table envelope, which includes the layer's own curve (ratios >= 1 by construction; at layers 28-31 the layer itself is
the envelope at most accuracies). The unrestricted column is net 4000's base.

The nets differ by up to 2x at mid layers (layer 12 at 10% error: 44 to 97 bits; layer 16: 27 to 62); net 4001 needs
the most at every layer and also has the costliest output (27 bits at 10%); net 4000 needs the least.

Against your analytic linear reference (the same held-out inputs, realised through the net): the local code needs
0.91 of its bits at layer 0 and 10% error (0.99 at 0.1%), 0.78 at layer 8, 0.83 at layer 16, 0.75 at layer 31; near
the output the learned codes save 25-40% at nMSE 0.03-0.01 (the output is sparse and nonnegative). At layer 0 the
local code is close to linear: the learned nonlinearity buys 1-9%.

### What sets a layer's bits: the future-weighted spectrum
For each layer, the eigenvalues lam_i of the whitened future Gram E[J^T J] (J = da/dh_l), in units of the output
variance V (figs/spectra_arc_s4000.png), say how strongly the output depends on each direction of the layer. The
closed-form reverse water-filling rate over them, R_wf(eps) = sum_i max(0, 1/2 log2(lam_i / theta)) with
sum_i min(lam_i, theta) = eps V, is a linear-Gaussian prediction that needs no training. Net 4000 (all nets:
results/spectrum_table.md, scratch/spectrum_table.py):

| layer | tr G / V | participation ratio | directions above the water level, 10% / 1% | R_wf, 10% / 1% | learned local, 10% / 1% | learned / R_wf |
|---|---|---|---|---|---|---|
| 0 | 16.0 | 91 | 241 / 252 | 730 / 1146 | 575 / 1084 | 0.79 / 0.95 |
| 4 | 5.3 | 23 | 160 / 210 | 302 / 640 | 192 / 612 | 0.64 / 0.96 |
| 8 | 3.0 | 11 | 92 / 162 | 145 / 396 | 82 / 360 | 0.57 / 0.91 |
| 16 | 1.7 | 5.5 | 39 / 98 | 53 / 199 | 27 / 170 | 0.51 / 0.85 |
| 24 | 1.2 | 3.0 | 21 / 65 | 25 / 119 | 17 / 94 | 0.67 / 0.78 |
| 31 | 1.0 | 2.2 | 17 / 79 | 19 / 128 | 15 / 90 | 0.81 / 0.70 |

At 1% error the learned local count is 0.70-0.95 of this closed form on net 4000 and 0.69-1.01 over every measured
layer and net; at 10% error 0.51-0.83 (there the net is more forgiving than its linearisation and the learned code uses
the output's non-Gaussian shape). The profile of bits over depth follows, within these factors, the profile of the
spectrum (a HEURISTIC reading: the closed form predicts, it does not prove that the spectrum fixes the optimum): the
output depends on ~240 directions of layer 0 with a total linearised sensitivity 16 times its variance, and on a few directions of layer
31 with sensitivity equal to its variance. MEASURED (the closed form is HEURISTIC as a predictor; the ratios are
measured).

## 3. Narrow-deep 8 x 256

| layer | 10% error | 1% | 0.1% | x output, 10% | slope near 2% / 0.2% | unrestricted 10% / 1% |
|---|---|---|---|---|---|---|
| 0 | 9.5 [5.0-13.8] | 23 [19-29] | 37 [33-43] | 7.6 [4.0-12] | 8.6 / 8.5 | 1.4 [0.8-1.5] / 2.9 [2.0-3.2] |
| 8 | 1.4 [0.8-1.7] | 3.0 [2.2-7.4] | 5.8 [4.0-12.5] | 1.0 [1.0-1.2] | 1.0 / 2.3 | not run (A18 cut) |
| 16 | 1.5 [0.8-1.7] | 3.1 [2.0-5.7] | 5.1 [3.6-9.6] | 1.0 [1.0-1.4] | 1.0 / 1.4 | 1.4 [0.8-1.5] / 2.9 [2.0-3.2] |
| 32-255 | 1.4 [0.8-1.5] | 2.9 [2.0-3.2] | 4.5 [3.6-4.8] | 1.0 | 0.93 / 0.99 | 1.4 / 2.9 (64, 255) |
| output | 1.3 [0.8-1.5] | 2.9 [2.0-3.1] | 4.5 [3.6-4.8] | 1 | 0.93 / 1.0 | Gaussian bound 1.7 / 3.3 / 5.0 |

The local count is the better of the reference start (base at 0, 8, 64, 192, 255; cut at 16, 32, 128 by A18) and the
look-ahead start (every layer). From layer 32 on, every measured net, class, start and restart gives the output's curve within
0.1 bit, and every code uses one latent: the state is one number times a fixed direction (the relevance lane's N = 1).
Nets 216 and 223 collapse by layer 8, nets 203 and 221 later (layer 8 at 0.1% error: 9-18 bits against 4.7).

## 4. Is it the network or us?

### 4.1 The convergence subset (A18): ARC net 4000, layers 0, 16, 31, both classes, at 10% and 1% error
Rule (registered): a count is converged when doubling the training steps, and separately the depth (unrestricted: also
the width), changes it by < 3%, and, for local cells, no other start (reference, look-ahead, backward, random with
encoder learning rates 3e-4 and 1e-3) is lower by 3% or more. Doubling partners must match the run in every other
setting (A20); a missing partner leaves that axis unverified. The doublings are checked on the best run among the
reference, look-ahead and backward starts (restarts averaged); another run within 3% of it is reported, one lower by
3% or more makes the cell search-limited. results/conv_subset.md (scratch/conv_subset.py) lists every run;
figs/convergence_arc_s4000.png shows each start's count against steps. Rounds: round 1 (A18), round 2 (8x steps,
A22, after a preemption lost the first attempt).

| layer | class | error | registered count | lowest found (run) | last doubling of the run giving it | verdict |
|---|---|---|---|---|---|---|
| 0 | local | 10% | 575 | 573 (h1c2s2) | steps -0% (h1c2 -> h1c2s2); depth -0% (h1s2 -> h1c2s2) | converged |
| 0 | local | 1% | 1084 | 1065 (h1c2s2) | steps -2% (h1c2 -> h1c2s2); depth -0% (h1s2 -> h1c2s2) | converged |
| 0 | unres | 10% | 17.8 | 17.4 (cap2steps4) | steps -1% (cap2steps2 -> cap2steps4); depth -0% (steps4 -> cap2steps4); width unverified | unverified: no matched width partner |
| 0 | unres | 1% | - | - | - | not reached |
| 16 | local | 10% | 27 | 24 (h1s8) | steps -3% (h1s4 -> h1s8); depth unverified | unverified: no matched depth partner |
| 16 | local | 1% | 170 | 150 (h1s8) | steps -5% (h1s4 -> h1s8); depth unverified | not converged: steps -5% per doubling |
| 16 | unres | 10% | 16.4 | 15.4 (cap2steps8) | steps -1% (cap2steps4 -> cap2steps8); depth unverified; width unverified | unverified: no matched depth, width partner |
| 16 | unres | 1% | 111 | 81 (cap2steps8) | steps -5% (cap2steps4 -> cap2steps8); depth unverified; width unverified | not converged: steps -5% per doubling |
| 31 | local | 10% | 15.4 | 15.0 (cap2steps8) | steps -0% (cap2steps4 -> cap2steps8); depth unverified | unverified: no matched depth partner |
| 31 | local | 1% | 90 | 75 (cap2steps8) | steps -3% (cap2steps4 -> cap2steps8); depth unverified | not converged: steps -3% per doubling |
| 31 | unres | 10% | 15.5 | 14.7 (cap2steps8) | steps -1% (cap2steps4 -> cap2steps8); depth unverified; width unverified | unverified: no matched depth, width partner |
| 31 | unres | 1% | 88 | 70 (cap2steps8) | steps -4% (cap2steps4 -> cap2steps8); depth unverified; width unverified | not converged: steps -4% per doubling |

Run names: h1 = look-ahead start, c2 = depth 6, s2 / s4 / s8 = 2x / 4x / 8x steps (cap2steps8 = depth 6 with 8x
steps). The counts that are not converged are still falling 3-5% per doubling at 8x steps and depth 6; their converged
values are below the lowest found, by an amount we cannot bound from these runs. At 10% error every cell moved by at
most 3% at its last doubling, but at 8x steps the matched depth (and, for unrestricted codes, width) partners were not
run (the width partners at 4x were lost with bitarc-7), so the rule leaves them unverified.

### 4.2 Your backward curriculum (A19)
Implementation (scripts/run_vr.py, --init bwd): the hidden layer computes h_{l+1} (as in the look-ahead start); the
readout is first the least-squares fit of the later code's latents on h_{l+1}; then the whole one-ReLU-layer encoder
(every price anchor) is distilled for 3000 steps onto the later code's signal g * nu, weighted by its gains, which
projects the composed code onto the local class (gain-weighted R^2 0.98-1.0 at six of seven price anchors on the
layer-28 test); the gains and the prior are copied; the decoder starts at the least-squares linear map from the new
latents to the layer's state; no lambda annealing, so the later code's per-price structure is kept. Chain: layer 31's
best code (depth 6, 4x steps) starts layer 28, then 24, 20, 16, 12, 8, 4, 2, 0, each from the one before. This is an
approximate, projected warm start, not an exact composition: the one-ReLU-layer class cannot hold the later code
composed with the layer map, the fit R^2 is measured on its fitting sample at the anchors only, the gain function
differs by the 1% between the two layers' output-variance estimates (review r2 #10), and the source code is itself
not converged at 1% error. Its results below cannot rule out an exact curriculum from converged later codes. A
candidate reason it loses precision down the chain (HEURISTIC): the latents become features of ever later layers, so
the decoder has to invert more layers, the same step that limits our unrestricted codes at early layers.

| layer | backward start, 10% / 1% / 0.1% | better of the other two starts | change |
|---|---|---|---|
| 28 | 15.8 / 89 / not reached | 15.7 / 84 / 260 | +1% / +7% |
| 24 | 19.6 / not reached / not reached | 16.9 / 94 / 281 | +16% / - |
| 20 | 42 / not reached / not reached | 19.7 / 125 / 357 | +116% / - |
| 16 (one step from layer 20's look-ahead code, A21) | 28 / not reached / not reached | 27 / 170 / 431 | +3% / - |
| 0 (one step from layer 2's look-ahead code, A21) | not reached / not reached / not reached | 575 / 1084 / 1577 | - |

The chain stopped after layer 20: the VM running it (bitarc-7, spot) was preempted at ~14:32Z and the rest of the chain
was not rerun (A22), because the one-step variant (A21) had already answered at layers 16 and 0.

The backward start lowers no cell it has reached: it matches the look-ahead start at 10% error one or two layers below
its source (layers 28, 16) and loses precision from there (no 1% error at layers 24, 20, 16; no better than 60% error at
layer 0, at any rate).

### 4.3 Consistency with the noiseless-net orders (A19 flags; results/consistency.md)
Every code of any layer is also a code of the output, so no count can be below the output's true curve (PROVEN). If
in addition the wanted reconstructions are reachable through the remaining layers, the unrestricted counts all equal
it (R_l <= R_l' for l < l', and R_l >= R_a). Measured departures (tolerance 3% and 0.5 bit, as in C3), read as
undertraining, a decoder limit, or a structural limit of reachability:
- the output cell's own code sits above the best-known envelope of its net (all runs) by up to 55% (net 4000: +5%,
  +37%, +55% at 10%, 1%, 0.1% error; nets 4001-4003: +11-22% at <= 1%); the envelope comes from the layer-31 depth-6
  8x-steps runs (net 4000) and the layer-28 base (nets 4001-4003);
- unrestricted base counts rise toward the input between consecutive measured layers (net 4000 at 10% error: 17.8,
  17.1, 16.4, 15.9, 15.5 bits at layers 0, 8, 16, 24, 31; at 1%: not reached, 185, 111, 84, 88), and sit 5-21% (10%)
  and 20-163% (1%) above the best-known envelope;
- narrow-deep: every unrestricted layer and the output cell agree within 0-6%.

### 4.4 Training length, capacity and restarts (C3 and the added runs), ARC net 4000
results/conv_table.md (scratch/conv_table.py); bits, change against the base:

| layer | class | run | 10% | 3% | 1% | 0.3% | 0.1% |
|---|---|---|---|---|---|---|---|
| 0 | local | base | 642 | 905 | 1126 | 1363 | 1577 |
| 0 | local | restart 2 / 2x steps / 2x depth | +0 / +1 / +0% | -0 / +0 / -0% | -0 / +0 / -0% | +0 / +0 / +0% | +0 / -0 / +0% |
| 0 | local | look-ahead start; 2x steps; depth 6; depth 6 + 2x steps | 575; 573; 574; 573 | 849; 842; 849; 842 | 1084; 1065; 1084; 1065 | 1365; 1314; 1364; 1313 | not reached; 1552; not reached; 1551 |
| 0 | unrestricted | base / 2x steps / 2x depth | 18.4 / -1 / -0% | 64 / -12 / -6% | not reached | | |
| 16 | local | base | 32 | 97 | 185 | 306 | 431 |
| 16 | local | look-ahead start; 2x; 4x; 8x steps | 27; 26; 25; 24 | 84; 81; 78; 75 | 170; 164; 157; 150 | 298; 290; 279; 265 | 432; 423; 410; 395 |
| 16 | unrestricted | base; 2x; 4x steps; 2x depth; depth 6 + 2x, 4x, 8x steps | 16.4; 16.0; 15.8; 16.1; 15.8, 15.6, 15.4 | 42; 41; 40; 41; 40, 40, 39 | 111; 94; 89; 96; 89, 85, 81 | not reached | not reached |
| 31 | local | base | 15.4 | 43 | 90 | 181 | 299 |
| 31 | local | restart 2 / 2x steps / 2x depth | +0 / -1 / -1% | -0 / -4 / -4% | -0 / -6 / -5% | +0 / -11 / -8% | +0 / -11 / -6% |
| 31 | local | 4x steps (A9) | 15.1 (-2%) | 40 (-5%) | 81 (-10%) | 150 (-17%) | 244 (-18%) |
| 31 | local | depth 6 + 2x steps (A20 partner) | 15.2 (-2%) | 40 (-6%) | 81 (-10%) | 152 (-16%) | 248 (-17%) |
| 31 | local | depth 6 + 4x steps (A11) | 15.1 (-3%) | 40 (-8%) | 78 (-14%) | 141 (-22%) | 229 (-24%) |
| 31 | local | depth 6 + 8x steps (A16, A22) | 15.0 (-3%) | 39 (-9%) | 75 (-16%) | 134 (-26%) | 215 (-28%) |
| 31 | unrestricted | base; 2x; 4x steps; 2x depth; depth 6 + 2x, 4x, 8x steps | 15.5; 15.3; 15.1; 15.3; 15.1, 14.9, 14.7 | 43; 41; 40; 41; 39, 38, 37 | 88; 82; 78; 84; 78, 73, 70 | 169; 154; 142; 162; 144, 133, 124 | 274; 244; 225; 263; 231, 213, 198 |
| 24 | local | base / 4x steps (A14) | 17.1 / -1% | 48 / -5% | 99 / -9% | 191 / -13% | 291 / -7% |
| output | out | base / restart 2 / 4x steps (A14) | 15.5 / +0 / -3% | 44 / -0 / -10% | 96 / -0 / -17% | 192 / +0 / -19% | 308 / +0 / -18% |

### 4.5 Your single variable-rate encoder against dedicated models (C7)
results/c7_table.md (scratch/c7_table.py). Layer 0 local, net 4000:

| bit price t | dedicated model | variable-rate curve at its nMSE, same start | look-ahead start's curve | verdict |
|---|---|---|---|---|
| -5 | 1447 bits, nMSE 0.0018 | 1460 (+1%) | 1491 | pass |
| -3.5 | 733 bits, nMSE 0.065 | 741 (+1%) | 671 | pass |
| -2.75 (A12) | 273 bits, nMSE 0.42 | 277 (+2%; adjacent-point rule 285) | 174 | pass (fail under the A4 rule) |
| -2.5 (A12) | 121 bits, nMSE 0.69 | 76 (adjacent-point rule 138) | 2.6 | pass (fail under the A4 rule) |
| -2.25 (A12) | 20 bits, nMSE 0.69 | 75 (138) | 2.6 | fail |
| -2 | 4.0 bits, nMSE 0.73 | 43 (117: a gap, 3.3 bits at 0.785, then 116 at 0.734) | 1.7 | fail |

(Variable-rate bits by time-sharing over the convex hull of its points, A20; the registered A4 rule interpolated
between adjacent Pareto points, given in brackets where it differs.)

Dedicated models from the look-ahead start (A12), against the look-ahead start's variable-rate curve:

| bit price t | dedicated model | variable-rate curve at its nMSE, same start | verdict |
|---|---|---|---|
| -2.75 | 169 bits, nMSE 0.40 | 188 (+11%) | fail |
| -2.5 | 66 bits, nMSE 0.57 | 66 (+1%; A4 rule 92) | pass (fail under the A4 rule) |
| -2.25 | 15 bits, nMSE 0.56 | 73 (A4 rule 96) | fail |

So above ~30% error at layer 0 the variable-rate curves can be far above what dedicated models find (15 bits for nMSE
0.56 where the curves need 73-75), and the dedicated models disagree with each other (20 and 121 bits for nMSE 0.69
from the reference start): that region is not certified for any start, and the true cost there may be several times
below our curves. At 6.5% error and below the variable-rate curve and dedicated models agree within 1% (layer 0).
The layer-31 comparison below is at unequal training budgets (three dedicated models of 12000 steps each against one
variable-rate model of 12000 to 48000 steps).

Layer 31 local (dedicated models at 12000 steps; the variable-rate model at 12000 / 24000 / 48000 steps):

| bit price t | dedicated model | variable-rate, 12000 steps | 24000 | 48000 | verdict (registered: 12000) |
|---|---|---|---|---|---|
| -5 | 262 bits, nMSE 0.0009 | 317 (+21%) | 284 | 259 (-1%) | fail |
| -3.5 | 67 bits, nMSE 0.0126 | 79 (+18%) | 73 | 70 (+5%) | fail |
| -2 | 11.2 bits, nMSE 0.127 | 11.9 (+6%) | 11.6 | 11.5 (+3%) | fail |

Layer 31 unrestricted (same setup): dedicated 252 / 65 / 11.2 bits at nMSE 0.0009 / 0.012 / 0.124 against the
variable-rate model's 293 / 80 / 12.3 after 12000 steps (+16 / +23 / +10%), 261 / 74 / 12.0 after 24000 and
238 / 69 / 11.8 after 48000 (-6 / +7 / +5%). Narrow-deep net 203 layer 0 local: dedicated 50 / 28 / 4.0 bits at nMSE
0.0007 / 0.016 / 0.48 against the variable-rate curve's 51 / 29 / 5.9 (-0 / -2% pass; -32% fail at the cheap end, where
the look-ahead start's curve needs 2.6 bits).

### 4.6 Search beyond the informed starts (A13, A15, A17)
A 10x encoder learning rate from the look-ahead start does not lower ARC layers 0, 8 and 16 (-5 to +7%), nor from the
reference start at layer 8 (+0 to +3%). Random starts with a 10-33x encoder learning rate find cheaper codes at >= 20%
error where they find anything: ARC layer 8, 6-7 against 9 bits at 30% error and 22-30 against 30 at 20%; ARC layer 16,
27 bits at 10% (the look-ahead start's 27); narrow-deep layer 0, 2-3x fewer bits at 30% error (0.9-2.5 against
2.1-6.3) and half at 10% on one net (2.6 against 5.0). They do not reach 1% error, and at ARC layer 0 they reach
nothing (nMSE > 0.5 at every price). So the low-precision counts at early layers are search-limited, and below 10%
error the informed starts are the best we have. These runs do not enter the registered count (the better of the two
informed starts); the export lists the lowest individual run of every cell as bits_best_known with its run name (for
example narrow-deep net 217, layer 0 at 10% error: 2.5 bits from a random start against the registered 5.0).

### 4.7 The planted toy (C5): bits at nMSE 0.1 / 0.01 / 0.001, known rate in brackets

| k | local, random start (registered) | local, reference start (A6) | unrestricted |
|---|---|---|---|
| 4 [6.5 / 13.2 / 19.8] | 7.8 / not reached / not reached | 6.8 / 13.4 / 20.1 | 6.8 / 13.4 / 20.1 |
| 16 [25.2 / 51.8 / 78.3] | 36.3 / not reached / not reached | 25.7 / 52.2 / 78.8 | 25.7 / 52.2 / 78.8 |
| 64 [80.1 / 186.3 / 292.6] | 203 / not reached / not reached | 82.0 / 188.2 / 294.5 | 82.3 / 188.4 / 295.1 |

Both informed starts are the toy's optimum (a linear Gaussian future), so their pass shows that training, the prior
and the rate estimate keep a correct code (within 0.2-1.9 bits); the random-start failure shows that training refines
a start and does not search the local class (encoder learning rate 3e-5, chosen for precision).

### 4.8 The other certificates
- C1 (unrestricted reaches the output's curve; registered baseline: the output cell): layer 31's unrestricted base is
  within 1% of the output cell at 10% error and 6-12% below it at 1% and 0.1% (the output cell is the undertrained one);
  against the best-known envelope it is 5% / 25% / 38% above (net 4000). At layers 0-24 the shortfall grows toward the
  input and with precision (section 4.3). Post-hoc decoder diagnostic (not registered; scripts/refine_posthoc.py): at
  260-355 bits the layer-0 unrestricted code predicts the output to nMSE 0.003-0.004 (its output-space head), while the
  plain decoder gives 0.055-0.11 and 1000 steps of preimage refinement 0.007-0.020: the information is there; turning
  it back into a layer-0 state through 32 layers is what fails. At layers 8, 16 and 24 (~210 bits) refinement recovers
  the head's accuracy (layer 8: plain 0.028, head 0.0062, refined 0.0053), but even the head's accuracy costs about
  twice the output's bits at layer 8 (the code was trained with the weaker decoder in the loop).
- C2 (the output's curve below Gaussian water-filling over Cov a): passes on all nets; the envelope is 0.55-0.80 of
  the bound at nMSE 0.01-0.001.
- C4 (refitted priors): the best refitted prior is never more than 0.6% below the trained prior's rate.
- C6 (your analytic reference): learned codes are at or below it at 555 of 557 checked points (505 of 507 without the
  output cells; nMSE 0.1, 0.03, 0.01, 0.003, 0.001); the exceptions are the unrestricted curve of net 4000 at layer 24 just before it stops (0.003)
  and narrow-deep net 220 layer 8 at 0.001 (at 0.005, outside that grid, also net 4000's layer-16 unrestricted
  curve). The reference is a baseline, not a matched ablation (its decoder is linear).
- C8 (decoder): at layer 0 local the head beats the decoder by > 20% at 4% of the points; at layer 0 unrestricted at
  44-56%.

### 4.9 What the slope above the dimension means (diagnostic, not registered)
Perturbing the state isotropically in whitened coordinates and measuring the output nMSE through the real net
(scratch/perturb_exponent.py, results/diag/): at layer 0 of ARC nets 4000 and 4001 the output error grows as the state
error to the power 0.97-0.98 at output nMSE ~0.002, 0.93-0.94 at ~0.02 and 0.78-0.82 at ~0.1; at layer 31 the power
is 1.00 (MEASURED, for isotropic perturbations). If a code's error scaled the same way, resolving all 256 directions
would cost about 256 / 0.95 = 270 bits per halving of the output RMS error near 1-2%, most of the measured 274-302; that
inference is HEURISTIC (a code's error is neither isotropic nor unbiased, and optimisation and prior overheads are not
excluded).

## 5. Comparisons

- Relevance lane (work/relevance), context only: its N(l, eps) counts noiseless linear features under a Gaussian state
  convention, while our number is the codes' active latents (> 0.05 bit) on the true state distribution, taken from the
  reference start; they are different quantities. Median over nets, at nMSE 0.1 / 0.01 / 0.001: layer 0: 240 / 251 / 254 against 208 / 256 / 256; layer 8: 82 / 171 / 200
  against 56 / 128 / 192; layer 16: 32 / 106 / 153 against 20 / 80 / 128; layer 24: 18 / 58 / 116 against 12 / 48 / 96;
  layer 31: 17 / 60 / 124 against 12 / 48 / 96. Its water-filling numbers use only the top 96 eigenvalues of its
  spectra and the linearised distortion, so they are not comparable where N > 96 (layers 0-8); our closed form over
  the full spectrum (section 2) replaces them.
- Bits-bounds formula (outbox/bits-bounds-2026-11-01.md, 3.5-3.6; net 4000, d = PR): 224 / 112 / 45 bits at nMSE 0.1
  for layers 4 / 8 / 16 against the reference-start base's 236 / 100 / 32 and the look-ahead start's 192 / 82 / 27: the
  formula is within a factor 1.2 / 1.4 / 1.7 of the registered count (the better start; P1.7, within a factor 2: pass).
- Sibling lane (work/bit-accounting/bit-accounting.md section 4.1), context only (channel capacities of fixed-rate
  codes and information rates with a learned prior are different upper bounds): its earlier fixed-budget local counts
  on these nets (> 256 / 135 / 43 / 25 / 24 at layers 0 / 8 / 16 / 24 / 31 for 10% error;
  210 / 119 / 135 at layers 16 / 24 / 31 for 1%), against ours (629 / 111 / 39 / 21 / 19 and 205 / 103 / 95): ours are
  0.7-1.0 of theirs from layer 8 on and resolve layer 0 (its codes were capped at 256 bits).

## 6. Pre-registration scorecard

Three scorers on the same runs (results/scorecard.md, scratch/scorecard.py): the registered v1 (scratch/score_P1.py,
on a merged copy of the runs), the A4 scorer v2 (as it was, without the A20 normalisation) and the A20 scorer v3 (pass /
fail / not evaluable kept apart; the registered local count = the better of the two starts).

| prediction | registered scorer (v1) | A4 scorer (v2) | A20 scorer (v3) |
|---|---|---|---|
| P1.1: C1 passes at L - l <= 8 (both curves reached) on >= 75% of cells | pass (hits 53, n 54, need 41) | pass (hits 53, n 54, need 41) | pass (hits 53, n 54, need 41) |
| P1.2: ARC local R(0.1) at l in {0, 2, 4} >= 4 x R_out(0.1), 4 of 4 nets | pass (hits 4, n 4, need 4) | pass (hits 4, n 4, need 4) | pass (hits 4, n 4, need 4) |
| P1.3: ARC local l = 31 within 15% of R_out at 0.1 and 0.01, >= 3 of 4 nets | pass (hits 4, n 4, need 3) | pass (hits 4, n 4, need 3) | pass (hits 4, n 4, need 3) |
| P1.4: local R(0.1) non-increasing in l (10% + 1 bit), ARC >= 3 of 4, narrow-deep >= 5 of 6 | pass (arc_hits 4, nd_hits 6) | fail | pass |
| P1.5: slopes: ARC l = 0 above l = 31 near 0.01 (4 of 4); narrow-deep l >= 32 in [0.8, 1.6] (>= 5 of 6) | pass (arc_hits 4, arc_n 4, nd_hits 6, nd_n 6) | fail (arc_hits 4, arc_n 4, nd_hits 0, nd_n 0) | pass (arc_hits 4, arc_n 4, nd_hits 6, nd_n 6) |
| P1.6: ARC local slope near 0.01 within [0.5, 2] x N(l, 0.01) at l = 16-31 on >= 75% | pass (hits 16, n 16, need 12) | pass (hits 16, n 16, need 12) | pass (hits 15, n 16, need 12) |
| P1.7: ARC 4000 local R(0.1) within 2x of the bits-bounds formula (d = PR) at l = 4, 8, 16 | pass (hits 3, n 3, need 3) | pass (hits 3, n 3, need 3) | pass (hits 3, n 3, need 3) |
| P1.8: C6 passes on >= 90%; layer 31 local within 10% of the gram reference at 0.1 and 0.01 (>= 3 of 4) | fail (c6_pass 555, c6_n 557, l31_hits 0, l31_n 4) | fail (c6_pass 555, c6_n 557, l31_hits 0, l31_n 4) | fail (c6_pass 555, c6_n 557, l31_hits 0, l31_n 4) |
| P1.9: C2 passes on all nets; R_out(0.01) <= 0.8 R_G(0.01) on >= 3 of 4 ARC nets | pass (ratio_hits 4) | pass (c2_pass 120, c2_n 120, ratio_hits 4) | pass (c2_pass 120, c2_n 120, ratio_hits 4) |
| P1.10: narrow-deep: both classes at l >= 32 within max(0.5 bit, 10%) of R_out; local R(0.1) at l = 0 >= 3 R_out | fail (n 6, sync_hits 0, input_hits 6) | fail (n 6, sync_hits 0, input_hits 6) | pass (n 6, sync_hits 6, input_hits 6) |
| P1.11: output-mean error < nMSE / 10 on >= 90% of resolved points with nMSE <= 0.1 | pass (hits 5477, n 5862, need 5276) | pass (hits 5477, n 5862, need 5276) | pass (hits 5480, n 5865, need 5279) |
| P1.12: C5 (planted toy) passes for both classes at k = 4 and 16 (registered random local start) | fail | fail | fail |

v3's P1.5 (narrow-deep part) and P1.10 (synchronisation part) are passes on the measured combinations only: the
unrestricted class ran at layers 64 and 255 of the five layers >= 32 (7 of 10 layer-class combinations per net); the
registered full-coverage predictions are not evaluable. Misses that hold under every scorer: P1.8's second part (the learned layer-31 code is 25-35% below the analytic
reference, not within 10%: one layer of future is not "nearly linear" for a sparse nonnegative output) and P1.12 (the
registered random-start local toy fails; see 4.7). The differences between scorers come from cells cut by A8/A18:
v1 and v2 count a missing cell as a failure (P1.10's synchronisation part, v2's P1.4 and P1.5), v3 scores the cells
that ran.

## 7. Process

- Registered 04:30Z (PREREG Part P1) after dev runs on dev nets only (ARC 9100, narrow-deep 111, toy seed 9). Nineteen
  amendments, each hashed before the runs it affects (prereg_hashes.txt; notes/DEVIATIONS.md lists every deviation,
  including kills by PID, preemptions and scheduling):
  - A1 (04:44Z) the analytic gain cap SNR 256 -> 65536: the old cap stopped narrow-deep codes at nMSE 0.004.
  - A2 (04:50Z) fp32 for every precision-critical matmul: TF32 rounding in a linear decoder that contains W^{-1}
    raised the initial nMSE from 1.6e-4 to 0.033 on the dev net.
  - A3 (05:01Z) the look-ahead local start, after the sibling lane's code review found reference-started local
    encoders effectively affine.
  - A4 (05:20Z) fixes from this lane's code review (gpt-6-astra, review/review_r1.raw, 14 findings): the Jacobian
    statistics on the data's span, the output head clipped separately, and scoring rules (trained-prior rate; refined
    decoder scored on its own subset; time-sharing interpolation; restart 2 kept out of the base; a cell inventory).
  - A5 (05:28Z) cheaper prior refits and references.
  - A6 (05:31Z) the planted toy also from the nets' reference start.
  - A7 (06:36Z) the look-ahead start on ARC nets 4001-4003 at layers 0-8.
  - A8 (08:39Z) a reduced plan after three spot preemptions (restart 0 for the remaining ARC local layers; fewer
    unrestricted layers; steps/capacity checks at a few layers).
  - A9 (09:17Z) 4x steps at layer 31 local. A10 (09:29Z) scheduling over both VMs; the look-ahead start on nets
    4001-4003 at layers 12-28 (cut again by A18 before most of it ran). A11 (09:39Z) depth 6 with 4x steps at layer 31.
    A12 (09:50Z) dedicated models at more low-rate points of layer 0. A13 (09:55Z) a 10x encoder learning rate.
    A14 (10:16Z) 4x steps for the output cell and layer 24. A15 (10:26Z) random starts with a 33x encoder learning
    rate. A16 (10:55Z) depth 6 with 8x steps at layer 31. A17 (11:34Z) random starts with a 10x rate at ARC layers 0
    and 8 and narrow-deep layer 0.
  - A18 (11:40Z, main's request) the convergence subset, with cuts of low-value cells to make room: narrow-deep local
    base at layers 16, 32, 128, narrow-deep unrestricted at layer 8, some narrow-deep checks, the look-ahead start on
    nets 4001-4003 at layers 12-28.
  - A19 (12:25Z, main relaying your proposal) the backward curriculum start and the consistency flags. Two 300-step
    smoke tests of the backward start on net 4000 layer 28 (12:19Z, 12:24Z) produced held-out curves before the A19
    hash; they are disclosed pilot evaluations (kept in /root/runs/BWDTEST, synced with bitarc-6), not counted.
  - A20 (13:16Z) the analysis corrections from review round 2 (common normalisation, convex-hull time-sharing, named
    baselines, scorer v3, matched doubling partners, export statuses, wording).
- Compute: L4 VMs bitarc-1 (spot, preempted after 2.6 h), bitarc-2 (spot, preempted after 28 min), bitarc-3 (spot,
  preempted after 16 min), bitarc-4 (spot, 5.0 h), bitarc-5 (on-demand, 5.0 h), bitarc-6 (on-demand, 4.7 h, stopped when
  its queue was done), bitarc-7 (spot from 12:30Z, on-demand being stocked out in all 13 zones; preempted at ~14:32Z,
  final sync failed, results pulled up to 14:30Z kept), bitarc-9 (on-demand, 14:49-16:51Z, round 2 only, A22); at most
  two at once. Costs at the on-demand list price ($0.87/h, an upper bound for spot) in GPU-LOG.md: $17.50 for the
  first seven VMs, $19.27 in all with bitarc-9 (2.0 h); every VM deleted (gcloud list empty at 16:52Z). Cap $25
  (main allowed up to $40 with notice; not needed).
- Review: gpt-6-astra (codex-any, "model: gpt-6-astra" checked in each transcript). Round 1 on the method and code during
  the runs (review/review_r1.raw; "fixable issues"; applied as A4). Round 2 on the draft report and the analysis code
  (review/review_r2.raw; "fixable issues", 14 findings; applied as A20: no table cell moved by more than 5%, but
  several claims were narrowed). Round 3 on the near-final report (review/review_r3.raw; "fixable issues", 8 findings:
  convergence wording beyond the checks, output baselines in C1, export completeness, normalisation of the benchmarks,
  stale numbers, scorer coverage labels; all applied; the 12 headline counts, the convergence verdicts and the backward
  counts reproduced from the run files).
- Saved models (.pt, state dict and arguments, for the later analysis of the identity-anchored codes): runs/bitarc-*/P1/
  VR_<family>_l<layer>_<class>_<variant>_r<restart>_s<first seed>.pt; one file holds every net and restart of its
  process (the name carries the first seed). Kept: ARC local base at layers 0, 2, 8, 12, 16, 20, 24, 28, 31 (all nets;
  layer 4's was lost with bitarc-7), the look-ahead codes at every layer of net 4000 and layers 0-8 of nets 4001-4003,
  the convergence runs, the unrestricted codes at layers 0, 8, 16, 24, 31, the output cell; narrow-deep local base and
  look-ahead codes at every layer that ran, unrestricted at 0, 16, 64, 255; round 2 (bitarc-9): layer 31 local and
  layers 16 and 31 unrestricted at depth 6 with 8x steps, layer 16 local look-ahead with 8x steps, layer 0 unrestricted
  at depth 6 with 4x steps. 71 files, 3.9 GB.

## 8. Files
work/bit-accounting-arc/: this report; PREREG.md (Part P1, amendments A1-A22) and prereg_hashes.txt; notes/DEVIATIONS.md
(every deviation, kill and preemption), notes/DESIGN.md, notes/QUESTIONS.md; GPU-LOG.md; gpu_src/ (the code that ran:
lib/vrlib.py, scripts/run_vr.py, ref_curve.py, refine_posthoc.py; queue files in queue_done/); lib/vra.py, lib/vrb.py and
scratch/ (analysis: score_P1.py, score_P1_v2.py, score_P1_v3.py, report_tables.py, export_curves.py, conv_subset.py,
conv_table.py, c7_table.py, consistency.py, spectrum_table.py, perturb_exponent.py, scorecard.py, table_diff.py, figure
scripts); runs/bitarc-*/ (raw results per VM); results/ (P1_tables_final.md, P1_scores_v1/v2/v3.json, scorecard.md,
conv_subset.md/.json, conv_table.md, c7_table.md, consistency.md, spectrum_table.md, certified_curves.json, diag/,
update_vs_1133Z.md); figs/; review/ (prompts, raw reviews, status).
