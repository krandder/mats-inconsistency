# How many bits each layer needs (bit-accounting lane, 1-2 November 2026)

Status: REPORT v4 (phase 1 final; phase 2: P2.0 parts 1-5 with the code-map v2 rerun, corrected unscrambling rerun U-c, partial extra exploration 13/53). Coverage: main cells 159/159, extra certificates 55/159. Phase 1 (Kelvin 02:55Z): the best bits-vs-accuracy curve of every layer, one
lambda-conditioned variable-rate code per layer and encoder class, with certificates on every curve (PREREG Part P1,
registered 04:01Z). Phase 2 (registered as Part P2.0: Kelvin's near-identity hypothesis; then the unscrambling loss at
fixed rate) is in sections 3.1-3.5. Earlier measurements (fixed-budget codes, scrambling tests, estimation) are kept in
sections 4-6 as background.

Work directory work/bit-accounting/: PREREG.md + prereg_hashes.txt, FORMAL.md (theory), notes/DEVIATIONS.md,
GPU-LOG.md, results/ (P1_table.md: every cell; P1_summary.md: by layer; P1_scores.json), figs/P1_curves_*.png, runs/P1/
(raw JSON per cell). Labels: PROVEN, MEASURED, HEURISTIC.

## Summary for Kelvin

1. **What was measured (your 02:55Z request).** For each layer of the small random nets (16x4, 32x8, 16x16, 32x32;
   three seeds; layers 0, L/4, L/2, 3L/4, L-1 and the output) and two controls, the fewest bits a code of that layer
   needs so that decoding it and running the real rest of the net gives the output within a given error: the whole
   bits-versus-accuracy curve. The registered method is one model per layer that takes the bit price as an input;
   the reported count R* is the best of several codes of the class (that model's restarts, a 2x-steps version,
   explorer restarts, and single-price models where they are better), and the single registered model r0 is given
   next to it. The code is real
   numbers sent through fixed unit noise with encoder-chosen power (the same thing as fixed power with chosen noise:
   only the signal-to-noise ratio matters); bits = information under a learned prior. Each model starts as the linear
   reference code (your identity anchor in its water-filled form: at a high budget every direction the output
   depends on is sent at SNR 256, an almost perfect reconstruction) and the price rises during training.
2. **An encoder that can compute anything usually needs about the output's own bits at every layer.** Across layers
   and nets the median gap between the unrestricted curve and the output's own curve of the same net is 0.04 (max 0.63) bit at
   10% error and 0.14 (max 77.18) bit at 1% (57 layer-net pairs). There are 9 exceptions; the largest: 32x8 s5002 l=0 at 1%: 100.3 against 23.1; 32x8 s5001 l=0 at 1%: 88.7 against 20.8; 32x8 s5000 l=0 at 1%: 66.8 against 20.4; 16x4 s5001 l=0 at 1%: 47.5 against 17.4 bits. In these cases the trained
   decoders did not find precise enough preimages; the theory (FORMAL 4.1) only bounds reachability, so these
   gaps are observations, not proofs that the shortcut fails. The bits are small: 1.7-2.6 for 10% and 5.6-7.0 for 1% on
   16x16 (output 1.7-2.5 / 5.6-6.5).
3. **A local encoder (one ReLU layer as wide as the net) needs more bits the closer the layer is to the input:
   a little more at L/2, clearly more at L/4, and very many more at the input** (best-found curves; on all 12 ReLU
   nets the count does not rise with depth, P1.3). Bits for 10% / 1% error (median over seeds; the single registered
   model r0 is in section 2): 
   - 16x4: l = 0: 12.5 / 34.6; l = 1: 9.6 / 25.4; l = 2: 7.7 / 22.4; l = 3: 7.3 / 20.7; output: 6.6 / 17.4
   - 32x8: l = 0: 27.1 / 80.2; l = 2: 10.9 / 45.2; l = 4: 7.6 / 27.9; l = 6: 7.2 / 23.7; l = 7: 7.1 / 22.5; output: 6.9 / 20.8
   - 16x16: l = 0: 17.2 / 48.8; l = 4: 3.2 / 11.9; l = 8: 2.6 / 7.4; l = 12: 2.5 / 6.5; l = 15: 2.5 / 6.6; output: 2.4 / 6.1
   - 32x32: l = 0: 41.3 / 112.0; l = 8: 2.0 / 12.6; l = 16: 1.9 / 6.3; l = 24: 1.9 / 5.6; l = 31: 1.9 / 5.5; output: 1.9 / 5.4
4. **Slopes (bits per halving of the output error, 10% -> 1%).** At the input layer at least n/2 bits per halving
   (8.9 for n = 16, 21.4 for n = 32): essentially every input direction has to be refined. At the output 1.2 (16x16), 1.1 (32x32) bits (two
   to three directions). In between the slope falls fast with depth (2.6 at L/4 for 16x16).
5. **The anchored local codes were not the best in the class at early layers; codes whose ReLU gates start inside
   the data cloud need fewer bits.** The registered runs start every local encoder from the linear reference code, with
   its gates outside the data cloud (affine), and they end close to the linear reference at the input layer
   (single anchored model r0: 16x16 27.1 / 55.9 bits against the linear reference 26.9 / 55.6; 32x32 52.8 / 112.6 against 53.1 / 111.4). Local encoders started from the net's own next-layer gates (amendment P1-A, 'explorer') need fewer bits:
   median saving at 10% / 1% error, by layer: input 32% / 12% (n = 12); L/4 32% / 19% (n = 12); L/2 11% / 14% (n = 12); late 1% / 3% (n = 19). Widening only the decoder and prior gave little typical
   improvement (width-n encoder with the wide decoder minus the base model: median -0.02 bit, range -1.56 to +0.25 (110 points)), which supports, but does not
   prove, that the gains of the 2x-wide local encoder come from the encoder's own width. The counts in this report
   include the explorer codes; better explorations of the class could lower the early-layer counts further.
6. **How sure we are: these are best-found curves, not certified optima, and the one-model requirement is not met
   everywhere.** The certificates below test the anchored models; the explorer codes that set many early-layer counts
   got only their own restarts.
   On every curve: three restarts, a model trained twice as long, a model twice as wide, refitted priors, an
   output-space centroid, the linear reference, and three separately trained single-price models (section 3).
   Restarts agree on 805/812 points (but they all start from the same anchor; see point 5). A model trained twice as long is within tolerance of the base model on 795/824
   points. The single registered model r0 sits above the reported counts by a median of input 9.7; L/4 3.9; L/2 1.5; late 0.1 bits (input, L/4,
   L/2, later local layers; mostly because the explorer and 2x-steps codes are better). The 2x-wide unrestricted model is within tolerance on 434/453.
   Refitted priors are no better (within 5%) on 2149/2156 points (the largest exception is 13% lower). The learned code
   is within 0.3 bit or 5% of the linear reference or below it on 823/823 of the points where it reaches the error level;
   70 points exceed it within that tolerance (by up to 2.6 bits). Known answers: the output codes are
   within 0.3 bit of the Gaussian bound or below it (36 points, all hit); unrestricted matches output on 105/114 points; the
   linear net gives the Gaussian answer within 0.12 bit at every layer for both classes (33 points, all hit). Limits:
   (a) A local encoder twice as wide (outside the class; decoder and prior also doubled) needs fewer bits on
   35% of points, mostly at L/4 to L/2; the amendment suggests it is the encoder's width (point 5), so there the
   count depends on the width-n restriction.
   (b) One model for all prices is within 3% of the separately trained single-price models on 390/477 spot checks
   (443/477 for the 2x-steps model). Your requirement is therefore not met everywhere. The reported counts take the
   lower envelope of both, as registered; section 2 shows the single model too.
   (c) 129 (target, cell) points are not covered: the learned curves stop before the target within the trained
   price range (breakdown in section 3).
   (d) Uncertainty: the spread between restarts (median 0.02 (90th percentile 0.25) bit) is the only measured one. The selection over
   about ten curves per cell is not corrected for.
   The independent review (gpt-6-astra) asked for three more checks: encoders whose ReLU gates start inside the data
   cloud (point 5), an output-space residual correction around r0's reconstruction (the model is within 1.4% of the corrected output (median; worst 6%, 165 points); a fitted correction
   improved r0 little, which is consistent with, not proof of, a near-optimal decoder), and the encoder's width
   separated from the decoder's (point 5). Coverage of these extra checks: 55/159 cells (55 of the 57 ReLU local cells; the controls' local cells and all unrestricted cells were not reached within the GPU budget). The explorer initial fit used evaluation samples in 12 of
   these cells (pilot samples afterwards).
7. **Controls.** In the chaotic tanh net the local count grows steeply toward the input (7.8, 10.3, 18.2, 34.4, 65.7 bits for 10%
   error at l = 15, 12, 8, 4, 0; output 6.6). The trained unrestricted codes follow the output's curve only down
   to an error that grows toward the input; below it they need about the local bits (the decoders did not find
   precise preimages in the chaotic map; figs/P1_curves_tanh4_16x16.png). In the Boolean Feistel circuit, layers 12-15 need about the 16 bits of the state
   itself (l = 12 and 15: 14.5, 16.6 bits for 10%; output 16.6), and at l <= 8 none of the trained real-valued codes reached 10% error within the price range. In
   the linear net every layer, in both classes, needs the Gaussian answer to within 0.12 bit (4.2 / 11.5 bits against the Gaussian 4.1 / 11.4). One seed
   per control.
8. **Phase 2** (sections 3.1-3.5; Kelvin's near-identity hypothesis P2.0, the unscrambling loss, more
   exploration; reviewed by gpt-6-astra, round 3).
   - Per layer, a linear code costs 1.25-1.7 times the best code's bits. A learned mixture of linear pieces (gates
     inside the data cloud) needs 0.6-0.8 times the linear code's bits.
   - Across one layer, the anchored codes' means are fitted by an affine map to R^2 0.98-1.00 from L/4 on on ReLU
     (an empirical fit, not a certificate; with rank-truncated fits this now includes the late 32x32 codes). On the
     ReLU nets the codes as received (with the channel noise) predict the next layer's gates no better than the
     constant base rate; on the chaotic net they predict them well above it.
   - The registered near-identity predictions H2-H4 missed: the chaotic control's one-step maps look as simple as
     ReLU's.
   - Forcing the next code to be an affine function of this one (corrected penalty, rerun U-c) costs ReLU 0.03 / 0.19
     / 0.38 / 0.42 bits and chaotic tanh 0.18 / 0.40 / 1.08 / 2.25 bits at mu = 0.1 / 0.3 / 1 / 3 (median over pairs).
     The cost is extra rate; the pair's distortion barely moves. From L/4 on at mu >= 1, the tanh pair costs more than
     every ReLU pair at each layer position (one tanh seed). As a fraction of its larger code it does not (U2 missed).
   - In the noisy nets, maps fitted at one layer transfer poorly to deeper layers (H5 missed).
   - Training twice as long still lowers the input-layer local counts by 4-9% (B1 missed), so those counts are upper
     bounds.

## 1. What phase 1 measures (method, plain)

For one layer l of a fixed random net, a code describes the layer's state h_l with a few real numbers z, each received
through unit Gaussian noise; the encoder chooses how loud each number is (fixed noise with chosen power is the same
thing as fixed power with chosen noise: only the signal-to-noise ratio matters). The decoder turns z back into a
state at layer l, and the real rest of the net turns that into an output. The cost of a code is its information in
bits (log2 q(z | h) - log2 p(z), averaged; p a learned prior), the accuracy is the output's error on fresh inputs,
nMSE = E|a_hat - a|^2 / tr Var(a). The bits-vs-accuracy curve of the layer is the fewest bits that reach each error.

One model per layer and encoder class traces the whole curve: the bit price lambda is an input to the encoder, the
decoder and the prior. Training starts from a near-perfect identity code (each coordinate of the whitened state sent
at signal-to-noise 256, about 4 bits per coordinate, decoded by the inverse) and raises the price gradually, so the
code sheds the least useful directions first. Each price counts equally in the loss.

Encoder classes: unrestricted (any network that also sees the exact output; it can take the shortcut) and local (one
ReLU layer as wide as the net, then affine: it can compute the next layer but not the rest of the net).

How we know a curve is as good as reasonably possible (certificates, every curve):
- three independent restarts; a model trained twice as long; a model twice as wide (prior twice as rich);
- the rate re-measured with fresh, richer priors fitted on held-out codes (is the learned prior loose?);
- an output-space regressor E[a | z] trained on held-out codes (is the decoder optimal for its codes?);
- three separately trained single-price models (is one model for all prices worse than one model per price?);
- the analytic linear reference: send the directions of the cloud-averaged, future-weighted Jacobian spectrum with
  reverse-water-filled power, decode by linear MMSE, run the real net; a learned code must match or beat it;
- known answers: the output's own curve against its Gaussian bound; every unrestricted curve against the output's;
  a linear net, where the curve is known exactly (Gaussian reverse water-filling) for both classes.

## 2. Bits per layer (phase 1; median over seeds [range]; "unreached" = beyond the trained price range)

Counts are conservative (time-sharing between achieved points). "Bits" columns = R*, the best code of the class found by any variant (an envelope over several models); "single model r0" = the registered single lambda-conditioned model, which is often several bits higher at the early layers. Every cell, seed, source and certificate flag: results/P1_table.md; curves: figs/P1_curves_*.png.

| net | layer | class | bits for 10% error | bits for 1% | bits for 0.1% | bits per halving (10% -> 1%) | single model r0, 10% / 1% | linear reference 10% / 1% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| relu 16x4 | 0 | local | 12.5 [8.8-18.8] | 34.6 [30.1-44.4] | unreached | 6.66 | 22.0 [17.4-27.6] / 48.0 [44.1-53.3] | 22.2 [17.8-27.3] / 47.6 [44.0-53.1] | 3 |
| relu 16x4 | 0 | unres | 7.2 [4.9-7.4] | 21.6 [15.3-47.5] | unreached | 4.26 | 7.4 [5.1-7.5] / 33.1 [16.9-49.5] | 22.2 [17.8-27.4] / 47.6 [44.0-53.2] | 3 |
| relu 16x4 | 1 | local | 9.6 [6.0-11.7] | 25.4 [22.3-32.6] | unreached | 4.93 | 12.4 [8.9-18.2] / 32.7 [30.0-42.7] | 14.6 [11.6-22.3] / 36.8 [34.6-49.3] | 3 |
| relu 16x4 | 1 | unres | 7.0 [4.7-7.3] | 19.7 [14.8-21.0] | unreached | 3.73 | 7.2 [4.8-7.5] / 20.8 [15.2-25.6] | 14.6 [11.6-22.3] / 36.8 [34.5-49.3] | 3 |
| relu 16x4 | 2 | local | 7.7 [5.3-8.1] | 22.4 [18.6-23.4] | unreached | 4.32 | 9.7 [6.1-10.8] / 25.1 [22.7-29.1] | 11.9 [8.0-15.4] / 30.1 [26.9-38.5] | 3 |
| relu 16x4 | 2 | unres | 6.7 [4.7-7.2] | 17.9 [14.6-18.8] | 28.4 (2 unreached) | 3.37 | 6.8 [4.8-7.4] / 19.0 [14.9-19.3] | 11.8 [8.0-15.4] / 30.1 [26.8-38.5] | 3 |
| relu 16x4 | 3 | local | 7.3 [4.8-7.7] | 20.7 [15.7-21.0] | 31.8 (2 unreached) | 3.91 | 7.8 [5.3-7.9] / 21.2 [18.3-22.5] | 10.0 [6.6-11.8] / 27.5 [21.6-33.5] | 3 |
| relu 16x4 | 3 | unres | 6.5 [4.6-7.2] | 17.0 [14.4-18.5] | 31.8 [26.9-33.9] | 3.16 | 6.6 [4.7-7.2] / 17.5 [14.6-18.9] | 10.0 [6.6-11.8] / 27.5 [21.5-33.6] | 3 |
| relu 16x4 | output | unres | 6.6 [4.6-7.2] | 17.4 [14.6-18.7] | 27.1 (2 unreached) | 3.24 | 6.8 [4.7-7.3] / 18.3 [14.8-19.2] | 10.3 [6.4-12.4] / 31.0 [22.8-36.2] | 3 |
| relu 32x8 | 0 | local | 27.1 [22.0-33.8] | 80.2 [78.2-89.4] | unreached | 16.73 | 50.1 [43.6-52.0] / 105.2 [98.3-108.3] | 50.3 [44.1-51.8] / 104.4 [97.8-107.3] | 3 |
| relu 32x8 | 0 | unres | 7.1 [6.0-7.5] | 88.7 [66.8-100.3] | unreached | 24.91 | 7.1 [6.1-7.8] / 100.0 [99.5-106.9] | 50.3 [44.1-51.9] / 104.4 [97.8-107.4] | 3 |
| relu 32x8 | 2 | local | 10.9 [8.0-13.3] | 45.2 [38.9-51.8] | unreached | 10.35 | 17.1 [12.4-21.0] / 59.0 [51.4-64.8] | 21.1 [16.2-25.5] / 63.8 [57.7-71.9] | 3 |
| relu 32x8 | 2 | unres | 7.1 [5.8-7.4] | 23.2 [21.6-29.0] | unreached | 5.22 | 7.1 [5.9-7.6] / 25.6 [22.3-51.8] | 21.2 [16.2-25.6] / 63.8 [57.7-72.1] | 3 |
| relu 32x8 | 4 | local | 7.6 [6.3-8.9] | 27.9 [25.7-31.5] | unreached | 6.48 | 9.0 [7.2-10.4] / 33.4 [33.3-37.4] | 10.9 [9.4-13.2] / 41.4 [38.7-44.0] | 3 |
| relu 32x8 | 4 | unres | 7.0 [5.8-7.2] | 21.3 [21.0-24.2] | unreached | 4.69 | 7.0 [5.8-7.3] / 21.9 [21.5-25.7] | 10.9 [9.4-13.3] / 41.5 [38.7-44.1] | 3 |
| relu 32x8 | 6 | local | 7.2 [6.1-7.5] | 23.7 [21.8-26.1] | unreached | 5.31 | 7.4 [6.1-8.0] / 24.3 [22.8-27.9] | 8.0 [7.5-10.7] / 31.1 [26.6-37.1] | 3 |
| relu 32x8 | 6 | unres | 6.9 [5.7-6.9] | 20.8 [20.3-23.1] | unreached | 4.53 | 7.0 [5.8-7.0] / 21.2 [20.6-23.7] | 8.0 [7.5-10.8] / 31.1 [26.5-37.2] | 3 |
| relu 32x8 | 7 | local | 7.1 [5.9-7.2] | 22.5 [21.4-25.7] | unreached | 5.00 | 7.3 [5.9-7.6] / 23.0 [22.1-26.9] | 8.2 [7.1-10.0] / 30.5 [27.3-36.3] | 3 |
| relu 32x8 | 7 | unres | 6.9 [5.7-6.9] | 20.7 [20.2-22.9] | unreached | 4.52 | 6.9 [5.7-7.0] / 21.2 [20.5-23.7] | 8.2 [7.1-10.0] / 30.5 [27.3-36.3] | 3 |
| relu 32x8 | output | unres | 6.9 [5.7-6.9] | 20.8 [20.4-23.1] | unreached | 4.57 | 6.9 [5.7-7.0] / 21.4 [20.8-24.0] | 8.7 [7.0-10.3] / 35.8 [35.0-43.6] | 3 |
| relu 16x16 | 0 | local | 17.2 [15.7-28.8] | 48.8 [46.9-58.0] | unreached | 8.93 | 27.1 [24.0-35.1] / 55.9 [55.0-62.4] | 26.9 [24.7-34.8] / 55.6 [54.9-62.3] | 3 |
| relu 16x16 | 0 | unres | 2.5 [1.8-2.6] | 6.9 [6.7-7.0] | unreached | 1.36 | 2.6 [1.8-2.7] / 7.9 [7.1-25.0] | 26.9 [24.7-34.8] / 55.6 [54.9-62.3] | 3 |
| relu 16x16 | 4 | local | 3.2 [3.0-3.9] | 11.9 [11.4-18.7] | unreached | 2.60 | 4.2 [3.7-6.0] / 16.4 [13.5-22.2] | 5.9 [5.4-10.9] / 20.0 [17.6-31.3] | 3 |
| relu 16x16 | 4 | unres | 2.4 [1.8-2.5] | 6.5 [6.4-6.6] | 12.6 [12.3-12.8] (1 unreached) | 1.26 | 2.5 [1.8-2.6] / 6.7 [6.5-6.7] | 5.9 [5.4-10.9] / 20.0 [17.6-31.3] | 3 |
| relu 16x16 | 8 | local | 2.6 [1.8-2.7] | 7.4 [7.2-7.6] | 17.0 [16.6-18.4] | 1.44 | 2.7 [2.1-2.9] / 9.3 [8.6-9.9] | 4.0 [4.0-4.7] / 14.0 [13.3-16.1] | 3 |
| relu 16x16 | 8 | unres | 2.4 [1.7-2.5] | 6.2 [6.0-6.6] | 12.0 [11.7-12.5] | 1.25 | 2.5 [1.7-2.5] / 6.3 [6.1-6.6] | 4.0 [4.0-4.7] / 14.0 [13.3-16.1] | 3 |
| relu 16x16 | 12 | local | 2.5 [1.7-2.6] | 6.5 [5.8-6.8] | 12.8 [12.4-13.3] | 1.23 | 2.6 [1.8-2.7] / 6.7 [6.2-7.4] | 3.3 [2.2-3.4] / 9.5 [9.3-9.9] | 3 |
| relu 16x16 | 12 | unres | 2.4 [1.7-2.5] | 6.1 [5.6-6.6] | 11.3 [11.2-12.3] | 1.18 | 2.5 [1.7-2.5] / 6.2 [5.7-6.6] | 3.3 [2.2-3.4] / 9.5 [9.2-9.9] | 3 |
| relu 16x16 | 15 | local | 2.5 [1.7-2.5] | 6.6 [5.7-6.8] | 12.5 [11.7-12.9] | 1.23 | 2.5 [1.7-2.7] / 6.8 [5.9-6.8] | 3.0 [2.0-3.7] / 9.2 [7.0-10.5] | 3 |
| relu 16x16 | 15 | unres | 2.4 [1.7-2.5] | 6.1 [5.6-6.6] | 11.8 [11.1-12.4] | 1.17 | 2.5 [1.7-2.5] / 6.2 [5.6-6.6] | 3.0 [2.0-3.7] / 9.2 [7.0-10.5] | 3 |
| relu 16x16 | output | unres | 2.4 [1.7-2.5] | 6.1 [5.6-6.5] | 12.0 [11.2-12.3] | 1.17 | 2.4 [1.7-2.5] / 6.5 [5.6-6.6] | 3.0 [2.0-3.5] / 9.2 [7.1-11.1] | 3 |
| relu 32x32 | 0 | local | 41.3 [37.4-46.8] | 112.0 [111.1-118.0] | unreached | 21.43 | 52.8 [51.6-57.5] / 112.6 [111.3-118.1] | 53.1 [51.8-58.3] / 111.4 [110.3-116.9] | 3 |
| relu 32x32 | 0 | unres | 1.9 [1.6-2.4] | 5.6 [4.2-6.2] | unreached | 1.13 | 1.9 [1.7-2.5] / 5.7 [4.2-6.3] | 53.1 [51.7-58.3] / 111.4 [110.2-116.9] | 3 |
| relu 32x32 | 8 | local | 2.0 [1.7-2.7] | 12.6 [6.1-16.8] | unreached | 2.98 | 3.4 [2.0-3.6] / 16.0 [13.2-20.1] | 4.7 [3.1-7.8] / 21.0 [18.5-27.3] | 3 |
| relu 32x32 | 8 | unres | 1.8 [1.6-2.4] | 5.5 [4.2-6.1] | 11.9 [9.2-12.0] | 1.11 | 1.9 [1.7-2.4] / 5.6 [4.3-6.2] | 4.7 [3.1-7.8] / 21.0 [18.4-27.3] | 3 |
| relu 32x32 | 16 | local | 1.9 [1.7-2.5] | 6.3 [4.3-6.5] | 14.7 [11.0-17.6] | 1.21 | 1.9 [1.7-2.6] / 7.1 [5.1-7.9] | 2.1 [1.9-3.1] / 8.4 [7.5-10.1] | 3 |
| relu 32x32 | 16 | unres | 1.9 [1.6-2.4] | 5.5 [4.1-6.0] | 11.5 [8.8-11.7] | 1.10 | 1.9 [1.6-2.4] / 5.5 [4.2-6.1] | 2.1 [1.9-3.1] / 8.4 [7.5-10.1] | 3 |
| relu 32x32 | 24 | local | 1.9 [1.6-2.4] | 5.6 [4.1-6.2] | 11.9 [9.0-12.2] | 1.11 | 1.9 [1.7-2.5] / 5.6 [4.2-6.4] | 1.9 [1.7-2.9] / 6.1 [4.5-8.9] | 3 |
| relu 32x32 | 24 | unres | 1.9 [1.6-2.4] | 5.5 [4.1-6.0] | 11.3 [8.7-11.6] | 1.09 | 1.9 [1.6-2.4] / 5.5 [4.1-6.1] | 1.9 [1.7-2.9] / 6.1 [4.5-9.0] | 3 |
| relu 32x32 | 31 | local | 1.9 [1.6-2.4] | 5.5 [4.1-6.1] | 11.7 [8.8-12.4] | 1.09 | 1.9 [1.7-2.4] / 5.5 [4.2-6.2] | 1.9 [1.7-2.8] / 5.9 [4.4-9.1] | 3 |
| relu 32x32 | 31 | unres | 1.9 [1.6-2.4] | 5.5 [4.1-6.0] | 11.3 [8.6-11.8] | 1.08 | 1.9 [1.6-2.4] / 5.5 [4.1-6.1] | 1.9 [1.7-2.8] / 5.9 [4.4-9.1] | 3 |
| relu 32x32 | output | unres | 1.9 [1.6-2.4] | 5.4 [4.1-6.0] | 11.3 [8.7-11.9] | 1.08 | 1.9 [1.6-2.4] / 5.5 [4.1-6.1] | 1.9 [1.7-2.8] / 6.0 [4.4-9.4] | 3 |
| tanh4 16x16 | 0 | local | 65.7 | 99.2 | unreached | 10.09 | 66.0 / 99.8 | 65.4 / 99.3 | 1 |
| tanh4 16x16 | 0 | unres | 64.6 | 98.5 | unreached | 10.20 | 64.7 / 98.8 | 65.4 / 99.3 | 1 |
| tanh4 16x16 | 4 | local | 34.4 | 67.9 | unreached | 10.10 | 36.5 / 68.3 | 38.4 / 68.3 | 1 |
| tanh4 16x16 | 4 | unres | 9.0 | 67.0 | unreached | 17.45 | 26.1 / 67.8 | 38.4 / 68.3 | 1 |
| tanh4 16x16 | 8 | local | 18.2 | 47.5 | unreached | 8.81 | 20.3 / 49.4 | 26.3 / 51.3 | 1 |
| tanh4 16x16 | 8 | unres | 7.6 | 38.5 | unreached | 9.30 | 8.0 / 43.9 | 26.3 / 51.3 | 1 |
| tanh4 16x16 | 12 | local | 10.3 | 26.6 | unreached | 4.90 | 10.6 / 28.6 | 15.6 / 38.4 | 1 |
| tanh4 16x16 | 12 | unres | 7.0 | 17.0 | unreached | 3.02 | 7.2 / 18.7 | 15.6 / 38.4 | 1 |
| tanh4 16x16 | 15 | local | 7.8 | 21.5 | unreached | 4.11 | 7.8 / 22.4 | 12.0 / 33.2 | 1 |
| tanh4 16x16 | 15 | unres | 6.9 | 16.5 | unreached | 2.89 | 7.1 / 18.0 | 12.0 / 33.2 | 1 |
| tanh4 16x16 | output | unres | 6.6 | 16.8 | unreached | 3.05 | 7.0 / 18.8 | 12.7 / 37.8 | 1 |
| feistel 16x16 | 0 | local | unreached | unreached | unreached | - | unreached / unreached | unreached / unreached | 1 |
| feistel 16x16 | 0 | unres | unreached | unreached | unreached | - | unreached / unreached | unreached / unreached | 1 |
| feistel 16x16 | 4 | local | unreached | unreached | unreached | - | unreached / unreached | unreached / unreached | 1 |
| feistel 16x16 | 4 | unres | unreached | unreached | unreached | - | unreached / unreached | unreached / unreached | 1 |
| feistel 16x16 | 8 | local | unreached | unreached | unreached | - | unreached / unreached | unreached / unreached | 1 |
| feistel 16x16 | 8 | unres | unreached | unreached | unreached | - | unreached / unreached | unreached / unreached | 1 |
| feistel 16x16 | 12 | local | 14.5 | 15.9 | 16.0 | 0.42 | 16.0 / 16.0 | 101.5 / 132.3 | 1 |
| feistel 16x16 | 12 | unres | 15.7 | 16.0 | 16.0 | 0.08 | 16.0 / 16.0 | 101.5 / 132.3 | 1 |
| feistel 16x16 | 15 | local | 16.6 | 21.0 | 22.8 | 1.31 | 17.1 / 21.2 | 32.9 / 59.0 | 1 |
| feistel 16x16 | 15 | unres | 16.5 | 18.2 | 22.4 | 0.50 | 16.7 / 20.7 | 32.9 / 59.0 | 1 |
| feistel 16x16 | output | unres | 16.6 | 22.9 | 26.8 | 1.88 | 17.4 / 24.3 | 23.6 / 47.3 | 1 |
| linear 16x16 | 0 | local | 4.2 | 11.5 | 20.3 | 2.20 | 4.2 / 11.5 | 4.2 / 11.4 | 1 |
| linear 16x16 | 0 | unres | 4.2 | 11.5 | 20.3 | 2.20 | 4.2 / 11.5 | 4.2 / 11.4 | 1 |
| linear 16x16 | 4 | local | 4.2 | 11.5 | 20.3 | 2.20 | 4.2 / 11.5 | 4.2 / 11.4 | 1 |
| linear 16x16 | 4 | unres | 4.2 | 11.5 | 20.3 | 2.20 | 4.2 / 11.5 | 4.2 / 11.4 | 1 |
| linear 16x16 | 8 | local | 4.2 | 11.5 | 20.3 | 2.20 | 4.2 / 11.5 | 4.2 / 11.4 | 1 |
| linear 16x16 | 8 | unres | 4.2 | 11.5 | 20.3 | 2.20 | 4.2 / 11.5 | 4.2 / 11.4 | 1 |
| linear 16x16 | 12 | local | 4.2 | 11.5 | 20.3 | 2.20 | 4.2 / 11.5 | 4.2 / 11.4 | 1 |
| linear 16x16 | 12 | unres | 4.2 | 11.5 | 20.3 | 2.20 | 4.2 / 11.5 | 4.2 / 11.4 | 1 |
| linear 16x16 | 15 | local | 4.2 | 11.5 | 20.3 | 2.20 | 4.2 / 11.5 | 4.2 / 11.4 | 1 |
| linear 16x16 | 15 | unres | 4.2 | 11.5 | 20.3 | 2.20 | 4.2 / 11.5 | 4.2 / 11.4 | 1 |
| linear 16x16 | output | unres | 4.2 | 11.5 | 20.3 | 2.20 | 4.2 / 11.5 | 4.2 / 11.4 | 1 |

## 3. What the certificates show, and what they do not (registered in PREREG P1, P1-A)

Share of (cell, error level) points where each check passes; "not covered" = the curves do not reach that error within the trained price range.

| certificate | passes | points | not covered |
|---|---|---|---|
| C1: three restarts agree | 99% | 812 | 142 |
| C2: 2x steps does not improve | 96% | 824 | 130 |
| C3: 2x width does not improve (unres) | 96% | 453 | 69 |
| C3_local: 2x width does not improve (local: width 2n leaves the class) | 65% | 362 | 70 |
| C4: refitted priors no better | 100% | 2156 | 0 |
| C5: centroid (first version) | 99% | 477 | 0 |
| C5v2: output-space residual centroid (amendment) | 100% | 165 | 0 |
| C6_r0: one model for all prices within 3% of single-price models | 82% | 477 | 0 |
| C6_long: same, for the 2x-steps model | 93% | 477 | 0 |
| C7: learned code at or below the linear reference | 100% | 823 | 129 |

Registered predictions (PREREG P1; scored by gpu_src/scripts/score_P1v2.py):

- KA1: output codes within 0.3 bit of the Gaussian bound or below, every ReLU net: hit (36/36)
- KA2 (empirical comparison): unrestricted within max(0.5 bit, 15%) of the output curve on >= 90%: hit (105/114)
- KA3: linear net: the Gaussian answer at every layer, both classes: hit (33/33)
- P1.1: certificates C1, C2, C3 (unres), C4, C5 (v2), C7 each pass on >= 90%: pending
- P1.2: local count at the input >= 2x the output count on >= 5 of the 6 L = n nets: hit (6/6)
- P1.3: local count not rising with depth on >= 80% of the 12 ReLU nets: hit (12/12)
- P1.4: output slope smaller on the deeper net of each width, every seed: hit (6/6)
- P1.5: controls (tanh4, feistel, linear): True (parts {'tanh4': True, 'feistel': True, 'linear': True})

What the checks show, in words:
- Restarts, 2x steps, refitted priors: the curves are reproducible; the 2x-steps model is within tolerance of the
  base model at almost every point but is often the best code, so the reported counts (R*) are a few tenths of a bit
  below the single registered model r0 there (both are in section 2 and results/P1_table.md). Fresh, richer priors
  fitted on held-out codes give a lower rate at 7 of 2156 points (up to 10%), so the learned priors are tight almost
  everywhere but not everywhere.
- Linear reference: wherever a learned curve reaches an error level, it needs at most the linear reference's bits.
  At the input layer the anchored local code coincides with it and the explorer codes need fewer bits; from L/4 on
  every nonlinear local code needs clearly fewer.
- 2x width, local: a local encoder of width 2n (outside the class; its decoder and prior are doubled too) needs
  fewer bits at L/4 to L/2. The amendment's width-n encoder with the wide decoder and prior is no better than the
  base model (median -0.02 bit), so the encoder's own width causes it: there the count depends on the class.
- One model for all prices against single-price models: the variable-rate model is within 3% of the separately
  trained single-price models at most spot checks but not all (worse by up to 27%); training it twice as long closes
  part of the gap. The requirement is not met everywhere. The reported counts take the lower envelope of the
  variable-rate curves and the single-price points (the registered fallback); the table's "source" column says
  where a single-price point ('fixed') or a combination of curves ('pooled') sets the count.
- Restarts all start from the same analytic anchor, whose local ReLU gates sit outside the data cloud (the encoder
  starts affine), so restart agreement alone does not show that the nonlinear part of the local class was explored.
  The amendment's explorer restarts start with the gates inside the cloud (the net's own next-layer gates) and keep
  3-100% of them (median 69%) active on the data; they find codes with clearly fewer bits at the early layers (summary point 5),
  so the anchored curves were not the best in the class there, and the explorer curves themselves are best-found,
  not certified optimal (their own restarts differ by up to 14% at the input layer, median 2%).
- Interpolation: counts use time-sharing between achieved points (conservative). The registered log-linear
  interpolation over the same candidates is in results/P1_scores.json (R_loglinear).
- Uncertainty: the only measured one is the spread between restarts (summary point 6). Neither the selection of the
  minimum over about ten curves on the same held-out samples nor the sampling noise of the crossings is corrected.
  The rates are one-draw estimates of an information upper bound, not confidence bounds.
- The unrestricted-equals-output comparison is empirical, not a theorem here: decoding at layer l can only produce
  outputs reachable from layer l (FORMAL 4.1), and the trained decoder may not find the preimage, so a gap need not
  mean undertraining of the encoder.

Where the learned curves do not reach an error level within the trained price range (lambda >= 1e-4):

| net | error level | cells not reached by learned codes | of these, reached by the linear reference |
|---|---|---|---|
| feistel 16x16 | 0.001 | 6 | 0 |
| feistel 16x16 | 0.003 | 6 | 0 |
| feistel 16x16 | 0.01 | 6 | 0 |
| feistel 16x16 | 0.03 | 6 | 0 |
| feistel 16x16 | 0.1 | 6 | 0 |
| feistel 16x16 | 0.3 | 6 | 0 |
| relu 16x16 | 0.001 | 10 | 10 |
| relu 16x4 | 0.001 | 21 | 21 |
| relu 32x32 | 0.001 | 9 | 9 |
| relu 32x32 | 0.003 | 3 | 3 |
| relu 32x8 | 0.001 | 33 | 33 |
| relu 32x8 | 0.003 | 5 | 5 |
| tanh4 16x16 | 0.001 | 11 | 11 |
| tanh4 16x16 | 0.003 | 1 | 1 |

### 3.1 Phase 2, first measurement (P2.0 part 1, CPU, approved by main 15:5xZ): how linear are the anchored codes?

Per net, layer and class: results/P2_identity_table.md (script scratch/report_P2_identity.py; raw:
results/P2_identity_part0/1.json from scratch/p2_identity.py). ReLU nets, medians over the four shapes, the seeds and the
prices lambda = 0.1 and 0.01 (error levels 10% and 1% for the bit ratios):

| layer | class | linear share of the code | alignment with G: cos^2 / energy captured | bits: linear reference / best code | bits: best code with gates in the cloud / linear reference | units switching inside the data cloud (explorer codes) |
|---|---|---|---|---|---|---|
| input | local | 1.000 | 0.99 / 0.99 | 1.25 | 0.80 | 100% |
| L/4 | local | 1.000 | 0.97 / 0.98 | 1.68 | 0.60 | 81% |
| L/2 | local | 1.000 | 0.97 / 0.98 | 1.49 | 0.67 | 62% |
| 3L/4 to L-1 | local | 1.000 | 0.98 / 1.00 | 1.30 | 0.77 | 38% |
| input | unres | 0.49 | 0.72 / 0.92 | 7.6 | - | - |
| L/4 | unres | 0.90 | 0.83 / 0.91 | 2.7 | - | - |
| L/2 | unres | 0.97 | 0.91 / 0.95 | 1.8 | - | - |
| 3L/4 to L-1 | unres | 0.99 | 0.96 / 0.99 | 1.4 | - | - |
| output | unres | 0.99 | 0.99 / 1.00 | 1.4 | - | - |

Linear share = R^2 of the best linear map from the layer's state to the code mean. G = E[J^T J], the Jacobian
linearisation from the output back to the layer. Alignment = cos^2 of the principal angles between the code's
directions and G's top directions, and the share of G's energy those directions capture relative to the best ones.
The linear reference is a linear code on G's directions with water-filled power and linear decoding, run through
the real net.
- The anchored local codes are exactly linear in the layer: none of their ReLU gates switches inside the data cloud,
  so as registered they test a linear encoder with a nonlinear decoder, not a mixture of linear pieces. They mostly
  read the linearisation's directions (median alignment 0.97-0.99). In 44 of 251 ReLU (code, price) points, mostly
  at the lowest rates (1-2 code directions), they read other directions (cos^2 below 0.5).
- From the output backwards, the linear reference needs 1.3-1.7 times the bits of the best code found at the local
  layers, and 1.4 times at the output itself.
- The anchored linear-encoder model, reading about the same directions, already needs about 1/1.3 of the linear
  reference's bits from L/4 on (bits ratio 1.26-1.30). That gain comes from power allocation and nonlinear decoding.
- Gates inside the data cloud, i.e. a learned mixture of linear pieces, give the largest further saving at L/4 and
  L/2: the explorer codes need 0.6-0.8 of the linear reference's bits. Most units switch inside the cloud at the
  early layers and fewer late (100% at the input, 38% late).
- The unrestricted codes are far from linear at the early layers (linear share 0.49 at the input), as expected for
  codes that carry the output's own coordinates.
- Controls (one seed each). Linear net: every statistic is 1.0 and the linear reference is optimal, which is the
  known answer. Chaotic tanh: the local codes are linear and the linear reference is about optimal at the input
  (1.0) but 1.5 times the best code late. Feistel: at l <= 8 no code direction is above the noise within the price
  range; at the late Boolean layers the linear reference needs about 4.9 times the best code's bits.
- Registered prediction H1 (P2.0) is missed. The ReLU part holds: linear share >= 0.8 on 54 of 54 ReLU (net,
  layer >= L/4, class) points at lambda = 0.01. The tanh part fails: the linear share is 0.52-1.0, not below 0.5. The
  reason is that the anchored local codes are linear on every net, because their gates never enter the cloud, so this
  statistic cannot separate the nets as registered. The useful contrast is the bits ratio, not the linear share.

### 3.2 Phase 2, P2.0 parts 2-4: code maps between consecutive layers (amendment P2.0-a; 56 cell pairs)

Per net, layer and class: results/P2_maps_table.md (scripts scratch/p2_maps.py, scratch/report_P2_maps.py; raw
results/P2_maps_all.json; v2 after review r3: rank-truncated fits and gate probes on the noisy codes, scratch/p2_maps_v2.py,
results/P2_maps_v2_table.md, raw results/P2_maps_v2_all.json; the numbers below are v2). For each grid layer l, the phase-1 code at l and a code trained the same way at l+1 (model
r0, lambda = 0.01, held-out inputs):
- **The next code is almost a deterministic function of the current one** (R^2 of an MLP map 0.94-1.00 everywhere).
  **From L/4 on, that map is almost exactly affine** for both classes (ReLU affine R^2 0.98-1.00, equal to the MLP's;
  chaotic tanh 0.93-0.98).
  The exception is the step from the input layer, where the local code map is 0.73-0.74 affine (the first ReLU
  acts on the code). The late 32x32 local codes, 0.72-0.79 in the first version, are 0.997-1.000 affine with rank
  truncation (MEASURED, v2; the first-version values were near-singular regressions).
- **The real layer, applied to the decoded code and re-encoded, reproduces the next code** with R^2 0.94-0.99 (0.76
  at the 32x32 input layer), and it agrees with the fitted MLP map at R^2 0.71-0.99.
- **Gates: on the code means, but not on the codes as received (v2, MEASURED).** A probe on the noiseless code means
  at l predicts every gate of layer l+1 of the ReLU nets with at least 95% held-out accuracy (local class); the
  unrestricted means predict none at the input, 25% / 59% at L/4, 62% / 94% at L/2 and 81% / 100% at 3L/4 (16x16 /
  32x32). Refitted on the codes as received (code mean plus the unit channel noise), the ReLU share falls to the
  constant base rate (medians over both nets): local 0.41 against base 0.38 at L/4, 0.58 / 0.58 at L/2, 0.77 / 0.75 late, and 0 / 0 at the
  input (unrestricted the same within 0.05). The mean probe accuracy gain over the base rate is 0.02-0.04 from L/4
  on (0.32 at the input layer, where no gate reaches 95%). So on the ReLU nets the received code does not determine the
  next layer's gates beyond what the base rate gives. The chaotic tanh net (one seed) differs: its received local
  codes still predict gates well above the base rate (mean gain 0.42-0.47; share at 95%: 1.00 input, 0.81 L/4,
  0.06-0.12 later, base 0 everywhere).
- **Rotation.** On the ReLU nets, the next code's subspace is the image of the current one under the cloud-averaged
  Jacobian only at the first step (cos^2 0.89-0.92). Later it rotates away (0.22-0.51, v2). Those later codes have few
  active directions, so the statistic is fragile there.
- **Registered predictions, all missed.** H2: agreement >= 0.8 holds on 36/36 ReLU pairs, but the chaotic tanh net
  also agrees (0.91-1.0, registered < 0.5). H3: compression does not make the map more linear (26/36 in v2, 25/36
  in the first version; registered >= 75%). H4 came out reversed: ReLU median 0.31 (v2; 0.24 first version), tanh 0.82.
- **The control contrast.** On the chaotic tanh net the per-layer code maps are as predictable, as affine and as
  close to the layer map as on the ReLU nets (the gate probe on received codes is the one step statistic that differs). In these runs, how simple one step looks does not separate ordered
  from chaotic nets; how the bit counts grow with depth (section 2) does.
- **In certified-linear terms, stated as empirical affine fits of the code means (they certify nothing):** across one
  layer the anchored codes' means are affine to R^2 0.98-1.00 from L/4 on, except at the first layer.
  The cost of linearity sits in the code itself rather than in the step between codes: per layer, the linear-piece
  ratio is 1.25-1.7 and the learned-mixture ratio is 0.6-0.8 (section 3.1).

### 3.3 Phase 2, unscrambling: the cost of forcing simple code dynamics (Part U, corrected rerun U-c; 84 runs)

**Extra bits for forcing the next code to be an affine function of this one** (corrected penalty, amendment U-c;
lambda = 0.01; price = (Lagrangian(mu) - Lagrangian(0)) / lambda summed over both layers of the pair = change in rate +
change in distortion / lambda; median over pairs; ReLU 24 pairs from 16x16 and 32x32, 3 seeds; chaotic tanh 4 pairs
from one network seed; MEASURED):

| mu | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|
| ReLU, all pairs | 0.03 | 0.19 | 0.38 | 0.42 |
| ReLU, L/4 on | 0.03 | 0.15 | 0.23 | 0.22 |
| chaotic tanh, all pairs | 0.18 | 0.40 | 1.08 | 2.25 |
| chaotic tanh, L/4 on | 0.06 | 0.18 | 1.05 | 1.50 |

**Rate and distortion, separately (MEASURED).** With the corrected penalty the cost is extra rate, not lost signal.
Over the pair (both layers summed), the distortion term barely moves, and the rate rises:
- ReLU, L/4 on, mu = 1: pair rate +0.00 to +0.69 bit, pair distortion term -0.01 to +0.15 (bit units).
- tanh at mu = 1: pair rate +1.16 (L/4), +1.13 (L/2), +0.44 (3L/4); distortion -0.08 to +0.02.
- Per layer (medians, mu = 1): the current code gains rate and loses distortion (ReLU L/4 on: rate +0.51, distortion
  -0.38; tanh: +2.50 / -1.74). The next code gives up a little rate and gains distortion by about the same amount in
  bit units (ReLU -0.34 / +0.44; tanh -1.37 / +1.66). It moves along its own rate-distortion curve rather than
  shrinking. A guard holds its variance at no less than half its starting value.
- Extra bits at matched distortion (pairs where both layers' nMSE moves by less than 5%, as registered): only 5 of 24
  ReLU pairs and no tanh pair qualify at mu = 1, so the table above is the main measure.
- At the input pair the cost is larger on both: ReLU 1.65 (1.32 to 3.14) and tanh 2.17 bits at mu = 1, 5.6 and 6.1 at
  mu = 3. Here the ranges overlap (4 of 6 ReLU pairs are below tanh).

**Does it separate ReLU from tanh?** In absolute bits, yes from L/4 on at mu >= 1. At each of L/4, L/2 and 3L/4, the
tanh pair costs more than all 6 ReLU pairs:
- mu = 1: 1.10 against ReLU at most 0.73; 1.05 against 0.50; 0.46 against 0.36.
- mu = 3: 3.00 against 1.19; 1.50 against 0.73; 0.97 against 0.28.
- It also holds at lambda = 0.1 (ReLU at most 0.19, tanh 0.32-1.17 at mu = 1) and for the identity-up-to-scale map
  (ReLU at most 1.16, tanh 1.20-2.07 at mu = 1).
- It does not hold at mu <= 0.3: at L/4 the tanh pair is below every ReLU pair.
- Relative to the pair's own Lagrangian it does not separate. The tanh codes carry about 4 times more bits (next code
  21.4 against 5.5 bits at mu = 0). At mu = 1 the relative rise is 1.3-1.9% for tanh from L/4 on, and 0.1-3.6% for
  ReLU.
- There is one tanh network seed, one pair per layer position.

Registered predictions (U1-U4 rescored unchanged on U-c; results/Uc_scores.json):
- U3 hit (28/28): the MLP ceiling does not move.
- U4 hit (12/12): identity up to scale reaches 0.8 of the MLP ceiling from L/2 on.
- U1 missed (7/18): from L/4 on, the affine map reaches 0.9 of the MLP ceiling on 14 of 18 ReLU pairs, but the
  Lagrangian rises by more than 2% on 8 of 18.
- U2 missed (0/4): the tanh maps stay affine-predictable (lin / mlp 0.86-0.97), and its Lagrangian rises by only
  1.3-2.8% (registered: > 5%). The registered contrast is relative, and in relative terms there is none.
- Numerical caveat: 26 of the 420 affine-predictability values (all ReLU) are negative, down to -227. These are
  near-singular regressions, as in the first version of the code maps (section 3.2). They can only lower the U1
  count.

Reading: forcing affine code dynamics costs a fraction of a bit on ReLU from L/4 on. On the chaotic net it costs
about 0.5-3 bits at mu >= 1 at the same layer positions, more than every ReLU pair, but about the same fraction of
its larger code. On this evidence (one tanh seed), the absolute cost of forcing separates ordered from chaotic
dynamics and the relative cost does not.

Tables: results/Uc_table.md (scratch/report_U.py Uc); rate and distortion per layer: results/Uc_split.md (lambda 0.01,
affine), results/Uc_split_0.1_linear.md, results/Uc_split_0.01_diag.md (scratch/report_U_split.py); ReLU range
against tanh: results/Uc_sep.md (scratch/report_Uc_sep.py). Runs: runs/U/Uc_*.json (on-demand VMs bitacct-16/17
after the spot VM bitacct-15 was preempted).

**Superseded: the first Part U run (runs/U/U_*.json, results/U_table.md).** Review r3 (gpt-6-astra) found that its
penalty's variance normaliser was detached from the gradient. That rewards shrinking the next code: there the next
code gave up rate while distortion rose (tanh, input pair, mu = 3: rate -73 bits, distortion +196 in bit units), so
its costs (ReLU 0.07 / 0.24 / 0.38 / 0.56, tanh 0.25 / 0.58 / 1.64 / 4.89 bits at mu = 0.1 / 0.3 / 1 / 3) mixed
simpler dynamics with loss of signal. Its scores were U1 6/18, U2 1/4, U3 28/28, U4 12/12.

### 3.4 More exploration of the local class (amendment P1-B; 13 of 53 cells, final for this report)

Results: runs/P1Y/Y_*.json, scored by score_P1v2.py (B1, B2), included in R* above. Repeated spot preemptions (the last,
bitacct-14, at 00:02Z on 3 November after it finished the noisy cells) and L4 stockouts cut this part short; no
further P1-B cell arrived, and the 40 remaining cells were not relaunched (the remaining GPU budget went to the
corrected Part U). Rescored at 00:06Z on 3 November: unchanged. On 13 early-layer
local cells (16x4, 32x8, 16x16):
- B1 missed (9 of 13). The registered test compares the best P1-B code, including the 12000-step explorers, with
  the P1-A explorers, and on 4 of 13 cells that code is more than 5% lower.
- The six extra 6000-step restarts alone move R*(0.1) by at most 5% on 12 of 13 cells.
- Training twice as long (12000 steps) is within tolerance on 10 of 13 cells (B2 hit), but lowers the input-layer
  counts by 4-9% (32x8 seed 5000: 29.4 to 27.1 bits with the primary crossing, 8.1%).
- So the input-layer local counts still fall with longer training. They are upper bounds, not converged values.

### 3.5 Phase 2, P2.0 part 5: do the code maps settle to one shared map in the noisy nets? (amendment P2.0-b)

Results: results/P2_noisy_maps.json (script scratch/p2_noisy_maps.py; summary numbers scratch/summ_noisy.py); all 36
noisy cells: 16x16 (layers 8-13) and 32x32 (layers 16-21), 3 seeds each. Codes trained with the shared-noise
distortion at six consecutive layers (lambda = 0.01):
- The code maps between consecutive layers are less affine than in the noiseless nets: affine R^2 0.55-0.80 on 16x16
  and 0.64-0.93 on 32x32, with the MLP fit of each step close to it (0.60-0.80, 0.68-0.93).
- A map fitted at layer l and applied at l+k, after orthogonal Procrustes alignment of the codes, transfers poorly.
  The MLP fit is below 0 on 43 of 60 transfers (at most 0.59). The affine fit is positive on 50 of 60 (at most 0.83);
  at l+4 it is 0.34-0.55 on 32x32 and -0.96 to 0.59 on 16x16.
- H5 (T fitted at l scores R^2 >= 0.8 at l+4 on the noisy 32x32 nets) is missed, 0 of 3, for either fit (the 32x32
  numbers are unchanged from the earlier partial analysis; the rerun reproduced the 16x16 seed 5000 values to within
  0.003).
- This is a failed transfer under this alignment. It is not proof that no shared dynamics exist: the codes at
  different layers can have different numbers of active directions, which Procrustes alignment cannot repair.


## 4. Earlier measurements (before phase 1): fixed-budget codes, ARC and narrow-deep

### 4.1 Fixed budgets, all nets: without the shortcut early layers need many more bits (MEASURED)
Best local code found for nMSE 0.1 (channel bits, median over nets; results/A1_table.md, S1_best.md):

| net | l = 0 | L/4 | L/2 | 3L/4 | L-1 | output's own count |
|---|---|---|---|---|---|---|
| ARC 256 x 32 (l = 0, 8, 16, 24, 31) | >256 | 135 | 43 | 25 | 24 | 25 |
| narrow-deep 8 x 256 (l = 0, 64, 128, 192, 255) | 15 | 1.6 | 1.7 | 1.6 | 1.7 | 1.7 |
| 16 x 4 | 22 | 15 | 12 | 10 | (= 3L/4) | 10 |
| 32 x 8 | >32 | 21 | 11 | 8 | 8 | 9 |
| 16 x 16 | 27 | 6 | 4 | 3 | 3 | 3 |
| 32 x 32 | >32 | 5 | 2 | 2 | 2 | 2 |

ARC also needs > 256 bits at l = 2 and 4. At 1% error ARC needs > 256 bits at every l <= 8, about 210 at l = 16, 119 at
l = 24, 135 at l = 31; the best codes at l = 24-31 beat the Gaussian bound on the output (median 170), so the output's
own count at 1% is at most about 114-168 bits on these nets; at 0.1% every ARC layer needs > 256 (Gaussian bound
404-500). Narrow-deep: the state is a ray from layer 8-32 on, so the output is one number times a fixed vector: 1.7,
3.3, 5.0 bits for 10%, 1%, 0.1%. An unrestricted encoder (shown the exact output) needs the same bits at every layer
of the small nets up to training (flat curves in figs/bits_by_layer.png; 1500 steps 4.6-5.9, 3000 steps 2.4-3.5 against
the output's 1.9-3.0).
Comparisons asked for in the brief:
- The relevance lane's linear-feature count N(l, 0.1) on ARC (208, 96, 56, 20, 12, 12 at l = 0, 4, 8, 16, 24, 31):
  about 2 bits per direction at l >= 16, many more where linear features are not enough.
- Gaussian reverse water-filling over the output covariance gives the output-space channel count exactly (19-39 bits
  at 10%, four ARC nets; learned output codes within 1-2%, A1.1). Over the true-law gradient-energy spectrum of layer
  l (the local high-resolution price) it predicts the local count within a factor 1.3 at l >= 8 and overprices early
  layers (334-770 bits where the measured count is > 256) (HEURISTIC, FORMAL 4.4). The degree-1 relevance spectrum
  prices only the linear part (FORMAL 4.3).

### 4.2 Were the earlier counts converged? (certificates of the fixed-budget codes)
- Known answers. (i) Output codes against the Gaussian bound: ARC learned output codes match it within 1-2%; the
  small-net learned output codes (no linear start) sit 0.5-2.4 bits above it (training). (ii) Unrestricted against
  the output's count: flat but 1.5-2.5x high at 1500 steps, within 0.5-1 bit at 3000 steps or with the S7 objective.
  (iii) Planted toy (a target that depends on exactly k sign bits): sign-bit training measures 2, 4-6 and 12 bits for
  k = 2, 4, 8 (unrestricted); the one-ReLU-layer encoder, which can represent the answer exactly, plateaus at nMSE
  0.08, 0.17 and 0.61 (S5.1 missed). Sign-bit training has real optimisation overhead.
- Quantiser optimality (S5, l = 0 and 4/8 so far): the per-code centroid improves on the learned decoder by <= 2% on
  97% of points (decoders are optimal for their encoders); one unrestricted Lloyd step cuts the error a lot (by > 20%
  on 93% of points, 6-30x at 8-16 bits), but re-training the local encoder to emit those codes recovers less than half
  of that on 94% of points (S5.5). So at early layers the local count is set by the restriction, not by optimisation.
  Restarts: sign-bit counts vary by more than 1.5 bits between restarts on 3 of 5 cells (dead codes at random rates,
  S5.4 missed).
- Joint quantisation (S6; 2^b-vector codebooks): b_VQ(10%) = 5.6, 7.5, 8.2 (16 x 4), 7.9, 6.7, 8.0 (32 x 8), 3.2
  (16 x 16) against Gaussian bounds 6.3, 12.3, 10.2, 8.7, 7.0, 10.4, 3.0: a joint binary index needs no more bits than
  the Gaussian channel count (below it on 6 of 8 nets; S6.2 missed in that direction). The earlier binary overhead (b/C
  ~1.3-1.4 for per-coordinate codes, 2-3x for learned sign bits) was encoder limitation; reachable codebooks (actual
  outputs as reconstructions) cost < 0.5 bit more on 6 of 8 nets.
- Bounds on the output's own rate (results/bounds_small.json): Shannon-type lower bound from the top 1-3 principal
  projections and the Gaussian upper bound: 16 x 16 1.2-1.4 <= R(10%) <= 1.9-3.4; 32 x 32 1.5-1.6 <= R <= 1.7-2.8;
  16 x 4 1.7-3.0 <= R <= 6.3-12.3; 32 x 8 2.4-3.4 <= R <= 7.0-10.4.

## 5. Background for phase 2: scrambling, the curriculum autoencoder, shared noise, estimation (earlier runs)

### 5.1 Binary codes in jointly trained chains (S2: 44 local and 12 unrestricted transition cells, held out)
- Whatever part of a next-code bit the current code predicts at all, a linear threshold function of the current
  code's bits predicts: it captures a median 96% (88-100% on non-degenerate cells) of the gap between the constant
  predictor and the Bayes limit. After a random relabelling of the same codes (same joint statistics, geometry
  destroyed) the same kind of map captures 27% (local) and 9% (unrestricted). Degree-2 maps add almost nothing; 3-input
  juntas come close but stay below: each next bit is a dense, smooth (threshold) function of many current bits.
- Registered in-sample statistics: degree-1 threshold accuracy >= 0.9 on 100% of cells and closer to the threshold
  baseline than to the permutation baseline on 99% (S2.1 hit). The exact Walsh spectrum over all 2^b codes reaches 0.8
  weight at degree <= 2 on only 39% of cells (S2.2 missed): it includes the transition's arbitrary behaviour on
  unreachable codes (coverage 0.2-14% at 16 bits), which the held-out statistics exclude.
- Unrestricted chains did not become identities (S2.3 missed): training does not find the identity chain that exists
  (FORMAL 4.2); their transitions are threshold-like too.

### 5.2 Real-valued codes (S9, 16 x 16 local chain so far)
Each learned map between consecutive layers' codes is 74-98% linear (degree-1 share in the Gaussian convention; fit
R^2 0.86-0.97), and each output coordinate depends on 3-8 of the 8 inputs: dense and smooth. Baselines: random linear
map 0.99-1.0, random sine map 0.02 (S9.1, S9.2 hit). The fixed isotropic noise pins the code coordinates up to
rotations and shifts, and these statistics are invariant to both.

### 5.3 ARC's output-relevant coordinates (B-ARC; dev net 9100, nets 4000 and 4001)
Between consecutive layers the whitened output-relevant coordinates map almost linearly: R^2 of the best linear map
from l = 20 on is 0.91-0.997 on nets 4000 and 4001 (0.92-0.995 on the dev net); 16 -> 20 is 0.885-0.914, and the
multi-step pairs 16 -> 24 and 16 -> 32 are 0.80-0.88, where an MLP adds up to 0.08 (BA.1 missed). Their sign bits
follow threshold maps within 0.03 of an MLP on every pair from l = 16 (BA.2 hit). Under an input perturbation of
correlation 0.999 the fraction of flipped sign bits grows from about 0.04 at l = 8 to 0.09-0.10 at l = 31 (factor
2.0-2.3; BA.3 registered at most 2, missed narrowly): gradual, not an avalanche (results/BARC_scores.json).

### 5.4 The loss is information, not noise
At 16 bits the whole next code is predictable from the current one only 8-37% of the time (82-93% per bit). One-step
decoding separates two losses (results/S2_posthoc_table.md, 14 transitions):
- the next code knows more than the current one (median gain 0.019 nMSE, up to 0.17: 0.35 -> 0.18 on 32 x 8's first
  step), because its encoder sees the state one layer later;
- the best prediction of the next code from the current one decodes worse than the current code itself (median loss
  0.034, 0.007-0.084): the two codes' cells do not nest, so each translation loses part of what the current code had.
The next code is a deterministic function of the state, so none of this is randomness; if the unpredictable bits were
output-irrelevant (training noise), the predicted code would decode about as well as the true next code, and it
decodes worse than even the current code.

### 5.5 Your curriculum autoencoder (S3: 16 x 16 three nets, 32 x 32 two nets)
Phase A plain autoencoder of h_l with 32 bits; phase B weight moves to the far horizons; phase C the budget shrinks
from 32 to 8 bits; reconstruction floors 0 and 0.05 (results/S3_table.md).
- The autoencoder stage alone is useless for the output at the input layer: 32 bits reconstruct x to 52-75% error in
  its own metric, and the output from the decoded x is worse than predicting the mean (nMSE 0.96-2.4).
- Moving the weight to the far horizons fixes this from l = L/4 on: 6.5-10 bits for 10% error (the output's own count
  is 2-3). At the input layer the encoder (it sees only x) never learns the shortcut: still > 32 bits.
- Shrinking the budget at the end does not lower the counts.
- Without the floor the code drifts away from describing the state (reconstruction error 0.12 -> 0.31 at l = 4, 0.20
  with the floor); that drift is toward the shortcut, which it does not reach at these depths (S3.1 missed: the drift
  is weaker than registered).
- Its transitions are threshold-like on reachable codes too (degree-1 accuracy 0.95-1.0 against 0.58-0.95 for a random
  permutation on the same codes).

### 5.6 Noisy nets scored with the same future noise (S4; 16 x 16 and 32 x 32, sigma = 1, three seeds)
The decoded state runs through the real future with the same noise as the original. The count becomes layer-dependent
for the unrestricted encoder too, in the direction of memory: 16 x 16: 1, 1.8, 7.1, 10.4, 18.9 bits for 10% error at
l = 0, 4, 8, 12, 15 (output 18.6), tracking the memory share 0.00, 0.04, 0.16, 0.55, 0.99; 32 x 32: 1, 1, 2.1, 15.6 at
l = 0, 8, 16, 24 (S4.2 hit). The local encoder needs more at l >= L/2 (S4.3 hit). The three scores are exactly nested
on all 1240 points (S4.1 hit): the per-sample shared-noise error is 3-30x the noise-averaged error of the same code, and
the output law (Gelbrich distance <= 0.01) and the output mean (error 1e-6 to 2e-4) are matched far better. So the
shared-noise score blocks the shortcut, but it asks for each sample's path, which is much more than the mean needs.

### 5.7 What it means for estimation
- Code propagation through the binary chains (local class; exact pushforward over all 2^b codes from the exact or
  factorised law of the input code): error of the mean 6e-4 to 0.1 at b = 4-16 with no downward trend in b
  (results/S2_table.md). Plain MC at the same number of evaluations has error 2^-b (measured N x error 0.3-1.6): the
  deductive estimate wins only at b = 4 and loses by 60-5000x at b = 16, and by more at equal compute (each code
  evaluation runs MLPs costing 30-400 MC samples).
- Compressed simulator with linear-Gaussian code dynamics (S10): the code law propagates in closed form and the last
  decoder is integrated over 8 dims; error 8e-5 (16 x 16) and 1.1e-4 (32 x 32) already at 256 points and flat to 65536
  (a bias floor, S10.2 hit). MC on the net reaches the same error at about 1e4 samples and RQMC (scrambled Sobol,
  2-22x better than MC on these nets) at about 10^3. Its per-layer fit costs little relative to free (MLP) dynamics:
  final nMSE 0.29 against 0.23 (16 x 16) and 0.48 against 0.41 (32 x 32), dominated by the 12-bit linear input code.
- The reason is section 5.4, not scrambling: the transitions are simple, but no step can add what the next layer's code
  knows and each step loses some of what it had. A deductive estimator would need an input code that already holds
  the output's bits, which is the problem itself; local input codes need far more bits than the output has (4.1).
- In the noisy regime the mean depends on the early state only through the noise-averaged map, whose error is tiny at
  low rates, so for the mean the relevant count is the noise-averaged one, not the shared-noise one (section 5.6).

## 6. Pre-registration scorecard (parts before phase 1)
- A1 (ARC): A1.1, A1.3, A1.5, A1.6, A1.7, A1.8, A1.9, A1.10, A1.11 hit; A1.2 missed (mis-specified: a layer-24/31
  code can beat a learned output-space code, which is not a bug, FORMAL 4.8); A1.4 missed.
- A2 (narrow-deep): A2.2 hit; A2.1 missed (3 of 6 nets have a layer >= 32 more than 0.6 bit above the one-number
  value: heavy-tailed radius, Gaussian quantiser design).
- S1: S1.2, S1.4, S1.5, S1.7 hit; S1.1, S1.3, S1.6 missed (learned sign/channel codes undertrained at 1500 steps).
- S2: S2.1 hit; S2.2, S2.3, S2.5 missed (see 4.1, 4.4).
- S3: S3.3 hit; S3.1, S3.2, S3.4, S3.5 missed (section 5).
- S4: S4.1, S4.2, S4.3 hit.
- S5-S10 (partial): S5.2, S5.3, S5.5, S6.1, S6.3, S9.1, S9.2, S10.2 hit; S5.1, S5.4, S6.2 missed; S7, S8, S10.1 pending.
- B-ARC (nets 4000, 4001; results/BARC_scores.json): BA.2 hit (sign-bit transitions within 0.03 of an MLP for every
  pair from l = 16 on); BA.1 missed (consecutive pairs from l = 20 on have linear R^2 0.91-0.997, but 16 -> 20 on net
  4001 is 0.885 and the multi-step pairs 16 -> 24 and 16 -> 32 are 0.80-0.88, with MLP gains up to 0.08); BA.3 missed
  narrowly (sign flips at rho = 0.999 grow by 2.02x and 2.29x from l = 8 to 31: gradual, but above the registered 2x).
- S11: deferred to phase 2 (plan change 02:55Z); S11 S7 reference rates are in runs/S11_*.

## 7. Process
- Plan changes by main/Kelvin during the night: small nets on a GPU first (23:05Z); Kelvin's curriculum (S3, 23:4xZ);
  shared-noise scoring (S4); convergence certificates (S5) and joint quantisation (S6, 01:3x-01:4xZ); the encoder
  chooses its bits (S7, 02:00Z, real-valued at 02:10Z) with Bernoulli and real-valued scrambling arms (S8, S9);
  simple-dynamics simulator (S10, 02:25Z); net-level controls (S11, 02:40Z); then phase 1 (02:55Z: the best curve
  per layer, one lambda-conditioned model per layer, certificates; 03:20Z: identity anchor annealed upward, the
  linear analytic reference, SNR equivalence) and Kelvin's near-identity hypothesis for phase 2 (registered P2.0).
  Each was registered before its runs; S9, S10 and the S11 spectra were stopped at 02:57Z (phase 2).
- Deviations (notes/DEVIATIONS.md): training steps cut (6000 -> 3000 -> 1500) for GPU time; two spot preemptions
  (results restored); CUDA OOM with 16 processes (jobs rerun with a memory gate); some unrestricted chains skipped for
  seeds 5001-5002; the straight-through surrogate standardised after dead codes; analysis on the monotone envelope.
- Compute: CPU at nice 19 (ARC, narrow-deep, linear codes, analyses). GPU: L4s only, one at a time until 02:55Z,
  then at most two: bitacct-1 to -4 (spot; two preempted), bitacct-5 to -9 (on-demand), $13.52 for VMs 1-6 and at
  most about $10.5 for VMs 7-9 at list price (GPU-LOG.md); cap $25 (raised from $15 at 02:55Z).
- Reviews (gpt-6-astra, model line verified each time): round 1 on FORMAL and the code (16 findings, applied in
  FORMAL 4.1-4.8 and the code). Round 2a on the phase-1 code (review/review_r2a.raw, 12 findings): the Feistel
  decoder squash and Jacobian were fixed before any Feistel run; the scorer was rewritten (conservative crossings,
  single model and best code, every error level with not-covered separated, one candidate set, pending until
  complete, KA2 relabelled empirical); the post-pass of amendment P1-A adds explorer restarts with gates inside the
  data cloud, the encoder width separated from the decoder width, a residual output-space centroid, held-out powers
  for the linear reference and gate activity; the one-model requirement (finding 5) is reported as not met
  everywhere. Round 2b on this report (review/review_r2b.raw, 9 findings): the provisional framing, the unrestricted
  exceptions, the single-model column, measured uncertainty only, counts instead of rounded percentages, the
  coverage table, softer control wording, the C5v2 convergence rule, correct source attribution and the same
  candidates for the log-linear secondary are applied; the explorer initialisation now uses pilot samples (cells
  finished before 12:28Z used evaluation samples for that initial fit only). Scorer v2 was test-run on the 12 cells
  already inspected before its first hash; each later change is hashed and logged.
  Round 2c (review/review_r2c.raw, final check after the post-pass, 'major revision' on wording): applied: R* is
  called an envelope over several models with the single model r0 beside it (median r0 - R* gaps given); the
  certificates are said to test the anchored models, not the explorer codes; the centroid result is stated as
  measured; the decoder-width attribution is softened; post-pass comparisons use the primary crossings; coverage
  is derived from the manifest (55 of 57 ReLU local cells); measured extrema replace rounded claims (prior refit up
  to 13%, 70 points above the linear reference within tolerance, linear control within 0.12 bit).
  Round 3 (review/review_r3.raw, phase 2, 'major revision'): applied: the unscrambling cost is called a
  Lagrangian-equivalent cost and split into rate and distortion; the detached-normalisation flaw of Part U is stated
  (a corrected rerun needs GPU budget); B1 rescored as registered (9/13, missed); the noisy transfers are reported
  for both fits; the code-map statistics are stated as empirical fits of the code means, with the numerical and
  noisy-code caveats (v2 rerun: scratch/p2_maps_v2.py); the ordered/chaotic contrast is softened (one tanh seed).
  Provenance: scratch/p2_identity.py and report_P2_identity.py were run before they were hashed;
  report_P2_maps.py was written after the analysis output existed (both recorded in prereg_hashes.txt).

