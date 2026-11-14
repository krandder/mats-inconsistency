# Very narrow, very deep networks (width 8-16, depth 64-1024): literature, random nets, deductive estimation, trained nets (29-30 October 2026)

Lane: narrow-deep. Work directory: `work/narrow-deep/`. Runs from 21:36Z on 29 October to 01:04Z on 30 October;
the follow-up of section 10 (CPU only) ran 01:14Z-02:18Z.
- Pre-registration: `PREREG.md`, Part 1 plus seven dated amendments, and Part 2 for the follow-up. The sha256 hashes
  and times are in `prereg_hashes.txt`, and those hash times are authoritative.
- Review: one GPT-6 review (`review/`). Its verdict was "major revision"; section 8 lists what it found and what
  changed, including two code fixes that were rerun. Section 10 had its own GPT-6 review (section 10.6).
- GPU: one on-demand L4 for 1.5 h, about $1.31 of the $5 cap. The VM was deleted and the deletion verified
  (`GPU-LOG.md`).
- Status labels used throughout: PROVEN, MEASURED, HEURISTIC.

Kelvin's ask (via main): "What is the theory/literature for very narrow but very deep networks? Can they be trained
for interesting tasks?" and "I would hope that the structure of the matrices, regardless of the gates, would give us
a clear bound on what dimensions can matter first. We should explore the narrow-but-deep networks; maybe n<=16, but
L>=64."

## Summary for Kelvin

**In brief.**
- Plain narrow-deep ReLU nets synchronise: the weights carry every input onto one trajectory. The angle between two
  inputs falls by about one e-fold every n layers, at a rate consistent with a half-width Lyapunov gap we can
  compute. After that, depth is cheap for estimation, and every difficulty we found sits in the first
  ~n ln(1/tolerance) layers.
- The gate-free constructions we tried (unmasked weight products, random masks) do not tell which directions
  matter. The gates of the inputs actually in use do, once they have synchronised.
- Residual narrow-deep nets do not synchronise. There, a sample-free Gaussian closure beats the samplers we tried at
  small budgets, at every depth from 16 to 1024. It has a bias floor, and it fails on trained nets.
- Narrow-deep nets can be trained. A width-16 RNN does 2nd argmax at T = 64 at 44% accuracy (28x chance). A width-16
  residual MLP of depth 64 sorts 16 numbers approximately. Plain nets of that width failed at depth 64 in our
  recipe, as the literature suggests.

1. **Literature (77 references checked, 4 corrected; section 1).**
   - What a narrow net can represent depends on the sense of approximation. In L^p on R^n, or uniformly, width at
     most the input dimension fails and one extra unit suffices. On compact domains in L^p the threshold is lower.
     Depth can replace width.
   - The depth/width ratio L/n governs initialisation. Fluctuations grow like exp(c L/n), angles between inputs
     collapse with an e-fold depth of about n, and nets are "born dead".
   - With skips or gates, narrow nets train at 100-10,000 layers: thin ResNets of 16-64 channels, 50-unit highway
     nets. Tiny recurrent states bridge 1,000-step lags.
   - We found no paper training a plain MLP of width <= 16 and depth >= 64 on a real task, and no learned narrow-deep
     sorting network.
2. **Random plain nets synchronise, then die whole (section 2).**
   - At n = 16 the median angle between two inputs shrinks at 0.0867 per layer, averaged over 64 nets (between-net
     standard error 0.005). The half-width Lyapunov gap predicts 0.0868; that exponent is proven only for infinitesimal
     separations, so this is a consistency check, not a precision test.
   - At n = 8: 39 of 64 nets lose every one of 4,096 sampled inputs by depth 256 (sigma = 0). The mean dead
     fraction, 0.63, matches 1 - (1 - 2^-8)^256.
   - After synchronisation, one direction per path survives, and one block of 2.5-4.5 directions on average (1-8 per
     net) covers 99% of the sensitivity of all inputs.
3. **Your hope (a bound from the matrices regardless of the gates): not with the constructions we tested
   (section 2.4).**
   - Unmasked weight products and random masks capture about chance (4-28%) of the sensitivity, against 52-93% for
     the best single direction.
   - Random-mask products point in directions spread over 7 of 8 and 13 of 16 dimensions.
   - Even the largest of 1,024 sampled random-mask gains exceeds the real paths' gain by e^5 to e^14.
   - The obstacle is that a path's masks are tied to its own state.
   - What works after synchronisation is "weights plus the gates of the inputs in use": marginal gate probabilities,
     or often a single reference trajectory, but not always (0.07 to 1.0 by net at n = 8).
   - Residual nets have no small relevant subspace.
   - In one trained net, the noise-trained RNN, a random-mask average of W_hh does find the single relevant direction.
4. **Deductive estimation on residual nets, the headline the coordinator asked for (section 3.1).**
   - Given the weights and the noise scales, the sample-free Gaussian closure (K2, Stein-linearised) gets E[output]
     to nMSE 2.3e-5 to 4.0e-4 for the counted cost of 18-35 sampled paths.
   - At that budget its MSE is 16-250x lower (median 43x, 24 nets) than the samplers tried (RQMC, RQMC and MC with a
     regression control variate), and 140-1,300x lower than plain Monte Carlo.
   - This holds at L = 16, 64, 256 and 1024: the median error at L = 1024 is 1.1-1.2x that at L = 64.
   - Its error is a bias: RQMC matches it at about 100-1,000 paths and wins beyond.
   - It underestimates tail probabilities 3x to 60,000x.
   - These nets are 82-87% linear and 96-97% quadratic in the input, while the input direction still carries 90-97%
     of the output variance. That fits "easy because the map is simple, not because the input is forgotten", but it
     is a diagnostic, not a proof of the mechanism.
5. **Plain nets: the easy case for depth, not for the input (section 3.2).**
   - Given the law at layer 16-64, a cheap deterministic deep part (K2 per radius bin) finishes the estimate to nMSE
     1e-7 to 2e-5 on the five healthy nets. This is a paired oracle that reuses the truth's paths.
   - The shallow part defeats every sample-free method we tried. K2 is off by nMSE 0.016-27, and naive cell
     enumeration faces >= 1e5 (n = 8) and ~1e6 (n = 16) gate histories.
   - Sampling only the shallow part (a hybrid) gains 1.6-4x over the best sampler, and nothing at large budgets.
   - These nets have not forgotten their input: its direction explains 80-92% of the output variance. The deep map
     is close to rank one.
6. **Trained nets (section 4).**
   - RNN width 16: 29-44% at T = 64 (chance 1.6%) and 33-40% at T = 32.
   - Sorting MLP: residual at depth 64 reaches nMSE 0.034-0.041. Plain at depth 64 fails (0.81-1.14), and the failed
     net has synchronised like a random one.
   - On all six trained nets estimated, K2's error is a floor of nMSE 0.013-0.22. That is about what MC gets at K2's
     own budget of ~35 paths, and 110-1,800x worse than RQMC at 1,024 paths.
7. **For M1/M3 (section 5): no major advance.** Narrow width moves the M1 obstacle without removing it:
   - plain nets: into a shallow block of depth about n ln(1/tol);
   - residual nets: into a depth-stable closure bias;
   - trained nets: the closure fails.
   - The concrete next step is a closure whose bias vanishes on the residual family, with K >= 3 or Gaussian
     mixtures split along the non-quadratic directions. The coordinator declined this for the random residual nets,
     which are 82-87% linear, and asked instead why K2 fails on the trained nets (item 8).
8. **Follow-up: why K2 fails on trained nets (section 10, pre-registered as `PREREG.md` Part 2).**
   - The failure is spread out, not concentrated in a few units. In the sorters, the worst 8 of 1,024 (layer, unit)
     pairs carry 5% of the gate-probability error, and the output error builds up over layers ~8-48.
   - In the RNNs the gate error is concentrated: one unit carries 38-46% of the one-step error, mostly in the first
     steps. But correcting the early steps helps little. Restarting K2 from exact moments after step 8 leaves the
     noiseless RNN's output error unchanged and lowers the noisy RNN's by 1.6x. In the noiseless RNN, the last 8
     steps alone regenerate the full error.
   - Splitting the Gaussian at the k <= 8 worst-gate units gains at most 1.13x over the same closure without splits
     (random splits: 0.92-1.08x). That is refuted by the 3x criterion, as pre-registered.
     - At equal FLOPs every such closure loses to RQMC: by >= 93x with exact moments, and by 4-51x with the cheaper
       Stein moments at k = 1-2.
     - An oracle with exactly conditioned pieces, added after the review, gains at most 1.13x on the RNNs at any
       layer, and at most 1.6x over K2 at the sorters' layers 2-8.
     - Late in the sorters (layer 48), exactly conditioned splits gain up to 10x, but only on top of exact moments up
       to there.
   - Exact radius integration (positive homogeneity) helps the task-relevant tails only modestly: 2.9-3.8x in
     variance for large sorting errors, and 1.5-1.8x for the RNNs' confident wrong answers.
   - For M1: on these trained nets, Gaussian closures with a few gate splits do not work. This holds for the
     sequential splits and for exactly conditioned single-layer splits, except late in the sorters, which no closure
     reaches accurately. One GPT-6 review, verdict "revise"; its findings are addressed in 10.6.

## 1. Literature: what is known about very narrow, very deep networks

Every reference was checked on 29 October against its arXiv abstract, its proceedings page or the full text.
`work/narrow-deep/notes/refs.md` has 77 entries: 65 verified as recalled, 4 corrected, 3 bibliographic only, and 5
with one unverified detail such as a venue.

**What narrow nets can represent.** The answer depends on the sense of approximation; the statements below keep
each paper's setting.
- Uniformly, or in L^p on all of R^n, ReLU nets whose hidden layers are no wider than the input fail, however deep:
  - width <= d_in fails (Lu et al. 2017, L1 on R^n; Hanin and Sellke 2017, uniform on compacta; Johnson 2019, for
    layered nets of width <= d_in with activations approximable by one-to-one maps);
  - every decision region of such a net is unbounded (Beise, Dias Da Cruz and Schroeder 2021);
  - under their assumptions (pyramidal widths, strictly monotone activations such as leaky ReLU), disconnected
    decision regions need a layer wider than the input (Nguyen, Mukkamala and Hein 2018).
- One extra unit suffices in those settings:
  - width d_in + 1 approximates any continuous scalar function on a compact set (Hanin and Sellke 2017);
  - width d_in + 4 does so in L1 on all of R^n (Lu et al. 2017);
  - for vector outputs, the minimum width is max(d_x + 1, d_y) in L^p on R^n (Park, Yun, Lee and Shin 2021).
- On compact domains in L^p the threshold is lower: max(d_x, d_y, 2) (Kim, Min and Park 2024; Cai 2023 for leaky
  ReLU). Width d_in + d_out + 2 is sufficient for general non-affine activations (Kidger and Lyons 2020).
- Depth can replace width cheaply:
  - any ReLU net is approximated by one of width O(d), at a linear-factor increase in parameters (Vardi, Yehudai and
    Shamir 2022);
  - a ResNet whose blocks have one hidden ReLU each is universal (Lin and Jegelka 2018).
- Our random nets have width equal to the input dimension, at or just below these thresholds depending on the
  setting. They are also bias-free, hence positively homogeneous, so neither they nor our residual variant can
  approximate arbitrary continuous targets whatever their width.
- Narrow deep nets compute functions shallow ones cannot: Telgarsky's sawtooth at width ~2-3 (Telgarsky 2016;
  Eldan and Shamir 2016). Gradient descent does not learn such functions when shallow nets cannot approximate them
  weakly (Malach, Yehudai, Shalev-Shwartz and Shamir 2021; Malach and Shalev-Shwartz 2019).

**Trainability at initialisation is set by depth/width.**
- In fully connected ReLU nets, the variance of the squared activation length, and the second moment of the squared
  input-output Jacobian entries, grow like exp(c sum_l 1/n_l), i.e. exp(c L/n) at constant width (Hanin and Rolnick
  2018; Hanin 2018). The mean of the squared entries stays fixed at He initialisation; their fluctuations grow.
- One input's squared output norm is approximately log-normal, with log-variance about 5 sum_l 1/n_l. This is an
  asymptotic statement with explicit error bounds (Hanin and Nica 2020, Comm. Math. Phys.). It is over random
  weights for one input; it is not a statement about one net across inputs.
- The NTK stays random and moves during training unless L/n is small (Hanin and Nica, ICLR 2020).
- The aspect ratio L/n controls the deviations from the Gaussian limit (Roberts, Yaida and Hanin 2022).
- Collapse:
  - at fixed width, the probability that a ReLU net is born dead tends to 1 with depth; with symmetric init,
    P <= 1 - prod_l (1 - 2^-n_l) (Lu, Shin, Su and Karniadakis 2020);
  - the angle between two inputs vanishes exponentially, with an e-fold depth of about n at finite width, against
    ~1/l at infinite width (Jakub and Nica 2024, JMLR). Their expansion is for n >> 1, and Monte Carlo checks it only
    at n = 256 and 1024;
  - unshaped ReLU covariances degenerate as depth/width grows (Li, Nica and Roy 2022);
  - representations lose rank with depth when there is no normalisation (Daneshmand et al. 2020; Feng et al. 2022).
- Narrow linear nets train slowly:
  - gradient descent on a width-1, depth-k linear net needs time exponential in k (Shamir 2019);
  - a sufficient condition for gradient descent on plain deep linear nets is width growing with depth (Du and Hu
    2019);
  - residual ones need only d_in + d_out (Zou, Long and Gu 2020).

**Residual and normalised narrow nets; infinite-depth limits.**
- Branch scales with bounded sum of squares, such as 1/sqrt(L), give a stable infinite-depth limit (Hayou et al.
  2021, "Stable ResNet"). With iid initialisations, 1/sqrt(L) is the only non-trivial scaling, and the limit is a
  neural SDE, not an ODE (Marion, Fermanian, Biau and Vert).
- In the joint limit of depth and width with d/n fixed, ReLU ResNet outputs are log-Gaussian, with parameters set by
  d/n (Li, Nica and Roy 2021). Width and depth limits commute under 1/sqrt(L) (Hayou and Yang 2023).
- Diffusion limits: Peluchetti and Favaro (2020). Neural ODEs: Chen et al. (2018); whether trained ResNets actually
  discretise them: Sander, Ablin and Peyre (2022), Cohen et al. (2021).
- Depth-muP (Yang, Yu, Zhu and Hayou 2024); depthwise hyperparameter transfer (Bordelon, Noci, Li, Hanin and
  Pehlevan 2024).
- Thin residual nets train without normalisation:
  - Fixup: 16/32/64-channel ResNets up to 10,000 layers on CIFAR-10;
  - SkipInit (De and Smith 2020);
  - ReZero: 10,000 fully connected layers fit to the CIFAR-10 training set, width not stated.
- Batch norm without skips makes gradients explode with depth (Yang et al. 2019).

**RNNs and SSMs: narrow state, deep in time, shared weights.**
- AlgZoo (ARC, January 2026): bias-free ReLU RNNs on 2nd argmax, hidden size 2-32, lengths 2-10. Accuracy falls
  with length unless width grows:
  - hidden 16 reaches 94.8% at length 10, and 95.3% for the blog's example M_{16,10};
  - hidden 8 reaches 42.7% at length 10;
  - no zoo model has a sequence longer than 11.
- Vanilla RNNs have vanishing and exploding gradients (Pascanu, Mikolov and Bengio 2013).
- An identity-initialised ReLU RNN with 100 units does the adding problem at T = 150-400 and pixel-by-pixel MNIST
  at 784 steps (Le, Jaitly and Hinton 2015).
- Gating widens the trainable region (Chen, Pennington and Schoenholz 2018). An LSTM with two memory cells bridges
  a 1,000-step lag among 1,000 distractors (Hochreiter and Schmidhuber 1997).
- SSMs keep a small state per channel but use many channels. S4 uses state size 64. Mamba's default per-channel
  state is 16, and its Table 10 shows perplexity still improving from N = 1 to 16. See also LRU (Orvieto et al.
  2023).

**Which interesting tasks narrow-deep nets have been trained on.**
- With skips or gates, routinely:
  - highway MLPs, 50 units x 100 layers, on MNIST, matching 10-layer nets in training loss; plain 71-unit nets
    degrade beyond 20 layers (Srivastava, Greff and Schmidhuber 2015);
  - ResNet-1202 with 16-64 filters: 7.93% CIFAR-10 test error;
  - Fixup at 10,000 layers;
  - a ResNet with one unit per block learns a 2-D disk that width-2 MLPs cannot (Lin and Jegelka 2018);
  - two-cell LSTMs on 1,000-step lags, and AlgZoo's tiny RNNs.
- Without skips, only short nets:
  - widths 10-15 with 10-25 layers reach about 95% on MNIST in one epoch, while 30 layers of width 15 drop to 82%
    (Jakub and Nica 2024, Tables 7-8);
  - widths 15-20 train on MNIST at a speed that tracks sum 1/n_l (Hanin and Rolnick 2018).
- Not found:
  - any plain MLP of width <= 16 and depth >= 64 trained to non-trivial accuracy on a real task;
  - any learned narrow-deep sorting network;
  - any gradient-descent learning of Telgarsky-type targets.
- Section 4 adds a width-16 RNN on 2nd argmax at T = 64 and a width-16 residual MLP of depth 64 that sorts 16
  numbers.

## 2. Random narrow-deep nets (n = 8 and 16, L = 64-256)

Nets (`lib/nd.py`):
- **Plain**: ARC's critical He-ReLU form, z_l = W_l h_{l-1}, a_l = ReLU(z_l), h_l = a_l + s_l eps_l, W_l iid
  N(0, 2/n), no bias.
- **Residual**: u_l = W_l h_{l-1}, h_l = h_{l-1} + V_l (ReLU(u_l) + s_l eps_l) / sqrt(L), with V_l iid N(0, 1/n).
- Noise: s_l = sigma x (RMS of layer l's clean activation), with sigma = 1e-3 unless stated.
- Sample sizes: 64 nets per width for the census, 4 per configuration otherwise. At n = 8 the per-net analyses use
  nets that are still alive at depth L (Amendment 1), so they condition on survival.

### 2.1 Collapse at initialisation: inputs synchronise, then narrow nets die whole

| plain, sigma | median angle between two inputs at depth 4 / 16 / 64 / 128 / 256 (rad) | gate disagreement between two inputs, same depths | nets (of 64) in which every one of 4,096 sampled inputs has passed an all-off layer, by L = 64 / 128 / 256 |
|---|---|---|---|
| n = 8, 0 | 0.62 / 0.015 / 0 / 0 / 0 | 25% / 4% / 0 / 0 / 0 | 10 / 25 / 39 (dead: output exactly 0) |
| n = 8, 1e-3 | 0.62 / 0.026 / 0.009 / 0.009 / 0.008 | 25% / 4.8% / 1.4% / 2.3% / 1.9% | 10 / 21 / 37 (> 95% of inputs; noise revives them, and no net has > 95% of its inputs dead at these depths) |
| n = 16, 0 | 0.72 / 0.15 / 3.1e-3 / 1.4e-5 / 0 | 27% / 8.3% / 0.67% / 0.08% / 0 | 0 / 0 / 0 |
| n = 16, 1e-3 | 0.72 / 0.15 / 0.013 / 0.012 / 0.008 | 27% / 8.3% / 1.2% / 1.1% / 1.0% | 0 / 0 / 0 |
| residual, n = 16, L = 256 | 1.6 / 1.6 / 1.5 / 1.5 / 1.3 | 50% / 50% / 48% / 46% / 43% | (paths do not die in a residual net) |

Angles and gate disagreements are computed over pairs of inputs that have never passed an all-off layer, so with
noise they describe a selected population.

- **Rate (MEASURED; consistent with a single-path exponent that is PROVEN for infinitesimal separations).**
  - The fitted log-slope of the median pair angle over layers 32-128, across 64 nets at n = 16, is -0.0867 per
    layer (median -0.0862). Between nets the standard error is about 0.005.
  - The prediction for two infinitesimally close inputs is the gap between the radial exponent and the first
    angular exponent of one path's Jacobian (Theorem C and Corollary C1 of the lyapunov lane), with the same
    conditioning m >= 2 for both: gap_1 = (1/2) E[psi(m/2) - psi((m-1)/2) | m >= 2], m ~ Bin(n, 1/2). It is
    0.0868 at n = 16 and 0.250 at n = 8, and approaches 1/n for large n.
  - The measured rate is consistent with gap_1 within that standard error. It is not a precision test: the angles are
    finite, the population is selected for survival, gates can cross, and observations are correlated across layers.
  - Jakub and Nica's large-n rate 1/n gives 0.0625 at n = 16.
  - At n = 8 the measured rate over layers 8-40 is -0.26, against gap_1 = 0.250. A layer with a single active unit
    also synchronises two inputs exactly: in 26 of 61 nets the median pair was exactly synchronised by depth 32.
- **Collective death (the mechanism is exact and noiseless; the rate is measured).**
  - While a noiseless path's state is nonzero, its next active count is Bin(n, 1/2) whatever its past (from the proof
    of Theorem C). It survives L layers with probability (1 - 2^-n)^L.
  - Inputs whose states lie exactly on one positive ray share every later gate, so they die at the same layer. Approximate
    synchronisation does not by itself imply identical gates.
  - At n = 8, the mean over 64 nets of the fraction of dead inputs is 0.19 / 0.41 / 0.63 at depth 64 / 128 / 256,
    against 1 - (1 - 2^-8)^L = 0.22 / 0.39 / 0.63.
  - Per net the fraction is bimodal. At L = 256, 20 nets have < 5% dead inputs, 39 have > 95%, and 5 are in between.
    This is the "born dead" collapse of Lu et al., here without biases. It is measured on the 4,096 sampled inputs,
    not proven for every input.
  - At n = 16 the per-path formula predicts 0.4% dead by depth 256, almost all of it from rare whole-net deaths; the
    expected count is 0.25 nets of 64. We measured 0.015% (sigma = 0).
  - With sigma = 1e-3 a dead path is revived by the next layer's noise wherever some paths survived. None of the 64
    noisy nets has more than 95% of its inputs dead at depth 64, 128 or 256. In the 4 of 16 noisy nets examined that came close to dying, noise carries 23-89% of the output
    variance. In the other 12 it carries 4% or less, and 1% or less in 10 of them.
- **Residual nets do not collapse.** 41-50% of gates disagree between inputs at every depth, and pair angles stay at
  1.3-1.6 rad.

### 2.2 Per-path surviving directions against remaining depth

N_surv(J; delta) counts the singular values of a path's Jacobian J_x = d a_L / d z_c that are within a factor delta
of the largest; 1,024 paths per net and cut. Plain nets, sigma = 1e-3, L = 256, mean over 4 nets:

| L' (layers to go) | 4 | 8 | 16 | 32 | 64 | 128 | 192 |
|---|---|---|---|---|---|---|---|
| n = 16: N_surv(0.01) | 5.0 | 3.9 | 2.5 | 1.7 | 1.3 | 1.1 | 1.0 |
| n = 8: N_surv(0.01) | 2.4 | 1.8 | 1.3 | 1.0 | 1.0 | 1.0 | 1.0 |
| n = 8: share of paths with s_2/s_1 < 1e-12 | 0.01 | 0.24 | 0.33 | 0.79 | 0.92 | 1.0 | 1.0 |

- **One direction per path survives deep in the net.** By Theorem C (lyapunov lane), J^T J is a product of
  independent half-width Gaussian matrices plus one rank-one radial term. The angular part decays relative to the
  radial one at gap_1 per layer.
  - Measured s_2/s_1 decays at 0.089-0.098 per layer for L' >= 64 at n = 16 (gap_1 = 0.087) and 0.26 at L' = 64 at
    n = 8 (gap_1 = 0.250).
  - The surviving input direction is the radial row r = J^T e_N of Theorem C. It collects leaks from the first
    layers after the cut, so it is not simply the state's own direction (Euler's identity fixes the output side,
    not the input side).
- rank(J_x) <= min_l m_l(x) exactly, where m_l is the number of active units at layer l (PROVEN, noiseless). The
  last row counts numerically rank-one Jacobians (s_2/s_1 < 1e-12), which include paths that passed a layer with one
  active unit.
  - For a noiseless path that survives 64 layers, the probability of such a layer is 1 - (1 - 8/255)^64 = 0.87.
    The measured 0.92 is consistent with it but a weak test: after synchronisation these events are shared by all
    inputs of a net, so 4 nets give little independent information.
- Residual nets keep every direction: N_surv(0.01) = n at every L'. The spectrum spreads slowly with depth, so
  N_surv(e^-1) falls from n to about 0.35 n by L' = 250.

### 2.3 How many directions one block for all inputs needs

- M = E_x J_x^T J_x is the sensitivity that one block for all inputs must cover (Theorems B and D of the lyapunov
  lane). It is computed over every path with a nonzero Jacobian, including paths revived by noise.
- S = Sigma^(1/2) M Sigma^(1/2) weights M by the spread of the state.
- k*_F(A; 1e-2) is the number of directions that leaves at most 1% of tr A outside the block.

Plain nets, sigma = 1e-3, L = 256, mean over 4 nets:

| cut c (layer) | 4 | 8 | 64 | 128 | 224 | 248 |
|---|---|---|---|---|---|---|
| n = 16: k*_F(M; 1e-2) / k*_F(S; 1e-2) | 12.3 / 7.0 | 10.8 / 4.3 | 3.3 / 1.0 | 2.5 / 1.0 | 2.8 / 1.3 | 4.0 / 1.3 |
| n = 8 | 5.3 / 3.8 | 3.5 / 2.3 | 2.8 / 1.0 | 3.5 / 1.3 | 3.3 / 1.3 | 3.0 / 1.0 |

- After synchronisation, one block needs 2.5-4.5 directions on average to cover M (more near the output, where few
  layers remain); single nets need 1-8. The spread-weighted S needs 1-1.3.
- At the input cut, one block needs most of the space.
- Wide nets are the opposite: there 185 of 195 directions were needed at depth 32.
- Residual nets need all n at every cut: k*_F(M; 1e-2) = n, and PR(M) is 9-16 at n = 16.

### 2.4 Kelvin's hope: a bound from the weights alone on which directions can matter

Candidate subspaces at a cut, k = 1 direction:
- **lin**: the unmasked product W_L ... W_{c+1};
- **rand**: random masks, E_D J_D^T J_D with D_l iid Bernoulli(1/2) drawn independently of the state. Almost none
  of these sequences is realised by an input: there are at most ~1e5-1e6 realised gate histories (3.2) against
  2^(n L') sequences;
- **avg**: the product with each layer's marginal gate probabilities, estimated from 1,024 trajectories;
- **att**: the Jacobian along one reference trajectory (one surviving input);
- **pcs**: the state's top principal component, for reference.

Each candidate is scored by the share of the sensitivity tr M it captures; the optimum is the top eigenvector of M.
Plain nets, sigma = 1e-3, mean over 4 nets:

| net and cut | optimum | lin | rand | avg | att | pcs | PR of top directions over random-mask products (zero products dropped) | log gain: actual median / largest of 1,024 random-mask products / product of norms |
|---|---|---|---|---|---|---|---|---|
| n = 16, L = 256, c = 4 (before sync) | 0.52 | 0.12 | 0.08 | 0.19 | 0.12 | 0.05 | 12.9 of 16 | -17.8 / -4.4 / +242 |
| n = 16, L = 256, c = 64 | 0.93 | 0.04 | 0.08 | 0.93 | 0.91 | 0.11 | 12.8 | -11.8 / -0.9 / +184 |
| n = 16, L = 256, c = 128 | 0.82 | 0.08 | 0.19 | 0.79 | 0.79 | 0.07 | 13.0 | -7.4 / +2.0 / +123 |
| n = 16, L = 64, c = 32 | 0.76 | 0.13 | 0.10 | 0.70 | 0.60 | 0.03 | 12.9 | -1.1 / +4.1 / +31 |
| n = 8, L = 256, c = 128 | 0.92 | 0.28 | 0.25 | 0.72 | 0.50 | 0.05 | 7.0 of 8 | -20.8 / -7.3 / +113 |
| n = 8, L = 64, c = 32 | 0.83 | 0.15 | 0.05 | 0.66 | 0.50 | 0.12 | 6.9 | -6.3 / +3.2 / +28 |

**What fails (MEASURED; tested constructions only).**
- The gate-free constructions we tried do not predict the relevant directions. The unmasked product and random masks
  capture 4-28% of the sensitivity, close to chance (1/n = 6%) at n = 16.
- The top directions of random-mask products spread over 7 of 8 and 13 of 16 dimensions, so no small subspace
  contains them.
- Gains:
  - the largest gain among 1,024 random mask sequences already exceeds the actual median gain by e^5 to e^14. That
    is a lower bound on how far a bound uniform over masks must sit above the real paths;
  - the product of operator norms is only an upper bound, e^30 to e^260 above.
- This does not prove that no weight-derived construction can work. It shows that these natural ones fail.
- Interpretation: a real path's masks are tied to its own state, because ReLU(W h) = D W h with D chosen so that the
  result is nonnegative. Random masks drop that tie, and most of their sequences are not realisable.

**What helps (MEASURED).**
- The gates of the inputs actually present, after synchronisation:
  - At n = 16 and c >= 64, the marginal-gate product (avg) is within 0.01-0.11 of the optimum on every net. The
    single reference trajectory (att) is as good on most nets and 0.1 below on one.
  - At n = 8 both lose more: avg captures 0.72 against 0.92 at c = 128. att ranges from 0.07 to 1.0 by net,
    depending on whether the reference path sits on the shared trajectory.
- So the useful object is "weights plus the gates of the inputs in use". A single trajectory often suffices, but not
  always.
- It helps only after synchronisation, about ln(theta_0 / theta) / gap_1 layers, i.e. roughly n layers per e-fold
  of angle. Before that (cut 4) the best candidate captures 0.19 against an optimum of 0.52.
- Residual nets have nothing concentrated to find. At k = 1 the optimum itself captures only 8-40%, and every
  candidate comes within a few points of it. But their whole state is only n-dimensional.

**The pre-registered score.**
- The pre-registered share of tr S makes every candidate look perfect wherever the state has collapsed: at c = 64-192
  (n = 16) the tail is 0.000-0.04 for lin, rand, avg and att alike. The state there is one-dimensional, so any block
  not orthogonal to it captures it. Near the output (c = 248) the tails reopen: lin 0.08, rand 0.12.
- So by the prereg's rule "a gate-independent bound exists" at those cuts, but only because the state has collapsed,
  not because the weights identify the sensitive directions. The sensitivity share above is the informative score
  (Amendment 1).

## 3. Deductive estimation where the whole state is small

**Setup.**
- Estimand: E[output], a vector. Score: nMSE = |estimate - truth|^2 / tr Var(output), as ARC uses.
- Truth: plain MC with 2^20-2^22 paths, whose own expected nMSE is 2-10e-7. All estimators are scored against the
  same truth, so differences below about 1e-6 are not resolved.
- Cost: an approximate operation count under the flopscope conventions of `work/arc-noisy` (matmul M N (2K - 1),
  normal draw 16, cdf 96, ...), quoted in units of one full MC path. "35 paths" means the counted FLOPs of 35 sampled
  paths. With N paths, plain MC's expected nMSE is exactly 1/N.
  - Not counted: binning, covariance estimation, regression fits, Sobol generation and inverse CDFs.
  - Residual HYB-lin is priced like a plain branch, and the deductive readout of the trained nets is not counted.
  - None of these changes a conclusion by more than a small factor. No wall-clock claim is made.
- Noise scales s_l are part of each net's definition. They come from a 2^14-path pilot that no estimator is charged
  for, so "sample-free" means: given the weights and these scales.
- Simple samplers:
  - MC; antithetic pairs (ANT); scrambled-Sobol RQMC on the input;
  - the same-sample regression control variate on the input x (E x = 0 is known), applied to MC and to RQMC
    (MC-reg, RQMC-reg). It is biased at O(1/N); its MSE is measured as is.
  - The "best simple sampler" is the lowest of these at each budget (16 replicates). In the own-budget comparison of
    3.1 only RQMC, RQMC-reg and MC-reg were run, with RQMC at the nearest power of 2 below the K2 budget (32 paths
    against 35, 16 against 18-19). ANT was never best in the full runs.
- Deductive estimators:
  - K2: ARC's Gaussian closure. Mean and covariance pass through each ReLU in closed form, using exact bivariate
    ReLU moments (copied from `work/arc-noisy`).
  - K2-Stein: the same with the covariance linearised (Stein's lemma; means and variances stay exact). It costs n^3
    rather than n^2 x 2,500 FLOPs per layer.
- Hybrid HYB(l0): MC for the first l0 layers, then a deductive deep part. The sampled states are split into 8
  log-radius bins, and each bin is pushed through the remaining layers by K2-Stein ("st") or by the noiseless map
  applied to the bin mean ("lin").

### 3.1 Residual nets (all n directions alive): a sample-free closure beats the best sampler at small budgets

K2-Stein against the samplers at K2-Stein's own budget (`runs/resdepth_*.json`; 4 nets per row at L = 16 and 64, 2 at
L = 256 and 1024; the 4 nets at n = 16, L = 256 of `runs/p3/` give 5.4e-5 to 1.3e-4):

| residual, sigma = 1e-3 | K2-Stein nMSE (per net) | K2-Stein cost (MC paths) | best of RQMC / RQMC-reg / MC-reg at that budget | plain MC at that budget | K2 (exact moments) nMSE, cost |
|---|---|---|---|---|---|
| n = 16, L = 16 | 1.2e-4, 3.6e-5, 2.9e-5, 5.7e-5 | 35 | 3.4e-3, 3.0e-3, 4.8e-3, 3.0e-3 (29-162x worse) | 0.029 | 1.0e-4, 4.4e-5, 2.9e-5, 6.9e-5; 512 |
| n = 16, L = 64 | 2.3e-5, 7.3e-5, 7.2e-5, 3.9e-5 | 35 | 3.2e-3, 3.3e-3, 4.0e-3, 3.4e-3 (45-140x worse) | 0.029 | 2.2e-5, 8.0e-5, 7.3e-5, 3.7e-5; 517 |
| n = 16, L = 256 | 8.6e-5, 1.1e-4 | 35 | 3.5e-3, 3.0e-3 (28-41x worse) | 0.029 | 8.7e-5, 1.1e-4; 518 |
| n = 16, L = 1024 | 4.5e-5, 9.0e-5 | 35 | 4.8e-3, 4.9e-3 (55-107x worse) | 0.029 | 4.5e-5, 9.0e-5; 518 |
| n = 8, L = 16 | 3.1e-4, 3.7e-4, 2.1e-4, 1.5e-4 | 18 | 4.8e-3, 7.0e-3, 6.4e-3, 7.2e-3 (16-49x worse) | 0.056 | 2.8e-4, 3.6e-4, 2.4e-4, 1.2e-4; 405 |
| n = 8, L = 64 | 1.1e-4, 3.3e-4, 2.1e-4, 2.0e-4 | 18 | 1.0e-2, 1.0e-2, 6.3e-3, 5.1e-3 (25-95x worse) | 0.056 | 1.0e-4, 3.1e-4, 2.1e-4, 1.9e-4; 411 |
| n = 8, L = 256 | 4.0e-4, 5.1e-5 | 19 | 9.9e-3, 1.2e-2 (25-246x worse) | 0.056 | 3.7e-4, 5.2e-5; 413 |
| n = 8, L = 1024 | 2.4e-4, 2.2e-4 | 19 | 8.2e-3, 5.8e-3 (26-35x worse) | 0.056 | 2.4e-4, 2.3e-4; 413 |

- **At its own budget, K2-Stein has 16-250x lower MSE than the samplers tested (median 43x over 24 nets), at every
  depth from 16 to 1024 (MEASURED).**
  - The range is conditional on this menu, the rounding of the RQMC budget and the cost model.
  - RQMC is the strongest sampler. Its error falls like N^-1.5 to N^-2, and it matches K2's accuracy at about
    100-1,000 paths (`runs/p3/*_samplers.json`).
  - Put as cost: K2-Stein reaches in 18-35 paths' FLOPs an accuracy that RQMC needs 5-30x more FLOPs for, and plain
    MC 2,500-43,000 paths.
- **It has a floor.** K2's error is a bias, nMSE 2e-5 to 4e-4. Beyond about 1,000 paths RQMC wins, reaching 1e-7 to
  1e-6 at 16,384 paths, though values that small are at the truth's own resolution.
  - The exact-moment K2 costs 15-23x more than K2-Stein for about the same accuracy. So it only ties RQMC at n = 16,
    and loses 2-4x at n = 8.
- **Depth does not hurt it (MEASURED; P10 hit).** The median nMSE relative to L = 64 is 0.83 / 1.76 / 1.21 at
  L = 16 / 256 / 1024 for n = 16, and 1.24 / 1.08 / 1.11 for n = 8. That is 2-4 nets per point, so the evidence is
  for these nets, not a general law.
  - Its cost grows linearly in L, as a sample's does, so the advantage at equal FLOPs does not shrink with depth.
  - Note that increasing L at fixed 1/sqrt(L) branch scale keeps the total residual variance fixed: it refines the
    same effective "time".
  - Why the error need not grow (HEURISTIC, martingale form): write the mean error as e_{l+1} = e_l + L^(-1/2) V_l b_l,
    where b_l is the closure discrepancy of layer l and is determined before V_l. Since E V_l^T V_l = I,
    E |e_L|^2 = L^(-1) sum_l E |b_l|^2, the average one-layer discrepancy, whatever L is. Bounding the discrepancies
    themselves, and the output variance, is what a proof would need.
  - (The L = 1024 nets share their first 256 W matrices with the others but use fresh V matrices.)
- **The problem is not trivial.**
  - The zero estimate has nMSE 0.17-0.32, and so does pushing the mean through the net.
  - A closure that keeps only variances, dropping all correlations, gets 2.2-2.8e-3, 20-100x worse than full K2
    (`logs/k2variants.log`).
- **Diagnostics consistent with why it works (MEASURED; they do not establish the mechanism).**
  - The input is not forgotten. Its direction explains 90-97% of the output variance and its norm 0.4-4%, and every
    one of 2^20 inputs has its own activation pattern by layer 8 (n = 16).
  - But the map is close to quadratic. A linear function of x explains 82-87% of the output variance and a quadratic
    one 96-97%, at every depth; the plain nets score 8-60% and 10-90%.
  - The pre-activations' marginals stay close to Gaussian (`runs/gauss_*.json`; L = 256, 2 nets per width). Mean
    |skewness|:
    - n = 16: 0.02-0.03 at layer 8, 0.04-0.05 at 32, 0.08-0.13 at 128;
    - n = 8: 0.03-0.05, 0.07-0.13 and 0.10-0.20 at the same layers;
    - it grows to 0.13-0.46 at the output layer;
    - plain nets: 1.0-1.7 from layer 8 on.
  - Small marginal skewness does not prove joint Gaussianity, nor accurate ReLU means near thresholds. The tails
    (3.3) show the closure is not right everywhere.
- **Hybrids do not help here.** Without synchronisation, the radius-binned deep part is biased: 4e-5 to 1e-4 for
  HYB-st, 0.05-0.4 for HYB-lin. HYB loses to RQMC by 10-100x at every budget.

### 3.2 Plain nets: the deep part is easy, the shallow part is the blocker

**K2 from the input fails (MEASURED).**
- It is off by nMSE 0.016-1.0 on the four n = 16, L = 256 nets and by 2-27 on the n = 8 nets. On a net whose output
  is mostly noise it reaches 1.2e7.
- The collapsed state is close to a ray whose radius is widely spread. A Gaussian with the same mean and covariance
  puts mass at negative radius.
- Dropping correlations makes it worse (16-218).

**The deep part, given the law at layer l0 estimated from the truth run's own paths.**
- This is a paired oracle: the same 2^21 paths give the law at l0 and the truth, so shared sampling error can cancel.
  Values near 1e-7 are not independent validations.
- Binned K2 runs from layer l0 to L = 256, one K2 propagation per log-radius bin (16 bins). The first column is the
  share of the output variance that the input explains.

| net (input share) | l0 = 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| n = 16 seed 0 (0.994) | 0.18 | 0.054 | 6.7e-7 | 3.3e-5 | 1.8e-7 | 2.0e-7 |
| n = 16 seed 1 (0.998) | 5e-4 | 4e-4 | 3.2e-4 | 3.6e-4 | 3.7e-7 | 3.7e-7 |
| n = 16 seed 3 (0.988) | 1.0e-3 | 1.7e-3 | 3.1e-3 | 2.8e-3 | 1.9e-5 | 2.0e-5 |
| n = 8 seed 7 (0.994) | 0.053 | 1.7e-3 | 1.4e-5 | 1.7e-5 | 1.7e-5 | 2.1e-5 |
| n = 8 seed 2 (0.964): binned K2 / mean pushed noiselessly | 3.9 / 7.9e-3 | 0.076 / 5.8e-5 | 0.068 / 7.1e-5 | 0.065 / 2.7e-5 | 0.031 / 8.5e-6 | 6.8e-4 / 1.4e-5 |
| n = 16 seed 2 (0.774, near-dead) | 0.24 | 0.024 | 1.8e-3 | 7.6e-3 | 1.9e-3 | 1.6e-3 |

- On the five nets whose input explains >= 96% of the output variance, a cheap deterministic deep part brings the
  error to 1e-7 to 2e-5 beyond l0 = 16-64 (MEASURED):
  - binned K2 on four of them;
  - on n = 8 seed 2, the noiseless push of the mean, not the binned K2.
- The l0 needed to get below 1e-4 is 8-16 at n = 8 and 32-64 at n = 16, consistent with a synchronisation length of
  about n.
- On the three near-dead nets, where noise carries 23-89% of the output variance, neither deep part works.

**Hybrid at equal FLOPs (column (b) of ARC's bar).** n = 16, L = 256, seed 0, nMSE:

| budget (MC paths) | MC | best simple sampler | best hybrid |
|---|---|---|---|
| 256 | 3.6e-3 | 1.3e-3 (RQMC-reg) | 3.1e-4 (HYB16-lin) |
| 1,024 | 6.2e-4 | 2.2e-4 (MC-reg) | 1.1e-4 (HYB32-st) |
| 4,096 | 3.5e-4 | 9.3e-5 (RQMC) | 5.8e-5 (HYB32-st) |
| 16,384 | 3.3e-5 | 1.6e-5 (RQMC-reg) | 2.1e-5 (HYB32-st) |

The hybrid gains 2-4x over the best sampler at 256-1,024 paths, 1.6x at 4,096 and nothing at 16k, where its deep-part
bias of about 1e-5 shows. This is MEASURED on one net; the planned second seed was dropped for time.

**Why no sample-free method we tried gets through the shallow part (MEASURED).**
- The shallow part is a narrow net of depth l0 ~ 16-64 on Gaussian input.
- An explicit enumeration of activation cells needs the joint gate histories of layers 1..l. Among 2^20 inputs:
  - n = 8 (sigma = 0): 3,600-5,300 distinct histories at layer 2, 77,000-82,000 at layer 8, and 90,000-116,000 from
    layer 16 on. Synchronisation stops further splitting. About half of these are seen once, so the true count is
    higher.
  - n = 16: about 1,000,000 by layer 8, i.e. nearly every sampled input has its own history.
  - Each cell would also need a Gaussian integral over a polyhedral cone in 8-16 dimensions.
  - This rules out naive enumeration. It is not a lower bound for every exact method: distinct histories can share a
    map or be aggregated.
- A grid at noise resolution needs (1/sigma)^(n-1) cells.
- So in the plain family the M1 difficulty does not disappear in narrow-deep nets for the methods tried. It moves
  into the first ~n ln(1/tolerance) layers: a net whose depth is proportional to its width, the regime M1 is about.

### 3.3 Tails: P(|output| > t) at the 1e-3 and 1e-4 quantiles

The thresholds are quantiles of part of the truth sample, and the same sample is reused for the exceedance
probability. The truth's relative error is 1.5-2% at q = 1e-3 and 5-7% at q = 1e-4.

**The Gaussian closure is wrong in the tails (MEASURED).**
- We computed the tail of K2's output Gaussian with Imhof's formula. Sampling that Gaussian checks it where its
  probability is >= 1e-6. At 1e-9 the GPT-6 reviewer checked it independently with an exponentially tilted inversion:
  1.898e-9 vs our 1.911e-9.
- At n = 16 the closure's tail is 3-77x too small, at n = 8 400-60,000x too small. The non-Gaussian, roughly
  quadratic part of the map makes the true tails heavier.

**Positive homogeneity gives a much better sampler (MEASURED).**
- A bias-free ReLU net satisfies out(r u) = r out(u) without noise. So the radius can be integrated exactly for each
  sampled direction ("RB"): P = E_u Q_chi_n(t / |out(u)|).
- At equal FLOPs RB lowers the relative MSE 2-90x against MC. Examples:
  - residual n = 16 at q = 1e-3 and 4,096 paths: 0.011 and 0.005 vs MC 0.18 and 0.29;
  - plain n = 8 at q = 1e-4 and 16,384 paths: 0.008 vs 0.45.
- With sigma = 1e-3 the identity is approximate: we rescale the noise to a typical radius, which changes the
  estimand slightly.
- Direct check on one residual net (n = 16, L = 64; 2^21 MC paths vs 2^19 RB directions, `logs/check_rb.log`):
  - q = 1e-3: agreement within 2% with and without noise;
  - q = 1e-4: within 1% with noise, and 12.6% apart without noise (1.7 combined standard errors).
  - The near-dead plain nets were not checked.

**The plain hybrid in the tails varies by net.**
- With the radius integrated, the attractor-linear deep map (HYB32-RB) is 9-200x better than MC on three nets: for
  example 0.0046 vs 0.94 at q = 1e-4, n = 16 seed 1.
- On the other two its error stays at 35-140% relative.

**Not M3.** None of these is a certified bound.

## 4. Trained narrow-deep nets

Training ran on one L4 (`gpu/`, `train/`, GPU-LOG.md).
- **RNNs.** ARC's AlgZoo architecture: bias-free ReLU, h_0 = 0, logits W_oh h_T, trained to output the 2nd argmax of
  a Gaussian sequence.
  - Width 16, T = 64 (4 seeds) and T = 32 (2 seeds), noise sigma = 0 and 0.01.
  - ARC's recipe at batch 4096, on 2^28 sequences (a quarter of ARC's budget).
- **Sorting MLP, the task of my choice.** Sort x ~ N(0, I_16) with width-16 bias-free ReLU MLPs:
  - residual (h <- h + V ReLU(W h), V initialised at 1/sqrt(L)), L = 16 and 64;
  - plain He-ReLU with a linear readout, L = 16 and 64;
  - sigma = 0 and 0.01; Adam, 2^15 steps.
- Why sorting: sorting networks are narrow-deep comparator circuits, ReLU(b - a) is one comparator, and the map is
  positively homogeneous like the nets. The literature check found no learned narrow-deep sorting network.

### 4.1 Trainability (MEASURED; finals on 2^18 fresh inputs, `runs/p4/eval_all.json`)

| model | seeds | result, noise on (per seed) | chance / reference | trains? (PREREG criterion) |
|---|---|---|---|---|
| RNN width 16, T = 64, sigma = 0 | 4 | accuracy 44.1%, 34.8%, 35.6%, 42.7% | 1.6% | yes (> 7.8%) |
| RNN width 16, T = 64, sigma = 0.01 | 4 | 31.2%, 34.4%, 29.3%, 33.1% | 1.6% | yes |
| RNN width 16, T = 32, sigma = 0 / 0.01 | 2 / 2 | 38.1%, 40.4% / 32.7%, 37.1% | 3.1% | yes |
| residual sorting MLP, L = 64, sigma = 0 / 0.01 | 2 / 2 | nMSE 0.036, 0.039 / 0.034, 0.041 | 1 = predict the mean | yes (<= 0.05) |
| residual sorting MLP, L = 16, sigma = 0 / 0.01 | 2 / 2 | nMSE 0.074, 0.082 / 0.075, 0.087 | | no (0.07-0.09) |
| plain sorting MLP, L = 64 | 2 + 1 | nMSE 0.81, 1.14; one run diverged (2e18) | | no |
| plain sorting MLP, L = 16 | 2 + 1 | nMSE 0.32, 0.40 / 0.41 | | no |

- **The RNNs learn, limited by capacity or by the training budget.**
  - Width 16 reaches 29-44% at T = 64, 19-28x chance. AlgZoo's width 16 reaches 94.8% at T = 10.
  - T = 64 is not worse than T = 32 (best 44.1% vs 40.4%), although the PREREG predicted it would be.
  - The best T = 64 seed was still improving at the end (35.7%, 39.8% and 44.1% at the last three checkpoints), so
    longer training may help.
  - Noise at sigma = 0.01 costs 0-13 points of accuracy (7 on average at T = 64).
- **In these recipes, sorting worked only with depth and a residual stream.**
  - Residual depth 64 beats depth 16 (nMSE 0.034-0.041 vs 0.074-0.087). The sort is still approximate: only 15-19%
    of output vectors have every coordinate within 0.1.
  - Plain nets of the same width fail at depth 64 (0.81 and 1.14: barely better than, or worse than, predicting the
    mean) and reach only 0.32-0.41 at depth 16.
  - The failed plain depth-64 net is synchronised exactly like the random ones (4.2). Its gate disagreement is
    0.000 from layer 16 on. Its output depends almost only on |x|: the input direction explains 0.12 of the output
    variance and the norm 0.87. A net like that cannot represent a sort.

### 4.2 Structure of the trained nets (MEASURED, `runs/p4/an_*.json`, best seed per configuration)

| model | gate disagreement between inputs at layer/step 1 / L/4 / L/2 / L | per-path N_surv(0.01) at L' = 4 / L/2 / L-4 | k*_F(M; 1e-2) | share of output variance: direction / norm |
|---|---|---|---|---|
| residual sort, L = 64 | 0.50 / 0.24 / 0.22 / 0.19 | 16 / 10.8 / 10.7 | 12-16 | 0.77 / 0.20 |
| plain sort, L = 64 (failed) | 0.50 / 0.000 / 0.000 / 0.000 | 5.0 / 1.0 / 1.0 | 1-5 | 0.12 / 0.87 |
| RNN T = 64, sigma = 0 | 0.50 / 0.09 / 0.05 / 0.03 | 11.8 / 10.6 / 8.8 | 11-12 | 0.98 / 0.01 |
| RNN T = 64, sigma = 0.01 | 0.50 / 0.08 / 0.05 / 0.05 | 4.3 / 1.8 / 5.5 | 1-2 | 0.75 / 0.20 |
| RNN T = 32, sigma = 0 | 0.50 / 0.19 / 0.11 / 0.07 | 11.8 / 10.5 / 8.5 | 12-13 | 0.96 / 0.03 |

- **The trained residual sorter keeps nearly every direction.** Path Jacobians keep 11-16 of 16 directions, one
  block for all inputs needs 12-16, and 19-24% of gates differ between inputs at every depth. A sorter's Jacobian is
  close to a permutation, so this is expected.
- **The RNNs do not synchronise their inputs.** The angle between two sequences' states stays at 0.2-0.4 rad. But
  the gates settle: by the last step only 3-7% of units differ between input sequences.
- **The noise-trained RNN has a single relevant direction.** For sigma = 0.01 its logits depend on the state at any
  step from 32 on through essentially one direction: PR(M) = 1.0, and the top direction carries 99-100% of the
  sensitivity. At step 4 it carries 84%.
  - There a gate-free object does find the direction: the random-mask average captures 96-100% of it (76% at step
    4), while the unmasked power of W_hh captures 12-87%.
  - Without noise the top direction carries 51-80%, and the random-mask average captures 28-72%.
  - So for this one trained net a gate-free construction (random masks) does find what matters. It reflects the
    net's specific structure, a single integrating direction, not a general property.

### 4.3 Estimation on trained nets (MEASURED, `runs/p4/est_*.json`)

| model | K2 nMSE (cost in MC paths) | K2-Stein nMSE (cost) | best simple sampler at 1,024 paths | best hybrid at 1,024 | K2 / best sampler |
|---|---|---|---|---|---|
| plain sort, L = 16 | 0.013 (1,183) | 0.013 (35) | 2.3e-5 (RQMC-reg) | 1.1e-3 | 570x |
| residual sort, L = 64, sigma = 0 | 0.024 (670) | 0.021 (45) | 8.4e-5 (RQMC-reg) | 6.3e-3 | 280x |
| residual sort, L = 64, sigma = 0.01 | 0.045 (523) | 0.030 (35) | 1.0e-4 (RQMC) | 5.0e-3 | 440x |
| RNN T = 32, sigma = 0 | 0.016 (1,120) | 0.017 (34) | 8.5e-5 (RQMC-reg) | 9.4e-3 | 190x |
| RNN T = 64, sigma = 0 | 0.039 (1,092) | 0.021 (34) | 3.5e-4 (RQMC-reg) | 0.020 | 110x |
| RNN T = 64, sigma = 0.01 | 0.22 (753) | 0.21 (23) | 1.3e-4 (RQMC-reg) | 0.061 | 1,800x |

- On trained nets the closure's error is a floor of nMSE 0.013-0.22.
  - At K2-Stein's own budget of 23-45 paths, that is within 2x of plain MC, and up to 5x worse for the noise-trained
    RNN.
  - Beyond that budget every sampler passes it. At 1,024 paths the best sampler is 110-1,800x better.
  - This matches what the algzoo-noisy lane found for ARC's AlgZoo models.
- The trained sorter's state is a partially sorted vector, far from Gaussian. The random residual nets of section
  3.1 are close to quadratic maps of a Gaussian. That contrast is consistent with K2 working there and failing here,
  but we did not isolate it.

## 5. What this means for M1 and M3

- **Narrow width moves the M1 difficulty without removing it, for every method we tried.**
  - **Plain nets.** The difficulty sits in the first ~n ln(1/tolerance) layers, the synchronisation length. After
    that, depth is cheap: a deterministic deep part per radius bin is enough on healthy nets.
    - Before it, the state is a net of depth proportional to width on Gaussian input. That is exactly M1's regime,
      now at width 8-16.
    - No sample-free method we tried works there. K2 is off by nMSE 0.016-27, and naive cell enumeration faces
      >= 1e5-1e6 gate histories.
  - **Residual nets.** Nothing synchronises, but a sample-free Gaussian closure is depth-stable: the median error at
    L = 1024 is 1.1-1.2x that at L = 64.
    - It beats the samplers tried below about 100-1,000 paths and loses above.
    - This is the closest this program has come to M1's "matches sampling at depths that grow", though only at small
      budgets and only at random initialisation.
  - **Trained nets.** The closure fails (4.3). Section 10 finds the failure spread out rather than concentrated in a
    few units, and gate splits at k <= 8 units do not fix it.
- **What would move it.**
  1. A closure whose bias vanishes on the residual family: cumulants beyond K = 2, or Gaussian mixtures split along
     the few directions where the map is not quadratic. The floor is a bias, not variance, so this is a
     well-defined next step. (The coordinator declined it for the random residual nets; section 10 tests the
     trained-net version.)
  2. For plain nets, a deductive method for the shallow block alone. That block's depth is ~n ln(1/tol), about 16-30
     layers at n = 8, and earlier lanes found no method in that regime either.
- **M3, certified tails: not reached.**
  - The closure gets the tails wrong by 3x to 60,000x.
  - The useful structure for tails is positive homogeneity: integrating the radius exactly makes tails 2-90x cheaper
    to sample. That gives a better estimate, not a certificate.

## 6. Pre-registered predictions: scorecard

gap_1 below uses the conditioning corrected after review (m >= 2): 0.250 at n = 8 and 0.0868 at n = 16.

| prediction | outcome |
|---|---|
| P1: per path, N_surv(0.01) <= 2 at L' >= 32 (n = 8) and L' >= 64 (n = 16) | hit (n = 8: 1.0; n = 16: 1.32 at L' = 64, 1.0 from L' = 192) |
| P1: slope of s2/s1 within 25% of -gap_1 at L' >= 64 | hit (n = 16: -0.089 to -0.098 vs -0.087; n = 8: -0.26 vs -0.25 at L' = 64; at L' >= 128 every n = 8 path is numerically rank one) |
| P1: rank-1 fraction near the binomial at L' = 64 (n = 8) | weak test: 0.92 vs 0.87 for paths conditioned to survive. The measure is s2/s1 < 1e-12, and the events are shared within a net |
| P2: angle < 0.05 rad by depth 64 (n = 8) and 128 (n = 16), sigma = 0 | hit (mean 1.8e-5 and 2.1e-3) |
| P2: log-slope within 30% of -gap_1 | hit (n = 16: -0.0867 vs -0.0868, between-net SE 0.005; n = 8: -0.26 vs -0.25 over layers 8-40, not the registered 32-128, where most n = 8 angles are exactly 0) |
| P2: noise floor of 1e-3 to 1e-2 rad at sigma = 1e-3 | mostly hit (0.008-0.013; the top of the range exceeds 1e-2) |
| P2: gate disagreement < 1% by depth 64 (n = 8) and 128 (n = 16) | hit (0 and 0.08%) |
| P2: dead fraction within 0.1 of 1 - (1 - 2^-n)^L, bimodal per net | hit (0.19 / 0.41 / 0.63 vs 0.22 / 0.39 / 0.63; 20 / 5 / 39 nets below 5% / between / above 95%) |
| P3: k*_F(M; 1e-2) <= 3 after synchronisation and >= n/2 at cut 4; k*_F(S; 1e-2) <= 2 | partial. After synchronisation (c >= 128 at n = 16, c >= 64 at n = 8) the mean k*_F(M) is 2.5-4.5 (n = 16) and 2.8-3.8 (n = 8), above 3 at 7 of 13 cuts, mostly the last few layers, and single nets reach 8. The >= n/2 at cut 4 holds, and k*_F(S) <= 1.3 holds |
| P4: residual nets do not collapse, gates differ > 10% | hit (angles 1.3-1.6 rad, gate disagreement 41-50%) |
| P4: residual N_surv(e^-1) >= n/2 at every L' | miss: it falls to ~0.35 n at L' >= 192 (N_surv(0.01) = n throughout) |
| P5: gate-free candidates capture <= 2k/n + 0.1 of tr S | miss as worded: wherever the state is one-dimensional every candidate captures nearly all of tr S (Amendment 1). On the sensitivity share (tr M), lin and rand capture 0.04-0.28, a hit (<= 0.225 at n = 16, <= 0.35 at n = 8) |
| P5: worst case over masks, PR >= n/2 and sampled sup-gain ratio >= 10 | hit after the review fix that drops zero products (PR 12.8-13.0 of 16, 6.9-7.0 of 8; the largest sampled gain is e^5 to e^14 above the actual median) |
| P5: avg and att capture >= 0.9 of tr S | hit, trivially as above. On tr M, avg is within 0.11 of the optimum at n = 16 after synchronisation but 0.2 below it at n = 8; att ranges from 0.07 to 1.0 by net |
| P6: deep part from the law at l0 reaches 1e-4 for l0 >= 48 (n = 8) and l0 >= 96 (n = 16) | hit on the 5 nets whose input explains >= 96% of the output variance (paired oracle), miss on the 3 near-dead nets. The sigma = 0 arm was not run |
| P7: hybrid >= 0.5 L/l0 better than MC at L = 256 | hit at 256-4,096 paths (5.6-11.6x); miss at 16,384 (1.6x). Against the best simple sampler, only 1.6-4x; one net |
| P8: K2 from the input misses by nMSE >= 1e-2 on plain nets; no sample-free method matches sampling at L = 64, n = 16 (plain) | hit (0.016-27); hit for the methods tried |
| P9: trained nets do not collapse; gates differ >= 10%; K2 loses to MC | residual sorter: hit / hit / hit. RNNs: no collapse (hit), gates settle to 3-7% (miss), K2 loses (hit) |
| P10 (Amendment 5): residual K2 nMSE ratio L = 1024 / L = 64 within 3x | hit (median 1.21 at n = 16, 1.11 at n = 8; 2 nets at L = 1024) |
| Trainability: T = 32 >= 50% | miss (38-40%) |
| Trainability: T = 64 above 5x chance but below T = 32 | first half hit; second half miss (best 44% vs 40%) |
| Trainability: residual sorter nMSE <= 0.05 at L = 64; plain fails at 64 and does better at 16 | hit; hit; hit |

Deviations (Amendments 1-7):
- Plain n = 8 analyses are conditioned on survival.
- The P5 score was supplemented with the sensitivity share.
- The main truths used 2^20-2^22 paths, not 2^24.
- Samplers were trimmed to 16 replicates, and only one plain-net seed ran the sampler comparison.
- The noiseless arm of P6 was not run.
- The MLP at L = 128 was stopped to free the GPU.
- Added after the first residual results: K2's depth scaling, the linearity shares, the Gaussianity check and the
  Imhof tail.
- The review led to two code fixes, now rerun, and a corrected gap_1 (Amendment 7):
  - sensitivity M over all paths with a nonzero Jacobian;
  - random-mask PR without zero products.

## 7. What is proven, measured, heuristic

- **PROVEN** (noiseless, from Theorem C of the lyapunov lane):
  - while a path's state is nonzero, its next active count is Bin(n, 1/2) whatever its past;
  - a path survives L layers with probability (1 - 2^-n)^L;
  - rank(J_x) <= min_l m_l;
  - inputs whose states lie exactly on one positive ray share every later gate, so they die together;
  - for two infinitesimally separated inputs, the log angle drifts at -gap_1 per layer, with both exponents
    conditioned on the chain keeping at least 2 active units;
  - a layer with one active unit synchronises two inputs on that unit exactly.
- **MEASURED**: everything else in sections 2-4. Sample sizes are stated where they appear:
  - 64 nets for the census, 4 for most per-net analyses, 2 for several diagnostics and the L >= 256 depth points;
  - 1 net for the plain sampler comparison, and the best seed per configuration for the trained-net estimates.
  - The synchronisation rate is a consistency check against an exponent proven for infinitesimal separations, not
    a precision test.
- **HEURISTIC**:
  - why K2's error need not grow with depth in 1/sqrt(L) residual nets. The martingale identity
    E|e_L|^2 = L^(-1) sum E|b_l|^2 is exact; bounding the per-layer discrepancies b_l is not done;
  - that the near-quadratic, near-Gaussian structure of those nets is what lets K2 work there.

## 8. Review

One GPT-6 review (Codex, high effort), on the near-final draft: `review/review_gpt6.md`, raw transcript in
`review/review_gpt6.raw`. Its verdict was "major revision". It reproduced 15 headline numbers (table at the end of
the review) and checked the Jacobian chains against autodiff (agreement to 4e-16). It also checked the residual Stein
cross-covariance against a one-step MC and the closure updates, and found no algebraic error. What it found and what
changed:
1. M excluded paths that had once died, though noise revives them (83-99% of the sensitivity on two n = 8 nets). The
   script is fixed and every plain run was redone. At n = 8, the mean k*_F(M) after synchronisation rose to 2.8-3.8 and the avg / att captures fell
   (sections 2.3-2.4).
2. Euler's identity does not identify the top input direction. Rewritten: the surviving direction is the radial row
   of Theorem C.
3. "No gate-free bound exists" overstated what was tested. Narrowed to the constructions tested.
4. gap_1 used inconsistent conditioning. Recomputed with m >= 2 for both exponents: 0.0868 (n = 16), 0.250 (n = 8).
   The n = 8 "excess" rate disappeared.
5. The random-mask PR counted zero products. Fixed: n = 8 PR is 6.9-7.0, not 2.2-3.8.
6. The worst-case gain evidence used an upper bound as if it were a lower bound. Now the largest sampled gain (e^5
   to e^14 above the actual median) is the evidence.
7. "One trajectory's gates" overstated what was shown: avg uses 1,024 trajectories, and att fails on one net.
   Narrowed.
8. The death statements needed qualifiers: "while the state is nonzero", "exactly synchronised", "sampled inputs".
   The noisy census had counted ever-dead paths as dead. Fixed.
9. The angle fit is a consistency check: between-net SE 0.005, acos resolution, selected population. Reworded.
10. The rank-one comparison needs survival conditioning (0.87). Corrected.
11. The paired oracle reuses the truth's paths. Stated.
12. The FLOP model is approximate, several items are not charged, and the noise-scale pilot is excluded for every
    estimator. All stated.
13. The own-budget comparison omits ANT and rounds RQMC down. Stated, and the 16-250x is called conditional.
14. The truth error is not resolved below ~1e-6. Stated.
15. The depth heuristic now has the martingale form, and the L = 1024 weight streams are described.
16. The mechanism diagnostics only show consistency, and the cell census is not a lower bound for every method.
    Reworded.
17. The RB check had claimed 2% agreement, but the noiseless 1e-4 point is 12.6% apart (1.7 SE). Corrected. The
    Imhof validation at 1e-9 now cites the reviewer's independent inversion.
18. Prereg scoring: P3 is now partial, the P2 floor mostly hit, P5 restated, P6's noiseless arm and the truth-budget
    change listed as deviations.
19. Literature: the settings (L^p on R^n, uniform, compact L^p) are separated; Nguyen et al.'s assumptions are
    stated; Kidger-Lyons is a sufficient width; bias-free nets cannot be universal; Hanin 2018 concerns the
    fluctuations of squared Jacobian entries; Hanin-Nica is asymptotic; Du-Hu is a sufficient condition;
    Li-Nica-Roy is a joint limit.
20. Two table values were transcribed wrongly and are corrected. "0.4% of paths die at n = 16" was the prediction,
    not the measurement. "Capacity-limited" and "sorting needs" are softened.

Not changed: the residual K2 result. The reviewer judged it "a promising exploratory mean-estimation result" that
largely reproduces, and agreed that M1/M3 have not been reached.

## 9. Files

- `PREREG.md`, `prereg_hashes.txt`, `GPU-LOG.md`; `notes/refs.md` (verified literature, 77 entries).
- Code:
  - `lib/nd.py`: random nets and FLOP counts.
  - `lib/k2.py`: Gaussian closures (plain; residual with the Stein cross-covariance; RNN).
  - `lib/gauss_relu.py`: copied unchanged from work/arc-noisy.
  - `lib/tnets.py`: trained nets.
  - `scratch/p2_*.py`: collapse, Jacobians, subspaces.
  - `scratch/p3_*.py`: truth, oracle deep part, samplers, tails, cells, variance split, linearity, Gaussianity, depth
    scaling, closure variants.
  - `scratch/p4_*.py`: trained nets.
  - `lib/k2mix.py`: Gaussian-mixture closure with gate splits (section 10).
  - `scratch/p5_*.py`: section 10 (locate, split, rb, tailcheck, rbvar, splitcheck, oracle); queues
    `scratch/q_p5*.sh`.
  - `train/` and `gpu/`: trainers and launch scripts.
- Results: `runs/*.json`, `runs/p3/*.json`, `runs/p4/*.json`, `runs/p5/*.json`. Tables in `results/p2_tables.md`
  and `results/resdepth_table.md`; logs in `logs/`. The pre-review Jacobian runs are kept in `runs/v1/`.
- Review: `review/review_gpt6.md` (findings), `review/review_gpt6.raw` (transcript), `review/prompt_gpt6.md`. Section
  10: `review/review_gpt6_r2.raw`, `review/prompt_gpt6_r2.md`.
- Trained weights: `train/gpu_out/*.npz`, finals and checkpoints.

## 10. Follow-up: why K2 fails on trained narrow nets, and whether the failure is concentrated

The coordinator's question (30 October, ~01:05Z), CPU only: why does K2 fail on the trained nets, and is the failure
concentrated (a few neurons or layers, which would allow a scalable fix) or spread out (a blocker for
Gaussian-plus-corrections families)?
- Nets: the four trained nets of section 4. Residual sorters: width 16, depth 64, sigma = 0 and 0.01. RNNs: width 16,
  T = 64, sigma = 0 and 0.01.
- Pre-registered as `PREREG.md` Part 2 (hash at 01:14:09Z), before any trained-net result of these analyses. The first
  result file was written at 01:14:47Z, and the code had been smoke-tested only on the barely trained pilot models.
  The Part 2 header says "written ~01:20Z". That time was written ahead; the hash time is authoritative.
- Code: `lib/k2mix.py` and `scratch/p5_*.py`. Results are in `runs/p5/` and logs in `logs/p5_*.log`. Runs from
  01:14Z to 02:18Z.
- Added after the pre-registered runs and not pre-registered:
  - `p5_rbvar.py`, which re-measures the tail estimators' errors with less noise (10.3);
  - `p5_splitcheck.py` and `p5_oracle.py`, prompted by the review (10.2).
- One GPT-6 review (10.6) found no bug in the split closure, layer indexing or weights. It narrowed several claims,
  and the text below is the revised version.

**Answer in brief.**
- The failure is spread out in the sorters. In the RNNs the gate error is concentrated, but removing the early errors
  helps little.
  - Sorters: the gate-probability error is spread over most (layer, unit) pairs. Restarting K2 from the exact
    moments at later and later layers lowers the output error gradually, over layers ~8-48.
  - RNNs: the gate error sits in a few units at the first steps. Restarting K2 from the exact moments after step 8
    leaves the noiseless RNN's output error unchanged and lowers the noisy RNN's by 1.6x. In the noiseless RNN, the
    last 8 steps of closure alone regenerate the full error.
- Splitting the Gaussian at the k worst-gate units (k <= 8) gains at most 1.13x over the same closure without splits,
  under any selection rule. The coordinator's criterion was 3x, so the fix is refuted, as pre-registered.
  - The tested closure re-Gaussianises each piece before the next split.
  - An oracle with exactly conditioned pieces fixes that. It still gains at most 1.13x on the RNNs, at any layer,
    over a single Gaussian with the same exact moments. On the sorters it stays at or below 1.6x over K2 when placed
    in layers 2-8.
- K2 also gets the tails wrong by 40-900x on three of the four nets, and the splits do not change that.
- Exact radius integration (positive homogeneity) applies to all four nets. For the task-relevant tail events the gain
  is modest: 2.9-3.8x in variance for large sorting errors, and 1.5-1.8x for the RNNs' confident wrong answers.

### 10.1 Locating the error (step 1, `runs/p5/locate_*.json`, 2^20 sampled paths per net)

For every (layer or step, unit) pair, K2's predicted pre-activation mean, variance and gate-on probability were
compared with sampled truth in two ways:
- accumulated: K2 run from the input;
- one-step: K2 applied to the exact moments of the previous state. Its gate error is then the non-Gaussianity of that
  pre-activation alone.
- The suffix error E_out(l0) is the output nMSE of K2 started from the exact Gaussian moment-match of the state after
  l0 layers or steps, and run to the output. The state moments and the output truth come from the same paths.
- "State error" below is the error of the state's mean, relative to the state's total variance. It does not measure
  covariance or shape errors.

| net | K2 output nMSE | share of summed accumulated gate error, worst 8 / 64 of 1,024 pairs | worst 4 of 16 units (uniform: 25%) | worst layer or step (uniform: 1.6%) | E_out(l0) at l0 = 0 / 8 / 16 / 32 / 48 / 56 |
|---|---|---|---|---|---|
| sorter, sigma = 0 | 0.024 | 4.9% / 29% | 29% | 4.3% | 0.024 / 0.022 / 0.012 / 0.0077 / 0.0017 / 7.6e-5 |
| sorter, sigma = 0.01 | 0.045 | 4.8% / 28% | 30% | 4.3% | 0.045 / 0.034 / 0.031 / 0.0046 / 4.5e-4 / 4.9e-5 |
| RNN, sigma = 0 | 0.039 | 4.9% / 24% | 50% | 3.8% | 0.039 / 0.041 / 0.041 / 0.041 / 0.028 / 0.057 |
| RNN, sigma = 0.01 | 0.22 | 8.3% / 37% | 47% | 8.5% | 0.22 / 0.14 / 0.14 / 0.072 / 0.0057 / 2.6e-4 |

**Sorters: spread (MEASURED).**
- Through layer 4, K2's accumulated state error is about 1e-6 of the state variance. After that it grows:
  - 7e-5 to 4e-4 by layer 8;
  - 0.005 by layer 16;
  - 0.02-0.1 by layer 32.
- The largest one-step errors, 1e-3 to 1.5e-2 at the worst six layers (mostly layers 12-33), stand out against a
  median of 6e-5 to 9e-5.
- Each of the 16 units carries 4-8% of the accumulated gate error (uniform: 6.25%). The worst layer carries 4.3%.
- Starting K2 from the exact state at a later layer lowers the output error step by step:
  - to about half at l0 = 16-24;
  - 3-10x lower at l0 = 32;
  - 14-100x lower at l0 = 48.
  - On a log scale, the largest single drop between grid points (48 to 56) is 43% (sigma = 0) and 26%
    (sigma = 0.01) of the total drop.
  - These are restarts, not an additive split of the error into layers: a restart changes both the starting moments
    and the later approximations.
- The closure gets the output's total variance right: closure/truth is 1.01 at sigma = 0 and 1.26 at sigma = 0.01.
- It gets the mean wrong (nMSE 0.024-0.045) and the tails very wrong. Its P(|output| > t) is too small:
  - 40-160x at the 1e-3 quantile (Imhof). Sampling the closure's own output law (`tailcheck_*.json`, 2^22 samples)
    agrees within its sampling error, with 26 and 100 exceedances.
  - 120-560x at 1e-4 (Imhof). The sampled law has 0 and 2 exceedances there, which confirms only "more than ~50x".
- Interpretation (HEURISTIC): a sorter's state becomes more ordered layer by layer, and no single unit or layer
  carries that structure.

**RNNs: concentrated gate errors, but the output error is not made where they are (MEASURED).**
- The one-step gate error is concentrated in a few units and in the early steps.
  - One unit carries 38-46% of it: unit 8 at sigma = 0, unit 0 at sigma = 0.01.
  - The first 9 of 64 steps carry 36% (sigma = 0) and 57% (sigma = 0.01) of it (uniform: 14%).
- The reason: x_1 is a scalar, so without noise the first state h_1 = ReLU(W_hi x_1) lies on one of two rays, far
  from Gaussian (with noise, close to them). K2's one-step state error is 0.06-0.07 of the state variance at step 2
  and falls to 1e-4 to 1e-3 by step 16.
- Removing these early errors helps little.
  - Noiseless RNN: restarting K2 from the exact moments after step 8, 16 or 32 leaves the output error unchanged
    (0.041 against 0.039).
  - Restarting after step 56 still gives 0.057. So eight steps of closure at the end regenerate the full error,
    even though the one-step state error there is only 3.5e-4 to 3.8e-4 of the state variance per step. Restarting
    after step 60 gives 0.012, and after step 62 0.0016.
  - This resembles the algzoo-noisy lane's finding on ARC's T = 10 RNN, whose errors are decided at the last steps.
  - The noiseless closure also overestimates the output variance 3.7x, so its tails come out 230-900x too large.
  - Noisy RNN: restarting after step 8 lowers the output error 1.6x (0.224 to 0.136), after step 32 3.1x, and after
    step 48 39x. Here the error is made over the whole run, early steps included.
  - Its closure variance is 1.12x the truth's. Its tails are 36% too high at q = 1e-3 and 31% too low at 1e-4.
- So the RNN gate errors are concentrated, but correcting them does not remove most of the output error. We did not
  test why small late-step errors matter so much for the output.

### 10.2 The gate-aligned fix: splitting at the worst-gate units (step 2, `runs/p5/split_*.json`)

**The fix.**
- At each selected (layer, unit) pair, every mixture component is split into its on and off pieces, and the gate is
  fixed in each piece. Pieces are never merged, so k splits give up to 2^k components.
- Each piece is replaced by the Gaussian with the first two moments of its half-space truncation.
  - A later split truncates that Gaussian replacement, not the truncated law, and the other units' ReLU moments are
    those of the replacement.
  - So with several splits at one layer, later pieces can contradict earlier gate signs. In the noisy RNN at S1,
    k = 8 (five splits at step 3), 26 of 32 components, with 15% of the weight, have a mean of the wrong sign on an
    earlier forced gate (review finding 1).
  - With one split per layer the issue does not arise within a layer. That is the case for the sorters' S1 sets,
    whose 8 pairs sit at 8 different layers.
- Selection rules:
  - S1: the worst k pairs by accumulated gate error (primary);
  - S2: the worst k by one-step error;
  - S0: k random pairs (control).
  - S1 and S2 use the 2^20-path truth pass of 10.1 to pick the units. That is an oracle step, not counted in the
    cost.
- k = 1, 2, 4 and 8. Exact ReLU moments for all three rules; the cheaper Stein-linearised moments for S1.
- Scoring:
  - E[output] nMSE against an independent 2^21-path truth;
  - against MC and RQMC at the closure's FLOPs. RQMC gets the largest power of 2 of paths within that budget,
    over R = 16 replicates.
  - The budgets count the propagation only. They leave out gate selection, the readout and mixing, and tail
    integration. At k = 0 the RNN mixture code counts about 8% more FLOPs than `k2.rnn_prop`, through an extra CDF
    vector. These are approximate propagation budgets.
- Checks (`scratch/p5_splitcheck.py`; the review repeated them independently):
  - with no splits the mixture code reproduces K2 to 1e-10 on all four nets;
  - one split at layer 0 of the noiseless sorter matches 2^20-path MC of the next layer within 2 standard errors,
    for the whole mixture and for each piece against MC conditioned on the gate.

| net | K2 nMSE, exact moments (cost in MC paths) | best gain from splitting at k <= 8 over the same closure without splits, by rule S1 / S2 / S0 (Stein S1) | exact-moment closures, k = 0-8: nMSE over RQMC's at equal FLOPs | Stein closures, k = 0 / 1-2 / 4-8: the same ratio |
|---|---|---|---|---|
| sorter, sigma = 0 | 0.024 (670) | 1.02x / 1.00x / 1.06x (1.05x) | 93-86,000x | 3.0x / 7-8x / 37-190x |
| sorter, sigma = 0.01 | 0.045 (523) | 1.05x / 1.05x / 1.04x (1.13x) | 190-33,000x | 4.1x / 5x / 9-110x |
| RNN, sigma = 0 | 0.039 (1,094) | 1.01x / 1.00x / 1.02x (1.00x) | 118-29,000x | 1.8x / 4-8x / 35-660x |
| RNN, sigma = 0.01 | 0.22 (755) | 0.97x / 1.00x / 1.08x (0.96x) | 540x or more | 6.9x / 32-51x / 580-6,000x |

- **Refuted (MEASURED): no k <= 8 gives even 1.2x over the same closure without splits, on any net, under any
  rule.**
  - Random splits do as well as targeted ones (0.92-1.08x).
  - No k <= 8 reaches 10x on any net, as pre-registered, so the k needed for 10x is beyond 8.
  - Against exact K2, the best closure of all is the Stein closure on the noiseless RNN: 1.85x better without
    splits, 1.86x with them. No closure tested reaches 3x over K2.
- At equal FLOPs every closure loses to RQMC.
  - With exact moments the loss is at least 93x. At k = 8 it is 6,000x or more, because the cost grows up to 2^k
    (9,000-250,000 paths at k = 8).
  - At k = 8, RQMC's error approaches the resolution of the 2^21-path truth (about 5e-7), and one corrected RQMC
    estimate is negative. The largest ratios are therefore unresolved, not lower bounds.
  - The Stein-linearised closures are 15-30x cheaper and on all four nets no less accurate: nMSE 0.021, 0.030, 0.021
    and 0.215, against 0.024, 0.045, 0.039 and 0.224.
  - They come closest to RQMC: 1.8-6.9x behind at k = 0, and 4-8x at k = 1-2 (32-51x on the noisy RNN).
- The splits leave the closure's tails as wrong as K2's.
  - Sorters: Imhof's tail probabilities change by under 3% between k = 0 and S1 at k = 8.
  - RNNs: sampled from the closure's law, they change by about 1% (noiseless) and by +11-16% (noisy).
  - Only k = 0 and S1 at k = 8 were sampled. For the noisy RNN the Imhof values at q = 1e-4 failed numerically, so
    its other configurations' tails are not validated.

**Oracle: exactly conditioned splits (`scratch/p5_oracle.py`, `runs/p5/oracle_*.json`; added after the review).**
- The oracle removes the re-Gaussianisation:
  - after l0 layers or steps, the sampled paths are grouped by the joint on/off pattern of the k units of that layer
    with the largest one-step gate error (k = 1, 2, 4, 8);
  - each pattern gets its exact conditional mean and covariance of the state, and K2 runs from each component to the
    output;
  - the components are mixed with the exact pattern probabilities. Patterns with fewer than 64 of the 2^20 paths
    are pooled into one component.
- It is compared, on the same paths, with a single Gaussian with the same exact moments (E_out(l0)), and with K2 from
  the input.

| net | best gain over the single exact Gaussian, max over k <= 8, at l0 = 2 / 4 / 8 / 16 / 32 / 48 / 56 | best gain over K2 from the input at l0 = 2-8 |
|---|---|---|
| sorter, sigma = 0 | 1.62x / 1.46x / 1.17x / 1.04x / 2.18x / 10.5x / 1.60x | 1.62x |
| sorter, sigma = 0.01 | 1.35x / 1.52x / 1.03x / 1.35x / 1.38x / 2.54x / 1.59x | 1.58x |
| RNN, sigma = 0 | 1.00x / 1.00x / 1.00x / 1.00x / 1.01x / 1.00x / 1.06x | 0.99x |
| RNN, sigma = 0.01 | 1.01x / 1.00x / 1.00x / 1.10x / 1.03x / 1.02x / 1.13x | 1.64x (from the exact restart; splits add nothing) |

- RNNs: even exactly conditioned splits at any single layer gain at most 1.13x over a single Gaussian with the same
  moments. The refutation extends to accurately conditioned gate mixtures of k <= 8 units at one layer, with K2
  afterwards.
- Sorters, layers 2-8, the only layers where the exact restart is not itself much better than K2: oracle splits gain
  at most 1.6x over K2, short of 3x.
- Sorters, later layers: exactly conditioned splits help more. At layer 48 of the noiseless sorter they gain 6x with
  one unit and 10.5x with eight over the single exact Gaussian. This holds only on top of exact moments up to that
  layer, which no closure here provides.
- So for the sorters the reviewer's point stands in part. Accurate gate mixtures carry useful structure late in the
  net, but reaching those layers accurately is the unsolved part.

**Why the fix fails (interpretation of the 10.1 numbers and the oracle, HEURISTIC).**
- In the sorters the gate error is spread over hundreds of pairs, so fixing 8 of 1,024 leaves the rest. The useful
  late splits sit behind 40-50 layers of accumulated error.
- In the RNNs the worst-gate units sit at the first steps, and correcting the early steps does not remove most of the
  output error. That holds even with exact moments and exact conditioning.

### 10.3 Tails with the radius integrated exactly (step 3, `runs/p5/rb_*.json` and `rbvar_*.json`; 2^22-path truth)

- All four nets are positively homogeneous in their input when noiseless. The residual MLP has no biases, and the RNN
  has no biases, h_0 = 0 and a linear readout.
- So are three events:
  - |output| > t;
  - for the sorters, max_i |out_i - sort(x)_i| > t, a large sorting error;
  - for the RNNs, the wrong-answer margin max_{j != y} logit_j - logit_y > t, where y is the position of the 2nd
    largest input (scale-invariant). This is a confident wrong answer.
- RB draws a direction u, computes the statistic G(u) on the unit-norm input, and integrates the radius exactly:
  P = E_u[Q_chi_n(t / G(u))], with n = 16 for the sorters and n = 64 for the RNNs (the input is the sequence). The
  thresholds are positive, and directions with G(u) <= 0 contribute 0.
  - Without noise it is unbiased.
  - With sigma = 0.01 the noise does not scale with the input. RB then uses the noise of a typical radius, which is an
    approximation, and it is biased.

The table gives the error at the FLOPs of 16,384 paths, at q = 1e-4 (q = 1e-3 in brackets).
- MC: the exact relative variance of the indicator, (1 - p)/(N p).
- RQMC: the relative MSE measured over R = 16 replicates, so noisy.
- RB: its relative variance, estimated from 2^20 directions. For the noiseless nets this is its MSE. For the noisy
  nets add the squared bias, discussed below.
- The last column compares RB's 2^20-direction mean with the truth (+- one standard error).

| net | event | MC | RQMC | RB (variance) | RB's variance gain over MC | RB mean vs truth |
|---|---|---|---|---|---|---|
| sorter, sigma = 0 | \|output\| | 0.61 (0.063) | 0.45 (0.10) | 1.6e-6 (9.7e-7) | 390,000x (65,000x) | +1% +- 5% (-1% +- 2%) |
| sorter, sigma = 0 | sorting error | 0.56 (0.063) | 0.66 (0.080) | 0.15 (0.018) | 3.8x (3.4x) | -12% +- 7% (-3% +- 2%) |
| sorter, sigma = 0.01 | \|output\| | 0.64 (0.060) | 1.46 (0.033) | 2.1e-6 (1.3e-6) | 310,000x (47,000x) | +3% +- 5% (+0.3% +- 1.5%) |
| sorter, sigma = 0.01 | sorting error | 0.61 (0.060) | 0.59 (0.039) | 0.17 (0.021) | 3.6x (2.9x) | +11% +- 7% (+7% +- 2%) |
| RNN, sigma = 0 | \|logits\| | 0.71 (0.063) | 0.62 (0.053) | 0.12 (0.013) | 5.9x (4.9x) | -1% +- 7% (+0.5% +- 2%) |
| RNN, sigma = 0 | wrong-answer margin | 0.58 (0.059) | 0.59 (0.059) | 0.33 (0.041) | 1.8x (1.5x) | +3% +- 9% (+1% +- 3%) |
| RNN, sigma = 0.01 | \|logits\| | 0.56 (0.058) | 0.38 (0.059) | 0.052 (0.0096) | 11x (6.0x) | +2% +- 6% (-0.1% +- 2%) |
| RNN, sigma = 0.01 | wrong-answer margin | 0.67 (0.061) | 0.96 (0.040) | 0.37 (0.037) | 1.8x (1.7x) | +4% +- 9% (-2% +- 3%) |

- The sorters' |output| gain is trivial. A sorter nearly preserves the norm (|sort(x)| = |x|), so the radius is
  almost the whole event.
- On the task-relevant events the gain is modest: 2.9-3.8x in variance for large sorting errors, and 1.5-1.8x for
  confident wrong answers. These events depend mostly on the input's direction, which RB still samples.
- The pre-registered measurement (R = 16 replicates of each estimator, `rb_*.json`) gave the same verdicts with
  noisier gains. For example, it gave 19x instead of 3.8x for the sorting error at q = 1e-4: 16 replicates miss RB's
  rare large contributions, and MC's measured error there was 2.4x its exact value.
- With noise, RB is biased by a few percent, so its MSE gain is below its variance gain.
  - The noisy sorter's sorting error has the clearest bias: two independent checks at q = 1e-3 gave +3%
    (2^19 directions) and +7% +- 2% (2^20).
  - With the point estimates of the bias added, its gains become 2.3x (q = 1e-3) and 3.4x (q = 1e-4). The noisy
    RNN's gains change by under 2%.
  - For the noisy sorter's |output| the variance gain (310,000x) is far beyond what the bias checks can resolve. A
    bias of 3% would cap the MSE gain near 700x.
- This is a better estimator, not a certified bound (M3).

### 10.4 Pre-registered predictions (Part 2)

| prediction | outcome |
|---|---|
| P11: sorters spread (worst 8 pairs < 20% of the gate error; no layer > 20%; no single gap > 50% of the suffix-error drop) | hit: 4.8-4.9%; 4.3%; 43% and 26% |
| P12: RNNs, worst 4 units > 50% of the gate error, worst 8 pairs < 20% | split. Accumulated error: worst 4 units 50.3% at sigma = 0 (borderline hit) and 46.8% at sigma = 0.01 (miss); worst 8 pairs 4.9% and 8.3% (hit). The one-step error, a separate reading, gives 65-70% and 10-13% |
| P13: refuted unless some k <= 8 gives 3x over K2. Predicted: refuted, and 10x needs k > 8 | hit for the tested closure and rules: refuted on all four nets (best 1.13x over the same closure without splits). The oracle (not pre-registered) extends this to exactly conditioned single-layer splits on the RNNs and at the sorters' layers 2-8 |
| P14: every split closure loses to RQMC by >= 30x at equal FLOPs | partial. Hit for exact moments (>= 108x for the splits; K2 itself 93x; the largest ratios unresolved). Miss for the Stein closures at k = 1-2 on three nets (4-8x) and at k = 4 on one sorter (9x) |
| P15: the closure's tail is wrong by > 2x at q = 1e-4 for every k on every net | partial. Hit on the sorters and the noiseless RNN (120-900x). Miss on the noisy RNN (24-31% low at k = 0 and S1 k = 8, the two configurations sampled; Imhof fails there at 1e-4). The registered test, relative squared error > 0.25, is not the same as "2x" for overestimates; no outcome changes |
| control: random splits <= 1.5x over K2 | hit (0.92-1.08x) |
| P16: RB >= 3x over MC at q = 1e-4 on the noiseless nets; on the noisy ones within 10% of the truth at q = 1e-3 and >= 3x | partial. Hit for \|output\| and the sorting error on the noiseless nets. On the noisy nets: the bias point estimates are within 10% (+3% in the pre-registered 2^19 check, +7% +- 2% later, whose 95% interval reaches 12%); the sorting error passes 3x at q = 1e-4 (3.4-3.6x) but not at 1e-3 (2.3-2.9x). Miss for the RNN margin on both RNNs (1.8x; 2.0x and 2.5x in the R = 16 measurement) |

### 10.5 What this means for M1 and M3

- On these trained nets, K2's failure is not a few-unit defect that a gate-aligned correction patches.
  - Sorters: the gate error is spread over the sorting layers. Exactly conditioned splits help only late in the net,
    and only on top of exact moments up to there.
  - RNNs: the visible gate errors are early, and correcting them leaves most of the output error. This holds even
    with exact moments and exact conditioning. The output error comes from the last steps (noiseless) or from the
    whole run (noisy).
- This is evidence against Gaussian-plus-few-splits closures on these nets: sequential splits, and exactly
  conditioned splits at one layer. It is not a proof against all closures. What remains are descriptions built on
  the algorithm's own variables, or on the steps that decide the output. The algzoo-noisy lane's algorithm-indexed
  descriptions are one example, though they are fitted rather than derived.
- For tails, positive homogeneity is cheap and exact without noise. But it helps the task-relevant events only
  1.5-4x, because they depend mostly on the input's direction.

### 10.6 Review

One GPT-6 review (`review/review_gpt6_r2.raw`, prompt `review/prompt_gpt6_r2.md`). Verdict: "revise before
finalizing". It found no layer-index, weight or normalisation bug. It reproduced k = 0 = K2 on all four nets and the
layer-0 split check. Its findings and what changed:
1. Repeated splits re-Gaussianise, so the result refutes this implementation, not accurately conditioned gate
   mixtures. We disclosed this in 10.2 and added the exactly conditioned oracle.
   - The oracle extends the refutation for the RNNs and for the sorters' early layers.
   - It shows that late sorter layers carry useful gate structure.
2. The restart (suffix) results are not an additive split, and "early errors do not matter" was too strong. We
   reworded: the restarts show how much removing early errors helps (1.0x noiseless RNN, 1.6x noisy RNN). We
   labelled state errors as mean errors.
3. For the noisy nets RB's column was variance, not MSE. We relabelled it, added the bias-inclusive gains, and noted
   that the noisy |output| gain is unresolved.
4. The noisy RNN's Imhof values at q = 1e-4 are numerical failures. Its sampled tails are trusted; the review's
   independent rescaled integral gives 5.9e-5, against 6.2e-5 sampled. We noted the configurations that are not
   validated.
5. The sorter tail checks rest on 26, 100, 0 and 2 exceedances. We reported the counts and dropped the precision
   claims.
6. The largest RQMC ratios are unresolved, not lower bounds. We reworded.
7. The budgets count propagation only, and gate selection uses a sampled oracle. We stated both.
8. The 1.13x is the gain over the same closure without splits; the Stein closure alone is 1.85x better than exact K2
   on the noiseless RNN. The radial law is chi_64 for the RNNs, and the code was right. Both are fixed in the text.
9. Scoring: P13 is qualified by findings 1 and 8, P15's registered test is asymmetric, and P16's noisy bias bound is
   a point estimate. All three are noted in 10.4. The Part 2 header time ("~01:20Z") is inconsistent with the hash
   (01:14:09Z), and the hash is authoritative.
