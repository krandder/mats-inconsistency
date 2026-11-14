# Can the k* important directions be mapped predictably through the inner layers? (kstar-frames, 30-31 October 2026)

Lane: kstar-frames (target M1). Work directory `work/kstar-frames/`, bead pool-8yk7c9, 23:20Z on 30 October to 03:23Z
on 31 October.
Pre-registration `PREREG.md`, hashed at 23:45Z before any run on a ReLU net; Amendment 1 (00:40Z, implementation
details) and Amendment 2 (02:48Z, after the review) declare what had been seen. Compute: General (CPU, nice 19) for
width 32; widths 64 and 128 on one on-demand L4 at a time (coordinator-approved, cap $3): four VMs in sequence (a fifth
launch hit a stockout and created nothing), all deleted and checked empty in the instance list, $2.09 in total at
list-price ceilings (`GPU-LOG.md`). Review: GPT-6 (gpt-6-astra), verdict "major revisions"; every finding was addressed
(section 10); a focused GPT-6 re-check returned "minor revisions", applied. Labels: PROVEN, MEASURED, HEURISTIC.

Kelvin (via main): "In case the proof attempt doesn't work, we can try to find a way to predictably map the 'K*
important dimensions' over the inner deep layers?"

## Summary for Kelvin

Along one input, yes: the k* directions move from layer to layer by the inverse transpose of the layer map (exact
when the downstream Jacobian has rank k*, approximate on the nets). But the weights still to come decide them, so
nothing computed from the forward pass so far predicts them better than a random frame plus one direction (proven for
the noiseless net). Across inputs, partly: a small learned map from
the state recovers 93-99% of a held-out input's top-k* sensitivity, but only 75-85% of its directions. None of the
methods we tested reaches the precision of an input's own frame, and none cleared the pre-registered gate for building
an estimator in the moving frame, so none was built. Two inputs' frames agree only while their states stay within about
1.4/L' radians, and in our tests with finite stores of frames the atlas needed grows steeply with width.

1. **One input, layer to layer (PROVEN; MEASURED).** The k*-frame at layer c (top-k* right singular vectors of the
   downstream Jacobian) is the image of the frame at layer c-1 under the inverse transpose of the layer map
   (Proposition 1; exact when the downstream Jacobian has rank k* and the active count does not grow). On the nets this
   "dual" step reproduces the frame with overlap 0.99 when the number of active units does not grow from one layer to
   the next. When it grows (about half the steps) part of the frame is invisible from the previous layer, and the
   overlap is 0.81-0.89 (less loss at larger width). The errors compound: after 8 steps the carried frame is close to
   chance (0.05 above it on average). Pushing a frame forward with the layer Jacobian itself (QR, the covariant-Lyapunov
   recipe) gives 0.44-0.80 after one step and chance after eight. So each input's frames come from its own backward
   pass.
2. **Why: the frame belongs to the future (PROVEN for sigma = 0; MEASURED for sigma = 1e-3).** In the noiseless net,
   given a spectral gap at the cutoff, any frame computed from the input and the weights up to layer c overlaps the
   true k*-frame, in expectation, at most as much as a random frame plus one direction, the state's own
   (Proposition 2). On the noisy nets the frame that forward QR converges to (the past's expanding directions) is at
   chance to within 0.01 on average.
3. **A new input's frame (MEASURED, held-out inputs, layer L/2).** A fixed frame (the leading directions of the averaged
   sensitivity, restricted to the input's active units) captures 0.87-0.90 of an input's top-k* sensitivity (overlap
   0.65-0.74); products with marginal gate probabilities do a little worse. A small MLP from the state direction (41
   thousand to 2.6 million parameters, polynomial in width) captures 0.93-0.99 (overlap 0.75-0.85), 0.07-0.13 more than
   the fixed frame on the same nets. On average it leaves 1.4-7% of an input's sensitivity outside the frame, against
   9-16% for the fixed frame and 0.01% for the input's own frame. It also beats the nearest of 8,192-16,384 stored
   inputs, by 0.04-0.16 (more at larger width). Its error falls slowly with training data, roughly as N^-0.1 to
   N^-0.25, and four times more epochs changed nothing.
4. **Coherence (MEASURED).** Two inputs' frames keep 90% captured sensitivity while their states are within about
   1.4/L' radians (0.05-0.06 rad with 24 layers to go, 0.09-0.11 with 16, 0.12-0.18 with 8), at all three widths. For
   90% overlap of the directions the radius is 4-5 times smaller, about 0.3/L'. At the 90%-sensitivity radius 2-7% of
   the gates differ per layer, about n/2 gate flips in total over the remaining depth.
5. **Atlas size (MEASURED for finite stores; HEURISTIC beyond).** At 90% captured sensitivity, layer L/2, 16 layers to
   go, the share of stored inputs whose frame fits a new input falls about tenfold per doubling of width (0.09, 0.013,
   0.0012 on average; 0.04, 0.004, 0.0002 for the median input). At width 128, 13-92% of new inputs (by net) have no
   fitting frame among 8,192 stored ones; with 8 layers to go, 66-99%. An uncalibrated heuristic from the states'
   small-ball masses gives about 10^1.5 / 10^4 / 10^6 frames at widths 32 / 64 / 128 (10^3.3 / 10^4 / 10^9.5 with 8
   layers to go). That is roughly exp((0.5-1) k*), but a high power of width fits as well, and between nets the
   estimate moves by up to five decades.
6. **Q3 not run.** The gate needed a method with captured sensitivity >= 0.9 and overlap >= 0.8 at widths 64 and 128
   and both depths (only width 64 at depth 16 passed) and an atlas of at most 10^4 frames at width 128 (not resolved
   directly; heuristic 10^5.7 to 10^10).
7. **For the premise.** These results are about the directions, not about the joint law on them. They support part of
   the picture: the k* directions are fixed by the downstream gates and are input-specific at a resolution whose
   atlas, in our finite-store tests, grows steeply with width. They cut against another part: the directions are not a
   hash of the input, since a polynomial-size learned map recovers most of their sensitivity weight. Neither result
   bounds the description length of the joint law on those directions, which is the proof lane's object, and neither
   is a lower bound on estimators.

## 1. Setting and scores

- Nets: critical He-ReLU, W_l iid N(0, 2/n), no bias, x ~ N(0, I_n), activation noise s_l = 1e-3 x RMS(a_l) after
  every ReLU (so on both sides of any cut). Widths 32, 64, 128; three nets each (seeds 400-402); depth 32, and depth 16
  as the first 16 layers of the same weights. Main layers c = 8, 16, 24 (L = 32) and 4, 8, 12 (L = 16); the gate layer
  is c = L/2.
- Window: at c = L/2 the median angle between independent inputs' states is 0.31-0.78 rad at widths 64 and 128 (P14).
  At width 32, depth 32, net 402 has largely merged by layer 16 (0.09 rad); it is kept in the means.
- The frame of input x at layer c: F_c(x), the top-k* right singular vectors of J_c = d a_L / d z_c along x's own noisy
  path; k* = N_surv(J_c; 0.01) averaged over 512 paths, fixed per net and layer. Measured k* matches Fuss-Catalan at
  half width plus about 1.5 at every layer with L' >= 8 (P1): 15 / 21-22 / 34-36 at L' = 24 / 16 / 8 for n = 128.
- Scores of a candidate frame G: overlap O = tr(P_G P_F)/k* (mean cos^2 of the principal angles; chance = k*/m_c,
  m_c active units); captured sensitivity CS = |J_c G|_F^2 / (energy of J_c's top k* directions), 1 for the input's own
  frame; tail T = share of |J_c|_F^2 outside G (1e-4 for the input's own frame). CS weighs the strong directions; O
  counts all k* equally, so the two can differ.
- Candidates (PREREG): (a) transport along the path from a neighbouring layer's frame; (b) fixed frames (marginal-gate
  products; the top-k* eigenvectors of the input-averaged normalised sensitivity, which maximise mean CS among unmasked
  fixed frames); (c) the frame of the nearest of R stored inputs (angle between states); (d) a learned map from the
  state to a frame. Fixed and stored frames are restricted to the new input's active units at layer c (its forward pass
  gives them) and re-orthonormalised; after that restriction the fixed frame is no longer optimal. Under Kelvin's
  constraint (scale to 10M+ parameters) only transports and polynomial-size learned maps count as methods; (c) and
  atlas counts are diagnostics.

## 2. One input, layer to layer (Q1a)

**Proposition 1 (dual covariance). PROVEN.** Let B_c = D_c W_c D_{c-1} be the layer map between active coordinates,
so J_{c-1} = J_c B_c. (a) If J_c has rank exactly k, the row space of J_{c-1} is B_c^T range(F_c), so the previous
layer's frame is orth(B_c^T F_c) when B_c^T keeps rank k on it. (b) The dual step (B_c^T)^+ maps that back onto the
projection of F_c onto range(B_c): exact iff F_c has no component in the null space of B_c^T within the active
coordinates, a subspace of dimension max(0, m_c - m_{c-1}) for generic weights. (c) The push-forward
orth(B_c F_{c-1}) = orth(B_c B_c^T F_c) returns range(F_c) only if range(F_c) is invariant under B_c B_c^T, which fails
for generic weights. Proof: row(J_c B_c) = B_c^T row(J_c), and (B^T)^+ B^T is the projector onto range(B). In Lyapunov
terms, right singular frames of the future (forward Lyapunov vectors) have covariant orthogonal complements, so they
are covariant under the inverse transpose. The measured k* is a thresholded effective rank, so on the nets the relation
holds only approximately, as the measured one-step overlaps below 1 show. (`notes/props.md`)

**Proposition 2 (a forward pass cannot see the frame). PROVEN for sigma = 0.** Assume a_c != 0, m_c >= 3, k fixed (or
a function of the past), and condition on the invariant event G that J_c has a spectral gap at the cutoff
(s_k > s_{k+1}, so its top-k frame F_c is unique). Let e_c = a_c/|a_c| and let S be any d-dimensional subspace of the
active coordinates at layer c that is a function of the input and W_1, ..., W_c only. Then
E[tr(P_{F_c} P_S) | past, G] <= |P_S e_c|^2 + k (d - |P_S e_c|^2)/(m_c - 1); for d = k the expected overlap is at most
k/(m_c - 1) + 1/k. Proof sketch: given the past, M = W_{c+1} B (B an orthonormal basis of the active coordinates
orthogonal to e_c) is Gaussian and independent of g = W_{c+1} e_c, and the future trajectory and gates depend on
W_{c+1} only through g; M O has the law of M for every rotation O, so on G the law of F_c is invariant under rotations
that fix e_c, and Schur's lemma gives E[P_F | past, G] = diag(alpha, beta I) with beta <= k/(m_c - 1). Checked by fixing
6 pasts at n = 32 and 64 and resampling 400 futures each: every conditional mean is 0.7-0.8 below the bound
(`logs/t2_prop2_cond.log`); the past frame there already contains the state direction, and the radial weight alpha is
0.30-0.43, so the radial term is real. It does not cover rules that use the future weights (marginal gates, stored
inputs, learned maps, backward passes).

Measured (run A, 512 paths per net; net means of O; every main layer in `logs/tables.md`):

| n | L | c (L') | k* | chance | push-forward 1 step | push-forward 8 steps | past frame | dual 1 step | dual 8 steps | backward 1 step | repeat with new noise |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 32 | 16 (16) | 5.3 | 0.37 | 0.61 | 0.37 | 0.35 | 0.91 | 0.42 | 0.99 | 0.98 |
| 64 | 32 | 16 (16) | 11.0 | 0.34 | 0.62 | 0.37 | 0.35 | 0.94 | 0.40 | 0.99 | 0.97 |
| 128 | 32 | 16 (16) | 21.0 | 0.35 | 0.61 | 0.38 | 0.35 | 0.96 | 0.43 | 0.99 | 0.98 |
| 128 | 32 | 24 (8) | 34.3 | 0.55 | 0.67 | 0.56 | 0.55 | 0.91 | 0.58 | 0.99 | 0.99 |
| 128 | 16 | 8 (8) | 33.3 | 0.52 | 0.66 | - | 0.52 | 0.91 | - | 0.99 | 0.99 |

Over every layer with L' >= 8 (net means): push-forward after 8 steps is 0.02 above chance on average (at most 0.06),
the past frame 0.00 (at most 0.05), dual transport after 8 steps 0.05 (at most 0.10). Dual one step, split by whether
the active count grows (pooled over layers and nets):

| n | O when m_c <= m_{c-1} | O when m_c > m_{c-1} | share of steps with growth |
|---|---|---|---|
| 32 | 0.99 | 0.81-0.83 | 0.44-0.49 |
| 64 | 0.99 | 0.85-0.86 | 0.45-0.46 |
| 128 | 0.99 | 0.88-0.89 | 0.48-0.49 |

So a frame follows almost exactly from the next layer's frame (backward step 0.99, Proposition 1a). Going forward,
each step that activates new units loses the part of the frame in the null space of B_c^T (generic combinations, not
the new units' own axes), and later steps do not recover it; that the inverse transpose also amplifies these errors is
plausible but was not measured separately. The pre-registered prediction that dual transport would hold over 8 steps
(P5, from a Gaussian surrogate with square blocks) failed. In this lane the frames were computed from each path's full
downstream Jacobians (O(L' n^3) per path).

## 3. A new input's frame (Q1b-d)

Held-out inputs (2048 per net), stored and training inputs from 16,384 others (8192 at n = 128), layer c = L/2; CS / O,
net means; T = share of sensitivity outside the frame, mean over inputs (`runs/Bres_*.json`, `runs/D_*.json`):

| n | L | k* | chance O | marginal gates | fixed frame | learned map (2 nets) | nearest of 256 stored | nearest of all stored | repeat with new noise | T: own frame / fixed / learned |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 32 | 6 | 0.37 | 0.82 / 0.71 | 0.89 / 0.69 | 0.97 / 0.77 | 0.91 / 0.80 | 0.95 / 0.87 | 1.00 / 0.98 | 1e-4 / 0.11 / 0.03 |
| 64 | 32 | 11 | 0.34 | 0.80 / 0.66 | 0.87 / 0.65 | 0.96 / 0.77 | 0.83 / 0.70 | 0.89 / 0.76 | 1.00 / 0.97 | 8e-5 / 0.13 / 0.04 |
| 128 | 32 | 22 | 0.35 | 0.83 / 0.68 | 0.87 / 0.68 | 0.95 / 0.77 | 0.82 / 0.68 | 0.86 / 0.72 | 1.00 / 0.98 | 9e-5 / 0.13 / 0.05 |
| 32 | 16 | 8 | 0.52 | 0.81 / 0.72 | 0.87 / 0.70 | 0.97 / 0.80 | 0.85 / 0.77 | 0.91 / 0.84 | 1.00 / 0.99 | 6e-5 / 0.13 / 0.03 |
| 64 | 16 | 17 | 0.55 | 0.87 / 0.74 | 0.90 / 0.74 | 0.98 / 0.83 | 0.84 / 0.75 | 0.88 / 0.79 | 1.00 / 0.99 | 5e-5 / 0.10 / 0.02 |
| 128 | 16 | 34 | 0.52 | 0.84 / 0.71 | 0.87 / 0.72 | 0.95 / 0.78 | 0.78 / 0.68 | 0.81 / 0.71 | 1.00 / 0.99 | 7e-5 / 0.13 / 0.05 |

- The learned map: an MLP (two hidden layers of width 4n, GELU) from u = z_c/|z_c| to an n x k* matrix, added to the
  fixed frame, restricted to the active units and orthonormalised; trained to maximise CS with early stopping on 10% of
  its training inputs, scored on the held-out inputs (a separate random stream, not used for stopping). Paired with the
  fixed frame on the same two nets, the gain is +0.07 to +0.13 CS and +0.08 to +0.12 overlap at every width and depth.
- Learning curve on net 400 (CS / O at N = 1024 / 4096 / max): width 32, L = 32: 0.939 / 0.956 / 0.966 (O 0.75 / 0.77 /
  0.78); width 64: 0.939 / 0.951 / 0.958; width 128: 0.910 / 0.925 / 0.931 (O 0.72 / 0.74 / 0.75). These are not
  end-to-end sample-size curves: the fixed frame the map starts from was computed from all stored inputs. A post-hoc run
  with four times the epoch cap and patience 30 (width 64, L = 32) stopped at epoch 437 with CS 0.959 / O 0.774
  (against 0.958 / 0.772).
- At the other layers of L = 32 (net 400 only): 24 layers before the end it is harder (learned CS 0.79-0.88 against
  0.59-0.73 for the fixed frame); 8 layers before the end easier (0.986-0.993 against 0.92-0.95).
- "Repeat with new noise" is the frame of the same state under a fresh downstream noise draw: a repeatability check,
  not a ceiling on prediction.
- The map was trained on CS only, while the gate also asked for overlap; its overlap shortfall sits in the weak
  directions near the 1% cutoff. Other objectives and architectures were not tried.

## 4. Coherence (Q2)

Pairs x' = x cos t + u sin t (u orthogonal to x, |u| = |x|), 256 per t, t from 1e-3 to pi/2, sharing their noise
draws; at each main layer the angle between the states, the overlap and captured sensitivity of the partner's frame,
and the Frobenius coherence of the two sensitivity matrices (`runs/C_*.json`). theta_coh = state angle at which the
median CS of the partner's frame (restricted to the input's active units) falls to 0.9; median over nets (a net whose
CS never falls below 0.9 in the observed range is right-censored, counted as infinite in the median):

| L' | width 32 | width 64 | width 128 | raw overlap 0.9 instead (widths 32 / 64 / 128) | gate disagreement per layer at theta_coh | random-mask prediction (before data) |
|---|---|---|---|---|---|---|
| 24 (L = 32, c = 8) | 0.053 | 0.052 | 0.059 | 0.013 / 0.013 / 0.015 | 0.016-0.023 | - |
| 16 (L = 32, c = 16) | 0.109 | 0.086 | 0.090 | 0.021 / 0.020 / 0.019 | 0.025-0.040 | 0.08 |
| 12 (L = 16, c = 4) | 0.111 | 0.116 | 0.104 | 0.026 / 0.032 / 0.021 | 0.035-0.039 | - |
| 8 (L = 32, c = 24; L = 16, c = 8) | 0.165; 0.120 | 0.152; 0.177 | 0.177; 0.151 | 0.031-0.041 | 0.047-0.073 | 0.15 |

- The radius varies by less than a factor 2 across the three widths (P11b) and scales like 1/L' for L' >= 8: about
  1.4/L' for 90% sensitivity (projected partner frame), 0.3/L' for 90% overlap (raw partner frame). At the sensitivity
  radius the total number of gate flips over the remaining depth is about n/2.
- The measured disagreement equals theta/pi to within 40% (Gaussian pre-activations at angle theta disagree in sign
  with probability theta/pi).
- The random-mask surrogate (independently flipped masks, raw mean CS) differs from this measurement (coupled real
  paths, projected median CS); its agreement is heuristic support, not a matched test.
- The annealed two-replica formula (lyapunov lane), fed with each pair's gate overlaps, predicts the net-mean Frobenius
  coherence within a factor 1.5 in 142 of 144 cells with coherence >= 0.1; the two failures are nearly independent
  inputs with 24 layers to go at width 32.

## 5. Atlas size, against exp(k*) (Q2)

For each held-out input, the share of stored inputs whose frame (restricted to its active units) captures at least
1 - delta of its top-k* sensitivity ("ball mass"); random-centre count R99 in the exact form 1 - mean (1 - p)^R = 0.99;
the union of all stored frames. Float64 recompute (`runs/Bcover2_*.json`; section 10); delta = 0.1 (CS >= 0.9), layer
L/2; all tolerances in `logs/tables.md`:

| n | L | k* | ball mass, mean over inputs (median over nets) | ball mass of the median input (median over nets) | inputs with no stored frame within delta (range over nets) | all stored frames cover (range) | R99, random centres |
|---|---|---|---|---|---|---|---|
| 32 | 32 | 6 | 0.09 | 0.043 | 0-0.5% | 99.5-100% | 6,100 / 7,400 / 14 |
| 64 | 32 | 11 | 0.013 | 0.0043 | 4-8% | 92-96% | > 16,384 |
| 128 | 32 | 22 | 0.0012 | 0.00024 | 13-92% | 8-87% | > 8,192 |
| 32 | 16 | 8 | 0.039 | 0.016 | 0-3% | 97-100% | > 16,384 / > 16,384 / 900 |
| 64 | 16 | 17 | 0.013 | 0.0051 | 2-21% | 80-98% | > 16,384 |
| 128 | 16 | 34 | 5e-6 | 0 | 66-99% | 1-34% | > 8,192 |

These describe random finite stores, not optimised atlases; an empirical zero is unresolved, not a population zero.
At delta = 0.2 the random-centre counts range from a few to 7,100 at widths 32 and 64 and are partly unresolved at 128.

States (run E, 65,536 states per net and layer): the share of state pairs within angle theta, C_c(theta), has local
slope d log C / d log theta at the coherence radius of about 4 (width 32; 1.2 on its merged net), 8 (width 64) and 13
(width 128) at c = 16, and 5-6.5, 10-11 and >= 17 at c = 8: the states' local dimension at the coherence scale grows in
proportion to width. HEURISTIC atlas count N(delta) ~ 1 / C_c(theta_coh(delta)), the inverse mean pair mass of the
states at the median coherence radius, extrapolated below the resolved range with the local slope there. It is not
calibrated as a covering number: at width 64 a typical frame ball is about 100 times larger than the state ball at
theta_coh (frames repeat across distant states), while 4-8% of inputs match no stored frame. At delta = 0.1:

| L' | width 32 | width 64 | width 128 | range at width 128 over binnings / per-net radii (by net) |
|---|---|---|---|---|
| 16 (L = 32) | 10^1.5-10^1.6 | 10^3.7-10^4.1 | 10^5.7-10^6.05 | 10^5.75-10^6.05 / 10^5.70 (10^3.5, 10^5.2, 10^8.3) |
| 8 (L = 16) | 10^3.1-10^3.4 | 10^3.9-10^4.2 | 10^9.2-10^10.1 | 10^9.5-10^10.1 / 10^9.2 (10^7.2, 10^10.2, 10^10.2) |

Against exp(k*): ln N / k* = 0.58 / 0.82 / 0.63 (L' = 16; k* = 6, 11, 22) and 0.95 / 0.54 / 0.64 (L' = 8; k* = 8, 17,
34), so N ~ exp((0.5-1) k*) describes the heuristic counts. But from width 64 to 128 at L' = 16 the count grows by about
2 decades where exp(k*) grows by 4.8, and a power of width near n^7 fits the L' = 16 counts as well; the mean ball
masses fall by factors of 7-11 per doubling at L' = 16 and 3, then 2,600, at L' = 8. Three widths, three nets and this
much variation between nets do not decide between exponential and polynomial growth.

## 6. Q3: the estimator in the moving frame

Pre-registered gate: (PREDICTABLE) a method-eligible predictor with held-out CS >= 0.9 and O >= 0.8 at c = L/2 for
both depths at widths 64 and 128; (COHERENT) an atlas of at most 10^4 frames at width 128, measured or estimated.
Outcome: the learned map passes only at width 64, depth 16 (0.98 / 0.83); elsewhere its overlap is 0.77-0.78. At width
128 no direct count resolves (the stored frames leave many held-out inputs uncovered), and the heuristic estimate is
10^5.7-10^6.05 (depth 32) and 10^9.2-10^10.1 (depth 16). Not run.

## 7. What this means

- For Kelvin's question: there is a predictable map along one input (the dual step, approximately, or each input's
  backward pass, exactly) and a compact approximate map across inputs (the learned MLP, 93-99% of the sensitivity
  weight). None of the shared maps we tested reaches an input's own frame: frames are fixed by the downstream gates,
  change on a scale of about 1.4/L' radians of state angle, and in our finite-store tests an atlas of them grows
  steeply with width.
- For an estimator (M1): a method that carries the distribution in the moving k*-frame needs each input's frame. The
  options seen here are an atlas (growing steeply with width), a per-input backward pass (affordable per sample, but
  the estimator must still integrate over inputs; this lane did not test such schemes), or a learned frame that leaves
  on average 1.4-7% of each input's sensitivity outside (against 0.01%). How that residual would enter an estimator's
  error was not measured.
- For the proof lane: Proposition 2 is a rigorous form of "the back's relevant directions are uniformly random with
  respect to everything the front computes, apart from the state's direction", consistent with the rotation invariance
  in its Lemma B. The coherence law (about 1.4/L' radians at 90% sensitivity, width-free) and the atlas growth are
  empirical inputs. The learned-map result says a smooth function of the state captures most of the frames' weight, so
  the directions alone look like an unlikely source of an exp(k*) lower bound; the result does not exclude one.

## 8. Pre-registered scorecard (`logs/scorecard.log`; net means as written, per-net counts as a supplement)

| prediction | outcome |
|---|---|
| P1 k* = Fuss-Catalan + 1.5 (n >= 64, L' >= 8) | held at every layer |
| P2 push-forward 1 step: O in [0.45, 0.80], CS in [0.45, 0.85] | held (0 of 90 cells; per net 2 of 270 outside) |
| P3 push-forward 8 steps <= chance + 0.15 | held (0 of 48) |
| P4 past frame <= chance + 0.12 | held (0 of 90) |
| P5 dual 1 step >= 0.85 / 0.97; 8 steps >= 0.70 / 0.93 | failed (68 of 90; 48 of 48): newly active units (Proposition 1b) |
| P6 backward 1 step >= 0.85 / 0.97 | held |
| P7 repeat with new noise: CS >= 0.95, O >= 0.80 | held |
| P8 fixed frames: O <= 0.65 (marginal gates), <= 0.70 (averaged-sensitivity frame), CS 0.08 below the noise repeat | failed in all four configurations, narrowly (O 0.66-0.74) |
| P9 learned map no better than the fixed frame + 0.05 CS, O <= 0.75 | failed: better than predicted (CS +0.08 to +0.09, two learned nets against three-net fixed means; paired on the same nets +0.07 to +0.10; O 0.77-0.83) |
| P10 nearest stored input: CS < 0.9 with all stored, slow power law | held |
| P11 theta_coh in range; width-free within x2; two-replica within x1.5 | held; held; failed in 2 of 144 cells |
| P12 state small-ball slope >= 8 at width 128 and >= 1.5 x width 32 | held (13.8 and 6.2 with the secant estimator; 15.0 and 6.7 with the local gradient) |
| P13 atlas at delta = 0.1: (a) unresolved at widths 64, 128; (b) heuristic >= 10^6 at 128; (c) its log grows >= 1.5x from 64 to 128 | (a) held; (b) held as scored (10^6.05, 10^9.5), but at depth 32 not robustly (10^5.70-10^5.98 under other binnings or per-net radii); (c) held (x1.55, x2.4), which with width 32 included does not show exponential growth |
| P14 window: median state angle >= 0.1 rad at L/2 (n >= 64) | held |

## 9. Status of every statement

| statement | status |
|---|---|
| Proposition 1 (dual covariance under rank k; exactness condition; push-forward not covariant) | PROVEN; on the nets approximate (effective rank) |
| Proposition 2 (forward-only frames at most chance plus one direction, given a spectral gap) | PROVEN for sigma = 0; checked conditionally; MEASURED at sigma = 1e-3 (past frame at chance) |
| Dual transport: 0.99 without growth, 0.81-0.89 with growth, near chance after 8 steps | MEASURED (three widths, three nets); error amplification not measured |
| Fixed, learned and stored-input frames (section 3) | MEASURED on held-out inputs (learned map on 2 nets per configuration, 1 at c = 8, 24) |
| theta_coh ~ 1.4/L' (sensitivity) and ~ 0.3/L' (overlap), width-free; disagreement ~ theta/pi | MEASURED (three widths, L' = 8-24, three nets) |
| Two-replica coherence within x1.5 | HEURISTIC formula; agreement MEASURED |
| Atlas coverage and ball masses (section 5, first table) | MEASURED for these finite random stores (not bounds on optimised atlases) |
| Heuristic atlas counts and their exp((0.5-1) k*) or n^7 readings | HEURISTIC, uncalibrated |
| Relation to the premise (directions, not the joint law; no estimator lower bound) | interpretation |

## 10. Review

GPT-6 (gpt-6-astra, high effort, read-only, 02:25-02:35Z; `review/review_gpt6.md`) reproduced 19 reported numbers,
passed both test suites, and returned "major revisions". What it found and what changed:
- Covering scores formed the Gram matrix in float32 and returned CS above 1 for ill-conditioned pairs (34 saved
  entries above 1.001; one case 2.155 against an exact 0.267). All covering statistics were recomputed in float64
  (`scratch/r2c_cover.py`, `lib/cover.py` v2; the reviewer's case is now a test). No score now exceeds 1, and no
  reported statistic moved by more than 0.002 (shares) or 0.063% (ball masses). Directions of a projected reference
  frame whose squared norm is below 1e-10 of the largest are dropped (a numerical-rank convention).
- "No predictable, shared map" overstated the evidence: replaced by "none of the tested methods cleared the gate";
  Proposition 2 covers only past-measurable rules; the learned map is evidence of substantial predictability; a
  per-input backward pass does not by itself force Monte Carlo.
- The heuristic atlas count is uncalibrated (state-space mean pair mass at a median radius): now labelled so, with its
  sensitivity to the unregistered binning and to per-net radii (table in section 5). P13(b) at depth 32 therefore does
  not hold robustly.
- Proposition 1 needed its rank conditions stated and the effective-rank caveat; Proposition 2 needed conditioning on a
  spectral-gap event and a fixed k, and "no better than chance" became "chance plus one direction". The old numerical
  check averaged over fixed nets; a conditional check (fixed pasts, resampled futures) was added.
- The scorecard tested each net, against the written rule (net means): rescored. P2 now holds (0 of 90); P5 fails 68 of
  90 and 48 of 48; P11c fails 2 of 144.
- "Median ball mass" was a median over nets of input means: renamed, with the median input's mass added.
- The learning curves reuse supervision through the fixed-frame start (computed from all stored inputs); comparisons
  mixed 2-net and 3-net means: noted, and paired gains reported.
- The averaged-sensitivity frame is optimal only before projection; the new-noise frame is a repeatability check, not a
  ceiling; dual transport after 8 steps is near chance, not at chance; greedy covers stop when the selection half is
  covered, so the union of all stored frames is the finite-store statistic; "a few dozen frames cover 99% at every
  width" was unsupported and is removed; R99 is now the exact fixed-R form; the COHERENT code now includes the
  direct-count alternative (outcome unchanged; the code requires both depths to pass the same way and every net to
  pass directly, stricter than the net-mean rule, which does not change the result).
- The coherence radius at 90% sensitivity is a loose tolerance: the 90%-overlap radius is now reported beside it.
- Not done, now declared (Amendment 2): the kernel-smoothed-reference diagnostic; learned maps at c = 8, 24 on net 401.
- Edge cases noted, not changed: orth() has no rank detection (all checked projections were full rank); n_surv counts
  an exactly zero Jacobian as full.
- Focused re-check (GPT-6, 03:18-03:22Z; `review/review_gpt6_r2.md`): verdict "minor revisions". It confirmed the
  float64 regression and 16,384 more pairs (agreement within 3e-8), the restated Proposition 2 and its conditional
  check, the net-mean rescoring, the ball-mass columns and all hashes. Applied: "exact rule" qualified in the summary;
  atlas growth limited to the finite-store tests; the radial weight alpha (0.30-0.43) corrected in `notes/props.md`;
  censored radii, raw overlap and the factor-2 width test stated; the P9 gain given paired; the numerical-rank
  convention and the COHERENT code's stricter logic disclosed; 0.063% for the largest ball-mass change.

## Files

`work/kstar-frames/`: `PREREG.md` and `prereg_hashes.txt`; `notes/questions.md`, `notes/theory.md`, `notes/props.md`;
`lib/kf.py` (frames, transports, scores), `lib/cover.py` (pairwise scores v2, covers), `lib/nd.py` (the net, unchanged
copy from the narrow-deep lane); `scratch/r1_layers.py` (run A), `r2_big.py`, `r2b_analyze.py` and `r2c_cover.py`
(run B; r2c is the float64 covering), `r3_pairs.py` (run C), `r4_mlp.py` (run D), `r5_states.py` (run E),
`r9_score.py` (scorecard), `r10_tables.py` (tables), `t0*.py`, `t1_prop2.py`, `t2_prop2_cond.py` (theory checks);
`tests/` (both suites exit 0); `gpu/` (L4 scripts); `logs/` (`logs/gpu_kstar*/` for the GPU runs; `logs/tables.md`,
`logs/scorecard.log`); `runs/` (`runs/posthoc/` for the longer training run); `GPU-LOG.md`; `review/`.
