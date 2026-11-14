# kstar-frames: can the k* important directions be mapped predictably through the inner layers? (pre-registration)

Written 2026-10-30 ~23:50Z, before any run on a ReLU net of this lane. The sha256 and time in `prereg_hashes.txt` are
authoritative. Amendments are appended below with dates and their own hashes.

Ask (Kelvin, via main): "In case the proof attempt doesn't work, we can try to find a way to predictably map the 'K*
important dimensions' over the inner deep layers?" Target M1. Constructive counterpart of the cut-rank proof lane.
Kelvin: "I am only interested in things that could scale to large (10M+ parameter) models": a method must be a
transport rule or a learned map whose cost is polynomial in width; atlases and covering counts are diagnostics only.

## Seen before writing this (declared)
- Reports: `outbox/lyapunov-dimension-2026-10-29.md` (Theorems C, D; the two-replica formula; 185 of 195 directions
  for one fixed block at 256 x 64), `outbox/narrow-deep-2026-10-29.md` (sections 2.1-2.4), `work/cut-rank/FORMAL.md`.
- Theory computed on surrogates only (no ReLU net of this lane run): `notes/theory.md`, logs `logs/t0*.log`
  (Gaussian half-width products; random-mask products; Fuss-Catalan counts).

## Objects
- Nets: `nd.Net('plain', seed, n, 32, sigma=1e-3)` from `work/narrow-deep/lib/nd.py` (copied unchanged to
  `lib/nd.py`): W_l iid N(0, 2/n), no bias, x ~ N(0, I_n), h_l = ReLU(W_l h_{l-1}) + s_l eps_l for l < L, output
  a_L; s_l = 1e-3 r_l (pilot RMS, 2^14 paths). Widths n = 32, 64, 128; seeds 400, 401, 402 (3 nets per width).
  Depth L = 32, and L = 16 as the first 16 layers of the same net (same W_1..16, r_1..16).
- Main layers: c = 8, 16, 24 (L = 32) and c = 4, 8, 12 (L = 16). Gate layer: c = L/2.
- True frame F_c(x): top-k right singular vectors of J_c(x) = d a_L / d z_c along x's own noisy path.
  k = k*(c) = round(mean over the run's paths of N_surv(J_c; 0.01)), fixed per net and layer for all predictors.
  m_c = number of active units at layer c; chance overlap = mean k / m_c.
- Scores of a frame G on path x: O = tr(P_G P_F)/k; CS = |J_c G|_F^2 / sum_{i<=k} s_i^2; T = 1 - |J_c G|_F^2 / |J_c|_F^2.
  Means over held-out paths, then over the 3 nets (between-net spread reported).
- "D_c-projected" G means orth(D_c G) with the test path's own gates at layer c (known from its forward pass to c).

## Runs
- A (all layers, 512 paths per net and depth): F_c and singular values at every c = 1..L-1, and
  a1 push-forward one step orth(D_c W_c F_{c-1}); a2 push-forward t = 2, 4, 8 steps from F_{c-t} (QR each step);
  dual one step orth(pinv(D_c W_c D_{c-1})^T F_{c-1}) and dual t = 2, 4, 8 steps; bwd one step
  orth(D_c W_{c+1}^T D_{c+1} F_{c+1}); blv = top-k left singular frame of the past product D_c d z_c / d x;
  noise ceiling at main layers: frame of the same z_c with the downstream noise (eps_c onward) redrawn.
  Median angle between independent inputs' z_c per layer (window check).
- B (main layers, references and held-out): N_A = 16384 training/reference paths (8192 at n = 128), N_B = 2048
  held-out paths per net and depth; frames, singular values (top k + 16), states z_c, gates of layer c.
  - Q1b fixed frames: marginal-gate frame (top-k right frame of diag(p_L) W_L ... W_{c+1} diag(p_c), p_l the marginal
    activation probabilities from 4096 separate paths); optimal fixed frame = top-k eigenvectors of
    Mt = mean over A of J^T J / sum_{i<=k} s_i^2 (maximises mean CS among fixed frames); both raw and D_c-projected.
  - Q1c nearest reference: the frame of the nearest of the first R paths of A by angle between z_c's, D_c-projected,
    R = 1, 4, 16, ..., N_A. Kernel-smoothed references reported as a diagnostic.
  - Q1d learned map: MLP u = z_c/|z_c| -> n x k matrix A(u); G = orth(D_c (G_fix + A(u))), G_fix the optimal fixed frame,
    output layer initialised at 0; 2 hidden layers of width 4n, GELU; loss -mean CS; Adam, batch 256, early stopping
    on 10% of the training set; N_train = 1024, 4096, 16384 (first paths of A); scored on B. Run at L = 32, c = 8, 16,
    24 and L = 16, c = 8, all widths, nets 400 and 401 (N_train = 16384), learning curve on net 400.
  - Covering at c = L/2, loss l(x, y) = 1 - CS_x(orth(D_c^x F_y)), delta = 0.3, 0.2, 0.1, 0.05: (i) greedy centres from A
    chosen on half of B and scored on the other half: centres for 50 / 90 / 99% coverage or "not reached with all of A";
    (ii) per held-out path the ball mass p(x) = share of A within delta; random-centre count R99 solving
    mean_x exp(-R p(x)) = 0.01, reported only if fewer than 1% of held-out paths have p(x) = 0.
- C (pairs, Q2): x' = x cos t + u sin t (u orthogonal to x, |u| = |x|), t = 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, pi/2;
  256 pairs per t per net and depth, same noise draws for both paths (coupled). At main layers: state angle theta_c,
  O(F, F'), CS of the partner's frame (both ways, averaged), rho = tr(A A') / sqrt(tr A^2 tr A'^2) with A = J^T J, and
  the per-layer gate overlaps for the two-replica recursion (as in the lyapunov lane's a2_theory.py).
- E (states): 2^16 states z_c at main layers per net; correlation integral C_c(theta) = share of pairs with angle
  <= theta; local slope d log C / d log theta. HEURISTIC extrapolation N_cover(delta) ~ 1 / C_c(theta_coh(delta)), with
  theta_coh(delta) the state angle at which the median CS of coupled pairs (run C) falls to 1 - delta.

## Predictions (numbers are path means, then net means)
- P1 (k*). At n >= 64 and L' >= 8: |k*_measured - (FC + 1.5)| <= max(3, 0.25 FC), FC = Fuss-Catalan at half width
  (`logs/t0d_kstar.log`: n = 64: 17.0 / 10.3 at L' = 8 / 16; n = 128: 34.1 / 20.6).
- P2 (push-forward, 1 step). At every layer with L' >= 8: O in [0.45, 0.80] and CS in [0.45, 0.85].
- P3 (push-forward, 8 steps). O <= chance + 0.15 at every layer c >= 9 with L' >= 8.
- P4 (past frame, blv). O <= chance + 0.12 at every layer with L' >= 8.
- P5 (dual transport). 1 step: O >= 0.85 and CS >= 0.97; 8 steps: O >= 0.70 and CS >= 0.93 (layers with L' >= 8).
- P6 (backward 1 step, consistency). O >= 0.85 and CS >= 0.97.
- P7 (noise ceiling). CS >= 0.95 and O >= 0.80 at c = L/2; CS >= 0.97 at L' = 8.
- P8 (fixed frames, held-out, D_c-projected, c = L/2, n = 64 and 128, both depths). Marginal-gate frame: O <= 0.65;
  optimal fixed frame: O <= 0.70 and CS at least 0.08 below the noise ceiling's CS.
- P9 (learned map, same configurations, N_train = 16384). CS <= CS(optimal fixed frame, D_c-projected) + 0.05 and
  O <= 0.75; the CS gain from N_train = 4096 to 16384 is <= 0.03.
- P10 (nearest reference, c = L/2, n >= 64). CS < 0.9 at R = N_A; a power-law fit of mean (1 - CS) against R over
  R = 64..N_A has exponent of magnitude < 0.25.
- P11 (coherence, run C). (a) theta_coh(CS = 0.9) in [0.03, 0.25] rad at L' = 16 and in [0.05, 0.40] at L' = 8
  (surrogate: 0.08, 0.15); (b) at fixed L' it varies by less than a factor 2 across n = 32, 64, 128;
  (c) measured rho within a factor 1.5 of the two-replica recursion on per-t pair means where mean rho >= 0.1.
- P12 (state small-ball, L = 32, c = 16). The local slope of log C_c at the smallest theta with >= 100 pairs is >= 8 at
  n = 128, and the slope at n = 128 is >= 1.5 x the slope at n = 32.
- P13 (covering, delta = 0.1, c = L/2). (a) n = 64 and 128, both depths: greedy 99% coverage not reached with all of A,
  and >= 1% of held-out paths have no delta-neighbour in A; (b) HEURISTIC extrapolation: log10 N_cover >= 6 at
  n = 128; (c) the extrapolated log N_cover grows by a factor >= 1.5 from n = 64 to n = 128 at fixed (c, L')
  (exponential-like, as exp(Theta(k*)); a polynomial of degree <= 4 would add at most 2.8 to the natural log).
- P14 (window). Median angle between independent inputs' states at c = L/2 is >= 0.1 rad for n >= 64.

## Decision rule for Q3 (fixed now)
- PREDICTABLE: for every n in {64, 128} and L in {16, 32}, at c = L/2, some method-eligible across-input predictor
  (fixed frames of Q1b, raw or D_c-projected; learned map of Q1d) has held-out net-mean CS >= 0.9 and O >= 0.8.
  (Transport rules are per-path: they need the path's own frame at another layer, so they are reported but do not
  count for this gate.)
- COHERENT: at n = 128, c = L/2, both depths, the covering count at delta = 0.1 is resolved at <= 1e4, or the
  HEURISTIC extrapolation gives <= 1e4.
- Q3 (an estimator in the moving k*-frame against MC and RQMC at equal FLOPs) runs only if both hold. Otherwise the
  coherence and covering numbers are reported as evidence on the premise, labelled, and sent to the proof lane via
  main. Neither outcome is a proof about the joint law on the k* directions (notes/questions.md, item 3).

## Process
CPU only, `nice -n 19 ionice -c3`, at most 3 heavy processes. Code in `lib/`, `scratch/`; logs in `logs/`; results in
`runs/`. A GPT-6 (gpt-6-astra) review before any claim. Report `outbox/kstar-frames-2026-10-30.md` with PROVEN,
MEASURED, HEURISTIC labels.

## Amendment 1 (2026-10-31 ~00:50Z; implementation details, declared with what had been seen)
Seen when writing this: width-32 results of runs A, B (fixed frames, references, covering) and C; run E at widths 64
and 128. Nothing at widths 64 or 128 from runs A-D.
1. k*(c) is computed once per net and depth from run A's 512 paths and used by runs B, C, D (decided before any
   data, when writing run B; the PREREG wording "the run's paths" meant run A).
2. At n = 128, N_A = 8192, so the largest learned-map training set there is 8192 (not 16384).
3. Widths 64 and 128 run on one L4 (coordinator-approved; General was at load ~30). On CUDA, right singular frames,
   singular values, past-product left frames and the pseudo-inverse of the dual transport come from eigh of J^T J
   (resp. A A^T, B^T B) because batched cusolver SVD of the masked Jacobians was ~170x slower; tests in
   `tests/test_kf.py` (section 8) show agreement with LAPACK SVD to 1e-9 on CPU. Random streams differ between CPU
   and CUDA (same distributions). Width 32 runs on General with LAPACK SVD.
4. Covering loss: the reference frame projected on the test path's active set is scored with the projector onto
   range(D_x F_y) (pseudo-inverse of the Gram matrix), which equals orth(D_x F_y) when D_x F_y has full rank k.
5. Scorecard: theta_coh is the median over nets (not a geometric mean), because a partly merged width-32 net never
   drops below CS 0.9 (theta_coh = inf). Decided after seeing the width-32 run C data.

## Amendment 2 (2026-10-31 ~03:00Z, after the GPT-6 review; everything had been seen)
1. Scorecard aggregation corrected to the written rule (path means, then net means, per (width, depth, layer) cell)
   for P2-P7 and P11c; per-net counts are reported as a supplement. The first scorecard tested each net separately.
2. Covering statistics recomputed with float64 pair scores (`scratch/r2c_cover.py`, `lib/cover.py` v2): the float32
   Gram matrix of v1 gave scores above 1 for ill-conditioned pairs. Random-centre R99 is also reported in the exact
   fixed-R form 1 - mean (1 - p)^R. P13(a) is scored on the union of all stored frames (net mean below 0.99) and the
   share without a stored frame within delta (net mean >= 1%).
3. P12 is scored with a secant slope over two grid intervals at the smallest angle with >= 100 pairs (the stored local
   gradient gives 15.0 and 6.7 instead of 13.8 and 6.2; both pass).
4. Not done: the kernel-smoothed-reference diagnostic of run B, and learned maps at c = 8, 24 on net 401 (net 400 only).
5. Post hoc, labelled as such: one learned-map run with a 600-epoch cap and patience 30 (width 64, L = 32, c = 16);
   a conditional check of Proposition 2 that fixes the past and resamples futures (`scratch/t2_prop2_cond.py`).
