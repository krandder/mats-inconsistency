# kstar-frames: assumptions, definitions, open questions (leaf pool-8yk7c9.1, 2026-10-30 ~23:45Z)

Lane: kstar-frames (M1). Kelvin (via main): "In case the proof attempt doesn't work, we can try to find a way to
predictably map the 'K* important dimensions' over the inner deep layers?" Constructive counterpart to the cut-rank
proof lane (`work/cut-rank/FORMAL.md`).

## Definitions fixed here
- Nets: plain critical He-ReLU (`work/narrow-deep/lib/nd.py` form): z_l = W_l h_{l-1}, a_l = ReLU(z_l),
  h_l = a_l + s_l eps_l (l < L), output a_L; W_l iid N(0, 2/n), no bias; x ~ N(0, I_n); s_l = sigma r_l, r_l the RMS
  of a_l from a seeded pilot; sigma = 1e-3 on every layer, so on both sides of any cut. Widths 32, 64, 128; depth 32,
  with depth 16 = the first 16 layers of the same weights (a separate net, frames recomputed for L = 16).
- True frame of path x at layer c: F_c(x) = top-k right singular vectors of J_c(x) = d a_L / d z_c
  = D_L W_L ... D_{c+1} W_{c+1} D_c along x's own noisy path (its own noise draw). It lies in range(D_c).
- k = k*(c): per net and layer, round(mean over paths of N_surv(J_c; 0.01)), N_surv = #{i : s_i >= 0.01 s_1}.
- Scores of a predicted orthonormal frame G (n x k) on path x:
  overlap O = tr(P_G P_F)/k (mean cos^2 of the principal angles);
  captured sensitivity CS = |J_c G|_F^2 / sum_{i<=k} s_i^2 (1 = as good as the true frame);
  tail T = 1 - |J_c G|_F^2 / |J_c|_F^2 (share of the path's sensitivity outside G; the true frame's T is the floor).
  Path means are primary; energy-weighted tails (ratio of sums) secondary.
- Method-eligible predictors (Kelvin: must scale to 10M+ parameters; no atlases): a transport rule or a learned map
  whose cost is polynomial in width. Atlases (frames of reference inputs) and covering counts are diagnostics only.

## Assumptions and open questions, each with its check
1. "Predictable" must mean predictable without the path's own downstream pass. One backward pass per path already
   gives F_c at every layer for O(L' n^2 k) FLOPs, polynomial in n, and is exactly sampling. So the content of the ask
   is whether frames are shared or learnable across inputs. Check: none needed (definition); stated in the report.
2. Transport direction. Right singular frames of the future product are covariant under the inverse transpose of the
   layer map, not the map itself (their orthogonal complements, the slowly growing directions, are covariant under
   the map). Push-forward by the layer Jacobian (QR) converges to the past's expanding frame (backward Lyapunov
   vectors), which by Theorem C is independent of the future frame in the angular part. Check: Gaussian surrogate
   (`scratch/t0_surrogate.py`, `t0b_dual.py`, logs/) before data; then measured on the nets (Q1a).
3. Does frame coherence bear on the premise? The premise is about the joint law on the k* directions; coherence and
   covering counts measure how fast the directions themselves change with the input, i.e. the size of any atlas of
   frames. Exponential covering counts support "the directions are input-specific at fine resolution"; they do not by
   themselves bound the description length of the joint law, and small counts would not prove a compact law. Check:
   stated as such in the report; no claim beyond it.
4. Downstream noise makes the frame random given the state. Any predictor from the state is capped by the frame of
   the same state under a fresh downstream noise draw. Check: measured noise ceiling (Q1b, "same state, new noise").
5. The interesting layers are those with k* well below n/2 (L' >= 8). Near the output k* -> n/2 and the frame is
   nearly the active set of layer c, which the state's own gates give. Check: k* and chance level k/m reported per
   layer; main analysis layers c = L/4, L/2, 3L/4 fixed in the PREREG.
6. Window. The ask is about the window before inputs merge. Check: median pair angle per layer reported for every
   (n, L); width 32 at depth 32 may be partly merged (Lyapunov e-fold ~ n layers).
7. Which state distance governs coherence: angle between z_c(x) and z_c(x'), or input angle. Check: both measured
   (Q2 pairs at fixed input angle, binned by state angle).
8. Covering counts beyond the sample size cannot be measured. Check: direct greedy/held-out cover where it resolves;
   otherwise a lower-bound-style estimate from small-ball masses, labelled HEURISTIC, with the extrapolation stated.
9. A good per-path frame may still not give an estimator: the law in the moving frame must be carried and pushed
   (the premise's object). Check: Q3 runs only if the pre-registered predictability and coherence gates pass.
