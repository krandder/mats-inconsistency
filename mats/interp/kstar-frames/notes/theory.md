# kstar-frames: theory before data, and the method choices (leaf pool-8yk7c9.2, 2026-10-30 ~23:55Z)

All numbers below come from Gaussian or random-mask surrogates (no ReLU net of this lane has been run). Logs:
`logs/t0_surrogate.log`, `logs/t0b_dual.log`, `logs/t0c_flip.log`, `logs/t0d_kstar.log`.

## 1. Transport along one path (Q1a)
Setting (Theorem C of the lyapunov lane, PROVEN for sigma = 0 on the no-dead-layer event): the angular part of a path's
downstream Jacobian is H_L ... H_{c+1}, independent Gaussian factors of half width m ~ n/2, entries of variance 1/m,
each independent of the past.
- Push-forward is the wrong covariance. Let F_c be the top-k right singular frame of J_c. Then
  F_{c-1} ~ orth(B_c^T F_c) (B_c the layer map), so a right frame is carried forward by B_c^{-T}, not B_c. In
  Lyapunov terms, right singular frames of the future (forward Lyapunov vectors) have covariant orthogonal
  complements; their own transport is the inverse transpose. Push-forward with QR converges instead to the top left
  singular frame of the past product (backward Lyapunov vectors, the span of the leading covariant vectors), and by
  Theorem C that frame is independent of F_c in the angular part: chance overlap k/m. HEURISTIC for the full J (one
  radial direction is exactly shared by both).
- Surrogate numbers (m = 16-64, L' = 8-24, 48-64 paths): one push-forward step from the true F_{c-1}: overlap
  O = 0.57-0.69, CS = 0.58-0.70; after 8 steps O = chance + 0.02-0.04; past frame (BLV): chance. Inverse-transpose
  ("dual") transport: O = 0.985-0.995 after one step, 0.97-0.98 after 8 steps, 0.91-0.92 after 16; CS >= 0.99.
  One backward step from F_{c+1}: O = 0.985-0.995.
- Consequence: per path, the k* frame CAN be carried predictably from layer to layer, by B_c^{-T}. But it must be
  initialised by the path's own frame at some layer, which needs that path's downstream pass; one backward pass gives
  all layers' frames anyway. So transport answers "is the frame sequence of one path simple" (yes, by this theory),
  not "can a new input's frame be predicted cheaply" (Q1b-d, Q2).

## 2. Coherence across inputs (Q1c, Q2)
- Random-mask surrogate (W shared, masks iid Bernoulli(1/2), path 2's masks = path 1's with a fraction f flipped at
  every layer from the cut on; f = theta/pi for two Gaussian pre-activation vectors at angle theta, the annealed
  assumption): at L' = 16, CS of the partner's frame = 0.995 / 0.986 / 0.95-0.97 / 0.88-0.89 / 0.66-0.69 / 0.41-0.44
  at f = 0.001 / 0.003 / 0.01 / 0.03 / 0.1 / 0.25; O = 0.97-0.98 / 0.94-0.95 / 0.86-0.87 / 0.74-0.75 / 0.54-0.56 /
  0.36-0.38 (chance 0.34-0.38). Identical at n = 64 and 128 within 0.02. At L' = 8 the decay is about half as fast.
  So CS = 0.9 at f ~ 0.025 (theta ~ 0.08 rad) at L' = 16 and f ~ 0.05 (theta ~ 0.15 rad) at L' = 8, width-free.
- Two-replica (annealed Wick recursion, lyapunov lane): predicts the Frobenius coherence
  rho = tr(A_x A_x') / sqrt(tr A_x^2 tr A_x'^2), A = J^T J, from the per-layer gate overlaps of the pair. Surrogate:
  rho ~ exp(-(L'+1) f) roughly.
- Size of any atlas. In the infinite-width (kernel) picture nearby inputs keep their angle (critical ReLU: theta' =
  theta - theta^2/(3 pi) + ...), so pairs with state angle below theta_coh ~ 0.08 must have input angle below
  ~0.08: the state distribution looks (n-1)-dimensional at that scale and its small-ball mass is ~ theta^(n-1). In
  the finite-width picture the past Jacobian (Fuss-Catalan at half width) folds most input directions away, and the
  local dimension at scale theta is the number of past singular directions stretched above theta, ~ (n/c) log(1/theta).
  Both predict a local dimension at the coherence scale that grows with n, hence covering counts exponential in n at
  fixed depth (exp(Theta(k*)) at fixed (c, L'), since k* ~ n G / (2 L')). Which picture holds is an empirical question.
- Predicted k* (Fuss-Catalan at half width; the lyapunov lane measured about +1.5 above it): at L' = 8 / 16 / 24:
  8.5 / 5.1 / 3.7 (n = 32), 17.0 / 10.3 / 7.3 (n = 64), 34.1 / 20.6 / 14.7 (n = 128).

## 3. Method choices
- Learned map (Q1d). Options: (i) MLP from the state direction to an n x k matrix, masked by the state's own gates
  D_c, orthonormalised by QR, trained to maximise CS; (ii) a kernel smoother of training sensitivities in state angle;
  (iii) a linear map from the state to a frame generator (n^2 k parameters). Chosen: (i), initialised at the best fixed
  frame (top-k of M from training paths) so it can only add to it, with parameters ~ n H + H^2 + H n k (H = 4n),
  polynomial in n: it is the method-eligible form Kelvin allows. (ii) is an atlas (stores every training frame); it
  is reported with the references of Q1c as a diagnostic. (iii) is dominated by (i) at these sizes.
- Covering counts (Q2). Options: (i) greedy cover of held-out paths by training frames (resolves counts up to the
  training size); (ii) random-centre estimate from per-path ball masses p(x) (solve E exp(-R p) = 0.01); (iii) beyond
  the sample, the state correlation integral at the measured coherence radius, with its local slope. Chosen: (i) and
  (ii) as MEASURED where they resolve; (iii) as HEURISTIC only, since frame-space ball masses can be smaller than
  state-space ones (distant states can share frames). The comparison of (ii) and (iii) where both resolve is reported.
