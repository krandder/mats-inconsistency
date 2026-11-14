# Stationary-regime lane: assumptions and open questions (13:05Z, before any data)

Each item: the assumption in the coordinator's heuristic model (or mine), why it may fail, and the check.

1. **Noise convention.** "Small Gaussian noise sigma on every layer" with "the radius walks down to the noise floor"
   only makes sense for absolute noise (Model S: h_l = relu(W_l h_{l-1}) + sigma xi_l). The sibling lanes use
   relative noise (Model R: s_l = sigma r_l, r_l the pilot RMS of the layer's activation).
   Check: state both; derive what each implies; measure both.
2. **Scale invariance (Model S).** relu is positively homogeneous and there are no biases, so h^(sigma) = sigma h^(1)
   pathwise once started at the scaled point. Then the stationary sample measure for sigma is the image of the one
   for sigma = 1 under h -> sigma h, and sigma is not a parameter of the stationary regime at all.
   Check: proof (one line); numerically, identical laws of h/sigma at sigma = 1e-3, 1e-2, 1e-1.
3. **Size of the noise relative to the state (Model S).** Back-of-envelope with the Lyapunov spectrum
   lambda_k ~ -(5/4 + (k-1))/n: E|h|^2 per coordinate is about (sigma^2/2) sum_k 1/(k + 1/4) ~ (sigma^2/2)(ln n + c), so
   one layer's noise is about sqrt(2/(ln n + c)) of the state: 0.6-0.9 at n = 8-256. If so, the stationary state is
   noise-dominated at every sigma, the cloud is not small, and the "few bits" regime does not exist there.
   Check: measure |h|^2/(n sigma^2), tau/sigma and the gate-flip fraction under one layer's noise, n = 8..256.
4. **"The attractor."** In Model S the noiseless dynamics decays to 0 (lambda_1 < 0), so there is no nonzero
   noiseless attractor to linearise around. With the noise fixed as well, pairs of trajectories still contract
   (the noise cancels in differences), so a random point attractor of the noisy system may exist.
   Check: prove pairwise mean-square contraction; measure coupling-from-the-past convergence of two starts under
   common noise and common weights.
5. **Lyapunov spectrum.** Half-width product heuristic: lambda_1 = E log|D W v|/|v| = -5/(4n) + O(n^-2) (chi-square and
   active-count fluctuations), gaps about 1/n. Along stationary noisy trajectories (Model S) the gates are
   noise-driven; is the spectrum the same?
   Check: QR (Benettin) along stationary trajectories, n = 8..256.
6. **Cloud covariance = Lyapunov equation.** Sigma = sigma^2 sum_j J_j J_j^T along the trajectory's own gates is an
   identity of the frozen-gate linear model, not of the net. Check: compare the measured noise covariance at fixed
   weights with (i) the same-trajectory Jacobian sum, (ii) the spectrum sigma^2/(2|lambda_k|).
7. **Radius law.** Model S: |h|^2 behaves like a Kesten perpetuity X = A X + B with E A = 1 (He criticality), so the
   stationary tail should be P(|h|^2 > x) ~ C/x (index kappa = 1, E|h|^2 = infinity) and E|h_l|^2 grows linearly
   from any start. Check: tail-index estimates over layers and nets; linear growth of the mean.
8. **Memory time of the radius.** 1/(2|lambda_1|) = 2n/5 layers for squared quantities. Check: decay of the
   coupling distance; autocorrelation of log|h_l| along the stationary chain.
9. **Births.** "n sqrt(2/pi) sigma/tau per layer". Needs a precise definition: the number of units whose gate
   differs between the trajectory and the same trajectory with the last layer's noise resampled. Check against the
   formula with tau and the noise's pre-activation scale measured.
10. **Age profile.** Linear picture: variance from noise of age j is sigma^2 ||J_j||_F^2 ~ sigma^2 n/(2j) for j < n, so
    every octave of ages carries the same share (scale-free), and the effective number of directions alive at age
    j is about n/(2j). Check: resample the noise of one layer at age j and measure the change of h_L, projected
    on the covariance eigenbasis.
11. **Live-bit budget, CLT vs enumeration.** If item 3 holds, Model S stationary is at the CLT end at every width.
    Is it then "easy"? Check: a Gaussian closure and a linear-response control variate against MC on it.
12. **Model R.** sigma is a genuine small parameter for the direction; the radius relative to the ensemble scale
    keeps the input's memory for a long time (heuristic: log-gain diffusion of order sigma^2 n / 2 per layer,
    restoring rate of order sigma^2). Check: measure the radius spread across noise at fixed input vs depth.
13. **Input bits (Model S with input).** Memory time T_mem ~ (4n/5) ln(1/(sigma sqrt n)) for the input radius to
    reach the stationary scale. Below it the noise is relatively small and the coordinator's small-cloud picture
    should apply. Check: measure relative noise vs depth from an input; ARC's 256 x 32 sits far below T_mem.
