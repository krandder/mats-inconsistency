# Literature and design choices (leaf r, 2026-10-31 ~01:20Z)

Sources opened (abstract or summary page) unless marked "memory". Prior lanes: cut-rank (outbox/cut-rank-lower-bound-
2026-10-30.md and work/cut-rank/notes/literature.md), arc-deep-networks (ARC's App. F), narrow-deep, estimability.

## What is known
- **Dual activations.** For unit-variance Gaussian pre-activations with correlation t, E[phi(X) phi(Y)] =
  sum_k c_k^2 t^k, c_k the normalised Hermite coefficients (Daniely-Frostig-Singer 2016, memory; Mehler's formula).
  Murray, Jin, Bowman, Montufar (ICLR 2023, arXiv 2211.07844) write the deep NTK as a power series whose coefficients
  are built from the activation's Hermite coefficients, and bound the NTK spectrum when those decay fast enough.
- **Kernel smoothness and eigenvalues on the sphere.** Bietti-Bach (ICLR 2021, arXiv 2009.14397): eigenvalue decay is
  read from the kernel's differentiability at t = +-1; deep ReLU kernels decay like shallow ones. Holzmuller-
  Scholpple (AISTATS 2026, arXiv 2506.22429): for activations non-smooth only at 0 (ReLU, LeakyReLU, ELU, SELU) the
  RKHS depends only on the degree of non-smoothness and not on depth; polynomial activations give depth-dependent
  RKHSs. No result for C-infinity non-polynomial activations is stated there.
- **Hermite coefficients and analyticity.** Hille (Duke Math. J., 1940; reviewed by Boyd, Math. Comp. 1980 and
  J. Comput. Phys. 1984): Hermite-function coefficients of a function analytic in a strip of half-width tau (with a
  decay condition in the strip) are O(n^{-1/4} exp(-tau sqrt(2n+1))); entire functions with Gaussian decay give
  geometric convergence. So tanh, sigmoid, softplus and SiLU (poles at finite imaginary distance) sit in the
  exp(-c sqrt k) class, Gaussian-type entire functions (erf, GELU, Gaussian-smoothed ReLU) in the geometric class.
- **Integration of smooth integrands.** Scrambled nets reach RMSE O(N^{-3/2} (log N)^{(d-1)/2}) for integrands with
  square-integrable mixed derivatives (Owen 1997, memory). Smoothness therefore also helps RQMC, which raises the bar a
  deductive estimator must clear. ReLU nets are piecewise linear (kinks), outside that class.
- **ARC.** Wu et al. 2026, App. F: GELU and tanh at n = 256, L = 2-12 are "qualitatively similar" to ReLU for cumulant
  propagation. Polynomial activations give O(n/eps^2) (their Thms 5.1-5.2).
- **Large models (Kelvin's 10M+ bar).** Shazeer 2020 (GLU variants): SwiGLU/GEGLU beat ReLU at matched parameters
  (log-perplexity 1.944 vs 1.997, via lit-tangledness lane). Mirzadeh et al. (ICLR 2024, arXiv 2310.04564, "ReLU
  strikes back"): ReLU, GELU and SiLU give similar LLM accuracy; ReLU's exact zeros give activation sparsity and faster
  inference. So the cost of a smooth gate at scale is mainly lost exact sparsity, not accuracy. Bilinear MLPs and
  chi-nets are near SwiGLU (lit-tangledness lane).

## Choices
1. **Deductive estimator.** Options: (a) Gaussian closure K2 (mean + covariance, exact bivariate Gaussian moments by
   Mehler series or quadrature); (b) harmonic/polynomial truncation; (c) ARC's K = 3-4 cumulant propagation.
   Chosen: (a) for running-time comparisons, since it is the program's and ARC's canonical sample-free estimator and
   costs O(n^3 L); (b) only as the theoretical summary protocol and in the exact ray-front computation, because its
   term count C(n + D, n) is too large to run at n >= 16; (c) not run (no general-activation implementation here, and
   ARC already report it activation-insensitive).
2. **Smoothness ladder.** Options: softplus_beta (explicit strip pi/beta, beta -> infinity is ReLU); Gaussian-smoothed
   ReLU (= ReLU with integrated pre-activation noise; entire); a C-infinity smooth step with a margin (exact zeros
   below -delta, identity above +delta: keeps ReLU's exact sparsity while being C-infinity). Chosen: all three, with
   softplus as the main ladder.
3. **Multiplicative gating.** SwiGLU (silu(W1 h) * (W2 h)), GEGLU, ReGLU and the bilinear GLU (no nonlinearity), with
   the hidden width cut to match parameters (Shazeer's 2/3 rule).
4. **Not chosen.** Learned/rational activations and polynomial activations of fixed degree as main arms: polynomial
   nets are exactly finite-rank (ARC Thms 5.1-5.2 cover them), so the bilinear GLU stands in for that end.
