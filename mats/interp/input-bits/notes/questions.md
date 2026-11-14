# Input-bits lane: assumptions and open questions, each with its check (31 Oct 2026, 17:55Z)

Kelvin's next step: "expand to the case where some of the bits come from the input". Standing direction: the
asymptotic case as compute -> infinity; better structured sampling counts as progress.

## Setting (fixed for the lane)
- Model: the stationary lane's Part C model (`work/stationary/scratch/partC.py`): x ~ N(0, I_n) at layer 0,
  h_l = relu(W_l h_{l-1}) + sigma xi_l for l = 1..L-1, output a = relu(W_L h_{L-1}) (no output noise), W_l = st.W_layer(seed, n, l).
  sigma = 0 is ARC's noiseless family; ARC's point is n = 256, L = 32, sigma = 1e-3.
- Regime: the input is not yet forgotten: L from n/8 to about T_mem = (4n/5) ln(1/(sigma sqrt(0.4 n))) (144, 271, 506,
  941 layers at n = 32, 64, 128, 256 for sigma = 1e-3).
- Seeds: dev seeds >= 9000 (calibration, not scored); evaluation seeds 4000-4003 (fresh; the stationary lane's Part C
  used 1000-1001, whose numbers I have read).
- Cost unit: one n x n matrix-vector product (mv); a pass costs L mv. Error: nMSE = |error|^2 / tr Var(a) per net.
- Annealed (averaged over weights; the exact replica Gram chain) and quenched (one fixed net) quantities are kept apart.

## Assumptions (each with how it is checked)
| id | assumption | check |
|---|---|---|
| A1 | sigma = 1e-3 is negligible for the input structure below T_mem/2 (the stationary lane: input share 0.999 at 256 x 32) | rerun the split and gate statistics at sigma = 1e-3 vs 0 on the same nets (coupled), report the change |
| A2 | for sigma = 0, a(x) = \|x\| g(x/\|x\|) exactly, so radius-exact (spherical) quantities are exact | code check: max relative error of a(x) - \|x\| g(u) at 1e-12 |
| A3 | the exact two-replica Gram chain gives the annealed values of every two-input quantity (angle, Hamming, split terms, single-coordinate Sobol effects) | chain vs nets (8 nets, KS / mean within 3 SE) as the numerics lane did for noise replicas |
| A4 | the gate statistics' marginal entropy sum is an upper bound on the joint entropy; the mean pairwise Hamming distance is the cleaner bit count (an exact annealed closed form) | report both; the Hamming count is the primary one |
| A5 | coordinate-level Sobol indices of a fixed net are exchangeable in law over coordinates (W_1 is rotation invariant), so sums over all n coordinates can be estimated from a random subset | compare a subset estimate with the full sum at n = 32, 64 |

## Open questions (each with its check)
| id | question | check |
|---|---|---|
| Q1 | What carries the input variance at depth L: the input radius \|x\|, the amplitude A(u) = \|g(u)\| (direction-dependent gain), or the output direction d(u)? Is "the radius carries the input" (stationary lane, not measured) right, and for which radius? | exact split V = V_rad + V_amp + V_dir on nets (unbiased U-statistics) and in the Gram chain; against the kernel + 1/n theory (FORMAL v1) |
| Q2 | How many gates depend on the input at each layer (no births when sigma = 0), and how fast do they decay with depth (algebraic 1/l from the finite-angle kernel contraction, or exponential at the Lyapunov rate)? Where do the 38 bits at 256 x 32 come from? | Hamming and marginal entropy per layer on nets vs kernel prediction n arccos(rho_{l-1})/pi and the Gaussian-cap entropy factor; Gram chain at finite n |
| Q3 | How do the input bits combine with the radius: is the amplitude variance set layer by layer by the input-decided gates (per-layer log-gain variance 8 theta_l^2/n)? | per-layer increments of Var log A on nets vs the theory sum Lambda(L)/(4n) |
| Q4 | What is the interaction order of the input-to-output map over input coordinates (mean dimension, first-order share), and of each of the three parts? | Saltelli/Jansen indices on nets; annealed single-coordinate refresh in the Gram chain; Stein degree-1/2 shares |
| Q5 | Is there a coordinate system (a rotation of the Gaussian input) in which the map is more additive (e.g. the eigenbasis of the degree-2 Hermite matrix of \|a\|^2)? | first-order share before/after rotation on the same nets |
| Q6 | Age and decay of input information by Lyapunov direction: how is the input cloud at layer l spread over the backward Lyapunov (QR) basis of a reference input, and how fast does each direction's share decay? Finite angles (kernel, algebraic) vs infinitesimal (gaps (k-1)/n)? | QR basis along a reference input; projections of input differences at layer l; small vs large input perturbations |
| Q7 | Does a structured method's gain follow the low-order share of the variance (RQMC: first-order share in its own coordinates; antithetic: the odd share, an identity; multilevel: the prefix-amplitude share), in both the noise and the input regime? | Part 2 (form registered before data) |
| Q8 | Which structured sampler for the input regime has a better rate or a gain growing with width, and how do constants scale with n at fixed L/n and at fixed L? | Part 3 (registered before data) |
| Q9 | Does noise at sigma = 1e-3 change Q1-Q6 below T_mem/2? | A1 |
