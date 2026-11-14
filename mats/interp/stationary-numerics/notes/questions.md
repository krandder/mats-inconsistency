# Numerics lane: assumptions and open questions (14:30Z 31 Oct, before any computation)

Model S throughout (absolute noise; sigma = 1 by scale invariance, FORMAL 2 of the stationary lane):
h_l = relu(W_l h_{l-1}) + xi_l, W_l iid N(0, 2/n), Q_l = |h_l|^2/n. Each item: assumption or question -> check.

## 1. Exact radius chain
1.1 Reduction (to be proven here): Q_l = (S_l/n) r_l with S_l ~ chi2_n independent of r_l, and r_l = 1 + A_l r_{l-1}
    (a perpetuity with B = 1), A_l = (2/n) chi2_M, M ~ Bin(n, 1/2). In u = log(r - 1) the step is
    u' = log A + softplus(u): a fixed smooth map followed by a translation by the fixed law of log A.
    -> proof from the Beta-Gamma independence; check the reduced chain's quantiles against the direct chain.
1.2 The stationary density of u is smooth with exponential decay at both ends (right: rate 1, the Kesten index;
    left: rate 1/2 from the M = 1 term, weight n 2^-n), so Nystrom with the trapezoid rule converges spectrally.
    -> halve the step, widen the domain; target 1e-10 on CDF values; GTH or positive power iteration so that
    tiny tail probabilities keep their relative accuracy.
1.3 Finite-n corrections: F_n(xn) = exp(-2/(5x)) + F_1(x)/n + O(n^-2), with F_1 from the Kramers-Moyal expansion of
    the chain to first order in 1/n. -> derive F_1 in closed form; check n (F_n - F_inf) -> F_1 on n = 64..1024.
1.4 Kesten constant C_n = 1/E[A log A] (astra P4): -> check x P(Q > x) -> C_n in the numerical law and measure the
    rate of approach; check the 1/n expansion 2n/5 + 13/75 against the KM tail of F_1.
1.5 Left tail (astra: "near-floor layers ~ exp(-0.4 n)" is a CONJECTURE): -> exact P(Q <= q) at n = 2..512, fit the
    exponential rate in n; derive it if the numbers suggest a form.
1.6 Simulation agreement: -> exact quantiles vs (a) >= 1e7 scalar-chain draws, (b) the stationary lane's net chains
    (runs/*radius*), within their Monte Carlo error (pre-registered).

## 2. Finite-width effective theory with noise
2.1 Assumption to test: the PDLT kernel/vertex recursions with an additive noise source have a stationary fixed
    point. Doubt: annealed, K_l = K_0 + 2 sigma^2 l grows forever (E Q = infinity), and V/K^2 ~ 5 l/n leaves the
    perturbative regime at l ~ n, which is exactly the stationary scale. -> algebra; state what does have a fixed
    point (normalised overlaps and vertices conditional on the radius).
2.2 Claim to prove: for R noise replicas sharing the weights, the R x R Gram matrix G_l = H_l^T H_l / n is an exact
    Markov chain (fresh W_l is rotation invariant on its input side), G' = (1/n) sum_i v_i v_i^T with v_i iid given G.
    Its 1/n expansion is the PDLT kernel + four-point-vertex recursion with the noise source. -> proof; check the
    Gram sampler against full-net replicas at small n.
2.3 Which quenched observables are replica observables: tr Cov_xi(h|W) (2 replicas), E_W tr C^2 (4 replicas, gives
    the participation ratio of the cloud spectrum), Sobol totals by age and mean dimension (2 replicas with partial
    noise sharing). -> identities, then compute from the Gram chain.
2.4 Does the perturbative (PDLT) closure reproduce the exact Gram chain's stationary overlap delta (astra's
    2.23 X^-2/3), radial CV and interaction order? -> solve both, compare by radius bin.
2.5 Does theory explain (a) the 5-8x cloud excess over sigma^2/(2|lambda_k|), (b) the mean interaction order 2-3
    (Part I) and its width dependence? -> predictions vs the stationary lane's measured numbers, same definitions.
2.6 The ~1/k covariance spectrum: is its shape a replica (4-point) consequence? -> participation ratio from 4
    replicas vs measured.

## 3. Exact quenched densities at n = 2, 3, 4
3.1 The grid transfer operator (push forward by relu(W h), then convolve with N(0, I)) is accurate to
    O((dx/sigma)^2) and the box captures the mass. -> refine the grid, enlarge the box, compare with a Monte Carlo
    histogram of the same net.
3.2 The pullback converges (synchronisation, plus exact resets with probability 2^-(n^2+n) per block).
    -> TV distance between pullbacks from different starts against depth T.
3.3 Where mass sits: by gate pattern of the last layer (mass on faces/axes/origin before the final noise).
    -> exact masses per pattern from the grid; density near the preactivation hyperplanes w_i.h = 0.
3.4 Non-Gaussianity: -> KL and TV of the exact density against the moment-matched Gaussian and a gate-pattern
    Gaussian mixture.

## 4. Response kernels and data collapse, n = 8-256
4.1 Definition: "Lyapunov direction k at layer L-j" = k-th right singular vector of the j-step tangent propagator
    (finite-time forward vector; its linear response is the k-th singular value) and the k-th backward Gram-Schmidt
    vector (depends on the past only). -> report both; pre-register which one the collapse uses.
4.2 Linear vs finite responses: tangent (infinitesimal), finite perturbations of size ~ sigma (noise scale), and
    one-gate flips. -> same trajectories, same subsequent noise and weights.
4.3 Expected collapse variables: (k-1) j/n for linear angular decay (gaps 1/n), theta_0 j for the finite-angle
    kernel contraction (algebraic), j/n for the radius. -> pre-register forms, fit with bootstrap CIs over nets.
4.4 Flip lifetimes: heavy-tailed? -> distribution of the time until the flipped and baseline gate patterns
    re-synchronise and until |dh|/|h| falls below the cloud scale; survival function on log-log axes.
