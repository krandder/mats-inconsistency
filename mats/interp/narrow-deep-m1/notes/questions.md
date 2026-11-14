# M1 width 8: assumptions and open questions (30 October 2026, ~22:45Z)

Lane: narrow-deep-m1. Target: a sample-free estimator of E[a_L] (and one tail probability near 1e-4) on random plain
He-ReLU nets of width 8 and depth 256, with small noise on every layer, that matches or beats RQMC at equal FLOPs.

Each item says how it is checked.

1. Net family: `nd.Net('plain', seed, 8, 256, sigma)` from `work/narrow-deep/lib/nd.py`, unchanged. W_l iid N(0, 2/n),
   no bias, x ~ N(0, I_8), h_l = ReLU(W_l h_{l-1}) + s_l eps_l for l < L, output a_L = ReLU(W_L h_{L-1}) (no noise after
   the output). s_l = sigma r_l, r_l the RMS of the clean activation a_l in the noisy net from a seeded 2^14-path pilot.
   Check: import the module from the old path, do not copy it; record r_l per net in every result file.
2. Noise convention: estimators and samplers use the same s_l. The pilot is part of the net and nobody is charged for
   it (as in both earlier lanes). Check: every estimator reads `net.s`.
3. Estimands: E[a_L] in R^8, nMSE = |est - truth|^2 / tr Var(a_L) (ARC's convention; plain MC with N paths scores
   1/N in expectation). Tail: p = P(|a_L| > t), t the (1 - 1e-4) quantile of |a_L| from the truth run; score relative
   MSE. Check: the truth file stores mean, per-coordinate variance, t and p with standard errors.
4. FLOPs: flopscope conventions as both earlier lanes (matmul (M,K)@(K,N) = M N (2K - 1), elementwise 1, normal draw
   16, exp/log/erf 16, normal cdf 96, pdf 54). For the deductive method everything is charged: forward, backward
   (value function), grids, interpolation, merging (a comparison or index operation counts 1), quadrature-node
   tables are charged when built per net. Fixed tables that do not depend on the net (e.g. 1D Gauss-Hermite nodes)
   are free. Check: a FLOP counter object threaded through every routine, unit-tested on a 2-layer net by hand count.
5. What counts as sample-free: no pseudo-random numbers anywhere, and the estimate is not an average of the network's
   output over a point set in input space (that would be QMC, which is a sampler). Deterministic grids over a
   low-dimensional value function, evaluated by deterministic forward passes from a cut, are allowed; so are
   deterministic quadrature rules for 1D/2D Gaussian integrals. Check: the estimator module imports no RNG; review.
6. Dying nets: with noise, a net whose noiseless version kills every input is not exactly zero. The census
   (`runs/census_*.json`) shows its output is noise-driven: the input explains about 0-10% of the output variance and
   no path is exactly 0. So "dead" nets are not trivial here; they are a noise-driven problem started at the death
   layer. Check: classify by the census before any estimator result, report the classes separately, and test the
   estimator on at least two dead nets as a secondary arm.
7. Truth: plain MC or RQMC with >= 2^24 paths per net; its own error must be at most 1/10 of the smallest nMSE we
   claim a ratio at. Check: replicated RQMC truth gives an error bar; MC truth gives 1/N.
8. Baselines at equal FLOPs: plain MC; RQMC (scrambled Sobol on the input, iid noise; the best simple sampler last
   time); the sample-the-shallow hybrid of the previous lane (MC to l0, radius-binned deterministic deep part).
   Check: same FLOP model; budgets 1e2-1e6 paths' worth.
9. Is "ARC's method until layer 5-8" valid at width 8? ARC's kprop drops diagrams by power counting in 1/n, which is
   not small at n = 8; at width 256 their factorized K = 3 beat sampling to depth ~8. Check: run ARC's kprop (K = 2,
   3, and 4 if the code supports it at n = 8) at depths 1-8 against exact (sampled, 2^24) moments of the layer-c law,
   with FLOPs, plus an exact-moment K2 closure.
10. Where to meet and with what representation: the previous lane's k*_F(M; 1e-2) at n = 8 is 5.3 at cut 4 and 3.5 at
    cut 8, i.e. the value function may be low-dimensional only from cut ~8. Check: oracle study on dev nets (sampled
    law at cut c, sampled value function) of the projected-meet error against k and c, and of the number of
    downstream-distinct classes against the raw count of gate histories.
11. Scaling (coordinator, 22:41Z): if width 8 succeeds, rerun the method at widths 16 and 32 with L = 32 n (64 if
    affordable on CPU), and report its cost against width (polynomial or exponential). Check: pre-registered in PREREG.
