Model: GPT-6 (Codex). The experiment is promising, but the Bayes mixture and shuffled-history criterion need correction.

1. **Shortcuts and useful regimes.** Large noise alone does not make structure inference useful: overwhelming common noise makes predicting near zero competitive. Exchangeable, equal-sized clusters can preserve equal aggregation weights; structure then matters mainly for uncertainty. Use unequal cluster sizes, matched marginal variances, substantial within-cluster correlation, and enough independent groups that informative aggregation remains possible. Add positive idiosyncratic noise to eliminate exact-copy ties. Compare against the actual history-blind Bayes predictor \(p(v\mid r,N)\), which can infer structure from query reports; a fixed linear rule is too weak. Measure expected predictive KL or excess log loss, not only predictive-mean variance.

2. **Exact Bayes.** For hypothesis \(h\),
   \[
   V_h=(1+\mathbf1^\top\Sigma_h^{-1}\mathbf1)^{-1},\qquad
   m_h=V_h\mathbf1^\top\Sigma_h^{-1}r.
   \]
   Crucially, mixture weights are
   \[
   p(h\mid H,r)\propto p(h\mid H)\,
   \mathcal N(r;0,\Sigma_h+\mathbf1\mathbf1^\top).
   \]
   Weighting components only by history is incorrect. Enumeration is cheap only for a genuinely small family; enumerating arbitrary subsets, partitions, or permutations explodes. Cache Cholesky factors; use log-sum-exp and strictly positive private variance.

   Under the standard IW convention, posterior parameters are \(\nu'=\nu+T\), \(\Psi'=\Psi+\sum_t e_te_t^\top\). Query error has a multivariate Student distribution with degrees of freedom \(d=\nu'-N+1\) and scale \(\Psi'/d\). The proposed normalized one-dimensional integral is correct. Require \(\nu>N-1\), or \(\nu>N+1\) for finite prior covariance mean. After one factorization, its quadratic form is \(av^2-2bv+c\), making quadrature cheap; verify normalization, tails, and tolerance convergence.

   \(T<N\) prevents unrestricted covariance recovery, but proper Bayesian prediction remains defined. Factor exposures themselves are generally unidentifiable; covariance is the relevant target. IW generates a different world from sparse shared sources. Define dimension-consistent priors before interpreting extrapolation in \(N\).

3. **Tokenization.** Start with report tokens containing error \(r_{t,i}-v_t\), trader key, round key, and history/query flag; query tokens contain \(r_i\). Explicit errors remove subtraction as a confound. Randomize keys and token order each episode. Test simultaneous relabeling invariance separately from history/query misbinding sensitivity. A trader token containing a fixed history vector hardcodes \(T\) and relies on coordinate alignment; independently pooling each trader’s history destroys cross-trader round alignment. An axial round-by-trader architecture is a useful diagnostic comparator. Full attention over \(N(T+1)\) tokens already means over 8,000 tokens at the proposed largest settings.

4. **Learnability and controls.** Weak correlations, short histories, ill-conditioned signed loadings, and scalar-only supervision can make the useful signal inaccessible. Begin with positive loadings, moderate correlations, small \(N\), and generous \(T\); add covariance or hypothesis-posterior auxiliary supervision, then ablate it. Train a distributional head: a single Gaussian cannot represent the general exact target. Supply \(N,T\) explicitly and vary both during training to expose counting and uncertainty scaling.

   Shuffling identities need not erase all history gains: history still reveals global noise and cluster composition. Compare against Bayes given the same corrupted information. Define “0.3 nats” as a specific predictive KL; report how frequently such cases occur.

5. **Concrete change and first pilot.** Replace the initial broad prior with four traders and six equally likely hypotheses: exactly one correlated pair, marginal variance 1, pair correlation 0.8, others independent. All hypotheses have identical spectra and marginal accuracies, isolating binding. Pilot \(T=0,8,32,128\), first verifying a substantial full-Bayes versus history-blind Bayes gap. Then train and measure the fraction of that gap recovered, relabeling invariance, and misbinding degradation before expanding \(N\).
