### 5.7 Estimation (M1/M3): two bond dimensions, alignment, expectations vs tails, noise

- **T11a, exact output law (PROVEN, as an algorithm).** If f has K_v classes along T and the input law is a latent
  tree along the same T with R_v latent states at node v (given u_v, the two child subtrees are independent), then
  Q_v(c | u_v) = sum over (c_a, c_b) merging to c of [sum_{u_a} P(u_a | u_v) Q_a(c_a | u_a)] [sum_{u_b} P(u_b | u_v) Q_b(c_b | u_b)],
  so the exact law of f(X) (every tail) costs O(sum_v R_v (R_a K_a + R_b K_b + K_a K_b)), i.e. O(R K (R + K)) per
  node for uniform sizes. This is a sufficient route, not a necessary one: algebraic structure (5.6) can bypass it,
  and a symmetric f can be evaluated along the law's own tree. What it shows is that alignment of the computation's
  hierarchy with the input law's makes exact tails cheap.
- **Misalignment.** If the law is a tree along T1 and the computation along T2, contraction of the union costs
  poly(size) x D^{O(w)}, with D the bond dimension and w the treewidth of T1 union T2 (Markov and Shi,
  arXiv:quant-ph/0511069; upper bound). **MEASURED** (N3a; random binary trees by random pairwise merging; treewidth
  of the union graph bracketed by a contraction lower bound (MMD+) and a min-fill upper bound; 5 pairs per N):

  | N leaves | T2 = copy of T1 | 4 random prune-and-regraft moves | 32 moves | T2 independent of T1 |
  |---|---|---|---|---|
  | 16 | 2 | 3.4-4.0 | 4.0-6.0 | 4.2-6.6 |
  | 64 | 2 | 3.8-4.2 | 5.4-14.2 | 6.8-23.6 |
  | 256 | 2 | 4.0-4.8 | 5.8-17.8 | 12.6-95.6 |

  Aligned hierarchies stay at treewidth 2; misalignment raises it steadily, and for independent trees the upper
  bound grows about linearly (0.37 N) while the lower bound grows slowly, so linear growth is not established
  (**CONJECTURE**: linear for independent random trees; a reviewer suggested an expander-minor argument, not checked).
- **Expectations need rank; tails need dimension (PROVEN).** A rank-chi tree network gives E_p f by contraction in
  O(N chi^3 R^3). Tails are expectations of indicators, whose rank can be exponentially larger: for integer weights,
  2^n P(sum_i a_i x_i > t) under the uniform law counts knapsack solutions, #P-complete in the worst case. With
  activation noise sigma and range R per coordinate, a d-dimensional message has about (R/sigma)^d distinguishable
  states, and the law of a sum is a convolution on that grid (pseudo-polynomial; exact up to the discretisation set by
  sigma).
- **Noise forces dimension to grow with scale (PROVEN UNDER** theta ~ N(0, 1); sources x_i | theta ~ N(theta, 1/I)
  independently, so the posterior is N(mu_N, v_N) with v_N = 1/(1 + N I); the root sees one m-dimensional message
  M = s(x) + noise with |s_k| <= R and independent N(0, sigma^2) noise per coordinate; its output is a Gaussian
  within forward KL delta of the posterior, in expectation**).** Forward KL <= delta allows mean-squared error up to
  e^{2 delta} v_N (the worst case over the output variance), so rate-distortion gives
  I(M; theta) >= (1/2) log2((1 + N I) e^{-2 delta}), while the Gaussian channel gives
  I(M; theta) <= I(M; s) <= (m/2) log2(1 + R^2/sigma^2). Hence m >= (log2(1 + N I) - 2 delta log2 e) / log2(1 + R^2/sigma^2).
  Bounded (normalised) noisy representations must widen like log N; unnormalised sums need not. So "larger
  structures need more dimensions" is a necessity, not a permission.
- **Additive messages give certified tails (PROVEN UNDER** finite moment generating function of each message near zero,
  e.g. bounded messages**).** Given the latents, a sum of independent messages has a computable cumulant generating
  function, so Chernoff bounds certify upper tails of any linear readout of the sum; saddlepoint approximations have
  relative error O(1/N) under the usual regularity (Daniels 1954). This applies to the additive log-odds message
  before any nonlinear readout. It is the M3 route for an SFU architecture.
