### 5.2 Fixed-size additive messages: exactly the exponential families; smooth families pay polylog

- **PROVEN UNDER** regularity (densities with support independent of theta and a continuously differentiable
  sufficient statistic; Koopman 1936, Pitman, Darmois): for iid sources, a sufficient statistic of fixed dimension for
  every N exists only for exponential families, where it is additive. Uniform(0, theta) shows the regularity is
  needed: its statistic max_i x_i merges by max, a monoid but not a sum.
- Distributed-statistics form: for combining local maximum-likelihood estimates, the asymptotic (second-order)
  information loss of the best one-shot combination is governed by how far the family is from a full exponential
  family, i.e. Efron's statistical curvature (Liu and Ihler, arXiv:1410.2653, under their regularity conditions).
  **PROVEN** (cited).
- Smooth families (**PROVEN UNDER** a prior supported on a compact box B, and l(theta; x) minus a constant c(x)
  analytic in theta on a common complex neighbourhood of B and bounded there by M uniformly in x): Chebyshev
  interpolation reaches sup error epsilon with O(log(M/epsilon)^d) terms. Per-source sup error epsilon gives
  log-posterior error at most 2 N epsilon, so KL <= 2 N epsilon, and O(log(M N/delta)^d) additive numbers per source
  suffice for KL <= delta. The tiny-PFN world meets this only approximately (Gaussian prior truncated to [-5.5, 5.5];
  unbounded readings make M grow slowly with the reading).
- **MEASURED** (N1e: the tiny-PFN world scaled to N sources; each source sends the r Chebyshev-interpolation
  coefficients of its log-likelihood of y* on [-5.5, 5.5]; the root sums them; KL(Bayes || approx) with the
  approximation normalised over the whole range, 95th percentile over 200 episodes at N = 3 and 64, 60 at N = 1024,
  25 at N = 4096; N1c gave the same numbers within a factor of about 1.5 with local normalisation):

  | N | r = 24 | r = 32 | r = 48 | r = 64 | r = 96 |
  |---|---|---|---|---|---|
  | 3 | 8.7e-3 | 2.7e-3 | 1.6e-4 | 8.0e-6 | 3.8e-8 |
  | 64 | 2.9e-2 | 1.2e-2 | 1.1e-3 | 7.0e-5 | 4.2e-7 |
  | 1024 | 7.3e-2 | 2.5e-2 | 1.0e-3 | 9.8e-5 | 8.7e-7 |
  | 4096 | 1.6e-1 | 4.7e-2 | 9.8e-4 | 1.0e-4 | 9.8e-7 |

  KL <= 1e-3 is crossed between r = 48 and r = 64 at every N from 64 to 4096 (r = 48 at N = 3).

  At fixed r = 64 the KL grows about tenfold from N = 3 to N = 64 and then barely moves up to N = 4096, far gentler
  than the worst-case 2 N epsilon; each extra 32 coefficients cut it about 100-fold, so the needed r grows like
  log N at most, as T3a says. The
  constant is large (tens of numbers) because the per-source log-likelihood switches between its three variance
  components over short distances.
