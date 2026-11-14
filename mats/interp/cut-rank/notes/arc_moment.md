# Task 4: ARC's moment class against sampling (cut-rank lane, 2026-10-31 ~00:05Z)

## 1. The arithmetic under ARC's conjectured error law (PROVEN as arithmetic; the law itself is ARC's conjecture)

ARC (Wu et al. 2026, Sec. 5 / App. D): order-K cumulant propagation has nMSE <= c_K (L/n)^K ("mean squared error",
normalised by the output variance); cost <= c'_K L n^{K+1} (basic) or c''_K L^2 n^K (factorised, K >= 3). Plain Monte
Carlo: nMSE = 1/N at cost 2 L n^2 N.

- Target RMSE eps (nMSE eps^2). Moments need (L/n)^K <= eps^2 / c_K, i.e. K >= 2 ln(1/eps) / ln(n/L) (constants dropped).
  Cost ~ L n^{K+1} = L n (1/eps)^{2 ln n / ln(n/L)}. Sampling: 2 L n^2 (1/eps)^2.
- Ratio (basic) = n^{K-1} eps^2 / 2. The exponent of 1/eps in the moment cost is 2 ln n / ln(n/L) > 2 for every L > 1,
  so with ARC's law read as an MSE bound the hierarchy loses to sampling as eps -> 0 at every depth L > 1. At a given
  eps it wins iff ln(1/eps) < ln(2n) ln(n/L) / (2 ln L): at L = sqrt(n) iff eps > (2n)^{-1/2} (fewer than ~2n
  samples matched); at L = n^{1/4} iff eps > (2n)^{-3/2}; at L = n^{3/4} iff eps > (2n)^{-1/6}.
- The coordinator's form "(1/eps)^{ln n / ln(n/L)} against (1/eps)^2, losing when L > sqrt(n)" is the same computation
  with (L/n)^K read as the RMS error rather than the MSE. Under the MSE reading (ARC's wording) the threshold
  "L > sqrt(n)" is where the hierarchy loses already at eps ~ n^{-1/2}; it loses at small enough eps for any L > 1.
- At fixed eps the classic statement is the depth crossover: basic order K loses beyond L* ~ n^{1/K}, factorised
  order K beyond L* ~ n^{2/(K+1)}; for K = 2 and factorised K = 3, L* ~ sqrt(n) (arc-deep-networks note, section 1.2).

## 2. Check against measured numbers (MEASURED, from earlier lanes; nothing new run here)

At fixed alpha = L/n the law predicts a constant nMSE c_K alpha^K, so the gain over equal-cost Monte Carlo should fall
like 1/n for K = 2 (basic, N_eq ~ n) and like 1/n^2 for factorised K = 3 (N_eq ~ alpha n^2).
- K = 2 (hardness report 6.3, alpha = 1/8, 16 seeds; 4 at n = 1024): gain 7.76, 4.06, 2.57, 1.91, 1.47, 1.09 at
  n = 32 ... 1024: it falls like n^{-0.56}, slower than the law's n^{-1} (measured nMSE_K2 itself falls like n^{-0.41}
  instead of staying constant). The law is pessimistic for K = 2 at L proportional to n (it is an upper bound).
- Factorised K = 3 (hardness 6.3, alpha = 1/8, 8 seeds; 3 at n = 256): gain 2.16, 0.80, 0.38, 0.08 at n = 32, 64, 128,
  256, i.e. like n^{-1.58}, close to the law's n^{-2}; at alpha = 1/4 it loses 27-247x. At 256 x 32 (L = 2 sqrt(n)) the
  arc-noisy lane measured factorised K = 3 about 11x behind plain Monte Carlo and K = 2 about at parity (gain 1.9 in
  the hardness table, parity at 2e9 FLOPs in arc-noisy).
- So the law's prediction "the hierarchy's higher orders lose increasingly with width at L proportional to n" is
  borne out for factorised K = 3; K = 2 does better than the law and roughly matches sampling at its own (single, low)
  budget, losing slowly with width.

## 3. What the cut-rank results say about the moment class (PROVEN / MEASURED, scope stated)

- A moment summary read out LINEARLY is a one-way linear protocol: order <= K moments of the cut law give rank
  C(n+K, K), so their error is at least the SVD tail of the cut operator at that rank (Fact 1). The harmonic version
  (astra A3) is the natural basis: truncating the back at harmonic degree m is a rank-S(n, m) protocol with error
  <= (n-1)/((m+1)(m+n-1)) (relative to E a^2); harmonic degree plays the role of moment order. PROVEN.
- MEASURED (this lane, held-out fronts, least squares on order-K cut moments, scored with independent input splits):
  the best linear readout from the cut's first moments (n features) leaves 8-9% of Vbar at n = 8 (c = L' = 4 and 8),
  21-27% at n = 16 (c = 8, L' = 2-8), 23-38% at n = 32 and 59-74% at n = 64 (128 training fronts for 65 features, so
  noisy); adding second moments (n = 8, 16) helps little or overfits at 512 fronts. In sampling units that is worth
  about 1.5-12 samples. ARC's K = 2, which reads the same second moments NONLINEARLY (Gaussian closure through the
  back) and propagates them from the input, has median nMSE 1.9e-2 at n = 32, depth 16 and 4.7e-3 at n = 64, depth 16
  (hardness report table 6.3), i.e. worth about 50-200 samples: 15-70x better than the best linear readout.
- Consequence (plain): the one-way linear bound is about linear readouts of a summary. ARC's closures are nonlinear
  readouts and are not covered; empirically they do far better than the best linear readout of the exact cut moments.
  A rank lower bound therefore cannot by itself show that ARC's hierarchy loses; the hierarchy's measured loss at depth
  comes from closure error (its bias), which needs a different bound (section 2 is consistent with ARC's law there).
