# The stationary regime of noisy critical ReLU stacks: what the stationary state is, and how well it can be estimated as compute grows (31 October 2026)

Lane: stationary regime (target M1), bead pool-qxszj1. Status: complete; revised after a GPT-6 review (section 6).
Work directory `work/stationary/`:
- `FORMAL.md` (theory v2.2; v1, frozen 13:15Z before the co-prover's answer, is `coprover/FORMAL_v1_as_written.md`);
- `PREREG.md` with Parts A, B, C, I and Amendments A2, A3, B1, B2, C1; hashes and times in `prereg_hashes.txt`
  (A 13:22Z, B 13:40Z, C 13:43Z, C1 13:51Z, A2 13:56Z, A3 13:58Z, I 14:10Z, B1 and B2 14:19Z, each before its runs);
- `notes/RESULTS_INDEX.md` (where every measured number lives);
- `coprover/` (gpt-6-astra rounds, prompts, raw logs, extracted answers, `PROVENANCE.md`);
- `lib/st.py`, `tests/test_st.py` (all pass, `logs/test_st.log`), `scratch/` (all scripts), `runs/` (JSON),
  `results/` (scorecards and tables), `review/`.

Labels: PROVEN, PROVEN UNDER <hypothesis>, MEASURED, HEURISTIC, CONJECTURE. Provenance: [C] Claude, [A] gpt-6-astra,
[C+A] both independently.

## Summary for Kelvin

**Short answer.** The stationary state of a critical ReLU stack with absolute noise is now pinned down, and most of the
coordinator's picture survives in shape but not in size. It is not an easier estimation problem. As compute grows, the
mean squared error of every unbiased method falls like 1/compute (plain MC needs a log for its burn-in). RQMC's early
exponents of 1.1-1.2 turn out pre-asymptotic: over N = 2^10..2^16 they are 1.03-1.11 and its gain saturates. The
deterministic closure has a floor. Structure buys constant factors (2-8x, at matched burn-in) that do not grow with
width.

What the stationary state is (Part A, pre-registered, n = 8-256):
1. **It exists and is unique** for fixed weights, and with the noise fixed too all trajectories merge onto one
   (PROVEN, with the co-prover).
2. **sigma does not matter**: the state for noise sigma is exactly sigma times the state for noise 1 (PROVEN). One
   layer's noise is about n^-1/2 of the state at every sigma, so "small noise" is set by width, not by sigma.
3. **The radius is an exact one-dimensional random walk** (a Kesten chain; PROVEN): the exact chain's median of
   |h|^2/(n sigma^2) is 0.575-0.579 n at every width from 8 to 256 (simulated; the large-width limit 0.5771 n is
   PROVEN), with a Pareto tail over layers, P(> x) ~ (2n/5 + 13/75 + O(1/n))/x. The nets' medians over layers are
   0.565-0.606 n, and their quantiles match the chain's within 6% at n <= 64, 7.5% at 128, 19% (90% quantile) at 256.
4. **For fixed weights the state is a thin cloud around a mean point**, increasingly Gaussian with width (radial skew
   0.3-0.4 at n = 256). The cloud is larger than the Lyapunov formula sigma^2/(2|lambda_k|): 1.2x at n = 8, 5-8x at
   n >= 64. Its size follows a finite-angle law instead: two noise replicas sit at angular distance
   delta = 2.23 X^(-2/3) (X = |h|^2/(n sigma^2)), and the radius varies by about 1.36 n^(-1/3) within a net. Both
   passed a fresh pre-registered test at n >= 32: measured/law medians 0.87 (range 0.72-1.24) and 0.80 (0.54-1.16).
5. **Bits are many at every width.** About 0.5-0.6 sqrt(n) gates flip per layer from the newest noise (the
   Gaussian-unit formula n arccos(Q/(Q+1))/pi fits within 2-9% at n >= 64; per fixed row the exact flip probability is
   2 Phi(t) Phi(-t), t = w.a/(sigma |w|)). One layer's noise goes on to flip 1.6 n (n = 16) to 3.8-4.5 n (n = 256) gates
   over the next 2n layers. There is no few-bits regime to enumerate.
6. **The interaction structure (your follow-up).**
   - The noise-to-state map has a mean interaction order of about 2-3 layers (finite-eps statistic, a lower bound;
     the bulk of the variance is at order about 2).
   - It lives in the angular noise (>= 96% of the effect); the noise along the radius barely matters, even for the
     radius.
   - Gate switching looks like the main mechanism (HEURISTIC reading of two controls): the pure radius chain has
     order 1.2; a net with the radius clamped keeps most of the order.
   - It is not log-additive. Whether it grows with width is unresolved: per-age sums in the original noise coordinates
     give 2.1 -> 3.9 from n = 16 to 256, the refresh statistic in radial/angular coordinates gives 2.5, 2.1, 3.1 (two
     different decompositions, 2 nets each). Only the output gate pattern's order clearly grows (5 -> 15).

Estimating E[a] and a 1% tail as compute grows (Part B):
| method | MSE vs compute | gain over MC (matched burn-in; setup excluded) |
|---|---|---|
| burn-in multilevel (optimal finite-level allocation) | 1/compute (removes a log) | worse than MC at bias^2 <= 1e-6 on 10 of 12 nets (better on both n = 256 nets: 1,667 and 1,991 against 2,049) |
| RQMC on recent noise | 1/compute asymptotically: fitted exponents 1.1-1.2 over N <= 4096 fall to 1.03-1.11 over N = 2^10..2^16 | saturates near 5-12x (n = 8), 5-7x (16), 3-4x (32); 2.1-2.6x at n = 128 |
| antithetic noise (xi and -xi) | 1/compute | 1.9x (small n) to 2.4x (n = 256-1024), free; levels off |
| Gaussian-surrogate control variate | 1/compute | 1.3-2x, flat |
| Gaussian closure (deterministic) | a floor of about 0.12/n | better than MC only near its own cost |
| tail: gradient-tilted importance sampling | 1/compute | 3-9.5x (at n = 256 4.8-5.5x at the actual budget with its pilot charged) |

The best gain at matched burn-in shrinks with width (7x at n = 8, 2.4x at n = 256): no width-growing advantage
(registered hypothesis HB holds). If the structured runs are charged their actual burn-in 8n against MC at the
minimal burn-in B_6, the best gains are 3.5, 3.1, 4.1, 0.8, 1.2, 2.4 (review finding 1): still no growth.

Input bits (Part C):
- The input is forgotten after about T_mem = (4n/5) ln(1/(sigma sqrt(0.4n))) layers, and input and noise variance add
  (interaction share at most 0.26).
- Gate "bits" of the input fade long before that: the sum of marginal gate entropies over inputs is 38 bits at layer
  31 of ARC's 256 x 32 (an upper bound on the joint entropy). Most of the input's remaining influence there is not in
  gate variability; its radius is the obvious carrier (not separately measured).
- At ARC's point with sigma = 1e-3 the noise is negligible (input share 0.9992 nested, 0.9995 crossed ANOVA; mean
  shift 3e-6 nMSE). The stationary-regime tricks do nothing there; RQMC over the input gives 2.1x.

What I would ask next (open questions, not recommendations):
- Can an estimator use the finite-angle law (cloud size known from the radius) without paying for the gate
  interactions?
- Is the absence of a width-growing gain fundamental here? Order-2-3 interactions spread over 2n-4n layers of
  n-dimensional noise suggest so, but nothing is proven.

## 1. Setting

- Nets: width n, W_l iid N(0, 2/n), no biases, relu. Every layer adds Gaussian noise:
  h_l = relu(W_l h_{l-1}) + s_l xi_l. Output a = relu(W_1 h_0) at the deep layer 0, after an infinite past.
- Two noise conventions.
  - Model S (the coordinator's model): s_l = sigma, absolute. The only one with a stationary state independent of the
    input.
  - Model R (the earlier lanes' convention): s_l = sigma r_l, with r_l the pilot RMS of that layer's activation.
- "Fixed weights": everything is conditional on one weight sequence. The stationary sample measure mu^W is the law
  of h_0 over the noise for those weights.
- Cost unit: one n x n matrix-vector product ("mv", 2n^2 flops). Errors in nMSE = |error|^2 / tr Var(a), so plain
  MC with N exact samples scores 1/N.

## 2. Part A: the stationary sample measure of Model S

Theory: `FORMAL.md` v2. Measurements: `results/partA_tables.md` and `partA_scores.json` (evaluation seeds 1000-1003,
n = 8, 16, 32, 64, 128, 256; registered in PREREG Part A before any evaluation run).

### 2.1 It exists, it is unique, and with the noise fixed all trajectories merge (PROVEN)
- [A] Two layers with an all-negative weight matrix after nonnegative noise map every state to the same point. This
  happens a.s. eventually, so all long pullbacks coincide exactly: a random point attractor, and a unique sample
  measure among all entrance laws. The waiting time is astronomical, so this is a correctness device only.
- [C identity, A constant] The practical contraction: one layer satisfies
  E_W |relu(Wh) - relu(Wh')|^2 = |h - h'|^2 - (2|h||h'|/pi)(sin t - t cos t) exactly, and p-th moments of differences
  contract uniformly for 1 <= p < 2, with -log rho = 5/(8n) + O(n^-2) per layer at p = 1 (the noise cancels in
  differences). Existence and the tail statements hold for almost every iid weight sequence.
- [A] Every fixed-weight sample measure has exponential squared-radius tails; the heavy tail below is across layers.
- MEASURED (PA4 hit; the fit uses the relative distance |hA - hB|/|hA|, not the registered absolute distance, a
  deviation): the coupling-from-the-past distance shrinks at n x rate = -1.15 to -1.54 (prediction -5/4).

### 2.2 sigma is not a parameter of the stationary state (PROVEN [C+A])
relu is positively homogeneous and there are no biases, so the stationary state for noise sigma is exactly sigma times
the one for sigma = 1 (checked to 5e-15, PA12). The noise-to-signal ratio of the stationary state is set by the width:
one layer's noise is about 1/sqrt(Q) of the state, Q ~ 0.58 n. The sigma axis (1e-3 to 1e-1) only matters on the
way down from an input (Part C) and in Model R.

### 2.3 The radius is an exact one-dimensional chain (PROVEN [C+A]; MEASURED)
- Q_l = |h_l|^2/(n sigma^2) satisfies Q_l = A_l Q_{l-1} + B_l with iid (A, B) built from chi-square variables
  (A = 2U/n, B = (U + V)/n, U ~ chi^2_M, V ~ chi^2_{n-M}, M ~ Bin(n, 1/2)). This holds averaged over the weights;
  by ergodicity it is also the distribution over the layers of one net.
- Consequences: E Q = infinity (He criticality, E A = 1); tail P(Q > x) ~ C_n/x with C_n = 2n/5 + 13/75 + O(1/n) [A];
  Q/n -> 2/(5Z) with Z ~ Exp(1) as n -> infinity, so the median tends to 0.5771 n [C heuristic, A proof].
- MEASURED (PA1 missed as registered: the 10%-everywhere prediction fails at n = 128-256, the 20% refutation threshold
  at n <= 64 is not crossed; PA2 median and Hill hit): the evaluation nets' radius over 40,000 layers matches the
  scalar chain (10/50/90% quantile ratios 0.94-1.05 at n <= 64, up to 1.075 at 128 and 1.19 at 256); the exact chain's
  median of Q/n is 0.575-0.579 at every width from 8 to 256 (the nets' 0.565-0.606); Hill tail index 0.8-1.2.
- So "the radius does a random walk down to the noise floor" is right, and exact: log Q drifts down at 5/(2n) per layer
  and is pushed up by one layer's noise; the stationary radius is about 0.76 sigma n, with a Pareto tail over layers.

### 2.4 Lyapunov exponents (PROVEN [A]; MEASURED)
- At fixed width the true infinite-time exponents are all -infinity: a layer with no active unit (probability 2^-n)
  zeroes the Jacobian [A] (v1 corrected).
- Conditioned on enough active units, gamma_k = -(k + 1/4)/n - (k^2 + k + 1/24)/n^2 + ... [A]: the coordinator's gaps
  (k-1)/n are right to first order.
- MEASURED (PA3 hit under the registered median rule: medians of n lambda_1 -1.42, -1.16, -1.06, -1.25, -1.28 at
  n = 16..256; individual trajectories -0.8 to -1.7, SE about 0.2; median gaps 1.93, 1.23, 1.06, 1.06 at n = 32..256);
  at n = 16 rank collapse breaks the exponents beyond k = 3. Caveat: the QR warm-up started from h = 0.

### 2.5 The attractor and the cloud (MEASURED)
For fixed weights the sample measure at a layer is a cloud around a mean point m (the noise-made "attractor").
| n | radial share of E\|h\|^2 (along the noiseless attractor direction) | \|m\|^2/E\|h\|^2 | radius CV within the net | radial skew | cloud / coordinator (k = 2-8) |
|---|---|---|---|---|---|
| 8 | 0.51-0.83 | 0.39-0.47 | 0.36-0.66 | 0.35-1.35 | 1.2 |
| 16 | 0.35-0.98 | 0.37-0.74 | 0.33-0.57 | 1.1-1.3 | 2.1 |
| 32 | 0.69-0.88 | 0.60-0.76 | 0.34-0.40 | 0.7-0.9 | 2.7 |
| 64 | 0.79-0.95 | 0.71-0.87 | 0.32-0.34 | 0.56-0.66 | 6.7 |
| 128 | 0.82-0.96 | 0.79-0.89 | 0.23-0.28 | 0.45-0.55 | 5.7 |
| 256 | 0.92-0.98 | 0.90-0.93 | 0.18-0.23 | 0.29-0.42 | 7.7 |
- The cloud becomes thin and near-Gaussian as width grows (angular projections |skew| <= 0.25 at n >= 32; the radial
  skew falls roughly like n^-0.4, so PA11's "radial stays non-Gaussian at 256" is refuted).
- The coordinator's cloud formula sigma^2/(2|lambda_k|) has roughly the right shape (log-log slope about -1, but
  -0.5 to -1.5 net to net: PA5 missed) and the wrong size: 1.2x too small at n = 8, 5-8x at n >= 64 (PA6 hit), more
  on layers whose radius is large; 14-160x for the radial direction. A covariance formula from Lyapunov exponents needs
  coefficients fixed independently of the noise, which gates are not [A]; frozen-gate recursions get the shape wrong.
- Post hoc: weighting the injected relative noise by the radius history (R_0/R_{-j})^2 explains part of the excess
  (remaining ratio 1.9-2.5 at n = 32, 2.4-11.6 at n = 256). The rest is open.

### 2.6 Bits: births and deaths (PROVEN formulas; MEASURED)
- Flip probabilities under the last layer's noise. Per fixed row, exactly: resampling 2 Phi(t) Phi(-t), deletion
  Phi(-|t|), t = w.a/(s|w|) [A]. Averaged in the independent Gaussian-unit model: resampling arccos(r)/pi [C],
  r = tau^2/(tau^2 + s_z^2). The coordinator's sqrt(2/pi) counts units in a one-SD band, not flips [A].
- MEASURED (PA8, PA9): flips per layer match the Gaussian-unit formula n arccos(Q/(Q+1))/pi with the path's own
  radius within 2-9% at n >= 64 (worst bin 18-24% at n <= 32: PA8 missed there). Births per layer are 0.48-0.64 sqrt(n)
  (PA9 hit), i.e. about 2 at n = 16 and 8-10 at n = 256. At a fixed layer the sum of marginal output-gate entropies is
  6 bits at n = 8 and 31-73 bits at n = 256 (about 3-4.5 sqrt(n); an upper bound on the joint gate entropy).
- Deaths (Amendment A2, registered 13:56Z): after one layer's noise is resampled, gates keep flipping at every later
  layer: from the first layer's 2-25 flips down to about 1-2 per layer by n/2 layers later, where they stay. The
  cascade over the next 2n layers totals 1.6-1.7 n (n = 16), 2.3-2.6 n (64) and 3.8-4.5 n (256) flips (PA13b's bound 3n
  exceeded at 256). The frozen-gate linear response to that resampling stays accurate for a few layers (correlation
  with the true change 0.94-0.98 at age 2 and 0.73-0.93 at age 8 for n >= 64; PA13c hit) and is lost by age about n
  (0.2-0.5). So bits do not die quickly: a perturbation keeps re-deciding gates for about n layers.
- So the stationary state of Model S is at the many-bits end at every width: there is no enumeration regime.

### 2.7 Age profile of information, direction by direction (MEASURED; `results/age_by_direction.json`)
Method: resample the noise of one layer j layers back, keep everything else, measure the change of h_0.
- The radius holds the oldest information: the radial share of the response rises from 1-5% at age 0 to 0.5-0.95 at
  ages n/2..n (PA10a hit); the median age of radial information is 0.3-1.1 n (prediction 0.28 n).
- The number of directions excited by noise of age j falls like n/j: participation ratio 1.3-1.9 x (n/(2j) + 1)
  (PA10b hit under the registered median rule; largest median discrepancy 1.91x).
- But the cloud's own principal directions do not have memories n/(2(k-1)): their median information ages are nearly
  flat in k (n = 256: 32, 28, 28, 29, 24 layers for k = 2, 3, 5, 9, 16 on one net, 78, 73, 77, 61, 57 on the other;
  the coordinator's model says 89, 44, 22, 11, 6).
- Strong interactions: the per-age total effects sum to 1.1-1.4 (n = 8), 1.9-2.4 (16-32), 2.5-2.6 (64), 3.0-3.1 (128)
  and 3.6-4.3 (256) times the variance; an additive (linear-Gaussian) state gives exactly 1 (PA10d hit). Ages >= n carry
  4-52% of the sum (PA10e missed): the deep past matters through the radius, and multiplicatively.

### 2.8 Where the near-Gaussian picture holds and fails
- Holds: within one net at large n (thin cloud, skews <= 0.4 at n = 256), and births follow the Gaussian-unit formula.
- Fails: (1) across layers and nets (the inverse-gamma radius, Pareto tail index 1); (2) at small width (radial skew
  about 1 at n = 16, near-floor layers at n = 8); (3) in how the state depends on the noise: strongly non-additive
  (total effects 2-4x), so a near-Gaussian marginal is not a linear-Gaussian model; (4) in the size of the cloud and in
  the memory of its principal directions (2.5, 2.7).

### 2.9 Model R (relative noise) in brief
- Exact scale/shape decomposition [A]: the global input scale is kept forever; there is no input-free stationary law.
- MEASURED: births match the Gaussian-unit formula with each path's own ratio (PR1 hit); the noise cloud at a fixed
  input is not proportional to sigma^2 and is steep at small sigma (PR2 missed); the spread of log|h| over the noise at
  a fixed input grows with depth even at sigma = 1e-3 (0.01-0.04 at L = 4n, 0.08-0.33 at L = 16n).

### 2.10 The interaction structure: which injections interact, and why (Part I; your follow-up)
Registered in PREREG Part I (14:10Z) before any of these runs; `scratch/partI.py`, scorer `scratch/analyze_I.py`,
`results/partI_scores.json`. Seeds 1000, 1001 at n = 16, 64, 256; window = the last 4n layers; one "block" = one
layer's noise, written in causal radial/angular coordinates (the component along the current pre-noise direction, and
the rest). Method: refresh each block independently with probability eps; G(eps) = E|f - f_eps|^2/(2V) =
sum_S (1 - (1-eps)^|S|) V_S/V. Its limit G(eps)/eps -> sum_S |S| V_S/V as eps -> 0 is the variance-weighted mean
interaction order (Owen's mean dimension); the reported D = G(0.01)/0.01 is a finite-eps lower bound of it (for the
output gates it drops from 13.2 at eps = 0.01 to 7.5 at 0.03, so for them the finite-eps effect is large); G(0.1)/0.1
measures the order of the bulk of the variance. These orders refer to the radial/angular block coordinates; Part A's
per-age sums used the original (Cartesian) noise blocks, a different decomposition. Pairs: Liu-Owen superset importance (all ANOVA terms
containing both ages).

| n | mean order D: h_0 / log\|h_0\| / direction / output a / output gates | bulk order (eps = 0.1), a | radial-noise share of the total effect (h, a / log\|h\|) | gates-only model (radius clamped): D of direction | 99% truncation dimension of a |
|---|---|---|---|---|---|
| 16 | 2.5 / 2.3-3.1 / 2.6-2.9 / 2.6-2.8 / 5.1-6.7 | 1.9-2.1 | 0.09-0.12 / 0.21-0.27 | 2.2-3.0 | 4n |
| 64 | 2.1-2.2 / 1.9-2.0 / 2.4-2.7 / 2.2-2.3 / 8.7-10.0 | 1.7-1.8 | 0.05-0.08 / 0.09-0.11 | 1.8-2.2 | 2n |
| 256 | 3.0-3.1 / 3.1-3.3 / 3.0 / 3.1-3.2 / 13.2-14.7 | 1.9 | 0.03 / 0.04-0.07 | 2.6-2.7 | 2n-4n |
Pure radius mechanism (the annealed Kesten chain, per-layer blocks): D of log Q = 1.22, 1.23, 1.25 at n = 16, 64, 256.

- **Where the variance lives (PI3 hit).** In the angular noise: all angular coordinates together have total effect
  >= 0.96 for h_0 and a at every width, the radial coordinates (one number per layer) 0.03-0.12. Even the radius log|h_0|
  depends on the radial noise for only 4-11% at n >= 64: the radius is set by the direction history through the
  direction-dependent gains, not by noise injected along the radius. So "radius-first" coordinates are the wrong ones.
- **Mechanisms (HEURISTIC reading of two imperfect controls).** (a) Multiplicative through the radius: the annealed
  Kesten chain has mean order 1.2 at every width (PI4 i hit), and the net's log-radius has more, 1.9-3.3 (PI4 ii hit).
  The comparison changes the randomness (annealed chain blocks against noise at fixed weights), so it bounds the
  scalar mechanism's own interactions rather than isolating it in the net. (b) Gate switching: clamping the radius after
  every layer keeps 66-113% of the direction's order (PI4 iii missed narrowly: 0.66 on one net); clamping is itself
  nonlinear and changes the relative noise, so this is not a clean gates-only model. Together with the output gate
  pattern being by far the most interacting observable (order 5-15) and the bit cascade (one resampled layer flips
  1.6-4.5 n gates over the next 2n layers, Amendment A2), the reading is that gate switching is the main source: an
  injection changes the Jacobian through which everything older is carried.
- **Not log-additive.** In log-radius + direction coordinates the interactions do not vanish: log|h_0| has order
  1.9-3.3 and the direction 2.4-3.0. The structure is not a product of near-independent factors.
- **Which pairs.** Per-pair superset importances are small and spread out: at n = 256 at most 0.004 of Var(h_0) for any
  pair (ages 8 and 32), decaying with the lag, versus single-age total effects of 0.06 for the last few layers. The
  interactions are diffuse, many weak couplings over lags up to ~n, not a few strong pairs. The output gates couple
  strongly to the most recent layers (0.12-0.18 for pairs with age <= 8).
- **The dimension distribution is moderately heavy-tailed (PI2 missed).** The bulk statistic (eps = 0.1) is 1.7-2.1
  against D = 2.1-3.2: most of the variance is in low-order terms and a small fraction in high-order ones;
  three-component fits are unstable and not used.
- **Does it grow with width? Unresolved (PI5 missed).** D for h_0 is 2.5, 2.1, 3.1 at n = 16, 64, 256 (radial/angular
  blocks), not monotone; Part A's per-age sums in Cartesian blocks (2.1, 2.55, 3.9) rise monotonically. The two
  decompositions differ, so neither the agreement within 30% (PI1) nor the difference settles the trend. The controls
  do not grow (Kesten 1.2-1.25; clamped model about 2.6). The output gate pattern's order does grow (about n^0.35).
  With 2 nets per width and a heavy-tailed radius, no width trend for the continuous state or output is claimed.
- **Effective dimension for sampling.** Truncation dimension (99% of Var(a) from the most recent d layers): 4n at
  n = 16, 2n at 64, 2n-4n at 256 (PI7: within [n/2, 4n] everywhere), i.e. 2-4 n^2 Gaussian coordinates. Superposition
  in layer blocks: bulk order about 2. The deep past older than n layers explains <= 5% of Var(a) on 5 of 6 nets, but 33%
  on one n = 256 net whose layer-0 radius is large (Q = 1,279, about 9 times the median 0.577 n; PI7 missed there):
  a large radius was built long ago, so large-radius layers have long memories.
- Reading for structured sampling: low order in layer blocks, but each block is n-dimensional and the relevant window
  is 2n-4n layers, so RQMC can exploit only the low-order structure in the most recent layers. This matches RQMC's
  gains shrinking with width (Part B).

## 3. Part B: estimating E[a] and a tail under the stationary measure, as compute grows

Registered in PREREG Part B (13:40Z) before any Part B run; scorer `scratch/analyze_B.py`, scorecard
`results/partB_scores.json`. Seeds 1000, 1001 at n = 8..256. Cost unit: one n x n matrix-vector product (mv). The
constant K of a method is its per-sample cost times its per-sample variance in units of tr Var(a): plain MC with
burn-in B has K = B + 1, and then nMSE = K / compute once the burn-in bias is negligible.
- **Matched burn-in.** The structured estimators were run at B = 8n, at or above every net's B_6 (the smallest dyadic
  burn-in with measured bias^2 <= 1e-6 trVar). Their variance ratios are properties of the stationary law, so the
  "gain" below is the variance ratio at equal per-sample cost and equal burn-in (CV: halved for its double cost). Only
  burn-in multilevel is compared with MC at B_6, because removing burn-in is its purpose. Charging the structured runs
  their actual 8n against MC at B_6 instead gives best gains 3.5, 3.1, 4.1, 0.8, 1.2, 2.4 at n = 8..256 (review).
- **Setup excluded.** Gains are asymptotic sampling constants: one-off setup (the CV's closure and covariance
  recursions, 6 dense n x n x n products per layer, i.e. about 6n mv per layer over 8n layers, not the 5n charged in
  the code; the tail's 512-path pilot) is excluded. At the actual tail budgets the pilot lowers the n = 256 gains
  from 6.2 and 7.3 to about 4.8 and 5.5.
- **Resolution.** RQMC variances come from R = 16 scrambles (8 at n = 128, at only three N), so fitted exponents carry
  roughly +-0.1; at n = 256 the K2 errors include the truth's own 1/8192 = 1.2e-4; B_6 is chosen on point estimates
  (n = 64, seed 1000: selected bias^2 9.4e-7 with an estimated noise contribution of 6.9e-7).

### 3.1 What every method's mean squared error does as compute grows
| method | exponent of nMSE in compute | constant, or gain over MC (matched burn-in, setup excluded) | trend with width |
|---|---|---|---|
| MC with burn-in B | 1, with a log: B must grow like (n/kappa) ln(1/eps^2), kappa = 1.1-3.4 (median about 2) | K = B + 1: 4n-8n mv at bias^2 <= 1e-6 | K grows like n |
| burn-in multilevel, optimal finite-level allocation of increment variances (the unbiased random-level version needs second moments and was not measured) | 1, no log [A: PROVEN UNDER decay bounds] | K = 4.7n-14n mv: worse than MC at bias^2 <= 1e-6 on 10 of 12 nets, better on both n = 256 nets (1,667 and 1,991 against 2,049) (PB1 hit) | the log it removes matters only at very small bias targets |
| RQMC on the noise of the recent layers (Sobol in recency order) | fitted 1.19-1.22 at n = 8, 1.10-1.13 at 16-32, 0.92-1.11 at 64-128 (N = 64-4096) | 4.1-8.0x (n = 8), 3.4-3.7 (16), 3.0 (32), 2.7-3.0 (64), 2.1-2.6 (128) at N = 1024 | shrinks with width (PB5 missed: still > 2 at n = 128) |
| antithetic noise (whole history xi and -xi) | 1 | 1.9-2.0x (n = 8-32), 2.2-2.4x (64-256) at no extra cost | grows slowly (PB7 hit) |
| Gaussian-surrogate control variate (K2-Stein closure, soft gates, same noise) + final-noise RB | 1 | residual 0.21-0.36 at twice the cost: 1.3-2.0x | flat (PB3 refuted) |
| final-noise Rao-Blackwellisation alone | 1 | 1.3-1.6x at n = 8, 1.01-1.07x at n >= 64 | vanishes (about n^-1/3 [A]) |
| K2 Gaussian closure (deterministic) | 0: a bias floor | nMSE 1.6e-3-1.1e-2 (n = 8-16), 3.3-3.9e-3 (32), 1.3-1.9e-3 (64), 8e-4 (128), 3.8-7.7e-4 (256): about 0.12/n | falls like 1/n (the CLT), but MC at the closure's own cost also gives O(1/n): a constant 3-8x window, then MC wins |
| tail P(\|a\| > t), p ~ 1e-2: MC | 1 | relative variance (1-p)/p ~ 99 per sample | |
| tail: radial tilt of the noise along the current direction | 1 | 0-2x, often biased low (heavy weights); the code had a likelihood-ratio bug when the pre-noise activation is all-off (review finding 14: possible only at small n, about 6% of paths at n = 8, negligible at n >= 32); fixed; reruns at n = 8, 16 (`runs/tailfix_*`) reproduce the original numbers to two digits | |
| tail: fixed gradient tilt (pilot mean of d\|a\|/d xi) | 1 | 3.1-9.5x (better of two tilt sizes), within 2.5 combined SE of the truth on every net (PB8 missed only at n = 16: 3.7x on both nets) | no trend |

- Best gain G(n) at matched burn-in among the measured methods (median over 2 nets): 7.0, 4.4, 4.1, 3.0, 2.4, 2.4 at
  n = 8..256 (RQMC up to 64, antithetic beyond). The registered main hypothesis HB (no width-growing gain) holds under
  either accounting.
- Antithetic at larger width (Amendment B1, 14:19Z; PB9 hit): pair ratio 0.43 at n = 512 and 0.42 at n = 1024, so the
  part of the variance that is even in the noise levels off near 21% and the free antithetic gain near 2.4x.
- RQMC as compute grows (Amendment B2, 14:19Z; PB10 hit): fitted over N = 2^10..2^16 (2^14 at n = 32) the variance
  exponent is 1.04-1.06 at n = 8, 1.09-1.11 at n = 16 and 1.03-1.10 at n = 32, against 1.19-1.22, 1.13 and 1.10-1.12 over
  2^6..2^12. The gain saturates: at n = 8 it is 7.6, 8.0, 12.5, 7.7 at N = 2^10, 2^12, 2^14, 2^16 (4.7-9.9 on the other
  net), 4.8-6.8 at n = 16 and 3.3-4.1 at n = 32. So the higher early exponents were pre-asymptotic; asymptotically
  RQMC is a constant factor, shrinking with width.
- Why, from Part A: the stationary state is made by noise over about n layers, each layer moving it by about
  n^-1/2 relative and flipping about 0.55 sqrt(n) gates; the noise-to-state map has mean interaction order 2-4 (Part I);
  the part of the variance that is even in the noise (antithetic ratio 0.41-0.55, i.e. 20-27% of the variance) does
  not shrink with width, and a linear-in-noise control cannot remove it.
- So in the stationary regime, structured sampling buys constants of 2-8x, none growing with width, and burn-in
  multilevel removes a logarithm that matters only at small bias targets (it wins at n = 256, loses below).
- Bug fix (reported both ways): the CV + RB combination divided by a near-zero surrogate variance on degenerate output
  coordinates at n = 128-256 (residual 2.2, 4e37, 4e72 as first run); guarded rerun gives 0.33-0.35, all other numbers
  identical (`runs/cv_*` vs `runs/cvfix_*`, `scratch/partB_as_registered.py`).

## 4. Part C: input bits below the memory time, down to ARC's 256 x 32

Registered in PREREG Part C (13:43Z) and Amendment C1 (13:51Z, crossed ANOVA) before any Part C run; scorer
`scratch/analyze_C.py`, `results/partC_scores.json`. Model S with an input x ~ N(0, I) at layer 0 and absolute noise
sigma; n = 32, 128, 256; sigma = 1e-3, 1e-2, 1e-1; L from n/8 to 8n.

- **The input is forgotten at about T_mem = (4n/5) ln(1/(sigma sqrt(0.4 n)))** (the radius decays to the noise floor):
  the input's share of Var(a_L) (nested estimate between/(between + within); the first version divided by a pooled
  variance and gave shares up to 1.015, review finding 7) is >= 0.9 at L <= T_mem/4 on 20 of 21 grid points and <= 0.2
  at L >= 2 T_mem on 17 of 17 (PC1); its half point lies between T_mem/2 and T_mem.
- **One budget, nearly additive.** In the crossed design (32 inputs x 32 shared noise sequences) the input x noise
  interaction share is <= 0.05 wherever the input dominates and at most 0.26 anywhere (at the transition) (PC7a, b).
  So input and noise contributions to the variance add, to within a quarter at worst.
- **Input "bits" fade long before the input stops mattering.** The sum of the input's marginal gate entropies at the
  last hidden layer (an upper bound on the joint gate entropy)
  (over inputs, noiseless) falls from 19-23 of 32 bits at L = 4 (n = 32) and 39-44 of 128 at L = 16 (n = 128) to 38 of
  256 at L = 32 (n = 256) and 6-17 bits by L = n/2, while at sigma = 1e-3 the input still carries > 99% of the
  variance there. So beyond about n/8 layers most of the input's influence is not in gate variability; the radius is
  the obvious carrier, but residual directional information was not measured (PC4's gate-entropy clause and PC6
  missed for this reason).
- **Noise births track the relative noise** eps(L) = sigma sqrt(n)/R_L, which grows like sigma exp(5L/(4n)) (PC2: 24
  of 26) and gives about (sqrt 2/pi) n eps flips per layer (PC3: 19 of 26 within 1.5x).
- **ARC's regime (n = 256, L = 32, sigma = 1e-3):** input share 0.9992 (nested; crossed ANOVA: input 0.9995, noise
  2e-4, interaction 4e-4); the noise moves the mean by nMSE 3e-6; 0.22 noise births per layer (about 7 in the whole net); the coupled
  noise correction has variance 8e-4 of the total, so any noiseless sampler can be used with a cheap noise correction.
- **Which structured gains survive.** Antithetic noise gives nothing where the input dominates (pair ratio 2.0, i.e. a
  loss; PC5a) and its 2x where the noise dominates. RQMC over the input gives 1.3-7.8x where the input dominates
  (2.1x at ARC's point; >= 2x on 13 of 20 points, PC5b missed) and nothing in the stationary regime. The noise is a
  cheap correction (coupled variance <= 0.01) at sigma = 1e-3 through L ~ n/2, but not at sigma = 1e-2 by L = n/2
  (PC5c: 12 of 21). At ARC's point the noiseless multilevel reference-trajectory sampler gains nothing (the unrestricted
  lane measured 1.01 at 256 x 32).

## 5. Scorecard of the registered predictions
Hash times in `work/stationary/prereg_hashes.txt` are authoritative (seven header times were first typed 1-21 minutes ahead of the
clock and rehashed with no data in between). "Median rule" = the registered aggregation over nets.

| part | hit | missed or refuted |
|---|---|---|
| A (13:22Z) | PA2 (median, Hill), PA3 (median rule), PA4 (relative-distance fit), PA6, PA7, PA9, PA10a, b (median rule), d, PA12, PR1 | PA1 (10% everywhere fails at n = 128-256; the 20% refutation threshold at n <= 64 is not crossed), PA2 monotone approach, PA5 (slope band), PA8 (worst bins at n <= 32), PA10c, e, PA11 (radial non-Gaussianity fades), PR2 |
| A2 bit cascade (13:56Z) | PA13c | PA13a (first-layer flips scatter with the radius), PA13b (cascade 3.8-4.5 n > 3 n at n = 256) |
| A3 finite-angle law (13:58Z) | PA14, PA15 | |
| B (13:40Z), B1, B2 (14:19Z) | PB1, PB7, HB, PB9, PB10 | PB2 (two widths below the band), PB3 (refuted: CV residual flat at 1/3), PB4 (one n = 256 net at 7.7e-4), PB5 (RQMC still 2.1-2.6x at 128; exponent 0.92 on one net), PB6 (RB ratio above 0.9 at n >= 64), PB8 (n = 16 only) |
| C (13:43Z, C1 13:51Z) | PC1b, PC5a, PC5b2, PC7a, PC7b | PC1a (20 of 21), PC2 (24 of 26), PC3 (19 of 26), PC4 (gate-entropy clause), PC5b1 (13 of 20), PC5c (12 of 21), PC6, PC7c (16 of 17) |
| I (14:10Z) | PI1, PI3, PI4 i-ii, PI6 (gates) | PI2, PI4 iii (0.66 on one net), PI5 (no growth of the isolated mechanisms), PI6 (monotonicity), PI7 (one long-memory net), PI8 (2.06 > 2) |

## 6. Process, deviations, compute
- **Co-prover (gpt-6-astra, xhigh, read-only; `coprover/PROVENANCE.md`).** Round 1 (13:03-13:28Z), an independent
  attempt from the questions alone; my v1 was frozen at 13:15Z before its answer (disclosure: I saw its last log lines
  at about 13:13Z). Independently found by both: scale invariance, the exact radius chain and its Pareto tail and
  inverse-gamma bulk, the input-forgetting time, births of order sqrt(n). New from astra: exact resets and uniqueness
  among all entrance laws, the explicit contraction constant, quenched light tails, -infinity exponents and the
  rank-conditioned exponent formula, the scope of the Lyapunov covariance formula, Model R's scale/shape
  decomposition, final-noise Rao-Blackwellisation. Round 2 (13:36-13:55Z): an audit of v1 (12 corrections, now in
  FORMAL v2.1), of my scoring (PA3 and PA10b pass under the registered median rule), and the finite-angle closure for
  the cloud, which I then tested on fresh nets (A3).
- **Review.** GPT-6 review (fresh session, high effort, 14:18-14:25Z; `review/review_gpt6.md`): verdict "major
  revision", 14 findings, all addressed in this version:
  1. gains vs MC at different burn-ins: now stated at matched burn-in, with the review's alternative accounting
     (3.5, 3.1, 4.1, 0.8, 1.2, 2.4) given; HB holds either way;
  2. the headline now separates MSE of unbiased methods, the burn-in log, finite-range RQMC fits and the closure floor;
  3. PA1 rescored as missed (refutation threshold not crossed); PA4's relative-distance fit disclosed; PA3 and PA10b
     text corrected to the median-rule hits;
  4. burn-in multilevel: "worse on 10 of 12 nets, better at n = 256", and it is the finite-level allocation, not the
     unbiased random-level estimator;
  5. gains labelled as asymptotic sampling constants with setup excluded; tail gains at the actual budget given; the
     CV setup charge corrected to 6n mv per layer;
  6. statistical caveats added (RQMC exponent uncertainty, truth error in the n = 256 K2 numbers, B_6 selection);
  7. Part C input share renormalised (between/(between + within)); PC counts unchanged; ARC's share 0.9992 nested,
     0.9995 crossed;
  8. Part I's D is a finite-eps lower bound of the mean order, in radial/angular blocks, a different decomposition
     from Part A's; the "coarse-grid artifact" claim withdrawn, the width trend called unresolved;
  9. radius numbers corrected (nets' medians 0.565-0.606 n; quantile agreement 6% / 7.5% / 19%; O(1/n) restored;
     median 0.577 n labelled as the chain's simulation plus the proven limit);
  10. flip formulas: exact per-row 2 Phi(t) Phi(-t), arccos as the Gaussian-unit average;
  11. "gate bits" renamed sums of marginal entropies; the radius-carrier claim qualified;
  12. mechanisms relabelled HEURISTIC (clamping and the annealed Kesten chain are imperfect controls); FORMAL 5a no
     longer says the closure derives the memories;
  13. FORMAL: contraction constant for 1 <= p < 2; almost every weight sequence; finite-window covariance identity;
  14. radial-tilt likelihood-ratio bug fixed (theta^2 |e|^2/2); it can matter only when a layer's pre-noise
     activation is all-off (about 6% of paths at n = 8, negligible at n >= 32); the reruns at n = 8, 16
     (`runs/tailfix_*`) reproduce the original numbers to two digits.
- **Deviations** (all reported, none hidden):
  - Part A's age, Lyapunov and coupling runs used 2 seeds, not 4, and the n >= 128 radius runs used 2 paths x 40,000
    layers instead of 4 x 20,000 (undeclared; found by astra's audit).
  - PA3 and PA10b were first scored every-net, then rescored with the registered median rule.
  - The PA7 scorer allowed a 0.02 slack (no effect). PB8 was first scored with |error| <= 30%, then with the
    registered 3-SE rule.
  - The CV + RB bug fix and rerun (section 3).
  - partA.py was edited after Amendment A2's hash (cloud mode only; the bitlife code it ran is unchanged).
  - The Lyapunov QR warm-up started from h = 0 with 2n layers discarded; Model R's pilot is noisy and recomputed per
    sigma, so its schedule changes with sigma beyond scaling (astra).
- **Not done.** Third-order Sobol indices (only all-order superset importances of pairs); the exact one-dimensional
  final-noise integration for the tail (astra's suggestion); RQMC at n = 256.
- **Compute.** CPU only, nice 19 and idle IO, at most 3 heavy processes, single-threaded BLAS; no GPU.
